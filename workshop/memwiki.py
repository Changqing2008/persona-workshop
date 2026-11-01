"""记忆层检索（只读）。

设计约束见 docs/记忆系统-LLM-wiki模式设计.md：
- 只扫 memory/，绝不读 raw/ —— 压缩对话原文不能出现在聊天提示词里。
- 只取「## 事实」段的行，不整页注入：记忆回答"发生过什么"，不是性格设定。
- 命中才返回，未命中返回空串，日常对话不占一个字。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import log

log_ = log.get("memwiki")

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DATE_FULL = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_DATE_CN = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")
_SOURCE_TAIL = re.compile(r"[（(]\s*source\s*:[^）)]*[）)]")
_MONTH_FILE = re.compile(r"^(\d{4})-(\d{2})\.md$")
_WIKILINK_ARROW = re.compile(r"\s*→\s*\[\[[^\]]+\]\]")
_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")

# 回忆意图词：出现时一次多取几条，没有时只取最相关的一条。
INTENT = ("记得", "上次", "那次", "以前", "之前", "我说过", "后来呢", "还记得", "旧事")

# 性格断言拦截（L4 闸门）。命中说明这条事实被写成了性格判断，跳过并告警。
JUDGEMENT = re.compile(r"她(总是|一向|从来|本质|其实很|这人|天生|根本)")

FACT_HEADING = "## 事实"
MIN_ALIAS = 2
"""单字别名不给建索引：'他'、'你' 这种会命中每一句话。"""

PLACEHOLDER = ("_待填_", "_待", "待归纳", "TODO")
"""占位行不许进提示词：模型看到 _待填_ 会自己编一个填上。"""


@dataclass(slots=True)
class MemPage:
    rel: str
    title: str
    type: str
    status: str
    confidence: str
    aliases: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    user_ids: list[str] = field(default_factory=list)
    """这个人在 QQ 上的 id。用来把"记忆里有没有他"接进关系档位判定。"""
    body: str = ""


@dataclass(slots=True)
class Hit:
    page: MemPage
    score: int


def _parse_front(text: str) -> tuple[dict[str, object], str]:
    """极简 frontmatter 解析。只支持本项目实际用到的形式，不引 yaml 依赖。"""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, object] = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if val.startswith("["):
            try:
                parsed = json.loads(val)
                meta[key] = [str(x) for x in parsed] if isinstance(parsed, list) else val
            except (json.JSONDecodeError, TypeError):
                inner = val[1:-1] if val.endswith("]") else val[1:]
                meta[key] = [x.strip().strip("\"'") for x in inner.split(",") if x.strip()]
        else:
            meta[key] = val.strip("\"'")
    return meta, text[m.end() :]


def _clean_line(s: str) -> str:
    """注入前剥掉链接语法：模型读 `[[那个熟人]]` 只会照着念出方括号。"""
    s = _WIKILINK_ARROW.sub("", s)
    s = _WIKILINK.sub(r"\1", s)
    return s.strip()


def _is_placeholder(s: str) -> bool:
    return any(tok in s for tok in PLACEHOLDER)


def _title_of(meta: dict[str, object], body: str, stem: str) -> str:
    title = str(meta.get("title") or "").strip()
    if title:
        return title
    m = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return m.group(1).strip() if m else stem


def _str_list(meta: dict[str, object], key: str) -> list[str]:
    val = meta.get(key)
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


class MemWiki:
    """memory/ 目录的只读检索视图。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.mem_dir = self.root / "memory"
        self._pages: list[MemPage] = []
        self._alias_map: dict[str, list[MemPage]] = {}
        self._tm: dict[str, float] = {}
        self.refresh()

    # ---------- 索引 ----------

    def _scan(self) -> list[MemPage]:
        if not self.mem_dir.is_dir():
            return []
        pages: list[MemPage] = []
        for path in sorted(self.mem_dir.rglob("*.md")):
            rel = path.relative_to(self.mem_dir).as_posix()
            if rel.startswith("pending/"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                log_.warning("记忆页读取失败 %s: %s", rel, exc)
                continue
            meta, body = _parse_front(text)
            # scope: system 的页只存档、不检索：里面是实现细节，
            # 被注入会让角色说出"我的记忆库走 wiki 模式"这类 OOC 内容。
            if str(meta.get("scope") or "").strip() == "system":
                continue
            title = _title_of(meta, body, path.stem)
            aliases = _str_list(meta, "aliases")
            # 页名与标题本身也是检索键，不用在 frontmatter 里重复写
            for auto in (title, path.stem):
                if auto and auto not in aliases:
                    aliases.append(auto)
            pages.append(
                MemPage(
                    rel=rel,
                    title=title,
                    type=str(meta.get("type") or ""),
                    status=str(meta.get("status") or ""),
                    confidence=str(meta.get("confidence") or ""),
                    aliases=aliases,
                    links=_str_list(meta, "links"),
                    user_ids=_str_list(meta, "user_id"),
                    body=body,
                )
            )
        return pages

    def refresh(self) -> None:
        self._pages = self._scan()
        table: dict[str, list[MemPage]] = {}
        for page in self._pages:
            for alias in page.aliases:
                if len(alias) < MIN_ALIAS:
                    continue
                table.setdefault(alias, []).append(page)
        # 别名撞车：两页抢同一个词，取 rel 排序第一，保证结果确定（doctor 负责报出来）
        for alias, owners in table.items():
            if len(owners) > 1:
                log_.warning(
                    "别名冲突「%s」：%s", alias, ", ".join(p.rel for p in owners)
                )
        self._alias_map = table

    @property
    def pages(self) -> list[MemPage]:
        return list(self._pages)

    @property
    def ready(self) -> bool:
        return bool(self._pages)

    def has_person(self, user_id: str) -> bool:
        """这个 QQ 号在 people/ 下有没有"带事实"的记忆页。

        关系档位靠它判定：有共同经历才算熟人，聊得再多但没留下事实就还是陌生人。
        只认 people/ 且必须有「## 事实」行——空壳页（stub 但没料）不算认识。
        """
        if not user_id:
            return False
        for page in self._pages:
            if page.type != "person":
                continue
            if user_id in page.user_ids and self.fact_lines(page):
                return True
        return False

    # ---------- 检索 ----------

    def hits(self, text: str) -> list[Hit]:
        """别名命中。长别名优先——"截图那位"比"熟人"更具体。"""
        if not text:
            return []
        best: dict[str, Hit] = {}
        for alias, owners in self._alias_map.items():
            if alias not in text:
                continue
            page = owners[0]
            cur = best.get(page.rel)
            score = len(alias)
            if cur is None or score > cur.score:
                best[page.rel] = Hit(page=page, score=score)
        return sorted(best.values(), key=lambda h: (-h.score, h.page.rel))

    def month_file(self, month: int, year: int | None = None) -> Path | None:
        tl = self.mem_dir / "timeline"
        if not tl.is_dir():
            return None
        cands: list[tuple[int, Path]] = []
        for path in sorted(tl.glob("*.md")):
            m = _MONTH_FILE.match(path.name)
            if not m:
                continue
            y, mo = int(m.group(1)), int(m.group(2))
            if mo != month or (year is not None and y != year):
                continue
            cands.append((y, path))
        return cands[-1][1] if cands else None

    def timeline_lines(self, text: str) -> list[str]:
        """显式日期命中：2026-10-24、10月24号。取该月份页里对应日的行。"""
        if not text:
            return []
        year: int | None = None
        day = month = 0
        m = _DATE_FULL.search(text)
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            m = _DATE_CN.search(text)
            if not m:
                return []
            month, day = int(m.group(1)), int(m.group(2))
        path = self.month_file(month, year)
        if path is None:
            return []
        try:
            _, body = _parse_front(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            return []
        # timeline 行格式是 `- MM-DD · 内容`，月份必须一起匹配，只比 day 会串月
        tags = (f"- {month:02d}-{day:02d}", f"- {month}-{day}")
        out: list[str] = []
        for line in body.splitlines():
            s = line.strip()
            for tag in tags:
                if s.startswith(tag):
                    out.append(s[len(tag) :].lstrip(" ·-—").strip())
                    break
        return [x for x in out if x]

    # ---------- 组装 ----------

    @staticmethod
    def fact_lines(page: MemPage) -> list[str]:
        """只取「## 事实」段的行，剥掉 source 尾巴，拦性格断言。"""
        out: list[str] = []
        inside = False
        for line in page.body.splitlines():
            if line.startswith("## "):
                inside = line.strip().startswith(FACT_HEADING)
                continue
            if not inside:
                continue
            s = line.strip()
            if not s.startswith("- "):
                continue
            s = _SOURCE_TAIL.sub("", s[2:]).strip()
            if not s or _is_placeholder(s):
                continue
            if JUDGEMENT.search(s):
                log_.warning("记忆含性格断言，已跳过：%s / %s", page.rel, s)
                continue
            out.append(s)
        return out

    @staticmethod
    def table_lines(page: MemPage) -> list[str]:
        """hub 页没有「## 事实」，内容在表格里：表头之后每行拼成一句。"""
        out: list[str] = []
        header_skipped = False
        for line in page.body.splitlines():
            s = line.strip()
            if not s.startswith("|"):
                continue
            if set(s) <= set("|-: "):
                continue
            if not header_skipped:
                header_skipped = True
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            cells = [c for c in cells if c]
            # 空模板库的表格全是 _待填_，注入进去等于让模型自己编一行
            if not cells or any(_is_placeholder(c) for c in cells):
                continue
            out.append(" · ".join(cells))
        return out

    def page_lines(self, page: MemPage) -> list[str]:
        """一页要注入的内容：hub 走表格，其余只走「## 事实」段。"""
        if page.type == "hub":
            return [x for x in (_clean_line(s) for s in self.table_lines(page)) if x]
        return [x for x in (_clean_line(s) for s in self.fact_lines(page)) if x]

    def recall(self, text: str, budget: int = 800, top: int | None = None) -> list[str]:
        """命中项拍平成待注入的行。每条自带出处标题，便于模型区分人和事。"""
        if not text or not self._pages:
            return []
        intent = any(w in text for w in INTENT)
        limit = top if top is not None else (3 if intent else 1)
        lines: list[str] = []
        for hit in self.hits(text)[:limit]:
            for fact in self.page_lines(hit.page):
                lines.append(f"{fact}（{hit.page.title}）")
        for item in self.timeline_lines(text):
            clean = _clean_line(item)
            if clean:
                lines.append(f"{clean}（时间线）")
        if not lines:
            return []
        out: list[str] = []
        used = 0
        for line in lines:
            if used + len(line) > budget and out:
                out.append("…（更多细节，问的时候再说）")
                break
            out.append(line)
            used += len(line)
        return out

    def render(self, text: str, budget: int = 800, top: int | None = None) -> str:
        """生成注入段。未命中返回空串——日常对话一个字都不加。"""
        lines = self.recall(text, budget=budget, top=top)
        if not lines:
            return ""
        body = "\n".join(f"- {line}" for line in lines)
        return (
            "## 相关记忆（来自记忆库，不是本轮对话）\n"
            "> 这些是你记得的事，不是你的性格；不要因此改变说话方式。\n"
            f"{body}"
        )

    def describe(self) -> str:
        counts: dict[str, int] = {}
        for page in self._pages:
            counts[page.type or "未知"] = counts.get(page.type or "未知", 0) + 1
        head = "记忆库：%s（%d 页）" % (self.mem_dir, len(self._pages))
        detail = "  " + " / ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        return f"{head}\n{detail}" if counts else head

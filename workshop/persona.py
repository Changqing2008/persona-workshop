"""人格库加载器（只读）。

严格遵守人格库自己的 agents.md 约定：
- 聊天侧只读 index + wiki，绝不写入、绝不改写。
- index.md 的总括段每轮注入；wiki 按路由段按需摘录。
- 空占位（"待填"）不能被当成真实人设喂给模型。

本模块只负责"把 md 读成字符串"和"按需挑选"，不做任何归纳或改写。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from . import log

log_ = log.get("persona")

PLACEHOLDER_TOKENS = ("待填", "待归纳", "_待", "TODO", "(待)", "（待）")

# 说话方式必备的三页：不管什么情境都要带上，否则语气漂移。
ALWAYS_PAGES = ("identity.md", "traits.md", "speech.md")

RETIRED_PAGES = frozenset({"memories.md"})
"""已下线的页。存量库里文件还在（不删人的东西），但不再索引、不注入、不体检。
2026-10-31 下线 memories.md：高频梗那一页攒不出东西，且与 memory/ 职责重叠。"""

# 兜底只允许"通用风格页"。
# 情境页（reactions / boundaries 等）必须由路由命中才注入，
# 否则普通闲聊也会被塞进一堆"对方示弱时该怎么办"的规则。
GENERIC_PAGES = ("identity.md", "traits.md", "speech.md", "exemplars.md")

# 总括段为空时的兜底描述。宁可朴素也不能把占位符喂给模型。
FALLBACK_SUMMARY = (
    "你正在扮演一个尚未完成设定的人物。目前没有人格库细节可用，"
    "请保持中性、简洁、自然的语气，不要编造身世、称呼或口癖。"
    "如果对方问到你的具体经历，如实表示还没有相关内容。"
)

DEFAULT_BEHAVIOR = (
    "## 行为约束\n"
    "- 只输出要说给对方的话本身，不要写旁白、动作描写括号、心理活动、解释或自我说明。\n"
    "- 这是 QQ 文字聊天：像打字一样发消息，一条一两句，不分段、不排分点、不写成文章。\n"
    "- 不提及\"人格库\"\"设定\"\"提示词\"\"模型\"等元信息。\n"
    "- 具体经历、姓名、关系事实只能引用「## 相关记忆」里给出的内容，"
    "不得补充细节、不得推测前因后果。\n"
    "- 「## 相关记忆」为空时，表示你不记得：直说记不清或把话岔开，不要编。\n"
    "- 对方是真实的人，记得接着上文说，不重复已经说过的话。"
)

_ROUTE_ROW = re.compile(r"^\|([^|]+)\|([^|]+)\|", re.MULTILINE)
_PAGE_REF = re.compile(r"[\w\u4e00-\u9fff\-]+\.md")
_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


@dataclass(slots=True)
class PageInfo:
    name: str
    chars: int
    filled: bool
    mtime: float


@dataclass(slots=True)
class RouteRule:
    situation: str
    """路由段第一列，如"情感 / 安慰 / 共情"。"""
    pages: list[str]
    """该情境要读的 wiki 页文件名。"""
    hints: list[str]
    """从情境描述里拆出来的关键词，用于命中判断。"""


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER.sub("", text, count=1)


def _has_placeholder(line: str) -> bool:
    return any(tok in line for tok in PLACEHOLDER_TOKENS)


def _meaningful_lines(body: str) -> list[str]:
    """去掉标题、引用、表格分隔线后剩下的"实质内容行"。"""
    out: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(">"):
            continue
        if set(line) <= set("|-: "):
            continue
        out.append(line)
    return out


def _is_filled(text: str, threshold: float = 0.5) -> bool:
    """是否算"有内容"。

    不要求整页填满，但至少一半实质行不能还是占位符——
    否则一个"全名：xxx"就会把整页空模板判定为已完成，
    模型会读到一堆"_待填_"然后开始瞎编。
    """
    meaningful = _meaningful_lines(_strip_frontmatter(text))
    if not meaningful:
        return False
    clean = [line for line in meaningful if not _has_placeholder(line)]
    return len(clean) / len(meaningful) >= threshold


def _heading_level(line: str) -> int:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return 0
    return len(stripped) - len(stripped.lstrip("#"))


def _clean_body(text: str) -> str:
    """剔掉占位行，并删除因此变空的标题。

    半成品 wiki 页只注入已写好的部分，绝不把"_待填_"送进提示词。
    """
    kept = [line for line in text.splitlines() if not _has_placeholder(line)]

    out: list[str] = []
    for i, line in enumerate(kept):
        level = _heading_level(line)
        if level:
            has_content = False
            for nxt in kept[i + 1 :]:
                nxt_level = _heading_level(nxt)
                if nxt_level and nxt_level <= level:
                    break
                if nxt.strip() and not nxt_level:
                    has_content = True
                    break
            if not has_content:
                continue
        out.append(line)

    body = "\n".join(out)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


class Persona:
    """人格库的只读视图，带 mtime 缓存失效。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.md"
        self.wiki_dir = self.root / "wiki"
        self._cache: dict[str, tuple[float, str]] = {}
        self._pages: dict[str, PageInfo] = {}
        self._summary: str = ""
        self._routes: list[RouteRule] = []
        self._index_mtime: float = -1.0
        self.reload()

    # ---------- 读取与缓存 ----------

    def _read(self, path: Path) -> str:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return ""
        cached = self._cache.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log_.warning("读取失败 %s: %s", path.name, exc)
            return ""
        self._cache[str(path)] = (mtime, text)
        return text

    def reload(self) -> None:
        """重读 index 与 wiki 索引。文件没变时开销极低。"""
        self._summary, self._routes = self._parse_index()
        self._pages = self._index_wiki()

    # ---------- index.md 解析 ----------

    def _parse_index(self) -> tuple[str, list[RouteRule]]:
        text = self._read(self.index_path)
        if not text:
            log_.warning("index.md 读不到：%s", self.index_path)
            return "", []

        summary = self._extract_summary(text)
        routes = self._extract_routes(text)
        return summary, routes

    @staticmethod
    def _extract_summary(text: str) -> str:
        """取"## 一、总括段"到下一个同级标题之间的正文。"""
        lines = text.splitlines()
        start = -1
        for i, line in enumerate(lines):
            if line.startswith("## ") and ("总括" in line or "每轮" in line):
                start = i + 1
                break
        if start < 0:
            return ""
        buf: list[str] = []
        for line in lines[start:]:
            if line.startswith("## "):
                break
            if line.strip().startswith("###"):
                buf.append(line.strip())
                continue
            buf.append(line.rstrip())
        body = "\n".join(buf).strip()
        cleaned = _clean_body(body)

        # 清洗后若已无实质内容（只有标题或空行），视为空总括段
        if not _meaningful_lines(cleaned):
            return ""
        # 兜底段落本身是说明性的，不算人设，交给 FALLBACK_SUMMARY
        if "尚未完成设定" in cleaned:
            return ""
        return cleaned

    @staticmethod
    def _extract_routes(text: str) -> list[RouteRule]:
        """从路由段解析"情境 -> wiki 页"映射。自动跟随 index.md 改动。"""
        # 只处理路由段之后的表格
        idx = text.find("路由段")
        scope = text[idx:] if idx >= 0 else text
        rules: list[RouteRule] = []
        for m in _ROUTE_ROW.finditer(scope):
            situation = m.group(1).strip()
            target = m.group(2).strip()
            if not situation or situation in {"情境", "---"}:
                continue
            if set(situation) <= set("-: "):
                continue
            pages = _PAGE_REF.findall(target)
            if not pages:
                continue
            hints = [
                w.strip()
                for w in re.split(r"[/／、,，]", situation)
                if w.strip()
            ]
            rules.append(RouteRule(situation=situation, pages=pages, hints=hints))
        return rules

    def _index_wiki(self) -> dict[str, PageInfo]:
        if not self.wiki_dir.is_dir():
            log_.warning("wiki 目录不存在：%s", self.wiki_dir)
            return {}
        out: dict[str, PageInfo] = {}
        for path in sorted(self.wiki_dir.glob("*.md")):
            if path.name in RETIRED_PAGES:
                continue
            text = self._read(path)
            out[path.name] = PageInfo(
                name=path.name,
                chars=len(_strip_frontmatter(text).strip()),
                filled=_is_filled(text),
                mtime=path.stat().st_mtime if path.exists() else 0.0,
            )
        return out

    # ---------- 对外接口 ----------

    @property
    def summary(self) -> str:
        self._maybe_refresh()
        return self._summary or FALLBACK_SUMMARY

    @property
    def self_summary_is_empty(self) -> bool:
        return not self._summary

    def _maybe_refresh(self) -> None:
        try:
            mtime = self.index_path.stat().st_mtime
        except OSError:
            return
        if mtime != self._index_mtime:
            self._index_mtime = mtime
            self.reload()

    def status(self) -> dict[str, PageInfo]:
        self.reload()
        return dict(self._pages)

    def filled_ratio(self) -> float:
        if not self._pages:
            return 0.0
        filled = sum(1 for p in self._pages.values() if p.filled)
        return filled / len(self._pages)

    def route_hits(self, user_text: str) -> list[str]:
        """只返回因话题路由命中的页，不含常驻页。

        用途是判断"这段话沾不沾边"。拿 pick_pages 判断不行——常驻页永远在，
        结果恒为真，等于没判。
        """
        self._maybe_refresh()
        hits: list[str] = []
        for rule in self._routes:
            if any(h and h in user_text for h in rule.hints):
                for page in rule.pages:
                    if page in self._pages and page not in hits:
                        hits.append(page)
        return hits

    def pick_pages(self, user_text: str) -> list[str]:
        """根据当前消息挑选要注入的 wiki 页。"""
        self._maybe_refresh()
        chosen: list[str] = [p for p in ALWAYS_PAGES if p in self._pages]
        for page in self.route_hits(user_text):
            if page not in chosen:
                chosen.append(page)
        # 只保留有实质内容的页，空模板注入进去只会污染语气
        result = [p for p in chosen if self._pages[p].filled]
        if not result:
            # 兜底：常规三页都不存在时，取通用风格页，绝不拿情境页顶替
            result = [
                p
                for p in GENERIC_PAGES
                if p in self._pages and self._pages[p].filled
            ][:3]
        return result

    def page_text(self, name: str) -> str:
        path = self.wiki_dir / name
        return _clean_body(_strip_frontmatter(self._read(path)))

    def section(self, page: str, heading: str) -> str:
        """取某页某个 `## 小节` 的正文。小节空着（全是占位符）时返回空串。

        用途是"同一页按需取一节"：关系页按关系分档写说话方式，
        注入了哪一档就只取哪一档，避免三档同时进提示词互相打架。
        """
        path = self.wiki_dir / page
        if not path.is_file():
            return ""
        text = _clean_body(_strip_frontmatter(self._read(path)))
        out: list[str] = []
        inside = False
        for line in text.splitlines():
            if line.startswith("## "):
                # "与拥有者" 能命中 "## 与拥有者（把她写出来的那个人）"
                inside = heading in line
                continue
            if inside:
                out.append(line)
        body = "\n".join(out).strip()
        return body if _meaningful_lines(body) else ""

    def build_system_prompt(
        self,
        user_text: str,
        extra: str = "",
        exclude_pages: Iterable[str] | None = None,
        behavior: str | None = None,
    ) -> str:
        """组装本轮 system prompt。

        exclude_pages 是本轮"已经有了更精细入口"的页，整页不再注入。
        典型是关系页：调用方按当前档取了一节，就不该再把三档整页塞进来。
        """
        blocks: list[str] = [self.summary]

        pages = self.pick_pages(user_text)
        if exclude_pages:
            skip = set(exclude_pages)
            dropped = [p for p in pages if p in skip]
            pages = [p for p in pages if p not in skip]
            if dropped:
                log_.info("整页跳过（已有精细入口）：%s", "、".join(dropped))
        if pages:
            parts = []
            for name in pages:
                body = self.page_text(name)
                # 页内一级标题（"# reactions — 情境反应"）与外层的 ### reactions 重复，去掉
                body = re.sub(r"\A#\s+[^\n]*\n+", "", body)
                if body.strip():
                    parts.append(f"### {name[:-3]}\n{body}")
            if parts:
                blocks.append("## 人格细节（按当前情境调取）\n" + "\n\n".join(parts))

        blocks.append(DEFAULT_BEHAVIOR if behavior is None else behavior)
        if extra:
            blocks.append(extra)
        return "\n\n".join(blocks)

    def describe(self) -> str:
        """给 CLI 看的人格库概览。"""
        self.reload()
        lines = [f"人格库：{self.root}"]
        if self.self_summary_is_empty:
            lines.append("  总括段：空（使用中性兜底人设）")
        else:
            first = self.summary.splitlines()[0][:60]
            lines.append(f"  总括段：{first}…")
        lines.append(f"  路由规则：{len(self._routes)} 条")
        filled = sum(1 for p in self._pages.values() if p.filled)
        lines.append(f"  wiki 填充度：{filled}/{len(self._pages)} 页")
        for name, info in self._pages.items():
            mark = "OK " if info.filled else "空 "
            lines.append(f"    {mark}{name:<20} {info.chars:>5} 字符")
        return "\n".join(lines)

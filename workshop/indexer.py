"""index.md 总括段重建：把 wiki 里的新内容搬进「每轮注入」的那四小节。

一键归纳把新内容写进 wiki 之后，总括段如果还停在旧内容，聊天时根本看不见 ——
收了等于没收。这里补上这一步。

**确定性重建，不调模型**，理由有三：
  1. 总括段短、结构固定（四小节），能靠结构抽，不必花钱；
  2. 模型每跑一次结果都不一样，总括段天天变，等于人格天天变；
  3. 只增不改这一条，程序能自己守，不用再写一套闸。

三条铁律：
  · 人工写的行**永久保留**，程序不删、不改写；
  · 程序只覆盖自己上次写进去的行（记在 `data/index_state.json`）；
  · 小节里只要出现非清单的人工段落，整节不碰。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from . import log

log_ = log.get("indexer")

INDEX_NAME = "index.md"
STATE_REL = Path("data") / "index_state.json"
PLACEHOLDER = "_待填_"

MIN_LINE = 6
MAX_LINE = 64
SIM = 0.75

_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
_PAREN = re.compile(r"（[^（）]*）|\([^()]*\)")
_SRC = re.compile(r"raw/[\w\u4e00-\u9fff\.·\-]+\.md|见 log\s*\S*|conflict:\s*\S+")
_KEEP = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")

# 页码太薄的行不要（"定义："、"证据：…"这类只剩标签的）
_THIN = re.compile(r"^[\u4e00-\u9fff]{2,4}[：:]\s*$")


@dataclass(slots=True)
class Part:
    """总括段的一个小节，以及它从哪一页取料。"""

    title: str
    page: str
    limit: int
    sections: tuple[str, ...] = ()  # 只从这些 ## 节里抽；空 = 全页
    skip: tuple[str, ...] = ("待证", "矛盾")  # 这些 ## 节跳过
    bullet: bool = True  # False = 整段文字（一句话人设），只填空、不覆盖


PARTS: tuple[Part, ...] = (
    Part("一句话人设", "identity.md", 1, bullet=False),
    Part("核心特质", "traits.md", 7),
    Part("语言风格要点", "speech.md", 4),
    Part("底线", "boundaries.md", 2),
)


# ---------- 通用小工具 ----------


def _norm(text: str) -> str:
    return _KEEP.sub("", str(text))


def _similar(a: str, b: str) -> bool:
    """两行说的是不是同一件事。短行要求更严 —— 免得「短」和「直」被判成同义。"""
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return False
    if x in y or y in x:
        return min(len(x), len(y)) >= 4
    if min(len(x), len(y)) < 8:
        return False
    return SequenceMatcher(None, x, y, autojunk=False).ratio() >= SIM


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER.sub("", text, count=1)


def _blocks(text: str) -> list[tuple[int, str, list[str]]]:
    """按 ## / ### 切块，返回 (级别, 标题, 正文行)。级别 0 是文件头。"""
    out: list[tuple[int, str, list[str]]] = []
    level, title, buf = 0, "", []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m and len(m.group(1)) <= 3:
            out.append((level, title, buf))
            level, title, buf = len(m.group(1)), m.group(2).strip(), []
        else:
            buf.append(line)
    out.append((level, title, buf))
    return out


def _bullets(lines: list[str]) -> list[str]:
    return [ln.strip() for ln in lines if ln.strip().startswith(("-", "*", "·"))]


def _has_placeholder(line: str) -> bool:
    return PLACEHOLDER in line


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    for sep in "，；。、":
        i = head.rfind(sep)
        if i >= limit // 2:
            return head[:i]
    return head


def _compress(text: str, limit: int = MAX_LINE) -> str:
    """把 wiki 的一行长句压成总括段能用的短语。"""
    s = str(text).strip().lstrip("-*·").strip()
    s = _PAREN.sub("", s)  # 括号里基本是批注、强度、来源
    s = _SRC.sub("", s)
    s = s.replace("**", "").replace("`", "")
    s = s.strip(" 　，。；、：:")
    if _THIN.match(s) or len(s) < MIN_LINE:
        return ""
    return _cut(s, limit)


def _short(text: str, limit: int = 24) -> str:
    """名字、称呼这类短标签：只去注释，不套「至少 6 字」那道门。"""
    s = str(text).strip().lstrip("-*·").strip()
    s = _PAREN.sub("", s)
    s = s.replace("**", "").replace("`", "")
    return _cut(s.strip(" 　，。；、：:"), limit)


# 这些 label 后面跟的是出处、退役说明、批注，不是判断本身
NOISE_LABELS = frozenset(
    {
        "证据",
        "证据摘录",
        "来源",
        "辅助证据",
        "母本佐证",
        "退役",
        "校准",
        "conflict",
        "边界",
        "触发条件",
        "行为侧",
    }
)


def _is_noise(label: str) -> bool:
    if not label:
        return False
    return label in NOISE_LABELS or any(k in label for k in ("证据", "来源", "退役", "冲突", "边界"))


def _first_sentences(text: str, n: int = 2, limit: int = 110) -> str:
    """截到前 n 句。一句话人设要读起来像一句判断，不是一整段背景。"""
    s = str(text).strip()
    cuts = [i for i, ch in enumerate(s) if ch in "。！？"]
    if len(cuts) >= n:
        s = s[: cuts[n - 1] + 1]
    return _cut(s.rstrip("。"), limit)


def _label_value(line: str) -> tuple[str, str]:
    """拆「- 定义：xxx」成 ("定义", "xxx")；没冒号返回 ("", 整行)。"""
    body = line.strip().lstrip("-*·").strip()
    m = re.match(r"^([^：:]{1,12})[：:]\s*(.+)$", body)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", body


# ---------- 从 wiki 抽候选 ----------


def _from_identity(text: str) -> list[str]:
    """一句话人设：名字 + 设定背景第一条，拼成一句。"""
    blocks = _blocks(_strip_frontmatter(text))
    name, background = "", ""
    for _lvl, title, lines in blocks:
        if not name and "名字" in title:
            for ln in _bullets(lines):
                label, value = _label_value(ln)
                if label in {"对外叫", "对外", "名字", "本名"}:
                    name = _short(value)
                    break
            if not name:
                picked = [v for _l, v in map(_label_value, _bullets(lines)) if _l]
                name = _short(picked[0]) if picked else ""
        if not background and any(k in title for k in ("设定背景", "背景", "来历")):
            for ln in _bullets(lines):
                label, value = _label_value(ln)
                if label in {"外观", "载体", "在线时间", "活动半径"} or _is_noise(label):
                    continue
                background = _first_sentences(value)
                if background:
                    break
    if not name and not background:
        return []
    if name and background:
        bare = name.strip("「」『』\"'“”")
        return [f"「{bare}」——{background}"]
    return [name or background]


def _from_traits(text: str) -> list[str]:
    """核心特质：每个 ### 特质取它的「定义」一行，拼成「名：定义」。"""
    blocks = _blocks(_strip_frontmatter(text))
    heads = [t for lvl, t, _ in blocks if lvl == 2]
    strict = any("主特质" in t or "底色" in t for t in heads)

    out: list[str] = []
    section = ""
    for lvl, title, lines in blocks:
        if lvl == 2:
            keep = (not strict) or any(k in title for k in ("主特质", "底色"))
            section = title if keep and not any(k in title for k in ("待证", "矛盾")) else ""
            continue
        if lvl != 3 or not section:
            continue
        name = _short(re.split(r"[（(]", title)[0], 16)
        definition = ""
        for ln in _bullets(lines):
            label, value = _label_value(ln)
            if label in {"定义", "底色", "说明"}:
                definition = _compress(value)
                break
        if name and definition:
            out.append(f"- {name}：{definition}")
    if out:
        return out

    # 没有 ### 分层的旧页：退一步把所有「定义：」行收进来（待证/矛盾节的仍然不收）
    section = ""
    for lvl, title, lines in blocks:
        if lvl == 2:
            section = "" if any(k in title for k in ("待证", "矛盾")) else title
            continue
        if lvl != 3 and lvl != 0 and section == "":
            continue
        if not section:
            continue
        for ln in _bullets(lines):
            label, value = _label_value(ln)
            if label != "定义":
                continue
            v = _compress(value)
            if v and not any(_similar(v, o) for o in out):
                out.append(f"- {v}")
    return out


def _from_boundaries(text: str) -> list[str]:
    """底线：优先「不做的事」，没有就第一个 ## 节。"""
    blocks = _blocks(_strip_frontmatter(text))
    picked: list[str] = []
    for want in ("不做的事", "红线", ""):
        for lvl, title, lines in blocks:
            if lvl != 2:
                continue
            if want and want not in title:
                continue
            for ln in _bullets(lines):
                label, value = _label_value(ln)
                if _is_noise(label):
                    continue
                v = _compress(value)
                if v:
                    picked.append(f"- {v}")
            if picked:
                return picked
    return picked


def _from_speech(text: str) -> list[str]:
    """语言风格：每个 ## 节挑一条「像判断而不像术语表」的清单行。"""
    blocks = _blocks(_strip_frontmatter(text))
    # 排在前面的更"原则"，先占名额；同一节只取一条
    order = (("句式", "句长", "节奏"), ("消息体量", "体量"), ("语气语域", "语气"), ("口头禅",))
    out: list[str] = []
    for keys in order:
        for lvl, title, lines in blocks:
            if lvl != 2 or not any(k in title for k in keys):
                continue
            for ln in _bullets(lines):
                label, value = _label_value(ln)
                if _is_noise(label):
                    continue
                v = _compress(value if label else ln)
                # 术语表式的短行（「惊讶：！」）在这里被长度和标点挡掉
                if not v or len(v) < 14 or not re.search(r"[，。；]", v):
                    continue
                if any(_similar(v, o) for o in out):
                    continue
                out.append(f"- {v}")
                break
    return out


_EXTRACTORS = {
    "identity.md": _from_identity,
    "traits.md": _from_traits,
    "speech.md": _from_speech,
    "boundaries.md": _from_boundaries,
}


def candidates(root: Path | str, part: Part) -> list[str]:
    path = Path(root) / "wiki" / part.page
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log_.warning("读不到 %s：%s", path, exc)
        return []
    if PLACEHOLDER in text and not any(
        not _has_placeholder(ln) for ln in _bullets(_strip_frontmatter(text).splitlines())
    ):
        return []  # 整页还是空骨架
    fn = _EXTRACTORS.get(part.page)
    return fn(text) if fn else []


# ---------- 定位 index.md 里的小节 ----------


@dataclass(slots=True)
class Spot:
    """一个小节在 index.md 里的位置：标题行 + 正文行区间。"""

    title: str
    head: int  # 标题行号
    start: int  # 正文起始行号（含）
    end: int  # 正文结束行号（不含）
    lines: list[str] = field(default_factory=list)


def _spots(lines: list[str]) -> dict[str, Spot]:
    out: dict[str, Spot] = {}
    for part in PARTS:
        for i, line in enumerate(lines):
            if not line.startswith("### "):
                continue
            title = line[4:].strip()
            if part.title not in title:
                continue
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("## ") or lines[j].startswith("### "):
                    end = j
                    break
            out[part.title] = Spot(part.title, i, i + 1, end, lines[i + 1 : end])
            break
    return out


def _state_path(root: Path | str) -> Path:
    return Path(root) / STATE_REL


def read_state(root: Path | str) -> dict[str, Any]:
    path = _state_path(root)
    if not path.is_file():
        return {"sections": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"sections": {}}
    if not isinstance(data, dict) or not isinstance(data.get("sections"), dict):
        return {"sections": {}}
    return data


def write_state(root: Path | str, state: dict[str, Any]) -> Path:
    from . import ingest  # 复用原子写，避免两套实现漂移

    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ingest._atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2))  # noqa: SLF001
    return path


# ---------- 合并 ----------


def merge(cur: list[str], cands: list[str], written: list[str], limit: int) -> tuple[list[str], list[str], list[str]]:
    """返回 (新清单, 新增的行, 被淘汰的行)。

    `cur` 是当前小节里的清单行，`written` 是上次程序自己写的那些。
    `cur` 里不在 `written` 的一律当人工行：留下，且不参与淘汰。
    """
    owned = set(written)
    keep: list[str] = []
    dropped: list[str] = []
    for line in cur:
        if _has_placeholder(line):
            continue  # 模板占位，第一次写就被换掉
        if line in owned:
            if any(_similar(line, c) for c in cands):
                keep.append(line)  # 来源还在，继续用
            else:
                dropped.append(line)  # 页里删了这条，总括段也跟着退场
        else:
            keep.append(line)
    added: list[str] = []
    for cand in cands:
        if len(keep) + len(added) >= limit:
            break
        if any(_similar(cand, k) for k in keep + added):
            continue
        added.append(cand)
    return keep + added, added, dropped


# ---------- 主流程 ----------


def rebuild(
    root: Path | str,
    *,
    run_id: str = "",
    stamp: str = "",
    at: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """重建 index.md 总括段。没变化就不写盘，也不留备份目录。"""
    from . import ingest  # 延迟导入：ingest.run 里用到本模块

    root = Path(root)
    path = root / INDEX_NAME
    if not path.is_file():
        return {"ok": False, "error": f"没有 {INDEX_NAME}"}

    now = at or time.time()
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    date = time.strftime("%Y-%m-%d", time.localtime(now))

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"读不到 {INDEX_NAME}：{exc}"}

    lines = text.splitlines()
    spots = _spots(lines)
    state = read_state(root)
    sec_state: dict[str, Any] = dict(state.get("sections") or {})

    notes: list[str] = []
    plan: dict[str, list[str]] = {}  # 小节 → 新正文行
    spot_of: dict[str, Spot] = {}
    prog_next: dict[str, list[str]] = {}  # 小节 → 本次算作「程序写的」那些行
    changed_parts: list[str] = []

    for part in PARTS:
        spot = spots.get(part.title)
        if spot is None:
            notes.append(f"「{part.title}」小节在 index.md 里找不到，跳过")
            continue
        body = spot.lines
        cur = [ln.strip() for ln in body if ln.strip().startswith(("-", "*", "·"))]
        extra = [ln for ln in body if ln.strip() and ln.strip() not in cur]
        prev = sec_state.get(part.title) or {}
        written = [str(x) for x in (prev.get("lines") or [])]

        if not part.bullet:
            # 整段式（一句话人设）：只要已有实质文字就永久不碰
            if any(ln.strip() and not _has_placeholder(ln) for ln in body):
                continue
            cands = candidates(root, part)
            if not cands:
                notes.append(f"「{part.title}」：{part.page} 里还没有可用内容")
                continue
            plan[part.title] = cands[:1]
            spot_of[part.title] = spot
            prog_next[part.title] = cands[:1]
            changed_parts.append(f"{part.title} 首次填充")
            continue

        if extra:
            notes.append(f"「{part.title}」里有人工段落，整节不碰")
            continue

        cands = candidates(root, part)
        if not cands:
            notes.append(f"「{part.title}」：{part.page} 里还没有可用内容")
            continue

        new, added, dropped = merge(cur, cands, written, part.limit)
        if new == cur:
            continue
        human = [ln for ln in cur if ln not in set(written) and not _has_placeholder(ln)]
        plan[part.title] = new
        spot_of[part.title] = spot
        prog_next[part.title] = [ln for ln in new if ln not in human]
        if added:
            changed_parts.append(f"{part.title} +{len(added)}")
        if dropped:
            changed_parts.append(f"{part.title} −{len(dropped)}")

    if not plan:
        return {
            "ok": True,
            "changed": False,
            "reason": "；".join(notes) or "总括段已经是最新（wiki 里的内容它都提到了）",
            "notes": notes,
            "pages": [],
        }

    # 逐行改写：只换小节正文，其余字节不动（从后往前替换，行号才不漂）
    out = list(lines)
    for title, new in sorted(plan.items(), key=lambda kv: -spot_of[kv[0]].head):
        spot = spot_of[title]
        out[spot.start : spot.end] = [""] + new + [""]

    new_text = "\n".join(out)
    if not new_text.endswith("\n"):
        new_text += "\n"

    if dry_run:
        return {"ok": True, "changed": True, "dry_run": True, "summary": "、".join(changed_parts),
                "notes": notes, "pages": []}

    # 备份 → 写 → 追 log → 记 state（顺序同 ingest.commit）
    hist = ingest.history_dir(root) / stamp
    files = hist / "files"
    ingest._copy_into(path, files / INDEX_NAME)  # noqa: SLF001
    log_path = root / "log.md"
    if log_path.is_file():
        ingest._copy_into(log_path, files / "log.md")
    # 「哪些行是程序写的」也要能撤回来。原本没有就放一份空的 ——
    # 否则撤完 index.md 回到空稿，state 还记着程序写过一堆行，下次会认错主人。
    st_path = _state_path(root)
    if st_path.is_file():
        ingest._copy_into(st_path, files / STATE_REL)  # noqa: SLF001
    else:
        ingest._atomic_write(files / STATE_REL, json.dumps({"sections": {}}, ensure_ascii=False))  # noqa: SLF001
    hist.mkdir(parents=True, exist_ok=True)
    meta = {
        "stamp": stamp,
        "run_id": run_id,
        "at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "status": "writing",
        "files": [INDEX_NAME, "log.md", str(STATE_REL).replace("\\", "/")],
        "renames": [],
        # 本模块不碰归纳入库水位：让 undo 别把它清空
        "state_existed": True,
        "kind": "index",
        "parts": changed_parts,
    }
    ingest._atomic_write(hist / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))  # noqa: SLF001

    ingest._atomic_write(path, new_text)  # noqa: SLF001

    if log_path.is_file():
        old = log_path.read_text(encoding="utf-8", errors="replace")
        line = f"## [{date}] index | 重建总括段"
        detail = "\n".join(f"- {c}" for c in changed_parts) or "- 无"
        ingest._atomic_write(log_path, old.rstrip("\n") + f"\n\n{line}\n{detail}\n")  # noqa: SLF001

    new_state: dict[str, Any] = {}
    for title, prog in prog_next.items():
        state_entry = {"lines": prog, "at": time.strftime("%Y-%m-%d %H:%M", time.localtime(now))}
        new_state[title] = state_entry
    state["sections"] = {**sec_state, **new_state}
    state["at"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(now))
    state["run_id"] = run_id
    write_state(root, state)

    meta["status"] = "done"
    ingest._atomic_write(hist / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))  # noqa: SLF001

    return {
        "ok": True,
        "changed": True,
        "stamp": stamp,
        "run_id": run_id,
        "pages": [INDEX_NAME],
        "accepted": len(plan),
        "summary": "、".join(changed_parts),
        "notes": notes,
        "history": str(hist),
    }

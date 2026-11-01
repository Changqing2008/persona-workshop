"""人格库体检：确定性 lint，一条模型都不调。

归纳跑完直接落盘，中间没有人眼看。所以这道体检的定位是**事后拦一次**：
结构坏了、证据丢了、反应卡越界了，这些机器一眼能看出来的问题，别等到聊起来才发现。

只报能确定的事，不做"我觉得这段写得不好"这种判断。宁可少报，不可误报 ——
误报多了人就学会无视它，那这道闸就白设了。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import log
from .ingest_rules import judgement_hits

log_ = log.get("lint")

LEVEL_ORDER = {"error": 0, "warn": 1, "info": 2}
PLACEHOLDER = "_待填_"
FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
HEADING = re.compile(r"^(#{1,3})\s+(.*)$")

# 反应卡只许进这三页（agents.md 第九节）
CARD_PAGES = frozenset({"reactions.md", "speech.md", "relationships.md"})
CARD_MARKS = ("形态：反应样式", "形态:反应样式", "（反应样式稿", "(反应样式稿", "来源：反应")

# reactions.md 里并存两种卡，字段集不一样，任一成套就算合格：
# 人工卡（典型反应 / 情绪强度 + 出处）与蒸馏器的反应样式稿（条件 / 情绪 / 回应 / 露出）。
REACTION_SHAPES = (
    ("人工卡", (("典型反应",), ("情绪强度",))),
    ("反应样式卡", (("条件", "场景"), ("情绪",), ("回应",), ("露出",))),
)

# 这几节本来就不挂 raw（平台安全规则、结构说明），别报"没证据"
NO_EVIDENCE_SKIP = ("安全", "合规", "衔接")

# 这些页里的判定句必须有出处
EVIDENCE_PAGES = frozenset({"traits.md", "values.md", "boundaries.md"})

_LABEL = re.compile(r"^[\s>*\-•·]*([^：:]{1,10})[：:]\s*(.*)$")


@dataclass(slots=True)
class Finding:
    level: str  # error / warn / info
    code: str
    page: str
    line: int
    text: str
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "page": self.page,
            "line": self.line,
            "text": self.text,
            "why": self.why,
        }


def _body(text: str) -> str:
    return FRONTMATTER.sub("", text, count=1)


def _substantive(line: str) -> bool:
    s = line.strip()
    if not s or s.startswith((">", "|")):
        return False
    if set(s) <= set("|-: "):
        return False
    return True


def _blocks(text: str) -> list[tuple[int, str, list[tuple[int, str]]]]:
    """按 ## 切块：(级别, 标题, [(行号, 行内容)])。行号从 1 起，指原文。"""
    out: list[tuple[int, str, list[tuple[int, str]]]] = []
    level, title, buf = 0, "", []
    for i, line in enumerate(text.splitlines(), start=1):
        m = HEADING.match(line)
        if m and len(m.group(1)) <= 2:
            out.append((level, title, buf))
            level, title, buf = len(m.group(1)), m.group(2).strip(), []
        else:
            buf.append((i, line))
    out.append((level, title, buf))
    return out


def _label(line: str) -> str:
    m = _LABEL.match(line)
    return m.group(1).strip() if m else ""


# ---------- 各条检查 ----------


def check_judgement(page: str, text: str) -> list[Finding]:
    """性格断言：「她总是/一向/本质」这类词把情境规则写死了，agents.md 明令禁止。"""
    out: list[Finding] = []
    for i, line in enumerate(_body(text).splitlines(), start=1):
        if not _substantive(line):
            continue
        hits = judgement_hits(line)
        if hits:
            out.append(
                Finding("error", "judgement", page, i, line.strip()[:90],
                        f"性格断言「{'、'.join(hits)}」—— 该写成「在 X 情境 → Y 表现」的条件句")
            )
    return out


def check_placeholder(page: str, text: str) -> list[Finding]:
    """占位符残留：说明这页还没填，注入时会被整段剔掉。"""
    lines = [ln for ln in _body(text).splitlines() if PLACEHOLDER in ln]
    if not lines:
        return []
    return [
        Finding("warn", "placeholder", page, 0, f"{len(lines)} 行",
                "还有 _待填_，这页注入时会被整段剔掉")
    ]


def check_empty_sections(page: str, text: str) -> list[Finding]:
    """空节：`##` 底下一条实质内容都没有，结构残破。"""
    out: list[Finding] = []
    for level, title, buf in _blocks(_body(text)):
        if level != 2 or not title:
            continue
        if any(_substantive(ln) for _i, ln in buf):
            continue
        line_no = next((i for i, ln in buf if ln.strip()), 0)
        out.append(Finding("warn", "empty-section", page, line_no, f"## {title}",
                           "这一节底下没有内容"))
    return out


def check_no_evidence(page: str, text: str) -> list[Finding]:
    """整节没有出处：判定句全都不挂 raw 引文，等于凭印象写的。"""
    if page not in EVIDENCE_PAGES:
        return []
    out: list[Finding] = []
    for level, title, buf in _blocks(_body(text)):
        if level != 2 or not title:
            continue
        rows = [(i, ln) for i, ln in buf if _substantive(ln)]
        if len(rows) < 2:
            continue
        if any(k in title for k in NO_EVIDENCE_SKIP):
            continue
        body = "".join(ln for _i, ln in rows)
        if "raw/" in body or "来源" in body or "证据" in body:
            continue
        out.append(Finding("warn", "no-evidence", page, rows[0][0], f"## {title}",
                           f"这一节 {len(rows)} 行，没有任何 raw 引用"))
    return out


def check_reaction_fields(page: str, text: str) -> list[Finding]:
    """反应卡缺字段：`### 情境：…` 块必须齐 条件/情绪/回应/露出。"""
    if page != "reactions.md":
        return []
    lines = _body(text).splitlines()
    out: list[Finding] = []
    head, row, block = "", 0, []

    def flush() -> None:
        nonlocal head, row, block
        if not head:
            return
        labels = {_label(ln) for _i, ln in block if _substantive(ln)}
        if labels:
            name, groups = max(
                REACTION_SHAPES,
                key=lambda sh: sum(1 for g in sh[1] if any(x in labels for x in g)),
            )
            missing = [g[0] for g in groups if not any(x in labels for x in g)]
            if missing:
                out.append(Finding("warn", "reaction-missing", page, row, head,
                                   f"{name}缺 {'/'.join(missing)}"))
        head, block = "", []

    for i, line in enumerate(lines, start=1):
        if line.startswith("### "):
            flush()
            head, row = line[4:].strip(), i
            continue
        if line.startswith("## ") and head:
            flush()
            continue
        if head:
            block.append((i, line))
    flush()
    return out


def check_card_scope(page: str, text: str) -> list[Finding]:
    """反应卡越界：identity / boundaries 不该出现反应样式稿的东西。"""
    if page in CARD_PAGES:
        return []
    out: list[Finding] = []
    for i, line in enumerate(_body(text).splitlines(), start=1):
        if not _substantive(line):
            continue
        for mark in CARD_MARKS:
            if mark in line:
                out.append(Finding("error", "card-scope", page, i, line.strip()[:90],
                                   f"反应样式稿的内容进了 {page}（只许进 {'/'.join(sorted(CARD_PAGES))}）"))
                break
    return out


def check_index(root: Path) -> list[Finding]:
    """index.md：四小节在不在、填没填。总括段是每轮注入的东西，缺了就白收。"""
    path = root / "index.md"
    if not path.is_file():
        return [Finding("error", "index-missing", "index.md", 0, "", "没有 index.md，聊天时没有人格可用")]
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [Finding("error", "index-missing", "index.md", 0, "", f"读不到：{exc}")]

    from .indexer import PARTS  # 延迟导入：indexer 反过来要用 ingest

    out: list[Finding] = []
    lines = text.splitlines()
    for part in PARTS:
        hit = next((i for i, ln in enumerate(lines, start=1)
                    if ln.startswith("### ") and part.title in ln), 0)
        if not hit:
            out.append(Finding("error", "index-section", "index.md", 0, f"### {part.title}",
                               "总括段缺这一节，每轮注入会少一块"))
            continue
        body: list[str] = []
        for ln in lines[hit:]:
            if ln.startswith("## ") or ln.startswith("### "):
                break
            body.append(ln)
        if not any(_substantive(ln) and PLACEHOLDER not in ln for ln in body):
            out.append(Finding("warn", "index-empty", "index.md", hit, f"### {part.title}",
                               "这一节还是空的"))
    return out


_BULLET = re.compile(r"^[\s>*\-•·]+")


def _key(line: str) -> str:
    """去掉列表前缀再比 —— 人工可能把 `- x` 改成 `· x`，那还是同一条。"""
    return _BULLET.sub("", str(line)).strip()


def _clip(text: str, limit: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def check_index_body(root: Path) -> list[Finding]:
    """总括段的内容对不对得上 wiki，以及自己内部有没有重复。

    `index.md` 是每轮注入的那份，`data/index_state.json` 记着哪些行是程序写的。
    两件事这里能确定：

    - **脱节**：程序写过某行、这行现在还在 index.md 里，但 wiki 那页已经抽不出相近的
      候选了 —— 页改过、总括段没跟着重建。（人把整节换掉不算：那是他的权利，一条都
      对不上就跳过。）
    - **重复**：同一节两行归一化后一样。跟 wiki 无关也照样是错，白占名额。
    """
    path = root / "index.md"
    if not path.is_file():
        return []  # 缺文件已由 check_index 报 index-missing
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    from . import indexer  # 延迟导入：indexer 反过来要用 ingest

    spots = indexer._spots(text.splitlines())  # noqa: SLF001
    sec_state = (indexer.read_state(root).get("sections") or {})
    out: list[Finding] = []

    for part in indexer.PARTS:
        spot = spots.get(part.title)
        if spot is None:
            continue

        # 1) 同节重复
        seen: set[str] = set()
        for i, ln in enumerate(spot.lines):
            key = indexer._norm(_key(ln))  # noqa: SLF001
            if len(key) < 4:
                continue
            if key in seen:
                out.append(Finding("warn", "index-dupe", "index.md", spot.head + i + 2,
                                   _clip(_key(ln)),
                                   "这一节里有同样的两条，每轮注入白占一位"))
                break  # 一节只报一次，别刷屏
            seen.add(key)

        # 2) 与 wiki 脱节
        written = [str(x) for x in ((sec_state.get(part.title) or {}).get("lines") or [])]
        if not written:
            continue  # 程序没接管过这节，没得比
        here = {_key(ln) for ln in spot.lines if ln.strip()}
        live = [w for w in written if _key(w) in here]
        if not live or len(live) < len(written):
            # 有的已经被人删了：说明这节人在管，别拿程序的老账去说事
            continue
        cands = indexer.candidates(root, part)
        if not cands:
            continue  # 页本身空着，脱节无从谈起（另有 index-empty 报）
        stale = [w for w in live
                 if not any(indexer._similar(w, c) for c in cands)]  # noqa: SLF001
        if stale:
            out.append(Finding(
                "warn", "index-drift", "index.md", spot.head + 1, f"### {part.title}",
                f"{len(stale)} 条在 {part.page} 里找不到出处了（如「{_clip(_key(stale[0]))}」）"
                f" —— wiki 改过，总括段没跟着重建",
            ))
    return out


# ---------- 库契约检查（v1，见 人格工坊/contract/库契约.md） ----------

# 契约要求的顶层文件与目录。raw/ 缺了素材无处落，是 error；
# 其余缺了库还能转，降为 warn。
CONTRACT_TOP_FILES = ("agents.md", "index.md", "log.md", "preferences.toml")

# sessions.db 冻结 schema（v1）：缺表缺列是 error，多列是 warn。
SESSIONS_REQUIRED_TABLES = frozenset(
    {"messages", "sessions", "summaries", "seen", "profiles"}
)
SESSIONS_MESSAGE_COLUMNS = frozenset(
    {"id", "session_id", "role", "content", "ts", "msg_key", "user_id"}
)


def check_contract(root: Path) -> list[Finding]:
    """库契约：目录结构 + sessions.db 冻结 schema。只读，不做任何 DDL。"""
    out: list[Finding] = []
    for name in CONTRACT_TOP_FILES:
        if not (root / name).is_file():
            out.append(Finding("warn", "contract-missing", name, 0, "",
                               "契约要求的文件缺失"))
    if not (root / "raw").is_dir():
        out.append(Finding("error", "contract-missing", "raw/", 0, "",
                           "没有 raw/，素材无处落"))
    if not (root / "memory").is_dir():
        out.append(Finding("warn", "contract-missing", "memory/", 0, "", "记忆层缺失"))
    elif not (root / "memory" / "_hub").is_dir():
        out.append(Finding("warn", "contract-missing", "memory/_hub/", 0, "",
                           "缺四张分类索引（people/projects/domains/timeline）"))
    wiki = root / "wiki"
    if wiki.is_dir():
        for child in sorted(wiki.iterdir()):
            if child.is_dir():
                out.append(Finding("warn", "contract-layout", f"wiki/{child.name}", 0, "",
                                   "wiki/ 不该有子目录（记忆层与 wiki/ 平级，见 agents.md）"))
    db = root / "data" / "sessions.db"
    if db.is_file():
        out.extend(_check_sessions_schema(db))
    return out


def _check_sessions_schema(db: Path) -> list[Finding]:
    """对照冻结 schema 只读检查。WAL 库优先 ro 模式，失败再普通连接（只 SELECT）。"""
    import sqlite3

    out: list[Finding] = []
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        try:
            conn = sqlite3.connect(str(db))
        except sqlite3.Error as exc:
            return [Finding("error", "contract-schema", "data/sessions.db", 0, "",
                            f"打不开：{exc}")]
    assert conn is not None
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        missing = SESSIONS_REQUIRED_TABLES - tables
        if missing:
            out.append(Finding("error", "contract-schema", "data/sessions.db", 0, "",
                               f"缺表：{'、'.join(sorted(missing))}（schema 已冻结）"))
        if "messages" in tables:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
            lack = SESSIONS_MESSAGE_COLUMNS - cols
            if lack:
                out.append(Finding("error", "contract-schema", "data/sessions.db", 0, "",
                                   f"messages 缺列：{'、'.join(sorted(lack))}（schema 已冻结）"))
            extra = cols - SESSIONS_MESSAGE_COLUMNS
            if extra:
                out.append(Finding("warn", "contract-schema", "data/sessions.db", 0, "",
                                   f"messages 有契约外的列：{'、'.join(sorted(extra))}"
                                   "（通用性走命名空间约定，别加列）"))
    except sqlite3.Error as exc:
        out.append(Finding("error", "contract-schema", "data/sessions.db", 0, "",
                           f"读取失败：{exc}"))
    finally:
        conn.close()
    return out


# ---------- 总入口 ----------


PAGE_FILES = (
    "identity.md", "traits.md", "speech.md", "reactions.md", "values.md",
    "relationships.md", "boundaries.md", "exemplars.md",
)


def run(root: Path | str, pages: Iterable[str] | None = None) -> dict[str, Any]:
    """扫一遍 wiki 八页 + index.md。不写盘、不改任何东西。"""
    root = Path(root)
    wiki = root / "wiki"
    findings: list[Finding] = []

    for name in pages or PAGE_FILES:
        path = wiki / name
        if not path.is_file():
            findings.append(Finding("error", "page-missing", name, 0, "", "这一页不存在"))
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append(Finding("error", "page-missing", name, 0, "", f"读不到：{exc}"))
            continue
        if not text.strip():
            findings.append(Finding("error", "page-missing", name, 0, "", "这一页是空的"))
            continue
        for fn in (check_judgement, check_card_scope, check_placeholder,
                   check_empty_sections, check_no_evidence, check_reaction_fields):
            findings.extend(fn(name, text))

    findings.extend(check_index(root))
    findings.extend(check_index_body(root))
    findings.extend(check_contract(root))
    findings.sort(key=lambda f: (LEVEL_ORDER.get(f.level, 9), f.page, f.line))

    errors = sum(1 for f in findings if f.level == "error")
    warns = sum(1 for f in findings if f.level == "warn")
    log_.info("体检：%d error / %d warn", errors, warns)
    return {
        "ok": True,
        "root": str(root),
        "errors": errors,
        "warns": warns,
        "items": [f.to_dict() for f in findings],
        "summary": _summary(errors, warns),
    }


def _summary(errors: int, warns: int) -> str:
    if not errors and not warns:
        return "体检通过"
    bits = []
    if errors:
        bits.append(f"{errors} error")
    if warns:
        bits.append(f"{warns} warn")
    return "体检 " + " ".join(bits)


def brief(result: dict[str, Any] | None) -> str:
    """结果条上那一小截文案。"""
    if not result or not result.get("ok"):
        return ""
    return _summary(int(result.get("errors") or 0), int(result.get("warns") or 0))

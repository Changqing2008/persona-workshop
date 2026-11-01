"""一键归纳：扫 `raw/` 判状态、出补丁、过闸、落盘。

三段分工：

1. `scan()`（只读）—— 扫目录、解析成品头、比对水位、从 `log.md` 反查历史
2. `ingest_one()`（读 + 调模型）—— 一份素材 → 补丁 → 三道闸拦一遍
3. `commit()` / `undo()`（写）—— 备份 → 原子写 → 追 log → 推水位 → 改名，可整份回滚

水位（`data/ingest_state.json`）与文件名后缀（`*.已归纳.md`）互为备份：
- 资源管理器里一眼看出哪些收过 → 靠后缀
- 状态文件丢了能整份恢复 → 靠后缀反推
- 后缀被手删了也能认出来 → 靠水位

写入面只有三个：`wiki/*.md`、`log.md`（只追加）、`data/`（水位与备份）。
`raw/` 的内容一个字节都不改，只改名加 `.已归纳` —— 这是硬约束，写在这里防后人破例。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from . import ingest_prompt
from . import ingest_rules as R
from .llm import LLMClient, LLMError

RAW_DIR = "raw"
ASSETS_DIR = "assets"
WIKI_DIR = "wiki"
STATE_REL = Path("data") / "ingest_state.json"
HISTORY_REL = Path("data") / "wiki_history"

MIN_BODY = 50
"""正文短于此视为空素材（只有标题和头），收不出东西，别浪费一次调用。"""

QUOTE_MIN = 8
"""正文引文最短长度。太短的引文在哪都能搜到，等于没证据。"""

HEAD_QUOTE_MIN = 5
"""成品头字段引文的最短长度。

反应卡的料全在头里（`话量：少（一句一条）` 归一化后只有 7 个字），
按正文的 8 字卡会把整份有效素材拦掉。头字段因此单独放行，但**只认逐字包含**，
不给模糊兜底 —— 头字段是结构化原文，改写它就是编。
"""

INGEST_MAX_TOKENS = 6000
"""归纳单份的输出上限。

四条补丁本身约 900 token，但推理型模型（deepseek-flash / R1 这类）会先花 3~4k 思考，
给 2400 的结果是「思考用完，正文是空的」。6000 对非推理模型没有坏处（它不会写满），
对推理模型刚好够。
"""

STATUS_NEW = "new"
STATUS_DONE = "done"
STATUS_CHANGED = "changed"
STATUS_BLOCKED = "blocked"

STATUS_LABEL = {
    STATUS_NEW: "待归纳",
    STATUS_DONE: "已归纳",
    STATUS_CHANGED: "改过",
    STATUS_BLOCKED: "拦住",
}

_LOG_INGEST = re.compile(r"ingest\s*[|｜]\s*(.+)$", re.IGNORECASE)
_LOG_NOTE = re.compile(r"[（(][^）)]*[）)]")
_LOG_DATE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_RAW_PREFIX = re.compile(r"^\s*raw[/\\]")
_MEM_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_MEM_TYPE = re.compile(r"^type:\s*(person|project|domain|event|timeline)\b", re.MULTILINE | re.IGNORECASE)
_SPEAKER = re.compile(r"\*\*[^*\n]{1,24}\*\*\s*[:：]")


@dataclass(slots=True)
class Asset:
    name: str
    """磁盘上的文件名，可能带 `.已归纳`。"""
    plain: str
    """去掉后缀的名字，水位与 log 比对都用它。"""
    stem: str
    """不带扩展名，log 里记的就是这个。"""
    kind: str
    level: str
    head: dict[str, str]
    status: str
    reason: str = ""
    sha: str = ""
    chars: int = 0
    mtime: float = 0.0
    cap: str = R.CONF_MEDIUM
    """这份素材最高能撑到几级置信。"""
    via: str = ""
    """状态依据：`suffix` / `state` / `log` / `content` / `identity`。"""
    text: str = field(default="", repr=False)
    """原文，只在扫料这一轮暂存，不进 to_dict —— 否则接口一次吐几十 KB 正文。"""

    @property
    def label(self) -> str:
        return STATUS_LABEL.get(self.status, self.status)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text", None)  # 正文留在内存里给 ingest_one 用，接口不吐几十 KB
        d["label"] = self.label
        d["kind_label"] = R.KIND_LABEL.get(self.kind, self.kind)
        return d


@dataclass(slots=True)
class Scan:
    root: str
    items: list[Asset] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    log_hits: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out = {STATUS_NEW: 0, STATUS_DONE: 0, STATUS_CHANGED: 0, STATUS_BLOCKED: 0}
        for it in self.items:
            out[it.status] = out.get(it.status, 0) + 1
        return out

    def pending(self) -> list[Asset]:
        """真正该跑的那批：只算 `new`。`changed` 只提示，不自动重收。"""
        return [it for it in self.items if it.status == STATUS_NEW]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "items": [it.to_dict() for it in self.items],
            "counts": self.counts,
            "pending": len(self.pending()),
            "log_hits": self.log_hits,
            "warnings": self.warnings,
            "state_path": str(Path(self.root) / STATE_REL),
        }


# ---------- 水位 ----------


def state_path(root: Path | str) -> Path:
    return Path(root) / STATE_REL


def read_state(root: Path | str) -> dict[str, Any]:
    path = state_path(root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write(path: Path, text: str) -> None:
    """先写同目录临时文件再 `os.replace`。中途断电/被杀，原文件一个字都不变。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".ingest-tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_state(root: Path | str, state: dict[str, Any]) -> Path:
    path = state_path(root)
    _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2))
    return path


# ---------- log 反查 ----------


def log_ingested(root: Path | str) -> list[str]:
    """`log.md` 里被 `ingest |` 记过的素材 stem。

    库里两种写法都有，都得认：
        ## [2026-10-24] ingest | 2026-10-23_群里那两条还没回
        - 2026-10-24 ingest | raw/2026-10-24_替人背锅.md
    括号注释（`raw/identity.md（身份母本）`）先剥掉再取 basename。
    """
    return log_groups(root)[0]


def log_groups(root: Path | str) -> tuple[list[str], set[str]]:
    """返回 (精确命中的 stem 列表, 按批收过的日期集合)。

    第二种是给整批记录的兜底：`## [2026-10-25] ingest | 反应样式稿五份（…）`
    这种没写文件名的流水，靠"日期 + 反应样式稿 + N 份"三项对上当天那批反应卡，
    免得一键归纳把已经整批收过的素材再收一遍（白烧五次调用）。
    **只对反应样式稿生效**，别的人格素材宁可不认，也不能漏收。
    """
    path = Path(root) / "log.md"
    if not path.is_file():
        return [], set()
    stems: list[str] = []
    reaction_days: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LOG_INGEST.search(line)
        if not m:
            continue
        raw_desc = m.group(1)
        got = _LOG_NOTE.sub("", raw_desc).strip()
        got = _RAW_PREFIX.sub("", got)
        got = got.replace("\\", "/").split("/")[-1].strip()
        got = R.plain_stem(got)
        if got and got not in stems:
            stems.append(got)
        if "反应" in raw_desc and re.search(r"[一二三四五六七八九十\d]+\s*份", raw_desc):
            day = _LOG_DATE.search(line.split("|")[0])
            if day:
                reaction_days.add(day.group(1))
    return stems, reaction_days


# ---------- 扫料 ----------


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _body(text: str) -> str:
    """去掉一级标题与成品头之后剩下的正文，用来判"是不是空素材"。"""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(">"):
            continue
        lines.append(s)
    return "\n".join(lines)


def _is_mem_page(text: str) -> bool:
    """记忆层页被误塞进 raw 的识别（frontmatter 里有 `type: person` 之类）。

    这种文件是 `memory/people/xxx.md` 走错了地方，收进 wiki 会把"她是谁"写成一堆事实条目。
    """
    m = _MEM_FRONT.match(text)
    return bool(m and _MEM_TYPE.search(m.group(1)))


def _has_content(asset: Asset, text: str) -> tuple[bool, str]:
    """这份素材到底有没有东西可收。比"字数够不够"准，因为反应卡正文本来就是空的。"""
    if asset.head.get("审", "").strip() == R.AUDIT_FAIL:
        return False, "成品头 `审：未过` —— 没过审的整份不收"
    if asset.kind == R.KIND_REACTION:
        # 反应样式稿的料全在头里（第九节），正文为空是正常形态
        if not R.dirty_head(asset.head.get("刺激", "")) and (
            not R.dirty_head(asset.head.get("场景", ""))
            or not R.dirty_head(asset.head.get("回应", ""))
        ):
            return True, ""
        return False, "反应样式稿的头是空的（`刺激` 与 `场景/回应` 都没填）"
    if _is_mem_page(text):
        return False, "这是记忆层的页（frontmatter 标了 type），该进 memory/ 不是 wiki/"
    body = _body(text)
    if len(body) >= MIN_BODY or _SPEAKER.search(body):
        return True, ""
    return False, f"正文只有 {len(body)} 字，也没有对白行"


def _log_anchor(log_stem: str) -> str:
    """log 里常写缩短名：`第一卷1章·……唔～……`，取 `·` 之后那段当锚。"""
    if "·" not in log_stem:
        return ""
    anchor = log_stem.rsplit("·", 1)[-1].strip()
    return anchor if len(anchor) >= 4 else ""


def log_match(stem: str, log_stems: list[str], all_stems: list[str]) -> str:
    """素材是否被 log 记过；命中返回 log 里那条的名字，没命中返回空串。

    精确比对之外再认一种：log 写了缩短名（`第一卷1章·……唔～……`），
    素材全名以 `·` 后那段结尾。锚太短的不要（`·第1章` 会命中一堆），
    同一锚命中多份也不认 —— 宁可多收一遍，不能漏收。
    """
    for ls in log_stems:
        if ls == stem:
            return ls
    for ls in log_stems:
        anchor = _log_anchor(ls)
        if not anchor or not stem.endswith(anchor):
            continue
        if sum(1 for other in all_stems if other.endswith(anchor)) > 1:
            continue
        return ls
    return ""


def judge(
    asset: Asset,
    state: dict[str, Any],
    hits: tuple[list[str], set[str]],
    text: str,
    all_stems: list[str],
) -> None:
    """定状态。顺序即优先级，改顺序要想清楚。"""
    stems, reaction_days = hits
    rec = state.get(asset.stem) or {}

    if R.is_ingested(asset.name):
        asset.status = STATUS_DONE
        asset.via = "suffix"
        if rec.get("sha") and rec["sha"] != asset.sha:
            asset.status = STATUS_CHANGED
            asset.via = "suffix+sha"
            asset.reason = "收了之后内容又改过（母本反复改写属正常，其余素材不该变）"
        return

    if rec:
        if rec.get("sha") == asset.sha:
            asset.status = STATUS_DONE
            asset.via = "state"
        else:
            asset.status = STATUS_CHANGED
            asset.via = "state"
            asset.reason = "水位里有记录但内容变了"
        return

    hit = log_match(asset.stem, stems, all_stems)
    if hit:
        asset.status = STATUS_DONE
        asset.via = "log"
        if hit != asset.stem:
            asset.reason = f"log 里记的是缩短名「{hit}」"
        return

    if asset.kind == R.KIND_REACTION and asset.stem[:10] in reaction_days:
        asset.status = STATUS_DONE
        asset.via = "log-group"
        asset.reason = "log 里按「反应样式稿 N 份」整批收过（只对反应卡这么认）"
        return

    ok, why = _has_content(asset, text)
    if not ok:
        asset.status = STATUS_BLOCKED
        asset.via = "content"
        asset.reason = why
        return

    asset.status = STATUS_NEW
    asset.via = ""


def scan(root: Path | str) -> Scan:
    """扫一遍 `raw/`。**零副作用**：不建目录、不写状态、不改文件名。"""
    root = Path(root)
    out = Scan(root=str(root))
    raw = root / RAW_DIR
    if not raw.is_dir():
        out.warnings.append(f"没有人格素材目录：{raw}")
        return out

    out.state = read_state(root)
    hits = log_groups(root)
    out.log_hits = hits[0]

    for path in sorted(raw.glob("*.md")):
        if path.name.startswith("."):
            continue
        try:
            data = path.read_bytes()
            text = data.decode("utf-8", errors="replace")
        except OSError as exc:
            out.warnings.append(f"读不了 {path.name}：{exc}")
            continue

        head = R.parse_head(text)
        out.items.append(Asset(
            name=path.name,
            plain=R.plain_name(path.name),
            stem=R.plain_stem(path.name),
            kind=R.kind_of(path.name, head),
            level=R.level_of(head),
            head=head,
            status=STATUS_NEW,
            sha=_sha(data),
            chars=len(text),
            mtime=path.stat().st_mtime,
            cap=R.cap_for(head),
            text=text,
        ))

    # 判状态要看到全部素材名（log 缩短名匹配要防同一锚命中多份），所以扫完再判
    all_stems = [it.stem for it in out.items]
    for asset in out.items:
        judge(asset, out.state, hits, asset.text, all_stems)
        if asset.kind == R.KIND_IDENTITY:
            # 母本人工反复改写，不进队列；归纳时只作"参考现状"读进来
            asset.status = STATUS_DONE
            asset.via = "identity"
            asset.reason = "身份母本由人工维护，不做一键归纳"

    if not out.items:
        out.warnings.append("raw/ 里还没有 .md 素材")
    return out


def pending(root: Path | str) -> list[Asset]:
    return scan(root).pending()


def state_for_commit(scan_result: Scan, items: list[Asset], at: float | None = None) -> dict[str, Any]:
    """给 commit 用的新水位：在旧水位上合并这批素材。纯函数，不落盘。"""
    state = dict(scan_result.state)
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(at or time.time()))
    for it in items:
        old = state.get(it.stem) or {}
        state[it.stem] = {
            "sha": it.sha,
            "at": stamp,
            "name": it.plain,
            "kind": it.kind,
            "pages": old.get("pages", []),
            "items": old.get("items", 0),
            "rejected": old.get("rejected", 0),
        }
    return state


# ---------- 补丁 ----------

_KEEP = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
"""比对时只留字：空白、标点、markdown 修饰符全去掉。

标点必须一起去 —— 模型抄长句时常把 `，` 写成 `,` 或干脆不加，
留着标点比就只有"逐字照抄"能过，真素材会被误拦。
"""


def _norm(text: str) -> str:
    return _KEEP.sub("", str(text))


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+]\s+")
_ANY_HEADING = re.compile(r"^\s*#{1,6}\s", re.MULTILINE)
_KNOB_FULL = (
    "称呼", "直接度", "话量", "解释不解释", "主动追问", "情绪露出",
    # 模板里写成问句/长名，两种都得认，否则模型照模板抄反而被拦
    "会不会主动追问", "允许露出的情绪",
)
_KNOB_LINE = re.compile(
    r"^(?:[-*+]\s*)?("
    + "|".join(re.escape(k) for k in sorted(_KNOB_FULL, key=len, reverse=True))
    + r")\s*[：:]"
)


@dataclass(slots=True)
class Patch:
    """一条补丁。`reject` 非空 = 被闸拦下，不落盘但要进报告给人看。"""

    op: str = "append"
    page: str = ""
    section: str = ""
    title: str = ""
    text: str = ""
    quote: str = ""
    confidence: str = ""
    why: str = ""
    source: str = ""
    """素材文件名，写进 conflict 标记行做溯源。"""
    reject: str = ""
    mode: str = ""
    """落盘时的实际动作：`append` / `new-sub` / `merge` / `new-section`。"""

    @property
    def ok(self) -> bool:
        return not self.reject

    @property
    def writes(self) -> bool:
        return not self.reject and self.op != "skip"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op, "page": self.page, "section": self.section,
            "title": self.title, "text": self.text, "quote": self.quote,
            "confidence": self.confidence, "why": self.why,
            "reject": self.reject, "mode": self.mode, "ok": self.ok,
        }


def _json_object(raw: str) -> dict[str, Any] | None:
    """模型常包一层 ```json，也不保证只输出 JSON。取第一个完整对象。"""
    text = _FENCE.sub("", str(raw).strip()).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_patch(raw: str) -> tuple[list[Patch], str]:
    """模型输出 → 补丁列表 + 整体备注。解析失败返回空列表，由调用方按失败处理。"""
    data = _json_object(raw)
    if data is None:
        return [], ""
    out: list[Patch] = []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        out.append(Patch(
            op=str(item.get("op") or "append").strip().lower(),
            page=str(item.get("page") or "").strip(),
            section=R.header_line(item.get("section") or "", 2),
            title=str(item.get("title") or "").strip(),
            text=str(item.get("text") or "").strip(),
            quote=str(item.get("quote") or "").strip(),
            confidence=str(item.get("confidence") or R.CONF_MEDIUM).strip().lower(),
            why=str(item.get("why") or "").strip(),
        ))
    return out, str(data.get("note") or "").strip()


def _is_head_quote(quote: str) -> bool:
    """`话量：少（一句一条）` —— 抄的是成品头某一行，不是正文。"""
    head = str(quote).strip().lstrip(">").strip()
    return any(head.startswith(key) for key in R.HEAD_KEYS)


def quote_ok(quote: str, source: str) -> bool:
    """证据回查。这是唯一能挡住"模型自己编一条"的闸。

    两种放行：
    - 成品头字段（`话量：少（一句一条）`）：短，但**只认逐字包含** —— 头是结构化原文，改写就是编
    - 正文引文：要求**有一段连续 ≥8 字对得上**，且整句只错几个字。
      只看"总字数对得上"不行 —— 素材两千字，随便挑十几个常用字都能凑够，
      那样这道闸等于没有。
    """
    q = _norm(quote)
    src = _norm(source)
    # 头字段先放行：它短，但只认逐字包含（头是结构化原文，改写就是编）。
    # 这一步是**额外的放行条件**，不是分支 —— 「对方」既是头字段也是正文常见开头，
    # 正文引文从「对方」起头时得能落到下面的正文规则上。
    if _is_head_quote(quote) and len(q) >= HEAD_QUOTE_MIN and q in src:
        return True
    if len(q) < QUOTE_MIN:
        return False
    if q in src:
        return True

    matcher = SequenceMatcher(None, q, src, autojunk=False)
    blocks = matcher.get_matching_blocks()
    longest = max((b.size for b in blocks), default=0)
    total = sum(b.size for b in blocks)
    return longest >= QUOTE_MIN and total >= len(q) - max(2, len(q) // 10)


def _bullets(text: str) -> list[str]:
    return [ln.strip() for ln in str(text).splitlines() if ln.strip()]


def _dup(text: str, page: str) -> bool:
    """页里已有同义条目就不再加一遍。比的是去掉符号后的整行包含。"""
    page_n = _norm(page)
    for line in _bullets(text):
        key = _norm(_BULLET.sub("", line))
        if len(key) >= 10 and key in page_n:
            return True
    return False


def gate(patch: Patch, asset: Asset, pages: dict[str, str]) -> str:
    """三道闸。返回拒收原因，空串表示通过。顺序即优先级。"""
    if patch.op not in R.OPS:
        return f"未知动作「{patch.op}」，只认 {'/'.join(R.OPS)}"
    if patch.op == "skip":
        return "" if patch.why else "skip 没写理由"
    if not patch.page:
        return "没给 page"
    if patch.page not in R.WIKI_PAGES:
        return f"页名不在白名单：{patch.page}"
    if patch.page not in R.allowed_pages(asset.kind):
        return f"这类素材不许写 {patch.page}"

    page = pages.get(patch.page) or ""
    if not page.strip():
        return f"页不存在或为空：{patch.page}"
    if not patch.text:
        return "text 空着"
    if _ANY_HEADING.search(patch.text):
        return "text 里带了标题行（结构由程序加，模型不许改）"
    if not patch.section:
        return "没给 section"

    existing = [R.header_line(s, 2) for s in R.sections_of(page, 2)]
    if patch.section not in existing and patch.page not in R.NEW_SECTION_PAGES:
        return f"{patch.page} 里没有这一节：{patch.section}"

    if not quote_ok(patch.quote, asset.text):
        return "quote 在素材原文里找不到（证据对不上，不收）"

    hits = R.judgement_hits(patch.text) or R.judgement_hits(patch.title)
    if hits:
        return f"含性格断言「{hits[0]}」"

    if asset.kind == R.KIND_REACTION:
        hits = R.proper_hits(patch.text + " " + patch.title)
        if hits:
            return f"含专名/称谓「{hits[0]}」（反应卡不许带具体身份）"
        if patch.page == "exemplars.md":
            if asset.head.get("范例", "").strip() != "已核":
                return "反应卡没勾「范例：已核」，不许进 exemplars"
            if patch.section != R.header_line("行为范例", 2):
                return "反应卡只能进 exemplars 的「## 行为范例」"
        if patch.page == "relationships.md":
            bad = [ln for ln in _bullets(patch.text) if not _KNOB_LINE.match(ln)]
            if bad:
                return f"关系页只收六个旋钮，这条不是：{bad[0][:24]}"

    if _dup(patch.text, page):
        return "页里已有同义条目"
    return ""


# ---------- 合页 ----------


def _tidy(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if not line.strip():
            if out and not out[-1].strip():
                continue
            out.append("")
        else:
            out.append(line.rstrip())
    while out and not out[-1].strip():
        out.pop()
    return out


def _block_of(patch: Patch) -> list[str]:
    body = [ln.rstrip() for ln in patch.text.splitlines()]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return []
    if patch.op == "conflict":
        src = patch.source or "素材"
        body = body + ["", f"- ⚠️ 与既有说法冲突，两说并存待人工裁（source: {src}）"]
    return body


def _touch_updated(text: str, date: str) -> str:
    """刷新 frontmatter 的 `updated:`。缺失就补在 `---` 之前 —— 只动这一行。"""
    if re.search(r"^updated:.*$", text, flags=re.MULTILINE):
        return re.sub(r"^updated:.*$", f"updated: {date}", text, count=1, flags=re.MULTILINE)
    return text


def apply_patch(page_text: str, patch: Patch) -> tuple[str, str]:
    """把一条补丁合进页文本。返回 (新文本, 实际动作)。

    只做插入，不做替换 —— 函数里没有任何一行会覆盖已有内容。
    """
    lines = page_text.rstrip("\n").splitlines()
    block = _block_of(patch)
    if not block:
        return page_text, ""

    idx = next((i for i, ln in enumerate(lines) if ln.rstrip() == patch.section), None)
    if idx is None:  # 新开一节：exemplars / reactions 才允许，闸里已经放过
        return "\n".join(_tidy(lines + ["", patch.section, ""] + block)) + "\n", "new-section"

    end = len(lines)
    for i in range(idx + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    while end > idx + 1 and not lines[end - 1].strip():
        end -= 1

    if patch.title:
        t_line = R.header_line(patch.title, 3)
        j = next((i for i in range(idx + 1, end) if lines[i].rstrip() == t_line), None)
        if j is not None:  # 并进已有子节的末尾
            sub_end = end
            for i in range(j + 1, end):
                if lines[i].startswith("### "):
                    sub_end = i
                    break
            while sub_end > j + 1 and not lines[sub_end - 1].strip():
                sub_end -= 1
            out = lines[:sub_end] + [""] + block + lines[sub_end:]
            return "\n".join(_tidy(out)) + "\n", "merge"
        out = lines[:end] + ["", t_line, ""] + block + lines[end:]
        return "\n".join(_tidy(out)) + "\n", "new-sub"

    out = lines[:end] + [""] + block + lines[end:]
    return "\n".join(_tidy(out)) + "\n", "append"


# ---------- 一份素材 ----------


@dataclass(slots=True)
class IngestResult:
    asset: Asset | None = field(default=None, repr=False)
    patches: list[Patch] = field(default_factory=list)
    note: str = ""
    error: str = ""
    raw: str = field(default="", repr=False)
    mark_done: bool = False
    """收过了 = 下一轮不再进来。模型异常/解析失败时为 False，素材留在队列。"""

    @property
    def accepted(self) -> list[Patch]:
        return [p for p in self.patches if p.writes]

    @property
    def rejected(self) -> list[Patch]:
        return [p for p in self.patches if not p.ok]

    @property
    def pages(self) -> list[str]:
        return sorted({p.page for p in self.accepted})

    def to_dict(self) -> dict[str, Any]:
        a = self.asset
        return {
            "asset": a.plain if a else "",
            "kind": a.kind if a else "",
            "kind_label": R.KIND_LABEL.get(a.kind, "") if a else "",
            "head": dict(a.head) if a else {},
            "caps": a.cap if a else "",
            "note": self.note,
            "error": self.error,
            "mark_done": self.mark_done,
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "conflicts": sum(1 for p in self.accepted if p.op == "conflict"),
            "pages": self.pages,
            "patches": [p.to_dict() for p in self.patches],
        }


def read_pages(root: Path | str) -> dict[str, str]:
    root = Path(root)
    out: dict[str, str] = {}
    for page in R.WIKI_PAGES:
        path = root / WIKI_DIR / page
        if path.is_file():
            out[page] = path.read_text(encoding="utf-8", errors="replace")
    return out


async def ingest_one(
    llm: LLMClient,
    root: Path | str,
    asset: Asset,
    *,
    model: str = "",
    max_tokens: int = 0,
) -> IngestResult:
    """一份素材 → 补丁 → 过闸。**不写任何文件**，落盘在 `commit()`。"""
    root = Path(root)
    res = IngestResult(asset=asset)
    try:
        messages = ingest_prompt.build_messages(root, asset)
    except Exception as exc:  # noqa: BLE001
        res.error = f"拼提示词失败：{exc}"
        return res

    budget = max_tokens or INGEST_MAX_TOKENS
    # 归纳比聊天慢得多：素材 + 整页内容进提示词，推理型模型还要先想几千字
    wait = max(float(getattr(llm.cfg, "timeout", 60.0)), 180.0)
    try:
        raw = await llm.chat(
            messages, temperature=0.2, max_tokens=budget, model=model or None, timeout=wait
        )
    except LLMError as exc:
        msg = str(exc)
        # 推理型模型思考长度不固定：同一份素材这次 3.7k、下次 6k+。
        # 撞上"思考吃光预算"时翻倍再来一次，比让人去翻配置强。
        if "全花在思考上" in msg and budget < 16000:
            bigger = min(budget * 2, 16000)
            try:
                raw = await llm.chat(
                    messages, temperature=0.2, max_tokens=bigger, model=model or None, timeout=wait
                )
            except LLMError as exc2:
                res.error = f"{exc2}（已把上限从 {budget} 提到 {bigger}，还是不够）"
                return res
        else:
            res.error = msg
            return res

    res.raw = raw
    res.patches, res.note = parse_patch(raw)
    if not res.patches and not res.note:
        res.error = "模型没给条目也没给原因（按失败处理，素材留在队列）"
        return res

    pages = read_pages(root)
    for patch in res.patches:
        patch.source = asset.plain
        patch.reject = gate(patch, asset, pages)
        if patch.writes:
            patch.confidence = R.clamp(patch.confidence, asset.head)
    res.mark_done = True
    return res


# ---------- 落盘 ----------


def history_dir(root: Path | str) -> Path:
    return Path(root) / HISTORY_REL


def _copy_into(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def log_block(results: list[IngestResult], date: str) -> str:
    """log 追加块。一段一份素材，和库里既有 `## [日期] ingest | 名字` 同格式。"""
    rows: list[str] = []
    for r in results:
        a = r.asset
        if a is None:
            continue
        body: list[str] = []
        if r.accepted:
            per: dict[str, int] = {}
            for p in r.accepted:
                per[p.page] = per.get(p.page, 0) + 1
            body.append(f"- 收 {len(r.accepted)} 条：" + " / ".join(f"{k} {v}" for k, v in per.items()))
            conflict = sum(1 for p in r.accepted if p.op == "conflict")
            if conflict:
                body.append(f"- 冲突 {conflict} 条（两说并存，已标待裁，没覆盖旧条目）")
        else:
            body.append(f"- 无可收条目：{r.note or '模型判断这份素材收不出东西'}")
        if r.rejected:
            reasons: dict[str, int] = {}
            for p in r.rejected:
                key = p.reject.split("（")[0].split("：")[0]
                reasons[key] = reasons.get(key, 0) + 1
            body.append("- 拦下 " + str(len(r.rejected)) + " 条：" + "、".join(reasons))
        rows.append("\n".join([f"## [{date}] ingest | {a.stem}"] + body))
    return "\n\n".join(rows) + "\n"


def commit(
    root: Path | str,
    results: list[IngestResult],
    *,
    run_id: str = "",
    stamp: str = "",
    at: float | None = None,
) -> dict[str, Any]:
    """落盘一轮。顺序：备份 → 写页 → 追 log → 推水位 → 改名（最后）。

    中间任何一步抛错，`meta.json` 已经在备份目录里写好，`undo()` 能整份还原。
    """
    root = Path(root)
    now = at or time.time()
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    date = time.strftime("%Y-%m-%d", time.localtime(now))
    hist = history_dir(root) / stamp
    files = hist / "files"

    todos = [r for r in results if r.asset is not None]
    if not todos:
        return {"ok": False, "error": "没有可提交的结果"}

    # 1) 快照原件（备份与改写用同一份，避免边写边读）
    originals: dict[str, str] = {}
    for r in todos:
        for p in r.accepted:
            if p.page in originals:
                continue
            path = root / WIKI_DIR / p.page
            originals[p.page] = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    if any(not v.strip() for v in originals.values()):
        missing = [k for k, v in originals.items() if not v.strip()]
        return {"ok": False, "error": f"页不存在或为空：{'、'.join(missing)}"}

    # 2) 合页
    new_texts = dict(originals)
    for r in todos:
        for p in r.accepted:
            new_texts[p.page], p.mode = apply_patch(new_texts[p.page], p)
    for page in new_texts:
        new_texts[page] = _touch_updated(new_texts[page], date)

    # 3) 改名计划（先算好写进 meta，改名真的做了没有都要能对上）
    renames: list[list[str]] = []
    for r in todos:
        if not r.mark_done or r.asset is None:
            continue
        a = r.asset
        src = root / RAW_DIR / a.name
        dst = root / RAW_DIR / R.ingested_name(a.name)
        if a.kind == R.KIND_IDENTITY or R.is_ingested(a.name) or src == dst or not src.is_file():
            continue
        renames.append([str(src.relative_to(root)), str(dst.relative_to(root))])

    state_existed = state_path(root).is_file()
    meta: dict[str, Any] = {
        "stamp": stamp,
        "run_id": run_id,
        "at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "status": "writing",
        "files": sorted(new_texts) + ["log.md"] + (["data/ingest_state.json"] if state_existed else []),
        "renames": renames,
        "state_existed": state_existed,
        "items": [
            {
                "asset": r.asset.plain if r.asset else "",
                "kind": r.asset.kind if r.asset else "",
                "accepted": len(r.accepted),
                "rejected": len(r.rejected),
                "pages": r.pages,
                "note": r.note,
                "error": r.error,
            }
            for r in todos
        ],
    }

    # 4) 备份（先于任何写入）
    for page, text in originals.items():
        _copy_into(root / WIKI_DIR / page, files / WIKI_DIR / page)
    log_path = root / "log.md"
    if log_path.is_file():
        _copy_into(log_path, files / "log.md")
    if state_existed:
        _copy_into(state_path(root), files / STATE_REL)
    hist.mkdir(parents=True, exist_ok=True)
    _atomic_write(hist / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

    # 5) 写页
    for page, text in new_texts.items():
        _atomic_write(root / WIKI_DIR / page, text)

    # 6) 追加 log（只追加，不重写历史）
    old = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else "# log\n"
    _atomic_write(log_path, old.rstrip("\n") + "\n\n" + log_block(todos, date))

    # 7) 推水位
    state = read_state(root)
    for r in todos:
        if not r.mark_done or r.asset is None:
            continue
        a = r.asset
        state[a.stem] = {
            "sha": a.sha,
            "at": time.strftime("%Y-%m-%d %H:%M", time.localtime(now)),
            "name": a.plain,
            "kind": a.kind,
            "pages": r.pages,
            "items": len(r.accepted),
            "rejected": len(r.rejected),
            "empty": not r.accepted,
        }
    write_state(root, state)

    # 8) 改名（最后一步：改名没成 = 这轮不算完，撤销时反向改名即可还原）
    renamed: list[list[str]] = []
    warnings: list[str] = []
    for src_rel, dst_rel in renames:
        try:
            os.replace(root / src_rel, root / dst_rel)
            renamed.append([src_rel, dst_rel])
        except OSError as exc:
            warnings.append(f"改名失败 {src_rel}：{exc}")

    meta["status"] = "done"
    meta["renamed"] = renamed
    meta["warnings"] = warnings
    meta["written"] = sorted(new_texts)
    _atomic_write(hist / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

    return {
        "ok": True,
        "stamp": stamp,
        "run_id": run_id,
        "pages": sorted(new_texts),
        "accepted": sum(len(r.accepted) for r in todos),
        "rejected": sum(len(r.rejected) for r in todos),
        "renamed": renamed,
        "warnings": warnings,
        "history": str(hist),
    }


# ---------- 撤销 ----------


def history(root: Path | str) -> list[dict[str, Any]]:
    """历史轮次，新的在前。给「撤回」按钮列可选。"""
    base = history_dir(root)
    if not base.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({
            "stamp": d.name,
            "at": meta.get("at", ""),
            "run_id": meta.get("run_id", ""),
            "status": meta.get("status", ""),
            "undone": bool(meta.get("undone")),
            "items": meta.get("items", []),
        })
    return out


def undo(root: Path | str, run_id: str = "") -> dict[str, Any]:
    """按备份整份回滚：文件字节级还原 + 文件名反向改回 + 水位回退。

    `run_id` 为空时只撤最后一轮；给了就撤这一轮的所有分片（新的先撤）。
    备份目录不删 —— 撤错了还能再对着看。
    """
    root = Path(root)
    base = history_dir(root)
    if not base.is_dir():
        return {"ok": False, "error": "没有可撤销的记录"}

    dirs = [d for d in sorted(base.iterdir(), key=lambda p: p.name, reverse=True) if d.is_dir()]
    if run_id:
        dirs = [d for d in dirs if d.name.startswith(run_id)]
    if not dirs:
        return {"ok": False, "error": f"没找到这轮记录：{run_id}"}

    picked: list[Path] = []
    for d in dirs:
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("undone") or meta.get("status") != "done":
            continue
        picked.append(d)
        if not run_id:
            break
    if not picked:
        return {"ok": False, "error": "这轮已经撤过了，或者当时没写完"}

    restored: list[str] = []
    renamed: list[list[str]] = []
    warnings: list[str] = []
    for d in picked:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        files = d / "files"
        if files.is_dir():
            for src in files.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(files)
                target = root / rel
                try:
                    _copy_into(src, target)  # 二进制原样拷回，连换行都不动
                    restored.append(str(rel))
                except OSError as exc:
                    warnings.append(f"还原失败 {rel}：{exc}")
        # 水位原本不存在 → 清空，而不是留一份"已收过"的假水位
        if not meta.get("state_existed"):
            write_state(root, {})
        # 反向改名
        for src_rel, dst_rel in meta.get("renamed") or []:
            old, new = root / src_rel, root / dst_rel
            if new.is_file() and not old.exists():
                try:
                    os.replace(new, old)
                    renamed.append([dst_rel, src_rel])
                except OSError as exc:
                    warnings.append(f"反向改名失败 {dst_rel}：{exc}")
            elif new.is_file() and old.exists():
                warnings.append(f"{src_rel} 已存在，{dst_rel} 保留没动")
        meta["undone"] = True
        meta["undone_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _atomic_write(d / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

    return {
        "ok": True,
        "stamps": [d.name for d in picked],
        "restored": sorted(set(restored)),
        "renamed": renamed,
        "warnings": warnings,
    }


# ---------- 跑一轮 ----------

_JOB: dict[str, Any] = {
    "running": False,
    "run_id": "",
    "root": "",
    "phase": "idle",
    "total": 0,
    "current": 0,
    "item": "",
    "started": 0.0,
    "lines": [],
    "results": [],
    "commits": [],
    "index": {},
    "lint": {},
    "error": "",
}
"""一轮归纳的运行态。进程内单例 —— 同时只跑一轮，两个按钮点两下不会并发写库。"""

_MAX_LINES = 300


def job() -> dict[str, Any]:
    out = dict(_JOB)
    out["lines"] = list(_JOB["lines"])
    out["results"] = list(_JOB["results"])
    out["commits"] = list(_JOB["commits"])
    return out


def running() -> bool:
    return bool(_JOB["running"])


def claim(run_id: str, root: str) -> bool:
    """先把位置占住再交给事件循环。

    `run()` 里的 running 检查要等协程真的被调度才执行，这中间有一次 HTTP 往返 ——
    连点两下按钮会两个请求都判"没在跑"，然后顺序执行两轮、把同一批素材写两遍。
    """
    if _JOB["running"]:
        return False
    _JOB.update({
        "running": True, "run_id": run_id, "root": root, "phase": "starting",
        "total": 0, "current": 0, "item": "", "started": time.time(),
        "lines": [], "results": [], "commits": [], "error": "",
    })
    return True


def _emit(text: str, level: str = "info") -> None:
    _JOB["lines"].append({"t": time.strftime("%H:%M:%S"), "level": level, "text": text})
    if len(_JOB["lines"]) > _MAX_LINES:
        del _JOB["lines"][: len(_JOB["lines"]) - _MAX_LINES]


async def run(
    root: Path | str,
    llm: LLMClient,
    *,
    limit: int = 0,
    model: str = "",
    max_tokens: int = 0,
    dry_run: bool = False,
    run_id: str = "",
    claimed: bool = False,
    auto_index: bool = True,
) -> dict[str, Any]:
    """一键归纳：逐份归纳、逐份落盘。

    一份一落盘而不是攒到最后一起写，原因是第二份的提示词要看见第一份的成果 ——
    攒着写会让同一批素材各写一遍同样的条目，去重也拦不住（每份补丁各自合法）。

    auto_index=False 时跳过总括段重建：库里 [ingest] auto_index 关掉就是它。
    """
    root = Path(root)
    if _JOB["running"] and not claimed:
        return {"ok": False, "error": "上一轮还没跑完"}

    run_id = run_id or _JOB.get("run_id") or time.strftime("%Y%m%d-%H%M%S")
    _JOB.update({
        "running": True, "run_id": run_id, "root": str(root),
        "phase": _JOB.get("phase") if claimed else "scan",
        "total": 0, "current": 0, "item": "",
        "started": time.time(), "error": "",
    })
    if not claimed:
        _JOB.update({"lines": [], "results": [], "commits": [], "index": {}, "lint": {}})
    try:
        scan_result = scan(root)
        queue = scan_result.pending()
        if limit and limit > 0:
            queue = queue[:limit]
        _JOB.update({"phase": "run", "total": len(queue)})
        _emit(f"队列 {len(queue)} 份（拦住 {scan_result.counts.get(STATUS_BLOCKED, 0)} 份跳过）")
        if not queue:
            _JOB["phase"] = "done"
            return {"ok": True, "run_id": run_id, "total": 0, "message": "没有待归纳的素材"}

        for i, asset in enumerate(queue, start=1):
            _JOB.update({"current": i, "item": asset.plain})
            _emit(f"[{i}/{len(queue)}] {asset.plain}")
            res = await ingest_one(llm, root, asset, model=model, max_tokens=max_tokens)
            _JOB["results"].append(res.to_dict())
            if res.error:
                _emit(f"    ✗ {res.error}", "error")
                continue
            _emit(
                f"    收 {len(res.accepted)} 条 / 拦 {len(res.rejected)} 条"
                + (f" · {res.note}" if res.note else "")
            )
            # 全是被拦/全判重复（accepted 空）也要落一次盘：把素材改名、推水位，
            # 否则下次点按钮它又算「新的」，每次白烧一次模型调用。
            if dry_run or not res.mark_done:
                continue
            got = commit(root, [res], run_id=run_id, stamp=f"{run_id}-{i:02d}")
            if got.get("ok"):
                _JOB["commits"].append(got)
                if got["pages"]:
                    _emit(f"    落盘 {got['stamp']}：{'/'.join(got['pages'])}")
                else:
                    _emit(f"    落盘 {got['stamp']}：页没动，只把素材标成已归纳")
                for w in got.get("warnings") or []:
                    _emit(f"    ⚠ {w}", "warn")
            else:
                _emit(f"    ✗ 落盘失败：{got.get('error')}", "error")

        # 全部素材收完后重建一次总括段。逐份重建没意义 —— 每份都要重排一遍
        # 那四小节，白跑；而且最后一份的内容会盖掉前面几份的判断。
        ix: dict[str, Any] = {}
        if not dry_run and not auto_index:
            _emit("总括段跳过（库内 [ingest] auto_index = false）")
        elif not dry_run:
            from . import indexer  # 延迟导入：indexer 反过来要用本模块的工具

            _JOB["phase"] = "index"
            _emit("重建 index.md 总括段…")
            try:
                ix = indexer.rebuild(root, run_id=run_id, stamp=f"{run_id}-ix")
            except Exception as exc:  # noqa: BLE001
                ix = {"ok": False, "error": str(exc)}
            _JOB["index"] = ix
            if not ix.get("ok"):
                _emit(f"    ⚠ 总括段没重建：{ix.get('error')}", "warn")
            elif ix.get("changed"):
                _emit(f"    总括段更新：{ix.get('summary') or '已写'}")
                for note in ix.get("notes") or []:
                    _emit(f"    · {note}")
            else:
                _emit(f"    总括段没动：{ix.get('reason')}")

        # 体检：确定性 lint，只报能确定的事。落盘后再扫一遍，
        # 因为归纳是无人值守的，结构坏了得留个后手能看见。
        if not dry_run:
            from . import lint as lint_mod

            _JOB["phase"] = "lint"
            _emit("体检中…")
            try:
                got_lint = lint_mod.run(root)
            except Exception as exc:  # noqa: BLE001
                got_lint = {"ok": False, "error": str(exc), "summary": ""}
            _JOB["lint"] = got_lint
            if got_lint.get("ok"):
                _emit(f"    {got_lint.get('summary')}", "error" if got_lint.get("errors") else "info")
                for item in (got_lint.get("items") or [])[:6]:
                    _emit(f"    · [{item['level']}] {item['page']}：{item['why']}", item["level"])
            else:
                _emit(f"    ⚠ 体检没跑成：{got_lint.get('error')}", "warn")

        _JOB["phase"] = "done"
        total_accepted = sum(c["accepted"] for c in _JOB["commits"])
        total_rejected = sum(r.get("rejected", 0) for r in _JOB["results"])
        marks: list[str] = []
        if ix.get("changed"):
            marks.append("总括段已更新")
        elif not dry_run and not auto_index:
            marks.append("总括段跳过")
        lint_result = _JOB.get("lint") or {}
        if lint_result.get("ok") and lint_result.get("summary"):
            marks.append(str(lint_result["summary"]))
        if dry_run:
            marks.append("dry-run，未落盘")
        tail = "".join(f" · {m}" for m in marks if m)
        _emit(f"完成：{len(queue)} 份 → 改 {total_accepted} 条 · 拦 {total_rejected} 条{tail}")
        return {
            "ok": True,
            "run_id": run_id,
            "total": len(queue),
            "accepted": total_accepted,
            "rejected": total_rejected,
            "index": ix,
            "lint": _JOB.get("lint") or {},
            "dry_run": dry_run,
        }
    except Exception as exc:  # noqa: BLE001
        _JOB["error"] = str(exc)
        _JOB["phase"] = "error"
        _emit(f"✗ 中断：{exc}", "error")
        return {"ok": False, "error": str(exc), "run_id": run_id}
    finally:
        _JOB["running"] = False

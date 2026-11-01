"""身份母本（raw/identity.md）的读写通道。

人格库对程序是只读的，这里是唯一例外，而且只允许写这一个路径：
`<persona_dir>/raw/identity.md`。

分工：
- 母本由人写。程序只负责落盘、备份、报状态，不生成、不改写内容。
- 归纳AI 读母本提炼成 wiki/identity.md，可重排结构与润色措辞，
  但不得删改事实点；冲突标 conflict 交人工裁决。
- 身份页保存后即落盘，归纳AI 何时 ingest 由人决定。

留白的节写 `_待填_`，与 wiki 的占位符同一套约定（复用 persona.PLACEHOLDER_TOKENS），
这样归纳AI 和聊天端都能用同一套规则识别"这节还没写"。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from . import log
from .persona import PLACEHOLDER_TOKENS

log_ = log.get("seed")

# 母本文件名固定，不受 raw 的 `YYYY-MM-DD_描述.md` 命名规范约束。
IDENTITY_REL = Path("raw") / "identity.md"

# 存储上限。够写一份很细的人设，也挡住误粘贴整本书。
MAX_CHARS = 20000

# 历史备份保留份数。多了没用，翻不到那么远。
KEEP_HISTORY = 50

# 七节骨架。顺序有讲究：先定"她是谁"，再定"她日子怎么过"，
# 最后用"不存在的东西"收口，避免模型自己补一套人生。
# 场景锚在 QQ 文字聊天：关系只分三档，不写具体人物网；
# 三档怎么说话由 wiki/relationships.md 管，母本只给"谁算哪档"的事实。
SECTIONS: tuple[tuple[str, str], ...] = (
    ("一句话是谁", "年龄 / 职业 / 在哪 / 现在在忙什么"),
    ("生活半径", "常去的三个地方、日常节奏、一天里什么时候挂着 QQ"),
    ("关系三档", "拥有者 / 熟人 / 陌生人各一句，不列人名清单"),
    ("说话习惯（QQ 文字）", "发几条多长、口头禅、标点表情、称呼、群里还是私聊"),
    ("底线雷区", "绝对不做的三件事"),
    ("价值观与立场", "信念 / 在意 / 不在意 / 底线执念，各一句"),
    ("想要与怕", "今年想达成什么、怕什么"),
    ("不存在的东西", "这个世界里没有的人 / 事 / 说法"),
)

TEMPLATE = (
    "# identity — 身份母本\n"
    "\n"
    "> 人工撰写。归纳AI 只许完善措辞与结构，不许删改事实点；冲突标 conflict，交人工裁决。\n"
    "> 这份人设只在 QQ 上活着：文字消息、日常闲聊、群和私聊。关系只有三档，不建复杂的人物网。\n"
    "\n"
    "## 一句话是谁\n"
    "_待填_（年龄 / 职业 / 在哪 / 现在在忙什么）\n"
    "\n"
    "## 生活半径\n"
    "_待填_（常去的三个地方、日常节奏、一天里什么时候挂着 QQ）\n"
    "\n"
    "## 关系三档\n"
    "> 只写\"这一档在她眼里算什么、默认保持多远\"，不列人名清单，也不写谁跟谁是什么关系。\n"
    "> 三档具体怎么说话（称呼 / 直接度 / 话量 / 解释 / 追问 / 情绪）落在 wiki/relationships.md，这里只给事实。\n"
    "\n"
    "- 拥有者：_待填_（唯一不用客气的那个，以及她默认怎么叫他）\n"
    "- 熟人：_待填_（聊得起来、会接话的那类人；聊到什么程度算熟）\n"
    "- 陌生人：_待填_（没打过交道的人；她默认保持的距离）\n"
    "\n"
    "## 说话习惯（QQ 文字）\n"
    "- 一次发几条 / 每条多长：_待填_（一条说完，还是想到哪发到哪）\n"
    "- 口头禅 / 语气词：_待填_\n"
    "- 标点 / 表情 / 贴图 / 语音：_待填_（用不用句号、爱不爱发表情包、会不会发语音）\n"
    "- 怎么称呼人 / 怎么自称：_待填_\n"
    "- 群里还是私聊：_待填_（群里搭不搭话、被 @ 才回还是自己凑热闹）\n"
    "- 打字习惯：_待填_（错别字、缩写、句尾拖长音、用拼音代替字）\n"
    "\n"
    "## 底线雷区\n"
    "- _待填_（绝对不做的三件事）\n"
    "\n"
    "## 价值观与立场\n"
    "> 这一节是 `wiki/values.md` 的唯一权威源，**人工写死**。归纳AI 只做搬运与细化，\n"
    "> 不许从对话里给她发明新立场；对话里出现相反的，标 conflict 交人工裁。\n"
    "\n"
    "- 信念 / 立场：_待填_（她认什么理、不认什么理，一条一句）\n"
    "- 在意的事：_待填_（值得她花力气的东西）\n"
    "- 不在意的事：_待填_（别人在意、她偏不在意的）\n"
    "- 底线 / 执念：_待填_（碰了就翻脸的那一条）\n"
    "\n"
    "## 想要与怕\n"
    "- 今年想达成：_待填_\n"
    "- 怕的东西：_待填_\n"
    "\n"
    "## 不存在的东西\n"
    "- _待填_（这个世界里没有的人 / 事 / 说法）\n"
)


def _is_placeholder(line: str) -> bool:
    return any(tok in line for tok in PLACEHOLDER_TOKENS)


def _meaningful(text: str) -> list[str]:
    """去掉标题、引用、空行、纯分隔线后剩下的实质行。"""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(">"):
            continue
        if set(line) <= set("|-: "):
            continue
        out.append(line)
    return out


@dataclass(slots=True)
class Section:
    name: str
    chars: int
    filled: bool


def _split_sections(text: str) -> list[Section]:
    """按 `## ` 切节，统计每节有多少非占位内容。"""
    out: list[Section] = []
    name = ""
    body: list[str] = []

    def flush() -> None:
        if not name:
            return
        lines = _meaningful("\n".join(body))
        clean = [ln for ln in lines if not _is_placeholder(ln)]
        out.append(
            Section(
                name=name,
                chars=sum(len(ln) for ln in clean),
                filled=bool(clean),
            )
        )

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            name = line[3:].strip()
            body = []
            continue
        body.append(line)
    flush()
    return out


def normalize(text: str) -> str:
    """统一行尾、去 BOM、补尾换行。不碰内容本身。"""
    s = (text or "").replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.rstrip() + "\n" if s.strip() else ""


class Seed:
    """母本的唯一写者。除 save() 外全部只读。"""

    def __init__(self, root: Path | str, history_dir: Path | str) -> None:
        self.root = Path(root)
        self.path = self.root / IDENTITY_REL
        self.history_dir = Path(history_dir)
        self._cache = ""
        self._mtime = -1.0

    # ---------- 读 ----------

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def read(self) -> str:
        """按 mtime 缓存。手改了文件、或别的进程写了，都能立刻看到。"""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._cache, self._mtime = "", -1.0
            return ""
        if mtime != self._mtime:
            try:
                self._cache = self.path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log_.warning("母本读取失败：%s", exc)
                self._cache = ""
            self._mtime = mtime
        return self._cache

    def brief(self, max_chars: int = 800) -> str:
        """给生成提示词用的母本摘要：剔掉占位行与说明性引用，只留写好的部分。

        母本还没写时返回空串——调用方据此决定"能不能造这一档的料"。
        """
        text = self.read()
        if not text:
            return ""
        kept: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(">"):
                continue
            if _is_placeholder(stripped):
                continue
            kept.append(line.rstrip())
        body = "\n".join(kept).strip()
        return body[:max_chars]

    def sections(self) -> list[Section]:
        return _split_sections(self.read() or TEMPLATE)

    def history(self) -> list[str]:
        if not self.history_dir.is_dir():
            return []
        names = [p.name for p in self.history_dir.glob("*.md") if p.is_file()]
        return sorted(names, reverse=True)

    def status(self) -> dict:
        text = self.read()
        secs = _split_sections(text or TEMPLATE)
        try:
            saved = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(self.path.stat().st_mtime)
            )
        except OSError:
            saved = ""
        return {
            "exists": self.exists,
            "path": str(self.path),
            "content": text,
            "chars": len(text),
            "max_chars": MAX_CHARS,
            "saved_at": saved,
            "sections": [
                {"name": s.name, "chars": s.chars, "filled": s.filled} for s in secs
            ],
            "filled": sum(1 for s in secs if s.filled),
            "total": len(secs),
            "history": self.history()[:5],
            "history_count": len(self.history()),
        }

    # ---------- 写 ----------

    def _backup(self, text: str) -> str:
        """把即将被覆盖的旧版存一份。返回备份文件名，失败返回空串。"""
        if not text.strip():
            return ""
        try:
            self.history_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"{stamp}.md"
            target = self.history_dir / name
            # 同一秒内连续保存（比如误点两下）不该互相覆盖
            n = 1
            while target.exists():
                name = f"{stamp}-{n}.md"
                target = self.history_dir / name
                n += 1
            target.write_text(text, encoding="utf-8", newline="\n")
            self._prune()
            return name
        except OSError as exc:
            log_.warning("母本备份失败（不影响本次保存）：%s", exc)
            return ""

    def _prune(self) -> None:
        files = sorted(
            (p for p in self.history_dir.glob("*.md") if p.is_file()),
            key=lambda p: p.name,
            reverse=True,
        )
        for old in files[KEEP_HISTORY:]:
            try:
                old.unlink()
            except OSError:
                pass

    def save(self, text: str) -> dict:
        """落盘。抛 ValueError 表示这次不该写。"""
        body = normalize(text)
        if not body.strip():
            raise ValueError("母本不能是空的。至少写一句「她是谁」")
        if len(body) > MAX_CHARS:
            raise ValueError(f"母本 {len(body)} 字，超过上限 {MAX_CHARS} 字")

        old = self.read()
        backup = "" if body == normalize(old) else self._backup(old)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".md.tmp")
        try:
            tmp.write_text(body, encoding="utf-8", newline="\n")
            os.replace(tmp, self.path)
        except OSError as exc:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise ValueError(f"写入失败：{exc}") from exc

        # 缓存失效，下一次 read 立刻读到新内容
        self._mtime = -1.0
        self._cache = ""
        log_.info("身份母本已保存：%d 字%s", len(body), "（旧版已备份）" if backup else "")
        out = self.status()
        out["backup"] = backup
        return out

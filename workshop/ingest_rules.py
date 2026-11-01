"""归纳规则常量：`agents.md` 里程序要判的那几条，抽成代码。

`agents.md` 是给模型看的散文，模型能读、程序不能。这个文件只放**程序必须自己判**的东西：
页名白名单、成品头字段、禁令正则、置信度上限、改名后缀。

改这里的表必须同时改 `agents.md`（和 `bridge/seed.py` 的 SECTIONS 一个先例）：
两边对不上时，`ingest.lint()` 会报「规程版本不一致」。

本模块零 IO、零依赖（只 import 正则与 library 的页表），可以随便被任何层引用。
"""

from __future__ import annotations

import re

from .library import WIKI_PAGES
from .memwiki import JUDGEMENT

# ---------- 改名后缀 ----------

SUFFIX = "已归纳"
"""收过的素材改名为 `<原名>.已归纳.md`。

插在 `.md` 之前，保证还是 .md：双击能开、编辑器认得出、资源管理器能一眼看出哪些没走。
"""

_SUFFIX_TAIL = f".{SUFFIX}.md"


def is_ingested(name: str) -> bool:
    return name.endswith(_SUFFIX_TAIL)


def ingested_name(name: str) -> str:
    """`a.md` → `a.已归纳.md`。已带后缀的原样返回，不出双后缀。"""
    if is_ingested(name):
        return name
    return name[: -len(".md")] + _SUFFIX_TAIL if name.endswith(".md") else name + _SUFFIX_TAIL


def plain_name(name: str) -> str:
    """去掉后缀，用于水位比对与 log 反查：`a.已归纳.md` → `a.md`。"""
    if is_ingested(name):
        return name[: -len(_SUFFIX_TAIL)] + ".md"
    return name


def plain_stem(name: str) -> str:
    """`a.已归纳.md` → `a`。log 里记的是不带扩展名的素材名。"""
    base = plain_name(name)
    return base[:-3] if base.endswith(".md") else base


# ---------- 素材类型 ----------

KIND_IDENTITY = "identity"
KIND_REACTION = "reaction"
KIND_DIALOG = "dialog"

KIND_LABEL = {
    KIND_IDENTITY: "身份母本",
    KIND_REACTION: "反应样式",
    KIND_DIALOG: "对话稿",
}

# ---------- 成品头 ----------

HEAD_KEYS = (
    # 反应样式稿（agents.md 第九节）
    "形态", "档位", "刺激", "场景", "情绪", "回应", "话量", "句长", "追问",
    "露出", "语言", "不做", "置信", "审", "范例", "例句",
    # 对话稿（第六节）
    "用途", "分类", "情境", "张力", "对方", "构成", "来源", "批注",
    # 程序自己写的溯源行，不参与判断，但要能读出来给人看
    "证据", "生成",
)

HEAD_LIMIT = 40
"""只在前 40 行里找成品头。头在标题下方，正文再长也不会跑到后面去。"""

_HEAD_LINE = re.compile(r"^>\s*([^：:\s]{1,8})\s*[：:]\s*(.*)$")

LEVELS = ("拥有者", "熟人", "陌生人")
LEVEL_NONE = "—"
STIMULI = ("被逼近", "被轻视", "被关心", "被示好", "被道歉", "被逗",
           "被依赖", "被误解", "被索取", "被试探", "被指责", "被冷落")
"""与 `react_style.STIMULI` 必须逐字一致。反应卡那条线从这里读，两边分叉会导致
成品头的值过不了归纳校验。test_distill 有一条断言守这个不变式。"""
CATEGORIES = (
    "日常搭话", "情绪反应", "冲突对峙", "价值判断",
    "回忆往事", "关系试探", "边界触碰", "自我介绍",
)

REL_SECTIONS = {
    "拥有者": "## 与拥有者",
    "熟人": "## 对熟人",
    "陌生人": "## 对陌生人",
}

# 读取侧别名：早期模板把后两档写成"## 与熟人 / ## 与陌生人"，
# 与 REL_SECTIONS 对不上，`_slice_section` 取不到就退化成"整页三档一起给模型"，
# 正是 agents.md 第八节要防的串档。老库不改造，读取侧多试一个名字即可。
REL_SECTION_ALIASES = {
    "拥有者": ("## 与拥有者",),
    "熟人": ("## 对熟人", "## 与熟人"),
    "陌生人": ("## 对陌生人", "## 与陌生人"),
}

AUDIT_PASS = "已过"
AUDIT_FAIL = "未过"

# ---------- 置信度 ----------

CONF_HIGH, CONF_MEDIUM, CONF_LOW = "high", "medium", "low"
LEVEL_OF_CONF = {CONF_HIGH: 3, CONF_MEDIUM: 2, CONF_LOW: 1}


def cap_for(head: dict[str, str]) -> str:
    """这份素材最多能撑到几级置信。

    agents.md 第六节口径五条的可执行版：
    - 有「人工原话 N 句」(N>0) → 可 high
    - 反应样式稿 → 恒 medium 上限（模型抽象，非原话）
    - 小说「原文 N 句」→ medium，永远不是 high
    - 全篇只有 AI 采纳 → medium
    - 批注里有「推定」→ low
    """
    comp = head.get("构成", "")
    note = head.get("批注", "")
    if "推定" in note:
        return CONF_LOW
    # 反应样式稿先判：它自己就是模型抽象出来的，就算头里写了原话也不许升过 medium
    if head.get("形态", "").startswith("反应样式"):
        return CONF_MEDIUM
    m = re.search(r"人工原话\s*(\d+)", comp)
    if m and int(m.group(1)) > 0:
        return CONF_HIGH
    if re.search(r"原文\s*(\d+)", comp):
        return CONF_MEDIUM
    if re.search(r"AI\s*采纳\s*(\d+)", comp):
        return CONF_MEDIUM
    return CONF_MEDIUM


def clamp(conf: str, head: dict[str, str]) -> str:
    """把模型自报的置信压到素材允许的天花板以下。"""
    want = LEVEL_OF_CONF.get(str(conf).strip().lower(), 2)
    cap = LEVEL_OF_CONF[cap_for(head)]
    return {3: CONF_HIGH, 2: CONF_MEDIUM, 1: CONF_LOW}[min(want, cap)]


# ---------- 写权边界 ----------

FORBIDDEN_PAGES = {
    KIND_REACTION: ("identity.md", "boundaries.md"),
}
"""反应样式稿的两条禁令（agents.md 第九节）：不许进身份 / 边界。"""

NO_GROW = ("identity.md", "traits.md", "speech.md", "values.md", "boundaries.md")
"""记忆层归档器不许写这几页；要改只能写 memory/pending 提案（第十一节写权边界）。"""

RELATION_KNOBS = ("称呼", "直接度", "话量", "解释不解释", "主动追问", "情绪露出")
"""relationships.md 只允许动这六个旋钮，别的都是基调漂移。"""

OPS = ("append", "merge", "conflict", "skip")
"""补丁只有这四种动作。**没有 delete、没有 replace** —— 不是提示词约定，是 schema 里不存在。"""


def allowed_pages(kind: str) -> tuple[str, ...]:
    banned = FORBIDDEN_PAGES.get(kind, ())
    return tuple(p for p in WIKI_PAGES if p not in banned)


# ---------- 禁令正则 ----------

NAME_MIN_LEN = 2
"""单字不给当专名拦：'他'、'你' 会命中每一句（memwiki 同款口径）。"""

# 称谓 / 职业 / 书名号：反应卡的第三条禁令（`agents.md` 第九节）
#
# 这一层拦的是"模型替素材补出来的具体身份"。反应卡只有档位没有对象，
# 写出「喊他老周」就是编的 —— 素材里根本没有这个人。
#
# 只做**高精度低召回**：宁漏不误杀，因为误杀会让真素材收不进来。
# 地名故意不进正则：`城市 / 市场 / 小镇` 和"北京市"没有可分的形状，
# 硬写一条只会把正常句子拦掉。地名交给 quote 回查兜底 ——
# 编出来的地名不可能在素材里找到原句，真出现的地名本来就是素材自己写的。
TITLE_WORDS = (
    "老周", "母亲大人", "大人", "陛下", "殿下", "阁下", "师尊", "师父", "师傅",
    "老爷", "夫人", "小姐", "少爷", "公子", "姑娘", "公主", "王子", "国王",
    "皇帝", "皇后", "将军", "掌柜", "管家", "侍女", "仆人", "骑士",
)
OCCUPATION_WORDS = (
    "医生", "律师", "校长", "总裁", "董事长", "经理", "老板", "警察",
    "士兵", "祭司", "神官", "学者", "船长", "队长", "教习",
)

_TITLE_PAT = re.compile("|".join(re.escape(w) for w in sorted(TITLE_WORDS, key=len, reverse=True)))
_OCCUPATION_PAT = re.compile("|".join(re.escape(w) for w in OCCUPATION_WORDS))
_BOOK_PAT = re.compile(r"《[^》]{1,24}》")


def proper_hits(text: str) -> list[str]:
    """专名 / 称谓 / 职业 / 书名号，命中即拒收。去重保序。"""
    out: list[str] = []
    for m in _TITLE_PAT.finditer(text):
        out.append(m.group(0))
    for m in _OCCUPATION_PAT.finditer(text):
        out.append(m.group(0))
    for m in _BOOK_PAT.finditer(text):
        out.append(m.group(0))
    seen: set[str] = set()
    return [w for w in out if not (w in seen or seen.add(w))]


def judgement_hits(text: str) -> list[str]:
    """性格断言（她总是 / 一向 / 本质…）。命中即拒收，`agents.md` 第十一节句法禁令。"""
    return [m.group(0) for m in JUDGEMENT.finditer(text)]


# ---------- 页内结构 ----------

NEW_SECTION_PAGES = ("exemplars.md",)
"""只给 exemplars 开这个口子：`## 行为范例` 是人定的新节，模板里还没有。

reactions.md **不给**：真库里它是「`## 高频情境反应` + `### 情境：xxx`」两层结构，
让模型自己开 `##` 节，会写出一条"刺激—反应（反应样式稿，medium 上限）"这种
只有模型看得懂的平行结构，把整页拆成两套写法。反应卡一律进 `## 高频情境反应`，
新情境用 `### 情境：…` 作 title。

其余页（traits / speech / values / relationships / boundaries / identity）
的节是人工定好的骨架，模型不许自造 —— 造了就是拿一格虚构结构替换整页写法。
"""


def sections_of(text: str, level: int = 2) -> list[str]:
    """页里所有 `## 标题` / `### 标题`（不含前缀）。按出现顺序。"""
    marks = "#" * level + " "
    out: list[str] = []
    for line in text.splitlines():
        s = line.rstrip()
        if s.startswith(marks) and not s.startswith("#" * (level + 1) + " "):
            out.append(s[len(marks):].strip())
    return out


def header_line(section: str, level: int = 2) -> str:
    """`高频情境反应` / `## 高频情境反应` → 统一成 `## 高频情境反应`。"""
    name = re.sub(r"^#+\s*", "", str(section).strip()).strip()
    return ("#" * level + " " + name) if name else ""


def dirty_head(value: str) -> bool:
    """头字段里的空值写法：`—`、`-`、`无`、空串都算没填。"""
    return str(value).strip() in ("", "—", "-", "无", "N/A", "n/a")


# ---------- 成品头解析 ----------


def parse_head(text: str) -> dict[str, str]:
    """把 `> 键：值` 那一段读成字典。

    只认 HEAD_KEYS 里的键 —— 素材正文里也有引用块，不加白名单会把正文当头部读。
    同一键出现两次以第一次为准（`> 生成：` 这类程序行可能重复，不影响判断）。
    """
    head: dict[str, str] = {}
    for line in text.splitlines()[:HEAD_LIMIT]:
        m = _HEAD_LINE.match(line.strip())
        if not m:
            continue
        key, value = m.group(1).strip(), m.group(2).strip()
        if key in HEAD_KEYS and key not in head:
            head[key] = value
    return head


def kind_of(name: str, head: dict[str, str]) -> str:
    """素材类型。判定顺序即优先级：母本 > 反应样式 > 对话稿。"""
    if name == "identity.md":
        return KIND_IDENTITY
    if head.get("形态", "").startswith("反应样式"):
        return KIND_REACTION
    return KIND_DIALOG


def level_of(head: dict[str, str]) -> str:
    """关系档。反应卡看 `档位：`，对话稿看 `情境：<档位>｜…` 的首段。

    返回 "—" 表示没分档（不归关系页），不是"缺字段"。
    """
    got = head.get("档位", "").strip()
    if got in LEVELS:
        return got
    scene = head.get("情境", "")
    first = re.split(r"[｜|]", scene)[0].strip() if scene else ""
    return first if first in LEVELS else LEVEL_NONE

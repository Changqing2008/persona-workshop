"""角色实体发现：不依赖「XX 说」的四个信号。

为什么不能靠"从说话动词前面捞名字"：这本书 99% 的台词段里根本没有"某某说"，
实测捞出来的 top25 全是 `换句话`×11、`老实`×9、`现在`×8、`殿下`×3 —— 一个真名都没有。
所以换四个在这类译本上真的能命中的信号：

1. **间隔号全名** `尤莉安娜·玛赞塔` —— 最可靠，一眼就是人名，几乎不会误判；
2. **前缀绑定** `尤菲` ⊂ `尤莉安娜·玛赞塔` —— 只认前缀，后缀是切分碎片
   （`菲亚` 属于 `艾妮丝菲雅`，按子串匹配就会把两个人的名字缝在一起）；
3. **敬称后缀** `「艾妮丝」小姐`、`「格兰特」公爵` —— 简称的第二个来源；
4. **称呼语** 引语开头的 `「伊莉丝，……」` —— 没全名也没敬称的角色靠这个兜住。

产出是**待人确认的清单**，不是自动结论：

- A 档：有全名锚，默认勾选；
- B 档：只靠敬称/称呼语推定，默认勾选但标出来看；
- C 档：`殿下`、`公爵` 这类泛称，两位公主都被叫"殿下"，必须人来选，默认留空。

合并错了等于把两个人的台词混成一个人，代价比漏合高得多，所以一律留人工闸门。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from . import text as T

#: 间隔号全名：尤莉安娜·玛赞塔 / 亚尔加德·伯纳·帕雷提亚
FULLNAME_RE = re.compile(r"[\u4e00-\u9fa5]{2,6}·[\u4e00-\u9fa5]{1,6}(?:·[\u4e00-\u9fa5]{1,6}){0,2}")

#: 敬称：只在人后面接这些，说明前面那截是称呼
HONOR_SUFFIX = ("小姐", "公爵", "殿下", "大人", "先生", "公主", "王女", "陛下", "同学", "老师", "夫人")
HONOR_RE = re.compile(r"(?=%s)" % "|".join(HONOR_SUFFIX))

#: 泛称：听着像人，但一屋里可能有好几个，不能自动归并
GENERIC = {"殿下", "大人", "阁下", "陛下", "夫人", "公主", "公主殿下", "公爵", "小姐",
           "少爷", "那家伙", "这家伙", "父王陛下", "国王陛下"}

#: 出现在名字里就不可能是名字的字（虚词、代词、连词）
_BAD_CHARS = "的了着是在和跟与对向朝把被从为就也都还又才便很更最太当没不我你他她它们这那所以如果但因为虽然可是不过而且然后于是只有还要让给甚连再怎"
#: 名字左边常粘着的字，剥掉才是名字
_LEFT_JUNK = "的了让是和跟与对向朝把被在从为就也都还又才便很更最太当看见听想要说问叫是没不"
#: 名字右边常粘着的字
_RIGHT_JUNK = "的了着地吗呢吧啊呀哦喔和跟与对向朝把被"

#: 高频非人名词：引语开头的"知道，""明白，"这类会被称呼语信号误收
_STOP_WORDS = {
    "知道", "明白", "时候", "其实", "身上", "这样", "那样", "什么", "怎么", "为什",
    "现在", "已经", "可能", "应该", "觉得", "认为", "一直", "立刻", "忽然", "突然",
    "终于", "只是", "不过", "可以", "没有", "不会", "不能", "不要", "还是", "就是",
    "也是", "都是", "于是", "然后", "接着", "最后", "同时", "当时", "此时", "这时",
    "那时", "当然", "果然", "居然", "竟然", "难道", "似乎", "仿佛", "好像", "大概",
    "也许", "或者", "而且", "并且", "虽然", "但是", "可是", "因为", "所以", "如果",
    "只要", "除了", "关于", "对于", "由于", "对吧", "是吧", "是啊", "没错", "真的",
    "抱歉", "谢谢", "对不起", "没问题", "好的", "不行", "等等", "那里", "这里",
}

#: 一个字至少出现这么多次，才值得当候选
MIN_COUNT = 5
#: 同段共现达到这个数，才作为"应该是同一个人"的旁证
COOCCUR_MIN = 10


@dataclass(slots=True)
class Entity:
    canonical: str
    """标准名：优先取带间隔号的全名，没有就用最高频写法。"""
    label: str
    """显示名：出现最多的那个写法，界面上用它。"""
    count: int
    tier: str
    """A=有全名锚，B=敬称/称呼语推定，C=泛称待选。"""
    aliases: dict[str, int] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def all_forms(self) -> list[str]:
        out = [self.canonical, self.label]
        out += [a for a in self.aliases if a not in out]
        return [x for x in out if x]

    def to_dict(self) -> dict:
        return {
            "name": self.canonical,
            "label": self.label,
            "count": self.count,
            "tier": self.tier,
            "aliases": [{"name": k, "count": v} for k, v in
                        sorted(self.aliases.items(), key=lambda kv: -kv[1])],
            "signals": list(self.signals),
            "conflicts": list(self.conflicts),
        }


@dataclass(slots=True)
class Report:
    entities: list[Entity] = field(default_factory=list)
    pending: list[dict] = field(default_factory=list)

    def find(self, name: str) -> Entity | None:
        name = (name or "").strip()
        if not name:
            return None
        for e in self.entities:
            if name in e.all_forms():
                return e
        # 用户填的可能是全名的一个片段
        best: Entity | None = None
        for e in self.entities:
            if any(name in f or f in name for f in e.all_forms()):
                if best is None or e.count > best.count:
                    best = e
        return best

    def resolve(self, name: str) -> tuple[str, list[str]]:
        """用户填的名字 → (标准名, 别名)。找不到就把原名当标准名。"""
        e = self.find(name)
        if not e:
            return (name or "").strip(), []
        return e.canonical, [f for f in e.all_forms() if f != e.canonical]

    def cast(self, character: str, extra: list[str] | None = None) -> dict:
        """构造 extract 需要的几份名单。"""
        standard, auto = self.resolve(character)
        names = list(dict.fromkeys(
            [n for n in [standard] + list(extra or []) + auto if n]
        ))

        others: list[str] = []
        for e in self.entities:
            forms = e.all_forms()
            if any(f in names for f in forms):
                continue
            others += forms
        others = list(dict.fromkeys([n for n in others if n and n not in names]))

        return {
            "standard": standard,
            "aliases": [n for n in names if n != standard],
            "names": names,
            "others": others,
            "roster": [e.label for e in self.entities][:40],
            "report": self,
        }

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "pending": list(self.pending),
        }


# ---------------------------------------------------------------- 内部

def _trim(token: str) -> str:
    s = token.strip()
    s = re.sub(r"^[%s]+" % _LEFT_JUNK, "", s)
    s = re.sub(r"[%s]+$" % _RIGHT_JUNK, "", s)
    return s


def _looks_like_name(s: str, text: str, min_count: int = MIN_COUNT) -> bool:
    """一个串像不像人名。宁可拒绝，也不要把「可以请」这种人名收进来。"""
    if not re.fullmatch(r"[\u4e00-\u9fa5]{2,6}", s):
        return False
    if s in GENERIC or s in _STOP_WORDS or s in T._STOP_NAMES:
        return False
    if any(ch in _BAD_CHARS for ch in s):
        return False
    if any(ch in T.PRONOUNS for ch in s):
        return False
    return text.count(s) >= min_count


def _tail(p: str, maxlen: int = 4) -> str:
    """取尾部最长的合法片段：「甚至连萨兰」→「萨兰」。"""
    for n in range(min(maxlen, len(p)), 1, -1):
        cand = p[-n:]
        if not any(ch in _BAD_CHARS for ch in cand):
            return cand
    return ""


def _head(p: str, maxlen: int = 4) -> str:
    """取头部最长的合法片段：「玛赞塔小姐」→「玛赞塔」。"""
    for n in range(min(maxlen, len(p)), 1, -1):
        cand = p[:n]
        if not any(ch in _BAD_CHARS for ch in cand):
            return cand
    return ""


def _clean_fullname(raw: str) -> str:
    """修剪间隔号全名两端的粘连。

    正则的第二段是 {2,6}，会把「玛赞塔只能在」「千金尤莉安娜」这类整串吃进去。
    第一段往前粘连（「甚至连萨兰」），末尾段往后粘连（「玛赞塔小姐」），各取合法的一端。
    """
    parts = [_trim(p) for p in raw.split("·")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return ""
    # 「奈雪儿·玛赞塔公爵夫人」：末尾段先把敬称剥干净，否则会剩下「玛赞塔公」
    last_raw = parts[-1]
    for h in HONOR_SUFFIX:
        idx = last_raw.find(h)
        if idx >= 2:
            last_raw = last_raw[:idx]
            break
    changed = True
    while changed:
        changed = False
        for h in HONOR_SUFFIX:
            if last_raw.endswith(h):
                last_raw = last_raw[: -len(h)]
                changed = True
    first = _tail(parts[0], 5)
    last = _head(last_raw, 4)
    if not first or not last:
        return ""
    middle = [p for p in parts[1:-1] if 1 <= len(p) <= 4]
    return "·".join([first] + middle + [last])


def _honor_candidates(full: str) -> Counter[str]:
    """从「XX 小姐 / XX 公爵」里取名字。

    不能拿正则的贪婪匹配直接用——它会把「所以可以请殿下」整串都吃进去。
    改成从敬称位置往前取 2~5 字，挑最长的那个"像名字"的：
    「……可以请殿下」→ 试「可以请」含虚词，否；试「以请」否；试「请」太短 → 全部拒绝。
    """
    out: Counter[str] = Counter()
    for m in HONOR_RE.finditer(full):
        end = m.start()
        window = full[max(0, end - 6):end]
        for n in (5, 4, 3, 2):
            if len(window) < n:
                continue
            cand = _trim(window[-n:])
            if _looks_like_name(cand, full):
                out[cand] += 1
                break
    return out


def _vocatives(full: str, narr: str) -> Counter[str]:
    """引语开头的称呼语：「伊莉丝，我想用魔法。」

    没有全名、也没有敬称的角色（侍女、骑士、龙）靠这个兜住。
    多一道闸：这个名字必须在**旁白**里也出现过——「知道，」「明白，」只会出现在台词里，
    真人在旁白里总会被提到（"伊莉丝示意要尤菲坐下"）。
    """
    out: Counter[str] = Counter()
    for q in T.QUOTE_RE.findall(full):
        m = re.match(r"^\s*([\u4e00-\u9fa5]{2,4})[，、]", q)
        if not m:
            continue
        cand = m.group(1)
        if _looks_like_name(cand, full, min_count=3) and narr.count(cand) >= 3:
            out[cand] += 1
    return out


def _bind_prefixes(e: Entity, full: str) -> None:
    """把全名的高频前缀当别名，跳掉"每次出现都被更长前缀覆盖"的碎片。

    尤莉安娜·玛赞塔 出现 2 次：
      尤莉安娜 163 次  ← 比全名多，说明有独立用法，收
      尤菲莉     163 次  ← 和尤莉安娜一样多，说明只是它的切分，丢
      尤菲       619 次  ← 独立用法，收
    """
    canon = e.canonical
    prev = None
    for n in range(len(canon), 1, -1):
        p = canon[:n]
        if "·" in p:
            prev = full.count(p)
            continue
        c = full.count(p)
        if prev is not None and c == prev:
            prev = c
            continue
        if c >= MIN_COUNT and p != canon:
            e.aliases[p] = c
            e.signals.append(f"前缀×{c}")
        prev = c


def discover(chapters: list[T.Chapter]) -> Report:
    """扫全书，产出实体清单。chapters 用 text.split_chapters 的结果。"""
    paras = [p for c in chapters for p in c.paras]
    full = "\n".join(paras)

    # ---------- 信号 1：间隔号全名 ----------
    anchors: Counter[str] = Counter()
    for m in FULLNAME_RE.finditer(full):
        name = _clean_fullname(m.group(0))
        if name:
            anchors[name] += 1

    entities: list[Entity] = []
    for n, c in anchors.items():
        # 首段必须能在别处独立出现：「蕾妮」17 次是真名，
        # 「挚友格兰特」只在那个全名里出现过一次，是左侧粘连的叙述，不是名字。
        if full.count(n.split("·")[0]) < 2:
            continue
        entities.append(Entity(canonical=n, label=n, count=c, tier="A", signals=["间隔号全名"]))

    # ---------- 信号 2：前缀绑定 ----------
    for e in entities:
        _bind_prefixes(e, full)

    # ---------- 信号 3、4：敬称与称呼语 ----------
    narr = T.QUOTE_RE.sub(" ", full)
    for src, tag in ((_honor_candidates(full), "敬称"), (_vocatives(full, narr), "称呼语")):
        for short, cnt in src.items():
            owner = None
            for e in entities:
                if e.canonical.startswith(short) or short in e.aliases:
                    owner = e
                    break
            if owner:
                owner.aliases.setdefault(short, full.count(short))
                if f"{tag}×{cnt}" not in owner.signals:
                    owner.signals.append(f"{tag}×{cnt}")
                continue
            if any(e.canonical == short for e in entities):
                continue
            entities.append(Entity(
                canonical=short, label=short, count=full.count(short),
                tier="B", signals=[f"{tag}×{cnt}"],
            ))

    # ---------- 信号 5：口语体「某某说」 ----------
    # 原创中文小说常常满篇"老周说"，那种书从说话动词前面捞名字是有效的。
    # 但本书只有 1% 的台词段这么写，直接捞出来全是「换句话」「老实」。
    # 所以先量一下体例，够密才启用这条——宁可少一个信号，也不要把垃圾名塞进名单。
    quoted = [p for p in paras if T.QUOTE_RE.search(p)]
    cue_ratio = (
        sum(1 for p in quoted if T.VERB_RE.search(T.QUOTE_RE.sub("", p))) / len(quoted)
        if quoted else 0.0
    )
    if cue_ratio >= 0.15:
        # 这里只按"说话动词前出现过"取候选，频次交给 _looks_like_name 用全文计数把关：
        # 短篇里「老周说」可能只出现一次，但「老周」全文出现十几次，那仍然是真名。
        for name, cnt in T.discover_names(paras, 200, min_count=1):
            if not _looks_like_name(name, full, min_count=3):
                continue
            if any(e.canonical == name or name in e.all_forms() for e in entities):
                continue
            entities.append(Entity(
                canonical=name, label=name, count=full.count(name),
                tier="B", signals=[f"说话动词×{cnt}"],
            ))

    # ---------- 同段共现：只作旁证，不单独作依据 ----------
    for e in entities:
        if e.tier != "A":
            continue
        for other in entities:
            if other is e or other.tier != "B":
                continue
            short = other.canonical
            co = sum(1 for p in paras if short in p and e.label in p)
            if co >= COOCCUR_MIN and short not in e.aliases:
                e.aliases[short] = other.count
                e.signals.append(f"共现×{co}")

    absorbed = {a for e in entities if e.tier == "A" for a in e.aliases}
    entities = [e for e in entities if not (e.tier == "B" and e.canonical in absorbed)]

    # ---------- 反证：同块内一 a 一 b 接续对答 ----------
    forms = [f for e in entities for f in e.all_forms()]
    for ch in chapters:
        cmap = T.cue_map(ch.paras, forms)
        slots = sorted(cmap.items())
        for (i, a), (j, b) in zip(slots, slots[1:]):
            if j - i > 3 or a == b:
                continue
            pair = {a, b}
            for e in entities:
                if e.tier != "A" or e.canonical not in pair:
                    continue
                rest = pair - {e.canonical}
                if rest & set(e.aliases):
                    other = rest.pop()
                    msg = f"第 {i}/{j} 段由「{e.canonical}」与「{other}」分别接话，疑似两人"
                    if msg not in e.conflicts:
                        e.conflicts.append(msg)

    # ---------- 定稿：计数与显示名 ----------
    for e in entities:
        forms_e = [e.canonical] + list(e.aliases)
        counts = {f: full.count(f) for f in forms_e if full.count(f)}
        if counts:
            e.count = max(counts.values())
            e.label = max(counts, key=lambda k: counts[k])

    # ---------- C 档：泛称待选 ----------
    top = [e for e in entities if e.tier in ("A", "B")][:6]
    pending = []
    for g in ("殿下", "大人", "陛下", "公爵", "小姐", "夫人", "阁下"):
        cnt = full.count(g)
        if cnt < 5:
            continue
        pending.append({
            "name": g,
            "count": cnt,
            "tier": "C",
            "options": [e.label for e in top][:4],
            "reason": "两人以上共用这个称呼，自动归并会混人",
        })

    # B 档只有称呼语/敬称一条证据，频次又低的多半是「笑容」「一样」这类词。
    # 自动名单只留高频的；但"说话动词"信号本身就是强证据，不受频次门槛约束。
    entities = [
        e for e in entities
        if e.tier == "A" or e.count >= 60
        or any(s.startswith("说话动词") for s in e.signals)
    ]
    entities.sort(key=lambda e: -e.count)
    return Report(entities=entities[:40], pending=pending)

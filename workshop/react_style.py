"""反应卡内核：把一段对白抽象成"反应样式"。

为什么要有这一层：直接把小说对白当素材喂进人格库，人名、关系、剧情会一起进去，
最后她会"记得"自己从没经历过的事。这里只留一件事——
**面对这类刺激，她用什么方式、什么情绪、什么语言回。**

三道闸，缺一道都会漏：
1. 入口遮罩：送模型之前就把人名换成「她 / 某人」，关系档与章节名根本不传。
   模型看不见，就编不出来。
2. 结构化输出：字段里没有 name / relation / quote 的位置，想写也没处放。
3. 出口校验：专名、关系称呼、与原文 8 字重合，命中就拒。

三道都不是提示词，而是代码。提示词只是让模型少犯错，代码保证错了也出不去。

2026-10-27：本模块从 distill/react.py 上移。
理由：素材台也要出反应卡，而 distill 已经 import sim，
sim 再回头 import distill.react 就成了三向循环。内核放这里是唯一的单向解。
distill/react.py 保留为 re-export 壳，不删。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

# 刺激类型固定。自由发挥会让三十张卡写出三十种说法，既没法统计也没法去重。
#
# 十类而不是八类（2026-10-31）：原八类装不下 wiki 里两类高频情境 ——
# 「被示好」（表白 / 夸奖 / 被感谢 / 当面承诺）与「被道歉」（对方认错、递台阶）。
# 这两类在示例角色库里有 high 级原话证据，却只能被硬塞进"被关心"或"被索取"，
# 分类统计与去重都跟着失真。其余边界见 COVERAGE。
#
# 再加两类（同日）：聊天里对方的态度不止"施压"和"索取"，还有两种常见走向
# 原八类接不住 —— 「被逗」（开玩笑、调侃，没有目的）与「被依赖」（示弱、求助、
# 撒娇，把软的一面交过来）。后者正是"她不太会安慰人"这个特征唯一能被看到的场合。
STIMULI = (
    "被逼近", "被轻视", "被关心", "被示好",
    "被道歉", "被逗", "被依赖", "被误解",
    "被索取", "被试探", "被指责", "被冷落",
)

STIM_LIST = " / ".join(STIMULI)

# 情境 → 归类。写死这张表是为了让「覆盖完全」可检验：
# 模型和人都按同一张表落类，不在表里的情境一律不进反应卡（进 exemplars 的行为范例）。
# 左边是 wiki 里会出现的说法，右边必须是 STIMULI 之一，空串 = 明确不归反应卡。
COVERAGE = {
    "被逼表态 / 被问站哪边": "被逼近",
    "被催 / 被追问没回": "被逼近",
    "被突然点名 / 被提高音量": "被逼近",
    "被堵在当场 / 不给退路": "被逼近",
    "被贬低 / 被看不起": "被轻视",
    "示好被客套挡回": "被轻视",
    "对方答应的事不做": "被轻视",
    "被关心 / 被问状态": "被关心",
    "被安慰 / 被照顾": "被关心",
    "被示好 / 被表白": "被示好",
    "被夸奖 / 被感谢": "被示好",
    "被分享好消息": "被示好",
    "被道歉 / 对方递台阶": "被道歉",
    "被调侃 / 被开玩笑": "被逗",
    "被逗笑 / 被撩": "被逗",
    "被示弱 / 对方难过": "被依赖",
    "被求助 / 要她安慰": "被依赖",
    "被撒娇 / 被卖萌": "被依赖",
    "被需要 / 对方离不开她": "被依赖",
    "被误解 / 被曲解": "被误解",
    "被冤枉 / 被当成另一个人": "被误解",
    "被求评价 / 被要一句判断": "被索取",
    "被交代任务 / 被支使": "被索取",
    "被索取时间 / 被索取情绪": "被索取",
    "被试探 / 被绕弯评价": "被试探",
    "被抓包 / 被撞见": "被试探",
    "被翻旧账": "被指责",
    "被指出做错 / 被追责": "被指责",
    "被说教 / 被纠正做法": "被指责",
    "被推锅 / 边界被碰": "被指责",
    "冷场 / 没人理": "被冷落",
    "被晾着 / 消息不回": "被冷落",
    # 空串：她主动做的，不是"被"刺激，结构上不属于反应卡
    "她主动安慰人": "",
    "她主动催对方": "",
    "她主动开话题": "",
    "开心 / 兴奋": "",
}

# 关系三档。这里是"对多远的人"，不是"对谁"——具体是谁永远不写进卡里。
LEVELS = ("拥有者", "熟人", "陌生人")

# 关系称呼：出现在输出里就说明它把"对谁"写进来了，必须拒
RELATION_WORDS = (
    "父亲", "母亲", "爸爸", "妈妈", "爹", "娘", "师父", "师傅", "师兄", "师姐",
    "师弟", "师妹", "上司", "老板", "领导", "同事", "同学", "朋友", "恋人",
    "男友", "女友", "丈夫", "妻子", "老公", "老婆", "儿子", "女儿", "哥哥",
    "姐姐", "弟弟", "妹妹", "叔叔", "阿姨", "舅舅", "老师", "学生", "主人",
    "大人", "殿下", "陛下", "将军", "夫人", "少爷", "小姐", "前辈", "晚辈",
)

SYSTEM = """你在把一段对白抽象成"面对某类情绪刺激时的回应方式"。

你只写方式，不写内容：
- 不写任何人名、称呼、亲属关系、职业、地名、具体事件
- 不引用原句，不复述对方说了什么
- 不写她是谁、经历过什么、身边有谁

回应方式必须可检验：写"先反问，再给结论"，不写"温柔一点""比较冷淡"这类形容词。
写不出可检验的就留空字符串。

刺激类型只能从这十类里选一个：""" + STIM_LIST + """

只输出 JSON 对象，字段固定，不要解释、不要代码块：
{"stimulus":"","scene":"","emotion_from":"","emotion_to":"","moves":[""],
 "volume":"","length":"","probe":"","emotion_shown":"","language":[""],"avoid":[""]}"""

USER = """下面这段对白已经去过名字，你看到的只有「她」和「某人」。

场景类型：{category}
台词：
{lines}

按字段要求输出。scene 不超过 20 字，只描述"被什么逼到"，不要出现任何专名。"""

# 素材台用这套：没有台词，只有设定 + 条件，要她"会怎么回"。
# 和上面 USER 的分工：上面是"给你一段对白，抽象它"；这里是"给你条件，推她怎么回"。
# 后者没有原句可抄，所以更依赖母本——母本里「说话习惯」那一节就是话量/句长/语言的来源。
SYSTEM_SIMULATE = """你在推断一个人遇到某类情绪刺激时会怎么回应，产出一张"反应样式"卡。

你看得到的是她的设定，和这一次逼到她面前的条件。你写**方式**，不写内容：
- 不写任何人名、称呼、亲属关系、职业、地名、具体事件
- 不写台词，不照抄条件里的措辞，不复述对方说了什么
- 设定里没写的，留空字符串——不要替她补一套人生

回应方式必须可检验：写"先反问，再给结论"，不写"温柔一点""比较冷淡"。
话量 / 句长 / 语言三项只能从她的说话习惯推，推不出来就留空。

刺激类型只能从这十类里选一个：""" + STIM_LIST + """

只输出 JSON 对象，字段固定，不要解释、不要代码块：
{"stimulus":"","scene":"","emotion_from":"","emotion_to":"","moves":[""],
 "volume":"","length":"","probe":"","emotion_shown":"","language":[""],"avoid":[""]}"""

NO_PROFILE = """## 她是谁
（还没有设定。只写普遍成立的回应方式，不要编造身份、年龄、职业、身边有谁。）"""

SIMULATE_USER = """{who}

## 这一次的条件
关系档：{relation}
场合：{setting}
对方想要：{goal}
卡在哪：{tension}
对方开口第一句：{opening}

按字段要求输出。scene 不超过 20 字，只写"被什么逼到"，
不要出现任何专名，也不要照抄上面任何一行的措辞。"""


def build_simulate_user(
    profile: str,
    relation: str,
    setting: str,
    goal: str,
    tension: str,
    opening: str,
) -> str:
    """把母本与卡片条件拼成生成请求。空字段写占位符，不留空行给模型猜。"""
    body = (profile or "").strip()
    who = "## 她是谁\n" + body if body else NO_PROFILE
    return SIMULATE_USER.format(
        who=who,
        relation=relation or "（未标）",
        setting=setting or "（未写）",
        goal=goal or "（未写）",
        tension=tension or "（未写）",
        opening=opening or "（未写）",
    )

# 素材台现在的主路：不给情境卡，只给母本 + 刺激/档位，一次要几张。
# 为什么一次要几张而不是一张：一张一调要三次往返，且模型容易连着写同一个套路。
SYSTEM_BATCH = """你在推断一个人遇到某类情绪刺激时会怎么回应，一次产出 {n} 张"反应样式"卡。

你看得到的只有她的设定，和这次要补的两项。你写**方式**，不写内容：
- 不写任何人名、称呼、亲属关系、职业、地名、机构名、具体事件
- 不写台词，不写引号里的内容，不写心理活动
- 设定里没写的，留空字符串——不要替她补一套人生

回应方式必须可检验：写"先反问，再给结论"，不写"温柔一点""比较冷淡"。
话量 / 句长 / 语言三项只能从她的说话习惯推，推不出来就留空。

{n} 张之间必须真的不同：刺激不重复，场景不重复，回应的路子也不能是同一个。

字段固定：
- stimulus：刺激类型，只能从 """ + STIM_LIST.replace(" / ", "/") + """ 里选一个
- level：关系档，只能从 拥有者/熟人/陌生人 里选一个
- scene：什么把她逼到这一步，一句话，不超过 20 字
- emotion_from / emotion_to：从什么情绪被推到哪里，各不超过 8 字
- moves：她先做什么后做什么，2~4 个短动作
- volume / length / probe / emotion_shown：话量 / 句长 / 答完追不追问 / 露出什么情绪
- language：语言层面的习惯，1~4 条
- avoid：这类情况下她不会做的，1~3 条

只输出 JSON 数组，{n} 个对象，不要解释、不要代码块：
[{{"stimulus":"","level":"","scene":"","emotion_from":"","emotion_to":"","moves":[""],
  "volume":"","length":"","probe":"","emotion_shown":"","language":[""],"avoid":[""]}}]"""

BATCH_USER = """{who}

## 这一次的要求
{lines}

产出 {n} 张。scene 只写"被什么逼到"，不要出现任何专名，也不要照抄上面任何一行的措辞。"""


def build_batch_user(profile: str, stimulus: str = "", level: str = "", n: int = 3) -> str:
    """只给母本和两个下拉的选择。

    刻意没有场合 / 对方 / 卡点 / 开场白——这四样是最容易顺着头行进 wiki 的污染源，
    不传进来，模型就编不出来。
    """
    body = (profile or "").strip()
    who = "## 她是谁\n" + body if body else NO_PROFILE
    lines = [
        f"刺激类型：{stimulus}" if stimulus else f"刺激类型：不限，你自己挑，{n} 张不许重复",
        f"关系档：{level}"
        if level
        else "关系档：不限，按她的关系三档各挑一档，别几张都写成同一档",
    ]
    return BATCH_USER.format(who=who, lines="\n".join(lines), n=n)


FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_WS = re.compile(r"\s+")

# render 里 "不做 / 用途" 之间插范例行的位置锚点，见 render_card
_JOIN = " / "


def mask(seg: dict, names: list[str], her: str) -> tuple[str, str]:
    """遮罩台词：人名换掉，其余一个字不改。

    返回 (给模型看的台词文本, 原文文本)。原文文本只用于后面的重合校验，不送给模型。
    只保留 role / action / text；speaker、other_role、章节名一概不带。
    """
    aliases = sorted({n for n in names if n and len(n) >= 2}, key=len, reverse=True)

    def scrub(text: str) -> str:
        out = text or ""
        for name in aliases:
            if name in out:
                out = out.replace(name, "她" if name == her else "某人")
        return out

    lines, raw = [], []
    for line in seg.get("lines") or []:
        text = scrub(line.get("text") or "")
        action = scrub(line.get("action") or "")
        who = "她" if line.get("role") == "self" else "某人"
        lines.append(f"{who}：{'（' + action + '）' if action else ''}{text}")
        raw.append(line.get("text") or "")
        raw.append(line.get("action") or "")
    return "\n".join(lines), "\n".join(raw)


def parse(text: str) -> dict[str, Any]:
    """从模型输出里抠出卡片对象。失败返回空 dict，不抛。"""
    s = (text or "").strip()
    if not s:
        return {}
    for cand in (s, (FENCE.search(s).group(1).strip() if FENCE.search(s) else "")):
        if not cand:
            continue
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return normalize(data)
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            data = json.loads(s[i : j + 1])
            if isinstance(data, dict):
                return normalize(data)
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _scan_objects(text: str) -> list[str]:
    """扫出所有顶层 {...}。用于抢救被截断的输出。

    从 sim/cards.py 上移：两处都要"模型吐了一串对象，切在最后一个字"这种场面。
    逐字符走，认得字符串与转义，所以引号里的花括号不会打乱层级。
    只取已经闭合的对象——没闭合的那个就是被切掉的，救不回来也不硬凑。
    """
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1
    return out


def parse_array(text: str) -> list:
    """从模型输出里抠出一个 JSON 数组。失败返回空列表，不抛异常。

    顺序：整段 → 代码块 → 首尾方括号 → 逐个对象抢救。
    最后一层是为截断准备的：三张卡缺一张，也好过一张都没有。
    """
    s = (text or "").strip()
    if not s:
        return []

    candidates: list[str] = [s]
    m = FENCE.search(s)
    if m:
        candidates.append(m.group(1).strip())
    i, j = s.find("["), s.rfind("]")
    if 0 <= i < j:
        candidates.append(s[i : j + 1])

    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("cards", "items", "result", "data"):
                if isinstance(data.get(key), list):
                    return data[key]

    salvaged: list = []
    for chunk in _scan_objects(s):
        try:
            obj = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            salvaged.append(obj)
    return salvaged


def parse_many(text: str) -> list[dict[str, Any]]:
    """一次请求里的多张卡。空壳（既没场景也没回应）直接丢，不占位。"""
    out: list[dict[str, Any]] = []
    for obj in parse_array(text):
        if not isinstance(obj, dict):
            continue
        card = normalize(obj)
        if card["scene"] or card["moves"]:
            out.append(card)
    return out


def _list(value: Any, limit: int = 4, cap: int = 24) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[，,、/；;]", value)
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        s = str(item or "").strip()[:cap]
        if s:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    """把模型输出修成固定形状。多出来的字段一律丢掉，免得多带东西进库。"""
    stimulus = str(data.get("stimulus") or "").strip()
    if stimulus not in STIMULI:
        # 一、模型把整条情境名填进 stimulus（"被示好 / 被表白"）→ 按覆盖表归一
        mapped = COVERAGE.get(stimulus) or next(
            (
                v
                for k, v in COVERAGE.items()
                if v and len(stimulus) >= 3 and (k in stimulus or stimulus in k)
            ),
            "",
        )
        # 二、值里含着某个标准类的子串（"被轻视（当众）"）→ 取长的那一类
        stimulus = mapped or next(
            (s for s in sorted(STIMULI, key=len, reverse=True) if s in stimulus), ""
        )
        # 三、仍然认不出就留空，成品头写「未分类」。
        # 绝不兜成"被逼近"：那会把超纲情境伪装成标准类，统计与去重全跟着错，且看不出错。
    level = str(data.get("level") or "").strip()
    return {
        # 档位由人判断，模型不给；这里只在人工写入时保留合法值
        "level": level if level in LEVELS else "",
        "stimulus": stimulus,
        "scene": str(data.get("scene") or "").strip()[:24],
        "emotion_from": str(data.get("emotion_from") or "").strip()[:8],
        "emotion_to": str(data.get("emotion_to") or "").strip()[:8],
        "moves": _list(data.get("moves"), 4),
        "volume": str(data.get("volume") or "").strip()[:16],
        "length": str(data.get("length") or "").strip()[:16],
        "probe": str(data.get("probe") or "").strip()[:20],
        "emotion_shown": str(data.get("emotion_shown") or "").strip()[:12],
        "language": _list(data.get("language"), 4),
        "avoid": _list(data.get("avoid"), 3),
    }


def _grams(text: str, n: int = 8) -> set[str]:
    s = _WS.sub("", text or "")
    return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}


def name_hits(text: str, names: list[str]) -> list[str]:
    """专名命中。名字长度 >= 2 才算，单字名会误伤正文。"""
    hits: list[str] = []
    for name in sorted({n for n in names if n and len(n) >= 2}, key=len, reverse=True):
        if name in (text or ""):
            hits.append(f"出现专名「{name}」")
    return hits


def relation_hits(text: str) -> list[str]:
    """关系称呼命中。"""
    return [f"出现关系称呼「{w}」" for w in RELATION_WORDS if w in (text or "")]


def violations(card: dict[str, Any], names: list[str], source: str) -> list[str]:
    """三类硬校验。返回人话写的命中原由，空列表表示通过。"""
    hits: list[str] = []
    fields = [str(card.get(k) or "") for k in ("stimulus", "scene", "emotion_from", "emotion_to",
                                               "volume", "length", "probe", "emotion_shown")]
    for key in ("moves", "language", "avoid"):
        fields.extend(str(x) for x in (card.get(key) or []))
    blob = "".join(fields)

    hits.extend(name_hits(blob, names))
    hits.extend(relation_hits(blob))
    source_grams = _grams(source)
    if source_grams:
        for field in fields:
            if _grams(field) & source_grams:
                hits.append(f"搬了原文：「{field[:12]}」")
                break
    # 同一个词重复报没意义，保留前四条即可
    return list(dict.fromkeys(hits))[:4]


def example_hits(example: str, names: list[str]) -> list[str]:
    """例句的校验：**只查专名**。

    关系称呼故意不查。例句是人挑的一句原话，原话里带称呼本来就是常态
    （"爸，别说了"），而黑名单里的"小姐"连"本小姐"这种自称都会误伤——
    实测就撞上了。三道闸防的是模型编出来的卡面，不是人挑的原句：
    后者还有 `范例：已核` 的勾选与归纳AI 复核兜着。

    专名照查不误：人名进了 few-shot 就是"她认识这个人"，那是真污染。
    """
    text = (example or "").strip()
    if not text:
        return []
    return list(dict.fromkeys(name_hits(text, names)))[:4]


def render_head(
    card: dict[str, Any],
    evidence_lines: list[str] | None = None,
    reviewed: bool = False,
    exemplar: bool = False,
    example: str = "",
    generated_by: str = "蒸馏器·反应样式",
    level_hint: str = "",
) -> list[str]:
    """反应卡头部。两条产线共用，差别只在证据行与生成者署名。

    刻意不叫「情境」：那个字段在对话稿成品头里代表关系档，用同名会诱导归纳AI
    把「被逼近」当档位读。
    """

    def join(key: str) -> str:
        return _JOIN.join(card.get(key) or []) or "—"

    level = card.get("level") or level_hint
    pages = ["reactions.md", "speech.md"]
    if level:
        pages.append("relationships.md")
    # 用途跟着范例开关走：没勾就绝不出现 exemplars.md
    if exemplar:
        pages.append("exemplars.md")

    head = [
        f"# {time.strftime('%Y-%m-%d')} 反应·{card.get('stimulus') or '未分类'}",
        "",
        "> 形态：反应样式",
        f"> 档位：{level or '—'}",
        f"> 刺激：{card.get('stimulus') or '—'}",
        f"> 场景：{card.get('scene') or '—'}",
        f"> 情绪：{card.get('emotion_from') or '—'} → {card.get('emotion_to') or '—'}",
        f"> 回应：{join('moves')}",
        f"> 话量：{card.get('volume') or '—'}"
        + (f"（{card['length']}）" if card.get("length") else ""),
        f"> 追问：{card.get('probe') or '—'}",
        f"> 露出：{card.get('emotion_shown') or '—'}",
        f"> 语言：{join('language')}",
        f"> 不做：{join('avoid')}",
    ]
    if exemplar:
        head.append("> 范例：已核")
        if example.strip():
            head.append(f"> 例句：{example.strip()}")
    head.append(f"> 用途：{_JOIN.join(pages)}")
    head.append("> 置信：medium（模型抽象，非原话）")
    for line in evidence_lines or []:
        head.append(f"> 证据：{line}")
    head.append(f"> 生成：{generated_by} {time.strftime('%Y-%m-%d %H:%M')}")
    head.append(f"> 审：{'已过' if reviewed else '未过'}")
    return head


def render(
    card: dict[str, Any],
    meta: dict,
    seg: dict,
    reviewed: bool,
    exemplar: bool = False,
    example: str = "",
) -> str:
    """蒸馏台成品。没有对白，没有身份，没有关系。"""
    paras = seg.get("paras") or ["0", "0"]
    evidence = (
        f"小说《{meta.get('name', '')}》{seg.get('chapter_title') or ''}"
        f"（第 {paras[0]}~{paras[-1]} 段）"
    )
    return (
        "\n".join(
            render_head(
                card,
                [evidence],
                reviewed,
                exemplar=exemplar,
                example=example,
                generated_by="蒸馏器·反应样式",
            )
        )
        + "\n"
    )

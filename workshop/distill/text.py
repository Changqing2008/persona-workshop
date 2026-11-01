"""小说文本层：解码、章节、段落、引语与说话人线索。

这里全部是可解释的规则，不调模型。理由很实际：
一本书几十万字，逐段问模型既贵又不稳，而且一旦错了没法解释是哪一步错的。
规则能抽的（"某某说"这类）抽干净，抽不出的留空交给上层按下一条规则处理，
最后仍认不出的直接丢掉——宁可少收，不可猜错。

三条口径：
- 引号里的才是台词；引号外的旁白只用来取"谁说的"和"什么神态"。
- 说话人只认能落到名字上的；只有"她/他"一律算未识别。
- 动作括号只收短状语（低声、头也不抬），心理描写和环境描写都不收。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# 章节标题：第X章/节/回/卷/部/篇，或 Chapter N。行首行尾都算，但长度不超过 44 字，
# 免得把正文里"第一章讲的是……"这种句子认成标题。
CHAPTER_RE = re.compile(
    r"^\s*(?:"
    r"第\s*[0-9零一二三四五六七八九十百千两]{1,6}\s*[章节回卷部篇]"
    r"|Chapter\s+\d{1,4}\b"
    r"|CHAPTER\s+\d{1,4}\b"
    r")[^\n]{0,40}$"
)

# 中文引号。只收成对且不跨引号的；网络小说里「」和“”混用很常见，两种都认。
QUOTE_RE = re.compile(r"[“「]([^“”「」]{1,600})[”」]")

# 说话动词。长的放前面，正则替换时先匹配"低声说"再匹配"说"，动作才不会被切错。
SPEECH_VERB = (
    "低声说|轻声说|冷笑道|笑着说|沉声道|开口道|接着道|继续说道|解释道|补充道|反问道|提醒道|警告道"
    "|说道|问道|答道|回道|应道|骂道|喊道|叫道|叹道|念道|回答|回应|接口|嘀咕|嘟囔|咕哝|低语|喃喃"
    "|冷笑|冷哼|嗤笑|苦笑|失笑|低笑"
    "|说|道|问|答|喊|叫|嚷|骂|吼|叹|应|回|念|开口|补充|反问|解释|提醒|警告"
)
VERB_RE = re.compile(r"(?:%s)" % SPEECH_VERB)

# 单字「道」有一半时候是名词：下水道 / 味道 / 街道 / 知道 / 难道。
# 不排掉就会把「我曾向父王提过我对下水道的知识」认成"父王说"。
_DAO_AS_NOUN = set("水味街通轨管渠河车跑航板叉隧夹过知难地")

# 名字前面是这些介词，它就是宾语，不是说话人：「我曾向父王提过」——
# 说话的是"我"，父王只是听的人。
_PREPOSITIONS = set("向对跟和与给替被让把朝冲为同")

PRONOUNS = ("她", "他", "它", "我", "你")

# 句末标点：一段话以这些结尾，就认为段落结束（应对按 80 字符硬换行的 txt）
_TERMINAL = re.compile(r"[。！？….”』」]\s*$")

# 认名字时要排除的常用词。宁可漏，也不要把"于是她"当成人名。
_STOP_NAMES = {
    "他们", "她们", "我们", "你们", "自己", "大家", "众人", "两人", "二人", "三人",
    "某人", "有人", "那人", "这人", "旁人", "对方", "所有人", "两个", "一个",
    "什么", "怎么", "这样", "那样", "于是", "然后", "接着", "突然", "最后",
    "只是", "但是", "可是", "如果", "因为", "所以", "虽然", "不过", "而且",
    "这时", "那时", "此时", "当时", "同时", "一直", "一边", "一面", "终于",
    "不由", "不禁", "忍不住", "轻声", "低声", "大声", "小声", "冷声", "沉声",
    "笑着", "冷笑", "淡淡", "缓缓", "慢慢", "忽然", "猛然", "随即", "立刻",
    "点头", "摇头", "抬头", "低头", "回头", "转身", "皱眉", "叹了", "顿了",
}
_STOP_PREFIX = ("但", "却", "于", "又", "便", "就", "也", "还", "只", "正", "刚", "才",
                "忽", "猛", "突", "然", "可", "所", "因", "为", "如", "与", "和", "跟",
                "给", "让", "被", "把", "在", "从", "对", "向", "朝", "冲",
                "没", "不", "看", "听", "想", "笑", "哭", "站", "坐", "转", "抬", "低", "点")
# 动作括号里不许出现的东西：心理、推断、因果、残余动词
# 动词自身带的语气：「低声说」「冷笑道」里的"低声""冷笑"其实就是状态
_MANNER = ("低声", "轻声", "大声", "小声", "冷声", "沉声", "柔声", "厉声",
           "冷笑", "嗤笑", "苦笑", "失笑", "低笑", "笑着", "喃喃")

_ACTION_BAN = re.compile(
    r"想|觉得|心想|知道|明白|意识|感到|似乎|仿佛|好像|因为|所以|如果|但是|然后|于是"
    r"|说|道|问|答|喊|叫|骂|吼|叹|念|看|望|听|见|是|的"
)
# 这些不是状态，是量词/虚词：「冷笑一声」里的「一声」不能进括号
_ACTION_JUNK = {
    "一声", "一句", "两句", "几句", "一下", "半天", "片刻", "会儿", "一顿",
    "一句句", "一句半句", "两声", "三声",
}

# 段间分隔符：＊　＊　＊ / ──── 这类只用来断开场景，不是内容
SEP_RE = re.compile(r"^[＊*◇◆○●※\-=─—\s]{2,}$")

# 引号后面紧跟这些，说明整段是在说"那个词"，不是在记人说的话
META_AFTER = re.compile(
    r"^[\s]*的?(?:表情|神情|神色|眼神|样子|声音|这句话|这个词"
    r"|这两个字|这三个字|两字|二字|一词|之说|与否)"
)
# 引号前面出现这些，同上
META_BEFORE = re.compile(r"(?:所谓|名为|称作|冠上|叫作|叫做|那句|一句|二字|一词)[^\n]{0,3}$")


def is_speech(text: str, m: "re.Match[str]") -> bool:
    """这个引号里的是人在说话，还是被强调引用的一个词？

    这本书里「魔法省」「超级」「剑」「卡住」全被当成台词收下来——它们是机构名、
    形容、物件、动词，只是用引号强调。这类误收比漏收更糟：它会让挑选界面里
    混进一堆根本不是对话的行，人还得一条条删。

    三条同时成立才判为非台词，宁可放过不可错杀：

    1. 引语很短（≤8 个正文字符）——强调引用都很短，真对话很少这么短又嵌在句中间；
    2. 引号前面紧挨着汉字——说明它嵌在叙述句内部，不是段落/句子开头起话；
    3. 紧邻 8 字内没有说话动词，且整段旁白远多于引语——旁白才是这段话的主角。

    第 3 条的窗口必须窄：拿整段去找动词会被"顾问""说明""如他本人所说"这类
    普通词骗过去，把整段的引号都判成台词。真正的归属线索只可能在引号两侧几个字内。
    """
    q = m.group(1).strip()
    bare = re.sub(r"[\W_]", "", q)
    if not bare:
        return False  # 「────」这种，连字都没有

    if META_AFTER.match(text[m.end() : m.end() + 8]):
        return False
    if META_BEFORE.search(text[max(0, m.start() - 8) : m.start()]):
        return False

    if len(bare) > 8:
        return True  # 这么长的引语不会是强调引用

    prev = text[m.start() - 1] if m.start() else ""
    if not prev or not re.match(r"[\u4e00-\u9fa5]", prev):
        return True  # 段首或标点后起话，是对话

    near = text[max(0, m.start() - 8) : m.start()] + text[m.end() : m.end() + 8]
    if VERB_RE.search(near):
        return True  # 紧挨着"说/问/道"，引语就是这句话的内容

    after = text[m.end() :].strip()
    if len(after) <= 8:
        return False  # 引语后面立刻就收句了，说明它是名词短语的一部分

    narr = QUOTE_RE.sub("", text)
    return len(narr) <= max(20, len(bare) * 4)  # 旁白不够长，仍算对白


def para_kind(text: str) -> str:
    """段落角色：SEP 分隔符 / DIALOGUE 纯台词 / DIALOGUE+ 台词带叙述 / CUE 线索 / NARR 纯叙述。

    这本书 53% 的段是纯台词段，说话线索全在紧邻的旁白段里。
    把 CUE 和 NARR 混成一类，就等于永远看不到线索。
    """
    s = text.strip()
    if SEP_RE.match(s):
        return "SEP"
    if QUOTE_RE.search(text):
        return "DIALOGUE" if len(QUOTE_RE.sub("", text).strip()) <= 4 else "DIALOGUE+"
    return "CUE" if VERB_RE.search(text) else "NARR"


def cue_speaker(text: str, names: list[str]) -> str:
    """纯旁白段里的说话人：名字必须是说话动词的主语。

    只在这三种情况返回名字，其余一律空：
    - 名字紧贴动词（中间不超过 6 字），排除"她"/"他"作宾语的句子；
    - 名字后面不是「看着她 / 问他」这种宾语结构。

    宁可空着也不算——线索段写的是"某某变了脸色询问我"时，
    那个某某是在追问，台词其实出自旁白里的"我"。
    """
    if QUOTE_RE.search(text):
        return ""
    for m in VERB_RE.finditer(text):
        if m.group(0) == "道" and m.start() and text[m.start() - 1] in _DAO_AS_NOUN:
            continue  # 「下水道」「知道」里的道不是动词
        # 只往前看到上一个标点，跨句的主语不算这句话的主语
        head = re.split(r"[，。！？；：\n]", text[: m.start()])[-1]
        cands = [(head.rfind(n), n) for n in names if n in head]
        if not cands:
            continue
        pos, name = max(cands)
        if pos and head[pos - 1] in _PREPOSITIONS:
            continue  # 「向父王提过」——父王是宾语，说话的是别人
        tail = head[pos + len(name) :]
        if len(tail) > 6:
            continue
        if re.search(r"[她他]", tail):
            continue
        return name
    return ""


def decode_bytes(data: bytes) -> tuple[str, str]:
    """把字节解成文本，返回 (文本, 编码名)。

    顺序不能反：gb18030 几乎什么字节都能解，放前面会把 UTF-8 解成乱码。
    """
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        # 再兜一层：解出来一半是替换符，说明选错编码了
        if text.count("\ufffd") <= max(2, len(text) // 2000):
            return text, enc
    return data.decode("utf-8", "replace"), "utf-8?"


@dataclass(slots=True)
class Chapter:
    index: int
    title: str
    paras: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Utterance:
    """一句台词。speaker 为空表示没认出来。"""

    text: str
    speaker: str = ""
    action: str = ""
    mode: str = "explicit"
    """explicit = 原文写了；alternate = 按轮次推的；unknown = 没认出来。"""


def _reflow(lines: list[str]) -> list[str]:
    """把硬换行的 txt 拼回段落。以句末标点收尾才算一段结束。"""
    out: list[str] = []
    buf = ""
    for line in lines:
        s = line.strip()
        if not s:
            if buf:
                out.append(buf)
                buf = ""
            continue
        buf = f"{buf}{s}" if buf else s
        if _TERMINAL.search(s) or len(buf) >= 400:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def split_chapters(text: str) -> list[Chapter]:
    """按章节标题切。认不出标题就整本当一章，不硬猜。"""
    lines = text.splitlines()
    marks: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s and len(s) <= 44 and CHAPTER_RE.match(s):
            marks.append((i, s))

    if len(marks) < 2:
        return [Chapter(index=0, title="全文", paras=_reflow(lines))]

    chapters: list[Chapter] = []
    if marks[0][0] > 0:
        head = _reflow(lines[: marks[0][0]])
        if head:
            chapters.append(Chapter(index=0, title="卷首", paras=head))
    for k, (i, title) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        chapters.append(
            Chapter(index=len(chapters), title=title, paras=_reflow(lines[i + 1 : end]))
        )
    return chapters


def discover_names(
    paragraphs: list[str], limit: int = 200, min_count: int = 2
) -> list[tuple[str, int]]:
    """从"XX说/道/问"里捞人名候选，按出现次数排序。

    只做候选，不做判定——真正用哪个由人在界面上点。
    min_count=1 用于归属阶段的"这本书里出现过谁"，只出现一次的名字也算。
    """
    counter: Counter[str] = Counter()
    for para in paragraphs:
        for m in re.finditer(r"([\u4e00-\u9fa5]{2,4})(?:%s)" % SPEECH_VERB, para):
            name = m.group(1)
            if not _plausible_name(name):
                continue
            counter[name] += 1
    return [(n, c) for n, c in counter.most_common(limit) if c >= min_count]


def _plausible_name(name: str) -> bool:
    if not 2 <= len(name) <= 4:
        return False
    if name in _STOP_NAMES:
        return False
    if any(p in name for p in PRONOUNS):
        return False
    if name[-1] in "地着了儿的":
        return False
    # 名字中间夹"了/着/的"的，几乎都是半句话（"看过了还"），不是人名
    if any(ch in "地着了" for ch in name[1:-1]):
        return False
    if name.startswith(_STOP_PREFIX):
        return False
    return True


_NEG_PRE = re.compile(r"(?:没有|没|未|不)(?:立刻|马上|直接|再|多)?$")
# 这些动词被否定时只是叙述（"没有立刻回答"），不构成归属
_NARRATIVE_VERBS = {"回答", "回应", "开口", "出声", "应声", "作声", "解释", "搭理", "说话"}


def _verb_match(clause: str):
    """找一个"真的在标记说话"的动词。

    "没有立刻回答"里的"回答"不算——它是叙述，不是归属。
    只否定这一种搭配，不能见"不"就当否定：「头也不抬地说」里的"不"是状语的一部分。
    """
    for m in VERB_RE.finditer(clause):
        if m.group(0) == "道" and m.start() and clause[m.start() - 1] in _DAO_AS_NOUN:
            continue  # 「下水道」「知道」里的道不是动词
        if m.group(0) in _NARRATIVE_VERBS:
            pre = clause[max(0, m.start() - 10) : m.start()]
            if _NEG_PRE.search(pre):
                continue
        return m
    return None


def _pick_clause(before: str, after: str) -> str:
    """决定归属线索取引号前还是引号后。有说话动词的那一边才算线索。"""
    if _verb_match(after[:18]):
        return after
    if _verb_match(before[-36:]):
        return before[-36:]
    return ""


def _find_speaker(clause: str, names: list[str]) -> str:
    for name in sorted(names, key=len, reverse=True):
        if name and name in clause:
            return name
    return ""


def _clean_action(text: str) -> str:
    s = text.strip(" \t\r\n，,。：:；;、！？!?“”「」")
    s = re.sub(r"^[的地着了]+|[的地着了]+$", "", s)
    if not 1 <= len(s) <= 12:
        return ""
    if s in _ACTION_JUNK:
        return ""
    if _ACTION_BAN.search(s):
        return ""
    return s


def _find_action(clause: str, speaker: str) -> str:
    """从线索句里抠一个短状语：「她头也不抬地说」→「头也不抬」。

    只取紧贴动词前的那一小段（遇到标点就断），动宾太长就放弃；心理描写一律不要。
    """
    m = _verb_match(clause)
    if not m:
        return ""
    head = re.split(r"[，。！？；：\n“”「」]", clause[: m.start()])[-1]
    for token in ([speaker] if speaker else []) + list(PRONOUNS):
        if token and token in head:
            head = head.replace(token, "", 1)
            break
    head = re.sub(r"^(?:又|才|便|就|也|还|只|却|正|刚|先|再|忽然|突然|随即|不禁|不由)+", "", head.strip())
    action = _clean_action(head)
    if action:
        return action
    # 状语被并进动词里了：「低声说」「冷笑道」
    verb = m.group(0)
    for manner in _MANNER:
        if verb.startswith(manner):
            action = _clean_action(manner)
            if action:
                return action
            break
    # 动词后的短补语也算状态：「说，声音有点低」
    tail = clause[m.end() :]
    tail = re.split(r"[。！？\n“”「」]", tail, maxsplit=1)[0]
    return _clean_action(tail)


def _narrate(text: str) -> str:
    """引语之外的旁白。

    判断"这段里有谁"必须只看旁白：引语里出现的名字是被称呼、被提到的人，
    恰恰不是说话人（「我要跟父王说我想学魔法！」的说话人不是父王）。
    这本书 53% 的段是纯引语段，这个区分是归属对错的分水岭。
    """
    return QUOTE_RE.sub(" ", text)


# 敬称：名字后面常跟一个，然后是标点。「尤莉安娜小姐，请坐。」里的呼语是
# 「尤莉安娜小姐」整体，只看到「小姐」还判不出来。
_TITLES = (
    "骑士团长|女骑士|大小姐|小姐|先生|殿下|大人|少爷|公主|王子|夫人|阁下"
    "|公爵|侯爵|伯爵|子爵|男爵|将军|陛下|老师|师父|师傅|哥哥|姐姐|弟弟|妹妹"
)
# 名字（可带敬称）后面紧跟这些，就说明整句话是在叫这个人
_ADDR_TAIL = re.compile(r"^(?:(?:%s))?[\s，,。！？!?…、；;：:）)」』—－]" % _TITLES)
# 自我介绍：「我叫尤莉安娜。」名字指的是自己，不是被称呼的对象。
# 必须紧贴名字，中间不许夹别的字：「我是人喔，伊莉丝」不是自我介绍。
_SELF_INTRO = re.compile(r"(?:我叫|我是|本人|名字是|名叫|在下|吾名|自称|人家叫)\s*$")


def name_spans(text: str, names: list[str]) -> list[tuple[int, int, str]]:
    """台词里出现的人名，最长优先、不重叠。

    别名是前缀关系（尤菲 / 尤莉安娜）。短的先匹配会把「尤莉安娜小姐」切成
    「尤菲」+「莉亚小姐」，后面跟的就不是标点了，呼语判据整个失效。
    """
    spans: list[tuple[int, int, str]] = []
    taken = bytearray(len(text))
    for n in sorted((x for x in names if x), key=len, reverse=True):
        for m in re.finditer(re.escape(n), text):
            if any(taken[m.start() : m.end()]):
                continue
            taken[m.start() : m.end()] = b"\x01" * (m.end() - m.start())
            spans.append((m.start(), m.end(), n))
    return sorted(spans)


def is_addressed(text: str, name: str, names: list[str]) -> bool:
    """这句台词是不是在「叫」这个人 —— 如果是，说话的就绝不是他。

    这是全篇最强的一条反向证据，而且不依赖任何上下文：
    「尤莉安娜小姐，请坐。」是别人在请她坐；「父王。」是子女在喊父王；
    「我是人喔，伊莉丝。」是在对伊莉丝说话。
    把这类句子记到被称呼者头上，正是"台词里一出现角色名就当成他在说"的病根。

    判据：名字（可带一个敬称）后面紧跟句读标点，或台词到此为止。
    「我叫尤莉安娜。」这类自我介绍要放过，否则会把自己的台词判丢。
    """
    if not name or name not in text:
        return False
    for start, end, n in name_spans(text, names):
        # 别名同族：尤菲 / 尤莉安娜 指的是同一个人
        if not (n == name or n.startswith(name) or name.startswith(n)):
            continue
        if _SELF_INTRO.search(text[max(0, start - 6) : start]):
            continue
        if _ADDR_TAIL.match(text[end:]):
            return True
    return False


def cue_map(paras: list[str], names: list[str]) -> dict[int, str]:
    """块内线索绑定：段下标 → 由相邻旁白指认的说话人。

    只绑"一侧是台词"的线索段。两侧都是台词时，分不清旁白描述的是前一句还是后一句，
    这种情况一律不绑——猜错的代价远高于空着。

    例：把「「……女儿明白。」」和下一段的「看到尤莉安娜小姐微微点头」连起来，
    才认得出这句话是尤菲说的。段内永远看不到这条线索。
    """
    kinds = [para_kind(p) for p in paras]
    out: dict[int, str] = {}
    for i, p in enumerate(paras):
        if kinds[i] != "CUE":
            continue
        name = cue_speaker(p, names)
        if not name:
            continue
        has_prev = i > 0 and kinds[i - 1].startswith("DIALOGUE")
        has_next = i + 1 < len(kinds) and kinds[i + 1].startswith("DIALOGUE")
        if has_next and not has_prev:
            out[i + 1] = name
        elif has_prev and not has_next:
            out[i - 1] = name
    return out


def parse_paragraph(
    text: str,
    names: list[str],
    genders: dict[str, str] | None = None,
    recent: list[str] | None = None,
) -> list[Utterance]:
    """抽出一段里的所有台词。

    认人有三层，越往后越保守：
    1. 线索句里直接写着名字 → explicit；
    2. 只有"她/他" → 用同一段里唯一性别吻合的名字反推（pronoun）；
    3. 同段上一句已经认出来了 → 承接同一人（same），中文小说里一段内的连续引语极少换人；
    4. 认不出 → 留空，交给上层按轮次交替推（或不收）。

    第 2 层是必要的：中文小说写起来满篇"她说"，不认代词几乎收不到东西。
    但只在"这一段只有一个性别吻合的人"时才敢认，两个女性角色同场就退回第 3/4 层。

    "这段里有谁"一律取旁白（_narrate），不取整段：引语里的人名是被称呼者。
    """
    quotes = list(QUOTE_RE.finditer(text))
    if not quotes:
        return []

    genders = genders or {}
    narr = _narrate(text)
    present = [n for n in names if n in narr]
    # 这一段只写"她"，人名在上一段：拿上一段的人做代词消歧的候选，但不用于"整段唯一人名"
    pool = present or list(recent or [])
    out: list[Utterance] = []
    prev_speaker = ""
    for i, m in enumerate(quotes):
        if not is_speech(text, m):
            continue  # 强调引用的词不是台词，连行都不该产
        before = text[quotes[i - 1].end() if i else 0 : m.start()]
        after = text[m.end() : quotes[i + 1].start() if i + 1 < len(quotes) else len(text)]
        clause = _pick_clause(before, after)
        speaker = _find_speaker(clause, names) if clause else ""
        mode = "explicit"
        if not speaker and clause:
            speaker = _resolve_pronoun(clause, narr, pool, genders)
            if speaker:
                mode = "pronoun"
        if not speaker:
            speaker = _resolve_topic(narr, present)
            if speaker:
                mode = "topic"
        if not speaker and prev_speaker:
            speaker, mode = prev_speaker, "same"
        action = _find_action(clause, speaker) if clause else ""
        out.append(
            Utterance(
                text=m.group(1).strip(),
                speaker=speaker,
                action=action,
                mode=mode if speaker else "unknown",
            )
        )
        if speaker:
            prev_speaker = speaker
    return out


def addresses_other(
    text: str, names: list[str], others: list[str], known: list[str]
) -> bool:
    """这句台词在叫「别人」，没有叫目标角色 —— 多半是目标角色在说话。

    纯引语段（本书占 53%）段内一个线索都没有，只能靠 A→B→A→B 推；
    一旦某一句推错，后面会整串跟着错。这是那种段落里少有的**正向**证据：
    「艾妮丝菲雅公主，请多指教。」—— 在叫对方，说话的就是她。

    仍然只是推测，所以不作锚点，只用来盖掉轮次交替给出的答案。
    同时叫了两个人的句子一律放弃：分不清在跟谁说话。
    """
    if any(is_addressed(text, n, known) for n in names):
        return False  # 在叫目标角色，那说话人确定不是她
    return any(is_addressed(text, n, known) for n in others)


def _resolve_pronoun(
    clause: str, para: str, present: list[str], genders: dict[str, str]
) -> str:
    """「她低声说」里的她是谁：同一段只有一个候选人才敢认。

    两个候选就看性别；性别判不出来宁可放弃。
    还有一种要排除：名字后面跟着"看着她/问他"——这时代词是宾语，说话的是另一个人。
    """
    if not present:
        return ""
    if "她" in clause and "他" not in clause:
        want, pronoun = "female", "她"
    elif "他" in clause and "她" not in clause:
        want, pronoun = "male", "他"
    else:
        return ""
    matched = [n for n in present if genders.get(n) == want]
    if len(matched) == 1:
        return matched[0]
    if matched:
        return ""
    if len(present) != 1:
        return ""
    name = present[0]
    if re.search(re.escape(name) + r"[^。！？]{0,6}(?:看|望|转向|对|冲|朝|向|问|叫)" + pronoun, para):
        return ""
    return name


def _resolve_topic(para: str, present: list[str]) -> str:
    """整段只提到一个人，也没写"谁说"，就认为这句话是他说的。

    中文小说里这种段落极多（"老周没有抬头，笔尖还在纸上走。'那份东西我看过了。'"），
    不认就收不到几句。但名字只是宾语时必须放弃——"她看着他"里的他，说话的是第三个人。

    para 必须是**旁白**（_narrate 的结果），present 也来自旁白。
    拿整段去数名字会把"被称呼的人"当成说话人：
    「我要跟父王说我想学魔法！」——父王出现在引语里，但说这话的是艾妮丝。
    """
    if len(present) != 1:
        return ""
    name = present[0]
    if re.search(
        re.escape(name) + r"[^。！？]{0,4}(?:看着|看向|望着|转向|对着|冲着|朝|向|问|叫)[她他]", para
    ):
        return ""
    return name


def infer_genders(paragraphs: list[str], names: list[str]) -> dict[str, str]:
    """按"名字附近用的是她还是他"粗判性别，只用于代词消歧。

    判不出来的留空，不硬猜——猜错了会把台词记到另一个人头上。
    """
    scores: dict[str, list[int]] = {n: [0, 0] for n in names}
    for para in paragraphs:
        for name in names:
            idx = para.find(name)
            if idx < 0:
                continue
            window = para[max(0, idx - 8) : idx + len(name) + 16]
            scores[name][0] += window.count("她")
            scores[name][1] += window.count("他")
    out: dict[str, str] = {}
    for name, (she, he) in scores.items():
        out[name] = "female" if she > he else "male" if he > she else ""
    return out


def guess_category(lines: list[str]) -> str:
    """按关键词猜分类，只用来给个默认值，界面上可以改。

    宁可猜"日常搭话"，也不要为了显得聪明乱贴标签。
    """
    blob = "\n".join(lines)
    rules = (
        ("冲突对峙", ("滚", "闭嘴", "凭什么", "不可能", "少来", "你算", "威胁", "翻脸", "吵")),
        ("边界触碰", ("不许", "不准", "别管", "到此为止", "最后一次", "别再", "不要再说")),
        ("关系试探", ("是不是", "你到底", "我们算", "什么关系", "在乎", "喜欢我")),
        ("回忆往事", ("记得", "当年", "以前", "那时", "小时候", "从前", "后来")),
        ("价值判断", ("值得", "应该", "对错", "原则", "底线", "值不值")),
        ("自我介绍", ("我叫", "我是", "叫我")),
        ("情绪反应", ("哭", "笑", "生气", "难过", "委屈", "害怕")),
    )
    for category, words in rules:
        if any(w in blob for w in words):
            return category
    return "日常搭话"

"""素材台的离线测试。

覆盖三件事：
1. 模型输出解析 —— 它会加代码块、加前缀、偶尔不听话，这些都得能兜住。
2. 收稿格式 —— 头部标注决定归纳AI 怎么读，字段缺一个就白写。
3. HTTP 全流程 —— 开卡、对演、挑选、撤销、收稿、覆盖率。

不需要 API Key：模型被替换成假的，只看管道通不通。

    python test_sim.py
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workshop import config, library, log, webui
from workshop.memory import Memory
from workshop.persona import Persona
from workshop.sim import Studio, dims
from workshop.sim.cards import Card, _system_prompt, generate_cards, parse_cards
from workshop.sim.studio import Turn, _parse_candidates

PASS, FAIL = 0, 0

# 测试自带一个示例人格库，不读任何人的真实库。
# 以前这里指向作者本机的一个私人库，换台机器就跑不了，开源也不能带。
_DEMO_NAME = "示例角色"
PERSONA_DIR = Path(tempfile.mkdtemp(prefix="sim-persona-")) / _DEMO_NAME
library.create_persona(PERSONA_DIR, name=_DEMO_NAME)
(PERSONA_DIR / "raw" / "identity.md").write_text(
    "# 身份母本\n\n"
    "## 这个人是谁\n"
    "示例角色，素材台测试用的固定人格。\n"
    "断言全部对着这份内容写，不依赖外部人格库。\n",
    encoding="utf-8",
)

CARDS_JSON = json.dumps(
    [
        {
            "name": "走廊那句",
            "category": "冲突对峙",
            "relation": "同组同事，合作三个月",
            "setting": "复盘会散场后的走廊",
            "other_role": "认为方案有硬伤的同事",
            "other_goal": "要她拿出证据",
            "tension": "她嘴上不认输，但确实漏了一处",
            "opening": ["你那套东西，真跑过吗？"],
        },
        {
            "name": "深夜借两万",
            "category": "关系试探",
            "relation": "认识八年的老同学",
            "setting": "凌晨两点的微信",
            "other_role": "老同学",
            "other_goal": "借钱又不想说用途",
            "tension": "她刚交完房租",
            "opening": ["在吗，睡了吗"],
        },
        {
            "name": "被拿来比较",
            "category": "情绪反应",
            "relation": "表姐",
            "setting": "年夜饭桌上",
            "other_role": "表姐",
            "other_goal": "当众证明自己过得更好",
            "tension": "长辈都在场",
            "opening": ["你现在一个月多少钱"],
        },
    ],
    ensure_ascii=False,
)

CANDS_JSON = json.dumps(
    ["行啊，那你指出哪崩。", "我没跑过，但我知道它会在哪崩。", "先说说你觉得哪不对。"],
    ensure_ascii=False,
)

# 一张干净的反应卡：没有任何人名、称呼、事件
REACT_JSON = json.dumps(
    {
        "stimulus": "被试探",
        "scene": "对方递来一件她早就知道的事",
        "emotion_from": "被试探",
        "emotion_to": "不接",
        "moves": ["先反问", "只认一半", "不顺着说"],
        "volume": "一到两句",
        "length": "6~14 字",
        "probe": "答完就停",
        "emotion_shown": "不耐烦",
        "language": ["句末少用问号"],
        "avoid": ["不道歉"],
    },
    ensure_ascii=False,
)

# 脏卡：把"对谁"写进了场景，必须被出口校验拦住
REACT_BAD_JSON = json.dumps(
    {
        "stimulus": "被逼近",
        "scene": "被父亲在书房连续追问",
        "emotion_from": "被逼",
        "emotion_to": "沉默",
        "moves": ["不说话"],
        "volume": "一句",
        "length": "4 字",
        "probe": "—",
        "emotion_shown": "冷",
        "language": [],
        "avoid": [],
    },
    ensure_ascii=False,
)

# 卡点很长的卡：用来验"照抄条件"会被 8 字重合校验抓住
CARD_COPY = {
    "name": "十分钟内答复",
    "category": "冲突对峙",
    "relation": "熟人",
    "setting": "深夜的私聊窗口，对方连发三条",
    "other_role": "追问的熟人",
    "other_goal": "要她当场表个态",
    "tension": "她必须在十分钟内给答复，否则对方会把截图发到群里",
    "opening": ["你就说一句，站哪边"],
}

COPY_JSON = json.dumps(
    {
        "stimulus": "被逼近",
        "scene": "她必须在十分钟内给答复",
        "emotion_from": "被逼",
        "emotion_to": "烦",
        "moves": ["先反问"],
        "volume": "一句",
        "length": "8 字",
        "probe": "不追",
        "emotion_shown": "冷",
        "language": [],
        "avoid": [],
    },
    ensure_ascii=False,
)


BATCH_JSON = """[
 {"stimulus":"被试探","level":"熟人","scene":"对方递来一件她早就知道的事","emotion_from":"被试探","emotion_to":"不接",
  "moves":["先反问","只认一半"],"volume":"一到两句","length":"6~14 字","probe":"答完就停",
  "emotion_shown":"不耐烦","language":["句末少用问号"],"avoid":["不道歉"]},
 {"stimulus":"被轻视","level":"陌生人","scene":"对方把她说的话当玩笑","emotion_from":"被轻视","emotion_to":"压着火",
  "moves":["把话说完","不接茬"],"volume":"两三句","length":"短句","probe":"不追问",
  "emotion_shown":"冷淡","language":["不用语气词"],"avoid":["不解释"]},
 {"stimulus":"被关心","level":"拥有者","scene":"对方连着问了三遍她怎么了","emotion_from":"被关心","emotion_to":"想躲",
  "moves":["只说一半","岔开"],"volume":"一句","length":"6 字以内","probe":"答完就停",
  "emotion_shown":"敷衍","language":["用「没事」收尾"],"avoid":["不诉苦"]}
]"""


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \033[32m✓\033[0m {name}")
    else:
        FAIL += 1
        print(f"  \033[31m✗\033[0m {name}  {detail}")


class FakeLLM:
    """按 system 内容分发：要卡给卡，要候选给候选。"""

    def __init__(self) -> None:
        self.cfg = config.LLMConfig(api_key="sk-test")
        self.calls = 0

    @property
    def ready(self) -> bool:
        return True

    async def chat(self, messages, **kw) -> str:
        self.calls += 1
        system = messages[0]["content"] if messages else ""
        if "情境卡" in system:
            # 故意套一层代码块，验证解析能剥
            return "好的，下面是三张卡：\n```json\n" + CARDS_JSON + "\n```"
        return CANDS_JSON

    async def complete(self, *a, **kw) -> str:
        return "摘要"

    def reload(self, cfg) -> None:
        self.cfg = cfg


class ReactLLM(FakeLLM):
    """只回一张固定卡片，用来验管道与校验，不验模型水平。"""

    def __init__(self, payload: str = REACT_JSON) -> None:
        super().__init__()
        self.payload = payload

    async def chat(self, messages, **kw) -> str:
        self.calls += 1
        return self.payload


def http(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data or None, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sim-test-"))
    persona = Persona(PERSONA_DIR)
    cfg = config.Config(root=tmp, sim=config.SimConfig(dir="sim", candidates=3), )
    cfg.llm.api_key = "sk-test"
    cfg.bot.persona_dir = str(PERSONA_DIR)
    llm = FakeLLM()
    studio = Studio(cfg, persona, llm)
    memory = Memory(tmp / "test.db")

    try:
        # ---------- 解析 ----------
        print("\n[解析]")
        cards = parse_cards("```json\n" + CARDS_JSON + "\n```")
        check("代码块包裹能解", len(cards) == 3, str(len(cards)))
        check("分类归一化", cards[0].category == "冲突对峙", cards[0].category)
        check("开场白保留", cards[0].opening[0].startswith("你那套东西"), cards[0].opening)

        check("缺开场白的卡被丢", len(parse_cards('[{"name":"空卡","opening":[]}]')) == 0)
        check("乱七八糟的输入不炸", parse_cards("我不太会写 JSON") == [])
        check("对象包裹也能取", len(parse_cards('{"cards":[' + CARDS_JSON[1:-1] + "]}")) == 3)

        cands = _parse_candidates('```\n1. 甲\n2. 乙\n```')
        check("候选退化成按行切", cands == ["甲", "乙"], str(cands))
        check("引号被剥掉", _parse_candidates('["「在。」"]') == ["在。"], str(_parse_candidates('["「在。」"]')))

        scene = cards[0].scene()
        check("场景含关系与卡点", "同组同事" in scene and "漏了一处" in scene, scene[:40])
        check("分类映射到 traits", "traits" in cards[0].pages(), str(cards[0].pages()))

        # 输出被 token 上限切断是这个功能最常见的故障，必须能救回完整的卡
        truncated = CARDS_JSON[: int(len(CARDS_JSON) * 0.78)]
        salvaged = parse_cards(truncated)
        check("截断时能救回完整的卡", 1 <= len(salvaged) < 3, str(len(salvaged)))

        weird = '[{"name":"带花括号","category":"日常搭话","opening":["他说「{不行}」"]}]'
        check("字符串里的花括号不误判", len(parse_cards(weird)) == 1, str(parse_cards(weird)))

        # ---------- 生成失败的处理 ----------
        print("\n[生成容错]")

        class FlakyLLM:
            """第一次不听话，第二次给一张卡。"""

            def __init__(self):
                self.cfg = config.LLMConfig(api_key="sk-test")
                self.n = 0
                self.last_finish = "length"

            async def chat(self, messages, **kw):
                self.n += 1
                if self.n == 1:
                    return "抱歉，我无法按要求输出"
                one = json.loads(CARDS_JSON)[0]
                return json.dumps([one], ensure_ascii=False)

        flaky = FlakyLLM()
        got = await generate_cards(flaky, 3)
        check("失败后退化为单张重试", len(got) == 1 and flaky.n == 2, f"{len(got)} 次调用={flaky.n}")

        class DumbLLM:
            def __init__(self):
                self.cfg = config.LLMConfig(api_key="sk-test")
                self.last_finish = "length"

            async def chat(self, messages, **kw):
                return "就是不写 JSON"

        try:
            await generate_cards(DumbLLM(), 3)
            check("彻底失败时抛错", False, "居然没抛")
        except ValueError as exc:
            check("错误里说清是截断", "截断" in str(exc), str(exc))

        # ---------- 引擎 ----------
        print("\n[引擎]")
        draft = studio.new_draft(cards[0], "other")
        check("开场自动落成第一句", len(draft.turns) == 1 and draft.turns[0].source == "card")
        check("显示名从目录推导", studio.self_label == _DEMO_NAME, studio.self_label)

        items = await studio.candidates(draft, "self", 3)
        check("出三条候选", len(items) == 3, str(items))
        studio.commit(draft, "other", "我不信，你就是在拖。", "human")
        studio.commit(draft, "self", items[1], "ai", dropped=[items[0], items[2]])
        c = draft.counts()
        check("人工与采纳分开计数", c["human"] == 1 and c["ai"] == 1, str(c))
        check("未采纳被记录", c["dropped"] == 2, str(c["dropped"]))

        result = studio.export(draft)
        md = Path(result["path"]).read_text(encoding="utf-8")
        check("文件名符合 raw 规范", result["name"].startswith("20") and result["name"].endswith(".md"), result["name"])
        for key in ("用途：", "情境：", "张力：", "对方：", "构成：", "生成："):
            check(f"头部含「{key.strip('：')}」", key in md)
        check("正文带角色名", f"**{studio.self_label}**：" in md, md[-80:])
        check("未采纳数写进头部", "未采纳 2 条" in md, md[:400])
        check("分类写进头部便于追溯用途", "> 分类：冲突对峙" in md, md[:400])
        check("不写 raw 目录", not (tmp / "raw").exists())
        check("草稿落在 drafts/", Path(result["path"]).parent.name == "drafts", result["path"])

        again = studio.export(draft)
        check("同名自动加序号", again["name"].endswith("_2.md"), again["name"])

        before = len(draft.turns)
        studio.undo(draft)
        check("撤销回退一句", len(draft.turns) == before - 1)

        # 就地改一句：开场白也要能改，改成空等于删掉
        e_draft = studio.new_draft(cards[1], "other")
        studio.edit_turn(e_draft, 0, "在吗，睡了没")
        check("引擎能改句", e_draft.turns[0].text == "在吗，睡了没", e_draft.turns[0].text)
        check("改过算手写", e_draft.turns[0].source == "human", e_draft.turns[0].source)
        studio.edit_turn(e_draft, 9, "越界的")
        check("越界不动", len(e_draft.turns) == 1, str(len(e_draft.turns)))
        studio.edit_turn(e_draft, 0, "   ")
        check("改成空等于删掉", len(e_draft.turns) == 0, str(len(e_draft.turns)))

        cov = {x["page"]: x for x in studio.coverage()}
        check("覆盖率统计到 traits", cov["traits.md"]["segments"] == 2, str(cov["traits.md"]))
        check("未生产的页为 0", cov["boundaries.md"]["segments"] == 0, str(cov["boundaries.md"]))

        # ---------- 条件与句子结构全可改 ----------
        print("\n[编辑能力]")
        m_draft = studio.new_draft(cards[2], "other")
        studio.edit_turn(m_draft, 0, "你现在一个月多少钱（改）")
        studio.insert_turn(m_draft, 0, "self", "我妈问的")
        check("能在开头插一句",
              len(m_draft.turns) == 2 and m_draft.turns[0].role == "self", str(len(m_draft.turns)))
        studio.insert_turn(m_draft, 99, "other", "末尾追加")
        check("越界位置当追加", m_draft.turns[-1].text == "末尾追加", str(len(m_draft.turns)))
        studio.insert_turn(m_draft, 1, "self", "   ")
        check("空话不插", len(m_draft.turns) == 3, str(len(m_draft.turns)))

        studio.set_turn_role(m_draft, 0, "other")
        check("换人后记到对方名下",
              m_draft.turns[0].role == "other" and m_draft.turns[0].source == "human",
              str(m_draft.turns[0]))
        studio.move_turn(m_draft, 0, 1)
        check("下移交换相邻两句", m_draft.turns[1].text == "我妈问的", m_draft.turns[1].text)
        studio.move_turn(m_draft, 0, -1)
        check("顶到边界不动", m_draft.turns[0].text == "你现在一个月多少钱（改）", m_draft.turns[0].text)
        studio.move_turn(m_draft, 2, 1)
        check("到底边界不动", m_draft.turns[-1].text == "末尾追加", m_draft.turns[-1].text)

        # 卡点/关系是注入的一部分，改完下一轮候选必须跟着变
        before_sys = studio._self_system(m_draft, 3)
        new_card = Card.from_dict(dict(
            m_draft.card.as_dict(), tension="她必须在十分钟内答复", relation="拥有者",
            category="价值判断",
        ))
        studio.set_card(m_draft, new_card)
        check("条件能整卡换掉", m_draft.card.tension.startswith("她必须"), m_draft.card.tension)
        check("换条件不影响已有句子", len(m_draft.turns) == 3, str(len(m_draft.turns)))
        after_sys = studio._self_system(m_draft, 3)
        check("新卡点进注入", "十分钟内答复" in after_sys and "十分钟内答复" not in before_sys)
        check("新关系档进注入", "对方是拥有者" in after_sys)
        check("改分类后用途页跟着变", m_draft.card.pages() == ["values", "relationships"],
              str(m_draft.card.pages()))

        try:
            studio.set_card(m_draft, Card.from_dict({"name": "  "}))
            check("空标题的条件被拒", False, "居然没抛")
        except ValueError:
            check("空标题的条件被拒", True)

        # ---------- 谁在演谁 ----------
        print("\n[谁在演谁]")
        w = studio.new_draft(cards[1], "other")
        check("演对方时 AI 演她", w.mode == "other" and w.ai_role == "self", w.mode)
        studio.set_mode(w, "self")
        check("换边后 AI 演对面", w.mode == "self" and w.ai_role == "other", w.mode)
        check("换边落盘", studio.get(w.id).mode == "self", studio.get(w.id).mode)
        studio.set_mode(w, "self")
        check("换到同一边是空操作", w.mode == "self", w.mode)

        # 「对方」这个词以前同时指两个人，模型就在这儿把她的台词当对面交上来
        w.card.other_role = "老周"
        studio.commit(w, "self", "嗯，她是个好人", "human")
        her_view = studio._transcript(w, "self")
        his_view = studio._transcript(w, "other")
        check("她那一侧看到的自己是「你」", "你：" in her_view, her_view)
        check("她那一侧看到的对面带真名", "老周" in her_view, her_view)
        check("对面那一侧看到的自己是「你」", "你：" in his_view, his_view)
        check("她的话在对面视角里用真名", f"{_DEMO_NAME}：" in his_view, his_view)
        other_sys = studio._other_system(w)
        check("对面注入写明谁是谁",
              "你：老周" in other_sys and "对面：示例角色" in other_sys, other_sys[:240])
        check("对面注入不再用「对方」指她", "对方" not in other_sys, other_sys)

        # 说话人标错的候选必须丢掉
        mixed = json.dumps(
            [
                {"speaker": "老周", "text": "你大半夜说这个干什么"},
                {"speaker": "示例角色", "text": "我说她是好人就是好人"},
                {"speaker": "示例角色", "text": "别的我也说不清"},
            ],
            ensure_ascii=False,
        )
        only_him = _parse_candidates(mixed, expect="other", her="示例角色", other="老周")
        check("说话人不对的候选被丢", only_him == ["你大半夜说这个干什么"], str(only_him))
        only_her = _parse_candidates(mixed, expect="self", her="示例角色", other="老周")
        check("反过来也认得出", only_her == ["我说她是好人就是好人", "别的我也说不清"], str(only_her))
        check("没有 speaker 时不过滤",
              len(_parse_candidates(CANDS_JSON, expect="other", her="示例角色", other="老周")) == 3)

        # ---------- 卡片库 ----------
        print("\n[卡片库]")
        saved = studio.save_card(cards[0])
        check("收进库", saved["ok"] and saved["path"].endswith(".toml"), str(saved))
        check("库里能读到", len(studio.library()) == 1, str(len(studio.library())))

        again_save = studio.save_card(cards[0])
        check("重复收藏是幂等的", again_save.get("existed") and len(studio.library()) == 1, str(again_save))

        twin = Card.from_dict(dict(cards[0].as_dict(), id="c-twin-0001"))
        twin_save = studio.save_card(twin)
        check("同名不同 id 不重复入", twin_save.get("existed") and len(studio.library()) == 1, str(twin_save))

        drawn = studio.draw_cards(1)
        check("从库抽卡", len(drawn) == 1 and drawn[0].id == cards[0].id, str(drawn))
        studio.new_draft(drawn[0], "other")
        check("开演一次计数 +1", studio.library()[0].used == 1, str(studio.library()[0].used))

        check("出库", studio.delete_card(cards[0].id) and len(studio.library()) == 0)
        check("删除不存在的卡返回 False", studio.delete_card("不存在") is False)
        check("id 里的路径分隔符被清掉", studio._card_path("../../etc/passwd").name == "etcpasswd.toml")

        # ---------- 关系三档 ----------
        print("\n[关系三档]")
        rng = random.Random(7)
        b = dims.brief(rng, "陌生人")
        check("指定关系就用它", b["relation"] == "陌生人", b["relation"])
        check("不指定时随机一档", dims.brief(rng)["relation"] in dims.RELATIONS)
        check("原料清单标出关系", "关系：陌生人" in dims.brief_text([b]))
        check("关系只有三档", dims.RELATIONS == ("拥有者", "熟人", "陌生人"), str(dims.RELATIONS))

        check("关系不是三档时不给档位", Card.from_dict({"relation": "同组同事"}).level == "")
        card_d = Card.from_dict({
            "name": "刚被介绍认识", "category": "关系试探", "relation": "陌生人",
            "setting": "包间里", "other_role": "陌生人",
            "other_goal": "摸清她是谁", "tension": "她不想被问太多", "opening": ["你是做什么的"],
        })
        check("关系认得出档位", card_d.level == "stranger", card_d.level)
        check("三档关系也记进关系页", "relationships" in card_d.pages(), str(card_d.pages()))

        class CaptureLLM(FakeLLM):
            """记下喂给模型的原料清单，同时照常返回固定卡。"""

            def __init__(self) -> None:
                super().__init__()
                self.sent = ""

            async def chat(self, messages, **kw) -> str:
                self.sent = messages[-1]["content"] if messages else ""
                return await super().chat(messages, **kw)

        cap = CaptureLLM()
        got = await generate_cards(cap, 2, relation="陌生人")
        check("生成请求里带上了关系", "关系：陌生人" in cap.sent, cap.sent[:160])
        check("返回的卡被打上关系", all(c.relation == "陌生人" for c in got), str([c.relation for c in got]))

        draft_d = studio.new_draft(card_d, mode="other")
        res_d = studio.export(draft_d)
        md_d = Path(res_d["path"]).read_text(encoding="utf-8")
        check("成品情境行首段是关系", "> 情境：陌生人｜" in md_d, md_d[:300])

        cov_d = {(x["page"], x["level"]): x for x in studio.coverage()}
        check("关系页仍统计总段数", cov_d[("relationships.md", "")]["segments"] >= 1,
              str(cov_d[("relationships.md", "")]))
        check("陌生人档单独计数", cov_d[("relationships.md", "stranger")]["segments"] >= 1,
              str(cov_d[("relationships.md", "stranger")]))
        check("拥有者档显示为缺", cov_d[("relationships.md", "owner")]["segments"] == 0,
              str(cov_d[("relationships.md", "owner")]))
        check("三档条目常驻", all(("relationships.md", d) in cov_d for d in ("owner", "familiar", "stranger")))
        check("三档有中文标签", cov_d[("relationships.md", "owner")]["label"] == "关系·拥有者",
              cov_d[("relationships.md", "owner")]["label"])

        sys_d = studio._self_system(draft_d, 3)
        check("对演提示词写明关系", "对方是陌生人" in sys_d, sys_d[-400:])
        check("对演仍带人格约束", "行为约束" in sys_d)
        check("对演也带母本", "这个人是谁" in sys_d, sys_d[:400])

        # 母本必须进生成提示词，否则卡与她无关
        sys_p = _system_prompt("## 一句话是谁\n小小白，14 岁，只在 QQ 上活着")
        check("母本进生成提示词", "小小白" in sys_p, sys_p[-200:])
        check("母本块有明确标题", "这个人是谁" in sys_p)
        check("空母本给兜底说明", "还没有设定" in _system_prompt(""))
        check("空母本不让编身世", "不要编造身份" in _system_prompt(""))

        # 模型写了别的档位：标签跟内容走，绝不硬贴
        class WrongLLM(FakeLLM):
            async def chat(self, messages, **kw) -> str:
                return json.dumps([{
                    "name": "饭桌上那句", "category": "关系试探", "relation": "熟人",
                    "setting": "深夜的私聊窗口", "other_role": "老同学",
                    "other_goal": "问她最近怎么样", "tension": "她不想提",
                    "opening": ["你最近还好吧"],
                }], ensure_ascii=False)

        ws = await generate_cards(WrongLLM(), 1, relation="拥有者")
        check("档位不符时不硬改", ws[0].relation == "熟人", ws[0].relation)
        check("档位缺失时才补请求值",
              (await generate_cards(CaptureLLM(), 1, relation="熟人"))[0].relation == "熟人")

        # ---------- 反应卡：只留方式，不带场合/对方/台词 ----------
        print("\n[反应卡]")
        st_r = Studio(cfg, persona, ReactLLM())
        dr = st_r.new_draft(card_d, mode="other")
        dr.turns.append(Turn(role="self", text="你这截图我早上就看见了。", source="human"))
        st_r.save(dr)
        dr_id = dr.id

        got_r = await st_r.to_reaction(st_r.get(dr_id))
        check("干净对白能转出卡", got_r["status"] == "ok", str(got_r.get("violations")))

        md_r = st_r.preview_reaction(st_r.get(dr_id))
        check("首行是形态行", "> 形态：反应样式" in md_r, md_r[:200])
        check("没有情境行", "> 情境：" not in md_r, md_r[:400])
        check("没有对方行", "> 对方：" not in md_r, md_r[:400])
        check("没有正文台词", "**" not in md_r and "你这截图" not in md_r, md_r[-200:])
        check("不带角色名", st_r.self_label not in md_r, md_r[:300])
        check("档位取自卡的关系", "> 档位：陌生人" in md_r, md_r[:400])
        check("未勾范例时用途不含 exemplars", "exemplars.md" not in md_r, md_r[:600])
        check("证据指向草稿而非书名", "素材台对演" in md_r and "小说《" not in md_r, md_r[-300:])
        check("生成者署名是素材台", "素材台·反应样式" in md_r, md_r[-200:])

        try:
            st_r.export(st_r.get(dr_id), form="reaction")
            check("未过审不许收反应卡", False, "居然收了")
        except ValueError as exc:
            check("未过审不许收反应卡", "已过审" in str(exc), str(exc))

        try:
            st_r.update_reaction(st_r.get(dr_id), exemplar=True)
            check("未过审不许标范例", False, "居然标上了")
        except ValueError as exc:
            check("未过审不许标范例", "已过审" in str(exc), str(exc))

        st_r.update_reaction(st_r.get(dr_id), reviewed=True)
        out_r = st_r.export(st_r.get(dr_id), form="reaction")
        check("文件名按反应卡规则", "_反应_" in out_r["name"] and out_r["name"].startswith("20"),
              out_r["name"])
        check("收反应卡不顺手写对话稿", not st_r.get(dr_id).exported,
              str(st_r.get(dr_id).exported))

        # 范例开关：勾了才进 exemplars，例句是原句也不查 8 字重合
        st_r.update_reaction(st_r.get(dr_id), exemplar=True, example="你这截图我早上就看见了")
        md_ex = st_r.preview_reaction(st_r.get(dr_id))
        check("勾范例后用途含 exemplars", "exemplars.md" in md_ex, md_ex[:600])
        check("成品标出范例已核", "> 范例：已核" in md_ex, md_ex[:700])
        check("例句写进成品", "> 例句：你这截图我早上就看见了" in md_ex, md_ex[:700])

        out_ex = st_r.export(st_r.get(dr_id), form="reaction")
        check("范例卡能导出", Path(out_ex["path"]).is_file(), out_ex.get("path"))
        md_file = Path(out_ex["path"]).read_text(encoding="utf-8")
        check("导出的范例卡仍无场合对方", "> 情境：" not in md_file and "> 对方：" not in md_file)

        # 例句是人挑的原话：带称呼正常（"本小姐"都在里面），只拦人名
        st_r.update_reaction(st_r.get(dr_id), example="说了就说了，别指望本小姐夸你")
        check("例句带自称不算错", True)
        try:
            st_r.update_reaction(st_r.get(dr_id), example="示例角色也这么说过")
            check("例句夹带专名被拒", False, "居然放过")
        except ValueError as exc:
            check("例句夹带专名被拒", "专名" in str(exc), str(exc))

        # ---------- 例句自动生成 ----------
        print("\n[例句自动生成]")

        class ExampleLLM(ReactLLM):
            """按调用次序回不同的例句，验生成与自检回路。"""

            def __init__(self, payloads: list[str]) -> None:
                super().__init__()
                self.payloads = list(payloads)
                self.n = 0
                self.sent = ""

            async def chat(self, messages, **kw) -> str:
                self.n += 1
                self.sent = messages[-1]["content"] if messages else ""
                return self.payloads[min(self.n, len(self.payloads)) - 1]

        # 干净例句：一次通过，请求里带母本和卡面
        ex_ok = ExampleLLM(["「你这截图我早上就看见了。」"])
        st_e = Studio(cfg, persona, ex_ok)
        got_e = await st_e.generate_example(st_e.get(dr_id))
        check("干净例句一次通过", got_e == "你这截图我早上就看见了。", got_e)
        check("请求带母本与卡面", "## 她是谁" in ex_ok.sent and "刺激" in ex_ok.sent, ex_ok.sent[:200])
        check("请求要求不写人名", "不得出现任何人名" in ex_ok.sent, ex_ok.sent[-200:])

        # 带专名 → 带着原因重试一次 → 第二次干净
        ex_retry = ExampleLLM(["示例角色也这么说过", "说了就说了，别指望本小姐夸你"])
        st_e2 = Studio(cfg, persona, ex_retry)
        got_e2 = await st_e2.generate_example(st_e2.get(dr_id))
        check("带名字重试后通过", got_e2 == "说了就说了，别指望本小姐夸你", got_e2)
        check("重试带了原因", "上次不合格" in ex_retry.sent and "名字" in ex_retry.sent,
              ex_retry.sent[-120:])

        # 两次都带专名 → 放弃并让人手写
        ex_bad = ExampleLLM(["示例角色也这么说过", "又是示例角色说过的话"])
        st_e3 = Studio(cfg, persona, ex_bad)
        try:
            await st_e3.generate_example(st_e3.get(dr_id))
            check("两次违规就放弃", False, "居然放行")
        except ValueError as exc:
            check("两次违规就放弃", "手写" in str(exc), str(exc))

        # 没转反应卡、没 Key，都不许生成
        st_e4 = Studio(cfg, persona, ExampleLLM(["在。"]))
        try:
            await st_e4.generate_example(st_e4.new_draft(card_d, opening=False))
            check("没转卡不许生成例句", False, "居然出句")
        except ValueError as exc:
            check("没转卡不许生成例句", "先转一次" in str(exc), str(exc))
        class NoKeyYet:
            """没配 Key 的引擎：ready=False，不该能出例句。"""

            def __init__(self) -> None:
                self.cfg = config.LLMConfig(api_key="")

            @property
            def ready(self) -> bool:
                return False

        try:
            await Studio(cfg, persona, NoKeyYet()).generate_example(st_r.get(dr_id))
            check("没 Key 不许生成例句", False, "居然出句")
        except ValueError as exc:
            check("没 Key 不许生成例句", "API Key" in str(exc), str(exc))

        # 脏卡：模型把"对谁"写进场景，出口校验必须拦住
        st_b = Studio(cfg, persona, ReactLLM(REACT_BAD_JSON))
        db = st_b.new_draft(card_d, mode="other")
        db.turns.append(Turn(role="self", text="我不说。", source="human"))
        st_b.save(db)
        got_b = await st_b.to_reaction(st_b.get(db.id))
        check("带称呼的卡被拦", got_b["status"] == "blocked", str(got_b.get("violations")))
        try:
            st_b.update_reaction(st_b.get(db.id), reviewed=True)
            check("违规卡不许标已审", False, "居然让标了")
        except ValueError as exc:
            check("违规卡不许标已审", "违规" in str(exc), str(exc))

        # 覆盖看板要认「档位」，且 exemplars 拆两项
        cov_r = {(x["page"], x["level"]): x for x in st_r.coverage()}
        check("反应卡计入反应页", cov_r[("reactions.md", "")]["segments"] >= 1,
              str(cov_r.get(("reactions.md", ""))))
        check("反应卡按档位计入关系页", cov_r[("relationships.md", "stranger")]["segments"] >= 1,
              str(cov_r.get(("relationships.md", "stranger"))))
        check("行为范例单独计数", cov_r[("exemplars.md", "style")]["segments"] >= 1,
              str(cov_r.get(("exemplars.md", "style"))))
        check("exemplars 拆出的两项常驻",
              ("exemplars.md", "dialog") in cov_r and ("exemplars.md", "style") in cov_r)
        check("行为范例有中文标签",
              cov_r[("exemplars.md", "style")]["label"] == "exemplars·行为范例",
              cov_r[("exemplars.md", "style")]["label"])

        # ---------- 模拟：没有台词，只靠母本 + 卡的条件 ----------
        print("\n[模拟]")

        class SimCapture(ReactLLM):
            """记下喂给模型的请求，同时照常返回固定卡。"""

            def __init__(self) -> None:
                super().__init__()
                self.sent = ""

            async def chat(self, messages, **kw) -> str:
                self.sent = messages[-1]["content"] if messages else ""
                return await super().chat(messages, **kw)

        cap_s = SimCapture()
        st_s = Studio(cfg, persona, cap_s)
        ds = st_s.new_draft(card_d, opening=False)
        check("开稿不落开场白", len(ds.turns) == 0, str(len(ds.turns)))

        got_s = await st_s.simulate(st_s.get(ds.id))
        check("没台词也能模拟出卡", got_s["status"] == "ok", str(got_s.get("violations")))
        check("模拟请求带母本", "## 她是谁" in cap_s.sent, cap_s.sent[:120])
        check("模拟请求带卡的场合", "包间里" in cap_s.sent, cap_s.sent[:400])
        check("模拟请求带卡的卡点", "她不想被问太多" in cap_s.sent, cap_s.sent[:400])
        check("模拟请求没有台词区", "台词：" not in cap_s.sent, cap_s.sent[:300])
        check("模拟用生成式提示词", "推断" in cap_s.sent or "条件" in cap_s.sent, cap_s.sent[:200])

        md_s = st_s.preview_reaction(st_s.get(ds.id))
        check("模拟稿证据行写明推断", "按身份母本推" in md_s, md_s[-320:])
        check("模拟稿仍无场合对方", "> 情境：" not in md_s and "> 对方：" not in md_s, md_s[:400])
        check("模拟稿档位跟着卡", "> 档位：陌生人" in md_s, md_s[:400])

        # 照抄卡点 = 没抽象，必须被重合校验拦
        st_c = Studio(cfg, persona, ReactLLM(COPY_JSON))
        dc = st_c.new_draft(Card.from_dict(CARD_COPY), opening=False)
        got_c = await st_c.simulate(st_c.get(dc.id))
        check("照抄卡点被拦",
              got_c["status"] == "blocked" and "搬了原文" in str(got_c.get("violations")),
              str(got_c.get("violations")))

        print("\n[批量生成]")

        class BatchLLM(FakeLLM):
            """一次回三张卡；可以换成空串/坏 JSON 验兜底。"""

            def __init__(self, payload: str = BATCH_JSON) -> None:
                super().__init__()
                self.payload = payload
                self.sent = ""

            async def chat(self, messages, **kw) -> str:
                self.calls += 1
                self.sent = messages[-1]["content"] if messages else ""
                return self.payload

        batch = BatchLLM()
        st_g = Studio(cfg, persona, batch)
        cards = await st_g.generate_reactions("", "", 3)
        check("一次出三张", len(cards) == 3, str(len(cards)))
        check("三张都干净", all(x["status"] == "ok" for x in cards),
              str([x["violations"] for x in cards]))
        check("刺激各不相同", len({x["stimulus"] for x in cards}) == 3,
              str([x["stimulus"] for x in cards]))
        check("档位跟着卡走", {x["level"] for x in cards} == {"熟人", "陌生人", "拥有者"},
              str([x["level"] for x in cards]))

        # 每张都落成一稿，界面上才能一张张改
        d0 = st_g.get(cards[0]["id"])
        check("每张都落了稿", d0 is not None and bool((d0.reactions or {}).get("card")))
        check("稿里没有场合", d0.card.setting == "", repr(d0.card.setting))
        check("稿里没有对方", d0.card.other_role == "" and d0.card.other_goal == "", d0.card.other_role)
        check("稿里没有开场白", d0.card.opening == [], str(d0.card.opening))
        check("稿里没有台词", len(d0.turns) == 0, str(len(d0.turns)))

        md_g = st_g.preview_reaction(d0)
        check("成品无情境行", "> 情境：" not in md_g, md_g[:400])
        check("成品无对方行", "> 对方：" not in md_g, md_g[:400])
        check("成品有档位行", "> 档位：熟人" in md_g, md_g[:400])
        check("成品证据行写母本推断", "母本推断" in md_g and "无台词依据" in md_g, md_g[-300:])

        # 请求里不该出现任何场合类字段
        check("请求带母本", "## 她是谁" in batch.sent, batch.sent[:150])
        check("请求没有场合", "场合：" not in batch.sent, batch.sent[:400])
        check("请求没有对方想要", "对方想要" not in batch.sent, batch.sent[:400])

        # 下拉指定了就以指定为准
        batch2 = BatchLLM()
        st_g2 = Studio(cfg, persona, batch2)
        cards2 = await st_g2.generate_reactions("被冷落", "熟人", 2)
        check("指定刺激就全覆盖", all(x["stimulus"] == "被冷落" for x in cards2),
              str([x["stimulus"] for x in cards2]))
        check("指定档位就全覆盖", all(x["level"] == "熟人" for x in cards2),
              str([x["level"] for x in cards2]))
        check("指定项进了请求", "刺激类型：被冷落" in batch2.sent and "关系档：熟人" in batch2.sent,
              batch2.sent[:300])

        # 模型输出为空 → 报错，不硬放行
        st_bad = Studio(cfg, persona, BatchLLM("我不知道该写什么"))
        try:
            await st_bad.generate_reactions("", "", 3)
            check("空输出直接报错", False, "没报错")
        except ValueError as exc:
            check("空输出直接报错", "JSON" in str(exc), str(exc))

        # 缺 Key → 报错，不静默
        class NoKey(BatchLLM):
            @property
            def ready(self) -> bool:
                return False

        try:
            await Studio(cfg, persona, NoKey()).generate_reactions("", "", 3)
            check("缺 Key 报错", False, "没报错")
        except ValueError as exc:
            check("缺 Key 报错", "API Key" in str(exc), str(exc))

        # 乱填刺激 → 直接拒绝，不要发给模型
        try:
            await st_g.generate_reactions("被夸", "", 3)
            check("非法刺激被拒", False, "没报错")
        except ValueError as exc:
            check("非法刺激被拒", "刺激类型" in str(exc), str(exc))

    finally:
        memory.close()
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n结果：{PASS} 通过 / {FAIL} 失败\n")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    log.setup("INFO", None)
    raise SystemExit(asyncio.run(main()))

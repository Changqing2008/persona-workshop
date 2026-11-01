"""小说角色蒸馏器的离线测试。

覆盖四段：
1. 文本层 —— 解码、章节、硬换行、人名候选、代词消歧、动作括号。
2. 归属与成段 —— 谁能认出来、认不出的怎么丢、段怎么断。
3. 落稿 —— 素材台成品头字段一个不能少，且不能写 raw。
4. 互通与接口 —— 开进素材台、收进卡片库，以及 HTTP 全流程。
5. 块级裁判 —— 视图渲染、解析闸门、候选集封闭、规则档不调模型、失败静默回退。

不需要 API Key：模型调用一律用假客户端顶上，验的是管道不是模型水平。

    python test_distill.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workshop import config, webui
from workshop.distill import Distiller
from workshop.distill import cast
from workshop.distill import entities as EM
from workshop.distill import judge as DJ
from workshop.distill import react
from workshop.distill import text as T
from workshop.llm import LLMClient
from workshop.persona import Persona
from workshop.sim import Studio
from workshop.sim.cards import Card

PASS, FAIL = 0, 0

NOVEL = """第一章 夜谈

示例角色推开门，没有进来，只站在门框那儿。“你找我？”她低声说。

“嗯。”老周没有抬头，笔尖还在纸上走，“那份东西我看过了。”

“看过了还问我？”示例角色冷笑一声，“你是想听我自己说一遍。”

老周合上本子，终于看向她。“我想听实话。”

示例角色没有立刻回答。她把手插进衣袋，靠住了门框。“实话值钱吗？”

“值。”老周说，“至少比你现在这套值。”

她笑了一下，那笑没能撑到最后。“我要是说了，你打算怎么办？”

老周没有接这句，只把本子推过来。“你自己看。”

第二章 走廊

第二天早上，走廊里没有人。示例角色靠在窗边，手里捏着那张纸。“他昨晚没睡。”她说。

“谁没睡？”老周从拐角出来，声音有点哑。

“你说呢。”她把纸折成两半，塞进衣袋。
"""


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \033[32m✓\033[0m {name}")
    else:
        FAIL += 1
        print(f"  \033[31m✗\033[0m {name}  {detail}")


class FakeReactLLM:
    """只回一张固定卡片，用来验管道与校验，不验模型水平。"""

    ready = True

    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kw) -> str:
        self.calls.append(messages)
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload, ensure_ascii=False)


class FakeJudgeLLM:
    """按提示词里的在场人物回一份固定判定——验管道，不验模型水平。

    它自己从 USER 提示里读候选集，所以"候选集封闭"这条闸门同时被测到了：
    如果候选集没把说话人放进去，它会退到第一个候选，跟真模型的行为一致。
    `fail=True` 时每次调用都抛异常，验"裁判挂了也不许中断提取"。
    """

    ready = True

    def __init__(self, prefer: str = "老周", fail: bool = False) -> None:
        self.prefer = prefer
        self.fail = fail
        self.calls: list[list[dict]] = []
        self.picked: list[str] = []

    async def chat(self, messages, **kw) -> str:
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("假客户端：故意挂掉")
        user = str(messages[-1].get("content") or "")
        cands: list[str] = []
        n = 0
        for ln in user.splitlines():
            if ln.startswith("在场人物："):
                cands = [x for x in ln[len("在场人物："):].split("、")
                         if x and not x.startswith("（")]
            elif ln.startswith("共 ") and "句台词" in ln:
                try:
                    n = int(ln[2:].split(" ")[0])
                except ValueError:
                    n = 0
        who = self.prefer if self.prefer in cands else (cands[0] if cands else "")
        self.picked.append(who)
        return json.dumps(
            {"lines": [{"i": i, "who": who} for i in range(max(n, 1))]},
            ensure_ascii=False,
        )


REACT_OK = {
    "stimulus": "被逼近",
    "scene": "被连续追问，手里有一件不想提的事",
    "emotion_from": "被逼近",
    "emotion_to": "不认输，退半步",
    "moves": ["先反问", "再给结论", "不解释动机"],
    "volume": "一到两句",
    "length": "6~14 字",
    "probe": "答完就停",
    "emotion_shown": "不耐烦",
    "language": ["句末少用问号", "不用语气词"],
    "avoid": ["不道歉", "不复述对方的话"],
}


def make_env(tmp: Path) -> tuple[config.Config, Studio, Distiller]:
    persona_dir = tmp / "persona"
    (persona_dir / "wiki").mkdir(parents=True)
    (persona_dir / "index.md").write_text(
        "---\ntype: identity\n---\n示例角色，嘴硬心软。", encoding="utf-8"
    )
    cfg = config.Config(root=tmp)
    cfg.bot.persona_dir = str(persona_dir)
    cfg.sim.dir = "sim"
    cfg.distill.dir = "distill"
    persona = Persona(persona_dir)
    studio = Studio(cfg, persona, LLMClient(cfg.llm))
    return cfg, studio, Distiller(cfg, studio)


def http(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def main() -> int:
    print("\n=== 蒸馏器离线测试 ===\n")
    tmp = Path(tempfile.mkdtemp(prefix="qab-distill-"))
    try:
        cfg, studio, d = make_env(tmp)
        genders = {"示例角色": "female", "老周": "male"}

        # ---------- 文本层 ----------
        print("[文本层]")
        text, enc = T.decode_bytes("第一章\n示例角色说：“走吧。”".encode("utf-8"))
        check("UTF-8 认得出来", "示例角色" in text and enc.startswith("utf-8"), enc)
        text, enc = T.decode_bytes("第一章\n示例角色说：“走吧。”".encode("gb18030"))
        check("GBK 认得出来", "示例角色" in text and enc == "gb18030", enc)

        wrapped = "第一章 开头\n\n示例角色推开门，\n没有进来，只站在门口。\n她抬眼看了看屋里的人。\n\n第二章 后来\n\n老周合上本子。\n"
        chapters = T.split_chapters(wrapped)
        check("认出两章", len(chapters) == 2, str(len(chapters)))
        check("硬换行拼成完整句",
              chapters[0].paras[0] == "示例角色推开门，没有进来，只站在门口。",
              str(chapters[0].paras))
        check("单章时不硬切", len(T.split_chapters("只有一段话，没有标题。")) == 1)

        found = dict(T.discover_names(
            [p for c in T.split_chapters(NOVEL) for p in c.paras], min_count=1
        ))
        check("候选里有示例角色", found.get("示例角色", 0) >= 1, str(found))
        check("候选里有老周", "老周" in found, str(found))
        check("不把'于是她'当人名", "于是她" not in found, str(found))
        check("不把'看过了还'当人名", "看过了还" not in found, str(found))

        standard, aliases = cast.resolve("示例", [("示例角色", 5), ("示例", 3), ("老周", 4)])
        check("别名归到最长写法", standard == "示例角色", standard)
        check("短写法进别名", "示例" in aliases, str(aliases))
        check("别人不进别名", "老周" not in aliases, str(aliases))

        utts = T.parse_paragraph("“值。”老周说，“至少比你现在这套值。”", ["示例角色", "老周"], genders)
        check("同段两句都归老周", [u.speaker for u in utts] == ["老周", "老周"], str([u.speaker for u in utts]))
        check("显式归属不标推定", all(u.mode == "explicit" for u in utts), str([u.mode for u in utts]))

        utts = T.parse_paragraph("“我不想说。”她头也不抬地说。", ["示例角色", "老周"], genders, ["示例角色"])
        check("代词能落到人", utts and utts[0].speaker == "示例角色", str(utts[0].speaker if utts else ""))
        check("动作括号取短状语", utts and utts[0].action == "头也不抬", utts[0].action if utts else "")
        check("代词归属标推定", utts and utts[0].mode == "pronoun", utts[0].mode if utts else "")

        utts = T.parse_paragraph("“随便。”她冷冷地说。", ["示例角色", "老周"], genders, ["示例角色"])
        check("动词里的语气也进括号", utts and utts[0].action == "冷冷", utts[0].action if utts else "")

        utts = T.parse_paragraph("“嗯。”老周没有抬头，“那份东西我看过了。”", ["示例角色", "老周"], genders)
        check("整段只有一个人时归他", all(u.speaker == "老周" for u in utts), str([u.speaker for u in utts]))

        utts = T.parse_paragraph(
            "示例角色没有立刻回答。她把手插进衣袋，靠住了门框。“实话值钱吗？”",
            ["示例角色", "老周"], genders,
        )
        check("'没有立刻回答'不算说话", utts and utts[0].action == "", utts[0].action if utts else "")
        check("这句靠唯一人名兜住", utts and utts[0].speaker == "示例角色", str(utts and utts[0].speaker))

        cat = T.guess_category(["你凭什么这么说", "闭嘴"])
        check("冲突词判成冲突对峙", cat == "冲突对峙", cat)
        check("没线索时回落到日常搭话", T.guess_category(["今天天气不错"]) == "日常搭话")

        # 强调引用不是台词：这本书里「魔法省」「超级」「剑」全被当成台词收下来了
        def _speeches(s: str, names=("示例角色", "老周")) -> list[str]:
            return [u.text for u in T.parse_paragraph(s, list(names), genders)]

        check("机构名不当台词",
              _speeches("他是掌管国家机关「魔法省」的伯爵之子。") == [], "")
        check("强调词不当台词", _speeches("可以在前头再冠上「超级」两字的美少女。") == [], "")
        check("纯符号不当台词", _speeches("「────」") == [], "")
        check("强调引用的神态不当台词",
              _speeches("他用「问题不在那里吧？」的眼神刺向我。") == [], "")
        check("正常的短对白照收", _speeches("「嗯。」她点头。") == ["嗯。"], "")
        check("句中有短引语但带说话动词照样收",
              _speeches("她冷冷地说「不行」。") == ["不行"], "")
        check("段首起话的短句照收",
              _speeches("「走吧。」他站起来，拿起外套，往门口走去。") == ["走吧。"], "")

        # 反向归属：引语里出现的名字是被称呼的人，不是说话人。
        # 用户报的「话里有角色名就被当成本角色说话」就出在这儿，是全篇最硬的
        # 一条反向证据，不依赖任何上下文。
        KN = ["示例角色", "老周", "伊莉丝", "父王"]
        for said, who, want, why in [
            ("示例角色小姐，请坐。", "示例角色", True, "开头称呼 + 敬称"),
            ("我是人喔，伊莉丝，你的视力没问题吗？", "伊莉丝", True, "句中称呼"),
            ("父王。", "父王", True, "整句就是一声喊"),
            ("示例角色，你要去哪里？", "示例角色", True, "名字后直接跟逗号"),
            ("我叫示例角色。", "示例角色", False, "自我介绍不是被称呼"),
            ("我是示例角色。", "示例角色", False, "自我介绍不是被称呼"),
            ("示例角色是我的名字。", "示例角色", False, "名字作主语，后面不是标点"),
            ("这都要感谢示例角色小姐的帮助。", "示例角色", False, "所有格，后面接的是「的」"),
            ("那都要怪示例角色啦……", "示例角色", False, "名字后跟语气词，不是呼语"),
        ]:
            got = T.is_addressed(said, who, KN)
            check(f"呼语判据·{why}", got is want, f"{said} → {got}")

        check("别名同族：全名被称呼也算",
              T.is_addressed("示例角色小姐，请坐。", "示例", ["示例", "示例角色", "老周"]) is True, "")
        check("台词里没这个人：不算被称呼",
              T.is_addressed("今天天气不错。", "示例角色", KN) is False, "")

        # 线索段里的名字必须是主语，不能是介词宾语
        check("「向父王提过」：父王是宾语，不算他说",
              T.cue_speaker("我曾向父王提过下水道的知识。", KN) == "", "")
        check("「跟父王说过」：同上",
              T.cue_speaker("都是因为我跟父王说过。", KN) == "", "")
        check("「下水道」里的道不是说话动词",
              T.cue_speaker("父王提过下水道。", KN) == "", "")
        check("「知道」里的道也不是",
              T.cue_speaker("父王知道他来了。", KN) == "", "")
        check("真的「低声道」照样认",
              T.cue_speaker("示例角色低声道。", KN) == "示例角色", "")
        check("真的「问道」照样认",
              T.cue_speaker("老周问道。", KN) == "老周", "")

        # 正向证据：台词里在叫别人、没叫自己 —— 纯引语段里唯一能打断交替传染的东西
        ME_NAMES, OTHER_NAMES = ["示例角色"], ["老周", "伊莉丝"]
        check("在叫别人：当成目标角色说的",
              T.addresses_other("老周，你过来。", ME_NAMES, OTHER_NAMES, KN) is True, "")
        check("在叫自己：不能当成她说的",
              T.addresses_other("示例角色，你过来。", ME_NAMES, OTHER_NAMES, KN) is False, "")
        check("同时叫两个人：放弃判断",
              T.addresses_other("示例角色、老周，你们都过来。", ME_NAMES, OTHER_NAMES, KN) is False, "")
        check("谁也没叫：放弃判断",
              T.addresses_other("今天天气不错。", ME_NAMES, OTHER_NAMES, KN) is False, "")
        check("只是提到别人、不是在叫：放弃判断",
              T.addresses_other("我把老周的书弄丢了。", ME_NAMES, OTHER_NAMES, KN) is False, "")

        # ---------- 归属与成段 ----------
        print("\n[归属与成段]")
        book = d.import_text("测试小说", NOVEL)
        check("导入后有两章", book["chapters"] == 2, str(book))
        again = d.import_text("换了个名字", NOVEL)
        check("同一本书不重复入库", again["reused"] and again["id"] == book["id"], str(again))

        out = d.extract(book["id"], "示例角色", level="熟人")
        stats = out["stats"]
        check("引号台词都数到了", stats["quotes"] == 14, str(stats))
        check("全部收下（这段没有歧义）", stats["kept"] == 14 and stats["dropped"] == 0, str(stats))
        check("两侧都有", stats["self"] > 0 and stats["other"] > 0, str(stats))
        check("成段不为空", out["total"] >= 2, str(out["total"]))

        segs = out["segments"]
        check("每段都有来有回",
              all({x["role"] for x in s["lines"]} == {"self", "other"} for s in segs), "")
        check("段 id 唯一", len({s["id"] for s in segs}) == len(segs), str([s["id"] for s in segs]))

        # 整段原文没提到目标角色 = 她不在场，这些段全是交替推出来的幻觉，必须剔掉
        all_p = [p for c in T.split_chapters(NOVEL) for p in c.paras]

        def _in_span(s: dict) -> bool:
            a, z = int(s["paras"][0]), int(s["paras"][-1])
            return "示例角色" in "".join(all_p[a - 1 : z])

        check("留下来的段，原文里都真有她", all(_in_span(s) for s in segs), "")

        off_book = d.import_text(
            "旁人戏",
            "第一章 旁人\n\n"
            "「你吃了吗？」老周问。\n\n"
            "「吃了。你呢？」伊莉丝回答。\n\n"
            "「也没吃。」老周把碗放下，「今天厨房不知道怎么回事，一直没有人来。」\n\n"
            "「那我给你留一点吧。」伊莉丝站起来，「你等着，我马上就回来。」\n\n"
            "「不用了，我自己去就行。」老周也起身，「顺便看看那边到底出了什么事。」\n\n"
            "「你坐着吧。」伊莉丝按住他的肩膀，「你昨天一夜没睡，我看得出来。」\n\n"
            "老周没有再坚持，只是点了点头。「那就麻烦你了。」\n\n"
            "伊莉丝走出去，门在她身后关上了。廊下很安静，只有风从窗缝里钻进来。\n\n"
            "第二章 又是旁人\n\n"
            "「这里的窗子是不是坏了？」老周伸手推了推，木框晃得厉害。\n\n"
            "「去年就坏了。」伊莉丝把盘子放下，「一直没人来修，说了也没用。」\n\n"
            "「那我明天带工具来。」老周蹲下去看那道缝，「钉子松了，重新打一次就好。」\n\n"
            "「你还会这个？」伊莉丝有些意外地看着他。\n\n"
            "「会一点。」老周站起来，「总比一直漏风强。」\n\n"
            "伊莉丝笑了一下，没有再说什么。窗外的天色慢慢暗下来。\n",
        )
        try:
            off_out = d.extract(off_book["id"], "示例角色")
            off_total = off_out["total"]
        except Exception:  # noqa: BLE001
            off_total = 0  # 书里根本没这个人，连候选都排不出来，同样不该出段
        check("整本书没她：一段都不出", off_total == 0, str(off_total))

        # 成品：一段台词一条、空行分开（一段成品 = 一段对话，别挤成一整块）
        meta_b = d._load_meta(book["id"])
        md = d.render(meta_b, segs[0])
        rows = md.splitlines()
        at = [i for i, x in enumerate(rows) if x.startswith("**")]
        check("一句台词一条，段内不合并",
              len(at) == len(segs[0]["lines"]), f"{len(at)} vs {len(segs[0]['lines'])}")
        check("台词之间用空行分开",
              all(rows[at[k] + 1] == "" for k in range(len(at) - 1)),
              str([rows[at[0] + 1][:8]]))
        titles = {c["index"]: c["title"] for c in d.get(book["id"])["chapters"]}
        check("段落在同一章", all(titles.get(s["chapter"]) == s["chapter_title"] for s in segs), "")
        check("对方名落到段上", segs[0]["other_role"] == "老周", segs[0]["other_role"])
        check("关系档只有一个", all(s["level"] in ("拥有者", "熟人", "陌生人") for s in segs), "")

        # 轮次交替的两种档：chain 会一路 A→B→A→B 推下去，anchor 只推一句
        chain_novel = (
            "第一章 试\n\n示例角色说：“你来了。”\n\n“我来了。”\n\n“坐吧。”\n\n“不坐。”\n"
        )
        cn = T.split_chapters(chain_novel)
        cp = [p for c in cn for p in c.paras]
        for tag, want in (("chain", 4), ("anchor", 2), ("off", 1)):
            got, _ = d._attribute(cn, ["示例角色"], ["老周"], 3, cp, None, False, tag)
            check(f"轮次推测 {tag} 档收 {want} 句", len(got) == want, f"{len(got)} 句")
        got_anchor, _ = d._attribute(cn, ["示例角色"], ["老周"], 3, cp, None, False, "anchor")
        check("anchor 推的那句仍是对方",
              len(got_anchor) > 1 and got_anchor[1]["role"] == "other",
              str([(x["role"], x["mode"]) for x in got_anchor]))

        # 限步：steps 越大越多，0 = 不限
        def _cnt(steps: int) -> int:
            return len(d._attribute(
                cn, ["示例角色"], ["老周"], 3, cp, None, False, "chain", steps
            )[0])

        n0, n1, n2 = _cnt(0), _cnt(1), _cnt(2)
        check("steps=1 比不限少", n1 < n0, f"{n1} vs {n0}")
        check("steps=2 落在两者之间", n1 <= n2 <= n0, f"{n1} {n2} {n0}")

        got_deep, _ = d._attribute(cn, ["示例角色"], ["老周"], 3, cp, None, False, "chain")
        check("每句都带推演深度", all("depth" in x for x in got_deep), "")
        check("第一句是锚点，深度为 0", got_deep and got_deep[0]["depth"] == 0,
              str([x["depth"] for x in got_deep]))
        check("越往后推得越远", [x["depth"] for x in got_deep] == sorted(x["depth"] for x in got_deep),
              str([x["depth"] for x in got_deep]))

        # ---------- 落稿 ----------
        print("\n[落稿]")
        res = d.export(book["id"], [segs[0]["id"]])
        md = res["files"][0]["markdown"]
        for key in ("用途：", "分类：", "情境：", "张力：", "对方：", "构成：", "来源：", "生成："):
            check(f"头部含「{key.strip('：')}」", key in md)
        check("情境首段是三档", "> 情境：熟人｜" in md, md[:400])
        check("不带原作关系词", "师父" not in md and "同门" not in md, "")
        check("构成写「原文 N 句」", "> 构成：原文 " in md and "人工原话 0 句" in md, md[:500])
        check("正文带角色名", "**示例角色**：" in md, md[-200:])
        check("正文带对方", "**对方**：" in md, "")
        check("成品落在 out/", Path(res["files"][0]["path"]).parent.name == "out", res["files"][0]["path"])
        check("不写 raw", not (tmp / "raw").exists(), "")
        check("同名自动加序号", d.export(book["id"], [segs[0]["id"]])["files"][0]["name"].endswith("_2.md"), "")

        d.update_segment(book["id"], segs[0]["id"], {"category": "冲突对峙", "tension": "她不想提"})
        got = d.get(book["id"])
        check("分类能改", got["segments"][0]["category"] == "冲突对峙", got["segments"][0]["category"])
        check("用途跟着分类走", "traits.md" in d.render(got["book"], got["segments"][0]), "")

        # ---------- 互通 ----------
        print("\n[与素材台互通]")
        d.update_line(book["id"], segs[0]["id"], 0, "other")
        line0 = d.get(book["id"])["segments"][0]["lines"][0]
        check("换人后标 manual", line0["role"] == "other" and line0["mode"] == "manual", str(line0))
        before = len(d.get(book["id"])["segments"][0]["lines"])
        d.update_line(book["id"], segs[0]["id"], 0, "drop")
        check("删句生效", len(d.get(book["id"])["segments"][0]["lines"]) == before - 1, "")

        sim = d.to_sim(book["id"], segs[0]["id"])
        draft = studio.get(sim["draft_id"])
        check("草稿落在素材台", draft is not None and draft.card.name, "")
        check("句子来源记成 novel", all(t.source == "novel" for t in draft.turns), str([t.source for t in draft.turns[:2]]))
        check("素材台计数含原文", draft.counts()["novel"] == len(draft.turns), str(draft.counts()))
        composed = studio._render(draft)
        check("素材台收稿写「原文 N 句」", "原文 " in composed and "未采纳 0 条" in composed, composed[:400])

        lib = d.to_library(book["id"], segs[0]["id"])
        check("卡片进库", lib.get("ok") and Path(lib["path"]).parent.name == "cards", str(lib))
        check("卡片关系是三档", any(c.relation in ("拥有者", "熟人", "陌生人") for c in studio.library()), "")

        # ---------- 章节范围 ----------
        print("\n[章节范围]")
        full_total = out["total"]
        part = d.extract(book["id"], "示例角色", chapters=[1])
        check("只摘选中的章",
              all(s["chapter"] == 1 for s in part["segments"]),
              str([s["chapter"] for s in part["segments"]]))
        check("范围回传", part["chapters"] == [1], str(part["chapters"]))
        check("段数少于全书", 0 < part["total"] < full_total, f"{part['total']} / {full_total}")
        check("段号仍是全书位置",
              bool(part["segments"]) and part["segments"][0]["paras"][0] >= 9,
              str(part["segments"][0]["paras"] if part["segments"] else ""))
        check("get 返回本次范围", d.get(book["id"])["selected"] == [1], "")

        got = d.get(book["id"])
        check("默认不填关系档", got["segments"][0]["level"] == "", str(got["segments"][0]["level"]))
        blank_md = d.render(got["book"], got["segments"][0])
        check("情境位置写占位", "> 情境：—｜" in blank_md, blank_md[:300])
        check("用途不硬塞关系页", "relationships.md" not in blank_md, blank_md[:300])
        blank_draft = d.to_sim(book["id"], got["segments"][0]["id"])
        check("开进素材台时关系为空",
              studio.get(blank_draft["draft_id"]).card.relation == "",
              str(studio.get(blank_draft["draft_id"]).card.relation))
        # 模拟用户在素材台里划档：改条件 → 收稿，情境首段就该是那一档
        bd = studio.get(blank_draft["draft_id"])
        studio.set_card(bd, Card.from_dict(dict(bd.card.as_dict(), relation="陌生人")))
        check("素材台划档后写进情境",
              "> 情境：陌生人｜" in studio._render(studio.get(bd.id)),
              studio._render(studio.get(bd.id))[:260])

        bad = ""
        try:
            d.extract(book["id"], "示例角色", chapters=[99])
        except ValueError as exc:
            bad = str(exc)
        check("越界章被拒", "章" in bad, bad)

        check("范围串解析（1 基）", webui._parse_chapters("3-5,9") == [2, 3, 4, 8],
              str(webui._parse_chapters("3-5,9")))
        check("下标列表原样", webui._parse_chapters([1, 2]) == [1, 2], "")
        check("空值等于全书", webui._parse_chapters("") == [] and webui._parse_chapters([]) == [], "")

        # ---------- 块级裁判 ----------
        print("\n[块级裁判]")
        check("默认配置不开裁判（不偷偷烧 token）",
              config.Config(root=tmp).distill.judge == "rules",
              config.Config(root=tmp).distill.judge)
        check("设置页能切裁判档",
              any(f["path"] == "distill.judge" for f in webui.FIELDS), "")
        shown, n_show = DJ.render([
            ("示例角色推开门。“谁在？”", ["谁在？"]),
            ("老周抬起头。“是我。”", ["是我。"]),
        ])
        check("裁判视图编号从 0 起", "[0] 「谁在？」" in shown and "[1] 「是我。」" in shown, shown)
        check("裁判视图保留旁白", "【示例角色推开门。】" in shown, shown)
        check("台词句数对得上", n_show == 2, str(n_show))

        got = DJ.parse(
            '{"lines":[{"i":0,"who":"老周"},{"i":1,"who":"路人甲"},{"i":9,"who":"老周"}]}',
            2, ["示例角色", "老周"],
        )
        check("集外名字丢弃", got == {0: "老周"}, str(got))
        check("坏 JSON 不炸", DJ.parse("我看是老周说的", 2, ["老周"]) == {}, "")
        check("空值原样留白", DJ.parse('{"lines":[{"i":0,"who":""}]}', 1, ["老周"]) == {0: ""}, "")
        check("代码块包着也能解",
              DJ.parse('```json\n{"lines":[{"i":0,"who":"老周"}]}\n```', 1, ["老周"]) == {0: "老周"}, "")

        cands = DJ.pick_candidates(
            ["老周走进来，示例角色在窗边。"], ["甲", "乙", "丙", "丁", "戊", "己", "老周", "示例角色"]
        )
        check("候选集只放本块在场者", cands == ["老周", "示例角色"], str(cands))
        cands = DJ.pick_candidates(
            ["老周走进来。"], ["甲", "乙", "丙", "丁", "戊", "己", "老周"], prev="阿七"
        )
        check("候选集封顶 6 个", len(cands) <= DJ.MAX_CANDIDATES, str(cands))
        check("上一块说话人在候选里", "阿七" in cands, str(cands))

        # rules 档：一个字都不问模型
        cfg.distill.judge = "rules"
        quiet = FakeJudgeLLM()
        dr = Distiller(cfg, studio, quiet)
        base = dr.extract(book["id"], "示例角色")
        check("rules 档不调模型", not quiet.calls, str(len(quiet.calls)))
        check("rules 档统计归零", base["stats"]["judge_called"] == 0, str(base["stats"]))

        # 规则结果原样留一份，用来比对"模型能改什么、不能改什么"
        chapters = T.split_chapters((dr._dir(book["id"]) / "source.txt").read_text(encoding="utf-8"))
        allp = [p for c in chapters for p in c.paras]
        picked = EM.discover(chapters).cast("示例角色")
        KNOWN = list(dict.fromkeys(picked["names"] + picked["others"][:60]))
        rule_lines, _ = dr._attribute(chapters, picked["names"], picked["others"], 3, allp)
        # 原文写着"某某说"的行：模型说不同也压不过原文
        fixed_by_book = {(ln["ch"], ln["para"], ln["text"]): ln["speaker"]
                         for ln in rule_lines if ln["mode"] == "explicit"}
        check("样书里确有显式归属", len(fixed_by_book) >= 2, str(len(fixed_by_book)))

        # auto 档：真调模型，只覆盖靠猜的那几档。
        # 先清缓存，否则上一轮留下的判定会让"新调了几次"永远是 0。
        shutil.rmtree(tmp / "distill" / "cache", ignore_errors=True)
        cfg.distill.judge = "auto"
        fake = FakeJudgeLLM(prefer="老周")
        da = Distiller(cfg, studio, fake)
        judged = da.extract(book["id"], "示例角色")
        check("auto 档真的调了模型", bool(fake.calls), str(len(fake.calls)))
        check("一块一次调用", len(fake.calls) == judged["stats"]["judge_blocks"], str(judged["stats"]))
        check("改了至少一句", judged["stats"]["judge_fixed"] > 0, str(judged["stats"]))

        # 拿裁判的原始输出来验，而不是 segments：段现在最多 3 句，样本小说里
        # 有些段只有一方说话会被整段丢掉，llm 行也就跟着不在了。
        jlines, jstat = da._judge([dict(x) for x in rule_lines], allp, KNOWN, picked["standard"],
                                    tuple(picked["names"]))
        check("产出带 llm 行", any(ln["mode"] == "llm" for ln in jlines), "")
        check("llm 行落在候选集内",
              all(ln["speaker"] in ("示例角色", "老周") for ln in jlines if ln["mode"] == "llm"), "")

        kept = [(k, ln["speaker"]) for ln in jlines
                if (k := (ln["ch"], ln["para"], ln["text"])) in fixed_by_book]
        check("显式归属不被模型压过",
              all(who == fixed_by_book[k] for k, who in kept), str(kept[:4]))
        check("改判的行标 llm", any(ln["mode"] == "llm" for ln in jlines),
              str(sorted({ln["mode"] for ln in jlines})))
        check("模型没答的句子保持原样",
              sum(1 for a, b in zip(rule_lines, jlines) if a["speaker"] == b["speaker"]) > 0, "")

        # 缓存：同样的块再跑一遍不再花钱，结论还得一致
        again_lines, again_stat = da._judge([dict(x) for x in rule_lines], allp, KNOWN, picked["standard"],
                                    tuple(picked["names"]))
        check("同书重提取吃缓存", again_stat["judge_cache"] > 0 and not again_stat["judge_called"],
              str(again_stat))
        check("缓存结果与首次一致",
              [ln["speaker"] for ln in again_lines] == [ln["speaker"] for ln in jlines], "")

        # 失败回退：裁判整体挂掉，提取照常出结果
        cfg.distill.judge_cache = False
        boom = FakeJudgeLLM(fail=True)
        db = Distiller(cfg, studio, boom)
        fallback = db.extract(book["id"], "示例角色")
        check("裁判挂了不影响出段", fallback["total"] > 0, str(fallback["total"]))
        check("裁判挂了记失败数", fallback["stats"]["judge_failed"] > 0, str(fallback["stats"]))
        check("裁判挂了没有 llm 行",
              not any(ln["mode"] == "llm" for s in fallback["segments"] for ln in s["lines"]), "")
        cfg.distill.judge_cache = True

        cfg.distill.judge = "rules"
        # 下游反应卡测试要接着用 extract.json，把状态恢复成"只摘选中的章"
        d.extract(book["id"], "示例角色", chapters=[1])

        # ---------- 反应卡 ----------
        print("\n[反应卡]")
        toy = {
            "chapter": 0, "chapter_title": "第一章", "paras": [1, 2],
            "lines": [
                {"role": "self", "text": "示例角色，你别装了", "action": ""},
                {"role": "other", "text": "老周，你闭嘴", "action": "冷笑"},
            ],
        }
        masked, raw = react.mask(toy, ["示例角色", "老周"], "示例角色")
        check("主角换成「她」", "她，你别装了" in masked, masked)
        check("别人换成「某人」", "某人，你闭嘴" in masked, masked)
        check("遮罩后没有专名", "示例角色" not in masked and "老周" not in masked, masked)
        check("原文留给校验用", "示例角色" in raw and "老周" in raw, raw)

        cur = d.get(book["id"])
        sid = cur["segments"][0]["id"]
        ok_llm = FakeReactLLM(REACT_OK)
        dr = Distiller(cfg, studio, ok_llm)
        res = asyncio.run(dr.to_reaction(book["id"], sid))
        check("生成成功", res["status"] == "ok" and res["card"]["stimulus"] == "被逼近", str(res)[:160])
        sent = "\n".join(m["content"] for m in ok_llm.calls[0])
        check("提示词里没有人名", "示例角色" not in sent and "老周" not in sent, sent[:120])
        check("提示词里没有关系档", "熟人" not in sent and "拥有者" not in sent, "")
        check("反应卡头写形态", "> 形态：反应样式" in res["preview"], res["preview"][:200])
        check("反应卡没有对白", "**示例角色**" not in res["preview"] and "**对方**" not in res["preview"], "")
        check("用途锁死", "> 用途：reactions.md / speech.md" in res["preview"], "")
        check("未审就标明", "> 审：未过" in res["preview"], "")

        hits = react.violations({**REACT_OK, "moves": ["我要是说了，你打算怎么办"]}, [], "我要是说了，你打算怎么办？")
        check("搬原句被拒", any("搬了原文" in h for h in hits), str(hits))

        bad = Distiller(cfg, studio, FakeReactLLM({**REACT_OK, "scene": "老周站在门口"}))
        res = asyncio.run(bad.to_reaction(book["id"], sid))
        check("专名命中被拒",
              res["status"] == "blocked" and any("专名" in h for h in res["violations"]),
              str(res["violations"]))

        rel = Distiller(cfg, studio, FakeReactLLM({**REACT_OK, "scene": "被师父逼问"}))
        res = asyncio.run(rel.to_reaction(book["id"], sid))
        check("关系称呼命中被拒",
              res["status"] == "blocked" and any("关系称呼" in h for h in res["violations"]),
              str(res["violations"]))

        try:
            bad.update_reaction(book["id"], sid, reviewed=True)
            check("违规卡不能标已审", False, "居然过了")
        except ValueError as exc:
            check("违规卡不能标已审", "违规" in str(exc), str(exc))

        dr = Distiller(cfg, studio, FakeReactLLM(REACT_OK))
        asyncio.run(dr.to_reaction(book["id"], sid))
        try:
            dr.export_react(book["id"], [sid])
            check("未过审导不出去", False, "居然导出了")
        except ValueError as exc:
            check("未过审导不出去", "过审" in str(exc), str(exc))

        up = dr.update_reaction(book["id"], sid, reviewed=True)
        check("勾审后状态写入", up["reaction"]["reviewed"] is True, str(up["reaction"])[:120])
        check("档位默认不分档", "> 档位：—" in up["preview"], up["preview"][:120])

        lv = dr.update_reaction(book["id"], sid, card={"level": "熟人"})
        check("能选档位", lv["reaction"]["card"]["level"] == "熟人", str(lv["reaction"]["card"].get("level")))
        check("档位写进成品头", "> 档位：熟人" in lv["preview"], lv["preview"][:200])
        check("有档位就多挂关系页",
              "> 用途：reactions.md / speech.md / relationships.md" in lv["preview"],
              lv["preview"][:260])
        bad_lv = dr.update_reaction(book["id"], sid, card={"level": "师父"})
        check("非法档位被丢弃", bad_lv["reaction"]["card"]["level"] == "", str(bad_lv["reaction"]["card"].get("level")))
        dr.update_reaction(book["id"], sid, card={"level": "熟人"})

        again = Distiller(cfg, studio, FakeReactLLM(REACT_OK))
        re = asyncio.run(again.to_reaction(book["id"], sid))
        check("重转保留已选档位", re["card"]["level"] == "熟人", str(re["card"].get("level")))
        again.update_reaction(book["id"], sid, reviewed=True)

        out = again.export_react(book["id"], [sid])
        md = out["files"][0]["markdown"]
        check("导出到 react/", Path(out["files"][0]["path"]).parent.name == "react", out["files"][0]["path"])
        check("导出件标已过审", "> 审：已过" in md, md[:200])
        check("导出件带档位", "> 档位：熟人" in md, md[:200])
        check("导出件用途含关系页",
              "> 用途：reactions.md / speech.md / relationships.md" in md, md[:300])
        check("导出件不含对白", "**" not in md, md[-160:])

        shown = d.get(book["id"])
        check("反应卡按段回显", (shown["reactions"].get(sid) or {}).get("status") == "ok", str(list(shown["reactions"]))[:80])
        d.extract(book["id"], "示例角色", chapters=[1])
        again2 = d.get(book["id"])
        check("重新提取后卡片仍挂对段",
              bool(again2["reactions"].get(again2["segments"][0]["id"])),
              str(list(again2["reactions"]))[:80])

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n结果：{PASS} 通过 / {FAIL} 失败\n")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""一键归纳 · 扫料层测试。

守的是"第一次点按钮时看到什么"：待收几份、已收几份、拦住几份、为什么。
判错了只有两种后果，都很难看：
- 判成 `new` 判多了 → 已经收过的素材被重新收一遍，白烧调用还可能污染 wiki
- 判成 `done` 判多了 → 新素材永远收不进去，且没有任何提示

所以这里对每一类素材的判定都钉一条，尤其是 `log.md` 里那些没写全名的流水。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from workshop import ingest, library  # noqa: E402
from workshop import ingest_prompt  # noqa: E402
from workshop import ingest_rules as R  # noqa: E402
from workshop.llm import LLMError  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \033[32m✓\033[0m {name}")
    else:
        FAIL += 1
        print(f"  \033[31m✗\033[0m {name}  {detail}")


REACTION = """# 2026-10-27 反应·被关心

> 形态：反应样式
> 档位：拥有者
> 刺激：被关心
> 场景：对方把她手里的事接过去让她歇着
> 情绪：惊讶 → 满足
> 回应：先推掉，被坚持才接受
> 话量：少（一句一条）
> 追问：不追问
> 露出：惊讶转满足
> 语言：—
> 不做：—
> 用途：reactions.md / speech.md / relationships.md
> 置信：medium（模型抽象，非原话）
> 审：已过
"""

DIALOG = """# 2026-10-24 替人背锅

> 用途：boundaries.md / relationships.md
> 分类：边界触碰
> 情境：陌生人｜群里被@，两条消息晾着没回
> 张力：第三个人在听，她不想接又不好当场撕
> 对方：同群的不熟的人
> 构成：人工原话 1 句 / AI 采纳 0 句 / 未采纳 0 条
> 生成：素材台 2026-10-24 02:21

**对方**：这事你就顺手帮我认了吧

**示例角色**：滚
"""

DIALOG_AI_ONLY = """# 2026-10-25 推算稿

> 用途：speech.md
> 分类：日常搭话
> 情境：熟人｜网上闲聊
> 构成：原文 3 句 / 人工原话 0 句 / AI 采纳 2 句 / 未采纳 1 条
> 批注：推定 2 句

**对方**：在吗

**示例角色**：说
"""

MEM_PAGE = """---
type: person
aliases: ["示例角色"]
source: "group:123#1-9"
---

# 示例角色

## 事实

- 自称 14 岁（source: group:123#1-9）
"""

NO_HEAD = """# 2026-10-23 群里那两条还没回

**对方**：你昨天说的事我想了想

**示例角色**：嗯，你说
"""

EMPTY_BODY = """# 2026-10-26 只有头没有正文

> 用途：speech.md
> 分类：日常搭话
> 情境：熟人｜闲聊
"""

AUDIT_FAIL = """# 2026-10-26 没过审的卡

> 形态：反应样式
> 档位：熟人
> 刺激：被试探
> 场景：被绕弯问话
> 回应：反问
> 审：未过
"""


def build(root: Path) -> Path:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "identity.md").write_text("# identity — 身份母本\n\n## 一句话是谁\n\n她。\n", encoding="utf-8")
    (raw / "2026-10-27_反应_被关心.md").write_text(REACTION, encoding="utf-8")
    (raw / "2026-10-24_替人背锅.md").write_text(DIALOG, encoding="utf-8")
    (raw / "2026-10-25_推算稿.md").write_text(DIALOG_AI_ONLY, encoding="utf-8")
    (raw / "2026-10-27_归档误抽的自我描述.md").write_text(MEM_PAGE, encoding="utf-8")
    (raw / "2026-10-23_群里那两条还没回.md").write_text(NO_HEAD, encoding="utf-8")
    (raw / "2026-10-26_只有头没有正文.md").write_text(EMPTY_BODY, encoding="utf-8")
    (raw / "2026-10-26_没过审的卡.md").write_text(AUDIT_FAIL, encoding="utf-8")
    return root


def find(scan, name: str):
    for it in scan.items:
        if it.name == name:
            return it
    return None


# ---------- 一、后缀 ----------


def suite_suffix() -> None:
    print("\n[已归纳后缀]")
    check("a.md → a.已归纳.md", R.ingested_name("a.md") == "a.已归纳.md")
    check("后缀插在 .md 之前（还是 md）", R.ingested_name("x.md").endswith(".已归纳.md"))
    check("已带后缀不重复加", R.ingested_name("a.已归纳.md") == "a.已归纳.md")
    check("不出双后缀", "已归纳.已归纳" not in R.ingested_name(R.ingested_name("a.md")))
    check("认得出带后缀的文件", R.is_ingested("a.已归纳.md") and not R.is_ingested("a.md"))
    check("去后缀还原原名", R.plain_name("a.已归纳.md") == "a.md")
    check("stem 去掉扩展名", R.plain_stem("2026-10-27_反应.已归纳.md") == "2026-10-27_反应")


# ---------- 二、成品头 ----------


def suite_head() -> None:
    print("\n[成品头解析]")
    h = R.parse_head(REACTION)
    check("反应卡读出 6 个以上字段", len(h) >= 6, str(len(h)))
    check("形态=反应样式", h.get("形态") == "反应样式")
    check("档位=拥有者", h.get("档位") == "拥有者")
    check("刺激=被关心", h.get("刺激") == "被关心")

    d = R.parse_head(DIALOG)
    check("对话稿读出分类", d.get("分类") == "边界触碰")
    check("引用块之外的正文不算头", "对方**" not in str(d))

    check("反应卡 kind", R.kind_of("x.md", h) == R.KIND_REACTION)
    check("对话稿 kind", R.kind_of("x.md", d) == R.KIND_DIALOG)
    check("母本 kind", R.kind_of("identity.md", {}) == R.KIND_IDENTITY)
    check("无头素材默认对话稿", R.kind_of("x.md", {}) == R.KIND_DIALOG)

    check("反应卡档位", R.level_of(h) == "拥有者")
    check("对话稿档位取情境首段", R.level_of(d) == "陌生人")
    check("没分档是 —（不是缺字段）", R.level_of({}) == R.LEVEL_NONE)
    check("档位写 — 时回落到情境", R.level_of({"档位": "—", "情境": "熟人｜闲聊"}) == "熟人")


def suite_cap() -> None:
    print("\n[置信上限]")
    check("人工原话可到 high", R.cap_for({"构成": "人工原话 2 句 / AI 采纳 0 句"}) == R.CONF_HIGH)
    check("只有 AI 采纳 → medium", R.cap_for({"构成": "AI 采纳 3 句"}) == R.CONF_MEDIUM)
    check("小说原文 → medium（不是 high）", R.cap_for({"构成": "原文 3 句 / 人工原话 0 句 / AI 采纳 2 句"}) == R.CONF_MEDIUM)
    check("反应样式稿恒 medium", R.cap_for({"形态": "反应样式", "构成": "人工原话 2 句"}) == R.CONF_MEDIUM)
    check("推定 → low", R.cap_for({"构成": "人工原话 2 句", "批注": "推定 2 句"}) == R.CONF_LOW)
    check("模型自报 high 压到素材天花板", R.clamp("high", {"形态": "反应样式"}) == R.CONF_MEDIUM)
    check("low 不会被抬高", R.clamp("low", {"构成": "人工原话 2 句"}) == R.CONF_LOW)
    check("禁页表：反应卡不许进身份/边界",
          set(R.FORBIDDEN_PAGES[R.KIND_REACTION]) == {"identity.md", "boundaries.md"})
    check("补丁动作里没有 delete", "delete" not in R.OPS and "replace" not in R.OPS)


# ---------- 三、扫料判定 ----------


def suite_scan() -> None:
    print("\n[扫料判定]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ingest-"))
    try:
        root = build(tmp / "库A")
        s = ingest.scan(root)

        check("扫到 8 份素材", len(s.items) == 8, str(len(s.items)))
        check("母本不进队列", find(s, "identity.md").status == ingest.STATUS_DONE)
        check("母本标 via=identity", find(s, "identity.md").via == "identity")

        card = find(s, "2026-10-27_反应_被关心.md")
        check("反应卡待收", card.status == ingest.STATUS_NEW, card.status)
        check("反应卡类型", card.kind == R.KIND_REACTION)
        check("反应卡有档位", card.level == "拥有者")
        check("反应卡上限 medium", card.cap == R.CONF_MEDIUM)

        dlg = find(s, "2026-10-24_替人背锅.md")
        check("对话稿待收", dlg.status == ingest.STATUS_NEW)
        check("对话稿可到 high", dlg.cap == R.CONF_HIGH)

        ai = find(s, "2026-10-25_推算稿.md")
        check("只有 AI 采纳的稿上限 low（批注里有推定）", ai.cap == R.CONF_LOW)
        check("但它仍然可以收", ai.status == ingest.STATUS_NEW)

        mem = find(s, "2026-10-27_归档误抽的自我描述.md")
        check("记忆层页被拦住", mem.status == ingest.STATUS_BLOCKED)
        check("拦住理由说得清", "memory/" in mem.reason, mem.reason)

        nh = find(s, "2026-10-23_群里那两条还没回.md")
        check("无头但有对白的素材能收", nh.status == ingest.STATUS_NEW)
        check("无头素材档位是 —", nh.level == R.LEVEL_NONE)

        eb = find(s, "2026-10-26_只有头没有正文.md")
        check("只有头没有正文 → 拦住", eb.status == ingest.STATUS_BLOCKED)

        af = find(s, "2026-10-26_没过审的卡.md")
        check("审：未过 → 拦住", af.status == ingest.STATUS_BLOCKED)
        check("拦住理由是没过审", "未过" in af.reason, af.reason)

        check("pending 只算 new", len(s.pending()) == 4, str(len(s.pending())))
        check("counts 合计等于素材数", sum(s.counts.values()) == len(s.items))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_no_side_effect() -> None:
    print("\n[扫料零副作用]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ingest-"))
    try:
        root = build(tmp / "库A")
        before = sorted(p.name for p in (root / "raw").iterdir())
        s1 = ingest.scan(root)
        s2 = ingest.scan(root)
        after = sorted(p.name for p in (root / "raw").iterdir())
        check("文件名一个没变", before == after)
        check("两次结果一致", s1.counts == s2.counts and len(s1.items) == len(s2.items))
        check("不写状态文件", not (root / "data" / "ingest_state.json").exists())
        check("to_dict 不带正文", all("text" not in it for it in s1.to_dict()["items"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- 四、log 反查 ----------


def suite_log() -> None:
    print("\n[log 反查]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ingest-"))
    try:
        root = build(tmp / "库A")
        (root / "log.md").write_text(
            "# log.md\n\n"
            "## 建库\n\n- 2026-10-26 建库 | agents.md | 空骨架\n\n"
            "## [2026-10-24] ingest | raw/2026-10-24_替人背锅.md\n- 并入 boundaries.md：1 条\n\n"
            "## [2026-10-24] ingest | 第一卷1章·……唔～……\n\n"
            "## [2026-10-25] ingest | 反应样式稿五份（第一卷 2 章 / 4 章）\n",
            encoding="utf-8",
        )
        stems, days = ingest.log_groups(root)
        check("带路径的 digest 能认", "2026-10-24_替人背锅" in stems)
        check("缩短名也在里面", "第一卷1章·……唔～……" in stems)
        check("记到整批收过的日期", days == {"2026-10-25"}, str(days))

        s = ingest.scan(root)
        check("精确命中 → 已收", find(s, "2026-10-24_替人背锅.md").status == ingest.STATUS_DONE)
        check("命中依据是 log", find(s, "2026-10-24_替人背锅.md").via == "log")

        (root / "raw" / "2026-10-24_第一卷1章转生公主·……唔～…….md").write_text(
            "# 对话\n\n**对方**：你在干什么\n\n**示例角色**：唔～\n", encoding="utf-8"
        )
        s = ingest.scan(root)
        short = find(s, "2026-10-24_第一卷1章转生公主·……唔～…….md")
        check("缩短名对得上全名 → 已收", short.status == ingest.STATUS_DONE, short.status)
        check("并说明 log 记的是缩短名", "缩短名" in short.reason, short.reason)

        # 同一锚命中多份就不认：宁可多收一遍，也不能漏收
        (root / "raw" / "2026-10-26_另一份……唔～…….md").write_text(
            "# 对话\n\n**对方**：嗯\n\n**示例角色**：嗯\n", encoding="utf-8"
        )
        s = ingest.scan(root)
        amb = find(s, "2026-10-26_另一份……唔～…….md")
        check("锚命中多份时不认（保守）", amb.status == ingest.STATUS_NEW, amb.status)

        # 整批反查只对反应卡生效
        (root / "raw" / "2026-10-25_反应_被轻视_某章.md").write_text(REACTION, encoding="utf-8")
        (root / "raw" / "2026-10-25_普通对话稿.md").write_text(DIALOG, encoding="utf-8")
        s = ingest.scan(root)
        check("同日反应卡按整批算已收",
              find(s, "2026-10-25_反应_被轻视_某章.md").via == "log-group")
        check("同日非反应卡不吃整批这条",
              find(s, "2026-10-25_普通对话稿.md").status == ingest.STATUS_NEW)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- 五、后缀与水位 ----------


def suite_done_by_suffix() -> None:
    print("\n[后缀与水位的互为备份]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ingest-"))
    try:
        root = build(tmp / "库A")
        raw = root / "raw"
        (raw / "2026-10-27_反应_被关心.md").rename(raw / "2026-10-27_反应_被关心.已归纳.md")
        s = ingest.scan(root)
        got = find(s, "2026-10-27_反应_被关心.已归纳.md")
        check("带后缀的直接算已收", got.status == ingest.STATUS_DONE)
        check("依据是后缀", got.via == "suffix")
        check("stem 与未加后缀时一致", got.stem == "2026-10-27_反应_被关心")

        # 水位：内容变了 → changed
        st = ingest.read_state(root)
        st["2026-10-27_反应_被关心"] = {"sha": "deadbeef", "at": "2026-10-27 21:00"}
        ingest.write_state(root, st)
        s = ingest.scan(root)
        got = find(s, "2026-10-27_反应_被关心.已归纳.md")
        check("内容改过 → changed", got.status == ingest.STATUS_CHANGED, got.status)
        check("并说清是内容改过", "改过" in got.reason, got.reason)

        # 后缀被手删了，水位还在 → 仍然认得出
        (raw / "2026-10-27_反应_被关心.已归纳.md").rename(raw / "2026-10-27_反应_被关心.md")
        stamp = ingest.scan(root)
        got = find(stamp, "2026-10-27_反应_被关心.md")
        cur = got.sha
        ingest.write_state(root, {"2026-10-27_反应_被关心": {"sha": cur, "at": "2026-10-27 21:00"}})
        s = ingest.scan(root)
        got = find(s, "2026-10-27_反应_被关心.md")
        check("后缀删了靠水位认出来", got.status == ingest.STATUS_DONE and got.via == "state")
        check("水位不会被扫料悄悄改写",
              ingest.read_state(root)["2026-10-27_反应_被关心"]["sha"] == cur)

        state = ingest.state_for_commit(s, s.pending())
        check("commit 水位保留旧字段", "sha" in state["2026-10-27_反应_被关心"])
        check("commit 水位带上时间", bool(state["2026-10-27_反应_被关心"]["at"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_state_io() -> None:
    print("\n[水位读写]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ingest-"))
    try:
        root = tmp / "库A"
        root.mkdir(parents=True)
        check("没有状态文件返回空", ingest.read_state(root) == {})
        ingest.write_state(root, {"a": {"sha": "1"}})
        check("写进去读得出来", ingest.read_state(root)["a"]["sha"] == "1")
        (root / "data" / "ingest_state.json").write_text("{坏掉的 json", encoding="utf-8")
        check("坏文件不炸，当空处理", ingest.read_state(root) == {})
        (root / "data" / "ingest_state.json").write_text(json.dumps(["数组"]), encoding="utf-8")
        check("结构不对也当空处理", ingest.read_state(root) == {})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- 六、补丁与闸 ----------

QUOTE = "对方把她手里的事接过去让她歇着"
"""这段能在 REACTION 的成品头里逐字找到（`场景：` 那一行）。用来做「证据回查」的正面样本。"""

GOOD_ITEMS = [
    {
        "op": "append", "page": "reactions.md", "section": "高频情境反应",
        "title": "情境：被关心 / 对方替她把事接走",
        "text": "- 条件：对方不等她开口就把事接走\n- 回应：先愣一下，再一句「我自己来」\n- 不做：不说谢，不解释为什么不用",
        "quote": QUOTE, "confidence": "high", "why": "被关心这一档的回应序列",
    },
    {
        "op": "append", "page": "speech.md", "section": "口头禅 / 语气词",
        "text": "- 能一个字说完的不说两个字",
        "quote": "先推掉，被坚持才接受", "confidence": "medium", "why": "句长的直接证据",
    },
    {
        "op": "append", "page": "relationships.md", "section": "与拥有者",
        "text": "- 称呼：不喊，直接说事\n- 直接度：高，不铺垫",
        "quote": QUOTE, "confidence": "medium", "why": "拥有者档的旋钮",
    },
]


def _reply(items=None, note: str = "") -> str:
    return json.dumps(
        {"items": GOOD_ITEMS if items is None else items, "note": note},
        ensure_ascii=False,
    )


class FakeLLM:
    """假模型：只回一段固定 JSON。`ingest_one` 只用得到 `chat()`。"""

    def __init__(self, reply) -> None:
        self.reply = reply
        self.cfg = SimpleNamespace(model="fake-chat", ingest_model="", ingest_max_tokens=2400)
        self.ready = True
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **kw):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _make_lib(tmp: Path) -> Path:
    """按真库路径建一套骨架（页、节、frontmatter 都是模板里的），再丢一份反应卡。"""
    root = tmp / "甲"
    library.create_persona(root, name="甲")
    (root / "raw" / "2026-10-27_反应_被关心.md").write_text(REACTION, encoding="utf-8")
    return root


def _asset(root: Path, name: str = "2026-10-27_反应_被关心.md"):
    for it in ingest.scan(root).items:
        if it.name == name:
            return it
    raise AssertionError(f"扫不到素材 {name}")


def _page(root: Path, page: str) -> str:
    return (root / "wiki" / page).read_text(encoding="utf-8")


def _kept(before: str, after: str) -> bool:
    """原文每一行都还在 —— 「只增不改」的机器可验版。"""
    tail = [ln.strip() for ln in after.splitlines() if ln.strip()]
    return all(ln.strip() in tail for ln in before.splitlines() if ln.strip())


def suite_patch() -> None:
    print("\n[补丁解析]")
    items, note = ingest.parse_patch(_reply())
    check("解析出 3 条", len(items) == 3, str(len(items)))
    check("备留守空", note == "")
    check("section 补上 ## 前缀", items[0].section == "## 高频情境反应", items[0].section)
    check("op 认出来", items[0].op == "append")
    check("title 保留", items[0].title.startswith("情境："))

    fenced = "```json\n" + _reply() + "\n```"
    check("包了代码块也认", len(ingest.parse_patch(fenced)[0]) == 3)
    check("前后有废话也认", len(ingest.parse_patch("好，我看看：\n" + _reply() + "\n以上。")[0]) == 3)
    check("不是 JSON → 空", ingest.parse_patch("我收不出东西")[0] == [])
    check("坏 JSON → 空", ingest.parse_patch('{"items": [') [0] == [])
    check("非字典条目被丢掉", len(ingest.parse_patch('{"items": ["x", {"page": "speech.md"}]}')[0]) == 1)
    check("op 缺省 append", ingest.parse_patch('{"items": [{"page": "speech.md"}]}')[0][0].op == "append")
    check("note 读出来", ingest.parse_patch(_reply([], "整份都是闲聊"))[1] == "整份都是闲聊")


def suite_quote() -> None:
    print("\n[证据回查]")
    src = REACTION
    check("逐字命中", ingest.quote_ok(QUOTE, src))
    check("多空格也命中", ingest.quote_ok("对方把她手里 的事接过去\n让她歇着", src))
    check("抄错一两个字仍过（切片兜底）", ingest.quote_ok("对方把她手里的事情接过去让她歇着", src))
    check("编的话拦下", not ingest.quote_ok("她冷冷地瞥了一眼没有说话", src))
    check("太短不算证据", not ingest.quote_ok("被关心", src))
    check("空 quote 拦下", not ingest.quote_ok("", src))
    check("成品头短字段放行", ingest.quote_ok("话量：少（一句一条）", src))
    check("头字段改写就拦下", not ingest.quote_ok("话量：一句一条都没有", src))
    check("头字段前带 > 也认", ingest.quote_ok("> 刺激：被关心", src))


def suite_gate() -> None:
    print("\n[三道闸]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-gate-"))
    try:
        root = _make_lib(tmp)
        asset = _asset(root)
        pages = ingest.read_pages(root)

        def g(**kw):
            patch = ingest.Patch(**kw)
            patch.source = asset.plain
            return ingest.gate(patch, asset, pages)

        base = dict(op="append", page="reactions.md", section="## 高频情境反应",
                    text="- 回应：先愣一下", quote=QUOTE, confidence="medium")
        check("合法补丁放行", g(**base) == "", g(**base))

        check("反应卡不许进 identity",
              "不许写" in g(**{**base, "page": "identity.md"}))
        check("反应卡没勾范例不许进 exemplars",
              "已核" in g(**{**base, "page": "exemplars.md", "section": "## 行为范例"}))
        check("页名不在白名单",
              "白名单" in g(**{**base, "page": "notes.md"}))
        check("未知动作拦下",
              "未知动作" in g(**{**base, "op": "delete"}))
        check("自造节拦下（非情境页）",
              "没有这一节" in g(**{**base, "page": "speech.md", "section": "## 自造的节"}))
        check("reactions 也不许自造节（只能进已有的 ## 高频情境反应）",
              "没有这一节" in g(**{**base, "section": "## 被夸奖"}))
        check("反应卡能进已有的节",
              g(**{**base, "section": "## 高频情境反应"}) == "")
        check("模板不给旧结构留后门（新建库必须与真库同构）",
              "没有这一节" in g(**{**base, "section": "## 被关心 / 被安慰"}))
        check("text 带标题行拦下",
              "标题行" in g(**{**base, "text": "## 我自己造一节"}))
        check("性格断言拦下",
              "断言" in g(**{**base, "text": "- 她总是先推掉再接受"}))
        check("称谓/专名拦下",
              "专名" in g(**{**base, "text": "- 回应：喊他老周"}))
        check("关系页只收六个旋钮",
              "六个旋钮" in g(**{**base, "page": "relationships.md",
                                 "section": "## 与拥有者", "text": "- 她跟对方很亲近"}))
        check("关系页旋钮写法放行",
              g(**{**base, "page": "relationships.md",
                   "section": "## 与拥有者", "text": "- 会不会主动追问：不追"}) == "")
        check("quote 对不上拦下",
              "找不到" in g(**{**base, "quote": "这段素材里根本没有这句话"}))
        check("text 空着拦下", "空着" in g(**{**base, "text": ""}))
        check("没给 section 拦下", "section" in g(**{**base, "section": ""}))
        check("skip 要有理由", "理由" in g(**{**base, "op": "skip", "text": "", "why": ""}))
        check("skip 带理由放行", g(**{**base, "op": "skip", "text": "", "why": "纯闲聊"}) == "")

        check("同义行判重", ingest._dup(
            "- 条件：对方不等她开口就把事接走",
            "- 条件：对方不等她开口就把事接走，直接做",
        ))
        check("不同说法不算重", not ingest._dup("- 回应：只说两个字", "- 回应：一个字都不给"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_apply() -> None:
    print("\n[合页]")
    page = """---
updated: 2026-01-01
---

# reactions

## 被关心 / 被安慰
- _待填_

## 被催 / 被追问没回
- _待填_
"""
    p = ingest.Patch(op="append", page="reactions.md", section="## 被关心 / 被安慰",
                     text="- 回应：先愣一下\n- 不做：不说谢")
    out, mode = ingest.apply_patch(page, p)
    check("动作是 append", mode == "append", mode)
    check("插进了对的节", out.index("先愣一下") < out.index("## 被催"), "顺序不对")
    check("原文一行不少", _kept(page, out))

    p2 = ingest.Patch(op="append", page="reactions.md", section="## 被关心 / 被安慰",
                      title="情境：被关心 / 替她接事", text="- 条件：对方接过去")
    out2, mode2 = ingest.apply_patch(out, p2)
    check("新子节是 new-sub", mode2 == "new-sub", mode2)
    check("子节标题写成 ###", "### 情境：被关心 / 替她接事" in out2)
    check("第二次仍只增不改", _kept(out, out2))

    out3, mode3 = ingest.apply_patch(out2, p2)
    check("同名子节并进去", mode3 == "merge", mode3)
    check("并进去是加一条，不是替换", out3.count("### 情境：被关心 / 替她接事") == 1)

    pc = ingest.Patch(op="conflict", page="reactions.md", section="## 被关心 / 被安慰",
                      text="- 回应：反过来接受", source="2026-10-27_反应.md")
    out4, _ = ingest.apply_patch(out3, pc)
    check("冲突条带待裁标记", "待人工裁" in out4 and "2026-10-27_反应.md" in out4)

    pn = ingest.Patch(op="append", page="reactions.md", section="## 被夸奖", text="- 条件：被夸")
    out5, mode5 = ingest.apply_patch(page, pn)
    check("新开一节", mode5 == "new-section" and "## 被夸奖" in out5)
    check("新开节也不动原文", _kept(page, out5))

    check("updated 刷新", "updated: 2026-10-28" in ingest._touch_updated(page, "2026-10-28"))


def suite_prompt() -> None:
    print("\n[提示词]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-prompt-"))
    try:
        root = _make_lib(tmp)
        asset = _asset(root)
        msgs = ingest_prompt.build_messages(root, asset)
        check("两条消息", len(msgs) == 2)
        check("系统提示讲了反应样式稿", "反应样式稿" in msgs[0]["content"])
        check("系统提示写了 quote 必须逐字", "逐字" in msgs[0]["content"])
        check("系统提示说明了没有 delete", "只准追加" in msgs[0]["content"])
        user = msgs[1]["content"]
        check("用户消息带素材原文", "素材原文" in user and "被关心" in user)
        check("用户消息带成品头", "刺激：被关心" in user)
        check("用户消息带可写页", "reactions.md" in user)
        check("反应卡不给身份页", "identity.md" not in user)
        check("反应卡不给梗页", "memories.md" not in user)
        check("给了现有节名（好让它挑）", "## 高频情境反应" in user)
        check("给了关系档对应那一节", "与拥有者" in user)
        check("别的关系档不串进来", "对熟人" not in user)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_commit() -> None:
    print("\n[落盘]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-commit-"))
    try:
        root = _make_lib(tmp)
        raw = root / "raw"
        asset = _asset(root)
        before = {p: _page(root, p) for p in ("reactions.md", "speech.md", "relationships.md")}
        log_before = (root / "log.md").read_text(encoding="utf-8") if (root / "log.md").is_file() else ""

        llm = FakeLLM(_reply())
        res = asyncio.run(ingest.ingest_one(llm, root, asset))
        check("三条全过闸", len(res.accepted) == 3, [p.reject for p in res.patches])
        check("没被拦", len(res.rejected) == 0, [p.reject for p in res.rejected])
        check("置信压到 medium（反应卡天花板）",
              all(p.confidence == R.CONF_MEDIUM for p in res.accepted),
              [p.confidence for p in res.accepted])
        check("送进模型的是整份规程+素材", "素材原文" in llm.calls[0][1]["content"])

        got = ingest.commit(root, [res], run_id="T1", stamp="T1-01")
        check("落盘成功", bool(got.get("ok")), str(got.get("error")))
        after = {p: _page(root, p) for p in before}
        check("新条目进了页", "我自己来" in after["reactions.md"])
        check("原文一行不少", _kept(before["reactions.md"], after["reactions.md"]))
        idx = after["reactions.md"].index
        # 新子节追加在节尾，不插队到已有情境之前
        check("落在对的节里",
              idx("## 高频情境反应") < idx("### 情境：被关心 / 被安慰") < idx("我自己来"),
              "位置不对")
        check("第二页也写进去了", "能一个字说完" in after["speech.md"])
        check("关系页按旋钮写", "不喊，直接说事" in after["relationships.md"])

        log_after = (root / "log.md").read_text(encoding="utf-8")
        check("log 追加了 ingest 行", "ingest | 2026-10-27_反应_被关心" in log_after)
        check("log 旧内容保留", log_after.startswith(log_before[:40]))
        check("log 记了改哪页", "reactions.md" in log_after.split("ingest")[-1])

        renamed = raw / "2026-10-27_反应_被关心.已归纳.md"
        check("素材改名加后缀", renamed.is_file())
        check("原名不见了", not (raw / "2026-10-27_反应_被关心.md").exists())
        check("素材内容一个字节没变", renamed.read_text(encoding="utf-8") == REACTION)

        state = ingest.read_state(root)
        check("水位推进", state.get("2026-10-27_反应_被关心", {}).get("sha") == asset.sha)
        check("水位记了页", set(state["2026-10-27_反应_被关心"]["pages"]) == set(before))

        hist = root / "data" / "wiki_history" / "T1-01"
        check("有备份目录", hist.is_dir())
        check("备份里是原件",
              (hist / "files" / "wiki" / "reactions.md").read_text(encoding="utf-8") == before["reactions.md"])
        check("meta 记了改名", any("已归纳" in b for a, b in
                                 json.loads((hist / "meta.json").read_text(encoding="utf-8"))["renamed"]))
        check("历史列表能列出来", any(h["stamp"] == "T1-01" for h in ingest.history(root)))

        s = ingest.scan(root)
        check("收完不再排队", len(s.pending()) == 0, str([i.name for i in s.pending()]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_undo() -> None:
    print("\n[撤回]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-undo-"))
    try:
        root = _make_lib(tmp)
        raw = root / "raw"
        asset = _asset(root)
        before = {p: _page(root, p) for p in ("reactions.md", "speech.md", "relationships.md")}
        log_before = (root / "log.md").read_text(encoding="utf-8") if (root / "log.md").is_file() else ""

        res = asyncio.run(ingest.ingest_one(FakeLLM(_reply()), root, asset))
        ingest.commit(root, [res], run_id="T2", stamp="T2-01")
        got = ingest.undo(root, "T2")
        check("撤回成功", bool(got.get("ok")), str(got.get("error")))
        check("三个页字节级还原",
              all(_page(root, p) == before[p] for p in before),
              "页没还原")
        check("log 也还原", (root / "log.md").read_text(encoding="utf-8") == log_before)
        check("文件名改回去了",
              (raw / "2026-10-27_反应_被关心.md").is_file()
              and not (raw / "2026-10-27_反应_被关心.已归纳.md").exists())
        check("水位退回空", ingest.read_state(root) == {})
        check("备份留着没删", (root / "data" / "wiki_history" / "T2-01").is_dir())
        check("同一轮撤两次会拒", ingest.undo(root, "T2").get("ok") is False)
        check("撤完素材重新排队", len(ingest.scan(root).pending()) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_run_dry() -> None:
    print("\n[跑一轮]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-run-"))
    try:
        root = _make_lib(tmp)
        asset = _asset(root)
        before = _page(root, "reactions.md")
        llm = FakeLLM(_reply())
        out = asyncio.run(ingest.run(root, llm, dry_run=True, run_id="DRY"))
        check("跑完不报错", bool(out.get("ok")), str(out.get("error")))
        check("队列 1 份", out.get("total") == 1, str(out.get("total")))
        check("dry-run 不写页", _page(root, "reactions.md") == before)
        check("dry-run 不改名", (root / "raw" / "2026-10-27_反应_被关心.md").is_file())
        check("跑完释放锁", ingest.running() is False)
        job = ingest.job()
        check("过程有日志", len(job["lines"]) >= 2, str(len(job["lines"])))
        check("日志带每份结果", any("收 3 条" in ln["text"] for ln in job["lines"]))
        check("结果留在 job 里", len(job["results"]) == 1)

        out2 = asyncio.run(ingest.run(root, llm, run_id="REAL"))
        check("真跑落盘", out2.get("accepted") == 3, str(out2))
        check("真跑改了页", "我自己来" in _page(root, "reactions.md"))
        check("真跑改了名", (root / "raw" / "2026-10-27_反应_被关心.已归纳.md").is_file())

        out3 = asyncio.run(ingest.run(root, FakeLLM(_reply()), run_id="AGAIN"))
        check("没料了立刻返回", out3.get("total") == 0, str(out3))

        # 全判重复（accepted 空）也必须落一次盘：改名 + 推水位，
        # 否则下次点按钮它又算「新的」，每次白烧一次模型调用。
        root2 = _make_lib(tmp / "b")
        out4 = asyncio.run(ingest.run(root2, FakeLLM(_reply([], "三个落点都在页里")), run_id="SKIP"))
        check("全判重复也算跑完", bool(out4.get("ok")) and out4.get("accepted") == 0, str(out4))
        check("全判重复也改名", (root2 / "raw" / "2026-10-27_反应_被关心.已归纳.md").is_file())
        st = ingest.read_state(root2)
        check("全判重复也推水位", any(v.get("empty") for v in st.values()), str(list(st)))
        check("全判重复也留备份",
              any((d / "meta.json").is_file() for d in (root2 / "data" / "wiki_history").iterdir()))
        out5 = asyncio.run(ingest.run(root2, FakeLLM(_reply()), run_id="SKIP2"))
        check("再点不再排队", out5.get("total") == 0, str(out5))

        bad = asyncio.run(ingest.ingest_one(FakeLLM(LLMError("HTTP 500")), root, asset))
        check("模型失败不标收过", bad.mark_done is False and "HTTP" in bad.error)
        junk = asyncio.run(ingest.ingest_one(FakeLLM("我想不出来"), root, asset))
        check("吐不出 JSON 也算失败", junk.mark_done is False and bool(junk.error))
        nothing = asyncio.run(ingest.ingest_one(FakeLLM(_reply([], "整份都是闲聊")), root, asset))
        check("说清收不出东西也算收过", nothing.mark_done is True and nothing.note == "整份都是闲聊")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_run_switches() -> None:
    print("\n[库内开关]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-sw-"))
    try:
        # auto_index=False：页照收，总括段一个字不动
        root = _make_lib(tmp / "a")
        index_before = (root / "index.md").read_text(encoding="utf-8")
        out = asyncio.run(
            ingest.run(root, FakeLLM(_reply()), run_id="NOIX", auto_index=False)
        )
        check("关掉总括段也照常收", out.get("accepted") == 3, str(out))
        check("index.md 一个字节没动",
              (root / "index.md").read_text(encoding="utf-8") == index_before)
        check("index 结果留空", out.get("index") == {}, str(out.get("index")))
        texts = [ln["text"] for ln in ingest.job()["lines"]]
        check("日志说清为什么跳过", any("总括段跳过" in t for t in texts), str(texts[-5:]))
        check("汇总行带总括段跳过",
              any(t.startswith("完成") and "总括段跳过" in t for t in texts), str(texts[-3:]))

        # limit：只收 N 份，剩下的留在队列，下次点按钮接着收
        root2 = tmp / "b"
        library.create_persona(root2, name="乙")
        (root2 / "raw" / "2026-10-27_反应_被关心.md").write_text(REACTION, encoding="utf-8")
        (root2 / "raw" / "2026-10-28_反应_被指责.md").write_text(
            REACTION.replace("2026-10-27", "2026-10-28").replace("被关心", "被指责"),
            encoding="utf-8",
        )
        check("两份排队", len(ingest.scan(root2).pending()) == 2)
        out2 = asyncio.run(ingest.run(root2, FakeLLM(_reply()), run_id="LIM", limit=1))
        check("只收一份", out2.get("total") == 1, str(out2.get("total")))
        check("收的是第一份", (root2 / "raw" / "2026-10-27_反应_被关心.已归纳.md").is_file())
        check("剩一份还在队列", len(ingest.scan(root2).pending()) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 60)
    print("一键归纳 · 扫料层 + 补丁 + 落盘")
    print("=" * 60)
    suite_suffix()
    suite_head()
    suite_cap()
    suite_scan()
    suite_no_side_effect()
    suite_log()
    suite_done_by_suffix()
    suite_state_io()
    suite_patch()
    suite_quote()
    suite_gate()
    suite_apply()
    suite_prompt()
    suite_commit()
    suite_undo()
    suite_run_dry()
    suite_run_switches()
    print()
    print(f"通过 {PASS} · 失败 {FAIL}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

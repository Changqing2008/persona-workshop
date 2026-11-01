"""index.md 总括段重建测试。

这块的风险不在"生成的句子难不难看"，在两条硬规矩上：
- 人工写的行被程序冲掉 → 你手写的判断悄悄消失，且看不出来
- 该更新的不更新 → 收进来的特质聊天时看不见，收了等于没收

所以每条规矩都钉一个测试，外加幂等、备份、撤回。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from workshop import indexer, ingest, library  # noqa: E402

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


def _lib(tmp: Path) -> Path:
    root = tmp / "甲"
    library.create_persona(root, name="甲")
    return root


def _index(root: Path) -> str:
    return (root / "index.md").read_text(encoding="utf-8")


def _write(root: Path, rel: str, text: str) -> None:
    (root / rel).write_text(text, encoding="utf-8")


def _fill_section(root: Path, rel: str, keyword: str, body: str) -> None:
    """把某页某个 ## 节的正文换成给定内容（模板里的占位行去掉）。"""
    p = root / rel
    lines = p.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        out.append(lines[i])
        if lines[i].startswith("## ") and keyword in lines[i]:
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            out.append("")
            out.extend(body.splitlines())
            out.append("")
            continue
        i += 1
    p.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")


def _section(root: Path, title: str) -> str:
    """取总括段某小节的正文（不含标题行）。"""
    lines = _index(root).splitlines()
    for i, line in enumerate(lines):
        if line.startswith("### ") and title in line:
            buf = []
            for ln in lines[i + 1 :]:
                if ln.startswith("## ") or ln.startswith("### "):
                    break
                buf.append(ln)
            return "\n".join(buf).strip()
    raise AssertionError(f"找不到小节 {title}")


# ---------- 纯函数 ----------


def suite_merge() -> None:
    print("\n[合并规则]")
    cur = ["- 人工A：一直在这儿", "- 程序B：上次写的"]
    cands = ["- 程序B：上次写的", "- 程序C：页里新来的"]
    new, added, dropped = indexer.merge(cur, cands, ["- 程序B：上次写的"], 7)
    check("人工行留下", "- 人工A：一直在这儿" in new)
    check("来源还在的程序行留下", "- 程序B：上次写的" in new)
    check("新条追加进来", "- 程序C：页里新来的" in new)
    check("没有淘汰", dropped == [], str(dropped))

    new2, _added2, dropped2 = indexer.merge(
        ["- 人工A：一直在这儿", "- 程序B：页里删了它"], ["- 人工A：一直在这儿"], ["- 程序B：页里删了它"], 7
    )
    check("源没了的程序行被淘汰", "- 程序B：页里删了它" not in new2)
    check("淘汰有记账", dropped2 == ["- 程序B：页里删了它"], str(dropped2))
    check("人工行不受淘汰影响", "- 人工A：一直在这儿" in new2)

    new3, added3, _ = indexer.merge(["- 人工A：一", "- 人工B：二"], ["- 程序C：三"], [], 2)
    check("满额不再追加", added3 == [] and new3 == ["- 人工A：一", "- 人工B：二"], str(new3))

    new4, added4, _ = indexer.merge([], ["- 重复的：同一句话的意思", "- 重复的：同一句话的意思"], [], 7)
    check("同义只留一条", len(new4) == 1 and added4 == new4, str(new4))

    check("占位行被换掉", "- _待填_（性格底色）" not in indexer.merge(["- _待填_（性格底色）"], ["- 新的：内容"], [], 7)[0])


def suite_extract() -> None:
    print("\n[从 wiki 抽句子]")
    traits = """---
type: trait
---
# traits

## 稳定主特质（长期稳定的性格底色）

### 自持（强度：中；原「傲」）
- 定义：有底气、不轻贱自己；但不争面子、不顶嘴。
- 证据摘录：自称「本小姐」（raw/identity.md）

## 待证观察
### 还没证据的
- 定义：这行不该进总括段
"""
    got = indexer._from_traits(traits)  # noqa: SLF001
    check("### 名 + 定义 拼成一条", got and got[0].startswith("- 自持："), str(got))
    check("括号注释被去掉", "强度" not in " ".join(got), str(got))
    check("待证观察不收", not any("不该进" in g for g in got), str(got))
    check("证据行不收", not any("本小姐" in g for g in got), str(got))

    speech = """# speech
## 口头禅 / 惯用词
- 惊讶：！
- 疑问：？
## 消息体量
- 一条为主，偶尔补一句，想到哪发到哪，不攒成一段再发。
## 句式节奏
- 先反应，后信息：被问、被夸，第一下先把反应打出来。
"""
    got2 = indexer._from_speech(speech)  # noqa: SLF001
    check("术语表式短行被挡掉", not any("惊讶" in g for g in got2), str(got2))
    check("整句的收到", len(got2) >= 2, str(got2))

    ident = """# identity
## 名字 / 自称
- 对外叫：「小小白」
- 自称：本小姐
## 设定背景
- 女生，14 岁。ChatGPT 这个模型的娘化。
"""
    got3 = indexer._from_identity(ident)  # noqa: SLF001
    check("一句话人设拿名字当引子", got3 and got3[0].startswith("「小小白」——"), str(got3))
    check("不拿「外观」当背景", got3 and "外观" not in got3[0], str(got3))

    empty = """# traits
## 稳定主特质
### 空
- _待填_（特质 → 证据摘录 → source）
"""
    check("全占位的页抽不出东西", indexer._from_traits(empty) == [], str(indexer._from_traits(empty)))  # noqa: SLF001


# ---------- 真库骨架上的行为 ----------


def suite_empty_library() -> None:
    print("\n[空库：不写空话]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ix-"))
    try:
        root = _lib(tmp)
        before = _index(root)
        out = indexer.rebuild(root, run_id="E1")
        check("跑完不报错", bool(out.get("ok")), str(out))
        check("空 wiki 不写盘", out.get("changed") is False, str(out.get("changed")))
        check("index.md 一个字节没动", _index(root) == before)
        check("原因说清了", "还没有可用内容" in (out.get("reason") or ""), str(out.get("reason")))
        check("没留备份目录", not (root / "data" / "wiki_history").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_fill() -> None:
    print("\n[有料就填]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ix-"))
    try:
        root = _lib(tmp)
        _fill_section(
            root,
            "wiki/traits.md",
            "底色",
            "### 自持\n- 定义：有底气、不轻贱自己；但不争面子、不顶嘴。\n\n"
            "### 冷\n- 定义：默认档就是省字，没关系的人只分到最少的字。",
        )
        _fill_section(
            root,
            "wiki/boundaries.md",
            "不做的",
            "- 不替别人认不属于自己的账。被推锅时直接拒，不协商。",
        )
        before = _index(root)
        out = indexer.rebuild(root, run_id="F1")
        check("真的写了", out.get("changed") is True, str(out))
        traits = _section(root, "核心特质")
        check("核心特质填了自持", "- 自持：" in traits, traits)
        check("核心特质填了冷", "- 冷：" in traits, traits)
        check("底线填了一条", "不替别人认" in _section(root, "底线"), _section(root, "底线"))
        check("占位行没了", "_待填_" not in traits, traits)
        check("小结里报了改哪几节", "核心特质 +" in (out.get("summary") or ""), str(out.get("summary")))

        after = _index(root)
        check("节标题与路由段原样保留", "## 二、路由段" in after and "| 情境 | 去读 |" in after)
        check("小节外没被重排", after.split("## 二、路由段")[1] == before.split("## 二、路由段")[1])

        logs = (root / "log.md").read_text(encoding="utf-8")
        check("log 追加了一条 index 记录", "] index | 重建总括段" in logs)

        out2 = indexer.rebuild(root, run_id="F1")
        check("再跑一次幂等", out2.get("changed") is False, str(out2.get("reason")))
        check("第二次不动文件", _index(root) == after)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_human_lines() -> None:
    print("\n[人工写的行不许动]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ix-"))
    try:
        root = _lib(tmp)
        _fill_section(root, "wiki/traits.md", "底色", "### 自持\n- 定义：有底气、不轻贱自己。")
        text = _index(root)
        text = text.replace(
            "- _待填_（性格底色，用形容词 + 一句限定，不写完整人设段落）",
            "- 我手写的：这条谁都不许动",
        )
        _write(root, "index.md", text)

        out = indexer.rebuild(root, run_id="H1")
        traits = _section(root, "核心特质")
        check("人工行还在", "我手写的：这条谁都不许动" in traits, traits)
        check("新条追加在人工行后面", traits.index("我手写的") < traits.index("- 自持："), traits)
        check("这次确实改了（追加）", out.get("changed") is True, str(out))

        # 人工段落（不是清单）→ 整节不碰
        text = _index(root)
        text = text.replace(
            "- 我手写的：这条谁都不许动",
            "这段是我自己写的一段话，不是清单，程序别碰它。",
        )
        _write(root, "index.md", text)
        guarded = _section(root, "核心特质")
        out2 = indexer.rebuild(root, run_id="H2")
        check("有人工段落时整节原样不动", _section(root, "核心特质") == guarded, _section(root, "核心特质"))
        check("跳过原因写进 notes", any("人工段落" in n for n in (out2.get("notes") or [])), str(out2.get("notes")))

        # 一句话人设：已有文字就不覆盖
        text = _index(root)
        text = text.replace("_待填_", "「我自己起的」——人工写的整段人设。")
        _write(root, "index.md", text)
        _fill_section(root, "wiki/identity.md", "名字", "- 对外叫：「模型起的」")
        indexer.rebuild(root, run_id="H3")
        check(
            "一句话人设不被覆盖",
            "「我自己起的」" in _section(root, "一句话人设"),
            _section(root, "一句话人设"),
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_undo() -> None:
    print("\n[备份与撤回]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ix-"))
    try:
        root = _lib(tmp)
        _fill_section(root, "wiki/traits.md", "底色", "### 自持\n- 定义：有底气、不轻贱自己。")
        ingest.write_state(root, {"某素材": {"sha": "abc", "empty": False}})
        before = _index(root)

        out = indexer.rebuild(root, run_id="U1", stamp="U1-ix")
        check("重建成功", out.get("changed") is True, str(out))
        check("备份目录建了", (root / "data" / "wiki_history" / "U1-ix" / "files" / "index.md").is_file())
        check("变了", _index(root) != before)

        undone = ingest.undo(root, "U1")
        check("撤回成功", bool(undone.get("ok")), str(undone))
        check("index.md 字节级还原", _index(root) == before)
        check(
            "撤完「程序写过哪些行」也清掉",
            indexer.read_state(root).get("sections") == {},
            str(indexer.read_state(root)),
        )
        check(
            "入库水位没被清空（state_existed=True 保护）",
            ingest.read_state(root).get("某素材") is not None,
            str(ingest.read_state(root)),
        )
        again = ingest.undo(root, "U1")
        check("撤两次被拒", again.get("ok") is False, str(again))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_dry_run() -> None:
    print("\n[dry-run]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ix-"))
    try:
        root = _lib(tmp)
        _fill_section(root, "wiki/traits.md", "底色", "### 自持\n- 定义：有底气、不轻贱自己。")
        before = _index(root)
        out = indexer.rebuild(root, run_id="D1", dry_run=True)
        check("dry-run 说会改", out.get("changed") is True and out.get("dry_run") is True, str(out))
        check("dry-run 不写盘", _index(root) == before)
        check("dry-run 不留备份", not (root / "data" / "wiki_history").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_no_index() -> None:
    print("\n[没有 index.md]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-ix-"))
    try:
        root = _lib(tmp)
        (root / "index.md").unlink()
        out = indexer.rebuild(root, run_id="N1")
        check("说不清就不干", out.get("ok") is False and "index.md" in str(out.get("error")), str(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 60)
    print("index.md 总括段重建")
    print("=" * 60)
    suite_merge()
    suite_extract()
    suite_empty_library()
    suite_fill()
    suite_human_lines()
    suite_undo()
    suite_dry_run()
    suite_no_index()
    print()
    total = PASS + FAIL
    if FAIL == 0:
        print(f"\033[32m通过 {PASS} · 失败 0\033[0m  （共 {total} 项）")
        return 0
    print(f"\033[31m通过 {PASS} · 失败 {FAIL}\033[0m  （共 {total} 项）")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""体检（确定性 lint）测试。

这道闸的价值全在"报了就该改、不报就是没问题"。所以测试分两半：
- 该报的必须报（性格断言、反应卡越界、页缺失、总括段缺节）
- 不该报的一条都不许报（三种卡形态、安全节不挂 raw、健康的库应当 0 error）
误报比漏报更伤 —— 误报多了人就学会无视它，闸就白设了。
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from workshop import library, lint  # noqa: E402

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


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _codes(result: dict, level: str = "") -> list[str]:
    return [i["code"] for i in result["items"] if not level or i["level"] == level]


# ---------- 单条检查 ----------


def suite_judgement() -> None:
    print("\n[性格断言]")
    hits = lint.check_judgement("traits.md", "# t\n\n- 她总是先顶一句\n- 她一向不认输\n")
    check("断言被报", len(hits) == 2, str(len(hits)))
    check("级别是 error", all(h.level == "error" for h in hits))
    check("原句带出来", "她总是" in hits[0].text, hits[0].text)
    check("说清怎么改", "条件句" in hits[0].why, hits[0].why)
    ok = lint.check_judgement("traits.md", "# t\n\n- 被质疑时先顶一句，再认账\n")
    check("条件句不报", ok == [], str(ok))
    quote = lint.check_judgement("values.md", "# v\n\n> 她从来不说软话（这句是引用说明）\n")
    check("引用块不报", quote == [], str(quote))


def suite_card_scope() -> None:
    print("\n[反应卡越界]")
    bad = lint.check_card_scope("identity.md", "# i\n\n- 场景：被关心\n- 形态：反应样式\n")
    check("identity 里出现反应卡 → error", len(bad) == 1 and bad[0].level == "error", str(bad))
    ok = lint.check_card_scope("reactions.md", "# r\n\n- 形态：反应样式\n")
    check("reactions 里允许", ok == [], str(ok))
    ok2 = lint.check_card_scope("speech.md", "# s\n\n- 来源：反应样式稿 3 份\n")
    check("speech 里允许", ok2 == [], str(ok2))
    bad2 = lint.check_card_scope("memories.md", "# m\n\n- 来源：反应\n")
    check("memories 里出现 → error", len(bad2) == 1, str(bad2))


def suite_structure() -> None:
    print("\n[结构]")
    ph = lint.check_placeholder("speech.md", "# s\n\n- _待填_\n- _待填_（句长）\n")
    check("占位符报一条 warn", len(ph) == 1 and ph[0].level == "warn", str(ph))
    check("占位符说清后果", "剔掉" in ph[0].why, ph[0].why)

    empty = lint.check_empty_sections("traits.md", "# t\n\n## 有货\n- 一条\n\n## 空的\n\n## 也有货\n- 两条\n")
    check("空节报出来", len(empty) == 1 and "空的" in empty[0].text, str(empty))

    no = lint.check_no_evidence("traits.md", "# t\n\n## 没出处的一节\n- 判断一\n- 判断二\n")
    check("整节无出处 → warn", len(no) == 1 and no[0].code == "no-evidence", str(no))
    has = lint.check_no_evidence("traits.md", "# t\n\n## 有出处\n- 判断一\n- 来源：raw/a.md\n")
    check("挂了 raw 就不报", has == [], str(has))
    safe = lint.check_no_evidence(
        "boundaries.md", "# b\n\n## 与现实安全规则的衔接\n- 涉及自伤，按平台规则\n- 破人设优先\n"
    )
    check("安全节不报（本来就不挂 raw）", safe == [], str(safe))
    other = lint.check_no_evidence("speech.md", "# s\n\n## 随便一节\n- 判断一\n- 判断二\n")
    check("非证据页不查", other == [], str(other))


def suite_reaction_shapes() -> None:
    print("\n[反应卡三种形态都认]")
    hand = "# r\n\n### 情境：被道歉\n\n- 典型反应：不接\n- 情绪强度：平\n- 来源：raw/a.md\n"
    check("人工卡（典型反应+情绪强度）不报", lint.check_reaction_fields("reactions.md", hand) == [])

    style_card = (
        "# r\n\n### 被关心（拥有者）\n\n- 场景：被当面关心\n- 情绪：未表现\n"
        "- 回应：停一拍\n- 露出：不外露\n"
    )
    check("样式卡用「场景」也不报", lint.check_reaction_fields("reactions.md", style_card) == [])

    new_card = (
        "# r\n\n### 情境：被冷落 / 没人接\n\n- 条件：说完没人应\n- 情绪：先当没发生\n"
        "- 回应：收回去\n- 露出：不外露\n"
    )
    check("样式卡用「条件」也不报", lint.check_reaction_fields("reactions.md", new_card) == [])

    broken = "# r\n\n### 情境：缺一条\n\n- 条件：x\n- 情绪：y\n- 回应：z\n"
    got = lint.check_reaction_fields("reactions.md", broken)
    check("缺「露出」要报", len(got) == 1 and "露出" in got[0].why, str(got))
    check("说清是哪张卡", "反应样式卡" in got[0].why, got[0].why)

    not_reactions = lint.check_reaction_fields("traits.md", broken)
    check("不是 reactions 页就不管", not_reactions == [], str(not_reactions))


def suite_index() -> None:
    print("\n[总括段]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-lint-"))
    try:
        root = _lib(tmp)
        ok = lint.check_index(root)
        check("空骨架库：四节都在但都空", all(f.code == "index-empty" for f in ok), str([f.code for f in ok]))

        text = (root / "index.md").read_text(encoding="utf-8").replace("### 底线（1~2 条）", "### 别的东西")
        _write(root, "index.md", text)
        got = lint.check_index(root)
        check("缺一节 → error", any(f.code == "index-section" and f.level == "error" for f in got),
              str([(f.code, f.level) for f in got]))

        (root / "index.md").unlink()
        gone = lint.check_index(root)
        check("没有 index.md → error", gone and gone[0].code == "index-missing", str(gone))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_index_body() -> None:
    print("\n[总括段与 wiki 脱节]")
    from workshop import indexer

    tmp = Path(tempfile.mkdtemp(prefix="qab-lint-"))
    try:
        root = _lib(tmp)
        traits = ("# traits\n\n## 主特质\n\n### 认账\n- 定义：被点出来就当场认，不辩解，不推给条件\n- 证据：raw/a.md\n\n"
                  "### 自持\n- 定义：有底气、不轻贱自己，不靠讨好换位置\n- 证据：raw/b.md\n")
        _write(root, "wiki/traits.md", traits)
        _write(root, "wiki/identity.md", "# identity\n\n## 名字\n- 「小甲」—— 测试用的假人，专门给体检当靶子\n")
        _write(root, "wiki/speech.md", "# speech\n\n## 说话\n- 短、平，不客套\n")
        _write(root, "wiki/boundaries.md", "# boundaries\n\n## 底线\n- 不替别人认不属于自己的账\n")

        part = next(p for p in indexer.PARTS if p.title == "核心特质")
        cands = indexer.candidates(root, part)
        check("夹具：特质页抽得出两条", len(cands) >= 2, str(cands))
        bare = [re.sub(r"^[\s>*\-•·]+", "", c) for c in cands[:2]]

        head = "# index.md\n\n## 总括段\n\n### 一句话人设\n小甲，测试用假人。\n\n### 核心特质\n"
        tail = "\n### 语言风格要点\n- 短、平，不客套\n\n### 底线\n- 不替别人认账\n"
        _write(root, "index.md", head + "\n".join(cands[:2]) + "\n" + tail)
        indexer.write_state(root, {"sections": {"核心特质": {"lines": cands[:2]}}})
        check("一致时一句不报", lint.check_index_body(root) == [], str(lint.check_index_body(root)))

        # 换个前缀（- 改 · ）还是同一条，别当成"内容变了"
        alt = head + "\n".join("· " + b for b in bare) + "\n" + tail
        _write(root, "index.md", alt)
        check("换个列表前缀仍算同一条",
              not any(f.code == "index-drift" for f in lint.check_index_body(root)))

        # 同节两条一样
        _write(root, "index.md", head + cands[0] + "\n" + cands[0] + "\n" + tail)
        got = lint.check_index_body(root)
        check("同节两条一样 → index-dupe", any(f.code == "index-dupe" for f in got),
              str([f.code for f in got]))
        check("重复是 warn", all(f.level == "warn" for f in got if f.code == "index-dupe"))

        # wiki 整页换掉：总括段还留着老话 → 脱节
        _write(root, "wiki/traits.md",
               "# traits\n\n## 主特质\n\n### 爱攒东西\n- 定义：抽屉里全是没拆的快递箱，一个都舍不得扔\n\n"
               "### 记性差\n- 定义：刚说过的事转头就忘，得别人再提一遍\n")
        _write(root, "index.md", head + "\n".join(cands[:2]) + "\n" + tail)
        got = lint.check_index_body(root)
        drift = [f for f in got if f.code == "index-drift"]
        check("wiki 改了、总括段没跟上 → index-drift", len(drift) == 1, str([f.code for f in got]))
        check("脱节是 warn", bool(drift) and drift[0].level == "warn")
        check("说清几条、指向哪页", bool(drift) and "2 条" in drift[0].why
              and "traits.md" in drift[0].why, drift[0].why if drift else "")
        check("行号指到那一节的标题", bool(drift) and drift[0].line > 0, str(drift[0].line) if drift else "")

        # 人手动把这两条删了：这节他在管，别拿程序的老账说事
        _write(root, "index.md",
               head + "- 我自己写的一条\n" + tail)
        check("人把整节换掉 → 不报", lint.check_index_body(root) == [],
              str(lint.check_index_body(root)))

        # 页本身空着：脱节无从谈起（另有 index-empty 报）
        _write(root, "wiki/traits.md", "# traits\n\n## 底色\n_待填_\n")
        _write(root, "index.md", head + "\n".join(cands[:2]) + "\n" + tail)
        check("wiki 页是空骨架 → 不报脱节",
              not any(f.code == "index-drift" for f in lint.check_index_body(root)))

        # 程序从没接手过（没有 state）
        (root / "data" / "index_state.json").unlink()
        check("没有 index_state → 不报脱节",
              not any(f.code == "index-drift" for f in lint.check_index_body(root)))

        (root / "index.md").unlink()
        check("没有 index.md 时不重复报", lint.check_index_body(root) == [],
              str(lint.check_index_body(root)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- 总入口 ----------


def suite_run() -> None:
    print("\n[跑一遍整库]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-lint-"))
    try:
        root = _lib(tmp)
        _write(root, "wiki/traits.md",
               "# traits\n\n## 底色\n- 她总是先顶一句\n- 来源：raw/a.md\n")
        _write(root, "wiki/identity.md", "# identity\n\n## 名字\n- 形态：反应样式\n")
        before = {p.name: p.read_bytes() for p in (root / "wiki").iterdir()}

        out = lint.run(root)
        check("跑完 ok", out.get("ok") is True)
        check("数出 error", out["errors"] >= 2, str(out["errors"]))
        check("有 warn", out["warns"] >= 1, str(out["warns"]))
        check("summary 有人话", out["summary"].startswith("体检 "), out["summary"])
        codes = _codes(out, "error")
        check("断言进了 error", "judgement" in codes, str(codes))
        check("越界进了 error", "card-scope" in codes, str(codes))
        check("排序：error 排前面", out["items"][0]["level"] == "error", out["items"][0]["level"])
        after = {p.name: p.read_bytes() for p in (root / "wiki").iterdir()}
        check("体检不写盘", before == after)

        (root / "wiki" / "values.md").unlink()
        out2 = lint.run(root)
        check("缺页 → error", "page-missing" in _codes(out2, "error"), str(_codes(out2)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def suite_healthy() -> None:
    print("\n[健康的库不该报 error]")
    tmp = Path(tempfile.mkdtemp(prefix="qab-lint-"))
    try:
        root = _lib(tmp)
        for name in lint.PAGE_FILES:
            path = root / "wiki" / name
            if path.is_file():
                path.write_text(
                    f"---\ntype: x\n---\n\n# {name}\n\n## 一节\n- 条件句：被质疑时先顶一句\n- 来源：raw/a.md\n",
                    encoding="utf-8",
                )
        text = (root / "index.md").read_text(encoding="utf-8").replace("_待填_", "填好的内容")
        _write(root, "index.md", text)
        out = lint.run(root)
        check("0 error", out["errors"] == 0, str([(i["page"], i["code"]) for i in out["items"]]))
        check("brief 可用", lint.brief(out) in {"体检通过", "体检 0 error 0 warn"} or "warn" in lint.brief(out),
              lint.brief(out))
        check("brief 空输入安全", lint.brief(None) == "" and lint.brief({"ok": False}) == "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 60)
    print("人格库体检 · 确定性 lint")
    print("=" * 60)
    suite_judgement()
    suite_card_scope()
    suite_structure()
    suite_reaction_shapes()
    suite_index()
    suite_index_body()
    suite_run()
    suite_healthy()
    print()
    total = PASS + FAIL
    if FAIL == 0:
        print(f"\033[32m通过 {PASS} · 失败 0\033[0m  （共 {total} 项）")
        return 0
    print(f"\033[31m通过 {PASS} · 失败 {FAIL}\033[0m  （共 {total} 项）")
    return 1


if __name__ == "__main__":
    sys.exit(main())

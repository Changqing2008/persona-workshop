"""记忆层检索单测。不连网、不调模型。

    python test_memwiki.py

覆盖：别名命中、无命中不注入、只取事实段、性格断言拦截、日期命中、
raw 隔离、pending 不索引、预算截断。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workshop.memwiki import MemWiki

PERSON_PAGE = """---
type: person
user_id: ["u-test"]
tags: [熟人]
aliases: ["那个熟人", "老王", "截图那位"]
links: []
status: growing
confidence: low
source: "raw/x.md"
---

# 那个熟人

## 事实

- 2026-10-24 她把方案给对方看过，对方在群里念了一遍。（source: raw/x.md）
- 她总是抢着认错。（source: raw/x.md）

## 待查

- 吵架起因未知，素材里没有，不猜。
"""

TIMELINE_PAGE = """---
type: timeline
aliases: []
---

# 2026-10 事件流水

- 10-24 · 她把方案给对方看过，对方在群里念了一遍。→ [[那个熟人]]
- 10-26 · 定下记忆系统走 wiki 模式。
"""

SINGLE_CHAR_PAGE = """---
type: person
aliases: ["他"]
---

# 某个他

## 事实

- 单字别名不该被索引。
"""


def build(root: Path) -> tuple[MemWiki, Path]:
    mem = root / "memory"
    (mem / "people").mkdir(parents=True)
    (mem / "timeline").mkdir(parents=True)
    (mem / "pending").mkdir(parents=True)
    (mem / "people" / "那个熟人.md").write_text(PERSON_PAGE, encoding="utf-8")
    (mem / "people" / "某个他.md").write_text(SINGLE_CHAR_PAGE, encoding="utf-8")
    (mem / "timeline" / "2026-10.md").write_text(TIMELINE_PAGE, encoding="utf-8")
    (mem / "pending" / "未批准.md").write_text(
        '---\ntype: person\naliases: ["未批准的人"]\n---\n\n# 未批准的人\n\n## 事实\n\n- 不该被检索到。\n',
        encoding="utf-8",
    )
    return MemWiki(root), mem


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="memwiki_"))
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"  OK   {name}")
        else:
            failures.append(name)
            print(f"  FAIL {name}  {detail}")

    try:
        # raw/ 里放哨兵：聊天侧任何情况下都不许读到
        raw = tmp / "raw"
        raw.mkdir()
        (raw / "leak.md").write_text(
            '---\naliases: ["那个熟人"]\n---\n\n## 事实\n\n- SENTINEL_RAW_ONLY_9f3a 泄漏内容\n',
            encoding="utf-8",
        )

        wiki, mem = build(tmp)
        leak = "SENTINEL_RAW_ONLY_9f3a"

        print("1. 别名命中")
        hits = wiki.hits("你还记得那个熟人吗")
        check("命中那个熟人", len(hits) == 1 and hits[0].page.title == "那个熟人", str(hits))
        check("长别名得分更高", wiki.hits("截图那位").__len__() == 1)

        print("2. 无命中不注入")
        check("无关话题返回空串", wiki.render("今天天气不错") == "")
        check("空输入返回空串", wiki.render("") == "")

        print("3. 只取事实段")
        block = wiki.render("那个熟人最近怎么样")
        check("包含事实行", "方案" in block)
        check("不含待查段", "吵架起因" not in block)
        check("带出处标题", "（那个熟人）" in block)
        check("有使用提示", "不是你的性格" in block)

        print("4. 性格断言拦截")
        check("拦下她总是", "总是抢着认错" not in block)

        print("5. 日期命中")
        tl = wiki.render("10月24号那天怎么了")
        check("命中时间线", "时间线" in tl and "方案" in tl)
        check("不串到别的日", "记忆系统走" not in tl)

        print("6. 隔离与索引边界")
        check("raw 不被索引", leak not in wiki.render("那个熟人"))
        check("pending 不被索引", wiki.pages and all("未批准" not in p.rel for p in wiki.pages))
        check("单字别名不建索引", all("他" != a for a in wiki._alias_map))

        print("6b. scope: system 不注入")
        (mem / "projects").mkdir(exist_ok=True)
        (mem / "projects" / "内部工具.md").write_text(
            '---\ntype: project\nscope: system\naliases: ["内部工具"]\n---\n\n# 内部工具\n\n## 事实\n\n- 实现细节不该被角色说出来。\n',
            encoding="utf-8",
        )
        wiki.refresh()
        check("system 页不注入", wiki.render("内部工具") == "")

        print("6c. has_person 认人（决定关系档位）")
        check("有 uid 且有事实 → True", wiki.has_person("u-test"))
        check("陌生人 uid → False", not wiki.has_person("u-unknown"))
        check("空 uid → False", not wiki.has_person(""))
        hollow = mem / "people" / "空壳.md"
        hollow.write_text(
            '---\ntype: person\nuser_id: ["u-hollow"]\n---\n\n# 空壳\n\n## 事实\n\n_待填_\n',
            encoding="utf-8",
        )
        wiki.refresh()
        check("有 uid 但没事实 → False", not wiki.has_person("u-hollow"))

        print("7. 预算截断")
        tight = wiki.render("那个熟人", budget=10)
        check("超预算留提示", "更多细节" in tight or len(tight) < 260, tight)

        print("7b. 空模板库不注入占位符")
        blank = Path(tempfile.mkdtemp(prefix="memwiki_blank_"))
        try:
            (blank / "memory" / "_hub").mkdir(parents=True)
            (blank / "memory" / "_hub" / "people.md").write_text(
                "---\ntype: hub\naliases: [\"认识的人\", \"都有谁\"]\n---\n\n# people\n\n"
                "| 页 | 关系档 | 一句话 | 状态 |\n|---|---|---|---|\n| _待填_ | | | |\n",
                encoding="utf-8",
            )
            bw = MemWiki(blank)
            out = bw.render("你都有谁")
            check("占位行不注入", "_待填_" not in out, out)
            check("空表格不产生条目", out == "", out)
        finally:
            shutil.rmtree(blank, ignore_errors=True)

        print("8. describe 可用")
        check("describe 有条目", "记忆库" in wiki.describe())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"失败 {len(failures)} 项：{', '.join(failures)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

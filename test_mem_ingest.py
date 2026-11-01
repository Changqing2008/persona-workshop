"""归档器单测：不连网、不调真实模型。

    python test_mem_ingest.py

覆盖：JSON 解析容错、三道校验（白名单 / quote 核对 / 性格断言）、
合并幂等、新建页、时间线落盘、水位线、缺事实段拒收。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

import mem_ingest as MI  # noqa: E402

PAGE = """---
type: person
relation: 熟人
aliases: ["那个熟人", "老王"]
status: growing
confidence: low
first_seen: 2026-10-24
updated: 2026-10-24
source: "raw/x.md"
---

# 那个熟人

## 事实

- 2026-10-24 她把方案给对方看过。（source: raw/x.md）

## 待查

- 起因未知。
"""

TRANSCRIPT = "[1] 对方：你还记得那个熟人吗\n[2] 她：嗯\n[3] 对方：他把你的方案念了一遍"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="memingest_"))
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"  OK   {name}")
        else:
            failures.append(name)
            print(f"  FAIL {name}  {detail}")

    try:
        mem = tmp / "memory"
        (mem / "people").mkdir(parents=True)
        (mem / "timeline").mkdir(parents=True)
        (mem / "pending").mkdir(parents=True)
        page = mem / "people" / "那个熟人.md"
        page.write_text(PAGE, encoding="utf-8")
        today = time.strftime("%Y-%m-%d")

        print("1. JSON 解析容错")
        ok = MI.parse_events('{"events":[{"page":"people/那个熟人","facts":["他念了方案"],"quote":"方案念了一遍"}]}')
        check("纯 JSON", len(ok) == 1 and ok[0].facts == ["他念了方案"])
        fenced = MI.parse_events(
            '```json\n{"events":[{"page":"","facts":["某事"],"quote":"方案"}]}\n```'
        )
        check("带代码块包裹", len(fenced) == 1)
        check("空事件", MI.parse_events('{"events": []}') == [])
        check("乱码不崩", MI.parse_events("模型跑飞了") == [])

        print("2. 三道校验")
        good = MI.Event(page="people/那个熟人", title="t", date=today, facts=["他把方案念了一遍"], quote="方案念了一遍")
        check("正常通过", MI.validate(good, TRANSCRIPT, True) is None)
        bad_dir = MI.Event(page="wiki/identity", title="t", date=today, facts=["x"], quote="方案念了一遍")
        check("目录白名单", "白名单" in (MI.validate(bad_dir, TRANSCRIPT, True) or ""))
        no_quote = MI.Event(page="people/那个熟人", title="t", date=today, facts=["x"], quote="")
        check("缺 quote 拒收", "quote" in (MI.validate(no_quote, TRANSCRIPT, True) or ""))
        fake = MI.Event(page="people/那个熟人", title="t", date=today, facts=["他打了她一顿"], quote="他把方案撕了")
        check("quote 核对拦编造", "找不到" in (MI.validate(fake, TRANSCRIPT, True) or ""))
        judge = MI.Event(page="people/那个熟人", title="t", date=today, facts=["她总是不接示好"], quote="方案念了一遍")
        check("性格断言拒收", "性格断言" in (MI.validate(judge, TRANSCRIPT, True) or ""))
        need_new = MI.Event(page="NEW:people/新角色", title="新角色", date=today, facts=["出现了"], quote="方案念了一遍")
        check("新页需批准", "未获批准" in (MI.validate(need_new, TRANSCRIPT, False) or ""))
        check("批准后可建", MI.validate(need_new, TRANSCRIPT, True) is None)

        print("3. 合并与幂等")
        ev = MI.Event(page="people/那个熟人", title="那个熟人", date=today, facts=["他把方案念了一遍"], quote="方案念了一遍")
        done, rej = MI.apply_events(mem, [ev], "private:1#1-3", today, TRANSCRIPT, False, False)
        text1 = page.read_text(encoding="utf-8")
        check("写入成功", "他把方案念了一遍" in text1 and not rej, str(rej))
        check("带 source", "private:1#1-3" in text1)
        check("updated 刷新", f"updated: {today}" in text1)
        before = text1
        done2, rej2 = MI.apply_events(mem, [ev], "private:1#1-3", today, TRANSCRIPT, False, False)
        check("重跑不重复", page.read_text(encoding="utf-8") == before and not rej2, str(done2))

        print("4. 新建页")
        ev2 = MI.Event(page="NEW:people/新角色", title="新角色", date=today, facts=["在群里出现过一次"], quote="方案念了一遍", confidence="low")
        done3, rej3 = MI.apply_events(mem, [ev2], "private:1#1-3", today, TRANSCRIPT, True, False)
        newp = mem / "people" / "新角色.md"
        check("页已创建", newp.is_file(), str(rej3))
        check("含事实段", "## 事实" in newp.read_text(encoding="utf-8"))
        check("status=stub", "status: stub" in newp.read_text(encoding="utf-8"))

        print("5. 时间线")
        ev3 = MI.Event(page="", title="", date=today, facts=["那天她没回消息"], quote="方案念了一遍")
        MI.apply_events(mem, [ev3], "private:1#1-3", today, TRANSCRIPT, False, False)
        tl = mem / "timeline" / f"{today[:7]}.md"
        check("月份页已建", tl.is_file())
        check("条目已写入", "那天她没回消息" in tl.read_text(encoding="utf-8"))

        print("6. 缺事实段拒收")
        bare = mem / "projects"
        bare.mkdir()
        (bare / "裸页.md").write_text("---\ntype: project\n---\n\n# 裸页\n\n没有事实段\n", encoding="utf-8")
        ev4 = MI.Event(page="projects/裸页", title="裸页", date=today, facts=["一条"], quote="方案念了一遍")
        _, rej4 = MI.apply_events(mem, [ev4], "private:1#1-3", today, TRANSCRIPT, False, False)
        check("无事实段被拒", any("事实" in r.reason for r in rej4), str(rej4))

        print("7. dry-run 不写盘")
        snapshot = page.read_text(encoding="utf-8")
        ev5 = MI.Event(page="people/那个熟人", title="t", date=today, facts=["dry 测试行"], quote="方案念了一遍")
        MI.apply_events(mem, [ev5], "private:2#9-9", today, TRANSCRIPT, False, True)
        check("dry-run 未改文件", page.read_text(encoding="utf-8") == snapshot)

        print("8. 水位线")
        MI.write_state(mem, {"sessions": {"private:1": 3}})
        st = MI.read_state(mem)
        check("写入并读回", st["sessions"].get("private:1") == 3)
        (mem / MI.STATE_NAME).write_text("{坏 json", encoding="utf-8")
        check("坏文件降级", MI.read_state(mem)["sessions"] == {})

        print("9. 定时归档入口：make_args → run")
        # run.py 的定时任务走的就是这条路。不许连网，所以把 extract 换成假的：
        # 这里要验的是参数装配和整条落盘链路（库定位、水位、统计、hub），
        # 不是模型抽得好不好——那个由第 1~6 项负责。
        e2e = tmp / "e2e"
        (e2e / "memory" / "people").mkdir(parents=True)
        (e2e / "memory" / "timeline").mkdir(parents=True)
        (e2e / "data").mkdir(parents=True)
        (e2e / "memory" / "people" / "那个熟人.md").write_text(PAGE, encoding="utf-8")
        cfg_path = e2e / "config.toml"
        cfg_path.write_text("[memory]\ndb_path = \"data/bridge.db\"\n", encoding="utf-8")
        # 库内 data/sessions.db 优先于 config 里的 db_path，跟线上一致
        db = e2e / "data" / "sessions.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE sessions (session_id TEXT, last_ts REAL)")
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT,"
            " role TEXT, content TEXT, ts REAL, user_id TEXT)"
        )
        conn.execute("INSERT INTO sessions VALUES ('private:42', 1.0)")
        conn.execute(
            "INSERT INTO messages VALUES (1,'private:42','user',"
            "'你还记得那个熟人吗，他把你的方案念了一遍',1.0,'42')"
        )
        conn.execute("INSERT INTO messages VALUES (2,'private:42','assistant','嗯',2.0,'')")
        conn.commit()
        conn.close()

        async def fake_extract(llm, transcript, pages):  # noqa: ANN001, ARG001
            return [
                MI.Event(
                    page="people/那个熟人",
                    title="那个熟人",
                    date=time.strftime("%Y-%m-%d"),
                    facts=["他把方案念了一遍"],
                    quote="方案念了一遍",
                )
            ]

        real_extract = MI.extract
        MI.extract = fake_extract
        try:
            args = MI.make_args(persona=str(e2e), config=str(cfg_path), rebuild_hub=True)
            check("cli 与定时任务共用参数定义", args.stats is False and args.limit == MI.RECENT_DEFAULT)
            code = asyncio.run(MI.run(args))
            check("run 返回 0", code == 0, str(code))
            body = (e2e / "memory" / "people" / "那个熟人.md").read_text(encoding="utf-8")
            check("事实已落盘", "他把方案念了一遍" in body)
            check("水位线推进", MI.read_state(e2e / "memory")["sessions"].get("private:42") == 2)
            check("统计已记一笔", (e2e / "memory" / "ingest_stats.jsonl").is_file())
            check("hub 已重建", (e2e / "memory" / "_hub" / "people.md").is_file())
            # 再跑一轮：没有新消息，不该重复写
            check("空跑仍返回 0", asyncio.run(MI.run(args)) == 0)
            check(
                "没有重复行",
                (e2e / "memory" / "people" / "那个熟人.md").read_text(encoding="utf-8") == body,
            )
        finally:
            MI.extract = real_extract

        print("10. 模型给的日期为空")
        # 实测模型常把 date 留空（它不知道今天几号）。原样用下去，只进时间线的
        # 条目会写到 timeline/.md —— 文件名是一个空的点 md，既不被 hub 索引
        # 也检索不到，等于事实凭空消失。
        check("_norm_date 合法值原样", MI._norm_date("2026-01-02", today) == "2026-01-02")
        check("_norm_date 空值兜底", MI._norm_date("", today) == today)
        check("_norm_date 乱码兜底", MI._norm_date("十月中旬", today) == today)
        ev_date = MI.Event(page="", title="", date="", facts=["那天她换了头像"], quote="方案念了一遍")
        MI.apply_events(mem, [ev_date], "private:1#1-3", today, TRANSCRIPT, False, False)
        month = (mem / "timeline" / f"{today[:7]}.md").read_text(encoding="utf-8")
        check("空日期落到今天", "那天她换了头像" in month)
        check("没有空的 .md 文件", not (mem / "timeline" / ".md").exists())

        print("11. 单会话失败不拖塌整批")
        e2e2 = tmp / "e2e2"
        (e2e2 / "memory" / "people").mkdir(parents=True)
        (e2e2 / "memory" / "timeline").mkdir(parents=True)
        (e2e2 / "memory" / "people" / "那个熟人.md").write_text(PAGE, encoding="utf-8")
        (e2e2 / "data").mkdir(parents=True)
        cfg2 = e2e2 / "config.toml"
        cfg2.write_text("[memory]\ndb_path = \"data/x.db\"\n", encoding="utf-8")
        conn = sqlite3.connect(e2e2 / "data" / "sessions.db")
        conn.execute("CREATE TABLE sessions (session_id TEXT, last_ts REAL)")
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT,"
            " role TEXT, content TEXT, ts REAL, user_id TEXT)"
        )
        # last_ts 大的先处理：坏会话排在前面，好会话必须照跑
        conn.execute("INSERT INTO sessions VALUES ('private:1', 1.0)")
        conn.execute("INSERT INTO sessions VALUES ('private:2', 9.0)")
        conn.execute(
            "INSERT INTO messages VALUES (1,'private:1','user',"
            "'他还记得吗，他把你的方案念了一遍',1.0,'1')"
        )
        conn.execute("INSERT INTO messages VALUES (2,'private:1','assistant','嗯',2.0,'')")
        conn.execute(
            "INSERT INTO messages VALUES (3,'private:2','user','坏会话，他把你的方案念了一遍',9.0,'2')"
        )
        conn.commit()
        conn.close()

        async def flaky(llm, transcript, pages):  # noqa: ANN001, ARG001
            if "坏会话" in transcript:
                raise MI.LLMError("模型返回空内容")
            return [
                MI.Event(
                    page="people/那个熟人",
                    title="那个熟人",
                    date="",
                    facts=["他把方案念了一遍"],
                    quote="方案念了一遍",
                )
            ]

        MI.extract = flaky
        try:
            check("整批仍返回 0", asyncio.run(MI.run(MI.make_args(persona=str(e2e2), config=str(cfg2)))) == 0)
        finally:
            MI.extract = real_extract
        st2 = MI.read_state(e2e2 / "memory")
        check("好会话照跑", "他把方案念了一遍" in (e2e2 / "memory" / "people" / "那个熟人.md").read_text(encoding="utf-8"))
        check("好会话水位推进", st2["sessions"].get("private:1") == 2)
        check("坏会话水位不动", "private:2" not in st2["sessions"])
        stats = [
            json.loads(x)
            for x in (e2e2 / "memory" / "ingest_stats.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        check("失败也记了统计", any(s.get("error") for s in stats), str(stats))

        print("12. 不给角色本人建页")
        # 群聊里她说的话，模型分不清"她"和"对方"，实测会给角色自己建 people 页。
        # 那页一旦被检索命中，注入内容就成了「## 相关记忆 - 示例角色 14岁」，
        # 人格事实混进回忆里，还会让她把自己的底细当旧事讲出来。
        pd = tmp / "示例角色.辛.厄尔难.特羽克洛斯"
        pd.mkdir()
        names = MI.own_names(pd)
        check("目录名算自己", "示例角色.辛.厄尔难.特羽克洛斯" in names)
        check("拆段也算自己", "示例角色" in names)
        self_ev = MI.Event(
            page="NEW:people/示例角色", title="t", date="", facts=["自称14岁"], quote="方案念了一遍"
        )
        check("拒绝给自己建页", "角色本人" in (MI.validate(self_ev, TRANSCRIPT, True, names) or ""))
        other_ev = MI.Event(
            page="NEW:people/老王", title="t", date="", facts=["一条"], quote="方案念了一遍"
        )
        check("别人照建", MI.validate(other_ev, TRANSCRIPT, True, names) is None)
        check("不传 self_names 时按旧行为", MI.validate(self_ev, TRANSCRIPT, True) is None)
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

"""人格库隔离单测：切库时哪些东西必须跟着走、哪些绝对不能串。

    python test_library.py

背景：人格库最早只有 wiki/index/raw 三棵树是"一人一份"，会话库和漂移状态
都落在程序目录里，两个角色共用一个文件。这个测试守的就是那条线——
库内的东西永远只属于这个库。

分别覆盖：
    1~2  建库时该生成什么，会话库会不会回退到别人身上
    3~4  漂移与记忆的落点是否物理分家
    5~6  换库往返：A 的经历不会搬到 B，切回 A 也还在
    7~8  基线随库配置，热更新只挪目标不抹经历
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import tempfile
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workshop import library
from workshop.agent import Agent
from workshop.config import (
    BatchConfig,
    BotConfig,
    Config,
    GateConfig,
    LLMConfig,
    MemoryConfig,
    RateLimitConfig,
    ReplyConfig,
    TriggerConfig,
)
from workshop.drift import DriftConfig, DriftEvent, DriftStore, Knobs, knobs_from
from workshop.events import IncomingMessage
from workshop.memory import Memory
from workshop.persona import Persona, RETIRED_PAGES
from workshop import ingest_rules, react_style, relation

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(label)


def note(msg: str) -> None:
    print(f"       {msg}")


def make_lib(tmp: Path, name: str, gate: dict | None = None) -> Path:
    """按 create_persona 建一个真库。gate 传什么，新库的 preferences 就是什么。"""
    prefs = None
    if gate:
        prefs = {"gate": gate}
    entry = library.create_persona(tmp / name, name=name, summary="", prefs=prefs)
    return entry.path


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        print("1. 建库：骨架齐了，会话库也一并建好")
        a = make_lib(tmp, "甲")
        b = make_lib(tmp, "乙")
        check("wiki 八页都在", len(list((a / "wiki").glob("*.md"))) == len(library.WIKI_PAGES))
        # 2026-10-31 下线「高频梗」页：模板不再产出，存量库里的旧文件靠 RETIRED_PAGES 跳过
        check("新建库不再生成 memories.md", not (a / "wiki" / "memories.md").exists())
        check("存量库的 memories.md 会被跳过", "memories.md" in RETIRED_PAGES)
        check("index.md 生成", (a / "index.md").is_file())
        check("raw/identity.md 生成", (a / "raw" / "identity.md").is_file())
        check("preferences.toml 生成", library.prefs_path(a).is_file())
        check("身份母本备份目录生成", library.history_path(a).is_dir())
        # 这一条是整个测试的重点：不建它，新库就会去读启动库的聊天记录
        check("空会话库已建", library.sessions_path(a).is_file())

        # 骨架必须与"真库"（跑过几个月、人工改过的那套）同构。
        # 不同构的后果不是报错，是静默失效：归纳AI 按规程写进「## 高频情境反应」，
        # 闸门回一句"这一节不存在"，整条补丁作废，而界面上只说"收不出东西"。
        rx = (a / "wiki" / "reactions.md").read_text(encoding="utf-8")
        check("reactions 有「## 高频情境反应」", "## 高频情境反应" in rx,
              "" if "## 高频情境反应" in rx else "模板退回旧结构，反应卡落不了盘")
        check("reactions 情境按 ### 分条", "### 情境：" in rx)
        rel = (a / "wiki" / "relationships.md").read_text(encoding="utf-8")
        for lv, heading in relation.SECTIONS.items():
            check(f"关系页有「{heading}」({lv})", heading in rel)

        # 同一张刺激表散在四个地方（内核 / 归纳规则 / UI 两个下拉 / 契约文档），
        # 改一处漏三处不会报错，只会让下拉里选不到、成品头过不了校验。
        print("1b. 反应卡刺激表：四处必须逐字一致")
        check("react_style 与 ingest_rules 同源",
              react_style.STIMULI == ingest_rules.STIMULI,
              f"{len(react_style.STIMULI)} vs {len(ingest_rules.STIMULI)}")
        here = Path(__file__).resolve().parent
        ui = (here / "workshop" / "ui" / "index.html").read_text(encoding="utf-8")
        for var in ("STIMULI", "STIM_OPTS"):
            m = re.search(rf"const {var}\s*=\s*\[(.*?)\];", ui, re.S)
            got = [x.strip().strip("'\"") for x in m.group(1).split(",")] if m else []
            check(f"UI 的 {var} 与后端一致", got == list(react_style.STIMULI),
                  f"UI {len(got)} 项 / 后端 {len(react_style.STIMULI)} 项")
        agents = (here / "workshop" / "templates" / "agents.md").read_text(encoding="utf-8")
        lack = [s for s in react_style.STIMULI if s not in agents]
        check("模板 agents.md 写全十类", not lack, ("缺 " + "/".join(lack)) if lack else "")

        print("2. resolve_db：两库各用各的，谁也不掉进回退分支")
        fallback = tmp / "global.db"
        Memory(fallback).close()  # 造一个"旧全局库"当诱饵
        da = library.resolve_db(a, fallback)
        db = library.resolve_db(b, fallback)
        check("甲用自己的库", da == library.sessions_path(a), str(da))
        check("乙用自己的库", db == library.sessions_path(b), str(db))
        check("两者不是同一个文件", da != db)
        check("都没落到全局库", fallback not in (da, db))
        check("目录名不会互相污染", library.sessions_path(a) != library.sessions_path(b))

        print("3. 记忆隔离：甲写的话乙看不到")
        ma = Memory(da)
        ma.append("group:1", "user", "甲这边说的话")
        check("甲读得到自己写的", len(ma.recent("group:1", 10)) == 1)
        ma.close()
        mb = Memory(db)
        check("乙读不到甲的话", len(mb.recent("group:1", 10)) == 0)
        mb.close()

        print("4. 漂移落点：一人一份，不共用程序目录的文件")
        pa = library.drift_path(a)
        pb = library.drift_path(b)
        check("路径不同", pa != pb)
        check("都在库内 data/ 下", pa.parent == a / "data" and pb.parent == b / "data")
        check("未启用时还不存在", not pa.exists() and not pb.exists())

        print("5. 换库往返：甲的经历不会搬到乙，切回来也还在")
        # 甲的旋钮基线：怕生 0.35
        gate_a = {"talkativeness": 0.55, "curiosity": 0.6, "shyness": 0.35, "assertiveness": 0.5}
        # 乙的旋钮基线故意设得不一样：怕生 0.80
        gate_b = {"talkativeness": 0.30, "curiosity": 0.4, "shyness": 0.80, "assertiveness": 0.2}

        cfg_a = DriftConfig(enabled=True)
        store_a = DriftStore(pa, cfg_a, baseline=knobs_from(_Gate(**gate_a)))
        # 甲连续被冷落 20 次
        for i in range(20):
            store_a.record(DriftEvent(kind="ignored", ts=1000.0 + i))
        store_a.save()
        cold = store_a.state.knobs
        note(f"甲 20 次被忽略后：怕生 {cold.shyness:.3f}（基线 0.35）")

        # 切到乙：读库内自己那份，应该是初始状态
        store_b = DriftStore(pb, DriftConfig(enabled=True), baseline=knobs_from(_Gate(**gate_b)))
        check("乙的事件数是 0，没继承甲的历史", store_b.state.turns == 0, f"turns={store_b.state.turns}")
        check("乙的怕生还是自己的基线", abs(store_b.state.knobs.shyness - 0.80) < 1e-9, f"{store_b.state.knobs.shyness}")
        check("乙的基线是乙的配置，不是甲的", abs(store_b.state.baseline.shyness - 0.80) < 1e-9)

        # 乙攒一点自己的经历
        for i in range(5):
            store_b.record(DriftEvent(kind="replied", ts=2000.0 + i, by_owner=True))
        store_b.save()

        # 切回甲：20 次记录必须还在，且没被乙的 5 次污染
        back_a = DriftStore(pa, cfg_a, baseline=knobs_from(_Gate(**gate_a)))
        check("切回甲，事件数还在", back_a.state.turns == 20, f"turns={back_a.state.turns}")
        check("切回甲的旋钮与刷盘时一致", back_a.state.knobs == cold, f"{back_a.state.knobs.shyness:.3f}")
        check("甲没沾上乙的事件", back_a.state.last_kind == "ignored", back_a.state.last_kind)

        print("6. 两库并行：一边冷一边热，各自演化互不影响")
        far_a = DriftStore(pa, cfg_a, baseline=knobs_from(_Gate(**gate_a)))
        far_b = DriftStore(pb, DriftConfig(enabled=True), baseline=knobs_from(_Gate(**gate_b)))
        for i in range(30):
            far_a.record(DriftEvent(kind="ignored", ts=3000.0 + i))
        for i in range(30):
            far_b.record(DriftEvent(kind="replied", ts=3000.0 + i, by_owner=True))
        check(
            "甲变冷（怕生高于基线）",
            far_a.state.knobs.shyness > far_a.state.baseline.shyness,
            f"{far_a.state.knobs.shyness - far_a.state.baseline.shyness:+.3f}",
        )
        check(
            "乙变热（怕生低于基线）",
            far_b.state.knobs.shyness < far_b.state.baseline.shyness,
            f"{far_b.state.knobs.shyness - far_b.state.baseline.shyness:+.3f}",
        )
        check("方向相反，说明两条链各自独立", far_a.state.turns == 50 and far_b.state.turns == 35)

        print("7. 基线随库配置：改了底子，只挪回归目标，不抹经历")
        # 用户把乙的怕生从 0.80 改成 0.50
        new_base = knobs_from(_Gate(talkativeness=0.30, curiosity=0.4, shyness=0.50, assertiveness=0.2))
        before_knobs = far_b.state.knobs
        turns_before = far_b.state.turns
        far_b.rebase(new_base)
        check("基线换成新配置", abs(far_b.state.baseline.shyness - 0.50) < 1e-9)
        check("当前旋钮没被重置", far_b.state.knobs == before_knobs, f"{far_b.state.knobs.shyness:.3f}")
        check("事件数没被清零", far_b.state.turns == turns_before, f"turns={far_b.state.turns}")
        far_b.save()
        reread = DriftStore(pb, DriftConfig(enabled=True), baseline=new_base)
        check("重新读盘不再报基线漂移", not reread.baseline_moved)
        check("旧文件的基线已被改写", abs(reread.state.baseline.shyness - 0.50) < 1e-9)

        print("8. drift_summary：只读，绝不因为看一眼就建文件")
        fresh = make_lib(tmp, "丙")
        check("没有文件时返回 None", library.drift_summary(fresh) is None)
        check("看一眼没把文件看出来", not library.drift_path(fresh).exists())
        summary = library.drift_summary(a)
        check("有文件的库能读出摘要", summary is not None and summary["turns"] > 0, str(summary and summary["turns"]))
        check("摘要带得出旋钮", summary is not None and "shyness" in summary["knobs"])

        print("9. 建库护栏：非空目录一律不碰")
        try:
            library.create_persona(a, name="甲")
            check("已存在的库被拒绝", False, "居然建成功了")
        except ValueError as exc:
            check("已存在的库被拒绝", True, str(exc)[:30])
        check("拒绝后原文件没被动", (a / "index.md").is_file())

        print("10. 真机往返：两个 Agent 交替上线，状态各归各的")
        gate_a = {"talkativeness": 0.55, "curiosity": 0.6, "shyness": 0.35, "assertiveness": 0.5}
        gate_b = {"talkativeness": 0.30, "curiosity": 0.4, "shyness": 0.80, "assertiveness": 0.2}
        d = make_lib(tmp, "丁", gate_a)
        e = make_lib(tmp, "戊", gate_b)

        d1 = asyncio.run(online(d, tmp, ["在吗", "在吗", "在吗"]))
        e1 = asyncio.run(online(e, tmp, ["在吗", "在吗"]))
        # 隔 12 小时才回来：能触发 idle，就说明 last_ts 是从文件里读回来的
        d2 = asyncio.run(online(d, tmp, ["在吗"], ts_offset=12 * 3600))

        check("丁上线后落了漂移文件", d1["drift_file"])
        check("丁确实记了事件", d1["turns"] >= 3, f"turns={d1['turns']}")
        check("丁真的发出了回复", d1["sent"] >= 1, f"sent={d1['sent']}")

        # 戊的消息更少，事件数必须比丁少；若继承了丁的历史，这里会更大
        check(
            "戊没继承丁的事件数",
            2 <= e1["turns"] < d1["turns"],
            f"戊 {e1['turns']} vs 丁 {d1['turns']}",
        )
        check(
            "戊的基线是库里的 0.80",
            abs(e1["baseline"].shyness - 0.80) < 1e-9,
            f"{e1['baseline'].shyness}",
        )
        check(
            "丁的基线是库里的 0.35",
            abs(d2["baseline"].shyness - 0.35) < 1e-9,
            f"{d2['baseline'].shyness}",
        )
        check("戊落的是自己的文件", library.drift_path(e).is_file())
        check(
            "两个文件互不相同",
            library.drift_path(d).read_text(encoding="utf-8")
            != library.drift_path(e).read_text(encoding="utf-8"),
        )

        check("丁再次上线接着自己的计数", d2["turns"] > d1["turns"], f"{d1['turns']} → {d2['turns']}")
        # 一条 at 消息最多记两笔：idle（隔了 12 小时）+ named（被点名）。
        # 若 last_ts 没从文件读回来，gap 会算成 0，就只剩 named 那一笔。
        check(
            "读到旧的 last_ts，判出了空窗",
            d2["turns"] - d1["turns"] >= 2,
            f"{d1['turns']} → {d2['turns']}（末笔 {d2['last_kind']}）",
        )
        check(
            "戊的对话没写进丁的会话库",
            e1["msgs"] < d1["msgs"],
            f"戊 {e1['msgs']} 条 vs 丁 {d1['msgs']} 条",
        )

        # 两个库：下面几节要验证「一个库的偏好不外溢到另一个库」
        from workshop import webui

        source = cfg_for(tmp / "seed", tmp)
        made = library.create_persona(
            tmp / "己", name="己",
            prefs=library.prefs_from_config(source, "己", "", webui.ALL_PATHS),
        ).path
        other = library.create_persona(
            tmp / "庚", name="庚",
            prefs=library.prefs_from_config(source, "庚", "", webui.ALL_PATHS),
        ).path

        # 一键归纳的三个开关也走 prefs：库内 [ingest] 段盖全局
        library.write_prefs(
            other,
            {
                **library.read_prefs(other),
                "ingest": {"auto_index": False, "max_per_run": 3, "preview_first": True},
            },
        )
        ci = cfg_for(other, tmp)
        library.apply_prefs(ci, other)
        check("ingest 段被认出来", ci.ingest.auto_index is False, str(ci.ingest))
        check("整数开关也对齐", ci.ingest.max_per_run == 3, str(ci.ingest.max_per_run))
        check("preview_first 也认", ci.ingest.preview_first is True)
        cj = cfg_for(made, tmp)
        library.apply_prefs(cj, made)
        check("别的库不受影响", cj.ingest.max_per_run == 0, str(cj.ingest))

        print("12. 启动库随切换写回：切到哪，开机就读哪")
        # 全局 config.toml 只放一份最小骨架，persist_boot_dir 只该动 [bot] 两行
        conf = tmp / "config.toml"
        conf.write_text(
            '[bot]\npersona_dir = "D:/别处/旧库"\nnickname = "旧库"\n\n[gate]\nmode = "hybrid"\n',
            encoding="utf-8",
        )
        r1 = library.persist_boot_dir(conf, a)
        check("第一次写回成功", r1.get("ok") is True and r1.get("changed") is True, str(r1.get("error")))
        written = tomllib.loads(conf.read_text(encoding="utf-8"))
        check("启动库指向新库", written["bot"]["persona_dir"] == str(a), str(written["bot"]["persona_dir"]))
        # 库内 preferences 有 nickname（甲）：全局兜底值也该跟着换，
        # 否则会出现"开机读甲、兜底自称旧库"
        check("兜底昵称同步成新库的", written["bot"]["nickname"] == "甲", str(written["bot"]["nickname"]))
        check("别的段一个字节没动", written["gate"]["mode"] == "hybrid")
        check("留了 .bak 备份", conf.with_suffix(".toml.bak").is_file())
        check("备份里还是旧值",
              tomllib.loads(conf.with_suffix(".toml.bak").read_text(encoding="utf-8"))
              ["bot"]["persona_dir"] == "D:/别处/旧库")

        # 幂等：目标已经在文件里就不再写盘，切多少次都不动配置
        mtime = conf.stat().st_mtime_ns
        r2 = library.persist_boot_dir(conf, a)
        check("重复切同一个库不写盘", r2.get("changed") is False and conf.stat().st_mtime_ns == mtime)

        # 乙没有 nickname 时（prefs 里空着）不动全局昵称，但启动库照写
        prefs_b_raw = tomllib.loads(library.prefs_path(b).read_text(encoding="utf-8"))
        library.write_prefs(b, {**prefs_b_raw, "bot": {**prefs_b_raw.get("bot", {}), "nickname": ""}})
        r3 = library.persist_boot_dir(conf, b)
        written3 = tomllib.loads(conf.read_text(encoding="utf-8"))
        check("启动库换成乙", written3["bot"]["persona_dir"] == str(b))
        check("乙没昵称就不动兜底值", written3["bot"]["nickname"] == "甲", str(written3["bot"]["nickname"]))

        # 文件不存在 / 坏 TOML：报错返回，不抛异常
        miss = library.persist_boot_dir(tmp / "没有.toml", a)
        check("文件不存在安静失败", miss.get("ok") is False and "不存在" in str(miss.get("error")))
        badp = tmp / "bad.toml"
        badp.write_text("[bot\n", encoding="utf-8")
        bad = library.persist_boot_dir(badp, a)
        check("坏 TOML 安静失败", bad.get("ok") is False and "读取配置失败" in str(bad.get("error")))

    print()
    if failures:
        print(f"失败 {len(failures)} 项：{', '.join(failures)}")
        return 1
    print("全部通过")
    return 0


class _Gate:
    """假门控配置：只提供漂移要的四个旋钮，不拖 Config 进来。"""

    def __init__(self, talkativeness: float, curiosity: float, shyness: float, assertiveness: float) -> None:
        self.talkativeness = talkativeness
        self.curiosity = curiosity
        self.shyness = shyness
        self.assertiveness = assertiveness


class FakeLLM:
    """不联网的假模型。只回答固定内容，够走完发送链路。"""

    def __init__(self) -> None:
        self.cfg = LLMConfig()

    @property
    def ready(self) -> bool:
        return True

    async def chat(self, messages, **kwargs) -> str:
        return "嗯。"

    async def complete(self, system: str, user: str, **kwargs) -> str:
        return "（摘要）"


def cfg_for(root: Path, tmp: Path) -> Config:
    """一份"干净"的全局配置。库级差异交给 preferences.toml，跟真实启动一致。"""
    return Config(
        bot=BotConfig(self_id="999", persona_dir=str(root), nickname=root.name),
        trigger=TriggerConfig(private="always", group="at", keywords=[]),
        batch=BatchConfig(enabled=False),
        gate=GateConfig(mode="rule"),
        # 故意指向一个全局库：resolve_db 要是回退了，记忆就会串到这里
        memory=MemoryConfig(db_path=str(tmp / "global.db"), recent_turns=12, summarize_after=999),
        reply=ReplyConfig(),
        rate_limit=RateLimitConfig(per_session_per_min=99, min_interval_sec=0.0),
        drift=DriftConfig(enabled=True),
        channel="mock",
        root=tmp,
    )


async def online(root: Path, tmp: Path, texts: list[str], ts_offset: float = 0.0) -> dict:
    """让某个库真正上线：读自己的 preferences → 收消息 → 落盘下班。

    刻意走 apply_prefs：这样库的性格旋钮是从 preferences.toml 来的，
    和线上启动路径一致，才能验证"漂移基线跟着库走"。
    ts_offset 用来把消息时间往后推，制造"隔了很久才回来"的空窗。
    """
    cfg = cfg_for(root, tmp)
    library.apply_prefs(cfg, root)
    mem = Memory(library.resolve_db(root, tmp / "global.db"))
    sent: list = []
    base_ts = time.time() + ts_offset

    async def sender(out) -> None:
        sent.append(out)

    ag = Agent(cfg, Persona(root), mem, FakeLLM(), sender)
    for i, text in enumerate(texts):
        await ag.handle(
            IncomingMessage(
                channel="mock",
                scope="group",
                user_id="10001",
                user_name="小朱",
                text=text,
                session_id="group:8",
                group_id="8",
                # 带上起点时间戳：同一个库二次上线时，消息 id 不能跟上次重名，
                # 否则会被 memory 的去重表当成重放事件整条丢掉
                message_id=f"{root.name}-{int(base_ts)}-{i}",
                timestamp=base_ts + i,
                is_at_me=True,
            )
        )
    ag.flush_drift()
    out = {
        "turns": ag.drift.state.turns if ag.drift else -1,
        "knobs": ag.drift.state.knobs if ag.drift else None,
        "baseline": ag.drift.state.baseline if ag.drift else None,
        "last_kind": ag.drift.state.last_kind if ag.drift else "",
        "sent": len(sent),
        "drift_file": library.drift_path(root).is_file(),
        "msgs": len(mem.recent("group:8", 50)),
    }
    # Windows 下 sqlite 句柄不关，临时目录删不掉
    mem.close()
    return out


if __name__ == "__main__":
    sys.exit(main())

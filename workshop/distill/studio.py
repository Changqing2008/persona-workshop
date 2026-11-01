"""蒸馏器内核：小说 → 台词 → 对话段 → 素材台能直接读的成品稿。

刻意和素材台保持同一套世界观：
- 文件即真相，不建数据库。一本书三个文件（source / meta / extract），
  记事本能打开、能改、重启不丢。
- 不写 raw。导出只落 distill/out/，搬不搬由人决定。
- 只读原文。source.txt 存一份快照，之后不再改动。

两条硬规则（对应 agents.md 第六节）：
- 原文直取，不转写。台词只做"去旁白"，动作只在括号里补一个短状语。
- 小说原文算中间置信度（medium）：它是设定级证据，不是本人聊天原话。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .. import log
from ..sim import dims
from ..sim.cards import Card
from ..sim.studio import Turn
from . import cast
from . import entities
from . import judge
from . import react
from . import text as textmod

log_ = log.get("distill")

_BAD_FILENAME = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def _safe(name: str, limit: int = 32) -> str:
    s = _BAD_FILENAME.sub("", name or "").strip().replace(" ", "")
    return (s or "未命名")[:limit]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _today() -> str:
    return time.strftime("%Y-%m-%d")


class Distiller:
    """一本书一次提取。所有方法同步执行，只碰磁盘，不碰模型。"""

    def __init__(self, cfg, studio=None, llm=None) -> None:
        self.cfg = cfg
        self.studio = studio
        self.llm = llm
        """只有反应卡用得到；对话形态全程不调模型。"""
        self.root = cfg.path(cfg.distill.dir)
        self.book_dir = self.root / "books"
        self.out_dir = self.root / "out"
        self.react_dir = self.root / "react"
        for d in (self.root, self.book_dir, self.out_dir, self.react_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- 基础 ----------

    def _dir(self, book_id: str) -> Path:
        safe = re.sub(r"[^0-9a-f]", "", (book_id or "").lower())[:16]
        if not safe:
            raise ValueError("书 id 不合法")
        return self.book_dir / safe

    def _write(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) if not isinstance(data, str) else data,
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def _load_meta(self, book_id: str) -> dict:
        path = self._dir(book_id) / "meta.json"
        if not path.is_file():
            raise ValueError("这本书还没导入")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_extract(self, book_id: str) -> dict | None:
        path = self._dir(book_id) / "extract.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log_.warning("提取结果读取失败 %s：%s", book_id, exc)
            return None

    def _save_extract(self, book_id: str, data: dict) -> None:
        self._write(self._dir(book_id) / "extract.json", data)

    def _load_react(self, book_id: str) -> dict:
        path = self._dir(book_id) / "react.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log_.warning("反应卡读取失败 %s：%s", book_id, exc)
            return {}
        return data if isinstance(data, dict) else {}

    def _save_react(self, book_id: str, data: dict) -> None:
        self._write(self._dir(book_id) / "react.json", data)

    @staticmethod
    def _seg_sig(seg: dict) -> str:
        """反应卡按"段落位置"存，不按 s0001 这种序号存。

        重新提取时段 id 会从 s0001 重排，用 id 当键会把旧卡错配到别的段上。
        位置签名在同样文本下稳定，章节改了最多变成孤儿卡，不会张冠李戴。
        """
        paras = seg.get("paras") or [0, 0]
        return f"c{seg.get('chapter', 0)}-p{paras[0]}-{paras[-1]}"

    # ---------- 导入 ----------

    def import_bytes(self, name: str, data: bytes) -> dict[str, Any]:
        text, encoding = textmod.decode_bytes(data)
        return self.import_text(name, text, encoding=encoding)

    def import_text(self, name: str, text: str, encoding: str = "utf-8") -> dict[str, Any]:
        body = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(body) < 200:
            raise ValueError("文本太短了，至少要有几百字才值得蒸馏")
        sha = hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
        book_dir = self.book_dir / sha
        meta_path = book_dir / "meta.json"

        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            log_.info("这本书已经在库里：%s", meta.get("name"))
            return {**self._brief(meta), "reused": True}

        book_dir.mkdir(parents=True, exist_ok=True)
        (book_dir / "source.txt").write_text(body, encoding="utf-8")
        chapters = textmod.split_chapters(body)
        meta = {
            "id": sha,
            "name": (name or "").strip()[:40] or "未命名小说",
            "sha1": sha,
            "chars": len(body),
            "encoding": encoding,
            "imported_at": _now(),
            "chapters": [
                {
                    "index": c.index,
                    "title": c.title[:44],
                    "paras": len(c.paras),
                    "chars": sum(len(p) for p in c.paras),
                }
                for c in chapters
            ],
        }
        self._write(meta_path, meta)
        log_.info("已导入《%s》：%d 字 / %d 章", meta["name"], meta["chars"], len(meta["chapters"]))
        return {**self._brief(meta), "reused": False}

    def _brief(self, meta: dict) -> dict:
        extract = self._load_extract(meta["id"]) or {}
        return {
            "id": meta["id"],
            "name": meta["name"],
            "chars": meta["chars"],
            "chapters": len(meta.get("chapters") or []),
            "imported_at": meta.get("imported_at", ""),
            "character": (extract.get("character") or ""),
            "segments": len(extract.get("segments") or []),
            "stats": extract.get("stats") or {},
        }

    def books(self) -> list[dict]:
        out = []
        for path in self.book_dir.glob("*/meta.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append(self._brief(meta))
        out.sort(key=lambda x: x.get("imported_at") or "", reverse=True)
        return out

    # ---------- 提取 ----------

    def _fresh_distill(self):
        """每次提取前从磁盘重读一次蒸馏配置。

        工坊只在启动那一刻读一次 config.toml；更糟的是 `apply_prefs` 还会拿
        **库级 preferences.toml 覆盖全局**。于是"改了参数 → 重启 → 重新提取"
        依然拿到老结果，且看不出是哪一层盖住了谁（实际连踩三次）。

        这里按线上同一口径重算一遍（库级优先、全局兜底），保证点一次提取就是最新参数。
        读两个 toml 的开销可以忽略；读不到就沿用当前内存里的，不让配置问题挡住提取。
        """
        src = getattr(self.cfg, "source_path", None)
        persona_dir = getattr(getattr(self.cfg, "bot", None), "persona_dir", "")
        if src is None or not persona_dir:
            return self.cfg.distill
        try:
            from .. import config as config_mod
            from .. import library

            fresh = config_mod.load(src)
            library.apply_prefs(fresh, Path(persona_dir))
            return fresh.distill
        except Exception as exc:  # noqa: BLE001
            log_.warning("重读蒸馏配置失败，沿用内存里的：%s", exc)
            return self.cfg.distill

    def extract(
        self,
        book_id: str,
        character: str,
        aliases: list[str] | None = None,
        level: str = "",
        limit: int = 200,
        chapters: list[int] | None = None,
    ) -> dict[str, Any]:
        """摘某个角色的台词。chapters 为空表示全书，否则只在这些章里摘。

        章节过滤只影响"读哪几章"，不影响段落编号：段号始终是全书位置，
        这样成品头的「来源（第 a~b 段）」回原文对得上。
        """
        # 先把磁盘上的最新参数换成当前生效的，再动手算
        fresh = self._fresh_distill()
        if fresh is not self.cfg.distill:
            self.cfg.distill = fresh
        character = (character or "").strip()
        if not character:
            raise ValueError("要先填一个角色名")
        # 关系档留空是常态：导出时先空着，档位由人在素材台里划
        if level and level not in dims.RELATIONS:
            raise ValueError(f"关系只能是 {' / '.join(dims.RELATIONS)}，或留空")

        meta = self._load_meta(book_id)
        text = (self._dir(book_id) / "source.txt").read_text(encoding="utf-8")
        book_chapters = textmod.split_chapters(text)
        all_paras = [p for c in book_chapters for p in c.paras]

        known_idx = [c.index for c in book_chapters]
        scope: list[int] = []
        if chapters:
            scope = sorted({int(i) for i in chapters if int(i) in known_idx})
            if not scope:
                raise ValueError("选的章节不在书里，刷新一下重新选")

        # 名单来自实体层：全名锚 + 前缀 + 敬称 + 称呼语。
        # 不再从"XX说"里捞——这本书 99% 的台词段没有这个句式，
        # 捞出来的全是「换句话」「现在」「殿下」，直接导致台词被挂到非人名上。
        report = entities.discover(book_chapters)
        picked = report.cast(character, aliases)
        standard = picked["standard"]
        names = picked["names"]
        others = picked["others"]
        roster = picked["roster"]

        gap = int(getattr(self.cfg.distill, "gap_paragraphs", 3))
        # 轮次跨度单独一个值：它决定"能猜多远"（影响召回），gap 决定"段落多长"
        # （影响好不好读）。混用一个值必然二选一，实测吃过这个亏。
        turn_gap = int(getattr(self.cfg.distill, "turn_gap", gap) or gap)
        turn_steps = int(getattr(self.cfg.distill, "turn_steps", 0) or 0)
        guess = str(getattr(self.cfg.distill, "guess", "chain")).lower()
        judge_on = (
            str(getattr(self.cfg.distill, "judge", "auto")).lower() not in ("rules", "off")
            and bool(self.llm)
            and bool(getattr(self.llm, "ready", False))
        )
        # 开着裁判时先把认不出的句子全留下（标 unknown），判完再由 _prune 统一丢
        lines, stats = self._attribute(
            book_chapters, names, others, turn_gap, all_paras, scope,
            keep_unknown=judge_on, guess=guess, steps=turn_steps,
        )
        lines, jstats = self._judge(
            lines,
            all_paras,
            list(dict.fromkeys(names + others[:60])),
            standard,
            tuple(names),
        )
        stats.update(jstats)
        if judge_on:
            lines = self._prune(lines, turn_gap, others, all_paras, stats)
        segments = self._segment(lines, level, gap)
        # 整段原文里一次都没提到目标角色：她不在场，这些 self 全是交替推出来的幻觉。
        # 实测本书 137 段里有 38 段属于这种，含 110 句"她说的"台词——这种段拿去
        # 做反应卡，等于凭空编了一半对话。"她是否在场"是能从原文确定的事，不该猜。
        def _on_stage(seg: dict) -> bool:
            pr = seg.get("paras") or []
            if not pr:
                return True
            a, z = int(pr[0]), int(pr[-1])
            span = "".join(all_paras[max(0, a - 1) : z])
            return any(n in span for n in names)

        off_stage = [s for s in segments if not _on_stage(s)]
        if off_stage:
            stats["off_stage"] = len(off_stage)
            segments = [s for s in segments if _on_stage(s)]
        # id 在这里一次定死：界面、导出、开进素材台都靠它认段
        for i, seg in enumerate(segments, 1):
            seg["id"] = f"s{i:04d}"

        payload = {
            "book": book_id,
            "character": standard,
            "aliases": [n for n in names if n != standard],
            "level": level,
            "chapters": scope,
            "roster": roster[:40],
            "entities": report.to_dict()["entities"],
            "pending": report.pending,
            "created": _now(),
            "stats": stats,
            "segments": segments,
        }
        self._save_extract(book_id, payload)
        log_.info(
            "蒸馏《%s》%s：台词 %d 句 → %d 段（丢 %d 句没认出来）",
            meta["name"], standard, stats["kept"], len(segments), stats["dropped"],
        )
        return {
            "ok": True,
            "book": self._brief(meta),
            "character": standard,
            "aliases": payload["aliases"],
            "level": level,
            "chapters": scope,
            "stats": stats,
            "total": len(segments),
            "segments": segments[: max(1, min(int(limit or 200), 400))],
        }

    def _attribute(
        self,
        chapters: list[textmod.Chapter],
        names: list[str],
        others: list[str],
        gap: int,
        all_paras: list[str],
        scope: list[int] | None = None,
        keep_unknown: bool = False,
        guess: str = "anchor",
        steps: int = 0,
        beyond: int = 0,
    ) -> tuple[list[dict], dict]:
        """逐段认台词。显式归优先，其次按人称代词推，最后按轮次交替，其余丢掉。

        交替推只在一个前提下做：这一段里没有出现第三个已知人名。
        否则"她说"到底是谁说就说不清了，宁可丢。

        `guess` 决定轮次交替能推多远：
        - anchor（默认）：上一句必须真有名字才敢推下一句，且只推一句。
          本书 421 句交替里 374 句（89%）是从上一次猜测起跳的，A→B→A→B 一路传下去，
          金标准实测 0 正确。换个说法：那不是推断，是掷硬币接龙。
        - chain：老行为，一路交替到底。
        - off：完全不用轮次，宁可留空。

        `steps` 是比 guess 好用的旋钮：一个锚点之后最多连猜几步。0 = 不限（等同
        chain），1 = 只推一句（等同 anchor）。**它才是"多"和"准"之间真正能调的那一档**
        —— guess 只能表达两个极端，steps 能表达中间。

        `beyond` 是"在场窗口"：目标角色超过这么多段没被提到，就不再把她推成
        说话人（只影响 self，other 照推）。0 = 不限。

        scope 是要摘的章号；不在其中的章整章跳过，但段号照常往前推，
        否则成品头里的「第 a~b 段」会跟原文对不上。
        """
        # 代词消歧要用到别人的名字和性别；前 60 个足够覆盖主要角色，
        # 再多只是把"这段里到底有谁"搞得更模糊。
        known = list(dict.fromkeys(names + others[:60]))
        genders = textmod.infer_genders(all_paras, known)

        lines: list[dict] = []
        total = kept = self_n = other_n = alt = pro = same = topic = cue_n = voc = outc = 0
        last_role = ""
        last_named = False
        last_para = -999
        run = 0  # 距上一个锚点已经连猜了几步
        last_self = -999  # 目标角色最近一次被提到是在哪一段
        gidx = 0
        recent: list[str] = []

        for ch in chapters:
            if scope and ch.index not in scope:
                gidx += len(ch.paras)
                last_role, last_named, last_para, recent = "", False, -999, []
                run = 0
                continue

            # 本章的线索段预扫：旁白里写着"某某说"，而相邻段只有台词时，
            # 把这句话指给那段台词。判定逻辑在 text.cue_map，评测脚本用的是同一份。
            cue_here = {gidx + i + 1: n for i, n in textmod.cue_map(ch.paras, known).items()}

            for para in ch.paras:
                gidx += 1
                here = [n for n in known if n in para]
                if any(n in names for n in here):
                    last_self = gidx  # 她又出场了，重置"在场"的判定基准
                for qi, utt in enumerate(textmod.parse_paragraph(para, known, genders, recent)):
                    total += 1
                    speaker = utt.speaker
                    mode = utt.mode
                    if not speaker and gidx in cue_here:
                        speaker = cue_here[gidx]
                        mode = "cue_bound"
                        cue_n += 1
                    # 台词里在称呼这个名字，说话的就绝不是他。这是全篇最硬的一条
                    # 反向证据，不依赖上下文，所以不走后面的猜测。
                    forced = ""
                    if speaker and textmod.is_addressed(utt.text, speaker, known):
                        if speaker in names:
                            # 目标角色正被别人叫着 —— 这句确定不是他说的，记成 other
                            forced, mode = "other", "vocative"
                        speaker = ""
                    if speaker:
                        role = "self" if speaker in names else "other"
                    elif forced:
                        role = forced
                    elif textmod.addresses_other(utt.text, names, others, known):
                        # 台词里在叫别人、没叫自己 —— 纯引语段里少有的正向证据。
                        # 不走交替：交替已经被上一次猜测绑架了，这里能把它掰回来。
                        speaker, role, mode = "", "self", "call_out"
                        outc += 1
                    elif keep_unknown:
                        # 先全留下、标成"还不知道"，等裁判判完再决定丢不丢。
                        # 规则认不出的恰恰是最需要模型的那批；先丢再判等于没判。
                        speaker, role, mode = "", "unknown", "unknown"
                    else:
                        # 一段里有两个以上"别人"时，轮次交替就不成立了：宁可丢
                        if sum(1 for n in others if n in para) >= 2:
                            continue
                        if guess == "off":
                            continue
                        # 离上一个锚点太远了就别再接龙：传得越长越像掷硬币。
                        if steps and run >= steps:
                            continue
                        # anchor 档要求上一句是真名，chain 档什么都能接
                        able = last_named if guess == "anchor" else bool(last_role)
                        if not able or gidx - last_para > gap:
                            continue
                        speaker = ""
                        role = "other" if last_role == "self" else "self"
                        mode = "alternate"
                        # 目标角色好久没在这段情节里露面了，就别再把她推成说话人。
                        # 交替推最离谱的一类错就是：一段别人之间的对话，一路 A→B→A→B
                        # 传到某个位置正好轮到 self，却压根没她出场。
                        if (
                            role == "self"
                            and beyond
                            and gidx - last_self > beyond
                        ):
                            role = "other"
                        alt += 1
                        # 台词里在喊目标角色，就确定不是他说的——哪怕轮次推到了 self。
                        # 「呃，尤莉安娜小姐，这是什么状况？」这种人称得上是明证，
                        # 比 A→B→A→B 的交替可靠得多。
                        if role == "self" and any(
                            textmod.is_addressed(utt.text, n, known) for n in names
                        ):
                            role, mode = "other", "vocative"
                            voc += 1
                    if not utt.text:
                        continue
                    if mode == "pronoun":
                        pro += 1
                    elif mode == "same":
                        same += 1
                    elif mode == "topic":
                        topic += 1
                    elif mode == "vocative":
                        voc += 1
                    lines.append(
                        {
                            "ch": ch.index,
                            "ch_title": ch.title,
                            "para": gidx,
                            "qi": qi,
                            "role": role,
                            "text": utt.text,
                            "action": utt.action,
                            "speaker": speaker,
                            "mode": mode,
                            # 这句之前已经连猜了几步。0 = 有名字依据。
                            # 交替推一句错、后面整串跟着错，depth 就是"错了多远"，
                            # 界面靠它把"猜了十步"的句子跟"只猜了一步"的分开显示。
                            "depth": run,
                        }
                    )
                    if mode == "unknown":
                        continue  # 还没定论，不参与轮次判断
                    last_role, last_para = role, gidx
                    # 只有真名才配当下一句交替的依据；猜出来的那行本身不作数
                    last_named = bool(speaker)
                    # 有名字或被呼语证据纠正过 = 锚点，计数器清零；纯猜则累加
                    run = 0 if (speaker or mode == "vocative") else run + 1
                    kept += 1
                    if role == "self":
                        self_n += 1
                    else:
                        other_n += 1
                if here:
                    recent = here

        return lines, {
            "quotes": total,
            "kept": kept,
            "self": self_n,
            "other": other_n,
            "alternate": alt,
            "pronoun": pro,
            "same": same,
            "topic": topic,
            "cue_bound": cue_n,
            "vocative": voc,
            "call_out": outc,
            "dropped": max(0, total - kept),
        }

    def _prune(
        self,
        lines: list[dict],
        gap: int,
        others: list[str],
        all_paras: list[str],
        stats: dict,
    ) -> list[dict]:
        """裁判之后做减法：还是没人认的句子，按轮次交替兜一手，兜不住就丢。

        这一步和 _attribute 里的老逻辑同口径——只是把"判定"和"丢弃"拆成了两步，
        好让裁判有机会在丢弃之前插进来。
        """
        out: list[dict] = []
        last_role, last_para = "", -999
        alt = dropped = 0
        for ln in lines:
            if ln["speaker"]:
                out.append(ln)
                last_role, last_para = ln["role"], ln["para"]
                continue
            para = all_paras[ln["para"] - 1] if 0 < ln["para"] <= len(all_paras) else ""
            if sum(1 for n in others if n in para) >= 2 or not last_role:
                dropped += 1
                continue
            if ln["para"] - last_para > gap:
                dropped += 1
                continue
            ln["role"] = "other" if last_role == "self" else "self"
            ln["mode"] = "alternate"
            alt += 1
            out.append(ln)
            last_role, last_para = ln["role"], ln["para"]

        stats["kept"] = len(out)
        stats["self"] = sum(1 for x in out if x["role"] == "self")
        stats["other"] = sum(1 for x in out if x["role"] == "other")
        stats["alternate"] = stats.get("alternate", 0) + alt
        stats["dropped"] = max(0, stats.get("quotes", len(lines)) - len(out))
        return out

    # ---------- 块级裁判（P3） ----------

    # 规则里靠猜的档。只有这几档允许模型改判——显式命名、同段承接、
    # 线索绑定都是原文写着的，模型说不同也压不过原文。
    GUESS_MODES = frozenset({"alternate", "topic", "pronoun", "unknown", ""})

    def _scene_jobs(
        self, lines: list[dict], all_paras: list[str], names: list[str], character: str = ""
    ) -> list[dict]:
        """把台词切成场景块，每块渲染成一次调用。

        切块只看三件事：换章、两句之间隔太远、块太长。不用 NARR 硬断——
        台词之间的旁白往往正是"谁在说"的唯一线索，断了就把信息切没了。

        块大小**不能**复用 max_lines：那个是"成品一段多少句"（现在默认 3），
        拿它切块会把模型每次能看到的上下文砍成三句，判断依据全没了。
        """
        scene_gap = int(getattr(self.cfg.distill, "judge_scene_gap", 6))
        max_lines = int(getattr(self.cfg.distill, "judge_max_lines", 24))

        blocks: list[list[int]] = []
        cur: list[int] = []
        for idx, ln in enumerate(lines):
            if cur:
                prev = lines[cur[-1]]
                if (
                    ln["ch"] != prev["ch"]
                    or ln["para"] - prev["para"] > scene_gap
                    or len(cur) >= max_lines
                ):
                    blocks.append(cur)
                    cur = []
            cur.append(idx)
        if cur:
            blocks.append(cur)

        jobs: list[dict] = []
        for block in blocks:
            grouped: list[tuple[int, list[str]]] = []
            for idx in block:
                ln = lines[idx]
                if not grouped or grouped[-1][0] != ln["para"]:
                    grouped.append((ln["para"], []))
                grouped[-1][1].append(ln["text"])
            paras = [all_paras[p - 1] for p, _ in grouped if 0 < p <= len(all_paras)]
            if not paras:
                continue
            block_text, n = judge.render(
                [(all_paras[p - 1], qs) for p, qs in grouped if 0 < p <= len(all_paras)]
            )
            prev = lines[block[0] - 1]["speaker"] if block[0] > 0 else ""
            jobs.append({
                "idxs": block,
                "block": block_text,
                "n": n,
                "candidates": judge.pick_candidates(
                    paras, names, prev, must=(character,)
                ),
                "first": lines[block[0]]["para"],
            })
        return jobs

    def _judge(
        self,
        lines: list[dict],
        all_paras: list[str],
        names: list[str],
        character: str = "",
        own: tuple[str, ...] = (),
    ) -> tuple[list[dict], dict]:
        """让模型按块判"这句是谁说的"，只覆盖规则靠猜的行。

        `names` 是全剧组（主角别名 + 其他人），模型从这里选人；
        `own` 是主角自己的别名，用来定 self/other——两者不能混，
        否则别人说的话会被记到主角头上。

        任何失败都静默回退：模型挂了、JSON 坏了、答了集外的人名，
        都保留规则结果。提取不能因为裁判失败而中断。
        """
        mode = str(getattr(self.cfg.distill, "judge", "auto")).lower()
        stat = {"judge_mode": mode, "judge_blocks": 0, "judge_called": 0,
                "judge_cache": 0, "judge_failed": 0, "judge_fixed": 0, "judge_lines": 0}
        if mode in ("rules", "off") or not self.llm or not getattr(self.llm, "ready", False):
            return lines, stat

        jobs = self._scene_jobs(lines, all_paras, names, character)
        if not jobs:
            return lines, stat

        cache = None
        if getattr(self.cfg.distill, "judge_cache", True):
            try:
                cache = judge.Cache(self.root / "cache" / "judge")
            except OSError:
                cache = None

        verdicts, jstat = judge.judge_blocks_sync(
            self.llm,
            jobs,
            cache,
            max_tokens=int(getattr(self.cfg.distill, "judge_max_tokens", 4096)),
        )
        stat["judge_blocks"] = len(jobs)
        stat["judge_called"] = jstat["called"]
        stat["judge_cache"] = jstat["cache_hits"]
        stat["judge_failed"] = jstat["failed"]
        stat["judge_lines"] = jstat["lines"]
        if jstat.get("aborted"):
            stat["judge_aborted"] = jstat["aborted"]
            log_.warning("裁判提前中止（%s），后面的句子按规则结果处理", jstat["aborted"])

        own_set = set(own) or {character}
        cast = list(names)
        for job, got in zip(jobs, verdicts):
            if not got:
                continue
            for i, who in got.items():
                if not who or not 0 <= i < len(job["idxs"]):
                    continue
                who = judge.resolve(who, cast, job["candidates"])
                if not who:
                    continue
                ln = lines[job["idxs"][i]]
                if ln["mode"] not in self.GUESS_MODES:
                    continue
                if ln["speaker"] == who:
                    continue
                ln["speaker"] = who
                ln["role"] = "self" if who in own_set else "other"
                ln["mode"] = "llm"
                stat["judge_fixed"] += 1

        log_.info(
            "裁判：%d 块（新调 %d / 缓存 %d / 失败 %d），改了 %d 句",
            stat["judge_blocks"], stat["judge_called"],
            stat["judge_cache"], stat["judge_failed"], stat["judge_fixed"],
        )
        return lines, stat

    def _segment(self, lines: list[dict], level: str, gap: int) -> list[dict]:
        """把台词连成段。三件事会断开：换章、隔太远、换了对话的人。

        长短完全按 max_lines 卡死（默认 3 句）：一段就是一个回合，满了立刻断，
        不再"等对方开口"——那会让段长飘到上限的两倍。
        """
        min_lines = int(getattr(self.cfg.distill, "min_lines", 2))
        max_lines = int(getattr(self.cfg.distill, "max_lines", 3))

        segments: list[dict] = []
        bucket: list[dict] = []
        counterpart = ""

        def flush() -> None:
            nonlocal bucket, counterpart
            roles = {x["role"] for x in bucket}
            if len(bucket) >= min_lines and {"self", "other"} <= roles:
                segments.append(self._make_segment(bucket, level, counterpart))
            bucket = []
            counterpart = ""

        for line in lines:
            if bucket:
                prev = bucket[-1]
                if (
                    line["ch"] != prev["ch"]
                    or line["para"] - prev["para"] > gap
                    or len(bucket) >= max_lines
                    or (
                        line["role"] == "other"
                        and line["speaker"]
                        and counterpart
                        and line["speaker"] != counterpart
                    )
                ):
                    flush()
            bucket.append(line)
            if line["role"] == "other" and line["speaker"] and not counterpart:
                counterpart = line["speaker"]
        flush()
        return segments

    def _make_segment(self, bucket: list[dict], level: str, counterpart: str) -> dict:
        first = bucket[0]
        mine = next((x["text"] for x in bucket if x["role"] == "self"), first["text"])
        title = f"{first['ch_title']}·{mine[:8]}"
        if not counterpart:
            counterpart = next((x["speaker"] for x in bucket if x["role"] == "other" and x["speaker"]), "")
        setting = first["ch_title"][:24]
        if setting and setting != "全文":
            setting = f"{setting}·原文对话"
        else:
            setting = "书里的一场对话"
        return {
            "id": "",
            "chapter": first["ch"],
            "chapter_title": first["ch_title"],
            "paras": [bucket[0]["para"], bucket[-1]["para"]],
            "name": title[:24],
            "category": textmod.guess_category([x["text"] for x in bucket]),
            "level": level,
            "setting": setting,
            "tension": "—",
            "other_role": counterpart or "对方",
            "other_goal": "—",
            "note": "",
            "confidence": "medium",
            "lines": [
                {
                    "role": x["role"],
                    "text": x["text"],
                    "action": x["action"],
                    "speaker": x["speaker"],
                    "mode": x["mode"],
                    "depth": x.get("depth", 0),
                }
                for x in bucket
            ],
        }

    # ---------- 读取 ----------

    def get(self, book_id: str, offset: int = 0, limit: int = 200) -> dict:
        meta = self._load_meta(book_id)
        extract = self._load_extract(book_id) or {}
        segs = extract.get("segments") or []
        for i, seg in enumerate(segs, 1):
            seg["id"] = seg.get("id") or f"s{i:04d}"
        offset = max(0, int(offset or 0))
        limit = max(1, min(int(limit or 200), 400))
        return {
            "ok": True,
            "book": self._brief(meta),
            "character": extract.get("character", ""),
            "aliases": extract.get("aliases", []),
            "level": extract.get("level", ""),
            "stats": extract.get("stats", {}),
            "created": extract.get("created", ""),
            "total": len(segs),
            "offset": offset,
            "segments": segs[offset : offset + limit],
            "chapters": meta.get("chapters", []),
            "selected": extract.get("chapters", []),
            "reactions": self._reactions_by_id(book_id, segs),
            "relations": list(dims.RELATIONS),
            "categories": list(dims.CATEGORIES),
            "out_dir": str(self.out_dir),
        }

    def _reactions_by_id(self, book_id: str, segs: list[dict]) -> dict:
        """把按位置存的反应卡翻成按段 id 的映射，界面直接用。"""
        store = self._load_react(book_id)
        out: dict[str, dict] = {}
        for seg in segs:
            entry = store.get(self._seg_sig(seg))
            if isinstance(entry, dict):
                out[seg["id"]] = entry
        return out

    def _find(self, book_id: str, seg_id: str) -> tuple[dict, dict]:
        payload = self._load_extract(book_id)
        if payload is None:
            raise ValueError("还没有提取结果，先选角色提取台词")
        for seg in payload.get("segments") or []:
            if seg.get("id") == seg_id:
                return payload, seg
        raise ValueError("这一段已经不在书里了，刷新一下")

    # ---------- 编辑 ----------

    def update_segment(self, book_id: str, seg_id: str, changes: dict) -> dict:
        payload, seg = self._find(book_id, seg_id)
        allowed = {
            "name": 24, "category": 12, "level": 6, "setting": 40,
            "tension": 60, "other_role": 24, "other_goal": 60, "note": 200,
        }
        for key, limit in allowed.items():
            if key not in changes:
                continue
            value = str(changes.get(key) or "").strip()[:limit]
            if key == "level" and value and value not in dims.RELATIONS:
                continue
            if key == "category" and value and value not in dims.CATEGORIES:
                continue
            seg[key] = value or ("—" if key in {"tension", "other_goal"} else seg.get(key, ""))
        self._save_extract(book_id, payload)
        return {"ok": True, "segment": seg}

    def update_line(self, book_id: str, seg_id: str, index: int, role: str) -> dict:
        """改一句的归属，或删掉它。人手改过就不再是推定。"""
        payload, seg = self._find(book_id, seg_id)
        lines = seg.get("lines") or []
        if not 0 <= index < len(lines):
            raise ValueError("这句已经不在了，刷新一下")
        if role == "drop":
            lines.pop(index)
        else:
            lines[index]["role"] = "self" if role == "self" else "other"
            lines[index]["mode"] = "manual"
        self._save_extract(book_id, payload)
        return {"ok": True, "segment": seg}

    # ---------- 落稿 ----------

    @staticmethod
    def _line_text(line: dict) -> str:
        text = (line.get("text") or "").strip()
        action = (line.get("action") or "").strip()
        if action and not text.startswith("（"):
            return f"（{action}）{text}"
        return text

    def render(self, meta: dict, seg: dict) -> str:
        """素材台成品头。字段顺序与素材台一致，多一行「来源」便于溯源。"""
        pages = list(dims.CATEGORY_PAGES.get(seg.get("category", ""), ["speech"]))
        if seg.get("level") and "relationships" not in pages:
            pages.append("relationships")
        lines = seg.get("lines") or []
        others = sum(1 for x in lines if x.get("role") == "other")
        mine = len(lines) - others
        alt = sum(1 for x in lines if x.get("mode") == "alternate")
        pro = sum(1 for x in lines if x.get("mode") == "pronoun")
        same = sum(1 for x in lines if x.get("mode") == "same")
        topic = sum(1 for x in lines if x.get("mode") == "topic")
        cue = sum(1 for x in lines if x.get("mode") == "cue_bound")
        llm = sum(1 for x in lines if x.get("mode") == "llm")
        notes = []
        if alt or pro or same or topic or cue or llm:
            bits = []
            if llm:
                bits.append(f"{llm} 句由模型判场景")
            if cue:
                bits.append(f"{cue} 句由相邻旁白指认")
            if pro:
                bits.append(f"{pro} 句按人称代词")
            if topic:
                bits.append(f"{topic} 句按整段唯一人名")
            if same:
                bits.append(f"{same} 句承接同段上句")
            if alt:
                bits.append(f"{alt} 句按轮次交替")
            notes.append(f"推定 {alt + pro + same + topic + cue + llm} 句（{'、'.join(bits)}），按低置信看")
        if seg.get("note"):
            notes.append(str(seg["note"]).strip())

        head = [
            f"# {_today()} {seg.get('name') or '无题'}",
            "",
            f"> 用途：{' / '.join(f'{p}.md' for p in pages)}",
            f"> 分类：{seg.get('category') or '日常搭话'}",
            f"> 情境：{seg.get('level') or '—'}｜{seg.get('setting') or '原文对话'}",
            f"> 张力：{seg.get('tension') or '—'}",
            f"> 对方：{seg.get('other_role') or '对方'}（{seg.get('other_goal') or '—'}）",
            f"> 构成：原文 {mine} 句 / 人工原话 0 句 / AI 采纳 0 句 / 未采纳 0 条",
            f"> 来源：小说《{meta.get('name', '')}》{seg.get('chapter_title') or ''}"
            f"（第 {seg.get('paras', ['0', '0'])[0]}~{seg.get('paras', ['0', '0'])[-1]} 段）",
            f"> 生成：蒸馏器 {_now()}",
        ]
        if notes:
            head.append(f"> 批注：{'；'.join(notes)}")
        label = meta.get("self_label") or "角色"
        # 一段台词一条，空行分开：一段成品就是「一段对话 + 一段话轮替」，
        # 挤成一行行接在一起反而看不清是谁在说话。
        body = [
            f"**{label}**：{self._line_text(x)}"
            if x.get("role") == "self"
            else f"**对方**：{self._line_text(x)}"
            for x in lines
        ]
        return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"

    def export(self, book_id: str, seg_ids: list[str] | None = None) -> dict:
        meta = self._load_meta(book_id)
        payload = self._load_extract(book_id)
        if payload is None:
            raise ValueError("还没有提取结果")
        segs = payload.get("segments") or []
        wanted = set(seg_ids or [])
        picked = [s for s in segs if not wanted or s.get("id") in wanted]
        if not picked:
            raise ValueError("没有可导出的段")
        meta = {**meta, "self_label": payload.get("character") or "角色"}

        written = []
        for seg in picked:
            markdown = self.render(meta, seg)
            base = f"{_today()}_{_safe(meta['name'], 20)}_{_safe(seg.get('name') or '段', 16)}"
            path = self.out_dir / f"{base}.md"
            n = 2
            while path.exists():
                path = self.out_dir / f"{base}_{n}.md"
                n += 1
            path.write_text(markdown, encoding="utf-8")
            written.append({"name": path.name, "path": str(path), "markdown": markdown})
        log_.info("蒸馏导出 %d 段 → %s", len(written), self.out_dir)
        return {"ok": True, "files": written, "count": len(written), "out_dir": str(self.out_dir)}

    # ---------- 反应卡 ----------
    #
    # 与对话形态的分工：
    # 对话稿产出「她说过什么」，可以当 exemplars 的原料；
    # 反应卡只产出「遇到这类刺激怎么回」，只进 reactions / speech。
    # 后者要调模型，前者不用——所以缺 Key 时只有这条走不通。

    async def to_reaction(self, book_id: str, seg_id: str) -> dict:
        if self.llm is None or not getattr(self.llm, "ready", False):
            raise ValueError("缺少 API Key：反应卡要调模型做抽象，先在「配置」页填上")
        meta = self._load_meta(book_id)
        payload, seg = self._find(book_id, seg_id)

        her = payload.get("character") or ""
        names = [her] + list(payload.get("aliases") or []) + list(payload.get("roster") or [])
        names = [n for n in dict.fromkeys(names) if n]
        masked, source = react.mask(seg, names, her)

        user = react.USER.format(category=seg.get("category") or "日常搭话", lines=masked)
        messages = [
            {"role": "system", "content": react.SYSTEM},
            {"role": "user", "content": user},
        ]

        # 档位不由模型判断；重转时把人工上次选的那一档带过来，别让人重选一遍
        prior = self._load_react(book_id).get(self._seg_sig(seg)) or {}
        prior_level = (prior.get("card") or {}).get("level") or ""

        card: dict[str, Any] = {}
        hits: list[str] = []
        # 第一次不干净就带着"具体哪里不行"再要一次；两次都不行就标 blocked，不硬放行
        for _attempt in (1, 2):
            text = await self.llm.chat(messages, temperature=0.3, max_tokens=700)
            card = react.parse(text)
            if not card:
                hits = ["模型没按 JSON 输出"]
                messages = messages[:1] + [
                    {"role": "user", "content": user + "\n\n只输出 JSON 对象，不要解释、不要代码块。"}
                ]
                continue
            hits = react.violations(card, names, source)
            if not hits:
                break
            messages = messages[:1] + [
                {
                    "role": "user",
                    "content": user
                    + "\n\n上一次输出里有这些不能出现的东西："
                    + "；".join(hits)
                    + "。重写，去掉它们，只输出 JSON。",
                }
            ]

        status = "ok" if card and not hits else "blocked"
        if card:
            card["level"] = prior_level if prior_level in react.LEVELS else ""
        store = self._load_react(book_id)
        store[self._seg_sig(seg)] = {
            "status": status,
            "card": card,
            "violations": hits,
            "reviewed": False,
            "at": _now(),
        }
        self._save_react(book_id, store)
        log_.info("反应卡 %s：%s%s", seg_id, status, f"（{hits[0]}）" if hits else "")
        return {
            "ok": True,
            "status": status,
            "card": card,
            "violations": hits,
            "reviewed": False,
            "preview": react.render(card, meta, seg, False) if card else "",
            "reactions": store,
        }

    def update_reaction(
        self,
        book_id: str,
        seg_id: str,
        card: dict | None = None,
        reviewed: bool | None = None,
        exemplar: bool | None = None,
        example: str | None = None,
    ) -> dict:
        """手改卡片，或勾/取消"已过审"，或勾/取消"可作范例"。改完会重跑校验，违规就退回 blocked。

        范例开关（exemplar）是人工裁决，模型不参与：勾了才允许进 exemplars 的「行为范例」小节。
        顺序不能反——没过审的卡不许勾范例，否则等于用一个开关绕开第三道闸。
        """
        store = self._load_react(book_id)
        payload, seg = self._find(book_id, seg_id)
        key = self._seg_sig(seg)
        entry = store.get(key)
        if not isinstance(entry, dict):
            raise ValueError("这段还没有反应卡，先转一次")

        her = payload.get("character") or ""
        names = [her] + list(payload.get("aliases") or []) + list(payload.get("roster") or [])
        names = [n for n in dict.fromkeys(names) if n]

        if isinstance(card, dict):
            merged = dict(entry.get("card") or {})
            merged.update({k: v for k, v in card.items() if k in react.normalize({}).keys()})
            entry["card"] = react.normalize(merged)
            _masked, source = react.mask(seg, names, her)
            entry["violations"] = react.violations(entry["card"], names, source)
            entry["status"] = "ok" if not entry["violations"] else "blocked"
            if entry["status"] == "blocked":
                entry["reviewed"] = False
                entry["exemplar"] = False

        if example is not None:
            text = str(example or "").strip()[:40]
            # 例句：查专名与关系称呼，不查 8 字重合（它本来就是原话）
            hits = react.example_hits(text, names)
            if hits:
                raise ValueError("例句里" + "；".join(hits) + "——例句不写人名，换一句")
            entry["example"] = text

        if reviewed is not None:
            if reviewed and entry.get("status") != "ok":
                raise ValueError("还有违规内容，改完再标已审")
            entry["reviewed"] = bool(reviewed)
            if not entry["reviewed"]:
                # 撤审就连带撤范例：范例的准入前提是审过
                entry["exemplar"] = False

        if exemplar is not None:
            want = bool(exemplar)
            if want and not entry.get("reviewed"):
                raise ValueError("先勾「已过审」，再把这张标成可作范例")
            if want:
                hits = react.example_hits(str(entry.get("example") or ""), names)
                if hits:
                    raise ValueError("例句里" + "；".join(hits) + "，先改掉再标范例")
            entry["exemplar"] = want

        entry["at"] = _now()
        store[key] = entry
        self._save_react(book_id, store)

        meta = self._load_meta(book_id)
        return {
            "ok": True,
            "reaction": entry,
            "preview": react.render(
                entry.get("card") or {},
                meta,
                seg,
                bool(entry.get("reviewed")),
                exemplar=bool(entry.get("exemplar")),
                example=str(entry.get("example") or ""),
            ),
            "reactions": store,
        }

    def export_react(self, book_id: str, seg_ids: list[str] | None = None) -> dict:
        """只导出「已过审且无违规」的卡片。没勾的一律不写——这是最后一道闸。"""
        meta = self._load_meta(book_id)
        store = self._load_react(book_id)
        payload = self._load_extract(book_id) or {}
        segs = {s["id"]: s for s in payload.get("segments") or []}
        by_sig = {self._seg_sig(s): s for s in segs.values()}
        wanted = set(seg_ids or [])
        wanted_sigs = (
            {self._seg_sig(segs[i]) for i in wanted if i in segs} if wanted else set()
        )

        written, skipped = [], 0
        for sig, entry in store.items():
            if wanted_sigs and sig not in wanted_sigs:
                continue
            if not entry.get("reviewed") or entry.get("status") != "ok":
                skipped += 1
                continue
            seg = by_sig.get(sig)
            if not seg:
                continue
            exemplar = bool(entry.get("exemplar"))
            example = str(entry.get("example") or "")
            markdown = react.render(
                entry.get("card") or {}, meta, seg, True, exemplar=exemplar, example=example
            )
            base = (
                f"{_today()}_反应_{_safe(entry['card'].get('stimulus') or '', 8)}"
                f"_{_safe(seg.get('name') or '段', 12)}"
            )
            path = self.react_dir / f"{base}.md"
            n = 2
            while path.exists():
                path = self.react_dir / f"{base}_{n}.md"
                n += 1
            path.write_text(markdown, encoding="utf-8")
            written.append({"name": path.name, "path": str(path), "markdown": markdown})

        if not written:
            raise ValueError("没有已过审的反应卡：先把卡片改干净，勾上「已过审」再导")
        log_.info("反应卡导出 %d 张 → %s", len(written), self.react_dir)
        return {
            "ok": True,
            "files": written,
            "count": len(written),
            "skipped": skipped,
            "out_dir": str(self.react_dir),
        }

    # ---------- 与素材台互通 ----------

    def _card(self, seg: dict) -> Card:
        lines = seg.get("lines") or []
        opening = [self._line_text(x) for x in lines if x.get("role") == "other"][:1]
        if not opening:
            opening = [self._line_text(lines[0])] if lines else ["（空）"]
        return Card.from_dict(
            {
                "name": seg.get("name") or "原文段",
                "category": seg.get("category") or "日常搭话",
                "relation": seg.get("level") or "",
                "setting": seg.get("setting") or "原文对话",
                "other_role": seg.get("other_role") or "对方",
                "other_goal": seg.get("other_goal") or "—",
                "tension": seg.get("tension") or "—",
                "opening": opening,
                "origin": "novel",
            }
        )

    def _require_studio(self):
        if self.studio is None:
            raise ValueError("素材台未启用（config.toml 里 [sim] enabled = false）")
        return self.studio

    def to_sim(self, book_id: str, seg_id: str) -> dict:
        """把这一段开成素材台草稿：句子原样带过去，之后随便改、接着演。"""
        studio = self._require_studio()
        meta = self._load_meta(book_id)
        _payload, seg = self._find(book_id, seg_id)
        card = self._card(seg)
        draft = studio.new_draft(card, "other")
        draft.turns = [
            Turn(role=x["role"], text=self._line_text(x), source="novel")
            for x in (seg.get("lines") or [])
        ]
        draft.note = f"源自《{meta.get('name', '')}》{seg.get('chapter_title', '')}"
        studio.save(draft)
        log_.info("蒸馏段开进素材台：%s（%d 句）", draft.id, len(draft.turns))
        return {
            "ok": True,
            "draft_id": draft.id,
            "draft": draft.as_dict(),
            "preview": studio.preview(draft),
        }

    def to_library(self, book_id: str, seg_id: str) -> dict:
        """把这一段的情境条件收进卡片库。卡只写条件，换人格库也能抽。"""
        studio = self._require_studio()
        _meta = self._load_meta(book_id)
        _payload, seg = self._find(book_id, seg_id)
        card = self._card(seg)
        result = studio.save_card(card)
        result["total"] = len(studio.library())
        return result

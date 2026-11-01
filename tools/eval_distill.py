#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""蒸馏台台词抽取的度量基座（P0）。

只读，不写业务数据：不碰 data/、不碰任何 meta.json / extract.json，
只在临时目录建几个空文件夹，为的是能直接调用 Distiller._attribute 的**真实实现**。
度量的必须是线上跑的那份代码，不能是复刻版——否则改完代码数字会骗人。

用法：
    python tools/eval_distill.py
    python tools/eval_distill.py --book "D:/.../books/d4b9173c23625a6e"
    python tools/eval_distill.py --character "尤莉安娜·玛赞塔" --verbose
    python tools/eval_distill.py --json tools/eval/2026-10-30.json

五层指标：
    L1 文本层  章节/段落/引号切分，引号变体覆盖
    L2 实体层  人名候选发现质量
    L3 归属层  线上口径（character=…）的说话人覆盖率
    L4 结构层  场景块线索覆盖率（P1 的目标基线）
    L5 金标准  60 段人工标注的逐句归属准确率与误收率
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workshop.distill import cast as castmod  # noqa: E402
from workshop.distill import entities as EM  # noqa: E402
from workshop.distill import text as T  # noqa: E402
from workshop.distill.studio import Distiller  # noqa: E402

DEFAULT_BOOK = None  # 必须用 --book 指定，不给默认值
DEFAULT_GOLD = Path(__file__).with_name("gold_distill.jsonl")

# 金标准用到的规范名与别名。别名只放"真名类"称呼，
# 「殿下/公爵」这种泛称不能进——它们会匹配到所有同类角色，等于作弊。
GOLD_CAST = {
    "艾妮丝菲雅": ["艾妮丝菲雅", "艾妮丝"],
    "尤莉安娜": ["尤莉安娜", "尤菲"],
    "伊莉丝": ["伊莉丝"],
    "格兰特": ["格兰特"],
    "国王": ["国王", "父王"],
    "骑士团长": ["骑士团长"],
    "龙": ["龙"],
}
ALIAS_TO_STD = {a: std for std, als in GOLD_CAST.items() for a in als}
GOLD_KNOWN = list(ALIAS_TO_STD)


# ---------------------------------------------------------------- 基础设施

def make_distiller(guess: str = "chain", turn_gap: int = 20, gap: int = 3) -> Distiller:
    """造一个只落在临时目录里的 Distiller，复用线上 _attribute。

    guess / turn_gap 的默认值必须和线上 config.DistillConfig 一模一样，
    否则量出来的不是线上跑的那份口径，改完代码数字会骗人。
    """
    root = Path(tempfile.mkdtemp(prefix="eval-distill-"))
    cfg = SimpleNamespace(root=root)
    cfg.distill = SimpleNamespace(
        dir="distill",
        min_lines=2,
        max_lines=24,
        gap_paragraphs=gap,
        turn_gap=turn_gap,
        guess=guess,
    )

    def _path(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else root / p

    cfg.path = _path
    return Distiller(cfg)


def load_book(book: Path) -> tuple[dict, str, list[T.Chapter], list[str]]:
    meta = json.loads((book / "meta.json").read_text(encoding="utf-8"))
    src = (book / "source.txt").read_text(encoding="utf-8")
    chapters = T.split_chapters(src)
    all_paras = [p for c in chapters for p in c.paras]
    return meta, src, chapters, all_paras


def pct(a: float, b: float) -> str:
    return f"{a / b:.1%}" if b else "—"


# ---------------------------------------------------------------- L1 文本层

def layer_text(src: str, chapters: list[T.Chapter], all_paras: list[str]) -> dict:
    quotes = [q for p in all_paras for q in T.QUOTE_RE.findall(p)]
    lens = sorted(len(q) for q in quotes)

    # 引号变体：正文里有、但 QUOTE_RE 抓不到的那几类
    variants = {
        "「」": (src.count("「"), src.count("」")),
        "『』": (src.count("『"), src.count("』")),
        "“”": (src.count("“"), src.count("”")),
        '""': (src.count('"'), None),
    }
    missed = {
        "『』": len(re.findall(r"『[^『』]{1,600}』", src)),
        "“”": len(re.findall(r"“[^“”]{1,600}”", src)) - len(T.QUOTE_RE.findall(src)),
    }
    missed["“”"] = max(0, missed["“”"])

    # 章节样式：本书实际用哪种标题行，现有 CHAPTER_RE 能不能认
    heads = [ln.strip() for ln in src.splitlines() if ln.strip() and len(ln.strip()) <= 44]
    style = {
        "第X章": sum(1 for h in heads if re.match(r"^第\s*[0-9零一二三四五六七八九十百千两]{1,6}\s*[章节回卷部篇]", h)),
        "Chapter N": sum(1 for h in heads if re.match(r"^Chapter\s+\d", h, re.I)),
        "markdown #": sum(1 for h in heads if h.startswith("#")),
        "楔子/序章/后记": sum(1 for h in heads if h in ("楔子", "序章", "后记", "尾声")),
        "已识别": sum(1 for h in heads if T.CHAPTER_RE.match(h)),
    }

    pure = [p for p in all_paras
            if T.QUOTE_RE.search(p) and len(T.QUOTE_RE.sub("", p).strip()) <= 4]
    pure_one_name = [p for p in pure if sum(1 for n in GOLD_KNOWN if n in p) == 1]

    return {
        "chars": len(src),
        "chapters": len(chapters),
        "paras": len(all_paras),
        "quotes": len(quotes),
        "pure_dialogue": len(pure),
        "pure_one_name": len(pure_one_name),
        "quote_pairs": variants,
        "quote_missed": missed,
        "quote_len": {"median": lens[len(lens) // 2] if lens else 0,
                      "max": lens[-1] if lens else 0,
                      "over600": sum(1 for x in lens if x > 600)},
        "chapter_style": style,
    }


# ---------------------------------------------------------------- L2 实体层

def layer_entity(chapters) -> dict:
    """实体层：四信号发现的名单质量。这里报的是"能不能认出人来"，不是归属。"""
    rep = EM.discover(chapters)
    labels = [e.label for e in rep.entities]
    forms = [f for e in rep.entities for f in e.all_forms()]

    recall = {}
    for std, aliases in GOLD_CAST.items():
        hit = [a for a in aliases if any(a == f or a in f for f in forms)]
        recall[std] = hit

    return {
        "entities": len(rep.entities),
        "tier_a": sum(1 for e in rep.entities if e.tier == "A"),
        "tier_b": sum(1 for e in rep.entities if e.tier == "B"),
        "top": [(e.label, e.count, e.tier) for e in rep.entities[:12]],
        "gold_hit": sum(1 for v in recall.values() if v),
        "gold_total": len(GOLD_CAST),
        "recall": recall,
        "pending": rep.pending,
        "conflicts": [(e.canonical, e.conflicts[0]) for e in rep.entities if e.conflicts][:5],
    }


# ---------------------------------------------------------------- L3 归属层

def layer_attr(character: str, chapters, all_paras) -> dict:
    """完全照 extract() 的口径跑一遍，拿线上真实数字。"""
    d = make_distiller()
    # 与线上同一条路：名单来自实体层，不再从"XX说"捞
    picked = EM.discover(chapters).cast(character)
    standard = picked["standard"]
    names = picked["names"]
    others = picked["others"]
    roster = picked["roster"]
    turn_gap = int(getattr(d.cfg.distill, "turn_gap", 3))
    guess = str(getattr(d.cfg.distill, "guess", "chain"))
    lines, stats = d._attribute(
        chapters, names, others, turn_gap, all_paras, None, guess=guess
    )

    total = stats["quotes"]
    named = stats["kept"] - stats["alternate"]
    # 线上 known 名单里混着「换句话」「现在」这种从"XX说"里误捞的词，
    # 所以"有名字"不等于"归对人"。只有落在真实人物上的才算数。
    gmap: dict[int, str] = {}
    gi = 0
    for ch in chapters:
        for p in ch.paras:
            gi += 1
            gmap[gi] = p

    def _pure(ln: dict) -> bool:
        p = gmap.get(ln["para"], "")
        return bool(T.QUOTE_RE.search(p)) and len(T.QUOTE_RE.sub("", p).strip()) <= 4

    topic_pure = sum(1 for ln in lines if ln["mode"] == "topic" and _pure(ln))
    trusted = sum(1 for ln in lines if ln["speaker"] in ALIAS_TO_STD)
    bad = Counter(ln["speaker"] for ln in lines
                  if ln["speaker"] and ln["speaker"] not in ALIAS_TO_STD)
    return {
        "character": character,
        "resolved": standard,
        "aliases": [a for a in names if a != standard],
        "roster_size": len(roster),
        **stats,
        "named": named,
        "named_rate": named / total if total else 0.0,
        "topic_pure": topic_pure,
        "trusted": trusted,
        "trusted_rate": trusted / total if total else 0.0,
        "top_bad_speakers": bad.most_common(8),
        "alternate_rate": stats["alternate"] / total if total else 0.0,
        "dropped_rate": stats["dropped"] / total if total else 0.0,
        "_lines": lines,
    }


# ---------------------------------------------------------------- L4 结构层

def para_kind(p: str) -> str:
    """段落角色，直接用 text 层的实现——评测和线上必须是同一套判定。"""
    return T.para_kind(p)


def layer_block(chapters: list[T.Chapter]) -> dict:
    """场景块：连续的台词段 + 夹在其中的线索段。块是跨段绑定的最小单位。

    只有 NARR（纯叙述）和 SEP 能切断块——线索段必须留在块内，
    否则「台词在下一段、线索在这一段」这种最常见的中文小说写法就永远绑不上。
    """
    out = {"blocks": 0, "blocks_with_cue": 0, "blocks_with_cue_strict": 0,
           "dialogue": 0, "in_cue_block": 0, "in_cue_block_strict": 0,
           "bindable_paras": 0, "bindable_lines": 0, "block_lens": Counter()}
    for ch in chapters[1:]:
        kinds = [para_kind(p) for p in ch.paras]
        # 真正能落到"哪一句"的线索（text.cue_map 的口径），比"块内有线索"严得多
        cmap = T.cue_map(ch.paras, GOLD_KNOWN)
        for idx in cmap:
            out["bindable_paras"] += 1
            out["bindable_lines"] += len(T.QUOTE_RE.findall(ch.paras[idx]))
        i = 0
        while i < len(kinds):
            if kinds[i] in ("NARR", "SEP"):
                i += 1
                continue
            j = i
            while j < len(kinds) and kinds[j] not in ("NARR", "SEP"):
                j += 1
            block = ch.paras[i:j]
            n_dlg = sum(1 for k in kinds[i:j] if k.startswith("DIALOGUE"))
            if n_dlg:
                out["blocks"] += 1
                out["dialogue"] += n_dlg
                loose = any(T.VERB_RE.search(T.QUOTE_RE.sub("", p)) for p in block)
                strict = any(T.cue_speaker(p, GOLD_KNOWN) for p in block)
                if loose:
                    out["blocks_with_cue"] += 1
                    out["in_cue_block"] += n_dlg
                if strict:
                    out["blocks_with_cue_strict"] += 1
                    out["in_cue_block_strict"] += n_dlg
                out["block_lens"][min(n_dlg, 9)] += 1
            i = j
    d = out["dialogue"]
    out["cue_rate"] = out["in_cue_block"] / d if d else 0.0
    out["cue_rate_strict"] = out["in_cue_block_strict"] / d if d else 0.0
    out["bindable_rate"] = out["bindable_lines"] / d if d else 0.0
    out["block_lens"] = dict(sorted(out["block_lens"].items()))
    return out


# ---------------------------------------------------------------- L5 金标准

def load_gold(path: Path) -> list[dict]:
    items = []
    for ln, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            items.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"金标准第 {ln} 行不是合法 JSON：{exc}")
    return items


def norm(name: str) -> str:
    """把模型写的写法归到金标准的规范名。

    名单里两种写法都算对：模型偏爱全名（尤莉安娜·玛赞塔），金标准写短名。
    只看间隔号前那一段，正好对上——这是同一套"最长写法才算全名"的规矩，
    不该因为模型多打了几个字就判它错。
    """
    name = (name or "").strip()
    if not name:
        return ""
    if name in ALIAS_TO_STD:
        return ALIAS_TO_STD[name]
    base = name.split("·")[0].strip()
    return ALIAS_TO_STD.get(base, name)


def layer_gold(chapters, all_paras, gold: list[dict], verbose: bool = False) -> dict:
    genders = T.infer_genders(all_paras, GOLD_KNOWN)
    cmaps: dict[int, dict[int, str]] = {}
    per_type = {}
    hits = miss = wrong = extra = fp = 0
    expect_total = got_total = 0
    bad = []

    for item in gold:
        ch_i, pa_i = item["ch"], item["para"]
        ch = chapters[ch_i]
        if ch_i not in cmaps:
            cmaps[ch_i] = T.cue_map(ch.paras, GOLD_KNOWN)
        para = ch.paras[pa_i]
        # 与线上同一条路：段内判不出时，由相邻旁白指认
        cue = cmaps[ch_i].get(pa_i, "")
        us = T.parse_paragraph(para, GOLD_KNOWN, genders)
        expect = item["expect"]
        expect_total += len(expect)
        got_total += len(us)
        t = per_type.setdefault(item["type"], {"n": 0, "expect": 0, "hit": 0,
                                               "miss": 0, "wrong": 0, "extra": 0, "fp": 0})
        t["n"] += 1
        t["expect"] += len(expect)

        if not expect:
            if us:
                fp += len(us)
                t["fp"] += len(us)
                bad.append((item, para, [u.speaker or "-" for u in us], "误收"))
            continue

        for k, exp in enumerate(expect):
            if k < len(us):
                got = norm(us[k].speaker) or norm(cue)
                if got == exp:
                    hits += 1
                    t["hit"] += 1
                elif not got:
                    miss += 1
                    t["miss"] += 1
                    if verbose:
                        bad.append((item, para, [u.speaker or "-" for u in us], f"未识别（应为 {exp}）"))
                else:
                    wrong += 1
                    t["wrong"] += 1
                    if verbose:
                        bad.append((item, para, [u.speaker or "-" for u in us], f"归错（应为 {exp}，实为 {got}）"))
            else:
                miss += 1
                t["miss"] += 1
                if verbose:
                    bad.append((item, para, [u.speaker or "-" for u in us], f"漏句（应为 {exp}）"))
        if len(us) > len(expect):
            extra += len(us) - len(expect)
            t["extra"] += len(us) - len(expect)

    total = hits + miss + wrong
    return {
        "paragraphs": len(gold),
        "expect_lines": expect_total,
        "got_lines": got_total,
        "hit": hits, "miss": miss, "wrong": wrong,
        "extra": extra, "false_positive": fp,
        "accuracy": hits / total if total else 0.0,
        "named_rate": (hits + wrong) / total if total else 0.0,
        "by_type": per_type,
        "bad": bad,
    }


# ---------------------------------------------------------------- L6 裁判层

def layer_gold_judged(
    gold: list[dict],
    lines: list[dict],
    chapters: list[T.Chapter],
    verbose: bool = False,
    self_names: tuple[str, ...] = (),
) -> dict:
    """同一份金标准，换成"规则 + 块级裁判"之后的逐句准确率。

    **口径**：只看"这句该不该由目标角色说"，不看能不能报出具体是谁。
    线上的轮次交替只产 role=self/other，speaker 是空的；如果拿"具体名字"
    去比，1321 句里 1274 句都会被判成"未识别"，准确率永远卡在 9%——
    那个数字量的是"能不能点名"，不是"认没认对"，据此调优必然走反方向。

    所以：expect 里出现目标角色 → 期望 role=self，否则期望 role=other。

    金标准里的 (ch, para) 是「章在列表里的下标 + 章内段号」，
    而跑出来的行记的是「章序号 + 全书段号」。这里先换算，再按 qi 对位——
    顺序猜不得，差一节整份结论就反了。
    """
    offsets: dict[int, int] = {}
    running = 0
    for i, ch in enumerate(chapters):
        offsets[i] = running
        running += len(ch.paras)

    got_map: dict[tuple[int, int], dict[int, dict]] = {}
    for ln in lines:
        got_map.setdefault((ln["ch"], ln["para"]), {})[ln["qi"]] = ln

    me = {norm(x) for x in self_names if x}
    per_type: dict[str, dict] = {}
    hits = miss = wrong = extra = 0
    bad = []
    for item in gold:
        ch_i, pa_i = item["ch"], item["para"]
        key = (chapters[ch_i].index, offsets[ch_i] + pa_i + 1)
        expect = item["expect"]
        here = got_map.get(key, {})
        t = per_type.setdefault(item["type"], {"n": 0, "expect": 0, "hit": 0,
                                               "miss": 0, "wrong": 0, "extra": 0})
        t["n"] += 1
        t["expect"] += len(expect)
        if not expect:
            if here:
                extra += len(here)
                t["extra"] += len(here)
                if verbose:
                    bad.append((item, "", [f"{k}" for k in here], "误收"))
            continue
        for k, exp in enumerate(expect):
            ln = here.get(k)
            want_self = norm(exp) in me if me else False
            if ln is None:
                miss += 1
                t["miss"] += 1
                if verbose:
                    bad.append((item, "", "-", f"未识别（应为 {exp}）"))
                continue
            got_self = ln.get("role") == "self"
            if got_self == want_self:
                hits += 1
                t["hit"] += 1
            else:
                wrong += 1
                t["wrong"] += 1
                if verbose:
                    side = "self" if got_self else "other"
                    bad.append((item, "", side, f"归错（应为 {exp}）"))
        if len(here) > len(expect):
            extra += len(here) - len(expect)
            t["extra"] += len(here) - len(expect)

    total = hits + miss + wrong
    return {
        "paragraphs": len(gold),
        "expect_lines": sum(len(i["expect"]) for i in gold),
        "got_lines": hits + wrong + miss,
        "hit": hits, "miss": miss, "wrong": wrong, "extra": extra,
        "accuracy": hits / total if total else 0.0,
        "by_type": per_type,
        "bad": bad,
    }


# ---------------------------------------------------------------- 报告

def report(meta, book, res: dict, verbose: bool) -> None:
    say = print
    say("=" * 78)
    say(f"蒸馏台·台词抽取度量   {meta.get('name', book.name)}")
    say(f"  {book}")
    say("=" * 78)

    l1 = res["L1"]
    say("\n【L1 文本层】")
    say(f"  字数 {l1['chars']:,}   章 {l1['chapters']}   段 {l1['paras']:,}   台词 {l1['quotes']:,}")
    ql = l1["quote_len"]
    say(f"  台词长度：中位 {ql['median']}   最长 {ql['max']}   超 600 字 {ql['over600']}")
    for k, (o, c) in l1["quote_pairs"].items():
        c = "-" if c is None else c
        say(f"  引号 {k}: 开 {o} / 闭 {c}")
    for k, v in l1["quote_missed"].items():
        if v:
            say(f"  ⚠ 未收录的引号形态 {k}: {v} 处（当前抽取器抓不到）")
    say(f"  纯引语段（除台词空无一字）{l1['pure_dialogue']:,} 段，"
        f"其中段内恰好出现一个人名 {l1['pure_one_name']:,} 段")
    say(f"  章节标题样式：{l1['chapter_style']}")

    l2 = res["L2"]
    say("\n【L2 实体层】四信号发现")
    say(f"  实体 {l2['entities']} 个（A 档全名锚 {l2['tier_a']} / B 档推定 {l2['tier_b']}）")
    say("  " + "  ".join(f"{n}({t})×{c}" for n, c, t in l2["top"][:8]))
    say(f"  金标准人物命中：{l2['gold_hit']}/{l2['gold_total']}")
    miss = [k for k, v in l2["recall"].items() if not v]
    if miss:
        say(f"  ⚠ 未识别：{'、'.join(miss)}")
    if l2["pending"]:
        say("  待人工确认（C 档泛称）："
            + "  ".join(f"{p['name']}×{p['count']}" for p in l2["pending"][:4]))
    if l2["conflicts"]:
        say(f"  ⚠ 合并反证：{l2['conflicts'][0][0]} — {l2['conflicts'][0][1][:40]}")

    l3 = res["L3"]
    say("\n【L3 归属层】线上口径")
    say(f"  character={l3['character']} → 标准名 {l3['resolved']}"
        f"  别名 {l3['aliases'] or '（无）'}")
    say(f"  台词 {l3['quotes']:,}   保留 {l3['kept']:,}   丢弃 {l3['dropped']:,}")
    say(f"  ├ 有名字     {l3['named']:,}（{pct(l3['named'], l3['quotes'])}）"
        f"  ← 代词 {l3['pronoun']} / 同段承接 {l3['same']} / 整段唯一人 {l3['topic']}")
    say(f"  │   ⚠ 整段唯一人里 {l3['topic_pure']:,} 句落在纯引语段："
        f"那个名字出现在引语内部，是被称呼的人，不是说话人")
    say(f"  │   其中落在真实人物上：{l3['trusted']:,}（{pct(l3['trusted'], l3['quotes'])}）")
    if l3["top_bad_speakers"]:
        say("  │   ⚠ 被当成说话人的非人名："
            + "  ".join(f"{n}×{c}" for n, c in l3["top_bad_speakers"]))
    say(f"  ├ 轮次猜测   {l3['alternate']:,}（{pct(l3['alternate'], l3['quotes'])}）")
    say(f"  └ 未识别丢弃 {l3['dropped']:,}（{pct(l3['dropped'], l3['quotes'])}）")

    l4 = res["L4"]
    say("\n【L4 结构层】场景块（P1 目标基线）")
    say(f"  场景块 {l4['blocks']} 个，覆盖台词 {l4['dialogue']:,} 句")
    say(f"  块长分布（按台词句数）{l4['block_lens']}")
    say(f"  块内有说话动词：{l4['blocks_with_cue']} 块 → 覆盖 {l4['in_cue_block']:,} 句"
        f"（{l4['cue_rate']:.0%}）")
    say(f"  块内有『真名+动词』：{l4['blocks_with_cue_strict']} 块 → 覆盖"
        f" {l4['in_cue_block_strict']:,} 句（{l4['cue_rate_strict']:.0%}）")
    say(f"  ⚠ 以上是「块内有线索」的粗口径。真正能绑到具体一句的："
        f"{l4['bindable_paras']} 段 / {l4['bindable_lines']} 句"
        f"（{l4['bindable_rate']:.0%}）——剩下只能交模型")

    l5 = res["L5"]
    l5b = res.get("L5b") or {}
    say("\n【L5 金标准】" + (f"共 {l5['paragraphs']} 段" if l5 else "（未加载）"))
    if l5:
        say(f"  期望台词 {l5['expect_lines']} 句；实际抽出 {l5['got_lines']} 句，"
            f"其中 {l5['false_positive']} 句来自本该无台词的段（误收）")
        say(f"  逐句归属：正确 {l5['hit']} / 未识别 {l5['miss']} / 归错 {l5['wrong']}"
            f"  → 准确率 {l5['accuracy']:.0%}")
        say(f"  按类型：")
        for t, v in l5["by_type"].items():
            n = v["expect"] or 0
            say(f"    {t:12} 段 {v['n']:2}  句 {n:2}  正确 {v['hit']:2}"
                f"  {'—' if not n else pct(v['hit'], n)}"
                + (f"  误收 {v['fp']}" if v["fp"] else ""))
        if verbose and l5["bad"]:
            say("\n  明细（仅 --verbose）：")
            for item, para, got, why in l5["bad"][:40]:
                say(f"    {item['id']} c{item['ch']}p{item['para']}: {why}")
                say(f"        原文：{para[:70]}")
                say(f"        抽出：{got}")

    say("\n【L5b 金标准·线上口径】" + ("（未加载）" if not l5b else ""))
    if l5b:
        say(f"  只看「该不该由目标角色说」，不看能不能报出具体是谁")
        say(f"  逐句：正确 {l5b['hit']} / 未识别 {l5b['miss']} / 归错 {l5b['wrong']}"
            f"  → 判对率 {l5b['accuracy']:.0%}")
        say("  按类型：")
        for t, v in l5b["by_type"].items():
            n = v["expect"] or 0
            say(f"    {t:12} 段 {v['n']:2}  句 {n:2}  正确 {v['hit']:2}"
                f"  {'—' if not n else pct(v['hit'], n)}")
        if verbose and l5b["bad"]:
            say("\n  明细（仅 --verbose）：")
            for item, _para, got, why in l5b["bad"][:40]:
                say(f"    {item['id']} c{item['ch']}p{item['para']}: {why}（实为 {got or '-'}）")

    say("\n【结论】")
    say(f"  真人物命中 {pct(l3['trusted'], l3['quotes'])}"
        f" ｜ 反向归属 {pct(l3['topic_pure'], l3['quotes'])}"
        f" ｜ 轮次猜测 {pct(l3['alternate'], l3['quotes'])}"
        f" ｜ 未识别 {pct(l3['dropped'], l3['quotes'])}")
    if l5b:
        say(f"  线上口径判对率 {l5b['accuracy']:.0%}"
            f"（{l5b['expect_lines']} 句对 {l5b['hit']} 句），误收 {l5b['extra']} 句")
    elif l5:
        say(f"  金标准逐句准确率 {l5['accuracy']:.0%}"
            f"（{l5['expect_lines']} 句对 {l5['hit']} 句），误收 {l5['false_positive']} 句")

    l6 = res.get("L6")
    if l6:
        say("\n【L6 块级裁判】规则 + 模型")
        st = l6["stat"]
        say(f"  调模型 {st['judge_called']} 次（缓存 {st['judge_cache']}，失败 {st['judge_failed']}），"
            f"改了 {st['judge_fixed']} 句，覆盖 {st['judge_lines']} 句")
        say(f"  金标准逐句：正确 {l6['hit']} / 未识别 {l6['miss']} / 归错 {l6['wrong']}"
            f"  → 准确率 {l6['accuracy']:.0%}")
        if "rule_accuracy" in l6:
            say(f"  同口径下裁判前 {l6['rule_accuracy']:.0%} → 裁判后 {l6['accuracy']:.0%}"
                f"（{(l6['accuracy'] - l6['rule_accuracy']) * 100:+.1f} 个点）")
            say(f"  逐句：正确 {l6['rule_hit']} → {l6['hit']}"
                f"｜未识别 {l6['rule_miss']} → {l6['miss']}"
                f"｜归错 {l6['rule_wrong']} → {l6['wrong']}")
        if verbose and l6["bad"]:
            say("\n  明细（仅 --verbose）：")
            for item, _para, got, why in l6["bad"][:40]:
                say(f"    {item['id']} c{item['ch']}p{item['para']}: {why}  {got[:60]}")
    say("=" * 78)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="蒸馏台台词抽取度量（P0）")
    ap.add_argument("--book", type=Path, required=True, help="书目录（含 meta.json / source.txt）")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="金标准 jsonl")
    ap.add_argument("--character", required=True, help="线上口径要摘的角色名")
    ap.add_argument("--json", type=Path, default=None, help="把指标落成 JSON，便于版本间对比")
    ap.add_argument("--verbose", action="store_true", help="打印每条误判")
    ap.add_argument("--judge", action="store_true",
                    help="额外跑一遍块级裁判（真实调模型，L6）")
    ap.add_argument("--judge-no-cache", action="store_true", help="裁判不复用缓存，强制重判")
    args = ap.parse_args()

    if not (args.book / "source.txt").is_file():
        raise SystemExit(f"找不到书：{args.book}")

    meta, src, chapters, all_paras = load_book(args.book)
    res = {
        "book": {"dir": str(args.book), "name": meta.get("name"), "chars": meta.get("chars")},
        "L1": layer_text(src, chapters, all_paras),
        "L2": layer_entity(chapters),
        "L3": layer_attr(args.character, chapters, all_paras),
        "L4": layer_block(chapters),
        "L5": {},
    }
    gold = load_gold(args.gold) if args.gold.is_file() else []
    if gold:
        res["L5"] = layer_gold(chapters, all_paras, gold, args.verbose)
        # L5 量的是"段内线索能点出几个人名"，不看上层轮次。线上真正交付的是
        # self/other，所以再按同一份金标准量一遍线上产出口径——这才是调优要看的那把尺。
        me = tuple(EM.discover(chapters).cast(args.character)["names"])
        res["L5b"] = layer_gold_judged(
            gold, res["L3"]["_lines"], chapters, args.verbose, self_names=me
        )

    if args.judge:
        from workshop import config as C
        from workshop.llm import LLMClient

        real = C.load()
        if not real.llm.resolve_key():
            raise SystemExit("没配 API Key，跑不了裁判档")

        d = make_distiller()
        d.llm = LLMClient(real.llm)
        d.cfg.distill.judge = "auto"
        d.cfg.distill.judge_cache = not args.judge_no_cache
        d.root = Path(tempfile.gettempdir()) / "qab-judge-cache"
        d.root.mkdir(parents=True, exist_ok=True)

        rule_lines = [dict(x) for x in res["L3"]["_lines"]]
        # 与线上同一条路：候选集给全剧组，主角别名单独用来定 self/other；
        # 认不出的句子先全留下，判完再统一丢
        picked = EM.discover(chapters).cast(args.character)
        known = list(dict.fromkeys(picked["names"] + picked["others"][:60]))
        turn_gap = int(getattr(d.cfg.distill, "turn_gap", 3))
        seed, seed_stats = d._attribute(
            chapters, picked["names"], picked["others"], turn_gap, all_paras,
            None, True, str(getattr(d.cfg.distill, "guess", "chain")),
        )
        print(f"\n裁判中：{len(seed)} 句待判 / 候选人 {len(known)} 名，按场景块调用…",
              flush=True)
        judged, jstat = d._judge(
            seed, all_paras, known, picked["standard"], tuple(picked["names"])
        )
        judged = d._prune(judged, turn_gap, picked["others"], all_paras, seed_stats)
        jstat["kept"] = seed_stats["kept"]
        jstat["dropped"] = seed_stats["dropped"]
        res["L6"] = layer_gold_judged(
            gold, judged, chapters, args.verbose,
            self_names=tuple(picked["names"]),
        ) if gold else {}
        res["L6"]["stat"] = jstat
        if gold:
            # 同一套对位口径下，裁判前的数字才算得上有对比意义
            before = layer_gold_judged(gold, rule_lines, chapters)
            res["L6"]["rule_accuracy"] = before["accuracy"]
            res["L6"]["rule_hit"] = before["hit"]
            res["L6"]["rule_miss"] = before["miss"]
            res["L6"]["rule_wrong"] = before["wrong"]

    report(meta, args.book, res, args.verbose)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(json.dumps(res, ensure_ascii=False, default=str))
        payload["L3"].pop("_lines", None)
        payload["L5"].pop("bad", None)
        if res.get("L6"):
            payload["L6"].pop("bad", None)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"指标已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

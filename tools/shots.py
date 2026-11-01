#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""截图入库：`raw/assets/` 里的聊天截图 → `raw/` 里的对话稿。

默认只扫不转（不烧调用）。要真转得显式加 `--run`。

用法：
    python tools/shots.py                    # 看哪些图还没转
    python tools/shots.py --dry-run          # 转写但不落盘，打印全文
    python tools/shots.py --run              # 转写并落进 raw/（审：未过）
    python tools/shots.py --run --file a.png # 只转某一张
    python tools/shots.py --persona "D:/…/长青"

写出来的稿一律 `审：未过` + `情境：—`，被 `ingest` 挡在归纳门外。人核一遍，
把档位和审各改一处，再走一键归纳——这是设计，不是漏做。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workshop import shots, vision  # noqa: E402
from workshop.config import load  # noqa: E402
from workshop.llm import LLMClient, LLMError  # noqa: E402


def _size(n: int) -> str:
    return f"{n / 1024:.0f}KB" if n < 1048576 else f"{n / 1048576:.1f}MB"


def print_scan(sc: shots.Scan) -> None:
    if not sc.items:
        print(f"没有可转的图。{sc.warnings[0] if sc.warnings else ''}")
        return

    todo = sc.pending()
    print(f"{sc.root} · raw/assets 共 {len(sc.items)} 张，待转 {len(todo)} 张")
    for s in sc.items:
        tag = shots.STATUS_LABEL.get(s.status, s.status)
        extra = f" → {s.md}" if s.md else ""
        note = f"  （{s.reason}）" if s.reason else ""
        print(f"  [{tag}] {s.name}  {_size(s.size)}{extra}{note}")
    for w in sc.warnings:
        print(f"  ! {w}")


async def run(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    root = cfg.path(args.persona or cfg.bot.persona_dir)
    if not root.is_dir():
        print(f"人格库不存在：{root}")
        return 2

    sc = shots.scan(root)
    print_scan(sc)

    todo = sc.pending()
    if args.file:
        want = Path(args.file).name
        todo = [s for s in todo if s.name == want]
        if not todo:
            print(f"\n没有待转的 `{want}`（已转过、或不在 raw/assets 里）")
            return 0

    if not args.run and not args.dry_run:
        if todo:
            print("\n加 --dry-run 先看效果，加 --run 才真落盘")
        return 0

    if not todo:
        print("\n没有要转的图。")
        return 0

    model = args.model or cfg.llm.vision_model or cfg.llm.model

    llm = LLMClient(cfg.llm)
    done: list[tuple[shots.Shot, vision.Transcript]] = []
    failed = 0

    print(f"\n模型：{model}")
    for i, shot in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] 转写 {shot.name} …", end=" ", flush=True)
        try:
            tr = await vision.describe(shot.path, llm, model=args.model)
        except (vision.VisionError, LLMError) as exc:
            print(f"失败：{exc}")
            failed += 1
            continue

        if tr.kind == vision.KIND_NOT_CHAT:
            print("不是聊天截图，跳过")
        elif tr.kind == vision.KIND_UNREADABLE:
            print(f"读不出（{tr.unknown} 条），照样出一份待补稿")
        else:
            print(f"ok（{tr.readable} 条，读不出 {tr.unknown}）")
        done.append((shot, tr))

    if not done:
        print(f"\n没有可落的稿（失败 {failed} 张）。")
        return 1 if failed else 0

    results = shots.commit(root, done, date=args.date, dry=args.dry_run)

    print()
    if args.dry_run:
        print("─" * 20 + " 预览（未落盘） " + "─" * 20)
        print(results[0]["text"])
        print("─" * 54)
        print(f"共 {len(results)} 份待写。确认无误后去掉 --dry-run 再跑一次。")
        return 0

    for r in results:
        if r.get("skipped"):
            print(f"  [跳过] {r['shot']}")
        else:
            print(f"  [写入] raw/{r['md']}  （{r['turns']} 条，审：未过）")

    print(
        "\n下一步：打开稿子核三件事 —— "
        "把 `情境：—` 改成档位、把 `审：未过` 改成 `已过`、扫一眼有没有真名。\n"
        "然后走一键归纳。"
    )
    if failed:
        print(f"另有 {failed} 张转写失败，图还在原地，重跑同一条命令即可。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="把 raw/assets 里的聊天截图转成 raw/ 里的对话稿")
    ap.add_argument("--persona", default="", help="人格库目录，默认取 config.toml 的 bot.persona_dir")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--run", action="store_true", help="真转写并落盘")
    ap.add_argument("--dry-run", action="store_true", help="转写但不落盘，打印全文")
    ap.add_argument("--file", default="", help="只处理某一张（文件名）")
    ap.add_argument("--model", default="", help="临时指定看图模型，覆盖 config 的 vision_model（留空则跟着 model）")
    ap.add_argument("--date", default="", help="素材日期，默认今天（YYYY-MM-DD）")
    args = ap.parse_args()

    started = time.time()
    code = asyncio.run(run(args))
    if code == 0:
        print(f"\n用时 {time.time() - started:.1f}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

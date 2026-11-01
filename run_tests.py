#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人格工坊测试矩阵：跑库侧全部套件，一个入口看结论。

测的是「库」这一侧——库结构、归纳、记忆 wiki、素材台、蒸馏台、契约校验。
通道侧的套件（触发/合流/门控/适配器/漂移）在通道器仓库自己的 run_tests.py 里。

用法：
    python run_tests.py            # 跑全部
    python run_tests.py test_sim   # 只跑指定套件
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: 库侧套件：(文件名, 说明)
SUITES: tuple[tuple[str, str], ...] = (
    ("test_relation", "关系距离分档"),
    ("test_lint", "库体检 + 契约校验"),
    ("test_index", "index.md 重建"),
    ("test_library", "多库注册 / 切库 / 偏好隔离"),
    ("test_memwiki", "记忆 wiki 判定"),
    ("test_ingest", "归纳管线"),
    ("test_shots", "截图入库（转写与闸门）"),
    ("test_mem_ingest", "记忆归档器"),
    ("test_sim", "素材台（母本直出反应卡）"),
    ("test_distill", "蒸馏台"),
)

#: 各套件汇总行格式不统一（"通过 202 · 失败 0" / "结果：167 通过 / 0 失败" /
#: "全部通过" / unittest 的 "OK"），统一按「有成功标志且无失败标志」判定。
#: 漏判会造成假阴性（实际通过却报失败），所以这里必须覆盖全部已知格式。
FAIL_MARKERS = ("失败 1", "失败 2", "失败 3", "失败 4", "失败 5", "失败 6", "失败 7",
                "失败 8", "失败 9", "FAILED", "Traceback", "AssertionError")
PASS_MARKERS = ("失败 0", "0 失败", "0 error", "全部通过", "验收通过")


def judge(out: str, code: int) -> bool:
    """零失败判定。返回码非 0 一律不算过；有失败标志也不算过。"""
    if code != 0:
        return False
    if any(m in out for m in FAIL_MARKERS):
        return False
    if re.search(r"^OK", out, re.M):
        return True
    return any(m in out for m in PASS_MARKERS)


def run_one(name: str) -> tuple[bool, str, float]:
    path = HERE / f"{name}.py"
    if not path.is_file():
        return False, "文件不存在", 0.0
    started = time.time()
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(HERE), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    elapsed = time.time() - started
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = ""
    for line in reversed(out.strip().splitlines()):
        if any(m in line for m in PASS_MARKERS) or re.search(r"^(OK|FAILED)", line.strip())                 or "失败" in line or "Traceback" in line:
            tail = line.strip()
            break
    ok = judge(out, proc.returncode)
    return ok, tail or (f"返回码 {proc.returncode}"), elapsed


def main() -> int:
    wanted = sys.argv[1:]
    suites = [(n, d) for n, d in SUITES if not wanted or n in wanted]
    if not suites:
        print(f"没有匹配的套件：{'、'.join(wanted)}")
        print("可用：" + "、".join(n for n, _ in SUITES))
        return 2

    print(f"人格工坊测试矩阵 · {len(suites)} 个套件")
    print("─" * 66)
    failed: list[str] = []
    total = 0.0
    for name, desc in suites:
        ok, tail, elapsed = run_one(name)
        total += elapsed
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name:<18} {desc:<26} {elapsed:5.1f}s  {tail}")
        if not ok:
            failed.append(name)
    print("─" * 66)
    print(f"合计 {total:.1f}s")

    if failed:
        print(f"\n{len(failed)} 个套件未通过：{'、'.join(failed)}")
        print("单独复跑：python " + failed[0] + ".py")
        return 1
    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

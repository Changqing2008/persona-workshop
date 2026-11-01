"""关系档位判定单测。

    python test_relation.py

核心约定：记忆层启用后，熟人档看的是"记忆里有没有这个人"，不是聊过多少条。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workshop.relation import level_of

OWNER = "u-owner"
GUEST = "u-guest"


def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"  OK   {name}")
        else:
            failures.append(name)
            print(f"  FAIL {name}  {detail}")

    print("1. 拥有者优先，与记忆无关")
    check("owner 无记忆仍是 owner", level_of(OWNER, [OWNER], 0, 12, has_memory=False) == "owner")
    check("owner 有记忆仍是 owner", level_of(OWNER, [OWNER], 99, 12, has_memory=True) == "owner")

    print("2. 记忆层启用：按共同经历")
    check("有记忆 → 熟人（只聊过 0 条也算）", level_of(GUEST, [OWNER], 0, 12, has_memory=True) == "familiar")
    check("无记忆 → 陌生人（聊了 100 条也不算）", level_of(GUEST, [OWNER], 100, 12, has_memory=False) == "stranger")

    print("3. 记忆层未启用：退回老规则")
    check("turns 达到阈值 → 熟人", level_of(GUEST, [OWNER], 12, 12, has_memory=None) == "familiar")
    check("turns 不足 → 陌生人", level_of(GUEST, [OWNER], 3, 12, has_memory=None) == "stranger")
    check("阈值 0 时永不升档", level_of(GUEST, [OWNER], 999, 0, has_memory=None) == "stranger")

    print("4. 边界")
    check("owner_ids 为空不会认主", level_of(OWNER, [], 99, 12, has_memory=True) == "familiar")
    check("空 id 无记忆即陌生人", level_of("", [OWNER], 99, 12, has_memory=False) == "stranger")

    print()
    if failures:
        print(f"失败 {len(failures)} 项：{', '.join(failures)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

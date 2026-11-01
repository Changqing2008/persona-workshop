"""角色名册：把候选名归到一个人名下，并给出别名。

中文小说里同一个人有七八种叫法（全名、姓氏、职称、外号），
不合并就等于把一个人的台词劈成几份。这里只做"包含关系"的合并，
不做语义猜测——"队长"和"老周"是不是一个人，交给人在界面上补别名。
"""

from __future__ import annotations

from .text import discover_names


def variants(name: str, candidates: list[tuple[str, int]], limit: int = 8) -> list[str]:
    """取与 name 有包含关系的候选名，按出现次数排序。

    「林知夏」会把「知夏」「小知夏」都收进来；
    「陈默」不会把「陈默的师父」收进来——那个长过头了。
    """
    name = (name or "").strip()
    if not name:
        return []
    out: list[str] = []
    for cand, _count in candidates:
        if cand == name or not 2 <= len(cand) <= 6:
            continue
        if name in cand or cand in name:
            if abs(len(cand) - len(name)) <= 3:
                out.append(cand)
        if len(out) >= limit:
            break
    return out


def resolve(name: str, candidates: list[tuple[str, int]]) -> tuple[str, list[str]]:
    """返回 (标准名, 别名)。标准名优先取候选里最长的那个写法。

    短写法本身就是别名：用户填「知夏」、书里写「林知夏」，
    那「知夏」也必须能被认出来，否则开头那几段会整段漏掉。
    """
    name = (name or "").strip()
    if not name:
        return "", []
    picked = variants(name, candidates)
    standard = max([name] + picked, key=len)
    aliases = [x for x in dict.fromkeys([name] + picked) if x != standard]
    return standard, aliases


__all__ = ["discover_names", "variants", "resolve"]

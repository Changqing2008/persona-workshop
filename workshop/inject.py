"""注入仲裁：同一份人格内容有多个入口时，只留最精细的那一个。

现状：relationships.md 有两个入口同时生效。
- pick_pages()：路由命中就注入整页，三档说话方式全给；
- relation.voice()：按当前关系档只取一节。

两边一起进 system，模型看到三档并列，「这一句该按哪个说话」没有依据，
表现就是语气忽冷忽热。section() 当初单独抽一档，本来就是为了避免这件事。

规则和"记忆覆盖人格"一致：能用精细的就别再用笼统的。
本模块只做一件事——算出本轮该跳过的整页。
"""

from __future__ import annotations

from collections.abc import Iterable

# 支持"按小节精细注入"的页。这些页一旦走了精细入口，整页就不该再进来。
SECTIONED_PAGES: tuple[str, ...] = ("relationships.md",)


def excluded_pages(*, sections_used: Iterable[str]) -> frozenset[str]:
    """本轮已通过 section() 精细注入过的页里，需要跳过整页的那部分。"""
    return frozenset(p for p in sections_used if p in SECTIONED_PAGES)

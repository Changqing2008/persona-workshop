"""兼容壳：反应卡内核已上移到 `bridge/react_style.py`。

2026-10-27 上移的原因：素材台也要出反应卡，而 `distill/studio.py` 已经 import `sim`，
sim 再回头 import `distill.react` 就成了三向循环。内核放在 `bridge/react_style.py`
是唯一的单向解。

本文件**保留且不删**：`distill/studio.py`、`test_distill.py` 以及外部脚本按旧路径
`from . import react` / `react.render(...)` 引用照样能跑。新代码请直接
`from workshop import react_style`。
"""

from __future__ import annotations

from ..react_style import (  # noqa: F401
    FENCE,
    LEVELS,
    RELATION_WORDS,
    STIMULI,
    SYSTEM,
    USER,
    example_hits,
    mask,
    name_hits,
    normalize,
    parse,
    relation_hits,
    render,
    render_head,
    violations,
)

__all__ = [
    "FENCE",
    "LEVELS",
    "RELATION_WORDS",
    "STIMULI",
    "SYSTEM",
    "USER",
    "example_hits",
    "mask",
    "name_hits",
    "normalize",
    "parse",
    "relation_hits",
    "render",
    "render_head",
    "violations",
]

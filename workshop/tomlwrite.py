"""最小 TOML 序列化器。

标准库只有 tomllib（只读），改配置要写回文件就得自己序列化。
这里只覆盖本项目的配置形态：标量、标量数组、最多两层嵌套表。
"""

from __future__ import annotations

import json
from typing import Any

_BARE_KEY = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _key(k: str) -> str:
    if k and all(c in _BARE_KEY for c in k):
        return k
    return json.dumps(k, ensure_ascii=False)


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    if isinstance(v, str):
        # TOML 的字符串转义与 JSON 基本一致，可直接复用
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if v is None:
        return '""'
    return json.dumps(str(v), ensure_ascii=False)


def _emit(table: dict[str, Any], path: list[str], out: list[str]) -> None:
    scalars = {k: v for k, v in table.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in table.items() if isinstance(v, dict)}

    if path:
        out.append("")
        out.append("[" + ".".join(_key(p) for p in path) + "]")
    for k, v in scalars.items():
        out.append(f"{_key(k)} = {_value(v)}")
    for k, v in tables.items():
        _emit(v, path + [k], out)


def dumps(data: dict[str, Any]) -> str:
    out: list[str] = []
    _emit(data, [], out)
    text = "\n".join(out).strip() + "\n"
    # 折叠连续空行，保留一个
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    """按 a.b.c 写值，中间缺的表自动创建。"""
    keys = path.split(".")
    cur = data
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

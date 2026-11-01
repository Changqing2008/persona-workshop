"""终端日志。带颜色但可关，输出到 stdout 与 data/bridge.log 双写。

另外维护一个内存环形缓冲，供 WebUI 实时拉取。
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

_CONFIGURED = False

_COLORS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[36m",
    "WARN": "\033[33m",
    "ERROR": "\033[31m",
    "RESET": "\033[0m",
}
_LEVEL_ALIAS = {"WARN": "WARNING"}


class _ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-5s %(name)-12s %(message)s", "%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if record.levelname == "WARNING":
            record.levelname = "WARN"
        text = super().format(record)
        if self.use_color:
            color = _COLORS.get(record.levelname, "")
            if color:
                text = f"{color}{text}{_COLORS['RESET']}"
        return text


class RingBuffer(logging.Handler):
    """环形日志缓冲：WebUI 靠它做增量拉取，不落盘、不阻塞。"""

    def __init__(self, capacity: int = 800) -> None:
        super().__init__()
        self._items: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._total = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            msg = "<格式化失败>"
        with self._lock:
            self._total += 1
            self._items.append(
                {
                    "i": self._total,
                    "t": record.created,
                    "lvl": record.levelname,
                    "src": record.name,
                    "msg": msg,
                }
            )

    def since(self, index: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            return [x for x in self._items if x["i"] > index], self._total

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


RING = RingBuffer()


def setup(level: str = "INFO", log_file: Path | None = None) -> None:
    """初始化根 logger，幂等。"""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return

    root.setLevel(logging.DEBUG)
    use_color = sys.stdout.isatty()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, _LEVEL_ALIAS.get(level.upper(), level.upper()), logging.INFO))
    console.setFormatter(_ColorFormatter(use_color))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)-12s %(message)s"))
        root.addHandler(fh)

    RING.setLevel(logging.INFO)
    root.addHandler(RING)

    for noisy in ("websockets", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)

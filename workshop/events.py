"""统一消息事件模型。

所有通道适配器（OneBot / 官方平台 / 终端模拟）都把上游格式翻译成
IncomingMessage，智能体内核只认这一种结构。这样换接入方式时，
persona / memory / agent 三层完全不用改。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

Scope = Literal["private", "group"]


@dataclass(slots=True)
class IncomingMessage:
    """一条进来的用户消息（已归一化）。"""

    channel: str
    """来源通道标识：onebot / official / mock。"""

    scope: Scope
    """私聊还是群聊。"""

    user_id: str
    """发送者 QQ 号（字符串，避免大整数精度丢失）。"""

    text: str
    """纯文本正文，已剥离 CQ 码 / markdown 包裹。"""

    session_id: str
    """会话键。私聊 = private:<uid>；群聊 = group:<gid>。记忆按此隔离。"""

    user_name: str = ""
    group_id: str | None = None
    message_id: str | None = None
    is_at_me: bool = False
    """是否明确 @ 了机器人。"""

    is_reply_to_me: bool = False
    """是否引用了机器人的消息。"""

    images: list[str] = field(default_factory=list)
    """图片地址列表（暂不解析内容，仅记录存在）。"""

    timestamp: float = field(default_factory=time.time)
    raw: dict[str, Any] = field(default_factory=dict)
    """原始事件，排障时写日志用。"""

    @property
    def display_name(self) -> str:
        if self.user_name:
            return f"{self.user_name}({self.user_id})"
        return self.user_id

    def brief(self) -> str:
        """单行摘要，用于日志。"""
        tag = "@" if self.is_at_me else ""
        text = self.text.replace("\n", " ")[:60]
        return f"[{self.channel}/{self.scope}] {self.session_id} {tag}{self.display_name}: {text}"


@dataclass(slots=True)
class OutgoingMessage:
    """一条待发送的回复。"""

    session_id: str
    text: str
    scope: Scope
    group_id: str | None = None
    user_id: str | None = None
    reply_to: str | None = None
    """要引用的 message_id，None 表示不引用。"""

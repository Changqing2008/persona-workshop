"""对话合流。

原来一条消息走一次模型、回一条话，群里话一密就变成逐句接话的复读机。
这里把同一会话的消息先攒进缓冲，等静默窗过去（或撞上硬上限）再一次性交出去，
让模型看见的是一段完整对话，而不是一句孤零零的话。

两个窗口配合：
- 静默窗：每来一条就重置。语义是"等大家把话说完"。
- 硬上限：从本批第一条算起。有人连刷时靠它兜底，否则永远等不到静默。

关掉时（batch.enabled = false）退化成"来一条交一条"，链路不变。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from . import log
from .config import BatchConfig
from .events import IncomingMessage

log_ = log.get("batch")

Flusher = Callable[["Batch"], Awaitable[None]]

_GAP_NOTE_SEC = 90.0
"""两句之间隔这么久，在对话块里插一行说明，让模型知道中间断过。"""


@dataclass(slots=True)
class Batch:
    """一批待回应的对话。字段在 push 过程中原地补全。"""

    session_id: str
    scope: str
    group_id: str | None
    messages: list[IncomingMessage] = field(default_factory=list)
    first_id: int = 0
    """本批第一条落库后的行 id。构建上下文时以它为边界。"""
    last_id: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    deferred: int = 0
    """被限速推迟过几次。用来兜底防止等待时间算错时无限重试。"""
    retry_at: float = 0.0
    """最早可以重试的墙上时间。等待额度期间来的新消息不能把这个时刻提前。"""

    @property
    def last(self) -> IncomingMessage:
        return self.messages[-1]

    @property
    def anchor(self) -> IncomingMessage:
        """真正该被回应的人：最后一个明确叫它的人。

        原来一律用 last。群里 A @ 了它、B 随后插一句闲话，last 就变成 B——
        回复挂到 B 的 msg_id 上，内容却在答 A，两边都错。
        没人叫它（随机插话）时才退回最后发言者。
        """
        for m in reversed(self.messages):
            if m.is_at_me or m.is_reply_to_me:
                return m
        return self.messages[-1]

    @property
    def callers(self) -> list[IncomingMessage]:
        """批内所有明确叫过它的人，按人去重（同一人保留最后一次）。"""
        seen: dict[str, IncomingMessage] = {}
        for m in self.messages:
            if m.is_at_me or m.is_reply_to_me:
                seen[m.user_id] = m
        return list(seen.values())

    @property
    def named(self) -> bool:
        """批内是否有人明确叫它。"""
        return any(m.is_at_me for m in self.messages)

    @property
    def replied_to(self) -> bool:
        """批内是否有人引用了它的消息。"""
        return any(m.is_reply_to_me for m in self.messages)

    @property
    def speakers(self) -> int:
        """批内有多少个不同的人在说话。人多时怕生的角色会更收敛。"""
        return len({m.user_id for m in self.messages})

    @property
    def span_sec(self) -> float:
        return max(0.0, self.last_ts - self.first_ts)

    def text(self, limit: int = 2000) -> str:
        """整批拼接文本，用于话题匹配与门控判断。"""
        return "\n".join(m.text for m in self.messages)[:limit]

    def render(self, with_name: bool = True) -> str:
        """渲染成对话块。时间戳本身就是节奏信息，不再另加间隔说明。

        群聊（with_name=True）按发言人分段：同一人连着说的话并成一段，
        换人空一行。原来是一行一条的流水账，三个人各说两句就糊成一片，
        模型分不清"谁在叫我"和"谁在旁边聊"。段落边界就是人的边界，
        这是最省字数的隔离方式，也不需要拆会话。
        """
        if not with_name:
            # 私聊只有一个人，逐行排下来即可
            lines: list[str] = []
            prev = 0.0
            for m in self.messages:
                body = (m.text or "").replace("\n", " ").strip()
                if not body:
                    continue
                if prev and m.timestamp - prev >= _GAP_NOTE_SEC:
                    lines.append(f"（这里隔了 {int((m.timestamp - prev) // 60)} 分钟）")
                stamp = time.strftime("%H:%M", time.localtime(m.timestamp))
                mark = "（@你）" if m.is_at_me else ("（引用你的话）" if m.is_reply_to_me else "")
                lines.append(f"[{stamp}] {body}{mark}")
                prev = m.timestamp
            return "\n".join(lines)

        paras: list[str] = []
        cur_uid = ""
        cur_name = ""
        cur: list[str] = []
        prev = 0.0

        def flush() -> None:
            if cur:
                paras.append(f"【{cur_name}】\n" + "\n".join(cur))

        for m in self.messages:
            body = (m.text or "").replace("\n", " ").strip()
            if not body:
                continue
            # 换人、或同一人隔了很久，都另起一段——断点本身就是信息
            if m.user_id != cur_uid or (prev and m.timestamp - prev >= _GAP_NOTE_SEC):
                flush()
                cur_uid = m.user_id
                cur_name = m.user_name or m.user_id
                cur = []
            stamp = time.strftime("%H:%M", time.localtime(m.timestamp))
            mark = "（@你）" if m.is_at_me else ("（引用你的话）" if m.is_reply_to_me else "")
            cur.append(f"{stamp} {body}{mark}")
            prev = m.timestamp
        flush()
        return "\n\n".join(paras)

    def brief(self) -> str:
        return f"{self.session_id} {len(self.messages)}条/{self.speakers}人/{self.span_sec:.0f}s"


class Coalescer:
    """按会话攒消息，静默或到顶就交给回调。"""

    def __init__(self, cfg: BatchConfig, on_flush: Flusher) -> None:
        self.cfg = cfg
        self.on_flush = on_flush
        self._bufs: dict[str, Batch] = {}
        self._timers: dict[str, asyncio.Task[None]] = {}
        self.closing = False
        """退出中。此时不再排队等限速，最后一批直接发。"""

    # ---------- 外部接口 ----------

    async def push(self, msg: IncomingMessage, msg_id: int = 0) -> bool:
        """收一条消息。返回 True 表示这一批已经交出去了（立即模式或攒满）。"""
        sid = msg.session_id
        buf = self._bufs.get(sid)
        if buf is None:
            buf = Batch(
                session_id=sid,
                scope=msg.scope,
                group_id=msg.group_id,
                first_id=msg_id,
                last_id=msg_id,
                first_ts=msg.timestamp,
                last_ts=msg.timestamp,
            )
            self._bufs[sid] = buf

        buf.messages.append(msg)
        buf.last_ts = msg.timestamp
        if msg_id:
            buf.last_id = msg_id
            if not buf.first_id:
                buf.first_id = msg_id

        if not self.cfg.enabled:
            self._cancel(sid)
            await self._emit(buf)
            return True

        if (
            self.cfg.max_messages
            and len(buf.messages) >= self.cfg.max_messages
            and buf.retry_at <= time.time()
        ):
            log_.debug("攒满 %d 条，立即交出 %s", len(buf.messages), sid)
            self._cancel(sid)
            await self._emit(buf)
            return True

        self._reschedule(buf)
        return False

    def pending(self, session_id: str) -> int:
        buf = self._bufs.get(session_id)
        return len(buf.messages) if buf else 0

    def snapshot(self) -> list[dict[str, object]]:
        """给控制台看：哪些会话正在攒，攒了多少。"""
        now = time.time()
        return [
            {
                "session_id": b.session_id,
                "count": len(b.messages),
                "speakers": b.speakers,
                "waited": round(now - b.first_ts, 1),
            }
            for b in self._bufs.values()
        ]

    def defer(self, buf: Batch, delay: float) -> None:
        """额度没恢复：这批不丢，放回缓冲，等冷却结束再一起处理。

        限速管的是"开口次数"，不是"能处理多少条消息"。被推回来的这批会在
        冷却期间继续接新消息，到点一次性发出——所以连问十个问题也只是
        攒成一批，而不是丢掉八个。
        """
        sid = buf.session_id
        buf.deferred += 1

        current = self._bufs.get(sid)
        if current is not None and current is not buf:
            # 冷却期间又来了新消息、已经开张了新的一批：把旧的接回前面，保持时间顺序
            current.messages = buf.messages + current.messages
            current.first_id = buf.first_id or current.first_id
            current.first_ts = buf.first_ts or current.first_ts
            current.deferred = buf.deferred
            target = current
        else:
            self._bufs[sid] = buf
            target = buf

        target.retry_at = time.time() + delay
        self._cancel(sid)
        self._timers[sid] = asyncio.create_task(
            self._wait(target, max(0.1, delay)), name=f"batch:{sid}"
        )
        log_.debug(
            "限速推迟 %s，%.1fs 后重试（第 %d 次，积压 %d 条）",
            sid,
            delay,
            buf.deferred,
            len(target.messages),
        )

    async def flush_all(self) -> None:
        """退出前把没到点的缓冲全部交出去，别让消息烂在内存里。

        先把 closing 立起来：agent 那边看到它就不再看限速，最后一批不再排队。
        """
        self.closing = True
        for sid in list(self._bufs):
            buf = self._bufs.get(sid)
            if buf is not None:
                self._cancel(sid)
                await self._emit(buf)

    def close(self) -> None:
        for sid in list(self._timers):
            self._cancel(sid)

    # ---------- 内部 ----------

    def _delay(self, buf: Batch, now: float) -> float:
        # 被限速推回来过：必须等到额度恢复。中间来再多消息也不能提前，
        # 更不能被静默窗或 max_wait 覆盖——那样每条新消息都会重试一次。
        if buf.retry_at > now:
            return buf.retry_at - now
        quiet = self.cfg.private_quiet_sec if buf.scope == "private" else self.cfg.group_quiet_sec
        if self.cfg.max_wait_sec > 0:
            remain = buf.first_ts + self.cfg.max_wait_sec - now
            quiet = min(quiet, max(0.05, remain))
        return max(0.0, quiet)

    def _reschedule(self, buf: Batch) -> None:
        self._cancel(buf.session_id)
        delay = self._delay(buf, time.time())
        task = asyncio.create_task(
            self._wait(buf, delay), name=f"batch:{buf.session_id}"
        )
        self._timers[buf.session_id] = task

    async def _wait(self, buf: Batch, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        self._timers.pop(buf.session_id, None)
        await self._emit(buf)

    def _cancel(self, session_id: str) -> None:
        task = self._timers.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _emit(self, buf: Batch) -> None:
        # 同一会话已经换了新缓冲（或已被别的路径交出去），这批作废
        if self._bufs.get(buf.session_id) is not buf:
            return
        self._bufs.pop(buf.session_id, None)
        if not buf.messages:
            return
        try:
            await self.on_flush(buf)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log_.exception("交出对话失败 %s: %s", buf.brief(), exc)

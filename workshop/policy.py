"""触发策略与限速。

机器人最容易翻车的地方不是模型，是"什么时候该说话"。
这里把判断集中到一处，方便按群调教。
"""

from __future__ import annotations

import random
import time
from collections import defaultdict, deque

from .config import BotConfig, RateLimitConfig, TriggerConfig
from .events import IncomingMessage

WINDOW_SEC = 60.0
"""配额统计的滑动窗口长度（秒）。"""


class Trigger:
    def __init__(self, cfg: TriggerConfig, bot: BotConfig) -> None:
        self.cfg = cfg
        self.bot = bot
        self._keywords = [k for k in cfg.keywords if k]
        if bot.nickname:
            self._keywords.append(bot.nickname)

    def _hit_keyword(self, text: str) -> bool:
        return any(k and k in text for k in self._keywords)

    def hard(self, msg: IncomingMessage) -> tuple[bool, str]:
        """确定性判断：@ 了、叫了名字、私聊、被拉黑。不含随机。

        合批之后要拿它逐条扫一遍——"有人叫过它"和"掷骰子碰巧想说话"
        是两件事，不能混在一次随机里。
        """
        cfg = self.cfg

        if self.bot.self_id and msg.user_id == str(self.bot.self_id):
            return False, "self"
        if msg.user_id in cfg.ignore_users:
            return False, "blacklist"
        if not msg.text.strip():
            return False, "empty"

        if msg.scope == "private":
            mode = cfg.private
            if mode == "off":
                return False, "private_off"
            if mode == "always":
                return True, "private_always"
            return (True, "private_keyword") if self._hit_keyword(msg.text) else (False, "no_keyword")

        # 群聊
        mode = cfg.group
        if mode == "off":
            return False, "group_off"
        if msg.is_at_me or msg.is_reply_to_me:
            return True, "group_at_me"
        if mode == "always":
            return True, "group_always"
        # at 与 keyword 两种模式都把"叫到名字"视为点名：
        # 群里不 @ 直接喊名字是最常见的用法，漏掉会显得耳聋。
        if self._hit_keyword(msg.text):
            return True, "group_keyword"
        return False, "not_mentioned"

    def decide(self, msg: IncomingMessage) -> tuple[bool, str]:
        """返回 (是否回复, 原因)。原因用于日志排障。"""
        ok, reason = self.hard(msg)
        if ok:
            return True, reason
        # 只有群聊"没被叫到"才给随机插话的机会；私聊关键词没命中就是没命中
        if msg.scope == "group" and reason == "not_mentioned":
            return self._chance(msg)
        return False, reason

    def _chance(self, msg: IncomingMessage) -> tuple[bool, str]:
        p = self.cfg.group_chance
        if p <= 0:
            return False, "not_mentioned"
        if random.random() < p:
            return True, "chance"
        return False, "not_mentioned"


class RateLimiter:
    """按会话防连发、按人算配额。

    以前两者都挂在会话上，群里就成了"整群共用一个嗓子"：
    A 问一句、B 再问一句，第三句不管是冲谁来的都发不出去——
    谁先开口谁把额度用光。现在配额落到人头上，每个人每分钟各有各的份；
    只有最小间隔仍按整会话算，同一个群短时间连发几条容易被风控。
    """

    def __init__(self, cfg: RateLimitConfig) -> None:
        self.cfg = cfg
        self._last: dict[str, float] = {}
        """会话 → 上次开口的单调时间。只管最小间隔。"""
        self._quota: dict[str, deque[float]] = defaultdict(deque)
        """会话|人 → 本分钟已经回过的时刻。配额按这个算。"""

    @staticmethod
    def _key(session_id: str, speaker: str) -> str:
        """没给说话人就退化成整会话一份额度（协议端的旧行为）。"""
        return f"{session_id}|{speaker}" if speaker else session_id

    def allow(
        self,
        session_id: str,
        speaker: str = "",
        ignore_gap: bool = False,
    ) -> tuple[bool, str]:
        """ignore_gap：同一批按人拆出来的后续回复。

        那些回复是同一次"该开口"的产物，彼此间隔不到最小间隔是正常的；
        但配额照样计，免得拆批变成绕开限速的后门。
        """
        now = time.monotonic()

        if not ignore_gap:
            last = self._last.get(session_id)
            if last is not None and now - last < self.cfg.min_interval_sec:
                return False, "too_fast"

        q = self._quota[self._key(session_id, speaker)]
        while q and now - q[0] > WINDOW_SEC:
            q.popleft()
        if len(q) >= self.cfg.per_session_per_min:
            return False, "quota"

        q.append(now)
        self._last[session_id] = now
        return True, "ok"

    def cooldown(self, session_id: str) -> float:
        """距离可以再发还有多久（秒）。只算最小间隔，不看每分钟配额。"""
        last = self._last.get(session_id)
        if last is None:
            return 0.0
        return max(0.0, self.cfg.min_interval_sec - (time.monotonic() - last))

    def wait_until(self, session_id: str, speaker: str = "") -> float:
        """距离这个人下次真的能发言还有多久（秒），0 表示现在就行。

        和 cooldown 的区别：这里把每分钟配额也算进去。配额用完时要等最早的
        那次滑出 60 秒窗口，而不是干等 1.5 秒就重试——那样会被连着拒几十次。
        """
        now = time.monotonic()
        gap = self.cooldown(session_id)

        q = self._quota.get(self._key(session_id, speaker))
        if not q:
            return gap
        while q and now - q[0] > WINDOW_SEC:
            q.popleft()
        if len(q) >= self.cfg.per_session_per_min:
            quota = max(0.0, q[0] + WINDOW_SEC - now)
            return max(gap, quota)
        return gap

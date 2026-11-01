"""智能体内核：把一段时间里的对话变成一次回应。

流水线：
    去重 -> 全量落库 -> 入合流缓冲 -> 静默/到顶交出 -> 意愿门控
    -> 组装上下文 -> 调模型 -> 清洗与分段 -> 发送 -> 落库 -> 必要时压缩摘要

关键约束：
- 没点名的消息也要落库。模型得先看得见整段对话，才可能"整体回应"。
- 同一会话串行处理，避免两批并发生成导致上下文错位、回复乱序。
- 人格库只读：本模块从不写 index.md / wiki。
- 模型输出必须清洗，否则 QQ 里会出现 markdown 星号和自述旁白。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

from . import inject, library, log
from .batch import Batch, Coalescer
from .config import Config
from .drift import DriftEvent, DriftStore, knobs_from
from .events import IncomingMessage, OutgoingMessage
from .gate import Decision, Gate
from .llm import LLMClient, LLMError
from .memory import Memory
from .memwiki import MemWiki
from .persona import Persona
from .policy import RateLimiter
from .relation import level_of, voice as relation_voice

log_ = log.get("agent")

Sender = Callable[[OutgoingMessage], Awaitable[None]]

# 调试台用的独立会话，不污染真实聊天记忆
PREVIEW_SESSION = "webui:debug"

# 模型偶尔会把整句话用引号包起来，或者加动作旁白，这里统一剥掉。
_QUOTE_WRAP = re.compile(r'^["“「『](.*)["”」』]$', re.DOTALL)
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_PAREN_ACTION = re.compile(r"[（(](?:微笑|叹气|笑|沉默|停顿|点头|摇头|皱眉|歪头|想了想|顿了顿)[）)]")
_LEAD_NAME = re.compile(r"^\s*(?:assistant|AI)\s*[:：]\s*", re.IGNORECASE)


def _lead_name_re(name: str) -> re.Pattern[str]:
    """生成剥离"角色名前缀"的正则。

    模型偶尔会在回复开头加上「某某：」。名字不写死在代码里——由调用方传入
    人格库的昵称；名字为空时只兜住 assistant / AI 这类通用前缀。
    name 走 re.escape，避免昵称里的 . * ( ) 把正则炸掉。
    """
    name = (name or "").strip()
    if not name:
        return _LEAD_NAME
    return re.compile(rf"^\s*(?:{re.escape(name)}|assistant|AI)\s*[:：]\s*", re.IGNORECASE)


SUMMARY_SYSTEM = (
    "你在压缩一段聊天记录，供后续对话复用。只输出摘要正文，不要开场白。\n"
    "保留：双方关系与称呼、发生过的事件、明确的承诺或约定、情绪基调与未解决的话题。\n"
    "丢弃：寒暄、重复内容、单纯的情绪感叹。\n"
    "要求：中文，第三人称，不分点，不超过 200 字。"
)

# 关系分层的注入文本。写"分寸"不写台词——具体怎么说话归人格库。
# 这里只回答一个问题：对面这个人，值不值得你认真对待。
OWNER_BLOCK = (
    "## 对方是谁\n"
    "「{name}」就是把你写出来、一直在调教你的人——你的拥有者。\n"
    "对他不必客套，也不用演：可以直说、可以顶嘴、可以懒得敷衍。"
)

OWNER_IN_GROUP = "现在是在群里，旁边还有别人。跟他的关系不用装作不认识，但别拿来讲，也别把他的事摊开。"

FAMILIAR_BLOCK = (
    "## 对方是谁\n"
    "「{name}」跟你聊过 {turns} 轮，算个熟人。比陌生人松一点，但还没到能交底的地步。"
)

STRANGER_BLOCK = (
    "## 对方是谁\n"
    "「{name}」不是你的拥有者，是个不熟的人。礼貌，但留着：不交底、不替人做决定、不主动示好。"
)

GROUP_NOTE = "这条是群聊，旁边还有别人在看你说话。"

BATCH_BLOCK = (
    "## 这一轮怎么回\n"
    "下面给你的是一段时间里累积的整段对话，不是单独一句话。\n"
    "- 把它当成一件事来回应：只接真正轮到你的那部分，不必每句都应。\n"
    "- 不要逐条复述，不要按发言顺序一一作答，也不要把每句话都接一遍。\n"
    "- 一次只说一轮。篇幅看内容，通常一到三句；确实需要分条讲再分段。\n"
    "- 别人之间的话，你插不进去就不插，装着没看见也可以。\n"
)

SILENCE_TOKEN = "<沉默>"
"""模型用它表示"这一轮我不接"。整条回复只有这个词时，不发送、不落库。"""

MAX_DEFER = 5
"""一批最多被限速退回来几次。超过就放弃，防止等待时间算错时无限转圈。"""

REPLY_FREEDOM = (
    "## 回不回，你说了算\n"
    "有人叫你不代表你必须答。懒得答、不想答、觉得没意思，都可以直接拒绝。\n"
    "拒绝就一句话，别解释、别找补、别道歉——「不说」「不想回答」「滚」「烦」「你自己查」。\n"
    "别每次都同一个词，看你当时的心情。\n"
    "但大多数时候该答还是答：拒绝是态度，不是习惯。\n"
)

SILENCE_HINT = f"真的一句都不想给，就只输出 {SILENCE_TOKEN}，别的什么都别写。\n"
"""只在允许沉默时追加。关掉它，她至少得给句话，但可以不客气。"""


CARRY_BLOCK = (
    "## 他和你的私下对话（只作参考，不要在这里复述）\n"
    "{summary}\n"
    "只在接住他的情绪、或他提到旧事时用。别人问起这些内容，不要讲。"
)


class Agent:
    def __init__(
        self,
        cfg: Config,
        persona: Persona,
        memory: Memory,
        llm: LLMClient,
        sender: Sender,
        requires_context: bool = False,
    ) -> None:
        self.cfg = cfg
        self.persona = persona
        # 剥"角色名前缀"时用的名字：优先 [bot] nickname，退回到人格库目录名。
        # 从配置来，不写死在代码里——换个人格库不用改源码。
        self.self_name = (
            (getattr(cfg.bot, "nickname", "") or "").strip() or persona.root.name
        )
        # 记忆层：只读 persona/memory/，命中才注入。开关关着时不建对象，
        # 行为与旧版逐字节一致（回归测试靠这条）。
        self.memwiki = MemWiki(persona.root) if cfg.memory.wiki_enabled else None
        self.memory = memory
        self.llm = llm
        self.sender = sender
        self.requires_context = requires_context
        """通道是否要求每段回复都挂在原消息上。

        官方平台只认"被动回复"，不带 msg_id 就变成主动消息并受限频约束，
        所以分段时必须每段都带同一个 msg_id（平台靠 msg_seq 区分）。
        """
        self.gate = Gate(cfg.gate, cfg.trigger, cfg.bot, persona, llm)
        self.trigger = self.gate.trigger
        """旧入口，指向门控里的硬规则层。保留是为了不弄断已有脚本。"""
        self.limiter = RateLimiter(cfg.rate_limit)
        self.coalescer = Coalescer(cfg.batch, self._flush)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._summarizing: set[str] = set()
        """正在压缩摘要的会话。压缩是一次额外的模型往返（最长 60s），
        不能让同一会话的两条消息同时压，否则互相覆盖摘要和 upto_id。"""
        self._last_spoke: dict[str, float] = {}
        """每个会话上一次开口的时间。自冷却用它，防止它对着同一群人连珠炮。"""

        self.drift: DriftStore | None = None
        """人格漂移。关着时是 None，门控完全按配置里的旋钮走，行为与旧版一致。"""
        self._pending: dict[str, float] = {}
        """她在某会话说完话、还没等到回音。下一条消息进来时判"被接"还是"被忽略"。"""
        self._drift_dirty = 0
        """自上次刷盘以来记了几个事件。攒够再写，别每条消息都碰磁盘。"""
        if cfg.drift.enabled:
            self.drift = self._open_drift(cfg)
            self.gate.bind_drift(self.drift.state.knobs)
            log_.info(
                "人格漂移已启用 | 库 %s | 旋钮 %s | 累计 %d 次互动",
                self.persona.root.name,
                self.drift.state.knobs.as_dict(),
                self.drift.state.turns,
            )

        self.stats = {
            "in": 0,
            "replied": 0,
            "skipped": 0,
            "errors": 0,
            "silent": 0,
            "deferred": 0,
        }
    # ---------- 人格漂移 ----------

    def _open_drift(self, cfg: Config) -> DriftStore:
        """开一份属于当前人格库的漂移状态。

        两个关键点，缺一个就会串库：
        - 落点是库内 data/drift.json，不跟程序目录走。切库时两个角色的
          状态必须物理分家，否则甲的怕生会被当成乙的。
        - 基线取本库 config 里的旋钮（preferences.toml 覆盖后的值），
          不取出厂常量。基线写死的话，漂移一开工就把人工调过的性格抹平。
        """
        store = DriftStore(
            library.drift_path(self.persona.root),
            cfg.drift,
            baseline=knobs_from(cfg.gate),
        )
        if store.baseline_moved:
            log_.info(
                "漂移基线已跟随库配置 | %s → %s",
                self.persona.root.name,
                store.baseline.as_dict(),
            )
        return store

    def _drift(
        self,
        kind: str,
        *,
        ts: float,
        gap_hours: float = 0.0,
        by_owner: bool = False,
        scope: str = "group",
    ) -> None:
        """记一个事件，把新旋钮推给门控。

        内核是纯函数（bridge/drift.py），这里只负责两件它管不着的事：
        什么算一个事件、什么时候刷盘。关掉漂移时整个函数什么都不做。
        """
        if self.drift is None:
            return
        try:
            state = self.drift.record(
                DriftEvent(kind=kind, ts=ts, gap_hours=gap_hours, by_owner=by_owner, scope=scope)
            )
        except ValueError as exc:
            log_.warning("漂移事件被拒：%s", exc)
            return
        self.gate.bind_drift(state.knobs)
        self._drift_dirty += 1
        if self._drift_dirty >= 10:
            self.drift.save()
            self._drift_dirty = 0
            log_.debug("漂移落盘 %s | %s", state.last_kind, state.knobs.as_dict())

    def flush_drift(self) -> None:
        """把攒着没写的漂移落盘。退出前调一次。

        `last_ts` 是靠 touch 在内存里推进的，不占 dirty 计数（否则每收一条消息
        就要写一次盘）。所以这里不能只看 dirty：只要有 last_ts 就说明真收过消息，
        得落，否则重启后第一条消息会把"上次互动到现在"重新判一遍。
        """
        if self.drift is not None and (self._drift_dirty or self.drift.state.last_ts):
            self.drift.save()
            self._drift_dirty = 0

    # ---------- 主入口 ----------

    async def handle(self, msg: IncomingMessage) -> None:
        """收一条消息。只做落库和入缓冲，不在这里生成回复。

        生成交给合流器：等这一轮话说完了，再一次性回应。
        """
        self.stats["in"] += 1

        key = f"{msg.channel}:{msg.message_id}" if msg.message_id else None
        if not self.memory.claim(key):
            log_.debug("重复事件，跳过 %s", key)
            return

        drop = self._drop(msg)
        if drop:
            self.stats["skipped"] += 1
            log_.debug("不记录(%s) %s", drop, msg.brief())
            return

        text = msg.text[: self.cfg.trigger.max_input_chars]
        first_time = self.memory.profile(msg.user_id) is None
        self.memory.touch_user(msg.user_id, msg.user_name)
        if first_time:
            # 新面孔第一次说话时把 id 打出来，填 owner.ids 时照着抄
            log_.info(
                "首次出现的用户 id=%s 昵称=%r（要设为拥有者就填进 owner.ids）",
                msg.user_id,
                msg.user_name,
            )

        # 全量落库。没点名的也记——模型得先看得见整段对话，
        # 才可能"总结到一块"再回应。这也是本次改造的前提。
        row_id = self.memory.append(
            msg.session_id, "user", self._annotate(msg, text), msg.message_id, msg.user_id
        )

        if self.drift is not None:
            self._track(msg)

        await self.coalescer.push(msg, row_id)

    def _track(self, msg: IncomingMessage) -> None:
        """从一条进来的消息里读漂移事件。只在这里判，别处不重复判。

        两种情况：
            空窗  离上次收到消息超过 idle_hours —— 隔得越久影响越大
            回音  她上一条话音刚落，这条就是回音：接住了还是没接

        两条判完都要 touch：last_ts 记的是"最后一次收到消息"，不是
        "最后一次产生事件"。漏了这一步，从没开过口的角色会被判成从未久别，
        而开口过的角色又会在密集聊天里被误判成空窗。
        """
        state = self.drift.state
        gap_h = (msg.timestamp - state.last_ts) / 3600.0 if state.last_ts else 0.0
        if gap_h >= self.cfg.drift.idle_hours:
            self._drift("idle", ts=msg.timestamp, gap_hours=gap_h, scope=msg.scope)

        # 只看紧接着的那一条。她在群里说一句，下一个人接不接，就是最真实的反馈。
        if self._pending.pop(msg.session_id, None) is not None:
            hit = msg.is_reply_to_me or msg.is_at_me
            self._drift(
                "replied" if hit else "ignored",
                ts=msg.timestamp,
                by_owner=self._is_owner(msg),
                scope=msg.scope,
            )

        self.drift.touch(msg.timestamp)

    def _drop(self, msg: IncomingMessage) -> str:
        """连记录都不必的消息：自己发的、拉黑的、空文本。

        这些不该进记忆库，否则上下文里混进噪声，还会污染门控的打分。
        """
        if self.cfg.bot.self_id and msg.user_id == str(self.cfg.bot.self_id):
            return "self"
        if msg.user_id in self.cfg.trigger.ignore_users:
            return "blacklist"
        if not msg.text.strip():
            return "empty"
        return ""

    async def _flush(self, batch: Batch) -> None:
        """合流器交来一批对话。这是唯一的生成入口。"""
        lock = self._locks[batch.session_id]
        if lock.locked():
            log_.debug("会话忙，排队 session=%s", batch.session_id)

        outcome, wait = "denied", 0.0
        replied_any = False
        async with lock:
            try:
                parts = self._fanout(batch)
                for i, part in enumerate(parts):
                    outcome, wait = await self._reply(part, skip_rate=i > 0)
                    if outcome == "replied":
                        replied_any = True
                    if outcome == "deferred":
                        break
                    # 连着两条挂在不同的 msg_id 上，拉开一点再发，别挤成一坨
                    if i < len(parts) - 1:
                        await asyncio.sleep(0.6)
            except asyncio.CancelledError:
                raise
            except LLMError as exc:
                self.stats["errors"] += 1
                log_.error("模型调用失败 session=%s: %s", batch.session_id, exc)
            except Exception as exc:  # noqa: BLE001
                self.stats["errors"] += 1
                log_.exception("处理异常 session=%s: %s", batch.session_id, exc)

        if outcome == "deferred":
            # 额度没恢复：这批退回合流器继续攒，到点一块儿说。
            # 退太多次说明等待时间算错了，别无限转下去。
            # 注意退的是原批次，不是拆出来的那一片——否则其他叫过她的人就丢了。
            if batch.deferred >= MAX_DEFER:
                log_.warning(
                    "推迟 %d 次仍未发出，放弃 session=%s（%d 条已留上下文）",
                    batch.deferred,
                    batch.session_id,
                    len(batch.messages),
                )
                return
            self.coalescer.defer(batch, wait)
            return

        # 压缩摘要放在会话锁之外：它是第二次模型往返，压在锁上会让这个会话
        # 的下一条消息白等几十秒。同一会话的并发压缩由 _summarizing 挡住。
        if replied_any:
            await self._maybe_summarize(batch.session_id)

    def _fanout(self, batch: Batch) -> list[Batch]:
        """群里几个人同时叫她时，按人拆成几批，各回各的。

        官方通道的被动回复只能挂一条 msg_id。几个人一起叫还合成一次回复，
        就注定有人被冒名——答案挂到别人名下（日志里看着"回给 Master"，
        内容却在答薇尔）。拆开之后每批只有一个点名的人，回谁就是谁。

        没点名的消息当背景每批都带上，她得知道旁边在聊什么；
        其他人的点名不重复带进来，免得又混。
        """
        if batch.scope != "group":
            return [batch]
        callers = batch.callers
        if len(callers) <= 1:
            return [batch]

        caller_ids = {c.user_id for c in callers}
        bystanders = [m for m in batch.messages if m.user_id not in caller_ids]

        parts: list[Batch] = []
        for caller in callers:
            mine = [m for m in batch.messages if m.user_id == caller.user_id]
            msgs = sorted(mine + bystanders, key=lambda m: m.timestamp)
            if not msgs:
                continue
            parts.append(
                Batch(
                    session_id=batch.session_id,
                    scope=batch.scope,
                    group_id=batch.group_id,
                    messages=msgs,
                    first_id=batch.first_id,
                    last_id=batch.last_id,
                    first_ts=batch.first_ts,
                    last_ts=batch.last_ts,
                )
            )
        if len(parts) <= 1:
            return [batch]

        log_.info(
            "群里 %d 人同时叫她，拆成 %d 批分别回：%s",
            len(parts),
            len(parts),
            "、".join(c.user_name or c.user_id for c in callers),
        )
        return parts

    # ---------- 回复生成 ----------

    async def _reply(self, batch: Batch, skip_rate: bool = False) -> tuple[str, float]:
        """对一批对话生成并发送一次回应。

        返回 (结果, 等待秒数)。结果取值：
            replied  发了
            denied   不该开口（门控拦下）
            deferred 额度没恢复，这批要退回去排队
            silent   她选择不接
            empty    清洗后为空

        skip_rate：同一批按人拆出来的第 2、3 份。跳过最小间隔——它们本来就是
        同一次开口的产物；但配额照计，别让拆批变成绕开限速的后门。
        """
        started = time.monotonic()
        sid = batch.session_id

        last = self._last_spoke.get(sid)
        since_last_spoke = (time.time() - last) if last else 1e9

        decision = await self.gate.decide(batch, since_last_spoke)
        if not decision.speak:
            # 一群人热闹、没人叫她、她也一个字没说 —— 这是"没位置"，不是"被忽略"。
            # 被忽略是说了没人接；这里连口都没开，怕生涨得更快。
            if batch.speakers >= 3 and not batch.named and not batch.replied_to:
                self._drift("crowded", ts=time.time(), scope=batch.scope)
            self.stats["skipped"] += 1
            log_.debug("不开口(%s) %s", decision.reason, batch.brief())
            return "denied", 0.0

        # 配额按人算：群里 A 和 B 各有各的两条，谁也不会把谁的额度用光。
        # 最小间隔仍按整会话——同一个群短时间连发几条容易被风控。
        speaker = batch.anchor.user_id
        allow, why = self.limiter.allow(sid, speaker, ignore_gap=skip_rate)
        if not allow:
            if not self.coalescer.closing:
                # 不丢。退回缓冲继续攒，等额度恢复了连新消息一起说。
                wait = self.limiter.wait_until(sid, speaker)
                self.stats["deferred"] += 1
                log_.info(
                    "限速(%s) %s：积压 %d 条，%.0fs 后一起回",
                    why,
                    sid,
                    len(batch.messages),
                    wait,
                )
                return "deferred", wait
            log_.info("退出中，无视限速发出最后一批 %s", sid)

        # 是被叫到才开口的，记一笔 named。主动插话不算——没人叫她自己凑上去
        # 和被人点名，对参数的意义完全不同。
        if decision.by == "rule" or batch.named or batch.replied_to:
            self._drift(
                "named",
                ts=time.time(),
                by_owner=self._is_owner(batch.anchor),
                scope=batch.scope,
            )

        messages = await self._build_messages(batch, decision)

        raw = await self.llm.chat(messages)
        if self._is_silence(raw):
            # 她选择这一轮一个字都不给。东西不发，但对话照样留在记忆里，
            # 下一轮她还记得刚才有人叫过她。
            self.stats["silent"] += 1
            log_.info("她不接 session=%s | %d 条已留上下文", sid, len(batch.messages))
            return "silent", 0.0

        reply = self._sanitize(raw, self.self_name)[: self.cfg.reply.max_chars]
        if not reply:
            log_.warning("清洗后为空，放弃回复 session=%s", sid)
            return "empty", 0.0

        # 被动回复只认一条 msg_id，合批后挂在"叫她的人"那条上。
        # 不能用 last：群里后面有人插一句闲话，回复就挂错了人。
        anchor = batch.anchor
        chunks = self._split(
            reply,
            self.cfg.reply.split_threshold,
            self.cfg.reply.max_chunks,
            self.cfg.reply.split_on_blank,
        )
        for i, chunk in enumerate(chunks):
            if self.requires_context:
                # 官方平台：每一段都要挂在原消息上，否则退化成主动消息
                reply_to = anchor.message_id
            elif i == 0 and self.cfg.reply.quote_when_group and batch.scope == "group":
                reply_to = anchor.message_id
            else:
                reply_to = None

            out = OutgoingMessage(
                session_id=sid,
                text=chunk,
                scope=batch.scope,
                group_id=batch.group_id,
                user_id=anchor.user_id,
                reply_to=reply_to,
            )
            await self.sender(out)
            if len(chunks) > 1:
                await asyncio.sleep(0.4)

        self.memory.append(sid, "assistant", reply)
        self._last_spoke[sid] = time.time()
        if self.drift is not None:
            # 挂个待判：下一条进来的消息决定这次开口是"被接住"还是"没人理"
            self._pending[sid] = time.time()
        self.stats["replied"] += 1
        cost = (time.monotonic() - started) * 1000
        # 关系档也打出来：不认人的时候，第一眼看的就是这行是 owner 还是 stranger
        level = self._level(anchor)[0]
        log_.info(
            "回复 %s → %s | %.0fms | %d 字 | 合 %d 条 · %s | %s | %s",
            sid,
            anchor.user_name or anchor.user_id,
            cost,
            len(reply),
            len(batch.messages),
            level,
            decision.by,
            reply.replace("\n", " ")[:50],
        )
        return "replied", 0.0

    def _annotate(self, msg: IncomingMessage, text: str) -> str:
        """群聊里给用户消息加说话人前缀，否则模型分不清谁在说话。"""
        if msg.scope == "private":
            return text
        name = msg.user_name or msg.user_id
        return f"{name}: {text}"

    def _is_owner(self, msg: IncomingMessage) -> bool:
        return self._level(msg)[0] == "owner"

    def _has_owner(self, batch: Batch) -> bool:
        """批内是否有拥有者开口。决定要不要带上私聊摘要。"""
        ids = set(self.cfg.owner.ids)
        return bool(ids) and any(m.user_id in ids for m in batch.messages)

    def _level(self, msg: IncomingMessage) -> tuple[str, int]:
        """返回 (主人/熟人/陌生人, 已聊轮数)。判定规则与素材台共用 relation 模块。

        记忆层启用后，熟人档看的是"记忆里有没有这个人"，不再看聊过多少条：
        刷了一百句的路人仍是陌生人，只聊过三次但记住了他生日就是熟人。
        """
        owner = self.cfg.owner
        turns = 0
        if owner.familiar_turns > 0:
            prof = self.memory.profile(msg.user_id)
            turns = int(prof["turns"]) if prof else 0
        has_mem = None if self.memwiki is None else self.memwiki.has_person(msg.user_id)
        return level_of(
            msg.user_id, owner.ids, turns, owner.familiar_turns, has_memory=has_mem
        ), turns

    def _voice(self, level: str) -> str:
        """按关系档取"怎么说"。只换这一节，人格那几页原样注入，基调不动。"""
        return relation_voice(self.persona, level)

    def _relation(self, msg: IncomingMessage) -> tuple[str, set[str]]:
        """主人 / 熟人 / 陌生人，拼成注入的身份块。

        第二个返回值是本轮已精细注入的关系页小节归属。
        调用方据此跳过整页——整页会把三档同时塞进去，和这一节打架。
        """
        owner = self.cfg.owner
        name = msg.user_name or msg.user_id
        level, turns = self._level(msg)

        if level == "owner":
            block = OWNER_BLOCK.format(name=name)
            call = owner.call_me.strip()
            if call:
                block += f"你称他「{call}」。"
            if msg.scope == "group":
                block += "\n" + OWNER_IN_GROUP
        elif level == "familiar":
            block = FAMILIAR_BLOCK.format(name=name, turns=turns)
            if msg.scope == "group":
                block += "\n" + GROUP_NOTE
        else:
            block = STRANGER_BLOCK.format(name=name)
            if msg.scope == "group":
                block += "\n" + GROUP_NOTE

        voice = self._voice(level)
        if voice:
            return f"{block}\n\n{voice}", {"relationships.md"}
        return block, set()

    def _crowd(
        self,
        batch: Batch,
        anchor: IncomingMessage,
        callers: list[IncomingMessage],
    ) -> str:
        """这一段里有谁，各自是什么身份。

        官方通道拿不到真昵称，一排「群友5E812A」在模型眼里长得差不多，
        容易把甲说的话算到乙头上。这里把人和角色对上号：
        谁在跟她说话、谁也叫了她、谁只是旁边聊自己的。
        """
        people: dict[str, IncomingMessage] = {}
        for m in batch.messages:
            people.setdefault(m.user_id, m)
        if len(people) <= 1:
            return ""

        caller_ids = {c.user_id for c in callers}
        # 排序确定：正在说话的人排第一，其次也叫了她的人，最后是旁听的。
        # 人多时只留 8 条——这一块是给模型分人用的，不该盖过对话本身。
        order: list[str] = [anchor.user_id]
        for c in callers:
            if c.user_id not in order:
                order.append(c.user_id)
        for uid in people:
            if uid not in order:
                order.append(uid)

        rows: list[str] = []
        for uid in order[:8]:
            m = people.get(uid)
            if m is None:
                continue
            name = m.user_name or uid
            if uid == anchor.user_id:
                rows.append(f"- {name}：正在跟你说话，你这一句是回他")
            elif uid in caller_ids:
                rows.append(f"- {name}：也叫了你")
            else:
                rows.append(f"- {name}：在旁边，没叫你")
        if not rows:
            return ""
        return "## 这一段里有谁\n" + "\n".join(rows)

    def _member_block(self, anchor: IncomingMessage, past: list[dict]) -> str:
        """她和这个人单独说过的话。

        群聊里她看到的是一整锅对话，问她"上次你跟我说什么"是答不出来的。
        这段按人捞出来、标明是他说的还是她说，让她对具体的人有连续的记忆。
        只作参考，且明令不许当众复述——群里有别人在听。
        """
        name = anchor.user_name or anchor.user_id
        prefix = f"{name}: "
        lines: list[str] = []
        for r in past:
            body = (r.get("content") or "").replace("\n", " ").strip()
            if not body:
                continue
            # 落库时给群消息加过"名字: "前缀，这里剥掉，否则读起来是"他：群友5E812A: xxx"
            if r.get("role") != "assistant" and body.startswith(prefix):
                body = body[len(prefix) :]
            who = "你" if r.get("role") == "assistant" else "他"
            stamp = time.strftime("%H:%M", time.localtime(float(r.get("ts") or 0)))
            lines.append(f"{stamp} {who}：{body}")
        if not lines:
            return ""
        return (
            f"## 你和「{name}」之前单独说过的\n"
            "（只用来记住聊到哪儿了。这是过去的事，不要当众复述，也别重新回答一遍）\n"
            + "\n".join(lines)
        )

    def _owner_private_summary(self) -> str:
        """主人与她的私聊摘要。官方通道私聊与群聊是两个 openid，逐个试。"""
        for oid in self.cfg.owner.ids:
            text, _ = self.memory.get_summary(f"private:{oid}")
            if text.strip():
                return text.strip()[:600]
        return ""

    async def _build_messages(self, batch: Batch, decision: Decision) -> list[dict[str, str]]:
        sid = batch.session_id
        summary, upto = self.memory.get_summary(sid)
        anchor = batch.anchor
        who = anchor.user_name or anchor.user_id

        if batch.scope == "group":
            callers = batch.callers
            if len(callers) > 1:
                names = "、".join(f"「{c.user_name or c.user_id}」" for c in callers)
                scene = (
                    f"## 当前场景\n"
                    f"这是群聊（群号 {batch.group_id}），你在跟一群人说话。"
                    f"刚才 {names} 都在叫你，不止一个人。"
                    f"挑真正需要你答的那句接，别挨个点名作答——另一句当没听见也可以。"
                    f"不要替别人说话，也不要复述群成员名字列表。"
                )
            else:
                scene = (
                    f"## 当前场景\n"
                    f"这是群聊（群号 {batch.group_id}），你在跟一群人说话。"
                    f"现在叫你的是「{who}」。"
                    f"只回应该说的那部分，不要替别人说话，也不要复述群成员名字列表。"
                )
            crowd = self._crowd(batch, anchor, callers)
            if crowd:
                scene += "\n\n" + crowd
            # 整群共享一条上下文，人一多就分不清跟谁聊到哪儿了。
            # 按人抽一段单独补给她，隔着几百条也能接上话。
            past = self.memory.member_exchanges(
                sid, anchor.user_id, before_id=batch.first_id or 1
            )
            if past:
                recall = self._member_block(anchor, past)
                if recall:
                    scene += "\n\n" + recall
        else:
            scene = f"## 当前场景\n这是私聊，对方是「{who}」。"

        relation_block, sections_used = self._relation(anchor)
        extra = scene + "\n\n" + relation_block

        # 群里主人开口时，把他私聊里聊过的带过来，让她记得这段交情。
        # 单向、只在他说话时生效——不在别人面前暴露私聊内容。
        if batch.scope == "group" and self.cfg.owner.carry_summary and self._has_owner(batch):
            carried = self._owner_private_summary()
            if carried:
                extra += "\n\n" + CARRY_BLOCK.format(summary=carried)

        system = self.persona.build_system_prompt(
            batch.text(),
            extra=extra,
            exclude_pages=inject.excluded_pages(sections_used=sections_used),
        )
        if summary:
            system += f"\n\n## 你们之前聊过的（摘要）\n{summary}"
        # 记忆层：命中才出现这一段。未命中不加一个字，日常闲聊不受影响。
        if self.memwiki is not None:
            recall = self.memwiki.render(
                batch.text(), budget=self.cfg.memory.wiki_budget
            )
            if recall:
                system += "\n\n" + recall

        # 这一轮怎么回：整体回，不逐句接
        system += "\n\n" + BATCH_BLOCK
        if decision.hint:
            system += f"- {decision.hint}\n"
        system += "\n" + REPLY_FREEDOM
        if self.cfg.gate.allow_silence:
            system += SILENCE_HINT

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        # 历史按 id 边界取：本批第一条之前、摘要覆盖点之后。
        # 不能"取最后 N 条再删几条"——批内消息随时可能被新消息追加，会切错。
        rows = self.memory.recent_before(
            sid,
            batch.first_id or 1,
            self.cfg.memory.recent_turns,
            after_id=upto,
        )
        for r in rows:
            messages.append({"role": r["role"], "content": r["content"]})

        block = batch.render(with_name=(batch.scope == "group"))
        messages.append({"role": "user", "content": f"（以下是刚刚的对话）\n{block}"})
        return messages

    # ---------- 输出清洗 ----------

    @staticmethod
    def _is_silence(text: str) -> bool:
        """整条回复就是沉默标记时，视为"这一轮不接"。

        只有整条等于标记才认。她要是写了别的内容、只是在话里带了标记，
        那就是在说话，照常发出去——别因为她用了这个词就把整条吞掉。
        """
        s = (text or "").strip().strip("*`\"'「」“”‘’《》（）()。.，,！!~ \n")
        return s in {SILENCE_TOKEN, "沉默", "不说话", "silence", "<silence>"}

    @staticmethod
    def _sanitize(text: str, name: str = "") -> str:
        s = text.strip()
        # 去掉模型自带的角色名前缀
        s = _lead_name_re(name).sub("", s)
        # 去掉 markdown 粗体（QQ 不渲染）
        s = _BOLD.sub(r"\1", s)
        # 去掉典型的动作旁白，如（微笑）
        s = _PAREN_ACTION.sub("", s)
        # 去掉整段被引号包裹的情况（成对出现才剥）
        m = _QUOTE_WRAP.match(s)
        if m and not s[0] in "（(":
            inner = m.group(1).strip()
            if inner.count("「") == inner.count("」") and inner.count("“") == inner.count("”"):
                s = inner
        # 去掉残留的 markdown 标题与列表符号
        lines = []
        for line in s.splitlines():
            line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
            line = re.sub(r"^\s{0,3}[-*+]\s+", "", line)
            lines.append(line.rstrip())
        s = "\n".join(lines).strip()
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s

    @staticmethod
    def _split(
        text: str,
        threshold: int,
        max_chunks: int = 4,
        on_blank: bool = True,
    ) -> list[str]:
        """把一条回复拆成要发的几条消息。

        和以前不一样：以前只看总长度，两百字以内哪怕分了三段也全挤在一个气泡里。
        现在先按她自己分的段拆——聊天里分段就是"分几条说"，真人也这么发。
        长度只是兜底：单段太长才硬切，免得被平台截断。

        只有一段、又不超长时仍然只发一条，不为了拆而拆。
        """
        body = text.strip()
        if not body:
            return []

        if on_blank:
            paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        else:
            paras = [body]
        if not paras:
            return []

        out: list[str] = []
        for para in paras:
            while len(para) > threshold:
                out.append(para[:threshold])
                para = para[threshold:]
            if para:
                out.append(para)

        if len(out) > max_chunks:
            # 条数超了：把后几段并回一条，宁可长点也别丢内容
            keep = out[: max_chunks - 1]
            merged = "\n".join(out[max_chunks - 1 :])
            out = keep + [merged]
        return out

    def apply_config(self, cfg: Config) -> None:
        """热更新可热的部分。通道与凭证不在这里处理，需要重启。"""
        self.cfg = cfg
        self.gate = Gate(cfg.gate, cfg.trigger, cfg.bot, self.persona, self.llm)
        self.trigger = self.gate.trigger
        self.limiter = RateLimiter(cfg.rate_limit)
        self.coalescer.cfg = cfg.batch
        self.llm.reload(cfg.llm)
        # 记忆层开关在面板上就能改，检索器得跟着重建：
        # 关掉时置 None，注入点立刻退回旧行为，不用重启整个机器人
        self.memwiki = MemWiki(self.persona.root) if cfg.memory.wiki_enabled else None

        # 漂移同样支持热切换：新开的门控实例要重新挂上旋钮，
        # 关掉则解绑，旋钮立刻回到 preferences.toml 里的值。
        if cfg.drift.enabled and self.drift is None:
            self.drift = self._open_drift(cfg)
        elif not cfg.drift.enabled:
            # 关之前先落一次：攒着没刷的那几个事件不该随解绑一起消失。
            # 没收到过消息就别建文件，免得平白多一个只有出厂值的 drift.json。
            if self.drift is not None and self.drift.state.last_ts:
                self.drift.save()
                self._drift_dirty = 0
            self.drift = None
        if self.drift is not None:
            self.drift.cfg = cfg.drift
            # 库里的性格旋钮被改过：回归目标跟着挪，但近况不重置。
            # 用 reset 就是"改个配置她就失忆"，那不该是保存配置的副作用。
            new_base = knobs_from(cfg.gate)
            if self.drift.baseline != new_base:
                self.drift.rebase(new_base)
                log_.info("漂移基线已更新 | %s", new_base.as_dict())
            self.gate.bind_drift(self.drift.state.knobs)
            log_.info("漂移热更新 | 旋钮 %s", self.drift.state.knobs.as_dict())

    # ---------- 调试台预览 ----------

    async def preview(self, text: str, remember: bool = False) -> dict:
        """不经过 QQ，直接看"这一句会被怎么处理"。

        返回回复、实际注入的 system prompt、命中的 wiki 页、耗时。
        调人格时看这个，比看日志快得多。
        """
        started = time.monotonic()
        extra = "## 当前场景\n这是网页调试台的单轮预览，不涉及真实 QQ 会话，不必寒暄客套。"
        system = self.persona.build_system_prompt(text, extra=extra)

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if remember:
            for item in self.memory.recent(PREVIEW_SESSION, self.cfg.memory.recent_turns):
                messages.append(item)
        messages.append({"role": "user", "content": text})

        reply = self._sanitize(await self.llm.chat(messages), self.self_name)

        if remember:
            self.memory.append(PREVIEW_SESSION, "user", text)
            self.memory.append(PREVIEW_SESSION, "assistant", reply)

        return {
            "reply": reply,
            "system_prompt": system,
            "pages": self.persona.pick_pages(text),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "model": self.cfg.llm.model,
            "chars": len(reply),
        }

    def clear_preview(self) -> None:
        self.memory.clear_session(PREVIEW_SESSION)

    # ---------- 摘要压缩 ----------

    async def summarize_session(self, session_id: str, force: bool = False) -> dict[str, Any]:
        """把还没压进摘要的旧消息折成一条摘要。

        force=False 是自动压缩：不到阈值、或只剩零星几条，就不动。
        force=True 供控制台手动压缩：不看阈值，有多少压多少（一次最多 summarize_after 条），
        压完消息原文仍在库里，只是上下文改从摘要接。
        """
        cfg = self.cfg.memory
        pending = self.memory.pending_count(session_id)
        if not force and pending < cfg.summarize_after:
            return {
                "ok": False,
                "error": f"未达自动压缩阈值（{pending}/{cfg.summarize_after} 条），暂时不用压",
            }

        _, upto = self.memory.get_summary(session_id)
        rows = self.memory.tail_after(session_id, upto, limit=cfg.summarize_after)
        if not rows:
            return {"ok": False, "error": "这个会话没有还没压缩的对话"}
        if len(rows) < 6 and not force:
            return {"ok": False, "error": f"只剩 {len(rows)} 条，压了没收益"}

        transcript = "\n".join(f"{r['role']}: {r['content']}" for r in rows)
        old_summary, _ = self.memory.get_summary(session_id)
        user_block = transcript
        if old_summary:
            user_block = f"已有摘要：\n{old_summary}\n\n新增对话：\n{transcript}"

        try:
            new_summary = await self.llm.complete(SUMMARY_SYSTEM, user_block, max_tokens=350)
        except LLMError as exc:
            log_.warning("摘要失败 session=%s: %s", session_id, exc)
            return {"ok": False, "error": f"摘要失败：{exc}"}

        summary = new_summary.strip()
        self.memory.set_summary(session_id, summary, int(rows[-1]["id"]))
        log_.info(
            "已压缩摘要 session=%s，覆盖 %d 条%s", session_id, len(rows),
            "（手动）" if force else "",
        )
        return {"ok": True, "covered": len(rows), "upto": int(rows[-1]["id"]), "summary": summary}

    async def _maybe_summarize(self, session_id: str) -> None:
        """到阈值才压，且同一会话同一时刻只压一次。"""
        if session_id in self._summarizing:
            return
        self._summarizing.add(session_id)
        try:
            await self.summarize_session(session_id)
        finally:
            self._summarizing.discard(session_id)

"""发言意愿门控。

机器人最难看的地方不是答得不好，是"每句话都要应一声"。
这里只回答一个问题：眼前这一批对话，轮到它开口吗？

三级漏斗，先便宜后贵：
    1) 硬规则 — 点名、引用、关键词。确定性命中，零成本。
    2) 情景打分 — 有没有人提问、话题沾不沾边、几个人在聊、有没有情绪。
       再用四个性格旋钮去缩放分数线，替掉原来那个固定的插话概率。
    3) 模型判定 — 只有分数落在阈值附近的模糊地带才问一次模型。
       把人格摘要递过去，让它自己说想不想接。一次只回一个词。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from . import log
from .batch import Batch
from .config import BotConfig, GateConfig, TriggerConfig
from .drift import Knobs
from .llm import LLMClient, LLMError
from .persona import Persona
from .policy import Trigger

log_ = log.get("gate")

_QUESTION = re.compile(
    r"[?？]|吗|呢|怎么|为什么|为啥|如何|是不是|能不能|有没有|要不要|"
    r"多少|哪(?!怕)|什么|会不会|可不可以|行不行|对不对|几时|咋|啥"
)
"""提问信号。宁滥勿缺——它只是加分项，不单独决定说不说。

别只认问号：技术群里最常见的问法是"活化温度多少合适"这种不带标点的。
"哪"排除"哪怕"，那是让步不是提问。
"""

_EMOTION = re.compile(r"烦|难过|崩溃|累死|无语|生气|哭了|郁闷|焦虑|救命|离谱|卧槽|受不了|顶不住")

JUDGE_RULE = (
    "现在群里有人聊天，没提你的名字，也没人问你。\n"
    "你看到了。要不要插一句？\n"
    "按你自己的性格来。不想说就别说，没人规定你必须接话。\n"
    "只回一个字或两个字：说 / 不说。不要解释，不要加标点以外的任何内容。"
)


@dataclass(slots=True)
class Decision:
    speak: bool
    reason: str
    score: float = 0.0
    by: str = "rule"
    """判定来源：rule / score / model。排障时看这个知道是谁做的决定。"""
    hint: str = ""
    """给生成层的语气提示。同样是不说话，被点名和主动插话不一样。"""


def _parse_verdict(ans: str) -> bool | None:
    """解析模型的一词回复。认不出来返回 None，交给分数决定。"""
    t = (ans or "").strip().strip("*。.，,！!~ \n")
    if not t:
        return None
    low = t.lower()
    if t.startswith("不") or "不说" in t or low.startswith("skip") or low.startswith("no"):
        return False
    if t.startswith("说") or low.startswith("speak") or low.startswith("yes"):
        return True
    return None


class Gate:
    def __init__(
        self,
        cfg: GateConfig,
        trigger_cfg: TriggerConfig,
        bot: BotConfig,
        persona: Persona,
        llm: LLMClient | None = None,
    ) -> None:
        self.cfg = cfg
        self.trigger = Trigger(trigger_cfg, bot)
        self.persona = persona
        self.llm = llm
        self.last_by = ""
        """最近一次判定的来源，控制台用来看门控有没有在走模型。"""
        self.drift: Knobs | None = None
        """漂移覆盖层。为 None 时完全按配置里的旋钮算，行为与没有漂移时一致。"""

    # ---------- 旋钮读取 ----------

    def _k(self, name: str) -> float:
        """读一个性格旋钮：漂移覆盖层优先，没绑就用配置值。

        覆盖层只在本进程生效，preferences.toml 里的人工值永远是"出厂设置"。
        """
        if self.drift is not None:
            return getattr(self.drift, name)
        return getattr(self.cfg, name)

    def bind_drift(self, knobs: Knobs | None) -> None:
        """挂上漂移后的旋钮。传 None 解绑，回到配置文件的值。"""
        self.drift = knobs

    # ---------- 阈值与打分 ----------

    def threshold(self) -> float:
        """性格旋钮直接调分数线：怕生的抬高，话多的压低。"""
        t = self.cfg.speak_threshold
        t += (self._k("shyness") - 0.5) * 0.30
        t -= (self._k("talkativeness") - 0.5) * 0.50
        return min(0.95, max(0.05, t))

    def score(self, batch: Batch) -> tuple[float, list[str]]:
        talk = self._k("talkativeness")
        curious = self._k("curiosity")
        shy = self._k("shyness")
        assertive = self._k("assertiveness")
        text = batch.text()
        notes: list[str] = []

        # 底噪：话多的角色，不说话本身就难受
        s = (talk - 0.5) * 0.30
        notes.append(f"底噪{s:+.2f}")

        if _QUESTION.search(text):
            add = 0.22 + curious * 0.18
            s += add
            notes.append(f"提问{add:+.2f}")

        hits = self.persona.route_hits(text) if self.persona else []
        if hits:
            add = 0.14 + curious * 0.16
            s += add
            notes.append(f"话题{add:+.2f}[{','.join(hits[:2])}]")

        if _EMOTION.search(text):
            add = 0.10 + assertive * 0.10
            s += add
            notes.append(f"情绪{add:+.2f}")

        if batch.speakers <= 1:
            s += 0.10
            notes.append("一对一+0.10")
        elif batch.speakers >= 3:
            sub = shy * 0.28
            s -= sub
            notes.append(f"人多-{sub:.2f}")

        # 聊得热闹但没人叫它：话多的凑上去，怕生的往边上站
        if len(batch.messages) >= 3:
            add = (talk - shy) * 0.15
            s += add
            notes.append(f"热聊{add:+.2f}")

        return s, notes

    # ---------- 主判定 ----------

    async def decide(self, batch: Batch, since_last_speak: float = 1e9) -> Decision:
        cfg = self.cfg
        scope_mode = self.trigger.cfg.private if batch.scope == "private" else self.trigger.cfg.group
        if scope_mode == "off":
            return Decision(False, f"{batch.scope}_off")

        # 1. 硬规则：批内任一条叫到它就算
        hit = ""
        for m in batch.messages:
            ok, why = self.trigger.hard(m)
            if ok:
                hit = why
                break
        if not hit and batch.named:
            hit = "named"
        if not hit and batch.replied_to:
            hit = "replied_to"

        if hit:
            self.last_by = "rule"
            return Decision(True, hit, 1.0, "rule", self._hint(batch, named=True))

        # 2. 自冷却：刚说完话，没人叫就别再自己凑上去
        if since_last_speak < cfg.self_cooldown_sec:
            self.last_by = "rule"
            return Decision(False, f"cooldown {since_last_speak:.0f}s/{cfg.self_cooldown_sec:.0f}s")

        if cfg.mode == "rule":
            self.last_by = "rule"
            return Decision(False, "not_named")

        # 3. 情景打分
        s, notes = self.score(batch)
        thr = self.threshold()
        detail = f"{s:.2f}/{thr:.2f} {' '.join(notes)}"

        if cfg.mode == "score":
            self.last_by = "score"
            return Decision(s >= thr, detail, s, "score", self._hint(batch, named=False))

        if s >= thr + cfg.fuzzy_margin:
            self.last_by = "score"
            return Decision(True, f"明确过线 {detail}", s, "score", self._hint(batch, named=False))
        if s <= thr - cfg.fuzzy_margin:
            self.last_by = "score"
            return Decision(False, f"明确不够 {detail}", s, "score")

        # 4. 模糊地带，问模型
        return await self._judge(batch, s, thr, detail)

    def _hint(self, batch: Batch, named: bool) -> str:
        """给生成层的语境提示。同样开口，被点名和主动插话分寸完全不同。"""
        if named:
            return "这批是直接冲你来的，正常接话就行。"
        text = batch.text()
        if _EMOTION.search(text):
            return "没人叫你，但有人在闹情绪。可以主动接一句，别讲道理，别劝。"
        if _QUESTION.search(text):
            return "没人点名，但话里有个问题悬着。可以从旁边接一句，不必给完整答案。"
        return "没人叫你，是你自己想插一句。别抢戏，一句话就够。"

    async def _judge(self, batch: Batch, score: float, thr: float, detail: str) -> Decision:
        """模糊地带交给模型：把人格摘要递过去，让它以角色身份答一个字。"""
        if self.llm is None or not getattr(self.llm, "ready", False):
            self.last_by = "score"
            return Decision(score >= thr, f"无可用模型，按分数 {detail}", score, "score")

        system = (self.persona.summary if self.persona else "") + "\n\n" + JUDGE_RULE
        user = batch.render() + "\n\n（没人叫你的名字，也没人直接问你。）"

        try:
            raw = await asyncio.wait_for(
                self.llm.complete(system, user, max_tokens=self.cfg.judge_max_tokens),
                timeout=self.cfg.judge_timeout,
            )
        except (LLMError, asyncio.TimeoutError) as exc:
            log_.debug("判定调用失败，回退分数：%s", exc)
            self.last_by = "score"
            return Decision(score >= thr, f"判定失败，按分数 {detail}", score, "score")

        verdict = _parse_verdict(raw)
        if verdict is None:
            self.last_by = "score"
            return Decision(score >= thr, f"判定无法解析({raw[:6]})，按分数 {detail}", score, "score")

        self.last_by = "model"
        log_.debug("模型判定 %s ← %r", "说" if verdict else "不说", raw.strip()[:12])
        return Decision(
            verdict,
            f"模型判定 {detail}",
            score,
            "model",
            self._hint(batch, named=False) if verdict else "",
        )

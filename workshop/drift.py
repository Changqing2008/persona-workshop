"""人格漂移内核：确定性的行为参数演化。

核心是一个纯函数 `transition(state, event) -> new_state`：
零 LLM 调用、不读时钟、不碰磁盘、无随机数。同样的初始状态 + 同样的事件序列
喂进去，出来的状态序列逐位相同 —— 可以整段回放，也可以拿线上日志复现现场。

它只改四个数（gate.py 里参与开口评分的旋钮）：话多、好奇、怕生、主动。
人格文本（wiki/ index/ raw/）一个字节都不动，那棵树的唯一写者仍然是归纳 AI。
这条边界是硬的：本模块改的是"她今天愿不愿意开口"，不是"她是谁"。

三条稳态约定，缺一条就会漂成另一个人：

    1. 有界   每个旋钮夹在 [lo, hi]，两头都不许碰 0 和 1
    2. 有回归 每步都朝 baseline 拉 pull 比例，均衡点落在 baseline 上下约 0.1~0.4
    3. 有钝化 同类事件连着来，敏感度按 decay 递减到 floor —— 被忽略一百次
              不会把怕生顶到 0.95 然后永久闭麦

权重表 `_IMPACT` 是行为设计决策，不是从文献里推出来的数字。改它就是在改
她的人格倾向，改之前先想清楚要她变成什么样。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

KNOBS: tuple[str, ...] = ("talkativeness", "curiosity", "shyness", "assertiveness")
"""四个旋钮的名字，顺序固定，序列化按这个顺序。"""


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Knobs:
    """行为参数。默认值 = 出厂设置，不随漂移改变。"""

    talkativeness: float = 0.55
    """话多话少。越高越愿意开口。"""

    curiosity: float = 0.60
    """好奇心。越高越容易被提问和话题勾住。"""

    shyness: float = 0.35
    """怕生。越高在人多时越收敛。"""

    assertiveness: float = 0.50
    """主动性。越高越偏向自己起头。"""

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in KNOBS}

    @classmethod
    def from_mapping(cls, m: dict[str, Any], fallback: "Knobs | None" | None = None) -> "Knobs":
        base = (fallback or cls()).as_dict()
        for k in KNOBS:
            if k in m:
                try:
                    base[k] = float(m[k])
                except (TypeError, ValueError):
                    pass
        return cls(**base)


@dataclass(frozen=True, slots=True)
class DriftState:
    """漂移状态。不可变，每次 transition 返回新对象。

    `baseline` 是回归目标：创建后不再改变。想让她换个底子，改配置里的
    初始值再重置，不要偷偷改 baseline —— 那等于把"回归"这件事废掉。
    """

    knobs: Knobs
    baseline: Knobs
    ignored_streak: int = 0
    """连续被排斥的次数。ignored（说了没人接）与 crowded（群里没位置）共用。

    两者连着来是同一种疼，共用一个计数才不会互相叠加成双倍伤害。
    """

    engaged_streak: int = 0
    """连续被接话的次数。"""

    turns: int = 0
    """已回放的事件数。调试和统计用。"""

    last_ts: float = 0.0
    """最后一条事件的时间戳。只用于记录，永不倒流。"""

    last_kind: str = ""
    """最后一条事件的类型。排障时看这个知道停在哪一步。"""

    def as_dict(self) -> dict[str, Any]:
        return {
            "knobs": self.knobs.as_dict(),
            "baseline": self.baseline.as_dict(),
            "ignored_streak": self.ignored_streak,
            "engaged_streak": self.engaged_streak,
            "turns": self.turns,
            "last_ts": round(float(self.last_ts), 3),
            "last_kind": self.last_kind,
        }

    def offset_from_baseline(self) -> dict[str, float]:
        """当前偏离基线多少。正数 = 比出厂设置更极端，负数 = 更收。"""
        cur = self.knobs.as_dict()
        base = self.baseline.as_dict()
        return {k: round(cur[k] - base[k], 4) for k in KNOBS}

    def touch(self, ts: float) -> DriftState:
        """只推进 last_ts，不动旋钮、不计轮次。纯函数。

        `last_ts` 是「最后一次收到消息的时间」，不是「最后一次产生漂移事件的时间」。
        这两者混用会坏两头：

            - 从没产生过事件的库，last_ts 永远是 0，"久别重逢"永不触发；
            - 产生过事件之后，last_ts 停在那一刻不动，之后哪怕每分钟都有人在聊，
              过 6 小时也会被判成"空窗"。

        所以每收一条消息都要 touch 一次，不管它有没有落成事件。
        """
        if not ts or ts <= self.last_ts:
            # 不回退：乱序消息、时钟回拨都不该让时间倒流
            return self
        return replace(self, last_ts=float(ts))


@dataclass(frozen=True, slots=True)
class DriftEvent:
    """一条事件。纯数据，不带行为。

    `gap_hours` 由调用方填（它知道上次说话是什么时候），内核不读时钟 ——
    这是可回放的前提。
    """

    kind: str
    ts: float = 0.0
    gap_hours: float = 0.0
    """距上一次同类互动的间隔小时数。只有 idle 用它。"""

    by_owner: bool = False
    """这次互动是不是拥有者发起的。是的话正反馈权重乘 owner_weight。"""

    scope: str = "group"
    """private / group。目前只记录，不参与计算。"""


@dataclass(slots=True)
class DriftConfig:
    """漂移参数。全部可调，改完建议跑一遍 test_drift.py。

    注意这里**不能**加 frozen：它是配置，配置要在保存时被逐项覆盖
    （webui 的 set_cfg_value 走 setattr，apply_prefs 也是）。
    真正该不可变的是状态 Knobs / DriftState，不是参数表。
    """

    enabled: bool = False
    """默认关。

    内核层（`DEFAULT`）显式开——纯函数回放得先能算，否则 demo 和单测全是空转。
    配置层（config.load）也显式传，两层都不依赖这个默认值。
    留 False 是为了防止别处 `DriftConfig()` 一构造就把漂移打开。
    """

    step: float = 0.02
    """单次事件的最大位移。0.02 意味着 50 次同向事件走满 1.0 的理论上限。"""

    pull: float = 0.05
    """每步朝基线的回归比例。相对 step 越大，漂得越近。"""

    lo: float = 0.05
    """下界。不许到 0：一个 0 话量的人格再也接不回来。"""

    hi: float = 0.95
    """上界。同理不许到 1。"""

    decay: float = 0.92
    """同类事件连续发生时，敏感度每次乘这个数。

    0.85 太狠：连着被忽略十次就麻了，怕生只涨 0.02。真实的人际里持续冷落
    是累积的，钝化只该防"一次坏日子把参数打歪"，不该替她习惯被忽略。"""

    decay_floor: float = 0.45
    """钝化下限。再连着来也不会低到这个比例以下。"""

    idle_hours: float = 6.0
    """多久没互动算"长时间"。idle 事件的敏感度以此为 1.0 的基准。"""

    owner_weight: float = 1.6
    """拥有者的正反馈加权。被主人接话比被路人接话更让她放松。"""


DEFAULT = DriftConfig(enabled=True)
"""内核默认参数：开着算。这是"计算"用的默认值，不是"上线"用的默认值。

线上开不开由配置决定（config.load 的 `drift.enabled` 默认 False），
两件事别混：这里的 enabled=False 会让 transition 直接空转，demo 和单测全废。
"""


# --------------------------------------------------------------------------
# 权重表
# --------------------------------------------------------------------------

_IMPACT: dict[str, dict[str, float]] = {
    # 被直接叫到：怕生往下掉、主动往上走。被期待会让人放开一点。
    "named": {
        "talkativeness": 0.30,
        "curiosity": 0.20,
        "shyness": -0.50,
        "assertiveness": 0.40,
    },
    # 她的话被人接住了：最有效的正反馈，但也钝化得最快。
    "replied": {
        "talkativeness": 1.00,
        "curiosity": 0.50,
        "shyness": -0.60,
        "assertiveness": 0.30,
    },
    # 说完没人理：唯一会真正把她推回壳里的事件。
    "ignored": {
        "talkativeness": -1.00,
        "curiosity": -0.30,
        "shyness": 0.80,
        "assertiveness": -0.50,
    },
    # 一群人热闹但没她的位置：怕生涨得比被忽略还快，主动性掉得更多。
    "crowded": {
        "talkativeness": -0.40,
        "curiosity": 0.10,
        "shyness": 0.90,
        "assertiveness": -0.60,
    },
    # 长时间没人找：想说话（好奇涨）、但更不敢多话（话量微降），会自己起头。
    "idle": {
        "talkativeness": -0.20,
        "curiosity": 0.80,
        "shyness": 0.30,
        "assertiveness": 0.70,
    },
    # 什么都没发生：只走回归。
    "quiet": {},
}

EVENTS: tuple[str, ...] = tuple(_IMPACT)
"""合法事件类型，同时也是文档清单。"""


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _decay(streak: int, factor: float, floor: float) -> float:
    """同类事件连着来，敏感度递减到 floor 为止。

    streak 是含本次在内的连续计数：第 1 次返回 1.0，第 2 次返回 factor。
    钝化是稳态的关键 —— 没有它，一次坏日子的连续暴击就能把人设打歪。
    """
    return max(floor, factor ** max(0, streak - 1))


def _sensitivity(event: DriftEvent, cfg: DriftConfig, rejected: int, engaged: int) -> float:
    """这次事件该打几折。0 表示不产生位移。"""
    kind = event.kind
    if kind == "named":
        s = 1.0
    elif kind == "replied":
        s = _decay(engaged, cfg.decay, cfg.decay_floor)
    elif kind in ("ignored", "crowded"):
        # 两种都是"被排斥"，共用一个连续计数：说了没人接，和在群里没位置，
        # 连着来是同一种疼，不该各记各的然后双倍生效。
        s = _decay(rejected, cfg.decay, cfg.decay_floor)
    elif kind == "idle":
        # 隔了 3 小时和隔了 30 小时不是一回事，但封顶 2 倍，别让久别重逢变成换人。
        s = min(2.0, event.gap_hours / cfg.idle_hours) if cfg.idle_hours > 0 else 1.0
    else:  # quiet
        s = 0.0

    if event.by_owner and kind in ("named", "replied"):
        s *= cfg.owner_weight
    return s


# --------------------------------------------------------------------------
# 内核
# --------------------------------------------------------------------------


def initial(knobs: Knobs | None = None) -> DriftState:
    """出厂状态。knobs 给的是旋钮值，同时也是回归基线。"""
    k = knobs or Knobs()
    return DriftState(knobs=k, baseline=k)


def knobs_from(obj: Any, fallback: Knobs | None = None) -> Knobs:
    """从任意带四个旋钮属性的对象取一组值（鸭子类型，不 import 配置模块）。

    用途是把人格库自己的性格配置拿来当基线。基线不能是硬编码的出厂值：
    两个库的 shyness 只要不一样，漂移一开工就会把人工调过的那份盖掉。
    """
    base = fallback or Knobs()
    mapped = {k: getattr(obj, k) for k in KNOBS if hasattr(obj, k)}
    return Knobs.from_mapping(mapped, base)


def transition(
    state: DriftState,
    event: DriftEvent,
    cfg: DriftConfig | None = None,
) -> DriftState:
    """吃一个事件，吐一个新状态。纯函数：无副作用、无随机、不读时钟。

    两件事同时发生：
        位移   按权重表推一把
        回归   再朝 baseline 拉回 pull 比例
    回归是后做的，所以单步不会冲过界；长期看收敛在基线上下的有限带内。
    """
    c = cfg or DEFAULT

    if event.kind not in _IMPACT:
        raise ValueError(f"未知事件类型 {event.kind!r}，只认 {'/'.join(EVENTS)}")
    if not c.enabled:
        return state

    # 连续被排斥的计数：ignored 与 crowded 共用，换类才断
    rejected = state.ignored_streak + 1 if event.kind in ("ignored", "crowded") else 0
    engaged = state.engaged_streak + 1 if event.kind == "replied" else 0

    scale = _sensitivity(event, c, rejected, engaged)
    weights = _IMPACT[event.kind]

    cur = state.knobs.as_dict()
    base = state.baseline.as_dict()

    if scale > 0.0:
        for name in KNOBS:
            w = weights.get(name, 0.0)
            if not w:
                continue
            v = cur[name] + c.step * w * scale
            v += c.pull * (base[name] - v)
            cur[name] = _clamp(v, c.lo, c.hi)
    else:
        # 纯回归事件（quiet，或 gap 为 0 的 idle）：只回家，不位移
        for name in KNOBS:
            v = cur[name] + c.pull * (base[name] - cur[name])
            cur[name] = _clamp(v, c.lo, c.hi)

    return DriftState(
        knobs=Knobs(**{k: round(cur[k], 6) for k in KNOBS}),
        baseline=state.baseline,
        ignored_streak=rejected,
        engaged_streak=engaged,
        turns=state.turns + 1,
        last_ts=max(state.last_ts, event.ts),
        last_kind=event.kind,
    )


def replay(
    events: Iterable[DriftEvent],
    start: DriftState | None = None,
    cfg: DriftConfig | None = None,
) -> list[DriftState]:
    """按序回放，返回状态序列（含起始状态，长度 = 事件数 + 1）。

    这就是验证入口：把线上日志翻译成事件，重放一遍，看她的参数有没有
    漂到你没设计过的地方。
    """
    s = start or initial()
    out = [s]
    for e in events:
        s = transition(s, e, cfg)
        out.append(s)
    return out


def apply_to(gate_cfg: Any, state: DriftState) -> Any:
    """把漂移后的旋钮叠到 GateConfig 上，返回副本。

    不修改传入对象 —— preferences.toml 里的人工值是"出厂设置"，
    现场跑的时候用这份副本，文件本身不动。
    """
    return replace(gate_cfg, **state.knobs.as_dict())


def validate(cfg: DriftConfig) -> list[str]:
    """配置体检。返回问题列表，空列表表示通过。"""
    problems: list[str] = []
    if not 0.0 < cfg.step <= 0.2:
        problems.append(f"[drift] step 应在 (0, 0.2]，当前 {cfg.step}")
    if not 0.0 <= cfg.pull <= 0.5:
        problems.append(f"[drift] pull 应在 [0, 0.5]，当前 {cfg.pull}")
    if not 0.0 <= cfg.lo < cfg.hi <= 1.0:
        problems.append(f"[drift] 需要 0 <= lo < hi <= 1，当前 lo={cfg.lo} hi={cfg.hi}")
    if not 0.0 < cfg.decay < 1.0:
        problems.append(f"[drift] decay 应在 (0, 1)，当前 {cfg.decay}")
    if not 0.0 < cfg.decay_floor <= 1.0:
        problems.append(f"[drift] decay_floor 应在 (0, 1]，当前 {cfg.decay_floor}")
    if cfg.idle_hours <= 0:
        problems.append(f"[drift] idle_hours 必须为正，当前 {cfg.idle_hours}")
    if cfg.owner_weight <= 1.0:
        problems.append(f"[drift] owner_weight 应 > 1（拥有者加权），当前 {cfg.owner_weight}")
    return problems


# --------------------------------------------------------------------------
# 序列化
# --------------------------------------------------------------------------


def to_dict(state: DriftState) -> dict[str, Any]:
    return state.as_dict()


def from_dict(d: dict[str, Any], fallback: DriftState | None = None) -> DriftState:
    """从 dict 还原。字段缺了或坏了就退回 fallback 的同名字段，不抛。"""
    f = fallback or initial()
    try:
        knobs = Knobs.from_mapping(d.get("knobs") or {}, f.knobs)
        baseline = Knobs.from_mapping(d.get("baseline") or {}, f.baseline)
        return DriftState(
            knobs=knobs,
            baseline=baseline,
            ignored_streak=int(d.get("ignored_streak", 0) or 0),
            engaged_streak=int(d.get("engaged_streak", 0) or 0),
            turns=int(d.get("turns", 0) or 0),
            last_ts=float(d.get("last_ts", 0.0) or 0.0),
            last_kind=str(d.get("last_kind", "") or ""),
        )
    except (TypeError, ValueError):
        return f


class DriftStore:
    """状态落盘的轻薄包装。

    内核保持纯函数，读写文件全部收在这里。写盘走临时文件 + 原子替换，
    中途断电不会留下半截 JSON。
    """

    def __init__(
        self,
        path: str | Path,
        cfg: DriftConfig | None = None,
        baseline: Knobs | None = None,
    ) -> None:
        self.path = Path(path)
        self.cfg = cfg or DEFAULT
        self.baseline = baseline
        """回归目标。None 表示用出厂值——只有不受库约束的场景（测试、演示）该这么用。"""
        self.baseline_moved = False
        """读盘时是否因为配置变了而改过基线。调用方可以据此记一笔日志。"""
        self._state = self._load()

    @property
    def state(self) -> DriftState:
        return self._state

    def _load(self) -> DriftState:
        fallback = initial(self.baseline)
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                loaded = from_dict(json.load(fh))
        except (OSError, json.JSONDecodeError, ValueError):
            # 文件不在 / 坏了，都当出厂状态。漂移是可重建的，不值得为它崩。
            return fallback

        # 基线跟着库的配置走：用户改了 preferences.toml 的性格底子，就该朝新底子
        # 回归。但她的近况（knobs）不因此被抹平——只把回归目标挪过去，
        # 然后由后续每一步的 pull 慢慢拉。直接重置等于"改了配置她就失忆"。
        if self.baseline is not None and loaded.baseline != self.baseline:
            self.baseline_moved = True
            loaded = replace(loaded, baseline=self.baseline)
        return loaded

    def record(self, event: DriftEvent) -> DriftState:
        """吃一个事件，返回新状态。不自动写盘 —— 由调用方决定刷盘时机。"""
        self._state = transition(self._state, event, self.cfg)
        return self._state

    def touch(self, ts: float) -> DriftState:
        """推进"最后一次收到消息的时间"。不产生轮次，不写盘。

        和 record 分开是必须的：每收一条消息都该 touch，但绝大多数消息
        不该落成漂移事件。混在一起会让 turns 虚高、旋钮被噪声推动。
        """
        self._state = self._state.touch(ts)
        return self._state

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(to_dict(self._state), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def reset(self, knobs: Knobs | None = None) -> DriftState:
        """重置。不传 knobs 就回到当前基线，而不是硬编码的出厂值。"""
        self._state = initial(knobs if knobs is not None else self.baseline)
        return self._state

    def rebase(self, baseline: Knobs) -> DriftState:
        """换回归目标，保留当前旋钮值。

        热更新改了库的性格配置时用这个，不要用 reset —— reset 会把
        她这阵子的经历一并抹掉，那等于"改个配置她就失忆"。
        """
        self.baseline = baseline
        self._state = replace(self._state, baseline=baseline)
        return self._state

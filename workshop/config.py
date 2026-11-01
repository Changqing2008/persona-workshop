"""配置加载。TOML 格式，标准库 tomllib 解析，零第三方依赖。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .drift import DriftConfig
from .drift import validate as validate_drift

DEFAULT_CONFIG_CANDIDATES = (
    "config.toml",
    "config.example.toml",
)


def _get(d: dict[str, Any], key: str, default: Any) -> Any:
    """按 a.b.c 路径取值，缺失返回默认。"""
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass(slots=True)
class BotConfig:
    self_id: str = ""
    persona_dir: str = ""
    nickname: str = ""
    log_level: str = "INFO"
    log_file: str = "data/bridge.log"


@dataclass(slots=True)
class TriggerConfig:
    private: str = "always"  # always | keyword | off
    group: str = "at"  # at | keyword | always | off
    keywords: list[str] = field(default_factory=list)
    group_chance: float = 0.0
    """群聊未 @ 时随机插话的概率（0~1）。0 表示从不主动。"""
    ignore_users: list[str] = field(default_factory=list)
    """黑名单 QQ 号。"""
    max_input_chars: int = 800


@dataclass(slots=True)
class BatchConfig:
    """对话合流。攒一段时间再一次性回应，替掉"来一句回一句"。"""

    enabled: bool = True
    group_quiet_sec: float = 4.0
    """群聊静默窗：这么久没新消息就交出去。有人说话会重置计时。"""
    private_quiet_sec: float = 1.5
    """私聊静默窗。一对一没有"插话"问题，短一点更像正常回消息。"""
    max_wait_sec: float = 20.0
    """硬上限。从本批第一条算起，无论还在不在聊都交出去，防刷屏饿死。"""
    max_messages: int = 12
    """一批最多攒这么多条，满了立刻交。0 表示不限。"""


@dataclass(slots=True)
class GateConfig:
    """发言意愿门控：这一批对话轮到它开口吗。

    四个性格旋钮不是抽象标签，直接参与阈值与加分计算：
    话多的阈值低、怕生的在人多时更收敛、好奇的更容易被话题勾住、
    强势的冷场时更愿意自己起头。
    """

    mode: str = "hybrid"  # rule | score | hybrid
    speak_threshold: float = 0.45
    """基础开口分数线。会被性格旋钮上下调整。"""
    fuzzy_margin: float = 0.15
    """模糊地带宽度。分数落在阈值上下这个范围内，才交给模型判。"""
    talkativeness: float = 0.55
    """话多话少 0~1。越高阈值越低。"""
    curiosity: float = 0.6
    """好奇心 0~1。越高越容易被提问和话题勾住。"""
    shyness: float = 0.35
    """怕生 0~1。越高在人多时越收敛，阈值越高。"""
    assertiveness: float = 0.5
    """主动性 0~1。越高越偏向自己起头。"""
    self_cooldown_sec: float = 25.0
    """自己刚发过言后的静默期。这段时间内没被点名就不再主动开口。"""
    judge_max_tokens: int = 16
    """模型判定只回一个词。给到 8 有些模型会直接返回空内容，触发白跑一轮重试。"""
    allow_silence: bool = True
    """允许她"看到了但一个字都不给"。关掉则必须给点反应（哪怕是一句冷话）。"""
    judge_timeout: float = 8.0


@dataclass(slots=True)
class OwnerConfig:
    """拥有者：把她写出来、一直在调教的那个人。

    用来在群里区分"主客"——对主人和对路人，分寸不该一样。

    官方通道的 openid 按场景分域：私聊走 user_openid、群聊走 member_openid，
    同一个人在这两处是两个不同的 id，所以 ids 要把两个都填上。
    OneBot 直接填 QQ 号即可，跨场景天然一致。
    """

    ids: list[str] = field(default_factory=list)
    call_me: str = ""
    """她怎么称呼拥有者。留空则按「你」处理。"""
    familiar_turns: int = 12
    """非主人聊够多少轮算熟人。0 表示不做熟人这一层。"""
    carry_summary: bool = True
    """群里主人说话时，是否带上他与她的私聊摘要（单向，且不许复述）。"""


@dataclass(slots=True)
class RateLimitConfig:
    per_session_per_min: int = 12
    """每人每分钟回复上限。按会话里的每个发言人各算一份，不是整群共用。"""

    min_interval_sec: float = 1.2
    """整会话的最小发言间隔。防的是群内连发被风控，所以按会话算。"""


@dataclass(slots=True)
class LLMConfig:
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    temperature: float = 0.85
    max_tokens: int = 400
    timeout: float = 60.0
    max_retries: int = 2
    ingest_model: str = ""
    """一键归纳用的模型。留空 = 跟聊天同一个。

    归纳吃长上下文、不吃"温度"，跟聊天模型是两种用法：
    聊天要的是稳中有变（temperature 0.85），归纳要的是照着规程抄证据。
    指一个便宜的 128K 模型（deepseek-chat / qwen-long 之类）能省一大截。
    """
    ingest_max_tokens: int = 6000
    """归纳单份的输出上限。推理型模型会先花 3~4k 思考，给小了正文会是空的。"""
    vision_model: str = ""
    """看图的模型。留空 = 跟着 `model` 走。

    单列这个字段，只为「聊天模型收不了图、得换一个多模态的」这种情况留出口。
    同一个模型就能看图时（deepseek-flash 实测可以）留空最省事，不用重复填。
    填了才跟 `model` 分开；base_url 和 api_key 照旧共用。
    """
    vision_max_tokens: int = 2000
    """单张截图的转写上限。一屏聊天四十来条，2000 够；推理型模型要再放大。"""

    def resolve_key(self) -> str:
        if self.api_key:
            return self.api_key.strip()
        return os.environ.get(self.api_key_env, "").strip()


@dataclass(slots=True)
class MemoryConfig:
    db_path: str = "data/bridge.db"
    recent_turns: int = 12
    """注入提示词的最近消息条数（user+assistant 合计）。"""
    summarize_after: int = 40
    """单会话未压缩消息超过此数时，触发滚动摘要。"""
    wiki_enabled: bool = False
    """记忆层（persona/memory/）检索开关。关着时行为与旧版完全一致。"""
    wiki_budget: int = 800
    """单次注入字数上限。超了截断——记忆不许挤占人格的预算。"""
    ingest_interval_hours: float = 0.0
    """定时归档间隔。0 = 不自动跑；>0 则每 N 小时把新聊天沉淀进 memory/。"""
    ingest_allow_new: bool = False
    """定时归档是否允许自动新建页。默认关：无人看着时先落 pending，不擅自开页。"""


@dataclass(slots=True)
class IngestConfig:
    """一键归纳的行为开关。放库内 preferences.toml 的 [ingest] 段，一人一套。

    默认值就是"什么都不改"的那一套：收完重建总括段、每轮不限份数、直接落盘。
    三个开关都是"想收着点"的时候才去动，不是必填项。
    """

    auto_index: bool = True
    """收完素材后重建 index.md 总括段。关掉则只写 wiki，总括段靠人工维护。"""
    max_per_run: int = 0
    """一轮最多收几份。0 = 不限。设正数时超出的留在队列里，下次点按钮接着收。"""
    preview_first: bool = False
    """开成 true 就是"先预演"：只出清单不落盘（等同于每次带 dry_run）。"""


@dataclass(slots=True)
class OneBotConfig:
    ws_url: str = "ws://127.0.0.1:3001"
    access_token: str = ""
    reconnect_delay: float = 3.0
    max_reconnect_delay: float = 60.0


@dataclass(slots=True)
class OfficialConfig:
    app_id: str = ""
    app_secret: str = ""
    sandbox: bool = False
    api_base: str = ""
    """留空按 sandbox 自动选正式/沙箱域名，特殊情况下可手动覆盖。"""

    intents: list[str] = field(default_factory=lambda: ["GROUP_AND_C2C_EVENT"])
    """事件订阅。

    默认只订阅群聊 + 单聊（1<<25），这是最小可用集合。
    注意：订阅了没有权限的事件类型，网关会直接断开连接，
    所以宁可少订阅，要用频道再加 PUBLIC_GUILD_MESSAGES。
    """

    shard: list[int] = field(default_factory=lambda: [0, 1])
    heartbeat_timeout: float = 30.0
    """多久没收到心跳 ACK 就认为连接已死。"""


@dataclass(slots=True)
class ReplyConfig:
    split_threshold: int = 180
    """超过此长度按段落拆成多条发送，避免被截断。"""
    max_chars: int = 900
    """硬上限，超出直接截断。"""
    quote_when_group: bool = False
    """群聊回复时是否引用原消息。"""
    split_on_blank: bool = True
    """她分了段就分几条发，不挤在一个气泡里。真人在聊天里也是这么发的。"""
    max_chunks: int = 4
    """一次最多拆成几条。官方平台同一条消息最多回 5 次，留一条余量。"""


@dataclass(slots=True)
class WebUIConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8770


@dataclass(slots=True)
class SimConfig:
    """素材台。产出走文件，不写数据库。"""

    enabled: bool = True
    dir: str = "sim"
    """素材台工作目录，下分 cards/（收藏的卡）与 drafts/（草稿与成品）。"""
    candidates: int = 3
    """每次出几条候选。挑选用，不建议超过 5。"""
    temperature: float = 1.0
    """候选要够散才有的挑，比聊天温度高。"""
    self_label: str = ""
    """人格本人在这段对话里的显示名。留空则取人格库目录名的第一段。"""
    card_count: int = 3
    """一次现生成几张情境卡。"""
    card_max_tokens: int = 2200
    """生成情境卡的输出上限。三张卡约 1200~1500 token，给小了会被截断成半截 JSON。"""


@dataclass(slots=True)
class DistillConfig:
    """小说蒸馏器。只产成品稿，不写 raw。"""

    enabled: bool = True
    dir: str = "distill"
    """工作目录：books/（原文与提取结果）与 out/（成品 md）。"""
    min_lines: int = 2
    """少于这么多句不成段。一段至少要有来有回，否则没法当素材。"""
    max_lines: int = 3
    """一段最多这么多句。太长一条拖进 raw 不好读。"""
    gap_paragraphs: int = 3
    """两句之间隔超过这么多段就断开，不硬连成一段。只影响「怎么分段」，不影响认谁说的。"""
    turn_gap: int = 20
    """轮次交替能跨多少段。

    和 gap_paragraphs 分开是有实测依据的：这两个值一个管"能猜多远"，一个管
    "段落多长"，混用会互相绑架。实测同一本书——
        turn_gap=3   留 334 句 / 32 段 / 段均 10 句
        turn_gap=20  留 1321 句 / 63 段 / 段均 21 句
    turn_gap 放大到 20 能把召回从 334 拉到 1321（金标准判对率 46.7%，比 3 时的
    17.8% 还高），代价是段落变长。拆开以后就能"召回按大的来、段落按小的切"。
    """
    turn_steps: int = 0
    """一个锚点之后，最多连着猜几步。0 表示不限（**默认**）。

    这是"多"和"准"之间的旋钮，但实测它不是个权衡：
        步数   1    3    8   不限
        句数  98  188  404  1405
        判对 11%  13%  27%   62%
    步数越小判对率越低——限步只是把句子砍掉，并不会把错的筛掉。所以默认不限，
    真要少收再改小。真正让"错得可分辨"的是 line 上的 depth（连猜了几步），
    界面按它把「远推 12」和「推定」分开显示。
    """
    judge: str = "rules"
    """块级裁判（模型判每句谁说的）。

    rules — 只用规则，不调模型，与 P2a 完全一致（**默认**，不会产生任何费用）；
    auto  — 一个场景块一次调用，只覆盖规则里靠猜的那几档（alternate/topic/pronoun/unknown），
            显式命名和线索绑定不动。15 万字约 80 次调用、同步多等一两分钟；
    off   — 同 rules，但连候选集都不算，纯旧行为。

    默认值刻意保守：不打招呼就替人烧 token 是最不该有的默认。要开就写进 config.toml：
        [distill]
        judge = "auto"
    """
    judge_max_tokens: int = 6000
    """一块判一次的输出上限。

    给小了会被推理型模型吃光：deepseek-flash 有近一半的块把 1200 token 全花在
    思考上，正文一个字没有。6000 起步，真不够时 judge_block 自己会加预算重试。
    """
    judge_scene_gap: int = 6
    """两句相隔超过这么多段就当成两个场景，不放进同一次调用（上下文太长模型会飘）。"""
    judge_max_lines: int = 24
    """一次调用最多带多少句台词给模型。

    和 max_lines（成品一段几句子）是两回事，别混用：max_lines 是切给人看的，
    这个是一口气给模型多少上下文。共用会把模型的视野砍成三句，判断依据全没了。
    """
    judge_cache: bool = True
    """按「场景文本 + 候选集」缓存判定结果，同一本书重提取不再重复烧钱。"""
    guess: str = "chain"
    """认不出说话人时，要不要按轮次交替往下推。

    chain（默认）— A→B→A→B 一路推。中文小说的对话几乎都是交替的，这是规则能做的
                  最后一招，也是召回的主力：本书 1321 句留存里有 1264 句靠它；
    anchor       — 上一句必须真有名字才敢推一句。金标准上"看起来"更准（误收少），
                  但代价是把 1321 句砍到 94 句 —— 蒸馏台是给人挑素材的，
                  召回砍到 6% 等于让人没得挑，这个方向是错的；
    off          — 完全不推。

    judge="rules"（默认不调模型）时，chain 是唯一能撑起召回的规则。
    """


@dataclass(slots=True)
class Config:
    bot: BotConfig = field(default_factory=BotConfig)
    owner: OwnerConfig = field(default_factory=OwnerConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    """人格漂移。默认关闭：不开启时 gate 的旋钮完全按 [gate] 段来，行为与旧版一致。"""
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    onebot: OneBotConfig = field(default_factory=OneBotConfig)
    official: OfficialConfig = field(default_factory=OfficialConfig)
    reply: ReplyConfig = field(default_factory=ReplyConfig)
    webui: WebUIConfig = field(default_factory=WebUIConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)

    channel: str = "onebot"
    root: Path = field(default_factory=Path.cwd)
    """配置文件所在目录，用于解析相对路径。"""
    source_path: Path | None = None
    """实际加载的配置文件路径，WebUI 保存时写回这里。"""

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    def validate(self) -> list[str]:
        """返回问题列表，空列表表示通过。"""
        problems: list[str] = []
        if not self.bot.persona_dir:
            problems.append("[bot] persona_dir 未配置（人格库目录）")
        elif not self.path(self.bot.persona_dir).is_dir():
            problems.append(f"[bot] persona_dir 不存在：{self.path(self.bot.persona_dir)}")
        if self.trigger.private not in {"always", "keyword", "off"}:
            problems.append(f"[trigger] private 取值非法：{self.trigger.private}")
        if self.trigger.group not in {"at", "keyword", "always", "off"}:
            problems.append(f"[trigger] group 取值非法：{self.trigger.group}")
        if self.gate.mode not in {"rule", "score", "hybrid"}:
            problems.append(f"[gate] mode 取值非法：{self.gate.mode}（rule / score / hybrid）")
        if self.batch.group_quiet_sec < 0 or self.batch.private_quiet_sec < 0:
            problems.append("[batch] 静默窗不能为负")
        if self.ingest.max_per_run < 0:
            problems.append("[ingest] max_per_run 不能为负（0 = 不限）")
        if self.batch.max_wait_sec and self.batch.max_wait_sec < self.batch.group_quiet_sec:
            problems.append("[batch] max_wait_sec 小于静默窗，合流永远不会等到静默")
        if self.drift.enabled:
            problems.extend(validate_drift(self.drift))
        if not self.llm.resolve_key():
            problems.append(
                f"[llm] 没有 API Key：请填 api_key，或设置环境变量 {self.llm.api_key_env}"
            )
        if self.channel == "official":
            if not self.official.app_id:
                problems.append("[adapter.official] 缺少 app_id（QQ 开放平台 → 开发设置）")
            if not self.official.app_secret:
                problems.append("[adapter.official] 缺少 app_secret（ClientSecret）")
        elif self.channel == "onebot":
            if not self.onebot.ws_url:
                problems.append("[adapter.onebot] 缺少 ws_url")
        return problems


def load(path: str | Path | None = None) -> Config:
    """加载配置。path 为 None 时按候选名查找当前目录。"""
    cfg_path: Path | None = None
    if path is not None:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"配置文件不存在：{cfg_path}")
    else:
        for name in DEFAULT_CONFIG_CANDIDATES:
            cand = Path(name)
            if cand.is_file():
                cfg_path = cand
                break

    raw: dict[str, Any] = {}
    root = Path.cwd()
    if cfg_path is not None:
        with cfg_path.open("rb") as fh:
            raw = tomllib.load(fh)
        root = cfg_path.resolve().parent

    cfg = Config(
        bot=BotConfig(
            self_id=str(_get(raw, "bot.self_id", "")),
            persona_dir=_get(raw, "bot.persona_dir", ""),
            nickname=_get(raw, "bot.nickname", ""),
            log_level=_get(raw, "bot.log_level", "INFO"),
            log_file=_get(raw, "bot.log_file", "data/bridge.log"),
        ),
        owner=OwnerConfig(
            ids=[str(u).strip() for u in _get(raw, "owner.ids", []) if str(u).strip()],
            call_me=str(_get(raw, "owner.call_me", "")),
            familiar_turns=int(_get(raw, "owner.familiar_turns", 12)),
            carry_summary=bool(_get(raw, "owner.carry_summary", True)),
        ),
        trigger=TriggerConfig(
            private=_get(raw, "trigger.private", "always"),
            group=_get(raw, "trigger.group", "at"),
            keywords=list(_get(raw, "trigger.keywords", [])),
            group_chance=float(_get(raw, "trigger.group_chance", 0.0)),
            ignore_users=[str(u) for u in _get(raw, "trigger.ignore_users", [])],
            max_input_chars=int(_get(raw, "trigger.max_input_chars", 800)),
        ),
        batch=BatchConfig(
            enabled=bool(_get(raw, "batch.enabled", True)),
            group_quiet_sec=float(_get(raw, "batch.group_quiet_sec", 4.0)),
            private_quiet_sec=float(_get(raw, "batch.private_quiet_sec", 1.5)),
            max_wait_sec=float(_get(raw, "batch.max_wait_sec", 20.0)),
            max_messages=int(_get(raw, "batch.max_messages", 12)),
        ),
        gate=GateConfig(
            mode=str(_get(raw, "gate.mode", "hybrid")),
            speak_threshold=float(_get(raw, "gate.speak_threshold", 0.45)),
            fuzzy_margin=float(_get(raw, "gate.fuzzy_margin", 0.15)),
            talkativeness=float(_get(raw, "gate.talkativeness", 0.55)),
            curiosity=float(_get(raw, "gate.curiosity", 0.6)),
            shyness=float(_get(raw, "gate.shyness", 0.35)),
            assertiveness=float(_get(raw, "gate.assertiveness", 0.5)),
            self_cooldown_sec=float(_get(raw, "gate.self_cooldown_sec", 25.0)),
            judge_max_tokens=int(_get(raw, "gate.judge_max_tokens", 16)),
            allow_silence=bool(_get(raw, "gate.allow_silence", True)),
            judge_timeout=float(_get(raw, "gate.judge_timeout", 8.0)),
        ),
        drift=DriftConfig(
            enabled=bool(_get(raw, "drift.enabled", False)),
            step=float(_get(raw, "drift.step", 0.02)),
            pull=float(_get(raw, "drift.pull", 0.05)),
            lo=float(_get(raw, "drift.lo", 0.05)),
            hi=float(_get(raw, "drift.hi", 0.95)),
            decay=float(_get(raw, "drift.decay", 0.92)),
            decay_floor=float(_get(raw, "drift.decay_floor", 0.45)),
            idle_hours=float(_get(raw, "drift.idle_hours", 6.0)),
            owner_weight=float(_get(raw, "drift.owner_weight", 1.6)),
        ),
        rate_limit=RateLimitConfig(
            per_session_per_min=int(_get(raw, "rate_limit.per_session_per_min", 12)),
            min_interval_sec=float(_get(raw, "rate_limit.min_interval_sec", 1.2)),
        ),
        llm=LLMConfig(
            base_url=str(_get(raw, "llm.base_url", LLMConfig.base_url)).rstrip("/"),
            api_key=_get(raw, "llm.api_key", ""),
            api_key_env=_get(raw, "llm.api_key_env", "DEEPSEEK_API_KEY"),
            model=_get(raw, "llm.model", "deepseek-chat"),
            temperature=float(_get(raw, "llm.temperature", 0.85)),
            max_tokens=int(_get(raw, "llm.max_tokens", 400)),
            timeout=float(_get(raw, "llm.timeout", 60.0)),
            max_retries=int(_get(raw, "llm.max_retries", 2)),
            ingest_model=str(_get(raw, "llm.ingest_model", "")),
            ingest_max_tokens=int(_get(raw, "llm.ingest_max_tokens", 6000)),
            vision_model=str(_get(raw, "llm.vision_model", "")),
            vision_max_tokens=int(_get(raw, "llm.vision_max_tokens", 2000)),
        ),
        memory=MemoryConfig(
            db_path=_get(raw, "memory.db_path", "data/bridge.db"),
            recent_turns=int(_get(raw, "memory.recent_turns", 12)),
            summarize_after=int(_get(raw, "memory.summarize_after", 40)),
            wiki_enabled=bool(_get(raw, "memory.wiki_enabled", False)),
            wiki_budget=int(_get(raw, "memory.wiki_budget", 800)),
            ingest_interval_hours=float(_get(raw, "memory.ingest_interval_hours", 0.0)),
            ingest_allow_new=bool(_get(raw, "memory.ingest_allow_new", False)),
        ),
        ingest=IngestConfig(
            auto_index=bool(_get(raw, "ingest.auto_index", True)),
            max_per_run=int(_get(raw, "ingest.max_per_run", 0)),
            preview_first=bool(_get(raw, "ingest.preview_first", False)),
        ),
        onebot=OneBotConfig(
            ws_url=_get(raw, "adapter.onebot.ws_url", "ws://127.0.0.1:3001"),
            access_token=_get(raw, "adapter.onebot.access_token", ""),
            reconnect_delay=float(_get(raw, "adapter.onebot.reconnect_delay", 3.0)),
            max_reconnect_delay=float(_get(raw, "adapter.onebot.max_reconnect_delay", 60.0)),
        ),
        official=OfficialConfig(
            app_id=str(_get(raw, "adapter.official.app_id", "")),
            app_secret=_get(raw, "adapter.official.app_secret", ""),
            sandbox=bool(_get(raw, "adapter.official.sandbox", False)),
            api_base=_get(raw, "adapter.official.api_base", ""),
            intents=list(_get(raw, "adapter.official.intents", ["GROUP_AND_C2C_EVENT"])),
            shard=list(_get(raw, "adapter.official.shard", [0, 1])),
            heartbeat_timeout=float(_get(raw, "adapter.official.heartbeat_timeout", 30.0)),
        ),
        reply=ReplyConfig(
            split_threshold=int(_get(raw, "reply.split_threshold", 180)),
            max_chars=int(_get(raw, "reply.max_chars", 900)),
            quote_when_group=bool(_get(raw, "reply.quote_when_group", False)),
            split_on_blank=bool(_get(raw, "reply.split_on_blank", True)),
            max_chunks=int(_get(raw, "reply.max_chunks", 4)),
        ),
        webui=WebUIConfig(
            enabled=bool(_get(raw, "webui.enabled", True)),
            host=_get(raw, "webui.host", "127.0.0.1"),
            port=int(_get(raw, "webui.port", 8770)),
        ),
        sim=SimConfig(
            enabled=bool(_get(raw, "sim.enabled", True)),
            dir=str(_get(raw, "sim.dir", "sim")),
            candidates=int(_get(raw, "sim.candidates", 3)),
            temperature=float(_get(raw, "sim.temperature", 1.0)),
            self_label=str(_get(raw, "sim.self_label", "")),
            card_count=int(_get(raw, "sim.card_count", 3)),
            card_max_tokens=int(_get(raw, "sim.card_max_tokens", 2200)),
        ),
        distill=DistillConfig(
            enabled=bool(_get(raw, "distill.enabled", True)),
            dir=str(_get(raw, "distill.dir", "distill")),
            min_lines=int(_get(raw, "distill.min_lines", 2)),
            max_lines=int(_get(raw, "distill.max_lines", 3)),
            gap_paragraphs=int(_get(raw, "distill.gap_paragraphs", 3)),
            turn_gap=int(_get(raw, "distill.turn_gap", 20)),
            turn_steps=int(_get(raw, "distill.turn_steps", 0)),
            judge=str(_get(raw, "distill.judge", "rules")).lower(),
            judge_max_tokens=int(_get(raw, "distill.judge_max_tokens", 6000)),
            judge_scene_gap=int(_get(raw, "distill.judge_scene_gap", 6)),
            judge_max_lines=int(_get(raw, "distill.judge_max_lines", 24)),
            judge_cache=bool(_get(raw, "distill.judge_cache", True)),
            guess=str(_get(raw, "distill.guess", "chain")).lower(),
        ),
        channel=_get(raw, "adapter.channel", "onebot"),
        root=root,
        source_path=cfg_path.resolve() if cfg_path else None,
    )
    if cfg_path is not None:
        cfg.channel = _get(raw, "adapter.channel", cfg.channel)
    return cfg

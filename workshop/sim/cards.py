"""情境卡：结构、生成、解析。

卡只描述"条件"，不描述角色会怎么回答 —— 回答归人格库。
这样同一张卡换任何人格都能用，也不会把模型的想象写成设定。
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field

from .. import log
from ..relation import level_of_label
from ..react_style import parse_array as parse_json_array  # noqa: F401  旧 import 路径不断
from . import dims

log_ = log.get("sim")

# 模型很爱在 JSON 外面包一层 ```json，或者前面加一句"好的"。
_LEAD_NOISE = re.compile(r"^[^\[{]*")

# 三张卡大约 1200~1500 token（中文 + 键名很吃 token）。
# 之前给 1200，结果 JSON 正好在第三张卡中间被切断，解析全废。
CARD_MAX_TOKENS = 2200


@dataclass(slots=True)
class Card:
    name: str
    category: str
    relation: str
    setting: str
    other_role: str
    other_goal: str
    tension: str
    opening: list[str] = field(default_factory=list)
    id: str = ""
    created: str = ""
    origin: str = "generated"
    """generated = 现生成的；library = 从库里抽的。"""
    used: int = 0
    """被抽中/开演过几次。库里按它排序，用得多的排前面。"""
    saved_at: str = ""
    """收进库的时间。空表示还没收藏。"""

    # ---------- 序列化 ----------

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "relation": self.relation,
            "setting": self.setting,
            "other_role": self.other_role,
            "other_goal": self.other_goal,
            "tension": self.tension,
            "opening": list(self.opening),
            "created": self.created,
            "origin": self.origin,
            "used": self.used,
            "saved_at": self.saved_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "Card":
        return Card(
            name=str(d.get("name") or "未命名情境").strip()[:24],
            category=str(d.get("category") or "日常搭话"),
            relation=str(d.get("relation") or ""),
            setting=str(d.get("setting") or ""),
            other_role=str(d.get("other_role") or ""),
            other_goal=str(d.get("other_goal") or ""),
            tension=str(d.get("tension") or ""),
            opening=[str(x) for x in (d.get("opening") or []) if str(x).strip()][:3],
            id=str(d.get("id") or ""),
            created=str(d.get("created") or ""),
            origin=str(d.get("origin") or "generated"),
            used=int(d.get("used") or 0),
            saved_at=str(d.get("saved_at") or ""),
        )

    # ---------- 派生 ----------

    def scene(self) -> str:
        """渲染成注入用的场景描述。"""
        return (
            "## 当前场景\n"
            f"关系：{self.relation}\n"
            f"地点：{self.setting}\n"
            f"对方：{self.other_role}，{self.other_goal}\n"
            f"卡在哪：{self.tension}\n"
        )

    def pages(self) -> list[str]:
        pages = list(dims.CATEGORY_PAGES.get(self.category, ["speech"]))
        # 关系是三档之一的卡，也是关系页的素材
        if self.level and "relationships" not in pages:
            pages.append("relationships")
        return pages

    @property
    def level(self) -> str:
        """「陌生人」→ stranger。关系不是三档之一时返回空串。"""
        return level_of_label(self.relation)

    def summary(self) -> str:
        bits = [self.relation, self.setting]
        return "｜".join(b for b in bits if b)


def _clean_category(value: str) -> str:
    v = (value or "").strip()
    for c in dims.CATEGORIES:
        if v == c or c in v:
            return c
    return "日常搭话"


def normalize(raw: dict, index: int = 0) -> Card:
    """把模型吐出来的一条对象修成合法卡。缺字段就丢，不瞎补故事。"""
    card = Card.from_dict(raw)
    card.category = _clean_category(card.category)
    ts = time.strftime("%Y%m%d%H%M")
    # 同一分钟内生成两批卡很常见，只靠时间+序号会撞 id，收藏时互相覆盖
    card.id = card.id or f"c{ts}{index:02d}{random.randrange(0x10000):04x}"
    card.created = card.created or time.strftime("%Y-%m-%d %H:%M")
    return card


def parse_cards(text: str) -> list[Card]:
    """解析模型输出为卡列表，过滤缺关键字段的。"""
    out: list[Card] = []
    seen: set[str] = set()
    for i, item in enumerate(parse_json_array(text)):
        if not isinstance(item, dict):
            continue
        card = normalize(item, i)
        # 没有开场白的卡没法直接开演，属于废卡
        if not card.name or not card.opening:
            continue
        if card.name in seen:
            continue
        seen.add(card.name)
        out.append(card)
    return out


CARD_SYSTEM = """你在为一个人格素材库设计情境卡。

这些卡用来让一个具体的人（下文称"她"）开口说话，把她的反应沉淀成素材。
所以卡只写"条件"：谁、在哪、因为什么事、被逼到什么位置。
绝不写她会怎么回答、有什么心理活动——那是她本人的事。

硬要求：
- 具体。地点、关系、当场发生的事都要能立刻想象出画面。禁止"某天""两人交谈""发生了矛盾"这类抽象说法。
- 有压力。对方必须有明确诉求，而她又处在不好回答的位置。没有压力的场景产不出素材。
- 开场只写对方说的第一句话。口语，一到两句，不要旁白、不要动作描写、不要引号。
- 分类只能从下面选一个：日常搭话 / 情绪反应 / 冲突对峙 / 价值判断 / 回忆往事 / 关系试探 / 边界触碰 / 自我介绍。
- relation 只能从下面选一个，照抄原料里给的，不许自己换：拥有者 / 熟人 / 陌生人。
  这三档对她分别意味着谁、默认保持多远，看上文「这个人是谁」里的关系三档；不要自己发明关系。
- 场景必须落在她的生活里。她只以文字存在：场合就是某块屏幕上的聊天，
  不写线下地点、身体动作、外貌、天气。原料里的场合只当作"这段对话的性质"，按她的真实处境改写。
- 不要写任何她在设定里没有的东西（职业、同事、同学、城市、交通工具）。
  设定里没写的生活，就不要出现。
- name 是短标题，12 字以内，像给人看的目录名。
- 写短。relation / setting / tension / other_goal 各一句话，40 字以内；opening 只写一句。写长会把输出撑爆被截断。

只输出 JSON 数组，不要任何解释、不要代码块。每个元素字段：
{"name":"","category":"","relation":"","setting":"","other_role":"","other_goal":"","tension":"","opening":[""]}

其中 other_role 说明对方是谁（八个字内），other_goal 说明对方想拿到什么（一句话）。"""

# 母本为空时的补充说明。宁可写通用情境，也不让模型替她编一套人生。
NO_PROFILE = """## 这个人是谁
（还没有设定。只写普遍成立的聊天情境，不要编造身份、年龄、职业、住哪、身边有谁。）"""


def _system_prompt(profile: str) -> str:
    body = profile.strip()
    if not body:
        return CARD_SYSTEM + "\n\n" + NO_PROFILE
    return (
        CARD_SYSTEM
        + "\n\n## 这个人是谁（写卡必须贴着这些条件，不许改）\n"
        + body
    )


def _build_user_prompt(briefs: list[dict[str, str]], avoid: list[str] | None) -> str:
    user = (
        f"下面是 {len(briefs)} 组随机原料，一组写一张卡：\n\n"
        f"{dims.brief_text(briefs)}\n\n"
    )
    if avoid:
        names = "、".join(a for a in avoid[:24] if a)
        if names:
            user += f"已经有这些卡，别写重样的题目：{names}\n\n"
    user += f"输出 {len(briefs)} 个对象的 JSON 数组，顺序对应。relation 照抄原料里给的关系。"
    return user


async def _generate_once(
    llm,
    count: int,
    rng,
    avoid: list[str] | None,
    max_tokens: int,
    relation: str = "",
    profile: str = "",
) -> tuple[list[Card], str]:
    briefs = [dims.brief(rng, relation) for _ in range(max(1, count))]
    text = await llm.chat(
        [
            {"role": "system", "content": _system_prompt(profile)},
            {"role": "user", "content": _build_user_prompt(briefs, avoid)},
        ],
        temperature=1.0,
        max_tokens=max_tokens,
    )
    return parse_cards(text), text


async def generate_cards(
    llm,
    count: int = 3,
    rng=None,
    avoid: list[str] | None = None,
    max_tokens: int = CARD_MAX_TOKENS,
    relation: str = "",
    profile: str = "",
) -> list[Card]:
    """现生成若干张情境卡。avoid 是库里已有的题目，喂给模型避免撞题。

    失败不会直接报死：先看能不能从截断的输出里救回完整的卡，
    救不回来再退化成"只生成一张"重试一次——单张输出短，几乎必成。
    """
    cards, raw = await _generate_once(llm, count, rng, avoid, max_tokens, relation, profile)
    if not cards and count > 1:
        log_.warning("一次生成 %d 张失败，退化为单张重试", count)
        cards, raw = await _generate_once(
            llm, 1, rng, avoid, max(800, max_tokens // 2), relation, profile
        )

    if not cards:
        finish = getattr(llm, "last_finish", None)
        reason = (
            "输出被 token 上限截断"
            if finish == "length"
            else "输出里没有可解析的卡"
        )
        snippet = " ".join(raw.split())[:160]
        log_.warning("情境卡生成失败（%s，%d 字）：%s", reason, len(raw), snippet)
        raise ValueError(f"{reason}（返回 {len(raw)} 字）。把 sim.card_max_tokens 调大再试，或换一次")

    if relation:
        wrong = 0
        for c in cards:
            if c.relation not in dims.RELATIONS:
                # 没写或写了别的词，补成请求档
                c.relation = relation
            elif c.relation != relation:
                # 写了另一档：以卡面为准，标签跟着内容走，绝不硬贴
                wrong += 1
        if wrong:
            log_.warning("有 %d 张卡的关系与请求不符，按卡面保留（标签跟内容走）", wrong)

    if avoid:
        taken = set(avoid)
        cards = [c for c in cards if c.name not in taken] or cards
    return cards[:count]

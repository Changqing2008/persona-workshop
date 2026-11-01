"""关系档：主人 / 熟人 / 陌生人。

这个模块只做两件事：
1. 判定当前这条消息属于哪一档。
2. 从关系页抽出那一档的"说话方式"。

抽出来的原因是它有两个消费者：真实聊天（agent）和素材台（sim）。
两边必须用同一套判定与同一段语域，否则造出来的素材跟线上表现对不上。
"""

from __future__ import annotations

from typing import Iterable

# 档位名 → 展示名。素材台收稿头、覆盖看板、配置提示都用这套中文。
LABELS = {
    "owner": "拥有者",
    "familiar": "熟人",
    "stranger": "陌生人",
}

# 档位 → relationships.md 里对应的那一节。
# 第一项是标准名（agents.md 第八节定的）；后面的别名只为兼容早期模板产物——
# 早期模板把"对熟人 / 对陌生人"写成了"与熟人 / 与陌生人"，那一档的说话方式
# 就永远取不到，静默退回内置分寸。子串匹配，"与熟人"与"对熟人"互不命中，
# 只能靠这张表兜。不改老库文件，读取侧兼容。
SECTIONS_ALIASES = {
    "owner": ("与拥有者",),
    "familiar": ("对熟人", "与熟人"),
    "stranger": ("对陌生人", "与陌生人"),
}

# 兼容旧引用：只给标准名
SECTIONS = {k: v[0] for k, v in SECTIONS_ALIASES.items()}

LEVELS = tuple(LABELS)

# 反向表：卡片和成品里写的是中文，代码里用英文档位名
LEVEL_OF_LABEL = {v: k for k, v in LABELS.items()}


def level_of_label(text: str) -> str:
    """「陌生人」→ stranger。认不出返回空串。"""
    return LEVEL_OF_LABEL.get((text or "").strip(), "")


def level_of(
    user_id: str,
    owner_ids: Iterable[str],
    turns: int = 0,
    familiar_turns: int = 0,
    has_memory: bool | None = None,
) -> str:
    """判定档位。

    has_memory 决定熟人档的依据：
    - None  → 记忆层没启用，退回老规则（按聊过的条数）
    - True  → memory/ 里有这个人的记忆页且有事实 → 熟人
    - False → 没有共同经历 → 陌生人

    为什么不再用条数：聊得多不等于认识。群里刷了 50 条也可能是路人；
    只聊过三次但记住了他的生日和工作，才是处出来的关系。
    """
    ids = {str(i).strip() for i in owner_ids if str(i).strip()}
    if ids and str(user_id) in ids:
        return "owner"
    if has_memory is not None:
        return "familiar" if has_memory else "stranger"
    if familiar_turns > 0 and turns >= familiar_turns:
        return "familiar"
    return "stranger"


def label(level: str) -> str:
    return LABELS.get(level, "")


def voice(persona, level: str) -> str:
    """取该档的说话方式。关系页没填这一节就返回空串，调用方走内置分寸。"""
    if persona is None:
        return ""
    body = ""
    for heading in SECTIONS_ALIASES.get(level, ()):
        body = persona.section("relationships.md", heading)
        if body:
            break
    if not body:
        return ""
    return f"## 对他说话的方式\n{body}"

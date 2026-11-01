"""素材台：现生成情境，按身份母本模拟反应卡，产出可进 raw 的素材。

与 agent.py 的分工：
- agent.py 负责"真实聊天"，人格库只读、记忆落 SQLite。
- 本包负责"造素材"，不连任何通道，产出是 md 文件，由人拖进 raw。

唯一的共用点是 Persona.build_system_prompt —— 模拟与真机走同一条注入链路，
否则会出现"模拟台里像、QQ 里不像"的偏差。
"""

from .cards import Card, generate_cards, parse_cards
from .dims import CATEGORY_PAGES, brief, brief_text
from .studio import Draft, Studio, Turn

__all__ = [
    "Card",
    "Draft",
    "Studio",
    "Turn",
    "CATEGORY_PAGES",
    "brief",
    "brief_text",
    "generate_cards",
    "parse_cards",
]

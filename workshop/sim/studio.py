"""素材台内核：一稿的状态、按母本模拟反应卡、收稿。

设计上刻意不建数据库。
一段对话就是一份 json（过程）+ 一份 md（成品），文件即真相：
用户随时能用记事本打开改，程序重启也不会丢，导出后拖进 raw 就是素材。

三个约束：
- 人格库只读，本模块不写 index / wiki 任何一个字。
- 收稿只写 sim/drafts/，不碰 raw/。raw 的写入权归人。
- 注入链路与真实聊天共用 Persona.build_system_prompt。
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import log, tomlwrite, library
from .. import react_style as rs
from ..relation import LABELS as LEVEL_LABELS
from ..relation import LEVELS, level_of_label
from ..relation import voice as relation_voice
from ..seed import Seed
from .cards import Card, parse_json_array

log_ = log.get("sim")

_BAD_FILENAME = re.compile(r'[\\/:*?"<>|\r\n\t]+')

# 覆盖率看板的固定顺序，跟 agents.md 的目录结构一致
_PAGE_ORDER = {
    name: i
    for i, name in enumerate(
        [
            "identity.md",
            "traits.md",
            "speech.md",
            "reactions.md",
            "values.md",
            "relationships.md",
            "boundaries.md",
            "exemplars.md",
        ]
    )
}

# 三档在覆盖看板里的排序：总项在前，然后是拥有者 / 熟人 / 陌生人
# dialog / style 是 exemplars.md 拆出来的两项（原话段 / 行为范例张），借 level 槽位装
_DISTANCE_ORDER = {"": -1, "owner": 0, "familiar": 1, "stranger": 2, "dialog": 0, "style": 1}

CANDIDATE_HINT = """## 这一轮要做的事
下面是到目前为止的对话，其中「你」就是你自己说过的话。
请给出 {n} 条不同的说法，作为你接下来会说的那句话。
- 每条一到三句，口语，长度可以有差别
- 几条之间要明显不同：态度、是否反问、说多少，都别写成同一个腔调
- 不写旁白、动作、心理、解释，不加引号
- 只输出 JSON 数组，每条写成 {{"speaker":"你的名字","text":"那句台词"}}"""

# 「对方」这个词在两边各指一个人，模型会搞混（曾经把她的话当对面的话交上来）。
# 这里一律用真名 + "你/她"，把指代钉死。
# 另外 other_role 常常是她对那个人的称呼（"老周"），直接说"你在演老周"
# 会把模型带回她的视角，所以先声明"你是另一个人，她这么叫你"。
OTHER_SYSTEM = """你在演这场对话里的另一个人。不是助手，不是旁白，也不是{her}。

这一场对话只有两个人：
- 你：{other}（说「我」的那个人）
- 对面：{her}

{her}管你叫「{other}」；你是被她称呼的那一方，不是她本人。
你想要：{goal}
你们卡在：{tension}
关系：{relation}
场合：{setting}

规则：
- 只写你（{other}）会打出去的话，一到三句，口语，自称用「我」
- 绝不写{her}的台词：不替她回答，不复述她的话，不总结这段对话
- 她的自称、口头禅、句子长度都是她的，你一个字都别跟着用
- 不写旁白、动作、心理、括号里的补充说明，不加引号
- 可以追问、可以敷衍、可以装听不懂，也可以连发两条
- 你是在跟一个熟人说话，不是在服务用户"""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M")


def _safe(name: str, limit: int = 40) -> str:
    s = _BAD_FILENAME.sub("", name).strip().replace(" ", "")
    return (s or "未命名")[:limit]


@dataclass(slots=True)
class Turn:
    role: str
    """self = 人格本人；other = 对面那个人。"""
    text: str
    source: str
    """human = 人打的；ai = 采纳的候选；card = 卡片自带开场；novel = 蒸馏器带进来的小说原文。"""

    def as_dict(self) -> dict:
        return {"role": self.role, "text": self.text, "source": self.source}

    @staticmethod
    def from_dict(d: dict) -> "Turn":
        return Turn(
            role=str(d.get("role") or "other"),
            text=str(d.get("text") or ""),
            source=str(d.get("source") or "human"),
        )


@dataclass(slots=True)
class Draft:
    id: str
    card: Card
    mode: str = "other"
    """other = 我在演对方；self = 我在演她本人。"""
    turns: list[Turn] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    note: str = ""
    created: str = ""
    exported: str = ""
    """已收稿的对话稿文件名。空表示还没收。"""
    exported_reaction: str = ""
    """已收稿的反应卡文件名。空表示还没收。与 exported 各记各的，一份稿可以两个出口都出。"""
    reactions: dict = field(default_factory=dict)
    """反应卡状态：status / card / violations / reviewed / exemplar / example / at。

    和蒸馏台的 react.json 一个结构，只是键换成草稿 id。空表示还没转过。
    """

    # ---------- 序列化 ----------

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "card": self.card.as_dict(),
            "mode": self.mode,
            "turns": [t.as_dict() for t in self.turns],
            "dropped": list(self.dropped),
            "note": self.note,
            "created": self.created,
            "exported": self.exported,
            "exported_reaction": self.exported_reaction,
            "reactions": dict(self.reactions),
        }

    @staticmethod
    def from_dict(d: dict) -> "Draft":
        reactions = d.get("reactions")
        return Draft(
            id=str(d.get("id") or ""),
            card=Card.from_dict(d.get("card") or {}),
            mode=str(d.get("mode") or "other"),
            turns=[Turn.from_dict(t) for t in (d.get("turns") or []) if isinstance(t, dict)],
            dropped=[str(x) for x in (d.get("dropped") or [])],
            note=str(d.get("note") or ""),
            created=str(d.get("created") or ""),
            exported=str(d.get("exported") or ""),
            exported_reaction=str(d.get("exported_reaction") or ""),
            reactions=dict(reactions) if isinstance(reactions, dict) else {},
        )

    # ---------- 统计 ----------

    @property
    def ai_role(self) -> str:
        """AI 演的那一方：我演谁，反过来就是它。候选永远只出这一边。"""
        return "self" if self.mode == "other" else "other"

    def counts(self) -> dict[str, int]:
        human = sum(1 for t in self.turns if t.source == "human")
        ai = sum(1 for t in self.turns if t.source == "ai")
        novel = sum(1 for t in self.turns if t.source == "novel")
        return {
            "human": human,
            "ai": ai,
            "novel": novel,
            "turns": len(self.turns),
            "dropped": len(self.dropped),
        }

    def summary(self) -> dict:
        c = self.counts()
        r = self.reactions or {}
        return {
            "id": self.id,
            "name": self.card.name,
            "category": self.card.category,
            "mode": self.mode,
            "turns": c["turns"],
            "human": c["human"],
            "ai": c["ai"],
            "created": self.created,
            "has_reaction": bool(r.get("card")),
            "reaction_status": str(r.get("status") or ""),
        }


class Studio:
    """一次能开多稿的素材台。所有方法都是同步的，除候选生成外。"""

    def __init__(self, cfg, persona, llm) -> None:
        self.cfg = cfg
        self.persona = persona
        self.llm = llm
        self.root = cfg.path(cfg.sim.dir)
        self.card_dir = self.root / "cards"
        self.draft_dir = self.root / "drafts"
        for d in (self.root, self.card_dir, self.draft_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.self_label = cfg.sim.self_label.strip() or self._derive_label()
        # 母本：造料要贴着她本人，不能等 wiki 填完再造
        self.seed = Seed(
            persona.root,
            library.resolve_history(persona.root, cfg.path("data") / "identity_history"),
        )

    # ---------- 基础 ----------

    def _derive_label(self) -> str:
        """从人格库目录名取一个称呼，取不到就用「角色」。"""
        name = Path(self.persona.root).name or ""
        first = re.split(r"[.．·]\s*", name)[0].strip()
        return first[:12] or "角色"

    def _draft_path(self, draft_id: str) -> Path:
        return self.draft_dir / f"{draft_id}.json"

    def save(self, draft: Draft) -> None:
        path = self._draft_path(draft.id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(draft.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def get(self, draft_id: str) -> Draft | None:
        path = self._draft_path(draft_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log_.warning("草稿读取失败 %s: %s", draft_id, exc)
            return None
        return Draft.from_dict(data)

    def list_drafts(self) -> list[dict]:
        out: list[dict] = []
        for path in self.draft_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            draft = Draft.from_dict(data)
            if draft.id:
                item = draft.summary()
                item["exported"] = bool(draft.exported)
                item["exported_name"] = draft.exported
                item["exported_reaction"] = bool(draft.exported_reaction)
                item["exported_reaction_name"] = draft.exported_reaction
                out.append(item)
        out.sort(key=lambda x: x.get("created") or "", reverse=True)
        return out

    # ---------- 开稿 ----------

    def new_draft(self, card: Card, mode: str = "other", opening: bool = True) -> Draft:
        """开一稿。opening=False 时不落开场白——模拟反应卡那条路没有台词，落了也是废句。"""
        draft_id = time.strftime("%Y%m%d-%H%M%S") + "-" + f"{random.randint(0, 0xFFFF):04x}"
        draft = Draft(
            id=draft_id,
            card=card,
            mode="self" if mode == "self" else "other",
            turns=[],
            created=_now(),
        )
        # 卡片自带的开场白直接落成第一句，否则开局是空的，不知道谁先说话
        first = card.opening[0] if card.opening else ""
        if opening and first:
            draft.turns.append(Turn(role="other", text=first, source="card"))
        self.save(draft)
        # 从库里来的卡，开一次记一次，用来排"复用最多"
        if card.origin == "library":
            self.touch_card(card.id)
        return draft

    # ---------- 对演 ----------

    def _other_name(self, draft: Draft) -> str:
        """对面的称呼。卡上写了就用卡上的，没写才退回「对面那个人」。"""
        return (draft.card.other_role or "").strip() or "对面那个人"

    def _speaker(self, draft: Draft, role: str, viewer: str = "") -> str:
        """把某一方渲染成一句话里的称呼。

        viewer 是"这份记录给谁看"：给他自己那一侧写成「你」，
        另一侧用真名，避免「对方」同时指两个人。
        """
        if viewer and role == viewer:
            return "你"
        if role == "self":
            return self.self_label
        return f"对方（{self._other_name(draft)}）"

    def _transcript(self, draft: Draft, viewer: str = "") -> str:
        lines = []
        for t in draft.turns:
            lines.append(f"{self._speaker(draft, t.role, viewer)}：{t.text}")
        return "\n".join(lines)

    def _last_other(self, draft: Draft) -> str:
        for t in reversed(draft.turns):
            if t.role == "other":
                return t.text
        return draft.card.opening[0] if draft.card.opening else draft.card.name

    def _self_system(self, draft: Draft, n: int) -> str:
        """她这一侧的注入。带上卡的关系档，语域才跟线上一致。"""
        parts = [draft.card.scene()]
        profile = self.seed.brief(900)
        if profile:
            parts.append("## 这个人是谁\n" + profile)
        level = draft.card.level
        if level:
            parts.append(f"## 这段的关系\n对方是{draft.card.relation}。")
            spoken = relation_voice(self.persona, level)
            if spoken:
                parts.append(spoken)
        parts.append(CANDIDATE_HINT.format(n=n))
        return self.persona.build_system_prompt(
            self._last_other(draft), extra="\n\n".join(parts)
        )

    def _other_system(self, draft: Draft) -> str:
        card = draft.card
        return OTHER_SYSTEM.format(
            other=self._other_name(draft),
            her=self.self_label,
            goal=card.other_goal or "想弄明白一件事",
            tension=card.tension or "气氛有点僵",
            relation=card.relation or "熟人",
            setting=card.setting or "QQ 私聊",
        )

    async def candidates(self, draft: Draft, role: str, n: int = 3) -> list[str]:
        """生成候选台词。只出 AI 演的那一方，role 与草稿设定冲突时以草稿为准。"""
        n = max(1, min(int(n or 3), 5))
        role = "self" if role == "self" else "other"
        if role != draft.ai_role:
            log_.warning("候选角色 %s 与草稿设定 %s 不符，按草稿走", role, draft.ai_role)
            role = draft.ai_role

        name = self.self_label if role == "self" else self._other_name(draft)
        system = self._self_system(draft, n) if role == "self" else self._other_system(draft)
        transcript = self._transcript(draft, role) or "（还没开始，谁都没说话）"
        head = f"到目前为止的对话（「你」是你自己说过的话）：\n{transcript}\n\n"
        if role == "other":
            # 对面这一侧没有「母本」可用，只能拿他自己说过的话当口气锚点；
            # 同时把她的最后一句点明为"她的话"，否则模型会顺着她的腔调写。
            his = [t.text for t in draft.turns if t.role == "other"][-3:]
            hers = [t.text for t in draft.turns if t.role == "self"]
            if his:
                head += "你之前说过的（照这个口气，用「我」）：\n" + "\n".join(
                    f"- {x}" for x in his
                ) + "\n\n"
            if hers:
                head += f"她刚说的是：「{hers[-1]}」——那是她的话，别顺着她的口气写。\n\n"
        user = (
            head
            + f"现在轮到你（{name}）说话。给出 {n} 条不同的说法。\n"
            + f'只输出 JSON 数组，每条写成 {{"speaker":"{name}","text":"那句台词"}}'
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        # 模型偶尔会滑回她的腔调，或者复读已经说过的话。
        # 第一次出不来干净的，就带一句更硬的约束再要一次。
        said = {t.text.strip() for t in draft.turns}
        items: list[str] = []
        for attempt in (1, 2):
            text = await self.llm.chat(
                messages,
                temperature=max(0.8, float(getattr(self.cfg.sim, "temperature", 1.0))),
                max_tokens=400,
            )
            items = [
                x for x in _parse_candidates(
                    text, expect=role, her=self.self_label, other=self._other_name(draft)
                )
                if x not in said
            ]
            if items:
                return items[:n]
            if attempt == 1:
                log_.warning("第一次没拿到%s的台词，加硬约束重试一次", name)
            messages = messages[:1] + [
                {
                    "role": "user",
                    "content": user
                    + f"\n\n注意：只写{name}的话。上面这些已经是她说过的话，不要再重复。",
                }
            ]
        raise ValueError(f"模型没能给出{name}的台词，重试一次或自己写")

    def commit(
        self,
        draft: Draft,
        role: str,
        text: str,
        source: str = "human",
        dropped: list[str] | None = None,
    ) -> Draft:
        text = (text or "").strip()
        if text:
            draft.turns.append(
                Turn(role="self" if role == "self" else "other", text=text, source=source)
            )
        for item in dropped or []:
            item = (item or "").strip()
            if item and item != text:
                draft.dropped.append(item)
        self.save(draft)
        return draft

    def undo(self, draft: Draft) -> Draft:
        if draft.turns:
            draft.turns.pop()
        self.save(draft)
        return draft

    def edit_turn(self, draft: Draft, index: int, text: str) -> Draft:
        """改写已经说过的一句。index 是 turns 下标；改成空文本等于删掉这句。

        改过的句子来源一律记成 human：既然人手改过，它就该按原话算进成品统计，
        不能再顶着"AI 采纳"的名头。
        """
        if not 0 <= index < len(draft.turns):
            return draft
        text = (text or "").strip()
        if not text:
            draft.turns.pop(index)
        else:
            turn = draft.turns[index]
            if turn.text.strip() == text:
                return draft
            turn.text = text
            turn.source = "human"
        self.save(draft)
        return draft

    def set_turn_role(self, draft: Draft, index: int, role: str) -> Draft:
        """改「这句是谁说的」。人手动过的，来源一律记回 human。"""
        if not 0 <= index < len(draft.turns):
            return draft
        role = "self" if role == "self" else "other"
        turn = draft.turns[index]
        if turn.role == role:
            return draft
        turn.role = role
        turn.source = "human"
        self.save(draft)
        return draft

    def insert_turn(
        self, draft: Draft, at: int, role: str, text: str, source: str = "human"
    ) -> Draft:
        """在 at 之前插一句；at 落到末尾之外就是追加。空文本不插。"""
        text = (text or "").strip()
        if not text:
            return draft
        at = max(0, min(int(at), len(draft.turns)))
        draft.turns.insert(
            at, Turn(role="self" if role == "self" else "other", text=text, source=source)
        )
        self.save(draft)
        return draft

    def move_turn(self, draft: Draft, index: int, delta: int) -> Draft:
        """上下挪一句，把对话顺序调顺。来源标记跟着句子走，不改写。"""
        j = index + (1 if delta > 0 else -1)
        if not (0 <= index < len(draft.turns)) or not (0 <= j < len(draft.turns)):
            return draft
        draft.turns[index], draft.turns[j] = draft.turns[j], draft.turns[index]
        self.save(draft)
        return draft

    def set_card(self, draft: Draft, card: Card) -> Draft:
        """换掉这一稿的情境条件。

        条件本身就是注入的一部分（见 _self_system），所以改完对下一轮候选即时生效：
        把卡点从"她不想提"改成"她必须在十分钟内答复"，出来的话就不一样。
        """
        if not card.name:
            raise ValueError("卡片标题不能为空")
        draft.card = card
        self.save(draft)
        return draft

    def set_mode(self, draft: Draft, mode: str) -> Draft:
        """改「我在演谁」。这是谁说话的单一真相：候选出哪一边、标签写谁，全看它。

        换边不是改历史，已经落下的句子原样留着，只影响下一轮。
        """
        mode = "self" if mode == "self" else "other"
        if mode != draft.mode:
            draft.mode = mode
            self.save(draft)
            log_.info("对演换边：我在演%s，候选改出%s", mode, draft.ai_role)
        return draft

    # ---------- 收稿 ----------

    def _render(self, draft: Draft) -> str:
        """成品 markdown。头部标注决定归纳AI 怎么读，字段别随意增删。"""
        date = time.strftime("%Y-%m-%d")
        c = draft.counts()
        pages = " / ".join(f"{p}.md" for p in draft.card.pages())
        head = [
            f"# {date} {draft.card.name}",
            "",
            f"> 用途：{pages}",
            f"> 分类：{draft.card.category}",
            # 关系留空时写占位符，不要写成一个空的首段：ingest 靠这一段判档
            f"> 情境：{draft.card.relation or '—'}｜{draft.card.setting}",
            f"> 张力：{draft.card.tension}",
            f"> 对方：{draft.card.other_role}（{draft.card.other_goal}）",
            f"> 构成：原文 {c['novel']} 句 / 人工原话 {c['human']} 句 / AI 采纳 {c['ai']} 句 / 未采纳 {c['dropped']} 条",
            f"> 生成：素材台 {draft.created}",
        ]
        if draft.note.strip():
            head.append(f"> 批注：{draft.note.strip()}")
        body = [
            f"**{self.self_label if t.role == 'self' else '对方'}**：{t.text}"
            for t in draft.turns
        ]
        return "\n".join(head) + "\n\n" + "\n\n".join(body) + "\n"

    def export(self, draft: Draft, form: str = "dialog") -> dict[str, Any]:
        """收稿。两个出口，都只落 sim/drafts/，绝不写 raw —— 那一步归人。

        form="dialog"：对话稿，带场合/对方/台词。能喂 exemplars 的原话区。
        form="reaction"：反应卡，只有"怎么回应"。人勾了范例才追加 exemplars。
        """
        if str(form or "dialog").strip().lower() in {"reaction", "react", "style"}:
            return self.export_reaction(draft)
        return self._export_dialog(draft)

    def _export_dialog(self, draft: Draft) -> dict[str, Any]:
        """写一份对话稿 md 到 sim/drafts/。"""
        date = time.strftime("%Y-%m-%d")
        base = f"{date}_{_safe(draft.card.name)}"
        path = self.draft_dir / f"{base}.md"
        n = 2
        while path.exists():
            path = self.draft_dir / f"{base}_{n}.md"
            n += 1

        markdown = self._render(draft)
        path.write_text(markdown, encoding="utf-8")
        draft.exported = path.name
        self.save(draft)
        log_.info("已收稿（对话稿）：%s（%d 句）", path.name, draft.counts()["turns"])
        return {"path": str(path), "name": path.name, "markdown": markdown, "form": "dialog"}

    def preview(self, draft: Draft) -> str:
        """不落盘，只看成品长什么样。"""
        return self._render(draft)

    # ---------- 反应卡 ----------
    #
    # 与对话稿的分工：
    # 对话稿产出「她说过什么」，带场合与对方，能喂 exemplars 的原话区；
    # 反应卡只产出「遇到这类刺激怎么回」，成品里没有名字、没有场合、没有一句台词。
    #
    # 内核与蒸馏台共用 bridge/react_style.py：同一套 SYSTEM、同一套出口校验、同一份头部。
    # 两条产线只在「证据行」「生成者署名」上不同。

    def _reaction_seg(self, draft: Draft) -> dict:
        """把草稿摊成 react_style 认识的段结构。空句不送——送了只会让模型复述空行。"""
        return {
            "lines": [
                {"role": t.role, "text": t.text, "action": ""}
                for t in draft.turns
                if (t.text or "").strip()
            ]
        }

    def _reaction_names(self, draft: Draft) -> tuple[list[str], str]:
        """要遮掉的名字：她的显示名 + 卡上对那人的称呼。

        卡上没有称呼时返回空，不要拿「对面那个人」这种占位词去当专名拦——
        那会把模型正常的指代判成违规。这条路上本来也只剩她自己的名字要防。
        """
        her = self.self_label
        other = (draft.card.other_role or "").strip()
        return [n for n in dict.fromkeys([her, other]) if n], her

    def _reaction_level(self, draft: Draft) -> str:
        """卡的关系档 → 反应卡的中文档位。不是三档之一就返回空（不分档）。"""
        return LEVEL_LABELS.get(draft.card.level, "")

    def _render_reaction(self, draft: Draft) -> str:
        entry = draft.reactions or {}
        c = draft.counts()
        if c["turns"]:
            evidence = (
                f"素材台对演 {draft.id}（人工原话 {c['human']} 句 / AI 采纳 {c['ai']} 句"
                + (f" / 未采纳 {c['dropped']} 条" if c["dropped"] else "")
                + "）"
            )
        else:
            # 没有台词：来源只有母本。写明"推断"，让归纳AI 知道这不是她说过的话。
            evidence = f"素材台·母本推断 {draft.id}（按身份母本推，无台词依据）"
        head = rs.render_head(
            entry.get("card") or {},
            [evidence],
            bool(entry.get("reviewed")),
            exemplar=bool(entry.get("exemplar")),
            example=str(entry.get("example") or ""),
            generated_by="素材台·反应样式",
            level_hint=self._reaction_level(draft),
        )
        return "\n".join(head) + "\n"

    def preview_reaction(self, draft: Draft) -> str:
        """不落盘，只看反应卡长什么样。没转过就返回空串。"""
        if not (draft.reactions or {}).get("card"):
            return ""
        return self._render_reaction(draft)

    async def _abstract(
        self, draft: Draft, system: str, user: str, source: str, names: list[str]
    ) -> dict:
        """跑一次抽象并落盘。两个入口（对演稿 / 卡片条件）共用同一套重试与校验。

        第一次不干净就带着"具体哪里不行"再要一次；两次都不行标 blocked，不硬放行。
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        card: dict[str, Any] = {}
        hits: list[str] = []
        for _attempt in (1, 2):
            text = await self.llm.chat(messages, temperature=0.3, max_tokens=700)
            card = rs.parse(text)
            if not card:
                hits = ["模型没按 JSON 输出"]
                messages = messages[:1] + [
                    {"role": "user", "content": user + "\n\n只输出 JSON 对象，不要解释、不要代码块。"}
                ]
                continue
            hits = rs.violations(card, names, source)
            if not hits:
                break
            messages = messages[:1] + [
                {
                    "role": "user",
                    "content": user
                    + "\n\n上一次输出里有这些不能出现的东西："
                    + "；".join(hits)
                    + "。重写，去掉它们，只输出 JSON。",
                }
            ]

        status = "ok" if card and not hits else "blocked"
        prior = draft.reactions or {}
        # 重转等于换了一张卡：审与范例标记一律作废，例句留着（那是人写的，不该被模型覆盖）
        draft.reactions = {
            "status": status,
            "card": card,
            "violations": hits,
            "reviewed": False,
            "exemplar": False,
            "example": str(prior.get("example") or ""),
            "at": _now(),
        }
        self.save(draft)
        log_.info("反应卡 %s：%s%s", draft.id, status, f"（{hits[0]}）" if hits else "")
        return {
            "ok": True,
            "status": status,
            "reaction": draft.reactions,
            "violations": hits,
            "preview": self._render_reaction(draft) if card else "",
        }

    def _simulate_source(self, draft: Draft) -> str:
        """重合校验的基准：卡上的场合 / 卡点 / 对方诉求 / 开场白。

        模型看得到这些条件，很容易把原句抄进 scene。抄了就说明它没抽象。
        """
        c = draft.card
        return "\n".join(
            [
                c.setting or "",
                c.tension or "",
                c.other_goal or "",
                (c.opening[0] if c.opening else ""),
            ]
        )

    async def simulate(self, draft: Draft) -> dict:
        """按母本 + 卡片条件直接推一张反应卡。不需要台词，也不需要人先对演。

        和对演稿那条路的差别只在一个地方：依据不是"她说过什么"，而是"她是谁"。
        所以母本里「说话习惯」那一节是话量 / 句长 / 语言的唯一来源——母本空着，
        这三项就该留空，而不是让模型编一套腔调。
        """
        if self.llm is None or not getattr(self.llm, "ready", False):
            raise ValueError("缺少 API Key：模拟反应卡要调模型，先在「配置」页填上")

        card = draft.card
        profile = self.seed.brief(1200)
        opening = card.opening[0] if card.opening else ""
        user = rs.build_simulate_user(
            profile, card.relation, card.setting, card.other_goal, card.tension, opening
        )
        names, _her = self._reaction_names(draft)
        return await self._abstract(
            draft, rs.SYSTEM_SIMULATE, user, self._simulate_source(draft), names
        )

    def _card_from_reaction(self, card: dict[str, Any], level: str) -> Card:
        """反应卡倒推出一张最小的卡。

        2026-10-27 之后素材台没有情境卡了，但 Draft 仍然要挂一张 Card：
        `name` 用来命名成品文件，`relation` 用来出档位。
        场合 / 卡点 / 对方 / 开场白一个都不填——这条路上没有它们的位置，
        填了就是留一条后门，早晚会顺着成品头漏进 wiki。
        """
        return Card(
            name=(card.get("scene") or "反应卡")[:24],
            category="反应卡",
            relation=(card.get("level") or level or ""),
            setting="",
            other_role="",
            other_goal="",
            tension="",
            opening=[],
        )

    async def generate_reactions(
        self, stimulus: str = "", level: str = "", n: int = 3
    ) -> list[dict[str, Any]]:
        """按母本 + 刺激/档位直接出 n 张反应卡，每张落成一稿。

        这是素材台现在的主路。输入只有两样东西：身份母本、两个下拉。
        没有场合、没有对方、没有开场白——不传进来，模型就编不出来，
        也就不存在"条件被抄进成品头"这条路。

        一次调用要 n 张而不是一张一张要：省往返，而且几张摆在一起模型才会真去差异化。
        """
        if self.llm is None or not getattr(self.llm, "ready", False):
            raise ValueError("缺少 API Key：生成反应卡要调模型，先在「配置」页填上")

        n = max(1, min(int(n or 3), 6))
        stimulus = (stimulus or "").strip()
        level = (level or "").strip()
        if stimulus and stimulus not in rs.STIMULI:
            raise ValueError(f"刺激类型只能是：{' / '.join(rs.STIMULI)}")
        if level and level not in LEVEL_LABELS.values():
            raise ValueError(f"关系档只能是：{' / '.join(LEVEL_LABELS.values())}")

        user = rs.build_batch_user(self.seed.brief(1200), stimulus, level, n)
        messages = [
            {"role": "system", "content": rs.SYSTEM_BATCH.format(n=n)},
            {"role": "user", "content": user},
        ]
        cards: list[dict[str, Any]] = []
        for _attempt in (1, 2):
            text = await self.llm.chat(messages, temperature=0.85, max_tokens=2600)
            cards = rs.parse_many(text)
            if cards:
                break
            messages = messages[:1] + [
                {
                    "role": "user",
                    "content": user + "\n\n只输出 JSON 数组，不要解释、不要代码块。",
                }
            ]
        if not cards:
            raise ValueError("模型没按 JSON 数组输出，再点一次「生成反应卡」试试")

        names = [x for x in dict.fromkeys([self.self_label]) if x]
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for card in cards[:n]:
            # 下拉里指定了就以指定为准：模型偶尔会换一个，界面上对不上就是骗人
            if stimulus:
                card["stimulus"] = stimulus
            if level:
                card["level"] = level
            key = (card["stimulus"], card["scene"])
            if key in seen:
                continue
            seen.add(key)
            draft = self.new_draft(self._card_from_reaction(card, level), mode="other", opening=False)
            # 没有条件原文可比，重合校验这条就空着跑：只剩专名与关系称呼两道
            hits = rs.violations(card, names, "")
            draft.reactions = {
                "status": "ok" if not hits else "blocked",
                "card": card,
                "violations": hits,
                "reviewed": False,
                "exemplar": False,
                "example": "",
                "at": _now(),
            }
            self.save(draft)
            out.append(
                {
                    "id": draft.id,
                    "name": draft.card.name,
                    "stimulus": card["stimulus"],
                    "level": card["level"] or "不分档",
                    "scene": card["scene"],
                    "status": draft.reactions["status"],
                    "violations": hits,
                    "preview": self._render_reaction(draft),
                    "draft": draft.as_dict(),
                }
            )

        if not out:
            raise ValueError("这一批全是重复的，换一组刺激或档位再试")
        log_.info("批量反应卡：要 %s 张，成 %s 张", n, len(out))
        return out

    async def to_reaction(self, draft: Draft) -> dict:
        """把已有的对演稿抽象成反应卡。台词先遮名字，再送模型，出口再校验一遍。

        对演面板已经撤了，这条入口留给老稿和蒸馏台带进来的段。
        """
        if self.llm is None or not getattr(self.llm, "ready", False):
            raise ValueError("缺少 API Key：反应卡要调模型做抽象，先在「配置」页填上")

        seg = self._reaction_seg(draft)
        if not seg["lines"]:
            raise ValueError("这稿还没有台词，用「模拟反应卡」直接按母本生成")

        names, her = self._reaction_names(draft)
        masked, source = rs.mask(seg, names, her)
        user = rs.USER.format(category=draft.card.category or "日常搭话", lines=masked)
        return await self._abstract(draft, rs.SYSTEM, user, source, names)

    async def generate_example(self, draft: Draft) -> str:
        """给反应卡生成例句：一句她的原话，≤40 字，能直接当 few-shot。

        例句校验只查专名（关系称呼不查——原话里带称呼是常态）。
        所以模型回来先过 example_hits：带名字 / 超长 / 空输出，
        就带着原因重试一次；两次都不干净就报错让人手写，
        绝不把污染句塞进 exemplars。
        """
        if self.llm is None or not getattr(self.llm, "ready", False):
            raise ValueError("缺少 API Key：生成例句要调模型，先在「配置」页填上")
        card = dict((draft.reactions or {}).get("card") or {})
        if not card:
            raise ValueError("这稿还没转反应卡，先转一次")
        names, _her = self._reaction_names(draft)

        user = "\n".join(
            [
                "## 她是谁",
                self.seed.brief(900),
                "",
                "## 这张反应卡",
                f"- 刺激：{card.get('stimulus') or ''}（场景：{card.get('scene') or ''}）",
                f"- 情绪：{card.get('emotion_from') or ''} → {card.get('emotion_to') or ''}，露出：{card.get('emotion_shown') or ''}",
                f"- 回应：{'；'.join(card.get('moves') or [])}",
                f"- 语言：{'；'.join(card.get('language') or [])}",
                f"- 不做：{'；'.join(card.get('avoid') or [])}",
                "",
                "## 任务",
                "替她说一句面对这种刺激时会说的话。要求：",
                "- 一句原话，第一人称，不超过 40 个字",
                "- 贴着上面的回应方式与语言习惯走，说了她不会说的就是废句",
                "- 不得出现任何人名、地名、机构名；关系称呼（爸/姐/老师这类）可以有",
                "- 只输出那一句话，不要解释，不要用引号把整句包起来",
            ]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你在给一张反应卡补例句。例句是她的原话，将作为 few-shot"
                    "示范她怎么说话，所以一个不该出现的名字都是污染。只输出一句话。"
                ),
            },
            {"role": "user", "content": user},
        ]
        reason = ""
        for _attempt in (1, 2):
            text = (await self.llm.chat(messages, temperature=0.85, max_tokens=120)).strip()
            text = text.strip("\"“”'「」『』 \t")
            text = text.splitlines()[0].strip() if text else ""
            hits = rs.example_hits(text, names)
            if not text:
                reason = "上一次输出是空的"
            elif len(text) > 40:
                reason = f"上一次写了 {len(text)} 个字"
            elif hits:
                reason = "上一次出现了名字：" + "；".join(hits)
            else:
                return text
            messages = messages[:2] + [
                {"role": "user", "content": user + f"\n\n重来。上次不合格：{reason}。"}
            ]
        raise ValueError(f"例句两次都不合格（{reason}），手写一句吧")

    def update_reaction(
        self,
        draft: Draft,
        card: dict | None = None,
        reviewed: bool | None = None,
        exemplar: bool | None = None,
        example: str | None = None,
    ) -> Draft:
        """手改卡、勾审、勾范例、填例句。改完重跑校验，违规就退回 blocked。

        顺序不能反：没过审的卡不许标范例，否则等于用一个开关绕开第三道闸。
        """
        entry = dict(draft.reactions or {})
        if not entry.get("card"):
            raise ValueError("这稿还没转反应卡，先转一次")

        names, her = self._reaction_names(draft)
        _masked, source = rs.mask(self._reaction_seg(draft), names, her)

        if isinstance(card, dict):
            merged = dict(entry.get("card") or {})
            merged.update({k: v for k, v in card.items() if k in rs.normalize({}).keys()})
            entry["card"] = rs.normalize(merged)
            entry["violations"] = rs.violations(entry["card"], names, source)
            entry["status"] = "ok" if not entry["violations"] else "blocked"
            if entry["status"] == "blocked":
                entry["reviewed"] = False
                entry["exemplar"] = False

        if example is not None:
            text = str(example or "").strip()[:40]
            hits = rs.example_hits(text, names)
            if hits:
                raise ValueError("例句里" + "；".join(hits) + "——例句不写人名，换一句")
            entry["example"] = text

        if reviewed is not None:
            if reviewed and entry.get("status") != "ok":
                raise ValueError("还有违规内容，改完再标已审")
            entry["reviewed"] = bool(reviewed)
            if not entry["reviewed"]:
                entry["exemplar"] = False

        if exemplar is not None:
            want = bool(exemplar)
            if want and not entry.get("reviewed"):
                raise ValueError("先勾「已过审」，再把这张标成可作范例")
            if want:
                hits = rs.example_hits(str(entry.get("example") or ""), names)
                if hits:
                    raise ValueError("例句里" + "；".join(hits) + "，先改掉再标范例")
            entry["exemplar"] = want

        entry["at"] = _now()
        draft.reactions = entry
        self.save(draft)
        return draft

    def export_reaction(self, draft: Draft) -> dict[str, Any]:
        """收反应卡。没转、有违规、没过审，三种情况一律不收——这是最后一道闸。"""
        entry = draft.reactions or {}
        if not entry.get("card"):
            raise ValueError("这稿还没转反应卡，先转一次")
        if entry.get("status") != "ok":
            raise ValueError("卡片还有违规内容，改干净再收")
        if not entry.get("reviewed"):
            raise ValueError("先勾「已过审」再收反应卡")

        # 例题兜底：改过卡片之后例句可能夹带名字，收稿前再查一遍
        example = str(entry.get("example") or "")
        names, _her = self._reaction_names(draft)
        hits = rs.example_hits(example, names)
        if hits:
            raise ValueError("例句里" + "；".join(hits) + "，先改掉再收")

        stimulus = _safe(str((entry.get("card") or {}).get("stimulus") or "未分类"), 8)
        base = f"{time.strftime('%Y-%m-%d')}_反应_{stimulus}_{_safe(draft.card.name, 16)}"
        path = self.draft_dir / f"{base}.md"
        n = 2
        while path.exists():
            path = self.draft_dir / f"{base}_{n}.md"
            n += 1

        markdown = self._render_reaction(draft)
        path.write_text(markdown, encoding="utf-8")
        draft.exported_reaction = path.name
        self.save(draft)
        log_.info(
            "已收稿（反应卡）：%s%s",
            path.name,
            "（含范例）" if entry.get("exemplar") else "",
        )
        return {
            "path": str(path),
            "name": path.name,
            "markdown": markdown,
            "form": "reaction",
            "exemplar": bool(entry.get("exemplar")),
        }


    # ---------- 卡片库 ----------
    #
    # 库放在 sim/cards/，与人格库目录无关：
    # 卡只写条件（谁、在哪、卡在哪），不写她怎么回答，
    # 所以换一个人格库照样能用，这也是它值得被收藏的原因。

    def _card_path(self, card_id: str) -> Path | None:
        """卡 id 只允许字母数字下划线连字符，挡住路径穿越。"""
        safe = re.sub(r"[^A-Za-z0-9_-]", "", card_id or "")[:40]
        if not safe:
            return None
        return self.card_dir / f"{safe}.toml"

    def library(self) -> list[Card]:
        import tomllib

        cards: list[Card] = []
        for path in self.card_dir.glob("*.toml"):
            try:
                with path.open("rb") as fh:
                    data = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                log_.warning("卡片读取失败 %s: %s", path.name, exc)
                continue
            card = Card.from_dict(data)
            card.id = card.id or path.stem
            if not card.saved_at:
                card.saved_at = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)
                )
            cards.append(card)
        cards.sort(key=lambda c: c.saved_at, reverse=True)
        cards.sort(key=lambda c: -c.used)
        return cards

    def save_card(self, card: Card) -> dict[str, Any]:
        """收进库。同名视为重复，幂等返回，不产生第二份。"""
        if not card.id:
            card.id = f"c{time.strftime('%Y%m%d%H%M')}{random.randrange(0x10000):04x}"
        path = self._card_path(card.id)
        if path is None:
            return {"ok": False, "error": "卡片 id 非法"}

        for other in self.library():
            if other.name == card.name and other.id != card.id:
                return {
                    "ok": True,
                    "existed": True,
                    "id": other.id,
                    "path": str(self._card_path(other.id) or ""),
                }

        existed = path.exists()
        card.saved_at = card.saved_at or _now()
        card.origin = card.origin or "generated"
        path.write_text(tomlwrite.dumps(card.as_dict()), encoding="utf-8")
        log_.info("卡片%s：%s", "更新" if existed else "入库", card.name)
        return {"ok": True, "existed": existed, "id": card.id, "path": str(path)}

    def delete_card(self, card_id: str) -> bool:
        path = self._card_path(card_id)
        if path is None or not path.is_file():
            return False
        try:
            path.unlink()
        except OSError as exc:
            log_.warning("删除卡片失败 %s: %s", card_id, exc)
            return False
        log_.info("卡片出库：%s", path.stem)
        return True

    def touch_card(self, card_id: str) -> None:
        """用一次记一次，库里按使用次数排序，"优质"就不用额外打分。

        只读这一张卡的文件。以前走 library() 全库扫描，
        而它每次开稿都被调用一次——库大了就是白读几百个 toml。
        """
        import tomllib

        path = self._card_path(card_id)
        if path is None or not path.is_file():
            return
        try:
            with path.open("rb") as fh:
                card = Card.from_dict(tomllib.load(fh))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log_.warning("卡片读取失败 %s: %s", card_id, exc)
            return
        card.id = card.id or path.stem
        card.used += 1
        try:
            path.write_text(tomlwrite.dumps(card.as_dict()), encoding="utf-8")
        except OSError as exc:
            log_.warning("卡片计数写回失败 %s: %s", card_id, exc)

    def draw_cards(self, n: int = 3) -> list[Card]:
        """从库里抽 n 张。抽中即计数，抽到的最多半分钟内还是那几张，不会重样。"""
        lib = self.library()
        if not lib:
            return []
        n = max(1, min(int(n or 3), len(lib)))
        picked = lib[:n] if len(lib) <= n else random.sample(lib, n)
        for card in picked:
            card.origin = "library"
        return picked

    # ---------- 覆盖率 ----------

    @staticmethod
    def _slot(page: str, level: str = "", label: str = "") -> dict:
        if not label:
            base = page[:-3] if page.endswith(".md") else page
            label = f"关系·{LEVEL_LABELS.get(level, level)}" if level else base
        return {
            "page": page,
            "level": level,
            "label": label,
            "segments": 0,
            "turns": 0,
            "last": "",
        }

    def coverage(self) -> list[dict]:
        """按用途页统计已收稿素材。

        关系页额外按三档拆开统计：写了"关系"但三档全缺，跟没写其实一样。
        三档条目常驻显示（哪怕 0 段），缺哪一档一眼看到。

        exemplars 也拆两项：`原话`（对话稿，能当 few-shot）与 `行为范例`（反应卡勾了范例）。
        两项都真实显示，不给"这页填了"的假完成。
        """
        pages: dict[tuple[str, str], dict] = {}
        names = list(self.persona.status().keys()) or [
            "identity.md",
            "traits.md",
            "speech.md",
            "reactions.md",
            "values.md",
            "relationships.md",
            "boundaries.md",
            "exemplars.md",
        ]
        for name in names:
            pages[(name, "")] = self._slot(name)
        for d in LEVELS:
            pages.setdefault(("relationships.md", d), self._slot("relationships.md", d))
        for key, label in (("dialog", "exemplars·原话"), ("style", "exemplars·行为范例")):
            pages.setdefault(("exemplars.md", key), self._slot("exemplars.md", key, label))

        for path in self.draft_dir.glob("*.md"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            turns = sum(1 for line in text.splitlines() if line.startswith("**"))
            used: list[str] = []
            dist = ""
            is_style = False
            # 反应卡头部比对话稿长（含档位/刺激/场景…），窗口给到 24 行
            for line in text.splitlines()[:24]:
                if line.startswith("> 形态："):
                    is_style = "反应样式" in line
                elif line.startswith("> 用途："):
                    used = [x.strip() for x in line[5:].split("/") if x.strip()]
                elif line.startswith("> 档位：") and not dist:
                    # 反应卡的档位就是关系档，和对话稿的「情境」首段同权
                    dist = level_of_label(line[5:].strip())
                elif line.startswith("> 情境：") and not dist:
                    # 情境的第一段就是关系档，例如「陌生人｜深夜的微信对话框」
                    dist = level_of_label(line[5:].split("｜")[0])
            day = path.name[:10]
            for page in used:
                slot = pages.setdefault((page, ""), self._slot(page))
                slot["segments"] += 1
                slot["turns"] += turns
                slot["last"] = max(slot["last"], day)
                if page == "relationships.md" and dist:
                    dslot = pages.setdefault((page, dist), self._slot(page, dist))
                    dslot["segments"] += 1
                    dslot["turns"] += turns
                    dslot["last"] = max(dslot["last"], day)
                if page == "exemplars.md":
                    key = "style" if is_style else "dialog"
                    xslot = pages.setdefault(("exemplars.md", key), self._slot("exemplars.md", key))
                    xslot["segments"] += 1
                    xslot["turns"] += turns
                    xslot["last"] = max(xslot["last"], day)

        return sorted(
            pages.values(),
            key=lambda x: (
                _PAGE_ORDER.get(x["page"], 99),
                _DISTANCE_ORDER.get(x["level"], -1),
                x["page"],
            ),
        )


def _speaker_side(speaker: str, her: str = "", other: str = "") -> str:
    """判断一条候选标的是谁在说。认不出来就返回空串（不过滤）。"""
    s = (speaker or "").strip()
    if not s:
        return ""
    if her and her in s:
        return "self"
    if other and other in s:
        return "other"
    low = s.lower()
    if low in {"self", "她", "她本人", "本人", "assistant"}:
        return "self"
    if low in {"other", "对方", "你", "user", "对面"}:
        return "other"
    return ""


def _parse_candidates(
    text: str, expect: str = "", her: str = "", other: str = ""
) -> list[str]:
    """解析候选数组。模型偶尔不听话，退化成按行切。

    expect 是"这一轮该谁说话"：候选自带的 speaker 明确指向另一边时直接丢掉，
    这是防止「我在演她、AI 却把她的台词当对面交上来」的最后一道闸。
    """
    data = parse_json_array(text)
    out: list[str] = []
    wrong = 0
    for item in data:
        if isinstance(item, str) and item.strip():
            out.append(_tidy(item))
        elif isinstance(item, dict):
            if expect:
                side = _speaker_side(
                    str(item.get("speaker") or item.get("who") or ""), her, other
                )
                if side and side != expect:
                    wrong += 1
                    continue
            for key in ("text", "reply", "content", "line"):
                if isinstance(item.get(key), str) and item[key].strip():
                    out.append(_tidy(item[key]))
                    break

    if wrong:
        log_.warning("丢弃 %d 条说话人不对的候选", wrong)

    if not out:
        for line in (text or "").splitlines():
            s = re.sub(r"^\s*(?:[-*]|\d+[.、)）])\s*", "", line).strip()
            s = s.strip("\"“”「」'")
            if not s or s.startswith(("{", "}", "[", "]", "```")):
                continue
            out.append(_tidy(s))

    # 去重但保序
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def _tidy(text: str) -> str:
    s = text.strip().strip("\"“”「」'")
    s = re.sub(r"^\s*(?:她|assistant|AI)\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    return s.strip()

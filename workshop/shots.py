"""截图入库：`raw/assets/` 里的图 → `raw/` 里的对话稿。

三段式，跟 `ingest.py` 一个风格：
    1. `scan()`     只读，算 sha，比对水位，说清楚哪张还没转
    2. `compose()`  纯函数，把一次转写组装成成品稿文本，不落盘
    3. `commit()`   只写 `raw/*.md` 与 `data/shots_state.json`，不碰 wiki

**写出来的稿一律 `审：未过` + `情境：—`**，两处都是故意的：

- `审：未过` 让 `ingest.judge()` 判 `blocked`，整份挡在归纳门外。截图转写是全链
  唯一会"看错字"的环节，而 wiki 里没有"这句是猜的"这种标记，混进去就查不出来。
- `情境：—` 让 `level_of()` 返回 `—`，不归关系页。档位按 `agents.md` 只能人工判，
  模型给的档位宁可不写，也不能写错一节。

人核稿的动作就是这两处：把 `—` 改成档位、把 `未过` 改成 `已过`。一次编辑，三件事
（判档、核原话句数、扫一眼专名）一起完成。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import vision
from .ingest import ASSETS_DIR, RAW_DIR
from .ingest_rules import ingested_name

STATE_REL = Path("data") / "shots_state.json"
"""图片水位。键是图片 sha，不是文件名——QQ 截图叫 `8CB84CE1...(20261030-094627).png`
这种哈希名，人一改名按文件名就对不上了，而 sha 不受改名影响。"""

STATUS_NEW = "new"
STATUS_DONE = "done"
STATUS_LOST = "lost"
"""水位里有记录、稿却不在了。判 lost 而不是 new，是为了让原因说得清——
"转过但稿被删了"和"从没转过"是两种事，处理动作也不一样（前者是找回或重写）。"""

STATUS_LABEL = {
    STATUS_NEW: "待转写",
    STATUS_DONE: "已转写",
    STATUS_LOST: "稿丢了",
}

SHOT_SUFFIXES = tuple(vision.MIME)

USE_HEAD = "exemplars.md / reactions.md / speech.md / relationships.md"
"""对话稿的作业方向。照 `PRIORITY[dialog]` 的顺序写，但只是提示——
`agents.md` 第六节明写「用途不是结论，内容撑不起那一页就别硬编」。"""

NAME_MAX = 16
_BAD = re.compile(r'[\\/:*?"<>|\r\n\t]')


class ShotError(RuntimeError):
    pass


@dataclass(slots=True)
class Shot:
    name: str
    """磁盘文件名。"""
    path: Path
    sha: str
    size: int
    status: str
    md: str = ""
    """已生成的稿文件名（`done` 时有值）。"""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "status_label": STATUS_LABEL.get(self.status, self.status),
            "md": self.md,
            "reason": self.reason,
            "size": self.size,
            "sha": self.sha,
        }


@dataclass(slots=True)
class Scan:
    root: str
    items: list[Shot] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def pending(self) -> list[Shot]:
        return [s for s in self.items if s.status != STATUS_DONE]

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.items:
            out[s.status] = out.get(s.status, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "items": [s.to_dict() for s in self.items],
            "counts": self.counts,
            "pending": len(self.pending()),
            "warnings": self.warnings,
        }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_state(root: Path | str) -> dict:
    path = Path(root) / STATE_REL
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def write_state(root: Path | str, state: dict) -> None:
    path = Path(root) / STATE_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))


def _md_exists(raw: Path, md: str) -> bool:
    """稿还在不在。**要认带 `.已归纳` 的名字**：归纳收完之后素材会改名，
    水位里记的却还是转写当时的原名 —— 只按原名找会把收过稿的图全判成「稿丢了」，
    接着重转一份同名的，跟已归纳的那份并排躺在 raw/ 里。
    """
    if not md:
        return False
    return (raw / md).is_file() or (raw / ingested_name(md)).is_file()


def _occupied(raw: Path, name: str) -> bool:
    """这个稿名被占了没有 —— 原名和 `.已归纳` 两种形态都算。"""
    return (raw / name).exists() or (raw / ingested_name(name)).exists()


def scan(root: Path | str) -> Scan:
    """扫 `raw/assets/`。**零副作用**：不建目录、不写水位、不改名。"""
    root = Path(root)
    out = Scan(root=str(root))
    assets = root / RAW_DIR / ASSETS_DIR
    if not assets.is_dir():
        out.warnings.append(f"没有截图目录：{assets}")
        return out
    if not (root / RAW_DIR / "identity.md").is_file():
        # 称呼是从 identity.md 的「关系三档」读的，读不到会静默退成「对方」。
        # 稿子看着没问题，归完才发现说话人对不上，所以这里先喊一声。
        out.warnings.append("没有 raw/identity.md，正文里的称呼会退成「对方」，先把母本放回去再转")

    state = read_state(root)
    raw = root / RAW_DIR

    for path in sorted(assets.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in SHOT_SUFFIXES:
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            out.warnings.append(f"读不了 {path.name}：{exc}")
            continue

        sha = _sha(data)
        rec = state.get(sha) or {}
        shot = Shot(name=path.name, path=path, sha=sha, size=len(data), status=STATUS_NEW)

        # 同名换过内容（sha 变了）在这里认不出来：scan 是零副作用的，没法把旧名字
        # 记回内存再比对。判 new 的代价只是多出一份稿（落盘时自动加 -2），
        # 漏转的代价是这张图永远进不了库 —— 宁可重转。
        if not rec:
            out.items.append(shot)
            continue

        md = str(rec.get("md") or "")
        if rec.get("skipped"):
            shot.status = STATUS_DONE
            shot.reason = "上次判定不是聊天截图，跳过"
        elif md and not _md_exists(raw, md):
            shot.status = STATUS_LOST
            shot.md = md
            shot.reason = "水位里有记录，但那份稿已经不在了（被删或被改名）"
        elif md and rec.get("name") != path.name:
            shot.status = STATUS_DONE
            shot.md = md
            shot.reason = f"同一张图改过名，原来叫「{rec.get('name')}」"
        else:
            shot.status = STATUS_DONE
            shot.md = md
        out.items.append(shot)

    if not out.items:
        out.warnings.append(f"raw/{ASSETS_DIR}/ 里没有图片（支持 {'/'.join(SHOT_SUFFIXES)}）")
    return out


def _labels(root: Path | str) -> tuple[str, str]:
    """(拥有者, 她的名字)。两个都从库里已有的事实读，不写死在代码里。

    - 她的名字：库目录名第一段（`示例角色.辛.厄尔难.特羽克洛斯` → `示例角色`）
    - 拥有者：`raw/identity.md` 里「关系三档」的 `- 拥有者：老周，…`
    """
    root = Path(root)
    self_name = root.name.split(".")[0].strip() or "本人"

    owner = "对方"
    identity = root / RAW_DIR / "identity.md"
    if identity.is_file():
        text = identity.read_text(encoding="utf-8", errors="replace")
        hit = re.search(r"拥有者[：:]\s*([^\s，,。；;（(、]+)", text)
        if hit:
            owner = hit.group(1).strip() or owner
    return owner, self_name


def _title_of(turns: list[vision.Turn]) -> str:
    """标题取她说的第一句，截 NAME_MAX 字。

    取她的话而不是对方的话：这份素材是"她怎么反应"，标题该由她的动作定。
    她在整屏里一句没说（单方面陈述、转发的记录）才退到对方第一句。
    全读不出就给「待补」，别让文件名假装有内容。
    """
    def pick(items: list[vision.Turn]) -> str:
        for t in items:
            body = _BAD.sub("", t.text).strip(" 。，、！？…~～.,!?")
            if len(body) >= 2:
                return body[:NAME_MAX]
        return ""

    got = pick([t for t in turns if t.side == vision.SIDE_LEFT])
    if not got:
        signed = [t for t in turns if t.side != vision.SIDE_LEFT]
        got = pick(signed)
    return got or "待补"


def _clock_of(turns: list[vision.Turn]) -> str:
    for t in turns:
        if t.time:
            return t.time
    for t in turns:
        if t.side == vision.SIDE_CENTER and re.fullmatch(r"\d{1,2}:\d{2}", t.text.strip()):
            return t.text.strip()
    return ""


def compose(
    tr: vision.Transcript,
    *,
    date: str,
    image_rel: str,
    root: Path | str,
    at: str = "",
) -> tuple[str, str]:
    """转写 → (标题, 成品稿全文)。纯函数，不碰磁盘（除读 identity.md 取称呼）。"""
    owner, self_name = _labels(root)
    turns = list(tr.turns)

    her = sum(1 for t in turns if t.side == vision.SIDE_LEFT)
    clock = _clock_of(turns)
    scene = tr.scene or "待补场景"
    if clock:
        scene = f"{clock}，{scene}"

    title = _title_of(turns)
    head = [
        f"> 用途：{USE_HEAD}",
        f"> 分类：{tr.category or '日常搭话'}",
        f"> 情境：—｜{scene}",
        "> 张力：—",
        "> 对方：—",
        f"> 构成：人工原话 {her} 句 / AI 采纳 0 句 / 未采纳 0 条",
        f"> 生成：截图转写 {at or time.strftime('%Y-%m-%d %H:%M')}",
        f"> 证据：{image_rel}",
        "> 审：未过",
    ]

    body: list[str] = []
    for t in turns:
        if t.side == vision.SIDE_CENTER:
            continue
        who = owner if t.side == vision.SIDE_RIGHT else self_name
        text = t.text.replace("**", "＊＊")
        body.append(f"**{who}**：{text}")

    parts = [f"# {date} {title}", "", "\n".join(head), ""]
    if body:
        parts.append("\n\n".join(body))
        parts.append("")
    if tr.unknown:
        why = f"：{tr.note}" if tr.note else ""
        parts.append(f"【待补】有 {tr.unknown} 条读不出{why}。补齐后再把「审：未过」改成「已过」")
        parts.append("")
    if not tr.category:
        parts.append("【待定】分类没认出来，按内容从八类里挑一个再收")
        parts.append("")

    return title, "\n".join(parts).rstrip() + "\n"


def _free_name(raw: Path, base: str) -> str:
    """避开同名。同一天截了两张、首句又一样时，加 -2 而不是覆盖旧稿。"""
    name = f"{base}.md"
    if not _occupied(raw, name):
        return name
    for i in range(2, 100):
        name = f"{base}-{i}.md"
        if not _occupied(raw, name):
            return name
    raise ShotError(f"base 名撞太多次：{base}")


def commit(
    root: Path | str,
    done: list[tuple[Shot, vision.Transcript]],
    *,
    date: str = "",
    at: str = "",
    dry: bool = False,
) -> list[dict]:
    """落稿 + 推水位。返回每张图的处理结果。

    `dry=True` 只算不写，返回里带 `text` 供人预览。
    """
    root = Path(root)
    raw = root / RAW_DIR
    date = date or time.strftime("%Y-%m-%d")
    at = at or time.strftime("%Y-%m-%d %H:%M")
    state = read_state(root)
    results: list[dict] = []
    if not dry:
        raw.mkdir(parents=True, exist_ok=True)

    for shot, tr in done:
        title, text = compose(
            tr,
            date=date,
            image_rel=f"{RAW_DIR}/{ASSETS_DIR}/{shot.name}",
            root=root,
            at=at,
        )
        base = f"{date}_{title}"
        name = _free_name(raw, base)

        row = {
            "shot": shot.name,
            "title": title,
            "md": name,
            "kind": tr.kind,
            "turns": tr.readable,
            "unknown": tr.unknown,
            "text": text,
            "dry": dry,
        }
        results.append(row)

        # 不是聊天截图也要记水位，否则下次 scan 又把它当新的转一遍，白烧一次调用
        if tr.kind == vision.KIND_NOT_CHAT:
            row["skipped"] = True
            if not dry:
                state[shot.sha] = {"md": "", "name": shot.name, "at": at, "skipped": True}
            continue

        if dry:
            continue

        _atomic_write(raw / name, text)
        state[shot.sha] = {
            "md": name,
            "name": shot.name,
            "at": at,
            "turns": tr.readable,
            "unknown": tr.unknown,
        }

    if not dry:
        write_state(root, state)
    return results

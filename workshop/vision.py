"""截图转写：图片 → 结构化对话。

只做一件事：把一张聊天截图读成「哪一侧说了什么」，**不判档、不写稿、不落盘**。
判档与组装成品头在 `shots.py`——只有那里知道这个库的称呼和素材规范。

为什么单独一层：
- 换视觉模型、调 prompt、改 JSON schema 是常态，而"这张图到底读得准不准"
  要能单独验（`test_shots.py` 注入 stub，不花真钱）。
- 这里不 import `ingest`，避免转写层被归纳规则绑死。
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ingest_rules import CATEGORIES
from .llm import LLMClient, LLMError


class VisionError(RuntimeError):
    pass


SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDE_CENTER = "center"
SIDES = (SIDE_LEFT, SIDE_RIGHT, SIDE_CENTER)

KIND_CHAT = "chat"
"""是聊天截图，读出了内容。"""
KIND_NOT_CHAT = "not_chat"
"""不是聊天界面：风景、表情包、文档、网页。跳过，不产稿。"""
KIND_UNREADABLE = "unreadable"
"""是聊天但糊到读不出。产一份「待补」稿，别让人以为程序没看见这张图。"""
KINDS = (KIND_CHAT, KIND_NOT_CHAT, KIND_UNREADABLE)

MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

MAX_BYTES = 8 * 1024 * 1024
"""单张上限。base64 后膨胀 1/3，8MB 的图要发 11MB 请求体，多数网关会掐。"""

SYSTEM = """你是聊天截图转写器。只做转写，不做理解、不做总结、不做推断。

铁律：
1. 只输出一个 JSON 对象。不要 markdown 代码块，不要任何解释文字。
2. 逐条照抄气泡原文。错别字、标点、语气词全部保留，不修正、不补全、不改写。
3. 看不清的字写 null，不要猜。猜出来的字比空着危害大得多。
4. 不认人。只判气泡在左边还是右边，不写任何名字、昵称、群名、备注。
5. 不描述头像，不描述背景。

JSON 结构：
{"kind": "chat" | "not_chat" | "unreadable",
 "category": "" | "日常搭话" | "情绪反应" | "冲突对峙" | "价值判断"
             | "回忆往事" | "关系试探" | "边界触碰" | "自我介绍",
 "scene": "一句话说清这是什么场景，不带人名",
 "turns": [{"side": "left" | "right" | "center", "text": "...", "time": "23:38" | ""}],
 "unknown": 0,
 "note": ""}

- kind=not_chat：不是聊天界面，turns 给 []。
- kind=unreadable：是聊天但大部分读不出，能读的照写，unknown 记读不出的条数。
- category：只从上面八个里选一个，拿不准就给空串，不要自创。
- scene：说事件不说评价（"深夜催对方放下手机"，不是"她很关心人"）。不带人名、不带时间。
- side=center：居中系统提示（时间戳、撤回、加好友），照原文记进 text。
- time：只在气泡旁真有时间戳时填，否则空串。时间戳本身也单独记一条 center。
- note：一句话说明异常（"顶部两条被裁掉"），正常留空。
"""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJ = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class Turn:
    side: str
    text: str
    time: str = ""


@dataclass(slots=True)
class Transcript:
    kind: str = KIND_CHAT
    category: str = ""
    """八类之一，模型给的建议。落在枚举外就当没给——它只决定作业方向，不是结论。"""
    scene: str = ""
    """一句话场景，不带人名与时间。`情境` 的后半句。"""
    turns: list[Turn] = field(default_factory=list)
    unknown: int = 0
    """读不出的条数。大于 0 时 `shots` 会把稿标成待补，不放行。"""
    note: str = ""
    raw: str = ""
    """模型原始输出。解析失败或读数可疑时靠它复盘。"""

    @property
    def readable(self) -> int:
        return len(self.turns)


def data_uri(path: Path | str) -> str:
    """图片 → data URI。格式与大小在这里一次拦掉，别等网关报错。"""
    path = Path(path)
    mime = MIME.get(path.suffix.lower())
    if not mime:
        raise VisionError(f"不认的图片格式：{path.suffix or '(无扩展名)'}（支持 {'/'.join(sorted(MIME))}）")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VisionError(f"读不了 {path.name}：{exc}") from exc
    if not data:
        raise VisionError(f"{path.name} 是空文件")
    if len(data) > MAX_BYTES:
        raise VisionError(f"{path.name} 有 {len(data) / 1048576:.1f}MB，上限 {MAX_BYTES // 1048576}MB")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _json_of(text: str) -> dict:
    """从模型输出里抠 JSON。容忍 ```json 包裹和前后废话，不容忍读不出。"""
    body = (text or "").strip()
    if not body:
        raise VisionError("模型返回空内容")
    hit = _FENCE.search(body)
    if hit:
        body = hit.group(1).strip()
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    hit = _OBJ.search(body)
    if hit:
        try:
            obj = json.loads(hit.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise VisionError(f"模型没吐合法 JSON：{body[:200]}")


def parse(text: str) -> Transcript:
    """JSON → Transcript。字段烂一条丢一条，不整份崩。

    丢掉的 null 条数与模型自报的 unknown 取大值：宁可多记一条待补，
    不能让读不出的内容无声无息地消失。
    """
    obj = _json_of(text)

    kind = str(obj.get("kind") or KIND_CHAT).strip().lower()
    if kind not in KINDS:
        kind = KIND_CHAT

    turns: list[Turn] = []
    dropped = 0
    for item in obj.get("turns") or []:
        if not isinstance(item, dict):
            dropped += 1
            continue
        body = item.get("text")
        if body is None:
            dropped += 1
            continue
        body = str(body).strip()
        if not body:
            continue
        side = str(item.get("side") or "").strip().lower()
        if side not in SIDES:
            side = SIDE_LEFT
        turns.append(Turn(side=side, text=body, time=str(item.get("time") or "").strip()))

    try:
        said = int(obj.get("unknown") or 0)
    except (TypeError, ValueError):
        said = 0

    category = str(obj.get("category") or "").strip()
    if category not in CATEGORIES:
        category = ""

    return Transcript(
        kind=kind,
        category=category,
        scene=str(obj.get("scene") or "").strip(),
        turns=turns,
        unknown=max(said, dropped, 0),
        note=str(obj.get("note") or "").strip(),
        raw=text,
    )


async def describe(
    path: Path | str,
    llm: LLMClient,
    *,
    model: str = "",
    max_tokens: int = 0,
) -> Transcript:
    """读一张图。失败抛 `VisionError`（配置/格式问题）或 `LLMError`（调用问题）。

    模型优先级：显式 `model` > `vision_model` > `model`。留空回退到聊天模型，
    是因为不少接入点同一个模型就能收图（deepseek-flash 实测可以），
    没必要逼人多配一行。真遇上纯文本模型，HTTP 400 会在这里翻译成人话。
    """
    use = (model or llm.cfg.vision_model or llm.cfg.model or "").strip()
    if not use:
        raise VisionError("没有可用模型：config.toml 的 [llm] 里 model 和 vision_model 都是空的")

    path = Path(path)
    uri = data_uri(path)
    try:
        text = await llm.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"转写这张截图：{path.name}"},
            ],
            model=use,
            max_tokens=max_tokens or llm.cfg.vision_max_tokens,
            temperature=0.0,
            images=[uri],
        )
    except LLMError as exc:
        if "HTTP 400" in str(exc):
            raise VisionError(
                f"{use} 拒收了带图请求。纯文本模型收不了图，"
                f"把 [llm] 的 vision_model 指到一个多模态模型再试。"
                f"原始报错：{exc}"
            ) from exc
        raise
    return parse(text)

"""块级裁判：把一整块台词交给模型，让它说每句是谁说的。

规则层能精确绑定的只有 4%（53 / 1507 句）。原因是 155 个真名线索段里 72 个两侧都是台词：

    p195  我用斩钉截铁的语气给出这个答案。   ← 描述的是 p194
    p196  「我其实有好几次远远地观察你呢。」
    p197  「……有这种事？」

旁白到底在说前一句还是后一句，规则判不出来；但人读一眼就知道，因为读得懂一问一答。
这正是该交给模型的活。

三条硬约束（都是为防止模型把事搞坏）：

1. **模型不输出原文**。它只回 `{"i": 3, "who": "尤莉安娜·玛赞塔"}`，台词由调用方按索引取。
   从根上杜绝"模型顺手把台词改了"。
2. **候选集封闭**。只给本块在场者 + 上一块的说话人，最多 6 个。模型答了集外的名字一律丢弃。
3. **失败静默回退**。超时、JSON 坏、答案不在候选里 → 保留规则结果，不上报错误、不中断提取。

调用量：一个场景块一次，这本书 330 块。不是逐句问，也不是整本一次问（上下文太长会飘）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

from . import text as T

SYSTEM = """你是小说对话归属标注器。给你一个场景片段，判断每句台词是谁说的。

片段格式：旁白用【】括起，台词按出现顺序编号，形如 [0] 「……」。
在场人物会列在开头。

判断要点：
- 台词里出现的人名通常是被称呼或被谈论的对象，不是说话人。
- 一问一答要看语义：回答的人通常是被问的人；接着上一句反驳的人通常是对手。
- 「我」指谁，看同段旁白的视角：谁的眼睛在看，谁就是「我」。
- 连续多句台词且中间没有旁白时，多半是一人说几句，或两人交替。

只输出 JSON，不要解释、不要代码块：
{"lines": [{"i": 0, "who": "名字"}, {"i": 1, "who": ""}]}

每句都要给。中文小说里绝大多数台词都能从上下文判断出是谁说的——
旁白紧挨着谁、谁在被称呼、谁刚被问到——请按这些线索尽力判断。
实在判断不出（比如整段只有连续引语、完全没有旁白）才留空字符串，
不要为了保守而大面积留空，那样等于没标。"""

USER = """在场人物：{candidates}

场景：
{block}

共 {n} 句台词。请为每句给出说话人，只能从在场人物里选，或留空。"""

MAX_CANDIDATES = 6

PROMPT_SIG = hashlib.sha1((SYSTEM + "\x00" + USER).encode("utf-8")).hexdigest()[:8]
"""提示词的指纹。缓存键里带上它，改提示词就不会命中旧判定。"""


# ---------------------------------------------------------------- 组块

def render(para_lines: list[tuple[str, list[str]]]) -> tuple[str, int]:
    """把一块渲染成给模型看的场景，返回 (文本, 台词句数)。

    para_lines 是 [(该段原文, 该段里已收下的台词文本)]，按原文顺序。
    编号只编台词，旁白用【】括起，段落顺序原样保留——"旁白紧挨在哪句台词旁边"
    正是要模型判断的信息。

    段内顺序做了简化：段里的旁白合成一条，放在该段台词之前。
    一段内多句台词谁说的，是 `explicit` / `same` 两档规则就能定的（写在同一段里，
    名字或"她"就在旁边），裁判本来也不覆盖这两档。跨段才是它要解决的。
    """
    out: list[str] = []
    idx = 0
    for ptext, quotes in para_lines:
        narr = T.QUOTE_RE.sub("", ptext).strip()
        if narr:
            out.append(f"【{narr}】")
        for q in quotes:
            out.append(f"[{idx}] 「{q.strip()}」")
            idx += 1
    return "\n".join(out), idx


def pick_candidates(
    paras: list[str], names: list[str], prev: str = "", must: tuple[str, ...] = ()
) -> list[str]:
    """本块在场人物。优先取旁白里提到的，不够再补引语里被称呼的。

    只看旁白会把"全篇都被叫殿下"的角色漏掉，只看全文又会把只在台词里被提及的人
    （可能根本不在场）塞进来，所以先旁白后全文，且总量封顶。

    `must` 是必须先占位的名字，第一位是主角本人。第一人称小说里叙述者几乎不说话
    名——"我把她拉走"——但每句独白都是他说的，候选集里没有他就等于让模型无解。
    """
    narr = " ".join(T.QUOTE_RE.sub("", p) for p in paras)
    whole = " ".join(paras)
    here = [n for n in names if n in narr]
    if len(here) < 3:
        here += [n for n in names if n in whole and n not in here]
    if prev and prev not in here:
        here.append(prev)

    head = [n for n in must if n][:2]
    rest = [n for n in here if n not in head]
    return (head + rest)[:MAX_CANDIDATES]


def parse(text: str, n: int, allow: set[str] | None = None) -> dict[int, str]:
    """解析模型输出。只保证索引合法；"这个名字算不算数"交给调用方。

    早先在这里就用候选集硬过滤，结果模型回答「艾妮丝菲雅·温·帕雷提亚」（全名）、
    候选集里只有「艾妮丝菲雅」（简称）时整块判定被丢光。名字对不对得上得靠
    resolve() 做前缀匹配，不是字符串相等的活。
    """
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return {}

    out: dict[int, str] = {}
    for item in data.get("lines") or []:
        if not isinstance(item, dict):
            continue
        i, who = item.get("i"), (item.get("who") or "").strip()
        if not isinstance(i, int) or not 0 <= i < n:
            continue
        if who and allow is not None and who not in allow:
            continue
        out[i] = who
    return out


def resolve(who: str, cast: list[str], preferred: list[str]) -> str:
    """把模型写出来的名字对回名单里的一个写法。

    模型可能写全名、简称、或者带上"小姐"。三种都收：
    完全相等 → 优先候选；包含关系 → 取最长的那个（全名含简称，取全名）。
    对不上的返回空串，当作"模型也认不出"，不让它污染结果。
    """
    who = (who or "").strip()
    if not who:
        return ""
    if who in cast:
        return who
    for pool in (preferred, cast):
        hit = [c for c in pool if c and (c in who or who in c)]
        if hit:
            return max(hit, key=len)
    return ""


# ---------------------------------------------------------------- 缓存

class Cache:
    """按「场景内容 + 候选集」做键。同一本书重提取时不会重复烧钱。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, block: str, candidates: list[str]) -> str:
        """键里带提示词指纹：改了提示词，旧判定必须作废，否则测的是上一版。"""
        raw = PROMPT_SIG + "\x00" + block + "\x00" + "|".join(candidates)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def get(self, block: str, candidates: list[str]) -> dict[int, str] | None:
        path = self.root / f"{self._key(block, candidates)}.json"
        if not path.is_file():
            self.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return {int(k): v for k, v in (data.get("lines") or {}).items()}

    def put(self, block: str, candidates: list[str], verdicts: dict[int, str]) -> None:
        path = self.root / f"{self._key(block, candidates)}.json"
        try:
            path.write_text(
                json.dumps({"lines": {str(k): v for k, v in verdicts.items()}},
                           ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass


# ---------------------------------------------------------------- 裁判

class AuthFailed(RuntimeError):
    """余额用完 / Key 失效。这类错误重试和换块都没用，必须整轮停掉。"""


_FATAL = ("http 401", "http 402", "insufficient balance", "invalid_api_key",
          "authentication", "unauthorized")


def _fatal(msg: str) -> bool:
    low = msg.lower()
    return any(k in low for k in _FATAL)


async def judge_block(
    llm, block: str, n: int, candidates: list[str], *, max_tokens: int = 4096
) -> dict[int, str]:
    """判一块。任何异常都吞掉，返回空 dict 让调用方回退规则结果。

    `max_tokens` 给得偏大是故意的：推理型模型（deepseek-flash 就是）会先把预算
    花在思考上，1200 的预算有一半的块直接被思考吃光、正文一个字没有。
    真被吃到时再给一次三倍预算——这类块本身就不多，重试比全量调大便宜。
    """
    user = USER.format(candidates="、".join(candidates) or "（未识别到人名）",
                       block=block, n=n)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    budgets = (max_tokens, min(max_tokens * 3, 8192))
    for k, budget in enumerate(budgets):
        try:
            text = await llm.chat(messages, temperature=0.1, max_tokens=budget)
        except Exception as exc:  # noqa: BLE001 - 网络/配额/超时都回退
            if _fatal(str(exc)):
                raise AuthFailed(str(exc)) from exc
            if "思考" in str(exc) and k + 1 < len(budgets):
                continue  # 预算被思考吃光，换大预算再试一次
            return {}
        got = parse(text, n)
        if got:
            return got
        if k + 1 < len(budgets) and getattr(llm, "last_finish", None) == "length":
            continue  # 输出被截成半截 JSON，同样值得加预算重来
        break
    return {}


def judge_blocks_sync(
    llm,
    jobs: list[dict],
    cache: Cache | None = None,
    *,
    concurrency: int = 6,
    max_tokens: int = 4096,
) -> tuple[list[dict], dict]:
    """同步跑一串块。jobs 每项：{"block": str, "n": int, "candidates": [str]}。

    返回 (与 jobs 等长的 verdict 列表, 统计)。串行跑这本书 330 块要十来分钟，
    并发 6 路压到两分钟出头；再高 DeepSeek 容易 429，得不偿失。
    """
    results: list[dict[int, str]] = [{} for _ in jobs]
    pending: list[int] = []
    stat = {"blocks": len(jobs), "cache_hits": 0, "called": 0, "failed": 0,
            "lines": 0, "aborted": ""}

    for i, job in enumerate(jobs):
        hit = cache.get(job["block"], job["candidates"]) if cache else None
        if hit is not None:
            results[i] = hit
            stat["cache_hits"] += 1
        else:
            pending.append(i)

    if not pending:
        stat["lines"] = sum(1 for r in results for _ in r)
        return results, stat

    sem = asyncio.Semaphore(max(1, concurrency))
    stop = asyncio.Event()

    async def _one(i: int) -> None:
        if stop.is_set():
            return
        async with sem:
            if stop.is_set():
                return
            job = jobs[i]
            try:
                got = await judge_block(
                    llm, job["block"], job["n"], job["candidates"], max_tokens=max_tokens
                )
            except AuthFailed as exc:
                # 余额/鉴权问题：后面几十块再问一遍只会把日志刷满，直接停
                stat["aborted"] = str(exc)[:120]
                stop.set()
                return

        results[i] = got
        stat["called"] += 1
        if not got:
            stat["failed"] += 1
        elif cache:
            cache.put(job["block"], job["candidates"], got)

    async def _run() -> None:
        await asyncio.gather(*(_one(i) for i in pending))

    try:
        asyncio.run(_run())
    except RuntimeError:
        # 已经在事件循环里（WebUI 的异步入口）：起一个独立线程跑
        import threading

        def _worker() -> None:
            try:
                asyncio.run(_run())
            except Exception:  # noqa: BLE001 - 裁判整体失败不上抛
                pass

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

    stat["lines"] = sum(1 for r in results for _ in r)
    return results, stat


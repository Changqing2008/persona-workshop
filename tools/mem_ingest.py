"""记忆归档器：把聊天里值得记的事实沉淀进 persona/memory/。

    python tools/mem_ingest.py --persona <dir> [--session private:123] [--dry-run]
                               [--limit 60] [--allow-new] [--rebuild-hub]

只读 sessions.db（mode=ro）；写入只发生在 persona/memory/ 下。
水位线存 memory/archive_state.json，最后一步才推进：中途失败可重放，不会重复写。

三道校验挡住模型编造，任何一条不过就不落盘：
1. page 必须落在 people / projects / domains 之下
2. 每条 fact 必须带 quote，且 quote 要能在对话原文里找到
3. fact 不得含性格断言（她总是 / 一向 / 从来 / 本质 / 其实很 / 这人 / 天生 / 根本）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workshop import library  # noqa: E402
from workshop.config import load  # noqa: E402
from workshop.llm import LLMClient, LLMError  # noqa: E402
from workshop.memwiki import JUDGEMENT, MemWiki  # noqa: E402

ALLOWED_DIRS = ("people", "projects", "domains")
STATE_NAME = "archive_state.json"
RECENT_DEFAULT = 60
MAX_FAILS = 3
"""同一会话连续抽取失败几次后放弃并推进水位。

有了它才敢在不推进水位的前提下重试：否则一个永远失败的会话
会每一轮都白烧一次调用；而没有它，失败即推进等于把那批聊天永久丢掉。
"""

EXTRACT_SYSTEM = """你是记忆归档器。从一段 QQ 对话里抽取"三个月后还用得上"的事实，输出严格 JSON。

只抽这三类：
1. 对方稳定的事实：身份、职业、作息、明确说过的偏好与忌讳
2. 关系本身的变化：称呼怎么变的、亲密度、明确表态（"别叫我宝贝"这类）
3. 事件的结论或约定：某件事定了什么、答应了什么（不要过程）

一律不抽：
- 某天聊了什么话题、说了哪句话（哪怕当时很有意思）
- 一次性的抱怨、情绪、玩笑、撒娇
- 她自己的说话方式、行为规则、定的标准（那属于人格层，不是记忆层）
- 过程细节（聊了多久、几点睡的、躺了几分钟）

== 最重要的一条 ==
对话里标着「她」的是本角色自己。关于她自己的任何内容——年龄、姓名、家人、
底线、喜好、她说过"我不接受什么"——一个字都不要抽，更不要给她建 people 页。
她是谁写在 wiki/ 里，不归记忆层。只记对话里的其他人（标着「对方」的，或群聊里
其他发言人）。拿不准某句话是她说的还是别人说的，就不记。

判断标准：这个事实三个月后还有用吗？没用就不写。
对方只要透露了稳定信息（职业、作息、家人、健康、住址、生日、明确喜好与忌讳、长期计划），就必须抽出来；
纯闲聊可以返回空。

每条事实必须附 quote：从对话里原样摘的一句话，用来核对。找不到依据就不要写。
pages 是已有页面清单，能归到已有页就填它的路径；确实是新的对象才写 NEW:<目录>/<名字>。
只值得记进时间线、不配单独成页的，page 填空字符串。

输出格式（只输出 JSON，不要解释）：
{"events":[{"page":"people/某人或NEW:people/某人或空串","title":"一句话",
"date":"YYYY-MM-DD","facts":["一条事实"],"quote":"原文片段","confidence":"high|medium|low"}]}

没有值得记的内容就输出 {"events": []}。"""


@dataclass
class Event:
    page: str
    title: str
    date: str
    facts: list[str] = field(default_factory=list)
    quote: str = ""
    confidence: str = "medium"


@dataclass
class Reject:
    reason: str
    event: Event


# ---------- 读库（只读） ----------


def open_ro(db: Path) -> sqlite3.Connection:
    uri = f"file:{db.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def list_sessions(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT session_id FROM sessions ORDER BY last_ts DESC LIMIT 20"
    ).fetchall()
    return [str(r["session_id"]) for r in rows]


def fetch(conn: sqlite3.Connection, session_id: str, after_id: int, limit: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, role, content, ts, user_id FROM messages "
        "WHERE session_id=? AND id>? ORDER BY id ASC LIMIT ?",
        (session_id, after_id, limit),
    ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "role": str(r["role"]),
            "content": str(r["content"]),
            "user_id": str(r["user_id"] or ""),
        }
        for r in rows
    ]


# ---------- 水位线 ----------


def read_state(mem_dir: Path) -> dict:
    path = mem_dir / STATE_NAME
    if not path.is_file():
        return {"sessions": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"sessions": {}}
    if not isinstance(data, dict):
        return {"sessions": {}}
    data.setdefault("sessions", {})
    return data


def write_state(mem_dir: Path, state: dict) -> None:
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = mem_dir / STATE_NAME
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ---------- 抽取 ----------


def render_transcript(rows: list[dict]) -> str:
    out = []
    for r in rows:
        who = "她" if r["role"] == "assistant" else "对方"
        out.append(f"[{r['id']}] {who}：{r['content']}")
    return "\n".join(out)


def parse_events(raw: str) -> list[Event]:
    """模型偶尔会包一层 ```json，也不保证只输出 JSON。取第一个完整对象。"""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    events: list[Event] = []
    for item in data.get("events") or []:
        if not isinstance(item, dict):
            continue
        facts = [str(x).strip() for x in (item.get("facts") or []) if str(x).strip()]
        if not facts:
            continue
        events.append(
            Event(
                page=str(item.get("page") or "").strip(),
                title=str(item.get("title") or "").strip(),
                date=str(item.get("date") or "").strip() or time.strftime("%Y-%m-%d"),
                facts=facts[:6],
                quote=str(item.get("quote") or "").strip(),
                confidence=str(item.get("confidence") or "medium").strip(),
            )
        )
    return events


def validate(
    ev: Event, transcript: str, allow_new: bool, self_names: set[str] | None = None
) -> str | None:
    """返回拒收原因；None 表示通过。"""
    # NEW: 是「待新建」前缀，先剥掉再判目录，否则 NEW:people/x 会被当成非法路径
    rel = ev.page[4:] if ev.page.startswith("NEW:") else ev.page
    if rel and not rel.startswith(ALLOWED_DIRS):
        return f"目标目录不在白名单：{rel}"
    # 角色本人不归记忆层管。群聊里她自己说的话（年龄、姓氏、底线）
    # 模型分不清"她"和"对方"，实测会给角色自己建一张 people 页。
    # 那页一旦被检索命中，注入内容就成了「## 相关记忆 - 角色名 14岁」，
    # 把人格事实混进回忆里 —— 既违反隔离，也会让她把自己的底细当旧事讲出来。
    if rel and self_names:
        top, _, rest = rel.partition("/")
        stem = Path(rest).stem if rest else ""
        if top == "people" and stem and stem in self_names:
            return f"不能给角色本人建 people 页：{rel}（她是谁属于 wiki/，不归记忆层）"
    if ev.page.startswith("NEW:") and not allow_new:
        return "新页面未获批准（加 --allow-new 才建）"
    if not ev.quote or len(ev.quote) < 4:
        return "缺 quote，无法核对出处"
    # 引文核对：中文对话里模型常抄错一两个字，取前 10 字做包含检查
    probe = ev.quote[:10]
    if probe not in transcript:
        return f"quote 在对话里找不到：{ev.quote[:20]}…"
    for fact in ev.facts:
        if JUDGEMENT.search(fact):
            return f"含性格断言：{fact[:24]}"
    return None


async def extract(llm: LLMClient, transcript: str, pages: list[str]) -> list[Event]:
    user = "已有页面：\n" + ("\n".join(pages) if pages else "（无）")
    user += "\n\n对话原文：\n" + transcript
    raw = await llm.complete(EXTRACT_SYSTEM, user, max_tokens=900)
    return parse_events(raw)


# ---------- 写入 ----------


def _touch_updated(text: str, date: str) -> str:
    if re.search(r"^updated:.*$", text, flags=re.MULTILINE):
        return re.sub(
            r"^updated:.*$", f"updated: {date}", text, count=1, flags=re.MULTILINE
        )
    return text


def _existing_facts(text: str) -> set[str]:
    out: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- "):
            out.add(re.sub(r"[（(]\s*source\s*:[^）)]*[）)]", "", s[2:]).strip())
    return out


def insert_facts(text: str, lines: list[str]) -> str | None:
    """把 facts 追加到「## 事实」段末尾。没有该段返回 None。"""
    rows = text.splitlines()
    start = next(
        (i for i, ln in enumerate(rows) if ln.strip().startswith("## 事实")), None
    )
    if start is None:
        return None
    end = start + 1
    for i in range(start + 1, len(rows)):
        if rows[i].startswith("## "):
            break
        if rows[i].strip():
            end = i + 1
    return "\n".join(rows[:end] + lines + rows[end:]).rstrip("\n") + "\n"


def new_page_text(
    ev: Event, rel: str, name: str, source: str, today: str, user_id: str = ""
) -> str:
    aliases = json.dumps([name], ensure_ascii=False)
    facts = "\n".join(f"- {f}（source: {source}）" for f in ev.facts)
    # type 必须按剥离前缀后的相对路径判，用 ev.page 会把 NEW:people/x 判成 domain
    ptype = {"people": "person", "projects": "project", "domains": "domain"}.get(
        rel.split("/")[0], "note"
    )
    # user_id 决定"关系档位能不能认出他"：没有这行，归档了也当陌生人
    uids = json.dumps([user_id], ensure_ascii=False) if user_id else "[]"
    return f"""---
type: {ptype}
user_id: {uids}
tags: []
aliases: {aliases}
links: []
status: stub
confidence: {ev.confidence}
first_seen: {_norm_date(ev.date, today)}
updated: {today}
source: "{source}"
---

# {name}

## 事实

{facts}

## 待查

- 只从一次对话里抽出，未经互证。
"""


def append_timeline(mem_dir: Path, date: str, lines: list[str]) -> None:
    month = date[:7]
    path = mem_dir / "timeline" / f"{month}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(
            f"---\ntype: timeline\ntags: [时间线]\naliases: []\nlinks: []\n"
            f"status: growing\nconfidence: medium\nfirst_seen: {date}\n"
            f"updated: {date}\nsource: \"\"\n---\n\n# {month} 事件流水\n\n",
            encoding="utf-8",
        )
    text = path.read_text(encoding="utf-8")
    text = _touch_updated(text, date)
    add = [ln for ln in lines if ln not in text]
    if add:
        text = text.rstrip("\n") + "\n" + "\n".join(add) + "\n"
    path.write_text(text, encoding="utf-8")


def _norm_date(value: str, today: str) -> str:
    """把模型给的日期收拾成 YYYY-MM-DD。

    模型并不知道"今天"是几号，实测经常返回空串。原样用下去，
    只进时间线的条目会写到 `timeline/.md`——文件名是一场空的点 md，
    而这个文件既不会被 hub 索引也不会被检索到，等于事实凭空消失。
    """
    text = (value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    return today


def apply_events(
    mem_dir: Path,
    events: list[Event],
    source: str,
    today: str,
    transcript: str,
    allow_new: bool,
    dry_run: bool,
    user_id: str = "",
    self_names: set[str] | None = None,
) -> tuple[list[str], list[Reject]]:
    done: list[str] = []
    rejects: list[Reject] = []
    planned: set[str] = set()
    """本次已计划新建的页。dry-run 下文件并不存在，靠它让预演与实跑一致。"""
    for ev in events:
        reason = validate(ev, transcript, allow_new, self_names)
        if reason:
            rejects.append(Reject(reason, ev))
            continue

        # 只进时间线
        if not ev.page:
            date = _norm_date(ev.date, today)
            day = date[5:]  # MM-DD：条目行里不重复年份
            add = [f"- {day} · {f}" for f in ev.facts]
            if not dry_run:
                append_timeline(mem_dir, date, add)
            done.append(f"时间线 {date}：{len(add)} 条")
            continue

        creating = ev.page.startswith("NEW:")
        rel = ev.page[4:] if creating else ev.page
        if not rel.endswith(".md"):
            rel += ".md"
        path = mem_dir / rel

        if creating and not path.exists() and rel not in planned:
            name = Path(rel).stem
            if not dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    new_page_text(ev, rel, name, source, today, user_id), encoding="utf-8"
                )
            planned.add(rel)
            done.append(f"新建 {rel}：{len(ev.facts)} 条")
            continue

        # dry-run 时同一页的第二条起，按"并入本次新建页"预览，别重复报新建
        if dry_run and rel in planned:
            done.append(f"并入 {rel}（本次新建）：{len(ev.facts)} 条")
            continue

        if not path.is_file():
            rejects.append(Reject(f"目标页不存在：{rel}", ev))
            continue

        text = path.read_text(encoding="utf-8")
        seen = _existing_facts(text)
        fresh = [f for f in ev.facts if f not in seen]
        if not fresh:
            done.append(f"{rel}：0 条（已全部存在）")
            continue
        lines = [f"- {f}（source: {source}）" for f in fresh]
        merged = insert_facts(_touch_updated(text, today), lines)
        if merged is None:
            rejects.append(Reject(f"{rel} 没有「## 事实」段，拒绝合并", ev))
            continue
        if not dry_run:
            path.write_text(merged, encoding="utf-8")
        done.append(f"并入 {rel}：{len(fresh)} 条")
    return done, rejects


def write_rejects(mem_dir: Path, rejects: list[Reject], source: str) -> None:
    if not rejects:
        return
    path = mem_dir / "pending" / "rejected.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%d %H:%M")
    block = [f"\n## {now} · {source}\n"]
    for r in rejects:
        block.append(f"- 拒收：{r.reason}")
        for f in r.event.facts:
            block.append(f"  - 原事实：{f}")
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(block) + "\n")


def append_log(persona_dir: Path, done: list[str], source: str) -> None:
    """归档成功后在 log.md 追加一条 merge 记录（契约要求，别靠人记）。"""
    path = persona_dir / "log.md"
    if not path.is_file():
        return
    now = time.strftime("%Y-%m-%d %H:%M")
    block = f"\n## [{now}] merge | 记忆归档 {source}\n"
    block += "\n".join(f"- {line}" for line in done) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


STATS_NAME = "ingest_stats.jsonl"


def append_stats(mem_dir: Path, record: dict) -> None:
    """每次批量归档记一行。没有它，"抽多抽少"永远只能靠人肉看 dry-run。"""
    path = mem_dir / STATS_NAME
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_stats(mem_dir: Path, last: int = 8) -> int:
    path = mem_dir / STATS_NAME
    if not path.is_file():
        print(f"还没有归档记录：{path}")
        return 0
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        print("统计文件为空")
        return 0
    msg = sum(int(r.get("messages", 0)) for r in rows)
    ev = sum(int(r.get("events", 0)) for r in rows)
    wr = sum(int(r.get("written", 0)) for r in rows)
    rj = sum(int(r.get("rejected", 0)) for r in rows)
    hits = sum(1 for r in rows if int(r.get("events", 0)) > 0)
    print(f"归档 {len(rows)} 批 / 消息 {msg} 条 / 事件 {ev} 条 / 写入事实 {wr} 条 / 拒收 {rj} 条")
    print(f"有产出的批次 {hits} 批（{hits / len(rows) * 100:.0f}%）")
    print(f"\n最近 {min(last, len(rows))} 批：")
    for r in rows[-last:]:
        print(
            f"  {r.get('ts', '?')}  {r.get('session', '?')[:28]:<28} "
            f"消息 {r.get('messages', 0):>3}  事件 {r.get('events', 0):>2}  "
            f"写入 {r.get('written', 0):>2}  拒收 {r.get('rejected', 0):>2}"
        )
    return 0


# ---------- 主流程 ----------


def rebuild_hub(mem_dir: Path) -> str:
    """按实际页面重写四张 hub 索引。返回变更摘要。"""
    from tools.mem_doctor import rebuild_all

    return rebuild_all(mem_dir)


async def run(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    persona_dir = Path(args.persona or cfg.bot.persona_dir)
    mem_dir = persona_dir / "memory"
    if not mem_dir.is_dir():
        print(f"记忆层不存在：{mem_dir}")
        return 2

    db = library.resolve_db(persona_dir, cfg.path(cfg.memory.db_path))
    if not db.is_file():
        print(f"会话库不存在：{db}")
        return 2

    wiki = MemWiki(persona_dir)
    pages = [p.rel for p in wiki.pages]
    state = read_state(mem_dir)
    today = time.strftime("%Y-%m-%d")
    self_names = own_names(persona_dir)

    conn = open_ro(db)
    try:
        sessions = [args.session] if args.session else list_sessions(conn)
        total_events = 0
        for sid in sessions:
            after = int(state["sessions"].get(sid, 0))
            rows = fetch(conn, sid, after, args.limit)
            if not rows:
                print(f"[{sid}] 无新消息（水位 {after}）")
                continue
            transcript = render_transcript(rows)
            span = f"{sid}#{rows[0]['id']}-{rows[-1]['id']}"
            # 私聊的对象就是 sid 后半段；群聊有多个发言人，不自动认人
            uid = sid.split(":", 1)[1] if sid.startswith("private:") else ""
            print(f"[{sid}] 新消息 {len(rows)} 条 → 抽取中…")

            try:
                events = await extract(LLMClient(cfg.llm), transcript, pages)
            except (LLMError, OSError) as exc:
                # 一个会话抽不出来不该带塌整批：剩下的会话还没处理呢。
                # 水位不推进 = 下一轮重试；连续失败 MAX_FAILS 次才放弃并推进，
                # 否则一个坏会话会永远挂在那里，每轮白烧一次调用。
                fails = state.setdefault("fails", {})
                n = int(fails.get(sid, 0)) + 1
                fails[sid] = n
                give_up = n >= MAX_FAILS
                print(
                    f"[{sid}] 抽取失败（第 {n} 次）：{exc}"
                    + ("，放弃并推进水位" if give_up else "，保留水位下轮重试")
                )
                if not args.dry_run:
                    append_stats(
                        mem_dir,
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M"),
                            "session": sid,
                            "messages": len(rows),
                            "events": 0,
                            "written": 0,
                            "rejected": 0,
                            "error": str(exc)[:160],
                        },
                    )
                    if give_up:
                        state["sessions"][sid] = rows[-1]["id"]
                        fails.pop(sid, None)
                    write_state(mem_dir, state)
                continue
            state.setdefault("fails", {}).pop(sid, None)
            if not events:
                print(f"[{sid}] 没有值得记的事实")
                if not args.dry_run:
                    append_stats(
                        mem_dir,
                        {
                            "ts": time.strftime("%Y-%m-%d %H:%M"),
                            "session": sid,
                            "messages": len(rows),
                            "events": 0,
                            "written": 0,
                            "rejected": 0,
                        },
                    )
                    state["sessions"][sid] = rows[-1]["id"]
                    write_state(mem_dir, state)
                continue

            done, rejects = apply_events(
                mem_dir,
                events,
                span,
                today,
                transcript,
                args.allow_new,
                args.dry_run,
                uid,
                self_names,
            )
            if args.dry_run:
                # dry-run 的意义就是让人看见会写进什么，只报条数等于没验
                for ev in events:
                    print(f"    · [{ev.page or '仅时间线'}] {ev.date} {ev.title}")
                    for f in ev.facts:
                        print(f"        - {f}")
                    print(f"        quote: {ev.quote[:40]}")
            for line in done:
                print("   ", line)
            for r in rejects:
                print("    拒收：", r.reason)
            total_events += len(events)
            if not args.dry_run:
                write_rejects(mem_dir, rejects, span)
                if done:
                    append_log(persona_dir, done, span)
                append_stats(
                    mem_dir,
                    {
                        "ts": time.strftime("%Y-%m-%d %H:%M"),
                        "session": sid,
                        "messages": len(rows),
                        "events": len(events),
                        "written": len(done),
                        "rejected": len(rejects),
                    },
                )
                state["sessions"][sid] = rows[-1]["id"]
                write_state(mem_dir, state)
    finally:
        conn.close()

    if args.rebuild_hub and not args.dry_run:
        print("重建 hub：", rebuild_hub(mem_dir))

    print(f"\n完成：{total_events} 条事件" + ("（dry-run，未写盘）" if args.dry_run else ""))
    if not args.dry_run and total_events:
        print("已写入 log.md（前缀 merge）")
    return 0


def own_names(persona_dir: Path) -> set[str]:
    """角色自己的各种叫法：目录名、按点拆开的各段、库内昵称。

    用来挡住「给角色本人建 people 页」——群聊里她说的话，模型分不清
    是"她"还是"对方"，会一本正经地给她自己归档。
    """
    names = {persona_dir.name.strip()}
    names |= {p.strip() for p in persona_dir.name.split(".") if p.strip()}
    try:
        nick = str(library.read_prefs(persona_dir).get("bot", {}).get("nickname") or "")
    except Exception:  # noqa: BLE001
        nick = ""
    if nick.strip():
        names.add(nick.strip())
    return {n for n in names if n}


def make_args(
    persona: str = "",
    config: str = "config.toml",
    *,
    session: str = "",
    limit: int = RECENT_DEFAULT,
    dry_run: bool = False,
    allow_new: bool = False,
    rebuild_hub: bool = False,
) -> argparse.Namespace:
    """组装 run() 需要的参数。

    命令行和 run.py 的定时归档都从这里取，免得哪边加了参数另一边漏掉——
    历史上就因为两处各写一份，dry-run 和实跑的行为偷偷不一致过。
    """
    return argparse.Namespace(
        persona=persona,
        config=config,
        session=session,
        limit=limit,
        dry_run=dry_run,
        allow_new=allow_new,
        rebuild_hub=rebuild_hub,
        stats=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="把聊天事实归档进 memory/")
    ap.add_argument("--persona", default="", help="人格库目录，默认取 config.toml")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--session", default="", help="只处理某个会话，默认最近 20 个")
    ap.add_argument("--limit", type=int, default=RECENT_DEFAULT, help="单会话最多取多少条消息")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写盘、不推进水位")
    ap.add_argument("--allow-new", action="store_true", help="允许新建页面")
    ap.add_argument("--rebuild-hub", action="store_true", help="结束后重建 hub 索引")
    ap.add_argument("--stats", action="store_true", help="只看历史归档统计，不跑归档")
    args = ap.parse_args()
    if args.stats:
        cfg = load(args.config)
        persona_dir = Path(args.persona or cfg.bot.persona_dir)
        return print_stats(persona_dir / "memory")
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())

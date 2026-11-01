"""记忆层体检 + hub 重建。

    python tools/mem_doctor.py --persona <dir> [--fix]

--fix 只做安全动作：重建四张 hub 索引。其余只报告，不改内容页。
退出码：0 无问题 / 1 有 error。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workshop import library  # noqa: E402
from workshop.config import load  # noqa: E402
from workshop.memwiki import JUDGEMENT, MemWiki, _parse_front  # noqa: E402

MAX_PAGE_CHARS = 4000
WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
HUBS = ("people", "projects", "domains", "timeline")

HEADERS = {
    "people": "| 页 | 关系档 | 一句话 | 状态 |",
    "projects": "| 页 | 一句话 | 状态 |",
    "domains": "| 页 | 一句话 | 状态 |",
    "timeline": "| 月份 | 页 | 条目数 |",
}


# ---------- 扫描 ----------


def _facts_of(body: str) -> list[str]:
    out: list[str] = []
    inside = False
    for line in body.splitlines():
        if line.startswith("## "):
            inside = line.strip().startswith("## 事实")
            continue
        if inside and line.strip().startswith("- "):
            out.append(line.strip()[2:].strip())
    return out


def _first_fact(body: str, width: int = 40) -> str:
    facts = _facts_of(body)
    if not facts:
        return ""
    s = re.sub(r"[（(]\s*source\s*:[^）)]*[）)]", "", facts[0]).strip()
    return s if len(s) <= width else s[:width] + "…"


def scan(mem_dir: Path) -> list[dict]:
    pages: list[dict] = []
    if not mem_dir.is_dir():
        return pages
    for path in sorted(mem_dir.rglob("*.md")):
        rel = path.relative_to(mem_dir).as_posix()
        if rel.startswith("pending/") or rel.startswith("_hub/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        meta, body = _parse_front(text)
        links = meta.get("links")
        if isinstance(links, str):
            links = [links]
        pages.append(
            {
                "rel": rel,
                "path": path,
                "top": rel.split("/")[0],
                "name": path.stem,
                "title": str(meta.get("title") or "") or path.stem,
                "type": str(meta.get("type") or ""),
                "status": str(meta.get("status") or ""),
                "confidence": str(meta.get("confidence") or ""),
                "scope": str(meta.get("scope") or ""),
                "relation": str(meta.get("relation") or ""),
                "user_ids": (
                    [str(x) for x in meta["user_id"]]
                    if isinstance(meta.get("user_id"), list)
                    else ([str(meta["user_id"])] if meta.get("user_id") else [])
                ),
                "summary": str(meta.get("summary") or ""),
                "aliases": [str(x) for x in (meta.get("aliases") or [])]
                if isinstance(meta.get("aliases"), list)
                else [],
                "links": [str(x) for x in (links or [])],
                "body": body,
                "chars": len(body),
                "text": text,
            }
        )
    return pages


# ---------- hub 重建 ----------


def _row_for(hub: str, page: dict) -> str:
    one = page["summary"] or _first_fact(page["body"]) or "—"
    status = page["status"] or "—"
    if hub == "people":
        return f"| [[{page['rel'][:-3]}]] | {page['relation'] or '—'} | {one} | {status} |"
    if hub == "timeline":
        month = page["name"]
        count = len(re.findall(r"^- \d", page["body"], flags=re.MULTILINE))
        return f"| {month} | [[{page['rel'][:-3]}]] | {count} |"
    return f"| [[{page['rel'][:-3]}]] | {one} | {status} |"


def _replace_table(text: str, header: str, rows: list[str]) -> str:
    lines = text.splitlines()
    idx = [i for i, ln in enumerate(lines) if ln.strip().startswith("|")]
    cols = header.count("|") - 1
    sep = "|" + "|".join(["---"] * cols) + "|"
    block = [header, sep] + (rows or ["| _暂无_ |" + " — |" * (cols - 1)])
    if not idx:
        return text.rstrip("\n") + "\n\n" + "\n".join(block) + "\n"
    first, last = idx[0], idx[-1]
    return "\n".join(lines[:first] + block + lines[last + 1 :]) + "\n"


def rebuild_all(mem_dir: Path) -> str:
    pages = scan(mem_dir)
    out: list[str] = []
    for hub in HUBS:
        rows = [_row_for(hub, p) for p in pages if p["top"] == hub and p["scope"] != "system"]
        path = mem_dir / "_hub" / f"{hub}.md"
        header = HEADERS[hub]
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            path.write_text(_replace_table(text, header, rows), encoding="utf-8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            body = (
                f"---\ntype: hub\ntags: [索引]\naliases: []\nstatus: stable\n"
                f"confidence: high\nupdated: {__import__('time').strftime('%Y-%m-%d')}\n---\n\n"
                f"# {hub} — 分类索引\n\n> 由 mem_doctor.py 自动重建。\n\n"
                + _replace_table("", header, rows)
            )
            path.write_text(body, encoding="utf-8")
        out.append(f"{hub} {len(rows)} 行")
    return " / ".join(out)


# ---------- 体检 ----------


def check(persona_dir: Path) -> list[tuple[str, str]]:
    mem_dir = persona_dir / "memory"
    wiki_dir = persona_dir / "wiki"
    issues: list[tuple[str, str]] = []
    if not mem_dir.is_dir():
        return [("error", f"记忆层不存在：{mem_dir}")]

    pages = scan(mem_dir)
    if not pages:
        return [("warn", "记忆层为空（只有 hub）")]

    by_rel = {p["rel"]: p for p in pages}

    # 1. 必填字段 + 目录与 type 一致
    EXPECT_TYPE = {
        "people": "person",
        "projects": "project",
        "domains": "domain",
        "timeline": "timeline",
    }
    for p in pages:
        for key in ("type", "status", "confidence"):
            if not p[key]:
                issues.append(("error", f"{p['rel']} 缺 {key}"))
        want = EXPECT_TYPE.get(p["top"])
        if want and p["type"] and p["type"] != want:
            issues.append(("error", f"{p['rel']} type 应为 {want}，实际 {p['type']}"))

    # 2. 别名冲突
    owners: dict[str, list[str]] = {}
    for p in pages:
        for alias in p["aliases"] or [p["title"]]:
            if len(alias) >= 2:
                owners.setdefault(alias, []).append(p["rel"])
    for alias, rels in sorted(owners.items()):
        if len(rels) > 1:
            issues.append(("error", f"别名冲突「{alias}」：{', '.join(rels)}"))

    # 3. 断链
    # 短名（[[那个熟人]]）先在本层找，找不到再回退 wiki —— 时间线里写的都是短名，
    # 直接按 wiki 页判会把正常链接报成断链。
    names = {p["name"]: p["rel"] for p in pages}

    def resolves(t: str) -> bool:
        t = t.strip()
        if not t:
            return True
        if "/" in t:
            return (mem_dir / (t if t.endswith(".md") else t + ".md")).is_file()
        if t in names:
            return True
        return (wiki_dir / (t if t.endswith(".md") else t + ".md")).is_file()

    for p in pages:
        targets = list(p["links"]) + WIKILINK.findall(p["body"])
        for t in targets:
            if not resolves(t):
                issues.append(("error", f"{p['rel']} 断链 → {t.strip()}"))

    # 4. 孤页：没有任何 hub 或页链接到它
    linked: set[str] = set()
    hub_text = "\n".join(
        (mem_dir / "_hub" / f"{h}.md").read_text(encoding="utf-8")
        for h in HUBS
        if (mem_dir / "_hub" / f"{h}.md").is_file()
    )
    for t in WIKILINK.findall(hub_text):
        linked.add(t.strip())
        linked.add(Path(t.strip()).stem)
    for p in pages:
        for t in p["links"] + WIKILINK.findall(p["body"]):
            linked.add(t.strip())
            linked.add(Path(t.strip()).stem)
    for p in pages:
        if p["scope"] == "system":
            continue
        if p["rel"][:-3] not in linked and p["name"] not in linked:
            issues.append(("warn", f"{p['rel']} 孤页：没有任何索引或页面链接到它"))

    # 5. 超长页
    for p in pages:
        if p["chars"] > MAX_PAGE_CHARS:
            issues.append(("warn", f"{p['rel']} 超长 {p['chars']} 字（上限 {MAX_PAGE_CHARS}），该拆页"))

    # 6. 占位符残留
    for p in pages:
        if "_待填_" in p["text"]:
            issues.append(("warn", f"{p['rel']} 仍有 _待填_"))

    # 7. 性格断言
    for p in pages:
        for fact in _facts_of(p["body"]):
            if JUDGEMENT.search(fact):
                issues.append(("error", f"{p['rel']} 事实行含性格断言：{fact[:24]}"))

    # 8. system 页不该出现在 hub 里
    for p in pages:
        if p["scope"] == "system" and p["rel"][:-3] in linked:
            issues.append(("warn", f"{p['rel']} 是 scope: system，却被索引引用"))

    # 9. people 页要能被认出来：缺 user_id 则关系档位永远判成陌生人
    for p in pages:
        if p["top"] != "people":
            continue
        if not p["user_ids"]:
            issues.append(("warn", f"{p['rel']} 缺 user_id：档位认不出这个人，归档了也当陌生人"))
        if not p["relation"]:
            issues.append(("warn", f"{p['rel']} 缺 relation 字段（拥有者/熟人/陌生人）"))

    # 同一条问题会被多行链接重复触发，去重且保持原顺序
    return list(dict.fromkeys(issues))


def main() -> int:
    ap = argparse.ArgumentParser(description="记忆层体检")
    ap.add_argument("--persona", default="", help="人格库目录，默认取 config.toml")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--fix", action="store_true", help="重建四张 hub 索引")
    args = ap.parse_args()

    cfg = load(args.config)
    persona_dir = Path(args.persona or cfg.bot.persona_dir)

    if args.fix:
        print("重建 hub：", rebuild_all(persona_dir / "memory"))

    issues = check(persona_dir)
    if not issues:
        print(f"体检通过：{persona_dir.name}/memory")
        return 0

    errors = [i for i in issues if i[0] == "error"]
    warns = [i for i in issues if i[0] == "warn"]
    for level, msg in issues:
        print(f"  {'ERROR' if level == 'error' else 'WARN '} {msg}")
    print(f"\n合计：{len(errors)} 错误 / {len(warns)} 警告")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

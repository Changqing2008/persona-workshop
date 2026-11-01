"""人格库注册表与骨架生成。

程序侧唯一知道"有哪些人格库"的地方。人格库仍按 agents.md 的约定保持只读，
例外只有三处：raw/identity.md（身份母本）、preferences.toml（关系配置）、data/（会话与备份）。

三件事：
1. 注册表 data/personas.index.toml —— 记录库路径、显示名、归档标记。
   用注册表而不是固定根目录，是因为人格库可以在任意位置（现有库就放在工作区根下）。
2. preferences.toml —— 随库走的 relationship 配置（昵称 / 拥有者 / 称呼）。
3. create_persona() —— 按 templates/ 生成一套空骨架，即"还没填原素材"的状态。
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import log, tomlwrite

log_ = log.get("library")

TEMPLATE_DIR = Path(__file__).parent / "templates"

DEFAULT_INDEX_REL = Path("data") / "personas.index.toml"
PREFS_NAME = "preferences.toml"
DATA_DIR = "data"
SESSIONS_NAME = "sessions.db"
HISTORY_DIR = "identity_history"
DRIFT_NAME = "drift.json"

# 与 persona.py 的 wiki 索引保持一致：八页人格知识。
# 2026-10-31 下线 memories.md（高频梗）：那页只放"她反复提的那几句"，
# 攒了半年仍是空的，且与 memory/ 的职责重叠 —— 往事归记忆层，wiki 不设转述页。
# 存量库里的 memories.md 不删，靠 persona.RETIRED_PAGES 跳过。
WIKI_PAGES = (
    "identity.md",
    "traits.md",
    "speech.md",
    "reactions.md",
    "values.md",
    "relationships.md",
    "boundaries.md",
    "exemplars.md",
)

# 记忆层骨架。与 wiki/ 平级：wiki 是"她是谁"，memory 是"她记得什么"。
# 混在一棵树里迟早被当成人格页注入，所以物理分家（见 docs/记忆系统-LLM-wiki模式设计.md）。
MEM_DIRS = ("_hub", "people", "projects", "domains", "timeline", "pending")
MEM_PAGES = (
    "_hub/people.md",
    "_hub/projects.md",
    "_hub/domains.md",
    "_hub/timeline.md",
)

# 目录名不能带这些字符，Windows 上是硬限制
_BAD_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def today() -> str:
    return time.strftime("%Y-%m-%d")


def _render(name: str, **fields: str) -> str:
    """模板渲染。只做字面替换，不用 format，避免 md 里的花括号被吃掉。"""
    path = TEMPLATE_DIR / name
    text = path.read_text(encoding="utf-8")
    for key, val in fields.items():
        text = text.replace("{{" + key.upper() + "}}", str(val))
    return text


# ---------- 路径约定 ----------


def sessions_path(persona_dir: Path | str) -> Path:
    return Path(persona_dir) / DATA_DIR / SESSIONS_NAME


def history_path(persona_dir: Path | str) -> Path:
    return Path(persona_dir) / DATA_DIR / HISTORY_DIR


def prefs_path(persona_dir: Path | str) -> Path:
    return Path(persona_dir) / PREFS_NAME


def drift_path(persona_dir: Path | str) -> Path:
    """行为参数状态：一人一份，永远落在库内。

    这里**故意不提供全局回退**。会话库缺了回退到旧库只是"记忆串了"，
    漂移缺了回退到旧库等于把甲的性格状态直接套到乙头上——两者不可比。
    漂移本身可从事件流重建，丢了重建一遍就是出厂状态，代价远小于串库。
    """
    return Path(persona_dir) / DATA_DIR / DRIFT_NAME


def ensure_db(persona_dir: Path | str) -> Path:
    """确保库内有自己的会话库，缺了就建空的（连表结构一起）。

    建库时不建它，resolve_db 就会回退到全局 db，新建的人格库一启动
    就能读到上一个角色的全部聊天记录——这是最不该发生的串库。
    """
    from .memory import Memory  # 延迟导入：memory 不依赖本模块，但反过来容易成环

    db = sessions_path(persona_dir)
    if not db.is_file():
        db.parent.mkdir(parents=True, exist_ok=True)
        Memory(db).close()
        log_.info("已建空会话库：%s", db)
    return db


def resolve_db(persona_dir: Path | str, fallback: Path) -> Path:
    """会话库位置：库内有 data/sessions.db 就用它，否则回退到旧的全局 db。

    回退是为了让未迁移的老库行为完全不变——迁移与否都能启动。
    新建库由 create_persona / _import_persona 预先建好空库，走不到回退分支。
    """
    db = sessions_path(persona_dir)
    return db if db.is_file() else fallback


def drift_summary(persona_dir: Path | str) -> dict[str, Any] | None:
    """库内漂移状态的一览，总端卡片用。没有就返回 None。

    只读 JSON，不实例化 DriftStore：总端要列全部库，不该为一个数字
    把每个库的状态机都构造一遍。
    """
    import json

    path = drift_path(persona_dir)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    knobs = raw.get("knobs") or {}
    base = raw.get("baseline") or {}
    if not isinstance(knobs, dict):
        return None
    return {
        "turns": int(raw.get("turns", 0) or 0),
        "last_kind": str(raw.get("last_kind", "") or ""),
        "knobs": {k: float(knobs.get(k, 0.0) or 0.0) for k in knobs if isinstance(knobs.get(k), (int, float))},
        "baseline": {k: float(base.get(k, 0.0) or 0.0) for k in base if isinstance(base.get(k), (int, float))},
    }


def resolve_history(persona_dir: Path | str, fallback: Path) -> Path:
    """身份母本备份目录：库内有 data/identity_history 就用它，否则回退全局。"""
    hist = history_path(persona_dir)
    return hist if hist.is_dir() else fallback


# ---------- preferences.toml ----------


def read_prefs(persona_dir: Path | str) -> dict[str, Any]:
    path = prefs_path(persona_dir)
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        log_.warning("读取 %s 失败：%s", path, exc)
        return {}


def write_prefs(persona_dir: Path | str, prefs: dict[str, Any]) -> Path:
    path = prefs_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlwrite.dumps(prefs), encoding="utf-8")
    return path


# 一份 preferences.toml = 一个库的整份配置。
#
# 例外只有一个：bot.persona_dir 是自指的（"程序启动时加载哪个库"），
# 写进库内就等于"用 A 库的偏好把自己指成 B"，永远不能跟库走。
GLOBAL_PATHS: frozenset[str] = frozenset({"bot.persona_dir"})


def _cfg_paths(path: str) -> list[str]:
    """表单路径 → Config 属性路径。`adapter.official.x` 在 Config 上是 `official.x`。"""
    parts = path.split(".")
    return parts[1:] if parts and parts[0] == "adapter" else parts


def _walk_leaves(data: dict[str, Any], prefix: str = ""):
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _walk_leaves(value, path)
        else:
            yield path, value


def get_cfg_value(cfg: Any, path: str, default: Any = None) -> Any:
    """按表单路径从 Config 取值。"""
    cur: Any = cfg
    for part in _cfg_paths(path):
        cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def set_cfg_value(cfg: Any, path: str, value: Any) -> bool:
    """按表单路径往 Config 写值，带类型对齐。成功返回 True。

    类型对齐不是洁癖：TOML 里 `10` 和 `10.0` 是两个类型，
    `int(float)` 会静默丢小数，反过来 int 字段收到 float 又会在
    range 比较时报错。这里按现有值的类型归一，不对就整条跳过并记一笔。
    """
    parts = _cfg_paths(path)
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return False
    key = parts[-1]
    if not hasattr(obj, key):
        return False
    cur = getattr(obj, key)
    try:
        if isinstance(cur, bool):
            value = bool(value)
        elif isinstance(cur, int):
            value = int(value)
        elif isinstance(cur, float):
            value = float(value)
        elif isinstance(cur, list):
            value = [str(x).strip() for x in (value or []) if str(x).strip()]
        elif isinstance(cur, str):
            value = str(value)
    except (TypeError, ValueError):
        log_.warning("preferences 值类型不对，跳过：%s = %r", path, value)
        return False
    setattr(obj, key, value)
    return True


def export_cfg_paths(cfg: Any, paths: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """把当前生效的配置按给定路径导出成一份 prefs 骨架。

    新建库时用它打底：新库不该从一个空配置开始，否则点连接只会提示缺凭证。
    导出的是"现在跑着的这一套"，用户想分开再自己去库里改。
    """
    out: dict[str, Any] = {}
    for path in paths:
        if path in GLOBAL_PATHS:
            continue
        value = get_cfg_value(cfg, path, None)
        if value is not None:
            tomlwrite.set_path(out, path, value)
    return out


def prefs_from_config(
    cfg: Any,
    display_name: str = "",
    summary: str = "",
    paths: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """把配置抽成一份库内 preferences。

    paths 为空时退化成老行为（只含关系项），保证老调用方不受影响；
    webui 会传全部表单路径，于是新库拿到的是整份配置。
    """
    prefs: dict[str, Any] = {
        "persona": {"name": display_name or cfg.bot.nickname, "summary": summary},
    }
    if paths:
        prefs.update(export_cfg_paths(cfg, paths))
        return prefs
    return {
        **prefs,
        "bot": {"nickname": cfg.bot.nickname},
        "owner": {
            "ids": list(cfg.owner.ids),
            "call_me": cfg.owner.call_me,
            "familiar_turns": cfg.owner.familiar_turns,
            "carry_summary": cfg.owner.carry_summary,
        },
    }


def persist_boot_dir(cfg_path: Path | str, persona_dir: Path | str) -> dict[str, Any]:
    """把「开机读哪个库」写回全局 config.toml。切库成功后调用。

    启动库不是某个库的属性：一个进程同时只能跑一个人，写进库内就等于
    "用 A 的偏好把自己指成 B"。所以它只存在全局文件里，并且只由「切换」
    这一个动作改写——配置页那一格是只读的。

    幂等：目标值已经在文件里就一个字节都不写，避免每次切库都动配置。

    顺带同步 `bot.nickname`：全局那份是"库内缺 nickname 时的兜底"，
    不同步就会留着上一个库的名字，出现"启动长青、自称示例角色"。
    """
    path = Path(cfg_path)
    if not path.is_file():
        return {"ok": False, "error": f"配置文件不存在：{path}"}
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {"ok": False, "error": f"读取配置失败：{exc}"}

    target = str(Path(persona_dir))
    nickname = str(tomlwrite.get_path(read_prefs(persona_dir), "bot.nickname", "") or "")
    changed: list[str] = []
    if str(tomlwrite.get_path(raw, "bot.persona_dir", "") or "") != target:
        tomlwrite.set_path(raw, "bot.persona_dir", target)
        changed.append("bot.persona_dir")
    if nickname and str(tomlwrite.get_path(raw, "bot.nickname", "") or "") != nickname:
        tomlwrite.set_path(raw, "bot.nickname", nickname)
        changed.append("bot.nickname")
    if not changed:
        return {"ok": True, "changed": False, "raw": raw}

    try:
        path.with_suffix(".toml.bak").write_bytes(path.read_bytes())
        path.write_text(tomlwrite.dumps(raw), encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"写入配置失败：{exc}", "raw": raw}
    log_.info("启动库已随切换改写：%s（%s）", target, ", ".join(changed))
    return {"ok": True, "changed": True, "raw": raw}


def apply_prefs(cfg: Any, persona_dir: Path | str, set_dir: bool = True) -> Any:
    """库级覆盖：把 preferences.toml 里出现的每一项都盖到 cfg 上。

    只覆盖"文件里真有"的键，缺的键沿用 config.toml —— 所以老的
    五字段 prefs、或者手动删掉几行的 prefs，都不会把配置打成默认值。

    set_dir=False 用于"保存配置"这种场景：用户可能刚在全局里改了 persona_dir
    想换启动库，此时不能又被库内路径盖回去。
    """
    root = Path(persona_dir)
    if set_dir:
        cfg.bot.persona_dir = str(root)
    prefs = read_prefs(root)
    if not prefs:
        return cfg

    for path, value in _walk_leaves(prefs):
        if path in GLOBAL_PATHS or path.startswith("persona."):
            continue
        set_cfg_value(cfg, path, value)
    return cfg


# ---------- 注册表 ----------


@dataclass(slots=True)
class PersonaEntry:
    id: str
    """人格库 id，取目录名，URL 里用原样编码。"""
    name: str
    """显示名。总端卡片与标题用这个。"""
    path: Path
    created: str = ""
    archived: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": str(self.path),
            "created": self.created,
            "archived": self.archived,
            "summary": self.summary,
        }


class Library:
    """人格库注册表。线程内同步读写，条目数很少，不值得上锁。"""

    def __init__(self, index_path: Path | str = DEFAULT_INDEX_REL) -> None:
        self.index_path = Path(index_path)
        self._entries: dict[str, PersonaEntry] = {}
        self.load()

    # ---------- 持久化 ----------

    def load(self) -> None:
        self._entries = {}
        if not self.index_path.is_file():
            return
        try:
            with self.index_path.open("rb") as fh:
                raw = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            log_.warning("读取注册表失败 %s：%s", self.index_path, exc)
            return
        table = raw.get("personas", {})
        if not isinstance(table, dict):
            return
        for pid, item in table.items():
            if not isinstance(item, dict):
                continue
            path = Path(str(item.get("path", "")))
            if not str(item.get("path", "")).strip():
                continue
            self._entries[pid] = PersonaEntry(
                id=pid,
                name=str(item.get("name") or pid),
                path=path,
                created=str(item.get("created", "")),
                archived=bool(item.get("archived", False)),
                summary=str(item.get("summary", "")),
            )

    def save(self) -> None:
        data: dict[str, Any] = {"personas": {}}
        for pid, e in self._entries.items():
            data["personas"][pid] = {
                "name": e.name,
                "path": str(e.path),
                "created": e.created,
                "archived": e.archived,
                "summary": e.summary,
            }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(tomlwrite.dumps(data), encoding="utf-8")

    # ---------- 查询 ----------

    def entries(self, include_archived: bool = False) -> list[PersonaEntry]:
        out = [e for e in self._entries.values() if include_archived or not e.archived]
        return sorted(out, key=lambda e: (e.created or "", e.id))

    def get(self, pid: str) -> PersonaEntry | None:
        return self._entries.get(pid)

    def by_path(self, path: Path | str) -> PersonaEntry | None:
        target = Path(path).resolve()
        for e in self._entries.values():
            try:
                if e.path.resolve() == target:
                    return e
            except OSError:
                continue
        return None

    def exists(self, pid: str) -> bool:
        return pid in self._entries

    # ---------- 变更 ----------

    def register(
        self,
        path: Path | str,
        name: str = "",
        summary: str = "",
        archived: bool = False,
    ) -> PersonaEntry:
        """登记一个已存在的目录。重复登记时只更新显示名。"""
        root = Path(path)
        pid = root.name
        entry = PersonaEntry(
            id=pid,
            name=name or pid,
            path=root,
            created=today(),
            archived=archived,
            summary=summary,
        )
        old = self._entries.get(pid)
        if old is not None:
            entry.created = old.created or entry.created
            entry.archived = archived or old.archived
            if not name:
                entry.name = old.name
            if not summary:
                entry.summary = old.summary
        self._entries[pid] = entry
        self.save()
        return entry

    def archive(self, pid: str) -> PersonaEntry:
        """归档：只在注册表里记一笔，不动磁盘上的任何文件。"""
        entry = self._entries.get(pid)
        if entry is None:
            raise KeyError(f"人格库不存在：{pid}")
        entry.archived = True
        self.save()
        log_.info("人格库已归档（文件未动）：%s", entry.path)
        return entry

    def unarchive(self, pid: str) -> PersonaEntry:
        entry = self._entries.get(pid)
        if entry is None:
            raise KeyError(f"人格库不存在：{pid}")
        entry.archived = False
        self.save()
        return entry

    def forget(self, pid: str) -> PersonaEntry:
        """从注册表摘掉一条，磁盘上的文件一个都不动。

        用途是清理失效条目（目录已挪走/删掉），不是"删库"。
        """
        entry = self._entries.pop(pid, None)
        if entry is None:
            raise KeyError(f"人格库不存在：{pid}")
        self.save()
        log_.info("注册表已移除条目（文件未动）：%s", entry.path)
        return entry

    def update_meta(self, pid: str, name: str = "", summary: str = "") -> PersonaEntry:
        entry = self._entries.get(pid)
        if entry is None:
            raise KeyError(f"人格库不存在：{pid}")
        if name:
            entry.name = name
        if summary:
            entry.summary = summary
        self.save()
        return entry


# ---------- 状态统计 ----------


def _wiki_stats(root: Path) -> dict[str, Any]:
    """wiki 填充度。判定复用 persona 的同一套占位符规则，避免两处口径打架。"""
    from .persona import _is_filled  # 延迟导入，避免与 persona 互相牵制

    wiki = root / "wiki"
    if not wiki.is_dir():
        return {"pages": 0, "filled": 0}
    pages = 0
    filled = 0
    for path in wiki.glob("*.md"):
        pages += 1
        try:
            if _is_filled(path.read_text(encoding="utf-8")):
                filled += 1
        except (OSError, UnicodeDecodeError):
            continue
    return {"pages": pages, "filled": filled}


def _session_stats(root: Path) -> dict[str, Any]:
    """会话概览。直接查库，不实例化 Memory——总端要列全部库，不能每个都开连接。"""
    db = sessions_path(root)
    if not db.is_file():
        return {"sessions": 0, "messages": 0, "last_ts": 0.0}
    try:
        conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return {"sessions": 0, "messages": 0, "last_ts": 0.0}
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS s, COALESCE(SUM(n),0) AS m, COALESCE(MAX(last_ts),0) AS t "
            "FROM sessions"
        ).fetchone()
        return {
            "sessions": int(row[0] or 0),
            "messages": int(row[1] or 0),
            "last_ts": float(row[2] or 0.0),
        }
    except sqlite3.Error:
        return {"sessions": 0, "messages": 0, "last_ts": 0.0}
    finally:
        conn.close()


def stats(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    out: dict[str, Any] = {"name": root.name, "path": str(root), "exists": root.is_dir()}
    out.update(_wiki_stats(root))
    out.update(_session_stats(root))
    out["has_index"] = (root / "index.md").is_file()
    out["has_agents"] = (root / "agents.md").is_file()
    out["has_identity"] = (root / "raw" / "identity.md").is_file()
    out["has_prefs"] = prefs_path(root).is_file()
    out["has_drift"] = drift_path(root).is_file()
    return out


# ---------- 建库 ----------


def create_persona(
    root: Path | str,
    name: str = "",
    summary: str = "",
    prefs: dict[str, Any] | None = None,
) -> PersonaEntry:
    """生成一套空骨架——还没填任何原素材的状态。

    root 是库目录本身（不存在则创建）。name 是显示名，留空取目录名。
    已存在且非空的目录直接拒绝，绝不覆盖别人的东西。
    """
    root = Path(root)
    pid = root.name
    if not pid.strip():
        raise ValueError("人格库目录名不能为空")
    if _BAD_NAME.search(pid):
        raise ValueError('目录名不能包含 \\ / : * ? " < > | 等字符')
    if root.exists():
        existing = [p for p in root.iterdir()]
        if existing:
            raise ValueError(f"目录已存在且非空：{root}")

    display = (name or pid).strip()
    date = today()

    (root / "wiki").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "assets").mkdir(parents=True, exist_ok=True)
    (root / DATA_DIR / HISTORY_DIR).mkdir(parents=True, exist_ok=True)
    # 空会话库：不建它就等于把新库的对话写进启动库，
    # 两个人格的聊天记录会混在同一条时间线上，且没有任何提示。
    ensure_db(root)

    # 契约文档与总括页
    (root / "agents.md").write_text(
        _render("agents.md", persona=pid), encoding="utf-8"
    )
    (root / "index.md").write_text(
        _render("index.md", persona=display, summary=summary.strip() or "_待填_"),
        encoding="utf-8",
    )
    (root / "log.md").write_text(
        _render("log.md", date=date), encoding="utf-8"
    )

    # wiki 九页空壳
    for page in WIKI_PAGES:
        src = TEMPLATE_DIR / "wiki" / page
        (root / "wiki" / page).write_text(
            _render(f"wiki/{page}", date=date, persona=display), encoding="utf-8"
        )
        del src

    # 记忆层骨架（空目录 + 四个 hub 索引）
    for sub in MEM_DIRS:
        (root / "memory" / sub).mkdir(parents=True, exist_ok=True)
    for page in MEM_PAGES:
        (root / "memory" / page).parent.mkdir(parents=True, exist_ok=True)
        (root / "memory" / page).write_text(
            _render(f"memory/{page}", date=date), encoding="utf-8"
        )

    # 身份母本：复用 seed.TEMPLATE，保证与「身份」页的七节骨架同源
    from .seed import TEMPLATE as IDENTITY_TEMPLATE

    (root / "raw" / "identity.md").write_text(IDENTITY_TEMPLATE, encoding="utf-8")

    # 关系配置
    base = prefs or {}
    base.setdefault("persona", {})
    base["persona"]["name"] = display
    base["persona"]["summary"] = summary.strip()
    base.setdefault("bot", {})
    if not base["bot"].get("nickname"):
        base["bot"]["nickname"] = display
    base.setdefault(
        "owner",
        {"ids": [], "call_me": "", "familiar_turns": 12, "carry_summary": True},
    )
    write_prefs(root, base)

    log_.info("已生成人格库骨架：%s（wiki %d 页 / raw 空 / data 就绪）", root, len(WIKI_PAGES))

    return PersonaEntry(
        id=pid,
        name=display,
        path=root,
        created=date,
        summary=summary.strip(),
    )


def validate_name(name: str) -> str:
    """校验库名，返回错误信息；空串表示通过。"""
    if not name.strip():
        return "库名不能为空"
    if _BAD_NAME.search(name):
        return '库名不能包含 \\ / : * ? " < > | 等字符'
    if name in {".", ".."}:
        return "库名非法"
    return ""


def copy_session_db(src: Path | str, persona_dir: Path | str) -> Path:
    """把旧的全局 db 复制进库内。只复制，不移动、不删除源文件。"""
    src = Path(src)
    dst = sessions_path(persona_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖：{dst}")
    shutil.copy2(src, dst)
    log_.info("会话库已复制：%s → %s", src, dst)
    return dst

"""会话记忆。SQLite 单文件，同步访问（写入量极小，不值得引入异步 ORM）。

隔离粒度是 session_id：
- 私聊：private:<uid>，每个人独立
- 群聊：group:<gid>，整群共享一条上下文

另外两张表用于工程可靠性：
- seen 表做消息去重（协议端重连时可能重放事件）
- summaries 表存滚动摘要（长会话不会无限膨胀上下文）
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from . import log

log_ = log.get("memory")

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    ts         REAL    NOT NULL,
    msg_key    TEXT,
    -- 群聊里这句话是谁说的。原来自带在 content 的"名字: "前缀里，
    -- 取不出来就没法按人聚，群聊一热闹就糊成一团。assistant 行为 NULL。
    user_id    TEXT
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, id);

-- 会话索引。没有它，列会话要做一次全表 GROUP BY：
-- 12 万条消息时实测 478ms，而且卡在智能体与面板共用的那条连接上。
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    n          INTEGER NOT NULL DEFAULT 0,
    last_ts    REAL    NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    upto_id    INTEGER NOT NULL,
    updated    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS seen (
    key  TEXT PRIMARY KEY,
    ts   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    user_id    TEXT PRIMARY KEY,
    user_name  TEXT,
    first_seen REAL,
    last_seen  REAL,
    turns      INTEGER DEFAULT 0
);
"""


class Memory:
    def __init__(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # 写放大是这里最大的隐性开销：默认 rollback journal 下每次 commit 都要
        # fsync 日志文件，Windows 上实测单次 7.8ms，一条消息 5 次提交就是 31ms，
        # 而这 31ms 是同步跑在事件循环线程上的。WAL + NORMAL 后同口径 0.3ms（100 倍）。
        # 代价是掉电可能丢最后几条消息——聊天记忆可以接受，权衡后取性能。
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        # 这条连接被事件循环线程和 WebUI 的多个 HTTP 线程共用，必须自己串行化：
        # sqlite3 的 check_same_thread=False 只是关掉检查，不提供互斥。
        # 用 RLock 是因为 list_sessions 之类的方法内部还会调其他方法。
        self._lock = threading.RLock()
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._ensure_columns()
        self._backfill_sessions()
        self._prune_seen()
        log_.info("记忆库就绪：%s", db_path)

    def _ensure_columns(self) -> None:
        """老库升级：messages 补 user_id 列。

        索引必须在这里建，不能写进 SCHEMA——老库的 messages 还没有这一列，
        executescript 阶段建索引会直接报 no such column。
        """
        with self._lock:
            cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(messages)")}
            added = False
            if "user_id" not in cols:
                self.conn.execute("ALTER TABLE messages ADD COLUMN user_id TEXT")
                added = True
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(session_id, user_id, id)"
            )
            self.conn.commit()
        if added:
            log_.info("消息表已升级：新增 user_id 列（老消息该列为空，不影响读取）")

    def _prune_seen(self, keep_days: int = 7) -> None:
        """去重表只对"重连重放"有意义，留一周足够。

        不清理的话它跟消息表一起无限涨：每来一条消息一行，永远不删。
        """
        cutoff = time.time() - keep_days * 86400
        with self._lock:
            cur = self.conn.execute("DELETE FROM seen WHERE ts < ?", (cutoff,))
            self.conn.commit()
        if cur.rowcount:
            log_.info("清理过期去重记录 %d 条", cur.rowcount)

    def _backfill_sessions(self) -> None:
        """老库升级：sessions 表是后加的，首次启动时从 messages 补一次索引。"""
        row = self.conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()
        if row and int(row["c"]) > 0:
            return
        row = self.conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()
        if not row or int(row["c"]) == 0:
            return
        with self._lock:
            self.conn.execute(
                "INSERT INTO sessions(session_id, n, last_ts) "
                "SELECT session_id, COUNT(*), MAX(ts) FROM messages GROUP BY session_id "
                "ON CONFLICT(session_id) DO UPDATE SET n=excluded.n, last_ts=excluded.last_ts"
            )
            self.conn.commit()
        log_.info("已补建会话索引：%d 条记录", int(row["c"]))

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass

    # ---------- 去重 ----------

    def claim(self, key: str | None) -> bool:
        """抢占一个消息键。返回 True 表示是新消息，False 表示已处理过。"""
        if not key:
            return True
        with self._lock:
            try:
                self.conn.execute("INSERT INTO seen(key, ts) VALUES (?, ?)", (key, time.time()))
                self.conn.commit()
                return True
            except sqlite3.IntegrityError:
                # 失败会留下一个未提交的事务，必须回滚，否则它挂着写锁等下一次 commit
                self.conn.rollback()
                return False

    # ---------- 消息 ----------

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        msg_key: str | None = None,
        user_id: str | None = None,
    ) -> int:
        now = time.time()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO messages(session_id, role, content, ts, msg_key, user_id) "
                "VALUES (?,?,?,?,?,?)",
                (session_id, role, content, now, msg_key, user_id),
            )
            # 会话索引与消息同一次提交：要么都进，要么都不进
            self.conn.execute(
                "INSERT INTO sessions(session_id, n, last_ts) VALUES (?, 1, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET n=n+1, last_ts=excluded.last_ts",
                (session_id, now),
            )
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def recent(self, session_id: str, limit: int) -> list[dict[str, str]]:
        """最近 limit 条，按时间正序返回（旧的在前）。"""
        if limit <= 0:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def tail_after(self, session_id: str, after_id: int, limit: int = 200) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT id, role, content FROM messages "
                "WHERE session_id=? AND id>? ORDER BY id ASC LIMIT ?",
                (session_id, after_id, limit),
            ).fetchall()

    def member_exchanges(
        self,
        session_id: str,
        user_id: str,
        before_id: int = 0,
        turns: int = 4,
    ) -> list[dict[str, Any]]:
        """某人在这个会话里与她的最近几轮往来，时间正序。

        群聊的上下文是按整群存的，人一多就糊在一起：她既记不住跟某个人聊到
        哪儿了，也分不清哪句是冲自己来的。这里按人抽一段——只留这个人说的话
        和他之后紧跟着的那条回复，别人的插话一律不进。

        before_id 传当前这批的起点：这批正在说的话马上会出现在对话块里，
        没必要再抄一遍。
        """
        if not user_id:
            return []
        turns = max(1, int(turns))
        cut = int(before_id) if before_id else 1 << 62
        with self._lock:
            rows = self.conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id=? AND user_id=? AND id<? ORDER BY id DESC LIMIT ?",
                (session_id, user_id, cut, turns),
            ).fetchall()
            if not rows:
                return []
            ids = [int(r["id"]) for r in rows]
            lo, hi = min(ids), max(ids)
            # 多给一点尾巴，让最后一条他说的后面那句回复也能带进来
            span = self.conn.execute(
                "SELECT id, role, content, ts, user_id FROM messages "
                "WHERE session_id=? AND id BETWEEN ? AND ? "
                "AND (user_id=? OR role='assistant') ORDER BY id",
                (session_id, lo, hi + 4, user_id),
            ).fetchall()

        out: list[dict[str, Any]] = []
        awaiting = False
        for r in span:
            if r["user_id"] == user_id:
                out.append(
                    {"role": "user", "content": r["content"], "ts": float(r["ts"] or 0)}
                )
                awaiting = True
            elif r["role"] == "assistant" and awaiting:
                out.append(
                    {"role": "assistant", "content": r["content"], "ts": float(r["ts"] or 0)}
                )
                awaiting = False
        return out

    def last_id(self, session_id: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(MAX(id),0) AS m FROM messages WHERE session_id=?", (session_id,)
            ).fetchone()
        return int(row["m"]) if row else 0

    def recent_before(
        self,
        session_id: str,
        before_id: int,
        limit: int,
        after_id: int = 0,
    ) -> list[dict[str, Any]]:
        """取 id 落在 (after_id, before_id) 区间里的最近 limit 条，时间正序。

        合批场景必须按 id 边界切，不能"取最后 N 条再去掉几条"——批内消息落库后
        随时可能又来新消息，按条数删会把别人的话一起切掉。
        """
        if limit <= 0 or before_id <= 0:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, role, content FROM messages "
                "WHERE session_id=? AND id<? AND id>? ORDER BY id DESC LIMIT ?",
                (session_id, before_id, after_id, limit),
            ).fetchall()
        return [
            {"id": int(r["id"]), "role": str(r["role"]), "content": str(r["content"])}
            for r in reversed(rows)
        ]

    def last_ts(self, session_id: str) -> float:
        """该会话最后一条消息的时间戳。没有返回 0。"""
        with self._lock:
            row = self.conn.execute(
                "SELECT last_ts FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        return float(row["last_ts"]) if row else 0.0

    def pending_count(self, session_id: str) -> int:
        """自上次摘要之后累积了多少条。"""
        with self._lock:
            row = self.conn.execute(
                "SELECT upto_id FROM summaries WHERE session_id=?", (session_id,)
            ).fetchone()
            upto = int(row["upto_id"]) if row else 0
            cnt = self.conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id=? AND id>?",
                (session_id, upto),
            ).fetchone()
        return int(cnt["c"]) if cnt else 0

    # ---------- 摘要 ----------

    def get_summary(self, session_id: str) -> tuple[str, int]:
        with self._lock:
            row = self.conn.execute(
                "SELECT summary, upto_id FROM summaries WHERE session_id=?", (session_id,)
            ).fetchone()
        if not row:
            return "", 0
        return str(row["summary"]), int(row["upto_id"])

    def set_summary(self, session_id: str, summary: str, upto_id: int) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO summaries(session_id, summary, upto_id, updated) VALUES (?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "summary=excluded.summary, upto_id=excluded.upto_id, updated=excluded.updated",
                (session_id, summary, upto_id, time.time()),
            )
            self.conn.commit()

    # ---------- 用户画像（极简） ----------

    def touch_user(self, user_id: str, user_name: str = "") -> None:
        now = time.time()
        with self._lock:
            self.conn.execute(
                "INSERT INTO profiles(user_id, user_name, first_seen, last_seen, turns) "
                "VALUES (?,?,?,?,1) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "last_seen=excluded.last_seen, turns=turns+1, "
                "user_name=COALESCE(NULLIF(excluded.user_name,''), user_name)",
                (user_id, user_name, now, now),
            )
            self.conn.commit()

    def profile(self, user_id: str) -> dict[str, Any] | None:
        """取一个人的极简画像。没聊过返回 None。"""
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id, user_name, first_seen, last_seen, turns "
                "FROM profiles WHERE user_id=?",
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "user_id": str(row["user_id"]),
            "user_name": str(row["user_name"] or ""),
            "first_seen": float(row["first_seen"] or 0),
            "last_seen": float(row["last_seen"] or 0),
            "turns": int(row["turns"] or 0),
        }

    def stats(self) -> dict[str, Any]:
        """全部走 sessions 小表，不做 messages 全表 COUNT —— 那条查询随消息量线性变慢，
        而它被控制台每 3 秒拉一次。"""

        def one(sql: str) -> int:
            with self._lock:
                row = self.conn.execute(sql).fetchone()
            return int(row[0]) if row else 0

        return {
            "path": str(self.db_path),
            "messages": one("SELECT COALESCE(SUM(n),0) FROM sessions"),
            "sessions": one("SELECT COUNT(*) FROM sessions"),
            "users": one("SELECT COUNT(*) FROM profiles"),
        }

    def clear_session(self, session_id: str) -> int:
        """清空某个会话的记忆与摘要，返回删除的消息条数。"""
        with self._lock:
            cur = self.conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            self.conn.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
            self.conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            self.conn.commit()
            return cur.rowcount or 0

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        """会话列表。走 sessions 索引表，不再对 messages 做全表 GROUP BY。

        12 万条消息下老写法 478ms，且卡在与智能体共用的连接上；现在只扫几十行。
        每条记录的 pending 用带索引的相关子查询取，避免 N+1 次往返。
        """
        sql = (
            "SELECT s.session_id, s.n, s.last_ts, "
            "       COALESCE(m.summary, '') AS summary, "
            "       (SELECT COUNT(*) FROM messages x "
            "         WHERE x.session_id=s.session_id AND x.id>COALESCE(m.upto_id,0)) AS pending "
            "FROM sessions s LEFT JOIN summaries m ON m.session_id=s.session_id "
            "ORDER BY s.last_ts DESC LIMIT ?"
        )
        with self._lock:
            rows = self.conn.execute(sql, (limit,)).fetchall()
        return [
            {
                "session_id": str(r["session_id"]),
                "count": int(r["n"]),
                "last": r["last_ts"],
                "pending": int(r["pending"] or 0),
                "has_summary": bool(str(r["summary"]).strip()),
            }
            for r in rows
        ]

    def session_messages(self, session_id: str, limit: int = 200) -> tuple[list[dict[str, Any]], int]:
        """某会话的消息，最新 limit 条、旧的在前，带 id/ts 供界面展示。

        返回 (消息列表, 总条数)。列表可能比总数短 —— 更早的仍在库里，只是没翻出来。
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, role, content, ts, user_id FROM messages WHERE session_id=? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, max(1, limit)),
            ).fetchall()
            cnt = self.conn.execute(
                "SELECT n FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        total = int(cnt["n"]) if cnt else len(rows)
        msgs = [
            {
                "id": int(r["id"]),
                "role": str(r["role"]),
                "content": str(r["content"]),
                "ts": float(r["ts"]),
                "user_id": str(r["user_id"] or ""),
            }
            for r in reversed(rows)
        ]
        return msgs, total

-- sessions.db schema v1（冻结）
-- 与通道器侧 bridge/memory.py 的 SCHEMA 一字不差。改库结构 = 违反库契约。
-- 聊天程序首次连接时若库不存在，按本文件建库；已存在则禁止任何 DDL。

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

-- 会话索引。没有它，列会话要做一次全表 GROUP BY。
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

-- 冻结后补建的索引（老库升级时由程序补，新库可直接带上）
CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(session_id, user_id, id);

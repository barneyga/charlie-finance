import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS series_meta (
    series_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    frequency       TEXT NOT NULL,
    units           TEXT,
    source          TEXT NOT NULL DEFAULT 'fred',
    last_updated    TEXT,
    observation_start TEXT,
    observation_end   TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    series_id   TEXT NOT NULL,
    date        TEXT NOT NULL,
    value       REAL,
    PRIMARY KEY (series_id, date),
    FOREIGN KEY (series_id) REFERENCES series_meta(series_id)
);

CREATE INDEX IF NOT EXISTS idx_obs_date ON observations(date);
CREATE INDEX IF NOT EXISTS idx_obs_series ON observations(series_id);

CREATE TABLE IF NOT EXISTS derived_indicators (
    indicator_id TEXT NOT NULL,
    date         TEXT NOT NULL,
    value        REAL,
    metadata     TEXT,
    PRIMARY KEY (indicator_id, date)
);
"""


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_schema(self):
        self.conn.executescript(SCHEMA_SQL)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

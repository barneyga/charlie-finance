from datetime import datetime

import pandas as pd

from charlie.storage.db import Database


def upsert_series_meta(
    db: Database,
    series_id: str,
    name: str,
    category: str,
    frequency: str,
    units: str | None = None,
    source: str = "fred",
):
    db.conn.execute(
        """INSERT INTO series_meta (series_id, name, category, frequency, units, source)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(series_id) DO UPDATE SET
               name=excluded.name,
               category=excluded.category,
               frequency=excluded.frequency,
               units=COALESCE(excluded.units, series_meta.units),
               source=excluded.source
        """,
        (series_id, name, category, frequency, units, source),
    )
    db.conn.commit()


def upsert_observations(db: Database, series_id: str, df: pd.DataFrame):
    """Upsert a DataFrame with DatetimeIndex and a single value column."""
    if df.empty:
        return

    rows = []
    for date, value in df.items():
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        val = float(value) if pd.notna(value) else None
        rows.append((series_id, date_str, val))

    db.conn.executemany(
        "INSERT OR REPLACE INTO observations (series_id, date, value) VALUES (?, ?, ?)",
        rows,
    )

    # Update observation range in metadata
    dates = [r[1] for r in rows if r[2] is not None]
    if dates:
        min_date = min(dates)
        max_date = max(dates)
        db.conn.execute(
            """UPDATE series_meta SET
                   observation_start = MIN(COALESCE(observation_start, ?), ?),
                   observation_end = MAX(COALESCE(observation_end, ?), ?),
                   last_updated = ?
               WHERE series_id = ?""",
            (min_date, min_date, max_date, max_date, datetime.now().isoformat(), series_id),
        )
    db.conn.commit()


def get_latest_date(db: Database, series_id: str) -> str | None:
    row = db.conn.execute(
        "SELECT observation_end FROM series_meta WHERE series_id = ?",
        (series_id,),
    ).fetchone()
    return row["observation_end"] if row else None


def query_series(
    db: Database,
    series_id: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Return a pandas Series with DatetimeIndex for the given FRED series."""
    sql = "SELECT date, value FROM observations WHERE series_id = ?"
    params: list = [series_id]

    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)

    sql += " ORDER BY date"
    rows = db.conn.execute(sql, params).fetchall()

    if not rows:
        return pd.Series(dtype=float, name=series_id)

    dates = [r["date"] for r in rows]
    values = [r["value"] for r in rows]
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id, dtype=float)


def query_multiple_series(
    db: Database,
    series_ids: list[str],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with columns for each series, aligned by date."""
    frames = {}
    for sid in series_ids:
        frames[sid] = query_series(db, sid, start, end)
    return pd.DataFrame(frames)


def get_all_series_meta(db: Database) -> list[dict]:
    rows = db.conn.execute("SELECT * FROM series_meta ORDER BY category, series_id").fetchall()
    return [dict(r) for r in rows]

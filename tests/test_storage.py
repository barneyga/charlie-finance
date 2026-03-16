"""Tests for charlie.storage — Database and models."""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.storage.db import Database
from charlie.storage.models import (
    upsert_series_meta,
    upsert_observations,
    get_latest_date,
    query_series,
    query_multiple_series,
    get_all_series_meta,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a fresh in-memory-backed Database for each test."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.init_schema()
    yield db
    db.close()


def _insert_series(db, series_id="TEST", name="Test Series", category="test", frequency="daily"):
    upsert_series_meta(db, series_id, name, category, frequency)


def _make_obs(values, start="2020-01-01", freq="D"):
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, name="obs")


# ── Database schema ───────────────────────────────────────────────────────────

class TestDatabaseSchema:
    def test_init_schema_creates_tables(self, tmp_db):
        tables = tmp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "series_meta" in table_names
        assert "observations" in table_names
        assert "derived_indicators" in table_names

    def test_context_manager(self, tmp_path):
        db_path = tmp_path / "ctx.db"
        with Database(db_path) as db:
            db.init_schema()
            # Should be usable inside context
            tables = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            assert len(tables) > 0
        # After __exit__ the connection is closed
        assert db._conn is None

    def test_init_schema_is_idempotent(self, tmp_db):
        # Calling init_schema a second time should not raise
        tmp_db.init_schema()
        tmp_db.init_schema()

    def test_parent_directory_created(self, tmp_path):
        nested_path = tmp_path / "a" / "b" / "c" / "test.db"
        db = Database(nested_path)
        db.init_schema()
        assert nested_path.exists()
        db.close()


# ── upsert_series_meta ────────────────────────────────────────────────────────

class TestUpsertSeriesMeta:
    def test_insert_new_row(self, tmp_db):
        upsert_series_meta(tmp_db, "DGS10", "10-Year Treasury", "yield_curve", "daily")
        row = tmp_db.conn.execute(
            "SELECT * FROM series_meta WHERE series_id='DGS10'"
        ).fetchone()
        assert row is not None
        assert row["name"] == "10-Year Treasury"
        assert row["source"] == "fred"

    def test_update_existing_row(self, tmp_db):
        upsert_series_meta(tmp_db, "DGS10", "Old Name", "yield_curve", "daily")
        upsert_series_meta(tmp_db, "DGS10", "New Name", "yield_curve", "daily")
        row = tmp_db.conn.execute(
            "SELECT name FROM series_meta WHERE series_id='DGS10'"
        ).fetchone()
        assert row["name"] == "New Name"

    def test_custom_source(self, tmp_db):
        upsert_series_meta(tmp_db, "SPY", "S&P 500", "indices", "daily", source="yahoo")
        row = tmp_db.conn.execute(
            "SELECT source FROM series_meta WHERE series_id='SPY'"
        ).fetchone()
        assert row["source"] == "yahoo"

    def test_units_stored(self, tmp_db):
        upsert_series_meta(tmp_db, "CPIAUCSL", "CPI", "inflation", "monthly", units="Index")
        row = tmp_db.conn.execute(
            "SELECT units FROM series_meta WHERE series_id='CPIAUCSL'"
        ).fetchone()
        assert row["units"] == "Index"


# ── upsert_observations ───────────────────────────────────────────────────────

class TestUpsertObservations:
    def test_insert_observations(self, tmp_db):
        _insert_series(tmp_db)
        obs = _make_obs([1.0, 2.0, 3.0])
        upsert_observations(tmp_db, "TEST", obs)
        rows = tmp_db.conn.execute(
            "SELECT COUNT(*) as cnt FROM observations WHERE series_id='TEST'"
        ).fetchone()
        assert rows["cnt"] == 3

    def test_upsert_overwrites_existing(self, tmp_db):
        _insert_series(tmp_db)
        obs1 = _make_obs([1.0, 2.0], start="2020-01-01")
        obs2 = _make_obs([99.0, 99.0], start="2020-01-01")
        upsert_observations(tmp_db, "TEST", obs1)
        upsert_observations(tmp_db, "TEST", obs2)
        rows = tmp_db.conn.execute(
            "SELECT value FROM observations WHERE series_id='TEST' ORDER BY date"
        ).fetchall()
        assert rows[0]["value"] == pytest.approx(99.0)
        assert rows[1]["value"] == pytest.approx(99.0)

    def test_updates_observation_end_in_meta(self, tmp_db):
        _insert_series(tmp_db)
        obs = _make_obs([1.0, 2.0, 3.0], start="2021-06-01")
        upsert_observations(tmp_db, "TEST", obs)
        row = tmp_db.conn.execute(
            "SELECT observation_end FROM series_meta WHERE series_id='TEST'"
        ).fetchone()
        assert row["observation_end"] is not None
        assert row["observation_end"] >= "2021-06-01"

    def test_empty_dataframe_is_no_op(self, tmp_db):
        _insert_series(tmp_db)
        obs = pd.Series(dtype=float)
        upsert_observations(tmp_db, "TEST", obs)
        rows = tmp_db.conn.execute(
            "SELECT COUNT(*) as cnt FROM observations WHERE series_id='TEST'"
        ).fetchone()
        assert rows["cnt"] == 0

    def test_nan_values_stored_as_null(self, tmp_db):
        _insert_series(tmp_db)
        obs = _make_obs([1.0, float("nan"), 3.0])
        upsert_observations(tmp_db, "TEST", obs)
        rows = tmp_db.conn.execute(
            "SELECT value FROM observations WHERE series_id='TEST' ORDER BY date"
        ).fetchall()
        assert rows[1]["value"] is None


# ── get_latest_date ───────────────────────────────────────────────────────────

class TestGetLatestDate:
    def test_returns_none_for_unknown_series(self, tmp_db):
        result = get_latest_date(tmp_db, "UNKNOWN")
        assert result is None

    def test_returns_latest_date_after_insert(self, tmp_db):
        _insert_series(tmp_db)
        obs = _make_obs([1.0, 2.0, 3.0], start="2022-01-01")
        upsert_observations(tmp_db, "TEST", obs)
        latest = get_latest_date(tmp_db, "TEST")
        assert latest is not None
        assert latest >= "2022-01-01"


# ── query_series ──────────────────────────────────────────────────────────────

class TestQuerySeries:
    def _setup(self, db, values, start="2020-01-01"):
        _insert_series(db)
        obs = _make_obs(values, start=start)
        upsert_observations(db, "TEST", obs)

    def test_returns_series_with_datetime_index(self, tmp_db):
        self._setup(tmp_db, [1.0, 2.0, 3.0])
        s = query_series(tmp_db, "TEST")
        assert isinstance(s, pd.Series)
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_returns_correct_values(self, tmp_db):
        self._setup(tmp_db, [10.0, 20.0, 30.0])
        s = query_series(tmp_db, "TEST")
        assert list(s.values) == pytest.approx([10.0, 20.0, 30.0])

    def test_empty_for_unknown_series(self, tmp_db):
        s = query_series(tmp_db, "UNKNOWN")
        assert s.empty

    def test_start_filter(self, tmp_db):
        self._setup(tmp_db, [1.0, 2.0, 3.0, 4.0, 5.0], start="2020-01-01")
        s = query_series(tmp_db, "TEST", start="2020-01-03")
        assert len(s) == 3
        assert s.index[0] >= pd.Timestamp("2020-01-03")

    def test_end_filter(self, tmp_db):
        self._setup(tmp_db, [1.0, 2.0, 3.0, 4.0, 5.0], start="2020-01-01")
        s = query_series(tmp_db, "TEST", end="2020-01-03")
        assert len(s) == 3
        assert s.index[-1] <= pd.Timestamp("2020-01-03")

    def test_series_name_is_series_id(self, tmp_db):
        self._setup(tmp_db, [1.0])
        s = query_series(tmp_db, "TEST")
        assert s.name == "TEST"


# ── query_multiple_series ─────────────────────────────────────────────────────

class TestQueryMultipleSeries:
    def test_returns_dataframe_with_columns(self, tmp_db):
        for sid, vals in [("A", [1.0, 2.0, 3.0]), ("B", [4.0, 5.0, 6.0])]:
            upsert_series_meta(tmp_db, sid, sid, "test", "daily")
            upsert_observations(tmp_db, sid, _make_obs(vals))
        df = query_multiple_series(tmp_db, ["A", "B"])
        assert "A" in df.columns
        assert "B" in df.columns

    def test_missing_series_returns_nan_column(self, tmp_db):
        upsert_series_meta(tmp_db, "REAL", "Real", "test", "daily")
        upsert_observations(tmp_db, "REAL", _make_obs([1.0, 2.0]))
        df = query_multiple_series(tmp_db, ["REAL", "MISSING"])
        assert "MISSING" in df.columns
        assert df["MISSING"].isna().all()


# ── get_all_series_meta ───────────────────────────────────────────────────────

class TestGetAllSeriesMeta:
    def test_returns_empty_list_on_fresh_db(self, tmp_db):
        result = get_all_series_meta(tmp_db)
        assert result == []

    def test_returns_inserted_meta(self, tmp_db):
        upsert_series_meta(tmp_db, "DGS10", "10Y Treasury", "yield_curve", "daily")
        upsert_series_meta(tmp_db, "CPIAUCSL", "CPI", "inflation", "monthly")
        result = get_all_series_meta(tmp_db)
        ids = [r["series_id"] for r in result]
        assert "DGS10" in ids
        assert "CPIAUCSL" in ids

    def test_returns_list_of_dicts(self, tmp_db):
        upsert_series_meta(tmp_db, "DGS10", "10Y Treasury", "yield_curve", "daily")
        result = get_all_series_meta(tmp_db)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

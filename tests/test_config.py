"""Tests for charlie.config — Settings loading."""
import pytest
import os
import tempfile
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.config import (
    Settings,
    SeriesConfig,
    TickerConfig,
    CalendarRelease,
    SubredditConfig,
    SentimentConfig,
)


# ── Dataclass sanity checks ───────────────────────────────────────────────────

class TestSeriesConfig:
    def test_creation(self):
        sc = SeriesConfig(id="DGS10", name="10Y Treasury", frequency="daily", category="yield_curve")
        assert sc.id == "DGS10"
        assert sc.name == "10Y Treasury"
        assert sc.frequency == "daily"
        assert sc.category == "yield_curve"

    def test_frozen_prevents_mutation(self):
        sc = SeriesConfig(id="DGS10", name="10Y Treasury", frequency="daily", category="yield_curve")
        with pytest.raises(Exception):  # FrozenInstanceError
            sc.id = "DGS2"  # type: ignore


class TestTickerConfig:
    def test_creation(self):
        tc = TickerConfig(symbol="SPY", name="S&P 500", category="indices")
        assert tc.symbol == "SPY"
        assert tc.name == "S&P 500"
        assert tc.category == "indices"

    def test_frozen(self):
        tc = TickerConfig(symbol="SPY", name="S&P 500", category="indices")
        with pytest.raises(Exception):
            tc.symbol = "QQQ"  # type: ignore


class TestCalendarRelease:
    def test_creation(self):
        cr = CalendarRelease(id=10, name="CPI", full_name="Consumer Price Index", importance="high")
        assert cr.id == 10
        assert cr.importance == "high"
        assert cr.fixed_dates == ()

    def test_with_fixed_dates(self):
        cr = CalendarRelease(
            id=0,
            name="Test",
            full_name="Test Release",
            importance="low",
            fixed_dates=("2026-01-15", "2026-02-15"),
        )
        assert len(cr.fixed_dates) == 2


# ── Settings methods ──────────────────────────────────────────────────────────

def _make_settings(**overrides):
    defaults = dict(
        fred_api_key="test_key",
        db_path=Path("/tmp/test.db"),
        series=(
            SeriesConfig(id="DGS10", name="10Y Treasury", frequency="daily", category="yield_curve"),
            SeriesConfig(id="CPIAUCSL", name="CPI", frequency="monthly", category="inflation"),
            SeriesConfig(id="UNRATE", name="Unemployment", frequency="monthly", category="labor"),
        ),
        categories=("yield_curve", "inflation", "labor"),
        tickers=(
            TickerConfig(symbol="SPY", name="S&P 500", category="indices"),
            TickerConfig(symbol="GLD", name="Gold", category="commodities"),
        ),
        ticker_categories=("indices", "commodities"),
        calendar_releases=(
            CalendarRelease(id=10, name="CPI", full_name="Consumer Price Index", importance="high"),
        ),
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestSettingsMethods:
    def test_series_by_category_returns_matching(self):
        s = _make_settings()
        result = s.series_by_category("yield_curve")
        assert len(result) == 1
        assert result[0].id == "DGS10"

    def test_series_by_category_empty_for_unknown(self):
        s = _make_settings()
        result = s.series_by_category("unknown_category")
        assert result == []

    def test_series_by_category_multiple_matches(self):
        s = _make_settings(
            series=(
                SeriesConfig(id="A", name="A", frequency="daily", category="cat1"),
                SeriesConfig(id="B", name="B", frequency="daily", category="cat1"),
                SeriesConfig(id="C", name="C", frequency="daily", category="cat2"),
            )
        )
        result = s.series_by_category("cat1")
        assert len(result) == 2
        ids = [r.id for r in result]
        assert "A" in ids and "B" in ids

    def test_series_by_id_returns_correct(self):
        s = _make_settings()
        result = s.series_by_id("CPIAUCSL")
        assert result is not None
        assert result.name == "CPI"

    def test_series_by_id_returns_none_for_unknown(self):
        s = _make_settings()
        result = s.series_by_id("UNKNOWN_SERIES")
        assert result is None

    def test_tickers_by_category_returns_matching(self):
        s = _make_settings()
        result = s.tickers_by_category("indices")
        assert len(result) == 1
        assert result[0].symbol == "SPY"

    def test_tickers_by_category_empty_for_unknown(self):
        s = _make_settings()
        result = s.tickers_by_category("missing")
        assert result == []

    def test_ticker_by_symbol_returns_correct(self):
        s = _make_settings()
        result = s.ticker_by_symbol("GLD")
        assert result is not None
        assert result.name == "Gold"

    def test_ticker_by_symbol_returns_none_for_unknown(self):
        s = _make_settings()
        result = s.ticker_by_symbol("NONEXISTENT")
        assert result is None


# ── get_settings integration (with real config files) ─────────────────────────

class TestGetSettings:
    def test_loads_series_from_config(self):
        """get_settings should load series from the real config/series.yaml."""
        # Clear lru_cache to get a fresh load
        from charlie.config import get_settings
        get_settings.cache_clear()

        # Set minimal env vars to avoid failures
        os.environ.setdefault("FRED_API_KEY", "test_key")

        try:
            settings = get_settings()
            assert len(settings.series) > 0
        except FileNotFoundError:
            pytest.skip("Config files not available in this context")
        finally:
            get_settings.cache_clear()

    def test_loads_tickers_from_config(self):
        from charlie.config import get_settings
        get_settings.cache_clear()
        os.environ.setdefault("FRED_API_KEY", "test_key")

        try:
            settings = get_settings()
            assert len(settings.tickers) > 0
        except FileNotFoundError:
            pytest.skip("Config files not available in this context")
        finally:
            get_settings.cache_clear()

    def test_loads_calendar_from_config(self):
        from charlie.config import get_settings
        get_settings.cache_clear()
        os.environ.setdefault("FRED_API_KEY", "test_key")

        try:
            settings = get_settings()
            assert len(settings.calendar_releases) > 0
        except FileNotFoundError:
            pytest.skip("Config files not available in this context")
        finally:
            get_settings.cache_clear()

    def test_result_is_cached(self):
        from charlie.config import get_settings
        get_settings.cache_clear()
        os.environ.setdefault("FRED_API_KEY", "test_key")

        try:
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2  # same object from lru_cache
        except FileNotFoundError:
            pytest.skip("Config files not available in this context")
        finally:
            get_settings.cache_clear()

    def test_series_categories_match_loaded_series(self):
        from charlie.config import get_settings
        get_settings.cache_clear()
        os.environ.setdefault("FRED_API_KEY", "test_key")

        try:
            settings = get_settings()
            series_cats = {s.category for s in settings.series}
            for cat in series_cats:
                assert cat in settings.categories
        except FileNotFoundError:
            pytest.skip("Config files not available in this context")
        finally:
            get_settings.cache_clear()

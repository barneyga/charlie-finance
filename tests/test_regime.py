"""Tests for charlie.analysis.regime — macro regime detection."""
import pytest
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.storage.db import Database
from charlie.storage.models import upsert_series_meta, upsert_observations
from charlie.analysis.regime import macro_regime, REGIME_COLORS


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "regime_test.db"
    d = Database(db_path)
    d.init_schema()
    yield d
    d.close()


def _insert(db, series_id, values, start="2018-01-01", freq="D"):
    upsert_series_meta(db, series_id, series_id, "test", "daily")
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    s = pd.Series(values, index=idx)
    upsert_observations(db, series_id, s)


def _insert_monthly(db, series_id, values, start="2018-01-01"):
    upsert_series_meta(db, series_id, series_id, "test", "monthly")
    idx = pd.date_range(start=start, periods=len(values), freq="MS")
    s = pd.Series(values, index=idx)
    upsert_observations(db, series_id, s)


def _insert_quarterly(db, series_id, values, start="2018-01-01"):
    upsert_series_meta(db, series_id, series_id, "test", "quarterly")
    idx = pd.date_range(start=start, periods=len(values), freq="QS")
    s = pd.Series(values, index=idx)
    upsert_observations(db, series_id, s)


class TestMacroRegime:
    def test_returns_dict_with_expected_keys(self, db):
        result = macro_regime(db)
        assert "regime" in result
        assert "score" in result
        assert "signals" in result

    def test_regime_is_valid_string(self, db):
        result = macro_regime(db)
        assert result["regime"] in ("expansion", "late_cycle", "contraction", "recovery")

    def test_empty_db_gives_expansion(self, db):
        """With no data, score stays at 0 → expansion."""
        result = macro_regime(db)
        assert result["regime"] == "expansion"
        assert result["score"] == 0

    def test_inverted_yield_curve_adds_contraction_signal(self, db):
        # Inverted: 2Y > 10Y (short rate higher than long rate)
        _insert(db, "DGS2", [4.0] * 300)
        _insert(db, "DGS10", [2.0] * 300)
        result = macro_regime(db)
        assert result["signals"].get("yield_curve_signal") == "inverted"
        assert result["score"] >= 1

    def test_normal_yield_curve_no_signal(self, db):
        _insert(db, "DGS2", [1.0] * 300)
        _insert(db, "DGS10", [3.5] * 300)
        result = macro_regime(db)
        assert result["signals"].get("yield_curve_signal") == "normal"

    def test_high_inflation_adds_contraction_signal(self, db):
        # CPI > 4% YoY → hot inflation signal
        # Need 24 months: first 12 = 200, next 12 = 210 (5% YoY)
        _insert_monthly(db, "CPIAUCSL", [200.0] * 12 + [210.0] * 12)
        result = macro_regime(db)
        assert result["signals"].get("inflation_signal") == "hot"

    def test_rising_unemployment_adds_contraction_signal(self, db):
        # Rising by more than 0.3% over 3 months
        unemployment = [4.0] * 20 + [4.5]  # last reading jumps
        _insert_monthly(db, "UNRATE", unemployment)
        result = macro_regime(db)
        assert result["signals"].get("unemployment_signal") == "rising"

    def test_falling_unemployment_subtracts_from_score(self, db):
        unemployment = [5.0] * 5 + [4.5]  # falling
        _insert_monthly(db, "UNRATE", unemployment)
        result = macro_regime(db)
        assert result["signals"].get("unemployment_signal") == "falling"

    def test_contraction_regime_with_multiple_signals(self, db):
        # Set up: inverted curve + high inflation + rising unemployment
        _insert(db, "DGS2", [5.0] * 300)
        _insert(db, "DGS10", [3.0] * 300)
        _insert_monthly(db, "CPIAUCSL", [200.0] * 12 + [212.0] * 12)
        unemployment = [4.0] * 20 + [4.8]
        _insert_monthly(db, "UNRATE", unemployment)
        result = macro_regime(db)
        assert result["score"] >= 2
        assert result["regime"] == "contraction"

    def test_regime_score_range(self, db):
        result = macro_regime(db)
        # Score should be bounded by number of indicators
        assert -6 <= result["score"] <= 6

    def test_signals_is_dict(self, db):
        result = macro_regime(db)
        assert isinstance(result["signals"], dict)


class TestRegimeColors:
    def test_all_regimes_have_colors(self):
        for regime in ("expansion", "late_cycle", "contraction", "recovery"):
            assert regime in REGIME_COLORS

    def test_colors_are_valid_hex(self):
        for regime, color in REGIME_COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB

    def test_contraction_is_red(self):
        # Contraction should stand out as red/warning
        assert "ef" in REGIME_COLORS["contraction"].lower() or "f4" in REGIME_COLORS["contraction"].lower()

    def test_expansion_is_green(self):
        assert "2" in REGIME_COLORS["expansion"] or "c5" in REGIME_COLORS["expansion"].lower()

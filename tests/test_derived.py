"""Tests for charlie.analysis.derived — derived macro indicators."""
import pytest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.storage.db import Database
from charlie.storage.models import upsert_series_meta, upsert_observations
from charlie.analysis.derived import (
    yield_curve_spread,
    yield_curve_shape,
    real_rate,
    cpi_yoy,
    payrolls_mom_change,
    m2_yoy,
    hy_ig_spread,
    credit_impulse,
    fed_balance_sheet_change,
    gold_silver_ratio,
    stock_bond_correlation,
    spy_rsp_spread,
    sector_returns,
)


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    d = Database(db_path)
    d.init_schema()
    yield d
    d.close()


def _insert(db, series_id, values, start="2020-01-01", freq="D", category="test"):
    upsert_series_meta(db, series_id, series_id, category, "daily")
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    s = pd.Series(values, index=idx)
    upsert_observations(db, series_id, s)


def _insert_monthly(db, series_id, values, start="2010-01-01"):
    upsert_series_meta(db, series_id, series_id, "test", "monthly")
    idx = pd.date_range(start=start, periods=len(values), freq="MS")
    s = pd.Series(values, index=idx)
    upsert_observations(db, series_id, s)


# ── yield_curve_spread ────────────────────────────────────────────────────────

class TestYieldCurveSpread:
    def test_basic_spread(self, db):
        _insert(db, "DGS2", [1.0] * 10)
        _insert(db, "DGS10", [3.0] * 10)
        spread = yield_curve_spread(db, "DGS2", "DGS10")
        assert ((spread - 2.0).abs() < 1e-6).all()

    def test_inverted_spread_is_negative(self, db):
        _insert(db, "DGS2", [4.0] * 10)
        _insert(db, "DGS10", [2.0] * 10)
        spread = yield_curve_spread(db, "DGS2", "DGS10")
        assert (spread < 0).all()

    def test_spread_name(self, db):
        _insert(db, "DGS2", [1.0] * 5)
        _insert(db, "DGS10", [3.0] * 5)
        spread = yield_curve_spread(db, "DGS2", "DGS10")
        assert "DGS10" in spread.name and "DGS2" in spread.name

    def test_spread_handles_missing_data(self, db):
        # Only one series — should return empty or NaN-only
        _insert(db, "DGS2", [1.0] * 5)
        spread = yield_curve_spread(db, "DGS2", "DGS10")
        # Either empty or all-NaN
        assert spread.empty or spread.isna().all()


# ── yield_curve_shape ─────────────────────────────────────────────────────────

class TestYieldCurveShape:
    def _insert_all_tenors(self, db, base_value=2.0):
        tenors = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
        for i, t in enumerate(tenors):
            _insert(db, t, [base_value + i * 0.1] * 10)

    def test_returns_series_with_correct_labels(self, db):
        self._insert_all_tenors(db)
        shape = yield_curve_shape(db)
        assert not shape.empty
        assert "1M" in shape.index
        assert "30Y" in shape.index

    def test_shape_is_normal_upward_sloping(self, db):
        self._insert_all_tenors(db)
        shape = yield_curve_shape(db)
        # Long end should be higher than short end
        assert shape["30Y"] > shape["1M"]

    def test_empty_db_returns_empty_series(self, db):
        shape = yield_curve_shape(db)
        assert shape.empty


# ── real_rate ─────────────────────────────────────────────────────────────────

class TestRealRate:
    def test_basic_real_rate(self, db):
        _insert(db, "DGS10", [4.5] * 10)
        _insert(db, "T10YIE", [2.5] * 10)
        rr = real_rate(db)
        assert ((rr - 2.0).abs() < 1e-6).all()

    def test_negative_real_rate(self, db):
        _insert(db, "DGS10", [1.0] * 10)
        _insert(db, "T10YIE", [3.0] * 10)
        rr = real_rate(db)
        assert (rr < 0).all()

    def test_name_is_real_rate_10y(self, db):
        _insert(db, "DGS10", [3.0] * 5)
        _insert(db, "T10YIE", [2.0] * 5)
        rr = real_rate(db)
        assert rr.name == "real_rate_10y"


# ── cpi_yoy ───────────────────────────────────────────────────────────────────

class TestCpiYoy:
    def test_doubling_gives_100_pct(self, db):
        # 12 months of 100, then 12 months of 200 → 100% YoY
        values = [100.0] * 12 + [200.0] * 12
        _insert_monthly(db, "CPIAUCSL", values)
        yoy = cpi_yoy(db)
        assert not yoy.empty
        assert yoy.iloc[-1] == pytest.approx(100.0, abs=0.1)

    def test_stable_cpi_gives_zero_yoy(self, db):
        values = [200.0] * 24
        _insert_monthly(db, "CPIAUCSL", values)
        yoy = cpi_yoy(db)
        clean = yoy.dropna()
        assert (clean.abs() < 0.001).all()

    def test_name_contains_yoy(self, db):
        _insert_monthly(db, "CPIAUCSL", [200.0] * 24)
        yoy = cpi_yoy(db)
        assert "yoy" in yoy.name


# ── payrolls_mom_change ───────────────────────────────────────────────────────

class TestPayrollsMomChange:
    def test_basic_mom(self, db):
        values = [140_000, 140_200, 140_500]
        _insert_monthly(db, "PAYEMS", values)
        change = payrolls_mom_change(db)
        clean = change.dropna()
        assert not clean.empty
        assert clean.iloc[-1] == pytest.approx(300.0)

    def test_name_is_payrolls_mom(self, db):
        _insert_monthly(db, "PAYEMS", [100.0] * 5)
        change = payrolls_mom_change(db)
        assert change.name == "payrolls_mom"


# ── m2_yoy ────────────────────────────────────────────────────────────────────

class TestM2Yoy:
    def test_stable_m2_gives_zero(self, db):
        _insert_monthly(db, "M2SL", [20000.0] * 24)
        yoy = m2_yoy(db)
        assert (yoy.dropna().abs() < 0.001).all()

    def test_growing_m2_gives_positive(self, db):
        base = 20000.0
        values = [base] * 12 + [base * 1.10] * 12
        _insert_monthly(db, "M2SL", values)
        yoy = m2_yoy(db)
        assert yoy.iloc[-1] == pytest.approx(10.0, abs=0.1)


# ── hy_ig_spread ──────────────────────────────────────────────────────────────

class TestHyIgSpread:
    def test_spread_calculation(self, db):
        _insert(db, "BAMLH0A0HYM2", [5.0] * 10)
        _insert(db, "BAMLC0A0CM", [2.0] * 10)
        spread = hy_ig_spread(db)
        assert ((spread - 3.0).abs() < 1e-6).all()

    def test_spread_name(self, db):
        _insert(db, "BAMLH0A0HYM2", [5.0] * 5)
        _insert(db, "BAMLC0A0CM", [2.0] * 5)
        spread = hy_ig_spread(db)
        assert spread.name == "hy_ig_spread"


# ── gold_silver_ratio ─────────────────────────────────────────────────────────

class TestGoldSilverRatio:
    def test_ratio_calculation(self, db):
        _insert(db, "GLD", [200.0] * 10)
        _insert(db, "SLV", [20.0] * 10)
        ratio = gold_silver_ratio(db)
        assert ((ratio - 10.0).abs() < 1e-6).all()

    def test_rising_ratio_is_risk_off(self, db):
        # Rising GLD relative to SLV → risk-off signal
        gld_vals = list(range(100, 110))
        slv_vals = [10.0] * 10
        _insert(db, "GLD", gld_vals)
        _insert(db, "SLV", slv_vals)
        ratio = gold_silver_ratio(db)
        assert ratio.is_monotonic_increasing

    def test_name_is_gold_silver_ratio(self, db):
        _insert(db, "GLD", [200.0] * 5)
        _insert(db, "SLV", [20.0] * 5)
        ratio = gold_silver_ratio(db)
        assert ratio.name == "gold_silver_ratio"


# ── stock_bond_correlation ────────────────────────────────────────────────────

class TestStockBondCorrelation:
    def test_perfectly_correlated_returns_one(self, db):
        # Same values → correlation of 1
        values = [100 + i * 0.5 for i in range(200)]
        _insert(db, "SPY", values)
        _insert(db, "TLT", values)
        corr = stock_bond_correlation(db, window=63)
        # Perfect positive correlation
        assert corr.dropna().iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_inversely_correlated_returns_neg_one(self, db):
        # Build series whose daily returns are perfectly inversely correlated
        np.random.seed(7)
        shocks = np.random.randn(200)
        spy_prices = np.cumprod(1 + shocks * 0.01) * 100
        tlt_prices = np.cumprod(1 - shocks * 0.01) * 100  # opposite shocks
        _insert(db, "SPY", spy_prices.tolist())
        _insert(db, "TLT", tlt_prices.tolist())
        corr = stock_bond_correlation(db, window=63)
        assert corr.dropna().iloc[-1] == pytest.approx(-1.0, abs=0.01)

    def test_name_is_spy_tlt_correlation(self, db):
        values = [100 + i for i in range(200)]
        _insert(db, "SPY", values)
        _insert(db, "TLT", values)
        corr = stock_bond_correlation(db)
        assert corr.name == "spy_tlt_correlation"

    def test_empty_when_no_data(self, db):
        corr = stock_bond_correlation(db)
        assert corr.empty


# ── spy_rsp_spread ────────────────────────────────────────────────────────────

class TestSpyRspSpread:
    def test_equal_performance_gives_100(self, db):
        # SPY and RSP grow at same rate → normalized ratio = 100
        values = [100 * (1.001 ** i) for i in range(50)]
        _insert(db, "SPY", values)
        _insert(db, "RSP", values)
        ratio = spy_rsp_spread(db)
        assert ((ratio.dropna() - 100.0).abs() < 0.01).all()

    def test_spy_outperforming_gives_ratio_above_100(self, db):
        spy = [100 * (1.01 ** i) for i in range(50)]
        rsp = [100 * (1.001 ** i) for i in range(50)]
        _insert(db, "SPY", spy)
        _insert(db, "RSP", rsp)
        ratio = spy_rsp_spread(db)
        assert ratio.dropna().iloc[-1] > 100

    def test_name_is_spy_rsp_ratio(self, db):
        _insert(db, "SPY", [100.0] * 10)
        _insert(db, "RSP", [100.0] * 10)
        ratio = spy_rsp_spread(db)
        assert ratio.name == "spy_rsp_ratio"

    def test_empty_returns_empty_series(self, db):
        ratio = spy_rsp_spread(db)
        assert ratio.empty


# ── sector_returns ────────────────────────────────────────────────────────────

class TestSectorReturns:
    def _insert_sector(self, db, symbol, n=100):
        values = [100 * (1.001 ** i) for i in range(n)]
        _insert(db, symbol, values, start="2023-01-01")

    def test_returns_dataframe(self, db):
        self._insert_sector(db, "XLF")
        result = sector_returns(db)
        assert isinstance(result, pd.DataFrame)

    def test_columns_include_periods(self, db):
        self._insert_sector(db, "XLF")
        result = sector_returns(db)
        if not result.empty:
            for col in ["1W", "1M"]:
                assert col in result.columns

    def test_empty_db_returns_empty_dataframe(self, db):
        result = sector_returns(db)
        assert result.empty

    def test_multiple_sectors(self, db):
        for sym in ["XLF", "XLE", "XLK"]:
            self._insert_sector(db, sym)
        result = sector_returns(db)
        assert len(result) == 3

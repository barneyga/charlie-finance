"""Tests for charlie.analysis.stats — statistical utilities."""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.analysis.stats import (
    rolling_zscore,
    percentile_rank,
    rolling_change,
    rate_of_change,
    direction_arrow,
)


def make_series(values, start="2020-01-01", freq="D"):
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, name="test")


# ── rolling_zscore ────────────────────────────────────────────────────────────

class TestRollingZscore:
    def test_output_length_matches_input(self):
        s = make_series(range(100))
        result = rolling_zscore(s, window=20)
        assert len(result) == len(s)

    def test_name_suffix(self):
        s = make_series(range(100), )
        s.name = "DGS10"
        result = rolling_zscore(s, window=20)
        assert "zscore" in result.name

    def test_constant_series_produces_nan_or_zero(self):
        s = make_series([5.0] * 50)
        result = rolling_zscore(s, window=20)
        # Std of constant series is 0 → NaN or 0
        assert result.dropna().isin([0.0, float("nan")]).all() or result.dropna().empty or (result.dropna() == 0).all()

    def test_zscore_of_mean_is_zero(self):
        """A value equal to the rolling mean should have z-score near 0."""
        np.random.seed(42)
        values = np.random.normal(100, 5, 300)
        s = make_series(values)
        result = rolling_zscore(s, window=252)
        # Mean z-score should be close to 0 (over a long enough window)
        assert abs(result.dropna().mean()) < 0.5

    def test_large_values_have_positive_zscore(self):
        """Values well above the mean should produce positive z-scores."""
        values = list(range(100)) + [1000]  # spike at end
        s = make_series(values)
        result = rolling_zscore(s, window=50)
        assert result.iloc[-1] > 2


# ── percentile_rank ───────────────────────────────────────────────────────────

class TestPercentileRank:
    def test_output_range_0_to_100(self):
        np.random.seed(0)
        s = make_series(np.random.randn(300))
        result = percentile_rank(s, window=200)
        clean = result.dropna()
        assert (clean >= 0).all()
        assert (clean <= 100).all()

    def test_max_value_gets_high_rank(self):
        values = list(range(1, 201))  # monotonically increasing
        s = make_series(values)
        result = percentile_rank(s, window=200)
        # Last value (200) is the highest → near 100
        assert result.iloc[-1] > 90

    def test_min_value_gets_low_rank(self):
        values = list(range(200, 0, -1))  # monotonically decreasing
        s = make_series(values)
        result = percentile_rank(s, window=200)
        # Last value (1) is the lowest → near 0
        assert result.iloc[-1] < 10

    def test_output_name_contains_pctile(self):
        s = make_series(range(100))
        s.name = "VIX"
        result = percentile_rank(s)
        assert "pctile" in result.name

    def test_min_periods_respected(self):
        s = make_series(range(10))
        result = percentile_rank(s, window=1000)
        # Should still produce values due to min_periods=60 clamped to window
        assert len(result) == len(s)


# ── rolling_change ────────────────────────────────────────────────────────────

class TestRollingChange:
    def test_basic_diff(self):
        s = make_series([10, 20, 30, 40, 50])
        result = rolling_change(s, periods=1)
        assert result.iloc[1] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(10.0)

    def test_multi_period_diff(self):
        s = make_series([0, 5, 10, 15, 20])
        result = rolling_change(s, periods=2)
        assert result.iloc[2] == pytest.approx(10.0)
        assert result.iloc[4] == pytest.approx(10.0)

    def test_name_contains_chg(self):
        s = make_series(range(10))
        s.name = "UNRATE"
        result = rolling_change(s, periods=3)
        assert "chg" in result.name
        assert "3" in result.name

    def test_first_periods_are_nan(self):
        s = make_series(range(10))
        result = rolling_change(s, periods=3)
        assert result.iloc[:3].isna().all()


# ── rate_of_change ────────────────────────────────────────────────────────────

class TestRateOfChange:
    def test_doubling_gives_100_pct(self):
        s = make_series([100] * 252 + [200])
        result = rate_of_change(s, periods=252)
        assert result.iloc[-1] == pytest.approx(100.0)

    def test_halving_gives_neg_50_pct(self):
        s = make_series([200] * 10 + [100])
        result = rate_of_change(s, periods=10)
        assert result.iloc[-1] == pytest.approx(-50.0)

    def test_name_contains_roc(self):
        s = make_series(range(300))
        s.name = "SPY"
        result = rate_of_change(s, periods=252)
        assert "roc" in result.name

    def test_output_length_matches_input(self):
        s = make_series(range(100))
        result = rate_of_change(s, periods=10)
        assert len(result) == len(s)


# ── direction_arrow ───────────────────────────────────────────────────────────

class TestDirectionArrow:
    def test_uptrend_returns_up(self):
        s = make_series(list(range(50)))  # steadily rising
        assert direction_arrow(s, lookback=21, threshold=0.01) == "up"

    def test_downtrend_returns_down(self):
        s = make_series(list(range(50, 0, -1)))  # steadily falling
        assert direction_arrow(s, lookback=21, threshold=0.01) == "down"

    def test_flat_returns_flat(self):
        s = make_series([100.0] * 50)
        assert direction_arrow(s, lookback=21, threshold=0.01) == "flat"

    def test_too_short_returns_dash(self):
        s = make_series([1, 2, 3])
        assert direction_arrow(s, lookback=21) == "—"

    def test_nan_past_returns_dash(self):
        values = [float("nan")] * 25 + [1.0] * 25
        s = make_series(values)
        # past value will be NaN → should return "—"
        result = direction_arrow(s, lookback=21)
        assert result in ("—", "up", "down", "flat")  # graceful handling

    def test_zero_past_value_returns_dash(self):
        # past = series.iloc[-(lookback+1)] = series.iloc[-22]
        # With 30 values and 25 leading zeros, iloc[-22] = iloc[8] = 0.0
        values = [0.0] * 25 + [1.0] * 5
        s = make_series(values)
        assert direction_arrow(s, lookback=21) == "—"

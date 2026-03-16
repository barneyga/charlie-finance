"""Tests for charlie.analysis.composite — Fear/Greed composite score."""
import pytest
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.storage.db import Database
from charlie.storage.models import upsert_series_meta, upsert_observations
from charlie.analysis.composite import fear_greed_score, _label_and_color, _compute_component


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "composite_test.db"
    d = Database(db_path)
    d.init_schema()
    yield d
    d.close()


def _insert(db, series_id, values, start="2015-01-01", freq="D"):
    upsert_series_meta(db, series_id, series_id, "test", "daily")
    idx = pd.date_range(start=start, periods=len(values), freq=freq)
    s = pd.Series(values, index=idx)
    upsert_observations(db, series_id, s)


def _make_series(n=500, value=50.0):
    idx = pd.date_range(start="2015-01-01", periods=n, freq="D")
    return pd.Series([value] * n, index=idx)


# ── _label_and_color ──────────────────────────────────────────────────────────

class TestLabelAndColor:
    def test_extreme_greed_at_low_score(self):
        label, color = _label_and_color(10)
        assert label == "Extreme Greed"

    def test_extreme_fear_at_high_score(self):
        label, color = _label_and_color(95)
        assert label == "Extreme Fear"

    def test_neutral_at_midpoint(self):
        label, color = _label_and_color(50)
        assert label == "Neutral"

    def test_greed_band(self):
        label, _ = _label_and_color(30)
        assert label == "Greed"

    def test_fear_band(self):
        label, _ = _label_and_color(70)
        assert label == "Fear"

    def test_all_labels_return_hex_color(self):
        for score in [10, 30, 50, 70, 90]:
            _, color = _label_and_color(score)
            assert color.startswith("#")
            assert len(color) == 7

    def test_exact_threshold_boundary(self):
        # score == 20 should give "Extreme Greed" (threshold ≤ 20)
        label, _ = _label_and_color(20)
        assert label == "Extreme Greed"
        # score == 21 gives "Greed"
        label, _ = _label_and_color(21)
        assert label == "Greed"


# ── _compute_component ────────────────────────────────────────────────────────

class TestComputeComponent:
    def test_output_range_0_to_100(self):
        np.random.seed(42)
        s = pd.Series(np.random.randn(500), index=pd.date_range("2015-01-01", periods=500))
        result = _compute_component(s)
        clean = result.dropna()
        assert (clean >= 0).all()
        assert (clean <= 100).all()

    def test_invert_flips_values(self):
        np.random.seed(0)
        s = pd.Series(np.random.randn(500), index=pd.date_range("2015-01-01", periods=500))
        normal = _compute_component(s)
        inverted = _compute_component(s, invert=True)
        # Inverted + normal should sum to ~100
        diff = (normal + inverted).dropna()
        assert ((diff - 100.0).abs() <= 1.0).all()


# ── fear_greed_score ──────────────────────────────────────────────────────────

class TestFearGreedScore:
    def test_empty_db_returns_neutral(self, db):
        result = fear_greed_score(db)
        assert result["score"] == 50.0
        assert result["label"] == "Neutral"
        assert result["color"] == "#eab308"
        assert result["components"] == {}
        assert result["history"].empty

    def test_returns_dict_with_expected_keys(self, db):
        result = fear_greed_score(db)
        assert "score" in result
        assert "label" in result
        assert "color" in result
        assert "components" in result
        assert "history" in result

    def test_score_in_valid_range(self, db):
        # Insert enough data for VIX component
        np.random.seed(42)
        vix_vals = np.random.uniform(10, 40, 400)
        _insert(db, "VIXCLS", vix_vals)
        result = fear_greed_score(db)
        assert 0 <= result["score"] <= 100

    def test_vix_component_present_when_data_available(self, db):
        np.random.seed(1)
        vix_vals = np.random.uniform(10, 40, 400)
        _insert(db, "VIXCLS", vix_vals)
        result = fear_greed_score(db)
        assert "VIX" in result["components"]

    def test_component_has_score_raw_value_description(self, db):
        np.random.seed(1)
        vix_vals = np.random.uniform(10, 40, 400)
        _insert(db, "VIXCLS", vix_vals)
        result = fear_greed_score(db)
        comp = result["components"]["VIX"]
        assert "score" in comp
        assert "raw_value" in comp
        assert "description" in comp

    def test_history_is_series(self, db):
        np.random.seed(2)
        vix_vals = np.random.uniform(10, 40, 400)
        _insert(db, "VIXCLS", vix_vals)
        result = fear_greed_score(db)
        assert isinstance(result["history"], pd.Series)

    def test_high_vix_environment_gives_fear_label(self, db):
        """When VIX is persistently at its highest recorded levels, score should be high (fear)."""
        # First 300 days: low VIX (calm)
        # Last 100 days: very high VIX (panic)
        low_vix = [12.0] * 300
        high_vix = [80.0] * 100
        _insert(db, "VIXCLS", low_vix + high_vix)
        result = fear_greed_score(db)
        # VIX component score should reflect elevated fear
        vix_comp_score = result["components"]["VIX"]["score"]
        assert vix_comp_score > 50  # above midpoint → fear territory

    def test_multiple_components_compute_average(self, db):
        """With multiple components, score = average."""
        np.random.seed(3)
        n = 400

        # Insert VIX
        _insert(db, "VIXCLS", np.random.uniform(10, 40, n))
        # Insert credit spread data
        _insert(db, "BAMLH0A0HYM2", np.random.uniform(3, 8, n))
        _insert(db, "BAMLC0A0CM", np.random.uniform(0.5, 2, n))
        # Insert SPY and RSP for breadth
        _insert(db, "SPY", np.cumprod(1 + np.random.normal(0.0003, 0.01, n)) * 100)
        _insert(db, "RSP", np.cumprod(1 + np.random.normal(0.0003, 0.01, n)) * 50)

        result = fear_greed_score(db)
        # Multiple components should be present
        assert len(result["components"]) >= 2

        # Score should match simple average of component scores
        component_scores = [c["score"] for c in result["components"].values()]
        expected = sum(component_scores) / len(component_scores)
        assert result["score"] == pytest.approx(expected, abs=0.01)

    def test_label_matches_score(self, db):
        """Label should match the score bucket."""
        np.random.seed(4)
        _insert(db, "VIXCLS", np.random.uniform(10, 40, 400))
        result = fear_greed_score(db)
        score = result["score"]
        label = result["label"]

        if score <= 20:
            assert label == "Extreme Greed"
        elif score <= 40:
            assert label == "Greed"
        elif score <= 60:
            assert label == "Neutral"
        elif score <= 80:
            assert label == "Fear"
        else:
            assert label == "Extreme Fear"

    def test_color_is_valid_hex(self, db):
        result = fear_greed_score(db)
        assert result["color"].startswith("#")
        assert len(result["color"]) == 7

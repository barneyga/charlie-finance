"""Composite Fear/Greed score — real-time market regime gauge."""
import pandas as pd

from charlie.storage.db import Database
from charlie.storage.models import query_series
from charlie.analysis.derived import (
    hy_ig_spread, spy_rsp_spread, stock_bond_correlation,
    gold_silver_ratio, yield_curve_spread,
)
from charlie.analysis.stats import percentile_rank


LABELS = [
    (20, "Extreme Greed", "#22c55e"),
    (40, "Greed", "#86efac"),
    (60, "Neutral", "#eab308"),
    (80, "Fear", "#f97316"),
    (100, "Extreme Fear", "#ef4444"),
]


def _label_and_color(score: float) -> tuple[str, str]:
    for threshold, label, color in LABELS:
        if score <= threshold:
            return label, color
    return "Extreme Fear", "#ef4444"


def _compute_component(
    series: pd.Series, window: int = 1260, invert: bool = False
) -> pd.Series:
    """Percentile-rank a series (0-100). If invert, flip so low raw = high score."""
    pct = percentile_rank(series, window=window)
    if invert:
        pct = 100 - pct
    return pct


def fear_greed_score(db: Database) -> dict:
    """Compute composite fear/greed score from 9 market-based components.

    Returns dict with:
        score: float (0-100, 0=extreme greed, 100=extreme fear)
        label: str
        color: str (hex)
        components: dict of component name → {score, raw_value, description}
        history: pd.Series of rolling composite over time
    """
    components = {}
    component_series = {}

    # 1. VIX Level — high VIX = fear
    try:
        vix = query_series(db, "VIXCLS")
        if len(vix) >= 60:
            pct = _compute_component(vix)
            component_series["VIX"] = pct
            components["VIX"] = {
                "score": round(pct.iloc[-1], 1),
                "raw_value": round(vix.iloc[-1], 2),
                "description": "Volatility index level",
            }
    except Exception:
        pass

    # 2. Credit Stress — wide HY-IG spread = fear
    try:
        hi = hy_ig_spread(db)
        if len(hi) >= 60:
            pct = _compute_component(hi)
            component_series["Credit Stress"] = pct
            components["Credit Stress"] = {
                "score": round(pct.iloc[-1], 1),
                "raw_value": round(hi.iloc[-1], 2),
                "description": "HY-IG OAS spread",
            }
    except Exception:
        pass

    # 3. Breadth — high SPY/RSP ratio = concentration = fear
    try:
        breadth = spy_rsp_spread(db)
        if len(breadth) >= 60:
            pct = _compute_component(breadth)
            component_series["Breadth"] = pct
            components["Breadth"] = {
                "score": round(pct.iloc[-1], 1),
                "raw_value": round(breadth.iloc[-1], 2),
                "description": "SPY/RSP concentration ratio",
            }
    except Exception:
        pass

    # 4. Stock/Bond Correlation — positive correlation = fear (no diversification)
    try:
        corr = stock_bond_correlation(db)
        if len(corr) >= 60:
            pct = _compute_component(corr)
            component_series["Correlation"] = pct
            components["Correlation"] = {
                "score": round(pct.iloc[-1], 1),
                "raw_value": round(corr.iloc[-1], 3),
                "description": "SPY-TLT 63d correlation",
            }
    except Exception:
        pass

    # 5. Safe Haven Demand — high gold/silver ratio = fear
    try:
        gsr = gold_silver_ratio(db)
        if len(gsr) >= 60:
            pct = _compute_component(gsr)
            component_series["Safe Haven"] = pct
            components["Safe Haven"] = {
                "score": round(pct.iloc[-1], 1),
                "raw_value": round(gsr.iloc[-1], 2),
                "description": "Gold/Silver ratio",
            }
    except Exception:
        pass

    # 6. Yield Curve — low/negative spread = fear (inverted)
    try:
        yc = yield_curve_spread(db, "DGS2", "DGS10")
        if len(yc) >= 60:
            pct = _compute_component(yc, invert=True)
            component_series["Yield Curve"] = pct
            components["Yield Curve"] = {
                "score": round(pct.iloc[-1], 1),
                "raw_value": round(yc.iloc[-1], 3),
                "description": "10Y-2Y spread (inverted)",
            }
    except Exception:
        pass

    # 7. Market Momentum — price below trend = fear
    try:
        spy = query_series(db, "SPY")
        if len(spy) >= 200:
            ma50 = spy.rolling(50).mean()
            ma200 = spy.rolling(200).mean()
            momentum = ma50 / ma200 * 100  # >100 = above trend
            momentum.name = "spy_momentum"
            # Invert: low momentum ratio = high fear score
            pct = _compute_component(momentum, invert=True)
            component_series["Momentum"] = pct
            components["Momentum"] = {
                "score": round(pct.iloc[-1], 1),
                "raw_value": round(momentum.iloc[-1], 2),
                "description": "SPY 50d/200d MA ratio (inverted)",
            }
    except Exception:
        pass

    # 8. Put/Call Ratio — high ratio = fear (defensive hedging)
    try:
        pcr = query_series(db, "PCR_EQUITY")
        if len(pcr) >= 60:
            pcr.name = "put_call_ratio"
            # Higher put/call = more fear (no inversion needed)
            pct = _compute_component(pcr, invert=False)
            component_series["Put/Call"] = pct
            components["Put/Call"] = {
                "score": round(pct.iloc[-1], 1),
                "raw_value": round(pcr.iloc[-1], 3),
                "description": "CBOE equity put/call ratio",
            }
    except Exception:
        pass

    # 9. COT S&P 500 Positioning — high net long = complacency (greed)
    try:
        cot_es = query_series(db, "COT_ES_PCT")
        if len(cot_es) >= 26:
            cot_es.name = "cot_es_pct"
            # High net long = greed → invert so high long = high fear score
            pct = _compute_component(cot_es, invert=True)
            component_series["COT Positioning"] = pct
            components["COT Positioning"] = {
                "score": round(pct.iloc[-1], 1),
                "raw_value": round(cot_es.iloc[-1], 1),
                "description": "S&P 500 futures net speculator % (inverted)",
            }
    except Exception:
        pass

    # Composite score = simple average of all available components
    if not components:
        return {
            "score": 50.0,
            "label": "Neutral",
            "color": "#eab308",
            "components": {},
            "history": pd.Series(dtype=float),
        }

    current_score = sum(c["score"] for c in components.values()) / len(components)
    label, color = _label_and_color(current_score)

    # Build history from aligned component series
    if component_series:
        aligned = pd.DataFrame(component_series).dropna()
        history = aligned.mean(axis=1)
        history.name = "fear_greed"
    else:
        history = pd.Series(dtype=float)

    return {
        "score": round(current_score, 1),
        "label": label,
        "color": color,
        "components": components,
        "history": history,
    }

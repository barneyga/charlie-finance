"""Simple macro regime detection based on indicator scoring."""
import pandas as pd

from charlie.storage.db import Database
from charlie.storage.models import query_series
from charlie.analysis.derived import yield_curve_spread, cpi_yoy
from charlie.analysis.stats import rolling_zscore


def macro_regime(db: Database) -> dict:
    """
    Classify the current macro regime based on a scoring system.

    Returns dict with:
        regime: str - "expansion", "late_cycle", "contraction", "recovery"
        score: int - raw score (-4 to +4, negative = contractionary)
        signals: dict - individual signal values
    """
    signals = {}
    contraction_score = 0

    # 1. Yield curve: inverted 10Y-2Y spread
    try:
        spread = yield_curve_spread(db, "DGS2", "DGS10")
        if len(spread) >= 63:  # ~3 months of trading days
            recent_avg = spread.iloc[-63:].mean()
            signals["yield_curve_avg_3m"] = round(recent_avg, 3)
            if recent_avg < 0:
                contraction_score += 1
                signals["yield_curve_signal"] = "inverted"
            else:
                signals["yield_curve_signal"] = "normal"
    except Exception:
        signals["yield_curve_signal"] = "unavailable"

    # 2. Credit spreads z-score
    try:
        baa = query_series(db, "BAA10Y")
        if len(baa) >= 252:
            z = rolling_zscore(baa, 252)
            current_z = z.iloc[-1]
            signals["credit_spread_zscore"] = round(current_z, 2)
            if current_z > 1.5:
                contraction_score += 1
                signals["credit_signal"] = "stressed"
            elif current_z > 1.0:
                signals["credit_signal"] = "elevated"
            else:
                signals["credit_signal"] = "normal"
    except Exception:
        signals["credit_signal"] = "unavailable"

    # 3. Unemployment trend
    try:
        unrate = query_series(db, "UNRATE")
        if len(unrate) >= 4:
            change_3m = unrate.iloc[-1] - unrate.iloc[-4]
            signals["unemployment_3m_change"] = round(change_3m, 2)
            if change_3m > 0.3:
                contraction_score += 1
                signals["unemployment_signal"] = "rising"
            elif change_3m < -0.2:
                contraction_score -= 1
                signals["unemployment_signal"] = "falling"
            else:
                signals["unemployment_signal"] = "stable"
    except Exception:
        signals["unemployment_signal"] = "unavailable"

    # 4. Inflation
    try:
        cpi = cpi_yoy(db)
        if len(cpi) > 0:
            current_cpi = cpi.iloc[-1]
            signals["cpi_yoy"] = round(current_cpi, 2)
            if current_cpi > 4.0:
                contraction_score += 1  # late cycle / tightening pressure
                signals["inflation_signal"] = "hot"
            elif current_cpi < 1.5:
                contraction_score -= 1
                signals["inflation_signal"] = "cool"
            else:
                signals["inflation_signal"] = "moderate"
    except Exception:
        signals["inflation_signal"] = "unavailable"

    # 5. HY OAS z-score
    try:
        hy_oas = query_series(db, "BAMLH0A0HYM2")
        if len(hy_oas) >= 252:
            z = rolling_zscore(hy_oas, 252)
            current_z = z.iloc[-1]
            signals["hy_oas_zscore"] = round(current_z, 2)
            signals["hy_oas_level"] = round(hy_oas.iloc[-1], 2)
            if current_z > 1.5:
                contraction_score += 1
                signals["hy_oas_signal"] = "stressed"
            elif current_z > 1.0:
                signals["hy_oas_signal"] = "elevated"
            else:
                signals["hy_oas_signal"] = "normal"
    except Exception:
        signals["hy_oas_signal"] = "unavailable"

    # 6. Loan delinquency trend
    try:
        delinq = query_series(db, "DRALACBS")
        if len(delinq) >= 5:
            change_1y = delinq.iloc[-1] - delinq.iloc[-5]  # 4 quarters back
            signals["delinquency_rate"] = round(delinq.iloc[-1], 2)
            signals["delinquency_1y_change"] = round(change_1y, 2)
            if change_1y > 0.3:
                contraction_score += 1
                signals["delinquency_signal"] = "rising"
            elif change_1y < -0.2:
                contraction_score -= 1
                signals["delinquency_signal"] = "improving"
            else:
                signals["delinquency_signal"] = "stable"
    except Exception:
        signals["delinquency_signal"] = "unavailable"

    # Classify regime
    if contraction_score >= 2:
        regime = "contraction"
    elif contraction_score == 1:
        regime = "late_cycle"
    elif contraction_score <= -2:
        regime = "recovery"
    else:
        regime = "expansion"

    return {
        "regime": regime,
        "score": contraction_score,
        "signals": signals,
    }


REGIME_COLORS = {
    "expansion": "#22c55e",    # green
    "late_cycle": "#eab308",   # yellow
    "contraction": "#ef4444",  # red
    "recovery": "#3b82f6",     # blue
}

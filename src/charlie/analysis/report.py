"""Weekly macro report generator — template-based narrative from existing analysis."""
from datetime import datetime

import pandas as pd

from charlie.storage.db import Database
from charlie.storage.models import query_series, query_multiple_series
from charlie.analysis.regime import macro_regime
from charlie.analysis.composite import fear_greed_score
from charlie.analysis.derived import (
    yield_curve_spread, real_rate, cpi_yoy, payrolls_mom_change,
    hy_ig_spread, credit_impulse, gold_silver_ratio, oil_gold_ratio,
    spy_rsp_spread, sector_returns, cot_summary_table, m2_yoy,
)
from charlie.analysis.stats import rolling_zscore, direction_arrow


# Thresholds for alert detection (mirrors dashboard _THRESHOLDS)
_ALERT_THRESHOLDS = {
    "vix": (25, "VIX above 25 — elevated volatility"),
    "hy_oas": (600, "HY OAS above 600 bps — credit stress"),
    "cpi": (4, "CPI YoY above 4% — high inflation"),
    "unemployment": (6, "Unemployment above 6%"),
    "put_call": (1.0, "Put/Call ratio above 1.0 — heavy hedging"),
}


def _wow(series: pd.Series, periods: int = 5) -> dict | None:
    """Week-over-week change for a series."""
    if len(series) < periods + 1:
        return None
    current = series.iloc[-1]
    prev = series.iloc[-(periods + 1)]
    change = current - prev
    pct = (change / abs(prev) * 100) if prev != 0 else 0
    return {"current": current, "prev": prev, "change": change, "pct": pct}


def _arrow(val: float) -> str:
    if val > 0.01:
        return "↑"
    elif val < -0.01:
        return "↓"
    return "→"


def generate_weekly_report(db: Database, fred_api_key: str = "") -> dict:
    """Generate a structured weekly macro report from all available data.

    Returns dict with keys: generated_at, sections, alerts, markdown.
    """
    now = datetime.now()
    sections = []
    alerts = []

    # ── 1. Macro Regime ──────────────────────────────────────
    try:
        regime = macro_regime(db)
        label = regime["regime"].replace("_", " ").title()
        score = regime["score"]
        signals = regime["signals"]
        body = f"**Regime: {label}** (score: {score})\n\n"
        if "yield_curve_avg_3m" in signals:
            yc = signals["yield_curve_avg_3m"]
            body += f"- Yield curve 3M avg: {yc:.3f}% "
            body += f"({'inverted' if yc < 0 else 'normal'})\n"
        if "cpi_yoy" in signals:
            body += f"- CPI YoY: {signals['cpi_yoy']:.1f}%\n"
        if "credit_spread_zscore" in signals:
            body += f"- Credit spread Z-score: {signals['credit_spread_zscore']:.2f}\n"
        sections.append({"title": "Macro Regime", "body": body})
    except Exception:
        sections.append({"title": "Macro Regime", "body": "Data unavailable."})

    # ── 2. Fear/Greed ────────────────────────────────────────
    try:
        fg = fear_greed_score(db)
        body = f"**{fg['label']}** — Score: {fg['score']:.0f}/100\n\n"
        if fg["components"]:
            # Sort by score to find most fearful/greedy
            sorted_comps = sorted(fg["components"].items(), key=lambda x: x[1]["score"])
            most_greed = sorted_comps[0]
            most_fear = sorted_comps[-1]
            body += f"- Most fearful signal: {most_fear[0]} ({most_fear[1]['score']:.0f})\n"
            body += f"- Most greedy signal: {most_greed[0]} ({most_greed[1]['score']:.0f})\n"
        if fg["score"] >= 80:
            alerts.append("Fear/Greed at Extreme Fear (contrarian bullish)")
        elif fg["score"] <= 20:
            alerts.append("Fear/Greed at Extreme Greed (contrarian bearish)")
        sections.append({"title": "Fear / Greed Gauge", "body": body})
    except Exception:
        sections.append({"title": "Fear / Greed Gauge", "body": "Data unavailable."})

    # ── 3. Yield Curve & Rates ───────────────────────────────
    try:
        spread = yield_curve_spread(db, "DGS2", "DGS10")
        body = ""
        if not spread.empty:
            val = spread.iloc[-1]
            arrow = direction_arrow(spread)
            body += f"- 10Y-2Y spread: {val:.2f}% ({arrow})\n"
            if val < 0:
                alerts.append("Yield curve inverted (10Y-2Y < 0)")
        real = real_rate(db)
        if not real.empty:
            body += f"- Real rate (10Y): {real.iloc[-1]:.2f}%\n"
        sections.append({"title": "Yield Curve & Rates", "body": body or "Data unavailable."})
    except Exception:
        sections.append({"title": "Yield Curve & Rates", "body": "Data unavailable."})

    # ── 4. Inflation ─────────────────────────────────────────
    try:
        cpi = cpi_yoy(db, "CPIAUCSL")
        core = cpi_yoy(db, "CPILFESL")
        body = ""
        if not cpi.empty:
            v = cpi.iloc[-1]
            body += f"- CPI YoY: {v:.1f}% ({direction_arrow(cpi)})\n"
            if v > _ALERT_THRESHOLDS["cpi"][0]:
                alerts.append(_ALERT_THRESHOLDS["cpi"][1])
        if not core.empty:
            body += f"- Core CPI YoY: {core.iloc[-1]:.1f}%\n"
        sections.append({"title": "Inflation", "body": body or "Data unavailable."})
    except Exception:
        sections.append({"title": "Inflation", "body": "Data unavailable."})

    # ── 5. Labor Market ──────────────────────────────────────
    try:
        body = ""
        unrate = query_series(db, "UNRATE")
        if not unrate.empty:
            v = unrate.iloc[-1]
            body += f"- Unemployment: {v:.1f}% ({direction_arrow(unrate)})\n"
            if v > _ALERT_THRESHOLDS["unemployment"][0]:
                alerts.append(_ALERT_THRESHOLDS["unemployment"][1])
        nfp = payrolls_mom_change(db)
        if not nfp.empty:
            body += f"- Latest NFP change: {nfp.iloc[-1]:+.0f}K\n"
        claims = query_series(db, "ICSA")
        if not claims.empty:
            body += f"- Initial claims: {claims.iloc[-1]:,.0f}\n"
        sections.append({"title": "Labor Market", "body": body or "Data unavailable."})
    except Exception:
        sections.append({"title": "Labor Market", "body": "Data unavailable."})

    # ── 6. Credit Conditions ─────────────────────────────────
    try:
        body = ""
        hy = query_series(db, "BAMLH0A0HYM2")
        if not hy.empty:
            v = hy.iloc[-1]
            body += f"- HY OAS: {v:.0f} bps ({direction_arrow(hy)})\n"
            if v > _ALERT_THRESHOLDS["hy_oas"][0]:
                alerts.append(_ALERT_THRESHOLDS["hy_oas"][1])
            if len(hy) >= 252:
                z = rolling_zscore(hy, 252)
                body += f"- HY OAS Z-score: {z.iloc[-1]:.2f}\n"
        hi = hy_ig_spread(db)
        if not hi.empty:
            body += f"- HY-IG spread: {hi.iloc[-1]:.0f} bps\n"
        vix = query_series(db, "VIXCLS")
        if not vix.empty:
            v = vix.iloc[-1]
            body += f"- VIX: {v:.1f} ({direction_arrow(vix)})\n"
            if v > _ALERT_THRESHOLDS["vix"][0]:
                alerts.append(_ALERT_THRESHOLDS["vix"][1])
        sections.append({"title": "Credit & Volatility", "body": body or "Data unavailable."})
    except Exception:
        sections.append({"title": "Credit & Volatility", "body": "Data unavailable."})

    # ── 7. Market Breadth ────────────────────────────────────
    try:
        body = ""
        ratio = spy_rsp_spread(db)
        if not ratio.empty:
            current = ratio.iloc[-1]
            body += f"- SPY/RSP ratio: {current:.1f} ({direction_arrow(ratio)})\n"
            if len(ratio) >= 22:
                chg = current - ratio.iloc[-22]
                signal = "narrowing" if chg > 0.5 else ("broadening" if chg < -0.5 else "neutral")
                body += f"- Breadth signal: {signal} (1M change: {chg:+.1f})\n"
        # Index returns
        for sym, name in [("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"), ("IWM", "Russell 2000")]:
            s = query_series(db, sym)
            w = _wow(s)
            if w:
                body += f"- {name}: {w['pct']:+.1f}% weekly\n"
        sections.append({"title": "Market Breadth", "body": body or "Data unavailable."})
    except Exception:
        sections.append({"title": "Market Breadth", "body": "Data unavailable."})

    # ── 8. Sector Rotation ───────────────────────────────────
    try:
        sec = sector_returns(db)
        body = ""
        if not sec.empty and "1M" in sec.columns:
            sorted_sec = sec.sort_values("1M", ascending=False)
            top3 = sorted_sec.head(3)
            bot3 = sorted_sec.tail(3)
            body += "**Leaders (1M):**\n"
            for name, row in top3.iterrows():
                body += f"- {name} ({row['Symbol']}): {row['1M']:+.1f}%\n"
            body += "\n**Laggards (1M):**\n"
            for name, row in bot3.iterrows():
                body += f"- {name} ({row['Symbol']}): {row['1M']:+.1f}%\n"
        sections.append({"title": "Sector Rotation", "body": body or "Data unavailable."})
    except Exception:
        sections.append({"title": "Sector Rotation", "body": "Data unavailable."})

    # ── 9. Commodities & Energy ──────────────────────────────
    try:
        body = ""
        wti = query_series(db, "DCOILWTICO")
        if not wti.empty:
            body += f"- WTI Crude: ${wti.iloc[-1]:.2f} ({direction_arrow(wti)})\n"
        gsr = gold_silver_ratio(db)
        if not gsr.empty:
            body += f"- Gold/Silver ratio: {gsr.iloc[-1]:.1f}\n"
        ogr = oil_gold_ratio(db)
        if not ogr.empty:
            body += f"- Oil/Gold ratio: {ogr.iloc[-1]:.3f}\n"
        gld = query_series(db, "GLD")
        if not gld.empty:
            body += f"- Gold (GLD): ${gld.iloc[-1]:.2f} ({direction_arrow(gld)})\n"
        sections.append({"title": "Commodities & Energy", "body": body or "Data unavailable."})
    except Exception:
        sections.append({"title": "Commodities & Energy", "body": "Data unavailable."})

    # ── 10. COT Positioning ──────────────────────────────────
    try:
        cot = cot_summary_table(db)
        body = ""
        if not cot.empty:
            extremes = cot[cot["Z-Score"].abs() >= 1.5]
            if not extremes.empty:
                body += "**Notable positioning:**\n"
                for _, row in extremes.iterrows():
                    direction = "long" if row["Z-Score"] > 0 else "short"
                    body += f"- {row['Contract']}: {row['Net %']:+.1f}% (Z: {row['Z-Score']:+.2f}, extreme {direction})\n"
                    if abs(row["Z-Score"]) >= 2.0:
                        alerts.append(f"COT: {row['Contract']} at extreme Z={row['Z-Score']:+.2f}")
            else:
                body += "All contracts within normal positioning range.\n"
            # Quick summary of all
            body += "\n**All contracts:**\n"
            for _, row in cot.iterrows():
                body += f"- {row['Contract']}: net {row['Net %']:+.1f}%, Z={row['Z-Score']:+.2f} {row['Direction']}\n"
        sections.append({"title": "COT Positioning", "body": body or "No COT data available."})
    except Exception:
        sections.append({"title": "COT Positioning", "body": "Data unavailable."})

    # ── 11. Put/Call Ratio ───────────────────────────────────
    try:
        body = ""
        pcr = query_series(db, "PCR_EQUITY")
        if not pcr.empty:
            v = pcr.iloc[-1]
            body += f"- Equity PCR: {v:.2f} ({direction_arrow(pcr)})\n"
            if len(pcr) >= 10:
                ma10 = pcr.rolling(10).mean().iloc[-1]
                body += f"- 10-day MA: {ma10:.2f}\n"
            if v > _ALERT_THRESHOLDS["put_call"][0]:
                alerts.append(_ALERT_THRESHOLDS["put_call"][1])
        sections.append({"title": "Put/Call Ratio", "body": body or "Data unavailable."})
    except Exception:
        sections.append({"title": "Put/Call Ratio", "body": "Data unavailable."})

    # ── 12. Currencies ───────────────────────────────────────
    try:
        body = ""
        dxy = query_series(db, "DX=F")
        if not dxy.empty:
            body += f"- DXY: {dxy.iloc[-1]:.2f} ({direction_arrow(dxy)})\n"
        btc = query_series(db, "BTC-USD")
        if not btc.empty:
            body += f"- BTC: ${btc.iloc[-1]:,.0f} ({direction_arrow(btc)})\n"
        sections.append({"title": "Currencies", "body": body or "Data unavailable."})
    except Exception:
        sections.append({"title": "Currencies", "body": "Data unavailable."})

    # ── 13. Upcoming Calendar ────────────────────────────────
    try:
        if fred_api_key:
            from charlie.analysis.calendar import get_economic_calendar
            from charlie.config import get_settings
            settings = get_settings()
            cal = get_economic_calendar(fred_api_key, settings.calendar_releases, 14)
            body = ""
            if not cal.empty:
                high = cal[cal["importance"] == "high"]
                if not high.empty:
                    body += "**Key releases next 2 weeks:**\n"
                    for _, row in high.iterrows():
                        body += f"- {row['date']} — {row['name']} ({row['full_name']})\n"
                medium = cal[cal["importance"] == "medium"].head(5)
                if not medium.empty:
                    body += "\n**Other notable:**\n"
                    for _, row in medium.iterrows():
                        body += f"- {row['date']} — {row['name']}\n"
            sections.append({"title": "Upcoming Calendar", "body": body or "No upcoming releases found."})
        else:
            sections.append({"title": "Upcoming Calendar", "body": "FRED API key not set — calendar unavailable."})
    except Exception:
        sections.append({"title": "Upcoming Calendar", "body": "Calendar data unavailable."})

    # ── Build markdown ───────────────────────────────────────
    md_lines = [f"# Charlie Finance — Weekly Macro Report", f"*Generated: {now.strftime('%Y-%m-%d %H:%M')}*\n"]

    if alerts:
        md_lines.append("## Alerts")
        for a in alerts:
            md_lines.append(f"- **{a}**")
        md_lines.append("")

    for sec in sections:
        md_lines.append(f"## {sec['title']}")
        md_lines.append(sec["body"])
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("*Report auto-generated by Charlie Finance.*")

    markdown = "\n".join(md_lines)

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "sections": sections,
        "alerts": alerts,
        "markdown": markdown,
    }

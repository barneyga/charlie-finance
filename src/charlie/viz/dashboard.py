"""Charlie Finance — Macro Dashboard (Streamlit)."""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Ensure src is on path when run via streamlit
_src = str(Path(__file__).resolve().parent.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

import streamlit as st
import pandas as pd

from charlie.config import get_settings
from charlie.storage.db import Database
from charlie.storage.models import query_series, query_multiple_series, get_all_series_meta
from charlie.analysis.derived import (
    yield_curve_spread, yield_curve_shape, real_rate,
    cpi_yoy, payrolls_mom_change, m2_yoy,
    hy_ig_spread, credit_impulse,
    gold_silver_ratio, oil_gold_ratio, brent_wti_spread,
    stock_bond_correlation, spy_rsp_spread, sector_returns,
)
from charlie.analysis.stats import rolling_zscore, percentile_rank, direction_arrow
from charlie.analysis.regime import macro_regime, REGIME_COLORS
from charlie.analysis.composite import fear_greed_score
from charlie.analysis.calendar import get_economic_calendar
from charlie.analysis.sentiment import sentiment_summary, ticker_sentiment_ranking, sentiment_vs_price
from charlie.viz.charts import (
    time_series_chart, yield_curve_snapshot, bar_chart, dual_axis_chart,
    normalized_returns_chart, horizontal_bar_chart, gauge_chart,
)

st.set_page_config(page_title="Charlie Finance", page_icon="$", layout="wide")

# Smooth scrolling CSS + abbr tooltip styling
st.markdown("""<style>
html { scroll-behavior: smooth; }
abbr { text-decoration: underline dotted; cursor: help; }
</style>""", unsafe_allow_html=True)

# Acronym glossary — hover to see definition
_GLOSSARY = {
    "CPI": "Consumer Price Index",
    "PCE": "Personal Consumption Expenditures",
    "NFP": "Nonfarm Payrolls",
    "FOMC": "Federal Open Market Committee",
    "GDP": "Gross Domestic Product",
    "YoY": "Year-over-Year",
    "MoM": "Month-over-Month",
    "VIX": "CBOE Volatility Index",
    "HY": "High Yield",
    "IG": "Investment Grade",
    "OAS": "Option-Adjusted Spread",
    "DXY": "US Dollar Index",
    "NFCI": "National Financial Conditions Index",
    "STLFSI": "St. Louis Fed Financial Stress Index",
    "RSP": "Equal-Weight S&P 500 ETF",
    "SPY": "S&P 500 ETF (cap-weighted)",
    "TLT": "20+ Year Treasury Bond ETF",
    "GICS": "Global Industry Classification Standard",
    "M2": "M2 Money Supply",
    "BTC": "Bitcoin",
    "FX": "Foreign Exchange",
    "TIPS": "Treasury Inflation-Protected Securities",
    "WTI": "West Texas Intermediate (US crude benchmark)",
    "GLD": "Gold ETF (SPDR)",
    "SLV": "Silver ETF (iShares)",
    "COPX": "Copper Miners ETF",
    "USO": "United States Oil Fund ETF",
    "EFA": "Developed Markets ex-US ETF",
    "EEM": "Emerging Markets ETF",
    "VGK": "European Stocks ETF",
    "ITA": "US Aerospace & Defense ETF",
    "QQQ": "Nasdaq 100 ETF",
    "IGV": "Software ETF",
    "SOXX": "Semiconductor ETF",
    "NVDA": "NVIDIA Corporation",
    "WAL": "Write-Ahead Logging",
    "FRED": "Federal Reserve Economic Data",
    "VADER": "Valence Aware Dictionary for Sentiment Reasoning",
    "PCR": "Put/Call Ratio",
    "CBOE": "Chicago Board Options Exchange",
    "PRAW": "Python Reddit API Wrapper",
    "ETF": "Exchange-Traded Fund",
    "MA": "Moving Average",
    "HYG": "High Yield Corporate Bond ETF",
    "LQD": "Investment Grade Corporate Bond ETF",
}


def _abbr(text: str) -> str:
    """Replace known acronyms with <abbr> hover tooltips."""
    import re
    for acr, defn in sorted(_GLOSSARY.items(), key=lambda x: -len(x[0])):
        text = re.sub(
            rf"\b{re.escape(acr)}\b",
            f'<abbr title="{defn}">{acr}</abbr>',
            text,
        )
    return text

# Section definitions for navigation
SECTIONS = {
    "Macro Overview": [
        ("regime", "Macro Regime"),
        ("calendar", "Economic Calendar"),
    ],
    "Fixed Income & Policy": [
        ("yield_curve", "Yield Curve"),
        ("inflation", "Inflation"),
        ("labor", "Labor Market"),
        ("credit", "Credit Deep Dive"),
        ("monetary", "Monetary Policy"),
    ],
    "Equities & Sectors": [
        ("breadth", "Market Breadth"),
        ("sectors", "Sector Scorecard"),
    ],
    "Cross-Asset": [
        ("metals", "Commodities & Energy"),
        ("divergence", "Cross-Asset Divergence"),
        ("geo", "Geographic Rotation"),
        ("tech", "AI & Tech Sub-sectors"),
    ],
    "FX & Sentiment": [
        ("currencies", "Currencies"),
        ("sentiment", "Reddit Sentiment"),
    ],
}


@st.cache_resource
def get_db():
    settings = get_settings()
    db = Database(settings.db_path)
    db.init_schema()
    return db


def _data_freshness(db):
    """Query last update timestamps per source."""
    rows = db.conn.execute(
        "SELECT source, MAX(last_updated) as latest FROM series_meta GROUP BY source"
    ).fetchall()
    result = {}
    for r in rows:
        if r["latest"]:
            try:
                dt = datetime.fromisoformat(r["latest"])
                hours_ago = (datetime.now() - dt).total_seconds() / 3600
                result[r["source"]] = (r["latest"], hours_ago)
            except (ValueError, TypeError):
                result[r["source"]] = (r["latest"], 999)
    return result


def _anchor(section_id: str):
    """Insert an HTML anchor for sidebar navigation."""
    st.markdown(f'<div id="{section_id}"></div>', unsafe_allow_html=True)


def _section_info(explanation: str):
    """Render a clickable info popover inside a section."""
    with st.popover("ℹ️"):
        st.markdown(_abbr(explanation), unsafe_allow_html=True)


def _alert_badge(value: float, thresholds: dict[str, tuple[float, float]]) -> str:
    """Return emoji badge based on value and threshold ranges.

    thresholds = {"green": (lo, hi), "yellow": (lo, hi), "red": (lo, hi)}
    """
    for color, (lo, hi) in thresholds.items():
        if lo <= value < hi:
            return {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(color, "⚪")
    return "⚪"


# Section explanations
_INFO = {
    "regime": (
        "**What:** Classifies the economy into 4 states — Expansion, Late Cycle, Contraction, "
        "or Recovery — based on 6 signals.\n\n"
        "**Signals used:** Yield curve inversion, credit spread z-scores, CPI YoY, HY OAS level, "
        "unemployment trend (3-month change), and loan delinquency rates.\n\n"
        "**How to read:** Score ranges from -4 (deep contraction) to +4 (strong expansion). "
        "Transitions between regimes matter more than the regime itself — Late Cycle to Contraction "
        "is when risk assets typically sell off. Recovery to Expansion is when you want to be long."
    ),
    "fear_greed": (
        "**What:** A composite 0-100 score measuring market sentiment from 7 price-based signals. "
        "**0 = Extreme Greed** (complacent, risky), **100 = Extreme Fear** (panic, often a buying "
        "opportunity).\n\n"
        "**Components:** VIX level, HY-IG credit stress, SPY/RSP breadth divergence, SPY-TLT "
        "correlation, gold/silver safe haven demand, yield curve shape, and SPY momentum "
        "(50-day vs 200-day MA).\n\n"
        "**How to read:** Each component is percentile-ranked over a 5-year window and "
        "equal-weighted. Extremes (below 20 or above 80) are contrarian signals — when everyone "
        "is fearful, it's often near a bottom. When everyone is greedy, risk is elevated."
    ),
    "calendar": (
        "**What:** Upcoming economic data releases from FRED.\n\n"
        "**Importance levels:** 🔴 High = market-moving (CPI, NFP, FOMC, GDP, PCE). "
        "🟡 Medium = notable (retail sales, housing starts, industrial production). "
        "⚪ Low = context (minor regional surveys).\n\n"
        "**Why it matters:** Markets move on *surprises* relative to expectations. Knowing when "
        "data drops lets you position ahead or avoid being caught off-guard."
    ),
    "yield_curve": (
        "**What:** The yield curve plots Treasury bond yields by maturity (1 month to 30 years). "
        "Normally it slopes upward — you earn more for lending money longer.\n\n"
        "**The 10Y-2Y spread** is the most-watched indicator. When it goes **below zero (inverts)**, "
        "short-term rates exceed long-term rates. This means the bond market expects the Fed will "
        "need to cut rates because the economy is weakening.\n\n"
        "**Why it matters:** An inverted yield curve has preceded every US recession in the last "
        "50 years, typically by 12-18 months. The *un-inversion* (spread going back positive) "
        "often signals the recession is imminent or starting — not that things are better.\n\n"
        "**How to read the charts:** Left chart shows the curve's current shape vs 3 months and "
        "1 year ago. A flattening or inverting shape = caution. Right chart tracks the 10Y-2Y "
        "spread over time — the dashed line at zero is your warning level."
    ),
    "inflation": (
        "**What:** Measures how fast prices are rising.\n\n"
        "**CPI YoY:** Headline Consumer Price Index, year-over-year change. Includes everything "
        "you buy — food, gas, rent, etc. **Core CPI** strips out food and energy because they're "
        "volatile, giving a cleaner trend.\n\n"
        "**Breakeven rates:** The difference between regular Treasury yields and TIPS yields. "
        "This is what the *bond market* thinks inflation will average over 5 or 10 years. "
        "When breakevens rise, markets expect more inflation ahead.\n\n"
        "**Why it matters:** The Fed targets ~2% inflation. Above 3% = Fed likely tightening "
        "(bad for stocks/bonds). Below 1% = deflation risk (bad for economy).\n\n"
        "**Alert thresholds:** 🟢 CPI < 3%, 🟡 3-4%, 🔴 > 4%."
    ),
    "labor": (
        "**What:** Four key employment indicators.\n\n"
        "**Unemployment rate (U-3):** Percentage of people actively looking for work who can't "
        "find it. Below 4% is historically tight. Rising unemployment triggers Fed rate cuts.\n\n"
        "**NFP (Nonfarm Payrolls):** Jobs added/lost last month. The single most market-moving "
        "data point each month. Positive = growth, negative = contraction.\n\n"
        "**Initial Claims:** Weekly count of *new* unemployment filings. Leading indicator — "
        "spikes precede rising unemployment by weeks. Under 250K = healthy.\n\n"
        "**Continued Claims:** People still receiving benefits. Shows how long it takes to find "
        "new work. Rising = labor market deteriorating.\n\n"
        "**Alert thresholds:** 🟢 Unemployment < 4.5%, 🟡 4.5-6%, 🔴 > 6%."
    ),
    "credit": (
        "**What:** Credit markets often signal trouble before stocks do.\n\n"
        "**HY OAS:** The extra yield investors demand to hold risky corporate bonds over safe "
        "Treasuries. Think of it as the market's price for default risk. Under 400 bps = calm, "
        "over 600 bps = stress, over 800 bps = crisis.\n\n"
        "**HY-IG spread:** Gap between high-yield and investment-grade spreads. When this widens, "
        "the riskiest companies are getting punished — early warning of trouble.\n\n"
        "**Z-scores:** How far current spreads are from their 1-year average in standard "
        "deviations. Above 1.5 = unusually wide, something is brewing.\n\n"
        "**VIX:** Implied volatility of S&P 500 options. Under 20 = calm, over 25 = elevated, "
        "over 35 = panic.\n\n"
        "**Put/Call Ratio (PCR):** Ratio of put option volume to call option volume from CBOE. "
        "High PCR (>1.0) = heavy put buying (hedging/fear) — contrarian bullish signal. "
        "Low PCR (<0.7) = heavy call buying (complacency) — contrarian bearish signal. "
        "The 10-day moving average smooths daily noise.\n\n"
        "**NFCI / STLFSI:** Financial conditions indices. Positive = tighter than average "
        "(restrictive). Negative = loose (accommodative).\n\n"
        "**Alert thresholds:** HY OAS: 🟢 < 400, 🟡 400-600, 🔴 > 600. VIX: 🟢 < 20, 🟡 20-25, 🔴 > 25. "
        "Put/Call: 🟢 < 0.7, 🟡 0.7-1.0, 🔴 > 1.0."
    ),
    "monetary": (
        "**What:** Fed policy and money supply.\n\n"
        "**Fed Funds Rate:** The interest rate banks charge each other overnight. The Fed's "
        "primary tool — raising it slows the economy, lowering it stimulates. Impacts every "
        "other rate in the economy.\n\n"
        "**M2 Money Supply (YoY):** Total money in the economy (cash, checking, savings, money "
        "market funds). Normally grows 5-7%/year. Negative M2 growth is historically rare and "
        "signals active liquidity drain — restrictive for asset prices.\n\n"
        "**Fed Balance Sheet:** Total assets held by the Fed. Growing (QE) = injecting money. "
        "Shrinking (QT) = pulling money out. More liquidity = bullish for risk assets."
    ),
    "breadth": (
        "**What:** Compares SPY (market-cap-weighted) vs RSP (equal-weight S&P 500).\n\n"
        "**Why this matters:** SPY gives huge weight to the top 7-10 stocks (Apple, Microsoft, "
        "NVIDIA, etc.). RSP treats all 500 equally. When SPY rises but RSP lags, only a few "
        "mega-caps are driving gains — the rally is narrow and fragile.\n\n"
        "**How to read:** Rising SPY/RSP ratio = increasing concentration (top-heavy, risky). "
        "Falling ratio = broad participation (healthier, more sustainable rally). "
        "**Crown's key principle: breadth reveals what the index hides.**"
    ),
    "sectors": (
        "**What:** Performance of all 11 GICS sector ETFs.\n\n"
        "**Sector rotation** is a key macro signal. Early expansion: cyclicals lead (XLY, XLF, "
        "XLI). Late cycle: energy and materials (XLE, XLB). Contraction: defensives hold up "
        "(XLU, XLP, XLV). Recovery: tech and discretionary lead again.\n\n"
        "**How to read:** Green = positive return, red = negative. Sorted by 1-month return. "
        "Which sectors lead/lag tells you where institutional money is flowing.\n\n"
        "**Sectors:** XLF (Financials), XLE (Energy), XLK (Tech), XLV (Healthcare), XLU (Utilities), "
        "XLB (Materials), XLI (Industrials), XLY (Discretionary), XLP (Staples), "
        "XLC (Comms), XLRE (Real Estate)."
    ),
    "metals": (
        "**Oil:**\n\n"
        "**WTI** = US crude benchmark. **Brent** = international benchmark. Brent usually trades "
        "at a premium. **Brent-WTI spread widening** = global supply disruption. Narrowing = "
        "US-specific dynamics.\n\n"
        "**Oil/Gold ratio:** Rising = growth/risk-on (demand driving oil). Falling = stagflation "
        "risk or flight to safety.\n\n"
        "**WTI thresholds:** 🟢 $50-80, 🟡 $30-50 or $80-100, 🔴 > $100 or < $30.\n\n"
        "---\n\n"
        "**Metals:**\n\n"
        "**Gold/Silver ratio:** Rising = risk-off. Above 80 = elevated fear. Above 90 = extreme.\n\n"
        "**Real yields vs gold:** Inversely correlated. Falling real rates = gold more attractive "
        "(lower opportunity cost of holding non-yielding asset).\n\n"
        "**Silver vs copper:** Both industrial metals. Divergence signals shift between industrial "
        "vs monetary demand."
    ),
    "divergence": (
        "**What:** Cross-asset divergences are early warning signals.\n\n"
        "**SPY-TLT correlation** (63-day rolling): Normally stocks and bonds move inversely — "
        "when stocks fall, bonds rally as a safe haven. **When correlation turns positive**, "
        "stocks and bonds are falling *together*. This usually means macro stress — inflation "
        "forcing the Fed to tighten while growth slows. Breaks 60/40 portfolio diversification.\n\n"
        "**HY-IG spread:** When high-yield bonds underperform investment-grade, credit markets "
        "are pricing in rising default risk. Credit often leads equities by weeks."
    ),
    "geo": (
        "**What:** Relative performance of US vs international markets.\n\n"
        "**SPY vs EFA vs EEM:** SPY = US, EFA = developed ex-US (Europe, Japan, Australia), "
        "EEM = emerging (China, India, Brazil). When EFA/EEM outperform SPY, global capital is "
        "rotating away from the US — often driven by dollar weakness or relative valuation.\n\n"
        "**VGK (Europe):** European equities specifically. Tracks EU economic divergence.\n\n"
        "**ITA (Defense) vs SPY:** Outperformance signals geopolitical risk being priced in — "
        "useful for monitoring war/conflict positioning."
    ),
    "tech": (
        "**What:** Tech sub-sector breakdown — 'tech' is not monolithic.\n\n"
        "**QQQ** = Nasdaq 100 (mega-cap tech-heavy). **IGV** = pure software. **SOXX** = "
        "semiconductors. These can diverge sharply — AI hype lifts SOXX/NVDA while software "
        "may lag if growth is slowing.\n\n"
        "**NVDA vs QQQ:** NVIDIA is QQQ's largest weight. Divergence shows whether the AI trade "
        "is broadening or narrowing.\n\n"
        "**Why it matters:** Buying QQQ for 'AI exposure' gives heavy Apple/Microsoft weight "
        "with diluted semi exposure. Sub-sector selection matters more than the index."
    ),
    "currencies": (
        "**What:** Currency markets and their macro signals.\n\n"
        "**DXY (Dollar Index):** USD vs 6 major currencies (heavy Euro weight). Strong dollar = "
        "headwind for emerging markets (dollar debt), commodities (priced in USD), and US "
        "multinationals (foreign revenue worth less). Weak dollar = tailwind for all three.\n\n"
        "**FX pairs:** USD/JPY rising = risk-on (yen weakening). EUR/USD rising = dollar "
        "weakening.\n\n"
        "**BTC:** Risk/liquidity barometer. Correlates with risk-on assets, inversely with real "
        "rates. Not a hedge — more like leveraged risk appetite."
    ),
    "sentiment": (
        "**What:** Reddit sentiment from finance-focused subreddits.\n\n"
        "**Sources:** r/wallstreetbets, r/stocks, r/investing. Each post scored using VADER "
        "sentiment analysis, averaged daily.\n\n"
        "**Score:** 0 = Very Bearish, 50 = Neutral, 100 = Very Bullish. Per-ticker mentions "
        "tracked by keyword matching.\n\n"
        "**How to use:** Reddit sentiment is *contrarian* at extremes. Retail overwhelmingly "
        "bullish (>70) = caution. Fear dominant (<30) = often marks bottoms. Mid-range = noise."
    ),
}

# Alert thresholds for key metrics
_THRESHOLDS = {
    "vix": {"green": (0, 20), "yellow": (20, 25), "red": (25, float("inf"))},
    "spread_10y2y": {"red": (-float("inf"), 0), "yellow": (0, 0.5), "green": (0.5, float("inf"))},
    "hy_oas": {"green": (0, 400), "yellow": (400, 600), "red": (600, float("inf"))},
    "hy_oas_z": {"green": (-float("inf"), 1.0), "yellow": (1.0, 1.5), "red": (1.5, float("inf"))},
    "cpi_yoy": {"green": (0, 3), "yellow": (3, 4), "red": (4, float("inf"))},
    "unemployment": {"green": (0, 4.5), "yellow": (4.5, 6), "red": (6, float("inf"))},
    "gold_silver": {"green": (0, 80), "yellow": (80, 90), "red": (90, float("inf"))},
    "put_call": {"green": (0, 0.7), "yellow": (0.7, 1.0), "red": (1.0, float("inf"))},
}


def main():
    db = get_db()
    settings = get_settings()

    # -- Sidebar --
    st.sidebar.title("Charlie Finance")
    st.sidebar.caption("Macro Analysis Dashboard")

    # Data freshness
    freshness = _data_freshness(db)
    if freshness:
        st.sidebar.markdown("**Data Freshness**")
        for source, (ts, hours) in freshness.items():
            if hours < 12:
                icon = "🟢"
            elif hours < 24:
                icon = "🟡"
            else:
                icon = "🔴"
            label = source.upper()
            st.sidebar.caption(f"{icon} {label}: {hours:.0f}h ago")
        st.sidebar.divider()

    # Navigation
    st.sidebar.markdown("**Navigation**")
    for group, items in SECTIONS.items():
        st.sidebar.markdown(f"*{group}*")
        for sid, name in items:
            st.sidebar.markdown(f"[{name}](#{sid})", unsafe_allow_html=True)

    st.sidebar.divider()

    # Glossary
    with st.sidebar.expander("📖 Glossary"):
        for acr, defn in sorted(_GLOSSARY.items()):
            st.markdown(f"**{acr}** — {defn}")

    st.sidebar.divider()

    default_start = (datetime.now() - timedelta(days=365 * 2)).date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start, datetime.now().date()),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = str(date_range[0])
        end_date = str(date_range[1])
    else:
        start_date = str(default_start)
        end_date = str(datetime.now().date())

    if st.sidebar.button("Refresh FRED Data"):
        with st.sidebar.status("Fetching FRED data..."):
            from charlie.ingest.fred import FredIngester
            ingester = FredIngester(settings, db)
            count = ingester.fetch_all()
            st.sidebar.success(f"Fetched {count} observations")
            st.sidebar.text(ingester.report())
            st.cache_resource.clear()
            st.rerun()

    if st.sidebar.button("Refresh Market Data"):
        with st.sidebar.status("Fetching Yahoo data..."):
            from charlie.ingest.yahoo import YahooIngester
            ingester = YahooIngester(settings, db)
            count = ingester.fetch_all()
            st.sidebar.success(f"Fetched {count} observations")
            st.sidebar.text(ingester.report())
            st.cache_resource.clear()
            st.rerun()

    if settings.reddit_client_id and settings.sentiment:
        if st.sidebar.button("Refresh Sentiment"):
            with st.sidebar.status("Fetching Reddit sentiment..."):
                from charlie.ingest.sentiment import SentimentIngester
                ingester = SentimentIngester(settings, db)
                count = ingester.fetch_all()
                st.sidebar.success(f"Scored {count} posts")
                st.sidebar.text(ingester.report())
                st.cache_resource.clear()
                st.rerun()

    # Check if we have data
    meta = get_all_series_meta(db)
    if not meta:
        st.warning(
            "No data loaded yet. Click **Refresh Data from FRED** in the sidebar, "
            "or run `python scripts/fetch.py --all` from the terminal."
        )
        st.info(
            "Make sure you have a `.env` file with your FRED_API_KEY. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        return

    # ============================================================
    # MACRO OVERVIEW
    # ============================================================
    st.markdown("## Macro Overview")

    # Section 1: Macro Regime Summary + Fear/Greed
    _anchor("regime")
    with st.expander("Macro Regime + Fear/Greed", expanded=True):
        _section_info(_INFO["regime"] + "\n\n---\n\n**Fear/Greed:** " + _INFO["fear_greed"])
        regime_data = macro_regime(db)
        regime_label = regime_data["regime"].replace("_", " ").title()
        regime_color = REGIME_COLORS.get(regime_data["regime"], "#888")

        fg = fear_greed_score(db)

        r1_col1, r1_col2, r1_col3 = st.columns([2, 2, 1])
        with r1_col1:
            st.markdown(
                f"### <span style='color:{regime_color}'>{regime_label}</span> "
                f"(score: {regime_data['score']})",
                unsafe_allow_html=True,
            )
            signals = regime_data["signals"]
            metric_cols = st.columns(2)
            with metric_cols[0]:
                if "cpi_yoy" in signals:
                    _cpi = signals['cpi_yoy']
                    st.metric(f"{_alert_badge(_cpi, _THRESHOLDS['cpi_yoy'])} CPI YoY", f"{_cpi:.1f}%")
            with metric_cols[1]:
                if "credit_spread_zscore" in signals:
                    st.metric("Credit Spread Z", f"{signals['credit_spread_zscore']:.2f}")

        with r1_col2:
            st.plotly_chart(
                gauge_chart(fg["score"], "Fear / Greed", fg["label"]),
                width="stretch",
            )

        with r1_col3:
            st.metric("Score", f"{fg['score']:.0f} / 100")
            st.markdown(
                f"<span style='color:{fg['color']};font-size:1.2em;font-weight:bold'>"
                f"{fg['label']}</span>",
                unsafe_allow_html=True,
            )

        if fg["components"]:
            st.subheader("Fear/Greed Components")
            comp_rows = []
            for name, data in fg["components"].items():
                comp_rows.append({
                    "Component": name,
                    "Score": f"{data['score']:.0f}",
                    "Raw Value": str(data["raw_value"]),
                    "Description": data["description"],
                })
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, width="stretch")

        if not fg["history"].empty:
            hist = fg["history"].loc[start_date:end_date]
            if not hist.empty:
                fig = time_series_chart(hist, "Fear/Greed History", yaxis_title="Score (0=Greed, 100=Fear)")
                fig.add_hline(y=20, line_dash="dot", line_color="#22c55e", annotation_text="Greed")
                fig.add_hline(y=80, line_dash="dot", line_color="#ef4444", annotation_text="Fear")
                fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                st.plotly_chart(fig, width="stretch")

        signal_rows = []
        for key, val in regime_data["signals"].items():
            if not key.endswith("_signal"):
                continue
            indicator = key.replace("_signal", "").replace("_", " ").title()
            signal_rows.append({"Indicator": indicator, "Signal": val})
        if signal_rows:
            st.subheader("Regime Signals")
            st.dataframe(pd.DataFrame(signal_rows), hide_index=True, width="stretch")

    # Section 2: Economic Calendar
    _anchor("calendar")
    with st.expander("Economic Calendar", expanded=True):
        _section_info(_INFO["calendar"])
        @st.cache_data(ttl=3600)
        def _load_calendar(_api_key, _releases, _days):
            return get_economic_calendar(_api_key, _releases, _days)

        cal_df = _load_calendar(settings.fred_api_key, settings.calendar_releases, 30)

        if not cal_df.empty:
            key_releases = ["CPI", "NFP", "FOMC", "GDP", "PCE"]
            cols = st.columns(len(key_releases))
            for i, name in enumerate(key_releases):
                match = cal_df[cal_df["name"] == name]
                with cols[i]:
                    if not match.empty:
                        row = match.iloc[0]
                        days = row["days_until"]
                        label = "Today" if days == 0 else f"in {days}d"
                        st.metric(name, row["date"], label)
                    else:
                        st.metric(name, "—")

            show_all = st.checkbox("Show all releases", value=False, key="cal_show_all")
            if show_all:
                display_df = cal_df
            else:
                display_df = cal_df[cal_df["importance"].isin(["high", "medium"])]

            importance_map = {"high": "🔴", "medium": "🟡", "low": "⚪"}
            styled = display_df.copy()
            styled["importance"] = styled["importance"].map(
                lambda x: f"{importance_map.get(x, '')} {x}"
            )
            styled = styled.rename(columns={
                "date": "Date",
                "name": "Event",
                "full_name": "Description",
                "importance": "Impact",
                "days_until": "Days Until",
            })

            st.dataframe(styled, hide_index=True, width="stretch")
        else:
            st.info("No upcoming releases found. Check your FRED API key.")

    # ============================================================
    # FIXED INCOME & POLICY
    # ============================================================
    st.markdown("## Fixed Income & Policy")

    # Section 3: Yield Curve
    _anchor("yield_curve")
    with st.expander("Yield Curve", expanded=True):
        _section_info(_INFO["yield_curve"])
        yc_col1, yc_col2 = st.columns(2)

        with yc_col1:
            curves = {}
            today_curve = yield_curve_shape(db)
            if not today_curve.empty:
                curves["Current"] = today_curve

            date_3m = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            curve_3m = yield_curve_shape(db, date=date_3m)
            if not curve_3m.empty:
                curves["3 Months Ago"] = curve_3m

            date_1y = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            curve_1y = yield_curve_shape(db, date=date_1y)
            if not curve_1y.empty:
                curves["1 Year Ago"] = curve_1y

            if curves:
                st.plotly_chart(
                    yield_curve_snapshot(curves, "Yield Curve Shape"),
                    width="stretch",
                )

        with yc_col2:
            spread_10y2y = yield_curve_spread(db, "DGS2", "DGS10")
            if not spread_10y2y.empty:
                spread_filtered = spread_10y2y.loc[start_date:end_date]
                latest_spread = spread_filtered.iloc[-1] if not spread_filtered.empty else None
                if latest_spread is not None:
                    badge = _alert_badge(latest_spread, _THRESHOLDS["spread_10y2y"])
                    st.metric(f"{badge} 10Y-2Y Spread", f"{latest_spread:.2f}%")
                fig = time_series_chart(
                    spread_filtered, "10Y-2Y Spread", db=db, yaxis_title="%"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                st.plotly_chart(fig, width="stretch")

    # Section 4: Inflation
    _anchor("inflation")
    with st.expander("Inflation", expanded=True):
        _section_info(_INFO["inflation"])
        inf_col1, inf_col2 = st.columns(2)

        with inf_col1:
            cpi = cpi_yoy(db, "CPIAUCSL")
            core_cpi = cpi_yoy(db, "CPILFESL")
            if not cpi.empty or not core_cpi.empty:
                inf_df = pd.DataFrame({"CPI YoY": cpi, "Core CPI YoY": core_cpi}).loc[start_date:end_date]
                st.plotly_chart(
                    time_series_chart(inf_df, "CPI Year-over-Year %", db=db, yaxis_title="%"),
                    width="stretch",
                )

        with inf_col2:
            breakevens = query_multiple_series(db, ["T5YIE", "T10YIE"], start=start_date, end=end_date)
            if not breakevens.empty:
                breakevens.columns = ["5Y Breakeven", "10Y Breakeven"]
                st.plotly_chart(
                    time_series_chart(breakevens, "Breakeven Inflation Rates", db=db, yaxis_title="%"),
                    width="stretch",
                )

    # Section 5: Labor Market
    _anchor("labor")
    with st.expander("Labor Market", expanded=True):
        _section_info(_INFO["labor"])
        lab_col1, lab_col2 = st.columns(2)

        with lab_col1:
            unrate = query_series(db, "UNRATE", start=start_date, end=end_date)
            if not unrate.empty:
                unrate.name = "Unemployment Rate"
                latest_unrate = unrate.iloc[-1]
                badge = _alert_badge(latest_unrate, _THRESHOLDS["unemployment"])
                st.metric(f"{badge} Unemployment Rate", f"{latest_unrate:.1f}%")
                st.plotly_chart(
                    time_series_chart(unrate, "Unemployment Rate", db=db, yaxis_title="%"),
                    width="stretch",
                )

        with lab_col2:
            nfp = payrolls_mom_change(db)
            if not nfp.empty:
                nfp_filtered = nfp.loc[start_date:end_date]
                st.plotly_chart(
                    bar_chart(nfp_filtered, "Nonfarm Payrolls MoM Change (thousands)"),
                    width="stretch",
                )

        claims = query_multiple_series(db, ["ICSA", "CCSA"], start=start_date, end=end_date)
        if not claims.empty:
            claims.columns = ["Initial Claims", "Continued Claims"]
            st.plotly_chart(
                dual_axis_chart(
                    claims["Initial Claims"], claims["Continued Claims"],
                    "Jobless Claims",
                    y1_title="Initial Claims", y2_title="Continued Claims",
                    db=db,
                ),
                width="stretch",
            )

    # Section 6: Credit Deep Dive
    _anchor("credit")
    with st.expander("Credit Deep Dive", expanded=True):
        _section_info(_INFO["credit"])
        hy_oas = query_series(db, "BAMLH0A0HYM2")
        ig_oas = query_series(db, "BAMLC0A0CM")
        delinq_all = query_series(db, "DRALACBS")

        if not hy_oas.empty:
            m1, m2_col, m3, m4 = st.columns(4)
            with m1:
                _hy = hy_oas.iloc[-1]
                st.metric(f"{_alert_badge(_hy, _THRESHOLDS['hy_oas'])} HY OAS", f"{_hy:.0f} bps")
            with m2_col:
                if not ig_oas.empty:
                    spread_val = hy_oas.iloc[-1] - ig_oas.iloc[-1]
                    st.metric("HY-IG Spread", f"{spread_val:.0f} bps")
            with m3:
                if len(hy_oas) >= 252:
                    z = rolling_zscore(hy_oas, 252)
                    _z_val = z.iloc[-1]
                    st.metric(f"{_alert_badge(_z_val, _THRESHOLDS['hy_oas_z'])} HY OAS Z-Score", f"{_z_val:.2f}")
            with m4:
                if not delinq_all.empty:
                    st.metric("All Loans Delinquency", f"{delinq_all.iloc[-1]:.2f}%")

        oas_col1, oas_col2 = st.columns(2)

        with oas_col1:
            oas_df = query_multiple_series(
                db, ["BAMLH0A0HYM2", "BAMLC0A0CM"], start=start_date, end=end_date
            )
            if not oas_df.empty:
                oas_df.columns = ["HY OAS", "IG OAS"]
                st.plotly_chart(
                    time_series_chart(oas_df, "OAS Spreads (bps)", db=db, yaxis_title="bps"),
                    width="stretch",
                )

        with oas_col2:
            try:
                hy_ig = hy_ig_spread(db)
                if not hy_ig.empty:
                    hy_ig_filtered = hy_ig.loc[start_date:end_date]
                    hy_ig_z = rolling_zscore(hy_ig_filtered)
                    hy_ig_filtered.name = "HY-IG Spread"
                    hy_ig_z.name = "Z-Score"
                    st.plotly_chart(
                        dual_axis_chart(
                            hy_ig_filtered, hy_ig_z,
                            "HY-IG Spread Differential",
                            y1_title="Spread (bps)", y2_title="Z-Score",
                            db=db,
                        ),
                        width="stretch",
                    )
            except Exception:
                pass

        yd_col1, yd_col2 = st.columns(2)

        with yd_col1:
            hy_yield = query_series(db, "BAMLH0A0HYM2EY", start=start_date, end=end_date)
            if not hy_yield.empty:
                hy_yield.name = "HY Effective Yield"
                st.plotly_chart(
                    time_series_chart(hy_yield, "High Yield Effective Yield", db=db, yaxis_title="%"),
                    width="stretch",
                )

        with yd_col2:
            delinq_df = query_multiple_series(
                db, ["DRALACBS", "DRTSCILM", "DRSFRMACBS"], start=start_date, end=end_date
            )
            if not delinq_df.empty:
                delinq_df.columns = ["All Loans", "C&I Loans", "SF Residential"]
                st.plotly_chart(
                    time_series_chart(delinq_df, "Loan Delinquency Rates", db=db, yaxis_title="%"),
                    width="stretch",
                )

        cv_col1, cv_col2 = st.columns(2)

        with cv_col1:
            try:
                impulse = credit_impulse(db)
                if not impulse.empty:
                    impulse_filtered = impulse.loc[start_date:end_date]
                    fig = bar_chart(impulse_filtered, "Credit Impulse (Total Loans YoY %)")
                    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    st.plotly_chart(fig, width="stretch")
            except Exception:
                pass

        with cv_col2:
            vix = query_series(db, "VIXCLS", start=start_date, end=end_date)
            if not vix.empty:
                vix.name = "VIX"
                latest_vix = vix.iloc[-1]
                badge = _alert_badge(latest_vix, _THRESHOLDS["vix"])
                st.metric(f"{badge} VIX", f"{latest_vix:.1f}")
                st.plotly_chart(
                    time_series_chart(vix, "VIX (Volatility Index)", db=db),
                    width="stretch",
                )

        pcr_col1, pcr_col2 = st.columns(2)

        with pcr_col1:
            pcr_equity = query_series(db, "PCR_EQUITY", start=start_date, end=end_date)
            if not pcr_equity.empty:
                _pcr_val = pcr_equity.iloc[-1]
                badge = _alert_badge(_pcr_val, _THRESHOLDS["put_call"])
                st.metric(f"{badge} Equity Put/Call", f"{_pcr_val:.2f}")
                # Add 10-day MA overlay
                pcr_ma = pcr_equity.rolling(10).mean()
                pcr_df = pd.DataFrame({"Put/Call Ratio": pcr_equity, "10d MA": pcr_ma}).dropna()
                if not pcr_df.empty:
                    st.plotly_chart(
                        time_series_chart(pcr_df, "CBOE Equity Put/Call Ratio", yaxis_title="Ratio"),
                        width="stretch",
                    )

        with pcr_col2:
            pcr_total = query_series(db, "PCR_TOTAL", start=start_date, end=end_date)
            if not pcr_total.empty:
                st.metric("Total Put/Call", f"{pcr_total.iloc[-1]:.2f}")
                pcr_t_ma = pcr_total.rolling(10).mean()
                pcr_t_df = pd.DataFrame({"Put/Call Ratio": pcr_total, "10d MA": pcr_t_ma}).dropna()
                if not pcr_t_df.empty:
                    st.plotly_chart(
                        time_series_chart(pcr_t_df, "CBOE Total Put/Call Ratio", yaxis_title="Ratio"),
                        width="stretch",
                    )

        fc = query_multiple_series(db, ["NFCI", "STLFSI2"], start=start_date, end=end_date)
        if not fc.empty:
            fc.columns = ["Chicago Fed NFCI", "St. Louis Fed Stress"]
            st.plotly_chart(
                time_series_chart(fc, "Financial Conditions Indices", db=db),
                width="stretch",
            )

    # Section 7: Monetary Policy
    _anchor("monetary")
    with st.expander("Monetary Policy", expanded=True):
        _section_info(_INFO["monetary"])
        mp_col1, mp_col2 = st.columns(2)

        with mp_col1:
            ff = query_series(db, "DFF", start=start_date, end=end_date)
            if not ff.empty:
                ff.name = "Fed Funds Rate"
                st.plotly_chart(
                    time_series_chart(ff, "Effective Fed Funds Rate", db=db, yaxis_title="%"),
                    width="stretch",
                )

        with mp_col2:
            m2 = m2_yoy(db)
            if not m2.empty:
                m2_filtered = m2.loc[start_date:end_date]
                m2_filtered.name = "M2 YoY %"
                fig = time_series_chart(m2_filtered, "M2 Money Supply YoY Growth", db=db, yaxis_title="%")
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                st.plotly_chart(fig, width="stretch")

        walcl = query_series(db, "WALCL", start=start_date, end=end_date)
        if not walcl.empty:
            walcl.name = "Fed Total Assets ($M)"
            st.plotly_chart(
                time_series_chart(walcl, "Fed Balance Sheet Total Assets", db=db, yaxis_title="$ Millions"),
                width="stretch",
            )

    # ============================================================
    # EQUITIES & SECTORS
    # ============================================================
    st.markdown("## Equities & Sectors")

    # Section 8: Market Breadth
    _anchor("breadth")
    with st.expander("Market Breadth", expanded=True):
        _section_info(_INFO["breadth"])
        spy = query_series(db, "SPY", start=start_date, end=end_date)
        if not spy.empty:
            breadth_col1, breadth_col2 = st.columns(2)

            with breadth_col1:
                breadth_syms = ["SPY", "RSP"]
                breadth_names = {"SPY": "S&P 500 (Cap-Weight)", "RSP": "S&P 500 (Equal-Weight)"}
                breadth_df = query_multiple_series(db, breadth_syms, start=start_date, end=end_date)
                if not breadth_df.empty:
                    breadth_df.columns = [breadth_names.get(c, c) for c in breadth_df.columns]
                    st.plotly_chart(
                        normalized_returns_chart(breadth_df, "SPY vs RSP — Breadth"),
                        width="stretch",
                    )

            with breadth_col2:
                ratio = spy_rsp_spread(db)
                if not ratio.empty:
                    ratio_filtered = ratio.loc[start_date:end_date]
                    fig = time_series_chart(ratio_filtered, "SPY/RSP Ratio (Concentration)", yaxis_title="Ratio")
                    fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    st.plotly_chart(fig, width="stretch")

            if not ratio.empty and len(ratio) >= 22:
                bm1, bm2, bm3 = st.columns(3)
                current_ratio = ratio.iloc[-1]
                month_ago_ratio = ratio.iloc[-22] if len(ratio) >= 22 else ratio.iloc[0]
                ratio_chg = current_ratio - month_ago_ratio
                signal = "Narrowing" if ratio_chg > 0.5 else ("Broadening" if ratio_chg < -0.5 else "Neutral")
                with bm1:
                    st.metric("SPY/RSP Ratio", f"{current_ratio:.1f}")
                with bm2:
                    st.metric("1M Change", f"{ratio_chg:+.2f}")
                with bm3:
                    st.metric("Breadth Signal", signal)

            idx_col1, idx_col2 = st.columns(2)

            with idx_col1:
                index_syms = ["SPY", "QQQ", "DIA", "IWM"]
                index_names = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Russell 2000"}
                idx_df = query_multiple_series(db, index_syms, start=start_date, end=end_date)
                if not idx_df.empty:
                    idx_df.columns = [index_names.get(c, c) for c in idx_df.columns]
                    st.plotly_chart(
                        normalized_returns_chart(idx_df, "Index Performance (Normalized)"),
                        width="stretch",
                    )

            with idx_col2:
                idx_returns = {}
                for sym in index_syms:
                    s = query_series(db, sym, start=start_date, end=end_date)
                    if len(s) >= 2:
                        ret = ((s.iloc[-1] / s.iloc[0]) - 1) * 100
                        idx_returns[index_names.get(sym, sym)] = ret
                if idx_returns:
                    st.plotly_chart(
                        horizontal_bar_chart(idx_returns, "Index Returns (Period)"),
                        width="stretch",
                    )
        else:
            st.info("No market data loaded. Click **Refresh Market Data** in the sidebar.")

    # Section 9: Sector Scorecard
    _anchor("sectors")
    with st.expander("Sector Scorecard", expanded=True):
        _section_info(_INFO["sectors"])
        sec_df = sector_returns(db)
        if not sec_df.empty:
            value_cols = [c for c in ["1W", "1M", "3M", "YTD"] if c in sec_df.columns]
            display_df = sec_df[["Symbol"] + value_cols].sort_values("1M", ascending=False)

            def _color_cell(v):
                if pd.isna(v):
                    return ""
                color = "#22c55e33" if v > 0 else "#ef444433"
                return f"background-color: {color}"

            styled = display_df.style.map(_color_cell, subset=value_cols).format(
                "{:+.1f}%", subset=value_cols, na_rep="—"
            )
            st.dataframe(styled, width="stretch", height=450)

            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                if "1M" in sec_df.columns:
                    month_rets = sec_df["1M"].dropna().to_dict()
                    if month_rets:
                        st.plotly_chart(
                            horizontal_bar_chart(month_rets, "1-Month Sector Returns"),
                            width="stretch",
                        )
            with sc_col2:
                if "YTD" in sec_df.columns:
                    ytd_rets = sec_df["YTD"].dropna().to_dict()
                    if ytd_rets:
                        st.plotly_chart(
                            horizontal_bar_chart(ytd_rets, "YTD Sector Returns"),
                            width="stretch",
                        )

    # ============================================================
    # CROSS-ASSET
    # ============================================================
    st.markdown("## Cross-Asset")

    # Section 10: Commodities & Energy
    _anchor("metals")
    with st.expander("Commodities & Energy", expanded=True):
        _section_info(_INFO["metals"])

        # --- Metric cards ---
        gld = query_series(db, "GLD", start=start_date, end=end_date)
        slv = query_series(db, "SLV", start=start_date, end=end_date)
        wti = query_series(db, "DCOILWTICO", start=start_date, end=end_date)
        brent = query_series(db, "DCOILBRENTEU", start=start_date, end=end_date)
        gsr = gold_silver_ratio(db)
        copx = query_series(db, "COPX", start=start_date, end=end_date)

        met1, met2, met3, met4, met5, met6 = st.columns(6)
        with met1:
            if not wti.empty:
                _wti_val = wti.iloc[-1]
                _wti_badge = "🟢" if 50 <= _wti_val <= 80 else ("🔴" if _wti_val > 100 or _wti_val < 30 else "🟡")
                st.metric(f"{_wti_badge} WTI Crude", f"${_wti_val:.2f}",
                           f"{direction_arrow(wti)}" if len(wti) >= 2 else None)
        with met2:
            if not brent.empty:
                st.metric("Brent Crude", f"${brent.iloc[-1]:.2f}",
                           f"{direction_arrow(brent)}" if len(brent) >= 2 else None)
        with met3:
            if not gld.empty:
                st.metric("Gold (GLD)", f"${gld.iloc[-1]:.2f}",
                           f"{direction_arrow(gld)}" if len(gld) >= 2 else None)
        with met4:
            if not slv.empty:
                st.metric("Silver (SLV)", f"${slv.iloc[-1]:.2f}",
                           f"{direction_arrow(slv)}" if len(slv) >= 2 else None)
        with met5:
            if not gsr.empty:
                _gsr_val = gsr.iloc[-1]
                st.metric(f"{_alert_badge(_gsr_val, _THRESHOLDS['gold_silver'])} Gold/Silver", f"{_gsr_val:.1f}")
        with met6:
            if not copx.empty:
                st.metric("Copper (COPX)", f"${copx.iloc[-1]:.2f}",
                           f"{direction_arrow(copx)}" if len(copx) >= 2 else None)

        # --- Row 1: Oil charts ---
        oil_col1, oil_col2 = st.columns(2)

        with oil_col1:
            oil_df = query_multiple_series(db, ["DCOILWTICO", "DCOILBRENTEU"], start=start_date, end=end_date)
            if not oil_df.empty:
                oil_df.columns = ["WTI Crude", "Brent Crude"]
                st.plotly_chart(
                    time_series_chart(oil_df, "WTI vs Brent Crude Oil", db=db, yaxis_title="$/barrel"),
                    width="stretch",
                )

        with oil_col2:
            ogr = oil_gold_ratio(db)
            if not ogr.empty:
                ogr_filtered = ogr.loc[start_date:end_date]
                st.plotly_chart(
                    time_series_chart(ogr_filtered, "Oil/Gold Ratio (WTI/GLD)", yaxis_title="Ratio"),
                    width="stretch",
                )

        # --- Row 2: Metal charts ---
        metals_col1, metals_col2 = st.columns(2)

        with metals_col1:
            if not gsr.empty:
                gsr_filtered = gsr.loc[start_date:end_date]
                st.plotly_chart(
                    time_series_chart(gsr_filtered, "Gold/Silver Ratio", yaxis_title="Ratio"),
                    width="stretch",
                )

        with metals_col2:
            real = real_rate(db)
            if not real.empty and not gld.empty:
                st.plotly_chart(
                    dual_axis_chart(
                        real.loc[start_date:end_date], gld,
                        "Real Yields vs Gold",
                        y1_title="Real Rate %", y2_title="GLD Price",
                    ),
                    width="stretch",
                )

        # --- Row 3: Commodity comparisons ---
        metals_col3, metals_col4 = st.columns(2)

        with metals_col3:
            slv_copx = query_multiple_series(db, ["SLV", "COPX"], start=start_date, end=end_date)
            if not slv_copx.empty:
                slv_copx.columns = ["Silver (SLV)", "Copper Miners (COPX)"]
                st.plotly_chart(
                    normalized_returns_chart(slv_copx, "Silver vs Copper Miners"),
                    width="stretch",
                )

        with metals_col4:
            comm_df = query_multiple_series(db, ["GLD", "SLV", "USO"], start=start_date, end=end_date)
            if not comm_df.empty:
                comm_df.columns = ["Gold", "Silver", "Crude Oil (USO)"]
                st.plotly_chart(
                    normalized_returns_chart(comm_df, "Commodities Normalized"),
                    width="stretch",
                )

    # Section 11: Cross-Asset Divergence
    _anchor("divergence")
    with st.expander("Cross-Asset Divergence", expanded=True):
        _section_info(_INFO["divergence"])
        ca_col1, ca_col2 = st.columns(2)

        with ca_col1:
            corr = stock_bond_correlation(db)
            if not corr.empty:
                corr_filtered = corr.loc[start_date:end_date]
                fig = time_series_chart(
                    corr_filtered, "SPY-TLT Rolling Correlation (63d)",
                    yaxis_title="Correlation",
                )
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                st.plotly_chart(fig, width="stretch")

        with ca_col2:
            sb_df = query_multiple_series(db, ["SPY", "TLT"], start=start_date, end=end_date)
            if not sb_df.empty:
                sb_df.columns = ["S&P 500", "20Y+ Treasury"]
                st.plotly_chart(
                    normalized_returns_chart(sb_df, "Stocks vs Bonds"),
                    width="stretch",
                )

        ca_col3, ca_col4 = st.columns(2)

        with ca_col3:
            hi_spread = hy_ig_spread(db)
            if not hi_spread.empty:
                st.plotly_chart(
                    time_series_chart(
                        hi_spread.loc[start_date:end_date], "HY-IG OAS Spread",
                        db=db, yaxis_title="bps",
                    ),
                    width="stretch",
                )

        with ca_col4:
            credit_df = query_multiple_series(db, ["HYG", "LQD", "TLT"], start=start_date, end=end_date)
            if not credit_df.empty:
                credit_df.columns = ["High Yield", "Inv. Grade", "20Y+ Treasury"]
                st.plotly_chart(
                    normalized_returns_chart(credit_df, "Credit & Duration ETFs"),
                    width="stretch",
                )

    # Section 12: Geographic Rotation
    _anchor("geo")
    with st.expander("Geographic Rotation", expanded=True):
        _section_info(_INFO["geo"])
        geo_col1, geo_col2 = st.columns(2)

        with geo_col1:
            geo_df = query_multiple_series(db, ["SPY", "EFA", "EEM"], start=start_date, end=end_date)
            if not geo_df.empty:
                geo_df.columns = ["US (SPY)", "Developed ex-US (EFA)", "Emerging (EEM)"]
                st.plotly_chart(
                    normalized_returns_chart(geo_df, "US vs Developed vs Emerging"),
                    width="stretch",
                )

        with geo_col2:
            eu_df = query_multiple_series(db, ["VGK", "EFA"], start=start_date, end=end_date)
            if not eu_df.empty:
                eu_df.columns = ["Europe (VGK)", "Developed ex-US (EFA)"]
                st.plotly_chart(
                    normalized_returns_chart(eu_df, "European Focus"),
                    width="stretch",
                )

        geo_col3, geo_col4 = st.columns(2)

        with geo_col3:
            geo_rets = {}
            for sym, label in [("SPY", "US"), ("EFA", "Developed"), ("EEM", "Emerging"), ("VGK", "Europe")]:
                s = query_series(db, sym, start=start_date, end=end_date)
                if len(s) >= 2:
                    geo_rets[label] = ((s.iloc[-1] / s.iloc[0]) - 1) * 100
            if geo_rets:
                st.plotly_chart(
                    horizontal_bar_chart(geo_rets, "Geographic Returns (Period)"),
                    width="stretch",
                )

        with geo_col4:
            def_df = query_multiple_series(db, ["ITA", "SPY"], start=start_date, end=end_date)
            if not def_df.empty:
                def_df.columns = ["Aerospace & Defense (ITA)", "S&P 500 (SPY)"]
                st.plotly_chart(
                    normalized_returns_chart(def_df, "Defense vs S&P 500"),
                    width="stretch",
                )

    # Section 13: AI & Tech Sub-sectors
    _anchor("tech")
    with st.expander("AI & Tech Sub-sectors", expanded=True):
        _section_info(_INFO["tech"])
        tech_col1, tech_col2 = st.columns(2)

        with tech_col1:
            tech_df = query_multiple_series(db, ["QQQ", "IGV", "SOXX"], start=start_date, end=end_date)
            if not tech_df.empty:
                tech_df.columns = ["Nasdaq 100 (QQQ)", "Software (IGV)", "Semis (SOXX)"]
                st.plotly_chart(
                    normalized_returns_chart(tech_df, "QQQ vs Software vs Semiconductors"),
                    width="stretch",
                )

        with tech_col2:
            nvda_df = query_multiple_series(db, ["NVDA", "QQQ"], start=start_date, end=end_date)
            if not nvda_df.empty:
                nvda_df.columns = ["NVIDIA", "Nasdaq 100 (QQQ)"]
                st.plotly_chart(
                    normalized_returns_chart(nvda_df, "NVIDIA vs QQQ"),
                    width="stretch",
                )

        tech_rets = {}
        for sym, label in [("QQQ", "Nasdaq 100"), ("SOXX", "Semis"), ("IGV", "Software"),
                            ("NVDA", "NVIDIA"), ("XLK", "Tech Sector")]:
            s = query_series(db, sym, start=start_date, end=end_date)
            if len(s) >= 2:
                tech_rets[label] = ((s.iloc[-1] / s.iloc[0]) - 1) * 100
        if tech_rets:
            st.plotly_chart(
                horizontal_bar_chart(tech_rets, "Tech Sub-sector Returns (Period)"),
                width="stretch",
            )

    # ============================================================
    # FX & SENTIMENT
    # ============================================================
    st.markdown("## FX & Sentiment")

    # Section 14: Currencies
    _anchor("currencies")
    with st.expander("Currencies", expanded=True):
        _section_info(_INFO["currencies"])
        dxy = query_series(db, "DX=F", start=start_date, end=end_date)
        if not dxy.empty:
            ccy_col1, ccy_col2 = st.columns(2)

            with ccy_col1:
                dxy.name = "DXY"
                st.plotly_chart(
                    time_series_chart(dxy, "US Dollar Index (DXY)", yaxis_title="Index"),
                    width="stretch",
                )

            with ccy_col2:
                fx_pairs = [
                    ("EURUSD=X", "EUR/USD", False),
                    ("GBPUSD=X", "GBP/USD", False),
                    ("USDJPY=X", "USD/JPY", True),
                ]
                ccy_rows = []
                for symbol, label, usd_direct in fx_pairs:
                    s = query_series(db, symbol, start=start_date, end=end_date)
                    if len(s) >= 22:
                        current = s.iloc[-1]
                        month_ago = s.iloc[-22] if len(s) >= 22 else s.iloc[0]
                        pct_chg = ((current / month_ago) - 1) * 100
                        if usd_direct:
                            usd_signal = "Strengthening" if pct_chg > 0.5 else ("Weakening" if pct_chg < -0.5 else "Stable")
                        else:
                            usd_signal = "Strengthening" if pct_chg < -0.5 else ("Weakening" if pct_chg > 0.5 else "Stable")
                        ccy_rows.append({
                            "Pair": label,
                            "Current": f"{current:.4f}",
                            "1M Change": f"{pct_chg:+.2f}%",
                            "USD vs Pair": usd_signal,
                        })

                if ccy_rows:
                    st.subheader("Currency Strength vs USD")
                    st.dataframe(pd.DataFrame(ccy_rows), hide_index=True, width="stretch")

            btc = query_series(db, "BTC-USD", start=start_date, end=end_date)
            if not btc.empty:
                btc.name = "BTC/USD"
                st.plotly_chart(
                    time_series_chart(btc, "Bitcoin", yaxis_title="USD"),
                    width="stretch",
                )
        else:
            st.info("No currency data loaded. Click **Refresh Market Data** in the sidebar.")

    # Section 15: Reddit Sentiment
    _anchor("sentiment")
    with st.expander("Reddit Sentiment", expanded=True):
        _section_info(_INFO["sentiment"])
        sent = sentiment_summary(db)

        if not sent["available"]:
            if not settings.reddit_client_id:
                st.info(
                    "Reddit sentiment requires API credentials. "
                    "Add `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` to your `.env` file. "
                    "Create an app at https://www.reddit.com/prefs/apps/"
                )
            else:
                st.info("No sentiment data yet. Click **Refresh Sentiment** in the sidebar.")
        else:
            # Row 1: Overall gauge + per-subreddit scores
            sent_col1, sent_col2 = st.columns([2, 3])

            with sent_col1:
                st.plotly_chart(
                    gauge_chart(sent["overall_score"], "Reddit Sentiment", sent["label"]),
                    width="stretch",
                )

            with sent_col2:
                if sent["subreddit_scores"]:
                    sub_cols = st.columns(len(sent["subreddit_scores"]))
                    for i, (name, score) in enumerate(sent["subreddit_scores"].items()):
                        with sub_cols[i]:
                            delta = None
                            if sent["trend"] != 0:
                                delta = f"{sent['trend']:+.1f}"
                            st.metric(name, f"{score:.1f}", delta)

            # Row 2: Sentiment history
            if not sent["history"].empty:
                hist = sent["history"].loc[start_date:end_date]
                if not hist.empty:
                    fig = time_series_chart(hist, "Overall Reddit Sentiment History", yaxis_title="Score (0=Bearish, 100=Bullish)")
                    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    fig.add_hline(y=70, line_dash="dot", line_color="#22c55e", annotation_text="Bullish")
                    fig.add_hline(y=30, line_dash="dot", line_color="#ef4444", annotation_text="Bearish")
                    st.plotly_chart(fig, width="stretch")

            # Row 3: Ticker sentiment ranking
            if settings.sentiment:
                ranking = ticker_sentiment_ranking(
                    db, list(settings.sentiment.tracked_tickers),
                    settings.sentiment.ticker_series_prefix,
                )
                if not ranking.empty:
                    st.subheader("Ticker Sentiment Ranking")
                    st.dataframe(ranking, hide_index=True, width="stretch")

                    # Row 4: Sentiment vs Price for top ticker
                    top_ticker = ranking.iloc[0]["Ticker"]
                    svp = sentiment_vs_price(
                        db, top_ticker,
                        settings.sentiment.ticker_series_prefix,
                    )
                    if not svp.empty:
                        st.plotly_chart(
                            dual_axis_chart(
                                svp["Sentiment"], svp["Price"],
                                f"{top_ticker}: Sentiment vs Price",
                                y1_title="Sentiment Score", y2_title="Price ($)",
                            ),
                            width="stretch",
                        )

    # Footer
    st.divider()
    st.caption(
        f"Data sources: FRED, Yahoo Finance, Reddit | "
        f"{len(meta)} series loaded"
    )


if __name__ == "__main__":
    main()

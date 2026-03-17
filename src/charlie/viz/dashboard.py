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
    gold_silver_ratio, stock_bond_correlation, spy_rsp_spread, sector_returns,
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

# Smooth scrolling CSS
st.markdown("""<style>html { scroll-behavior: smooth; }</style>""", unsafe_allow_html=True)

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
        ("metals", "Metals & Commodities"),
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
                    st.metric("CPI YoY", f"{signals['cpi_yoy']:.1f}%")
            with metric_cols[1]:
                if "credit_spread_zscore" in signals:
                    st.metric("Credit Spread Z", f"{signals['credit_spread_zscore']:.2f}")

        with r1_col2:
            st.plotly_chart(
                gauge_chart(fg["score"], "Fear / Greed", fg["label"]),
                use_container_width=True,
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
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        if not fg["history"].empty:
            hist = fg["history"].loc[start_date:end_date]
            if not hist.empty:
                fig = time_series_chart(hist, "Fear/Greed History", yaxis_title="Score (0=Greed, 100=Fear)")
                fig.add_hline(y=20, line_dash="dot", line_color="#22c55e", annotation_text="Greed")
                fig.add_hline(y=80, line_dash="dot", line_color="#ef4444", annotation_text="Fear")
                fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                st.plotly_chart(fig, use_container_width=True)

        signal_rows = []
        for key, val in regime_data["signals"].items():
            if not key.endswith("_signal"):
                continue
            indicator = key.replace("_signal", "").replace("_", " ").title()
            signal_rows.append({"Indicator": indicator, "Signal": val})
        if signal_rows:
            st.subheader("Regime Signals")
            st.dataframe(pd.DataFrame(signal_rows), hide_index=True, use_container_width=True)

    # Section 2: Economic Calendar
    _anchor("calendar")
    with st.expander("Economic Calendar", expanded=True):
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

            st.dataframe(styled, hide_index=True, use_container_width=True)
        else:
            st.info("No upcoming releases found. Check your FRED API key.")

    # ============================================================
    # FIXED INCOME & POLICY
    # ============================================================
    st.markdown("## Fixed Income & Policy")

    # Section 3: Yield Curve
    _anchor("yield_curve")
    with st.expander("Yield Curve", expanded=True):
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
                    use_container_width=True,
                )

        with yc_col2:
            spread_10y2y = yield_curve_spread(db, "DGS2", "DGS10")
            if not spread_10y2y.empty:
                spread_filtered = spread_10y2y.loc[start_date:end_date]
                fig = time_series_chart(
                    spread_filtered, "10Y-2Y Spread", db=db, yaxis_title="%"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                st.plotly_chart(fig, use_container_width=True)

    # Section 4: Inflation
    _anchor("inflation")
    with st.expander("Inflation", expanded=True):
        inf_col1, inf_col2 = st.columns(2)

        with inf_col1:
            cpi = cpi_yoy(db, "CPIAUCSL")
            core_cpi = cpi_yoy(db, "CPILFESL")
            if not cpi.empty or not core_cpi.empty:
                inf_df = pd.DataFrame({"CPI YoY": cpi, "Core CPI YoY": core_cpi}).loc[start_date:end_date]
                st.plotly_chart(
                    time_series_chart(inf_df, "CPI Year-over-Year %", db=db, yaxis_title="%"),
                    use_container_width=True,
                )

        with inf_col2:
            breakevens = query_multiple_series(db, ["T5YIE", "T10YIE"], start=start_date, end=end_date)
            if not breakevens.empty:
                breakevens.columns = ["5Y Breakeven", "10Y Breakeven"]
                st.plotly_chart(
                    time_series_chart(breakevens, "Breakeven Inflation Rates", db=db, yaxis_title="%"),
                    use_container_width=True,
                )

    # Section 5: Labor Market
    _anchor("labor")
    with st.expander("Labor Market", expanded=True):
        lab_col1, lab_col2 = st.columns(2)

        with lab_col1:
            unrate = query_series(db, "UNRATE", start=start_date, end=end_date)
            if not unrate.empty:
                unrate.name = "Unemployment Rate"
                st.plotly_chart(
                    time_series_chart(unrate, "Unemployment Rate", db=db, yaxis_title="%"),
                    use_container_width=True,
                )

        with lab_col2:
            nfp = payrolls_mom_change(db)
            if not nfp.empty:
                nfp_filtered = nfp.loc[start_date:end_date]
                st.plotly_chart(
                    bar_chart(nfp_filtered, "Nonfarm Payrolls MoM Change (thousands)"),
                    use_container_width=True,
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
                use_container_width=True,
            )

    # Section 6: Credit Deep Dive
    _anchor("credit")
    with st.expander("Credit Deep Dive", expanded=True):
        hy_oas = query_series(db, "BAMLH0A0HYM2")
        ig_oas = query_series(db, "BAMLC0A0CM")
        delinq_all = query_series(db, "DRALACBS")

        if not hy_oas.empty:
            m1, m2_col, m3, m4 = st.columns(4)
            with m1:
                st.metric("HY OAS", f"{hy_oas.iloc[-1]:.0f} bps")
            with m2_col:
                if not ig_oas.empty:
                    spread_val = hy_oas.iloc[-1] - ig_oas.iloc[-1]
                    st.metric("HY-IG Spread", f"{spread_val:.0f} bps")
            with m3:
                if len(hy_oas) >= 252:
                    z = rolling_zscore(hy_oas, 252)
                    st.metric("HY OAS Z-Score", f"{z.iloc[-1]:.2f}")
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
                    use_container_width=True,
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
                        use_container_width=True,
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
                    use_container_width=True,
                )

        with yd_col2:
            delinq_df = query_multiple_series(
                db, ["DRALACBS", "DRTSCILM", "DRSFRMACBS"], start=start_date, end=end_date
            )
            if not delinq_df.empty:
                delinq_df.columns = ["All Loans", "C&I Loans", "SF Residential"]
                st.plotly_chart(
                    time_series_chart(delinq_df, "Loan Delinquency Rates", db=db, yaxis_title="%"),
                    use_container_width=True,
                )

        cv_col1, cv_col2 = st.columns(2)

        with cv_col1:
            try:
                impulse = credit_impulse(db)
                if not impulse.empty:
                    impulse_filtered = impulse.loc[start_date:end_date]
                    fig = bar_chart(impulse_filtered, "Credit Impulse (Total Loans YoY %)")
                    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        with cv_col2:
            vix = query_series(db, "VIXCLS", start=start_date, end=end_date)
            if not vix.empty:
                vix.name = "VIX"
                st.plotly_chart(
                    time_series_chart(vix, "VIX (Volatility Index)", db=db),
                    use_container_width=True,
                )

        fc = query_multiple_series(db, ["NFCI", "STLFSI2"], start=start_date, end=end_date)
        if not fc.empty:
            fc.columns = ["Chicago Fed NFCI", "St. Louis Fed Stress"]
            st.plotly_chart(
                time_series_chart(fc, "Financial Conditions Indices", db=db),
                use_container_width=True,
            )

    # Section 7: Monetary Policy
    _anchor("monetary")
    with st.expander("Monetary Policy", expanded=True):
        mp_col1, mp_col2 = st.columns(2)

        with mp_col1:
            ff = query_series(db, "DFF", start=start_date, end=end_date)
            if not ff.empty:
                ff.name = "Fed Funds Rate"
                st.plotly_chart(
                    time_series_chart(ff, "Effective Fed Funds Rate", db=db, yaxis_title="%"),
                    use_container_width=True,
                )

        with mp_col2:
            m2 = m2_yoy(db)
            if not m2.empty:
                m2_filtered = m2.loc[start_date:end_date]
                m2_filtered.name = "M2 YoY %"
                fig = time_series_chart(m2_filtered, "M2 Money Supply YoY Growth", db=db, yaxis_title="%")
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                st.plotly_chart(fig, use_container_width=True)

        walcl = query_series(db, "WALCL", start=start_date, end=end_date)
        if not walcl.empty:
            walcl.name = "Fed Total Assets ($M)"
            st.plotly_chart(
                time_series_chart(walcl, "Fed Balance Sheet Total Assets", db=db, yaxis_title="$ Millions"),
                use_container_width=True,
            )

    # ============================================================
    # EQUITIES & SECTORS
    # ============================================================
    st.markdown("## Equities & Sectors")

    # Section 8: Market Breadth
    _anchor("breadth")
    with st.expander("Market Breadth", expanded=True):
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
                        use_container_width=True,
                    )

            with breadth_col2:
                ratio = spy_rsp_spread(db)
                if not ratio.empty:
                    ratio_filtered = ratio.loc[start_date:end_date]
                    fig = time_series_chart(ratio_filtered, "SPY/RSP Ratio (Concentration)", yaxis_title="Ratio")
                    fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                    st.plotly_chart(fig, use_container_width=True)

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
                        use_container_width=True,
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
                        use_container_width=True,
                    )
        else:
            st.info("No market data loaded. Click **Refresh Market Data** in the sidebar.")

    # Section 9: Sector Scorecard
    _anchor("sectors")
    with st.expander("Sector Scorecard", expanded=True):
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
            st.dataframe(styled, use_container_width=True, height=450)

            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                if "1M" in sec_df.columns:
                    month_rets = sec_df["1M"].dropna().to_dict()
                    if month_rets:
                        st.plotly_chart(
                            horizontal_bar_chart(month_rets, "1-Month Sector Returns"),
                            use_container_width=True,
                        )
            with sc_col2:
                if "YTD" in sec_df.columns:
                    ytd_rets = sec_df["YTD"].dropna().to_dict()
                    if ytd_rets:
                        st.plotly_chart(
                            horizontal_bar_chart(ytd_rets, "YTD Sector Returns"),
                            use_container_width=True,
                        )

    # ============================================================
    # CROSS-ASSET
    # ============================================================
    st.markdown("## Cross-Asset")

    # Section 10: Metals & Commodities
    _anchor("metals")
    with st.expander("Metals & Commodities", expanded=True):
        gld = query_series(db, "GLD", start=start_date, end=end_date)
        if not gld.empty:
            met1, met2, met3, met4 = st.columns(4)
            with met1:
                st.metric("Gold (GLD)", f"${gld.iloc[-1]:.2f}",
                           f"{direction_arrow(gld)}" if len(gld) >= 2 else None)
            slv = query_series(db, "SLV", start=start_date, end=end_date)
            with met2:
                if not slv.empty:
                    st.metric("Silver (SLV)", f"${slv.iloc[-1]:.2f}",
                               f"{direction_arrow(slv)}" if len(slv) >= 2 else None)
            gsr = gold_silver_ratio(db)
            with met3:
                if not gsr.empty:
                    st.metric("Gold/Silver Ratio", f"{gsr.iloc[-1]:.2f}")
            copx = query_series(db, "COPX", start=start_date, end=end_date)
            with met4:
                if not copx.empty:
                    st.metric("Copper (COPX)", f"${copx.iloc[-1]:.2f}",
                               f"{direction_arrow(copx)}" if len(copx) >= 2 else None)

            metals_col1, metals_col2 = st.columns(2)

            with metals_col1:
                if not gsr.empty:
                    gsr_filtered = gsr.loc[start_date:end_date]
                    st.plotly_chart(
                        time_series_chart(gsr_filtered, "Gold/Silver Ratio", yaxis_title="Ratio"),
                        use_container_width=True,
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
                        use_container_width=True,
                    )

            metals_col3, metals_col4 = st.columns(2)

            with metals_col3:
                slv_copx = query_multiple_series(db, ["SLV", "COPX"], start=start_date, end=end_date)
                if not slv_copx.empty:
                    slv_copx.columns = ["Silver (SLV)", "Copper Miners (COPX)"]
                    st.plotly_chart(
                        normalized_returns_chart(slv_copx, "Silver vs Copper Miners"),
                        use_container_width=True,
                    )

            with metals_col4:
                comm_df = query_multiple_series(db, ["GLD", "SLV", "USO"], start=start_date, end=end_date)
                if not comm_df.empty:
                    comm_df.columns = ["Gold", "Silver", "Crude Oil"]
                    st.plotly_chart(
                        normalized_returns_chart(comm_df, "Commodities Normalized"),
                        use_container_width=True,
                    )

    # Section 11: Cross-Asset Divergence
    _anchor("divergence")
    with st.expander("Cross-Asset Divergence", expanded=True):
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
                st.plotly_chart(fig, use_container_width=True)

        with ca_col2:
            sb_df = query_multiple_series(db, ["SPY", "TLT"], start=start_date, end=end_date)
            if not sb_df.empty:
                sb_df.columns = ["S&P 500", "20Y+ Treasury"]
                st.plotly_chart(
                    normalized_returns_chart(sb_df, "Stocks vs Bonds"),
                    use_container_width=True,
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
                    use_container_width=True,
                )

        with ca_col4:
            credit_df = query_multiple_series(db, ["HYG", "LQD", "TLT"], start=start_date, end=end_date)
            if not credit_df.empty:
                credit_df.columns = ["High Yield", "Inv. Grade", "20Y+ Treasury"]
                st.plotly_chart(
                    normalized_returns_chart(credit_df, "Credit & Duration ETFs"),
                    use_container_width=True,
                )

    # Section 12: Geographic Rotation
    _anchor("geo")
    with st.expander("Geographic Rotation", expanded=True):
        geo_col1, geo_col2 = st.columns(2)

        with geo_col1:
            geo_df = query_multiple_series(db, ["SPY", "EFA", "EEM"], start=start_date, end=end_date)
            if not geo_df.empty:
                geo_df.columns = ["US (SPY)", "Developed ex-US (EFA)", "Emerging (EEM)"]
                st.plotly_chart(
                    normalized_returns_chart(geo_df, "US vs Developed vs Emerging"),
                    use_container_width=True,
                )

        with geo_col2:
            eu_df = query_multiple_series(db, ["VGK", "EFA"], start=start_date, end=end_date)
            if not eu_df.empty:
                eu_df.columns = ["Europe (VGK)", "Developed ex-US (EFA)"]
                st.plotly_chart(
                    normalized_returns_chart(eu_df, "European Focus"),
                    use_container_width=True,
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
                    use_container_width=True,
                )

        with geo_col4:
            def_df = query_multiple_series(db, ["ITA", "SPY"], start=start_date, end=end_date)
            if not def_df.empty:
                def_df.columns = ["Aerospace & Defense (ITA)", "S&P 500 (SPY)"]
                st.plotly_chart(
                    normalized_returns_chart(def_df, "Defense vs S&P 500"),
                    use_container_width=True,
                )

    # Section 13: AI & Tech Sub-sectors
    _anchor("tech")
    with st.expander("AI & Tech Sub-sectors", expanded=True):
        tech_col1, tech_col2 = st.columns(2)

        with tech_col1:
            tech_df = query_multiple_series(db, ["QQQ", "IGV", "SOXX"], start=start_date, end=end_date)
            if not tech_df.empty:
                tech_df.columns = ["Nasdaq 100 (QQQ)", "Software (IGV)", "Semis (SOXX)"]
                st.plotly_chart(
                    normalized_returns_chart(tech_df, "QQQ vs Software vs Semiconductors"),
                    use_container_width=True,
                )

        with tech_col2:
            nvda_df = query_multiple_series(db, ["NVDA", "QQQ"], start=start_date, end=end_date)
            if not nvda_df.empty:
                nvda_df.columns = ["NVIDIA", "Nasdaq 100 (QQQ)"]
                st.plotly_chart(
                    normalized_returns_chart(nvda_df, "NVIDIA vs QQQ"),
                    use_container_width=True,
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
                use_container_width=True,
            )

    # ============================================================
    # FX & SENTIMENT
    # ============================================================
    st.markdown("## FX & Sentiment")

    # Section 14: Currencies
    _anchor("currencies")
    with st.expander("Currencies", expanded=True):
        dxy = query_series(db, "DX=F", start=start_date, end=end_date)
        if not dxy.empty:
            ccy_col1, ccy_col2 = st.columns(2)

            with ccy_col1:
                dxy.name = "DXY"
                st.plotly_chart(
                    time_series_chart(dxy, "US Dollar Index (DXY)", yaxis_title="Index"),
                    use_container_width=True,
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
                    st.dataframe(pd.DataFrame(ccy_rows), hide_index=True, use_container_width=True)

            btc = query_series(db, "BTC-USD", start=start_date, end=end_date)
            if not btc.empty:
                btc.name = "BTC/USD"
                st.plotly_chart(
                    time_series_chart(btc, "Bitcoin", yaxis_title="USD"),
                    use_container_width=True,
                )
        else:
            st.info("No currency data loaded. Click **Refresh Market Data** in the sidebar.")

    # Section 15: Reddit Sentiment
    _anchor("sentiment")
    with st.expander("Reddit Sentiment", expanded=True):
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
                    use_container_width=True,
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
                    st.plotly_chart(fig, use_container_width=True)

            # Row 3: Ticker sentiment ranking
            if settings.sentiment:
                ranking = ticker_sentiment_ranking(
                    db, list(settings.sentiment.tracked_tickers),
                    settings.sentiment.ticker_series_prefix,
                )
                if not ranking.empty:
                    st.subheader("Ticker Sentiment Ranking")
                    st.dataframe(ranking, hide_index=True, use_container_width=True)

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
                            use_container_width=True,
                        )

    # Footer
    st.divider()
    st.caption(
        f"Data sources: FRED, Yahoo Finance, Reddit | "
        f"{len(meta)} series loaded"
    )


if __name__ == "__main__":
    main()

"""Derived macro indicators from raw FRED data."""
import numpy as np
import pandas as pd

from charlie.storage.db import Database
from charlie.storage.models import query_series, query_multiple_series


def yield_curve_spread(
    db: Database, short: str = "DGS2", long: str = "DGS10"
) -> pd.Series:
    """Compute long-short yield spread (e.g., 10Y-2Y)."""
    df = query_multiple_series(db, [short, long])
    spread = df[long] - df[short]
    spread.name = f"{long}_{short}_spread"
    return spread.dropna()


def yield_curve_shape(db: Database, date: str | None = None) -> pd.Series:
    """Get the full yield curve for a given date (or most recent)."""
    tenors = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30"]
    tenor_labels = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]

    df = query_multiple_series(db, tenors)
    if df.empty:
        return pd.Series(dtype=float)

    if date:
        # Find nearest available date
        target = pd.Timestamp(date)
        idx = df.index.get_indexer([target], method="ffill")[0]
        if idx < 0:
            idx = 0
        row = df.iloc[idx]
    else:
        row = df.iloc[-1]

    values = [row.get(t) for t in tenors]
    return pd.Series(values, index=tenor_labels, name=str(row.name)[:10])


def real_rate(
    db: Database, nominal: str = "DGS10", inflation: str = "T10YIE"
) -> pd.Series:
    """Real rate = nominal yield - breakeven inflation."""
    df = query_multiple_series(db, [nominal, inflation])
    real = df[nominal] - df[inflation]
    real.name = "real_rate_10y"
    return real.dropna()


def cpi_yoy(db: Database, series_id: str = "CPIAUCSL") -> pd.Series:
    """Year-over-year percentage change in CPI."""
    s = query_series(db, series_id)
    yoy = s.pct_change(periods=12, fill_method=None) * 100
    yoy.name = f"{series_id}_yoy"
    return yoy.dropna()


def payrolls_mom_change(db: Database) -> pd.Series:
    """Month-over-month change in nonfarm payrolls (thousands)."""
    s = query_series(db, "PAYEMS")
    change = s.diff()
    change.name = "payrolls_mom"
    return change.dropna()


def m2_yoy(db: Database) -> pd.Series:
    """Year-over-year percentage change in M2 money supply."""
    s = query_series(db, "M2SL")
    yoy = s.pct_change(periods=12, fill_method=None) * 100
    yoy.name = "m2_yoy"
    return yoy.dropna()


def hy_ig_spread(db: Database) -> pd.Series:
    """HY OAS minus IG OAS — widening means risk aversion increasing."""
    df = query_multiple_series(db, ["BAMLH0A0HYM2", "BAMLC0A0CM"])
    spread = df["BAMLH0A0HYM2"] - df["BAMLC0A0CM"]
    spread.name = "hy_ig_spread"
    return spread.dropna()


def credit_impulse(db: Database) -> pd.Series:
    """Year-over-year percentage change in total loans — credit expansion/contraction."""
    s = query_series(db, "TOTLL")
    yoy = s.pct_change(periods=12, fill_method=None) * 100
    yoy.name = "credit_impulse"
    return yoy.dropna()


def fed_balance_sheet_change(db: Database) -> pd.Series:
    """Year-over-year percentage change in Fed balance sheet."""
    s = query_series(db, "WALCL")
    yoy = s.pct_change(periods=52, fill_method=None) * 100  # weekly data
    yoy.name = "walcl_yoy"
    return yoy.dropna()


# ── Crown Macro Letter derived metrics ──────────────────────────


def gold_silver_ratio(db: Database) -> pd.Series:
    """GLD / SLV price ratio. Rising = gold outperforming silver (risk-off)."""
    df = query_multiple_series(db, ["GLD", "SLV"])
    ratio = df["GLD"] / df["SLV"]
    ratio.name = "gold_silver_ratio"
    return ratio.dropna()


def gold_copper_ratio(db: Database) -> pd.Series:
    """GLD / COPX ratio. Rising = safe haven outperforming cyclical metal (risk-off)."""
    df = query_multiple_series(db, ["GLD", "COPX"])
    if df.empty or "GLD" not in df.columns or "COPX" not in df.columns:
        return pd.Series(dtype=float)
    ratio = df["GLD"] / df["COPX"]
    ratio.name = "gold_copper_ratio"
    return ratio.dropna()


def gold_real_yield_divergence(db: Database, window: int = 63) -> pd.Series:
    """Rolling correlation of gold returns vs real yield changes.

    Normally negative (gold falls when real rates rise). When correlation
    turns positive, gold is rising DESPITE rising real rates = stress signal.
    """
    rr = real_rate(db)
    gld = query_series(db, "GLD")
    if rr.empty or gld.empty:
        return pd.Series(dtype=float)

    gld_ret = gld.pct_change().dropna()
    rr_chg = rr.diff().dropna()

    # Align
    aligned = pd.DataFrame({"gld_ret": gld_ret, "rr_chg": rr_chg}).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    corr = aligned["gld_ret"].rolling(window, min_periods=30).corr(aligned["rr_chg"])
    corr.name = "gold_real_yield_corr"
    return corr.dropna()


def gold_momentum(db: Database) -> pd.Series:
    """GLD 50d/200d MA ratio. Rising gold trend = safe haven demand."""
    gld = query_series(db, "GLD")
    if gld.empty or len(gld) < 200:
        return pd.Series(dtype=float)
    ma50 = gld.rolling(50).mean()
    ma200 = gld.rolling(200).mean()
    momentum = ma50 / ma200 * 100
    momentum.name = "gold_momentum"
    return momentum.dropna()


def oil_gold_ratio(db: Database) -> pd.Series:
    """WTI Crude / GLD ratio. Rising = growth/risk-on, falling = stagflation risk."""
    df = query_multiple_series(db, ["DCOILWTICO", "GLD"])
    if df.empty or "DCOILWTICO" not in df.columns or "GLD" not in df.columns:
        return pd.Series(dtype=float)
    ratio = df["DCOILWTICO"] / df["GLD"]
    ratio.name = "oil_gold_ratio"
    return ratio.dropna()


def brent_wti_spread(db: Database) -> pd.Series:
    """Brent - WTI spread. Widening = international supply tightness."""
    df = query_multiple_series(db, ["DCOILBRENTEU", "DCOILWTICO"])
    if df.empty or "DCOILBRENTEU" not in df.columns or "DCOILWTICO" not in df.columns:
        return pd.Series(dtype=float)
    spread = df["DCOILBRENTEU"] - df["DCOILWTICO"]
    spread.name = "brent_wti_spread"
    return spread.dropna()


def stock_bond_correlation(db: Database, window: int = 63) -> pd.Series:
    """Rolling correlation of SPY vs TLT daily returns (63d ≈ 3 months).

    Positive = stocks and bonds moving together (unusual, risk-off regime).
    Negative = normal diversification benefit.
    """
    df = query_multiple_series(db, ["SPY", "TLT"])
    returns = df.pct_change().dropna()
    corr = returns["SPY"].rolling(window).corr(returns["TLT"])
    corr.name = "spy_tlt_correlation"
    return corr.dropna()


def spy_rsp_spread(db: Database) -> pd.Series:
    """SPY/RSP normalized ratio — breadth signal.

    Rising = cap-weighted outperforming equal-weight (narrowing breadth,
    concentration risk). Falling = breadth broadening (healthier market).
    """
    df = query_multiple_series(db, ["SPY", "RSP"])
    df = df.dropna()
    if df.empty:
        return pd.Series(dtype=float, name="spy_rsp_ratio")
    first = df.index[0]
    spy_norm = df["SPY"] / df["SPY"].loc[first] * 100
    rsp_norm = df["RSP"] / df["RSP"].loc[first] * 100
    ratio = spy_norm / rsp_norm * 100
    ratio.name = "spy_rsp_ratio"
    return ratio


def sector_returns(db: Database) -> pd.DataFrame:
    """Multi-period return table for all 11 GICS sector ETFs.

    Returns DataFrame indexed by sector name with columns: Symbol, 1W, 1M, 3M, YTD.
    """
    sectors = {
        "XLF": "Financials", "XLE": "Energy", "XLK": "Technology",
        "XLV": "Healthcare", "XLU": "Utilities", "XLB": "Materials",
        "XLI": "Industrials", "XLY": "Cons. Discr.",
        "XLP": "Cons. Staples", "XLC": "Comm. Svcs.", "XLRE": "Real Estate",
    }
    periods = {"1W": 5, "1M": 21, "3M": 63}

    rows = []
    for sym, name in sectors.items():
        s = query_series(db, sym)
        if s.empty or len(s) < 2:
            continue
        row = {"Sector": name, "Symbol": sym}
        for label, n in periods.items():
            if len(s) >= n + 1:
                row[label] = ((s.iloc[-1] / s.iloc[-n - 1]) - 1) * 100
            else:
                row[label] = None
        # YTD
        year_start = s.index[s.index >= f"{s.index[-1].year}-01-01"]
        if not year_start.empty:
            row["YTD"] = ((s.iloc[-1] / s.loc[year_start[0]]) - 1) * 100
        else:
            row["YTD"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("Sector")
    return df


# ── COT Positioning ──────────────────────────────────────────

# Contract prefixes and display names
COT_CONTRACTS = {
    "COT_ES": "S&P 500",
    "COT_NQ": "Nasdaq",
    "COT_GC": "Gold",
    "COT_CL": "Crude Oil",
    "COT_ZN": "10Y Treasury",
    "COT_EC": "Euro FX",
}


def cot_summary_table(db: Database, window: int = 52) -> pd.DataFrame:
    """Build a summary table of COT positioning across all tracked contracts.

    Returns DataFrame with columns: Contract, Net %, Z-Score, Direction.
    """
    from charlie.analysis.stats import rolling_zscore, direction_arrow

    rows = []
    for prefix, name in COT_CONTRACTS.items():
        pct = query_series(db, f"{prefix}_PCT")
        if pct.empty:
            continue

        current = pct.iloc[-1]
        z = rolling_zscore(pct, window)
        z_val = z.iloc[-1] if not z.empty else 0.0
        arrow = direction_arrow(pct) if len(pct) >= 2 else ""

        rows.append({
            "Contract": name,
            "Net %": round(current, 1),
            "Z-Score": round(z_val, 2),
            "Direction": arrow,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── ETF Flows ────────────────────────────────────────────────

# Category display names
_FLOW_CATEGORIES = {
    "equity_us": "US Equity",
    "fixed_income": "Fixed Income",
    "commodities": "Commodities",
    "international": "International",
    "sectors": "Sectors",
}


def etf_flow_summary(db: Database) -> pd.DataFrame:
    """Build a summary of ETF dollar volume activity per ETF.

    Uses dollar volume (price × volume) as a flow proxy. Computes the
    recent dollar volume vs 20-day average to identify demand surges.

    Returns DataFrame with columns: Symbol, Name, Category,
    Latest ($M), 20d Avg ($M), vs Avg (%), 1W Cum ($M), 1M Cum ($M).
    """
    try:
        from charlie.config import get_settings
        settings = get_settings()
    except Exception:
        return pd.DataFrame()

    rows = []
    for etf in settings.etf_flow_tickers:
        dvol = query_series(db, f"FLOW_{etf.symbol}_DVOL")
        davg = query_series(db, f"FLOW_{etf.symbol}_DAVG")
        cum = query_series(db, f"FLOW_{etf.symbol}_CUM")
        if dvol.empty:
            continue

        latest = float(dvol.iloc[-1])
        avg20 = float(davg.iloc[-1]) if not davg.empty else 0
        vs_avg = ((latest / avg20) - 1) * 100 if avg20 > 0 else 0

        def _sum_last_n(s, n):
            if s.empty:
                return 0.0
            return float(s.iloc[-n:].sum()) if len(s) >= n else float(s.sum())

        # Net volume flow (deviation from 20d avg) summed over periods
        deviation = dvol - davg if not davg.empty else dvol * 0
        deviation = deviation.dropna()

        rows.append({
            "Symbol": etf.symbol,
            "Name": etf.name,
            "Category": _FLOW_CATEGORIES.get(etf.category, etf.category),
            "Latest ($M)": round(latest, 0),
            "20d Avg ($M)": round(avg20, 0),
            "vs Avg (%)": round(vs_avg, 1),
            "1W Net ($M)": round(_sum_last_n(deviation, 5), 0),
            "1M Net ($M)": round(_sum_last_n(deviation, 21), 0),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def etf_flow_by_category(db: Database) -> pd.DataFrame:
    """Aggregate ETF net volume flows by asset class category.

    Returns DataFrame with columns: Category, 1W Net ($M), 1M Net ($M).
    """
    summary = etf_flow_summary(db)
    if summary.empty:
        return pd.DataFrame()

    grouped = summary.groupby("Category")[["1W Net ($M)", "1M Net ($M)"]].sum()
    return grouped.reset_index().sort_values("1M Net ($M)", ascending=False)


# ── Crown Macro Signal Functions ────────────────────────────────


def vix_vs_realized_vol(db: Database) -> dict:
    """VIX minus realized volatility — complacency when negative.

    Returns dict with: vix, realized_vol, premium, premium_zscore, signal,
    history (DataFrame with VIX, Realized, Premium columns).
    """
    vix = query_series(db, "VIXCLS")
    spy = query_series(db, "SPY")

    if vix.empty or spy.empty:
        return {"available": False}

    # Realized vol: 21-day rolling std of daily returns, annualized
    spy_ret = spy.pct_change().dropna()
    realized = spy_ret.rolling(21, min_periods=15).std() * np.sqrt(252) * 100
    realized.name = "Realized Vol"

    # Align on common dates
    df = pd.DataFrame({"VIX": vix, "Realized": realized}).dropna()
    if df.empty or len(df) < 30:
        return {"available": False}

    df["Premium"] = df["VIX"] - df["Realized"]

    current_vix = float(df["VIX"].iloc[-1])
    current_rv = float(df["Realized"].iloc[-1])
    premium = float(df["Premium"].iloc[-1])

    # Z-score of premium over 252-day window
    from charlie.analysis.stats import rolling_zscore
    prem_z = rolling_zscore(df["Premium"], 252)
    z_val = float(prem_z.iloc[-1]) if not prem_z.empty else 0.0

    # Signal classification
    if premium < 0:
        signal = "complacency"
    elif premium < 5:
        signal = "normal"
    elif premium < 8:
        signal = "elevated"
    else:
        signal = "fear_overshoot"

    return {
        "available": True,
        "vix": current_vix,
        "realized_vol": current_rv,
        "premium": premium,
        "premium_zscore": z_val,
        "signal": signal,
        "history": df,
    }


def breadth_above_200d_ma(db: Database) -> dict:
    """Percentage of tracked ETFs above their 200-day moving average.

    Uses 15 ETFs already in DB as a proxy for market breadth.

    Returns dict with: current_pct, above_count, total_count, detail (list),
    history (Series), signal.
    """
    etfs = {
        "XLF": "Financials", "XLE": "Energy", "XLK": "Technology",
        "XLV": "Healthcare", "XLU": "Utilities", "XLB": "Materials",
        "XLI": "Industrials", "XLY": "Cons. Discr.",
        "XLP": "Cons. Staples", "XLC": "Comm. Svcs.", "XLRE": "Real Estate",
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
        "RSP": "S&P Equal Wt",
    }

    detail = []
    # For historical time series, track daily above/below per ETF
    above_ts: dict[str, pd.Series] = {}

    for sym, name in etfs.items():
        s = query_series(db, sym)
        if s.empty or len(s) < 200:
            continue
        ma200 = s.rolling(200).mean()
        current_price = float(s.iloc[-1])
        current_ma = float(ma200.iloc[-1])
        pct_from_ma = ((current_price / current_ma) - 1) * 100

        is_above = current_price > current_ma
        detail.append({
            "Symbol": sym,
            "Name": name,
            "Price": current_price,
            "200d MA": current_ma,
            "% from MA": round(pct_from_ma, 1),
            "Above": is_above,
        })

        # Historical: 1 if above 200d MA, 0 if below
        above_series = (s > ma200).astype(float)
        above_ts[sym] = above_series

    if not detail:
        return {"available": False}

    above_count = sum(1 for d in detail if d["Above"])
    total_count = len(detail)
    current_pct = (above_count / total_count) * 100

    # Historical time series of % above 200d MA
    if above_ts:
        above_df = pd.DataFrame(above_ts).dropna()
        if not above_df.empty:
            history = above_df.mean(axis=1) * 100
            history.name = "pct_above_200d"
        else:
            history = pd.Series(dtype=float)
    else:
        history = pd.Series(dtype=float)

    # Signal classification
    if current_pct >= 80:
        signal = "strong"
    elif current_pct >= 60:
        signal = "healthy"
    elif current_pct >= 40:
        signal = "weakening"
    else:
        signal = "critical"

    return {
        "available": True,
        "current_pct": current_pct,
        "above_count": above_count,
        "total_count": total_count,
        "detail": sorted(detail, key=lambda d: d["% from MA"], reverse=True),
        "history": history,
        "signal": signal,
    }


def exhaustion_signal(
    db: Database,
    api_key: str,
    releases: list | tuple,
) -> dict:
    """'Good News Stops Working' — market exhaustion detection.

    Checks SPY reaction to high-importance economic releases. If markets
    sell off on good news, the move is exhausted.

    Returns dict with: score, count_negative, count_total,
    events (list of {date, name, spy_return}), signal.
    """
    from charlie.analysis.calendar import get_past_release_dates

    if not api_key:
        return {"available": False}

    # Get past release dates for high-importance releases
    high_releases = [r for r in releases if r.importance == "high"]
    if not high_releases:
        return {"available": False}

    past_events = get_past_release_dates(api_key, high_releases, days_back=180)

    spy = query_series(db, "SPY")
    if spy.empty or not past_events:
        return {"available": False}

    # Compute SPY daily returns
    spy_ret = spy.pct_change() * 100

    events = []
    for ev in past_events[-15:]:  # Last 15 events
        dt = pd.Timestamp(ev["date"])
        # Find nearest trading day return
        idx = spy_ret.index.get_indexer([dt], method="nearest")[0]
        if idx >= 0 and idx < len(spy_ret):
            ret = float(spy_ret.iloc[idx])
            events.append({
                "date": ev["date"],
                "name": ev["name"],
                "spy_return": round(ret, 2),
            })

    if not events:
        return {"available": False}

    # Score: fraction of last 10 releases with negative SPY returns
    recent = events[-10:] if len(events) >= 10 else events
    count_neg = sum(1 for e in recent if e["spy_return"] < 0)
    count_total = len(recent)
    score = count_neg / count_total if count_total > 0 else 0.0

    if score < 0.4:
        signal = "normal"
    elif score < 0.6:
        signal = "caution"
    else:
        signal = "exhaustion"

    return {
        "available": True,
        "score": score,
        "count_negative": count_neg,
        "count_total": count_total,
        "events": events,
        "signal": signal,
    }


# COT-to-price mapping for crowded trade detection
COT_PRICE_MAP = {
    "COT_ES": "SPY",
    "COT_NQ": "QQQ",
    "COT_GC": "GLD",
    "COT_CL": "USO",
    "COT_ZN": "TLT",
    "COT_EC": "EFA",  # Euro FX proxy
}


def crowded_trade_unwind(db: Database, z_threshold: float = 2.0) -> list[dict]:
    """Detect crowded trade unwinding — extreme positioning + adverse price action.

    Compound signal: COT z-score > threshold AND price below 20d MA (for longs)
    or COT z-score < -threshold AND price above 20d MA (for shorts).

    Returns list of dicts per contract with: contract, prefix, z_score, direction,
    price_vs_ma_pct, unwinding, signal_type.
    """
    from charlie.analysis.stats import rolling_zscore

    results = []
    for prefix, name in COT_CONTRACTS.items():
        pct = query_series(db, f"{prefix}_PCT")
        price_sym = COT_PRICE_MAP.get(prefix)
        if pct.empty or not price_sym:
            continue

        price = query_series(db, price_sym)
        if price.empty or len(price) < 20:
            continue

        z = rolling_zscore(pct, 52)
        if z.empty:
            continue

        z_val = float(z.iloc[-1])
        current_price = float(price.iloc[-1])
        ma20 = float(price.rolling(20).mean().iloc[-1])
        price_vs_ma_pct = ((current_price / ma20) - 1) * 100

        unwinding = False
        signal_type = "none"

        if z_val > z_threshold and current_price < ma20:
            unwinding = True
            signal_type = "crowded_long_unwinding"
        elif z_val < -z_threshold and current_price > ma20:
            unwinding = True
            signal_type = "crowded_short_squeeze"

        results.append({
            "contract": name,
            "prefix": prefix,
            "price_symbol": price_sym,
            "z_score": round(z_val, 2),
            "direction": "long" if z_val > 0 else "short",
            "price_vs_ma_pct": round(price_vs_ma_pct, 1),
            "unwinding": unwinding,
            "signal_type": signal_type,
        })

    return results


# Sector ETF symbols for rank reversal detection
_SECTOR_ETFS = {
    "XLF": "Financials", "XLE": "Energy", "XLK": "Technology",
    "XLV": "Healthcare", "XLU": "Utilities", "XLB": "Materials",
    "XLI": "Industrials", "XLY": "Cons. Discr.",
    "XLP": "Cons. Staples", "XLC": "Comm. Svcs.", "XLRE": "Real Estate",
}


def sector_rank_reversal(db: Database, window: int = 63) -> dict:
    """Detect 'Factory Reset' — major sector rank reversals over a quarter.

    Compares sector return rankings now vs one quarter ago.
    Flags sectors with |rank_change| >= 5.

    Returns dict with: reversals (list), has_reversal (bool),
    current_rankings (dict), previous_rankings (dict), details (list).
    """
    current_rets = {}
    prev_rets = {}

    for sym in _SECTOR_ETFS:
        s = query_series(db, sym)
        if s.empty or len(s) < window * 2 + 1:
            continue

        # Current quarter return (last 63 trading days)
        current_rets[sym] = ((s.iloc[-1] / s.iloc[-window - 1]) - 1) * 100

        # Previous quarter return (63d before that)
        prev_rets[sym] = ((s.iloc[-window - 1] / s.iloc[-2 * window - 1]) - 1) * 100

    if len(current_rets) < 5:
        return {"available": False}

    # Rank (1=best, N=worst)
    curr_sorted = sorted(current_rets.items(), key=lambda x: x[1], reverse=True)
    prev_sorted = sorted(prev_rets.items(), key=lambda x: x[1], reverse=True)

    curr_rank = {sym: i + 1 for i, (sym, _) in enumerate(curr_sorted)}
    prev_rank = {sym: i + 1 for i, (sym, _) in enumerate(prev_sorted)}

    details = []
    reversals = []
    for sym in current_rets:
        name = _SECTOR_ETFS[sym]
        cr = curr_rank.get(sym, 0)
        pr = prev_rank.get(sym, 0)
        rank_change = pr - cr  # positive = improved (moved up)

        entry = {
            "symbol": sym,
            "name": name,
            "current_rank": cr,
            "previous_rank": pr,
            "rank_change": rank_change,
            "current_return": round(current_rets[sym], 1),
            "previous_return": round(prev_rets.get(sym, 0), 1),
        }
        details.append(entry)

        if abs(rank_change) >= 5:
            reversals.append(entry)

    details.sort(key=lambda d: d["current_rank"])

    return {
        "available": True,
        "reversals": reversals,
        "has_reversal": len(reversals) > 0,
        "current_rankings": curr_rank,
        "previous_rankings": prev_rank,
        "details": details,
    }

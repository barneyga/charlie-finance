"""Derived macro indicators from raw FRED data."""
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

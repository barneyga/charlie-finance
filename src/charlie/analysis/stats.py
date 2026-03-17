"""Statistical utilities for macro analysis."""
import pandas as pd


def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """How many std devs the current value is from its rolling mean."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std
    z.name = f"{series.name}_zscore"
    return z


def percentile_rank(series: pd.Series, window: int = 1260) -> pd.Series:
    """Where the current value sits in its trailing distribution (0-100)."""
    def _rank(arr):
        if len(arr) < 2:
            return 50.0
        current = arr[-1]
        count_below = (arr[:-1] < current).sum()
        return (count_below / (len(arr) - 1)) * 100

    result = series.rolling(window, min_periods=min(60, window)).apply(_rank, raw=True)
    result.name = f"{series.name}_pctile"
    return result


def rolling_change(series: pd.Series, periods: int = 21) -> pd.Series:
    """Absolute change over N periods."""
    change = series.diff(periods)
    change.name = f"{series.name}_chg{periods}"
    return change


def rate_of_change(series: pd.Series, periods: int = 252) -> pd.Series:
    """Percentage change over N periods."""
    roc = series.pct_change(periods) * 100
    roc.name = f"{series.name}_roc{periods}"
    return roc


def direction_arrow(series: pd.Series, lookback: int = 21, threshold: float = 0.01) -> str:
    """Return an arrow indicating recent direction of a series."""
    if len(series) < lookback + 1:
        return "—"
    recent = series.iloc[-1]
    past = series.iloc[-(lookback + 1)]
    if pd.isna(recent) or pd.isna(past) or past == 0:
        return "—"
    pct_change = (recent - past) / abs(past)
    if pct_change > threshold:
        return "up"
    elif pct_change < -threshold:
        return "down"
    return "flat"

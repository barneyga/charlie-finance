"""Sentiment analysis functions for dashboard consumption."""
import pandas as pd

from charlie.storage.db import Database
from charlie.storage.models import query_series


def _sentiment_label(score: float) -> tuple[str, str]:
    """Return (label, color) for a 0-100 sentiment score."""
    if score >= 70:
        return "Very Bullish", "#22c55e"
    elif score >= 60:
        return "Bullish", "#86efac"
    elif score >= 40:
        return "Neutral", "#fbbf24"
    elif score >= 30:
        return "Bearish", "#f87171"
    else:
        return "Very Bearish", "#ef4444"


def sentiment_summary(db: Database) -> dict:
    """Aggregate sentiment summary across all subreddits.

    Returns dict with: overall_score, label, color, trend,
    subreddit_scores {name: score}, history Series.
    """
    overall = query_series(db, "SENT_REDDIT_ALL")

    if overall.empty:
        return {
            "available": False,
            "overall_score": 50.0,
            "label": "No Data",
            "color": "#888",
            "trend": 0.0,
            "subreddit_scores": {},
            "history": pd.Series(dtype=float),
        }

    current = overall.iloc[-1]
    label, color = _sentiment_label(current)

    # 7-day trend
    trend = 0.0
    if len(overall) >= 7:
        week_ago = overall.iloc[-7]
        trend = current - week_ago

    # Per-subreddit scores (latest value)
    sub_ids = {
        "SENT_REDDIT_WSB": "r/wallstreetbets",
        "SENT_REDDIT_STOCKS": "r/stocks",
        "SENT_REDDIT_INVESTING": "r/investing",
    }
    subreddit_scores = {}
    for sid, name in sub_ids.items():
        s = query_series(db, sid)
        if not s.empty:
            subreddit_scores[name] = s.iloc[-1]

    return {
        "available": True,
        "overall_score": current,
        "label": label,
        "color": color,
        "trend": trend,
        "subreddit_scores": subreddit_scores,
        "history": overall,
    }


def ticker_sentiment_ranking(
    db: Database,
    tickers: list[str],
    prefix: str = "SENT_TICKER_",
) -> pd.DataFrame:
    """Return DataFrame with ticker sentiment scores, sorted by score.

    Columns: Ticker, Score, 7d_Change
    """
    rows = []
    for t in tickers:
        sid = f"{prefix}{t}"
        s = query_series(db, sid)
        if s.empty:
            continue
        current = s.iloc[-1]
        change_7d = 0.0
        if len(s) >= 7:
            change_7d = current - s.iloc[-7]
        label, _ = _sentiment_label(current)
        rows.append({
            "Ticker": t,
            "Score": round(current, 1),
            "Label": label,
            "7d Change": round(change_7d, 1),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    return df.reset_index(drop=True)


def sentiment_vs_price(
    db: Database,
    ticker: str,
    prefix: str = "SENT_TICKER_",
    days: int = 90,
) -> pd.DataFrame:
    """Aligned DataFrame with sentiment + price for a single ticker."""
    sent_sid = f"{prefix}{ticker}"
    sent = query_series(db, sent_sid)
    price = query_series(db, ticker)

    if sent.empty or price.empty:
        return pd.DataFrame()

    df = pd.DataFrame({"Sentiment": sent, "Price": price}).dropna()
    if len(df) > days:
        df = df.iloc[-days:]
    return df

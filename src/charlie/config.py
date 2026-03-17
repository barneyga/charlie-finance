from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import yaml
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class SeriesConfig:
    id: str
    name: str
    frequency: str
    category: str


@dataclass(frozen=True)
class TickerConfig:
    symbol: str
    name: str
    category: str


@dataclass(frozen=True)
class CalendarRelease:
    id: int
    name: str
    full_name: str
    importance: str  # "high", "medium", "low"
    fixed_dates: tuple[str, ...] = ()  # optional hardcoded dates (YYYY-MM-DD)


@dataclass(frozen=True)
class SubredditConfig:
    name: str
    series_id: str
    display_name: str
    fetch_limit: int = 100
    sort: str = "hot"


@dataclass(frozen=True)
class SentimentConfig:
    subreddits: tuple[SubredditConfig, ...]
    tracked_tickers: tuple[str, ...]
    ticker_series_prefix: str
    aggregate_series_id: str


@dataclass(frozen=True)
class Settings:
    fred_api_key: str
    db_path: Path
    series: tuple[SeriesConfig, ...]
    categories: tuple[str, ...]
    tickers: tuple[TickerConfig, ...]
    ticker_categories: tuple[str, ...]
    calendar_releases: tuple[CalendarRelease, ...]
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "charlie-finance/0.1"
    sentiment: SentimentConfig | None = None

    def series_by_category(self, category: str) -> list[SeriesConfig]:
        return [s for s in self.series if s.category == category]

    def series_by_id(self, series_id: str) -> SeriesConfig | None:
        for s in self.series:
            if s.id == series_id:
                return s
        return None

    def tickers_by_category(self, category: str) -> list[TickerConfig]:
        return [t for t in self.tickers if t.category == category]

    def ticker_by_symbol(self, symbol: str) -> TickerConfig | None:
        for t in self.tickers:
            if t.symbol == symbol:
                return t
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    fred_api_key = os.getenv("FRED_API_KEY", "")
    db_path = Path(os.getenv("DB_PATH", "data/charlie.db"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    # Load FRED series
    config_path = PROJECT_ROOT / "config" / "series.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    all_series = []
    categories = []
    for cat_name, cat_data in raw.get("categories", {}).items():
        categories.append(cat_name)
        for s in cat_data.get("series", []):
            all_series.append(SeriesConfig(
                id=s["id"],
                name=s["name"],
                frequency=s["frequency"],
                category=cat_name,
            ))

    # Load Yahoo tickers
    tickers_path = PROJECT_ROOT / "config" / "tickers.yaml"
    all_tickers = []
    ticker_categories = []
    if tickers_path.exists():
        with open(tickers_path) as f:
            ticker_raw = yaml.safe_load(f)
        for cat_name, cat_data in ticker_raw.get("categories", {}).items():
            ticker_categories.append(cat_name)
            for t in cat_data.get("tickers", []):
                all_tickers.append(TickerConfig(
                    symbol=t["symbol"],
                    name=t["name"],
                    category=cat_name,
                ))

    # Load calendar releases
    calendar_path = PROJECT_ROOT / "config" / "calendar.yaml"
    all_releases = []
    if calendar_path.exists():
        with open(calendar_path) as f:
            cal_raw = yaml.safe_load(f)
        for r in cal_raw.get("releases", []):
            fixed = tuple(r.get("fixed_dates", []))
            all_releases.append(CalendarRelease(
                id=r["id"],
                name=r["name"],
                full_name=r["full_name"],
                importance=r["importance"],
                fixed_dates=fixed,
            ))

    # Load sentiment config
    sentiment_path = PROJECT_ROOT / "config" / "sentiment.yaml"
    sentiment_cfg = None
    if sentiment_path.exists():
        with open(sentiment_path) as f:
            sent_raw = yaml.safe_load(f)
        if sent_raw:
            subs = []
            for s in sent_raw.get("subreddits", []):
                subs.append(SubredditConfig(
                    name=s["name"],
                    series_id=s["series_id"],
                    display_name=s["display_name"],
                    fetch_limit=s.get("fetch_limit", 100),
                    sort=s.get("sort", "hot"),
                ))
            tt = sent_raw.get("ticker_tracking", {})
            agg = sent_raw.get("aggregate", {})
            sentiment_cfg = SentimentConfig(
                subreddits=tuple(subs),
                tracked_tickers=tuple(tt.get("symbols", [])),
                ticker_series_prefix=tt.get("series_id_prefix", "SENT_TICKER_"),
                aggregate_series_id=agg.get("series_id", "SENT_REDDIT_ALL"),
            )

    # Reddit credentials from .env
    reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "charlie-finance/0.1")

    return Settings(
        fred_api_key=fred_api_key,
        db_path=db_path,
        series=tuple(all_series),
        categories=tuple(categories),
        tickers=tuple(all_tickers),
        ticker_categories=tuple(ticker_categories),
        calendar_releases=tuple(all_releases),
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        reddit_user_agent=reddit_user_agent,
        sentiment=sentiment_cfg,
    )

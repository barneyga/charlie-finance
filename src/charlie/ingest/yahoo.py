"""Yahoo Finance data ingestion via yfinance."""
import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from charlie.config import Settings, TickerConfig
from charlie.storage.db import Database
from charlie.storage.models import (
    get_latest_date,
    upsert_observations,
    upsert_series_meta,
)

logger = logging.getLogger(__name__)


class YahooIngester:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.errors: list[tuple[str, str]] = []

    def fetch_ticker(self, tc: TickerConfig, force_full: bool = False) -> int:
        """Fetch a single ticker. Returns number of observations upserted."""
        try:
            upsert_series_meta(
                self.db, tc.symbol, tc.name, tc.category,
                frequency="daily", units="USD", source="yahoo",
            )

            # Determine start date for incremental fetch
            start = "2000-01-01"
            if not force_full:
                latest = get_latest_date(self.db, tc.symbol)
                if latest:
                    start = (
                        datetime.strptime(latest, "%Y-%m-%d") - timedelta(days=3)
                    ).strftime("%Y-%m-%d")

            ticker = yf.Ticker(tc.symbol)
            hist = ticker.history(start=start, auto_adjust=True)

            if hist is None or hist.empty:
                logger.warning(f"No data returned for {tc.symbol}")
                return 0

            # Use Close price (already adjusted since auto_adjust=True)
            close = hist["Close"].dropna()
            close.index = close.index.tz_localize(None)  # strip timezone
            upsert_observations(self.db, tc.symbol, close)
            logger.info(f"Fetched {len(close)} observations for {tc.symbol}")
            return len(close)

        except Exception as e:
            logger.error(f"Failed to fetch {tc.symbol}: {e}")
            self.errors.append((tc.symbol, str(e)))
            return 0

    def fetch_category(self, category: str, force_full: bool = False) -> int:
        """Fetch all tickers in a category."""
        tickers = self.settings.tickers_by_category(category)
        total = 0
        for tc in tickers:
            total += self.fetch_ticker(tc, force_full)
        return total

    def fetch_all(self, force_full: bool = False) -> int:
        """Fetch all configured tickers."""
        total = 0
        for tc in self.settings.tickers:
            total += self.fetch_ticker(tc, force_full)
        return total

    def report(self) -> str:
        if not self.errors:
            return "All tickers fetched successfully."
        lines = ["Errors encountered:"]
        for sym, err in self.errors:
            lines.append(f"  {sym}: {err}")
        return "\n".join(lines)

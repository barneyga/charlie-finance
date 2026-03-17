"""ETF flow tracking via dollar volume (yfinance).

Dollar volume = price × volume.  We compute a rolling z-score of dollar
volume to identify unusual activity, and track relative dollar volume
across ETFs as a proxy for fund-flow direction.  No API key required.
"""
import logging
import time

import pandas as pd
import yfinance as yf

from charlie.config import Settings, ETFFlowConfig
from charlie.storage.db import Database
from charlie.storage.models import upsert_series_meta, upsert_observations

logger = logging.getLogger(__name__)


class ETFFlowIngester:
    """Fetch ETF price + volume and compute dollar volume flow metrics."""

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.errors: list[tuple[str, str]] = []

    def fetch_ticker(self, etf: ETFFlowConfig) -> int:
        """Fetch price/volume for one ETF and compute flow metrics.

        Stores three series per ETF:
          FLOW_{symbol}_DVOL   — daily dollar volume ($M)
          FLOW_{symbol}_DAVG   — 20-day moving average of dollar volume ($M)
          FLOW_{symbol}_CUM    — cumulative net dollar volume change vs 20d avg ($M)

        Returns total observation count across all three series.
        """
        symbol = etf.symbol
        total = 0

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start="2020-01-01", auto_adjust=True)

            if hist.empty or "Close" not in hist.columns or "Volume" not in hist.columns:
                logger.warning(f"{symbol}: no price/volume data")
                self.errors.append((symbol, "no price/volume data"))
                return 0

            price = hist["Close"].dropna()
            volume = hist["Volume"].dropna()

            # Strip timezone
            if hasattr(price.index, "tz") and price.index.tz is not None:
                price.index = price.index.tz_localize(None)
            if hasattr(volume.index, "tz") and volume.index.tz is not None:
                volume.index = volume.index.tz_localize(None)

            # Align
            common = price.index.intersection(volume.index)
            price = price.loc[common]
            volume = volume.loc[common]

            # Dollar volume in millions
            dvol = (price * volume) / 1e6
            dvol = dvol.dropna()

            # 20-day moving average
            dvol_avg = dvol.rolling(20, min_periods=10).mean().dropna()

            # Net flow proxy = cumulative deviation from 20d average
            # Positive = above-average activity (demand), negative = below-average
            deviation = (dvol - dvol_avg).dropna()
            cum_flow = deviation.cumsum()

            # Store daily dollar volume
            dvol_sid = f"FLOW_{symbol}_DVOL"
            upsert_series_meta(
                self.db, dvol_sid, f"{etf.name} Dollar Volume ($M)",
                etf.category, "daily", "millions_usd", source="etf_flow",
            )
            upsert_observations(self.db, dvol_sid, dvol)
            total += len(dvol)

            # Store 20d moving average
            davg_sid = f"FLOW_{symbol}_DAVG"
            upsert_series_meta(
                self.db, davg_sid, f"{etf.name} 20d Avg Dollar Volume ($M)",
                etf.category, "daily", "millions_usd", source="etf_flow",
            )
            upsert_observations(self.db, davg_sid, dvol_avg)
            total += len(dvol_avg)

            # Store cumulative net flow proxy
            cum_sid = f"FLOW_{symbol}_CUM"
            upsert_series_meta(
                self.db, cum_sid, f"{etf.name} Cumulative Net Volume Flow ($M)",
                etf.category, "daily", "millions_usd", source="etf_flow",
            )
            upsert_observations(self.db, cum_sid, cum_flow)
            total += len(cum_flow)

            logger.info(
                f"{symbol}: {len(dvol)} days, "
                f"latest dvol={dvol.iloc[-1]:,.0f}M, "
                f"20d avg={dvol_avg.iloc[-1]:,.0f}M"
            )

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            self.errors.append((symbol, str(e)))

        return total

    def fetch_all(self) -> int:
        """Fetch flows for all configured ETFs."""
        total = 0
        for etf in self.settings.etf_flow_tickers:
            total += self.fetch_ticker(etf)
            time.sleep(0.5)  # Rate limiting
        return total

    def report(self) -> str:
        if not self.errors:
            return "All ETF flow data fetched successfully."
        lines = ["Errors encountered:"]
        for sym, err in self.errors:
            lines.append(f"  {sym}: {err}")
        return "\n".join(lines)

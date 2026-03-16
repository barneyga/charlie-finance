"""FRED data ingestion with incremental updates."""
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from fredapi import Fred

from charlie.config import Settings, SeriesConfig
from charlie.storage.db import Database
from charlie.storage.models import (
    get_latest_date,
    upsert_observations,
    upsert_series_meta,
)

logger = logging.getLogger(__name__)


class FredIngester:
    def __init__(self, settings: Settings, db: Database):
        self.fred = Fred(api_key=settings.fred_api_key)
        self.settings = settings
        self.db = db
        self.errors: list[tuple[str, str]] = []

    def fetch_series(self, sc: SeriesConfig, force_full: bool = False) -> int:
        """Fetch a single FRED series. Returns number of observations upserted."""
        try:
            # Ensure metadata row exists
            units = None
            try:
                info = self.fred.get_series_info(sc.id)
                units = info.get("units", None)
            except Exception:
                pass

            upsert_series_meta(
                self.db, sc.id, sc.name, sc.category, sc.frequency, units
            )

            # Determine start date for incremental fetch
            kwargs = {}
            if not force_full:
                latest = get_latest_date(self.db, sc.id)
                if latest:
                    overlap = (
                        datetime.strptime(latest, "%Y-%m-%d") - timedelta(days=7)
                    )
                    kwargs["observation_start"] = overlap.strftime("%Y-%m-%d")

            data: pd.Series = self.fred.get_series(sc.id, **kwargs)

            if data is not None and not data.empty:
                upsert_observations(self.db, sc.id, data)
                logger.info(f"Fetched {len(data)} observations for {sc.id}")
                return len(data)
            else:
                logger.warning(f"No data returned for {sc.id}")
                return 0

        except Exception as e:
            logger.error(f"Failed to fetch {sc.id}: {e}")
            self.errors.append((sc.id, str(e)))
            return 0

    def fetch_category(self, category: str, force_full: bool = False) -> int:
        """Fetch all series in a category. Returns total observations."""
        series_list = self.settings.series_by_category(category)
        total = 0
        for i, sc in enumerate(series_list):
            if i > 0:
                time.sleep(0.6)  # rate limit: 120 req/min
            total += self.fetch_series(sc, force_full)
        return total

    def fetch_all(self, force_full: bool = False) -> int:
        """Fetch all configured series. Returns total observations."""
        total = 0
        first = True
        for sc in self.settings.series:
            if not first:
                time.sleep(0.6)
            first = False
            total += self.fetch_series(sc, force_full)
        return total

    def report(self) -> str:
        if not self.errors:
            return "All series fetched successfully."
        lines = ["Errors encountered:"]
        for sid, err in self.errors:
            lines.append(f"  {sid}: {err}")
        return "\n".join(lines)

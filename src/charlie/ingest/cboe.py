"""CBOE put/call ratio ingestion from free CSV endpoints."""
import logging
from io import StringIO

import pandas as pd
import requests

from charlie.config import Settings
from charlie.storage.db import Database
from charlie.storage.models import upsert_series_meta, upsert_observations

logger = logging.getLogger(__name__)

# CBOE publishes these CSVs daily, no API key needed
_SOURCES = {
    "PCR_TOTAL": {
        "url": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv",
        "name": "CBOE Total Put/Call Ratio",
    },
    "PCR_EQUITY": {
        "url": "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv",
        "name": "CBOE Equity Put/Call Ratio",
    },
}


class CBOEIngester:
    """Fetch put/call ratio data from CBOE's free CSV endpoints."""

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.errors: list[tuple[str, str]] = []

    def fetch_series(self, series_id: str) -> int:
        """Fetch a single put/call CSV. Returns observation count."""
        src = _SOURCES[series_id]
        try:
            resp = requests.get(src["url"], timeout=30)
            resp.raise_for_status()

            # CSV has 2 header lines (disclaimer + product line) before column names
            df = pd.read_csv(StringIO(resp.text), skiprows=2)

            # Columns: DATE, CALLS, PUTS, TOTAL, P/C Ratio
            df["DATE"] = pd.to_datetime(df["DATE"], format="mixed")
            ratio = df.set_index("DATE")["P/C Ratio"].dropna()
            ratio = pd.to_numeric(ratio, errors="coerce").dropna()
            ratio.index = ratio.index.tz_localize(None)

            upsert_series_meta(
                self.db, series_id, src["name"],
                "options", "daily", "ratio", source="cboe",
            )
            upsert_observations(self.db, series_id, ratio)

            logger.info(f"Fetched {len(ratio)} observations for {series_id}")
            return len(ratio)

        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            self.errors.append((series_id, str(e)))
            return 0

    def fetch_all(self) -> int:
        """Fetch all CBOE put/call series."""
        total = 0
        for sid in _SOURCES:
            total += self.fetch_series(sid)
        return total

    def report(self) -> str:
        if not self.errors:
            return "All CBOE series fetched successfully."
        lines = ["Errors encountered:"]
        for name, err in self.errors:
            lines.append(f"  {name}: {err}")
        return "\n".join(lines)

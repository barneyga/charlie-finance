"""CFTC Commitment of Traders (COT) ingestion via cot_reports library."""
import logging

import pandas as pd
import cot_reports as cot

from charlie.config import Settings
from charlie.storage.db import Database
from charlie.storage.models import upsert_series_meta, upsert_observations

logger = logging.getLogger(__name__)

# Contracts to track: (CFTC name, series prefix, display name)
_CONTRACTS = [
    ("E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE", "COT_ES", "S&P 500 E-mini"),
    ("NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE", "COT_NQ", "Nasdaq 100"),
    ("GOLD - COMMODITY EXCHANGE INC.", "COT_GC", "Gold"),
    ("CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE", "COT_CL", "WTI Crude Oil"),
    ("UST 10Y NOTE - CHICAGO BOARD OF TRADE", "COT_ZN", "10Y Treasury"),
    ("EURO FX - CHICAGO MERCANTILE EXCHANGE", "COT_EC", "Euro FX"),
]

# Legacy report columns
_DATE_COL = "As of Date in Form YYYY-MM-DD"
_NAME_COL = "Market and Exchange Names"
_OI_COL = "Open Interest (All)"
_NONCOMM_LONG = "Noncommercial Positions-Long (All)"
_NONCOMM_SHORT = "Noncommercial Positions-Short (All)"


class CFTCIngester:
    """Fetch COT positioning data from CFTC via cot_reports library."""

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.errors: list[tuple[str, str]] = []

    def fetch_all(self) -> int:
        """Fetch legacy futures COT report and extract tracked contracts."""
        try:
            logger.info("Downloading CFTC COT legacy futures report...")
            df = cot.cot_all(cot_report_type="legacy_fut")
            logger.info(f"Downloaded {len(df)} total COT records")
        except Exception as e:
            logger.error(f"Failed to download COT data: {e}")
            self.errors.append(("COT_DOWNLOAD", str(e)))
            return 0

        total = 0
        for cftc_name, prefix, display in _CONTRACTS:
            try:
                total += self._process_contract(df, cftc_name, prefix, display)
            except Exception as e:
                logger.error(f"Failed to process {prefix}: {e}")
                self.errors.append((prefix, str(e)))

        return total

    def _process_contract(
        self, df: pd.DataFrame, cftc_name: str, prefix: str, display: str
    ) -> int:
        """Extract and store positioning data for a single contract."""
        mask = df[_NAME_COL] == cftc_name
        contract_df = df.loc[mask].copy()

        if contract_df.empty:
            logger.warning(f"No data found for {cftc_name}")
            return 0

        contract_df[_DATE_COL] = pd.to_datetime(contract_df[_DATE_COL])
        contract_df = contract_df.sort_values(_DATE_COL).set_index(_DATE_COL)
        contract_df.index = contract_df.index.tz_localize(None)

        # Net non-commercial positioning (Long - Short)
        net = (
            pd.to_numeric(contract_df[_NONCOMM_LONG], errors="coerce")
            - pd.to_numeric(contract_df[_NONCOMM_SHORT], errors="coerce")
        ).dropna()

        # Open interest
        oi = pd.to_numeric(contract_df[_OI_COL], errors="coerce").dropna()

        # Net as % of open interest
        pct = (net / oi * 100).dropna()

        count = 0

        # Store net positioning
        sid_net = f"{prefix}_NET"
        upsert_series_meta(
            self.db, sid_net, f"{display} Net Speculator",
            "cot", "weekly", "contracts", source="cftc",
        )
        upsert_observations(self.db, sid_net, net)
        count += len(net)

        # Store open interest
        sid_oi = f"{prefix}_OI"
        upsert_series_meta(
            self.db, sid_oi, f"{display} Open Interest",
            "cot", "weekly", "contracts", source="cftc",
        )
        upsert_observations(self.db, sid_oi, oi)
        count += len(oi)

        # Store net % of OI (key signal)
        sid_pct = f"{prefix}_PCT"
        upsert_series_meta(
            self.db, sid_pct, f"{display} Net % of OI",
            "cot", "weekly", "% of OI", source="cftc",
        )
        upsert_observations(self.db, sid_pct, pct)
        count += len(pct)

        logger.info(f"Processed {prefix}: {len(net)} weeks, latest net%={pct.iloc[-1]:.1f}%")
        return count

    def report(self) -> str:
        if not self.errors:
            return "All CFTC contracts fetched successfully."
        lines = ["Errors encountered:"]
        for name, err in self.errors:
            lines.append(f"  {name}: {err}")
        return "\n".join(lines)

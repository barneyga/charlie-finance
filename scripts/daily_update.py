"""Daily incremental data update script. Schedule via Task Scheduler or cron."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.config import get_settings
from charlie.storage.db import Database
from charlie.ingest.fred import FredIngester
from charlie.ingest.yahoo import YahooIngester


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    settings = get_settings()

    with Database(settings.db_path) as db:
        db.init_schema()

        # FRED data
        if settings.fred_api_key:
            fred = FredIngester(settings, db)
            fred_count = fred.fetch_all()
            logging.info(f"FRED update: {fred_count} observations")
            if fred.errors:
                logging.warning(fred.report())
        else:
            logging.warning("FRED_API_KEY not set, skipping FRED fetch")

        # Yahoo Finance data
        yahoo = YahooIngester(settings, db)
        yahoo_count = yahoo.fetch_all()
        logging.info(f"Yahoo update: {yahoo_count} observations")
        if yahoo.errors:
            logging.warning(yahoo.report())


if __name__ == "__main__":
    main()

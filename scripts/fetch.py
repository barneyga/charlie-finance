"""CLI script to fetch FRED and/or Yahoo Finance data."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.config import get_settings
from charlie.storage.db import Database
from charlie.ingest.fred import FredIngester
from charlie.ingest.yahoo import YahooIngester


def fetch_fred(settings, db, args):
    if not settings.fred_api_key:
        print("Error: FRED_API_KEY not set. Copy .env.example to .env and add your key.")
        print("Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    ingester = FredIngester(settings, db)

    if args.series:
        sc = settings.series_by_id(args.series)
        if not sc:
            print(f"Unknown series: {args.series}")
            return
        count = ingester.fetch_series(sc, force_full=args.full)
        print(f"[FRED] Fetched {count} observations for {args.series}")
    elif args.category and args.category in settings.categories:
        count = ingester.fetch_category(args.category, force_full=args.full)
        print(f"[FRED] Fetched {count} observations for category '{args.category}'")
    else:
        count = ingester.fetch_all(force_full=args.full)
        print(f"[FRED] Fetched {count} total observations")

    print(ingester.report())


def fetch_yahoo(settings, db, args):
    ingester = YahooIngester(settings, db)

    if args.category and args.category in settings.ticker_categories:
        count = ingester.fetch_category(args.category, force_full=args.full)
        print(f"[Yahoo] Fetched {count} observations for category '{args.category}'")
    else:
        count = ingester.fetch_all(force_full=args.full)
        print(f"[Yahoo] Fetched {count} total observations")

    print(ingester.report())


def main():
    parser = argparse.ArgumentParser(description="Fetch macro and market data")
    parser.add_argument(
        "--source", choices=["fred", "yahoo", "all"], default="fred",
        help="Data source to fetch (default: fred)",
    )
    parser.add_argument("--category", "-c", help="Fetch a specific category only")
    parser.add_argument("--series", "-s", help="Fetch a single FRED series by ID")
    parser.add_argument("--full", action="store_true", help="Force full history fetch")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    settings = get_settings()

    with Database(settings.db_path) as db:
        db.init_schema()

        if args.source in ("fred", "all"):
            fetch_fred(settings, db, args)
        if args.source in ("yahoo", "all"):
            fetch_yahoo(settings, db, args)


if __name__ == "__main__":
    main()

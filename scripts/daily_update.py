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

        # CBOE put/call ratio (no API key needed)
        from charlie.ingest.cboe import CBOEIngester
        cboe = CBOEIngester(settings, db)
        cboe_count = cboe.fetch_all()
        logging.info(f"CBOE update: {cboe_count} observations")
        if cboe.errors:
            logging.warning(cboe.report())

        # CFTC COT positioning (no API key needed, weekly data)
        from charlie.ingest.cftc import CFTCIngester
        cftc = CFTCIngester(settings, db)
        cftc_count = cftc.fetch_all()
        logging.info(f"CFTC update: {cftc_count} observations")
        if cftc.errors:
            logging.warning(cftc.report())

        # ETF flow data (no API key needed)
        if settings.etf_flow_tickers:
            from charlie.ingest.etf_flows import ETFFlowIngester
            flows = ETFFlowIngester(settings, db)
            flow_count = flows.fetch_all()
            logging.info(f"ETF flows update: {flow_count} observations")
            if flows.errors:
                logging.warning(flows.report())

        # StockTwits sentiment (no API key required)
        if settings.stocktwits:
            from charlie.ingest.stocktwits import StockTwitsIngester
            st_ing = StockTwitsIngester(settings, db)
            st_count = st_ing.fetch_all()
            logging.info(f"StockTwits update: {st_count} symbols scored")
            if st_ing.errors:
                logging.warning(st_ing.report())
        else:
            logging.info("No StockTwits config, skipping")

        # Reddit sentiment data
        if settings.reddit_client_id and settings.sentiment:
            from charlie.ingest.sentiment import SentimentIngester
            sent = SentimentIngester(settings, db)
            sent_count = sent.fetch_all()
            logging.info(f"Sentiment update: {sent_count} posts scored")
            if sent.errors:
                logging.warning(sent.report())
        else:
            logging.info("Reddit credentials not set, skipping sentiment fetch")

        # Alert evaluation (state-transition detection)
        from charlie.analysis.alerts import check_alerts
        from charlie.analysis.notify import send_alert_email

        new_alerts = check_alerts(db, settings)
        if new_alerts:
            logging.info(f"Alerts triggered: {len(new_alerts)}")
            for a in new_alerts:
                logging.info(f"  [{a['level'].upper()}] {a['message']}")
            red_alerts = [a for a in new_alerts if a["level"] == "red"]
            if red_alerts and settings.smtp_host:
                send_alert_email(settings, db, red_alerts)
        else:
            logging.info("No new alert transitions")


if __name__ == "__main__":
    main()

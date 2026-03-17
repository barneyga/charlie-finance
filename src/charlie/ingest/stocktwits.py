"""StockTwits sentiment ingestion — free API, uses curl_cffi for Cloudflare bypass."""
import logging
import time
from datetime import datetime, timezone

import pandas as pd
from curl_cffi import requests as cffi_requests

from charlie.config import Settings
from charlie.storage.db import Database
from charlie.storage.models import upsert_series_meta, upsert_observations

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.stocktwits.com/api/2"


class StockTwitsIngester:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.errors: list[tuple[str, str]] = []
        self.token = settings.stocktwits_access_token or ""

    def _fetch_symbol(self, symbol: str, limit: int = 30) -> dict | None:
        """Fetch recent messages for a symbol. Returns parsed JSON or None."""
        url = f"{_BASE_URL}/streams/symbol/{symbol}.json"
        params = {"limit": limit}
        if self.token:
            params["access_token"] = self.token

        try:
            resp = cffi_requests.get(
                url, params=params, impersonate="chrome", timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"StockTwits fetch failed for {symbol}: {e}")
            self.errors.append((symbol, str(e)))
            return None

    def _score_messages(self, data: dict) -> dict:
        """Score messages by counting bullish/bearish tags.

        Returns dict with: score (0-100), bull_count, bear_count, total, bull_pct.
        """
        messages = data.get("messages", [])
        bull = 0
        bear = 0
        total = 0

        for msg in messages:
            sentiment = msg.get("entities", {}).get("sentiment")
            if sentiment:
                basic = sentiment.get("basic")
                if basic == "Bullish":
                    bull += 1
                elif basic == "Bearish":
                    bear += 1
            total += 1

        tagged = bull + bear
        if tagged == 0:
            score = 50.0  # neutral if no tagged messages
        else:
            score = (bull / tagged) * 100

        return {
            "score": score,
            "bull_count": bull,
            "bear_count": bear,
            "total": total,
            "bull_pct": (bull / tagged * 100) if tagged else 50.0,
            "msg_count": total,
        }

    def fetch_symbol(self, symbol: str) -> float | None:
        """Fetch and score a single symbol. Returns score or None on error."""
        cfg = self.settings.stocktwits
        if not cfg:
            return None

        data = self._fetch_symbol(symbol, limit=cfg.fetch_limit)
        if not data:
            return None

        result = self._score_messages(data)
        score = result["score"]

        # Store in DB
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sid = f"{cfg.series_id_prefix}{symbol}"

        upsert_series_meta(
            self.db, sid, f"{symbol} StockTwits Sentiment",
            "sentiment", "daily", "score (0-100)", source="stocktwits",
        )
        obs = pd.Series(
            [score],
            index=pd.DatetimeIndex([today]),
            name=sid,
        )
        upsert_observations(self.db, sid, obs)

        logger.info(
            f"StockTwits {symbol}: score={score:.1f} "
            f"(bull={result['bull_count']}, bear={result['bear_count']}, "
            f"total={result['total']})"
        )
        return score

    def fetch_all(self) -> int:
        """Fetch all configured symbols. Returns count of symbols scored."""
        cfg = self.settings.stocktwits
        if not cfg:
            logger.warning("No StockTwits config loaded")
            return 0

        scores = []
        for symbol in cfg.symbols:
            score = self.fetch_symbol(symbol)
            if score is not None:
                scores.append(score)
            time.sleep(0.5)  # rate limiting: stays under 200 req/hr

        # Compute aggregate
        if scores:
            agg_score = sum(scores) / len(scores)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            agg_sid = cfg.aggregate_series_id

            upsert_series_meta(
                self.db, agg_sid, "StockTwits Overall Sentiment",
                "sentiment", "daily", "score (0-100)", source="stocktwits",
            )
            agg_obs = pd.Series(
                [agg_score],
                index=pd.DatetimeIndex([today]),
                name=agg_sid,
            )
            upsert_observations(self.db, agg_sid, agg_obs)

        return len(scores)

    def report(self) -> str:
        if not self.errors:
            return "All StockTwits symbols fetched successfully."
        lines = ["Errors encountered:"]
        for sym, err in self.errors:
            lines.append(f"  {sym}: {err}")
        return "\n".join(lines)

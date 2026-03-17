"""Reddit sentiment ingestion using PRAW + VADER."""
import logging
import re
from datetime import datetime, timezone

import pandas as pd
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from charlie.config import Settings, SubredditConfig
from charlie.storage.db import Database
from charlie.storage.models import upsert_series_meta, upsert_observations

logger = logging.getLogger(__name__)

# Regex to match $TICKER or standalone uppercase symbols
_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")


class SentimentIngester:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.errors: list[tuple[str, str]] = []
        self.analyzer = SentimentIntensityAnalyzer()

        self.reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )

    def _score_text(self, text: str) -> float:
        """VADER compound score for a text string. Returns [-1, +1]."""
        if not text:
            return 0.0
        return self.analyzer.polarity_scores(text)["compound"]

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract tracked ticker symbols from text."""
        if not text or not self.settings.sentiment:
            return []
        tracked = set(self.settings.sentiment.tracked_tickers)
        # Match $TICKER patterns
        found = set(_TICKER_RE.findall(text.upper()))
        # Also match bare uppercase words that are in our tracked list
        words = set(re.findall(r"\b([A-Z]{2,5})\b", text))
        found.update(words & tracked)
        return sorted(found & tracked)

    def _normalize_score(self, compound: float) -> float:
        """Convert VADER compound [-1, +1] to [0, 100]."""
        return (compound + 1) * 50

    def fetch_subreddit(self, sc: SubredditConfig) -> int:
        """Fetch and score posts from a single subreddit. Returns post count."""
        try:
            subreddit = self.reddit.subreddit(sc.name)

            if sc.sort == "hot":
                posts = subreddit.hot(limit=sc.fetch_limit)
            elif sc.sort == "new":
                posts = subreddit.new(limit=sc.fetch_limit)
            else:
                posts = subreddit.top(limit=sc.fetch_limit, time_filter="day")

            scores = []
            ticker_scores: dict[str, list[float]] = {}
            count = 0

            for post in posts:
                text = f"{post.title} {post.selftext or ''}"
                compound = self._score_text(text)
                scores.append(compound)

                # Track per-ticker sentiment
                tickers = self._extract_tickers(text)
                for t in tickers:
                    ticker_scores.setdefault(t, []).append(compound)

                count += 1

            if not scores:
                return 0

            # Store subreddit-level daily mean
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            mean_score = self._normalize_score(sum(scores) / len(scores))

            upsert_series_meta(
                self.db, sc.series_id, sc.display_name,
                "sentiment", "daily", "score (0-100)", source="sentiment",
            )
            obs = pd.Series(
                [mean_score],
                index=pd.DatetimeIndex([today]),
                name=sc.series_id,
            )
            upsert_observations(self.db, sc.series_id, obs)

            # Store per-ticker sentiment
            if self.settings.sentiment:
                prefix = self.settings.sentiment.ticker_series_prefix
                for ticker, t_scores in ticker_scores.items():
                    t_mean = self._normalize_score(sum(t_scores) / len(t_scores))
                    t_sid = f"{prefix}{ticker}"
                    upsert_series_meta(
                        self.db, t_sid, f"{ticker} Reddit Sentiment",
                        "sentiment", "daily", "score (0-100)", source="sentiment",
                    )
                    t_obs = pd.Series(
                        [t_mean],
                        index=pd.DatetimeIndex([today]),
                        name=t_sid,
                    )
                    upsert_observations(self.db, t_sid, t_obs)

            logger.info(f"Scored {count} posts from r/{sc.name}, mean={mean_score:.1f}")
            return count

        except Exception as e:
            logger.error(f"Failed to fetch r/{sc.name}: {e}")
            self.errors.append((sc.name, str(e)))
            return 0

    def fetch_all(self) -> int:
        """Fetch all configured subreddits and compute aggregate. Returns total posts."""
        if not self.settings.sentiment:
            logger.warning("No sentiment config loaded")
            return 0

        total = 0
        for sc in self.settings.sentiment.subreddits:
            total += self.fetch_subreddit(sc)

        # Compute aggregate score across all subreddits
        agg_sid = self.settings.sentiment.aggregate_series_id
        sub_ids = [sc.series_id for sc in self.settings.sentiment.subreddits]
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        agg_scores = []
        for sid in sub_ids:
            row = self.db.conn.execute(
                "SELECT value FROM observations WHERE series_id = ? AND date = ?",
                (sid, today),
            ).fetchone()
            if row and row["value"] is not None:
                agg_scores.append(row["value"])

        if agg_scores:
            agg_mean = sum(agg_scores) / len(agg_scores)
            upsert_series_meta(
                self.db, agg_sid, "Reddit Overall Sentiment",
                "sentiment", "daily", "score (0-100)", source="sentiment",
            )
            agg_obs = pd.Series(
                [agg_mean],
                index=pd.DatetimeIndex([today]),
                name=agg_sid,
            )
            upsert_observations(self.db, agg_sid, agg_obs)

        return total

    def report(self) -> str:
        if not self.errors:
            return "All subreddits fetched successfully."
        lines = ["Errors encountered:"]
        for name, err in self.errors:
            lines.append(f"  r/{name}: {err}")
        return "\n".join(lines)

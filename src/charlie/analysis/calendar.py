"""Economic calendar via FRED Release Dates API."""
import logging
from datetime import date, timedelta

import pandas as pd
import requests

from charlie.config import CalendarRelease

logger = logging.getLogger(__name__)


def fetch_release_dates(
    api_key: str,
    release_id: int,
    start: str,
    end: str,
) -> list[str]:
    """Fetch scheduled release dates for a single FRED release.

    Returns list of ISO date strings (YYYY-MM-DD).
    """
    url = "https://api.stlouisfed.org/fred/release/dates"
    params = {
        "api_key": api_key,
        "file_type": "json",
        "release_id": release_id,
        "realtime_start": start,
        "realtime_end": end,
        "include_release_dates_with_no_data": "true",
        "sort_order": "asc",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [rd["date"] for rd in data.get("release_dates", [])]
    except Exception as e:
        logger.warning("Failed to fetch release dates for id=%s: %s", release_id, e)
        return []


def get_economic_calendar(
    api_key: str,
    releases: tuple[CalendarRelease, ...] | list[CalendarRelease],
    days_ahead: int = 30,
) -> pd.DataFrame:
    """Build a calendar of upcoming economic releases.

    Returns DataFrame with columns:
        date, name, full_name, importance, days_until
    Sorted by date ascending, filtered to today and future only.
    """
    today = date.today()
    start = today.isoformat()
    end = (today + timedelta(days=days_ahead)).isoformat()

    rows = []
    for rel in releases:
        # Use fixed dates if provided (id=0), otherwise query FRED
        if rel.fixed_dates:
            dates = [d for d in rel.fixed_dates if start <= d <= end]
        elif rel.id > 0:
            dates = fetch_release_dates(api_key, rel.id, start, end)
        else:
            dates = []

        for d in dates:
            dt = date.fromisoformat(d)
            if dt >= today:
                rows.append({
                    "date": d,
                    "name": rel.name,
                    "full_name": rel.full_name,
                    "importance": rel.importance,
                    "days_until": (dt - today).days,
                })

    if not rows:
        return pd.DataFrame(columns=["date", "name", "full_name", "importance", "days_until"])

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_next_release(
    api_key: str,
    releases: tuple[CalendarRelease, ...] | list[CalendarRelease],
    short_name: str,
) -> date | None:
    """Get the next upcoming date for a specific release by short name."""
    today = date.today()
    start = today.isoformat()
    end = (today + timedelta(days=90)).isoformat()

    for rel in releases:
        if rel.name == short_name:
            dates = fetch_release_dates(api_key, rel.id, start, end)
            for d in dates:
                dt = date.fromisoformat(d)
                if dt >= today:
                    return dt
    return None

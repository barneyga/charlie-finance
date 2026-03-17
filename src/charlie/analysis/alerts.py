"""Alert engine — state-transition detection for macro thresholds.

Evaluates all configured AlertThresholds against current data, detects
level transitions (green→yellow→red or vice versa), persists alert
records to SQLite, and returns newly triggered alerts for notification.
"""
import logging
from datetime import datetime, timezone

from charlie.config import Settings, AlertThreshold
from charlie.storage.db import Database
from charlie.storage.models import query_series

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value resolution — compute the current metric value
# ---------------------------------------------------------------------------

def _resolve_value(db: Database, threshold: AlertThreshold) -> float | None:
    """Compute the current value for a given alert metric."""
    mid = threshold.metric_id
    sid = threshold.series_id

    try:
        # Direct series (not derived)
        if not sid.startswith("_"):
            s = query_series(db, sid)
            return float(s.iloc[-1]) if len(s) else None

        # --- Derived metrics ---
        if mid == "spread_10y2y":
            from charlie.analysis.derived import yield_curve_spread
            spread = yield_curve_spread(db)
            return float(spread.iloc[-1]) if len(spread) else None

        if mid == "hy_oas_z":
            from charlie.analysis.stats import rolling_zscore
            hy = query_series(db, "BAMLH0A0HYM2")
            if len(hy) < 52:
                return None
            z = rolling_zscore(hy, 252).dropna()
            return float(z.iloc[-1]) if len(z) else None

        if mid == "cpi_yoy":
            from charlie.analysis.derived import cpi_yoy
            cpi = cpi_yoy(db)
            return float(cpi.iloc[-1]) if len(cpi) else None

        if mid == "gold_silver":
            from charlie.analysis.derived import gold_silver_ratio
            gs = gold_silver_ratio(db)
            return float(gs.iloc[-1]) if len(gs) else None

        if mid == "cot_z":
            from charlie.analysis.derived import cot_summary_table
            tbl = cot_summary_table(db)
            if tbl.empty:
                return None
            return float(tbl["Z-Score"].abs().max())

        if mid in ("fear_greed_fear", "fear_greed_greed"):
            from charlie.analysis.composite import fear_greed_score
            result = fear_greed_score(db)
            return float(result["score"]) if result else None

    except Exception as exc:
        logger.warning("Could not resolve value for %s: %s", mid, exc)
        return None

    return None


# ---------------------------------------------------------------------------
# Level classification
# ---------------------------------------------------------------------------

_LEVEL_ORDER = {"green": 0, "yellow": 1, "red": 2}


def _classify_level(value: float, threshold: AlertThreshold) -> str:
    """Return 'green', 'yellow', or 'red' for a given value."""
    for level_name in ("red", "yellow", "green"):
        bounds = getattr(threshold, level_name)
        lo = bounds[0] if bounds[0] is not None else float("-inf")
        hi = bounds[1] if bounds[1] is not None else float("inf")
        if lo <= value < hi:
            return level_name
    return "green"


def _is_worse(new_level: str, old_level: str) -> bool:
    """Return True if new_level is worse (higher severity) than old_level."""
    return _LEVEL_ORDER.get(new_level, 0) > _LEVEL_ORDER.get(old_level, 0)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_current_state(db: Database, metric_id: str) -> dict | None:
    row = db.conn.execute(
        "SELECT current_level, value, updated_at FROM alert_state WHERE metric_id = ?",
        (metric_id,),
    ).fetchone()
    if row is None:
        return None
    return {"level": row["current_level"], "value": row["value"], "updated_at": row["updated_at"]}


def _update_state(db: Database, metric_id: str, level: str, value: float):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    db.conn.execute(
        "INSERT OR REPLACE INTO alert_state (metric_id, current_level, value, updated_at) "
        "VALUES (?, ?, ?, ?)",
        (metric_id, level, value, now),
    )
    db.conn.commit()


def _create_alert(db: Database, metric_id: str, level: str, message: str, value: float) -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cur = db.conn.execute(
        "INSERT INTO alerts (metric_id, level, message, value, triggered_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (metric_id, level, message, value, now),
    )
    db.conn.commit()
    return cur.lastrowid


def _resolve_alerts(db: Database, metric_id: str):
    """Mark all unresolved alerts for a metric as resolved."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    db.conn.execute(
        "UPDATE alerts SET resolved_at = ? WHERE metric_id = ? AND resolved_at IS NULL",
        (now, metric_id),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_alerts(db: Database, settings: Settings) -> list[dict]:
    """Evaluate all thresholds, detect state transitions, return new alerts.

    On first run for a metric, the initial state is recorded but no alert
    is fired (avoids a flood of alerts on first deployment).
    """
    new_alerts: list[dict] = []

    for threshold in settings.alert_thresholds:
        value = _resolve_value(db, threshold)
        if value is None:
            continue

        level = _classify_level(value, threshold)
        prev = _get_current_state(db, threshold.metric_id)

        if prev is None:
            # First run — record state, don't alert
            _update_state(db, threshold.metric_id, level, value)
            logger.info("Alert state initialized: %s = %s (%.2f)", threshold.metric_id, level, value)
            continue

        old_level = prev["level"]
        if level == old_level:
            # No transition — just update the value
            _update_state(db, threshold.metric_id, level, value)
            continue

        # State transition detected
        _update_state(db, threshold.metric_id, level, value)

        if _is_worse(level, old_level):
            # Escalation — create alert
            msg = f"{threshold.name}: {threshold.description} (value={value:.2f}, {old_level}→{level})"
            alert_id = _create_alert(db, threshold.metric_id, level, msg, value)
            alert = {
                "id": alert_id,
                "metric_id": threshold.metric_id,
                "name": threshold.name,
                "level": level,
                "message": msg,
                "value": value,
                "old_level": old_level,
            }
            new_alerts.append(alert)
            logger.info("ALERT: %s", msg)
        else:
            # De-escalation — resolve open alerts
            _resolve_alerts(db, threshold.metric_id)
            logger.info("RESOLVED: %s returned to %s (%.2f)", threshold.name, level, value)

    return new_alerts


# ---------------------------------------------------------------------------
# Query helpers for dashboard
# ---------------------------------------------------------------------------

def get_active_alerts(db: Database) -> list[dict]:
    """Return all unresolved alerts, newest first."""
    rows = db.conn.execute(
        "SELECT id, metric_id, level, message, value, triggered_at "
        "FROM alerts WHERE resolved_at IS NULL ORDER BY triggered_at DESC",
    ).fetchall()
    return [dict(r) for r in rows]


def get_alert_history(db: Database, limit: int = 50) -> list[dict]:
    """Return recent alerts (both active and resolved)."""
    rows = db.conn.execute(
        "SELECT id, metric_id, level, message, value, triggered_at, resolved_at, notified "
        "FROM alerts ORDER BY triggered_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def build_threshold_dict(settings: Settings) -> dict:
    """Build the dashboard _THRESHOLDS dict from unified config.

    Returns a dict like: {"vix": {"green": (0, 20), "yellow": (20, 25), "red": (25, inf)}}
    """
    thresholds = {}
    for t in settings.alert_thresholds:
        thresholds[t.metric_id] = {
            "green": (
                t.green[0] if t.green[0] is not None else float("-inf"),
                t.green[1] if t.green[1] is not None else float("inf"),
            ),
            "yellow": (
                t.yellow[0] if t.yellow[0] is not None else float("-inf"),
                t.yellow[1] if t.yellow[1] is not None else float("inf"),
            ),
            "red": (
                t.red[0] if t.red[0] is not None else float("-inf"),
                t.red[1] if t.red[1] is not None else float("inf"),
            ),
        }
    return thresholds

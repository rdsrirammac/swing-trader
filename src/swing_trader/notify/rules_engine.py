"""Alert suppression rules — rating threshold, dedup, quiet hours (SRS 3.10, AL-006).

Centralizes the "should this alert actually be sent" decision so
`notify.engine` (and any future caller) share one policy.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import Alert
from swing_trader.logging_setup import get_logger

logger = get_logger("notify.rules_engine")

# Ordinal urgency ordering for SignalRating.rating comparisons, least to
# most actionable/urgent. The SRS does not define an explicit ordering for
# `alerts.min_rating_for_signal_alert`; this is a reasonable choice so that
# the default threshold ("Buy") reads naturally as "Buy or Strong Buy
# triggers a signal alert" while Watch/Trim/Hold/Sell do not.
RATING_ORDER = ["Sell", "Hold", "Watch", "Trim", "Buy", "Strong Buy"]


def _rating_rank(rating: str) -> int:
    try:
        return RATING_ORDER.index(rating)
    except ValueError:
        logger.warning("Unknown rating %r not in RATING_ORDER; treating as lowest rank", rating)
        return -1


def should_alert(
    session: Session,
    category: str,
    dedup_key: str,
    min_rating: str | None = None,
    actual_rating: str | None = None,
    severity: str = "info",
) -> bool:
    """AL-006: decide whether an alert should actually be dispatched.

    Checks, in order:
      1. Rating threshold — if both `min_rating` and `actual_rating` are
         given, `actual_rating` must rank >= `min_rating` on `RATING_ORDER`.
      2. Dedup — suppress if an `Alert` row with the same `dedup_key` was
         created within the last `alerts.dedup_window_minutes` minutes.
      3. Quiet hours — if `alerts.quiet_hours.enabled` and the current
         local time (per `app.timezone`, resolved via `zoneinfo`) falls
         within `[start, end)`, suppress — UNLESS `category == "risk"` and
         `severity == "critical"`, which always gets through.

    Returns True iff the alert should be sent through the enabled channels.
    """
    settings = get_settings()

    if min_rating is not None and actual_rating is not None:
        if _rating_rank(actual_rating) < _rating_rank(min_rating):
            logger.info("Suppressing alert: rating %r below threshold %r", actual_rating, min_rating)
            return False

    dedup_window_minutes = settings.get("alerts.dedup_window_minutes", 240)
    cutoff = dt.datetime.utcnow() - dt.timedelta(minutes=dedup_window_minutes)
    stmt = (
        select(Alert)
        .where(Alert.dedup_key == dedup_key, Alert.created_at >= cutoff)
        .limit(1)
    )
    recent = session.execute(stmt).scalars().first()
    if recent is not None:
        logger.info(
            "Suppressing alert: dedup_key %r fired within %s-minute window", dedup_key, dedup_window_minutes
        )
        return False

    quiet_cfg = settings.get("alerts.quiet_hours", {}) or {}
    if quiet_cfg.get("enabled", False):
        if category == "risk" and severity == "critical":
            return True

        tz_name = settings.get("app.timezone", "America/New_York")
        try:
            now_local = dt.datetime.now(ZoneInfo(tz_name)).time()
        except Exception as e:
            logger.warning("Failed to resolve timezone %r: %s; skipping quiet-hours check", tz_name, e)
            return True

        start = _parse_hhmm(quiet_cfg.get("start", "22:00"))
        end = _parse_hhmm(quiet_cfg.get("end", "07:00"))
        if _in_quiet_window(now_local, start, end):
            logger.info("Suppressing alert: within quiet hours [%s, %s)", start, end)
            return False

    return True


def _parse_hhmm(value: str) -> dt.time:
    hour, minute = value.split(":")
    return dt.time(int(hour), int(minute))


def _in_quiet_window(now: dt.time, start: dt.time, end: dt.time) -> bool:
    """Handle quiet windows that wrap midnight (e.g. 22:00 -> 07:00)."""
    if start <= end:
        return start <= now < end
    return now >= start or now < end

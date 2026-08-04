"""Calendar-side earnings-avoidance rules (SRS 3.14, EC-004).

Mirrors the semantics of `signals/earnings_blackout.py` (SR-005), owned by
a different, concurrently-built package, but framed for calendar-triggered
automation — e.g. a scheduled morning job that walks every ticker in the
universe and needs blackout / close-suggestion flags without depending on
that day's signal-generation run having happened first.

Deliberately self-contained: does NOT import from `swing_trader.signals`
(that package may not exist yet in a concurrent build, and importing it
here would create a fragile cross-package coupling). Both modules are
expected to independently compute similar blackout logic from the same
`earnings.*` config block — that duplication is intentional and
documented here, not a bug.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import EarningsEvent
from swing_trader.logging_setup import get_logger

logger = get_logger("calendar_data.rules")


def _nearest_future_earnings(session: Session, ticker: str, as_of: dt.date) -> EarningsEvent | None:
    stmt = (
        select(EarningsEvent)
        .where(EarningsEvent.ticker == ticker.upper(), EarningsEvent.earnings_date >= as_of)
        .order_by(EarningsEvent.earnings_date.asc())
        .limit(1)
    )
    return session.execute(stmt).scalars().first()


def earnings_avoidance_check(session: Session, ticker: str, as_of: dt.date) -> dict:
    """EC-004: compute earnings-driven position-management flags for
    `ticker` as of `as_of`.

    Reads `earnings.blackout_days_before`, `earnings.suggest_close_days_before`,
    and `earnings.auto_close_on_earnings_day` from settings, plus the
    nearest upcoming `EarningsEvent` for the ticker.

    Returns:
        {
          "blackout_new_positions": bool,  # don't open new positions
          "suggest_close_existing": bool,  # flag existing positions for review
          "auto_close_today": bool,        # earnings.auto_close_on_earnings_day AND today IS the earnings date
        }

    If no upcoming earnings event is on record, all three flags are False
    (nothing to avoid). Callers relying on this for safety should ensure
    `calendar_data.earnings.sync_earnings_calendar` runs regularly so
    events don't go stale/missing for actively-traded tickers.
    """
    if isinstance(as_of, dt.datetime):
        as_of = as_of.date()

    settings = get_settings()
    blackout_days_before = settings.get("earnings.blackout_days_before", 5)
    suggest_close_days_before = settings.get("earnings.suggest_close_days_before", 2)
    auto_close_on_earnings_day = settings.get("earnings.auto_close_on_earnings_day", False)

    event = _nearest_future_earnings(session, ticker, as_of)
    if event is None:
        return {
            "blackout_new_positions": False,
            "suggest_close_existing": False,
            "auto_close_today": False,
        }

    days_until = (event.earnings_date - as_of).days

    blackout_new_positions = 0 <= days_until <= blackout_days_before
    suggest_close_existing = 0 <= days_until <= suggest_close_days_before
    auto_close_today = bool(auto_close_on_earnings_day) and days_until == 0

    return {
        "blackout_new_positions": blackout_new_positions,
        "suggest_close_existing": suggest_close_existing,
        "auto_close_today": auto_close_today,
    }

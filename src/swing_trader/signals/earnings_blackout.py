"""SR-005 / EC-004: earnings-blackout gating for new signals and positions.

Blocks new Buy/Strong Buy signals within `earnings.blackout_days_before`
calendar days of a ticker's next known `EarningsEvent`, and separately flags
when it's time to suggest closing/trimming an existing position ahead of the
print (`earnings.suggest_close_days_before`).
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import EarningsEvent


def _nearest_upcoming_earnings(session: Session, ticker: str, as_of: dt.date) -> EarningsEvent | None:
    stmt = (
        select(EarningsEvent)
        .where(EarningsEvent.ticker == ticker, EarningsEvent.earnings_date >= as_of)
        .order_by(EarningsEvent.earnings_date.asc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def days_until_earnings(session: Session, ticker: str, as_of: dt.date) -> int | None:
    """EC-004: calendar days from `as_of` to the next known earnings date for
    `ticker`, or None if no upcoming `EarningsEvent` row exists."""
    event = _nearest_upcoming_earnings(session, ticker, as_of)
    if event is None:
        return None
    return (event.earnings_date - as_of).days


def is_earnings_blackout(session: Session, ticker: str, as_of: dt.date) -> bool:
    """SR-005: True if `as_of` falls within `earnings.blackout_days_before`
    days of the next earnings date, in which case new Buy/Strong Buy signals
    should be suppressed/blocked by the caller."""
    days = days_until_earnings(session, ticker, as_of)
    if days is None:
        return False
    blackout_days = get_settings().get("earnings.blackout_days_before", 5)
    return 0 <= days <= blackout_days


def should_suggest_close(session: Session, ticker: str, as_of: dt.date) -> bool:
    """EC-004: True if `as_of` is within `earnings.suggest_close_days_before`
    days of the next earnings date -- the dashboard/CLI/alerts layer should
    surface a suggestion to close or trim ahead of the print."""
    days = days_until_earnings(session, ticker, as_of)
    if days is None:
        return False
    suggest_days = get_settings().get("earnings.suggest_close_days_before", 2)
    return 0 <= days <= suggest_days

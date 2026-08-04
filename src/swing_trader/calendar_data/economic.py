"""Economic calendar sync & high-impact-event lookahead (SRS 3.14, EC-003).

Thin orchestration over `swing_trader.data.economic_calendar.EconomicCalendarClient`
— this module owns persistence (upsert into `EconomicEvent`) and the
"what's coming up soon" query; the client owns feed parsing.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.data.economic_calendar import EconomicCalendarClient
from swing_trader.db.models import EconomicEvent
from swing_trader.logging_setup import get_logger

logger = get_logger("calendar_data.economic")


def sync_economic_calendar(session: Session) -> list[EconomicEvent]:
    """EC-003: fetch upcoming high-impact economic events and upsert `EconomicEvent` rows.

    Dedupes on (event_name, same calendar day as event_date) since the
    RSS-derived timestamps aren't guaranteed to line up exactly across
    fetches. Never raises — a feed-fetch failure logs a warning and
    returns an empty list.
    """
    client = EconomicCalendarClient()
    try:
        raw_events = client.fetch_upcoming_events(lookback_days=7)
    except Exception as e:
        logger.warning("Failed to fetch economic calendar: %s", e)
        return []

    result: list[EconomicEvent] = []
    for raw in raw_events:
        event_name = raw.get("event_name")
        event_date = raw.get("event_date")
        if event_name is None or event_date is None:
            continue

        day_start = dt.datetime.combine(event_date.date(), dt.time.min)
        day_end = dt.datetime.combine(event_date.date(), dt.time.max)

        existing = (
            session.execute(
                select(EconomicEvent).where(
                    EconomicEvent.event_name == event_name,
                    EconomicEvent.event_date >= day_start,
                    EconomicEvent.event_date <= day_end,
                )
            )
            .scalars()
            .first()
        )

        if existing is not None:
            existing.timing = raw.get("timing") or existing.timing
            existing.historical_reaction_notes = (
                raw.get("historical_reaction_notes") or existing.historical_reaction_notes
            )
            result.append(existing)
            continue

        event = EconomicEvent(
            event_name=event_name,
            event_date=event_date,
            timing=raw.get("timing"),
            historical_reaction_notes=raw.get("historical_reaction_notes"),
            source=raw.get("source", "rss"),
        )
        session.add(event)
        result.append(event)

    session.flush()
    logger.info("Synced %d economic calendar events", len(result))
    return result


def get_upcoming_high_impact_events(session: Session, within_hours: int = 24) -> list[EconomicEvent]:
    """EC-003: return economic events falling within the next `within_hours`
    hours (relative to now, UTC) — drives the "alert 24 hours before major
    events" requirement.
    """
    now = dt.datetime.utcnow()
    horizon = now + dt.timedelta(hours=within_hours)
    stmt = (
        select(EconomicEvent)
        .where(EconomicEvent.event_date >= now, EconomicEvent.event_date <= horizon)
        .order_by(EconomicEvent.event_date.asc())
    )
    return list(session.execute(stmt).scalars().all())

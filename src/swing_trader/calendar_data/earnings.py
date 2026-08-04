"""Earnings calendar sync & post-earnings move prediction (SRS 3.14, EC-001, EC-002)."""
from __future__ import annotations

import datetime as dt
import statistics
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.data.yf_client import YFinanceClient
from swing_trader.db.models import EarningsEvent
from swing_trader.logging_setup import get_logger

logger = get_logger("calendar_data.earnings")


def _coerce_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def _extract_earnings_date(calendar: Any) -> dt.date | None:
    """Best-effort extraction of the next earnings date from yfinance's
    `.calendar`, whose return shape has changed across yfinance versions
    (dict with an 'Earnings Date' key -> list of dates, OR a DataFrame with
    an 'Earnings Date' row/column). Returns the earliest parseable date, or
    None if nothing usable is present.
    """
    try:
        if isinstance(calendar, dict):
            raw = calendar.get("Earnings Date")
            if raw is None:
                return None
            candidates = list(raw) if isinstance(raw, (list, tuple)) else [raw]
            dates = [d for d in (_coerce_date(c) for c in candidates) if d is not None]
            return min(dates) if dates else None

        if isinstance(calendar, pd.DataFrame):
            if "Earnings Date" in calendar.index:
                row = calendar.loc["Earnings Date"]
                values = row.tolist() if hasattr(row, "tolist") else [row]
            elif "Earnings Date" in calendar.columns:
                values = calendar["Earnings Date"].tolist()
            else:
                return None
            dates = [d for d in (_coerce_date(v) for v in values) if d is not None]
            return min(dates) if dates else None
    except Exception as e:
        logger.warning("Failed to parse calendar shape (%s): %s", type(calendar), e)
    return None


def _extract_field(calendar: Any, keys: tuple[str, ...]) -> float | None:
    """Generic best-effort scalar extractor across the dict/DataFrame calendar shapes."""
    try:
        if isinstance(calendar, dict):
            for key in keys:
                val = calendar.get(key)
                if val is not None:
                    if isinstance(val, (list, tuple)):
                        val = val[0] if val else None
                    if val is not None:
                        return float(val)
            return None
        if isinstance(calendar, pd.DataFrame):
            for key in keys:
                if key in calendar.index:
                    row = calendar.loc[key]
                    v = row.iloc[0] if hasattr(row, "iloc") else row
                    if v is not None and not (hasattr(pd, "isna") and pd.isna(v)):
                        return float(v)
                if key in calendar.columns:
                    col = calendar[key]
                    v = col.iloc[0] if hasattr(col, "iloc") else col
                    if v is not None and not (hasattr(pd, "isna") and pd.isna(v)):
                        return float(v)
    except Exception as e:
        logger.warning("Failed to parse calendar field %s: %s", keys, e)
    return None


def sync_earnings_calendar(session: Session, ticker: str) -> EarningsEvent | None:
    """EC-001: fetch the next confirmed/estimated earnings date for `ticker`
    from yfinance and upsert an `EarningsEvent` row (matched on
    ticker+earnings_date).

    Returns None (and logs a warning) if yfinance has no calendar data for
    the ticker or its shape can't be parsed — never raises.
    """
    ticker = ticker.upper()
    try:
        calendar = YFinanceClient.instance().get_calendar(ticker)
    except Exception as e:
        logger.warning("Failed to fetch calendar for %s: %s", ticker, e)
        return None

    earnings_date = _extract_earnings_date(calendar)
    if earnings_date is None:
        logger.warning("No parseable earnings date in calendar for %s", ticker)
        return None

    eps_estimate = _extract_field(calendar, ("EPS Estimate", "Earnings Average"))
    revenue_estimate = _extract_field(calendar, ("Revenue Estimate", "Revenue Average"))

    existing = (
        session.execute(
            select(EarningsEvent).where(
                EarningsEvent.ticker == ticker, EarningsEvent.earnings_date == earnings_date
            )
        )
        .scalars()
        .first()
    )

    if existing is not None:
        if eps_estimate is not None:
            existing.eps_estimate = eps_estimate
        if revenue_estimate is not None:
            existing.revenue_estimate = revenue_estimate
        session.flush()
        return existing

    event = EarningsEvent(
        ticker=ticker,
        earnings_date=earnings_date,
        confirmed=False,
        eps_estimate=eps_estimate,
        revenue_estimate=revenue_estimate,
    )
    session.add(event)
    session.flush()
    logger.info("Synced earnings event for %s on %s", ticker, earnings_date)
    return event


def record_earnings_result(
    session: Session, ticker: str, eps_actual: float, revenue_actual: float | None = None
) -> EarningsEvent:
    """EC-001: record the actual reported EPS/revenue against a ticker's
    most recent (nearest not-in-the-future) `EarningsEvent` row, and
    compute `surprise_pct = (eps_actual - eps_estimate) / abs(eps_estimate)`.

    If no prior `EarningsEvent` row exists on or before today for this
    ticker, a new one is created dated today (confirmed=True) so the
    result isn't lost.
    """
    ticker = ticker.upper()
    today = dt.date.today()
    event = (
        session.execute(
            select(EarningsEvent)
            .where(EarningsEvent.ticker == ticker, EarningsEvent.earnings_date <= today)
            .order_by(EarningsEvent.earnings_date.desc())
        )
        .scalars()
        .first()
    )

    if event is None:
        event = EarningsEvent(ticker=ticker, earnings_date=today, confirmed=True)
        session.add(event)

    event.eps_actual = eps_actual
    if revenue_actual is not None:
        event.revenue_actual = revenue_actual

    if event.eps_estimate:
        event.surprise_pct = (eps_actual - event.eps_estimate) / abs(event.eps_estimate)
    else:
        event.surprise_pct = None

    session.flush()
    logger.info("Recorded earnings result for %s: eps_actual=%s surprise_pct=%s", ticker, eps_actual, event.surprise_pct)
    return event


def straddle_implied_move(call_price: float, put_price: float, stock_price: float) -> float:
    """EC-002: ATM straddle-implied move as a fraction of stock price.

    implied_move_pct = (call_price + put_price) / stock_price
    """
    if stock_price <= 0:
        raise ValueError("stock_price must be positive")
    return (call_price + put_price) / stock_price


def predict_post_earnings_move(
    session: Session,
    ticker: str,
    call_price: float | None = None,
    put_price: float | None = None,
    stock_price: float | None = None,
) -> dict:
    """EC-002: best-effort post-earnings move prediction.

    Combines (a) historical `post_earnings_drift_5d` values across past
    `EarningsEvent` rows for this ticker (mean + stdev), and (b), if an
    ATM straddle is supplied by the caller (`call_price`/`put_price`/
    `stock_price` — typically sourced from
    `YFinanceClient.instance().get_options_chain(...)` by the caller),
    the options-implied move via `straddle_implied_move`.

    Returns:
        {
          "historical_avg_move_pct": float | None,
          "historical_stdev_move_pct": float | None,
          "implied_move_pct": float | None,
          "confidence": "low" | "medium" | "high",
          "sample_size": int,
        }

    `confidence` reflects how many historical events are available:
    <2 -> "low", 2-4 -> "medium", 5+ -> "high".
    """
    ticker = ticker.upper()
    events = (
        session.execute(
            select(EarningsEvent).where(
                EarningsEvent.ticker == ticker, EarningsEvent.post_earnings_drift_5d.is_not(None)
            )
        )
        .scalars()
        .all()
    )

    drifts = [e.post_earnings_drift_5d for e in events if e.post_earnings_drift_5d is not None]
    n = len(drifts)

    historical_avg_move_pct = statistics.mean(drifts) if n >= 1 else None
    historical_stdev_move_pct = statistics.stdev(drifts) if n >= 2 else None

    implied_move_pct = None
    if call_price is not None and put_price is not None and stock_price is not None:
        try:
            implied_move_pct = straddle_implied_move(call_price, put_price, stock_price)
        except Exception as e:
            logger.warning("Failed to compute straddle implied move for %s: %s", ticker, e)

    if n < 2:
        confidence = "low"
    elif n <= 4:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "historical_avg_move_pct": historical_avg_move_pct,
        "historical_stdev_move_pct": historical_stdev_move_pct,
        "implied_move_pct": implied_move_pct,
        "confidence": confidence,
        "sample_size": n,
    }

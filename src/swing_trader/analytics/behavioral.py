"""Behavioral trading-discipline diagnostics (SRS 3.12, PA-004).

Flags patterns associated with poor trading discipline: exiting winners too
early, letting stops drift, revenge trading after a loss, and overtrading
relative to a planned cadence. Several of these are necessarily
*approximations* given the current schema -- each such gap is documented
inline with a suggested schema change as a backlog/ROADMAP item.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import Holding, Trade
from swing_trader.logging_setup import get_logger

logger = get_logger("analytics.behavioral")

DEFAULT_PLANNED_TRADES_PER_WEEK = 3


def _closed_trades(session: Session, portfolio_id: int) -> list[Trade]:
    return (
        session.execute(
            select(Trade)
            .where(Trade.portfolio_id == portfolio_id, Trade.exit_date.is_not(None))
            .order_by(Trade.entry_date)
        )
        .scalars()
        .all()
    )


def _early_exit_rate(trades: list[Trade]) -> float:
    """Fraction of manual exits that sold before reaching the original target.

    Approximation: for trades with `exit_reason == MANUAL` and a recorded
    `target`, we treat it as an "early exit" if the realized move
    (exit_price - entry_price) was positive but smaller than the
    target-implied move (target - entry_price). This only catches early
    exits on trades that were, on net, winners at the time of exit.
    """
    eligible = [t for t in trades if t.exit_reason is not None and t.exit_reason.value == "manual" and t.target is not None]
    if not eligible:
        return 0.0
    early = 0
    for t in eligible:
        if t.exit_price is None or t.entry_price is None:
            continue
        realized_move = t.exit_price - t.entry_price
        target_move = t.target - t.entry_price
        if target_move > 0 and 0 < realized_move < target_move:
            early += 1
    return early / len(eligible)


def _stop_violation_rate(session: Session, trades: list[Trade]) -> float:
    """Fraction of trades whose stop appears to have been moved *further*
    from entry (i.e. loosened, adding risk) between open and close.

    LIMITATION: the schema does not retain a history of stop-loss edits --
    only the `Trade.stop_loss` recorded at trade open and the *current*
    `Holding.stop_loss` (which may have been updated any number of times,
    including tightened via a trailing stop, which is healthy risk
    management and should NOT be flagged as a violation). As a proxy, we
    compare `Trade.stop_loss` at open against the linked `Holding.stop_loss`
    at the time of this query, and only count it as a violation if the
    stop moved AWAY from entry_price (i.e. more risk was taken on, not
    less). This will under/over-count relative to true intra-trade
    behavior.

    ROADMAP / backlog: add a `stop_loss_history` table (holding_id,
    changed_at, old_stop, new_stop) to make this exact.
    """
    violations = 0
    eligible = 0
    for t in trades:
        if t.holding_id is None or t.stop_loss is None or t.entry_price is None:
            continue
        holding = session.get(Holding, t.holding_id)
        if holding is None or holding.stop_loss is None:
            continue
        eligible += 1
        original_risk = abs(t.entry_price - t.stop_loss)
        current_risk = abs(t.entry_price - holding.stop_loss)
        if current_risk > original_risk:
            violations += 1
    return (violations / eligible) if eligible else 0.0


def _revenge_trading_score(trades: list[Trade]) -> float:
    """Fraction of trades that look like "revenge trades": entered within
    24h of the prior trade's exit, where the prior trade was a loss, AND
    position size (shares) increased vs. the prior trade.

    Trades are compared in `entry_date` order (consecutive pairs).
    """
    ordered = sorted(trades, key=lambda t: t.entry_date)
    if len(ordered) < 2:
        return 0.0

    flagged = 0
    comparisons = 0
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.exit_date is None or prev.realized_pnl is None:
            continue
        comparisons += 1
        gap = cur.entry_date - prev.exit_date
        was_loss = prev.realized_pnl < 0
        size_increased = (cur.shares or 0) > (prev.shares or 0)
        if was_loss and gap <= dt.timedelta(hours=24) and size_increased:
            flagged += 1
    return (flagged / comparisons) if comparisons else 0.0


def _overtrading_ratio(trades: list[Trade]) -> float:
    """Trades per week vs. a configured planned cadence.

    Reads `analytics.planned_trades_per_week` from settings; if that key is
    absent (it is not currently defined in config/settings.yaml), defaults
    to 3 trades/week.

    ROADMAP / backlog: add `analytics.planned_trades_per_week` to
    config/settings.yaml so this is operator-configurable without a code
    change.
    """
    settings = get_settings()
    planned_per_week = settings.get("analytics.planned_trades_per_week", DEFAULT_PLANNED_TRADES_PER_WEEK)

    if not trades:
        return 0.0
    entry_dates = sorted(t.entry_date for t in trades)
    span_days = max((entry_dates[-1] - entry_dates[0]).days, 1)
    weeks = max(span_days / 7.0, 1 / 7.0)
    trades_per_week = len(trades) / weeks
    return trades_per_week / planned_per_week if planned_per_week else 0.0


def behavioral_report(session: Session, portfolio_id: int) -> dict:
    """PA-004: behavioral trading-discipline diagnostics for a portfolio.

    Returns:
        {
          "early_exit_rate": float,
          "stop_violation_rate": float,
          "revenge_trading_score": float,
          "overtrading_ratio": float,
        }
    """
    trades = _closed_trades(session, portfolio_id)
    return {
        "early_exit_rate": _early_exit_rate(trades),
        "stop_violation_rate": _stop_violation_rate(session, trades),
        "revenge_trading_score": _revenge_trading_score(trades),
        "overtrading_ratio": _overtrading_ratio(trades),
    }

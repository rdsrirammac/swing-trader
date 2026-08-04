"""Trade execution journal — open/close trade lifecycle (SRS 3.9, TE-001, TE-002).

TE-001: record every trade entry (thesis, stop, target, expected R-multiple,
chart screenshot reference) at the moment of execution.
TE-002: record trade exits and compute realized P&L / R-multiple / holding
period / slippage against the expected fill price.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.db.models import ExitReason, Trade
from swing_trader.logging_setup import get_logger

logger = get_logger("execution.journal")


def open_trade(
    session: Session,
    portfolio_id: int,
    ticker: str,
    entry_price: float,
    shares: float,
    stop_loss: float | None,
    target: float | None,
    thesis: str | None,
    expected_r_multiple: float | None = None,
    holding_id: int | None = None,
    is_paper: bool = True,
    chart_screenshot_path: str | None = None,
) -> Trade:
    """Open (journal) a new trade. TE-001.

    Creates and persists a `Trade` row with `entry_date=utcnow()`. Does not
    touch `Holding`/`Portfolio` cash balances or create a `Holding` row —
    that is the portfolio module's responsibility. `holding_id` is an
    optional FK linking this trade to a live `Holding` row if one already
    exists.

    Caller is expected to be inside a `session_scope()` context; this
    function flushes (not commits) so the returned `Trade.id` is populated.
    """
    trade = Trade(
        portfolio_id=portfolio_id,
        holding_id=holding_id,
        ticker=ticker.upper(),
        entry_date=dt.datetime.utcnow(),
        entry_price=entry_price,
        shares=shares,
        stop_loss=stop_loss,
        target=target,
        thesis=thesis,
        expected_r_multiple=expected_r_multiple,
        chart_screenshot_path=chart_screenshot_path,
        is_paper=is_paper,
    )
    session.add(trade)
    session.flush()
    logger.info(
        "Opened trade %s ticker=%s shares=%s entry=%.2f", trade.id, trade.ticker, shares, entry_price
    )
    return trade


def close_trade(
    session: Session,
    trade_id: int,
    exit_price: float,
    exit_reason: str,
    expected_fill_price: float | None = None,
) -> Trade:
    """Close an open trade and compute realized performance. TE-002.

    `exit_reason` should be one of the `ExitReason` enum values
    (stop/target/manual/earnings/regime_change), passed as a plain string
    for caller convenience and coerced to the enum here (raises
    `ValueError` if not a valid member — same as any bad enum value).

    Computes:
      realized_pnl = (exit_price - entry_price) * shares
      realized_r_multiple = (exit_price - entry_price) / (entry_price - stop_loss)
          if stop_loss is set and != entry_price, else None
      holding_period_days = (exit_date - entry_date).days
      slippage = exit_price - expected_fill_price, if expected_fill_price given
    """
    trade = session.get(Trade, trade_id)
    if trade is None:
        raise ValueError(f"Trade {trade_id} not found")

    exit_date = dt.datetime.utcnow()
    trade.exit_date = exit_date
    trade.exit_price = exit_price
    trade.exit_reason = ExitReason(exit_reason)
    trade.realized_pnl = (exit_price - trade.entry_price) * trade.shares

    if trade.stop_loss is not None and trade.entry_price != trade.stop_loss:
        risk_per_share = trade.entry_price - trade.stop_loss
        trade.realized_r_multiple = (exit_price - trade.entry_price) / risk_per_share
    else:
        trade.realized_r_multiple = None

    trade.holding_period_days = (exit_date - trade.entry_date).days

    if expected_fill_price is not None:
        trade.expected_fill_price = expected_fill_price
        trade.slippage = exit_price - expected_fill_price

    session.flush()
    logger.info(
        "Closed trade %s ticker=%s exit=%.2f reason=%s pnl=%.2f",
        trade.id,
        trade.ticker,
        exit_price,
        trade.exit_reason.value,
        trade.realized_pnl,
    )
    return trade


def get_open_trades(session: Session, portfolio_id: int) -> list[Trade]:
    """Return all trades for a portfolio with no `exit_date` set."""
    stmt = select(Trade).where(Trade.portfolio_id == portfolio_id, Trade.exit_date.is_(None))
    return list(session.execute(stmt).scalars().all())


def get_closed_trades(
    session: Session,
    portfolio_id: int,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> list[Trade]:
    """Return closed trades for a portfolio, optionally bounded by `exit_date` range."""
    stmt = select(Trade).where(Trade.portfolio_id == portfolio_id, Trade.exit_date.is_not(None))
    if start is not None:
        stmt = stmt.where(Trade.exit_date >= start)
    if end is not None:
        stmt = stmt.where(Trade.exit_date <= end)
    return list(session.execute(stmt).scalars().all())

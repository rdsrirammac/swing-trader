"""SRS 3.1 Portfolio Management (PF-001..PF-007).

Holding lifecycle (open/trim/adjust-stop/close), ticker-universe admin
(add/remove), portfolio summary, and the PF-007 watchlist trigger evaluator.

Separation of concerns: `add_ticker` only inserts a `TickerUniverse` row
(status=pending) -- it does NOT run the TB-001..006 auto-backfill pipeline.
That is `swing_trader.portfolio.backfill.run_backfill`, invoked separately
(e.g. by the CLI right after `add_ticker`, or by a scheduler).
"""
from __future__ import annotations

import datetime as dt
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import (
    ExitReason,
    Holding,
    Portfolio,
    PositionStatus,
    TickerStatus,
    TickerUniverse,
    Trade,
    WatchlistItem,
)
from swing_trader.logging_setup import get_logger
from swing_trader.signals.risk_manager import portfolio_heat

logger = get_logger("portfolio.manager")


# ---------------------------------------------------------------------------
# PF-001: create portfolio
# ---------------------------------------------------------------------------

def create_portfolio(
    session: Session,
    name: str,
    cash_balance: float = 100_000.0,
    is_paper: bool = True,
) -> Portfolio:
    """PF-001: create a new (by default paper) portfolio, seeded with the
    system-wide risk defaults from config so risk_manager has sane fallbacks
    even if the Portfolio row's own fields aren't consulted everywhere yet."""
    settings = get_settings()
    portfolio = Portfolio(
        name=name,
        cash_balance=cash_balance,
        max_risk_per_trade_pct=settings.get("risk.max_risk_per_trade_pct", 0.02),
        max_portfolio_heat_pct=settings.get("risk.max_portfolio_heat_pct", 0.20),
        is_paper=is_paper,
    )
    session.add(portfolio)
    session.flush()
    return portfolio


# ---------------------------------------------------------------------------
# PF-002 / PF-003: ticker universe admin
# ---------------------------------------------------------------------------

def add_ticker(session: Session, ticker: str) -> TickerUniverse:
    """PF-002: register `ticker` in the universe.

    - If `ticker` is not yet present, inserts a new row with status=pending
      (liquidity/price/ATR screening is deferred to TB-006's `screen_ticker`,
      invoked from `backfill.run_backfill` -- a fresh ticker has no known
      avg_daily_volume yet, so there's nothing to validate here).
    - If `ticker` already exists and its avg_daily_volume is already known
      (populated by a prior backfill run), it IS validated against
      `ticker_universe.min_avg_daily_volume`, raising ValueError if it now
      fails the liquidity floor. A previously-inactive ticker that still
      passes is reactivated to status=pending (ready for another backfill).
    """
    settings = get_settings()
    ticker = ticker.upper()
    min_volume = settings.get("ticker_universe.min_avg_daily_volume", 1_000_000)

    existing = session.execute(
        select(TickerUniverse).where(TickerUniverse.ticker == ticker)
    ).scalar_one_or_none()

    if existing is not None:
        if existing.avg_daily_volume is not None and existing.avg_daily_volume < min_volume:
            raise ValueError(
                f"{ticker} avg_daily_volume {existing.avg_daily_volume:,.0f} is below the "
                f"liquidity floor {min_volume:,.0f} (ticker_universe.min_avg_daily_volume)"
            )
        if existing.status == TickerStatus.INACTIVE:
            existing.status = TickerStatus.PENDING
        return existing

    row = TickerUniverse(ticker=ticker, status=TickerStatus.PENDING)
    session.add(row)
    session.flush()
    return row


def remove_ticker(session: Session, ticker: str, retain_history: bool = True) -> None:
    """PF-003: remove `ticker` from active use.

    - Closes any non-closed Holdings for `ticker` (administrative close, not
      a market exit: no exit price is fabricated; a note is left instead).
      If an open (exit_date is null) Trade journal row references the
      holding, its exit_date/exit_reason are set so it isn't left dangling.
    - Deletes all WatchlistItem rows for `ticker`.
    - If retain_history=True, sets TickerUniverse.status=inactive (row kept
      for historical price/feature/prediction data); if False, deletes the
      TickerUniverse row outright.
    """
    ticker = ticker.upper()
    note = "closed on ticker removal"

    active_holdings = session.execute(
        select(Holding).where(Holding.ticker == ticker, Holding.status != PositionStatus.CLOSED)
    ).scalars().all()
    for h in active_holdings:
        h.status = PositionStatus.CLOSED
        h.notes = f"{h.notes}; {note}" if h.notes else note

        open_trade = session.execute(
            select(Trade).where(Trade.holding_id == h.id, Trade.exit_date.is_(None))
        ).scalar_one_or_none()
        if open_trade is not None:
            open_trade.exit_date = dt.datetime.utcnow()
            open_trade.exit_reason = ExitReason.MANUAL
            open_trade.notes = f"{open_trade.notes}; {note}" if open_trade.notes else note

    watchlist_items = session.execute(
        select(WatchlistItem).where(WatchlistItem.ticker == ticker)
    ).scalars().all()
    for item in watchlist_items:
        session.delete(item)

    ticker_row = session.execute(
        select(TickerUniverse).where(TickerUniverse.ticker == ticker)
    ).scalar_one_or_none()
    if ticker_row is None:
        return

    if retain_history:
        ticker_row.status = TickerStatus.INACTIVE
        ticker_row.retain_history_on_removal = True
    else:
        ticker_row.retain_history_on_removal = False
        session.delete(ticker_row)


# ---------------------------------------------------------------------------
# PF-004 / PF-005: holding lifecycle
# ---------------------------------------------------------------------------

def enter_position(
    session: Session,
    portfolio_id: int,
    ticker: str,
    shares: float,
    entry_price: float,
    stop_loss: float | None = None,
    atr_14: float | None = None,
    thesis: str | None = None,
) -> Holding:
    """PF-004: open a new Holding.

    If `stop_loss` is None it is auto-calculated as
    `entry_price - positions.default_stop_atr_multiple * atr_14` (requires
    `atr_14`; raises ValueError if both are None). take_profit_1/2 are
    auto-calculated at `positions.target_1_atr_multiple` /
    `positions.target_2_atr_multiple` x ATR whenever `atr_14` is available
    (independent of whether stop_loss was supplied explicitly).
    """
    settings = get_settings()
    stop_mult = settings.get("positions.default_stop_atr_multiple", 2.0)
    target1_mult = settings.get("positions.target_1_atr_multiple", 2.0)
    target2_mult = settings.get("positions.target_2_atr_multiple", 3.5)

    if stop_loss is None:
        if atr_14 is None:
            raise ValueError(
                "enter_position: stop_loss is None and atr_14 is None -- cannot "
                "auto-calculate a stop. Supply one or the other."
            )
        stop_loss = entry_price - (stop_mult * atr_14)

    take_profit_1 = take_profit_2 = None
    if atr_14 is not None:
        take_profit_1 = entry_price + (target1_mult * atr_14)
        take_profit_2 = entry_price + (target2_mult * atr_14)

    holding = Holding(
        portfolio_id=portfolio_id,
        ticker=ticker.upper(),
        shares=shares,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        thesis=thesis,
        status=PositionStatus.ACTIVE,
    )
    session.add(holding)
    session.flush()
    return holding


def trim_position(session: Session, holding_id: int, pct: float = 0.5) -> Holding:
    """PF-005: reduce shares by `pct` (default positions.trim_target_1_pct
    convention of 50%) and mark status=trimmed."""
    holding = session.get(Holding, holding_id)
    if holding is None:
        raise ValueError(f"Holding {holding_id} not found")
    if holding.status == PositionStatus.CLOSED:
        raise ValueError(f"Holding {holding_id} is already closed; cannot trim")
    if not 0 < pct < 1:
        raise ValueError("pct must be strictly between 0 and 1")

    holding.shares = holding.shares * (1 - pct)
    holding.status = PositionStatus.TRIMMED
    session.flush()
    return holding


def adjust_stop(session: Session, holding_id: int, new_stop: float) -> Holding:
    """PF-005: manual/administrative stop-loss setter with a status guard.

    Real trailing-stop ratchet logic lives in
    `signals.risk_manager.update_trailing_stops`; this is just the plain
    setter used for manual adjustments (e.g. CLI `adjust-stop` command).
    """
    holding = session.get(Holding, holding_id)
    if holding is None:
        raise ValueError(f"Holding {holding_id} not found")
    if holding.status == PositionStatus.CLOSED:
        raise ValueError(f"Holding {holding_id} is closed; cannot adjust stop")

    holding.stop_loss = new_stop
    session.flush()
    return holding


def close_position(session: Session, holding_id: int, exit_price: float, exit_reason: str) -> Holding:
    """PF-005: close a Holding.

    Only updates the Holding itself (status=closed + a note). Writing the
    corresponding closing `Trade` journal row (realized_pnl, r_multiple,
    holding_period_days, ...) belongs to the execution/journal module.
    """
    holding = session.get(Holding, holding_id)
    if holding is None:
        raise ValueError(f"Holding {holding_id} not found")

    holding.status = PositionStatus.CLOSED
    note = f"closed @ {exit_price} ({exit_reason})"
    holding.notes = f"{holding.notes}; {note}" if holding.notes else note
    session.flush()

    # TODO(execution): write closing Trade row (exit_date/exit_price/
    # exit_reason/realized_pnl/realized_r_multiple/holding_period_days).
    return holding


# ---------------------------------------------------------------------------
# PF-006: portfolio summary
# ---------------------------------------------------------------------------

def portfolio_summary(session: Session, portfolio_id: int) -> dict:
    """PF-006: cash, estimated holdings market value, portfolio heat, and
    win/loss stats from closed Trade rows.

    NOTE: holdings market value uses entry_price (no live mark-to-market
    quote feed is wired into this module) -- a documented approximation,
    same caveat as `risk_manager.portfolio_heat`.
    """
    portfolio = session.get(Portfolio, portfolio_id)
    if portfolio is None:
        raise ValueError(f"Portfolio {portfolio_id} not found")

    holdings = session.execute(
        select(Holding).where(Holding.portfolio_id == portfolio_id, Holding.status != PositionStatus.CLOSED)
    ).scalars().all()
    holdings_market_value = sum(h.shares * h.entry_price for h in holdings)

    heat = portfolio_heat(session, portfolio_id)

    closed_trades = session.execute(
        select(Trade).where(Trade.portfolio_id == portfolio_id, Trade.exit_date.is_not(None))
    ).scalars().all()
    wins = [t for t in closed_trades if (t.realized_pnl or 0) > 0]
    win_rate = (len(wins) / len(closed_trades)) if closed_trades else None

    return {
        "cash_balance": portfolio.cash_balance,
        "holdings_market_value_estimate": holdings_market_value,
        "total_value_estimate": portfolio.cash_balance + holdings_market_value,
        "portfolio_heat_pct": heat,
        "num_holdings": len(holdings),
        "closed_trade_count": len(closed_trades),
        "win_count": len(wins),
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# PF-007: watchlist
# ---------------------------------------------------------------------------

def add_to_watchlist(session: Session, portfolio_id: int, ticker: str, trigger_condition: str) -> WatchlistItem:
    """PF-007: add a ticker + trigger_condition string (e.g.
    "RSI14<30 AND prob_5pct_up_10d>0.65") to the portfolio's watchlist."""
    item = WatchlistItem(
        portfolio_id=portfolio_id,
        ticker=ticker.upper(),
        trigger_condition=trigger_condition,
        triggered=False,
    )
    session.add(item)
    session.flush()
    return item


# Restricted grammar: FIELD OP VALUE (AND FIELD OP VALUE)* (OR ...)*
# OP in <, >, <=, >=, ==. No eval() of arbitrary text -- a small hand-rolled
# tokenizer/evaluator instead. OR has lower precedence than AND (standard
# boolean short-circuit semantics: true if any OR-group has all its AND
# clauses true).
_COND_CLAUSE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")


def _eval_clause(clause: str, context: dict) -> bool:
    match = _COND_CLAUSE_RE.match(clause)
    if not match:
        raise ValueError(f"unsupported watchlist condition clause: {clause!r}")
    field, op, value_str = match.groups()
    if field not in context or context[field] is None:
        return False
    lhs, value = float(context[field]), float(value_str)
    if op == "<":
        return lhs < value
    if op == ">":
        return lhs > value
    if op == "<=":
        return lhs <= value
    if op == ">=":
        return lhs >= value
    return lhs == value  # op == "=="


def _evaluate_condition(condition: str, context: dict) -> bool:
    or_groups = re.split(r"\s+OR\s+", condition, flags=re.IGNORECASE)
    for group in or_groups:
        and_clauses = re.split(r"\s+AND\s+", group, flags=re.IGNORECASE)
        if all(_eval_clause(c, context) for c in and_clauses):
            return True
    return False


def evaluate_watchlist(
    session: Session,
    portfolio_id: int,
    feature_row: dict,
    prediction_row: dict,
) -> list[WatchlistItem]:
    """PF-007: evaluate every untriggered WatchlistItem for `portfolio_id`
    against a merged {**feature_row, **prediction_row} context dict using the
    restricted safe-eval grammar above, setting/persisting `triggered=True`
    on any that fire.

    Unparsable trigger_condition strings are logged and skipped (not raised)
    so one bad watchlist entry doesn't block evaluating the rest.

    Returns the list of items that newly triggered on this call.
    """
    context = {**(feature_row or {}), **(prediction_row or {})}

    items = session.execute(
        select(WatchlistItem).where(
            WatchlistItem.portfolio_id == portfolio_id, WatchlistItem.triggered.is_(False)
        )
    ).scalars().all()

    newly_triggered: list[WatchlistItem] = []
    for item in items:
        try:
            if _evaluate_condition(item.trigger_condition, context):
                item.triggered = True
                newly_triggered.append(item)
        except ValueError as e:
            logger.warning("Skipping unparsable watchlist condition for %s: %s", item.ticker, e)

    session.flush()
    return newly_triggered

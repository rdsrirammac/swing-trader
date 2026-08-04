"""Performance analytics & trade journaling on real (closed) trades (SRS 3.12, PA-001/PA-002).

Reuses the exact metric formulas from `swing_trader.backtest.engine
.compute_backtest_metrics` (rather than re-deriving them) so that
backtested and live performance numbers are always computed identically.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.backtest.engine import compute_backtest_metrics
from swing_trader.db.models import Portfolio, Trade
from swing_trader.logging_setup import get_logger

logger = get_logger("analytics.performance")

DEFAULT_STARTING_CAPITAL = 100000.0


def _closed_trades_query(portfolio_id: int, start: dt.date | None, end: dt.date | None):
    stmt = (
        select(Trade)
        .where(Trade.portfolio_id == portfolio_id, Trade.exit_date.is_not(None))
        .order_by(Trade.exit_date)
    )
    if start is not None:
        stmt = stmt.where(Trade.exit_date >= start)
    if end is not None:
        stmt = stmt.where(Trade.exit_date <= end)
    return stmt


def trade_journal_view(
    session: Session,
    portfolio_id: int,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> list[dict]:
    """PA-001: closed-trade journal for a portfolio.

    Args:
        session: an active SQLAlchemy session (see `swing_trader.db.base
            .session_scope`).
        portfolio_id: the `Portfolio.id` to journal.
        start / end: optional inclusive date bounds on `Trade.exit_date`.

    Returns:
        List of dicts (one per closed trade, ordered by exit_date), each
        with: ticker, entry_date, entry_price, exit_date, exit_price,
        shares, stop_loss, target, pnl, r_multiple, holding_period_days,
        exit_reason, thesis, notes.
    """
    trades = session.execute(_closed_trades_query(portfolio_id, start, end)).scalars().all()
    return [
        {
            "trade_id": t.id,
            "ticker": t.ticker,
            "entry_date": t.entry_date,
            "entry_price": t.entry_price,
            "exit_date": t.exit_date,
            "exit_price": t.exit_price,
            "shares": t.shares,
            "stop_loss": t.stop_loss,
            "target": t.target,
            "pnl": t.realized_pnl,
            "r_multiple": t.realized_r_multiple,
            "holding_period_days": t.holding_period_days,
            "exit_reason": t.exit_reason.value if t.exit_reason else None,
            "thesis": t.thesis,
            "notes": t.notes,
        }
        for t in trades
    ]


def _trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    """Adapt closed `Trade` ORM rows into the dataframe shape expected by
    `compute_backtest_metrics` (entry_date/exit_date/pnl/r_multiple)."""
    rows = [
        {
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.realized_pnl if t.realized_pnl is not None else 0.0,
            "r_multiple": t.realized_r_multiple,
            "exit_reason": t.exit_reason.value if t.exit_reason else None,
            "holding_period_days": t.holding_period_days,
        }
        for t in trades
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "pnl",
            "r_multiple",
            "exit_reason",
            "holding_period_days",
        ],
    )


def _r_multiple_histogram(r_values: pd.Series, n_bins: int = 10) -> list[tuple[float, int]]:
    """Best-effort histogram of R-multiples as (bin_left_edge, count) pairs."""
    r_values = r_values.dropna()
    if len(r_values) == 0:
        return []
    counts, edges = np.histogram(r_values.to_numpy(), bins=n_bins)
    return [(float(edges[i]), int(counts[i])) for i in range(len(counts))]


def compute_performance_metrics(session: Session, portfolio_id: int) -> dict:
    """PA-002: full performance metrics for a portfolio's real closed trades.

    Reuses `swing_trader.backtest.engine.compute_backtest_metrics` for the
    core return/risk formulas so live and simulated metrics never diverge;
    adds journal-specific breakdowns (R-multiple histogram, time-in-market,
    drawdown duration) on top.

    Returns:
        {
          "overall": {"total_return", "cagr", "sharpe", "sortino", "calmar"},
          "win_loss": {"win_rate", "avg_win", "avg_loss", "profit_factor"},
          "r_multiples": {"avg_r", "max_r", "histogram": list[(bin_edge, count)]},
          "time": {"avg_hold_time_days", "time_in_market_pct"},
          "drawdown": {"max_dd", "dd_duration_days"},
        }
    """
    portfolio = session.get(Portfolio, portfolio_id)
    starting_capital = (
        portfolio.cash_balance if portfolio and portfolio.cash_balance else DEFAULT_STARTING_CAPITAL
    )

    trades = session.execute(_closed_trades_query(portfolio_id, None, None)).scalars().all()
    if not trades:
        empty_metrics = compute_backtest_metrics(pd.DataFrame(columns=["entry_date", "exit_date", "pnl"]))
        return {
            "overall": {
                "total_return": empty_metrics["total_return"],
                "cagr": empty_metrics["cagr"],
                "sharpe": empty_metrics["sharpe_ratio"],
                "sortino": empty_metrics["sortino_ratio"],
                "calmar": None,
            },
            "win_loss": {
                "win_rate": empty_metrics["win_rate"],
                "avg_win": None,
                "avg_loss": None,
                "profit_factor": empty_metrics["profit_factor"],
            },
            "r_multiples": {"avg_r": None, "max_r": None, "histogram": []},
            "time": {"avg_hold_time_days": None, "time_in_market_pct": None},
            "drawdown": {"max_dd": empty_metrics["max_drawdown"], "dd_duration_days": None},
        }

    df = _trades_to_frame(trades)
    base_metrics = compute_backtest_metrics(df, starting_capital=starting_capital)

    calmar = None
    if base_metrics["cagr"] is not None and base_metrics["max_drawdown"]:
        calmar = base_metrics["cagr"] / base_metrics["max_drawdown"]

    pnl = df["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) else None
    avg_loss = float(losses.mean()) if len(losses) else None

    r = df["r_multiple"].dropna().astype(float)
    avg_r = float(r.mean()) if len(r) else None
    max_r = float(r.max()) if len(r) else None
    histogram = _r_multiple_histogram(df["r_multiple"])

    hold_days = df["holding_period_days"].dropna().astype(float)
    avg_hold = float(hold_days.mean()) if len(hold_days) else None

    entry_dates = pd.to_datetime(df["entry_date"])
    exit_dates = pd.to_datetime(df["exit_date"])
    span_days = max((exit_dates.max() - entry_dates.min()).days, 1)
    time_in_market_pct = float(hold_days.sum() / span_days) if len(hold_days) and span_days else None

    # Drawdown duration: best-effort, computed from the realized_pnl
    # cumulative series ordered by exit_date -- i.e. the number of days
    # between a peak in the (exit-date-indexed) equity curve and its
    # subsequent recovery. This is an approximation of true intraday/
    # intraperiod drawdown duration since we only observe equity at trade
    # close events, not continuously.
    equity = starting_capital + pnl.cumsum().to_numpy()
    dd_duration_days = None
    if len(equity) > 0:
        running_max = np.maximum.accumulate(equity)
        peak_idx = 0
        max_duration = 0
        exit_dates_arr = pd.to_datetime(df["exit_date"]).to_numpy()
        for i in range(len(equity)):
            if equity[i] >= running_max[i]:
                peak_idx = i
            else:
                duration = (exit_dates_arr[i] - exit_dates_arr[peak_idx]).astype("timedelta64[D]").astype(int)
                max_duration = max(max_duration, duration)
        dd_duration_days = int(max_duration)

    return {
        "overall": {
            "total_return": base_metrics["total_return"],
            "cagr": base_metrics["cagr"],
            "sharpe": base_metrics["sharpe_ratio"],
            "sortino": base_metrics["sortino_ratio"],
            "calmar": calmar,
        },
        "win_loss": {
            "win_rate": base_metrics["win_rate"],
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": base_metrics["profit_factor"],
        },
        "r_multiples": {"avg_r": avg_r, "max_r": max_r, "histogram": histogram},
        "time": {"avg_hold_time_days": avg_hold, "time_in_market_pct": time_in_market_pct},
        "drawdown": {"max_dd": base_metrics["max_drawdown"], "dd_duration_days": dd_duration_days},
    }


def summarize_model_performance(session: Session, portfolio_id: int | None = None) -> dict:
    """Dashboard integration convenience wrapper (Model Performance panel,
    DV-001) around `compute_performance_metrics`, flattened to the handful
    of headline numbers the dashboard's `st.metric` tiles want: win_rate,
    avg_r_multiple, sharpe_ratio. If `portfolio_id` is omitted, uses the
    first portfolio found. Returns an all-None dict if there are no
    portfolios / no closed trades yet, rather than raising, so the caller
    can render a friendly "no data yet" state.
    """
    if portfolio_id is None:
        portfolio = session.execute(select(Portfolio).order_by(Portfolio.id)).scalars().first()
        if portfolio is None:
            return {"win_rate": None, "avg_r_multiple": None, "sharpe_ratio": None}
        portfolio_id = portfolio.id

    metrics = compute_performance_metrics(session, portfolio_id)
    return {
        "win_rate": metrics["win_loss"]["win_rate"],
        "avg_r_multiple": metrics["r_multiples"]["avg_r"],
        "sharpe_ratio": metrics["overall"]["sharpe"],
    }

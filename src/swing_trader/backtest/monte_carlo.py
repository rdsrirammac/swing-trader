"""Monte Carlo simulation & Kelly sizing on a historical trade log (SRS 3.11, BT-004).

Both functions here are purely statistical post-processors over a trade log
(as produced by `swing_trader.backtest.engine.simulate_trades` or adapted
from real `Trade` rows) -- neither touches the database directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_trader.config import get_settings
from swing_trader.logging_setup import get_logger

logger = get_logger("backtest.monte_carlo")


def _extract_trade_pnl(trade_log: pd.DataFrame) -> np.ndarray:
    if "pnl" not in trade_log.columns:
        raise ValueError("trade_log must contain a 'pnl' column")
    return trade_log["pnl"].astype(float).to_numpy()


def run_monte_carlo(
    trade_log: pd.DataFrame,
    n_runs: int | None = None,
    starting_capital: float = 100000.0,
) -> dict:
    """BT-004: resample historical trades to build a distribution of outcomes.

    Resamples-with-replacement the historical per-trade dollar P&L
    `n_runs` times (bootstrap), each run drawing the same number of trades
    as the original trade_log, to build a distribution of simulated equity
    curves. This treats trade outcomes as i.i.d. draws from the empirical
    distribution of historical trades -- a standard (if simplified)
    Monte Carlo bootstrap; it does not model serial correlation between
    trades (e.g. regime persistence).

    Args:
        trade_log: dataframe with at least a `pnl` column (dollar P&L per
            trade); `entry_date`/`exit_date` are used if present to estimate
            an annualization factor for `annual_return_ci`.
        n_runs: number of bootstrap resamples. Defaults to
            `backtest.monte_carlo_runs`.
        starting_capital: notional starting account size for the simulated
            equity curves.

    Returns:
        {
          "probability_of_ruin": float,   # fraction of runs where equity <= 0 at any point
          "drawdown_distribution": {"p5": .., "p50": .., "p95": ..},
          "annual_return_ci": {"p5": .., "p50": .., "p95": ..},
        }
    """
    settings = get_settings()
    n_runs = n_runs if n_runs is not None else settings.get("backtest.monte_carlo_runs", 1000)

    pnl = _extract_trade_pnl(trade_log)
    n_trades = len(pnl)
    if n_trades == 0:
        return {
            "probability_of_ruin": 0.0,
            "drawdown_distribution": {"p5": 0.0, "p50": 0.0, "p95": 0.0},
            "annual_return_ci": {"p5": 0.0, "p50": 0.0, "p95": 0.0},
        }

    # Annualization factor: how many "trade-log spans" fit in a year, based
    # on the observed date range. Falls back to 1.0 (no annualization) if
    # date columns are unavailable.
    annualization_factor = 1.0
    if "entry_date" in trade_log.columns and "exit_date" in trade_log.columns:
        entry = pd.to_datetime(trade_log["entry_date"])
        exit_ = pd.to_datetime(trade_log["exit_date"])
        span_days = max((exit_.max() - entry.min()).days, 1)
        annualization_factor = 365.25 / span_days

    rng = np.random.default_rng()
    max_drawdowns = np.empty(n_runs)
    annual_returns = np.empty(n_runs)
    ruin_count = 0

    for i in range(n_runs):
        sample = rng.choice(pnl, size=n_trades, replace=True)
        equity_curve = starting_capital + np.cumsum(sample)

        running_max = np.maximum.accumulate(np.concatenate(([starting_capital], equity_curve)))[1:]
        drawdowns = np.where(running_max > 0, (running_max - equity_curve) / running_max, 0.0)
        max_drawdowns[i] = drawdowns.max() if len(drawdowns) else 0.0

        if equity_curve.min() <= 0:
            ruin_count += 1

        total_return = (equity_curve[-1] - starting_capital) / starting_capital
        annual_returns[i] = (
            (1 + total_return) ** annualization_factor - 1 if total_return > -1 else -1.0
        )

    probability_of_ruin = ruin_count / n_runs

    def _pctiles(arr: np.ndarray) -> dict:
        return {
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "probability_of_ruin": float(probability_of_ruin),
        "drawdown_distribution": _pctiles(max_drawdowns),
        "annual_return_ci": _pctiles(annual_returns),
    }


def optimal_kelly_fraction(trade_log: pd.DataFrame) -> float:
    """BT-004: Kelly-optimal position size fraction from historical trade stats.

    f* = W - (1 - W) / R

    where W = historical win rate and R = avg_win / avg_loss (ratio of mean
    winning trade size to mean losing trade size, both in absolute dollar
    terms).

    Full Kelly (f*) is well known to be far too aggressive for practical use
    -- it assumes exact, stationary knowledge of the true win-rate/payoff
    distribution and produces large equity swings even when correct. This
    function therefore clips the result to [0, 0.25] ("quarter-Kelly at
    most") as a safety bound; callers wanting a more conservative sizing
    should further scale this down (e.g. half of the returned value).

    Returns:
        Kelly fraction in [0, 0.25]. Returns 0.0 if there is insufficient
        data (no losing trades, no winning trades, or an empty trade_log).
    """
    pnl = _extract_trade_pnl(trade_log)
    if len(pnl) == 0:
        return 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    win_rate = len(wins) / len(pnl)
    avg_win = float(wins.mean())
    avg_loss = float(-losses.mean())
    if avg_loss == 0:
        return 0.0

    payoff_ratio = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / payoff_ratio

    return float(np.clip(kelly, 0.0, 0.25))

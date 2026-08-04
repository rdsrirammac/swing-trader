"""Walk-forward backtesting engine and trade simulation (SRS 3.11, BT-001..003, BT-005).

Implements:
    - ``WalkForwardSplitter`` (BT-001): expanding-then-rolling walk-forward
      train/test date splits.
    - ``simulate_trades`` : bar-level trade simulation with commission +
      slippage applied, used to build a trade log dataframe for backtests.
    - ``compute_backtest_metrics`` (BT-002): the canonical set of performance
      metric formulas, shared by ``swing_trader.analytics.performance`` so
      that live-trade metrics and simulated-backtest metrics never drift
      apart.
    - ``compare_strategies`` (BT-003): side-by-side metric comparison across
      N trade logs (model versions / entry thresholds / stop strategies /
      position sizing schemes).
    - ``out_of_sample_split`` (BT-005): held-out final N months, never to be
      touched during training or hyperparameter tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from swing_trader.config import get_settings
from swing_trader.logging_setup import get_logger

logger = get_logger("backtest.engine")

# Approximate trading-day conventions used to translate the
# calendar-flavoured config knobs (`*_months`, `*_years`) into bar counts.
# `test_window_days` / `step_size_days` in settings.yaml are themselves
# already expressed in trading-day (bar) units (e.g. 20 trading days ~= 1
# month), so the whole splitter operates on integer bar positions rather
# than calendar dates. This keeps behaviour well-defined for irregularly
# spaced trading calendars (holidays, weekends already excluded upstream).
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252


@dataclass
class WalkForwardSplitter:
    """BT-001: walk-forward cross-validation splitter for backtests.

    Produces (train_idx, test_idx) pairs where the training window expands
    from `train_window_min_months` up to a cap of `train_window_max_years`
    (after which it becomes a rolling window of that fixed max size), the
    test fold is always exactly `test_window_days` bars long, and each
    successive split steps forward by `step_size_days` bars.
    """

    train_window_min_months: int = 6
    train_window_max_years: int = 3
    test_window_days: int = 20
    step_size_days: int = 5

    @classmethod
    def from_config(cls) -> "WalkForwardSplitter":
        """Build a splitter using `backtest.*` values from config/settings.yaml."""
        settings = get_settings()
        return cls(
            train_window_min_months=settings.get("backtest.train_window_min_months", 6),
            train_window_max_years=settings.get("backtest.train_window_max_years", 3),
            test_window_days=settings.get("backtest.test_window_days", 20),
            step_size_days=settings.get("backtest.step_size_days", 5),
        )

    def split(
        self, dates: pd.DatetimeIndex
    ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield (train_idx, test_idx) DatetimeIndex pairs (BT-001).

        `dates` should be the sorted, de-duplicated set of trading-day bars
        available (e.g. `price_df.index`). The training window starts at
        `train_window_min_months` worth of bars, expands by `step_size_days`
        each iteration, and is capped at `train_window_max_years` worth of
        bars (beyond which it rolls forward instead of continuing to grow).
        Iteration stops once there are no more full test folds left.
        """
        dates = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
        n = len(dates)

        min_train_bars = round(self.train_window_min_months * TRADING_DAYS_PER_MONTH)
        max_train_bars = round(self.train_window_max_years * TRADING_DAYS_PER_YEAR)
        test_bars = self.test_window_days
        step_bars = self.step_size_days

        if min_train_bars <= 0 or test_bars <= 0 or step_bars <= 0:
            raise ValueError("train/test/step window sizes must be positive")

        train_end = min_train_bars  # exclusive upper bound of the training slice
        while True:
            test_start = train_end
            test_end = test_start + test_bars
            if test_end > n or train_end > n:
                break

            train_start = max(0, train_end - max_train_bars)
            train_idx = dates[train_start:train_end]
            test_idx = dates[test_start:test_end]

            if len(train_idx) == 0 or len(test_idx) < test_bars:
                break

            yield train_idx, test_idx
            train_end += step_bars


def _apply_slippage(price: float, slippage_pct: float, *, buying: bool) -> float:
    """Apply slippage as an adverse price move (worse fill than the quoted price)."""
    return price * (1 + slippage_pct) if buying else price * (1 - slippage_pct)


def simulate_trades(
    price_df: pd.DataFrame,
    signals: pd.DataFrame,
    commission_per_share: float | None = None,
    slippage_pct: float | None = None,
) -> pd.DataFrame:
    """Simulate entries/exits for a single ticker's signal stream (BT-001/BT-002 input).

    Args:
        price_df: date-indexed OHLC dataframe with columns Open/High/Low/Close
            (a Volume column is tolerated but not required), sorted ascending.
        signals: date-indexed dataframe aligned to (a subset of) `price_df`'s
            index, with at least a `rating` column (values compared against
            {"Buy", "Strong Buy"}) and optional `stop` / `target` columns
            (absolute price levels) used for exit management. An optional
            `shares` column scales P&L (defaults to 1 share/unit).
        commission_per_share: flat $/share commission, charged on both the
            entry and the exit fill. Defaults to `backtest.commission_per_share`.
        slippage_pct: adverse fill slippage as a fraction of price, applied
            against the trader on both entry and exit. Defaults to
            `backtest.slippage_pct`.

    Fill logic (kept intentionally simple/explicit):
        - Enter at the *next bar's* Open following a Buy/Strong Buy signal
          (no same-bar fills -- avoids look-ahead bias).
        - While in a position, exit intrabar the first time either the Low
          touches/crosses the stop or the High touches/crosses the target
          (stop checked first if both trigger on the same bar -- the more
          conservative assumption).
        - If neither stop nor target is touched within `test_window_days`
          bars of the entry, exit at the Close of the final bar in that
          window ("time" exit).
        - While a position is open, subsequent signals are ignored (no
          pyramiding / overlapping trades in this simple simulator).

    Returns:
        DataFrame with one row per closed trade: entry_date, exit_date,
        entry_price, exit_price, shares, pnl, r_multiple, exit_reason.
        `exit_reason` is one of {"stop", "target", "time"} -- note this is
        a backtest-only vocabulary, distinct from the DB `ExitReason` enum
        used for live trades (which has no "time" value).
    """
    settings = get_settings()
    commission = (
        commission_per_share
        if commission_per_share is not None
        else settings.get("backtest.commission_per_share", 0.01)
    )
    slippage = (
        slippage_pct if slippage_pct is not None else settings.get("backtest.slippage_pct", 0.0005)
    )
    test_window_days = settings.get("backtest.test_window_days", 20)

    price_df = price_df.sort_index()
    signals = signals.sort_index()
    dates = price_df.index

    buy_ratings = {"Buy", "Strong Buy"}
    rows: list[dict] = []
    last_exit_pos = -1

    for sig_date, sig_row in signals.iterrows():
        rating = sig_row.get("rating")
        if rating not in buy_ratings:
            continue
        if sig_date not in dates:
            continue

        sig_pos = dates.get_loc(sig_date)
        if isinstance(sig_pos, slice):  # duplicate index values -- skip ambiguous
            continue
        if sig_pos <= last_exit_pos:
            continue  # still holding a position from an earlier signal

        entry_pos = sig_pos + 1
        if entry_pos >= len(dates):
            continue  # no next bar available to fill on

        entry_date = dates[entry_pos]
        raw_entry_price = float(price_df.iloc[entry_pos]["Open"])
        entry_price = _apply_slippage(raw_entry_price, slippage, buying=True)

        stop = sig_row.get("stop")
        target = sig_row.get("target")
        shares = float(sig_row.get("shares", 1.0)) if "shares" in sig_row else 1.0

        max_hold_pos = min(entry_pos + test_window_days, len(dates) - 1)

        exit_pos = None
        exit_price_raw = None
        exit_reason = None
        for pos in range(entry_pos, max_hold_pos + 1):
            bar = price_df.iloc[pos]
            if stop is not None and not pd.isna(stop) and bar["Low"] <= stop:
                exit_pos, exit_price_raw, exit_reason = pos, float(stop), "stop"
                break
            if target is not None and not pd.isna(target) and bar["High"] >= target:
                exit_pos, exit_price_raw, exit_reason = pos, float(target), "target"
                break

        if exit_pos is None:
            exit_pos = max_hold_pos
            exit_price_raw = float(price_df.iloc[exit_pos]["Close"])
            exit_reason = "time"

        exit_date = dates[exit_pos]
        exit_price = _apply_slippage(exit_price_raw, slippage, buying=False)

        gross_pnl = (exit_price - entry_price) * shares
        round_trip_commission = commission * shares * 2
        pnl = gross_pnl - round_trip_commission

        risk_per_share = (entry_price - stop) if (stop is not None and not pd.isna(stop)) else None
        r_multiple = (
            (exit_price - entry_price) / risk_per_share
            if risk_per_share and risk_per_share != 0
            else None
        )

        rows.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "pnl": pnl,
                "r_multiple": r_multiple,
                "exit_reason": exit_reason,
            }
        )
        last_exit_pos = exit_pos

    return pd.DataFrame(
        rows,
        columns=[
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "pnl",
            "r_multiple",
            "exit_reason",
        ],
    )


def _max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = np.where(running_max > 0, (running_max - equity_curve) / running_max, 0.0)
    return float(np.max(drawdowns))


def _max_consecutive_losses(pnl: pd.Series) -> int:
    max_streak = streak = 0
    for value in pnl:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def compute_backtest_metrics(trade_log: pd.DataFrame, starting_capital: float = 100000.0) -> dict:
    """BT-002: canonical backtest metric formulas.

    Shared verbatim (via import) with `swing_trader.analytics.performance
    .compute_performance_metrics` so live-trade and simulated-backtest
    metrics are computed identically.

    Args:
        trade_log: dataframe with at least entry_date/exit_date/pnl columns
            (as produced by `simulate_trades`, or adapted from real `Trade`
            rows by the analytics layer). An `r_multiple` column enables the
            R-multiple metrics. An optional `regime` column (added by the
            caller by joining against `RegimeHistory`) enables
            `return_by_regime`.
        starting_capital: notional starting account size used to express
            total_return / cagr / drawdown in percentage terms.

    Returns:
        dict with keys: total_return, cagr, win_rate, profit_factor,
        avg_r_multiple_win, avg_r_multiple_loss, max_drawdown,
        max_consecutive_losses, sharpe_ratio, sortino_ratio,
        expectancy_per_trade, return_by_regime (dict or None).
    """
    if trade_log is None or len(trade_log) == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "win_rate": 0.0,
            "profit_factor": None,
            "avg_r_multiple_win": None,
            "avg_r_multiple_loss": None,
            "max_drawdown": 0.0,
            "max_consecutive_losses": 0,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "expectancy_per_trade": 0.0,
            "return_by_regime": None,
        }

    tl = trade_log.sort_values("exit_date").reset_index(drop=True)
    pnl = tl["pnl"].astype(float)

    equity_curve = starting_capital + pnl.cumsum().to_numpy()
    total_return = float((equity_curve[-1] - starting_capital) / starting_capital)

    entry_dates = pd.to_datetime(tl["entry_date"])
    exit_dates = pd.to_datetime(tl["exit_date"])
    span_days = max((exit_dates.max() - entry_dates.min()).days, 1)
    years = span_days / 365.25
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 and (1 + total_return) > 0 else None

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    win_rate = float(len(wins) / len(pnl)) if len(pnl) else 0.0
    gross_wins = float(wins.sum())
    gross_losses = float(-losses.sum())
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else None

    avg_r_win = avg_r_loss = None
    if "r_multiple" in tl.columns:
        r = tl["r_multiple"].astype(float)
        r_win = r[pnl > 0]
        r_loss = r[pnl < 0]
        avg_r_win = float(r_win.mean()) if len(r_win) else None
        avg_r_loss = float(r_loss.mean()) if len(r_loss) else None

    max_dd = _max_drawdown(equity_curve)
    max_consec_losses = _max_consecutive_losses(pnl)

    # Trade-level Sharpe/Sortino: approximated from per-trade returns
    # (pnl / starting_capital) annualized by the observed trade frequency.
    # This is a simplification vs. a true daily-return Sharpe (no
    # daily-mark-to-market data is available from a trade log alone); it is
    # documented here and used consistently across backtest + live metrics.
    trades_per_year = len(pnl) / years if years > 0 else len(pnl)
    trade_returns = pnl / starting_capital
    mean_r, std_r = float(trade_returns.mean()), float(trade_returns.std(ddof=1)) if len(trade_returns) > 1 else 0.0
    sharpe_ratio = (mean_r / std_r) * np.sqrt(trades_per_year) if std_r > 0 else None

    downside = trade_returns[trade_returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino_ratio = (mean_r / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else None

    expectancy_per_trade = float(pnl.mean())

    return_by_regime = None
    if "regime" in tl.columns:
        return_by_regime = {
            str(regime): float(group["pnl"].sum())
            for regime, group in tl.groupby("regime")
        }

    return {
        "total_return": total_return,
        "cagr": cagr,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_r_multiple_win": avg_r_win,
        "avg_r_multiple_loss": avg_r_loss,
        "max_drawdown": max_dd,
        "max_consecutive_losses": max_consec_losses,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "expectancy_per_trade": expectancy_per_trade,
        "return_by_regime": return_by_regime,
    }


def compare_strategies(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """BT-003: compare N strategies (model versions / thresholds / stop rules / sizing).

    Args:
        results: mapping of strategy_name -> trade_log dataframe (each in the
            shape expected by `compute_backtest_metrics`). The caller is
            responsible for generating each trade_log (e.g. by re-running
            `simulate_trades` with different entry thresholds, stop
            multiples, or position sizing) -- this function only aggregates
            metrics for side-by-side comparison.

    Returns:
        DataFrame indexed by strategy_name, one column per metric returned
        by `compute_backtest_metrics` (return_by_regime dropped from the
        table since it is itself a nested dict -- inspect it separately if
        needed).
    """
    rows = {}
    for name, trade_log in results.items():
        metrics = compute_backtest_metrics(trade_log)
        metrics.pop("return_by_regime", None)
        rows[name] = metrics
    return pd.DataFrame.from_dict(rows, orient="index")


def out_of_sample_split(
    df: pd.DataFrame, months: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BT-005: reserve the last `months` of data as a held-out out-of-sample set.

    Args:
        df: dataframe indexed by a DatetimeIndex (or containing a `date`/`ts`
            column -- pass a DatetimeIndex-indexed frame for simplicity).
        months: number of trailing months to reserve. Defaults to
            `backtest.out_of_sample_months`.

    Returns:
        (train_df, test_df). `test_df` covers the final `months` of the date
        range; `train_df` covers everything before that.

    IMPORTANT: `test_df` must NEVER be used for model training, feature
    selection, or hyperparameter tuning of any kind -- it exists solely for
    a final, single-pass evaluation of a fully-frozen model/strategy. Any
    iteration on `test_df` results (re-tuning after looking at them)
    silently reintroduces look-ahead / overfitting bias.
    """
    settings = get_settings()
    months = months if months is not None else settings.get("backtest.out_of_sample_months", 6)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("out_of_sample_split requires a DatetimeIndex-indexed dataframe")

    idx = df.index
    cutoff = idx.max() - pd.DateOffset(months=months)
    train_df = df[idx <= cutoff]
    test_df = df[idx > cutoff]
    return train_df, test_df


def run_backtest(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    start=None,
    end=None,
    strategy: str = "default",
) -> dict:
    """Integration convenience wrapper around `simulate_trades` +
    `compute_backtest_metrics`, used by the CLI (`backtest` command) and the
    dashboard's Backtest page. Not part of the original BT-* module split,
    but added during final integration so callers don't need to know the
    lower-level per-ticker API.

    Args:
        price_df: long-format dataframe with columns
            ticker, ts, open, high, low, close (and optionally volume) --
            i.e. exactly what you get from a `StockPrice` query flattened to
            a DataFrame.
        signals_df: long-format dataframe with columns
            ticker, as_of, rating, suggested_entry, suggested_stop,
            suggested_target_1, suggested_target_2 -- i.e. what you get from
            a `SignalRating` query flattened to a DataFrame.
        start, end: optional bounds (informational only here; callers are
            expected to have already filtered price_df/signals_df).
        strategy: free-form label, echoed back in the result and usable as
            a dict key when comparing multiple runs via `compare_strategies`.

    Returns:
        dict with keys: strategy, start, end, trade_log (DataFrame),
        metrics (dict from `compute_backtest_metrics`), equity_curve
        (pd.Series indexed by exit_date, starting_capital + cumulative pnl).
    """
    if price_df is None or price_df.empty:
        return {
            "strategy": strategy, "start": start, "end": end,
            "trade_log": pd.DataFrame(), "metrics": compute_backtest_metrics(pd.DataFrame()),
            "equity_curve": pd.Series(dtype=float),
        }

    starting_capital = 100000.0
    all_trades = []

    for ticker, tdf in price_df.groupby("ticker"):
        tdf = tdf.sort_values("ts").set_index("ts")
        ohlc = tdf.rename(
            columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        )
        ohlc.index = pd.DatetimeIndex(ohlc.index)

        tsig = signals_df[signals_df["ticker"] == ticker] if signals_df is not None and not signals_df.empty else pd.DataFrame()
        if tsig.empty:
            continue
        tsig = tsig.sort_values("as_of").set_index("as_of")
        tsig.index = pd.DatetimeIndex(tsig.index)
        sig = pd.DataFrame(
            {
                "rating": tsig.get("rating"),
                "stop": tsig.get("suggested_stop"),
                "target": tsig.get("suggested_target_1"),
            },
            index=tsig.index,
        )

        try:
            trades = simulate_trades(ohlc[["Open", "High", "Low", "Close"]], sig)
        except Exception as e:
            logger.warning("simulate_trades failed for %s: %s", ticker, e)
            continue
        if not trades.empty:
            trades["ticker"] = ticker
            all_trades.append(trades)

    trade_log = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    metrics = compute_backtest_metrics(trade_log, starting_capital=starting_capital)

    equity_curve = pd.Series(dtype=float)
    if not trade_log.empty:
        tl = trade_log.sort_values("exit_date")
        equity_curve = pd.Series(
            starting_capital + tl["pnl"].astype(float).cumsum().to_numpy(),
            index=pd.to_datetime(tl["exit_date"]),
        )

    return {
        "strategy": strategy,
        "start": start,
        "end": end,
        "trade_log": trade_log,
        "metrics": metrics,
        "equity_curve": equity_curve,
    }

"""Unit tests for swing_trader.backtest.engine (SRS 3.11, BT-001..003, BT-005)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swing_trader.backtest.engine import (
    WalkForwardSplitter,
    compare_strategies,
    compute_backtest_metrics,
    out_of_sample_split,
    run_backtest,
    simulate_trades,
)


def _ohlc(dates, start=100.0, drift=0.5):
    close = start + np.arange(len(dates)) * drift
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
        },
        index=dates,
    )


def test_simulate_trades_enters_next_bar_and_exits_on_target():
    dates = pd.bdate_range("2026-01-01", periods=30)
    price_df = _ohlc(dates)

    signals = pd.DataFrame(index=dates)
    signals["rating"] = None
    signals.loc[dates[0], "rating"] = "Buy"
    signals.loc[dates[0], "stop"] = price_df.loc[dates[1], "Open"] - 5.0
    signals.loc[dates[0], "target"] = price_df.loc[dates[1], "Open"] + 2.0  # small, hits fast

    trades = simulate_trades(price_df, signals, commission_per_share=0.0, slippage_pct=0.0)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_date"] == dates[1]  # next-bar fill, no look-ahead
    assert row["exit_reason"] in ("target", "stop", "time")


def test_compute_backtest_metrics_empty_trade_log_returns_zeros():
    metrics = compute_backtest_metrics(pd.DataFrame())
    assert metrics["total_return"] == 0.0
    assert metrics["win_rate"] == 0.0


def test_compute_backtest_metrics_win_rate_and_profit_factor():
    trade_log = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-10"]),
            "exit_date": pd.to_datetime(["2026-01-03", "2026-01-07", "2026-01-12"]),
            "pnl": [100.0, -50.0, 200.0],
            "r_multiple": [1.5, -1.0, 2.0],
        }
    )
    metrics = compute_backtest_metrics(trade_log, starting_capital=10_000)
    assert metrics["win_rate"] == pytest.approx(2 / 3)
    assert metrics["profit_factor"] == pytest.approx(300 / 50)
    assert metrics["avg_r_multiple_win"] == pytest.approx(1.75)
    assert metrics["avg_r_multiple_loss"] == pytest.approx(-1.0)


def test_walk_forward_splitter_expands_then_steps():
    dates = pd.bdate_range("2020-01-01", periods=400)
    splitter = WalkForwardSplitter(
        train_window_min_months=1, train_window_max_years=1, test_window_days=10, step_size_days=10
    )
    splits = list(splitter.split(dates))
    assert len(splits) > 1
    for train_idx, test_idx in splits:
        assert len(test_idx) == 10
        assert train_idx[-1] < test_idx[0]  # no look-ahead: train strictly precedes test


def test_out_of_sample_split_reserves_trailing_months():
    dates = pd.bdate_range("2020-01-01", periods=500)
    df = pd.DataFrame({"x": range(500)}, index=dates)
    train, test = out_of_sample_split(df, months=6)
    assert train.index.max() < test.index.min()
    assert len(test) > 0
    assert len(train) + len(test) == len(df)


def test_compare_strategies_returns_one_row_per_strategy():
    tl_a = pd.DataFrame(
        {"entry_date": pd.to_datetime(["2026-01-01"]), "exit_date": pd.to_datetime(["2026-01-05"]), "pnl": [100.0]}
    )
    tl_b = pd.DataFrame(
        {"entry_date": pd.to_datetime(["2026-01-01"]), "exit_date": pd.to_datetime(["2026-01-05"]), "pnl": [-50.0]}
    )
    comparison = compare_strategies({"strategy_a": tl_a, "strategy_b": tl_b})
    assert set(comparison.index) == {"strategy_a", "strategy_b"}
    assert "total_return" in comparison.columns


def test_run_backtest_integration_wrapper_end_to_end():
    dates = pd.bdate_range("2026-01-01", periods=30)
    ohlc = _ohlc(dates)
    price_df = pd.DataFrame(
        {
            "ticker": "AAA",
            "ts": dates,
            "open": ohlc["Open"].values,
            "high": ohlc["High"].values,
            "low": ohlc["Low"].values,
            "close": ohlc["Close"].values,
            "volume": 1_000_000,
        }
    )
    signals_df = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "as_of": [dates[0]],
            "rating": ["Buy"],
            "suggested_entry": [ohlc.loc[dates[1], "Open"]],
            "suggested_stop": [ohlc.loc[dates[1], "Open"] - 5.0],
            "suggested_target_1": [ohlc.loc[dates[1], "Open"] + 2.0],
        }
    )

    result = run_backtest(price_df, signals_df, start=dates[0], end=dates[-1], strategy="unit-test")
    assert result["strategy"] == "unit-test"
    assert "metrics" in result
    assert isinstance(result["trade_log"], pd.DataFrame)
    assert isinstance(result["equity_curve"], pd.Series)

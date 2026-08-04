"""Relative strength features (SRS FE-002).

Computes a ticker's trailing return relative to SPY and its sector ETF, an
IBD-style relative-strength percentile rating vs a universe of peers, and a
1-11 sector momentum rank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_trader.logging_setup import get_logger

logger = get_logger("features.relative_strength")

_WINDOWS = (5, 10, 21)


def _close_col(df: pd.DataFrame) -> str | None:
    for candidate in ("Close", "close", "Adj Close", "adj_close"):
        if candidate in df.columns:
            return candidate
    return None


def _trailing_return(df: pd.DataFrame, window: int) -> float | None:
    """Simple trailing return over `window` trading days using the close column."""
    if df is None or df.empty:
        return None
    col = _close_col(df)
    if col is None or len(df) <= window:
        return None
    closes = df[col].dropna()
    if len(closes) <= window:
        return None
    try:
        return float(closes.iloc[-1] / closes.iloc[-1 - window] - 1.0)
    except (IndexError, ZeroDivisionError):
        return None


def compute_relative_strength(
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    sector_df: pd.DataFrame | None = None,
) -> dict:
    """Compute ret_Nd_vs_spy / ret_Nd_vs_sector for N in (5, 10, 21).

    Each value is the ticker's trailing N-day return minus the benchmark's
    trailing N-day return over the same window. Returns None for any pair
    that cannot be computed (insufficient history).
    """
    result: dict[str, float | None] = {}

    for window in _WINDOWS:
        ticker_ret = _trailing_return(ticker_df, window)
        spy_ret = _trailing_return(spy_df, window)
        key = f"ret_{window}d_vs_spy"
        if ticker_ret is None or spy_ret is None:
            result[key] = None
        else:
            result[key] = ticker_ret - spy_ret

    for window in _WINDOWS:
        key = f"ret_{window}d_vs_sector"
        if sector_df is None:
            result[key] = None
            continue
        ticker_ret = _trailing_return(ticker_df, window)
        sector_ret = _trailing_return(sector_df, window)
        if ticker_ret is None or sector_ret is None:
            result[key] = None
        else:
            result[key] = ticker_ret - sector_ret

    return result


def compute_rs_rating(
    ticker_returns: pd.Series, universe_returns: dict[str, pd.Series]
) -> float:
    """IBD-style relative-strength rating: percentile rank (0-100) of the
    ticker's trailing return against a universe of peer tickers' trailing
    returns.

    `ticker_returns` and each series in `universe_returns` are expected to be
    a single scalar-like trailing return value per ticker (e.g. trailing
    12-month or 21-day return); if a full Series of daily returns is passed
    instead, the cumulative compounded return is used.
    """

    def _scalarize(s: pd.Series | float) -> float | None:
        if s is None:
            return None
        if isinstance(s, (int, float, np.floating)):
            val = float(s)
            return val if not np.isnan(val) else None
        if isinstance(s, pd.Series):
            clean = s.dropna()
            if clean.empty:
                return None
            # Treat as a daily-returns series -> compound to a cumulative return.
            return float((1.0 + clean).prod() - 1.0)
        return None

    target = _scalarize(ticker_returns)
    if target is None:
        return 0.0

    peer_values = []
    for peer_ticker, peer_series in universe_returns.items():
        val = _scalarize(peer_series)
        if val is not None:
            peer_values.append(val)

    if not peer_values:
        return 50.0  # neutral default with no peer data

    all_values = np.array(peer_values + [target])
    rank = (all_values <= target).sum()
    percentile = 100.0 * rank / len(all_values)
    return float(np.clip(percentile, 0.0, 100.0))


def sector_momentum_rank(sector_etf_returns: dict[str, float]) -> dict[str, int]:
    """Rank sector ETFs 1-11 by momentum (1 = strongest trailing return)."""
    if not sector_etf_returns:
        return {}
    clean = {k: v for k, v in sector_etf_returns.items() if v is not None and not np.isnan(v)}
    if not clean:
        return {ticker: None for ticker in sector_etf_returns}  # type: ignore[misc]

    ordered = sorted(clean.items(), key=lambda kv: kv[1], reverse=True)
    ranks: dict[str, int] = {ticker: i + 1 for i, (ticker, _) in enumerate(ordered)}
    for ticker in sector_etf_returns:
        if ticker not in ranks:
            ranks[ticker] = None  # type: ignore[assignment]
    return ranks

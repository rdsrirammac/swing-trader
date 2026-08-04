"""Volatility features (SRS FE-003).

Computes 20-day annualized realized volatility, its trailing-252-day
percentile, ATR-as-percent-of-price, and the historical-vol-vs-implied-vol
spread (when an IV estimate is supplied).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swing_trader.logging_setup import get_logger

logger = get_logger("features.volatility")

_TRADING_DAYS_PER_YEAR = 252


def _close_col(df: pd.DataFrame) -> str | None:
    for candidate in ("Close", "close", "Adj Close", "adj_close"):
        if candidate in df.columns:
            return candidate
    return None


def _log_returns(closes: pd.Series) -> pd.Series:
    return np.log(closes / closes.shift(1))


def _annualized_realized_vol(closes: pd.Series, window: int) -> pd.Series:
    """Rolling `window`-day annualized stdev of log returns."""
    log_ret = _log_returns(closes)
    return log_ret.rolling(window=window, min_periods=max(2, window // 2)).std() * np.sqrt(
        _TRADING_DAYS_PER_YEAR
    )


def compute_volatility_features(df: pd.DataFrame, iv: float | None = None) -> dict:
    """Compute realized_vol_20d, realized_vol_pctile, atr_pct, hv_iv_spread.

    Parameters
    ----------
    df: OHLC(V) dataframe for a single ticker, most recent row last.
    iv: optional current implied volatility estimate (e.g. from an ATM
        option or IV rank source). If None, hv_iv_spread is returned as None
        because yfinance does not expose a clean IV series (known gap, see
        SRS_Refinement Section 3).
    """
    result = {
        "realized_vol_20d": None,
        "realized_vol_pctile": None,
        "atr_pct": None,
        "hv_iv_spread": None,
    }

    if df is None or df.empty:
        return result

    close_col = _close_col(df)
    if close_col is None:
        logger.warning("compute_volatility_features: no close column found")
        return result

    closes = df[close_col].dropna()
    if len(closes) < 5:
        return result

    vol_series = _annualized_realized_vol(closes, window=20)
    latest_vol = vol_series.iloc[-1] if not vol_series.empty else np.nan
    if pd.notna(latest_vol):
        result["realized_vol_20d"] = float(latest_vol)

        trailing = vol_series.dropna().tail(_TRADING_DAYS_PER_YEAR)
        if len(trailing) >= 5:
            pctile = 100.0 * (trailing <= latest_vol).sum() / len(trailing)
            result["realized_vol_pctile"] = float(np.clip(pctile, 0.0, 100.0))

    # ATR percent of close, computed inline (avoids importing technical.py's
    # pandas_ta dependency here; a simple Wilder-style ATR is sufficient).
    try:
        high_col = next((c for c in df.columns if str(c).lower() == "high"), None)
        low_col = next((c for c in df.columns if str(c).lower() == "low"), None)
        if high_col and low_col:
            high = df[high_col]
            low = df[low_col]
            prev_close = df[close_col].shift(1)
            tr = pd.concat(
                [
                    (high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_14 = tr.rolling(window=14, min_periods=7).mean().iloc[-1]
            last_close = closes.iloc[-1]
            if pd.notna(atr_14) and last_close:
                result["atr_pct"] = float(atr_14 / last_close)
    except Exception as e:
        logger.warning("atr_pct computation failed: %s", e)

    if iv is not None and result["realized_vol_20d"] is not None:
        result["hv_iv_spread"] = float(result["realized_vol_20d"] - iv)
    # else: leave as None -- IV not available from yfinance without an
    # options-chain-derived estimate; caller may supply one explicitly.

    return result

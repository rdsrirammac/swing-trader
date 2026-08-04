"""Macro/market-wide features (SRS FE-006).

Computes VIX level & percentile, whether SPY is above its 20-day EMA,
percent of the 11 SPDR sector ETFs trading above their own 50-day SMA
(breadth), and the 10y-2y treasury yield curve spread.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from swing_trader.logging_setup import get_logger

logger = get_logger("features.macro")

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLB", "XLRE", "XLU", "XLC"]


def _close_col(df: pd.DataFrame) -> str | None:
    for candidate in ("Close", "close", "Adj Close", "adj_close"):
        if candidate in df.columns:
            return candidate
    return None


def _above_ema(df: pd.DataFrame, span: int) -> bool | None:
    col = _close_col(df) if df is not None else None
    if df is None or df.empty or col is None:
        return None
    closes = df[col].dropna()
    if len(closes) < span:
        return None
    ema = closes.ewm(span=span, adjust=False).mean()
    try:
        return bool(closes.iloc[-1] > ema.iloc[-1])
    except IndexError:
        return None


def _above_sma(df: pd.DataFrame, window: int) -> bool | None:
    col = _close_col(df) if df is not None else None
    if df is None or df.empty or col is None:
        return None
    closes = df[col].dropna()
    if len(closes) < window:
        return None
    sma = closes.rolling(window=window, min_periods=window).mean()
    try:
        last_sma = sma.iloc[-1]
        if pd.isna(last_sma):
            return None
        return bool(closes.iloc[-1] > last_sma)
    except IndexError:
        return None


def compute_macro_features(
    as_of: dt.date,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    sector_etf_dfs: dict[str, pd.DataFrame],
    treasury_10y: float | None,
    treasury_2y_proxy: float | None,
) -> dict:
    """Returns vix_level, vix_percentile, spy_above_ema20, sector_breadth_pct,
    yield_curve_10y_2y.

    NOTE on yield_curve_10y_2y: yfinance does not cleanly expose a 2-year
    treasury yield series. `treasury_2y_proxy` lets the caller supply an
    external value (e.g. derived from ^IRX as a short-rate proxy, or a
    FRED-sourced DGS2 series wired up later) — this is a known data gap per
    SRS_Refinement Section 3. If either input is missing we return None
    rather than fabricating a spread.
    """
    result: dict = {
        "vix_level": None,
        "vix_percentile": None,
        "spy_above_ema20": None,
        "sector_breadth_pct": None,
        "yield_curve_10y_2y": None,
    }

    vix_close_col = _close_col(vix_df) if vix_df is not None else None
    if vix_df is not None and not vix_df.empty and vix_close_col is not None:
        vix_closes = vix_df[vix_close_col].dropna()
        if not vix_closes.empty:
            latest_vix = float(vix_closes.iloc[-1])
            result["vix_level"] = latest_vix
            trailing = vix_closes.tail(252)
            if len(trailing) >= 5:
                pctile = 100.0 * (trailing <= latest_vix).sum() / len(trailing)
                result["vix_percentile"] = float(np.clip(pctile, 0.0, 100.0))

    result["spy_above_ema20"] = _above_ema(spy_df, span=20)

    if sector_etf_dfs:
        above_count = 0
        total_evaluable = 0
        for etf in SECTOR_ETFS:
            etf_df = sector_etf_dfs.get(etf)
            above = _above_sma(etf_df, window=50)
            if above is not None:
                total_evaluable += 1
                if above:
                    above_count += 1
        if total_evaluable > 0:
            result["sector_breadth_pct"] = float(100.0 * above_count / total_evaluable)

    if treasury_10y is not None and treasury_2y_proxy is not None:
        try:
            result["yield_curve_10y_2y"] = float(treasury_10y - treasury_2y_proxy)
        except (TypeError, ValueError):
            result["yield_curve_10y_2y"] = None
    else:
        logger.debug(
            "yield_curve_10y_2y skipped: treasury_10y=%s treasury_2y_proxy=%s "
            "(2y yield not cleanly available from yfinance; see SRS_Refinement Section 3)",
            treasury_10y,
            treasury_2y_proxy,
        )

    return result

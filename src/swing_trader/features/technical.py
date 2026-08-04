"""Technical indicator computation (SRS FE-001).

Computes RSI(2), RSI(14), MACD(12,26,9), ATR(14), Bollinger Bands(20,2) +
bandwidth, EMA(20), SMA(50), ADX(14), ROC(5,10,21), OBV, and a 20-day volume
ratio from an OHLCV dataframe, using the `pandas_ta` technical-analysis
library. Output columns are named to match `swing_trader.db.models.StockFeature`
field names exactly so callers can dict()-assign directly onto the ORM model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - pandas_ta may be unavailable in some envs
    ta = None

from swing_trader.logging_setup import get_logger

logger = get_logger("features.technical")

_TECHNICAL_COLUMNS = [
    "rsi_2",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "bb_upper",
    "bb_lower",
    "bb_bandwidth",
    "ema_20",
    "sma_50",
    "adx_14",
    "roc_5",
    "roc_10",
    "roc_21",
    "obv",
    "volume_ratio_20d",
]


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to the lowercase form pandas_ta expects."""
    rename_map = {}
    for col in df.columns:
        lower = str(col).lower()
        if lower in ("open", "high", "low", "close", "volume"):
            rename_map[col] = lower
    out = df.rename(columns=rename_map)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"OHLCV dataframe missing required columns: {missing}")
    return out


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the FE-001 technical indicator set for a single ticker's OHLCV history.

    Parameters
    ----------
    df: pd.DataFrame
        Indexed by date (or any monotonic index), with Open/High/Low/Close/Volume
        columns (case-insensitive).

    Returns
    -------
    pd.DataFrame with the same index as `df` and columns matching
    `StockFeature`'s technical fields (rsi_2, rsi_14, macd, macd_signal,
    macd_hist, atr_14, bb_upper, bb_lower, bb_bandwidth, ema_20, sma_50,
    adx_14, roc_5, roc_10, roc_21, obv, volume_ratio_20d).

    Early rows before an indicator has warmed up will be NaN, which is
    expected and handled downstream (feature_completeness accounts for it).
    """
    out = pd.DataFrame(index=df.index, columns=_TECHNICAL_COLUMNS, dtype="float64")

    if df is None or df.empty:
        logger.warning("compute_technical_indicators called with empty dataframe")
        return out

    work = _normalize_ohlcv(df)

    if ta is None:
        logger.warning("pandas_ta not installed; returning all-NaN technical indicators")
        return out

    try:
        rsi_2 = ta.rsi(work["close"], length=2)
        if rsi_2 is not None:
            out["rsi_2"] = rsi_2
    except Exception as e:
        logger.warning("rsi_2 computation failed: %s", e)

    try:
        rsi_14 = ta.rsi(work["close"], length=14)
        if rsi_14 is not None:
            out["rsi_14"] = rsi_14
    except Exception as e:
        logger.warning("rsi_14 computation failed: %s", e)

    try:
        macd_df = ta.macd(work["close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            # pandas_ta names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            macd_col = next((c for c in macd_df.columns if c.startswith("MACD_")), None)
            hist_col = next((c for c in macd_df.columns if c.startswith("MACDh_")), None)
            signal_col = next((c for c in macd_df.columns if c.startswith("MACDs_")), None)
            if macd_col:
                out["macd"] = macd_df[macd_col]
            if signal_col:
                out["macd_signal"] = macd_df[signal_col]
            if hist_col:
                out["macd_hist"] = macd_df[hist_col]
    except Exception as e:
        logger.warning("macd computation failed: %s", e)

    try:
        atr_14 = ta.atr(work["high"], work["low"], work["close"], length=14)
        if atr_14 is not None:
            out["atr_14"] = atr_14
    except Exception as e:
        logger.warning("atr_14 computation failed: %s", e)

    try:
        bb_df = ta.bbands(work["close"], length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            upper_col = next((c for c in bb_df.columns if c.startswith("BBU_")), None)
            lower_col = next((c for c in bb_df.columns if c.startswith("BBL_")), None)
            bw_col = next((c for c in bb_df.columns if c.startswith("BBB_")), None)
            if upper_col:
                out["bb_upper"] = bb_df[upper_col]
            if lower_col:
                out["bb_lower"] = bb_df[lower_col]
            if bw_col:
                out["bb_bandwidth"] = bb_df[bw_col]
            elif upper_col and lower_col:
                mid_col = next((c for c in bb_df.columns if c.startswith("BBM_")), None)
                if mid_col:
                    out["bb_bandwidth"] = (
                        (bb_df[upper_col] - bb_df[lower_col]) / bb_df[mid_col]
                    ) * 100
    except Exception as e:
        logger.warning("bbands computation failed: %s", e)

    try:
        ema_20 = ta.ema(work["close"], length=20)
        if ema_20 is not None:
            out["ema_20"] = ema_20
    except Exception as e:
        logger.warning("ema_20 computation failed: %s", e)

    try:
        sma_50 = ta.sma(work["close"], length=50)
        if sma_50 is not None:
            out["sma_50"] = sma_50
    except Exception as e:
        logger.warning("sma_50 computation failed: %s", e)

    try:
        adx_df = ta.adx(work["high"], work["low"], work["close"], length=14)
        if adx_df is not None and not adx_df.empty:
            adx_col = next((c for c in adx_df.columns if c.startswith("ADX_")), None)
            if adx_col:
                out["adx_14"] = adx_df[adx_col]
    except Exception as e:
        logger.warning("adx_14 computation failed: %s", e)

    try:
        out["roc_5"] = ta.roc(work["close"], length=5)
        out["roc_10"] = ta.roc(work["close"], length=10)
        out["roc_21"] = ta.roc(work["close"], length=21)
    except Exception as e:
        logger.warning("roc computation failed: %s", e)

    try:
        obv = ta.obv(work["close"], work["volume"])
        if obv is not None:
            out["obv"] = obv
    except Exception as e:
        logger.warning("obv computation failed: %s", e)

    try:
        vol_sma_20 = work["volume"].rolling(window=20, min_periods=1).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = work["volume"] / vol_sma_20.replace(0, np.nan)
        out["volume_ratio_20d"] = ratio
    except Exception as e:
        logger.warning("volume_ratio_20d computation failed: %s", e)

    return out

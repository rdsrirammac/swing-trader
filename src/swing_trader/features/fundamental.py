"""Fundamental features (SRS FE-005).

Computes PE percentile rank vs sector peers and vs the ticker's own
historical PE, a best-effort short-interest-as-percent-of-float, and an
earnings surprise streak counter, all from yfinance-shaped inputs.
"""
from __future__ import annotations

import numpy as np

from swing_trader.logging_setup import get_logger

logger = get_logger("features.fundamental")


def _percentile_rank(value: float | None, population: list[float] | None) -> float | None:
    if value is None or population is None:
        return None
    clean = [v for v in population if v is not None and not np.isnan(v)]
    if not clean:
        return None
    arr = np.array(clean + [value])
    rank = (arr <= value).sum()
    return float(np.clip(100.0 * rank / len(arr), 0.0, 100.0))


def compute_fundamental_features(
    info: dict,
    sector_pe_values: list[float] | None,
    historical_pe: list[float] | None,
    earnings_history: list[dict] | None,
) -> dict:
    """Returns pe_percentile_sector, pe_percentile_history,
    short_interest_pct_float, earnings_surprise_streak.

    `info` is the dict shape returned by `YFinanceClient.get_info`.
    `earnings_history` is a list of {"actual": float, "estimate": float}
    dicts ordered oldest-to-newest (most recent last).
    """
    result: dict = {
        "pe_percentile_sector": None,
        "pe_percentile_history": None,
        "short_interest_pct_float": None,
        "earnings_surprise_streak": None,
    }

    info = info or {}
    trailing_pe = info.get("trailingPE")
    if trailing_pe is not None:
        try:
            trailing_pe = float(trailing_pe)
        except (TypeError, ValueError):
            trailing_pe = None

    if trailing_pe is not None:
        result["pe_percentile_sector"] = _percentile_rank(trailing_pe, sector_pe_values)
        result["pe_percentile_history"] = _percentile_rank(trailing_pe, historical_pe)

    # Short interest as % of float: yfinance's `info` typically only exposes
    # `shortRatio` (days-to-cover), not pct-of-float directly. We approximate
    # using sharesShort / floatShares when both are present (best-effort,
    # since yfinance's schema for this varies by ticker/version).
    try:
        shares_short = info.get("sharesShort")
        float_shares = info.get("floatShares")
        if shares_short is not None and float_shares:
            result["short_interest_pct_float"] = float(shares_short) / float(float_shares)
        else:
            result["short_interest_pct_float"] = None  # genuinely unavailable
    except (TypeError, ValueError, ZeroDivisionError) as e:
        logger.warning("short_interest_pct_float computation failed: %s", e)
        result["short_interest_pct_float"] = None

    # Earnings surprise streak: count consecutive beats (actual > estimate)
    # or misses (actual < estimate) working backwards from the most recent
    # entry. Positive = beat streak, negative = miss streak, 0 = mixed/none.
    try:
        if earnings_history:
            streak = 0
            direction = 0  # +1 beat, -1 miss
            for row in reversed(earnings_history):
                actual = row.get("actual")
                estimate = row.get("estimate")
                if actual is None or estimate is None:
                    break
                if actual > estimate:
                    this_dir = 1
                elif actual < estimate:
                    this_dir = -1
                else:
                    this_dir = 0

                if direction == 0 and this_dir != 0:
                    direction = this_dir
                    streak = 1
                elif this_dir == direction and this_dir != 0:
                    streak += 1
                else:
                    break
            result["earnings_surprise_streak"] = streak * direction if direction != 0 else 0
    except Exception as e:
        logger.warning("earnings_surprise_streak computation failed: %s", e)
        result["earnings_surprise_streak"] = None

    return result

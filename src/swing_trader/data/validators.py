"""Data quality validation (DC-006).

Every batch of ingested price/feature data passes through here before it's
written to the database. Validation failures are logged and, if the failure
rate exceeds config thresholds, the batch is rejected outright.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd

from swing_trader.config import get_settings
from swing_trader.logging_setup import get_logger

logger = get_logger("data.validators")


@dataclass
class ValidationResult:
    is_valid: bool
    missing_pct: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_price_dataframe(df: pd.DataFrame, ticker: str) -> ValidationResult:
    """Validate an OHLCV dataframe per DC-006:
      - missing values (reject batch if > 5% missing)
      - out-of-range values (negative prices, zero volume)
      - timestamp consistency (no future dates, no duplicates)
      - cross-field validation (high >= low, close within [low, high])
    """
    settings = get_settings()
    max_missing_pct = settings.get("data_quality.max_missing_pct_batch", 0.05)

    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        return ValidationResult(is_valid=False, missing_pct=1.0, errors=["empty dataframe"])

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")
        return ValidationResult(is_valid=False, missing_pct=1.0, errors=errors)

    total_cells = len(df) * len(required_cols)
    missing_cells = df[list(required_cols)].isna().sum().sum()
    missing_pct = float(missing_cells) / total_cells if total_cells else 1.0

    if missing_pct > max_missing_pct:
        errors.append(f"{missing_pct:.1%} missing values exceeds {max_missing_pct:.1%} threshold")

    negative_prices = (df[["Open", "High", "Low", "Close"]] < 0).any().any()
    if negative_prices:
        errors.append("negative price values found")

    zero_volume_days = int((df["Volume"] == 0).sum())
    if zero_volume_days:
        warnings.append(f"{zero_volume_days} day(s) with zero volume")

    now = pd.Timestamp.utcnow()
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        tz_now = now.tz_localize(None) if idx.tz is None else now
        future_dates = (idx > tz_now).sum() if idx.tz is not None else (idx > pd.Timestamp(dt.datetime.utcnow())).sum()
        if future_dates:
            errors.append(f"{future_dates} future-dated row(s)")
        if idx.duplicated().any():
            errors.append("duplicate timestamps found")

    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl:
        errors.append(f"{bad_hl} row(s) with High < Low")

    bad_close = ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum()
    if bad_close:
        errors.append(f"{bad_close} row(s) with Close outside [Low, High]")

    is_valid = len(errors) == 0
    if not is_valid:
        logger.warning("Validation failed for %s: %s", ticker, "; ".join(errors))

    return ValidationResult(is_valid=is_valid, missing_pct=missing_pct, errors=errors, warnings=warnings)


def data_freshness_ok(latest_ts: dt.datetime, max_staleness_trading_days: int = 2) -> bool:
    """TB-003: last price must be within N trading days (approximated via
    calendar days * 1.5 to loosely account for weekends)."""
    if latest_ts is None:
        return False
    max_age = dt.timedelta(days=max_staleness_trading_days * 1.6)
    now = dt.datetime.utcnow()
    if latest_ts.tzinfo is not None:
        now = now.replace(tzinfo=latest_ts.tzinfo)
    return (now - latest_ts) <= max_age

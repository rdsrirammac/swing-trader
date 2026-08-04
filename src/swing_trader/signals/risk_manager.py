"""SRS 3.8 Risk Management (RM-001..RM-007).

All numeric thresholds are read from `config/settings.yaml` (`risk.*` /
`positions.*`), never hardcoded, so operators can retune risk posture
without a code change.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import Holding, Portfolio, PositionStatus, TickerUniverse
from swing_trader.logging_setup import get_logger

logger = get_logger("signals.risk_manager")

_ACTIVE_STATUSES = (PositionStatus.ACTIVE, PositionStatus.TRIMMED)


# ---------------------------------------------------------------------------
# RM-001: position sizing
# ---------------------------------------------------------------------------

def position_size(
    portfolio_value: float,
    entry: float,
    stop: float,
    risk_pct: float | None = None,
) -> float:
    """RM-001: fixed-fractional position sizing.

    shares = (portfolio_value * risk_pct) / (entry - stop)

    `risk_pct` defaults to `risk.max_risk_per_trade_pct` and is clamped into
    [`risk.min_risk_per_trade_pct`, `risk.max_risk_per_trade_pct_ceiling`].
    The resulting position's dollar value is additionally capped at
    `risk.max_single_position_pct` of `portfolio_value` (shares are clipped
    down if the fixed-fractional sizing would exceed that cap).

    Returns 0.0 for degenerate inputs (entry <= stop, portfolio_value <= 0).
    """
    settings = get_settings()
    default_risk = settings.get("risk.max_risk_per_trade_pct", 0.02)
    min_risk = settings.get("risk.min_risk_per_trade_pct", 0.01)
    max_risk_ceiling = settings.get("risk.max_risk_per_trade_pct_ceiling", 0.03)
    max_single_position_pct = settings.get("risk.max_single_position_pct", 0.10)

    effective_risk_pct = default_risk if risk_pct is None else risk_pct
    effective_risk_pct = max(min_risk, min(effective_risk_pct, max_risk_ceiling))

    risk_per_share = entry - stop
    if risk_per_share <= 0 or portfolio_value <= 0:
        return 0.0

    shares = (portfolio_value * effective_risk_pct) / risk_per_share

    max_position_value = portfolio_value * max_single_position_pct
    if entry > 0 and shares * entry > max_position_value:
        shares = max_position_value / entry

    return max(shares, 0.0)


# ---------------------------------------------------------------------------
# RM-002: portfolio heat
# ---------------------------------------------------------------------------

def portfolio_heat(session: Session, portfolio_id: int) -> float:
    """RM-002: portfolio heat = sum((entry_price - stop_loss) * shares) over
    active/trimmed Holdings, divided by an approximate total portfolio value
    (cash_balance + sum(shares * entry_price)).

    NOTE: uses entry_price as a stand-in for live mark-to-market value since
    this module has no quote feed wired in; swapping in real-time prices for
    the denominator is a backlog item and would make this more accurate
    during periods of large unrealized gains/losses.
    """
    portfolio = session.get(Portfolio, portfolio_id)
    if portfolio is None:
        raise ValueError(f"Portfolio {portfolio_id} not found")

    holdings = session.execute(
        select(Holding).where(Holding.portfolio_id == portfolio_id, Holding.status.in_(_ACTIVE_STATUSES))
    ).scalars().all()

    total_risk_dollars = sum(max(h.entry_price - h.stop_loss, 0.0) * h.shares for h in holdings)
    holdings_value = sum(h.entry_price * h.shares for h in holdings)
    total_portfolio_value = portfolio.cash_balance + holdings_value

    if total_portfolio_value <= 0:
        return 0.0
    return total_risk_dollars / total_portfolio_value


def heat_status(heat_pct: float) -> str:
    """RM-002 status band: green (<10%), yellow (10%..max_portfolio_heat_pct),
    red (> max_portfolio_heat_pct)."""
    max_heat = get_settings().get("risk.max_portfolio_heat_pct", 0.20)
    if heat_pct < 0.10:
        return "green"
    if heat_pct <= max_heat:
        return "yellow"
    return "red"


def blocks_new_buys(heat_pct: float) -> bool:
    """RM-002: True if heat exceeds risk.max_portfolio_heat_pct -- new Buy
    signals should be suppressed/blocked by the caller."""
    max_heat = get_settings().get("risk.max_portfolio_heat_pct", 0.20)
    return heat_pct > max_heat


# ---------------------------------------------------------------------------
# RM-003: sector concentration
# ---------------------------------------------------------------------------

def sector_concentration(session: Session, portfolio_id: int) -> dict[str, float]:
    """RM-003: {sector: pct_of_portfolio} across active/trimmed Holdings,
    joined to TickerUniverse.sector (tickers with unknown sector are bucketed
    under "Unknown")."""
    portfolio = session.get(Portfolio, portfolio_id)
    if portfolio is None:
        raise ValueError(f"Portfolio {portfolio_id} not found")

    holdings = session.execute(
        select(Holding).where(Holding.portfolio_id == portfolio_id, Holding.status.in_(_ACTIVE_STATUSES))
    ).scalars().all()
    if not holdings:
        return {}

    holdings_value = sum(h.entry_price * h.shares for h in holdings)
    total_portfolio_value = portfolio.cash_balance + holdings_value
    if total_portfolio_value <= 0:
        return {}

    sector_value: dict[str, float] = {}
    for h in holdings:
        ticker_row = session.execute(
            select(TickerUniverse).where(TickerUniverse.ticker == h.ticker)
        ).scalar_one_or_none()
        sector = ticker_row.sector if ticker_row and ticker_row.sector else "Unknown"
        sector_value[sector] = sector_value.get(sector, 0.0) + h.entry_price * h.shares

    return {sector: value / total_portfolio_value for sector, value in sector_value.items()}


def sector_limit_breached(concentration: dict[str, float]) -> list[str]:
    """RM-003: sectors whose concentration exceeds risk.max_sector_concentration_pct."""
    limit = get_settings().get("risk.max_sector_concentration_pct", 0.30)
    return [sector for sector, pct in concentration.items() if pct > limit]


# ---------------------------------------------------------------------------
# RM-004: correlation check
# ---------------------------------------------------------------------------

def correlation_check(
    session: Session,
    portfolio_id: int,
    candidate_ticker: str,
    price_lookup: dict[str, pd.Series],
) -> tuple[bool, dict[str, float]]:
    """RM-004: pairwise Pearson correlation of daily returns between
    `candidate_ticker` and each active/trimmed Holding.

    `price_lookup` maps ticker -> a (caller-supplied, typically ~60-day)
    close-price Series; this module intentionally does not fetch price data
    itself to avoid coupling to the data layer.

    Returns:
        (reject, correlations) where reject=True if any |correlation| exceeds
        risk.correlation_reject_threshold.
    """
    threshold = get_settings().get("risk.correlation_reject_threshold", 0.80)

    holdings = session.execute(
        select(Holding).where(Holding.portfolio_id == portfolio_id, Holding.status.in_(_ACTIVE_STATUSES))
    ).scalars().all()

    correlations: dict[str, float] = {}
    candidate_prices = price_lookup.get(candidate_ticker)
    if candidate_prices is None or candidate_prices.empty:
        logger.warning("No price series for candidate %s; skipping correlation check", candidate_ticker)
        return False, correlations

    candidate_returns = candidate_prices.pct_change().dropna()

    held_tickers = {h.ticker for h in holdings if h.ticker != candidate_ticker}
    for ticker in held_tickers:
        series = price_lookup.get(ticker)
        if series is None or series.empty:
            continue
        holding_returns = series.pct_change().dropna()
        aligned = pd.concat([candidate_returns, holding_returns], axis=1, join="inner")
        if len(aligned) < 5:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if corr is not None and not pd.isna(corr):
            correlations[ticker] = float(corr)

    reject = any(abs(c) > threshold for c in correlations.values())
    return reject, correlations


# ---------------------------------------------------------------------------
# RM-005: drawdown controls
# ---------------------------------------------------------------------------

def drawdown_controls(
    session: Session,
    portfolio_id: int,
    portfolio_value_history: pd.Series,
) -> dict:
    """RM-005: compute daily/weekly drawdown and a circuit-breaker size
    multiplier from a caller-supplied time series of daily portfolio value.

    `portfolio_value_history` must be indexed by date (ascending or
    descending; it is sorted internally) with the most recent value as the
    latest entry.

    Returns dict with keys: daily_dd_pct, weekly_dd_pct, alert_daily,
    alert_weekly, size_multiplier (1.0 normal, 0.5 if drawdown-from-peak
    exceeds risk.drawdown_reduce_50_trigger_pct, 0.25 if it exceeds
    risk.drawdown_reduce_75_trigger_pct).
    """
    settings = get_settings()
    daily_alert_pct = settings.get("risk.daily_drawdown_alert_pct", 0.05)
    weekly_alert_pct = settings.get("risk.weekly_drawdown_alert_pct", 0.10)
    reduce_50_trigger = settings.get("risk.drawdown_reduce_50_trigger_pct", 0.10)
    reduce_75_trigger = settings.get("risk.drawdown_reduce_75_trigger_pct", 0.15)

    if portfolio_value_history is None or portfolio_value_history.empty:
        return {
            "daily_dd_pct": 0.0,
            "weekly_dd_pct": 0.0,
            "alert_daily": False,
            "alert_weekly": False,
            "size_multiplier": 1.0,
        }

    history = portfolio_value_history.sort_index()
    current_value = float(history.iloc[-1])

    prev_value = float(history.iloc[-2]) if len(history) >= 2 else current_value
    daily_dd_pct = (prev_value - current_value) / prev_value if prev_value else 0.0

    lookback = min(5, len(history) - 1)
    week_ago_value = float(history.iloc[-1 - lookback]) if lookback > 0 else current_value
    weekly_dd_pct = (week_ago_value - current_value) / week_ago_value if week_ago_value else 0.0

    running_peak = float(history.cummax().iloc[-1])
    peak_dd_pct = (running_peak - current_value) / running_peak if running_peak else 0.0

    size_multiplier = 1.0
    if peak_dd_pct >= reduce_75_trigger:
        size_multiplier = 0.25
    elif peak_dd_pct >= reduce_50_trigger:
        size_multiplier = 0.5

    return {
        "daily_dd_pct": daily_dd_pct,
        "weekly_dd_pct": weekly_dd_pct,
        "alert_daily": daily_dd_pct >= daily_alert_pct,
        "alert_weekly": weekly_dd_pct >= weekly_alert_pct,
        "size_multiplier": size_multiplier,
    }


# ---------------------------------------------------------------------------
# RM-006: volatility-adjusted sizing
# ---------------------------------------------------------------------------

def volatility_adjusted_sizing(
    base_shares: float,
    regime: str | None,
    vix: float | None = None,
) -> dict:
    """RM-006: shrink size and widen stop/target ATR multiples in a
    high-volatility regime (RegimeType.HIGH_VOLATILITY, the SRS's literal
    "high_vol", or VIX > risk.high_vol_vix_threshold).

    Returns dict with keys: shares, stop_atr_multiple, target_atr_multiple.
    """
    settings = get_settings()
    vix_threshold = settings.get("risk.high_vol_vix_threshold", 25)
    size_multiplier = settings.get("risk.high_vol_size_multiplier", 0.5)
    high_vol_stop_mult = settings.get("risk.high_vol_stop_atr_multiple", 2.5)
    high_vol_target_mult = settings.get("risk.high_vol_target_atr_multiple", 2.0)
    default_stop_mult = settings.get("positions.default_stop_atr_multiple", 2.0)
    default_target_mult = settings.get("positions.target_1_atr_multiple", 2.0)

    regime_norm = (regime or "").lower()
    is_high_vol = regime_norm in {"high_vol", "high_volatility"} or (
        vix is not None and vix > vix_threshold
    )

    if is_high_vol:
        return {
            "shares": base_shares * size_multiplier,
            "stop_atr_multiple": high_vol_stop_mult,
            "target_atr_multiple": high_vol_target_mult,
        }
    return {
        "shares": base_shares,
        "stop_atr_multiple": default_stop_mult,
        "target_atr_multiple": default_target_mult,
    }


# ---------------------------------------------------------------------------
# RM-007: trailing stops
# ---------------------------------------------------------------------------

def update_trailing_stops(
    session: Session,
    portfolio_id: int,
    current_prices: dict[str, float],
    atr_by_ticker: dict[str, float],
) -> list[Holding]:
    """RM-007: activate/ratchet trailing stops for active/trimmed Holdings.

    For each holding up more than `risk.trailing_stop_activation_pct` from
    entry (using today's close in `current_prices` as the reference), sets
    `trailing_stop_active=True` and ratchets `trailing_stop` up to
    `max(existing trailing_stop, close - risk.trailing_stop_atr_multiple * atr)`
    -- it is NEVER lowered.

    This module does not track intraday highs; per the SRS Section 6.3 EOD
    schedule, `update_trailing_stops` is expected to be called once daily
    with the day's close, and the monotonic ratchet across repeated daily
    calls is what approximates trailing off the running high.

    Returns the list of updated Holding objects; the caller is responsible
    for committing the session (this function only flushes for consistency
    of returned objects, it does not commit).
    """
    settings = get_settings()
    activation_pct = settings.get("risk.trailing_stop_activation_pct", 0.03)
    trailing_atr_mult = settings.get("risk.trailing_stop_atr_multiple", 1.5)

    holdings = session.execute(
        select(Holding).where(Holding.portfolio_id == portfolio_id, Holding.status.in_(_ACTIVE_STATUSES))
    ).scalars().all()

    updated: list[Holding] = []
    for h in holdings:
        price = current_prices.get(h.ticker)
        atr = atr_by_ticker.get(h.ticker)
        if price is None or atr is None or h.entry_price <= 0:
            continue

        gain_pct = (price - h.entry_price) / h.entry_price
        if gain_pct < activation_pct:
            continue

        h.trailing_stop_active = True
        candidate_stop = price - trailing_atr_mult * atr
        if h.trailing_stop is None or candidate_stop > h.trailing_stop:
            h.trailing_stop = candidate_stop
        updated.append(h)

    session.flush()
    return updated

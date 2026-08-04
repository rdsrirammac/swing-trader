"""Market Regime Detection (SRS Section 3.5, MR-001..004).

Classifies the current market regime from SPY/VIX/breadth metrics,
detects day-over-day regime transitions, exposes the per-regime
position-sizing/hold-time adjustments used by risk management, and tracks
realized trading performance broken out by regime.
"""
from __future__ import annotations

import datetime as dt
import statistics

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import RegimeHistory, RegimePerformance, RegimeType, StockFeature, Trade
from swing_trader.logging_setup import get_logger

logger = get_logger("models.regime_detector")


# ---------------------------------------------------------------------------
# MR-001 classification
# ---------------------------------------------------------------------------

def classify_regime(
    spy_adx: float,
    vix: float,
    sector_breadth_pct: float | None,
    bb_width_pct: float | None,
    atr_expansion_pct: float | None,
    pct_sp500_reporting_next_2wk: float | None,
) -> RegimeType:
    """Classify today's market regime per MR-001.

    Rules (thresholds read from config/regimes.yaml via `get_settings().regimes`,
    falling back to the SRS defaults below if a key is missing):
      - Strong Trend:   ADX > 25, VIX < 20, breadth > 60%
      - Weak Trend:     ADX in [15, 25], VIX in [20, 25]
      - Range-Bound:    ADX < 20, VIX < 22, BB width < 10%
      - High Volatility: VIX > 25 OR ATR expansion > 150%
      - Earnings Season: > 20% of S&P 500 reporting in the next 2 weeks

    Precedence when multiple conditions match (SRS does not specify this;
    documented choice): high_volatility > earnings_season > strong_trend >
    weak_trend > range_bound. High volatility is checked first because it's
    the regime with the largest risk-management impact (position-size cuts);
    earnings season next because it's a scheduled, ticker-agnostic risk
    window independent of trend/vol state.
    """
    settings = get_settings()
    regimes_cfg = (settings.regimes or {}).get("regimes", {})

    st = regimes_cfg.get("strong_trend", {})
    wt = regimes_cfg.get("weak_trend", {})
    rb = regimes_cfg.get("range_bound", {})
    hv = regimes_cfg.get("high_volatility", {})
    es = regimes_cfg.get("earnings_season", {})

    st_adx_min = st.get("spy_adx_min", 25)
    st_vix_max = st.get("vix_max", 20)
    st_breadth_min = st.get("sector_breadth_min_pct", 60)

    wt_adx_min = wt.get("spy_adx_min", 15)
    wt_adx_max = wt.get("spy_adx_max", 25)
    wt_vix_min = wt.get("vix_min", 20)
    wt_vix_max = wt.get("vix_max", 25)

    rb_adx_max = rb.get("spy_adx_max", 20)
    rb_vix_max = rb.get("vix_max", 22)
    rb_bb_width_max = rb.get("bb_width_max_pct", 10)

    hv_vix_min = hv.get("vix_min", 25)
    hv_atr_expansion_min = hv.get("atr_expansion_pct_min", 150)

    es_pct_min = es.get("sp500_reporting_pct_min", 20)

    is_high_vol = (vix is not None and vix > hv_vix_min) or (
        atr_expansion_pct is not None and atr_expansion_pct > hv_atr_expansion_min
    )
    is_earnings_season = (
        pct_sp500_reporting_next_2wk is not None and pct_sp500_reporting_next_2wk > es_pct_min
    )
    is_strong_trend = (
        spy_adx is not None
        and vix is not None
        and spy_adx > st_adx_min
        and vix < st_vix_max
        and sector_breadth_pct is not None
        and sector_breadth_pct > st_breadth_min
    )
    is_weak_trend = (
        spy_adx is not None
        and vix is not None
        and wt_adx_min <= spy_adx <= wt_adx_max
        and wt_vix_min <= vix <= wt_vix_max
    )
    is_range_bound = (
        spy_adx is not None
        and vix is not None
        and spy_adx < rb_adx_max
        and vix < rb_vix_max
        and bb_width_pct is not None
        and bb_width_pct < rb_bb_width_max
    )

    if is_high_vol:
        return RegimeType.HIGH_VOLATILITY
    if is_earnings_season:
        return RegimeType.EARNINGS_SEASON
    if is_strong_trend:
        return RegimeType.STRONG_TREND
    if is_weak_trend:
        return RegimeType.WEAK_TREND
    if is_range_bound:
        return RegimeType.RANGE_BOUND

    # No rule matched (e.g. ADX/VIX fall between bands). Default to
    # weak_trend as the most neutral non-extreme classification since the
    # SRS doesn't define an explicit catch-all.
    logger.debug(
        "classify_regime: no rule matched (adx=%s vix=%s breadth=%s bb_width=%s); defaulting to weak_trend",
        spy_adx, vix, sector_breadth_pct, bb_width_pct,
    )
    return RegimeType.WEAK_TREND


# ---------------------------------------------------------------------------
# MR-002 transition detection
# ---------------------------------------------------------------------------

def detect_transition(
    session: Session,
    as_of: dt.date,
    advance_decline_up_volume_pct: float | None = None,
) -> tuple[bool, str | None]:
    """Detect a market-regime transition around `as_of` (MR-002).

    Checks, each independent and additive to the transition reason string:
      1. VIX spike: |today's VIX - trailing 20-day mean| > N std devs
         (N from regimes.yaml `transitions.vix_spike_std_devs`).
      2. SPY EMA20/SMA50 crossover, sourced from the `StockFeature` rows for
         ticker "SPY" (the two most recent trading days as of `as_of`).
      3. Sector rotation (top-3 sector ranks changed within N days): SKIPPED
         -- the current schema has no per-sector-ETF rank history table
         (`RegimeHistory` only stores an aggregate `sector_breadth_pct`, not
         per-sector momentum ranks), so this check cannot be computed without
         a new table. Documented gap.
      4. Breadth thrust (>=90% up volume vs down volume): requires
         advance/decline volume data the caller must supply via
         `advance_decline_up_volume_pct`; skipped (None) if not provided.

    Returns (transitioned: bool, reason: str | None).
    """
    settings = get_settings()
    transitions_cfg = (settings.regimes or {}).get("transitions", {})
    vix_spike_std_devs = transitions_cfg.get("vix_spike_std_devs", 2)
    breadth_thrust_volume_pct = transitions_cfg.get("breadth_thrust_volume_pct", 90)

    reasons: list[str] = []

    # 1. VIX spike vs trailing 20-day VIX.
    try:
        today_row = session.execute(
            select(RegimeHistory).where(RegimeHistory.ts == as_of)
        ).scalar_one_or_none()
        trailing_rows = (
            session.execute(
                select(RegimeHistory)
                .where(RegimeHistory.ts < as_of)
                .order_by(RegimeHistory.ts.desc())
                .limit(20)
            )
            .scalars()
            .all()
        )
        vix_values = [r.vix for r in trailing_rows if r.vix is not None]
        if today_row is not None and today_row.vix is not None and len(vix_values) >= 5:
            mean_vix = statistics.mean(vix_values)
            std_vix = statistics.pstdev(vix_values)
            if std_vix > 0 and abs(today_row.vix - mean_vix) > vix_spike_std_devs * std_vix:
                reasons.append(
                    f"VIX spike: {today_row.vix:.1f} vs trailing mean {mean_vix:.1f} "
                    f"(+/-{std_vix:.1f}, >{vix_spike_std_devs} std devs)"
                )
    except Exception as e:
        logger.warning("detect_transition: VIX spike check failed: %s", e)

    # 2. SPY EMA20/SMA50 crossover.
    try:
        spy_rows = (
            session.execute(
                select(StockFeature)
                .where(StockFeature.ticker == "SPY", StockFeature.ts <= as_of)
                .order_by(StockFeature.ts.desc())
                .limit(2)
            )
            .scalars()
            .all()
        )
        if len(spy_rows) == 2:
            newest, prior = spy_rows[0], spy_rows[1]
            if None not in (newest.ema_20, newest.sma_50, prior.ema_20, prior.sma_50):
                newest_diff = newest.ema_20 - newest.sma_50
                prior_diff = prior.ema_20 - prior.sma_50
                if (newest_diff > 0) != (prior_diff > 0):
                    direction = "bullish" if newest_diff > 0 else "bearish"
                    reasons.append(f"SPY EMA20/SMA50 {direction} crossover")
    except Exception as e:
        logger.warning("detect_transition: SPY EMA/SMA crossover check failed: %s", e)

    # 3. sector rotation -- see docstring; not computable with current schema.

    # 4. breadth thrust
    if (
        advance_decline_up_volume_pct is not None
        and advance_decline_up_volume_pct >= breadth_thrust_volume_pct
    ):
        reasons.append(f"breadth thrust: {advance_decline_up_volume_pct:.0f}% up volume")

    if reasons:
        return True, "; ".join(reasons)
    return False, None


# ---------------------------------------------------------------------------
# MR-003 regime-driven adjustments
# ---------------------------------------------------------------------------

def regime_adjustments(regime: RegimeType) -> dict:
    """Position-size / stop / target / hold-time multipliers for `regime`
    (MR-003), read from `regimes.yaml`'s `regime_adjustments` block. Regimes
    without an explicit override (or missing individual keys) fall back to
    1.0 multipliers / the global default ATR multiples from settings.yaml.
    """
    settings = get_settings()
    adjustments_cfg = (settings.regimes or {}).get("regime_adjustments", {})
    regime_cfg = adjustments_cfg.get(regime.value, {})

    default_stop_atr = settings.get("positions.default_stop_atr_multiple", 2.0)
    default_target_atr = settings.get("positions.target_1_atr_multiple", 2.0)

    return {
        "position_size_multiplier": regime_cfg.get("position_size_multiplier", 1.0),
        "stop_atr_multiple": regime_cfg.get("stop_atr_multiple", default_stop_atr),
        "target_atr_multiple": regime_cfg.get("target_atr_multiple", default_target_atr),
        "expected_hold_days_multiplier": regime_cfg.get("expected_hold_days_multiplier", 1.0),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def record_regime(
    session: Session,
    as_of: dt.date,
    regime: RegimeType,
    vix: float | None,
    spy_adx: float | None,
    sector_breadth_pct: float | None,
    transition_flag: bool,
    transition_reason: str | None,
) -> RegimeHistory:
    """Upsert the `RegimeHistory` row for `as_of`."""
    existing = session.execute(
        select(RegimeHistory).where(RegimeHistory.ts == as_of)
    ).scalar_one_or_none()

    row = existing or RegimeHistory(ts=as_of)
    row.regime = regime
    row.vix = vix
    row.spy_adx = spy_adx
    row.sector_breadth_pct = sector_breadth_pct
    row.transition_flag = transition_flag
    row.transition_reason = transition_reason

    if existing is None:
        session.add(row)

    return row


# ---------------------------------------------------------------------------
# MR-004 regime performance tracking
# ---------------------------------------------------------------------------

def track_regime_performance(session: Session, regime: RegimeType, as_of: dt.date) -> RegimePerformance:
    """Compute and upsert `RegimePerformance` for `regime` as of `as_of`.

    Matches closed `Trade` rows to regime-days by joining each trade's
    `entry_date` (truncated to a date) against `RegimeHistory.ts` rows whose
    `regime` equals the target regime. Metrics:
      - win_rate: fraction of matched trades with realized_pnl > 0.
      - avg_r_multiple: mean of `realized_r_multiple` across matched trades.
      - max_drawdown: worst peak-to-trough drop in *cumulative dollar PnL*
        across matched trades ordered by exit_date (not normalized by
        account equity, since no equity curve/capital baseline is available
        to this function -- documented assumption).
      - sharpe_ratio: mean(realized_r_multiple) / stdev(realized_r_multiple)
        across matched trades, NOT annualized (per-trade Sharpe proxy;
        documented assumption since trade cadence is irregular).
    """
    try:
        regime_dates = set(
            session.execute(
                select(RegimeHistory.ts).where(RegimeHistory.regime == regime)
            )
            .scalars()
            .all()
        )
    except Exception as e:
        logger.warning("track_regime_performance: failed to load regime dates: %s", e)
        regime_dates = set()

    try:
        closed_trades = (
            session.execute(select(Trade).where(Trade.exit_date.isnot(None)))
            .scalars()
            .all()
        )
    except Exception as e:
        logger.warning("track_regime_performance: failed to load trades: %s", e)
        closed_trades = []

    matched = [t for t in closed_trades if t.entry_date is not None and t.entry_date.date() in regime_dates]
    matched.sort(key=lambda t: t.exit_date or t.entry_date)

    trade_count = len(matched)
    win_rate = None
    avg_r_multiple = None
    max_drawdown = None
    sharpe_ratio = None

    if trade_count > 0:
        wins = sum(1 for t in matched if (t.realized_pnl or 0) > 0)
        win_rate = wins / trade_count

        r_multiples = [t.realized_r_multiple for t in matched if t.realized_r_multiple is not None]
        if r_multiples:
            avg_r_multiple = statistics.mean(r_multiples)
            if len(r_multiples) >= 2:
                stdev = statistics.pstdev(r_multiples)
                sharpe_ratio = (avg_r_multiple / stdev) if stdev > 0 else None

        pnl_series = [t.realized_pnl or 0.0 for t in matched]
        cumulative = 0.0
        running_max = 0.0
        worst_drawdown = 0.0
        for pnl in pnl_series:
            cumulative += pnl
            running_max = max(running_max, cumulative)
            drawdown = cumulative - running_max
            worst_drawdown = min(worst_drawdown, drawdown)
        max_drawdown = worst_drawdown

    existing = session.execute(
        select(RegimePerformance).where(
            RegimePerformance.regime == regime, RegimePerformance.as_of == as_of
        )
    ).scalar_one_or_none()

    row = existing or RegimePerformance(regime=regime, as_of=as_of)
    row.win_rate = win_rate
    row.avg_r_multiple = avg_r_multiple
    row.max_drawdown = max_drawdown
    row.sharpe_ratio = sharpe_ratio
    row.trade_count = trade_count

    if existing is None:
        session.add(row)

    return row

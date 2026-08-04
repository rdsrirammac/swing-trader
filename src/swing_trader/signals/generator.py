"""SR-001 / SR-004: signal generation -- rating + bracket-order computation.

`generate_signal` pulls together the latest `Prediction`, `StockFeature`,
`DailyMetric` and `RegimeHistory` rows for a ticker/date, scores them via
`signals.rating.compute_rating_score`, computes SR-004 bracket orders
(stop-loss / take-profit 1 & 2 / position size), and persists a
`SignalRating` row linked back to the `Prediction`.

Design note: rather than requiring the caller to pre-fetch entry_price /
atr_14 / portfolio_value as explicit args, this function queries the latest
`StockPrice` close and the latest `StockFeature.atr_14` itself (it already
needs a `session` to load Prediction/StockFeature/etc, so this keeps the
call site simple: `generate_signal(session, ticker, as_of)`). Only
`portfolio_value` is an explicit (defaulted) argument, since sizing
inherently depends on portfolio state the caller controls.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import (
    DailyMetric,
    Prediction,
    Rating,
    RegimeHistory,
    SignalRating,
    StockFeature,
    StockPrice,
)
from swing_trader.logging_setup import get_logger
from swing_trader.signals.earnings_blackout import is_earnings_blackout
from swing_trader.signals.rating import compute_rating_score

logger = get_logger("signals.generator")


def _latest(session: Session, model, ticker: str, as_of: dt.date, ts_attr: str):
    col = getattr(model, ts_attr)
    stmt = (
        select(model)
        .where(model.ticker == ticker, col <= as_of)
        .order_by(col.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def generate_signal(
    session: Session,
    ticker: str,
    as_of: dt.date,
    portfolio_value: float = 100_000.0,
) -> SignalRating:
    """SR-001/SR-004: generate (or refresh) the SignalRating for `ticker` as of `as_of`.

    Steps:
      1. Load latest Prediction / StockFeature / RegimeHistory at/before `as_of`
         (raises ValueError if no Prediction exists -- there's nothing to rate).
      2. Load latest StockPrice close at/before `as_of` as the entry price.
      3. sentiment_score <- StockFeature.news_sentiment_3d_avg (0.0 if missing).
      4. pe_percentile <- StockFeature.pe_percentile_sector, falling back to
         StockFeature.pe_percentile_history.
      5. Score via `rating.compute_rating_score`.
      6. SR-004 bracket orders (multiples read from config, not hardcoded):
             stop_loss     = entry_price - positions.default_stop_atr_multiple * atr_14
             take_profit_1 = entry_price + positions.target_1_atr_multiple   * atr_14
             take_profit_2 = entry_price + positions.target_2_atr_multiple   * atr_14
             risk_per_share = entry_price - stop_loss
             position_shares = (portfolio_value * risk.max_risk_per_trade_pct) / risk_per_share
         Left as None if atr_14 is unavailable (feature engineering hasn't run yet).
      7. SR-005/EC-004 earnings blackout flag via `earnings_blackout.is_earnings_blackout`.
      8. Upsert-by-prediction_id and persist the `SignalRating` row.

    Raises:
        ValueError: no Prediction found, or no StockPrice found to price the signal.
    """
    settings = get_settings()

    prediction = _latest(session, Prediction, ticker, as_of, "as_of")
    if prediction is None:
        raise ValueError(f"No Prediction found for {ticker} as of {as_of}; cannot generate a signal")

    feature = _latest(session, StockFeature, ticker, as_of, "ts")
    _metric = _latest(session, DailyMetric, ticker, as_of, "ts")  # reserved for future use (e.g. IV rank)

    regime_row = session.execute(
        select(RegimeHistory)
        .where(RegimeHistory.ts <= as_of)
        .order_by(RegimeHistory.ts.desc())
        .limit(1)
    ).scalar_one_or_none()

    price_row = session.execute(
        select(StockPrice)
        .where(StockPrice.ticker == ticker, StockPrice.interval == "1d", StockPrice.ts <= as_of)
        .order_by(StockPrice.ts.desc())
        .limit(1)
    ).scalar_one_or_none()
    if price_row is None:
        raise ValueError(f"No StockPrice found for {ticker} as of {as_of}; cannot price the signal")
    entry_price = float(price_row.close)

    sentiment_score = 0.0
    pe_percentile: float | None = None
    atr_14: float | None = None
    if feature is not None:
        sentiment_score = float(feature.news_sentiment_3d_avg or 0.0)
        pe_percentile = feature.pe_percentile_sector
        if pe_percentile is None:
            pe_percentile = feature.pe_percentile_history
        atr_14 = feature.atr_14

    regime_value = regime_row.regime.value if regime_row is not None else None
    predicted_return = prediction.expected_return_10d or 0.0
    ci_lower = prediction.ci_lower if prediction.ci_lower is not None else entry_price
    ci_upper = prediction.ci_upper if prediction.ci_upper is not None else entry_price

    score, rating_label = compute_rating_score(
        predicted_return=predicted_return,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        price=entry_price,
        sentiment_score=sentiment_score,
        pe_percentile=pe_percentile,
        regime=regime_value,
        cfg=settings,
    )

    stop_atr_mult = settings.get("positions.default_stop_atr_multiple", 2.0)
    target1_atr_mult = settings.get("positions.target_1_atr_multiple", 2.0)
    target2_atr_mult = settings.get("positions.target_2_atr_multiple", 3.5)
    max_risk_pct = settings.get("risk.max_risk_per_trade_pct", 0.02)

    suggested_stop = suggested_target_1 = suggested_target_2 = None
    suggested_shares = suggested_pct = None
    if atr_14:
        suggested_stop = entry_price - (stop_atr_mult * atr_14)
        suggested_target_1 = entry_price + (target1_atr_mult * atr_14)
        suggested_target_2 = entry_price + (target2_atr_mult * atr_14)
        risk_per_share = entry_price - suggested_stop
        if risk_per_share > 0:
            suggested_shares = (portfolio_value * max_risk_pct) / risk_per_share
            suggested_pct = (
                (suggested_shares * entry_price) / portfolio_value if portfolio_value else None
            )
    else:
        logger.info("No ATR-14 available for %s as of %s; bracket orders left unset", ticker, as_of)

    blackout = is_earnings_blackout(session, ticker, as_of)

    existing = session.execute(
        select(SignalRating).where(SignalRating.prediction_id == prediction.id)
    ).scalar_one_or_none()

    signal = existing or SignalRating(prediction_id=prediction.id, ticker=ticker, as_of=as_of)
    signal.ticker = ticker
    signal.as_of = as_of
    signal.score = score
    signal.rating = Rating(rating_label)
    signal.suggested_entry = entry_price
    signal.suggested_stop = suggested_stop
    signal.suggested_target_1 = suggested_target_1
    signal.suggested_target_2 = suggested_target_2
    signal.suggested_position_shares = suggested_shares
    signal.suggested_position_pct = suggested_pct
    signal.earnings_blackout = blackout

    session.add(signal)
    session.flush()
    return signal


def kelly_position_size(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float,
    portfolio_value: float,
    risk_per_share: float,
) -> float:
    """Alternative position sizer to the fixed 2%-risk approach (SR-004 extension).

    Computes the Kelly fraction f* = W - (1 - W) / b, where W is the
    historical win rate and b = avg_win_r / avg_loss_r is the win/loss payoff
    ratio expressed in R-multiples (avg_loss_r must be > 0, i.e. expressed as
    a positive magnitude).

    This function applies HALF-KELLY (f*/2) rather than full Kelly. Full
    Kelly is provably growth-optimal only under exact, stationary knowledge
    of win_rate/avg_win_r/avg_loss_r; in practice those are noisy, finite-
    sample estimates from a live trading system, and full Kelly is well
    documented to produce large, painful drawdowns when the edge is
    overestimated or the underlying edge decays. Half-Kelly gives up some
    long-run growth rate in exchange for materially lower variance/drawdown,
    which is the standard, conservative practitioner compromise -- and it
    composes with (does not replace) the hard caps enforced elsewhere
    (`risk_manager.position_size`'s max_single_position_pct clamp, portfolio
    heat, etc.).

    Returns:
        Non-negative share count. Returns 0.0 if the edge is non-positive or
        inputs are degenerate (avg_loss_r <= 0, risk_per_share <= 0, or
        portfolio_value <= 0).
    """
    if avg_loss_r <= 0 or risk_per_share <= 0 or portfolio_value <= 0:
        return 0.0

    b = avg_win_r / avg_loss_r
    if b <= 0:
        return 0.0

    kelly_fraction = win_rate - (1 - win_rate) / b
    half_kelly_fraction = max(kelly_fraction, 0.0) / 2.0

    dollar_risk = portfolio_value * half_kelly_fraction
    return max(dollar_risk / risk_per_share, 0.0)

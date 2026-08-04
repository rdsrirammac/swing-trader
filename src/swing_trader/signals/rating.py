"""SR-002 / SR-003: composite signal scoring and rating-label lookup.

SR-002 defines a fixed pseudocode scoring recipe (momentum 40%, confidence
20%, sentiment 20%, valuation 10%, volatility regime 10%) whose internal
break points (0.05/0.02 momentum bands, 0.03/0.08 CI-width bands, the 1.5x
sentiment weight, the 0.8/0.2 PE percentile bands, ...) are given as fixed
constants in the SRS itself -- they are NOT present in config/settings.yaml
and are therefore transcribed verbatim below as module constants. Only the
final score -> rating-label cutoffs are config-able (via `rating.*` in
settings.yaml), so those are read from `get_settings()` (or an injected
`cfg` with a compatible `.get()` for testability) instead of being
hardcoded.
"""
from __future__ import annotations

from swing_trader.config import get_settings
from swing_trader.logging_setup import get_logger

logger = get_logger("signals.rating")

# --- SR-002 fixed pseudocode constants (not settings.yaml-configurable) ---
_MOMENTUM_STRONG_UP = 0.05
_MOMENTUM_UP = 0.02
_MOMENTUM_STRONG_DOWN = -0.05
_MOMENTUM_DOWN = -0.02
_MOMENTUM_STRONG_SCORE = 2.0
_MOMENTUM_SCORE = 1.0

_CI_WIDTH_TIGHT = 0.03
_CI_WIDTH_WIDE = 0.08
_CI_TIGHT_MULTIPLIER = 1.2
_CI_WIDE_MULTIPLIER = 0.8

_SENTIMENT_WEIGHT = 1.5

_PE_HIGH_PCTILE = 0.8
_PE_LOW_PCTILE = 0.2
_PE_ADJUSTMENT = 0.5

_HIGH_VOL_PENALTY = 0.5

# The SR-002 pseudocode uses the literal token "high_vol"; the DB's
# RegimeType enum instead uses RegimeType.HIGH_VOLATILITY.value ==
# "high_volatility". Both spellings are treated as equivalent here.
_HIGH_VOL_REGIME_VALUES = {"high_vol", "high_volatility"}


def compute_rating_score(
    predicted_return: float,
    ci_lower: float,
    ci_upper: float,
    price: float,
    sentiment_score: float,
    pe_percentile: float | None,
    regime: str | None,
    cfg=None,
) -> tuple[float, str]:
    """SR-002: compute the composite signal score and map it to a rating label.

    Faithful transcription of the SR-002 pseudocode:

        Score = 0
        # Price momentum (40% weight)
        if predicted_return > 0.05:        score += 2.0
        elif predicted_return > 0.02:      score += 1.0
        elif predicted_return < -0.05:     score -= 2.0
        elif predicted_return < -0.02:     score -= 1.0
        # Confidence (20% weight)
        ci_width = (upper - lower) / price
        if ci_width < 0.03:                score *= 1.2
        elif ci_width > 0.08:              score *= 0.8
        # Sentiment (20% weight)
        score += sentiment_score * 1.5
        # Valuation (10% weight)
        if pe_percentile > 0.8:            score -= 0.5
        elif pe_percentile < 0.2:          score += 0.5
        # Volatility regime (10% weight)
        if regime == "high_vol" and predicted_return < 0: score -= 0.5

    The final score -> rating-label mapping uses the config-able cutoffs
    `rating.strong_buy_score` / `rating.buy_score` / `rating.sell_score` /
    `rating.weak_sell_score` (defaults 1.5 / 0.5 / -1.5 / -0.5) instead of
    the SRS's hardcoded 1.5/0.5/-1.5/-0.5, so operators can tune them.

    Args:
        predicted_return: model's expected return (e.g. Prediction.expected_return_10d).
        ci_lower, ci_upper: prediction confidence interval bounds.
        price: current price, used to normalize the CI width.
        sentiment_score: e.g. StockFeature.news_sentiment_3d_avg, range ~[-1, 1].
        pe_percentile: e.g. StockFeature.pe_percentile_sector (or history fallback);
            None skips the valuation adjustment entirely.
        regime: RegimeType value string (or the SRS's literal "high_vol"); any
            other value (or None) is treated as not-high-vol.
        cfg: optional Settings-like object (must implement `.get(path, default)`);
            defaults to `get_settings()`.

    Returns:
        (score, rating_label) where rating_label is one of
        "Strong Buy" / "Buy" / "Hold" / "Sell".
    """
    settings = cfg if cfg is not None else get_settings()

    score = 0.0

    # Price momentum (40% weight)
    if predicted_return > _MOMENTUM_STRONG_UP:
        score += _MOMENTUM_STRONG_SCORE
    elif predicted_return > _MOMENTUM_UP:
        score += _MOMENTUM_SCORE
    elif predicted_return < _MOMENTUM_STRONG_DOWN:
        score -= _MOMENTUM_STRONG_SCORE
    elif predicted_return < _MOMENTUM_DOWN:
        score -= _MOMENTUM_SCORE

    # Confidence (20% weight)
    if price:
        ci_width = (ci_upper - ci_lower) / price
        if ci_width < _CI_WIDTH_TIGHT:
            score *= _CI_TIGHT_MULTIPLIER
        elif ci_width > _CI_WIDTH_WIDE:
            score *= _CI_WIDE_MULTIPLIER
    else:
        logger.warning("compute_rating_score called with price<=0; skipping CI-width adjustment")

    # Sentiment (20% weight)
    score += (sentiment_score or 0.0) * _SENTIMENT_WEIGHT

    # Valuation (10% weight) -- skipped entirely if pe_percentile is unknown
    if pe_percentile is not None:
        if pe_percentile > _PE_HIGH_PCTILE:
            score -= _PE_ADJUSTMENT
        elif pe_percentile < _PE_LOW_PCTILE:
            score += _PE_ADJUSTMENT

    # Volatility regime (10% weight)
    regime_norm = (regime or "").lower()
    if regime_norm in _HIGH_VOL_REGIME_VALUES and predicted_return < 0:
        score -= _HIGH_VOL_PENALTY

    # Map to rating (config-able cutoffs)
    strong_buy_cut = settings.get("rating.strong_buy_score", 1.5)
    buy_cut = settings.get("rating.buy_score", 0.5)
    sell_cut = settings.get("rating.sell_score", -1.5)
    weak_sell_cut = settings.get("rating.weak_sell_score", -0.5)

    if score >= strong_buy_cut:
        rating = "Strong Buy"
    elif score >= buy_cut:
        rating = "Buy"
    elif score <= sell_cut:
        rating = "Sell"
    elif score <= weak_sell_cut:
        rating = "Hold"  # weak sell per SR-002 note; SR-003 has no separate
        # "Weak Sell" label so this collapses into Hold, same as the neutral case.
    else:
        rating = "Hold"

    return score, rating


# --- SR-003: rating definitions lookup -------------------------------------

RATING_DEFINITIONS: dict[str, dict] = {
    "Strong Buy": {
        "criteria": (
            "Composite score >= rating.strong_buy_score (default 1.5): strong "
            "positive predicted return, tight confidence interval, supportive "
            "sentiment/valuation, and not in a high-volatility regime with a "
            "negative predicted return."
        ),
        "suggested_position_size_pct": 0.10,
    },
    "Buy": {
        "criteria": (
            "Composite score >= rating.buy_score (default 0.5) but below the "
            "Strong Buy cutoff: moderately positive conviction."
        ),
        "suggested_position_size_pct": 0.05,
    },
    "Hold": {
        "criteria": (
            "Composite score between rating.weak_sell_score and rating.buy_score "
            "(default -0.5..0.5), or a mild negative score above the Sell cutoff: "
            "no clear edge either direction; maintain existing exposure, do not add."
        ),
        "suggested_position_size_pct": None,
    },
    "Trim": {
        "criteria": (
            "Held position whose signal has weakened materially since entry "
            "(e.g. rating dropped from Buy/Strong Buy toward Hold, or a target "
            "has been reached) without a full Sell signal: take partial profits "
            "per positions.trim_target_1_pct rather than exiting outright."
        ),
        "suggested_position_size_pct": None,
    },
    "Sell": {
        "criteria": (
            "Composite score <= rating.sell_score (default -1.5): strong "
            "negative conviction. Exit any existing position; do not open new ones."
        ),
        "suggested_position_size_pct": None,
    },
    "Watch": {
        "criteria": (
            "Ticker does not currently meet full Buy/Strong Buy criteria but is "
            "close, or a WatchlistItem trigger_condition has not yet fired "
            "(PF-007): monitor, no position taken."
        ),
        "suggested_position_size_pct": None,
    },
}

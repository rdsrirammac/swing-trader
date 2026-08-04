"""Unit tests for the SR-002 rating algorithm (swing_trader.signals.rating).

These transcribe the exact pseudocode branches from
SRS Swing_Trading_SRS_v1.0.md Section 3.7 (SR-002) and assert the output
matches by hand-computed expectation, not just "doesn't crash".
"""
from __future__ import annotations

from swing_trader.signals.rating import RATING_DEFINITIONS, compute_rating_score


def test_strong_positive_momentum_tight_ci_positive_sentiment_is_strong_buy():
    score, rating = compute_rating_score(
        predicted_return=0.06,  # > 0.05 -> +2.0
        ci_lower=0.04,
        ci_upper=0.08,  # ci_width = 0.04/100 = 0.0004 -> < 0.03 -> *1.2
        price=100.0,
        sentiment_score=0.5,  # +0.75
        pe_percentile=0.5,  # no adjustment
        regime="strong_trend",
    )
    assert rating == "Strong Buy"
    assert score > 1.5


def test_strong_negative_momentum_high_vol_is_sell():
    score, rating = compute_rating_score(
        predicted_return=-0.08,  # < -0.05 -> -2.0
        ci_lower=-0.10,
        ci_upper=-0.06,
        price=100.0,
        sentiment_score=-0.5,  # -0.75
        pe_percentile=0.9,  # overvalued -> -0.5
        regime="high_volatility",  # and predicted_return < 0 -> -0.5
    )
    assert rating == "Sell"
    assert score <= -1.5


def test_flat_prediction_is_hold():
    score, rating = compute_rating_score(
        predicted_return=0.001,
        ci_lower=-0.01,
        ci_upper=0.01,
        price=50.0,
        sentiment_score=0.0,
        pe_percentile=None,
        regime="range_bound",
    )
    assert rating == "Hold"
    assert -0.5 < score < 0.5


def test_moderate_positive_momentum_is_buy():
    score, rating = compute_rating_score(
        predicted_return=0.03,  # > 0.02 -> +1.0
        ci_lower=0.01,
        ci_upper=0.05,
        price=100.0,
        sentiment_score=0.1,
        pe_percentile=0.1,  # undervalued -> +0.5
        regime="weak_trend",
    )
    assert rating in ("Buy", "Strong Buy")  # depends on ci_width multiplier, but must be bullish
    assert score >= 0.5


def test_wide_confidence_interval_dampens_score():
    """A very wide CI should shrink the score via the *0.8 multiplier (branch
    only fires once score is nonzero, so use a momentum case to see the effect)."""
    # price=100 => ci_width = (upper-lower)/price; need >0.08 to hit the wide
    # branch, i.e. (upper-lower) > 8 in absolute price-unit terms.
    tight, _ = compute_rating_score(
        predicted_return=0.06, ci_lower=0.059, ci_upper=0.061, price=100.0,
        sentiment_score=0.0, pe_percentile=None, regime="range_bound",
    )
    wide, _ = compute_rating_score(
        predicted_return=0.06, ci_lower=-5.0, ci_upper=5.0, price=100.0,
        sentiment_score=0.0, pe_percentile=None, regime="range_bound",
    )
    assert wide < tight


def test_regime_high_vol_alias_matches_enum_value():
    """The SRS pseudocode checks `regime == "high_vol"`; our RegimeType enum
    value is "high_volatility" -- compute_rating_score must treat these as
    equivalent (explicit mapping, not a literal string match)."""
    score_enum_style, _ = compute_rating_score(
        predicted_return=-0.01, ci_lower=-0.02, ci_upper=0.0, price=100.0,
        sentiment_score=0.0, pe_percentile=None, regime="high_volatility",
    )
    score_no_vol, _ = compute_rating_score(
        predicted_return=-0.01, ci_lower=-0.02, ci_upper=0.0, price=100.0,
        sentiment_score=0.0, pe_percentile=None, regime="strong_trend",
    )
    assert score_enum_style < score_no_vol


def test_rating_definitions_cover_all_six_ratings():
    assert set(RATING_DEFINITIONS.keys()) == {
        "Strong Buy", "Buy", "Hold", "Trim", "Sell", "Watch",
    }
    assert RATING_DEFINITIONS["Strong Buy"]["suggested_position_size_pct"] == 0.10
    assert RATING_DEFINITIONS["Buy"]["suggested_position_size_pct"] == 0.05

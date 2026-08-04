"""Unit tests for swing_trader.models.regime_detector (SRS 3.5, MR-001..004).

Thresholds come from config/regimes.yaml (percent-unit fields like
sector_breadth_min_pct: 60 expect a 0-100 value, not a 0-1 fraction).
"""
from __future__ import annotations

from swing_trader.db.models import RegimeType
from swing_trader.models.regime_detector import classify_regime, regime_adjustments


def test_strong_trend_classification():
    regime = classify_regime(
        spy_adx=30, vix=15, sector_breadth_pct=70,
        bb_width_pct=None, atr_expansion_pct=None, pct_sp500_reporting_next_2wk=None,
    )
    assert regime == RegimeType.STRONG_TREND


def test_high_volatility_takes_precedence_over_strong_trend():
    # Even though ADX/breadth would qualify as strong trend, VIX > 25 wins.
    regime = classify_regime(
        spy_adx=30, vix=32, sector_breadth_pct=70,
        bb_width_pct=None, atr_expansion_pct=None, pct_sp500_reporting_next_2wk=None,
    )
    assert regime == RegimeType.HIGH_VOLATILITY


def test_high_volatility_via_atr_expansion():
    regime = classify_regime(
        spy_adx=10, vix=15, sector_breadth_pct=40,
        bb_width_pct=5, atr_expansion_pct=180, pct_sp500_reporting_next_2wk=None,
    )
    assert regime == RegimeType.HIGH_VOLATILITY


def test_range_bound_classification():
    regime = classify_regime(
        spy_adx=12, vix=18, sector_breadth_pct=45,
        bb_width_pct=6, atr_expansion_pct=90, pct_sp500_reporting_next_2wk=None,
    )
    assert regime == RegimeType.RANGE_BOUND


def test_weak_trend_classification():
    regime = classify_regime(
        spy_adx=20, vix=22, sector_breadth_pct=50,
        bb_width_pct=15, atr_expansion_pct=100, pct_sp500_reporting_next_2wk=None,
    )
    assert regime == RegimeType.WEAK_TREND


def test_earnings_season_precedence_over_strong_trend():
    regime = classify_regime(
        spy_adx=30, vix=15, sector_breadth_pct=70,
        bb_width_pct=None, atr_expansion_pct=None, pct_sp500_reporting_next_2wk=25,
    )
    assert regime == RegimeType.EARNINGS_SEASON


def test_regime_adjustments_reduce_size_in_high_vol():
    adj = regime_adjustments(RegimeType.HIGH_VOLATILITY)
    assert adj["position_size_multiplier"] < 1.0

    normal_adj = regime_adjustments(RegimeType.STRONG_TREND)
    assert normal_adj["position_size_multiplier"] >= adj["position_size_multiplier"]

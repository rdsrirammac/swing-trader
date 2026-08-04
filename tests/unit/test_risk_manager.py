"""Unit tests for swing_trader.signals.risk_manager (SRS 3.8, RM-001..007)."""
from __future__ import annotations

import pandas as pd
import pytest

from swing_trader.db.models import Holding, PositionStatus, TickerUniverse, TickerStatus
from swing_trader.signals.risk_manager import (
    blocks_new_buys,
    correlation_check,
    heat_status,
    portfolio_heat,
    position_size,
    sector_concentration,
    sector_limit_breached,
    volatility_adjusted_sizing,
)


def test_position_size_basic_risk_math():
    # $100k portfolio, 2% risk = $2000 risk budget; $15 risk/share (entry 50,
    # stop 35) -> 133.33 shares; position value = 133.33*$50 = $6,667, safely
    # under the 10%-of-portfolio ($10,000) cap, so the raw risk formula is
    # what determines the result here (see the capped case below).
    shares = position_size(portfolio_value=100_000, entry=50.0, stop=35.0, risk_pct=0.02)
    assert shares == pytest.approx(2_000 / 15, rel=0.01)


def test_position_size_basic_risk_math_when_cap_binds():
    # Same $2000 risk budget but a tight $2 stop -> raw sizing wants 1000
    # shares ($50,000 position, 50% of the portfolio); RM-001's 10%-of-
    # portfolio cap clips this down to $10,000 / $50 = 200 shares.
    shares = position_size(portfolio_value=100_000, entry=50.0, stop=48.0, risk_pct=0.02)
    assert shares == pytest.approx(200.0, rel=0.01)


def test_position_size_never_exceeds_max_single_position_cap():
    # Huge risk_pct request should still be clipped so position value <= 10% of portfolio.
    shares = position_size(portfolio_value=100_000, entry=10.0, stop=9.99, risk_pct=0.5)
    position_value = shares * 10.0
    assert position_value <= 100_000 * 0.10 * 1.001  # small tolerance


def test_heat_status_thresholds():
    # Bands: green <10%, yellow 10%..max_portfolio_heat_pct (inclusive,
    # default 20%), red strictly above max_portfolio_heat_pct.
    assert heat_status(0.05) == "green"
    assert heat_status(0.099) == "green"
    assert heat_status(0.10) == "yellow"
    assert heat_status(0.19) == "yellow"
    assert heat_status(0.20) == "yellow"  # == max_portfolio_heat_pct is still yellow, not red
    assert heat_status(0.21) == "red"
    assert heat_status(0.35) == "red"


def test_blocks_new_buys():
    assert blocks_new_buys(0.25) is True
    assert blocks_new_buys(0.05) is False


def test_volatility_adjusted_sizing_reduces_in_high_vol_regime():
    normal = volatility_adjusted_sizing(base_shares=200, regime="strong_trend", vix=15)
    stressed = volatility_adjusted_sizing(base_shares=200, regime="high_volatility", vix=30)
    assert stressed["shares"] < normal["shares"]
    assert stressed["stop_atr_multiple"] >= normal["stop_atr_multiple"]


def test_volatility_adjusted_sizing_triggers_on_vix_even_without_regime_label():
    stressed = volatility_adjusted_sizing(base_shares=100, regime="weak_trend", vix=30)
    assert stressed["shares"] < 100


def test_portfolio_heat_and_sector_concentration(db_session, portfolio):
    # Zero out cash so sector/heat percentages are driven purely by holdings
    # (portfolio_heat/sector_concentration both use cash_balance + holdings
    # value as the portfolio-value denominator -- with the fixture's default
    # $100k cash, a few thousand dollars of holdings would barely move the
    # percentages, which isn't a useful test of the concentration math).
    portfolio.cash_balance = 0.0
    db_session.add_all(
        [
            TickerUniverse(ticker="AAA", status=TickerStatus.ACTIVE, sector="Technology"),
            TickerUniverse(ticker="BBB", status=TickerStatus.ACTIVE, sector="Technology"),
            TickerUniverse(ticker="CCC", status=TickerStatus.ACTIVE, sector="Energy"),
        ]
    )
    db_session.add_all(
        [
            Holding(
                portfolio_id=portfolio.id, ticker="AAA", shares=100, entry_price=50.0,
                stop_loss=45.0, status=PositionStatus.ACTIVE,
            ),
            Holding(
                portfolio_id=portfolio.id, ticker="BBB", shares=50, entry_price=100.0,
                stop_loss=95.0, status=PositionStatus.ACTIVE,
            ),
            Holding(
                portfolio_id=portfolio.id, ticker="CCC", shares=20, entry_price=30.0,
                stop_loss=28.0, status=PositionStatus.ACTIVE,
            ),
        ]
    )
    db_session.commit()

    heat = portfolio_heat(db_session, portfolio.id)
    assert heat > 0

    concentration = sector_concentration(db_session, portfolio.id)
    assert "Technology" in concentration
    assert "Energy" in concentration
    # Technology = AAA(5000) + BBB(5000) = 10000; Energy = CCC(600); total ~10600
    assert concentration["Technology"] > concentration["Energy"]

    breached = sector_limit_breached(concentration)
    assert "Technology" in breached  # ~94% >> 30% limit


def test_correlation_check_rejects_highly_correlated_candidate(db_session, portfolio):
    db_session.add(
        Holding(
            portfolio_id=portfolio.id, ticker="AAA", shares=10, entry_price=100.0,
            stop_loss=90.0, status=PositionStatus.ACTIVE,
        )
    )
    db_session.commit()

    dates = pd.bdate_range("2026-01-01", periods=60)
    base = pd.Series(range(100, 160), index=dates, dtype=float)
    price_lookup = {
        "AAA": base,
        "NEW": base * 1.01,  # near-perfectly correlated
    }

    reject, correlations = correlation_check(db_session, portfolio.id, "NEW", price_lookup)
    assert reject is True
    assert correlations["AAA"] > 0.8

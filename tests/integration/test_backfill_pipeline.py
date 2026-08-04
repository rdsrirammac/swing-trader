"""Integration test for the TB-001..TB-004 auto-backfill pipeline
(swing_trader.portfolio.backfill.run_backfill), with `YFinanceClient`
fully mocked out -- no network access, no real yfinance calls.

This exercises the real orchestration logic (phase sequencing, BackfillJob
progress rows, StockPrice ingestion, the TB-003 quality gate) end-to-end
against an in-memory SQLite database.

KNOWN GAP this test documents rather than hides: Phase 5 (FEATURES) calls
`swing_trader.features.engineering.build_feature_row(session, ticker)`, but
that function's actual signature (built by a different workstream) takes
many more required positional arguments (price_history, spy_history, info,
news_rows, ...). The call currently raises TypeError, which
`_phase_features` catches and logs as a non-critical failure -- so no
StockFeature row ever gets created during backfill today, which means the
TB-003 quality gate's feature_completeness check can never pass, and every
newly-backfilled ticker ends up `status=failed` even with perfect price
data. This is tracked as a GitHub issue (see ROADMAP.md /
.github/ISSUE_TEMPLATE backlog) -- "Wire Phase 5 feature warm-up to the
real build_feature_row signature." The test below asserts *today's actual
behavior* (price data lands correctly, gate fails on missing features) so
it will fail loudly -- as a useful regression signal -- once that gap is
fixed and should be updated at that point to assert `status == ACTIVE`.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from swing_trader.data.yf_client import YFinanceClient
from swing_trader.db.models import BackfillJob, StockPrice, TickerStatus, TickerUniverse
from swing_trader.portfolio.backfill import check_quality_gate, run_backfill, screen_ticker


class _FakeOptionChain:
    def __init__(self):
        self.calls = pd.DataFrame({"volume": [100, 200]})
        self.puts = pd.DataFrame({"volume": [50, 60]})


class FakeYFinanceClient:
    """Stands in for `YFinanceClient.instance()` -- no network calls."""

    def __init__(self, n_days: int = 260):
        end = pd.Timestamp.now().normalize()
        dates = pd.bdate_range(end=end, periods=n_days)
        close = 100 + np.cumsum(np.random.default_rng(7).normal(0.1, 1.0, size=n_days))
        self._daily = pd.DataFrame(
            {
                "Open": close - 0.3,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": np.random.default_rng(7).integers(2_000_000, 6_000_000, size=n_days),
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            },
            index=dates,
        )

    def get_history(self, ticker, period="1y", interval="1d", prepost=False):
        if interval == "1d":
            return self._daily
        # intraday: a tiny frame is fine, it's non-critical
        idx = pd.date_range(end=pd.Timestamp.now(), periods=5, freq="30min")
        return pd.DataFrame(
            {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 10000}, index=idx
        )

    def get_info(self, ticker):
        return {
            "trailingPE": 25.0, "marketCap": 2_000_000_000, "sector": "Technology",
            "regularMarketPrice": float(self._daily["Close"].iloc[-1]), "quoteType": "EQUITY",
        }

    def get_quarterly_financials(self, ticker):
        return pd.DataFrame()

    def get_quarterly_balance_sheet(self, ticker):
        return pd.DataFrame()

    def get_quarterly_earnings(self, ticker):
        return pd.DataFrame()

    def get_calendar(self, ticker):
        return {"Earnings Date": [dt.date.today() + dt.timedelta(days=20)]}

    def get_recommendations(self, ticker):
        return pd.DataFrame()

    def get_upgrades_downgrades(self, ticker):
        return pd.DataFrame()

    def get_option_expirations(self, ticker):
        return (dt.date.today() + dt.timedelta(days=14)).isoformat(),

    def get_options_chain(self, ticker, expiration):
        return _FakeOptionChain()

    def get_news(self, ticker):
        return [
            {"title": "Company announces new product", "providerPublishTime": int(pd.Timestamp.now().timestamp()), "publisher": "TestWire"},
        ]

    def get_actions(self, ticker):
        return pd.DataFrame()

    def get_institutional_holders(self, ticker):
        return pd.DataFrame()


@pytest.fixture()
def fake_yf(monkeypatch):
    fake = FakeYFinanceClient()
    monkeypatch.setattr(YFinanceClient, "instance", staticmethod(lambda: fake))
    return fake


def test_run_backfill_ingests_price_data_and_tracks_phases(db_session, fake_yf, monkeypatch):
    # run_backfill opens its own session_scope() internally when not given
    # one; redirect that to our test's in-memory session/engine so we can
    # inspect the results without needing a real Postgres connection.
    import swing_trader.portfolio.backfill as backfill_module

    monkeypatch.setattr(backfill_module, "session_scope", lambda: _passthrough_scope(db_session))

    run_backfill("TESTX")

    bars = db_session.query(StockPrice).filter(
        StockPrice.ticker == "TESTX", StockPrice.interval == "1d"
    ).count()
    assert bars >= 200  # TB-003 min_daily_bars

    jobs = db_session.query(BackfillJob).filter(BackfillJob.ticker == "TESTX").all()
    phases_seen = {j.phase for j in jobs}
    assert {"price", "fundamentals", "news", "options", "features", "model_warmup"} <= phases_seen

    price_job = next(j for j in jobs if j.phase == "price")
    assert price_job.status == "done"

    ticker_row = db_session.query(TickerUniverse).filter(TickerUniverse.ticker == "TESTX").one()
    # See module docstring: Phase 5 (features) does not yet match
    # build_feature_row's real signature, so the quality gate currently
    # always fails on missing feature_completeness even with good price data.
    passed, reasons = check_quality_gate(db_session, "TESTX")
    assert passed is False
    assert any("feature_completeness" in r or "StockFeature" in r for r in reasons)
    assert ticker_row.status == TickerStatus.FAILED


def test_screen_ticker_accepts_valid_candidate():
    ok, reason = screen_ticker(
        info={"regularMarketPrice": 150.0, "quoteType": "EQUITY"},
        avg_daily_volume=5_000_000,
        atr_pct=0.03,
    )
    assert ok is True, reason


def test_screen_ticker_rejects_illiquid_candidate():
    ok, reason = screen_ticker(
        info={"regularMarketPrice": 150.0, "quoteType": "EQUITY"},
        avg_daily_volume=10_000,  # below 1,000,000 floor
        atr_pct=0.03,
    )
    assert ok is False
    assert "avg_daily_volume" in reason


def test_screen_ticker_rejects_out_of_price_band():
    ok, reason = screen_ticker(
        info={"regularMarketPrice": 5.0, "quoteType": "EQUITY"},
        avg_daily_volume=5_000_000,
        atr_pct=0.03,
    )
    assert ok is False
    assert "price" in reason


class _passthrough_scope:
    """A `session_scope()`-compatible context manager that reuses an
    already-open test session instead of opening a new engine connection
    (there is no real Postgres in the test environment)."""

    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.session.commit()
        else:
            self.session.rollback()
        return False

"""Tests for `features.engineering.recalculate_recent_features` -- the
TB-005 nightly rolling feature recalc that `portfolio.backfill.
run_daily_incremental_update` calls, and that `scripts/
backfill_historical_features.py` also uses (with a larger `days`) for the
one-time bulk historical backfill needed before `predict` can train on a
freshly-added ticker.

`YFinanceClient` is fully mocked out via the same `FakeYFinanceClient` used
by the backfill integration tests -- no network access, no real yfinance
calls.
"""
from __future__ import annotations

import pytest

from swing_trader.data.yf_client import YFinanceClient
from swing_trader.db.models import StockFeature
from swing_trader.features.engineering import recalculate_recent_features

from tests.integration.test_backfill_pipeline import FakeYFinanceClient


@pytest.fixture()
def fake_yf(monkeypatch):
    fake = FakeYFinanceClient(n_days=260)
    monkeypatch.setattr(YFinanceClient, "instance", staticmethod(lambda: fake))
    return fake


def test_recalculate_recent_features_writes_one_row_per_trading_day(db_session, fake_yf):
    written = recalculate_recent_features(db_session, "TESTY", days=60)
    db_session.commit()

    assert written == 60
    rows = db_session.query(StockFeature).filter(StockFeature.ticker == "TESTY").all()
    assert len(rows) == 60
    assert all(r.feature_completeness is not None for r in rows)


def test_recalculate_recent_features_is_point_in_time_correct(db_session, fake_yf):
    """Regression test for the "no look-ahead" guarantee described in
    `recalculate_recent_features`'s docstring: each recomputed day's row
    must only reflect price history up to and including that day.

    `ret_21d_vs_spy` needs at least 21 trailing closes on both the ticker
    and SPY sides (`relative_strength._trailing_return` returns None
    otherwise). Recomputing the *entire* 260-day fake history means the
    very first date recomputed only has itself (1 row) sliced in --
    nowhere near enough -- while the last date has the full window behind
    it. If `recalculate_recent_features` were (incorrectly) using the full,
    unsliced price history for every date instead of slicing per-date, the
    first row would have a non-None value too, since the *unsliced* history
    always has 260 rows available.
    """
    written = recalculate_recent_features(db_session, "TESTZ", days=260)
    db_session.commit()

    rows = (
        db_session.query(StockFeature)
        .filter(StockFeature.ticker == "TESTZ")
        .order_by(StockFeature.ts)
        .all()
    )
    assert len(rows) == written == 260
    assert rows[0].ret_21d_vs_spy is None
    assert rows[-1].ret_21d_vs_spy is not None


def test_recalculate_recent_features_handles_missing_price_history(db_session, monkeypatch):
    class _EmptyYF:
        def get_history(self, *a, **kw):
            import pandas as pd

            return pd.DataFrame()

    monkeypatch.setattr(YFinanceClient, "instance", staticmethod(lambda: _EmptyYF()))
    written = recalculate_recent_features(db_session, "NOPE", days=60)
    assert written == 0

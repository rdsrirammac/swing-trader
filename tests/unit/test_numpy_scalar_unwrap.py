"""Regression tests for the global `before_flush` numpy-scalar unwrap hook
(`db/base.py::_unwrap_numpy_scalars`).

Context: SQLite (used everywhere in this test suite) silently accepts
numpy scalar types (`np.float64`, `np.int64`, `np.bool_`, ...) on ORM
attributes, but psycopg2/PostgreSQL does not have a default adapter for
them and fails the insert with `InvalidSchemaName: schema "np" does not
exist`. This bug has hit `StockFeature` and `RegimeHistory` inserts in
production (real Postgres) without ever failing here in SQLite-backed
tests -- which is exactly why the fix needs its own explicit test: it
can't rely on "the existing tests still pass" as a signal, since SQLite
never exposed the bug in the first place.

IMPORTANT test-design note: SQLite's own storage/retrieval already
discards numpy typing on the way back out (`np.float64` is itself a
`float` subclass, and sqlite3's REAL column round-trip hands back a plain
`float` regardless of whether the hook ran) -- so asserting the type of a
value *read back from the DB* would pass even with the hook completely
disabled, and wouldn't actually be testing anything. The meaningful
assertion is the type of the ORM object's attribute checked in-memory,
immediately after `flush()`, before any reload/expire -- that reflects
whether the `before_flush` hook's `setattr(...)` actually ran.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from swing_trader.db.models import DailyMetric, RegimeHistory, RegimeType


def test_before_flush_unwraps_numpy_scalars_on_new_objects(db_session):
    row = RegimeHistory(
        ts=dt.date(2026, 8, 5),
        regime=RegimeType.WEAK_TREND,
        vix=np.float64(16.5),
        spy_adx=np.float64(22.317),
        sector_breadth_pct=np.float64(90.9090909090909),
        transition_flag=bool(np.bool_(False)),
    )
    assert type(row.vix) is np.float64  # sanity: confirms the object really is numpy-typed pre-flush

    db_session.add(row)
    db_session.flush()  # this is where psycopg2 would choke in production; the hook fires here

    # Check the in-memory object immediately, not a DB round-trip: SQLite's
    # own REAL-column storage already discards numpy typing on read-back
    # (np.float64 is itself a float subclass), so a post-reload type check
    # alone would pass even with the hook disabled -- see module docstring.
    assert type(row.vix) is float
    assert type(row.spy_adx) is float
    assert type(row.sector_breadth_pct) is float
    assert row.vix == 16.5


def test_before_flush_unwraps_numpy_scalars_on_dirty_objects(db_session):
    """Same guarantee on UPDATE (session.dirty), not just INSERT (session.new)."""
    today = dt.date(2026, 8, 5)
    metric = DailyMetric(ticker="AAPL", ts=today, market_cap=1_000_000.0)
    db_session.add(metric)
    db_session.commit()

    metric.market_cap = np.float64(2_500_000.5)
    assert type(metric.market_cap) is np.float64  # sanity, pre-flush

    db_session.flush()

    assert type(metric.market_cap) is float
    assert metric.market_cap == 2_500_000.5


def test_before_flush_leaves_plain_python_values_alone(db_session):
    """Sanity check: the hook shouldn't mangle ordinary Python values (in
    particular, must not treat strings as having a numpy-scalar `.item()`)."""
    row = RegimeHistory(
        ts=dt.date(2026, 8, 6),
        regime=RegimeType.RANGE_BOUND,
        vix=15.2,
        transition_flag=True,
        transition_reason="test transition",
    )
    db_session.add(row)
    db_session.flush()

    db_session.expire_all()
    reloaded = db_session.query(RegimeHistory).filter(RegimeHistory.ts == dt.date(2026, 8, 6)).one()
    assert reloaded.transition_reason == "test transition"
    assert reloaded.vix == 15.2
    assert reloaded.transition_flag is True

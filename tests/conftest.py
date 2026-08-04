"""Shared pytest fixtures.

Tests run against an in-memory SQLite database (not TimescaleDB/Postgres) --
fast, no external services required. This means TimescaleDB-specific SQL
(hypertable creation in scripts/init_db.py) is NOT exercised by the unit
test suite; that's covered by manually running `make init-db` against a
real `docker compose up -d` Postgres instance (see README "Verification").
The SQLAlchemy table DEFINITIONS themselves (db/models.py) are fully
exercised here, since `Base.metadata.create_all()` works identically on
SQLite.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Allow `import swing_trader...` without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swing_trader.db.base import Base  # noqa: E402
from swing_trader.db import models  # noqa: E402,F401


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def db_session(engine):
    Session = sessionmaker(bind=engine, future=True)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def portfolio(db_session):
    from swing_trader.db.models import Portfolio

    p = Portfolio(name="Test Portfolio", cash_balance=100_000.0, is_paper=True)
    db_session.add(p)
    db_session.commit()
    return p


@pytest.fixture()
def sample_ohlcv():
    """40 business days of gently-trending synthetic OHLCV data."""
    import numpy as np
    import pandas as pd

    dates = pd.bdate_range("2026-01-01", periods=60)
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.15, 1.0, size=len(dates)))
    high = close + rng.uniform(0.1, 1.5, size=len(dates))
    low = close - rng.uniform(0.1, 1.5, size=len(dates))
    open_ = close - rng.uniform(-0.5, 0.5, size=len(dates))
    volume = rng.integers(1_000_000, 5_000_000, size=len(dates))

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture()
def today() -> dt.date:
    return dt.date(2026, 8, 4)

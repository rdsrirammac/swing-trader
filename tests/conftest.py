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


class _FakeOptionChain:
    def __init__(self):
        import pandas as pd

        self.calls = pd.DataFrame({"volume": [100, 200]})
        self.puts = pd.DataFrame({"volume": [50, 60]})


class FakeYFinanceClient:
    """Stands in for `YFinanceClient.instance()` -- no network calls.

    Shared across test modules via the `fake_yf` fixture below rather than
    imported directly from one test file into another: `tests/` has no
    `__init__.py` (only `tests/integration/__init__.py` does), so
    `tests.integration.<module>` only resolves as an importable dotted path
    when pytest happens to be invoked in a way that puts the project root on
    `sys.path` (e.g. `python -m pytest`, which auto-adds cwd) -- it breaks
    under a plain `pytest` invocation (e.g. the Makefile's `$(BIN)/pytest`),
    which doesn't. Defining it once here and exposing it as a fixture avoids
    that class of import fragility entirely.
    """

    def __init__(self, n_days: int = 260):
        import numpy as np
        import pandas as pd

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
        import pandas as pd

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
        import pandas as pd

        return pd.DataFrame()

    def get_quarterly_balance_sheet(self, ticker):
        import pandas as pd

        return pd.DataFrame()

    def get_quarterly_earnings(self, ticker):
        import pandas as pd

        return pd.DataFrame()

    def get_calendar(self, ticker):
        return {"Earnings Date": [dt.date.today() + dt.timedelta(days=20)]}

    def get_recommendations(self, ticker):
        import pandas as pd

        return pd.DataFrame()

    def get_upgrades_downgrades(self, ticker):
        import pandas as pd

        return pd.DataFrame()

    def get_option_expirations(self, ticker):
        return (dt.date.today() + dt.timedelta(days=14)).isoformat(),

    def get_options_chain(self, ticker, expiration):
        return _FakeOptionChain()

    def get_news(self, ticker):
        import pandas as pd

        return [
            {"title": "Company announces new product", "providerPublishTime": int(pd.Timestamp.now().timestamp()), "publisher": "TestWire"},
        ]

    def get_actions(self, ticker):
        import pandas as pd

        return pd.DataFrame()

    def get_institutional_holders(self, ticker):
        import pandas as pd

        return pd.DataFrame()


@pytest.fixture()
def fake_yf(monkeypatch):
    from swing_trader.data.yf_client import YFinanceClient

    fake = FakeYFinanceClient()
    monkeypatch.setattr(YFinanceClient, "instance", staticmethod(lambda: fake))
    return fake

"""Structural tests for the database schema (SRS Section 2.1 / 5)."""
from __future__ import annotations

import datetime as dt

from swing_trader.db.base import Base
from swing_trader.db.models import (
    Alert,
    BackfillJob,
    Holding,
    Portfolio,
    PositionStatus,
    Prediction,
    SignalRating,
    StockFeature,
    StockPrice,
    TickerStatus,
    TickerUniverse,
    Trade,
)

EXPECTED_TABLES = {
    "ticker_universe", "backfill_jobs", "stock_prices", "stock_features",
    "daily_metrics", "news_sentiment", "regime_history", "regime_performance",
    "predictions", "model_performance", "signal_ratings", "portfolios",
    "holdings", "watchlist", "trades", "alerts", "notifications",
    "earnings_events", "economic_events", "pipeline_runs", "api_rate_limit_log",
}


def test_all_expected_tables_registered():
    assert EXPECTED_TABLES.issubset(set(Base.metadata.tables.keys()))


def test_tables_create_on_sqlite(engine):
    # `engine` fixture already ran create_all(); just confirm no tables were
    # silently skipped by re-running against the same engine (idempotent).
    Base.metadata.create_all(engine, checkfirst=True)
    with engine.connect() as conn:
        from sqlalchemy import inspect

        inspector = inspect(conn)
        table_names = set(inspector.get_table_names())
    assert EXPECTED_TABLES.issubset(table_names)


def test_portfolio_holding_relationship(db_session, portfolio):
    h = Holding(
        portfolio_id=portfolio.id, ticker="AAPL", shares=10, entry_price=190.0,
        stop_loss=180.0, status=PositionStatus.ACTIVE,
    )
    db_session.add(h)
    db_session.commit()
    db_session.refresh(portfolio)
    assert len(portfolio.holdings) == 1
    assert portfolio.holdings[0].ticker == "AAPL"


def test_prediction_signal_rating_relationship(db_session):
    as_of = dt.date(2026, 8, 1)
    pred = Prediction(ticker="AAPL", as_of=as_of, model_version="v1", expected_return_10d=0.05)
    db_session.add(pred)
    db_session.commit()

    rating = SignalRating(
        prediction_id=pred.id, ticker="AAPL", as_of=as_of, score=2.0, rating="Strong Buy",
    )
    db_session.add(rating)
    db_session.commit()
    db_session.refresh(pred)
    assert pred.rating is not None
    assert pred.rating.rating.value == "Strong Buy"


def test_ticker_universe_status_enum_roundtrip(db_session):
    t = TickerUniverse(ticker="MSFT", status=TickerStatus.PENDING)
    db_session.add(t)
    db_session.commit()
    db_session.refresh(t)
    assert t.status == TickerStatus.PENDING


def test_backfill_job_tracks_progress(db_session):
    db_session.add(TickerUniverse(ticker="NVDA", status=TickerStatus.BACKFILLING))
    job = BackfillJob(ticker="NVDA", phase="price", status="running", pct_complete=50.0)
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)
    assert job.pct_complete == 50.0
    assert job.attempt == 1

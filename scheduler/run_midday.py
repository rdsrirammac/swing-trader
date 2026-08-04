"""Midday job (SRS Section 6.3 / com.swingtrader.midday, 12:00 ET).

Refreshes 30-minute intraday bars for all active tickers via
`swing_trader.data.yf_client.YFinanceClient.get_history(period="5d",
interval="30m")` and upserts them into the `stock_prices` hypertable
(unique on ticker/ts/interval per `db.models.StockPrice`).

Runnable standalone: `python scheduler/run_midday.py`.
"""
from __future__ import annotations

import contextlib
import datetime as dt

from swing_trader.db.base import session_scope
from swing_trader.db.models import PipelineRun, StockPrice, TickerStatus, TickerUniverse
from swing_trader.logging_setup import get_logger, setup_logging

logger = get_logger("scheduler.midday")
JOB_NAME = "midday"


@contextlib.contextmanager
def _tracked_run(session, job_name: str):
    run = PipelineRun(job_name=job_name, started_at=dt.datetime.utcnow(), records_processed=0)
    session.add(run)
    session.flush()
    try:
        yield run
        if run.success is None:
            run.success = True
    except Exception as exc:
        run.success = False
        run.error_message = str(exc)[:2000]
        logger.exception("%s job failed", job_name)
        raise
    finally:
        run.finished_at = dt.datetime.utcnow()
        session.commit()


def _upsert_bar(session, ticker: str, ts: dt.datetime, row) -> None:
    existing = (
        session.query(StockPrice)
        .filter(StockPrice.ticker == ticker, StockPrice.ts == ts, StockPrice.interval == "30m")
        .first()
    )
    if existing is None:
        existing = StockPrice(ticker=ticker, ts=ts, interval="30m")
        session.add(existing)
    existing.open = float(row.get("Open"))
    existing.high = float(row.get("High"))
    existing.low = float(row.get("Low"))
    existing.close = float(row.get("Close"))
    existing.adj_close = float(row.get("Adj Close", row.get("Close")))
    existing.volume = float(row.get("Volume", 0.0) or 0.0)
    existing.source = "yfinance"


def main() -> None:
    setup_logging()
    logger.info("Starting %s job", JOB_NAME)
    try:
        with session_scope() as db:
            with _tracked_run(db, JOB_NAME) as run:
                tickers = [
                    t.ticker
                    for t in db.query(TickerUniverse).filter(TickerUniverse.status == TickerStatus.ACTIVE).all()
                ]
                if not tickers:
                    logger.info("No active tickers; nothing to do.")
                    return

                from swing_trader.data.yf_client import YFinanceClient

                client = YFinanceClient.instance()
                processed = 0
                for ticker in tickers:
                    try:
                        df = client.get_history(ticker, period="5d", interval="30m")
                        if df is None or df.empty:
                            continue
                        for ts, row in df.iterrows():
                            py_ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                            _upsert_bar(db, ticker, py_ts, row)
                            processed += 1
                    except Exception as exc:
                        logger.warning("Midday refresh failed for %s: %s", ticker, exc)
                run.records_processed = processed
                logger.info("Upserted %d intraday bars", processed)
    except Exception:
        logger.exception("%s job aborted", JOB_NAME)
        raise SystemExit(1)
    logger.info("Finished %s job", JOB_NAME)


if __name__ == "__main__":
    main()

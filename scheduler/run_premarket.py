"""Pre-market job (SRS Section 6.3 / com.swingtrader.premarket, 08:00 ET).

Fetches pre-market % change (from the prior close) for every active ticker
using the already-built `swing_trader.data.yf_client.YFinanceClient`. Every
run is recorded as a `PipelineRun` row (job_name="premarket") so
`swing_trader.cli system-health` and the dashboard's System Health panel can
report success rates -- the row is always closed out (finished_at/success
set) even if the job body raises, via the `_tracked_run` context manager.

Runnable standalone: `python scheduler/run_premarket.py`.
"""
from __future__ import annotations

import contextlib
import datetime as dt

from swing_trader.db.base import session_scope
from swing_trader.db.models import PipelineRun, TickerStatus, TickerUniverse
from swing_trader.logging_setup import get_logger, setup_logging

logger = get_logger("scheduler.premarket")
JOB_NAME = "premarket"


@contextlib.contextmanager
def _tracked_run(session, job_name: str):
    """Open a PipelineRun row, always close it out (success/finished_at) on
    exit -- even on exception -- and re-raise so `main()` can log+exit(1)."""
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
                        hist = client.get_history(ticker, period="1d", prepost=True)
                        if hist is not None and not hist.empty and hist["Open"].iloc[0]:
                            pct_change = (hist["Close"].iloc[-1] / hist["Open"].iloc[0] - 1) * 100
                            logger.info("%s pre-market change: %.2f%%", ticker, pct_change)
                        processed += 1
                    except Exception as exc:
                        logger.warning("Pre-market fetch failed for %s: %s", ticker, exc)
                run.records_processed = processed
    except Exception:
        logger.exception("%s job aborted", JOB_NAME)
        raise SystemExit(1)
    logger.info("Finished %s job", JOB_NAME)


if __name__ == "__main__":
    main()

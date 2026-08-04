"""End-of-day job (SRS Section 6.3 / com.swingtrader.eod, 16:35 ET).

Calls `swing_trader.portfolio.backfill.run_daily_incremental_update` for
every active ticker (defensive import -- that package is being built
concurrently). If it isn't available yet, the job logs a clear "not yet
available" warning and closes out its PipelineRun row as a no-op rather
than crashing.

Runnable standalone: `python scheduler/run_eod.py`.
"""
from __future__ import annotations

import contextlib
import datetime as dt

from swing_trader.db.base import session_scope
from swing_trader.db.models import PipelineRun, TickerStatus, TickerUniverse
from swing_trader.logging_setup import get_logger, setup_logging

logger = get_logger("scheduler.eod")
JOB_NAME = "eod"


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

                try:
                    from swing_trader.portfolio.backfill import run_daily_incremental_update
                except ImportError as exc:
                    logger.warning(
                        "swing_trader.portfolio.backfill.run_daily_incremental_update not yet "
                        "available (%s); skipping EOD update.",
                        exc,
                    )
                    return

                # NOTE: run_daily_incremental_update's real signature is
                # (tickers: list[str]) -- it batches all active tickers in one
                # call and opens its own session internally, rather than taking
                # a (session, ticker) pair per call.
                try:
                    run_daily_incremental_update(tickers)
                    run.records_processed = len(tickers)
                except Exception as exc:
                    logger.warning("EOD incremental update failed: %s", exc)
    except Exception:
        logger.exception("%s job aborted", JOB_NAME)
        raise SystemExit(1)
    logger.info("Finished %s job", JOB_NAME)


if __name__ == "__main__":
    main()

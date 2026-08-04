"""Prediction job (SRS Section 6.3 / com.swingtrader.predict, 18:30 ET).

Pipeline: run the daily self-tuning model pipeline
(`swing_trader.models.pipeline.run_daily_self_tuning_pipeline`), generate a
fresh signal rating per active ticker (`swing_trader.signals.generator.
generate_signal`), then fire alerts for newly-rated Strong Buy / Buy
tickers (`swing_trader.notify.engine.alert_signal_change`). Every stage is
a defensive import -- those packages are being built concurrently -- so a
missing stage is logged and skipped rather than crashing the whole job.

Runnable standalone: `python scheduler/run_predict.py`.
"""
from __future__ import annotations

import contextlib
import datetime as dt

from swing_trader.db.base import session_scope
from swing_trader.db.models import PipelineRun, Rating, TickerStatus, TickerUniverse
from swing_trader.logging_setup import get_logger, setup_logging

logger = get_logger("scheduler.predict")
JOB_NAME = "predict"


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

                # Stage 1: daily self-tuning model pipeline.
                try:
                    from swing_trader.models.pipeline import (
                        build_context_from_db,
                        run_daily_self_tuning_pipeline,
                    )

                    context = build_context_from_db(db, tickers)
                    if context is None:
                        logger.warning("Not enough feature/price history to build a training context; skipping pipeline")
                    else:
                        run_daily_self_tuning_pipeline(tickers, context)
                except ImportError as exc:
                    logger.warning("models.pipeline not yet available: %s", exc)
                except Exception as exc:
                    logger.error("Prediction pipeline failed: %s", exc)

                # Stage 2: generate a fresh signal rating per ticker.
                new_ratings: dict[str, object] = {}
                try:
                    from swing_trader.signals.generator import generate_signal
                except ImportError as exc:
                    logger.warning("signals.generator.generate_signal not yet available: %s", exc)
                    generate_signal = None  # type: ignore[assignment]

                processed = 0
                today = dt.date.today()
                if generate_signal is not None:
                    for ticker in tickers:
                        try:
                            rating = generate_signal(db, ticker, today)
                            new_ratings[ticker] = rating
                            processed += 1
                        except Exception as exc:
                            logger.warning("Signal generation failed for %s: %s", ticker, exc)
                run.records_processed = processed

                # Stage 3: alert on new Strong Buy / Buy ratings.
                try:
                    from swing_trader.notify.engine import alert_signal_change
                except ImportError as exc:
                    logger.warning("notify.engine.alert_signal_change not yet available: %s", exc)
                    alert_signal_change = None  # type: ignore[assignment]

                if alert_signal_change is not None:
                    for ticker, rating in new_ratings.items():
                        rating_value = getattr(rating, "rating", rating)
                        if rating_value in (Rating.STRONG_BUY, Rating.BUY, "Strong Buy", "Buy"):
                            try:
                                alert_signal_change(db, ticker, rating_value)
                            except Exception as exc:
                                logger.warning("Alert dispatch failed for %s: %s", ticker, exc)
    except Exception:
        logger.exception("%s job aborted", JOB_NAME)
        raise SystemExit(1)
    logger.info("Finished %s job", JOB_NAME)


if __name__ == "__main__":
    main()

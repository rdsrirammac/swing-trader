"""Nightly DB backup job (SRS Section 6.3 / com.swingtrader.backup, 02:00 ET daily).

Runs `pg_dump` against the configured database (see
`swing_trader.config.get_settings().raw["database"]`) and writes a
timestamped SQL dump to `BASE/backups/`, pruning dumps older than 30 days.
`pg_dump` may not be installed in a dev sandbox -- failures are caught,
logged as a warning, and recorded on the PipelineRun row; they never crash
the process.

Runnable standalone: `python scheduler/run_backup.py`.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import glob
import os
import subprocess
import time
from pathlib import Path

from swing_trader.config import REPO_ROOT, get_settings
from swing_trader.db.base import session_scope
from swing_trader.db.models import PipelineRun
from swing_trader.logging_setup import get_logger, setup_logging

logger = get_logger("scheduler.backup")
JOB_NAME = "backup"
RETENTION_DAYS = 30


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
        # Note: deliberately not re-raised -- a failed pg_dump (e.g. missing
        # binary in a dev sandbox) should not crash the scheduler process.
    finally:
        run.finished_at = dt.datetime.utcnow()
        session.commit()


def _run_pg_dump(backups_dir: Path) -> str | None:
    settings = get_settings()
    db_cfg = settings.raw.get("database", {})
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = backups_dir / f"swing_trader_{timestamp}.sql"

    env = os.environ.copy()
    if db_cfg.get("password"):
        env["PGPASSWORD"] = str(db_cfg["password"])

    cmd = [
        "pg_dump",
        "-h", str(db_cfg.get("host", "localhost")),
        "-p", str(db_cfg.get("port", 5432)),
        "-U", str(db_cfg.get("user", "swing")),
        "-d", str(db_cfg.get("name", "swing_trader")),
        "-f", str(out_path),
    ]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.warning("pg_dump exited %s: %s", result.returncode, (result.stderr or "")[:2000])
            return None
        logger.info("Backup written to %s", out_path)
        return str(out_path)
    except FileNotFoundError:
        logger.warning("pg_dump not found on PATH; skipping backup (install the postgresql-client package).")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("pg_dump timed out after 600s; skipping backup.")
        return None
    except Exception as exc:
        logger.warning("pg_dump failed: %s", exc)
        return None


def _prune_old_backups(backups_dir: Path) -> int:
    cutoff = time.time() - RETENTION_DAYS * 86400
    pruned = 0
    for path in glob.glob(str(backups_dir / "swing_trader_*.sql")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                pruned += 1
                logger.info("Pruned old backup %s", path)
        except OSError as exc:
            logger.warning("Failed to prune %s: %s", path, exc)
    return pruned


def main() -> None:
    setup_logging()
    logger.info("Starting %s job", JOB_NAME)
    backups_dir = REPO_ROOT / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    with session_scope() as db:
        with _tracked_run(db, JOB_NAME) as run:
            out_path = _run_pg_dump(backups_dir)
            run.records_processed = 1 if out_path else 0
            run.success = out_path is not None

            pruned = _prune_old_backups(backups_dir)
            logger.info("Pruned %d backup(s) older than %d days", pruned, RETENTION_DAYS)

    logger.info("Finished %s job", JOB_NAME)


if __name__ == "__main__":
    main()

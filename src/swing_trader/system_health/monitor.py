"""System health & monitoring helpers (SRS Section 7.8).

Used by `swing_trader.cli system-health`, the dashboard's System Health
panel, and (for `record_api_call`) by data-collection code that wants to
track provider rate-limit usage in the `api_rate_limit_log` table.
"""
from __future__ import annotations

import datetime as dt
import shutil
from typing import Any

from sqlalchemy.orm import Session

from swing_trader.db.models import ApiRateLimitLog, PipelineRun
from swing_trader.logging_setup import get_logger

logger = get_logger("system_health.monitor")

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]


def pipeline_success_rate(session: Session, job_name: str | None = None, days: int = 30) -> float:
    """Fraction (0.0-1.0) of `PipelineRun` rows with `success=True` in the
    trailing `days` window, optionally filtered to a single `job_name`.

    Runs that are still in-flight (`success is None`) are excluded from both
    numerator and denominator. Returns 0.0 if no completed runs exist.
    """
    since = dt.datetime.utcnow() - dt.timedelta(days=days)
    query = session.query(PipelineRun).filter(PipelineRun.started_at >= since)
    if job_name:
        query = query.filter(PipelineRun.job_name == job_name)
    runs = query.all()

    completed = [r for r in runs if r.success is not None]
    if not completed:
        return 0.0
    successes = sum(1 for r in completed if r.success)
    return successes / len(completed)


def record_api_call(
    session: Session,
    provider: str,
    calls_made: int = 1,
    limit_per_window: int | None = None,
    window_seconds: int | None = None,
) -> None:
    """Write a row to `api_rate_limit_log` for the given `provider`.

    Callers (data-collection clients) should call this once per outbound API
    request/batch so `api_rate_limit_status` can compute usage windows.
    """
    log = ApiRateLimitLog(
        provider=provider,
        ts=dt.datetime.utcnow(),
        calls_made=calls_made,
        limit_per_window=limit_per_window,
        window_seconds=window_seconds,
    )
    session.add(log)
    session.flush()


def api_rate_limit_status(session: Session, provider: str, window_seconds: int = 60) -> dict[str, Any]:
    """Summarize recent API usage for `provider` within the last `window_seconds`.

    Returns `{"calls_in_window", "limit", "pct_used", "near_limit"}`. `limit`
    and `pct_used` are None if no `limit_per_window` has been recorded for
    this provider recently. `near_limit` is True once usage reaches 80% of
    the limit.
    """
    since = dt.datetime.utcnow() - dt.timedelta(seconds=window_seconds)
    rows = (
        session.query(ApiRateLimitLog)
        .filter(ApiRateLimitLog.provider == provider, ApiRateLimitLog.ts >= since)
        .all()
    )
    calls_in_window = sum(r.calls_made or 0 for r in rows)
    limit = next((r.limit_per_window for r in rows if r.limit_per_window), None)
    pct_used = (calls_in_window / limit) if limit else None
    return {
        "calls_in_window": calls_in_window,
        "limit": limit,
        "pct_used": pct_used,
        "near_limit": bool(pct_used is not None and pct_used >= 0.8),
    }


def system_resource_snapshot() -> dict[str, float | None]:
    """Return `{"cpu_pct", "memory_pct", "disk_free_gb"}` for the host.

    Prefers `psutil` (see requirements.txt) for CPU/memory/disk. Falls back
    to stdlib `shutil.disk_usage` (always available) for disk space if
    `psutil` isn't installed. Any metric that can't be determined is
    returned as None rather than raising -- this feeds a best-effort
    dashboard panel that must never crash the page.
    """
    result: dict[str, float | None] = {"cpu_pct": None, "memory_pct": None, "disk_free_gb": None}

    if psutil is not None:
        try:
            result["cpu_pct"] = psutil.cpu_percent(interval=0.1)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("psutil.cpu_percent failed: %s", exc)
        try:
            result["memory_pct"] = psutil.virtual_memory().percent
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("psutil.virtual_memory failed: %s", exc)
        try:
            disk = psutil.disk_usage("/")
            result["disk_free_gb"] = disk.free / (1024**3)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("psutil.disk_usage failed: %s", exc)
    else:
        logger.info("psutil not installed; falling back to shutil.disk_usage for disk_free_gb only.")
        try:
            usage = shutil.disk_usage("/")
            result["disk_free_gb"] = usage.free / (1024**3)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("shutil.disk_usage failed: %s", exc)

    return result

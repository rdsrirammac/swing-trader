"""Alert dispatch orchestrator (SRS 3.10, AL-001..004) tying together
`rules_engine`, `macos`, and `email_notifier`.

`dispatch_alert` is the single entry point every other module should call
to raise an alert; it handles persistence (`Alert` row), the suppression
decision, fan-out to enabled channels, and a `NotificationLog` row per
channel attempted.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import Alert, AlertChannel, NotificationLog
from swing_trader.logging_setup import get_logger
from swing_trader.notify import macos, rules_engine
from swing_trader.notify.email_notifier import send_email

logger = get_logger("notify.engine")


def dispatch_alert(
    session: Session,
    category: str,
    message: str,
    ticker: str | None = None,
    severity: str = "info",
    dedup_key: str | None = None,
    min_rating: str | None = None,
    actual_rating: str | None = None,
) -> Alert:
    """AL-001..004: create and (maybe) send an alert through all enabled channels.

    Always creates and persists an `Alert` row (audit trail) regardless of
    whether `rules_engine.should_alert` allows the alert through — the
    suppression decision only affects whether channels are actually
    notified, so the dashboard's alert history stays complete.

    `category` should be one of `AlertCategory`'s values (signal/risk/
    data/earnings); `dedup_key` defaults to
    `f"{category}:{ticker}:{message[:50]}"` if not supplied.
    """
    settings = get_settings()

    if dedup_key is None:
        dedup_key = f"{category}:{ticker}:{message[:50]}"

    alert = Alert(
        category=category,
        ticker=ticker,
        message=message,
        severity=severity,
        dedup_key=dedup_key,
    )
    session.add(alert)
    session.flush()

    allowed = rules_engine.should_alert(
        session,
        category=category,
        dedup_key=dedup_key,
        min_rating=min_rating,
        actual_rating=actual_rating,
        severity=severity,
    )
    if not allowed:
        logger.info("Alert %s suppressed by rules_engine (dedup_key=%s)", alert.id, dedup_key)
        return alert

    channels_cfg = settings.get("alerts.channels", {}) or {}
    title = f"[{category}] {ticker or ''}".strip()

    if channels_cfg.get("macos", False):
        success = macos.send_macos_notification(title=title, message=message)
        _log_notification(session, alert.id, AlertChannel.MACOS, success)

    if channels_cfg.get("email", False):
        success = send_email(subject=title or "swing-trader alert", body=message)
        _log_notification(session, alert.id, AlertChannel.EMAIL, success)

    if channels_cfg.get("sms", False):
        logger.info("SMS channel not implemented — see ROADMAP.md")
        _log_notification(
            session,
            alert.id,
            AlertChannel.SMS,
            False,
            error_message="SMS channel not implemented — see ROADMAP.md",
        )

    session.flush()
    return alert


def _log_notification(
    session: Session,
    alert_id: int,
    channel: AlertChannel,
    success: bool,
    error_message: str | None = None,
) -> None:
    log = NotificationLog(
        alert_id=alert_id,
        channel=channel,
        success=success,
        error_message=error_message,
    )
    session.add(log)


# --- SRS-category convenience wrappers -------------------------------------


def alert_signal_change(session: Session, ticker: str, new_rating: str) -> Alert:
    """AL-001: notify when a ticker's signal rating changes.

    Applies `alerts.min_rating_for_signal_alert` as the suppression
    threshold via `rules_engine`.
    """
    settings = get_settings()
    min_rating = settings.get("alerts.min_rating_for_signal_alert", "Buy")
    return dispatch_alert(
        session,
        category="signal",
        message=f"{ticker} rating changed to {new_rating}",
        ticker=ticker,
        severity="info",
        min_rating=min_rating,
        actual_rating=new_rating,
    )


def alert_stop_hit(session: Session, ticker: str, price: float) -> Alert:
    """AL-002 family: notify when a position's stop loss is hit. Always critical."""
    return dispatch_alert(
        session,
        category="risk",
        message=f"{ticker} stop loss hit at {price:.2f}",
        ticker=ticker,
        severity="critical",
    )


def alert_risk_breach(session: Session, kind: str, value: float, limit: float) -> Alert:
    """AL-002: notify on portfolio risk-limit breaches (drawdown, heat,
    sector concentration, correlation, etc). Always critical.
    """
    return dispatch_alert(
        session,
        category="risk",
        message=f"Risk breach ({kind}): value={value:.4f} exceeds limit={limit:.4f}",
        severity="critical",
    )


def alert_backfill_complete(session: Session, ticker: str, success: bool) -> Alert:
    """AL-003: notify when a ticker's auto-backfill job finishes."""
    status = "succeeded" if success else "failed"
    return dispatch_alert(
        session,
        category="data",
        message=f"Backfill for {ticker} {status}",
        ticker=ticker,
        severity="info" if success else "warning",
    )


def alert_earnings_warning(session: Session, ticker: str, days_until: int) -> Alert:
    """AL-004: notify at `earnings.warning_days` thresholds ahead of a
    ticker's earnings date. `dedup_key` includes `days_until` so each
    warning threshold (e.g. 10/5/1 days) fires independently rather than
    being deduped against each other.
    """
    return dispatch_alert(
        session,
        category="earnings",
        message=f"{ticker} reports earnings in {days_until} day(s)",
        ticker=ticker,
        severity="warning" if days_until <= 1 else "info",
        dedup_key=f"earnings:{ticker}:{days_until}",
    )

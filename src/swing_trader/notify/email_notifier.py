"""Email notifications (SRS 3.10 AL-005; 3.13 PA-005 daily/weekly reports).

Uses stdlib `smtplib` + `email.mime.text` only (no third-party mail
dependency). Missing SMTP credentials are treated as a soft failure: log
a warning and return False rather than raising, so a mis-/un-configured
mail account never crashes a caller.
"""
from __future__ import annotations

import datetime as dt
import smtplib
from email.mime.text import MIMEText

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.db.models import Holding, PositionStatus, SignalRating, Trade
from swing_trader.logging_setup import get_logger

logger = get_logger("notify.email_notifier")


def send_email(subject: str, body: str, to_address: str | None = None) -> bool:
    """AL-005: send a plain-text email via SMTP.

    Reads `email.smtp_host` / `email.smtp_port` / `email.username` /
    `email.password` / `email.to_address` from `get_settings().secret(...)`.
    Returns False (and logs a warning) if credentials are missing or the
    send fails for any reason — never raises.
    """
    settings = get_settings()
    smtp_host = settings.secret("email.smtp_host")
    smtp_port = settings.secret("email.smtp_port")
    username = settings.secret("email.username")
    password = settings.secret("email.password")
    recipient = to_address or settings.secret("email.to_address")

    if not smtp_host or not username or not password or not recipient:
        logger.warning("Email credentials not configured; skipping send of %r", subject)
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = recipient

    try:
        port = int(smtp_port) if smtp_port else 587
        with smtplib.SMTP(smtp_host, port, timeout=15) as server:
            server.starttls()
            server.login(username, password)
            server.sendmail(username, [recipient], msg.as_string())
        return True
    except Exception as e:
        logger.warning("Failed to send email %r: %s", subject, e)
        return False


def send_daily_summary_email(session: Session, portfolio_id: int) -> bool:
    """PA-005: send a daily plain-text summary of open positions, new signal
    ratings, and today's realized P&L.

    First-cut implementation: simple queries over `Holding`/`SignalRating`/
    `Trade`, plain-text body, no charts/HTML. Intended to be invoked once
    per day near market close by the scheduler.
    """
    today = dt.date.today()

    holdings = list(
        session.execute(
            select(Holding).where(
                Holding.portfolio_id == portfolio_id, Holding.status == PositionStatus.ACTIVE
            )
        )
        .scalars()
        .all()
    )
    new_signals = list(
        session.execute(select(SignalRating).where(SignalRating.as_of == today)).scalars().all()
    )
    closed_today = list(
        session.execute(
            select(Trade).where(
                Trade.portfolio_id == portfolio_id,
                Trade.exit_date.is_not(None),
                Trade.exit_date >= dt.datetime.combine(today, dt.time.min),
            )
        )
        .scalars()
        .all()
    )

    lines = [f"Daily Summary — {today.isoformat()}", ""]
    lines.append(f"Open positions ({len(holdings)}):")
    for h in holdings:
        lines.append(f"  {h.ticker}: {h.shares} sh @ {h.entry_price:.2f}, stop {h.stop_loss:.2f}")
    lines.append("")
    lines.append(f"New signal ratings today ({len(new_signals)}):")
    for s in new_signals:
        lines.append(f"  {s.ticker}: {s.rating.value} (score {s.score:.2f})")
    lines.append("")
    lines.append(f"Trades closed today ({len(closed_today)}):")
    total_pnl = 0.0
    for t in closed_today:
        pnl = t.realized_pnl or 0.0
        total_pnl += pnl
        reason = t.exit_reason.value if t.exit_reason else "?"
        lines.append(f"  {t.ticker}: pnl {pnl:.2f} ({reason})")
    lines.append(f"  Total realized P&L today: {total_pnl:.2f}")

    body = "\n".join(lines)
    return send_email(f"[swing-trader] Daily Summary {today.isoformat()}", body)


def send_weekly_report_email(session: Session, portfolio_id: int) -> bool:
    """PA-005: send a weekly plain-text performance report.

    First-cut implementation: aggregates trades closed in the trailing 7
    days (win rate, avg R-multiple, total realized P&L) plus a snapshot of
    currently open positions.
    """
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=7)

    trades = list(
        session.execute(
            select(Trade).where(
                Trade.portfolio_id == portfolio_id,
                Trade.exit_date.is_not(None),
                Trade.exit_date >= start,
                Trade.exit_date <= end,
            )
        )
        .scalars()
        .all()
    )
    holdings = list(
        session.execute(
            select(Holding).where(
                Holding.portfolio_id == portfolio_id, Holding.status == PositionStatus.ACTIVE
            )
        )
        .scalars()
        .all()
    )

    n = len(trades)
    wins = sum(1 for t in trades if (t.realized_pnl or 0.0) > 0)
    total_pnl = sum(t.realized_pnl or 0.0 for t in trades)
    r_values = [t.realized_r_multiple for t in trades if t.realized_r_multiple is not None]
    avg_r = (sum(r_values) / len(r_values)) if r_values else None

    lines = [
        f"Weekly Report — {start.date().isoformat()} to {end.date().isoformat()}",
        "",
        f"Trades closed: {n}",
        f"Win rate: {(wins / n * 100):.1f}%" if n else "Win rate: n/a",
        f"Avg R-multiple: {avg_r:.2f}" if avg_r is not None else "Avg R-multiple: n/a",
        f"Total realized P&L: {total_pnl:.2f}",
        "",
        f"Currently open positions: {len(holdings)}",
    ]
    for h in holdings:
        lines.append(f"  {h.ticker}: {h.shares} sh @ {h.entry_price:.2f}")

    body = "\n".join(lines)
    return send_email(f"[swing-trader] Weekly Report {end.date().isoformat()}", body)

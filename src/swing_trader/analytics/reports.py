"""Plain-text weekly / monthly performance reports (SRS 3.12, PA-005).

These functions build and return report bodies as plain strings only --
they contain NO email-sending or delivery logic. Delivery (email, Slack,
etc.) is owned by `swing_trader.notify.email_notifier` (a different,
concurrently-developed module) which this module deliberately does not
import; the caller is responsible for taking the returned string and
handing it to whatever delivery mechanism it wants.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.analytics.attribution import (
    attribution_by_month,
    attribution_by_regime,
    attribution_by_ticker,
)
from swing_trader.analytics.behavioral import behavioral_report
from swing_trader.analytics.performance import compute_performance_metrics, trade_journal_view
from swing_trader.db.models import Holding, ModelPerformance, PositionStatus, RegimeHistory, SignalRating
from swing_trader.logging_setup import get_logger

logger = get_logger("analytics.reports")


def _fmt_pct(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def _fmt_num(value: float | None, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if value is not None else "n/a"


def _fmt_money(value: float | None) -> str:
    return f"${value:,.2f}" if value is not None else "n/a"


def _open_positions_section(session: Session, portfolio_id: int) -> str:
    holdings = (
        session.execute(
            select(Holding).where(
                Holding.portfolio_id == portfolio_id, Holding.status != PositionStatus.CLOSED
            )
        )
        .scalars()
        .all()
    )
    lines = ["OPEN POSITIONS", "-" * 40]
    if not holdings:
        lines.append("(none)")
    for h in holdings:
        lines.append(
            f"{h.ticker:6s} shares={h.shares:>8.2f} entry={_fmt_money(h.entry_price)} "
            f"stop={_fmt_money(h.stop_loss)} tp1={_fmt_money(h.take_profit_1)} status={h.status.value}"
        )
    return "\n".join(lines)


def _new_signals_section(session: Session, start: dt.date, end: dt.date) -> str:
    signals = (
        session.execute(
            select(SignalRating).where(SignalRating.as_of >= start, SignalRating.as_of <= end)
        )
        .scalars()
        .all()
    )
    lines = [f"NEW SIGNALS ({start} to {end})", "-" * 40]
    if not signals:
        lines.append("(none)")
    for s in signals:
        lines.append(
            f"{s.ticker:6s} {s.rating.value:12s} score={_fmt_num(s.score)} "
            f"entry={_fmt_money(s.suggested_entry)} stop={_fmt_money(s.suggested_stop)}"
        )
    return "\n".join(lines)


def _pnl_summary_section(session: Session, portfolio_id: int, start: dt.date, end: dt.date) -> str:
    closed = trade_journal_view(session, portfolio_id, start=start, end=end)
    total_pnl = sum(t["pnl"] for t in closed if t["pnl"] is not None)
    wins = [t for t in closed if (t["pnl"] or 0) > 0]
    lines = [f"P&L SUMMARY ({start} to {end})", "-" * 40]
    lines.append(f"Closed trades: {len(closed)}  |  Wins: {len(wins)}  |  Total P&L: {_fmt_money(total_pnl)}")
    for t in closed:
        lines.append(
            f"{t['ticker']:6s} exit={t['exit_date']} pnl={_fmt_money(t['pnl'])} "
            f"R={_fmt_num(t['r_multiple'])} reason={t['exit_reason']}"
        )
    return "\n".join(lines)


def generate_weekly_report(session: Session, portfolio_id: int) -> str:
    """PA-005: weekly report body -- open positions, new signals, P&L summary.

    Covers the trailing 7 days ending today. Returns a plain-text string;
    the caller decides how (or whether) to deliver it.
    """
    today = dt.date.today()
    start = today - dt.timedelta(days=7)

    sections = [
        f"SWING-TRADER WEEKLY REPORT — Portfolio #{portfolio_id}",
        f"Period: {start} to {today}",
        "=" * 60,
        "",
        _open_positions_section(session, portfolio_id),
        "",
        _new_signals_section(session, start, today),
        "",
        _pnl_summary_section(session, portfolio_id, start, today),
    ]
    return "\n".join(sections)


def _model_accuracy_section(session: Session) -> str:
    latest_per_model: dict[str, ModelPerformance] = {}
    rows = session.execute(select(ModelPerformance).order_by(ModelPerformance.as_of)).scalars().all()
    for row in rows:
        latest_per_model[row.model_version] = row  # keep the latest (rows ordered ascending)

    lines = ["MODEL ACCURACY", "-" * 40]
    if not latest_per_model:
        lines.append("(no model performance records)")
    for version, perf in latest_per_model.items():
        deployed = " [DEPLOYED]" if perf.deployed else ""
        lines.append(
            f"{version}{deployed}: MAPE={_fmt_num(perf.mape, 4)} "
            f"dir_acc={_fmt_pct(perf.directional_accuracy)} sharpe={_fmt_num(perf.sharpe_ratio)} "
            f"max_dd={_fmt_pct(perf.max_drawdown)} calmar={_fmt_num(perf.calmar_ratio)}"
        )
    return "\n".join(lines)


def _regime_summary_section(session: Session, start: dt.date, end: dt.date) -> str:
    rows = (
        session.execute(
            select(RegimeHistory)
            .where(RegimeHistory.ts >= start, RegimeHistory.ts <= end)
            .order_by(RegimeHistory.ts)
        )
        .scalars()
        .all()
    )
    lines = [f"REGIME SUMMARY ({start} to {end})", "-" * 40]
    if not rows:
        lines.append("(no regime history records)")
    else:
        counts: dict[str, int] = {}
        for r in rows:
            counts[r.regime.value] = counts.get(r.regime.value, 0) + 1
        for regime, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"{regime:20s} {count} day(s)")
        transitions = [r for r in rows if r.transition_flag]
        if transitions:
            lines.append(f"Regime transitions this period: {len(transitions)}")
            for t in transitions:
                lines.append(f"  {t.ts}: -> {t.regime.value} ({t.transition_reason or 'n/a'})")
    return "\n".join(lines)


def generate_monthly_report(session: Session, portfolio_id: int) -> str:
    """PA-005: monthly report body -- full performance, model accuracy, regime summary.

    Covers the trailing 30 days ending today. Returns a plain-text string;
    the caller decides how (or whether) to deliver it.
    """
    today = dt.date.today()
    start = today - dt.timedelta(days=30)

    metrics = compute_performance_metrics(session, portfolio_id)
    overall = metrics["overall"]
    win_loss = metrics["win_loss"]
    r_multiples = metrics["r_multiples"]
    time_metrics = metrics["time"]
    drawdown = metrics["drawdown"]

    behavior = behavioral_report(session, portfolio_id)

    perf_lines = [
        "FULL PERFORMANCE",
        "-" * 40,
        f"Total return: {_fmt_pct(overall['total_return'])}  CAGR: {_fmt_pct(overall['cagr'])}  "
        f"Sharpe: {_fmt_num(overall['sharpe'])}  Sortino: {_fmt_num(overall['sortino'])}  "
        f"Calmar: {_fmt_num(overall['calmar'])}",
        f"Win rate: {_fmt_pct(win_loss['win_rate'])}  Avg win: {_fmt_money(win_loss['avg_win'])}  "
        f"Avg loss: {_fmt_money(win_loss['avg_loss'])}  Profit factor: {_fmt_num(win_loss['profit_factor'])}",
        f"Avg R: {_fmt_num(r_multiples['avg_r'])}  Max R: {_fmt_num(r_multiples['max_r'])}",
        f"Avg hold: {_fmt_num(time_metrics['avg_hold_time_days'], 1)} days  "
        f"Time in market: {_fmt_pct(time_metrics['time_in_market_pct'])}",
        f"Max drawdown: {_fmt_pct(drawdown['max_dd'])}  DD duration: {drawdown['dd_duration_days']} days",
        "",
        "BEHAVIORAL DIAGNOSTICS",
        "-" * 40,
        f"Early exit rate: {_fmt_pct(behavior['early_exit_rate'])}  "
        f"Stop violation rate: {_fmt_pct(behavior['stop_violation_rate'])}",
        f"Revenge trading score: {_fmt_pct(behavior['revenge_trading_score'])}  "
        f"Overtrading ratio: {_fmt_num(behavior['overtrading_ratio'])}",
        "",
        "ATTRIBUTION -- TOP TICKERS",
        "-" * 40,
    ]

    ticker_attr = attribution_by_ticker(session, portfolio_id)
    if ticker_attr.empty:
        perf_lines.append("(no closed trades)")
    else:
        for _, row in ticker_attr.head(10).iterrows():
            perf_lines.append(
                f"{row['ticker']:6s} trades={int(row['trade_count'])} "
                f"total_pnl={_fmt_money(row['total_pnl'])} win_rate={_fmt_pct(row['win_rate'])}"
            )

    perf_lines.append("")
    perf_lines.append("ATTRIBUTION -- BY REGIME")
    perf_lines.append("-" * 40)
    regime_attr = attribution_by_regime(session, portfolio_id)
    if regime_attr.empty:
        perf_lines.append("(no closed trades)")
    else:
        for _, row in regime_attr.iterrows():
            perf_lines.append(
                f"{row['regime']:16s} trades={int(row['trade_count'])} "
                f"total_pnl={_fmt_money(row['total_pnl'])} win_rate={_fmt_pct(row['win_rate'])}"
            )

    perf_lines.append("")
    perf_lines.append("ATTRIBUTION -- SEASONALITY (BY MONTH)")
    perf_lines.append("-" * 40)
    month_attr = attribution_by_month(session, portfolio_id)
    if month_attr.empty:
        perf_lines.append("(no closed trades)")
    else:
        for _, row in month_attr.iterrows():
            perf_lines.append(
                f"{str(row['month']):12s} trades={int(row['trade_count'])} total_pnl={_fmt_money(row['total_pnl'])}"
            )

    sections = [
        f"SWING-TRADER MONTHLY REPORT — Portfolio #{portfolio_id}",
        f"Period: {start} to {today}",
        "=" * 60,
        "",
        "\n".join(perf_lines),
        "",
        _model_accuracy_section(session),
        "",
        _regime_summary_section(session, start, today),
    ]
    return "\n".join(sections)

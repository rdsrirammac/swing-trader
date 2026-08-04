"""Streamlit dashboard main entrypoint (SRS 3.15 DV-001, DV-004).

Run with: `streamlit run src/swing_trader/dashboard/app.py` (or `make dashboard`).

Layout is a single page of stacked sections per DV-001: Portfolio Summary,
Market Regime, Alerts, Active Positions, Watchlist/Signals, Model
Performance, Sector Rotation, Correlation, Backtest Results, Trade Journal,
System Health. Additional focused pages live in `dashboard/pages/`
(Streamlit native multipage convention): Ticker Detail, Backtest, Trade
Journal.

Every section is wrapped in its own try/except so a missing sibling package
(signals/portfolio/models/backtest/analytics/notify/calendar_data/execution)
or an empty/unreachable database never crashes the whole page -- it just
shows an `st.info(...)` placeholder for that one panel.
"""
from __future__ import annotations

import datetime as dt
import importlib

import pandas as pd
import streamlit as st

from swing_trader.db.base import session_scope
from swing_trader.db.models import (
    Alert,
    Holding,
    Portfolio,
    PositionStatus,
    SignalRating,
    TickerUniverse,
    Trade,
    WatchlistItem,
)
from swing_trader.logging_setup import get_logger, setup_logging

try:
    from swing_trader.dashboard import charts
except ImportError:
    charts = None  # type: ignore[assignment]

setup_logging()
logger = get_logger("dashboard.app")

st.set_page_config(page_title="Swing Trader Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optional_import(module_path: str):
    """Best-effort import of a sibling package that may not exist yet."""
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        logger.info("Optional module %s not available yet: %s", module_path, exc)
        return None


def _not_available(msg: str) -> None:
    st.info(f"Not yet available: {msg}.")


def _get_default_portfolio(db) -> Portfolio | None:
    return db.query(Portfolio).order_by(Portfolio.id).first()


# ---------------------------------------------------------------------------
# Header: title + last-updated timestamp / refresh (DV-004)
# ---------------------------------------------------------------------------

def _render_header() -> None:
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.title("Swing Trader Dashboard")

    autorefresh_mod = _optional_import("streamlit_autorefresh")
    with col2:
        if autorefresh_mod is not None and hasattr(autorefresh_mod, "st_autorefresh"):
            autorefresh_mod.st_autorefresh(interval=60_000, key="dashboard_autorefresh")
            st.session_state["last_updated"] = dt.datetime.now()
            st.caption("Auto-refreshing every 60s")
        else:
            if st.button("Refresh"):
                st.session_state["last_updated"] = dt.datetime.now()
                st.rerun()

    if "last_updated" not in st.session_state:
        st.session_state["last_updated"] = dt.datetime.now()
    with col3:
        st.caption(f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")


# ---------------------------------------------------------------------------
# Portfolio Summary
# ---------------------------------------------------------------------------

def _render_portfolio_summary() -> None:
    st.header("Portfolio Summary")
    try:
        with session_scope() as db:
            portfolio = _get_default_portfolio(db)
            if portfolio is None:
                st.info(
                    "No portfolio yet. Create one: "
                    "`python -m swing_trader.cli init-portfolio <NAME>`."
                )
                return

            holdings = (
                db.query(Holding)
                .filter(Holding.portfolio_id == portfolio.id, Holding.status != PositionStatus.CLOSED)
                .all()
            )
            positions_value = sum((h.shares * h.entry_price) for h in holdings)
            total_value = portfolio.cash_balance + positions_value

            manager = _optional_import("swing_trader.portfolio.manager")
            heat_pct = None
            daily_pnl = None
            if manager is not None and hasattr(manager, "portfolio_summary"):
                try:
                    summary = manager.portfolio_summary(db, portfolio.id)
                    if isinstance(summary, dict):
                        heat_pct = summary.get("heat_pct")
                        daily_pnl = summary.get("daily_pnl")
                        total_value = summary.get("total_value", total_value)
                except Exception as exc:
                    logger.warning("portfolio_summary() failed: %s", exc)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Value", f"${total_value:,.2f}")
            c1.metric("Cash", f"${portfolio.cash_balance:,.2f}")
            c2.metric("Daily P&L", f"${daily_pnl:,.2f}" if daily_pnl is not None else "N/A")
            c2.metric("Open Positions", str(len(holdings)))

            with c3:
                st.caption("Portfolio Heat")
                if heat_pct is None:
                    st.write("N/A")
                    _not_available("portfolio heat (swing_trader.portfolio.manager.portfolio_summary)")
                else:
                    limit = portfolio.max_portfolio_heat_pct
                    color = "green" if heat_pct < 0.5 * limit else ("orange" if heat_pct < limit else "red")
                    st.markdown(f"<h2 style='color:{color}'>{heat_pct * 100:.1f}%</h2>", unsafe_allow_html=True)
                    st.caption(f"limit {limit * 100:.0f}%")
            with c4:
                st.caption("Mode")
                st.write("Paper" if portfolio.is_paper else "Live")
    except Exception as exc:
        logger.exception("Portfolio summary panel failed")
        st.error(f"Could not load portfolio summary: {exc}")


# ---------------------------------------------------------------------------
# Market Regime
# ---------------------------------------------------------------------------

def _render_market_regime() -> None:
    st.header("Market Regime")
    try:
        from swing_trader.db.models import RegimeHistory

        with session_scope() as db:
            latest = db.query(RegimeHistory).order_by(RegimeHistory.ts.desc()).first()
            if latest is None:
                st.info("No regime history yet. It's populated by the market regime detector job.")
                return
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Regime", latest.regime.value.replace("_", " ").title())
            c2.metric("VIX", f"{latest.vix:.1f}" if latest.vix is not None else "N/A")
            c3.metric("SPY ADX", f"{latest.spy_adx:.1f}" if latest.spy_adx is not None else "N/A")
            c4.metric(
                "Sector Breadth", f"{latest.sector_breadth_pct:.0f}%" if latest.sector_breadth_pct is not None else "N/A"
            )
            if latest.transition_flag:
                st.warning(f"Regime transition detected: {latest.transition_reason or 'see logs'}")
    except Exception as exc:
        logger.exception("Market regime panel failed")
        st.error(f"Could not load market regime: {exc}")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def _render_alerts() -> None:
    st.header("Recent Alerts")
    try:
        with session_scope() as db:
            alerts = db.query(Alert).order_by(Alert.created_at.desc()).limit(10).all()
            if not alerts:
                st.info("No alerts yet.")
                return
            df = pd.DataFrame(
                [
                    {
                        "time": a.created_at, "category": a.category.value, "ticker": a.ticker,
                        "severity": a.severity, "message": a.message,
                    }
                    for a in alerts
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        logger.exception("Alerts panel failed")
        st.error(f"Could not load alerts: {exc}")


# ---------------------------------------------------------------------------
# Active Positions
# ---------------------------------------------------------------------------

def _render_active_positions() -> None:
    st.header("Active Positions")
    try:
        with session_scope() as db:
            portfolio = _get_default_portfolio(db)
            if portfolio is None:
                st.info("No portfolio yet.")
                return
            holdings = (
                db.query(Holding)
                .filter(Holding.portfolio_id == portfolio.id, Holding.status != PositionStatus.CLOSED)
                .all()
            )
            if not holdings:
                st.info("No active positions.")
                return
            df = pd.DataFrame(
                [
                    {
                        "ticker": h.ticker, "shares": h.shares, "entry_price": h.entry_price,
                        "entry_date": h.entry_date, "stop_loss": h.stop_loss,
                        "take_profit_1": h.take_profit_1, "take_profit_2": h.take_profit_2,
                        "trailing_stop": h.trailing_stop, "status": h.status.value,
                    }
                    for h in holdings
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        logger.exception("Active positions panel failed")
        st.error(f"Could not load active positions: {exc}")


# ---------------------------------------------------------------------------
# Watchlist / Signals
# ---------------------------------------------------------------------------

def _render_watchlist_signals() -> None:
    st.header("Watchlist & Signals")
    col1, col2 = st.columns(2)
    try:
        with session_scope() as db:
            portfolio = _get_default_portfolio(db)
            with col1:
                st.subheader("Watchlist")
                if portfolio is None:
                    st.info("No portfolio yet.")
                else:
                    items = db.query(WatchlistItem).filter(WatchlistItem.portfolio_id == portfolio.id).all()
                    if not items:
                        st.info("Watchlist is empty.")
                    else:
                        df = pd.DataFrame(
                            [{"ticker": i.ticker, "condition": i.trigger_condition, "triggered": i.triggered} for i in items]
                        )
                        st.dataframe(df, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Recent Signal Ratings")
                signals = db.query(SignalRating).order_by(SignalRating.as_of.desc()).limit(20).all()
                if not signals:
                    st.info("No signal ratings yet.")
                else:
                    df = pd.DataFrame(
                        [
                            {
                                "ticker": s.ticker, "as_of": s.as_of, "rating": s.rating.value, "score": s.score,
                                "entry": s.suggested_entry, "stop": s.suggested_stop,
                            }
                            for s in signals
                        ]
                    )
                    st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        logger.exception("Watchlist/signals panel failed")
        st.error(f"Could not load watchlist/signals: {exc}")


# ---------------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------------

def _render_model_performance() -> None:
    st.header("Model Performance")
    performance_mod = _optional_import("swing_trader.analytics.performance")
    try:
        with session_scope() as db:
            if performance_mod is not None and hasattr(performance_mod, "summarize_model_performance"):
                try:
                    summary = performance_mod.summarize_model_performance(db)
                    if isinstance(summary, dict):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Win Rate", f"{summary.get('win_rate', 0) * 100:.1f}%")
                        c2.metric("Avg R", f"{summary.get('avg_r_multiple', 0):.2f}")
                        c3.metric("Sharpe", f"{summary.get('sharpe_ratio', 0):.2f}")
                        return
                except Exception as exc:
                    logger.warning("summarize_model_performance() failed: %s", exc)

            _not_available("model performance (swing_trader.analytics.performance); showing basic trade stats instead")
            closed = db.query(Trade).filter(Trade.exit_date.isnot(None)).all()
            if not closed:
                st.info("No closed trades yet to compute win rate / avg R.")
                return
            wins = [t for t in closed if (t.realized_pnl or 0) > 0]
            win_rate = len(wins) / len(closed)
            r_values = [t.realized_r_multiple for t in closed if t.realized_r_multiple is not None]
            avg_r = sum(r_values) / len(r_values) if r_values else None
            c1, c2 = st.columns(2)
            c1.metric("Win Rate (trades)", f"{win_rate * 100:.1f}%")
            c2.metric("Avg R (trades)", f"{avg_r:.2f}" if avg_r is not None else "N/A")
    except Exception as exc:
        logger.exception("Model performance panel failed")
        st.error(f"Could not load model performance: {exc}")


# ---------------------------------------------------------------------------
# Sector Rotation
# ---------------------------------------------------------------------------

def _render_sector_rotation() -> None:
    st.header("Sector Rotation")
    analytics_mod = _optional_import("swing_trader.analytics.sector")
    try:
        with session_scope() as db:
            if analytics_mod is not None and hasattr(analytics_mod, "sector_rotation_summary"):
                try:
                    df = analytics_mod.sector_rotation_summary(db)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        return
                except Exception as exc:
                    logger.warning("sector_rotation_summary() failed: %s", exc)

            _not_available("sector rotation ranking (swing_trader.analytics.sector); showing universe breakdown instead")
            tickers = db.query(TickerUniverse).filter(TickerUniverse.sector.isnot(None)).all()
            if not tickers:
                st.info("No sector data yet in the ticker universe.")
                return
            df = pd.DataFrame([{"sector": t.sector, "ticker": t.ticker} for t in tickers])
            counts = df.groupby("sector").size().reset_index(name="ticker_count")
            st.dataframe(counts, use_container_width=True, hide_index=True)
    except Exception as exc:
        logger.exception("Sector rotation panel failed")
        st.error(f"Could not load sector rotation: {exc}")


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def _render_correlation() -> None:
    st.header("Correlation Summary")
    corr_mod = _optional_import("swing_trader.analytics.correlation")
    if corr_mod is None or not hasattr(corr_mod, "portfolio_correlation_summary"):
        _not_available("correlation summary (swing_trader.analytics.correlation.portfolio_correlation_summary)")
        return
    try:
        with session_scope() as db:
            portfolio = _get_default_portfolio(db)
            if portfolio is None:
                st.info("No portfolio yet.")
                return
            summary = corr_mod.portfolio_correlation_summary(db, portfolio.id)
            if isinstance(summary, pd.DataFrame):
                st.dataframe(summary, use_container_width=True)
            else:
                st.write(summary)
    except Exception as exc:
        logger.exception("Correlation panel failed")
        st.error(f"Could not load correlation summary: {exc}")


# ---------------------------------------------------------------------------
# Backtest Results
# ---------------------------------------------------------------------------

def _render_backtest_results() -> None:
    st.header("Backtest Results")
    engine_mod = _optional_import("swing_trader.backtest.engine")
    if engine_mod is None or not hasattr(engine_mod, "get_latest_backtest_summary"):
        _not_available(
            "backtest results summary (swing_trader.backtest.engine.get_latest_backtest_summary) -- "
            "use the Backtest page to run one on demand"
        )
        return
    try:
        with session_scope() as db:
            summary = engine_mod.get_latest_backtest_summary(db)
            if not summary:
                st.info("No backtest runs recorded yet.")
                return
            if isinstance(summary, dict):
                cols = st.columns(min(4, len(summary)))
                for i, (k, v) in enumerate(summary.items()):
                    cols[i % len(cols)].metric(k, str(v))
            else:
                st.write(summary)
    except Exception as exc:
        logger.exception("Backtest results panel failed")
        st.error(f"Could not load backtest results: {exc}")


# ---------------------------------------------------------------------------
# Trade Journal (last 5)
# ---------------------------------------------------------------------------

def _render_trade_journal() -> None:
    st.header("Trade Journal (Last 5)")
    try:
        with session_scope() as db:
            trades = db.query(Trade).order_by(Trade.entry_date.desc()).limit(5).all()
            if not trades:
                st.info("No trades recorded yet.")
                return
            df = pd.DataFrame(
                [
                    {
                        "ticker": t.ticker, "entry_date": t.entry_date, "entry_price": t.entry_price,
                        "shares": t.shares, "exit_date": t.exit_date, "exit_price": t.exit_price,
                        "realized_pnl": t.realized_pnl, "realized_r_multiple": t.realized_r_multiple,
                        "exit_reason": t.exit_reason.value if t.exit_reason else None,
                    }
                    for t in trades
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        logger.exception("Trade journal panel failed")
        st.error(f"Could not load trade journal: {exc}")


# ---------------------------------------------------------------------------
# System Health
# ---------------------------------------------------------------------------

def _render_system_health() -> None:
    st.header("System Health")
    try:
        from swing_trader.db.models import PipelineRun
        from swing_trader.system_health.monitor import pipeline_success_rate, system_resource_snapshot

        with session_scope() as db:
            jobs = sorted({row[0] for row in db.query(PipelineRun.job_name).distinct().all()})
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Pipeline Success Rate (30d)")
                if not jobs:
                    st.info("No pipeline runs recorded yet.")
                else:
                    df = pd.DataFrame(
                        [{"job": j, "success_rate_pct": pipeline_success_rate(db, j, days=30) * 100} for j in jobs]
                    )
                    st.dataframe(df, use_container_width=True, hide_index=True)
            with c2:
                st.subheader("Host Resources")
                snap = system_resource_snapshot()
                st.metric("CPU %", f"{snap['cpu_pct']:.1f}%" if snap["cpu_pct"] is not None else "N/A")
                st.metric("Memory %", f"{snap['memory_pct']:.1f}%" if snap["memory_pct"] is not None else "N/A")
                st.metric(
                    "Disk Free (GB)", f"{snap['disk_free_gb']:.1f}" if snap["disk_free_gb"] is not None else "N/A"
                )
    except Exception as exc:
        logger.exception("System health panel failed")
        st.error(f"Could not load system health: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _render_header()
    st.divider()
    _render_portfolio_summary()
    st.divider()
    _render_market_regime()
    st.divider()
    _render_alerts()
    st.divider()
    _render_active_positions()
    st.divider()
    _render_watchlist_signals()
    st.divider()
    _render_model_performance()
    st.divider()
    _render_sector_rotation()
    st.divider()
    _render_correlation()
    st.divider()
    _render_backtest_results()
    st.divider()
    _render_trade_journal()
    st.divider()
    _render_system_health()


main()

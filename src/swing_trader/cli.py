"""Admin CLI for swing-trader (NFR 4.6: "CLI for all admin operations").

Invoked as `python -m swing_trader.cli <command>` (see Makefile targets:
add-ticker, remove-ticker, list-portfolio, backfill, predict, backtest, ...)
or `python -m swing_trader` via `__main__.py`.

Every sibling package this module touches (portfolio, models, signals,
backtest, analytics, notify) may not exist yet -- other workstreams build
those concurrently -- so every cross-package import below is wrapped
defensively and, if unavailable, the command prints a clear "not yet
available" message instead of crashing (see `_optional_import` /
`_not_available`). Likewise every command catches exceptions at the
top level and prints an actionable suggested fix rather than a raw
traceback, per NFR 4.6.
"""
from __future__ import annotations

import datetime as dt
import importlib
from typing import Any

import click

from swing_trader.db.base import session_scope
from swing_trader.db.models import (
    ApiRateLimitLog,
    Holding,
    PipelineRun,
    Portfolio,
    PositionStatus,
    SignalRating,
    StockPrice,
    TickerStatus,
    TickerUniverse,
    WatchlistItem,
)
from swing_trader.logging_setup import get_logger, setup_logging

logger = get_logger("cli")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optional_import(module_path: str):
    """Best-effort import of a sibling package that may not exist yet.

    Returns the module, or None if it isn't importable. Callers must handle
    the None case (see `_not_available`) rather than let ImportError bubble
    up -- this is standard practice in this repo while workstreams build
    concurrently.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        logger.info("Optional module %s not available yet: %s", module_path, exc)
        return None


def _not_available(feature: str, module_path: str) -> None:
    click.secho(
        f"[not yet available] '{feature}' requires `{module_path}`, which hasn't "
        "been built yet by its owning workstream. Try again once that module lands.",
        fg="yellow",
    )


def _friendly_error(exc: Exception, suggestion: str) -> None:
    """Print an actionable error instead of a raw traceback (NFR 4.6)."""
    click.secho(f"Error: {exc}", fg="red", err=True)
    click.secho(f"Suggested fix: {suggestion}", fg="yellow", err=True)


def _resolve_portfolio(db, name: str | None) -> Portfolio | None:
    query = db.query(Portfolio)
    if name:
        return query.filter(Portfolio.name == name).first()
    return query.order_by(Portfolio.id).first()


def _run_backfill(ticker: str, verb: str) -> None:
    ticker = ticker.upper()
    backfill = _optional_import("swing_trader.portfolio.backfill")
    if backfill is None or not hasattr(backfill, "run_backfill"):
        _not_available("backfill", "swing_trader.portfolio.backfill.run_backfill")
        return
    try:
        with session_scope() as db:
            click.echo(f"{verb} {ticker}...")
            # NOTE: run_backfill's real signature is (ticker, session=None) --
            # opens its own session_scope() internally if none is passed.
            # We pass ours explicitly so it participates in the same
            # transaction as the surrounding CLI command.
            result = backfill.run_backfill(ticker, session=db)
            click.secho(f"Done: {result}", fg="green")
    except Exception as exc:
        _friendly_error(
            exc,
            "Check BackfillJob rows (phase/status/error_message) for the ticker, "
            "verify network access to data providers, then retry with "
            f"`python -m swing_trader.cli retry-backfill {ticker}`.",
        )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """swing-trader admin CLI (NFR 4.6)."""
    setup_logging()


# --- Ticker / portfolio management (PF, TB) --------------------------------

@cli.command("add-ticker")
@click.argument("ticker")
def add_ticker_cmd(ticker: str) -> None:
    """Add TICKER to the universe and kick off its backfill (TB-001/002)."""
    ticker = ticker.upper()
    manager = _optional_import("swing_trader.portfolio.manager")
    if manager is None or not hasattr(manager, "add_ticker"):
        _not_available("add-ticker", "swing_trader.portfolio.manager.add_ticker")
        return

    try:
        with session_scope() as db:
            click.echo(f"Adding {ticker} to ticker universe...")
            manager.add_ticker(db, ticker)
            click.secho(f"{ticker} added.", fg="green")
    except Exception as exc:
        _friendly_error(
            exc,
            "Check that the ticker symbol is valid and meets the liquidity/price "
            "filters in config/settings.yaml (ticker_universe.*).",
        )
        return

    _run_backfill(ticker, "Backfilling")


@cli.command("remove-ticker")
@click.argument("ticker")
def remove_ticker_cmd(ticker: str) -> None:
    """Remove TICKER from the active universe (PF)."""
    ticker = ticker.upper()
    manager = _optional_import("swing_trader.portfolio.manager")
    if manager is None or not hasattr(manager, "remove_ticker"):
        _not_available("remove-ticker", "swing_trader.portfolio.manager.remove_ticker")
        return
    try:
        with session_scope() as db:
            manager.remove_ticker(db, ticker)
            click.secho(f"{ticker} removed.", fg="green")
    except Exception as exc:
        _friendly_error(
            exc,
            "Check the ticker exists in ticker_universe and has no open holdings "
            "blocking removal.",
        )


@cli.command("list-portfolio")
@click.option("--name", default=None, help="Portfolio name (defaults to first portfolio found).")
def list_portfolio_cmd(name: str | None) -> None:
    """Print a formatted table of holdings + summary for a portfolio (PF)."""
    try:
        with session_scope() as db:
            portfolio = _resolve_portfolio(db, name)
            if portfolio is None:
                click.secho(
                    "No portfolio found. Create one first: "
                    "`python -m swing_trader.cli init-portfolio <NAME>`.",
                    fg="yellow",
                )
                return

            holdings = (
                db.query(Holding)
                .filter(Holding.portfolio_id == portfolio.id, Holding.status != PositionStatus.CLOSED)
                .all()
            )

            click.echo(
                f"\nPortfolio: {portfolio.name}  "
                f"(cash=${portfolio.cash_balance:,.2f}, paper={portfolio.is_paper})"
            )
            header = f"{'TICKER':<8}{'SHARES':>10}{'ENTRY':>10}{'STOP':>10}{'TP1':>10}{'TP2':>10}{'STATUS':>10}"
            click.echo("-" * len(header))
            click.echo(header)
            click.echo("-" * len(header))
            if not holdings:
                click.echo("(no active holdings)")
            for h in holdings:
                click.echo(
                    f"{h.ticker:<8}{h.shares:>10.2f}{h.entry_price:>10.2f}{h.stop_loss:>10.2f}"
                    f"{(h.take_profit_1 or 0):>10.2f}{(h.take_profit_2 or 0):>10.2f}{h.status.value:>10}"
                )

            manager = _optional_import("swing_trader.portfolio.manager")
            if manager is not None and hasattr(manager, "portfolio_summary"):
                try:
                    summary = manager.portfolio_summary(db, portfolio.id)
                    click.echo("\nSummary:")
                    if isinstance(summary, dict):
                        for k, v in summary.items():
                            click.echo(f"  {k}: {v}")
                    else:
                        click.echo(f"  {summary}")
                except Exception as exc:
                    click.secho(f"(portfolio_summary raised {exc})", fg="yellow")
            else:
                _not_available("portfolio summary", "swing_trader.portfolio.manager.portfolio_summary")
    except Exception as exc:
        _friendly_error(exc, "Check DB connectivity (`make infra-up`) and that `make init-db` has run.")


@cli.command("init-portfolio")
@click.argument("name")
@click.option("--cash", default=100000.0, type=float, help="Starting cash balance.")
def init_portfolio_cmd(name: str, cash: float) -> None:
    """Create a new portfolio (PF-001)."""
    manager = _optional_import("swing_trader.portfolio.manager")
    if manager is not None and hasattr(manager, "create_portfolio"):
        try:
            with session_scope() as db:
                manager.create_portfolio(db, name, cash_balance=cash)
                click.secho(f"Portfolio '{name}' created (cash=${cash:,.2f}).", fg="green")
        except Exception as exc:
            _friendly_error(exc, "Check that a portfolio with this name doesn't already exist.")
        return

    # Fallback: create directly via the ORM so init-portfolio still works
    # before swing_trader.portfolio.manager lands.
    _not_available("create_portfolio", "swing_trader.portfolio.manager.create_portfolio")
    click.echo("Falling back to a direct DB insert...")
    try:
        with session_scope() as db:
            existing = db.query(Portfolio).filter(Portfolio.name == name).first()
            if existing:
                click.secho(f"Portfolio '{name}' already exists.", fg="yellow")
                return
            db.add(Portfolio(name=name, cash_balance=cash))
            click.secho(f"Portfolio '{name}' created via DB fallback (cash=${cash:,.2f}).", fg="green")
    except Exception as exc:
        _friendly_error(exc, "Check DB connectivity (`make infra-up`) and that `make init-db` has run.")


# --- Backfill (TB) -----------------------------------------------------------

@cli.command("backfill")
@click.argument("ticker")
def backfill_cmd(ticker: str) -> None:
    """Run/re-run full backfill for TICKER (TB-002)."""
    _run_backfill(ticker, "Backfilling")


@cli.command("retry-backfill")
@click.argument("ticker")
def retry_backfill_cmd(ticker: str) -> None:
    """Manually retry a failed backfill (TB-004)."""
    _run_backfill(ticker, "Retrying backfill for")


# --- Prediction (PM) ---------------------------------------------------------

@cli.command("predict")
@click.option("--tickers", default=None, help="Comma-separated tickers (default: all active tickers).")
def predict_cmd(tickers: str | None) -> None:
    """Run the daily self-tuning prediction pipeline (PM)."""
    pipeline = _optional_import("swing_trader.models.pipeline")
    if pipeline is None or not hasattr(pipeline, "run_daily_self_tuning_pipeline"):
        _not_available("predict", "swing_trader.models.pipeline.run_daily_self_tuning_pipeline")
        return
    try:
        with session_scope() as db:
            if tickers:
                ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            else:
                ticker_list = [
                    t.ticker
                    for t in db.query(TickerUniverse).filter(TickerUniverse.status == TickerStatus.ACTIVE).all()
                ]
            if not ticker_list:
                click.secho("No active tickers found. Add some with `add-ticker` first.", fg="yellow")
                return
            if not hasattr(pipeline, "build_context_from_db"):
                _not_available("predict", "swing_trader.models.pipeline.build_context_from_db")
                return
            click.echo(f"Building training context for {len(ticker_list)} ticker(s)...")
            context = pipeline.build_context_from_db(db, ticker_list)
            if context is None:
                click.secho(
                    "Not enough stored feature/price history to build a training context. "
                    "Run `backfill` for these tickers first.",
                    fg="yellow",
                )
                return
            click.echo(f"Running prediction pipeline for {len(ticker_list)} ticker(s)...")
            pipeline.run_daily_self_tuning_pipeline(ticker_list, context)
            click.secho("Prediction pipeline complete.", fg="green")
    except Exception as exc:
        _friendly_error(
            exc,
            "Ensure features are backfilled for these tickers and that "
            "swing_trader.models.pipeline is fully implemented.",
        )


# --- Backtest (BT) -----------------------------------------------------------

@cli.command("backtest")
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option("--strategy", default="default", help="Strategy identifier passed to the backtest engine.")
def backtest_cmd(start: str, end: str, strategy: str) -> None:
    """Run a historical backtest over stored prices/signals (BT)."""
    engine = _optional_import("swing_trader.backtest.engine")
    if engine is None or not hasattr(engine, "run_backtest"):
        _not_available("backtest", "swing_trader.backtest.engine.run_backtest")
        return

    try:
        start_date = dt.date.fromisoformat(start)
        end_date = dt.date.fromisoformat(end)
    except ValueError as exc:
        _friendly_error(exc, "Dates must be in YYYY-MM-DD format.")
        return

    try:
        import pandas as pd

        with session_scope() as db:
            prices = (
                db.query(StockPrice)
                .filter(StockPrice.ts >= start_date, StockPrice.ts <= end_date, StockPrice.interval == "1d")
                .all()
            )
            if not prices:
                click.secho(
                    f"No stored 1d price data between {start} and {end}. Run backfill first.", fg="yellow"
                )
                return
            signals = (
                db.query(SignalRating)
                .filter(SignalRating.as_of >= start_date, SignalRating.as_of <= end_date)
                .all()
            )

            price_df = pd.DataFrame(
                [
                    {
                        "ticker": p.ticker, "ts": p.ts, "open": p.open, "high": p.high,
                        "low": p.low, "close": p.close, "volume": p.volume,
                    }
                    for p in prices
                ]
            )
            signals_df = pd.DataFrame(
                [
                    {
                        "ticker": s.ticker, "as_of": s.as_of, "score": s.score, "rating": s.rating.value,
                        "suggested_entry": s.suggested_entry, "suggested_stop": s.suggested_stop,
                        "suggested_target_1": s.suggested_target_1, "suggested_target_2": s.suggested_target_2,
                    }
                    for s in signals
                ]
            )

            click.echo(
                f"Running backtest strategy={strategy} from {start} to {end} "
                f"({len(price_df)} price rows, {len(signals_df)} signal rows)..."
            )
            result = engine.run_backtest(
                price_df=price_df, signals_df=signals_df, start=start_date, end=end_date, strategy=strategy
            )
            trade_log = result.get("trade_log")
            n_trades = 0 if trade_log is None else len(trade_log)
            click.echo(f"\nBacktest result ({n_trades} simulated trades):")
            for k, v in (result.get("metrics") or {}).items():
                click.echo(f"  {k}: {v}")
    except Exception as exc:
        _friendly_error(
            exc,
            "Check swing_trader.backtest.engine.run_backtest(price_df, signals_df, start, "
            "end, strategy) signature, and that enough price history is stored.",
        )


# --- Reports (analytics) ------------------------------------------------------

@cli.command("report")
@click.option("--portfolio", "portfolio_name", default=None, help="Portfolio name (defaults to first found).")
@click.option("--period", type=click.Choice(["weekly", "monthly"]), default="weekly")
def report_cmd(portfolio_name: str | None, period: str) -> None:
    """Print a weekly/monthly performance report."""
    reports = _optional_import("swing_trader.analytics.reports")
    fn_name = "generate_weekly_report" if period == "weekly" else "generate_monthly_report"
    if reports is None or not hasattr(reports, fn_name):
        _not_available(f"{period} report", f"swing_trader.analytics.reports.{fn_name}")
        return
    try:
        with session_scope() as db:
            portfolio = _resolve_portfolio(db, portfolio_name)
            if portfolio is None:
                click.secho("No portfolio found. Create one first with init-portfolio.", fg="yellow")
                return
            fn = getattr(reports, fn_name)
            result = fn(db, portfolio.id)
            click.echo(result)
    except Exception as exc:
        _friendly_error(exc, "Ensure trades exist for this portfolio and swing_trader.analytics is implemented.")


# --- Watchlist (PF-003) -------------------------------------------------------

@cli.command("watchlist-add")
@click.argument("ticker")
@click.argument("condition")
@click.option("--portfolio", "portfolio_name", default=None)
def watchlist_add_cmd(ticker: str, condition: str, portfolio_name: str | None) -> None:
    """Add TICKER to the watchlist with a trigger CONDITION, e.g. 'RSI<30'."""
    ticker = ticker.upper()
    manager = _optional_import("swing_trader.portfolio.manager")
    try:
        with session_scope() as db:
            portfolio = _resolve_portfolio(db, portfolio_name)
            if portfolio is None:
                click.secho("No portfolio found. Create one first with init-portfolio.", fg="yellow")
                return
            if manager is not None and hasattr(manager, "add_to_watchlist"):
                manager.add_to_watchlist(db, portfolio.id, ticker, condition)
            else:
                db.add(WatchlistItem(portfolio_id=portfolio.id, ticker=ticker, trigger_condition=condition))
            click.secho(f"Added {ticker} to watchlist: {condition}", fg="green")
    except Exception as exc:
        _friendly_error(exc, "Check the trigger condition syntax and that the portfolio exists.")


@cli.command("watchlist-list")
@click.option("--portfolio", "portfolio_name", default=None)
def watchlist_list_cmd(portfolio_name: str | None) -> None:
    """List watchlist items (PF-003)."""
    try:
        with session_scope() as db:
            portfolio = _resolve_portfolio(db, portfolio_name)
            if portfolio is None:
                click.secho("No portfolio found.", fg="yellow")
                return
            items = db.query(WatchlistItem).filter(WatchlistItem.portfolio_id == portfolio.id).all()
            if not items:
                click.echo("Watchlist is empty.")
                return
            click.echo(f"{'TICKER':<8}{'TRIGGERED':<12}CONDITION")
            for it in items:
                click.echo(f"{it.ticker:<8}{str(it.triggered):<12}{it.trigger_condition}")
    except Exception as exc:
        _friendly_error(exc, "Check DB connectivity (`make infra-up`).")


# --- System health (7.8) ------------------------------------------------------

@cli.command("system-health")
def system_health_cmd() -> None:
    """Print recent pipeline success rates and API rate-limit summary."""
    try:
        from swing_trader.system_health.monitor import pipeline_success_rate
    except ImportError as exc:
        _friendly_error(exc, "swing_trader.system_health.monitor should always be importable; check for syntax errors.")
        return

    try:
        with session_scope() as db:
            jobs = sorted({row[0] for row in db.query(PipelineRun.job_name).distinct().all()})
            click.echo("Pipeline success rate (last 30 days):")
            if not jobs:
                click.echo("  No pipeline runs recorded yet.")
            for job in jobs:
                rate = pipeline_success_rate(db, job_name=job, days=30)
                click.echo(f"  {job:<20} {rate * 100:5.1f}%")

            click.echo("\nAPI calls (last 24h):")
            since = dt.datetime.utcnow() - dt.timedelta(hours=24)
            rows = (
                db.query(ApiRateLimitLog.provider, ApiRateLimitLog.calls_made)
                .filter(ApiRateLimitLog.ts >= since)
                .all()
            )
            totals: dict[str, int] = {}
            for provider, calls in rows:
                totals[provider] = totals.get(provider, 0) + (calls or 0)
            if not totals:
                click.echo("  No API calls logged in the last 24h.")
            for provider, total in totals.items():
                click.echo(f"  {provider:<20} {total} calls")
    except Exception as exc:
        _friendly_error(exc, "Check DB connectivity; pipeline_runs/api_rate_limit_log are created by `make init-db`.")


if __name__ == "__main__":
    cli()

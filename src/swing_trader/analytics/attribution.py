"""Trade P&L attribution breakdowns (SRS 3.12, PA-003).

All functions operate on a portfolio's closed `Trade` rows and slice P&L by
a different dimension: ticker, sector, market regime, signal rating, and
calendar month (seasonality).
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.db.models import RegimeHistory, SignalRating, TickerUniverse, Trade
from swing_trader.logging_setup import get_logger

logger = get_logger("analytics.attribution")


def _closed_trades_df(session: Session, portfolio_id: int) -> pd.DataFrame:
    trades = (
        session.execute(
            select(Trade)
            .where(Trade.portfolio_id == portfolio_id, Trade.exit_date.is_not(None))
            .order_by(Trade.exit_date)
        )
        .scalars()
        .all()
    )
    rows = [
        {
            "trade_id": t.id,
            "ticker": t.ticker,
            "entry_date": pd.Timestamp(t.entry_date) if t.entry_date else pd.NaT,
            "exit_date": pd.Timestamp(t.exit_date) if t.exit_date else pd.NaT,
            "pnl": t.realized_pnl if t.realized_pnl is not None else 0.0,
            "r_multiple": t.realized_r_multiple,
        }
        for t in trades
    ]
    return pd.DataFrame(
        rows, columns=["trade_id", "ticker", "entry_date", "exit_date", "pnl", "r_multiple"]
    )


def _summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[group_col, "trade_count", "total_pnl", "avg_pnl", "win_rate", "avg_r_multiple"]
        )

    def _agg(g: pd.DataFrame) -> pd.Series:
        wins = (g["pnl"] > 0).sum()
        return pd.Series(
            {
                "trade_count": len(g),
                "total_pnl": g["pnl"].sum(),
                "avg_pnl": g["pnl"].mean(),
                "win_rate": wins / len(g) if len(g) else 0.0,
                "avg_r_multiple": g["r_multiple"].dropna().mean(),
            }
        )

    result = df.groupby(group_col, dropna=False).apply(_agg, include_groups=False).reset_index()
    return result.sort_values("total_pnl", ascending=False).reset_index(drop=True)


def attribution_by_ticker(session: Session, portfolio_id: int) -> pd.DataFrame:
    """PA-003: P&L attribution by ticker."""
    df = _closed_trades_df(session, portfolio_id)
    return _summarize(df, "ticker")


def attribution_by_sector(session: Session, portfolio_id: int) -> pd.DataFrame:
    """PA-003: P&L attribution by sector (joins `TickerUniverse.sector`)."""
    df = _closed_trades_df(session, portfolio_id)
    if df.empty:
        return _summarize(df, "sector")

    tickers = df["ticker"].unique().tolist()
    sector_rows = session.execute(
        select(TickerUniverse.ticker, TickerUniverse.sector).where(TickerUniverse.ticker.in_(tickers))
    ).all()
    sector_map = {ticker: sector for ticker, sector in sector_rows}
    df = df.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")
    return _summarize(df, "sector")


def attribution_by_regime(session: Session, portfolio_id: int) -> pd.DataFrame:
    """PA-003: P&L attribution by market regime at trade entry.

    Joins each trade's `entry_date` against `RegimeHistory.ts` using an
    as-of (backward) match -- i.e. the regime in effect on or immediately
    before the entry date.
    """
    df = _closed_trades_df(session, portfolio_id)
    if df.empty:
        return _summarize(df, "regime")

    regime_rows = session.execute(select(RegimeHistory.ts, RegimeHistory.regime)).all()
    regime_df = pd.DataFrame(regime_rows, columns=["ts", "regime"])
    if regime_df.empty:
        df = df.copy()
        df["regime"] = "Unknown"
        return _summarize(df, "regime")

    regime_df["ts"] = pd.to_datetime(regime_df["ts"])
    regime_df = regime_df.sort_values("ts")
    regime_df["regime"] = regime_df["regime"].map(lambda r: r.value if hasattr(r, "value") else r)

    df = df.copy()
    df["entry_date_norm"] = pd.to_datetime(df["entry_date"]).dt.normalize()
    df = df.sort_values("entry_date_norm")

    merged = pd.merge_asof(
        df, regime_df, left_on="entry_date_norm", right_on="ts", direction="backward"
    )
    merged["regime"] = merged["regime"].fillna("Unknown")
    return _summarize(merged, "regime")


def attribution_by_rating(session: Session, portfolio_id: int) -> pd.DataFrame:
    """PA-003: P&L attribution by the `SignalRating` in effect at trade entry.

    Joins ticker + nearest `SignalRating.as_of` <= `Trade.entry_date`
    (per-ticker as-of match), letting callers compare e.g. whether
    "Strong Buy"-rated entries outperformed "Buy"-rated entries.
    """
    df = _closed_trades_df(session, portfolio_id)
    if df.empty:
        return _summarize(df, "rating")

    tickers = df["ticker"].unique().tolist()
    rating_rows = session.execute(
        select(SignalRating.ticker, SignalRating.as_of, SignalRating.rating).where(
            SignalRating.ticker.in_(tickers)
        )
    ).all()
    rating_df = pd.DataFrame(rating_rows, columns=["ticker", "as_of", "rating"])
    if rating_df.empty:
        df = df.copy()
        df["rating"] = "Unknown"
        return _summarize(df, "rating")

    rating_df["as_of"] = pd.to_datetime(rating_df["as_of"])
    rating_df["rating"] = rating_df["rating"].map(lambda r: r.value if hasattr(r, "value") else r)
    rating_df = rating_df.sort_values("as_of")

    df = df.copy()
    df["entry_date_norm"] = pd.to_datetime(df["entry_date"]).dt.normalize()
    df = df.sort_values("entry_date_norm")

    merged = pd.merge_asof(
        df,
        rating_df,
        left_on="entry_date_norm",
        right_on="as_of",
        by="ticker",
        direction="backward",
    )
    merged["rating"] = merged["rating"].fillna("Unknown")
    return _summarize(merged, "rating")


def attribution_by_month(session: Session, portfolio_id: int) -> pd.DataFrame:
    """PA-003: seasonality -- P&L attribution grouped by calendar exit month.

    Aggregates across all years present in the trade history (e.g. all
    "January" trades regardless of year are combined) to surface seasonal
    patterns.
    """
    df = _closed_trades_df(session, portfolio_id)
    if df.empty:
        return _summarize(df, "month")

    df = df.copy()
    df["month"] = pd.to_datetime(df["exit_date"]).dt.month_name()
    result = _summarize(df, "month")

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    result["month"] = pd.Categorical(result["month"], categories=month_order, ordered=True)
    return result.sort_values("month").reset_index(drop=True)

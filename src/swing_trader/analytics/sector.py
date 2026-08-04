"""Sector rotation summary -- dashboard integration convenience module.

Not part of the original SRS module split (Section 3.13's CC-* requirements
are covered by `swing_trader.analytics.correlation`); added during final
integration to back the dashboard's "Sector Rotation" panel (DV-001) with
real per-sector momentum data instead of just a ticker-count table.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.db.models import StockFeature, TickerUniverse
from swing_trader.logging_setup import get_logger

logger = get_logger("analytics.sector")


def sector_rotation_summary(session: Session) -> pd.DataFrame:
    """Average `ret_5d_vs_sector` / `ret_10d_vs_sector` and the most recent
    `sector_momentum_rank` for each sector represented in the ticker
    universe, using each ticker's latest `StockFeature` row.

    Returns a DataFrame with columns: sector, ticker_count, avg_ret_5d,
    avg_ret_10d, avg_momentum_rank -- sorted by avg_ret_5d descending (best
    momentum first). Empty DataFrame if no sector/feature data exists yet.
    """
    tickers = session.execute(
        select(TickerUniverse.ticker, TickerUniverse.sector).where(TickerUniverse.sector.is_not(None))
    ).all()
    if not tickers:
        return pd.DataFrame()

    sector_map = {ticker: sector for ticker, sector in tickers}
    rows = []

    for ticker, sector in sector_map.items():
        latest = session.execute(
            select(StockFeature)
            .where(StockFeature.ticker == ticker)
            .order_by(StockFeature.ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest is None:
            continue
        rows.append(
            {
                "sector": sector,
                "ticker": ticker,
                "ret_5d_vs_sector": latest.ret_5d_vs_sector,
                "ret_10d_vs_sector": latest.ret_10d_vs_sector,
                "sector_momentum_rank": latest.sector_momentum_rank,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("sector")
        .agg(
            ticker_count=("ticker", "count"),
            avg_ret_5d=("ret_5d_vs_sector", "mean"),
            avg_ret_10d=("ret_10d_vs_sector", "mean"),
            avg_momentum_rank=("sector_momentum_rank", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values("avg_ret_5d", ascending=False, na_position="last").reset_index(drop=True)

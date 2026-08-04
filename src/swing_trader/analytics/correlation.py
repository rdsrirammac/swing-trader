"""Correlation & concentration risk analytics (SRS 3.13, CC-001..004).

All functions take price data as caller-supplied `price_lookup` dicts
(ticker -> close-price pandas Series) rather than fetching prices
themselves, to keep this module decoupled from the data-collection layer.
Sector composition is recomputed directly from `Holding` + `TickerUniverse`
here (rather than importing `swing_trader.signals`) to avoid a cross-agent
import dependency, per the module ownership boundaries for this build.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.db.models import Holding, PositionStatus, StockPrice, TickerUniverse, WatchlistItem
from swing_trader.logging_setup import get_logger

logger = get_logger("analytics.correlation")


def _price_lookup_from_db(session: Session, tickers: list[str], window_days: int = 60) -> dict[str, pd.Series]:
    """Build a `price_lookup` dict (ticker -> close Series) straight from
    stored `StockPrice` rows, so dashboard/CLI callers don't need to build
    it by hand. Pulls a bit more than `window_days` of calendar history to
    ensure enough trading days survive after dropping weekends/holidays.
    """
    lookup: dict[str, pd.Series] = {}
    for ticker in tickers:
        rows = (
            session.execute(
                select(StockPrice.ts, StockPrice.close)
                .where(StockPrice.ticker == ticker, StockPrice.interval == "1d")
                .order_by(StockPrice.ts.desc())
                .limit(window_days * 2)
            )
            .all()
        )
        if not rows:
            continue
        series = pd.Series({ts: close for ts, close in rows}).sort_index()
        lookup[ticker] = series
    return lookup


def portfolio_correlation_summary(session: Session, portfolio_id: int, window_days: int = 60) -> pd.DataFrame:
    """Dashboard integration convenience wrapper (Correlation panel, DV-001 /
    CC-001) that fetches prices for the portfolio's active holdings itself
    (via `_price_lookup_from_db`) and returns the resulting correlation
    matrix, so the caller doesn't need to assemble `price_lookup` manually.
    Returns an empty DataFrame if there are fewer than 2 holdings with
    price history.
    """
    holdings = _active_holdings(session, portfolio_id)
    tickers = [h.ticker for h in holdings]
    if len(tickers) < 2:
        return pd.DataFrame()
    price_lookup = _price_lookup_from_db(session, tickers, window_days=window_days)
    return correlation_matrix(price_lookup, window_days=window_days)


def correlation_matrix(price_lookup: dict[str, pd.Series], window_days: int = 60) -> pd.DataFrame:
    """CC-001: Pearson correlation matrix of daily returns across tickers.

    Args:
        price_lookup: ticker -> close-price Series (DatetimeIndex-ish,
            ascending). Series shorter than `window_days` are used in full.
        window_days: number of most-recent trading days of returns to use.

    Returns:
        Ticker x ticker correlation dataframe. Empty dataframe if fewer
        than 2 tickers have usable data.
    """
    if not price_lookup:
        return pd.DataFrame()

    series = {}
    for ticker, prices in price_lookup.items():
        prices = prices.dropna()
        if len(prices) < 2:
            continue
        returns = prices.pct_change().dropna().tail(window_days)
        series[ticker] = returns

    if len(series) < 2:
        return pd.DataFrame()

    combined = pd.DataFrame(series)
    return combined.corr(method="pearson")


def _active_holdings(session: Session, portfolio_id: int) -> list[Holding]:
    return (
        session.execute(
            select(Holding).where(
                Holding.portfolio_id == portfolio_id, Holding.status != PositionStatus.CLOSED
            )
        )
        .scalars()
        .all()
    )


def _holding_market_value(holding: Holding, price_lookup: dict[str, pd.Series]) -> float:
    """Best-effort current market value: last known price if available, else entry_price."""
    prices = price_lookup.get(holding.ticker)
    if prices is not None and len(prices.dropna()) > 0:
        last_price = float(prices.dropna().iloc[-1])
    else:
        last_price = holding.entry_price
    return holding.shares * last_price


def _sector_weights(
    session: Session, holdings: list[Holding], price_lookup: dict[str, pd.Series]
) -> dict[str, float]:
    """Portfolio weight (0-1) per sector, recomputed via Holding + TickerUniverse join."""
    if not holdings:
        return {}
    values = {h.ticker: _holding_market_value(h, price_lookup) for h in holdings}
    total_value = sum(values.values())
    if total_value <= 0:
        return {}

    tickers = list(values.keys())
    sector_rows = session.execute(
        select(TickerUniverse.ticker, TickerUniverse.sector).where(TickerUniverse.ticker.in_(tickers))
    ).all()
    sector_map = {ticker: (sector or "Unknown") for ticker, sector in sector_rows}

    sector_totals: dict[str, float] = {}
    for ticker, value in values.items():
        sector = sector_map.get(ticker, "Unknown")
        sector_totals[sector] = sector_totals.get(sector, 0.0) + value

    return {sector: value / total_value for sector, value in sector_totals.items()}


def _avg_pairwise_correlation(
    holdings: list[Holding], price_lookup: dict[str, pd.Series], window_days: int = 60
) -> float | None:
    tickers = [h.ticker for h in holdings]
    sub_lookup = {t: price_lookup[t] for t in tickers if t in price_lookup}
    corr = correlation_matrix(sub_lookup, window_days=window_days)
    if corr.empty or corr.shape[0] < 2:
        return None
    # Average of the upper triangle (excluding diagonal).
    mask = ~pd.DataFrame(
        [[i == j for j in range(corr.shape[0])] for i in range(corr.shape[0])],
        index=corr.index,
        columns=corr.columns,
    )
    values = corr.where(mask).stack()
    return float(values.mean()) if len(values) else None


def _portfolio_beta(
    holdings: list[Holding], price_lookup: dict[str, pd.Series], beta_lookup: dict[str, float]
) -> float | None:
    values = {h.ticker: _holding_market_value(h, price_lookup) for h in holdings}
    total_value = sum(values.values())
    if total_value <= 0:
        return None
    weighted_beta = 0.0
    weighted_total = 0.0
    for ticker, value in values.items():
        beta = beta_lookup.get(ticker)
        if beta is None:
            continue
        weighted_beta += value * beta
        weighted_total += value
    if weighted_total <= 0:
        return None
    return weighted_beta / weighted_total


def diversification_score(
    session: Session,
    portfolio_id: int,
    price_lookup: dict[str, pd.Series],
    beta_lookup: dict[str, float] | None = None,
) -> float:
    """CC-002: composite 0-100 diversification score.

    Components (25 points each):
        - holdings count: full credit for 5-10 holdings, tapering off below
          5 (too concentrated) and above 10 (diminishing benefit).
        - sector concentration: full credit unless the largest sector
          exceeds `risk.max_sector_concentration_pct` (30%), penalized
          1 point per percentage-point over.
        - average pairwise correlation: full credit below 0.5, penalized
          1 point per 0.01 over.
        - portfolio beta: full credit (no penalty) if `beta_lookup` is not
          supplied (unknown -- not penalized); otherwise full credit below
          1.5, penalized 0.5 points per 0.01 over.
    """
    holdings = _active_holdings(session, portfolio_id)
    n = len(holdings)

    if n == 0:
        return 0.0

    # -- count score --
    if 5 <= n <= 10:
        count_score = 25.0
    elif n < 5:
        count_score = 25.0 * (n / 5.0)
    else:
        count_score = max(0.0, 25.0 - (n - 10) * 2.0)

    # -- sector score --
    sector_weights = _sector_weights(session, holdings, price_lookup)
    max_sector_pct = max(sector_weights.values()) if sector_weights else 0.0
    sector_penalty = max(0.0, (max_sector_pct - 0.30) * 100.0)
    sector_score = max(0.0, 25.0 - sector_penalty)

    # -- correlation score --
    avg_corr = _avg_pairwise_correlation(holdings, price_lookup)
    if avg_corr is None:
        corr_score = 25.0  # insufficient data -- don't penalize
    else:
        corr_penalty = max(0.0, (avg_corr - 0.5) * 100.0)
        corr_score = max(0.0, 25.0 - corr_penalty)

    # -- beta score --
    if beta_lookup is None:
        beta_score = 25.0
    else:
        beta = _portfolio_beta(holdings, price_lookup, beta_lookup)
        if beta is None:
            beta_score = 25.0
        else:
            beta_penalty = max(0.0, (beta - 1.5) * 50.0)
            beta_score = max(0.0, 25.0 - beta_penalty)

    total = count_score + sector_score + corr_score + beta_score
    return float(max(0.0, min(100.0, total)))


def concentration_alerts(
    session: Session,
    portfolio_id: int,
    price_lookup: dict[str, pd.Series],
    beta_lookup: dict[str, float] | None = None,
) -> list[dict]:
    """CC-003: concentration risk alerts for a portfolio.

    Checks: single position > 15% of portfolio value, single sector > 30%,
    portfolio beta > 1.3 (if `beta_lookup` given), average pairwise
    correlation > 0.6.

    Returns:
        List of {"type": str, "message": str, "severity": str} dicts,
        suitable for handing to the alerting/notification module by the
        caller (this module does not send alerts itself).
    """
    holdings = _active_holdings(session, portfolio_id)
    alerts: list[dict] = []
    if not holdings:
        return alerts

    values = {h.ticker: _holding_market_value(h, price_lookup) for h in holdings}
    total_value = sum(values.values())

    if total_value > 0:
        for ticker, value in values.items():
            pct = value / total_value
            if pct > 0.15:
                alerts.append(
                    {
                        "type": "single_position_concentration",
                        "message": f"{ticker} is {pct:.1%} of portfolio value (> 15% threshold).",
                        "severity": "warning",
                    }
                )

    sector_weights = _sector_weights(session, holdings, price_lookup)
    for sector, pct in sector_weights.items():
        if pct > 0.30:
            alerts.append(
                {
                    "type": "sector_concentration",
                    "message": f"Sector '{sector}' is {pct:.1%} of portfolio value (> 30% threshold).",
                    "severity": "warning",
                }
            )

    if beta_lookup is not None:
        beta = _portfolio_beta(holdings, price_lookup, beta_lookup)
        if beta is not None and beta > 1.3:
            alerts.append(
                {
                    "type": "high_portfolio_beta",
                    "message": f"Portfolio beta is {beta:.2f} (> 1.3 threshold).",
                    "severity": "warning",
                }
            )

    avg_corr = _avg_pairwise_correlation(holdings, price_lookup)
    if avg_corr is not None and avg_corr > 0.6:
        alerts.append(
            {
                "type": "high_correlation",
                "message": f"Average pairwise holdings correlation is {avg_corr:.2f} (> 0.6 threshold).",
                "severity": "warning",
            }
        )

    return alerts


def suggest_replacements(
    session: Session,
    portfolio_id: int,
    high_corr_ticker: str,
    watchlist_price_lookup: dict[str, pd.Series],
) -> list[str]:
    """CC-004: suggest lower-correlation replacements from the watchlist.

    Ranks the portfolio's `WatchlistItem` tickers by correlation to
    `high_corr_ticker` (lowest first) and returns the top 3 candidates.

    Args:
        watchlist_price_lookup: must include a price Series for
            `high_corr_ticker` itself (used as the correlation baseline) in
            addition to the watchlist tickers' series.
    """
    if high_corr_ticker not in watchlist_price_lookup:
        logger.warning(
            "suggest_replacements: no price series for high_corr_ticker=%s; cannot rank.",
            high_corr_ticker,
        )
        return []

    watchlist_tickers = (
        session.execute(
            select(WatchlistItem.ticker).where(WatchlistItem.portfolio_id == portfolio_id)
        )
        .scalars()
        .all()
    )
    watchlist_tickers = [t for t in dict.fromkeys(watchlist_tickers) if t != high_corr_ticker]
    if not watchlist_tickers:
        return []

    candidate_lookup = {
        t: watchlist_price_lookup[t] for t in watchlist_tickers if t in watchlist_price_lookup
    }
    candidate_lookup[high_corr_ticker] = watchlist_price_lookup[high_corr_ticker]

    corr = correlation_matrix(candidate_lookup)
    if corr.empty or high_corr_ticker not in corr.columns:
        return []

    ranked = (
        corr[high_corr_ticker]
        .drop(labels=[high_corr_ticker], errors="ignore")
        .dropna()
        .sort_values(ascending=True)
    )
    return ranked.head(3).index.tolist()

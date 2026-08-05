"""Feature engineering orchestrator (SRS FE-007).

`build_feature_row` calls every FE-001..006 feature function for a single
ticker/as_of date, merges the results into one flat dict keyed by
`StockFeature` column names, and computes `feature_completeness`. It does
NOT write to the database — callers (the daily pipeline / backfill jobs)
own the session and call `upsert_feature_row` explicitly so this module
stays easily unit-testable without a DB.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from swing_trader.db.models import StockFeature
from swing_trader.logging_setup import get_logger

from swing_trader.features.fundamental import compute_fundamental_features
from swing_trader.features.macro import compute_macro_features
from swing_trader.features.relative_strength import compute_relative_strength
from swing_trader.features.sentiment import (
    analyst_rating_velocity,
    get_scorer,
    options_put_call_skew,
)
from swing_trader.features.technical import compute_technical_indicators
from swing_trader.features.volatility import compute_volatility_features

logger = get_logger("features.engineering")

# Best-effort GICS sector name -> SPDR sector ETF mapping, used only to look
# up this ticker's own sector_momentum_rank from `sector_etf_histories`.
# yfinance `info["sector"]` strings vary in exact wording across versions,
# hence the loose/normalized matching in `_map_sector_to_etf`.
_SECTOR_TO_ETF = {
    "technology": "XLK",
    "financial services": "XLF",
    "financials": "XLF",
    "energy": "XLE",
    "consumer cyclical": "XLY",
    "consumer discretionary": "XLY",
    "consumer defensive": "XLP",
    "consumer staples": "XLP",
    "healthcare": "XLV",
    "health care": "XLV",
    "industrials": "XLI",
    "basic materials": "XLB",
    "materials": "XLB",
    "real estate": "XLRE",
    "utilities": "XLU",
    "communication services": "XLC",
}

# The ~40 numeric/boolean feature columns that make up feature_completeness's
# denominator (everything on StockFeature except id/ticker/ts/feature_completeness).
_FEATURE_COLUMNS = [
    "rsi_2", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14",
    "bb_upper", "bb_lower", "bb_bandwidth", "ema_20", "sma_50", "adx_14",
    "roc_5", "roc_10", "roc_21", "obv", "volume_ratio_20d",
    "ret_5d_vs_spy", "ret_10d_vs_spy", "ret_21d_vs_spy",
    "ret_5d_vs_sector", "ret_10d_vs_sector", "ret_21d_vs_sector",
    "rs_rating", "sector_momentum_rank",
    "realized_vol_20d", "realized_vol_pctile", "atr_pct", "hv_iv_spread",
    "news_sentiment_3d_avg", "news_volume_velocity", "analyst_rating_velocity",
    "options_put_call_skew",
    "pe_percentile_sector", "pe_percentile_history",
    "short_interest_pct_float", "earnings_surprise_streak",
    "vix_level", "vix_percentile", "spy_above_ema20", "sector_breadth_pct",
    "yield_curve_10y_2y",
]


def _map_sector_to_etf(sector_name: str | None) -> str | None:
    if not sector_name:
        return None
    return _SECTOR_TO_ETF.get(str(sector_name).strip().lower())


def _sector_etf_trailing_returns(
    sector_etf_histories: dict[str, pd.DataFrame], window: int = 21
) -> dict[str, float]:
    from swing_trader.features.relative_strength import _trailing_return

    returns: dict[str, float] = {}
    for etf, df in (sector_etf_histories or {}).items():
        ret = _trailing_return(df, window)
        if ret is not None:
            returns[etf] = ret
    return returns


def _last_row_dict(df: pd.DataFrame, columns: list[str]) -> dict:
    out = {c: None for c in columns}
    if df is None or df.empty:
        return out
    last = df.iloc[-1]
    for c in columns:
        if c in df.columns:
            val = last[c]
            out[c] = None if pd.isna(val) else val
    return out


def build_feature_row(
    ticker: str,
    as_of: dt.date,
    price_history: pd.DataFrame,
    spy_history: pd.DataFrame,
    sector_history: pd.DataFrame | None,
    vix_history: pd.DataFrame,
    info: dict,
    news_rows: list[dict],
    recommendations_df,
    options_chain,
    sector_etf_histories: dict[str, pd.DataFrame],
) -> dict:
    """Compute the full FE-001..006 feature set for one ticker/date.

    Returns a flat dict of `StockFeature` column names -> values (does not
    include id/ticker/ts; the caller supplies those to `upsert_feature_row`).
    """
    feature_dict: dict = {c: None for c in _FEATURE_COLUMNS}

    # --- FE-001 technical ---
    try:
        tech_df = compute_technical_indicators(price_history)
        tech_cols = [
            "rsi_2", "rsi_14", "macd", "macd_signal", "macd_hist", "atr_14",
            "bb_upper", "bb_lower", "bb_bandwidth", "ema_20", "sma_50",
            "adx_14", "roc_5", "roc_10", "roc_21", "obv", "volume_ratio_20d",
        ]
        feature_dict.update(_last_row_dict(tech_df, tech_cols))
    except Exception as e:
        logger.warning("[%s] technical indicators failed: %s", ticker, e)

    # --- FE-002 relative strength ---
    try:
        rs = compute_relative_strength(price_history, spy_history, sector_history)
        feature_dict.update(rs)
    except Exception as e:
        logger.warning("[%s] relative strength failed: %s", ticker, e)

    try:
        sector_etf_returns = _sector_etf_trailing_returns(sector_etf_histories)
        from swing_trader.features.relative_strength import sector_momentum_rank as _rank_fn

        ranks = _rank_fn(sector_etf_returns)
        etf = _map_sector_to_etf((info or {}).get("sector"))
        if etf and etf in ranks:
            feature_dict["sector_momentum_rank"] = ranks[etf]
    except Exception as e:
        logger.warning("[%s] sector_momentum_rank failed: %s", ticker, e)

    # rs_rating requires a peer universe of returns which this per-ticker
    # orchestrator call doesn't have (it operates on one ticker at a time);
    # left None here — a batch-level caller can compute it separately via
    # `compute_rs_rating` across the full ticker universe and overwrite it.

    # --- FE-003 volatility ---
    try:
        vol = compute_volatility_features(price_history, iv=None)
        feature_dict.update(vol)
    except Exception as e:
        logger.warning("[%s] volatility features failed: %s", ticker, e)

    # --- FE-004 sentiment / analyst / options ---
    try:
        scorer = get_scorer()
        # news_rows may already carry a `sentiment_score`/`published_at`
        # (populated upstream against NewsSentiment); if a row is missing a
        # score but has text, score it on the fly.
        enriched_rows = []
        for row in news_rows or []:
            r = dict(row)
            if r.get("sentiment_score") is None and r.get("headline"):
                scored = scorer.score(r["headline"])
                r["sentiment_score"] = scored["score"]
            enriched_rows.append(r)
        agg = scorer.aggregate_news_features(enriched_rows)
        feature_dict.update(agg)
    except Exception as e:
        logger.warning("[%s] sentiment aggregation failed: %s", ticker, e)

    try:
        feature_dict["analyst_rating_velocity"] = analyst_rating_velocity(recommendations_df)
    except Exception as e:
        logger.warning("[%s] analyst_rating_velocity failed: %s", ticker, e)

    try:
        if options_chain is not None:
            calls = getattr(options_chain, "calls", None)
            puts = getattr(options_chain, "puts", None)
            feature_dict["options_put_call_skew"] = options_put_call_skew(calls, puts)
    except Exception as e:
        logger.warning("[%s] options_put_call_skew failed: %s", ticker, e)

    # --- FE-005 fundamental ---
    try:
        fund = compute_fundamental_features(
            info=info or {},
            sector_pe_values=None,
            historical_pe=None,
            earnings_history=None,
        )
        feature_dict.update(fund)
    except Exception as e:
        logger.warning("[%s] fundamental features failed: %s", ticker, e)

    # --- FE-006 macro ---
    try:
        macro = compute_macro_features(
            as_of=as_of,
            spy_df=spy_history,
            vix_df=vix_history,
            sector_etf_dfs=sector_etf_histories or {},
            treasury_10y=None,
            treasury_2y_proxy=None,
        )
        feature_dict.update(macro)
    except Exception as e:
        logger.warning("[%s] macro features failed: %s", ticker, e)

    non_null = sum(1 for c in _FEATURE_COLUMNS if feature_dict.get(c) is not None)
    feature_dict["feature_completeness"] = round(non_null / len(_FEATURE_COLUMNS), 4)

    return feature_dict


def _to_python_scalar(value):
    """Unwrap numpy scalar types (np.float64, np.int64, np.bool_, ...) to
    native Python equivalents.

    pandas/pandas_ta computations routinely hand back numpy scalars rather
    than plain floats/ints/bools. SQLite (used in tests) accepts these
    transparently, but psycopg2/PostgreSQL does not have a default adapter
    for them -- it renders `np.float64(...)` as a literal in the SQL text
    and the insert fails with "schema np does not exist". `.item()` is the
    numpy-supported way to get the native Python type back; plain Python
    values (and None) don't have `.item()` and pass through unchanged.
    """
    if value is not None and hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def upsert_feature_row(session: Session, ticker: str, as_of: dt.date, feature_dict: dict) -> StockFeature:
    """Insert or update the `StockFeature` row for (ticker, as_of)."""
    existing = session.execute(
        select(StockFeature).where(StockFeature.ticker == ticker, StockFeature.ts == as_of)
    ).scalar_one_or_none()

    row = existing or StockFeature(ticker=ticker, ts=as_of)

    for key, value in feature_dict.items():
        if hasattr(StockFeature, key):
            setattr(row, key, _to_python_scalar(value))

    if existing is None:
        session.add(row)

    return row

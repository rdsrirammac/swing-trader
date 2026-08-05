"""SRS 3.2 Ticker Universe & Auto-Backfill (TB-001..TB-006).

The auto-backfill orchestrator: given a new ticker, runs six phases (price,
fundamentals, news, options, features, model warm-up), tracking progress via
`BackfillJob` rows, then applies the TB-003 data-quality gate to decide
whether the ticker graduates to `TickerUniverse.status = active`.

Failure handling (TB-004, NFR 4.2 graceful degradation):
  - PRICE is the only *critical* phase. If it fails `_MAX_PHASE_ATTEMPTS`
    times in a row, the ticker is marked `failed` and remaining phases are
    skipped entirely.
  - fundamentals / news / options / features / model_warmup are
    non-critical: a failure (even after retries) is logged into that
    phase's `BackfillJob` row but does NOT abort the run -- later phases
    still execute, and the quality gate only cares about price/feature
    completeness/freshness, not about non-critical phase status.
  - features / model_warmup additionally guard against `swing_trader.features`
    / `swing_trader.models` not existing yet (parallel build) via
    try/except ImportError.
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from swing_trader.config import get_settings
from swing_trader.data.newsapi_client import NewsApiClient
from swing_trader.data.validators import data_freshness_ok, validate_price_dataframe
from swing_trader.data.yf_client import YFinanceClient
from swing_trader.db.base import session_scope
from swing_trader.db.models import (
    BackfillJob,
    DailyMetric,
    EarningsEvent,
    NewsSentiment,
    StockFeature,
    StockPrice,
    TickerStatus,
    TickerUniverse,
)
from swing_trader.logging_setup import get_logger

logger = get_logger("portfolio.backfill")

_CRITICAL_PHASES = {"price"}
_MAX_PHASE_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _get_or_create_ticker(session: Session, ticker: str) -> TickerUniverse:
    row = session.execute(select(TickerUniverse).where(TickerUniverse.ticker == ticker)).scalar_one_or_none()
    if row is None:
        row = TickerUniverse(ticker=ticker, status=TickerStatus.PENDING)
        session.add(row)
        session.flush()
    return row


def _to_naive_datetime(ts) -> dt.datetime:
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if isinstance(ts, dt.datetime) and ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)
    return ts


def _parse_published_at(value) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, (int, float)):
        try:
            return dt.datetime.utcfromtimestamp(value)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            parsed = pd.Timestamp(value).to_pydatetime()
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except Exception:
            return None
    return None


def _extract_earnings_date(calendar) -> dt.date | None:
    """yfinance's `.calendar` shape varies by version: a dict with an
    'Earnings Date' key (list of dates) in newer versions, or a
    DataFrame-like object in older ones. Best-effort extraction; returns
    None on any unrecognized shape rather than raising."""
    if calendar is None:
        return None
    try:
        if isinstance(calendar, dict):
            dates = calendar.get("Earnings Date")
            if isinstance(dates, (list, tuple)) and dates:
                d = dates[0]
                return d if isinstance(d, dt.date) and not isinstance(d, dt.datetime) else pd.Timestamp(d).date()
        elif hasattr(calendar, "loc"):
            d = calendar.loc["Earnings Date"]
            d = d.iloc[0] if hasattr(d, "iloc") else d
            return pd.Timestamp(d).date()
    except Exception:
        return None
    return None


def _extract_eps_estimate(calendar) -> float | None:
    if not isinstance(calendar, dict):
        return None
    for key in ("EPS Estimate Avg", "Earnings Average"):
        val = calendar.get(key)
        if isinstance(val, (list, tuple)) and val:
            try:
                return float(val[0])
            except (TypeError, ValueError):
                return None
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _run_phase(session: Session, ticker: str, phase: str, fn) -> BackfillJob:
    """TB-002/TB-004: run a single backfill phase with up to
    `_MAX_PHASE_ATTEMPTS` retries (manual loop + exponential-ish backoff),
    tracked via one `BackfillJob` row per phase (its `attempt` field records
    the retry count). `fn` takes no args and returns records_ingested (int).
    """
    job = BackfillJob(ticker=ticker, phase=phase, status="running", attempt=1, pct_complete=0.0)
    session.add(job)
    session.flush()

    last_error: str | None = None
    for attempt in range(1, _MAX_PHASE_ATTEMPTS + 1):
        job.attempt = attempt
        try:
            records = fn()
            job.status = "done"
            job.pct_complete = 100.0
            job.records_ingested = records
            job.error_message = None
            job.finished_at = dt.datetime.utcnow()
            session.flush()
            return job
        except Exception as e:  # noqa: BLE001 - any data-source failure should retry, not crash the run
            last_error = str(e)
            logger.warning("Backfill phase=%s ticker=%s attempt=%d failed: %s", phase, ticker, attempt, e)
            if attempt < _MAX_PHASE_ATTEMPTS:
                time.sleep(min(2**attempt, 10))

    job.status = "failed"
    job.error_message = last_error
    job.finished_at = dt.datetime.utcnow()
    session.flush()
    return job


# ---------------------------------------------------------------------------
# Phase 1: PRICE (critical)
# ---------------------------------------------------------------------------

def _phase_price(session: Session, ticker: str) -> int:
    settings = get_settings()
    client = YFinanceClient.instance()
    period = f"{settings.get('ticker_universe.backfill_years', 1)}y"
    df = client.get_history(ticker, period=period, interval="1d")

    result = validate_price_dataframe(df, ticker)
    if not result.is_valid:
        raise RuntimeError(f"price validation failed for {ticker}: {'; '.join(result.errors)}")

    records = 0
    for ts, row in df.iterrows():
        ts_val = _to_naive_datetime(ts)
        exists = session.execute(
            select(StockPrice).where(
                StockPrice.ticker == ticker, StockPrice.ts == ts_val, StockPrice.interval == "1d"
            )
        ).scalar_one_or_none()
        if exists is not None:
            continue
        session.add(
            StockPrice(
                ticker=ticker,
                ts=ts_val,
                interval="1d",
                open=float(row.get("Open", 0.0) or 0.0),
                high=float(row.get("High", 0.0) or 0.0),
                low=float(row.get("Low", 0.0) or 0.0),
                close=float(row.get("Close", 0.0) or 0.0),
                adj_close=float(row["Adj Close"]) if "Adj Close" in row and pd.notna(row["Adj Close"]) else None,
                volume=float(row.get("Volume", 0.0) or 0.0),
                dividends=float(row.get("Dividends", 0.0) or 0.0),
                splits=float(row.get("Stock Splits", 0.0) or 0.0),
                source="yfinance",
            )
        )
        records += 1
    session.flush()

    # Best-effort 60-day intraday bars; failure here does not fail the (critical) price phase.
    try:
        intraday_days = settings.get("ticker_universe.intraday_days", 60)
        intraday_df = client.get_history(ticker, period=f"{intraday_days}d", interval="30m")
        for ts, row in intraday_df.iterrows():
            ts_val = _to_naive_datetime(ts)
            exists = session.execute(
                select(StockPrice).where(
                    StockPrice.ticker == ticker, StockPrice.ts == ts_val, StockPrice.interval == "30m"
                )
            ).scalar_one_or_none()
            if exists is not None:
                continue
            session.add(
                StockPrice(
                    ticker=ticker,
                    ts=ts_val,
                    interval="30m",
                    open=float(row.get("Open", 0.0) or 0.0),
                    high=float(row.get("High", 0.0) or 0.0),
                    low=float(row.get("Low", 0.0) or 0.0),
                    close=float(row.get("Close", 0.0) or 0.0),
                    adj_close=None,
                    volume=float(row.get("Volume", 0.0) or 0.0),
                    source="yfinance",
                )
            )
            records += 1
        session.flush()
    except Exception as e:
        logger.warning("Intraday backfill (non-critical) failed for %s: %s", ticker, e)

    ticker_row = _get_or_create_ticker(session, ticker)
    if not df.empty:
        ticker_row.last_price = float(df["Close"].iloc[-1])
        ticker_row.avg_daily_volume = float(df["Volume"].tail(20).mean())
        ticker_row.last_updated = dt.datetime.utcnow()
    session.flush()

    return records


# ---------------------------------------------------------------------------
# Phase 2: FUNDAMENTALS
# ---------------------------------------------------------------------------

def _phase_fundamentals(session: Session, ticker: str) -> int:
    client = YFinanceClient.instance()
    info = client.get_info(ticker) or {}
    records = 0

    today = dt.date.today()
    metric = session.execute(
        select(DailyMetric).where(DailyMetric.ticker == ticker, DailyMetric.ts == today)
    ).scalar_one_or_none()
    if metric is None:
        metric = DailyMetric(ticker=ticker, ts=today)
        session.add(metric)

    metric.market_cap = info.get("marketCap")
    metric.pe_ratio = info.get("trailingPE")
    metric.peg_ratio = info.get("pegRatio")
    metric.ps_ratio = info.get("priceToSalesTrailing12Months")
    metric.short_ratio = info.get("shortRatio")
    metric.float_shares = info.get("floatShares")
    records += 1

    # Primes the yf_client disk cache for features.fundamental (FE-005);
    # not persisted to its own table here (out of this module's scope).
    try:
        client.get_quarterly_financials(ticker)
        client.get_quarterly_earnings(ticker)
    except Exception as e:
        logger.warning("quarterly financials/earnings fetch failed for %s: %s", ticker, e)

    try:
        calendar = client.get_calendar(ticker) or {}
        earnings_date = _extract_earnings_date(calendar)
        if earnings_date is not None:
            existing_event = session.execute(
                select(EarningsEvent).where(
                    EarningsEvent.ticker == ticker, EarningsEvent.earnings_date == earnings_date
                )
            ).scalar_one_or_none()
            if existing_event is None:
                session.add(
                    EarningsEvent(
                        ticker=ticker,
                        earnings_date=earnings_date,
                        confirmed=False,
                        eps_estimate=_extract_eps_estimate(calendar),
                    )
                )
                records += 1
    except Exception as e:
        logger.warning("calendar fetch failed for %s: %s", ticker, e)

    session.flush()
    return records


# ---------------------------------------------------------------------------
# Phase 3: NEWS
# ---------------------------------------------------------------------------

def _phase_news(session: Session, ticker: str) -> int:
    settings = get_settings()
    client = YFinanceClient.instance()
    records = 0

    items: list[dict] = []
    try:
        yf_news = client.get_news(ticker) or []
    except Exception as e:
        logger.warning("yfinance news fetch failed for %s: %s", ticker, e)
        yf_news = []

    for n in yf_news:
        content = n.get("content", n) if isinstance(n, dict) else {}
        headline = content.get("title") or n.get("title")
        if not headline:
            continue
        published = content.get("pubDate") or n.get("providerPublishTime")
        provider = content.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else n.get("publisher")
        canonical = content.get("canonicalUrl")
        url = canonical.get("url") if isinstance(canonical, dict) else n.get("link")
        items.append(
            {
                "headline": headline,
                "publisher": publisher,
                "url": url,
                "published_at": published,
                "source": "yfinance",
            }
        )

    news_history_days = settings.get("ticker_universe.news_history_days", 90)
    if len(items) < 5:
        # yfinance's .news is effectively "latest only" -- fall back to
        # NewsAPI for a real historical backfill window (gracefully returns
        # [] if no API key is configured; see NewsApiClient).
        items.extend(NewsApiClient().get_historical_news(ticker, days=news_history_days))

    for item in items:
        published_at = _parse_published_at(item.get("published_at"))
        headline = item.get("headline")
        if published_at is None or not headline:
            continue
        exists = session.execute(
            select(NewsSentiment).where(
                NewsSentiment.ticker == ticker,
                NewsSentiment.headline == headline,
                NewsSentiment.published_at == published_at,
            )
        ).scalar_one_or_none()
        if exists is not None:
            continue
        session.add(
            NewsSentiment(
                ticker=ticker,
                published_at=published_at,
                headline=headline,
                publisher=item.get("publisher"),
                url=item.get("url"),
                source=item.get("source", "yfinance"),
                # scored later by features.sentiment.SentimentScorer during
                # feature engineering -- deliberately left unscored here to
                # avoid importing the features package from this module.
                sentiment_label=None,
                sentiment_score=None,
            )
        )
        records += 1

    session.flush()
    return records


# ---------------------------------------------------------------------------
# Phase 4: OPTIONS
# ---------------------------------------------------------------------------

def _phase_options(session: Session, ticker: str) -> int:
    client = YFinanceClient.instance()
    expirations = client.get_option_expirations(ticker) or ()
    if not expirations:
        logger.info("No option expirations available for %s; skipping options phase", ticker)
        return 0

    nearest = expirations[0]
    chain = client.get_options_chain(ticker, nearest)
    calls = getattr(chain, "calls", None)
    puts = getattr(chain, "puts", None)
    calls_vol = float(calls["volume"].fillna(0).sum()) if calls is not None and not calls.empty else 0.0
    puts_vol = float(puts["volume"].fillna(0).sum()) if puts is not None and not puts.empty else 0.0

    # NOTE: put_call_ratio_5d here is actually a single-day snapshot ratio
    # from the nearest expiration's chain (yfinance gives no 5-day option
    # volume history from one call) -- a true 5-day rolling ratio requires
    # accumulating this daily via run_daily_incremental_update. Documented
    # simplification per TB-001 Phase 4.
    put_call_ratio = (puts_vol / calls_vol) if calls_vol > 0 else None

    today = dt.date.today()
    metric = session.execute(
        select(DailyMetric).where(DailyMetric.ticker == ticker, DailyMetric.ts == today)
    ).scalar_one_or_none()
    if metric is None:
        metric = DailyMetric(ticker=ticker, ts=today)
        session.add(metric)
    metric.put_call_ratio_5d = put_call_ratio
    session.flush()
    return 1


# ---------------------------------------------------------------------------
# Phase 5: FEATURES (extension point)
# ---------------------------------------------------------------------------

def _phase_features(session: Session, ticker: str) -> int:
    try:
        from swing_trader.features.engineering import (  # type: ignore
            _map_sector_to_etf,
            build_feature_row,
            upsert_feature_row,
        )
        from swing_trader.features.macro import SECTOR_ETFS  # type: ignore
    except ImportError:
        logger.info("features.engineering not available yet; skipping feature warm-up for %s", ticker)
        return 0

    try:
        settings = get_settings()
        client = YFinanceClient.instance()
        period = f"{settings.get('ticker_universe.backfill_years', 1)}y"

        price_history = client.get_history(ticker, period=period, interval="1d")
        if price_history is None or price_history.empty:
            logger.warning("feature warm-up: no price history for %s; skipping", ticker)
            return 0
        as_of = _to_naive_datetime(price_history.index[-1]).date()

        spy_history = client.get_history("SPY", period=period, interval="1d")
        vix_history = client.get_history("^VIX", period=period, interval="1d")

        try:
            info = client.get_info(ticker) or {}
        except Exception as e:
            logger.warning("feature warm-up: get_info failed for %s: %s", ticker, e)
            info = {}

        sector_etf_histories: dict = {}
        for etf in SECTOR_ETFS:
            try:
                etf_df = client.get_history(etf, period=period, interval="1d")
                if etf_df is not None and not etf_df.empty:
                    sector_etf_histories[etf] = etf_df
            except Exception as e:
                logger.warning("feature warm-up: sector ETF %s history failed: %s", etf, e)

        sector_etf = _map_sector_to_etf(info.get("sector"))
        sector_history = sector_etf_histories.get(sector_etf) if sector_etf else None

        news_rows_orm = session.execute(
            select(NewsSentiment)
            .where(NewsSentiment.ticker == ticker)
            .order_by(NewsSentiment.published_at.desc())
            .limit(50)
        ).scalars().all()
        news_rows = [
            {
                "headline": n.headline,
                "published_at": n.published_at,
                "sentiment_score": n.sentiment_score,
            }
            for n in news_rows_orm
        ]

        try:
            recommendations_df = client.get_recommendations(ticker)
        except Exception as e:
            logger.warning("feature warm-up: recommendations fetch failed for %s: %s", ticker, e)
            recommendations_df = None

        options_chain = None
        try:
            expirations = client.get_option_expirations(ticker) or ()
            if expirations:
                options_chain = client.get_options_chain(ticker, expirations[0])
        except Exception as e:
            logger.warning("feature warm-up: options chain fetch failed for %s: %s", ticker, e)

        feature_dict = build_feature_row(
            ticker=ticker,
            as_of=as_of,
            price_history=price_history,
            spy_history=spy_history,
            sector_history=sector_history,
            vix_history=vix_history,
            info=info,
            news_rows=news_rows,
            recommendations_df=recommendations_df,
            options_chain=options_chain,
            sector_etf_histories=sector_etf_histories,
        )
        upsert_feature_row(session, ticker, as_of, feature_dict)
        session.flush()
        return 1
    except Exception as e:
        logger.warning("feature warm-up failed for %s: %s", ticker, e)
        return 0


# ---------------------------------------------------------------------------
# Phase 6: MODEL WARM-UP (extension point)
# ---------------------------------------------------------------------------

def _phase_model_warmup(session: Session, ticker: str) -> int:
    try:
        from swing_trader.models import regime_detector  # type: ignore
    except ImportError:
        logger.info("models.regime_detector not available yet; skipping model warm-up for %s", ticker)
        return 0
    try:
        warm_up = getattr(regime_detector, "warm_up", None)
        if warm_up is not None:
            warm_up(session, ticker)
        return 1
    except Exception as e:
        logger.warning("model warm-up failed for %s: %s", ticker, e)
        return 0


# ---------------------------------------------------------------------------
# TB-003: quality gate
# ---------------------------------------------------------------------------

def check_quality_gate(session: Session, ticker: str) -> tuple[bool, list[str]]:
    """TB-003: min_daily_bars, feature_completeness, freshness, and no
    critical (price) BackfillJob failures. Sets
    `TickerUniverse.status = active` if all checks pass, else `failed`.

    If no StockFeature row exists yet for the ticker, that specific check
    fails (feature_completeness treated as unavailable == failing) but the
    other checks are still evaluated independently.
    """
    settings = get_settings()
    reasons: list[str] = []

    min_bars = settings.get("data_quality.min_daily_bars", 200)
    bar_count = session.execute(
        select(func.count()).select_from(StockPrice).where(StockPrice.ticker == ticker, StockPrice.interval == "1d")
    ).scalar_one()
    if bar_count < min_bars:
        reasons.append(f"only {bar_count} daily bars ingested, need >= {min_bars}")

    min_completeness = settings.get("data_quality.min_feature_completeness", 0.80)
    latest_feature = session.execute(
        select(StockFeature).where(StockFeature.ticker == ticker).order_by(StockFeature.ts.desc()).limit(1)
    ).scalar_one_or_none()
    if latest_feature is None or latest_feature.feature_completeness is None:
        reasons.append("no StockFeature row / feature_completeness available yet")
    elif latest_feature.feature_completeness < min_completeness:
        reasons.append(
            f"feature_completeness {latest_feature.feature_completeness:.2f} below {min_completeness:.2f}"
        )

    max_staleness = settings.get("data_quality.max_staleness_trading_days", 2)
    latest_price = session.execute(
        select(StockPrice)
        .where(StockPrice.ticker == ticker, StockPrice.interval == "1d")
        .order_by(StockPrice.ts.desc())
        .limit(1)
    ).scalar_one_or_none()
    if latest_price is None or not data_freshness_ok(latest_price.ts, max_staleness):
        reasons.append("latest price data is stale or missing")

    # Only the *most recent* attempt at each critical phase counts here --
    # not an all-time failure count. A ticker that failed a few times in the
    # past (e.g. during a transient yfinance/network issue, or before a bug
    # fix landed) but has since had a clean, successful re-run should not be
    # permanently blocked from `active` status by that history.
    stale_critical_phases: list[str] = []
    for phase in _CRITICAL_PHASES:
        latest_job = session.execute(
            select(BackfillJob)
            .where(BackfillJob.ticker == ticker, BackfillJob.phase == phase)
            .order_by(BackfillJob.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest_job is not None and latest_job.status == "failed":
            stale_critical_phases.append(phase)
    if stale_critical_phases:
        reasons.append(
            f"most recent attempt at critical phase(s) failed: {', '.join(stale_critical_phases)}"
        )

    passed = len(reasons) == 0

    ticker_row = _get_or_create_ticker(session, ticker)
    ticker_row.status = TickerStatus.ACTIVE if passed else TickerStatus.FAILED
    if passed and ticker_row.activated_at is None:
        ticker_row.activated_at = dt.datetime.utcnow()
    ticker_row.screening_notes = "; ".join(reasons) if reasons else None
    session.flush()

    return passed, reasons


# ---------------------------------------------------------------------------
# TB-006: screening
# ---------------------------------------------------------------------------

def screen_ticker(info: dict, avg_daily_volume: float, atr_pct: float) -> tuple[bool, str]:
    """TB-006: screen a candidate ticker's price / liquidity / volatility
    band, plus a best-effort bankruptcy/non-equity exclusion.

    yfinance does not reliably expose a bankruptcy/liquidation flag; this
    checks `info.get('bankruptcy')` / `info.get('isDelisted')` if present
    (rare) and falls back to excluding `quoteType != 'EQUITY'` as a rough
    proxy for ETFs/indices/etc. that slipped into the candidate list. This
    is a known gap -- a dedicated corporate-actions data source would close
    it properly.
    """
    settings = get_settings()
    price_min = settings.get("ticker_universe.price_min", 10)
    price_max = settings.get("ticker_universe.price_max", 500)
    min_volume = settings.get("ticker_universe.min_avg_daily_volume", 1_000_000)
    atr_pct_min = settings.get("ticker_universe.atr_pct_min", 0.015)
    atr_pct_max = settings.get("ticker_universe.atr_pct_max", 0.08)

    price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
    if price is None:
        return False, "no price available in info"
    if not (price_min <= price <= price_max):
        return False, f"price {price} outside [{price_min}, {price_max}]"

    if avg_daily_volume < min_volume:
        return False, f"avg_daily_volume {avg_daily_volume} below floor {min_volume}"

    if not (atr_pct_min <= atr_pct <= atr_pct_max):
        return False, f"atr_pct {atr_pct} outside [{atr_pct_min}, {atr_pct_max}]"

    if info.get("bankruptcy") or info.get("isDelisted"):
        return False, "flagged as bankruptcy/delisted in info"
    quote_type = info.get("quoteType")
    if quote_type is not None and quote_type != "EQUITY":
        return False, f"quoteType {quote_type} is not EQUITY"

    return True, "passed screening"


# ---------------------------------------------------------------------------
# TB-001..004: orchestrator
# ---------------------------------------------------------------------------

def _run_backfill_impl(session: Session, ticker: str) -> None:
    ticker_row = _get_or_create_ticker(session, ticker)
    ticker_row.status = TickerStatus.BACKFILLING
    session.flush()

    price_job = _run_phase(session, ticker, "price", lambda: _phase_price(session, ticker))
    if price_job.status != "done":
        ticker_row.status = TickerStatus.FAILED
        ticker_row.screening_notes = f"critical price phase failed: {price_job.error_message}"
        session.flush()
        logger.error("Aborting backfill for %s: critical price phase failed after retries", ticker)
        return

    # Non-critical phases: failures are logged into their own BackfillJob
    # row (by _run_phase) but do not abort the run (NFR 4.2 graceful degradation).
    _run_phase(session, ticker, "fundamentals", lambda: _phase_fundamentals(session, ticker))
    _run_phase(session, ticker, "news", lambda: _phase_news(session, ticker))
    _run_phase(session, ticker, "options", lambda: _phase_options(session, ticker))
    _run_phase(session, ticker, "features", lambda: _phase_features(session, ticker))
    _run_phase(session, ticker, "model_warmup", lambda: _phase_model_warmup(session, ticker))

    passed, reasons = check_quality_gate(session, ticker)
    if passed:
        logger.info("Backfill quality gate PASSED for %s", ticker)
    else:
        logger.warning("Backfill quality gate FAILED for %s: %s", ticker, "; ".join(reasons))


def run_backfill(ticker: str, session: Session | None = None) -> None:
    """TB-001..TB-004: run the full 6-phase auto-backfill pipeline for `ticker`.

    Phases: price (critical) -> fundamentals -> news -> options -> features
    (extension point) -> model warm-up (extension point). Sets
    `TickerUniverse.status` to `backfilling` at the start, then to
    `active`/`failed` per the TB-003 quality gate (or `failed` immediately
    if the critical price phase exhausts its retries).

    If `session` is not supplied, opens and commits its own
    `session_scope()`; if supplied, the caller owns the transaction (this
    function will `flush()` but not `commit()`/`close()` it).
    """
    ticker = ticker.upper()
    if session is not None:
        _run_backfill_impl(session, ticker)
        return
    with session_scope() as db:
        _run_backfill_impl(db, ticker)


# ---------------------------------------------------------------------------
# TB-005: daily incremental update
# ---------------------------------------------------------------------------

def run_daily_incremental_update(tickers: list[str]) -> None:
    """TB-005: EOD incremental refresh for already-active tickers.

    For each ticker with TickerUniverse.status == active: fetch latest EOD
    price, mark any EarningsEvent whose date has passed as confirmed, and
    ingest today's news (reuses `_phase_news`, which is idempotent by
    headline+published_at). Rolling feature recalculation (last 60 days)
    and prediction regeneration are extension points guarded by
    try/except ImportError since `features`/`models` are owned by other
    modules and may not exist yet.
    """
    client = YFinanceClient.instance()

    with session_scope() as session:
        for raw_ticker in tickers:
            ticker = raw_ticker.upper()
            ticker_row = session.execute(
                select(TickerUniverse).where(TickerUniverse.ticker == ticker)
            ).scalar_one_or_none()
            if ticker_row is None or ticker_row.status != TickerStatus.ACTIVE:
                logger.info("Skipping incremental update for inactive/unknown ticker %s", ticker)
                continue

            try:
                df = client.get_history(ticker, period="5d", interval="1d")
                result = validate_price_dataframe(df, ticker)
                if result.is_valid and not df.empty:
                    last_ts = _to_naive_datetime(df.index[-1])
                    exists = session.execute(
                        select(StockPrice).where(
                            StockPrice.ticker == ticker, StockPrice.ts == last_ts, StockPrice.interval == "1d"
                        )
                    ).scalar_one_or_none()
                    if exists is None:
                        row = df.iloc[-1]
                        session.add(
                            StockPrice(
                                ticker=ticker,
                                ts=last_ts,
                                interval="1d",
                                open=float(row.get("Open", 0.0) or 0.0),
                                high=float(row.get("High", 0.0) or 0.0),
                                low=float(row.get("Low", 0.0) or 0.0),
                                close=float(row.get("Close", 0.0) or 0.0),
                                adj_close=float(row["Adj Close"])
                                if "Adj Close" in row and pd.notna(row["Adj Close"])
                                else None,
                                volume=float(row.get("Volume", 0.0) or 0.0),
                                source="yfinance",
                            )
                        )
                    ticker_row.last_price = float(df["Close"].iloc[-1])
                    ticker_row.last_updated = dt.datetime.utcnow()
            except Exception as e:
                logger.warning("Incremental price update failed for %s: %s", ticker, e)

            try:
                calendar = client.get_calendar(ticker) or {}
                earnings_date = _extract_earnings_date(calendar)
                if earnings_date is not None and earnings_date <= dt.date.today():
                    event = session.execute(
                        select(EarningsEvent).where(
                            EarningsEvent.ticker == ticker, EarningsEvent.earnings_date == earnings_date
                        )
                    ).scalar_one_or_none()
                    if event is not None and not event.confirmed:
                        event.confirmed = True
            except Exception as e:
                logger.warning("Incremental fundamentals/earnings check failed for %s: %s", ticker, e)

            try:
                _phase_news(session, ticker)
            except Exception as e:
                logger.warning("Incremental news ingest failed for %s: %s", ticker, e)

            # TODO: recalculate rolling features for the last 60 days and
            # regenerate predictions once features/models packages exist.
            try:
                from swing_trader.features.engineering import recalculate_recent_features  # type: ignore

                recalculate_recent_features(session, ticker, days=60)  # type: ignore[call-arg]
            except ImportError:
                pass
            except Exception as e:
                logger.warning("Rolling feature recalculation failed for %s: %s", ticker, e)

        session.flush()

"""ORM models for every table in SRS Section 2.1 / 5.

Tables marked (hypertable) are converted to TimescaleDB hypertables by
scripts/init_db.py after creation (partitioned on their timestamp column).
"""
from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from swing_trader.db.base import Base


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TickerStatus(str, enum.Enum):
    PENDING = "pending"
    BACKFILLING = "backfilling"
    ACTIVE = "active"
    FAILED = "failed"
    INACTIVE = "inactive"


class PositionStatus(str, enum.Enum):
    ACTIVE = "active"
    TRIMMED = "trimmed"
    CLOSED = "closed"


class Rating(str, enum.Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    TRIM = "Trim"
    SELL = "Sell"
    WATCH = "Watch"


class RegimeType(str, enum.Enum):
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    EARNINGS_SEASON = "earnings_season"


class AlertChannel(str, enum.Enum):
    MACOS = "macos"
    EMAIL = "email"
    SMS = "sms"
    DASHBOARD = "dashboard"


class AlertCategory(str, enum.Enum):
    SIGNAL = "signal"
    RISK = "risk"
    DATA = "data"
    EARNINGS = "earnings"


class ExitReason(str, enum.Enum):
    STOP = "stop"
    TARGET = "target"
    MANUAL = "manual"
    EARNINGS = "earnings"
    REGIME_CHANGE = "regime_change"


# ---------------------------------------------------------------------------
# 3.2 Ticker Universe & Auto-Backfill (TB)
# ---------------------------------------------------------------------------

class TickerUniverse(Base):
    __tablename__ = "ticker_universe"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), unique=True, index=True)
    status: Mapped[TickerStatus] = mapped_column(
        Enum(TickerStatus), default=TickerStatus.PENDING
    )
    sector: Mapped[str | None] = mapped_column(String(64), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True)
    avg_daily_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    added_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    activated_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    last_updated: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    feature_completeness: Mapped[float | None] = mapped_column(Float, nullable=True)
    retain_history_on_removal: Mapped[bool] = mapped_column(Boolean, default=True)
    screening_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    backfill_jobs: Mapped[list["BackfillJob"]] = relationship(back_populates="ticker_ref")


class BackfillJob(Base):
    """Tracks TB-002 progress and TB-004 retry/failure handling."""

    __tablename__ = "backfill_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(
        String(10), ForeignKey("ticker_universe.ticker"), index=True
    )
    phase: Mapped[str] = mapped_column(String(32))  # price|fundamentals|news|options|features|warmup
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|running|done|failed
    pct_complete: Mapped[float] = mapped_column(Float, default=0.0)
    records_ingested: Mapped[int] = mapped_column(Integer, default=0)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    ticker_ref: Mapped["TickerUniverse"] = relationship(back_populates="backfill_jobs")


# ---------------------------------------------------------------------------
# 3.3 Data Collection — price / fundamentals / news (hypertables)
# ---------------------------------------------------------------------------

class StockPrice(Base):
    """(hypertable, partitioned on `ts`) Daily + intraday OHLCV."""

    __tablename__ = "stock_prices"
    __table_args__ = (UniqueConstraint("ticker", "ts", "interval", name="uq_price_ticker_ts_interval"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    interval: Mapped[str] = mapped_column(String(8), default="1d")  # 1d|30m|1m
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float] = mapped_column(Float)
    dividends: Mapped[float] = mapped_column(Float, default=0.0)
    splits: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32), default="yfinance")


class StockFeature(Base):
    """(hypertable, partitioned on `ts`) Feature store — FE-001..006, 30+ columns."""

    __tablename__ = "stock_features"
    __table_args__ = (UniqueConstraint("ticker", "ts", name="uq_feature_ticker_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    ts: Mapped[dt.date] = mapped_column(Date, index=True)

    # --- FE-001 technical ---
    rsi_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    rsi_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    macd_hist: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    bb_bandwidth: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    adx_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_5: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_10: Mapped[float | None] = mapped_column(Float, nullable=True)
    roc_21: Mapped[float | None] = mapped_column(Float, nullable=True)
    obv: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_ratio_20d: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- FE-002 relative strength ---
    ret_5d_vs_spy: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_10d_vs_spy: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_21d_vs_spy: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_5d_vs_sector: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_10d_vs_sector: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret_21d_vs_sector: Mapped[float | None] = mapped_column(Float, nullable=True)
    rs_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector_momentum_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- FE-003 volatility ---
    realized_vol_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_vol_pctile: Mapped[float | None] = mapped_column(Float, nullable=True)
    atr_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    hv_iv_spread: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- FE-004 sentiment ---
    news_sentiment_3d_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    news_volume_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)
    analyst_rating_velocity: Mapped[float | None] = mapped_column(Float, nullable=True)
    options_put_call_skew: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- FE-005 fundamental ---
    pe_percentile_sector: Mapped[float | None] = mapped_column(Float, nullable=True)
    pe_percentile_history: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_interest_pct_float: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_surprise_streak: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- FE-006 macro ---
    vix_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    vix_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)
    spy_above_ema20: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    sector_breadth_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    yield_curve_10y_2y: Mapped[float | None] = mapped_column(Float, nullable=True)

    feature_completeness: Mapped[float | None] = mapped_column(Float, nullable=True)


class DailyMetric(Base):
    """Rollup metrics per ticker/day not tied to the feature store (e.g. market cap)."""

    __tablename__ = "daily_metrics"
    __table_args__ = (UniqueConstraint("ticker", "ts", name="uq_metric_ticker_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    ts: Mapped[dt.date] = mapped_column(Date, index=True)
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)
    pe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    peg_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    ps_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    float_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    put_call_ratio_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_pain_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv_rank: Mapped[float | None] = mapped_column(Float, nullable=True)
    iv_percentile: Mapped[float | None] = mapped_column(Float, nullable=True)


class NewsSentiment(Base):
    __tablename__ = "news_sentiment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    published_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    headline: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(String(128), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="yfinance")  # yfinance|newsapi|rss
    sentiment_label: Mapped[str | None] = mapped_column(String(16), nullable=True)  # positive|neutral|negative
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # -1..1
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)


# ---------------------------------------------------------------------------
# 3.5 Market Regime Detection (MR)
# ---------------------------------------------------------------------------

class RegimeHistory(Base):
    __tablename__ = "regime_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[dt.date] = mapped_column(Date, unique=True, index=True)
    regime: Mapped[RegimeType] = mapped_column(Enum(RegimeType))
    vix: Mapped[float | None] = mapped_column(Float, nullable=True)
    spy_adx: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector_breadth_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    transition_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    transition_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)


class RegimePerformance(Base):
    """MR-004 model performance tracked per regime."""

    __tablename__ = "regime_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    regime: Mapped[RegimeType] = mapped_column(Enum(RegimeType), index=True)
    as_of: Mapped[dt.date] = mapped_column(Date, index=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)


# ---------------------------------------------------------------------------
# 3.6 Predictive Modeling Engine (PM)
# ---------------------------------------------------------------------------

class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("ticker", "as_of", "model_version", name="uq_pred_ticker_asof_ver"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    as_of: Mapped[dt.date] = mapped_column(Date, index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    prob_3pct_up_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_5pct_up_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_10pct_up_21d: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_return_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    optimal_hold_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_drawdown_10d: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime: Mapped[RegimeType | None] = mapped_column(Enum(RegimeType), nullable=True)

    rating: Mapped["SignalRating | None"] = relationship(
        back_populates="prediction", uselist=False
    )


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    as_of: Mapped[dt.date] = mapped_column(Date, index=True)
    mape: Mapped[float | None] = mapped_column(Float, nullable=True)
    directional_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    calmar_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    deployed: Mapped[bool] = mapped_column(Boolean, default=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


# ---------------------------------------------------------------------------
# 3.7 Swing Signal Generation & Rating (SR)
# ---------------------------------------------------------------------------

class SignalRating(Base):
    __tablename__ = "signal_ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    as_of: Mapped[dt.date] = mapped_column(Date, index=True)
    score: Mapped[float] = mapped_column(Float)
    rating: Mapped[Rating] = mapped_column(Enum(Rating))
    suggested_entry: Mapped[float | None] = mapped_column(Float, nullable=True)
    suggested_stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    suggested_target_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    suggested_target_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    suggested_position_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    suggested_position_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    earnings_blackout: Mapped[bool] = mapped_column(Boolean, default=False)

    prediction: Mapped["Prediction"] = relationship(back_populates="rating")


# ---------------------------------------------------------------------------
# 3.1 Portfolio Management (PF)
# ---------------------------------------------------------------------------

class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    cash_balance: Mapped[float] = mapped_column(Float, default=100000.0)
    max_risk_per_trade_pct: Mapped[float] = mapped_column(Float, default=0.02)
    max_portfolio_heat_pct: Mapped[float] = mapped_column(Float, default=0.20)
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)  # TE-003
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    holdings: Mapped[list["Holding"]] = relationship(back_populates="portfolio")
    watchlist_items: Mapped[list["WatchlistItem"]] = relationship(back_populates="portfolio")
    trades: Mapped[list["Trade"]] = relationship(back_populates="portfolio")


class Holding(Base):
    __tablename__ = "holdings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    shares: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    entry_date: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    trailing_stop: Mapped[float | None] = mapped_column(Float, nullable=True)
    trailing_stop_active: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[PositionStatus] = mapped_column(Enum(PositionStatus), default=PositionStatus.ACTIVE)
    thesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="holdings")


class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    trigger_condition: Mapped[str] = mapped_column(Text)  # e.g. "RSI<30 AND prob_5pct_up_10d>0.65"
    triggered: Mapped[bool] = mapped_column(Boolean, default=False)
    added_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="watchlist_items")


# ---------------------------------------------------------------------------
# 3.9 Trade Execution & Order Management (TE)
# ---------------------------------------------------------------------------

class Trade(Base):
    """Combines TE-001 (journal) and TE-002 (exit tracking)."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), index=True)
    holding_id: Mapped[int | None] = mapped_column(ForeignKey("holdings.id"), nullable=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)

    entry_date: Mapped[dt.datetime] = mapped_column(DateTime)
    entry_price: Mapped[float] = mapped_column(Float)
    shares: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    target: Mapped[float | None] = mapped_column(Float, nullable=True)
    thesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    chart_screenshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    exit_date: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[ExitReason | None] = mapped_column(Enum(ExitReason), nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_period_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expected_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="trades")


# ---------------------------------------------------------------------------
# 3.10 Alerts & Notification System (AL)
# ---------------------------------------------------------------------------

class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[AlertCategory] = mapped_column(Enum(AlertCategory))
    ticker: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    message: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(16), default="info")  # info|warning|critical
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    dedup_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)


class NotificationLog(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id"), index=True)
    channel: Mapped[AlertChannel] = mapped_column(Enum(AlertChannel))
    sent_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# 3.14 Earnings & Economic Calendar (EC)
# ---------------------------------------------------------------------------

class EarningsEvent(Base):
    __tablename__ = "earnings_events"
    __table_args__ = (UniqueConstraint("ticker", "earnings_date", name="uq_earnings_ticker_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), index=True)
    earnings_date: Mapped[dt.date] = mapped_column(Date, index=True)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    eps_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    eps_actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    surprise_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    implied_move_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    post_earnings_drift_5d: Mapped[float | None] = mapped_column(Float, nullable=True)


class EconomicEvent(Base):
    __tablename__ = "economic_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_name: Mapped[str] = mapped_column(String(64), index=True)  # CPI|FOMC|NFP|GDP|PPI
    event_date: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    timing: Mapped[str | None] = mapped_column(String(16), nullable=True)  # pre-market|post-market
    historical_reaction_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="rss")


# ---------------------------------------------------------------------------
# 7.8 System Health & Monitoring
# ---------------------------------------------------------------------------

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_name: Mapped[str] = mapped_column(String(64), index=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    records_processed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ApiRateLimitLog(Base):
    __tablename__ = "api_rate_limit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    calls_made: Mapped[int] = mapped_column(Integer, default=0)
    limit_per_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    window_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

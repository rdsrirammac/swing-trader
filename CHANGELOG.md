# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versioning
follows [SemVer](https://semver.org/) once tagged releases start.

## [Unreleased]

Nothing yet -- this is where your next PR's summary goes, under
"Added"/"Changed"/"Fixed" as appropriate.

## [0.1.0] - 2026-08-04

Initial build. Implements all 15 functional areas of
`docs/SRS/Swing_Trading_SRS_v1.0.md` (as refined by
`docs/SRS/SRS_Refinement_v1.1_yfinance.md`'s yfinance-native architecture):

### Added
- PostgreSQL + TimescaleDB schema (21 tables) via SQLAlchemy, with hypertable conversion for `stock_prices` / `stock_features`.
- yfinance-native data collection layer (price, fundamentals, options, news, analyst data) with disk caching, rate limiting, and retry/backoff; NewsAPI, SEC EDGAR, and RSS economic-calendar clients as secondary sources.
- Feature engineering: technical indicators (pandas-ta), relative strength, volatility, sentiment (FinBERT with VADER fallback), fundamental, and macro features.
- Market regime detection (Strong Trend / Weak Trend / Range-Bound / High Volatility / Earnings Season) with documented precedence rules.
- Predictive modeling: LightGBM, ARIMA-X, LSTM, Random Forest base models, Ridge-meta-learner stacking ensemble, Optuna hyperparameter tuning, MLflow tracking, and the full PM-004 10-step daily self-tuning pipeline.
- Signal generation & the SR-002 rating algorithm, transcribed exactly from the SRS pseudocode, with configurable rating-cutoff thresholds.
- Full risk management: position sizing, portfolio heat, sector concentration, correlation rejection, drawdown circuit breakers, volatility-adjusted sizing, trailing stops.
- Portfolio management and the 6-phase auto-backfill orchestrator with progress tracking, retry/backoff, and the TB-003 quality gate.
- Trade journal, paper trading simulator, and an inert broker interface (Alpaca skeleton) for future live/paper broker integration.
- Alerts: rules engine (dedup, quiet hours, rating thresholds), macOS notifications, email.
- Walk-forward backtesting engine, Monte Carlo simulation, strategy comparison, out-of-sample splitting.
- Performance analytics, attribution analysis, behavioral bias detection, correlation/concentration/diversification scoring.
- Earnings and economic calendar integration with avoidance rules.
- Streamlit dashboard (portfolio summary, regime, alerts, positions, watchlist, model performance, sector rotation, correlation, backtest, trade journal, system health) plus ticker-detail/backtest/journal sub-pages.
- Admin CLI (`python -m swing_trader.cli ...`) covering all documented admin operations.
- launchd job scripts + plists matching the SRS's daily schedule, plus install/uninstall helpers.
- 40 unit + integration tests (pytest, in-memory SQLite, mocked yfinance for the backfill integration test).
- GitHub Actions CI (lint, type-check, tests, schema-creation check on 3.11/3.12).
- Issue templates, PR template, label taxonomy, and a seeded backlog of known integration gaps.

### Known limitations
See `ROADMAP.md` "Known Integration Gaps" -- most notably, the backfill
pipeline's Phase 5 (feature warm-up) doesn't yet call the feature-store
orchestrator with the correct arguments, so newly-backfilled tickers
currently fail the TB-003 quality gate on `feature_completeness` even with
good price data. This is the top-priority next fix.

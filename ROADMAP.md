# Roadmap

Status of the original 8-phase implementation plan from
`docs/SRS/Swing_Trading_SRS_v1.0.md` Section 8, plus what to do next.

## Where things stand (v0.1)

All 15 functional areas (Section 3 of the SRS) have a working
implementation: database schema, yfinance-native data collection, feature
engineering, market regime detection, a real (LightGBM/ARIMA-X/LSTM/RF)
ensemble with Optuna tuning and MLflow tracking, the SR-002 rating
algorithm transcribed exactly, full risk management (position sizing,
heat, sector concentration, correlation, drawdown circuit breakers,
trailing stops), portfolio management + 6-phase auto-backfill, trade
journal + paper trading, alerts (macOS/email, rules engine), walk-forward
backtesting + Monte Carlo, performance analytics + correlation analysis,
earnings/economic calendar, a Streamlit dashboard, an admin CLI, and
launchd scheduling. 40 unit/integration tests pass against an in-memory
SQLite database, and the schema has been verified to create correctly.

This maps roughly to Phases 1-6 of the SRS's original 24-week plan,
compressed. Phases 7-8 (polish/automation, performance profiling, CoreML
optimization, load testing at 50 tickers) are genuinely "run it for a
while on the real Mac Mini and see" work -- not something to fake.

## Known Integration Gaps

These were identified during the build and are seeded as GitHub issues
(`scripts/seed_github_issues.sh`) -- check there for the live/closed
status rather than this static list:

1. **Backfill Phase 5 (features) signature mismatch** -- `_phase_features()`
   calls `build_feature_row()` with the wrong arguments; feature rows never
   actually get created during backfill today, so the TB-003 quality gate
   always fails on `feature_completeness`. This is the single
   highest-priority fix -- until it's resolved, no ticker will ever reach
   `status=active` through the normal backfill flow. See
   `tests/integration/test_backfill_pipeline.py`'s docstring for the
   regression test that documents this.
2. No persisted backtest run history (dashboard shows on-demand results only).
3. Historical short interest unavailable via yfinance (current value only).
4. FinBERT vs. VADER sentiment quality not yet evaluated on real data.
5. No Alembic migrations (raw `create_all()` only).
6. Regime inputs (SPY ADX/VIX/breadth) only populate if SPY is itself a
   tracked ticker; no independent market-snapshot table.
7. Alpaca broker integration is an inert skeleton (by design, see
   `execution/alpaca_broker.py` docstring -- financial-safety policy).
8. SMS alert channel unimplemented (logs and no-ops).
9. No clean 2Y Treasury yield source for the yield-curve feature.
10. Sector ETFs (XLK/XLF/.../XLC) must be manually added as tickers for
    sector breadth/rotation to populate.
11. Economic calendar doesn't capture precise pre/post-market timing.
12. `analytics.planned_trades_per_week` config key doesn't exist yet
    (hardcoded default of 3 in `analytics/behavioral.py`).
13. No stop-loss change history -- `PA-004`'s stop-violation detection is
    an approximation.

## Next steps (priority order)

1. Fix the backfill Phase 5 wiring (#1 above) -- unblocks everything downstream.
2. Run `make install && make infra-up && make init-db` on the real Mac
   Mini and backfill a handful of real tickers end-to-end; fix whatever
   yfinance edge cases show up (this was explicitly anticipated in
   `docs/SRS/SRS_Refinement_v1.1_yfinance.md` Section 9 -- "you will
   discover edge cases only by building").
3. Paper-trade for a few weeks via `execution/paper_trading.py` before
   trusting any signal.
4. Revisit `PM-004`'s hyperparameter/feature-selection thresholds once
   there's real walk-forward performance data -- the current defaults are
   the SRS's starting guesses, not tuned values.
5. Work through the remaining backlog issues as they become relevant to
   your actual trading, not necessarily in the order listed above.

## Original 8-phase plan (for reference)

| Phase | Weeks | Focus | Status |
|---|---|---|---|
| 1 | 1-3 | Foundation: DB, scaffolding, price collection, CLI | Done |
| 2 | 4-6 | Features & sentiment | Done (FinBERT/VADER fallback wired, not yet evaluated on real data) |
| 3 | 7-10 | Modeling: LightGBM/ARIMA-X/LSTM, ensemble, backtesting, MLflow | Done (untrained on real data yet) |
| 4 | 11-13 | Signals & risk | Done |
| 5 | 14-16 | Portfolio, alerts, earnings calendar, correlation | Done |
| 6 | 17-19 | Dashboard & analytics | Done |
| 7 | 20-22 | launchd automation, paper trading, reports, docs, backups | Done (scripts written, not yet run on a real schedule) |
| 8 | 23-24 | Performance profiling, CoreML, query optimization, 50-ticker load test | Not started -- needs a real running system first |

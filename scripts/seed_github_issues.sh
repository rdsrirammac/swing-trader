#!/usr/bin/env bash
# Seed the initial backlog of known gaps / follow-up work as GitHub issues.
# These are the concrete items identified while building v0.1 (see
# ROADMAP.md "Known Integration Gaps" for the full writeup of each).
#
# Requires: `gh auth login` already run, repo created & pushed.
# Usage: bash scripts/seed_github_issues.sh
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI ('gh') not found. Install it (https://cli.github.com) and run 'gh auth login' first." >&2
  echo "You can also just copy the items from ROADMAP.md into issues by hand." >&2
  exit 1
fi

create_issue() {
  local title="$1" body="$2" labels="$3"
  echo "Creating: $title"
  gh issue create --title "$title" --body "$body" --label "$labels" || \
    echo "  (skipped -- may already exist, or label(s) not yet created: run scripts/seed_github_labels.sh first)"
}

create_issue \
  "Wire backfill Phase 5 (features) to the real build_feature_row signature" \
  "swing_trader/portfolio/backfill.py's _phase_features() calls
build_feature_row(session, ticker), but the actual implementation in
swing_trader/features/engineering.py takes many more required arguments
(price_history, spy_history, sector_history, vix_history, info, news_rows,
recommendations_df, options_chain, sector_etf_histories). The mismatched
call raises TypeError, which is caught and logged as a non-critical
failure -- so no StockFeature row is ever created during backfill today,
which means the TB-003 quality gate can never pass (feature_completeness
always unavailable) and every newly-backfilled ticker ends up
status=failed even with perfect price data.

See tests/integration/test_backfill_pipeline.py's module docstring for a
regression test that currently asserts this failure mode -- update it to
assert status=ACTIVE once this is fixed.

Fix: have backfill.py assemble the required inputs (it already fetches
price/info/news/options during phases 1-4 -- pass them through) and call
build_feature_row with the correct signature." \
  "bug,srs:TB,srs:FE"

create_issue \
  "Add a BacktestRun table so backtest history persists across sessions" \
  "swing_trader/backtest/engine.py's run_backtest() returns results in-memory
only. The dashboard's 'Backtest Results' panel (DV-001) looks for
swing_trader.backtest.engine.get_latest_backtest_summary(), which doesn't
exist, so it always shows a 'run one on demand' fallback.

Add a BacktestRun model (strategy, start, end, metrics JSON, created_at)
and have run_backtest() optionally persist to it, plus a
get_latest_backtest_summary(session) query for the dashboard." \
  "enhancement,srs:BT,srs:DV"

create_issue \
  "Historical short interest via SEC EDGAR (yfinance only gives current shortRatio)" \
  "Per SRS_Refinement_v1.1 Section 3, yfinance has no historical short
interest -- only the current shortRatio. FE-005's
short_interest_pct_float feature is therefore point-in-time only, with no
trend signal (short_interest_trend from the original SRS FE-005 spec is
unimplemented). Exchange short-interest data is published bi-weekly and
could be scraped/ingested via a dedicated collector." \
  "enhancement,srs:DC,srs:FE"

create_issue \
  "Evaluate FinBERT vs VADER sentiment quality on real news backfill" \
  "features/sentiment.py lazily loads FinBERT (transformers) and falls back
to VADER if unavailable. Once running with real news data, compare
sentiment_score distributions/labels between the two on a sample of
tickers and decide whether FinBERT's extra latency/dependency weight is
worth it for this system's news volume, or whether VADER is good enough." \
  "enhancement,srs:FE"

create_issue \
  "Set up Alembic migrations instead of raw create_all()" \
  "scripts/init_db.py currently uses Base.metadata.create_all(), which has
no migration/versioning story -- any future schema change requires a
manual ALTER or a full drop/recreate. Add Alembic (already in
requirements.txt) with an initial migration matching the current schema,
and document the migration workflow in CONTRIBUTING.md." \
  "infra,good-first-issue"

create_issue \
  "Add a dedicated MarketSnapshot table for SPY/VIX regime inputs" \
  "models/pipeline.py's build_context_from_db() currently derives regime
inputs (spy_adx, vix_level, sector_breadth_pct) from SPY's own
StockFeature row IF SPY happens to be a tracked ticker in the portfolio,
else they're left None. A dedicated MarketSnapshot table (independent of
any traded ticker) would make regime detection reliable even for
portfolios that never add SPY/QQQ/sector ETFs as tracked tickers." \
  "enhancement,srs:MR"

create_issue \
  "Alpaca broker integration behind an explicit human-confirmation step" \
  "execution/alpaca_broker.py is an intentionally inert skeleton (every
method raises NotImplementedError) per TE-004's 'future integration'
scope and this build's trading-safety policy (no automated order
placement). When ready to implement real paper/live trading, design an
explicit confirmation UX (dashboard button + CLI --confirm flag) rather
than wiring it into any scheduled job." \
  "enhancement,srs:TE"

create_issue \
  "SMS alert channel (Twilio) is a stub" \
  "notify/engine.py's dispatch_alert() logs 'SMS channel not implemented'
for any alert when alerts.channels.sms is enabled. AL-005 lists SMS/Push
as 'Critical' priority for drawdown/circuit-breaker alerts -- implement a
Twilio (or similar) integration, config keys already stubbed in
config/api_keys.yaml.example under 'sms:'." \
  "enhancement,srs:AL"

create_issue \
  "2-year Treasury yield source for the yield curve feature" \
  "features/macro.py's compute_macro_features() accepts a
treasury_2y_proxy parameter that nothing currently supplies -- yfinance
has no clean 2Y yield ticker. Options: FRED API (free, no key for basic
series), or a proxy like ^IRX (13-week T-bill, imperfect). Wire one of
these into the daily data collection job and pass it through to
yield_curve_10y_2y." \
  "enhancement,srs:FE"

create_issue \
  "Auto-track the 11 SPDR sector ETFs for sector breadth / rotation" \
  "MR-001's sector_breadth_pct and FE-002's sector_momentum_rank both need
XLK/XLF/XLE/XLY/XLP/XLV/XLI/XLB/XLRE/XLU/XLC price history. Right now
these only get populated if a user manually adds each ETF as a tracked
ticker. Consider auto-seeding them into ticker_universe on first run (make
install / init-db) with a distinct 'reference' flag so they don't count
against the 50-ticker portfolio limit (NFR 4.3)." \
  "enhancement,srs:MR,srs:FE"

create_issue \
  "Economic calendar: precise pre/post-market timing" \
  "data/economic_calendar.py's EconomicCalendarClient parses Fed/BLS RSS
feeds for CPI/PPI/NFP/GDP/FOMC headlines but doesn't reliably capture
exact release time-of-day (EC-003's 'pre-market vs post-market timing').
Cross-reference against the official BLS/Fed release calendar pages (fixed
schedule, e.g. CPI always 8:30am ET) to fill in `timing` accurately." \
  "enhancement,srs:EC"

create_issue \
  "Add analytics.planned_trades_per_week to settings.yaml" \
  "analytics/behavioral.py's overtrading_ratio() defaults to 3 planned
trades/week because that key doesn't exist in config/settings.yaml yet.
Small config addition + docs update." \
  "infra,good-first-issue,srs:PA"

create_issue \
  "Add a stop_loss_history table for accurate stop-violation detection" \
  "analytics/behavioral.py's stop_violation_rate() approximates 'did the
trader move their stop away from the original risk level' by comparing
Trade.stop_loss at open vs. the linked Holding.stop_loss at close time --
there's no true history of intermediate stop adjustments. A
stop_loss_history table (holding_id, changed_at, old_stop, new_stop)
written by portfolio/manager.py's adjust_stop() would make this exact." \
  "enhancement,srs:PA,srs:PF"

echo ""
echo "Backlog seeded. Run scripts/seed_github_labels.sh first if labels don't exist yet."

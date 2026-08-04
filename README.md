# swing-trader

A swing trading (3-21 day holds) stock analysis and prediction system:
yfinance-native data collection, technical/sentiment/fundamental feature
engineering, a regime-aware ML ensemble (LightGBM + ARIMA-X + LSTM,
stacked), a rating/signal engine, full risk management, portfolio tracking
with auto-backfill, backtesting, alerts, and a Streamlit dashboard.

Built from `docs/SRS/Swing_Trading_SRS_v1.0.md` (81 requirements across 15
functional areas) as refined by `docs/SRS/SRS_Refinement_v1.1_yfinance.md`
(yfinance-first architecture, $0/month, one optional free API key).

**Not financial advice.** Personal research/education tool. See `LICENSE`.

## Status

v0.1 -- all 15 functional areas have a real implementation and 40 tests
pass, but this has **not yet been run against real market data end-to-end**
on a live Postgres/TimescaleDB instance. There is one known
high-priority integration gap (backfill's feature warm-up phase) that
blocks tickers from reaching `active` status until fixed -- see
`ROADMAP.md` before you start using this for anything real.

## Quickstart

```bash
git clone <your-fork-url> swing-trader && cd swing-trader
make install                 # venv + deps + .env / api_keys.yaml scaffolding
# edit .env and config/api_keys.yaml (NewsAPI key is optional but recommended)
make infra-up                 # docker compose: TimescaleDB + Redis + MLflow
make init-db                    # create tables + hypertables
make add-ticker TICKER=AAPL      # backfill + activate a ticker
make dashboard                     # streamlit run ...
```

Requires Docker (for TimescaleDB/Redis/MLflow) and Python 3.11+. Designed
for a Mac Mini Pro per the SRS, but nothing in the stack is macOS-specific
except the `pync` notification channel (which no-ops gracefully elsewhere)
and the `launchd` scheduling (use cron/systemd on Linux instead --
`scheduler/run_*.py` scripts are plain Python entrypoints either way).

## What's actually implemented

Full detail in `ROADMAP.md`; summary:

- **Data**: yfinance client (cached, rate-limited, retried) for
  price/fundamentals/options/news/analyst data; NewsAPI (optional key) for
  90-day news history; SEC EDGAR and RSS economic-calendar clients, both
  keyless.
- **Features**: ~35 technical/relative-strength/volatility/sentiment
  (FinBERT with VADER fallback)/fundamental/macro columns per ticker/day.
- **Regime detection**: Strong Trend / Weak Trend / Range-Bound / High
  Volatility / Earnings Season, config-driven thresholds.
- **Modeling**: LightGBM + ARIMA-X + LSTM + Random Forest base models,
  Ridge-meta-learner stacking, Optuna tuning, MLflow tracking, the full
  PM-004 10-step daily self-tuning pipeline.
- **Signals & risk**: the SR-002 rating algorithm transcribed exactly from
  the SRS pseudocode; position sizing, portfolio heat, sector
  concentration, correlation rejection, drawdown circuit breakers,
  trailing stops (RM-001..007).
- **Portfolio**: multi-portfolio CRUD, watchlist with a small safe
  trigger-condition evaluator, 6-phase auto-backfill with progress
  tracking and a data-quality gate.
- **Execution**: trade journal, paper trading simulator, an intentionally
  inert broker interface (see "Financial safety" below).
- **Alerts**: rules engine (dedup, quiet hours, rating thresholds), macOS
  Notification Center, email; SMS is a stub (see `ROADMAP.md`).
- **Backtesting**: walk-forward splitter, bar-level trade simulation with
  commission/slippage, Monte Carlo probability-of-ruin/drawdown
  distribution, strategy comparison, out-of-sample splitting.
- **Analytics**: the same metrics engine shared between backtests and live
  trades, attribution (ticker/sector/regime/rating/month), behavioral bias
  detection, correlation matrix + diversification score.
- **Dashboard**: Streamlit, single-page layout matching the SRS's DV-001
  mockup, plus ticker-detail/backtest/journal sub-pages.
- **CLI**: `python -m swing_trader.cli --help` for every admin operation.
- **Scheduling**: launchd plists + scripts matching the SRS's daily
  schedule (pre-market, midday, EOD, predict, weekly retrain, backup).

## Verification

What's been checked in this environment (no real network/Postgres
available at build time):

- `python -m py_compile` across every module -- no syntax errors.
- Full 21-table schema creates successfully against SQLite
  (`Base.metadata.create_all()`), confirming the SQLAlchemy models are
  structurally valid.
- 40 pytest unit/integration tests pass, including: the SR-002 rating
  algorithm against hand-computed expected scores, risk-management math
  (position sizing, heat bands, sector concentration, correlation
  rejection), the regime classifier against all five regime types, the
  backtest engine (trade simulation, metrics, walk-forward splitting,
  out-of-sample splitting), and an integration test running the real
  6-phase backfill orchestrator against a mocked `YFinanceClient`.

What you should verify yourself before trusting this with real trades:

```bash
make infra-up && make init-db     # real TimescaleDB, not SQLite
make add-ticker TICKER=AAPL         # real yfinance calls -- watch for edge cases
make test                             # should still be 40/40 on your machine
make lint                               # ruff
```

Then read `ROADMAP.md` "Known Integration Gaps" -- fix #1 (backfill Phase
5) before relying on the auto-backfill quality gate, and paper-trade for a
while (`execution/paper_trading.py`) before anything else.

## Managing future work (this was the other half of the ask)

This repo is set up so ongoing fixes/enhancements don't pile up as loose
chat history:

- **Issue templates** (`.github/ISSUE_TEMPLATE/`) for bugs, SRS-mapped
  enhancements (dropdown of the 15 functional areas), and out-of-scope
  feature requests.
- **Labels** (`.github/labels.yml` + `scripts/seed_github_labels.sh`) --
  one `srs:XX` label per functional area so the backlog can be filtered by
  requirement area, plus `bug`/`enhancement`/`feature-request`/`infra`/`good-first-issue`.
- **A pre-seeded backlog** (`scripts/seed_github_issues.sh`) of the 13
  concrete gaps found while building this (also listed in `ROADMAP.md`) --
  run it once after creating the repo so you start with real, actionable
  issues instead of a blank tracker.
- **CI** (`.github/workflows/ci.yml`) -- lint + tests + a schema-creation
  check on every push/PR.
- **PR template** with an SRS-requirement-touched field and a checklist.
- **CONTRIBUTING.md** -- branching/commit conventions, where modules live,
  the config-not-hardcoding rule, the financial-safety rule.
- **CHANGELOG.md** -- Keep-a-Changelog format, `[Unreleased]` section ready
  for your next PR.

### Getting this onto GitHub

```bash
cd swing-trader
git init -b main            # already done for you -- see below
gh repo create swing-trader --private --source=. --remote=origin   # or create on github.com and add the remote manually
git push -u origin main
bash scripts/seed_github_labels.sh    # requires `gh auth login` first
bash scripts/seed_github_issues.sh
```

(This build already ran `git init` and made an initial commit locally --
see "What's in this delivery" below. You just need to create the remote
and push.)

## Financial safety

`src/swing_trader/execution/broker_base.py` and `alpaca_broker.py` are
intentionally inert -- every method raises `NotImplementedError`. No code
path in this repository places a real order. Paper trading
(`execution/paper_trading.py`) simulates fills against stored price data
only. Read `ROADMAP.md`'s note on broker integration before changing this.

## Project layout

See `docs/ARCHITECTURE.md` for the full module map and design-decision
notes. Directory structure follows the SRS's Section 6.2 layout with one
change: source lives under `src/swing_trader/` as a proper installable
package rather than a loose `src/` tree.

## Docs

- `docs/SRS/Swing_Trading_SRS_v1.0.md` -- the original requirements spec.
- `docs/SRS/SRS_Refinement_v1.1_yfinance.md` -- why/how yfinance became the primary data source.
- `docs/ARCHITECTURE.md` -- module map and design decisions.
- `ROADMAP.md` -- phase status + known integration gaps + next steps.
- `CONTRIBUTING.md` -- how to work on this repo.
- `CHANGELOG.md` -- release notes.

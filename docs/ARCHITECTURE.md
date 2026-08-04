# Architecture

See `docs/SRS/Swing_Trading_SRS_v1.0.md` for the full requirements and
`docs/SRS/SRS_Refinement_v1.1_yfinance.md` for why yfinance is the primary
data source (free, no API key, sufficient breadth for a 5-50 ticker
personal system).

## Data flow

```
yfinance (primary) + NewsAPI/SEC EDGAR/RSS (secondary)
        |
        v
data/ collectors (cached, rate-limited, retried)
        |
        v
db/models.py  (PostgreSQL + TimescaleDB)
        |
        v
features/  ---->  models/ (regime, ensemble, daily pipeline)  ---->  predictions
        |                                                                |
        v                                                                v
portfolio/ (backfill orchestrator, CRUD)                    signals/ (rating, risk, bracket orders)
        |                                                                |
        v                                                                v
execution/ (trade journal, paper trading)  <-------------------  notify/ (alerts)
        |
        v
analytics/ + backtest/  ---->  dashboard/ (Streamlit) + cli.py
```

## Module map

| Package | SRS section | Owns |
|---|---|---|
| `db/` | 2.1, 5 | SQLAlchemy models for all 21 tables |
| `data/` | 3.3 | yfinance/NewsAPI/SEC-EDGAR/RSS clients, validation |
| `features/` | 3.4 | Technical/relative-strength/volatility/sentiment/fundamental/macro features |
| `models/` | 3.5, 3.6 | Regime detection, base ML models, stacking ensemble, Optuna tuning, MLflow, the daily pipeline |
| `signals/` | 3.7, 3.8 | Rating algorithm, signal generation, risk management, earnings blackout |
| `portfolio/` | 3.1, 3.2 | Portfolio CRUD, watchlist, 6-phase auto-backfill |
| `execution/` | 3.9 | Trade journal, paper trading, broker interface (inert) |
| `notify/` | 3.10 | Alert rules engine, macOS/email notifications |
| `backtest/` | 3.11 | Walk-forward engine, Monte Carlo |
| `analytics/` | 3.12, 3.13 | Performance metrics, attribution, behavioral analytics, correlation/concentration |
| `calendar_data/` | 3.14 | Earnings & economic calendar (named to avoid shadowing stdlib `calendar`) |
| `dashboard/` | 3.15 | Streamlit app |
| `cli.py` | NFR 4.6 | Admin CLI |
| `scheduler/` | 6.3 | launchd-invoked job scripts + plists |

## Design decisions worth knowing

- **Config over hardcoding.** Every numeric threshold tied to an SRS
  requirement lives in `config/settings.yaml` / `config/regimes.yaml`,
  read via `get_settings().get("a.b.c", default)`. Exception: the SR-002
  rating algorithm's own internal pseudocode constants (momentum/CI-width
  bands) aren't in the SRS's config table, so they're module constants in
  `signals/rating.py`, documented there.
- **Defensive cross-module imports during the initial build.** This
  codebase was built by several workstreams in parallel against a shared
  DB schema and config contract. Some integration points (documented in
  `ROADMAP.md` "Known Integration Gaps") have signature mismatches that
  degrade gracefully (logged + skipped) rather than crashing. The
  CLI/dashboard/scheduler layer wraps sibling-package calls in
  `hasattr`/try-except checks for this reason -- that pattern is fine to
  keep going forward for genuinely optional integrations, but new code
  within a single well-defined module shouldn't need it.
- **`compute_backtest_metrics` is shared** between `backtest/engine.py`
  and `analytics/performance.py` so live-trade and simulated-backtest
  metrics are computed identically -- never re-derive the same formulas
  twice.
- **No live mark-to-market pricing** in risk/portfolio calculations
  (`portfolio_heat`, `sector_concentration`, `diversification_score` all
  use `entry_price` as a stand-in for current value). This is a documented
  simplification, not an oversight -- wiring in real-time quotes is a
  reasonable enhancement once the system is running live.
- **Financial safety:** `execution/broker_base.py` and
  `execution/alpaca_broker.py` are deliberately inert. No code path in
  this repository places a real trade. See their module docstrings.

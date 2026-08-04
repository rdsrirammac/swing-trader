# Contributing

This is a personal swing-trading system, but it's built to be maintained
like a real project: issues track work, PRs go through CI, and the SRS
docs stay the source of truth for *why* something works the way it does.

## Filing work

Use the issue templates (`.github/ISSUE_TEMPLATE/`) -- don't just leave a
comment somewhere and forget it:

- **Bug** -- something is broken (crash, wrong number, bad data).
- **Enhancement** -- an SRS requirement that's partially or not yet
  implemented. Tag it with the right `srs:XX` label (see the dropdown in
  the template) so the backlog stays organized by functional area.
- **Feature request** -- something new, outside the original SRS.

Check `ROADMAP.md` first -- the "Known Integration Gaps" section lists
things already identified during the initial build; your issue might
already be tracked there (or worth linking to it).

## Branching & commits

- Branch off `main`: `git checkout -b fix/rm-002-heat-rounding` or
  `feat/srs-ec-003-precise-timing` (short, descriptive, referencing the
  requirement ID when there is one).
- Small, focused commits. Conventional-commit-style prefixes are
  encouraged but not enforced: `fix:`, `feat:`, `docs:`, `test:`,
  `refactor:`, `chore:`.
- Open a PR against `main` using the PR template -- fill in the SRS
  requirement(s) touched and how you tested it.

## Local setup

```bash
make install        # venv + deps + .env / api_keys.yaml scaffolding
make infra-up        # docker compose: TimescaleDB + Redis + MLflow
make init-db          # create tables + hypertables
make test              # pytest, in-memory SQLite, no external services
make lint                # ruff
```

`make test` runs against an in-memory SQLite database and never touches
TimescaleDB or the network -- it should be fast and safe to run on every
commit. Anything that genuinely needs Postgres/TimescaleDB (hypertable
conversion, real backfill against yfinance) is a manual verification step,
documented in `README.md`'s "Verification" section, not part of the
default test suite.

## Where things live

See `docs/ARCHITECTURE.md` for the module map. Short version:

- `src/swing_trader/data/` -- external data clients (yfinance, NewsAPI, SEC EDGAR, econ calendar)
- `src/swing_trader/features/` -- technical/fundamental/sentiment/macro feature engineering
- `src/swing_trader/models/` -- regime detection, base models, ensemble, tuning, the daily pipeline
- `src/swing_trader/signals/` -- rating algorithm, signal generation, risk management, earnings blackout
- `src/swing_trader/portfolio/` -- portfolio CRUD, auto-backfill orchestrator, watchlist
- `src/swing_trader/execution/` -- trade journal, paper trading, broker interface (inert by design)
- `src/swing_trader/notify/` -- alert rules engine, macOS/email notifications
- `src/swing_trader/backtest/` -- walk-forward engine, Monte Carlo
- `src/swing_trader/analytics/` -- performance metrics, attribution, correlation, behavioral analytics
- `src/swing_trader/calendar_data/` -- earnings & economic calendar (named to avoid shadowing stdlib `calendar`)
- `src/swing_trader/dashboard/` -- Streamlit app
- `src/swing_trader/cli.py` -- admin CLI
- `scheduler/` -- launchd-invoked job scripts + the plists themselves

Every module's docstring cites the SRS requirement ID(s) it implements
(e.g. "RM-001..007") -- when in doubt about *why* code does something a
particular way, check the docstring first, then `docs/SRS/`.

## Config, not hardcoding

Numeric thresholds tied to an SRS requirement (risk percentages, ATR
multiples, rating cutoffs, backfill windows, ...) belong in
`config/settings.yaml` / `config/regimes.yaml`, read via
`get_settings().get("a.b.c", default)` -- not hardcoded in Python. The
SR-002 rating algorithm's own internal pseudocode constants (the 0.05/0.02
momentum bands, etc.) are a deliberate exception -- see the comment at the
top of `signals/rating.py` for why.

## Financial-safety rule

`execution/broker_base.py` / `execution/alpaca_broker.py` must never place
real trades from this codebase without an explicit human-in-the-loop
confirmation step. Don't wire broker calls into any scheduled job.

.PHONY: install infra-up infra-down init-db add-ticker backfill backfill-features predict list-predictions backtest \
        dashboard test lint fmt typecheck run-premarket run-midday run-eod \
        run-predict run-weekly-retrain run-backup schedule-install schedule-uninstall clean

PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

## Single-command setup (NFR 4.6)
install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .
	mkdir -p logs backups cache
	cp -n .env.example .env || true
	cp -n config/api_keys.yaml.example config/api_keys.yaml || true
	@echo "Install complete. Edit .env and config/api_keys.yaml, then: make infra-up init-db"

infra-up:
	docker compose up -d
	@echo "Waiting for Postgres to become healthy..."
	sleep 5

infra-down:
	docker compose down

init-db:
	$(BIN)/python scripts/init_db.py

## --- CLI shortcuts (src/cli.py) ---
add-ticker:
	$(BIN)/python -m swing_trader.cli add-ticker $(TICKER)

remove-ticker:
	$(BIN)/python -m swing_trader.cli remove-ticker $(TICKER)

list-portfolio:
	$(BIN)/python -m swing_trader.cli list-portfolio

backfill:
	$(BIN)/python -m swing_trader.cli backfill $(TICKER)

## One-time bulk historical feature backfill (run once per ticker after
## add-ticker, so `predict` has enough training history immediately instead
## of waiting weeks for the nightly EOD job to accumulate it). TICKERS is
## optional space-separated list; defaults to all active tickers.
backfill-features:
	$(BIN)/python scripts/backfill_historical_features.py $(TICKERS)

predict:
	$(BIN)/python -m swing_trader.cli predict

list-predictions:
	$(BIN)/python -m swing_trader.cli list-predictions $(if $(TICKERS),--tickers "$(TICKERS)",)

backtest:
	$(BIN)/python -m swing_trader.cli backtest $(ARGS)

dashboard:
	$(BIN)/streamlit run src/swing_trader/dashboard/app.py

## --- Scheduled jobs (mirrors launchd plists in scheduler/launchd) ---
run-premarket:
	$(BIN)/python scheduler/run_premarket.py

run-midday:
	$(BIN)/python scheduler/run_midday.py

run-eod:
	$(BIN)/python scheduler/run_eod.py

run-predict:
	$(BIN)/python scheduler/run_predict.py

run-weekly-retrain:
	$(BIN)/python scheduler/run_weekly_retrain.py

run-backup:
	$(BIN)/python scheduler/run_backup.py

schedule-install:
	bash scheduler/install_launchd.sh

schedule-uninstall:
	bash scheduler/uninstall_launchd.sh

## --- Quality gates ---
test:
	$(BIN)/pytest --cov=src --cov-report=term-missing

lint:
	$(BIN)/ruff check src tests

fmt:
	$(BIN)/ruff format src tests

typecheck:
	$(BIN)/mypy src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage

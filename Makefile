.PHONY: install infra-up infra-down init-db add-ticker backfill predict backtest \
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

predict:
	$(BIN)/python -m swing_trader.cli predict

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

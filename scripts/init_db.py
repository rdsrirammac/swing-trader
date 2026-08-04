#!/usr/bin/env python3
"""Create all tables and convert time-series tables to TimescaleDB hypertables.

Usage:
    python scripts/init_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402

from swing_trader.db.base import Base, get_engine  # noqa: E402
from swing_trader.db import models  # noqa: E402,F401  (import registers all tables)

# Tables that should become TimescaleDB hypertables, partitioned on this column.
HYPERTABLES = {
    "stock_prices": "ts",
    "stock_features": "ts",
}


def main() -> None:
    engine = get_engine()

    print("Creating tables from SQLAlchemy metadata...")
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"WARNING: could not enable timescaledb extension: {e}")
            print("Continuing with plain PostgreSQL tables.")
            return

        for table, time_col in HYPERTABLES.items():
            try:
                conn.execute(
                    text(
                        "SELECT create_hypertable(:table, :col, "
                        "if_not_exists => TRUE, migrate_data => TRUE)"
                    ),
                    {"table": table, "col": time_col},
                )
                print(f"  -> {table} is now a hypertable (partitioned on {time_col})")
            except Exception as e:  # pragma: no cover
                print(f"  -> WARNING: could not convert {table} to hypertable: {e}")

    print("Database initialization complete.")


if __name__ == "__main__":
    main()

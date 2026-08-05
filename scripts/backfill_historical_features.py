#!/usr/bin/env python3
"""One-time bulk historical feature backfill.

`portfolio.backfill.run_backfill` (via `_phase_features`) only ever writes
ONE day's `StockFeature` row per ticker -- today's. The nightly EOD job
(`portfolio.backfill.run_daily_incremental_update`) rolls that forward one
row at a time via `features.engineering.recalculate_recent_features`, so a
freshly-backfilled ticker would otherwise need ~2-3 months of live nightly
runs before `models.pipeline.build_context_from_db` has enough history
(`modeling.walk_forward_test_days`, default 60, plus the prediction
horizon) to actually train anything.

This script closes that gap once: for each given ticker (default: every
`active` ticker), it calls `recalculate_recent_features(session, ticker,
days=N)` across as much of that ticker's already-downloaded price history
as is available (default: all of it, capped by --days), writing one
`StockFeature` row per trading day. Point-in-time correctness (no
look-ahead) is handled inside `recalculate_recent_features` itself -- see
its docstring.

Usage:
    python scripts/backfill_historical_features.py                  # all active tickers, full history
    python scripts/backfill_historical_features.py AAPL MSFT         # specific tickers
    python scripts/backfill_historical_features.py --days 260 AAPL   # cap history window
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swing_trader.db.base import session_scope  # noqa: E402
from swing_trader.db.models import TickerStatus, TickerUniverse  # noqa: E402
from swing_trader.features.engineering import recalculate_recent_features  # noqa: E402
from swing_trader.logging_setup import get_logger, setup_logging  # noqa: E402

logger = get_logger("scripts.backfill_historical_features")

# Comfortably more than a year of trading days -- effectively "all available
# history" for the default 1y yfinance backfill window (ticker_universe.backfill_years).
_DEFAULT_DAYS = 400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", help="Tickers to backfill (default: all active tickers).")
    parser.add_argument(
        "--days", type=int, default=_DEFAULT_DAYS,
        help=f"Max trading days of history to recompute per ticker (default: {_DEFAULT_DAYS}).",
    )
    args = parser.parse_args()

    setup_logging()

    with session_scope() as db:
        if args.tickers:
            tickers = [t.upper() for t in args.tickers]
        else:
            tickers = [
                t.ticker
                for t in db.query(TickerUniverse).filter(TickerUniverse.status == TickerStatus.ACTIVE).all()
            ]

        if not tickers:
            print("No tickers to backfill. Pass tickers explicitly or `make add-ticker TICKER=...` first.")
            return

        print(f"Backfilling historical features for {len(tickers)} ticker(s), up to {args.days} days each...")
        total_written = 0
        for i, ticker in enumerate(tickers, start=1):
            print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
            start = time.monotonic()
            try:
                written = recalculate_recent_features(db, ticker, days=args.days)
                db.commit()
                total_written += written
                print(f"{written} rows written ({time.monotonic() - start:.1f}s)")
            except Exception as e:
                db.rollback()
                logger.exception("Historical feature backfill failed for %s", ticker)
                print(f"FAILED: {e}")

        print(f"Done. {total_written} total StockFeature rows written across {len(tickers)} ticker(s).")
        print("Run `make predict` (or `python -m swing_trader.cli predict`) now that history exists.")


if __name__ == "__main__":
    main()

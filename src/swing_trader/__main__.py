"""Entrypoint for `python -m swing_trader` (equivalent to `python -m swing_trader.cli`)."""
from __future__ import annotations

from swing_trader.cli import cli

if __name__ == "__main__":
    cli()

"""PF-007 watchlist re-export.

SRS Section 6.2 lists both `portfolio/manager.py` and a `watchlist.py` file;
the actual implementation lives in `manager.py` (it needs no state/imports
that manager.py doesn't already have), so this module simply re-exports the
watchlist functions for anyone importing `swing_trader.portfolio.watchlist`
directly.
"""
from __future__ import annotations

from swing_trader.portfolio.manager import add_to_watchlist, evaluate_watchlist

__all__ = ["add_to_watchlist", "evaluate_watchlist"]

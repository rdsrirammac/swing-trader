"""Paper trading simulator (SRS 3.9, TE-003).

Wraps `execution.journal` but only ever operates on `Portfolio` rows
flagged `is_paper=True`, and provides fill simulation + A/B comparison
utilities for comparing strategies / model versions without risking real
capital.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping

from sqlalchemy.orm import Session

from swing_trader.db.models import Portfolio, Trade
from swing_trader.execution import journal
from swing_trader.logging_setup import get_logger

logger = get_logger("execution.paper_trading")


class PaperTradingSimulator:
    """TE-003: simulate trade execution against paper (`is_paper=True`) portfolios.

    This class never touches a live (`is_paper=False`) portfolio — both
    `open_trade` and `close_trade` raise `ValueError` if pointed at one.
    """

    def _require_paper_portfolio(self, session: Session, portfolio_id: int) -> Portfolio:
        portfolio = session.get(Portfolio, portfolio_id)
        if portfolio is None:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        if not portfolio.is_paper:
            raise ValueError(
                f"Portfolio {portfolio_id} ({portfolio.name}) is not a paper portfolio; "
                "PaperTradingSimulator refuses to operate on live portfolios."
            )
        return portfolio

    def open_trade(self, session: Session, portfolio_id: int, **kwargs: Any) -> Trade:
        """Open a paper trade. Delegates to `execution.journal.open_trade`
        after verifying `portfolio_id` refers to a paper portfolio.
        """
        self._require_paper_portfolio(session, portfolio_id)
        kwargs.setdefault("is_paper", True)
        return journal.open_trade(session, portfolio_id=portfolio_id, **kwargs)

    def close_trade(self, session: Session, trade_id: int, **kwargs: Any) -> Trade:
        """Close a paper trade. Delegates to `execution.journal.close_trade`
        after verifying the trade's portfolio is a paper portfolio.
        """
        trade = session.get(Trade, trade_id)
        if trade is None:
            raise ValueError(f"Trade {trade_id} not found")
        self._require_paper_portfolio(session, trade.portfolio_id)
        return journal.close_trade(session, trade_id=trade_id, **kwargs)

    @staticmethod
    def simulate_fill(
        price_series_today: Mapping[str, float],
        side: Literal["open", "close"],
        timing: Literal["open", "close"] = "close",
    ) -> float:
        """Return a simulated fill price from today's OHLCV row.

        `price_series_today` is any dict-like/`pd.Series` exposing
        'Open'/'Close' keys (e.g. a row from a yfinance history
        DataFrame). `side` (open vs close of a *position*) is informational
        only; `timing` selects which of today's prices ('Open' or 'Close')
        to use as the simulated fill.
        """
        key = "Open" if timing == "open" else "Close"
        if key not in price_series_today:
            raise KeyError(f"price_series_today missing '{key}' key (side={side!r})")
        return float(price_series_today[key])

    @staticmethod
    def run_ab_comparison(session: Session, portfolio_id_a: int, portfolio_id_b: int) -> dict:
        """Compare closed-trade performance between two paper portfolios.

        Useful for A/B-testing two model versions or strategy variants,
        each run through its own paper portfolio. Returns:

            {
              "portfolio_a": {"portfolio_id": ..., "trade_count": ..., "win_rate": ..., "avg_r": ..., "total_pnl": ...},
              "portfolio_b": {...},
            }
        """

        def _stats(trades: list[Trade]) -> dict:
            n = len(trades)
            if n == 0:
                return {"trade_count": 0, "win_rate": None, "avg_r": None, "total_pnl": 0.0}
            wins = sum(1 for t in trades if (t.realized_pnl or 0.0) > 0)
            r_values = [t.realized_r_multiple for t in trades if t.realized_r_multiple is not None]
            total_pnl = sum(t.realized_pnl or 0.0 for t in trades)
            return {
                "trade_count": n,
                "win_rate": wins / n,
                "avg_r": (sum(r_values) / len(r_values)) if r_values else None,
                "total_pnl": total_pnl,
            }

        trades_a = journal.get_closed_trades(session, portfolio_id_a)
        trades_b = journal.get_closed_trades(session, portfolio_id_b)
        return {
            "portfolio_a": {"portfolio_id": portfolio_id_a, **_stats(trades_a)},
            "portfolio_b": {"portfolio_id": portfolio_id_b, **_stats(trades_b)},
        }

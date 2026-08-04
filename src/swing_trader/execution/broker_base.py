"""Abstract broker interface (SRS 3.9, TE-004) — FUTURE integration only.

This module defines the *shape* of a broker integration for a future
phase. Live/paper broker order routing is explicitly deferred per SRS
Section 9. No concrete class defined anywhere in this codebase is wired
into any scheduled job, CLI command, dashboard action, or automated
pipeline — implementations (e.g. `AlpacaBroker`) exist only as
unconnected skeletons pending an explicit, human-reviewed integration
decision.

Financial safety: no code path in this package may place a real order
without an explicit human-in-the-loop confirmation step upstream of the
call. Treat every concrete `BrokerInterface` implementation as inert
until that review has happened.
"""
from __future__ import annotations

import abc
from typing import Literal


class BrokerInterface(abc.ABC):
    """Abstract broker / order-routing interface. Deferred per SRS Section 9.

    Concrete implementations are expected to wrap a specific broker's API
    (e.g. Alpaca — see `alpaca_broker.py`). None of the methods here are
    called by any scheduler, CLI command, or dashboard action in this
    codebase today; this class exists purely to document the intended
    integration surface for a future phase.
    """

    @abc.abstractmethod
    def place_bracket_order(
        self,
        ticker: str,
        shares: float,
        entry_type: Literal["market", "limit"],
        stop_loss: float,
        take_profit: float,
    ) -> str:
        """Place a bracket order (entry + stop + target).

        Returns the broker's order_id. MUST NOT be invoked automatically
        by any pipeline — callers are responsible for obtaining explicit
        human confirmation immediately before calling this, every time.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_positions(self) -> list[dict]:
        """Return current broker-side open positions."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_buying_power(self) -> float:
        """Return available buying power at the broker."""
        raise NotImplementedError

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order by broker order_id. Returns True on success."""
        raise NotImplementedError

    def handle_order_update(self, payload: dict) -> None:
        """Webhook handler stub for broker order-status callbacks.

        Default implementation is a no-op; concrete brokers may override
        to parse fill/cancel/reject events. Not wired to any HTTP route in
        this codebase — a future web handler would need to call this
        explicitly, after verifying webhook signatures, before any such
        wiring should be considered safe.
        """
        return None

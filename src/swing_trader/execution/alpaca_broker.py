"""Alpaca broker skeleton (SRS 3.9, TE-004) — UNCONNECTED, Phase-2 backlog item.

Financial safety: this module must never be wired into an automated
pipeline without an explicit human-in-the-loop confirmation step, per the
assistant's own trading-safety policy. Every public method below raises
`NotImplementedError` unconditionally once past the credential check —
this is a documentation skeleton, not a working integration. No order has
ever been placed, tested, or exercised against a live or paper Alpaca
account through this code.

`alpaca-py` is intentionally NOT added to requirements. The commented-out
bodies below sketch the intended plain-`requests` REST call shape against
Alpaca's *paper* trading API (`https://paper-api.alpaca.markets`) only,
for a future engineer to pick up deliberately — they are not executed.
"""
from __future__ import annotations

from typing import Literal

from swing_trader.config import get_settings
from swing_trader.execution.broker_base import BrokerInterface
from swing_trader.logging_setup import get_logger

logger = get_logger("execution.alpaca_broker")

_ROADMAP_MSG = "Alpaca integration is a Phase-2 backlog item — see ROADMAP.md"


class AlpacaBroker(BrokerInterface):
    """Unconnected Alpaca skeleton. DO NOT use for live trading.

    Reads `broker.alpaca_api_key` / `broker.alpaca_api_secret` /
    `broker.alpaca_base_url` from `get_settings().secret(...)`. Regardless
    of whether credentials are configured, every method raises
    `NotImplementedError` — configuring credentials only changes whether
    that error fires before or after a (never-reached) credential check,
    so this class can never accidentally place a trade.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.secret("broker.alpaca_api_key")
        self._api_secret = settings.secret("broker.alpaca_api_secret")
        self._base_url = (
            settings.secret("broker.alpaca_base_url") or "https://paper-api.alpaca.markets"
        )

    def _require_credentials(self) -> None:
        if not self._api_key or not self._api_secret:
            raise NotImplementedError(_ROADMAP_MSG)

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self._api_key or "",
            "APCA-API-SECRET-KEY": self._api_secret or "",
        }

    def place_bracket_order(
        self,
        ticker: str,
        shares: float,
        entry_type: Literal["market", "limit"],
        stop_loss: float,
        take_profit: float,
    ) -> str:
        """Intended shape of a bracket-order POST to Alpaca. NOT exercised.

        Never call this without an explicit human-in-the-loop confirmation
        step upstream of this method — none exists in this codebase today,
        and this method always raises regardless.
        """
        self._require_credentials()
        raise NotImplementedError(_ROADMAP_MSG)
        # Intended (never-executed) implementation sketch, left inert:
        #
        # import requests
        # payload = {
        #     "symbol": ticker,
        #     "qty": str(shares),
        #     "side": "buy",
        #     "type": entry_type,
        #     "time_in_force": "day",
        #     "order_class": "bracket",
        #     "stop_loss": {"stop_price": str(stop_loss)},
        #     "take_profit": {"limit_price": str(take_profit)},
        # }
        # resp = requests.post(
        #     f"{self._base_url}/v2/orders", json=payload, headers=self._headers(), timeout=10
        # )
        # resp.raise_for_status()
        # return resp.json()["id"]

    def get_positions(self) -> list[dict]:
        self._require_credentials()
        raise NotImplementedError(_ROADMAP_MSG)
        # import requests
        # resp = requests.get(f"{self._base_url}/v2/positions", headers=self._headers(), timeout=10)
        # resp.raise_for_status()
        # return resp.json()

    def get_buying_power(self) -> float:
        self._require_credentials()
        raise NotImplementedError(_ROADMAP_MSG)
        # import requests
        # resp = requests.get(f"{self._base_url}/v2/account", headers=self._headers(), timeout=10)
        # resp.raise_for_status()
        # return float(resp.json()["buying_power"])

    def cancel_order(self, order_id: str) -> bool:
        self._require_credentials()
        raise NotImplementedError(_ROADMAP_MSG)
        # import requests
        # resp = requests.delete(
        #     f"{self._base_url}/v2/orders/{order_id}", headers=self._headers(), timeout=10
        # )
        # return resp.status_code == 204

    def handle_order_update(self, payload: dict) -> None:
        """Webhook stub — Alpaca does not push webhooks natively; a real
        integration would back a future polling or SSE-based order-update
        loop. No-op today beyond logging.
        """
        logger.warning(
            "AlpacaBroker.handle_order_update called but integration is not implemented: %s",
            _ROADMAP_MSG,
        )
        return None

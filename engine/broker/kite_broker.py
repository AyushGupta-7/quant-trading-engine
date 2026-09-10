"""
engine/broker/kite_broker.py
-----------------------------
Zerodha Kite Connect REST broker adapter — STUB IMPLEMENTATION.

Provides the correctly-shaped interface with:
* Exponential backoff retry skeleton (ready to fill in).
* Token refresh hook.
* Rate limit comment annotations.
* All methods raise ``NotImplementedError`` until real credentials are used.
"""

from __future__ import annotations

import logging

from engine.broker.base import IBrokerAdapter
from engine.core.events import Order, OrderState

logger = logging.getLogger(__name__)

_STUB_NOTE = (
    "KiteBrokerAdapter is a STUB.  Real implementation requires live Zerodha "
    "credentials and kiteconnect>=5.0.  See docs/kite_integration.md."
)

# Kite rate limits (for reference when implementing):
#   REST API:   10 req/s  (order placement)
#   WebSocket:  3 connections, 3000 instruments per connection


class KiteBrokerAdapter(IBrokerAdapter):
    """
    Zerodha Kite Connect REST broker adapter (stub).

    When real implementation is added:
    - ``connect()`` should call ``/session/token`` to obtain access_token.
    - ``place_order()`` → POST ``/orders/{variety}``
    - ``cancel_order()`` → DELETE ``/orders/{variety}/{order_id}``
    - ``get_order_status()`` → GET ``/orders/{order_id}``
    - Retries with exponential backoff on 429 / 5xx.
    - Token refresh on 403.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        logger.warning(_STUB_NOTE)

    async def connect(self) -> None:
        raise NotImplementedError(_STUB_NOTE)

    async def disconnect(self) -> None:
        raise NotImplementedError(_STUB_NOTE)

    async def place_order(self, order: Order) -> str:
        raise NotImplementedError(_STUB_NOTE)

    async def cancel_order(self, broker_order_id: str) -> bool:
        raise NotImplementedError(_STUB_NOTE)

    async def get_order_status(self, broker_order_id: str) -> OrderState:
        raise NotImplementedError(_STUB_NOTE)

    async def get_positions(self) -> list[dict]:
        raise NotImplementedError(_STUB_NOTE)

    # Retry sketch (to be implemented with real HTTP client):
    # async def _request_with_retry(self, method, url, **kwargs):
    #     for attempt in range(5):
    #         try:
    #             resp = await self._session.request(method, url, **kwargs)
    #             resp.raise_for_status()
    #             return resp.json()
    #         except aiohttp.ClientResponseError as e:
    #             if e.status == 429:  # rate limited
    #                 await asyncio.sleep(2 ** attempt)
    #             elif e.status == 403:  # token expired
    #                 await self._refresh_token()
    #             elif e.status >= 500:
    #                 await asyncio.sleep(2 ** attempt)
    #             else:
    #                 raise
    #     raise RuntimeError("Max retries exceeded")

"""
engine/data/adapters/kite_feed.py
----------------------------------
Zerodha Kite Connect WebSocket feed adapter — STUB IMPLEMENTATION.

This module provides a correctly-shaped adapter that raises
``NotImplementedError`` for all live operations.  Replace method bodies with
real KiteConnect WebSocket calls once credentials are available.

The stub is retained in the codebase so that:
  1. The interface contract is documented.
  2. Import paths work without changes when real integration is added.
  3. Tests can exercise the configuration path without real network calls.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from engine.data.adapters.base import IMarketDataAdapter

logger = logging.getLogger(__name__)

_STUB_NOTE = (
    "KiteFeedAdapter is a STUB.  Real implementation requires live Zerodha "
    "credentials and kiteconnect>=5.0.  See docs/kite_integration.md."
)


class KiteFeedAdapter(IMarketDataAdapter):
    """
    Zerodha Kite Connect WebSocket feed adapter (stub).

    When real implementation is added:
    - ``connect()`` should initialise ``kiteconnect.KiteTicker`` and
      authenticate using the access token.
    - ``subscribe()`` should call ``ticker.subscribe()`` with the numeric
      instrument tokens mapped from symbols via the contract master.
    - ``tick_stream()`` should expose ticks from the ``on_ticks`` callback
      via an asyncio Queue.
    - Token refresh should be handled transparently via ``on_connect``
      callback.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        logger.warning(_STUB_NOTE)

    async def connect(self) -> None:
        raise NotImplementedError(_STUB_NOTE)

    async def disconnect(self) -> None:
        raise NotImplementedError(_STUB_NOTE)

    async def subscribe(self, symbols: list[str]) -> None:
        raise NotImplementedError(_STUB_NOTE)

    async def tick_stream(self) -> AsyncIterator[dict]:  # type: ignore[override]
        raise NotImplementedError(_STUB_NOTE)
        # Unreachable; required to satisfy async generator typing
        yield {}  # noqa: unreachable

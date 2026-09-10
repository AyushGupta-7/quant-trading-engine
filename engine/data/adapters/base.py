"""
engine/data/adapters/base.py
----------------------------
Abstract base class for all market-data adapters.

Any concrete adapter (MockFeedAdapter, KiteFeedAdapter, …) implements this
interface.  The rest of the engine only ever talks to ``IMarketDataAdapter``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class IMarketDataAdapter(ABC):
    """
    Contract for market-data adapters.

    Lifecycle::

        adapter = ConcreteAdapter(config)
        await adapter.connect()
        await adapter.subscribe(["CRUDEOIL", "GOLD"])

        async for raw_tick in adapter.tick_stream():
            ...   # process tick

        await adapter.disconnect()
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Cleanly disconnect and release resources."""

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to tick updates for *symbols*."""

    @abstractmethod
    def tick_stream(self) -> AsyncIterator[dict]:
        """
        Async generator that yields raw tick dicts indefinitely.

        The caller is responsible for passing each dict through
        ``TickNormaliser`` before use.
        """

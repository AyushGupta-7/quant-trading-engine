"""
engine/broker/base.py
----------------------
Abstract broker adapter interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.core.events import Fill, Order, OrderState


class IBrokerAdapter(ABC):
    """
    Contract for broker adapters.

    All methods are async to support REST/WebSocket implementations.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Authenticate and connect to the broker."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close connections."""

    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """
        Submit *order* to the broker.

        Returns
        -------
        str
            The broker-assigned order ID.

        Raises
        ------
        Exception
            On rejection or connectivity failure.
        """

    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order.  Returns True on success."""

    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> OrderState:
        """Return the current state of an order."""

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Return current open positions as a list of raw dicts."""

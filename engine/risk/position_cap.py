"""
engine/risk/position_cap.py
----------------------------
Enforces maximum net position per symbol and total portfolio position.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from engine.core.events import OrderIntent, Side

logger = logging.getLogger(__name__)


class PositionCapGuard:
    """
    Rejects order intents that would breach per-symbol or portfolio position caps.

    Parameters
    ----------
    max_position_lots:
        Maximum net lots per symbol (absolute value).
    max_open_orders:
        Maximum number of simultaneous open orders across all symbols.
    """

    def __init__(
        self,
        max_position_lots: int = 10,
        max_open_orders: int = 20,
    ) -> None:
        self._max_pos   = max_position_lots
        self._max_orders = max_open_orders
        self._positions: dict[str, int] = {}   # symbol → net lots
        self._open_order_count: int = 0

    def update_position(self, symbol: str, delta_lots: int) -> None:
        """Called after a fill; updates net position tracker."""
        self._positions[symbol] = self._positions.get(symbol, 0) + delta_lots

    def increment_open_orders(self, n: int = 1) -> None:
        self._open_order_count += n

    def decrement_open_orders(self, n: int = 1) -> None:
        self._open_order_count = max(0, self._open_order_count - n)

    def approve(self, intent: OrderIntent) -> tuple[bool, str]:
        """
        Returns (True, "") if approved; (False, reason) if rejected.
        """
        sym = intent.symbol
        current = self._positions.get(sym, 0)
        new_pos = current + (intent.qty if intent.side == Side.BUY else -intent.qty)

        if abs(new_pos) > self._max_pos:
            reason = (
                f"PositionCapGuard: {sym} position {new_pos} would exceed "
                f"cap {self._max_pos}"
            )
            logger.warning(reason)
            return False, reason

        if self._open_order_count >= self._max_orders:
            reason = (
                f"PositionCapGuard: open order count {self._open_order_count} "
                f">= max {self._max_orders}"
            )
            logger.warning(reason)
            return False, reason

        return True, ""

    def net_position(self, symbol: str) -> int:
        return self._positions.get(symbol, 0)

    def all_positions(self) -> dict[str, int]:
        return dict(self._positions)

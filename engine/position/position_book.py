"""
engine/position/position_book.py
---------------------------------
Maintains net position per instrument.  Updated on every ``Fill`` event.

Position is tracked in lots (not contracts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from engine.core.events import Fill, Side

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    net_lots: int = 0                   # positive = long, negative = short
    avg_entry_price: Decimal = Decimal(0)
    total_cost: Decimal = Decimal(0)    # absolute cost basis
    last_updated: datetime | None = None

    def is_flat(self) -> bool:
        return self.net_lots == 0

    def market_value(self, ltp: Decimal, lot_size: int) -> Decimal:
        return ltp * self.net_lots * lot_size


class PositionBook:
    """
    Tracks net positions across all instruments.

    Parameters
    ----------
    lot_size_fn:
        Callable(symbol) → int returning the instrument's lot size.
        If None, lot_size defaults to 1.
    """

    def __init__(self, lot_size_fn=None) -> None:
        self._positions: dict[str, Position] = {}
        self._lot_size_fn = lot_size_fn or (lambda s: 1)

    def on_fill(self, fill: Fill) -> Position:
        """Apply a fill and return the updated position."""
        sym = fill.symbol
        if sym not in self._positions:
            self._positions[sym] = Position(symbol=sym)

        pos = self._positions[sym]
        prev_lots = pos.net_lots

        if fill.side == Side.BUY:
            new_lots = prev_lots + fill.fill_qty
            # Weighted average entry on adds
            if new_lots != 0 and prev_lots >= 0:
                pos.avg_entry_price = (
                    pos.avg_entry_price * prev_lots + fill.fill_price * fill.fill_qty
                ) / new_lots
            pos.net_lots = new_lots
        else:
            new_lots = prev_lots - fill.fill_qty
            # On partial close of long, avg entry stays; on reversal, reset
            if new_lots < 0 and prev_lots >= 0:
                # Full close + reversal into short
                pos.avg_entry_price = fill.fill_price
            pos.net_lots = new_lots

        pos.total_cost += fill.fees
        pos.last_updated = fill.ts

        logger.debug(
            "PositionBook: %s  net_lots=%d  avg_entry=%.2f",
            sym, pos.net_lots, pos.avg_entry_price,
        )
        return pos

    def get(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol=symbol))

    def all(self) -> dict[str, Position]:
        return dict(self._positions)

    def net_exposure(self) -> dict[str, int]:
        return {sym: pos.net_lots for sym, pos in self._positions.items() if not pos.is_flat()}

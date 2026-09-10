"""
engine/position/pnl_engine.py
------------------------------
Computes realised and unrealised P&L.

Realised P&L: computed on close/reduction of a position (FIFO matching).
Unrealised P&L: computed using current LTP and average entry price.

All arithmetic uses ``Decimal`` for paisa-level accuracy.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from engine.core.events import Fill, Side
from engine.position.position_book import Position

logger = logging.getLogger(__name__)


class PnLEngine:
    """
    Tracks realised and unrealised P&L per symbol and total portfolio.

    Parameters
    ----------
    lot_size_fn:
        Callable(symbol) → int.  Defaults to always returning 1.
    """

    def __init__(self, lot_size_fn=None) -> None:
        self._lot_size_fn = lot_size_fn or (lambda s: 1)
        self._realised:   dict[str, Decimal] = {}
        self._unrealised: dict[str, Decimal] = {}
        self._total_fees: dict[str, Decimal] = {}

    def on_fill(self, fill: Fill, position_before: Position) -> Decimal:
        """
        Compute realised P&L from a closing/reducing fill.

        Returns the realised P&L delta for this fill (zero on pure entry).
        """
        sym = fill.symbol
        lot_size = self._lot_size_fn(sym)
        prev_lots = position_before.net_lots
        prev_entry = position_before.avg_entry_price

        realised = Decimal(0)

        if fill.side == Side.SELL and prev_lots > 0:
            # Closing long position (partial or full)
            closing_qty = min(fill.fill_qty, prev_lots)
            realised = (fill.fill_price - prev_entry) * closing_qty * lot_size

        elif fill.side == Side.BUY and prev_lots < 0:
            # Closing short position
            closing_qty = min(fill.fill_qty, abs(prev_lots))
            realised = (prev_entry - fill.fill_price) * closing_qty * lot_size

        self._realised[sym] = self._realised.get(sym, Decimal(0)) + realised - fill.fees
        self._total_fees[sym] = self._total_fees.get(sym, Decimal(0)) + fill.fees

        if realised != 0:
            logger.debug(
                "PnLEngine: realised=%.2f sym=%s side=%s @%.2f",
                realised, sym, fill.side.value, fill.fill_price,
            )

        return realised

    def update_unrealised(self, symbol: str, position: Position, ltp: Decimal) -> Decimal:
        """Compute unrealised P&L from current LTP."""
        lot_size = self._lot_size_fn(symbol)
        if position.net_lots == 0 or position.avg_entry_price == 0:
            self._unrealised[symbol] = Decimal(0)
            return Decimal(0)

        pnl = (ltp - position.avg_entry_price) * position.net_lots * lot_size
        self._unrealised[symbol] = pnl
        return pnl

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def total_realised(self) -> Decimal:
        return sum(self._realised.values(), Decimal(0))

    def total_unrealised(self) -> Decimal:
        return sum(self._unrealised.values(), Decimal(0))

    def total_equity(self) -> Decimal:
        return self.total_realised() + self.total_unrealised()

    def total_fees(self) -> Decimal:
        return sum(self._total_fees.values(), Decimal(0))

    def realised_by_symbol(self) -> dict[str, Decimal]:
        return dict(self._realised)

    def unrealised_by_symbol(self) -> dict[str, Decimal]:
        return dict(self._unrealised)

    def summary(self) -> dict:
        return {
            "total_realised":   float(self.total_realised()),
            "total_unrealised": float(self.total_unrealised()),
            "total_equity":     float(self.total_equity()),
            "total_fees":       float(self.total_fees()),
        }

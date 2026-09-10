"""
engine/backtest/fill_model.py
------------------------------
Bar-accurate fill model for backtesting.

Rules (no lookahead)
~~~~~~~~~~~~~~~~~~~~
* **MARKET orders**: fill at the OPEN of the *next* bar.
* **LIMIT orders**: fill if the bar's range crosses the limit price.
  Fill price = limit price (no additional slippage on limit orders in sim).
* **SL orders**: fill if bar's low ≤ stop (for long SL) or high ≥ stop (for short SL).
* Slippage is applied as a fixed number of ticks on MARKET fills.
* Commission is computed via the CostModel.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Callable

from engine.core.events import Bar, Fill, Order, OrderState, OrderType, Side
from engine.core.instrument import Instrument
from engine.position.cost_model import CostModel


class BarFillModel:
    """
    Simulates fills from OHLCV bars.

    Parameters
    ----------
    cost_model:
        Used to compute fees on each fill.
    slippage_ticks:
        Ticks of slippage applied to MARKET orders.
    on_fill:
        Callback invoked with each simulated ``Fill``.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        slippage_ticks: int = 1,
        on_fill: Callable[[Fill], None] | None = None,
    ) -> None:
        self._cost_model     = cost_model
        self._slippage_ticks = slippage_ticks
        self._on_fill        = on_fill
        self._pending: list[tuple[Order, Instrument]] = []

    def submit(self, order: Order, instrument: Instrument) -> None:
        """Register an order for fill evaluation on the next bar."""
        self._pending.append((order, instrument))

    def process_bar(self, bar: Bar) -> list[Fill]:
        """
        Evaluate all pending orders against *bar*.

        Returns the list of fills produced.
        """
        fills: list[Fill] = []
        remaining: list[tuple[Order, Instrument]] = []

        for order, instr in self._pending:
            if order.symbol != bar.symbol:
                remaining.append((order, instr))
                continue

            fill = self._try_fill(order, instr, bar)
            if fill:
                fills.append(fill)
                if self._on_fill:
                    self._on_fill(fill)
            else:
                remaining.append((order, instr))

        self._pending = remaining
        return fills

    # ------------------------------------------------------------------

    def _try_fill(self, order: Order, instr: Instrument, bar: Bar) -> Fill | None:
        fill_price: Decimal | None = None

        if order.order_type == OrderType.MARKET:
            # Fill at bar open + slippage (bar open IS next bar's open)
            base = bar.open
            slip = instr.tick_size * self._slippage_ticks
            fill_price = base + slip if order.side == Side.BUY else base - slip

        elif order.order_type == OrderType.LIMIT:
            if order.price is None:
                return None
            if order.side == Side.BUY and bar.low <= order.price:
                fill_price = order.price
            elif order.side == Side.SELL and bar.high >= order.price:
                fill_price = order.price

        elif order.order_type in (OrderType.SL, OrderType.SL_M):
            if order.price is None:
                return None
            if order.side == Side.SELL and bar.low <= order.price:
                fill_price = order.price
            elif order.side == Side.BUY and bar.high >= order.price:
                fill_price = order.price

        if fill_price is None:
            return None

        fill_price = max(fill_price, instr.tick_size)  # floor at one tick

        fees = Decimal(0)
        if self._cost_model:
            # Build a temporary fill without fees to compute cost
            temp_fill = Fill(
                order_id=order.client_order_id,
                broker_order_id="",
                symbol=order.symbol,
                side=order.side,
                fill_price=fill_price,
                fill_qty=order.qty,
                fees=Decimal(0),
                ts=bar.ts,
            )
            fees = self._cost_model.compute(temp_fill, instr)

        return Fill(
            order_id=order.client_order_id,
            broker_order_id=f"BT-{uuid.uuid4().hex[:8]}",
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_qty=order.qty,
            fees=fees,
            ts=bar.ts,
        )

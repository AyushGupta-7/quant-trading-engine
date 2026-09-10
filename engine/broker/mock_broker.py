"""
engine/broker/mock_broker.py
-----------------------------
Realistic mock broker adapter for simulation and backtesting.

Features
~~~~~~~~
* Fills at ``ltp ± gaussian_slippage`` (configurable ticks).
* Configurable fill probability (simulates partial/rejected fills).
* Configurable fill latency (asyncio sleep).
* Configurable rejection rate.
* Maintains an in-memory order book and position registry.
* Emits ``FillEvent`` via callback on fill.

This adapter is the default for all backtest and simulation runs.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Optional

from engine.broker.base import IBrokerAdapter
from engine.core.events import Fill, Order, OrderState, OrderType, Side
from engine.core.instrument import InstrumentRegistry

logger = logging.getLogger(__name__)


class MockBrokerAdapter(IBrokerAdapter):
    """
    Simulated broker.

    Parameters
    ----------
    slippage_ticks:
        Number of ticks of slippage applied symmetrically.
    fill_probability:
        Probability [0,1] that an order gets filled (rest are rejected).
    fill_latency_ms:
        Simulated fill latency in milliseconds.
    rejection_rate:
        Rate at which orders are randomly rejected (separate from fill_prob).
    on_fill:
        Callback invoked with each ``Fill`` object.
    instrument_registry:
        Used to look up tick size for slippage calculation.
    seed:
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        slippage_ticks: int = 1,
        fill_probability: float = 0.98,
        fill_latency_ms: int = 50,
        rejection_rate: float = 0.01,
        on_fill: Callable[[Fill], None] | None = None,
        instrument_registry: InstrumentRegistry | None = None,
        seed: int | None = None,
    ) -> None:
        self._slippage_ticks = slippage_ticks
        self._fill_prob      = fill_probability
        self._latency_ms     = fill_latency_ms
        self._rejection_rate = rejection_rate
        self._on_fill        = on_fill
        self._registry       = instrument_registry
        self._rng = random.Random(seed)

        # State
        self._orders:    dict[str, Order] = {}     # broker_id → Order
        self._ltp:       dict[str, Decimal] = {}   # symbol → last price
        self._connected: bool = False

    # ------------------------------------------------------------------
    # IBrokerAdapter
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True
        logger.info("MockBrokerAdapter: connected (simulation mode)")

    async def disconnect(self) -> None:
        self._connected = False
        logger.info("MockBrokerAdapter: disconnected")

    async def place_order(self, order: Order) -> str:
        """Simulate order placement and fill."""
        # Random rejection
        if self._rng.random() < self._rejection_rate:
            raise RuntimeError(f"MockBroker: random rejection for {order.client_order_id}")

        # Assign broker ID
        broker_id = f"MOCK-{uuid.uuid4().hex[:10].upper()}"
        self._orders[broker_id] = order
        logger.debug(
            "MockBroker: received order coid=%s broker_id=%s sym=%s side=%s qty=%d",
            order.client_order_id, broker_id, order.symbol, order.side.value, order.qty,
        )

        # Schedule fill in background
        asyncio.create_task(
            self._fill_order(broker_id, order), name=f"mock-fill-{broker_id}"
        )
        return broker_id

    async def cancel_order(self, broker_order_id: str) -> bool:
        order = self._orders.get(broker_order_id)
        if order and order.state == OrderState.OPEN:
            order.state = OrderState.CANCELLED
            logger.debug("MockBroker: cancelled broker_id=%s", broker_order_id)
            return True
        return False

    async def get_order_status(self, broker_order_id: str) -> OrderState:
        order = self._orders.get(broker_order_id)
        return order.state if order else OrderState.REJECTED

    async def get_positions(self) -> list[dict]:
        """Return a list of non-zero positions inferred from filled orders."""
        positions: dict[str, int] = {}
        for order in self._orders.values():
            if order.state == OrderState.COMPLETE:
                delta = order.filled_qty if order.side == Side.BUY else -order.filled_qty
                positions[order.symbol] = positions.get(order.symbol, 0) + delta
        return [
            {"symbol": sym, "net_lots": lots}
            for sym, lots in positions.items()
            if lots != 0
        ]

    # ------------------------------------------------------------------
    # Market data integration
    # ------------------------------------------------------------------

    def update_ltp(self, symbol: str, price: Decimal) -> None:
        """Feed the latest price so MARKET orders fill at correct price."""
        self._ltp[symbol] = price

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fill_order(self, broker_id: str, order: Order) -> None:
        """Simulate latency then attempt fill."""
        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000.0)

        order = self._orders[broker_id]
        if order.state not in (OrderState.OPEN, OrderState.PENDING):
            return

        # Fill probability check
        if self._rng.random() > self._fill_prob:
            order.state = OrderState.REJECTED
            logger.debug("MockBroker: fill rejected (prob) broker_id=%s", broker_id)
            return

        fill_price = self._compute_fill_price(order)
        fill = Fill(
            order_id=order.client_order_id,
            broker_order_id=broker_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            fill_qty=order.qty,
            fees=Decimal(0),   # fees computed in CostModel
            ts=datetime.now(tz=timezone.utc),
        )

        order.state          = OrderState.COMPLETE
        order.filled_qty     = order.qty
        order.avg_fill_price = fill_price

        if self._on_fill:
            self._on_fill(fill)

        logger.debug(
            "MockBroker: FILL broker_id=%s sym=%s side=%s @%.2f",
            broker_id, order.symbol, order.side.value, fill_price,
        )

    def _compute_fill_price(self, order: Order) -> Decimal:
        """Determine fill price with slippage."""
        if order.order_type == OrderType.LIMIT and order.price is not None:
            base = order.price
        elif order.symbol in self._ltp:
            base = self._ltp[order.symbol]
        else:
            raise ValueError(f"No LTP available for {order.symbol}")

        # Tick size for slippage
        tick = Decimal("1.0")
        if self._registry:
            try:
                tick = self._registry.get(order.symbol).tick_size
            except KeyError:
                pass

        # Gaussian slippage: mean=0, std=slippage_ticks
        std_ticks = max(0, self._slippage_ticks)
        slip_ticks = Decimal(str(round(self._rng.gauss(0, std_ticks), 1)))
        slip = slip_ticks * tick

        # Slippage always hurts the buyer (BUY fills higher, SELL fills lower)
        if order.side == Side.BUY:
            return base + abs(slip)
        else:
            return base - abs(slip)

    @classmethod
    def from_config(cls, cfg: dict, **kwargs) -> "MockBrokerAdapter":
        mock_cfg = cfg.get("mock", {})
        return cls(
            slippage_ticks=int(mock_cfg.get("slippage_ticks", 1)),
            fill_probability=float(mock_cfg.get("fill_probability", 0.98)),
            fill_latency_ms=int(mock_cfg.get("fill_latency_ms", 50)),
            rejection_rate=float(mock_cfg.get("rejection_rate", 0.01)),
            **kwargs,
        )

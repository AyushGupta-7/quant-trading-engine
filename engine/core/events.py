"""
engine/core/events.py
---------------------
Typed event dataclasses that flow through the asyncio EventBus.

All events are immutable (frozen=True) to prevent accidental mutation in
multi-consumer scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"

    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"            # stop-loss
    SL_M = "SL_M"        # stop-loss market


class OrderState(Enum):
    PENDING = auto()
    OPEN = auto()
    COMPLETE = auto()
    CANCELLED = auto()
    REJECTED = auto()


class RegimeState(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    CRISIS = "CRISIS"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Market data events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tick:
    """Normalised market tick."""
    symbol: str
    ltp: Decimal            # last traded price
    bid: Decimal
    ask: Decimal
    volume: int             # total session volume at this tick
    oi: int                 # open interest
    ts: datetime            # exchange timestamp (UTC)

    def spread(self) -> Decimal:
        return self.ask - self.bid


@dataclass(frozen=True)
class Bar:
    """OHLCV bar produced by BarBuilder."""
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    ts: datetime            # bar close timestamp (UTC)

    def typical_price(self) -> Decimal:
        return (self.high + self.low + self.close) / Decimal(3)

    def range(self) -> Decimal:
        return self.high - self.low


# ---------------------------------------------------------------------------
# Order events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderIntent:
    """
    A strategy's request to place an order.  Not yet validated or committed.
    """
    symbol: str
    side: Side
    qty: int                # lots
    order_type: OrderType
    price: Decimal | None   # None for MARKET orders
    strategy_id: str
    tag: str = ""           # free-form label for blotter


@dataclass
class Order:
    """Committed order with OMS state."""
    client_order_id: str
    symbol: str
    side: Side
    qty: int
    order_type: OrderType
    price: Decimal | None
    strategy_id: str
    tag: str
    state: OrderState = OrderState.PENDING
    broker_order_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    filled_qty: int = 0
    avg_fill_price: Decimal = Decimal(0)

    def is_terminal(self) -> bool:
        return self.state in (
            OrderState.COMPLETE, OrderState.CANCELLED, OrderState.REJECTED
        )


@dataclass(frozen=True)
class Fill:
    """A single execution report from the broker."""
    order_id: str           # client_order_id
    broker_order_id: str
    symbol: str
    side: Side
    fill_price: Decimal
    fill_qty: int           # lots
    fees: Decimal           # total transaction costs in INR
    ts: datetime


# ---------------------------------------------------------------------------
# System events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KillSwitchEvent:
    """Emitted when a kill switch is engaged (manual or automatic)."""
    reason: str
    source: str             # "operator" | "circuit_breaker" | "drawdown"
    ts: datetime


@dataclass(frozen=True)
class RegimeChangeEvent:
    """Emitted when regime scorer transitions state."""
    previous: RegimeState
    current: RegimeState
    ts: datetime
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ErrorEvent:
    """Wraps unexpected exceptions for central error handling."""
    source: str
    error: str
    ts: datetime
    details: dict[str, Any] = field(default_factory=dict)

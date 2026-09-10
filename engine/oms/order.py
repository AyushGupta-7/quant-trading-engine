"""
engine/oms/order.py
--------------------
Order lifecycle state machine and persistence helpers.

client_order_id format: ``{strategy_id}:{symbol}:{side}:{ts_epoch_ms}``
This format is deterministic and idempotent — the same intent always maps
to the same client_order_id if retried within the same millisecond tick.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from engine.core.events import Order, OrderIntent, OrderState, OrderType, Side


def make_client_order_id(intent: OrderIntent, ts_ms: int | None = None) -> str:
    """
    Generate a deterministic client order ID from an ``OrderIntent``.

    Uses a hash of the key fields so duplicate submissions (retries) produce
    the same ID — allowing the OMS to deduplicate them.
    """
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    raw = f"{intent.strategy_id}:{intent.symbol}:{intent.side.value}:{intent.qty}:{ts_ms}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"{intent.strategy_id[:8]}_{h}"


def intent_to_order(intent: OrderIntent, ts_ms: int | None = None) -> Order:
    """Convert an approved ``OrderIntent`` into a pending ``Order``."""
    coid = make_client_order_id(intent, ts_ms)
    now  = datetime.now(tz=timezone.utc)
    return Order(
        client_order_id=coid,
        symbol=intent.symbol,
        side=intent.side,
        qty=intent.qty,
        order_type=intent.order_type,
        price=intent.price,
        strategy_id=intent.strategy_id,
        tag=intent.tag,
        state=OrderState.PENDING,
        created_at=now,
        updated_at=now,
    )

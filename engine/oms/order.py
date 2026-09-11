"""
engine/oms/order.py
--------------------
Order lifecycle state machine and persistence helpers.

client_order_id derivation
~~~~~~~~~~~~~~~~~~~~~~~~~~
The ID is an SHA-1 hash of::

    strategy_id : symbol : side : qty : order_type : price : tag : bar_ts_ms

Using the bar timestamp (not wall-clock time) guarantees that two retries of
the SAME intent within the same bar produce the SAME ID, enabling true
idempotency.  Different bars always produce different timestamps → different IDs
even when all other intent fields are identical.

Callers SHOULD pass ``bar_ts_ms`` explicitly.  When omitted (e.g. interactive
or test code), we fall back to the current second (not millisecond) to avoid
accidental collision during a single bar but still allow meaningful replay.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from engine.core.events import Order, OrderIntent, OrderState, OrderType, Side


def make_client_order_id(intent: OrderIntent, bar_ts_ms: int | None = None) -> str:
    """
    Generate a deterministic client order ID from an ``OrderIntent``.

    The hash includes the bar's epoch-millisecond timestamp so that:
    * Two calls for the **same intent in the same bar** → **same ID** (idempotent).
    * Two calls for the **same intent in different bars** → different IDs.

    Parameters
    ----------
    intent:
        The order intent to identify.
    bar_ts_ms:
        Epoch milliseconds of the bar that generated this intent.  Pass
        ``int(bar.ts.timestamp() * 1000)`` from the engine.  When omitted,
        falls back to the current second (×1000) — adequate for interactive
        or test usage, but callers should always supply this explicitly.
    """
    if bar_ts_ms is None:
        # Fall back to current second (not millisecond) so tests without a
        # bar timestamp still work, while reducing accidental collisions.
        bar_ts_ms = int(time.time()) * 1000

    price_str = str(intent.price) if intent.price is not None else "MKT"
    raw = (
        f"{intent.strategy_id}:{intent.symbol}:{intent.side.value}:"
        f"{intent.qty}:{intent.order_type.value}:{price_str}:{intent.tag}:{bar_ts_ms}"
    )
    h = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"{intent.strategy_id[:8]}_{h}"


def intent_to_order(intent: OrderIntent, bar_ts_ms: int | None = None) -> Order:
    """Convert an approved ``OrderIntent`` into a pending ``Order``.

    Parameters
    ----------
    bar_ts_ms:
        Epoch milliseconds of the bar that generated *intent*.  Forwarded to
        :func:`make_client_order_id` to produce a deterministic, idempotent ID.
    """
    coid = make_client_order_id(intent, bar_ts_ms)
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

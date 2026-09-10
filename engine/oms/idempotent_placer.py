"""
engine/oms/idempotent_placer.py
--------------------------------
Wraps the broker adapter with idempotency guarantees.

On every placement:
  1. Compute ``client_order_id`` from intent.
  2. Check state store — if already COMPLETE/OPEN/PENDING, skip re-submission.
  3. Persist PENDING state.
  4. Call broker.
  5. Update state to OPEN.

On crash-restart:
  The reconciler reads PENDING/OPEN orders from the store and syncs with the
  broker to determine their true final state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from engine.core.events import Fill, Order, OrderIntent, OrderState
from engine.oms.order import intent_to_order
from engine.oms.state_store import OrderStateStore

logger = logging.getLogger(__name__)


class IdempotentPlacer:
    """
    Idempotent order placement layer.

    Parameters
    ----------
    store:
        The persistent order state store.
    broker:
        Any object implementing ``IBrokerAdapter`` (or duck-typed equivalent).
    """

    def __init__(self, store: OrderStateStore, broker) -> None:
        self._store  = store
        self._broker = broker

    async def place(self, intent: OrderIntent) -> Order | None:
        """
        Place *intent* via the broker with idempotency.

        Returns the ``Order`` object if submission succeeded, or ``None`` if
        it was a duplicate or was rejected.
        """
        order = intent_to_order(intent)
        coid  = order.client_order_id

        # --- Dedup check ---
        existing = self._store.get(coid)
        if existing is not None:
            logger.debug(
                "IdempotentPlacer: duplicate coid=%s state=%s — skipping",
                coid, existing.state.value,
            )
            return existing

        # --- Persist PENDING before touching broker ---
        self._store.upsert(order)
        logger.debug("IdempotentPlacer: persisted PENDING coid=%s", coid)

        try:
            broker_id = await self._broker.place_order(order)
        except Exception as exc:
            order.state = OrderState.REJECTED
            order.updated_at = datetime.now(tz=timezone.utc)
            self._store.upsert(order)
            logger.error("IdempotentPlacer: broker rejected coid=%s error=%s", coid, exc)
            return None

        # --- Update to OPEN ---
        order.broker_order_id = broker_id
        order.state           = OrderState.OPEN
        order.updated_at      = datetime.now(tz=timezone.utc)
        self._store.upsert(order)
        logger.info(
            "IdempotentPlacer: OPEN coid=%s broker_id=%s sym=%s side=%s qty=%d",
            coid, broker_id, order.symbol, order.side.value, order.qty,
        )
        return order

    def record_fill(self, fill: Fill) -> None:
        """Update order state to COMPLETE after receiving a fill."""
        order = self._store.get(fill.order_id)
        if order is None:
            logger.warning("IdempotentPlacer: fill for unknown coid=%s", fill.order_id)
            return
        order.filled_qty     += fill.fill_qty
        order.avg_fill_price  = fill.fill_price
        order.state           = (
            OrderState.COMPLETE if order.filled_qty >= order.qty else OrderState.OPEN
        )
        order.updated_at = fill.ts
        self._store.upsert(order)
        logger.debug(
            "IdempotentPlacer: fill recorded coid=%s filled=%d/%d",
            fill.order_id, order.filled_qty, order.qty,
        )

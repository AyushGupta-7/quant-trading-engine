"""
engine/oms/reconciler.py
-------------------------
On-restart reconciliation: resolves discrepancies between the OMS state store
and the broker's view of orders.

Recovery logic
~~~~~~~~~~~~~~
* PENDING orders in store → query broker; if not found, mark REJECTED.
* OPEN orders in store → query broker; apply final state.
* COMPLETE orders → skip (terminal state).

In the mock environment the broker confirms all persistent open orders.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from engine.core.events import OrderState
from engine.oms.state_store import OrderStateStore

logger = logging.getLogger(__name__)


class ReconciliationEngine:
    """
    Synchronises OMS state store with broker on startup.
    """

    def __init__(self, store: OrderStateStore, broker) -> None:
        self._store  = store
        self._broker = broker

    async def reconcile(self) -> dict:
        """
        Query the broker for all non-terminal orders and update the store.

        Returns a summary dict ``{resolved: int, skipped: int, errors: int}``.
        """
        pending = self._store.get_by_state(OrderState.PENDING)
        open_   = self._store.get_by_state(OrderState.OPEN)
        to_check = pending + open_

        resolved = skipped = errors = 0

        for order in to_check:
            if order.broker_order_id is None:
                # Was never sent to broker — mark rejected
                order.state      = OrderState.REJECTED
                order.updated_at = datetime.now(tz=timezone.utc)
                self._store.upsert(order)
                resolved += 1
                logger.warning(
                    "Reconciler: no broker_id for coid=%s → REJECTED",
                    order.client_order_id,
                )
                continue

            try:
                broker_state = await self._broker.get_order_status(order.broker_order_id)
                if broker_state != order.state:
                    order.state      = broker_state
                    order.updated_at = datetime.now(tz=timezone.utc)
                    self._store.upsert(order)
                    resolved += 1
                    logger.info(
                        "Reconciler: coid=%s updated to %s",
                        order.client_order_id, broker_state.value,
                    )
                else:
                    skipped += 1
            except Exception as exc:
                errors += 1
                logger.error(
                    "Reconciler: error querying broker for coid=%s: %s",
                    order.client_order_id, exc,
                )

        logger.info(
            "Reconciler: complete  resolved=%d  skipped=%d  errors=%d",
            resolved, skipped, errors,
        )
        return {"resolved": resolved, "skipped": skipped, "errors": errors}

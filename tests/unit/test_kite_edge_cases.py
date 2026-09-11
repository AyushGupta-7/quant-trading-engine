"""
tests/unit/test_kite_edge_cases.py
-----------------------------------
Tests specifically designed to verify production edge cases for the Kite integration,
such as network timeouts during order placement, reconciliation recovery, and tick
stream exception handling.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.core.events import OrderIntent, OrderState, OrderType, Side
from engine.oms.idempotent_placer import IdempotentPlacer
from engine.oms.order import intent_to_order
from engine.oms.reconciler import ReconciliationEngine
from engine.oms.state_store import OrderStateStore
from engine.data.adapters.kite_feed import KiteFeedAdapter


# ---------------------------------------------------------------------------
# Order Management Edge Cases
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    return OrderStateStore(":memory:")


def _make_intent(coid: str = "test_entry") -> OrderIntent:
    return OrderIntent(
        symbol="CRUDEOIL",
        side=Side.BUY,
        qty=1,
        order_type=OrderType.MARKET,
        price=None,
        strategy_id="s1",
        tag=coid,
    )


@pytest.mark.asyncio
async def test_placement_timeout_leaves_order_pending(store):
    """
    Edge Case 12: Network timeout/uncertain result after order placement.
    If place_order raises an exception, we must NEVER blindly mark it REJECTED,
    because it might have reached the broker. It must stay PENDING.
    """
    mock_broker = AsyncMock()
    # Simulate a network timeout (e.g. read timeout)
    mock_broker.place_order.side_effect = TimeoutError("Network timeout")

    placer = IdempotentPlacer(store=store, broker=mock_broker)
    intent = _make_intent("timeout_test")

    result = await placer.place(intent)
    assert result is None  # Placement did not return an order

    # Verify it was saved as PENDING and NOT updated to REJECTED
    saved_order = store.get(intent_to_order(intent).client_order_id)
    assert saved_order is not None
    assert saved_order.state == OrderState.PENDING


@pytest.mark.asyncio
async def test_reconciler_recovers_missing_broker_id_via_tag_match(store):
    """
    Edge Case 20: Broker truth reconciles with local state.
    If an order is PENDING with NO broker_order_id (e.g. due to previous timeout),
    the reconciler should match it against broker orders using the tag.
    """
    order = intent_to_order(_make_intent("recovery_test"))
    order.state = OrderState.PENDING
    order.broker_order_id = None
    store.upsert(order)

    mock_broker = AsyncMock()
    # Simulate broker returning the order successfully with a matching tag
    mock_broker.get_orders.return_value = [
        {
            "order_id": "1234567890",
            "tag": order.client_order_id[:20],
            "status": "OPEN",
        }
    ]
    mock_broker.get_order_status.return_value = OrderState.OPEN

    reconciler = ReconciliationEngine(store=store, broker=mock_broker)
    stats = await reconciler.reconcile()

    # The order should now be linked and updated
    saved_order = store.get(order.client_order_id)
    assert saved_order.broker_order_id == "1234567890"
    assert saved_order.state == OrderState.OPEN
    assert stats["resolved"] == 1


@pytest.mark.asyncio
async def test_reconciler_marks_rejected_if_not_found_on_broker(store):
    """
    If the order is missing broker_order_id and NOT found on the broker,
    we can safely assume it never reached the exchange.
    """
    order = intent_to_order(_make_intent("not_found_test"))
    order.state = OrderState.PENDING
    order.broker_order_id = None
    store.upsert(order)

    mock_broker = AsyncMock()
    # Simulate broker returning no matching orders
    mock_broker.get_orders.return_value = [
        {
            "order_id": "9999",
            "tag": "unrelated_tag",
            "status": "OPEN",
        }
    ]

    reconciler = ReconciliationEngine(store=store, broker=mock_broker)
    await reconciler.reconcile()

    saved_order = store.get(order.client_order_id)
    assert saved_order.broker_order_id is None
    assert saved_order.state == OrderState.REJECTED


# ---------------------------------------------------------------------------
# Tick Stream Edge Cases
# ---------------------------------------------------------------------------

def test_tick_stream_callback_exception_isolation(caplog):
    """
    Edge Case 11: Exceptions inside tick callbacks must not silently kill the stream.
    """
    import logging

    def failing_callback(ticks):
        raise ValueError("Simulated callback failure")

    adapter = KiteFeedAdapter(on_tick_raw=failing_callback)
    
    # We need a loop to run `call_soon_threadsafe`, mock it.
    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = False
    adapter._loop = mock_loop

    raw_tick = {
        "tradingsymbol": "CRUDEOIL",
        "last_price": 6000.0,
        "exchange_timestamp": datetime.now(timezone.utc),
    }

    with caplog.at_level(logging.ERROR):
        # Call the ticker callback directly
        adapter._on_ticks(ws=None, ticks=[raw_tick])

    # The callback exception should be caught and logged
    assert any("callback raised" in r.message for r in caplog.records)

    # The normalisation and enqueue should STILL happen despite the callback failing
    mock_loop.call_soon_threadsafe.assert_called_once()

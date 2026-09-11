"""
tests/unit/test_live_polling.py
-------------------------------
Tests for the background order polling task in LiveEngine.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.core.events import Fill, OrderIntent, OrderState, OrderType, Side
from engine.core.live_engine import LiveEngine
from engine.oms.idempotent_placer import IdempotentPlacer
from engine.oms.order import intent_to_order
from engine.oms.state_store import OrderStateStore


@pytest.fixture
def store():
    return OrderStateStore(":memory:")


def _make_pending_order(store: OrderStateStore, client_id: str, broker_id: str | None = None) -> str:
    intent = OrderIntent(
        symbol="CRUDEOIL",
        side=Side.BUY,
        qty=10,
        order_type=OrderType.MARKET,
        price=None,
        strategy_id="s1",
        tag=client_id,
    )
    order = intent_to_order(intent)
    order.state = OrderState.PENDING
    order.broker_order_id = broker_id
    store.upsert(order)
    return order.client_order_id


@pytest.mark.asyncio
async def test_polling_detects_completed_order(store):
    coid = _make_pending_order(store, "test1", "broker1")
    
    broker = AsyncMock()
    broker.get_orders.return_value = [
        {
            "order_id": "broker1",
            "filled_quantity": 10,
            "average_price": "6000.50",
            "status": "COMPLETE",
        }
    ]
    
    placer = IdempotentPlacer(store, broker)
    strategy = MagicMock()
    
    engine = LiveEngine(
        feed=MagicMock(), broker=broker, placer=placer, strategies=[strategy],
        normaliser=MagicMock(), bar_builder=MagicMock(), risk_gate=MagicMock(), indicators_state={},
        poll_interval_seconds=0
    )
    
    await engine._poll_orders_cycle()
    
    assert strategy.on_fill.call_count > 0
    fill = strategy.on_fill.call_args[0][0]
    assert fill.fill_qty == 10
    assert fill.fill_price == Decimal("6000.50")
    
    saved = store.get(coid)
    assert saved.state == OrderState.COMPLETE
    assert saved.filled_qty == 10


@pytest.mark.asyncio
async def test_polling_idempotent_fills(store):
    coid = _make_pending_order(store, "test2", "broker2")
    
    broker = AsyncMock()
    broker.get_orders.return_value = [{"order_id": "broker2", "filled_quantity": 10, "average_price": "6000.0", "status": "COMPLETE"}]
    
    placer = IdempotentPlacer(store, broker)
    strategy = MagicMock()
    engine = LiveEngine(feed=MagicMock(), broker=broker, placer=placer, strategies=[strategy], normaliser=MagicMock(), bar_builder=MagicMock(), risk_gate=MagicMock(), indicators_state={}, poll_interval_seconds=0)
    
    # First poll creates fill
    await engine._poll_orders_cycle()
    assert strategy.on_fill.call_count > 0
    strategy.on_fill.reset_mock()
    
    # Second poll should do NOTHING because incremental qty is 0 (order is COMPLETE and filled_qty matches)
    await engine._poll_orders_cycle()
    assert strategy.on_fill.call_count == 0


@pytest.mark.asyncio
async def test_polling_partial_fills(store):
    coid = _make_pending_order(store, "test3", "broker3")
    
    broker = AsyncMock()
    broker.get_orders.side_effect = [
        [{"order_id": "broker3", "filled_quantity": 4, "average_price": "6000.0", "status": "OPEN"}],
        [{"order_id": "broker3", "filled_quantity": 10, "average_price": "6001.0", "status": "COMPLETE"}],
        [{"order_id": "broker3", "filled_quantity": 10, "average_price": "6001.0", "status": "COMPLETE"}],
    ]
    
    placer = IdempotentPlacer(store, broker)
    strategy = MagicMock()
    engine = LiveEngine(feed=MagicMock(), broker=broker, placer=placer, strategies=[strategy], normaliser=MagicMock(), bar_builder=MagicMock(), risk_gate=MagicMock(), indicators_state={}, poll_interval_seconds=0)
    
    # Poll 1 -> 4 lots
    await engine._poll_orders_cycle()
    assert strategy.on_fill.call_count > 0
    assert store.get(coid).filled_qty == 4
    strategy.on_fill.reset_mock()
    
    # Poll 2 -> 6 lots more
    await engine._poll_orders_cycle()
    assert strategy.on_fill.call_count > 0
    second_fill = strategy.on_fill.call_args[0][0]
    assert second_fill.fill_qty == 6
    assert store.get(coid).filled_qty == 10
    assert store.get(coid).state == OrderState.COMPLETE


@pytest.mark.asyncio
async def test_polling_rejected_status(store):
    coid = _make_pending_order(store, "test4", "broker4")
    broker = AsyncMock()
    broker.get_orders.return_value = [{"order_id": "broker4", "filled_quantity": 0, "status": "REJECTED"}]
    
    placer = IdempotentPlacer(store, broker)
    engine = LiveEngine(feed=MagicMock(), broker=broker, placer=placer, strategies=[], normaliser=MagicMock(), bar_builder=MagicMock(), risk_gate=MagicMock(), indicators_state={}, poll_interval_seconds=0)
    
    await engine._poll_orders_cycle()
    assert store.get(coid).state == OrderState.REJECTED


@pytest.mark.asyncio
async def test_polling_missing_broker_id_recovery(store):
    # Order has NO broker_id yet
    coid = _make_pending_order(store, "test_recovery", None)
    
    broker = AsyncMock()
    broker.get_orders.return_value = [{"order_id": "b_recovered", "tag": coid[:20], "filled_quantity": 0, "status": "OPEN"}]
    
    placer = IdempotentPlacer(store, broker)
    engine = LiveEngine(feed=MagicMock(), broker=broker, placer=placer, strategies=[], normaliser=MagicMock(), bar_builder=MagicMock(), risk_gate=MagicMock(), indicators_state={}, poll_interval_seconds=0)
    
    await engine._poll_orders_cycle()
    assert store.get(coid).broker_order_id == "b_recovered"


@pytest.mark.asyncio
async def test_polling_exception_isolation(caplog, store):
    import logging
    coid = _make_pending_order(store, "test_exc", "broker_exc")
    
    broker = AsyncMock()
    broker.get_orders.side_effect = Exception("API down")
    
    placer = IdempotentPlacer(store, broker)
    engine = LiveEngine(feed=MagicMock(), broker=broker, placer=placer, strategies=[], normaliser=MagicMock(), bar_builder=MagicMock(), risk_gate=MagicMock(), indicators_state={}, poll_interval_seconds=0)
    
    with caplog.at_level(logging.ERROR):
        await engine._poll_orders_cycle()
        
    # Should log error and not crash
    assert "polling error: API down" in caplog.text
    # State should remain untouched (PENDING)
    assert store.get(coid).state == OrderState.PENDING


@pytest.mark.asyncio
async def test_placer_polling_race_condition(store):
    from engine.core.events import OrderIntent, Side, OrderType
    
    intent = OrderIntent(
        symbol="CRUDEOIL",
        side=Side.BUY,
        qty=10,
        order_type=OrderType.MARKET,
        price=None,
        strategy_id="s1",
        tag="race_test"
    )
    
    broker = AsyncMock()
    placer = IdempotentPlacer(store, broker)
    engine = LiveEngine(feed=MagicMock(), broker=broker, placer=placer, strategies=[], normaliser=MagicMock(), bar_builder=MagicMock(), risk_gate=MagicMock(), indicators_state={}, poll_interval_seconds=0)
    
    async def mock_place_order(order):
        # Simulate polling waking up and detecting a complete fill BEFORE place_order returns
        broker.get_orders.return_value = [{"order_id": "b_race", "tag": order.client_order_id[:20], "filled_quantity": 10, "average_price": "6000", "status": "COMPLETE"}]
        await engine._poll_orders_cycle()
        return "b_race"
        
    broker.place_order.side_effect = mock_place_order
    
    placed = await placer.place(intent)
    assert placed is not None
    
    # Assert the DB didn't get overwritten back to OPEN/0-qty
    saved = store.get(placed.client_order_id)
    assert saved.state == OrderState.COMPLETE
    assert saved.filled_qty == 10
    assert saved.broker_order_id == "b_race"

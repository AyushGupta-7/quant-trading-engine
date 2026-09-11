"""
tests/unit/test_live_engine.py
-------------------------------
Tests for the LiveEngine concurrency, backpressure, exception isolation,
and graceful shutdown logic.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.core.events import Bar, Tick, OrderIntent
from engine.core.live_engine import LiveEngine


class MockFeed:
    def __init__(self, num_ticks: int = 3):
        self.num_ticks = num_ticks
        self.connected = False
        self.disconnected = False
        self.stream_completed = False
        
    async def connect(self):
        self.connected = True
        
    async def disconnect(self):
        self.disconnected = True
        
    async def tick_stream(self):
        for i in range(self.num_ticks):
            yield {"raw": i}
            await asyncio.sleep(0.01)
        self.stream_completed = True


class MockStrategy:
    def __init__(self, intents: list[OrderIntent] = None, fail_on_bar: bool = False):
        self.intents = intents or []
        self.fail_on_bar = fail_on_bar
        self.bars_processed = 0
        self._config = {"symbols": ["CRUDEOIL"]}
        self._name = "MockStrategy"
        
    def on_bar(self, bar, indicators):
        if self.fail_on_bar:
            raise RuntimeError("Strategy crashed")
        self.bars_processed += 1
        return self.intents


@pytest.mark.asyncio
async def test_live_engine_startup_shutdown():
    feed = MockFeed()
    broker = AsyncMock()
    placer = AsyncMock()
    normaliser = MagicMock()
    bar_builder = MagicMock()
    risk_gate = MagicMock()
    
    engine = LiveEngine(
        feed=feed,
        broker=broker,
        placer=placer,
        strategies=[],
        normaliser=normaliser,
        bar_builder=bar_builder,
        risk_gate=risk_gate,
        indicators_state={}
    )
    
    await engine.start()
    assert feed.connected
    broker.connect.assert_awaited_once()
    assert engine._running
    
    await engine.stop()
    assert not engine._running
    assert feed.disconnected
    broker.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_engine_processes_ticks_sequentially():
    feed = MockFeed(num_ticks=2)
    broker = AsyncMock()
    broker.update_ltp = MagicMock()
    placer = AsyncMock()
    
    normaliser = MagicMock()
    normaliser.normalise.return_value = Tick(symbol="CRUDEOIL", ltp=100.0, volume=1, bid=100.0, ask=100.0, oi=0, ts=None)
    
    bar_builder = MagicMock()
    # Return a bar on every tick for testing
    bar_builder.update.return_value = Bar(symbol="CRUDEOIL", open=100, high=100, low=100, close=100, volume=1, ts=None)
    
    intent = OrderIntent(symbol="CRUDEOIL", side=None, qty=1, order_type=None, price=None, strategy_id="mock", tag="")
    strategy = MockStrategy(intents=[intent])
    
    risk_gate = MagicMock()
    risk_gate.approve.return_value = (True, "OK")
    
    engine = LiveEngine(
        feed=feed,
        broker=broker,
        placer=placer,
        strategies=[strategy],
        normaliser=normaliser,
        bar_builder=bar_builder,
        risk_gate=risk_gate,
        indicators_state={}
    )
    
    await engine.start()
    # Wait for processing task to finish all ticks
    await engine._process_task
    
    # 2 ticks = 2 bars = 2 strategy calls = 2 places
    assert strategy.bars_processed == 2
    assert placer.place.call_count == 2
    
    await engine.stop()


@pytest.mark.asyncio
async def test_live_engine_exception_isolation(caplog):
    import logging
    feed = MockFeed(num_ticks=1)
    broker = AsyncMock()
    broker.update_ltp = MagicMock()
    placer = AsyncMock()
    
    normaliser = MagicMock()
    normaliser.normalise.return_value = Tick(symbol="CRUDEOIL", ltp=100.0, volume=1, bid=100.0, ask=100.0, oi=0, ts=None)
    
    bar_builder = MagicMock()
    bar_builder.update.return_value = Bar(symbol="CRUDEOIL", open=100, high=100, low=100, close=100, volume=1, ts=None)
    
    # Strategy fails!
    strategy = MockStrategy(fail_on_bar=True)
    
    engine = LiveEngine(
        feed=feed,
        broker=broker,
        placer=placer,
        strategies=[strategy],
        normaliser=normaliser,
        bar_builder=bar_builder,
        risk_gate=MagicMock(),
        indicators_state={}
    )
    
    with caplog.at_level(logging.ERROR):
        await engine.start()
        await engine._process_task
        
    # The stream should have completed despite the error (task didn't crash)
    assert feed.stream_completed
    
    # Should log the specific error
    assert any("error in strategy MockStrategy: Strategy crashed" in r.message for r in caplog.records)
    
    await engine.stop()


@pytest.mark.asyncio
async def test_live_engine_cancellation_during_processing():
    # Provide a feed that runs forever until cancelled
    class InfiniteFeed(MockFeed):
        async def tick_stream(self):
            while True:
                yield {"raw": 1}
                await asyncio.sleep(1.0)
                
    feed = InfiniteFeed()
    broker = AsyncMock()
    broker.update_ltp = MagicMock()
    engine = LiveEngine(
        feed=feed,
        broker=broker,
        placer=AsyncMock(),
        strategies=[],
        normaliser=MagicMock(),
        bar_builder=MagicMock(),
        risk_gate=MagicMock(),
        indicators_state={}
    )
    
    await engine.start()
    
    # Let it run briefly
    await asyncio.sleep(0.1)
    
    # Stop triggers cancellation
    await engine.stop()
    
    assert not engine._running
    assert feed.disconnected

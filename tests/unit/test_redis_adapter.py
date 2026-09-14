import asyncio
import pytest
import json
from datetime import datetime
from decimal import Decimal
from engine.core.events import Side, OrderState, RegimeState
from engine.infrastructure.redis_adapter import RedisAdapter, EventEncoder

@pytest.fixture
def sample_event():
    return {
        "price": Decimal("100.5"),
        "ts": datetime(2023, 1, 1, 12, 0, 0),
        "side": Side.BUY,
        "state": OrderState.PENDING,
        "regime": RegimeState.BULL
    }

def test_event_encoder(sample_event):
    data = json.dumps(sample_event, cls=EventEncoder)
    parsed = json.loads(data)
    assert parsed["price"] == 100.5
    assert parsed["ts"] == "2023-01-01T12:00:00"
    assert parsed["side"] == "BUY"
    assert parsed["state"] == 1  # OrderState.PENDING is auto(), usually 1 if it's the first
    assert parsed["regime"] == "BULL"

@pytest.mark.asyncio
async def test_redis_adapter_no_connection():
    # Should not crash if redis is unavailable (using a dummy port)
    adapter = RedisAdapter("redis://localhost:9999/0")
    await adapter.connect()
    assert adapter._redis is None

    # Publish should silently ignore
    adapter.publish_nowait("test_channel", {"data": "test"})

    # Should disconnect without error
    await adapter.disconnect()

"""tests/unit/test_grid_kill_switch.py

Verifies that GridEngine correctly clears and emits CANCEL intents for all
pending limits when the kill switch is engaged.
"""

from decimal import Decimal
from typing import Any

from engine.core.events import Bar, OrderIntent, OrderType, Side
from engine.strategy.grid_engine import GridEngine


def _bar(close: float = 6000.0) -> Bar:
    from datetime import datetime, timezone
    p = Decimal(str(close))
    return Bar("CRUDEOIL", p, p + 5, p - 5, p, 1000, datetime.now(timezone.utc))


def test_grid_emits_cancel_when_kill_switch_engaged():
    cfg = {
        "atr_period": 5,
        "atr_multiplier": 1.0,
        "max_grid_levels": 2,
        "order_type": "LIMIT",
        "symbols": ["CRUDEOIL"],
    }
    grid = GridEngine("grid-test", cfg)

    ind = {"atr_5": 20.0}
    bar0 = _bar(6000.0)
    
    # 1. Normal bar produces limits
    intents = grid.on_bar(bar0, ind)
    assert len(intents) == 4
    assert all(i.order_type == OrderType.LIMIT for i in intents)
    
    assert len(grid.pending_levels) == 4

    # 2. Engage kill switch
    grid.on_kill_switch()
    assert grid.kill_switch_engaged is True
    
    # 3. Next bar should produce exactly 4 CANCEL intents and clear levels
    bar1 = _bar(6000.0)
    cancel_intents = grid.on_bar(bar1, ind)
    
    assert len(cancel_intents) == 4
    assert all(i.order_type == OrderType.CANCEL for i in cancel_intents)
    assert len(grid.pending_levels) == 0

    # 4. Subsequent bars should produce NO intents
    bar2 = _bar(6000.0)
    empty_intents = grid.on_bar(bar2, ind)
    assert len(empty_intents) == 0

def test_grid_no_duplicate_cancels():
    cfg = {
        "atr_period": 5,
        "atr_multiplier": 1.0,
        "max_grid_levels": 1,
    }
    grid = GridEngine("grid-test", cfg)
    grid.on_bar(_bar(), {"atr_5": 20.0})
    
    assert len(grid.pending_levels) == 2
    
    grid.on_kill_switch()
    intents1 = grid.on_bar(_bar(), {"atr_5": 20.0})
    assert len(intents1) == 2
    
    # Should be completely empty now
    intents2 = grid.on_bar(_bar(), {"atr_5": 20.0})
    assert len(intents2) == 0

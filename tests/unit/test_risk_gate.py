"""tests/unit/test_risk_gate.py"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from engine.core.events import OrderIntent, OrderType, Side
from engine.risk.drawdown_guard import DrawdownGuard
from engine.risk.kill_switch import KillSwitch
from engine.risk.position_cap import PositionCapGuard
from engine.risk.risk_gate import RiskGate


def make_intent(symbol: str = "CRUDEOIL", side: Side = Side.BUY, qty: int = 1) -> OrderIntent:
    return OrderIntent(
        symbol=symbol, side=side, qty=qty,
        order_type=OrderType.MARKET, price=None,
        strategy_id="test", tag="test",
    )


def make_gate(max_pos: int = 10, max_daily_loss: float = 1e9,
              max_dd: float = 1.0) -> RiskGate:
    ks  = KillSwitch(switch_file="NUL")
    ks._engaged = False
    dd  = DrawdownGuard(max_drawdown_pct=max_dd, max_daily_loss=max_daily_loss)
    pc  = PositionCapGuard(max_position_lots=max_pos, max_open_orders=100)
    return RiskGate(kill_switch=ks, drawdown_guard=dd, position_cap=pc)


class TestKillSwitch:
    def test_engage_blocks_orders(self):
        gate = make_gate()
        gate.engage_kill_switch("test")
        ok, reason = gate.approve(make_intent())
        assert not ok
        assert "kill switch" in reason.lower()

    def test_not_engaged_by_default(self):
        gate = make_gate()
        ok, _ = gate.approve(make_intent())
        assert ok

    def test_reset_unblocks_orders(self):
        gate = make_gate()
        gate.engage_kill_switch("test")
        gate.kill_switch.reset()
        ok, _ = gate.approve(make_intent())
        assert ok


class TestPositionCapGuard:
    def test_rejects_when_cap_exceeded(self):
        gate = make_gate(max_pos=2)
        gate.position_cap.update_position("CRUDEOIL", 2)  # already at cap
        ok, reason = gate.approve(make_intent(qty=1))
        assert not ok
        assert "cap" in reason.lower()

    def test_approves_when_within_cap(self):
        gate = make_gate(max_pos=5)
        gate.position_cap.update_position("CRUDEOIL", 2)
        ok, _ = gate.approve(make_intent(qty=1))
        assert ok

    def test_sell_reduces_position(self):
        gate = make_gate(max_pos=5)
        gate.position_cap.update_position("CRUDEOIL", 5)
        # Selling should be allowed (reduces position)
        ok, _ = gate.approve(make_intent(side=Side.SELL, qty=1))
        assert ok


class TestDrawdownGuard:
    def test_blocks_on_large_drawdown(self):
        gate = make_gate(max_dd=0.05)
        gate.drawdown_guard.update_equity(Decimal("100000"))
        gate.drawdown_guard.update_equity(Decimal("94000"))  # 6% drawdown
        ok, reason = gate.approve(make_intent())
        assert not ok
        assert "drawdown" in reason.lower()

    def test_allows_when_within_limit(self):
        gate = make_gate(max_dd=0.10)
        gate.drawdown_guard.update_equity(Decimal("100000"))
        gate.drawdown_guard.update_equity(Decimal("95000"))  # 5% dd
        ok, _ = gate.approve(make_intent())
        assert ok

    def test_blocks_on_daily_loss(self):
        gate = make_gate(max_daily_loss=1000.0)
        gate.drawdown_guard.update_equity(Decimal("10000"))
        gate.drawdown_guard.update_equity(Decimal("8500"))   # 1500 loss
        ok, reason = gate.approve(make_intent())
        assert not ok


class TestRiskGateCounters:
    def test_counters_increment(self):
        gate = make_gate(max_pos=1)
        gate.approve(make_intent())  # approved
        gate.position_cap.update_position("CRUDEOIL", 1)
        gate.approve(make_intent())  # rejected (cap)
        assert gate.approved_count == 1
        assert gate.rejected_count == 1

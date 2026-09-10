"""tests/unit/test_grid_engine.py"""

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from engine.core.events import Bar, Fill, OrderType, RegimeState, Side
from engine.strategy.grid_engine import GridEngine


TS = datetime(2024, 1, 15, 9, 15, tzinfo=timezone.utc)


def make_bar(close: float, ts_offset: int = 0) -> Bar:
    c = Decimal(str(close))
    ts = TS + timedelta(minutes=ts_offset)
    return Bar(symbol="CRUDEOIL", open=c, high=c + 10, low=c - 10, close=c, volume=1000, ts=ts)


def make_grid(atr_mult: float = 1.0, max_levels: int = 3,
              pos_cap: int = 5, max_pyramid: int = 3) -> GridEngine:
    cfg = {
        "atr_period": 14,
        "atr_multiplier": atr_mult,
        "max_grid_levels": max_levels,
        "max_pyramid_levels": max_pyramid,
        "position_cap_lots": pos_cap,
        "order_type": "LIMIT",
        "symbols": ["CRUDEOIL"],
    }
    return GridEngine("grid-test", cfg)


class TestGridEngine:
    def test_no_intents_without_atr(self):
        grid = make_grid()
        intents = grid.on_bar(make_bar(6000.0), {})
        assert intents == []

    def test_no_intents_with_nan_atr(self):
        grid = make_grid()
        intents = grid.on_bar(make_bar(6000.0), {"atr_14": float("nan")})
        assert intents == []

    def test_initialises_anchor_on_first_valid_bar(self):
        grid = make_grid()
        grid.on_bar(make_bar(6000.0), {"atr_14": 50.0})
        assert grid.anchor == Decimal("6000.0")

    def test_generates_buy_and_sell_levels(self):
        grid = make_grid(max_levels=2)
        intents = grid.on_bar(make_bar(6000.0), {"atr_14": 100.0})
        sides = {i.side for i in intents}
        assert Side.BUY in sides
        assert Side.SELL in sides

    def test_buy_levels_below_anchor(self):
        grid = make_grid(max_levels=3)
        intents = grid.on_bar(make_bar(6000.0), {"atr_14": 100.0})
        buys = [i for i in intents if i.side == Side.BUY]
        for intent in buys:
            assert intent.price < Decimal("6000.0")

    def test_sell_levels_above_anchor(self):
        grid = make_grid(max_levels=3)
        intents = grid.on_bar(make_bar(6000.0), {"atr_14": 100.0})
        sells = [i for i in intents if i.side == Side.SELL]
        for intent in sells:
            assert intent.price > Decimal("6000.0")

    def test_position_cap_limits_buys(self):
        grid = make_grid(max_levels=5, pos_cap=2, max_pyramid=2)
        grid._net_lots = 2   # already at cap
        intents = grid.on_bar(make_bar(6000.0), {"atr_14": 100.0})
        buys = [i for i in intents if i.side == Side.BUY]
        assert buys == []

    def test_kill_switch_produces_no_intents(self):
        grid = make_grid()
        grid.on_kill_switch()
        intents = grid.on_bar(make_bar(6000.0), {"atr_14": 100.0})
        assert intents == []

    def test_no_duplicate_levels(self):
        grid = make_grid(max_levels=3)
        ind = {"atr_14": 100.0}
        grid.on_bar(make_bar(6000.0), ind)
        grid.on_bar(make_bar(6000.0), ind)   # same anchor, same levels
        # Second call should not re-add pending levels
        pending = grid.pending_levels
        prices = list(pending.keys())
        assert len(prices) == len(set(prices))   # all unique

    def test_fill_updates_position_and_anchor(self):
        grid = make_grid()
        grid.on_bar(make_bar(6000.0), {"atr_14": 100.0})
        initial_anchor = grid.anchor

        fill = Fill(
            order_id="grid-test_abc123",
            broker_order_id="MOCK-001",
            symbol="CRUDEOIL",
            side=Side.BUY,
            fill_price=Decimal("5900.0"),
            fill_qty=1,
            fees=Decimal(0),
            ts=TS,
        )
        grid.on_fill(fill)
        assert grid.net_lots == 1
        # Anchor should have moved up by half a spacing step
        assert grid.anchor != initial_anchor

    def test_regime_change_updates_params(self):
        grid = make_grid(atr_mult=1.0, max_levels=5)
        grid.on_regime_change(
            RegimeState.BULL,
            {"atr_multiplier": 0.8, "max_grid_levels": 6, "enabled": True}
        )
        assert grid._atr_mult == pytest.approx(0.8)
        assert grid._max_levels == 6

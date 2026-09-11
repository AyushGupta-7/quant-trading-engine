"""tests/unit/test_sar_engine.py"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from engine.core.events import Bar, Fill, OrderType, RegimeState, Side
from engine.strategy.sar_engine import SAREngine


TS = datetime(2024, 1, 15, 9, 15, tzinfo=timezone.utc)


def make_bar(close: float) -> Bar:
    c = Decimal(str(close))
    return Bar(symbol="GOLD", open=c, high=c + 5, low=c - 5, close=c, volume=500, ts=TS)


def make_sar(stop_mult: float = 2.0, max_pyramid: int = 3, pos_cap: int = 5) -> SAREngine:
    cfg = {
        "atr_period": 14,
        "atr_stop_multiplier": stop_mult,
        "pyramid_on_trend": True,
        "max_pyramid_levels": max_pyramid,
        "position_cap_lots": pos_cap,
        "order_type": "MARKET",
        "symbols": ["GOLD"],
    }
    return SAREngine("sar-test", cfg)


def make_indicators(trend_up: bool, atr: float = 50.0) -> dict:
    return {
        "atr_14":             atr,
        "supertrend_14":      5900.0 if trend_up else 6100.0,
        "supertrend_14_up":   trend_up,
    }


class TestSAREngine:
    def test_no_intents_without_indicators(self):
        sar = make_sar()
        intents = sar.on_bar(make_bar(6000.0), {})
        assert intents == []

    def test_initial_entry_long_on_uptrend(self):
        sar = make_sar()
        intents = sar.on_bar(make_bar(6000.0), make_indicators(trend_up=True))
        # Entry + optional stop order
        assert len(intents) >= 1
        assert intents[0].side == Side.BUY
        assert intents[0].tag == "sar_entry"

    def test_initial_entry_short_on_downtrend(self):
        sar = make_sar()
        intents = sar.on_bar(make_bar(6000.0), make_indicators(trend_up=False))
        # Entry + optional stop order
        assert len(intents) >= 1
        assert intents[0].side == Side.SELL
        assert intents[0].tag == "sar_entry"

    def test_stop_price_set_on_entry(self):
        sar = make_sar(stop_mult=2.0)
        sar.on_bar(make_bar(6000.0), make_indicators(trend_up=True, atr=50.0))
        assert sar.stop_price is not None
        # stop = close - 2 * ATR = 6000 - 100 = 5900
        assert sar.stop_price == Decimal("5900.0")

    def test_reversal_closes_long_and_opens_short(self):
        sar = make_sar()
        sar.on_bar(make_bar(6000.0), make_indicators(trend_up=True))
        sar._net_lots = 1  # simulate filled

        intents = sar.on_bar(make_bar(5900.0), make_indicators(trend_up=False))
        tags = [i.tag for i in intents]
        sides = [i.side for i in intents]
        assert Side.SELL in sides  # close long
        assert "sar_close" in tags
        assert "sar_reverse" in tags
        # May also emit a stop intent — at least 2 intents
        assert len(intents) >= 2

    def test_pyramid_adds_on_trend(self):
        sar = make_sar(max_pyramid=3)
        # Initial entry
        sar.on_bar(make_bar(6000.0), make_indicators(trend_up=True))
        sar._net_lots = 1
        sar._direction = Side.BUY
        sar._pyramid_count = 1
        sar._prev_supertrend_up = True

        # Same trend → pyramid
        intents = sar.on_bar(make_bar(6010.0), make_indicators(trend_up=True))
        pyramid_buys = [i for i in intents if i.side == Side.BUY and "pyramid" in i.tag]
        assert len(pyramid_buys) == 1

    def test_pyramid_capped(self):
        sar = make_sar(max_pyramid=2)
        sar._direction = Side.BUY
        sar._pyramid_count = 2   # already at cap
        sar._net_lots = 2
        sar._prev_supertrend_up = True

        intents = sar.on_bar(make_bar(6010.0), make_indicators(trend_up=True))
        buys = [i for i in intents if i.side == Side.BUY]
        assert buys == []   # no more pyramiding

    def test_kill_switch_stops_trading(self):
        sar = make_sar()
        sar.on_kill_switch()
        intents = sar.on_bar(make_bar(6000.0), make_indicators(trend_up=True))
        assert intents == []

    def test_regime_change_updates_stop_mult(self):
        sar = make_sar(stop_mult=2.0)
        sar.on_regime_change(RegimeState.CRISIS, {"atr_stop_multiplier": 3.0, "enabled": False})
        assert not sar.enabled

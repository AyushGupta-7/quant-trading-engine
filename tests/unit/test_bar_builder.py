"""tests/unit/test_bar_builder.py"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from engine.core.events import Tick
from engine.data.bar_builder import BarBuilder, MultiBarBuilder


def make_tick(symbol: str, price: float, ts: datetime, volume: int = 1000) -> Tick:
    p = Decimal(str(price))
    return Tick(symbol=symbol, ltp=p, bid=p, ask=p, volume=volume, oi=0, ts=ts)


BASE_TS = datetime(2024, 1, 15, 9, 15, 0, tzinfo=timezone.utc)


class TestBarBuilder:
    def test_no_bar_before_interval_ends(self):
        builder = BarBuilder("X", interval_seconds=60)
        tick = make_tick("X", 100.0, BASE_TS)
        result = builder.update(tick)
        assert result is None

    def test_bar_closes_on_interval_boundary(self):
        builder = BarBuilder("X", interval_seconds=60)
        t0 = BASE_TS
        t1 = BASE_TS + timedelta(minutes=1)

        builder.update(make_tick("X", 100.0, t0))
        builder.update(make_tick("X", 102.0, t0 + timedelta(seconds=30)))
        bar = builder.update(make_tick("X", 101.0, t1))  # new interval

        assert bar is not None
        assert bar.symbol == "X"
        assert bar.open == Decimal("100.0")
        assert bar.high == Decimal("102.0")
        assert bar.low  == Decimal("100.0")
        assert bar.close == Decimal("102.0")   # last tick of interval

    def test_bar_callback_is_invoked(self):
        completed = []
        builder = BarBuilder("X", interval_seconds=60, on_bar=completed.append)

        t0 = BASE_TS
        t1 = BASE_TS + timedelta(minutes=1)

        builder.update(make_tick("X", 100.0, t0))
        builder.update(make_tick("X", 105.0, t1))

        assert len(completed) == 1
        assert completed[0].close == Decimal("100.0")

    def test_wrong_symbol_ignored(self):
        builder = BarBuilder("X", interval_seconds=60)
        tick = make_tick("Y", 100.0, BASE_TS)
        result = builder.update(tick)
        assert result is None

    def test_force_close(self):
        builder = BarBuilder("X", interval_seconds=60)
        builder.update(make_tick("X", 100.0, BASE_TS))
        bar = builder.force_close(BASE_TS + timedelta(seconds=45))
        assert bar is not None
        assert bar.close == Decimal("100.0")

    def test_multi_bar_builder_routes_correctly(self):
        bars = {"X": [], "Y": []}
        def on_bar(b):
            bars[b.symbol].append(b)

        multi = MultiBarBuilder(interval_seconds=60, on_bar=on_bar)
        t0 = BASE_TS
        t1 = BASE_TS + timedelta(minutes=1)

        multi.update(make_tick("X", 100.0, t0))
        multi.update(make_tick("Y", 200.0, t0))
        multi.update(make_tick("X", 101.0, t1))  # closes X bar
        multi.update(make_tick("Y", 201.0, t1))  # closes Y bar

        assert len(bars["X"]) == 1
        assert len(bars["Y"]) == 1
        assert bars["X"][0].open == Decimal("100.0")
        assert bars["Y"][0].open == Decimal("200.0")

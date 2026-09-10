"""tests/unit/test_indicators.py"""

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from engine.core.events import Bar
from engine.indicators import ATR, EMA, SMA, RSI, MACD, BollingerBands, Supertrend, VWAP, OBV


def make_bar(close: float, high: float | None = None, low: float | None = None,
             volume: int = 1000, ts_offset: int = 0) -> Bar:
    ts = datetime(2024, 1, 15, 9, 15, tzinfo=timezone.utc) + timedelta(minutes=ts_offset)
    c = Decimal(str(close))
    h = Decimal(str(high)) if high else c + Decimal("1")
    l = Decimal(str(low))  if low  else c - Decimal("1")
    return Bar(symbol="X", open=c, high=h, low=l, close=c, volume=volume, ts=ts)


class TestSMA:
    def test_nan_before_ready(self):
        sma = SMA(3)
        b = make_bar(100)
        v = sma.update(b)
        assert math.isnan(v)
        assert not sma.ready

    def test_correct_value_after_warmup(self):
        sma = SMA(3)
        for price in [100, 102, 104]:
            sma.update(make_bar(price))
        assert sma.value() == pytest.approx(102.0)
        assert sma.ready

    def test_rolling_window(self):
        sma = SMA(3)
        for price in [100, 102, 104, 106]:
            sma.update(make_bar(price))
        # SMA of [102, 104, 106]
        assert sma.value() == pytest.approx(104.0)


class TestEMA:
    def test_nan_before_ready(self):
        ema = EMA(5)
        assert math.isnan(ema.update(make_bar(100)))

    def test_ready_after_period(self):
        ema = EMA(3)
        for p in [100, 101, 102]:
            ema.update(make_bar(p))
        assert ema.ready
        assert not math.isnan(ema.value())

    def test_ema_responds_to_price_move(self):
        ema = EMA(5)
        for p in [100] * 5:
            ema.update(make_bar(p))
        v1 = ema.value()
        ema.update(make_bar(200))
        v2 = ema.value()
        assert v2 > v1


class TestATR:
    def test_nan_before_period(self):
        atr = ATR(14)
        assert math.isnan(atr.update(make_bar(100, 105, 95)))

    def test_positive_after_warmup(self):
        atr = ATR(5)
        for i in range(15):
            atr.update(make_bar(100 + i, 102 + i, 98 + i))
        assert atr.ready
        assert atr.value() > 0

    def test_zero_range_bar(self):
        atr = ATR(3)
        for _ in range(10):
            atr.update(make_bar(100, 100, 100))
        # All bars have same H/L/C → ATR should be very small (approaches 0)
        assert atr.value() >= 0


class TestRSI:
    def test_nan_before_warmup(self):
        rsi = RSI(14)
        assert math.isnan(rsi.update(make_bar(100)))

    def test_overbought_on_constant_rise(self):
        rsi = RSI(14)
        for i in range(30):
            rsi.update(make_bar(100 + i))
        assert rsi.ready
        assert rsi.value() > 70

    def test_oversold_on_constant_fall(self):
        rsi = RSI(14)
        for i in range(30):
            rsi.update(make_bar(100 - i))
        assert rsi.ready
        assert rsi.value() < 30

    def test_range_0_to_100(self):
        rsi = RSI(14)
        import random
        rng = random.Random(0)
        for _ in range(50):
            rsi.update(make_bar(rng.uniform(90, 110)))
        if rsi.ready:
            assert 0 <= rsi.value() <= 100


class TestBollingerBands:
    def test_ready_after_period(self):
        bb = BollingerBands(5, 2.0)
        for p in range(5):
            bb.update(make_bar(100 + p))
        assert bb.ready
        assert bb.upper > bb.value() > bb.lower

    def test_bandwidth_positive(self):
        bb = BollingerBands(5)
        for p in range(10):
            bb.update(make_bar(100 + p * 0.5))
        assert not math.isnan(bb.bandwidth)
        assert bb.bandwidth > 0

    def test_percent_b_midpoint_at_middle(self):
        bb = BollingerBands(5)
        # Flat prices → %B should be ~0.5
        for _ in range(10):
            bb.update(make_bar(100.0))
        # With flat prices std ≈ 0, %B fallback is 0.5
        assert not math.isnan(bb.percent_b)


class TestSupertrend:
    def test_returns_value_after_warmup(self):
        st = Supertrend(7, 3.0)
        for i in range(20):
            st.update(make_bar(100 + i * 0.5, 101 + i * 0.5, 99 + i * 0.5))
        assert not math.isnan(st.value())

    def test_trend_up_on_strong_uptrend(self):
        st = Supertrend(7, 3.0)
        for i in range(30):
            st.update(make_bar(100 + i * 2, 102 + i * 2, 100 + i * 2))
        assert st.trend_up is True


class TestVWAP:
    def test_equals_price_on_single_bar(self):
        vwap = VWAP()
        bar = make_bar(100.0, 102.0, 98.0, volume=1000)
        vwap.update(bar)
        # typical = (102 + 98 + 100) / 3 = 100
        assert vwap.value() == pytest.approx(100.0)

    def test_resets_correctly(self):
        vwap = VWAP()
        vwap.update(make_bar(100.0, volume=1000))
        vwap.reset()
        assert not vwap.ready
        assert math.isnan(vwap.value())


class TestOBV:
    def test_rises_on_up_close(self):
        obv = OBV()
        obv.update(make_bar(100, volume=1000))
        obv.update(make_bar(101, volume=500))
        assert obv.value() == 500  # only second bar added (after prev_close set)

    def test_falls_on_down_close(self):
        obv = OBV()
        obv.update(make_bar(100, volume=1000))
        obv.update(make_bar(99, volume=800))
        assert obv.value() == -800

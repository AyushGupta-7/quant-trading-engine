"""
engine/indicators/momentum.py
------------------------------
Momentum indicators: RSI, MACD, Stochastic Oscillator.
"""

from __future__ import annotations

import math
from collections import deque

from engine.core.events import Bar
from engine.indicators.base import IIndicator
from engine.indicators.trend import EMA, WilderSMA


class RSI(IIndicator):
    """
    Relative Strength Index (Wilder smoothing, standard 14-period).

    Values:
        > 70 → overbought
        < 30 → oversold
    """

    def __init__(self, period: int = 14) -> None:
        super().__init__(period)
        self._gain_smoother = WilderSMA(period)
        self._loss_smoother = WilderSMA(period)
        self._prev_close: float | None = None
        self._value = float("nan")

    def _calculate(self, bar: Bar) -> float:
        close = float(bar.close)

        if self._prev_close is None:
            self._prev_close = close
            return float("nan")

        delta = close - self._prev_close
        self._prev_close = close

        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)

        avg_gain = self._gain_smoother.update_value(gain)
        avg_loss = self._loss_smoother.update_value(loss)

        if math.isnan(avg_gain) or math.isnan(avg_loss):
            return float("nan")

        if avg_loss == 0:
            self._value = 100.0
        else:
            rs = avg_gain / avg_loss
            self._value = 100.0 - (100.0 / (1 + rs))

        return self._value

    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._gain_smoother.reset()
        self._loss_smoother.reset()
        self._prev_close = None
        self._value = float("nan")
        self._bar_count = 0


class MACD(IIndicator):
    """
    MACD = EMA(fast) − EMA(slow).
    Signal = EMA(MACD, signal_period).
    Histogram = MACD − Signal.

    Default: 12, 26, 9.
    """

    def __init__(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> None:
        super().__init__(slow)   # warm-up = slow period
        self._fast_ema   = EMA(fast)
        self._slow_ema   = EMA(slow)
        self._signal_ema = EMA(signal)
        self._macd_val   = float("nan")
        self._signal_val = float("nan")
        self._hist_val   = float("nan")

    def _calculate(self, bar: Bar) -> float:
        fast_val = self._fast_ema._calculate(bar)
        slow_val = self._slow_ema._calculate(bar)

        if not self._slow_ema._initialised:
            return float("nan")

        self._macd_val = fast_val - slow_val

        # Feed MACD into signal EMA
        from engine.core.events import Bar as _Bar
        from decimal import Decimal
        proxy_bar = _Bar(
            symbol=bar.symbol,
            open=Decimal(str(self._macd_val)),
            high=Decimal(str(self._macd_val)),
            low=Decimal(str(self._macd_val)),
            close=Decimal(str(self._macd_val)),
            volume=bar.volume,
            ts=bar.ts,
        )
        self._signal_val = self._signal_ema._calculate(proxy_bar)

        if not math.isnan(self._signal_val):
            self._hist_val = self._macd_val - self._signal_val

        return self._macd_val

    @property
    def macd(self) -> float:
        return self._macd_val

    @property
    def signal(self) -> float:
        return self._signal_val

    @property
    def histogram(self) -> float:
        return self._hist_val

    def value(self) -> float:
        return self._macd_val

    def reset(self) -> None:
        self._fast_ema.reset()
        self._slow_ema.reset()
        self._signal_ema.reset()
        self._macd_val = self._signal_val = self._hist_val = float("nan")
        self._bar_count = 0


class Stochastic(IIndicator):
    """
    Stochastic Oscillator %K and %D.

    Parameters
    ----------
    k_period:
        Lookback for raw %K (default 14).
    d_period:
        Smoothing for %D (SMA of %K, default 3).
    smooth_k:
        Additional smoothing of %K (1 = no extra smoothing).
    """

    def __init__(
        self, k_period: int = 14, d_period: int = 3, smooth_k: int = 1
    ) -> None:
        super().__init__(k_period)
        self._highs: deque[float] = deque(maxlen=k_period)
        self._lows:  deque[float] = deque(maxlen=k_period)
        self._k_buffer: deque[float] = deque(maxlen=smooth_k)
        self._d_buffer: deque[float] = deque(maxlen=d_period)
        self._d_period = d_period
        self._pct_k = float("nan")
        self._pct_d = float("nan")

    def _calculate(self, bar: Bar) -> float:
        self._highs.append(float(bar.high))
        self._lows.append(float(bar.low))
        close = float(bar.close)

        if len(self._highs) < self._period:
            return float("nan")

        highest = max(self._highs)
        lowest  = min(self._lows)
        rng = highest - lowest

        raw_k = 100.0 * (close - lowest) / rng if rng > 0 else 50.0
        self._k_buffer.append(raw_k)
        self._pct_k = sum(self._k_buffer) / len(self._k_buffer)

        self._d_buffer.append(self._pct_k)
        if len(self._d_buffer) == self._d_period:
            self._pct_d = sum(self._d_buffer) / self._d_period

        return self._pct_k

    @property
    def pct_k(self) -> float:
        return self._pct_k

    @property
    def pct_d(self) -> float:
        return self._pct_d

    def value(self) -> float:
        return self._pct_k

    def reset(self) -> None:
        self._highs.clear()
        self._lows.clear()
        self._k_buffer.clear()
        self._d_buffer.clear()
        self._pct_k = self._pct_d = float("nan")
        self._bar_count = 0

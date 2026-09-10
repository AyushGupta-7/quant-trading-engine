"""
engine/indicators/trend.py
--------------------------
Trend-following indicators: SMA, EMA, ADX, Supertrend.

All use Wilder smoothing where indicated (matching TA-Lib defaults).
"""

from __future__ import annotations

import math
from collections import deque

from engine.core.events import Bar
from engine.indicators.base import IIndicator


class SMA(IIndicator):
    """Simple Moving Average."""

    def __init__(self, period: int) -> None:
        super().__init__(period)
        self._window: deque[float] = deque(maxlen=period)
        self._value = float("nan")

    def _calculate(self, bar: Bar) -> float:
        self._window.append(float(bar.close))
        if len(self._window) == self._period:
            self._value = sum(self._window) / self._period
        return self._value

    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._window.clear()
        self._value = float("nan")
        self._bar_count = 0


class EMA(IIndicator):
    """
    Exponential Moving Average.

    Uses the standard multiplier k = 2 / (period + 1).
    Initialised with SMA of the first *period* bars.
    """

    def __init__(self, period: int) -> None:
        super().__init__(period)
        self._k = 2.0 / (period + 1)
        self._init_buffer: deque[float] = deque(maxlen=period)
        self._value = float("nan")
        self._initialised = False

    def _calculate(self, bar: Bar) -> float:
        close = float(bar.close)
        if not self._initialised:
            self._init_buffer.append(close)
            if len(self._init_buffer) == self._period:
                self._value = sum(self._init_buffer) / self._period
                self._initialised = True
        else:
            self._value = close * self._k + self._value * (1 - self._k)
        return self._value

    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._init_buffer.clear()
        self._value = float("nan")
        self._initialised = False
        self._bar_count = 0


class WilderSMA(IIndicator):
    """
    Wilder's Smoothed Moving Average (used inside ATR, ADX, RSI).

    Multiplier: k = 1 / period.
    """

    def __init__(self, period: int) -> None:
        super().__init__(period)
        self._k = 1.0 / period
        self._init_buffer: deque[float] = deque(maxlen=period)
        self._value = float("nan")
        self._initialised = False

    def update_value(self, v: float) -> float:
        """Accept a pre-computed value (used by ATR, ADX, RSI internally)."""
        if not self._initialised:
            self._init_buffer.append(v)
            if len(self._init_buffer) == self._period:
                self._value = sum(self._init_buffer) / self._period
                self._initialised = True
        else:
            self._value = v * self._k + self._value * (1 - self._k)
        self._bar_count += 1
        return self._value

    def _calculate(self, bar: Bar) -> float:
        return self.update_value(float(bar.close))

    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._init_buffer.clear()
        self._value = float("nan")
        self._initialised = False
        self._bar_count = 0


class ADX(IIndicator):
    """
    Average Directional Index (Wilder, 14-period default).

    Exposes ``adx``, ``plus_di``, ``minus_di`` after burn-in.
    """

    def __init__(self, period: int = 14) -> None:
        super().__init__(period)
        self._prev_high: float | None = None
        self._prev_low:  float | None = None
        self._prev_close: float | None = None
        self._atr_smoother  = WilderSMA(period)
        self._pdm_smoother  = WilderSMA(period)  # +DM
        self._ndm_smoother  = WilderSMA(period)  # -DM
        self._dx_smoother   = WilderSMA(period)
        self._adx  = float("nan")
        self._pdi  = float("nan")
        self._ndi  = float("nan")

    def _calculate(self, bar: Bar) -> float:
        high  = float(bar.high)
        low   = float(bar.low)
        close = float(bar.close)

        if self._prev_close is None:
            self._prev_high, self._prev_low, self._prev_close = high, low, close
            return float("nan")

        # True range
        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low  - self._prev_close),
        )

        # Directional movement
        up   = high - self._prev_high
        down = self._prev_low - low
        pdm  = up   if (up > down and up > 0)   else 0.0
        ndm  = down if (down > up and down > 0) else 0.0

        self._prev_high, self._prev_low, self._prev_close = high, low, close

        atr_val = self._atr_smoother.update_value(tr)
        pdm_val = self._pdm_smoother.update_value(pdm)
        ndm_val = self._ndm_smoother.update_value(ndm)

        if atr_val and atr_val > 0:
            self._pdi = 100.0 * pdm_val / atr_val
            self._ndi = 100.0 * ndm_val / atr_val
            di_sum = self._pdi + self._ndi
            if di_sum > 0:
                dx = 100.0 * abs(self._pdi - self._ndi) / di_sum
                adx_val = self._dx_smoother.update_value(dx)
                if self._dx_smoother.ready:
                    self._adx = adx_val

        return self._adx

    @property
    def adx(self) -> float:
        return self._adx

    @property
    def plus_di(self) -> float:
        return self._pdi

    @property
    def minus_di(self) -> float:
        return self._ndi

    def value(self) -> float:
        return self._adx

    def reset(self) -> None:
        self._prev_high = self._prev_low = self._prev_close = None
        self._atr_smoother.reset()
        self._pdm_smoother.reset()
        self._ndm_smoother.reset()
        self._dx_smoother.reset()
        self._adx = self._pdi = self._ndi = float("nan")
        self._bar_count = 0


class Supertrend(IIndicator):
    """
    Supertrend indicator based on ATR.

    Parameters
    ----------
    period:
        ATR period (default 7).
    multiplier:
        Band multiplier (default 3.0).
    """

    def __init__(self, period: int = 7, multiplier: float = 3.0) -> None:
        super().__init__(period)
        self._mult = multiplier
        self._atr_smoother = WilderSMA(period)
        self._prev_close: float | None = None
        self._prev_high:  float | None = None
        self._prev_low:   float | None = None

        self._upper_band = float("nan")
        self._lower_band = float("nan")
        self._supertrend = float("nan")
        self._trend_up  = True     # True = uptrend, False = downtrend

    def _calculate(self, bar: Bar) -> float:
        high  = float(bar.high)
        low   = float(bar.low)
        close = float(bar.close)
        hl2   = (high + low) / 2

        if self._prev_close is None:
            self._prev_high, self._prev_low, self._prev_close = high, low, close
            return float("nan")

        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low  - self._prev_close),
        )
        atr = self._atr_smoother.update_value(tr)
        self._prev_high, self._prev_low, self._prev_close = high, low, close

        if math.isnan(atr):
            return float("nan")

        basic_upper = hl2 + self._mult * atr
        basic_lower = hl2 - self._mult * atr

        # Band tightening rule
        prev_upper = self._upper_band if not math.isnan(self._upper_band) else basic_upper
        prev_lower = self._lower_band if not math.isnan(self._lower_band) else basic_lower
        prev_trend_up = self._trend_up

        self._upper_band = basic_upper if basic_upper < prev_upper or close > prev_upper else prev_upper
        self._lower_band = basic_lower if basic_lower > prev_lower or close < prev_lower else prev_lower

        if prev_trend_up:
            self._trend_up = close >= self._lower_band
        else:
            self._trend_up = close > self._upper_band

        self._supertrend = self._lower_band if self._trend_up else self._upper_band
        return self._supertrend

    def value(self) -> float:
        return self._supertrend

    @property
    def trend_up(self) -> bool:
        return self._trend_up

    def reset(self) -> None:
        self._atr_smoother.reset()
        self._prev_close = self._prev_high = self._prev_low = None
        self._upper_band = self._lower_band = self._supertrend = float("nan")
        self._trend_up = True
        self._bar_count = 0

"""
engine/indicators/volume.py
----------------------------
Volume indicators: VWAP, OBV, Volume Moving Average.
"""

from __future__ import annotations

import math
from collections import deque

from engine.core.events import Bar
from engine.indicators.base import IIndicator


class VWAP(IIndicator):
    """
    Volume-Weighted Average Price (session-based, resets each day).

    Tracks cumulative (price × volume) / cumulative volume since reset.
    Call ``reset()`` at the start of each trading session.
    """

    def __init__(self) -> None:
        super().__init__(period=1)   # No warmup needed
        self._cum_pv  = 0.0
        self._cum_vol = 0
        self._value   = float("nan")

    def _calculate(self, bar: Bar) -> float:
        tp = float(bar.typical_price())  # (H+L+C)/3
        vol = bar.volume
        self._cum_pv  += tp * vol
        self._cum_vol += vol
        if self._cum_vol > 0:
            self._value = self._cum_pv / self._cum_vol
        return self._value

    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._cum_pv = 0.0
        self._cum_vol = 0
        self._value = float("nan")
        self._bar_count = 0

    @property
    def ready(self) -> bool:
        return self._cum_vol > 0


class OBV(IIndicator):
    """
    On-Balance Volume.

    OBV rises on up-bars (close > prev_close), falls on down-bars.
    """

    def __init__(self) -> None:
        super().__init__(period=2)   # Needs one previous close
        self._prev_close: float | None = None
        self._obv = 0.0

    def _calculate(self, bar: Bar) -> float:
        close = float(bar.close)
        vol   = bar.volume

        if self._prev_close is None:
            self._prev_close = close
            return float("nan")

        if close > self._prev_close:
            self._obv += vol
        elif close < self._prev_close:
            self._obv -= vol
        # Unchanged close → OBV unchanged

        self._prev_close = close
        return self._obv

    def value(self) -> float:
        return self._obv

    def reset(self) -> None:
        self._prev_close = None
        self._obv = 0.0
        self._bar_count = 0


class VolumeMA(IIndicator):
    """
    Simple Moving Average of volume.

    Useful for detecting volume surges (bar_volume > k × VolumeMA).
    """

    def __init__(self, period: int = 20) -> None:
        super().__init__(period)
        self._window: deque[float] = deque(maxlen=period)
        self._value = float("nan")

    def _calculate(self, bar: Bar) -> float:
        self._window.append(float(bar.volume))
        if len(self._window) == self._period:
            self._value = sum(self._window) / self._period
        return self._value

    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._window.clear()
        self._value = float("nan")
        self._bar_count = 0

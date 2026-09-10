"""
engine/indicators/volatility.py
--------------------------------
Volatility indicators: ATR, Bollinger Bands, Historical Volatility.

ATR uses Wilder smoothing (matches TA-Lib / TradingView default).
"""

from __future__ import annotations

import math
from collections import deque

from engine.core.events import Bar
from engine.indicators.base import IIndicator
from engine.indicators.trend import WilderSMA


class ATR(IIndicator):
    """
    Average True Range (Wilder smoothing).

    ATR is a primary input for grid spacing, stop placement, and supertrend.
    """

    def __init__(self, period: int = 14) -> None:
        super().__init__(period)
        self._smoother = WilderSMA(period)
        self._prev_close: float | None = None
        self._value = float("nan")

    def _calculate(self, bar: Bar) -> float:
        high  = float(bar.high)
        low   = float(bar.low)
        close = float(bar.close)

        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low  - self._prev_close),
            )

        self._prev_close = close
        self._value = self._smoother.update_value(tr)
        return self._value

    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._smoother.reset()
        self._prev_close = None
        self._value = float("nan")
        self._bar_count = 0


class BollingerBands(IIndicator):
    """
    Bollinger Bands: middle (SMA), upper, lower.

    Parameters
    ----------
    period:
        SMA period (default 20).
    std_dev:
        Number of standard deviations for bands (default 2.0).
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0) -> None:
        super().__init__(period)
        self._k = std_dev
        self._window: deque[float] = deque(maxlen=period)
        self._middle = float("nan")
        self._upper  = float("nan")
        self._lower  = float("nan")
        self._bandwidth = float("nan")
        self._percent_b = float("nan")

    def _calculate(self, bar: Bar) -> float:
        close = float(bar.close)
        self._window.append(close)

        if len(self._window) < self._period:
            return float("nan")

        mean = sum(self._window) / self._period
        variance = sum((x - mean) ** 2 for x in self._window) / self._period
        std = math.sqrt(variance)

        self._middle = mean
        self._upper  = mean + self._k * std
        self._lower  = mean - self._k * std
        self._bandwidth = (self._upper - self._lower) / mean if mean else float("nan")
        self._percent_b = (close - self._lower) / (self._upper - self._lower) if (self._upper != self._lower) else 0.5

        return self._middle

    def value(self) -> float:
        return self._middle

    @property
    def upper(self) -> float:
        return self._upper

    @property
    def lower(self) -> float:
        return self._lower

    @property
    def bandwidth(self) -> float:
        return self._bandwidth

    @property
    def percent_b(self) -> float:
        return self._percent_b

    def reset(self) -> None:
        self._window.clear()
        self._middle = self._upper = self._lower = float("nan")
        self._bandwidth = self._percent_b = float("nan")
        self._bar_count = 0


class HistoricalVolatility(IIndicator):
    """
    Annualised close-to-close Historical Volatility (log returns).

    Parameters
    ----------
    period:
        Lookback for standard deviation of log returns.
    trading_periods:
        Annualisation factor (252 for equities/commodities).
    """

    def __init__(self, period: int = 20, trading_periods: int = 252) -> None:
        super().__init__(period)
        self._ann = math.sqrt(trading_periods)
        self._log_returns: deque[float] = deque(maxlen=period)
        self._prev_close: float | None = None
        self._value = float("nan")

    def _calculate(self, bar: Bar) -> float:
        close = float(bar.close)

        if self._prev_close is not None and self._prev_close > 0:
            lr = math.log(close / self._prev_close)
            self._log_returns.append(lr)

        self._prev_close = close

        if len(self._log_returns) < self._period:
            return float("nan")

        mean = sum(self._log_returns) / len(self._log_returns)
        variance = sum((r - mean) ** 2 for r in self._log_returns) / (len(self._log_returns) - 1)
        self._value = math.sqrt(variance) * self._ann
        return self._value

    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._log_returns.clear()
        self._prev_close = None
        self._value = float("nan")
        self._bar_count = 0

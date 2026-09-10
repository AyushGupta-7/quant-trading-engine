"""
engine/core/clock.py
--------------------
Abstract clock interface with two concrete implementations:

  SimClock  – deterministic, controlled by the backtest engine.
  LiveClock – delegates to system time.

All timestamps produced by the engine flow through a Clock instance so that
backtests are provably free of look-ahead: the clock never advances beyond the
bar being processed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone


class Clock(ABC):
    """Abstract time source for the engine."""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current logical time as a timezone-aware UTC datetime."""

    @abstractmethod
    def now_ts(self) -> float:
        """Return the current logical time as a POSIX timestamp (seconds)."""


class LiveClock(Clock):
    """Wall-clock implementation for live trading."""

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def now_ts(self) -> float:
        return time.time()


class SimClock(Clock):
    """
    Simulated clock for backtesting.

    The backtest engine advances the clock before processing each bar so that
    strategy code never has access to future time.
    """

    def __init__(self, start: datetime | None = None) -> None:
        if start is None:
            start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        if start.tzinfo is None:
            raise ValueError("SimClock requires a timezone-aware datetime.")
        self._current: datetime = start

    def now(self) -> datetime:
        return self._current

    def now_ts(self) -> float:
        return self._current.timestamp()

    def advance(self, dt: datetime) -> None:
        """Advance the clock to *dt*.  Raises if dt < current (no backwards travel)."""
        if dt < self._current:
            raise ValueError(
                f"SimClock cannot go backwards: current={self._current}, requested={dt}"
            )
        self._current = dt

    def advance_seconds(self, seconds: float) -> None:
        from datetime import timedelta
        self._current += timedelta(seconds=seconds)

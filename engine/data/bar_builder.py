"""
engine/data/bar_builder.py
--------------------------
Aggregates a stream of ``Tick`` events into fixed-interval OHLCV ``Bar`` events.

Key design choices
~~~~~~~~~~~~~~~~~~
* One ``BarBuilder`` instance per symbol.
* The builder is called with each normalised tick.
* When the bar interval is complete it calls the registered ``on_bar``
  callback — making it easy to unit-test without asyncio.
* The backtest engine controls when bars close by calling
  ``force_close(ts)``; this guarantees no look-ahead.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

from engine.core.events import Bar, Tick


OnBarCallback = Callable[[Bar], None]


class BarBuilder:
    """
    Builds OHLCV bars from ticks.

    Parameters
    ----------
    symbol:
        The instrument symbol to aggregate.
    interval_seconds:
        Bar duration in seconds (default 60 = 1-minute bars).
    on_bar:
        Callback invoked with the completed ``Bar``.
    """

    def __init__(
        self,
        symbol: str,
        interval_seconds: int = 60,
        on_bar: OnBarCallback | None = None,
    ) -> None:
        self.symbol = symbol
        self.interval = timedelta(seconds=interval_seconds)
        self.on_bar = on_bar

        # Current bar state
        self._open: Decimal | None = None
        self._high: Decimal | None = None
        self._low: Decimal | None = None
        self._close: Decimal | None = None
        self._volume: int = 0
        self._bar_start: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tick: Tick) -> Bar | None:
        """
        Process a tick.  Returns a completed ``Bar`` if the current bar
        interval has elapsed, otherwise returns ``None``.
        """
        if tick.symbol != self.symbol:
            return None

        # Compute bar bucket for this tick
        bar_ts = self._bucket_start(tick.ts)

        # First tick ever
        if self._bar_start is None:
            self._start_bar(bar_ts, tick)
            return None

        # Still within the same bar
        if bar_ts == self._bar_start:
            self._update_ohlcv(tick)
            return None

        # New bar started → close the current one, start a new one
        completed = self._close_bar()
        self._start_bar(bar_ts, tick)
        if self.on_bar and completed:
            self.on_bar(completed)
        return completed

    def force_close(self, ts: datetime) -> Bar | None:
        """
        Force-close the current bar at *ts* (used by backtest engine at EOD
        or on session end).  Returns the bar or ``None`` if no data exists.
        """
        if self._open is None:
            return None
        bar = Bar(
            symbol=self.symbol,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            ts=ts,
        )
        self._reset()
        if self.on_bar:
            self.on_bar(bar)
        return bar

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bucket_start(self, ts: datetime) -> datetime:
        """Floor *ts* to the nearest bar interval boundary (UTC epoch-based)."""
        epoch = ts.timestamp()
        interval_sec = self.interval.total_seconds()
        floored = int(epoch // interval_sec) * interval_sec
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    def _start_bar(self, bar_start: datetime, tick: Tick) -> None:
        self._bar_start = bar_start
        self._open  = tick.ltp
        self._high  = tick.ltp
        self._low   = tick.ltp
        self._close = tick.ltp
        self._volume = tick.volume

    def _update_ohlcv(self, tick: Tick) -> None:
        if tick.ltp > self._high:
            self._high = tick.ltp
        if tick.ltp < self._low:
            self._low = tick.ltp
        self._close  = tick.ltp
        self._volume = tick.volume  # session cumulative volume

    def _close_bar(self) -> Bar | None:
        if self._open is None:
            return None
        bar = Bar(
            symbol=self.symbol,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            ts=self._bar_start + self.interval,  # bar close = bar_start + interval
        )
        self._reset()
        return bar

    def _reset(self) -> None:
        self._open = None
        self._high = None
        self._low  = None
        self._close = None
        self._volume = 0
        self._bar_start = None


class MultiBarBuilder:
    """
    Manages one ``BarBuilder`` per symbol.  Accepts ticks for any symbol
    and routes them to the correct builder.
    """

    def __init__(self, interval_seconds: int = 60, on_bar: OnBarCallback | None = None) -> None:
        self._interval = interval_seconds
        self._on_bar = on_bar
        self._builders: dict[str, BarBuilder] = {}

    def update(self, tick: Tick) -> Bar | None:
        if tick.symbol not in self._builders:
            self._builders[tick.symbol] = BarBuilder(
                symbol=tick.symbol,
                interval_seconds=self._interval,
                on_bar=self._on_bar,
            )
        return self._builders[tick.symbol].update(tick)

    def force_close_all(self, ts: datetime) -> list[Bar]:
        bars = []
        for builder in self._builders.values():
            bar = builder.force_close(ts)
            if bar:
                bars.append(bar)
        return bars

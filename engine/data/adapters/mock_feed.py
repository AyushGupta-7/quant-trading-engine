"""
engine/data/adapters/mock_feed.py
----------------------------------
Mock market-data adapter with two tick sources:

1. **CSV replay** – reads a pre-recorded tick CSV file and replays rows at
   configurable speed.  Good for deterministic testing against real data.

2. **Synthetic / GBM** – generates prices via Geometric Brownian Motion.
   Useful for unit tests and stress-testing without a tick file.

Both sources implement ``IMarketDataAdapter`` so they are interchangeable.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import AsyncIterator

from engine.data.adapters.base import IMarketDataAdapter

logger = logging.getLogger(__name__)


class CsvFeedAdapter(IMarketDataAdapter):
    """
    Replays ticks from a CSV file.

    Expected CSV columns (header required)::

        timestamp,symbol,last_price,bid,ask,volume,oi

    *timestamp* may be ISO-8601 or POSIX epoch (float).

    Parameters
    ----------
    csv_path:
        Path to the tick CSV file.
    replay_speed:
        1.0 = real-time; 0.0 = as fast as possible; 2.0 = 2× real-time.
    symbols:
        If provided, only rows matching these symbols are emitted.
    """

    def __init__(
        self,
        csv_path: str | Path,
        replay_speed: float = 0.0,
        symbols: list[str] | None = None,
    ) -> None:
        self._path = Path(csv_path)
        self._replay_speed = replay_speed
        self._symbols: set[str] | None = {s.upper() for s in symbols} if symbols else None
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=5000)
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        logger.info("CsvFeedAdapter: connected (file=%s)", self._path)

    async def disconnect(self) -> None:
        self._connected = False
        await self._queue.put(None)  # sentinel
        logger.info("CsvFeedAdapter: disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        if self._symbols is None:
            self._symbols = {s.upper() for s in symbols}
        else:
            self._symbols.update(s.upper() for s in symbols)
        # Start background reader
        asyncio.create_task(self._read_csv(), name="csv-feed-reader")

    async def tick_stream(self) -> AsyncIterator[dict]:  # type: ignore[override]
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    # ------------------------------------------------------------------

    async def _read_csv(self) -> None:
        if not self._path.exists():
            logger.error("CsvFeedAdapter: file not found: %s", self._path)
            await self._queue.put(None)
            return

        prev_ts: datetime | None = None

        with self._path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row["symbol"].upper()
                if self._symbols and symbol not in self._symbols:
                    continue

                tick: dict = {
                    "symbol": symbol,
                    "last_price": float(row["last_price"]),
                    "bid":   float(row.get("bid", row["last_price"])),
                    "ask":   float(row.get("ask", row["last_price"])),
                    "volume": int(float(row.get("volume", 0))),
                    "oi":    int(float(row.get("oi", 0))),
                    "timestamp": row["timestamp"],
                }

                # Timing delay for real-time replay
                if self._replay_speed > 0 and prev_ts is not None:
                    from engine.data.normaliser import TickNormaliser
                    cur_ts = TickNormaliser._parse_timestamp(row["timestamp"])
                    gap = (cur_ts - prev_ts).total_seconds()
                    if gap > 0:
                        await asyncio.sleep(gap / self._replay_speed)
                    prev_ts = cur_ts
                elif prev_ts is None:
                    from engine.data.normaliser import TickNormaliser
                    prev_ts = TickNormaliser._parse_timestamp(row["timestamp"])

                await self._queue.put(tick)

        await self._queue.put(None)  # EOF sentinel
        logger.info("CsvFeedAdapter: finished replaying %s", self._path)


class SyntheticFeedAdapter(IMarketDataAdapter):
    """
    Generates synthetic tick prices via Geometric Brownian Motion.

    Parameters
    ----------
    symbols:
        Symbols to generate.  Each gets its own independent GBM process.
    initial_prices:
        Starting price per symbol.  Defaults to 1000.0.
    drift:
        Annual drift (μ) used in GBM.  Converted to per-tick.
    volatility:
        Annual volatility (σ) used in GBM.  Converted to per-tick.
    tick_interval_seconds:
        How many seconds between synthetic ticks.
    ticks_per_year:
        For converting annualised params to per-tick (default: 252×390).
    seed:
        Random seed for reproducibility.
    max_ticks:
        Stop after this many ticks per symbol (0 = infinite).
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        initial_prices: dict[str, float] | None = None,
        drift: float = 0.0001,
        volatility: float = 0.002,
        tick_interval_seconds: float = 1.0,
        ticks_per_year: int = 252 * 390,
        seed: int = 42,
        max_ticks: int = 0,
    ) -> None:
        self._symbols: list[str] = symbols or []
        self._init_prices = initial_prices or {}
        self._drift = drift
        self._vol = volatility
        self._tick_interval = tick_interval_seconds
        self._ticks_per_year = ticks_per_year
        self._rng = random.Random(seed)
        self._max_ticks = max_ticks
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=10000)
        self._stop_event = asyncio.Event()

    async def connect(self) -> None:
        logger.info("SyntheticFeedAdapter: connected (symbols=%s)", self._symbols)

    async def disconnect(self) -> None:
        self._stop_event.set()
        await self._queue.put(None)

    async def subscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            if s not in self._symbols:
                self._symbols.append(s)
        asyncio.create_task(self._generate(), name="synthetic-feed-generator")

    async def tick_stream(self) -> AsyncIterator[dict]:  # type: ignore[override]
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    # ------------------------------------------------------------------

    async def _generate(self) -> None:
        dt = 1.0 / self._ticks_per_year
        mu_dt  = (self._drift - 0.5 * self._vol ** 2) * dt
        sig_dt = self._vol * math.sqrt(dt)

        prices = {s: self._init_prices.get(s, 1000.0) for s in self._symbols}
        volumes = {s: 0 for s in self._symbols}

        start = datetime.now(tz=timezone.utc)
        tick_count = 0

        while not self._stop_event.is_set():
            ts = start + timedelta(seconds=tick_count * self._tick_interval)
            for sym in self._symbols:
                z = self._rng.gauss(0, 1)
                prices[sym] *= math.exp(mu_dt + sig_dt * z)
                prices[sym] = max(prices[sym], 0.01)  # floor
                volumes[sym] += self._rng.randint(10, 500)
                spread = prices[sym] * 0.0002  # 2 bps spread

                tick = {
                    "symbol": sym,
                    "last_price": round(prices[sym], 2),
                    "bid":  round(prices[sym] - spread / 2, 2),
                    "ask":  round(prices[sym] + spread / 2, 2),
                    "volume": volumes[sym],
                    "oi": 0,
                    "timestamp": ts.isoformat(),
                }
                await self._queue.put(tick)

            tick_count += 1
            if self._max_ticks and tick_count >= self._max_ticks:
                break

            if self._tick_interval > 0:
                await asyncio.sleep(self._tick_interval)
            else:
                await asyncio.sleep(0)  # yield to event loop

        await self._queue.put(None)
        logger.info("SyntheticFeedAdapter: finished generating %d ticks", tick_count)


def build_feed_adapter(cfg: dict) -> IMarketDataAdapter:
    """Factory: build a feed adapter from the ``feed`` config section."""
    adapter_type = cfg.get("adapter", "mock").lower()

    if adapter_type == "mock":
        mock_cfg = cfg.get("mock", {})
        source = mock_cfg.get("source", "synthetic").lower()

        if source == "csv":
            return CsvFeedAdapter(
                csv_path=mock_cfg["csv_path"],
                replay_speed=float(mock_cfg.get("replay_speed", 0.0)),
            )
        elif source == "synthetic":
            syn = mock_cfg.get("synthetic", {})
            return SyntheticFeedAdapter(
                drift=float(syn.get("drift", 0.0001)),
                volatility=float(syn.get("volatility", 0.002)),
                seed=int(syn.get("seed", 42)),
                tick_interval_seconds=0.0,  # as-fast-as-possible in backtests
            )
        else:
            raise ValueError(f"Unknown mock source: {source!r}")

    elif adapter_type == "kite":
        from engine.data.adapters.kite_feed import KiteFeedAdapter
        return KiteFeedAdapter(cfg.get("kite", {}))

    else:
        raise ValueError(f"Unknown feed adapter type: {adapter_type!r}")

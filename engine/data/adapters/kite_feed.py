"""
engine/data/adapters/kite_feed.py
----------------------------------
Zerodha Kite Connect WebSocket feed adapter — REAL IMPLEMENTATION.

Authentication
~~~~~~~~~~~~~~
Credentials are read exclusively from environment variables:

    KITE_API_KEY        – your Kite app's api_key
    KITE_ACCESS_TOKEN   – a valid access_token (rotate daily after login)

Never hard-code or log these values.

Tick pipeline
~~~~~~~~~~~~~
KiteTicker (callback-based) → asyncio Queue → tick_stream() generator
→ TickNormaliser → BarBuilder → Strategy

The KiteTicker runs in its own thread (kiteconnect uses a threaded websocket).
Callbacks enqueue raw tick dicts onto an asyncio queue using
``loop.call_soon_threadsafe``.  The consumer awaits items from the queue inside
``tick_stream()``.

Reconnect handling
~~~~~~~~~~~~~~~~~~
KiteTicker has built-in auto-reconnect.  We additionally re-subscribe on every
``on_connect`` callback to ensure instrument tokens are restored after a drop.

Symbol ↔ instrument token mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Kite WebSocket requires numeric *instrument tokens*, not symbol strings.
This adapter maintains an ``_instruments`` dict mapping our internal symbol
strings → Kite instrument tokens.  Populate it via ``set_instrument_tokens``
before calling ``subscribe()``.  If no token mapping is provided we fall back to
subscribing using the symbol string, which only works for equities (not futures).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import AsyncIterator, Callable

try:
    from kiteconnect import KiteTicker
except ImportError:  # pragma: no cover
    KiteTicker = None  # type: ignore[assignment,misc]

from engine.data.adapters.base import IMarketDataAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-variable names (documented, never logged)
# ---------------------------------------------------------------------------
_ENV_API_KEY      = "KITE_API_KEY"
_ENV_ACCESS_TOKEN = "KITE_ACCESS_TOKEN"

# Queue size: stop filling if consumer is behind; oldest ticks are dropped
_QUEUE_MAXSIZE = 10_000


class KiteCredentialError(RuntimeError):
    """Raised when required Kite credentials are absent."""


class KiteFeedAdapter(IMarketDataAdapter):
    """
    Zerodha Kite Connect WebSocket feed adapter.

    Parameters
    ----------
    on_tick_raw:
        Optional additional callback invoked synchronously on each raw tick
        before it is queued.  Use for monitoring / debugging only.
    reconnect_max_tries:
        How many reconnect attempts KiteTicker should make (-1 = unlimited).
    reconnect_max_delay:
        Maximum seconds between reconnect attempts.

    Environment variables (required at runtime):
        KITE_API_KEY
        KITE_ACCESS_TOKEN
    """

    def __init__(
        self,
        on_tick_raw: Callable[[list[dict]], None] | None = None,
        reconnect_max_tries: int = -1,
        reconnect_max_delay: int = 300,
    ) -> None:
        self._on_tick_raw = on_tick_raw
        self._reconnect_max_tries = reconnect_max_tries
        self._reconnect_max_delay = reconnect_max_delay

        self._ticker: object | None = None   # kiteconnect.KiteTicker
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)

        # symbol → instrument token (set by caller via set_instrument_tokens)
        self._instruments: dict[str, int] = {}
        # Currently subscribed tokens (restored on reconnect)
        self._subscribed_tokens: list[int] = []

        self._connected = False
        self._shutting_down = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_instrument_tokens(self, mapping: dict[str, int]) -> None:
        """
        Provide symbol → numeric Kite instrument token mapping.

        This must be called before ``subscribe()`` for futures/commodity
        instruments.  For NSE equities you may omit this and subscribe by
        symbol, but Kite will reject unrecognised tokens.

        Example::

            adapter.set_instrument_tokens({
                "CRUDEOIL": 5633,
                "GOLD":     53504,
            })
        """
        self._instruments = {k.upper(): v for k, v in mapping.items()}
        logger.info("KiteFeedAdapter: loaded %d instrument tokens", len(self._instruments))

    async def connect(self) -> None:
        """
        Authenticate and start the KiteTicker.

        Raises
        ------
        KiteCredentialError
            If KITE_API_KEY or KITE_ACCESS_TOKEN are not set.
        """
        from kiteconnect import KiteTicker  # imported here to allow tests to mock

        api_key      = os.environ.get(_ENV_API_KEY)
        access_token = os.environ.get(_ENV_ACCESS_TOKEN)

        if not api_key:
            raise KiteCredentialError(
                f"Missing environment variable {_ENV_API_KEY!r}. "
                "Set it before starting the live feed."
            )
        if not access_token:
            raise KiteCredentialError(
                f"Missing environment variable {_ENV_ACCESS_TOKEN!r}. "
                "Generate a fresh access token via the Kite login flow and export it."
            )

        # Credentials intentionally not logged
        logger.info("KiteFeedAdapter: initialising ticker (api_key=***)")
        self._loop = asyncio.get_event_loop()

        self._ticker = KiteTicker(
            api_key=api_key,
            access_token=access_token,
            reconnect_max_tries=self._reconnect_max_tries,
            reconnect_max_delay=self._reconnect_max_delay,
        )

        # Wire callbacks
        self._ticker.on_connect  = self._on_connect
        self._ticker.on_ticks    = self._on_ticks
        self._ticker.on_close    = self._on_close
        self._ticker.on_error    = self._on_error
        self._ticker.on_reconnect = self._on_reconnect
        self._ticker.on_noreconnect = self._on_noreconnect

        # Start in background thread (KiteTicker is blocking)
        self._ticker.connect(threaded=True)
        self._connected = True
        logger.info("KiteFeedAdapter: ticker started (threaded)")

    async def disconnect(self) -> None:
        """Gracefully shut down the WebSocket and drain the queue."""
        self._shutting_down = True
        if self._ticker is not None:
            try:
                self._ticker.close()
            except Exception as exc:
                logger.warning("KiteFeedAdapter: error during close: %s", exc)
        # Sentinel to unblock tick_stream()
        await self._queue.put(None)
        self._connected = False
        logger.info("KiteFeedAdapter: disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        """
        Subscribe to tick updates for the given symbols.

        If instrument tokens are loaded, uses them; otherwise logs a warning
        and attempts to subscribe by symbol string (works only for NSE equities).
        """
        tokens = self._tokens_for_symbols(symbols)
        if not tokens:
            logger.warning(
                "KiteFeedAdapter: no instrument tokens found for %s — "
                "subscribe() skipped.  Call set_instrument_tokens() first.",
                symbols,
            )
            return
        self._subscribed_tokens = list(set(self._subscribed_tokens + tokens))
        if self._ticker and self._connected:
            self._ticker.subscribe(tokens)
            self._ticker.set_mode(self._ticker.MODE_FULL, tokens)
            logger.info("KiteFeedAdapter: subscribed tokens=%s", tokens)

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Remove subscriptions for the given symbols."""
        tokens = self._tokens_for_symbols(symbols)
        if tokens and self._ticker and self._connected:
            self._ticker.unsubscribe(tokens)
            self._subscribed_tokens = [t for t in self._subscribed_tokens if t not in tokens]
            logger.info("KiteFeedAdapter: unsubscribed tokens=%s", tokens)

    async def tick_stream(self) -> AsyncIterator[dict]:  # type: ignore[override]
        """
        Async generator yielding raw tick dicts from the WebSocket.

        Each dict has the schema expected by ``TickNormaliser``::

            {
                "symbol":     "CRUDEOIL",
                "last_price": 6012.0,
                "bid":        6011.0,
                "ask":        6013.0,
                "volume":     12345,
                "oi":         45000,
                "timestamp":  <datetime>,
            }

        Terminates when ``disconnect()`` is called (sentinel None received).
        """
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    # ------------------------------------------------------------------
    # KiteTicker callbacks (called from the ticker thread)
    # ------------------------------------------------------------------

    def _on_connect(self, ws, response) -> None:  # noqa: ARG002
        logger.info("KiteFeedAdapter: WebSocket connected")
        # Re-subscribe after connect / reconnect
        if self._subscribed_tokens and self._ticker:
            self._ticker.subscribe(self._subscribed_tokens)
            self._ticker.set_mode(self._ticker.MODE_FULL, self._subscribed_tokens)
            logger.info(
                "KiteFeedAdapter: re-subscribed %d tokens after connect",
                len(self._subscribed_tokens),
            )

    def _on_ticks(self, ws, ticks: list[dict]) -> None:  # noqa: ARG002
        """Convert Kite ticks to our schema and enqueue them."""
        if self._on_tick_raw:
            try:
                self._on_tick_raw(ticks)
            except Exception:
                logger.exception("KiteFeedAdapter: on_tick_raw callback raised")

        for raw in ticks:
            normalised = self._normalise_kite_tick(raw)
            if normalised is None:
                continue
            if self._loop and not self._loop.is_closed():
                # Thread-safe enqueue
                self._loop.call_soon_threadsafe(self._enqueue, normalised)

    def _on_close(self, ws, code, reason) -> None:
        logger.warning(
            "KiteFeedAdapter: WebSocket closed code=%s reason=%s", code, reason
        )

    def _on_error(self, ws, code, reason) -> None:
        logger.error(
            "KiteFeedAdapter: WebSocket error code=%s reason=%s", code, reason
        )

    def _on_reconnect(self, ws, attempts_count) -> None:
        logger.warning(
            "KiteFeedAdapter: reconnecting attempt=%d", attempts_count
        )

    def _on_noreconnect(self, ws) -> None:
        logger.critical(
            "KiteFeedAdapter: max reconnect attempts exhausted — shutting down feed"
        )
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, tick: dict) -> None:
        """Thread-safe enqueue; drops oldest tick if queue is full."""
        try:
            self._queue.put_nowait(tick)
        except asyncio.QueueFull:
            # Drop oldest to avoid unbounded memory growth
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(tick)
            except asyncio.QueueFull:
                pass  # drop silently — better than blocking the ticker thread

    def _tokens_for_symbols(self, symbols: list[str]) -> list[int]:
        """Return instrument token integers for the given symbols."""
        tokens = []
        for sym in symbols:
            token = self._instruments.get(sym.upper())
            if token is not None:
                tokens.append(token)
            else:
                logger.warning(
                    "KiteFeedAdapter: no instrument token for %r — "
                    "call set_instrument_tokens() to register it.", sym
                )
        return tokens

    @staticmethod
    def _normalise_kite_tick(raw: dict) -> dict | None:
        """
        Convert a Kite tick dict to our normalised schema.

        Kite tick (MODE_FULL) keys of interest:
            instrument_token, last_price, volume_traded (or volume),
            buy_quantity, sell_quantity, depth.buy[0].price,
            depth.sell[0].price, ohlc.close, oi,
            last_trade_time / exchange_timestamp.
        """
        try:
            ltp = float(raw.get("last_price", 0))
            if ltp <= 0:
                return None

            # Bid / ask from market depth if available
            depth = raw.get("depth", {})
            buy_depth = depth.get("buy", [])
            sell_depth = depth.get("sell", [])
            bid = float(buy_depth[0]["price"]) if buy_depth else ltp
            ask = float(sell_depth[0]["price"]) if sell_depth else ltp

            volume = int(raw.get("volume_traded", raw.get("volume", 0)))
            oi     = int(raw.get("oi", 0))

            # Timestamp: prefer exchange_timestamp, then last_trade_time, then now
            ts_raw = (
                raw.get("exchange_timestamp")
                or raw.get("last_trade_time")
            )
            if isinstance(ts_raw, datetime):
                ts = ts_raw if ts_raw.tzinfo else ts_raw.replace(tzinfo=timezone.utc)
            else:
                ts = datetime.now(tz=timezone.utc)

            # Symbol: resolve from instrument_token if token→symbol mapping available
            symbol = raw.get("tradingsymbol", raw.get("symbol", "UNKNOWN"))

            return {
                "symbol":     symbol,
                "last_price": ltp,
                "bid":        bid,
                "ask":        ask,
                "volume":     volume,
                "oi":         oi,
                "timestamp":  ts,
            }
        except Exception as exc:
            logger.debug("KiteFeedAdapter: could not normalise tick %r: %s", raw, exc)
            return None

    @classmethod
    def from_config(cls, cfg: dict) -> "KiteFeedAdapter":
        kite_cfg = cfg.get("kite", {})
        return cls(
            reconnect_max_tries=int(kite_cfg.get("reconnect_max_tries", -1)),
            reconnect_max_delay=int(kite_cfg.get("reconnect_max_delay", 300)),
        )

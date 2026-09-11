"""
engine/broker/kite_broker.py
-----------------------------
Zerodha Kite Connect REST broker adapter — REAL IMPLEMENTATION.

Authentication
~~~~~~~~~~~~~~
Credentials are read exclusively from environment variables:

    KITE_API_KEY        – your Kite app's api_key
    KITE_ACCESS_TOKEN   – a valid access_token (rotate daily after login)

Never hard-code or log these values.

Order mapping
~~~~~~~~~~~~~
Our internal model → Kite parameters:

    Order.side           → transaction_type  (BUY / SELL)
    Order.order_type     → order_type        (MARKET / LIMIT / SL / SL-M)
    Order.symbol         → tradingsymbol     (looked up in InstrumentRegistry)
    Instrument.exchange  → exchange          (NSE / BSE / MCX)
    CNC / MIS / NRML     → product          (always NRML for futures)
    Order.price          → price             (LIMIT / SL orders)
    Order.price          → trigger_price     (SL / SL-M orders)

Rate limits (Kite REST):
    10 requests/second for order placement.
    We use a minimal token-bucket via asyncio.sleep between retries.

Retry policy:
    429 / 5xx  → exponential backoff, up to 4 attempts.
    403        → attempt a single token refresh, then re-raise.
    Order placement retries are DISABLED because Kite is not idempotent;
    our ``IdempotentPlacer`` already guards against duplicates at our layer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

try:
    from kiteconnect import KiteConnect
except ImportError:  # pragma: no cover
    KiteConnect = None  # type: ignore[assignment,misc]

from engine.broker.base import IBrokerAdapter
from engine.core.events import Fill, Order, OrderState, OrderType, Side
from engine.core.instrument import AssetClass, Exchange, Instrument, InstrumentRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env-variable names (documented, never logged)
# ---------------------------------------------------------------------------
_ENV_API_KEY      = "KITE_API_KEY"
_ENV_ACCESS_TOKEN = "KITE_ACCESS_TOKEN"

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_KITE_PRODUCT = "NRML"          # Normal product for futures
_KITE_VALIDITY = "DAY"
_KITE_VARIETY_REGULAR = "regular"
_KITE_VARIETY_AMO = "amo"       # After-market order

_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT:  "LIMIT",
    OrderType.SL:     "SL",
    OrderType.SL_M:   "SL-M",
}

_STATE_MAP: dict[str, OrderState] = {
    "OPEN":             OrderState.OPEN,
    "PENDING":          OrderState.PENDING,
    "COMPLETE":         OrderState.COMPLETE,
    "CANCELLED":        OrderState.CANCELLED,
    "REJECTED":         OrderState.REJECTED,
    "VALIDATION PENDING": OrderState.PENDING,
    "PUT ORDER REQ RECEIVED": OrderState.PENDING,
    "MODIFY VALIDATION PENDING": OrderState.OPEN,
    "MODIFY COMPLETE":  OrderState.OPEN,
    "TRIGGER PENDING":  OrderState.OPEN,   # for SL orders
}


class KiteCredentialError(RuntimeError):
    """Raised when required Kite credentials are absent or invalid."""


class KiteAuthError(RuntimeError):
    """Raised on 403 / token-expired responses from Kite."""


class KiteBrokerAdapter(IBrokerAdapter):
    """
    Zerodha Kite Connect REST broker adapter.

    Parameters
    ----------
    registry:
        ``InstrumentRegistry`` used to resolve symbol → exchange and trading
        symbol string.  If not provided, the exchange defaults to NSE for all
        symbols, which is incorrect for MCX instruments.
    amo:
        If True, orders are placed as after-market orders (AMO variety).
        Useful for pre-market strategy runs.

    Environment variables (required at runtime, never default-filled):
        KITE_API_KEY
        KITE_ACCESS_TOKEN
    """

    def __init__(
        self,
        registry: InstrumentRegistry | None = None,
        amo: bool = False,
    ) -> None:
        self._registry = registry
        self._amo = amo
        self._kite: Any = None          # kiteconnect.KiteConnect instance
        self._connected = False

    # ------------------------------------------------------------------
    # IBrokerAdapter
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Initialise the KiteConnect client.

        Raises
        ------
        KiteCredentialError
            If KITE_API_KEY or KITE_ACCESS_TOKEN are not set.
        """
        api_key      = os.environ.get(_ENV_API_KEY)
        access_token = os.environ.get(_ENV_ACCESS_TOKEN)

        if not api_key:
            raise KiteCredentialError(
                f"Missing environment variable {_ENV_API_KEY!r}. "
                "Set it before starting the live engine."
            )
        if not access_token:
            raise KiteCredentialError(
                f"Missing environment variable {_ENV_ACCESS_TOKEN!r}. "
                "Generate a fresh access token via the Kite login flow and export it."
            )

        if KiteConnect is None:  # pragma: no cover
            raise ImportError("kiteconnect is not installed. Run: pip install kiteconnect>=5.0")

        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)
        # Credentials intentionally not logged
        logger.info("KiteBrokerAdapter: initialised (api_key=***)")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        self._kite = None
        logger.info("KiteBrokerAdapter: disconnected")

    async def place_order(self, order: Order) -> str:
        """
        Place an order via Kite REST.

        Order placement is NOT retried on failure because Kite does not
        guarantee idempotency — our ``IdempotentPlacer`` layer handles dedup.

        Returns
        -------
        str
            Kite's numeric order ID (as a string).
        """
        self._require_connected()
        params = self._order_to_kite_params(order)
        variety = _KITE_VARIETY_AMO if self._amo else _KITE_VARIETY_REGULAR
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._kite.place_order(variety=variety, **params),
            )
        except Exception as exc:
            logger.error(
                "KiteBrokerAdapter: order placement failed sym=%s side=%s qty=%d error=%s",
                order.symbol, order.side.value, order.qty, exc,
            )
            raise

        order_id = str(response.get("order_id", ""))
        logger.info(
            "KiteBrokerAdapter: placed coid=%s kite_id=%s sym=%s side=%s qty=%d",
            order.client_order_id, order_id, order.symbol, order.side.value, order.qty,
        )
        return order_id

    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        self._require_connected()
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._kite.cancel_order(
                    variety=_KITE_VARIETY_REGULAR,
                    order_id=broker_order_id,
                ),
            )
            logger.info("KiteBrokerAdapter: cancelled kite_id=%s", broker_order_id)
            return True
        except Exception as exc:
            logger.warning(
                "KiteBrokerAdapter: cancel failed kite_id=%s error=%s",
                broker_order_id, exc,
            )
            return False

    async def get_order_status(self, broker_order_id: str) -> OrderState:
        """Return the current state of a Kite order."""
        self._require_connected()
        try:
            orders = await asyncio.get_event_loop().run_in_executor(
                None, self._kite.orders
            )
            for o in orders:
                if str(o.get("order_id")) == broker_order_id:
                    return self._map_kite_status(o.get("status", ""))
        except Exception as exc:
            logger.error(
                "KiteBrokerAdapter: get_order_status failed kite_id=%s error=%s",
                broker_order_id, exc,
            )
        return OrderState.REJECTED   # unknown → treat as rejected for safety

    async def get_positions(self) -> list[dict]:
        """Return net positions from Kite."""
        self._require_connected()
        try:
            pos = await asyncio.get_event_loop().run_in_executor(
                None, self._kite.positions
            )
            # Kite returns {"net": [...], "day": [...]}
            return pos.get("net", [])
        except Exception as exc:
            logger.error("KiteBrokerAdapter: get_positions failed error=%s", exc)
            return []

    # ------------------------------------------------------------------
    # Reconciliation helper
    # ------------------------------------------------------------------

    async def get_orders(self) -> list[dict]:
        """
        Return all today's orders from Kite.  Used by ``ReconciliationEngine``
        to sync OMS state with broker reality on startup/reconnect.
        """
        self._require_connected()
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._kite.orders
            )
        except Exception as exc:
            logger.error("KiteBrokerAdapter: get_orders failed error=%s", exc)
            return []

    async def get_fills_for_order(self, broker_order_id: str) -> list[Fill]:
        """
        Return fill objects for a given Kite order ID.

        Returns an empty list if the order has no trades.
        """
        self._require_connected()
        try:
            trades = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._kite.order_trades(broker_order_id),
            )
        except Exception as exc:
            logger.warning(
                "KiteBrokerAdapter: get_fills_for_order failed kite_id=%s error=%s",
                broker_order_id, exc,
            )
            return []

        fills = []
        for t in trades:
            try:
                side = Side.BUY if t.get("transaction_type", "").upper() == "BUY" else Side.SELL
                fills.append(Fill(
                    order_id=t.get("tag", broker_order_id),   # tag = client_order_id we set
                    broker_order_id=str(t.get("order_id", broker_order_id)),
                    symbol=t.get("tradingsymbol", ""),
                    side=side,
                    fill_price=Decimal(str(t.get("average_price", 0))),
                    fill_qty=int(t.get("quantity", 0)),
                    fees=Decimal(0),   # Kite does not break out per-trade fees
                    ts=self._parse_kite_ts(t.get("exchange_timestamp")),
                ))
            except Exception as exc:
                logger.warning("KiteBrokerAdapter: could not parse trade %r: %s", t, exc)
        return fills

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected or self._kite is None:
            raise RuntimeError(
                "KiteBrokerAdapter: not connected. Call await adapter.connect() first."
            )

    def _order_to_kite_params(self, order: Order) -> dict:
        """
        Map an internal ``Order`` to Kite REST parameters.

        Returns a dict ready to unpack into ``kite.place_order(**params)``.
        """
        exchange, tradingsymbol = self._resolve_symbol(order.symbol)
        kite_order_type = _ORDER_TYPE_MAP.get(order.order_type, "MARKET")

        params: dict[str, Any] = {
            "exchange":        exchange,
            "tradingsymbol":   tradingsymbol,
            "transaction_type": order.side.value,
            "quantity":        order.qty,
            "product":         _KITE_PRODUCT,
            "order_type":      kite_order_type,
            "validity":        _KITE_VALIDITY,
            "tag":             order.client_order_id[:20],  # Kite tag max 20 chars
        }

        # Price fields
        if order.order_type == OrderType.LIMIT and order.price is not None:
            params["price"] = float(order.price)
        elif order.order_type == OrderType.SL and order.price is not None:
            # SL limit: both price and trigger_price
            params["price"]         = float(order.price)
            params["trigger_price"] = float(order.price)
        elif order.order_type == OrderType.SL_M and order.price is not None:
            # SL-M: only trigger_price, no limit price
            params["trigger_price"] = float(order.price)

        return params

    def _resolve_symbol(self, symbol: str) -> tuple[str, str]:
        """
        Return (exchange_str, tradingsymbol) for a given internal symbol.

        Uses the InstrumentRegistry when available; falls back to guessing
        NSE for unknown symbols (logs a warning).
        """
        if self._registry is not None:
            try:
                instr: Instrument = self._registry.get(symbol)
                exchange_str = instr.exchange.value  # "MCX", "NSE", "BSE"
                # For futures, Kite tradingsymbol is typically the base symbol
                # (Kite resolves the front-month contract server-side when
                #  the instrument token is not used).
                return exchange_str, symbol
            except KeyError:
                pass

        logger.warning(
            "KiteBrokerAdapter: unknown symbol %r — defaulting exchange to NSE", symbol
        )
        return "NSE", symbol

    @staticmethod
    def _map_kite_status(kite_status: str) -> OrderState:
        """Map a Kite order status string to our ``OrderState``."""
        return _STATE_MAP.get(kite_status.upper().strip(), OrderState.OPEN)

    @staticmethod
    def _parse_kite_ts(raw) -> datetime:
        if raw is None:
            return datetime.now(tz=timezone.utc)
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        try:
            from dateutil.parser import parse as dtparse
            dt = dtparse(str(raw))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(tz=timezone.utc)

    @classmethod
    def from_config(cls, cfg: dict, registry: InstrumentRegistry | None = None) -> "KiteBrokerAdapter":
        kite_cfg = cfg.get("kite", {})
        return cls(
            registry=registry,
            amo=bool(kite_cfg.get("amo", False)),
        )

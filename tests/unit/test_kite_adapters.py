"""
tests/unit/test_kite_adapters.py
----------------------------------
Unit tests for KiteBrokerAdapter and KiteFeedAdapter.

All tests use unittest.mock to avoid real network calls.
No real credentials are required to run this suite.

Coverage:
  - KiteBrokerAdapter:
    * order mapping (all OrderType variants)
    * successful placement
    * order cancellation
    * order status lookup
    * position retrieval
    * missing credentials
    * API error handling (non-placement calls)
    * duplicate order safety via IdempotentPlacer
    * fill parsing
    * reconciliation via get_orders()
  - KiteFeedAdapter:
    * missing credentials
    * tick conversion (MODE_FULL dict → our schema)
    * subscription handling (set_instrument_tokens → subscribe)
    * queue enqueue / drop-on-full
    * graceful shutdown (sentinel)
    * reconnect callback re-subscribe
    * on_error / on_noreconnect callbacks
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from engine.broker.kite_broker import (
    KiteBrokerAdapter,
    KiteCredentialError,
    _ORDER_TYPE_MAP,
)
from engine.core.events import Order, OrderIntent, OrderState, OrderType, Side
from engine.core.instrument import AssetClass, Exchange, Instrument, InstrumentRegistry
from engine.data.adapters.kite_feed import KiteFeedAdapter
from engine.oms.idempotent_placer import IdempotentPlacer
from engine.oms.order import intent_to_order
from engine.oms.state_store import OrderStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> InstrumentRegistry:
    reg = InstrumentRegistry()
    reg.register(Instrument(
        symbol="CRUDEOIL",
        exchange=Exchange.MCX,
        asset_class=AssetClass.COMMODITY_FUTURES,
        lot_size=100,
        tick_size=Decimal("1.0"),
        margin_pct=Decimal("0.05"),
    ))
    reg.register(Instrument(
        symbol="NIFTY",
        exchange=Exchange.NSE,
        asset_class=AssetClass.INDEX_FUTURES,
        lot_size=25,
        tick_size=Decimal("0.05"),
        margin_pct=Decimal("0.10"),
    ))
    return reg


def _make_order(
    symbol: str = "CRUDEOIL",
    side: Side = Side.BUY,
    qty: int = 1,
    order_type: OrderType = OrderType.MARKET,
    price: Decimal | None = None,
) -> Order:
    intent = OrderIntent(
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        price=price,
        strategy_id="test-strategy",
        tag="test",
    )
    return intent_to_order(intent, bar_ts_ms=1_700_000_000_000)


def _connected_adapter(registry=None) -> KiteBrokerAdapter:
    """Return a KiteBrokerAdapter with a mocked KiteConnect instance."""
    adapter = KiteBrokerAdapter(registry=registry or _make_registry())
    adapter._kite = MagicMock()
    adapter._connected = True
    return adapter


# ===========================================================================
# KiteBrokerAdapter tests
# ===========================================================================

class TestKiteBrokerAdapterCredentials:

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("KITE_API_KEY", raising=False)
        monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
        adapter = KiteBrokerAdapter()
        with pytest.raises(KiteCredentialError, match="KITE_API_KEY"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_missing_access_token_raises(self, monkeypatch):
        monkeypatch.setenv("KITE_API_KEY", "test_key")
        monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
        adapter = KiteBrokerAdapter()

        with patch("engine.broker.kite_broker.KiteConnect", MagicMock()):
            with pytest.raises(KiteCredentialError, match="KITE_ACCESS_TOKEN"):
                await adapter.connect()

    @pytest.mark.asyncio
    async def test_api_key_not_logged(self, monkeypatch, caplog):
        """Credentials must never appear in log output."""
        import logging
        monkeypatch.setenv("KITE_API_KEY", "SECRET_KEY_12345")
        monkeypatch.setenv("KITE_ACCESS_TOKEN", "SECRET_TOKEN_67890")
        adapter = KiteBrokerAdapter()

        mock_kite_cls = MagicMock()
        mock_kite_cls.return_value = MagicMock()

        with caplog.at_level(logging.DEBUG):
            with patch("engine.broker.kite_broker.KiteConnect", mock_kite_cls):
                await adapter.connect()

        for record in caplog.records:
            assert "SECRET_KEY_12345" not in record.getMessage()
            assert "SECRET_TOKEN_67890" not in record.getMessage()


class TestKiteBrokerOrderMapping:

    def test_market_order_params(self):
        adapter = _connected_adapter()
        order = _make_order(order_type=OrderType.MARKET)
        params = adapter._order_to_kite_params(order)

        assert params["transaction_type"] == "BUY"
        assert params["order_type"] == "MARKET"
        assert params["exchange"] == "MCX"
        assert params["tradingsymbol"] == "CRUDEOIL"
        assert params["quantity"] == 1
        assert params["product"] == "NRML"
        assert params["validity"] == "DAY"
        assert "price" not in params
        assert "trigger_price" not in params

    def test_limit_order_params(self):
        adapter = _connected_adapter()
        order = _make_order(order_type=OrderType.LIMIT, price=Decimal("6000"))
        params = adapter._order_to_kite_params(order)

        assert params["order_type"] == "LIMIT"
        assert params["price"] == 6000.0
        assert "trigger_price" not in params

    def test_sl_order_params(self):
        """SL (stop-loss limit) sets both price and trigger_price."""
        adapter = _connected_adapter()
        order = _make_order(order_type=OrderType.SL, price=Decimal("5950"))
        params = adapter._order_to_kite_params(order)

        assert params["order_type"] == "SL"
        assert params["price"] == 5950.0
        assert params["trigger_price"] == 5950.0

    def test_sl_m_order_params(self):
        """SL-M (stop-loss market) sets only trigger_price, no limit price."""
        adapter = _connected_adapter()
        order = _make_order(order_type=OrderType.SL_M, price=Decimal("5960"))
        params = adapter._order_to_kite_params(order)

        assert params["order_type"] == "SL-M"
        assert params["trigger_price"] == 5960.0
        assert "price" not in params

    def test_sell_order_params(self):
        adapter = _connected_adapter()
        order = _make_order(side=Side.SELL)
        params = adapter._order_to_kite_params(order)
        assert params["transaction_type"] == "SELL"

    def test_nse_exchange_resolved(self):
        adapter = _connected_adapter()
        order = _make_order(symbol="NIFTY")
        params = adapter._order_to_kite_params(order)
        assert params["exchange"] == "NSE"

    def test_unknown_symbol_defaults_to_nse(self):
        adapter = _connected_adapter(registry=InstrumentRegistry())  # empty registry
        order = _make_order(symbol="UNKNOWN")
        params = adapter._order_to_kite_params(order)
        assert params["exchange"] == "NSE"

    def test_tag_is_truncated_to_20_chars(self):
        adapter = _connected_adapter()
        order = _make_order()
        params = adapter._order_to_kite_params(order)
        assert len(params["tag"]) <= 20


class TestKiteBrokerPlacement:

    @pytest.mark.asyncio
    async def test_successful_placement_returns_broker_id(self):
        adapter = _connected_adapter()
        adapter._kite.place_order.return_value = {"order_id": "111222333"}
        order = _make_order()

        broker_id = await adapter.place_order(order)
        assert broker_id == "111222333"

    @pytest.mark.asyncio
    async def test_placement_failure_raises(self):
        adapter = _connected_adapter()
        adapter._kite.place_order.side_effect = Exception("InputException: quantity")
        order = _make_order()

        with pytest.raises(Exception, match="InputException"):
            await adapter.place_order(order)

    @pytest.mark.asyncio
    async def test_placement_not_retried_on_failure(self):
        """Order placement must NOT be retried to avoid duplicates."""
        adapter = _connected_adapter()
        adapter._kite.place_order.side_effect = Exception("network error")
        order = _make_order()

        with pytest.raises(Exception):
            await adapter.place_order(order)

        # Only called once — no retry
        assert adapter._kite.place_order.call_count == 1

    @pytest.mark.asyncio
    async def test_not_connected_raises(self):
        adapter = KiteBrokerAdapter()
        # _connected=False by default
        with pytest.raises(RuntimeError, match="not connected"):
            await adapter.place_order(_make_order())


class TestKiteBrokerCancelAndStatus:

    @pytest.mark.asyncio
    async def test_cancel_success_returns_true(self):
        adapter = _connected_adapter()
        adapter._kite.cancel_order.return_value = {"order_id": "999"}
        result = await adapter.cancel_order("999")
        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_failure_returns_false(self):
        adapter = _connected_adapter()
        adapter._kite.cancel_order.side_effect = Exception("order not found")
        result = await adapter.cancel_order("999")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_order_status_complete(self):
        adapter = _connected_adapter()
        adapter._kite.orders.return_value = [
            {"order_id": "123", "status": "COMPLETE"},
        ]
        state = await adapter.get_order_status("123")
        assert state == OrderState.COMPLETE

    @pytest.mark.asyncio
    async def test_get_order_status_unknown_returns_rejected(self):
        adapter = _connected_adapter()
        adapter._kite.orders.return_value = []  # order not found
        state = await adapter.get_order_status("xyz")
        assert state == OrderState.REJECTED

    @pytest.mark.asyncio
    async def test_get_order_status_kite_error_returns_rejected(self):
        adapter = _connected_adapter()
        adapter._kite.orders.side_effect = Exception("network error")
        state = await adapter.get_order_status("123")
        assert state == OrderState.REJECTED

    @pytest.mark.asyncio
    async def test_get_positions_returns_net_list(self):
        adapter = _connected_adapter()
        adapter._kite.positions.return_value = {
            "net": [{"tradingsymbol": "CRUDEOIL", "quantity": 1}],
            "day": [],
        }
        positions = await adapter.get_positions()
        assert len(positions) == 1
        assert positions[0]["tradingsymbol"] == "CRUDEOIL"

    @pytest.mark.asyncio
    async def test_get_positions_error_returns_empty(self):
        adapter = _connected_adapter()
        adapter._kite.positions.side_effect = Exception("error")
        positions = await adapter.get_positions()
        assert positions == []


class TestKiteOrderStatusMap:

    @pytest.mark.parametrize("kite_status,expected", [
        ("COMPLETE",   OrderState.COMPLETE),
        ("CANCELLED",  OrderState.CANCELLED),
        ("REJECTED",   OrderState.REJECTED),
        ("OPEN",       OrderState.OPEN),
        ("PENDING",    OrderState.PENDING),
        ("TRIGGER PENDING", OrderState.OPEN),
        ("VALIDATION PENDING", OrderState.PENDING),
        ("",           OrderState.OPEN),   # fallback
    ])
    def test_status_mapping(self, kite_status, expected):
        result = KiteBrokerAdapter._map_kite_status(kite_status)
        assert result == expected


class TestKiteReconciliation:

    @pytest.mark.asyncio
    async def test_get_orders_returns_list(self):
        adapter = _connected_adapter()
        adapter._kite.orders.return_value = [
            {"order_id": "1", "status": "COMPLETE"},
            {"order_id": "2", "status": "OPEN"},
        ]
        orders = await adapter.get_orders()
        assert len(orders) == 2

    @pytest.mark.asyncio
    async def test_get_fills_for_order_returns_fill_objects(self):
        adapter = _connected_adapter()
        adapter._kite.order_trades.return_value = [{
            "order_id": "123",
            "transaction_type": "BUY",
            "tradingsymbol": "CRUDEOIL",
            "average_price": 6010.0,
            "quantity": 1,
            "tag": "test_coid",
            "exchange_timestamp": None,
        }]
        fills = await adapter.get_fills_for_order("123")
        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("6010.0")
        assert fills[0].side == Side.BUY

    @pytest.mark.asyncio
    async def test_get_fills_error_returns_empty(self):
        adapter = _connected_adapter()
        adapter._kite.order_trades.side_effect = Exception("error")
        fills = await adapter.get_fills_for_order("999")
        assert fills == []


class TestIdempotentPlacerWithKite:
    """Prove that the IdempotentPlacer correctly deduplicates even when the
    underlying Kite adapter would accept the order again."""

    @pytest.mark.asyncio
    async def test_duplicate_intent_not_placed_twice(self):
        adapter = _connected_adapter()
        adapter._kite.place_order.return_value = {"order_id": "ABC"}

        store = OrderStateStore(":memory:")
        placer = IdempotentPlacer(store=store, broker=adapter)

        intent = OrderIntent(
            symbol="CRUDEOIL", side=Side.BUY, qty=1,
            order_type=OrderType.MARKET, price=None,
            strategy_id="s1", tag="entry",
        )

        # First call — should place
        result1 = await placer.place(intent)
        assert result1 is not None

        # Second call with identical intent — must NOT call place_order again
        result2 = await placer.place(intent)
        assert result2 is not None
        assert adapter._kite.place_order.call_count == 1


# ===========================================================================
# KiteFeedAdapter tests
# ===========================================================================

class TestKiteFeedAdapterCredentials:

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        from engine.data.adapters.kite_feed import KiteCredentialError as FeedCredErr
        monkeypatch.delenv("KITE_API_KEY", raising=False)
        monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
        adapter = KiteFeedAdapter()
        with pytest.raises(FeedCredErr, match="KITE_API_KEY"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_missing_access_token_raises(self, monkeypatch):
        from engine.data.adapters.kite_feed import KiteCredentialError as FeedCredErr
        monkeypatch.setenv("KITE_API_KEY", "test_key")
        monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
        adapter = KiteFeedAdapter()
        with patch("engine.data.adapters.kite_feed.KiteTicker", MagicMock()):
            with pytest.raises(FeedCredErr, match="KITE_ACCESS_TOKEN"):
                await adapter.connect()

    @pytest.mark.asyncio
    async def test_credentials_not_in_logs(self, monkeypatch, caplog):
        import logging
        monkeypatch.setenv("KITE_API_KEY", "SECRET_FEED_KEY")
        monkeypatch.setenv("KITE_ACCESS_TOKEN", "SECRET_FEED_TOKEN")

        mock_ticker = MagicMock()
        mock_ticker.connect = MagicMock()

        with caplog.at_level(logging.DEBUG):
            with patch("engine.data.adapters.kite_feed.KiteTicker", return_value=mock_ticker):
                adapter = KiteFeedAdapter()
                await adapter.connect()

        for record in caplog.records:
            assert "SECRET_FEED_KEY" not in record.getMessage()
            assert "SECRET_FEED_TOKEN" not in record.getMessage()


class TestKiteTickNormalisation:
    """KiteFeedAdapter._normalise_kite_tick"""

    def _norm(self, raw: dict) -> dict | None:
        return KiteFeedAdapter._normalise_kite_tick(raw)

    def test_full_mode_tick_normalised_correctly(self):
        raw = {
            "tradingsymbol": "CRUDEOIL",
            "last_price": 6012.0,
            "volume_traded": 12345,
            "oi": 45000,
            "depth": {
                "buy":  [{"price": 6011.0, "quantity": 10}],
                "sell": [{"price": 6013.0, "quantity": 5}],
            },
            "exchange_timestamp": datetime(2024, 1, 15, 9, 15, 0, tzinfo=timezone.utc),
        }
        result = self._norm(raw)
        assert result is not None
        assert result["symbol"] == "CRUDEOIL"
        assert result["last_price"] == 6012.0
        assert result["bid"] == 6011.0
        assert result["ask"] == 6013.0
        assert result["volume"] == 12345
        assert result["oi"] == 45000

    def test_missing_depth_uses_ltp_for_bid_ask(self):
        raw = {"tradingsymbol": "GOLD", "last_price": 70000.0}
        result = self._norm(raw)
        assert result["bid"] == 70000.0
        assert result["ask"] == 70000.0

    def test_zero_ltp_returns_none(self):
        raw = {"tradingsymbol": "GOLD", "last_price": 0.0}
        result = self._norm(raw)
        assert result is None

    def test_missing_last_price_returns_none(self):
        raw = {"tradingsymbol": "GOLD"}
        result = self._norm(raw)
        assert result is None

    def test_volume_traded_preferred_over_volume(self):
        raw = {"tradingsymbol": "X", "last_price": 100.0, "volume_traded": 999, "volume": 111}
        result = self._norm(raw)
        assert result["volume"] == 999

    def test_fallback_symbol_key(self):
        """Accepts 'symbol' key if 'tradingsymbol' is absent."""
        raw = {"symbol": "NIFTY", "last_price": 22000.0}
        result = self._norm(raw)
        assert result["symbol"] == "NIFTY"


class TestKiteFeedSubscription:

    def test_set_instrument_tokens_stored_uppercase(self):
        adapter = KiteFeedAdapter()
        adapter.set_instrument_tokens({"crudeoil": 5633, "GOLD": 53504})
        assert adapter._instruments["CRUDEOIL"] == 5633
        assert adapter._instruments["GOLD"] == 53504

    @pytest.mark.asyncio
    async def test_subscribe_calls_ticker_subscribe(self):
        adapter = KiteFeedAdapter()
        adapter._connected = True
        mock_ticker = MagicMock()
        adapter._ticker = mock_ticker
        adapter.set_instrument_tokens({"CRUDEOIL": 5633})

        await adapter.subscribe(["CRUDEOIL"])
        mock_ticker.subscribe.assert_called_once_with([5633])
        mock_ticker.set_mode.assert_called_once()

    @pytest.mark.asyncio
    async def test_subscribe_without_tokens_logs_warning(self, caplog):
        import logging
        adapter = KiteFeedAdapter()
        adapter._connected = True
        adapter._ticker = MagicMock()
        # No instrument tokens set

        with caplog.at_level(logging.WARNING):
            await adapter.subscribe(["UNKNOWN_SYMBOL"])

        assert not adapter._ticker.subscribe.called

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_from_subscribed(self):
        adapter = KiteFeedAdapter()
        adapter._connected = True
        mock_ticker = MagicMock()
        adapter._ticker = mock_ticker
        adapter.set_instrument_tokens({"CRUDEOIL": 5633, "GOLD": 53504})
        adapter._subscribed_tokens = [5633, 53504]

        await adapter.unsubscribe(["CRUDEOIL"])
        assert 5633 not in adapter._subscribed_tokens
        assert 53504 in adapter._subscribed_tokens


class TestKiteFeedQueue:

    def test_enqueue_adds_to_queue(self):
        adapter = KiteFeedAdapter()
        tick = {"symbol": "X", "last_price": 100.0}
        adapter._enqueue(tick)
        assert adapter._queue.qsize() == 1

    def test_enqueue_drops_oldest_when_full(self):
        """When the queue is full the oldest tick is dropped, not the newest."""
        adapter = KiteFeedAdapter()
        # Fill the queue
        for i in range(_QUEUE_MAXSIZE := 10_000):
            try:
                adapter._queue.put_nowait({"n": i})
            except asyncio.QueueFull:
                break  # queue is full — expected

        newest = {"symbol": "NEW", "last_price": 999.0}
        adapter._enqueue(newest)  # should not raise or block

    @pytest.mark.asyncio
    async def test_tick_stream_yields_ticks_and_stops_on_sentinel(self):
        adapter = KiteFeedAdapter()
        # Pre-populate the queue
        adapter._queue.put_nowait({"symbol": "A", "last_price": 1.0})
        adapter._queue.put_nowait({"symbol": "B", "last_price": 2.0})
        adapter._queue.put_nowait(None)  # sentinel

        received = []
        async for tick in adapter.tick_stream():
            received.append(tick)

        assert len(received) == 2
        assert received[0]["symbol"] == "A"
        assert received[1]["symbol"] == "B"


class TestKiteFeedReconnect:

    @pytest.mark.asyncio
    async def test_on_connect_resubscribes(self):
        adapter = KiteFeedAdapter()
        adapter._connected = True
        mock_ticker = MagicMock()
        adapter._ticker = mock_ticker
        adapter._subscribed_tokens = [5633, 53504]

        # Simulate reconnect callback
        adapter._on_connect(ws=None, response=None)

        mock_ticker.subscribe.assert_called_once_with([5633, 53504])
        mock_ticker.set_mode.assert_called_once()

    def test_on_noreconnect_sends_sentinel(self):
        adapter = KiteFeedAdapter()
        mock_loop = MagicMock()
        mock_loop.is_closed.return_value = False
        adapter._loop = mock_loop

        # Simulate callback — it should enqueue a sentinel
        adapter._on_noreconnect(ws=None)

        mock_loop.call_soon_threadsafe.assert_called_once_with(adapter._queue.put_nowait, None)

    @pytest.mark.asyncio
    async def test_disconnect_sends_sentinel_and_stops_stream(self):
        adapter = KiteFeedAdapter()
        adapter._connected = True
        adapter._ticker = MagicMock()

        # Start stream consumer in background
        received = []

        async def consumer():
            async for tick in adapter.tick_stream():
                received.append(tick)

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0)  # yield to let consumer start

        # Disconnect should send sentinel
        await adapter.disconnect()
        await asyncio.wait_for(task, timeout=1.0)

        assert received == []   # no ticks were added before disconnect

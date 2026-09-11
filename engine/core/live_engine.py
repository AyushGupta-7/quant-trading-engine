"""
engine/core/live_engine.py
---------------------------
Asynchronous live execution engine.

Coordinates real-time tick streaming, bar building, indicator updates,
strategy evaluation, and idempotent order placement.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from engine.core.events import Bar
from engine.data.bar_builder import MultiBarBuilder
from engine.data.normaliser import TickNormaliser
from engine.oms.idempotent_placer import IdempotentPlacer
from engine.risk.risk_gate import RiskGate

logger = logging.getLogger(__name__)


class LiveEngine:
    """
    Coordinates asynchronous market data processing and strategy execution.
    """

    def __init__(
        self,
        feed: Any,
        broker: Any,
        placer: IdempotentPlacer,
        strategies: list[Any],
        normaliser: TickNormaliser,
        bar_builder: MultiBarBuilder,
        risk_gate: RiskGate,
        indicators_state: dict[str, dict[str, Any]],
        pos_book: Any = None,
        pnl_engine: Any = None,
        blotter: Any = None,
        poll_interval_seconds: float = 3.0,
    ) -> None:
        self._feed = feed
        self._broker = broker
        self._placer = placer
        self._strategies = strategies
        self._normaliser = normaliser
        self._bar_builder = bar_builder
        self._risk_gate = risk_gate
        self._indicators_state = indicators_state
        self._pos_book = pos_book
        self._pnl_engine = pnl_engine
        self._blotter = blotter
        self._poll_interval = poll_interval_seconds
        
        self._running = False
        self._process_task: asyncio.Task | None = None
        self._poll_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the engine and begin processing ticks."""
        if self._running:
            return
        
        self._running = True
        logger.info("LiveEngine: starting")
        
        # Connect adapters if not already connected
        if hasattr(self._feed, "connect"):
            await self._feed.connect()
        if hasattr(self._broker, "connect"):
            await self._broker.connect()
            
        self._process_task = asyncio.create_task(self._process_ticks())
        if self._poll_interval > 0:
            self._poll_task = asyncio.create_task(self._poll_orders())

    async def stop(self) -> None:
        """Gracefully stop the engine and adapters."""
        if not self._running:
            return
            
        logger.info("LiveEngine: initiating graceful shutdown")
        self._running = False
        
        if hasattr(self._feed, "disconnect"):
            await self._feed.disconnect()

        if self._process_task:
            try:
                # Wait briefly for the task to finish its current iteration
                await asyncio.wait_for(self._process_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._process_task.cancel()
                
        if self._poll_task:
            try:
                await asyncio.wait_for(self._poll_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._poll_task.cancel()
                
        if hasattr(self._broker, "disconnect"):
            await self._broker.disconnect()
            
        logger.info("LiveEngine: shutdown complete")

    async def _process_ticks(self) -> None:
        """Asynchronous stream consumer."""
        try:
            async for raw in self._feed.tick_stream():
                if not self._running:
                    break
                    
                try:
                    tick = self._normaliser.normalise(raw)
                    if not tick:
                        continue
                        
                    # Update LTP in broker for mock fill logic or tracking
                    if hasattr(self._broker, "update_ltp"):
                        self._broker.update_ltp(tick.symbol, tick.ltp)
                        
                    bar = self._bar_builder.update(tick)
                    if bar:
                        # Process sequentially to avoid race conditions
                        await self._process_bar(bar)
                        
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("LiveEngine: error processing tick: %s", exc, exc_info=True)
                    
        except asyncio.CancelledError:
            logger.info("LiveEngine: tick processing cancelled")
        except Exception as exc:
            logger.critical("LiveEngine: fatal stream error: %s", exc, exc_info=True)
        finally:
            self._running = False

    async def _process_bar(self, bar: Bar) -> None:
        """Updates indicators and routes to strategies."""
        ind_set = self._indicators_state.get(bar.symbol, {})
        current_inds = {}
        
        for key, ind in ind_set.items():
            try:
                val = ind.update(bar)
                current_inds[key] = val
                if key.startswith("supertrend_") and hasattr(ind, "trend_up"):
                    current_inds[f"{key}_up"] = ind.trend_up
            except Exception as exc:
                logger.error("LiveEngine: error updating indicator %s: %s", key, exc)
                
        for strategy in self._strategies:
            try:
                syms = set(s.upper() for s in getattr(strategy, "_config", {}).get("symbols", []))
                if not syms or bar.symbol.upper() in syms:
                    intents = strategy.on_bar(bar, current_inds)
                    for intent in intents:
                        ok, reason = self._risk_gate.approve(intent)
                        if ok:
                            order = await self._placer.place(intent)
                            if order:
                                self._risk_gate.position_cap.increment_open_orders()
                        else:
                            logger.debug("LiveEngine: intent rejected: %s", reason)
            except Exception as exc:
                logger.error("LiveEngine: error in strategy %s: %s", getattr(strategy, "_name", "unknown"), exc, exc_info=True)

    def on_fill(self, fill: Any) -> None:
        """Route a fill to the placer, observability, and strategies."""
        from engine.core.events import OrderState
        
        self._placer.record_fill(fill)
        
        if self._pos_book and self._pnl_engine:
            prev = self._pos_book.get(fill.symbol)
            self._pnl_engine.on_fill(fill, prev)
            self._pos_book.on_fill(fill)
            
        if self._blotter:
            self._blotter.record(fill)
            
        for strategy in self._strategies:
            if hasattr(strategy, "on_fill"):
                strategy.on_fill(fill)

    async def _poll_orders(self) -> None:
        """Background task to poll REST API for order statuses and create fills."""
        while self._running:
            await asyncio.sleep(self._poll_interval)
            await self._poll_orders_cycle()
            
    async def _poll_orders_cycle(self) -> None:
        """Single iteration of the polling logic. Isolated for testing."""
        from engine.core.events import Fill, OrderState
        from datetime import datetime, timezone
        from decimal import Decimal

        # 1. Fetch local active orders
        pending = self._placer._store.get_by_state(OrderState.PENDING)
        open_ = self._placer._store.get_by_state(OrderState.OPEN)
        active = pending + open_
        
        if not active:
            return

        get_orders = getattr(self._broker, "get_orders", None)
        if not callable(get_orders):
            return  # Polling requires get_orders
            
        try:
            # 2. Query broker
            broker_orders = await get_orders()
            # Map broker orders by id and tag
            by_id = {str(o.get("order_id")): o for o in broker_orders if "order_id" in o}
            by_tag = {str(o.get("tag")): o for o in broker_orders if "tag" in o}
            
            for local_order in active:
                b_o = None
                if local_order.broker_order_id:
                    b_o = by_id.get(local_order.broker_order_id)
                elif local_order.client_order_id[:20] in by_tag:
                    b_o = by_tag[local_order.client_order_id[:20]]
                    local_order.broker_order_id = str(b_o["order_id"])
                    self._placer._store.upsert(local_order)
                    
                if not b_o:
                    continue
                    
                # 3. Handle incremental fills
                broker_filled = int(b_o.get("filled_quantity", 0))
                incremental = broker_filled - local_order.filled_qty
                
                if incremental > 0:
                    if local_order.broker_order_id is None:
                        logger.warning("LiveEngine: skipping fill creation because broker_order_id is None for coid=%s", local_order.client_order_id)
                        continue

                    avg_price = Decimal(str(b_o.get("average_price", 0)))
                    fill = Fill(
                        order_id=local_order.client_order_id,
                        broker_order_id=local_order.broker_order_id,
                        symbol=local_order.symbol,
                        side=local_order.side,
                        fill_price=avg_price,
                        fill_qty=incremental,
                        fees=Decimal("0"),
                        ts=datetime.now(timezone.utc)
                    )
                    # Route through standard local fill handler
                    self.on_fill(fill)
                    
                # 4. Handle terminal state rejections/cancellations
                # If it's fully filled, record_fill already set it to COMPLETE
                status = b_o.get("status", "").upper()
                # Only check if local order isn't COMPLETE yet after possible fill
                current_local = self._placer._store.get(local_order.client_order_id)
                if current_local and current_local.state != OrderState.COMPLETE:
                    if status == "CANCELLED":
                        current_local.state = OrderState.CANCELLED
                        self._placer._store.upsert(current_local)
                    elif status == "REJECTED":
                        current_local.state = OrderState.REJECTED
                        self._placer._store.upsert(current_local)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("LiveEngine: polling error: %s", exc)

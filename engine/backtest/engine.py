"""
engine/backtest/engine.py
--------------------------
Event-driven backtest engine.

The backtest engine drives historical bar data through the SAME code path
as the live engine:

    BarEvent → IndicatorEngine → Strategy → RiskGate → FillModel → PnL

The only differences from live:
* ``SimClock`` is advanced to each bar's timestamp before processing.
* ``BarFillModel`` is used instead of ``MockBrokerAdapter``.
* No asyncio — runs synchronously for simplicity and speed.

No-lookahead guarantee
~~~~~~~~~~~~~~~~~~~~~~
The clock is advanced BEFORE the bar is passed to the strategy.
The bar's CLOSE is never used for fills (only bar OPEN of the *next* bar
is used for MARKET orders in ``BarFillModel``).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

from engine.core.clock import SimClock
from engine.core.events import Bar, Fill, Order, OrderIntent, OrderState, Side
from engine.core.instrument import InstrumentRegistry
from engine.indicators import ATR, EMA, RSI, MACD, BollingerBands, Supertrend, VWAP
from engine.backtest.fill_model import BarFillModel
from engine.oms.order import intent_to_order
from engine.oms.state_store import OrderStateStore
from engine.position.position_book import PositionBook
from engine.position.pnl_engine import PnLEngine
from engine.position.cost_model import CostModel
from engine.regime.circuit_breaker import CircuitBreaker
from engine.risk.risk_gate import RiskGate
from engine.strategy.base import BaseStrategy
from engine.observability.blotter import TradeBlotter

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Synchronous event-driven backtest engine.

    Parameters
    ----------
    strategies:
        List of strategy instances to run.
    risk_gate:
        Pre-built RiskGate instance.
    registry:
        InstrumentRegistry for instrument lookups.
    cost_model:
        CostModel for fee calculations.
    slippage_ticks:
        Ticks of slippage on MARKET orders.
    blotter:
        Optional blotter for recording trades.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        risk_gate: RiskGate,
        registry: InstrumentRegistry,
        cost_model: CostModel | None = None,
        slippage_ticks: int = 1,
        blotter: "TradeBlotter | None" = None,
        circuit_breaker: CircuitBreaker | None = None,   # B9: wired into fill path
        regime_engine=None,                              # RegimeEngine | None
    ) -> None:
        self._strategies  = strategies
        self._risk_gate   = risk_gate
        self._registry    = registry
        self._cost_model  = cost_model
        self._blotter     = blotter
        self._circuit_breaker = circuit_breaker          # B9
        self._regime_engine   = regime_engine            # regime integration
        self._clock       = SimClock()

        self._fill_model  = BarFillModel(
            cost_model=cost_model,
            slippage_ticks=slippage_ticks,
            on_fill=self._on_fill,
        )
        self._position_book = PositionBook(
            lot_size_fn=lambda s: self._get_lot_size(s)
        )
        self._pnl_engine  = PnLEngine(
            lot_size_fn=lambda s: self._get_lot_size(s)
        )
        self._store       = OrderStateStore(":memory:")
        self._fills: list[Fill] = []
        self._bars_processed = 0

        # Per-symbol indicator sets (lazily initialised)
        self._indicators: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, bars: list[Bar]) -> dict:
        """
        Run the backtest on a list of bars (sorted ascending by ts).

        Returns a summary dict with PnL and trade statistics.
        """
        logger.info("BacktestEngine: starting run with %d bars", len(bars))
        pending_orders: list[Order] = []   # orders waiting for next bar

        for i, bar in enumerate(bars):
            # Advance simulated clock (no lookahead: clock == bar time)
            self._clock.advance(bar.ts)
            # B1: derive a stable ms timestamp from the bar (not wall-clock)
            bar_ts_ms = int(bar.ts.timestamp() * 1000)

            # 0. Regime engine — update macro state before strategies run
            if self._regime_engine is not None:
                self._regime_engine.on_bar(bar.ts)

            # 1. Fill orders from PREVIOUS bar against THIS bar
            fills_this_bar = self._fill_model.process_bar(bar)
            for f in fills_this_bar:
                self._update_risk_tracking(f)

            # 2. Update indicators
            ind = self._update_indicators(bar)

            # 3. Run strategies → collect order intents
            for strategy in self._strategies:
                sym_filter = self._strategy_symbols(strategy)  # B4: None = match-all
                if sym_filter is None or bar.symbol in sym_filter:
                    intents = strategy.on_bar(bar, ind)
                    for intent in intents:
                        approved, reason = self._risk_gate.approve(intent)
                        if approved:
                            # B1: pass bar_ts_ms so ID is deterministic per bar
                            order = intent_to_order(intent, bar_ts_ms)
                            self._store.upsert(order)
                            self._fill_model.submit(order, self._registry.get(intent.symbol))
                        else:
                            logger.debug("BacktestEngine: rejected intent — %s", reason)

            self._bars_processed += 1

        logger.info(
            "BacktestEngine: run complete  bars=%d  fills=%d",
            self._bars_processed, len(self._fills),
        )
        return self.summary()

    def summary(self) -> dict:
        pnl = self._pnl_engine.summary()
        pnl["bars_processed"] = self._bars_processed
        pnl["total_fills"] = len(self._fills)
        pnl["approved_orders"] = self._risk_gate.approved_count
        pnl["rejected_orders"] = self._risk_gate.rejected_count
        return pnl

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def pnl_engine(self) -> PnLEngine:
        return self._pnl_engine

    @property
    def position_book(self) -> PositionBook:
        return self._position_book

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_fill(self, fill: Fill) -> None:
        self._fills.append(fill)
        prev_pos = self._position_book.get(fill.symbol)
        realised = self._pnl_engine.on_fill(fill, prev_pos)
        self._position_book.on_fill(fill)

        # B9: feed realised P&L into circuit breaker so daily limits are enforced
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_pnl(realised, fill.ts)

        # Notify strategies
        for strategy in self._strategies:
            if strategy.strategy_id in fill.order_id:
                strategy.on_fill(fill)

        # Update risk tracking
        self._risk_gate.position_cap.update_position(
            fill.symbol, fill.fill_qty if fill.side == Side.BUY else -fill.fill_qty
        )

        if self._blotter:
            self._blotter.record(fill)

        logger.debug(
            "BacktestEngine: fill sym=%s side=%s qty=%d @%.2f  realised=%.2f  fees=%.2f",
            fill.symbol, fill.side.value, fill.fill_qty,
            fill.fill_price, realised, fill.fees,
        )

    def _update_risk_tracking(self, fill: Fill) -> None:
        equity = self._pnl_engine.total_equity()
        self._risk_gate.drawdown_guard.update_equity(equity)

    def _update_indicators(self, bar: Bar) -> dict[str, Any]:
        sym = bar.symbol
        if sym not in self._indicators:
            self._indicators[sym] = self._build_indicators()

        ind = self._indicators[sym]
        out: dict[str, Any] = {}

        for key, indicator in ind.items():
            val = indicator.update(bar)
            out[key] = val

            # Expose Supertrend trend direction
            if key.startswith("supertrend_") and hasattr(indicator, "trend_up"):
                out[f"{key}_up"] = indicator.trend_up

        return out

    def _build_indicators(self) -> dict[str, Any]:
        return {
            "atr_14":      ATR(14),
            "ema_20":      EMA(20),
            "ema_50":      EMA(50),
            "rsi_14":      RSI(14),
            "macd":        MACD(12, 26, 9),
            "bb_20":       BollingerBands(20, 2.0),
            "supertrend_7": Supertrend(7, 3.0),
            "supertrend_14": Supertrend(14, 3.0),
            "vwap":        VWAP(),
        }

    def _strategy_symbols(self, strategy: BaseStrategy) -> set[str] | None:
        """Return the set of upper-cased symbols this strategy listens to.

        Returns ``None`` (match-all) when the strategy has no explicit symbol
        list configured.  Previously returned ``{""}`` which never matched any
        real symbol — B4 fix.
        """
        symbols = strategy._config.get("symbols", [])
        if not symbols:
            return None   # match every bar symbol
        return {s.upper() for s in symbols}

    def _get_lot_size(self, symbol: str) -> int:
        try:
            return self._registry.get(symbol).lot_size
        except KeyError:
            return 1

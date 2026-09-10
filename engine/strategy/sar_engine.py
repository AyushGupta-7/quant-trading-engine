"""
engine/strategy/sar_engine.py
------------------------------
Stop-and-Reverse (SAR) strategy with ATR-based stop placement and pyramiding.

Algorithm
~~~~~~~~~
1. Determine trend direction using Supertrend or EMA cross.
2. Enter LONG on uptrend signal (MARKET order, 1 lot).
3. Place ATR-based trailing stop.
4. On trend reversal: close existing position (MARKET) + open opposite.
5. Pyramid: add 1 lot on each new confirmed bar in the same direction,
   up to ``max_pyramid_levels``.
6. Position cap enforced independently by RiskGate.

The strategy never cancels its own stops — it relies on the broker's SL order
mechanism or the mock broker's stop simulation.
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal
from typing import Any

from engine.core.events import Bar, Fill, OrderIntent, OrderType, RegimeState, Side
from engine.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class SAREngine(BaseStrategy):
    """
    Stop-and-Reverse strategy.

    Config keys (all under ``strategies.sar``)::

        atr_period             : int   = 14
        atr_stop_multiplier    : float = 2.0
        pyramid_on_trend       : bool  = true
        max_pyramid_levels     : int   = 3
        position_cap_lots      : int   = 5
        order_type             : str   = "MARKET"
    """

    def __init__(self, strategy_id: str, config: dict) -> None:
        super().__init__(strategy_id, config)
        self._atr_period    = int(config.get("atr_period", 14))
        self._stop_mult     = float(config.get("atr_stop_multiplier", 2.0))
        self._pyramid       = bool(config.get("pyramid_on_trend", True))
        self._max_pyramid   = int(config.get("max_pyramid_levels", 3))
        self._pos_cap       = int(config.get("position_cap_lots", 5))
        self._order_type    = OrderType(config.get("order_type", "MARKET"))

        # State
        self._direction: Side | None = None   # current trend direction
        self._net_lots: int = 0
        self._pyramid_count: int = 0
        self._stop_price: Decimal | None = None
        self._prev_supertrend_up: bool | None = None

    # ------------------------------------------------------------------
    # BaseStrategy hooks
    # ------------------------------------------------------------------

    def _on_bar(self, bar: Bar, indicators: dict[str, Any]) -> list[OrderIntent]:
        atr_key  = f"atr_{self._atr_period}"
        st_key   = f"supertrend_{self._atr_period}"
        st_up_key = f"supertrend_{self._atr_period}_up"

        atr = indicators.get(atr_key)
        st_up = indicators.get(st_up_key)

        if atr is None or math.isnan(atr) or st_up is None:
            return []

        intents: list[OrderIntent] = []
        trend_changed = (self._prev_supertrend_up is not None and st_up != self._prev_supertrend_up)

        if trend_changed:
            intents.extend(self._handle_reversal(bar, atr, st_up))
        elif self._direction is not None:
            intents.extend(self._handle_pyramid(bar, atr, st_up))

        # First signal ever
        if self._direction is None and st_up is not None:
            intents.extend(self._enter_initial(bar, atr, st_up))

        self._prev_supertrend_up = st_up
        return intents

    def _on_fill(self, fill: Fill) -> None:
        if fill.side == Side.BUY:
            self._net_lots += fill.fill_qty
            self._pyramid_count += 1
        else:
            self._net_lots -= fill.fill_qty
            self._pyramid_count = max(0, self._pyramid_count - 1)

        logger.debug(
            "SAREngine[%s]: fill side=%s qty=%d @%.2f  net_lots=%d",
            self.strategy_id, fill.side.value, fill.fill_qty,
            fill.fill_price, self._net_lots,
        )

    def _on_regime_change(self, new_state: RegimeState, overrides: dict) -> None:
        self._stop_mult   = float(self._config.get("atr_stop_multiplier", 2.0))
        self._max_pyramid = int(self._config.get("max_pyramid_levels", 3))
        self._pos_cap     = int(self._config.get("position_cap_lots", 5))
        logger.info(
            "SAREngine[%s]: params updated for regime=%s  stop_mult=%.2f",
            self.strategy_id, new_state.value, self._stop_mult,
        )

    # ------------------------------------------------------------------
    # Internal signal logic
    # ------------------------------------------------------------------

    def _enter_initial(self, bar: Bar, atr: float, trend_up: bool) -> list[OrderIntent]:
        """Place the very first position."""
        side = Side.BUY if trend_up else Side.SELL
        self._direction = side
        self._update_stop(bar, atr, trend_up)
        logger.info(
            "SAREngine[%s]: initial entry side=%s @%.2f",
            self.strategy_id, side.value, bar.close,
        )
        return [self._make_intent(bar, side, qty=1, order_type=self._order_type, tag="sar_entry")]

    def _handle_reversal(self, bar: Bar, atr: float, trend_up: bool) -> list[OrderIntent]:
        """Close current position and reverse."""
        intents = []

        # Close existing position
        if self._net_lots != 0:
            close_side = Side.SELL if self._net_lots > 0 else Side.BUY
            close_qty  = abs(self._net_lots)
            intents.append(
                self._make_intent(bar, close_side, qty=close_qty,
                                  order_type=OrderType.MARKET, tag="sar_close")
            )

        # Open new position in reversed direction
        new_side = Side.BUY if trend_up else Side.SELL
        self._direction = new_side
        self._pyramid_count = 0
        self._update_stop(bar, atr, trend_up)

        intents.append(
            self._make_intent(bar, new_side, qty=1, order_type=self._order_type, tag="sar_reverse")
        )
        logger.info(
            "SAREngine[%s]: reversal → side=%s @%.2f stop=%.2f",
            self.strategy_id, new_side.value, bar.close, self._stop_price or 0,
        )
        return intents

    def _handle_pyramid(self, bar: Bar, atr: float, trend_up: bool) -> list[OrderIntent]:
        """Optionally add to the existing position (pyramid)."""
        if not self._pyramid:
            return []
        if self._pyramid_count >= self._max_pyramid:
            return []
        if abs(self._net_lots) >= self._pos_cap:
            return []

        self._update_stop(bar, atr, trend_up)
        logger.debug(
            "SAREngine[%s]: pyramid #%d side=%s",
            self.strategy_id, self._pyramid_count + 1, self._direction.value,
        )
        return [
            self._make_intent(
                bar, self._direction, qty=1,
                order_type=self._order_type, tag=f"sar_pyramid_{self._pyramid_count + 1}"
            )
        ]

    def _update_stop(self, bar: Bar, atr: float, trend_up: bool) -> None:
        stop_dist = Decimal(str(round(atr * self._stop_mult, 2)))
        if trend_up:
            self._stop_price = bar.close - stop_dist
        else:
            self._stop_price = bar.close + stop_dist

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def direction(self) -> Side | None:
        return self._direction

    @property
    def net_lots(self) -> int:
        return self._net_lots

    @property
    def stop_price(self) -> Decimal | None:
        return self._stop_price

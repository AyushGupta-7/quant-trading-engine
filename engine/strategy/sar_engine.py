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
        super().__init__(strategy_id, config, config_prefix="sar")   # B10
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
        # B3: track inflight reversal so we don't issue duplicate close+open orders
        self._reversal_pending: bool = False  # True while close order is in flight
        # Side of the CLOSING order (opposite to the NEW direction after reversal)
        self._reversal_close_side: Side | None = None
        # Stop-order deduplication: track the price of the currently-submitted stop.
        # A new stop intent is only emitted when the stop price CHANGES.
        # Set to None when no stop is in flight (initial state, post-fill, post-reversal).
        self._pending_stop_price: Decimal | None = None
        self._stop_tag: str = ""   # tag of the currently-pending stop order

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

        # B3: while a close order is in flight, do not issue more orders
        if self._reversal_pending:
            self._prev_supertrend_up = st_up
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
        # Detect stop-order fill: a fill that arrives at the pending stop price
        # with the side that would close (not extend) the current position.
        # This is unambiguous because stop orders always have the OPPOSITE side
        # to the current direction, and the price must match the pending stop.
        is_stop_fill = (
            self._pending_stop_price is not None
            and fill.fill_price == self._pending_stop_price
            and self._direction is not None
            and fill.side != self._direction  # opposite side = closing
        )

        if fill.side == Side.BUY:
            self._net_lots += fill.fill_qty
            if not is_stop_fill:
                self._pyramid_count += 1
        else:
            self._net_lots -= fill.fill_qty
            if not is_stop_fill:
                self._pyramid_count = max(0, self._pyramid_count - 1)

        # Stop fill: position was closed by stop-loss; reset state
        if is_stop_fill:
            self._pending_stop_price = None
            self._stop_tag = ""
            self._stop_price = None
            self._pyramid_count = 0
            self._direction = None
            logger.info(
                "SAREngine[%s]: stop-loss FILLED side=%s qty=%d @%.2f  net_lots=%d",
                self.strategy_id, fill.side.value, fill.fill_qty,
                fill.fill_price, self._net_lots,
            )
        else:
            # B3: detect the closing fill of a reversal.
            if (
                self._reversal_pending
                and self._reversal_close_side is not None
                and fill.side == self._reversal_close_side
            ):
                # This is the closing fill — clear guard and reset counter
                self._reversal_pending = False
                self._reversal_close_side = None
                self._pending_stop_price = None   # stop cancelled by reversal
                self._stop_tag = ""
                self._pyramid_count = 0
                logger.debug(
                    "SAREngine[%s]: reversal close confirmed, resuming normal operation",
                    self.strategy_id,
                )

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
        """Place the very first position and its protective stop."""
        side = Side.BUY if trend_up else Side.SELL
        self._direction = side
        self._update_stop(bar, atr, trend_up)
        logger.info(
            "SAREngine[%s]: initial entry side=%s @%.2f stop=%.2f",
            self.strategy_id, side.value, bar.close, self._stop_price,
        )
        intents = [self._make_intent(bar, side, qty=1, order_type=self._order_type, tag="sar_entry")]
        intents.extend(self._make_stop_intent(bar, trend_up))
        return intents

    def _handle_reversal(self, bar: Bar, atr: float, trend_up: bool) -> list[OrderIntent]:
        """Close current position and reverse.

        B3: sets ``_reversal_pending = True`` so subsequent bars are frozen until
        the close fill confirms.  ``_pyramid_count`` is NOT reset here — it will
        be reset in ``_on_fill`` when the closing fill is recognised, avoiding
        the race that previously drove ``pyramid_count`` negative.
        """
        intents = []

        # Cancel pending stop tracking: the close order overrides the stop.
        # Clearing _pending_stop_price means the reversal's new stop entry
        # will always be emitted even if the price is numerically the same.
        self._pending_stop_price = None
        self._stop_tag = ""

        # Close existing position
        if self._net_lots != 0:
            close_side = Side.SELL if self._net_lots > 0 else Side.BUY
            close_qty  = abs(self._net_lots)
            intents.append(
                self._make_intent(bar, close_side, qty=close_qty,
                                  order_type=OrderType.MARKET, tag="sar_close")
            )
            # B3: record the closing side so _on_fill can identify the close fill
            self._reversal_close_side = close_side
            # B3: arm the guard BEFORE returning so no second reversal fires
            self._reversal_pending = True

        # Update direction for the new position
        new_side = Side.BUY if trend_up else Side.SELL
        self._direction = new_side
        # NOTE: _pyramid_count is intentionally NOT reset here (B3 fix).
        #       It is reset in _on_fill once the close fill is confirmed.
        self._update_stop(bar, atr, trend_up)

        intents.append(
            self._make_intent(bar, new_side, qty=1, order_type=self._order_type, tag="sar_reverse")
        )
        # Emit stop for the new position (only after reversal, not while pending)
        intents.extend(self._make_stop_intent(bar, trend_up))
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
        intents = [
            self._make_intent(
                bar, self._direction, qty=1,
                order_type=self._order_type, tag=f"sar_pyramid_{self._pyramid_count + 1}"
            )
        ]
        # Update stop for the enlarged position
        intents.extend(self._make_stop_intent(bar, trend_up))
        return intents

    def _update_stop(self, bar: Bar, atr: float, trend_up: bool) -> None:
        """Recompute stop price. Does NOT touch the dedup state.

        Deduplication is handled in _make_stop_intent by comparing
        _stop_price against _pending_stop_price.
        """
        stop_dist = Decimal(str(round(atr * self._stop_mult, 2)))
        if trend_up:
            self._stop_price = bar.close - stop_dist
        else:
            self._stop_price = bar.close + stop_dist

    def _make_stop_intent(self, bar: Bar, trend_up: bool) -> list[OrderIntent]:
        """Emit a SL_M stop intent for the current position.

        A new stop is only emitted when the computed stop price differs from
        the price of the currently-submitted stop.  This correctly deduplicates
        without clearing state in _update_stop (which would cause a new stop
        to be emitted every bar even when the price is unchanged).
        """
        if self._stop_price is None:
            return []
        # Deduplicate: skip if the same price is already in flight
        if self._pending_stop_price == self._stop_price:
            return []

        # Stop side is opposite to the current position
        stop_side = Side.SELL if trend_up else Side.BUY
        tag = f"sar_stop_{self._pyramid_count}"
        self._pending_stop_price = self._stop_price
        self._stop_tag = tag
        logger.debug(
            "SAREngine[%s]: submitting stop side=%s @%.2f tag=%s",
            self.strategy_id, stop_side.value, self._stop_price, tag,
        )
        return [
            self._make_intent(
                bar, stop_side, qty=abs(self._net_lots) + 1,  # +1 for the pending entry
                order_type=OrderType.SL_M,
                price=self._stop_price,
                tag=tag,
            )
        ]

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

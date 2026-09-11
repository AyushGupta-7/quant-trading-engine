"""
engine/strategy/grid_engine.py
--------------------------------
Grid trading strategy with ATR-based spacing, pyramiding, and position caps.

Algorithm
~~~~~~~~~
1. On first bar after ATR is ready: set initial grid anchor at current price.
2. Calculate grid levels: ``anchor ± n × (ATR × atr_multiplier)``
3. Place LIMIT BUY orders at each lower grid level, LIMIT SELL at upper.
4. On fill: move anchor to fill price; recalculate grid (pyramid into trend).
5. Enforce ``max_grid_levels`` and ``position_cap_lots``.
6. Kill switch: cancel all pending orders and freeze.

Pyramiding logic:
  * After a BUY fill at grid level N, the new anchor is pulled up by one
    grid step, allowing additional BUY orders to be placed lower.
  * ``max_pyramid_levels`` caps the total number of open BUY (or SELL) lots.
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal
from typing import Any

from engine.core.events import Bar, Fill, OrderIntent, OrderType, RegimeState, Side
from engine.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class GridEngine(BaseStrategy):
    """
    ATR-spaced grid trading strategy.

    Config keys (all under ``strategies.grid``)::

        atr_period          : int   = 14
        atr_multiplier      : float = 1.0
        max_grid_levels     : int   = 5
        max_pyramid_levels  : int   = 3
        position_cap_lots   : int   = 5
        order_type          : str   = "LIMIT"
    """

    def __init__(self, strategy_id: str, config: dict) -> None:
        super().__init__(strategy_id, config, config_prefix="grid")  # B10
        self._atr_period     = int(config.get("atr_period", 14))
        self._atr_mult       = float(config.get("atr_multiplier", 1.0))
        self._max_levels     = int(config.get("max_grid_levels", 5))
        self._max_pyramid    = int(config.get("max_pyramid_levels", 3))
        self._pos_cap        = int(config.get("position_cap_lots", 5))
        self._order_type     = OrderType(config.get("order_type", "LIMIT"))

        # State
        self._anchor: Decimal | None = None
        self._grid_spacing: Decimal = Decimal(0)
        self._net_lots: int = 0          # positive = long, negative = short
        self._pyramid_count: int = 0

        # Track which grid levels already have pending orders
        # key = Decimal price, value = Side
        self._pending_levels: dict[Decimal, Side] = {}

    # ------------------------------------------------------------------
    # BaseStrategy hooks
    # ------------------------------------------------------------------

    def _on_bar(self, bar: Bar, indicators: dict[str, Any]) -> list[OrderIntent]:
        atr = indicators.get(f"atr_{self._atr_period}")
        if atr is None or math.isnan(atr) or atr <= 0:
            return []

        spacing = Decimal(str(round(atr * self._atr_mult, 2)))
        if spacing <= 0:
            return []

        self._grid_spacing = spacing

        if self._anchor is None:
            # Initialise anchor at current mid-price
            self._anchor = bar.close
            logger.info(
                "GridEngine[%s]: anchor initialised at %.2f, spacing=%.2f",
                self.strategy_id, self._anchor, self._grid_spacing,
            )

        # B2: prune levels that are now outside the valid grid range
        self._prune_stale_levels()

        return self._compute_orders(bar)

    def _on_fill(self, fill: Fill) -> None:
        if fill.side == Side.BUY:
            self._net_lots += fill.fill_qty
            self._pyramid_count += 1
        else:
            self._net_lots -= fill.fill_qty
            self._pyramid_count = max(0, self._pyramid_count - 1)

        # Remove only the exact filled level
        self._pending_levels.pop(fill.fill_price, None)

        # Move anchor towards fill price (trend-following adjustment)
        if self._anchor is not None and self._grid_spacing > 0:
            if fill.side == Side.BUY:
                # Anchor moves up by half a grid step on buys
                self._anchor = fill.fill_price + self._grid_spacing / 2
            else:
                self._anchor = fill.fill_price - self._grid_spacing / 2

            # B2: anchor has moved — all old levels are now stale; clear them.
            # Fresh levels will be computed in _compute_orders on the next bar.
            self._pending_levels.clear()
            logger.debug(
                "GridEngine[%s]: anchor moved to %.2f — pending_levels cleared",
                self.strategy_id, self._anchor,
            )

        logger.debug(
            "GridEngine[%s]: fill side=%s qty=%d @%.2f  net_lots=%d anchor=%.2f",
            self.strategy_id, fill.side.value, fill.fill_qty,
            fill.fill_price, self._net_lots, self._anchor,
        )

    def _on_regime_change(self, new_state: RegimeState, overrides: dict) -> None:
        # Reload parameters that may have changed via regime overrides
        self._atr_mult    = float(self._config.get("atr_multiplier", 1.0))
        self._max_levels  = int(self._config.get("max_grid_levels", 5))
        self._pos_cap     = int(self._config.get("position_cap_lots", 5))
        logger.info(
            "GridEngine[%s]: params updated for regime=%s  "
            "atr_mult=%.2f max_levels=%d pos_cap=%d",
            self.strategy_id, new_state.value,
            self._atr_mult, self._max_levels, self._pos_cap,
        )

    # ------------------------------------------------------------------
    # Grid computation
    # ------------------------------------------------------------------

    def _compute_orders(self, bar: Bar) -> list[OrderIntent]:
        """Generate LIMIT buy/sell OrderIntents for each unfilled grid level."""
        intents: list[OrderIntent] = []

        for i in range(1, self._max_levels + 1):
            buy_price  = self._anchor - i * self._grid_spacing
            sell_price = self._anchor + i * self._grid_spacing

            # --- BUY levels (below anchor) ---
            if (
                buy_price > 0
                and buy_price not in self._pending_levels
                and self._net_lots < self._pos_cap
                and self._pyramid_count < self._max_pyramid
            ):
                intents.append(
                    self._make_intent(
                        bar, Side.BUY, qty=1,
                        order_type=self._order_type,
                        price=buy_price,
                        tag=f"grid_buy_L{i}",
                    )
                )
                self._pending_levels[buy_price] = Side.BUY

            # --- SELL levels (above anchor) ---
            if (
                sell_price not in self._pending_levels
                and self._net_lots > -self._pos_cap
            ):
                intents.append(
                    self._make_intent(
                        bar, Side.SELL, qty=1,
                        order_type=self._order_type,
                        price=sell_price,
                        tag=f"grid_sell_L{i}",
                    )
                )
                self._pending_levels[sell_price] = Side.SELL

        return intents

    def _prune_stale_levels(self) -> None:
        """B2: Remove any pending level whose price is outside the current grid range.

        A level is stale when the anchor has moved far enough that the level
        is now more than ``max_grid_levels`` spacings away.  We drop it here
        so it is never presented as a duplicate guard in ``_compute_orders``
        and the corresponding fill-model order becomes an orphan that will
        simply expire without filling.
        """
        if self._anchor is None or self._grid_spacing <= 0:
            return
        max_dist = (self._max_levels + 1) * self._grid_spacing   # +1 for rounding
        stale = [
            price for price in list(self._pending_levels)
            if abs(price - self._anchor) > max_dist
        ]
        if stale:
            for price in stale:
                del self._pending_levels[price]
            logger.debug(
                "GridEngine[%s]: pruned %d stale pending levels",
                self.strategy_id, len(stale),
            )

    # ------------------------------------------------------------------
    # State accessors (for tests / observability)
    # ------------------------------------------------------------------

    @property
    def anchor(self) -> Decimal | None:
        return self._anchor

    @property
    def grid_spacing(self) -> Decimal:
        return self._grid_spacing

    @property
    def net_lots(self) -> int:
        return self._net_lots

    @property
    def pending_levels(self) -> dict[Decimal, Side]:
        return dict(self._pending_levels)

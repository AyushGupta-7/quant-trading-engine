"""
engine/observability/alerts.py
-------------------------------
Rule-based alert manager.

Checks predefined thresholds and fires alert callbacks (logs + optional webhook).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Callable

logger = logging.getLogger(__name__)

AlertCallback = Callable[[str, str, dict], None]   # (level, message, context)


def _log_alert(level: str, message: str, context: dict) -> None:
    fn = getattr(logger, level.lower(), logger.warning)
    fn("ALERT [%s]: %s | %s", level.upper(), message, context)


class AlertManager:
    """
    Fires alerts when metrics cross configured thresholds.

    Parameters
    ----------
    thresholds:
        Dict from ``observability.alert_thresholds`` config section.
    callbacks:
        List of alert handlers.  Defaults to logging only.
    """

    def __init__(
        self,
        thresholds: dict | None = None,
        callbacks: list[AlertCallback] | None = None,
    ) -> None:
        thresholds = thresholds or {}
        self._dd_pct_threshold  = float(thresholds.get("drawdown_pct", 0.03))
        self._pos_util_threshold = float(thresholds.get("position_utilisation_pct", 0.80))
        self._callbacks = callbacks or [_log_alert]
        self._fired: set[str] = set()   # deduplicate repeated alerts

    def check_drawdown(self, drawdown_pct: float, context: dict | None = None) -> None:
        key = "drawdown"
        if drawdown_pct >= self._dd_pct_threshold and key not in self._fired:
            self._fire(
                "WARNING", key,
                f"Drawdown {drawdown_pct:.2%} exceeds threshold {self._dd_pct_threshold:.2%}",
                context or {},
            )
        elif drawdown_pct < self._dd_pct_threshold:
            self._fired.discard(key)

    def check_position_utilisation(
        self, current_lots: int, max_lots: int, symbol: str = ""
    ) -> None:
        util = current_lots / max_lots if max_lots else 0
        key = f"pos_util_{symbol}"
        if util >= self._pos_util_threshold and key not in self._fired:
            self._fire(
                "WARNING", key,
                f"Position utilisation {util:.0%} for {symbol} (max={max_lots})",
                {"symbol": symbol, "current_lots": current_lots, "max_lots": max_lots},
            )
        elif util < self._pos_util_threshold:
            self._fired.discard(key)

    def fire_kill_switch_alert(self, reason: str) -> None:
        self._fire("CRITICAL", "kill_switch", f"Kill switch engaged: {reason}", {})

    def _fire(self, level: str, key: str, message: str, context: dict) -> None:
        self._fired.add(key)
        for cb in self._callbacks:
            try:
                cb(level, message, context)
            except Exception:
                logger.exception("AlertManager: callback failed")

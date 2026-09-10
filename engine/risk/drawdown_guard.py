"""
engine/risk/drawdown_guard.py
------------------------------
Monitors intraday and rolling drawdown from the equity peak.

Integrated into the RiskGate approval chain.
"""

from __future__ import annotations

import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class DrawdownGuard:
    """
    Tracks the rolling equity peak and computes drawdown.

    Parameters
    ----------
    max_drawdown_pct:
        Fractional drawdown allowed before blocking new orders (e.g. 0.05 = 5%).
    max_daily_loss:
        Absolute daily loss limit in INR.
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.05,
        max_daily_loss: float = 50_000.0,
    ) -> None:
        self._max_dd_pct  = Decimal(str(max_drawdown_pct))
        self._max_daily   = Decimal(str(max_daily_loss))
        self._peak_equity = Decimal(0)
        self._current_equity = Decimal(0)
        self._daily_loss  = Decimal(0)
        self._blocked     = False

    def update_equity(self, equity: Decimal) -> None:
        """Call after every fill with the latest total equity."""
        if equity > self._peak_equity:
            self._peak_equity = equity
        self._current_equity = equity

        pnl_change = equity - self._peak_equity
        if pnl_change < 0:
            self._daily_loss += abs(pnl_change)

    def approve(self, _intent=None) -> tuple[bool, str]:
        """Returns (True, "") if drawdown limits are within bounds."""
        if self._blocked:
            return False, "DrawdownGuard: already blocked (reset required)"

        if self._peak_equity > 0:
            dd = (self._peak_equity - self._current_equity) / self._peak_equity
            if dd >= self._max_dd_pct:
                self._blocked = True
                reason = (
                    f"DrawdownGuard: drawdown {dd:.2%} >= max {self._max_dd_pct:.2%}"
                )
                logger.critical(reason)
                return False, reason

        if self._daily_loss >= self._max_daily:
            self._blocked = True
            reason = (
                f"DrawdownGuard: daily loss ₹{self._daily_loss:.2f} "
                f">= limit ₹{self._max_daily:.2f}"
            )
            logger.critical(reason)
            return False, reason

        return True, ""

    def reset_daily(self) -> None:
        self._daily_loss = Decimal(0)
        self._blocked = False

    @property
    def drawdown_pct(self) -> Decimal:
        if self._peak_equity <= 0:
            return Decimal(0)
        return (self._peak_equity - self._current_equity) / self._peak_equity

    @property
    def blocked(self) -> bool:
        return self._blocked

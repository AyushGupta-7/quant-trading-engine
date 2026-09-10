"""
engine/regime/circuit_breaker.py
---------------------------------
Circuit breaker that monitors daily P&L and drawdown.

When a limit is breached it engages the kill switch by emitting a
``KillSwitchEvent`` onto the event bus and setting the kill-switch flag.

Tracked limits
~~~~~~~~~~~~~~
* ``daily_loss_limit``   : total realised + unrealised loss since midnight.
* ``max_consecutive_losses``: number of consecutive losing fills.
* These are independent of (but complementary to) the RiskGate drawdown guard.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal

from engine.core.events import KillSwitchEvent

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Monitors daily loss and consecutive losing fills.

    Parameters
    ----------
    daily_loss_limit:
        Maximum allowable loss (INR) before engaging kill switch.
    max_consecutive_losses:
        Maximum consecutive losing fills before engaging kill switch.
    on_breached:
        Optional callback invoked with a ``KillSwitchEvent`` when a limit fires.
    """

    def __init__(
        self,
        daily_loss_limit: float = 50_000.0,
        max_consecutive_losses: int = 5,
        on_breached=None,
    ) -> None:
        self._daily_loss_limit = Decimal(str(daily_loss_limit))
        self._max_consecutive_losses = max_consecutive_losses
        self._on_breached = on_breached

        self._daily_loss: Decimal = Decimal(0)
        self._consecutive_losses: int = 0
        self._engaged: bool = False
        self._session_date: str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def daily_loss(self) -> Decimal:
        return self._daily_loss

    def record_pnl(self, pnl: Decimal, ts: datetime | None = None) -> None:
        """
        Record a P&L event (can be negative for a loss).

        Called after every fill by the P&L engine.
        """
        ts = ts or datetime.now(tz=timezone.utc)
        self._check_day_rollover(ts)

        if pnl < 0:
            self._daily_loss += abs(pnl)
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._check_limits(ts)

    def reset_day(self) -> None:
        """Call at session open to reset daily counters."""
        self._daily_loss = Decimal(0)
        self._consecutive_losses = 0
        self._engaged = False
        self._session_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        logger.info("CircuitBreaker: daily counters reset.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_day_rollover(self, ts: datetime) -> None:
        today = ts.strftime("%Y-%m-%d")
        if today != self._session_date:
            self.reset_day()
            self._session_date = today

    def _check_limits(self, ts: datetime) -> None:
        if self._engaged:
            return

        reason: str | None = None

        if self._daily_loss >= self._daily_loss_limit:
            reason = (
                f"Daily loss limit breached: ₹{self._daily_loss:.2f} "
                f">= ₹{self._daily_loss_limit:.2f}"
            )
        elif self._consecutive_losses >= self._max_consecutive_losses:
            reason = (
                f"Consecutive losses limit: {self._consecutive_losses} "
                f">= {self._max_consecutive_losses}"
            )

        if reason:
            self._engaged = True
            logger.critical("CircuitBreaker ENGAGED: %s", reason)
            event = KillSwitchEvent(reason=reason, source="circuit_breaker", ts=ts)
            if self._on_breached:
                self._on_breached(event)

    @classmethod
    def from_config(cls, regime_cfg: dict, **kwargs) -> "CircuitBreaker":
        cb_cfg = regime_cfg.get("circuit_breaker", {})
        return cls(
            daily_loss_limit=float(cb_cfg.get("daily_loss_limit", 50_000)),
            max_consecutive_losses=int(cb_cfg.get("max_consecutive_losses", 5)),
            **kwargs,
        )

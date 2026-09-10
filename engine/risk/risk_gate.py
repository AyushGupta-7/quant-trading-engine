"""
engine/risk/risk_gate.py
-------------------------
Composes all risk guards into a single ``approve(OrderIntent) → bool`` call.

Guards checked in order (fast-fail):
  1. KillSwitch
  2. DrawdownGuard
  3. PositionCapGuard

A rejection at any step short-circuits the rest.
"""

from __future__ import annotations

import logging

from engine.core.events import OrderIntent
from engine.risk.drawdown_guard import DrawdownGuard
from engine.risk.kill_switch import KillSwitch
from engine.risk.position_cap import PositionCapGuard

logger = logging.getLogger(__name__)


class RiskGate:
    """
    Single entry point for all risk checks.

    Usage::

        gate = RiskGate.from_config(risk_cfg)
        approved, reason = gate.approve(intent)
        if approved:
            oms.place(intent)
    """

    def __init__(
        self,
        kill_switch: KillSwitch,
        drawdown_guard: DrawdownGuard,
        position_cap: PositionCapGuard,
    ) -> None:
        self._ks  = kill_switch
        self._dd  = drawdown_guard
        self._pc  = position_cap

        # Counters for observability
        self.approved_count = 0
        self.rejected_count = 0

    def approve(self, intent: OrderIntent) -> tuple[bool, str]:
        """
        Run all guards.  Returns ``(True, "")`` or ``(False, reason)``.
        """
        # 1. Kill switch
        if not self._ks.check():
            self.rejected_count += 1
            reason = "RiskGate: kill switch engaged"
            logger.warning("%s | symbol=%s side=%s", reason, intent.symbol, intent.side.value)
            return False, reason

        # 2. Drawdown
        ok, reason = self._dd.approve(intent)
        if not ok:
            self.rejected_count += 1
            return False, reason

        # 3. Position cap
        ok, reason = self._pc.approve(intent)
        if not ok:
            self.rejected_count += 1
            return False, reason

        self.approved_count += 1
        return True, ""

    def engage_kill_switch(self, reason: str = "manual") -> None:
        self._ks.engage(reason)

    @property
    def kill_switch(self) -> KillSwitch:
        return self._ks

    @property
    def drawdown_guard(self) -> DrawdownGuard:
        return self._dd

    @property
    def position_cap(self) -> PositionCapGuard:
        return self._pc

    @classmethod
    def from_config(cls, risk_cfg: dict, switch_file: str = "data/kill_switch.lock") -> "RiskGate":
        return cls(
            kill_switch=KillSwitch(switch_file=switch_file),
            drawdown_guard=DrawdownGuard(
                max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 0.05)),
                max_daily_loss=float(risk_cfg.get("max_daily_loss", 50_000)),
            ),
            position_cap=PositionCapGuard(
                max_position_lots=int(risk_cfg.get("max_position_lots", 10)),
                max_open_orders=int(risk_cfg.get("max_open_orders", 20)),
            ),
        )

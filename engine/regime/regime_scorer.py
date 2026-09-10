"""
engine/regime/regime_scorer.py
-------------------------------
Scores macro snapshots into ``RegimeState`` enum values using a rule-based
decision tree.

Rule table (configurable via config thresholds):

  CRISIS   → VIX ≥ bear_threshold AND USD/INR ≥ stress_threshold
  BEAR     → VIX ≥ bear_threshold (but not full CRISIS)
  BULL     → VIX < bull_threshold AND FII flow > 0
  SIDEWAYS → otherwise
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from engine.core.events import RegimeChangeEvent, RegimeState
from engine.regime.macro_ingester import MacroSnapshot

logger = logging.getLogger(__name__)


class RegimeScorer:
    """
    Converts a ``MacroSnapshot`` → ``RegimeState``.

    Parameters
    ----------
    cfg:
        The ``regime.proxies`` section of config.
    """

    def __init__(self, cfg: dict) -> None:
        vix_cfg  = cfg.get("india_vix", {})
        inr_cfg  = cfg.get("usd_inr", {})
        fii_cfg  = cfg.get("fii_flow_crore", {})

        self._vix_bull     = float(vix_cfg.get("bull_threshold", 20.0))
        self._vix_bear     = float(vix_cfg.get("bear_threshold", 25.0))
        self._inr_stress   = float(inr_cfg.get("stress_threshold", 85.0))
        self._fii_outflow  = float(fii_cfg.get("outflow_threshold", -1000.0))

        self._current = RegimeState.UNKNOWN
        self._history: list[tuple[datetime, RegimeState]] = []

    def score(self, snap: MacroSnapshot) -> RegimeState:
        """Score a ``MacroSnapshot`` and return the new ``RegimeState``."""
        vix = snap.india_vix
        inr = snap.usd_inr
        fii = snap.fii_flow_crore

        if vix >= self._vix_bear and inr >= self._inr_stress:
            new_state = RegimeState.CRISIS
        elif vix >= self._vix_bear:
            new_state = RegimeState.BEAR
        elif vix < self._vix_bull and fii > 0:
            new_state = RegimeState.BULL
        else:
            new_state = RegimeState.SIDEWAYS

        if new_state != self._current:
            logger.info(
                "Regime change: %s → %s  (vix=%.1f, usd_inr=%.2f, fii=%.0f)",
                self._current.value, new_state.value, vix, inr, fii,
            )
            self._history.append((snap.ts, new_state))
            self._current = new_state

        return self._current

    @property
    def current(self) -> RegimeState:
        return self._current

    def make_change_event(self, previous: RegimeState, current: RegimeState, snap: MacroSnapshot) -> RegimeChangeEvent:
        return RegimeChangeEvent(
            previous=previous,
            current=current,
            ts=snap.ts,
            details=snap.to_dict(),
        )

    def history(self) -> list[tuple[datetime, RegimeState]]:
        return list(self._history)

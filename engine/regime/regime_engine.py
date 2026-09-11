"""
engine/regime/regime_engine.py
--------------------------------
Thin orchestrator that wires the regime pipeline into the backtest/live loop.

Full pipeline::

    MockMacroIngester
        ↓  get_snapshot()
    RegimeScorer
        ↓  score(snapshot) → RegimeState
    ParameterOverrideRegistry
        ↓  get_overrides(state) → dict
    strategy.on_regime_change(state, overrides)
        ↓
    CircuitBreaker (called by BacktestEngine._on_fill)
        ↓  record_pnl(realised, ts)
    RiskGate.engage_kill_switch()

The ``RegimeEngine`` is designed to be called ONCE PER BAR from the
backtest loop (or on a timer in the live path).  It keeps track of the
previous regime state so it only calls ``on_regime_change`` when the regime
ACTUALLY CHANGES — strategies are not spammed with redundant updates.

Usage
~~~~~
::

    regime_engine = RegimeEngine.from_config(cfg, strategies, risk_gate)

    # inside the bar loop:
    regime_engine.on_bar(bar)   # feeds macro, scores, propagates if changed
"""

from __future__ import annotations

import logging
from datetime import datetime

from engine.core.events import RegimeState
from engine.regime.circuit_breaker import CircuitBreaker
from engine.regime.macro_ingester import MockMacroIngester, MacroSnapshot
from engine.regime.param_override import ParameterOverrideRegistry
from engine.regime.regime_scorer import RegimeScorer
from engine.risk.risk_gate import RiskGate
from engine.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class RegimeEngine:
    """
    End-to-end regime pipeline orchestrator.

    Parameters
    ----------
    ingester:
        Source of macro proxy data.
    scorer:
        Converts a MacroSnapshot → RegimeState.
    override_registry:
        Maps RegimeState → parameter override dict.
    strategies:
        List of strategies to notify on regime changes.
    circuit_breaker:
        CircuitBreaker instance (may be None; record_pnl must be called
        separately by the backtest engine after each fill).
    risk_gate:
        RiskGate whose kill switch the circuit breaker engages.
    regime_update_every_n_bars:
        Re-score macro every N bars (default 1 = every bar).
        In practice a 5-minute bar engine might re-score every 60 bars.
    """

    def __init__(
        self,
        ingester: MockMacroIngester,
        scorer: RegimeScorer,
        override_registry: ParameterOverrideRegistry,
        strategies: list[BaseStrategy],
        circuit_breaker: CircuitBreaker | None = None,
        risk_gate: RiskGate | None = None,
        regime_update_every_n_bars: int = 1,
    ) -> None:
        self._ingester   = ingester
        self._scorer     = scorer
        self._registry   = override_registry
        self._strategies = strategies
        self._cb         = circuit_breaker
        self._risk_gate  = risk_gate
        self._update_every = max(1, regime_update_every_n_bars)

        self._bar_count = 0
        self._current_state: RegimeState = RegimeState.UNKNOWN

        # Wire circuit-breaker → kill switch (if both present)
        if self._cb is not None and self._risk_gate is not None:
            # Wrap the existing callback or set a new one
            _orig = self._cb._on_breached
            def _combined_cb(event):
                self._risk_gate.engage_kill_switch("CB: " + event.reason)
                if _orig:
                    _orig(event)
            self._cb._on_breached = _combined_cb

    # ------------------------------------------------------------------
    # Public API — call from bar loop
    # ------------------------------------------------------------------

    def on_bar(self, bar_ts: datetime | None = None) -> RegimeState:
        """
        Called once per bar by the backtest engine.

        Fetches a macro snapshot every ``regime_update_every_n_bars`` bars,
        scores it, and propagates any state change to strategies.

        Returns the current regime state (may be unchanged from previous call).
        """
        self._bar_count += 1
        if self._bar_count % self._update_every != 0:
            return self._current_state

        snap = self._ingester.get_snapshot(ts=bar_ts)
        new_state = self._scorer.score(snap)

        if new_state != self._current_state:
            prev = self._current_state
            self._current_state = new_state
            overrides = self._registry.get_overrides(new_state)
            logger.info(
                "RegimeEngine: state %s → %s  overrides=%s",
                prev.value, new_state.value, overrides,
            )
            for strategy in self._strategies:
                strategy.on_regime_change(new_state, overrides)

        return self._current_state

    @property
    def current_state(self) -> RegimeState:
        return self._current_state

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        return self._cb

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        strategies: list[BaseStrategy],
        risk_gate: RiskGate | None = None,
        regime_update_every_n_bars: int = 1,
    ) -> "RegimeEngine":
        """
        Build a ``RegimeEngine`` from the ``regime`` section of config.

        Parameters
        ----------
        cfg:
            Full config dict (``regime`` section extracted internally).
        strategies:
            Strategy instances to receive regime updates.
        risk_gate:
            RiskGate whose kill switch the circuit breaker engages on breach.
        regime_update_every_n_bars:
            Re-score every N bars (default 1 = every bar).
        """
        regime_cfg = cfg.get("regime", {})

        ingester  = MockMacroIngester(regime_cfg.get("proxies", {}))
        scorer    = RegimeScorer(regime_cfg.get("proxies", {}))
        overrides = ParameterOverrideRegistry.from_config(regime_cfg)

        # Build CircuitBreaker — the on_breached callback is wired in __init__
        cb = CircuitBreaker.from_config(regime_cfg)

        return cls(
            ingester=ingester,
            scorer=scorer,
            override_registry=overrides,
            strategies=strategies,
            circuit_breaker=cb,
            risk_gate=risk_gate,
            regime_update_every_n_bars=regime_update_every_n_bars,
        )

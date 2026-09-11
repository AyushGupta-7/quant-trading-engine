"""
engine/strategy/base.py
------------------------
Abstract base class for all trading strategies.

Strategies are pure signal generators: they receive bar/tick/fill events
and return ``list[OrderIntent]``.  They never interact with the broker
or risk gate directly — that is the engine's responsibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from engine.core.events import Bar, Fill, OrderIntent, RegimeState, Tick


class BaseStrategy(ABC):
    """
    Abstract trading strategy.

    Lifecycle::

        strategy = ConcreteStrategy("my-strategy", config)
        strategy.on_regime_change(RegimeState.BULL)
        for bar in bars:
            intents = strategy.on_bar(bar, indicators)
            # pass intents to RiskGate
        strategy.on_fill(fill)

    Parameters
    ----------
    strategy_id:
        Unique identifier (used in order tags, logs, blotter).
    config:
        Strategy-specific config dict (from config YAML).
    """

    def __init__(self, strategy_id: str, config: dict, config_prefix: str = "") -> None:
        self.strategy_id = strategy_id
        self._config = dict(config)
        self._config_prefix = config_prefix   # B10: e.g. "grid" or "sar"
        self._enabled: bool = bool(config.get("enabled", True))
        self._kill_switch_engaged: bool = False

    # ------------------------------------------------------------------
    # Strategy hooks (implement in subclass)
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar, indicators: dict[str, Any]) -> list[OrderIntent]:
        """
        Called on each new OHLCV bar.

        Parameters
        ----------
        bar:
            Completed bar for the strategy's symbol.
        indicators:
            Dict of indicator name → current value (pre-computed by engine).

        Returns
        -------
        list[OrderIntent]
            Zero or more order intentions.  May be empty.
        """
        if not self._enabled:
            return []
        if self._kill_switch_engaged:
            return self._on_kill_switch_bar(bar, indicators)
        return self._on_bar(bar, indicators)

    def on_tick(self, tick: Tick) -> list[OrderIntent]:
        """Called on every tick (optional, override if needed)."""
        if not self._enabled or self._kill_switch_engaged:
            return []
        return self._on_tick(tick)

    def on_fill(self, fill: Fill) -> None:
        """Called when one of this strategy's orders is filled."""
        self._on_fill(fill)

    def on_regime_change(self, new_state: RegimeState, overrides: dict) -> None:
        """
        Called by the engine when the macro regime transitions.

        B10 fix: *overrides* may contain dot-notation keys such as
        ``{"grid.atr_multiplier": 0.8, "sar.enabled": False}``.  Only keys
        whose prefix matches this strategy's ``_config_prefix`` (or that have
        no prefix at all) are merged.  The merged dict uses **flat** keys
        (``"atr_multiplier"``) so that subclass ``_on_regime_change``
        implementations can read them directly from ``self._config``.
        """
        flat: dict = {}
        for k, v in overrides.items():
            if "." in k:
                prefix, leaf = k.split(".", 1)
                # Accept the key only when it belongs to this strategy type
                if self._config_prefix and prefix == self._config_prefix:
                    flat[leaf] = v
                elif not self._config_prefix:
                    # No prefix registered — accept all dot-keys (legacy behaviour)
                    flat[leaf] = v
                # else: belongs to a different strategy — silently skip
            else:
                # Plain key (no dot) — apply to every strategy
                flat[k] = v

        self._config.update(flat)
        self._enabled = bool(self._config.get("enabled", True))
        self._on_regime_change(new_state, flat)

    def on_kill_switch(self) -> None:
        """Engage kill switch — strategy will stop issuing intents."""
        self._kill_switch_engaged = True
        self._on_kill_switch()

    def reset_kill_switch(self) -> None:
        """Operator-reset of kill switch (use cautiously)."""
        self._kill_switch_engaged = False

    # ------------------------------------------------------------------
    # Overridable hooks (default no-ops)
    # ------------------------------------------------------------------

    @abstractmethod
    def _on_bar(self, bar: Bar, indicators: dict[str, Any]) -> list[OrderIntent]:
        ...

    def _on_tick(self, tick: Tick) -> list[OrderIntent]:
        return []

    def _on_fill(self, fill: Fill) -> None:
        pass

    def _on_regime_change(self, new_state: RegimeState, overrides: dict) -> None:
        pass

    def _on_kill_switch(self) -> None:
        pass

    def _on_kill_switch_bar(self, bar: Bar, indicators: dict[str, Any]) -> list[OrderIntent]:
        return []

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill_switch_engaged

    def _make_intent(
        self, bar: Bar, side, qty: int, order_type, price=None, tag: str = ""
    ) -> OrderIntent:
        from engine.core.events import OrderIntent as _OI
        return _OI(
            symbol=bar.symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            price=price,
            strategy_id=self.strategy_id,
            tag=tag,
        )

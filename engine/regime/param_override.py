"""
engine/regime/param_override.py
--------------------------------
Per-regime parameter override registry.

When the regime changes, the ParameterOverrideRegistry returns a merged
parameter dict that strategies use to update their live parameters.

Override keys use dot-notation: ``"grid.atr_multiplier"``, ``"sar.enabled"``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from engine.core.events import RegimeState

logger = logging.getLogger(__name__)


class ParameterOverrideRegistry:
    """
    Stores per-regime parameter overrides and merges them onto base configs.

    Usage::

        registry = ParameterOverrideRegistry.from_config(regime_cfg)
        overrides = registry.get_overrides(RegimeState.BULL)
        # → {"grid.atr_multiplier": 0.8, "grid.max_grid_levels": 6, ...}
    """

    def __init__(self) -> None:
        self._overrides: dict[RegimeState, dict[str, Any]] = {
            s: {} for s in RegimeState
        }

    def register(self, state: RegimeState, key: str, value: Any) -> None:
        self._overrides[state][key] = value

    def get_overrides(self, state: RegimeState) -> dict[str, Any]:
        """Return a copy of the override dict for *state*."""
        return copy.deepcopy(self._overrides.get(state, {}))

    def apply(self, base: dict[str, Any], state: RegimeState) -> dict[str, Any]:
        """
        Merge regime overrides onto *base* config dict.

        Dot-notation keys (e.g. ``"grid.atr_multiplier"``) are expanded into
        nested dict updates.  Returns a new dict; *base* is not mutated.
        """
        result = copy.deepcopy(base)
        overrides = self._overrides.get(state, {})

        for dot_key, val in overrides.items():
            parts = dot_key.split(".")
            target = result
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = val

        if overrides:
            logger.info(
                "ParameterOverrideRegistry: applied %d overrides for regime=%s",
                len(overrides), state.value,
            )

        return result

    @classmethod
    def from_config(cls, regime_cfg: dict) -> "ParameterOverrideRegistry":
        """Build registry from the ``regime.overrides`` config section."""
        registry = cls()
        overrides_cfg = regime_cfg.get("overrides", {})

        for state_name, kvs in overrides_cfg.items():
            try:
                state = RegimeState(state_name)
            except ValueError:
                logger.warning("Unknown regime state in config: %s", state_name)
                continue
            for key, val in kvs.items():
                registry.register(state, key, val)

        return registry

"""
engine/indicators/base.py
--------------------------
Abstract base for all technical indicators.

Design principles
~~~~~~~~~~~~~~~~~
* **Streaming / stateful**: each indicator maintains internal state and
  processes one bar at a time — no future data is ever accessed.
* **Burn-in safety**: ``ready`` property returns ``False`` until the
  minimum number of bars have been seen; strategies must check this before
  using values.
* **Numerical accuracy**: internal accumulators use ``float`` (IEEE-754
  double) which gives sufficient precision for indicator mathematics.
  Trade-level monetary calculations use ``Decimal`` instead (see pnl_engine).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.core.events import Bar


class IIndicator(ABC):
    """
    Streaming technical indicator.

    Subclasses must implement ``_calculate`` and expose ``period``.
    """

    def __init__(self, period: int) -> None:
        if period < 1:
            raise ValueError(f"Indicator period must be >= 1, got {period}")
        self._period = period
        self._bar_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, bar: Bar) -> float:
        """
        Process *bar* and return the current indicator value.

        Returns ``float('nan')`` while the indicator is warming up
        (i.e., ``ready`` is ``False``).
        """
        self._bar_count += 1
        return self._calculate(bar)

    @property
    def period(self) -> int:
        return self._period

    @property
    def ready(self) -> bool:
        """True once enough bars have been seen for a valid reading."""
        return self._bar_count >= self._period

    @abstractmethod
    def value(self) -> float:
        """Return the latest computed indicator value."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (e.g. when switching instruments)."""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @abstractmethod
    def _calculate(self, bar: Bar) -> float:
        """Compute and store the new value from *bar*; return it."""

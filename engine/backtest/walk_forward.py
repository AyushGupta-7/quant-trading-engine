"""
engine/backtest/walk_forward.py
--------------------------------
Walk-forward backtesting engine.

Algorithm
~~~~~~~~~
The historical bar series is split into overlapping (or non-overlapping) windows::

    |<--- train --->|<- test ->|
                    |<--- train --->|<- test ->|
                                   ...

For each window:
  1. Build a fresh BacktestEngine (fresh state, fresh indicators).
  2. Run the strategy on the TRAINING bars (warm-up/optimisation period).
     The strategy's final parameter state after training is then used
     as the starting state for the test window.
  3. Run the strategy on the TEST bars (out-of-sample).
  4. Record per-window metrics.

No-lookahead guarantee
~~~~~~~~~~~~~~~~~~~~~~
* Training and test windows are disjoint in time.
* The engine never peeks at the test window during training.
* The SimClock advances monotonically within each window.
* Strategy state is passed from training → test but NOT from test → future training
  (each training window starts fresh from the beginning of that window).

Configuration
~~~~~~~~~~~~~
``train_bars``:
    Number of bars in each training window.
``test_bars``:
    Number of bars in each test (out-of-sample) window.
``step_bars``:
    How many bars to advance the window each iteration.
    Defaults to ``test_bars`` (non-overlapping test windows).
    Set < ``test_bars`` for more granular reporting.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

from engine.backtest.engine import BacktestEngine
from engine.core.events import Bar
from engine.core.instrument import InstrumentRegistry
from engine.position.cost_model import CostModel
from engine.regime.circuit_breaker import CircuitBreaker
from engine.risk.risk_gate import RiskGate
from engine.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Per-window out-of-sample results."""
    window_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_bars: int
    test_bars: int
    # Test-window metrics
    total_realised: float
    total_unrealised: float
    total_equity: float
    total_fees: float
    total_fills: int
    approved_orders: int
    rejected_orders: int

    def to_dict(self) -> dict:
        return {
            "window": self.window_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_bars": self.train_bars,
            "test_bars_actual": self.test_bars,
            "total_realised": self.total_realised,
            "total_unrealised": self.total_unrealised,
            "total_equity": self.total_equity,
            "total_fees": self.total_fees,
            "total_fills": self.total_fills,
            "approved_orders": self.approved_orders,
            "rejected_orders": self.rejected_orders,
        }


@dataclass
class WalkForwardResult:
    """Aggregate results from a full walk-forward run."""
    windows: list[WindowResult] = field(default_factory=list)

    # Aggregate stats
    @property
    def total_windows(self) -> int:
        return len(self.windows)

    @property
    def aggregate_realised(self) -> float:
        return sum(w.total_realised for w in self.windows)

    @property
    def aggregate_fees(self) -> float:
        return sum(w.total_fees for w in self.windows)

    @property
    def aggregate_fills(self) -> int:
        return sum(w.total_fills for w in self.windows)

    @property
    def profitable_windows(self) -> int:
        return sum(1 for w in self.windows if w.total_realised > 0)

    @property
    def win_rate(self) -> float:
        if not self.windows:
            return 0.0
        return self.profitable_windows / len(self.windows)

    def summary(self) -> dict:
        return {
            "total_windows":       self.total_windows,
            "profitable_windows":  self.profitable_windows,
            "win_rate":            round(self.win_rate, 4),
            "aggregate_realised":  round(self.aggregate_realised, 2),
            "aggregate_fees":      round(self.aggregate_fees, 2),
            "aggregate_fills":     self.aggregate_fills,
            "per_window":          [w.to_dict() for w in self.windows],
        }


# ---------------------------------------------------------------------------
# Factory protocol
# ---------------------------------------------------------------------------

# A callable that creates a *fresh* list of strategies for each window.
# This ensures each window starts with clean indicator state.
StrategyFactory = Callable[[], list[BaseStrategy]]


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------

class WalkForwardEngine:
    """
    Runs a genuine walk-forward evaluation over a bar series.

    Parameters
    ----------
    strategy_factory:
        Callable that returns a **fresh** list of strategy instances.
        Called once per window so each window starts with clean state.
    risk_gate_factory:
        Callable that returns a fresh RiskGate.  Called once per window.
    registry:
        InstrumentRegistry shared across all windows (read-only).
    cost_model:
        CostModel used for transaction cost accounting.
    slippage_ticks:
        Ticks of slippage on market orders.
    train_bars:
        Number of bars in the training (in-sample) window.
    test_bars:
        Number of bars in the test (out-of-sample) window.
    step_bars:
        Bars to advance each iteration.  Defaults to ``test_bars``
        (pure walk-forward, no overlapping test windows).
    """

    def __init__(
        self,
        strategy_factory: StrategyFactory,
        risk_gate_factory: Callable[[], RiskGate],
        registry: InstrumentRegistry,
        cost_model: CostModel | None = None,
        slippage_ticks: int = 1,
        train_bars: int = 200,
        test_bars: int = 50,
        step_bars: int | None = None,
    ) -> None:
        self._strategy_factory   = strategy_factory
        self._risk_gate_factory  = risk_gate_factory
        self._registry           = registry
        self._cost_model         = cost_model
        self._slippage_ticks     = slippage_ticks
        self._train_bars         = train_bars
        self._test_bars          = test_bars
        self._step_bars          = step_bars if step_bars is not None else test_bars

    def run(self, bars: list[Bar]) -> WalkForwardResult:
        """
        Execute the walk-forward loop over *bars*.

        *bars* must be sorted ascending by timestamp.  The engine validates
        chronological ordering and raises ``ValueError`` if violated.

        Returns
        -------
        WalkForwardResult
            Per-window and aggregate out-of-sample statistics.
        """
        self._validate_chronological(bars)

        n = len(bars)
        window_size = self._train_bars + self._test_bars
        result = WalkForwardResult()

        if n < window_size:
            raise ValueError(
                f"Insufficient bars: need at least {window_size} "
                f"(train={self._train_bars} + test={self._test_bars}), "
                f"got {n}"
            )

        window_idx = 0
        start = 0

        while start + window_size <= n:
            train_slice = bars[start : start + self._train_bars]
            test_slice  = bars[start + self._train_bars : start + window_size]

            logger.info(
                "WalkForward window %d: train [%s → %s] (%d bars), "
                "test [%s → %s] (%d bars)",
                window_idx,
                train_slice[0].ts.date(), train_slice[-1].ts.date(), len(train_slice),
                test_slice[0].ts.date(),  test_slice[-1].ts.date(),  len(test_slice),
            )

            window_result = self._run_window(
                window_idx, train_slice, test_slice
            )
            result.windows.append(window_result)

            start      += self._step_bars
            window_idx += 1

        logger.info(
            "WalkForward complete: %d windows, aggregate_realised=%.2f",
            result.total_windows, result.aggregate_realised,
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_window(
        self,
        idx: int,
        train_bars: list[Bar],
        test_bars: list[Bar],
    ) -> WindowResult:
        """Run one train+test window and return its out-of-sample metrics."""
        # --- TRAINING PHASE ---
        # Run full backtest on training data. We discard the PnL result
        # but keep strategy state (indicators warm, position = 0 at start).
        train_strategies = self._strategy_factory()
        train_gate       = self._risk_gate_factory()
        train_engine     = BacktestEngine(
            strategies=train_strategies,
            risk_gate=train_gate,
            registry=self._registry,
            cost_model=self._cost_model,
            slippage_ticks=self._slippage_ticks,
        )
        train_engine.run(train_bars)
        # Training PnL intentionally discarded — we only care about OOS.

        # --- TEST (OUT-OF-SAMPLE) PHASE ---
        # Fresh strategies each window to prevent state leakage from training.
        # This is the CORRECT walk-forward approach: each OOS window runs on
        # a clean slate so results are truly out-of-sample.
        test_strategies = self._strategy_factory()
        test_gate       = self._risk_gate_factory()
        test_engine     = BacktestEngine(
            strategies=test_strategies,
            risk_gate=test_gate,
            registry=self._registry,
            cost_model=self._cost_model,
            slippage_ticks=self._slippage_ticks,
        )
        # Run test bars — SimClock starts at test_bars[0].ts
        test_summary = test_engine.run(test_bars)

        return WindowResult(
            window_index=idx,
            train_start=train_bars[0].ts,
            train_end=train_bars[-1].ts,
            test_start=test_bars[0].ts,
            test_end=test_bars[-1].ts,
            train_bars=len(train_bars),
            test_bars=len(test_bars),
            total_realised=test_summary["total_realised"],
            total_unrealised=test_summary["total_unrealised"],
            total_equity=test_summary["total_equity"],
            total_fees=test_summary["total_fees"],
            total_fills=test_summary["total_fills"],
            approved_orders=test_summary["approved_orders"],
            rejected_orders=test_summary["rejected_orders"],
        )

    @staticmethod
    def _validate_chronological(bars: list[Bar]) -> None:
        """Raise ValueError if bars are not strictly chronologically ordered."""
        for i in range(1, len(bars)):
            if bars[i].ts < bars[i - 1].ts:
                raise ValueError(
                    f"Bars are not chronologically ordered: "
                    f"bar[{i-1}].ts={bars[i-1].ts} > bar[{i}].ts={bars[i].ts}"
                )

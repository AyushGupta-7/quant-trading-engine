"""tests/integration/test_walk_forward.py

Integration tests for the WalkForwardEngine.

Proofs:
1. Chronological ordering is validated — reversed bars raise ValueError.
2. Window slicing is correct (train_start/end, test_start/end).
3. No future data is visible during any training window.
4. Multiple windows are produced and metrics are non-negative.
5. Aggregate results are consistent with per-window results.
6. Configurable step_bars produces correct window count.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from engine.backtest.walk_forward import WalkForwardEngine, WalkForwardResult
from engine.core.events import Bar
from engine.core.instrument import AssetClass, Exchange, Instrument, InstrumentRegistry
from engine.risk.drawdown_guard import DrawdownGuard
from engine.risk.kill_switch import KillSwitch
from engine.risk.position_cap import PositionCapGuard
from engine.risk.risk_gate import RiskGate
from engine.strategy.sar_engine import SAREngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(symbol: str, i: int, close: float = 6000.0) -> Bar:
    ts = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
    p = Decimal(str(close))
    return Bar(symbol=symbol, open=p, high=p + 10, low=p - 10, close=p, volume=1000, ts=ts)


def _make_bars(n: int, symbol: str = "CRUDEOIL", start_close: float = 6000.0,
               trend: float = 1.0) -> list[Bar]:
    """Generate n bars with optional linear price trend."""
    return [_make_bar(symbol, i, close=start_close + i * trend) for i in range(n)]


def _strategy_factory():
    return [SAREngine("sar-test", {
        "atr_period": 5,
        "atr_stop_multiplier": 2.0,
        "pyramid_on_trend": True,
        "max_pyramid_levels": 2,
        "position_cap_lots": 5,
        "order_type": "MARKET",
        "symbols": ["CRUDEOIL"],
    })]


def _risk_gate_factory(tmp_path, suffix=""):
    ks = KillSwitch(str(tmp_path / f"ks_{suffix}.lock"))
    return RiskGate(
        kill_switch=ks,
        drawdown_guard=DrawdownGuard(max_drawdown_pct=1.0, max_daily_loss=1e9),
        position_cap=PositionCapGuard(max_position_lots=10, max_open_orders=200),
    )


def _make_registry() -> InstrumentRegistry:
    reg = InstrumentRegistry()
    reg.register(Instrument(
        symbol="CRUDEOIL", exchange=Exchange.MCX,
        asset_class=AssetClass.COMMODITY_FUTURES,
        lot_size=100, tick_size=Decimal("1.0"),
        margin_pct=Decimal("0.05"),
    ))
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWalkForwardChronology:
    """Verify that chronological ordering guarantees are enforced."""

    def test_reversed_bars_raise_value_error(self, tmp_path):
        bars = _make_bars(100)
        bars_reversed = bars[::-1]  # reverse order = future before past

        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "rev"),
            registry=reg,
            train_bars=50,
            test_bars=20,
        )
        with pytest.raises(ValueError, match="chronologically"):
            wf.run(bars_reversed)

    def test_out_of_order_bar_raises(self, tmp_path):
        bars = _make_bars(100)
        # Swap two adjacent bars to create a local disorder
        bars[50], bars[49] = bars[49], bars[50]

        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "oop"),
            registry=reg,
            train_bars=40,
            test_bars=20,
        )
        with pytest.raises(ValueError):
            wf.run(bars)

    def test_correctly_ordered_bars_succeed(self, tmp_path):
        bars = _make_bars(100)
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "ok"),
            registry=reg,
            train_bars=50,
            test_bars=30,
        )
        result = wf.run(bars)
        assert result.total_windows >= 1


class TestWalkForwardWindowing:
    """Verify window slicing produces correct date ranges."""

    def test_single_window_timestamps(self, tmp_path):
        bars = _make_bars(100)
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "sw"),
            registry=reg,
            train_bars=60,
            test_bars=30,
        )
        result = wf.run(bars)
        assert result.total_windows >= 1

        w = result.windows[0]
        # Train window is first train_bars bars
        assert w.train_start == bars[0].ts
        assert w.train_end == bars[59].ts
        # Test window immediately follows
        assert w.test_start == bars[60].ts
        assert w.test_end == bars[89].ts
        assert w.train_bars == 60
        assert w.test_bars == 30

    def test_no_overlap_between_train_and_test(self, tmp_path):
        bars = _make_bars(200)
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "no"),
            registry=reg,
            train_bars=80,
            test_bars=40,
        )
        result = wf.run(bars)
        for w in result.windows:
            # Test must start strictly AFTER train ends
            assert w.test_start > w.train_end, (
                f"Window {w.window_index}: test_start={w.test_start} "
                f"must be after train_end={w.train_end}"
            )

    def test_windows_are_non_overlapping_by_default(self, tmp_path):
        bars = _make_bars(300)
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "non"),
            registry=reg,
            train_bars=100,
            test_bars=50,
            step_bars=50,   # default: step == test_bars → non-overlapping test windows
        )
        result = wf.run(bars)
        assert result.total_windows >= 2
        for i in range(1, len(result.windows)):
            prev = result.windows[i - 1]
            curr = result.windows[i]
            # Current test start must be AFTER previous test end
            assert curr.test_start > prev.test_end, (
                f"Test windows {i-1} and {i} overlap in time"
            )

    def test_step_bars_controls_window_count(self, tmp_path):
        bars = _make_bars(200)
        reg = _make_registry()

        # step=test_bars → ceil((200-100-50)/50)+1 windows
        wf_default = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "sc1"),
            registry=reg,
            train_bars=100,
            test_bars=50,
            step_bars=50,
        )
        r_default = wf_default.run(bars)

        # step=25 → more windows
        wf_dense = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "sc2"),
            registry=reg,
            train_bars=100,
            test_bars=50,
            step_bars=25,
        )
        r_dense = wf_dense.run(bars)

        assert r_dense.total_windows > r_default.total_windows, (
            "Smaller step_bars must produce more windows"
        )

    def test_insufficient_bars_raises(self, tmp_path):
        bars = _make_bars(50)  # need 80+20=100 bars
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "ins"),
            registry=reg,
            train_bars=80,
            test_bars=20,
        )
        with pytest.raises(ValueError, match="Insufficient"):
            wf.run(bars)


class TestWalkForwardNoLookahead:
    """Prove that test data is never visible during training."""

    def test_test_bars_timestamps_always_after_train(self, tmp_path):
        """Every single test bar timestamp must be strictly after every train bar."""
        bars = _make_bars(300)
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "la"),
            registry=reg,
            train_bars=150,
            test_bars=50,
        )
        result = wf.run(bars)
        for w in result.windows:
            # All train bars precede all test bars
            assert w.train_end < w.test_start, (
                f"Window {w.window_index}: train_end={w.train_end} "
                f">= test_start={w.test_start} — LOOKAHEAD!"
            )


class TestWalkForwardResults:
    """Verify output structure and aggregate calculations."""

    def test_metrics_are_non_negative(self, tmp_path):
        bars = _make_bars(200)
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "mn"),
            registry=reg,
            train_bars=100,
            test_bars=50,
        )
        result = wf.run(bars)
        for w in result.windows:
            assert w.total_fills >= 0
            assert w.total_fees >= 0.0
            assert w.approved_orders >= 0
            assert w.rejected_orders >= 0

    def test_aggregate_fills_sum(self, tmp_path):
        bars = _make_bars(300)
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "af"),
            registry=reg,
            train_bars=100,
            test_bars=50,
        )
        result = wf.run(bars)
        assert result.aggregate_fills == sum(w.total_fills for w in result.windows)

    def test_win_rate_between_0_and_1(self, tmp_path):
        bars = _make_bars(300)
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "wr"),
            registry=reg,
            train_bars=100,
            test_bars=50,
        )
        result = wf.run(bars)
        assert 0.0 <= result.win_rate <= 1.0

    def test_summary_dict_contains_all_keys(self, tmp_path):
        bars = _make_bars(200)
        reg = _make_registry()
        wf = WalkForwardEngine(
            strategy_factory=_strategy_factory,
            risk_gate_factory=lambda: _risk_gate_factory(tmp_path, "sk"),
            registry=reg,
            train_bars=100,
            test_bars=50,
        )
        result = wf.run(bars)
        summary = result.summary()
        for key in ("total_windows", "profitable_windows", "win_rate",
                    "aggregate_realised", "aggregate_fees", "per_window"):
            assert key in summary, f"Missing key in summary: {key}"

"""tests/integration/test_regime_integration.py

End-to-end integration tests for the regime pipeline.

Verified flows:
1. MacroSnapshot → RegimeScorer → RegimeState transition
2. RegimeState → ParameterOverrideRegistry → strategy.on_regime_change
3. Grid dot-notation overrides actually reach GridEngine._atr_mult
4. SAR dot-notation overrides actually reach SAREngine._stop_mult
5. Regime change does NOT affect strategies when no overrides configured
6. CircuitBreaker engages kill switch → RiskGate blocks orders → strategies halt
7. BacktestEngine runs with regime_engine wired (end-to-end bar loop)
8. Regime change happens mid-backtest and strategies receive it
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from engine.backtest.engine import BacktestEngine
from engine.core.events import Bar, OrderType, RegimeState, Side
from engine.core.instrument import AssetClass, Exchange, Instrument, InstrumentRegistry
from engine.regime.circuit_breaker import CircuitBreaker
from engine.regime.macro_ingester import MacroSnapshot, MockMacroIngester
from engine.regime.param_override import ParameterOverrideRegistry
from engine.regime.regime_engine import RegimeEngine
from engine.regime.regime_scorer import RegimeScorer
from engine.risk.drawdown_guard import DrawdownGuard
from engine.risk.kill_switch import KillSwitch
from engine.risk.position_cap import PositionCapGuard
from engine.risk.risk_gate import RiskGate
from engine.strategy.grid_engine import GridEngine
from engine.strategy.sar_engine import SAREngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(i: int, close: float = 6000.0, symbol: str = "CRUDEOIL") -> Bar:
    ts = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
    p = Decimal(str(close))
    return Bar(symbol=symbol, open=p, high=p + 5, low=p - 5, close=p, volume=1000, ts=ts)


def _make_registry() -> InstrumentRegistry:
    reg = InstrumentRegistry()
    reg.register(Instrument(
        symbol="CRUDEOIL", exchange=Exchange.MCX,
        asset_class=AssetClass.COMMODITY_FUTURES,
        lot_size=100, tick_size=Decimal("1.0"),
        margin_pct=Decimal("0.05"),
    ))
    return reg


def _make_gate(tmp_path, suffix=""):
    ks = KillSwitch(str(tmp_path / f"ks_{suffix}.lock"))
    return RiskGate(
        kill_switch=ks,
        drawdown_guard=DrawdownGuard(max_drawdown_pct=1.0, max_daily_loss=1e9),
        position_cap=PositionCapGuard(max_position_lots=20, max_open_orders=200),
    )


def _grid_strategy() -> GridEngine:
    return GridEngine("grid-1", {
        "atr_period": 5,
        "atr_multiplier": 1.0,
        "max_grid_levels": 3,
        "max_pyramid_levels": 5,
        "position_cap_lots": 20,
        "order_type": "LIMIT",
    })


def _sar_strategy() -> SAREngine:
    return SAREngine("sar-1", {
        "atr_period": 5,
        "atr_stop_multiplier": 2.0,
        "pyramid_on_trend": True,
        "max_pyramid_levels": 3,
        "position_cap_lots": 10,
        "order_type": "MARKET",
    })


# ---------------------------------------------------------------------------
# 1. Macro → Scorer → State
# ---------------------------------------------------------------------------

class TestRegimeScorerTransitions:
    """RegimeScorer correctly classifies macro snapshots."""

    def test_bull_when_vix_low_and_fii_positive(self):
        scorer = RegimeScorer({
            "india_vix": {"bull_threshold": 20.0, "bear_threshold": 25.0},
            "usd_inr": {"stress_threshold": 85.0},
            "fii_flow_crore": {"outflow_threshold": -1000.0},
        })
        snap = MacroSnapshot(india_vix=16.0, usd_inr=83.0, fii_flow_crore=500.0)
        assert scorer.score(snap) == RegimeState.BULL

    def test_bear_when_vix_high(self):
        scorer = RegimeScorer({
            "india_vix": {"bull_threshold": 20.0, "bear_threshold": 25.0},
            "usd_inr": {"stress_threshold": 85.0},
            "fii_flow_crore": {"outflow_threshold": -1000.0},
        })
        snap = MacroSnapshot(india_vix=28.0, usd_inr=83.0, fii_flow_crore=100.0)
        assert scorer.score(snap) == RegimeState.BEAR

    def test_crisis_when_vix_and_inr_stressed(self):
        scorer = RegimeScorer({
            "india_vix": {"bull_threshold": 20.0, "bear_threshold": 25.0},
            "usd_inr": {"stress_threshold": 85.0},
            "fii_flow_crore": {"outflow_threshold": -1000.0},
        })
        snap = MacroSnapshot(india_vix=30.0, usd_inr=88.0, fii_flow_crore=-500.0)
        assert scorer.score(snap) == RegimeState.CRISIS

    def test_sideways_otherwise(self):
        scorer = RegimeScorer({
            "india_vix": {"bull_threshold": 20.0, "bear_threshold": 25.0},
            "usd_inr": {"stress_threshold": 85.0},
            "fii_flow_crore": {"outflow_threshold": -1000.0},
        })
        snap = MacroSnapshot(india_vix=22.0, usd_inr=83.0, fii_flow_crore=-200.0)
        assert scorer.score(snap) == RegimeState.SIDEWAYS


# ---------------------------------------------------------------------------
# 2. State → Overrides → Strategy
# ---------------------------------------------------------------------------

class TestOverrideRoutingToStrategies:
    """Dot-notation overrides from ParameterOverrideRegistry reach the right strategy."""

    def test_bull_regime_updates_grid_atr_mult(self):
        grid = _grid_strategy()
        assert grid._atr_mult == 1.0

        registry = ParameterOverrideRegistry()
        registry.register(RegimeState.BULL, "grid.atr_multiplier", 0.5)

        overrides = registry.get_overrides(RegimeState.BULL)
        grid.on_regime_change(RegimeState.BULL, overrides)

        assert grid._atr_mult == 0.5

    def test_bear_regime_updates_sar_stop_mult(self):
        sar = _sar_strategy()
        assert sar._stop_mult == 2.0

        registry = ParameterOverrideRegistry()
        registry.register(RegimeState.BEAR, "sar.atr_stop_multiplier", 3.5)

        overrides = registry.get_overrides(RegimeState.BEAR)
        sar.on_regime_change(RegimeState.BEAR, overrides)

        assert sar._stop_mult == 3.5

    def test_sar_override_does_not_affect_grid(self):
        grid = _grid_strategy()
        registry = ParameterOverrideRegistry()
        registry.register(RegimeState.BEAR, "sar.atr_stop_multiplier", 9.9)

        overrides = registry.get_overrides(RegimeState.BEAR)
        grid.on_regime_change(RegimeState.BEAR, overrides)

        # Grid must be unchanged since the key is SAR-prefixed
        assert grid._atr_mult == 1.0

    def test_plain_key_disables_both_strategies(self):
        grid = _grid_strategy()
        sar  = _sar_strategy()
        registry = ParameterOverrideRegistry()
        registry.register(RegimeState.CRISIS, "enabled", False)

        overrides = registry.get_overrides(RegimeState.CRISIS)
        grid.on_regime_change(RegimeState.CRISIS, overrides)
        sar.on_regime_change(RegimeState.CRISIS, overrides)

        assert not grid.enabled
        assert not sar.enabled

    def test_multiple_strategy_overrides_in_one_dict(self):
        grid = _grid_strategy()
        sar  = _sar_strategy()

        overrides = {
            "grid.atr_multiplier": 0.8,
            "grid.max_grid_levels": 7,
            "sar.atr_stop_multiplier": 3.0,
            "sar.max_pyramid_levels": 5,
        }
        grid.on_regime_change(RegimeState.BULL, overrides)
        sar.on_regime_change(RegimeState.BULL, overrides)

        assert grid._atr_mult == 0.8
        assert grid._max_levels == 7
        assert sar._stop_mult == 3.0
        assert sar._max_pyramid == 5


# ---------------------------------------------------------------------------
# 3. RegimeEngine end-to-end wiring
# ---------------------------------------------------------------------------

class TestRegimeEngineOrchestration:
    """RegimeEngine correctly drives the macro → scorer → strategy pipeline."""

    def _make_bull_ingester(self):
        """Returns a BULL-state macro snapshot every call."""
        cfg = {
            "india_vix": {"mock_value": 16.0, "bull_threshold": 20.0, "bear_threshold": 25.0},
            "usd_inr": {"mock_value": 83.0, "stress_threshold": 85.0},
            "fii_flow_crore": {"mock_value": 500.0, "outflow_threshold": -1000.0},
        }
        return MockMacroIngester(cfg)

    def _make_bear_ingester(self):
        cfg = {
            "india_vix": {"mock_value": 28.0, "bull_threshold": 20.0, "bear_threshold": 25.0},
            "usd_inr": {"mock_value": 83.0, "stress_threshold": 85.0},
            "fii_flow_crore": {"mock_value": 100.0, "outflow_threshold": -1000.0},
        }
        return MockMacroIngester(cfg)

    def _make_scorer(self):
        return RegimeScorer({
            "india_vix": {"bull_threshold": 20.0, "bear_threshold": 25.0},
            "usd_inr": {"stress_threshold": 85.0},
            "fii_flow_crore": {"outflow_threshold": -1000.0},
        })

    def test_regime_engine_transitions_to_bull(self):
        grid = _grid_strategy()
        registry = ParameterOverrideRegistry()
        registry.register(RegimeState.BULL, "grid.atr_multiplier", 0.5)

        re = RegimeEngine(
            ingester=self._make_bull_ingester(),
            scorer=self._make_scorer(),
            override_registry=registry,
            strategies=[grid],
        )
        ts = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
        re.on_bar(ts)  # should transition UNKNOWN → BULL

        assert re.current_state == RegimeState.BULL
        assert grid._atr_mult == 0.5, "Bull override must update grid ATR multiplier"

    def test_regime_engine_no_double_notification(self):
        """If state doesn't change, strategy must NOT be called again."""
        call_count = [0]
        grid = _grid_strategy()
        _orig = grid.on_regime_change
        def _tracked(state, overrides):
            call_count[0] += 1
            _orig(state, overrides)
        grid.on_regime_change = _tracked

        registry = ParameterOverrideRegistry()
        re = RegimeEngine(
            ingester=self._make_bull_ingester(),
            scorer=self._make_scorer(),
            override_registry=registry,
            strategies=[grid],
        )
        ts = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
        re.on_bar(ts)  # → BULL, notifies once
        re.on_bar(ts)  # → same BULL, must NOT notify again
        re.on_bar(ts)  # → same BULL, must NOT notify again

        assert call_count[0] == 1, (
            f"Expected 1 notification, got {call_count[0]} (no duplicate calls on stable regime)"
        )

    def test_regime_engine_from_config(self, tmp_path):
        """from_config builds a valid RegimeEngine from a config dict."""
        cfg = {
            "regime": {
                "proxies": {
                    "india_vix": {
                        "mock_value": 16.0,
                        "bull_threshold": 20.0,
                        "bear_threshold": 25.0,
                    },
                    "usd_inr": {"mock_value": 83.0, "stress_threshold": 85.0},
                    "fii_flow_crore": {"mock_value": 500.0, "outflow_threshold": -1000.0},
                },
                "overrides": {
                    "BULL": {"grid.atr_multiplier": 0.7},
                },
                "circuit_breaker": {
                    "daily_loss_limit": 100_000,
                    "max_consecutive_losses": 10,
                },
            }
        }
        grid = _grid_strategy()
        gate = _make_gate(tmp_path, "fc")
        re = RegimeEngine.from_config(cfg, strategies=[grid], risk_gate=gate)

        ts = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
        state = re.on_bar(ts)

        assert state == RegimeState.BULL
        assert grid._atr_mult == 0.7, "Grid must receive BULL override from from_config"

    def test_circuit_breaker_wired_via_regime_engine(self, tmp_path):
        """When CB fires via RegimeEngine, RiskGate kill switch engages."""
        gate = _make_gate(tmp_path, "cbw")
        cb = CircuitBreaker(
            daily_loss_limit=1.0,
            max_consecutive_losses=100,
        )
        re = RegimeEngine(
            ingester=self._make_bull_ingester(),
            scorer=self._make_scorer(),
            override_registry=ParameterOverrideRegistry(),
            strategies=[],
            circuit_breaker=cb,
            risk_gate=gate,
        )
        # Directly trigger breach via CB
        cb.record_pnl(Decimal("-5000"), datetime.now(tz=timezone.utc))

        assert cb.engaged, "CircuitBreaker must engage"
        assert gate.kill_switch.engaged, "RiskGate kill switch must engage"


# ---------------------------------------------------------------------------
# 4. BacktestEngine + RegimeEngine integrated bar loop
# ---------------------------------------------------------------------------

class TestBacktestRegimeIntegration:
    """Regime engine is called each bar inside BacktestEngine.run()."""

    def test_regime_changes_mid_backtest(self, tmp_path):
        """
        Simulate a backtest where the regime changes from BULL to BEAR mid-run.
        Grid strategy should have its atr_mult updated at regime change time.
        """
        # Build a macro ingester that yields BULL for first 5 calls, BEAR thereafter
        # by using a custom ingester backed by a CSV-style list
        class _SequentialIngester:
            def __init__(self, snaps):
                self._snaps = snaps
                self._idx = 0
            def get_snapshot(self, ts=None):
                s = self._snaps[min(self._idx, len(self._snaps) - 1)]
                self._idx += 1
                return s
            def reset(self):
                self._idx = 0

        bull_snap = MacroSnapshot(india_vix=16.0, usd_inr=83.0, fii_flow_crore=500.0)
        bear_snap = MacroSnapshot(india_vix=28.0, usd_inr=83.0, fii_flow_crore=100.0)
        snaps = [bull_snap] * 5 + [bear_snap] * 20

        ingester = _SequentialIngester(snaps)
        scorer = RegimeScorer({
            "india_vix": {"bull_threshold": 20.0, "bear_threshold": 25.0},
            "usd_inr": {"stress_threshold": 85.0},
            "fii_flow_crore": {"outflow_threshold": -1000.0},
        })
        registry = ParameterOverrideRegistry()
        registry.register(RegimeState.BULL, "grid.atr_multiplier", 0.5)
        registry.register(RegimeState.BEAR, "grid.atr_multiplier", 1.5)

        grid = _grid_strategy()
        gate = _make_gate(tmp_path, "mbt")
        re = RegimeEngine(
            ingester=ingester,
            scorer=scorer,
            override_registry=registry,
            strategies=[grid],
        )
        engine = BacktestEngine(
            strategies=[grid],
            risk_gate=gate,
            registry=_make_registry(),
            regime_engine=re,
        )

        bars = [_bar(i) for i in range(25)]
        engine.run(bars)

        # After 25 bars (5 BULL + 20 BEAR), final regime should be BEAR
        assert re.current_state == RegimeState.BEAR
        assert grid._atr_mult == 1.5, (
            f"Grid must have BEAR atr_mult=1.5, got {grid._atr_mult}"
        )

    def test_backtest_runs_without_regime_engine(self, tmp_path):
        """BacktestEngine must work normally when regime_engine=None."""
        grid = _grid_strategy()
        gate = _make_gate(tmp_path, "noreg")
        engine = BacktestEngine(
            strategies=[grid],
            risk_gate=gate,
            registry=_make_registry(),
            regime_engine=None,
        )
        bars = [_bar(i) for i in range(30)]
        summary = engine.run(bars)
        assert summary["bars_processed"] == 30

    def test_circuit_breaker_blocks_trading_in_backtest(self, tmp_path):
        """After CB fires mid-backtest, remaining orders are rejected."""
        from engine.core.events import OrderIntent

        gate = _make_gate(tmp_path, "cbb")
        cb = CircuitBreaker(
            daily_loss_limit=1.0,   # trivially small — will breach immediately
            max_consecutive_losses=100,
        )
        re = RegimeEngine(
            ingester=MockMacroIngester({
                "india_vix": {"mock_value": 16.0, "bull_threshold": 20.0, "bear_threshold": 25.0},
                "usd_inr": {"mock_value": 83.0, "stress_threshold": 85.0},
                "fii_flow_crore": {"mock_value": 500.0, "outflow_threshold": -1000.0},
            }),
            scorer=RegimeScorer({
                "india_vix": {"bull_threshold": 20.0, "bear_threshold": 25.0},
                "usd_inr": {"stress_threshold": 85.0},
                "fii_flow_crore": {"outflow_threshold": -1000.0},
            }),
            override_registry=ParameterOverrideRegistry(),
            strategies=[],
            circuit_breaker=cb,
            risk_gate=gate,
        )

        # Trigger breach BEFORE run starts
        cb.record_pnl(Decimal("-10000"), datetime.now(tz=timezone.utc))
        assert gate.kill_switch.engaged

        grid = _grid_strategy()
        engine = BacktestEngine(
            strategies=[grid],
            risk_gate=gate,
            registry=_make_registry(),
            circuit_breaker=cb,
            regime_engine=re,
        )
        bars = [_bar(i) for i in range(30)]
        summary = engine.run(bars)

        # All orders must be rejected once kill switch is engaged
        assert summary["approved_orders"] == 0, (
            "After kill switch, no orders should be approved"
        )

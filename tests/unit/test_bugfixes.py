"""tests/unit/test_bugfixes.py

Regression tests for the six critical bugs fixed:

  B1 — client_order_id determinism
  B2 — Grid _pending_levels stale level pruning
  B3 — SAR reversal race / pyramid_count negative
  B4 — _strategy_symbols returns None (match-all) instead of {""}
  B9 — CircuitBreaker wired into fill pipeline
  B10 — Regime override dot-keys correctly routed to strategies
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from engine.core.events import (
    Bar, Fill, KillSwitchEvent, OrderIntent, OrderType, RegimeState, Side,
)
from engine.core.instrument import AssetClass, Exchange, Instrument, InstrumentRegistry
from engine.oms.order import intent_to_order, make_client_order_id
from engine.regime.circuit_breaker import CircuitBreaker
from engine.risk.drawdown_guard import DrawdownGuard
from engine.risk.kill_switch import KillSwitch
from engine.risk.position_cap import PositionCapGuard
from engine.risk.risk_gate import RiskGate
from engine.strategy.grid_engine import GridEngine
from engine.strategy.sar_engine import SAREngine
from engine.strategy.base import BaseStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(symbol: str = "CRUDEOIL", close: float = 6000.0, i: int = 0) -> Bar:
    ts = datetime(2024, 1, 2, 9, 15, tzinfo=timezone.utc) + timedelta(minutes=i)
    p = Decimal(str(close))
    return Bar(symbol=symbol, open=p, high=p + 5, low=p - 5, close=p, volume=1000, ts=ts)


def _fill(symbol: str, side: Side, price: float, qty: int = 1, order_id: str = "oid") -> Fill:
    return Fill(
        order_id=order_id,
        broker_order_id="bid",
        symbol=symbol,
        side=side,
        fill_price=Decimal(str(price)),
        fill_qty=qty,
        fees=Decimal("0"),
        ts=datetime.now(tz=timezone.utc),
    )


def _intent(strategy_id: str = "s1", symbol: str = "X",
            side: Side = Side.BUY, price: float | None = None) -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,
        qty=1,
        order_type=OrderType.LIMIT if price is not None else OrderType.MARKET,
        price=Decimal(str(price)) if price is not None else None,
        strategy_id=strategy_id,
        tag="test",
    )


def _grid(extra_cfg: dict | None = None) -> GridEngine:
    cfg = {
        "atr_period": 5,
        "atr_multiplier": 1.0,
        "max_grid_levels": 3,
        "max_pyramid_levels": 5,
        "position_cap_lots": 20,
        "order_type": "LIMIT",
        "symbols": ["CRUDEOIL"],
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    return GridEngine("grid-test", cfg)


def _sar(extra_cfg: dict | None = None) -> SAREngine:
    cfg = {
        "atr_period": 5,
        "atr_stop_multiplier": 2.0,
        "pyramid_on_trend": True,
        "max_pyramid_levels": 3,
        "position_cap_lots": 10,
        "order_type": "MARKET",
        "symbols": ["CRUDEOIL"],
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    return SAREngine("sar-test", cfg)


# Ready indicators (enough warmup)
_INDICATORS_UP = {
    "atr_5": 20.0,
    "supertrend_5": 5800.0,
    "supertrend_5_up": True,
}
_INDICATORS_DOWN = {
    "atr_5": 20.0,
    "supertrend_5": 6200.0,
    "supertrend_5_up": False,
}


# ===========================================================================
# B1 — client_order_id determinism
# ===========================================================================

class TestB1IdempotentOrderId:
    """client_order_id must be the same for two calls with the same intent+bar_ts."""

    def test_same_bar_ts_produces_same_id(self):
        intent = _intent()
        bar_ts_ms = 1_700_000_000_000
        id1 = make_client_order_id(intent, bar_ts_ms)
        id2 = make_client_order_id(intent, bar_ts_ms)
        assert id1 == id2, "Same intent + same bar_ts_ms must produce the same ID"

    def test_different_bar_ts_produces_different_id(self):
        intent = _intent()
        id1 = make_client_order_id(intent, 1_700_000_000_000)
        id2 = make_client_order_id(intent, 1_700_000_060_000)   # 1 minute later
        assert id1 != id2, "Different bar timestamps must produce different IDs"

    def test_different_intents_same_ts_differ(self):
        bar_ts_ms = 1_700_000_000_000
        id_buy  = make_client_order_id(_intent(side=Side.BUY,  price=100.0), bar_ts_ms)
        id_sell = make_client_order_id(_intent(side=Side.SELL, price=100.0), bar_ts_ms)
        assert id_buy != id_sell

    def test_different_prices_same_ts_differ(self):
        bar_ts_ms = 1_700_000_000_000
        id1 = make_client_order_id(_intent(price=100.0), bar_ts_ms)
        id2 = make_client_order_id(_intent(price=101.0), bar_ts_ms)
        assert id1 != id2

    def test_intent_to_order_uses_bar_ts(self):
        intent = _intent()
        bar_ts_ms = 1_700_000_000_000
        o1 = intent_to_order(intent, bar_ts_ms)
        o2 = intent_to_order(intent, bar_ts_ms)
        assert o1.client_order_id == o2.client_order_id

    def test_market_and_limit_differ(self):
        """MARKET and LIMIT intents for the same symbol/side should have different IDs."""
        bar_ts_ms = 1_700_000_000_000
        market = OrderIntent("X", Side.BUY, 1, OrderType.MARKET, None, "s", "t")
        limit  = OrderIntent("X", Side.BUY, 1, OrderType.LIMIT, Decimal("100"), "s", "t")
        assert make_client_order_id(market, bar_ts_ms) != make_client_order_id(limit, bar_ts_ms)


# ===========================================================================
# B2 — Grid _pending_levels lifecycle
# ===========================================================================

class TestB2GridPendingLevels:
    """Stale grid levels must not accumulate indefinitely."""

    def test_pending_levels_cleared_on_fill(self):
        """After a fill that moves the anchor, _pending_levels must be empty."""
        grid = _grid()
        bar = _bar(close=6000.0, i=0)
        ind = {"atr_5": 20.0}
        grid.on_bar(bar, ind)
        assert len(grid.pending_levels) > 0

        # Simulate a fill at the buy level
        assert grid.anchor is not None, "anchor must be set after on_bar with valid ATR"
        buy_price = float(grid.anchor) - 20.0
        fill = _fill("CRUDEOIL", Side.BUY, buy_price)
        grid.on_fill(fill)
        # After fill the anchor moves and all levels should be cleared
        assert len(grid.pending_levels) == 0, (
            "pending_levels must be cleared when anchor moves after a fill"
        )

    def test_stale_levels_pruned_on_bar(self):
        """Levels far from the current anchor are pruned on the next bar."""
        grid = _grid()
        # First bar establishes anchor and levels
        bar0 = _bar(close=6000.0, i=0)
        ind = {"atr_5": 20.0}
        grid.on_bar(bar0, ind)
        assert len(grid.pending_levels) > 0

        # Move anchor manually far away (simulates price jumping)
        grid._anchor = Decimal("10000")

        # Second bar triggers _prune_stale_levels
        bar1 = _bar(close=10000.0, i=1)
        grid.on_bar(bar1, ind)

        # All old levels (around 6000) are > max_levels spacings from anchor (10000)
        for price in grid.pending_levels:
            dist = abs(price - grid._anchor)
            assert dist <= (grid._max_levels + 1) * grid._grid_spacing, (
                f"Stale level at {price} was not pruned (anchor={grid._anchor})"
            )

    def test_no_unbounded_growth_over_many_bars(self):
        """Run 100 bars with fills; pending_levels must stay bounded."""
        grid = _grid()
        ind = {"atr_5": 20.0}
        close = 6000.0
        for i in range(100):
            bar = _bar(close=close, i=i)
            grid.on_bar(bar, ind)
            # Simulate a fill at a level (to trigger anchor move)
            if i % 5 == 0 and grid.pending_levels:
                p, s = next(iter(grid.pending_levels.items()))
                fill = _fill("CRUDEOIL", s, float(p))
                grid.on_fill(fill)
            close += 10.0

        max_possible = 2 * grid._max_levels
        assert len(grid.pending_levels) <= max_possible, (
            f"pending_levels grew to {len(grid.pending_levels)} > {max_possible}"
        )


# ===========================================================================
# B3 — SAR reversal race / pyramid_count
# ===========================================================================

class TestB3SARReversal:
    """Reversal pending flag prevents duplicate orders; pyramid_count stays >= 0."""

    def test_reversal_pending_blocks_second_reversal(self):
        """While a reversal close order is in-flight, no new reversal is issued."""
        sar = _sar()
        bar0 = _bar(close=6000.0, i=0)

        # Initial entry (uptrend)
        sar.on_bar(bar0, _INDICATORS_UP)
        sar.on_fill(_fill("CRUDEOIL", Side.BUY, 6000.0))

        # Trend flips — first reversal
        bar1 = _bar(close=5900.0, i=1)
        intents1 = sar.on_bar(bar1, _INDICATORS_DOWN)
        reversal_tags = [i.tag for i in intents1]
        assert "sar_close" in reversal_tags, "Reversal must produce a close intent"
        assert sar._reversal_pending, "_reversal_pending must be True after reversal"

        # Next bar still shows downtrend — must NOT produce another reversal
        bar2 = _bar(close=5880.0, i=2)
        intents2 = sar.on_bar(bar2, _INDICATORS_DOWN)
        assert intents2 == [], (
            "While reversal is pending, no new intents should be issued"
        )

    def test_pyramid_count_never_negative(self):
        """pyramid_count must remain >= 0 through reversal + fill cycle."""
        sar = _sar()

        # Build up position: 2 long lots
        bar0 = _bar(close=6000.0, i=0)
        sar.on_bar(bar0, _INDICATORS_UP)
        sar.on_fill(_fill("CRUDEOIL", Side.BUY, 6000.0))

        bar1 = _bar(close=6010.0, i=1)
        sar.on_bar(bar1, _INDICATORS_UP)
        sar.on_fill(_fill("CRUDEOIL", Side.BUY, 6010.0))

        assert sar._pyramid_count == 2
        assert sar._net_lots == 2

        # Trigger reversal (trend flips)
        bar2 = _bar(close=5900.0, i=2)
        sar.on_bar(bar2, _INDICATORS_DOWN)

        # Close fill arrives (SELL 2 lots)
        close_fill = _fill("CRUDEOIL", Side.SELL, 5900.0, qty=2)
        sar.on_fill(close_fill)

        assert sar._pyramid_count >= 0, (
            f"pyramid_count must not be negative, got {sar._pyramid_count}"
        )
        assert not sar._reversal_pending, "_reversal_pending must clear after close fill"

    def test_reversal_pending_cleared_after_close_fill(self):
        """After the close fill arrives, the strategy resumes normal operation."""
        sar = _sar()
        bar0 = _bar(close=6000.0, i=0)
        sar.on_bar(bar0, _INDICATORS_UP)
        sar.on_fill(_fill("CRUDEOIL", Side.BUY, 6000.0))

        # Reversal
        bar1 = _bar(close=5900.0, i=1)
        sar.on_bar(bar1, _INDICATORS_DOWN)
        assert sar._reversal_pending

        # Close fill (SELL) clears the pending flag
        sar.on_fill(_fill("CRUDEOIL", Side.SELL, 5900.0))
        assert not sar._reversal_pending, "Flag must clear after close fill"


# ===========================================================================
# B4 — _strategy_symbols match-all for unconfigured symbols
# ===========================================================================

class TestB4StrategySymbols:
    """Strategies without explicit symbols must receive all bars (match-all)."""

    def test_no_symbols_config_returns_none(self, tmp_path):
        from engine.backtest.engine import BacktestEngine
        from engine.position.cost_model import CostModel

        ks = KillSwitch(str(tmp_path / "ks_b4a.lock"))
        gate = RiskGate(
            kill_switch=ks,
            drawdown_guard=DrawdownGuard(max_drawdown_pct=1.0, max_daily_loss=1e9),
            position_cap=PositionCapGuard(max_position_lots=20, max_open_orders=200),
        )
        reg = InstrumentRegistry()
        reg.register(Instrument(
            symbol="CRUDEOIL", exchange=Exchange.MCX,
            asset_class=AssetClass.COMMODITY_FUTURES,
            lot_size=100, tick_size=Decimal("1.0"),
            margin_pct=Decimal("0.05"),
        ))
        engine = BacktestEngine(strategies=[], risk_gate=gate, registry=reg)

        # Strategy with NO symbols config → match-all
        grid = _grid(extra_cfg={"symbols": []})
        sym_filter = engine._strategy_symbols(grid)
        assert sym_filter is None, "No symbols config must return None (match-all)"

    def test_with_symbols_returns_set(self, tmp_path):
        from engine.backtest.engine import BacktestEngine

        ks = KillSwitch(str(tmp_path / "ks_b4b.lock"))
        gate = RiskGate(
            kill_switch=ks,
            drawdown_guard=DrawdownGuard(max_drawdown_pct=1.0, max_daily_loss=1e9),
            position_cap=PositionCapGuard(max_position_lots=20, max_open_orders=200),
        )
        reg = InstrumentRegistry()
        reg.register(Instrument(
            symbol="CRUDEOIL", exchange=Exchange.MCX,
            asset_class=AssetClass.COMMODITY_FUTURES,
            lot_size=100, tick_size=Decimal("1.0"),
            margin_pct=Decimal("0.05"),
        ))
        engine = BacktestEngine(strategies=[], risk_gate=gate, registry=reg)

        grid = _grid()  # has symbols=["CRUDEOIL"]
        sym_filter = engine._strategy_symbols(grid)
        assert sym_filter == {"CRUDEOIL"}

    def test_empty_string_not_in_symbol_set(self, tmp_path):
        """The old bug returned {""} which never matched real symbols."""
        from engine.backtest.engine import BacktestEngine

        ks = KillSwitch(str(tmp_path / "ks_b4c.lock"))
        gate = RiskGate(
            kill_switch=ks,
            drawdown_guard=DrawdownGuard(max_drawdown_pct=1.0, max_daily_loss=1e9),
            position_cap=PositionCapGuard(max_position_lots=20, max_open_orders=200),
        )
        reg = InstrumentRegistry()
        reg.register(Instrument(
            symbol="CRUDEOIL", exchange=Exchange.MCX,
            asset_class=AssetClass.COMMODITY_FUTURES,
            lot_size=100, tick_size=Decimal("1.0"),
            margin_pct=Decimal("0.05"),
        ))
        engine = BacktestEngine(strategies=[], risk_gate=gate, registry=reg)

        grid = _grid(extra_cfg={"symbols": []})
        sym_filter = engine._strategy_symbols(grid)
        # Must be None (match-all), NOT {""} which matched nothing
        assert sym_filter != {""}, "Bug regression: symbol filter must not be {''}"


# ===========================================================================
# B9 — CircuitBreaker wired into fill pipeline
# ===========================================================================

class TestB9CircuitBreakerWired:
    """CircuitBreaker must block further trading once daily loss limit is breached."""

    def _make_engine_with_cb(self, daily_loss_limit: float, tmp_path=None):
        from engine.backtest.engine import BacktestEngine
        import tempfile, os

        # Use a real temp file so KillSwitch.engage() can write to it
        if tmp_path is not None:
            ks_file = str(tmp_path / "kill_switch_test.lock")
        else:
            # fallback: unique temp name that doesn’t exist yet
            fd, ks_file = tempfile.mkstemp(suffix=".lock")
            os.close(fd)
            os.unlink(ks_file)   # delete so switch starts disengaged

        ks = KillSwitch(ks_file)
        assert not ks.engaged, "KillSwitch must start disengaged"

        gate = RiskGate(
            kill_switch=ks,
            drawdown_guard=DrawdownGuard(max_drawdown_pct=1.0, max_daily_loss=1e9),
            position_cap=PositionCapGuard(max_position_lots=100, max_open_orders=200),
        )
        reg = InstrumentRegistry()
        reg.register(Instrument(
            symbol="CRUDEOIL", exchange=Exchange.MCX,
            asset_class=AssetClass.COMMODITY_FUTURES,
            lot_size=100, tick_size=Decimal("1.0"),
            margin_pct=Decimal("0.05"),
        ))

        # CB fires its callback which engages the kill switch
        cb = CircuitBreaker(
            daily_loss_limit=daily_loss_limit,
            max_consecutive_losses=1000,
            on_breached=lambda e: gate.engage_kill_switch("CB: " + e.reason),
        )

        sar = _sar()
        engine = BacktestEngine(
            strategies=[sar],
            risk_gate=gate,
            registry=reg,
            cost_model=None,
            circuit_breaker=cb,
        )
        return engine, gate, cb

    def test_circuit_breaker_fires_on_daily_loss(self, tmp_path):
        """When cumulative losses exceed limit, CB engages kill switch."""
        engine, gate, cb = self._make_engine_with_cb(daily_loss_limit=1.0,
                                                      tmp_path=tmp_path)
        cb.record_pnl(Decimal("-5000"), datetime.now(tz=timezone.utc))
        assert cb.engaged, "CircuitBreaker must be engaged after loss > limit"

    def test_kill_switch_engages_via_callback(self, tmp_path):
        """The on_breached callback must propagate to RiskGate.kill_switch."""
        engine, gate, cb = self._make_engine_with_cb(daily_loss_limit=1.0,
                                                      tmp_path=tmp_path)

        # Before breach: orders allowed
        intent = OrderIntent("CRUDEOIL", Side.BUY, 1, OrderType.MARKET, None, "s", "")
        approved_before, _ = gate.approve(intent)
        assert approved_before

        # Trigger breach via callback
        cb.record_pnl(Decimal("-5000"), datetime.now(tz=timezone.utc))

        approved_after, reason = gate.approve(intent)
        assert not approved_after, "Kill switch must block orders after CB fires"
        assert "kill switch" in reason.lower()

    def test_circuit_breaker_is_wired_in_backtest_engine(self, tmp_path):
        """BacktestEngine must have a _circuit_breaker attribute (B9 wiring)."""
        engine, gate, cb = self._make_engine_with_cb(daily_loss_limit=50_000,
                                                      tmp_path=tmp_path)
        assert engine._circuit_breaker is cb, (
            "BacktestEngine must store the circuit_breaker reference"
        )

    def test_no_circuit_breaker_does_not_crash(self, tmp_path):
        """BacktestEngine without circuit_breaker must work fine (backward compat)."""
        from engine.backtest.engine import BacktestEngine
        import tempfile, os

        ks_file = str(tmp_path / "ks2.lock")
        ks = KillSwitch(ks_file)
        gate = RiskGate(
            kill_switch=ks,
            drawdown_guard=DrawdownGuard(max_drawdown_pct=1.0, max_daily_loss=1e9),
            position_cap=PositionCapGuard(max_position_lots=20, max_open_orders=200),
        )
        reg = InstrumentRegistry()
        reg.register(Instrument(
            symbol="CRUDEOIL", exchange=Exchange.MCX,
            asset_class=AssetClass.COMMODITY_FUTURES,
            lot_size=100, tick_size=Decimal("1.0"),
            margin_pct=Decimal("0.05"),
        ))
        engine = BacktestEngine(strategies=[], risk_gate=gate, registry=reg)
        assert engine._circuit_breaker is None


# ===========================================================================
# B10 — Regime override dot-key routing
# ===========================================================================

class TestB10RegimeOverrideDotKeys:
    """Dot-notation regime overrides must reach the correct strategy parameter."""

    def test_grid_atr_multiplier_updated_by_regime(self):
        grid = _grid()
        assert grid._atr_mult == 1.0

        # Simulate a BULL regime change with grid-prefixed override
        overrides = {"grid.atr_multiplier": 0.5, "sar.atr_stop_multiplier": 3.0}
        grid.on_regime_change(RegimeState.BULL, overrides)

        assert grid._atr_mult == 0.5, (
            f"atr_mult should be 0.5 after BULL regime override, got {grid._atr_mult}"
        )

    def test_sar_override_ignored_by_grid(self):
        """sar.* overrides must not affect GridEngine."""
        grid = _grid()
        grid.on_regime_change(RegimeState.BEAR, {"sar.atr_stop_multiplier": 9.9})
        # Grid's stop_mult doesn't exist but more importantly its own params unchanged
        assert grid._atr_mult == 1.0  # unchanged

    def test_grid_override_ignored_by_sar(self):
        """grid.* overrides must not affect SAREngine."""
        sar = _sar()
        orig_stop_mult = sar._stop_mult
        sar.on_regime_change(RegimeState.BULL, {"grid.atr_multiplier": 0.1})
        assert sar._stop_mult == orig_stop_mult

    def test_sar_stop_multiplier_updated_by_regime(self):
        sar = _sar()
        sar.on_regime_change(RegimeState.BEAR, {"sar.atr_stop_multiplier": 3.5})
        assert sar._stop_mult == 3.5

    def test_plain_key_applies_to_all_strategies(self):
        """A plain (non-dot) override key must apply to every strategy."""
        grid = _grid()
        sar  = _sar()
        grid.on_regime_change(RegimeState.CRISIS, {"enabled": False})
        sar.on_regime_change(RegimeState.CRISIS,  {"enabled": False})
        assert not grid.enabled
        assert not sar.enabled

    def test_grid_max_levels_updated(self):
        grid = _grid()
        grid.on_regime_change(RegimeState.SIDEWAYS, {"grid.max_grid_levels": 2})
        assert grid._max_levels == 2

    def test_config_prefix_stored(self):
        grid = _grid()
        sar  = _sar()
        assert grid._config_prefix == "grid"
        assert sar._config_prefix == "sar"

    def test_multiple_overrides_applied_atomically(self):
        grid = _grid()
        overrides = {
            "grid.atr_multiplier": 1.5,
            "grid.max_grid_levels": 7,
            "grid.position_cap_lots": 15,
        }
        grid.on_regime_change(RegimeState.BULL, overrides)
        assert grid._atr_mult == 1.5
        assert grid._max_levels == 7
        assert grid._pos_cap == 15

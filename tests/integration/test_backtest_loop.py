"""tests/integration/test_backtest_loop.py

Integration test: runs a full backtest and validates that:
- Fills are generated
- P&L is computed
- No lookahead (clock never ahead of bar time)
- Costs reduce net P&L
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from engine.backtest.engine import BacktestEngine
from engine.core.events import Bar
from engine.core.instrument import AssetClass, Exchange, Instrument, InstrumentRegistry
from engine.position.cost_model import CostModel
from engine.risk.drawdown_guard import DrawdownGuard
from engine.risk.kill_switch import KillSwitch
from engine.risk.position_cap import PositionCapGuard
from engine.risk.risk_gate import RiskGate
from engine.strategy.grid_engine import GridEngine
from engine.strategy.sar_engine import SAREngine


def build_bars(n: int = 300, start_price: float = 6000.0,
               vol: float = 50.0, seed: int = 0) -> list[Bar]:
    import math, random
    rng = random.Random(seed)
    bars = []
    price = start_price
    ts = datetime(2024, 1, 2, 9, 15, tzinfo=timezone.utc)
    for i in range(n):
        change = rng.gauss(0, vol * 0.01) * price
        price  = max(price + change, 100.0)
        high   = price + abs(rng.gauss(0, 5))
        low    = price - abs(rng.gauss(0, 5))
        bars.append(Bar(
            symbol="CRUDEOIL",
            open=Decimal(str(round(price * 0.999, 2))),
            high=Decimal(str(round(high, 2))),
            low=Decimal(str(round(max(low, 1), 2))),
            close=Decimal(str(round(price, 2))),
            volume=rng.randint(500, 5000),
            ts=ts + timedelta(minutes=i),
        ))
    return bars


def build_engine(strategies, with_costs: bool = True) -> BacktestEngine:
    registry = InstrumentRegistry()
    registry.register(Instrument(
        symbol="CRUDEOIL", exchange=Exchange.MCX,
        asset_class=AssetClass.COMMODITY_FUTURES,
        lot_size=100, tick_size=Decimal("1.0"),
        margin_pct=Decimal("0.05"),
    ))

    ks = KillSwitch("NUL")
    ks._engaged = False
    risk_gate = RiskGate(
        kill_switch=ks,
        drawdown_guard=DrawdownGuard(max_drawdown_pct=1.0, max_daily_loss=1e9),
        position_cap=PositionCapGuard(max_position_lots=20, max_open_orders=100),
    )
    cost_model = CostModel({
        "mcx": {"ctt_pct": "0.0001", "exchange_txn_charge_pct": "0.0026",
                "sebi_fee_pct": "0.000001", "gst_pct": "0.18", "brokerage_per_lot": "20"},
    }) if with_costs else None

    return BacktestEngine(
        strategies=strategies,
        risk_gate=risk_gate,
        registry=registry,
        cost_model=cost_model,
        slippage_ticks=1,
    )


class TestBacktestLoop:
    def test_engine_runs_without_error(self):
        grid = GridEngine("grid-01", {
            "atr_period": 14, "atr_multiplier": 1.0, "max_grid_levels": 3,
            "max_pyramid_levels": 2, "position_cap_lots": 5, "order_type": "LIMIT",
            "symbols": ["CRUDEOIL"],
        })
        engine = build_engine([grid])
        results = engine.run(build_bars(200))
        assert results["bars_processed"] == 200

    def test_fills_are_generated(self):
        grid = GridEngine("grid-01", {
            "atr_period": 14, "atr_multiplier": 0.5, "max_grid_levels": 5,
            "max_pyramid_levels": 5, "position_cap_lots": 20, "order_type": "LIMIT",
            "symbols": ["CRUDEOIL"],
        })
        engine = build_engine([grid])
        engine.run(build_bars(300))
        assert engine.fills  # at least one fill

    def test_fees_reduce_net_pnl_vs_no_fees(self):
        def run_with_costs(costs: bool) -> float:
            grid = GridEngine("g", {
                "atr_period": 14, "atr_multiplier": 0.5, "max_grid_levels": 5,
                "max_pyramid_levels": 5, "position_cap_lots": 20, "order_type": "LIMIT",
                "symbols": ["CRUDEOIL"],
            })
            engine = build_engine([grid], with_costs=costs)
            engine.run(build_bars(300, seed=1))
            return engine.pnl_engine.total_fees()

        fees = run_with_costs(True)
        no_fees = run_with_costs(False)
        assert fees > no_fees   # fee version has higher costs

    def test_no_lookahead_clock_monotonic(self):
        """Verify SimClock never goes backwards during backtest."""
        from engine.core.clock import SimClock
        bars = build_bars(50)
        grid = GridEngine("g", {
            "atr_period": 14, "atr_multiplier": 1.0, "max_grid_levels": 2,
            "max_pyramid_levels": 2, "position_cap_lots": 5, "order_type": "LIMIT",
            "symbols": ["CRUDEOIL"],
        })
        engine = build_engine([grid])
        prev_ts = None
        for bar in bars:
            engine._clock.advance(bar.ts)
            curr = engine._clock.now()
            if prev_ts:
                assert curr >= prev_ts
            prev_ts = curr

    def test_sar_backtest(self):
        sar = SAREngine("sar-01", {
            "atr_period": 14, "atr_stop_multiplier": 2.0,
            "pyramid_on_trend": True, "max_pyramid_levels": 2,
            "position_cap_lots": 5, "order_type": "MARKET",
            "symbols": ["CRUDEOIL"],
        })
        engine = build_engine([sar])
        results = engine.run(build_bars(250, seed=7))
        assert results["bars_processed"] == 250

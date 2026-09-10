"""
scripts/run_backtest.py
------------------------
CLI entry point for running a backtest.

Usage::

    python scripts/run_backtest.py --config config/default.yaml --symbol CRUDEOIL --bars 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from engine.backtest.engine import BacktestEngine
from engine.core.events import Bar
from engine.core.instrument import InstrumentRegistry
from engine.observability.blotter import TradeBlotter
from engine.observability.logger import setup_logging
from engine.position.cost_model import CostModel
from engine.risk.risk_gate import RiskGate
from engine.strategy.grid_engine import GridEngine
from engine.strategy.sar_engine import SAREngine


def generate_synthetic_bars(
    symbol: str,
    n: int = 500,
    start_price: float = 6000.0,
    drift: float = 0.0,
    vol: float = 0.002,
    seed: int = 42,
) -> list[Bar]:
    """Generate synthetic OHLCV bars via GBM for backtesting."""
    import math
    import random

    rng = random.Random(seed)
    bars = []
    price = start_price
    start_ts = datetime(2024, 1, 2, 9, 15, 0, tzinfo=timezone.utc)

    dt = 1.0 / (252 * 390)
    mu_dt  = (drift - 0.5 * vol ** 2) * dt
    sig_dt = vol * math.sqrt(dt)

    for i in range(n):
        ts = start_ts + timedelta(minutes=i)
        z    = rng.gauss(0, 1)
        close = price * math.exp(mu_dt + sig_dt * z)
        close = max(close, 0.01)
        high  = close * (1 + abs(rng.gauss(0, 0.001)))
        low   = close * (1 - abs(rng.gauss(0, 0.001)))
        open_ = price

        bars.append(Bar(
            symbol=symbol.upper(),
            open=Decimal(str(round(open_, 2))),
            high=Decimal(str(round(high, 2))),
            low=Decimal(str(round(low, 2))),
            close=Decimal(str(round(close, 2))),
            volume=rng.randint(100, 5000),
            ts=ts,
        ))
        price = close

    return bars


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a backtest")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--symbol", default="CRUDEOIL")
    parser.add_argument("--bars",   type=int, default=500)
    parser.add_argument("--strategy", choices=["grid", "sar", "both"], default="both")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    setup_logging(args.log_level, fmt="console")
    logger = logging.getLogger(__name__)

    # Build instrument registry
    registry = InstrumentRegistry.from_config(cfg["instruments"])

    # Build strategies
    strategies = []
    sym = args.symbol.upper()

    if args.strategy in ("grid", "both"):
        grid_cfg = cfg["strategies"]["grid"]
        grid_cfg["symbols"] = [sym]
        strategies.append(GridEngine("grid-01", grid_cfg))

    if args.strategy in ("sar", "both"):
        sar_cfg = cfg["strategies"]["sar"]
        sar_cfg["symbols"] = [sym]
        strategies.append(SAREngine("sar-01", sar_cfg))

    # Build supporting components
    risk_gate   = RiskGate.from_config(cfg["risk"], switch_file=":memory:")

    # Patch kill switch for backtest (no file persistence)
    from engine.risk.kill_switch import KillSwitch
    risk_gate._ks._path = Path("/dev/null") if sys.platform != "win32" else Path("NUL")

    cost_model  = CostModel.from_config(cfg)
    blotter     = TradeBlotter(csv_path=":memory:", db_path=":memory:")

    # Generate bars
    logger.info("Generating %d synthetic bars for %s", args.bars, sym)
    bars = generate_synthetic_bars(sym, n=args.bars)

    # Run backtest
    engine = BacktestEngine(
        strategies=strategies,
        risk_gate=risk_gate,
        registry=registry,
        cost_model=cost_model,
        slippage_ticks=1,
        blotter=blotter,
    )

    results = engine.run(bars)

    # Print results
    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<28}: {v:>12,.2f}")
        else:
            print(f"  {k:<28}: {v:>12,}")
    print("=" * 55)


if __name__ == "__main__":
    main()

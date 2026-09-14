"""
scripts/run_live.py
--------------------
Async live trading entry point (mock adapters by default).

Usage::

    python scripts/run_live.py --config config/default.yaml
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from engine.core.event_bus import EventBus, Topics
from engine.core.events import Bar, Fill, Tick
from engine.core.instrument import InstrumentRegistry
from engine.data.adapters.mock_feed import SyntheticFeedAdapter
from engine.data.bar_builder import MultiBarBuilder
from engine.data.normaliser import TickNormaliser
from engine.indicators import ATR, EMA, RSI, BollingerBands, Supertrend, VWAP
from engine.observability.blotter import TradeBlotter
from engine.observability.logger import setup_logging
from engine.oms.idempotent_placer import IdempotentPlacer
from engine.oms.state_store import OrderStateStore
from engine.position.cost_model import CostModel
from engine.position.position_book import PositionBook
from engine.position.pnl_engine import PnLEngine
from engine.risk.risk_gate import RiskGate
from engine.strategy.grid_engine import GridEngine
from engine.strategy.sar_engine import SAREngine
from engine.broker.mock_broker import MockBrokerAdapter

logger = logging.getLogger(__name__)

# Global shutdown event
_shutdown = asyncio.Event()


def _handle_signal(sig, frame):
    logger.info("Signal %s received — initiating graceful shutdown", sig)
    _shutdown.set()


async def main_async(config_path: str = "config/default.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["system"].get("log_level", "INFO"), cfg["system"].get("log_format", "console"))



    # Build components
    registry   = InstrumentRegistry.from_config(cfg["instruments"])
    symbols    = [i["symbol"] for i in cfg["instruments"]]
    risk_gate  = RiskGate.from_config(cfg["risk"])
    cost_model = CostModel.from_config(cfg)
    blotter    = TradeBlotter(
        cfg["observability"]["blotter_path"],
        cfg["observability"]["db_path"],
    )
    store      = OrderStateStore(cfg["observability"]["db_path"])
    pos_book   = PositionBook(lot_size_fn=lambda s: registry.get(s).lot_size if s in [i.symbol for i in registry.all()] else 1)
    pnl_engine = PnLEngine(lot_size_fn=lambda s: registry.get(s).lot_size if s in [i.symbol for i in registry.all()] else 1)

    # Mock broker
    broker = MockBrokerAdapter.from_config(cfg["broker"])

    def on_fill(fill: Fill) -> None:
        prev = pos_book.get(fill.symbol)
        pnl_engine.on_fill(fill, prev)
        pos_book.on_fill(fill)
        blotter.record(fill)
        logger.info(
            "FILL sym=%s side=%s qty=%d @%.2f fees=%.2f  equity=%.2f",
            fill.symbol, fill.side.value, fill.fill_qty,
            fill.fill_price, fill.fees, pnl_engine.total_equity(),
        )

    broker._on_fill = on_fill
    await broker.connect()

    placer = IdempotentPlacer(store=store, broker=broker)

    # Strategies
    strategies = []
    grid_cfg = cfg["strategies"]["grid"]
    strategies.append(GridEngine("grid-01", grid_cfg))
    sar_cfg  = cfg["strategies"]["sar"]
    strategies.append(SAREngine("sar-01", sar_cfg))

    # Market data
    normaliser   = TickNormaliser()
    bar_builder  = MultiBarBuilder(
        interval_seconds=cfg["bar_builder"]["interval_seconds"],
    )

    # Per-symbol indicator state
    indicators_state: dict[str, dict] = {sym: {
        "atr_14":       ATR(14),
        "ema_20":       EMA(20),
        "rsi_14":       RSI(14),
        "bb_20":        BollingerBands(20),
        "supertrend_14": Supertrend(14, 3.0),
        "vwap":         VWAP(),
    } for sym in symbols}

    feed = SyntheticFeedAdapter(
        symbols=symbols,
        initial_prices={s: 6000.0 for s in symbols},
        tick_interval_seconds=0.1,
        max_ticks=500,
    )

    from engine.core.live_engine import LiveEngine

    engine = LiveEngine(
        feed=feed,
        broker=broker,
        placer=placer,
        strategies=strategies,
        normaliser=normaliser,
        bar_builder=bar_builder,
        risk_gate=risk_gate,
        indicators_state=indicators_state,
        pos_book=pos_book,
        pnl_engine=pnl_engine,
        blotter=blotter,
        poll_interval_seconds=3.0,
    )
    
    # Wire MockBrokerAdapter to use LiveEngine's fill handler
    broker._on_fill = engine.on_fill

    # Register signal handlers
    def _handle_signal(sig, frame):
        logger.info("Signal %s received — initiating graceful shutdown", sig)
        asyncio.create_task(engine.stop())

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Live engine started.  Symbols=%s  Ctrl-C to stop.", symbols)

    try:
        await engine.start()
        # Feed completion or timeout
        await asyncio.wait_for(engine._process_task, timeout=300)
    except asyncio.TimeoutError:
        logger.info("Feed completed / timeout reached.")
    except asyncio.CancelledError:
        logger.info("Engine cancelled.")
    finally:
        await engine.stop()
        blotter.close()

    summary = pnl_engine.summary()
    logger.info("Session complete: %s", summary)
    print("\n=== SESSION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    asyncio.run(main_async(args.config))


if __name__ == "__main__":
    main()

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

    # Register signal handlers
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

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
    await feed.connect()
    await feed.subscribe(symbols)

    logger.info("Live engine started.  Symbols=%s  Ctrl-C to stop.", symbols)

    async def process_ticks():
        async for raw in feed.tick_stream():
            if _shutdown.is_set():
                break
            try:
                tick = normaliser.normalise(raw)
                broker.update_ltp(tick.symbol, tick.ltp)
                bar = bar_builder.update(tick)
                if bar:
                    await process_bar(bar)
            except Exception:
                logger.exception("Error processing tick")

    async def process_bar(bar: Bar):
        ind_set = indicators_state.get(bar.symbol, {})
        current_inds = {}
        for key, ind in ind_set.items():
            val = ind.update(bar)
            current_inds[key] = val
            if key.startswith("supertrend_") and hasattr(ind, "trend_up"):
                current_inds[f"{key}_up"] = ind.trend_up

        for strategy in strategies:
            syms = set(s.upper() for s in strategy._config.get("symbols", []))
            if bar.symbol.upper() in syms or not syms:
                intents = strategy.on_bar(bar, current_inds)
                for intent in intents:
                    ok, reason = risk_gate.approve(intent)
                    if ok:
                        order = await placer.place(intent)
                        if order:
                            risk_gate.position_cap.increment_open_orders()
                    else:
                        logger.debug("Rejected: %s", reason)

    try:
        await asyncio.wait_for(process_ticks(), timeout=300)
    except asyncio.TimeoutError:
        logger.info("Feed completed / timeout reached.")

    # Shutdown
    _shutdown.set()
    await broker.disconnect()
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

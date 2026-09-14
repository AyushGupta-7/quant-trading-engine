# Quant Trading Engine

A modular, testable Python quant trading engine for MCX and NSE F&O markets.

> **Live trading is fully implemented** using Zerodha Kite Connect, but all broker and market-data adapters are **simulated by default** for safety.
> Real-money live trading has not been validated in a production environment. Use mock adapters for local testing.

---

## Architecture

```
Tick → BarBuilder → IndicatorEngine → MacroRegimeEngine
                                           │
                                    StrategyLayer (Grid / SAR)
                                           │
                                       RiskGate
                                           │
                                     OMS (Idempotent)
                                           │
                                    BrokerAdapter (Mock)
                                           │
                               PositionBook / PnLEngine / CostModel
                                           │
                                     Observability
```

See [`implementation_plan.md`](docs/architecture.md) for the full data-flow diagram.

---

## Project Layout

```
engine/
├── core/          # Clock, EventBus, typed events, Instrument
├── data/          # TickNormaliser, BarBuilder, MockFeedAdapter, ContractMaster
├── indicators/    # ATR, EMA, RSI, MACD, Bollinger, Supertrend, VWAP, OBV …
├── regime/        # MacroIngester, RegimeScorer, ParamOverrideRegistry, CircuitBreaker
├── strategy/      # BaseStrategy, GridEngine, SAREngine
├── risk/          # KillSwitch, DrawdownGuard, PositionCapGuard, RiskGate
├── oms/           # IdempotentPlacer, SQLite StateStore, Reconciler
├── broker/        # IBrokerAdapter, MockBrokerAdapter, KiteBrokerAdapter
├── position/      # PositionBook, PnLEngine, CostModel (MCX CTT / NSE STT)
├── backtest/      # BacktestEngine (SimClock), BarFillModel (no lookahead)
└── observability/ # structlog, TradeBlotter (CSV+SQLite), AlertManager

scripts/
├── run_backtest.py   # CLI backtest runner
└── run_live.py       # Async live execution (mock or Kite Connect)

tests/
├── unit/             # Indicators, BarBuilder, GridEngine, SAREngine, Risk, PnL
├── integration/      # Full backtest loop, regime transitions
└── regression/       # Golden-file tests per strategy
```

---

## Quickstart

### Install

```bash
# Python 3.11+ required
python -m pip install -e ".[dev]"
```

### Run Tests

```bash
python -m pytest tests/ -v
# Expected: 89 passed in ~0.5s
```

### Run a Backtest (CLI)

```bash
python scripts/run_backtest.py --symbol CRUDEOIL --bars 500 --strategy both
```

Output:
```
=======================================================
  BACKTEST RESULTS
=======================================================
  total_realised              :       xxx.xx
  total_unrealised            :       xxx.xx
  total_equity                :       xxx.xx
  total_fees                  :       xxx.xx
  bars_processed              :          500
  total_fills                 :           xx
  approved_orders             :           xx
  rejected_orders             :            x
=======================================================
```

### Run Live Simulation

```bash
python scripts/run_live.py --config config/default.yaml
# Generates synthetic ticks, runs both strategies, prints session summary on Ctrl-C
```

---

## Key Design Principles

| Principle | Implementation |
|---|---|
| **No lookahead** | `SimClock` advances before each bar; fills on next bar's OPEN |
| **Decimal arithmetic** | All price/PnL/fee calculations use `decimal.Decimal` |
| **Idempotent orders** | SHA1 `client_order_id` deduplicates retries in OMS |
| **Crash recovery** | SQLite WAL state store + startup `ReconciliationEngine` |
| **Same code path** | Backtest and live use identical strategy/risk/OMS code |
| **Pluggable broker** | `IBrokerAdapter` ABC — swap mock → Kite in one line |
| **Regime-aware** | `ParameterOverrideRegistry` patches strategy params on regime change |

---

## Strategies

### Grid Engine (`engine/strategy/grid_engine.py`)
- ATR-based grid spacing: `anchor ± n × (ATR × multiplier)`
- Pyramiding: anchor moves on each fill to follow trend
- Position cap: `position_cap_lots` hard limit per symbol
- Regime override: `atr_multiplier`, `max_grid_levels` updated on regime change
- Kill switch: immediately stops issuing new order intents

### SAR Engine (`engine/strategy/sar_engine.py`)
- Supertrend-driven directional entry
- ATR stop: `close ± atr_stop_multiplier × ATR`
- Reversal: closes full position then enters opposite direction in one bar
- Pyramiding: adds lots on each confirmed trend bar up to `max_pyramid_levels`

---

## Risk Controls

| Guard | Trigger | Effect |
|---|---|---|
| `KillSwitch` | Operator or `CircuitBreaker` | All `approve()` calls return False |
| `DrawdownGuard` | DD ≥ `max_drawdown_pct` or daily loss ≥ `max_daily_loss` | Blocks orders |
| `PositionCapGuard` | `abs(net_lots) ≥ max_position_lots` | Blocks buy/sell that would breach |
| `CircuitBreaker` | Daily loss limit or consecutive losses | Fires `KillSwitchEvent` |

---

## Cost Model (Indian Markets)

**MCX Commodity Futures** (per executed lot):
- CTT: 0.01% on sell side
- Exchange charge: 0.0026% of notional
- SEBI fee: 0.0001% of notional
- Brokerage: ₹20 flat
- GST: 18% on brokerage + exchange charges

**NSE F&O** (per executed lot):
- STT: 0.0125% on sell side
- Exchange charge: 0.0019% of notional
- SEBI fee: 0.0001% of notional
- Brokerage: ₹20 flat
- GST: 18% on brokerage + exchange charges

---

## Configuration

All parameters are in [`config/default.yaml`](config/default.yaml).  
Use [`config/test.yaml`](config/test.yaml) for deterministic test overrides.

Key sections:
- `feed.adapter`: `mock` (default) or `kite`
- `broker.adapter`: `mock` (default) or `kite`
- `strategies.grid` / `strategies.sar`: strategy-specific params
- `regime.overrides`: per-regime parameter patches
- `risk`: position caps, drawdown limits

---

## Live Execution & Zerodha Kite Connect

The engine features a fully implemented, asyncio-based live execution pipeline:
* **Live Engine Flow**: Asynchronous processing pipeline `Market Data → BarBuilder → Strategy → RiskGate → OMS`.
* **Background Polling**: Background REST order-status polling and reconciliation ensures terminal order states are safely synced without blocking the tick stream.
* **Resilience**: Supports graceful shutdown, safe crash/restart recovery using the SQLite WAL state store, and deterministic order reconciliation.

### Kite Connect REST (Broker)
`KiteBrokerAdapter` is fully implemented and provides:
* Order placement and active cancellation
* Order status polling and reconciliation
* Authentication via API credentials and daily access tokens
* Idempotent OMS integration using deterministic `client_order_id`

### Kite Connect WebSocket (Feed)
`KiteFeedAdapter` is fully implemented and provides:
* Real-time tick streaming and parsing
* Automatic WebSocket disconnect and reconnect handling
* Automatic restoration of subscriptions after a reconnect
* Bounded `asyncio.Queue` integration with back-pressure handling

### Safe Local Testing vs Live Execution
By default, the engine is configured to use `MockBrokerAdapter` and `SyntheticFeedAdapter` to safely simulate fills and market data without risk.

To activate the real Zerodha Kite Connect integration:
1. Set `feed.adapter: kite` and `broker.adapter: kite` in your configuration file.
2. Provide your real API credentials by setting the `KITE_API_KEY`, `KITE_API_SECRET`, and `KITE_ACCESS_TOKEN` environment variables (see `.env.example`).

*Warning: Real-money live trading performance and profitability have not been validated.*

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit only
python -m pytest tests/unit/ -v

# Integration
python -m pytest tests/integration/ -v

# Regression (golden files)
python -m pytest tests/regression/ -v

# With coverage
python -m pytest tests/ --cov=engine --cov-report=term-missing
```

---

## Observability

- **Structured logs**: JSON-lines via `structlog` (set `log_format: json` in config)
- **Trade blotter**: appended to `data/blotter.csv` and `data/trading.db`
- **Alerts**: drawdown and position utilisation thresholds
- **Kill switch**: persisted to `data/kill_switch.lock` — survives restarts

---

## What's Mocked / Simulated

While the live adapters (Kite) are fully implemented, the following components are used by default for safe local simulation:

| Component | Status |
|---|---|
| Market data feed | `SyntheticFeedAdapter` (GBM) + `CsvFeedAdapter` |
| Broker fills | `MockBrokerAdapter` (Gaussian slippage, configurable latency) |
| Macro proxies | Static values from config or CSV time-series |
| SPAN margin | Formula-based approximation |

---

## Docker Support

The engine includes a production-ready `Dockerfile` to easily run the project in an isolated container without affecting your local environment.
By default, the Docker setup uses the safe simulation mode with mock adapters, meaning it does **not** contain or require real trading credentials.

### Build the Image
```bash
docker build -t quant-trading-engine .
```

### Run Tests in Docker
```bash
docker run --rm quant-trading-engine python -m pytest tests/
```

### Run Backtest in Docker
```bash
docker run --rm quant-trading-engine qte-backtest
```

### Run Live Simulation in Docker
```bash
docker run --rm quant-trading-engine qte-live
```

### Docker Compose

You can also use Docker Compose for a simpler reproducible workflow. The Compose stack includes an optional **Redis** service acting as a lightweight event distribution layer for live events (ticks, fills). *Note: Redis does NOT replace the SQLite OMS persistence, which remains fully intact.*

```bash
# Build the image
docker compose build

# Start the stack (Engine + Redis) in the background
docker compose up -d

# Check logs
docker compose logs -f

# Stop the stack
docker compose down
```

The engine gracefully handles Redis unavailability, meaning you can still run `docker compose run --rm engine qte-live` even if Redis isn't running.

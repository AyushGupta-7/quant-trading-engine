"""
engine/core/instrument.py
-------------------------
Immutable instrument specification dataclass.

An Instrument captures the static properties of a tradeable contract:
exchange rules, lot size, tick size, cost regime.  All dynamic state
(positions, orders) lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum


class Exchange(Enum):
    MCX = "MCX"
    NSE = "NSE"
    BSE = "BSE"


class AssetClass(Enum):
    COMMODITY_FUTURES = "commodity_futures"
    INDEX_FUTURES = "index_futures"
    EQUITY_FUTURES = "equity_futures"
    EQUITY_OPTIONS = "equity_options"
    CURRENCY_FUTURES = "currency_futures"


@dataclass(frozen=True)
class Instrument:
    """
    Describes a single tradeable instrument.

    All price arithmetic (tick rounding, lot value) uses Decimal to avoid
    floating-point drift.
    """
    symbol: str                   # e.g. "CRUDEOIL", "GOLD", "NIFTY"
    exchange: Exchange
    asset_class: AssetClass
    lot_size: int                 # contracts per lot
    tick_size: Decimal            # minimum price movement
    currency: str = "INR"
    margin_pct: Decimal = Decimal("0.10")  # approx SPAN margin fraction

    # Optional fields for F&O
    expiry: str | None = None     # "YYYY-MM-DD"
    strike: Decimal | None = None
    option_type: str | None = None  # "CE" | "PE"

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {self.lot_size}")
        if self.tick_size <= Decimal(0):
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def round_to_tick(self, price: Decimal) -> Decimal:
        """Round *price* to the nearest valid tick."""
        ticks = (price / self.tick_size).to_integral_value()
        return ticks * self.tick_size

    def lot_value(self, price: Decimal) -> Decimal:
        """Notional value of one lot at *price*."""
        return price * self.lot_size

    def margin_required(self, price: Decimal, lots: int) -> Decimal:
        """Approximate initial margin for *lots* at *price*."""
        return self.lot_value(price) * self.margin_pct * lots

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @property
    def full_symbol(self) -> str:
        """e.g. MCX:CRUDEOIL or NSE:NIFTY25JANFUT"""
        base = f"{self.exchange.value}:{self.symbol}"
        if self.expiry:
            base += f":{self.expiry}"
        return base


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class InstrumentRegistry:
    """
    In-memory registry loaded from config at startup.

    Lookup by symbol (case-insensitive).  The contract master module extends
    this with expiry-aware instruments loaded from MCX/NSE master files.
    """

    def __init__(self) -> None:
        self._store: dict[str, Instrument] = {}

    def register(self, instrument: Instrument) -> None:
        key = instrument.symbol.upper()
        self._store[key] = instrument

    def get(self, symbol: str) -> Instrument:
        key = symbol.upper()
        if key not in self._store:
            raise KeyError(f"Instrument not found: {symbol!r}")
        return self._store[key]

    def all(self) -> list[Instrument]:
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    @classmethod
    def from_config(cls, instruments_cfg: list[dict]) -> "InstrumentRegistry":
        """Build registry from the instruments section of config YAML."""
        registry = cls()
        for cfg in instruments_cfg:
            instrument = Instrument(
                symbol=cfg["symbol"],
                exchange=Exchange(cfg["exchange"]),
                asset_class=AssetClass(cfg["asset_class"]),
                lot_size=int(cfg["lot_size"]),
                tick_size=Decimal(str(cfg["tick_size"])),
                currency=cfg.get("currency", "INR"),
                margin_pct=Decimal(str(cfg.get("margin_pct", "0.10"))),
            )
            registry.register(instrument)
        return registry

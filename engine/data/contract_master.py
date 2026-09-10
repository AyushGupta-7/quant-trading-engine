"""
engine/data/contract_master.py
-------------------------------
MCX and NSE contract specifications + expiry/rollover logic.

In a live system this module would fetch the contract master CSV from the
exchange or from Kite Connect's ``/instruments`` API.  In the mock/MVP, it
uses:
  1. A hardcoded table of common MCX commodities and NSE F&O instruments.
  2. Simple deterministic expiry calculation (last Thursday of expiry month
     for NSE F&O; last calendar day for MCX).

Design
~~~~~~
* ``ContractMaster`` is the single source of truth for instrument specs.
* It extends ``InstrumentRegistry`` so that strategies can look up contracts
  by symbol alone; the master selects the front-month contract automatically.
* ``rollover()`` returns the symbol string for the next-expiry contract.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal

from engine.core.instrument import AssetClass, Exchange, Instrument, InstrumentRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded MCX commodity specs (lot size, tick size, margin)
# ---------------------------------------------------------------------------

_MCX_SPECS: list[dict] = [
    dict(symbol="CRUDEOIL",  lot_size=100,  tick_size="1.00",  margin_pct="0.05"),
    dict(symbol="GOLD",      lot_size=100,  tick_size="1.00",  margin_pct="0.04"),
    dict(symbol="SILVER",    lot_size=30,   tick_size="1.00",  margin_pct="0.04"),
    dict(symbol="COPPER",    lot_size=2500, tick_size="0.05",  margin_pct="0.05"),
    dict(symbol="NATURALGAS",lot_size=1250, tick_size="0.10",  margin_pct="0.06"),
    dict(symbol="ALUMINIUM", lot_size=5000, tick_size="0.05",  margin_pct="0.05"),
    dict(symbol="ZINC",      lot_size=5000, tick_size="0.05",  margin_pct="0.05"),
]

_NSE_FO_SPECS: list[dict] = [
    dict(symbol="NIFTY",     lot_size=25,   tick_size="0.05",  margin_pct="0.10"),
    dict(symbol="BANKNIFTY", lot_size=15,   tick_size="0.05",  margin_pct="0.12"),
    dict(symbol="FINNIFTY",  lot_size=40,   tick_size="0.05",  margin_pct="0.11"),
]


def _last_thursday(year: int, month: int) -> date:
    """Return the last Thursday of *month*/*year* (NSE F&O expiry rule)."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    # weekday(): Monday=0 … Thursday=3
    offset = (d.weekday() - 3) % 7
    return d - timedelta(days=offset)


def _last_biz_day(year: int, month: int) -> date:
    """Return the last working day of *month*/*year* (simplified MCX rule)."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    # Roll back if weekend
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


class ContractMaster:
    """
    Resolves trading symbols to ``Instrument`` objects.

    Usage::

        master = ContractMaster()
        instr = master.get_front_month("CRUDEOIL", as_of=date.today())
    """

    def __init__(self) -> None:
        self._registry = InstrumentRegistry()
        self._load_defaults()

    def _load_defaults(self) -> None:
        for spec in _MCX_SPECS:
            base = Instrument(
                symbol=spec["symbol"],
                exchange=Exchange.MCX,
                asset_class=AssetClass.COMMODITY_FUTURES,
                lot_size=spec["lot_size"],
                tick_size=Decimal(spec["tick_size"]),
                margin_pct=Decimal(spec["margin_pct"]),
            )
            self._registry.register(base)

        for spec in _NSE_FO_SPECS:
            base = Instrument(
                symbol=spec["symbol"],
                exchange=Exchange.NSE,
                asset_class=AssetClass.INDEX_FUTURES,
                lot_size=spec["lot_size"],
                tick_size=Decimal(spec["tick_size"]),
                margin_pct=Decimal(spec["margin_pct"]),
            )
            self._registry.register(base)

    def get(self, symbol: str) -> Instrument:
        """Look up a base instrument (no expiry)."""
        return self._registry.get(symbol)

    def get_front_month(self, symbol: str, as_of: date | None = None) -> Instrument:
        """
        Return an ``Instrument`` with the current front-month expiry set.

        Front month = current month if expiry > as_of, else next month.
        """
        as_of = as_of or date.today()
        base = self._registry.get(symbol)

        if base.exchange == Exchange.MCX:
            expiry = self._mcx_front_month(as_of)
        else:
            expiry = self._nse_front_month(as_of)

        return Instrument(
            symbol=base.symbol,
            exchange=base.exchange,
            asset_class=base.asset_class,
            lot_size=base.lot_size,
            tick_size=base.tick_size,
            margin_pct=base.margin_pct,
            currency=base.currency,
            expiry=expiry.strftime("%Y-%m-%d"),
        )

    def rollover(self, symbol: str, as_of: date | None = None) -> Instrument:
        """Return the *next*-month contract (used for rollover)."""
        as_of = as_of or date.today()
        # Move to next month
        if as_of.month == 12:
            next_month_date = date(as_of.year + 1, 1, 1)
        else:
            next_month_date = date(as_of.year, as_of.month + 1, 1)
        return self.get_front_month(symbol, as_of=next_month_date)

    def days_to_expiry(self, symbol: str, as_of: date | None = None) -> int:
        """Calendar days until front-month expiry."""
        as_of = as_of or date.today()
        instr = self.get_front_month(symbol, as_of)
        expiry_date = datetime.strptime(instr.expiry, "%Y-%m-%d").date()
        return (expiry_date - as_of).days

    def should_rollover(self, symbol: str, as_of: date | None = None, days_before: int = 3) -> bool:
        """Return True if contract should be rolled (within *days_before* of expiry)."""
        return self.days_to_expiry(symbol, as_of) <= days_before

    def all_base_symbols(self) -> list[str]:
        return [i.symbol for i in self._registry.all()]

    # ------------------------------------------------------------------
    # Expiry calculation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nse_front_month(as_of: date) -> date:
        expiry = _last_thursday(as_of.year, as_of.month)
        if as_of > expiry:
            # Roll to next month
            if as_of.month == 12:
                expiry = _last_thursday(as_of.year + 1, 1)
            else:
                expiry = _last_thursday(as_of.year, as_of.month + 1)
        return expiry

    @staticmethod
    def _mcx_front_month(as_of: date) -> date:
        expiry = _last_biz_day(as_of.year, as_of.month)
        if as_of >= expiry:
            if as_of.month == 12:
                expiry = _last_biz_day(as_of.year + 1, 1)
            else:
                expiry = _last_biz_day(as_of.year, as_of.month + 1)
        return expiry

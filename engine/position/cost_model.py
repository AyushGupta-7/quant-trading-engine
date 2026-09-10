"""
engine/position/cost_model.py
------------------------------
Computes transaction costs (STT, CTT, brokerage, exchange charges) for Indian
markets with Decimal precision.

MCX (Commodity Futures):
  * CTT (Commodity Transaction Tax): on sell side only.
  * Exchange transaction charge: per executed value.
  * SEBI fee.
  * GST: on brokerage + exchange charges.
  * Flat brokerage per executed lot.

NSE F&O (Index/Equity Futures):
  * STT: on sell side only (futures sell).
  * Exchange transaction charge.
  * SEBI fee.
  * GST.
  * Flat brokerage per executed lot.

All rates are stored as Decimal to avoid floating-point drift.
"""

from __future__ import annotations

from decimal import Decimal

from engine.core.events import Fill, Side
from engine.core.instrument import AssetClass, Exchange, Instrument


class CostModel:
    """
    Computes all-in transaction cost for a fill in INR.

    Parameters
    ----------
    costs_cfg:
        The ``costs`` section of config.
    """

    def __init__(self, costs_cfg: dict) -> None:
        mcx = costs_cfg.get("mcx", {})
        nse = costs_cfg.get("nse_fo", {})

        # MCX rates
        self._mcx_ctt         = Decimal(str(mcx.get("ctt_pct", "0.0001")))
        self._mcx_exc_charge  = Decimal(str(mcx.get("exchange_txn_charge_pct", "0.0026")))
        self._mcx_sebi_fee    = Decimal(str(mcx.get("sebi_fee_pct", "0.000001")))
        self._mcx_gst         = Decimal(str(mcx.get("gst_pct", "0.18")))
        self._mcx_brokerage   = Decimal(str(mcx.get("brokerage_per_lot", "20")))

        # NSE F&O rates
        self._nse_stt         = Decimal(str(nse.get("stt_pct", "0.000125")))
        self._nse_exc_charge  = Decimal(str(nse.get("exchange_txn_charge_pct", "0.0019")))
        self._nse_sebi_fee    = Decimal(str(nse.get("sebi_fee_pct", "0.000001")))
        self._nse_gst         = Decimal(str(nse.get("gst_pct", "0.18")))
        self._nse_brokerage   = Decimal(str(nse.get("brokerage_per_lot", "20")))

    def compute(self, fill: Fill, instrument: Instrument) -> Decimal:
        """
        Compute total transaction cost for *fill* in INR.

        Parameters
        ----------
        fill:
            The executed fill.
        instrument:
            The instrument spec (to determine exchange and lot size).
        """
        notional = fill.fill_price * instrument.lot_size * fill.fill_qty

        if instrument.exchange == Exchange.MCX:
            return self._mcx_cost(fill, instrument, notional)
        elif instrument.exchange in (Exchange.NSE, Exchange.BSE):
            return self._nse_fo_cost(fill, instrument, notional)
        else:
            return Decimal(0)

    # ------------------------------------------------------------------

    def _mcx_cost(
        self, fill: Fill, instrument: Instrument, notional: Decimal
    ) -> Decimal:
        brokerage = self._mcx_brokerage * fill.fill_qty
        exc_charge = notional * self._mcx_exc_charge / 100
        sebi_fee   = notional * self._mcx_sebi_fee / 100
        ctt        = notional * self._mcx_ctt / 100 if fill.side == Side.SELL else Decimal(0)
        gst        = (brokerage + exc_charge) * self._mcx_gst
        return brokerage + exc_charge + sebi_fee + ctt + gst

    def _nse_fo_cost(
        self, fill: Fill, instrument: Instrument, notional: Decimal
    ) -> Decimal:
        brokerage  = self._nse_brokerage * fill.fill_qty
        exc_charge = notional * self._nse_exc_charge / 100
        sebi_fee   = notional * self._nse_sebi_fee / 100
        stt        = notional * self._nse_stt / 100 if fill.side == Side.SELL else Decimal(0)
        gst        = (brokerage + exc_charge) * self._nse_gst
        return brokerage + exc_charge + sebi_fee + stt + gst

    @classmethod
    def from_config(cls, cfg: dict) -> "CostModel":
        return cls(cfg.get("costs", {}))

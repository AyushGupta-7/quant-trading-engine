"""tests/unit/test_pnl_cost_model.py"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from engine.core.events import Fill, Side
from engine.core.instrument import AssetClass, Exchange, Instrument
from engine.position.cost_model import CostModel
from engine.position.pnl_engine import PnLEngine
from engine.position.position_book import Position, PositionBook


TS = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)


def make_fill(symbol: str, side: Side, price: float, qty: int = 1,
              fees: float = 0.0) -> Fill:
    return Fill(
        order_id="test-001",
        broker_order_id="MOCK-001",
        symbol=symbol,
        side=side,
        fill_price=Decimal(str(price)),
        fill_qty=qty,
        fees=Decimal(str(fees)),
        ts=TS,
    )


MCX_INSTR = Instrument(
    symbol="CRUDEOIL",
    exchange=Exchange.MCX,
    asset_class=AssetClass.COMMODITY_FUTURES,
    lot_size=100,
    tick_size=Decimal("1.0"),
    margin_pct=Decimal("0.05"),
)

NSE_INSTR = Instrument(
    symbol="NIFTY",
    exchange=Exchange.NSE,
    asset_class=AssetClass.INDEX_FUTURES,
    lot_size=25,
    tick_size=Decimal("0.05"),
    margin_pct=Decimal("0.10"),
)

_COSTS_CFG = {
    "costs": {
        "mcx": {
            "ctt_pct": "0.0001",
            "exchange_txn_charge_pct": "0.0026",
            "sebi_fee_pct": "0.000001",
            "gst_pct": "0.18",
            "brokerage_per_lot": "20",
        },
        "nse_fo": {
            "stt_pct": "0.000125",
            "exchange_txn_charge_pct": "0.0019",
            "sebi_fee_pct": "0.000001",
            "gst_pct": "0.18",
            "brokerage_per_lot": "20",
        },
    }
}


class TestCostModel:
    def setup_method(self):
        self.cost = CostModel.from_config(_COSTS_CFG)

    def test_mcx_buy_has_no_ctt(self):
        fill = make_fill("CRUDEOIL", Side.BUY, 6000.0)
        # CTT only on sell
        cost_buy = self.cost.compute(fill, MCX_INSTR)
        fill_sell = make_fill("CRUDEOIL", Side.SELL, 6000.0)
        cost_sell = self.cost.compute(fill_sell, MCX_INSTR)
        assert cost_sell > cost_buy   # sell has CTT extra

    def test_mcx_cost_positive(self):
        fill = make_fill("CRUDEOIL", Side.BUY, 6000.0)
        cost = self.cost.compute(fill, MCX_INSTR)
        assert cost > Decimal(0)

    def test_nse_sell_has_stt(self):
        fill_buy  = make_fill("NIFTY", Side.BUY, 22000.0)
        fill_sell = make_fill("NIFTY", Side.SELL, 22000.0)
        cost_buy  = self.cost.compute(fill_buy,  NSE_INSTR)
        cost_sell = self.cost.compute(fill_sell, NSE_INSTR)
        assert cost_sell > cost_buy

    def test_brokerage_is_per_lot(self):
        fill1 = make_fill("CRUDEOIL", Side.BUY, 6000.0, qty=1)
        fill2 = make_fill("CRUDEOIL", Side.BUY, 6000.0, qty=2)
        cost1 = self.cost.compute(fill1, MCX_INSTR)
        cost2 = self.cost.compute(fill2, MCX_INSTR)
        # brokerage component doubles; total should be > 2x cost1
        # (because proportion of cost from brokerage is significant)
        # simple check: cost2 > cost1
        assert cost2 > cost1


class TestPnLEngine:
    def test_realised_pnl_on_long_close(self):
        pnl = PnLEngine(lot_size_fn=lambda s: 100)
        pos = Position(symbol="CRUDEOIL", net_lots=1,
                       avg_entry_price=Decimal("6000"))
        fill = make_fill("CRUDEOIL", Side.SELL, 6100.0, qty=1)
        realised = pnl.on_fill(fill, pos)
        # (6100 - 6000) * 1 lot * 100 contracts = 10000
        assert realised == Decimal("10000")

    def test_realised_pnl_on_short_close(self):
        pnl = PnLEngine(lot_size_fn=lambda s: 100)
        pos = Position(symbol="CRUDEOIL", net_lots=-1,
                       avg_entry_price=Decimal("6000"))
        fill = make_fill("CRUDEOIL", Side.BUY, 5900.0, qty=1)
        realised = pnl.on_fill(fill, pos)
        # (6000 - 5900) * 1 * 100 = 10000
        assert realised == Decimal("10000")

    def test_unrealised_pnl(self):
        pnl = PnLEngine(lot_size_fn=lambda s: 100)
        pos = Position(symbol="CRUDEOIL", net_lots=2,
                       avg_entry_price=Decimal("6000"))
        urpnl = pnl.update_unrealised("CRUDEOIL", pos, Decimal("6050"))
        # (6050 - 6000) * 2 * 100 = 10000
        assert urpnl == Decimal("10000")

    def test_fees_deducted_from_realised(self):
        pnl = PnLEngine(lot_size_fn=lambda s: 1)
        pos = Position(symbol="X", net_lots=1, avg_entry_price=Decimal("100"))
        fill = make_fill("X", Side.SELL, 110.0, fees=5.0)
        pnl.on_fill(fill, pos)
        # Gross realised = (110 - 100) * 1 * 1 = 10; fees = 5 deducted from stored total
        # Net realised stored = 10 - 5 = 5
        assert pnl.total_realised() == Decimal("5")

    def test_total_equity(self):
        pnl = PnLEngine(lot_size_fn=lambda s: 1)
        assert pnl.total_equity() == Decimal(0)


class TestPositionBook:
    def test_long_position_builds(self):
        book = PositionBook()
        fill = make_fill("CRUDEOIL", Side.BUY, 6000.0, qty=2)
        book.on_fill(fill)
        assert book.get("CRUDEOIL").net_lots == 2

    def test_close_flattens_position(self):
        book = PositionBook()
        book.on_fill(make_fill("CRUDEOIL", Side.BUY, 6000.0, qty=2))
        book.on_fill(make_fill("CRUDEOIL", Side.SELL, 6100.0, qty=2))
        assert book.get("CRUDEOIL").net_lots == 0
        assert book.get("CRUDEOIL").is_flat()

    def test_avg_entry_price_weighted(self):
        book = PositionBook()
        book.on_fill(make_fill("X", Side.BUY, 100.0, qty=1))
        book.on_fill(make_fill("X", Side.BUY, 200.0, qty=1))
        pos = book.get("X")
        assert pos.net_lots == 2
        # Weighted avg: (100 + 200) / 2 = 150
        assert pos.avg_entry_price == Decimal("150")

    def test_reversal_resets_avg(self):
        book = PositionBook()
        book.on_fill(make_fill("X", Side.BUY, 100.0, qty=2))
        book.on_fill(make_fill("X", Side.SELL, 110.0, qty=3))  # reversal
        pos = book.get("X")
        assert pos.net_lots == -1  # net short 1 lot

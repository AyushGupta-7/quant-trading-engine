"""tests/unit/test_contract_master.py"""

from datetime import date

import pytest

from engine.data.contract_master import ContractMaster, _last_thursday, _last_biz_day


class TestExpiry:
    def test_last_thursday_jan_2024(self):
        d = _last_thursday(2024, 1)
        assert d.weekday() == 3          # Thursday
        assert d.month == 1

    def test_last_thursday_dec_2023(self):
        d = _last_thursday(2023, 12)
        assert d.weekday() == 3
        assert d.month == 12

    def test_last_biz_day_not_weekend(self):
        d = _last_biz_day(2024, 1)
        assert d.weekday() < 5          # Mon-Fri


class TestContractMaster:
    def setup_method(self):
        self.master = ContractMaster()

    def test_get_base_instrument(self):
        instr = self.master.get("CRUDEOIL")
        assert instr.symbol == "CRUDEOIL"
        assert instr.lot_size == 100

    def test_get_front_month_has_expiry(self):
        instr = self.master.get_front_month("GOLD")
        assert instr.expiry is not None

    def test_front_month_expiry_in_future(self):
        instr = self.master.get_front_month("NIFTY", as_of=date(2024, 1, 10))
        expiry = date.fromisoformat(instr.expiry)
        assert expiry >= date(2024, 1, 10)

    def test_rollover_returns_next_month(self):
        curr = self.master.get_front_month("CRUDEOIL", as_of=date(2024, 1, 10))
        next_ = self.master.rollover("CRUDEOIL", as_of=date(2024, 1, 10))
        curr_expiry = date.fromisoformat(curr.expiry)
        next_expiry = date.fromisoformat(next_.expiry)
        assert next_expiry > curr_expiry

    def test_should_rollover_within_3_days(self):
        # Pick a date just before expiry
        instr = self.master.get_front_month("CRUDEOIL", as_of=date(2024, 1, 10))
        expiry = date.fromisoformat(instr.expiry)
        two_days_before = expiry - __import__("datetime").timedelta(days=2)
        assert self.master.should_rollover("CRUDEOIL", as_of=two_days_before, days_before=3)

    def test_all_base_symbols(self):
        symbols = self.master.all_base_symbols()
        assert "CRUDEOIL" in symbols
        assert "NIFTY" in symbols
        assert len(symbols) >= 5

from datetime import date, timedelta

import pytest

from optrisk import MarketSnapshot, OptionContract

VAL_DATE = date(2026, 1, 5)


@pytest.fixture
def market() -> MarketSnapshot:
    return MarketSnapshot("X", VAL_DATE, 100.0, 0.04, 0.015, 0.25)


@pytest.fixture
def make_contract():
    def _make(kind: str = "call", strike: float = 100.0, days: int = 90, exercise: str = "european") -> OptionContract:
        return OptionContract("X", strike, VAL_DATE + timedelta(days=days), kind, exercise, f"{kind}{strike:g}-{days}")

    return _make

"""Controlled validation grid: strikes x maturities x volatilities x rates x dividend yields x type."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ..contracts import OptionContract, OptionType
from ..generate import SAMPLE_VALUATION_DATE
from ..market import MarketSnapshot

SPOT = 100.0

FULL = dict(
    moneyness=(0.70, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30),  # K / S
    days=(7, 30, 90, 180, 365, 730),
    vol=(0.10, 0.20, 0.40, 0.80),
    rate=(-0.005, 0.0, 0.02, 0.05),
    div=(0.0, 0.02, 0.05),
)
QUICK = dict(
    moneyness=(0.85, 1.00, 1.15),
    days=(7, 90, 365),
    vol=(0.10, 0.40),
    rate=(0.0, 0.05),
    div=(0.0, 0.03),
)


@dataclass(frozen=True)
class Grid:
    """Parallel 1-D arrays, one entry per (contract, market) combination."""

    sign: np.ndarray
    spot: np.ndarray
    strike: np.ndarray
    days: np.ndarray
    tau: np.ndarray
    rate: np.ndarray
    div: np.ndarray
    vol: np.ndarray
    moneyness: np.ndarray

    def __len__(self) -> int:
        return int(self.sign.shape[0])

    def subset(self, idx: np.ndarray) -> "Grid":
        return Grid(**{f: getattr(self, f)[idx] for f in self.__dataclass_fields__})

    def every(self, step: int, offset: int = 0) -> "Grid":
        return self.subset(np.arange(offset, len(self), step))

    def contract_and_market(self, i: int, valuation_date: date = SAMPLE_VALUATION_DATE) -> tuple[OptionContract, MarketSnapshot]:
        typ = OptionType.CALL if self.sign[i] == 1 else OptionType.PUT
        contract = OptionContract(
            "GRID", float(self.strike[i]), valuation_date + timedelta(days=int(self.days[i])), typ, contract_id=f"G{i}"
        )
        market = MarketSnapshot("GRID", valuation_date, float(self.spot[i]), float(self.rate[i]), float(self.div[i]), float(self.vol[i]))
        return contract, market

    def describe(self, i: int) -> str:
        t = "call" if self.sign[i] == 1 else "put"
        return (
            f"{t} K/S={self.moneyness[i]:.2f} T={int(self.days[i])}d vol={self.vol[i]:.2f} "
            f"r={self.rate[i]:+.3f} q={self.div[i]:.2f}"
        )


def validation_grid(quick: bool = False) -> Grid:
    g = QUICK if quick else FULL
    rows = list(itertools.product(g["moneyness"], g["days"], g["vol"], g["rate"], g["div"], (1.0, -1.0)))
    m, d, v, r, q, s = (np.array(col, dtype=float) for col in zip(*rows))
    return Grid(
        sign=s,
        spot=np.full(len(rows), SPOT),
        strike=SPOT * m,
        days=d.astype(int),
        tau=d / 365.0,
        rate=r,
        div=q,
        vol=v,
        moneyness=m,
    )

"""Generated, fully synthetic data. No market-data subscription is needed.

Everything is deterministic: the sample portfolio is hand-built, the quote chain is computed
from a known volatility smile (so implied-vol recovery can be checked against the truth), and
random books use an explicit seed.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .contracts import ExerciseStyle, OptionContract, OptionType
from .engines.black_scholes import bsm_scalar
from .conventions import year_fraction
from .io import write_records
from .market import MarketSnapshot
from .portfolio import Portfolio, Position

#: Fixed valuation date so every run of the demo is byte-for-byte comparable.
SAMPLE_VALUATION_DATE = date(2026, 1, 5)


def sample_markets(valuation_date: date = SAMPLE_VALUATION_DATE) -> dict[str, MarketSnapshot]:
    spec = {
        # name: (spot, rate, dividend yield, vol)
        "SIM_A": (100.0, 0.040, 0.015, 0.22),
        "SIM_B": (250.0, 0.040, 0.000, 0.35),
        "SIM_C": (50.0, 0.040, 0.030, 0.28),
    }
    return {n: MarketSnapshot(n, valuation_date, s, r, q, v) for n, (s, r, q, v) in spec.items()}


# (contract_id, underlying, type, strike, days to expiry, quantity). Positive quantity is long.
_SAMPLE_POSITIONS = [
    ("A1", "SIM_A", "call", 100.0, 4, +30),  # near-expiry, at the money: large gamma
    ("A2", "SIM_A", "put", 95.0, 4, -25),
    ("A3", "SIM_A", "call", 100.0, 30, +20),  # call spread
    ("A4", "SIM_A", "call", 110.0, 30, -20),
    ("A5", "SIM_A", "put", 90.0, 90, +40),  # protective puts
    ("A6", "SIM_A", "put", 100.0, 90, -15),
    ("A7", "SIM_A", "call", 105.0, 180, +25),
    ("A8", "SIM_A", "put", 85.0, 180, +10),
    ("B1", "SIM_B", "call", 250.0, 30, -10),  # short straddle-ish: short gamma / short vega
    ("B2", "SIM_B", "put", 240.0, 30, -10),
    ("B3", "SIM_B", "call", 270.0, 90, +15),
    ("B4", "SIM_B", "put", 220.0, 90, +20),
    ("B5", "SIM_B", "call", 250.0, 180, +10),
    ("B6", "SIM_B", "put", 250.0, 180, -5),
    ("C1", "SIM_C", "call", 50.0, 4, +50),
    ("C2", "SIM_C", "call", 55.0, 30, -40),
    ("C3", "SIM_C", "put", 45.0, 90, +30),
    ("C4", "SIM_C", "put", 50.0, 180, -20),
]


def sample_portfolio(valuation_date: date = SAMPLE_VALUATION_DATE) -> Portfolio:
    positions = [
        Position(
            OptionContract(und, strike, valuation_date + timedelta(days=days), typ, ExerciseStyle.EUROPEAN, cid),
            quantity=qty,
            multiplier=100.0,
        )
        for cid, und, typ, strike, days, qty in _SAMPLE_POSITIONS
    ]
    return Portfolio(tuple(positions), name="sample_portfolio")


def sample_quote_chain(
    markets: dict[str, MarketSnapshot] | None = None,
) -> tuple[list[OptionContract], dict[str, float], dict[str, float], dict[str, str]]:
    """Option chain on SIM_A with quotes generated from a skewed smile.

    Returns ``(contracts, quotes, true_vols, notes)``. The true volatility is
    ``0.20 - 0.25 * ln(K/F) + 0.40 * ln(K/F)^2`` (skew plus curvature), so the implied-vol solver
    has a non-flat truth to recover. A few deliberately invalid quotes (``notes`` != "") are
    appended to exercise the explicit failure statuses.
    """
    markets = markets or sample_markets()
    m = markets["SIM_A"]
    contracts: list[OptionContract] = []
    quotes: dict[str, float] = {}
    truth: dict[str, float] = {}
    notes: dict[str, str] = {}
    for days in (30, 90, 180):
        expiry = m.valuation_date + timedelta(days=days)
        tau = year_fraction(m.valuation_date, expiry)
        fwd = m.spot * math.exp((m.rate - m.dividend_yield) * tau)
        for strike in range(80, 125, 5):
            x = math.log(strike / fwd)
            vol = 0.20 - 0.25 * x + 0.40 * x * x
            for typ in (OptionType.CALL, OptionType.PUT):
                cid = f"Q{days}{typ.value[0].upper()}{strike}"
                contract = OptionContract("SIM_A", float(strike), expiry, typ, contract_id=cid)
                contracts.append(contract)
                quotes[cid] = bsm_scalar(typ.sign, m.spot, strike, tau, m.rate, m.dividend_yield, vol)[0]
                truth[cid] = vol

    expiry = m.valuation_date + timedelta(days=90)
    tau = year_fraction(m.valuation_date, expiry)
    disc_fwd = m.spot * math.exp(-m.dividend_yield * tau)
    disc_k = 100.0 * math.exp(-m.rate * tau)
    bad = [
        ("BAD_below_intrinsic", OptionType.CALL, 80.0, 0.5 * (disc_fwd - 80.0 * math.exp(-m.rate * tau)), "call quoted below discounted intrinsic"),
        ("BAD_above_upper", OptionType.CALL, 100.0, disc_fwd + 0.5, "call quoted above S*exp(-qT)"),
        ("BAD_put_above_upper", OptionType.PUT, 100.0, disc_k + 0.5, "put quoted above K*exp(-rT)"),
        ("BAD_negative", OptionType.PUT, 100.0, -0.25, "negative price"),
        ("BAD_nan", OptionType.CALL, 100.0, float("nan"), "NaN price"),
    ]
    for cid, typ, strike, price, note in bad:
        contracts.append(OptionContract("SIM_A", strike, expiry, typ, contract_id=cid))
        quotes[cid] = price
        notes[cid] = note
    return contracts, quotes, truth, notes


def random_portfolio(
    n_positions: int, n_underlyings: int = 20, seed: int = 2026, valuation_date: date = SAMPLE_VALUATION_DATE
) -> tuple[Portfolio, dict[str, MarketSnapshot]]:
    """Reproducible random book of European options for scaling benchmarks."""
    rng = np.random.default_rng(seed)
    spots = np.round(rng.uniform(20.0, 500.0, n_underlyings), 2)
    markets = {
        f"U{i:03d}": MarketSnapshot(
            f"U{i:03d}",
            valuation_date,
            float(spots[i]),
            float(rng.uniform(0.0, 0.05)),
            float(rng.uniform(0.0, 0.04)),
            float(rng.uniform(0.12, 0.60)),
        )
        for i in range(n_underlyings)
    }
    names = list(markets)
    und_idx = rng.integers(0, n_underlyings, n_positions)
    days = rng.integers(3, 731, n_positions)
    log_k = rng.uniform(-0.30, 0.30, n_positions)
    is_call = rng.random(n_positions) < 0.5
    qty = rng.integers(1, 51, n_positions) * rng.choice([-1, 1], n_positions)
    positions = []
    for i in range(n_positions):
        u = names[und_idx[i]]
        strike = round(float(markets[u].spot * math.exp(log_k[i])), 2)
        contract = OptionContract(
            u,
            max(strike, 0.01),
            valuation_date + timedelta(days=int(days[i])),
            OptionType.CALL if is_call[i] else OptionType.PUT,
            contract_id=f"R{i:07d}",
        )
        positions.append(Position(contract, float(qty[i]), 100.0))
    return Portfolio(tuple(positions), name=f"random_{n_positions}"), markets


# ---------------------------------------------------------------------------- writers
def contracts_rows(contracts: list[OptionContract]) -> list[dict[str, object]]:
    return [
        {
            "contract_id": c.contract_id,
            "underlying": c.underlying,
            "option_type": c.option_type.value,
            "strike": c.strike,
            "expiry": c.expiry,
            "exercise": c.exercise.value,
        }
        for c in contracts
    ]


def positions_rows(portfolio: Portfolio) -> list[dict[str, object]]:
    return [
        {"contract_id": p.contract.contract_id, "quantity": p.quantity, "multiplier": p.multiplier} for p in portfolio
    ]


def markets_rows(markets: dict[str, MarketSnapshot]) -> list[dict[str, object]]:
    return [
        {
            "underlying": m.underlying,
            "valuation_date": m.valuation_date,
            "spot": m.spot,
            "rate": m.rate,
            "dividend_yield": m.dividend_yield,
            "volatility": m.volatility,
        }
        for m in markets.values()
    ]


def write_sample_data(outdir: str | Path) -> dict[str, Path]:
    """Write the sample inputs used by the demo into ``outdir`` and return their paths."""
    out = Path(outdir)
    markets = sample_markets()
    portfolio = sample_portfolio()
    chain, quotes, truth, notes = sample_quote_chain(markets)
    paths = {
        "market": write_records(out / "market.csv", markets_rows(markets)),
        "contracts": write_records(out / "contracts.csv", contracts_rows([p.contract for p in portfolio])),
        "positions": write_records(out / "positions.csv", positions_rows(portfolio)),
        "chain_contracts": write_records(out / "chain_contracts.csv", contracts_rows(chain)),
        "chain_quotes": write_records(
            out / "chain_quotes.csv",
            [
                {"contract_id": cid, "price": quotes[cid], "true_vol": truth.get(cid, ""), "note": notes.get(cid, "")}
                for cid in quotes
            ],
        ),
    }
    return paths

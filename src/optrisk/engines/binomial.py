"""Cox-Ross-Rubinstein binomial tree.

    dt = T/n,  u = exp(sigma sqrt dt),  d = 1/u,  p = (e^{(r-q) dt} - d) / (u - d)

European and American exercise are supported. Backward induction is vectorised
across the nodes of each time step, so the cost is n NumPy operations on arrays
of length <= n+1 (O(n^2) arithmetic).

Known behaviour, not a bug: CRR prices converge at O(1/n) but with an odd/even
oscillation caused by where the strike falls relative to the terminal nodes. The
validation report quantifies the error against Black-Scholes.

Greeks
------
* delta, gamma, theta come from the tree nodes at steps 1 and 2 (Hull's method),
  so they cost nothing extra and are consistent with the tree price.
* vega and rho use central differences, re-building the tree four times.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ..contracts import ExerciseStyle, OptionContract
from ..conventions import DAYS_PER_YEAR, RHO_UNIT, VEGA_UNIT
from ..errors import InvalidInputError, NumericalError
from ..market import MarketSnapshot
from ..results import Greeks, PricingResult
from .base import PricingEngine

MIN_STEPS = 3
MAX_STEPS = 200_000
#: Central-difference bump for rho (absolute rate, 1 basis point).
RHO_BUMP = 1e-4


class TreeValues(NamedTuple):
    price: float
    v1: tuple[float, float] | None  # values at step 1: (down, up)
    v2: tuple[float, float, float] | None  # values at step 2: (down-down, mid, up-up)
    u: float
    d: float
    p: float


def crr_tree(
    sign: int,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    div: float,
    vol: float,
    steps: int,
    american: bool = False,
    nodes: bool = False,
) -> TreeValues:
    """Value an option on an ``steps``-step CRR tree. ``nodes`` also returns steps 1 and 2."""
    dt = tau / steps
    sqrt_dt = math.sqrt(dt)
    u = math.exp(vol * sqrt_dt)
    d = 1.0 / u
    growth = math.exp((rate - div) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise NumericalError(
            f"CRR risk-neutral probability p={p:.6g} is outside (0, 1) with {steps} steps; "
            "increase the number of steps (|r-q|*sqrt(dt) must be smaller than sigma)"
        )
    disc = math.exp(-rate * dt)
    pu = disc * p
    pd = disc * (1.0 - p)

    # Terminal nodes: j up-moves out of n  ->  S * exp(sigma sqrt(dt) (2j - n))
    j = np.arange(steps + 1, dtype=float)
    spots = spot * np.exp(vol * sqrt_dt * (2.0 * j - steps))
    values = np.maximum(sign * (spots - strike), 0.0)

    v1 = v2 = None
    for k in range(steps - 1, -1, -1):
        values = pu * values[1:] + pd * values[:-1]
        if american:
            spots = spots[1:] * d  # node j at step k is node j+1 at step k+1 times d
            np.maximum(values, sign * (spots - strike), out=values)
        if nodes:
            if k == 2:
                v2 = (float(values[0]), float(values[1]), float(values[2]))
            elif k == 1:
                v1 = (float(values[0]), float(values[1]))
    return TreeValues(float(values[0]), v1, v2, u, d, p)


class BinomialEngine(PricingEngine):
    """CRR tree pricing for European and American vanilla options."""

    name = "binomial_crr"
    supports_american = True

    def __init__(self, steps: int = 500) -> None:
        if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
            raise InvalidInputError(f"steps must be an integer, got {steps!r}")
        if not (MIN_STEPS <= steps <= MAX_STEPS):
            raise InvalidInputError(f"steps must be in [{MIN_STEPS}, {MAX_STEPS}], got {steps}")
        self.steps = int(steps)

    def _price(self, contract: OptionContract, market: MarketSnapshot, tau: float, greeks: bool) -> PricingResult:
        sign = contract.option_type.sign
        american = contract.exercise is ExerciseStyle.AMERICAN
        s, k, r, q, vol = market.spot, contract.strike, market.rate, market.dividend_yield, market.volatility
        tree = crr_tree(sign, s, k, tau, r, q, vol, self.steps, american, nodes=greeks)
        diagnostics = {"steps": self.steps, "american": american, "u": tree.u, "d": tree.d, "p": tree.p}
        if not greeks:
            return PricingResult(price=tree.price, method=self.name, diagnostics=diagnostics)

        assert tree.v1 is not None and tree.v2 is not None
        dt = tau / self.steps
        up, dn = tree.u, tree.d
        delta = (tree.v1[1] - tree.v1[0]) / (s * up - s * dn)
        s_uu, s_dd = s * up * up, s * dn * dn
        gamma = ((tree.v2[2] - tree.v2[1]) / (s_uu - s) - (tree.v2[1] - tree.v2[0]) / (s - s_dd)) / (0.5 * (s_uu - s_dd))
        theta_year = (tree.v2[1] - tree.price) / (2.0 * dt)

        def value(rate_: float, vol_: float) -> float:
            return crr_tree(sign, s, k, tau, rate_, q, vol_, self.steps, american).price

        h_vol = min(0.005, 0.25 * vol)
        vega = (value(r, vol + h_vol) - value(r, vol - h_vol)) / (2.0 * h_vol)
        rho = (value(r + RHO_BUMP, vol) - value(r - RHO_BUMP, vol)) / (2.0 * RHO_BUMP)
        return PricingResult(
            price=tree.price,
            method=self.name,
            greeks=Greeks(
                delta=delta,
                gamma=gamma,
                vega=vega * VEGA_UNIT,
                theta=theta_year / DAYS_PER_YEAR,
                rho=rho * RHO_UNIT,
            ),
            diagnostics=diagnostics,
        )

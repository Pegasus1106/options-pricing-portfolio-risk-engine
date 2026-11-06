"""Black-Scholes-Merton analytic pricing for European options.

Model: dS = (r - q) S dt + sigma S dW, constant r, q, sigma.

    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T),   d2 = d1 - sigma sqrt T
    price = phi * [ S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2) ]       (phi = +1 call, -1 put)

Two implementations of the same formulas are provided:

* ``bsm_scalar``  pure Python (``math``), fastest for a handful of options;
* ``bsm_arrays``  vectorised NumPy/SciPy, for large batches and scenario grids.

Tests assert they agree. Both return Greeks in the units of ``conventions.py``
(vega per vol point, rho per rate point, theta per calendar day).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import ndtr

from ..contracts import OptionContract
from ..conventions import DAYS_PER_YEAR, RHO_UNIT, VEGA_UNIT
from ..market import MarketSnapshot
from ..results import Greeks, PricingResult
from .base import PricingEngine, expired_price_and_greeks

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _ncdf(x: float) -> float:
    # erfc keeps relative accuracy in the lower tail, unlike 0.5 * (1 + erf(x / sqrt 2)).
    return 0.5 * math.erfc(-x / _SQRT2)


def _npdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def bsm_scalar(
    sign: int, spot: float, strike: float, tau: float, rate: float, div: float, vol: float
) -> tuple[float, Greeks]:
    """Price and Greeks for one European option. ``tau <= 0`` returns the expired value."""
    if tau <= 0.0:
        return expired_price_and_greeks(sign, spot, strike)
    sqrt_t = math.sqrt(tau)
    vol_sqrt_t = vol * sqrt_t
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol * vol) * tau) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    disc_r = math.exp(-rate * tau)
    disc_q = math.exp(-div * tau)
    nd1 = _ncdf(sign * d1)
    nd2 = _ncdf(sign * d2)
    pdf1 = _npdf(d1)

    fwd_leg = spot * disc_q
    strike_leg = strike * disc_r
    price = sign * (fwd_leg * nd1 - strike_leg * nd2)
    delta = sign * disc_q * nd1
    gamma = disc_q * pdf1 / (spot * vol_sqrt_t)
    vega = fwd_leg * pdf1 * sqrt_t
    theta_year = -fwd_leg * pdf1 * vol / (2.0 * sqrt_t) - sign * rate * strike_leg * nd2 + sign * div * fwd_leg * nd1
    rho = sign * strike * tau * disc_r * nd2
    return price, Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega * VEGA_UNIT,
        theta=theta_year / DAYS_PER_YEAR,
        rho=rho * RHO_UNIT,
    )


def bsm_arrays(
    sign: Any, spot: Any, strike: Any, tau: Any, rate: Any, div: Any, vol: Any, *, greeks: bool = True
) -> dict[str, np.ndarray]:
    """Vectorised BSM. Inputs broadcast against each other; ``tau <= 0`` entries get expired values.

    Returns a dict with ``price`` and, if ``greeks``, ``delta gamma vega theta rho``.
    """
    sign, spot, strike, tau, rate, div, vol = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (sign, spot, strike, tau, rate, div, vol))
    )
    live = tau > 0.0
    t = np.where(live, tau, 1.0)  # dummy value keeps the expired lanes free of NaN/inf
    sqrt_t = np.sqrt(t)
    vol_sqrt_t = vol * sqrt_t
    d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol * vol) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    disc_r = np.exp(-rate * t)
    disc_q = np.exp(-div * t)
    nd1 = ndtr(sign * d1)
    nd2 = ndtr(sign * d2)
    fwd_leg = spot * disc_q
    strike_leg = strike * disc_r

    intrinsic = np.maximum(sign * (spot - strike), 0.0)
    out: dict[str, np.ndarray] = {"price": np.where(live, sign * (fwd_leg * nd1 - strike_leg * nd2), intrinsic)}
    if not greeks:
        return out

    pdf1 = _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1)
    delta = sign * disc_q * nd1
    gamma = disc_q * pdf1 / (spot * vol_sqrt_t)
    vega = fwd_leg * pdf1 * sqrt_t
    theta_year = -fwd_leg * pdf1 * vol / (2.0 * sqrt_t) - sign * rate * strike_leg * nd2 + sign * div * fwd_leg * nd1
    rho = sign * strike * t * disc_r * nd2
    zero = np.zeros_like(intrinsic)
    out["delta"] = np.where(live, delta, np.where(intrinsic > 0.0, sign, 0.0))
    out["gamma"] = np.where(live, gamma, zero)
    out["vega"] = np.where(live, vega * VEGA_UNIT, zero)
    out["theta"] = np.where(live, theta_year / DAYS_PER_YEAR, zero)
    out["rho"] = np.where(live, rho * RHO_UNIT, zero)
    return out


class BlackScholesEngine(PricingEngine):
    """Closed-form European pricing with analytic Greeks."""

    name = "black_scholes"

    def _price(self, contract: OptionContract, market: MarketSnapshot, tau: float, greeks: bool) -> PricingResult:
        price, g = bsm_scalar(
            contract.option_type.sign,
            market.spot,
            contract.strike,
            tau,
            market.rate,
            market.dividend_yield,
            market.volatility,
        )
        return PricingResult(
            price=price,
            method=self.name,
            greeks=g if greeks else None,
            diagnostics={"tau_years": tau},
        )

    # The vectorised entry point used by RiskEngine and the benchmarks.
    price_arrays = staticmethod(bsm_arrays)

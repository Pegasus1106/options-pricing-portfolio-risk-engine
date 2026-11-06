"""Implied volatility under the Black-Scholes-Merton model (European options).

Solves  BSM_price(sigma) = observed_price  for sigma.

Why a bracketing solver: the BSM price is strictly increasing in sigma (vega > 0), so a
sign change of ``f(sigma) = BSM(sigma) - quote`` over ``[lo, hi]`` guarantees exactly one
root. Bisection is slow but cannot fail once bracketed; Brent's method (SciPy ``brentq``)
keeps that guarantee and usually converges in far fewer evaluations.

Model-consistent price bounds (no volatility can reproduce a quote outside them):

    call:  max(S e^{-qT} - K e^{-rT}, 0)  <  C  <  S e^{-qT}
    put :  max(K e^{-rT} - S e^{-qT}, 0)  <  P  <  K e^{-rT}

The bounds are open: the lower bound is the limit as sigma -> 0 and the upper bound the
limit as sigma -> infinity, so a quote exactly on a bound has no positive-volatility solution.

Failure is reported through :class:`IVStatus`; the solver does not raise on bad quotes.
Note that for deep in/out-of-the-money options vega is tiny, so the price tolerance is
met by a wide range of volatilities: the *volatility* is then ill-conditioned even though
the *price* round trip is exact. ``IVResult.vega`` lets callers see this.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .contracts import ExerciseStyle, OptionContract
from .conventions import MAX_VOLATILITY, year_fraction
from .engines.black_scholes import bsm_arrays, bsm_scalar
from .market import MarketSnapshot

DEFAULT_VOL_LO = 1e-6
DEFAULT_VOL_HI = 5.0
DEFAULT_XTOL = 1e-12
DEFAULT_MAX_ITER = 200
#: A solution is flagged ill-conditioned when a 1-vol-point move changes the price by less than this fraction of spot.
ILL_CONDITIONED_VEGA_FRACTION = 1e-6


class IVStatus(str, Enum):
    CONVERGED = "converged"
    INVALID_INPUT = "invalid_input"  # non-finite/non-positive inputs, expired option, American exercise, unknown method
    PRICE_BELOW_LOWER_BOUND = "price_below_lower_bound"  # quote <= discounted intrinsic bound: no sigma > 0 fits
    PRICE_ABOVE_UPPER_BOUND = "price_above_upper_bound"  # quote >= S e^{-qT} (call) or K e^{-rT} (put)
    BRACKET_FAILED = "bracket_failed"  # quote inside bounds but outside [price(lo), price(hi)] up to MAX_VOLATILITY
    MAX_ITERATIONS = "max_iterations"  # iteration cap reached before xtol


@dataclass(frozen=True)
class IVResult:
    status: IVStatus
    vol: float  # NaN unless status is CONVERGED
    method: str
    iterations: int = 0
    function_evals: int = 0
    residual: float = float("nan")  # BSM(vol) - quote at the returned vol
    vega: float = float("nan")  # per unit volatility, at the returned vol (conditioning hint)
    bracket: tuple[float, float] | None = None
    message: str = ""
    runtime_s: float = 0.0
    #: False when the price barely depends on volatility at the solution (vega per vol point below
    #: ``ILL_CONDITIONED_VEGA_FRACTION * spot``). The price is reproduced but the volatility is not identifiable.
    well_conditioned: bool = True

    @property
    def ok(self) -> bool:
        return self.status is IVStatus.CONVERGED


def price_bounds(sign: int, spot: float, strike: float, tau: float, rate: float, div: float) -> tuple[float, float]:
    """Open no-arbitrage interval ``(lower, upper)`` for a European option price under BSM."""
    fwd_leg = spot * math.exp(-div * tau)
    strike_leg = strike * math.exp(-rate * tau)
    if sign == 1:
        return max(fwd_leg - strike_leg, 0.0), fwd_leg
    return max(strike_leg - fwd_leg, 0.0), strike_leg


def bisect(f: Callable[[float], float], lo: float, hi: float, xtol: float, max_iter: int) -> tuple[float, int, int, bool]:
    """Bisection for an increasing ``f`` with ``f(lo) < 0 < f(hi)``.

    Returns ``(root, iterations, function_evals, converged)``. Converged means the bracket
    width fell to ``xtol`` (or ``f`` hit exactly zero).
    """
    evals = 0
    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        evals += 1
        if fm == 0.0:
            return mid, it, evals, True
        if fm < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= xtol:
            return 0.5 * (lo + hi), it, evals, True
    return 0.5 * (lo + hi), max_iter, evals, False


def _fail(status: IVStatus, method: str, message: str, start: float, **kw: object) -> IVResult:
    return IVResult(status=status, vol=float("nan"), method=method, message=message, runtime_s=time.perf_counter() - start, **kw)  # type: ignore[arg-type]


def implied_volatility_raw(
    price: float,
    sign: int,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    div: float,
    *,
    method: str = "bisection",
    vol_lo: float = DEFAULT_VOL_LO,
    vol_hi: float = DEFAULT_VOL_HI,
    xtol: float = DEFAULT_XTOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> IVResult:
    """Implied volatility from plain numbers. ``method`` is ``"bisection"`` or ``"brentq"``."""
    start = time.perf_counter()
    if method not in ("bisection", "brentq"):
        return _fail(IVStatus.INVALID_INPUT, method, f"unknown method {method!r}", start)
    inputs = (price, spot, strike, tau, rate, div)
    if sign not in (1, -1) or not all(isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(x) for x in inputs):
        return _fail(IVStatus.INVALID_INPUT, method, "all inputs must be finite numbers and sign must be +1/-1", start)
    if price < 0.0:
        return _fail(IVStatus.INVALID_INPUT, method, f"price must be >= 0, got {price}", start)
    if spot <= 0.0 or strike <= 0.0:
        return _fail(IVStatus.INVALID_INPUT, method, "spot and strike must be > 0", start)
    if tau <= 0.0:
        return _fail(IVStatus.INVALID_INPUT, method, "option is expired or has non-positive time to expiry", start)
    if not (0.0 < vol_lo < vol_hi <= MAX_VOLATILITY) or xtol <= 0.0 or max_iter < 1:
        return _fail(IVStatus.INVALID_INPUT, method, "invalid solver settings", start)

    lower, upper = price_bounds(sign, spot, strike, tau, rate, div)
    bounds = (lower, upper)
    if price <= lower:
        return _fail(
            IVStatus.PRICE_BELOW_LOWER_BOUND, method,
            f"price {price:.10g} is <= the model lower bound {lower:.10g}", start, bracket=bounds,
        )
    if price >= upper:
        return _fail(
            IVStatus.PRICE_ABOVE_UPPER_BOUND, method,
            f"price {price:.10g} is >= the model upper bound {upper:.10g}", start, bracket=bounds,
        )

    evals = 0

    def f(vol: float) -> float:
        nonlocal evals
        evals += 1
        return bsm_scalar(sign, spot, strike, tau, rate, div, vol)[0] - price

    lo, hi = vol_lo, vol_hi
    f_lo = f(lo)
    if f_lo > 0.0:
        return _fail(
            IVStatus.BRACKET_FAILED, method,
            f"price {price:.10g} is below the model price {price + f_lo:.10g} at vol={lo:g}", start,
            function_evals=evals, bracket=(lo, hi),
        )
    f_hi = f(hi)
    while f_hi < 0.0 and hi < MAX_VOLATILITY:  # widen upwards before giving up
        lo, f_lo = hi, f_hi
        hi = min(2.0 * hi, MAX_VOLATILITY)
        f_hi = f(hi)
    if f_hi < 0.0:
        return _fail(
            IVStatus.BRACKET_FAILED, method,
            f"model price at vol={hi:g} is still below the quote", start, function_evals=evals, bracket=(lo, hi),
        )
    bracket = (lo, hi)

    if method == "bisection":
        vol, iters, _evals, converged = bisect(f, lo, hi, xtol, max_iter)
    else:
        try:
            vol, info = brentq(f, lo, hi, xtol=xtol, maxiter=max_iter, full_output=True, disp=False)
        except (ValueError, RuntimeError) as exc:  # defensive: sign change was checked above
            return _fail(IVStatus.BRACKET_FAILED, method, str(exc), start, function_evals=evals, bracket=bracket)
        iters, converged = int(info.iterations), bool(info.converged)

    price_at, g = bsm_scalar(sign, spot, strike, tau, rate, div, vol)
    well = g.vega >= ILL_CONDITIONED_VEGA_FRACTION * spot
    if not converged:
        message = f"no convergence to xtol={xtol:g} within {max_iter} iterations"
    elif not well:
        message = f"price is reproduced but volatility is poorly determined (vega per vol point = {g.vega:.2e})"
    else:
        message = ""
    return IVResult(
        status=IVStatus.CONVERGED if converged else IVStatus.MAX_ITERATIONS,
        vol=vol if converged else float("nan"),
        method=method,
        iterations=iters,
        function_evals=evals,
        residual=price_at - price,
        vega=g.vega / 0.01,
        bracket=bracket,
        message=message,
        runtime_s=time.perf_counter() - start,
        well_conditioned=well,
    )


def implied_volatility(
    price: float, contract: OptionContract, market: MarketSnapshot, *, method: str = "bisection", **kwargs: float
) -> IVResult:
    """Implied volatility of a European contract. ``market.volatility`` is ignored (it is the unknown)."""
    start = time.perf_counter()
    if contract.exercise is not ExerciseStyle.EUROPEAN:
        return _fail(IVStatus.INVALID_INPUT, method, "implied volatility is only supported for European options", start)
    if contract.underlying != market.underlying:
        return _fail(IVStatus.INVALID_INPUT, method, "contract and market underlyings differ", start)
    tau = year_fraction(market.valuation_date, contract.expiry)
    return implied_volatility_raw(
        price, contract.option_type.sign, market.spot, contract.strike, tau, market.rate, market.dividend_yield,
        method=method, **kwargs,  # type: ignore[arg-type]
    )


def implied_volatility_arrays(
    price: np.ndarray,
    sign: np.ndarray,
    spot: np.ndarray,
    strike: np.ndarray,
    tau: np.ndarray,
    rate: np.ndarray,
    div: np.ndarray,
    *,
    vol_lo: float = DEFAULT_VOL_LO,
    vol_hi: float = DEFAULT_VOL_HI,
    iterations: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised bisection over many quotes. Returns ``(vol, status_code)``.

    A fixed number of iterations shrinks ``[vol_lo, vol_hi]`` by 2**-iterations (60 -> ~4e-18 of the
    width, i.e. to double precision). Status codes: 0 converged, 1 invalid input,
    2 below lower bound, 3 above upper bound, 4 bracket failed. Failed lanes get NaN.
    Unlike the scalar solver this one does not widen ``vol_hi``.
    """
    price, sign, spot, strike, tau, rate, div = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (price, sign, spot, strike, tau, rate, div))
    )
    status = np.zeros(price.shape, dtype=np.int8)
    finite = np.isfinite(price) & np.isfinite(spot) & np.isfinite(strike) & np.isfinite(tau) & np.isfinite(rate) & np.isfinite(div)
    with np.errstate(invalid="ignore"):
        invalid = ~finite | (price < 0) | (spot <= 0) | (strike <= 0) | (tau <= 0) | ~np.isin(sign, (-1.0, 1.0))
    status[invalid] = 1

    t = np.where(tau > 0, tau, 1.0)
    fwd_leg = spot * np.exp(-div * t)
    strike_leg = strike * np.exp(-rate * t)
    is_call = sign == 1.0
    lower = np.where(is_call, np.maximum(fwd_leg - strike_leg, 0.0), np.maximum(strike_leg - fwd_leg, 0.0))
    upper = np.where(is_call, fwd_leg, strike_leg)
    with np.errstate(invalid="ignore"):
        status[(status == 0) & (price <= lower)] = 2
        status[(status == 0) & (price >= upper)] = 3

    lo = np.full(price.shape, vol_lo)
    hi = np.full(price.shape, vol_hi)
    safe_sign = np.where(invalid, 1.0, sign)
    safe_spot = np.where(invalid, 1.0, spot)
    safe_strike = np.where(invalid, 1.0, strike)
    safe_price = np.where(invalid, 0.0, price)
    f_lo = bsm_arrays(safe_sign, safe_spot, safe_strike, t, rate, div, lo, greeks=False)["price"] - safe_price
    f_hi = bsm_arrays(safe_sign, safe_spot, safe_strike, t, rate, div, hi, greeks=False)["price"] - safe_price
    status[(status == 0) & ((f_lo > 0) | (f_hi < 0))] = 4

    active = status == 0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = bsm_arrays(safe_sign, safe_spot, safe_strike, t, rate, div, mid, greeks=False)["price"] - safe_price
        go_up = f_mid < 0
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    vol = np.where(active, 0.5 * (lo + hi), np.nan)
    return vol, status

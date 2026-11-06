"""Validation checks. Each returns a :class:`CheckResult`; none raise on a numerical failure.

Tolerances live in ``tolerances.py`` and were fixed before final reports were generated.
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..contracts import ExerciseStyle, OptionContract, OptionType
from ..engines import BinomialEngine, BlackScholesEngine, MonteCarloEngine
from ..engines.base import PricingEngine
from ..engines.binomial import crr_tree
from ..engines.black_scholes import bsm_arrays
from ..errors import InvalidInputError, NumericalError, UnsupportedInstrumentError
from ..generate import SAMPLE_VALUATION_DATE
from ..implied_vol import (
    IVStatus,
    implied_volatility,
    implied_volatility_arrays,
    implied_volatility_raw,
    price_bounds,
)
from ..market import MarketSnapshot
from . import tolerances as T
from .grid import Grid, validation_grid
from .quantlib_ref import QuantLibReference, quantlib_available

GREEKS = ("delta", "gamma", "vega", "theta", "rho")


@dataclass
class CheckResult:
    name: str
    title: str
    what: str  # what the check demonstrates
    tolerance: str
    status: str  # "passed" | "failed" | "skipped"
    n_cases: int = 0
    n_failed: int = 0
    max_abs_error: float = float("nan")
    max_utilisation: float = float("nan")  # max over cases of error / allowed error; <= 1 passes
    worst_case: str = ""
    notes: str = ""
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def _skipped(name: str, title: str, what: str, reason: str) -> CheckResult:
    return CheckResult(name, title, what, "n/a", "skipped", notes=reason)


def _compare(
    name: str,
    title: str,
    what: str,
    tol_text: str,
    values: np.ndarray,
    refs: np.ndarray,
    allowed: np.ndarray,
    describe: Callable[[int], str],
    notes: str = "",
    tables: dict[str, list[dict[str, Any]]] | None = None,
) -> CheckResult:
    """``allowed`` is the per-case permitted absolute error."""
    values, refs, allowed = (np.asarray(x, dtype=float) for x in (values, refs, allowed))
    err = np.abs(values - refs)
    bad = ~np.isfinite(err)
    util = np.where(bad, np.inf, err / allowed)
    worst = int(np.argmax(util)) if len(util) else 0
    n_failed = int((util > 1.0).sum())
    return CheckResult(
        name,
        title,
        what,
        tol_text,
        "passed" if n_failed == 0 and len(util) else "failed",
        n_cases=len(util),
        n_failed=n_failed,
        max_abs_error=float(np.nanmax(err)) if len(err) else float("nan"),
        max_utilisation=float(util.max()) if len(util) else float("nan"),
        worst_case=describe(worst) if len(util) else "",
        notes=notes,
        tables=tables or {},
    )


def _allowed(tol: T.Tol, refs: np.ndarray) -> np.ndarray:
    return tol.atol + tol.rtol * np.abs(refs)


def _bsm(grid: Grid, greeks: bool = True) -> dict[str, np.ndarray]:
    return bsm_arrays(grid.sign, grid.spot, grid.strike, grid.tau, grid.rate, grid.div, grid.vol, greeks=greeks)


# ------------------------------------------------------------------------------ analytic engine
def check_put_call_parity(grid: Grid) -> CheckResult:
    ones = np.ones(len(grid))
    args = (grid.spot, grid.strike, grid.tau, grid.rate, grid.div, grid.vol)
    call = bsm_arrays(ones, *args, greeks=False)["price"]
    put = bsm_arrays(-ones, *args, greeks=False)["price"]
    ref = grid.spot * np.exp(-grid.div * grid.tau) - grid.strike * np.exp(-grid.rate * grid.tau)
    res = _compare(
        "parity_bsm",
        "Put-call parity (analytic)",
        "Call and put implementations are mutually consistent: C - P = S e^{-qT} - K e^{-rT}.",
        T.PARITY.describe(),
        call - put,
        ref,
        _allowed(T.PARITY, ref),
        lambda i: grid.describe(i),
    )
    # same identity on the tree: it must hold to rounding because the tree is built to price the forward exactly
    sub = grid.every(8)
    diffs, refs2 = [], []
    for i in range(len(sub)):
        c = crr_tree(1, sub.spot[i], sub.strike[i], sub.tau[i], sub.rate[i], sub.div[i], sub.vol[i], 200).price
        p = crr_tree(-1, sub.spot[i], sub.strike[i], sub.tau[i], sub.rate[i], sub.div[i], sub.vol[i], 200).price
        diffs.append(c - p)
        refs2.append(sub.spot[i] * math.exp(-sub.div[i] * sub.tau[i]) - sub.strike[i] * math.exp(-sub.rate[i] * sub.tau[i]))
    tree = _compare(
        "parity_tree", "", "", "", np.array(diffs), np.array(refs2), _allowed(T.PARITY, np.array(refs2)),
        lambda i: sub.describe(i),
    )
    res.n_cases += tree.n_cases
    res.n_failed += tree.n_failed
    res.max_abs_error = max(res.max_abs_error, tree.max_abs_error)
    if tree.max_utilisation > res.max_utilisation:
        res.max_utilisation, res.worst_case = tree.max_utilisation, "tree: " + tree.worst_case
    res.status = "passed" if res.n_failed == 0 else "failed"
    res.notes = f"Analytic: {len(grid)} cases. CRR tree (200 steps): {tree.n_cases} cases."
    return res


def check_price_bounds(grid: Grid) -> CheckResult:
    price = _bsm(grid, greeks=False)["price"]
    lower = np.where(
        grid.sign == 1,
        np.maximum(grid.spot * np.exp(-grid.div * grid.tau) - grid.strike * np.exp(-grid.rate * grid.tau), 0.0),
        np.maximum(grid.strike * np.exp(-grid.rate * grid.tau) - grid.spot * np.exp(-grid.div * grid.tau), 0.0),
    )
    upper = np.where(grid.sign == 1, grid.spot * np.exp(-grid.div * grid.tau), grid.strike * np.exp(-grid.rate * grid.tau))
    violation = np.maximum(lower - price, 0.0) + np.maximum(price - upper, 0.0)
    res = _compare(
        "bounds_bsm",
        "No-arbitrage price bounds",
        "Prices lie inside the model bounds: max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT} (puts analogous), and are positive.",
        f"violation <= {T.BOUNDS_SLACK:g}",
        violation,
        np.zeros_like(violation),
        np.full_like(violation, T.BOUNDS_SLACK),
        lambda i: grid.describe(i),
    )
    # tree on a subset
    sub = grid.every(16)
    eng = BinomialEngine(200)
    viol = []
    for i in range(len(sub)):
        c, m = sub.contract_and_market(i)
        p = eng.price(c, m, greeks=False).price
        lo, hi = price_bounds(int(sub.sign[i]), m.spot, c.strike, sub.tau[i], m.rate, m.dividend_yield)
        viol.append(max(lo - p, 0.0) + max(p - hi, 0.0))
    n_tree_bad = int(sum(v > T.BOUNDS_SLACK for v in viol))
    res.n_cases += len(viol)
    res.n_failed += n_tree_bad
    res.status = "passed" if res.n_failed == 0 else "failed"
    res.notes = f"Analytic: {len(grid)} cases; CRR tree (200 steps): {len(viol)} cases, {n_tree_bad} violations."
    return res


def check_greeks_finite_difference(grid: Grid) -> CheckResult:
    s = T.FD_STEPS
    base = _bsm(grid)
    args = dict(sign=grid.sign, spot=grid.spot, strike=grid.strike, tau=grid.tau, rate=grid.rate, div=grid.div, vol=grid.vol)

    def price(**over: np.ndarray) -> np.ndarray:
        a = {**args, **over}
        return bsm_arrays(a["sign"], a["spot"], a["strike"], a["tau"], a["rate"], a["div"], a["vol"], greeks=False)["price"]

    # Richardson extrapolation of central differences, (4 D(h/2) - D(h)) / 3, cancels the O(h^2)
    # truncation term. A plain central difference was tried first and its error in the tails (deep
    # out-of-the-money delta) exceeded the tolerance while shrinking 4x per halving of h, i.e. the
    # reference, not the analytic Greek, was the inaccurate side.
    def richardson(first: Callable[[float], np.ndarray]) -> np.ndarray:
        return (4.0 * first(0.5) - first(1.0)) / 3.0

    width = grid.spot * grid.vol * np.sqrt(grid.tau)
    h = s["spot_width_fraction"] * width
    p0 = price()

    def d_spot(f: float) -> np.ndarray:
        return (price(spot=grid.spot + f * h) - price(spot=grid.spot - f * h)) / (2 * f * h)

    def d2_spot(f: float) -> np.ndarray:
        return (price(spot=grid.spot + f * h) - 2 * p0 + price(spot=grid.spot - f * h)) / ((f * h) ** 2)

    ht = s["time_fraction"] * grid.tau
    hv, hr = s["vol"], s["rate"]

    def d_vol(f: float) -> np.ndarray:
        return (price(vol=grid.vol + f * hv) - price(vol=grid.vol - f * hv)) / (2 * f * hv)

    def d_time(f: float) -> np.ndarray:
        return (price(tau=grid.tau + f * ht) - price(tau=grid.tau - f * ht)) / (2 * f * ht)

    def d_rate(f: float) -> np.ndarray:
        return (price(rate=grid.rate + f * hr) - price(rate=grid.rate - f * hr)) / (2 * f * hr)

    fd = {
        "delta": richardson(d_spot),
        "gamma": richardson(d2_spot),
        "vega": richardson(d_vol) * 0.01,
        "theta": -richardson(d_time) / 365.0,
        "rho": richardson(d_rate) * 0.01,
    }
    values = np.concatenate([base[g] for g in GREEKS])
    refs = np.concatenate([fd[g] for g in GREEKS])
    n = len(grid)
    return _compare(
        "greeks_fd",
        "Analytic Greeks vs finite differences",
        "Delta, gamma, vega, theta, rho (all in the library's reported units) match central finite differences of the price.",
        f"{T.FD_GREEK.describe()}; Richardson-extrapolated central differences, steps {s}",
        values,
        refs,
        _allowed(T.FD_GREEK, refs),
        lambda i: f"{GREEKS[i // n]}: {grid.describe(i % n)}",
        notes=f"{n} grid points x 5 Greeks. The finite-difference reference is itself only accurate to about its truncation error, which the tolerance accounts for.",
    )


# ------------------------------------------------------------------------------ implied volatility
def check_implied_vol(grid: Grid) -> CheckResult:
    price = _bsm(grid, greeks=False)["price"]
    n = len(grid)
    lower = np.empty(n)
    upper = np.empty(n)
    for i in range(n):
        lower[i], upper[i] = price_bounds(int(grid.sign[i]), grid.spot[i], grid.strike[i], grid.tau[i], grid.rate[i], grid.div[i])

    status_counts: dict[str, dict[str, int]] = {"bisection": {}, "brentq": {}}
    evals = {"bisection": [], "brentq": []}
    vols: dict[str, np.ndarray] = {m: np.full(n, np.nan) for m in status_counts}
    resid = {m: np.full(n, np.nan) for m in status_counts}
    unsolvable_ok = {m: True for m in status_counts}
    unsolvable_worst: dict[str, str] = {}
    for method in status_counts:
        for i in range(n):
            r = implied_volatility_raw(
                price[i], int(grid.sign[i]), grid.spot[i], grid.strike[i], grid.tau[i], grid.rate[i], grid.div[i], method=method
            )
            status_counts[method][r.status.value] = status_counts[method].get(r.status.value, 0) + 1
            if r.ok:
                vols[method][i] = r.vol
                resid[method][i] = abs(r.residual)
                evals[method].append(r.function_evals)
            else:
                near = (price[i] - lower[i] <= T.IV_ILLPOSED_SLACK) or (upper[i] - price[i] <= T.IV_ILLPOSED_SLACK)
                if not near:
                    unsolvable_ok[method] = False
                    unsolvable_worst[method] = grid.describe(i)

    vega_unit = np.where(
        np.isfinite(vols["bisection"]),
        bsm_arrays(grid.sign, grid.spot, grid.strike, grid.tau, grid.rate, grid.div, np.where(np.isfinite(vols["bisection"]), vols["bisection"], 0.2))["vega"] / 0.01,
        0.0,
    )
    conv = np.isfinite(vols["bisection"])
    well = conv & (vega_unit >= T.IV_MIN_VEGA)

    # (1) price round trip, both solvers, converged cases
    rt_tol = T.IV_PRICE_ROUNDTRIP
    rt_allowed = _allowed(rt_tol, price)
    worst_rt = max(
        float(np.nanmax(resid[m] / rt_allowed)) if np.isfinite(resid[m]).any() else 0.0 for m in status_counts
    )
    rt_fail = int(sum((np.nan_to_num(resid[m]) > rt_allowed).sum() for m in status_counts))
    # (2) vol recovery where well conditioned
    vol_err = np.abs(vols["bisection"] - grid.vol)[well]
    vol_err_b = np.abs(vols["brentq"] - grid.vol)[well]
    vol_fail = int((vol_err > T.IV_VOL_RECOVERY.atol).sum() + (vol_err_b > T.IV_VOL_RECOVERY.atol).sum())
    # (3) solver agreement
    agree = np.abs(vols["bisection"] - vols["brentq"])[well]
    agree_fail = int((agree > T.IV_BRENT_VS_BISECTION.atol).sum())
    # (4) vectorised
    v_vec, st = implied_volatility_arrays(price, grid.sign, grid.spot, grid.strike, grid.tau, grid.rate, grid.div)
    both = conv & (st == 0)
    # Volatilities are only comparable where vega is not tiny; on flat-price lanes any vol in the flat region
    # reproduces the price, so there the vectorised solver is judged by its price round trip instead.
    comparable = both & well
    vec_err = np.abs(v_vec - vols["bisection"])[comparable]
    vec_price = bsm_arrays(
        grid.sign, grid.spot, grid.strike, grid.tau, grid.rate, grid.div, np.where(both, v_vec, 0.2), greeks=False
    )["price"]
    vec_rt_err = np.abs(vec_price - price)[both]
    vec_rt_fail = int((vec_rt_err > rt_allowed[both]).sum())
    vec_fail = int((vec_err > T.IV_VECTOR_VS_SCALAR.atol).sum()) + vec_rt_fail
    n_flat = int((both & ~well).sum())

    failures = rt_fail + vol_fail + agree_fail + vec_fail + int(not all(unsolvable_ok.values()))
    n_unsolved = int((~conv).sum())
    avg = {m: (float(np.mean(evals[m])) if evals[m] else float("nan")) for m in evals}
    return CheckResult(
        "implied_vol",
        "Price -> implied volatility -> price round trip",
        "The solver recovers a volatility that reproduces the price; bisection, Brent and the vectorised solver agree; unsolvable quotes get explicit statuses.",
        f"price round trip {rt_tol.describe()}; vol {T.IV_VOL_RECOVERY.describe()} where vega>={T.IV_MIN_VEGA}; solver agreement {T.IV_BRENT_VS_BISECTION.atol:g}",
        "passed" if failures == 0 else "failed",
        n_cases=n,
        n_failed=failures,
        max_abs_error=float(vol_err.max()) if len(vol_err) else float("nan"),
        max_utilisation=worst_rt,
        worst_case="" if failures == 0 else "; ".join(f"{k}: {v}" for k, v in unsolvable_worst.items()) or "see counts",
        notes=(
            f"{n} grid prices solved by each method. Converged: {int(conv.sum())}; well-conditioned (vega>={T.IV_MIN_VEGA}): {int(well.sum())}; "
            f"max |vol error| on those: bisection {vol_err.max():.2e}, brentq {vol_err_b.max():.2e}; "
            f"max |bisection-brentq|: {agree.max():.2e}; vectorised vs scalar vol max diff {vec_err.max():.2e} on {int(comparable.sum())} well-conditioned lanes; "
            f"vectorised price round trip max error {vec_rt_err.max():.2e} on all {int(both.sum())} converged lanes "
            f"({n_flat} of them have vega < {T.IV_MIN_VEGA}, where the vol is not identifiable and only the price is compared). "
            f"Not solved: {n_unsolved} (all within {T.IV_ILLPOSED_SLACK:g} of a model bound: {all(unsolvable_ok.values())}). "
            "Max utilisation is for the price round trip."
        ),
        tables={
            "iv_solver_statuses": [
                {"method": m, **{k: v for k, v in sorted(status_counts[m].items())}, "mean_function_evals": avg[m]}
                for m in status_counts
            ]
        },
    )


# ------------------------------------------------------------------------------ binomial tree
def check_tree_vs_analytic(grid: Grid) -> CheckResult:
    sub = grid.every(4)
    steps = T.TREE_STEPS
    tree = np.array([crr_tree(int(sub.sign[i]), sub.spot[i], sub.strike[i], sub.tau[i], sub.rate[i], sub.div[i], sub.vol[i], steps).price for i in range(len(sub))])
    ref = _bsm(sub, greeks=False)["price"]
    allowed = np.array([T.tree_price_tol(sub.spot[i], sub.vol[i], sub.tau[i], steps).limit(ref[i]) for i in range(len(sub))])
    scaled = np.abs(tree - ref) / (sub.spot * sub.vol * np.sqrt(sub.tau) / steps)
    res = _compare(
        "tree_vs_analytic",
        f"CRR binomial ({steps} steps) vs analytic",
        "The European tree converges to Black-Scholes-Merton at the expected O(1/n) rate across the grid.",
        f"|tree-BSM| <= {T.TREE_PRICE_SCALE} * S*sigma*sqrt(T)/n + {T.TREE_PRICE_RTOL:g}*|BSM|",
        tree,
        ref,
        allowed,
        lambda i: sub.describe(i),
        notes=f"Every 4th grid point ({len(sub)} cases). Largest error in units of S*sigma*sqrt(T)/n: {scaled.max():.3f}; median {np.median(scaled):.3f}.",
        tables={"tree_convergence": _convergence_table()},
    )
    return res


def _convergence_table() -> list[dict[str, Any]]:
    rows = []
    cases = [("ATM call, T=1y, vol 20%", 1, 100.0, 1.0, 0.2), ("OTM put 90, T=0.25y, vol 30%", -1, 90.0, 0.25, 0.3), ("ITM call 85, T=2y, vol 40%", 1, 85.0, 2.0, 0.4)]
    from ..engines.black_scholes import bsm_scalar

    for label, sign, k, tau, vol in cases:
        ref = bsm_scalar(sign, 100.0, k, tau, 0.03, 0.01, vol)[0]
        for n in (10, 20, 50, 100, 200, 500, 1000, 1001, 2000, 4000):
            t0 = time.perf_counter()
            p = crr_tree(sign, 100.0, k, tau, 0.03, 0.01, vol, n).price
            dt = time.perf_counter() - t0
            rows.append({"case": label, "steps": n, "tree": p, "bsm": ref, "error": p - ref, "abs_error_x_n": abs(p - ref) * n, "seconds": dt})
    return rows


def check_tree_greeks(grid: Grid) -> CheckResult:
    sub = grid.every(14)
    eng = BinomialEngine(T.TREE_GREEK_STEPS)
    ref = _bsm(sub)
    got = {g: np.empty(len(sub)) for g in GREEKS}
    for i in range(len(sub)):
        c, m = sub.contract_and_market(i)
        gr = eng.price(c, m).greeks
        assert gr is not None
        for g in GREEKS:
            got[g][i] = getattr(gr, g)
    s, sig, tau = sub.spot, sub.vol, sub.tau
    # Natural scale of each Greek: the error is reported as a fraction of it.
    scale = {
        "delta": np.ones(len(sub)),
        "gamma": 1.0 / (s * sig * np.sqrt(tau)),
        "vega": 0.01 * s * np.sqrt(tau),
        "theta": s * sig / (365.0 * np.sqrt(tau)),
        "rho": 0.01 * s * tau,
    }
    n = len(sub)
    values = np.concatenate([got[g] for g in GREEKS])
    refs = np.concatenate([ref[g] for g in GREEKS])
    allowed = np.concatenate([T.TREE_GREEK_SCALED[g] * scale[g] for g in GREEKS])
    return _compare(
        "tree_greeks",
        f"CRR binomial Greeks ({T.TREE_GREEK_STEPS} steps) vs analytic",
        "Tree-derived delta/gamma/theta and bump-and-revalue vega/rho agree with analytic Greeks to a fraction of their natural scale.",
        f"|error| <= fraction * natural scale; fractions {T.TREE_GREEK_SCALED}",
        values,
        refs,
        allowed,
        lambda i: f"{GREEKS[i // n]}: {sub.describe(i % n)}",
        notes=f"{n} grid points x 5 Greeks.",
    )


def _american_cases() -> list[tuple[OptionContract, MarketSnapshot]]:
    vd = SAMPLE_VALUATION_DATE
    out = []
    for typ in (OptionType.PUT, OptionType.CALL):
        for r, q in ((0.02, 0.0), (0.05, 0.0), (0.05, 0.05), (0.02, 0.06)):
            for k in (90.0, 100.0, 110.0):
                c = OptionContract("AM", k, vd + timedelta(days=183), typ, ExerciseStyle.AMERICAN, f"AM{typ.value[0]}{k:g}r{r}q{q}")
                out.append((c, MarketSnapshot("AM", vd, 100.0, r, q, 0.30)))
    return out


def check_american() -> list[CheckResult]:
    cases = _american_cases()
    eng = BinomialEngine(1000)
    ame = np.array([eng.price(c, m, greeks=False).price for c, m in cases])
    eur = np.array([eng.price(OptionContract(c.underlying, c.strike, c.expiry, c.option_type, ExerciseStyle.EUROPEAN), m, greeks=False).price for c, m in cases])
    intrinsic = np.array([c.payoff(m.spot) for c, m in cases])
    below = np.maximum(eur - ame, 0.0) + np.maximum(intrinsic - ame, 0.0)
    internal = _compare(
        "american_internal",
        "American tree: dominance and no-early-exercise cases",
        "American value >= European value and >= intrinsic; an American call on a non-dividend underlying equals the European call.",
        "violation <= 1e-9",
        below,
        np.zeros_like(below),
        np.full_like(below, 1e-9),
        lambda i: cases[i][0].label,
    )
    nodiv_calls = [i for i, (c, m) in enumerate(cases) if c.is_call and m.dividend_yield == 0.0]
    diff = np.array([abs(ame[i] - eur[i]) for i in nodiv_calls])
    if len(diff) and diff.max() > 1e-9:
        internal.status = "failed"
        internal.n_failed += int((diff > 1e-9).sum())
    internal.n_cases += len(diff)
    internal.notes = f"{len(cases)} contracts (put/call, r,q variations); {len(diff)} non-dividend calls checked for equality with European (max diff {diff.max():.2e})."

    if not quantlib_available():
        return [internal, _skipped("american_quantlib", "American tree vs QuantLib", "", "QuantLib not installed")]
    ref = QuantLibReference()
    ql_tree = np.array([ref.tree_price(c, m, 1000) for c, m in cases])
    ql_fd = np.array([ref.finite_difference_price(c, m) for c, m in cases])
    scale = np.array([m.spot * m.volatility * math.sqrt(year_frac(c, m)) / 1000.0 for c, m in cases])
    r1 = _compare(
        "american_quantlib",
        "American CRR tree vs QuantLib",
        "American exercise (an extension beyond the core scope) agrees with QuantLib's CRR tree and its finite-difference engine.",
        f"vs QL tree: <= {T.AMERICAN_VS_QL_TREE_SCALE} * S*sigma*sqrt(T)/n; vs QL finite-difference: {T.AMERICAN_VS_QL_FD.describe()}",
        ame,
        ql_tree,
        T.AMERICAN_VS_QL_TREE_SCALE * scale + 1e-12,
        lambda i: cases[i][0].label,
    )
    fd = _compare("fd", "", "", "", ame, ql_fd, _allowed(T.AMERICAN_VS_QL_FD, ql_fd), lambda i: cases[i][0].label)
    r1.n_cases += fd.n_cases
    r1.n_failed += fd.n_failed
    if fd.max_utilisation > r1.max_utilisation:
        r1.max_utilisation, r1.worst_case = fd.max_utilisation, "vs finite-difference: " + fd.worst_case
    r1.status = "passed" if r1.n_failed == 0 else "failed"
    r1.notes = f"{len(cases)} contracts, two references. Max |tree - QL FD| = {np.abs(ame - ql_fd).max():.2e}; max |tree - QL tree| = {np.abs(ame - ql_tree).max():.2e}."
    return [internal, r1]


def year_frac(c: OptionContract, m: MarketSnapshot) -> float:
    return (c.expiry - m.valuation_date).days / 365.0


# ------------------------------------------------------------------------------ Monte Carlo
def _mc_subset(grid: Grid) -> Grid:
    keep = (grid.moneyness >= 0.85) & (grid.moneyness <= 1.15) & (grid.days >= 30) & (grid.vol >= 0.2)
    return grid.subset(np.flatnonzero(keep)).every(10)


def check_mc_accuracy(grid: Grid) -> CheckResult:
    sub = _mc_subset(grid)
    eng = MonteCarloEngine(n_paths=200_000, seed=2026, antithetic=True)
    ref = _bsm(sub)
    prices, ses, ref_p = [], [], []
    g_val = {g: [] for g in GREEKS}
    g_se = {g: [] for g in GREEKS}
    g_ref = {g: [] for g in GREEKS}
    for i in range(len(sub)):
        c, m = sub.contract_and_market(i)
        # distinct seed per contract so errors are not shared across the subset
        res = eng.with_settings(seed=2026 + i).price(c, m)
        prices.append(res.price)
        ses.append(res.std_error)
        ref_p.append(ref["price"][i])
        for g in GREEKS:
            g_val[g].append(getattr(res.greeks, g))
            g_se[g].append(res.diagnostics["greeks_std_error"][g])
            g_ref[g].append(ref[g][i])
    n = len(sub)
    values = np.concatenate([prices] + [g_val[g] for g in GREEKS])
    refs = np.concatenate([ref_p] + [g_ref[g] for g in GREEKS])
    se = np.concatenate([ses] + [g_se[g] for g in GREEKS])
    names = ("price",) + GREEKS
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (values - refs) / se
    # A standard error of ~0 is legitimate, not a bug: e.g. with zero risk-neutral drift (r - q = sigma^2/2) and
    # K = S, antithetic pairs make the pathwise rho estimator exactly constant. The z-score is then rounding noise
    # divided by rounding noise, so such cases are excluded from the z summary (they still must pass the absolute test).
    degenerate = se < 1e-10 * (1.0 + np.abs(refs))
    zq = np.where(degenerate, np.nan, z)
    per_quantity = [
        {
            "quantity": name,
            "mean_z": float(np.nanmean(zq[j * n : (j + 1) * n])),
            "std_z": float(np.nanstd(zq[j * n : (j + 1) * n])),
            "max_abs_z": float(np.nanmax(np.abs(zq[j * n : (j + 1) * n]))),
            "zero_variance_cases": int(degenerate[j * n : (j + 1) * n].sum()),
        }
        for j, name in enumerate(names)
    ]
    result = _compare(
        "mc_accuracy",
        "Monte Carlo vs analytic, in standard errors",
        "MC price and Greek estimates are statistically consistent with the analytic values; error is judged in standard errors, not currency.",
        f"|MC - BSM| <= {T.MC_Z:g} * standard_error",
        values,
        refs,
        T.MC_Z * se + T.MC_ATOL,
        lambda i: f"{names[i // n]}: {sub.describe(i % n)} (z={z[i]:+.2f})",
        notes=(
            f"{n} contracts x 6 quantities, 200,000 paths, antithetic. Over the {int((~degenerate).sum())} cases with non-degenerate standard error: "
            f"mean z = {np.nanmean(zq):+.3f}, std of z = {np.nanstd(zq):.3f}, max |z| = {np.nanmax(np.abs(zq)):.2f} "
            "(a correct estimator with correct standard errors gives about 0, 1 and 3 to 3.5 for ~1000 samples). "
            f"{int(degenerate.sum())} case(s) have a zero-variance estimator (exact result) and are judged by the absolute floor only."
        ),
        tables={"mc_accuracy_z_by_quantity": per_quantity},
    )
    return result


_REPEAT_CASES = [
    # (label, sign, strike, days, vol, rate, div)
    ("ATM call 90d", 1, 100.0, 90, 0.25, 0.03, 0.01),
    ("ATM put 90d", -1, 100.0, 90, 0.25, 0.03, 0.01),
    ("ITM call 1y", 1, 90.0, 365, 0.30, 0.04, 0.02),
    ("ITM put 1y", -1, 110.0, 365, 0.30, 0.04, 0.02),
    ("OTM call 6m", 1, 110.0, 180, 0.20, 0.02, 0.0),
    ("OTM put 6m", -1, 90.0, 180, 0.20, 0.02, 0.0),
    ("ATM call 7d", 1, 100.0, 7, 0.20, 0.03, 0.0),
    ("OTM call 1y high vol", 1, 130.0, 365, 0.60, 0.03, 0.0),
]


def _repeat_stats(n_paths: int, n_seeds: int, antithetic: bool, n_cases: int | None = None) -> list[dict[str, Any]]:
    out = []
    # Spread the reduced case set across the list (calls, puts, ITM/ATM/OTM) instead of taking the first few.
    cases = _REPEAT_CASES if n_cases is None else [_REPEAT_CASES[i] for i in np.linspace(0, len(_REPEAT_CASES) - 1, n_cases).round().astype(int)]
    for label, sign, k, days, vol, r, q in cases:
        vd = SAMPLE_VALUATION_DATE
        contract = OptionContract("REP", k, vd + timedelta(days=days), OptionType.CALL if sign == 1 else OptionType.PUT)
        market = MarketSnapshot("REP", vd, 100.0, r, q, vol)
        ref = BlackScholesEngine().price(contract, market, greeks=False).price
        est, se, cover, times = [], [], 0, []
        for s in range(n_seeds):
            res = MonteCarloEngine(n_paths=n_paths, seed=s, antithetic=antithetic).price(contract, market, greeks=False)
            est.append(res.price)
            se.append(res.std_error)
            lo, hi = res.confidence_interval
            cover += lo <= ref <= hi
            times.append(res.runtime_s)
        est_a, se_a = np.array(est), np.array(se)
        out.append(
            {
                "case": label,
                "reference": ref,
                "mean_estimate": float(est_a.mean()),
                "bias_z": float((est_a.mean() - ref) / (est_a.std(ddof=1) / math.sqrt(n_seeds))),
                "empirical_std": float(est_a.std(ddof=1)),
                "mean_reported_se": float(se_a.mean()),
                "se_ratio": float(se_a.mean() / est_a.std(ddof=1)),
                "coverage_95": cover / n_seeds,
                "estimator_variance": float(est_a.var(ddof=1)),
                "mean_reported_variance": float((se_a**2).mean()),
                "median_runtime_s": float(np.median(times)),
            }
        )
    return out


def check_mc_repeated_seeds(n_seeds: int = 1000, n_paths: int = 20_000, n_cases: int | None = None) -> list[CheckResult]:
    anti = _repeat_stats(n_paths, n_seeds, True, n_cases)
    plain = _repeat_stats(n_paths, n_seeds, False, n_cases)
    cov = np.array([r["coverage_95"] for r in anti])
    pooled = float(cov.mean())
    se_ratio = np.array([r["se_ratio"] for r in anti])
    bias_z = np.array([r["bias_z"] for r in anti])
    single_bad = int(((cov < T.MC_COVERAGE_SINGLE[0]) | (cov > T.MC_COVERAGE_SINGLE[1])).sum())
    se_bad = int(((se_ratio < T.MC_SE_RATIO[0]) | (se_ratio > T.MC_SE_RATIO[1])).sum())
    bias_bad = int((np.abs(bias_z) > T.MC_BIAS_Z).sum())
    pooled_bad = int(not (T.MC_COVERAGE_POOLED[0] <= pooled <= T.MC_COVERAGE_POOLED[1]))
    fails = single_bad + se_bad + bias_bad + pooled_bad
    stats = CheckResult(
        "mc_repeated_seeds",
        "Monte Carlo behaviour across repeated seeds",
        "Across independent seeds the estimator is unbiased, the 95% interval covers the true price about 95% of the time, and the reported standard error matches the empirical spread.",
        f"pooled coverage in {T.MC_COVERAGE_POOLED}; per-case coverage in {T.MC_COVERAGE_SINGLE}; mean SE / empirical std in {T.MC_SE_RATIO}; |bias z| <= {T.MC_BIAS_Z:g}",
        "passed" if fails == 0 else "failed",
        n_cases=len(anti) * 3,
        n_failed=fails,
        max_abs_error=float(np.max(np.abs([r["mean_estimate"] - r["reference"] for r in anti]))),
        max_utilisation=float("nan"),
        worst_case="" if fails == 0 else "see table",
        notes=f"{len(anti)} contracts x {n_seeds} seeds x {n_paths:,} paths (antithetic). Pooled 95% coverage: {pooled:.3f}.",
        tables={"mc_repeated_seed_stats_antithetic": anti},
    )
    ratio_rows = []
    for a, p in zip(anti, plain):
        ratio_rows.append(
            {
                "case": a["case"],
                "empirical_variance_plain": p["estimator_variance"],
                "empirical_variance_antithetic": a["estimator_variance"],
                "empirical_ratio": p["estimator_variance"] / a["estimator_variance"],
                "variance_ratio_plain_over_antithetic": p["mean_reported_variance"] / a["mean_reported_variance"],
                "se_plain_reported": p["mean_reported_se"],
                "se_antithetic_reported": a["mean_reported_se"],
                "median_runtime_plain_s": p["median_runtime_s"],
                "median_runtime_antithetic_s": a["median_runtime_s"],
                "plain_coverage_95": p["coverage_95"],
                "antithetic_coverage_95": a["coverage_95"],
            }
        )
    ratios = np.array([r["variance_ratio_plain_over_antithetic"] for r in ratio_rows])
    vr_fail = int((ratios < T.MC_VR_MIN_RATIO).sum())
    vr = CheckResult(
        "mc_variance_reduction",
        "Antithetic variates at equal simulation budget",
        "At the same number of simulated terminal prices, antithetic variates do not increase estimator variance, and usually reduce it.",
        f"plain variance / antithetic variance >= {T.MC_VR_MIN_RATIO} (variances = mean reported SE^2 over seeds)",
        "passed" if vr_fail == 0 else "failed",
        n_cases=len(ratios),
        n_failed=vr_fail,
        worst_case=min(ratio_rows, key=lambda r: r["variance_ratio_plain_over_antithetic"])["case"],
        notes=(
            f"Same {n_paths:,} paths per estimate, {n_seeds} seeds each. Variance ratio (plain/antithetic, from mean reported SE^2): "
            f"min {ratios.min():.2f}, median {np.median(ratios):.2f}, max {ratios.max():.2f}. The 'empirical_ratio' column uses the spread of "
            "estimates across seeds instead; it is much noisier for skewed payoffs (high-vol OTM options), so it is shown but not used for pass/fail. "
            "Max error / allowed column shows the minimum ratio."
        ),
        tables={"mc_variance_reduction": ratio_rows},
    )
    vr.max_utilisation = float(ratios.min())
    return [stats, vr]


def check_mc_reproducibility() -> CheckResult:
    vd = SAMPLE_VALUATION_DATE
    contract = OptionContract("REP", 105.0, vd + timedelta(days=120), OptionType.CALL)
    market = MarketSnapshot("REP", vd, 100.0, 0.03, 0.01, 0.25)
    notes, fails = [], 0

    def record(label: str, ok: bool) -> None:
        nonlocal fails
        fails += int(not ok)
        notes.append(f"{label}: {'ok' if ok else 'FAILED'}")

    a = MonteCarloEngine(50_000, seed=7).price(contract, market)
    b = MonteCarloEngine(50_000, seed=7).price(contract, market)
    c = MonteCarloEngine(50_000, seed=8).price(contract, market)
    record("same seed -> identical price and Greeks", a.price == b.price and a.greeks == b.greeks)
    record("different seed -> different price", a.price != c.price)
    prices = {}
    for w in (1, 2, 4):
        with MonteCarloEngine(50_000, seed=7, n_workers=w) as eng:
            prices[w] = eng.price(contract, market)
    record("1, 2 and 4 workers -> bitwise identical price and Greeks", all(prices[w].price == a.price and prices[w].greeks == a.greeks for w in prices))
    ss = np.random.SeedSequence(7).spawn(2)
    x, y = (np.random.default_rng(s).standard_normal(200_000) for s in ss)
    corr = float(np.corrcoef(x, y)[0, 1])
    record(f"spawned streams uncorrelated (|corr|={abs(corr):.4f} <= {4 / math.sqrt(len(x)):.4f})", abs(corr) <= 4 / math.sqrt(len(x)))
    x2, y2 = (np.random.default_rng(7).standard_normal(1000) for _ in range(2))
    record("(contrast) naive same-seed workers would be identical streams", bool(np.array_equal(x2, y2)))
    return CheckResult(
        "mc_reproducibility",
        "Monte Carlo reproducibility and independent streams",
        "Results are reproducible from a seed and independent of worker count; parallel workers use spawned independent streams, not repeated seeds.",
        "exact equality / |corr| <= 4/sqrt(n)",
        "passed" if fails == 0 else "failed",
        n_cases=len(notes),
        n_failed=fails,
        notes="; ".join(notes),
    )


# ------------------------------------------------------------------------------ QuantLib
def check_quantlib_bsm(grid: Grid) -> CheckResult:
    if not quantlib_available():
        return _skipped("quantlib_bsm", "Analytic engine vs QuantLib", "", "QuantLib not installed")
    ref = QuantLibReference()
    ours = _bsm(grid)
    names = ("price",) + GREEKS
    theirs = {k: np.empty(len(grid)) for k in names}
    for i in range(len(grid)):
        c, m = grid.contract_and_market(i)
        r = ref.european_analytic(c, m)
        for k in names:
            theirs[k][i] = r[k]
    n = len(grid)
    values = np.concatenate([ours[k] for k in names])
    refs = np.concatenate([theirs[k] for k in names])
    allowed = np.concatenate([_allowed(T.QL_BSM_PRICE, theirs["price"])] + [_allowed(T.QL_BSM_GREEK, theirs[g]) for g in GREEKS])
    per_quantity = []
    for j, k in enumerate(names):
        e = np.abs(ours[k] - theirs[k])
        rel = e / np.maximum(np.abs(theirs[k]), 1e-12)
        per_quantity.append({"quantity": k, "max_abs_error": float(e.max()), "max_rel_error_where_ref_gt_1e-6": float(rel[np.abs(theirs[k]) > 1e-6].max(initial=0.0)), "cases": n})
    return _compare(
        "quantlib_bsm",
        "Analytic price and Greeks vs QuantLib AnalyticEuropeanEngine",
        "Independent reference validation with matching conventions (Actual365Fixed, NullCalendar, flat continuous curves), full grid.",
        f"price {T.QL_BSM_PRICE.describe()}; Greeks {T.QL_BSM_GREEK.describe()}",
        values,
        refs,
        allowed,
        lambda i: f"{names[i // n]}: {grid.describe(i % n)}",
        notes=f"{n} contracts x 6 quantities = {6 * n} comparisons.",
        tables={"quantlib_bsm_by_quantity": per_quantity},
    )


def check_quantlib_tree(grid: Grid) -> CheckResult:
    if not quantlib_available():
        return _skipped("quantlib_tree", "Tree vs QuantLib", "", "QuantLib not installed")
    ref = QuantLibReference()
    sub = grid.every(14)
    steps = T.TREE_GREEK_STEPS
    ours = np.empty(len(sub))
    theirs = np.empty(len(sub))
    for i in range(len(sub)):
        c, m = sub.contract_and_market(i)
        ours[i] = crr_tree(int(sub.sign[i]), m.spot, c.strike, sub.tau[i], m.rate, m.dividend_yield, m.volatility, steps).price
        theirs[i] = ref.tree_price(c, m, steps, "crr")
    scale = sub.spot * sub.vol * np.sqrt(sub.tau) / steps
    return _compare(
        "quantlib_tree",
        "CRR tree vs QuantLib BinomialVanillaEngine('crr')",
        "Tree implementation vs QuantLib's CRR tree at equal step count. QuantLib's 'crr' uses the same up/down factors but sets the up-probability from the log-drift, p = 1/2 + (r - q - sigma^2/2) sqrt(dt) / (2 sigma), whereas this tree sets p so the one-step mean equals the forward exactly. Both converge to the same limit, so exact equality at finite n is not expected; the difference should be well below the tree's own error.",
        f"|diff| <= {T.TREE_VS_QL_TREE_SCALE} * S*sigma*sqrt(T)/n",
        ours,
        theirs,
        T.TREE_VS_QL_TREE_SCALE * scale + 1e-12,
        lambda i: sub.describe(i),
        notes=f"{len(sub)} contracts, {steps} steps.",
    )


def check_quantlib_mc(grid: Grid) -> CheckResult:
    if not quantlib_available():
        return _skipped("quantlib_mc", "Monte Carlo vs QuantLib", "", "QuantLib not installed")
    ref = QuantLibReference()
    sub = _mc_subset(grid).every(3)
    vals, refs, ses = [], [], []
    for i in range(len(sub)):
        c, m = sub.contract_and_market(i)
        res = MonteCarloEngine(n_paths=200_000, seed=99 + i).price(c, m, greeks=False)
        qlp = ref.european_analytic(c, m)["price"]
        vals.append(res.price)
        refs.append(qlp)
        ses.append(res.std_error)
    vals, refs, ses = map(np.array, (vals, refs, ses))
    z = (vals - refs) / ses
    return _compare(
        "quantlib_mc",
        "Monte Carlo price vs QuantLib analytic reference",
        "Statistical consistency of MC prices with the independent QuantLib price.",
        f"|MC - QL| <= {T.MC_Z:g} * standard_error",
        vals,
        refs,
        T.MC_Z * ses + T.MC_ATOL,
        lambda i: f"{sub.describe(i)} (z={z[i]:+.2f})",
        notes=f"{len(sub)} contracts, 200,000 antithetic paths. Mean z={z.mean():+.3f}, std z={z.std():.3f}.",
    )


# ------------------------------------------------------------------------------ invalid input / boundaries
def check_invalid_inputs() -> CheckResult:
    vd = SAMPLE_VALUATION_DATE
    good_c = OptionContract("X", 100.0, vd + timedelta(days=90), "call")
    good_m = MarketSnapshot("X", vd, 100.0, 0.03, 0.01, 0.2)
    bsm, tree, mc = BlackScholesEngine(), BinomialEngine(100), MonteCarloEngine(1000, seed=1)

    cases: list[tuple[str, Callable[[], Any], type | None]] = []

    def raises(label: str, fn: Callable[[], Any], exc: type) -> None:
        cases.append((label, fn, exc))

    raises("strike = 0", lambda: OptionContract("X", 0.0, vd, "call"), InvalidInputError)
    raises("strike negative", lambda: OptionContract("X", -5.0, vd, "call"), InvalidInputError)
    raises("strike NaN", lambda: OptionContract("X", float("nan"), vd, "call"), InvalidInputError)
    raises("strike not a number", lambda: OptionContract("X", "abc", vd, "call"), InvalidInputError)  # type: ignore[arg-type]
    raises("unknown option type", lambda: OptionContract("X", 100.0, vd, "straddle"), InvalidInputError)
    raises("expiry is a datetime, not a date", lambda: OptionContract("X", 100.0, __import__("datetime").datetime(2026, 6, 1), "call"), InvalidInputError)
    raises("empty underlying", lambda: OptionContract("", 100.0, vd, "call"), InvalidInputError)
    raises("spot = 0", lambda: MarketSnapshot("X", vd, 0.0, 0.03, 0.0, 0.2), InvalidInputError)
    raises("spot negative", lambda: MarketSnapshot("X", vd, -1.0, 0.03, 0.0, 0.2), InvalidInputError)
    raises("volatility = 0", lambda: MarketSnapshot("X", vd, 100.0, 0.03, 0.0, 0.0), InvalidInputError)
    raises("volatility negative", lambda: MarketSnapshot("X", vd, 100.0, 0.03, 0.0, -0.2), InvalidInputError)
    raises("volatility given in percent (25)", lambda: MarketSnapshot("X", vd, 100.0, 0.03, 0.0, 25.0), InvalidInputError)
    raises("rate given in percent (5)", lambda: MarketSnapshot("X", vd, 100.0, 5.0, 0.0, 0.2), InvalidInputError)
    raises("rate NaN", lambda: MarketSnapshot("X", vd, 100.0, float("nan"), 0.0, 0.2), InvalidInputError)
    raises("dividend yield inf", lambda: MarketSnapshot("X", vd, 100.0, 0.03, float("inf"), 0.2), InvalidInputError)
    expired_c = OptionContract("X", 100.0, vd - timedelta(days=1), "call")
    for eng in (bsm, tree, mc):
        raises(f"valuation after expiry ({eng.name})", lambda eng=eng: eng.price(expired_c, good_m), InvalidInputError)
    raises("underlying mismatch", lambda: bsm.price(OptionContract("Y", 100.0, vd + timedelta(days=9), "call"), good_m), InvalidInputError)
    am = OptionContract("X", 100.0, vd + timedelta(days=90), "put", "american")
    raises("American option on analytic engine", lambda: bsm.price(am, good_m), UnsupportedInstrumentError)
    raises("American option on Monte Carlo engine", lambda: mc.price(am, good_m), UnsupportedInstrumentError)
    raises("tree with 2 steps", lambda: BinomialEngine(2), InvalidInputError)
    raises("tree steps not an integer", lambda: BinomialEngine(100.5), InvalidInputError)  # type: ignore[arg-type]
    extreme_m = MarketSnapshot("X", vd, 100.0, 0.50, 0.0, 0.01)
    raises("tree where risk-neutral p leaves (0,1)", lambda: BinomialEngine(5).price(OptionContract("X", 100.0, vd + timedelta(days=730), "call"), extreme_m), NumericalError)
    raises("MC odd n_paths with antithetic", lambda: MonteCarloEngine(1001, antithetic=True), InvalidInputError)
    raises("MC n_paths = 1", lambda: MonteCarloEngine(1, antithetic=False, n_streams=1).price(good_c, good_m), InvalidInputError)
    raises("MC negative seed", lambda: MonteCarloEngine(1000, seed=-1), InvalidInputError)
    raises("MC confidence = 1.5", lambda: MonteCarloEngine(1000, confidence=1.5), InvalidInputError)
    raises("scenario with spot shock <= -100%", lambda: good_m.shocked(spot_pct=-1.0), InvalidInputError)

    # solver outcomes are statuses, not exceptions
    status_cases = []
    tau = 0.25
    lower, upper = price_bounds(1, 100.0, 90.0, tau, 0.03, 0.01)
    status_cases += [
        ("IV: price below intrinsic lower bound", implied_volatility_raw(lower * 0.5, 1, 100.0, 90.0, tau, 0.03, 0.01), IVStatus.PRICE_BELOW_LOWER_BOUND),
        ("IV: price equal to lower bound", implied_volatility_raw(lower, 1, 100.0, 90.0, tau, 0.03, 0.01), IVStatus.PRICE_BELOW_LOWER_BOUND),
        ("IV: price above upper bound", implied_volatility_raw(upper + 1.0, 1, 100.0, 90.0, tau, 0.03, 0.01), IVStatus.PRICE_ABOVE_UPPER_BOUND),
        ("IV: negative price", implied_volatility_raw(-1.0, 1, 100.0, 90.0, tau, 0.03, 0.01), IVStatus.INVALID_INPUT),
        ("IV: NaN price", implied_volatility_raw(float("nan"), 1, 100.0, 90.0, tau, 0.03, 0.01), IVStatus.INVALID_INPUT),
        ("IV: expired option", implied_volatility_raw(1.0, 1, 100.0, 90.0, 0.0, 0.03, 0.01), IVStatus.INVALID_INPUT),
        ("IV: unknown method", implied_volatility_raw(1.0, 1, 100.0, 90.0, tau, 0.03, 0.01, method="newton"), IVStatus.INVALID_INPUT),
        ("IV: American contract", implied_volatility(5.0, am, good_m), IVStatus.INVALID_INPUT),
        # Inside the model bounds, but needs a volatility above the solver maximum (10 = 1000%), even after widening.
        ("IV: quote needs vol above solver maximum", implied_volatility_raw(100.0 * (1 - 1e-12), 1, 100.0, 100.0, 0.1, 0.0, 0.0, vol_hi=1.0), IVStatus.BRACKET_FAILED),
        ("IV: iteration cap hit", implied_volatility_raw(6.0, 1, 100.0, 100.0, 1.0, 0.03, 0.0, max_iter=3), IVStatus.MAX_ITERATIONS),
    ]

    # loader errors
    loader_cases: list[tuple[str, Callable[[], Any]]] = []
    from .. import io as oio

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "c_missing_col.csv").write_text("contract_id,underlying,option_type,strike\nA,X,call,100\n")
        (d / "c_bad_date.csv").write_text("contract_id,underlying,option_type,strike,expiry\nA,X,call,100,2026-13-45\n")
        (d / "c_dup.csv").write_text("contract_id,underlying,option_type,strike,expiry\nA,X,call,100,2026-06-01\nA,X,put,100,2026-06-01\n")
        (d / "c_neg_strike.csv").write_text("contract_id,underlying,option_type,strike,expiry\nA,X,call,-100,2026-06-01\n")
        (d / "c_ok.csv").write_text("contract_id,underlying,option_type,strike,expiry\nA,X,call,100,2026-06-01\n")
        (d / "p_unknown.csv").write_text("contract_id,quantity\nZ,1\n")
        (d / "m_pct.csv").write_text("underlying,valuation_date,spot,rate,dividend_yield,volatility\nX,2026-01-05,100,0.03,0,25\n")
        (d / "bad.json").write_text("{not json")
        loader_cases = [
            ("loader: missing column", lambda: oio.load_contracts(d / "c_missing_col.csv")),
            ("loader: malformed date", lambda: oio.load_contracts(d / "c_bad_date.csv")),
            ("loader: duplicate contract id", lambda: oio.load_contracts(d / "c_dup.csv")),
            ("loader: negative strike (row reported)", lambda: oio.load_contracts(d / "c_neg_strike.csv")),
            ("loader: unknown contract id in positions", lambda: oio.load_positions(d / "p_unknown.csv", oio.load_contracts(d / "c_ok.csv"))),
            ("loader: volatility in percent", lambda: oio.load_markets(d / "m_pct.csv")),
            ("loader: invalid JSON", lambda: oio.read_records(d / "bad.json", "contracts")),
            ("loader: file not found", lambda: oio.load_markets(d / "nope.csv")),
        ]
        loader_results = []
        for label, fn in loader_cases:
            try:
                fn()
                loader_results.append((label, False, "no error raised"))
            except InvalidInputError as exc:
                loader_results.append((label, str(d) in str(exc), str(exc).replace(str(d), "<dir>")))

    rows: list[dict[str, Any]] = []
    fails = 0
    for label, fn, exc in cases:
        try:
            fn()
            ok, got = False, "no error raised"
        except exc as e:  # type: ignore[misc]
            ok, got = True, f"{type(e).__name__}"
        except Exception as e:  # noqa: BLE001 - recording the wrong type is the point
            ok, got = False, f"wrong error type {type(e).__name__}"
        fails += int(not ok)
        rows.append({"case": label, "expected": exc.__name__, "result": got, "ok": ok})
    for label, res, expected in status_cases:
        ok = res.status is expected and math.isnan(res.vol)
        fails += int(not ok)
        rows.append({"case": label, "expected": expected.value, "result": res.status.value, "ok": ok})
    for label, ok, msg in loader_results:
        fails += int(not ok)
        rows.append({"case": label, "expected": "InvalidInputError naming the file", "result": msg[:110], "ok": ok})
    return CheckResult(
        "invalid_inputs",
        "Invalid-input handling",
        "Bad inputs fail loudly with a specific error type (library calls), an explicit status (implied-vol solver), or a message naming the file and row (loaders).",
        "expected exception type / status for every case",
        "passed" if fails == 0 else "failed",
        n_cases=len(rows),
        n_failed=fails,
        worst_case="" if fails == 0 else next(r["case"] for r in rows if not r["ok"]),
        tables={"invalid_input_cases": rows},
    )


def check_expiry_boundaries() -> CheckResult:
    vd = SAMPLE_VALUATION_DATE
    fails, rows = 0, []
    engines: list[PricingEngine] = [BlackScholesEngine(), BinomialEngine(50), MonteCarloEngine(1000, seed=1)]
    for typ in ("call", "put"):
        for spot in (110.0, 100.0, 90.0):
            c = OptionContract("X", 100.0, vd, typ)
            m = MarketSnapshot("X", vd, spot, 0.03, 0.01, 0.3)
            intrinsic = c.payoff(spot)
            label = "ITM" if intrinsic > 0 else ("ATM" if spot == c.strike else "OTM")  # moneyness depends on option type
            exp_delta = (1.0 if typ == "call" else -1.0) if intrinsic > 0 else 0.0
            for eng in engines:
                r = eng.price(c, m)
                g = r.greeks
                ok = (
                    r.price == intrinsic
                    and g is not None
                    and g.delta == exp_delta
                    and (g.gamma, g.vega, g.theta, g.rho) == (0.0, 0.0, 0.0, 0.0)
                    and r.diagnostics.get("expired") is True
                    and (not eng.statistical or r.std_error == 0.0)
                )
                fails += int(not ok)
                rows.append({"case": f"expiry today, {label} {typ}, {eng.name}", "price": r.price, "expected": intrinsic, "delta": getattr(g, "delta", None), "ok": ok})
    # one day before expiry should be close to (and consistent with) intrinsic for deep ITM / OTM
    for typ, spot in (("call", 150.0), ("call", 60.0), ("put", 60.0), ("put", 150.0)):
        c = OptionContract("X", 100.0, vd + timedelta(days=1), typ)
        m = MarketSnapshot("X", vd, spot, 0.03, 0.0, 0.3)
        r = BlackScholesEngine().price(c, m)
        ok = abs(r.price - c.payoff(spot)) < 0.01
        fails += int(not ok)
        rows.append({"case": f"1 day to expiry, deep {'ITM' if c.payoff(spot) > 0 else 'OTM'} {typ} S={spot:g}", "price": r.price, "expected": c.payoff(spot), "delta": r.greeks.delta, "ok": ok})
    return CheckResult(
        "expiry_boundaries",
        "Expired and near-expiry options",
        "At expiry every engine returns intrinsic value, step-function delta and zero other Greeks; one day before expiry deep ITM/OTM prices are within 0.01 of intrinsic.",
        "exact equality at expiry; 0.01 one day out",
        "passed" if fails == 0 else "failed",
        n_cases=len(rows),
        n_failed=fails,
        tables={"expiry_boundary_cases": rows},
    )


# ------------------------------------------------------------------------------ risk engine
class _ScalarOnlyBSM(PricingEngine):
    """BSM that hides ``price_arrays``, forcing RiskEngine onto its position-by-position route."""

    name = "bsm_scalar_route"

    def __init__(self) -> None:
        self._inner = BlackScholesEngine()

    def _price(self, contract, market, tau, greeks):  # type: ignore[no-untyped-def]
        return self._inner._price(contract, market, tau, greeks)


def check_risk_routes() -> CheckResult:
    from ..generate import random_portfolio, sample_markets, sample_portfolio
    from ..risk import RiskEngine, default_scenarios

    fails, notes = 0, []
    worst = 0.0
    for label, (pf, mk) in {"sample": (sample_portfolio(), sample_markets()), "random-300": random_portfolio(300, 6, seed=5)}.items():
        scenarios = default_scenarios()
        fast = RiskEngine(BlackScholesEngine()).run_scenarios(pf, mk, scenarios)
        slow = RiskEngine(_ScalarOnlyBSM()).run_scenarios(pf, mk, scenarios)
        diffs = np.array([[abs(a.full_pnl - b.full_pnl), abs(a.approx_pnl - b.approx_pnl)] for a, b in zip(fast, slow)])
        scale = max(abs(a.base_value) for a in fast) + 1.0
        rel = float(diffs.max() / scale)
        worst = max(worst, rel)
        bad = rel > 1e-11
        fails += int(bad)
        notes.append(f"{label}: {len(scenarios)} scenarios, max route difference / portfolio value = {rel:.1e}")
    return CheckResult(
        "risk_routes",
        "Portfolio scenario revaluation: vectorised vs position-by-position",
        "The fast (vectorised) and generic (engine-agnostic) revaluation routes produce the same scenario P&L, including expiry inside the horizon.",
        "max |difference| / portfolio value <= 1e-11",
        "passed" if fails == 0 else "failed",
        n_cases=2 * len(default_scenarios()),
        n_failed=fails,
        max_utilisation=worst / 1e-11,
        notes="; ".join(notes),
    )


# ------------------------------------------------------------------------------ driver
def run_all(quick: bool = False, progress: Callable[[str], None] | None = None) -> tuple[list[CheckResult], Grid]:
    grid = validation_grid(quick=quick)
    say = progress or (lambda s: None)
    results: list[CheckResult] = []

    def run(label: str, fn: Callable[[], CheckResult | list[CheckResult]]) -> None:
        say(f"  {label} ...")
        t0 = time.perf_counter()
        out = fn()
        for r in out if isinstance(out, list) else [out]:
            r.notes = (r.notes + f" [{time.perf_counter() - t0:.1f}s]").strip() if r.status != "skipped" else r.notes
            results.append(r)

    run("put-call parity", lambda: check_put_call_parity(grid))
    run("price bounds", lambda: check_price_bounds(grid))
    run("Greeks vs finite differences", lambda: check_greeks_finite_difference(grid))
    run("implied volatility", lambda: check_implied_vol(grid))
    run("tree vs analytic", lambda: check_tree_vs_analytic(grid))
    run("tree Greeks", lambda: check_tree_greeks(grid))
    run("American exercise", check_american)
    run("Monte Carlo accuracy", lambda: check_mc_accuracy(grid))
    run("Monte Carlo repeated seeds", lambda: check_mc_repeated_seeds(600, n_cases=4) if quick else check_mc_repeated_seeds(1000))
    run("Monte Carlo reproducibility", check_mc_reproducibility)
    run("QuantLib analytic", lambda: check_quantlib_bsm(grid))
    run("QuantLib tree", lambda: check_quantlib_tree(grid))
    run("QuantLib Monte Carlo", lambda: check_quantlib_mc(grid))
    run("invalid inputs", check_invalid_inputs)
    run("expiry boundaries", check_expiry_boundaries)
    run("risk engine routes", check_risk_routes)
    return results, grid

"""Pass/fail tolerances for the validation suite.

These were fixed BEFORE the final reports were generated. A value passes when

    |value - reference| <= atol + rtol * |reference|

Both terms are used: ``rtol`` alone fails for reference prices near zero (deep out-of-the-money,
short-dated), and ``atol`` alone would be too loose for large prices. All absolute tolerances are
quoted for a spot of 100 (the validation grid spot), i.e. in currency units per unit of underlying.

Where a method has a known error law, the tolerance follows it instead of a flat number:

* CRR tree price error is O(1/n) with constant ~ S*sigma*sqrt(T). Exploratory runs on the grid
  showed |tree - BSM| <= 0.21 * S*sigma*sqrt(T)/n (n=200) and 0.13 (n=1000); the tolerance uses
  0.3, about twice the worst observed.
* Monte Carlo is a statistical estimate, so the test is in standard errors (z = 4 -> a false alarm
  probability of ~6e-5 per check), not in currency units.

If a check fails, the right response is to investigate, not to widen the tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Tol:
    atol: float
    rtol: float

    def limit(self, reference: float) -> float:
        return self.atol + self.rtol * abs(reference)

    def ok(self, value: float, reference: float) -> bool:
        return abs(value - reference) <= self.limit(reference)

    def describe(self) -> str:
        return f"atol={self.atol:g}, rtol={self.rtol:g}"


# -- analytic engine vs QuantLib AnalyticEuropeanEngine (same formulas, double precision)
QL_BSM_PRICE = Tol(1e-10, 1e-9)
QL_BSM_GREEK = Tol(1e-9, 1e-8)

# -- internal consistency of the analytic engine
PARITY = Tol(1e-10, 1e-12)
BOUNDS_SLACK = 1e-10  # numerical slack on no-arbitrage bounds (price units, S=100)
FD_GREEK = Tol(1e-7, 1e-5)  # analytic Greek vs central finite difference of the price

# -- implied volatility
IV_PRICE_ROUNDTRIP = Tol(1e-9, 1e-9)
IV_VOL_RECOVERY = Tol(1e-8, 0.0)  # only enforced where vega (per unit vol) >= IV_MIN_VEGA
IV_MIN_VEGA = 0.5
IV_BRENT_VS_BISECTION = Tol(1e-9, 0.0)  # the two solvers must agree on the vol (same cond. as above)
# Vectorised bisection vs scalar bisection volatility, where both converge AND vega >= IV_MIN_VEGA.
# CHANGE LOG (check design, tolerance unchanged): the first version compared volatilities on every
# converged lane. On the full grid 96 lanes (deep in-the-money, 7-day options, vega <= 1.2e-4) differed
# by up to 6.5e-4, because the price is flat in vol there and the scalar (math.erfc) and vectorised
# (scipy ndtr) solvers legitimately stop at different points of the flat region. On the 3,260
# well-conditioned lanes the two agree to 2.7e-13. The vol comparison now has the same vega condition as
# the other vol comparisons, and every converged lane (including the flat ones) must pass the price round trip.
IV_VECTOR_VS_SCALAR = Tol(1e-10, 0.0)

# -- CRR binomial tree
TREE_STEPS = 1000
TREE_PRICE_SCALE = 0.3  # tolerance = SCALE * S * sigma * sqrt(T) / n  (+ rtol term)
TREE_PRICE_RTOL = 1e-4
TREE_GREEK_STEPS = 500
# Tree Greek errors are judged against each Greek's natural scale (see checks.py), as a fraction of it.
TREE_GREEK_SCALED = {"delta": 2e-3, "gamma": 3e-3, "vega": 2e-2, "theta": 3e-3, "rho": 2e-3}
TREE_VS_QL_TREE_SCALE = 0.15  # my CRR vs QuantLib's CRR variant at equal steps, same scaling as above
AMERICAN_PUT_VS_QL = Tol(2e-3, 1e-3)  # American CRR (n=1000) vs QuantLib finite-difference

# -- Monte Carlo
MC_Z = 4.0  # |MC - reference| <= MC_Z * standard_error + MC_ATOL
MC_ATOL = 1e-8
# CHANGE LOG (statistical power, bands unchanged): the repeated-seed experiment first used 200 seeds.
# A 3000-seed probe showed the standard errors are right (mean SE / empirical std = 0.998 plain, 1.000
# antithetic; 95.3% coverage), but with 200 seeds the empirical std has ~8% sampling noise, so the +-15%
# SE band was only ~1.9 sigma wide and failed by chance (1.1506). The experiment now uses 1000 seeds
# (600 seeds x 4 cases in --quick), which makes the same bands 3-4 sigma wide. No band was widened.
MC_COVERAGE_POOLED = (0.93, 0.97)  # empirical coverage of the 95% interval, pooled over contracts
MC_COVERAGE_SINGLE = (0.88, 0.995)  # per contract, with fewer repetitions
MC_SE_RATIO = (0.85, 1.15)  # mean reported SE / empirical std of the estimate across seeds
MC_BIAS_Z = 4.0

# Finite-difference steps for the Greek checks (central differences everywhere).
# Spot steps scale with the local width of the price curve, S*sigma*sqrt(T): a fixed relative step
# like 1e-4*S is far too wide for a 7-day, 10%-vol option and far too narrow for a 2-year, 80%-vol one.
FD_STEPS = {
    "spot_width_fraction": 5e-3,  # h_S = fraction * S * sigma * sqrt(T)  (delta and gamma)
    "vol": 1e-4,  # absolute volatility
    "time_fraction": 1e-3,  # h_T = fraction * T
    "rate": 1e-5,  # absolute rate
}

# Preregistered monte-carlo thresholds that are not z-scores.
# (plain variance) / (antithetic variance) must be >= this. Theory says >= 1 for a payoff monotone in Z.
# CHANGE LOG: the first draft measured the variances as the empirical spread of estimates over 100-200
# seeds with a 0.8 floor. That quantity is too noisy for skewed payoffs (an OTM high-vol call failed at
# 0.65 on pure sampling noise), so the ratio is now taken from mean reported variance (mean SE^2),
# which estimates the same quantity with ~1-2% noise, and the floor is 0.95. The SE itself is validated
# against the empirical spread separately (MC_SE_RATIO).
MC_VR_MIN_RATIO = 0.95
IV_ILLPOSED_SLACK = 1e-6  # a quote this close to a model bound is considered unsolvable (price units, S=100)
AMERICAN_VS_QL_FD = Tol(5e-3, 2e-3)
AMERICAN_VS_QL_TREE_SCALE = 0.15


def tree_price_tol(spot: float, vol: float, tau: float, steps: int, scale: float = TREE_PRICE_SCALE) -> Tol:
    return Tol(scale * spot * vol * math.sqrt(tau) / steps, TREE_PRICE_RTOL)

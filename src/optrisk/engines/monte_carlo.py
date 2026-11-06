"""Monte Carlo pricing of European options by exact terminal-price simulation.

    S_T = S exp[(r - q - sigma^2/2) T + sigma sqrt(T) Z],   Z ~ N(0, 1)

No time-stepping is needed for a European payoff, so there is no discretisation bias.

Estimators
----------
Price is the sample mean of the discounted payoff. Greeks use estimators that need no
re-simulation: pathwise derivatives for delta, vega, rho and theta, and the
likelihood-ratio method for gamma (the payoff's second derivative is a delta function,
so pathwise gamma is wrong). Each Greek comes with a standard error, in
``diagnostics["greeks_std_error"]``.

Variance reduction
------------------
``antithetic=True`` evaluates each draw Z together with -Z and averages the pair. The
pair average is one sample, so ``n_paths`` terminal prices give ``n_paths / 2`` samples
and the standard error is computed from pair averages (treating the two halves as
independent samples would understate the error).

Reproducible parallelism
------------------------
The total sample count is split into ``n_streams`` fixed blocks. Block ``i`` uses the
generator built from child ``i`` of ``SeedSequence(seed).spawn(n_streams)``, which gives
statistically independent streams (NumPy's recommended approach for parallel work;
giving every worker the same seed would repeat the same numbers). Workers only decide
*who* runs a block, never *what* a block contains, and block results are merged in
block order. The price is therefore identical for any ``n_workers``.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtri

from ..contracts import OptionContract
from ..conventions import DAYS_PER_YEAR, RHO_UNIT, VEGA_UNIT
from ..errors import InvalidInputError
from ..market import MarketSnapshot
from ..results import Greeks, PricingResult
from .base import PricingEngine

# Column order of the per-sample estimator matrix.
_COLS = ("price", "delta", "gamma", "vega", "rho", "theta")
_BLOCK = 250_000  # samples per vectorised block; bounds memory at roughly 6 * 8 * block bytes


def _estimators(
    z: np.ndarray, sign: int, spot: float, strike: float, tau: float, rate: float, div: float, vol: float, want_greeks: bool
) -> np.ndarray:
    """Per-sample estimators for draws ``z``. Rows follow ``_COLS``; only row 0 if not ``want_greeks``."""
    sqrt_t = math.sqrt(tau)
    disc = math.exp(-rate * tau)
    drift = rate - div - 0.5 * vol * vol
    s_t = spot * np.exp(drift * tau + vol * sqrt_t * z)
    moneyness = sign * (s_t - strike)
    payoff = np.maximum(moneyness, 0.0)
    if not want_greeks:
        return (disc * payoff)[None, :]
    itm = (moneyness > 0.0).astype(float)
    sst = sign * s_t * itm  # signed S_T on in-the-money paths, 0 elsewhere
    out = np.empty((6, z.shape[0]))
    out[0] = disc * payoff
    out[1] = disc * sst / spot
    out[2] = disc * payoff * ((z * z - 1.0) / (spot * spot * vol * vol * tau) - z / (spot * spot * vol * sqrt_t))
    out[3] = disc * sst * (sqrt_t * z - vol * tau)
    out[4] = disc * tau * (sst - payoff)
    # theta = -dV/dT (calendar time passing shortens T)
    dv_dt = -rate * disc * payoff + disc * sst * (drift + vol * z / (2.0 * sqrt_t))
    out[5] = -dv_dt
    return out


def _moments(x: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n = x.shape[1]
    mean = x.mean(axis=1)
    m2 = ((x - mean[:, None]) ** 2).sum(axis=1)
    return n, mean, m2


def _merge(a: tuple[int, np.ndarray, np.ndarray] | None, b: tuple[int, np.ndarray, np.ndarray]):
    """Chan et al. pairwise merge of (count, mean, sum of squared deviations)."""
    if a is None:
        return b
    na, ma, m2a = a
    nb, mb, m2b = b
    n = na + nb
    delta = mb - ma
    return n, ma + delta * (nb / n), m2a + m2b + delta * delta * (na * nb / n)


def _simulate_stream(args: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    """Simulate one stream. Module-level so it can be pickled for worker processes."""
    sign, spot, strike, tau, rate, div, vol, n_samples, antithetic, want_greeks, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    acc = None
    remaining = n_samples
    while remaining > 0:
        m = min(_BLOCK, remaining)
        z = rng.standard_normal(m)
        x = _estimators(z, sign, spot, strike, tau, rate, div, vol, want_greeks)
        if antithetic:
            x = 0.5 * (x + _estimators(-z, sign, spot, strike, tau, rate, div, vol, want_greeks))
        acc = _merge(acc, _moments(x))
        remaining -= m
    assert acc is not None
    return acc


def split_counts(total: int, parts: int) -> list[int]:
    """Split ``total`` into ``parts`` near-equal non-negative integers (earlier parts get the remainder)."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


class MonteCarloEngine(PricingEngine):
    """European option pricing by Monte Carlo, with standard error and confidence interval.

    Parameters
    ----------
    n_paths      total simulated terminal prices (must be even when ``antithetic``).
    seed         non-negative int for reproducible output; ``None`` draws fresh entropy
                 (recorded in ``diagnostics["seed_entropy"]``).
    antithetic   use antithetic variates (Z and -Z) as one averaged sample.
    n_streams    number of independent RNG streams; fixes the result regardless of ``n_workers``.
    n_workers    1 runs in-process; >1 uses a persistent process pool created on first use.
                 Call :meth:`close` (or use ``with``) to release it.
    confidence   two-sided confidence level of the reported interval.
    """

    name = "monte_carlo"
    statistical = True

    def __init__(
        self,
        n_paths: int = 200_000,
        seed: int | None = 12345,
        antithetic: bool = True,
        n_streams: int = 8,
        n_workers: int = 1,
        confidence: float = 0.95,
    ) -> None:
        for label, value, lo in (("n_paths", n_paths, 2), ("n_streams", n_streams, 1), ("n_workers", n_workers, 1)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lo:
                raise InvalidInputError(f"{label} must be an integer >= {lo}, got {value!r}")
        if antithetic and n_paths % 2:
            raise InvalidInputError(f"n_paths must be even with antithetic variates, got {n_paths}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
            raise InvalidInputError(f"seed must be a non-negative integer or None, got {seed!r}")
        if not (0.0 < confidence < 1.0):
            raise InvalidInputError(f"confidence must be in (0, 1), got {confidence!r}")
        self.n_paths = int(n_paths)
        self.seed = None if seed is None else int(seed)
        self.antithetic = bool(antithetic)
        self.n_streams = int(n_streams)
        self.n_workers = int(n_workers)
        self.confidence = float(confidence)
        self._pool: ProcessPoolExecutor | None = None

    # -- pool lifecycle ---------------------------------------------------------------
    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.n_workers)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "MonteCarloEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def with_settings(self, **changes: Any) -> "MonteCarloEngine":
        """New engine with some settings changed (does not share the process pool)."""
        settings = dict(
            n_paths=self.n_paths,
            seed=self.seed,
            antithetic=self.antithetic,
            n_streams=self.n_streams,
            n_workers=self.n_workers,
            confidence=self.confidence,
        )
        settings.update(changes)
        return MonteCarloEngine(**settings)

    # -- pricing ----------------------------------------------------------------------
    def _price(self, contract: OptionContract, market: MarketSnapshot, tau: float, greeks: bool) -> PricingResult:
        n_samples = self.n_paths // 2 if self.antithetic else self.n_paths
        seed_seq = np.random.SeedSequence(self.seed)
        children = seed_seq.spawn(self.n_streams)
        counts = split_counts(n_samples, self.n_streams)
        base = (
            contract.option_type.sign,
            market.spot,
            contract.strike,
            tau,
            market.rate,
            market.dividend_yield,
            market.volatility,
        )
        jobs = [base + (c, self.antithetic, greeks, child) for c, child in zip(counts, children) if c > 0]

        results: Sequence[tuple[int, np.ndarray, np.ndarray]]
        if self.n_workers > 1 and len(jobs) > 1:
            results = list(self._executor().map(_simulate_stream, jobs))
        else:
            results = [_simulate_stream(job) for job in jobs]

        acc = None
        for part in results:  # fixed order -> bitwise-reproducible regardless of worker count
            acc = _merge(acc, part)
        assert acc is not None
        n, mean, m2 = acc
        if n < 2:
            raise InvalidInputError("need at least 2 samples to estimate a standard error")
        std_err = np.sqrt(m2 / (n - 1) / n)

        z = float(ndtri(0.5 + 0.5 * self.confidence))
        price = float(mean[0])
        se = float(std_err[0])
        diagnostics: dict[str, Any] = {
            "n_paths": self.n_paths,
            "n_samples": n,
            "antithetic": self.antithetic,
            "n_streams": self.n_streams,
            "n_workers": self.n_workers,
            "seed": self.seed,
            "seed_entropy": seed_seq.entropy,
            "confidence": self.confidence,
            "estimator_variance": se * se,
        }
        g = None
        if greeks:
            scale = np.array([1.0, 1.0, 1.0, VEGA_UNIT, RHO_UNIT, 1.0 / DAYS_PER_YEAR])
            scaled_mean = mean * scale
            g = Greeks(**{name: float(scaled_mean[i]) for i, name in enumerate(_COLS) if i > 0})
            diagnostics["greeks_std_error"] = {
                name: float(std_err[i] * scale[i]) for i, name in enumerate(_COLS) if i > 0
            }
        return PricingResult(
            price=price,
            method=self.name,
            greeks=g,
            std_error=se,
            confidence_interval=(price - z * se, price + z * se),
            diagnostics=diagnostics,
        )

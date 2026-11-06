# Conventions

Every choice another developer would otherwise have to guess. The code that enforces each one is named so you can check it.

## Time

| Question | Answer | Where |
| --- | --- | --- |
| How is time to expiry computed? | `(expiry_date - valuation_date).days / 365`, i.e. ACT/365 Fixed, whole calendar days. | `conventions.year_fraction` |
| Holidays, business days, intraday time? | Not modelled. Expiry is a calendar date; the option expires at the end of that day's accounting, with no time-of-day. | |
| Leap years? | A leap year is 366/365 of a year. That is what ACT/365F means; it matches QuantLib's `Actual365Fixed`. | tested in `test_black_scholes.py` |
| Rates and volatility | Decimals (0.05 = 5 %), annualised, continuously compounded. | `MarketSnapshot` |

## Units of results

All results are **per one unit of the underlying**, not per contract. Position-level figures are `unit figure x quantity x multiplier` (`Position.scale`).

| Quantity | Unit | Why |
| --- | --- | --- |
| `price` | currency per unit of underlying | contract size lives on `Position.multiplier`, not on the contract |
| `delta` | dV/dS, per 1.0 of spot | |
| `gamma` | d2V/dS2, per 1.0 of spot, squared | |
| `vega` | change in price for **+1 volatility point** (+0.01 absolute vol) | readable trade-desk unit; multiply by 100 for per-unit-vol |
| `theta` | change in price per **calendar day** (negative for a typical long option) | per-year theta divided by 365 |
| `rho` | change in price for **+1 rate point** (+0.01 absolute rate) | |

The unit tests check these definitions by bumping the inputs (`test_units_vega_per_point_theta_per_day_rho_per_point`), and the QuantLib comparison converts QuantLib's per-unit-vol, per-year and per-unit-rate figures into these units in exactly one place (`validation/quantlib_ref.py`).

Portfolio aggregates use position units: value in currency, delta in share-equivalents, gamma in share-equivalents per 1.0 of spot, vega in currency per vol point, theta in currency per day, rho in currency per rate point. `delta_notional = delta x spot`.

## Expired options and invalid inputs

| Situation | Behaviour |
| --- | --- |
| `valuation_date == expiry` (T = 0) | Expired. Price = intrinsic value. Delta = +1 (call) or -1 (put) if strictly in the money, else 0. Gamma, vega, theta and rho are 0. `diagnostics["expired"] = True`. At-the-money (S = K) has delta 0. Monte Carlo reports standard error 0. |
| `valuation_date > expiry` | `InvalidInputError`. A stale contract in a position file is almost always a data error, so it is not silently priced at intrinsic. |
| Option expires **inside a scenario horizon** | Revalued at intrinsic value at its expiry date (no discounting of the payoff beyond expiry). The scenario result counts these in `n_expired`. |
| Non-finite, non-numeric, or out-of-range numbers | `InvalidInputError` when the object is constructed, so invalid objects cannot exist. Bounds: spot > 0, strike > 0, 0 < vol <= 10, abs(rate) <= 1, abs(dividend yield) <= 1. The upper bounds catch percent-versus-decimal mix-ups (a volatility of `25` instead of `0.25`). |
| Zero volatility | Rejected (vol must be > 0). The zero-vol limit is a deterministic forward payoff and has no meaningful vega or gamma. Scenario vol shocks are floored at `1e-4` (`MIN_SCENARIO_VOL`) so a large down-shock never produces an invalid market. |
| Negative rates | Allowed (`-1 <= r <= 1`). The validation grid includes `r = -0.5 %`. |
| `datetime` where a `date` is expected | Rejected. A `datetime` is a subclass of `date`, so this is checked explicitly to stop time-of-day silently changing the day count. |
| Engine cannot price the instrument | `UnsupportedInstrumentError` (American exercise on the analytic or Monte Carlo engine). |
| Numerical method breaks down | `NumericalError`. Example: the CRR risk-neutral probability leaves (0, 1) because there are too few steps for the drift and volatility. |
| Contract and market underlying differ | `InvalidInputError`. |

Batch commands do not fail the whole run for one bad row of contracts: `optrisk price` records `error: <message>` in that row's `status` column and exits with code 3. Malformed *files* (missing column, bad date, duplicate id, unknown id) fail immediately with a message naming the file and row, exit code 2.

## Implied volatility

* European contracts only.
* Model-consistent bounds are **open** intervals: `max(S e^-qT - K e^-rT, 0) < C < S e^-qT` for calls, and the analogue for puts. A quote exactly on a bound has no positive-volatility solution.
* Outcomes are statuses, not exceptions: `converged`, `invalid_input`, `price_below_lower_bound`, `price_above_upper_bound`, `bracket_failed`, `max_iterations`. Failed solves return `vol = NaN`.
* `IVResult.well_conditioned` is `False` when vega per vol point is below `1e-6 x spot`. The price round trip is still exact there, but the volatility is not identifiable (the price is flat in volatility), so do not compare volatilities on those quotes.
* Default bracket `[1e-6, 5.0]`, widened upwards (doubling, up to 10) if the quote needs more. Default `xtol = 1e-12` in volatility.

## Randomness

* `MonteCarloEngine(seed=...)` is reproducible. `seed=None` draws fresh entropy and records it in `diagnostics["seed_entropy"]`.
* The path budget is split into `n_streams` fixed blocks, each with its own generator from `SeedSequence(seed).spawn(n_streams)`. Workers choose who runs a block, never what is in it, and block results are merged in block order, so **the result is bitwise identical for any `n_workers`**.
* `n_paths` counts simulated terminal prices. With antithetic variates one sample is a pair (Z, -Z) and `n_paths` must be even.
* Pricing different contracts with the same seed reuses the same random numbers (common random numbers). That is useful for scenario comparisons and means errors across contracts are correlated; use different seeds if you need independent errors.
* On macOS and Windows, `multiprocessing` starts workers by importing your main module. Scripts that use `n_workers > 1` need an `if __name__ == "__main__":` guard.

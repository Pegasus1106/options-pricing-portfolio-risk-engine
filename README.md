# optrisk: options pricing and portfolio risk engine

A Python package that prices European options three ways, backs out implied volatility, and revalues a portfolio under spot, volatility and time shocks. The numerical methods are implemented from scratch. QuantLib is used only as an independent reference in the validation suite.

| | |
| --- | --- |
| Instruments | European calls and puts (American exercise on the binomial engine as an extension) |
| Market model | Constant rate, dividend yield and volatility, per underlying |
| Pricing methods | Black-Scholes-Merton (analytic), CRR binomial tree, Monte Carlo with standard errors |
| Risk measures | Delta, gamma, vega, theta, rho |
| Calibration | Implied volatility by bisection (own implementation) and Brent (SciPy), with explicit failure statuses |
| Portfolio | Valuation, aggregate Greeks, exposure by underlying and expiry, scenario grid, Greek approximation error |
| I/O | CSV or JSON contracts, positions, market snapshots and quotes; CSV, JSON and Markdown outputs |

## Quick start

```bash
make install        # Python 3.11 virtualenv, installs the package with pytest and QuantLib
make test           # 164 tests
make demo           # the main demonstration, about a minute, writes reports/demo/
make reproduce      # demo + full validation + QuantLib comparison + benchmarks -> reports/
```

Without `make`: `python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev,reference]"`, then `.venv/bin/optrisk demo` or `.venv/bin/optrisk reproduce`. QuantLib is optional (`pip install -e ".[dev]"` skips it); without it the QuantLib checks report `skipped`.

### Commands

| Command | What it does |
| --- | --- |
| `optrisk demo` | Generates data, prices one option three ways, solves implied vols on a quote chain (including bad quotes), values the sample portfolio, runs the scenario grid, runs the quick validation |
| `optrisk reproduce` | Everything above, plus the full validation grid, the QuantLib comparison, and the benchmarks, all written to `reports/` |
| `optrisk generate --outdir data` | Writes sample `contracts.csv`, `positions.csv`, `market.csv`, `chain_contracts.csv`, `chain_quotes.csv` |
| `optrisk price --contracts C --market M --engine bsm\|binomial\|mc` | Price and Greeks per contract; `--steps`, `--paths`, `--seed`, `--workers`, `--output out.csv` |
| `optrisk implied-vol --contracts C --market M --quotes Q` | Implied vol with both solvers, one status per quote |
| `optrisk risk --contracts C --positions P --market M` | Portfolio value, aggregate Greeks, exposures |
| `optrisk scenario --contracts C --positions P --market M` | `scenario_report.md` and `scenario_grid.csv` |
| `optrisk validate [--quick]` | Validation suite and report |
| `optrisk quantlib-compare` | QuantLib comparison only |
| `optrisk benchmark [--quick]` | Benchmarks and report |

Exit codes: 0 success, 1 a validation check failed, 2 a malformed input file or invalid argument, 3 some contracts in a `price` run could not be priced.

### Library use

```python
from datetime import date
from optrisk import BinomialEngine, BlackScholesEngine, MarketSnapshot, MonteCarloEngine, OptionContract
from optrisk.implied_vol import implied_volatility

market = MarketSnapshot("ABC", date(2026, 1, 5), spot=100.0, rate=0.04, dividend_yield=0.015, volatility=0.25)
call = OptionContract("ABC", strike=105.0, expiry=date(2026, 7, 5), option_type="call")

for engine in (BlackScholesEngine(), BinomialEngine(steps=1000), MonteCarloEngine(n_paths=400_000, seed=1)):
    r = engine.price(call, market)              # the whole engine interface: price(contract, market)
    print(r.method, r.price, r.greeks.delta, r.std_error)

iv = implied_volatility(r.price, call, market, method="brentq")   # iv.status, iv.vol, iv.iterations ...
```

Portfolios: `Position(contract, quantity, multiplier)` and `Portfolio`; `RiskEngine(engine).value(...)` and `.run_scenarios(...)`.
Monte Carlo with several workers needs `with MonteCarloEngine(..., n_workers=4) as eng:` and an `if __name__ == "__main__":` guard on macOS/Windows.

## Design

Small objects, composition, one narrow interface.

| Component | Module | Responsibility |
| --- | --- | --- |
| `OptionContract` | `contracts.py` | Underlying, strike, expiry, call/put, exercise style. Validates on construction. |
| `MarketSnapshot` | `market.py` | Valuation date, spot, rate, dividend yield, volatility. Validates on construction; builds shocked copies. |
| `PricingEngine` | `engines/base.py` | `price(contract, market)`. Owns validation, expiry handling and timing so every engine behaves the same. |
| `BlackScholesEngine` | `engines/black_scholes.py` | Analytic price and Greeks; a scalar `math` version and a vectorised NumPy version of the same formulas |
| `BinomialEngine` | `engines/binomial.py` | CRR tree, European and American; tree Greeks |
| `MonteCarloEngine` | `engines/monte_carlo.py` | Exact terminal-price simulation, antithetic variates, standard errors and CIs, reproducible parallel streams |
| `PricingResult` | `results.py` | Price, Greeks, method, std error, CI, diagnostics, runtime |
| implied vol | `implied_vol.py` | Own bisection, SciPy `brentq`, vectorised bisection, price bounds, `IVStatus` |
| `Portfolio`, `Position` | `portfolio.py` | Quantities and contract multipliers |
| `RiskEngine` | `risk.py` | Valuation, aggregation, exposures, scenario revaluation, Greek approximation |

Conventions (time, units, expiry, invalid input) are in [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md); assumptions and what this tool does not do are in [`docs/ASSUMPTIONS_LIMITATIONS.md`](docs/ASSUMPTIONS_LIMITATIONS.md). The short version: time is ACT/365F in calendar days; prices are per unit of underlying (the multiplier lives on the position); vega is per vol point; rho is per rate point; theta is per calendar day; an option valued on its expiry date is worth intrinsic value; valuation after expiry raises.

## Results

All numbers below come from `optrisk reproduce` on an Apple M5 (10 logical CPUs, 24 GB), Python 3.11, NumPy 2.4.6, SciPy 1.17.1, QuantLib 1.44. The reports contain the full tables, tolerances, and variability. Re-running gives the same validation outcomes; timings vary a little from run to run.

### Correctness: [`reports/validation_report.md`](reports/validation_report.md), [`reports/quantlib_comparison.md`](reports/quantlib_comparison.md)

18 checks, all passing, on a grid of 4,032 contracts (7 strikes x 6 expiries x 4 volatilities x 4 rates x 3 dividend yields x call/put). Tolerances were fixed before the final run, as `|error| <= atol + rtol * |reference|`.

| Check | Cases | Result |
| --- | ---: | --- |
| Analytic price and 5 Greeks vs QuantLib `AnalyticEuropeanEngine` | 24,192 | max abs error 5.0e-14 (tolerance 1e-10 + 1e-9 x price for price, 1e-9 + 1e-8 x value for Greeks) |
| Analytic Greeks vs Richardson-extrapolated finite differences | 20,160 | max abs error 3.5e-9 |
| Put-call parity (analytic and tree) | 4,536 | max abs error 1.8e-12 |
| Price inside no-arbitrage bounds | 4,284 | 0 violations |
| Price to implied vol to price | 4,032 | 3,936 converge; price round-trip error <= 1.2e-11 for both solvers; vol recovered to 2.5e-13 where vega is not tiny. The other 96 quotes sit exactly on the lower bound (deep in the money, 7 days) and correctly get `price_below_lower_bound` |
| CRR tree (1,000 steps) vs analytic | 1,008 | max error 0.124 x S sigma sqrt(T)/n, the expected O(1/n) law |
| CRR tree vs QuantLib `BinomialVanillaEngine("crr")` | 288 | max difference 0.054 x S sigma sqrt(T)/n (tolerance 0.15) |
| Monte Carlo vs analytic, price and 5 Greeks, 200k paths | 1,080 | z-scores: mean -0.018, std 0.976, max abs 3.52 (limit 4) |
| Monte Carlo across 1,000 seeds, 8 contracts | 8 | pooled 95% interval coverage 96.3% (band 93% to 97%); reported standard error 2% to 4% above the empirical spread. The 8 contracts reuse the same 1,000 seeds, so they are not independent evidence |
| Antithetic vs plain, same 20,000 paths | 8 | variance reduced by 1.14x to 5.36x, median 1.80x |
| Worker count 1, 2, 4 | 3 | bitwise-identical price and Greeks |
| Invalid inputs | 47 | every case raises the expected error type, status, or file-and-row message |
| American CRR vs QuantLib CRR and finite-difference | 48 | max difference 2.8e-5 vs the tree, 2.2e-3 vs finite-difference |

Two things the validation work turned up are recorded rather than hidden (see the change log in `src/optrisk/validation/tolerances.py`): finite-difference references needed Richardson extrapolation in the far tails, and two checks were redesigned because they were badly powered or compared volatilities where the price is flat in volatility. No tolerance was loosened to pass.

### Portfolio scenarios: [`reports/demo/scenario_report.md`](reports/demo/scenario_report.md)

The sample book has 18 long and short positions on three underlyings across four expiries (4, 30, 90 and 180 days). Value 39,858; aggregate delta 6,637 share-equivalents; vega 2,339 per vol point; theta -797 per day. The report has the 5 x 3 x 3 grid (spot -10% to +10%, vol -5 to +5 points, 0, 1 and 7 days) with full revaluation, the Greek approximation and their difference, and exposure by underlying, by expiry and by both.

Where the Greek approximation fails, measured on this book and on single options:

* **Shock size.** With no vol or time move, the delta-gamma error is 2,961 to 3,392 at +-5% spot and 24,774 to 27,675 at +-10%: 7x to 9x larger for twice the shock (a third-order error predicts 8x). Relative to the full P&L that is 7% to 14% at +-5% and 28% to 60% at +-10%.
* **Near expiry.** Positions with at most 7 days to expiry (3 of 18) account for 72% of the net error in the worst grid cell. For one at-the-money call, the delta-gamma error at +-10% spot is 42% to 334% of the true price change at 4 days, 6.6% to 8.4% at 30 days, 0.6% to 2.5% at 90 days and 0.9% at 365 days.
* **Out of the money.** For a 30-day call and a +5% move, the delta-gamma error is 0.04% of the change at strike 80, 1.5% at 100, 9% at 110 and 27.5% at 120.
* **Volatility.** Vega alone is accurate near the money (max error 0.17% of the change) and poor out of the money (up to 94% at strike 120, 30 days).
* **Time.** Theta alone misses 10.7% of 7-day decay at 4 days to expiry and 0.4% at 365 days; options that expire inside the horizon are valued at intrinsic and cannot be represented by a linear theta.

### Performance: [`reports/benchmark_report.md`](reports/benchmark_report.md)

Compute time is timed separately from file loading and process start-up. Medians over 3 to 15 repeats, with coefficients of variation in the report.

| Workload | Baseline | Optimised | Ratio |
| --- | --- | --- | ---: |
| Price + 5 Greeks for 1,000,000 contracts | 1.45 s, pure-Python loop | 37.0 ms, one NumPy call | 39x |
| Same, 100 contracts | 85.5 µs | 18.4 µs | 4.7x |
| Same, 10 contracts | 8.6 µs | 16.0 µs | **0.54x (slower)** |
| Same, 1 contract | 1.0 µs | 15.6 µs | **0.07x (slower)** |
| 45-scenario revaluation, 2,000 positions (92,000 pricings) | 490 ms, one engine call per position | 4.0 ms, vectorised | 122x |
| 45-scenario revaluation, 200,000 positions (9.2M pricings) | not run (linear extrapolation: about 49 s) | 363 ms | n/a |
| Monte Carlo, 8,000,000 paths | 59.2 ms, 1 worker | 11.5 ms, 10 workers | 5.2x (52% efficiency) |
| Monte Carlo, 1,000,000 paths | 8.0 ms, 1 worker | 3.2 ms, 10 workers | 2.5x |
| Monte Carlo, 100,000 paths | 1.24 ms, 1 worker | 2.26 ms, 2 workers | **0.55x (slower)** |
| Monte Carlo, 10,000 paths | 0.57 ms, 1 worker | 2.23 ms, 2 workers | **0.25x (slower)** |
| 50 contracts x 100,000 paths, one pool call each | 61 ms, 1 worker | 115 ms, 4 workers | **0.53x (slower)** |

Other measured costs: importing `optrisk` takes about 211 ms (mostly SciPy); creating and warming a pool takes about 230 ms for 2 workers and 380 ms for 8; loading 100,000 positions from CSV takes 578 ms against 181 ms to compute the full 45-scenario grid. At a 1 ms budget on an at-the-money call the 1,000-step tree reaches an error of about 1e-3, where antithetic Monte Carlo with 100,000 paths reaches about 2e-2.

The 122x row compares against the generic one-call-per-position route, which pays for validation and result objects on every call; it is not a comparison against the best possible scalar code. The 39x row is against a bare `math`-based loop. The 8M-path Monte Carlo result is bitwise identical for every worker count. Parallelism is over the path budget of one price and does not help small jobs.

## Resume bullets (measured values only)

* Implemented an object-oriented Python options pricing and risk engine with analytic, binomial and Monte Carlo methods; validated 4,032 contracts (24,192 price and Greek comparisons) against QuantLib to within 5e-14 absolute on every quantity, against a pre-set price tolerance of 1e-10 + 1e-9 x price.
* Developed implied-volatility calibration (own bisection vs SciPy Brent; 45 vs 15 function evaluations on average; 3,936 round trips with price error under 1.3e-11) and portfolio scenario analysis; cut a 45-scenario revaluation of a 2,000-position book from 490 ms to 4.0 ms (122x versus per-position engine calls) and an 8M-path Monte Carlo price from 59 ms to 11.5 ms on 10 workers (5.2x) with bitwise-identical results.

## Repository layout

```
src/optrisk/
  contracts.py market.py results.py conventions.py errors.py
  engines/            base.py black_scholes.py binomial.py monte_carlo.py
  implied_vol.py portfolio.py risk.py io.py generate.py
  scenario_report.py benchmarks.py reporting.py cli.py
  validation/         grid.py tolerances.py checks.py quantlib_ref.py report.py
tests/                pytest suite (also runs the quick validation)
docs/                 CONVENTIONS.md  ASSUMPTIONS_LIMITATIONS.md
reports/              validation_report.md quantlib_comparison.md benchmark_report.md demo/
```

## Not included

Finite-difference engine, C++ kernel and the inventory-aware quote generator (the optional extensions); volatility smiles; discrete dividends; live market data. See `docs/ASSUMPTIONS_LIMITATIONS.md`.

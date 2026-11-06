# Benchmark report

## Setup

| Item | Value |
| --- | --- |
| CPU | Apple M5 |
| Logical CPUs | 10 |
| Memory | 24.0 GB |
| OS | Darwin 27.2.0 |
| Python | 3.11.17 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| QuantLib | 1.44 |
| optrisk | 0.1.0 |
| Run date (UTC) | 2026-11-05 08:04:11 |

**Method.** Wall-clock `perf_counter`; one warm-up call; each cell repeats until the minimum count and a time budget are met (the `n` in each cell). Calls under 0.2 ms are batched so a sample lasts about 2 ms. Cells show the **median**; CV is the coefficient of variation across samples, so a large CV means the median is soft. Nothing else was paused on the machine. Speedup = ratio of medians. Process pools are created once and reused; their start-up is reported separately. Inputs come from fixed seeds. Absolute numbers apply to this machine only.

## 1. Scalar Python vs vectorised NumPy (price + all five Greeks)

`scalar` is a pure-Python loop over `bsm_scalar` (inputs pre-converted to Python lists, so the loop is not penalised for NumPy scalar access). `vectorised` is one `bsm_arrays` call. `object interface` loops over `BlackScholesEngine.price(contract, market)` and includes validation and result-object creation (contract objects built outside the timed region).

| Contracts | Scalar Python loop | NumPy vectorised | Speedup | Object interface loop | Max abs diff (price, delta) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 µs (CV 1%, n=15) | 15.6 µs (CV 2%, n=15) | 0.07x | 2.5 µs (CV 2%, n=15) | 1.3e-15 |
| 10 | 8.6 µs (CV 1%, n=15) | 16.0 µs (CV 3%, n=15) | 0.54x | 24.1 µs (CV 1%, n=15) | 1.4e-14 |
| 100 | 85.5 µs (CV 1%, n=15) | 18.4 µs (CV 1%, n=15) | 4.7x | 243.4 µs (CV 1%, n=15) | 1.4e-14 |
| 1,000 | 881.8 µs (CV 1%, n=15) | 43.0 µs (CV 1%, n=15) | 20.5x | 2.48 ms (CV 1%, n=15) | 2.8e-14 |
| 10,000 | 9.29 ms (CV 21%, n=15) | 340.1 µs (CV 5%, n=15) | 27.3x | 29.09 ms (CV 19%, n=15) | 5.7e-14 |
| 100,000 | 121.64 ms (CV 5%, n=15) | 3.52 ms (CV 1%, n=15) | 34.5x | not run | 5.7e-14 |
| 1,000,000 | 1.45 s (CV 6%, n=3) | 37.00 ms (CV 2%, n=15) | 39.3x | not run | 5.7e-14 |

Vectorised code is **slower** than the scalar loop at 1, 10 contract(s) (down to 0.07x): a NumPy call has a fixed overhead of tens of microseconds that a handful of options cannot amortise.

## 2. Portfolio scenario revaluation (45 scenarios + base Greeks)

`vectorised`: `RiskEngine` with the Black-Scholes engine, whole book per scenario in one NumPy call. `position-by-position`: the same engine behind the generic interface (one `price(contract, market)` call per position per scenario). `book build` is the part of the vectorised run spent flattening the portfolio into arrays.

| Positions | Pricings | Book build | Vectorised total | Position-by-position | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 | 920 | 8.9 µs (CV 1%, n=11) | 790.4 µs (CV 2%, n=11) | 5.33 ms (CV 1%, n=5) | 6.7x |
| 200 | 9,200 | 69.4 µs (CV 1%, n=11) | 1.07 ms (CV 2%, n=11) | 49.82 ms (CV 0%, n=5) | 46.4x |
| 2,000 | 92,000 | 643.7 µs (CV 3%, n=11) | 4.02 ms (CV 2%, n=11) | 489.96 ms (CV 0%, n=3) | 122.0x |
| 20,000 | 920,000 | 6.11 ms (CV 1%, n=11) | 36.12 ms (CV 1%, n=11) | not run | n/a |
| 200,000 | 9,200,000 | 64.20 ms (CV 1%, n=11) | 363.27 ms (CV 0%, n=6) | not run | n/a |

## 3. Monte Carlo across worker processes

One European call, antithetic, price only, 40 independent RNG streams split across the workers. The result is identical for every worker count (column *same price*), so any timing difference is pure parallel overhead or gain. *First call* includes creating the pool; *compute* is the median of later calls.

| Paths | Workers | Compute (median) | Speedup vs 1 | Parallel efficiency | First call | Same price |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 565.0 µs (CV 1%, n=11) | 1.0x | 100% | 713.2 µs | yes |
| 10,000 | 2 | 2.23 ms (CV 4%, n=11) | 0.25x | 13% | 290.45 ms | yes |
| 10,000 | 4 | 2.40 ms (CV 3%, n=11) | 0.24x | 6% | 215.84 ms | yes |
| 10,000 | 8 | 2.46 ms (CV 13%, n=11) | 0.23x | 3% | 298.47 ms | yes |
| 10,000 | 10 | 2.66 ms (CV 33%, n=11) | 0.21x | 2% | 319.23 ms | yes |
| 100,000 | 1 | 1.24 ms (CV 3%, n=11) | 1.0x | 100% | 1.44 ms | yes |
| 100,000 | 2 | 2.26 ms (CV 2%, n=11) | 0.55x | 27% | 207.91 ms | yes |
| 100,000 | 4 | 2.47 ms (CV 5%, n=11) | 0.50x | 13% | 215.01 ms | yes |
| 100,000 | 8 | 2.59 ms (CV 10%, n=11) | 0.48x | 6% | 296.81 ms | yes |
| 100,000 | 10 | 2.90 ms (CV 48%, n=11) | 0.43x | 4% | 317.60 ms | yes |
| 1,000,000 | 1 | 8.03 ms (CV 1%, n=11) | 1.0x | 100% | 8.22 ms | yes |
| 1,000,000 | 2 | 5.08 ms (CV 1%, n=11) | 1.6x | 79% | 212.18 ms | yes |
| 1,000,000 | 4 | 3.37 ms (CV 2%, n=11) | 2.4x | 60% | 228.81 ms | yes |
| 1,000,000 | 8 | 3.15 ms (CV 9%, n=11) | 2.6x | 32% | 302.59 ms | yes |
| 1,000,000 | 10 | 3.19 ms (CV 18%, n=11) | 2.5x | 25% | 337.52 ms | yes |
| 8,000,000 | 1 | 59.23 ms (CV 0%, n=11) | 1.0x | 100% | 60.12 ms | yes |
| 8,000,000 | 2 | 31.31 ms (CV 4%, n=11) | 1.9x | 95% | 241.12 ms | yes |
| 8,000,000 | 4 | 17.35 ms (CV 1%, n=11) | 3.4x | 85% | 241.77 ms | yes |
| 8,000,000 | 8 | 12.13 ms (CV 1%, n=11) | 4.9x | 61% | 343.92 ms | yes |
| 8,000,000 | 10 | 11.49 ms (CV 2%, n=11) | 5.2x | 52% | 392.63 ms | yes |

**Batches of small jobs.** 50 contracts x 100,000 paths each, priced one after another, each call dispatched to the pool:

| Workers | Total (median) | Speedup vs 1 |
| --- | ---: | ---: |
| 1 | 61.13 ms (CV 1%, n=7) | 1.0x |
| 4 | 115.30 ms (CV 1%, n=7) | 0.53x |
| 8 | 115.63 ms (CV 1%, n=7) | 0.53x |

## 4. Start-up and file loading, separate from compute

**Process and import costs** (fresh subprocess, wall time):

| Step | Median (CV, n) |
| --- | --- |
| python interpreter only | 15.04 ms (CV 3%, n=7) |
| import numpy | 42.65 ms (CV 2%, n=7) |
| import scipy special optimize | 201.62 ms (CV 1%, n=7) |
| import optrisk | 208.40 ms (CV 0%, n=7) |

**Worker pool** creation + one task per worker + shutdown:

| Workers | Median (CV, n) |
| --- | ---: |
| 2 | 228.68 ms (CV 1%, n=7) |
| 4 | 253.57 ms (CV 1%, n=7) |
| 8 | 376.60 ms (CV 1%, n=7) |

**CSV loading versus computing** (vectorised Black-Scholes engine). Loading includes CSV parsing, validation and object construction for contracts, positions and markets:

| Positions | CSV MB | Load files | Valuation + Greeks | 45-scenario grid | Load / scenario grid |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.07 | 4.46 ms (CV 3%, n=7) | 5.03 ms (CV 2%, n=7) | 2.52 ms (CV 8%, n=7) | 1.8x |
| 10,000 | 0.68 | 47.96 ms (CV 10%, n=7) | 38.41 ms (CV 11%, n=7) | 18.35 ms (CV 1%, n=7) | 2.6x |
| 100,000 | 6.77 | 577.53 ms (CV 4%, n=3) | 390.60 ms (CV 6%, n=3) | 181.01 ms (CV 0%, n=6) | 3.2x |

## 5. Accuracy at comparable computational budgets

Error against the analytic price. Tree error is deterministic. Monte Carlo error is the **RMSE over repeated seeds**, which is the quantity that matters when you cannot know the seed's luck in advance. Runtime is the median for one pricing call (single process, price only).

**ATM call 90d** (analytic engine: 2.4 µs per call)

| Method | Setting | Median runtime | Error | Error kind |
| --- | --- | --- | --- | --- |
| tree | 10 steps | 11.4 µs | 0.121 | abs error (deterministic) |
| tree | 25 steps | 22.3 µs | 0.049 | abs error (deterministic) |
| tree | 50 steps | 39.4 µs | 0.025 | abs error (deterministic) |
| tree | 100 steps | 74.2 µs | 0.012 | abs error (deterministic) |
| tree | 250 steps | 186.0 µs | 0.005 | abs error (deterministic) |
| tree | 500 steps | 391.4 µs | 0.002 | abs error (deterministic) |
| tree | 1000 steps | 861.5 µs | 0.001 | abs error (deterministic) |
| tree | 2000 steps | 1.96 ms | 6.2e-04 | abs error (deterministic) |
| mc_plain | 1,000 paths | 96.0 µs | 0.279 | RMSE over 60 seeds |
| mc_plain | 10,000 paths | 163.5 µs | 0.075 | RMSE over 60 seeds |
| mc_plain | 100,000 paths | 966.3 µs | 0.026 | RMSE over 60 seeds |
| mc_plain | 1,000,000 paths | 9.05 ms | 0.008 | RMSE over 60 seeds |
| mc_antithetic | 1,000 paths | 114.5 µs | 0.173 | RMSE over 60 seeds |
| mc_antithetic | 10,000 paths | 168.6 µs | 0.070 | RMSE over 60 seeds |
| mc_antithetic | 100,000 paths | 830.5 µs | 0.020 | RMSE over 60 seeds |
| mc_antithetic | 1,000,000 paths | 7.53 ms | 0.005 | RMSE over 60 seeds |

**OTM put 6m** (analytic engine: 2.4 µs per call)

| Method | Setting | Median runtime | Error | Error kind |
| --- | --- | --- | --- | --- |
| tree | 10 steps | 11.5 µs | 0.006 | abs error (deterministic) |
| tree | 25 steps | 21.4 µs | 0.025 | abs error (deterministic) |
| tree | 50 steps | 38.9 µs | 0.013 | abs error (deterministic) |
| tree | 100 steps | 73.0 µs | 0.004 | abs error (deterministic) |
| tree | 250 steps | 180.8 µs | 0.002 | abs error (deterministic) |
| tree | 500 steps | 378.4 µs | 0.001 | abs error (deterministic) |
| tree | 1000 steps | 855.3 µs | 1.8e-04 | abs error (deterministic) |
| tree | 2000 steps | 1.96 ms | 1.2e-04 | abs error (deterministic) |
| mc_plain | 1,000 paths | 94.3 µs | 0.107 | RMSE over 60 seeds |
| mc_plain | 10,000 paths | 161.6 µs | 0.035 | RMSE over 60 seeds |
| mc_plain | 100,000 paths | 976.7 µs | 0.012 | RMSE over 60 seeds |
| mc_plain | 1,000,000 paths | 9.03 ms | 0.004 | RMSE over 60 seeds |
| mc_antithetic | 1,000 paths | 114.2 µs | 0.100 | RMSE over 60 seeds |
| mc_antithetic | 10,000 paths | 168.3 µs | 0.040 | RMSE over 60 seeds |
| mc_antithetic | 100,000 paths | 828.5 µs | 0.012 | RMSE over 60 seeds |
| mc_antithetic | 1,000,000 paths | 7.44 ms | 0.003 | RMSE over 60 seeds |

**ITM call 1y** (analytic engine: 2.4 µs per call)

| Method | Setting | Median runtime | Error | Error kind |
| --- | --- | --- | --- | --- |
| tree | 10 steps | 11.0 µs | 0.242 | abs error (deterministic) |
| tree | 25 steps | 21.2 µs | 0.082 | abs error (deterministic) |
| tree | 50 steps | 38.8 µs | 0.020 | abs error (deterministic) |
| tree | 100 steps | 74.5 µs | 0.011 | abs error (deterministic) |
| tree | 250 steps | 179.3 µs | 0.004 | abs error (deterministic) |
| tree | 500 steps | 374.3 µs | 0.002 | abs error (deterministic) |
| tree | 1000 steps | 845.5 µs | 0.002 | abs error (deterministic) |
| tree | 2000 steps | 1.97 ms | 3.8e-05 | abs error (deterministic) |
| mc_plain | 1,000 paths | 95.2 µs | 0.855 | RMSE over 60 seeds |
| mc_plain | 10,000 paths | 161.1 µs | 0.219 | RMSE over 60 seeds |
| mc_plain | 100,000 paths | 967.2 µs | 0.081 | RMSE over 60 seeds |
| mc_plain | 1,000,000 paths | 8.99 ms | 0.025 | RMSE over 60 seeds |
| mc_antithetic | 1,000 paths | 115.1 µs | 0.471 | RMSE over 60 seeds |
| mc_antithetic | 10,000 paths | 168.0 µs | 0.193 | RMSE over 60 seeds |
| mc_antithetic | 100,000 paths | 813.8 µs | 0.054 | RMSE over 60 seeds |
| mc_antithetic | 1,000,000 paths | 7.35 ms | 0.015 | RMSE over 60 seeds |

**Best error within a runtime budget** (best configuration of each method that finishes within the budget):

| case | runtime_budget | tree | mc_plain | mc_antithetic |
| --- | ---: | ---: | ---: | ---: |
| ATM call 90d | 1.00 ms | 1.23e-03 (1000 steps) | 2.62e-02 (100,000 paths) | 1.98e-02 (100,000 paths) |
| ATM call 90d | 10.00 ms | 6.16e-04 (2000 steps) | 8.31e-03 (1,000,000 paths) | 5.50e-03 (1,000,000 paths) |
| ATM call 90d | 100.00 ms | 6.16e-04 (2000 steps) | 8.31e-03 (1,000,000 paths) | 5.50e-03 (1,000,000 paths) |
| OTM put 6m | 1.00 ms | 1.80e-04 (1000 steps) | 1.15e-02 (100,000 paths) | 1.15e-02 (100,000 paths) |
| OTM put 6m | 10.00 ms | 1.17e-04 (2000 steps) | 4.25e-03 (1,000,000 paths) | 3.32e-03 (1,000,000 paths) |
| OTM put 6m | 100.00 ms | 1.17e-04 (2000 steps) | 4.25e-03 (1,000,000 paths) | 3.32e-03 (1,000,000 paths) |
| ITM call 1y | 1.00 ms | 2.35e-03 (1000 steps) | 8.06e-02 (100,000 paths) | 5.39e-02 (100,000 paths) |
| ITM call 1y | 10.00 ms | 3.81e-05 (2000 steps) | 2.51e-02 (1,000,000 paths) | 1.53e-02 (1,000,000 paths) |
| ITM call 1y | 100.00 ms | 3.81e-05 (2000 steps) | 2.51e-02 (1,000,000 paths) | 1.53e-02 (1,000,000 paths) |

## Reading these results

- Treat any speedup with a high CV as approximate. Ratios of medians from fewer than ~5 repeats are weak evidence.
- Process-level parallelism pays a fixed cost per call (serialising arguments, scheduling, returning results). Small jobs can be slower with more workers; this is a finding, not a defect.
- The tree's error is deterministic but not monotone in the step count (odd/even oscillation), so a larger tree is not guaranteed to beat a smaller one at every setting.
- For a single vanilla European option the analytic engine is exact and fastest; tree and Monte Carlo exist to validate it and to extend the framework to payoffs without closed forms.

# Assumptions and limitations

## Model assumptions

* **Black-Scholes-Merton dynamics.** One underlying with constant volatility, a constant continuously compounded risk-free rate and a constant continuous dividend yield. Log-normal terminal prices, no jumps, no stochastic volatility.
* **One flat volatility per underlying.** No smile or term structure. A portfolio revalued under this model prices every strike on an underlying with the same volatility, and a "vol shock" is a parallel shift. Real books have skew, so scenario P&L from this tool will not match a desk's smile-aware numbers. The implied-vol solver does produce one volatility per quote, so you can see the smile, but the pricing and scenario engines do not use it.
* **Continuous dividends.** Discrete cash dividends are not supported.
* **Frictionless.** No transaction costs, borrow costs, margin, or bid-ask spread.
* **Calendar-day, date-only time.** ACT/365F, no holidays, no intraday expiry (see `CONVENTIONS.md`).

## Instruments and methods

* European calls and puts are the core scope. American exercise is a later extension: it works on the binomial engine only, and is validated against QuantLib's CRR tree and finite-difference engine, not against a closed form. American Greeks come from the same tree nodes as European ones and have had no separate accuracy study. Implied volatility does not support American exercise.
* The CRR tree converges at O(1/n) with an odd/even oscillation (see the convergence table in the validation report). A single step count is not guaranteed to beat a smaller one. The tree uses up factor `exp(sigma sqrt(dt))` and sets the up-probability so the one-step mean equals the forward exactly (which makes put-call parity hold to rounding). QuantLib's `"crr"` uses the same factors but takes the probability from the log-drift, `1/2 + (r - q - sigma^2/2) sqrt(dt) / (2 sigma)`; recomputing a tree with that probability reproduces QuantLib's price to about 1e-11 in a spot check. The two variants converge to the same limit and differ by a small amount at finite n (the validation tolerance for that difference is stated and was met).
* Tree vega and rho are central differences of rebuilt trees, so they are about five times as expensive as price alone.
* Monte Carlo simulates the terminal price exactly. It is not suitable for path-dependent payoffs without extension (no time stepping). Greeks use pathwise estimators (delta, vega, rho, theta) and a likelihood-ratio estimator (gamma). The gamma estimator has much higher variance than the others; its standard error is reported but expect it to be the noisiest figure.
* Some Monte Carlo estimators can have zero variance in special cases (for example K = S with zero risk-neutral drift makes the antithetic pathwise rho exactly constant). Such results are exact, and their reported standard error is rounding noise.

## Scenario analysis

* The Greek approximation is the standard second-order spot expansion plus linear vega and theta. It leaves out third-order spot terms, vanna, volga, charm and the expiry cap. The scenario report quantifies where this matters on the sample portfolio and on single options; it does not isolate the contribution of each omitted term.
* Spot shocks are relative and applied equally to every underlying (no correlation structure, no per-underlying shock). Rates and dividend yields are held constant.
* A scenario is a set of instantaneous revaluations, not a distribution. This is not a VaR or expected-shortfall model.
* Options expiring inside a time-advance scenario are valued at intrinsic value at their expiry date and receive no interest after it.

## Validation scope

* The QuantLib comparison proves agreement with an independent implementation under matched conventions on the stated grid. It does not prove the model is a good description of any market.
* Grid spot is fixed at 100. Absolute tolerances are quoted for that spot.
* The data is synthetic. No live or historical market data has been used, so nothing here says anything about performance against real quotes or real calibration stability.
* Tolerances were fixed before the final reports were generated. Where an early version of a check was badly designed (for example, comparing volatilities where the price is flat in volatility, or a standard-error test with too few seeds), the check was corrected and the reason is recorded in the change log at the top of `validation/tolerances.py`. No tolerance band was widened to make a result pass.

## Performance scope

* Timings come from one machine (reported in the benchmark report) with other processes running normally. Treat absolute numbers as indicative and ratios with a high coefficient of variation as approximate.
* The vectorised speedup compares against a pure-Python loop and against the object-based interface; it says nothing about Numba, C++ or GPU implementations, which were not tried.
* Process-level parallelism is applied to the path budget of one Monte Carlo price. It is not applied across contracts or scenarios. Parallelising across contracts would likely help the many-small-jobs case but was not built or measured.

## Not implemented

Finite-difference engine, C++ kernel, quote generator with inventory and risk limits (the optional extensions); volatility surfaces; discrete dividends; stochastic rates; multi-asset options; portfolio VaR.

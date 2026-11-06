# Portfolio scenario report

Pricing engine: `black_scholes`. Valuation date: 2026-01-05. Positions: 18. Underlyings: SIM_A, SIM_B, SIM_C.

Units: values in currency; delta in share-equivalents; gamma in share-equivalents per 1.0 move in spot; vega per +1 volatility point; theta per calendar day; rho per +1 rate point. Position figures = unit figure x quantity x multiplier.

## Market snapshots

| Underlying | Spot | Rate | Dividend yield | Volatility |
| --- | ---: | ---: | ---: | ---: |
| SIM_A | 100.0000 | 0.0400 | 0.0150 | 0.2200 |
| SIM_B | 250.0000 | 0.0400 | 0 | 0.3500 |
| SIM_C | 50.0000 | 0.0400 | 0.0300 | 0.2800 |

## Portfolio value and aggregate Greeks

| Measure | Value |
| --- | ---: |
| Total value | 39,857.94 |
| Delta (share-equivalents) | 6,636.57 |
| Delta notional (currency) | 662,113.72 |
| Gamma | 1,861.41 |
| Vega (per vol point) | 2,338.76 |
| Theta (per day) | -796.55 |
| Rho (per rate point) | 1,632.13 |

## Exposure by underlying

| Underlying | Positions | Value | Delta | Delta notional | Gamma | Vega | Theta/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SIM_A | 8 | 17,136.1 | 3,551.1 | 355,110.9 | 669.8 | 1,233.7 | -466.5 |
| SIM_B | 6 | 25,772.4 | 763.7 | 190,912.5 | -1.1 | 1,187.8 | -6.3 |
| SIM_C | 4 | -3,050.6 | 2,321.8 | 116,090.3 | 1,192.8 | -82.8 | -323.8 |

## Exposure by expiry

| Expiry | Positions | Value | Delta | Gamma | Vega | Theta/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-01-09 | 3 | 5,708.2 | 4,094.1 | 1,845.4 | 221.3 | -700.4 |
| 2026-02-04 | 5 | -11,890.9 | 171.4 | -156.0 | -510.3 | 315.0 |
| 2026-04-05 | 5 | 25,400.8 | -279.5 | 172.6 | 1,738.3 | -321.8 |
| 2026-07-04 | 5 | 20,639.9 | 2,650.6 | -0.6 | 889.5 | -89.4 |

## Exposure by underlying and expiry

| Underlying | Expiry | Positions | Value | Delta | Gamma | Vega | Theta/day |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SIM_A | 2026-01-09 | 2 | 2,772.2 | 1,558.3 | 485.0 | 116.9 | -332.0 |
| SIM_A | 2026-02-04 | 2 | 4,821.9 | 902.1 | 81.7 | 147.8 | -59.8 |
| SIM_A | 2026-04-05 | 2 | -2,747.9 | 118.9 | 27.4 | 148.5 | -19.3 |
| SIM_A | 2026-07-04 | 2 | 12,289.9 | 971.8 | 75.6 | 820.4 | -55.4 |
| SIM_B | 2026-02-04 | 2 | -15,726.2 | -220.9 | -30.0 | -538.6 | 318.5 |
| SIM_B | 2026-04-05 | 2 | 25,690.8 | 194.3 | 25.6 | 1,383.4 | -271.5 |
| SIM_B | 2026-07-04 | 2 | 15,807.8 | 790.2 | 3.2 | 343.0 | -53.3 |
| SIM_C | 2026-01-09 | 1 | 2,935.9 | 2,535.9 | 1,360.4 | 104.4 | -368.4 |
| SIM_C | 2026-02-04 | 1 | -986.6 | -509.9 | -207.8 | -119.6 | 56.4 |
| SIM_C | 2026-04-05 | 1 | 2,457.9 | -592.7 | 119.6 | 206.4 | -31.0 |
| SIM_C | 2026-07-04 | 1 | -7,457.8 | 888.6 | -79.4 | -273.9 | 19.3 |

## Scenario results

Each scenario is a joint relative spot shock, absolute vol shock (in vol points) and a number of calendar days forward, applied to every underlying. *Full revaluation* re-prices every position. *Greek approximation* is `delta*dS + 0.5*gamma*dS^2 + vega*dVol + theta*days`, using Greeks at the base market. Values are portfolio P&L versus the base value.

### 0 day(s) forward

**Full revaluation P&L**

|  | spot -10% | spot -5% | spot +0% | spot +5% | spot +10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol +5pt | -33,237 | -12,401 | 12,032 | 51,297 | 96,649 |
| vol +0pt | -45,841 | -24,487 | 0 | 42,156 | 89,484 |
| vol -5pt | -57,965 | -35,725 | -11,203 | 34,309 | 83,552 |

**Greek approximation P&L**

|  | spot -10% | spot -5% | spot +0% | spot +5% | spot +10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol +5pt | -6,472 | -9,400 | 11,694 | 56,811 | 125,951 |
| vol +0pt | -18,165 | -21,094 | 0 | 45,117 | 114,257 |
| vol -5pt | -29,859 | -32,788 | -11,694 | 33,423 | 102,563 |

**Approximation error (full - approximation)**

|  | spot -10% | spot -5% | spot +0% | spot +5% | spot +10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol +5pt | -26,766 | -3,001 | 338 | -5,514 | -29,302 |
| vol +0pt | -27,675 | -3,392 | 0 | -2,961 | -24,774 |
| vol -5pt | -28,105 | -2,937 | 491 | 886 | -19,011 |

### 1 day(s) forward

**Full revaluation P&L**

|  | spot -10% | spot -5% | spot +0% | spot +5% | spot +10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol +5pt | -33,468 | -12,441 | 11,050 | 51,022 | 96,638 |
| vol +0pt | -46,064 | -24,455 | -852 | 42,013 | 89,486 |
| vol -5pt | -58,161 | -35,654 | -11,898 | 34,243 | 83,549 |

**Greek approximation P&L**

|  | spot -10% | spot -5% | spot +0% | spot +5% | spot +10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol +5pt | -7,268 | -10,197 | 10,897 | 56,014 | 125,155 |
| vol +0pt | -18,962 | -21,891 | -797 | 44,321 | 113,461 |
| vol -5pt | -30,656 | -33,585 | -12,490 | 32,627 | 101,767 |

**Approximation error (full - approximation)**

|  | spot -10% | spot -5% | spot +0% | spot +5% | spot +10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol +5pt | -26,200 | -2,244 | 153 | -4,993 | -28,516 |
| vol +0pt | -27,102 | -2,565 | -56 | -2,308 | -23,975 |
| vol -5pt | -27,506 | -2,070 | 593 | 1,616 | -18,218 |

### 7 day(s) forward

**Full revaluation P&L**

|  | spot -10% | spot -5% | spot +0% | spot +5% | spot +10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol +5pt | -35,038 | -11,466 | 4,526 | 50,693 | 96,786 |
| vol +0pt | -47,439 | -23,602 | -6,345 | 41,867 | 89,648 |
| vol -5pt | -59,324 | -34,905 | -16,288 | 34,147 | 83,637 |

**Greek approximation P&L**

|  | spot -10% | spot -5% | spot +0% | spot +5% | spot +10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol +5pt | -12,048 | -14,976 | 6,118 | 51,235 | 120,375 |
| vol +0pt | -23,741 | -26,670 | -5,576 | 39,541 | 108,681 |
| vol -5pt | -35,435 | -38,364 | -17,270 | 27,848 | 96,988 |

**Approximation error (full - approximation)**

|  | spot -10% | spot -5% | spot +0% | spot +5% | spot +10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| vol +5pt | -22,990 | 3,511 | -1,592 | -542 | -23,589 |
| vol +0pt | -23,698 | 3,068 | -769 | 2,325 | -19,033 |
| vol -5pt | -23,889 | 3,459 | 982 | 6,299 | -13,351 |

### P&L attribution of the approximation (days = 0)

| Scenario | Delta | Gamma | Vega | Theta | Approx total | Full | Error | Error % of full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spot -10%, vol +0pt, 0d | -66,211 | 48,046 | 0 | 0 | -18,165 | -45,841 | -27,675 | -60 |
| spot -5%, vol +0pt, 0d | -33,106 | 12,011 | 0 | 0 | -21,094 | -24,487 | -3,392 | -14 |
| spot +5%, vol +0pt, 0d | 33,106 | 12,011 | 0 | 0 | 45,117 | 42,156 | -2,961 | -7 |
| spot +10%, vol +0pt, 0d | 66,211 | 48,046 | 0 | 0 | 114,257 | 89,484 | -24,774 | -28 |

## Where the Greek approximation breaks down

Largest absolute error in the grid: **-29,302** at *spot +10%, vol +5pt, 0d* (full 96,649, approximation 125,951).

Largest position-level errors in that scenario (*spot +10%, vol +5pt, 0d*):

| Position | Option | Days to expiry | Quantity | Full P&L | Approx P&L | Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A1 | call 100 | 4 | 30 | 27,281 | 41,876 | -14,594 |
| C1 | call 50 | 4 | 50 | 22,091 | 30,205 | -8,114 |
| A5 | put 90 | 90 | 40 | -1,887 | 673 | -2,560 |
| A4 | call 110 | 30 | -20 | -6,599 | -4,096 | -2,503 |
| A3 | call 100 | 30 | 20 | 15,963 | 17,943 | -1,979 |
| B4 | put 220 | 90 | 20 | -4,118 | -2,219 | -1,900 |

Error by time to expiry in the same scenario:

| Bucket | Net error |
| --- | ---: |
| <= 7 days | -21,228 |
| 8-30 days | -3,472 |
| 31-90 days | -4,126 |
| > 90 days | -476 |

### Single-option experiments

The portfolio mixes many effects, so these isolate them on one at-the-money call (S = K = 100, vol 25%, r 4%, no dividend), exact BSM.

**Spot shocks.** Delta-only versus delta-gamma, by time to expiry. Error is full change minus approximation.

| days_to_expiry | spot_shock | option_value | full_change | delta_only_error | delta_gamma_error | delta_gamma_error_pct_of_abs_full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | -10% | 1.0659 | -1.0659 | 4.0531 | -3.5652 | 334.4937 |
| 4 | -5% | 1.0659 | -1.0407 | 1.5188 | -0.3858 | 37.0706 |
| 4 | +5% | 1.0659 | 4.0091 | 1.4496 | -0.4550 | 11.3499 |
| 4 | +10% | 1.0659 | 8.9780 | 3.8590 | -3.7594 | 41.8730 |
| 30 | -10% | 3.0211 | -2.7853 | 2.5403 | -0.2335 | 8.3828 |
| 30 | -5% | 3.0211 | -1.9730 | 0.6898 | -0.0037 | 0.1874 |
| 30 | +5% | 3.0211 | 3.3079 | 0.6451 | -0.0483 | 1.4604 |
| 30 | +10% | 3.0211 | 7.5972 | 2.2716 | -0.5022 | 6.6101 |
| 90 | -10% | 5.4313 | -3.9463 | 1.6164 | 0.0256 | 0.6486 |
| 90 | -5% | 5.4313 | -2.3747 | 0.4066 | 0.0089 | 0.3767 |
| 90 | +5% | 5.4313 | 3.1606 | 0.3792 | -0.0185 | 0.5846 |
| 90 | +10% | 5.4313 | 6.9805 | 1.4178 | -0.1730 | 2.4789 |
| 365 | -10% | 11.8370 | -5.3085 | 0.8132 | 0.0471 | 0.8875 |
| 365 | -5% | 11.8370 | -2.8629 | 0.1979 | 0.0064 | 0.2239 |
| 365 | +5% | 11.8370 | 3.2453 | 0.1844 | -0.0072 | 0.2204 |
| 365 | +10% | 11.8370 | 6.8288 | 0.7070 | -0.0591 | 0.8653 |

**Volatility shocks.** Vega is a first-order approximation in volatility. Compared at the money (strike 100) and out of the money (strike 120).

| strike | days_to_expiry | vol_shock_pts | full_change | vega_approx | error | error_pct_of_abs_full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100.0000 | 30 | -5.0000 | -0.5699 | -0.5700 | 8.16e-05 | 0.0143 |
| 100.0000 | 30 | 5.0000 | 0.5700 | 0.5700 | 2.19e-05 | 0.0038 |
| 100.0000 | 30 | 15.0000 | 1.7098 | 1.7099 | -0.0001 | 0.0068 |
| 100.0000 | 90 | -5.0000 | -0.9802 | -0.9806 | 0.0004 | 0.0429 |
| 100.0000 | 90 | 5.0000 | 0.9807 | 0.9806 | 0.0001 | 0.0115 |
| 100.0000 | 90 | 15.0000 | 2.9413 | 2.9419 | -0.0006 | 0.0203 |
| 100.0000 | 365 | -5.0000 | -1.9120 | -1.9153 | 0.0033 | 0.1742 |
| 100.0000 | 365 | 5.0000 | 1.9162 | 1.9153 | 0.0009 | 0.0467 |
| 100.0000 | 365 | 15.0000 | 5.7412 | 5.7460 | -0.0047 | 0.0824 |
| 120.0000 | 30 | -5.0000 | -0.0143 | -0.0276 | 0.0133 | 93.6075 |
| 120.0000 | 30 | 5.0000 | 0.0482 | 0.0276 | 0.0206 | 42.6765 |
| 120.0000 | 30 | 15.0000 | 0.3030 | 0.0828 | 0.2202 | 72.6728 |
| 120.0000 | 90 | -5.0000 | -0.3266 | -0.4106 | 0.0840 | 25.7291 |
| 120.0000 | 90 | 5.0000 | 0.4837 | 0.4106 | 0.0732 | 15.1229 |
| 120.0000 | 90 | 15.0000 | 1.7896 | 1.2317 | 0.5579 | 31.1736 |
| 120.0000 | 365 | -5.0000 | -1.7386 | -1.8072 | 0.0687 | 3.9492 |
| 120.0000 | 365 | 5.0000 | 1.8538 | 1.8072 | 0.0465 | 2.5108 |
| 120.0000 | 365 | 15.0000 | 5.7308 | 5.4217 | 0.3090 | 5.3927 |

**Time decay.** Theta is the instantaneous rate. Near expiry it changes quickly, and once the option expires inside the horizon the decay is capped at the remaining time value.

| days_to_expiry | days_forward | full_change | theta_approx | error | error_pct_of_abs_full |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | -0.1453 | -0.1359 | -0.0094 | 6.4502 |
| 4 | 7 | -1.0659 | -0.9516 | -0.1143 | 10.7203 |
| 30 | 1 | -0.0534 | -0.0530 | -0.0004 | 0.7586 |
| 30 | 7 | -0.3932 | -0.3710 | -0.0222 | 5.6434 |
| 90 | 1 | -0.0328 | -0.0327 | -7.77e-05 | 0.2368 |
| 90 | 7 | -0.2331 | -0.2292 | -0.0039 | 1.6900 |
| 365 | 1 | -0.0185 | -0.0185 | -9.92e-06 | 0.0535 |
| 365 | 7 | -0.1302 | -0.1297 | -0.0005 | 0.3764 |

**Moneyness.** A +5% spot move on a 30-day call at different strikes (delta-gamma approximation).

| strike | option_value | full_change_for_+5% | delta_gamma_approx | error | error_pct_of_abs_full |
| --- | ---: | ---: | ---: | ---: | ---: |
| 80.0000 | 20.2640 | 4.9987 | 5.0007 | -0.0020 | 0.0403 |
| 90.0000 | 10.4872 | 4.8425 | 4.9069 | -0.0644 | 1.3294 |
| 100.0000 | 3.0211 | 3.3079 | 3.3563 | -0.0483 | 1.4604 |
| 110.0000 | 0.3534 | 0.9366 | 0.8493 | 0.0873 | 9.3186 |
| 120.0000 | 0.0158 | 0.0939 | 0.0681 | 0.0258 | 27.4864 |

## Interpretation

Every number below is computed from the tables above for this run.

- **Spot shock size (portfolio).** With no vol or time move, the absolute error is 2,961 to 3,392 at +-5% and 24,774 to 27,675 at +-10%, i.e. 7.3x to 9.3x larger for twice the shock. As a share of the full P&L the error is 14% / 7% at -5% / +5% and 60% / 28% at -10% / +10%. The first neglected spot term is third order (speed), which predicts roughly 8x for a doubled shock.
- **Effect of the vol shock on the portfolio error.** At fixed +-10% spot and 0 days, the absolute error over the three vol shocks (-5, 0, +5 points) ranges between 26,766 and 28,105 at -10%; 19,011 and 29,302 at +10%. Compare with the spot-only errors above: the vol move changes the error, but the spot move sets its scale.
- **Time to expiry (single at-the-money call).** Delta-gamma error as a share of the full price change at +-10% spot: 4d: 41.9% to 334.5%; 30d: 6.6% to 8.4%; 90d: 0.6% to 2.5%; 365d: 0.9% to 0.9%. Doubling the shock from 5% to 10% multiplied the absolute error by 2.9x to 63.2x across tenors and directions (a pure third-order error would give 8x).
- **Moneyness (30-day call, +5% spot).** Delta-gamma error as a share of the full change, by strike: K=80: 0.0%, K=90: 1.3%, K=100: 1.5%, K=110: 9.3%, K=120: 27.5%. Out-of-the-money options have small deltas but strongly curved prices, so they are the first to break.
- **Volatility shocks (single option).** Vega alone has a maximum error of 0.17% of the change at the money (strike 100) and 93.6% out of the money (strike 120) over -5/+5/+15 vol points and 30 to 365 days. The first-order vega approximation therefore works near the money and degrades away from it, where volga is non-zero.
- **Time decay (single at-the-money call, 7 days forward).** Theta error as a share of the true decay: 4d: 10.7%, 30d: 5.6%, 90d: 1.7%, 365d: 0.4%. Theta is an instantaneous rate and decay accelerates toward expiry, so the approximation is worst for short-dated options.
- **Positions expiring inside the horizon.** Number of positions at or past expiry by days forward: 0d: 0, 1d: 0, 7d: 3. Their revaluation is intrinsic value, and linear theta cannot represent that cap (see `positions_expired` in `scenario_grid.csv`).
- **Where the largest error sits.** In *spot +10%, vol +5pt, 0d*, positions with at most 7 days to expiry account for -21,228 of the net error of -29,302 (72%). The net error is 29,302 while the sum of absolute position errors is 40,359, so offsetting long and short positions hide 27% of the position-level error in the total.
- **What the approximation omits.** Third-and-higher-order spot terms, vanna (spot x vol), volga (vol squared), charm (spot x time) and the expiry cap. This report does not separate their individual contributions; the single-option tables above isolate spot, vol and time one at a time.

Files: `portfolio_valuation.csv`, `portfolio_exposures.csv`, `scenario_grid.csv`.

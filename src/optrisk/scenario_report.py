"""Portfolio scenario report: valuation, exposures, full revaluation vs Greek approximation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .conventions import DAYS_PER_YEAR
from .engines.black_scholes import bsm_arrays
from .io import write_records
from .market import MarketSnapshot
from .portfolio import Portfolio
from .reporting import fmt, md_table
from .risk import PortfolioValuation, RiskEngine, Scenario, ScenarioResult, default_scenarios


def scenario_rows(results: Sequence[ScenarioResult]) -> list[dict[str, object]]:
    return [
        {
            "spot_shock_pct": r.scenario.spot_pct * 100,
            "vol_shock_pts": r.scenario.vol_abs * 100,
            "days_forward": r.scenario.days,
            "base_value": r.base_value,
            "shocked_value": r.shocked_value,
            "full_pnl": r.full_pnl,
            "approx_pnl": r.approx_pnl,
            "delta_pnl": r.delta_pnl,
            "gamma_pnl": r.gamma_pnl,
            "vega_pnl": r.vega_pnl,
            "theta_pnl": r.theta_pnl,
            "approx_error": r.approx_error,
            "approx_error_pct_of_full": r.approx_error_pct,
            "positions_expired": r.n_expired,
        }
        for r in results
    ]


def valuation_rows(v: PortfolioValuation) -> list[dict[str, object]]:
    return [p.__dict__ | {"expiry": p.expiry.isoformat()} for p in v.positions]


def exposure_rows(v: PortfolioValuation) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind, group in (("underlying", v.by_underlying), ("expiry", v.by_expiry), ("underlying+expiry", v.by_underlying_expiry)):
        for g in group:
            rows.append(
                {
                    "grouping": kind,
                    "underlying": g.underlying if kind != "expiry" else "ALL",
                    "expiry": g.expiry.isoformat() if g.expiry else "",
                    "n_positions": g.n_positions,
                    "value": g.value,
                    "delta": g.delta,
                    "delta_notional": g.delta_notional,
                    "gamma": g.gamma,
                    "vega": g.vega,
                    "theta": g.theta,
                    "rho": g.rho,
                }
            )
    return rows


def _matrix(results: Sequence[ScenarioResult], days: int, attr: str) -> str:
    sel = [r for r in results if r.scenario.days == days]
    spots = sorted({r.scenario.spot_pct for r in sel})
    vols = sorted({r.scenario.vol_abs for r in sel}, reverse=True)
    lookup = {(r.scenario.vol_abs, r.scenario.spot_pct): r for r in sel}
    rows = []
    for v in vols:
        rows.append([f"vol {v * 100:+.0f}pt"] + [fmt(float(getattr(lookup[(v, s)], attr)), 0) for s in spots])
    return md_table(["", *[f"spot {s:+.0%}" for s in spots]], rows, align_right_from=1)


def approximation_study() -> dict[str, list[dict[str, object]]]:
    """Single-option experiments that isolate where the Greek approximation degrades (BSM, S=K=100)."""
    s0, k, vol, r, q = 100.0, 100.0, 0.25, 0.04, 0.0

    def val(spot: float, v: float, tau: float, sign: float = 1.0, strike: float = k) -> float:
        return float(bsm_arrays(sign, spot, strike, tau, r, q, v, greeks=False)["price"])

    def greeks(spot: float, v: float, tau: float, sign: float = 1.0, strike: float = k) -> dict[str, np.ndarray]:
        return bsm_arrays(sign, spot, strike, tau, r, q, v)

    spot_rows, vol_rows, time_rows, moneyness_rows = [], [], [], []
    for days in (4, 30, 90, 365):
        tau = days / DAYS_PER_YEAR
        base = val(s0, vol, tau)
        g = greeks(s0, vol, tau)
        for shock in (-0.10, -0.05, 0.05, 0.10):
            ds = s0 * shock
            full = val(s0 + ds, vol, tau) - base
            approx_d = float(g["delta"]) * ds
            approx_dg = approx_d + 0.5 * float(g["gamma"]) * ds * ds
            spot_rows.append(
                {
                    "days_to_expiry": days,
                    "spot_shock": f"{shock:+.0%}",
                    "option_value": base,
                    "full_change": full,
                    "delta_only_error": full - approx_d,
                    "delta_gamma_error": full - approx_dg,
                    "delta_gamma_error_pct_of_abs_full": 100.0 * abs(full - approx_dg) / abs(full),
                }
            )
        for fwd in (1, 7):
            full = val(s0, vol, max(tau - fwd / DAYS_PER_YEAR, 0.0)) - base
            approx = float(g["theta"]) * fwd
            time_rows.append(
                {
                    "days_to_expiry": days,
                    "days_forward": fwd,
                    "full_change": full,
                    "theta_approx": approx,
                    "error": full - approx,
                    "error_pct_of_abs_full": 100.0 * abs(full - approx) / abs(full) if abs(full) > 1e-12 else float("nan"),
                }
            )
    # Vega is exact only to first order; its error depends on strike (volga is ~0 at the money).
    for strike in (100.0, 120.0):
        for days in (30, 90, 365):
            tau = days / DAYS_PER_YEAR
            base = val(s0, vol, tau, 1.0, strike)
            g = greeks(s0, vol, tau, 1.0, strike)
            for dv in (-0.05, 0.05, 0.15):
                full = val(s0, vol + dv, tau, 1.0, strike) - base
                approx = float(g["vega"]) * dv * 100
                vol_rows.append(
                    {
                        "strike": strike,
                        "days_to_expiry": days,
                        "vol_shock_pts": dv * 100,
                        "full_change": full,
                        "vega_approx": approx,
                        "error": full - approx,
                        "error_pct_of_abs_full": 100.0 * abs(full - approx) / abs(full),
                    }
                )
    # Moneyness: a 5% spot move for 30-day options at different strikes
    tau = 30 / DAYS_PER_YEAR
    for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
        base = val(s0, vol, tau, 1.0, strike)
        g = greeks(s0, vol, tau, 1.0, strike)
        ds = 5.0
        full = val(s0 + ds, vol, tau, 1.0, strike) - base
        approx = float(g["delta"]) * ds + 0.5 * float(g["gamma"]) * ds * ds
        moneyness_rows.append(
            {
                "strike": strike,
                "option_value": base,
                "full_change_for_+5%": full,
                "delta_gamma_approx": approx,
                "error": full - approx,
                "error_pct_of_abs_full": 100.0 * abs(full - approx) / abs(full),
            }
        )
    return {"spot": spot_rows, "vol": vol_rows, "time": time_rows, "moneyness": moneyness_rows}


def _interpretation(
    results: Sequence[ScenarioResult],
    study: dict[str, list[dict[str, object]]],
    biggest: ScenarioResult,
    position_err: np.ndarray,
    bucket_err: Mapping[str, float],
) -> list[str]:
    """Bullets whose numbers are all computed from this run's results (no hard-coded findings)."""
    out: list[str] = []

    # 1. spot-shock size, portfolio level (no time, no vol move)
    pure = {round(r.scenario.spot_pct, 6): r for r in results if r.scenario.days == 0 and r.scenario.vol_abs == 0.0}
    if {-0.1, -0.05, 0.05, 0.1} <= pure.keys():
        e5 = [abs(pure[s].approx_error) for s in (-0.05, 0.05)]
        e10 = [abs(pure[s].approx_error) for s in (-0.1, 0.1)]
        out.append(
            f"- **Spot shock size (portfolio).** With no vol or time move, the absolute error is {min(e5):,.0f} to {max(e5):,.0f} at +-5% and "
            f"{min(e10):,.0f} to {max(e10):,.0f} at +-10%, i.e. {min(e10) / max(e5):.1f}x to {max(e10) / min(e5):.1f}x larger for twice the shock. "
            f"As a share of the full P&L the error is {abs(pure[-0.05].approx_error_pct):.0f}% / {abs(pure[0.05].approx_error_pct):.0f}% at -5% / +5% and "
            f"{abs(pure[-0.1].approx_error_pct):.0f}% / {abs(pure[0.1].approx_error_pct):.0f}% at -10% / +10%. "
            "The first neglected spot term is third order (speed), which predicts roughly 8x for a doubled shock."
        )

    # 2. vol and time barely change the spot-driven error
    spread = []
    for s in (-0.1, 0.1):
        errs = [abs(r.approx_error) for r in results if r.scenario.days == 0 and abs(r.scenario.spot_pct - s) < 1e-9]
        if errs:
            spread.append((s, min(errs), max(errs)))
    if spread:
        out.append(
            "- **Effect of the vol shock on the portfolio error.** At fixed +-10% spot and 0 days, the absolute error over the three vol shocks (-5, 0, +5 points) ranges between "
            + "; ".join(f"{lo:,.0f} and {hi:,.0f} at {s:+.0%}" for s, lo, hi in spread)
            + ". Compare with the spot-only errors above: the vol move changes the error, but the spot move sets its scale."
        )

    # 3. single-option spot study by tenor
    spot_rows = study["spot"]
    by_days: dict[int, list[dict[str, object]]] = defaultdict(list)
    for r in spot_rows:
        by_days[int(r["days_to_expiry"])].append(r)  # type: ignore[arg-type]
    parts = []
    for d, rows in sorted(by_days.items()):
        big = [float(r["delta_gamma_error_pct_of_abs_full"]) for r in rows if r["spot_shock"] in ("-10%", "+10%")]  # type: ignore[arg-type]
        parts.append(f"{d}d: {min(big):.1f}% to {max(big):.1f}%")
    ratios = []
    for d, rows in by_days.items():
        for sign in ("-", "+"):
            small = next(abs(float(r["delta_gamma_error"])) for r in rows if r["spot_shock"] == f"{sign}5%")  # type: ignore[arg-type]
            large = next(abs(float(r["delta_gamma_error"])) for r in rows if r["spot_shock"] == f"{sign}10%")  # type: ignore[arg-type]
            ratios.append(large / small)
    out.append(
        "- **Time to expiry (single at-the-money call).** Delta-gamma error as a share of the full price change at +-10% spot: " + "; ".join(parts) + ". "
        f"Doubling the shock from 5% to 10% multiplied the absolute error by {min(ratios):.1f}x to {max(ratios):.1f}x across tenors and directions "
        "(a pure third-order error would give 8x)."
    )

    # 4. moneyness
    mon = study["moneyness"]
    out.append(
        "- **Moneyness (30-day call, +5% spot).** Delta-gamma error as a share of the full change, by strike: "
        + ", ".join(f"K={float(r['strike']):g}: {float(r['error_pct_of_abs_full']):.1f}%" for r in mon)  # type: ignore[arg-type]
        + ". Out-of-the-money options have small deltas but strongly curved prices, so they are the first to break."
    )

    # 5. vol study
    vol_rows = study["vol"]
    atm = [float(r["error_pct_of_abs_full"]) for r in vol_rows if float(r["strike"]) == 100.0]  # type: ignore[arg-type]
    otm = [float(r["error_pct_of_abs_full"]) for r in vol_rows if float(r["strike"]) == 120.0]  # type: ignore[arg-type]
    out.append(
        f"- **Volatility shocks (single option).** Vega alone has a maximum error of {max(atm):.2f}% of the change at the money (strike 100) and {max(otm):.1f}% out of the money "
        f"(strike 120) over -5/+5/+15 vol points and 30 to 365 days. The first-order vega approximation therefore works near the money and degrades away from it, where volga is non-zero."
    )

    # 6. time study
    time_rows = study["time"]
    seven = [(int(r["days_to_expiry"]), float(r["error_pct_of_abs_full"])) for r in time_rows if int(r["days_forward"]) == 7]  # type: ignore[arg-type]
    out.append(
        "- **Time decay (single at-the-money call, 7 days forward).** Theta error as a share of the true decay: "
        + ", ".join(f"{d}d: {e:.1f}%" for d, e in seven)
        + ". Theta is an instantaneous rate and decay accelerates toward expiry, so the approximation is worst for short-dated options."
    )

    # 7. expiry inside the horizon
    by_horizon = {r.scenario.days: r.n_expired for r in results if r.scenario.spot_pct == 0.0 and r.scenario.vol_abs == 0.0}
    out.append(
        "- **Positions expiring inside the horizon.** Number of positions at or past expiry by days forward: "
        + ", ".join(f"{d}d: {n}" for d, n in sorted(by_horizon.items()))
        + ". Their revaluation is intrinsic value, and linear theta cannot represent that cap (see `positions_expired` in `scenario_grid.csv`)."
    )

    # 8. where the worst-scenario error sits
    total = float(sum(bucket_err.values()))
    short = bucket_err.get("<= 7 days", 0.0)
    gross = float(np.abs(position_err).sum())
    out.append(
        f"- **Where the largest error sits.** In *{biggest.scenario.label}*, positions with at most 7 days to expiry account for {short:,.0f} of the net error of {total:,.0f} "
        f"({100 * short / total:.0f}%). The net error is {abs(total):,.0f} while the sum of absolute position errors is {gross:,.0f}, so offsetting long and short positions hide "
        f"{100 * (1 - abs(total) / gross):.0f}% of the position-level error in the total."
        if total
        else "- **Where the largest error sits.** Net error is zero."
    )

    # 9. what is omitted
    out.append(
        "- **What the approximation omits.** Third-and-higher-order spot terms, vanna (spot x vol), volga (vol squared), charm (spot x time) and the expiry cap. "
        "This report does not separate their individual contributions; the single-option tables above isolate spot, vol and time one at a time."
    )
    return out


def _bucket(days: float) -> str:
    if days <= 7:
        return "<= 7 days"
    if days <= 30:
        return "8-30 days"
    if days <= 90:
        return "31-90 days"
    return "> 90 days"


def build_scenario_report(
    portfolio: Portfolio,
    markets: Mapping[str, MarketSnapshot],
    outdir: str | Path,
    risk: RiskEngine | None = None,
    scenarios: Sequence[Scenario] | None = None,
) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    risk = risk or RiskEngine()
    scenarios = list(scenarios) if scenarios is not None else default_scenarios()
    valuation = risk.value(portfolio, markets)
    results = risk.run_scenarios(portfolio, markets, scenarios, detail=True)

    write_records(out / "portfolio_valuation.csv", valuation_rows(valuation))
    write_records(out / "portfolio_exposures.csv", exposure_rows(valuation))
    write_records(out / "scenario_grid.csv", scenario_rows(results))

    lines: list[str] = ["# Portfolio scenario report", ""]
    first = next(iter(markets.values()))
    lines += [
        f"Pricing engine: `{valuation.engine}`. Valuation date: {first.valuation_date}. "
        f"Positions: {len(portfolio)}. Underlyings: {', '.join(portfolio.underlyings)}.",
        "",
        "Units: values in currency; delta in share-equivalents; gamma in share-equivalents per 1.0 move in spot; "
        "vega per +1 volatility point; theta per calendar day; rho per +1 rate point. "
        "Position figures = unit figure x quantity x multiplier.",
        "",
        "## Market snapshots",
        "",
        md_table(
            ["Underlying", "Spot", "Rate", "Dividend yield", "Volatility"],
            [[m.underlying, m.spot, m.rate, m.dividend_yield, m.volatility] for m in markets.values()],
            digits=4,
        ),
        "",
        "## Portfolio value and aggregate Greeks",
        "",
        md_table(
            ["Measure", "Value"],
            [
                ("Total value", valuation.total_value),
                ("Delta (share-equivalents)", valuation.totals["delta"]),
                ("Delta notional (currency)", valuation.totals["delta_notional"]),
                ("Gamma", valuation.totals["gamma"]),
                ("Vega (per vol point)", valuation.totals["vega"]),
                ("Theta (per day)", valuation.totals["theta"]),
                ("Rho (per rate point)", valuation.totals["rho"]),
            ],
            digits=2,
        ),
        "",
        "## Exposure by underlying",
        "",
        md_table(
            ["Underlying", "Positions", "Value", "Delta", "Delta notional", "Gamma", "Vega", "Theta/day"],
            [[g.underlying, g.n_positions, g.value, g.delta, g.delta_notional, g.gamma, g.vega, g.theta] for g in valuation.by_underlying],
            digits=1,
        ),
        "",
        "## Exposure by expiry",
        "",
        md_table(
            ["Expiry", "Positions", "Value", "Delta", "Gamma", "Vega", "Theta/day"],
            [[g.expiry.isoformat() if g.expiry else "", g.n_positions, g.value, g.delta, g.gamma, g.vega, g.theta] for g in valuation.by_expiry],
            digits=1,
        ),
        "",
        "## Exposure by underlying and expiry",
        "",
        md_table(
            ["Underlying", "Expiry", "Positions", "Value", "Delta", "Gamma", "Vega", "Theta/day"],
            [
                [g.underlying, g.expiry.isoformat() if g.expiry else "", g.n_positions, g.value, g.delta, g.gamma, g.vega, g.theta]
                for g in valuation.by_underlying_expiry
            ],
            digits=1,
        ),
        "",
        "## Scenario results",
        "",
        "Each scenario is a joint relative spot shock, absolute vol shock (in vol points) and a number of calendar days forward, "
        "applied to every underlying. *Full revaluation* re-prices every position. *Greek approximation* is "
        "`delta*dS + 0.5*gamma*dS^2 + vega*dVol + theta*days`, using Greeks at the base market. Values are portfolio P&L versus the base value.",
        "",
    ]
    for days in sorted({s.days for s in scenarios}):
        lines += [
            f"### {days} day(s) forward",
            "",
            "**Full revaluation P&L**",
            "",
            _matrix(results, days, "full_pnl"),
            "",
            "**Greek approximation P&L**",
            "",
            _matrix(results, days, "approx_pnl"),
            "",
            "**Approximation error (full - approximation)**",
            "",
            _matrix(results, days, "approx_error"),
            "",
        ]

    biggest = max(results, key=lambda r: abs(r.approx_error))
    lines += [
        "### P&L attribution of the approximation (days = 0)",
        "",
        md_table(
            ["Scenario", "Delta", "Gamma", "Vega", "Theta", "Approx total", "Full", "Error", "Error % of full"],
            [
                [r.scenario.label, r.delta_pnl, r.gamma_pnl, r.vega_pnl, r.theta_pnl, r.approx_pnl, r.full_pnl, r.approx_error, r.approx_error_pct]
                for r in results
                if r.scenario.days == 0 and r.scenario.vol_abs == 0.0 and r.scenario.spot_pct != 0.0
            ],
            digits=0,
        ),
        "",
        "## Where the Greek approximation breaks down",
        "",
        f"Largest absolute error in the grid: **{biggest.approx_error:,.0f}** at *{biggest.scenario.label}* "
        f"(full {biggest.full_pnl:,.0f}, approximation {biggest.approx_pnl:,.0f}).",
        "",
    ]

    # Which positions and expiry buckets drive the error in the worst scenario?
    assert biggest.position_full_pnl is not None and biggest.position_approx_pnl is not None
    err = biggest.position_full_pnl - biggest.position_approx_pnl
    pos_rows = []
    bucket_err: dict[str, float] = defaultdict(float)
    for i, p in enumerate(portfolio):
        m = markets[p.contract.underlying]
        dte = (p.contract.expiry - m.valuation_date).days
        bucket_err[_bucket(dte)] += float(err[i])
        pos_rows.append(
            [p.contract.label, f"{p.contract.option_type.value} {p.contract.strike:g}", dte, p.quantity,
             float(biggest.position_full_pnl[i]), float(biggest.position_approx_pnl[i]), float(err[i])]
        )
    pos_rows.sort(key=lambda r: -abs(r[-1]))
    lines += [
        f"Largest position-level errors in that scenario (*{biggest.scenario.label}*):",
        "",
        md_table(["Position", "Option", "Days to expiry", "Quantity", "Full P&L", "Approx P&L", "Error"], pos_rows[:6], digits=0, align_right_from=2),
        "",
        "Error by time to expiry in the same scenario:",
        "",
        md_table(
            ["Bucket", "Net error"],
            [[b, bucket_err[b]] for b in ("<= 7 days", "8-30 days", "31-90 days", "> 90 days") if b in bucket_err],
            digits=0,
        ),
        "",
    ]

    study = approximation_study()

    def table(rows: list[dict[str, object]], cols: Sequence[str], digits: int = 4) -> str:
        return md_table(list(cols), [[r[c] for c in cols] for r in rows], digits=digits, align_right_from=1)

    lines += [
        "### Single-option experiments",
        "",
        "The portfolio mixes many effects, so these isolate them on one at-the-money call (S = K = 100, vol 25%, r 4%, no dividend), exact BSM.",
        "",
        "**Spot shocks.** Delta-only versus delta-gamma, by time to expiry. Error is full change minus approximation.",
        "",
        table(study["spot"], ["days_to_expiry", "spot_shock", "option_value", "full_change", "delta_only_error", "delta_gamma_error", "delta_gamma_error_pct_of_abs_full"]),
        "",
        "**Volatility shocks.** Vega is a first-order approximation in volatility. Compared at the money (strike 100) and out of the money (strike 120).",
        "",
        table(study["vol"], ["strike", "days_to_expiry", "vol_shock_pts", "full_change", "vega_approx", "error", "error_pct_of_abs_full"]),
        "",
        "**Time decay.** Theta is the instantaneous rate. Near expiry it changes quickly, and once the option expires inside the horizon the decay is capped at the remaining time value.",
        "",
        table(study["time"], ["days_to_expiry", "days_forward", "full_change", "theta_approx", "error", "error_pct_of_abs_full"]),
        "",
        "**Moneyness.** A +5% spot move on a 30-day call at different strikes (delta-gamma approximation).",
        "",
        table(study["moneyness"], ["strike", "option_value", "full_change_for_+5%", "delta_gamma_approx", "error", "error_pct_of_abs_full"]),
        "",
        "## Interpretation",
        "",
        "Every number below is computed from the tables above for this run.",
        "",
        *_interpretation(results, study, biggest, err, bucket_err),
        "",
        "Files: `portfolio_valuation.csv`, `portfolio_exposures.csv`, `scenario_grid.csv`.",
        "",
    ]
    path = out / "scenario_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

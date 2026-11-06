"""Command-line interface: ``optrisk <command>`` (or ``python -m optrisk <command>``).

Commands
--------
generate     write sample contracts, positions, market snapshots and an implied-vol quote chain
price        price contracts with the analytic, binomial or Monte Carlo engine
implied-vol  solve implied volatility for observed quotes
risk         value a portfolio: total value, aggregate Greeks, exposures
scenario     portfolio scenario report (full revaluation vs Greek approximation)
validate     run the full validation suite (includes the QuantLib comparison)
quantlib-compare  run only the QuantLib comparison
benchmark    run the performance benchmarks
demo         the main demonstration, end to end, in about a minute
reproduce    demo + full validation + full benchmarks (writes everything under reports/)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .engines import BinomialEngine, BlackScholesEngine, MonteCarloEngine, PricingEngine
from .errors import OptRiskError
from .generate import sample_markets, sample_portfolio, sample_quote_chain, write_sample_data
from .implied_vol import implied_volatility
from .io import load_contracts, load_markets, load_positions, load_quotes, write_records
from .reporting import fmt, md_table
from .risk import RiskEngine, Scenario, default_scenarios
from .scenario_report import build_scenario_report, exposure_rows, valuation_rows


# ------------------------------------------------------------------------------ helpers
def _print_table(rows: Sequence[dict[str, Any]], columns: Sequence[str] | None = None, digits: int = 5) -> None:
    if not rows:
        print("(no rows)")
        return
    cols = list(columns) if columns else list(rows[0].keys())
    cells = [[fmt(r.get(c, ""), digits) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(v.rjust(w) if i else v.ljust(w) for i, (v, w) in enumerate(zip(row, widths))))


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("engine")
    g.add_argument("--engine", choices=("bsm", "binomial", "mc"), default="bsm", help="pricing engine (default: bsm)")
    g.add_argument("--steps", type=int, default=500, help="binomial tree steps (default 500)")
    g.add_argument("--paths", type=int, default=200_000, help="Monte Carlo paths (default 200000)")
    g.add_argument("--seed", type=int, default=12345, help="Monte Carlo seed (default 12345)")
    g.add_argument("--no-antithetic", action="store_true", help="disable antithetic variates")
    g.add_argument("--workers", type=int, default=1, help="Monte Carlo worker processes (default 1)")


def _make_engine(args: argparse.Namespace) -> PricingEngine:
    if args.engine == "bsm":
        return BlackScholesEngine()
    if args.engine == "binomial":
        return BinomialEngine(args.steps)
    return MonteCarloEngine(args.paths, seed=args.seed, antithetic=not args.no_antithetic, n_workers=args.workers)


def _write_if(path: str | None, rows: list[dict[str, Any]]) -> None:
    if path:
        write_records(path, rows)
        print(f"wrote {path}")


# ------------------------------------------------------------------------------ commands
def cmd_generate(args: argparse.Namespace) -> int:
    paths = write_sample_data(args.outdir)
    for name, p in paths.items():
        print(f"{name:16s} {p}")
    return 0


def cmd_price(args: argparse.Namespace) -> int:
    contracts = load_contracts(args.contracts)
    markets = load_markets(args.market)
    engine = _make_engine(args)
    rows: list[dict[str, Any]] = []
    n_err = 0
    try:
        for cid, c in contracts.items():
            row: dict[str, Any] = {"contract_id": cid, "type": c.option_type.value, "strike": c.strike, "expiry": c.expiry.isoformat()}
            try:
                m = markets.get(c.underlying)
                if m is None:
                    raise OptRiskError(f"no market snapshot for underlying {c.underlying!r}")
                res = engine.price(c, m, greeks=not args.no_greeks)
                row.update(res.to_dict())
                row["status"] = "ok"
            except OptRiskError as exc:
                n_err += 1
                row["status"] = f"error: {exc}"
            rows.append(row)
    finally:
        if isinstance(engine, MonteCarloEngine):
            engine.close()
    cols = ["contract_id", "type", "strike", "expiry", "price", "std_error", "delta", "gamma", "vega", "theta", "rho", "status"]
    _print_table(rows, [c for c in cols if any(c in r for r in rows)])
    print(f"\nengine={engine.name}; priced {len(rows) - n_err}/{len(rows)}; price per unit of underlying; "
          "vega per vol point, theta per day, rho per rate point")
    _write_if(args.output, rows)
    return 3 if n_err else 0


def cmd_implied_vol(args: argparse.Namespace) -> int:
    contracts = load_contracts(args.contracts)
    markets = load_markets(args.market)
    quotes = load_quotes(args.quotes)
    methods = ("bisection", "brentq") if args.method == "both" else (args.method,)
    rows: list[dict[str, Any]] = []
    for cid, price in quotes.items():
        c = contracts.get(cid)
        if c is None:
            rows.append({"contract_id": cid, "quote": price, "status": "unknown_contract"})
            continue
        m = markets.get(c.underlying)
        if m is None:
            rows.append({"contract_id": cid, "quote": price, "status": "no_market"})
            continue
        for method in methods:
            r = implied_volatility(price, c, m, method=method)
            rows.append({"contract_id": cid, "quote": price, "method": method, "status": r.status.value, "implied_vol": r.vol,
                         "iterations": r.iterations, "function_evals": r.function_evals, "residual": r.residual, "vega": r.vega, "message": r.message})
    _print_table(rows, ["contract_id", "quote", "method", "status", "implied_vol", "iterations", "function_evals", "residual"], digits=8)
    n_ok = sum(r["status"] == "converged" for r in rows)
    print(f"\n{n_ok}/{len(rows)} solves converged; failures carry an explicit status (see --output for messages)")
    _write_if(args.output, rows)
    return 0


def cmd_risk(args: argparse.Namespace) -> int:
    contracts = load_contracts(args.contracts)
    markets = load_markets(args.market)
    portfolio = load_positions(args.positions, contracts)
    engine = _make_engine(args)
    try:
        v = RiskEngine(engine).value(portfolio, markets)
    finally:
        if isinstance(engine, MonteCarloEngine):
            engine.close()
    print(md_table(["Measure", "Value"], [["Total value", v.total_value], *[[k, x] for k, x in v.totals.items()]], digits=2, align_right_from=1))
    print("\nBy underlying:")
    _print_table([g.__dict__ | {"expiry": ""} for g in v.by_underlying], ["underlying", "n_positions", "value", "delta", "gamma", "vega", "theta"], digits=1)
    print("\nBy underlying and expiry:")
    _print_table([g.__dict__ | {"expiry": g.expiry.isoformat() if g.expiry else ""} for g in v.by_underlying_expiry],
                 ["underlying", "expiry", "n_positions", "value", "delta", "gamma", "vega", "theta"], digits=1)
    if args.output_dir:
        out = Path(args.output_dir)
        write_records(out / "portfolio_valuation.csv", valuation_rows(v))
        write_records(out / "portfolio_exposures.csv", exposure_rows(v))
        print(f"\nwrote {out}/portfolio_valuation.csv and portfolio_exposures.csv")
    return 0


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def cmd_scenario(args: argparse.Namespace) -> int:
    contracts = load_contracts(args.contracts)
    markets = load_markets(args.market)
    portfolio = load_positions(args.positions, contracts)
    scenarios = default_scenarios(
        [x / 100 for x in _floats(args.spot_shocks)],
        [x / 100 for x in _floats(args.vol_shocks)],
        [int(x) for x in _floats(args.days)],
    )
    engine = _make_engine(args)
    try:
        path = build_scenario_report(portfolio, markets, args.outdir, risk=RiskEngine(engine), scenarios=scenarios)
    finally:
        if isinstance(engine, MonteCarloEngine):
            engine.close()
    print(f"wrote {path} ({len(scenarios)} scenarios)")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .validation.report import run_validation

    results, md = run_validation(args.outdir, quick=args.quick, progress=print)
    for r in results:
        print(f"  [{r.status:7s}] {r.title}: {r.n_cases - r.n_failed}/{r.n_cases} cases ok" if r.status != "skipped" else f"  [skipped] {r.title}: {r.notes}")
    failed = [r for r in results if r.status == "failed"]
    print(f"\nreport: {md}")
    print("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED")
    return 1 if failed else 0


def cmd_quantlib_compare(args: argparse.Namespace) -> int:
    from .validation.report import run_quantlib_comparison

    try:
        results, md = run_quantlib_comparison(args.outdir, quick=args.quick, progress=print)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for r in results:
        print(f"  [{r.status:7s}] {r.title}: {r.n_cases - r.n_failed}/{r.n_cases} comparisons within tolerance")
    failed = [r for r in results if r.status != "passed"]
    print(f"\nreport: {md}")
    return 1 if failed else 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmarks import run_benchmarks

    path = run_benchmarks(args.outdir, quick=args.quick, progress=print)
    print(f"report: {path}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .implied_vol import implied_volatility
    from .validation.report import run_validation

    out = Path(args.outdir)
    t_start = time.perf_counter()
    print(f"optrisk {__version__} demo -> {out}\n")

    print("1) Generating controlled sample data")
    paths = write_sample_data(out / "data")
    for name, p in paths.items():
        print(f"   {name:16s} {p}")

    markets = sample_markets()
    portfolio = sample_portfolio()
    chain, quotes, truth, notes = sample_quote_chain(markets)

    print("\n2) Three pricing methods on one option (SIM_A call, K=105, 180 days)")
    contract = next(p.contract for p in portfolio if p.contract.contract_id == "A7")
    rows = []
    with MonteCarloEngine(400_000, seed=2026) as mc:
        for eng in (BlackScholesEngine(), BinomialEngine(1000), mc):
            res = eng.price(contract, markets["SIM_A"])
            g = res.greeks
            rows.append({"engine": res.method, "price": res.price, "std_error": res.std_error if res.std_error is not None else "", "delta": g.delta,
                         "gamma": g.gamma, "vega": g.vega, "theta": g.theta, "rho": g.rho, "runtime": f"{res.runtime_s * 1e3:.2f} ms"})
    _print_table(rows)
    write_records(out / "pricing_comparison.csv", rows)

    print("\n3) Implied volatility on a generated quote chain (bisection vs Brent)")
    stats = {m: {"ok": 0, "evals": 0, "max_err": 0.0} for m in ("bisection", "brentq")}
    failures = []
    iv_rows = []
    for c in chain:
        for method in stats:
            r = implied_volatility(quotes[c.contract_id], c, markets["SIM_A"], method=method)
            iv_rows.append({"contract_id": c.contract_id, "method": method, "quote": quotes[c.contract_id], "status": r.status.value,
                            "implied_vol": r.vol, "true_vol": truth.get(c.contract_id, ""), "iterations": r.iterations,
                            "function_evals": r.function_evals, "residual": r.residual, "note": notes.get(c.contract_id, "")})
            if r.ok:
                stats[method]["ok"] += 1
                stats[method]["evals"] += r.function_evals
                stats[method]["max_err"] = max(stats[method]["max_err"], abs(r.vol - truth[c.contract_id]))
            elif method == "bisection":
                failures.append((c.contract_id, r.status.value))
    write_records(out / "implied_vol_results.csv", iv_rows)
    for m, s in stats.items():
        print(f"   {m:9s}: {s['ok']} solved, mean function evaluations {s['evals'] / max(s['ok'], 1):.1f}, max |vol - true vol| {s['max_err']:.1e}")
    print("   explicit failure statuses: " + ", ".join(f"{cid}={st}" for cid, st in failures))

    print("\n4) Portfolio valuation, exposures and scenarios")
    report = build_scenario_report(portfolio, markets, out)
    v = RiskEngine().value(portfolio, markets)
    print(f"   total value {v.total_value:,.0f}; delta {v.totals['delta']:,.0f}; gamma {v.totals['gamma']:,.0f}; vega {v.totals['vega']:,.0f}; theta/day {v.totals['theta']:,.0f}")
    res = RiskEngine().run_scenarios(portfolio, markets, [Scenario(-0.10, 0.05, 7)])[0]
    print(f"   stress scenario ({res.scenario.label}): full {res.full_pnl:,.0f}, Greek approximation {res.approx_pnl:,.0f}, error {res.approx_error:,.0f}")
    print(f"   report: {report}")

    if not args.skip_validate:
        print("\n5) Validation against analytic identities, finite differences and QuantLib (quick grid)")
        results, md = run_validation(out, quick=True)
        for r in results:
            if r.status == "skipped":
                print(f"   [skipped] {r.title}: {r.notes}")
        n_pass = sum(r.status == "passed" for r in results)
        n_fail = sum(r.status == "failed" for r in results)
        print(f"   {n_pass} passed, {n_fail} failed, {sum(r.status == 'skipped' for r in results)} skipped; report: {md}")
        if n_fail:
            return 1
    print(f"\ndone in {time.perf_counter() - t_start:.1f}s")
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    from .benchmarks import run_benchmarks
    from .validation.report import run_validation

    out = Path(args.outdir)
    code = cmd_demo(argparse.Namespace(outdir=str(out / "demo"), skip_validate=True))
    print("\nFull validation grid")
    results, md = run_validation(out, quick=False, progress=print)
    failed = [r for r in results if r.status == "failed"]
    print(f"  report: {md}; {'all checks passed' if not failed else str(len(failed)) + ' FAILED'}")
    print("\nQuantLib comparison")
    ql_code = cmd_quantlib_compare(argparse.Namespace(outdir=str(out), quick=False))
    print("\nBenchmarks (several minutes)")
    path = run_benchmarks(out, quick=False, progress=print)
    print(f"  report: {path}")
    return code or ql_code or (1 if failed else 0)


# ------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="optrisk", description="European option pricing, implied volatility and portfolio scenario risk.")
    p.add_argument("--version", action="version", version=f"optrisk {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("generate", help="write sample input files")
    s.add_argument("--outdir", default="data")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("price", help="price contracts")
    s.add_argument("--contracts", required=True)
    s.add_argument("--market", required=True)
    s.add_argument("--no-greeks", action="store_true")
    s.add_argument("--output", help="write results to .csv or .json")
    _add_engine_args(s)
    s.set_defaults(func=cmd_price)

    s = sub.add_parser("implied-vol", help="implied volatility from quotes")
    s.add_argument("--contracts", required=True)
    s.add_argument("--market", required=True)
    s.add_argument("--quotes", required=True)
    s.add_argument("--method", choices=("bisection", "brentq", "both"), default="both")
    s.add_argument("--output", help="write results to .csv or .json")
    s.set_defaults(func=cmd_implied_vol)

    s = sub.add_parser("risk", help="portfolio value, Greeks and exposures")
    s.add_argument("--contracts", required=True)
    s.add_argument("--positions", required=True)
    s.add_argument("--market", required=True)
    s.add_argument("--output-dir")
    _add_engine_args(s)
    s.set_defaults(func=cmd_risk)

    s = sub.add_parser("scenario", help="scenario report")
    s.add_argument("--contracts", required=True)
    s.add_argument("--positions", required=True)
    s.add_argument("--market", required=True)
    s.add_argument("--outdir", default="reports")
    s.add_argument("--spot-shocks", default="-10,-5,0,5,10", help="percent, comma separated")
    s.add_argument("--vol-shocks", default="-5,0,5", help="vol points, comma separated")
    s.add_argument("--days", default="0,1,7", help="calendar days forward, comma separated")
    _add_engine_args(s)
    s.set_defaults(func=cmd_scenario)

    s = sub.add_parser("validate", help="run the validation suite")
    s.add_argument("--outdir", default="reports")
    s.add_argument("--quick", action="store_true", help="smaller grid, for a fast check")
    s.set_defaults(func=cmd_validate)

    s = sub.add_parser("quantlib-compare", help="compare against QuantLib only")
    s.add_argument("--outdir", default="reports")
    s.add_argument("--quick", action="store_true", help="smaller grid")
    s.set_defaults(func=cmd_quantlib_compare)

    s = sub.add_parser("benchmark", help="run performance benchmarks")
    s.add_argument("--outdir", default="reports")
    s.add_argument("--quick", action="store_true", help="reduced sizes and repeats; not for reporting")
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("demo", help="main demonstration")
    s.add_argument("--outdir", default="reports/demo")
    s.add_argument("--skip-validate", action="store_true")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("reproduce", help="demo + full validation + full benchmarks")
    s.add_argument("--outdir", default="reports")
    s.set_defaults(func=cmd_reproduce)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except OptRiskError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # e.g. `optrisk price ... | head`
        return 0


if __name__ == "__main__":
    sys.exit(main())

"""Performance benchmarks.

Method
------
* Wall-clock time from ``time.perf_counter``; one warm-up call before timing.
* Each measurement repeats until ``min_repeats`` samples exist and a time budget is used up (capped at
  ``max_repeats``). Calls faster than ~0.2 ms are batched so one sample lasts about 2 ms.
* Statistics reported: median, min, max, inter-quartile range and coefficient of variation (CV).
  Speedups use medians; read them together with the CV. Nothing else was paused during runs, so
  background load on the machine is part of the variability.
* Compute time is measured separately from (a) file loading and object construction, and
  (b) process start-up and worker-pool creation. Pools are created once, warmed, then reused.
* Inputs are generated from fixed seeds so workloads are identical across runs.
"""

from __future__ import annotations

import math
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import OptionContract, OptionType
from .engines import BinomialEngine, BlackScholesEngine, MonteCarloEngine
from .engines.base import PricingEngine
from .engines.black_scholes import bsm_arrays, bsm_scalar
from .generate import SAMPLE_VALUATION_DATE, contracts_rows, markets_rows, positions_rows, random_portfolio
from .io import load_contracts, load_markets, load_positions, write_json, write_records
from .market import MarketSnapshot
from .reporting import dict_table, environment, environment_table, md_table
from .risk import RiskEngine, _Book, default_scenarios


# ------------------------------------------------------------------------------ timing helper
@dataclass
class Timing:
    samples: list[float]  # seconds per call

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def min(self) -> float:
        return min(self.samples)

    @property
    def max(self) -> float:
        return max(self.samples)

    @property
    def cv(self) -> float:
        if self.n < 2:
            return float("nan")
        mean = statistics.fmean(self.samples)
        return statistics.stdev(self.samples) / mean if mean else float("nan")

    @property
    def iqr(self) -> float:
        if self.n < 4:
            return self.max - self.min
        q = statistics.quantiles(self.samples, n=4)
        return q[2] - q[0]

    def as_dict(self) -> dict[str, float]:
        return {"median_s": self.median, "min_s": self.min, "max_s": self.max, "iqr_s": self.iqr, "cv": self.cv, "repeats": self.n}


def measure(fn: Callable[[], Any], *, min_repeats: int = 5, max_repeats: int = 25, budget_s: float = 2.0, warmup: int = 1) -> Timing:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    fn()
    single = time.perf_counter() - t0
    inner = max(1, math.ceil(2e-3 / single)) if single < 2e-4 else 1
    samples: list[float] = []
    start = time.perf_counter()
    while len(samples) < max_repeats:
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
        if len(samples) >= min_repeats and time.perf_counter() - start > budget_s:
            break
    return Timing(samples)


def human(seconds: float) -> str:
    if seconds != seconds:
        return "n/a"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def _x(ratio: float) -> str:
    return f"{ratio:.2f}x" if ratio < 1 else f"{ratio:.1f}x"


def _t(timing: Timing) -> str:
    return f"{human(timing.median)} (CV {timing.cv * 100:.0f}%, n={timing.n})"


# ------------------------------------------------------------------------------ B1: scalar vs vectorised
def _random_arrays(n: int, seed: int = 11) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    spot = rng.uniform(50.0, 150.0, n)
    return {
        "sign": rng.choice([-1.0, 1.0], n),
        "spot": spot,
        "strike": spot * np.exp(rng.uniform(-0.3, 0.3, n)),
        "tau": rng.uniform(0.01, 2.0, n),
        "rate": rng.uniform(0.0, 0.05, n),
        "div": rng.uniform(0.0, 0.03, n),
        "vol": rng.uniform(0.10, 0.60, n),
    }


def bench_vectorization(sizes: list[int], oo_limit: int, fast: bool) -> list[dict[str, Any]]:
    rows = []
    for n in sizes:
        arr = _random_arrays(n)
        lists = [arr[k].tolist() for k in ("sign", "spot", "strike", "tau", "rate", "div", "vol")]

        def scalar() -> list:
            return [bsm_scalar(int(s), a, k, t, r, q, v) for s, a, k, t, r, q, v in zip(*lists)]

        def vector() -> dict:
            return bsm_arrays(arr["sign"], arr["spot"], arr["strike"], arr["tau"], arr["rate"], arr["div"], arr["vol"])

        kw = dict(min_repeats=3, max_repeats=9 if fast else 15, budget_s=0.5 if fast else 2.0)
        ts, tv = measure(scalar, **kw), measure(vector, **kw)
        sc, ve = scalar(), vector()
        max_diff = max(
            float(np.max(np.abs(np.array([x[0] for x in sc]) - ve["price"]))),
            float(np.max(np.abs(np.array([x[1].delta for x in sc]) - ve["delta"]))),
        )
        row: dict[str, Any] = {
            "contracts": n,
            "scalar_python_loop": ts,
            "numpy_vectorised": tv,
            "speedup_median": ts.median / tv.median,
            "max_abs_diff_price_delta": max_diff,
        }
        if n <= oo_limit:
            engine = BlackScholesEngine()
            contracts, markets = [], []
            for i in range(n):
                typ = OptionType.CALL if lists[0][i] == 1 else OptionType.PUT
                contracts.append(OptionContract("B", lists[2][i], SAMPLE_VALUATION_DATE + timedelta(days=max(1, round(lists[3][i] * 365))), typ))
                markets.append(MarketSnapshot("B", SAMPLE_VALUATION_DATE, lists[1][i], lists[4][i], lists[5][i], lists[6][i]))

            def oo() -> list:
                return [engine.price(c, m) for c, m in zip(contracts, markets)]

            row["object_interface_loop"] = measure(oo, **kw)
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------ B2: portfolio scenarios
class _GenericBSM(PricingEngine):
    """BSM behind the generic interface, so RiskEngine prices position by position."""

    name = "bsm_generic_route"

    def __init__(self) -> None:
        self._inner = BlackScholesEngine()

    def _price(self, contract, market, tau, greeks):  # type: ignore[no-untyped-def]
        return self._inner._price(contract, market, tau, greeks)


def bench_portfolio(sizes: list[int], generic_limit: int, fast: bool) -> list[dict[str, Any]]:
    rows = []
    scenarios = default_scenarios()
    for n in sizes:
        pf, mk = random_portfolio(n, n_underlyings=20)
        fast_engine, slow_engine = RiskEngine(BlackScholesEngine()), RiskEngine(_GenericBSM())
        kw = dict(min_repeats=3, max_repeats=7 if fast else 11, budget_s=0.5 if fast else 2.0)
        row: dict[str, Any] = {"positions": n, "scenarios": len(scenarios), "pricings": n * (len(scenarios) + 1)}
        row["book_build"] = measure(lambda: _Book(pf, mk), **kw)
        row["vectorised_total"] = measure(lambda: fast_engine.run_scenarios(pf, mk, scenarios), **kw)
        if n <= generic_limit:
            row["position_by_position_total"] = measure(lambda: slow_engine.run_scenarios(pf, mk, scenarios), min_repeats=3, max_repeats=5, budget_s=1.0)
            row["speedup_median"] = row["position_by_position_total"].median / row["vectorised_total"].median
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------ B3: Monte Carlo workers
def _noop(_: int) -> int:
    return 0


def bench_mc_parallel(path_counts: list[int], workers: list[int], fast: bool) -> list[dict[str, Any]]:
    vd = SAMPLE_VALUATION_DATE
    contract = OptionContract("MC", 105.0, vd + timedelta(days=180), OptionType.CALL)
    market = MarketSnapshot("MC", vd, 100.0, 0.03, 0.01, 0.25)
    n_streams = 40  # divisible by 1, 2, 4, 5, 8, 10: even load per worker
    rows = []
    for n_paths in path_counts:
        reference_price = None
        for w in workers:
            with MonteCarloEngine(n_paths, seed=5, antithetic=True, n_streams=n_streams, n_workers=w) as eng:
                t0 = time.perf_counter()
                first = eng.price(contract, market, greeks=False)  # includes pool creation for w > 1
                first_call = time.perf_counter() - t0
                timing = measure(lambda: eng.price(contract, market, greeks=False), min_repeats=3, max_repeats=7 if fast else 11, budget_s=0.5 if fast else 2.0, warmup=1)
            if reference_price is None:
                reference_price = first.price
            rows.append(
                {
                    "paths": n_paths,
                    "workers": w,
                    "streams": n_streams,
                    "compute": timing,
                    "first_call_incl_pool_start_s": first_call,
                    "price": first.price,
                    "price_identical_to_1_worker": first.price == reference_price,
                }
            )
    base = {}
    for r in rows:
        if r["workers"] == workers[0]:
            base[r["paths"]] = r["compute"].median
    for r in rows:
        r["speedup_vs_1_worker"] = base[r["paths"]] / r["compute"].median
        r["efficiency"] = r["speedup_vs_1_worker"] / r["workers"]
    return rows


def bench_mc_batch(workers: list[int], n_contracts: int, n_paths: int, fast: bool) -> list[dict[str, Any]]:
    """Many small pricing calls: per-call overhead of parallel dispatch."""
    vd = SAMPLE_VALUATION_DATE
    market = MarketSnapshot("MC", vd, 100.0, 0.03, 0.01, 0.25)
    contracts = [OptionContract("MC", 80.0 + i * (40.0 / n_contracts), vd + timedelta(days=90), OptionType.CALL) for i in range(n_contracts)]
    rows = []
    for w in workers:
        with MonteCarloEngine(n_paths, seed=5, n_streams=40, n_workers=w) as eng:
            timing = measure(lambda: [eng.price(c, market, greeks=False) for c in contracts], min_repeats=3, max_repeats=5 if fast else 7, budget_s=1.0, warmup=1)
        rows.append({"contracts": n_contracts, "paths_each": n_paths, "workers": w, "total": timing})
    for r in rows:
        r["speedup_vs_1_worker"] = rows[0]["total"].median / r["total"].median
    return rows


# ------------------------------------------------------------------------------ B4: startup and I/O
def bench_startup(fast: bool) -> dict[str, Any]:
    reps = 3 if fast else 7

    def run(code: str) -> Timing:
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
            samples.append(time.perf_counter() - t0)
        return Timing(samples)

    interp = run("pass")
    result: dict[str, Any] = {
        "python_interpreter_only": interp,
        "import_numpy": run("import numpy"),
        "import_scipy_special_optimize": run("import scipy.special, scipy.optimize"),
        "import_optrisk": run("import optrisk, optrisk.risk, optrisk.implied_vol"),
    }
    pools = []
    for w in (2, 4, 8):
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=w) as pool:
                list(pool.map(_noop, range(w * 2)))  # force every worker to start
            samples.append(time.perf_counter() - t0)
        pools.append({"workers": w, "create_run_shutdown": Timing(samples)})
    result["pools"] = pools
    return result


def bench_io_vs_compute(sizes: list[int], fast: bool) -> list[dict[str, Any]]:
    rows = []
    scenarios = default_scenarios()
    for n in sizes:
        pf, mk = random_portfolio(n, n_underlyings=20)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_records(d / "contracts.csv", contracts_rows([p.contract for p in pf]))
            write_records(d / "positions.csv", positions_rows(pf))
            write_records(d / "market.csv", markets_rows(mk))
            size_mb = sum(f.stat().st_size for f in d.iterdir()) / 1e6

            def load() -> Any:
                contracts = load_contracts(d / "contracts.csv")
                markets = load_markets(d / "market.csv")
                return load_positions(d / "positions.csv", contracts), markets

            kw = dict(min_repeats=3, max_repeats=5 if fast else 7, budget_s=1.0)
            t_load = measure(load, **kw)
            loaded_pf, loaded_mk = load()
            engine = RiskEngine(BlackScholesEngine())
            t_val = measure(lambda: engine.value(loaded_pf, loaded_mk), **kw)
            t_scn = measure(lambda: engine.run_scenarios(loaded_pf, loaded_mk, scenarios), **kw)
        rows.append({"positions": n, "csv_mb": size_mb, "load_csv_and_build_objects": t_load, "valuation_with_greeks": t_val, "scenario_grid_45": t_scn,
                     "load_over_scenario": t_load.median / t_scn.median})
    return rows


# ------------------------------------------------------------------------------ B5: accuracy vs budget
_ACCURACY_CASES = [
    ("ATM call 90d", 1, 100.0, 90, 0.25, 0.03, 0.01),
    ("OTM put 6m", -1, 90.0, 180, 0.20, 0.02, 0.0),
    ("ITM call 1y", 1, 90.0, 365, 0.30, 0.04, 0.02),
]


def bench_accuracy(mc_paths: list[int], tree_steps: list[int], seeds: int, fast: bool) -> list[dict[str, Any]]:
    rows = []
    bsm = BlackScholesEngine()
    for label, sign, k, days, vol, r, q in _ACCURACY_CASES:
        vd = SAMPLE_VALUATION_DATE
        contract = OptionContract("ACC", k, vd + timedelta(days=days), OptionType.CALL if sign == 1 else OptionType.PUT)
        market = MarketSnapshot("ACC", vd, 100.0, r, q, vol)
        ref = bsm.price(contract, market, greeks=False).price
        kw = dict(min_repeats=3, max_repeats=7 if fast else 11, budget_s=0.3 if fast else 1.0)
        t_bsm = measure(lambda: bsm.price(contract, market, greeks=False), **kw)
        rows.append({"case": label, "method": "analytic", "setting": "-", "median_runtime_s": t_bsm.median, "error": 0.0, "error_kind": "reference"})
        for n in tree_steps:
            eng = BinomialEngine(n)
            t = measure(lambda: eng.price(contract, market, greeks=False), **kw)
            err = abs(eng.price(contract, market, greeks=False).price - ref)
            rows.append({"case": label, "method": "tree", "setting": f"{n} steps", "median_runtime_s": t.median, "error": err, "error_kind": "abs error (deterministic)"})
        for anti in (False, True):
            for n in mc_paths:
                errs = []
                for s in range(seeds):
                    errs.append(MonteCarloEngine(n, seed=s, antithetic=anti).price(contract, market, greeks=False).price - ref)
                eng = MonteCarloEngine(n, seed=0, antithetic=anti)
                t = measure(lambda: eng.price(contract, market, greeks=False), **kw)
                rows.append({"case": label, "method": "mc_antithetic" if anti else "mc_plain", "setting": f"{n:,} paths", "median_runtime_s": t.median,
                             "error": float(np.sqrt(np.mean(np.square(errs)))), "error_kind": f"RMSE over {seeds} seeds"})
    return rows


def accuracy_at_budget(rows: list[dict[str, Any]], budgets: list[float]) -> list[dict[str, Any]]:
    out = []
    for case in dict.fromkeys(r["case"] for r in rows):
        for budget in budgets:
            row: dict[str, Any] = {"case": case, "runtime_budget": human(budget)}
            for method in ("tree", "mc_plain", "mc_antithetic"):
                cands = [r for r in rows if r["case"] == case and r["method"] == method and r["median_runtime_s"] <= budget]
                best = min(cands, key=lambda r: r["error"]) if cands else None
                row[method] = f"{best['error']:.2e} ({best['setting']})" if best else "n/a (too slow)"
            out.append(row)
    return out


# ------------------------------------------------------------------------------ driver
def run_benchmarks(outdir: str | Path, quick: bool = False, progress: Callable[[str], None] | None = None) -> Path:
    say = progress or (lambda s: None)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ncpu = __import__("os").cpu_count() or 1
    workers = [w for w in (1, 2, 4, 8, 10) if w <= max(ncpu, 1)] or [1]

    cfg = {
        "vector_sizes": [1, 10, 100, 1_000, 10_000] if quick else [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000],
        "oo_limit": 1_000 if quick else 10_000,
        "portfolio_sizes": [20, 200, 2_000] if quick else [20, 200, 2_000, 20_000, 200_000],
        "generic_limit": 200 if quick else 2_000,
        "mc_paths": [10_000, 1_000_000] if quick else [10_000, 100_000, 1_000_000, 8_000_000],
        "workers": workers,
        "io_sizes": [1_000, 10_000] if quick else [1_000, 10_000, 100_000],
        "acc_mc_paths": [1_000, 10_000, 100_000] if quick else [1_000, 10_000, 100_000, 1_000_000],
        "acc_tree_steps": [10, 50, 200, 1000] if quick else [10, 25, 50, 100, 250, 500, 1000, 2000],
        "acc_seeds": 20 if quick else 60,
    }
    say("  vectorisation ...")
    b1 = bench_vectorization(cfg["vector_sizes"], cfg["oo_limit"], quick)
    say("  portfolio scenarios ...")
    b2 = bench_portfolio(cfg["portfolio_sizes"], cfg["generic_limit"], quick)
    say("  Monte Carlo workers ...")
    b3 = bench_mc_parallel(cfg["mc_paths"], cfg["workers"], quick)
    b3b = bench_mc_batch([w for w in cfg["workers"] if w in (1, 4, 8)] or [1], 50, 100_000, quick)
    say("  start-up and pools ...")
    b4 = bench_startup(quick)
    say("  I/O vs compute ...")
    b4b = bench_io_vs_compute(cfg["io_sizes"], quick)
    say("  accuracy vs budget ...")
    b5 = bench_accuracy(cfg["acc_mc_paths"], cfg["acc_tree_steps"], cfg["acc_seeds"], quick)

    env = environment()
    write_json(out / "benchmark_results.json", {
        "environment": env, "config": cfg, "quick": quick,
        "vectorisation": _jsonify(b1), "portfolio": _jsonify(b2), "mc_parallel": _jsonify(b3), "mc_batch": _jsonify(b3b),
        "startup": _jsonify(b4), "io_vs_compute": _jsonify(b4b), "accuracy": b5,
    })
    path = out / "benchmark_report.md"
    path.write_text(_render(env, cfg, quick, b1, b2, b3, b3b, b4, b4b, b5), encoding="utf-8")
    return path


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Timing):
        return obj.as_dict()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def _render(env, cfg, quick, b1, b2, b3, b3b, b4, b4b, b5) -> str:  # type: ignore[no-untyped-def]
    L: list[str] = ["# Benchmark report", ""]
    if quick:
        L += ["> **Quick mode**: reduced sizes and repeats. Use the full run for any reported number.", ""]
    L += [
        "## Setup",
        "",
        environment_table(env),
        "",
        "**Method.** Wall-clock `perf_counter`; one warm-up call; each cell repeats until the minimum count and a time budget are met "
        "(the `n` in each cell). Calls under 0.2 ms are batched so a sample lasts about 2 ms. Cells show the **median**; CV is the "
        "coefficient of variation across samples, so a large CV means the median is soft. Nothing else was paused on the machine. "
        "Speedup = ratio of medians. Process pools are created once and reused; their start-up is reported separately. "
        "Inputs come from fixed seeds. Absolute numbers apply to this machine only.",
        "",
        "## 1. Scalar Python vs vectorised NumPy (price + all five Greeks)",
        "",
        "`scalar` is a pure-Python loop over `bsm_scalar` (inputs pre-converted to Python lists, so the loop is not penalised for NumPy scalar access). "
        "`vectorised` is one `bsm_arrays` call. `object interface` loops over `BlackScholesEngine.price(contract, market)` and includes validation and result-object creation "
        "(contract objects built outside the timed region).",
        "",
    ]
    rows = []
    for r in b1:
        rows.append([r["contracts"], _t(r["scalar_python_loop"]), _t(r["numpy_vectorised"]), _x(r["speedup_median"]),
                     _t(r["object_interface_loop"]) if "object_interface_loop" in r else "not run", r["max_abs_diff_price_delta"]])
    L += [md_table(["Contracts", "Scalar Python loop", "NumPy vectorised", "Speedup", "Object interface loop", "Max abs diff (price, delta)"], rows, digits=3), ""]
    slower = [r for r in b1 if r["speedup_median"] < 0.95]
    flat = [r for r in b1 if 0.95 <= r["speedup_median"] < 1.5]
    if slower:
        L += [f"Vectorised code is **slower** than the scalar loop at {', '.join(str(r['contracts']) for r in slower)} contract(s) "
              f"(down to {min(r['speedup_median'] for r in slower):.2f}x): a NumPy call has a fixed overhead of tens of microseconds that a handful of options cannot amortise.", ""]
    if flat:
        L += [f"At {', '.join(str(r['contracts']) for r in flat)} contract(s) the two are within 1.5x of each other.", ""]

    L += ["## 2. Portfolio scenario revaluation (45 scenarios + base Greeks)", "",
          "`vectorised`: `RiskEngine` with the Black-Scholes engine, whole book per scenario in one NumPy call. `position-by-position`: the same engine behind the generic interface "
          "(one `price(contract, market)` call per position per scenario). `book build` is the part of the vectorised run spent flattening the portfolio into arrays.", ""]
    rows = []
    for r in b2:
        rows.append([r["positions"], r["pricings"], _t(r["book_build"]), _t(r["vectorised_total"]),
                     _t(r["position_by_position_total"]) if "position_by_position_total" in r else "not run",
                     _x(r["speedup_median"]) if "speedup_median" in r else "n/a"])
    L += [md_table(["Positions", "Pricings", "Book build", "Vectorised total", "Position-by-position", "Speedup"], rows, digits=3), ""]

    L += ["## 3. Monte Carlo across worker processes", "",
          "One European call, antithetic, price only, 40 independent RNG streams split across the workers. The result is identical for every worker count "
          "(column *same price*), so any timing difference is pure parallel overhead or gain. *First call* includes creating the pool; *compute* is the median of later calls.", ""]
    rows = []
    for r in b3:
        rows.append([r["paths"], r["workers"], _t(r["compute"]), _x(r["speedup_vs_1_worker"]), f"{r['efficiency'] * 100:.0f}%",
                     human(r["first_call_incl_pool_start_s"]), r["price_identical_to_1_worker"]])
    L += [md_table(["Paths", "Workers", "Compute (median)", "Speedup vs 1", "Parallel efficiency", "First call", "Same price"], rows, digits=3), ""]
    L += [f"**Batches of small jobs.** {b3b[0]['contracts']} contracts x {b3b[0]['paths_each']:,} paths each, priced one after another, each call dispatched to the pool:", "",
          md_table(["Workers", "Total (median)", "Speedup vs 1"], [[r["workers"], _t(r["total"]), _x(r["speedup_vs_1_worker"])] for r in b3b], digits=3), ""]

    L += ["## 4. Start-up and file loading, separate from compute", "",
          "**Process and import costs** (fresh subprocess, wall time):", "",
          md_table(["Step", "Median (CV, n)"], [[k.replace("_", " "), _t(v)] for k, v in b4.items() if isinstance(v, Timing)], digits=3, align_right_from=9), "",
          "**Worker pool** creation + one task per worker + shutdown:", "",
          md_table(["Workers", "Median (CV, n)"], [[p["workers"], _t(p["create_run_shutdown"])] for p in b4["pools"]], digits=3), "",
          "**CSV loading versus computing** (vectorised Black-Scholes engine). Loading includes CSV parsing, validation and object construction for contracts, positions and markets:", ""]
    rows = [[r["positions"], f"{r['csv_mb']:.2f}", _t(r["load_csv_and_build_objects"]), _t(r["valuation_with_greeks"]), _t(r["scenario_grid_45"]), _x(r["load_over_scenario"])] for r in b4b]
    L += [md_table(["Positions", "CSV MB", "Load files", "Valuation + Greeks", "45-scenario grid", "Load / scenario grid"], rows, digits=3), ""]

    L += ["## 5. Accuracy at comparable computational budgets", "",
          "Error against the analytic price. Tree error is deterministic. Monte Carlo error is the **RMSE over repeated seeds**, which is the quantity that matters when you cannot know the seed's luck in advance. "
          "Runtime is the median for one pricing call (single process, price only).", ""]
    for case in dict.fromkeys(r["case"] for r in b5):
        sel = [r for r in b5 if r["case"] == case and r["method"] != "analytic"]
        ana = next(r for r in b5 if r["case"] == case and r["method"] == "analytic")
        L += [f"**{case}** (analytic engine: {human(ana['median_runtime_s'])} per call)", "",
              md_table(["Method", "Setting", "Median runtime", "Error", "Error kind"],
                       [[r["method"], r["setting"], human(r["median_runtime_s"]), r["error"], r["error_kind"]] for r in sel], digits=3, align_right_from=9), ""]
    L += ["**Best error within a runtime budget** (best configuration of each method that finishes within the budget):", "",
          dict_table(accuracy_at_budget(b5, [1e-3, 1e-2, 1e-1]), digits=3), ""]
    L += [
        "## Reading these results",
        "",
        "- Treat any speedup with a high CV as approximate. Ratios of medians from fewer than ~5 repeats are weak evidence.",
        "- Process-level parallelism pays a fixed cost per call (serialising arguments, scheduling, returning results). Small jobs can be slower with more workers; this is a finding, not a defect.",
        "- The tree's error is deterministic but not monotone in the step count (odd/even oscillation), so a larger tree is not guaranteed to beat a smaller one at every setting.",
        "- For a single vanilla European option the analytic engine is exact and fastest; tree and Monte Carlo exist to validate it and to extend the framework to payoffs without closed forms.",
        "",
    ]
    return "\n".join(L)

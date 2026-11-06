"""The validation suite itself runs as a test, so a regression anywhere fails CI."""

import pytest

from optrisk.validation import tolerances as T
from optrisk.validation.checks import run_all
from optrisk.validation.grid import validation_grid
from optrisk.validation.quantlib_ref import quantlib_available


def test_tolerance_helper():
    tol = T.Tol(1e-10, 1e-9)
    assert tol.ok(1.0 + 5e-10, 1.0)
    assert not tol.ok(1.0 + 5e-9, 1.0)
    assert tol.ok(1e-11, 0.0)  # absolute term rescues near-zero references


def test_grids():
    assert len(validation_grid(quick=True)) == 3 * 3 * 2 * 2 * 2 * 2
    assert len(validation_grid()) == 7 * 6 * 4 * 4 * 3 * 2


@pytest.fixture(scope="module")
def quick_results():
    results, _ = run_all(quick=True)
    return {r.name: r for r in results}


def test_every_check_passes(quick_results):
    failed = {n: (r.worst_case, r.notes) for n, r in quick_results.items() if r.status == "failed"}
    assert not failed, failed


@pytest.mark.skipif(not quantlib_available(), reason="QuantLib not installed")
def test_quantlib_checks_ran_and_passed(quick_results):
    for name in ("quantlib_bsm", "quantlib_tree", "quantlib_mc", "american_quantlib"):
        assert quick_results[name].status == "passed", name
        assert quick_results[name].n_cases > 0


def test_expected_checks_present(quick_results):
    expected = {"parity_bsm", "bounds_bsm", "greeks_fd", "implied_vol", "tree_vs_analytic", "tree_greeks", "mc_accuracy",
                "mc_repeated_seeds", "mc_variance_reduction", "mc_reproducibility", "invalid_inputs", "expiry_boundaries", "risk_routes"}
    assert expected <= quick_results.keys()

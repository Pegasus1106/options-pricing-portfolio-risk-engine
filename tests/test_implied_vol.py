import math

import numpy as np
import pytest

from optrisk import BlackScholesEngine
from optrisk.engines.black_scholes import bsm_scalar
from optrisk.implied_vol import (
    IVStatus,
    bisect,
    implied_volatility,
    implied_volatility_arrays,
    implied_volatility_raw,
    price_bounds,
)


@pytest.mark.parametrize("method", ["bisection", "brentq"])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_round_trip_price_to_vol_to_price(method, sign, vol, strike):
    args = (sign, 100.0, strike, 0.5, 0.03, 0.01)
    price = bsm_scalar(*args, vol)[0]
    res = implied_volatility_raw(price, *args, method=method)
    assert res.status is IVStatus.CONVERGED
    repriced = bsm_scalar(*args, res.vol)[0]
    assert repriced == pytest.approx(price, abs=1e-9)
    if res.vega > 0.5:  # vol is only well determined when vega is not tiny
        assert res.vol == pytest.approx(vol, abs=1e-8)


def test_bisection_and_brentq_agree_and_brent_is_cheaper():
    args = (1, 100.0, 105.0, 0.75, 0.02, 0.0)
    price = bsm_scalar(*args, 0.31)[0]
    a = implied_volatility_raw(price, *args, method="bisection")
    b = implied_volatility_raw(price, *args, method="brentq")
    assert a.vol == pytest.approx(b.vol, abs=1e-10)
    assert b.function_evals < a.function_evals


def test_own_bisection_matches_known_root():
    root, iters, evals, ok = bisect(lambda x: x**3 - 2.0, 0.0, 2.0, 1e-12, 200)
    assert ok and root == pytest.approx(2 ** (1 / 3), abs=1e-11) and evals == iters


def test_contract_interface_ignores_market_vol(market, make_contract):
    c = make_contract("call", 105.0, 120)
    price = BlackScholesEngine().price(c, market).price
    res = implied_volatility(price, c, market.with_(volatility=0.9))  # wrong vol in snapshot must not matter
    assert res.vol == pytest.approx(market.volatility, abs=1e-9)


def test_status_price_below_lower_bound():
    lo, hi = price_bounds(1, 100.0, 90.0, 0.25, 0.03, 0.01)
    assert implied_volatility_raw(lo * 0.5, 1, 100.0, 90.0, 0.25, 0.03, 0.01).status is IVStatus.PRICE_BELOW_LOWER_BOUND
    assert implied_volatility_raw(lo, 1, 100.0, 90.0, 0.25, 0.03, 0.01).status is IVStatus.PRICE_BELOW_LOWER_BOUND  # bound is open


def test_status_price_above_upper_bound():
    _, hi = price_bounds(-1, 100.0, 110.0, 0.25, 0.03, 0.01)
    assert implied_volatility_raw(hi + 0.1, -1, 100.0, 110.0, 0.25, 0.03, 0.01).status is IVStatus.PRICE_ABOVE_UPPER_BOUND
    assert implied_volatility_raw(hi, -1, 100.0, 110.0, 0.25, 0.03, 0.01).status is IVStatus.PRICE_ABOVE_UPPER_BOUND


def test_status_invalid_inputs():
    base = dict(sign=1, spot=100.0, strike=100.0, tau=0.5, rate=0.03, div=0.0)
    for bad in (dict(price=-1.0), dict(price=float("nan")), dict(price=float("inf")), dict(price=5.0, tau=0.0), dict(price=5.0, tau=-1.0),
                dict(price=5.0, spot=0.0), dict(price=5.0, strike=-1.0), dict(price=5.0, sign=0)):
        kw = {**base, **bad}
        price = kw.pop("price")
        res = implied_volatility_raw(price, kw["sign"], kw["spot"], kw["strike"], kw["tau"], kw["rate"], kw["div"])
        assert res.status is IVStatus.INVALID_INPUT, bad
        assert math.isnan(res.vol)


def test_unknown_method_and_bad_settings_are_invalid_input():
    args = (5.0, 1, 100.0, 100.0, 0.5, 0.03, 0.0)
    assert implied_volatility_raw(*args, method="newton").status is IVStatus.INVALID_INPUT
    assert implied_volatility_raw(*args, vol_lo=0.5, vol_hi=0.1).status is IVStatus.INVALID_INPUT
    assert implied_volatility_raw(*args, xtol=0.0).status is IVStatus.INVALID_INPUT


def test_status_bracket_failed_when_vol_exceeds_solver_maximum():
    res = implied_volatility_raw(100.0 * (1 - 1e-12), 1, 100.0, 100.0, 0.1, 0.0, 0.0, vol_hi=1.0)
    assert res.status is IVStatus.BRACKET_FAILED


def test_status_bracket_failed_when_quote_is_below_price_at_min_vol():
    # The quote is inside the model bounds (a 10%-vol price) but the caller's lower volatility limit is 50%.
    price = bsm_scalar(1, 100.0, 100.0, 0.5, 0.03, 0.0, 0.10)[0]
    res = implied_volatility_raw(price, 1, 100.0, 100.0, 0.5, 0.03, 0.0, vol_lo=0.5)
    assert res.status is IVStatus.BRACKET_FAILED and math.isnan(res.vol)
    assert "below the model price" in res.message


def test_quote_one_ulp_above_lower_bound_is_flagged_ill_conditioned():
    """The price round-trips exactly, but the volatility is meaningless; the result must say so."""
    lo, _ = price_bounds(1, 100.0, 50.0, 0.01, 0.03, 0.0)
    res = implied_volatility_raw(lo * (1 + 1e-15) + 1e-15, 1, 100.0, 50.0, 0.01, 0.03, 0.0)
    assert res.status is IVStatus.CONVERGED
    assert not res.well_conditioned and "poorly determined" in res.message
    ok = implied_volatility_raw(bsm_scalar(1, 100.0, 100.0, 0.5, 0.03, 0.0, 0.2)[0], 1, 100.0, 100.0, 0.5, 0.03, 0.0)
    assert ok.well_conditioned and ok.message == ""


def test_status_max_iterations():
    res = implied_volatility_raw(6.0, 1, 100.0, 100.0, 1.0, 0.03, 0.0, max_iter=3)
    assert res.status is IVStatus.MAX_ITERATIONS and math.isnan(res.vol)


def test_widens_upper_bracket_for_very_high_vol():
    price = bsm_scalar(1, 100.0, 100.0, 1.0, 0.0, 0.0, 7.0)[0]
    res = implied_volatility_raw(price, 1, 100.0, 100.0, 1.0, 0.0, 0.0)  # default hi is 5.0
    assert res.status is IVStatus.CONVERGED and res.vol == pytest.approx(7.0, abs=1e-6)


def test_american_contract_not_supported(market, make_contract):
    res = implied_volatility(5.0, make_contract("put", 100.0, 30, "american"), market)
    assert res.status is IVStatus.INVALID_INPUT


def test_vectorised_solver_matches_scalar_and_flags_failures():
    rng = np.random.default_rng(1)
    n = 300
    sign = rng.choice([-1.0, 1.0], n)
    spot, strike = np.full(n, 100.0), 100.0 * np.exp(rng.uniform(-0.3, 0.3, n))
    tau, r, q, vol = rng.uniform(0.05, 2, n), np.full(n, 0.03), np.full(n, 0.01), rng.uniform(0.1, 0.8, n)
    price = np.array([bsm_scalar(int(sign[i]), 100.0, strike[i], tau[i], 0.03, 0.01, vol[i])[0] for i in range(n)])
    price[0], price[1], price[2] = -1.0, 1e9, np.nan
    v, status = implied_volatility_arrays(price, sign, spot, strike, tau, r, q)
    assert status[0] == 1 and status[1] == 3 and status[2] == 1
    assert np.isnan(v[:3]).all()
    ok = status == 0
    for i in np.flatnonzero(ok)[:50]:
        s = implied_volatility_raw(price[i], int(sign[i]), 100.0, strike[i], tau[i], 0.03, 0.01)
        assert v[i] == pytest.approx(s.vol, abs=1e-10)

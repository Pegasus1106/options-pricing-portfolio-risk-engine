import math
from datetime import timedelta

import numpy as np
import pytest

from optrisk import BlackScholesEngine, InvalidInputError, MarketSnapshot, OptionContract, UnsupportedInstrumentError
from optrisk.conventions import year_fraction
from optrisk.engines.black_scholes import bsm_arrays, bsm_scalar

from conftest import VAL_DATE


def test_textbook_values():
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5 -> c=4.76, p=0.81
    call, _ = bsm_scalar(1, 42.0, 40.0, 0.5, 0.10, 0.0, 0.20)
    put, _ = bsm_scalar(-1, 42.0, 40.0, 0.5, 0.10, 0.0, 0.20)
    assert call == pytest.approx(4.7594, abs=5e-5)
    assert put == pytest.approx(0.8086, abs=5e-5)
    # S=K=100, r=5%, sigma=20%, T=1
    assert bsm_scalar(1, 100.0, 100.0, 1.0, 0.05, 0.0, 0.2)[0] == pytest.approx(10.450584, abs=1e-6)
    assert bsm_scalar(-1, 100.0, 100.0, 1.0, 0.05, 0.0, 0.2)[0] == pytest.approx(5.573526, abs=1e-6)


def test_time_convention_is_act_365():
    assert year_fraction(VAL_DATE, VAL_DATE + timedelta(days=365)) == 1.0
    assert year_fraction(VAL_DATE, VAL_DATE + timedelta(days=73)) == pytest.approx(0.2)
    # a leap year is not 366/365 years
    from datetime import date

    assert year_fraction(date(2024, 1, 1), date(2025, 1, 1)) == pytest.approx(366 / 365)


@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(market, make_contract, q):
    m = market.with_(dividend_yield=q)
    eng = BlackScholesEngine()
    c = eng.price(make_contract("call", 95.0), m).price
    p = eng.price(make_contract("put", 95.0), m).price
    tau = 90 / 365
    assert c - p == pytest.approx(m.spot * math.exp(-q * tau) - 95.0 * math.exp(-m.rate * tau), abs=1e-12)


def test_units_vega_per_point_theta_per_day_rho_per_point(market, make_contract):
    """Greek units must match their documented definitions, checked by bumping the inputs."""
    c = make_contract("call", 105.0, 120)
    eng = BlackScholesEngine()
    base = eng.price(c, market)
    vega_bump = eng.price(c, market.with_(volatility=market.volatility + 0.01)).price - base.price
    assert base.greeks.vega == pytest.approx(vega_bump, rel=2e-2)  # one vol point
    rho_bump = eng.price(c, market.with_(rate=market.rate + 0.01)).price - base.price
    assert base.greeks.rho == pytest.approx(rho_bump, rel=2e-2)  # one rate point
    one_day = eng.price(c, market.with_(valuation_date=VAL_DATE + timedelta(days=1))).price - base.price
    assert base.greeks.theta == pytest.approx(one_day, rel=2e-2)  # one calendar day
    assert base.greeks.theta < 0


def test_analytic_greeks_match_central_differences(market, make_contract):
    eng = BlackScholesEngine()
    for kind in ("call", "put"):
        c = make_contract(kind, 98.0, 200)
        g = eng.price(c, market).greeks
        h = 0.01
        up = eng.price(c, market.with_(spot=market.spot + h)).price
        dn = eng.price(c, market.with_(spot=market.spot - h)).price
        mid = eng.price(c, market).price
        assert g.delta == pytest.approx((up - dn) / (2 * h), abs=1e-7)
        assert g.gamma == pytest.approx((up - 2 * mid + dn) / h**2, abs=1e-5)


def test_scalar_and_vectorised_agree():
    rng = np.random.default_rng(3)
    n = 500
    sign = rng.choice([-1.0, 1.0], n)
    spot, strike = rng.uniform(50, 150, n), rng.uniform(50, 150, n)
    tau, r, q, v = rng.uniform(0.01, 3, n), rng.uniform(-0.01, 0.06, n), rng.uniform(0, 0.05, n), rng.uniform(0.05, 0.9, n)
    arr = bsm_arrays(sign, spot, strike, tau, r, q, v)
    for i in range(n):
        price, g = bsm_scalar(int(sign[i]), spot[i], strike[i], tau[i], r[i], q[i], v[i])
        assert arr["price"][i] == pytest.approx(price, rel=1e-12, abs=1e-13)
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            assert arr[name][i] == pytest.approx(getattr(g, name), rel=1e-11, abs=1e-12)


def test_price_inside_no_arbitrage_bounds(market, make_contract):
    eng = BlackScholesEngine()
    for k in (60.0, 100.0, 150.0):
        for kind in ("call", "put"):
            p = eng.price(make_contract(kind, k, 30), market).price
            assert 0.0 <= p
            fwd = market.spot * math.exp(-market.dividend_yield * 30 / 365)
            disc_k = k * math.exp(-market.rate * 30 / 365)
            if kind == "call":
                assert max(fwd - disc_k, 0.0) - 1e-12 <= p <= fwd + 1e-12
            else:
                assert max(disc_k - fwd, 0.0) - 1e-12 <= p <= disc_k + 1e-12


def test_expired_option_returns_intrinsic_value(market, make_contract):
    eng = BlackScholesEngine()
    itm = eng.price(make_contract("call", 90.0, 0), market)
    assert itm.price == 10.0 and itm.greeks.delta == 1.0
    assert (itm.greeks.gamma, itm.greeks.vega, itm.greeks.theta, itm.greeks.rho) == (0.0, 0.0, 0.0, 0.0)
    assert itm.diagnostics["expired"] is True
    otm = eng.price(make_contract("put", 90.0, 0), market)
    assert otm.price == 0.0 and otm.greeks.delta == 0.0
    atm = eng.price(make_contract("call", 100.0, 0), market)
    assert atm.price == 0.0 and atm.greeks.delta == 0.0  # delta is the step function, 0 at the strike
    itm_put = eng.price(make_contract("put", 110.0, 0), market)
    assert itm_put.price == 10.0 and itm_put.greeks.delta == -1.0


def test_vectorised_handles_expired_lanes_without_nan():
    out = bsm_arrays(np.array([1.0, 1.0, -1.0]), 100.0, np.array([90.0, 110.0, 110.0]), np.array([0.0, 0.5, 0.0]), 0.03, 0.0, 0.2)
    assert np.all(np.isfinite(list(out.values())))
    assert out["price"][0] == 10.0 and out["price"][2] == 10.0


def test_deep_in_and_out_of_the_money_stay_finite(market, make_contract):
    eng = BlackScholesEngine()
    for k in (1.0, 1e5):
        r = eng.price(make_contract("call", k, 7), market.with_(volatility=0.05))
        assert math.isfinite(r.price) and all(math.isfinite(v) for v in r.greeks.as_dict().values())


def test_rejects_valuation_after_expiry(market, make_contract):
    with pytest.raises(InvalidInputError, match="before valuation date"):
        BlackScholesEngine().price(make_contract("call", 100.0, -1), market)


def test_rejects_underlying_mismatch(market):
    other = OptionContract("Y", 100.0, VAL_DATE + timedelta(days=30), "call")
    with pytest.raises(InvalidInputError, match="does not match"):
        BlackScholesEngine().price(other, market)


def test_rejects_american_exercise(market, make_contract):
    with pytest.raises(UnsupportedInstrumentError):
        BlackScholesEngine().price(make_contract("put", 100.0, 30, "american"), market)


@pytest.mark.parametrize(
    "kwargs",
    [dict(spot=0.0), dict(spot=-1.0), dict(volatility=0.0), dict(volatility=-0.1), dict(volatility=25.0), dict(rate=5.0),
     dict(rate=float("nan")), dict(dividend_yield=float("inf")), dict(spot="abc")],
)
def test_invalid_market_inputs_rejected(kwargs):
    base = dict(underlying="X", valuation_date=VAL_DATE, spot=100.0, rate=0.03, dividend_yield=0.0, volatility=0.2)
    base.update(kwargs)
    with pytest.raises(InvalidInputError):
        MarketSnapshot(**base)


@pytest.mark.parametrize("strike", [0.0, -1.0, float("nan"), float("inf"), "x"])
def test_invalid_strikes_rejected(strike):
    with pytest.raises(InvalidInputError):
        OptionContract("X", strike, VAL_DATE, "call")


def test_invalid_option_type_and_date_type_rejected():
    from datetime import datetime

    with pytest.raises(InvalidInputError):
        OptionContract("X", 100.0, VAL_DATE, "straddle")
    with pytest.raises(InvalidInputError):
        OptionContract("X", 100.0, datetime(2026, 6, 1), "call")

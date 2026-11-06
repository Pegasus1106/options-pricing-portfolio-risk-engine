import math

import pytest

from optrisk import BinomialEngine, BlackScholesEngine, InvalidInputError, NumericalError, MarketSnapshot

from conftest import VAL_DATE


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_tree_converges_to_bsm(market, make_contract, kind):
    c = make_contract(kind, 105.0, 180)
    exact = BlackScholesEngine().price(c, market).price
    errors = [abs(BinomialEngine(n).price(c, market, greeks=False).price - exact) for n in (50, 200, 800)]
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.02 * 1.0  # well inside 0.3 * S*sigma*sqrt(T)/n scale


def test_tree_error_follows_o_1_over_n(market, make_contract):
    c = make_contract("call", 100.0, 365)
    exact = BlackScholesEngine().price(c, market).price
    scale = market.spot * market.volatility * math.sqrt(1.0)
    for n in (100, 400, 1600):
        err = abs(BinomialEngine(n).price(c, market, greeks=False).price - exact)
        assert err <= 0.3 * scale / n


def test_tree_greeks_match_analytic(market, make_contract):
    c = make_contract("call", 100.0, 120)
    t = BinomialEngine(800).price(c, market).greeks
    a = BlackScholesEngine().price(c, market).greeks
    assert t.delta == pytest.approx(a.delta, abs=1e-3)
    assert t.gamma == pytest.approx(a.gamma, abs=1e-3)
    assert t.vega == pytest.approx(a.vega, abs=3e-3)
    assert t.theta == pytest.approx(a.theta, abs=1e-3)
    assert t.rho == pytest.approx(a.rho, abs=1e-3)


def test_tree_satisfies_put_call_parity_exactly(market, make_contract):
    eng = BinomialEngine(150)
    c = eng.price(make_contract("call", 97.0, 200), market, greeks=False).price
    p = eng.price(make_contract("put", 97.0, 200), market, greeks=False).price
    tau = 200 / 365
    assert c - p == pytest.approx(market.spot * math.exp(-market.dividend_yield * tau) - 97.0 * math.exp(-market.rate * tau), abs=1e-10)


def test_american_put_at_least_european_and_intrinsic(market, make_contract):
    eng = BinomialEngine(400)
    m = market.with_(rate=0.06, dividend_yield=0.0)
    eu = eng.price(make_contract("put", 110.0, 365), m, greeks=False).price
    am = eng.price(make_contract("put", 110.0, 365, "american"), m, greeks=False).price
    assert am > eu + 1e-3  # early exercise is valuable for a deep ITM put with r > 0
    assert am >= 10.0


def test_american_call_without_dividends_equals_european(market, make_contract):
    m = market.with_(dividend_yield=0.0)
    eng = BinomialEngine(300)
    eu = eng.price(make_contract("call", 95.0, 200), m, greeks=False).price
    am = eng.price(make_contract("call", 95.0, 200, "american"), m, greeks=False).price
    assert am == pytest.approx(eu, abs=1e-10)


def test_american_call_with_large_dividend_exceeds_european(market, make_contract):
    m = market.with_(dividend_yield=0.10)
    eng = BinomialEngine(400)
    eu = eng.price(make_contract("call", 80.0, 365), m, greeks=False).price
    am = eng.price(make_contract("call", 80.0, 365, "american"), m, greeks=False).price
    assert am > eu + 1e-3


def test_american_reference_value():
    """Independent reference: QuantLib CRR tree n=1000 gave 9.356822 for this put (S=100,K=105,r=4%,q=2%,vol=25%,180d)."""
    m = MarketSnapshot("X", VAL_DATE, 100.0, 0.04, 0.02, 0.25)
    from datetime import timedelta

    from optrisk import OptionContract

    c = OptionContract("X", 105.0, VAL_DATE + timedelta(days=180), "put", "american")
    assert BinomialEngine(1000).price(c, m, greeks=False).price == pytest.approx(9.356822, abs=1e-3)


def test_invalid_step_counts():
    for bad in (2, 0, -5, 100.5, True, 10**7):
        with pytest.raises(InvalidInputError):
            BinomialEngine(bad)


def test_probability_outside_unit_interval_raises(make_contract):
    m = MarketSnapshot("X", VAL_DATE, 100.0, 0.50, 0.0, 0.01)
    with pytest.raises(NumericalError, match="probability"):
        BinomialEngine(5).price(make_contract("call", 100.0, 730), m)

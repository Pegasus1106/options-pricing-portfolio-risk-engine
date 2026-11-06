from datetime import timedelta

import numpy as np
import pytest

from optrisk import BinomialEngine, BlackScholesEngine, InvalidInputError, MarketSnapshot, OptionContract
from optrisk.engines.base import PricingEngine
from optrisk.generate import random_portfolio, sample_markets, sample_portfolio
from optrisk.portfolio import Portfolio, Position
from optrisk.risk import RiskEngine, Scenario, default_scenarios

from conftest import VAL_DATE


class GenericBSM(PricingEngine):
    """Hides price_arrays so the RiskEngine takes its position-by-position route."""

    name = "generic_bsm"

    def __init__(self):
        self._inner = BlackScholesEngine()

    def _price(self, contract, market, tau, greeks):
        return self._inner._price(contract, market, tau, greeks)


@pytest.fixture(scope="module")
def sample():
    return sample_portfolio(), sample_markets()


def test_total_value_is_sum_of_position_values(sample):
    pf, mk = sample
    v = RiskEngine().value(pf, mk)
    eng = BlackScholesEngine()
    expected = sum(eng.price(p.contract, mk[p.contract.underlying]).price * p.quantity * p.multiplier for p in pf)
    assert v.total_value == pytest.approx(expected, rel=1e-12)
    assert len(v.positions) == len(pf)


def test_aggregate_greeks_are_position_weighted_sums(sample):
    pf, mk = sample
    v = RiskEngine().value(pf, mk)
    eng = BlackScholesEngine()
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        expected = sum(getattr(eng.price(p.contract, mk[p.contract.underlying]).greeks, name) * p.quantity * p.multiplier for p in pf)
        assert v.totals[name] == pytest.approx(expected, rel=1e-11, abs=1e-9)


def test_exposure_groups_sum_to_totals(sample):
    pf, mk = sample
    v = RiskEngine().value(pf, mk)
    for groups in (v.by_underlying, v.by_expiry, v.by_underlying_expiry):
        assert sum(g.value for g in groups) == pytest.approx(v.total_value)
        assert sum(g.n_positions for g in groups) == len(pf)
        assert sum(g.vega for g in groups) == pytest.approx(v.totals["vega"])
    assert {g.underlying for g in v.by_underlying} == {"SIM_A", "SIM_B", "SIM_C"}
    assert len({(g.underlying, g.expiry) for g in v.by_underlying_expiry}) == len(v.by_underlying_expiry)


def test_short_position_has_negative_value_and_opposite_greeks(market, make_contract):
    c = make_contract("call", 100.0, 60)
    pf = Portfolio((Position(c, 10, 100), Position(c, -10, 100)))
    v = RiskEngine().value(pf, {"X": market})
    assert v.total_value == pytest.approx(0.0, abs=1e-9)
    assert all(abs(v.totals[k]) < 1e-9 for k in ("delta", "gamma", "vega", "theta", "rho"))


def test_multiplier_scales_everything(market, make_contract):
    c = make_contract("put", 100.0, 60)
    one = RiskEngine().value(Portfolio((Position(c, 1, 1),)), {"X": market})
    hundred = RiskEngine().value(Portfolio((Position(c, 1, 100),)), {"X": market})
    assert hundred.total_value == pytest.approx(100 * one.total_value)
    assert hundred.totals["delta"] == pytest.approx(100 * one.totals["delta"])


def test_zero_shock_scenario_has_zero_pnl(sample):
    pf, mk = sample
    r = RiskEngine().run_scenarios(pf, mk, [Scenario()])[0]
    assert r.full_pnl == pytest.approx(0.0, abs=1e-9) and r.approx_pnl == 0.0


def test_scenario_full_revaluation_equals_manual_repricing(sample):
    pf, mk = sample
    sc = Scenario(-0.05, 0.05, 7)
    r = RiskEngine().run_scenarios(pf, mk, [sc])[0]
    eng = BlackScholesEngine()
    total = 0.0
    for p in pf:
        m = mk[p.contract.underlying]
        base = eng.price(p.contract, m, greeks=False).price
        shocked = m.shocked(sc.spot_pct, sc.vol_abs, 0)
        when = min(m.valuation_date + timedelta(days=sc.days), p.contract.expiry)
        new = eng.price(p.contract, shocked.with_(valuation_date=when), greeks=False).price
        total += (new - base) * p.scale
    assert r.full_pnl == pytest.approx(total, rel=1e-10)


def test_vectorised_and_generic_routes_agree_including_expiry_in_horizon():
    rand_pf, rand_mk = random_portfolio(250, n_underlyings=5, seed=8)
    sample_pf, sample_mk = sample_portfolio(), sample_markets()  # contains 4-day options that expire inside a 7-day horizon
    pf = Portfolio(rand_pf.positions + sample_pf.positions)
    mk = {**rand_mk, **sample_mk}
    scs = default_scenarios()
    fast = RiskEngine(BlackScholesEngine()).run_scenarios(pf, mk, scs)
    slow = RiskEngine(GenericBSM()).run_scenarios(pf, mk, scs)
    assert any(r.n_expired > 0 for r in fast)  # the random book has options expiring inside 7 days
    for a, b in zip(fast, slow):
        assert a.full_pnl == pytest.approx(b.full_pnl, rel=1e-9, abs=1e-6)
        assert a.approx_pnl == pytest.approx(b.approx_pnl, rel=1e-9, abs=1e-6)
        assert a.n_expired == b.n_expired


def test_greek_approximation_error_is_third_order_in_spot_shock(market, make_contract):
    pf = Portfolio((Position(make_contract("call", 100.0, 365), 1, 1),))
    errs = []
    for s in (0.02, 0.04, 0.08):
        r = RiskEngine().run_scenarios(pf, {"X": market}, [Scenario(s, 0.0, 0)])[0]
        errs.append(abs(r.approx_error))
    assert errs[1] / errs[0] == pytest.approx(8.0, rel=0.35)
    assert errs[2] / errs[1] == pytest.approx(8.0, rel=0.35)


def test_greek_approximation_is_worse_near_expiry(market, make_contract):
    def rel_error(days: int) -> float:
        pf = Portfolio((Position(make_contract("call", 100.0, days), 1, 1),))
        r = RiskEngine().run_scenarios(pf, {"X": market}, [Scenario(0.10, 0.0, 0)])[0]
        return abs(r.approx_error) / abs(r.full_pnl)

    assert rel_error(4) > 3 * rel_error(90) > 0


def test_expired_in_horizon_values_at_intrinsic(market, make_contract):
    c = make_contract("call", 90.0, 3)
    pf = Portfolio((Position(c, 1, 1),))
    r = RiskEngine().run_scenarios(pf, {"X": market}, [Scenario(0.0, 0.0, 7)])[0]
    assert r.n_expired == 1
    base = BlackScholesEngine().price(c, market).price
    assert r.shocked_value == pytest.approx(10.0)  # intrinsic S - K
    assert r.full_pnl == pytest.approx(10.0 - base)


def test_large_negative_vol_shock_is_floored_not_rejected(market, make_contract):
    pf = Portfolio((Position(make_contract("call", 100.0, 60), 1, 1),))
    r = RiskEngine().run_scenarios(pf, {"X": market}, [Scenario(0.0, -0.50, 0)])[0]
    assert np.isfinite(r.full_pnl)


def test_works_with_other_engines(sample):
    pf, mk = sample
    sub = Portfolio(pf.positions[:4])
    a = RiskEngine(BlackScholesEngine()).value(sub, mk)
    b = RiskEngine(BinomialEngine(400)).value(sub, mk)
    assert b.total_value == pytest.approx(a.total_value, rel=5e-3)
    assert b.totals["delta"] == pytest.approx(a.totals["delta"], rel=5e-3)
    r = RiskEngine(BinomialEngine(100)).run_scenarios(sub, mk, [Scenario(-0.05, 0.0, 1)])[0]
    ref = RiskEngine().run_scenarios(sub, mk, [Scenario(-0.05, 0.0, 1)])[0]
    assert r.full_pnl == pytest.approx(ref.full_pnl, rel=0.05)


def test_missing_market_and_empty_portfolio_rejected(sample, market, make_contract):
    pf, mk = sample
    with pytest.raises(InvalidInputError, match="no market snapshot"):
        RiskEngine().value(pf, {"SIM_A": mk["SIM_A"]})
    with pytest.raises(InvalidInputError, match="no positions"):
        RiskEngine().value(Portfolio(()), mk)


def test_expired_contract_before_valuation_date_rejected(market, make_contract):
    pf = Portfolio((Position(make_contract("call", 100.0, -2), 1, 1),))
    with pytest.raises(InvalidInputError, match="before the valuation date"):
        RiskEngine().value(pf, {"X": market})


def test_position_validation():
    c = OptionContract("X", 100.0, VAL_DATE + timedelta(days=30), "call")
    for bad in (dict(quantity=float("nan")), dict(quantity="a"), dict(quantity=1, multiplier=0), dict(quantity=1, multiplier=-100)):
        with pytest.raises(InvalidInputError):
            Position(c, **bad)


def test_scenario_validation():
    with pytest.raises(InvalidInputError):
        Scenario(days=-1)
    with pytest.raises(InvalidInputError):
        Scenario(spot_pct=-1.0)
    with pytest.raises(InvalidInputError):
        MarketSnapshot("X", VAL_DATE, 100, 0.03, 0.0, 0.2).shocked(spot_pct=-1.5)

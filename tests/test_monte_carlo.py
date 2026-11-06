import numpy as np
import pytest

from optrisk import BlackScholesEngine, InvalidInputError, MonteCarloEngine, UnsupportedInstrumentError
from optrisk.engines.monte_carlo import split_counts


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_and_greeks_within_four_standard_errors(market, make_contract, kind):
    c = make_contract(kind, 102.0, 150)
    exact = BlackScholesEngine().price(c, market)
    res = MonteCarloEngine(400_000, seed=11).price(c, market)
    assert abs(res.price - exact.price) <= 4 * res.std_error
    se = res.diagnostics["greeks_std_error"]
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert abs(getattr(res.greeks, name) - getattr(exact.greeks, name)) <= 4 * se[name], name


def test_confidence_interval_is_centered_and_scaled(market, make_contract):
    res = MonteCarloEngine(100_000, seed=1, confidence=0.95).price(make_contract(), market)
    lo, hi = res.confidence_interval
    assert (lo + hi) / 2 == pytest.approx(res.price)
    assert (hi - lo) / 2 == pytest.approx(1.959964 * res.std_error, rel=1e-5)
    wide = MonteCarloEngine(100_000, seed=1, confidence=0.99).price(make_contract(), market).confidence_interval
    assert wide[1] - wide[0] > hi - lo


def test_same_seed_reproduces_and_different_seed_differs(market, make_contract):
    c = make_contract()
    a = MonteCarloEngine(20_000, seed=5).price(c, market)
    b = MonteCarloEngine(20_000, seed=5).price(c, market)
    d = MonteCarloEngine(20_000, seed=6).price(c, market)
    assert a.price == b.price and a.greeks == b.greeks
    assert a.price != d.price


def test_result_is_independent_of_worker_count(market, make_contract):
    c = make_contract("put", 100.0, 90)
    ref = MonteCarloEngine(40_000, seed=3, n_streams=8, n_workers=1).price(c, market)
    for w in (2, 3):
        with MonteCarloEngine(40_000, seed=3, n_streams=8, n_workers=w) as eng:
            res = eng.price(c, market)
        assert res.price == ref.price
        assert res.greeks == ref.greeks
        assert res.std_error == ref.std_error


def test_spawned_streams_are_distinct():
    a, b = (np.random.default_rng(s).standard_normal(100_000) for s in np.random.SeedSequence(9).spawn(2))
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.01
    assert not np.array_equal(a, b)


def test_stream_count_changes_stream_layout_not_validity(market, make_contract):
    c = make_contract()
    exact = BlackScholesEngine().price(c, market).price
    for streams in (1, 4, 16):
        res = MonteCarloEngine(200_000, seed=2, n_streams=streams).price(c, market, greeks=False)
        assert abs(res.price - exact) <= 4.5 * res.std_error


def test_antithetic_reduces_estimator_variance(market, make_contract):
    c = make_contract("call", 100.0, 90)
    plain = [MonteCarloEngine(20_000, seed=s, antithetic=False).price(c, market, greeks=False).price for s in range(150)]
    anti = [MonteCarloEngine(20_000, seed=s, antithetic=True).price(c, market, greeks=False).price for s in range(150)]
    assert np.var(anti, ddof=1) < 0.7 * np.var(plain, ddof=1)


def test_reported_standard_error_matches_spread_across_seeds(market, make_contract):
    # 1500 seeds: the empirical std of the estimates then has ~3% sampling noise, so a +-10% band is a ~3 sigma test.
    # (With 200 seeds the noise is ~8% and a +-15% band fails by chance; measured: ratio 0.998 / 1.000 at 3000 seeds.)
    c = make_contract("call", 100.0, 180)
    for anti in (False, True):
        runs = [MonteCarloEngine(5_000, seed=s, antithetic=anti).price(c, market, greeks=False) for s in range(1500)]
        ratio = np.mean([r.std_error for r in runs]) / np.std([r.price for r in runs], ddof=1)
        assert 0.9 < ratio < 1.1, (anti, ratio)


def test_antithetic_se_uses_pairs_not_individual_paths(market, make_contract):
    """Treating the two halves of an antithetic pair as independent would understate the error."""
    c = make_contract("call", 100.0, 90)
    runs = [MonteCarloEngine(10_000, seed=s, antithetic=True).price(c, market, greeks=False) for s in range(300)]
    empirical = np.std([r.price for r in runs], ddof=1)
    assert np.mean([r.std_error for r in runs]) == pytest.approx(empirical, rel=0.15)


def test_expired_option_has_zero_error(market, make_contract):
    res = MonteCarloEngine(1000, seed=1).price(make_contract("call", 90.0, 0), market)
    assert res.price == 10.0 and res.std_error == 0.0 and res.confidence_interval == (10.0, 10.0)


def test_split_counts():
    assert split_counts(10, 3) == [4, 3, 3]
    assert sum(split_counts(1001, 40)) == 1001


def test_more_streams_than_samples(market, make_contract):
    res = MonteCarloEngine(6, seed=1, antithetic=True, n_streams=8).price(make_contract(), market, greeks=False)
    assert res.diagnostics["n_samples"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [dict(n_paths=1001, antithetic=True), dict(n_paths=1), dict(n_paths=0), dict(n_paths=100.5), dict(n_paths=1000, seed=-1),
     dict(n_paths=1000, confidence=1.5), dict(n_paths=1000, n_workers=0), dict(n_paths=1000, n_streams=0)],
)
def test_invalid_settings(kwargs):
    with pytest.raises(InvalidInputError):
        MonteCarloEngine(**kwargs)


def test_rejects_american(market, make_contract):
    with pytest.raises(UnsupportedInstrumentError):
        MonteCarloEngine(1000).price(make_contract("put", 100.0, 30, "american"), market)

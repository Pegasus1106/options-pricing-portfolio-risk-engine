"""QuantLib reference prices, built with conventions that match ``optrisk.conventions``.

Matching conventions (the comparison is meaningless without them):

* time:        Actual365Fixed day counter, NullCalendar (no holidays), valuation date set as the
               QuantLib evaluation date, expiry date as the exercise date;
* rates:       flat, continuously compounded, Actual365Fixed (``FlatForward``) for both the
               risk-free rate and the dividend yield;
* volatility:  ``BlackConstantVol`` (flat, Actual365Fixed);
* units:       QuantLib reports vega per 1.0 vol, rho per 1.0 rate, theta per year. They are
               converted to optrisk's per-vol-point, per-rate-point and per-calendar-day units
               here and nowhere else. Theta is ``theta() / 365`` (calendar days, matching the
               Actual365Fixed convention) rather than a QuantLib per-day accessor, so the
               conversion is explicit.

QuantLib is an optional dependency: import this module only inside functions that need it
via :func:`quantlib_available`.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ..contracts import ExerciseStyle, OptionContract
from ..conventions import DAYS_PER_YEAR, RHO_UNIT, VEGA_UNIT
from ..market import MarketSnapshot

try:  # pragma: no cover - exercised implicitly
    import QuantLib as ql
except ImportError:  # pragma: no cover
    ql = None


def quantlib_available() -> bool:
    return ql is not None


def quantlib_version() -> str:
    return ql.__version__ if ql is not None else "not installed"


def _qdate(d: date) -> "ql.Date":
    return ql.Date(d.day, d.month, d.year)


class QuantLibReference:
    """Builds QuantLib options from optrisk objects."""

    def __init__(self) -> None:
        if ql is None:
            raise RuntimeError("QuantLib is not installed; `pip install QuantLib` to run the reference comparison")
        self._calendar = ql.NullCalendar()
        self._dc = ql.Actual365Fixed()

    def _process(self, market: MarketSnapshot):  # type: ignore[no-untyped-def]
        today = _qdate(market.valuation_date)
        ql.Settings.instance().evaluationDate = today
        rate = ql.YieldTermStructureHandle(ql.FlatForward(today, market.rate, self._dc, ql.Continuous))
        div = ql.YieldTermStructureHandle(ql.FlatForward(today, market.dividend_yield, self._dc, ql.Continuous))
        vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, self._calendar, market.volatility, self._dc))
        spot = ql.QuoteHandle(ql.SimpleQuote(market.spot))
        return ql.BlackScholesMertonProcess(spot, div, rate, vol), today

    def _option(self, contract: OptionContract, today: "ql.Date"):  # type: ignore[no-untyped-def]
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if contract.is_call else ql.Option.Put, contract.strike)
        expiry = _qdate(contract.expiry)
        if contract.exercise is ExerciseStyle.AMERICAN:
            exercise = ql.AmericanExercise(today, expiry)
        else:
            exercise = ql.EuropeanExercise(expiry)
        return ql.VanillaOption(payoff, exercise)

    def european_analytic(self, contract: OptionContract, market: MarketSnapshot) -> dict[str, float]:
        """AnalyticEuropeanEngine price and Greeks in optrisk units."""
        process, today = self._process(market)
        option = self._option(contract, today)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return {
            "price": option.NPV(),
            "delta": option.delta(),
            "gamma": option.gamma(),
            "vega": option.vega() * VEGA_UNIT,
            "theta": option.theta() / DAYS_PER_YEAR,
            "rho": option.rho() * RHO_UNIT,
        }

    def tree_price(self, contract: OptionContract, market: MarketSnapshot, steps: int, kind: str = "crr") -> float:
        process, today = self._process(market)
        option = self._option(contract, today)
        option.setPricingEngine(ql.BinomialVanillaEngine(process, kind, steps))
        return option.NPV()

    def finite_difference_price(self, contract: OptionContract, market: MarketSnapshot, t_steps: int = 400, x_steps: int = 800) -> float:
        process, today = self._process(market)
        option = self._option(contract, today)
        option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, t_steps, x_steps))
        return option.NPV()

    def monte_carlo_price(self, contract: OptionContract, market: MarketSnapshot, samples: int, seed: int = 1) -> tuple[float, float]:
        """QuantLib's own MC engine (price, error estimate), for an apples-to-apples noise comparison."""
        process, today = self._process(market)
        option = self._option(contract, today)
        option.setPricingEngine(
            ql.MCEuropeanEngine(process, "PseudoRandom", timeSteps=1, antitheticVariate=True, requiredSamples=samples, seed=seed)
        )
        return option.NPV(), option.errorEstimate()


def versions() -> dict[str, Any]:
    return {"quantlib": quantlib_version()}

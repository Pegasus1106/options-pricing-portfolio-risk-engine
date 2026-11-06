"""Pricing engine interface.

Engines are small objects holding only numerical settings (steps, paths, seed...).
Contract and market data are passed per call: ``engine.price(contract, market)``.
The base class owns the behaviour that must be identical across engines:

* the contract's underlying must match the market's underlying;
* unsupported instruments are rejected explicitly;
* ``valuation_date > expiry``  -> :class:`InvalidInputError` (almost always a data error);
* ``valuation_date == expiry`` -> expired: price is intrinsic value, delta is the
  payoff step function (+-1 if strictly in the money, else 0), all other Greeks 0;
* wall-clock runtime is recorded on the result.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace

from ..contracts import ExerciseStyle, OptionContract
from ..conventions import year_fraction
from ..errors import InvalidInputError, UnsupportedInstrumentError
from ..market import MarketSnapshot
from ..results import Greeks, PricingResult


def expired_price_and_greeks(sign: int, spot: float, strike: float) -> tuple[float, Greeks]:
    """Value of an option at T = 0. Delta is the payoff step; the other Greeks are 0."""
    intrinsic = max(sign * (spot - strike), 0.0)
    delta = float(sign) if intrinsic > 0.0 else 0.0
    return intrinsic, Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)


class PricingEngine(ABC):
    #: Short method name stored on every result.
    name: str = "engine"
    #: Whether :meth:`_price` can handle American exercise.
    supports_american: bool = False
    #: Whether results carry sampling error (std_error / confidence_interval).
    statistical: bool = False

    def price(self, contract: OptionContract, market: MarketSnapshot, *, greeks: bool = True) -> PricingResult:
        """Price one contract. Result price is per one unit of the underlying."""
        market.check_contract(contract)
        if contract.exercise is ExerciseStyle.AMERICAN and not self.supports_american:
            raise UnsupportedInstrumentError(f"{self.name} engine cannot price American options")
        tau = year_fraction(market.valuation_date, contract.expiry)
        if tau < 0.0:
            raise InvalidInputError(
                f"{contract.label}: expiry {contract.expiry} is before valuation date {market.valuation_date}"
            )
        start = time.perf_counter()
        if tau == 0.0:
            result = self._expired(contract, market, greeks)
        else:
            result = self._price(contract, market, tau, greeks)
        return replace(result, runtime_s=time.perf_counter() - start)

    def _expired(self, contract: OptionContract, market: MarketSnapshot, greeks: bool) -> PricingResult:
        price, g = expired_price_and_greeks(contract.option_type.sign, market.spot, contract.strike)
        return PricingResult(
            price=price,
            method=self.name,
            greeks=g if greeks else None,
            std_error=0.0 if self.statistical else None,
            confidence_interval=(price, price) if self.statistical else None,
            diagnostics={"expired": True},
        )

    @abstractmethod
    def _price(self, contract: OptionContract, market: MarketSnapshot, tau: float, greeks: bool) -> PricingResult:
        """Price with ``tau > 0`` years to expiry. Inputs are already validated."""

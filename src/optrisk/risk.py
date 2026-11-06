"""Portfolio valuation, aggregate Greeks, exposures, and scenario revaluation.

Aggregation units
-----------------
Per-position figures are ``unit value * quantity * multiplier``. So for a portfolio:

    value        currency
    delta        units of underlying (share-equivalents)
    gamma        change in share-equivalent delta per 1.0 move in spot
    vega         currency per +1 vol point
    theta        currency per calendar day
    rho          currency per +1 rate point
    delta_notional  delta * spot, currency exposure to the underlying

Scenarios
---------
A scenario is a joint (relative spot shock, absolute vol shock, calendar days forward). Full
revaluation re-prices every position in the shocked market. The Greek approximation is the
second-order Taylor expansion actually used in practice:

    dV ~ delta*dS + 0.5*gamma*dS^2 + vega*d(vol points) + theta*days

It leaves out cross terms (vanna, volga/vomma, charm), higher orders in dS, and rho (rates do
not move in these scenarios). It also ignores that an option expiring inside the horizon stops
decaying, so the approximation is poorest for large shocks, big vol moves on long-dated options,
and short-dated options near the strike. ``approx_error = full - approx``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

from .contracts import OptionContract
from .conventions import DAYS_PER_YEAR, MIN_SCENARIO_VOL, year_fraction
from .engines.base import PricingEngine
from .engines.black_scholes import BlackScholesEngine
from .errors import InvalidInputError
from .market import MarketSnapshot
from .portfolio import Portfolio

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")
_FIELDS = ("price",) + GREEK_NAMES


@dataclass(frozen=True)
class Scenario:
    """Relative spot shock, absolute vol shock (decimal), calendar days forward."""

    spot_pct: float = 0.0
    vol_abs: float = 0.0
    days: int = 0

    def __post_init__(self) -> None:
        if self.days < 0 or int(self.days) != self.days:
            raise InvalidInputError(f"days must be a non-negative integer, got {self.days!r}")
        if self.spot_pct <= -1.0:
            raise InvalidInputError(f"spot_pct must be > -1, got {self.spot_pct!r}")

    @property
    def label(self) -> str:
        return f"spot {self.spot_pct:+.0%}, vol {self.vol_abs * 100:+.0f}pt, {self.days}d"


def default_scenarios(
    spot_pcts: Sequence[float] = (-0.10, -0.05, 0.0, 0.05, 0.10),
    vol_abs: Sequence[float] = (-0.05, 0.0, 0.05),
    days: Sequence[int] = (0, 1, 7),
) -> list[Scenario]:
    """Full grid. Ordered by days, then vol shock, then spot shock."""
    return [Scenario(s, v, d) for d, v, s in product(days, vol_abs, spot_pcts)]


@dataclass(frozen=True)
class PositionValuation:
    label: str
    underlying: str
    expiry: date
    option_type: str
    strike: float
    quantity: float
    multiplier: float
    unit_price: float
    value: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    delta_notional: float


@dataclass(frozen=True)
class ExposureRow:
    underlying: str
    expiry: date | None  # None when grouped by underlying only
    n_positions: int
    value: float
    delta: float
    delta_notional: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True)
class PortfolioValuation:
    portfolio_name: str
    engine: str
    total_value: float
    totals: dict[str, float]  # delta, gamma, vega, theta, rho, delta_notional
    positions: list[PositionValuation]
    by_underlying: list[ExposureRow]
    by_expiry: list[ExposureRow]
    by_underlying_expiry: list[ExposureRow]


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    base_value: float
    shocked_value: float
    full_pnl: float
    approx_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    n_expired: int  # positions whose expiry falls on or before the scenario date
    # Per-position P&L, filled only when detail=True.
    position_full_pnl: np.ndarray | None = field(default=None, repr=False, compare=False)
    position_approx_pnl: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def approx_error(self) -> float:
        """full - approx. Positive means the Greeks understated the P&L."""
        return self.full_pnl - self.approx_pnl

    @property
    def approx_error_pct(self) -> float:
        """Error as a percentage of |full P&L|; NaN when full P&L is ~0."""
        denom = abs(self.full_pnl)
        return float("nan") if denom < 1e-9 else 100.0 * self.approx_error / denom


class _Book:
    """Portfolio flattened into arrays, with the market each position uses."""

    def __init__(self, portfolio: Portfolio, markets: Mapping[str, MarketSnapshot]) -> None:
        if len(portfolio) == 0:
            raise InvalidInputError("portfolio has no positions")
        self.portfolio = portfolio
        self.contracts: list[OptionContract] = []
        self.markets: list[MarketSnapshot] = []
        for pos in portfolio:
            m = markets.get(pos.contract.underlying)
            if m is None:
                raise InvalidInputError(f"no market snapshot for underlying {pos.contract.underlying!r}")
            self.contracts.append(pos.contract)
            self.markets.append(m)
        self.sign = np.array([c.option_type.sign for c in self.contracts], dtype=float)
        self.strike = np.array([c.strike for c in self.contracts])
        self.tau = np.array([year_fraction(m.valuation_date, c.expiry) for c, m in zip(self.contracts, self.markets)])
        if (self.tau < 0).any():
            bad = self.contracts[int(np.argmax(self.tau < 0))]
            raise InvalidInputError(f"{bad.label}: expiry {bad.expiry} is before the valuation date")
        self.spot = np.array([m.spot for m in self.markets])
        self.rate = np.array([m.rate for m in self.markets])
        self.div = np.array([m.dividend_yield for m in self.markets])
        self.vol = np.array([m.volatility for m in self.markets])
        self.scale = np.array([p.scale for p in portfolio])


class RiskEngine:
    """Aggregates pricing-engine output over a portfolio and revalues it under scenarios.

    With a :class:`BlackScholesEngine` the whole book is priced in one vectorised call; any other
    engine is called position by position through the common ``price(contract, market)`` interface.
    Both routes apply identical shock rules and are tested to agree.
    """

    def __init__(self, engine: PricingEngine | None = None) -> None:
        self.engine = engine or BlackScholesEngine()

    # -- pricing a (possibly shocked) book --------------------------------------------
    def _price_book(self, book: _Book, scenario: Scenario, greeks: bool) -> dict[str, np.ndarray]:
        if isinstance(self.engine, BlackScholesEngine):
            spot = book.spot * (1.0 + scenario.spot_pct)
            vol = np.maximum(book.vol + scenario.vol_abs, MIN_SCENARIO_VOL)
            tau = np.maximum(book.tau - scenario.days / DAYS_PER_YEAR, 0.0)
            return self.engine.price_arrays(book.sign, spot, book.strike, tau, book.rate, book.div, vol, greeks=greeks)
        n = len(book.contracts)
        out = {k: np.empty(n) for k in (_FIELDS if greeks else ("price",))}
        for i, (contract, market) in enumerate(zip(book.contracts, book.markets)):
            shocked = market.shocked(scenario.spot_pct, scenario.vol_abs, 0)
            when = min(market.valuation_date + timedelta(days=scenario.days), contract.expiry)
            res = self.engine.price(contract, shocked.with_(valuation_date=when), greeks=greeks)
            out["price"][i] = res.price
            if greeks:
                assert res.greeks is not None
                for name in GREEK_NAMES:
                    out[name][i] = getattr(res.greeks, name)
        return out

    # -- valuation --------------------------------------------------------------------
    def value(self, portfolio: Portfolio, markets: Mapping[str, MarketSnapshot]) -> PortfolioValuation:
        book = _Book(portfolio, markets)
        arr = self._price_book(book, Scenario(), greeks=True)
        pos_vals: list[PositionValuation] = []
        for i, pos in enumerate(portfolio):
            s = book.scale[i]
            pos_vals.append(
                PositionValuation(
                    label=pos.contract.label,
                    underlying=pos.contract.underlying,
                    expiry=pos.contract.expiry,
                    option_type=pos.contract.option_type.value,
                    strike=pos.contract.strike,
                    quantity=pos.quantity,
                    multiplier=pos.multiplier,
                    unit_price=float(arr["price"][i]),
                    value=float(arr["price"][i] * s),
                    delta=float(arr["delta"][i] * s),
                    gamma=float(arr["gamma"][i] * s),
                    vega=float(arr["vega"][i] * s),
                    theta=float(arr["theta"][i] * s),
                    rho=float(arr["rho"][i] * s),
                    delta_notional=float(arr["delta"][i] * s * book.spot[i]),
                )
            )
        totals = {
            name: float(sum(getattr(p, name) for p in pos_vals)) for name in (*GREEK_NAMES, "delta_notional")
        }
        return PortfolioValuation(
            portfolio_name=portfolio.name,
            engine=self.engine.name,
            total_value=float(sum(p.value for p in pos_vals)),
            totals=totals,
            positions=pos_vals,
            by_underlying=_group(pos_vals, by_expiry=False),
            by_expiry=_group(pos_vals, by_expiry=True, by_underlying=False),
            by_underlying_expiry=_group(pos_vals, by_expiry=True),
        )

    # -- scenarios --------------------------------------------------------------------
    def run_scenarios(
        self,
        portfolio: Portfolio,
        markets: Mapping[str, MarketSnapshot],
        scenarios: Iterable[Scenario] | None = None,
        *,
        detail: bool = False,
    ) -> list[ScenarioResult]:
        """Full revaluation and Greek approximation for each scenario."""
        scenarios = list(scenarios) if scenarios is not None else default_scenarios()
        book = _Book(portfolio, markets)
        base = self._price_book(book, Scenario(), greeks=True)
        scale = book.scale
        base_pos_value = base["price"] * scale
        base_total = float(base_pos_value.sum())
        # Position-level Greeks in position units
        d, g, v, t = (base[k] * scale for k in ("delta", "gamma", "vega", "theta"))

        results: list[ScenarioResult] = []
        for sc in scenarios:
            shocked = self._price_book(book, sc, greeks=False)
            shocked_pos_value = shocked["price"] * scale
            full_pos = shocked_pos_value - base_pos_value

            d_spot = book.spot * sc.spot_pct
            d_vol_pts = (np.maximum(book.vol + sc.vol_abs, MIN_SCENARIO_VOL) - book.vol) * 100.0
            delta_pos = d * d_spot
            gamma_pos = 0.5 * g * d_spot * d_spot
            vega_pos = v * d_vol_pts
            theta_pos = t * sc.days
            approx_pos = delta_pos + gamma_pos + vega_pos + theta_pos

            results.append(
                ScenarioResult(
                    scenario=sc,
                    base_value=base_total,
                    shocked_value=float(shocked_pos_value.sum()),
                    full_pnl=float(full_pos.sum()),
                    approx_pnl=float(approx_pos.sum()),
                    delta_pnl=float(delta_pos.sum()),
                    gamma_pnl=float(gamma_pos.sum()),
                    vega_pnl=float(vega_pos.sum()),
                    theta_pnl=float(theta_pos.sum()),
                    n_expired=int((book.tau - sc.days / DAYS_PER_YEAR <= 0.0).sum()),
                    position_full_pnl=full_pos if detail else None,
                    position_approx_pnl=approx_pos if detail else None,
                )
            )
        return results


def _group(positions: list[PositionValuation], *, by_expiry: bool, by_underlying: bool = True) -> list[ExposureRow]:
    buckets: dict[tuple[str, date | None], list[PositionValuation]] = {}
    for p in positions:
        key = (p.underlying if by_underlying else "ALL", p.expiry if by_expiry else None)
        buckets.setdefault(key, []).append(p)

    def key_order(k: tuple[str, date | None]) -> tuple[str, date]:
        return k[0], k[1] or date.min

    rows = []
    for key in sorted(buckets, key=key_order):
        items = buckets[key]
        rows.append(
            ExposureRow(
                underlying=key[0],
                expiry=key[1],
                n_positions=len(items),
                value=sum(p.value for p in items),
                delta=sum(p.delta for p in items),
                delta_notional=sum(p.delta_notional for p in items),
                gamma=sum(p.gamma for p in items),
                vega=sum(p.vega for p in items),
                theta=sum(p.theta for p in items),
                rho=sum(p.rho for p in items),
            )
        )
    return rows

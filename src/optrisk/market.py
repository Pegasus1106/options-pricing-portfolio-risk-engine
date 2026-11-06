"""Market snapshot: the market inputs needed to value an option at one point in time."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

from .conventions import MAX_ABS_RATE, MAX_VOLATILITY, MIN_SCENARIO_VOL
from .contracts import OptionContract
from .errors import InvalidInputError


def _finite(name: str, value: object) -> float:
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise InvalidInputError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(x):
        raise InvalidInputError(f"{name} must be finite, got {value!r}")
    return x


@dataclass(frozen=True)
class MarketSnapshot:
    """Flat market assumptions for one underlying at one valuation date.

    All rates and volatilities are decimals (0.05 = 5 %), continuously compounded,
    annualised. One constant volatility applies to every strike and expiry (no smile).
    """

    underlying: str
    valuation_date: date
    spot: float
    rate: float
    dividend_yield: float
    volatility: float

    def __post_init__(self) -> None:
        if not isinstance(self.underlying, str) or not self.underlying.strip():
            raise InvalidInputError("underlying must be a non-empty string")
        if isinstance(self.valuation_date, datetime) or not isinstance(self.valuation_date, date):
            raise InvalidInputError(
                f"valuation_date must be a datetime.date, got {type(self.valuation_date).__name__}"
            )
        spot = _finite("spot", self.spot)
        rate = _finite("rate", self.rate)
        div = _finite("dividend_yield", self.dividend_yield)
        vol = _finite("volatility", self.volatility)
        if spot <= 0.0:
            raise InvalidInputError(f"spot must be > 0, got {spot}")
        if abs(rate) > MAX_ABS_RATE:
            raise InvalidInputError(f"rate must satisfy |rate| <= {MAX_ABS_RATE} (decimal, not percent), got {rate}")
        if abs(div) > MAX_ABS_RATE:
            raise InvalidInputError(
                f"dividend_yield must satisfy |q| <= {MAX_ABS_RATE} (decimal, not percent), got {div}"
            )
        if vol <= 0.0:
            raise InvalidInputError(f"volatility must be > 0, got {vol}")
        if vol > MAX_VOLATILITY:
            raise InvalidInputError(f"volatility must be <= {MAX_VOLATILITY} (decimal, not percent), got {vol}")
        object.__setattr__(self, "spot", spot)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "dividend_yield", div)
        object.__setattr__(self, "volatility", vol)

    def with_(self, **changes: object) -> "MarketSnapshot":
        """Copy with some fields replaced (validated again)."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def shocked(self, spot_pct: float = 0.0, vol_abs: float = 0.0, days: int = 0) -> "MarketSnapshot":
        """Return a scenario market.

        ``spot_pct``  relative spot change (-0.05 = -5 %).
        ``vol_abs``   absolute volatility change in decimals (+0.05 = +5 vol points).
                      The result is floored at ``MIN_SCENARIO_VOL`` so large down-shocks stay valid.
        ``days``      calendar days the valuation date moves forward.
        """
        if spot_pct <= -1.0:
            raise InvalidInputError(f"spot_pct must be > -1 (a -100% move gives zero spot), got {spot_pct}")
        return self.with_(
            spot=self.spot * (1.0 + spot_pct),
            volatility=max(self.volatility + vol_abs, MIN_SCENARIO_VOL),
            valuation_date=self.valuation_date + timedelta(days=int(days)),
        )

    def check_contract(self, contract: OptionContract) -> None:
        if contract.underlying != self.underlying:
            raise InvalidInputError(
                f"contract underlying {contract.underlying!r} does not match market underlying {self.underlying!r}"
            )

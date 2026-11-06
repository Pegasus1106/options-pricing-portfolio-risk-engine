"""Option contract definition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

from .errors import InvalidInputError


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> int:
        """+1 for calls, -1 for puts. Payoff is ``max(sign * (S - K), 0)``."""
        return 1 if self is OptionType.CALL else -1

    @classmethod
    def parse(cls, text: "str | OptionType") -> "OptionType":
        if isinstance(text, OptionType):
            return text
        key = str(text).strip().lower()
        if key in ("call", "c"):
            return cls.CALL
        if key in ("put", "p"):
            return cls.PUT
        raise InvalidInputError(f"unknown option type {text!r}; expected 'call' or 'put'")


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"

    @classmethod
    def parse(cls, text: "str | ExerciseStyle") -> "ExerciseStyle":
        if isinstance(text, ExerciseStyle):
            return text
        key = str(text).strip().lower()
        if key in ("european", "eur", "e"):
            return cls.EUROPEAN
        if key in ("american", "amer", "a"):
            return cls.AMERICAN
        raise InvalidInputError(f"unknown exercise style {text!r}; expected 'european' or 'american'")


@dataclass(frozen=True)
class OptionContract:
    """A vanilla option on one unit of an underlying.

    ``strike`` is in the same units as the underlying spot. The contract carries no
    quantity or multiplier; those belong to a :class:`~optrisk.portfolio.Position`.
    """

    underlying: str
    strike: float
    expiry: date
    option_type: OptionType
    exercise: ExerciseStyle = ExerciseStyle.EUROPEAN
    contract_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.underlying, str) or not self.underlying.strip():
            raise InvalidInputError("underlying must be a non-empty string")
        if isinstance(self.expiry, datetime) or not isinstance(self.expiry, date):
            raise InvalidInputError(f"expiry must be a datetime.date, got {type(self.expiry).__name__}")
        try:
            strike = float(self.strike)
        except (TypeError, ValueError):
            raise InvalidInputError(f"strike must be a number, got {self.strike!r}") from None
        if not math.isfinite(strike) or strike <= 0.0:
            raise InvalidInputError(f"strike must be finite and > 0, got {self.strike!r}")
        object.__setattr__(self, "strike", strike)
        object.__setattr__(self, "option_type", OptionType.parse(self.option_type))
        object.__setattr__(self, "exercise", ExerciseStyle.parse(self.exercise))

    @property
    def is_call(self) -> bool:
        return self.option_type is OptionType.CALL

    def payoff(self, spot: float) -> float:
        """Payoff per unit of underlying at a given terminal spot."""
        return max(self.option_type.sign * (spot - self.strike), 0.0)

    @property
    def label(self) -> str:
        return self.contract_id or (
            f"{self.underlying}-{self.expiry.isoformat()}-{self.option_type.value[0].upper()}{self.strike:g}"
        )

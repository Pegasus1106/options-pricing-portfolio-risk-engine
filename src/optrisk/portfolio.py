"""Positions and portfolios."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Iterator

from .contracts import OptionContract
from .errors import InvalidInputError

DEFAULT_MULTIPLIER = 100.0  # standard listed equity option: 100 shares per contract


@dataclass(frozen=True)
class Position:
    """``quantity`` contracts of ``contract``; positive is long, negative is short.

    Position value = unit price * quantity * multiplier, where the multiplier is the number of
    units of the underlying per contract.
    """

    contract: OptionContract
    quantity: float
    multiplier: float = DEFAULT_MULTIPLIER

    def __post_init__(self) -> None:
        try:
            qty = float(self.quantity)
            mult = float(self.multiplier)
        except (TypeError, ValueError):
            raise InvalidInputError(f"quantity and multiplier must be numbers, got {self.quantity!r}, {self.multiplier!r}") from None
        if not math.isfinite(qty):
            raise InvalidInputError(f"quantity must be finite, got {self.quantity!r}")
        if not math.isfinite(mult) or mult <= 0.0:
            raise InvalidInputError(f"multiplier must be finite and > 0, got {self.multiplier!r}")
        object.__setattr__(self, "quantity", qty)
        object.__setattr__(self, "multiplier", mult)

    @property
    def scale(self) -> float:
        """Units of underlying represented by the position (signed)."""
        return self.quantity * self.multiplier


@dataclass(frozen=True)
class Portfolio:
    positions: tuple[Position, ...] = field(default_factory=tuple)
    name: str = "portfolio"

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", tuple(self.positions))
        for p in self.positions:
            if not isinstance(p, Position):
                raise InvalidInputError(f"portfolio entries must be Position objects, got {type(p).__name__}")

    def __len__(self) -> int:
        return len(self.positions)

    def __iter__(self) -> Iterator[Position]:
        return iter(self.positions)

    @property
    def underlyings(self) -> list[str]:
        return sorted({p.contract.underlying for p in self.positions})

    @property
    def expiries(self) -> list[date]:
        return sorted({p.contract.expiry for p in self.positions})

    @classmethod
    def of(cls, positions: Iterable[Position], name: str = "portfolio") -> "Portfolio":
        return cls(tuple(positions), name)

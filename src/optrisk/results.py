"""Result containers returned by pricing engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Greeks:
    """Sensitivities of the per-unit option price. Units are fixed in ``conventions.py``.

    delta  per 1.0 of spot            gamma  per 1.0 of spot, squared
    vega   per +1 vol point (0.01)    theta  per calendar day
    rho    per +1 rate point (0.01)
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PricingResult:
    """Output of :meth:`PricingEngine.price`.

    ``price`` is per one unit of the underlying. ``std_error`` and ``confidence_interval``
    are only set by statistical methods (Monte Carlo). ``diagnostics`` holds engine-specific
    information (steps, path counts, seeds, expiry flags, Greek standard errors, ...).
    """

    price: float
    method: str
    greeks: Greeks | None = None
    std_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"price": self.price, "method": self.method, "runtime_s": self.runtime_s}
        if self.greeks is not None:
            out.update(self.greeks.as_dict())
        if self.std_error is not None:
            out["std_error"] = self.std_error
        if self.confidence_interval is not None:
            out["ci_low"], out["ci_high"] = self.confidence_interval
        return out

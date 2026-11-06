"""optrisk: European option pricing, implied volatility, and portfolio scenario risk."""

from .contracts import ExerciseStyle, OptionContract, OptionType
from .engines import BinomialEngine, BlackScholesEngine, MonteCarloEngine, PricingEngine
from .errors import InvalidInputError, NumericalError, OptRiskError, UnsupportedInstrumentError
from .market import MarketSnapshot
from .results import Greeks, PricingResult

__version__ = "0.1.0"

__all__ = [
    "OptionContract",
    "OptionType",
    "ExerciseStyle",
    "MarketSnapshot",
    "PricingEngine",
    "BlackScholesEngine",
    "BinomialEngine",
    "MonteCarloEngine",
    "PricingResult",
    "Greeks",
    "OptRiskError",
    "InvalidInputError",
    "NumericalError",
    "UnsupportedInstrumentError",
    "__version__",
]

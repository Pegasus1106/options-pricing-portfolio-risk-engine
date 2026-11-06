"""Explicit unit and time conventions used everywhere in the library.

Change nothing here without also updating docs/CONVENTIONS.md and the
QuantLib comparison, which converts QuantLib's units to these.

Reported units (``PricingResult.greeks`` and everything built on it)
--------------------------------------------------------------------
price   per one unit of the underlying (NOT per contract). Position value is
        ``price * quantity * multiplier``.
delta   dV/dS, per 1.0 of spot.
gamma   d2V/dS2, per 1.0 of spot squared.
vega    change in price for a +1 volatility point (+0.01 absolute vol).
theta   change in price per calendar day of time passing (negative for long
        options in the usual case).
rho     change in price for a +1 interest-rate point (+0.01 absolute rate).

Time
----
Time to expiry is ACT/365 Fixed: ``(expiry_date - valuation_date).days / 365``.
No holiday calendar, no intraday time. This matches QuantLib's
``Actual365Fixed`` with a ``NullCalendar``.
"""

from __future__ import annotations

from datetime import date

DAYS_PER_YEAR: float = 365.0

#: Multiply a per-unit-volatility vega by this to get "per volatility point".
VEGA_UNIT: float = 0.01
#: Multiply a per-unit-rate rho by this to get "per rate point".
RHO_UNIT: float = 0.01
#: Multiply a per-year theta by this to get "per calendar day".
THETA_PER_DAY: float = 1.0 / DAYS_PER_YEAR

#: Upper sanity limit for volatility (1000 %). Catches percent-vs-decimal mix-ups.
MAX_VOLATILITY: float = 10.0
#: Sanity limit for |rate| and |dividend yield| (100 %). Same purpose.
MAX_ABS_RATE: float = 1.0
#: Volatility floor applied by scenario shocks so that a large negative shock never produces vol <= 0.
MIN_SCENARIO_VOL: float = 1e-4


def year_fraction(start: date, end: date) -> float:
    """ACT/365 Fixed year fraction. Negative when ``end`` is before ``start``."""
    return (end - start).days / DAYS_PER_YEAR

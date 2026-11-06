"""Exception types raised by optrisk.

All library errors derive from :class:`OptRiskError`, so callers can catch a
single type. Input problems also derive from ``ValueError`` so that generic
Python error handling keeps working.
"""

from __future__ import annotations


class OptRiskError(Exception):
    """Base class for all optrisk errors."""


class InvalidInputError(OptRiskError, ValueError):
    """An input is malformed, non-finite, or outside its valid domain."""


class UnsupportedInstrumentError(OptRiskError, ValueError):
    """The chosen engine cannot price the requested instrument (e.g. BSM with American exercise)."""


class NumericalError(OptRiskError, ArithmeticError):
    """A numerical method cannot produce a trustworthy answer with the given settings."""

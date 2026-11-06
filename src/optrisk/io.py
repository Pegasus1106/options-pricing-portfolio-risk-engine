"""CSV / JSON input and output.

Input files (CSV with a header row, or JSON as a list of objects or ``{"<key>": [...]}``):

contracts   contract_id, underlying, option_type, strike, expiry [, exercise]
positions   contract_id, quantity [, multiplier]
market      underlying, valuation_date, spot, rate, dividend_yield, volatility
quotes      contract_id, price

Dates are ISO ``YYYY-MM-DD``. Rates and volatilities are decimals (0.05 = 5 %).
Every error names the file and the row so a bad record can be found quickly.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import ExerciseStyle, OptionContract, OptionType
from .errors import InvalidInputError
from .market import MarketSnapshot
from .portfolio import DEFAULT_MULTIPLIER, Portfolio, Position

Record = dict[str, Any]


def read_records(path: str | Path, key: str) -> list[tuple[int, Record]]:
    """Read rows as ``(row_number, record)``. Row numbers are 1-based data rows (header excluded)."""
    path = Path(path)
    if not path.exists():
        raise InvalidInputError(f"file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise InvalidInputError(f"{path}: empty file")
            rows = []
            for i, row in enumerate(reader, start=1):
                if None in row:
                    raise InvalidInputError(f"{path}: row {i} has more fields than the header")
                rows.append((i, {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}))
            return rows
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidInputError(f"{path}: invalid JSON ({exc})") from None
        if isinstance(data, dict):
            if key not in data:
                raise InvalidInputError(f"{path}: JSON object must contain a '{key}' list")
            data = data[key]
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise InvalidInputError(f"{path}: expected a list of objects")
        return list(enumerate(data, start=1))
    raise InvalidInputError(f"{path}: unsupported file type {suffix!r}; use .csv or .json")


def _require(path: Path, row: int, rec: Record, fields: Sequence[str]) -> None:
    missing = [f for f in fields if f not in rec or rec[f] in (None, "")]
    if missing:
        raise InvalidInputError(f"{path}: row {row} is missing required field(s): {', '.join(missing)}")


def _num(path: Path, row: int, rec: Record, name: str) -> float:
    try:
        value = float(rec[name])
    except (TypeError, ValueError):
        raise InvalidInputError(f"{path}: row {row}: {name} must be a number, got {rec[name]!r}") from None
    if not math.isfinite(value):
        raise InvalidInputError(f"{path}: row {row}: {name} must be finite, got {rec[name]!r}")
    return value


def _date(path: Path, row: int, rec: Record, name: str) -> date:
    try:
        return date.fromisoformat(str(rec[name]))
    except ValueError:
        raise InvalidInputError(f"{path}: row {row}: {name} must be an ISO date (YYYY-MM-DD), got {rec[name]!r}") from None


def _wrap(path: Path, row: int, build):  # type: ignore[no-untyped-def]
    try:
        return build()
    except InvalidInputError as exc:
        if str(exc).startswith(str(path)):
            raise
        raise InvalidInputError(f"{path}: row {row}: {exc}") from None


def load_contracts(path: str | Path) -> dict[str, OptionContract]:
    path = Path(path)
    out: dict[str, OptionContract] = {}
    for row, rec in read_records(path, "contracts"):
        _require(path, row, rec, ("contract_id", "underlying", "option_type", "strike", "expiry"))
        cid = str(rec["contract_id"])
        if cid in out:
            raise InvalidInputError(f"{path}: row {row}: duplicate contract_id {cid!r}")
        out[cid] = _wrap(
            path,
            row,
            lambda: OptionContract(
                underlying=str(rec["underlying"]),
                strike=_num(path, row, rec, "strike"),
                expiry=_date(path, row, rec, "expiry"),
                option_type=OptionType.parse(rec["option_type"]),
                exercise=ExerciseStyle.parse(rec.get("exercise") or "european"),
                contract_id=cid,
            ),
        )
    if not out:
        raise InvalidInputError(f"{path}: no contracts found")
    return out


def load_markets(path: str | Path) -> dict[str, MarketSnapshot]:
    path = Path(path)
    out: dict[str, MarketSnapshot] = {}
    for row, rec in read_records(path, "markets"):
        _require(path, row, rec, ("underlying", "valuation_date", "spot", "rate", "dividend_yield", "volatility"))
        name = str(rec["underlying"])
        if name in out:
            raise InvalidInputError(f"{path}: row {row}: duplicate market for underlying {name!r}")
        out[name] = _wrap(
            path,
            row,
            lambda: MarketSnapshot(
                underlying=name,
                valuation_date=_date(path, row, rec, "valuation_date"),
                spot=_num(path, row, rec, "spot"),
                rate=_num(path, row, rec, "rate"),
                dividend_yield=_num(path, row, rec, "dividend_yield"),
                volatility=_num(path, row, rec, "volatility"),
            ),
        )
    if not out:
        raise InvalidInputError(f"{path}: no market snapshots found")
    return out


def load_positions(path: str | Path, contracts: Mapping[str, OptionContract], name: str = "portfolio") -> Portfolio:
    path = Path(path)
    positions: list[Position] = []
    for row, rec in read_records(path, "positions"):
        _require(path, row, rec, ("contract_id", "quantity"))
        cid = str(rec["contract_id"])
        if cid not in contracts:
            raise InvalidInputError(f"{path}: row {row}: unknown contract_id {cid!r}")
        mult = _num(path, row, rec, "multiplier") if rec.get("multiplier") not in (None, "") else DEFAULT_MULTIPLIER
        positions.append(
            _wrap(path, row, lambda: Position(contracts[cid], _num(path, row, rec, "quantity"), mult))
        )
    if not positions:
        raise InvalidInputError(f"{path}: no positions found")
    return Portfolio(tuple(positions), name=name)


def load_quotes(path: str | Path) -> dict[str, float]:
    path = Path(path)
    out: dict[str, float] = {}
    for row, rec in read_records(path, "quotes"):
        _require(path, row, rec, ("contract_id", "price"))
        cid = str(rec["contract_id"])
        if cid in out:
            raise InvalidInputError(f"{path}: row {row}: duplicate quote for {cid!r}")
        try:
            # NaN/inf parse fine here on purpose: the solver reports them as INVALID_INPUT per quote
            # instead of one bad quote rejecting the whole file.
            out[cid] = float(rec["price"])
        except (TypeError, ValueError):
            raise InvalidInputError(f"{path}: row {row}: price must be a number, got {rec['price']!r}") from None
    return out


# ---------------------------------------------------------------------------- writers
def _jsonable(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_records(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    """Write rows to ``path`` as CSV or JSON, chosen by file extension. Creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames) if fieldnames else (list(rows[0].keys()) if rows else [])
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow({k: (repr(v) if isinstance(v, float) else _jsonable(v)) for k, v in r.items()})
    elif suffix == ".json":
        path.write_text(json.dumps([{k: _jsonable(v) for k, v in r.items()} for r in rows], indent=2) + "\n", encoding="utf-8")
    else:
        raise InvalidInputError(f"{path}: unsupported output type {suffix!r}; use .csv or .json")
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if isinstance(o, date):
            return o.isoformat()
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "value"):
            return o.value
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")

    def clean(o: Any) -> Any:  # strict JSON has no NaN/Infinity; write them as null
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o

    path.write_text(
        json.dumps(clean(payload), indent=2, default=default, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path

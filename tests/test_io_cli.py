import csv
import json

import pytest

from optrisk import InvalidInputError
from optrisk.cli import main
from optrisk.generate import sample_markets, sample_portfolio, write_sample_data
from optrisk.io import load_contracts, load_markets, load_positions, load_quotes, read_records, write_records


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    return d, write_sample_data(d)


def test_generated_files_load_back_to_the_same_portfolio(data):
    _, p = data
    contracts = load_contracts(p["contracts"])
    pf = load_positions(p["positions"], contracts)
    ref = sample_portfolio()
    assert len(pf) == len(ref)
    assert [x.contract for x in pf] == [x.contract for x in ref]
    assert [x.quantity for x in pf] == [x.quantity for x in ref]
    mk = load_markets(p["market"])
    assert mk == sample_markets()


def test_json_inputs_work(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"contracts": [
        {"contract_id": "A", "underlying": "X", "option_type": "Call", "strike": 100, "expiry": "2026-06-01", "exercise": "european"}]}))
    (tmp_path / "m.json").write_text(json.dumps([
        {"underlying": "X", "valuation_date": "2026-01-05", "spot": 100, "rate": 0.03, "dividend_yield": 0, "volatility": 0.2}]))
    (tmp_path / "p.json").write_text(json.dumps([{"contract_id": "A", "quantity": -3}]))
    c = load_contracts(tmp_path / "c.json")
    pf = load_positions(tmp_path / "p.json", c)
    assert pf.positions[0].quantity == -3 and pf.positions[0].multiplier == 100.0  # default multiplier
    assert load_markets(tmp_path / "m.json")["X"].spot == 100.0


def _write(path, text):
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    "text,message",
    [
        ("contract_id,underlying,option_type,strike\nA,X,call,100\n", "missing required field"),
        ("contract_id,underlying,option_type,strike,expiry\nA,X,call,100,2026-13-45\n", "ISO date"),
        ("contract_id,underlying,option_type,strike,expiry\nA,X,call,abc,2026-06-01\n", "strike must be a number"),
        ("contract_id,underlying,option_type,strike,expiry\nA,X,call,-5,2026-06-01\n", "row 1"),
        ("contract_id,underlying,option_type,strike,expiry\nA,X,butterfly,5,2026-06-01\n", "unknown option type"),
        ("contract_id,underlying,option_type,strike,expiry\nA,X,call,5,2026-06-01\nA,X,put,5,2026-06-01\n", "duplicate contract_id"),
        ("contract_id,underlying,option_type,strike,expiry\n", "no contracts"),
    ],
)
def test_contract_file_errors_name_file_and_row(tmp_path, text, message):
    p = _write(tmp_path / "c.csv", text)
    with pytest.raises(InvalidInputError, match=message) as exc:
        load_contracts(p)
    assert str(p) in str(exc.value)


def test_market_and_position_errors(tmp_path):
    with pytest.raises(InvalidInputError, match="volatility must be <="):
        load_markets(_write(tmp_path / "m.csv", "underlying,valuation_date,spot,rate,dividend_yield,volatility\nX,2026-01-05,100,0.03,0,25\n"))
    c = load_contracts(_write(tmp_path / "c.csv", "contract_id,underlying,option_type,strike,expiry\nA,X,call,100,2026-06-01\n"))
    with pytest.raises(InvalidInputError, match="unknown contract_id"):
        load_positions(_write(tmp_path / "p.csv", "contract_id,quantity\nZ,1\n"), c)
    with pytest.raises(InvalidInputError, match="multiplier"):
        load_positions(_write(tmp_path / "p2.csv", "contract_id,quantity,multiplier\nA,1,0\n"), c)


def test_unreadable_files(tmp_path):
    with pytest.raises(InvalidInputError, match="not found"):
        read_records(tmp_path / "missing.csv", "x")
    with pytest.raises(InvalidInputError, match="invalid JSON"):
        read_records(_write(tmp_path / "b.json", "{oops"), "x")
    with pytest.raises(InvalidInputError, match="unsupported file type"):
        read_records(_write(tmp_path / "b.txt", "x"), "x")


def test_quotes_pass_nan_through_for_per_quote_status(tmp_path):
    q = load_quotes(_write(tmp_path / "q.csv", "contract_id,price\nA,1.5\nB,nan\n"))
    assert q["A"] == 1.5 and q["B"] != q["B"]


def test_write_records_round_trip_preserves_float_precision(tmp_path):
    p = write_records(tmp_path / "o.csv", [{"a": 0.1 + 0.2, "b": "x"}])
    with p.open() as fh:
        row = next(csv.DictReader(fh))
    assert float(row["a"]) == 0.1 + 0.2


# ----------------------------------------------------------------------------- CLI
def test_cli_price_all_engines(data, tmp_path, capsys):
    _, p = data
    for extra in (["--engine", "bsm"], ["--engine", "binomial", "--steps", "100"], ["--engine", "mc", "--paths", "20000", "--seed", "1"]):
        out = tmp_path / f"price_{extra[1]}.csv"
        assert main(["price", "--contracts", str(p["contracts"]), "--market", str(p["market"]), "--output", str(out), *extra]) == 0
        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 18 and all(r["status"] == "ok" for r in rows)
    capsys.readouterr()


def test_cli_price_reports_row_errors_and_exit_code(tmp_path, data, capsys):
    _, p = data
    bad = _write(tmp_path / "c.csv", "contract_id,underlying,option_type,strike,expiry\nGOOD,SIM_A,call,100,2026-06-01\nSTALE,SIM_A,call,100,2025-01-01\nNOMKT,ZZZ,call,100,2026-06-01\n")
    out = tmp_path / "o.csv"
    code = main(["price", "--contracts", str(bad), "--market", str(p["market"]), "--output", str(out)])
    assert code == 3
    status = {r["contract_id"]: r["status"] for r in csv.DictReader(out.open())}
    assert status["GOOD"] == "ok" and status["STALE"].startswith("error") and "no market snapshot" in status["NOMKT"]
    capsys.readouterr()


def test_cli_implied_vol(data, tmp_path, capsys):
    _, p = data
    out = tmp_path / "iv.csv"
    assert main(["implied-vol", "--contracts", str(p["chain_contracts"]), "--market", str(p["market"]), "--quotes", str(p["chain_quotes"]), "--output", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    statuses = {r["status"] for r in rows}
    assert {"converged", "price_below_lower_bound", "price_above_upper_bound", "invalid_input"} <= statuses
    capsys.readouterr()


def test_cli_risk_and_scenario(data, tmp_path, capsys):
    _, p = data
    common = ["--contracts", str(p["contracts"]), "--positions", str(p["positions"]), "--market", str(p["market"])]
    assert main(["risk", *common, "--output-dir", str(tmp_path / "risk")]) == 0
    assert (tmp_path / "risk" / "portfolio_valuation.csv").exists()
    assert main(["scenario", *common, "--outdir", str(tmp_path / "scn"), "--days", "0,7"]) == 0
    text = (tmp_path / "scn" / "scenario_report.md").read_text()
    assert "Approximation error" in text and "Exposure by underlying and expiry" in text
    rows = list(csv.DictReader((tmp_path / "scn" / "scenario_grid.csv").open()))
    assert len(rows) == 5 * 3 * 2
    capsys.readouterr()


def test_cli_invalid_file_returns_error_code_2(tmp_path, capsys):
    code = main(["price", "--contracts", str(tmp_path / "nope.csv"), "--market", str(tmp_path / "nope2.csv")])
    assert code == 2
    assert "error:" in capsys.readouterr().err

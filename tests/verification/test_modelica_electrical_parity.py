"""Code-to-code comparison of noodl's circuits with OpenModelica on MSL and MBL examples.

The reference models are `Modelica.Electrical.Analog` (MSL 4.1.0) and `Buildings.Electrical`
(MBL v13.0.0, commit 55abf579) examples simulated by OpenModelica 1.27.1 at a DASSL tolerance
of 1e-12 (`tests/data/modelica_electrical/NOTICE.md`). Each model is read with
`read_modelica_circuit`, run with `simulate_circuit` on the reference CSV's own time grid, and
every compared column of the CSV is checked at EVERY row.

Algebraic models (no capacitor, no inductor): the circuit is algebraic in its sources and
signals, so noodl's steady solve at each output time is the problem the DAE solver solves, and
agreement is to round-off:

    |noodl - omc| <= 1e-9 |omc| + 1e-12   (units of the column: V, A or K).

The tolerance is not loosened for any model, column or row. The steady solves use the
runner's own Newton tolerances (`run.STEADY_ATOL`, `run.STEADY_RTOL`; `THERMAL_STEADY_ATOL`
for a circuit with a heat-port thermal network); nothing looser is passed.

MBL's `ACLine_RL` example is compared with its dynamic branch (`RL_3`, a `FixedZ_dynamic`
line, and its load `load_sc_3`) omitted; the source `E` feeds that branch too, so its current
columns are not this model's to compare and are skipped for that model only. Every other CSV
column must map to a noodl variable (`CircuitNames.value`).

The worst absolute and relative error per model and column is written, when the environment
variable `NOODL_RECORD_PARITY=1` is set, to the committed
`tests/data/modelica_electrical/parity-electrical.json` for the documentation. Recording is
opt-in: a plain checkout or a CI run never rewrites that file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.electrical.modelica import read_modelica_circuit
from noodl.apps.electrical.modelica.run import simulate_circuit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "modelica_electrical"
RECORD = DATA / "parity-electrical.json"

ALGEBRAIC = ("Analog.Examples.Resistor", "Analog.Examples.ShowVariableResistor",
             "DC.Loads.Examples.Resistor", "DC.Loads.Examples.VariableLoad",
             "AC.OnePhase.Loads.Examples.TestImpedance", "AC.OnePhase.Lines.Examples.ACLine_RL",
             "AC.OnePhase.Conversion.Examples.ACDCConverter",
             "AC.OnePhase.Conversion.Examples.ACACTransformer",
             "AC.ThreePhasesBalanced.Lines.Examples.ACLine")
DYNAMIC = ("Analog.Examples.SeriesResonance", "Analog.Examples.CauerLowPassAnalog",
           "Analog.Examples.CompareTransformers", "Analog.Examples.HeatingRectifier",
           "DC.Lines.Examples.RCModel")
OMIT = {"AC.OnePhase.Lines.Examples.ACLine_RL": ("RL_3", "load_sc_3")}
# E feeds the omitted RL_3 as well, so its current is not this model's to compare.
SKIPPED_COLUMNS = {"AC.OnePhase.Lines.Examples.ACLine_RL": ("E.terminal.i[1]",
                                                            "E.terminal.i[2]")}
RTOL_ALGEBRAIC, ATOL_ALGEBRAIC = 1e-9, 1e-12  # never loosened


def _csv(stem: str) -> tuple[list[str], np.ndarray]:
    lines = [ln for ln in (DATA / f"{stem}.csv").read_text().splitlines()
             if ln and not ln.startswith("#")]
    head = [h.strip('"') for h in lines[0].split(",")]
    data = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
    return head, data


def _record(stem: str, stats: dict, record: Path = RECORD) -> None:
    # Deliberate side effect: the measured errors go into the committed
    # tests/data/modelica_electrical/parity-electrical.json for the documentation, only
    # when NOODL_RECORD_PARITY=1 is set -- a plain checkout or a CI run never rewrites it.
    if os.environ.get("NOODL_RECORD_PARITY") != "1":
        return
    lock = record.with_suffix(".lock")
    for _ in range(6000):  # one writer at a time: models may be recorded in parallel
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        raise TimeoutError(f"{lock} held for 5 minutes")
    try:
        try:
            doc = json.loads(record.read_text()) if record.exists() else {}
        except json.JSONDecodeError:
            doc = {}
        doc[stem] = stats
        record.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    finally:
        os.close(fd)
        os.remove(lock)


# MSL's Resistor example couples a heat-port thermal network (Joule heat in, temperature-
# dependent resistance back) at every row: ~95 ms a row, ~48 s over its 501 rows. Its full
# comparison is marked slow; the default run compares every tenth row. Every model listed
# here is slow in full and subsampled in the default run.
SUBSAMPLED = {"Analog.Examples.Resistor": 10}


def _check(stem: str, column: str, times: np.ndarray, mine: np.ndarray,
           omc: np.ndarray) -> dict:
    """Assert `mine` is within the bound of `omc` at every row (a NaN or infinity on either
    side fails, naming the first such row); the worst absolute and relative error."""
    with np.errstate(invalid="ignore"):  # inf - inf: NaN, caught as off the bound below
        err = np.abs(mine - omc)
    within = err <= RTOL_ALGEBRAIC * np.abs(omc) + ATOL_ALGEBRAIC
    bad = ~(np.isfinite(mine) & np.isfinite(omc) & within)
    if bad.any():
        k = int(bad.argmax())
        raise AssertionError(f"{stem}: {column} off the bound from t = {float(times[k])!r}: "
                             f"noodl {float(mine[k])!r}, omc {float(omc[k])!r}")
    return {"max_abs": float(err.max()),
            "max_rel": float((err / np.maximum(np.abs(omc), ATOL_ALGEBRAIC)).max())}


def _compare(stem: str, every: int = 1) -> dict:
    """Run `stem` at every `every`-th CSV row and check each compared column against the
    bound; the worst absolute and relative error per column."""
    head, data = _csv(stem)
    data = data[::every]
    model, state, drivers, names = read_modelica_circuit(DATA / f"{stem}.json",
                                                         omit=OMIT.get(stem, ()))
    assert not names.dynamic
    times = torch.as_tensor(data[:, 0])
    torch.testing.assert_close(names.times[::every], times, rtol=0, atol=1e-12)
    out = simulate_circuit(model, state, drivers, names,
                           times=None if every == 1 else names.times[::every])
    skipped = set(SKIPPED_COLUMNS.get(stem, ()))
    omitted = {h for h in head[1:] if h.split(".")[0] in OMIT.get(stem, ())}
    stats = {}
    for j, h in enumerate(head[1:], start=1):
        if h in skipped or h in omitted:
            continue
        assert h in names.columns, f"{stem}: CSV column {h!r} maps to no noodl variable"
        stats[h] = _check(stem, h, data[:, 0], names.value(out, h).numpy(), data[:, j])
    if names.phasor:  # DC parts of a phasor model stay real
        dc = [n for n in names.nodes if n.endswith(("[1]", "[2]"))]
        for n in dc:
            assert float(out["electrical.phi"][..., names.nodes[n]].imag.abs().max()) <= 1e-12
    return stats


@pytest.mark.parametrize("stem", [pytest.param(s, marks=pytest.mark.slow)
                                  if s in SUBSAMPLED else s for s in ALGEBRAIC])
def test_parity_on_the_algebraic_models(stem):
    _record(stem, _compare(stem))


@pytest.mark.parametrize("stem", sorted(SUBSAMPLED))
def test_parity_on_the_algebraic_models_at_subsampled_rows(stem):
    _compare(stem, every=SUBSAMPLED[stem])


def test_the_bound_check_fails_on_a_nan_or_an_infinity():
    t = np.array([0.0, 1.0, 2.0])
    omc = np.array([1.0, 2.0, 3.0])
    assert _check("m", "c", t, omc.copy(), omc)["max_abs"] == 0.0
    for bad in (np.nan, np.inf):
        mine = omc.copy()
        mine[1] = bad
        with pytest.raises(AssertionError, match=r"m: c off the bound from t = 1\.0"):
            _check("m", "c", t, mine, omc)
        with pytest.raises(AssertionError, match=r"t = 1\.0"):
            _check("m", "c", t, omc, mine)


def test_every_fixture_is_parity_checked():
    stems = {p.stem for p in DATA.glob("*.json")} - {RECORD.stem}
    assert stems == set(ALGEBRAIC) | set(DYNAMIC)

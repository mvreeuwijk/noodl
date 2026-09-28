"""Parity with the Modelica Buildings Library on its multizone models.

The algebraic models are checked to a fixed tolerance (below); the dynamic models, those
with volumes, to the same relative tolerance against references simulated at a DASSL
tolerance of 1e-13 (`test_dynamic_parity_at_the_reference_precision`, at the end of this
module).

The reference implementation is MBL v13.0.0 (commit 55abf579) simulated by OpenModelica
1.27.1 (`tests/data/modelica/NOTICE.md`); agreement with it is parity. Each model below is
read with `read_modelica`, run with `run.simulate` on the reference CSV's own time grid, and
every compared variable of the CSV is checked at EVERY row against

    |noodl - omc| <= 1e-6 |omc| + 1e-9   (kg/s),

i.e. 1e-6 relative, or 1e-9 kg/s absolute near zero flow. The tolerance is not
loosened for any model or row.

The algebraic set is the models with no `MixingVolume`/`DelayFirstOrder` (checked from each
JSON by `test_the_algebraic_set_has_no_volumes`): between two or more pressure boundaries the
flows are algebraic in the boundary signals, so noodl's steady solve at each grid time is the
same problem the DAE solver solves, and parity is to round-off, not to the solver tolerance.
`OneEffectiveAirLeakageArea` has two volumes and a mass source, so it is a dynamic model.

At an output time exactly on a signal event (`DoorOpenClosed`: the `Step` at 0.5 s;
`OpenDoorPressure`/`OpenDoorTemperature`: the `TimeTable` knots every 3600 s) OpenModelica's
recorded row holds the pre-event value; `signals` evaluates the left limit there (its module
docstring), and these models align row for row with that convention.

Column mapping (`ModelicaNames`): `<inst>.m_flow` is `names.edges[inst]` (an in-line sensor's
own name maps to the flow of the element in series with it); `<inst>.m1_flow`/`m2_flow`
(= `mAB_flow`/`mBA_flow`) of a door, plain or discretised (whose two edges are its port
streams, `MBLDoorPortStream`), are `names.edges["<inst>.port_a1"]`/`["<inst>.port_a2"]`. A
CSV column that maps to nothing fails the test: nothing is skipped.

The maximum relative error per model and variable is printed (visible with `-s`) and, when
the environment variable `NOODL_RECORD_PARITY=1` is set, written to the committed
`tests/data/modelica/parity-algebraic.json` (dynamic models: `parity-dynamic.json`) for the
documentation. Recording is opt-in and off by default: a plain checkout
or a CI run never rewrites those files, only a deliberate re-recording pass does.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica.run import extrapolate, simulate
from noodl.apps.building_physics.modelica.schema import ModelicaImportError

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "modelica"
RECORD = DATA / "parity-algebraic.json"

RTOL, ATOL = 1e-6, 1e-9  # never loosened
ALGEBRAIC = ("OneWayFlow", "DoorOpenClosed", "OpenDoorPressure", "OpenDoorTemperature",
             "Orifice", "PowerLaw")
VOLUME_CLASSES = ("Buildings.Fluid.MixingVolumes.MixingVolume",
                  "Buildings.Fluid.Delays.DelayFirstOrder")


def _csv(model: str) -> tuple[list[str], np.ndarray]:
    lines = [ln for ln in (DATA / f"{model}.csv").read_text().splitlines()
             if ln and not ln.startswith("#")]
    head = [h.strip('"') for h in lines[0].split(",")]
    data = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]])
    return head, data


def _record(model: str, stats: dict, record: Path = RECORD) -> None:
    # Deliberate side effect: the measured errors go into the committed
    # tests/data/modelica/parity-*.json for the documentation, only when
    # NOODL_RECORD_PARITY=1 is set -- a plain checkout or a CI run never
    # rewrites the recorded numbers.
    if os.environ.get("NOODL_RECORD_PARITY") != "1":
        return
    lock = record.with_suffix(".lock")
    for _ in range(6000):  # one writer at a time: models may be re-recorded in parallel
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
        doc[model] = stats
        record.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    finally:
        os.close(fd)
        os.remove(lock)


# Node columns `<zone or boundary>.<var>` -> the `simulate` history holding them.
_NODE_VARS = {"T": "T", "p": "p", "Xi[1]": "X_w"}


def _noodl_columns(model: str, head: list[str], data: np.ndarray,
                   run: dict | None = None) -> dict[str, np.ndarray]:
    """noodl's value of every CSV column; `run`, when given, receives the `simulate` history
    (`"out"`) and the reader's `"names"` for the mechanism checks."""
    net, state, drivers, names = read_modelica(DATA / f"{model}.json", return_names=True)
    times = torch.tensor(data[:, 0], dtype=torch.float64)
    assert names.times.numel() == times.numel()
    assert torch.allclose(names.times, times, rtol=0.0, atol=1e-9)
    out = simulate(net, state, drivers, times)
    if run is not None:
        run.update(out=out, names=names)
    q = out["air.q"].numpy()
    cols: dict[str, np.ndarray] = {}
    for h in head[1:]:
        inst, var = h.rsplit(".", 1)
        if inst in names.nodes and (var in _NODE_VARS or var.startswith("C[")):
            # A volume's or boundary's T (K), p (Pa), Xi[1] (water, kg/kg) and C[k] (trace
            # substance k = the species layer's k-th species; water, when carried, is last).
            i = names.nodes[inst]
            if var.startswith("C["):
                cols[h] = out["C"][:, i, int(var[2:-1]) - 1].numpy()
            else:
                cols[h] = out[_NODE_VARS[var]][:, i].numpy()
            continue
        # A two-way element's mAB_flow/mBA_flow are its m1_flow/m2_flow (MBL
        # `PartialFourPortInterface`: `mAB_flow = port_a1.m_flow`, `mBA_flow = port_a2.m_flow`).
        key = {"m_flow": inst, "m1_flow": f"{inst}.port_a1", "m2_flow": f"{inst}.port_a2",
               "mAB_flow": f"{inst}.port_a1", "mBA_flow": f"{inst}.port_a2"}.get(var)
        assert key in names.edges, f"{model}: CSV column {h!r} maps to no noodl variable"
        cols[h] = sum(s * q[:, c] for c, s in names.edges[key])
    return cols


def test_the_algebraic_set_has_no_volumes():
    for model in ALGEBRAIC:
        doc = json.loads((DATA / f"{model}.json").read_text())
        volumes = [c["name"] for c in doc["components"] if c["class"] in VOLUME_CLASSES]
        assert volumes == [], f"{model} has volumes {volumes}; it belongs to the dynamic set"


@pytest.mark.parametrize("model", ALGEBRAIC)
def test_parity_on_the_algebraic_models(model):
    head, data = _csv(model)
    cols = _noodl_columns(model, head, data)
    assert set(cols) == set(head[1:])
    stats, failures = {}, []
    for j, h in enumerate(head[1:], start=1):
        omc, got = data[:, j], cols[h]
        err = np.abs(got - omc)
        tol = RTOL * np.abs(omc) + ATOL
        nonzero = np.abs(omc) > ATOL
        rel = float((err[nonzero] / np.abs(omc[nonzero])).max()) if nonzero.any() else 0.0
        stats[h] = {"max_rel": rel, "max_abs": float(err.max()),
                    "max_err_over_tol": float((err / tol).max()), "rows": int(omc.size)}
        print(f"{model} {h}: max rel {rel:.3e}, max abs {err.max():.3e} kg/s "
              f"({omc.size} rows)")
        bad = np.nonzero(err > tol)[0]
        if bad.size:
            k = int(bad[0])
            failures.append(f"{h}: {bad.size} rows, first t = {data[k, 0]!r}: noodl "
                            f"{got[k]!r} vs omc {omc[k]!r}")
    _record(model, stats)
    assert not failures, f"{model} outside |d| <= 1e-6 |omc| + 1e-9:\n" + "\n".join(failures)


def _decimals(value: float) -> int:
    text = repr(float(value))
    assert "e" not in text
    return len(text.split(".")[1])


def test_one_way_flow_against_the_embedded_contam_table():
    """`Validation/OneWayFlow.mo` embeds CONTAM's steady results (`TestData`, read by the
    observer `contamData`, a `CombiTable1Dv`): for each tested element the mass flow at
    `dp = -50 .. 50` Pa, to three significant figures. The ramp `-50 + 100 t/500` hits every
    tabulated `dp` exactly on the 1 s output grid, so noodl is compared at the knots, with no
    interpolation of the table.

    CONTAM and MBL are separate implementations (different media: the model uses
    `Buildings.Media.Specialized.Air.PerfectGas`), and MBL itself departs from the table by
    more than its last digit for 41 of the 104 entries (worst 0.74 %, `tabDat_m_flow` at
    -40 Pa; measured). The assertion is therefore that noodl's departure from
    CONTAM is MBL's own, to the parity tolerance: `|noodl - contam| <= |omc - contam| +
    1e-6 |omc| + 1e-9` at every entry. The maximum difference and how many entries agree
    within half a unit of the table's last digit are printed and recorded.
    """
    doc = json.loads((DATA / "OneWayFlow.json").read_text())
    (comp,) = [c for c in doc["components"] if c["name"] == "contamData"]
    assert comp["class"] == "Modelica.Blocks.Tables.CombiTable1Dv"
    table = comp["parameters"]["table"]
    head, data = _csv("OneWayFlow")
    cols = _noodl_columns("OneWayFlow", head, data)
    # TestData's column order (OneWayFlow.mo, m_flow_data / "Headers" comment).
    order = ["ela", "ori", "pow_1dat", "pow_2dat", "pow_m_flow", "pow_V_flow",
             "tabDat_m_flow", "tabDat_V_flow"]
    t = data[:, 0]
    max_abs, max_rel, within, total, worst = 0.0, 0.0, 0, 0, ""
    for row in table:
        dp = row[0]
        (k,) = np.nonzero(np.abs(-50.0 + 100.0 * t / 500.0 - dp) < 1e-12)[0]
        for name, contam in zip(order, row[1:], strict=True):
            noodl = cols[f"{name}.m_flow"][k]
            omc = data[k, head.index(f"{name}.m_flow")]
            d = float(abs(noodl - contam))
            assert d <= abs(omc - contam) + RTOL * abs(omc) + ATOL, (name, dp)
            total += 1
            within += int(d <= 0.5 * 10.0 ** -_decimals(contam) * (1 + 1e-9))
            if contam != 0.0 and d / abs(contam) > max_rel:
                max_rel, worst = d / abs(contam), f"{name} at dp = {dp:g} Pa"
            max_abs = max(max_abs, d)
    print(f"OneWayFlow vs CONTAM: max |noodl - contam| {max_abs:.3e} kg/s, max relative "
          f"{max_rel:.3e} ({worst}); {within}/{total} entries within the table's last digit")
    _record("OneWayFlow:CONTAM", {"max_abs": max_abs, "max_rel": max_rel, "worst": worst,
                                  "within_last_digit": within, "entries": total})
    assert total == 13 * 8


REFUSED = {
    # Wind pressure and weather data (Outside_CpLowRise, ReaderTMY3).
    "PressurizationData": {"east", "weaDat", "west"},
    # A feedback controller and its temperature sensor, besides the wind boundaries.
    "TrickleVent": {"con", "east", "temSen", "weaDat", "west"},
    # Two orifices in one column chain (no junction node), plus the controller.
    "ChimneyShaftNoVolume": {"con", "oriChiBot", "oriChiTop", "temSen"},
    # A mass- and heat-storing MediumColumnDynamic, plus the controller.
    "ChimneyShaftWithVolume": {"con", "sha", "temSen"},
}


@pytest.mark.parametrize("model", sorted(REFUSED))
def test_refused_models_name_their_offending_instances(model):
    with pytest.raises(ModelicaImportError) as exc:
        read_modelica(DATA / f"{model}.json")
    message = str(exc.value)
    named = set(re.findall(r"^\s+- (\S+) \(", message, flags=re.MULTILINE))
    assert named == REFUSED[model], message
    assert f"refused {len(REFUSED[model])} item" in message


# ===================================================================== dynamic models
RECORD_DYNAMIC = DATA / "parity-dynamic.json"
# Relative errors are |noodl - omc| / max(|omc|, floor). The floor for a flow is FLOOR_FRAC
# of the model's largest |flow| over the bounded rows, one floor for all its flow columns:
# a door's directional flows and a stack's orifices pass through zero, and there the
# relative error would measure the absolute error of the solution divided by a number near
# zero, not the agreement of the flow. For the other variables the floor is FLOOR_FRAC of
# the column's own largest |value| (it only acts on a trace substance, which starts at 0;
# T, p and Xi are never near zero).
FLOOR_FRAC = 1e-3


def _kind(column: str) -> str:
    var = column.rsplit(".", 1)[1]
    return "flow" if var.endswith("_flow") else var.split("[")[0]


def _dynamic_stats(head: list[str], data: np.ndarray, cols: dict[str, np.ndarray],
                   resolution: dict[str, float] | None = None) -> dict[str, dict]:
    """Per column: the worst relative and absolute error over every row after the first
    (t = StartTime, MBL's `p_start` initialisation, reported separately), the time of the
    worst relative error, the t = StartTime row's errors, and the worst relative error beyond
    the reference's resolution (`resolution[kind]`, absolute, REFERENCE_RESOLUTION): the
    part of the difference the reference resolves, which the tolerance applies to."""
    flows = [j for j, h in enumerate(head) if j and _kind(h) == "flow"]
    flow_floor = FLOOR_FRAC * float(np.abs(data[1:, flows]).max()) if flows else 0.0
    stats = {}
    for j, h in enumerate(head[1:], start=1):
        omc, got = data[:, j], np.asarray(cols[h], dtype=float)
        err = np.abs(got - omc)
        floor = flow_floor if _kind(h) == "flow" else FLOOR_FRAC * float(np.abs(omc[1:]).max())
        den = np.maximum(np.abs(omc), floor)
        rel = np.divide(err, den, out=np.zeros_like(err), where=den > 0)
        res = (resolution or {}).get(_kind(h), 0.0)
        beyond = np.divide(np.maximum(err - res, 0.0), den, out=np.zeros_like(err),
                           where=den > 0)
        k = int(np.argmax(rel[1:])) + 1
        kb = int(np.argmax(beyond[1:])) + 1
        stats[h] = {"kind": _kind(h), "max_rel": float(rel[1:].max()),
                    "max_abs": float(err[1:].max()), "t_max_rel": float(data[k, 0]),
                    "floor": floor, "t0_rel": float(rel[0]), "t0_abs": float(err[0]),
                    "rows": int(omc.size - 1), "resolution": res,
                    "max_rel_beyond_resolution": float(beyond[1:].max()),
                    "t_max_rel_beyond_resolution": float(data[kb, 0])}
    return stats


DYNAMIC = ("OpenDoorBuoyancyDynamic", "OpenDoorBuoyancyPressureDynamic", "ThreeRoomsContam",
           "ThreeRoomsContamDiscretizedDoor", "CO2TransportStep", "ClosedDoors",
           "NaturalVentilation", "OneOpenDoor", "OneRoom", "ReverseBuoyancy",
           "ReverseBuoyancy3Zones", "ZonalFlow", "OneEffectiveAirLeakageArea")
# The models whose dynamics MBL's volume mass storage dominates (a closed heated building, a
# start far from the airflow's balance, a source into sealed volumes): besides parity, each
# asserts that noodl reproduces the storage mechanism (`_check_storage_mechanism`).
STORAGE_DOMINATED = ("ClosedDoors", "OneOpenDoor", "ReverseBuoyancy",
                     "OneEffectiveAirLeakageArea")
UNITS = {"flow": "kg/s", "T": "K", "p": "Pa", "Xi": "kg/kg", "C": "kg/kg"}


def test_every_fixture_is_parity_checked_or_refused():
    # "parity-algebraic"/"parity-dynamic" are the committed recorded-error files this
    # module writes (see RECORD/RECORD_DYNAMIC, NOODL_RECORD_PARITY), not model fixtures.
    fixtures = {p.stem for p in DATA.glob("*.json")} - {"parity-algebraic", "parity-dynamic"}
    sets = (set(ALGEBRAIC), set(DYNAMIC), set(REFUSED))
    assert set().union(*sets) == fixtures
    assert sum(len(s) for s in sets) == len(fixtures)  # disjoint
    assert set(PARITY_ROWS) == set(DYNAMIC) == set(REFERENCE_RESOLUTION)
    assert set(PARITY_STORAGE) <= set(DYNAMIC) and set(PARITY_SUBSTEPS) <= set(DYNAMIC)
    assert set(DEFAULT_SUBSTEPS) <= set(DYNAMIC)
    assert set(STORAGE_DOMINATED) <= set(DYNAMIC)


# ------------------------------------------------ parity at the reference's precision
# Every dynamic model. The reference CSVs are simulated at a DASSL tolerance of 1e-13
# (`scripts/modelica_export.py --tolerance 1e-13`, `tests/data/modelica/NOTICE.md`; at 1e-14
# DASSL returns no trajectory): at the declared 1e-6 the reference's own error was of the
# order of the parity tolerance (ZonalFlow: 5.6e-5 K and 2.8e-8 kg/kg; CO2TransportStep's
# door flow 5e-5 relative after its pulse, still there at 1e-10). noodl reads each model as it
# is, with its volumes' mass storage (`modelica/storage.py`: every volume starts at MBL's
# `p_start` and stores what the airflow does not balance, with MBL's own balance forms), runs
# `scheme="midpoint"` (second order; the volumes' mass by the L-stable BDF2 rate, from a
# graded start, `run.GRADING_WINDOW`, or a smooth one when the start is in balance; every
# signal event and every switch of an element law or a flow's sign on a step boundary) at
# two step sizes in ratio 2 (PARITY_SUBSTEPS), combined by `run.extrapolate`, and must agree
# with every column at every row after t0 to PARITY_RTOL, relative with the FLOOR_FRAC floor
# of `_dynamic_stats`: 1e-6, the models' declared solver tolerance and the algebraic parity
# tolerance, beyond the reference's own resolution (REFERENCE_RESOLUTION). The t0 row (MBL's
# `p_start` initialisation, reproduced) is printed and recorded. The size of the
# extrapolation's correction (`own error` in the output, the finer run's own
# time-integration error) is printed and recorded.
PARITY_RTOL = 1e-6
# The reference's resolution, absolute, per model and variable kind: the largest change of
# any column of that kind, over the rows after t0, from the references simulated at a DASSL
# tolerance of 1e-12 (the CSVs of commit d6ea26a) to these at 1e-13, rounded up to two
# digits. A difference the reference itself moves by when its tolerance is tightened
# tenfold is not resolved by it, so the tolerance applies to the difference beyond this.
# It decides only where a model's relative error alone is above PARITY_RTOL (measured, at
# the full runs): OneOpenDoor's door flows at 28.8 s, 1.2e-8 kg/s from the reference where
# noodl's runs at 1, 2 and 4 substeps agree to 1e-10 kg/s and the reference itself moved by
# up to 2.9e-8 kg/s over its first rows; NaturalVentilation's orifices at their reversal,
# 5.0e-10 kg/s (unchanged from 1-2 to 2-4 substeps) where the reference moved by 1.8e-10 to
# 3.7e-9 kg/s from row to row there; ReverseBuoyancy's door flow at 612 s, 4.7e-9 kg/s,
# where the reference moved by 5.1e-9 kg/s.
REFERENCE_RESOLUTION: dict[str, dict[str, float]] = {
    "ClosedDoors": {"T": 1.2e-09, "Xi": 2.2e-14, "flow": 1.3e-11, "p": 1.5e-07},
    "CO2TransportStep": {"C": 6.0e-14, "T": 1.5e-12, "Xi": 5.3e-14, "flow": 3.3e-08,
                         "p": 1.3e-09},
    "NaturalVentilation": {"T": 2.8e-10, "Xi": 1.2e-15, "flow": 4.9e-09, "p": 1.1e-08},
    "OneEffectiveAirLeakageArea": {"T": 3.4e-09, "Xi": 8.8e-15, "flow": 3.1e-11,
                                   "p": 9.6e-08},
    "OneOpenDoor": {"T": 1.6e-08, "flow": 4.5e-08, "p": 3.6e-06},
    "OneRoom": {"T": 1.9e-12, "Xi": 1.8e-18, "flow": 1.9e-13, "p": 1.5e-11},
    "OpenDoorBuoyancyDynamic": {"T": 1.1e-08, "Xi": 1.5e-16, "flow": 1.5e-08, "p": 1.4e-09},
    "OpenDoorBuoyancyPressureDynamic": {"T": 7.1e-09, "Xi": 8.3e-15, "flow": 2.9e-09,
                                        "p": 1.9e-10},
    "ReverseBuoyancy": {"T": 2.8e-10, "Xi": 1.5e-14, "flow": 2.2e-08, "p": 8.6e-09},
    "ReverseBuoyancy3Zones": {"T": 1.2e-09, "Xi": 5.1e-16, "flow": 1.8e-08, "p": 4.3e-09},
    "ThreeRoomsContam": {"C": 0.0, "T": 2.3e-13, "Xi": 3.5e-14, "flow": 4.7e-09,
                         "p": 1.4e-10},
    "ThreeRoomsContamDiscretizedDoor": {"C": 0.0, "T": 2.3e-13, "Xi": 1.1e-13,
                                        "flow": 2.9e-10, "p": 1.2e-10},
    "ZonalFlow": {"T": 3.2e-10, "Xi": 1.6e-13, "flow": 0.0, "p": 0.0},
}
# The substeps per output interval of the two runs `run.extrapolate` combines: (1, 2), or
# (2, 4) where one step per output interval is too long for the extrapolation to be
# asymptotic (measured, worst flow error at (1, 2) and (2, 4) substeps): ClosedDoors'
# 14.4 s steps across its crack flows switching on (3.5e-10 and 1.2e-12 kg/s at 993.6 s,
# against a 1.3e-11 kg/s resolution), ReverseBuoyancy's 7.2 s steps across its orifices'
# reversal at 18 s (8.4e-8 and 1.6e-8 kg/s at 21.6 s).
PARITY_SUBSTEPS: dict[str, tuple[int, int]] = {"ClosedDoors": (2, 4),
                                               "ReverseBuoyancy": (2, 4)}
# Bounds above PARITY_RTOL (beyond the resolution), each with its demonstrated cause: none.
PARITY_STORAGE: dict[str, dict[str, float]] = {}
# The default run compares the first rows of every model (`-m slow` compares every row),
# the models in parallel (`default_windows`): where the start releases an imbalance, its
# first output intervals, most of whose cost is the graded start (`run.GRADING_WINDOW`);
# where it is in balance, a stretch past its smooth start. ReverseBuoyancy's window ends
# before its orifices reverse (18 s), the rows that need (2, 4) substeps (PARITY_SUBSTEPS),
# and runs at (1, 2) (DEFAULT_SUBSTEPS), a third of the cost.
PARITY_ROWS = {"ClosedDoors": 3, "CO2TransportStep": 2, "NaturalVentilation": 3,
               "OneEffectiveAirLeakageArea": 8, "OneOpenDoor": 4, "OneRoom": 21,
               "OpenDoorBuoyancyDynamic": 2, "OpenDoorBuoyancyPressureDynamic": 2,
               "ReverseBuoyancy": 2, "ReverseBuoyancy3Zones": 2, "ThreeRoomsContam": 3,
               "ThreeRoomsContamDiscretizedDoor": 3, "ZonalFlow": 11}
DEFAULT_SUBSTEPS: dict[str, tuple[int, int]] = {"ReverseBuoyancy": (1, 2)}


def _columns_of(out: dict, names, head: list[str], net, drivers) -> dict[str, np.ndarray]:
    q = out["air.q"].numpy()
    cols: dict[str, np.ndarray] = {}
    for h in head[1:]:
        inst, var = h.rsplit(".", 1)
        if inst in names.nodes and (var in _NODE_VARS or var.startswith("C[")):
            i = names.nodes[inst]
            cols[h] = (out["C"][:, i, int(var[2:-1]) - 1].numpy() if var.startswith("C[")
                       else out[_NODE_VARS[var]][:, i].numpy())
            continue
        key = {"m_flow": inst, "m1_flow": f"{inst}.port_a1", "m2_flow": f"{inst}.port_a2",
               "mAB_flow": f"{inst}.port_a1", "mBA_flow": f"{inst}.port_a2"}.get(var)
        assert key in names.edges, f"CSV column {h!r} maps to no noodl variable"
        cols[h] = sum(s * q[:, c] for c, s in names.edges[key])
    return cols


def _worst(stats: dict[str, dict]) -> dict[str, dict]:
    worst: dict[str, dict] = {}
    for h, s in stats.items():
        w = worst.setdefault(s["kind"], {"max_rel": -1.0, "beyond": -1.0})
        if s["max_rel"] > w["max_rel"]:
            w.update(max_rel=s["max_rel"], column=h, t=s["t_max_rel"], max_abs=s["max_abs"])
        if s.get("max_rel_beyond_resolution", s["max_rel"]) > w["beyond"]:
            w.update(beyond=s.get("max_rel_beyond_resolution", s["max_rel"]),
                     beyond_column=h,
                     beyond_t=s.get("t_max_rel_beyond_resolution", s["t_max_rel"]))
        w["t0_rel"] = max(w.get("t0_rel", 0.0), s["t0_rel"])
    return worst


def _check_parity(model: str, rows: int | None, threads: int | None = None,
                  substeps: tuple[int, int] | None = None) -> list[str]:
    """Run `model` over its first `rows` rows (every row: `None`, recorded) at `substeps`
    (default PARITY_SUBSTEPS) and return the failures against its tolerance (none: parity).
    `threads` sets torch's thread count (a worker of the default run's pool)."""
    if threads is not None:
        torch.set_num_threads(threads)
    head, data = _csv(model)
    data = data if rows is None else data[:rows]
    start = time.perf_counter()
    runs, cols = [], []
    substeps = substeps or PARITY_SUBSTEPS.get(model, (1, 2))
    for r in substeps:
        net, state, drivers, names = read_modelica(DATA / f"{model}.json", return_names=True,
                                                   substeps=r)
        times = names.times[:data.shape[0]]
        assert torch.allclose(times, torch.tensor(data[:, 0], dtype=torch.float64),
                              rtol=0.0, atol=1e-9)
        runs.append(simulate(net, state, drivers, times, scheme="midpoint"))
        cols.append(_columns_of(runs[-1], names, head, net, drivers))
    best = {h: cols[1][h] + (cols[1][h] - cols[0][h]) / 3.0 for h in head[1:]}  # extrapolate
    seconds = time.perf_counter() - start
    stats = _dynamic_stats(head, data, best, REFERENCE_RESOLUTION[model])
    own = _dynamic_stats(head, np.column_stack([data[:, 0]] + [best[h] for h in head[1:]]),
                         cols[1])
    worst, own_worst = _worst(stats), _worst(own)
    for kind, w in worst.items():
        print(f"{model} {kind}: max rel {w['max_rel']:.3e} ({w['column']} at t = {w['t']:g}), "
              f"max abs {w['max_abs']:.3e} {UNITS[kind]}; beyond the reference's resolution "
              f"{w['beyond']:.3e}; t0 row {w['t0_rel']:.1e}; own error "
              f"{own_worst[kind]['max_rel']:.3e}")
    print(f"{model}: {data.shape[0]} rows at {substeps} substeps in {seconds:.1f} s")
    mechanism = None
    if model in STORAGE_DOMINATED:
        mechanism = _check_storage_mechanism(
            model, head, data, {"names": names, "out": extrapolate(runs[0], runs[1])})
    if rows is None:
        _record(model, {"group": "storage" if model in STORAGE_DOMINATED else "parity",
                        "columns": stats, "worst": worst, "own_error": own_worst,
                        "rtol": PARITY_RTOL, "storage_bounds": PARITY_STORAGE.get(model, {}),
                        "floor_frac": FLOOR_FRAC, "mechanism": mechanism,
                        "reference_resolution": REFERENCE_RESOLUTION[model],
                        "substeps": list(substeps),
                        "reference_tolerance": 1e-13, "seconds": seconds}, RECORD_DYNAMIC)
    bound = {kind: PARITY_STORAGE.get(model, {}).get(kind, PARITY_RTOL) for kind in worst}
    failures = [f"{kind}: max rel beyond the reference's resolution {w['beyond']:.3e} "
                f"({w['beyond_column']} at t = {w['beyond_t']:g}; max rel {w['max_rel']:.3e}) > "
                f"{bound[kind]:.1e}" for kind, w in worst.items()
                if not w["beyond"] <= bound[kind]]
    return [f"{model} outside its tolerance:", *failures] if failures else []


@pytest.fixture(scope="module")
def default_windows() -> dict[str, list[str]]:
    """`_check_parity` on the first PARITY_ROWS rows of every dynamic model, the models in
    parallel worker processes (most of each window's time is its graded start, `run`
    module), so that the default run takes about as long as its slowest window."""
    models = sorted(PARITY_ROWS)
    cpus = os.cpu_count() or 1
    workers = max(1, min(len(models), cpus))
    threads = max(1, cpus // workers)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=context) as pool:
        futures = {m: pool.submit(_check_parity, m, PARITY_ROWS[m], threads,
                                  DEFAULT_SUBSTEPS.get(m)) for m in models}
        return {m: f.result() for m, f in futures.items()}


@pytest.mark.parametrize("model", sorted(PARITY_ROWS))
def test_dynamic_parity_at_the_reference_precision(model, default_windows):
    """The first `PARITY_ROWS[model]` rows of every dynamic model within PARITY_RTOL
    (see the block comment above PARITY_RTOL)."""
    failures = default_windows[model]
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.parametrize("model", DYNAMIC)
def test_dynamic_parity_at_the_reference_precision_every_row(model):
    """As `test_dynamic_parity_at_the_reference_precision`, over the whole experiment."""
    failures = _check_parity(model, None)
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.parametrize(("scheme", "order"), [("implicit", 1), ("midpoint", 2)])
def test_time_integration_error_falls_with_the_order_of_the_scheme(scheme, order):
    """ZonalFlow, without extrapolation: the error of `rooB.T` against the reference over
    the rows after its start falls by 2**order from 1 to 2 substeps per output interval:
    first order for `"implicit"` (flows at the step's end), second for `"midpoint"`, whose
    symmetric step is what `extrapolate` relies on. The ratio must be within 10 % of
    2**order (measured: 1.97 and 4.00). `rooB.T` is the column with the largest
    time-integration error; the zonal flows are prescribed, and the pressures sit at the
    reference's own resolution. ZonalFlow starts in balance (`run._balanced`), so its
    steps reach the grid step within the first two output intervals (steps doubling up
    from `run.GRADING_H0`)."""
    model, rows, column = "ZonalFlow", 20, "rooB.T"
    head, data = _csv(model)
    data = data[:rows]
    first = 3  # the first row past the smooth start
    errors = []
    for r in (1, 2):
        net, state, drivers, names = read_modelica(DATA / f"{model}.json", return_names=True,
                                                   substeps=r)
        out = simulate(net, state, drivers, names.times[:rows], scheme=scheme)
        cols = _columns_of(out, names, head, net, drivers)
        errors.append(float(np.abs(cols[column] - data[:, head.index(column)])[first:].max()))
    ratio = errors[0] / errors[1]
    print(f"{model} {column}, {scheme}: error {errors[0]:.3e} K at 1 substep, {errors[1]:.3e} "
          f"K at 2; ratio {ratio:.3f} (2**{order} = {2 ** order})")
    assert abs(ratio / 2 ** order - 1.0) <= 0.1, ratio


# ------------------------------------------------------- storage-dominated: the mechanism
def _volumes(doc: dict) -> dict[str, dict]:
    return {c["name"]: c["parameters"] for c in doc["components"]
            if c["class"] in VOLUME_CLASSES}


def _check_storage_mechanism(model: str, head: list[str], data: np.ndarray,
                             run: dict) -> dict:
    """Assert that noodl reproduces the storage mechanism of a storage-dominated model and
    return its numbers for the record."""
    doc = json.loads((DATA / f"{model}.json").read_text())
    if model == "ReverseBuoyancy":
        return _check_initial_imbalance(doc, head, data, run)
    if model == "OneEffectiveAirLeakageArea":
        return _check_injected_mass(doc, head, data, run)
    return _check_closed_heated_rooms(doc, head, data, run)


def _check_injected_mass(doc: dict, head: list[str], data: np.ndarray, run: dict) -> dict:
    """OneEffectiveAirLeakageArea: a `MassFlowSource_T` ramping to 0.01 kg/s (`Ramp`,
    1800-5400 s) feeds one of two volumes that exchange air only with each other through a
    crack, so every kilogram injected is stored by compression (the quasi-steady route had to
    refuse the model). `Buildings.Media.Air`'s mass is `V p dStp/pStp` (`Air.mo:210-215`), so
    `sum V dp dStp/pStp = int m_flow`: asserted to 1e-6 of the final injected mass for both
    (measured 1.3e-13 for MBL, 3.5e-14 for noodl, whose storage rate balances the ramp's
    point values and restarts at its corners, `run._Midpoint`)."""
    (ramp,) = [s["parameters"] for s in doc["signals"] if s["class"].endswith("Ramp")]
    t = data[:, 0]
    a, d, h = ramp["startTime"], ramp["duration"], ramp["height"]
    tau = np.clip(t - a, 0.0, d)
    injected = h * tau ** 2 / (2 * d) + h * np.maximum(t - a - d, 0.0)   # int of the ramp
    vols = _volumes(doc)
    names, out = run["names"], run["out"]
    p_n = out["p"].numpy()
    stored = {
        "mbl": sum(v["V"] * 1.2 / 101325.0
                   * (data[:, head.index(f"{z}.p")] - data[0, head.index(f"{z}.p")])
                   for z, v in vols.items()),
        "noodl": sum(v["V"] * 1.2 / 101325.0 * (p_n[:, names.nodes[z]] - p_n[0, names.nodes[z]])
                     for z, v in vols.items()),
    }
    scale = max(float(injected[-1]), 1.0)  # kg (none injected in the default rows)
    off = {k: float(np.abs(v - injected).max() / scale) for k, v in stored.items()}
    print(f"mechanism: stored vs injected mass off by {off['noodl']:.2e} (noodl), "
          f"{off['mbl']:.2e} (MBL) of the {injected[-1]:.1f} kg injected")
    assert off["noodl"] <= 1e-6 and off["mbl"] <= 1e-6
    return {"stored_vs_injected": off, "injected_kg": float(injected[-1])}


def _check_closed_heated_rooms(doc: dict, head: list[str], data: np.ndarray,
                               run: dict) -> dict:
    """ClosedDoors, OneOpenDoor: rooms of an ideal gas (PerfectGas, SimpleAir) with no
    boundary, heated by `Q = k A sin(2 pi f t)` (a `Sine` through a `Gain`).

    MBL (`ConservationEquation.mo`: `m = V medium.d`, `U = m medium.u`, `der(U) = Hb_flow +
    Q_flow`; `u = h - R T` for both ideal gases) heats the closed building at constant
    volume: with `p V = m R T` and `U = m cv T`, `sum V dp = (R/cv) int Q` whatever the door
    flows, and the zone temperatures rise cp/cv times faster than at constant pressure (the
    quasi-steady route, `mass_storage=False`, rose 1.40 times too slowly). Asserted, on
    noodl's own history: the same pressure balance to 1e-3 of its peak (as MBL's, measured
    3.7e-5 and 3.5e-8; noodl: the same), and the ratio of the V-weighted temperature rises
    MBL/noodl at the peak of `int Q` equal to 1 to 1e-4 (not cp/cv). `int Q` is the closed
    form of the JSON's signals."""
    from noodl.elements import medium as mbl_medium

    med = mbl_medium(doc["medium"]["class"])
    sine, gain = (next(s for s in doc["signals"] if s["class"].endswith(c))
                  for c in ("Sources.Sine", "Math.Gain"))
    assert sine["drives"] == f"{gain['name']}.u"
    ps = sine["parameters"]
    assert ps.get("phase", 0.0) == 0.0 and ps.get("offset", 0.0) == 0.0
    assert ps.get("startTime", 0.0) == 0.0
    t = data[:, 0]
    w = 2 * np.pi * ps["f"]
    int_q = gain["parameters"]["k"] * ps["amplitude"] * (1.0 - np.cos(w * t)) / w  # J
    vols = _volumes(doc)
    X = med.X_default[0] if med.has_moisture else 0.0
    first = next(iter(vols.values()))
    T0, p0 = float(first["T_start"]), float(first["p_start"])
    R = p0 / (float(med.density(torch.tensor(p0), T0, X)) * T0)  # the medium's p/(rho T)
    cp = med.specific_heat_cp(X)
    cv = cp - R
    names, out = run["names"], run["out"]
    p_n, T_n = out["p"].numpy(), out["T"].numpy()
    predicted = R / cv * int_q
    sum_vdp = {
        "mbl": sum(v["V"] * (data[:, head.index(f"{z}.p")] - data[0, head.index(f"{z}.p")])
                   for z, v in vols.items()),
        "noodl": sum(v["V"] * (p_n[:, names.nodes[z]] - p_n[0, names.nodes[z]])
                     for z, v in vols.items()),
    }
    balance = {key: float(np.abs(v - predicted).max() / np.abs(predicted).max())
               for key, v in sum_vdp.items()}
    k = int(np.argmax(np.abs(int_q)))
    rise_mbl = sum(v["V"] * (data[k, head.index(f"{z}.T")] - data[0, head.index(f"{z}.T")])
                   for z, v in vols.items())
    rise_noodl = sum(v["V"] * (T_n[k, names.nodes[z]] - T_n[0, names.nodes[z]])
                     for z, v in vols.items())
    ratio = float(rise_mbl / rise_noodl)
    print(f"mechanism: sum V dp vs (R/cv) int Q off by {balance['noodl']:.2e} of the peak "
          f"(MBL {balance['mbl']:.2e}); temperature-rise ratio MBL/noodl {ratio:.7f} "
          f"(cp/cv {cp / cv:.5f})")
    assert balance["noodl"] <= 1e-3 and balance["mbl"] <= 1e-3
    assert abs(ratio - 1.0) <= 1e-4
    return {"sum_V_dp_vs_R_over_cv_int_Q": balance, "rise_ratio_mbl_over_noodl": ratio,
            "cp_over_cv": cp / cv}


def _check_initial_imbalance(doc: dict, head: list[str], data: np.ndarray,
                             run: dict) -> dict:
    """ReverseBuoyancy: every zone starts at `p_start` = 101325 Pa against the outside
    boundary at 100000 Pa, in MBL (`FixedInitial`, `initialize_p`) and in noodl, whose
    t = StartTime row must hold the same pressures to 1e-9 Pa. Releasing the excess through
    storage cools the zones: the mass-weighted cooling by t = 21.6 s is the flow work
    `(pStp/dStp) ln(p0/p)` less the latent enthalpy of MBL's Xi drop (`Air.mo`
    `u = h - pStp/dStp`; `ConservationEquation.mo` `der(Xi) = mbXi_flow/m`), 0.830 K in MBL
    and none in the quasi-steady route, which started balanced. Asserted: noodl's cooling
    within 1e-4 of MBL's (measured 4e-6: 0.83010 K both)."""
    (bou,) = [c["parameters"] for c in doc["components"]
              if c["class"].endswith("Boundary_pT")]
    vols = _volumes(doc)
    names, out = run["names"], run["out"]
    p_n, T_n = out["p"].numpy(), out["T"].numpy()
    assert bou["p"] == 100000.0
    for z, v in vols.items():
        assert data[0, head.index(f"{z}.p")] == v["p_start"] == 101325.0
        assert abs(p_n[0, names.nodes[z]] - v["p_start"]) <= 1e-9
    k = int(np.argmin(np.abs(data[:, 0] - 21.6)))
    mass = cool_mbl = cool_noodl = 0.0
    for z, v in vols.items():
        m = v["V"] * 1.2 * v["p_start"] / 101325.0
        mass += m
        cool_mbl += m * (data[0, head.index(f"{z}.T")] - data[k, head.index(f"{z}.T")])
        cool_noodl += m * (T_n[0, names.nodes[z]] - T_n[k, names.nodes[z]])
    cool_mbl, cool_noodl = float(cool_mbl / mass), float(cool_noodl / mass)
    print(f"mechanism: mass-weighted cooling by t = {data[k, 0]:g} s: MBL {cool_mbl:.4f} K, "
          f"noodl {cool_noodl:.4f} K")
    assert abs(cool_noodl / cool_mbl - 1.0) <= 1e-4
    return {"t0_p_equals_p_start": True, "cooling_mbl_K": cool_mbl,
            "cooling_noodl_K": cool_noodl, "t": float(data[k, 0])}



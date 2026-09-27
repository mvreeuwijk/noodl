"""Parity with the Modelica Buildings Library on its multizone models.

The algebraic models are checked to a fixed tolerance (below); the dynamic models,
those with volumes, against per-model bounds set from measurement (`DYNAMIC_BOUNDS` and
`test_parity_on_the_dynamic_models`, at the end of this module), except the storage-dominated
ones, checked against the reference's own tolerance plus one output interval's change
(`_storage_tolerance`).

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
`OneEffectiveAirLeakageArea` has two volumes and a mass source (a dynamic, storage model), so
it is not in this set.

At an output time exactly on a signal event (`DoorOpenClosed`: the `Step` at 0.5 s;
`OpenDoorPressure`/`OpenDoorTemperature`: the `TimeTable` knots every 3600 s) OpenModelica's
recorded row holds the pre-event value; `signals` evaluates the left limit there (its module
docstring), and these models align row for row with that convention.

Column mapping (`ModelicaNames`): `<inst>.m_flow` is `names.edges[inst]` (an in-line sensor's
own name maps to the flow of the element in series with it); `<inst>.m1_flow`/`m2_flow` of
a `DoorOpen`/`DoorOperable` are `names.edges["<inst>.port_a1"]`/`["<inst>.port_a2"]`. A
discretised door's port flows are not a signed sum of its compartment edge flows (its
`smoothHeaviside` split), so its `m1_flow = mAB_flow` and `m2_flow = mBA_flow` are recomputed
exactly from the solved pressures by the door element's own `port_flows`. A CSV column that
maps to nothing fails the test: nothing is skipped.

The maximum relative error per model and variable is printed (visible with `-s`) and, when
the environment variable `NOODL_RECORD_PARITY=1` is set, written to the committed
`tests/data/modelica/parity-algebraic.json` (dynamic models: `parity-dynamic.json`) for the
documentation. Recording is opt-in and off by default: a plain checkout
or a CI run never rewrites those files, only a deliberate re-recording pass does.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica.run import simulate, step_drivers
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
    try:
        doc = json.loads(record.read_text()) if record.exists() else {}
    except json.JSONDecodeError:
        doc = {}
    doc[model] = stats
    record.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def _discretised_port_flows(model, drivers, names, out, inst: str) -> tuple[np.ndarray, ...]:
    """`(port_a1.m_flow, port_a2.m_flow)` of discretised door `inst` at every row of `out`,
    from the door element's `port_flows` at the solved pressures and the closure's node
    states (`p_abs`, `T`, `X_w`) of that row."""
    (kind,) = names.kinds[inst]
    layer = model.potential["air"]
    el, sl = layer.element_for(kind)
    grid = drivers["series:time"]
    m1, m2 = [], []
    n = out["p"].shape[-1]
    for k in range(out["time"].numel()):
        d = step_drivers(drivers, grid, float(out["time"][k]))
        d.update({"p_abs": out["p"][k], "T": out["T"][k],
                  "X_w": torch.as_tensor(out["X_w"][k], dtype=torch.float64).expand(n)})
        dp = layer.dp(out["air.phi"][k], d)[..., sl]
        a, b = el.port_flows(dp, d)
        m1.append(float(a))
        m2.append(float(b))
    return np.array(m1), np.array(m2)


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
    doors: dict[str, tuple[np.ndarray, ...]] = {}
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
        kinds = names.kinds.get(inst, ())
        if kinds and kinds[0].startswith("door_c:") and var in (
                "m1_flow", "m2_flow", "mAB_flow", "mBA_flow"):
            if inst not in doors:
                doors[inst] = _discretised_port_flows(net, drivers, names, out, inst)
            cols[h] = doors[inst][0 if var in ("m1_flow", "mAB_flow") else 1]
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


def _dynamic_stats(head: list[str], data: np.ndarray,
                   cols: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per column: the worst relative and absolute error over every row after the first
    (t = StartTime, reported separately: MBL's volumes re-balance their pressures through
    mass storage at initialisation, which noodl's quasi-steady airflow does not model), the
    time of the worst relative error, and the t = StartTime row's errors."""
    flows = [j for j, h in enumerate(head) if j and _kind(h) == "flow"]
    flow_floor = FLOOR_FRAC * float(np.abs(data[1:, flows]).max()) if flows else 0.0
    stats = {}
    for j, h in enumerate(head[1:], start=1):
        omc, got = data[:, j], np.asarray(cols[h], dtype=float)
        err = np.abs(got - omc)
        floor = flow_floor if _kind(h) == "flow" else FLOOR_FRAC * float(np.abs(omc[1:]).max())
        den = np.maximum(np.abs(omc), floor)
        rel = np.divide(err, den, out=np.zeros_like(err), where=den > 0)
        k = int(np.argmax(rel[1:])) + 1
        stats[h] = {"kind": _kind(h), "max_rel": float(rel[1:].max()),
                    "max_abs": float(err[1:].max()), "t_max_rel": float(data[k, 0]),
                    "floor": floor, "t0_rel": float(rel[0]), "t0_abs": float(err[0]),
                    "rows": int(omc.size - 1)}
    return stats


DYNAMIC = ("OpenDoorBuoyancyDynamic", "OpenDoorBuoyancyPressureDynamic", "ThreeRoomsContam",
           "ThreeRoomsContamDiscretizedDoor", "CO2TransportStep", "ClosedDoors",
           "NaturalVentilation", "OneOpenDoor", "OneRoom", "ReverseBuoyancy",
           "ReverseBuoyancy3Zones", "ZonalFlow", "OneEffectiveAirLeakageArea")
# Measured wall time above 30 s (38-315 s): `slow`, deselected by the
# default `-m "not slow"` addopts, run with `-m slow`. The other three took 25-55 s
# depending on machine load and stay in the default run, so that it checks one model of
# each kind: trace source (CO2TransportStep), stack (OneRoom), zonal flows (ZonalFlow).
SLOW_DYNAMIC = frozenset(DYNAMIC) - {"CO2TransportStep", "OneRoom", "ZonalFlow"}

# Three groups. "parity": noodl agrees to the reference's
# tolerance (CO2TransportStep outside the rows right after its source pulse). "step": the
# difference is noodl's first-order step at the CSV interval (halving the step halves it;
# asserted on OpenDoorBuoyancyDynamic by `test_step_limited_error_halves_with_the_step`).
# "storage": dominated by MBL's volume mass storage, which the reader models
# (`modelica/storage.py`); each is checked against `_storage_tolerance`, not a measured bound,
# and its test also asserts that noodl reproduces the storage mechanism
# (`_check_storage_mechanism`).
DYNAMIC_GROUP = {
    "ThreeRoomsContam": "parity", "ThreeRoomsContamDiscretizedDoor": "parity",
    "OneRoom": "parity", "ZonalFlow": "parity", "CO2TransportStep": "parity",
    "OpenDoorBuoyancyDynamic": "step", "OpenDoorBuoyancyPressureDynamic": "step",
    "NaturalVentilation": "step", "ReverseBuoyancy3Zones": "step",
    "ClosedDoors": "storage", "OneOpenDoor": "storage", "ReverseBuoyancy": "storage",
    "OneEffectiveAirLeakageArea": "storage",
}
UNITS = {"flow": "kg/s", "T": "K", "p": "Pa", "Xi": "kg/kg", "C": "kg/kg"}

# The bound on the worst relative error (FLOOR_FRAC floor) over every row after
# t = StartTime, per model and variable: 1.25 times the measured value, rounded up to two
# digits, 1e-12 where the measurement is round-off. Set once
# from that measurement and justified per model in the test docstring; never loosened.
DYNAMIC_BOUNDS = {
    "OpenDoorBuoyancyDynamic": {"flow": 3.1e-02, "T": 4.4e-05, "p": 2.6e-09, "Xi": 2.5e-07},
    "OpenDoorBuoyancyPressureDynamic": {"flow": 3.0e-02, "T": 4.8e-05, "p": 2.5e-09,
                                        "Xi": 5.8e-05},
    "ThreeRoomsContam": {"flow": 2.7e-06, "T": 8.9e-09, "p": 4.6e-10, "Xi": 4.4e-04,
                         "C": 1.0e-12},
    "ThreeRoomsContamDiscretizedDoor": {"flow": 3.0e-06, "T": 8.8e-09, "p": 4.6e-10,
                                        "Xi": 4.4e-04, "C": 1.0e-12},
    "CO2TransportStep": {"flow": 1.7e-05, "T": 8.9e-09, "p": 6.2e-10, "Xi": 4.4e-04,
                         "C": 2.2e+00},
    "NaturalVentilation": {"flow": 3.6e-01, "T": 4.7e-06, "p": 1.5e-06, "Xi": 1.5e-06},
    "OneRoom": {"flow": 2.0e-11, "T": 1.0e-12, "p": 1.0e-12, "Xi": 1.0e-12},
    "ReverseBuoyancy3Zones": {"flow": 1.2e+00, "T": 8.7e-05, "p": 2.1e-08, "Xi": 4.4e-04},
    "ZonalFlow": {"flow": 1.0e-12, "T": 4.4e-05, "p": 1.0e-12, "Xi": 3.3e-06},
}


def test_every_fixture_is_parity_checked_or_refused():
    # "parity-algebraic"/"parity-dynamic" are the committed recorded-error files this
    # module writes (see RECORD/RECORD_DYNAMIC, NOODL_RECORD_PARITY), not model fixtures.
    fixtures = {p.stem for p in DATA.glob("*.json")} - {"parity-algebraic", "parity-dynamic"}
    sets = (set(ALGEBRAIC), set(DYNAMIC), set(REFUSED))
    assert set().union(*sets) == fixtures
    assert sum(len(s) for s in sets) == len(fixtures)  # disjoint
    storage = {m for m, g in DYNAMIC_GROUP.items() if g == "storage"}
    assert set(DYNAMIC_BOUNDS) | storage == set(DYNAMIC) == set(DYNAMIC_GROUP)
    assert not set(DYNAMIC_BOUNDS) & storage


@pytest.mark.parametrize("model", [pytest.param(m, marks=pytest.mark.slow)
                                   if m in SLOW_DYNAMIC else m for m in DYNAMIC])
def test_parity_on_the_dynamic_models(model):
    """Every CSV column after t = StartTime within its model's bound (`DYNAMIC_BOUNDS`).

    OpenModelica solves each model with DASSL at the declared tolerance (1e-6 relative;
    1e-8 for the two OpenDoorBuoyancy*Dynamic) and adaptive steps; noodl stores each
    volume's mass by backward Euler (`modelica/storage.py`; the first interval on a graded
    sub-grid, `run.START_GRADING`) and steps heat and species with the exact scheme at the
    CSV interval, the flows held at their end-of-step values (`coupling="iterate"`), which is
    first order in the step. Relative errors use the
    FLOOR_FRAC floor (1e-3 of the model's largest flow for flows). The measurements behind
    each bound, the step-halving and mass-storage checks, are summarised here:

    * ThreeRoomsContam, ThreeRoomsContamDiscretizedDoor, OneRoom, ZonalFlow: pinned or
      mixing temperatures, flows to <= 2.4e-6 relative (the reference's own 1e-6 tolerance).
      Xi to 2.6e-6: MBL's Xi[1] of a volume moves by X dp/p while the volume's pressure
      re-balances through mass storage (volTop: 35 Pa of 101325), and so does noodl's with
      volume mass storage (3.5e-4 on the quasi-steady route, where it stayed).
      ZonalFlow's T is the one number in this group that is not round-off: rooB.T peaks at
      1.04e-2 K (3.5e-5 relative) at t = 36 s. Not solver tolerance: rooA and rooB start
      10 K and 0.005 kg/kg water apart (ZonalFlow.json), and noodl carries heat with one
      common `cp` instead of MBL's per-zone `cp(X)` (assemble.py's "Capacities" derivation,
      `|cp(X_in)/cp(X) - 1| <= 0.84 |dX_w|` here 0.84 * 0.005 = 4.2e-3 relative). Applied to
      the 10 K starting gap that bounds the error at about 0.04 K, the same order of
      magnitude as the measured 1.04e-2 K (about 4x tighter, plausibly because rooB's 1 m3
      is 1 % of rooA's 100 m3 and the gap decays as they mix). This is the most likely
      cause; it has not been isolated by rerunning with a per-zone `cp`.
    * OpenDoorBuoyancyDynamic, OpenDoorBuoyancyPressureDynamic: flows 2.4 %, temperatures
      0.011 K: noodl's first-order step (halving the step halves both errors, ratio 2.0).
    * CO2TransportStep: T, p as ThreeRoomsContam; flows 1.3e-5, six times ThreeRoomsContam's,
      in the pulse row, because the air source is a step mean too. C up to 170 % in the row
      after the 3.6 s source pulse, which noodl spreads over its 172.8 s step (the injected
      mass is exact: source drivers are step means; halving the step divides this by 2.9).
      A tight integration (DOP853, rtol 1e-12) of the same three-zone species equations,
      reusing noodl's flows (equal to MBL's to 4e-7) and the exact pulse, so independent in
      the time integration only, puts noodl 22 % off it two rows after the pulse (volTop,
      3801.6 s; MBL 1.8 %). From about t > 5000 s the reference's own error dominates: MBL
      is off it by 12 % at 46310 s where noodl is off by 0.2 % (trace substances are scaled
      by C_nominal = 0.01 against values ~1e-7).
    * ClosedDoors, OneOpenDoor ("storage", `_storage_tolerance`): closed rooms of an ideal
      gas (PerfectGas, SimpleAir) heated by a 100 W sine; the heated air expands against the
      closed doors (pressure up 243 and 366 Pa) and heats at constant volume. The
      quasi-steady route (`mass_storage=False`) was off by the cp/cv ratio of the
      temperature rises (0.31 K, 243 Pa, flows 73 % of the largest); with storage: ClosedDoors
      T 1.5e-4 K (5.0e-7 relative), p 2.0e-2 Pa (1.9e-7), Xi 4.9e-9, flows 1.4e-6 kg/s (1.3 %
      of the largest, 1.15e-4 kg/s); OneOpenDoor T 2.8e-3 K, p 8.2e-2 Pa (8.1e-7), flows
      6.9e-4 kg/s (0.8 % of 0.087). Halving the step: ClosedDoors flows / 1.99, T / 1.38,
      p / 1.28 (the rest, 2e-7 relative, is at the reference's tolerance); OneOpenDoor T
      / 1.99, flows / 1.41, p / 1.02 (8e-7 relative: the reference's own tolerance, 0.1 Pa).
    * NaturalVentilation, ReverseBuoyancy3Zones: absolute flow errors <= 0.15 % and 0.27 %
      of the largest flow, from the first-order step (halving it divides them by 1.3-1.4
      and 2.0); the relative bound is set where the flows reverse through zero.
    * ReverseBuoyancy ("storage"): the zones start at 101325 Pa against a 100000 Pa
      outside; MBL and noodl release the excess air through storage (V drho/dt 0.13 kg/s at
      t = 7.2 s) and cool the zones by the flow work less the latent heat of MBL's Xi drop
      (`_check_initial_imbalance`); the quasi-steady route started balanced and was off by
      0.90 K, 566 Pa. With storage the t = StartTime row agrees (flows 1e-10 relative)
      and the worst errors, all during the release at t = 14.4-21.6 s, are T 5.7e-2 K, p
      98 Pa (9.8e-4 relative), Xi 9.6e-6, flows 5.0e-2 kg/s: first order in the step once
      the release is resolved (`test_storage_models_converge_with_the_step`).
    """
    head, data = _csv(model)
    start = time.perf_counter()
    run: dict = {}
    cols = _noodl_columns(model, head, data, run)
    seconds = time.perf_counter() - start
    assert set(cols) == set(head[1:])
    stats = _dynamic_stats(head, data, cols)
    worst: dict[str, dict] = {}
    for h, s in stats.items():
        w = worst.setdefault(s["kind"], {"max_rel": -1.0})
        if s["max_rel"] > w["max_rel"]:
            w.update(max_rel=s["max_rel"], column=h, t=s["t_max_rel"])
        if s["max_abs"] >= w.get("max_abs", 0.0):
            w.update(max_abs=s["max_abs"], abs_column=h, unit=UNITS[s["kind"]])
        w["t0_rel"] = max(w.get("t0_rel", 0.0), s["t0_rel"])
        w["t0_abs"] = max(w.get("t0_abs", 0.0), s["t0_abs"])
    for kind, w in worst.items():
        print(f"{model} {kind}: max rel {w['max_rel']:.3e} ({w['column']} at t = {w['t']:g}), "
              f"max abs {w['max_abs']:.3e} {w['unit']} ({w['abs_column']}); t0 row rel "
              f"{w['t0_rel']:.3e}, abs {w['t0_abs']:.3e} {w['unit']}")
    if DYNAMIC_GROUP[model] == "storage":
        mechanism = _check_storage_mechanism(model, head, data, run)
        tol = _storage_tolerance(model, head, data)
        _record(model, {"group": "storage", "columns": stats, "worst": worst,
                        "tolerance": tol, "mechanism": mechanism, "seconds": seconds},
                RECORD_DYNAMIC)
        failures = []
        for j, h in enumerate(head[1:], start=1):
            err = np.abs(np.asarray(cols[h], dtype=float) - data[:, j])
            if not err.max() <= tol[h]:  # every row, t = StartTime included
                k = int(np.argmax(err))
                failures.append(f"{h}: {err.max():.3e} at t = {data[k, 0]:g} > {tol[h]:.3e}")
        assert not failures, f"{model} outside _storage_tolerance:\n" + "\n".join(failures)
        return
    _record(model, {"group": DYNAMIC_GROUP[model], "columns": stats, "worst": worst,
                    "bounds": DYNAMIC_BOUNDS[model], "floor_frac": FLOOR_FRAC,
                    "mechanism": None, "seconds": seconds}, RECORD_DYNAMIC)
    assert set(worst) == set(DYNAMIC_BOUNDS[model])
    failures = [f"{kind}: max rel {w['max_rel']:.3e} ({w['column']} at t = {w['t']:g}) > "
                f"{DYNAMIC_BOUNDS[model][kind]:.1e}" for kind, w in worst.items()
                if not w["max_rel"] <= DYNAMIC_BOUNDS[model][kind]]
    assert not failures, f"{model} outside its bounds:\n" + "\n".join(failures)


# ------------------------------------------------------- storage-dominated: the mechanism
def _volumes(doc: dict) -> dict[str, dict]:
    return {c["name"]: c["parameters"] for c in doc["components"]
            if c["class"] in VOLUME_CLASSES}


def _storage_tolerance(model: str, head: list[str], data: np.ndarray) -> dict[str, float]:
    """Per column: `Tolerance max|omc| + max_k |omc_k - omc_(k-1)|`, the reference's own
    declared relative tolerance (DASSL, `experiment.Tolerance`) plus the column's largest
    change over one output interval.

    The second term is the error budget of noodl's first-order step, justified rather than
    measured: the reader steps storage by backward Euler and heat and species with the
    flows held at their end-of-step values, so a flow reported at `t_k` is the storage
    rate averaged over `(t_(k-1), t_k)`, which lags the instantaneous one by about half an
    interval's change, and a state lags likewise; the error halves with the step
    (`test_storage_models_converge_with_the_step`). The bound holds at every row,
    t = StartTime included (the reader reproduces MBL's initial equations there)."""
    doc = json.loads((DATA / f"{model}.json").read_text())
    rtol = float(doc["experiment"]["Tolerance"])
    return {h: float(rtol * np.abs(data[:, j]).max() + np.abs(np.diff(data[:, j])).max())
            for j, h in enumerate(head) if j}


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
    `sum V dp dStp/pStp = int m_flow`: asserted to 1e-6 of the final injected mass for MBL and
    for noodl (whose source drivers are step means, so its injected mass is exact)."""
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
    off = {k: float(np.abs(v - injected).max() / injected[-1]) for k, v in stored.items()}
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
    6.1e-5 and 2.2e-4), and the ratio of the V-weighted temperature rises MBL/noodl at the
    peak of `int Q` equal to 1 to 1e-4 (not cp/cv). `int Q` is the closed form of the JSON's
    signals."""
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
    within 2 % of MBL's (a first-order step over a release lasting a few intervals; see
    `_storage_tolerance`)."""
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
    assert abs(cool_noodl / cool_mbl - 1.0) <= 0.02
    return {"t0_p_equals_p_start": True, "cooling_mbl_K": cool_mbl,
            "cooling_noodl_K": cool_noodl, "t": float(data[k, 0])}


@pytest.mark.slow
def test_step_limited_error_halves_with_the_step(tmp_path):
    """OpenDoorBuoyancyDynamic ("step" group): re-read with half the output interval, run on
    the fine grid and compared on the CSV rows, it has half the error: noodl's step is
    first order (flows held at their end-of-step values), and that is the whole difference.
    Over the first 24 rows past t0 (the error peaks at 173-518 s), the ratio of the worst
    absolute errors at r = 1 and r = 2 must lie in [1.8, 2.2] for the door flow and the zone
    temperature (measured over the full run: 1.97 for both on the quasi-steady route; 1.86
    and 1.83 over the 24 rows with volume mass storage)."""
    model, n = "OpenDoorBuoyancyDynamic", 25
    head, data = _csv(model)
    errs = {}
    for r in (1, 2):
        doc = json.loads((DATA / f"{model}.json").read_text())
        doc["experiment"]["Interval"] /= r
        path = tmp_path / f"{model}_r{r}.json"
        path.write_text(json.dumps(doc))
        net, state, drivers, names = read_modelica(path, return_names=True)
        out = simulate(net, state, drivers, names.times[:r * (n - 1) + 1])
        assert np.allclose(out["time"][::r].numpy(), data[:n, 0], atol=1e-9)
        q = out["air.q"][::r].numpy()
        flow = sum(s * q[:, c] for c, s in names.edges["doo.port_a1"])
        temp = out["T"][::r, names.nodes["bouA"]].numpy()
        errs[r] = (np.abs(flow - data[:n, head.index("doo.m1_flow")])[1:].max(),
                   np.abs(temp - data[:n, head.index("bouA.T")])[1:].max())
    ratios = [float(a / b) for a, b in zip(errs[1], errs[2], strict=True)]
    print(f"step halving on {model}: error ratios (flow, T) {ratios}")
    assert all(1.8 <= x <= 2.2 for x in ratios), ratios


def _worst_abs(model: str, kind: str, n: int, r: int, tmp_path) -> float:
    """Worst absolute error of `kind` over CSV rows 1..n-1, noodl run on the grid refined
    r times (the experiment Interval divided by r) and compared on the CSV rows."""
    head, data = _csv(model)
    data = data[:n]
    doc = json.loads((DATA / f"{model}.json").read_text())
    doc["experiment"]["Interval"] /= r
    path = tmp_path / f"{model}_r{r}.json"
    path.write_text(json.dumps(doc))
    net, state, drivers, names = read_modelica(path, return_names=True)
    out = simulate(net, state, drivers, names.times[:r * (n - 1) + 1])
    assert np.allclose(out["time"][::r].numpy(), data[:, 0], atol=1e-9)
    worst = 0.0
    for j, h in enumerate(head[1:], start=1):
        inst, var = h.rsplit(".", 1)
        if _kind(h) != kind or kind == "flow":
            continue
        got = out[_NODE_VARS[var]][::r, names.nodes[inst]].numpy()
        worst = max(worst, float(np.abs(got - data[:, j])[1:].max()))
    return worst


@pytest.mark.slow
@pytest.mark.parametrize("model, kind, n, r", [
    ("OneOpenDoor", "T", 80, 1),        # its worst T error, at t = 1123.2 s (row 78)
    ("ReverseBuoyancy", "p", 13, 4),    # the release, worst at t = 14.4 s (row 2)
])
def test_storage_models_converge_with_the_step(model, kind, n, r, tmp_path):
    """The storage models' remaining difference is noodl's first-order step
    (`_storage_tolerance`): refining the grid from r to 2r divides the worst error by 2
    within [1.7, 2.3]. Measured: OneOpenDoor's T 1.99 (r = 1 -> 2); ReverseBuoyancy's p
    1.34, 1.72, 1.88 for r = 1 -> 2 -> 4 -> 8 (98, 73, 43, 23 Pa: the release takes a few
    intervals and is first order only once they are resolved), so it is tested at r = 4."""
    coarse = _worst_abs(model, kind, n, r, tmp_path)
    fine = _worst_abs(model, kind, n, 2 * r, tmp_path)
    ratio = coarse / fine
    print(f"step halving on {model} {kind}: {coarse:.3e} -> {fine:.3e}, ratio {ratio:.3f}")
    assert 1.7 <= ratio <= 2.3, ratio

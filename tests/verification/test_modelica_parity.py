"""Parity with the Modelica Buildings Library on its algebraic multizone models (spec section 8).

The reference implementation is MBL v13.0.0 (commit 55abf579) simulated by OpenModelica
1.27.1 (`tests/data/modelica/NOTICE.md`); agreement with it is parity. Each model below is
read with `read_modelica`, run with `run.simulate` on the reference CSV's own time grid, and
every compared variable of the CSV is checked at EVERY row against

    |noodl - omc| <= 1e-6 |omc| + 1e-9   (kg/s),

the spec's "1e-6 relative, or 1e-9 kg/s absolute near zero flow". The tolerance is not
loosened for any model or row.

The algebraic set is the models with no `MixingVolume`/`DelayFirstOrder` (checked from each
JSON by `test_the_algebraic_set_has_no_volumes`): between two or more pressure boundaries the
flows are algebraic in the boundary signals, so noodl's steady solve at each grid time is the
same problem the DAE solver solves, and parity is to round-off, not to the solver tolerance.
`OneEffectiveAirLeakageArea` has two volumes and a mass source and is refused (below), so it
is not in this set.

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
the git-ignored ledger directory exists, recorded in
`.superpowers/sdd/2026-09-24-modelica-import/parity-algebraic.json` for the documentation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from noodl.apps.building_physics import read_modelica
from noodl.apps.building_physics.modelica.run import simulate, step_drivers
from noodl.apps.building_physics.modelica.schema import ModelicaImportError

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "modelica"
LEDGER = ROOT / ".superpowers" / "sdd" / "2026-09-24-modelica-import"
RECORD = LEDGER / "parity-algebraic.json"

RTOL, ATOL = 1e-6, 1e-9  # spec section 8; never loosened
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


def _record(model: str, stats: dict) -> None:
    if not LEDGER.is_dir():
        return
    try:
        doc = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    except json.JSONDecodeError:
        doc = {}
    doc[model] = stats
    RECORD.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


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


def _noodl_columns(model: str, head: list[str], data: np.ndarray) -> dict[str, np.ndarray]:
    net, state, drivers, names = read_modelica(DATA / f"{model}.json", return_names=True)
    times = torch.tensor(data[:, 0], dtype=torch.float64)
    assert names.times.numel() == times.numel()
    assert torch.allclose(names.times, times, rtol=0.0, atol=1e-9)
    out = simulate(net, state, drivers, times)
    q = out["air.q"].numpy()
    cols: dict[str, np.ndarray] = {}
    doors: dict[str, tuple[np.ndarray, ...]] = {}
    for h in head[1:]:
        inst, var = h.rsplit(".", 1)
        kinds = names.kinds.get(inst, ())
        if kinds and kinds[0].startswith("door_c:") and var in (
                "m1_flow", "m2_flow", "mAB_flow", "mBA_flow"):
            if inst not in doors:
                doors[inst] = _discretised_port_flows(net, drivers, names, out, inst)
            cols[h] = doors[inst][0 if var in ("m1_flow", "mAB_flow") else 1]
            continue
        key = {"m_flow": inst, "m1_flow": f"{inst}.port_a1",
               "m2_flow": f"{inst}.port_a2"}.get(var)
        assert key in names.edges, f"{model}: CSV column {h!r} maps to no noodl flow"
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
    -40 Pa; measured, 25 Sep 2026). The assertion is therefore that noodl's departure from
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
    # A mass source into two volumes with no boundary: only compressible storage could
    # take the injected mass (controller ruling: out of scope).
    "OneEffectiveAirLeakageArea": {"sou"},
}


@pytest.mark.parametrize("model", sorted(REFUSED))
def test_refused_models_name_their_offending_instances(model):
    with pytest.raises(ModelicaImportError) as exc:
        read_modelica(DATA / f"{model}.json")
    message = str(exc.value)
    named = set(re.findall(r"^\s+- (\S+) \(", message, flags=re.MULTILINE))
    assert named == REFUSED[model], message
    assert f"refused {len(REFUSED[model])} item" in message

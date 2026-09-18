"""SWMM parity on the committed tree (rows W1-W4), through pyswmm 2.1.0 / SWMM 5.2.4.

pyswmm ALWAYS writes a `.rpt` and a `.out` next to the model it runs (`pyswmm/swmm5.py`
silently derives both paths when they are not given), so every test here copies the fixture
into `tmp_path` and hands the engine explicit output paths there. Nothing is written into
the repository.
"""

import shutil
from pathlib import Path

import pytest
import torch

from tellegen.apps.sewer.inp import read_inp
from tellegen.apps.sewer.network import build_sewer_model

pyswmm = pytest.importorskip("pyswmm")
shared_enum = pytest.importorskip("swmm.toolkit.shared_enum")

F64 = torch.float64
DATA = Path(__file__).resolve().parents[1] / "data" / "sewer"
LINKS = ["C1", "C2", "C3", "C4", "C5"]


def _run_swmm(tmp_path, name):
    """Run the fixture and return the live link results plus the binary velocities."""
    inp = tmp_path / name
    shutil.copy(DATA / name, inp)
    out = {}
    with pyswmm.Simulation(
        str(inp), str(tmp_path / f"{inp.stem}.rpt"), str(tmp_path / f"{inp.stem}.out")
    ) as sim:
        for _ in sim:
            pass
        for link_id in LINKS:
            link = pyswmm.Links(sim)[link_id]
            entry = {"flow": link.flow, "depth": link.depth, "volume": link.volume}
            try:
                entry["pollut"] = dict(link.pollut_quality)
            except Exception:  # pragma: no cover - only when no pollutant is defined
                entry["pollut"] = {}
            out[link_id] = entry
        out["error"] = sim.flow_routing_error
        out["engine"] = sim.engine_version
    with pyswmm.Output(str(tmp_path / f"{inp.stem}.out")) as binary:
        for link_id in LINKS:
            series = binary.link_series(link_id, shared_enum.LinkAttribute.FLOW_VELOCITY)
            out[link_id]["velocity"] = list(series.items())[-1][1]
    return out


@pytest.fixture(scope="module")
def tellegen_steady():
    net, _, _ = read_inp(DATA / "tree_kinwave.inp")
    model, state, drivers = build_sewer_model(net, air=False, quality=False)
    resolved = model._apply_closures(state, drivers)
    return net, resolved


def test_the_engine_is_the_one_the_measurements_used(tmp_path):
    result = _run_swmm(tmp_path, "tree_kinwave.inp")
    # `Simulation.engine_version` returns a `packaging.version.Version` (the docstring's own
    # `LooseVersion` type hint is stale), which compares unequal to a bare `str` under `==`
    # (`NotImplemented`, so pytest reports it as a plain inequality) -- `str()` it first.
    assert str(result["engine"]) == "5.2.4"
    assert result["error"] == 0.0


def test_w1_pipe_flows(tmp_path, tellegen_steady):
    """Row W1, 1e-9 relative. Measured worst 1.370e-14."""
    net, resolved = tellegen_steady
    swmm = _run_swmm(tmp_path, "tree_kinwave.inp")
    order = [p.name for p in net.pipes]
    worst = 0.0
    for i, name in enumerate(order):
        mine = float(resolved["sewer.q"][i])
        worst = max(worst, abs(mine - swmm[name]["flow"]) / abs(swmm[name]["flow"]))
    assert worst < 1e-9, worst


def test_w2_normal_depths(tmp_path, tellegen_steady):
    """Row W2, 1e-3 relative. Measured worst 5.464e-4 (conduit C3), which is SWMM's own
    51-point circular lookup table (Ref. Man. Vol. II section 5.1.3), not solver noise."""
    net, resolved = tellegen_steady
    swmm = _run_swmm(tmp_path, "tree_kinwave.inp")
    worst = 0.0
    for i, name in enumerate(p.name for p in net.pipes):
        mine = float(resolved["sewer.h"][i])
        worst = max(worst, abs(mine - swmm[name]["depth"]) / swmm[name]["depth"])
    assert worst < 1e-3, worst
    assert worst > 1e-5, "the lookup-table difference has disappeared: check the geometry"


def test_w3_velocities(tmp_path, tellegen_steady):
    """Row W3, 1e-3 relative, against the BINARY output's FLOW_VELOCITY.

    Spec amendment A2: under KINWAVE, `Link.ups_xsection_area` is exactly 0.0 (SWMM fills
    it only under DYNWAVE), so the spec's `flow / ups_xsection_area` is 0/0. Measured worst
    6.257e-4.
    """
    net, resolved = tellegen_steady
    swmm = _run_swmm(tmp_path, "tree_kinwave.inp")
    worst = 0.0
    for i, name in enumerate(p.name for p in net.pipes):
        mine = float(resolved["sewer.v"][i])
        worst = max(worst, abs(mine - swmm[name]["velocity"]) / swmm[name]["velocity"])
    assert worst < 1e-3, worst


def test_conduit_volumes(tmp_path, tellegen_steady):
    """Not a numbered row, but the capacity every quality layer is built on. Measured worst
    6.438e-4 relative, the same lookup-table difference as W2."""
    net, resolved = tellegen_steady
    swmm = _run_swmm(tmp_path, "tree_kinwave.inp")
    worst = 0.0
    for i, name in enumerate(p.name for p in net.pipes):
        mine = float(resolved["sewer.V_wet"][i])
        worst = max(worst, abs(mine - swmm[name]["volume"]) / swmm[name]["volume"])
    assert worst < 1e-3, worst


def test_w4_tracer_concentration(tmp_path):
    """Row W4, 3e-5 relative (spec amendment A4; measured worst 7.811e-6 on conduit C5).

    The model's steady state IS the tank-in-series closed form: a manhole's balance is
    `sum(q_in C_in) = (q_out + k V) C`, which is exactly `C = C_mix / (1 + k tau)` with
    `tau = V / q`. SWMM's own conduit model is the same completely-mixed reactor (Ref. Man.
    Vol. III Eq. 5-4), which is why the row is this tight.
    """
    from tellegen.layers.reaction import FirstOrderDecay

    net, loads, pollutants = read_inp(DATA / "tree_kinwave_pollut.inp")
    model, state, drivers = build_sewer_model(net, air=False, quality=False)
    resolved = model._apply_closures(state, drivers)
    k = pollutants["Tracer"]["decay"]
    volume = resolved["sewer.V_wet"]
    q = resolved["sewer.q"]
    tau = volume / q
    order = {p.name: i for i, p in enumerate(net.pipes)}
    # The tree, resolved by hand: C1 and C2 mix at J3 into C3; C3 and C4 mix at J4 into C5.
    c = {}
    c["C1"] = 0.1 / (1.0 + k * float(tau[order["C1"]]))
    c["C2"] = 0.0
    c["C4"] = 0.0
    mix_j3 = (c["C1"] * 0.05 + c["C2"] * 0.08) / 0.13
    c["C3"] = mix_j3 / (1.0 + k * float(tau[order["C3"]]))
    mix_j4 = (c["C3"] * 0.13 + c["C4"] * 0.03) / 0.16
    c["C5"] = mix_j4 / (1.0 + k * float(tau[order["C5"]]))
    _ = FirstOrderDecay  # the framework reaction the app's own layer would carry
    swmm = _run_swmm(tmp_path, "tree_kinwave_pollut.inp")
    worst = 0.0
    for name in LINKS:
        theirs = swmm[name]["pollut"]["Tracer"] * 1e-3      # mg/L -> kg/m3
        mine = c[name]
        if mine == 0.0:
            assert abs(theirs) < 1e-12
            continue
        worst = max(worst, abs(mine - theirs) / mine)
    assert worst < 3e-5, worst

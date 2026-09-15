"""Parity with NIST's ContamX engine through contamxpy (spec section 9, amendments 14).

Skipped when contamxpy is not importable. Flows are compared after ContamX's initial
steady-state airflow solve; concentrations over a 24-step transient at the project's own
5-minute step with the implicit-Euler contaminant solver on both sides.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tellegen.apps.building.prj import project_to_model, read_prj

DATA = Path(__file__).resolve().parents[1] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
# Ruling R30: NOT NIST's `test_OneZoneSsStack-UseApi.prj`. That project attaches a
# constant-efficiency filter (`f# = 1`) removing 10% of sarin on path 1, which Task 12's
# reader refuses -- a filter is invisible to airflow but not to the species layer, so
# loading it would silently corrupt contaminant results. `test_OneZoneWthCtmStack-UseApi`
# is the same one-zone stack geometry from the same demo set with no filter on any path.
STACK = DATA / "test_OneZoneWthCtmStack-UseApi.prj"
# The stack project's own ambient is 293.15 K, exactly its zone temperature, so under its
# recorded conditions (and zero wind) every path flow is identically zero -- the engine
# confirms it -- and a direction test there would compare nothing but zeros. Cooling the
# ambient by 20 K makes the zone buoyant and the two openings a genuine stack pair.
STACK_AMBIENT_T = 273.15
F64 = torch.float64
pytestmark = pytest.mark.external


def test_contamx_module_names_the_missing_package(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "contamxpy", None)
    contamx = importlib.import_module("tellegen.apps.building.contamx")
    with pytest.raises(ImportError, match=r"contamxpy.*pip install"):
        contamx.run_steady(THREE, ambient={"Ta": 293.15, "Pb": 101325.0, "Ws": 0.0, "Wd": 0.0})


def _stack_case(contamx_run_steady):
    p = read_prj(STACK)
    amb = dict(p.ambient_conditions, Ta=STACK_AMBIENT_T)
    ref = contamx_run_steady(STACK, ambient=amb)
    model, state, drivers = project_to_model(p, ambient=amb)
    ss = model.steady(state, drivers)
    return p, ref, p.path_flows(ss["air.q"])


def test_stack_project_flow_directions_match_contamx(contamx):
    from tellegen.apps.building.contamx import run_steady

    p, ref, ours = _stack_case(run_steady)
    assert [int(n) for n in ref["path_nr"]] == [path.nr for path in p.paths]
    # Both paths run ambient -> zone, path 1 at relative height 0.0 and path 2 at 1.5 m.
    # With the zone 20 K warmer than outside, buoyancy admits cold air at the LOW opening
    # and expels warm air at the high one: the net is positive on path 1 and negative on
    # path 2, and that is known before either engine is consulted. This is the independent
    # check on the from->to sign convention `contamx.py` documents (Ruling R12).
    assert torch.equal(torch.sign(ours), torch.tensor([1.0, -1.0], dtype=F64))
    assert torch.equal(torch.sign(ours), torch.sign(ref["flow"]))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN, DIAGNOSED disagreement in the STACK term -- not a fault of this driver, and "
        "the tolerance is deliberately left at the value the milestone asks for rather than "
        "loosened to pass. `prj.py` converts CONTAM's turbulent coefficient for the "
        "`plr_orfc`/`plr_leak*`/`plr_crack` family with a FIXED reference density, "
        "C = turb * sqrt(RHO_0), RHO_0 = 1.2041; ContamX instead evaluates sqrt(rho) at the "
        "density of the air actually entering the path, which differs per path and with the "
        "flow direction. Hand-integrating the two-orifice stack balance with the upstream "
        "density reproduces ContamX to 4.3e-5 relative at every ambient from 273.15 K to "
        "313.15 K, and the same balance with the fixed RHO_0 reproduces tellegen's answer "
        "exactly, which pins the cause. The error in flow is about 0.086% per kelvin of "
        "|T_zone - T_ambient| (half the density error, since flow ~ sqrt(rho)): 1.7e-2 at "
        "the 20 K used here, and below 1e-3 for the isothermal wind-driven three-zone case, "
        "which is why that one passes. Fixing it means making the orifice coefficient track "
        "the upstream density, i.e. a solution-dependent PowerLaw coefficient -- an element- "
        "and Jacobian-level change to Task 12's converter, outside Task 14. `strict=True` so "
        "that this turns into a failure the moment that change lands."
    ),
)
def test_stack_project_flow_magnitudes_match_contamx(contamx):
    from tellegen.apps.building.contamx import run_steady

    _p, ref, ours = _stack_case(run_steady)
    torch.testing.assert_close(ours, ref["flow"], rtol=1e-3, atol=1e-6)


def test_three_zone_steady_flows_match_contamx(contamx):
    from tellegen.apps.building.contamx import run_steady

    p = read_prj(THREE)
    amb = dict(p.ambient_conditions, mf={0: 0.0023254})
    ref = run_steady(THREE, ambient=amb)
    model, state, drivers = project_to_model(p, ambient=amb)
    ss = model.steady(state, drivers)
    torch.testing.assert_close(p.path_flows(ss["air.q"]), ref["flow"], rtol=1e-3, atol=1e-6)


def test_three_zone_transient_concentrations_match_contamx(contamx):
    from tellegen.apps.building.contamx import run_transient

    p = read_prj(THREE)
    amb = dict(p.ambient_conditions, mf={0: 0.0023254})
    steps = 24
    ref = run_transient(THREE, steps=steps, ambient=amb)
    assert ref["dt"] == pytest.approx(300.0)
    model, state, drivers = project_to_model(p, ambient=amb, scheme="implicit")
    drivers["species.x_boundary"] = torch.tensor([[0.0023254]], dtype=F64)
    trace = [state["species.x"].clone()]
    for _ in range(steps):
        state = model.step(state, drivers, ref["dt"])
        trace.append(state["species.x"].clone())
    ours = torch.stack(trace)                                   # (steps+1, 3, 1)
    torch.testing.assert_close(ours, ref["mf"], rtol=2e-3, atol=1e-7)

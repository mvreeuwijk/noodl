"""Parity with NIST's ContamX engine through contamxpy (spec section 9, amendments 14).

Skipped when contamxpy is not importable. Flows are compared after ContamX's initial
steady-state airflow solve; concentrations over a 24-step transient at the project's own
5-minute step with the implicit-Euler contaminant solver on both sides.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from noodl.apps.building.prj import project_to_model, read_prj

DATA = Path(__file__).resolve().parents[1] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
# Ruling R30: NOT NIST's `test_OneZoneSsStack-UseApi.prj`. That project attaches a
# constant-efficiency filter (`f# = 1`) removing 10% of sarin on path 1, which Task 12's
# reader refuses -- a filter is invisible to airflow but not to the species layer, so
# loading it would silently corrupt contaminant results. `test_OneZoneWthCtmStack-UseApi`
# below is the same one-zone stack geometry from the same demo set with no filter on any
# path. That the filtered project is genuinely REFUSED, rather than merely unused, is
# asserted by `tests/apps/building/test_prj.py::
# test_the_filtered_one_zone_stack_project_is_refused_naming_the_filter`.
STACK = DATA / "test_OneZoneWthCtmStack-UseApi.prj"
MIXED = DATA / "doorway_damper_fan.prj"
# The stack project's own ambient is 293.15 K, exactly its zone temperature, so under its
# recorded conditions (and zero wind) every path flow is identically zero -- the engine
# confirms it -- and a direction test there would compare nothing but zeros. Cooling the
# ambient by 20 K makes the zone buoyant and the two openings a genuine stack pair.
STACK_AMBIENT_T = 273.15
# The whole sweep the upstream-density correction is measured over: +-20 K around the zone's
# own 293.15 K, both buoyancy directions and both flow directions through each opening.
STACK_AMBIENT_SWEEP = (273.15, 283.15, 303.15, 313.15)
# `doorway_damper_fan.prj` path 5 carries flow element 1, an `fan_cmf` constant-MASS-flow fan
# rated at this many kg/s, on a path declared zone -> ambient. Written once here and asserted
# against both the file and the engine below (see `contamx._snapshot`, Ruling R12).
FAN_CMF_RATING = 0.200683
FAN_CMF_PATH = 5
# The ambient mass fraction of the three-zone project's single species. ONE constant: it is
# both what the engine is told through its `mf` ambient dict and what the boundary condition
# of noodl's own species layer must be, and the two silently disagreeing would compare a
# transient against a different problem.
THREE_AMBIENT_MF = 0.0023254
F64 = torch.float64
# NOT a module-level `pytestmark`: the two tests below monkeypatch their way to the behaviour
# they pin and need no engine at all, so marking them `external` would misreport what this
# file requires (and would hide them from a `-m "not external"` run that could perfectly well
# execute them). `external` is carried by the engine-dependent tests themselves -- exactly
# those taking the `contamx` fixture. Nothing about the default selection changes: `addopts`
# deselects `slow` only, so `external` has never gated what runs.


def test_contamx_module_names_the_missing_package(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "contamxpy", None)
    contamx = importlib.import_module("noodl.apps.building.contamx")
    with pytest.raises(ImportError, match=r"contamxpy.*pip install"):
        contamx.run_steady(THREE, ambient={"Ta": 293.15, "Pb": 101325.0, "Ws": 0.0, "Wd": 0.0})


def test_a_refused_project_is_reported_by_the_path_the_caller_gave(monkeypatch):
    """`_isolated` hands the engine a COPY inside a temporary directory that is deleted
    before the exception reaches the caller, so naming that copy in the refusal message
    points at a path that no longer exists and that the caller never asked for. No engine is
    needed to pin this: a stub whose `setupSimulation` refuses is enough.
    """
    from noodl.apps.building import contamx

    class _Refusing:
        def __init__(self, *_a, **_k):
            pass

        def setVerbosity(self, _v):
            pass

        def setupSimulation(self, _n):
            return 1

        def endSimulation(self):
            pass

    monkeypatch.setattr(contamx, "_cxlib", lambda: _Refusing)
    with pytest.raises(RuntimeError) as exc:
        contamx.run_steady(THREE, ambient={"Ta": 293.15, "Pb": 0.0, "Ws": 0.0, "Wd": 0.0})
    assert str(THREE) in str(exc.value)
    assert "noodl-contamx-" not in str(exc.value)          # not the scratch copy


def _stack_case(contamx_run_steady, Ta=STACK_AMBIENT_T):
    p = read_prj(STACK)
    amb = dict(p.ambient_conditions, Ta=Ta)
    ref = contamx_run_steady(STACK, ambient=amb)
    model, state, drivers = project_to_model(p, ambient=amb)
    ss = model.steady(state, drivers)
    return p, ref, p.path_flows(ss["air.q"])


@pytest.mark.external
def test_stack_project_flow_directions_match_contamx(contamx, record_property):
    from noodl.apps.building.contamx import run_steady

    p, ref, ours = _stack_case(run_steady)
    assert [int(n) for n in ref["path_nr"]] == [path.nr for path in p.paths]
    # Both paths run ambient -> zone, path 1 at relative height 0.0 and path 2 at 1.5 m.
    # With the zone 20 K warmer than outside, buoyancy admits cold air at the LOW opening
    # and expels warm air at the high one: the net is positive on path 1 and negative on
    # path 2, and that is known before either engine is consulted. This is the independent
    # check on the from->to sign convention `contamx.py` documents (Ruling R12).
    assert torch.equal(torch.sign(ours), torch.tensor([1.0, -1.0], dtype=F64))
    assert torch.equal(torch.sign(ours), torch.sign(ref["flow"]))
    record_property("check", "Stack project, flow directions")
    record_property("tolerance", "exact signs")
    record_property("measured_status", "match")


@pytest.mark.external
@pytest.mark.parametrize("Ta", STACK_AMBIENT_SWEEP)
def test_stack_project_flow_magnitudes_match_contamx(contamx, Ta, record_property):
    """The non-isothermal parity case, live at the milestone's own 1e-3 (Task 14b).

    This was `xfail(strict=True)` through Task 14 at a measured 1.7e-2 -- seventeen times
    the tolerance -- because `prj.py` froze every orifice coefficient at the reference
    density RHO_0 while ContamX evaluates it at the density of the air entering the path.
    With `UpstreamDensityPowerLaw` carrying that correction the agreement across the whole
    +-20 K sweep is 4.1e-5 to 4.4e-5 relative, twenty-five times INSIDE the tolerance, and
    the residual is flat in temperature rather than growing with it -- i.e. what is left is
    no longer a density error. The sweep matters: a single ambient would not distinguish the
    correction from a constant rescaling, and both signs of the temperature difference are
    needed because the correction switches which endpoint it reads when the flow reverses.
    """
    from noodl.apps.building.contamx import run_steady

    _p, ref, ours = _stack_case(run_steady, Ta)
    torch.testing.assert_close(ours, ref["flow"], rtol=1e-3, atol=1e-6)
    worst_rel = ((ours - ref["flow"]) / ref["flow"]).abs().max().item()
    record_property("check", "Stack project, flow magnitudes, over a +-20 K ambient sweep")
    record_property("tolerance", "rel 1e-3")
    record_property("measured_rel", worst_rel)


@pytest.mark.external
def test_the_stack_residual_is_flat_across_the_sweep_not_proportional_to_dT(
    contamx, record_property
):
    """Guards the DIAGNOSIS, not just the tolerance: before the correction the relative
    error was proportional to |T_zone - T_ambient| (1.72e-2 at 20 K, 8.61e-3 at 10 K, 0 at
    0 K), which is the signature of the frozen density. Afterwards it must NOT scale with
    the temperature difference -- if a future change reintroduced a density error at, say, a
    tenth of the size, a fixed 1e-3 tolerance alone would not notice.
    """
    from noodl.apps.building.contamx import run_steady

    rel = []
    for Ta in STACK_AMBIENT_SWEEP:
        _p, ref, ours = _stack_case(run_steady, Ta)
        rel.append(((ours - ref["flow"]) / ref["flow"]).abs().max().item())
    assert max(rel) < 1e-4
    # 10 K and 20 K differ by less than a factor 1.5, against the factor 2.0 a residual
    # linear in the temperature difference would show.
    assert max(rel) / min(rel) < 1.5
    record_property("check", "Residual flatness across the sweep (max/min ratio)")
    record_property("tolerance", "< 1.5")
    record_property("measured_ratio", max(rel) / min(rel))


@pytest.mark.external
def test_a_constant_mass_flow_fan_delivers_its_rating_in_the_from_to_direction(
    contamx, record_property
):
    """The from->to sign convention `contamx._snapshot` documents, asserted rather than
    merely written down (hardening carried from the Task 14 review).

    `doorway_damper_fan.prj` path 5 is declared `n# 1` to `m# -1` (zone -> ambient) and
    carries a constant-MASS-flow fan. A fixed-flow fan has no freedom: it must deliver its
    rating, and the sign the engine reports for it is the sign of "from -> to". noodl's
    own reader must agree on both, which is what makes this a convention test and not just
    an engine smoke test.
    """
    from noodl.apps.building.contamx import run_steady

    p = read_prj(MIXED)
    amb = dict(p.ambient_conditions)
    ref = run_steady(MIXED, ambient=amb)
    i = ref["path_nr"].index(FAN_CMF_PATH)
    assert ref["from_zone"][i] != 0 and ref["to_zone"][i] == 0      # zone -> ambient
    assert ref["flow"][i].item() == pytest.approx(FAN_CMF_RATING, rel=1e-5)
    model, state, drivers = project_to_model(p, ambient=amb)
    ours = p.path_flows(model.steady(state, drivers)["air.q"])
    j = [path.nr for path in p.paths].index(FAN_CMF_PATH)
    assert ours[j].item() == pytest.approx(FAN_CMF_RATING, rel=1e-5)
    record_property("check", "Constant-mass-flow fan delivers its rating (0.200683 kg/s)")
    record_property("tolerance", "rel 1e-5")
    record_property("measured_rel", abs(ours[j].item() / FAN_CMF_RATING - 1.0))


@pytest.mark.external
def test_three_zone_steady_flows_match_contamx(contamx, record_property):
    from noodl.apps.building.contamx import run_steady

    p = read_prj(THREE)
    amb = dict(p.ambient_conditions, mf={0: THREE_AMBIENT_MF})
    ref = run_steady(THREE, ambient=amb)
    model, state, drivers = project_to_model(p, ambient=amb)
    ss = model.steady(state, drivers)
    ours = p.path_flows(ss["air.q"])
    torch.testing.assert_close(ours, ref["flow"], rtol=1e-3, atol=1e-6)
    worst_rel = ((ours - ref["flow"]) / ref["flow"].abs().clamp_min(1e-12)).abs().max().item()
    record_property("check", "Three-zone project, steady flows")
    record_property("tolerance", "rel 1e-3, abs 1e-6")
    record_property("measured_rel", worst_rel)


@pytest.mark.external
def test_three_zone_transient_concentrations_match_contamx(contamx, record_property):
    from noodl.apps.building.contamx import run_transient

    p = read_prj(THREE)
    amb = dict(p.ambient_conditions, mf={0: THREE_AMBIENT_MF})
    steps = 24
    ref = run_transient(THREE, steps=steps, ambient=amb)
    assert ref["dt"] == pytest.approx(300.0)
    model, state, drivers = project_to_model(p, ambient=amb, scheme="implicit")
    drivers["species.x_boundary"] = torch.tensor([[THREE_AMBIENT_MF]], dtype=F64)
    trace = [state["species.x"].clone()]
    for _ in range(steps):
        state = model.step(state, drivers, ref["dt"])
        trace.append(state["species.x"].clone())
    ours = torch.stack(trace)                                   # (steps+1, 3, 1)
    # Spec section 9's tolerance for zone mass fractions is 1e-3 relative; the measured
    # pointwise maximum relative error over the whole (25, 3, 1) trace is 6.5e-6, so the
    # tolerance is the spec's, not a looser one chosen to fit.
    torch.testing.assert_close(ours, ref["mf"], rtol=1e-3, atol=1e-7)
    worst_rel = ((ours - ref["mf"]) / ref["mf"].abs().clamp_min(1e-12)).abs().max().item()
    record_property("check", "Three-zone project, transient concentrations, 24 steps at 300 s")
    record_property("tolerance", "rel 1e-3")
    record_property("measured_rel", worst_rel)

"""Parity with NIST's ContamX engine through contamxpy.

Skipped when contamxpy is not importable. Flows are compared after ContamX's initial
steady-state airflow solve; concentrations over a 24-step transient at the project's own
5-minute step with the implicit-Euler contaminant solver on both sides.

TOLERANCES ARE SET BY THE REFERENCE'S PRECISION, not by the measured agreement. ContamX
holds project input data in SINGLE precision: `doorway_damper_fan.prj`'s fan rated 0.200683
kg/s comes back as 0.20068299770355225, which is float32(0.200683) exactly. Each input it
reads is therefore known to it only to the float32 unit roundoff `F32_U` = 2**-24 = 6.0e-8,
and each tolerance below is a first-order budget of such roundoffs through the quantity
compared (derived at each test). The solver-side contributions are pushed below that:
ContamX's airflow iteration is run at `TIGHT_AIRFLOW` rather than the sample projects'
1e-5 / 1e-6, and noodl physics' Newton at an absolute residual of `NOODL_ATOL` kg/s.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.prj import project_to_model, read_prj, steady

DATA = Path(__file__).resolve().parents[1] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
# NOT NIST's `test_OneZoneSsStack-UseApi.prj`. That project attaches a
# constant-efficiency filter (`f# = 1`) removing 10% of sarin on path 1, which the `.prj`
# reader refuses -- a filter is invisible to airflow but not to the species layer, so
# loading it would silently corrupt contaminant results. `test_OneZoneWthCtmStack-UseApi`
# below is the same one-zone stack geometry from the same demo set with no filter on any
# path. That the filtered project is genuinely REFUSED, rather than merely unused, is
# asserted by `tests/apps/building_physics/test_prj.py::
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
# against both the file and the engine below (see `contamx._snapshot`).
FAN_CMF_RATING = 0.200683
FAN_CMF_PATH = 5
# The ambient mass fraction of the three-zone project's single species. ONE constant: it is
# both what the engine is told through its `mf` ambient dict and what the boundary condition
# of noodl's own species layer must be, and the two silently disagreeing would compare a
# transient against a different problem.
THREE_AMBIENT_MF = 0.0023254
F64 = torch.float64
F32_U = 2.0**-24
# ContamX's airflow (afrcnvg, afacnvg): relative, and absolute in kg/s. See
# `contamx._set_airflow_convergence`.
TIGHT_AIRFLOW = (1e-10, 1e-12)
NOODL_ATOL = 1e-14
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
    contamx = importlib.import_module("noodl.apps.building_physics.contamx")
    with pytest.raises(ImportError, match=r"contamxpy.*pip install"):
        contamx.run_steady(THREE, ambient={"Ta": 293.15, "Pb": 101325.0, "Ws": 0.0, "Wd": 0.0})


def test_a_refused_project_is_reported_by_the_path_the_caller_gave(monkeypatch):
    """`_isolated` hands the engine a COPY inside a temporary directory that is deleted
    before the exception reaches the caller, so naming that copy in the refusal message
    points at a path that no longer exists and that the caller never asked for. No engine is
    needed to pin this: a stub whose `setupSimulation` refuses is enough.
    """
    from noodl.apps.building_physics import contamx

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


def _stack_case(contamx_run_steady, Ta=STACK_AMBIENT_T, *, zone_pressure_density=True):
    p = read_prj(STACK)
    if not zone_pressure_density:
        p = dataclasses.replace(p, density_uses_zone_pressure=False)
    amb = dict(p.ambient_conditions, Ta=Ta)
    ref = contamx_run_steady(STACK, ambient=amb, airflow_convergence=TIGHT_AIRFLOW)
    model, state, drivers = project_to_model(p, ambient=amb)
    ss = steady(model, state, drivers, atol=NOODL_ATOL)
    return p, ref, p.path_flows(ss["air.q"])


def _stack_budget(Ta, Tz=293.15):
    """First-order float32 budget for a two-opening stack flow, relative.

    F = C sqrt(rho_up) sqrt(dp) with dp ~ g h (rho_a - rho_z) and rho = Pb / (R T). A
    relative roundoff u in each input moves F by at most: C 1u; rho_up (R, T_up) 1u; g
    0.5u; R in the density difference 0.5u; and Ta, Tz through the DIFFERENCE
    (1/Ta - 1/Tz), which amplifies them to 0.5 (Ta + Tz) / |Tz - Ta| u. Pb (101325) and the
    opening heights (0, 1.5 m) are exact in float32. Over the sweep this is 17.2u = 1.02e-6
    (Ta = 273.15 K, 20 K below) to 32.8u = 1.96e-6 (Ta = 303.15 K, 10 K above).
    """
    return F32_U * (3.0 + 0.5 * (Ta + Tz) / abs(Tz - Ta))


@pytest.mark.external
def test_stack_project_flow_directions_match_contamx(contamx, record_property):
    from noodl.apps.building_physics.contamx import run_steady

    p, ref, ours = _stack_case(run_steady)
    assert [int(n) for n in ref["path_nr"]] == [path.nr for path in p.paths]
    # Both paths run ambient -> zone, path 1 at relative height 0.0 and path 2 at 1.5 m.
    # With the zone 20 K warmer than outside, buoyancy admits cold air at the LOW opening
    # and expels warm air at the high one: the net is positive on path 1 and negative on
    # path 2, and that is known before either engine is consulted. This is the independent
    # check on the from->to sign convention `contamx.py` documents.
    assert torch.equal(torch.sign(ours), torch.tensor([1.0, -1.0], dtype=F64))
    assert torch.equal(torch.sign(ours), torch.sign(ref["flow"]))
    record_property("check", "Stack project, flow directions")
    record_property("tolerance", "exact signs")
    record_property("measured_status", "match")


@pytest.mark.external
@pytest.mark.parametrize("Ta", STACK_AMBIENT_SWEEP)
def test_stack_project_flow_magnitudes_match_contamx(contamx, Ta, record_property):
    """The non-isothermal parity case, at the float32 budget `_stack_budget(Ta)`.

    Two density effects had to match ContamX's for this to hold:

    * Freezing every orifice coefficient at the reference density RHO_0 (while ContamX
      evaluates it at the density of the air ENTERING the path) measured 1.7e-2, growing
      linearly with |T_zone - T_ambient|. `UpstreamDensityPowerLaw` removes it.
    * The project sets `densZP = 1`: ContamX evaluates zone density at the zone's absolute
      pressure Pb + P_zone. Ignoring it left 4.1e-5 to 4.4e-5, FLAT across the sweep: the
      stack pressure is a density DIFFERENCE, which amplifies the |P_zone| / Pb ~ 6e-6
      density change by rho / |delta rho| ~ 14, and P_zone itself scales with that
      difference. `prj.ZonePressureDensity` removes it; see
      `test_the_zone_pressure_density_is_what_closed_the_stack_residual`.

    What remains is 7.2e-8 to 9.4e-8 (1.2-1.6 F32_U) against a budget of 17.2-32.8 F32_U. The
    sweep matters: both signs of the temperature difference are needed because the upstream
    density switches endpoint when the flow reverses.
    """
    from noodl.apps.building_physics.contamx import run_steady

    _p, ref, ours = _stack_case(run_steady, Ta)
    tol = _stack_budget(Ta)
    torch.testing.assert_close(ours, ref["flow"], rtol=tol, atol=1e-12)
    worst_rel = ((ours - ref["flow"]) / ref["flow"]).abs().max().item()
    record_property("check", "Stack project, flow magnitudes, over a +-20 K ambient sweep")
    record_property("tolerance", f"rel {tol:.2e} (float32 input budget)")
    record_property("measured_rel", worst_rel)


@pytest.mark.external
def test_the_zone_pressure_density_is_what_closed_the_stack_residual(
    contamx, record_property
):
    """Guards the DIAGNOSIS, not just the tolerance.

    With `densZP` ignored the residual must come back at its old size and shape -- 3e-5 to
    6e-5 at every ambient, and flat (max/min < 1.5, where a residual linear in the
    temperature difference would give 2.0 between 10 K and 20 K) -- and honouring it must
    remove at least 99 % of it. If the zone-pressure density were lost, or the 4e-5 ever
    closed by something else, this fails where a tolerance alone would not say why.
    """
    from noodl.apps.building_physics.contamx import run_steady

    off, on = [], []
    for Ta in STACK_AMBIENT_SWEEP:
        _p, ref, ours = _stack_case(run_steady, Ta, zone_pressure_density=False)
        off.append(((ours - ref["flow"]) / ref["flow"]).abs().max().item())
        _p, ref, ours = _stack_case(run_steady, Ta)
        on.append(((ours - ref["flow"]) / ref["flow"]).abs().max().item())
    assert all(3e-5 < r < 6e-5 for r in off), off
    assert max(off) / min(off) < 1.5
    assert all(r_on < 1e-2 * r_off for r_on, r_off in zip(on, off, strict=True)), (on, off)
    record_property("check", "Stack residual with and without the zone-pressure density")
    record_property("tolerance", "off: 3e-5..6e-5 and flat; on: < 1 % of off")
    record_property("measured_off_max", max(off))
    record_property("measured_on_max", max(on))


@pytest.mark.external
def test_a_constant_mass_flow_fan_delivers_its_rating_in_the_from_to_direction(
    contamx, record_property
):
    """The from->to sign convention `contamx._snapshot` documents, asserted rather than
    merely written down.

    `doorway_damper_fan.prj` path 5 is declared `n# 1` to `m# -1` (zone -> ambient) and
    carries a constant-MASS-flow fan. A fixed-flow fan has no freedom: it must deliver its
    rating, and the sign the engine reports for it is the sign of "from -> to". noodl's
    own reader must agree on both, which is what makes this a convention test and not just
    an engine smoke test.
    """
    from noodl.apps.building_physics.contamx import run_steady

    p = read_prj(MIXED)
    amb = dict(p.ambient_conditions)
    ref = run_steady(MIXED, ambient=amb)
    i = ref["path_nr"].index(FAN_CMF_PATH)
    assert ref["from_zone"][i] != 0 and ref["to_zone"][i] == 0      # zone -> ambient
    # The engine's value is the rating rounded to float32: within one unit roundoff.
    assert ref["flow"][i].item() == pytest.approx(FAN_CMF_RATING, rel=F32_U)
    model, state, drivers = project_to_model(p, ambient=amb)
    ours = p.path_flows(steady(model, state, drivers)["air.q"])
    j = [path.nr for path in p.paths].index(FAN_CMF_PATH)
    # noodl physics' is the rating in float64: a fixed flow has nothing to solve for.
    assert ours[j].item() == pytest.approx(FAN_CMF_RATING, rel=1e-12)
    record_property("check", "Constant-mass-flow fan delivers its rating (0.200683 kg/s)")
    record_property("tolerance", "engine rel 2**-24 (float32), noodl physics rel 1e-12")
    record_property("measured_rel", abs(ours[j].item() / FAN_CMF_RATING - 1.0))


@pytest.mark.external
def test_three_zone_steady_flows_match_contamx(contamx, record_property):
    from noodl.apps.building_physics.contamx import run_steady

    p = read_prj(THREE)
    amb = dict(p.ambient_conditions, mf={0: THREE_AMBIENT_MF})
    ref = run_steady(THREE, ambient=amb, airflow_convergence=TIGHT_AIRFLOW)
    model, state, drivers = project_to_model(p, ambient=amb)
    ss = steady(model, state, drivers, atol=NOODL_ATOL)
    ours = p.path_flows(ss["air.q"])
    # Isothermal and wind-driven: each flow is a product of powers (exponents <= 1) of about
    # ten single-precision inputs -- flow coefficients, wind speed, Cp, R, T -- with no
    # difference to amplify them, so <= ~10 F32_U first-order; the budget is 16 F32_U
    # (9.5e-7). Measured 9.1e-8.
    tol = 16 * F32_U
    torch.testing.assert_close(ours, ref["flow"], rtol=tol, atol=1e-12)
    worst_rel = ((ours - ref["flow"]) / ref["flow"].abs().clamp_min(1e-12)).abs().max().item()
    record_property("check", "Three-zone project, steady flows")
    record_property("tolerance", f"rel {tol:.2e} (16 float32 roundoffs)")
    record_property("measured_rel", worst_rel)


@pytest.mark.external
def test_three_zone_transient_concentrations_match_contamx(contamx, record_property):
    from noodl.apps.building_physics.contamx import run_transient

    p = read_prj(THREE)
    amb = dict(p.ambient_conditions, mf={0: THREE_AMBIENT_MF})
    steps = 24
    ref = run_transient(THREE, steps=steps, ambient=amb, airflow_convergence=TIGHT_AIRFLOW)
    assert ref["dt"] == pytest.approx(300.0)
    model, state, drivers = project_to_model(p, ambient=amb, scheme="implicit")
    drivers["species.x_boundary"] = torch.tensor([[THREE_AMBIENT_MF]], dtype=F64)
    trace = [state["species.x"].clone()]
    for _ in range(steps):
        state = model.step(state, drivers, ref["dt"])
        trace.append(state["species.x"].clone())
    ours = torch.stack(trace)                                   # (steps+1, 3, 1)
    # Budget: each zone's rate is flow / (rho V); flows carry 16 F32_U (steady test above)
    # and rho V two more (R, T; V and dt are exact). Early in a transient the k-th of k zones
    # in series responds as the PRODUCT of k such rates, so a relative rate error reaches it
    # up to k times: 3 zones x 18 F32_U = 54, budgeted as 64 F32_U (3.8e-6). Measured
    # 2.0e-7. It was 6.5e-6 while noodl physics held zone air mass at RHO_0 V rather than ContamX's
    # Pb / (R T) V (2.2e-6 apart even at 293.15 K), amplified about threefold, as above.
    tol = 64 * F32_U
    torch.testing.assert_close(ours, ref["mf"], rtol=tol, atol=1e-15)
    worst_rel = ((ours - ref["mf"]) / ref["mf"].abs().clamp_min(1e-12)).abs().max().item()
    record_property("check", "Three-zone project, transient concentrations, 24 steps at 300 s")
    record_property("tolerance", f"rel {tol:.2e} (64 float32 roundoffs)")
    record_property("measured_rel", worst_rel)

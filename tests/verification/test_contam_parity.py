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
import math
import re
import shutil
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.prj import project_to_model, read_prj, steady
from noodl.apps.building_physics.thermal import R_AIR

DATA = Path(__file__).resolve().parents[1] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
# The one-zone stack geometry with no filter on any path. NIST's
# `test_OneZoneSsStack-UseApi.prj`, which attaches a constant-efficiency filter (`f# = 1`)
# removing 10% of sarin on path 1, is a parity fixture of its own
# (`test_a_filtered_path_matches_contamx` below).
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


@pytest.mark.external
def test_a_constant_wind_pressure_matches_contamx(contamx, tmp_path, record_property):
    """Path 1 of the three-zone project with no wind profile and wPset = 5 Pa (TN 1887
    section 16). Budget as the three-zone steady flows: 16 F32_U (5 Pa is exact in float32)."""
    from noodl.apps.building_physics.contamx import run_steady

    text = THREE.read_text()
    p1 = "   1    1  -1   1   3   0   3   0   0   0   1  30.000   0.000   1.500 1 0 0.1225 270"
    assert p1 in text
    variant = tmp_path / "wpset.prj"
    variant.write_text(text.replace(
        p1, "   1    1  -1   1   3   0   0   0   0   0   1  30.000   0.000   1.500 1 5 0.1225 270",
        1))
    p = read_prj(variant)
    amb = dict(p.ambient_conditions, mf={0: THREE_AMBIENT_MF})
    ref = run_steady(variant, ambient=amb, airflow_convergence=TIGHT_AIRFLOW)
    model, state, drivers = project_to_model(p, ambient=amb)
    ours = p.path_flows(steady(model, state, drivers, atol=NOODL_ATOL)["air.q"])
    tol = 16 * F32_U
    torch.testing.assert_close(ours, ref["flow"], rtol=tol, atol=1e-12)
    record_property("check", "Three-zone project, constant wind pressure 5 Pa on path 1")
    record_property("tolerance", f"rel {tol:.2e}")
    record_property("measured_rel", ((ours - ref["flow"]) / ref["flow"]).abs().max().item())


WTH_CTM = DATA / "test_OneZoneWthCtm.prj"
STACK_SCHEDULED = DATA / "test_OneZoneWthCtmStack.prj"
# ContamX's airflow tolerances for the whole-project runs. At `TIGHT_AIRFLOW` the engine
# stops the step in which the 12:05 ambient jump collapses the stack of
# `test_OneZoneWthCtmStack.prj` (0.127 kg/s -> 1.5e-5 kg/s net) with 7.6e-7 kg/s of its
# exchange flow unresolved; at these it agrees with its own later steps, and with noodl
# physics, to 1e-11 kg/s (measured; 1e-5 / 1e-6 leaves 1.1e-3 kg/s).
PROJECT_AIRFLOW = (1e-13, 1e-16)


def _schedules_as_contamx_holds(prj: Path, dest: Path) -> Path:
    """A copy of `prj` in `dest`, with every day-schedule value rounded to float32, plus the
    weather and contaminant files its run control names.

    ContamX holds a day schedule's values in single precision but not the weather file's
    temperatures: with the zone scheduled at 293.15 K and the weather at 293.15 K, the engine
    drives a steady 2.9078e-5 kg/s exchange through the two openings of
    `test_OneZoneWthCtmStack.prj` (12:05-13:00), which is exactly the stack flow of a zone at
    float32(293.15) = 293.15 - 6.1e-6 K in a 293.15 K ambient, and it vanishes when both are
    293.125 K, which float32 represents exactly (measured, ContamX 3.4.1.7). At zero
    temperature difference that roundoff sets the whole flow, so no relative budget can cover
    it; noodl physics is given the value ContamX holds instead.
    """
    import numpy as np

    dest.mkdir(parents=True, exist_ok=True)
    lines = prj.read_text().splitlines(keepends=True)
    in_schedules = False
    n_schedules = rewritten = 0
    for i, line in enumerate(lines):
        if "! day-schedules:" in line:
            in_schedules = True
            n_schedules = int(line.split()[0])
        elif line.strip() == "-999":
            in_schedules = False
        elif in_schedules:
            tok = line.split()
            if len(tok) == 2 and tok[0].count(":") == 2:
                value = float(np.float32(float(tok[1])))
                lines[i] = f" {tok[0]} {value!r}\n"
                rewritten += 1
    assert rewritten >= n_schedules, (
        f"{prj.name}: {n_schedules} day schedules but {rewritten} values rewritten")
    (dest / prj.name).write_text("".join(lines))
    from noodl.apps.building_physics.contamx import _named_files

    for name in _named_files(prj):
        (dest / name).write_text((prj.parent / name).read_text())
    return dest / prj.name


def _project_parity(prj, tmp_path, *, rtol_flow, rtol_mf=None, record_property, label,
                    species_floor=False, atol=NOODL_ATOL, atol_flow=0.0, ref=None,
                    mf_bound=None):
    """ContamX's `run_project` (at `PROJECT_AIRFLOW`, its airflow tolerances tightened so its
    own iteration stays below the budgets) against `simulate_project` on the same project,
    with the schedule values ContamX holds (`_schedules_as_contamx_holds`).

    The absolute floor of the mass fractions is 64 u times the largest mass fraction; with
    `species_floor` it is 64 u times each species' own largest (a column floor), so a
    species far smaller than another is still checked relative to its own size. The zones'
    mass fractions are compared (a duct network's junctions follow them in the species
    state), and the duct terminal flows when the project has any. `atol` is noodl physics'
    airflow residual tolerance (kg/s). `atol_flow` (kg/s) is added to the flows' floor (a
    budget that is absolute, not relative); `ref` is ContamX's run when the caller already
    made it. `mf_bound(project, out)`, when given, replaces `rtol_mf` and the floor: it
    returns each interior row's allowed absolute mass-fraction error, (N, n_i, K), derived
    on the run (`_mass_fraction_bound`)."""
    from noodl.apps.building_physics.contamx import run_project
    from noodl.apps.building_physics.simulation import simulate_project

    if ref is None:
        ref = run_project(prj, airflow_convergence=PROJECT_AIRFLOW)
    noodl_prj = _schedules_as_contamx_holds(prj, tmp_path / "noodl")
    project = read_prj(noodl_prj)
    out = simulate_project(project, atol=atol, rtol=1e-14)
    bound = None if mf_bound is None else project.zone_mass_fractions(mf_bound(project, out))
    out["species.x"] = project.zone_mass_fractions(out["species.x"])
    torch.testing.assert_close(out["time"], ref["time"], rtol=0, atol=1e-9)
    scale = ref["flow"].abs().max()
    torch.testing.assert_close(out["path_flow"], ref["flow"], rtol=rtol_flow,
                               atol=(64 * F32_U * scale).item() + atol_flow)
    mf_scale = ref["mf"].abs().max()
    if species_floor:
        mf_scale = ref["mf"].abs().flatten(0, -2).amax(0)              # (K,)
    assert out["species.x"].dtype == ref["mf"].dtype == F64
    assert out["species.x"].shape == ref["mf"].shape
    mf_err = (out["species.x"] - ref["mf"]).abs()
    if bound is None:
        mf_allowed = 64 * F32_U * mf_scale + rtol_mf * ref["mf"].abs()
    else:
        mf_allowed = bound
        record_property("mf_budget_rel_max",
                        (bound[ref["mf"] != 0] / ref["mf"][ref["mf"] != 0].abs()).max().item())
    assert bool((mf_err <= mf_allowed).all()), (
        f"mass fractions: worst error {(mf_err / mf_allowed).max().item():.3g} of allowed")
    compared = [("flow", out["path_flow"], ref["flow"], scale, rtol_flow, atol_flow, None),
                ("mf", out["species.x"], ref["mf"], mf_scale, rtol_mf, 0.0, bound)]
    if ref["terminal_flow"].numel():
        t_scale = ref["terminal_flow"].abs().max()
        torch.testing.assert_close(out["terminal_flow"], ref["terminal_flow"],
                                   rtol=rtol_flow,
                                   atol=(64 * F32_U * t_scale).item() + atol_flow)
        compared.append(("terminal_flow", out["terminal_flow"], ref["terminal_flow"],
                         t_scale, rtol_flow, atol_flow, None))
    record_property("check", label)
    mf_text = "mf derived on the run" if bound is not None else f"mf rel {rtol_mf:.2e}"
    record_property("tolerance", f"flow rel {rtol_flow:.2e}, {mf_text}, "
                                 f"atol 64 u x max" + (" per species" if species_floor else "")
                                 + (f" + {atol_flow:.2e} kg/s on flows" if atol_flow else ""))
    for name, ours, theirs, top, rtol, extra, allowed in compared:
        # The relative error of the entries carrying at least 0.1 % of the largest (below
        # that the floor governs: the isothermal hours' exchange flow is a float32 roundoff
        # of the zone temperature), and the largest error of any entry as a fraction of what
        # the tolerance allows it.
        floor = 64 * F32_U * top + extra
        err = (ours - theirs).abs()
        big = theirs.abs() > 1e-3 * top
        if not big.any():                       # identically zero (no source, clean air)
            record_property(f"measured_rel_{name}", "all zero")
            continue
        record_property(f"measured_rel_{name}", (err[big] / theirs[big].abs()).max().item())
        if allowed is None:
            allowed = floor + rtol * theirs.abs()
        share = torch.where(allowed > 0, err / allowed.clamp_min(1e-300), 0.0)
        record_property(f"measured_{name}_of_allowed", share.max().item())   # 0 of 0: exact
    return ref, out


@pytest.mark.external
def test_a_run_with_its_own_weather_and_ambient_files_matches_contamx(
        contamx, tmp_path, record_property):
    """`test_OneZoneWthCtm.prj` for one day at 300 s: stack-driven while the weather holds the
    ambient 20 K below the zone (00:00-06:00), then wind-driven from 270 and 90 degrees, with
    the ambient sarin from the `.ctm` switching off at 14:05 and on at 05:05 the next day.
    Budget, flows: 33 u is the SUM of the two drives' first-order budgets, the stack's at
    20 K (`_stack_budget(273.15)` = 17.2 u: C, rho_up, g, R and the two temperatures
    through their difference) and the wind's (16 u, the three-zone steady budget: C, wind
    speed, Cp, R, T as a product of powers <= 1). It bounds a flow driven by either alone
    and by both together.
    Mass fractions: the one zone's rate is flow / (rho V), so the flows' 33 u plus 2 u for
    rho V (R, T; V and dt exact), plus 1 u for the ambient mass fraction from the `.ctm`:
    36 u, with no series amplification (one zone); budgeted at 64 u.
    Absolute floor: where the flow is near zero (calm, isothermal hours) a relative budget
    has nothing to scale; a float32 roundoff of an input moves the flow by at most its
    relative budget (<= 64 u) times the flow the same openings carry under the run's
    largest drive, so 64 u times the largest flow; likewise 64 u times the largest mass
    fraction."""
    _project_parity(WTH_CTM, tmp_path, rtol_flow=33 * F32_U, rtol_mf=64 * F32_U,
                    record_property=record_property,
                    label="One-zone project with its own .wth and .ctm, 288 steps")


@pytest.mark.external
def test_a_scheduled_zone_temperature_run_matches_contamx(contamx, tmp_path,
                                                          record_property):
    """`test_OneZoneWthCtmStack.prj`: zone on a temperature schedule (293.15 -> 303.15 K at
    13:00), weather 273.15 / 293.15 / 263.15 K, `densZP = 1`, `dodMdt = 1`. The flows in the
    step after the jump are set by the zone's air-mass change (ContamX puts dM/dt into the
    balance); its budget is the mass difference's: M = Pb V / (R T) at two temperatures
    3.4 % apart amplifies a float32 roundoff of R and the two T by about 30, so 64 u;
    elsewhere the stack budget at the smallest nonzero temperature difference (10 K:
    32.8 u), and 64 u on the mass fractions. From 12:05 to 13:00 zone and weather are both
    293.15 K and the only flow is the exchange set by ContamX's float32 schedule value (see
    `_schedules_as_contamx_holds`); without that value given to noodl physics the mass
    fractions drift 1.4e-4 apart over the hour."""
    ref, out = _project_parity(STACK_SCHEDULED, tmp_path, rtol_flow=64 * F32_U,
                               rtol_mf=64 * F32_U,
                               record_property=record_property,
                               label="One-zone stack with a scheduled zone temperature")
    jump = 157                                         # the step ending 13:05
    torch.testing.assert_close(out["path_flow"][jump].sum(), ref["flow"][jump].sum(),
                               rtol=64 * F32_U, atol=0)


@pytest.mark.external
@pytest.mark.slow
def test_a_two_day_run_across_midnight_matches_contamx(contamx, tmp_path, record_property):
    """The scheduled stack project run to the end of 1/2 (its `.wth`/`.ctm` cover both days):
    the calendar, the weather's day 2 and the left limit at midnight (the previous day's
    value until the first step after it). The weather file has no 00:00 row on 1/2: its
    1/1 24:00 row (263.15 K) is followed by 1/2 05:00 (293.15 K), and ContamX interpolates
    linearly across midnight between the two, as the reader does (measured: every step from
    00:05 to 05:00 of 1/2 inside the stack budget)."""
    for suffix in (".prj", ".wth", ".ctm"):
        src = STACK_SCHEDULED.with_suffix(suffix)
        text = src.read_text()
        if suffix == ".prj":
            one_day = "Jan01 00:00:00  Jan01 00:00:00  Jan01 24:00:00"
            assert one_day in text
            text = text.replace(one_day, "Jan01 00:00:00  Jan01 00:00:00  Jan02 24:00:00", 1)
        (tmp_path / src.name).write_text(text)
    _project_parity(tmp_path / STACK_SCHEDULED.name, tmp_path, rtol_flow=64 * F32_U,
                    rtol_mf=64 * F32_U, record_property=record_property,
                    label="Scheduled stack project over two days")


def test_a_project_naming_its_weather_file_with_a_directory_is_refused(tmp_path):
    from noodl.apps.building_physics.contamx import _named_files

    text = STACK_SCHEDULED.read_text()
    assert "test_OneZoneWthCtmStack.wth ! weather file" in text
    variant = tmp_path / "subdir.prj"
    variant.write_text(text.replace("test_OneZoneWthCtmStack.wth ! weather file",
                                    "weather/test_OneZoneWthCtmStack.wth ! weather file", 1))
    assert _named_files(STACK_SCHEDULED) == ["test_OneZoneWthCtmStack.wth",
                                             "test_OneZoneWthCtmStack.ctm"]
    with pytest.raises(ValueError, match=r"weather/test_OneZoneWthCtmStack\.wth.*bare file"):
        _named_files(variant)


@pytest.mark.external
def test_run_project_refuses_a_run_that_is_not_whole_time_steps(contamx, tmp_path):
    from noodl.apps.building_physics.contamx import run_project

    for suffix in (".prj", ".wth", ".ctm"):
        src = STACK_SCHEDULED.with_suffix(suffix)
        text = src.read_text()
        if suffix == ".prj":
            assert "Jan01 24:00:00  00:05:00" in text
            text = text.replace("Jan01 24:00:00  00:05:00", "Jan01 23:58:00  00:05:00", 1)
        (tmp_path / src.name).write_text(text)
    with pytest.raises(ValueError, match="not a whole number"):
        run_project(tmp_path / STACK_SCHEDULED.name)


SCHEDULED = DATA / "scheduled_three_zone.prj"


@pytest.mark.external
@pytest.mark.slow
def test_the_scheduled_three_zone_project_matches_contamx(contamx, tmp_path, record_property):
    """`scheduled_three_zone.prj`: a scheduled exhaust fan, sarin source, zone-2 temperature
    and window over 24 h, with diurnal weather (Ta, Pb, Ws, Wd) and an ambient sarin pulse,
    `dodMdt = 1`, `densZP = 0`. Budgets: flows 64 u (wind- and stack-driven flows, 16-33 u
    as in the rows above, and the mass-change steps of the zone-2 temperature ramps, 64 u as
    in the scheduled-zone-temperature test); mass fractions 64 u (the flows' budget, plus 2 u
    for rho V and 1 u for the ambient mass fraction, inside 64 u), each with the 64 u x max
    floor of `_project_parity`. The fan's flow is its rating times the schedule, exact in
    float64 (float32(0.100342) in ContamX).
    Two ContamX rules this run measured, without which it failed by 1e-3 relative: the wind
    pressure profiles of NIST's sample are curve fit 2, a periodic cubic spline in the wind
    angle (linear interpolation was 7.3e-5 kg/s off once the wind left 270 degrees), and the
    zone air mass of `dodMdt` is (Pb + P_zone) V / (R T) even with `densZP = 0`, in the
    airflow balance (5.9e-6 kg/s when the fan switches on) and for the contaminants (4.4e-9
    in the mass fractions when the window reopens)."""
    ref, out = _project_parity(SCHEDULED, tmp_path, rtol_flow=64 * F32_U, rtol_mf=64 * F32_U,
                               record_property=record_property,
                               label="Scheduled three-zone project, 24 h")
    assert out["path_flow"][95, 4].item() == 0.0                       # 07:55, fan off
    assert out["path_flow"][97, 4].item() == pytest.approx(0.100342, rel=1e-12)   # 08:05


REACTION = DATA / "reaction_three_zone.prj"


@pytest.mark.external
def test_kinetic_reactions_match_contamx(contamx, tmp_path, record_property):
    """`reaction_three_zone.prj`: two species, a three-coefficient reaction in zone one and a
    two-coefficient one in zone three, wind-driven flows, 24 steps of 300 s.

    Budget: the three-zone transient's 64 u (flows through three zones in series) plus the
    reaction coefficients, each a float32 input: an implicit decay over n steps responds to
    a relative error u in k as n k dt / (1 + k dt) u, at most 24 * 0.3 / 1.3 = 5.5 u for the
    fastest coefficient (1e-3/s); the product depends on three such coefficients, 5.5 + 5.5
    + 3.1 = 14 u. Total 80 u. The absolute floor is per species (64 u times the species'
    own largest mass fraction)."""
    _project_parity(REACTION, tmp_path, rtol_flow=16 * F32_U, rtol_mf=80 * F32_U,
                    record_property=record_property, species_floor=True,
                    label="Three-zone project with kinetic reactions in two zones, 24 steps")


SS_STACK = DATA / "test_OneZoneSsStack-UseApi.prj"
SS_STACK_AMBIENT_MF = 1e-3
FILTERS = DATA / "filters_three_zone.prj"


@pytest.mark.external
@pytest.mark.parametrize("Ta", (273.15, 313.15))
def test_a_filtered_path_matches_contamx(contamx, Ta, record_property):
    """NIST's `test_OneZoneSsStack-UseApi.prj`: steady airflow and contaminants, a 1e-4 kg/s
    sarin source, and a filter removing 10 % of the sarin crossing path 1 (the low opening,
    declared ambient -> zone), with 1e-3 kg/kg sarin outside. At 273.15 K the air enters
    through the filter; at 313.15 K it leaves through it and enters through path 2.

    Budget: x = ((1 - eta) F x_amb + G) / F. A relative roundoff reaches x through F (the
    stack budget at this ambient, `_stack_budget(Ta)`), through x_amb and G (1 u each) and
    through 1 - eta (u eta / (1 - eta) = 0.11 u): `_stack_budget(Ta) + 3 u`."""
    from noodl.apps.building_physics.contamx import run_steady

    p = read_prj(SS_STACK)
    amb = dict(p.ambient_conditions, Ta=Ta, mf={0: SS_STACK_AMBIENT_MF})
    ref = run_steady(SS_STACK, ambient=amb, airflow_convergence=TIGHT_AIRFLOW)
    model, state, drivers = project_to_model(p, ambient=amb)
    drivers["species.x_boundary"] = torch.tensor([[SS_STACK_AMBIENT_MF]], dtype=F64)
    ours = steady(model, state, drivers, atol=NOODL_ATOL)["species.x"]
    tol = _stack_budget(Ta) + 3 * F32_U
    torch.testing.assert_close(ours, ref["mf"], rtol=tol, atol=0)
    record_property("check", f"One-zone stack with a filtered path, steady, Ta = {Ta} K")
    record_property("tolerance", f"rel {tol:.2e}")
    record_property("measured_rel", ((ours - ref["mf"]) / ref["mf"]).abs().max().item())


@pytest.mark.external
def test_three_filters_match_contamx(contamx, tmp_path, record_property):
    """`filters_three_zone.prj`: sarin and 2 um particles entering a wind-driven three-zone
    building through a constant-efficiency filter, a particle filter crossed against its
    path's direction, and both in series; 24 steps of 300 s.

    Budget: the three-zone transient's 64 u, plus each pass fraction 1 - eta, which a
    roundoff in eta reaches as u eta / (1 - eta): 0.43 u (0.3), 1.9 u (0.65, a point of the
    particle curve), 2.3 u (both); along the chain to zone three at most 4.6 u, and 1 u for
    each ambient value: 72 u. The absolute floor is per species (64 u times the species'
    own largest mass fraction)."""
    _project_parity(FILTERS, tmp_path, rtol_flow=16 * F32_U, rtol_mf=72 * F32_U,
                    record_property=record_property, species_floor=True,
                    label="Three-zone project with constant-efficiency, particle and "
                          "series filters, 24 steps")


SINKS = DATA / "sinks_three_zone.prj"
PRESSURE_SOURCE = DATA / "pressure_source_stack.prj"


@pytest.mark.external
@pytest.mark.filterwarnings("ignore::noodl.apps.building_physics.prj.ContamCompatibilityWarning")
def test_sinks_and_emission_models_match_contamx(contamx, tmp_path, record_property):
    """`sinks_three_zone.prj`: a deposition velocity sink, a deposition rate sink on a
    schedule, a constant-coefficient source with a removal term, a cutoff source, NRCC
    power-law and peak sources, and a species decay constant ContamX does not apply; wind-
    driven, 24 steps of 300 s.

    Budget: the three-zone transient's 64 u, plus the sink rates, which a relative
    roundoff reaches over the run as k t u: at most 0.48 (deposition velocity, 2 inputs),
    2.7 (deposition rate) and 0.33 (removal term, R and the air mass): 4.6 u; plus the
    emissions, 1 u each in a, 0.35 u in the power law's exponent, and 2.1 u (t_p) and
    4.3 u (b) in the peak model: 80 u. The absolute floor is per species (64 u times the
    species' own largest mass fraction)."""
    _project_parity(SINKS, tmp_path, rtol_flow=16 * F32_U, rtol_mf=80 * F32_U,
                    record_property=record_property, species_floor=True,
                    label="Three-zone project with deposition sinks, removal terms and "
                          "NRCC emissions, 24 steps")


@pytest.mark.external
@pytest.mark.filterwarnings("ignore::noodl.apps.building_physics.prj.ContamCompatibilityWarning")
def test_a_constant_coefficient_source_scales_with_its_multiplier(contamx, tmp_path,
                                                                  record_property):
    """`sinks_three_zone.prj` with its constant-coefficient source's multiplier set to 2
    (measured on ContamX: the multiplier scales the generation and the removal term alike,
    the removal one implicitly, as for a multiplier of 1).

    Budget: as for the sinks fixture, the removal term's 0.33 u unchanged, a multiplier of 2
    being exact in float32: 80 u."""
    text = SINKS.read_text()
    old = "  3   3   3   0   0     1     0"
    assert old in text
    doubled = tmp_path / SINKS.name
    doubled.write_text(text.replace(old, "  3   3   3   0   0     2     0", 1))
    _project_parity(doubled, tmp_path, rtol_flow=16 * F32_U, rtol_mf=80 * F32_U,
                    record_property=record_property, species_floor=True,
                    label="Three-zone project with a constant-coefficient source at "
                          "multiplier 2, 24 steps")


@pytest.mark.external
@pytest.mark.slow
def test_a_pressure_driven_source_matches_contamx(contamx, tmp_path, record_property):
    """`pressure_source_stack.prj`: NIST's scheduled one-zone stack (weather and ambient
    files, zone schedule, densZP, dodMdt) with a pressure-driven source, 288 steps; the
    zone-to-ambient pressure difference changes with every weather and schedule jump.

    Budget: the scheduled stack's 64 u (`test_a_scheduled_zone_temperature_run_matches_
    contamx`), plus the source: S = G dP^n, and dP is a stack pressure, a density difference
    whose budget at the smallest nonzero temperature difference (10 K) is 32.8 u, so
    n 32.8 u = 21.3 u, and 1 u in G: 88 u. The absolute floor is per species."""
    _project_parity(PRESSURE_SOURCE, tmp_path, rtol_flow=64 * F32_U, rtol_mf=88 * F32_U,
                    record_property=record_property, species_floor=True,
                    label="One-zone stack with a pressure-driven source, 288 steps")


FAN_CURVE = DATA / "fan_curve_three_zone.prj"


@pytest.mark.external
@pytest.mark.slow
def test_a_scheduled_fan_curve_matches_contamx(contamx, tmp_path, record_property):
    """`fan_curve_three_zone.prj`: an exhaust fan curve (mult 1.5) from zone three at 303.15
    K on a speed schedule 1, 0.8, 0.05 (below its 0.1 off ratio: the shut-off orifice) and
    1.2; two hours at 300 s, wind 5.23 m/s from 270 degrees. Budget: the three-zone steady
    flows' 16 u for the orifice paths and the shut-off orifice, plus the fan's own 4 u (its
    coefficients, rating density and the zone temperature through r), doubled for the
    coupling of the fan's operating point with the zone pressure: 32 u on the flows; the
    three-zone transient's 64 u on the mass fractions. The absolute floor is per species."""
    _project_parity(FAN_CURVE, tmp_path, rtol_flow=32 * F32_U, rtol_mf=64 * F32_U,
                    record_property=record_property, species_floor=True,
                    label="Three-zone project with a scheduled fan curve, 24 steps")


SPLINES = DATA / "spline_three_zone.prj"


@pytest.mark.external
def test_the_four_spline_elements_match_contamx(contamx, tmp_path, record_property):
    """`spline_three_zone.prj`: the four cubic-spline elements on the three-zone paths, one
    crossed against its declared direction (the first segment extended to negative
    pressures), zone two at 303.15 K, a 0.02 kg/s supply. Budget: a float32 roundoff in
    each curve point moves a value inside a segment by at most 2 u, the zone densities (the
    volume forms) 1 u, the supply 1 u, and the wind pressures as in the three-zone steady
    test, 16 u: 32 u on the flows, 64 u on the mass fractions."""
    _project_parity(SPLINES, tmp_path, rtol_flow=32 * F32_U, rtol_mf=64 * F32_U,
                    record_property=record_property, species_floor=True,
                    label="Three-zone project with the four cubic-spline elements, 24 steps")


DUCTS = DATA / "duct_three_zone.prj"
# A duct junction's air mass is Pb V / (R T) with V its share of the segments' length x area:
# the float32 area (1 u) and length (1 u; 0.7 m is not exact), their product and the sum of
# the halves (u / 2 each): 3 u in V; R and T 2 u more. A zone's (and a plenum's) volume and
# Pb are exact: R and T, 2 u.
JUNCTION_MASS_U, ZONE_MASS_U = 5, 2


def _mass_roundoff(project, model):
    """Each interior row's relative air-mass error in ContamX's float32 inputs, (n_i,)."""
    lay = model.transport["species"]
    junctions = set(project.junctions)
    return torch.tensor([(JUNCTION_MASS_U if model.net.nodes[i] in junctions
                          else ZONE_MASS_U) * F32_U for i in lay.interior_idx.tolist()],
                        dtype=F64)


def _mass_fraction_bound(eps_flow, pass_fractions=None):
    """`_project_parity`'s `mf_bound` for a project whose air mass is constant and whose only
    species input is the ambient mass fraction (no source, clean start): the first-order
    bound of the mass-fraction error of noodl physics' implicit step against ContamX's, run
    through every step on noodl physics' own run.

    Each step solves B x_k = (M / dt) x_(k-1) + N x_b with B = M / dt + A, A holding each
    node's outflow on its diagonal and the transmitted inflows -eta q off it (eta: the
    filters' pass fraction on the edge). With dq, deta, dM, dx_b the input errors, to first
    order B dx_k = (M / dt) dx_(k-1) - dM / dt (x_k - x_(k-1)) - dA x_k + N dx_b. Both runs
    conserve mass at every interior node, so the flow errors do too, and row i of dA x_k is
    sum over the edges into i of dq (x_i - eta x_up) - deta q x_up. B is an M-matrix (a
    Z-matrix whose columns are diagonally dominant: what leaves a node arrives at most whole
    at the others), so B^-1 >= 0 and, step by step from dx_0 = 0 (a clean start),
    |dx_k| <= B^-1 [(M / dt) |dx_(k-1)| + |dM| / dt |x_k - x_(k-1)|
                    + sum_in |dq| |x_i - eta x_up| + sum_in |deta| q x_up
                    + sum_in from outdoors eta q |dx_b|].
    The bound follows a closed loop (the AHS's return air, a return duct) through B^-1
    rather than assuming a chain. Its terms: |dq| = `eps_flow` x the largest flow the edge
    carries over the run (the flow budget, its floor read edge by edge); |dM| =
    `_mass_roundoff` x M; |deta| = 2 u on a filtered edge (the float32 efficiency, at most
    u, the 1 - efficiency rounding, u / 2, and a second filter chained on the edge, u / 2
    more), 0 on an unfiltered one (eta = 1 exactly); |dx_b| = u |x_b| (the ambient mass
    fraction in float32). The preconditions (no species source, no generating occupant, no
    kinetic reaction, a zero initial mass fraction) are asserted, each naming the term it
    would need. `pass_fractions(project, model, drivers, out)`, when given, returns the pass
    fractions and their errors per step, two `(N, K, E)` tensors in the species layer's flow
    order, for filters whose pass fraction changes with the flows (`_uvgi_pass_fractions`);
    by default they are the layer's fixed transmission and the 2 u above."""

    def bound(project, out):
        assert not project.sources, (
            "the bound has no source term: the project's species sources would need one")
        assert not any(o.generation for o in project.occupants), (
            "the bound has no source term: the occupants' generation would need one")
        assert not project.zone_reactions, (
            "the bound has no kinetics term: the zones' reactions would need one")
        assert bool((out["species.x"][0] == 0).all()), (
            "the bound starts from dx_0 = 0: a nonzero initial mass fraction would need its "
            "float32 roundoff term")
        model, _, drivers = project_to_model(project)
        lay, net = model.transport["species"], model.net
        f = model.kirchhoff["air"].flows_of_kind(out["air.q"], lay.flow_kinds)
        src, tgt = zip(*(net.endpoints(k) for k in lay.flow_kinds), strict=True)
        src, tgt = torch.cat(src).tolist(), torch.cat(tgt).tolist()
        n_rows = out["species.x"].shape[0]
        if pass_fractions is None:
            eta = lay.transmission.to(F64).expand(lay.n_species, len(src))   # (K, E)
            deta = torch.where(eta < 1, 2 * F32_U, 0.0)
            eta, deta = eta.expand(n_rows, -1, -1), deta.expand(n_rows, -1, -1)
        else:
            eta, deta = pass_fractions(project, model, drivers, out)        # (N, K, E)
        dq = eps_flow * f.abs().amax(0)
        x = out["species.x"]                                                 # (N, n_i, K)
        cap = out["species.capacity"]
        assert bool((cap == cap[0]).all()), "the bound assumes a constant air mass"
        mass = torch.where(lay.storage_free_mask, 0.0, cap[0])
        dmass = _mass_roundoff(project, model) * mass
        xb = drivers["species.x_boundary"].to(F64).reshape(lay.n_b, lay.n_species)
        bpos = {int(b): j for j, b in enumerate(lay.boundary_idx.tolist())}
        pos = {node: i for i, node in enumerate(lay.interior_idx.tolist())}
        times = out["time"].tolist()
        dx = torch.zeros_like(x)
        for k in range(1, x.shape[0]):
            dt = times[k] - times[k - 1]
            b = torch.diag(mass / dt).expand(lay.n_species, -1, -1).clone()   # (K, n, n)
            rhs = (mass / dt)[:, None] * dx[k - 1] + (dmass / dt)[:, None] * (
                x[k] - x[k - 1]).abs()

            def upstream(node, k=k):
                return x[k, pos[node]] if node in pos else xb[bpos[node]]

            eta_k, deta_k = eta[k], deta[k]

            for e, q in enumerate(f[k].tolist()):
                ends = ([(src[e], tgt[e])] if q > 0 else [(tgt[e], src[e])] if q < 0
                        else [(src[e], tgt[e]), (tgt[e], src[e])])   # zero: either way
                for up, down in ends:
                    if q != 0 and up in pos:
                        b[:, pos[up], pos[up]] += abs(q)
                        if down in pos:
                            b[:, pos[down], pos[up]] -= eta_k[:, e] * abs(q)
                    if down not in pos:
                        continue
                    x_up = upstream(up)
                    term = dq[e] * (x[k, pos[down]] - eta_k[:, e] * x_up).abs()
                    term = term + deta_k[:, e] * abs(q) * x_up.abs()
                    if up not in pos:
                        term = term + eta_k[:, e] * abs(q) * F32_U * x_up.abs()
                    rhs[pos[down]] += term
            dx[k] = torch.linalg.solve(b, rhs.T.unsqueeze(-1)).squeeze(-1).T
        return dx

    return bound


@pytest.mark.external
@pytest.mark.slow
def test_a_duct_network_matches_contamx(contamx, tmp_path, record_property):
    """`duct_three_zone.prj` (fixture NOTICE): wind terminal, filtered Colebrook duct, fan
    curve on a speed schedule, vertical rise, split to two terminals (one half closed from
    01:00), damper and constant-volume exhaust; 24 steps.

    Budget, flows: ContamX evaluates the duct and terminal laws in single precision, 32 u in
    a pressure drop at a fixed flow (the element tests derive it); each duct flow is set by a
    balance of several such drops and the zone pressures, which at most doubles it: 64 u.
    (An independent model built from the measured rules agreed with ContamX's first step to
    32 u.) Mass fractions: `_mass_fraction_bound` with these flows' 64 u, derived on the
    run (the implicit step's first-order error under the flows', filters', air masses' and
    ambient mass fraction's float32 roundoff, through B^-1 >= 0)."""
    _project_parity(DUCTS, tmp_path, rtol_flow=64 * F32_U,
                    mf_bound=_mass_fraction_bound(64 * F32_U),
                    record_property=record_property, species_floor=True,
                    label="Three-zone project with a duct network, 24 steps")

UVGI = DATA / "uvgi_three_zone.prj"


def _pass_fraction_error(flt, species, air, eps_flow):
    """`(tau, dtau)`, each `(K,)`: a filter's pass fraction for the air passing and its
    first-order error under ContamX's float32 inputs and the flows' budget `eps_flow`.

    A fixed filter: 2 u (as in `_mass_fraction_bound`). Filters in series: the product rule.
    A UVGI filter, tau = exp(-X), X = k I_t: dtau = tau X eps_X, with eps_X the relative
    error of X, the sum of its factors' (`UVGIFilter`'s formulae):
    k, k_design, F_design, rho_std: 1 u each; ln(S_design): u / |ln S_design|; rho = Pb / (R T)
    (Pb exact in float32): 1 u through T; 1 / F: `eps_flow`; and OUT. OUT_TU = v / 100 off
    its clamps, v = C0 + C1 T + C2 U + C3 T^2 + C4 U^2 (T in deg C): each term's coefficient
    1 u, T through T_K (the kelvin value is the float32 input: |dv/dT| u T_K) and U through
    eps_U = `eps_flow` + 4 u (F_design, u_design, rho, rho_std), |dv/dU| U eps_U, over v; 0 on
    a clamp. OUT_age = a off its clamp, a = (K0 + K1 ln t) / 100: u (|K0| + |K1 ln t|) / |100 a|
    (t exact); 0 on the clamp or the 0.1 fallback. Against the path's direction X = k (dose
    1 J/m2): eps_X = u."""
    from noodl.apps.building_physics.filters import FilterSeries, UVGIFilter

    u = F32_U
    if isinstance(flt, FilterSeries):
        parts = [_pass_fraction_error(s, species, air, eps_flow) for s in flt.stages]
        tau = torch.stack([p[0] for p in parts]).prod(0)
        dtau = torch.zeros_like(tau)
        for i, (_, d) in enumerate(parts):
            others = torch.ones_like(tau)
            for j, (tj, _) in enumerate(parts):
                if j != i:
                    others = others * tj
            dtau = dtau + d * others
        return tau, dtau
    tau = 1.0 - flt.efficiency(species, air)
    if not isinstance(flt, UVGIFilter):
        return tau, torch.where(tau < 1, 2 * u, 0.0)
    X = -torch.log(tau)
    F = float(air.mass_flow)
    if F < 0:
        return tau, tau * X * u
    T_K = float(air.temperature)
    Tc = T_K - 273.15
    rho = float(air.density)
    U = F * 1.2041 * flt.design_velocity.item() / (rho * flt.design_mass_flow.item())
    c0, c1, c2, c3, c4 = flt.output_coefficients.tolist()
    v = c0 + c1 * Tc + c2 * U + c3 * Tc * Tc + c4 * U * U
    eps_out = 0.0
    if 0 < v < 100:
        eps_u = eps_flow + 4 * u
        dv = u * (abs(c0) + abs(c1 * Tc) + abs(c2 * U) + abs(c3) * Tc * Tc + abs(c4) * U * U)
        dv += abs(c1 + 2 * c3 * Tc) * u * T_K + abs(c2 + 2 * c4 * U) * U * eps_u
        eps_out = dv / v
    if flt.age_coefficients is not None:
        k0, k1 = flt.age_coefficients.tolist()
        hours = float(air.time) / 3600.0
        a = (k0 + k1 * math.log(hours)) / 100.0 if hours > 0 else None
        if a is not None and 0 < a < 1:
            eps_out += u * (abs(k0) + abs(k1 * math.log(hours))) / abs(100 * a)
    S = flt.design_survival.item()
    eps_x = u * (4 + 1 / abs(math.log(S))) + u + eps_flow + eps_out
    return tau, tau * X * eps_x


def _uvgi_pass_fractions(eps_flow):
    """`_mass_fraction_bound`'s `pass_fractions` for a project with UVGI filters: each
    step's pass fractions from the run's own flows, at the drivers' temperatures and
    densities and the step's time, and their errors (`_pass_fraction_error`). The project's
    temperatures and pressure must be constant (asserted: no input writes them)."""
    from noodl.apps.building_physics.filters import AirThroughFilter, FilterTransmission

    def pass_fractions(project, model, drivers, out):
        assert not any(i.key in ("T", "P_ref") for i in project.inputs), (
            "the pass fractions are taken at the drivers' constant temperatures and pressure")
        (closure,) = [c for c in model.closures if isinstance(c, FilterTransmission)]
        air = model.kirchhoff["air"]
        species = closure.species
        rho = drivers["P_ref"] / (R_AIR * drivers["T"])
        taus, dtaus = [], []
        for k, t in enumerate(out["time"].tolist()):
            f = air.flows_of_kind(out["air.q"][k], closure.flow_kinds)
            rows = [(r, torch.where(r < 1, 2 * F32_U, 0.0)) for r in closure.fixed]
            for row, src, flt in closure.varying:
                passing = AirThroughFilter(f[row], drivers["T"][src], rho[src], t)
                rows[row] = _pass_fraction_error(flt, species, passing, eps_flow)
            taus.append(torch.stack([r[0] for r in rows], dim=-1))          # (K, E)
            dtaus.append(torch.stack([r[1] for r in rows], dim=-1))
        return torch.stack(taus), torch.stack(dtaus)

    return pass_fractions


@pytest.mark.external
def test_uvgi_filters_match_contamx(contamx, tmp_path, record_property):
    """`uvgi_three_zone.prj` (fixture NOTICE): two species with UV susceptibilities 1e-3 and
    2e-2 m2/J entering a wind-driven three-zone building through a UVGI filter with an age
    model on a path whose flow a schedule cuts to 0.4 for 45 minutes, a UVGI filter in series
    with a constant-efficiency filter, and a UVGI filter crossed against its path's direction
    (ContamX's fixed 1 J/m2 dose, warned once); 24 steps of 300 s. The filters' pass fractions
    change with every airflow solve; the iterated coupling gives each step its own flows (with
    one pass, the transmission lags the flows by a step: 6 % in the mass fractions after the
    schedule's cut, measured).

    Budget, flows: the three-zone steady flows' 16 u (`test_three_filters_match_contamx`;
    the schedule's 0.4 is given to noodl physics as ContamX holds it). Mass fractions:
    `_mass_fraction_bound` with those flows, derived on the run, its pass fractions and their
    errors taken per step from the run's flows (`_uvgi_pass_fractions`): a UVGI pass
    fraction exp(-X) moves by exp(-X) X eps_X, eps_X the first-order float32 roundoff of the
    dose's inputs plus the flows' 16 u (`_pass_fraction_error`)."""
    import warnings

    from noodl.apps.building_physics.prj import ContamCompatibilityWarning

    eps_flow = 16 * F32_U
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _project_parity(UVGI, tmp_path, rtol_flow=eps_flow,
                        mf_bound=_mass_fraction_bound(eps_flow, _uvgi_pass_fractions(eps_flow)),
                        record_property=record_property, species_floor=True,
                        label="Three-zone project with UVGI filters, 24 steps")
    reverse = [str(w.message) for w in caught
               if issubclass(w.category, ContamCompatibilityWarning)]
    assert len(reverse) == 1 and "'uvDesign'" in reverse[0], reverse


AHS = DATA / "ahs_three_zone.prj"
NIST_AHS = DATA / "test_GetPrjInfo.prj"
SARIN_LINE = ("  1 1 0 140.0900  0.0000e+00  0.0000e+00  0.0000e+00  2.0000e-05  0.0000e+00"
              "  1.0000e+03")


@pytest.mark.external
def test_a_simple_ahs_matches_contamx(contamx, tmp_path, record_property):
    """`ahs_three_zone.prj` (fixture NOTICE). Budget: the AHS flows are sums and minima of
    float32 design flows and a float32 fraction (4 u); the envelope flows the three-zone
    steady 16 u: 32 u. Mass fractions: `_mass_fraction_bound` with these flows' 32 u,
    derived on the run; it follows the return air's closed loop through B^-1 >= 0 and
    counts the filters' pass fractions (2 u each filtered edge)."""
    _project_parity(AHS, tmp_path, rtol_flow=32 * F32_U,
                    mf_bound=_mass_fraction_bound(32 * F32_U),
                    record_property=record_property, species_floor=True,
                    label="Three-zone project with a simple AHS, 24 steps")


@pytest.mark.external
@pytest.mark.slow
def test_nist_s_ahs_and_duct_sample_matches_contamx(contamx, tmp_path, record_property):
    """NIST's `test_GetPrjInfo.prj` with zone 1's and segment 4's control references and the
    duct leakage removed, one hour, sarin outdoors at 1e-3: two simple AHS with zero-volume
    plenums (100 % outdoor air) and two fan-driven duct systems. Budget as the duct test:
    flows 64 u, mass fractions `_mass_fraction_bound` of that. All outdoor air, the zones
    approach the ambient 1e-3, where the bound is the ambient value's own float32 roundoff
    (u |x_b|) and ContamX's float32(1e-3) is 0.797 u off it: this row uses 80 % of the
    bound, by that one term.

    The airflow residual tolerance is 1e-12 kg/s, not 1e-14: each duct terminal carries
    3.3e-4 kg/s across a 5.8e-6 Pa drop (conductance 28 kg/(s Pa)) between junction and zone
    pressures of 2-30 Pa, so the last-place units of the two pressures (2 x 3.6e-15 Pa)
    move a node's balance by its conductance times that: 57 kg/(s Pa) at zone Three (the
    terminals of junctions 4 and 8), 64 at zone Two (the terminal of junction 6 and the
    nearly idle envelope path 12), so up to 64 x 7.1e-15 = 4.6e-13 kg/s: the float64
    floor. 1e-12 kg/s is 3e-9 of the smallest flow, far inside the budgets."""
    text = NIST_AHS.read_text()
    for old, new in (
        ("   1  3   0   3   0   1   0.000   600 293.15 0 One",
         "   1  3   0   0   0   1   0.000   600 293.15 0 One"),
        ("  4  0   5   6   2   0   0   5   2 5", "  4  0   5   6   2   0   0   0   2 5"),
        ("  Jan01 00:00:00  Jan01 00:00:00  Jan02 24:00:00",
         "  Jan01 00:00:00  Jan01 00:00:00  Jan01 01:00:00"),
        (SARIN_LINE, SARIN_LINE.replace("0.0000e+00  1.0000e+03", "1.0000e-03  1.0000e+03")),
    ):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    leak = " 0.2 0.628319 0.0314159 0.2 0.2 0 0.5 1\n"
    assert text.count(leak) == 1, leak
    text = text.replace(leak, " 0.2 0.628319 0.0314159 0.2 0.2 0 0 1\n")
    variant = tmp_path / "variant" / NIST_AHS.name
    variant.parent.mkdir()
    variant.write_text(text)
    _project_parity(variant, tmp_path, rtol_flow=64 * F32_U,
                    mf_bound=_mass_fraction_bound(64 * F32_U),
                    record_property=record_property, species_floor=True,
                    label="NIST test_GetPrjInfo (two AHS, two duct systems), 12 steps",
                    atol=1e-12)


DCV = DATA / "dcv_three_zone.prj"
PRESSURE_EXHAUST = DATA / "pressure_exhaust_three_zone.prj"
WINDOW = DATA / "hysteresis_window_three_zone.prj"


def _control_outputs_match(ref, out, rtol, record_property, label, atol=0.0):
    """Every named output within `rtol` relative plus `atol` absolute (either may be per
    column), with a floor of 64 u times the column's largest value (the outputs differ in
    scale by six orders: a CO2 reading in mg/kg and a fan signal). A column that is zero
    throughout with nothing allowed must be matched exactly."""
    want, got = ref["control"], out["control"]
    assert got.shape == want.shape
    allowed = rtol * want.abs() + 64 * F32_U * want.abs().amax(0) + atol
    err = (got - want).abs()
    exact = allowed == 0
    assert bool((err[exact] == 0).all()), f"{label}: an all-zero output is not zero"
    share = torch.where(exact, torch.zeros_like(err), err / torch.where(exact, 1.0, allowed))
    worst = int(torch.argmax(share.amax(0)))
    record_property("measured_control_of_allowed", share.max().item())
    assert bool((err <= allowed).all()), (
        f"{label}: control output {worst} off by {err[:, worst].max().item():.3e}")


def _pi_loop_response_sums(a, b, kp, ki_dt, steps):
    """The linearised error loop of a PI controller on a well-mixed zone, in the order
    ContamX evaluates it (the output acting over a step comes from the reading at the end of
    the step before): y_k = y_(k-1) + kp (s_(k-1) - s_(k-2)) + ki_dt (s_(k-1) + s_(k-2))/2,
    r_k = a r_(k-1) - b y_k, with s the reading as the controller sees it. Returns the sums
    over `steps` of |y| and |r| after a unit error in one reading, and after a unit error in
    one stored output: (y per reading, r per reading, y per output, r per output)."""
    sums = []
    for reading, output in ((1.0, 0.0), (0.0, 1.0)):
        r = y = s_prev = 0.0
        sum_y = sum_r = 0.0
        for k in range(steps):
            s = r + (reading if k == 0 else 0.0)
            y = y + kp * (s - s_prev) + ki_dt * (s + s_prev) / 2 + (output if k == 0 else 0.0)
            r = a * r - b * y
            s_prev = s
            sum_y, sum_r = sum_y + abs(y), sum_r + abs(r)
        sums += [sum_y, sum_r]
    return tuple(sums)


@pytest.mark.external
@pytest.mark.slow
def test_demand_controlled_ventilation_matches_contamx(contamx, tmp_path, record_property):
    """`dcv_three_zone.prj`: CO2 from occupants (08:00-18:00, a schedule node on the
    source), a PI controller (Kp 0.01, Ki 1e-5 /s, 300 s steps) on zone one's CO2 above
    650 mg/kg driving the exhaust fan (`fan_cmf`, R = 0.100342 kg/s at signal 1), never
    below 0.2, out of zone one (300 m3, M = 361.2 kg of air).

    Budget: first-order error propagation through the CLOSED loop. Two sources each step.
    (i) The reading the controller sees: ContamX computes control signals in float32
    (0.178749993 = fl32(0.17875), measured); its reading r = 1e6 x is rounded twice (x to
    float32, then the product), 2 u r, and e = r - 650 is exact (Sterbenz) but carries that
    absolute error; and noodl's controller reads noodl's own CO2, which differs from
    ContamX's by the plant's transient budget, 64 u r. So a reading error of
    n = 66 u r_max per step. (ii) The stored output: two float32 additions, each rounding at
    most u/2 since |y| < 1, plus the increment's own rounding and the float32 gains,
    u max|dy|: p = (1 + max|dy|) u per step.
    The plant, linearised at an operating point (exhaust Q, CO2 excess c over the 600 mg/kg
    ambient air replacing the exhausted air): one step of the implicit species update gives
    r_k = a r_(k-1) - b y_k with a = M / (M + dt Q) and b = dt R c / (M + dt Q) mg/kg per
    unit signal. The sums of the loop's response to one unit error
    (`_pi_loop_response_sums`) bound the error at a fixed operating point; the loop is
    time-varying, so as a first-order argument each sum is taken at its maximum over the
    range the reference visits: Q from 0.2 R to R times the largest fan signal (0.907), c
    over the unsaturated steps' excess (18.7 to 61.4 mg/kg), and the open loop b = 0, which
    the run visits too: for 11 unsaturated steps the PI output is below 0.2 and the fan
    takes the constant 0.2. The maxima: y per reading 0.876 and y per output 289 (open
    loop), r per reading 3.03 and r per output 985 (Q = 0.2 R, c = 29 and 24 mg/kg). Hence
    dy = 0.876 n + 289 p (about 38 500 u, on the PI output and the fan signal) and
    dr = 3.03 n + 985 p in mg/kg (about 222 u relative to the smallest reading). A clipped
    output (0 or 1) is exact on both sides, which only shortens these sums.
    Hence: flows 64 u plus dy R kg/s absolute (the fan's flow is its signal times R; the
    zone's other paths shift by no more); mass fractions 64 u plus dr / r_min relative; the
    reading that plus 2 u; the excess (r - 600) x 0.6584 the reading's absolute error times
    0.6584. ContamX's airflow is solved to `PROJECT_AIRFLOW`: at its own 1e-5 convergence
    its PI output at step 120 is 0.31324 against 0.31346 here, an error of ContamX's airflow
    iteration (0.313458 at the tight tolerance)."""
    from noodl.apps.building_physics.contamx import run_project

    ref = run_project(DCV, airflow_convergence=PROJECT_AIRFLOW)
    kp, ki_dt, rating, dt, air_mass = 0.01, 1e-5 * 300.0, 0.100342, 300.0, 300 * 1.2041
    reading, pi = ref["control"][:, 0], ref["control"][:, 1]
    r_max, r_min = reading.max().item(), reading.min().item()
    inside = (pi > 0) & (pi < 1)
    excess = reading[inside] - 600.0
    assert bool(((pi < 0.2) & inside).any())            # the open loop is visited
    q_top = rating * ref["control"][:, 2].max().item()
    points = [(0.0, 0.0)] + [                           # (Q, c); the first: the open loop
        (0.2 * rating + (q_top - 0.2 * rating) * i / 16,
         excess.min().item() + (excess.max() - excess.min()).item() * j / 16)
        for i in range(17) for j in range(17)]
    sums = torch.tensor([_pi_loop_response_sums(
        air_mass / (air_mass + dt * q), dt * rating * c / (air_mass + dt * q), kp, ki_dt,
        len(pi)) for q, c in points], dtype=F64)
    y_n, r_n, y_p, r_p = sums.amax(0).tolist()
    where = [points[i] for i in sums.argmax(0).tolist()]
    n = 66 * F32_U * r_max
    p = (1 + (pi[1:] - pi[:-1]).abs().max().item()) * F32_U
    dy, dr = y_n * n + y_p * p, r_n * n + r_p * p
    rtol_mf = 64 * F32_U + dr / r_min
    record_property("loop", f"maxima {y_n:.3g} {r_n:.3g} {y_p:.3g} {r_p:.3g} at (Q/R, c) "
                            + ", ".join(f"({q / rating:.3f}, {c:.1f})" for q, c in where)
                            + f"; dy {dy / F32_U:.0f} u, dr/r_min {dr / r_min / F32_U:.0f} u")
    ref, out = _project_parity(DCV, tmp_path, rtol_flow=64 * F32_U, rtol_mf=rtol_mf,
                               atol_flow=dy * rating, ref=ref,
                               record_property=record_property, species_floor=True,
                               label="Demand-controlled ventilation (PI), 288 steps")
    reading_err = (rtol_mf + 2 * F32_U) * r_max
    _control_outputs_match(
        ref, out, torch.tensor([rtol_mf + 2 * F32_U, 0.0, 0.0, 0.0], dtype=F64),
        record_property, "dcv",
        atol=torch.tensor([0.0, dy, dy, 0.6584 * reading_err], dtype=F64))


@pytest.mark.external
@pytest.mark.slow
def test_a_pressure_controlled_exhaust_matches_contamx(contamx, tmp_path, record_property):
    """`pressure_exhaust_three_zone.prj`: a scheduled supply fan; zone three's gauge
    pressure above -0.5 Pa (a limit control) through a P controller (Kp 0.8) drives the
    exhaust fan; a path pressure-difference and a flow sensor reported. No state carries a
    rounding from step to step (P control), so the three-zone transient budget, 64 u. The
    float32 gauge pressure p (the reference's largest |p|, 0.215 Pa) and set point -0.5 Pa
    enter the limit control p + 0.5 with an absolute u (|p|max + 0.5) = 0.72 u Pa, so the
    P output with 0.8 x that, 0.58 u, and the exhaust flow with 0.58 u R. Both sit inside
    their columns' floors as long as the largest P output exceeds 0.58 / 64 = 0.009 (it is
    0.40: a floor of 25.6 u) and the largest flow exceeds 0.009 R (it is 0.125 kg/s); both
    conditions are checked from the measured |p|max."""
    ref, out = _project_parity(PRESSURE_EXHAUST, tmp_path, rtol_flow=64 * F32_U,
                               rtol_mf=64 * F32_U, record_property=record_property,
                               species_floor=True,
                               label="Pressure-controlled exhaust (P), 288 steps")
    p_output_err = 0.8 * (ref["control"][:, 0].abs().max().item() + 0.5)     # in u
    assert ref["control"][:, 1].max() > p_output_err / 64
    assert ref["flow"].abs().max() > p_output_err / 64 * 0.100342
    _control_outputs_match(ref, out, 64 * F32_U, record_property, "pressure exhaust")


@pytest.mark.external
@pytest.mark.slow
def test_a_hysteresis_window_matches_contamx(contamx, tmp_path, record_property):
    """`hysteresis_window_three_zone.prj`: zone one's temperature ramps 293.15 -> 303.15 ->
    293.15 K through a schedule node; a window (path 5, an orifice) opens by hysteresis
    (slope 1.66667) and a band switch on the sensed temperature, times a schedule; every
    logical and arithmetic node type reported.

    Budget. The sensor reads x = (T - 293.15) x 0.1 in float32: ContamX holds the ramped
    temperature and the offset in single precision, each to u times itself, and the
    difference cancels them against a small x. So x is off by an ABSOLUTE
    dx = (0.1 (303.15 + 293.15) + 2) u = 61.6 u (the 2 u: the scale and the product), not by
    a relative 64 u (measured: 0.29166871 = 28/96 + 2.0e-6 in ContamX at 08:25). The
    switches decide alike: every threshold (0.49, 0.69, 0.29) lies at least 4.2e-4 from the
    sensed values' grid k/96, far beyond dx; the tests u > 0 meet x = 0 in the isothermal
    hours, where ContamX's float32 T and offset cancel exactly and noodl's float32-held
    schedule value lies below the float64 offset (x = -6.1e-7): both off. No output sums
    roundings over steps (the hysteresis stores one); the largest gain from x to an output
    is the sum node's x + hysteresis + set point, 1 + 1.66667: an absolute
    2.66667 dx on the control outputs, on top of 64 u. The window's flow is its opening (at
    most 1.66667 dx off) times the flow per unit opening F_open (the reference's largest
    |flow| / opening), and the other paths shift by no more: 1.66667 dx F_open absolute on
    the flows, on top of the transient's 64 u. The air is clean (no species source)."""
    from noodl.apps.building_physics.contamx import run_project

    ref = run_project(WINDOW, airflow_convergence=PROJECT_AIRFLOW)
    slope = 1.66667
    dx = (0.1 * (303.15 + 293.15) + 2) * F32_U
    opening, window_flow = ref["control"][:, 4], ref["flow"][:, 4]
    f_open = (window_flow.abs()[opening > 0] / opening[opening > 0]).max().item()
    ref, out = _project_parity(WINDOW, tmp_path, rtol_flow=64 * F32_U, rtol_mf=64 * F32_U,
                               atol_flow=slope * dx * f_open, ref=ref,
                               record_property=record_property, species_floor=True,
                               label="Hysteresis window, 288 steps")
    _control_outputs_match(ref, out, 64 * F32_U, record_property, "hysteresis window",
                           atol=(1 + slope) * dx)


def _through_schedule_nodes(src, records, tmp_path):
    """`src` with each record's schedule replaced by a control reference to a schedule node
    of the same week schedule (ContamX: bit-identical histories, measured)."""
    text = src.read_bytes().decode().replace("\r\n", "\n")
    body = "! # typ seq f n  c1  c2 name\n"
    for i, (old, new, week) in enumerate(records, 1):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
        body += f"  {i} sch {i} 0 0 0 0 <none>\nschedule node\n {week}\n"
    text = re.sub(r"^\d+ ! control nodes:\n.*?^-999\n",
                  f"{len(records)} ! control nodes:\n{body}-999\n", text,
                  flags=re.DOTALL | re.MULTILINE)
    out = tmp_path / "controlled" / src.name
    out.parent.mkdir(parents=True)
    out.write_bytes(text.replace("\n", "\r\n").encode())
    for suffix in (".wth", ".ctm"):
        if src.with_suffix(suffix).exists():
            shutil.copy(src.with_suffix(suffix), out.parent)
    return out


AHS_RECORDS = [("   8   16   4   5   0   1   0   0   1   0   1",
                "   8   16   4   5   0   1   0   0   0   1   1", 1),
               ("   5    8   5   1   0   0   0   1   2   0   1",
                "   5    8   5   1   0   0   0   1   0   2   1", 2)]
DUCT_RECORDS = [("  2  0   2   3   3   0   1   0", "  2  0   2   3   3   0   0   1", 1),
                ("  6  0 1  2   0   0   2   0", "  6  0 1  2   0   0   0   2", 2)]


@pytest.mark.parametrize(("src", "records"),
                         [(AHS, AHS_RECORDS),
                          pytest.param(DUCTS, DUCT_RECORDS, marks=pytest.mark.slow)])
def test_a_control_through_a_schedule_node_equals_the_schedule(src, records, tmp_path):
    """The AHS outdoor-air fraction and supply point, a duct fan's speed and a terminal's
    opening driven through schedule nodes: noodl's run is the scheduled run exactly, as
    ContamX's is (both paths write the same float64 value into the same driver entry)."""
    from noodl.apps.building_physics.simulation import simulate_project

    controlled = _through_schedule_nodes(src, records, tmp_path)
    a = simulate_project(read_prj(src), atol=NOODL_ATOL, rtol=1e-14)
    b = simulate_project(read_prj(controlled), atol=NOODL_ATOL, rtol=1e-14)
    for key in ("path_flow", "species.x"):
        torch.testing.assert_close(b[key], a[key], rtol=0, atol=0)


@pytest.mark.external
@pytest.mark.parametrize(("src", "records", "rtol_flow"),
                         [(AHS, AHS_RECORDS, 32 * F32_U),
                          pytest.param(DUCTS, DUCT_RECORDS, 64 * F32_U,
                                       marks=pytest.mark.slow)])
def test_controls_through_schedule_nodes_match_contamx(contamx, src, records, rtol_flow,
                                                       tmp_path, record_property):
    """The same variants against ContamX, at the budgets of the scheduled originals' parity
    tests (`test_a_simple_ahs_matches_contamx`: 32 u on flows;
    `test_a_duct_network_matches_contamx`: 64 u; mass fractions `_mass_fraction_bound` of
    those), since the variants change nothing ContamX computes."""
    controlled = _through_schedule_nodes(src, records, tmp_path)
    _project_parity(controlled, tmp_path, rtol_flow=rtol_flow,
                    mf_bound=_mass_fraction_bound(rtol_flow),
                    record_property=record_property, species_floor=True,
                    label=f"{src.stem} through controls")


OCCUPANTS = DATA / "occupants_three_zone.prj"
# ContamX writes its exposure, age and air-change-rate files with `%e` (7 significant
# digits): a printed value is within half a unit of its last digit, at most 5e-7 of itself.
PRINT = 5e-7
AGE_FLAG = ("!vol ach -bw cbw exp -bw age -bw\n  0   1   0   0   0   0   0   0",
            "!vol ach -bw cbw exp -bw age -bw\n  0   1   0   0   0   0   1   0")
ZONE_TWO_OUTSIDE = ("   2  3   0   0   0   1   0.000   600 293.15 0 two",
                    "   2 19   0   0   0   1   0.000   600 293.15 0 two")
# The duct project's exhaust terminal (junction 9) moved from the ambient into zone one: the
# constant-volume exhaust fan then draws air out of zone three through junctions 7, 8 and 9
# and returns it to zone one.
RETURN_DUCT = ("  9  0 1  -1   0   0   0   0   1", "  9  0 1  1   0   0   0   0   1")


def _variant(src, tmp_path, name, edits):
    """`src` with each `(old, new)` text edit made once, written with CRLF line ends as
    `tmp_path/<name>/<src.name>`."""
    text = src.read_bytes().decode().replace("\r\n", "\n")
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    out = tmp_path / name / src.name
    out.parent.mkdir(parents=True)
    out.write_bytes(text.replace("\n", "\r\n").encode())
    return out


def _age_and_air_change_allowed(model, out, conditioned, eps_flow, eps_mass, eps_volume):
    """The first-order bounds of `test_age_and_air_change_rate_match_contamx`'s docstring,
    evaluated on noodl physics' run. Returns the local mean age (s, (N - 1, n_i)) and its
    allowed error, the building air change rate (1/h, (N - 1,)) and its allowed error, and
    the conditioned volume (m3); print precision not included.

    Every edge's flow is allowed `eps_flow` times the largest flow that edge carries over
    the run (the flow budget, with its floor read edge by edge). The age solves
    A tau = M; to first order A dtau = dM - dA tau, and since both runs conserve mass at
    every node the flow errors do too, so row i of dA tau is the sum over the edges INTO i
    of dq (tau_i - tau_upstream) (tau = 0 outdoors). Hence
    |dtau| <= eps_mass tau + A^-1 r, r_i = sum_in |dq| |tau_i - tau_upstream|, with
    A^-1 >= 0 (an M-matrix). The air change rate is a sum of the flows crossing into the
    space over 1.2041 V: its error is at most the |dq| of the edges whose flow enters the
    space that step (to first order a flow leaving it adds nothing; a zero flow is counted
    either way) summed, over 1.2041 V, plus `eps_volume` of the rate for 1.2041 and V. An
    edge leaving the space on noodl physics' run with |q| <= |dq| may enter it on ContamX's
    (a reversal within the flow's own error): it is counted too, with its |dq|."""
    from noodl.apps.building_physics.outputs import building_air_change_rate, local_mean_age

    lay, net = model.transport["species"], model.net
    f = model.kirchhoff["air"].flows_of_kind(out["air.q"][1:], lay.flow_kinds)
    src, tgt = zip(*(net.endpoints(k) for k in lay.flow_kinds), strict=True)
    src, tgt = torch.cat(src).tolist(), torch.cat(tgt).tolist()
    dq = eps_flow * f.abs().amax(0)
    tau = local_mean_age(out, model)[1:]
    pos = {node: i for i, node in enumerate(lay.interior_idx.tolist())}
    allowed_tau = torch.empty_like(tau)
    for k in range(f.shape[0]):
        a = torch.zeros(lay.n_i, lay.n_i, dtype=F64)
        r = torch.zeros(lay.n_i, dtype=F64)

        def age(node, k=k):
            return tau[k, pos[node]].item() if node in pos else 0.0

        for e, q in enumerate(f[k].tolist()):
            if q != 0.0:
                up, down = (src[e], tgt[e]) if q > 0 else (tgt[e], src[e])
                if up in pos:
                    a[pos[up], pos[up]] += abs(q)
                    if down in pos:
                        a[pos[down], pos[up]] -= abs(q)
                ends = [(up, down)]
            else:                              # a zero flow: either way, to first order
                ends = [(src[e], tgt[e]), (tgt[e], src[e])]
            for up, down in ends:
                if down in pos:
                    r[pos[down]] += dq[e].item() * abs(age(down) - age(up))
        allowed_tau[k] = eps_mass * tau[k] + torch.linalg.solve(a, r)
    inside = {int(net.node_index(n)) for n in conditioned}
    into = torch.tensor([float(t in inside) - float(s in inside)     # +1 along the edge
                         for s, t in zip(src, tgt, strict=True)], dtype=F64)  # into it
    # Entering on noodl physics' run, or within its own error of zero (either way).
    entering = ((f * into >= 0) | (f.abs() <= dq)) & (into != 0)
    volume = sum(float(net.node_attr("volume", default=0.0)[i]) for i in inside)
    ach = building_air_change_rate(out, model)[1:]
    allowed_ach = ((dq * entering).sum(-1) / (1.2041 * volume) * 3600.0
                   + eps_volume * ach)
    return tau, allowed_tau, ach, allowed_ach, volume


def _age_and_air_change_match(ref, out, project, model, *, eps_flow, eps_mass, eps_volume,
                              record_property):
    """noodl physics' local mean age (in ContamX's zone order) and building air change rate
    against ContamX's `.age` and `.ach`, each within its derived first-order bound plus
    print precision; the conditioned volume against the `.ach` header (printed to 0.01 m3).
    Returns noodl physics' age (h) and rate (1/h)."""
    tau, allowed_tau, ach, allowed_ach, volume = _age_and_air_change_allowed(
        model, out, project.building_zones, eps_flow, eps_mass, eps_volume)
    interior = model.transport["species"].interior_idx.tolist()
    cols = [interior.index(model.net.node_index(z)) for z in project.zones]
    age, ref_age = tau[:, cols] / 3600.0, ref["age"]
    assert age.shape == ref_age.shape
    age_allowed = allowed_tau[:, cols] / 3600.0 + PRINT * ref_age.abs()
    ach_allowed = allowed_ach + PRINT * ref["air_change_rate"].abs()
    age_err, ach_err = (age - ref_age).abs(), (ach - ref["air_change_rate"]).abs()
    record_property("measured_rel_age", (age_err / ref_age.abs()).max().item())
    record_property("measured_age_of_allowed", (age_err / age_allowed).max().item())
    record_property("measured_rel_ach",
                    (ach_err / ref["air_change_rate"].abs()).max().item())
    record_property("measured_ach_of_allowed", (ach_err / ach_allowed).max().item())
    record_property("age_budget_rel_max", (age_allowed / ref_age.abs()).max().item())
    record_property("ach_budget_rel_max",
                    (ach_allowed / ref["air_change_rate"].abs()).max().item())
    assert bool((age_err <= age_allowed).all()), (
        f"age: worst error {(age_err / age_allowed).max().item():.3g} of allowed")
    assert bool((ach_err <= ach_allowed).all()), (
        f"air change rate: worst error {(ach_err / ach_allowed).max().item():.3g} of allowed")
    assert abs(volume - ref["building_volume"]) <= 0.005
    return age, ach


@pytest.mark.external
def test_occupant_exposure_age_of_air_and_air_change_rate_match_contamx(
        contamx, tmp_path, record_property):
    """`occupants_three_zone.prj`: two occupants (one moving mid-step at 01:02:30, out of the
    building from 02:00 to 03:00, generating 1e-6 kg/s of sarin on a schedule that jumps
    0 -> 0.5 at 00:32:30), zones at 303.15 / 293.15 / 283.15 K in a 263.15 K ambient, zone
    three left out of the building volume (flag 16), sarin 1e-4 in zone one at the start;
    48 steps of 300 s. Exposure, age and air change rate are ContamX's `.ebw`, `.age` and
    `.ach`.

    Budgets: first order in the float32 roundoff u of each input ContamX reads, every term
    stated; V (300, 600, 900 m3), Pb (101325 Pa), the step (300 s), the move times and the
    schedule value 0.5 are exact in float32.
    Flows: the four paths form ONE loop (ambient -> one -> two -> three -> ambient), every
    opening 1.5 m above the one floor, so a single flow q runs round it, driven by the wind
    on the two facades; each zone's (and the ambient's) hydrostatic term enters its two
    paths with opposite signs and cancels round the loop. The wind-driven product of powers
    is the three-zone steady budget, 16 u; the stack budget at the smallest zone-to-ambient
    difference (zone three, 20 K: `_stack_budget(263.15, 283.15)`, 16.7 u) bounds whatever
    the hydrostatic terms leave: 32.7 u, budgeted at 33 u as the own-file one-zone project.
    Mass fractions: each zone's rate q / M carries 33 u + 2 u (M = Pb V / (R T): R and T);
    three zones in series respond as the product of three rates (the three-zone transient's
    rule): 3 x 35 u = 105 u; plus the generation rate (1e-6, 1 u) and the initial mass
    fraction (1e-4, 1 u): 107 u, with `_project_parity`'s 64 u x max floor.
    Exposure: per step a weighted mean of the zone's start and end mass fractions with
    weights from the exact times (non-negative, summing to at most 1), so the mass
    fractions' 107 u relative and their floor, plus print; zero exactly while out.
    Age: along the loop tau_k = sum over zone k and the zones upstream of it of M_j / q, so
    its relative error is that of M plus that of q: 2 u + 33 u = 35 u, plus print. The
    general bound of `_age_and_air_change_allowed` gives exactly this for one flow (each
    |tau_i - tau_upstream| is M_i / q, so r_i = 33 u M_i and A^-1 r = 33 u tau).
    Air change rate: the loop's one flow crossing into the conditioned zones one and two
    (900 m3), 33 u, over 1.2041 (1 u) V (exact): 34 u, plus print."""
    project = read_prj(OCCUPANTS)
    assert project.building_zones == ["one", "two"]
    ref, out = _project_parity(OCCUPANTS, tmp_path, rtol_flow=33 * F32_U,
                               rtol_mf=107 * F32_U, record_property=record_property,
                               label="Three-zone project with two occupants, 48 steps")
    assert ref["building_volume"] == 900.0
    noodl_project = read_prj(tmp_path / "noodl" / OCCUPANTS.name)
    model, _, _ = project_to_model(noodl_project)
    ours, theirs = out["exposure"], ref["exposure"]
    assert ours.shape == theirs.shape == (48, 2, 1)
    assert bool((ours[theirs == 0] == 0).all())
    assert bool((theirs[:, 0] == 0).any()) and bool((theirs[:, 0] > 0).any())
    floor = 64 * F32_U * ref["mf"].abs().max()
    allowed = (107 * F32_U + PRINT) * theirs.abs() + floor
    err = (ours - theirs).abs()
    nonzero = theirs != 0
    record_property("measured_rel_exposure",
                    (err[nonzero] / theirs[nonzero].abs()).max().item())
    record_property("measured_exposure_of_allowed", (err / allowed).max().item())
    assert bool((err <= allowed).all()), (
        f"exposure: worst error {(err / allowed).max().item():.3g} of allowed")
    _age_and_air_change_match(ref, out, noodl_project, model, eps_flow=33 * F32_U,
                              eps_mass=2 * F32_U, eps_volume=F32_U,
                              record_property=record_property)


AGE_ACH_VARIANTS = {
    "ducts": (DUCTS, [AGE_FLAG], 64 * F32_U),
    "ahs": (AHS, [AGE_FLAG], 32 * F32_U),
    "ducts, zone two outside": (DUCTS, [AGE_FLAG, ZONE_TWO_OUTSIDE], 64 * F32_U),
    "ahs, zone two outside": (AHS, [AGE_FLAG, ZONE_TWO_OUTSIDE], 32 * F32_U),
    "return duct": (DUCTS, [AGE_FLAG, RETURN_DUCT], 64 * F32_U),
}


@pytest.mark.external
@pytest.mark.parametrize("case", [
    pytest.param(c, marks=pytest.mark.slow) if "ducts" in c or "return" in c else c
    for c in AGE_ACH_VARIANTS])
def test_age_and_air_change_rate_match_contamx(contamx, case, tmp_path, record_property):
    """The duct and simple-AHS projects with ContamX's age file switched on; each also with
    zone two flagged outside the building volume (flag 16); and the duct project with a
    return duct (`RETURN_DUCT`: zone three -> junctions 7, 8, 9 -> zone one).

    ContamX's conditioned space is every zone without flag 16 plus the whole mechanical
    system (each duct junction, each air handler's plenums: `project.building_zones`), and
    junction -> zone and zone -> junction flows are internal: the return duct moves about
    0.05 kg/s from zone three to zone one and none of it enters ContamX's rate, which the
    space without the junctions would count (checked below).

    Budgets: first order in float32 input roundoff u. Flows: those of
    `test_a_duct_network_matches_contamx` (64 u) and `test_a_simple_ahs_matches_contamx`
    (32 u); neither the flags nor the moved terminal changes an element law. Mass
    fractions: `_mass_fraction_bound` of those flow budgets, derived on each variant's run
    (it follows the AHS's return air and the return duct round their loops).
    Age: tau solves A tau = M (A: each node's outflow on the diagonal, its inflows off it; M
    the air mass, zero at a storage-free plenum). With dq the flow errors and dM the mass
    errors, A dtau = dM - dA tau to first order. Both runs conserve mass at every node, so
    row i of dA tau is the sum over the edges into i of dq (tau_i - tau_upstream), and
    A^-1 >= 0 gives |dtau| <= eps_mass tau + A^-1 r, r_i = sum_in |dq| |tau_i - tau_up|
    (`_age_and_air_change_allowed`, evaluated each step). |dq| is the flow budget times the
    largest flow the edge carries over the run (the flows' floor, read edge by edge).
    eps_mass: R and T of each node (2 u), and a junction's volume from float32 duct areas
    and lengths (3 u more: `_mass_roundoff`); zone volumes and Pb are exact. Without
    recirculation A^-1 r stays near eps_flow tau (each |tau_i - tau_up| is what row i
    adds: the chain argument of the occupants test); the AHS's return air raises it to
    about 2.7 eps_flow tau, the return duct to 1.9 (noodl physics' flows; computed here,
    not assumed). Plus print.
    Air change rate: the flows entering the space, each within its |dq|, summed, over
    1.2041 V; 1.2041 (1 u) and V, ContamX's float32 sum of float32 junction volumes (1 u):
    eps_volume = 2 u. Plus print. The volume itself is checked against the `.ach` header."""
    from noodl.apps.building_physics.contamx import run_project
    from noodl.apps.building_physics.outputs import building_air_change_rate

    src, edits, rtol_flow = AGE_ACH_VARIANTS[case]
    variant = _variant(src, tmp_path, "variant", edits)
    ref = run_project(variant, airflow_convergence=PROJECT_AIRFLOW)
    assert ref["age"].shape[0] == ref["air_change_rate"].shape[0] == len(ref["time"]) - 1
    ref, out = _project_parity(variant, tmp_path, rtol_flow=rtol_flow,
                               mf_bound=_mass_fraction_bound(rtol_flow),
                               record_property=record_property, species_floor=True, ref=ref,
                               label=f"{src.stem} ({case}): age of air and air change rate")
    project = read_prj(tmp_path / "noodl" / src.name)
    model, _, _ = project_to_model(project)
    assert ("two" in project.building_zones) == ("outside" not in case)
    _age_and_air_change_match(ref, out, project, model, eps_flow=rtol_flow,
                              eps_mass=_mass_roundoff(project, model),
                              eps_volume=2 * F32_U,
                              record_property=record_property)
    if case == "return duct":
        # The space without the junctions counts the terminals' flows as entering: far
        # off ContamX, which keeps them inside.
        zones_only = building_air_change_rate(out, model, conditioned=project.zones)[1:]
        assert bool(((zones_only - ref["air_change_rate"]).abs()
                     > 0.1 * ref["air_change_rate"]).all())

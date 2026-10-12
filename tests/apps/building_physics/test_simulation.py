"""The input mechanism and the run loop `simulate`, as measured against ContamX."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics import build_model
from noodl.apps.building_physics.inputs import (
    Input,
    apply_inputs,
    bind_inputs,
    scatter,
)
from noodl.apps.building_physics.prj import project_to_model, read_prj
from noodl.apps.building_physics.schedules import Calendar, TimeSeries
from noodl.apps.building_physics.simulation import simulate
from noodl.apps.building_physics.thermal import R_AIR

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
STACK = DATA / "test_OneZoneWthCtmStack-UseApi.prj"
F64 = torch.float64
TIGHT = dict(atol=1e-14, rtol=1e-14)


def _f(*v):
    return torch.tensor(v, dtype=F64)


# ------------------------------------------------------------------------------ scatter
def test_scatter_replaces_positions_and_broadcasts_batches():
    base = _f(1.0, 2.0, 3.0)
    assert torch.equal(scatter(base, (2, 0), _f(9.0, 8.0)), _f(8.0, 2.0, 9.0))
    assert torch.equal(scatter(base, None, torch.tensor(5.0, dtype=F64)), _f(5.0, 5.0, 5.0))
    batched = scatter(base, (1,), _f(7.0, 6.0).reshape(2, 1))
    assert torch.equal(batched, _f(1.0, 7.0, 3.0, 1.0, 6.0, 3.0).reshape(2, 3))
    assert torch.equal(scatter(torch.zeros(1, 2, dtype=F64), None, _f(4.0, 5.0)),
                       _f(4.0, 5.0).reshape(1, 2))
    v = torch.tensor([2.0], dtype=F64, requires_grad=True)
    scatter(base, (0,), v).sum().backward()
    assert v.grad.item() == 1.0
    assert torch.equal(base, _f(1.0, 2.0, 3.0))                      # never in place


# ------------------------------------------------------------------------------- inputs
def test_input_at_names_resolve_through_the_models_declared_inputs():
    model, _, drivers = project_to_model(read_prj(THREE))
    ramp = TimeSeries([0.0, 600.0], [293.15, 303.15])
    (bound,) = bind_inputs(model, [Input(ramp, "T", at="two")])
    assert bound.index == (model.net.node_index("two"),)
    d = apply_inputs(drivers, [bound], 300.0, Calendar())
    assert d["T"][model.net.node_index("two")].item() == pytest.approx(298.15, rel=1e-15)
    assert d["T"][model.net.node_index("one")].item() == 293.15
    assert d is not drivers and drivers["T"][model.net.node_index("two")].item() == 293.15


def test_overlapping_inputs_are_refused():
    model, _, _ = project_to_model(read_prj(THREE))
    a = Input(TimeSeries([0.0, 1.0], [1.0, 1.0]), "T", at="two", name="schedule")
    b = Input(TimeSeries([0.0, 1.0], [2.0, 2.0]), "T", at=["one", "two"], name="mine")
    with pytest.raises(ValueError, match=r"'schedule'.*'mine'.*'T'"):
        bind_inputs(model, [a, b])
    whole = Input(TimeSeries([0.0, 1.0], [1.0, 1.0]), "T", name="all")
    with pytest.raises(ValueError, match=r"'schedule'.*'all'"):
        bind_inputs(model, [a, whole])


def test_an_input_naming_an_undeclared_key_with_labels_is_refused():
    model, _, _ = project_to_model(read_prj(THREE))
    with pytest.raises(KeyError, match="no_such_key"):
        bind_inputs(model, [Input(TimeSeries([0.0, 1.0], [1.0, 1.0]), "no_such_key", at="two")])
    with pytest.raises(ValueError, match="index and at"):
        Input(TimeSeries([0.0, 1.0], [1.0, 1.0]), "T", index=0, at="two")


# ----------------------------------------------------------------------------- simulate
def _seeded(state):
    """The three-zone project starts clean with a clean ambient (CCdef = 0); give the zones
    some sarin so that the species histories are not identically zero."""
    return {**state, "species.x": torch.full_like(state["species.x"], 1e-3)}


def test_simulate_without_inputs_is_the_plain_step_loop():
    model, state, drivers = project_to_model(read_prj(THREE))
    state = _seeded(state)
    times = [300.0 * k for k in range(7)]
    out = simulate(model, state, drivers, times, **TIGHT)
    manual, s = [state["species.x"]], dict(state)
    for k in range(1, 7):
        s = model.step(s, {**drivers, "time": _f(times[k])[0]}, 300.0, t=times[k - 1], **TIGHT)
        manual.append(s["species.x"])
    assert out["time"].tolist() == times
    assert out["air.q"].shape == (7, 4) and out["species.x"].shape == (7, 3, 1)
    torch.testing.assert_close(out["species.x"], torch.stack(manual), rtol=1e-10, atol=1e-18)


def test_inputs_are_taken_at_the_end_of_each_step_and_breakpoints_can_split_steps():
    model, state, drivers = project_to_model(read_prj(THREE))
    state = _seeded(state)
    wind = TimeSeries([0.0, 450.0, 600.0], [5.23, 1.0, 1.0], interpolation="step")
    inp = [Input(wind, "V_met")]
    coarse = simulate(model, state, drivers, [0.0, 300.0, 600.0], inputs=inp,
                      land_on_breakpoints=False, **TIGHT)
    fine = simulate(model, state, drivers, [0.0, 300.0, 600.0], inputs=inp, **TIGHT)
    # Step 300 -> 600 ends after the jump: both take V_met = 1 for it. With
    # breakpoints landed, the step is split at 450 s and the 300 -> 450 part still has 5.23.
    manual = dict(state)
    for t0, t1, v in ((0.0, 300.0, 5.23), (300.0, 450.0, 5.23), (450.0, 600.0, 1.0)):
        manual = model.step(manual, {**drivers, "V_met": _f(v)[0], "time": _f(t1)[0]},
                            t1 - t0, t=t0, **TIGHT)
    torch.testing.assert_close(fine["species.x"][-1], manual["species.x"], rtol=1e-10, atol=1e-18)
    torch.testing.assert_close(coarse["air.q"][-1], fine["air.q"][-1], rtol=1e-12, atol=1e-15)
    assert not torch.allclose(coarse["species.x"][-1], fine["species.x"][-1], rtol=1e-6)


def test_a_batched_run_matches_its_instances_run_one_at_a_time():
    model, state, drivers = project_to_model(read_prj(THREE))
    state = _seeded(state)
    ramp = [Input(TimeSeries([0.0, 900.0], [293.15, 303.15]), "T", at="three")]
    times = [0.0, 300.0, 600.0, 900.0]
    both = simulate(model, state, {**drivers, "V_met": _f(5.23, 2.0)}, times, inputs=ramp, **TIGHT)
    assert both["species.x"].shape == (4, 2, 3, 1)
    for i, v in enumerate((5.23, 2.0)):
        one = simulate(model, state, {**drivers, "V_met": _f(v)[0]}, times, inputs=ramp, **TIGHT)
        torch.testing.assert_close(both["species.x"][:, i], one["species.x"],
                                   rtol=1e-10, atol=1e-18)
        torch.testing.assert_close(both["air.q"][:, i], one["air.q"], rtol=1e-10, atol=1e-15)


def test_gradients_reach_a_time_series():
    model, state, drivers = project_to_model(read_prj(THREE))
    state = {**state, "species.x": torch.full_like(state["species.x"], 1e-3)}
    temps = torch.tensor([293.15, 303.15], dtype=F64, requires_grad=True)
    inp = [Input(TimeSeries([0.0, 600.0], temps), "T", at="one")]
    out = simulate(model, state, drivers, [0.0, 300.0, 600.0], inputs=inp, **TIGHT)
    (g,) = torch.autograd.grad(out["species.x"][-1].sum(), temps)
    assert bool(torch.isfinite(g).all()) and g.abs().sum().item() > 0.0


def _set_dod_mdt(project, on):
    """The project as if its run control set (or cleared) dodMdt."""
    import dataclasses

    project.run_control = dataclasses.replace(project.run_control, zone_mass_change=on)


def test_the_zone_air_mass_change_enters_the_airflow_balance():
    """With `air_mass_change`, net inflow into a heated zone = (M_new - M_old) / dt, with M at
    the zone's absolute pressure (Pb + P_zone) V / (R T)."""
    project = read_prj(THREE)
    _set_dod_mdt(project, True)
    model, state, drivers = project_to_model(project)
    # "step" series: 293.15 over (0, 300], 303.15 over (300, 600] -- the jump lands in the
    # step ending 600 s (left limit at 300 s).
    step_up = [Input(TimeSeries([0.0, 300.0, 600.0], [293.15, 303.15, 303.15],
                                interpolation="step"), "T", at="one")]
    out = simulate(model, state, drivers, [0.0, 300.0, 600.0], inputs=step_up,
                   air_mass_change=True, **TIGHT)
    flows = project.path_flows(out["air.q"][-1])
    net_in = flows[0] - flows[1]                       # path 1 ambient->one, path 2 one->two
    V, i = 300.0, project.net.node_index("one")
    p_old, p_new = 101325.0 + out["air.phi"][-2, i], 101325.0 + out["air.phi"][-1, i]
    dM = V / R_AIR * (p_new / 303.15 - p_old / 293.15)
    assert net_in.item() == pytest.approx(dM / 300.0, rel=1e-9)
    without = simulate(model, state, drivers, [0.0, 300.0, 600.0], inputs=step_up,
                       air_mass_change=False, **TIGHT)
    f0 = project.path_flows(without["air.q"][-1])
    assert (f0[0] - f0[1]).abs().item() < 1e-12


def test_the_air_mass_change_follows_the_zone_pressure_without_zone_pressure_density():
    """ContamX's dodMdt mass is (Pb + P_zone) V / (R T) even with densZP = 0 (measured: when
    a 0.1 kg/s exhaust switches on, each zone of the scheduled three-zone project loses
    M dP_zone / (Pb dt), -6.5e-7 / -2.7e-6 / -5.7e-6 kg/s). A wind-speed jump moves the zone
    pressures with the temperatures fixed: net inflow = (P_zone change) V / (R T dt), at the
    pressures the step itself solves to."""
    project = read_prj(THREE)
    _set_dod_mdt(project, True)
    assert not project.density_uses_zone_pressure
    model, state, drivers = project_to_model(project)
    gust = [Input(TimeSeries([0.0, 300.0, 600.0], [5.23, 10.0, 10.0], interpolation="step"),
                  "V_met")]
    out = simulate(model, state, drivers, [0.0, 300.0, 600.0], inputs=gust,
                   air_mass_change=True, **TIGHT)
    f = project.path_flows(out["air.q"][-1])
    net_in = {"one": f[0] - f[1], "two": f[1] + f[2], "three": f[3] - f[2]}
    for zone, volume in (("one", 300.0), ("two", 600.0), ("three", 900.0)):
        i = project.net.node_index(zone)
        dp = out["air.phi"][-1, i] - out["air.phi"][-2, i]
        assert abs(dp.item()) > 1e-2
        expected = dp * volume / (R_AIR * 293.15) / 300.0
        assert net_in[zone].item() == pytest.approx(expected.item(), rel=1e-6)


@pytest.mark.parametrize("dt", [1e-3, 0.1, 300.0])
def test_the_air_mass_change_is_implicit_in_the_airflow_solve_at_any_step_length(dt):
    """The mass term's slope V / (R T dt) outgrows the envelope's conductance for short steps;
    it sits in the solve's Jacobian, so a 1 ms step converges like a 300 s one, and the mass
    each zone gains is what flowed in."""
    project = read_prj(THREE)
    _set_dod_mdt(project, True)
    model, state, drivers = project_to_model(project)
    gust = [Input(TimeSeries([0.0, dt, 2 * dt], [5.23, 10.0, 10.0], interpolation="step"),
                  "V_met")]
    # The withdrawal is a difference of masses of order 1e3 kg: the solve resolves it to
    # about 1e-16 x 1e3 kg / dt.
    atol = max(1e-14, 1e-12 / dt)
    out = simulate(model, state, drivers, [0.0, dt, 2 * dt], inputs=gust,
                   air_mass_change=True, atol=atol, rtol=1e-14)
    f = project.path_flows(out["air.q"][-1])
    net_in = {"one": f[0] - f[1], "two": f[1] + f[2], "three": f[3] - f[2]}
    for zone, volume in (("one", 300.0), ("two", 600.0), ("three", 900.0)):
        i = project.net.node_index(zone)
        dp = out["air.phi"][-1, i] - out["air.phi"][-2, i]
        gained = dp * volume / (R_AIR * 293.15)
        assert abs(gained.item()) > 1e-6
        assert (net_in[zone] * dt).item() == pytest.approx(gained.item(), rel=1e-6,
                                                           abs=10 * atol * dt)


def test_air_mass_change_needs_the_zone_air_mass_term_on_the_airflow_layer():
    """A model whose airflow layer carries no `ZoneAirMassChange` is refused by name."""
    project = read_prj(THREE)
    _set_dod_mdt(project, True)
    model, state, drivers = project_to_model(project)
    model.potential["air"]._node_sources = []
    with pytest.raises(ValueError, match="ZoneAirMassChange"):
        simulate(model, state, drivers, [0.0, 60.0], air_mass_change=True)


def test_zone_pressure_density_steps_are_iterated_to_their_fixed_point():
    model, state, drivers = project_to_model(read_prj(STACK), ambient={"Ta": 273.15})
    step_up = [Input(TimeSeries([0.0, 300.0], [293.15, 303.15]), "T", at="singleZone")]
    # The fixed point alone: the re-step below carries no start-of-step air mass, so the
    # project's dM/dt term (tested on its own above) is switched off here.
    out = simulate(model, state, drivers, [0.0, 300.0], inputs=step_up, air_mass_change=False,
                   **TIGHT)
    d = apply_inputs(drivers, bind_inputs(model, step_up), 300.0, Calendar())
    start = {**state, "species.x": out["species.x"][0], "air.phi": out["air.phi"][-1]}
    again = model.step(start, {**d, "time": _f(300.0)[0]}, 300.0, t=0.0, **TIGHT)
    assert (again["air.phi"] - out["air.phi"][-1]).abs().max().item() <= 1e-12 * 101325.0


def test_air_mass_change_needs_prescribed_temperatures():
    from noodl.topology import Network

    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0, volume=0.0, T0=283.15)
    net.add_node("z", z_ref=0.0, volume=50.0, T0=293.15, heat_capacity=0.0)
    net.add_edge("ambient", "z", kind="airpath")
    from noodl.elements import PowerLaw

    model, state, drivers = build_model(
        net, air_elements=[PowerLaw(_f(0.01)[0], _f(0.5)[0], kind="airpath")], drives=[],
        return_inputs=True)
    with pytest.raises(ValueError, match="air_mass_change"):
        simulate(model, state, drivers, [0.0, 60.0], air_mass_change=True)


def test_no_round_off_sub_steps_appear_next_to_a_breakpoint():
    """A breakpoint a few ulps from an output time (or from the step's start) is that time:
    the step is not split into a ~1e-11 s sliver."""
    import math

    from noodl.apps.building_physics.schedules import DAY, DaySchedule, WeekSchedule

    model, state, drivers = project_to_model(read_prj(THREE))
    cal = Calendar(start_time=1234.567)
    week = WeekSchedule([DaySchedule([0.0, 46800.1, DAY], [5.23, 1.0, 1.0])])
    bp = week.breakpoints(0.0, DAY, cal)[0]
    after = math.nextafter(math.nextafter(bp, math.inf), math.inf)
    before = math.nextafter(math.nextafter(bp, -math.inf), -math.inf)
    steps: list[float] = []
    plain = model.step

    def recording(s, d, dt, **kw):
        steps.append(dt)
        return plain(s, d, dt, **kw)

    model.step = recording
    inp = [Input(week, "V_met")]
    simulate(model, state, drivers, [0.0, after], inputs=inp, calendar=cal, **TIGHT)
    simulate(model, state, drivers, [0.0, before, before + 600.0], inputs=inp, calendar=cal,
             **TIGHT)
    assert min(steps) >= 1.0, steps


def test_zone_pressure_max_iter_must_be_at_least_one():
    model, state, drivers = project_to_model(read_prj(THREE))
    with pytest.raises(ValueError, match="zone_pressure_max_iter"):
        simulate(model, state, drivers, [0.0, 300.0], zone_pressure_max_iter=0)


def test_a_run_must_start_at_time_zero():
    model, state, drivers = project_to_model(read_prj(THREE))
    with pytest.raises(ValueError, match=r"times\[0\]"):
        simulate(model, state, drivers, [300.0, 600.0])


def test_input_positions_accept_numpy_and_torch_integers():
    import numpy as np

    signal = TimeSeries([0.0, 1.0], [1.0, 1.0])
    assert Input(signal, "T", index=np.int64(3)).index == (3,)
    assert Input(signal, "T", index=torch.tensor(2)).index == (2,)
    assert Input(signal, "T", index=[np.int64(1), 4]).index == (1, 4)


# ------------------------------------------------------------------------- simulate_project
def test_simulate_project_runs_the_projects_own_run_control_and_files():
    from noodl.apps.building_physics.simulation import simulate_project

    p = read_prj(DATA / "test_OneZoneWthCtmStack.prj")
    out = simulate_project(p, atol=1e-14, rtol=1e-14)
    assert out["time"].shape == (289,) and out["time"][-1].item() == 86400.0
    assert out["path_flow"].shape == (289, 2)
    # 13:05 (row 157): the zone has just jumped to 303.15 K (inputs take their left limit at
    # the end of a step) and its air-mass change sets the imbalance (dM/dt in the balance):
    # net inflow = Pb V / R (1/303.15 - 1/293.15) / 300 < 0.
    net = out["path_flow"][157].sum().item()
    assert net == pytest.approx(101325.0 * 600 / R_AIR * (1 / 303.15 - 1 / 293.15) / 300.0,
                                rel=2e-3)


def test_simulate_project_refuses_what_it_cannot_reproduce(tmp_path):
    from noodl.apps.building_physics.simulation import simulate_project

    text = (DATA / "test_OneZoneWthCtm.prj").read_text()
    head = "!sim_mf slae rs maxi   relcnvg   abscnvg relax gamma ucc\n    "
    assert head + "2" in text
    cyclic = tmp_path / "cyclic.prj"
    cyclic.write_text(text.replace(head + "2", head + "3", 1))
    with pytest.raises(ValueError, match="cyclic"):
        simulate_project(read_prj(cyclic))


def test_simulate_project_refuses_an_integration_weight_other_than_one(tmp_path):
    from noodl.apps.building_physics.simulation import simulate_project

    text = (DATA / "test_OneZoneWthCtm.prj").read_text()
    trace = "1.00e-15 1.100 1.000   0 ! (trace)"
    assert trace in text
    variant = tmp_path / "gamma.prj"
    variant.write_text(text.replace(trace, "1.00e-15 1.100 0.500   0 ! (trace)", 1))
    with pytest.raises(ValueError, match=r"weight 0\.5"):
        simulate_project(read_prj(variant))


def test_simulate_project_on_a_steady_run_control_is_the_steady_solve(tmp_path):
    from noodl.apps.building_physics.prj import steady
    from noodl.apps.building_physics.simulation import simulate_project

    text = THREE.read_text()
    af = "!sim_af afcalc afmaxi afrcnvg afacnvg afrelax uac Pbldg uPb\n     1"
    mf = "!sim_mf slae rs maxi   relcnvg   abscnvg relax gamma ucc\n    2"
    assert af in text and mf in text
    variant = tmp_path / "steady.prj"
    variant.write_text(text.replace(af, af[:-1] + "0", 1).replace(mf, mf[:-1] + "1", 1))
    p = read_prj(variant)
    out = simulate_project(p, **TIGHT)
    model, state, drivers = project_to_model(p)
    expected = p.path_flows(steady(model, state, drivers, **TIGHT)["air.q"])
    assert out["time"].tolist() == [0.0] and out["path_flow"].shape == (1, len(p.paths))
    torch.testing.assert_close(out["path_flow"][0], expected, rtol=1e-12, atol=1e-15)


def _run_control_variant(tmp_path, name, *edits):
    text = (DATA / "test_OneZoneWthCtm.prj").read_text()
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new, 1)
    variant = tmp_path / name
    variant.write_text(text)
    return read_prj(variant)


def test_simulate_project_refuses_a_method_other_than_trapezoid(tmp_path):
    from noodl.apps.building_physics.simulation import simulate_project

    head = "!mf_solver sim_1dz sim_1dd   celldx  sim_vjt udx\n     "
    p = _run_control_variant(tmp_path, "cvode.prj", (head + "0", head + "2"))
    with pytest.raises(ValueError, match="'cvode'"):
        simulate_project(p)


def test_simulate_project_refuses_transient_airflow_with_steady_contaminants(tmp_path):
    from noodl.apps.building_physics.simulation import simulate_project

    head = "!sim_mf slae rs maxi   relcnvg   abscnvg relax gamma ucc\n    "
    p = _run_control_variant(tmp_path, "steady_mf.prj", (head + "2", head + "1"))
    with pytest.raises(ValueError, match="transient airflow with steady contaminants"):
        simulate_project(p)


def test_simulate_project_refuses_a_run_that_is_not_whole_time_steps(tmp_path):
    from noodl.apps.building_physics.simulation import simulate_project

    p = _run_control_variant(tmp_path, "ragged.prj",
                             ("Jan01 24:00:00  00:05:00", "Jan01 23:58:00  00:05:00"))
    with pytest.raises(ValueError, match="not a whole number"):
        simulate_project(p)


def test_a_run_control_without_contaminants_builds_no_species_layer(tmp_path):
    import shutil

    from noodl.apps.building_physics.simulation import simulate_project

    for suffix in ("wth", "ctm"):
        shutil.copy(DATA / f"scheduled_three_zone.{suffix}", tmp_path)
    text = (DATA / "scheduled_three_zone.prj").read_text()
    head = "!sim_mf slae rs maxi   relcnvg   abscnvg relax gamma ucc\n    "
    assert head + "2" in text
    variant = tmp_path / "scheduled_three_zone.prj"
    variant.write_text(text.replace(head + "2", head + "0", 1))
    p = read_prj(variant)
    assert p.run_control.contaminants == "none" and p.sources and p.species
    model, _state, drivers = project_to_model(p)
    assert "species" not in model.transport
    assert not any(k.startswith("species.") for k in drivers)
    assert not any(i.key.startswith("species.") for i in p.inputs_for(model))
    assert any(i.key.startswith("species.") for i in p.inputs)
    out = simulate_project(p, **TIGHT)
    assert "air.q" in out and not any(k.startswith("species.") for k in out)


def test_the_zone_air_mass_source_needs_dodmdt_and_a_zone_with_volume():
    from noodl.apps.building_physics.simulation import _air_mass_source

    stack = DATA / "test_OneZoneWthCtmStack.prj"
    p = read_prj(stack)
    assert p.run_control.zone_mass_change
    assert _air_mass_source(project_to_model(p)[0]) is not None
    _set_dod_mdt(p, False)
    assert _air_mass_source(project_to_model(p)[0]) is None
    # dodMdt with no zone of positive volume: no source (an empty node set would fail).
    p = read_prj(stack)
    for node in p.net.graph.nodes:
        p.net.graph.nodes[node]["volume"] = 0.0
    p.net._cache.clear()
    assert _air_mass_source(project_to_model(p, species=False)[0]) is None


# ------------------------------------------------------------- final-review regressions
def test_air_mass_change_defaults_to_on_exactly_when_the_model_carries_the_source():
    """The default `air_mass_change=None` keeps dM/dt in the balance of a model that carries
    a `ZoneAirMassChange` source (a dodMdt project) and is off for one without; an explicit
    False opts out."""
    project = read_prj(THREE)
    _set_dod_mdt(project, True)
    model, state, drivers = project_to_model(project)
    step_up = [Input(TimeSeries([0.0, 300.0, 600.0], [293.15, 303.15, 303.15],
                                interpolation="step"), "T", at="one")]
    times = [0.0, 300.0, 600.0]
    default = simulate(model, state, drivers, times, inputs=step_up, **TIGHT)
    on = simulate(model, state, drivers, times, inputs=step_up, air_mass_change=True, **TIGHT)
    off = simulate(model, state, drivers, times, inputs=step_up, air_mass_change=False, **TIGHT)
    assert torch.equal(default["air.q"], on["air.q"])
    assert (default["air.q"][-1] - off["air.q"][-1]).abs().max().item() > 1e-4
    _set_dod_mdt(project, False)
    model, state, drivers = project_to_model(project)
    plain = simulate(model, state, drivers, times, inputs=step_up, **TIGHT)
    f = project.path_flows(plain["air.q"][-1])
    assert (f[0] - f[1]).abs().item() < 1e-12


@pytest.mark.parametrize("key", ["V_mett", "Tamb"])
def test_a_whole_driver_input_on_a_key_nobody_reads_is_refused(key):
    model, state, drivers = project_to_model(read_prj(THREE))
    typo = [Input(TimeSeries([0.0, 600.0], [1.0, 2.0]), key, name="wind typo")]
    with pytest.raises(KeyError, match=rf"wind typo.*{key}"):
        simulate(model, state, drivers, [0.0, 300.0], inputs=typo)


def _steady_variant(tmp_path, source):
    text = source.read_text()
    af = "!sim_af afcalc afmaxi afrcnvg afacnvg afrelax uac Pbldg uPb\n     1"
    mf = "!sim_mf slae rs maxi   relcnvg   abscnvg relax gamma ucc\n    2"
    assert af in text and mf in text
    variant = tmp_path / source.name
    variant.write_text(text.replace(af, af[:-1] + "0", 1).replace(mf, mf[:-1] + "1", 1))
    for extra in (".wth", ".ctm"):
        if source.with_suffix(extra).exists():
            variant.with_suffix(extra).write_text(source.with_suffix(extra).read_text())
    return read_prj(variant)


def test_simulate_project_steady_refuses_unread_and_overlapping_inputs(tmp_path):
    from noodl.apps.building_physics.simulation import simulate_project

    p = _steady_variant(tmp_path, DATA / "scheduled_three_zone.prj")
    with pytest.raises(KeyError, match="V_mett"):
        simulate_project(p, inputs=[Input(TimeSeries([0.0, 1.0], [1.0, 1.0]), "V_mett")])
    clash = Input(TimeSeries([0.0, 1.0], [300.0, 300.0]), "T", index=2, name="mine")
    with pytest.raises(ValueError, match="both write driver 'T'"):
        simulate_project(p, inputs=[clash])


@pytest.mark.parametrize("key, values", [("T", None), ("P_ref", (101325.0, 95000.0)),
                                         ("V_met", (5.23, 1.0))])
def test_a_batched_scheduled_project_with_a_constant_flow_fan_matches_its_instances(key, values):
    """scheduled_three_zone carries a scheduled constant-flow fan, a drive-free kind: a batch
    on `T`, `P_ref` or `V_met` runs, and every instance equals its own unbatched run."""
    p = read_prj(DATA / "scheduled_three_zone.prj")
    model, state, drivers = project_to_model(p)
    if values is None:
        batched = torch.stack([drivers["T"], drivers["T"] + 5.0])
    else:
        batched = torch.tensor(values, dtype=F64)
    times = [0.0, 300.0, 600.0]
    inputs = p.inputs_for(model)
    both = simulate(model, state, {**drivers, key: batched}, times, inputs=inputs, **TIGHT)
    assert both["air.q"].shape[1] == 2
    for i in range(2):
        one = simulate(model, state, {**drivers, key: batched[i]}, times, inputs=inputs, **TIGHT)
        torch.testing.assert_close(both["air.q"][:, i], one["air.q"], rtol=1e-10, atol=1e-15)
        torch.testing.assert_close(both["species.x"][:, i], one["species.x"],
                                   rtol=1e-10, atol=1e-18)


def test_the_start_mass_of_a_step_sees_the_temperatures_a_closure_writes():
    """A closure that writes `"T"` (here: zone one held at 303.15 K while the drivers say
    293.15 K) sets the temperature of both masses of dM/dt, so with nothing changing the
    zone's net inflow is zero in every step."""
    project = read_prj(THREE)
    _set_dod_mdt(project, True)
    model, state, drivers = project_to_model(project)
    i = project.net.node_index("one")

    class HoldZoneOne:
        outputs = ("T",)

        def __call__(self, state, drivers):
            return {"T": scatter(drivers["T"], (i,), torch.tensor(303.15, dtype=F64))}

    model.closures.insert(0, HoldZoneOne())
    out = simulate(model, state, drivers, [0.0, 300.0, 600.0], air_mass_change=True,
                   **TIGHT)
    for row in (1, 2):
        f = project.path_flows(out["air.q"][row])
        assert (f[0] - f[1]).abs().item() < 1e-12

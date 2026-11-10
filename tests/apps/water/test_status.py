"""Link status, settings and check valves: the per-link drivers `"water.status"`,
`"water.setting"` and `"water.one_way"`, and the laws that read them."""

import pytest
import torch

from noodl.apps.water.controls import ACTIVE, CLOSED, OPEN
from noodl.apps.water.elements import M_PER_FT, epanet_minor_coefficient
from noodl.apps.water.network import (
    Junction,
    Pump,
    Reservoir,
    Valve,
    WaterNetwork,
    WaterPipe,
    build_model,
    initial_drivers,
    link_states,
    water_steady,
)

F64 = torch.float64


def _pipe(name, u, v, status="OPEN"):
    return WaterPipe(name, u, v, 500.0, 0.3, 130.0, status=status)


def _heads(model, state):
    return {n: float(state["water.phi"][i]) for i, n in enumerate(model.node_names)}


def _flows(model, state):
    return {n: float(state["water.q"][i]) for i, n in enumerate(model.link_names)}


# ------------------------------------------------------------------ closed pipes
def _closed_pipe_network(**kwargs):
    """R1 (50 m) - P1 - J1 (5 L/s) - P2 - R2 (45 m)."""
    return WaterNetwork(
        junctions=(Junction("J1", 0.0, 0.005),),
        reservoirs=(Reservoir("R1", 50.0), Reservoir("R2", 45.0)),
        pipes=(_pipe("P1", "R1", "J1"), kwargs.pop("p2", _pipe("P2", "J1", "R2"))),
        **kwargs,
    )


@pytest.mark.parametrize("how", ["column", "status_section", "driver"])
def test_a_closed_pipe_carries_no_flow(how):
    if how == "column":
        net = _closed_pipe_network(p2=_pipe("P2", "J1", "R2", status="CLOSED"))
    elif how == "status_section":
        net = _closed_pipe_network(initial_status={"P2": "CLOSED"})
    else:
        net = _closed_pipe_network()
    model, state, drivers = build_model(net)
    if how == "driver":
        drivers = initial_drivers(model, values={"water.status": {"P2": CLOSED}})
    assert drivers["water.status"].tolist() == [OPEN, CLOSED]
    out = water_steady(model, state, drivers)
    q = _flows(model, out)
    assert q["P2"] == 0.0
    assert q["P1"] == pytest.approx(0.005, abs=1e-14)
    # J1 hangs from R1 alone: R1's head less P1's loss at the demand
    pipe = model.kirchhoff["water"].law_for("pipe")[0].pipe
    loss = float(pipe.head_loss(torch.tensor([0.005, 0.005], dtype=F64))[0])
    assert _heads(model, out)["J1"] == pytest.approx(50.0 - loss, abs=1e-9)
    assert link_states(model, out, drivers).tolist() == [OPEN, CLOSED]


# ------------------------------------------------------------------ check valves
def _check_valve_line(low_to_high):
    """R_lo (40 m) - P1 - J1 - P2 (CV) - J2 - P3 - R_hi (50 m), no demand; P2 points from the
    low reservoir to the high one when `low_to_high`, else the other way."""
    ends = ("J1", "J2") if low_to_high else ("J2", "J1")
    return WaterNetwork(
        junctions=(Junction("J1", 0.0), Junction("J2", 0.0)),
        reservoirs=(Reservoir("R_lo", 40.0), Reservoir("R_hi", 50.0)),
        pipes=(_pipe("P1", "R_lo", "J1"), _pipe("P2", *ends, status="CV"),
               _pipe("P3", "J2", "R_hi")),
    )


def test_a_check_valve_closes_against_reverse_head():
    model, state, drivers = build_model(_check_valve_line(low_to_high=True))
    assert drivers["water.one_way"].tolist() == [0.0, 1.0, 0.0]
    out = water_steady(model, state, drivers)
    q = _flows(model, out)
    assert q["P2"] == 0.0
    assert abs(q["P1"]) < 1e-12 and abs(q["P3"]) < 1e-12
    h = _heads(model, out)
    assert h["J1"] == pytest.approx(40.0, abs=1e-9)
    assert h["J2"] == pytest.approx(50.0, abs=1e-9)
    assert link_states(model, out, drivers).tolist() == [OPEN, CLOSED, OPEN]


def test_a_check_valve_opens_forward():
    """Pointing from the high reservoir to the low one, the check valve is an open pipe:
    the same heads and flows as the network without it."""
    model, state, drivers = build_model(_check_valve_line(low_to_high=False))
    out = water_steady(model, state, drivers)
    net = _check_valve_line(low_to_high=False)
    plain = WaterNetwork(junctions=net.junctions, reservoirs=net.reservoirs,
                         pipes=tuple(_pipe(p.name, p.u, p.v) for p in net.pipes))
    ref_model, ref_state, ref_drivers = build_model(plain)
    ref = water_steady(ref_model, ref_state, ref_drivers)
    assert _flows(model, out)["P2"] > 0.01
    torch.testing.assert_close(out["water.q"], ref["water.q"], rtol=0, atol=1e-12)
    torch.testing.assert_close(out["water.phi"], ref["water.phi"], rtol=0, atol=1e-10)
    assert link_states(model, out, drivers).tolist() == [OPEN, OPEN, OPEN]


def test_a_reverse_only_link_closes_against_forward_head():
    """`one_way = -1` (reverse flow only) on the same line: P2 points from the low reservoir
    to the high one, so the water runs against it and the link is open."""
    model, state, drivers = build_model(_check_valve_line(low_to_high=True))
    drivers = {**drivers, "water.one_way": torch.tensor([0.0, -1.0, 0.0], dtype=F64)}
    out = water_steady(model, state, drivers)
    assert _flows(model, out)["P2"] < -0.01
    model, state, drivers = build_model(_check_valve_line(low_to_high=False))
    drivers = {**drivers, "water.one_way": torch.tensor([0.0, -1.0, 0.0], dtype=F64)}
    out = water_steady(model, state, drivers)
    assert _flows(model, out)["P2"] == 0.0


def test_check_valves_and_closed_pipes_need_the_flow_form():
    for net in (_check_valve_line(low_to_high=True),
                _closed_pipe_network(initial_status={"P2": "CLOSED"})):
        with pytest.raises(ValueError, match=r"\['P2'\].*pipe_form='flow'"):
            build_model(net, pipe_form="potential")


# ------------------------------------------------------------------ cut-off junctions
def test_closing_a_link_that_cuts_off_a_junction_is_refused():
    net = WaterNetwork(
        junctions=(Junction("J1", 0.0, 0.002), Junction("J2", 0.0, 0.003)),
        reservoirs=(Reservoir("R1", 50.0),),
        pipes=(_pipe("P1", "R1", "J1"), _pipe("P2", "J1", "J2")),
        initial_status={"P2": "CLOSED"},
    )
    model, state, drivers = build_model(net)
    with pytest.raises(ValueError, match=r"water_steady: junction\(s\) \['J2'\] are cut off "
                                         r"from every reservoir and tank by closed links"):
        water_steady(model, state, drivers)
    # and the same closure given at solve time, as a driver
    model, state, drivers = build_model(_closed_pipe_network())
    open_ = water_steady(model, state, drivers)
    assert _flows(model, open_)["P2"] != 0.0
    closed = torch.tensor([CLOSED, CLOSED], dtype=F64)
    with pytest.raises(ValueError, match=r"\['J1'\] are cut off"):
        water_steady(model, state, {**drivers, "water.status": closed})


def test_a_demand_only_check_valves_pointing_away_reach_is_refused_before_the_solve():
    """R1 - P1 - J1 (demand) - P2 - J2 (demand): with P1 reverse-only (water may leave J1
    for R1, never come) no water reaches J1 or J2, though both are joined to R1 by open
    links; refused by name, in a batch too. A junction with an inflow feeds them
    instead: J1's inflow covers both demands and leaves the rest through P1."""
    net = WaterNetwork(
        junctions=(Junction("J1", 0.0, 0.002), Junction("J2", 0.0, 0.003)),
        reservoirs=(Reservoir("R1", 50.0),),
        pipes=(_pipe("P1", "R1", "J1"), _pipe("P2", "J1", "J2")),
    )
    model, state, drivers = build_model(net)
    away = torch.tensor([-1.0, 0.0], dtype=F64)
    with pytest.raises(ValueError, match=r"water_steady: junction\(s\) \['J1', 'J2'\] have "
                                         r"a demand that no reservoir, tank or inflow"):
        water_steady(model, state, {**drivers, "water.one_way": away})
    both = torch.stack([torch.zeros(2, dtype=F64), away])
    with pytest.raises(ValueError, match=r"\['J1', 'J2'\] have a demand"):
        water_steady(model, state, {**drivers, "water.one_way": both})
    toward = torch.tensor([1.0, 0.0], dtype=F64)
    assert _flows(model, water_steady(model, state, {**drivers, "water.one_way": toward}))[
        "P1"] == pytest.approx(0.005)
    inflow = initial_drivers(model, demand={"J1": -0.01, "J2": 0.003})
    out = water_steady(model, state, {**inflow, "water.one_way": away})
    assert _flows(model, out)["P1"] == pytest.approx(-0.007)


# ------------------------------------------------------------------ the drivers
def test_the_link_drivers_are_per_link():
    """Pipes, then pumps, then valves by kind: statuses from the CLOSED column and
    `[STATUS]`, settings from the valves (a numeric `[STATUS]` overriding one), valves
    ACTIVE, check valves forward-only."""
    net = WaterNetwork(
        junctions=(Junction("A", 0.0, 0.001), Junction("B", 0.0, 0.001),
                   Junction("C", 0.0, 0.001), Junction("D", 0.0, 0.001)),
        reservoirs=(Reservoir("R", 60.0),),
        pipes=(_pipe("P1", "R", "A", status="CV"), _pipe("P2", "A", "B"),
               _pipe("P3", "B", "C"), _pipe("P4", "C", "D"), _pipe("P5", "R", "D")),
        pumps=(Pump("PU", "R", "A", "C1", speed=0.9),),
        valves=(Valve("V1", "A", "B", 0.2, "TCV", 5.0),
                Valve("V2", "B", "C", 0.2, "FCV", 0.002),
                Valve("V3", "C", "D", 0.2, "PBV", 3.0)),
        curves={"C1": ((0.004, 10.0),)},
        initial_status={"P3": "CLOSED", "PU": "CLOSED", "V1": 7.5, "V3": "OPEN"},
    )
    model, _, drivers = build_model(net)
    assert model.link_names == ["P1", "P2", "P3", "P4", "P5", "PU", "V1", "V2", "V3"]
    assert drivers["water.status"].tolist() == [
        OPEN, OPEN, CLOSED, OPEN, OPEN, CLOSED, ACTIVE, ACTIVE, OPEN]
    assert drivers["water.one_way"].tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert drivers["water.setting"].tolist() == [0, 0, 0, 0, 0, 0.9, 7.5, 0.002, 3.0]
    assert set(model.refs.inputs) >= {"water.status", "water.setting", "water.one_way"}


# ------------------------------------------------------------------ valves and pumps
def _device_line(valve=None, pump=None, **kwargs):
    """R1 (50 m) - P1 - A (4 L/s) - DEVICE - B (3 L/s) - P2 - R2 (48 m)."""
    return WaterNetwork(
        junctions=(Junction("A", 0.0, 0.004), Junction("B", 0.0, 0.003)),
        reservoirs=(Reservoir("R1", 50.0), Reservoir("R2", 48.0)),
        pipes=(_pipe("P1", "R1", "A"), _pipe("P2", "B", "R2")),
        valves=() if valve is None else (valve,),
        pumps=() if pump is None else (pump,),
        **kwargs,
    )


_DEVICES = {
    "tcv": Valve("V", "A", "B", 0.2, "TCV", 5.0, minor_loss=2.0),
    "fcv": Valve("V", "A", "B", 0.2, "FCV", 0.002, minor_loss=2.0),
    "pbv": Valve("V", "A", "B", 0.2, "PBV", 1.0, minor_loss=2.0),
    "gpv": Valve("V", "A", "B", 0.2, "GPV", 0.0, minor_loss=2.0, curve="G"),
}
_GPV_CURVE = {"G": ((0.0, 0.0), (0.01, 1.0), (0.05, 5.0))}


@pytest.mark.parametrize("kind", sorted(_DEVICES))
def test_a_closed_valve_carries_no_flow(kind):
    net = _device_line(_DEVICES[kind], curves=_GPV_CURVE, initial_status={"V": "CLOSED"})
    model, state, drivers = build_model(net)
    out = water_steady(model, state, drivers)
    assert _flows(model, out)["V"] == 0.0
    assert link_states(model, out, drivers).tolist()[-1] == CLOSED


@pytest.mark.parametrize("kind", sorted(_DEVICES))
def test_an_open_valve_is_its_minor_loss(kind):
    """Status OPEN ignores the setting: the drop is EPANET's `Km q|q|` with
    `Km = 0.02517 K / d^4` (`epanet_minor_coefficient`)."""
    net = _device_line(_DEVICES[kind], curves=_GPV_CURVE, initial_status={"V": "OPEN"})
    model, state, drivers = build_model(net)
    out = water_steady(model, state, drivers)
    q = _flows(model, out)["V"]
    h = _heads(model, out)
    m = float(epanet_minor_coefficient(torch.tensor(2.0, dtype=F64),
                                       torch.tensor(0.2, dtype=F64)))
    assert abs(q) > 1e-3
    assert h["A"] - h["B"] == pytest.approx(m * q * abs(q), rel=1e-9)
    assert link_states(model, out, drivers).tolist()[-1] == OPEN


def test_an_active_valve_reads_its_setting_from_the_driver():
    """`"water.setting"` is the FCV's flow: changing it changes the solve."""
    model, state, drivers = build_model(_device_line(_DEVICES["fcv"]))
    setting = drivers["water.setting"].clone()
    setting[-1] = 0.0025
    out = water_steady(model, state, {**drivers, "water.setting": setting})
    assert _flows(model, out)["V"] == pytest.approx(0.0025, abs=1e-14)
    assert link_states(model, out, drivers).tolist()[-1] == ACTIVE


def test_a_closed_pump_carries_no_flow():
    net = _device_line(pump=Pump("PU", "A", "B", "C1"), curves={"C1": ((0.004, 10.0),)},
                       initial_status={"PU": "CLOSED"})
    model, state, drivers = build_model(net)
    out = water_steady(model, state, drivers)
    assert _flows(model, out)["PU"] == 0.0
    assert link_states(model, out, drivers).tolist()[-1] == CLOSED


def test_kappa_is_the_head_loss_slope_at_one_foot_per_second():
    """The row scale of a pipe's closed branch: its own `dh/dq` at EPANET's first iterate."""
    model, _, _ = build_model(_closed_pipe_network())
    law = model.kirchhoff["water"].law_for("pipe")[0]
    q = torch.tensor([torch.pi * 0.3**2 / 4 * M_PER_FT] * 2, dtype=F64, requires_grad=True)
    (slope,) = torch.autograd.grad(law.pipe.head_loss(q).sum(), q)
    torch.testing.assert_close(law.kappa, slope, rtol=1e-12, atol=0)


# ------------------------------------------------------------------ cut off, no demand
def test_a_cut_off_group_of_two_junctions_is_held_at_the_average_of_its_far_ends():
    """C1 - P4 - C2, no demand, joined to the rest only by the closed P5 (A - C1) and P6
    (C2 - B): both sit at EPANET's limit, the average of A and B, every flow into them
    exactly zero; the rest of the network is the network without them."""
    junctions = (Junction("A", 0.0, 0.005), Junction("B", 0.0, 0.003),
                 Junction("C1", 0.0), Junction("C2", 0.0))
    pipes = (_pipe("P1", "R1", "A"), _pipe("P2", "A", "B"), _pipe("P3", "B", "R2"),
             _pipe("P4", "C1", "C2"), _pipe("P5", "A", "C1", status="CLOSED"),
             _pipe("P6", "C2", "B", status="CLOSED"))
    reservoirs = (Reservoir("R1", 50.0), Reservoir("R2", 45.0))
    net = WaterNetwork(junctions=junctions, reservoirs=reservoirs, pipes=pipes)
    model, state, drivers = build_model(net)
    out = water_steady(model, state, drivers)
    h, q = _heads(model, out), _flows(model, out)
    assert h["C1"] == pytest.approx(0.5 * (h["A"] + h["B"]), abs=1e-12)
    assert h["C2"] == h["C1"]
    assert q["P4"] == 0.0 and q["P5"] == 0.0 and q["P6"] == 0.0
    plain = WaterNetwork(junctions=junctions[:2], reservoirs=reservoirs, pipes=pipes[:3])
    ref_model, ref_state, ref_drivers = build_model(plain)
    ref = _heads(ref_model, water_steady(ref_model, ref_state, ref_drivers))
    assert h["A"] == pytest.approx(ref["A"], abs=1e-12)
    assert h["B"] == pytest.approx(ref["B"], abs=1e-12)
    assert link_states(model, out, drivers).tolist() == [OPEN, OPEN, OPEN, OPEN, CLOSED,
                                                         CLOSED]


def test_water_step_names_itself_when_it_refuses():
    from noodl.apps.water.network import Tank, water_step

    net = WaterNetwork(
        junctions=(Junction("J1", 0.0, 0.002), Junction("J2", 0.0, 0.003)),
        reservoirs=(Reservoir("R1", 50.0),),
        tanks=(Tank("T", 40.0, 5.0, 0.0, 10.0, 5.0),),
        pipes=(_pipe("P1", "R1", "J1"), _pipe("P2", "J1", "T"),
               _pipe("P3", "J1", "J2", status="CLOSED")),
    )
    model, state, drivers = build_model(net, tank_scheme="backward_euler")
    from noodl.apps.water.network import initial_state

    with pytest.raises(ValueError, match=r"^water_step: junction\(s\) \['J2'\]"):
        water_step(model, initial_state(model), drivers, 60.0)


# ------------------------------------------------------------------ the limit program
def _reshut_network():
    """The re-shut case of the docstring above, with K (4 L/s) between R1 and R2 so that
    the solve has flow somewhere."""
    return WaterNetwork(
        junctions=(Junction("J1", 0.0), Junction("J2", 0.0), Junction("K", 0.0, 0.004)),
        reservoirs=(Reservoir("R1", 100.0), Reservoir("R2", 50.0), Reservoir("Ra", 90.0),
                    Reservoir("Rb", 60.0), Reservoir("Rc", 65.0),
                    *(Reservoir(f"Z{i}", 0.0) for i in range(10))),
        pipes=(_pipe("Pa", "Ra", "J1", "CV"), _pipe("P7", "J1", "J2", "CV"),
               _pipe("Pb", "Rb", "J2", "CV"), _pipe("Pc", "J1", "Rc", "CLOSED"),
               *(_pipe(f"PZ{i}", "J2", f"Z{i}", "CLOSED") for i in range(10)),
               _pipe("P3", "R1", "K"), _pipe("P4", "K", "R2")),
    )


def test_a_check_valve_opened_on_the_way_is_shut_again():
    """Through `water_steady`: J1 = J2 = 90 exactly (tied to Ra through the open Pa and
    P7), Pb CLOSED, every flow into the group exactly zero; the ten reservoirs at 0 and Rc
    do not enter."""
    model, state, drivers = build_model(_reshut_network())
    out = water_steady(model, state, drivers)
    h, q = _heads(model, out), _flows(model, out)
    assert h["J1"] == 90.0 and h["J2"] == 90.0
    assert all(q[name] == 0.0 for name in model.link_names if name not in ("P3", "P4"))
    states = dict(zip(model.link_names, link_states(model, out, drivers).tolist(),
                      strict=True))
    assert states["Pa"] == OPEN and states["P7"] == OPEN and states["Pb"] == CLOSED
    assert states["Pc"] == CLOSED and all(states[f"PZ{i}"] == CLOSED for i in range(10))


def test_a_demand_junction_behind_a_closed_check_valve_fed_otherwise_is_solved():
    """The refusal of a demand junction among the free nodes does not fire where the water
    reaches it another way (it is then reliable): D with a demand, a check valve D -> R1
    that must shut and an open pipe R1 -> D that feeds it."""
    net = WaterNetwork(
        junctions=(Junction("D", 0.0, 0.002), Junction("K", 0.0, 0.004)),
        reservoirs=(Reservoir("R1", 100.0), Reservoir("R2", 50.0)),
        pipes=(_pipe("P1", "D", "R1", "CV"), _pipe("P2", "R1", "D"), _pipe("P3", "R1", "K"),
               _pipe("P4", "K", "R2")),
    )
    model, state, drivers = build_model(net)
    out = water_steady(model, state, drivers)
    q = _flows(model, out)
    assert q["P1"] == 0.0 and q["P2"] == pytest.approx(0.002, abs=1e-12)
    assert link_states(model, out, drivers).tolist() == [CLOSED, OPEN, OPEN, OPEN]


def test_parallel_check_valves_open_together():
    """Two check valves with the same ends the same way share one constraint of the limit
    program (one multiplier): both OPEN, J tied to Ra; a duplicate row would otherwise never
    enter the active set and be called CLOSED."""
    net = WaterNetwork(
        junctions=(Junction("J", 0.0), Junction("K", 0.0, 0.004)),
        reservoirs=(Reservoir("R1", 100.0), Reservoir("R2", 50.0), Reservoir("Ra", 90.0)),
        pipes=(_pipe("A1", "Ra", "J", "CV"), _pipe("A2", "Ra", "J", "CV"),
               _pipe("B", "J", "R1", "CV"), _pipe("C", "R2", "J", "CV"),
               _pipe("P3", "R1", "K"), _pipe("P4", "K", "R2")),
    )
    model, state, drivers = build_model(net)
    out = water_steady(model, state, drivers)
    assert _heads(model, out)["J"] == 90.0
    assert link_states(model, out, drivers).tolist() == [OPEN, OPEN, CLOSED, CLOSED, OPEN, OPEN]


def test_drivers_made_in_inference_mode_are_read_uncached():
    """An inference-mode tensor counts no versions, so `link_flags` cannot remember it; the
    solve reads it every time instead of failing."""
    model, state, drivers = build_model(_closed_pipe_network())
    reference = water_steady(model, state, drivers)
    with torch.inference_mode():
        cloned = {k: v.clone() for k, v in drivers.items()}
    out = water_steady(model, state, cloned)
    torch.testing.assert_close(out["water.q"], reference["water.q"], rtol=0, atol=0)
    cloned["water.status"] = initial_drivers(model, values={"water.status": {"P2": CLOSED}})[
        "water.status"]
    assert _flows(model, water_steady(model, state, cloned))["P2"] == 0.0

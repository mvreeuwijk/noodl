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

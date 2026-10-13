"""Pumps and control rules: the curve arithmetic, the readers' refusals by name, SWMM's rule
semantics (controls.c) and the setting-dependent regulator flows. The live-SWMM parity of the
whole is `tests/verification/test_sewer_pumps_controls_parity.py`."""

import datetime
from pathlib import Path

import pytest
import torch

from noodl.apps.inpfile import InpLine
from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.controls import (
    ControlEngine,
    ControlSet,
    parse_rules,
    swmm_datetime,
)
from noodl.apps.sewer.network import Pump
from noodl.apps.sewer.pumps import (
    PumpSet,
    interval_lookup,
    mod_pump_flow,
    pump_curve_flow,
    table_slope,
)
from noodl.apps.sewer.structures import (
    Opening,
    OrificeSet,
    WeirSet,
    orifice_flow,
    outlet_flow,
    weir_flow,
)
from noodl.apps.sewer.xsect_tables import QCF

DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"
F64 = torch.float64


def _t(*x):
    return torch.tensor(x, dtype=F64)


# ----------------------------------------------------------------------------- curves
def test_interval_lookup_is_swmm_table_intervallookup():
    cx = torch.tensor([[1.0, 2.0, 3.0, torch.inf]], dtype=F64)
    cy = torch.tensor([[10.0, 20.0, 30.0, 30.0]], dtype=F64)
    # first entry whose x exceeds the argument; the last y beyond the last x
    got = [float(interval_lookup(_t(x), cx, cy)) for x in (0.0, 0.99, 1.0, 1.5, 2.0, 3.0, 9.0)]
    assert got == [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 30.0]


def test_table_slope_is_the_segment_containing_x():
    cx = torch.tensor([[0.0, 1.0, 3.0, torch.inf]], dtype=F64)
    cy = torch.tensor([[0.0, 2.0, 2.0, 2.0]], dtype=F64)
    n = torch.tensor([3])
    got = [float(table_slope(_t(x), cx, cy, n)) for x in (-1.0, 0.5, 1.0, 2.0, 5.0)]
    assert got == [2.0, 2.0, 2.0, 0.0, 0.0]


def _pump(kind, curve):
    return Pump("P", "A", "B", kind, curve)


def test_pump_curve_flow_by_type():
    """PUMP3 follows the head delivered and PUMP5 divides it by the speed squared; PUMP4 and
    PUMP2 the inlet depth, PUMP1 the volume; the flow is times the setting, never negative."""
    pumps = [_pump("type3", ((0.0, 2.0), (0.3, 0.1))), _pump("type5", ((0.0, 2.0), (0.3, 0.1))),
             _pump("type4", ((0.0, 1.0), (0.0, 0.2))), _pump("type2", ((0.5, 1.0), (0.1, 0.2))),
             _pump("type1", ((10.0, 20.0), (0.1, 0.2))), _pump("ideal", ())]
    ps = PumpSet.build(pumps)
    lift = _t(*[3.2808398950131235] * 6)         # h2 - h1 = 1 m, in ft
    y = _t(*[1.0 / 0.3048] * 6)                  # inlet depth 1 m
    vol = _t(*[15.0 / QCF] * 6)                  # 15 m3 (ft3)
    q, dqdh = pump_curve_flow(ps, _t(1.0, 0.5, 1.0, 1.0, 1.0, 1.0), y, vol, _t(0.0) * lift, lift)
    cms = q * QCF
    assert float(cms[0]) == pytest.approx(0.2, abs=1e-9)           # head 1 m on 0.3 -> 0.1 at 2
    # PUMP5 at speed 0.5: head/0.25 = 4 m, beyond the curve -> last flow, times the speed
    assert float(cms[1]) == pytest.approx(0.1 * 0.5, abs=1e-9)
    assert float(cms[2]) == pytest.approx(0.2, abs=1e-9)           # PUMP4: 1 m is the last x
    assert float(cms[3]) == pytest.approx(0.2)                      # PUMP2: beyond the last x
    assert float(cms[4]) == pytest.approx(0.2)  # PUMP1: 15 m3 -> 20
    assert float(cms[5]) == 0.0                                     # an ideal pump: the caller's
    assert float(dqdh[0]) > 0.0                                     # falling curve: dqdh >= 0


def test_mod_pump_flow_limits_to_the_water_at_the_inlet():
    kind = torch.tensor([2, 2, 0, 4])
    storage = torch.tensor([False, False, False, False])
    full = _t(0.0, 0.0, 100.0, 0.0)
    q = _t(2.0, 2.0, 2.0, 2.0)
    out = mod_pump_flow(kind, q, storage, full, inflow=_t(1.0, 1.0, 1.0, 1.0),
                        outflow=_t(0.0, 0.0, 0.0, 0.0), old_net_inflow=_t(0.0, 0.0, 0.0, 0.0),
                        old_depth=_t(0.1, 5.0, 0.0, 0.0), old_volume=_t(0.0, 0.0, 10.0, 0.0),
                        surf_area=_t(10.0, 10.0, 10.0, 10.0), dt=5.0)
    # depth type: 0.1 + 0.5 (0 + 1 - 2) 5 / 10 < 0 -> the inflow; deep enough -> as asked;
    # PUMP1 -> inflow + oldVolume / dt = 3 caps nothing; PUMP5 unlimited
    assert out.tolist() == [1.0, 2.0, 2.0, 2.0]
    capped = mod_pump_flow(torch.tensor([0]), _t(9.0), torch.tensor([True]), _t(5.0), _t(1.0),
                           _t(0.0), _t(0.0), _t(0.0), _t(10.0), _t(1.0), 5.0)
    assert float(capped) == 3.0                                       # 1 + 10 / 5


# ---------------------------------------------------------------------------- reading
def _fixture(tmp_path, key="A", replace=()):
    from tests.verification.test_sewer_pumps_controls_parity import _system  # noqa: PLC0415
    text = _system((DATA / "pumps_dynwave.inp").read_text(), key)
    for old, new in replace:
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "p.inp"
    path.write_text(text)
    return path


def test_reader_gives_pumps_and_their_curves_in_si(tmp_path):
    net, _, _ = read_swmm_inp(_fixture(tmp_path, "A"))
    (pump,) = net.pumps
    assert (pump.kind, pump.initial_on, pump.startup_depth, pump.shutoff_depth) == (
        "type3", False, 0.6, 0.15)
    assert pump.curve == ((0.0, 2.0, 4.0), (0.30, 0.25, 0.15))
    assert net.structures["pump"] == net.pumps
    net, _, _ = read_swmm_inp(_fixture(tmp_path, "F"))
    assert net.pumps[0].kind == "ideal" and net.pumps[0].curve == ()


def test_us_units_convert_a_pump_curve(tmp_path):
    text = _fixture(tmp_path, "C").read_text().replace("FLOW_UNITS           CMS",
                                                       "FLOW_UNITS           CFS")
    path = tmp_path / "us.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    ys = net.pumps[0].curve[1]
    assert ys[1] == pytest.approx(0.1 * QCF)         # a flow in cfs -> m3/s with SWMM's QCF
    assert net.pumps[0].curve[0][1] == pytest.approx(0.5 * 0.3048)


@pytest.mark.parametrize("old, new, message", [
    ("PC_A  OFF 0.6  0.15", "PC_X  OFF 0.6  0.15", "names curve 'PC_X'"),
    ("PC_A  OFF 0.6  0.15", "PC_A  MAYBE 0.6  0.15", "Status must be ON or OFF"),
    ("PC_A  OFF 0.6  0.15", "PC_A  OFF 0.1  0.15", "ERR_PUMP_LIMITS"),
    ("PC_A  OFF 0.6  0.15", "PC_A  OFF -1  0.15", "negative startup depth"),
    ("PA   WA  OA", "PA   WA  OZ", "names node 'OZ'"),
])
def test_reader_refuses_a_bad_pump_by_name(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        read_swmm_inp(_fixture(tmp_path, "A", [(old, new)]))


def test_reader_refuses_a_non_pump_curve_and_pumps_under_kinwave(tmp_path):
    with pytest.raises(ValueError, match="not one of PUMP1..PUMP5"):
        read_swmm_inp(_fixture(tmp_path, "A", [("PC_A  Pump3", "PC_A  Rating")]))
    with pytest.raises(ValueError, match="pumps are read under dynamic-wave routing only"):
        read_swmm_inp(_fixture(tmp_path, "A", [("FLOW_ROUTING         DYNWAVE",
                                                "FLOW_ROUTING         KINWAVE")]))


def test_network_refuses_an_outfall_inlet_and_a_duplicate_name(tmp_path):
    net, _, _ = read_swmm_inp(_fixture(tmp_path, "A"))
    import dataclasses
    bad = dataclasses.replace(net, pumps=(dataclasses.replace(net.pumps[0], u="OA", v="WA"),))
    with pytest.raises(ValueError, match="inlet 'OA' is an outfall"):
        bad.validate()
    dup = dataclasses.replace(net, pumps=(dataclasses.replace(net.pumps[0], name="CA"),))
    with pytest.raises(ValueError, match="duplicate link name"):
        dup.validate()


# --------------------------------------------------------------------------- controls
def _lines(text):
    return [InpLine("CONTROLS", i + 1, tuple(t.split()), t) for i, t in
            enumerate(text.strip().splitlines())]


LINKS = {"P1": "pump", "P2": "pump", "OR": "orifice", "C": "conduit"}
NODES = {"N1", "N2"}


def _rules(text):
    return parse_rules(_lines(text), links=LINKS, nodes=NODES, path="x.inp")


class _Ctx:
    def __init__(self, depth=1.0, settings=None, flow=0.0):
        self.depth, self.flow = depth, flow
        self.settings = settings or {}

    def node_value(self, name, attribute):
        return {"depth": self.depth, "head": self.depth + 10.0, "inflow": 0.0,
                "volume": 0.0}[attribute]

    def link_flow(self, name):
        return self.flow

    def link_setting(self, name):
        kind = LINKS[name]
        s = self.settings.get(name, 1.0)
        return kind, s, s, 0.0


def _engine(text, **kw):
    return ControlEngine(ControlSet(tuple(_rules(text)), **kw))


def test_premises_are_evaluated_left_to_right_without_precedence():
    # A AND B OR C: A false stops the AND chain and the OR is never reached (controls.c:469-483)
    text = """RULE R
IF NODE N1 DEPTH > 5
AND NODE N1 DEPTH > 0
OR NODE N1 DEPTH > 0.5
THEN PUMP P1 STATUS = OFF"""
    assert _engine(text).evaluate(_Ctx(1.0), 0.5, 0.5, 5.0 / 86400) == {}
    text = text.replace("AND NODE N1 DEPTH > 0\nOR NODE N1 DEPTH > 0.5", "OR NODE N1 DEPTH > 0.5")
    assert _engine(text).evaluate(_Ctx(1.0), 0.5, 0.5, 5.0 / 86400) == {"P1": 0.0}


def test_priority_and_action_order():
    text = """RULE low
IF NODE N1 DEPTH > 0
THEN PUMP P1 SETTING = 0.2
PRIORITY 1
RULE high
IF NODE N1 DEPTH > 0
THEN PUMP P1 SETTING = 0.7
PRIORITY 2
RULE tie
IF NODE N1 DEPTH > 0
THEN PUMP P1 SETTING = 0.9
PRIORITY 2"""
    # the higher priority wins; of equal priority the earlier rule
    assert _engine(text).evaluate(_Ctx(), 0.5, 0.5, 1e-4) == {"P1": 0.7}
    # within one rule the action written LAST is visited first and stays (equal priority)
    two = "RULE r\nIF NODE N1 DEPTH > 0\nTHEN PUMP P1 STATUS = ON\nAND PUMP P1 STATUS = OFF"
    assert _engine(two).evaluate(_Ctx(), 0.5, 0.5, 1e-4) == {"P1": 0.0}


def test_else_branch_and_unchanged_targets_are_dropped():
    text = "RULE r\nIF NODE N1 DEPTH > 5\nTHEN PUMP P1 STATUS = ON\nELSE PUMP P2 STATUS = OFF"
    out = _engine(text).evaluate(_Ctx(1.0, {"P1": 0.0}), 0.5, 0.5, 1e-4)
    assert out == {"P2": 0.0}
    again = _engine(text).evaluate(_Ctx(1.0, {"P2": 0.0}), 0.5, 0.5, 1e-4)
    assert again == {}


def test_curve_and_pid_actions_use_the_last_comparison():
    curve = ControlSet(tuple(_rules("RULE r\nIF NODE N1 DEPTH > 0\n"
                                    "THEN ORIFICE OR SETTING = CURVE C1")),
                       curves={"C1": ((0.0, 2.0), (0.0, 1.0))})
    assert ControlEngine(curve).evaluate(_Ctx(0.5), 0.5, 0.5, 1e-4) == {"OR": 0.25}
    pid = ControlEngine(ControlSet(tuple(_rules(
        "RULE r\nIF NODE N1 DEPTH <> 2.0\nTHEN ORIFICE OR SETTING = PID 0.5 0.0 0.0"))))
    # e0 = (2 - 1) / 2 = 0.5, update = Kp (e0 - e1) = 0.25 on the target 1 -> clipped at 1
    assert pid.evaluate(_Ctx(1.0, {"OR": 0.5}), 0.5, 0.5, 5.0 / 86400) == {"OR": 0.75}


def test_clock_premises_use_swmm_time_with_its_one_millisecond_offset():
    start = datetime.datetime(2026, 1, 1, 6, 0, 0)
    at = swmm_datetime(start, 0.0)
    assert at - int(at) == pytest.approx((6 * 3600 + 0.001) / 86400)
    text = "RULE r\nIF SIMULATION TIME >= 00:01:00\nTHEN PUMP P1 STATUS = OFF"
    engine = _engine(text, start=start)
    assert engine.evaluate(_Ctx(), at, 59.0 / 86400, 1e-4) == {}
    assert engine.evaluate(_Ctx(), at, 61.0 / 86400, 1e-4) == {"P1": 0.0}
    # a time EQUALS within half a step
    eq = _engine("RULE r\nIF SIMULATION TIME = 00:01:00\nTHEN PUMP P1 STATUS = OFF", start=start)
    assert eq.evaluate(_Ctx(), at, 62.0 / 86400, 5.0 / 86400) == {"P1": 0.0}
    assert eq.evaluate(_Ctx(), at, 64.0 / 86400, 5.0 / 86400) == {}


@pytest.mark.parametrize("text, message", [
    ("RULE r\nIF NODE N1 DEPTH > 1\nTHEN CONDUIT C STATUS = CLOSED", "closed conduit"),
    ("RULE r\nIF GAGE G1 INTENSITY > 1\nTHEN PUMP P1 STATUS = ON", "rain gage"),
    ("RULE r\nIF LINK C DEPTH > 1\nTHEN PUMP P1 STATUS = ON", "not modelled"),
    ("RULE r\nIF NODE N1 MAXDEPTH > 1\nTHEN PUMP P1 STATUS = ON", "not modelled"),
    ("VARIABLE v = NODE N1 DEPTH", "VARIABLE"),
    ("EXPRESSION e = v + 1", "EXPRESSION"),
    ("RULE r\nIF NODE N9 DEPTH > 1\nTHEN PUMP P1 STATUS = ON", "node 'N9' is unknown"),
    ("RULE r\nIF NODE N1 DEPTH > 1\nTHEN PUMP C STATUS = ON", "not a pump"),
    ("RULE r\nIF NODE N1 DEPTH > 1\nTHEN ORIFICE OR SETTING = 2", "between 0 and 1"),
    ("RULE r\nTHEN PUMP P1 STATUS = ON", "THEN must follow"),
    ("RULE r\nIF NODE N1 DEPTH >> 1\nTHEN PUMP P1 STATUS = ON", "relational operator"),
])
def test_rules_refuse_what_swmm_rejects_or_we_do_not_model(text, message):
    with pytest.raises(ValueError, match=message):
        _rules(text)


def test_control_set_validation_names_a_missing_curve():
    from noodl.apps.sewer.network import Manhole, Outfall, Pipe, SewerNetwork
    net = SewerNetwork(
        manholes=(Manhole("N1", 1.0),), pipes=(Pipe("C", "N1", "N2", 10.0, 0.5, 0.013, 0.01),),
        outfalls=(Outfall("N2", 0.0),))
    bad = ControlSet(tuple(_rules("RULE r\nIF NODE N1 DEPTH > 0\n"
                                  "THEN ORIFICE OR SETTING = CURVE NOPE")))
    with pytest.raises(ValueError, match="action names orifice 'OR'"):
        bad.validate(net)
    net, _, _ = read_swmm_inp(DATA / "controls_dynwave.inp")
    assert net.controls is not None and net.controls.rule_step == 0.0
    net.controls.validate(net)


def test_reader_checks_control_curves_and_files(tmp_path):
    from tests.verification.test_sewer_pumps_controls_parity import _system  # noqa: PLC0415
    text = _system((DATA / "controls_dynwave.inp").read_text(), "4")
    bad = tmp_path / "bad.inp"
    bad.write_text(text.replace("CC4 Control", "CC4 Rating"))
    with pytest.raises(ValueError, match="not a CONTROL curve"):
        read_swmm_inp(bad)
    kin = tmp_path / "kin.inp"
    kin.write_text(text.replace("FLOW_ROUTING         DYNWAVE", "FLOW_ROUTING         KINWAVE"))
    with pytest.raises(ValueError, match="read under dynamic-wave routing only"):
        read_swmm_inp(kin)


# ------------------------------------------------------- setting-dependent regulators
def _orifice(shape_bottom=False):
    opening = Opening.from_si([0], [0.5], [0.0])
    return OrificeSet.build(opening, [shape_bottom], [0.65], [False], 5.0)


@pytest.mark.parametrize("bottom", [False, True])
def test_orifice_setting_one_is_the_open_orifice_and_zero_closes_it(bottom):
    orf = _orifice(bottom)
    args = (_t(12.0), _t(9.0), _t(10.0), _t(8.0), _t(10.5))
    full = orifice_flow(orf, *args)
    same = orifice_flow(orf.at_setting(_t(1.0)), *args)
    assert float(same.flow) == pytest.approx(float(full.flow), rel=1e-14)
    half = orifice_flow(orf.at_setting(_t(0.5)), *args)
    assert 0.0 < float(half.flow) < float(full.flow)
    shut = orifice_flow(orf.at_setting(_t(0.0)), *args)
    assert float(shut.flow) == 0.0 and float(shut.dqdh) == 0.0


def test_weir_setting_raises_the_crest_and_zero_closes_it():
    opening = Opening.from_si([2], [1.0], [1.0])           # RECT_OPEN-like (shape 2)
    weir = WeirSet.build(opening, [0], [1.84], [0.0], [0.0], [True], [False])
    args = (_t(16.0), _t(8.0), _t(10.0), _t(8.0), _t(10.0))
    full = weir_flow(weir, *args)
    assert float(weir_flow(weir.at_setting(_t(1.0)), *args).flow) == pytest.approx(
        float(full.flow), rel=1e-14)
    raised = weir_flow(weir.at_setting(_t(0.5)), *args)   # crest 0.5 m higher: less head
    assert 0.0 < float(raised.flow) < float(full.flow)
    assert float(weir_flow(weir.at_setting(_t(0.0)), *args).flow) == 0.0


def test_outlet_flow_scales_with_its_setting():
    from noodl.apps.sewer.structures import OutletSet
    out = OutletSet.build([("functional", 0.1, 1.5)], [False], [False])
    args = (_t(12.0), _t(8.0), _t(10.0), _t(8.0), _t(10.0))
    full = outlet_flow(out, *args).flow
    assert float(outlet_flow(out, *args, setting=_t(0.4)).flow) == pytest.approx(
        0.4 * float(full))
    assert float(outlet_flow(out, *args, setting=_t(0.0)).flow) == 0.0


# --------------------------------------------------------------------- simulation
def test_pump_thresholds_orifice_rate_and_state_keys(tmp_path):
    net, _, _ = read_swmm_inp(_fixture(tmp_path, "A"))
    model, state, drivers = build_model(net, iterate_tol=1e-12, iterate_max=500)
    hyd = model.dynamic_wave
    assert hyd.controlled and hyd.setting_names == ["PA"]
    assert {"sewer.setting", "sewer.target_setting", "sewer.time_last_set"} <= set(state)
    assert float(state["sewer.setting"]) == 0.0                        # Status OFF
    _, hist = simulate_dynamic_wave(model, state, drivers, steps=60,
                                    record=("water.q", "sewer.setting"))
    i = model.link_names.index("PA")
    on = hist["sewer.setting"][:, 0]
    assert float(on.max()) == 1.0 and float(hist["water.q"][:, i].max()) > 0.1


def test_ideal_pump_flooding_inlet_is_refused_by_name(tmp_path):
    net, _, _ = read_swmm_inp(_fixture(tmp_path, "F"))
    model, state, drivers = build_model(net, iterate_tol=1e-10, iterate_max=500)
    hyd = model.dynamic_wave
    hyd.check_pump_inlets(state)                                      # below the top: fine
    flooded = dict(state, **{"sewer.junction_depth": hyd.flood_depth.clone()})
    with pytest.raises(RuntimeError, match="ideal pump flood"):
        hyd.check_pump_inlets(flooded)

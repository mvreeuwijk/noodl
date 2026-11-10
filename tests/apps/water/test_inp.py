"""The documented EPANET `.inp` subset."""

import dataclasses
from pathlib import Path

import pytest

from noodl.apps.water.controls import CLOSED, OPEN
from noodl.apps.water.inp import FLOW_UNITS, PSI_TO_M, read_epanet_inp
from noodl.apps.water.network import build_model, twoloop

DATA = Path(__file__).resolve().parents[2] / "data" / "water"
FOOT = 0.3048
INCH = 0.0254


def _edit(tmp_path, name, *pairs):
    text = (DATA / name).read_text()
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new, 1)
    path = tmp_path / name
    path.write_text(text)
    return path


# --------------------------------------------------------------------- happy path
def test_the_reader_reproduces_the_builders_fixture():
    """`twoloop()` and `read_epanet_inp(twoloop_si.inp)` must agree FIELD FOR FIELD.

    The fixture builder multiplies by `1e-3` rather than dividing by `1000.0` for exactly
    this reason: `9.0 / 1000.0` is 0.009 and `9.0 * 1e-3` is 0.009000000000000001, and the
    reader necessarily does the latter (it multiplies by the LPS conversion factor).
    """
    net = read_epanet_inp(DATA / "twoloop_si.inp")
    built = twoloop()
    assert net.junctions == built.junctions
    assert net.reservoirs == built.reservoirs
    assert net.pipes == built.pipes
    assert net.nodes() == built.nodes()
    assert net.notes["flow_units"] == "LPS"
    assert net.headloss == "H-W"
    assert net.duration == 0.0


def test_lps_lengths_are_metres_and_diameters_millimetres():
    net = read_epanet_inp(DATA / "twoloop_si.inp")
    pipe = next(p for p in net.pipes if p.name == "P1")
    assert pipe.length == 500.0
    assert pipe.diameter == pytest.approx(0.300, rel=1e-15)
    assert pipe.roughness == 130.0
    assert net.reservoirs[0].head == 50.0


def test_the_trace_variant_is_the_committed_file_with_four_edits():
    """`twoloop_trace.inp` differs from `twoloop_si.inp` only in its title, duration,
    trials and quality lines; the hydraulics it reads must be identical."""
    plain = read_epanet_inp(DATA / "twoloop_si.inp")
    trace = read_epanet_inp(DATA / "twoloop_trace.inp")
    assert trace.junctions == plain.junctions
    assert trace.pipes == plain.pipes
    assert trace.reservoirs == plain.reservoirs
    assert trace.duration == 6 * 3600.0


def test_net1_gpm_and_feet_are_converted_field_by_field():
    net = read_epanet_inp(DATA / "Net1.inp")
    assert net.notes["flow_units"] == "GPM"
    assert net.nodes() == ["10", "11", "12", "13", "21", "22", "23", "31", "32", "9", "2"]
    assert net.reservoirs[0].head == pytest.approx(800 * FOOT, rel=0.0)
    pipe = next(p for p in net.pipes if p.name == "10")
    assert pipe.length == pytest.approx(10530 * FOOT, rel=1e-15)
    assert pipe.diameter == pytest.approx(18 * INCH, rel=1e-15)
    assert pipe.roughness == 100.0
    junction = next(j for j in net.junctions if j.name == "11")
    assert junction.elevation == pytest.approx(710 * FOOT, rel=1e-15)
    assert junction.demand == pytest.approx(150 * FLOW_UNITS["GPM"], rel=1e-15)
    tank = net.tanks[0]
    assert (tank.elevation, tank.init_level, tank.min_level, tank.max_level) == (
        pytest.approx(850 * FOOT, rel=1e-15), pytest.approx(120 * FOOT, rel=1e-15),
        pytest.approx(100 * FOOT, rel=1e-15), pytest.approx(150 * FOOT, rel=1e-15),
    )
    assert tank.diameter == pytest.approx(50.5 * FOOT, rel=1e-15)


def test_net1_pump_curve_controls_pattern_and_times():
    net = read_epanet_inp(DATA / "Net1.inp")
    assert net.pumps == (
        type(net.pumps[0])("9", "9", "10", "1", None, 1.0),
    )
    assert net.curves["1"] == ((1500 * FLOW_UNITS["GPM"], 250 * FOOT),)
    assert [(c.link, c.status, c.trigger, c.node, c.above) for c in net.controls] == [
        ("9", OPEN, "node", "2", False), ("9", CLOSED, "node", "2", True)
    ]
    assert [c.value for c in net.controls] == pytest.approx(
        [110 * FOOT, 140 * FOOT], rel=1e-15
    )
    assert net.patterns["1"] == (
        1.0, 1.2, 1.4, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.6, 0.8
    )
    assert net.demand_pattern == "1"
    assert (net.duration, net.hydraulic_timestep, net.pattern_timestep,
            net.report_timestep) == (86400.0, 3600.0, 7200.0, 3600.0)


def test_every_flow_unit_is_a_positive_conversion():
    assert set(FLOW_UNITS) == {
        "CFS", "GPM", "MGD", "IMGD", "AFD", "LPS", "LPM", "MLD", "CMH", "CMD"
    }
    assert all(value > 0 for value in FLOW_UNITS.values())
    assert FLOW_UNITS["GPM"] == pytest.approx(6.309019640343977e-05, rel=1e-15)
    assert FLOW_UNITS["LPS"] == 1e-3
    assert FLOW_UNITS["CMH"] == pytest.approx(1.0 / 3600.0, rel=1e-15)


# ----------------------------------------------------------------------- refusals
def test_an_unknown_flow_unit_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp", (" Units              \tLPS",
                                              " Units              \tFURLONGS"))
    with pytest.raises(ValueError, match="UNITS 'FURLONGS'"):
        read_epanet_inp(path)


def test_an_emitters_section_with_content_is_read_and_validates(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[EMITTERS]\n;Junction        \tCoefficient",
                  "[EMITTERS]\n;Junction        \tCoefficient\n J1              \t0.5"))
    net = read_epanet_inp(path)
    net.validate()
    assert [e.node for e in net.emitters] == ["J1"]


def test_a_status_section_is_read_into_initial_status(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[STATUS]\n;ID              \tStatus/Setting",
                  "[STATUS]\n;ID              \tStatus/Setting\n P1              \tClosed"))
    net = read_epanet_inp(path)
    assert net.initial_status == {"P1": "CLOSED"}
    net.validate()


def test_an_unknown_section_with_content_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[END]", "[LEAKAGE]\n J1   0.001\n\n[END]"))
    with pytest.raises(ValueError, match=r"\[LEAKAGE\]"):
        read_epanet_inp(path)


def test_a_clocktime_control_is_read_and_accepted(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[CONTROLS]", "[CONTROLS]\n LINK P1 CLOSED AT CLOCKTIME 6 AM"))
    net = read_epanet_inp(path)
    assert net.controls[0].trigger == "clocktime"
    assert net.controls[0].value == 6 * 3600.0
    net.validate()


def test_a_malformed_control_is_refused(tmp_path):
    path = _edit(tmp_path, "Net1.inp",
                 (" LINK 9 OPEN IF NODE 2 BELOW 110",
                  " LINK 9 OPEN WHEN NODE 2 IS BELOW 110"))
    with pytest.raises(ValueError, match="only controls of the form"):
        read_epanet_inp(path)


def test_a_control_with_an_unknown_test_is_refused(tmp_path):
    path = _edit(tmp_path, "Net1.inp",
                 (" LINK 9 OPEN IF NODE 2 BELOW 110",
                  " LINK 9 OPEN IF NODE 2 NEAR 110"))
    with pytest.raises(ValueError, match="BELOW or ABOVE"):
        read_epanet_inp(path)


def test_a_constant_power_pump_is_read_in_kilowatts(tmp_path):
    path = _edit(tmp_path, "Net1.inp", (" 9               \t9               \t10"
                                        "              \tHEAD 1\t;",
                                        " 9               \t9               \t10"
                                        "              \tPOWER 50\t;"))
    net = read_epanet_inp(path)
    assert net.pumps[0].power == pytest.approx(50 * 0.745699872, rel=1e-15)
    net.validate()


def test_a_speed_patterned_pump_is_refused(tmp_path):
    path = _edit(tmp_path, "Net1.inp", ("\tHEAD 1\t;", "\tHEAD 1 PATTERN 1\t;"))
    net = read_epanet_inp(path)
    assert net.pumps[0].speed_pattern == "1"
    with pytest.raises(ValueError, match="SPEED PATTERN"):
        net.validate()


def test_an_unknown_pump_keyword_is_refused(tmp_path):
    path = _edit(tmp_path, "Net1.inp", ("\tHEAD 1\t;", "\tHEAD 1 TORQUE 3\t;"))
    with pytest.raises(ValueError, match="unknown keyword 'TORQUE'"):
        read_epanet_inp(path)


def test_a_volume_curve_that_does_not_cover_the_tank_levels_is_refused(tmp_path):
    """EPANET 2.2 refuses a tank whose minimum or maximum level lies outside its volume
    curve (`input1.c`, `validatetanks`, error 225)."""
    path = _edit(
        tmp_path, "Net1.inp",
        (" 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \t                \t;",
         " 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \tVC1             \t;"),
        ("[CURVES]", "[CURVES]\n VC1 0 0\n VC1 100 5000"),
    )
    net = read_epanet_inp(path)
    assert net.tanks[0].volume_curve == "VC1"
    with pytest.raises(ValueError, match=r"tank '2'.*levels \[30.48, 45.72\] m.*volume curve "
                                         r"'VC1'"):
        net.validate()


def test_a_demand_for_an_unknown_junction_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[CURVES]", "[DEMANDS]\n JX   5\n\n[CURVES]"))
    with pytest.raises(ValueError, match=r"\[DEMANDS\] names \['JX'\]"):
        read_epanet_inp(path)


def test_a_supplemental_demand_overrides_the_junction_demand(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[CURVES]", "[DEMANDS]\n J1   25   Pat9\n\n[CURVES]"))
    net = read_epanet_inp(path)
    junction = next(j for j in net.junctions if j.name == "J1")
    assert junction.demand == pytest.approx(25 * 1e-3, rel=1e-15)
    assert junction.pattern == "Pat9"


def test_a_short_record_is_refused_naming_the_line(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 (" J1              \t0           \t5           \t                \t;",
                  " J1"))
    with pytest.raises(ValueError, match="a junction needs 2 fields"):
        read_epanet_inp(path)


def test_a_non_numeric_field_is_refused_naming_the_field(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 (" J1              \t0           \t5           \t                \t;",
                  " J1              \tsurface     \t5"))
    with pytest.raises(ValueError, match="the elevation must be a number"):
        read_epanet_inp(path)


def test_a_zero_diameter_tank_is_refused(tmp_path):
    path = _edit(tmp_path, "Net1.inp", ("\t150         \t50.5        \t0",
                                        "\t150         \t0           \t0"))
    with pytest.raises(ValueError, match="a cylindrical tank needs a positive one"):
        read_epanet_inp(path)


def test_darcy_weisbach_roughness_is_converted_from_millimetres(tmp_path):
    path = _edit(
        tmp_path, "twoloop_si.inp",
        (" Headloss           \tH-W", " Headloss           \tD-W"),
        ("130         \t0           \tOpen", "0.26        \t0           \tOpen"),
    )
    net = read_epanet_inp(path)
    assert net.headloss == "D-W"
    assert net.pipes[0].roughness == pytest.approx(0.26e-3, rel=1e-15)


def test_darcy_weisbach_roughness_is_converted_from_millifeet(tmp_path):
    path = _edit(tmp_path, "Net1.inp",
                 (" Headloss           \tH-W", " Headloss           \tD-W"),
                 ("\t10530       \t18          \t100", "\t10530       \t18          \t0.85"))
    net = read_epanet_inp(path)
    pipe = next(p for p in net.pipes if p.name == "10")
    assert pipe.roughness == pytest.approx(0.85 * FOOT * 1e-3, rel=1e-15)


def test_the_flow_unit_is_carried_onto_the_options():
    assert read_epanet_inp(DATA / "twoloop_si.inp").options.flow_units == "LPS"
    assert read_epanet_inp(DATA / "Net1.inp").options.flow_units == "GPM"


@pytest.mark.parametrize(
    "name, value, ratio",
    [
        ("twoloop_si.inp", "1.5", 1.5),
        # at or below 1e-3 EPANET reads the ACTUAL kinematic viscosity (`input1.c`):
        # m2/s in an SI file, ft2/s in a US one
        ("twoloop_si.inp", "2.0439e-6", 2.0439e-6 / (1.1e-5 * FOOT**2)),
        ("Net1.inp", "2.2e-5", 2.0),
    ],
)
def test_viscosity_is_a_multiplier_or_an_actual_value_as_in_epanet(
    tmp_path, name, value, ratio
):
    old = " Viscosity          \t1.0"
    path = _edit(tmp_path, name, (old, f" Viscosity          \t{value}"))
    assert read_epanet_inp(path).options.viscosity == pytest.approx(ratio, rel=1e-12)


# --------------------------------------------------------------------- [OPTIONS]
def test_a_default_file_reads_dda_with_epanets_own_defaults():
    """Neither committed fixture writes `DEMAND MODEL`, `MINIMUM PRESSURE`, `REQUIRED
    PRESSURE` or `PRESSURE EXPONENT`, so all four fall back to `WaterOptions`'s defaults."""
    net = read_epanet_inp(DATA / "twoloop_si.inp")
    assert net.options.demand_model == "DDA"
    assert net.options.minimum_pressure == 0.0
    assert net.options.required_pressure == 0.1
    assert net.options.pressure_exponent == 0.5
    assert net.options.specific_gravity == 1.0
    assert net.options.viscosity == 1.0


def test_a_pda_file_reads_the_demand_model_and_the_three_pressures(tmp_path):
    path = _edit(
        tmp_path, "twoloop_si.inp",
        (" Pattern            \t1", " Pattern            \t1\n"
         " DEMAND MODEL       \tPDA\n MINIMUM PRESSURE   \t0\n"
         " REQUIRED PRESSURE  \t60\n PRESSURE EXPONENT  \t0.5"),
    )
    net = read_epanet_inp(path)
    assert net.options.demand_model == "PDA"
    assert net.options.minimum_pressure == 0.0
    assert net.options.required_pressure == 60.0
    assert net.options.pressure_exponent == 0.5
    from noodl.apps.water.network import build_model

    model, _, _ = build_model(net)
    assert len(model.kirchhoff["water"]._withdrawals) == 1


def test_an_unknown_demand_model_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 (" Pattern            \t1", " Pattern            \t1\n"
                  " DEMAND MODEL       \tPDD"))
    with pytest.raises(ValueError, match="DEMAND MODEL 'PDD'"):
        read_epanet_inp(path)


def test_an_unrecognised_option_is_recorded_not_dropped():
    """`Trials`, `Accuracy`, `CHECKFREQ`, ... are solver/report cosmetics this reader does
    not model, but they must be RECORDED rather than silently dropped."""
    net = read_epanet_inp(DATA / "twoloop_si.inp")
    assert "unrecognised_options" in net.notes
    assert "Trials" in net.notes["unrecognised_options"]
    assert "Quality" in net.notes["unrecognised_options"]


def test_required_pressure_in_a_us_units_file_is_converted_from_psi(tmp_path):
    path = _edit(
        tmp_path, "Net1.inp",
        (" Pattern            \t1", " Pattern            \t1\n"
         " DEMAND MODEL       \tPDA\n MINIMUM PRESSURE   \t0\n"
         " REQUIRED PRESSURE  \t20\n PRESSURE EXPONENT  \t0.5"),
    )
    net = read_epanet_inp(path)
    assert net.options.required_pressure == pytest.approx(20 * FOOT / 0.4333, rel=1e-15)


# ------------------------------------------------------- [DEMANDS] categories, [TIMES] units
def test_demand_categories_of_one_junction_add_up(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[CURVES]", "[DEMANDS]\n J1   5\n J1   7\n\n[CURVES]"))
    junction = next(j for j in read_epanet_inp(path).junctions if j.name == "J1")
    assert junction.demand == pytest.approx(12e-3, rel=1e-14)


def test_demand_categories_with_different_patterns_are_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[CURVES]", "[DEMANDS]\n J1   5   Pat1\n J1   7   Pat2\n\n[CURVES]"))
    with pytest.raises(ValueError, match="different patterns"):
        read_epanet_inp(path)


def _with_default_pattern(tmp_path, demands, option="Pat1"):
    pairs = [("[CURVES]", f"[PATTERNS]\n Pat1 1 2\n Pat2 2 1\n\n[DEMANDS]\n{demands}\n\n[CURVES]"),
             (" Pattern            	1", f" Pattern            	{option}")]
    return _edit(tmp_path, "twoloop_si.inp", *pairs)


@pytest.mark.parametrize("demands", [" J1   5\n J1   7   Pat1", " J1   5   Pat1\n J1   7"])
def test_no_pattern_equals_an_explicit_default_pattern_either_way_round(tmp_path, demands):
    """A demand with no pattern uses the [OPTIONS] default pattern, so it is the SAME pattern
    as one naming that default explicitly."""
    junction = next(j for j in read_epanet_inp(_with_default_pattern(tmp_path, demands)).junctions
                    if j.name == "J1")
    assert junction.demand == pytest.approx(12e-3, rel=1e-14)
    assert junction.pattern in ("Pat1", None)


def test_no_pattern_equals_pattern_1_when_no_default_option_is_given(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 (" Pattern            	1", ""),
                 ("[CURVES]", "[PATTERNS]\n 1 1 2\n\n[DEMANDS]\n J1 5\n J1 7 1\n\n[CURVES]"))
    junction = next(j for j in read_epanet_inp(path).junctions if j.name == "J1")
    assert junction.demand == pytest.approx(12e-3, rel=1e-14)


def test_no_pattern_differs_from_a_non_default_pattern(tmp_path):
    path = _with_default_pattern(tmp_path, " J1   5\n J1   7   Pat2")
    with pytest.raises(ValueError, match="different patterns"):
        read_epanet_inp(path)


def test_demand_categories_sharing_a_pattern_add_up(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[CURVES]", "[DEMANDS]\n J1   5   Pat1\n J1   7   Pat1\n\n[CURVES]"))
    junction = next(j for j in read_epanet_inp(path).junctions if j.name == "J1")
    assert junction.demand == pytest.approx(12e-3, rel=1e-14)
    assert junction.pattern == "Pat1"


def _times(tmp_path, body):
    text = (DATA / "twoloop_si.inp").read_text()
    assert "[TIMES]" in text
    head, tail = text.split("[TIMES]", 1)
    rest = tail.split("[", 1)[1] if "[" in tail else ""
    path = tmp_path / "times.inp"
    path.write_text(head + "[TIMES]\n" + body + "\n\n[" + rest)
    return read_epanet_inp(path)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("30 MINUTES", 1800.0), ("30 min", 1800.0), ("90 SECONDS", 90.0),
     ("2 HOURS", 7200.0), ("2", 7200.0), ("1 DAYS", 86400.0), ("1:30", 5400.0),
     ("30 MINS", 1800.0), ("90 secs", 90.0), ("2 hou", 7200.0), ("1 Dayx", 86400.0)],
)
def test_explicit_time_units_are_honoured(tmp_path, text, seconds):
    net = _times(tmp_path, f"Duration {text}\nHydraulic Timestep {text}\n"
                           f"Pattern Timestep {text}\nReport Timestep {text}")
    assert net.duration == seconds
    assert net.hydraulic_timestep == seconds
    assert net.pattern_timestep == seconds
    assert net.report_timestep == seconds


@pytest.mark.parametrize("unit", ["FORTNIGHTS", "HRS", "M", "SE"])
def test_an_unknown_time_unit_is_refused(tmp_path, unit):
    with pytest.raises(ValueError, match="unknown time unit"):
        _times(tmp_path, f"Duration 30 {unit}")


# ------------------------------------------------- every EPANET 2.2 hydraulic section
def _wntr_network(*parts):
    wntr = pytest.importorskip("wntr")
    return Path(wntr.__file__).parent.joinpath(*parts)


def test_net3_reads_status_time_controls_and_clock():
    net = read_epanet_inp(_wntr_network("library", "networks", "Net3.inp"))
    assert net.initial_status["10"] == "CLOSED"
    timed = [c for c in net.controls if c.trigger == "time"]
    assert len(timed) == 14
    assert (timed[0].link, timed[0].status, timed[0].value) == ("10", OPEN, 3600.0)
    assert timed[1].status == CLOSED
    assert timed[1].value == 15 * 3600.0
    assert net.start_clocktime == 0.0
    assert net.duration == 168 * 3600
    # the four tank-level controls are read as before, in metres (Net3 is a GPM file)
    tank_level = [c for c in net.controls if c.trigger == "node"]
    assert [(c.link, c.node, c.above) for c in tank_level] == [
        ("335", "1", False), ("335", "1", True), ("330", "1", False), ("330", "1", True)
    ]
    assert tank_level[0].value == pytest.approx(17.1 * FOOT, rel=1e-15)
    # the CLOSED pipe, the [STATUS] line and the time controls are modelled
    net.validate()


def test_net6_reads_prv_cv_and_power_pump():
    net = read_epanet_inp(_wntr_network("library", "networks", "Net6.inp"))
    prvs = [v for v in net.valves if v.kind == "PRV"]
    assert len(prvs) == 2
    # PRV setting 50 psi -> metres of head, as every other US pressure
    assert prvs[0].setting == pytest.approx(50 * PSI_TO_M, rel=1e-15)
    assert [p.status for p in net.pipes].count("CV") == 1
    powered = [p for p in net.pumps if p.power is not None]
    assert len(powered) == 1
    assert powered[0].curve is None
    assert powered[0].power == pytest.approx(15 * 0.745699872, rel=1e-12)  # hp -> kW


def test_ky10_reads_constant_power_pumps_in_kilowatts_and_the_energy_pattern():
    net = read_epanet_inp(_wntr_network("library", "networks", "ky10.inp"))
    assert all(p.power is not None for p in net.pumps)
    assert min(p.power for p in net.pumps) == pytest.approx(5 * 0.745699872, rel=1e-12)
    assert net.energy.global_pattern == "ENRG1"


def test_anytown_reads_speed_patterns_and_efficiency_curves():
    net = read_epanet_inp(_wntr_network("tests", "networks_for_testing", "Anytown.inp"))
    pumps = {p.name: p for p in net.pumps}
    assert [pumps[name].speed_pattern for name in ("78", "79", "80")] == ["2", "3", "4"]
    assert all(pumps[name].efficiency_curve == "E1" for name in ("78", "79", "80"))
    assert [y for _, y in net.curves["E1"]] == [0.0, 50.0, 65.0, 55.0, 40.0]
    assert [x for x, _ in net.curves["E1"]] == pytest.approx(
        [v * FLOW_UNITS["GPM"] for v in (0, 2000, 4000, 6000, 8000)], rel=1e-15
    )
    # the same file's head curve is converted as (flow, length)
    assert net.curves["2"][0][1] == pytest.approx(300 * FOOT, rel=1e-15)
    assert net.start_clocktime == 18 * 3600.0
    assert net.rule_timestep == net.hydraulic_timestep / 10
    with pytest.raises(ValueError, match="speed pattern"):
        net.validate()


def test_a_curve_used_as_head_and_volume_is_refused(tmp_path):
    path = _edit(
        tmp_path, "Net1.inp",
        (" 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \t                \t;",
         " 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \t1               \t;"),
    )
    with pytest.raises(ValueError, match="curve '1'"):
        read_epanet_inp(path)


def test_a_volume_curve_is_converted_to_metres_and_cubic_metres(tmp_path):
    path = _edit(
        tmp_path, "Net1.inp",
        (" 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \t                \t;",
         " 2               \t850         \t120         \t100         \t150         "
         "\t0           \t0           \tVC1             \tYES"),
        ("[CURVES]", "[CURVES]\n VC1 100 1000\n VC1 150 3000"),
    )
    net = read_epanet_inp(path)
    tank = net.tanks[0]
    assert (tank.volume_curve, tank.overflow) == ("VC1", True)
    assert net.curves["VC1"] == ((100 * FOOT, 1000 * FOOT**3), (150 * FOOT, 3000 * FOOT**3))
    net.validate()


def test_a_tank_overflow_flag_is_read(tmp_path):
    path = _edit(
        tmp_path, "Net1.inp",
        (" 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \t                \t;",
         " 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \t*               \tYES"),
    )
    net = read_epanet_inp(path)
    assert net.tanks[0].volume_curve is None
    assert net.tanks[0].overflow is True
    net.validate()


RULES = """[RULES]
RULE R1
IF NODE J1 PRESSURE ABOVE 20
AND LINK P2 STATUS IS OPEN
OR SYSTEM CLOCKTIME >= 8 PM
THEN LINK P3 STATUS IS CLOSED
AND LINK P4 STATUS IS CLOSED
ELSE LINK P3 STATUS IS OPEN
PRIORITY 2

RULE R2
IF SYSTEM TIME BELOW 6
THEN LINK P5 STATUS IS CLOSED

[CURVES]"""


def test_rules_are_parsed(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp", ("[CURVES]", RULES))
    net = read_epanet_inp(path)
    first, second = net.rules
    assert (first.name, first.priority) == ("R1", 2.0)
    assert [(p.conjunction, p.object, p.id, p.attribute, p.relation, p.value)
            for p in first.premises] == [
        ("IF", "NODE", "J1", "PRESSURE", ">", 20.0),
        ("AND", "LINK", "P2", "STATUS", "=", float(OPEN)),
        ("OR", "SYSTEM", "", "CLOCKTIME", ">=", 20 * 3600.0),
    ]
    assert [(a.link, a.attribute, a.value) for a in first.then_actions] == [
        ("P3", "STATUS", float(CLOSED)), ("P4", "STATUS", float(CLOSED))
    ]
    assert [(a.link, a.attribute, a.value) for a in first.else_actions] == [
        ("P3", "STATUS", float(OPEN))
    ]
    assert second.priority == 0.0
    assert second.premises[0].value == 6 * 3600.0
    assert second.premises[0].relation == "<"
    net.validate()   # rules are no longer refused


def test_a_rule_naming_an_unknown_link_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[CURVES]", "[RULES]\nRULE R\nIF SYSTEM TIME > 1\n"
                              "THEN LINK NOPE STATUS IS OPEN\n\n[CURVES]"))
    with pytest.raises(ValueError, match="NOPE"):
        read_epanet_inp(path)


def test_emitters_are_converted(tmp_path):
    path = _edit(tmp_path, "Net1.inp", ("[CURVES]", "[EMITTERS]\n 10 1.0\n\n[CURVES]"))
    net = read_epanet_inp(path)
    (emitter,) = net.emitters
    assert emitter.node == "10"
    assert emitter.coefficient == pytest.approx(6.309019640343977e-05 / PSI_TO_M**0.5, rel=1e-15)
    assert net.options.emitter_exponent == 0.5
    net.validate()


def test_an_si_emitter_is_only_converted_by_the_flow_unit(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 (" Emitter Exponent   	0.5", " Emitter Exponent   	0.6"),
                 ("[CURVES]", "[EMITTERS]\n J2 2.0\n\n[CURVES]"))
    net = read_epanet_inp(path)
    assert net.emitters[0].coefficient == pytest.approx(2.0e-3, rel=1e-15)
    assert net.options.emitter_exponent == 0.6


def test_an_emitter_at_a_non_junction_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp", ("[CURVES]", "[EMITTERS]\n R1 2.0\n\n[CURVES]"))
    with pytest.raises(ValueError, match="not a junction"):
        read_epanet_inp(path)


def test_the_chezy_manning_headloss_is_read_and_validates(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 (" Headloss           \tH-W", " Headloss           \tC-M"))
    net = read_epanet_inp(path)
    assert net.headloss == "C-M"
    net.validate()
    build_model(net)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("12 am", 0.0), ("12 AM", 0.0), ("12 pm", 12 * 3600.0), ("6:30 PM", 18.5 * 3600.0),
     ("18", 18 * 3600.0), ("1:15:30 am", 4530.0), ("0", 0.0)],
)
def test_start_clocktime_reads_am_pm_and_the_24_hour_clock(tmp_path, text, seconds):
    assert _times(tmp_path, f"Start ClockTime {text}").start_clocktime == seconds


def test_the_other_time_options_are_read(tmp_path):
    net = _times(tmp_path, "Hydraulic Timestep 1\nRule Timestep 0:05\nPattern Start 2\n"
                           "Report Start 3:30")
    assert net.rule_timestep == 300.0
    assert net.pattern_start == 7200.0
    assert net.report_start == 12600.0
    assert _times(tmp_path, "Hydraulic Timestep 1").rule_timestep == 360.0


def test_every_control_form_is_read(tmp_path):
    text = (" LINK 9 CLOSED AT CLOCKTIME 6:30 PM\n LINK 9 OPEN AT TIME 1:30\n"
            " LINK 9 0.5 AT TIME 2\n LINK 9 0.8 IF NODE 2 ABOVE 140")
    net = read_epanet_inp(_edit(tmp_path, "Net1.inp", ("[CONTROLS]", "[CONTROLS]\n" + text)))
    clock, timed, speed_timed, speed_level = net.controls[:4]
    assert (clock.trigger, clock.status, clock.setting, clock.value) == (
        "clocktime", CLOSED, None, 18.5 * 3600.0
    )
    assert (timed.trigger, timed.status, timed.value) == ("time", OPEN, 5400.0)
    assert (speed_timed.status, speed_timed.setting, speed_timed.value) == (None, 0.5, 7200.0)
    assert (speed_level.trigger, speed_level.node, speed_level.above, speed_level.setting) == (
        "node", "2", True, 0.8
    )
    assert speed_level.value == pytest.approx(140 * FOOT, rel=1e-15)
    net.validate()


def test_a_junction_pressure_control_is_converted_from_psi(tmp_path):
    path = _edit(tmp_path, "Net1.inp",
                 ("[CONTROLS]", "[CONTROLS]\n LINK 9 OPEN IF NODE 10 BELOW 30"))
    net = read_epanet_inp(path)
    assert net.controls[0].value == pytest.approx(30 * PSI_TO_M, rel=1e-15)
    net.validate()


def test_a_numeric_setting_on_a_pipe_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[CONTROLS]", "[CONTROLS]\n LINK P1 0.5 AT TIME 2"))
    with pytest.raises(ValueError, match="no meaning"):
        read_epanet_inp(path)


def test_a_status_line_reads_a_word_or_a_setting_in_si(tmp_path):
    path = _edit(tmp_path, "Net1.inp", ("[CURVES]", "[STATUS]\n 9 0.7\n 10 Closed\n\n[CURVES]"))
    net = read_epanet_inp(path)
    assert net.initial_status == {"9": 0.7, "10": "CLOSED"}
    with pytest.raises(ValueError, match=r"\[STATUS\]"):
        net.validate()


def test_energy_options_and_pump_overrides_are_read(tmp_path):
    path = _edit(tmp_path, "Net1.inp",
                 ("[CURVES]", "[CURVES]\n E1 1500 60\n"),
                 ("[END]", "[ENERGY]\n Global Efficiency 70\n Global Price 0.12\n"
                           " Global Pattern 1\n Demand Charge 4\n Pump 9 Efficiency E1\n"
                           " Pump 9 Price 0.2\n Pump 9 Pattern 1\n\n[END]"))
    net = read_epanet_inp(path)
    assert (net.energy.global_efficiency, net.energy.global_price,
            net.energy.global_pattern, net.energy.demand_charge) == (70.0, 0.12, "1", 4.0)
    pump = net.pumps[0]
    assert (pump.efficiency_curve, pump.price, pump.price_pattern) == ("E1", 0.2, "1")
    assert net.curves["E1"] == ((1500 * FLOW_UNITS["GPM"], 60.0),)
    net.validate()  # energy data is reporting only: it is never refused


def test_a_check_valve_pipe_is_read_and_validated(tmp_path):
    cv = read_epanet_inp(_edit(
        tmp_path, "twoloop_si.inp",
        ("130         \t0           \tOpen", "130         \t0           \tCV"),
    ))
    assert cv.pipes[0].status == "CV"
    cv.validate()


# ------------------------------------------------------------- review fix round 1
def test_a_us_emitter_coefficient_honours_the_emitter_exponent(tmp_path):
    """q = C p^n: C_si = f C_us / PSI_TO_M**n for any n (EPANET `convertunits`), not sqrt."""
    path = _edit(
        tmp_path, "Net1.inp",
        (" Emitter Exponent   	0.5", " Emitter Exponent   	0.6"),
        ("[CURVES]", "[EMITTERS]\n 10 2.0\n\n[CURVES]"),
    )
    net = read_epanet_inp(path)
    assert net.options.emitter_exponent == 0.6
    expected = 2.0 * FLOW_UNITS["GPM"] / PSI_TO_M**0.6
    assert net.emitters[0].coefficient == pytest.approx(expected, rel=1e-15)
    # the SI value reproduces the file's flow at a pressure of 10 psi
    head = 10 * PSI_TO_M
    assert net.emitters[0].coefficient * head**0.6 == pytest.approx(
        2.0 * 10**0.6 * FLOW_UNITS["GPM"], rel=1e-12
    )


def test_a_control_time_takes_an_optional_unit(tmp_path):
    def control(time):
        path = _edit(tmp_path, "Net1.inp",
                     ("[CONTROLS]", f"[CONTROLS]\n LINK 9 OPEN AT TIME {time}"))
        return read_epanet_inp(path).controls[0]

    assert control("90 MIN").value == 5400.0
    assert control("2 HOURS").value == 7200.0
    assert control("2").value == 7200.0
    assert control("1:30").value == 5400.0
    with pytest.raises(ValueError, match="unknown time unit"):
        control("90 FURLONGS")


def test_a_rule_system_time_takes_an_optional_unit_and_grade_is_head(tmp_path):
    rules = ("[RULES]\nRULE R1\nIF SYSTEM TIME > 90 MIN\nAND NODE J1 GRADE < 30\n"
             "THEN LINK P3 STATUS IS OPEN\n\n[CURVES]")
    net = read_epanet_inp(_edit(tmp_path, "twoloop_si.inp", ("[CURVES]", rules)))
    time, grade = net.rules[0].premises
    assert time.value == 5400.0
    assert (grade.attribute, grade.value) == ("HEAD", 30.0)
    bad = rules.replace("90 MIN", "90 FURLONGS")
    with pytest.raises(ValueError, match="unknown time unit"):
        read_epanet_inp(_edit(tmp_path, "twoloop_si.inp", ("[CURVES]", bad)))


def test_status_active_on_a_pump_or_pipe_control_is_refused_by_check(tmp_path):
    path = _edit(tmp_path, "Net1.inp", ("[CONTROLS]", "[CONTROLS]\n LINK 9 ACTIVE AT TIME 2"))
    with pytest.raises(ValueError, match="ACTIVE"):
        read_epanet_inp(path)


def test_an_energy_pattern_that_does_not_exist_is_refused(tmp_path):
    path = _edit(tmp_path, "Net1.inp",
                 ("[END]", "[ENERGY]\n Global Pattern NOPE\n\n[END]"))
    with pytest.raises(ValueError, match="NOPE"):
        read_epanet_inp(path)


def test_the_rule_timestep_follows_the_hydraulic_step_and_is_capped(tmp_path):
    net = _times(tmp_path, "Hydraulic Timestep 1")
    assert net.rule_timestep == 360.0
    longer = dataclasses.replace(net, hydraulic_timestep=7200.0)
    assert longer.rule_timestep == 720.0  # not stale after `replace`
    assert _times(tmp_path, "Hydraulic Timestep 1\nRule Timestep 2").rule_timestep == 3600.0
    assert _times(tmp_path, "Hydraulic Timestep 1\nRule Timestep 0:05").rule_timestep == 300.0


def test_unrecognised_times_are_recorded_and_the_tank_minimum_volume_is_read(tmp_path):
    net = _times(tmp_path, "Quality Timestep 0:05\nStatistic NONE")
    assert net.notes["unrecognised_times"] == "Quality Timestep 0:05; Statistic NONE"
    assert "unrecognised_times" not in _times(tmp_path, "Hydraulic Timestep 1").notes
    path = _edit(
        tmp_path, "Net1.inp",
        (" 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \t                \t;",
         " 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t5000        \t                \t;"),
    )
    assert read_epanet_inp(path).tanks[0].min_volume == pytest.approx(5000 * FOOT**3, rel=1e-15)
    assert read_epanet_inp(DATA / "twoloop_si.inp").tanks == ()

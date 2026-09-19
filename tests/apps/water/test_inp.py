"""The documented EPANET `.inp` subset (spec 13.5)."""

from pathlib import Path

import pytest

from noodl.apps.water.inp import FLOW_UNITS, read_epanet_inp
from noodl.apps.water.network import twoloop

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
    assert [(c.link, c.status, c.node, c.test) for c in net.controls] == [
        ("9", "OPEN", "2", "BELOW"), ("9", "CLOSED", "2", "ABOVE")
    ]
    assert [c.level for c in net.controls] == pytest.approx(
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


def test_chezy_manning_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp", (" Headloss           \tH-W",
                                              " Headloss           \tC-M"))
    with pytest.raises(ValueError, match="only H-W and D-W"):
        read_epanet_inp(path)


def test_a_rules_section_with_content_is_refused(tmp_path):
    rule = "[RULES]\nRULE 1\nIF TANK T1 LEVEL BELOW 5\nTHEN LINK P1 STATUS IS OPEN"
    path = _edit(tmp_path, "twoloop_si.inp", ("[RULES]", rule))
    with pytest.raises(ValueError, match=r"\[RULES\]"):
        read_epanet_inp(path)


def test_an_emitters_section_with_content_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[EMITTERS]\n;Junction        \tCoefficient",
                  "[EMITTERS]\n;Junction        \tCoefficient\n J1              \t0.5"))
    with pytest.raises(ValueError, match=r"\[EMITTERS\]"):
        read_epanet_inp(path)


def test_a_status_section_with_content_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[STATUS]\n;ID              \tStatus/Setting",
                  "[STATUS]\n;ID              \tStatus/Setting\n P1              \tClosed"))
    with pytest.raises(ValueError, match=r"\[STATUS\]"):
        read_epanet_inp(path)


def test_an_unknown_section_with_content_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[END]", "[LEAKAGE]\n J1   0.001\n\n[END]"))
    with pytest.raises(ValueError, match=r"\[LEAKAGE\]"):
        read_epanet_inp(path)


def test_a_time_based_control_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 ("[CONTROLS]", "[CONTROLS]\n LINK P1 CLOSED AT CLOCKTIME 6 AM"))
    with pytest.raises(ValueError, match="time-based control"):
        read_epanet_inp(path)


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


def test_a_constant_power_pump_is_refused(tmp_path):
    path = _edit(tmp_path, "Net1.inp", (" 9               \t9               \t10"
                                        "              \tHEAD 1\t;",
                                        " 9               \t9               \t10"
                                        "              \tPOWER 50\t;"))
    with pytest.raises(ValueError, match="constant-POWER pump"):
        read_epanet_inp(path)


def test_a_speed_patterned_pump_is_refused(tmp_path):
    path = _edit(tmp_path, "Net1.inp", ("\tHEAD 1\t;", "\tHEAD 1 PATTERN 1\t;"))
    with pytest.raises(ValueError, match="SPEED PATTERN"):
        read_epanet_inp(path)


def test_an_unknown_pump_keyword_is_refused(tmp_path):
    path = _edit(tmp_path, "Net1.inp", ("\tHEAD 1\t;", "\tHEAD 1 TORQUE 3\t;"))
    with pytest.raises(ValueError, match="unknown keyword 'TORQUE'"):
        read_epanet_inp(path)


def test_a_tank_with_a_volume_curve_is_refused(tmp_path):
    path = _edit(
        tmp_path, "Net1.inp",
        (" 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \t                \t;",
         " 2               \t850         \t120         \t100         \t150         "
         "\t50.5        \t0           \tVC1             \t;"),
    )
    with pytest.raises(ValueError, match="volume curve"):
        read_epanet_inp(path)


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


# --------------------------------------------------------------------- [OPTIONS] (N5)
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
    from noodl.apps.water.network import build_water_model

    model, _, _ = build_water_model(net)
    assert len(model.potential["water"]._node_sources) == 1


def test_an_unknown_demand_model_is_refused(tmp_path):
    path = _edit(tmp_path, "twoloop_si.inp",
                 (" Pattern            \t1", " Pattern            \t1\n"
                  " DEMAND MODEL       \tPDD"))
    with pytest.raises(ValueError, match="DEMAND MODEL 'PDD'"):
        read_epanet_inp(path)


def test_an_unrecognised_option_is_recorded_not_dropped():
    """`Trials`, `Accuracy`, `CHECKFREQ`, ... are solver/report cosmetics this reader does
    not model, but N5 requires they be RECORDED rather than silently dropped."""
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

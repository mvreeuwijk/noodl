"""CONTAM .prj reader on NIST's sample projects (the documented subset)."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import torch

from noodl.apps.building_physics.air_handling import AirHandlerFlows
from noodl.apps.building_physics.filters import (
    ConstantEfficiencyFilter,
    FilterSeries,
    FilterTransmission,
    ParticleFilter,
    UVGIFilter,
)
from noodl.apps.building_physics.occupants import OccupantGeneration
from noodl.apps.building_physics.prj import (
    _PATH_FIELDS,
    ContamCompatibilityWarning,
    PrescribedTemperatureDensity,
    Project,
    RunControl,
    ZonePressureDensity,
    project_to_model,
    read_prj,
)
from noodl.apps.building_physics.prj import steady as prj_steady
from noodl.apps.building_physics.schedules import Calendar, WeekSchedule
from noodl.apps.building_physics.sources import (
    ConstantSource,
    CutoffSource,
    DepositionRateSink,
    DepositionVelocitySink,
    PeakEmissionSource,
    PowerLawDecaySource,
    PressureDrivenSource,
    SpeciesSources,
    sources_from_project,
)
from noodl.apps.building_physics.species import KineticReaction, Species
from noodl.apps.building_physics.thermal import R_AIR
from noodl.elements import (
    AffinityFan,
    AirDuct,
    CubicSplineFlow,
    Damper,
    DuctTerminal,
    FixedFlow,
    FixedVolumeFlow,
    PrescribedFlow,
    Scaled,
    UpstreamDensityPowerLaw,
)

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
MIXED = DATA / "doorway_damper_fan.prj"
# NIST's own one-zone stack demo with a constant-efficiency filter on path 1.
SS_STACK = DATA / "test_OneZoneSsStack-UseApi.prj"
FILTERS = DATA / "filters_three_zone.prj"
SINKS = DATA / "sinks_three_zone.prj"
PRESSURE = DATA / "pressure_source_stack.prj"
F64 = torch.float64


def test_reads_the_three_zone_project_structure():
    p = read_prj(THREE)
    assert isinstance(p, Project)
    assert p.zones == ["one", "two", "three"]
    assert p.zone_volumes.tolist() == [300.0, 600.0, 900.0]
    assert p.T_zone.tolist() == pytest.approx([293.15] * 3)
    assert p.species == ["sarin"]
    assert [path.nr for path in p.paths] == [1, 2, 3, 4]
    assert p.ambient_conditions == pytest.approx(
        {"Ta": 293.15, "Pb": 101325.0, "Ws": 5.23, "Wd": 270.0}
    )
    assert p.g == pytest.approx(9.8055)
    assert sorted(p.profiles) == [1, 2, 3]
    assert p.profiles[2].angles.numel() == 13          # 12 knots + the appended 360


def test_network_edges_follow_the_paths_with_ambient_as_minus_one():
    p = read_prj(THREE)
    net = p.net
    assert net.nodes == ["ambient", "one", "two", "three"]
    assert [(u, v) for u, v, _ in net.edges] == [
        ("ambient", "one"), ("one", "two"), ("three", "two"), ("ambient", "three")
    ]
    assert set(net.edge_kinds()) == {"pl_3"}          # every path uses element 3
    assert net.edge_attr("z_path", "pl_3").tolist() == pytest.approx([1.5] * 4)
    assert net.edge_attr("Ch", "pl_3", default=0.0).tolist() == pytest.approx(
        [0.1225, 0.0, 0.0, 0.1225]
    )
    assert net.edge_attr("azimuth", "pl_3", default=0.0).tolist() == pytest.approx(
        [270.0, 0.0, 0.0, 90.0]
    )
    assert net.edge_attr("profile", "pl_3", default=0.0).tolist() == [3.0, 0.0, 0.0, 3.0]
    assert net.node_attr("volume", default=0.0).tolist() == [0.0, 300.0, 600.0, 900.0]


def test_element_coefficients_follow_contam_conventions():
    p = read_prj(THREE)
    (el,) = p.elements
    # `plr_orfc` is the sqrt(rho) CONTAM family: the reader must build the UPSTREAM-density
    # correction, not a plain `PowerLaw` (which `UpstreamDensityPowerLaw` satisfies too, so
    # `isinstance(el, PowerLaw)` alone would not catch a revert to the uncorrected element).
    assert isinstance(el, UpstreamDensityPowerLaw) and el.kind == "pl_3"
    turb_mass = 0.141421 * math.sqrt(1.2041)          # orifice: turb * sqrt(rho)
    torch.testing.assert_close(el.C, torch.full((4,), turb_mass, dtype=F64))
    torch.testing.assert_close(el.n, torch.full((4,), 0.5, dtype=F64))
    torch.testing.assert_close(el.m, torch.full((4,), 0.5, dtype=F64))
    src, tgt = p.net.endpoints("pl_3")
    torch.testing.assert_close(el.src, src)
    torch.testing.assert_close(el.tgt, tgt)
    lam, mu, rho = 0.00237882, 1.81625e-5, 1.2041
    assert el.dp_transition == pytest.approx((turb_mass * mu / (lam * rho)) ** (1 / 0.5))


def test_a_mass_flow_family_kind_gets_exponent_zero(tmp_path):
    """`plr_fcn` (and `plr_test1/2`, `plr_conn`, `plr_stair`, `plr_shaft`) is CONTAM's
    mass-flow family: C is already a mass-flow coefficient, so the upstream-density
    correction must apply exponent m = 0 -- i.e. no correction at all -- rather than the
    sqrt(rho) family's 1/2."""
    text = THREE.read_text()
    variant = _variant(text, "3 23 plr_orfc orfcPt01", "3 23 plr_fcn orfcPt01", tmp_path)
    p = read_prj(variant)
    (el,) = p.elements
    assert isinstance(el, UpstreamDensityPowerLaw)
    torch.testing.assert_close(el.m, torch.full((4,), 0.0, dtype=F64))


def test_a_volumetric_family_kind_gets_exponent_one(tmp_path):
    """`plr_qcn` is CONTAM's volumetric family (a volumetric rating turned into mass flow via
    C = mult * rho * turb), so the upstream-density correction must apply exponent m = 1."""
    text = THREE.read_text()
    variant = _variant(text, "3 23 plr_orfc orfcPt01", "3 23 plr_qcn orfcPt01", tmp_path)
    p = read_prj(variant)
    (el,) = p.elements
    assert isinstance(el, UpstreamDensityPowerLaw)
    torch.testing.assert_close(el.m, torch.full((4,), 1.0, dtype=F64))


def test_the_reader_builds_a_float64_application():
    """The default dtype is float32; every file value must arrive as float64."""
    p = read_prj(THREE)
    assert p.net.dtype is F64
    assert p.T_zone.dtype is F64 and p.zone_volumes.dtype is F64 and p.x0.dtype is F64
    (el,) = p.elements
    assert el.C.dtype is F64 and el.n.dtype is F64
    assert p.profiles[2].angles.dtype is F64


def test_project_to_model_solves_the_wind_driven_case_and_conserves_species():
    p = read_prj(THREE)
    model, state, drivers = project_to_model(p)
    assert set(model.layers) == {"air", "species"}
    assert drivers["V_met"].item() == pytest.approx(5.23)
    assert drivers["theta_w"].item() == pytest.approx(270.0)
    assert drivers["rho"].shape == (4,)
    ss = model.steady(state, drivers)
    flows = p.path_flows(ss["air.q"])
    assert flows.shape == (4,)
    assert flows[0] > 0 and flows[3] < 0                # west wind: in at 270, out at 90
    # This Newton solve lands the nodal balance at exactly 0.0, so the assertion is tight
    # rather than a slack 1e-9.
    assert model.residuals(ss, drivers)["air"].abs().max().item() < 1e-14
    drivers["species.x_boundary"] = torch.tensor([[0.0023254]], dtype=F64)
    ss2 = model.steady(state, drivers)
    torch.testing.assert_close(ss2["species.x"], torch.full((3, 1), 0.0023254, dtype=F64))


def test_reads_the_hand_written_doorway_damper_fan_project():
    p = read_prj(MIXED)
    kinds = {el.kind: el for el in p.elements}
    # Two DIFFERENT power-law kinds from this one file: a single-edge orifice (`pl_2`, from
    # `plr_orfc`) and a two-edge doorway (`door_3`, from `dor_pl2`). `isinstance(el, PowerLaw)`
    # alone would not catch a revert to the uncorrected element, since
    # `UpstreamDensityPowerLaw` is itself a `PowerLaw`.
    assert isinstance(kinds["pl_2"], UpstreamDensityPowerLaw)
    torch.testing.assert_close(kinds["pl_2"].m, torch.full((2,), 0.5, dtype=F64))
    src2, tgt2 = p.net.endpoints("pl_2")
    torch.testing.assert_close(kinds["pl_2"].src, src2)
    torch.testing.assert_close(kinds["pl_2"].tgt, tgt2)

    assert isinstance(kinds["door_3"], UpstreamDensityPowerLaw)
    assert p.net.edge_index("door_3").numel() == 2      # two-opening doorway -> two edges
    torch.testing.assert_close(kinds["door_3"].m, torch.full((2,), 0.5, dtype=F64))
    src3, tgt3 = p.net.endpoints("door_3")
    torch.testing.assert_close(kinds["door_3"].src, src3)
    torch.testing.assert_close(kinds["door_3"].tgt, tgt3)
    z = p.net.edge_attr("z_path", "door_3")
    assert (z[1] - z[0]).item() == pytest.approx(2 * 0.4444)
    assert isinstance(kinds["bd_4"], Damper)
    assert kinds["bd_4"].pos.C.item() == pytest.approx(0.02)
    assert kinds["bd_4"].neg.n.item() == pytest.approx(0.65)
    fans = [el for el in p.elements if isinstance(el, FixedFlow)]
    assert len(fans) <= 1


def test_a_doorway_reads_ht_wd_and_cd_from_fields_4_5_and_6():
    """Doorway field positions, verified against NIST's own numbers.

    `doorway_damper_fan.prj` element 5 is `reg_solverContTrace-mz-MH-trans-3day.prj`'s
    `dor_door` record copied verbatim: ` 0.148966 2.54558 0.5 0.01 2 0.9 1 0 0 0`, i.e.
    `lam turb expt dTmin ht wd cd`. Three independent assertions pin all three fields --
    the sum of C alone cannot, because CONTAM's identity turb = cd A sqrt(2) is symmetric
    under swapping `wd` and `cd`.
    """
    p = read_prj(MIXED)
    (door,) = [el for el in p.elements if el.kind == "door_5"]
    ht, wd, cd, turb, rho = 2.0, 0.9, 1.0, 2.54558, 1.2041

    # field 4 (ht): dor_door derives its half-separation as 2 ht / 9, nothing else.
    z = p.net.edge_attr("z_path", "door_5")
    assert (z[1] - z[0]).item() == pytest.approx(2 * (2 * ht / 9))
    # field 6 (cd) on its own, and field 5 (wd) through the half-opening area wd ht / 2:
    # swapping the two would give Cd 0.9 and area 1.0 instead.
    assert p.net.edge_attr("Cd", "door_5").tolist() == pytest.approx([cd, cd])
    assert p.net.edge_attr("area", "door_5").tolist() == pytest.approx([wd * ht / 2] * 2)
    # And the whole conversion against the record's OWN turbulent coefficient: the two
    # openings together must be turb sqrt(rho) (the file rounds turb to six figures).
    assert door.C.sum().item() == pytest.approx(turb * math.sqrt(rho), rel=1e-5)


def test_reads_a_multi_species_project():
    """A `N ! contaminants:` section lists its N indices on ONE line, not one per line.

    With a per-line loop the reader ate the `3 ! species:` header and then its comment
    line, and five NIST demo projects were unreadable with an error blaming the wrong
    record. `doorway_damper_fan.prj` carries three species so both failure modes are
    pinned (>= 2 eats the header, >= 3 eats the comment too).
    """
    p = read_prj(MIXED)
    assert p.species == ["sarin", "CO", "tracer"]
    assert p.x0.shape == (1, 3)
    torch.testing.assert_close(p.x0, torch.tensor([[0.0, 1e-4, 0.0]], dtype=F64))
    # And `initial zone concentrations` heads its ONE zone row with 3 -- that section's
    # count is zones x species, not rows (NIST heads two rows with 6 in the three-species
    # `test_OneFloorWpcAddMf.prj`). A reader taking it as a row count reads past the row.


def test_the_round_trip_keeps_levels_zone_numbers_and_sources():
    """`sources_from_project` resolves a source's zone through `zone_nr_to_name`, not by
    position."""
    p = read_prj(MIXED)
    assert p.zones == ["singleZone"]
    assert p.zone_nr_to_name == {1: "singleZone"}
    assert p.net.node_attr("z_ref", default=0.0).tolist() == [0.0, 0.0]   # level 1 refHt 0
    # relHt 0, 3, 1, 2, 2.5, 1.5 on a level whose refHt is 0 -> z_path = refHt + relHt,
    # the two doorways splitting theirs by +/- dH and +/- 2 ht / 9.
    assert sorted(p.net.edge_attr("z_path").tolist()) == pytest.approx(
        [0.0, 0.5556, 1.5 - 4 / 9, 1.4444, 1.5 + 4 / 9, 2.0, 2.5, 3.0]
    )
    (src,) = p.sources
    assert (src.nr, src.zone_nr, src.element_nr) == (1, 1, 1)
    assert src.source_type == "ccf"
    assert src.params == pytest.approx([0.0001, 0.0, 5.0, 0.0])
    assert src.mult == pytest.approx(1.0)
    # Every duct section of this project is PRESENT and empty; a project like that loads.
    assert "duct" not in {el.kind for el in p.elements}


def test_path_flows_index_q_in_the_layers_kind_block_layout():
    """`q` is laid out in element-KIND blocks, not in path order.

    `doorway_damper_fan.prj` has five kinds. Network edge order is path order --
    pl_2, pl_2, door_3, door_3, bd_4, fan_1, door_5, door_5 -- but `PotentialFlowLayer`
    concatenates per element, in `project.elements` order, so the two layouts differ.
    """
    p = read_prj(MIXED)
    offsets, offset = {}, 0
    for el in p.elements:
        offsets[el.kind] = offset
        offset += p.net.edge_index(el.kind).numel()
    assert offset == p.net.b == 8
    expected = {
        1: [offsets["pl_2"] + 0],
        2: [offsets["pl_2"] + 1],
        3: [offsets["door_3"] + 0, offsets["door_3"] + 1],
        4: [offsets["bd_4"] + 0],
        5: [offsets["fan_1"] + 0],
        6: [offsets["door_5"] + 0, offsets["door_5"] + 1],
    }
    assert {path.nr: path.edge_columns for path in p.paths} == expected
    # A naive running counter in PATH order would give the network's own edge order.
    naive = [[0], [1], [2, 3], [4], [5], [6, 7]]
    assert [path.edge_columns for path in p.paths] != naive

    q = torch.arange(p.net.b, dtype=F64)
    flows = p.path_flows(q)
    assert flows.tolist() == [float(sum(expected[nr])) for nr in (1, 2, 3, 4, 5, 6)]
    # And batched, so the ContamX comparison can compare a whole time series at once.
    batched = p.path_flows(q.expand(7, p.net.b))
    assert batched.shape == (7, 6)
    torch.testing.assert_close(batched[0], flows)


def _variant(text: str, old: str, new: str, tmp_path: Path) -> Path:
    assert old in text, old
    out = tmp_path / "variant.prj"
    out.write_text(text.replace(old, new, 1))
    return out


def test_ambient_conditions_come_from_the_steady_simulation_line(tmp_path):
    """The label picks the block, not its position among the `! Ta` blocks."""
    text = THREE.read_text()
    steady = "293.150 101325.0  5.230 270.0 0.000 1 2 0 0 1 ! steady simulation"
    windy = "293.150 101325.0 14.793 270.0 0.000 1 2 0 0 1 ! wind pressure test"
    # In the file the steady block comes FIRST; swapped, it must still be the one read.
    swapped = _variant(text, f"{steady}\n{windy}", f"{windy}\n{steady}", tmp_path)
    assert read_prj(swapped).ambient_conditions["Ws"] == pytest.approx(5.23)
    with pytest.raises(ValueError, match=r"steady simulation"):
        read_prj(_variant(text, "! steady simulation", "! design day", tmp_path))
    with pytest.raises(ValueError, match=r"dens"):
        read_prj(_variant(text, "!dens   grav", "!rho   grav", tmp_path))


def test_a_path_naming_an_absent_wind_profile_raises_naming_edge_and_number(tmp_path):
    """Profiles are keyed by CONTAM's profile NUMBER and the lookup is strict."""
    text = THREE.read_text()
    path1 = "   1    1  -1   1   3   0   3   0"
    with pytest.raises(KeyError, match=r"profile number 7"):
        read_prj(_variant(text, path1, "   1    1  -1   1   3   0   7   0", tmp_path))


def test_wind_profiles_keep_their_curve_fit_and_a_trigonometric_one_is_refused(tmp_path):
    """The profile record's type field: 1 linear, 2 cubic spline (NIST's samples), 3
    trigonometric (Walker and Wilson), which is refused naming the profile."""
    p = read_prj(THREE)
    assert [p.profiles[n].curve for n in (1, 2, 3)] == ["linear", "spline", "spline"]
    text = THREE.read_text()
    with pytest.raises(ValueError, match=r"wind pressure profile 3 .*trigonometric"):
        read_prj(_variant(text, "3 13 2 S&C_2_Short", "3 13 3 S&C_2_Short", tmp_path))
    with pytest.raises(ValueError, match=r"wind pressure profile 2 .*type 7"):
        read_prj(_variant(text, "2 13 2 S&C_2_Long", "2 13 7 S&C_2_Long", tmp_path))


def test_unsupported_records_raise_naming_the_offender(tmp_path):
    text = THREE.read_text()
    zone1 = "   1  3   0   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 0 0 0"
    with pytest.raises(ValueError, match=r"zone 1.*1-D"):
        read_prj(
            _variant(
                text,
                zone1,
                "   1  3   0   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 1 0 0",
                tmp_path,
            )
        )
    with pytest.raises(ValueError, match=r"duct"):
        read_prj(_variant(text, "0 ! duct segments:", "1 ! duct segments:", tmp_path))
    with pytest.raises(ValueError, match=r"element 3.*srv_jwa"):
        read_prj(_variant(text, "3 23 plr_orfc orfcPt01", "3 23 srv_jwa orfcPt01", tmp_path))
    with pytest.raises(ValueError, match=r"ContamW"):
        read_prj(_variant(text, "ContamW 3.4.0.4 0", "SomethingElse 1.0 0", tmp_path))


def test_the_refusal_messages_name_the_section_and_the_record(tmp_path):
    text = THREE.read_text()
    zone1 = "   1  3   0   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 0 0 0"
    with pytest.raises(KeyError) as exc:
        read_prj(
            _variant(
                text,
                zone1,
                "   1  3   0   0   2   1   0.000   300 293.15 0 one -1 0 2 0 0 0 0 0",
                tmp_path,
            )
        )
    message = str(exc.value)
    assert "zone 1" in message and "kinetic reaction" in message and "2" in message
    with pytest.raises(ValueError) as exc:
        read_prj(_variant(text, "0 ! duct segments:", "3 ! duct segments:", tmp_path))
    assert "duct segments" in str(exc.value) and "3" in str(exc.value)


def test_control_nodes_that_no_path_or_zone_references_load():
    """NIST's samples carry 33 sensor/logger control nodes (integrations among them); they
    are read, and none is built, as nothing uses them."""
    p = read_prj(THREE)
    assert len(p.paths) == 4 and p.zones == ["one", "two", "three"]
    assert len(p.control_nodes) == 33 and p.control_plan is None


def test_project_to_model_takes_an_ambient_override_and_can_drop_the_species_layer():
    p = read_prj(THREE)
    model, state, drivers = project_to_model(
        p, ambient={"Ta": 283.15, "Ws": 2.0, "Wd": 90.0, "rh": 0.5}, species=False
    )
    assert set(model.layers) == {"air"}
    assert state == {} and "species.x_boundary" not in drivers
    assert drivers["V_met"].item() == pytest.approx(2.0)
    assert drivers["theta_w"].item() == pytest.approx(90.0)
    # Ta moves the AMBIENT node's density; the zones keep the project's own T0.
    assert drivers["rho"][0].item() == pytest.approx(101325.0 / (287.055 * 283.15))
    assert drivers["rho"][1].item() == pytest.approx(101325.0 / (287.055 * 293.15))
    assert p.ambient_conditions["Ws"] == pytest.approx(5.23)      # the project is untouched
    flows = p.path_flows(model.steady(state, drivers)["air.q"])
    assert flows[0] < 0 and flows[3] > 0          # east wind: in at 90, out at 270


def test_a_truncated_file_raises_rather_than_returning_half_a_project(tmp_path):
    text = THREE.read_text()
    out = tmp_path / "truncated.prj"
    out.write_text(text.split("   2    0   1   2   3")[0])
    with pytest.raises(ValueError, match=r"ended in the middle of a record"):
        read_prj(out)


def _set_path_field(text: str, path_nr: int, index: int, value: str) -> tuple[str, str]:
    """(old line, rewritten line) with field `index` of path `path_nr` set to `value`.

    Rewriting a field in place keeps the record at its 30 fields, so the length check
    cannot stand in for the flag check the test is actually making.
    """
    for line in text.splitlines():
        tok = line.split()
        if len(tok) == _PATH_FIELDS and tok[0] == str(path_nr):
            new = list(tok)
            new[index] = value
            return line, " ".join(new)
    raise AssertionError(f"no {_PATH_FIELDS}-field path record numbered {path_nr}")


def test_the_path_record_carries_cdvf_and_cfd_at_fields_28_and_29(tmp_path):
    """A path line has 30 fields: an unnamed `clr` sits between `dir` and `u[4]`.

    ContamW's own header comment omits it, which suggests positions 27/28 instead.
    Every shipped fixture reads 0 at 27, 28 AND 29, so without this test a regression to
    27/28 leaves the whole file green -- while refusing most NIST projects, whose field 27
    (`u[4]`'s last unit code) takes values 0, 1, 3 and 4.
    """
    text = THREE.read_text()
    old, new = _set_path_field(text, 2, 28, "1")
    with pytest.raises(ValueError, match=r"path 2.*values file \(cdvf 1\)"):
        read_prj(_variant(text, old, new, tmp_path))
    old, new = _set_path_field(text, 2, 29, "1")
    with pytest.raises(ValueError, match=r"path 2.*CFD \(cfd 1\)"):
        read_prj(_variant(text, old, new, tmp_path))
    # Field 27 is a display-unit code, NOT cdvf: a nonzero there must load.
    old, new = _set_path_field(text, 2, 27, "4")
    assert [path.nr for path in read_prj(_variant(text, old, new, tmp_path)).paths] == [
        1, 2, 3, 4
    ]


def test_a_zone_reading_a_continuous_values_file_is_refused(tmp_path):
    """`cdvf` at zone field 17, between `axs` (16) and `cfd` (18), both already pinned."""
    text = THREE.read_text()
    zone1 = "   1  3   0   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 0 0 0"
    with pytest.raises(ValueError, match=r"zone 1.*continuous values file"):
        read_prj(
            _variant(
                text,
                zone1,
                "   1  3   0   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 0 1 0",
                tmp_path,
            )
        )


# --------------------------------------------------------------------------------------
# Density options (`!tsdens relax tsmaxi cnvgSS densZP stackD dodMdt`) and zone air mass.

WTH_STACK = DATA / "test_OneZoneWthCtmStack-UseApi.prj"
_TSDENS_WTH_STACK = "   1    0.75    20     1      1      0      1"


def test_dens_zp_is_read_and_stack_d_is_refused(tmp_path):
    assert read_prj(WTH_STACK).density_uses_zone_pressure is True
    assert read_prj(THREE).density_uses_zone_pressure is False
    model, _s, _d = project_to_model(read_prj(WTH_STACK))
    assert any(isinstance(c, ZonePressureDensity) for c in model.closures)
    model, _s, _d = project_to_model(read_prj(THREE))
    assert [type(c) for c in model.closures] == [PrescribedTemperatureDensity]
    text = WTH_STACK.read_text()
    with pytest.raises(ValueError, match=r"stackD = 1.*unsupported"):
        read_prj(_variant(text, _TSDENS_WTH_STACK, "   1    0.75    20     1      1      1      1",
                          tmp_path))
    with pytest.raises(ValueError, match=r"densZP stackD dodMdt.*4 fields"):
        read_prj(_variant(text, _TSDENS_WTH_STACK, "   1    0.75    20     1", tmp_path))


def test_zone_pressure_density_is_the_absolute_pressure_ideal_gas():
    from noodl.apps.building_physics.prj import ZonePressureDensity

    zp = ZonePressureDensity(101325.0)
    rho = torch.tensor([1.29, 1.20], dtype=F64)
    assert zp({}, {"rho": rho}) == {}                     # no pressures yet: no correction
    phi = torch.tensor([0.0, -0.65], dtype=F64)
    out = zp({"air.phi": phi}, {"rho": rho})["rho"]
    torch.testing.assert_close(out, rho * (101325.0 + phi) / 101325.0, rtol=1e-15, atol=0)


def test_prj_steady_converges_the_zone_pressure_density_to_its_fixed_point():
    """The converged state must be self-consistent: its densities are those of its OWN
    pressures, so one more pass changes nothing. A single `model.steady` is not (it solves
    at `Pb` alone), and differs by the |P_zone| / Pb term amplified through the stack.
    """
    from noodl.apps.building_physics.prj import steady

    p = read_prj(WTH_STACK)
    model, state, drivers = project_to_model(p, ambient=dict(p.ambient_conditions, Ta=273.15))
    one = model.steady(state, drivers, atol=1e-14)
    ss = steady(model, state, drivers, atol=1e-14)
    again = model.steady({**state, "air.phi": ss["air.phi"]}, drivers, atol=1e-14)
    torch.testing.assert_close(again["air.q"], ss["air.q"], rtol=1e-12, atol=0)
    rel = ((one["air.q"] - ss["air.q"]) / ss["air.q"]).abs().max().item()
    assert 3e-5 < rel < 6e-5
    with pytest.raises(RuntimeError, match=r"did not converge in 1 passes"):
        steady(model, state, drivers, rtol=0.0, max_iter=1, atol=1e-14)


def test_prj_steady_is_one_model_steady_without_dens_zp():
    from noodl.apps.building_physics.prj import steady

    model, state, drivers = project_to_model(read_prj(THREE))
    torch.testing.assert_close(
        steady(model, state, drivers)["air.q"], model.steady(state, drivers)["air.q"],
        rtol=0, atol=0,
    )


def test_a_steady_contaminant_solve_with_no_airflow_is_refused_with_the_weather():
    """At its own steady instant the one-zone stack project has Ta equal to the zone
    temperature and no wind: no air moves, so the zone keeps whatever enters it and the
    steady contaminant state is undefined. Refused naming the zone and the weather; a
    colder ambient drives a stack flow and the same project solves."""
    from noodl.layers import UndefinedSteadyState

    p = read_prj(DATA / "test_OneZoneWthCtmStack.prj")
    model, state, drivers = project_to_model(p)
    with pytest.raises(UndefinedSteadyState, match=r"\['singleZone'\]") as err:
        prj_steady(model, state, drivers)
    message = str(err.value)
    assert "Ta = 293.15 K" in message and "Ws = 0 m/s" in message
    assert err.value.layer == "species" and err.value.nodes == ["singleZone"]

    model, state, drivers = project_to_model(p, ambient=dict(p.ambient_conditions, Ta=273.15))
    out = prj_steady(model, state, drivers)
    assert bool(torch.isfinite(out["species.x"]).all())


def test_local_mean_age_of_a_zone_without_airflow_is_refused_by_name():
    from noodl.apps.building_physics.outputs import local_mean_age
    from noodl.layers import UndefinedSteadyState

    p = read_prj(DATA / "test_OneZoneWthCtmStack.prj")
    air, state, drivers = project_to_model(p, species=False)
    q = air.steady(state, drivers)["air.q"]
    assert bool((q == 0).all())
    model, _, _ = project_to_model(p)
    with pytest.raises(UndefinedSteadyState, match=r"\['singleZone'\]"):
        local_mean_age({"air.q": q}, model, capacity=model.transport["species"].capacity)


def test_species_capacity_is_the_zone_air_mass_at_its_own_density(tmp_path):
    """ContamX's zone air mass is Pb / (R T) V, not RHO_0 V: 1.2041 is that density only to
    five figures at 293.15 K, and 3.4 % off per 10 K away from it."""
    from noodl.apps.building_physics.thermal import R_AIR

    text = THREE.read_text()
    warm = _variant(text, "   2  3   0   0   0   1   0.000   600 293.15 0 two",
                    "   2  3   0   0   0   1   0.000   600 303.15 0 two", tmp_path)
    p = read_prj(warm)
    model, _s, _d = project_to_model(p, ambient=dict(p.ambient_conditions, Pb=100000.0))
    expected = 100000.0 / (R_AIR * torch.tensor([293.15, 303.15, 293.15], dtype=F64)) * (
        torch.tensor([300.0, 600.0, 900.0], dtype=F64)
    )
    torch.testing.assert_close(model.layers["species"].capacity, expected, rtol=1e-14, atol=0)


def test_densities_and_zone_air_masses_follow_the_prescribed_temperatures():
    model, state, drivers = project_to_model(read_prj(THREE))
    dens = model.closures[0]
    assert isinstance(dens, PrescribedTemperatureDensity)
    assert drivers["T"].shape == (4,) and drivers["P_ref"].item() == 101325.0
    T = drivers["T"].clone()
    two = model.net.node_index("two")
    T[two] = 303.15
    out = dens(state, {**drivers, "T": T})
    torch.testing.assert_close(out["rho"], 101325.0 / (R_AIR * T), rtol=1e-15, atol=0)
    assert out["rho_amb"].item() == pytest.approx(101325.0 / (R_AIR * 293.15), rel=1e-15)
    layer = model.transport["species"]
    expected = (101325.0 / (R_AIR * T) * model.net.node_attr("volume"))[layer.interior_idx]
    torch.testing.assert_close(out["species.capacity"], expected, rtol=1e-15, atol=0)


def test_the_initial_state_carries_the_zone_air_masses():
    model, state, drivers = project_to_model(read_prj(THREE))
    assert state["species.capacity"].shape == (3,)
    torch.testing.assert_close(
        state["species.capacity"], model.closures[0](state, drivers)["species.capacity"])


def test_a_batched_barometric_pressure_gives_batched_densities():
    model, state, drivers = project_to_model(read_prj(THREE))
    P = torch.tensor([101325.0, 90000.0], dtype=F64)
    out = model.closures[0](state, {**drivers, "P_ref": P})
    assert out["rho"].shape == (2, 4) and out["species.capacity"].shape == (2, 3)
    assert out["rho"][1, 1].item() == pytest.approx(90000.0 / (R_AIR * 293.15), rel=1e-15)


def test_zone_pressure_density_rescales_the_zone_air_mass_too():
    zp = ZonePressureDensity(101325.0, capacity_key="species.capacity",
                             capacity_nodes=torch.tensor([1], dtype=torch.long))
    phi = torch.tensor([0.0, 10.0], dtype=F64)
    rho = torch.tensor([1.2, 1.1], dtype=F64)
    cap = torch.tensor([660.0], dtype=F64)
    out = zp({"air.phi": phi},
             {"rho": rho, "species.capacity": cap, "P_ref": torch.tensor(101325.0, dtype=F64)})
    assert out["species.capacity"].item() == pytest.approx(660.0 * (1 + 10.0 / 101325.0), rel=1e-15)
    assert out["rho"][1].item() == pytest.approx(1.1 * (1 + 10.0 / 101325.0), rel=1e-15)


def test_zone_pressure_density_refuses_a_missing_zone_air_mass():
    """A configured `capacity_key` absent from the drivers is an error, not a skipped
    rescale (the species layer would otherwise see the zone air mass at Pb alone)."""
    zp = ZonePressureDensity(101325.0, capacity_key="species.capacity",
                             capacity_nodes=torch.tensor([1], dtype=torch.long))
    phi = torch.tensor([0.0, 10.0], dtype=F64)
    rho = torch.tensor([1.2, 1.1], dtype=F64)
    with pytest.raises(KeyError, match="species.capacity"):
        zp({"air.phi": phi}, {"rho": rho})


def test_zone_pressure_density_broadcasts_a_batched_reference_pressure():
    # B == n: a (B,) P_ref must align with the batch axis, not the node axis.
    zp = ZonePressureDensity(101325.0)
    phi = torch.tensor([[0.0, 10.0], [0.0, 20.0]], dtype=F64)
    rho = torch.tensor([[1.2, 1.1], [1.2, 1.1]], dtype=F64)
    P = torch.tensor([101325.0, 90000.0], dtype=F64)
    out = zp({"air.phi": phi}, {"rho": rho, "P_ref": P})
    assert out["rho"][1, 1].item() == pytest.approx(1.1 * (1 + 20.0 / 90000.0), rel=1e-15)
    assert out["rho"][0, 1].item() == pytest.approx(1.1 * (1 + 10.0 / 101325.0), rel=1e-15)


STACK_OWN = DATA / "test_OneZoneWthCtmStack.prj"
_NO_SCHEDULES = "0 ! day-schedules:\n-999\n0 ! week-schedules:\n-999\n"
_HALF_SCHEDULE = (
    "1 ! day-schedules:\n! # npts shap utyp ucnv name\n  1    3    0    1    0 Half\n\n"
    " 00:00:00 1\n 12:00:00 0.5\n 24:00:00 0.5\n-999\n"
    "1 ! week-schedules:\n! # utyp ucnv name\n  1    1    0 HalfWeek\n\n"
    " 1 1 1 1 1 1 1 1 1 1 1 1\n-999\n"
)
_STACK_DATES = "  Jan01 00:00:00  Jan01 00:00:00  Jan01 24:00:00"


def test_the_run_control_is_read():
    rc = read_prj(STACK_OWN).run_control
    assert isinstance(rc, RunControl)
    assert rc.transient_airflow and rc.contaminants == "transient"
    assert (rc.integration, rc.integration_weight) == ("trapezoid", 1.0)
    assert (rc.start_day, rc.start_time, rc.end_day, rc.end_time) == (1, 0.0, 1, 86400.0)
    assert (rc.time_step, rc.output_step, rc.day_type) == (300.0, 300.0, 1)
    assert rc.duration == 86400.0
    assert (rc.weather_file, rc.ambient_file) == (
        "test_OneZoneWthCtmStack.wth", "test_OneZoneWthCtmStack.ctm")
    assert rc.zone_mass_change
    assert read_prj(THREE).run_control.weather_file is None        # `null`
    assert not read_prj(THREE).run_control.zone_mass_change


def test_the_steady_day_type_stays_off_the_ambient_conditions():
    p = read_prj(STACK_OWN)
    assert set(p.ambient_conditions) == {"Ta", "Pb", "Ws", "Wd"}
    cal = p.run_control.steady_calendar()
    assert (cal.start_day, cal.start_time, cal.start_day_type) == (1, 0.0, 1)


def test_run_control_dates_are_days_of_the_year(tmp_path):
    p = read_prj(_variant(STACK_OWN.read_text(), _STACK_DATES,
                          "  Mar02 06:00:00  Feb28 12:00:00  Mar01 18:00:00", tmp_path))
    rc = p.run_control
    assert (rc.steady_day, rc.steady_time) == (61, 6 * 3600.0)
    assert (rc.start_day, rc.start_time, rc.end_day, rc.end_time) == (
        59, 12 * 3600.0, 60, 18 * 3600.0)
    assert rc.duration == 30 * 3600.0


def test_a_run_control_without_its_dates_or_with_a_bad_date_is_refused(tmp_path):
    text = STACK_OWN.read_text()
    with pytest.raises(ValueError, match=r"run-control section lacks steady_day"):
        read_prj(_variant(text, "!date_st time_st", "!dates_removed", tmp_path))
    with pytest.raises(ValueError, match=r"run-control section lacks weather_file"):
        read_prj(_variant(text, "! weather file", "! no such line", tmp_path))
    with pytest.raises(ValueError, match=r"contaminant simulation mode \(sim_mf\) 7"):
        read_prj(_variant(text, "    2             30", "    7             30", tmp_path))
    with pytest.raises(ValueError, match=r"'Xyz01' is not a CONTAM date"):
        read_prj(_variant(text, _STACK_DATES, "  Jan01 00:00:00  Xyz01 00:00:00  Jan01 24:00:00",
                          tmp_path))


def test_a_zone_temperature_schedule_becomes_an_input_on_T():
    p = read_prj(STACK_OWN)
    assert isinstance(p.schedules[1], WeekSchedule)
    (inp,) = p.inputs
    assert inp.key == "T" and inp.index == (p.net.node_index("singleZone"),)
    cal = Calendar()
    # Values at the END of a step; a jump's own instant keeps the value before it.
    assert inp.signal.value(13 * 3600.0, cal).item() == pytest.approx(293.15)
    assert inp.signal.value(13 * 3600.0 + 300.0, cal).item() == pytest.approx(303.15)
    assert p.ambient_mass_fractions.tolist() == pytest.approx([5.8135e-4])
    assert p.source_path == STACK_OWN.resolve()
    model, state, drivers = project_to_model(p)
    xb = drivers["species.x_boundary"]
    assert xb.shape == (1, 1) and xb.item() == pytest.approx(5.8135e-4)


def test_the_returned_drivers_are_at_the_steady_simulation_instant(tmp_path):
    text = STACK_OWN.read_text()
    # steady time 13:30 -> the zone is at 303.15 K in the returned drivers
    shifted = _variant(text, "  Jan01 00:00:00  Jan01 00:00:00", "  Jan01 13:30:00  Jan01 00:00:00",
                       tmp_path)
    p = read_prj(shifted)
    assert p.run_control.steady_time == 13.5 * 3600.0
    _, _, drivers = project_to_model(p)
    zone = p.net.node_index("singleZone")
    assert drivers["T"][zone].item() == pytest.approx(303.15)
    # the initial densities kept for code outside a step follow the scheduled temperature
    assert drivers["rho"][zone].item() == pytest.approx(101325.0 / (R_AIR * 303.15), rel=1e-15)


def _three_with_path_schedule(tmp_path) -> Path:
    text = THREE.read_text().replace(_NO_SCHEDULES, _HALF_SCHEDULE, 1)
    path2 = "   2    0   1   2   3   0   0   0   0   0   1"
    return _variant(text, path2, "   2    0   1   2   3   0   0   0   1   0   1", tmp_path)


def test_a_path_schedule_wraps_the_kind_in_scaled_and_writes_its_edge(tmp_path):
    p = read_prj(_three_with_path_schedule(tmp_path))
    (el,) = [e for e in p.elements if e.kind == "pl_3"]
    assert isinstance(el, Scaled) and el.key == "flow_multiplier.pl_3"
    assert p.multiplier_edges == {"pl_3": 4}
    (inp,) = p.inputs
    assert inp.key == "flow_multiplier.pl_3" and inp.index == (1,)   # path 2 is pl_3's 2nd edge
    model, state, drivers = project_to_model(p)
    assert drivers["flow_multiplier.pl_3"].tolist() == [1.0, 1.0, 1.0, 1.0]   # 00:00 steady
    assert model.refs.inputs["flow_multiplier.pl_3"].ordering == "edge order of kind 'pl_3'"


def test_a_scheduled_path_scales_its_flow_batched_and_unbatched(tmp_path):
    p = read_prj(_three_with_path_schedule(tmp_path))
    model, state, drivers = project_to_model(p)
    plain = read_prj(THREE)
    m0, s0, d0 = project_to_model(plain)
    q_ref = plain.path_flows(prj_steady(m0, s0, d0, atol=1e-14)["air.q"])
    q_one = p.path_flows(prj_steady(model, state, drivers, atol=1e-14)["air.q"])
    torch.testing.assert_close(q_one, q_ref, rtol=1e-12, atol=1e-15)
    s = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 1.0, 1.0]], dtype=F64)
    q_two = p.path_flows(prj_steady(model, state, {**drivers, "flow_multiplier.pl_3": s},
                                    atol=1e-14)["air.q"])
    assert q_two.shape == (2, 4)
    torch.testing.assert_close(q_two[0], q_ref, rtol=1e-12, atol=1e-15)
    assert not torch.allclose(q_two[1], q_ref)


def test_a_scheduled_fixed_flow_fan_gets_no_stack_drive(tmp_path):
    text = MIXED.read_text().replace(_NO_SCHEDULES, _HALF_SCHEDULE, 1)
    path5 = "   5    0   1  -1   1   0   0   0   0   0   1"
    p = read_prj(_variant(text, path5, "   5    0   1  -1   1   0   0   0   1   0   1", tmp_path))
    (fan,) = [e for e in p.elements if e.kind == "fan_1"]
    assert isinstance(fan, Scaled) and isinstance(fan.inner, FixedFlow)
    assert "fan_1" not in [d.kind for d in p.drives]


def test_schedule_references_that_name_nothing_are_refused(tmp_path):
    text = THREE.read_text()
    path2 = "   2    0   1   2   3   0   0   0   0   0   1"
    with pytest.raises(KeyError, match=r"path 2.*schedule 1"):
        read_prj(_variant(text, path2, "   2    0   1   2   3   0   0   0   1   0   1", tmp_path))
    zone1 = "   1  3   0   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 0 0 0"
    with pytest.raises(KeyError, match=r"zone 1.*schedule 2"):
        read_prj(_variant(text, zone1,
                          "   1  3   2   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 0 0 0",
                          tmp_path))
    bad_week = _HALF_SCHEDULE.replace(" 1 1 1 1 1 1 1 1 1 1 1 1", " 1 1 1 1 1 1 1 1 1 1 1 4")
    with pytest.raises(KeyError, match=r"week schedule 1.*day schedule\(s\) \[4\]"):
        read_prj(_variant(text, _NO_SCHEDULES, bad_week, tmp_path))
    bad_shape = _HALF_SCHEDULE.replace("  1    3    0    1    0 Half",
                                       "  1    3    2    1    0 Half")
    with pytest.raises(ValueError, match=r"day schedule 1 has shape 2"):
        read_prj(_variant(text, _NO_SCHEDULES, bad_shape, tmp_path))


def test_a_zone_held_at_constant_pressure_is_refused_by_name(tmp_path):
    zone1 = "   1  3   0   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 0 0 0"
    text = THREE.read_text()
    with pytest.raises(ValueError, match=r"zone 1 \('one'\).*flags 2.*variable-pressure"):
        read_prj(_variant(text, zone1, zone1.replace("   1  3", "   1  2", 1), tmp_path))
    p = read_prj(_variant(text, zone1, zone1.replace("   1  3", "   1  19", 1), tmp_path))
    assert p.zone_flags == {"one": 19, "two": 3, "three": 3}


def test_a_constant_wind_pressure_is_a_constant_drive_on_the_ambient_side(tmp_path):
    text = THREE.read_text()
    p1 = "   1    1  -1   1   3   0   3   0   0   0   1  30.000   0.000   1.500 1 0 0.1225 270"
    p = read_prj(_variant(
        text, p1,
        "   1    1  -1   1   3   0   0   0   0   0   1  30.000   0.000   1.500 1 5 0.1225 270",
        tmp_path))
    torch.testing.assert_close(p.constant_wind_pressure["pl_3"],
                               torch.tensor([5.0, 0.0, 0.0, 0.0], dtype=F64))
    model, state, drivers = project_to_model(p)
    assert torch.equal(drivers["wind_pressure.pl_3"], p.constant_wind_pressure["pl_3"])
    assert model.refs.inputs["wind_pressure.pl_3"].unit == "Pa"


def test_a_constant_wind_pressure_on_an_interior_path_is_refused(tmp_path):
    path2 = "   2    0   1   2   3   0   0   0   0   0   1  10.000   0.000   1.500 1 0 "
    text = THREE.read_text()
    with pytest.raises(ValueError, match=r"path 2.*constant wind pressure.*envelope"):
        read_prj(_variant(text, path2, path2[:-2] + "4 ", tmp_path))


def _scheduled_source_text(shape_npts, rows):
    text = (DATA / "doorway_damper_fan.prj").read_text()
    return text.replace(
        "0 ! day-schedules:\n-999\n0 ! week-schedules:\n-999\n",
        "1 ! day-schedules:\n! # npts shap utyp ucnv name\n"
        f"  1    {shape_npts}    0    1    0 On\n\n{rows}-999\n"
        "1 ! week-schedules:\n! # utyp ucnv name\n  1    1    0 OnWeek\n\n"
        " 1 1 1 1 1 1 1 1 1 1 1 1\n-999\n", 1)


_SRC_OFF = "  1   1   1   0   0     1     0"
_SRC_ON = "  1   1   1   1   0     1     0"


def test_a_source_schedule_becomes_an_input_on_the_source_multiplier(tmp_path):
    text = _scheduled_source_text(3, " 00:00:00 0\n 09:00:00 1\n 24:00:00 1\n")
    p = read_prj(_variant(text, _SRC_OFF, _SRC_ON, tmp_path))
    (inp,) = p.inputs
    assert inp.key == "species.source_multiplier" and inp.index == (0,)
    _, _, drivers = project_to_model(p)
    assert drivers["species.source_multiplier"].tolist() == [0.0]       # off at 00:00


def test_decaying_and_burst_sources_with_a_schedule_are_refused(tmp_path):
    text = _scheduled_source_text(2, " 00:00:00 1\n 24:00:00 1\n")
    text = text.replace("1 sarin ccf ccm\n\n 0.0001 0 5 0", "1 sarin brs burst\n\n 0.0001 0", 1)
    with pytest.raises(ValueError, match=r"source 1.*brs.*schedule"):
        read_prj(_variant(text, _SRC_OFF, _SRC_ON, tmp_path))


def test_a_source_naming_an_undefined_species_is_refused(tmp_path):
    text = (DATA / "doorway_damper_fan.prj").read_text()
    variant = _variant(text, "1 sarin ccf ccm", "1 nosuch ccf ccm", tmp_path)
    with pytest.raises(KeyError, match=r"source 1.*element 1.*'nosuch'"):
        read_prj(variant)


def test_a_zone_without_the_variable_contaminants_bit_is_refused_by_name(tmp_path):
    text = (DATA / "doorway_damper_fan.prj").read_text()
    variant = _variant(text, "   1  3   0   0   0   1   0.000   600",
                       "   1  1   0   0   0   1   0.000   600", tmp_path)
    with pytest.raises(ValueError, match=r"zone 1 \('singleZone'\).*variable-contaminants"):
        read_prj(variant)


def test_the_scheduled_three_zone_project_reads_its_four_inputs():
    p = read_prj(DATA / "scheduled_three_zone.prj")
    keys = sorted((i.key, i.index) for i in p.inputs)
    assert keys == sorted([
        ("T", (p.net.node_index("two"),)),
        ("flow_multiplier.pl_3", (0,)),                       # path 1, pl_3's first edge
        ("flow_multiplier.fan_2", (0,)),                      # path 5
        ("species.source_multiplier", (0,)),
    ])
    assert p.run_control.weather_file == "scheduled_three_zone.wth"


def test_a_zone_mass_change_project_holds_its_zone_air_mass_at_the_absolute_pressure():
    """ContamX with dodMdt = 1 and densZP = 0 (measured on the scheduled three-zone project):
    the densities the flows use stay at Pb, while the zone air mass the contaminants see is
    (Pb + P_zone) V / (R T) (its mass fractions then agree to 2.5e-11, against 4.4e-9 with the
    mass at Pb)."""
    p = read_prj(DATA / "scheduled_three_zone.prj")
    assert p.run_control.zone_mass_change and not p.density_uses_zone_pressure
    model, state, drivers = project_to_model(p)
    phi = torch.tensor([0.0, -1.0, -2.0, -3.0], dtype=torch.float64)
    amb = p.net.node_index(p.ambient)
    phi = torch.roll(phi, amb)                       # zero gauge pressure at the ambient
    out = model._apply_closures({**state, "air.phi": phi}, drivers)
    T, Pb = drivers["T"], drivers["P_ref"]
    torch.testing.assert_close(out["rho"], Pb / (R_AIR * T), rtol=1e-15, atol=0)
    nodes = model.transport["species"].interior_idx
    V = p.net.node_attr("volume", default=0.0)
    expected = ((Pb + phi) * V / (R_AIR * T))[nodes]
    torch.testing.assert_close(out["species.capacity"], expected, rtol=1e-15, atol=0)


def test_a_non_integer_option_code_names_the_option():
    from noodl.apps.building_physics.prj import _CONTAMINANT_MODES, _code

    with pytest.raises(ValueError, match=r"mode 'x' is not an integer"):
        _code(_CONTAMINANT_MODES, "x", "the mode")
    assert _code(_CONTAMINANT_MODES, "2", "the mode") == "transient"


def test_a_date_with_an_impossible_day_is_refused():
    from noodl.apps.building_physics.prj import _date

    assert _date("Feb28") == 59 and _date("Dec31") == 365
    for bad in ("Feb29", "Jan00", "Apr31", "Jan32"):
        with pytest.raises(ValueError, match="not a CONTAM date"):
            _date(bad)


REACTION = DATA / "reaction_three_zone.prj"


def _variants(text: str, pairs, tmp_path: Path) -> Path:
    """`_variant` applied once per `(old, new)` pair, each on the previous one's output."""
    for old, new in pairs:
        out = _variant(text, old, new, tmp_path)
        text = out.read_text()
    return out


def test_the_species_record_is_read_whole():
    p = read_prj(REACTION)
    assert p.species == ["sarin", "product"]
    sarin, product = p.species_records
    assert isinstance(sarin, Species) and sarin.name == "sarin"
    assert sarin.molar_mass == pytest.approx(140.09) and sarin.diffusivity == 2e-5
    assert product.molar_mass == pytest.approx(44.01) and product.description == (
        "reaction product")
    assert p.ambient_mass_fractions.tolist() == [0.0, 0.0]


def test_zone_reactions_become_the_species_layers_kinetics():
    p = read_prj(REACTION)
    assert set(p.zone_reactions) == {"one", "three"}
    assert isinstance(p.zone_reactions["one"], KineticReaction)
    model, _, _ = project_to_model(p)
    kin = model.transport["species"].kinetics                    # (n_i, K, K)
    torch.testing.assert_close(kin[0], torch.tensor([[-1e-3, 0.0], [1e-3, -5e-4]],
                                                    dtype=F64))
    assert kin[1].abs().sum().item() == 0.0
    torch.testing.assert_close(kin[2], torch.tensor([[-2e-4, 0.0], [1e-4, 0.0]], dtype=F64))


def test_a_species_decay_constant_is_kept_but_not_applied(tmp_path):
    text = THREE.read_text()
    with pytest.warns(ContamCompatibilityWarning):
        p = read_prj(_variant(text, "140.0900  0.0000e+00  0.0000e+00  0.0000e+00",
                              "140.0900  0.0000e+00  0.0000e+00  1.0000e-03", tmp_path))
    assert p.species_records[0].decay == 1e-3
    model, _, _ = project_to_model(p)
    assert model.transport["species"].kinetics is None          # ContamX ignores it


_DECAY_OLD = "140.0900  0.0000e+00  0.0000e+00  0.0000e+00"
_DECAY_NEW = "140.0900  0.0000e+00  0.0000e+00  1.0000e-03"


def test_a_nonzero_species_decay_warns_once_naming_species_and_constant(tmp_path):
    path = _variant(THREE.read_text(), _DECAY_OLD, _DECAY_NEW, tmp_path)
    with pytest.warns(ContamCompatibilityWarning) as record:
        read_prj(path)
    assert len(record) == 1
    text = str(record[0].message)
    assert "'sarin'" in text and "0.001 1/s" in text and "ContamX ignores" in text
    assert "DepositionRateSink" in text and "KineticReaction" in text


def test_no_decay_warning_when_the_decay_is_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ContamCompatibilityWarning)
        read_prj(THREE)


def test_the_decay_warning_category_is_filterable(tmp_path):
    path = _variant(THREE.read_text(), _DECAY_OLD, _DECAY_NEW, tmp_path)
    assert issubclass(ContamCompatibilityWarning, UserWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=ContamCompatibilityWarning)
        read_prj(path)


def test_unsimulated_and_non_trace_species(tmp_path):
    text = THREE.read_text()
    extra = ("0.0023254 kg/kg = 2800 mg/m3\n"
             "  2 0 0  28.0000  0.0000e+00  0.0000e+00  0.0000e+00  2.0000e-05  0.0000e+00"
             "  1.0000e+03  0.0000e+00 0 0 0 0 0 CO\n\n")
    p = read_prj(_variants(text, [("1 ! species:", "2 ! species:"),
                                  ("0.0023254 kg/kg = 2800 mg/m3\n", extra)], tmp_path))
    assert p.species == ["sarin"]                       # CO is not simulated
    with pytest.raises(ValueError, match=r"contaminants.*\[1\].*\[1, 2\]"):
        read_prj(_variants(text, [("1 ! species:", "2 ! species:"),
                                  ("0.0023254 kg/kg = 2800 mg/m3\n",
                                   extra.replace("  2 0 0  28.0000", "  2 1 0  28.0000"))],
                           tmp_path))
    with pytest.raises(ValueError, match=r"species 1 \('sarin'\) is non-trace"):
        read_prj(_variant(text, "  1 1 0 140.0900", "  1 1 1 140.0900", tmp_path))


def test_a_reaction_naming_a_species_the_project_lacks_is_refused(tmp_path):
    text = REACTION.read_text()
    with pytest.raises(KeyError, match=r"kinetic reaction 2.*\['ozone'\]"):
        read_prj(_variant(text, "product sarin 0.0001", "ozone sarin 0.0001", tmp_path))


def test_a_reaction_naming_an_unsimulated_species_is_refused(tmp_path):
    """Species 2 exists in the file but is not simulated (`s` = 0): a reaction on it has no
    concentration to act on."""
    text = REACTION.read_text()
    out = _variants(text, [("2 ! contaminants:\n   1 2", "1 ! contaminants:\n   1"),
                           ("  2 1 0  44.0100", "  2 0 0  44.0100")], tmp_path)
    with pytest.raises(KeyError, match=r"kinetic reaction 1.*\['product'\]"):
        read_prj(out)


def test_a_source_naming_an_unsimulated_species_is_refused(tmp_path):
    text = (DATA / "doorway_damper_fan.prj").read_text()
    out = _variants(text, [("3 ! contaminants:\n   1 2 3", "2 ! contaminants:\n   1 3"),
                           ("  2 1 0  28.0000", "  2 0 0  28.0000"),
                           ("1 sarin ccf ccm", "1 CO ccf ccm")], tmp_path)
    with pytest.raises(KeyError, match=r"source 1.*element 1.*'CO'.*does not simulate"):
        read_prj(out)


def test_a_reaction_pair_given_twice_is_refused_by_name(tmp_path):
    text = REACTION.read_text()
    out = _variant(text, "1 3 fast\n\nsarin sarin -0.001\n",
                   "1 4 fast\n\nsarin sarin -0.001\nsarin sarin -0.002\n", tmp_path)
    with pytest.raises(ValueError,
                       match=r"kinetic reaction 1 \('fast'\).*'sarin'.*'sarin'.*twice"):
        read_prj(out)


def _transmission_of(p, layer, path_nr):
    path = next(q for q in p.paths if q.nr == path_nr)
    return layer.transmission[:, path.edge_columns[0]]


def test_the_filtered_one_zone_stack_project_loads_its_filter():
    p = read_prj(SS_STACK)
    assert [q.filter_nr for q in p.paths] == [1, 0]
    (flt,) = set(p.edge_filters.values())
    assert isinstance(flt, ConstantEfficiencyFilter)
    model, _, _ = project_to_model(p)
    layer = model.transport["species"]
    torch.testing.assert_close(_transmission_of(p, layer, 1), torch.tensor([0.9], dtype=F64))
    torch.testing.assert_close(_transmission_of(p, layer, 2), torch.tensor([1.0], dtype=F64))


def test_three_filters_become_the_species_layers_transmission():
    p = read_prj(FILTERS)
    by_path = {q.nr: q.filter_nr for q in p.paths}
    assert by_path == {1: 1, 2: 3, 3: 2, 4: 0}
    kinds = {type(f) for f in p.edge_filters.values()}
    assert kinds == {ConstantEfficiencyFilter, FilterSeries, ParticleFilter}
    layer = project_to_model(p)[0].transport["species"]
    expected = {1: [0.7, 1.0], 2: [0.7, 0.35], 3: [1.0, 0.35], 4: [1.0, 1.0]}
    for nr, values in expected.items():
        torch.testing.assert_close(_transmission_of(p, layer, nr),
                                   torch.tensor(values, dtype=F64), rtol=1e-14, atol=1e-15)


def test_an_unreferenced_filter_element_is_read_and_unused():
    p = read_prj(MIXED)                     # one cef element and filter, on no path
    assert p.edge_filters == {}
    assert project_to_model(p)[0].transport["species"].transmission.eq(1.0).all()


def test_a_gaseous_filter_element_is_refused(tmp_path):
    text = SS_STACK.read_text()
    with pytest.raises(ValueError, match=r"filter element 1.*'gf0'.*loading"):
        read_prj(_variant(text, "1 cef 1 0.1 100 0 0 filt", "1 gf0 1 0.1 100 0 0 filt",
                          tmp_path))


UVGI = DATA / "uvgi_three_zone.prj"
UV_RECORD = "  0.1 0.25 0.01 0.002 1\n  40 1 2 -0.01 -0.02\n  100 -10 0 0\n"


def test_uvgi_filter_elements_are_read_as_one_stream_of_values(tmp_path):
    """ContamX reads a `uv1` element's 14 values whatever the line breaks (measured): S, u,
    F and k of the design, the age-model flag, C0..C4, K0, K1 and two display units."""
    p = read_prj(UVGI)
    assert [s.uv_susceptibility for s in p.species_records] == [1e-3, 2e-2]
    by_path = {q.nr: q.filter_nr for q in p.paths}
    assert by_path == {1: 1, 2: 3, 3: 2, 4: 0}
    flt = {type(f).__name__ for f in p.edge_filters.values()}
    assert flt == {"UVGIFilter", "FilterSeries"}
    (aged,) = [f for f in p.edge_filters.values() if f.name == "uvAged"]
    assert [aged.design_survival.item(), aged.design_velocity.item(),
            aged.design_mass_flow.item(), aged.design_rate_constant.item()] == [
        0.1, 0.25, 0.01, 0.002]
    assert aged.output_coefficients.tolist() == [40.0, 1.0, 2.0, -0.01, -0.02]
    assert aged.age_coefficients.tolist() == [100.0, -10.0]
    (series,) = [f for f in p.edge_filters.values() if isinstance(f, FilterSeries)]
    assert series.stages[1].age_coefficients is None             # flag 0: no age model
    one_line = UV_RECORD.replace("\n  ", " ")
    q = read_prj(_variant(UVGI.read_text(), UV_RECORD, one_line, tmp_path))
    (same,) = [f for f in q.edge_filters.values() if f.name == "uvAged"]
    assert same.output_coefficients.tolist() == aged.output_coefficients.tolist()


def test_malformed_uvgi_records_are_refused(tmp_path):
    text = UVGI.read_text()
    with pytest.raises(ValueError, match=r"element 1 \('uvAged'\): value 5 must be an integer"):
        read_prj(_variant(text, "0.002 1\n", "0.002 1.5\n", tmp_path))
    with pytest.raises(ValueError, match=r"element 1 \('uvAged'\): the age-model flag"):
        read_prj(_variant(text, "0.002 1\n", "0.002 2\n", tmp_path))
    with pytest.raises(ValueError, match=r"element 1 \('uvAged'\) needs 14 values, got 15"):
        read_prj(_variant(text, "100 -10 0 0\n", "100 -10 0 0 0\n", tmp_path))
    with pytest.raises(ValueError, match=r"filter element 1:.*design_survival must be"):
        read_prj(_variant(text, "  0.1 0.25 0.01", "  1.5 0.25 0.01", tmp_path))


def test_a_uvgi_filter_where_noodl_physics_cannot_follow_contamx_is_refused(tmp_path):
    duct = DUCTS.read_text().replace("1 cef 1 0.1 100 0 0 sarin30\n\n  1\n  sarin 0.3\n",
                                     "1 uv1 1 0.1 100 0 0 uv\n\n" + UV_RECORD)
    with pytest.raises(ValueError, match=r"duct segment 1 carries the UVGI filter 'uv'"):
        read_prj(_variant(duct, "", "", tmp_path))


def test_a_uvgi_project_iterates_each_step_on_its_own_flows():
    p = read_prj(UVGI)
    model, state, drivers = project_to_model(p)
    assert model.coupling == "iterate" and set(model.iterate_tol) == {"species"}
    (closure,) = [c for c in model.closures if isinstance(c, FilterTransmission)]
    assert model.closures[-1] is closure
    assert [type(f) for _, _, f in closure.varying] == [UVGIFilter, FilterSeries, UVGIFilter]
    plain = project_to_model(read_prj(FILTERS))[0]
    assert plain.coupling == "pingpong"


def test_air_crossing_a_uvgi_filter_backwards_is_warned_once_per_filter():
    from noodl.apps.building_physics.simulation import simulate_project

    p = read_prj(UVGI)
    with pytest.warns(ContamCompatibilityWarning, match=r"'uvDesign'.*1 J/m2") as caught:
        simulate_project(p)
    assert len([w for w in caught if issubclass(w.category, ContamCompatibilityWarning)]) == 1


def test_a_particle_filter_element_without_flat_ends_is_refused_by_number(tmp_path):
    text = FILTERS.read_text()
    with pytest.raises(ValueError, match=r"filter element 2.*first two and the last two"):
        read_prj(_variant(text, "  3e-07 0.1\n", "  3e-07 0.05\n", tmp_path))


def test_filters_naming_what_the_project_lacks_are_refused(tmp_path):
    text = FILTERS.read_text()
    with pytest.raises(KeyError, match=r"filter element 1.*\['ozone'\]"):
        read_prj(_variant(text, "  sarin 0.3\n", "  ozone 0.3\n", tmp_path))
    with pytest.raises(KeyError, match=r"path 2.*filter 1"):
        read_prj(_variant(THREE.read_text(), "   2    0   1   2   3   0   0",
                          "   2    0   1   2   3   1   0", tmp_path))


def test_an_unknown_filter_type_and_a_super_filter_containing_itself_are_refused(tmp_path):
    text = FILTERS.read_text()
    with pytest.raises(ValueError, match=r"filter element 1.*'xyz'.*unknown filter type"):
        read_prj(_variant(text, "1 cef 1 0.1 100 0 0 sarin30", "1 xyz 1 0.1 100 0 0 sarin30",
                          tmp_path))
    with pytest.raises(ValueError, match=r"filter element 3 \('both'\) contains itself"):
        read_prj(_variant(text, "  2\n  1 2\n", "  2\n  1 3\n", tmp_path))


def test_the_sink_and_emission_source_types_are_read():
    with pytest.warns(ContamCompatibilityWarning, match=r"'sarin'.*ContamX ignores"):
        p = read_prj(SINKS)                                   # its decay constant
    assert [s.source_type for s in p.sources] == ["dvs", "drs", "ccf", "cut", "plm", "pkm"]
    srcs = sources_from_project(p)
    assert [type(s) for s in srcs] == [DepositionVelocitySink, DepositionRateSink,
                                       ConstantSource, CutoffSource, PowerLawDecaySource,
                                       PeakEmissionSource]
    assert srcs[0].velocity.item() == pytest.approx(2e-3) and srcs[0].area.item() == 10.0
    assert srcs[1].rate.item() == pytest.approx(5e-4)
    assert srcs[4].t_p == 600.0 and srcs[5].t_peak == 900.0
    (inp,) = p.inputs
    assert inp.key == "species.source_multiplier" and inp.index == (1,)
    model, state, drivers = project_to_model(p)
    assert drivers["species.source_multiplier"].tolist() == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert model.transport["species"].kinetics is None        # the decay is data only
    (closure,) = [c for c in model.closures if isinstance(c, SpeciesSources)]
    assert closure(state, drivers)["species.removal"].shape == (3, 1)


def test_a_pressure_driven_source_is_read_with_the_node_heights(tmp_path):
    """The closure places the ambient pressure at each zone's own reference height: the
    zone's level raised to 4 m reaches it as that node's height, with the project's g."""
    p = read_prj(PRESSURE)
    (src,) = sources_from_project(p)
    assert isinstance(src, PressureDrivenSource)
    assert src.coefficient.item() == pytest.approx(2e-6) and src.exponent.item() == 0.65
    level = "  1   0.000   3.000 10 0 0 <1>"
    p = read_prj(_variant(PRESSURE.read_text(), level, level.replace("0.000", "4.000"),
                          tmp_path))
    model, _, _ = project_to_model(p)
    (closure,) = [c for c in model.closures if isinstance(c, SpeciesSources)]
    assert closure.reads_solved_pressure
    assert closure.heights.dtype == F64
    assert closure.heights.tolist() == [0.0, 4.0]                     # ambient, singleZone
    assert p.net.nodes == ["ambient", "singleZone"]
    assert closure.g == p.g == pytest.approx(9.8055)


@pytest.mark.filterwarnings("ignore::noodl.apps.building_physics.prj.ContamCompatibilityWarning")
def test_unsupported_sources_are_refused_by_name(tmp_path):
    text = SINKS.read_text()
    dvs = "1 sarin dvs deposit\n\n 0.001 10 0 0"
    for stype, why in (("bls", "boundary-layer"), ("dvr", "resuspension")):
        with pytest.raises(ValueError, match=rf"source 1.*'{stype}'.*{why}"):
            read_prj(_variant(text, dvs, dvs.replace("dvs", stype), tmp_path))
    with pytest.raises(ValueError, match=r"source 1.*'sup'.*super"):
        read_prj(_variant(text, dvs, "1 sarin sup deposit\n\n 2\n 3 4", tmp_path))
    plm = "  5   2   5   0   0     1     0"
    with pytest.raises(ValueError, match=r"source 5.*plm.*schedule 1"):
        read_prj(_variant(text, plm, "  5   2   5   1   0     1     0", tmp_path))


@pytest.mark.filterwarnings("ignore::noodl.apps.building_physics.prj.ContamCompatibilityWarning")
def test_an_unreferenced_super_element_ahead_of_a_used_one_is_read_and_ignored(tmp_path):
    """Only a source that references a super element is refused; the element's own data
    (a count, then that many sub-element numbers on the next line) is read past."""
    text = SINKS.read_text()
    head = "6 ! source/sink elements:\n1 sarin dvs deposit"
    p = read_prj(_variant(text, head,
                          "7 ! source/sink elements:\n7 sarin sup group\n\n 2\n 3 4\n"
                          "1 sarin dvs deposit", tmp_path))
    assert [s.source_type for s in p.sources] == ["dvs", "drs", "ccf", "cut", "plm", "pkm"]


def test_a_filter_with_its_load_pair_split_over_two_lines_is_refused_by_number(tmp_path):
    text = FILTERS.read_text()
    with pytest.raises(ValueError, match=r"filter 1 needs the pair `load nsload` on one line"):
        read_prj(_variant(text, "  1   1   1\n   0 0\n", "  1   1   1\n   0\n   0\n",
                          tmp_path))


FAN_CURVE = DATA / "fan_curve_three_zone.prj"


def test_a_fan_curve_path_is_an_affinity_fan_on_its_speed_schedule():
    p = read_prj(FAN_CURVE)
    (fan,) = [e for e in p.elements if isinstance(e, AffinityFan)]
    assert fan.kind == "fan_4" and fan.rho_ref == pytest.approx(1.2041)
    assert fan.off == pytest.approx(0.1) and fan.multiplier.tolist() == [1.5]
    assert fan.speed_key == "fan_speed.fan_4" and fan.shutoff is not None
    assert p.speed_edges == {"fan_4": 1}
    (inp,) = [i for i in p.inputs if i.key == "fan_speed.fan_4"]
    cal = Calendar()
    assert inp.signal.value(1800.0, cal).item() == pytest.approx(1.0)       # left limit
    assert inp.signal.value(2100.0, cal).item() == pytest.approx(0.8)
    model, state, drivers = project_to_model(p)
    assert drivers["fan_speed.fan_4"].tolist() == [1.0]
    assert model.refs.inputs["fan_speed.fan_4"].unit == "-"


def test_an_unscheduled_fan_curve_runs_at_its_rated_speed(tmp_path):
    text = FAN_CURVE.read_text()
    p5 = "   5    0   3  -1   4   0   0   0   1   0   1"
    p = read_prj(_variant(text, p5, "   5    0   3  -1   4   0   0   0   0   0   1", tmp_path))
    (fan,) = [e for e in p.elements if isinstance(e, AffinityFan)]
    assert fan.speed_key is None and p.speed_edges == {}


def test_a_batched_fan_speed_steps_as_per_instance_runs():
    """A control drives "fan_speed.<kind>" with a batch of speeds (running, slowed, stopped
    below its off ratio, overdriven); one batched step equals the per-instance steps."""
    p = read_prj(FAN_CURVE)
    model, state, drivers = project_to_model(p)
    speeds = [1.0, 0.8, 0.05, 1.2]
    batched = model.step(state, {**drivers, "fan_speed.fan_4": torch.tensor(
        [[s] for s in speeds], dtype=F64)}, 300.0, atol=1e-14)
    for i, s in enumerate(speeds):
        one = model.step(state, {**drivers, "fan_speed.fan_4": torch.tensor([s], dtype=F64)},
                         300.0, atol=1e-14)
        for key in ("air.q", "species.x"):
            torch.testing.assert_close(batched[key][i], one[key], rtol=1e-10, atol=1e-15)
    fan_flow = p.path_flows(batched["air.q"])[:, 4]                  # path 5
    assert len({round(v, 9) for v in fan_flow.tolist()}) == 4        # four distinct flows


SPLINES = DATA / "spline_three_zone.prj"


def test_the_four_spline_elements_are_read():
    p = read_prj(SPLINES)
    forms = {e.kind: (e.independent, e.volume) for e in p.elements
             if isinstance(e, CubicSplineFlow)}
    assert forms == {"sp_4": ("pressure", False), "sp_5": ("pressure", True),
                     "sp_6": ("flow", False), "sp_7": ("flow", True)}


def test_a_spline_record_contamx_would_refuse_is_refused_naming_its_element(tmp_path):
    text = SPLINES.read_text()
    head = "4 23 csf_fsp csf_fsp\nmeasured curve\n"
    rows = " 0 0\n 2 0.005\n 5 0.009\n 10 0.014\n 20 0.02\n"
    few = _variant(text, head + " 5 0 0\n" + rows,
                   head + " 3 0 0\n 0 0\n 2 0.005\n 5 0.009\n", tmp_path)
    with pytest.raises(ValueError, match=r"flow element 4.*at least four"):
        read_prj(few)
    flat = _variant(text, head + " 5 0 0\n" + rows,
                    head + " 5 0 0\n" + rows.replace("5 0.009", "5 0.005"), tmp_path)
    with pytest.raises(ValueError, match=r"flow element 4.*positive"):
        read_prj(flat)


def test_a_super_element_is_refused_naming_contamx_s_own_refusal(tmp_path):
    text = SPLINES.read_text()
    with pytest.raises(ValueError, match=r"element 4.*'sup_afe'.*ContamX"):
        read_prj(_variant(text, "4 23 csf_fsp csf_fsp\nmeasured curve\n 5 0 0\n",
                          "4 23 sup_afe super\n\n 2 0 0\n", tmp_path))


DUCTS = DATA / "duct_three_zone.prj"
A20_FILE, A15_FILE = 0.0314159, 0.0176715            # the duct elements' areas as written


def test_junctions_become_nodes_with_contamx_s_volumes():
    p = read_prj(DUCTS)
    assert p.junctions == [f"junction {k}" for k in range(1, 10)]
    v = {n: p.net.node_attr("volume")[p.net.node_index(n)].item() for n in p.junctions}
    expected = {1: 2.0 * A20_FILE, 2: 2.5 * A20_FILE, 3: 0.85 * A20_FILE,
                4: 2.85 * A20_FILE + 1.5 * A15_FILE, 5: 1.5 * A15_FILE, 6: 2.5 * A20_FILE,
                7: 0.5 * A20_FILE, 8: 3.5 * A20_FILE, 9: 3.0 * A20_FILE}
    for k, vol in expected.items():
        assert v[f"junction {k}"] == pytest.approx(vol, rel=1e-12), k
    z = p.net.node_attr("z_ref")
    assert z[p.net.node_index("junction 4")].item() == pytest.approx(2.7)
    assert p.junction_x0.shape == (9, 1)


def test_segments_and_terminals_become_their_elements():
    p = read_prj(DUCTS)
    by_kind = {e.kind: e for e in p.elements}
    assert isinstance(by_kind["duct_1"], AirDuct)
    assert by_kind["duct_1"].length.tolist() == [4.0, 0.7]
    assert by_kind["duct_1"].loss.tolist() == [0.0, 0.5]
    assert isinstance(by_kind["duct_3"], AffinityFan)
    assert by_kind["duct_3"].speed_key == "fan_speed.duct_3"
    assert isinstance(by_kind["duct_6"], FixedVolumeFlow)
    term = by_kind["duct_terminal"]
    assert isinstance(term, DuctTerminal)
    assert term.loss.tolist() == pytest.approx([0.5, 0.425, 0.125, 0.125, 1.0])
    assert term.opening_key == "terminal_opening.duct_terminal"
    assert [t.junction_nr for t in p.terminals] == [1, 5, 6, 7, 9]
    keys = sorted((i.key, i.index) for i in p.inputs)
    assert ("fan_speed.duct_3", (0,)) in keys
    assert ("terminal_opening.duct_terminal", (2,)) in keys
    assert len(p.edge_filters) == 2                     # segment 1 and terminal 6


def test_the_duct_model_runs_its_initial_solve():
    p = read_prj(DUCTS)
    model, state, drivers = project_to_model(p)
    out = model.steady(state, drivers, atol=1e-15, rtol=0.0)   # junctions balanced
    flows = p.terminal_flows(out["air.q"])
    assert flows.shape == (5,)
    assert flows[3].item() == pytest.approx(-flows[4].item(), rel=1e-12)   # exhaust balance
    assert flows[4].item() == pytest.approx(0.04 * 101325 / (R_AIR * 293.15), rel=1e-12)
    assert state["species.x"].shape == (3 + 9, 1)
    assert p.zone_mass_fractions(state["species.x"]).shape == (3, 1)
    assert p.segment_flows(out["air.q"]).shape == (7,)


def test_duct_records_contamx_cannot_run_or_ignores_are_refused(tmp_path):
    text = DUCTS.read_text()
    seg1 = "  1  0   1   2   1   1   0   0   4 4"
    with pytest.raises(ValueError, match=r"segment 1.*Darcy-Colebrook.*ignores"):
        read_prj(_variant(text, seg1, "  1  0   1   2   1   1   1   0   4 4", tmp_path))
    with pytest.raises(ValueError, match=r"segment 1 references control node 3 on "
                                         r"Darcy-Colebrook duct element 1"):
        read_prj(_variant(text, seg1, "  1  0   1   2   1   1   0   3   4 4", tmp_path))
    geo = " 0.2 0.628319 0.0314159 0.2 0.2 0 0 250\n 0 4 4 2 4 4 0 0\n2 23 dct_dwc"
    with pytest.raises(ValueError, match=r"duct element 1.*leakage"):
        read_prj(_variant(text, geo, geo.replace("0.2 0 0 250", "0.2 0 0.5 250", 1),
                          tmp_path))
    j1 = "  1  1 3  -1"
    with pytest.raises(ValueError, match=r"terminal 1.*wind.*flag value 1"):
        read_prj(_variant(text, j1, "  1  0 3  -1", tmp_path))
    j2 = "  2  0 0  1   0   0   0"
    with pytest.raises(ValueError, match=r"junction 2.*temperature"):
        read_prj(_variant(text, j2, "  2  0 0  1   0   0   2", tmp_path))
    dens = "   0    0.75    20     1      0      0      0"
    with pytest.raises(ValueError, match=r"densZP.*duct"):
        read_prj(_variant(text, dens, "   0    0.75    20     1      1      0      0",
                          tmp_path))
    with pytest.raises(ValueError, match=r"airflow mode is 2"):
        read_prj(_variant(text, "     1      1     30   1e-05",
                          "     2      1     30   1e-05", tmp_path))


def test_a_duct_network_with_the_zone_air_mass_change_is_refused(tmp_path):
    """Under `dodMdt = 1` the junctions' air mass is unmeasured: refused like `densZP`."""
    dens = "   0    0.75    20     1      0      0      0"
    with pytest.raises(ValueError, match=r"dodMdt = 1.*duct network"):
        read_prj(_variant(DUCTS.read_text(), dens,
                          "   0    0.75    20     1      0      0      1", tmp_path))


def test_a_junction_without_volume_is_a_storage_free_node(tmp_path):
    """Zero-length damper and exhaust-fan segments leave junctions 7, 8 and 9 without volume:
    they hold no sarin, so each carries exactly the mass fraction of the zone-three air
    passing through them, every step."""
    from noodl.apps.building_physics.simulation import simulate_project

    p = read_prj(_variants(DUCTS.read_text(), [
        ("  6  0   7   8   5   0   0   0   4 1 ", "  6  0   7   8   5   0   0   0   4 0 "),
        ("  7  0   8   9   6   0   0   0   4 6 ", "  7  0   8   9   6   0   0   0   4 0 "),
    ], tmp_path))
    volumes = p.net.node_attr("volume")
    free = [f"junction {k}" for k in (7, 8, 9)]
    assert [volumes[p.net.node_index(j)].item() for j in free] == [0.0, 0.0, 0.0]
    model, _, _ = project_to_model(p)
    assert model.transport["species"].storage_free_mask.sum().item() == 3
    out = simulate_project(p, atol=1e-15, rtol=0.0)     # junction inflow = outflow
    x = out["species.x"][1:, :, 0]
    rows = [3 + p.junctions.index(j) for j in free]
    zone_three = x[:, 2:3].expand(-1, 3)
    torch.testing.assert_close(x[:, rows], zone_three, rtol=1e-12, atol=0)
    assert bool((zone_three[-1] > 1e-4).all())


def test_a_steady_solve_with_an_idle_junction_without_volume_is_refused_by_name(tmp_path):
    """With the exhaust fan's flow set to 0 nothing passes through junctions 7, 8 and 9,
    which hold no air: a steady contaminant solve has no value for them."""
    from noodl.elements import FixedVolumeFlow

    p = read_prj(_variants(DUCTS.read_text(), [
        ("  6  0   7   8   5   0   0   0   4 1 ", "  6  0   7   8   5   0   0   0   4 0 "),
        ("  7  0   8   9   6   0   0   0   4 6 ", "  7  0   8   9   6   0   0   0   4 0 "),
    ], tmp_path))
    model, state, drivers = project_to_model(p)
    (fan,) = [el for el in model.potential["air"]._elements if isinstance(el, FixedVolumeFlow)]
    with torch.no_grad():
        fan.volume_flow.zero_()
    with pytest.raises(ValueError, match=r"TransportLayer 'species' steady: nothing passes "
                                         r"through storage-free node\(s\) \['junction 7', "
                                         r"'junction 8', 'junction 9'\]"):
        prj_steady(model, state, drivers)


def test_simulate_project_reports_the_terminal_flows(tmp_path):
    """A steady-airflow run of the duct project: one row, the terminals in junction order."""
    from noodl.apps.building_physics.simulation import simulate_project

    text = _variants(DUCTS.read_text(), [
        ("     1      1     30   1e-05", "     0      1     30   1e-05"),
        ("    2             30  1.00e-04", "    0             30  1.00e-04"),
    ], tmp_path)
    p = read_prj(text)
    out = simulate_project(p)
    assert out["terminal_flow"].shape == (1, 5)
    torch.testing.assert_close(out["terminal_flow"], p.terminal_flows(out["air.q"]))
    # The exhaust terminal carries the constant-volume fan's flow, to the solve's tolerance.
    assert out["terminal_flow"][0, 4].item() == pytest.approx(
        0.04 * 101325 / (R_AIR * 293.15), rel=1e-8)


_BALANCE = "4 23 dct_plr balance\n\n 0.000237882 0.0141421 0.5 0.01 0.112838 1 0 0\n"
_CURVE_ROWS = " 5 0 0\n 0 0\n 2 0.005\n 5 0.009\n 10 0.014\n 20 0.02\n"
_CURVE_X, _CURVE_Y = [0.0, 2.0, 5.0, 10.0, 20.0], [0.0, 0.005, 0.009, 0.014, 0.02]


@pytest.mark.parametrize("dtype, data", [
    ("dct_fcn", " 0.000237882 0.0141421 0.5\n"),
    ("dct_qcn", " 0.000237882 0.0141421 0.5\n"),
    ("dct_cmf", " 0.05 0\n"),
    ("dct_bdq", " 0.000237882 0.0141421 0.5 0.01 0.6\n"),
    ("dct_fsp", _CURVE_ROWS),
    ("dct_qsp", _CURVE_ROWS),
    ("dct_psf", _CURVE_ROWS),
    ("dct_psq", _CURVE_ROWS),
])
def test_every_duct_element_type_builds_its_element(dtype, data, tmp_path):
    """The fixture's balancing element (segment 5) rewritten into each other duct type: the
    power laws in the upstream density with exponent m (0 mass, 1 volume: rho_0 turb), the
    volume damper's coefficients scaled by rho_0, the mass-flow fan scheduled through
    `Scaled`, and the four curves on the element's points."""
    variants = [(_BALANCE, f"4 23 {dtype} balance\n\n{data}")]
    if dtype == "dct_cmf":
        variants.append(("  5  0   4   6   4   0   0   0   4 5",
                         "  5  0   4   6   4   0   1   0   4 5"))
    p = read_prj(_variants(DUCTS.read_text(), variants, tmp_path))
    el = {e.kind: e for e in p.elements}["duct_4"]
    rho0 = 1.2041
    if dtype in ("dct_fcn", "dct_qcn"):
        assert isinstance(el, UpstreamDensityPowerLaw)
        C, m = (0.0141421, 0.0) if dtype == "dct_fcn" else (rho0 * 0.0141421, 1.0)
        assert el.C.tolist() == pytest.approx([C], rel=1e-15)
        assert el.n.tolist() == [0.5] and el.m.item() == m
    elif dtype == "dct_cmf":
        assert isinstance(el, Scaled) and isinstance(el.inner, FixedFlow)
        assert el.inner.q0.tolist() == [0.05]
        assert el.key == "flow_multiplier.duct_4" and p.multiplier_edges["duct_4"] == 1
        assert ("flow_multiplier.duct_4", (0,)) in [(i.key, i.index) for i in p.inputs]
    elif dtype == "dct_bdq":
        assert isinstance(el, Damper)
        assert el.pos.C.item() == pytest.approx(rho0 * 0.0141421, rel=1e-15)
        assert el.neg.C.item() == pytest.approx(rho0 * 0.01, rel=1e-15)
        assert (el.pos.n.item(), el.neg.n.item()) == (0.5, 0.6)
        assert el.pos.m.item() == el.neg.m.item() == 1.0
    else:
        assert isinstance(el, CubicSplineFlow)
        form = {"fsp": ("pressure", False), "qsp": ("pressure", True),
                "psf": ("flow", False), "psq": ("flow", True)}[dtype[-3:]]
        assert (el.independent, el.volume) == form
        assert el.curve.x.tolist() == _CURVE_X and el.curve.y.tolist() == _CURVE_Y


def test_initial_concentrations_of_an_undefined_zone_or_junction_are_refused(tmp_path):
    text = DUCTS.read_text()
    with pytest.raises(KeyError, match=r"'initial junction concentrations'.*junction.*\[12\]"):
        read_prj(_variant(text, "   9  0.000e+00\n-999", "   12  0.000e+00\n-999", tmp_path))
    zone_rows = text[text.index("! initial zone concentrations:"):]
    first = zone_rows.splitlines()[2]                      # after the `! Z#` comment line
    assert first.split()[0] == "1", first
    with pytest.raises(KeyError, match=r"'initial zone concentrations'.*zone.*\[7\]"):
        read_prj(_variant(text, first, first.replace("1", "7", 1), tmp_path))


def test_a_junction_no_segment_or_terminal_reaches_is_refused_by_name(tmp_path):
    lone = ("  10  0 0  1   0   0   0   0   1   0.000   0.000   2.000 293.15 0 162 -1 0 0 2 0"
            " 0\n")
    p = read_prj(_variants(DUCTS.read_text(), [
        ("9 ! duct junctions:", "10 ! duct junctions:"),
        ("-999\n9 ! initial junction concentrations:",
         lone + "-999\n9 ! initial junction concentrations:"),
    ], tmp_path))
    with pytest.raises(ValueError, match=r"\['junction 10'\] carry no airflow path"):
        project_to_model(p)


AHS = DATA / "ahs_three_zone.prj"
NIST_AHS = DATA / "test_GetPrjInfo.prj"


def test_a_simple_ahs_becomes_plenums_prescribed_flows_and_its_closure():
    p = read_prj(AHS)
    (element,) = [e for e in p.elements if isinstance(e, PrescribedFlow)]
    assert element.kind == "ahs_1"
    (closure,) = p.closures
    assert isinstance(closure, AirHandlerFlows)
    assert closure.roles.tolist() == [0, 0, 1, 2, 3, 4]
    assert closure.design_flows.tolist() == pytest.approx([0.1, 0.05, 0.12, 0, 0, 0])
    assert closure.min_outdoor_air == pytest.approx(0.02)
    assert p.flow_balanced == ["ahs(Ret)", "ahs(Sup)"]
    assert p.storage_free == ["ahs(Ret)"]                     # the 0 m3 plenum
    assert p.air_handler_edges == {"ahs_1": 6}
    assert p.multiplier_edges == {}                          # the closure applies them
    keys = sorted((i.key, i.index) for i in p.inputs)
    assert ("flow_multiplier.ahs_1", (0,)) in keys            # supply point to zone one
    assert ("outdoor_air_fraction.ahs_1", None) in keys
    assert len(p.edge_filters) == 3                           # return, recirculation, OA
    assert [x.flags for x in p.paths[4:]] == [8, 8, 8, 16, 32, 64]
    assert [x.ahs_nr for x in p.paths[4:]] == [1, 1, 1, 0, 0, 0]
    assert p.paths[7].design_flow == pytest.approx(0.02)


def test_the_ahs_model_solves_with_its_plenums_on_the_boundary():
    p = read_prj(AHS)
    model, state, drivers = project_to_model(p)
    out = prj_steady(model, state, drivers)
    flows = p.path_flows(out["air.q"])
    torch.testing.assert_close(flows[4:], torch.tensor([0.1, 0.05, 0.12, 0.105, 0.045,
                                                        0.015], dtype=F64),
                               rtol=1e-12, atol=1e-15)
    assert drivers["air.phi_boundary"].shape == (3,)
    assert drivers["flow_multiplier.ahs_1"].shape == (6,)
    assert {"flow_multiplier.ahs_1", "outdoor_air_fraction.ahs_1"} <= set(model.refs.inputs)
    # The plenums are ordinary species nodes; the zero-volume one holds no air.
    species = model.transport["species"]
    rows = species.interior_idx.tolist()
    free = species.storage_free_mask.tolist()
    assert free[rows.index(p.net.node_index("ahs(Ret)"))]
    assert not free[rows.index(p.net.node_index("ahs(Sup)"))]


def test_ahs_records_contamx_mishandles_or_ignores_are_refused(tmp_path):
    text = AHS.read_text()
    with pytest.raises(ValueError, match=r"AHS 1.*path 9.*recirculation"):
        read_prj(_variant(text, "  1   4   5   8   9  10 -1 ahs",
                          "  1   4   5   9   8  10 -1 ahs", tmp_path))
    oa = "   9   32  -1   5   0   2   0   0   0   0   1"
    with pytest.raises(ValueError, match=r"path 9.*schedule.*ignores"):
        read_prj(_variant(text, oa, "   9   32  -1   5   0   2   0   0   1   0   1",
                          tmp_path))
    sup = "   5    8   5   1   0   0   0   1   2   0   1   0.000   0.000   0.000 1 0"
    with pytest.raises(ValueError, match=r"path 5.*mult.*ignores"):
        read_prj(_variant(text, sup, sup[:-3] + "2 0", tmp_path))


def test_implicit_paths_before_other_paths_are_refused(tmp_path):
    """ContamX takes the implicit paths by position; ContamW writes them last."""
    text = AHS.read_text()
    lines = text.splitlines(keepends=True)
    i = next(k for k, ln in enumerate(lines) if ln.startswith("  10   64"))
    j = next(k for k, ln in enumerate(lines) if ln.startswith("   5    8"))
    lines.insert(j, lines.pop(i))
    variant = tmp_path / "ahs_order.prj"
    variant.write_text("".join(lines))
    with pytest.raises(ValueError, match=r"implicit.*after every other path"):
        read_prj(variant)


def test_nist_s_ahs_sample_is_refused_for_its_duct_leakage():
    """Its duct elements leak (Qr = 0.5); zone 1 and segment 4 also reference control
    nodes. The parity test runs a copy without them."""
    with pytest.raises(ValueError, match=r"duct element 1.*leakage"):
        read_prj(NIST_AHS)


_ZONE3 = "   3  3   0   0   0   1   0.000   900 293.15 0 three"
_RETURN_PLENUM = "   4 10   0   0   0   1   0.000   0 293.15 0 ahs(Ret)"
_DENSITY_OPTIONS = "   0    0.75    20     1      0      0      0"


def test_only_ahs_plenums_may_lack_the_variable_pressure_bit(tmp_path):
    """The plenum zones (flags 10) are accepted because the AHS record names them; any
    other zone without the bit is still refused, after both sections are read."""
    text = AHS.read_text()
    assert read_prj(AHS).zone_flags["ahs(Ret)"] == 10
    with pytest.raises(ValueError, match=r"zone 3 \('three'\).*flags 10.*variable-pressure"):
        read_prj(_variant(text, _ZONE3, _ZONE3.replace("   3  3", "   3 10", 1), tmp_path))
    no_ahs = _variant(THREE.read_text(), "   1  3   0", "   1 10   0", tmp_path)
    with pytest.raises(ValueError, match=r"zone 1 \('one'\).*flags 10.*variable-pressure"):
        read_prj(no_ahs)


def test_an_ahs_with_zone_pressure_densities_is_refused(tmp_path):
    variant = _variant(AHS.read_text(), _DENSITY_OPTIONS,
                       "   0    0.75    20     1      1      0      0", tmp_path)
    with pytest.raises(ValueError, match=r"densZP = 1.*air-handling"):
        read_prj(variant)


def test_a_reaction_in_a_zero_volume_plenum_is_refused(tmp_path):
    variant = _variants(AHS.read_text(), [
        ("0 ! kinetic reactions:\n-999", "1 ! kinetic reactions:\n1 1 decay\n\n"
                                         "sarin sarin -0.001\n-999"),
        (_RETURN_PLENUM, _RETURN_PLENUM.replace("10   0   0   0", "10   0   0   1", 1)),
    ], tmp_path)
    with pytest.raises(ValueError, match=r"zone 4 .*zero-volume AHS plenum.*kinetic reaction"):
        read_prj(variant)


def test_ahs_paths_with_an_element_or_wind_are_refused(tmp_path):
    text = AHS.read_text()
    oa = "   9   32  -1   5   0   2   0   0   0   0   1"
    with pytest.raises(ValueError, match=r"path 9 .*AHS.*flow element 1"):
        read_prj(_variant(text, oa, "   9   32  -1   5   1   2   0   0   0   0   1", tmp_path))
    with pytest.raises(ValueError, match=r"path 9 .*AHS.*wind"):
        read_prj(_variant(text, oa, "   9   32  -1   5   0   2   1   0   0   0   1", tmp_path))


def test_a_leak_into_an_ahs_plenum_is_refused(tmp_path):
    """A plenum's pressure is not solved, so only the AHS's own flows may touch it."""
    p1 = "   1    1  -1   1   3   0   3   0   0   0   1"
    variant = _variant(AHS.read_text(), p1, "   1    1  -1   5   3   0   3   0   0   0   1",
                       tmp_path)
    with pytest.raises(ValueError, match=r"'ahs\(Sup\)'.*'ambient' -> 'ahs\(Sup\)'.*pl_3"):
        project_to_model(read_prj(variant))


def test_the_plenums_carry_no_zone_air_mass_change(tmp_path):
    """With dodMdt = 1 the zones' air-mass change enters the balance, but not at a plenum,
    whose pressure is not solved; the 2 m3 supply plenum keeps its air mass as a species
    node."""
    from noodl.apps.building_physics.prj import ZoneAirMassChange

    p = read_prj(_variant(AHS.read_text(), _DENSITY_OPTIONS,
                          "   0    0.75    20     1      0      0      1", tmp_path))
    model, state, drivers = project_to_model(p)
    (source,) = [s for s in model.potential["air"].node_sources
                 if isinstance(s, ZoneAirMassChange)]
    assert sorted(p.net.nodes[i] for i in source.nodes.tolist()) == ["one", "three", "two"]


def test_a_steady_contaminant_solve_with_the_ahs_off_is_refused(tmp_path):
    """A plenum with nothing passing through it holds its value; a steady solve has no
    value to give it (ContamX's behaviour there was not measured). `prj.steady` refuses it,
    and `simulate_project` with it."""
    from noodl.apps.building_physics.simulation import simulate_project

    sup1 = "   5    8   5   1   0   0   0   1   2   0   1   0.000   0.000   0.000 1 0 0 0 0.1"
    sup2 = "   6    8   5   2   0   0   0   1   0   0   1   0.000   0.000   0.000 1 0 0 0 0.05"
    variant = _variants(AHS.read_text(), [
        ("     1      1     30   1e-05", "     0      1     30   1e-05"),
        ("    2             30  1.00e-04", "    1             30  1.00e-04"),
        (sup1, sup1[:-3] + "0"), (sup2, sup2[:-4] + "0"),
    ], tmp_path)
    message = (r"air handler 'ahs_1': nothing passes through its supply plenum at the steady "
               r"instant")
    p = read_prj(variant)
    with pytest.raises(ValueError, match=message):
        prj_steady(*project_to_model(p))
    with pytest.raises(ValueError, match=message):
        simulate_project(p)
    airflow_only, state, drivers = project_to_model(p, species=False)
    prj_steady(airflow_only, state, drivers)       # no species: the idle handler is fine


_NIST_READABLE = (
    ("   1  3   0   3   0   1   0.000   600 293.15 0 One",
     "   1  3   0   0   0   1   0.000   600 293.15 0 One"),
    ("  4  0   5   6   2   0   0   5   2 5", "  4  0   5   6   2   0   0   0   2 5"),
    (" 0.2 0.628319 0.0314159 0.2 0.2 0 0.5 1\n", " 0.2 0.628319 0.0314159 0.2 0.2 0 0 1\n"),
)


def _nist(tmp_path, extra=()) -> Path:
    """NIST's two-AHS sample without its control references and duct leakage, plus
    `extra` replacements."""
    return _variants(NIST_AHS.read_text(), [*_NIST_READABLE, *extra], tmp_path)


def test_nist_s_two_ahs_are_read_in_record_order(tmp_path):
    p = read_prj(_nist(tmp_path))
    assert [c.kind for c in p.closures] == ["ahs_1", "ahs_2"]
    assert p.flow_balanced == ["ahsOne(Ret)", "ahsOne(Sup)", "ahsTwo(Ret)", "ahsTwo(Sup)"]


def test_implicit_paths_out_of_the_records_order_are_refused(tmp_path):
    """ContamX gives the k-th AHS record the k-th path of each implicit flag."""
    lines = _nist(tmp_path).read_text().splitlines(keepends=True)
    first = next(k for k, ln in enumerate(lines) if ln.startswith("  13   16"))
    lines[first:first + 6] = lines[first + 3:first + 6] + lines[first:first + 3]
    swapped = tmp_path / "swapped.prj"
    swapped.write_text("".join(lines))
    with pytest.raises(ValueError, match=r"AHS 1 \(AHS record 1\) names path 13 as its "
                                         r"recirculation path.*path 16"):
        read_prj(swapped)


def test_an_implicit_path_named_by_two_ahs_is_refused(tmp_path):
    """Its own plenum zones, no longer named, get the variable-pressure bit to pass."""
    variant = _nist(tmp_path, [("  2   6   7  16  17  18 -1 ahsTwo",
                                "  2   4   5  13  14  15 -1 ahsTwo"),
                               ("   6 10   0", "   6 11   0"), ("   7 10   0", "   7 11   0")])
    with pytest.raises(ValueError, match=r"AHS 2 \(AHS record 2\) names path 13.*path 16"):
        read_prj(variant)


def test_a_duct_terminal_into_an_ahs_plenum_is_refused(tmp_path):
    variant = _nist(tmp_path, [("   4  0 1   3   0", "   4  0 1   4   0")])
    with pytest.raises(ValueError, match=r"'ahsOne\(Ret\)'.*'junction 4' -> "
                                         r"'ahsOne\(Ret\)' of kind 'duct_terminal'"):
        project_to_model(read_prj(variant))


_RECIRCULATION = "   8   16   4   5   0   1   0   0   1   0   1"
_SUPPLY_TWO = "   6    8   5   2   0   0   0   1   0   0   1"


@pytest.mark.parametrize("old, new, error, message", [
    (_RECIRCULATION, "   8   16   4   5   0   1   0   1   1   0   1", ValueError,
     r"path 8 is an implicit AHS path and also names AHS 1"),
    (_SUPPLY_TWO, "   6    8   5   2   0   0   0   0   0   0   1", ValueError,
     r"path 6 is an AHS supply or return point without an AHS"),
    (_SUPPLY_TWO, "   6    8   5   2   0   0   0   2   0   0   1", KeyError,
     r"path 6 references AHS 2, which the 'simple AHS' section does not define"),
    (_RETURN_PLENUM, _RETURN_PLENUM.replace("   4 10", "   4  8", 1), ValueError,
     r"zone 4 \('ahs\(Ret\)'\) has flags 8, without the variable-contaminants bit"),
])
def test_ahs_path_and_plenum_records_are_checked_by_name(tmp_path, old, new, error,
                                                          message):
    with pytest.raises(error, match=message):
        read_prj(_variant(AHS.read_text(), old, new, tmp_path))


OCCUPANTS = DATA / "occupants_three_zone.prj"
MF_HEAD = "!sim_mf slae rs maxi   relcnvg   abscnvg relax gamma ucc\n    "


def _occ_variant(tmp_path, *pairs):
    text = OCCUPANTS.read_text()
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new, 1)
    out = tmp_path / "variant.prj"
    out.write_text(text)
    return out


def test_occupants_and_their_schedules_are_read():
    p = read_prj(OCCUPANTS)
    assert [o.name for o in p.occupants] == ["occupant 1", "occupant 2"]
    visitor, resident = p.occupants
    assert visitor.description == "visitor"
    cal = Calendar()
    assert [visitor.occupancy.zone_at(t, cal, left=False)
            for t in (0.0, 3750.0, 7200.0, 10800.0)] == ["one", "two", None, "three"]
    assert visitor.occupancy.zone_at(3750.0, cal, left=True) == "one"
    assert visitor.multiplier.item() == 2.0
    assert visitor.generation["sarin"].item() == pytest.approx(1e-6)
    assert visitor.schedules["sarin"].value(3600.0, cal).item() == 0.5
    assert resident.generation == {}
    assert resident.occupancy.zone_at(5000.0, cal, left=False) == "three"
    assert p.building_zones == ["one", "two"]
    keys = [i.key for i in p.inputs if "occupant" in i.key]
    assert keys == ["presence.occupant 1", "generation_multiplier.occupant 1"]


def test_project_to_model_attaches_the_occupants_generation():
    p = read_prj(OCCUPANTS)
    model, state, drivers = project_to_model(p)
    (gen,) = [c for c in model.closures if isinstance(c, OccupantGeneration)]
    assert gen.layer is model.transport["species"]
    assert [o.name for o in gen.occupants] == ["occupant 1"]
    assert drivers["presence.occupant 1"][model.net.node_index("one")].item() == 1.0
    assert drivers["generation_multiplier.occupant 1"].tolist() == [0.0]     # 00:00
    assert model.check(state, drivers).ok


def test_without_contaminants_the_occupants_inputs_are_left_out(tmp_path):
    p = read_prj(_occ_variant(tmp_path, (MF_HEAD + "2", MF_HEAD + "0")))
    assert p.run_control.contaminants == "none"
    model, state, drivers = project_to_model(p)
    assert "species" not in model.transport
    assert not any(isinstance(c, OccupantGeneration) for c in model.closures)
    keys = {i.key for i in p.inputs_for(model)}
    assert not keys & {"presence.occupant 1", "generation_multiplier.occupant 1"}
    assert "presence.occupant 1" not in drivers
    assert model.check(state, drivers).ok


def test_occupant_records_that_name_nothing_are_refused(tmp_path):
    cases = [
        ((" 01:02:30 2 0 0 0", " 01:02:30 9 0 0 0"), KeyError,
         r"occupancy schedule 1.*zone 9"),
        ((" 1 1 1 1 1 1 1 1 1 1 1 1\nsarin", " 3 1 1 1 1 1 1 1 1 1 1 1\nsarin"), KeyError,
         r"exposure 1.*occupancy schedule.*3"),
        (("sarin 1 1e-06 0 0", "CO2 1 1e-06 0 0"), KeyError, r"exposure 1.*'CO2'"),
        (("sarin 1 1e-06 0 0", "sarin 5 1e-06 0 0"), KeyError, r"exposure 1.*schedule 5"),
    ]
    for pair, exc, pattern in cases:
        with pytest.raises(exc, match=pattern):
            read_prj(_occ_variant(tmp_path, pair))


def test_unsupported_occupant_records_are_refused(tmp_path):
    with pytest.raises(ValueError, match=r"occupancy schedule 1.*24:00:00"):
        read_prj(_occ_variant(tmp_path, ("1 5 0 Moves", "1 4 0 Moves"),
                              (" 03:00:00 3 0 0 0\n 24:00:00 3 0 0 0\n",
                               " 03:00:00 3 0 0 0\n")))
    with pytest.raises(ValueError, match=r"exposure 1.*value file"):
        read_prj(_occ_variant(tmp_path, ("sarin 1 1e-06 0 0", "sarin 1 1e-06 0 1 node")))
    with pytest.raises(ValueError, match=r"exposure.*five fields"):
        read_prj(_occ_variant(tmp_path, ("1 1 1 2 0\nvisitor", "1 1 1 2\nvisitor")))
    with pytest.raises(ValueError, match=r"exposure 1.*2 occupancy schedules.*12"):
        read_prj(_occ_variant(tmp_path, (" 1 1 1 1 1 1 1 1 1 1 1 1\nsarin", " 1 1\nsarin")))


def test_an_occupant_that_generates_nothing_ignores_its_generation_lines(tmp_path):
    p = read_prj(_occ_variant(tmp_path, ("1 1 1 2 0\nvisitor", "1 0 1 2 0\nvisitor")))
    assert p.occupants[0].generation == {}
    assert not any("occupant" in i.key for i in p.inputs)
    model, _state, _drivers = project_to_model(p)
    assert not any(isinstance(c, OccupantGeneration) for c in model.closures)


def test_malformed_occupant_lines_are_refused_by_name(tmp_path):
    cases = [
        ((" 01:02:30 2 0 0 0", " 01:02:30"), r"occupancy schedule 1.*01:02:30"),
        ((" 01:02:30 2 0 0 0", " 01:02 2 0 0 0"), r"occupancy schedule 1.*01:02"),
        ((" 01:02:30 2 0 0 0", " 01:02:30 two 0 0 0"), r"occupancy schedule 1.*two"),
        (("sarin 1 1e-06 0 0", "sarin 1"), r"exposure 1.*generation line.*sarin 1"),
        (("sarin 1 1e-06 0 0", "sarin 1 lots 0 0"), r"exposure 1.*generation line.*lots"),
    ]
    for pair, pattern in cases:
        with pytest.raises(ValueError, match=pattern):
            read_prj(_occ_variant(tmp_path, pair))


def test_repeated_or_misnumbered_occupant_records_are_refused(tmp_path):
    with pytest.raises(ValueError, match=r"occupancy schedule 1 .*twice"):
        read_prj(_occ_variant(tmp_path, ("2 2 0 Stays", "1 2 0 Stays")))
    with pytest.raises(ValueError, match=r"exposure 3.*in order.*expected 2"):
        read_prj(_occ_variant(tmp_path, ("2 0 0 1 0\nresident", "3 0 0 1 0\nresident")))
    with pytest.raises(ValueError, match=r"exposure 1.*in order.*expected 2"):
        read_prj(_occ_variant(tmp_path, ("2 0 0 1 0\nresident", "1 0 0 1 0\nresident")))


def test_an_occupant_breathing_where_the_layer_holds_no_air_is_refused_at_build(tmp_path):
    text = AHS.read_text()
    old = "0 ! occupancy schedules:\n-999\n0 ! exposures:\n-999\n"
    assert old in text
    text = text.replace(old, "1 ! occupancy schedules:\n1 2 0 plenum\n\n"
                             " 00:00:00 4 0 0 0\n 24:00:00 4 0 0 0\n-999\n"
                             "1 ! exposures:\n1 0 0 1 0\nmaintenance\n"
                             " 1 1 1 1 1 1 1 1 1 1 1 1\n-999\n")
    variant = tmp_path / "ahs_occupant.prj"
    variant.write_text(text)
    p = read_prj(variant)
    assert p.occupants[0].generation == {}
    with pytest.raises(ValueError, match=r"occupant 'occupant 1'.*breathe"):
        project_to_model(p)


def test_a_steady_uvgi_project_uses_the_transmission_at_its_own_flows():
    p = read_prj(UVGI)
    model, state, drivers = project_to_model(p)
    with pytest.warns(ContamCompatibilityWarning):
        out = prj_steady(model, state, drivers)
    (closure,) = [c for c in model.closures if isinstance(c, FilterTransmission)]
    rho = drivers["P_ref"] / (R_AIR * drivers["T"])
    tau = closure(out, {**drivers, "rho": rho})["species.transmission"]
    layer = model.transport["species"]
    q = model.potential["air"].flows_of_kind(out["air.q"], layer.flow_kinds)
    sources = torch.zeros(model.net.n, 2, dtype=F64)
    expected = layer.steady(q, sources, drivers["species.x_boundary"], transmission=tau)
    torch.testing.assert_close(out["species.x"], expected, rtol=1e-12, atol=0)
    assert bool((tau < 1).any())

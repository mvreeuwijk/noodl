"""CONTAM .prj reader on NIST's sample projects (documented subset, spec section 8 and 14)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tellegen.apps.building.prj import (
    _PATH_FIELDS,
    Project,
    project_to_model,
    read_prj,
)
from tellegen.elements import Damper, FixedFlow, PowerLaw

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
MIXED = DATA / "doorway_damper_fan.prj"
# NIST's own one-zone stack demo. Ruling R30 replaced it with the unfiltered
# `test_OneZoneWthCtmStack-UseApi.prj` for the ContamX parity comparison because it attaches
# a constant-efficiency filter to path 1; it stays in the fixture set as the project that
# pins the refusal.
SS_STACK = DATA / "test_OneZoneSsStack-UseApi.prj"
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
    assert isinstance(el, PowerLaw) and el.kind == "pl_3"
    turb_mass = 0.141421 * math.sqrt(1.2041)          # orifice: turb * sqrt(rho) (spec 14)
    torch.testing.assert_close(el.C, torch.full((4,), turb_mass, dtype=F64))
    torch.testing.assert_close(el.n, torch.full((4,), 0.5, dtype=F64))
    lam, mu, rho = 0.00237882, 1.81625e-5, 1.2041
    assert el.dp_transition == pytest.approx((turb_mass * mu / (lam * rho)) ** (1 / 0.5))


def test_the_reader_builds_a_float64_application():
    """Ruling R7: the default dtype is float32; every file value must arrive as float64."""
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
    # Ruling R21: the brief asked for 1e-9; this Newton solve lands the nodal balance
    # at exactly 0.0, so the assertion is tightened rather than left slack.
    assert model.residuals(ss, drivers)["air"].abs().max().item() < 1e-14
    drivers["species.x_boundary"] = torch.tensor([[0.0023254]], dtype=F64)
    ss2 = model.steady(state, drivers)
    torch.testing.assert_close(ss2["species.x"], torch.full((3, 1), 0.0023254, dtype=F64))


def test_reads_the_hand_written_doorway_damper_fan_project():
    p = read_prj(MIXED)
    kinds = {el.kind: el for el in p.elements}
    assert isinstance(kinds["door_3"], PowerLaw)
    assert p.net.edge_index("door_3").numel() == 2      # two-opening doorway -> two edges
    z = p.net.edge_attr("z_path", "door_3")
    assert (z[1] - z[0]).item() == pytest.approx(2 * 0.4444)
    assert isinstance(kinds["bd_4"], Damper)
    assert kinds["bd_4"].pos.C.item() == pytest.approx(0.02)
    assert kinds["bd_4"].neg.n.item() == pytest.approx(0.65)
    fans = [el for el in p.elements if isinstance(el, FixedFlow)]
    assert len(fans) <= 1


def test_a_doorway_reads_ht_wd_and_cd_from_fields_4_5_and_6():
    """Ruling R9, verified against NIST's own numbers.

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
    """Ruling R5: Task 13 resolves a source's zone through `zone_nr_to_name`, not by position."""
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
    """Ruling R2: `q` is laid out in element-KIND blocks, not in path order.

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
    # And batched, so Task 14 can compare a whole time series at once.
    batched = p.path_flows(q.expand(7, p.net.b))
    assert batched.shape == (7, 6)
    torch.testing.assert_close(batched[0], flows)


def _variant(text: str, old: str, new: str, tmp_path: Path) -> Path:
    assert old in text, old
    out = tmp_path / "variant.prj"
    out.write_text(text.replace(old, new, 1))
    return out


def test_ambient_conditions_come_from_the_steady_simulation_line(tmp_path):
    """Ruling R8: the label picks the block, not its position among the `! Ta` blocks."""
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
    """Ruling R3: profiles are keyed by CONTAM's profile NUMBER and the lookup is strict."""
    text = THREE.read_text()
    path1 = "   1    1  -1   1   3   0   3   0"
    with pytest.raises(KeyError, match=r"profile number 7"):
        read_prj(_variant(text, path1, "   1    1  -1   1   3   0   7   0", tmp_path))


def test_the_filtered_one_zone_stack_project_is_refused_naming_the_filter():
    """Ruling R30, on NIST's own file rather than on a hand-edited variant.

    A filter is invisible to the airflow solve but not to the species layer, so loading this
    project would silently drop a 10% sarin sink on path 1 and return contaminant results
    that are wrong with nothing saying so. This is also the only test that references this
    fixture at all; without it the file sat in `tests/data/contam/` unused.
    """
    with pytest.raises(ValueError, match=r"path 1.*filter 1"):
        read_prj(SS_STACK)


def test_unsupported_records_raise_naming_the_offender(tmp_path):
    text = THREE.read_text()
    path2 = "   2    0   1   2   3   0   0   0   0   0   1"
    with pytest.raises(ValueError, match=r"path 2.*schedule"):
        read_prj(
            _variant(text, path2, "   2    0   1   2   3   0   0   0   1   0   1", tmp_path)
        )
    with pytest.raises(ValueError, match=r"path 2.*filter"):
        read_prj(
            _variant(text, path2, "   2    0   1   2   3   1   0   0   0   0   1", tmp_path)
        )
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
    with pytest.raises(ValueError, match=r"zone 1.*schedule"):
        read_prj(
            _variant(
                text,
                zone1,
                "   1  3   2   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 0 0 0",
                tmp_path,
            )
        )
    with pytest.raises(ValueError, match=r"duct"):
        read_prj(_variant(text, "0 ! duct segments:", "1 ! duct segments:", tmp_path))
    with pytest.raises(ValueError, match=r"element 3.*csf_fsp"):
        read_prj(_variant(text, "3 23 plr_orfc orfcPt01", "3 23 csf_fsp orfcPt01", tmp_path))
    with pytest.raises(ValueError, match=r"ContamW"):
        read_prj(_variant(text, "ContamW 3.4.0.4 0", "SomethingElse 1.0 0", tmp_path))


def test_the_refusal_messages_name_the_section_and_the_record(tmp_path):
    text = THREE.read_text()
    zone1 = "   1  3   0   0   0   1   0.000   300 293.15 0 one -1 0 2 0 0 0 0 0"
    with pytest.raises(ValueError) as exc:
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
    """NIST's samples carry 33 sensor/logger control nodes; skipping them is the whole point."""
    p = read_prj(THREE)
    assert len(p.paths) == 4 and p.zones == ["one", "two", "three"]


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

    ContamW's own header comment omits it, which is where the brief's 27/28 came from.
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

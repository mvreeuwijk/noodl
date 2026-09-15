"""CONTAM .prj reader on NIST's sample projects (documented subset, spec section 8 and 14)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tellegen.apps.building.prj import Project, project_to_model, read_prj
from tellegen.elements import Damper, FixedFlow, PowerLaw

DATA = Path(__file__).resolve().parents[2] / "data" / "contam"
THREE = DATA / "valThreeZonesWthCtm-UseApi.prj"
MIXED = DATA / "doorway_damper_fan.prj"
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


def test_the_round_trip_keeps_levels_zone_numbers_and_sources():
    """Ruling R5: Task 13 resolves a source's zone through `zone_nr_to_name`, not by position."""
    p = read_prj(MIXED)
    assert p.zones == ["singleZone"]
    assert p.zone_nr_to_name == {1: "singleZone"}
    assert p.net.node_attr("z_ref", default=0.0).tolist() == [0.0, 0.0]   # level 1 refHt 0
    # relHt 0, 3, 1, 2, 2.5 on a level whose refHt is 0 -> z_path = refHt + relHt.
    assert sorted(p.net.edge_attr("z_path").tolist()) == pytest.approx(
        [0.0, 0.5556, 1.4444, 2.0, 2.5, 3.0]
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

    `doorway_damper_fan.prj` has four kinds. Network edge order is path order --
    pl_2, pl_2, door_3, door_3, bd_4, fan_1 -- but `PotentialFlowLayer` concatenates
    per element, in `project.elements` order, so the two layouts differ.
    """
    p = read_prj(MIXED)
    offsets, offset = {}, 0
    for el in p.elements:
        offsets[el.kind] = offset
        offset += p.net.edge_index(el.kind).numel()
    assert offset == p.net.b == 6
    expected = {
        1: [offsets["pl_2"] + 0],
        2: [offsets["pl_2"] + 1],
        3: [offsets["door_3"] + 0, offsets["door_3"] + 1],
        4: [offsets["bd_4"] + 0],
        5: [offsets["fan_1"] + 0],
    }
    assert {path.nr: path.edge_columns for path in p.paths} == expected
    # A naive running counter in PATH order would give [0], [1], [2, 3], [4], [5].
    assert [path.edge_columns for path in p.paths] != [[0], [1], [2, 3], [4], [5]]

    q = torch.arange(p.net.b, dtype=F64)
    flows = p.path_flows(q)
    assert flows.tolist() == [float(sum(expected[nr])) if nr == 3 else float(expected[nr][0])
                              for nr in (1, 2, 3, 4, 5)]
    # And batched, so Task 14 can compare a whole time series at once.
    batched = p.path_flows(q.expand(7, p.net.b))
    assert batched.shape == (7, 5)
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

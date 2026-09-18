"""SewerNetwork validation, the builder and the steady driver."""

import pytest
import torch

from tellegen.apps.sewer.network import (
    Manhole,
    Outfall,
    Pipe,
    SewerNetwork,
    build_sewer_model,
    initial_state,
    sewer_steady,
    tree_steady,
)

F64 = torch.float64


def test_the_fixture_tree_is_the_research_network():
    net = tree_steady()
    assert [m.name for m in net.manholes] == ["J1", "J2", "J5", "J3", "J4"]
    assert [p.name for p in net.pipes] == ["C1", "C2", "C4", "C3", "C5"]
    assert [o.name for o in net.outfalls] == ["Outfall"]
    assert [p.diameter for p in net.pipes] == [0.30, 0.30, 0.30, 0.45, 0.45]
    assert [round(p.slope, 6) for p in net.pipes] == [0.01, 0.01, 0.01, 0.005, 0.005]
    net.validate()


def test_a_manhole_with_two_outgoing_pipes_is_refused():
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0), Manhole("J2", 11.0)),
        pipes=(Pipe("A", "J1", "J2", 100, 0.3, 0.013, 0.01),
               Pipe("B", "J1", "Out", 100, 0.3, 0.013, 0.01),
               Pipe("C", "J2", "Out", 100, 0.3, 0.013, 0.01)),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match=r"manhole 'J1' has 2 outgoing pipes"):
        net.validate()


def test_a_cycle_is_refused():
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0), Manhole("J2", 11.0)),
        pipes=(Pipe("A", "J1", "J2", 100, 0.3, 0.013, 0.01),
               Pipe("B", "J2", "J1", 100, 0.3, 0.013, 0.01)),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match="does not reach an outfall"):
        net.validate()


def test_a_component_without_an_outfall_is_refused():
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0), Manhole("J2", 11.0), Manhole("J3", 10.0)),
        pipes=(Pipe("A", "J1", "J2", 100, 0.3, 0.013, 0.01),
               Pipe("B", "J2", "Out", 100, 0.3, 0.013, 0.01),
               Pipe("C", "J3", "J3", 100, 0.3, 0.013, 0.01)),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match=r"'J3'"):
        net.validate()


@pytest.mark.parametrize("slope", [0.0, -0.001])
def test_a_zero_or_adverse_slope_is_refused(slope):
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0),),
        pipes=(Pipe("A", "J1", "Out", 100, 0.3, 0.013, slope),),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match=r"pipe 'A' has slope"):
        net.validate()


def test_a_duplicate_name_is_refused():
    net = SewerNetwork(
        manholes=(Manhole("J1", 12.0), Manhole("J1", 11.0)),
        pipes=(Pipe("A", "J1", "Out", 100, 0.3, 0.013, 0.01),),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(ValueError, match="duplicate node name 'J1'"):
        net.validate()


def test_builder_returns_a_model_a_state_and_drivers():
    model, state, drivers = build_sewer_model(tree_steady())
    assert set(model.potential) == {"air"}
    assert set(model.transport) == {"water_quality", "air_quality"}
    assert model.net.nodes == ["J1", "J2", "J5", "J3", "J4", "Outfall", "ambient"]
    for key in ("air.phi", "air.q", "water_quality.x", "air_quality.x"):
        assert key in state
    for key in ("inflow", "T_water", "T_head", "T_amb", "pH", "bod_in", "sulfide_in",
                "air.phi_boundary", "water_quality.x_boundary", "air_quality.x_boundary",
                "sewer.q_slope"):
        assert key in drivers, key


def test_the_air_layer_carries_every_kind():
    model, _, _ = build_sewer_model(tree_steady(), fans=("J3",))
    assert set(model.potential["air"].kinds) == {"headspace", "leak", "fan"}
    # one headspace edge per pipe, plus the outfall edge
    assert model.net.edge_index("headspace").numel() == 6
    assert model.net.edge_index("leak").numel() == 5
    assert model.net.edge_index("fan").numel() == 1


def test_air_false_builds_the_water_only_model():
    model, state, drivers = build_sewer_model(tree_steady(), air=False)
    assert model.potential == {}
    assert set(model.transport) == {"water_quality"}
    assert "ambient" not in model.net.nodes
    assert "air.phi_boundary" not in drivers


def test_quality_false_drops_both_quality_layers():
    model, _, _ = build_sewer_model(tree_steady(), quality=False)
    assert model.transport == {}


def test_storage_true_registers_the_closure_state_key():
    model, state, _ = build_sewer_model(tree_steady(), storage=True, dt_storage=60.0)
    assert "sewer.H" in model.closure_state_keys
    assert state["sewer.H"].shape == (5,)


def test_a_single_step_runs_and_conserves_the_air_balance():
    model, state, drivers = build_sewer_model(tree_steady())
    new = model.step(state, drivers, 60.0)
    residuals = model.residuals(new, drivers)
    assert float(residuals["air"].abs().max()) < 1e-10


def test_sewer_steady_reaches_a_fixed_point():
    model, state, drivers = build_sewer_model(tree_steady())
    final = sewer_steady(model, state, drivers)
    residuals = model.residuals(final, drivers)
    for name, value in residuals.items():
        assert float(value.abs().max()) < 1e-8, name


def test_initial_state_refuses_an_unknown_quantity():
    model, _, _ = build_sewer_model(tree_steady())
    model.transport["water_quality"].quantity = "nonsense"
    with pytest.raises(ValueError, match="nonsense"):
        initial_state(model)


def test_read_inp_sulfide_source_uses_each_manholes_own_outgoing_pipe():
    """M4-R4/M4-R9 amendment: `tree_steady.inp`'s `[CONDUITS]` order (C1..C5) does not match
    its `[JUNCTIONS]` order (J1, J2, J5, J3, J4), so `read_inp` gives the non-identity
    `out_pipe = [0, 1, 3, 2, 4]`. The model's sulfide source at each manhole must equal
    `sulfide_rate` evaluated on THAT manhole's own outgoing pipe's hydraulics, not on the
    pipe at the same position -- exercised end to end through `read_inp` and
    `build_sewer_model`."""
    from pathlib import Path

    from tellegen.apps.sewer.inp import read_inp
    from tellegen.apps.sewer.quality import sulfide_rate

    data = Path(__file__).resolve().parents[2] / "data" / "sewer"
    net, _, _ = read_inp(data / "tree_steady.inp")
    assert [p.name for p in net.pipes] == ["C1", "C2", "C3", "C4", "C5"]
    assert [m.name for m in net.manholes] == ["J1", "J2", "J5", "J3", "J4"]

    model, state, drivers = build_sewer_model(net, air=False, quality=True)
    assert model.out_pipe.tolist() == [0, 1, 3, 2, 4]

    reaction = dict(model.reactions)["water_quality"]
    resolved = model._apply_closures(state, drivers)

    bod = torch.tensor([1e-3, 2e-3, 3e-3, 4e-3, 5e-3], dtype=F64)
    sulfide = torch.tensor([1e-4, 2e-4, 3e-4, 4e-4, 5e-4], dtype=F64)
    x = torch.stack([bod, sulfide], dim=-1)
    dt = 60.0
    out = reaction.apply(x, dt, resolved)

    radius = resolved["sewer.R_h"].index_select(-1, model.out_pipe)
    velocity = resolved["sewer.v"].index_select(-1, model.out_pipe)
    depth = resolved["sewer.d_m"].index_select(-1, model.out_pipe)
    slope = drivers["sewer.q_slope"]
    d_sulfide = sulfide_rate(
        bod, sulfide, drivers["T_water"], radius, slope, velocity, depth
    )
    expected_sulfide = torch.clamp(sulfide + dt * d_sulfide, min=0.0)
    assert torch.allclose(out[..., 1], expected_sulfide)

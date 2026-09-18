"""The SewerHydraulics closure: tree flow, depths, capacities, storage (rows W1, W7)."""

from dataclasses import dataclass

import pytest
import torch

from tellegen.apps.sewer.hydraulics import CAPACITY_FLOOR, SewerHydraulics
from tellegen.topology import Network

F64 = torch.float64


@dataclass(frozen=True)
class _M:
    name: str
    invert: float
    inflow: float = 0.0


@dataclass(frozen=True)
class _P:
    name: str
    u: str
    v: str
    length: float
    diameter: float
    n: float
    slope: float


MANHOLES = [_M("J1", 12.0, 0.05), _M("J2", 12.0, 0.08), _M("J5", 11.0, 0.03),
            _M("J3", 10.0), _M("J4", 9.0)]
PIPES = [_P("C1", "J1", "J3", 200, 0.30, 0.013, 0.010),
         _P("C2", "J2", "J3", 200, 0.30, 0.013, 0.010),
         _P("C4", "J5", "J4", 200, 0.30, 0.013, 0.010),
         _P("C3", "J3", "J4", 200, 0.45, 0.013, 0.005),
         _P("C5", "J4", "Outfall", 200, 0.45, 0.013, 0.005)]


def _network():
    net = Network(dtype=F64)
    for m in MANHOLES:
        net.add_node(m.name)
    net.add_node("Outfall")
    net.add_node("ambient")
    for p in PIPES:
        net.add_edge(p.u, p.v, kind="pipe", name=p.name)
    return net


def _drivers(inflows=(0.05, 0.08, 0.03, 0.0, 0.0)):
    inflow = torch.zeros(7, dtype=F64)
    for i, value in enumerate(inflows):
        inflow[i] = value
    return {
        "inflow": inflow,
        "T_head": torch.tensor(293.15, dtype=F64),
        "T_amb": torch.tensor(283.15, dtype=F64),
    }


def test_construction_caches_the_level_order():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    assert closure.levels == [[0, 1, 2], [3], [4]]
    assert closure.upstream == {3: [0, 1], 4: [2, 3]}
    assert closure.state_keys == ()


def test_w1_tree_flow_is_the_net_upstream_inflow():
    """Row W1's own side: every pipe carries the sum of the inflows above it, exactly."""
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    out = closure({}, _drivers())
    assert out["sewer.q"].tolist() == pytest.approx(
        [0.05, 0.08, 0.03, 0.13, 0.16], abs=1e-15
    )


def test_depths_and_velocities_match_the_geometry():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    out = closure({}, _drivers())
    assert out["sewer.h"].tolist() == pytest.approx(
        [0.153007001, 0.208101322, 0.114723186, 0.262927897, 0.302698602], abs=1e-9
    )
    assert out["sewer.v"].tolist() == pytest.approx(
        [1.379502251, 1.528845136, 1.206842540, 1.347039431, 1.406247001], abs=1e-8
    )


def test_every_driver_the_conventions_name_is_written():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    out = closure({}, _drivers())
    for key in ("sewer.q", "sewer.h", "sewer.v", "sewer.A_air", "sewer.D_h", "sewer.T",
                "sewer.d_m", "sewer.R_h", "sewer.V_wet", "sewer.V_air",
                "water_quality.q", "water_quality.capacity", "air_quality.capacity",
                "rho_air_nodes"):
        assert key in out, key
        assert torch.isfinite(out[key]).all(), key
    assert out["sewer.q"].shape == (5,)
    assert out["water_quality.capacity"].shape == (5,)
    assert out["rho_air_nodes"].shape == (7,)


def test_capacities_are_the_outgoing_pipes_volumes():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    out = closure({}, _drivers())
    # manhole order J1 J2 J5 J3 J4 -> outgoing pipes C1 C2 C4 C3 C5 (positions 0 1 2 3 4)
    assert out["water_quality.capacity"].tolist() == pytest.approx(
        out["sewer.V_wet"].tolist(), abs=0.0
    )
    assert out["air_quality.capacity"].tolist() == pytest.approx(
        out["sewer.V_air"].tolist(), abs=0.0
    )


def test_air_density_is_full_node_and_ambient_off_the_manholes():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    rho = closure({}, _drivers())["rho_air_nodes"]
    assert rho[:5].tolist() == pytest.approx([1.2040972472143983] * 5, rel=1e-12)
    assert float(rho[5]) == pytest.approx(1.2466223133353378, rel=1e-12)
    assert float(rho[6]) == pytest.approx(1.2466223133353378, rel=1e-12)


def test_a_zero_flow_pipe_gets_the_capacity_floor_and_a_note():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    out = closure({}, _drivers(inflows=(0.0, 0.08, 0.03, 0.0, 0.0)))
    assert float(out["sewer.V_wet"][0]) == CAPACITY_FLOOR
    assert float(out["sewer.v"][0]) == 0.0
    assert "capacity_floor" in closure.notes
    assert "C1" in closure.notes["capacity_floor"]


def test_negative_inflow_is_refused_naming_the_manhole():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    with pytest.raises(ValueError, match=r"non-negative.*\['J2'\]"):
        closure({}, _drivers(inflows=(0.05, -0.01, 0.03, 0.0, 0.0)))


def test_a_wrong_shaped_inflow_is_refused():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    with pytest.raises(ValueError, match="FULL node order"):
        closure({}, {"inflow": torch.zeros(5, dtype=F64),
                     "T_head": torch.tensor(293.15, dtype=F64),
                     "T_amb": torch.tensor(283.15, dtype=F64)})


def test_storage_requires_a_positive_dt():
    with pytest.raises(ValueError, match="requires a positive dt"):
        SewerHydraulics(_network(), PIPES, MANHOLES, storage=True)


def test_w7_storage_reaches_the_quasi_steady_fixed_point():
    """Row W7. Measured by the plan writer: 5.7e-15 relative on the flows and 4.4e-15 on
    the depths after 200 steps of 60 s from a dry start."""
    net = _network()
    steady = SewerHydraulics(net, PIPES, MANHOLES)({}, _drivers())
    closure = SewerHydraulics(net, PIPES, MANHOLES, storage=True, dt=60.0)
    assert closure.state_keys == ("sewer.H",)
    state = {"sewer.H": torch.zeros(5, dtype=F64)}
    out = None
    for _ in range(200):
        out = closure(state, _drivers())
        state = {"sewer.H": out["sewer.H"]}
    q_rel = ((out["sewer.q"] - steady["sewer.q"]).abs() / steady["sewer.q"]).max()
    h_rel = ((out["sewer.h"] - steady["sewer.h"]).abs() / steady["sewer.h"]).max()
    assert float(q_rel) < 1e-9
    assert float(h_rel) < 1e-9


def test_storage_without_its_state_is_refused():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES, storage=True, dt=60.0)
    with pytest.raises(KeyError, match="state 'sewer.H'"):
        closure({}, _drivers())


def test_gradients_flow_through_the_whole_closure():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    inflow = torch.tensor([0.05, 0.08, 0.03, 0.0, 0.0, 0.0, 0.0], dtype=F64,
                          requires_grad=True)
    out = closure({}, {"inflow": inflow, "T_head": torch.tensor(293.15, dtype=F64),
                       "T_amb": torch.tensor(283.15, dtype=F64)})
    out["sewer.h"].sum().backward()
    assert torch.isfinite(inflow.grad).all()
    assert float(inflow.grad[0]) > 0.0
    assert float(inflow.grad[5]) == 0.0

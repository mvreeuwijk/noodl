"""The SewerHydraulics closure: tree flow, depths, capacities, storage (rows W1, W7)."""

from dataclasses import dataclass

import pytest
import torch

from noodl.apps.sewer.hydraulics import CAPACITY_FLOOR, SewerHydraulics
from noodl.model import StepContext
from noodl.topology import Network

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


def test_storage_without_a_declared_dt_is_now_allowed():
    """`storage=True` does not require a constructor `dt` -- the closure follows the
    model's own step interval (a `StepContext`) instead of one fixed at construction. `dt`
    stays an OPTIONAL declaration: when given, a step of a different interval is refused by
    name (`test_a_constructor_dt_that_disagrees_with_the_step_is_refused_by_name`,
    `tests/apps/sewer/test_storage_clock.py`); when omitted, as here, any interval steps."""
    closure = SewerHydraulics(_network(), PIPES, MANHOLES, storage=True)
    assert closure.dt is None
    assert closure.integrates is True


def test_w7_storage_reaches_the_quasi_steady_fixed_point():
    """Row W7. Measured: 5.7e-15 relative on the flows and 4.4e-15 on
    the depths after 200 steps of 60 s from a dry start.

    A direct closure call (bypassing `Model`) passes its own `StepContext` to make
    the closure ADVANCE -- without a `ctx`, `storage=True` evaluates the given state's
    levels as a query instead (see `test_storage_without_its_state_is_refused` below, which
    is unaffected because it never gets that far)."""
    net = _network()
    steady = SewerHydraulics(net, PIPES, MANHOLES)({}, _drivers())
    closure = SewerHydraulics(net, PIPES, MANHOLES, storage=True, dt=60.0)
    assert closure.state_keys == ("sewer.H",)
    ctx = StepContext(dt=60.0)
    state = {"sewer.H": torch.zeros(5, dtype=F64)}
    out = None
    for _ in range(200):
        out = closure(state, _drivers(), ctx)
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


# ------------------------------------------------------------------ vectorised sweep


def test_storage_sweep_is_vectorised_and_bit_identical_to_the_pinned_values():
    """The storage sweep's per-level upstream-inflow assembly is
    one gather (`index_select`) plus one scatter (`index_add`) per level, using the flat
    `level_idx`/`level_src`/`level_tgt` index tensors `__init__` precomputes -- no per-call
    Python loop over manholes. The pinned values below were computed ONCE from a
    per-call nested-loop reference implementation on this fixture, 3 steps of
    dt=60 s from a dry start; the vectorised sweep must reproduce them bit-for-bit."""
    net = _network()
    closure = SewerHydraulics(net, PIPES, MANHOLES, storage=True, dt=60.0)
    ctx = StepContext(dt=60.0)
    state = {"sewer.H": torch.zeros(5, dtype=F64)}
    out = None
    for _ in range(3):
        out = closure(state, _drivers(), ctx)
        state = {"sewer.H": out["sewer.H"]}
    assert out["sewer.q"].tolist() == [
        0.049996631074511494, 0.07999480508352401, 0.029996753614541634,
        0.1299788909045441, 0.15994840443943728,
    ]
    assert out["sewer.H"].tolist() == [
        0.15300087431334175, 0.20809111295985105, 0.11471647324658035,
        0.2629005273481634, 0.30262765439830236,
    ]


def test_non_finite_inflow_is_refused_naming_the_node():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    with pytest.raises(ValueError, match=r"non-finite.*\['J1'\].*instance\(s\) \[0\]"):
        closure({}, _drivers(inflows=(float("nan"), 0.08, 0.03, 0.0, 0.0)))


def test_storage_surcharge_is_refused_naming_the_manhole():
    """IMPORTANT 2: a level's required discharge is checked against the outgoing pipe's
    Manning capacity and refused BY NAME before `solve_monotone` ever sees an unbracketed
    root."""
    closure = SewerHydraulics(_network(), PIPES, MANHOLES, storage=True, dt=60.0)
    state = {"sewer.H": torch.zeros(5, dtype=F64)}
    with pytest.raises(ValueError, match=r"surcharge at manhole\(s\) \['J1'\]"):
        closure(state, _drivers(inflows=(50.0, 0.08, 0.03, 0.0, 0.0)), StepContext(dt=60.0))


def test_storage_surcharge_batched_names_only_the_surcharging_instance():
    """`capacity_flow` carries no batch dimension; the surcharge check must
    broadcast it up to `target`'s batch shape before indexing, or a later batch instance's
    lookup raises an unnamed `IndexError` instead of the named refusal. Instance 0 here is
    ordinary and must not be blamed; only instance 1 surcharges at J1."""
    closure = SewerHydraulics(_network(), PIPES, MANHOLES, storage=True, dt=60.0)
    state = {"sewer.H": torch.zeros(2, 5, dtype=F64)}
    inflow = torch.zeros(2, 7, dtype=F64)
    inflow[0, 0], inflow[0, 1], inflow[0, 2] = 0.05, 0.08, 0.03
    inflow[1, 0], inflow[1, 1], inflow[1, 2] = 50.0, 0.08, 0.03
    drivers = {
        "inflow": inflow,
        "T_head": torch.tensor(293.15, dtype=F64),
        "T_amb": torch.tensor(283.15, dtype=F64),
    }
    with pytest.raises(
        ValueError, match=r"surcharge at manhole\(s\) \['J1'\] instance\(s\) \[1\]"
    ):
        closure(state, drivers, StepContext(dt=60.0))


def test_non_positive_t_head_is_refused_naming_the_driver():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    drivers = _drivers()
    drivers["T_head"] = torch.tensor(-1.0, dtype=F64)
    with pytest.raises(ValueError, match="'T_head'"):
        closure({}, drivers)


def test_non_finite_t_amb_is_refused_naming_the_driver():
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    drivers = _drivers()
    drivers["T_amb"] = torch.tensor(float("nan"), dtype=F64)
    with pytest.raises(ValueError, match="'T_amb'"):
        closure({}, drivers)


def test_storage_gradients_are_finite_through_a_dry_branch():
    """IMPORTANT 4: the storage sweep's implicit-Euler residual is evaluated at h = 0
    exactly for a dry leaf manhole (zero inflow on J5, zero initial `sewer.H`), and
    `solve_monotone`'s backward differentiates that residual with respect to the pipe
    diameter, roughness and slope -- safe only because `geometry.hydraulic_radius`'s
    `** (2/3)` carries a both-branches-safe guard at h = 0."""
    net = _network()
    for target in ("sewer.H", "sewer.q"):
        closure = SewerHydraulics(net, PIPES, MANHOLES, storage=True, dt=60.0)
        closure.diameter = closure.diameter.clone().requires_grad_(True)
        closure.roughness = closure.roughness.clone().requires_grad_(True)
        closure.slope = closure.slope.clone().requires_grad_(True)
        inflow = torch.tensor([0.05, 0.08, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=F64,
                              requires_grad=True)
        state = {"sewer.H": torch.zeros(5, dtype=F64)}
        out = closure(state, {"inflow": inflow, "T_head": torch.tensor(293.15, dtype=F64),
                              "T_amb": torch.tensor(283.15, dtype=F64)}, StepContext(dt=60.0))
        grads = torch.autograd.grad(
            out[target].sum(),
            [inflow, closure.diameter, closure.roughness, closure.slope],
        )
        for g in grads:
            assert torch.isfinite(g).all()


# ------------------------------------------------------------------ nodal drivers


def test_t_head_full_node_batched_and_scalar_all_step_and_agree():
    """`T_head` is "full-node K; scalars broadcast", not scalar-or-nothing.
    A 0-d scalar, a `(n_nodes,)` full-node vector, a `(B, 1)` batched broadcast and a
    `(B, n_nodes)` batched full-node array must all step, and (being the same physical
    temperature everywhere) must all agree with the plain scalar case."""
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    base = _drivers()
    scalar_rho = closure({}, base)["rho_air_nodes"]

    full = dict(base)
    full["T_head"] = torch.full((7,), 293.15, dtype=F64)
    assert closure({}, full)["rho_air_nodes"].tolist() == pytest.approx(
        scalar_rho.tolist(), rel=1e-14
    )

    b1 = dict(base)
    b1["T_head"] = torch.tensor([[293.15], [293.15]], dtype=F64)
    out_b1 = closure({}, b1)["rho_air_nodes"]
    assert out_b1.shape == (2, 7)
    for row in out_b1:
        assert row.tolist() == pytest.approx(scalar_rho.tolist(), rel=1e-14)

    bn = dict(base)
    bn["T_head"] = torch.full((2, 7), 293.15, dtype=F64)
    out_bn = closure({}, bn)["rho_air_nodes"]
    assert out_bn.shape == (2, 7)
    for row in out_bn:
        assert row.tolist() == pytest.approx(scalar_rho.tolist(), rel=1e-14)


def test_t_head_with_a_layout_matching_neither_scalar_nor_full_node_is_refused_by_name():
    """A `(B,)` layout with `B != n_nodes` is neither a scalar, a full-node vector nor
    a trailing singleton, so it is refused by name rather than silently misinterpreted."""
    closure = SewerHydraulics(_network(), PIPES, MANHOLES)
    drivers = _drivers()
    drivers["T_head"] = torch.tensor([1.0, 2.0, 3.0], dtype=F64)
    with pytest.raises(ValueError, match=r"'T_head'.*trailing shape 3"):
        closure({}, drivers)


# ------------------------------------------------------------------ forests


def _forest():
    """Two independent one-pipe trees, each with its own outfall."""
    net = Network(dtype=F64)
    for name in ("A", "B", "OutA", "OutB"):
        net.add_node(name)
    net.add_edge("A", "OutA", kind="pipe", name="PA")
    net.add_edge("B", "OutB", kind="pipe", name="PB")
    return net


def test_a_two_component_forest_steps_and_each_component_keeps_its_own_inflow():
    """`_tree_flow` must not subtract the WHOLE network's lateral total at every
    outfall, which is wrong for more than one component (and raises an unnamed
    `index_add_` error once the outfall count and manhole count disagree with a
    single-`total` broadcast). Each one-pipe component's own pipe must carry exactly its
    own inflow, independent of the other component's."""
    net = _forest()
    manholes = [_M("A", 10.0, 0.05), _M("B", 8.0, 0.03)]
    pipes = [_P("PA", "A", "OutA", 100.0, 0.30, 0.013, 0.01),
             _P("PB", "B", "OutB", 100.0, 0.30, 0.013, 0.01)]
    closure = SewerHydraulics(net, pipes, manholes)
    inflow = torch.zeros(4, dtype=F64)
    inflow[0], inflow[1] = 0.05, 0.03
    out = closure({}, {"inflow": inflow, "T_head": torch.tensor(293.15, dtype=F64),
                       "T_amb": torch.tensor(283.15, dtype=F64)})
    assert out["sewer.q"].tolist() == pytest.approx([0.05, 0.03], abs=1e-15)

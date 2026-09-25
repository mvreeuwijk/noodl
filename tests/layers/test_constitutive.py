"""Tests for ConstitutiveLayer: the loop formulation for general branch laws.

``tests/golden/worked_examples.json`` (``GOLD``) holds a worked example's captured numeric
references for the same triangle (edges (0,1), (1,2), (2,0), kind "pipe") used throughout this
file.
"""

import json
from pathlib import Path

import pytest
import torch

from noodl.layers.constitutive import ConstitutiveLayer
from noodl.topology import Network

_GOLD_PATH = Path(__file__).parent.parent / "golden" / "worked_examples.json"
GOLD = json.loads(_GOLD_PATH.read_text())

F64 = torch.float64


def _triangle_net() -> Network:
    """Nodes 0, 1, 2; edges (0, 1), (1, 2), (2, 0), all kind 'pipe' -- the worked example's
    single mesh."""
    net = Network(dtype=F64)
    net.add_node(0)
    net.add_node(1)
    net.add_node(2)
    net.add_edge(0, 1, kind="pipe")
    net.add_edge(1, 2, kind="pipe")
    net.add_edge(2, 0, kind="pipe")
    return net


def _quadratic_loop_law(p: torch.Tensor, q: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """The worked example's ``loop``: a prescribed drive on edge 0, quadratic dissipation on
    edges 1, 2."""
    a0, a1, a2 = theta[0], theta[1], theta[2]
    return torch.stack(
        [
            p[0] - a0,
            p[1] - a1 * q[1].abs() * q[1],
            p[2] - a2 * q[2].abs() * q[2],
        ]
    )


def _spring_mass_damper_law(p: torch.Tensor, q: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """The worked example's ``spring_mass_damper``: inductor (edge 0), resistor (edge 1),
    capacitor (edge 2), stepped implicitly. ``theta = [L, R, C, q0_prev, p2_prev, dt]``; L
    does not appear explicitly (the worked example's own law does not use it, folding L = 1
    into the dt/L term as dt)."""
    _l, r, _c, q0_prev, p2_prev, dt = theta
    return torch.stack(
        [
            q[0] - q0_prev - dt * p[0],
            p[1] - r * q[1],
            p[2] - p2_prev - dt * q[2],
        ]
    )


# --------------------------------------------------------------------- (a) quadratic loop


def test_quadratic_loop_matches_the_worked_example_p_and_q():
    net = _triangle_net()
    layer = ConstitutiveLayer(net, "loop", kind="pipe", law=_quadratic_loop_law)

    theta = torch.tensor([1.0, 1.0, 1.0], dtype=F64)
    p, q = layer.solve(theta)

    want_p = torch.tensor(GOLD["quadratic_loop_a111"]["p"], dtype=F64)
    want_q = torch.tensor(GOLD["quadratic_loop_a111"]["q"], dtype=F64)
    torch.testing.assert_close(p, want_p, rtol=1e-8, atol=0.0)
    torch.testing.assert_close(q, want_q, rtol=1e-8, atol=0.0)


def test_quadratic_loop_jacobian_matches_the_worked_example_dp_da_dq_da():
    net = _triangle_net()
    layer = ConstitutiveLayer(net, "loop", kind="pipe", law=_quadratic_loop_law)

    def pq(theta: torch.Tensor) -> torch.Tensor:
        p, q = layer.solve(theta)
        return torch.cat([p, q])

    theta = torch.tensor([1.0, 1.0, 1.0], dtype=F64)
    jac = torch.autograd.functional.jacobian(pq, theta)  # (6, 3): rows p0,p1,p2,q0,q1,q2

    want_dp_da = torch.tensor(GOLD["quadratic_loop_a111"]["dp_da"], dtype=F64)
    want_dq_da = torch.tensor(GOLD["quadratic_loop_a111"]["dq_da"], dtype=F64)
    torch.testing.assert_close(jac[:3], want_dp_da, rtol=1e-6, atol=0.0)
    torch.testing.assert_close(jac[3:], want_dq_da, rtol=1e-6, atol=0.0)


# --------------------------------------------------------------- (b) spring-mass-damper


def test_spring_mass_damper_matches_the_worked_example_displacement_series():
    net = _triangle_net()
    layer = ConstitutiveLayer(net, "smd", kind="pipe", law=_spring_mass_damper_law)

    theta = torch.tensor([1.0, 0.2, 1.0, 0.0, 1.0, 0.2], dtype=F64)
    displacement = [1.0]  # p2(t0) / C, prepended (the worked example's own initial xs = [p2/C])
    z0 = None
    for _ in range(50):
        p, q = layer.solve(theta, z0=z0)
        theta = torch.stack([theta[0], theta[1], theta[2], q[0], p[2], theta[5]])
        displacement.append((p[2] / theta[2]).item())

    want = GOLD["spring_mass_damper_displacement"]
    assert len(displacement) == len(want) == 51
    got = torch.tensor(displacement, dtype=F64)
    torch.testing.assert_close(got, torch.tensor(want, dtype=F64), rtol=1e-8, atol=0.0)


# --------------------------------------------------------------------------- (c) refusals


def test_disconnected_network_is_refused_by_name():
    net = Network(dtype=F64)
    for n in (0, 1, 2, 3):
        net.add_node(n)
    net.add_edge(0, 1, kind="pipe")
    net.add_edge(2, 3, kind="pipe")  # a second, disconnected component
    with pytest.raises(ValueError, match="disconnected"):
        ConstitutiveLayer(net, "loop", kind="pipe", law=_quadratic_loop_law)


def test_unknown_kind_is_refused_by_name():
    net = _triangle_net()
    with pytest.raises(ValueError, match="unknown"):
        ConstitutiveLayer(net, "loop", kind="duct", law=_quadratic_loop_law)


def test_wrong_length_law_is_refused_at_first_solve():
    net = _triangle_net()

    def bad_law(p: torch.Tensor, q: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return torch.stack([p[0] - theta[0], p[1] - theta[1]])  # length 2, needs 3

    layer = ConstitutiveLayer(net, "loop", kind="pipe", law=bad_law)
    theta = torch.tensor([1.0, 1.0, 1.0], dtype=F64)
    with pytest.raises(ValueError, match="loop"):
        layer.solve(theta)


def test_model_refuses_a_constitutive_layer_by_name():
    """ConstitutiveLayer is not one of the three layer kinds Model steps -- it is a standalone
    block used directly through solve(), never handed to Model. Model.__init__ refuses any
    layer that is not a PotentialFlowLayer, TransportLayer or CapacitatedTransferLayer, naming
    the layer and its actual type -- which, for a ConstitutiveLayer, names it by name."""
    from noodl.model import Model

    net = _triangle_net()
    layer = ConstitutiveLayer(net, "loop", kind="pipe", law=_quadratic_loop_law)
    with pytest.raises(TypeError, match="ConstitutiveLayer"):
        Model(net, {"loop": layer})


# --------------------------------------------------------------------------- (d) gradcheck


def test_quadratic_loop_gradcheck():
    net = _triangle_net()
    layer = ConstitutiveLayer(net, "loop", kind="pipe", law=_quadratic_loop_law)

    theta = torch.tensor([1.0, 1.0, 1.0], dtype=F64, requires_grad=True)

    def f(th: torch.Tensor) -> torch.Tensor:
        p, q = layer.solve(th)
        return torch.cat([p, q])

    assert torch.autograd.gradcheck(f, (theta,), eps=1e-6, atol=1e-6)

"""Parity tests against captured worked-example golden references.

``tests/golden/worked_examples.json`` holds numeric results for three worked examples -- a
quadratic-drag loop, a spring-mass-damper stepped implicitly, and a three-zone contaminant
exchange -- computed independently of this codebase. Each test here reproduces one of the
examples with today's public layer API and checks the result against the captured reference,
not against a value re-derived from today's code -- that is what makes this a PARITY test
rather than an ordinary regression test.
"""

import json
from pathlib import Path

import torch

from noodl.elements.quadratic import Quadratic
from noodl.layers.constitutive import ConstitutiveLayer
from noodl.layers.potential import PotentialFlowLayer
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

GOLD = json.loads((Path(__file__).parent / "golden" / "worked_examples.json").read_text())

F64 = torch.float64


def test_three_zone_exchange_matches_the_backward_euler_series():
    """Three-zone contaminant exchange: nodes 0 (V=10), 1 (V=1), 2 (V=2), a cycle 0->1->2->0
    carrying constant flow Q=1 with upwind switching, backward Euler in the concentrations
    (dt=0.2), initial concentrations (0, 2, 1). The worked example's implicit relation is
    ``p_new - p_old + dt * q/V = 0`` with q_i the net outflow of node i -- exactly backward
    Euler on ``V dp/dt = inflow - outflow``, which is precisely
    ``TransportLayer(scheme="implicit")``.

    ``TransportLayer`` needs a boundary node (``active_interior`` in
    ``src/noodl/layers/transport.py`` makes any node touched by a flow-kind edge interior
    unless it is named in ``boundary=``, even when the edge carries zero flow), so a fourth
    node ``"amb"`` is added, joined to node 0 by an inert edge of the same flow kind carrying
    q=0, and named as the layer's only boundary node (``x_boundary=[0.0]``, inert since its
    edge carries no flow).

    The JSON series records rooms 1 and 2 (``[p1, p2]``) at each of 50 steps, with row 0 the
    initial condition ``[2.0, 1.0]``.
    """
    net = Network(dtype=F64)
    for n in (0, 1, 2, "amb"):
        net.add_node(n)
    net.add_edge(0, 1, kind="flow")
    net.add_edge(1, 2, kind="flow")
    net.add_edge(2, 0, kind="flow")
    net.add_edge(0, "amb", kind="flow")  # inert boundary edge, q=0

    capacity = torch.tensor([10.0, 1.0, 2.0], dtype=F64)  # V for nodes 0, 1, 2
    layer = TransportLayer(
        net, "species", capacity=capacity, flow_kind="flow", boundary=["amb"],
        scheme="implicit",
    )

    q = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=F64)  # cycle flows, then the inert edge
    sources = torch.zeros(4, dtype=F64)
    x_boundary = torch.tensor([0.0], dtype=F64)
    dt = 0.2

    rows = GOLD["three_zone_exchange_rooms"]
    assert rows[0] == [2.0, 1.0]  # sanity: row 0 is the initial condition

    x = torch.tensor([0.0, 2.0, 1.0], dtype=F64)  # nodes 0, 1, 2
    max_dev = 0.0
    for step, expected in enumerate(rows[1:], start=1):
        x = layer.step(x, q, sources, x_boundary, dt)
        got = torch.tensor([x[1].item(), x[2].item()], dtype=F64)
        want = torch.tensor(expected, dtype=F64)
        if step == 1:
            # If this very first comparison fails, stop rather than loosen tolerances: it
            # would mean a sign/scheme convention difference worth investigating, not a
            # numerical artifact. Print both rows and the rate at the initial state.
            if not torch.allclose(got, want, rtol=1e-10, atol=1e-12):
                rate0 = layer.rate(
                    torch.tensor([0.0, 2.0, 1.0], dtype=F64), q, sources, x_boundary
                )
                raise AssertionError(
                    f"BLOCKED: first-step mismatch. got={got.tolist()} "
                    f"want={want.tolist()} rate(x0)={rate0.tolist()}"
                )
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)
        max_dev = max(max_dev, float((got - want).abs().max()))


def test_quadratic_loop_matches_the_series_resistance_solution():
    """The quadratic-loop worked example: a single mesh with edges (0, 1), (1, 2), (2, 0) and
    branch laws p0 = a0 (a prescribed potential difference), p1 = a1 |q1| q1,
    p2 = a2 |q2| q2. With a = (1, 1, 1) the loop current is q = -0.7071 on every edge
    (``GOLD["quadratic_loop_a111"]``).

    Since a single mesh carries the SAME current through every branch, p1 and p2 combine as
    one series quadratic resistor of coefficient a1 + a2 (their potential drops add:
    (a1 + a2) |q| q = a1 |q1| q1 + a2 |q2| q2 when q1 == q2 == q). This is reproduced here as
    the SERIES-REDUCED form of the worked example -- same physics, one branch -- rather than
    the full three-edge mesh, which ``PotentialFlowLayer`` (a nodal-potential solver, not a
    mesh-current one) would otherwise need a third, purely-topological node for.

    Network: two nodes, both boundary (``n0`` at the prescribed drive a0, ``n1`` at 0), one
    edge n0->n1 of kind "pipe" carrying a single ``Quadratic`` element (see
    ``src/noodl/elements/quadratic.py``, whose law is ``dp = a q + b |q| q``; read
    ``tests/elements/test_quadratic.py`` for the sign/parameter convention). Setting the
    element's own linear coefficient a=0 and quadratic coefficient b = a1 + a2 = 2 makes
    ``dp = b |q| q``, matching the pure-quadratic branch laws exactly; then dp = a0 - 0 = a0,
    so q = sqrt(a0 / b) = sqrt(a0 / 2) in magnitude.

    With both endpoints boundary there is no interior unknown at all (the network is fully
    determined by the two prescribed potentials), so this drives ``layer.dp``/``layer.flows``
    directly instead of ``layer.solve`` -- there is nothing for Newton to solve, and
    ``PotentialFlowLayer.solve``'s own docs note ``phi0`` (and hence ``linear_init``, which
    divides by the element's linear coefficient a and would divide by zero here) is only ever
    a starting guess, never required. Differentiating d|q|/da0 through ``layer.flows`` is
    therefore ordinary reverse-mode autograd on the assembled potential, not the implicit
    adjoint -- exactly the same derivative the worked example's own analytic
    d|q|/da0 = 1 / (2 sqrt(2 a0)) gives.
    """
    net = Network(dtype=F64)
    net.add_node("n0")
    net.add_node("n1")
    net.add_edge("n0", "n1", kind="pipe")

    a1_plus_a2 = 2.0
    element = Quadratic(
        a=torch.tensor(0.0, dtype=F64), b=torch.tensor(a1_plus_a2, dtype=F64), kind="pipe"
    )
    layer = PotentialFlowLayer(net, "loop", [element], boundary=["n0", "n1"])
    assert layer.interior.numel() == 0  # fully determined by the two boundary potentials

    a0 = torch.tensor(1.0, dtype=F64, requires_grad=True)
    assert a0.item() == GOLD["quadratic_loop_a111"]["p"][0]  # p0 = a0, the prescribed drive

    phi_i = torch.zeros(0, dtype=F64)
    phi_b = torch.stack([a0, torch.zeros((), dtype=F64)])
    phi = layer.assemble(phi_i, phi_b)
    q = layer.flows(phi, {})
    assert q.shape == (1,)

    (dq_da0,) = torch.autograd.grad(q.abs().sum(), a0)

    want_q = abs(GOLD["quadratic_loop_a111"]["q"][1])
    want_dq_da0 = abs(GOLD["quadratic_loop_a111"]["dq_da"][1][0])
    torch.testing.assert_close(
        q.abs().item(), want_q, rtol=1e-8, atol=0.0,
    )
    torch.testing.assert_close(
        dq_da0.item(), want_dq_da0, rtol=1e-6, atol=0.0,
    )


def test_spring_mass_damper_matches_the_worked_example():
    """The spring-mass-damper worked example (``GOLD["spring_mass_damper_displacement"]``):
    a single mesh (edges (0, 1), (1, 2), (2, 0), kind "pipe") carrying an inductor, a
    resistor and a capacitor, stepped implicitly (``ConstitutiveLayer``).

    ``theta = [L, R, C, q0_prev, p2_prev, dt]``; the worked example's own branch law does
    not use ``L`` explicitly, folding ``dt / L = dt`` into the inductor term since ``L = 1``
    for this example.
    """
    net = Network(dtype=F64)
    net.add_node(0)
    net.add_node(1)
    net.add_node(2)
    net.add_edge(0, 1, kind="pipe")
    net.add_edge(1, 2, kind="pipe")
    net.add_edge(2, 0, kind="pipe")

    def spring_mass_damper(p, q, theta):
        _l, r, _c, q0_prev, p2_prev, dt = theta
        return torch.stack(
            [
                q[0] - q0_prev - dt * p[0],
                p[1] - r * q[1],
                p[2] - p2_prev - dt * q[2],
            ]
        )

    layer = ConstitutiveLayer(net, "smd", kind="pipe", law=spring_mass_damper)

    theta = torch.tensor([1.0, 0.2, 1.0, 0.0, 1.0, 0.2], dtype=F64)
    displacement = [1.0]  # p2(t0) / C, prepended -- the worked example's own initial xs = [p2/C]
    z0 = None
    for _ in range(50):
        p, q = layer.solve(theta, z0=z0)
        theta = torch.stack([theta[0], theta[1], theta[2], q[0], p[2], theta[5]])
        displacement.append((p[2] / theta[2]).item())

    want = GOLD["spring_mass_damper_displacement"]
    assert len(displacement) == len(want) == 51
    torch.testing.assert_close(
        torch.tensor(displacement, dtype=F64),
        torch.tensor(want, dtype=F64),
        rtol=1e-8,
        atol=0.0,
    )

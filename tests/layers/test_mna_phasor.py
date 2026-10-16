"""Modified Nodal Analysis in KirchhoffLayer: phasor layers (complex potentials and flows
solved in the real view)."""

import math

import pytest
import torch

from noodl.elements import Conductance, ConstitutiveLaw, ImplicitLaw, LawContext, PowerLaw
from noodl.layers._mna import MnaSystem
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.nodesources import NodeSource
from noodl.topology import Network
from tests._circuit_laws import Cap, DrivenSource, Ind, Source

F64 = torch.float64
C128 = torch.complex128


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


GROUND = torch.zeros(1, dtype=F64)


def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


class ConstantPower(ConstitutiveLaw):
    """A constant-power load: q = conj(S / dp), so dp conj(q) = S (not holomorphic in dp)."""

    phasor_capable = True

    def __init__(self, S, kind="load"):
        super().__init__(kind)
        self.S = self._param(S, False)

    def flow(self, dp, drivers=None):
        return torch.conj(self.S / dp)


class Impedance(ImplicitLaw):
    """dp = Z q, a series impedance (complex Z per edge)."""

    phasor_capable = True

    def __init__(self, Z, kind="line"):
        super().__init__(kind)
        self.Z = self._param(Z, False)

    def residual(self, dp, q, drivers, ctx):
        return dp - self.Z * q


class Ratio(ImplicitLaw):
    """Two-port ideal transformer: dp1 = n dp2, q2 = -n q1 (edges [primary, secondary])."""

    ports = 2
    phasor_capable = True

    def __init__(self, n, kind="xfmr"):
        super().__init__(kind)
        self.n = self._param(n, False)

    def residual(self, dp, q, drivers, ctx):
        g = dp.unflatten(-1, (-1, 2))
        h = q.unflatten(-1, (-1, 2))
        r1 = g[..., 0] - self.n * g[..., 1]
        r2 = h[..., 1] + self.n * h[..., 0]
        return torch.stack([r1, r2], -1).flatten(-2)


class LinearWithdrawal(NodeSource):
    """w = c phi at the given nodes."""

    def __init__(self, nodes, c):
        super().__init__(nodes)
        self.c = torch.nn.Parameter(torch.as_tensor(c, dtype=F64), requires_grad=False)

    def flow(self, phi_nodes, drivers=None):
        return self.c * phi_nodes


def _series_rlc(R=2.0, L=1e-3, C=1e-6):
    # source a -> gnd (1 V), resistor a -> b, inductor b -> c, capacitor c -> gnd
    net = _net(["gnd", "a", "b", "c"], [("a", "gnd", "source"), ("a", "b", "r"),
                                        ("b", "c", "ind"), ("c", "gnd", "cap")])
    layer = KirchhoffLayer(
        net, "e", [Source(torch.tensor([1.0])), Conductance(torch.tensor([1.0 / R]), kind="r"),
                   Ind(torch.tensor([L])), Cap(torch.tensor([C]))],
        boundary=["gnd"], phasor=True)
    return net, layer


def test_series_rlc_phasor_current_matches_impedance():
    R, L, C = 2.0, 1e-3, 1e-6
    net, layer = _series_rlc(R, L, C)
    f = torch.tensor([50.0, 5032.92, 1e5])
    phi, q = layer.solve(GROUND, {"e.frequency": f}, differentiable=False)
    assert phi.dtype == C128 and q.dtype == C128
    assert phi.shape == (3, net.n) and q.shape == (3, 4)
    w = 2 * math.pi * f.to(C128)
    expected = 1.0 / (R + 1j * w * L + 1.0 / (1j * w * C))
    current = q[..., layer.kind_slice("r")][..., 0]
    torch.testing.assert_close(current, expected, rtol=1e-12, atol=0)
    # the current is the same through every series element, and leaves a through the source
    torch.testing.assert_close(q[..., layer.kind_slice("cap")][..., 0], expected,
                               rtol=1e-12, atol=0)
    torch.testing.assert_close(q[..., layer.kind_slice("source")][..., 0], -expected,
                               rtol=1e-12, atol=0)


def _constant_power(S, Z):
    # source s -> gnd (1 V), series impedance s -> a, constant-power load a -> gnd
    net = _net(["gnd", "s", "a"], [("s", "gnd", "source"), ("s", "a", "line"),
                                   ("a", "gnd", "load")])
    layer = KirchhoffLayer(
        net, "e", [Source(torch.tensor([1.0])), Impedance(torch.tensor([Z])),
                   ConstantPower(torch.tensor([S]))],
        boundary=["gnd"], phasor=True)
    return net, layer


def test_constant_power_load_is_solved_in_real_view():
    S, Z = 0.5 + 0.2j, 0.01 + 0.05j
    net, layer = _constant_power(S, Z)
    # a constant-power load has no current at dp = 0, so start from the flat 1 V profile
    phi0 = torch.ones(len(layer.interior), dtype=C128)
    phi, q = layer.solve(GROUND, {"e.frequency": torch.tensor(50.0)}, phi0=phi0,
                         differentiable=False, atol=1e-14, rtol=0)
    v = complex(1.0)
    for _ in range(20):
        v = 1.0 - Z * (S / v).conjugate()
    v_a = phi[net.nodes.index("a")]
    assert abs(complex(v_a) - v) < 1e-12 * abs(v)
    # the load draws exactly S
    load = q[layer.kind_slice("load")][0]
    assert abs(complex(v_a * load.conj()) - S) < 1e-12


def test_complex_tellegen():
    _, layer = _series_rlc()
    drivers = {"e.frequency": torch.tensor([50.0, 5032.92, 1e5])}
    phi, q = layer.solve(GROUND, drivers, differentiable=False)
    p = layer.power_residual(phi, q, drivers)
    assert p.is_complex()
    assert float(p.real.abs().max()) < 1e-12 and float(p.imag.abs().max()) < 1e-12
    net, layer = _constant_power(0.5 + 0.2j, 0.01 + 0.05j)
    drivers = {"e.frequency": torch.tensor(50.0)}
    phi, q = layer.solve(GROUND, drivers, phi0=torch.ones(2, dtype=C128),
                         differentiable=False, atol=1e-14, rtol=0)
    p = layer.power_residual(phi, q, drivers)
    assert abs(float(p.real)) < 1e-12 and abs(float(p.imag)) < 1e-12


def test_phasor_refuses_nonlinear_real_law():
    net = _net(["gnd", "a"], [("a", "gnd", "source"), ("a", "gnd", "r")])
    with pytest.raises(ValueError, match=r"'r'.*PowerLaw"):
        KirchhoffLayer(net, "e", [Source(torch.tensor([1.0])),
                                      PowerLaw(torch.tensor([1.0]), 0.5, kind="r")],
                       boundary=["gnd"], phasor=True)


def test_phasor_refuses_node_sources():
    net = _net(["gnd", "a"], [("a", "gnd", "r")])
    with pytest.raises(ValueError, match="edges to ground"):
        KirchhoffLayer(net, "e", [Conductance(torch.tensor([1.0]), kind="r")],
                       node_sources=[LinearWithdrawal([net.nodes.index("a")], [0.1])],
                       boundary=["gnd"], phasor=True)


def test_structure_has_no_node_source_entries_in_real_view():
    # A phasor layer refuses node sources at construction; the pattern must not carry
    # entries for them either (their diagonal is laid out for one real unknown per node).
    net = _net(["gnd", "a"], [("a", "gnd", "r")])
    layer = KirchhoffLayer(net, "e", [Conductance(torch.tensor([1.0]), kind="r")],
                           boundary=["gnd"], phasor=True)
    layer._node_sources = [LinearWithdrawal([net.nodes.index("a")], [0.1])]
    mna = MnaSystem(layer)
    assert torch.equal(mna.row, layer._mna.row) and torch.equal(mna.col, layer._mna.col)
    assert int(mna.src.max()) < mna._node_slot


def test_missing_frequency_driver_named():
    _, layer = _series_rlc()
    with pytest.raises(KeyError, match="e.frequency"):
        layer.solve(GROUND, {}, differentiable=False)


def test_phasor_solve_refuses_steps_and_other_modes():
    _, layer = _series_rlc()
    drivers = {"e.frequency": torch.tensor(50.0)}
    with pytest.raises(ValueError, match="mode='dc'"):
        layer.solve(GROUND, drivers, mode="dc", differentiable=False)
    with pytest.raises(ValueError, match="drop dt and history"):
        layer.solve(GROUND, drivers, dt=0.1, differentiable=False)
    with pytest.raises(ValueError, match="drop dt and history"):
        layer.solve(GROUND, drivers, history=(torch.zeros(4), torch.zeros(4)),
                    differentiable=False)
    # an explicit mode="phasor" is the default
    a, _ = layer.solve(GROUND, drivers, differentiable=False)
    b, _ = layer.solve(GROUND, drivers, mode="phasor", differentiable=False)
    assert torch.equal(a, b)


def test_real_layer_refuses_phasor_mode():
    net = _net(["gnd", "a"], [("a", "gnd", "source"), ("a", "gnd", "r")])
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([1.0])),
                                          Conductance(torch.tensor([1.0]), kind="r")],
                           boundary=["gnd"])
    with pytest.raises(ValueError, match="phasor=True"):
        layer.solve(GROUND, mode="phasor", differentiable=False)


def test_history_that_is_not_a_pair_is_named():
    net = _net(["gnd", "a"], [("a", "gnd", "source"), ("a", "gnd", "cap")])
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([1.0])),
                                          Cap(torch.tensor([1.0]))], boundary=["gnd"])
    with pytest.raises(ValueError, match=r"history must be the pair \(phi, q\)"):
        layer.solve(GROUND, history=torch.zeros(3), dt=0.1, differentiable=False)


def test_complex_warm_start_and_boundary_are_accepted():
    net, layer = _series_rlc()
    drivers = {"e.frequency": torch.tensor(5032.92)}
    phi, q = layer.solve(GROUND, drivers, differentiable=False)
    diagnostics = {}
    phi2, q2 = layer.solve(torch.zeros(1, dtype=C128), drivers, differentiable=False,
                           phi0=phi[layer.interior], q0=q[layer.implicit_edges],
                           diagnostics=diagnostics)
    assert diagnostics["newton_iterations"] == 0
    torch.testing.assert_close(phi2, phi, rtol=0, atol=1e-12)
    torch.testing.assert_close(q2, q, rtol=0, atol=1e-12)


def test_phasor_gradcheck():
    # driven source s -> gnd (complex V), learnable conductance s -> a, capacitor a -> gnd:
    # v_a = V g / (g + j w C)
    net = _net(["gnd", "s", "a"], [("s", "gnd", "source"), ("s", "a", "r"),
                                   ("a", "gnd", "cap")])
    cond = Conductance(torch.tensor([1.0]), kind="r", learnable=True)
    layer = KirchhoffLayer(net, "e", [DrivenSource(), cond, Cap(torch.tensor([0.01]))],
                           boundary=["gnd"], phasor=True)
    a = net.nodes.index("a")
    f = torch.tensor(10.0)

    def magnitude(g, V):
        phi, _ = layer.solve(GROUND, {"V": V, "e.frequency": f})
        return phi[..., a].abs()

    V = torch.tensor([0.8 - 0.3j], dtype=C128, requires_grad=True)
    with torch.no_grad():
        expected = abs(complex(V) * 1.0 / (1.0 + 1j * 2 * math.pi * 10.0 * 0.01))
    assert float(magnitude(cond.g, V).detach()) == pytest.approx(expected, rel=1e-12)
    # gradcheck perturbs the conductance's own learnable parameter in place
    assert torch.autograd.gradcheck(magnitude, (cond.g, V))


def test_real_view_jacobian_matches_autograd_jacobian():
    # A non-holomorphic potential law, a 2-port implicit law, a capacitance, a series
    # impedance and a source: the real-view COO pattern against autograd.
    net = _net(["gnd", "s", "a", "b", "p"],
               [("s", "gnd", "source"), ("s", "a", "line"), ("a", "gnd", "load"),
                ("a", "gnd", "xfmr"), ("b", "gnd", "xfmr"), ("b", "gnd", "cap"),
                ("b", "p", "r"), ("p", "gnd", "r")])
    layer = KirchhoffLayer(
        net, "e", [Source(torch.tensor([1.0])), Impedance(torch.tensor([0.02 + 0.1j])),
                   ConstantPower(torch.tensor([0.4 + 0.3j])), Ratio(torch.tensor([2.0])),
                   Cap(torch.tensor([0.05])), Conductance(torch.tensor([1.5, 0.7]), kind="r")],
        boundary=["gnd"], phasor=True)
    mna = layer._mna
    assert mna.w == 2 and mna.n_I == 4 and mna.n_F == 5 and mna.n == 18
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(mna.n, generator=gen, dtype=F64)
    pb = torch.zeros(1, dtype=C128)
    sources = torch.randn(net.n, generator=gen, dtype=C128)
    ctx = LawContext(mode="phasor", omega=torch.tensor([2 * math.pi * 3.0]))
    call = mna._plain_call

    def residual(z):
        return mna.residual(z, pb, {}, sources, ctx, call)

    expected = torch.autograd.functional.jacobian(residual, x)
    op = mna.operator(x, pb, {}, ctx, call, None)
    J = op.assemble()
    torch.testing.assert_close(J, expected, rtol=1e-13, atol=1e-13)
    # the load block is not of the holomorphic form [[a, -b], [b, a]]
    a = net.nodes.index("a")
    ia = int(layer._interior_of_node[a])
    block = J[2 * ia: 2 * ia + 2, 2 * ia: 2 * ia + 2]
    assert not torch.allclose(block[0, 1], -block[1, 0])
    v = torch.randn(mna.n, generator=gen, dtype=F64)
    torch.testing.assert_close(op.rmatvec(v), expected.T @ v, rtol=1e-13, atol=1e-13)


def test_classic_power_residual_bit_for_bit():
    # A classic nonlinear layer with a drive and sources: the conjugating power_residual
    # equals the old real formula exactly.
    class Push:
        kind = "air"

        def __call__(self, drivers):
            return torch.tensor([0.3, -0.2, 0.1])

    net = _net(["gnd", "a", "b"], [("gnd", "a", "air"), ("a", "b", "air"), ("b", "gnd", "air")])
    layer = KirchhoffLayer(net, "e", [PowerLaw(torch.tensor([1.0, 0.7, 1.3]), 0.6,
                                                   kind="air")],
                           drives=[Push()], boundary=["gnd"])
    sources = torch.tensor([0.0, 0.2, -0.1])
    phi, q = layer.solve(torch.tensor([0.0]), {}, sources, differentiable=False)
    d = layer.dp(phi, {})
    drive_only = d - layer._difference(phi)
    old = ((d * q).sum(-1) - (drive_only * q).sum(-1)
           - (phi[..., layer.bound] * layer.net_outflow(q)[..., layer.bound]).sum(-1)
           - (phi[..., layer.interior] * sources[..., layer.interior]).sum(-1))
    assert torch.equal(layer.power_residual(phi, q, {}, sources), old)


def test_inductor_in_phasor_mode():
    # source a -> gnd, inductor a -> gnd: q = V / (j w L)
    net = _net(["gnd", "a"], [("a", "gnd", "source"), ("a", "gnd", "ind")])
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([2.0])), Ind(torch.tensor([0.5]))],
                           boundary=["gnd"], phasor=True)
    phi, q = layer.solve(GROUND, {"e.frequency": torch.tensor(1.0)}, differentiable=False)
    expected = 2.0 / (1j * 2 * math.pi * 0.5)
    assert abs(complex(q[layer.kind_slice("ind")][0]) - expected) < 1e-12 * abs(expected)

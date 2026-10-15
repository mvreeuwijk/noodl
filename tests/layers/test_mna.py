"""Modified Nodal Analysis in PotentialFlowLayer: static real circuits."""

import pytest
import torch

import noodl.solvers.select as select_module
from noodl.elements import Conductance, Element, FixedFlow, ImplicitLaw, LawContext
from noodl.layers.potential import PotentialFlowLayer
from noodl.nodesources import NodeSource
from noodl.solvers.implicit import TransposeOperator
from noodl.topology import Network
from tests._circuit_laws import DrivenSource, Source

F64 = torch.float64


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


class Ratio(ImplicitLaw):
    """Two-port ideal transformer: dp1 = n dp2, q2 = -n q1 (edges [primary, secondary])."""

    ports = 2

    def __init__(self, n, kind="xfmr", learnable=False):
        super().__init__(kind)
        self.n = self._param(n, learnable)

    def residual(self, dp, q, drivers, ctx):
        g = dp.unflatten(-1, (-1, 2))
        h = q.unflatten(-1, (-1, 2))
        r1 = g[..., 0] - self.n * g[..., 1]
        r2 = h[..., 1] + self.n * h[..., 0]
        return torch.stack([r1, r2], -1).flatten(-2)


class DrivenConductance(Element):
    """q = drivers["g"] dp."""

    def __init__(self, kind="r"):
        super().__init__(kind)

    def flow(self, dp, drivers=None):
        return drivers["g"] * dp


class LinearWithdrawal(NodeSource):
    """w = c phi at the given nodes."""

    def __init__(self, nodes, c):
        super().__init__(nodes)
        self.c = torch.nn.Parameter(torch.as_tensor(c, dtype=F64), requires_grad=False)

    def flow(self, phi_nodes, drivers=None):
        return self.c * phi_nodes


def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


def _at(net, phi, name):
    return phi[..., net.nodes.index(name)]


def _floating_source_net():
    # gnd -R1- a ==V== b -R2- gnd : a source between two NON-ground nodes
    return _net(["gnd", "a", "b"], [("gnd", "a", "r"), ("b", "gnd", "r"), ("a", "b", "source")])


def test_floating_voltage_source_matches_hand_analysis():
    net = _floating_source_net()
    layer = PotentialFlowLayer(net, "e", [Conductance(torch.tensor([1.0, 0.5]), kind="r"),
                                          Source(torch.tensor([3.0]))], boundary=["gnd"])
    assert layer.mna and not layer.dynamic and not layer.phasor
    assert layer.implicit_edges.tolist() == [2]
    phi, q = layer.solve(torch.zeros(1), differentiable=False)
    # Hand analysis (gnd = 0): q_r1 = 1*(0 - a), q_r2 = 0.5*(b - 0), a - b = 3.
    # KCL a: q_s - q_r1 = 0; KCL b: q_r2 - q_s = 0  =>  0.5 b = -a  =>  a = 1, b = -2,
    # and every edge flow is -1 (the current circulates against the edge directions).
    torch.testing.assert_close(_at(net, phi, "a"), torch.tensor(1.0), rtol=0, atol=1e-12)
    torch.testing.assert_close(_at(net, phi, "b"), torch.tensor(-2.0), rtol=0, atol=1e-12)
    torch.testing.assert_close(q, torch.tensor([-1.0, -1.0, -1.0]), rtol=0, atol=1e-12)
    torch.testing.assert_close(layer.power_residual(phi, q, {}), torch.tensor(0.0),
                               atol=1e-12, rtol=0)


def _transformer_layer(n=2.0, learnable=False):
    # source p->gnd (10 V), transformer [p->gnd primary, s->gnd secondary], load s->gnd
    net = _net(["gnd", "p", "s"], [("p", "gnd", "source"), ("p", "gnd", "xfmr"),
                                   ("s", "gnd", "xfmr"), ("s", "gnd", "load")])
    layer = PotentialFlowLayer(
        net, "e",
        [Source(torch.tensor([10.0])), Ratio(torch.tensor(n), learnable=learnable),
         Conductance(torch.tensor([0.25]), kind="load")],
        boundary=["gnd"])
    return net, layer


def test_ideal_transformer_reflects_load():
    net, layer = _transformer_layer()
    phi, q = layer.solve(torch.zeros(1), differentiable=False)
    # v_p = 10; v_p = 2 v_s => v_s = 5; i_load = 0.25 * 5 = 1.25 (s -> gnd);
    # KCL s: q2 + i_load = 0 => q2 = -1.25; q2 = -n q1 => q1 = 0.625 (primary winding);
    # KCL p: q_source + q1 = 0 => q_source = -0.625 (the source delivers 6.25 W).
    torch.testing.assert_close(_at(net, phi, "p"), torch.tensor(10.0), rtol=0, atol=1e-12)
    torch.testing.assert_close(_at(net, phi, "s"), torch.tensor(5.0), rtol=0, atol=1e-12)
    torch.testing.assert_close(q, torch.tensor([-0.625, 0.625, -1.25, 1.25]), rtol=0,
                               atol=1e-12)
    dp = layer.dp(phi, {})
    power_in = -(dp[0] * q[0])
    power_out = dp[3] * q[3]
    torch.testing.assert_close(power_in, torch.tensor(6.25), rtol=0, atol=1e-12)
    torch.testing.assert_close(power_out, power_in, rtol=0, atol=1e-12)
    torch.testing.assert_close(layer.power_residual(phi, q, {}), torch.tensor(0.0),
                               atol=1e-12, rtol=0)


def test_ports_must_divide_edge_count():
    net = _net(["gnd", "a", "b", "c"], [("a", "gnd", "xfmr"), ("b", "gnd", "xfmr"),
                                        ("c", "gnd", "xfmr")])
    with pytest.raises(ValueError, match="multiple of its law's ports=2"):
        PotentialFlowLayer(net, "e", [Ratio(torch.tensor(2.0))], boundary=["gnd"])


def test_floating_island_is_refused_naming_nodes():
    net = _net(["gnd", "p", "s1", "s2"], [("p", "gnd", "source"), ("p", "gnd", "xfmr"),
                                          ("s1", "s2", "xfmr"), ("s1", "s2", "load")])
    with pytest.raises(ValueError, match=r"'s1', 's2'"):
        PotentialFlowLayer(net, "e", [Source(torch.tensor([10.0])), Ratio(torch.tensor(2.0)),
                                      Conductance(torch.tensor([0.25]), kind="load")],
                           boundary=["gnd"])


def test_voltage_source_loop_is_singular_and_named():
    net = _net(["gnd", "a"], [("a", "gnd", "source"), ("a", "gnd", "source"),
                              ("a", "gnd", "r")])
    layer = PotentialFlowLayer(net, "e", [Source(torch.tensor([1.0, 2.0])),
                                          Conductance(torch.tensor([1.0]), kind="r")],
                               boundary=["gnd"])
    with pytest.raises(RuntimeError, match=r"singular.*'source'"):
        layer.solve(torch.zeros(1), differentiable=False)
    with pytest.raises(RuntimeError, match=r"singular.*'source'"):
        layer.solve(torch.zeros(1), differentiable=False, method="direct")


def test_cg_refused_on_mna_layer():
    with pytest.raises(ValueError, match="cg"):
        PotentialFlowLayer(_floating_source_net(), "e",
                           [Conductance(torch.tensor([1.0, 0.5]), kind="r"),
                            Source(torch.tensor([3.0]))],
                           boundary=["gnd"], linear_solver="cg")


def _driven_layer(**kwargs):
    net = _floating_source_net()
    layer = PotentialFlowLayer(net, "e", [DrivenConductance(kind="r"), DrivenSource()],
                               boundary=["gnd"], **kwargs)
    return net, layer


def test_differentiable_matches_finite_difference(monkeypatch):
    _, layer = _driven_layer()
    adjoint_ops = []
    original = select_module._sparse_direct

    def spy(op, b, where, triplet=None):
        adjoint_ops.append(type(op))
        return original(op, b, where, triplet)

    monkeypatch.setattr(select_module, "_sparse_direct", spy)

    def f(g, V, pb):
        phi, q = layer.solve(pb, {"g": g, "V": V})
        return torch.cat([phi, q], -1)

    g = torch.tensor([1.0, 0.5], requires_grad=True)
    V = torch.tensor([3.0], requires_grad=True)
    pb = torch.tensor([0.2], requires_grad=True)
    assert torch.autograd.gradcheck(f, (g, V, pb))
    # The backward solves the transposed MNA system with the method the layer resolved.
    assert TransposeOperator in adjoint_ops


def test_learnable_parameter_gradients_match_central_differences():
    net, layer = _transformer_layer(n=2.0, learnable=True)
    ratio = layer.element_for("xfmr")[0]
    w = torch.tensor([0.3, -1.1, 0.7, 2.0, 0.4, -0.5, 1.3])

    def loss():
        phi, q = layer.solve(torch.tensor([0.1]))
        return (torch.cat([phi, q], -1) * w).sum()

    loss().backward()
    grad = float(ratio.n.grad)
    h = 1e-6
    with torch.no_grad():
        ratio.n += h
        up = float(loss())
        ratio.n -= 2 * h
        down = float(loss())
        ratio.n += h
    assert grad == pytest.approx((up - down) / (2 * h), rel=1e-7)


def test_node_source_parameter_gradient_matches_central_differences():
    net = _net(["gnd", "a", "b", "c"],
               [("a", "gnd", "source"), ("a", "b", "r"), ("c", "gnd", "r"), ("b", "c", "f")])
    source = LinearWithdrawal(torch.tensor([net.nodes.index("b")]), [0.2])
    source.c.requires_grad_(True)
    layer = PotentialFlowLayer(
        net, "e",
        [Source(torch.tensor([2.0])), Conductance(torch.tensor([1.0, 0.5]), kind="r"),
         FixedFlow(torch.tensor([0.3]), kind="f")],
        boundary=["gnd"], node_sources=[source])

    def loss():
        phi, q = layer.solve(torch.zeros(1))
        return phi[net.nodes.index("b")] + 2.0 * q[0]

    loss().backward()
    grad = float(source.c.grad)
    h = 1e-6
    with torch.no_grad():
        source.c += h
        up = float(loss())
        source.c -= 2 * h
        down = float(loss())
        source.c += h
    # KCL b: b - 2 + 0.3 + c b = 0, so b = 1.7 / (1 + c) and db/dc = -1.7 / 1.2^2 at c = 0.2;
    # KCL a: q_source = b - 2, so the loss b + 2 q_source has gradient 3 db/dc.
    assert grad == pytest.approx(-1.7 / 1.2**2 * 3.0, rel=1e-12)
    assert grad == pytest.approx((up - down) / (2 * h), rel=1e-7)


def test_batched_drivers_solve_each_instance():
    net, layer = _driven_layer()
    g = torch.tensor([1.0, 0.5])
    V = torch.tensor([[1.0], [2.0], [3.0], [-4.0]])
    diagnostics = {}
    phi, q = layer.solve(torch.zeros(1), {"g": g, "V": V}, differentiable=False,
                         diagnostics=diagnostics)
    assert phi.shape == (4, net.n)
    assert q.shape == (4, 3)
    assert diagnostics["backend"] in ("sparse_direct", "direct")
    for k in range(4):
        phi_k, q_k = layer.solve(torch.zeros(1), {"g": g, "V": V[k]}, differentiable=False)
        torch.testing.assert_close(phi[k], phi_k, rtol=0, atol=1e-12)
        torch.testing.assert_close(q[k], q_k, rtol=0, atol=1e-12)
    # V = 3 is the hand-analysed case of the first test.
    torch.testing.assert_close(_at(net, phi[2], "a"), torch.tensor(1.0), rtol=0, atol=1e-12)


def test_classic_layer_is_untouched():
    net = _net(["gnd", "a", "b"], [("gnd", "a", "r"), ("a", "b", "r"), ("b", "gnd", "r")])
    layer = PotentialFlowLayer(net, "e", [Conductance(torch.tensor([1.0, 2.0, 0.5]), kind="r")],
                               boundary=["gnd"])
    assert layer.mna is False and layer.dynamic is False
    assert layer.implicit_edges.numel() == 0
    sources = torch.tensor([0.0, 1.0, -0.5])
    phi1, q1 = layer.solve(torch.zeros(1), {}, sources, differentiable=False)
    phi2, q2 = layer.solve(torch.zeros(1), {}, sources, differentiable=False,
                           integration="backward_euler", t=3.0, q0=None, history=None)
    assert torch.equal(phi1, phi2) and torch.equal(q1, q2)


def test_classic_methods_on_mna_layer_raise():
    _, layer = _transformer_layer()
    phi, _ = layer.solve(torch.zeros(1), differentiable=False)
    with pytest.raises(TypeError, match="Modified Nodal Analysis"):
        layer.flows(phi, {})
    with pytest.raises(TypeError, match="Modified Nodal Analysis"):
        layer.dflows(phi, {})
    with pytest.raises(TypeError, match="Modified Nodal Analysis"):
        layer.residual(phi[..., layer.interior], torch.zeros(1), {}, None)
    with pytest.raises(TypeError, match="Modified Nodal Analysis"):
        layer.linear_init(torch.zeros(1), {}, None)


def test_fixed_flow_and_node_source_still_work_in_mna():
    # gnd ==2V== a -r(1)- b -f(0.3)- c -r(0.5)- gnd ; withdrawal 0.2 phi at b and c
    net = _net(["gnd", "a", "b", "c"],
               [("a", "gnd", "source"), ("a", "b", "r"), ("c", "gnd", "r"), ("b", "c", "f")])
    nodes = torch.tensor([net.nodes.index("b"), net.nodes.index("c")])
    layer = PotentialFlowLayer(
        net, "e",
        [Source(torch.tensor([2.0])), Conductance(torch.tensor([1.0, 0.5]), kind="r"),
         FixedFlow(torch.tensor([0.3]), kind="f")],
        boundary=["gnd"], node_sources=[LinearWithdrawal(nodes, [0.2, 0.2])])
    sources = torch.tensor([0.0, 0.0, 0.1, 0.0])
    phi, q = layer.solve(torch.zeros(1), {}, sources, differentiable=False)
    withdrawal = torch.zeros(net.n)
    withdrawal[nodes] = 0.2 * phi[nodes]
    torch.testing.assert_close(layer._accumulate_interior(q),
                               (sources - withdrawal)[layer.interior], rtol=0, atol=1e-12)
    # Hand values: a = 2; KCL b: the inflow a - b equals 0.3 (f) + 0.2 b - 0.1 (source),
    # so b = 1.8 / 1.2 = 1.5; KCL c: 0.5 c + 0.2 c = 0.3 (f inflow), so c = 3/7.
    torch.testing.assert_close(_at(net, phi, "a"), torch.tensor(2.0), rtol=0, atol=1e-12)
    torch.testing.assert_close(_at(net, phi, "b"), torch.tensor(1.5), rtol=0, atol=1e-12)
    torch.testing.assert_close(_at(net, phi, "c"), torch.tensor(3 / 7), rtol=0, atol=1e-12)


# ------------------------------------------------------------------ Jacobian check
class Cubic(Element):
    """q = g dp + c dp^3, per edge."""

    def __init__(self, g, c, kind):
        super().__init__(kind)
        self.g = self._param(g, False)
        self.c = self._param(c, False)

    def flow(self, dp, drivers=None):
        return self.g * dp + self.c * dp**3


class CoupledPair(Element):
    """A two-port potential law: q1 = a dp1 + b dp2^2, q2 = b dp1 dp2 + a dp2."""

    ports = 2

    def __init__(self, a, b, kind):
        super().__init__(kind)
        self.a = self._param(a, False)
        self.b = self._param(b, False)

    def flow(self, dp, drivers=None):
        g = dp.unflatten(-1, (-1, 2))
        q1 = self.a * g[..., 0] + self.b * g[..., 1] ** 2
        q2 = self.b * g[..., 0] * g[..., 1] + self.a * g[..., 1]
        return torch.stack([q1, q2], -1).flatten(-2)


class FlowControlled(ImplicitLaw):
    """dp = R q + k q^3 (a flow-controlled nonlinear resistance)."""

    def __init__(self, R, k, kind):
        super().__init__(kind)
        self.R = self._param(R, False)
        self.k = self._param(k, False)

    def residual(self, dp, q, drivers, ctx):
        return dp - self.R * q - self.k * q**3


class CubicWithdrawal(NodeSource):
    def __init__(self, nodes, c):
        super().__init__(nodes)
        self.c = torch.nn.Parameter(torch.as_tensor(c, dtype=F64), requires_grad=False)

    def flow(self, phi_nodes, drivers=None):
        return self.c * phi_nodes + 0.1 * phi_nodes**3


def _mixed_layer():
    net = _net(["g", "n1", "n2", "n3", "n4", "n5"], [
        ("g", "n1", "r"), ("n1", "n2", "r"), ("n3", "g", "r"), ("n4", "n5", "r"),
        ("n5", "g", "r"),
        ("n2", "n3", "m"), ("n4", "g", "m"),
        ("n1", "n4", "d"),
        ("n2", "g", "x"), ("n5", "n3", "x"),
        ("n3", "n4", "f"),
    ])
    nodes = torch.tensor([net.nodes.index("n2"), net.nodes.index("n5")])
    layer = PotentialFlowLayer(
        net, "e",
        [Cubic(torch.tensor([1.0, 0.7, 1.3, 0.4, 2.0]), torch.tensor(0.2), kind="r"),
         CoupledPair(torch.tensor(0.9), torch.tensor(0.3), kind="m"),
         FlowControlled(torch.tensor([2.0]), torch.tensor([0.5]), kind="d"),
         Ratio(torch.tensor(1.7), kind="x"),
         FixedFlow(torch.tensor([0.25]), kind="f")],
        boundary=["g"], node_sources=[CubicWithdrawal(nodes, [0.3, 0.6])])
    return net, layer


def test_assembled_jacobian_matches_autograd_jacobian_of_the_residual():
    net, layer = _mixed_layer()
    mna = layer._mna
    assert mna.n_I == 5 and mna.n_F == 3 and mna.n == 8
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(mna.n, generator=gen, dtype=F64)
    pb = torch.tensor([0.4])
    sources = torch.randn(net.n, generator=gen, dtype=F64)
    ctx = LawContext(mode="dc")
    call = mna._plain_call

    def residual(z):
        return mna.residual(z, pb, {}, sources, ctx, call)

    expected = torch.autograd.functional.jacobian(residual, x)
    op = mna.operator(x, pb, {}, ctx, call, lambda phi: mna._node_slopes(phi, {}, call))
    torch.testing.assert_close(op.assemble(), expected, rtol=1e-13, atol=1e-13)
    # rmatvec is the transpose action
    v = torch.randn(mna.n, generator=gen, dtype=F64)
    torch.testing.assert_close(op.rmatvec(v), expected.T @ v, rtol=1e-13, atol=1e-13)


def test_mixed_nonlinear_network_converges_with_zero_residual():
    net, layer = _mixed_layer()
    pb = torch.tensor([0.4])
    sources = torch.tensor([0.0, 0.1, -0.2, 0.05, 0.0, 0.3])
    for method in ("sparse_direct", "direct", "gmres"):
        diagnostics = {}
        phi, q = layer.solve(pb, {}, sources, differentiable=False, method=method,
                             diagnostics=diagnostics, atol=1e-13, rtol=0)
        assert bool(diagnostics["converged"])
        mna = layer._mna
        x = torch.cat([phi[layer.interior], q[mna.f_edges]])
        r = mna.residual(x, pb, {}, sources, LawContext(mode="dc"), mna._plain_call)
        assert float(r.abs().max()) < 1e-12
        # Tellegen with the node sources: their withdrawal w leaves the balance as
        # -sum(phi_I w), the power the withdrawals take out of the network.
        withdrawn = torch.zeros(net.n)
        ns = layer.node_sources[0]
        withdrawn[ns.nodes] = ns.flow(phi[ns.nodes]) * phi[ns.nodes]
        torch.testing.assert_close(layer.power_residual(phi, q, {}, sources),
                                   -withdrawn.sum(), rtol=0, atol=1e-12)


def test_warm_start_is_used():
    _, layer = _transformer_layer()
    phi, q = layer.solve(torch.zeros(1), differentiable=False)
    diagnostics = {}
    phi2, q2 = layer.solve(torch.zeros(1), differentiable=False, diagnostics=diagnostics,
                           phi0=phi[layer.interior], q0=q[layer.implicit_edges])
    assert diagnostics["newton_iterations"] == 0
    torch.testing.assert_close(phi2, phi, rtol=0, atol=1e-12)
    torch.testing.assert_close(q2, q, rtol=0, atol=1e-12)


def test_on_failure_return_requires_diagnostics():
    _, layer = _transformer_layer()
    with pytest.raises(ValueError, match="diagnostics"):
        layer.solve(torch.zeros(1), differentiable=False, on_failure="return")


def test_dynamic_kind_may_not_carry_a_drive():
    class Store(Source):
        dynamic = True

    class Push:
        kind = "source"

        def __call__(self, drivers):
            return torch.tensor([1.0])

    net = _floating_source_net()
    with pytest.raises(ValueError, match="drive"):
        PotentialFlowLayer(net, "e", [Conductance(torch.tensor([1.0, 0.5]), kind="r"),
                                      Store(torch.tensor([3.0]))],
                           drives=[Push()], boundary=["gnd"])


def test_auto_factorises_sparse_whenever_scipy_is_importable(monkeypatch):
    import noodl.layers._mna as mna_module

    _, layer = _driven_layer()
    mna = layer._mna
    monkeypatch.setattr(mna_module, "_scipy_importable", lambda: True)
    assert mna.method() == "sparse_direct"
    assert mna.method("direct") == "direct"
    monkeypatch.setattr(mna_module, "_scipy_importable", lambda: False)
    assert mna.method() == "direct"
    assert mna.method("sparse_direct") == "sparse_direct"


def test_auto_runs_sparse_lu_on_a_batch_above_32():
    _, layer = _driven_layer()
    g = torch.tensor([1.0, 0.5]).expand(40, 2)
    diagnostics = {}
    phi, _ = layer.solve(torch.zeros(40, 1), {"g": g, "V": torch.tensor([3.0])},
                         diagnostics=diagnostics, differentiable=False)
    assert diagnostics["method"] == "auto" and diagnostics["backend"] == "sparse_direct"
    reference, _ = layer.solve(torch.zeros(1), {"g": g[0], "V": torch.tensor([3.0])},
                               differentiable=False)
    torch.testing.assert_close(phi, reference.expand(40, -1), rtol=1e-12, atol=1e-12)


def test_unknown_law_form_is_refused_naming_the_kind():
    class Odd(Conductance):
        law_form = "volume"

    with pytest.raises(ValueError, match=r"kind 'r'.*law_form='volume'.*'potential' or "
                                         r"'implicit'"):
        PotentialFlowLayer(_floating_source_net(), "e",
                           [Odd(torch.tensor([1.0, 0.5]), kind="r"),
                            Source(torch.tensor([3.0]))], boundary=["gnd"])


@pytest.mark.parametrize("kwargs, match", [
    ({"mode": "phasor"}, "mode='phasor'.*phasor=True"),
    ({"history": (torch.zeros(3), torch.zeros(2))}, "history"),
    ({"dt": 0.1}, "dt"),
])
def test_classic_layer_refuses_step_and_phasor_keywords(kwargs, match):
    net = _net(["gnd", "a", "b"], [("gnd", "a", "r"), ("a", "b", "r")])
    layer = PotentialFlowLayer(net, "e", [Conductance(torch.tensor([1.0, 2.0]), kind="r")],
                               boundary=["gnd", "b"])
    assert not layer.mna
    with pytest.raises(ValueError, match=rf"PotentialFlowLayer 'e'.*{match}"):
        layer.solve(torch.tensor([1.0, 0.0]), **kwargs)


def test_classic_layer_still_accepts_the_steady_keywords():
    net = _net(["gnd", "a", "b"], [("gnd", "a", "r"), ("a", "b", "r")])
    layer = PotentialFlowLayer(net, "e", [Conductance(torch.tensor([1.0, 2.0]), kind="r")],
                               boundary=["gnd", "b"])
    pb = torch.tensor([1.0, 0.0])
    phi, q = layer.solve(pb)
    for kwargs in ({"mode": "dc"}, {"mode": "initial"}, {"t": 3.0}, {"q0": torch.zeros(0)},
                   {"integration": "backward_euler"}):
        phi_k, q_k = layer.solve(pb, **kwargs)
        torch.testing.assert_close(phi_k, phi, rtol=0, atol=0)
        torch.testing.assert_close(q_k, q, rtol=0, atol=0)


_MIXED_SOURCES = (0.0, 0.1, -0.2, 0.05, 0.0, 0.3)


@pytest.mark.parametrize("differentiable", [False, True])
def test_return_primal_hands_back_the_last_iterate_where_nothing_is_differentiated(
        differentiable):
    _, layer = _mixed_layer()
    diagnostics = {}
    with torch.no_grad():
        phi, q = layer.solve(torch.tensor([0.4]), {}, torch.tensor(_MIXED_SOURCES),
                             differentiable=differentiable, diagnostics=diagnostics,
                             max_iter=1, atol=1e-13, rtol=0, on_failure="return_primal")
    assert not bool(diagnostics["converged"])
    assert bool(torch.isfinite(phi).all()) and bool(torch.isfinite(q).all())


def test_return_primal_still_raises_where_the_solve_is_differentiated():
    _, layer = _mixed_layer()
    sources = torch.tensor(_MIXED_SOURCES, requires_grad=True)
    with pytest.raises(RuntimeError, match="converge"):
        layer.solve(torch.tensor([0.4]), {}, sources, diagnostics={}, max_iter=1,
                    atol=1e-13, rtol=0, on_failure="return_primal")


def test_newton_defaults_reach_an_mna_solve():
    _, layer = _mixed_layer()
    layer.newton_defaults = {"max_iter": 1, "on_failure": "return"}
    diagnostics = {}
    layer.solve(torch.tensor([0.4]), {}, torch.tensor(_MIXED_SOURCES), differentiable=False,
                diagnostics=diagnostics, atol=1e-13, rtol=0)
    assert diagnostics["newton_iterations"] == 1
    assert not bool(diagnostics["converged"])


def test_return_primal_requires_diagnostics_on_an_mna_solve():
    _, layer = _mixed_layer()
    sources = torch.tensor(_MIXED_SOURCES, requires_grad=True)
    with pytest.raises(ValueError, match="diagnostics="):
        layer.solve(torch.tensor([0.4]), {}, sources, on_failure="return_primal")

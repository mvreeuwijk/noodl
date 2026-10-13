"""Modified Nodal Analysis in PotentialFlowLayer: dynamic laws stepped by companion forms."""

import math

import pytest
import torch

import noodl.layers._mna as mna_module
from noodl.elements import Conductance, Element
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network
from tests._circuit_laws import Cap, DrivenSource, Ind, Source

F64 = torch.float64


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


GROUND = torch.zeros(1, dtype=F64)


def _rc(V=1.0, R=1.0, C=1.0):
    # source a -> gnd (dp = phi_a = V), resistor a -> b, capacitor b -> gnd
    net = _net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "r"),
                                   ("b", "gnd", "cap")])
    layer = PotentialFlowLayer(
        net, "e", [Source(torch.tensor([V])), Conductance(torch.tensor([1.0 / R]), kind="r"),
                   Cap(torch.tensor([C]))], boundary=["gnd"])
    return net, layer


def _march(layer, phi, q, dt, steps, integration):
    for _ in range(steps):
        phi, q = layer.solve(GROUND, history=(phi, q), dt=dt, integration=integration,
                             differentiable=False)
    return phi, q


def _order(errors):
    return [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]


def test_rc_charge_backward_euler_is_exact_recurrence():
    net, layer = _rc()
    b = net.nodes.index("b")
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    torch.testing.assert_close(phi[b], torch.tensor(0.0), rtol=0, atol=1e-13)
    torch.testing.assert_close(q[2], torch.tensor(1.0), rtol=0, atol=1e-13)
    dt, v = 0.1, 0.0
    for _ in range(10):
        phi, q = _march(layer, phi, q, dt, 1, "backward_euler")
        v = (v + dt) / (1.0 + dt)          # R C = 1
        torch.testing.assert_close(phi[b], torch.tensor(v), rtol=0, atol=1e-13)
    # the capacitor current is the resistor current (KCL at b)
    torch.testing.assert_close(q[2], q[1], rtol=0, atol=1e-13)


def test_rc_trapezoidal_is_second_order():
    net, layer = _rc()
    b = net.nodes.index("b")
    errors = []
    for dt in (0.1, 0.05, 0.025):
        phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
        phi, q = _march(layer, phi, q, dt, round(1.0 / dt), "trapezoidal")
        errors.append(abs(phi[b].item() - (1.0 - math.exp(-1.0))))
    for p in _order(errors):
        assert abs(p - 2.0) < 0.1


def test_rlc_ringing_trapezoidal_second_order():
    R, L, C = 0.2, 1.0, 1.0
    net = _net(["gnd", "a", "b", "c"], [("a", "gnd", "source"), ("a", "b", "r"),
                                        ("b", "c", "ind"), ("c", "gnd", "cap")])
    layer = PotentialFlowLayer(
        net, "e", [Source(torch.tensor([1.0])), Conductance(torch.tensor([1.0 / R]), kind="r"),
                   Ind(torch.tensor([L])), Cap(torch.tensor([C]))], boundary=["gnd"])
    c = net.nodes.index("c")
    alpha = R / (2.0 * L)
    wd = math.sqrt(1.0 / (L * C) - alpha**2)
    t_end = 2.0
    exact = 1.0 - math.exp(-alpha * t_end) * (math.cos(wd * t_end)
                                              + alpha / wd * math.sin(wd * t_end))
    errors = []
    for dt in (0.1, 0.05, 0.025):
        phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
        phi, q = _march(layer, phi, q, dt, round(t_end / dt), "trapezoidal")
        errors.append(abs(phi[c].item() - exact))
    for p in _order(errors):
        assert abs(p - 2.0) < 0.1


def test_rl_backward_euler_is_exact_recurrence():
    # source a -> gnd (1 V), resistor a -> b (R = 1), inductor b -> gnd (L = 1)
    net = _net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "r"),
                                   ("b", "gnd", "ind")])
    layer = PotentialFlowLayer(
        net, "e", [Source(torch.tensor([1.0])), Conductance(torch.tensor([1.0]), kind="r"),
                   Ind(torch.tensor([1.0]))], boundary=["gnd"])
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    torch.testing.assert_close(q[2], torch.tensor(0.0), rtol=0, atol=1e-13)
    dt, i = 0.1, 0.0
    for _ in range(10):
        phi, q = _march(layer, phi, q, dt, 1, "backward_euler")
        i = (i + dt) / (1.0 + dt)          # L / R = 1
        torch.testing.assert_close(q[2], torch.tensor(i), rtol=0, atol=1e-13)


def test_dc_mode_opens_capacitor_and_shorts_inductor():
    # source a -> gnd (2 V), R1 a -> b, inductor b -> c, R2 c -> gnd, capacitor c -> gnd
    net = _net(["gnd", "a", "b", "c"], [("a", "gnd", "source"), ("a", "b", "r"),
                                        ("b", "c", "ind"), ("c", "gnd", "r"),
                                        ("c", "gnd", "cap")])
    layer = PotentialFlowLayer(
        net, "e", [Source(torch.tensor([2.0])), Conductance(torch.tensor([1.0, 1.0]), kind="r"),
                   Ind(torch.tensor([0.5])), Cap(torch.tensor([3.0]))], boundary=["gnd"])
    assert layer.dynamic
    phi, q = layer.solve(GROUND, differentiable=False)
    for node, value in (("a", 2.0), ("b", 1.0), ("c", 1.0)):
        torch.testing.assert_close(phi[net.nodes.index(node)], torch.tensor(value), rtol=0,
                                   atol=1e-13)
    dp = layer.dp(phi, {})
    s_ind, _ = layer._kind_slices["ind"]
    s_cap, _ = layer._kind_slices["cap"]
    torch.testing.assert_close(dp[s_ind], torch.tensor(0.0), rtol=0, atol=1e-13)
    torch.testing.assert_close(q[s_ind], torch.tensor(1.0), rtol=0, atol=1e-13)
    torch.testing.assert_close(q[s_cap], torch.tensor(0.0), rtol=0, atol=1e-13)
    # mode="dc" explicitly is the same operating point
    phi_dc, q_dc = layer.solve(GROUND, mode="dc", differentiable=False)
    assert torch.equal(phi_dc, phi) and torch.equal(q_dc, q)


def test_transient_without_history_is_refused():
    _, layer = _rc()
    with pytest.raises(ValueError, match=r"PotentialFlowLayer 'e': a transient step needs "
                                         r"history=\(phi, q\) at the step start"):
        layer.solve(GROUND, dt=0.1, differentiable=False)


def test_initial_mode_with_dt_is_refused():
    _, layer = _rc()
    with pytest.raises(ValueError, match="initial"):
        layer.solve(GROUND, mode="initial", dt=0.1, differentiable=False)


def test_unknown_mode_step_and_integration_are_refused():
    _, layer = _rc()
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    with pytest.raises(ValueError, match="mode='transient'"):
        layer.solve(GROUND, mode="transient", dt=0.1, history=(phi, q))
    with pytest.raises(ValueError, match="dt must be positive"):
        layer.solve(GROUND, dt=0.0, history=(phi, q))
    with pytest.raises(ValueError, match="integration='euler'"):
        layer.solve(GROUND, dt=0.1, history=(phi, q), integration="euler")
    with pytest.raises(ValueError, match="history=.*every edge's flow"):
        layer.solve(GROUND, dt=0.1, history=(phi, q[:2]))


def test_drive_on_dynamic_kind_is_refused():
    class Push:
        kind = "cap"

        def __call__(self, drivers):
            return torch.tensor([1.0])

    net = _net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "r"),
                                   ("b", "gnd", "cap")])
    with pytest.raises(ValueError, match="kind 'cap' has a dynamic law"):
        PotentialFlowLayer(net, "e", [Source(torch.tensor([1.0])),
                                      Conductance(torch.tensor([1.0]), kind="r"),
                                      Cap(torch.tensor([1.0]))],
                           drives=[Push()], boundary=["gnd"])


class StepRecorder(Element):
    """q = g dp for a static law that reads the context: records each mode it sees."""

    contextual = True

    def __init__(self, kind="r"):
        super().__init__(kind)
        self.seen = []

    def flow(self, dp, drivers=None, ctx=None):
        self.seen.append((ctx.mode, ctx.dp_prev is None))
        return dp


def test_static_contextual_layer_steps_without_history():
    # A layer with no dynamic law needs no history for a step; its laws see the step.
    net = _net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "r"),
                                   ("b", "gnd", "r")])
    recorder = StepRecorder()
    layer = PotentialFlowLayer(net, "e", [Source(torch.tensor([1.0])), recorder],
                               boundary=["gnd"])
    assert not layer.dynamic
    phi, _ = layer.solve(GROUND, dt=0.1, t=0.1, differentiable=False)
    torch.testing.assert_close(phi[net.nodes.index("b")], torch.tensor(0.5), rtol=0,
                               atol=1e-13)
    assert set(recorder.seen) == {("transient", True)}


def test_gradient_through_three_steps():
    # driven source a -> gnd, resistor a -> b, learnable capacitor b -> gnd
    net = _net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "r"),
                                   ("b", "gnd", "cap")])
    cap = Cap(torch.tensor([0.7]), learnable=True)
    layer = PotentialFlowLayer(
        net, "e", [DrivenSource(), Conductance(torch.tensor([2.0]), kind="r"), cap],
        boundary=["gnd"])
    b = net.nodes.index("b")

    def v_after_three_steps(C, V):
        drivers = {"V": V}
        phi, q = layer.solve(GROUND, drivers, mode="initial")
        for k in range(3):
            phi, q = layer.solve(GROUND, drivers, history=(phi, q), dt=0.1, t=0.1 * (k + 1))
        return phi[..., b]

    V = torch.tensor([1.3], requires_grad=True)
    # gradcheck perturbs the capacitor's own learnable parameter in place
    assert torch.autograd.gradcheck(v_after_three_steps, (cap.C, V))


def _voltage_loop():
    net = _net(["gnd", "a"], [("a", "gnd", "source"), ("a", "gnd", "source"),
                              ("a", "gnd", "r")])
    return PotentialFlowLayer(net, "e", [Source(torch.tensor([1.0, 2.0])),
                                         Conductance(torch.tensor([1.0]), kind="r")],
                              boundary=["gnd"])


def test_singular_message_names_flow_prescribed_parts():
    with pytest.raises(RuntimeError, match="only through flow-prescribed edges"):
        _voltage_loop().solve(GROUND, differentiable=False)


def test_rank_test_is_skipped_above_the_size_cap(monkeypatch):
    monkeypatch.setattr(mna_module, "_RANK_TEST_MAX_UNKNOWNS", 2)
    layer = _voltage_loop()          # 3 unknowns: phi_a and two source flows
    with pytest.raises(RuntimeError, match="failed to converge") as info:
        layer.solve(GROUND, differentiable=False)
    assert "singular" not in str(info.value)


def test_rank_test_runs_once_on_the_first_failing_instance(monkeypatch):
    # gnd -g- a ==V== b -g- gnd with g = 0 in the second instance only: there a and b are
    # tied to ground through the source alone, and their common level is undetermined.
    class DrivenConductance(Element):
        def flow(self, dp, drivers=None):
            return drivers["g"] * dp

    net = _net(["gnd", "a", "b"], [("gnd", "a", "r"), ("b", "gnd", "r"), ("a", "b", "source")])
    layer = PotentialFlowLayer(net, "e", [DrivenConductance("r"), Source(torch.tensor([3.0]))],
                               boundary=["gnd"])
    shapes = []
    original = torch.linalg.matrix_rank

    def spy(A, *args, **kwargs):
        shapes.append(tuple(A.shape))
        return original(A, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "matrix_rank", spy)
    g = torch.tensor([[1.0, 0.5], [0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(RuntimeError, match="singular"):
        layer.solve(torch.zeros(3, 1), {"g": g}, differentiable=False)
    assert shapes == [(3, 3)]


@pytest.mark.parametrize("differentiable", [False, True])
def test_diagnostics_report_the_requested_method(differentiable):
    _, layer = _rc()
    diagnostics = {}
    layer.solve(GROUND, diagnostics=diagnostics, differentiable=differentiable)
    assert diagnostics["method"] == "auto"
    assert diagnostics["backend"] == "sparse_direct"
    diagnostics = {}
    layer.solve(GROUND, diagnostics=diagnostics, differentiable=differentiable,
                method="direct")
    assert diagnostics["method"] == diagnostics["backend"] == "direct"


def test_cg_per_call_is_refused():
    _, layer = _rc()
    with pytest.raises(ValueError, match="PotentialFlowLayer 'e': method='cg'"):
        layer.solve(GROUND, method="cg", differentiable=False)

"""Modified Nodal Analysis in KirchhoffLayer: dynamic laws stepped by companion forms."""

import json
import math
from pathlib import Path

import pytest
import torch

import noodl.layers._mna as mna_module
from noodl.elements import Conductance, ConstitutiveLaw, ImplicitLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network
from tests._circuit_laws import Cap, DrivenSource, Ind, Source

F64 = torch.float64
GOLD = json.loads(
    (Path(__file__).parent.parent / "golden" / "worked_examples.json").read_text())


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
    layer = KirchhoffLayer(
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
    layer = KirchhoffLayer(
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
    layer = KirchhoffLayer(
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
    layer = KirchhoffLayer(
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
    with pytest.raises(ValueError, match=r"KirchhoffLayer 'e': a transient step needs "
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
        KirchhoffLayer(net, "e", [Source(torch.tensor([1.0])),
                                      Conductance(torch.tensor([1.0]), kind="r"),
                                      Cap(torch.tensor([1.0]))],
                       drives=[Push()], boundary=["gnd"])


class StepRecorder(ConstitutiveLaw):
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
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([1.0])), recorder],
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
    layer = KirchhoffLayer(
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
    return KirchhoffLayer(net, "e", [Source(torch.tensor([1.0, 2.0])),
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
    class DrivenConductance(ConstitutiveLaw):
        def flow(self, dp, drivers=None):
            return drivers["g"] * dp

    net = _net(["gnd", "a", "b"], [("gnd", "a", "r"), ("b", "gnd", "r"), ("a", "b", "source")])
    layer = KirchhoffLayer(net, "e", [DrivenConductance("r"), Source(torch.tensor([3.0]))],
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
    with pytest.raises(ValueError, match="KirchhoffLayer 'e': method='cg'"):
        layer.solve(GROUND, method="cg", differentiable=False)


# ------------------------------------------------- loop laws ported from the worked examples
#
# Both cases are a single mesh 0 -> 1 -> 2 -> 0 with node 0 as the reference (boundary) node.
# The edge potential difference is dp = phi[u] - phi[v], so around the mesh the three edge
# potential differences sum to zero and the three edge flows are equal, as in the loop
# formulation these cases were first written for.


class QuadraticLoss(ImplicitLaw):
    """dp = a |q| q on every edge of its kind, written as an implicit law."""

    def __init__(self, kind="loss"):
        super().__init__(kind)

    def residual(self, dp, q, drivers, ctx):
        return dp - drivers["a"] * q.abs() * q


def _mesh(kinds):
    return _net([0, 1, 2], [(0, 1, kinds[0]), (1, 2, kinds[1]), (2, 0, kinds[2])])


def _quadratic_loop():
    net = _mesh(["source", "loss", "loss"])
    layer = KirchhoffLayer(net, "loop", [DrivenSource(), QuadraticLoss()], boundary=[0])
    return net, layer


def _quadratic_loop_solve(theta, **kwargs):
    """theta = [a0, a1, a2]: the drive on edge 0 and the loss coefficients of edges 1, 2.
    Returns the edge potential differences and the edge flows."""
    _, layer = _quadratic_loop()
    drivers = {"V": theta[:1], "a": theta[1:]}
    # Newton from zero meets a singular Jacobian (d(a|q|q)/dq = 0 at q = 0): start from q = -1
    phi, q = layer.solve(GROUND, drivers, q0=-torch.ones(3, dtype=F64), **kwargs)
    p = torch.stack([phi[0] - phi[1], phi[1] - phi[2], phi[2] - phi[0]])
    return p, q


def test_quadratic_loop_matches_the_worked_example_p_and_q():
    # old expectation (GOLD["quadratic_loop_a111"]): p = [1, -1/2, -1/2], q = -1/sqrt(2) on all
    # three edges. Same physics; the unknowns are now the nodal potentials and edge flows.
    p, q = _quadratic_loop_solve(torch.ones(3, dtype=F64))
    torch.testing.assert_close(p, torch.tensor(GOLD["quadratic_loop_a111"]["p"], dtype=F64),
                               rtol=1e-8, atol=0.0)
    torch.testing.assert_close(q, torch.tensor(GOLD["quadratic_loop_a111"]["q"], dtype=F64),
                               rtol=1e-8, atol=0.0)
    assert q[0].item() == pytest.approx(-1.0 / math.sqrt(2.0), rel=1e-8)


def test_quadratic_loop_jacobian_matches_the_worked_example_dp_da_dq_da():
    def pq(theta):
        p, q = _quadratic_loop_solve(theta)
        return torch.cat([p, q])

    jac = torch.autograd.functional.jacobian(pq, torch.ones(3, dtype=F64))  # rows p0..p2, q0..q2
    torch.testing.assert_close(jac[:3], torch.tensor(GOLD["quadratic_loop_a111"]["dp_da"],
                                                      dtype=F64), rtol=1e-6, atol=0.0)
    torch.testing.assert_close(jac[3:], torch.tensor(GOLD["quadratic_loop_a111"]["dq_da"],
                                                      dtype=F64), rtol=1e-6, atol=0.0)


def test_quadratic_loop_gradcheck():
    theta = torch.ones(3, dtype=F64, requires_grad=True)

    def f(th):
        p, q = _quadratic_loop_solve(th)
        return torch.cat([p, q])

    assert torch.autograd.gradcheck(f, (theta,), eps=1e-6, atol=1e-6)


def test_spring_mass_damper_matches_the_worked_example_displacement_series():
    # old law, theta = [L, R, C, q0_prev, p2_prev, dt] = [1, .2, 1, 0, 1, .2]:
    #   q0 - q0_prev - dt p0 = 0   (inductor, L = 1, implicit Euler)
    #   p1 - R q1 = 0              (resistor)
    #   p2 - p2_prev - dt q2 = 0   (capacitor, C = 1, implicit Euler)
    # now: an inductance (edge 0), a conductance 1/R (edge 1) and a capacitance whose initial
    # voltage is p2(t0) = 1 (edge 2), marched by backward Euler from the "initial" state.
    R, dt = 0.2, 0.2
    net = _mesh(["ind", "r", "cap"])
    layer = KirchhoffLayer(
        net, "smd", [Ind(torch.tensor([1.0])), Conductance(torch.tensor([1.0 / R]), kind="r"),
                     Cap(torch.tensor([1.0]), v0=1.0)], boundary=[0])
    n2 = net.nodes.index(2)
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    displacement = [phi[n2].item()]     # p2(t0) / C = 1: the golden series' first entry
    for _ in range(50):
        phi, q = layer.solve(GROUND, history=(phi, q), dt=dt, integration="backward_euler",
                             differentiable=False)
        displacement.append(phi[n2].item())     # edge 2 is 2 -> 0 with phi[0] = 0
    want = GOLD["spring_mass_damper_displacement"]
    assert len(displacement) == len(want) == 51
    torch.testing.assert_close(torch.tensor(displacement, dtype=F64),
                               torch.tensor(want, dtype=F64), rtol=1e-8, atol=0.0)


class _WrongWidth(ImplicitLaw):
    def residual(self, dp, q, drivers, ctx):
        return dp[..., :1] - 1.0     # one value for a kind of two edges


class _ScalarResidual(ImplicitLaw):
    def residual(self, dp, q, drivers, ctx):
        return dp.sum() - 1.0        # 0-dim


class _WrongWidthFlow(ConstitutiveLaw):
    def flow(self, dp, drivers=None):
        return dp[..., :1]           # one value for a kind of two edges


@pytest.mark.parametrize("law, what", [(_WrongWidth("w"), "residual"),
                                        (_WrongWidthFlow("w"), "flow"),
                                        (_ScalarResidual("w"), "residual")])
def test_law_of_the_wrong_width_is_refused_by_layer_and_kind(law, what):
    # the source edge makes the layer an MNA layer even when the bad law is a potential law
    net = _net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "w"), ("b", "gnd", "w")])
    layer = KirchhoffLayer(net, "bad", [Source(torch.tensor([1.0])), law],
                           boundary=["gnd"])
    with pytest.raises(ValueError, match=rf"'bad'.*{what}.*'w'.*(width 1|a scalar).*2 edges"):
        layer.solve(GROUND, differentiable=False, phi0=torch.zeros(2, dtype=F64),
                    q0=torch.zeros(1, dtype=F64))


def test_batch_leading_law_output_passes_the_width_check():
    net = _net(["gnd", "a"], [("a", "gnd", "source"), ("a", "gnd", "r")])
    layer = KirchhoffLayer(net, "ok", [DrivenSource(), Conductance(torch.tensor([1.0]),
                                                                     kind="r")],
                           boundary=["gnd"])
    phi, q = layer.solve(GROUND, {"V": torch.tensor([[1.0], [2.0], [3.0]])},
                         differentiable=False)
    assert q.shape == (3, 2)
    torch.testing.assert_close(q[:, 1], torch.tensor([1.0, 2.0, 3.0]), rtol=0, atol=1e-13)


def test_dc_mode_with_dt_is_refused():
    _, layer = _rc()
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    with pytest.raises(ValueError, match=r"mode='dc'.*takes no step; drop dt"):
        layer.solve(GROUND, mode="dc", dt=0.1, history=(phi, q), differentiable=False)


def test_singular_dc_message_does_not_mention_the_initial_point():
    with pytest.raises(RuntimeError, match="singular") as info:
        _voltage_loop().solve(GROUND, differentiable=False)
    assert "v0" not in str(info.value)

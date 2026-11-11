"""The linear guess (`KirchhoffLayer.linear_init`) on every layer. A cold solve of a layer
with flow unknowns still starts from zeros; started from the guess instead, a nonlinear one
finds the same root."""

import math

import pytest
import torch

from noodl import LinearStorage
from noodl.apps.electrical import JunctionDiode, PrescribedVoltage, Resistance
from noodl.elements import Conductance, ConstitutiveLaw, ImplicitLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from tests._circuit_laws import Cap, Ind, Source
from tests.layers.test_mna import Orifice, _fan_layer, _net, _transformer_layer

F64 = torch.float64
C128 = torch.complex128
GROUND = torch.zeros(1, dtype=F64)


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _close(actual, expected, tol):
    torch.testing.assert_close(actual, expected, rtol=0, atol=tol)


def test_linear_guess_of_a_linear_layer_is_its_solution():
    # source + transformer + load: every law affine, so the guess is the solution
    _, layer = _transformer_layer()
    phi, q = layer.solve(GROUND, differentiable=False)
    phi_i, q_f = layer.linear_init(GROUND, {}, None, q=True)
    _close(phi_i, phi[..., layer.interior], 1e-12)
    _close(q_f, q[..., layer.implicit_edges], 1e-12)
    assert torch.equal(layer.linear_init(GROUND, {}, None), phi_i)


def test_fast_path_linear_guess_is_unchanged():
    # gnd(0) -1- a -2- b -0.5- c(4): the nodal linear solve, and no flow unknowns
    net = _net(["gnd", "a", "b", "c"], [("gnd", "a", "r"), ("a", "b", "r"), ("b", "c", "r")])
    g = torch.tensor([1.0, 2.0, 0.5])
    layer = KirchhoffLayer(net, "e", [Conductance(g, kind="r")], boundary=["gnd", "c"])
    pb = torch.tensor([0.0, 4.0])
    sources = torch.tensor([0.0, 0.3, -0.1, 0.0])
    phi_i = layer.linear_init(pb, {}, sources)
    phi_q, q_f = layer.linear_init(pb, {}, sources, q=True)
    assert torch.equal(phi_q, phi_i) and q_f.shape == (0,)
    # A_I diag(g) A_I^T phi_I = s_I + (boundary terms), by hand
    K = torch.tensor([[g[0] + g[1], -g[1]], [-g[1], g[1] + g[2]]])
    rhs = sources[1:3] + torch.tensor([g[0] * pb[0], g[2] * pb[1]])
    _close(phi_i, torch.linalg.solve(K, rhs), 1e-12)


def _surrogate_residual(layer, x, drivers, sources):
    step = layer._step(None, None, None, layer.scheme, None, None, drivers)
    tangents = layer._tangents(drivers, step.ctx)
    return layer._residual(x, GROUND, drivers, sources, step, layer._owner_call, tangents)[0]


def test_fan_guess_solves_the_surrogate_and_a_solve_from_it_finds_the_root():
    layer = _fan_layer()
    drivers = {"head": torch.tensor([5.0]), "c": torch.tensor([0.7])}
    src = torch.zeros(3)
    phi_i, q_f = layer.linear_init(GROUND, drivers, src, q=True)
    x = torch.cat([phi_i, q_f], -1)
    assert _surrogate_residual(layer, x, drivers, src).abs().max() < 1e-12
    # the surrogate replaces the orifices by their tangents at zero, so the guess is not
    # the root
    assert layer._residual(x, GROUND, drivers, src)[0].abs().max() > 1e-3

    diag: dict = {}
    phi, q = layer.solve(GROUND, drivers, src, phi0=phi_i, q0=q_f, differentiable=False,
                         diagnostics=diag)
    assert bool(diag["converged"].all())
    warm_phi, warm_q = layer.solve(GROUND, drivers, src, phi0=phi[..., layer.interior] * 1.05,
                                   q0=q[..., layer.implicit_edges] * 0.9,
                                   differentiable=False, atol=1e-14, rtol=0.0)
    _close(phi, warm_phi, 1e-10)
    _close(q, warm_q, 1e-10)


def _diode_circuits():
    # 1 V source a -> gnd, R a -> b, diode b -> gnd (forward), and two variants
    edges = [("a", "gnd", "voltage_source"), ("a", "b", "resistor"), ("b", "gnd", "diode")]
    for V, R in ((1.0, 100.0), (5.0, 1000.0), (-2.0, 10.0)):
        net = _net(["gnd", "a", "b"], edges)
        yield KirchhoffLayer(net, "e", [PrescribedVoltage(torch.tensor([V])),
                                        Resistance(torch.tensor([R])), JunctionDiode()],
                             boundary=["gnd"])


@pytest.mark.parametrize("layer", list(_diode_circuits()))
def test_diode_solve_from_the_guess_converges_to_the_root_of_the_cold_solve(layer):
    assert not layer.linear and layer.implicit_edges.numel() > 0
    phi_i, q_f = layer.linear_init(GROUND, {}, None, q=True)
    diag: dict = {}
    phi, q = layer.solve(GROUND, phi0=phi_i, q0=q_f, differentiable=False, diagnostics=diag,
                         atol=1e-15, rtol=0.0)
    assert bool(diag["converged"].all())
    # the cold solve starts from zeros
    phi_z, q_z = layer.solve(GROUND, differentiable=False, atol=1e-15, rtol=0.0)
    _close(phi, phi_z, 1e-10)
    _close(q, q_z, 1e-10)


def test_phasor_linear_guess_is_the_solution():
    R, L, C = 2.0, 1e-3, 1e-6
    net = _net(["gnd", "a", "b", "c"], [("a", "gnd", "source"), ("a", "b", "r"),
                                        ("b", "c", "ind"), ("c", "gnd", "cap")])
    layer = KirchhoffLayer(
        net, "e", [Source(torch.tensor([1.0])), Conductance(torch.tensor([1.0 / R]), kind="r"),
                   Ind(torch.tensor([L])), Cap(torch.tensor([C]))],
        boundary=["gnd"], phasor=True)
    drivers = {"e.frequency": torch.tensor([50.0, 5032.92, 1e5])}
    phi, q = layer.solve(GROUND, drivers, differentiable=False)
    phi_i, q_f = layer.linear_init(GROUND, drivers, None, q=True)
    assert phi_i.dtype == C128 and q_f.dtype == C128
    _close(phi_i, phi[..., layer.interior], 1e-12)
    _close(q_f, q[..., layer.implicit_edges], 1e-12)
    w = 2 * math.pi * drivers["e.frequency"].to(C128)
    current = 1.0 / (R + 1j * w * L + 1.0 / (1j * w * C))
    _close(q[..., layer.kind_slice("r")][..., 0], current, 1e-12)


def test_linear_guess_holds_the_held_rows_of_the_initial_point():
    # source s -> gnd (5 V), R s -> a; storage at a holds it at 1.5 V; a nonlinear orifice
    # a -> gnd
    V0 = 1.5
    net = _net(["gnd", "s", "a"], [("s", "gnd", "voltage_source"), ("s", "a", "resistor"),
                                   ("a", "gnd", "ori")])
    layer = KirchhoffLayer(
        net, "el", [PrescribedVoltage(5.0, kind="voltage_source"),
                    Resistance(2.0, kind="resistor"), Orifice()],
        boundary=["gnd"], storage=[LinearStorage([2], 0.3, initial=V0)])
    drivers = {"c": torch.tensor([0.7])}
    step = layer._step("initial", None, None, layer.scheme, None, None, drivers)
    phi_i = layer._linear_init_at(step, GROUND, drivers, None)
    a = int((layer.interior == 2).nonzero())
    assert phi_i[..., a].item() == V0
    phi, _ = layer.solve(GROUND, drivers, mode="initial", differentiable=False)
    assert phi[..., 2].item() == pytest.approx(V0, abs=1e-12)


def test_a_singular_linear_guess_is_diagnosed():
    # two ideal sources in parallel: the system is singular, which the guess reports by the
    # rank test, as a failed solve does
    net = _net(["gnd", "a"], [("a", "gnd", "source"), ("a", "gnd", "source"),
                              ("a", "gnd", "ori")])
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([1.0, 1.0])), Orifice()],
                           boundary=["gnd"])
    with pytest.raises(RuntimeError, match="singular"):
        layer.linear_init(GROUND, {"c": torch.tensor([0.7])}, None)
    with pytest.raises(RuntimeError, match="singular"):
        layer.solve(GROUND, {"c": torch.tensor([0.7])}, differentiable=False)


def test_the_guess_of_a_layer_with_flow_unknowns_is_detached():
    # the documented contract: no gradient reaches the inputs on this path, unlike on a
    # layer without flow unknowns
    _, layer = _transformer_layer()
    pb = torch.zeros(1, requires_grad=True)
    out = layer.linear_init(pb, {}, None)
    assert not out.requires_grad
    net = _net(["gnd", "a"], [("gnd", "a", "r")])
    fast = KirchhoffLayer(net, "f", [Conductance(torch.tensor([1.0]), kind="r")],
                          boundary=["gnd"])
    assert fast.linear_init(pb, {}, torch.tensor([0.0, 1.0], requires_grad=True)).requires_grad


def test_a_starting_point_refusal_comes_before_the_unreachable_tensor_refusal():
    # a law whose flow has the wrong width refuses at the starting point; the same layer
    # holds a tensor with requires_grad the differentiable solve cannot reach
    class Wide(ConstitutiveLaw):
        def __init__(self):
            super().__init__("wide")
            self.g = self._param(torch.tensor([1.0], requires_grad=True), False)

        def flow(self, dp, drivers=None):
            return torch.cat([self.g * dp, self.g * dp], -1)

    net = _net(["gnd", "a"], [("a", "gnd", "source"), ("a", "gnd", "wide")])
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([1.0])), Wide()], boundary=["gnd"])
    with pytest.raises(ValueError, match="the flow of the 'wide' law is width 2"):
        layer.solve(GROUND)


class _QuadraticLoss(ImplicitLaw):
    """dp = r q |q|, a loss with no slope at zero flow; its start flow is `start` (`None`:
    the default, zero) or the driver `"q_start"` when `start` is `"driver"`."""

    def __init__(self, r, start=None):
        super().__init__("loss")
        self.r = self._param(r, False)
        self.start = start

    def residual(self, dp, q, drivers, ctx):
        return dp - self.r * q * q.abs()

    def start_flow(self, drivers=None, ctx=None):
        if isinstance(self.start, str):
            return drivers[self.start]
        return None if self.start is None else torch.tensor([self.start])


def _loss_layer(law):
    # in(10) -loss- mid -g=1- out(0)
    net = _net(["in", "mid", "out"], [("in", "mid", "loss"), ("mid", "out", "pipe")])
    return KirchhoffLayer(net, "w", [law, Conductance(torch.tensor([1.0]), kind="pipe")],
                          boundary=["in", "out"])


PB = torch.tensor([10.0, 0.0], dtype=F64)


def test_the_guess_linearises_an_implicit_law_at_its_start_flow():
    # By hand, at q_s: 10 - phi - r q_s^2 - 2 r q_s (q - q_s) = 0 and q = g phi, so
    # phi = (10 + r q_s^2) / (1 + 2 r q_s g); r = 2, g = 1, q_s = 1 gives 12 / 5.
    phi_i, q_f = _loss_layer(_QuadraticLoss(torch.tensor([2.0]), 1.0)).linear_init(
        PB, {}, None, q=True)
    _close(phi_i, torch.tensor([2.4]), 1e-12)
    _close(q_f, torch.tensor([2.4]), 1e-12)
    # at the root, q_s = 2, the linearisation is exact: the guess is the solution
    phi_i, q_f = _loss_layer(_QuadraticLoss(torch.tensor([2.0]), 2.0)).linear_init(
        PB, {}, None, q=True)
    _close(phi_i, torch.tensor([2.0]), 1e-12)
    _close(q_f, torch.tensor([2.0]), 1e-12)


def test_a_law_without_a_start_flow_keeps_the_zero_flow_guess_bit_for_bit():
    # zero flow: no slope, so the loss carries the whole 10 at q = 10
    none = _loss_layer(_QuadraticLoss(torch.tensor([2.0]))).linear_init(PB, {}, None, q=True)
    zero = _loss_layer(_QuadraticLoss(torch.tensor([2.0]), 0.0)).linear_init(
        PB, {}, None, q=True)
    _close(none[1], torch.tensor([10.0]), 1e-12)
    assert torch.equal(none[0], zero[0]) and torch.equal(none[1], zero[1])


def test_a_start_flow_from_the_drivers_batches_the_guess():
    layer = _loss_layer(_QuadraticLoss(torch.tensor([2.0]), "q_start"))
    _, q_f = layer.linear_init(PB, {"q_start": torch.tensor([[1.0], [2.0]])}, None, q=True)
    _close(q_f, torch.tensor([[2.4], [2.0]]), 1e-12)


def test_a_start_flow_of_the_wrong_width_is_refused_by_name():
    layer = _loss_layer(_QuadraticLoss(torch.tensor([2.0]), "q_start"))
    with pytest.raises(ValueError, match="the start flow of the 'loss' law is width 2"):
        layer.linear_init(PB, {"q_start": torch.tensor([1.0, 2.0])}, None)


def test_the_start_flow_moves_no_solve_from_zeros():
    # a cold solve still starts from zeros, with or without a start flow
    plain = _loss_layer(_QuadraticLoss(torch.tensor([2.0]))).solve(PB, differentiable=False)
    given = _loss_layer(_QuadraticLoss(torch.tensor([2.0]), 1.0)).solve(
        PB, differentiable=False)
    assert torch.equal(plain[0], given[0]) and torch.equal(plain[1], given[1])

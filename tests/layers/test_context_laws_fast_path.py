"""Laws that read the law context stay on the fast path (core-unification M6, Task 30).

A law with a `ctx` parameter in `flow` still gives its flow from the potential difference: a
time-dependent current source, or a storing law written in flow form. Such a layer has no
flow unknowns; its flows and slopes are evaluated in the step's context, and its Jacobian is
the symmetric network matrix. Each case is checked against the same circuit forced onto the
general path by wrapping a law as a relation (`_AsRelation`: `r = q - flow(dp)`).

The initial point with held storage (`mode="initial"`, a storage whose `initial()` is not
`None`) on such a layer is solved, for that one call, on the sparse operator with the held
rows, in a layout built on the first such call (`_held_unknowns`).
"""

from __future__ import annotations

import pytest
import torch

import noodl.layers.kirchhoff as kirchhoff_module
from noodl.apps.electrical import Capacitance, PrescribedCurrent, Resistance, Sine
from noodl.elements import ConstitutiveLaw, ImplicitLaw, PowerLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.operators.graph import GraphLaplacianOperator
from noodl.operators.shifted import ShiftedOperator
from noodl.storage import LinearStorage
from noodl.topology import Network

F64 = torch.float64
GROUND = torch.zeros(1, dtype=F64)


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _t(values, grad=False):
    return torch.tensor(values, dtype=F64, requires_grad=grad)


def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


class _AsRelation(ImplicitLaw):
    """`law` written as a relation, `r = q - law(dp)`: the same circuit, on the general path."""

    def __init__(self, law):
        super().__init__(law.kind)
        self.law = law

    def residual(self, dp, q, drivers=None, ctx=None):
        return q - self.law(dp, drivers, ctx)


class _FlowFormCap(ConstitutiveLaw):
    """A backward-Euler capacitor in flow form, `q = C (dp - dp_prev) / dt` on a step (open
    otherwise): a dynamic law that gives its flow, so it adds no flow unknown."""

    dynamic = True
    linear = True

    def __init__(self, C, kind="cap", learnable=False):
        super().__init__(kind)
        self.C = self._param(C, learnable)

    def flow(self, dp, drivers=None, ctx=None):
        if ctx is None or ctx.mode != "transient":
            return torch.zeros_like(dp) * self.C
        return self.C * (dp - ctx.dp_prev) / ctx.dt


class _Spy(list):
    """The types of the operators the fast path builds, until `stop()`; building a
    `SparseOperator` meanwhile fails the test."""

    def __init__(self, monkeypatch):
        super().__init__()
        self._patch = monkeypatch
        original = KirchhoffLayer._operator_at

        def spy(layer, *args, **kwargs):
            op = original(layer, *args, **kwargs)
            self.append(type(op))
            return op

        def refuse(*args, **kwargs):
            raise AssertionError("a SparseOperator was built on the fast path")

        monkeypatch.setattr(KirchhoffLayer, "_operator_at", spy)
        monkeypatch.setattr(kirchhoff_module, "SparseOperator", refuse)

    def stop(self):
        self._patch.undo()


@pytest.fixture
def operators(monkeypatch):
    return _Spy(monkeypatch)


# ------------------------------------------------------------------ time-dependent source
def _source_layer(*, storage=False, general=False, amplitude=1.0, C=0.4):
    source = PrescribedCurrent(_t([amplitude]), waveforms=[Sine(2.0, phase=0.3)],
                               learnable=True)
    stores = [LinearStorage([1], _t([C]), learnable=True)] if storage else []
    law = _AsRelation(source) if general else source
    net = _net(["gnd", "a"], [("a", "gnd", "current_source"), ("a", "gnd", "resistor")])
    layer = KirchhoffLayer(net, "e", [law, Resistance(_t([2.0]))], boundary=["gnd"],
                           storage=stores)
    return layer, source, (stores[0] if storage else None)


def _source_run(layer, scheme=None):
    """The steady point at t = 0.3, or (with storage) two steps of `scheme` from a state."""
    if not layer.storage:
        return layer.solve(GROUND, t=0.3, differentiable=False)
    phi = torch.tensor([0.0, -0.5], dtype=F64)
    q = torch.zeros(2, dtype=F64)
    w = torch.zeros(1, dtype=F64)
    out = []
    for t in (0.3, 0.6):
        kw = {"dt": 0.3, "scheme": scheme, "history": (phi, q), "storage_flow": w}
        phi, q = layer.solve(GROUND, t=t, differentiable=False, **kw)
        w = layer.storage_flows(phi, {}, **kw)
        out += [phi, q, w]
    return tuple(out)


def test_a_time_dependent_source_keeps_the_fast_path(operators):
    layer, _, _ = _source_layer()
    assert layer.implicit_edges.numel() == 0
    phi, q = _source_run(layer)
    assert operators and all(op is GraphLaplacianOperator for op in operators)
    operators.stop()
    general, _, _ = _source_layer(general=True)
    assert general.implicit_edges.numel() == 1
    phi_g, q_g = general.solve(GROUND, t=0.3, differentiable=False)
    torch.testing.assert_close(phi, phi_g, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(q, q_g, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_a_time_dependent_source_with_storage_keeps_the_fast_path(operators, scheme):
    layer, _, _ = _source_layer(storage=True)
    assert layer.implicit_edges.numel() == 0
    fast = _source_run(layer, scheme)
    assert operators and all(op in (GraphLaplacianOperator, ShiftedOperator)
                             for op in operators)
    operators.stop()
    general, _, _ = _source_layer(storage=True, general=True)
    for a, b in zip(fast, _source_run(general, scheme), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_gradcheck_of_a_time_dependent_source_with_storage(scheme):
    layer, source, storage = _source_layer(storage=True)
    phi_start = torch.tensor([0.0, -0.5], dtype=F64)

    def run(amplitude, C):
        phi, q, w = phi_start, torch.zeros(2, dtype=F64), torch.zeros(1, dtype=F64)
        out = []
        for t in (0.3, 0.6):
            kw = {"dt": 0.3, "scheme": scheme, "history": (phi, q), "storage_flow": w}
            phi, q = layer.solve(GROUND, t=t, **kw)
            w = layer.storage_flows(phi, {}, **kw)
            out += [phi, q, w]
        return torch.cat(out)

    assert torch.autograd.gradcheck(run, (source.amplitude, storage.C))


def test_gradcheck_of_a_time_dependent_source_steady():
    layer, source, _ = _source_layer()
    assert torch.autograd.gradcheck(
        lambda a: torch.cat(layer.solve(GROUND, t=0.3)), (source.amplitude,))


# ------------------------------------------------------------------ flow-form capacitor
def _cap_layer(cap):
    """s -R- a -cap- b -R- gnd, with s prescribed: the capacitor joins two unknown nodes."""
    net = _net(["gnd", "s", "a", "b"], [("s", "a", "r"), ("a", "b", "cap"), ("b", "gnd", "r")])
    return KirchhoffLayer(net, "e", [Resistance(_t([1.0, 2.0]), kind="r"), cap],
                          boundary=["gnd", "s"], scheme="backward_euler")


PB = torch.tensor([0.0, 1.0], dtype=F64)


def _steps(layer, n, phi, q):
    out = []
    for i in range(n):
        phi, q = layer.solve(PB, dt=0.1, t=0.1 * (i + 1), history=(phi, q))
        out += [phi, q]
    return out


def test_a_flow_form_capacitor_stays_on_the_fast_path_and_matches_capacitance(operators):
    layer = _cap_layer(_FlowFormCap(_t([0.5])))
    assert layer.implicit_edges.numel() == 0 and layer.dynamic
    start = (torch.tensor([0.0, 1.0, 0.3, 0.1], dtype=F64), torch.zeros(3, dtype=F64))
    with torch.no_grad():
        fast = _steps(layer, 3, *start)
    assert operators and all(op is GraphLaplacianOperator for op in operators)
    operators.stop()
    reference = _cap_layer(Capacitance(_t([0.5]), kind="cap"))
    assert reference.implicit_edges.numel() == 1
    with torch.no_grad():
        for a, b in zip(fast, _steps(reference, 3, *start), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_the_flow_form_capacitor_slope_certifies_grounding():
    """`C / dt > 0` joins a and b, so the SPD certificate holds with b grounded only
    through the capacitor's slope and R."""
    cap = _FlowFormCap(_t([0.5]))
    layer = _cap_layer(cap)
    dq = layer._dflows(layer.assemble(torch.zeros(2), PB), {}, kirchhoff_module.LawContext(
        mode="transient", dt=0.1, scheme="backward_euler",
        dp_prev=torch.zeros(3), q_prev=torch.zeros(3)))
    torch.testing.assert_close(dq, torch.tensor([1.0, 0.5, 5.0], dtype=F64))


def test_gradcheck_of_a_flow_form_capacitor_through_two_steps():
    cap = _FlowFormCap(_t([0.5]), learnable=True)
    layer = _cap_layer(cap)
    phi_start = torch.tensor([0.0, 1.0, 0.3, 0.1], dtype=F64, requires_grad=True)
    q_start = torch.zeros(3, dtype=F64)
    assert torch.autograd.gradcheck(
        lambda C, p: torch.cat(_steps(layer, 2, p, q_start)), (cap.C, phi_start))


# ------------------------------------------------------------------ refusals and checks
def test_a_floating_node_is_refused_by_the_grounding_check_by_name():
    """b is reached only through a current source: refused at the start of the solve."""
    net = _net(["gnd", "a", "b"], [("a", "gnd", "r"), ("a", "b", "current_source")])
    layer = KirchhoffLayer(net, "e", [Resistance(_t([2.0]), kind="r"),
                                      PrescribedCurrent(_t([1.0]),
                                                        waveforms=[Sine(2.0)])],
                           boundary=["gnd"])
    with pytest.raises(RuntimeError, match=r"floating nodes .*\['b'\]"):
        layer.solve(GROUND, t=0.3, differentiable=False)


def test_cg_and_the_convex_line_search_are_accepted():
    layer, _, _ = _source_layer()
    reference, _ = layer.solve(GROUND, t=0.3, differentiable=False)
    for kwargs in ({"method": "cg"}, {"line_search": "convex"}):
        phi, _ = layer.solve(GROUND, t=0.3, differentiable=False, **kwargs)
        torch.testing.assert_close(phi, reference, rtol=1e-10, atol=1e-12)
    cg = KirchhoffLayer(layer.net, "e", list(layer._laws), boundary=["gnd"],
                        linear_solver="cg")
    torch.testing.assert_close(cg.solve(GROUND, t=0.3, differentiable=False)[0], reference,
                               rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------ held initial storage
def _held_layer(*, general=False, linear_solver="auto"):
    """A power-law network with storage holding initial potentials at two of its nodes."""
    net = _net(["w", "a", "b", "c", "l"], [("w", "a", "pipe"), ("a", "b", "pipe"),
                                           ("b", "c", "pipe"), ("c", "l", "pipe"),
                                           ("a", "c", "pipe")])
    law = PowerLaw(_t([0.010, 0.008, 0.012, 0.009, 0.011]), 0.65, dp_transition=1e-6,
                   kind="pipe")
    storage = LinearStorage([1, 2], _t([0.06, 0.03]), initial=_t([0.4, -0.2]))
    return KirchhoffLayer(net, "e", [_AsRelation(law) if general else law],
                          boundary=["w", "l"], storage=[storage], linear_solver=linear_solver)


PB_HELD = torch.tensor([6.0, 0.5], dtype=F64)
SOURCES = torch.tensor([0.0, 0.02, -0.01, 0.015, 0.0], dtype=F64)


def test_the_held_initial_point_matches_the_general_path():
    layer = _held_layer()
    assert layer.implicit_edges.numel() == 0
    # Newton to rounding on both, so the two layouts agree to 1e-12
    tight = {"atol": 1e-16, "rtol": 1e-15}
    phi, q = layer.solve(PB_HELD, {}, SOURCES, mode="initial", differentiable=False, **tight)
    w = layer.storage_flows(phi, {}, dt=None, mode="initial", q=q, sources=SOURCES)
    general = _held_layer(general=True)
    phi_g, q_g = general.solve(PB_HELD, {}, SOURCES, mode="initial", differentiable=False,
                               **tight)
    w_g = general.storage_flows(phi_g, {}, dt=None, mode="initial", q=q_g, sources=SOURCES)
    torch.testing.assert_close(phi[[1, 2]], torch.tensor([0.4, -0.2], dtype=F64),
                               rtol=0, atol=0)
    for a, b in ((phi, phi_g), (q, q_g), (w, w_g)):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_the_held_layout_is_built_lazily():
    layer = _held_layer()
    assert layer._held_unknowns is None
    phi, q = layer.solve(PB_HELD, {}, SOURCES, differentiable=False)
    for scheme in ("backward_euler", "trapezoidal"):
        phi, q = layer.solve(PB_HELD, {}, SOURCES, dt=0.2, scheme=scheme, history=(phi, q),
                             storage_flow=torch.zeros(2, dtype=F64), differentiable=False)
    layer.solve(PB_HELD, {}, SOURCES, mode="dc", differentiable=False)
    assert layer._held_unknowns is None
    layer.solve(PB_HELD, {}, SOURCES, mode="initial", differentiable=False)
    assert layer._held_unknowns is not None and layer._unknowns is None
    assert layer._held_unknowns.n_F == 0
    # an initial point that holds nothing stays on the fast path
    plain = KirchhoffLayer(layer.net, "e", list(layer._laws), boundary=["w", "l"],
                           storage=[LinearStorage([1, 2], _t([0.06, 0.03]))])
    plain.solve(PB_HELD, {}, SOURCES, mode="initial", differentiable=False)
    assert plain._held_unknowns is None


def test_the_held_initial_point_refuses_cg_in_plain_words():
    layer = _held_layer(linear_solver="cg")
    with pytest.raises(ValueError, match="the initial point with held storage"):
        layer.solve(PB_HELD, {}, SOURCES, mode="initial", differentiable=False)
    layer.solve(PB_HELD, {}, SOURCES, differentiable=False)   # every other solve takes cg


def test_gradcheck_of_the_held_initial_point():
    law = PowerLaw(_t([0.010, 0.008, 0.012, 0.009, 0.011]), 0.65, dp_transition=1e-6,
                   kind="pipe", learnable=True)
    layer = _held_layer()
    layer = KirchhoffLayer(layer.net, "e", [law], boundary=["w", "l"],
                           storage=list(layer.storage))
    sources = SOURCES.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda C, s: torch.cat(layer.solve(PB_HELD, {}, s, mode="initial")), (law.C, sources))

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
from noodl.elements._slopes import one_port_blocks
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
    """b's only link is the capacitor: on a step its slope `C / dt > 0` grounds b through a
    and R, so the step solves; at the steady point the capacitor is open, and b is refused
    as floating, by name."""
    net = _net(["s", "a", "b"], [("s", "a", "r"), ("a", "b", "cap")])
    layer = KirchhoffLayer(net, "e", [Resistance(_t([1.0]), kind="r"),
                                      _FlowFormCap(_t([0.5]))],
                           boundary=["s"], scheme="backward_euler")
    pb = torch.tensor([1.0], dtype=F64)
    start = (torch.tensor([1.0, 0.3, 0.1], dtype=F64), torch.zeros(2, dtype=F64))
    dq = layer._dflows(layer.assemble(torch.zeros(2), pb), {}, kirchhoff_module.LawContext(
        mode="transient", dt=0.1, scheme="backward_euler",
        dp_prev=torch.zeros(2), q_prev=torch.zeros(2)))
    torch.testing.assert_close(dq, torch.tensor([1.0, 5.0], dtype=F64))
    phi, q = layer.solve(pb, dt=0.1, history=start, differentiable=False)
    # b holds the capacitor's charge: no current through it, so a and b move together
    torch.testing.assert_close(phi[2] - phi[1], start[0][2] - start[0][1], rtol=0,
                               atol=1e-14)
    with pytest.raises(RuntimeError, match=r"floating nodes .*\['b'\]"):
        layer.solve(pb, t=0.0, differentiable=False)


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


class _AidedTimeLaw(ConstitutiveLaw):
    """A context law whose `jacobian` is a Newton aid, not its derivative: `q = g (1 + t)
    (dp + dp^3 / 3)`, with the slope `g (1 + t) (1 + 2 dp^2)`; it declares it
    (`exact_slope = False`), so the differentiable solve differentiates `flow` instead."""

    exact_slope = False

    def __init__(self, g, kind="aided"):
        super().__init__(kind)
        self.g = self._param(g, True)

    def flow(self, dp, drivers=None, ctx=None):
        return self.g * (1.0 + ctx.t) * (dp + dp ** 3 / 3.0)

    def jacobian(self, dp, q, drivers, ctx):
        return (one_port_blocks(self.g * (1.0 + ctx.t) * (1.0 + 2.0 * dp ** 2), dp),)


def test_gradcheck_of_a_context_law_with_an_inexact_slope():
    law = _AidedTimeLaw(_t([0.7]))
    net = _net(["s", "a", "gnd"], [("s", "a", "r"), ("a", "gnd", "aided")])
    layer = KirchhoffLayer(net, "e", [Resistance(_t([0.5]), kind="r"), law],
                           boundary=["s", "gnd"])
    pb = torch.tensor([2.0, 0.0], dtype=F64, requires_grad=True)
    # the two solves step with different slopes (the aid, autograd): Newton to rounding
    tight = {"atol": 1e-16, "rtol": 1e-15}
    primal, _ = layer.solve(pb.detach(), t=0.3, differentiable=False, **tight)
    phi, _ = layer.solve(pb, t=0.3, **tight)
    torch.testing.assert_close(phi.detach(), primal, rtol=1e-10, atol=1e-12)
    assert torch.autograd.gradcheck(
        lambda g, b: torch.cat(layer.solve(b, t=0.3)), (law.g, pb))


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


class _LearnedInitial(LinearStorage):
    """A storage whose initial potentials are a learnable parameter."""

    def __init__(self, nodes, capacity, v0):
        super().__init__(nodes, capacity, learnable=True)
        self.v0 = torch.nn.Parameter(torch.as_tensor(v0, dtype=F64))

    def initial(self, drivers=None):
        return self.v0


def test_gradcheck_of_the_held_initial_point():
    """The law, the sources, the held potentials and the capacities that split a held node's
    imbalance between its storage entries (a second, non-holding storage at a)."""
    law = PowerLaw(_t([0.010, 0.008, 0.012, 0.009, 0.011]), 0.65, dp_transition=1e-6,
                   kind="pipe", learnable=True)
    held = _LearnedInitial([1, 2], _t([0.06, 0.03]), [0.4, -0.2])
    extra = LinearStorage([1], _t([0.02]), learnable=True)
    layer = KirchhoffLayer(_held_layer().net, "e", [law], boundary=["w", "l"],
                           storage=[held, extra])
    sources = SOURCES.clone().requires_grad_(True)

    def initial(C, s, v0, cap, cap_extra):
        phi, q = layer.solve(PB_HELD, {}, s, mode="initial")
        w = layer.storage_flows(phi, {}, dt=None, mode="initial", q=q, sources=s)
        return torch.cat([phi, q, w])

    assert torch.autograd.gradcheck(initial, (law.C, sources, held.v0, held.C, extra.C))


def test_a_floating_node_is_refused_at_the_held_initial_point_by_name():
    """c is reached only through a current source; a is held. The held initial point is
    certified like every other solve: c is refused by name, a counts as grounded."""
    net = _net(["g", "a", "b", "c"], [("g", "a", "r"), ("a", "b", "r"),
                                      ("b", "c", "current_source")])
    laws = [Resistance(_t([1.0, 2.0]), kind="r"), PrescribedCurrent(_t([1.0]))]
    layer = KirchhoffLayer(net, "e", laws, boundary=["g"],
                           storage=[LinearStorage([1], _t([0.5]), initial=0.2)])
    with pytest.raises(RuntimeError, match=r"floating nodes .*\['c'\]"):
        layer.solve(GROUND, mode="initial", differentiable=False)
    # held rows count as grounded: a node joined only to a held node solves
    net = _net(["g", "a", "b"], [("a", "b", "r")])
    layer = KirchhoffLayer(net, "e", [Resistance(_t([2.0]), kind="r")], boundary=["g"],
                           storage=[LinearStorage([1], _t([0.5]), initial=0.2)])
    phi, _ = layer.solve(GROUND, mode="initial", differentiable=False)
    torch.testing.assert_close(phi, torch.tensor([0.0, 0.2, 0.2], dtype=F64), rtol=0,
                               atol=1e-14)


# ------------------------------------------------- residual, jacobian, adjoint at a time t
@pytest.mark.parametrize("general", [False, True])
def test_the_public_residual_is_evaluated_at_t(general):
    """`residual`, `jacobian` and `adjoint` take `solve`'s `t`: zero residual at the solved
    point of t = 0.3, and a time-dependent source refuses `t=None` by name."""
    layer, _, _ = _source_layer(general=general)
    phi, q = layer.solve(GROUND, t=0.3, differentiable=False)
    phi_i = phi[..., layer.interior]
    x = {"q": q} if general else {}
    r = layer.residual(phi_i, GROUND, {}, None, t=0.3, **x)
    assert float(r.abs().max()) < 1e-12
    assert float(layer.residual(phi_i, GROUND, {}, None, t=0.4, **x).abs().max()) > 1e-3
    width = r.shape[-1]
    J = layer.jacobian(phi_i, GROUND, {}, t=0.3, **x)
    assert J.shape[-1] == width
    lam = layer.adjoint(phi_i, GROUND, {}, torch.ones(width, dtype=F64), t=0.3, **x)
    torch.testing.assert_close(J.mT @ lam, torch.ones(width, dtype=F64), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="time-dependent waveforms; pass t"):
        layer.residual(phi_i, GROUND, {}, None, **x)


def test_the_public_methods_keep_no_step_start_on_the_layer():
    net = _net(["gnd", "a"], [("a", "gnd", "cap"), ("a", "gnd", "resistor")])
    layer = KirchhoffLayer(net, "e", [_FlowFormCap(_t([0.5])), Resistance(_t([2.0]))],
                           boundary=["gnd"])
    phi_prev = _t([0.0, 1.0], grad=True)
    history = (phi_prev, torch.zeros(2, dtype=F64))
    phi_i = _t([0.4])
    layer.residual(phi_i, GROUND, {}, None, dt=0.1, history=history)
    assert layer._law_ctxs == ()
    layer.jacobian(phi_i, GROUND, {}, dt=0.1, history=history)
    assert layer._law_ctxs == ()
    layer.adjoint(phi_i, GROUND, {}, _t([1.0]), dt=0.1, history=history)
    assert layer._law_ctxs == ()


def test_the_law_context_is_restricted_once_per_context_in_a_differentiable_step(
        monkeypatch):
    """A differentiable step of a law reading the step start evaluates it in three contexts
    (the step's own, the residual's and the operator's), the last two alternating over the
    Newton iterations: each law's restriction of each is built once (per grad mode), not on
    every switch. The gradient through the step start is unchanged by the cache."""
    import dataclasses
    import sys

    builds = []
    replace = dataclasses.replace

    def counting(obj, **changes):
        frame = sys._getframe(1)
        if frame.f_code.co_name == "_law_ctx":
            builds.append((id(obj), frame.f_locals["s"], torch.is_grad_enabled()))
        return replace(obj, **changes)

    source = PrescribedCurrent(_t([1.0]), waveforms=[Sine(2.0, phase=0.3)])
    net = _net(["gnd", "a", "b"], [("a", "gnd", "current_source"), ("a", "b", "cap"),
                                   ("b", "gnd", "resistor")])
    layer = KirchhoffLayer(net, "e", [_AsRelation(source), _FlowFormCap(_t([0.5])),
                                      PowerLaw(_t([2.0]), 1.5, kind="resistor")],
                           boundary=["gnd"])
    start = _t([0.0, 0.2, 0.1], grad=True)
    q_prev = torch.zeros(3, dtype=F64)
    diag: dict = {}
    monkeypatch.setattr(kirchhoff_module.dataclasses, "replace", counting)
    phi, _ = layer.solve(GROUND, {}, None, dt=0.1, t=0.3, history=(start, q_prev),
                         diagnostics=diag)
    monkeypatch.undo()
    assert diag["newton_iterations"] > 1
    assert builds and len(builds) == len(set(builds))
    (g,) = torch.autograd.grad(phi.sum(), start)
    h = 1e-6
    fd = []
    with torch.no_grad():
        for i in range(3):
            e = torch.zeros(3, dtype=F64)
            e[i] = h
            up = layer.solve(GROUND, {}, None, dt=0.1, t=0.3, history=(start + e, q_prev),
                             differentiable=False, atol=1e-14)[0]
            dn = layer.solve(GROUND, {}, None, dt=0.1, t=0.3, history=(start - e, q_prev),
                             differentiable=False, atol=1e-14)[0]
            fd.append(float((up - dn).sum()) / (2 * h))
    torch.testing.assert_close(g, torch.tensor(fd, dtype=F64), rtol=1e-6, atol=1e-9)

"""Refusals a KirchhoffLayer owes its caller: a stale topology and unreachable gradients."""

import dataclasses
import pickle

import pytest
import torch

from noodl.elements import Conductance, FixedFlow, PowerLaw
from noodl.elements.base import ConstitutiveLaw, ImplicitLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network
from noodl.withdrawals import NodeWithdrawal

F64 = torch.float64


def _net() -> Network:
    net = Network(dtype=F64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="conduction")
    return net


def _layer(net: Network) -> KirchhoffLayer:
    return KirchhoffLayer(
        net, "heat", [Conductance(torch.tensor(1.0, dtype=F64))], boundary=["a"]
    )


# ------------------------------------------------------------------ R08
def test_a_layer_refuses_to_solve_after_an_edge_is_added():
    net = _net()
    layer = _layer(net)
    net.add_node("c")
    net.add_edge("b", "c", kind="conduction")
    with pytest.raises(RuntimeError, match="topology.*changed"):
        layer.solve(torch.ones(1, dtype=F64), differentiable=False)


def test_a_layer_refuses_component_calls_after_a_node_is_added():
    net = _net()
    layer = _layer(net)
    net.add_node("c")
    with pytest.raises(RuntimeError, match="topology.*changed"):
        layer.flows(torch.zeros(2, dtype=F64), {})


def test_a_new_layer_over_the_changed_network_works():
    net = _net()
    _layer(net)
    net.add_node("c")
    net.add_edge("b", "c", kind="conduction")
    fresh = KirchhoffLayer(
        net, "heat", [Conductance(torch.ones(2, dtype=F64))], boundary=["a"]
    )
    phi, q = fresh.solve(torch.ones(1, dtype=F64), differentiable=False)
    assert torch.allclose(phi, torch.ones(3, dtype=F64))
    assert q.shape == (2,)


def test_re_adding_an_existing_node_with_attributes_is_not_a_topology_change():
    net = _net()
    layer = _layer(net)
    net.add_node("b", volume=3.0)
    phi, q = layer.solve(torch.ones(1, dtype=F64), differentiable=False)
    assert torch.allclose(q, torch.zeros(1, dtype=F64))


# ------------------------------------------------------------------ H01
class _BufferLaw(ConstitutiveLaw):
    def __init__(self, g):
        super().__init__("conduction")
        self.register_buffer("g", g)

    def flow(self, dp, drivers=None):
        return self.g * dp

    def dflow(self, dp, drivers=None):
        return self.g * torch.ones_like(dp)

    def linear_init(self, drivers=None):
        return torch.zeros_like(self.g), self.g


class _ContainerLaw(_BufferLaw):
    def __init__(self, g):
        ConstitutiveLaw.__init__(self, "conduction")
        self.coeffs = {"g": [g]}
        self.g = g.detach()

    def flow(self, dp, drivers=None):
        return self.coeffs["g"][0] * dp

    def dflow(self, dp, drivers=None):
        return self.coeffs["g"][0] * torch.ones_like(dp)

    def linear_init(self, drivers=None):
        g = self.coeffs["g"][0]
        return torch.zeros_like(g), g


@pytest.mark.parametrize("law_class", [_BufferLaw, _ContainerLaw])
def test_a_gradient_tensor_in_a_buffer_or_container_is_refused(law_class):
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [law_class(g)], boundary=["a"])
    with pytest.raises(ValueError, match="not a registered nn.Parameter"):
        layer.solve(torch.zeros(1, dtype=F64), sources=torch.tensor([0.0, 1.0], dtype=F64))


# H01: the exact check (`KirchhoffLayer._require_threaded`) evaluates the residual with every
# threaded tensor detached, so a tensor reached by any other route is refused.
@dataclasses.dataclass
class _Holder:
    g: torch.Tensor


class _ObjectLaw(ConstitutiveLaw):
    """Holds its coefficient inside a plain object (a dataclass)."""

    def __init__(self, g):
        super().__init__("conduction")
        self.h = _Holder(g)

    def flow(self, dp, drivers=None):
        return self.h.g * dp

    def linear_init(self, drivers=None):
        return torch.zeros_like(self.h.g), self.h.g.detach()


class _ModuleListLaw(ConstitutiveLaw):
    """Holds a learnable law in a plain Python list: its parameter is not registered."""

    def __init__(self, g):
        super().__init__("conduction")
        self.parts = [Conductance(torch.nn.Parameter(g))]

    def flow(self, dp, drivers=None):
        return self.parts[0].flow(dp)

    def linear_init(self, drivers=None):
        return self.parts[0].linear_init()


class _ClosureLaw(ConstitutiveLaw):
    """Captures its coefficient in a closure, where no attribute scan can see it."""

    def __init__(self, g):
        super().__init__("conduction")
        self._coefficient = lambda: g

    def flow(self, dp, drivers=None):
        return self._coefficient() * dp

    def linear_init(self, drivers=None):
        g = self._coefficient().detach()
        return torch.zeros_like(g), g


def _solve_unit_source(layer):
    return layer.solve(torch.zeros(1, dtype=F64), sources=torch.tensor([0.0, 1.0], dtype=F64))


def test_a_gradient_tensor_in_a_plain_object_attribute_is_refused():
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [_ObjectLaw(g)], boundary=["a"])
    with pytest.raises(ValueError, match=r"'h\.g'.*not a registered nn.Parameter"):
        _solve_unit_source(layer)


def test_a_module_in_a_plain_list_is_refused():
    layer = KirchhoffLayer(
        _net(), "heat", [_ModuleListLaw(torch.tensor([2.0], dtype=F64))], boundary=["a"]
    )
    with pytest.raises(ValueError, match=r"parts\[0\]\._parameters\['g'\]"):
        _solve_unit_source(layer)


def test_a_gradient_tensor_captured_by_a_closure_is_refused():
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [_ClosureLaw(g)], boundary=["a"])
    with pytest.raises(ValueError, match="residual depends on a tensor"):
        _solve_unit_source(layer)


def test_threaded_tensors_are_not_refused():
    """No false positive: a ParameterList, a grad-requiring driver and boundary, and a
    numeric table too long to scan all reach the residual only through `Function.apply`."""

    class _ParameterListLaw(ConstitutiveLaw):
        def __init__(self, g):
            super().__init__("conduction")
            self.gs = torch.nn.ParameterList([torch.nn.Parameter(g)])
            self.table = [float(i) for i in range(100_000)]

        def flow(self, dp, drivers=None):
            return self.gs[0] * drivers["scale"] * dp

        def linear_init(self, drivers=None):
            return torch.zeros_like(self.gs[0]), (self.gs[0] * drivers["scale"]).detach()

    law = _ParameterListLaw(torch.tensor([2.0], dtype=F64))
    layer = KirchhoffLayer(_net(), "heat", [law], boundary=["a"])
    scale = torch.tensor([1.0], dtype=F64, requires_grad=True)
    pb = torch.tensor([0.5], dtype=F64, requires_grad=True)
    phi, _ = layer.solve(pb, {"scale": scale}, sources=torch.tensor([0.0, 1.0], dtype=F64))
    # phi_b = pb + 1 / (g * scale)
    g_grad, s_grad, pb_grad = torch.autograd.grad(phi[1], (law.gs[0], scale, pb))
    assert torch.allclose(g_grad, torch.tensor([-0.25], dtype=F64))
    assert torch.allclose(s_grad, torch.tensor([-0.5], dtype=F64))
    assert torch.allclose(pb_grad, torch.tensor([1.0], dtype=F64))


class _BranchLaw(ConstitutiveLaw):
    """Unit slope for |dp| <= 1, `sign(dp) (1 + g (|dp| - 1))` beyond: `g` is read only on a
    branch the starting guess dp = 0 does not reach. `g` is held as `hold` says: a plain
    attribute, a closure, or a registered parameter."""

    def __init__(self, g, hold):
        super().__init__("conduction")
        if hold == "attribute":
            self.g = g
            self._g = lambda: self.g
        elif hold == "closure":
            self._g = lambda: g
        else:
            self.g = torch.nn.Parameter(g)
            self._g = lambda: self.g

    def flow(self, dp, drivers=None):
        if bool((dp.abs() > 1).any()):
            return dp.sign() * (1 + self._g() * (dp.abs() - 1))
        return dp

    def dflow(self, dp, drivers=None):
        return torch.where(dp.abs() > 1, self._g().detach(), torch.ones_like(dp))

    def linear_init(self, drivers=None):
        g = self._g().detach()
        return torch.zeros_like(g), torch.ones_like(g)


def _solve_from_zero(law):
    layer = KirchhoffLayer(_net(), "heat", [law], boundary=["a"])
    return layer.solve(
        torch.zeros(1, dtype=F64),
        sources=torch.tensor([0.0, 3.0], dtype=F64),
        phi0=torch.zeros(1, dtype=F64),
    )


def test_a_plain_gradient_attribute_is_refused_before_solving():
    """The review's reproduction: an unregistered `g` gave dq/dg = -1 (truth 0)."""
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    with pytest.raises(ValueError, match=r"tensor attribute 'g'.*not a registered nn.Parameter"):
        _solve_from_zero(_BranchLaw(g, "attribute"))


def test_a_closure_read_only_at_the_root_is_refused():
    """The starting guess never reaches `g`; the root does, so the solve is refused there."""
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    with pytest.raises(ValueError, match="residual depends on a tensor"):
        _solve_from_zero(_BranchLaw(g, "closure"))


def test_a_closure_is_refused_where_nothing_differentiable_reaches_the_solve():
    """`on_failure="return_primal"` with nothing threaded requiring grad: the solve is not
    differentiated (the primal Newton), and still refuses a closure-captured gradient tensor
    at the residual that reads it, the root here."""
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [_BranchLaw(g, "closure")], boundary=["a"])
    with pytest.raises(ValueError, match="residual depends on a tensor"):
        layer.solve(torch.zeros(1, dtype=F64), sources=torch.tensor([0.0, 3.0], dtype=F64),
                    phi0=torch.zeros(1, dtype=F64), on_failure="return_primal",
                    diagnostics={})


def test_a_registered_parameter_read_only_at_the_root_is_threaded():
    """No false positive, and the gradient is right: the source fixes the flow (dq/dg = 0)
    and phi_b = 1 + (3 - 1) / g (dphi_b/dg = -2 / g**2 = -0.5)."""
    law = _BranchLaw(torch.tensor([2.0], dtype=F64), "parameter")
    phi, q = _solve_from_zero(law)
    torch.testing.assert_close(phi, torch.tensor([0.0, 2.0], dtype=F64))
    dq, = torch.autograd.grad(q.sum(), law.g, retain_graph=True)
    dphi, = torch.autograd.grad(phi[1], law.g)
    torch.testing.assert_close(dq, torch.tensor([0.0], dtype=F64), rtol=0, atol=1e-12)
    torch.testing.assert_close(dphi, torch.tensor([-0.5], dtype=F64))


class _ImplicitBranchLaw(ImplicitLaw):
    """`_BranchLaw` as an implicit law, so the layer is solved with flow unknowns: q = f(dp)."""

    def __init__(self, g):
        super().__init__("conduction")
        self._g = lambda: g

    def residual(self, dp, q, drivers, ctx):
        f = dp
        if bool((dp.abs() > 1).any()):
            f = dp.sign() * (1 + self._g() * (dp.abs() - 1))
        return q - f


def test_a_closure_read_only_at_the_root_is_refused_on_the_mna_path():
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [_ImplicitBranchLaw(g)], boundary=["a"])
    assert layer.implicit_edges.numel() > 0
    sources = torch.tensor([0.0, 3.0], dtype=F64)
    # the primal solve reaches the branch that reads g
    phi, _ = layer.solve(torch.zeros(1, dtype=F64), sources=sources, differentiable=False)
    assert float(phi[1].abs()) > 1
    with pytest.raises(ValueError, match="residual depends on a tensor"):
        layer.solve(torch.zeros(1, dtype=F64), sources=sources)


def test_a_closure_is_refused_on_the_mna_path_where_nothing_differentiable_reaches():
    """`on_failure="return_primal"` with nothing threaded requiring grad on a layer with flow
    unknowns: the primal solve, which still refuses the closure-captured tensor by name
    rather than returning a detached result."""
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [_ImplicitBranchLaw(g)], boundary=["a"])
    sources = torch.tensor([0.0, 3.0], dtype=F64)
    with pytest.raises(ValueError, match="residual depends on a tensor"):
        layer.solve(torch.zeros(1, dtype=F64), sources=sources, on_failure="return_primal",
                    diagnostics={})
    with torch.no_grad():        # as on the fast path, under no_grad too
        with pytest.raises(ValueError, match="residual depends on a tensor"):
            layer.solve(torch.zeros(1, dtype=F64), sources=sources,
                        on_failure="return_primal", diagnostics={})


# ------------------------------------------------------------------ R08: old pickles
def _as_if_pickled_before_the_guard(layer: KirchhoffLayer) -> KirchhoffLayer:
    """Round-trip `layer` through pickle with the attributes older versions lacked removed."""
    del layer.net.__dict__["_version"]
    del layer.__dict__["_topology_version"]
    return pickle.loads(pickle.dumps(layer))


def test_an_old_pickle_without_topology_versions_still_solves():
    layer = _as_if_pickled_before_the_guard(_layer(_net()))
    assert layer.net.topology_version == 0
    phi, q = layer.solve(torch.ones(1, dtype=F64), differentiable=False)
    assert torch.allclose(q, torch.zeros(1, dtype=F64))


def test_an_old_pickled_network_counts_changes_and_its_layer_refuses_them():
    layer = _as_if_pickled_before_the_guard(_layer(_net()))
    layer.net.add_node("c")
    assert layer.net.topology_version == 1
    with pytest.raises(RuntimeError, match="topology.*changed"):
        layer.solve(torch.ones(1, dtype=F64), differentiable=False)


# The attribute scan is cached per layer (`KirchhoffLayer._held_scan`): a change to anything it
# read after a first, clean solve must still be refused.
@pytest.mark.parametrize("change", ["attribute", "requires_grad", "list", "nested", "buffer"])
def test_the_cached_attribute_scan_sees_a_later_change(change):
    holder = _Holder(torch.tensor([2.0], dtype=F64))
    law = _ObjectLaw(holder.g)
    law.extra = []
    layer = KirchhoffLayer(_net(), "heat", [law], boundary=["a"])
    _solve_unit_source(layer)                                  # clean: cached
    _solve_unit_source(layer)                                  # served from the cache
    g = torch.tensor([3.0], dtype=F64, requires_grad=True)
    if change == "attribute":
        law.other = g
    elif change == "requires_grad":
        law.h.g.requires_grad_(True)
    elif change == "list":
        law.extra.append(g)
    elif change == "nested":
        law.h.g = g
    else:
        law.register_buffer("b", g)
    with pytest.raises(ValueError, match="not a registered nn.Parameter"):
        _solve_unit_source(layer)


def test_the_closure_refusal_takes_the_primal_newton(monkeypatch):
    """The refusal above comes from the primal Newton: no `FunctionalLaws` is built."""
    from noodl.layers import kirchhoff

    built: list = []
    init = kirchhoff.FunctionalLaws.__init__

    def spy(self, *args, **kwargs):
        built.append(1)
        init(self, *args, **kwargs)

    monkeypatch.setattr(kirchhoff.FunctionalLaws, "__init__", spy)
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    layer = KirchhoffLayer(_net(), "heat", [_BranchLaw(g, "closure")], boundary=["a"])
    with pytest.raises(ValueError, match="residual depends on a tensor"):
        layer.solve(torch.zeros(1, dtype=F64), sources=torch.tensor([0.0, 3.0], dtype=F64),
                    phi0=torch.zeros(1, dtype=F64), on_failure="return_primal",
                    diagnostics={})
    assert not built


class _ForwardLaw(ConstitutiveLaw):
    """A law whose module call is not its `flow`: `forward` reads a closure-captured tensor
    that `flow` does not."""

    def __init__(self, g):
        super().__init__("conduction")
        self._g = lambda: g

    def flow(self, dp, drivers=None):
        return dp

    def forward(self, dp, drivers=None, ctx=None):
        return self._g() * dp

    def linear_init(self, drivers=None):
        return torch.zeros(1, dtype=F64), torch.ones(1, dtype=F64)


@pytest.mark.parametrize("how", ["forward", "hook"])
def test_a_law_whose_module_call_is_not_its_flow_is_called_as_a_module(how):
    """The fast path's primal Newton calls a law's own `flow` where that is its module call;
    where they may differ (`forward` overridden or hooked) it calls the module, as the
    differentiated solve does, so a solve that nothing differentiable reaches refuses the
    captured tensor as the differentiated solve would."""
    g = torch.tensor([2.0], dtype=F64, requires_grad=True)
    if how == "forward":
        law = _ForwardLaw(g)
    else:
        law = Conductance(torch.tensor([1.0], dtype=F64))
        law.register_forward_hook(lambda module, args, out: g * out)
    layer = KirchhoffLayer(_net(), "heat", [law], boundary=["a"])
    assert not layer._plain_calls()
    with pytest.raises(ValueError, match="residual depends on a tensor"):
        layer.solve(torch.zeros(1, dtype=F64), sources=torch.tensor([0.0, 1.0], dtype=F64),
                    on_failure="return_primal", diagnostics={})


def _hooked_power_law(learnable=False):
    """`2 C dp |dp|^0.5`: a power law whose forward hook doubles its flow, on a heat layer
    a -> b with 3 into b. The root is the scaled law's, `phi_b = (3 / (2 C))^(1 / 1.5)`."""
    law = PowerLaw(torch.tensor([1.0], dtype=F64), 1.5, kind="conduction",
                   learnable=learnable)
    law.register_forward_hook(lambda module, args, out: 2.0 * out)
    return law, KirchhoffLayer(_net(), "heat", [law], boundary=["a"])


HOOKED_SOURCES = torch.tensor([0.0, 3.0], dtype=F64)


def test_a_hooked_law_is_solved_on_its_module_call_on_the_fast_path():
    """The module call, hook included, is the user's law: on the fast path the residual
    and the slopes both come from it, with `differentiable=False`, a `return_primal` solve
    that nothing differentiable reaches and the differentiated solve alike. Before, the
    slopes were the law's own `dflow`, half the hooked one: `differentiable=False` solved
    the unhooked law (residual from `flow`), and the other two did not converge."""
    expected = 1.5 ** (1 / 1.5)
    out = {}
    for name, kw in (("primal", {"differentiable": False}),
                     ("return_primal", {"on_failure": "return_primal"}),
                     ("differentiated", {})):
        _, layer = _hooked_power_law()
        diag: dict = {}
        out[name] = layer.solve(torch.zeros(1, dtype=F64), sources=HOOKED_SOURCES,
                                diagnostics=diag, atol=1e-12, rtol=0.0, **kw), diag
        assert bool(diag["converged"].all()), name
        assert float(out[name][0][0][1]) == pytest.approx(expected, rel=1e-12), name
    (phi_p, q_p), diag_p = out["primal"]
    for name in ("return_primal", "differentiated"):
        (phi, q), diag = out[name]
        assert torch.equal(phi.detach(), phi_p) and torch.equal(q.detach(), q_p), name
        assert diag["newton_iterations"] == diag_p["newton_iterations"], name


def test_the_gradient_through_a_hooked_law_is_the_module_calls():
    """The adjoint's Jacobian is the hooked law's slope too, so the gradient is that of the
    scaled root: `d phi_b / d C = -phi_b / (1.5 C)`."""
    law, layer = _hooked_power_law(learnable=True)
    phi, _ = layer.solve(torch.zeros(1, dtype=F64), sources=HOOKED_SOURCES, atol=1e-12,
                         rtol=0.0)
    phi[1].backward()
    expected = -(1.5 ** (1 / 1.5)) / 1.5
    assert float(law.C.grad) == pytest.approx(expected, rel=1e-10)


# ------------------------------------------------- a hooked law is its module call everywhere
class _Leak(NodeWithdrawal):
    """`g phi` leaving at its nodes, with its own exact slope."""

    def __init__(self, nodes, g):
        super().__init__(nodes)
        self.g = torch.nn.Parameter(torch.tensor(g, dtype=F64), requires_grad=False)

    def flow(self, phi, drivers=None):
        return self.g * phi

    def dflow(self, phi, drivers=None):
        return self.g * torch.ones_like(phi)


def _chain(*kinds) -> Network:
    """a - b - c, one edge per kind in that order."""
    net = Network(dtype=F64)
    for n in "abc":
        net.add_node(n)
    for (u, v), kind in zip((("a", "b"), ("b", "c")), kinds, strict=True):
        net.add_edge(u, v, kind=kind)
    return net


def _hooked_chain(law):
    """`law` on both edges of a - b - c, boundary a, its flow doubled by a forward hook,
    and a leak `0.5 phi` at c that a hook triples."""
    law.register_forward_hook(lambda module, args, out: 2.0 * out)
    leak = _Leak([2], 0.5)
    leak.register_forward_hook(lambda module, args, out: 3.0 * out)
    layer = KirchhoffLayer(_chain("conduction", "conduction"), "heat", [law],
                           boundary=["a"], withdrawals=[leak])
    assert not layer._plain_calls()
    return layer


CHAIN_PB = torch.tensor([2.0], dtype=F64)
CHAIN_S = torch.tensor([0.0, 0.5, 0.0], dtype=F64)


def test_the_public_residual_and_slopes_of_a_hooked_law_are_its_module_calls():
    """`residual`, `flows`, `jacobian`, `dflows` and `adjoint` all follow the module call
    (hooks included): the residual is zero at the hooked root, `jacobian` is autograd of
    `residual`, `dflows` autograd of the hooked law, and `adjoint` solves with that
    Jacobian's transpose."""
    law = PowerLaw(torch.tensor([1.0], dtype=F64), 1.5, kind="conduction")
    layer = _hooked_chain(law)
    phi, q = layer.solve(CHAIN_PB, sources=CHAIN_S, atol=1e-13, rtol=0.0,
                         differentiable=False)
    phi_i = phi[1:]
    r = layer.residual(phi_i, CHAIN_PB, {}, CHAIN_S)
    assert float(r.abs().max()) < 1e-12
    torch.testing.assert_close(layer.flows(phi, {}), q, rtol=1e-14, atol=0)
    J = torch.autograd.functional.jacobian(
        lambda x: layer.residual(x, CHAIN_PB, {}, CHAIN_S), phi_i)
    torch.testing.assert_close(layer.jacobian(phi_i, CHAIN_PB, {}), J, rtol=1e-12, atol=0)
    dp = layer.dp(phi, {})
    slope = torch.autograd.functional.jacobian(lambda d: law(d, {}), dp).diagonal()
    torch.testing.assert_close(layer.dflows(phi, {}), slope, rtol=1e-12, atol=0)
    g = torch.tensor([1.0, -0.5], dtype=F64)
    lam = layer.adjoint(phi_i, CHAIN_PB, {}, g)
    torch.testing.assert_close(J.T @ lam, g, rtol=1e-10, atol=1e-12)


def test_a_differentiated_solve_returns_the_hooked_laws_flows():
    """With a learnable parameter the differentiated solve's flows come from the laws
    called with their graph: the hooked ones, as the primal solve's."""
    law = PowerLaw(torch.tensor([1.0], dtype=F64), 1.5, kind="conduction", learnable=True)
    layer = _hooked_chain(law)
    phi, q = layer.solve(CHAIN_PB, sources=CHAIN_S, atol=1e-13, rtol=0.0)
    assert q.requires_grad
    phi_p, q_p = layer.solve(CHAIN_PB, sources=CHAIN_S, atol=1e-13, rtol=0.0,
                             differentiable=False)
    torch.testing.assert_close(q.detach(), q_p, rtol=1e-12, atol=0)
    torch.testing.assert_close(q.detach(), layer.flows(phi_p, {}).detach(), rtol=1e-12,
                               atol=0)


def test_the_linear_guess_of_a_hooked_linear_law_is_the_hooked_root():
    """A linear law's tangent at zero is the law, so the linear guess is the solution: of
    the hooked law, its tangent taken from the module call (the doubled conductance and the
    tripled leak), not from its own `linear_init`."""
    layer = _hooked_chain(Conductance(torch.tensor([1.0], dtype=F64), kind="conduction"))
    phi, _ = layer.solve(CHAIN_PB, sources=CHAIN_S, atol=1e-13, rtol=0.0,
                         differentiable=False)
    torch.testing.assert_close(layer.linear_init(CHAIN_PB, {}, CHAIN_S), phi[1:],
                               rtol=1e-12, atol=0)


class _CtxConductance(ConstitutiveLaw):
    """`G dp`, written with a context it does not read: a law that reads the context, with
    no `jacobian` of its own."""

    def __init__(self, G):
        super().__init__("conduction")
        self.G = self._param(torch.tensor([G], dtype=F64), True)

    def flow(self, dp, drivers=None, ctx=None):
        return self.G * dp


@pytest.mark.parametrize("how", ["hooked", "context"])
def test_the_public_linear_guess_is_differentiable_in_a_tangent_taken_by_autograd(how):
    """The public `linear_init` keeps the graph of a tangent's slope taken by autograd (a
    hooked law's, a context law's): the guess `3 / (s G)` at b has `d / d G = -3 / (s G^2)`,
    as for a plain conductance, whose `linear_init` carries its graph."""
    if how == "hooked":
        law = Conductance(torch.tensor([1.0], dtype=F64), kind="conduction", learnable=True)
        law.register_forward_hook(lambda module, args, out: 2.0 * out)
        G, scale = law.g, 2.0
    else:
        law = _CtxConductance(1.0)
        G, scale = law.G, 1.0
    layer = KirchhoffLayer(_net(), "heat", [law], boundary=["a"])
    guess = layer.linear_init(torch.zeros(1, dtype=F64), {}, HOOKED_SOURCES)
    assert float(guess[0].detach()) == pytest.approx(3.0 / scale, rel=1e-14)
    (grad,) = torch.autograd.grad(guess.sum(), G)
    assert float(grad) == pytest.approx(-3.0 / scale, rel=1e-12)
    # a solve takes the same tangent without a graph: its root is unchanged
    phi, _ = layer.solve(torch.zeros(1, dtype=F64), sources=HOOKED_SOURCES,
                         differentiable=False)
    assert float(phi[1]) == pytest.approx(3.0 / scale, rel=1e-12)


def test_the_dp_dependence_error_names_a_hook_on_a_hooked_law():
    """A hooked law whose call loses its dependence on `dp` is told the hook may be why."""
    law = Conductance(torch.tensor([1.0], dtype=F64), kind="conduction")
    law.register_forward_hook(lambda module, args, out: out.detach())
    layer = KirchhoffLayer(_net(), "heat", [law], boundary=["a"])
    with pytest.raises(RuntimeError, match="flow\\(\\) or a forward hook on it is dropping"):
        layer.dflows(torch.tensor([1.0, 0.0], dtype=F64), {})


def test_the_linear_guess_takes_a_hooked_laws_slope_per_instance():
    """A hook that reads a batched driver gives each instance its own tangent: the guess of
    the conductance scaled by `s` is `3 / s` at b, instance by instance."""
    law = Conductance(torch.tensor([1.0], dtype=F64), kind="conduction")
    law.register_forward_hook(lambda module, args, out: args[1]["s"] * out)
    layer = KirchhoffLayer(_net(), "heat", [law], boundary=["a"])
    drivers = {"s": torch.tensor([[2.0], [3.0]], dtype=F64)}
    guess = layer.linear_init(torch.zeros(1, dtype=F64), drivers, HOOKED_SOURCES)
    torch.testing.assert_close(guess, torch.tensor([[1.5], [1.0]], dtype=F64), rtol=1e-14,
                               atol=0)


def test_the_storage_ramp_settles_on_the_hooked_law(monkeypatch):
    """The fallback's storage ramp takes the solve's own residual and Jacobian: the hooked
    law's, zero at the hooked root, with the hooked slope there."""
    import noodl.layers._fallback as fallback

    _, layer = _hooked_power_law()
    root = torch.tensor([1.5 ** (1 / 1.5)], dtype=F64)
    seen = {}

    def spy(residual, operator, x0, **kwargs):
        seen["r"] = residual(root)
        seen["op"] = operator(root).matvec(torch.ones(1, dtype=F64))
        kwargs["diagnostics"]["newton_iterations"] = 0
        return root.clone()

    monkeypatch.setattr(fallback, "storage_ramp", spy)
    diag: dict = {}
    layer.solve(torch.zeros(1, dtype=F64), sources=HOOKED_SOURCES, atol=1e-12, rtol=0.0,
                max_iter=1, fallback=["storage_ramp"], differentiable=False,
                diagnostics=diag)
    assert diag["fallback"] == "storage_ramp"
    assert float(seen["r"].abs().max()) < 1e-12
    # the hooked slope, 2 * 1.5 C sqrt(phi_b)
    torch.testing.assert_close(seen["op"], 3.0 * root.sqrt(), rtol=1e-12, atol=0)


class _Resist(ImplicitLaw):
    """`dp - R q = 0`, the flow an unknown."""

    def __init__(self, R, kind="loss"):
        super().__init__(kind)
        self.R = self._param(torch.tensor([R], dtype=F64), False)

    def residual(self, dp, q, drivers=None, ctx=None):
        return dp - self.R * q


def _flow_unknowns_layer(C, R, *, hooked, learnable=False):
    """a - b by a power law `C dp |dp|^0.5`, b - c by `_Resist(R)`, a at 1 and c at 0. With
    `hooked` the power law's hook doubles its flow and the resistance's subtracts `0.5 q`
    from its residual: the layer of `2 C` and `R + 0.5` unhooked."""
    law = PowerLaw(torch.tensor([C], dtype=F64), 1.5, kind="conduction", learnable=learnable)
    res = _Resist(R)
    if hooked:
        law.register_forward_hook(lambda module, args, out: 2.0 * out)
        res.register_forward_hook(lambda module, args, out: out - 0.5 * args[1])
    layer = KirchhoffLayer(_chain("conduction", "loss"), "w", [law, res],
                           boundary=["a", "c"])
    assert layer._unknowns is not None
    return law, layer


FU_PB = torch.tensor([1.0, 0.0], dtype=F64)


def _fu_root(C2, R2):
    """`phi_b` with `C2 (1 - p)^1.5 = p / R2`, by bisection to the last bit."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if C2 * (1 - mid) ** 1.5 - mid / R2 > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_a_flow_unknowns_layer_solves_a_hooked_law_on_its_module_call():
    """On a layer with flow unknowns the residual (a law's flow, an implicit law's
    residual) and the Jacobian both come from the module call: the hooked layer converges
    to the root of the equivalent unhooked one in as many Newton iterations."""
    p = _fu_root(2.0, 1.5)
    out = {}
    for hooked, (C, R) in ((True, (1.0, 1.0)), (False, (2.0, 1.5))):
        _, layer = _flow_unknowns_layer(C, R, hooked=hooked)
        diag: dict = {}
        phi, q = layer.solve(FU_PB, atol=1e-13, rtol=0.0, differentiable=False,
                             diagnostics=diag)
        assert bool(diag["converged"].all())
        assert float(phi[1]) == pytest.approx(p, rel=1e-12)
        out[hooked] = phi, q, diag["newton_iterations"]
    assert out[True][2] == out[False][2]
    torch.testing.assert_close(out[True][1], out[False][1], rtol=1e-12, atol=0)


def test_the_public_slopes_and_linear_guess_of_a_hooked_flow_unknowns_layer():
    """`jacobian` is autograd of the hooked `residual`, `adjoint` solves with its
    transpose, `dflows` gives the hooked slopes (the implied `1 / (R + 0.5)` of the
    resistance), and the linear guess is that of the equivalent unhooked layer."""
    _, layer = _flow_unknowns_layer(1.0, 1.0, hooked=True)
    phi, q = layer.solve(FU_PB, atol=1e-13, rtol=0.0, differentiable=False)
    x = torch.cat([phi[1:2], q[1:]])

    def residual(x):
        return layer.residual(x[:1], FU_PB, {}, None, q=torch.cat([q[:1], x[1:]]))

    J = torch.autograd.functional.jacobian(residual, x)
    torch.testing.assert_close(layer.jacobian(x[:1], FU_PB, {}, q=q), J, rtol=1e-12, atol=0)
    g = torch.tensor([1.0, -0.5], dtype=F64)
    torch.testing.assert_close(J.T @ layer.adjoint(x[:1], FU_PB, {}, g, q=q), g,
                               rtol=1e-10, atol=1e-12)
    dp = float(phi[0] - phi[1])
    torch.testing.assert_close(layer.dflows(phi, {}, q=q),
                               torch.tensor([2.0 * 1.5 * dp ** 0.5, 1 / 1.5], dtype=F64),
                               rtol=1e-12, atol=0)
    _, plain = _flow_unknowns_layer(2.0, 1.5, hooked=False)
    for a, b in zip(layer.linear_init(FU_PB, {}, None, q=True),
                    plain.linear_init(FU_PB, {}, None, q=True), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=0)


def test_the_gradient_through_a_hooked_law_with_flow_unknowns():
    """The adjoint of a layer with flow unknowns takes the hooked Jacobian: `d phi_b / d C`
    of `2 C (1 - p)^1.5 = p / 1.5`, by implicit differentiation."""
    law, layer = _flow_unknowns_layer(1.0, 1.0, hooked=True, learnable=True)
    phi, _ = layer.solve(FU_PB, atol=1e-13, rtol=0.0)
    phi[1].backward()
    p = _fu_root(2.0, 1.5)
    expected = -2 * (1 - p) ** 1.5 / (-3 * (1 - p) ** 0.5 - 1 / 1.5)
    assert float(law.C.grad) == pytest.approx(expected, rel=1e-10)


class _Sum(ConstitutiveLaw):
    """The sum of two laws' flows, with the exact slope."""

    def __init__(self, a, b):
        super().__init__(a.kind)
        self.a, self.b = a, b

    def flow(self, dp, drivers=None):
        return self.a.flow(dp, drivers) + self.b.flow(dp, drivers)

    def dflow(self, dp, drivers=None):
        return self.a.dflow(dp, drivers) + self.b.dflow(dp, drivers)


@pytest.mark.parametrize("kw", [{"differentiable": False}, {}])
def test_a_hook_that_makes_a_dp_independent_law_depend_on_dp_is_differentiated(kw):
    """`dp_independent` declares the law's own flow constant; a hook on it may add a term in
    `dp`, and the slope is then the module call's by autograd (`0.25` here), not the
    declared zero: the hooked layer solves as the equivalent unhooked one, iteration for
    iteration."""
    def layer_of(hooked):
        net = Network(dtype=F64)
        net.add_node("a")
        net.add_node("b")
        net.add_edge("a", "b", kind="conduction")
        net.add_edge("a", "b", kind="fixed")
        cond = PowerLaw(torch.tensor([1.0], dtype=F64), 1.5, kind="conduction")
        fixed = FixedFlow(torch.tensor([0.5], dtype=F64), kind="fixed")
        if hooked:
            fixed.register_forward_hook(lambda module, args, out: out + 0.25 * args[0])
        else:
            cond = _Sum(cond, Conductance(torch.tensor([0.25], dtype=F64), kind="conduction"))
        return KirchhoffLayer(net, "heat", [cond, fixed], boundary=["a"])

    out = {}
    for hooked in (True, False):
        layer = layer_of(hooked)
        diag: dict = {}
        phi, _ = layer.solve(torch.zeros(1, dtype=F64), sources=HOOKED_SOURCES, atol=1e-13,
                             rtol=0.0, diagnostics=diag, **kw)
        assert bool(diag["converged"].all())
        out[hooked] = phi.detach(), diag["newton_iterations"]
        if hooked:
            assert float(layer.dflows(phi.detach(), {})[1]) == 0.25
    assert out[True][1] == out[False][1]
    torch.testing.assert_close(out[True][0], out[False][0], rtol=1e-12, atol=0)


def test_a_hooked_dp_independent_law_still_has_its_declaration_checked():
    """The declaration is still checked against the law's own `dflow`, hook or not."""
    class _Wrong(FixedFlow):
        def dflow(self, dp, drivers=None):
            return torch.ones_like(dp)

    law = _Wrong(torch.tensor([0.5], dtype=F64), kind="conduction")
    law.register_forward_hook(lambda module, args, out: out + args[0])
    layer = KirchhoffLayer(_net(), "heat", [law], boundary=["a"])
    with pytest.raises(RuntimeError, match="declares dp_independent = True"):
        layer.solve(torch.zeros(1, dtype=F64), sources=HOOKED_SOURCES, differentiable=False)


def test_the_storage_columns_are_taken_once_with_the_graph_of_the_step_start():
    """A storage step's columns are taken when the step is built (`_storage_step`), with the
    graph of a grad-requiring `phi_prev`, so a read under `no_grad` (Newton) and one under
    grad mode (the result's storage flows) both see the same column, graph included."""
    import dataclasses

    from noodl.storage import LinearStorage

    layer = KirchhoffLayer(_net(), "heat", [Conductance(torch.tensor([1.0], dtype=F64))],
                           boundary=["a"],
                           storage=[LinearStorage([1], torch.tensor([0.5], dtype=F64))])
    st = layer._storage[0]
    phi_prev = torch.tensor([0.0, 2.0], dtype=F64, requires_grad=True)
    step = layer._storage_step(phi_prev, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.columns = ()
    phi = torch.tensor([0.0, 3.0], dtype=F64)
    with torch.no_grad():
        _, under_no_grad, previous = layer._storage_args(0, st, phi, step)
    assert previous is None
    _, start, _ = layer._storage_args(0, st, phi, step)
    assert start is under_no_grad and start.requires_grad
    (grad,) = torch.autograd.grad(start.sum(), phi_prev)
    assert torch.equal(grad, torch.tensor([0.0, 1.0], dtype=F64))

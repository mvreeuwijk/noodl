"""Refusals a KirchhoffLayer owes its caller: a stale topology and unreachable gradients."""

import dataclasses
import pickle

import pytest
import torch

from noodl.elements import Conductance
from noodl.elements.base import ConstitutiveLaw, ImplicitLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

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
    """`_BranchLaw` as an implicit law, so the layer is solved by MNA: q = f(dp)."""

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
def test_a_law_whose_module_call_is_not_its_flow_keeps_the_threaded_solve(how):
    """The primal Newton calls `flow`, the threaded one the module (`forward` and its
    hooks): where they may differ the solve stays threaded and refuses, as before."""
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


def test_the_storage_column_cache_keeps_the_graph_of_a_grad_requiring_start():
    """A step-start column read under no_grad is not kept when the step's `phi_prev`
    requires grad, so a later read under grad mode carries the graph."""
    from noodl.layers.kirchhoff import _StorageStep
    from noodl.storage import LinearStorage

    layer = KirchhoffLayer(_net(), "heat", [Conductance(torch.tensor([1.0], dtype=F64))],
                           boundary=["a"],
                           storage=[LinearStorage([1], torch.tensor([0.5], dtype=F64))])
    st = layer._storage[0]
    phi_prev = torch.tensor([0.0, 2.0], dtype=F64, requires_grad=True)
    step = _StorageStep(phi_prev, None)
    phi = torch.tensor([0.0, 3.0], dtype=F64)
    with torch.no_grad():
        _, start, _ = layer._storage_args(0, st, phi, step)
    assert not start.requires_grad
    _, start, _ = layer._storage_args(0, st, phi, step)
    assert start.requires_grad
    (grad,) = torch.autograd.grad(start.sum(), phi_prev)
    assert torch.equal(grad, torch.tensor([0.0, 1.0], dtype=F64))
    # a graph-free step is read once
    plain = _StorageStep(phi_prev.detach(), None)
    first = layer._storage_args(0, st, phi, plain)[1]
    assert layer._storage_args(0, st, phi, plain)[1] is first

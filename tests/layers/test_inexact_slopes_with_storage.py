"""`exact_slope` on laws and on node storage together, on the classic and the MNA path.

A law whose `dflow` is a Newton aid and a storage whose `rate_slope` is one (3x the true
slope), both declaring `exact_slope = False`, on one layer: the differentiable step solve
must give the same root as the exact twin layer (plain laws, plain `LinearStorage`) and the
same, correct gradients (`gradcheck`), because the Jacobian the adjoint uses takes autograd
for both. Also: the differentiable classic step solve that nothing reaches is the primal one,
storage flows included; and `FunctionalLaws.rebuild`'s memo hands back each call's own
step-start tensors.
"""

from __future__ import annotations

import pytest
import torch

from noodl.elements import PowerLaw
from noodl.layers._functional import FunctionalLaws
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.storage import LinearStorage
from noodl.topology import Network

F64 = torch.float64
TIGHT = dict(atol=1e-13, rtol=0.0)
DT = 0.3


def _t(values, grad=False):
    return torch.tensor(values, dtype=F64, requires_grad=grad)


class _AidedPowerLaw(PowerLaw):
    """A power law whose `dflow` is a Newton aid (twice the slope): not the derivative."""

    exact_slope = False

    def dflow(self, dp, drivers=None):
        return 2.0 * super().dflow(dp, drivers)


class _AidedStorage(LinearStorage):
    """A linear storage whose `rate_slope` is a Newton aid (3x the slope)."""

    exact_slope = False

    def rate_slope(self, phi_nodes, phi_start, previous, ctx, drivers=None):
        return 3.0 * super().rate_slope(phi_nodes, phi_start, previous, ctx, drivers)


def _net(edges):
    net = Network(dtype=F64)
    for n in ("g", "a", "b"):
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


def _layer(path: str, aided: bool, law_exact: bool = False, storage_exact: bool = False):
    """`(layer, law, storage)`: g -(r)- a -(load)- b -(r2)- g with storage at a and b on the
    classic path; on the MNA path an ideal source g -> a makes it an MNA layer, with storage at
    b. `aided` swaps in the Newton-aid law (on "load") and storage, with the given
    `exact_slope` declarations (a wrong declaration corrupts the gradients)."""
    law_cls = _AidedPowerLaw if aided else PowerLaw
    st_cls = _AidedStorage if aided else LinearStorage
    load = law_cls(_t([0.3]), _t(0.8), kind="load", dp_transition=1e-6, learnable=True)
    if aided:
        load.exact_slope = law_exact
    if path == "classic":
        net = _net([("g", "a", "r"), ("a", "b", "load"), ("b", "g", "r2")])
        laws = [PowerLaw(_t([0.5]), _t(0.65), kind="r", dp_transition=1e-6), load,
                PowerLaw(_t([0.4]), _t(0.55), kind="r2", dp_transition=1e-6)]
        storage = st_cls([1, 2], _t([0.06, 0.03]), learnable=True)
    else:
        from tests._circuit_laws import Source

        net = _net([("a", "g", "source"), ("a", "b", "r"), ("b", "g", "load")])
        laws = [Source(_t([1.0])), PowerLaw(_t([0.5]), _t(0.65), kind="r",
                                            dp_transition=1e-6), load]
        storage = st_cls([2], _t([0.03]), learnable=True)
    if aided:
        storage.exact_slope = storage_exact
    layer = KirchhoffLayer(net, "e", laws, boundary=["g"], storage=[storage])
    assert layer.mna == (path == "mna")
    return layer, load, storage


SOURCES = [0.0, 0.2, -0.1]


def _step(layer, phi_prev, flow_prev, law_C, st_C, sources, scheme):
    """One differentiable step; `law_C`/`st_C` are the parameters (perturbed in place by
    gradcheck), passed only so gradcheck sees them as inputs."""
    q_prev = torch.zeros(3, dtype=F64)    # every network here has three edges
    kw = dict(dt=DT, scheme=scheme, history=(phi_prev, q_prev))
    if scheme == "trapezoidal":
        kw["storage_flow"] = flow_prev
    phi, q = layer.solve(_t([0.0]), {}, sources, differentiable=True, **kw, **TIGHT)
    w = layer.storage_flows(phi, {}, **kw)
    return torch.cat([phi[..., layer.interior], q, w], dim=-1)


def _inputs(layer, load, storage):
    n_st = len(storage.nodes)
    return (_t([0.0, 0.4, -0.2], grad=True), _t([0.05] * n_st, grad=True), load.C,
            storage.C, _t(SOURCES, grad=True))


@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
@pytest.mark.parametrize("path", ["classic", "mna"])
def test_inexact_law_and_storage_slopes_keep_the_gradients_exact(path, scheme):
    layer, load, storage = _layer(path, aided=True)
    twin, twin_load, twin_storage = _layer(path, aided=False)
    inputs = _inputs(layer, load, storage)
    twin_inputs = (*inputs[:2], twin_load.C, twin_storage.C, inputs[4])

    def f(*args):
        return _step(layer, *args, scheme=scheme)

    def f_twin(*args):
        return _step(twin, *args, scheme=scheme)

    out, ref = f(*inputs), f_twin(*twin_inputs)
    # the same root: the aids change Newton's path only
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-12)
    # the same gradients as the exact twin, and correct ones
    for i in range(out.shape[-1]):
        g = torch.autograd.grad(out[i], inputs, allow_unused=True, retain_graph=True)
        g_ref = torch.autograd.grad(ref[i], twin_inputs, allow_unused=True,
                                    retain_graph=True)
        for a, b in zip(g, g_ref, strict=True):
            if a is None or b is None:
                assert a is None or not a.any()
                assert b is None or not b.any()
                continue
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
    assert torch.autograd.gradcheck(f, inputs, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("path", ["classic", "mna"])
@pytest.mark.parametrize("which", ["law", "storage"])
def test_a_newton_aid_declared_exact_corrupts_the_gradient(path, which):
    """The control: each aid declared exact hands the adjoint its wrong slope, so the
    gradients leave the exact twin's (the test above would fail without the gates)."""
    layer, load, storage = _layer(path, aided=True, law_exact=(which == "law"),
                                  storage_exact=(which == "storage"))
    twin, twin_load, twin_storage = _layer(path, aided=False)
    inputs = _inputs(layer, load, storage)
    twin_inputs = (*inputs[:2], twin_load.C, twin_storage.C, inputs[4])
    out = _step(layer, *inputs, scheme="backward_euler")
    ref = _step(twin, *twin_inputs, scheme="backward_euler")
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-12)
    g = torch.autograd.grad(out[1], inputs[3])[0]       # interior potential at b, storage C
    g_ref = torch.autograd.grad(ref[1], twin_inputs[3])[0]
    with pytest.raises(AssertionError):
        torch.testing.assert_close(g, g_ref, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_a_storage_step_nothing_differentiable_reaches_is_the_primal_step(scheme):
    """The differentiable classic step returns Newton's last `(phi, q)` when no gradient is
    wanted; the storage flow computed from them is the primal step's, bit for bit."""
    layer, load, storage = _layer("classic", aided=False)
    for p in (*load.parameters(), *storage.parameters()):
        p.requires_grad_(False)
    phi_prev, flow_prev = _t([0.0, 0.4, -0.2]), _t([0.05, 0.05])
    kw = dict(dt=DT, scheme=scheme, history=(phi_prev, None))
    if scheme == "trapezoidal":
        kw["storage_flow"] = flow_prev
    out = {}
    for differentiable in (True, False):
        diag: dict = {}
        phi, q = layer.solve(_t([0.0]), {}, _t(SOURCES), differentiable=differentiable,
                             diagnostics=diag, **kw, **TIGHT)
        out[differentiable] = phi, q, layer.storage_flows(phi, {}, **kw), diag
    for a, b in zip(out[True][:3], out[False][:3], strict=True):
        assert torch.equal(a, b)
    assert out[True][3]["newton_iterations"] == out[False][3]["newton_iterations"]


def test_the_rebuild_memo_returns_each_call_s_own_step():
    """`rebuild` keeps two entries (the forward Newton alternates the residual's detached
    tuple and the operator's); the threaded step comes back with the tuple's own tensors,
    whichever entry is hit."""
    layer, *_ = _layer("classic", aided=False)
    phi_prev, flow_prev = _t([0.0, 0.4, -0.2], grad=True), _t([0.05, 0.05], grad=True)
    step = layer._step(None, DT, None, "trapezoidal", (phi_prev, None), flow_prev, {})
    fn = FunctionalLaws(layer, {}, _t(SOURCES), _t([0.0]), step)
    a = fn.all_params
    b = tuple(p.detach() for p in a)
    c = tuple(p.detach().clone() for p in a)
    for params in (a, b, a, b, c, a, b):
        *_, st = fn.rebuild(params)
        assert st.storage.phi_prev is params[-2] and st.storage.previous is params[-1]

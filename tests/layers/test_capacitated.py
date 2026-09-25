"""Tests for CapacitatedTransferLayer: hard-clip mode, registered as a first-class Model layer."""

import pytest
import torch

from noodl.layers.capacitated import CapacitatedTransferLayer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64


def _chain_net() -> Network:
    """A -> B -> C, one edge kind 'link', two edges, three nodes."""
    net = Network()
    net.add_node("A")
    net.add_node("B")
    net.add_node("C")
    net.add_edge("A", "B", kind="link")
    net.add_edge("B", "C", kind="link")
    return net


def test_construction_rejects_wrong_c_arc_shape():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    bad_c_arc = torch.tensor([1.0, 2.0, 3.0], dtype=F64)  # 3 edges, network has 2
    with pytest.raises(ValueError, match="c_arc"):
        CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=bad_c_arc)


def test_construction_rejects_wrong_s_max_shape():
    net = _chain_net()
    bad_s_max = torch.full((2,), 10.0, dtype=F64)  # 2, network has 3 nodes
    c_arc = torch.full((2,), 1.0, dtype=F64)
    with pytest.raises(ValueError, match="s_max"):
        CapacitatedTransferLayer(net, "cap", "link", s_max=bad_s_max, c_arc=c_arc)


def test_construction_rejects_unknown_mode():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    with pytest.raises(ValueError, match="mode"):
        CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="bogus")


def test_construction_rejects_smooth_mode_without_tau():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    with pytest.raises(ValueError, match="requires tau"):
        CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="smooth")


@pytest.mark.parametrize("tau", [0.0, -1e-3])
def test_construction_rejects_non_positive_tau(tau):
    """`tau == 0` divides by zero in every smooth-mode helper; `tau < 0` is worse -- it
    turns `_clip`'s softmin into a softmax and reverses `_select`'s sigmoid blend, a
    silently WRONG answer rather than a crash."""
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    with pytest.raises(ValueError, match="tau must be > 0"):
        CapacitatedTransferLayer(
            net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="smooth", tau=tau
        )


def test_construction_rejects_n_passes_below_one():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    with pytest.raises(ValueError, match="n_passes"):
        CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc, n_passes=0)


def test_construction_rejects_edge_kind_absent_from_network():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    with pytest.raises(ValueError, match="no edges"):
        CapacitatedTransferLayer(net, "cap", "pipe", s_max=s_max, c_arc=c_arc)


def test_construction_rejects_wrong_preference_shape():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    bad_preference = torch.full((3,), 1.0, dtype=F64)  # 3 edges, network has 2
    with pytest.raises(ValueError, match="preference has trailing size"):
        CapacitatedTransferLayer(
            net, "cap", "link", s_max=s_max, c_arc=c_arc, preference=bad_preference
        )


def test_construction_rejects_non_positive_preference():
    """`mode="projection"` divides by `preference` (`lambda / preference_i`), so a zero
    weight gives a clean forward value and a silent NaN gradient -- the one failure the
    layer cannot report from its own output. Refused at construction instead."""
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    with pytest.raises(ValueError, match="preference must be > 0"):
        CapacitatedTransferLayer(
            net, "cap", "link", s_max=s_max, c_arc=c_arc,
            preference=torch.tensor([0.0, 1.0], dtype=F64),
        )


def test_step_rejects_wrong_shaped_requests_driver_by_name():
    """Mirrors `Model._kind_flows`' own shape refusal: a wrong-shaped `requests` must
    raise a named `ValueError`, not an unnamed torch broadcast `RuntimeError`."""
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s0 = torch.zeros(net.n, dtype=F64)
    with pytest.raises(ValueError, match=r"cap.requests.*trailing shape \(2,\)"):
        layer.step(s0, {"cap.requests": torch.ones(3, dtype=F64)}, dt=1.0)


def test_step_hard_clip_below_capacity_passes_request_through():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s0 = torch.zeros(net.n, dtype=F64)
    drivers = {"cap.requests": torch.tensor([2.0, 2.0], dtype=F64)}
    s1, f = layer.step(s0, drivers, dt=1.0)
    assert torch.allclose(f, torch.tensor([2.0, 2.0], dtype=F64))
    # A: only a source, loses 2. B: gains 2 from A, loses 2 to C, net 0. C: gains 2.
    assert torch.allclose(s1, torch.tensor([-2.0, 0.0, 2.0], dtype=F64))


def test_step_hard_clip_above_arc_capacity_is_capped():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 1.5, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s0 = torch.zeros(net.n, dtype=F64)
    drivers = {"cap.requests": torch.tensor([5.0, 5.0], dtype=F64)}
    _, f = layer.step(s0, drivers, dt=1.0)
    assert torch.allclose(f, torch.tensor([1.5, 1.5], dtype=F64))


def test_step_hard_clip_above_receiver_headroom_is_capped():
    net = _chain_net()
    # C already at s_max=1.0, so the B->C edge's headroom is 0.
    s_max = torch.tensor([100.0, 100.0, 1.0], dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s0 = torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    drivers = {"cap.requests": torch.tensor([2.0, 2.0], dtype=F64)}
    _, f = layer.step(s0, drivers, dt=1.0)
    assert torch.allclose(f, torch.tensor([2.0, 0.0], dtype=F64))


@pytest.mark.parametrize(("dt", "s_max_c"), [(2.0, 10.0), (86400.0, 86400.0)])
def test_step_receiver_headroom_clip_is_a_rate_at_dt_other_than_one(dt, s_max_c):
    """Regression: the receiver-headroom clip must bound the actual
    STORAGE increase at ANY `dt`, not only at `dt == 1.0`.

    `headroom` is `(s_max - s) / dt` -- a RATE, in the same m3/s units as `f`,
    `committed_in`, `avail` and everything else it is compared against -- while the
    storage update multiplies by `dt`. Before the fix it was the raw VOLUME `s_max - s`,
    which coincides with the rate only at `dt == 1.0` (every other test and benchmark
    here). At the two `dt` values below the pre-fix arithmetic gave
    `f = s_max_c` instead of `s_max_c / dt`, overshot `s_max` by a factor of `dt`, and
    reported the excess as `overflow` -- volume conserved, but the clip no longer
    bounding what it exists to bound. `dt = 86400` (one WSIMOD day in seconds) is the
    intended use, so this was not a hypothetical `dt`.

    Both `(dt, s_max_c)` pairs are chosen so `s_max_c / dt` is exact in binary floating
    point and the fill lands on `s_max` to the last bit, letting the assertions be exact
    rather than tolerance-guarded.
    """
    net = _chain_net()
    s_max = torch.tensor([1e9, 1e9, s_max_c], dtype=F64)
    c_arc = torch.full((2,), 1e9, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s0 = torch.zeros(net.n, dtype=F64)
    # B->C asks for far more than C can take; A->B asks for nothing, so C's fill comes
    # from exactly one edge and is hand-computable.
    drivers = {"cap.requests": torch.tensor([0.0, 1e9], dtype=F64)}
    diag: dict = {}
    s1, f = layer.step(s0, drivers, dt=dt, diagnostics=diag)
    assert f[1].item() == s_max_c / dt
    # C is filled EXACTLY to s_max: fully, and not past it.
    assert s1[2].item() == s_max_c
    assert torch.equal(diag["overflow"], torch.zeros(net.n, dtype=F64))


def test_step_reports_overflow_when_storage_starts_above_s_max():
    """Row W5's `overflow` diagnostic, asserted directly against a hand-computed value.

    With C1 fixed, the receiver-headroom clip genuinely bounds every INFLOW, so a node
    can only end a step above its own `s_max` if it began one there -- nothing in this
    layer (or in `Model`) forbids a caller handing in such a state, and this is what
    `overflow` exists to report rather than silently absorb into the final clamp.

    C starts at 5.0 with `s_max = 3.0`: its headroom is `(3 - 5)/2 = -1`, floored to 0 by
    `_nonneg`, so B->C carries nothing; C has no out-edge either, so its unclamped
    storage stays 5.0, is clamped back to 3.0, and the 2.0 m3 difference is reported as
    `2.0 / dt = 1.0` m3/s of overflow.
    """
    net = _chain_net()
    s_max = torch.tensor([100.0, 100.0, 3.0], dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    s0 = torch.tensor([0.0, 0.0, 5.0], dtype=F64)
    drivers = {"cap.requests": torch.tensor([2.0, 2.0], dtype=F64)}
    diag: dict = {}
    s1, f = layer.step(s0, drivers, dt=2.0, diagnostics=diag)
    assert torch.allclose(f, torch.tensor([2.0, 0.0], dtype=F64))
    assert torch.allclose(diag["overflow"], torch.tensor([0.0, 0.0, 1.0], dtype=F64))
    # A loses 2 m3/s for 2 s; B gains 2 and forwards none; C is clamped to its s_max.
    assert torch.allclose(s1, torch.tensor([-4.0, 4.0, 3.0], dtype=F64))


def test_step_missing_request_driver_raises_keyerror():
    net = _chain_net()
    s_max = torch.full((net.n,), 10.0, dtype=F64)
    c_arc = torch.full((2,), 1.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    with pytest.raises(KeyError, match="cap.requests"):
        layer.step(torch.zeros(net.n, dtype=F64), {}, dt=1.0)


def test_model_registers_capacitated_layer():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    model = Model(net, {"cap": layer})
    assert model.capacitated == {"cap": layer}


def test_model_refuses_capacitated_layer_as_a_transport_flow_owner_by_name():
    """`Model.__init__`'s ownership scan lets a capacitated layer own a transport layer's
    flow kinds (it writes `"<name>.q"` in the same key convention), but reading those flows
    would need `CapacitatedTransferLayer.flows_of_kind` -- species/quality transport on
    capacitated flows, deliberately not built. `_kind_flows` must not fall through to
    `self.potential[owner]` and raise a bare `KeyError` naming nothing; it must refuse by
    name instead.
    """
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    cap = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    # B and C are the active interior of the "link" edges once A is the boundary.
    spec = TransportLayer(
        net, "spec", capacity=torch.tensor([50.0, 50.0], dtype=F64), flow_kind="link",
        boundary=["A"], scheme="implicit",
    )
    model = Model(net, {"cap": cap, "spec": spec})
    assert model.flow_layer_of["spec"] == "cap"  # the ownership scan really does pick it
    state = {
        "cap.s": torch.zeros(net.n, dtype=F64),
        "spec.x": torch.zeros(2, dtype=F64),
    }
    drivers = {
        "cap.requests": torch.tensor([2.0, 2.0], dtype=F64),
        "spec.x_boundary": torch.tensor([1e-3], dtype=F64),
    }
    with pytest.raises(NotImplementedError, match="capacitated layer 'cap'"):
        model.step(state, drivers, dt=1.0)


def test_model_refuses_residuals_for_a_model_owning_a_capacitated_layer():
    """`residuals()` reports the balance whose zero `steady()` converges to, and `_pass`
    already refuses a steady pass for a capacitated layer by name. Silently returning `{}`
    (or the other layers' balances alone) would present a balance over part of the model as
    the model's."""
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    model = Model(net, {"cap": CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc
    )})
    with pytest.raises(ValueError, match="residuals"):
        model.residuals({"cap.s": torch.zeros(net.n, dtype=F64)}, {})


def test_model_rejects_capacitated_layer_without_dt():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    model = Model(net, {"cap": layer})
    state = {"cap.s": torch.zeros(net.n, dtype=F64)}
    drivers = {"cap.requests": torch.tensor([2.0, 2.0], dtype=F64)}
    with pytest.raises(ValueError, match="dt"):
        model._pass(state, drivers, None, {})


def _diamond_net() -> Network:
    """A -> B -> D, A -> C -> D: two paths sharing the sink D."""
    net = Network()
    for name in ("A", "B", "C", "D"):
        net.add_node(name)
    net.add_edge("A", "B", kind="link")
    net.add_edge("A", "C", kind="link")
    net.add_edge("B", "D", kind="link")
    net.add_edge("C", "D", kind="link")
    return net


def test_step_shares_scarce_receiver_headroom_by_preference():
    net = _diamond_net()
    s_max = torch.tensor([100.0, 100.0, 100.0, 3.0], dtype=F64)  # D's headroom is 3.0
    c_arc = torch.full((4,), 100.0, dtype=F64)
    # preference order matches edge insertion: A->B, A->C, B->D, C->D
    preference = torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=F64)  # B->D favoured 2:1
    layer = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, preference=preference,
        n_passes=5,
    )
    s0 = torch.zeros(net.n, dtype=F64)
    drivers = {"cap.requests": torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=F64)}
    _, f = layer.step(s0, drivers, dt=1.0)
    # D's headroom (3.0) is shared 2:1 between B->D and C->D -> 2.0 and 1.0. This is the
    # only genuine competition in this graph: B and C each have exactly ONE in-edge, so
    # there is nothing for A->B or A->C to share with.
    assert torch.allclose(f[2:], torch.tensor([2.0, 1.0], dtype=F64), atol=1e-6)
    # A->B and A->C are NOT capped to match B->D/C->D. B and C each have ample headroom
    # (s_max=100) and only one in-edge apiece, so their receiver-side sharing never
    # triggers -- each simply passes its full 10.0 request through, exactly like the
    # single-edge hard-clip. B and C then accumulate the surplus (10 - 2 = 8, and
    # 10 - 1 = 9 respectively) as storage this step; nothing in the layer's rule (sharing
    # applies only "where more than one out-edge draws on one node's supply", i.e. at the
    # CONVERGING node) or in WSIMOD's own per-arc push/pull
    # semantics (a node's accept decision is its own push_check against
    # its OWN storage headroom, never against its future ability to forward the flow
    # onward) caps an upstream edge to match a downstream bottleneck two hops away.
    assert torch.allclose(f[:2], torch.tensor([10.0, 10.0], dtype=F64), atol=1e-6)


def _star_net() -> Network:
    """X -> D, Y -> D, Z -> D: three independent sources sharing one sink D."""
    net = Network()
    for name in ("X", "Y", "Z", "D"):
        net.add_node(name)
    net.add_edge("X", "D", kind="link")
    net.add_edge("Y", "D", kind="link")
    net.add_edge("Z", "D", kind="link")
    return net


def test_step_conserves_with_n_passes_1_vs_5():
    """A single pass under-shares when one edge's request is below its fair share;
    later passes recover the difference by redistributing the freed preference weight
    among the still-competing edges.

    The diamond fixture (used elsewhere in this file) does NOT exercise this: with
    c_arc=100 >> D's headroom of 3.0 and both D-competing edges requesting far more
    than their fair share, pass 1 already saturates the exact max-min-fair split and
    n_passes=1/2/5 are bit-identical there -- n_passes only matters when some
    competitor's request falls short of what a naive fair share would hand it, freeing
    headroom that only a LATER pass redistributes to the others. This fixture (three
    edges into one sink, one deliberately under-demanding) is the minimal case that
    needs more than one pass, and was hand-verified pass-by-pass in the task report.
    """
    net = _star_net()
    s_max = torch.tensor([100.0, 100.0, 100.0, 3.0], dtype=F64)  # D's headroom is 3.0
    c_arc = torch.full((3,), 100.0, dtype=F64)
    preference = torch.full((3,), 1.0, dtype=F64)  # equal preference -> 1.0 fair share each
    # X requests only 0.5 (below its 1.0 fair share); Y and Z each request far more
    # than theirs, so they are the ones left holding X's unused headroom.
    drivers = {"cap.requests": torch.tensor([0.5, 10.0, 10.0], dtype=F64)}
    layer_1 = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, preference=preference, n_passes=1
    )
    layer_2 = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, preference=preference, n_passes=2
    )
    layer_5 = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, preference=preference, n_passes=5
    )
    _, f1 = layer_1.step(torch.zeros(net.n, dtype=F64), drivers, dt=1.0)
    _, f2 = layer_2.step(torch.zeros(net.n, dtype=F64), drivers, dt=1.0)
    _, f5 = layer_5.step(torch.zeros(net.n, dtype=F64), drivers, dt=1.0)
    # Pass 1: naive 3-way fair share is 1.0 each; X takes only what it asked for (0.5),
    # but that unused 0.5 is NOT yet redistributed -- Y and Z stay capped at 1.0, so
    # 0.5 of D's 3.0 headroom is left unused (sum = 2.5, provably under-shared).
    assert torch.allclose(f1, torch.tensor([0.5, 1.0, 1.0], dtype=F64), atol=1e-6)
    assert torch.allclose(f1.sum(), torch.tensor(2.5, dtype=F64), atol=1e-6)
    # Pass 2 (and every pass after): X's freed 0.5 preference share is redistributed
    # 50:50 between Y and Z, the only two edges still actively competing, each rising
    # from 1.0 to 1.25 -- D's headroom is now fully used (sum = 3.0, exactly).
    assert torch.allclose(f2, torch.tensor([0.5, 1.25, 1.25], dtype=F64), atol=1e-6)
    assert torch.allclose(f2.sum(), torch.tensor(3.0, dtype=F64), atol=1e-6)
    assert torch.allclose(f5, f2, atol=1e-6)  # already converged by pass 2, stable after


def test_smooth_mode_converges_to_hard_clip_as_tau_shrinks():
    """Single-out-edge chain: never touches the sharing `_select` site, only the
    `_clip`/`_nonneg` sites (avail's minimum, the tentative minimum, the final storage
    clamp) -- so this pins those three sites' convergence to hard-clip as tau shrinks.

    Once the arc's `c_arc` is exhausted (this fixture's request, 5.0, exceeds its
    capacity, 1.5), `c_arc - f` hovers at ~0 every pass thereafter, and `_nonneg`'s
    softplus has a built-in `tau * ln(2)` offset AT exactly zero (`softplus(0) =
    ln(2)`, not 0) -- so each of the fixed `n_passes=5` rounds re-adds that offset as
    a small phantom `avail`, compounding to a total error of `O(n_passes * tau)`
    rather than `O(tau)`. This is an inherent property of re-applying a smoothed
    clip `n_passes` times over an already-saturated arc, not a bug in `_clip`/
    `_nonneg` individually (each converges to its hard counterpart at a fixed point,
    just not instantaneously within one pass) -- confirmed by tracing the loop
    pass-by-pass at tau=0.01: avail/share stays a small tau-scaled residual every
    round instead of collapsing to exactly 0 after the first. Hence the tightest tau
    below is 1e-4, not 1e-2 as a naive single-clip reading would use.
    """
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 1.5, dtype=F64)
    drivers = {"cap.requests": torch.tensor([5.0, 5.0], dtype=F64)}
    hard = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="hard")
    _, f_hard = hard.step(torch.zeros(net.n, dtype=F64), drivers, dt=1.0)
    errors = []
    for tau in (1.0, 0.1, 1e-4):
        smooth = CapacitatedTransferLayer(
            net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="smooth", tau=tau
        )
        _, f_smooth = smooth.step(torch.zeros(net.n, dtype=F64), drivers, dt=1.0)
        errors.append((f_smooth - f_hard).abs().max().item())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-3


def test_smooth_mode_is_differentiable_through_requests():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 1.5, dtype=F64)
    layer = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="smooth", tau=0.1
    )
    r = torch.tensor([5.0, 5.0], dtype=F64, requires_grad=True)
    _, f = layer.step(torch.zeros(net.n, dtype=F64), {"cap.requests": r}, dt=1.0)
    f.sum().backward()
    assert r.grad is not None
    assert torch.isfinite(r.grad).all()


def test_smooth_mode_shares_scarce_receiver_headroom_by_preference():
    """Smooth-mode variant of `test_step_shares_scarce_receiver_headroom_by_preference`,
    covering the `_select` sharing site (the `over_subscribed_e` `torch.where`), which
    the chain fixtures above never exercise. A small `tau` should reproduce the hard
    2:1 split closely; the tolerance is scaled to `tau` rather than pinned to the hard
    mode's exact bound, since the softmin/sigmoid blend never converges bit-exactly.
    """
    net = _diamond_net()
    s_max = torch.tensor([100.0, 100.0, 100.0, 3.0], dtype=F64)
    c_arc = torch.full((4,), 100.0, dtype=F64)
    preference = torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=F64)
    tau = 0.01
    layer = CapacitatedTransferLayer(
        net,
        "cap",
        "link",
        s_max=s_max,
        c_arc=c_arc,
        preference=preference,
        mode="smooth",
        tau=tau,
        n_passes=5,
    )
    s0 = torch.zeros(net.n, dtype=F64)
    drivers = {"cap.requests": torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=F64)}
    _, f = layer.step(s0, drivers, dt=1.0)
    assert torch.allclose(f[2:], torch.tensor([2.0, 1.0], dtype=F64), atol=50 * tau)
    assert torch.allclose(f[:2], torch.tensor([10.0, 10.0], dtype=F64), atol=50 * tau)
    # Fully differentiable through the sharing site too.
    r = drivers["cap.requests"].clone().requires_grad_(True)
    _, f_grad = layer.step(s0, {"cap.requests": r}, dt=1.0)
    f_grad.sum().backward()
    assert r.grad is not None
    assert torch.isfinite(r.grad).all()


def test_projection_mode_converges_to_hard_clip_as_it_gets_tighter():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 1.5, dtype=F64)
    drivers = {"cap.requests": torch.tensor([5.0, 5.0], dtype=F64)}
    hard = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="hard")
    _, f_hard = hard.step(torch.zeros(net.n, dtype=F64), drivers, dt=1.0)
    projection = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="projection"
    )
    _, f_proj = projection.step(torch.zeros(net.n, dtype=F64), drivers, dt=1.0)
    assert torch.allclose(f_proj, f_hard, atol=1e-4)


def test_projection_mode_is_differentiable_through_requests():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 1.5, dtype=F64)
    layer = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="projection"
    )
    r = torch.tensor([5.0, 5.0], dtype=F64, requires_grad=True)
    _, f = layer.step(torch.zeros(net.n, dtype=F64), {"cap.requests": r}, dt=1.0)
    f.sum().backward()
    assert r.grad is not None
    assert torch.isfinite(r.grad).all()


def test_gradcheck_smooth_and_projection_on_diamond():
    net = _diamond_net()
    s_max = torch.tensor([100.0, 100.0, 100.0, 3.0], dtype=F64)
    c_arc = torch.full((4,), 100.0, dtype=F64)
    preference = torch.full((4,), 1.0, dtype=F64)
    for mode, kwargs in (("smooth", {"tau": 0.1}), ("projection", {})):
        layer = CapacitatedTransferLayer(
            net, "cap", "link", s_max=s_max, c_arc=c_arc, preference=preference,
            mode=mode, **kwargs,
        )

        def f_of_r(r, layer=layer):
            _, f = layer.step(torch.zeros(net.n, dtype=F64), {"cap.requests": r}, dt=1.0)
            return f

        r0 = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=F64, requires_grad=True)
        assert torch.autograd.gradcheck(f_of_r, (r0,), eps=1e-6, atol=1e-4)


def test_projection_mode_sharing_has_nonzero_cross_gradient():
    """Regression: a `mode="projection"` that reused hard-clip's own
    preference-proportional-share formula at the sharing site, which is a function of
    preference weights and total headroom alone -- never of any individual edge's own
    request -- would have a provably ZERO cross-gradient between competing edges, defeating
    the purpose of the mode (gradients should flow through which arc absorbs a constraint).
    Confirmed empirically (not just by inspection) via `torch.autograd.functional.jacobian`:
    such a `d(f_BD)/d(r_CD)` was byte-identical to `mode="hard"`'s own (zero) cross-term at
    this exact fixture, across 200 random trials too. `_share_via_qp`'s real coupled QP (via
    `solve_monotone`) fixes this: `d(f_BD)/d(r_CD)` must be genuinely nonzero, and its SIGN must be
    negative -- raising a competing edge's request can only ever grow ITS OWN share and shrink
    everyone else's, since the shared pool (`free_headroom`) does not grow.
    """
    net = _diamond_net()
    s_max = torch.tensor([100.0, 100.0, 100.0, 3.0], dtype=F64)
    c_arc = torch.full((4,), 100.0, dtype=F64)
    preference = torch.full((4,), 1.0, dtype=F64)
    layer = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, preference=preference, mode="projection"
    )

    def f_of_r(r):
        _, f = layer.step(torch.zeros(net.n, dtype=F64), {"cap.requests": r}, dt=1.0)
        return f

    r0 = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=F64)
    jac = torch.autograd.functional.jacobian(f_of_r, r0)
    # f_BD is edge index 2, r_CD is edge index 3 -- competing for D's headroom, D is
    # oversubscribed at this fixture's numbers (demand 4.0 > headroom 3.0).
    cross = jac[2, 3].item()
    assert cross < -1e-6, f"expected a genuinely negative cross-gradient, got {cross}"
    # Own-request sensitivity should be positive (more of my own request -> more of my own
    # share, up to the shared constraint).
    assert jac[2, 2].item() > 1e-6


def test_projection_mode_sharing_is_nan_free_under_random_fixtures():
    """`_share_via_qp`'s `solve_monotone`-based root-find has a genuine
    division (`weight = -grad_output / f_x` inside `solve_monotone`'s own backward) that
    blows up to NaN whenever every competing edge at a node is simultaneously boundary-
    pinned at the converged root -- observed directly during this fix, in two distinct
    forms: a single, arc-capacity-saturated edge at a non-competing node (caught by
    `test_projection_mode_is_differentiable_through_requests`'s chain fixture), and a
    genuinely multi-edge node whose competing edges are ALL already fully satisfied by an
    earlier `n_passes` round (`remaining == 0` for all of them) -- the second form was
    missed by every hand-written fixture above and only surfaced via randomised search: a
    first fix-attempt gated on in-degree alone still produced NaN gradients in 101/200
    random trials. This test pins that regression with a smaller, deterministic (seeded)
    sweep so a future change reintroducing either degenerate case fails CI directly, without
    needing a fresh randomised search to rediscover it.
    """
    net = _diamond_net()
    c_arc = torch.full((4,), 100.0, dtype=F64)
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        s_max = torch.tensor(
            [100.0, 100.0, 100.0, torch.rand(1, generator=generator).item() * 5 + 0.5],
            dtype=F64,
        )
        preference = torch.rand(4, generator=generator, dtype=F64) * 2 + 0.5
        layer = CapacitatedTransferLayer(
            net, "cap", "link", s_max=s_max, c_arc=c_arc, preference=preference,
            mode="projection",
        )

        def f_of_r(r, layer=layer):
            _, f = layer.step(torch.zeros(net.n, dtype=F64), {"cap.requests": r}, dt=1.0)
            return f

        r0 = torch.rand(4, generator=generator, dtype=F64) * 5 + 0.1
        jac = torch.autograd.functional.jacobian(f_of_r, r0)
        assert torch.isfinite(jac).all(), f"NaN/Inf Jacobian at s_max={s_max}, r0={r0}"


def test_model_steps_capacitated_layer_end_to_end():
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 10.0, dtype=F64)
    layer = CapacitatedTransferLayer(net, "cap", "link", s_max=s_max, c_arc=c_arc)
    model = Model(net, {"cap": layer})
    state = {"cap.s": torch.zeros(net.n, dtype=F64)}
    drivers = {"cap.requests": torch.tensor([2.0, 2.0], dtype=F64)}
    new_state = model.step(state, drivers, dt=1.0)
    assert torch.allclose(new_state["cap.q"], torch.tensor([2.0, 2.0], dtype=F64))
    assert torch.allclose(
        new_state["cap.s"], torch.tensor([-2.0, 0.0, 2.0], dtype=F64)
    )


def test_clip_projection_matches_box_clamp():
    """White-box: `_clip_projection` is the projection-mode QP machinery -- pin that
    it actually exists and computes the box-constrained least-squares projection (which has
    the closed form `clamp(r, 0, min(c_arc, headroom))`), not merely an alias for `_clip`.
    """
    net = _chain_net()
    s_max = torch.full((net.n,), 100.0, dtype=F64)
    c_arc = torch.full((2,), 1.5, dtype=F64)
    layer = CapacitatedTransferLayer(
        net, "cap", "link", s_max=s_max, c_arc=c_arc, mode="projection"
    )
    r = torch.tensor([0.5, 2.0, -1.0], dtype=F64)
    headroom = torch.tensor([1.0, 1.0, 1.0], dtype=F64)
    c_arc3 = torch.tensor([10.0, 10.0, 10.0], dtype=F64)
    out = layer._clip_projection(r, headroom, c_arc3)
    expected = torch.clamp(r, torch.zeros_like(r), torch.minimum(c_arc3, headroom))
    assert torch.allclose(out, expected, atol=1e-9)

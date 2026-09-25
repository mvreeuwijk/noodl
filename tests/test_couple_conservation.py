"""Independent references for two coupling properties: conservation (the donor receives
exactly the recipient's own integrated transfer) and convergence judged on the returned
state.

Two unit-capacity compartments. A: one edge zone->ambient that carries no flow, so A changes
only through the sources the coupler adds. B: zone->ambient and ambient->zone, both carrying
unit flow, so B exchanges with its boundary at rate q*(x_boundary - x_B). The two-way link
feeds A's concentration to B's boundary and B's net boundary inflow back into A's sources.
"""

from __future__ import annotations

import pytest
import torch

from noodl.couple import ValueLink, union
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64


def _t(values):
    return torch.tensor(values, dtype=F64)


def compartment(name, *, scheme="exact", circulation=False, initial=1.0, source=0.0):
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    if circulation:
        net.add_edge("ambient", "zone", kind="flow")
    layer = TransportLayer(
        net, name, capacity=_t([1.0]), flow_kind="flow", boundary=["ambient"],
        scheme=scheme, quantity="concentration", unit="kg/m3",
    )
    q = _t([1.0, 1.0] if circulation else [0.0])

    def closure(state, drivers):
        return {f"{name}.q": q}

    model = Model(net, {name: layer}, closures=[closure])
    state = {f"{name}.x": _t([initial])}
    sources = (torch.stack([torch.zeros((), dtype=F64), source])
               if isinstance(source, torch.Tensor) else _t([0.0, source]))
    drivers = {f"{name}.x_boundary": _t([0.0]), f"{name}.sources": sources}
    return model, state, drivers


def coupled_case(scheme, substeps=1, *, source=0.0, initial_b=0.0):
    a = compartment("a", scheme=scheme)
    b = compartment("b", scheme=scheme, circulation=True, initial=initial_b, source=source)
    model, state, drivers = union(
        {"A": a, "B": b},
        [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
        substeps={"B": substeps}, iterate_rtol=1e-12, iterate_atol=1e-14, iterate_max=200,
    )
    diagnostics: dict = {}
    result = model.step(state, drivers, 1.0, diagnostics=diagnostics)
    return result["A"]["a.x"].item(), result["B"]["b.x"].item(), diagnostics


@pytest.mark.parametrize(
    "scheme, substeps",
    [
        ("exact", 1),
        ("implicit", 1),  # control: a single implicit step's endpoint flux IS its integral
        ("implicit", 4),
        ("trapezoidal", 1),
    ],
)
def test_closed_two_way_exchange_conserves_the_total_amount(scheme, substeps):
    """No external source or sink anywhere: whatever leaves B must arrive in A."""
    a, b, diagnostics = coupled_case(scheme, substeps)
    assert bool(diagnostics["converged"].all())
    assert a + b == pytest.approx(1.0, abs=1e-10)


def test_two_way_iteration_returns_the_coupled_backward_euler_solution():
    """Both start at 1, B receives a unit source, one implicit step of one second each.

    A: a = 1 + (b - a)          ->  2a - b = 1
    B: b = 1 + 1 + (a - b)      -> -a + 2b = 2
    Solution (4/3, 5/3). The old code returned (1.5, 1.5) with converged=True.
    """
    a, b, diagnostics = coupled_case("implicit", 1, source=1.0, initial_b=1.0)
    assert bool(diagnostics["converged"].all())
    assert a == pytest.approx(4 / 3, abs=1e-9)
    assert b == pytest.approx(5 / 3, abs=1e-9)


def test_a_two_way_step_never_assembles_a_dense_topology_operator(monkeypatch):
    """The compartments carry no potential layer, so nothing on this path has a legitimate
    reason to form an (edges, nodes) matrix."""

    def forbidden(self, *args, **kwargs):
        raise AssertionError("dense topology operator assembled on the coupling path")

    monkeypatch.setattr(Network, "upwind", forbidden)
    monkeypatch.setattr(Network, "incidence", forbidden)
    monkeypatch.setattr(Network, "source_selector", forbidden)
    monkeypatch.setattr(Network, "target_selector", forbidden)
    a, b, _ = coupled_case("implicit", 1)
    assert a + b == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("scheme, substeps", [("exact", 1), ("implicit", 1), ("implicit", 4),
                                              ("trapezoidal", 1), ("trapezoidal", 3)])
def test_conservation_holds_before_convergence(scheme, substeps):
    """Conservation is by construction: cap the iteration at two passes and check the
    balance of what IS returned, on a pass-2 state that is still far from the fixed point
    (measured ``max_change`` is 0.11-0.13 for every parametrisation below, none of them
    anywhere near settled). ``iterate_rtol=0.5`` is loose enough that this residual still
    counts as "converged" by the per-pass criterion (so the run returns normally instead of
    raising), but the state itself is a genuinely early, unconverged pass -- the exact
    opposite of the tight-tolerance runs in the tests above. If a future change to the
    coupling makes some parametrisation settle below rtol 0.5 in two passes, the
    ``max_change`` assertion below will fail loudly rather than silently start testing a
    converged run instead."""
    a = compartment("a", scheme=scheme)
    b = compartment("b", scheme=scheme, circulation=True, initial=0.0)
    model, state, drivers = union(
        {"A": a, "B": b}, [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
        substeps={"B": substeps}, iterate_rtol=0.5, iterate_atol=0.0, iterate_max=2,
    )
    diagnostics: dict = {}
    result = model.step(state, drivers, 1.0, diagnostics=diagnostics)
    assert diagnostics["passes"] == 2
    assert max(c.max().item() for c in diagnostics["max_change"].values()) > 1e-3
    a_val = result["A"]["a.x"].item()
    b_val = result["B"]["b.x"].item()
    assert a_val + b_val == pytest.approx(1.0, abs=1e-10)


def test_unequal_capacities_nonzero_states_and_reversed_flow_conserve():
    a = compartment("a", scheme="implicit", initial=0.3)
    b = compartment("b", scheme="implicit", circulation=True, initial=2.0)
    a[0].transport["a"].capacity = _t([3.0])
    b[0].transport["b"].capacity = _t([0.5])
    # reverse B's circulation: both edges carry -1
    b_model, b_state, b_drivers = b
    b_model.closures[0] = lambda s, d: {"b.q": _t([-1.0, -1.0])}
    model, state, drivers = union(
        {"A": a, "B": (b_model, b_state, b_drivers)},
        [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
        substeps={"B": 2}, iterate_rtol=1e-12, iterate_atol=1e-14, iterate_max=200,
    )
    result = model.step(state, drivers, 0.7)
    total = 3.0 * result["A"]["a.x"].item() + 0.5 * result["B"]["b.x"].item()
    assert total == pytest.approx(3.0 * 0.3 + 0.5 * 2.0, abs=1e-10)


def test_external_sources_on_both_sides_enter_the_budget_exactly():
    a = compartment("a", scheme="trapezoidal", source=0.4)
    b = compartment("b", scheme="trapezoidal", circulation=True, source=-0.1, initial=1.0)
    model, state, drivers = union(
        {"A": a, "B": b}, [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
        iterate_rtol=1e-12, iterate_atol=1e-14, iterate_max=200,
    )
    result = model.step(state, drivers, 1.0)
    assert result["A"]["a.x"].item() + result["B"]["b.x"].item() == pytest.approx(
        2.0 + 1.0 * (0.4 - 0.1), abs=1e-10)


def test_two_recipients_of_one_donor_conserve_and_report_each_transfer():
    a = compartment("a", scheme="implicit")
    # initial=0.0, as the fixtures above do: with every compartment starting at the
    # same 1.0, A's boundary value would already equal each recipient's own state and no
    # transfer would ever be nonzero.
    b1 = compartment("b1", scheme="implicit", circulation=True, initial=0.0)
    b2 = compartment("b2", scheme="implicit", circulation=True, initial=0.0)
    model, state, drivers = union(
        {"A": a, "B1": b1, "B2": b2},
        [ValueLink("A", "a.x", 0, "B1", "b1.x_boundary", two_way=True),
         ValueLink("A", "a.x", 0, "B2", "b2.x_boundary", two_way=True)],
        iterate_rtol=1e-12, iterate_atol=1e-14, iterate_max=200,
    )
    diagnostics: dict = {}
    result = model.step(state, drivers, 1.0, diagnostics=diagnostics)
    total = sum(result[tag][k].item() for tag, k in (("A", "a.x"), ("B1", "b1.x"), ("B2", "b2.x")))
    assert total == pytest.approx(1.0, abs=1e-10)
    assert len(diagnostics["transfers"]) == 2


def test_a_model_that_is_both_recipient_and_donor_is_refused_by_name():
    a = compartment("a", scheme="implicit", circulation=True)
    b = compartment("b", scheme="implicit", circulation=True)
    with pytest.raises(ValueError, match="both a recipient and a donor"):
        union({"A": a, "B": b},
              [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True),
               ValueLink("B", "b.x", 0, "A", "a.x_boundary", two_way=True)])


def test_batch_with_mixed_convergence_is_reported_per_instance():
    a = compartment("a", scheme="implicit")
    # initial=0.0 (see the comment above): otherwise A's boundary already equals B's own
    # state and the coupling converges trivially on pass 1 for both instances.
    b = compartment("b", scheme="implicit", circulation=True, initial=0.0)
    b_model, b_state, b_drivers = b
    # Instance 1 exchanges 20x faster than instance 0. Measured (for the
    # recipient-first schedule): instance 1's faster exchange reaches its own
    # boundary equilibrium sooner and is the FIRST to satisfy the per-pass criterion (by
    # pass 8); instance 0's slower exchange is still short of it at pass 8 and needs 18 --
    # the opposite instance from a previous-pass-residual criterion.
    b_model.closures[0] = lambda s, d: {"b.q": torch.stack([_t([1.0, 1.0]), _t([20.0, 20.0])])}
    model, state, drivers = union(
        {"A": a, "B": (b_model, b_state, b_drivers)},
        [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
        iterate_rtol=1e-10, iterate_atol=1e-14, iterate_max=8,
    )
    diagnostics: dict = {}
    with pytest.raises(RuntimeError, match=r"instances \[0\]"):
        model.step(state, drivers, 1.0, diagnostics=diagnostics)


def test_cross_interface_gradient_matches_central_differences_for_each_scheme():
    for scheme, substeps in (("exact", 1), ("implicit", 2), ("trapezoidal", 1)):
        def run(q_val, scheme=scheme, substeps=substeps):
            a = compartment("a", scheme=scheme)
            b = compartment("b", scheme=scheme, circulation=True)
            b_model, b_state, b_drivers = b
            b_model.closures[0] = lambda s, d, q_val=q_val: {"b.q": torch.stack([q_val, q_val])}
            model, state, drivers = union(
                {"A": a, "B": (b_model, b_state, b_drivers)},
                [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
                substeps={"B": substeps}, iterate_rtol=1e-12, iterate_atol=1e-14, iterate_max=200,
            )
            return model.step(state, drivers, 1.0)["A"]["a.x"].sum()

        q = torch.tensor(0.8, dtype=F64, requires_grad=True)
        (grad,) = torch.autograd.grad(run(q), (q,))
        h = 1e-6
        hi = run(torch.tensor(0.8 + h, dtype=F64))
        lo = run(torch.tensor(0.8 - h, dtype=F64))
        fd = (hi - lo) / (2 * h)
        assert grad.item() == pytest.approx(fd.item(), rel=1e-5), scheme


def test_partitioned_error_against_a_monolithic_reference_shrinks_with_the_step():
    """Accuracy, separately from conservation: the partitioned implicit scheme with k
    recipient substeps converges to the monolithic two-zone implicit solution as dt shrinks."""
    def monolithic(dt, steps):
        net = Network(dtype=F64)
        for n in ("amb", "a", "b"):
            net.add_node(n)
        net.add_edge("a", "b", kind="flow")
        net.add_edge("b", "a", kind="flow")
        net.add_edge("b", "amb", kind="flow")      # inactive: q = 0
        layer = TransportLayer(net, "m", capacity=_t([1.0, 1.0]), flow_kind="flow",
                               boundary=["amb"], scheme="implicit")
        model = Model(net, {"m": layer}, closures=[lambda s, d: {"m.q": _t([1.0, 1.0, 0.0])}])
        state = {"m.x": _t([1.0, 0.0])}
        drivers = {"m.x_boundary": _t([0.0]), "m.sources": torch.zeros(3, dtype=F64)}
        for _ in range(steps):
            state = model.step(state, drivers, dt)
        return state["m.x"]

    def partitioned(dt, steps):
        a = compartment("a", scheme="implicit")
        # initial=0.0, matching the monolithic reference's own state = [1.0, 0.0] above:
        # otherwise A and B start at the same value, the boundary is already at equilibrium,
        # and the partitioned run never moves at all regardless of dt.
        b = compartment("b", scheme="implicit", circulation=True, initial=0.0)
        model, state, drivers = union(
            {"A": a, "B": b}, [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
            substeps={"B": 4}, iterate_rtol=1e-12, iterate_atol=1e-14, iterate_max=200,
        )
        for _ in range(steps):
            state = model.step(state, drivers, dt)
        return torch.stack([state["A"]["a.x"][0], state["B"]["b.x"][0]])

    fine = monolithic(1.0 / 64, 64)
    errors = [
        (partitioned(dt, int(1.0 / dt)) - fine).abs().max().item() for dt in (0.5, 0.25, 0.125)
    ]
    assert errors[1] < errors[0] and errors[2] < errors[1]


def test_a_recipient_with_an_extra_unlinked_layer_runs_step_with_transfer_only_on_the_linked_one(  # noqa: E501
    monkeypatch,
):
    """A recipient's OWN unlinked layer (an `exact`-scheme thermal layer in the
    real building/street coupling) must not pay `step_with_transfer`'s extra cost -- the
    coupler must ask only for the linked layer's transfer, never every transport layer of
    the recipient."""
    a = compartment("a", scheme="implicit")

    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("zone")
    net.add_edge("zone", "ambient", kind="flow")
    net.add_edge("ambient", "zone", kind="flow")
    net.add_node("amb2")
    net.add_node("z2")
    net.add_edge("z2", "amb2", kind="heat")
    net.add_edge("amb2", "z2", kind="heat")
    linked = TransportLayer(net, "b", capacity=_t([1.0]), flow_kind="flow",
                             boundary=["ambient"], scheme="implicit",
                             quantity="concentration", unit="kg/m3")
    extra = TransportLayer(net, "extra", capacity=_t([1.0]), flow_kind="heat",
                            boundary=["amb2"], scheme="exact")

    def closure(state, drivers):
        return {"b.q": _t([1.0, 1.0]), "extra.q": _t([0.5, 0.5])}

    b_model = Model(net, {"b": linked, "extra": extra}, closures=[closure])
    b_state = {"b.x": _t([0.0]), "extra.x": _t([1.0])}
    # Sources are FULL node order over the whole (shared) net: ambient, zone, amb2, z2.
    zeros4 = torch.zeros(4, dtype=F64)
    b_drivers = {
        "b.x_boundary": _t([0.0]), "b.sources": zeros4,
        "extra.x_boundary": _t([2.0]), "extra.sources": zeros4,
    }

    calls: list[str] = []
    original = TransportLayer.step_with_transfer

    def recording(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TransportLayer, "step_with_transfer", recording)

    model, state, drivers = union(
        {"A": a, "B": (b_model, b_state, b_drivers)},
        [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
        iterate_rtol=1e-10, iterate_atol=1e-14, iterate_max=50,
    )
    model.step(state, drivers, 1.0)
    assert calls and set(calls) == {"b"}  # never "extra"


def _coupled_with_source(s, *, iterate_rtol=1e-12):
    a = compartment("a", scheme="implicit")
    b = compartment("b", scheme="implicit", circulation=True, initial=1.0, source=s)
    model, state, drivers = union(
        {"A": a, "B": b},
        [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
        iterate_rtol=iterate_rtol, iterate_atol=1e-14, iterate_max=200,
    )
    diagnostics: dict = {}
    out = model.step(state, drivers, 1.0, diagnostics=diagnostics)
    return out["A"]["a.x"].sum(), out["B"]["b.x"].sum(), diagnostics


def test_gradient_at_a_converged_start_is_the_coupled_derivative():
    """Both compartments start at 1 and B's source s is 0: the start IS the fixed point, so
    the primal converges in one pass at (1, 1). The coupled backward-Euler solution is
    a = 1 + s/3, b = 1 + 2s/3, so the derivatives are 1/3 and 2/3, whatever the pass count."""
    s = torch.tensor(0.0, dtype=F64, requires_grad=True)
    a, b, diagnostics = _coupled_with_source(s)
    assert diagnostics["passes"] == 1
    (da,) = torch.autograd.grad(a, (s,), retain_graph=True)
    (db,) = torch.autograd.grad(b, (s,))
    assert da.item() == pytest.approx(1.0 / 3.0, rel=1e-8)
    assert db.item() == pytest.approx(2.0 / 3.0, rel=1e-8)


def test_gradient_does_not_depend_on_the_primal_tolerance():
    """s = 0.3 from the same start: a loose primal tolerance stops after a few passes. The
    returned VALUE may then be off by the primal tolerance, but the derivative of the fixed
    point is still 1/3 and 2/3 and must be returned to the adjoint's accuracy."""
    s = torch.tensor(0.3, dtype=F64, requires_grad=True)
    a, b, diagnostics = _coupled_with_source(s, iterate_rtol=1e-3)
    assert diagnostics["passes"] < 10
    (da,) = torch.autograd.grad(a, (s,), retain_graph=True)
    (db,) = torch.autograd.grad(b, (s,))
    assert da.item() == pytest.approx(1.0 / 3.0, rel=1e-8)
    assert db.item() == pytest.approx(2.0 / 3.0, rel=1e-8)


def test_diagnostics_report_the_adjoint_and_the_primal_pass_count():
    """`passes` counts PRIMAL passes on both paths, so the differentiable run's count must
    equal the forward-only run's: the extra pass that carries the adjoint is not one of them,
    and a run with nothing to differentiate does not pay for it at all (`adjoint is None`)."""
    s = torch.tensor(0.3, dtype=F64, requires_grad=True)
    _a, _b, diagnostics = _coupled_with_source(s)
    assert diagnostics["adjoint"] == "implicit" and diagnostics["passes"] >= 2
    with torch.no_grad():
        _a, _b, plain = _coupled_with_source(torch.tensor(0.3, dtype=F64))
    assert plain["adjoint"] is None and plain["passes"] == diagnostics["passes"]
    # Grad enabled but nothing requiring it: still no adjoint pass, still the same count.
    _a, _b, forward_only = _coupled_with_source(torch.tensor(0.3, dtype=F64))
    assert forward_only["adjoint"] is None and forward_only["passes"] == diagnostics["passes"]


def test_the_differentiated_pass_conserves_exactly_like_the_primal_one():
    """Conservation on the pass that is actually RETURNED when a gradient is wanted. The
    returned state comes from the extra differentiable pass rather than from the last primal
    one, so the
    conservation property has to hold there too -- it does, by construction, because both run
    the same `_one_pass` and the donor receives the recipient's own integrated transfer.

    Both compartments start at 1 and B carries an external source s over dt = 1, so the exact
    budget is 2 + s with nothing else entering or leaving.
    """
    for value in (0.0, 0.3):
        s = torch.tensor(value, dtype=F64, requires_grad=True)
        a, b, diagnostics = _coupled_with_source(s)
        assert diagnostics["adjoint"] == "implicit"
        assert a.item() + b.item() == pytest.approx(2.0 + value, abs=1e-10)
        with torch.no_grad():
            plain_a, plain_b, _ = _coupled_with_source(torch.tensor(value, dtype=F64))
        # ... and it is the SAME pass: the certified primal state, reproduced.
        assert a.item() == pytest.approx(plain_a.item(), rel=1e-12, abs=1e-15)
        assert b.item() == pytest.approx(plain_b.item(), rel=1e-12, abs=1e-15)


def test_a_one_way_link_riding_on_the_iteration_carries_its_own_gradient():
    """The fixed-point INTERFACE is every link's forward value, one-way links included, and
    this is the case that decides it. A --two-way--> B (B the recipient) and B --one-way--> C:
    the parameter is B's own source, and C sees it only through the one-way value read out of
    B's state. Hold that entry outside the interface -- the obvious reading of "only two-way
    links close a loop" -- and dC/ds collapses to 0, because C's boundary would then be frozen
    at B's start value. Central differences say otherwise.
    """
    def run(s):
        a = compartment("a", scheme="implicit")
        b = compartment("b", scheme="implicit", circulation=True, initial=1.0, source=s)
        c = compartment("c", scheme="implicit", circulation=True, initial=0.0)
        model, state, drivers = union(
            {"A": a, "B": b, "C": c},
            [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True),
             ValueLink("B", "b.x", 0, "C", "c.x_boundary")],
            iterate_rtol=1e-12, iterate_atol=1e-14, iterate_max=200,
        )
        out = model.step(state, drivers, 1.0)
        return tuple(out[tag][f"{tag.lower()}.x"].sum() for tag in ("A", "B", "C"))

    s = torch.tensor(0.3, dtype=F64, requires_grad=True)
    values = run(s)
    grads = [torch.autograd.grad(v, (s,), retain_graph=True)[0].item() for v in values]
    h = 1e-6
    hi, lo = run(torch.tensor(0.3 + h, dtype=F64)), run(torch.tensor(0.3 - h, dtype=F64))
    for name, g, up, down in zip("ABC", grads, hi, lo, strict=True):
        fd = ((up - down) / (2 * h)).item()
        assert g == pytest.approx(fd, rel=1e-6), name
    assert grads[2] == pytest.approx(1.0 / 3.0, rel=1e-8)   # not zero, and not B's 2/3


def test_the_adjoint_solves_a_batched_interface_per_instance():
    """The link's forward value carries the batch as its leading dim, so the adjoint
    solves one small system PER INSTANCE rather than flattening the whole batch into one --
    `diagnostics["adjoint_batched"]` says so. A batch of instances with DIFFERENT couplings
    still has to come back with each instance's own derivative either way. Weighted so a
    single scalar gradient cannot hide a per-instance error."""
    n = 4
    weights = torch.arange(1.0, n + 1.0, dtype=F64)

    def run(s, diagnostics=None):
        a = compartment("a", scheme="implicit")
        b_model, b_state, b_drivers = compartment(
            "b", scheme="implicit", circulation=True, initial=1.0)
        q = torch.stack([_t([1.0 + i, 1.0 + i]) for i in range(n)])
        b_model.closures[0] = lambda _s, _d: {"b.q": q}
        b_drivers = dict(b_drivers)
        b_drivers["b.sources"] = torch.stack([torch.zeros(n, dtype=F64), s * weights], dim=-1)
        model, state, drivers = union(
            {"A": a, "B": (b_model, b_state, b_drivers)},
            [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
            iterate_rtol=1e-12, iterate_atol=1e-14, iterate_max=200,
        )
        out = model.step(state, drivers, 1.0, diagnostics=diagnostics)
        return (out["A"]["a.x"].reshape(n) * weights).sum() + out["B"]["b.x"].sum()

    s = torch.tensor(0.3, dtype=F64, requires_grad=True)
    diagnostics: dict = {}
    (grad,) = torch.autograd.grad(run(s, diagnostics), (s,))
    assert diagnostics["adjoint"] == "implicit"
    assert diagnostics["adjoint_batched"] is True
    h = 1e-6
    fd = (run(torch.tensor(0.3 + h, dtype=F64)) - run(torch.tensor(0.3 - h, dtype=F64))) / (2 * h)
    assert grad.item() == pytest.approx(fd.item(), rel=1e-6)

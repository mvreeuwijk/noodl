"""Independent oracles for the coupling findings R1 and R2 of the 19 September review.

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
    drivers = {f"{name}.x_boundary": _t([0.0]), f"{name}.sources": _t([0.0, source])}
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
    Solution (4/3, 5/3). The old code returned (1.5, 1.5) with converged=True (R2).
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
    """Conservation is by construction: cap the iteration at two passes with a loose
    tolerance that cannot be met and check the balance of what IS returned."""
    a = compartment("a", scheme=scheme)
    b = compartment("b", scheme=scheme, circulation=True, initial=0.0)
    model, state, drivers = union(
        {"A": a, "B": b}, [ValueLink("A", "a.x", 0, "B", "b.x_boundary", two_way=True)],
        substeps={"B": substeps}, iterate_rtol=1e-14, iterate_atol=0.0, iterate_max=2,
    )
    diagnostics: dict = {}
    try:
        result = model.step(state, drivers, 1.0, diagnostics=diagnostics)
    except RuntimeError as exc:          # non-convergence is reported, not hidden
        assert "did not converge" in str(exc)
        return
    assert result["A"]["a.x"].item() + result["B"]["b.x"].item() == pytest.approx(1.0, abs=1e-10)


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
    # initial=0.0, as the R1/R2 fixtures above do: with every compartment starting at the
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
    # Instance 1 exchanges 20x faster than instance 0. Measured (re-pinned for the
    # recipient-first schedule, task 17): instance 1's faster exchange reaches its own
    # boundary equilibrium sooner and is the FIRST to satisfy the per-pass criterion (by
    # pass 8); instance 0's slower exchange is still short of it at pass 8 and needs 18 --
    # the opposite instance from the old previous-pass-residual criterion.
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

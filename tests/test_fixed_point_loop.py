"""`noodl._fixed_point_loop`: the pieces `Model._iterate` and `CoupledModel._iterate` share."""

import torch

from noodl._fixed_point_loop import (
    Verdict,
    confirm_gradient_dependence,
    detached,
    run_passes,
    unsolved_mask,
)

F64 = torch.float64


def test_detached_detaches_every_nested_tensor_and_keeps_the_rest():
    p = torch.tensor(2.0, dtype=F64, requires_grad=True)
    diag = {"a": {"x": p * 3.0, "n": 4, "s": "text"}, "b": p + 1.0, "c": [p]}
    out = detached(diag)
    assert not out["a"]["x"].requires_grad and out["a"]["x"].item() == 6.0
    assert out["a"]["n"] == 4 and out["a"]["s"] == "text"
    assert not out["b"].requires_grad
    assert out["c"] is diag["c"]                 # only dicts are walked, as before the move
    assert diag["b"].requires_grad               # the input is not modified


def test_unsolved_mask_is_none_when_every_named_solve_converged():
    diag = {"a": {"converged": torch.tensor([True, True])}, "b": {}, "t": {"substeps": 1}}
    assert unsolved_mask(diag, ["a", "b", "missing"]) is None


def test_unsolved_mask_ors_the_failures_of_the_named_layers_only():
    diag = {
        "a": {"converged": torch.tensor([True, False, True])},
        "b": {"converged": torch.tensor([False, True, True])},
        "c": {"converged": torch.tensor([False, False, False])},     # not named
    }
    assert unsolved_mask(diag, ["a", "b"]).tolist() == [True, True, False]
    assert unsolved_mask({"a": {"converged": False}}, ["a"]).item() is True


# --- run_passes / confirm_gradient_dependence on toy problems, with no Model -------------

class _Toy:
    """z -> g(z), judged on |g(z) - z| <= tol per instance and relaxed by `w`. `g` gets the
    fed iterate; every call records the grad mode it ran under."""

    def __init__(self, g, *, tol=1e-12, w=1.0, verdict_from=1, stall_at=None):
        self.g, self.tol, self.w = g, tol, w
        self.verdict_from, self.stall_at = verdict_from, stall_at
        self.runs: list[bool] = []
        self.judged: list[bool] = []
        self.advanced: list[bool] = []

    def run(self, z):
        self.runs.append(torch.is_grad_enabled())
        return self.g(z)

    def depends_on_grad(self, out):
        return out.requires_grad

    def detach(self, z, out):
        return z.detach(), out.detach()

    def judge(self, z, out, *, needs_adjoint):
        self.judged.append(torch.is_grad_enabled())
        if len(self.runs) < self.verdict_from:
            return None
        converged = (out - z).abs() <= self.tol
        stop = self.stall_at is not None and len(self.runs) >= self.stall_at
        return Verdict(converged=converged, change={"z": (out - z).abs()}, stop_early=stop)

    def advance(self, z, out, verdict):
        self.advanced.append(torch.is_grad_enabled())
        return z + self.w * (out - z)


def test_a_scalar_contraction_converges_and_certifies_its_last_fed_iterate():
    toy = _Toy(lambda z: 0.5 * z + 1.0, w=1.0)
    outcome = run_passes(toy, torch.tensor(0.0, dtype=F64), max_passes=100)
    assert outcome.converged and not outcome.stopped_early
    assert abs(outcome.certified.item() - 2.0) < 1e-11
    assert torch.equal(outcome.out, 0.5 * outcome.certified + 1.0)
    assert outcome.passes == len(toy.runs) == len(toy.judged) == len(toy.advanced) + 1
    # pass 1 runs under grad (grad mode is on), every other pass and all judging without
    assert toy.runs == [True] + [False] * (outcome.passes - 1)
    assert not any(toy.judged) and not any(toy.advanced)


def test_the_verdict_is_per_instance_and_the_loop_runs_until_all_converge():
    rate = torch.tensor([0.1, 0.9], dtype=F64)
    toy = _Toy(lambda z: rate * z + 1.0)
    outcome = run_passes(toy, torch.zeros(2, dtype=F64), max_passes=1000)
    assert outcome.verdict.converged.shape == (2,) and outcome.converged
    fast = run_passes(_Toy(lambda z: 0.1 * z + 1.0), torch.zeros((), dtype=F64),
                      max_passes=1000)
    assert fast.passes < outcome.passes


def test_no_verdict_on_early_passes_and_none_needed_to_run_on():
    toy = _Toy(lambda z: 0.5 * z + 1.0, verdict_from=2)
    outcome = run_passes(toy, torch.tensor(0.0, dtype=F64), max_passes=100)
    assert outcome.converged and outcome.passes >= 2


def test_non_convergence_stops_at_the_budget_without_a_next_iterate():
    toy = _Toy(lambda z: z + 1.0)
    outcome = run_passes(toy, torch.tensor(0.0, dtype=F64), max_passes=4)
    assert outcome.passes == 4 and not outcome.converged and not outcome.stopped_early
    assert len(toy.advanced) == 3                    # none formed after the last pass
    assert outcome.certified.item() == 3.0 and outcome.out.item() == 4.0
    assert outcome.needs_adjoint is False


def test_stop_early_ends_the_loop_and_is_reported():
    toy = _Toy(lambda z: z + 1.0, stall_at=3)
    outcome = run_passes(toy, torch.tensor(0.0, dtype=F64), max_passes=50)
    assert outcome.passes == 3 and outcome.stopped_early and not outcome.converged
    assert outcome.certified.item() == 2.0           # the fed iterate of the last pass run


def test_pass_one_observes_a_gradient_and_drops_its_graph():
    p = torch.tensor(0.5, dtype=F64, requires_grad=True)
    toy = _Toy(lambda z: p * z + 1.0)
    outcome = run_passes(toy, torch.tensor(0.0, dtype=F64), max_passes=200)
    assert outcome.needs_adjoint and outcome.converged
    assert not outcome.out.requires_grad and not outcome.certified.requires_grad
    calls = len(toy.runs)
    assert confirm_gradient_dependence(toy, outcome, roots=(p,)) is True
    assert len(toy.runs) == calls                    # pass 1's answer: no probe pass


def test_the_gradient_found_only_at_the_certified_point_is_confirmed_there():
    """The R01 branch closure: the map depends on `p` only once z has moved past 1."""
    p = torch.tensor(0.25, dtype=F64, requires_grad=True)

    def g(z):
        if z.item() > 1.0:
            return 1.5 + p * (z - z.detach())       # 1.5, but now carrying p's graph
        return 0.5 * z + 1.0

    toy = _Toy(g)
    outcome = run_passes(toy, torch.tensor(0.0, dtype=F64), max_passes=200)
    assert outcome.converged and not outcome.needs_adjoint
    passes = outcome.passes
    assert confirm_gradient_dependence(toy, outcome, roots=(p,)) is True
    assert len(toy.runs) == passes + 1 and toy.runs[-1] is True   # one probe, under grad
    assert outcome.passes == passes                                # not counted


def test_nothing_requiring_grad_reachable_skips_the_probe():
    toy = _Toy(lambda z: 0.5 * z + 1.0)
    outcome = run_passes(toy, torch.tensor(0.0, dtype=F64), max_passes=200)
    calls = len(toy.runs)
    assert confirm_gradient_dependence(toy, outcome, roots=(toy, 1.0)) is False
    assert len(toy.runs) == calls


def test_grad_mode_off_runs_every_pass_without_a_graph_and_never_probes():
    p = torch.tensor(0.5, dtype=F64, requires_grad=True)
    toy = _Toy(lambda z: p * z + 1.0)
    with torch.no_grad():
        outcome = run_passes(toy, torch.tensor(0.0, dtype=F64), max_passes=200)
        assert not outcome.needs_adjoint and not any(toy.runs)
        calls = len(toy.runs)
        assert confirm_gradient_dependence(toy, outcome, roots=(p,)) is False
    assert len(toy.runs) == calls

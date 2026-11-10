"""`KirchhoffLayer(fallback=)`: the ladder a potential solve climbs when Newton fails."""

import pytest
import torch

from noodl.elements import Conductance, PowerLaw
from noodl.layers import kirchhoff
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.model import Model
from noodl.solvers import ConvergenceError, GroundingError, SingularSystemError, implicit
from noodl.solvers.continuation import shift_diagonal
from noodl.topology import Network
from noodl.withdrawals import NodeWithdrawal
from tests.layers.test_mna import _mixed_layer

F64 = torch.float64
PB = torch.tensor([0.0], dtype=F64)
SOURCES = torch.tensor([0.0, 0.3, -0.1], dtype=F64)
FAR = torch.tensor([1e4, -1e4], dtype=F64)      # a warm start Newton cannot use in 10 steps


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _net():
    net = Network(dtype=F64)
    for n in ("ambient", "z1", "z2"):
        net.add_node(n)
    net.add_edge("ambient", "z1", kind="airpath")
    net.add_edge("z1", "z2", kind="airpath")
    net.add_edge("z2", "ambient", kind="airpath")
    return net


def _layer(fallback=(), learnable=False, **kwargs):
    law = PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65, learnable=learnable)
    return KirchhoffLayer(_net(), "zones", [law], boundary=["ambient"], fallback=fallback,
                          **kwargs)


def _fail_warm_newton(monkeypatch):
    """Newton fails, without iterating, on every solve that starts from a given `phi0`
    (both Newton entries: the primal solve's and the differentiable one's). Returns the
    list of the failed solves."""
    real_start = KirchhoffLayer._starting_point
    real_newton = kirchhoff.newton
    current = {"warm": False}
    failed = []

    def starting_point(self, phi0, *args, **kwargs):
        current["warm"] = phi0 is not None
        return real_start(self, phi0, *args, **kwargs)

    def newton(residual, operator, x0, **kwargs):
        if current["warm"]:
            failed.append(1)
            result = real_newton(residual, operator, x0,
                                 **{**kwargs, "max_iter": 0, "on_failure": "return"})
            result.converged = torch.zeros_like(result.converged)
            if kwargs.get("on_failure", "raise") == "raise":
                raise ConvergenceError("forced to fail")
            return result
        return real_newton(residual, operator, x0, **kwargs)

    monkeypatch.setattr(KirchhoffLayer, "_starting_point", starting_point)
    monkeypatch.setattr(kirchhoff, "newton", newton)
    monkeypatch.setattr(implicit, "newton", newton)
    return failed


# ------------------------------------------------------------------ the rungs and options
def test_unknown_or_repeated_rungs_and_a_bare_string_are_refused_by_name():
    with pytest.raises(ValueError, match=r"unknown fallback rung\(s\) \['restart'\]"):
        _layer(fallback=("restart",))
    with pytest.raises(ValueError, match="names a rung twice"):
        _layer(fallback=("linear_guess", "linear_guess"))
    with pytest.raises(TypeError, match=r"\('linear_guess',\)"):
        _layer(fallback="linear_guess")
    with pytest.raises(ValueError, match="unknown fallback rung"):
        _layer().solve(PB, {}, SOURCES, fallback=("restart",))


def test_no_fallback_takes_the_plain_solve(monkeypatch):
    """`fallback=()` (the default) never enters the ladder, and its diagnostics gain no
    entry; the result is the plain solve's, bit for bit."""
    entered = []
    real = KirchhoffLayer._solve_with_fallback

    def spy(self, *args, **kwargs):
        entered.append(1)
        return real(self, *args, **kwargs)

    reference = _layer(fallback=("linear_guess",)).solve(PB, {}, SOURCES, fallback=())
    monkeypatch.setattr(KirchhoffLayer, "_solve_with_fallback", spy)
    layer = _layer()
    diagnostics: dict = {}
    phi, q = layer.solve(PB, {}, SOURCES, diagnostics=diagnostics)
    assert entered == []
    assert "fallback" not in diagnostics
    assert torch.equal(phi, reference[0]) and torch.equal(q, reference[1])
    with pytest.raises(ConvergenceError):
        layer.solve(PB, {}, SOURCES, phi0=FAR, max_iter=10)
    assert entered == []


def test_a_converged_solve_reports_no_fallback():
    diagnostics: dict = {}
    _layer(fallback=("linear_guess", "storage_ramp")).solve(PB, {}, SOURCES,
                                                            diagnostics=diagnostics)
    assert diagnostics["fallback"] is None and diagnostics["fallback_iterations"] == 0


@pytest.mark.parametrize("kw", [{}, {"differentiable": False},
                                {"differentiable": False, "on_failure": "return"},
                                {"on_failure": "return_primal", "differentiable": False}])
def test_a_failed_warm_solve_retried_from_the_linear_guess_is_the_cold_solve(monkeypatch,
                                                                             kw):
    layer = _layer(fallback=("linear_guess",))
    cold = layer.solve(PB, {}, SOURCES, **kw, diagnostics={})
    warm = layer.solve(PB, {}, SOURCES, phi0=cold[0][layer.interior] + 1.0, **kw,
                       diagnostics={})
    failed = _fail_warm_newton(monkeypatch)
    diagnostics: dict = {}
    phi, q = layer.solve(PB, {}, SOURCES, phi0=cold[0][layer.interior] + 1.0, **kw,
                         diagnostics=diagnostics)
    assert failed == [1]
    assert diagnostics["fallback"] == "linear_guess"
    assert diagnostics["fallback_iterations"] == 0
    assert bool(diagnostics["converged"])
    assert torch.equal(phi, cold[0]) and torch.equal(q, cold[1])
    assert not torch.equal(phi, warm[0])       # the warm solve lands elsewhere in round-off


def test_the_linear_guess_is_skipped_when_the_failed_solve_started_there(monkeypatch):
    """A cold solve of a layer without flow unknowns starts from the linear guess: the
    rung would only repeat it, so the error stands."""
    layer = _layer(fallback=("linear_guess",))
    failed = []
    real_newton = kirchhoff.newton

    def newton(residual, operator, x0, **kwargs):
        failed.append(1)
        return real_newton(residual, operator, x0, **{**kwargs, "max_iter": 0})

    monkeypatch.setattr(kirchhoff, "newton", newton)
    monkeypatch.setattr(implicit, "newton", newton)
    with pytest.raises(ConvergenceError, match=r"after 0 iterations, residual norms \[[^]]*\]$"):
        layer.solve(PB, {}, SOURCES)
    assert failed == [1]


def test_when_no_rung_converges_the_first_failure_stands(monkeypatch):
    layer = _layer(fallback=("linear_guess",))
    real_newton = kirchhoff.newton

    def newton(residual, operator, x0, **kwargs):
        return real_newton(residual, operator, x0, **{**kwargs, "max_iter": 0})

    monkeypatch.setattr(kirchhoff, "newton", newton)
    monkeypatch.setattr(implicit, "newton", newton)
    with pytest.raises(ConvergenceError, match=r"after 0 iterations.*; the fallback "
                                               r"\['linear_guess'\] did not converge either"):
        layer.solve(PB, {}, SOURCES, phi0=FAR)
    diagnostics: dict = {}
    phi, _ = layer.solve(PB, {}, SOURCES, phi0=FAR, differentiable=False,
                         on_failure="return", diagnostics=diagnostics)
    assert diagnostics["fallback"] is None and not bool(diagnostics["converged"])
    assert torch.equal(phi[layer.interior], FAR)       # the first solve's own iterate


def test_an_error_other_than_non_convergence_passes_through(monkeypatch):
    layer = _layer(fallback=("linear_guess", "storage_ramp"))
    calls = []

    def newton(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("something else")

    monkeypatch.setattr(kirchhoff, "newton", newton)
    monkeypatch.setattr(implicit, "newton", newton)
    with pytest.raises(RuntimeError, match="^something else$"):
        layer.solve(PB, {}, SOURCES, phi0=FAR)
    assert calls == [1]


# ------------------------------------------------------------------ the storage ramp
def test_the_storage_ramp_rescues_a_solve_from_a_far_warm_start():
    reference, _ = _layer().solve(PB, {}, SOURCES, atol=1e-13, rtol=0.0)
    with pytest.raises(ConvergenceError):
        _layer().solve(PB, {}, SOURCES, phi0=FAR, max_iter=10, atol=1e-13, rtol=0.0)
    diagnostics: dict = {}
    phi, _ = _layer(fallback=("storage_ramp",)).solve(
        PB, {}, SOURCES, phi0=FAR, max_iter=10, atol=1e-13, rtol=0.0, diagnostics=diagnostics)
    assert diagnostics["fallback"] == "storage_ramp"
    assert diagnostics["fallback_iterations"] > 0
    torch.testing.assert_close(phi, reference, rtol=1e-10, atol=1e-10)


def test_the_storage_ramp_on_a_layer_without_flow_unknowns_certifies_spd():
    """The ramp's operator is the layer's Jacobian with a non-negative shift: the SPD
    certificate (and with it the symmetric solvers) still applies, also over a withdrawal's
    own shift."""
    class Leak(NodeWithdrawal):
        def flow(self, phi_nodes, drivers=None):
            return 1e-3 * phi_nodes

    net = _net()
    layer = KirchhoffLayer(net, "zones", [PowerLaw(torch.tensor([0.01, 0.02, 0.01]), 0.65)],
                           boundary=["ambient"],
                           withdrawals=[Leak(torch.tensor([net.node_index("z2")]))])
    x = torch.tensor([3.0, -2.0], dtype=F64)
    for op in (_layer()._operator(x, PB, {}), layer._operator(x, PB, {})):
        g = 0.5 * op.diagonal().abs()
        shifted = shift_diagonal(op, g)
        assert bool(torch.as_tensor(shifted.spd_certificate()).all())
        torch.testing.assert_close(shifted.assemble(), op.assemble() + torch.diag(g))


def test_a_differentiable_solve_rescued_by_the_storage_ramp_has_the_exact_gradient():
    layer = _layer(fallback=("storage_ramp",), learnable=True)
    sources = SOURCES.clone().requires_grad_(True)

    def solve(s):
        diagnostics: dict = {}
        phi, _ = layer.solve(PB, {}, s, phi0=FAR, max_iter=10, atol=1e-13, rtol=0.0,
                             diagnostics=diagnostics)
        assert diagnostics["fallback"] == "storage_ramp"
        return phi

    assert torch.autograd.gradcheck(solve, (sources,), eps=1e-6, atol=1e-6)
    # and the law's parameter: the same gradient as the solve that needed no fallback
    phi = solve(SOURCES)
    params = list(layer._laws[0].parameters())
    grad = torch.autograd.grad(phi.sum(), params)
    plain = layer.solve(PB, {}, SOURCES, atol=1e-13, rtol=0.0, fallback=())[0]
    expected = torch.autograd.grad(plain.sum(), params)
    assert len(params) == 2                            # the coefficients and the exponent
    for g, e in zip(grad, expected, strict=True):
        torch.testing.assert_close(g, e, rtol=1e-8, atol=1e-12)


def test_the_ladder_on_a_layer_with_flow_unknowns():
    """The linear guess is not the cold start there (zeros), so the rung runs from a cold
    failure too; the storage sits on the potential rows."""
    _, layer = _mixed_layer()
    pb = torch.tensor([0.4])
    sources = torch.tensor([0.0, 0.1, -0.2, 0.05, 0.0, 0.3])
    reference, q_ref = layer.solve(pb, {}, sources, differentiable=False, atol=1e-13, rtol=0)
    far = torch.full((5,), 50.0)
    with pytest.raises(ConvergenceError):
        layer.solve(pb, {}, sources, phi0=far, max_iter=12, atol=1e-13, rtol=0)
    for rung in ("linear_guess", "storage_ramp"):
        diagnostics: dict = {}
        phi, q = layer.solve(pb, {}, sources, phi0=far, max_iter=12, atol=1e-13, rtol=0,
                             diagnostics=diagnostics, fallback=(rung,))
        assert diagnostics["fallback"] == rung
        torch.testing.assert_close(phi, reference, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(q, q_ref, rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------ in a Model
def test_an_iterate_coupled_step_whose_first_pass_fails_reports_the_fallback(monkeypatch):
    net = Network(dtype=F64)
    net.add_node("out")
    net.add_node("z")
    net.add_edge("out", "z", kind="c")
    layer = KirchhoffLayer(net, "vent", [Conductance(torch.tensor([1e-3]), kind="c")],
                           boundary=["out"], fallback=("linear_guess",))
    model = Model(net, {"vent": layer}, coupling="iterate", iterate_tol={"vent": 1e-12})
    drivers = {"vent.phi_boundary": torch.tensor([0.0]),
               "vent.sources": torch.tensor([0.0, 0.02])}
    state = {"vent.phi": torch.tensor([0.0, 10.0]), "vent.q": torch.tensor([0.01])}
    reference: dict = {}
    expected = model.step(state, drivers, 300.0, diagnostics=reference)
    assert reference["fallbacks"] == {"vent": 0}
    real_newton = kirchhoff.newton
    calls = []

    def newton(residual, operator, x0, **kwargs):
        calls.append(1)
        if len(calls) == 1:                    # the first pass's warm solve
            kwargs = {**kwargs, "max_iter": 0}
        return real_newton(residual, operator, x0, **kwargs)

    monkeypatch.setattr(kirchhoff, "newton", newton)
    monkeypatch.setattr(implicit, "newton", newton)
    diagnostics: dict = {}
    out = model.step(state, drivers, 300.0, diagnostics=diagnostics)
    assert bool(torch.as_tensor(diagnostics["converged"]).all())
    assert diagnostics["fallbacks"] == {"vent": 1}
    torch.testing.assert_close(out["vent.phi"], expected["vent.phi"], rtol=1e-12, atol=1e-12)


def test_a_pingpong_step_reports_the_fallbacks_of_its_pass(monkeypatch):
    layer = _layer(fallback=("linear_guess",))
    model = Model(layer.net, {"zones": layer})
    drivers = {"zones.phi_boundary": PB, "zones.sources": SOURCES}
    state = {"zones.phi": torch.tensor([0.0, 1.0, -1.0])}
    diagnostics: dict = {}
    model.step(state, drivers, 1.0, diagnostics=diagnostics)
    assert diagnostics["fallbacks"] == {"zones": 0}
    _fail_warm_newton(monkeypatch)
    model.step(state, drivers, 1.0, diagnostics=diagnostics)
    assert diagnostics["fallbacks"] == {"zones": 1}


# ------------------------------------------------------------------ errors under the ladder
def _mixed_warm_failure():
    """The mixed layer with flow unknowns and a warm start it cannot use in 12 iterations."""
    _, layer = _mixed_layer()
    args = (torch.tensor([0.4]), {}, torch.tensor([0.0, 0.1, -0.2, 0.05, 0.0, 0.3]))
    kw = dict(phi0=torch.full((5,), 50.0), max_iter=12, atol=1e-13, rtol=0)
    return layer, args, kw


def test_a_rung_that_raises_is_a_failed_rung_and_the_first_failure_stands(monkeypatch):
    """A linear guess that cannot be formed (singular there) fails its rung; the error is
    still the first solve's non-convergence, and a return mode still returns it."""
    layer, args, kw = _mixed_warm_failure()

    def singular_guess(self, *a, **k):
        raise self._singular_error("dc")

    monkeypatch.setattr(KirchhoffLayer, "_linear_guess", singular_guess)
    with pytest.raises(ConvergenceError, match=r"failed to converge after 12 iterations.*; "
                                               r"the fallback \['linear_guess'\] did not "
                                               r"converge either: .*singular"):
        layer.solve(*args, fallback=("linear_guess",), **kw)
    diagnostics: dict = {}
    phi, _ = layer.solve(*args, fallback=("linear_guess",), differentiable=False,
                         on_failure="return_primal", diagnostics=diagnostics, **kw)
    assert diagnostics["fallback"] is None and not bool(diagnostics["converged"])
    assert torch.isfinite(phi).all()


def test_the_rank_test_waits_for_the_ladder(monkeypatch):
    """A failed warm solve whose last Jacobian the rank test calls singular: without a
    ladder the singular error is raised at once, as before; with one the rung rescues it."""
    layer, args, kw = _mixed_warm_failure()
    monkeypatch.setattr(kirchhoff, "singular", lambda *a: True)
    with pytest.raises(RuntimeError, match="is singular") as info:
        layer.solve(*args, **kw)
    assert not isinstance(info.value, ConvergenceError)
    diagnostics: dict = {}
    phi, _ = layer.solve(*args, fallback=("linear_guess",), diagnostics=diagnostics, **kw)
    assert diagnostics["fallback"] == "linear_guess"
    assert bool(diagnostics["converged"]) and torch.isfinite(phi).all()


def test_a_singular_layer_with_a_ladder_still_raises_the_singular_error():
    from tests._circuit_laws import Source

    net = Network(dtype=F64)
    for n in ("gnd", "a"):
        net.add_node(n)
    net.add_edge("a", "gnd", kind="source")
    net.add_edge("a", "gnd", kind="source")
    net.add_edge("a", "gnd", kind="r")
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([1.0, 2.0])),
                                      Conductance(torch.tensor([1.0]), kind="r")],
                           boundary=["gnd"], fallback=("linear_guess", "storage_ramp"))
    for kw in ({}, {"method": "direct"}):
        with pytest.raises(SingularSystemError, match=r"singular.*'source'") as info:
            layer.solve(torch.zeros(1), differentiable=False, **kw)
        assert not isinstance(info.value, ConvergenceError)


def test_a_rung_whose_law_raises_another_error_passes_it_through(monkeypatch):
    """Only non-convergence, a singular system or a grounding refusal fails a rung; a law's
    own error on a rung (a shape bug, say) is not reported as "the fallback did not converge
    either" but passes through as it is."""
    failed = _fail_warm_newton(monkeypatch)
    layer = _layer(fallback=("linear_guess",))
    real = KirchhoffLayer._starting_point

    def broken(self, phi0, *args, **kwargs):
        if phi0 is None:
            raise RuntimeError("a law's own bug")
        return real(self, phi0, *args, **kwargs)

    monkeypatch.setattr(KirchhoffLayer, "_starting_point", broken)
    with pytest.raises(RuntimeError, match="^a law's own bug$"):
        layer.solve(PB, {}, SOURCES, phi0=FAR, differentiable=False)
    assert failed


def test_a_floating_node_is_a_grounding_error():
    net = _net()
    net.add_node("island")
    net.add_node("island2")
    net.add_edge("island", "island2", kind="airpath")
    law = PowerLaw(torch.tensor([0.01, 0.02, 0.01, 0.01], dtype=F64), 0.65)
    layer = KirchhoffLayer(net, "zones", [law], boundary=["ambient"])
    with pytest.raises(GroundingError, match="floating nodes"):
        layer.solve(PB, {}, torch.zeros(5, dtype=F64))

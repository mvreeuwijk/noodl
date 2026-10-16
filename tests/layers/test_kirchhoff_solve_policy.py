"""K0 characterization of `KirchhoffLayer.solve`'s failure and differentiation policy.

The policy (how `on_failure`, `differentiable`, `analytic_primal`, `diagnostics=` and whether
anything differentiable reaches the solve combine) is implemented separately for the classic
(nodal Newton) path and the MNA path (real and phasor). This file runs the full matrix on one
small nonlinear layer of each kind, with and without a forced non-convergence, and compares
what comes back with `tests/golden/orchestration.json["solve_policy"]`, generated on the
unrefactored code by `benchmarks/regenerate_orchestration_golden.py`.

Per combination the record is the exception type and the first 80 characters of its message,
or the returned `(phi, q)` (values, dtype, `requires_grad`), the diagnostics dict, and the
gradient of a fixed loss with respect to the one leaf tensor that reaches the solve. The
comparison is exact. Extend the matrix (new keywords, new layer kinds) and regenerate only for
a deliberate change of behaviour.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from typing import Any

import pytest
import torch

from noodl.drives import ConstantDrive
from noodl.elements import ConstitutiveLaw, ImplicitLaw, PowerLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network
from tests._circuit_laws import Source
from tests.golden import load_golden
from tests.test_golden_orchestration import first_difference, outcome, ser

F64 = torch.float64
C128 = torch.complex128

#: The axes of the matrix. `reaches` is whether a tensor that requires grad is passed to the
#: solve; `grad_mode` the ambient `torch.is_grad_enabled()`; `forced` a one-iteration Newton
#: budget that cannot converge.
ON_FAILURE = (None, "raise", "return", "return_primal")
DIAGNOSTICS = (False, True)
DIFFERENTIABLE = (True, False)
ANALYTIC_PRIMAL = (False, True)
REACHES = (True, False)
FORCED = (False, True)
GRAD_MODE = (True, False)


# ------------------------------------------------------------------ one layer per kind


def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


def _classic(analytic_primal: bool):
    """ambient_w -> z1 -> z2 -> ambient_l, a power law per edge, a wind drive."""
    net = _net(["w", "z1", "z2", "l"], [("w", "z1", "airpath"), ("z1", "z2", "airpath"),
                                        ("z2", "l", "airpath")])
    element = PowerLaw(torch.tensor([0.010, 0.008, 0.012], dtype=F64), 0.65,
                       dp_transition=1e-6)
    layer = KirchhoffLayer(net, "e", [element], [ConstantDrive(kind="airpath", key="wind")],
                           boundary=["w", "l"], analytic_primal=analytic_primal)
    base = torch.tensor([12.0, 0.0, 0.0], dtype=F64)

    def call(reaches):
        leaf = torch.tensor(1.0, dtype=F64, requires_grad=reaches)
        return (torch.zeros(2, dtype=F64), {"wind": base * leaf}, None), {}, leaf

    return layer, call


def _mna(analytic_primal: bool):
    """A potential source in series with a nonlinear conductance and a cubic load (MNA)."""
    net = _net(["g", "a", "b"], [("a", "g", "source"), ("a", "b", "r"), ("b", "g", "load")])
    layer = KirchhoffLayer(
        net, "e", [Source(torch.tensor([1.0], dtype=F64)),
                   PowerLaw(torch.tensor([0.5], dtype=F64), 0.65, kind="r",
                            dp_transition=1e-6),
                   PowerLaw(torch.tensor([0.3], dtype=F64), 0.8, kind="load",
                            dp_transition=1e-6)],
        boundary=["g"], analytic_primal=analytic_primal)
    assert layer.mna

    def call(reaches):
        leaf = torch.tensor(1.0, dtype=F64, requires_grad=reaches)
        sources = torch.tensor([0.0, 0.2, -0.1], dtype=F64) * leaf
        return (torch.zeros(1, dtype=F64), {}, sources), {}, leaf

    return layer, call


class _ConstantPower(ConstitutiveLaw):
    """q = conj(S / dp): a constant-power load (not holomorphic in dp)."""

    phasor_capable = True

    def __init__(self, S, kind="load"):
        super().__init__(kind)
        self.S = self._param(S, False)

    def flow(self, dp, drivers=None):
        return torch.conj(self.S / dp)


class _Impedance(ImplicitLaw):
    phasor_capable = True

    def __init__(self, Z, kind="line"):
        super().__init__(kind)
        self.Z = self._param(Z, False)

    def residual(self, dp, q, drivers, ctx):
        return dp - self.Z * q


def _phasor(analytic_primal: bool):
    net = _net(["g", "s", "a"], [("s", "g", "source"), ("s", "a", "line"), ("a", "g", "load")])
    layer = KirchhoffLayer(
        net, "e", [Source(torch.tensor([1.0], dtype=F64)),
                   _Impedance(torch.tensor([0.01 + 0.05j], dtype=C128)),
                   _ConstantPower(torch.tensor([0.5 + 0.2j], dtype=C128))],
        boundary=["g"], phasor=True, analytic_primal=analytic_primal)
    phi0 = torch.ones(len(layer.interior), dtype=C128)

    def call(reaches):
        leaf = torch.tensor(1.0, dtype=F64, requires_grad=reaches)
        drivers = {"e.frequency": torch.tensor(50.0, dtype=F64) * leaf}
        return (torch.zeros(1, dtype=F64), drivers, None, phi0), {}, leaf

    return layer, call


LAYERS: dict[str, Callable[[bool], tuple[KirchhoffLayer, Callable]]] = {
    "classic": _classic, "mna": _mna, "phasor": _phasor,
}


# ------------------------------------------------------------------ the matrix


def _run(kind: str, *, differentiable, on_failure, with_diagnostics, analytic_primal,
         reaches, forced, grad_mode) -> dict[str, Any]:
    layer, call = LAYERS[kind](analytic_primal)
    args, _, leaf = call(reaches)
    kwargs: dict[str, Any] = {"differentiable": differentiable}
    if on_failure is not None:
        kwargs["on_failure"] = on_failure
    diagnostics: dict | None = {} if with_diagnostics else None
    if with_diagnostics:
        kwargs["diagnostics"] = diagnostics
    if forced:
        kwargs.update(max_iter=1, atol=1e-13, rtol=0.0)

    def go():
        with torch.set_grad_enabled(grad_mode):
            phi, q = layer.solve(*args, **kwargs)
            rec: dict[str, Any] = {
                "phi": ser(phi), "q": ser(q),
                "phi_requires_grad": bool(phi.requires_grad),
                "q_requires_grad": bool(q.requires_grad),
                "diagnostics": None if diagnostics is None else ser(diagnostics),
                "diagnostics_keys": None if diagnostics is None else sorted(diagnostics),
            }
            if phi.requires_grad and leaf.requires_grad:
                loss = phi.abs().sum() + q.abs().sum()
                (grad,) = torch.autograd.grad(loss, leaf, allow_unused=True)
                rec["grad"] = ser(grad)
            return rec

    return outcome(go)


def build_matrix(kind: str) -> dict[str, Any]:
    """Every combination of the policy axes for one layer kind, keyed by a readable id."""
    out: dict[str, Any] = {}
    for (diff, on_failure, diag, analytic, reaches, forced, mode) in itertools.product(
            DIFFERENTIABLE, ON_FAILURE, DIAGNOSTICS, ANALYTIC_PRIMAL, REACHES, FORCED,
            GRAD_MODE):
        key = (f"diff={diff}|on_failure={on_failure}|diag={diag}|analytic={analytic}|"
               f"reaches={reaches}|forced={forced}|grad_mode={mode}")
        out[key] = _run(kind, differentiable=diff, on_failure=on_failure,
                        with_diagnostics=diag, analytic_primal=analytic, reaches=reaches,
                        forced=forced, grad_mode=mode)
    return out


def build_all() -> dict[str, Any]:
    return {kind: build_matrix(kind) for kind in LAYERS}


# ------------------------------------------------------------------ the tests


@pytest.fixture(scope="module")
def golden():
    return load_golden("orchestration")["solve_policy"]


@pytest.mark.parametrize("kind", sorted(LAYERS))
def test_the_solve_policy_matrix_matches_the_golden_record(kind, golden):
    actual = json.loads(json.dumps(build_matrix(kind)))
    expected = golden[kind]
    assert set(actual) == set(expected), "the matrix axes changed: regenerate the golden"
    bad = [(k, first_difference(actual[k], expected[k])) for k in actual
           if first_difference(actual[k], expected[k])]
    assert not bad, f"{len(bad)} of {len(actual)} combinations differ; first: {bad[0]}"


@pytest.mark.parametrize("kind", sorted(LAYERS))
def test_the_matrix_exercises_errors_successes_and_gradients(kind, golden):
    records = list(golden[kind].values())
    assert any("error" in r for r in records)
    assert any("ok" in r for r in records)
    assert any("grad" in r.get("ok", {}) for r in records)
    assert any(r.get("ok", {}).get("diagnostics_keys") for r in records)
    # a forced non-convergence is reported through the diagnostics where the policy allows it
    unconverged = [r["ok"]["diagnostics"] for k, r in golden[kind].items()
                   if "ok" in r and "forced=True" in k and r["ok"]["diagnostics"]]
    assert unconverged

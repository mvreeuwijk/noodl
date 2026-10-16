"""K0 characterization of the Model / CoupledModel orchestration (issue #22, S02).

Every case runs a small, deterministic model through `Model.step` / `Model.steady` /
`CoupledModel.step` and records what the orchestration returns: every state tensor, the
whole `diagnostics` dict (pass counts, `converged`, `max_change`, `adjoint`,
`adjoint_batched`, per-layer Newton figures), the gradient of a fixed scalar loss with
respect to designated parameters, and, for the failure cases, the exception type and the
first 80 characters of its message.

The reference is `tests/golden/orchestration.json`, generated on the unrefactored code by
`benchmarks/regenerate_orchestration_golden.py`. The comparison is exact (floats are
compared with `==`, so a reordered floating-point operation fails it). A refactor that moves
code must not change one number; if a difference appears, find the operation-order cause
instead of loosening the comparison. Regenerate the file only for a deliberate change of
behaviour, and say so in the commit.

Exactness is a property of the recording platform, not of the code. Set `NOODL_GOLDEN_EXACT=1`
(on the recording machine, to prove a refactor changed no number) and floats are compared with
`==`. Without it (the default: GitHub's Linux, Windows and macOS/ARM runners all differ from the
recording in the last ulp or two, and run to run on Windows) the discrete content stays
exact -- keys, shapes, dtypes, ints, booleans, strings, error types and messages, pass counts --
and floats are compared with `|a - b| <= RTOL * |b| + RTOL * max|tensor|`, RTOL = 1e-12. Why
1e-12: a different BLAS or vectorised reduction reassociates sums, which moves a result by a few
ulp (1e-16 relative) per operation; a short chain of solves and reductions accumulates 1e-15 to
1e-14, with a few orders of margin below 1e-12. A real behaviour change moves values by more
than ~1e-10 relative, which this still rejects. The `max|tensor|` term (the largest golden
entry of the same tensor) keeps entries that are zero or cancel to rounding from being compared
against a bare relative tolerance. Convergence figures (`residual_norm`, `residual`,
`max_change`) are rounding noise once a solve has converged (1e-16 .. 1e-12) and differ by
order-one factors between platforms: they are compared with an absolute floor of `NOISE_ATOL`
= 1e-8 as well, which still tells "converged" from "not converged".
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from noodl.couple import ValueLink, union
from noodl.drives import ConstantDrive
from noodl.elements import PowerLaw
from noodl.elements.base import ConstitutiveLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.reaction import FirstOrderDecay
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network
from tests.golden import load_golden

F64 = torch.float64


# ------------------------------------------------------------------ recording helpers


def ser(x: Any) -> Any:
    """A JSON-ready, exactly round-tripping form of tensors, numbers, dicts and lists."""
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.is_complex():
            data = torch.view_as_real(t).tolist()
        else:
            data = t.tolist()
        return {"dtype": str(t.dtype), "shape": list(t.shape), "data": data}
    if isinstance(x, dict):
        return {str(k): ser(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ser(v) for v in x]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    raise TypeError(f"cannot record {type(x).__name__}: {x!r}")


#: Exact float comparison (the recording environment); see the module docstring.
EXACT = os.environ.get("NOODL_GOLDEN_EXACT", "") not in ("", "0")
RTOL = 1e-12
NOISE_ATOL = 1e-8
_NOISE_KEYS = frozenset({"residual_norm", "residual", "max_change"})


def _close(a: float, b: float, scale: float, noise: bool) -> bool:
    """`a` (actual) against `b` (golden) within the documented platform tolerance."""
    if a == b or (math.isnan(a) and math.isnan(b)):
        return True
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False
    atol = RTOL * scale + (NOISE_ATOL if noise else 0.0)
    return abs(a - b) <= RTOL * abs(b) + atol


def _flat_floats(x: Any):
    if isinstance(x, list):
        for v in x:
            yield from _flat_floats(v)
    elif isinstance(x, float) and math.isfinite(x):
        yield abs(x)


def _same(a: Any, b: Any, scale: float = 0.0, noise: bool = False) -> bool:
    """Equality: exact, or (floats only, unless EXACT) within the platform tolerance.

    `scale` is the largest golden magnitude of the tensor being compared; `noise` marks a
    convergence figure."""
    if isinstance(a, dict):
        if not (isinstance(b, dict) and set(a) == set(b)):
            return False
        if EXACT:
            return all(_same(a[k], b[k]) for k in a)
        if set(a) == {"dtype", "shape", "data"}:  # a recorded tensor
            return (a["dtype"] == b["dtype"] and a["shape"] == b["shape"]
                    and _same(a["data"], b["data"], max(_flat_floats(b["data"]), default=0.0),
                              noise))
        return all(_same(a[k], b[k], scale, k in _NOISE_KEYS or noise) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(
            _same(x, y, scale, noise) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        if EXACT:
            return a == b or (math.isnan(a) and math.isnan(b))
        return _close(a, b, max(scale, abs(b) if scale == 0.0 else 0.0), noise)
    return type(a) is type(b) and a == b


def first_difference(a: Any, b: Any, path: str = "") -> str | None:
    """The path of the first entry where two recorded structures differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(a)} != {sorted(b)}"
        if not EXACT and set(a) == {"dtype", "shape", "data"}:
            return None if _same(a, b) else f"{path}: {a!r} != {b!r}"
        for k in a:
            d = first_difference(a[k], b[k], f"{path}/{k}") if not (
                not EXACT and k in _NOISE_KEYS) else (
                None if _same(a[k], b[k], 0.0, True) else f"{path}/{k}: {a[k]!r} != {b[k]!r}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if _same(a, b) else f"{path}: {a!r} != {b!r}"


def outcome(fn: Callable[[], dict]) -> dict:
    """Run `fn`; return its record, or the exception type and first 80 message characters."""
    try:
        return {"ok": fn()}
    except Exception as exc:  # noqa: BLE001 - the failure IS the characterised outcome
        return {"error": {"type": type(exc).__name__, "message": str(exc)[:80]}}


def grads_of(loss: torch.Tensor, params: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Back-propagate `loss`; the gradients of `params` (None where nothing reached)."""
    for p in params.values():
        p.grad = None
    loss.backward()
    return {k: p.grad for k, p in params.items()}


def record(state: dict, diag: dict | None = None, grads: dict | None = None) -> dict:
    # The golden file is written with sorted keys, so the order of the returned dicts (which
    # is part of the contract: a returned dict follows the order of its keys) is recorded.
    out: dict[str, Any] = {"state": ser(dict(state)), "state_order": list(state)}
    if diag is not None:
        out["diagnostics"] = ser(diag)
        out["diagnostics_order"] = list(diag)
    if grads is not None:
        out["grads"] = ser(grads)
    return out


# ------------------------------------------------------------------ the shared air/gas model


def _net() -> Network:
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name, z_ref=0.0)
    net.add_edge("ambient", "z1", kind="airpath", z_path=0.0)
    net.add_edge("z1", "z2", kind="airpath", z_path=0.0)
    net.add_edge("z2", "ambient", kind="airpath", z_path=0.0)
    return net


def _air_gas(*, n_species=1, closures=(), reactions=(), learnable=True, **kw):
    """ambient -> z1 -> z2 -> ambient: a power-law air layer and a transport layer."""
    net = _net()
    el = PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65, learnable=learnable)
    air = KirchhoffLayer(net, "air", [el], drives=[ConstantDrive("airpath", "wind")],
                         boundary=["ambient"])
    gas = TransportLayer(net, "gas", capacity=torch.tensor([50.0, 80.0], dtype=F64),
                         flow_kind="airpath", boundary=["ambient"], scheme="implicit",
                         n_species=n_species)
    model = Model(net, {"air": air, "gas": gas}, closures=closures, reactions=reactions, **kw)
    k = (n_species,) if n_species > 1 else ()
    drivers = {
        "air.phi_boundary": torch.zeros(1, dtype=F64),
        "wind": torch.tensor([5.0, 0.0, 0.0], dtype=F64),
        "gas.x_boundary": torch.tensor([1e-3, 2e-3, 4e-3][:n_species], dtype=F64).reshape(1, *k),
        "gas.sources": torch.zeros(3, *k, dtype=F64),
    }
    state = {"gas.x": torch.zeros(2, *k, dtype=F64)}
    return model, state, drivers, {"C": el.C}


def _gain(state, drivers):
    """Wind that grows with z1's concentration: a genuine flow <- transport feedback."""
    x1 = state["gas.x"][..., 0]
    zero = torch.zeros_like(x1)
    return {"wind": drivers["wind"] + torch.stack([2e3 * x1, zero, zero], dim=-1)}


def _conditional(state, drivers):
    """The gain switches on z1's concentration (a branch closure, the R01 case)."""
    x1 = state["gas.x"][..., 0]
    zero = torch.zeros_like(x1)
    gain = torch.where(x1 > 5e-4, torch.full_like(x1, 3e3), torch.full_like(x1, 1e3))
    return {"wind": drivers["wind"] + torch.stack([gain * x1, zero, zero], dim=-1)}


def _winds(b):
    w = torch.zeros(b, 3, dtype=F64)
    w[:, 0] = torch.linspace(2.0, 9.0, b, dtype=F64)
    return w


TIGHT = {"atol": 1e-14, "rtol": 1e-14}


def _step_record(model, state, drivers, params, *, dt=600.0, steps=1, key="gas.x", **kw):
    diag: dict = {}
    out = state
    for _ in range(steps):
        out = model.step(out, drivers, dt, diagnostics=diag, **kw)
    return record(out, diag, grads_of(out[key].sum(), params))


# ------------------------------------------------------------------ cases: Model


def case_pingpong_chain():
    model, state, drivers, params = _air_gas(closures=[_gain])
    return _step_record(model, state, drivers, params, steps=3)


def case_pingpong_two_species_reaction():
    model, state, drivers, params = _air_gas(
        n_species=2, reactions=[("gas", FirstOrderDecay(1e-4))])
    return _step_record(model, state, drivers, params, steps=2)


def case_pingpong_substeps():
    model, state, drivers, params = _air_gas(closures=[_gain], substeps={"gas": 4})
    return _step_record(model, state, drivers, params, steps=2)


def case_iterate_reaction_before_transport():
    model, state, drivers, params = _air_gas(
        closures=[_gain], reactions=[("gas", FirstOrderDecay(1e-3))],
        reaction_order="before_transport", **_iterate_kw())
    return _step_record(model, state, drivers, params)


def case_iterate_substeps_with_reaction():
    model, state, drivers, params = _air_gas(
        closures=[_gain], reactions=[("gas", FirstOrderDecay(1e-3))], substeps={"gas": 3},
        **_iterate_kw())
    return _step_record(model, state, drivers, params)


def case_pingpong_batch3():
    model, state, drivers, params = _air_gas(closures=[_gain])
    return _step_record(model, state, dict(drivers, wind=_winds(3)), params)


def _iterate_kw(**extra):
    return dict(coupling="iterate", iterate_tol={"gas": 1e-13}, iterate_max=80, **extra)


def case_iterate_relaxation_half():
    model, state, drivers, params = _air_gas(closures=[_gain], **_iterate_kw(
        iterate_relaxation=0.5))
    return _step_record(model, state, drivers, params)


def case_iterate_relaxation_one():
    model, state, drivers, params = _air_gas(closures=[_gain], **_iterate_kw(
        iterate_relaxation=1.0))
    return _step_record(model, state, drivers, params)


def case_iterate_steady_relaxation_half():
    model, state, drivers, params = _air_gas(closures=[_gain], **_iterate_kw())
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, **TIGHT)
    return record(out, diag, grads_of(out["gas.x"].sum(), params))


def case_iterate_batch3():
    model, state, drivers, params = _air_gas(closures=[_gain], **_iterate_kw())
    return _step_record(model, state, dict(drivers, wind=_winds(3)), params)


def case_iterate_batch_mixed_convergence():
    """Per-instance verdicts: one instance meets the tolerance, the other does not."""
    model, state, drivers, params = _air_gas(
        closures=[_gain], coupling="iterate", iterate_tol={"gas": 1e-12}, iterate_max=11)
    winds = torch.tensor([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=F64)
    diag: dict = {}
    out = model.steady(state, dict(drivers, wind=winds), diagnostics=diag,
                       differentiable=False, on_failure="return", **TIGHT)
    return record(out, diag)


def case_iterate_batch_mixed_convergence_raises():
    model, state, drivers, _ = _air_gas(
        closures=[_gain], coupling="iterate", iterate_tol={"gas": 1e-12}, iterate_max=11)
    winds = torch.tensor([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=F64)
    return record(model.steady(state, dict(drivers, wind=winds), **TIGHT))


def case_iterate_branch_closure_cold():
    """R01: the branch closure's gradient is found only at the certified point."""
    model, state, drivers, params = _air_gas(closures=[_conditional], **_iterate_kw())
    return _step_record(model, state, drivers, params, warm_start=False)


def case_iterate_branch_closure_distant_warm_start():
    model, state, drivers, params = _air_gas(closures=[_conditional], **_iterate_kw())
    state = dict(state, **{"air.phi": torch.tensor([0.0, 40.0, -30.0], dtype=F64)})
    return _step_record(model, state, drivers, params, warm_start=True)


def case_iterate_no_grad_reaches():
    """No learnable parameter and no grad-carrying input, in grad mode: the primal passes
    plus the one uncounted probe pass, which finds nothing."""
    model, state, drivers, _ = _air_gas(closures=[_gain], learnable=False, **_iterate_kw())
    diag: dict = {}
    out = model.step(state, drivers, 600.0, diagnostics=diag)
    return record(out, diag)


def case_iterate_grad_enabled_off():
    model, state, drivers, _ = _air_gas(closures=[_gain], **_iterate_kw())
    diag: dict = {}
    with torch.no_grad():
        out = model.step(state, drivers, 600.0, diagnostics=diag)
    return record(out, diag)


def case_iterate_accept_non_converged():
    model, state, drivers, params = _air_gas(
        closures=[_gain], coupling="iterate", iterate_tol={"gas": 1e-15}, iterate_max=2,
        iterate_on_limit="accept")
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, **TIGHT)
    return record(out, diag, grads_of(out["gas.x"].sum(), params))


def case_iterate_accept_non_converged_no_grad():
    model, state, drivers, _ = _air_gas(
        closures=[_gain], learnable=False, coupling="iterate", iterate_tol={"gas": 1e-15},
        iterate_max=2, iterate_on_limit="accept")
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, **TIGHT)
    return record(out, diag)


def case_iterate_raise_non_converged():
    model, state, drivers, _ = _air_gas(
        closures=[_gain], coupling="iterate", iterate_tol={"gas": 1e-15}, iterate_max=2)
    return record(model.steady(state, drivers, **TIGHT))


def case_iterate_on_failure_return_with_diagnostics():
    model, state, drivers, _ = _air_gas(
        closures=[_gain], coupling="iterate", iterate_tol={"gas": 1e-15}, iterate_max=2)
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, differentiable=False,
                       on_failure="return", **TIGHT)
    return record(out, diag)


def case_iterate_on_failure_return_differentiable_is_refused():
    model, state, drivers, _ = _air_gas(
        closures=[_gain], coupling="iterate", iterate_tol={"gas": 1e-15}, iterate_max=2)
    return record(model.steady(state, drivers, diagnostics={}, on_failure="return", **TIGHT))


def case_iterate_stall_stops_early():
    """One Newton iteration per pass: a pass that fails again from within tolerance."""
    model, state, drivers, _ = _air_gas(
        closures=[_gain], coupling="iterate", iterate_tol={"gas": 1.0}, iterate_max=3)
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, differentiable=False,
                       on_failure="return", max_iter=1)
    return record(out, diag)


# ---- a Kirchhoff layer with head-dependent coefficients (Anderson and flow tolerances)


class _DriverConductance(ConstitutiveLaw):
    def __init__(self, kind):
        super().__init__(kind)

    def flow(self, dp, drivers=None):
        return drivers["toy.k"] * dp

    def dflow(self, dp, drivers=None):
        return drivers["toy.k"].expand_as(dp)


class _HeadDependentConductance:
    def __init__(self, src, k0):
        self.src, self.k0 = src, k0

    def __call__(self, state, drivers):
        return {"toy.k": self.k0 * (1.0 + state["toy.phi"][..., self.src] / 10.0)}


def _toy(k0, **kw):
    net = Network(dtype=F64)
    for n in ("a", "b", "c"):
        net.add_node(n)
    net.add_edge("a", "b", kind="link")
    net.add_edge("b", "c", kind="link")
    layer = KirchhoffLayer(net, "toy", [_DriverConductance("link")], boundary=["c"])
    src, _ = net.endpoints("link")
    model = Model(net, {"toy": layer}, closures=[_HeadDependentConductance(src, k0)], **kw)
    state = {"toy.phi": torch.zeros(3, dtype=F64), "toy.q": torch.zeros(2, dtype=F64)}
    drivers = {"toy.phi_boundary": torch.zeros(1, dtype=F64),
               "toy.sources": torch.tensor([20.0, 0.0, 0.0], dtype=F64)}
    return model, state, drivers


def _toy_record(acceleration, *, flow_tol=True, sources=None, **kw):
    k0 = torch.tensor(1.0, dtype=F64, requires_grad=True)
    model, state, drivers = _toy(
        k0, coupling="iterate", iterate_tol={"toy": 1e-13}, iterate_max=400,
        iterate_acceleration=acceleration,
        iterate_flow_tol={"toy": 1e-12} if flow_tol else None, **kw)
    if sources is not None:
        drivers["toy.sources"] = torch.tensor(sources, dtype=F64)
        batch = drivers["toy.sources"].shape[:1]
        state = {k: v.expand(batch + v.shape).clone() for k, v in state.items()}
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, **TIGHT)
    return record(out, diag, grads_of(out["toy.phi"][..., 0].sum(), {"k0": k0}))


def case_anderson_flow_tol():
    return _toy_record("anderson")


def case_plain_flow_tol():
    return _toy_record(None)


def case_plain_no_flow_tol():
    return _toy_record(None, flow_tol=False)


def case_anderson_batch2():
    return _toy_record("anderson", sources=[[0.5, 0.0, 0.0], [40.0, 0.0, 0.0]])


def case_anderson_memory_one_accept_non_converged():
    k0 = torch.tensor(1.0, dtype=F64, requires_grad=True)
    model, state, drivers = _toy(
        k0, coupling="iterate", iterate_tol={"toy": 1e-13}, iterate_max=3,
        iterate_acceleration="anderson", iterate_memory=1, iterate_flow_tol={"toy": 1e-12},
        iterate_on_limit="accept")
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, **TIGHT)
    return record(out, diag, grads_of(out["toy.phi"][..., 0].sum(), {"k0": k0}))


def case_anderson_memory_one_accept_non_converged_no_grad():
    model, state, drivers = _toy(
        torch.tensor(1.0, dtype=F64), coupling="iterate", iterate_tol={"toy": 1e-13},
        iterate_max=3, iterate_acceleration="anderson", iterate_memory=1,
        iterate_flow_tol={"toy": 1e-12}, iterate_on_limit="accept")
    diag: dict = {}
    return record(model.steady(state, drivers, diagnostics=diag, **TIGHT), diag)


# ---- the sewer applications


def case_sewer_tree_steady_and_step():
    from noodl.apps.sewer.network import build_model, sewer_steady, tree_steady

    model, state, drivers = build_model(tree_steady())
    final = sewer_steady(model, state, drivers)
    diag: dict = {}
    stepped = model.step(state, drivers, 60.0, diagnostics=diag)
    return {"steady": record(final), "step": record(stepped, diag)}


def case_sewer_dynwave_tree_anderson():
    """The dynamic-wave tree: Anderson mixing of the heads with a flow tolerance."""
    from noodl.apps.sewer import read_swmm_inp
    from noodl.apps.sewer.dynwave import build_dynamic_wave_model, simulate_dynamic_wave

    data = Path(__file__).resolve().parent / "data" / "sewer"
    net, _, _ = read_swmm_inp(data / "tree_dynwave.inp")
    model, state, drivers = build_dynamic_wave_model(net)
    diags: list = []
    final, hist = simulate_dynamic_wave(model, state, drivers, steps=4, diagnostics=diags)
    return {"acceleration": model.iterate_acceleration, "final": record(final),
            "history": ser(hist), "diagnostics": ser(diags)}


# ---- integrating closure with closure-carried state


class _Integrator:
    """Integrates a carried state `acc.v` from the transport state and feeds the wind."""

    integrates = True
    state_keys = ("acc.v",)

    def __call__(self, state, drivers, ctx):
        x1 = state["gas.x"][..., 0]
        zero = torch.zeros_like(x1)
        dt = 0.0 if ctx is None or ctx.dt is None else ctx.dt
        v = state["acc.v"] + dt * x1
        return {"acc.v": v,
                "wind": drivers["wind"] + torch.stack([1e2 * v + 1e3 * x1, zero, zero], dim=-1)}


def case_integrating_closure_pingpong():
    model, state, drivers, params = _air_gas(closures=[_Integrator()])
    state = dict(state, **{"acc.v": torch.zeros((), dtype=F64)})
    diag: dict = {}
    out = state
    for _ in range(3):
        out = model.step(out, drivers, 600.0, t=0.0, diagnostics=diag)
    return record(out, diag, grads_of(out["gas.x"].sum() + out["acc.v"], params))


def case_integrating_closure_iterate():
    model, state, drivers, params = _air_gas(
        closures=[_Integrator()], coupling="iterate", iterate_tol={"gas": 1e-10},
        iterate_max=80)
    state = dict(state, **{"acc.v": torch.zeros((), dtype=F64)})
    diag: dict = {}
    out = state
    for _ in range(2):
        out = model.step(out, drivers, 600.0, diagnostics=diag)
    return record(out, diag, grads_of(out["gas.x"].sum() + out["acc.v"], params))


# ---- MNA layers (dynamic and phasor)


def _mna_net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


def _rc_model(**kw):
    from noodl.elements import Conductance
    from tests._circuit_laws import Cap, Source

    net = _mna_net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "r"),
                                       ("b", "gnd", "cap")])
    cap = Cap(torch.tensor([1.0], dtype=F64), learnable=True)
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([1.0], dtype=F64)),
                                      Conductance(torch.tensor([1.0], dtype=F64), kind="r"),
                                      cap], boundary=["gnd"])
    return net, layer, Model(net, [layer], **kw), cap


def case_mna_dynamic_steps():
    ground = torch.zeros(1, dtype=F64)
    _, layer, model, cap = _rc_model()
    drivers = {"e.phi_boundary": ground}
    phi, q = layer.solve(ground, drivers, mode="initial", differentiable=False)
    state = {"e.phi": phi, "e.q": q}
    diag: dict = {}
    for i in range(4):
        state = model.step(state, drivers, dt=0.25, t=0.25 * i, diagnostics=diag)
    state_be = model.step({"e.phi": phi, "e.q": q}, drivers, dt=0.25, t=0.0,
                          integration="backward_euler")
    return record(state, diag, grads_of(state["e.phi"].sum(), {"C": cap.C})) | {
        "backward_euler": record(state_be)}


def case_mna_dynamic_steady_dc():
    _, _, model, cap = _rc_model()
    diag: dict = {}
    out = model.steady({}, {"e.phi_boundary": torch.zeros(1, dtype=F64)}, t=0.0,
                       diagnostics=diag)
    return record(out, diag)


def case_mna_dynamic_iterate_feedback():
    net = _mna_net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "r"),
                                       ("b", "gnd", "cap")])
    b = net.nodes.index("b")

    def feedback(state, drivers):
        return {"e.sources": torch.zeros(3, dtype=F64).index_put(
            (torch.tensor(b),), 0.05 * state["e.phi"][..., b])}

    ground = torch.zeros(1, dtype=F64)
    from noodl.elements import Conductance
    from tests._circuit_laws import Cap, Source

    cap = Cap(torch.tensor([1.0], dtype=F64), learnable=True)
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([1.0], dtype=F64)),
                                      Conductance(torch.tensor([1.0], dtype=F64), kind="r"),
                                      cap], boundary=["gnd"])
    model = Model(net, [layer], closures=[feedback], coupling="iterate",
                  iterate_tol={"e": 1e-13}, iterate_max=60)
    drivers = {"e.phi_boundary": ground, "e.sources": torch.zeros(3, dtype=F64)}
    phi, q = layer.solve(ground, drivers, mode="initial", differentiable=False)
    state = {"e.phi": phi, "e.q": q}
    diag: dict = {}
    out = model.step(state, drivers, dt=0.25, t=0.0, diagnostics=diag)
    return record(out, diag, grads_of(out["e.phi"].sum(), {"C": cap.C}))


def _rlc_phasor_model(**kw):
    from noodl.elements import Conductance
    from tests._circuit_laws import Cap, DrivenSource, Ind

    net = _mna_net(["gnd", "a", "b", "c"], [("a", "gnd", "source"), ("a", "b", "r"),
                                            ("b", "c", "ind"), ("c", "gnd", "cap")])
    ind = Ind(torch.tensor([1e-3], dtype=F64), learnable=True)
    layer = KirchhoffLayer(
        net, "e", [DrivenSource(), Conductance(torch.tensor([0.5], dtype=F64), kind="r"), ind,
                   Cap(torch.tensor([1e-6], dtype=F64))],
        boundary=["gnd"], phasor=True)
    return net, layer, Model(net, [layer], **kw), ind


def case_mna_phasor_iterate():
    net, layer, _, _ = _rlc_phasor_model()
    c = net.nodes.index("c")

    def feedback(state, drivers):
        return {"V": (1.0 + 0.01 * state["e.phi"][..., c:c + 1]).to(torch.complex128)}

    ground = torch.zeros(1, dtype=F64)
    _, layer, model, ind = _rlc_phasor_model(
        closures=[feedback], coupling="iterate", iterate_tol={"e": 1e-12}, iterate_max=60)
    drivers = {"e.phi_boundary": ground, "e.frequency": torch.tensor(1000.0, dtype=F64),
               "V": torch.ones(1, dtype=torch.complex128)}
    phi0 = layer.solve(ground, drivers, differentiable=False)[0]
    diag: dict = {}
    out = model.steady({"e.phi": phi0}, drivers, diagnostics=diag, atol=1e-13)
    return record(out, diag, grads_of(out["e.phi"].abs().sum(), {"L": ind.L}))


def case_mna_phasor_pingpong():
    ground = torch.zeros(1, dtype=F64)
    _, layer, model, ind = _rlc_phasor_model()
    drivers = {"e.phi_boundary": ground, "e.frequency": torch.tensor(1000.0, dtype=F64),
               "V": torch.ones(1, dtype=torch.complex128)}
    phi0 = layer.solve(ground, drivers, differentiable=False)[0]
    diag: dict = {}
    out = model.steady({"e.phi": phi0}, drivers, diagnostics=diag, atol=1e-13)
    return record(out, diag, grads_of(out["e.phi"].abs().sum(), {"L": ind.L}))


# ------------------------------------------------------------------ cases: CoupledModel


def _street():
    net = Network(dtype=F64)
    for n in ("atm", "seg0", "seg1"):
        net.add_node(n)
    net.add_edge("seg0", "atm", kind="vent")
    net.add_edge("seg1", "atm", kind="vent")
    layer = TransportLayer(net, "street", capacity=torch.tensor([10.0, 10.0], dtype=F64),
                           flow_kind="vent", boundary=["atm"], quantity="concentration",
                           unit="kg/m3")

    def closure(state, drivers):
        return {"street.q": torch.tensor([1.0, 1.0], dtype=F64)}

    model = Model(net, {"street": layer}, closures=[closure])
    state = {"street.x": torch.tensor([3.0, 4.0], dtype=F64)}
    drivers = {"street.x_boundary": torch.zeros(1, dtype=F64),
               "street.sources": torch.zeros(3, dtype=F64)}
    return model, state, drivers


def _building():
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z0")
    net.add_edge("z0", "ambient", kind="airpath")
    layer = TransportLayer(net, "species", capacity=torch.tensor([5.0], dtype=F64),
                           flow_kind="airpath", boundary=["ambient"],
                           quantity="mass_fraction", unit="kg/kg")

    def closure(state, drivers):
        return {"species.q": torch.tensor([-0.5], dtype=F64)}

    model = Model(net, {"species": layer}, closures=[closure])
    state = {"species.x": torch.tensor([0.1], dtype=F64)}
    drivers = {"species.x_boundary": torch.zeros(1, dtype=F64),
               "rho_amb": torch.tensor(1.2, dtype=F64)}
    return model, state, drivers


def _link(two_way):
    return ValueLink(
        from_model="street", from_key="street.x", from_index=0,
        to_model="building", to_key="species.x_boundary", to_index=0,
        convert="concentration_to_mass_fraction", two_way=two_way)


def _coupled(two_way, *, grad=False, **kw):
    sm, ss, sd = _street()
    bm, bs, bd = _building()
    leaf = None
    if grad:
        leaf = torch.tensor([0.0, 0.0, 0.5], dtype=F64, requires_grad=True)
        sd = dict(sd, **{"street.sources": leaf})
    coupled, state, drivers = union(
        {"street": (sm, ss, sd), "building": (bm, bs, bd)}, [_link(two_way)], **kw)
    return coupled, state, drivers, leaf


def _coupled_record(coupled, state, drivers, leaf, *, dt=1.0, **kw):
    diag: dict = {}
    out = coupled.step(state, drivers, dt, diagnostics=diag, **kw)
    grads = None
    if leaf is not None:
        grads = grads_of(out["building"]["species.x"].sum() + out["street"]["street.x"].sum(),
                         {"sources": leaf})
    return {"state": ser(out), "state_order": {k: list(v) for k, v in out.items()},
            "diagnostics": ser(diag), "diagnostics_order": list(diag)} | (
        {"grads": ser(grads)} if grads is not None else {})


def case_coupled_one_way():
    return _coupled_record(*_coupled(False, grad=True))


def case_coupled_two_way():
    return _coupled_record(*_coupled(True, grad=True, iterate_rtol=1e-12, iterate_max=100))


def case_coupled_two_way_substeps_and_clock():
    return _coupled_record(
        *_coupled(True, grad=True, substeps={"building": 6}, iterate_max=100),
        dt=60.0, t=30.0)


def case_coupled_two_way_batch():
    sm, ss, sd = _street()
    bm, bs, bd = _building()
    ss = {"street.x": torch.tensor([[3.0, 4.0], [30.0, 4.0]], dtype=F64)}
    coupled, state, drivers = union(
        {"street": (sm, ss, sd), "building": (bm, bs, bd)}, [_link(True)],
        iterate_rtol=1e-12, iterate_max=100)
    return _coupled_record(coupled, state, drivers, None)


def case_coupled_two_way_batch_mixed_convergence():
    sm, ss, sd = _street()
    bm, bs, bd = _building()
    ss = {"street.x": torch.tensor([[3.0, 4.0], [30.0, 4.0]], dtype=F64)}
    coupled, state, drivers = union(
        {"street": (sm, ss, sd), "building": (bm, bs, bd)}, [_link(True)],
        iterate_rtol=0.0, iterate_atol=0.05, iterate_max=6)
    return _coupled_record(coupled, state, drivers, None)


def case_coupled_two_way_non_converged():
    return _coupled_record(*_coupled(True, iterate_rtol=0.0, iterate_atol=0.0, iterate_max=2))


def case_coupled_electrical_union_time_dependent_law():
    from noodl.apps.electrical import Ramp
    from noodl.apps.electrical.analysis import initial_state
    from noodl.apps.electrical.circuit import (
        Capacitor,
        Circuit,
        Resistor,
        VoltageSource,
        build_model,
    )

    cap = Capacitor("C1", "b", "0", 0.1)
    circuit = Circuit([VoltageSource("V1", "a", "0", 1.0, Ramp(1.0)),
                       Resistor("R1", "a", "b", 1.0), cap])
    model, _, drivers = build_model(circuit)
    state = initial_state(model, drivers, t=0.0)
    out = []
    for t in (0.0, 0.2, 0.7):
        coupled, cs, cd = union({"c": (model, state, drivers)}, [])
        diag: dict = {}
        res = coupled.step(cs, cd, 0.1, t=t, diagnostics=diag)
        out.append({"t": t, "state": ser(res), "diagnostics": ser(diag)})
    return out


CASES: dict[str, Callable[[], Any]] = {
    name[len("case_"):]: fn for name, fn in sorted(globals().items())
    if name.startswith("case_") and callable(fn)
}


def build_all() -> dict[str, Any]:
    """Every case's record (used by the regeneration script and by the test)."""
    return {name: outcome(fn) for name, fn in CASES.items()}


# ------------------------------------------------------------------ the test


@pytest.fixture(scope="module")
def golden():
    return load_golden("orchestration")


@pytest.mark.parametrize("name", sorted(CASES))
def test_orchestration_matches_the_golden_record(name, golden):
    assert name in golden, f"{name}: not in tests/golden/orchestration.json (regenerate it)"
    import json

    actual = json.loads(json.dumps(outcome(CASES[name])))
    diff = first_difference(actual, golden[name])
    assert diff is None, f"{name}: first difference at {diff}"


def test_the_golden_file_holds_no_case_the_suite_dropped(golden):
    extra = sorted(set(golden) - set(CASES) - {"solve_policy"})
    assert not extra, f"golden entries without a case: {extra}"


def test_the_golden_cases_cover_the_failure_and_the_success_paths(golden):
    kinds = {name: next(iter(rec)) for name, rec in golden.items() if name != "solve_policy"}
    assert "error" in kinds.values() and "ok" in kinds.values()

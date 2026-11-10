"""`KirchhoffLayer` cases of the orchestration golden (`tests/golden/orchestration.json`).

The Model and CoupledModel cases of `tests/test_golden_orchestration.py` reach the layer only
through a Model. These cases run one small layer directly and record what the Model cases do
not reach:

- gradients through relations (ideal sources, a fan) and through a multi-port law (a
  transformer);
- the phasor solve (an RLC branch and a transformer);
- the edge-head driver on a layer with flow unknowns;
- withdrawals, on both kinds of layer, and their order against storage;
- storage, including the held initial point (`mode="initial"`);
- context-reading and dynamic flow laws on a layer without flow unknowns;
- `flows`, `dflows`, `residual`, `jacobian`, `adjoint` and `linear_init` on a layer with flow
  unknowns;
- the refusals worth keeping.

A case records the potentials `phi`, the flows `q`, the solve diagnostics
(`newton_iterations`, `linear_iterations`, `backend`, `converged`) and, where it names
parameters, the gradients of a fixed loss with respect to them. A refusal that is the recorded
behaviour is caught inside the case (`_attempt`). Every case runs with float64 as the default
dtype.

The cases came from the M6 pins (`potential_layer.json`, folded in and deleted at the end of
M6); the values were carried over bit for bit.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from noodl.apps.electrical import (
    PrescribedCurrent,
    PrescribedVoltage,
    Resistance,
    Sine,
    TransformerRatio,
)
from noodl.apps.water.demand import PressureDrivenDemand
from noodl.elements import Conductance, ConstitutiveLaw, ImplicitLaw, PowerLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.storage import LinearStorage
from noodl.topology import Network
from noodl.withdrawals import NodeWithdrawal
from tests._circuit_laws import Cap, Ind, Source
from tests.golden import grads_of, ser

F64 = torch.float64
C128 = torch.complex128
DIAG_KEYS = ("newton_iterations", "linear_iterations", "backend", "converged")
GROUND = torch.zeros(1, dtype=F64)


def _t(values, grad=False, dtype=F64):
    return torch.tensor(values, dtype=dtype, requires_grad=grad)


def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


# ------------------------------------------------------------------ recording helpers


def _diag_record(diag: dict) -> dict:
    return {k: ser(diag[k]) for k in DIAG_KEYS if k in diag}


def _loss(phi: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """A fixed scalar of a solve's output: every potential and the squared flows."""
    return phi.sum() + (q * q).sum()


def _solve(layer, args: tuple, *, params: dict | None = None, **kw) -> dict:
    """`solve(*args, **kw)`: phi, q, diagnostics, gradients of `_loss` w.r.t. `params`."""
    diag: dict = {}
    phi, q = layer.solve(*args, diagnostics=diag, **kw)
    out: dict[str, Any] = {"phi": ser(phi), "q": ser(q), "diagnostics": _diag_record(diag)}
    if params:
        out["grads"] = ser(grads_of(_loss(phi, q), params))
    return out


def _attempt(fn: Callable[[], dict]) -> dict:
    """`fn()`, or the exception type and first 80 message characters: a refusal is recorded."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the refusal IS the recorded behaviour
        return {"refused": {"type": type(exc).__name__, "message": str(exc)[:80]}}


# ------------------------------------------------------------------ test-only laws


class _Emitter(NodeWithdrawal):
    """q_out = C sign(phi) |phi|^n, the orifice-type emitter EPANET also offers."""

    def __init__(self, nodes, coeff, exponent=0.5):
        super().__init__(nodes)
        self.coeff = torch.nn.Parameter(torch.as_tensor(coeff, dtype=F64))
        self.exponent = float(exponent)

    def flow(self, phi_nodes, drivers=None):
        safe = torch.clamp(phi_nodes.abs(), min=1e-12)
        return self.coeff * torch.sign(phi_nodes) * safe**self.exponent


class _DrivenDemand(NodeWithdrawal):
    """w = drivers["demand"] phi at the given nodes: a withdrawal that reads a driver."""

    def flow(self, phi_nodes, drivers=None):
        return drivers["demand"] * phi_nodes


class _Head(ImplicitLaw):
    """A fan with a prescribed rise: dp + head = 0, the flow not in the residual at all."""

    linear = True

    def __init__(self, kind="fan"):
        super().__init__(kind)

    def residual(self, dp, q, drivers, ctx):
        return dp + drivers["head"]


class _Orifice(ConstitutiveLaw):
    """q = c sign(dp) sqrt(|dp|) (smoothed at zero), c from the drivers."""

    def __init__(self, kind="ori"):
        super().__init__(kind)

    def flow(self, dp, drivers=None):
        return drivers["c"] * dp / torch.sqrt(dp * dp + 1e-6) ** 0.5

    def dflow(self, dp, drivers=None):
        return torch.autograd.functional.jacobian(
            lambda d: self.flow(d, drivers).sum(), dp, create_graph=True)


class _PotentialCap(ConstitutiveLaw):
    """A capacitance written as a flow law: it reads the step context and gives no relation.
    Steady states carry no flow; a step gives `C (dp - dp_prev) / dt` (backward Euler) or
    `2 C (dp - dp_prev) / dt - q_prev` (trapezoidal)."""

    dynamic = True

    def __init__(self, C, kind="pcap", learnable=False):
        super().__init__(kind)
        self.C = self._param(C, learnable)

    def flow(self, dp, drivers=None, ctx=None):
        if ctx.mode in ("dc", "initial"):
            return torch.zeros_like(dp)
        if ctx.scheme == "backward_euler":
            return self.C * (dp - ctx.dp_prev) / ctx.dt
        return 2.0 * self.C / ctx.dt * (dp - ctx.dp_prev) - ctx.q_prev


# ------------------------------------------------------------------ layers


#: w -(e0)- z1 -(e1)- z2 -(e2)- z3 -(e3)- l, plus z1 -(e4)- z3: five power-law edges.
_FAST_NODES = ["w", "z1", "z2", "z3", "l"]
_FAST_EDGES = [("w", "z1", "pipe"), ("z1", "z2", "pipe"), ("z2", "z3", "pipe"),
               ("z3", "l", "pipe"), ("z1", "z3", "pipe")]


def _fast(*, withdrawals=(), storage=()):
    """A power-law network without flow unknowns (the fast path)."""
    net = _net(_FAST_NODES, _FAST_EDGES)
    law = PowerLaw(_t([0.010, 0.008, 0.012, 0.009, 0.011]), 0.65, dp_transition=1e-6,
                   kind="pipe", learnable=True)
    layer = KirchhoffLayer(net, "e", [law], boundary=["w", "l"], withdrawals=withdrawals,
                           storage=storage)
    return layer, law


def _fast_inputs(grad=False):
    return _t([6.0, 0.5]).requires_grad_(grad), _t([0.0, 0.02, -0.01, 0.015, 0.0], grad)


def _emitters():
    return [_Emitter(torch.tensor([1, 3]), torch.tensor([0.02, 0.015], dtype=F64))]


def _electrical(edges, laws, nodes=("gnd", "a", "b"), **kw):
    net = _net(list(nodes), edges)
    return KirchhoffLayer(net, "e", laws, boundary=["gnd"], **kw)


def _fan(linear_solver="auto"):
    """gnd =fan=> a -ori-> b -ori-> gnd: the fan's flow is fixed by the orifice chain."""
    net = _net(["gnd", "a", "b"], [("gnd", "a", "fan"), ("a", "b", "ori"),
                                   ("b", "gnd", "ori")])
    return KirchhoffLayer(net, "air", [_Head(), _Orifice()], boundary=["gnd"],
                          linear_solver=linear_solver)


def _fan_drivers(batch=False):
    c = _t([[0.6], [0.9]] if batch else [0.7], grad=True)
    return {"head": _t([5.0]), "c": c}, c


def _rc_storage_layer(withdrawal=False):
    """gnd <-V- s -R- a, a storage at a (and a second one at s and a when `withdrawal`)."""
    net = _net(["gnd", "s", "a"], [("s", "gnd", "voltage_source"), ("s", "a", "resistor")])
    storage = LinearStorage([2], _t([0.3]), initial=1.5, learnable=True)
    res = Resistance(_t([2.0]), learnable=True)
    ws = []
    storages = [storage]
    if withdrawal:
        ws = [_Emitter(torch.tensor([1, 2]), torch.tensor([0.02, 0.03], dtype=F64))]
        storages = [storage, LinearStorage([1, 2], _t([0.05, 0.04]), learnable=True)]
    layer = KirchhoffLayer(net, "e", [PrescribedVoltage(_t([5.0])), res], boundary=["gnd"],
                           withdrawals=ws, storage=storages)
    return layer, storage, res, ws


def _transformer_net():
    return _net(["gnd", "p", "s"], [("p", "gnd", "source"), ("p", "gnd", "xfmr"),
                                    ("s", "gnd", "xfmr"), ("s", "gnd", "load")])


def _pda(nodes, q_required):
    n = len(nodes)
    return PressureDrivenDemand(torch.tensor(nodes), torch.tensor(q_required, dtype=F64),
                                torch.zeros(n, dtype=F64), p_min=0.0, p_req=3.0,
                                learnable=True)


# ------------------------------------------------------------------ gradients through relations


def potential_relation_source_gradients():
    source = PrescribedVoltage(_t([2.0]), learnable=True)
    resistor = Resistance(_t([4.0, 3.0]), learnable=True)
    layer = _electrical([("a", "gnd", "voltage_source"), ("a", "b", "resistor"),
                         ("b", "gnd", "resistor")], [source, resistor])
    return _solve(layer, (GROUND,), params={"V": source.amplitude, "R": resistor.R})


def potential_relation_fan_ensemble_gradients():
    layer = _fan()
    c = _t([[0.6], [0.7], [0.9]], grad=True)
    drivers = {"head": _t([5.0]), "c": c}
    return _solve(layer, (GROUND, drivers, torch.zeros(3, dtype=F64)), params={"c": c})


def potential_relation_cap_ind_march():
    """A series RLC from its held initial point, one backward-Euler step, one trapezoidal."""
    cap = Cap(_t([1e-3]), learnable=True)
    ind = Ind(_t([2e-2]), learnable=True)
    layer = _electrical(
        [("a", "gnd", "source"), ("a", "b", "r"), ("b", "c", "ind"), ("c", "gnd", "cap")],
        [Source(_t([1.5])), Conductance(_t([0.5]), kind="r"), ind, cap],
        nodes=("gnd", "a", "b", "c"))
    phi, q = layer.solve(GROUND, mode="initial", differentiable=False)
    phi = phi.detach().clone().requires_grad_()
    q = q.detach().clone().requires_grad_()
    history = (phi, q)
    params = {"C": cap.C, "L": ind.L, "phi_hist": phi, "q_hist": q}
    out: dict[str, Any] = {"initial": {"phi": ser(phi), "q": ser(q)}}
    loss = 0.0
    for i, scheme in enumerate(["backward_euler", "trapezoidal"]):
        diag: dict = {}
        now = layer.solve(GROUND, history=history, dt=0.01, scheme=scheme, diagnostics=diag)
        out[f"step{i}"] = {"phi": ser(now[0]), "q": ser(now[1]),
                           "diagnostics": _diag_record(diag)}
        loss = loss + _loss(*now)
        history = now
    out["grads"] = ser(grads_of(loss, params))
    return out


def potential_relation_initial_conditions_gradients():
    cap = Cap(_t([1e-3]), v0=_t([0.4]))
    ind = Ind(_t([2e-2]), i0=_t([0.1]))
    cap.v0.requires_grad_(True)
    ind.i0.requires_grad_(True)
    layer = _electrical(
        [("a", "gnd", "source"), ("a", "b", "r"), ("b", "c", "ind"), ("c", "gnd", "cap")],
        [Source(_t([1.5])), Conductance(_t([0.5]), kind="r"), ind, cap],
        nodes=("gnd", "a", "b", "c"))
    return _solve(layer, (GROUND,), params={"v0": cap.v0, "i0": ind.i0}, mode="initial")


def potential_multiport_transformer_gradients():
    ratio = TransformerRatio(_t([2.0]), kind="xfmr", learnable=True)
    load = Resistance(_t([4.0]), kind="load", learnable=True)
    layer = KirchhoffLayer(_transformer_net(), "e",
                           [PrescribedVoltage(_t([10.0]), kind="source"), ratio, load],
                           boundary=["gnd"])
    return _solve(layer, (GROUND,), params={"n": ratio.n, "R": load.R})


# ------------------------------------------------------------------ phasor


def potential_phasor_rlc_gradients():
    cap = Cap(_t([1e-6]), learnable=True)
    ind = Ind(_t([1e-3]), learnable=True)
    res = Resistance(_t([2.0]), kind="r", learnable=True)
    layer = _electrical(
        [("a", "gnd", "source"), ("a", "b", "r"), ("b", "c", "ind"), ("c", "gnd", "cap")],
        [Source(_t([1.0])), res, ind, cap], nodes=("gnd", "a", "b", "c"), phasor=True)
    f = _t([50.0, 5032.92])
    diag: dict = {}
    phi, q = layer.solve(GROUND.to(C128), {"e.frequency": f}, diagnostics=diag)
    loss = phi.abs().sum() + q.abs().sum()
    return {"phi": ser(phi), "q": ser(q), "diagnostics": _diag_record(diag),
            "grads": ser(grads_of(loss, {"R": res.R, "L": ind.L, "C": cap.C}))}


def potential_phasor_transformer():
    layer = KirchhoffLayer(
        _transformer_net(), "e",
        [Source(_t([10.0])), TransformerRatio(_t([2.0]), kind="xfmr"),
         Resistance(_t([4.0]), kind="load")], boundary=["gnd"], phasor=True)
    return _solve(layer, (GROUND.to(C128), {"e.frequency": _t([50.0, 60.0])}),
                  differentiable=False)


# ------------------------------------------------------------------ the edge-head driver


def potential_edge_head_with_flow_unknowns():
    """gnd <-V- a -(r)- b -(load)- gnd with a head on each edge; a batch of two heads."""
    net = _net(["gnd", "a", "b"], [("a", "gnd", "voltage_source"), ("a", "b", "r"),
                                   ("b", "gnd", "load")])
    law = PowerLaw(_t([0.5]), _t(0.65), kind="r", dp_transition=1e-6, learnable=True)
    layer = KirchhoffLayer(
        net, "e", [PrescribedVoltage(_t([1.0])), law,
                   PowerLaw(_t([0.3]), _t(0.8), kind="load", dp_transition=1e-6)],
        boundary=["gnd"])
    head = _t([[0.0, 0.4, -0.2], [0.0, 0.1, 0.3]], grad=True)
    return _solve(layer, (GROUND, {"e.edge_head": head}, _t([0.0, 0.02, -0.01])),
                  params={"C": law.C, "head": head})


# ------------------------------------------------------------------ withdrawals


def potential_withdrawal_fast():
    ws = _emitters()
    layer, law = _fast(withdrawals=ws)
    pb, src = _fast_inputs(grad=True)
    return _solve(layer, (pb, {}, src),
                  params={"C": law.C, "coeff": ws[0].coeff, "sources": src})


def potential_withdrawal_fast_pressure_driven_demand():
    ws = [_pda([1, 3], [0.05, 0.04])]
    layer, law = _fast(withdrawals=ws)
    pb, src = _fast_inputs(grad=True)
    return _solve(layer, (pb, {}, src), params={"C": law.C, "q_required": ws[0].q_required})


def potential_withdrawal_driven_with_flow_unknowns():
    net = _net(["gnd", "s", "a"], [("s", "gnd", "voltage_source"), ("s", "a", "resistor"),
                                   ("a", "gnd", "resistor")])
    layer = KirchhoffLayer(
        net, "e", [PrescribedVoltage(_t([5.0])), Resistance(_t([2.0, 3.0]), learnable=True)],
        boundary=["gnd"], withdrawals=[_DrivenDemand(torch.tensor([1, 2]))])
    demand = _t([0.1, 0.2], grad=True)
    return _solve(layer, (GROUND, {"demand": demand}), params={"demand": demand})


def potential_withdrawal_and_storage_with_flow_unknowns():
    """A withdrawal and two storages on the same nodes: records the order the two are summed."""
    layer, storage, res, ws = _rc_storage_layer(withdrawal=True)
    phi0, q0 = layer.solve(GROUND, mode="initial", differentiable=False)
    w0 = layer.storage_flows(phi0, {}, dt=None, mode="initial", q=q0)
    flow = w0.detach().clone().requires_grad_()
    history = (phi0.detach().clone(), q0.detach().clone())
    kw = {"dt": 0.2, "scheme": "trapezoidal", "history": history, "storage_flow": flow}
    diag: dict = {}
    phi, q = layer.solve(GROUND, diagnostics=diag, **kw)
    w = layer.storage_flows(phi, {}, **kw)
    params = {"C": storage.C, "R": res.R, "coeff": ws[0].coeff, "storage_flow": flow}
    return {"initial": {"phi": ser(phi0), "q": ser(q0), "storage_flows": ser(w0)},
            "step": {"phi": ser(phi), "q": ser(q), "storage_flows": ser(w),
                     "diagnostics": _diag_record(diag)},
            "balance": ser(layer.balance(phi, q, None, {}, storage_flow=w)),
            "balance_steady": ser(layer.balance(phi, q, None, {})),
            "grads": ser(grads_of(_loss(phi, q) + (w * w).sum(), params))}


# ------------------------------------------------------------------ storage


def potential_storage_fast_trapezoidal_with_withdrawal():
    storage = LinearStorage([1, 2], _t([0.06, 0.03]), learnable=True)
    layer, law = _fast(storage=[storage], withdrawals=_emitters())
    pb, src = _fast_inputs()
    phi_prev = _t([6.0, 0.4, -0.2, 0.3, 0.5], grad=True)
    flow = _t([0.05, -0.02], grad=True)
    kw: dict[str, Any] = {"dt": 0.3, "scheme": "trapezoidal",
                          "history": (phi_prev, torch.zeros(5, dtype=F64)),
                          "storage_flow": flow}
    params = {"C": storage.C, "phi_prev": phi_prev, "law_C": law.C, "storage_flow": flow}
    diag: dict = {}
    phi, q = layer.solve(pb, {}, src, diagnostics=diag, **kw)
    w = layer.storage_flows(phi, {}, **kw)
    out = {"phi": ser(phi), "q": ser(q), "storage_flows": ser(w),
           "diagnostics": _diag_record(diag)}
    out["grads"] = ser(grads_of(_loss(phi, q) + (w * w).sum(), params))
    return out


def potential_storage_fast_held_initial():
    """The held initial point on a layer without flow unknowns (solved with the held rows)."""
    storage = LinearStorage([1, 2], _t([0.06, 0.03]), initial=_t([0.4, -0.2]), learnable=True)
    layer, _ = _fast(storage=[storage])
    pb, src = _fast_inputs()
    return _attempt(lambda: _solve(layer, (pb, {}, src), mode="initial",
                                   params={"C": storage.C}))


def potential_storage_held_initial_then_trapezoidal():
    layer, storage, res, _ = _rc_storage_layer()
    phi0, q0 = layer.solve(GROUND, mode="initial", differentiable=False)
    w0 = layer.storage_flows(phi0, {}, dt=None, mode="initial", q=q0)
    history = (phi0.detach().clone().requires_grad_(), q0.detach().clone().requires_grad_())
    flow = w0.detach().clone().requires_grad_()
    kw = {"dt": 0.2, "scheme": "trapezoidal", "history": history, "storage_flow": flow}
    diag: dict = {}
    phi, q = layer.solve(GROUND, diagnostics=diag, **kw)
    w = layer.storage_flows(phi, {}, **kw)
    params = {"C": storage.C, "R": res.R, "phi_hist": history[0], "storage_flow": flow}
    return {"initial": {"phi": ser(phi0), "q": ser(q0), "storage_flows": ser(w0)},
            "step": {"phi": ser(phi), "q": ser(q), "storage_flows": ser(w),
                     "diagnostics": _diag_record(diag)},
            "grads": ser(grads_of(_loss(phi, q) + (w * w).sum(), params))}


# ------------------------------------------------------------------ context and dynamic laws


def _current_layer(*extra_edges_and_laws, storage=None):
    """gnd <-I- a -R- gnd, a sine current source (a law that reads the time)."""
    source = PrescribedCurrent(_t([1.0]), waveforms=[Sine(2.0, phase=0.3)], learnable=True)
    edges = [("a", "gnd", "current_source"), ("a", "gnd", "resistor")]
    laws = [source, Resistance(_t([2.0]))]
    for edge, law in extra_edges_and_laws:
        edges.append(edge)
        laws.append(law)
    layer = KirchhoffLayer(_net(["gnd", "a"], edges), "e", laws, boundary=["gnd"],
                           storage=[storage] if storage else ())
    return layer, source


def potential_context_law_with_storage():
    """A time-reading source and a grounded storage: held initial point, two steps."""
    storage = LinearStorage([1], _t([0.4]), initial=0.0, learnable=True)
    layer, source = _current_layer(storage=storage)
    phi, q = layer.solve(GROUND, mode="initial", t=0.0, differentiable=False)
    w = layer.storage_flows(phi, {}, dt=None, mode="initial", q=q)
    out: dict[str, Any] = {"initial": {"phi": ser(phi), "q": ser(q), "storage_flows": ser(w)}}
    history = (phi.detach().clone(), q.detach().clone())
    loss = 0.0
    for i, t in enumerate((0.3, 0.6)):
        diag: dict = {}
        kw = {"dt": 0.3, "scheme": "trapezoidal", "history": history, "storage_flow": w,
              "t": t}
        phi, q = layer.solve(GROUND, diagnostics=diag, **kw)
        w = layer.storage_flows(phi, {}, **{k: v for k, v in kw.items() if k != "t"})
        out[f"step{i}"] = {"phi": ser(phi), "q": ser(q), "storage_flows": ser(w),
                           "diagnostics": _diag_record(diag)}
        loss = loss + _loss(phi, q)
        history = (phi, q)
    out["grads"] = ser(grads_of(loss, {"amplitude": source.amplitude, "C": storage.C}))
    return out


def potential_dynamic_flow_law():
    """A dynamic, context-reading flow law and no relation: a steady state, then one
    backward-Euler and one trapezoidal step."""
    pcap = _PotentialCap(_t([0.4]), learnable=True)
    layer, source = _current_layer((("a", "gnd", "pcap"), pcap))

    def run():
        phi, q = layer.solve(GROUND, t=0.0, differentiable=False)
        out: dict[str, Any] = {"initial": {"phi": ser(phi), "q": ser(q)}}
        history = (phi.detach().clone(), q.detach().clone())
        loss = 0.0
        for i, (scheme, t) in enumerate((("backward_euler", 0.2), ("trapezoidal", 0.4))):
            diag: dict = {}
            now = layer.solve(GROUND, history=history, dt=0.2, scheme=scheme, t=t,
                              diagnostics=diag)
            out[f"step{i}"] = {"phi": ser(now[0]), "q": ser(now[1]),
                               "diagnostics": _diag_record(diag)}
            loss = loss + _loss(*now)
            history = now
        out["grads"] = ser(grads_of(loss, {"amplitude": source.amplitude, "C": pcap.C}))
        return out

    return _attempt(run)


# ------------------------------------------------------------------ layer methods, refusals


def potential_layer_methods_with_flow_unknowns():
    """What a layer with flow unknowns answers about a solved point: its balance, linear
    guess, flows, slopes, residual, Jacobian and adjoint, and what it refuses without the
    flows `q`."""
    layer = _fan()
    drivers, _ = _fan_drivers()
    src = _t([0.0, 0.05, -0.02])
    phi, q = layer.solve(GROUND, drivers, src, differentiable=False)
    phi_i = phi[..., layer.interior]
    grad = _t([1.0, -0.5, 0.25])

    def probe(fn):
        return _attempt(lambda: {"value": ser(fn())})

    return {
        "power_residual": probe(lambda: layer.power_residual(phi, q, drivers, src)),
        "residual": probe(lambda: layer.residual(phi_i, GROUND, drivers, src, q=q)),
        "jacobian": probe(lambda: layer.jacobian(phi_i, GROUND, drivers, q=q)),
        "adjoint": probe(lambda: layer.adjoint(phi_i, GROUND, drivers, grad, q=q)),
        "linear_init": probe(lambda: layer.linear_init(GROUND, drivers, src)),
        "flows": probe(lambda: layer.flows(phi, drivers, q)),
        "flows_without_q": probe(lambda: layer.flows(phi, drivers)),
        # the fan is an ideal source: its implied slope is nan, recorded as a flag so that
        # the records stay finite
        "dflows_is_nan": probe(lambda: torch.isnan(layer.dflows(phi, drivers, q)).to(F64)),
        "dflows": probe(lambda: layer.dflows(phi, drivers, q)[..., 1:]),
    }


def potential_refuses_cg_with_flow_unknowns():
    drivers, c = _fan_drivers(batch=True)
    return _attempt(lambda: _solve(_fan("cg"), (GROUND, drivers, torch.zeros(3, dtype=F64)),
                                   params={"c": c}))


def potential_refuses_unconverged_with_flow_unknowns():
    drivers, _ = _fan_drivers()
    return _attempt(lambda: _solve(_fan(), (GROUND, drivers, torch.zeros(3, dtype=F64)),
                                   differentiable=False, on_failure="raise", max_iter=1))


# ------------------------------------------------------------------ the registry


def _float64(fn: Callable[[], dict]) -> Callable[[], dict]:
    """`fn` with float64 as the default dtype for its duration."""

    @functools.wraps(fn)
    def run() -> dict:
        previous = torch.get_default_dtype()
        torch.set_default_dtype(F64)
        try:
            return fn()
        finally:
            torch.set_default_dtype(previous)

    return run


CASES: dict[str, Callable[[], dict]] = {
    name: _float64(fn) for name, fn in sorted(globals().items())
    if name.startswith("potential_") and callable(fn)
}

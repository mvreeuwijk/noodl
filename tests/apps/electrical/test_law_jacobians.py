"""Every electrical law's analytic `jacobian` equals the colouring of its own `forward`.

Each case is a constructed law and the edge values, drivers and contexts it is checked in:
every mode the law supports (`dc`, `initial`, a transient step under both schemes, `phasor`),
batched inputs, and batched drivers where the law reads any. The blocks are checked against
`_colour_blocks` (to 1e-12 relative), against central differences, and `gradcheck` confirms
the law's autograd. The case list must cover every law defined in `noodl.apps.electrical`
(found by walking the package), and each of them must override `jacobian`.
"""

from __future__ import annotations

import importlib
import inspect
import math
import pkgutil
from dataclasses import dataclass, field

import pytest
import torch

import noodl.apps.electrical
from noodl.apps.electrical import (
    Capacitance,
    ConstantPowerLoad,
    DependentSource,
    Impedance,
    ImpedanceTransformer,
    Inductance,
    JunctionDiode,
    MutualInductance,
    PowerConverter,
    PrescribedCurrent,
    PrescribedVoltage,
    Resistance,
    Sine,
    SwitchedConductance,
    TransformerRatio,
)
from noodl.apps.electrical.power import PiBranch, Shunt, VoltageControlledGenerator
from noodl.elements import ConstitutiveLaw, LawContext
from noodl.elements._slopes import _colour_blocks
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64
C128 = torch.complex128
B = 3   # batch size of every case



def _t(values):
    return torch.tensor(values, dtype=F64)


def _rand(*shape, lo=-1.0, hi=1.0, seed=0, complex_=False):
    g = torch.Generator().manual_seed(seed)
    x = lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=F64)
    if complex_:
        y = lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=F64)
        return torch.complex(x, y)
    return x


def _contexts(E, *, real=True, transient=True, phasor=True, initial=True):
    """The contexts a law is checked in, by name."""
    out = {}
    if real:
        out["dc"] = LawContext(mode="dc", t=0.0)
        if initial:
            out["initial"] = LawContext(mode="initial", t=0.0)
        if transient:
            for scheme in ("backward_euler", "trapezoidal"):
                out[scheme] = LawContext(
                    mode="transient", dt=0.013, t=0.21, scheme=scheme,
                    dp_prev=_rand(B, E, seed=11), q_prev=_rand(B, E, seed=12))
    if phasor:
        out["phasor"] = LawContext(
            mode="phasor", t=0.0,
            omega=2 * math.pi * torch.tensor([[50.0], [60.0], [400.0]], dtype=F64))
    return out


@dataclass
class Case:
    name: str
    law: ConstitutiveLaw
    E: int
    contexts: dict
    drivers: dict = field(default_factory=dict)
    dp_range: tuple = (-1.0, 1.0)
    q_range: tuple = (-1.0, 1.0)


def _cases() -> list[Case]:
    E = 4
    T = 280.0 + 40.0 * _rand(B, E, lo=0.0, hi=1.0, seed=3)   # batched temperatures
    two = _contexts(E)
    cases = [
        Case("voltage_source", PrescribedVoltage(_t([1.0, 2.0, -1.0, 0.5])), E,
             _contexts(E)),
        Case("voltage_source_sine",
             PrescribedVoltage(2.0, waveforms=Sine(50.0, phase=0.3, start_time=0.01)), E,
             _contexts(E)),
        Case("current_source", PrescribedCurrent(_t([1.0, 2.0, -1.0, 0.5])), E,
             _contexts(E)),
        Case("resistor", Resistance(_t([1.0, 2.0, 5.0, 0.1])), E, _contexts(E)),
        Case("resistor_driven",
             Resistance(_t([1.0, 2.0, 5.0, 0.1]), alpha=0.004, temperature="T",
                        scale="s"), E, _contexts(E, phasor=False),
             drivers={"T": T, "s": 1.0 + _rand(B, E, lo=0.0, hi=1.0, seed=4)}),
        Case("capacitor", Capacitance(_t([1e-3, 2e-3, 0.5, 1.0]), v0=0.3), E,
             _contexts(E)),
        Case("inductor", Inductance(_t([1e-3, 2e-3, 0.5, 1.0]), i0=0.2), E,
             _contexts(E)),
        Case("diode", JunctionDiode(), E, _contexts(E, phasor=False), dp_range=(-1.0, 0.55)),
        Case("diode_above_maxexp", JunctionDiode(), E, _contexts(E, phasor=False),
             dp_range=(0.62, 1.0)),
        Case("diode_temperature", JunctionDiode(temperature="T"), E,
             _contexts(E, phasor=False), drivers={"T": T}, dp_range=(-1.0, 0.3)),
        Case("switch", SwitchedConductance("c", Ron=1e-2, Goff=1e-3), E, _contexts(E),
             drivers={"c": _t([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0],
                            [1.0, 1.0, 0.0, 0.0]])}),
        Case("ideal_transformer", TransformerRatio(_t([2.0, 0.5])), E, two),
        Case("mutual_inductance",
             MutualInductance(_t([1.0, 2.0]), _t([0.5, 3.0]),
                              _t([0.3, 1.1]), i0=(0.1, -0.2)), E, two),
    ]
    for control in ("voltage", "current"):
        for output in ("voltage", "current"):
            cases.append(Case(f"dependent_{control}_{output}",
                              DependentSource(_t([2.5, -0.7]), control=control,
                                              output=output), E, two))
    phasor_only = _contexts(E, real=False)
    cases += [
        Case("impedance", Impedance(_t([1.0, 2.0, 0.0, 0.5])), E,
             _contexts(E)),
        Case("impedance_reactive",
             Impedance(_t([1.0, 2.0, 0.0, 0.5]),
                       L=_t([1e-3, 0.0, 2e-3, 0.0]),
                       C=_t([1e-4, math.inf, math.inf, 2e-3])), E, phasor_only),
        Case("load", ConstantPowerLoad(_t([1.0, 2.0, 0.5, 3.0]),
                                       _t([0.0, 0.0, 0.0, 0.0])), E,
             _contexts(E), dp_range=(0.5, 1.5)),
        Case("load_reactive", ConstantPowerLoad(_t([1.0, 2.0, 0.5, 3.0]),
                                                _t([0.3, -0.2, 0.0, 1.0])), E,
             phasor_only, dp_range=(0.5, 1.5)),
        Case("load_nominal",
             ConstantPowerLoad(_t([1.0, 2.0, 0.5, 3.0]),
                               _t([0.3, -0.2, 0.0, 1.0]), V_nominal=1.0,
                               scale="p"), E, phasor_only,
             drivers={"p": 0.5 + _rand(B, E, lo=0.0, hi=1.0, seed=5)}, dp_range=(-1.0, 1.0)),
        Case("load_nominal_small",
             ConstantPowerLoad(_t([1.0, 2.0, 0.5, 3.0]),
                               _t([0.3, -0.2, 0.0, 1.0]), V_nominal=1000.0), E,
             phasor_only, dp_range=(-0.5, 0.5)),
        Case("load_real_nominal",
             ConstantPowerLoad(_t([1.0, 2.0, 0.5, 3.0]), V_nominal=1.0), E,
             _contexts(E, phasor=False), dp_range=(0.2, 1.5)),
        Case("load_dc_nominal",
             ConstantPowerLoad(_t([1.0, 2.0, 0.5, 3.0]), V_nominal=1000.0,
                               direct_current=True), E, _contexts(E),
             dp_range=(-1.0, 1.0)),
        Case("load_linearized_ac",
             ConstantPowerLoad(_t([1.0, 2.0, 0.5, 3.0]),
                               _t([0.3, -0.2, 0.0, 1.0]), V_nominal=1.0,
                               linearized=True), E, phasor_only),
        Case("load_linearized_dc",
             ConstantPowerLoad(_t([1.0, 2.0, 0.5, 3.0]), V_nominal=1.0,
                               linearized=True, direct_current=True), E, _contexts(E),
             dp_range=(0.0, 1.5)),
        Case("impedance_transformer",
             ImpedanceTransformer(_t([2.0, 0.5]),
                                  torch.tensor([0.1 + 0.3j, 0.2 + 0.05j], dtype=C128)), E,
             phasor_only),
        Case("power_converter", PowerConverter(_t([1.2, 0.8]),
                                               _t([0.9, 0.95])), E, phasor_only,
             q_range=(-0.3, 0.3)),
        Case("pi_branch",
             PiBranch(_t([0.01, 0.02]), _t([0.1, 0.05]),
                      _t([0.02, 0.0]), _t([0.0, 0.98]),
                      _t([0.0, 0.1])), E, phasor_only),
        Case("shunt", Shunt(_t([0.1, 0.0, 0.2, 0.05]),
                            _t([0.3, -0.2, 0.0, 1.0])), E, phasor_only),
        Case("generator", VoltageControlledGenerator(_t([0.5, 1.0, -0.3, 0.0]),
                                                     _t([1.0, 1.02, 0.98, 1.0])),
             E, phasor_only, dp_range=(0.5, 1.5)),
    ]
    return cases


CASES = _cases()
PARAMS = [pytest.param(c, name, id=f"{c.name}-{name}")
          for c in CASES for name in c.contexts]


def _values(case: Case, ctx: LawContext):
    phasor = ctx.mode == "phasor"
    dp = _rand(B, case.E, lo=case.dp_range[0], hi=case.dp_range[1], seed=1, complex_=phasor)
    q = _rand(B, case.E, lo=case.q_range[0], hi=case.q_range[1], seed=2, complex_=phasor)
    return dp, (q if case.law.law_form == "implicit" else None)


def _colour(law, dp, q, drivers, ctx):
    w = 2 if dp.is_complex() else 1
    if q is None:
        return _colour_blocks(lambda d: law(d, drivers, ctx), (dp,), law.ports, w)
    return _colour_blocks(lambda d, qq: law(d, qq, drivers, ctx), (dp, q), law.ports, w)


def _public_laws():
    """Every public `ConstitutiveLaw` subclass defined anywhere in `noodl.apps.electrical`."""
    pkg = noodl.apps.electrical
    found = set()
    for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if (inspect.isclass(obj) and issubclass(obj, ConstitutiveLaw)
                    and obj.__module__ == module.__name__ and not name.startswith("_")):
                found.add(obj)
    return found


def test_cases_cover_every_public_electrical_law():
    covered = {type(c.law) for c in CASES}
    missing = _public_laws() - covered
    assert not missing, f"no Jacobian case for {sorted(c.__name__ for c in missing)}"


@pytest.mark.parametrize("cls", sorted(_public_laws(), key=lambda c: c.__name__),
                         ids=lambda c: c.__name__)
def test_every_public_electrical_law_has_an_analytic_jacobian(cls):
    assert cls.jacobian is not ConstitutiveLaw.jacobian, (
        f"{cls.__name__} has no analytic jacobian")


def test_every_mode_is_checked_for_every_dynamic_and_phasor_law():
    for c in CASES:
        names = set(c.contexts)
        if c.law.dynamic:
            assert {"dc", "initial", "backward_euler", "trapezoidal"} <= names, c.name
        if c.law.phasor_capable:
            assert any("phasor" in n for case in CASES if type(case.law) is type(c.law)
                       for n in case.contexts), c.name


@pytest.mark.parametrize(("case", "mode"), PARAMS)
def test_analytic_blocks_equal_colouring(case, mode):
    ctx = case.contexts[mode]
    dp, q = _values(case, ctx)
    analytic = case.law.jacobian(dp, q, case.drivers, ctx)
    ref = _colour(case.law, dp, q, case.drivers, ctx)
    assert len(analytic) == len(ref)
    for a, r in zip(analytic, ref, strict=True):
        assert a.dtype == r.dtype
        assert a.shape[-5:] == r.shape[-5:]
        a, r = torch.broadcast_tensors(a, r)   # r may be wider: its batch includes dp_prev
        scale = float(r.abs().max()) or 1.0
        torch.testing.assert_close(a, r, rtol=1e-12, atol=1e-12 * scale)


def _real_view(x):
    return torch.view_as_real(x.resolve_conj()) if x.is_complex() else x.unsqueeze(-1)


def _central_blocks(law, dp, q, drivers, ctx, which, eps=1e-6):
    """Central differences of the law's output in groups of `ports`: perturbing `slot`
    (component `cl`) of every group at once gives column `(slot, cl)` of every block."""
    m = law.ports
    w = 2 if dp.is_complex() else 1
    x = (dp, q)[which]

    def out(xx):
        args = (xx, q) if which == 0 else (dp, xx)
        y = law(args[0], args[1], drivers, ctx) if q is not None else law(xx, drivers, ctx)
        return _real_view(y.to(C128) if dp.is_complex() else y)

    cols = []
    for slot in range(m):
        for cl in range(w):
            step = torch.zeros(x.shape[-1] // m, m, w, dtype=F64)
            step[:, slot, cl] = eps
            step = step.flatten(-3)
            step = torch.view_as_complex(step.unflatten(-1, (-1, 2))) if w == 2 else step
            diff = (out(x + step) - out(x - step)) / (2 * eps)       # (..., E, w)
            cols.append(diff.unflatten(-2, (-1, m)))                 # (..., G, m, w)
    # cols[slot * w + cl][..., g, k, ck] -> (..., G, k, l, ck, cl)
    stacked = torch.stack(cols, dim=-1).unflatten(-1, (m, w))       # (..., G, k, ck, l, cl)
    return stacked.permute(*range(stacked.dim() - 4), -4, -2, -3, -1)


@pytest.mark.parametrize(("case", "mode"), PARAMS)
def test_analytic_blocks_equal_central_differences(case, mode):
    ctx = case.contexts[mode]
    dp, q = _values(case, ctx)
    analytic = case.law.jacobian(dp, q, case.drivers, ctx)
    for which, a in enumerate(analytic):
        fd = _central_blocks(case.law, dp, q, case.drivers, ctx, which)
        a, fd = torch.broadcast_tensors(a, fd)
        scale = float(fd.abs().max()) or 1.0
        torch.testing.assert_close(a, fd, rtol=1e-6, atol=1e-7 * scale)


@pytest.mark.parametrize(("case", "mode"), PARAMS)
def test_law_autograd_passes_gradcheck(case, mode):
    ctx = case.contexts[mode]
    dp, q = _values(case, ctx)
    dp = dp.clone().requires_grad_(True)
    if q is None:
        assert torch.autograd.gradcheck(lambda d: case.law(d, case.drivers, ctx), (dp,))
    else:
        q = q.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(lambda d, qq: case.law(d, qq, case.drivers, ctx),
                                        (dp, q))


@pytest.mark.parametrize("dp", [0.55, 0.65], ids=["below_maxexp", "above_maxexp"])
def test_diode_slope_on_each_side_of_the_exlin_switch(dp):
    law = JunctionDiode(*(_t(v) for v in (1e-6, 0.04, 15.0, 1e8)))   # switch at dp = 0.6
    x = torch.tensor([dp], dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda d: law(d, {}, None), (x,))
    block, = law.jacobian(x.detach(), None, {}, None)
    (grad,) = torch.autograd.grad(law(x, {}, None).sum(), x)
    torch.testing.assert_close(block.reshape(1), grad, rtol=1e-14, atol=0)
    expected = 1e-6 * math.exp(min(dp / 0.04, 15.0)) / 0.04 + 1.0 / 1e8
    assert float(block) == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------------------ gradients through solves
def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


def test_gradient_through_transient_steps_of_a_small_circuit():
    # source a -> 0, resistor a -> b, capacitor b -> 0, inductor b -> 0, both schemes
    net = _net(["0", "a", "b"], [("a", "0", "voltage_source"), ("a", "b", "resistor"),
                                 ("b", "0", "capacitor"), ("b", "0", "inductor")])
    R = Resistance(_t([2.0]), learnable=True)
    C = Capacitance(_t([0.4]), learnable=True)
    L = Inductance(_t([0.9]), learnable=True)
    V = PrescribedVoltage(_t([1.0]), learnable=True)
    layer = KirchhoffLayer(net, "e", [V, R, C, L], boundary=["0"])
    b = net.nodes.index("b")
    ground = torch.zeros(1, dtype=F64)

    for scheme in ("backward_euler", "trapezoidal"):
        def v_after_steps(Vamp, Rv, Cv, Lv, scheme=scheme):
            phi, q = layer.solve(ground, {}, mode="initial", differentiable=True)
            for k in range(3):
                phi, q = layer.solve(ground, {}, history=(phi, q), dt=0.1, t=0.1 * (k + 1),
                                     scheme=scheme, differentiable=True)
            return phi[..., b]

        assert torch.autograd.gradcheck(v_after_steps, (V.amplitude, R.R, C.C, L.L))


def test_gradient_through_a_phasor_solve_of_a_small_circuit():
    net = _net(["0", "a", "b"], [("a", "0", "voltage_source"), ("a", "b", "impedance"),
                                 ("b", "0", "load")])
    Z = Impedance(_t([0.05]), L=_t([1e-4]), learnable=True)
    P = ConstantPowerLoad(_t([0.4]), _t([0.1]), learnable=True)
    layer = KirchhoffLayer(net, "e", [PrescribedVoltage(_t([1.0])), Z, P],
                           boundary=["0"], phasor=True)
    b = net.nodes.index("b")
    drivers = {"e.frequency": _t(50.0)}
    ground = torch.zeros(1, dtype=C128)
    phi0 = torch.ones(len(layer.interior), dtype=C128)

    def magnitude(Rv, Pv, Qv):
        phi, _ = layer.solve(ground, drivers, phi0=phi0, differentiable=True)
        return phi[..., b].abs()

    assert torch.autograd.gradcheck(magnitude, (Z.R, P.P, P.Q))

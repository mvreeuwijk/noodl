"""Every potential-form law's and node source's `dflow` against the autograd derivative of its
own `flow` (the audit behind `exact_slope`).

The differentiable classic solve uses `dflow` for the Newton Jacobian and the adjoint of a law
that declares `exact_slope = True` (the default), and autograd through `flow` for one that does
not (`noodl.layers._functional.FunctionalLaws.dflows`). The adjoint needs the exact derivative:
a `dflow` that is a Newton aid (a slope floor, a closed-pump slope) would silently corrupt the
gradients. So every law that keeps the default must have a `dflow` equal to the autograd
derivative of its `flow`, to 1e-10 relative, at sampled points on every smooth piece and inside
its regularisation bands; and a law that declares `exact_slope = False` must really differ
somewhere, so the declaration is not stale.

The classes are found by walking the `noodl` package; each one with a potential form must
have an entry in `CASES` (a constructor of instances, drivers and sample points) or in
`EXEMPT` (with the reason), so a new law cannot skip the audit. Every `NodeStorage` is audited
the same way (`rate_slope` against the autograd derivative of `rate`, `STORAGE_CASES`).
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
import torch

import noodl
from noodl.elements import ConstitutiveLaw
from noodl.nodesources import NodeSource

F64 = torch.float64
RTOL = 1e-10


@dataclass
class Case:
    """One law (or node source) with the drivers and the points it is checked at."""

    law: object
    x: torch.Tensor
    drivers: dict = field(default_factory=dict)


def _t(values):
    return torch.tensor(values, dtype=F64)


def _grid(n: int = 1, lo: float = -11.0, hi: float = 4.0, points: int = 121,
          seed: int = 0, centre: float = 0.0) -> torch.Tensor:
    """Points `centre +- 10^u` for u log-spaced over [lo, hi], with a seeded jitter so no point
    sits on a round-number switch; shape `(2 * points, n)`."""
    g = torch.Generator().manual_seed(seed)
    mag = 10.0 ** torch.linspace(lo, hi, points, dtype=F64)
    mag = mag * (1.0 + 0.1 * torch.rand(points, generator=g, dtype=F64))
    x = torch.cat([mag, -mag]).unsqueeze(-1).expand(-1, n)
    jitter = 1.0 + 1e-3 * torch.rand(x.shape, generator=g, dtype=F64)
    return centre + x * jitter


# ---------------------------------------------------------------------------- the cases
def _elements_cases() -> dict[type, Callable[[], list[Case]]]:
    from noodl.elements import (
        AirDuct,
        Conductance,
        CubicSplineFlow,
        Damper,
        Duct,
        DuctTerminal,
        FanCurve,
        FixedFlow,
        FixedVolumeFlow,
        OpenDoor,
        OperableDoor,
        PowerLaw,
        PrescribedFlow,
        Quadratic,
        RegularizedPowerLaw,
        Scaled,
        SplineFlowTable,
        UpstreamDensityPowerLaw,
        medium,
    )
    from noodl.elements.door_discretized import (
        DoorCompartment,
        DoorPortStream,
        OperableDoorCompartment,
        discretized_door,
        discretized_operable_door,
    )
    from noodl.elements.fan import AffinityFan

    air = medium("Buildings.Media.Air")
    rho = {"rho": _t([1.20, 1.17, 1.22])}
    T = {"T": _t([293.15, 297.0, 289.0])}
    door_drv = {"T": _t([293.15, 297.0, 289.0]), "p_abs": _t([101325.0, 101321.0, 101330.0]),
                "X_w": _t([0.01, 0.008, 0.012])}

    def fan(**kw):
        curve = FanCurve(_t([200.0, -100.0, -200.0, -800.0]), _t(0.5), kind="fan")
        return AffinityFan(curve, src=[0], tgt=[1], rho_ref=1.2041, kind="fan", **kw)

    def door_stream(operable: bool, y: float, direction: str):
        if operable:
            comp, head = discretized_operable_door(src=0, tgt=1, medium=air, y_key="y",
                                                   LClo=0.01, nCom=4)
        else:
            comp, head = discretized_door(src=0, tgt=1, medium=air, nCom=4)
        return Case(DoorPortStream(comp, head, direction, "door"), _grid(4, -8, 2),
                    {**door_drv, "y": _t(y)})

    return {
        Conductance: lambda: [Case(Conductance(_t([0.3, 2.0])), _grid(2))],
        PowerLaw: lambda: [
            Case(PowerLaw(_t([1e-3, 2e-2]), _t([0.65, 0.5])), _grid(2)),
            Case(PowerLaw(_t([1e-3]), _t([0.6]), regularised=0.01), _grid(1)),
        ],
        Quadratic: lambda: [Case(Quadratic(_t([3.0]), _t([0.5])), _grid(1))],
        RegularizedPowerLaw: lambda: [
            Case(RegularizedPowerLaw(C=1.3e-3, m=0.66, dp_turbulent=0.1, form="mass",
                                     rho_default=1.2), _grid(1)),
            Case(RegularizedPowerLaw(C=0.02, m=0.5, dp_turbulent=0.01, form="volume",
                                     rho_default=1.19), _grid(1)),
        ],
        UpstreamDensityPowerLaw: lambda: [Case(
            UpstreamDensityPowerLaw(_t([0.017, 0.002]), _t([0.5, 0.65]), src=[0, 1],
                                    tgt=[1, 2], m=_t([0.5, 0.65])), _grid(2), dict(rho))],
        Damper: lambda: [Case(Damper(_t(0.02), _t(0.5), _t(0.002), _t(0.65)), _grid(1))],
        Duct: lambda: [Case(Duct(_t([10.0]), _t([0.2]), _t([9e-5]), _t([1.5])),
                            _grid(1, -9, 3))],
        AirDuct: lambda: [Case(AirDuct(_t([10.0, 4.0]), _t([0.2, 0.1]), _t([9e-5, 1e-4]),
                                       _t([0.0, 2.0]), src=[0, 1], tgt=[1, 2],
                                       temperature_key="T"), _grid(2, -9, 3),
                               {**rho, **T})],
        DuctTerminal: lambda: [Case(DuctTerminal(_t([0.0314159]), _t([0.125]), src=[0],
                                                 tgt=[1]), _grid(1, -9, 3), dict(rho))],
        CubicSplineFlow: lambda: [
            Case(CubicSplineFlow([0.0, 2.0, 5.0, 10.0, 20.0], [0.0, 0.05, 0.09, 0.14, 0.2],
                                 src=[0], tgt=[1]), _grid(1, -6, 1.5), dict(rho)),
            Case(CubicSplineFlow([0.0, 0.05, 0.09, 0.14, 0.2], [0.0, 2.0, 5.0, 10.0, 20.0],
                                 independent="flow", volume=True, src=[0], tgt=[1]),
                 _grid(1, -6, 1.5), dict(rho)),
        ],
        SplineFlowTable: lambda: [Case(
            SplineFlowTable(_t([-50.0, -10.0, 0.0, 10.0, 50.0]),
                            _t([-0.4, -0.1, 0.0, 0.12, 0.5]), form="mass"),
            _grid(1, -6, 1.6))],
        FanCurve: lambda: [Case(FanCurve(_t([200.0, -100.0, -200.0, -800.0]), _t(0.5)),
                                _grid(1, -6, 2.6))],
        AffinityFan: lambda: [
            Case(fan(), _grid(1, -6, 2.6), dict(rho)),
            Case(fan(speed_key="s", off=0.1,
                     shutoff=PowerLaw(_t([1e-3]), _t([0.6]), kind="fan")),
                 _grid(1, -6, 2.6), {**rho, "s": _t([0.7])}),
            Case(fan(speed_key="s", off=0.1,
                     shutoff=PowerLaw(_t([1e-3]), _t([0.6]), kind="fan")),
                 _grid(1, -6, 2.6), {**rho, "s": _t([0.05])}),
        ],
        Scaled: lambda: [Case(Scaled(PowerLaw(_t([1e-3]), _t([0.65])), "s"), _grid(1),
                              {"s": _t([0.4])})],
        FixedFlow: lambda: [Case(FixedFlow(_t([0.1, -0.2])), _grid(2))],
        FixedVolumeFlow: lambda: [Case(FixedVolumeFlow(_t([0.04]), src=[0], tgt=[1]),
                                       _grid(1), dict(rho))],
        PrescribedFlow: lambda: [Case(PrescribedFlow("f", kind="ahs"), _grid(2),
                                      {"f": _t([0.3, -0.1])})],
        OpenDoor: lambda: [
            Case(OpenDoor(direction="ab", src=[0], tgt=[1], medium=air), _grid(1, -8, 2), T),
            Case(OpenDoor(direction="ba", src=[0, 1], tgt=[1, 2], medium=air,
                          wOpe=[0.9, 1.3]), _grid(2, -8, 2), T),
        ],
        OperableDoor: lambda: [
            Case(OperableDoor(direction="ab", src=[0], tgt=[1], medium=air, y_key="y",
                              LClo=0.01), _grid(1, -8, 2), {**T, "y": _t(y)})
            for y in (0.0, 0.4, 1.0)
        ],
        DoorPortStream: lambda: [door_stream(op, y, d) for op, y in
                                 ((False, 0.0), (True, 0.0), (True, 0.3))
                                 for d in ("ab", "ba")],
        DoorCompartment: lambda: [Case(
            DoorCompartment(src=[0, 0], tgt=[1, 1], medium=air, dA=[0.5, 0.5], kind="d"),
            _grid(2, -8, 2), door_drv)],
        OperableDoorCompartment: lambda: [Case(
            discretized_operable_door(src=0, tgt=1, medium=air, y_key="y", LClo=0.01,
                                      nCom=3)[0],
            _grid(3, -8, 2), {**door_drv, "y": _t(y)}) for y in (0.0, 0.3)],
    }


def _building_cases() -> dict[type, Callable[[], list[Case]]]:
    from noodl.apps.building_physics.modelica import storage
    from noodl.apps.building_physics.modelica.assemble import _HeldMover, _ZonalFlowEdge
    from noodl.elements import medium

    air = medium("Buildings.Media.Air")
    th = storage.thermo(air)
    k = th.k(_t([293.15, 297.0]), _t([0.0, 0.0]))
    zone_drv = {"T": _t([293.15, 297.0]), "X_w": _t([0.0, 0.0]),
                "air.storage_prev": torch.stack([_t([1.0, -2.0]), k], dim=-1),
                "air.storage_rate": _t([0.1, 0.1]), "air.storage_offset": _t([0.01, 0.0]),
                "air.storage_gain": _t([0.3, 0.0]), "air.storage_w_fed": _t([0.02, 0.0])}
    return {
        _ZonalFlowEdge: lambda: [Case(
            _ZonalFlowEdge(kind="z", direction="ab", src=0, tgt=1, medium=air, key="m"),
            _grid(1), {"m": _t(0.2)})],
        _HeldMover: lambda: [Case(_HeldMover("mover"), _grid(1))],
        storage.ZoneStorage: lambda: [Case(
            storage.ZoneStorage([0, 1], _t([50.0, 120.0]), 101325.0, th), _grid(2, -6, 3),
            zone_drv)],
    }


def _water_cases() -> dict[type, Callable[[], list[Case]]]:
    from noodl.apps.water.demand import PressureDrivenDemand
    from noodl.apps.water.elements import (
        CompositeDarcyWeisbach,
        HazenWilliams,
        MinorLoss,
        PumpCurve,
    )

    return {
        HazenWilliams: lambda: [
            Case(HazenWilliams(_t([1000.0, 300.0]), _t([0.3, 0.15]), _t([100.0, 130.0])),
                 _grid(2, -12, 3)),
            Case(HazenWilliams(_t([1000.0]), _t([0.3]), _t([100.0]), minor_loss=_t([2.0])),
                 _grid(1, -12, 3)),
        ],
        MinorLoss: lambda: [Case(MinorLoss(_t([3.0]), _t([0.2])), _grid(1, -12, 3))],
        PumpCurve: lambda: [
            Case(PumpCurve(_t([40.0]), _t([2000.0])), _grid(1, -9, 2.5, centre=-40.0)),
            Case(PumpCurve(_t([40.0]), _t([2000.0]), speed=_t([0.8]), status_key="st"),
                 _grid(1, -9, 2.5, centre=-25.6), {"st": _t([1.0])}),
        ],
        CompositeDarcyWeisbach: lambda: [
            Case(CompositeDarcyWeisbach(_t([500.0]), _t([0.25]), _t([0.0005])),
                 _grid(1, -12, 3)),
            Case(CompositeDarcyWeisbach(_t([500.0]), _t([0.25]), _t([0.0005]),
                                        minor_loss=_t([2.0])), _grid(1, -12, 3)),
        ],
        PressureDrivenDemand: lambda: [Case(
            PressureDrivenDemand(torch.tensor([0, 1]), _t([0.01, 0.02]), _t([0.0, 5.0]),
                                 p_min=10.0, p_req=30.0),
            _grid(2, -6, 1.6, centre=20.0))],
    }


def _sewer_cases() -> dict[type, Callable[[], list[Case]]]:
    from noodl.apps.sewer import geometry as g
    from noodl.apps.sewer.air import Headspace
    from noodl.apps.sewer.dynwave import DynamicWaveConduit, StructureLink

    n = 6
    conduit_drv = {
        "water.flow_offset": _t([0.1, 0.1, 0.1, 0.1, -0.2, 0.05]),
        "water.conveyance": _t([2.0, 2.0, 2.0, 2.0, 0.5, 1.0]),
        "water.normal_flow": _t([float("inf"), 0.15, 0.15, float("inf"), 0.3, 0.1]),
        "water.limited": _t([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        "water.slope_switch": _t([0.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
        "water.switch_drop": _t([0.0, 0.0, 0.05, 0.0, 0.0, -0.02]),
        "water.switch_slope": _t([0.0, 0.0, 100.0, 0.0, 0.0, 40.0]),
        "water.flow_last": _t([0.0, 0.0, 0.17, 0.0, 0.0, 0.08]),
        "water.dry_upstream": _t([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        "water.dry_downstream": _t([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        "water.max_flow": _t([float("inf"), 1.0, float("inf"), float("inf"), 0.25,
                              float("inf")]),
        "water.flap_block_reverse": _t([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    }
    culvert_drv = {**conduit_drv,
                   "water.cap_flow": _t([0.3, float("inf"), 0.3, float("inf"), float("inf"),
                                         float("inf")]),
                   "water.cap_slope": _t([0.7, 0.0, 0.7, 0.0, 0.0, 0.0]),
                   "water.cap_drop": _t([0.1, 0.0, 0.1, 0.0, 0.0, 0.0])}
    structure_drv = {"water.weir.flow_offset": _t([0.2, 0.2, -0.1]),
                     "water.weir.slope": _t([0.4, 0.4, 0.0]),
                     "water.weir.dp_fed": _t([0.1, 0.1, 0.0]),
                     "water.weir.flap_block_reverse": _t([0.0, 1.0, 0.0]),
                     "water.weir.flap_block_forward": _t([0.0, 0.0, 1.0])}

    def one(value, k=2):
        return torch.full((k,), value, dtype=F64)

    d, h = one(0.6, 1), one(0.18, 1)
    a_air, _, d_h = g.air_geometry(h, d)
    air_drv = {"sewer.A_air": a_air, "sewer.D_h": d_h, "sewer.T": g.top_width(h, d),
               "sewer.v": one(0.0, 1)}
    return {
        DynamicWaveConduit: lambda: [
            Case(DynamicWaveConduit(), _grid(n, -6, 1), conduit_drv),
            Case(DynamicWaveConduit(), _grid(n, -6, 1), culvert_drv),
        ],
        StructureLink: lambda: [Case(StructureLink("weir"), _grid(3, -6, 1), structure_drv)],
        Headspace: lambda: [Case(Headspace(_t([15.0]), torch.tensor([0])), _grid(1, -12, 2),
                                 air_drv)],
    }


def _electrical_cases() -> dict[type, Callable[[], list[Case]]]:
    from noodl.apps.electrical import (
        ConstantPowerLoad,
        JunctionDiode,
        PrescribedCurrent,
        Resistance,
        SwitchedConductance,
    )
    from noodl.apps.electrical.power import Shunt

    return {
        Resistance: lambda: [
            Case(Resistance(_t([10.0, 4.7])), _grid(2, -6, 3)),
            Case(Resistance(_t([10.0]), alpha=4e-3, temperature="T"), _grid(1, -6, 3),
                 {"T": _t([320.0])}),
        ],
        JunctionDiode: lambda: [
            Case(JunctionDiode(), _grid(1, -6, 0.3)),
            Case(JunctionDiode(temperature="T"), _grid(1, -6, 0.3), {"T": _t([320.0])}),
        ],
        SwitchedConductance: lambda: [Case(SwitchedConductance("c"), _grid(2, -6, 3),
                                           {"c": _t([1.0, 0.0])})],
        ConstantPowerLoad: lambda: [
            Case(ConstantPowerLoad(_t([100.0]), direct_current=True), _grid(1, 0, 3)),
        ],
        Shunt: lambda: [Case(Shunt(_t([0.1]), _t([0.0])), _grid(1, -6, 3))],
        PrescribedCurrent: lambda: [Case(PrescribedCurrent(_t([2.0])), _grid(1, -6, 3))],
    }


CASES: dict[type, Callable[[], list[Case]]] = {
    **_elements_cases(), **_building_cases(), **_water_cases(), **_sewer_cases(),
    **_electrical_cases(),
}

#: Classes with a potential form that `CASES` does not construct, and why.
EXEMPT: dict[str, str] = {
    # a mixin of the duct laws (`AirDuct`, `DuctTerminal`) with no `flow` of its own
    "noodl.elements.air_duct._UpstreamAir": "base class without a flow; see its subclasses",
    "noodl.elements.door._Door": "base class; see OpenDoor and OperableDoor",
    "noodl.elements.door_discretized._DoorCompartmentBase":
        "base class; see DoorCompartment and OperableDoorCompartment",
    # several ports: its derivative is the `jacobian` blocks, checked in
    # tests/apps/electrical/test_law_jacobians.py; it is solved on the MNA path only
    "noodl.apps.electrical.power.laws.PiBranch": "two-port law: jacobian, not dflow",
}


# ---------------------------------------------------------------------------- discovery
def _subclasses(base: type) -> set[type]:
    out, todo = set(), [base]
    while todo:
        for sub in todo.pop().__subclasses__():
            if sub not in out:
                out.add(sub)
                todo.append(sub)
    return out


def _discovered() -> list[type]:
    for info in pkgutil.walk_packages(noodl.__path__, "noodl."):
        importlib.import_module(info.name)
    found = [c for c in _subclasses(ConstitutiveLaw) | _subclasses(NodeSource)
             if c.__module__.startswith("noodl.")
             and getattr(c, "law_form", "potential") == "potential"]
    return sorted(found, key=lambda c: f"{c.__module__}.{c.__qualname__}")


def _name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def test_every_potential_form_law_is_audited():
    missing = [_name(c) for c in _discovered() if c not in CASES and _name(c) not in EXEMPT]
    assert not missing, f"add these to CASES (or EXEMPT, with the reason): {missing}"
    stale = [n for n in EXEMPT if n not in {_name(c) for c in _discovered()}]
    assert not stale, f"EXEMPT names classes that no longer exist: {stale}"


# ---------------------------------------------------------------------------- the check
def _call(law, x, drivers, method: str):
    fn = getattr(law, method)
    if getattr(law, "contextual", False) and method == "flow":
        from noodl.elements import LawContext
        return fn(x, drivers, LawContext(mode="dc", t=0.0))
    return fn(x, drivers)


def _autograd_slope(law, x, drivers) -> torch.Tensor:
    leaf = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        q = _call(law, leaf, drivers, "flow")
        if not q.is_complex():
            (grad,) = torch.autograd.grad(q.sum(), leaf)
            return grad
        # a complex coefficient on a real dp (a phasor law): the slope of each part
        (re,) = torch.autograd.grad(q.real.sum(), leaf, retain_graph=True)
        (im,) = torch.autograd.grad(q.imag.sum(), leaf)
    return torch.complex(re, im)


def _mismatch(case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(x, dflow, autograd)` at the points where they differ beyond `RTOL`."""
    law, x, drv = case.law, case.x, case.drivers
    with torch.no_grad():
        analytic = _call(law, x, drv, "dflow").expand_as(x)
    reference = _autograd_slope(law, x, drv)
    bad = (analytic - reference).abs() > RTOL * torch.maximum(analytic.abs(),
                                                              reference.abs())
    bad = bad | (torch.isfinite(analytic) != torch.isfinite(reference))
    return x[bad], analytic[bad], reference[bad]


def _params():
    return [pytest.param(cls, id=cls.__qualname__) for cls in CASES]


@pytest.mark.parametrize("cls", _params())
def test_dflow_is_the_derivative_of_flow_unless_declared(cls):
    cases = CASES[cls]()
    assert cases and all(type(c.law) is cls for c in cases)
    for case in cases:
        law = case.law
        if law.dp_independent if hasattr(law, "dp_independent") else False:
            # the solve never differentiates it: dflow is 0 and the flow does not move
            with torch.no_grad():
                assert torch.equal(law.dflow(case.x, case.drivers),
                                   torch.zeros_like(law.dflow(case.x, case.drivers)))
                assert torch.equal(_call(law, case.x, case.drivers, "flow"),
                                   _call(law, 1.37 * case.x + 0.1, case.drivers, "flow"))
            continue
        x, analytic, reference = _mismatch(case)
        if law.exact_slope:
            assert x.numel() == 0, (
                f"{cls.__qualname__}: dflow differs from the autograd derivative of flow at "
                f"{x[:5].tolist()}: dflow {analytic[:5].tolist()}, autograd "
                f"{reference[:5].tolist()}; fix dflow or declare exact_slope = False")
    if not getattr(cls, "exact_slope", True):
        # a declared Newton aid must really differ somewhere, or the flag is stale
        assert any(_mismatch(c)[0].numel() for c in cases), (
            f"{cls.__qualname__} declares exact_slope = False but its dflow equals the "
            f"autograd derivative at every audited point")


def test_a_wrapping_law_declares_its_inner_law_s_slope():
    from noodl.elements import FanCurve, PowerLaw, Scaled, medium
    from noodl.elements.door_discretized import DoorPortStream, discretized_door
    from noodl.elements.fan import AffinityFan

    comp, head = discretized_door(src=0, tgt=1, medium=medium("Buildings.Media.Air"), nCom=2)
    stream = DoorPortStream(comp, head, "ab", "door")
    assert not Scaled(stream, "s").exact_slope
    assert Scaled(PowerLaw(_t([1e-3]), _t([0.6])), "s").exact_slope
    curve = FanCurve(_t([200.0, -100.0, -200.0, -800.0]), _t(0.5), kind="door")
    assert AffinityFan(curve, src=[0], tgt=[1], rho_ref=1.2).exact_slope
    assert not AffinityFan(curve, src=[0], tgt=[1], rho_ref=1.2, shutoff=stream).exact_slope


# ---------------------------------------------------------------------------- node storage
# `NodeStorage.rate_slope` builds the differentiable solve's Jacobian and adjoint as `dflow`
# does a law's (classic and Modified Nodal Analysis alike), unless the storage declares
# `exact_slope = False`. The same audit: `rate_slope` against the autograd derivative of `rate`
# with respect to the end-of-step potential, under both implicit schemes.
@dataclass
class StorageCase:
    """One storage with the drivers, the step-start potentials and the points it is checked
    at (`phi_start` and `previous` broadcast against the points)."""

    storage: object
    x: torch.Tensor
    phi_start: torch.Tensor
    previous: torch.Tensor
    drivers: dict = field(default_factory=dict)
    schemes: tuple = ("backward_euler", "trapezoidal")


def _storage_cases() -> dict[type, Callable[[], list[StorageCase]]]:
    from noodl.apps.building_physics.prj import ZoneAirMassChange
    from noodl.apps.sewer.dynwave import JunctionStorage
    from noodl.storage import LinearStorage

    zone = {"P_ref": _t(101325.0), "T": _t([293.15, 297.0]),
            "air.mass_start": _t([60.0, 140.0])}

    def one(value, k=2):
        return torch.full((k,), value, dtype=F64)

    # the sewer junction storage (`phi_start` and `previous` are not read: its step start is
    # SWMM's, from the drivers)
    zero2 = _t([0.0, 0.0])
    store_drv = {"water.storage_area": one(4.0), "water.pond_area": one(4.5),
                 "water.head_start": one(0.46),
                 "water.net_inflow_start": one(0.02), "water.net_inflow_last": one(0.03),
                 "water.head_last": _t([0.46, 0.4]), "water.pond_switch_slope": one(40.0),
                 "water.ponded": one(0.0), "water.surcharged": _t([0.0, 1.0]),
                 "water.surcharge_slope": one(0.01), "water.flood_excess": one(0.002)}
    plain_drv = {"water.storage_area": _t([2.0, 2.0]), "water.head_start": _t([1.5, 1.0]),
                 "water.net_inflow_start": _t([0.3, 0.0])}
    # the drop-end area switch about H* = 1.1 m (junction 0; none at junction 1)
    switch_drv = {"water.storage_area": one(2.0), "water.head_start": one(1.0),
                  "water.net_inflow_start": one(0.1),
                  "water.area_switch_head": _t([1.1, float("inf")]),
                  "water.area_below": one(2.0), "water.area_above": one(3.0),
                  "water.area_switch_slope": one(10.0), "water.net_inflow_last": one(0.0)}
    swmm = ("trapezoidal",)     # SWMM's continuity: `JunctionStorage` refuses other schemes
    return {
        LinearStorage: lambda: [
            StorageCase(LinearStorage([0], 2.5), _grid(1), _t([0.3]), _t([0.01])),
            StorageCase(LinearStorage([0, 1], _t([1e-6, 40.0]), learnable=True),
                        _grid(2, -8, 3), _t([1.0, -2.0]), _t([0.0, 0.5])),
        ],
        JunctionStorage: lambda: [
            StorageCase(JunctionStorage(torch.tensor([0, 1]), bed=_t([1.0, 1.0]), bed_area=0.5),
                        _grid(2, -6, 0.5, centre=1.0), zero2, zero2, plain_drv, swmm),
            StorageCase(JunctionStorage(torch.tensor([0, 1]), bed=_t([0.0, 0.0]), bed_area=0.5),
                        _grid(2, -6, 0.3, centre=1.1), zero2, zero2, switch_drv, swmm),
            StorageCase(JunctionStorage(torch.arange(2), torch.zeros(2, dtype=F64),
                                        bed_area=12.566 * 0.0929, crown=one(0.3), top=one(0.5),
                                        surchargeable=torch.tensor([True, True])),
                        _grid(2, -6, 0.3, centre=0.4), zero2, zero2, store_drv, swmm),
            StorageCase(JunctionStorage(torch.arange(2), torch.zeros(2, dtype=F64),
                                        bed_area=12.566 * 0.0929, crown=one(0.3), top=one(0.5),
                                        surchargeable=torch.tensor([False, False]),
                                        pond_head=_t([0.5, float("inf")])),
                        _grid(2, -6, 0.3, centre=0.4), zero2, zero2,
                        {**store_drv, "water.surcharged": one(0.0)}, swmm),
        ],
        ZoneAirMassChange: lambda: [
            StorageCase(ZoneAirMassChange([0, 1], _t([50.0, 120.0])), _grid(2, -6, 3),
                        _t([0.0, 0.0]), _t([0.0, 0.0]), zone),
            # without "<layer>.mass_start" it stores nothing: rate and slope are both zero
            StorageCase(ZoneAirMassChange([0, 1], _t([50.0, 120.0])), _grid(2, -6, 3),
                        _t([0.0, 0.0]), _t([0.0, 0.0]), {"P_ref": _t(101325.0),
                                                         "T": _t([293.15, 297.0])}),
        ],
    }


STORAGE_CASES = _storage_cases()


def _storages() -> list[type]:
    from noodl.storage import NodeStorage

    for info in pkgutil.walk_packages(noodl.__path__, "noodl."):
        importlib.import_module(info.name)
    found = [c for c in _subclasses(NodeStorage) if c.__module__.startswith("noodl.")]
    return sorted(found, key=_name)


def test_every_node_storage_is_audited():
    missing = [_name(c) for c in _storages() if c not in STORAGE_CASES]
    assert not missing, f"add these to STORAGE_CASES: {missing}"


def _storage_mismatch(case: StorageCase, ctx):
    """`(x, rate_slope, autograd)` at the points where they differ beyond `RTOL`."""
    from noodl.storage import NodeStorage

    st, x, drv = case.storage, case.x, case.drivers
    phi_start = case.phi_start.expand_as(x)
    previous = case.previous.expand_as(x)
    assert isinstance(st, NodeStorage)
    with torch.no_grad():
        analytic = st.rate_slope(x, phi_start, previous, ctx, drv).expand_as(x)
    leaf = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        w = st.rate(leaf, phi_start, previous, ctx, drv)
        if w.requires_grad:
            (reference,) = torch.autograd.grad(w.sum(), leaf)
        else:   # a rate that does not depend on the potential (nothing stored)
            reference = torch.zeros_like(x)
    bad = (analytic - reference).abs() > RTOL * torch.maximum(analytic.abs(),
                                                              reference.abs())
    bad = bad | (torch.isfinite(analytic) != torch.isfinite(reference))
    return x[bad], analytic[bad], reference[bad]


@pytest.mark.parametrize("cls", [pytest.param(c, id=c.__qualname__) for c in STORAGE_CASES])
def test_rate_slope_is_the_derivative_of_rate_unless_declared(cls):
    from noodl.elements import LawContext

    cases = STORAGE_CASES[cls]()
    assert cases and all(type(c.storage) is cls for c in cases)
    differs = False
    for case in cases:
        contexts = [LawContext(mode="transient", dt=dt, t=dt, scheme=scheme)
                    for scheme in case.schemes for dt in (1e-3, 30.0)]
        for ctx in contexts:
            x, analytic, reference = _storage_mismatch(case, ctx)
            differs = differs or x.numel() > 0
            if case.storage.exact_slope:
                assert x.numel() == 0, (
                    f"{cls.__qualname__} ({ctx.scheme}, dt={ctx.dt}): rate_slope differs "
                    f"from the autograd derivative of rate at {x[:5].tolist()}: "
                    f"rate_slope {analytic[:5].tolist()}, autograd {reference[:5].tolist()}; "
                    f"fix rate_slope or declare exact_slope = False")
    if not getattr(cls, "exact_slope", True):
        assert differs, (
            f"{cls.__qualname__} declares exact_slope = False but its rate_slope equals the "
            f"autograd derivative at every audited point")


def test_node_storage_is_exact_by_default():
    from noodl.storage import LinearStorage, NodeStorage

    assert NodeStorage.exact_slope is True
    assert LinearStorage([0], 1.0).exact_slope

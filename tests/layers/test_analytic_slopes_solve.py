"""The differentiable classic solve with analytic slopes (`FunctionalLaws.dflows`).

A law (or node source) that declares `exact_slope` gives the Newton Jacobian and the adjoint
its own `dflow`; one that does not is differentiated by autograd. `gradcheck` of
`KirchhoffLayer.solve(differentiable=True)` with respect to a learnable law parameter, a
driver, `sources` and `phi_boundary` on a building, a water and a sewer network; and a law
whose `dflow` is a Newton aid, which gives correct gradients only because it declares
`exact_slope = False`.
"""

from __future__ import annotations

import pytest
import torch

from noodl.drives import ConstantDrive
from noodl.elements import OpenDoor, PowerLaw, medium
from noodl.layers._functional import FunctionalLaws
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64
TIGHT = dict(atol=1e-13, rtol=0.0)


def _t(values, grad=False):
    return torch.tensor(values, dtype=F64, requires_grad=grad)


def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


def _gradcheck(layer, phi_b, drivers, sources, params=(), keys=(), eps=None, solve_step=None):
    """gradcheck of the interior potentials and the flows with respect to each of the law
    parameters `params`, the drivers named `keys`, `sources` and `phi_boundary`, one at a
    time; `eps` maps an input's position in that order to its own step (a coefficient of
    order 100 needs a larger one than a source of order 0.01). `solve_step` holds the solve's step
    keywords (`dt`, `history`, `storage_flow`) for a layer with storage."""
    inputs = [*params, *(drivers[k] for k in keys), sources, phi_b]
    for i, x in enumerate(inputs):
        def f(x_, i=i):
            drv, src, pb = dict(drivers), sources, phi_b
            if len(params) <= i < len(params) + len(keys):
                drv[keys[i - len(params)]] = x_
            elif i == len(inputs) - 2:
                src = x_
            elif i == len(inputs) - 1:
                pb = x_
            # a law parameter is read from the law itself, which gradcheck perturbs in place
            phi, q = layer.solve(pb, drv, src, differentiable=True, **(solve_step or {}), **TIGHT)
            return phi[..., layer.interior], q

        step = (eps or {}).get(i, 1e-6)
        assert torch.autograd.gradcheck(f, (x,), eps=step, atol=1e-8, rtol=1e-6)


# ---------------------------------------------------------------------------- building
def _building():
    """ambient -> z1 -> z2 -> ambient with a learnable power law, z1 <-> z2 also through an
    open door (MBL's `DoorOpen`), and a wind drive on the power-law edges."""
    air = medium("Buildings.Media.Air")
    net = _net(["amb", "z1", "z2"], [("amb", "z1", "airpath"), ("z1", "z2", "airpath"),
                                     ("z2", "amb", "airpath"), ("z1", "z2", "door")])
    law = PowerLaw(_t([0.010, 0.008, 0.012]), _t(0.65), learnable=True)
    door = OpenDoor(direction="ab", src=[1], tgt=[2], medium=air, kind="door")
    layer = KirchhoffLayer(net, "air", [law, door], [ConstantDrive(kind="airpath", key="wind")],
                           boundary=["amb"])
    drivers = {"wind": _t([12.0, 0.0, -3.0], grad=True),
               "T": _t([288.15, 295.15, 291.15])}
    return layer, law, drivers


def test_gradcheck_building_network():
    layer, law, drivers = _building()
    sources = _t([0.0, 0.01, -0.02], grad=True)
    phi_b = _t([0.0], grad=True)
    _gradcheck(layer, phi_b, drivers, sources, (law.C,), keys=("wind",))


# ---------------------------------------------------------------------------- water
def _water():
    """A reservoir feeding three junctions through Hazen-Williams pipes (one with a minor
    loss), a pressure-driven demand at two of them."""
    from noodl.apps.water.demand import PressureDrivenDemand
    from noodl.apps.water.elements import HazenWilliams

    net = _net(["R", "J1", "J2", "J3"], [("R", "J1", "pipe"), ("J1", "J2", "pipe"),
                                         ("J1", "J3", "pipe"), ("J2", "J3", "pipe"),
                                         ("R", "J3", "valve")])
    pipes = HazenWilliams(_t([800.0, 400.0, 600.0, 300.0]), _t([0.3, 0.2, 0.2, 0.15]),
                          _t([110.0, 120.0, 100.0, 130.0]), learnable=True)
    valve = HazenWilliams(_t([50.0]), _t([0.1]), _t([100.0]), minor_loss=_t([4.0]),
                          kind="valve")
    demand = PressureDrivenDemand(torch.tensor([2, 3]), _t([0.02, 0.015]), _t([5.0, 8.0]),
                                  p_min=0.0, p_req=30.0)
    layer = KirchhoffLayer(net, "water", [pipes, valve], [ConstantDrive(kind="pipe",
                                                                        key="pump_head")],
                           boundary=["R"], node_sources=[demand])
    drivers = {"pump_head": _t([3.0, 0.0, 0.0, 0.0], grad=True)}
    return layer, pipes, drivers


def test_gradcheck_water_network():
    layer, pipes, drivers = _water()
    sources = _t([0.0, 0.01, 0.0, 0.005], grad=True)
    phi_b = _t([25.0], grad=True)
    phi, _ = layer.solve(phi_b, drivers, sources, differentiable=True, **TIGHT)
    pressure = phi[2:] - _t([5.0, 8.0])
    assert bool(((pressure > 0.0) & (pressure < 30.0)).all())  # inside the demand curve
    _gradcheck(layer, phi_b, drivers, sources, (pipes.roughness,), keys=("pump_head",),
               eps={0: 1e-3})


# ---------------------------------------------------------------------------- sewer
def _sewer():
    """Three junctions in a line draining to an outfall: SWMM's dynamic-wave conduits with
    their coefficients frozen for one pass, and the junctions' trapezoidal storage. The
    sewer laws hold no parameters (their coefficients are drivers), so the drivers stand in
    for one."""
    from noodl.apps.sewer.dynwave import DynamicWaveConduit, JunctionStorage

    net = _net(["J1", "J2", "J3", "O"], [("J1", "J2", "pipe"), ("J2", "J3", "pipe"),
                                         ("J3", "O", "pipe")])
    storage = JunctionStorage(torch.tensor([0, 1, 2]), bed=_t([0.0, -0.1, -0.2]),
                              bed_area=1.0)
    layer = KirchhoffLayer(net, "water", [DynamicWaveConduit()], boundary=["O"],
                           storage=[storage], quantity="head")
    three = torch.ones(3, dtype=F64)
    drivers = {
        "water.flow_offset": _t([0.02, 0.03, 0.05], grad=True),
        "water.conveyance": _t([0.8, 1.1, 1.5], grad=True),
        "water.normal_flow": three * float("inf"), "water.limited": 0.0 * three,
        "water.slope_switch": 0.0 * three, "water.switch_drop": 0.0 * three,
        "water.switch_slope": 0.0 * three, "water.flow_last": 0.0 * three,
        "water.dry_upstream": 0.0 * three, "water.dry_downstream": 0.0 * three,
        "water.storage_area": _t([3.0, 2.5, 4.0]), "water.head_start": _t([0.6, 0.5, 0.4]),
        "water.net_inflow_start": _t([0.01, 0.0, -0.01]),
    }
    return layer, drivers


def _sewer_step() -> dict:
    """The solve keywords of a 10 s step of `_sewer`; its junction storage reads the step
    start from the drivers, not from `history` or `storage_flow`."""
    return {"dt": 10.0, "history": (_t([0.6, 0.5, 0.4, 0.1]), None),
            "storage_flow": _t([0.0, 0.0, 0.0])}


def test_gradcheck_sewer_network():
    layer, drivers = _sewer()
    step = _sewer_step()
    sources = _t([0.03, 0.01, 0.0, 0.0], grad=True)
    phi_b = _t([0.1], grad=True)
    phi, _ = layer.solve(phi_b, drivers, sources, differentiable=True, **step, **TIGHT)
    assert bool((phi[:3] > _t([0.0, -0.1, -0.2])).all())  # above the beds: one smooth piece
    _gradcheck(layer, phi_b, drivers, sources, keys=("water.flow_offset", "water.conveyance"),
               solve_step=step)


# ---------------------------------------------------------------------------- the switch
class _AidedPowerLaw(PowerLaw):
    """A power law whose `dflow` is a Newton aid (twice the slope): not the derivative."""

    exact_slope = False

    def dflow(self, dp, drivers=None):
        return 2.0 * super().dflow(dp, drivers)


def _aided(exact: bool):
    layer, law, drivers = _building()
    aided = _AidedPowerLaw(law.C.detach().clone(), law.n.detach().clone(), learnable=True)
    aided.exact_slope = exact
    layer = KirchhoffLayer(layer.net, "air", [aided, layer._laws[1]], list(layer._drives),
                           boundary=["amb"])
    return layer, aided, drivers


def test_the_operator_uses_dflow_only_where_it_is_exact():
    """The slopes `FunctionalLaws` hands Newton and the adjoint: the law's own `dflow` for an
    exact law (bit for bit), autograd through its flow otherwise."""
    for exact in (True, False):
        layer, aided, drivers = _aided(exact)
        phi_b, sources = _t([0.0]), _t([0.0, 0.01, -0.02])
        fn = FunctionalLaws(layer, drivers, sources, phi_b)
        rebuilt, drv, _, _, _ = fn.rebuild(fn.all_params)
        phi = layer.assemble(_t([3.0, 1.0]), phi_b)
        with torch.no_grad():
            slopes = fn.dflows(phi, drv, rebuilt)
            analytic = layer.dflows(phi, drivers)
        door = layer.kind_slice("door")
        assert torch.equal(slopes[door], analytic[door])   # the door's dflow is exact
        airpath = layer.kind_slice("airpath")
        if exact:
            assert torch.equal(slopes[airpath], analytic[airpath])
        else:
            torch.testing.assert_close(slopes[airpath], 0.5 * analytic[airpath], rtol=1e-14,
                                       atol=0.0)


def test_a_newton_aid_declared_inexact_keeps_correct_gradients():
    layer, aided, drivers = _aided(False)
    sources, phi_b = _t([0.0, 0.01, -0.02], grad=True), _t([0.0], grad=True)
    _gradcheck(layer, phi_b, drivers, sources, (aided.C,), keys=("wind",))


def test_a_newton_aid_declared_exact_corrupts_the_gradient():
    """Why the declaration matters: the same law claiming `exact_slope` gives the adjoint
    twice its slope, and the gradient is wrong (the forward root is still right)."""
    layer, aided, drivers = _aided(True)
    phi_b, sources = _t([0.0]), _t([0.0, 0.01, -0.02])

    def potentials(C):
        with torch.no_grad():
            aided.C.copy_(C)
        return layer.solve(phi_b, drivers, sources, differentiable=True, **TIGHT)[0]

    C0 = aided.C.detach().clone()
    phi = layer.solve(phi_b, drivers, sources, differentiable=True, **TIGHT)[0]
    (adjoint,) = torch.autograd.grad(phi[1], aided.C)
    h = 1e-7
    central = torch.stack([
        (potentials(C0 + h * e)[1] - potentials(C0 - h * e)[1]) / (2 * h)
        for e in torch.eye(3, dtype=F64)])
    with pytest.raises(AssertionError):
        torch.testing.assert_close(adjoint, central, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("network", [_building, _water, _sewer])
def test_a_solve_nothing_differentiable_reaches_is_the_primal_solve(network):
    """With every slope exact, the differentiable forward Newton is the primal one: the same
    slopes, residuals and iterates, so the same `(phi, q)` bit for bit (the sewer relied on
    this through `analytic_primal`, which is gone)."""
    layer, *_, drivers = network()
    drivers = {k: v.detach() for k, v in drivers.items()}
    phi_b = _t([25.0]) if network is _water else _t([0.1])
    sources = torch.zeros(layer._n_nodes, dtype=F64)
    step = _sewer_step() if network is _sewer else {}
    for owner in (*layer._laws, *layer._node_sources):
        for p in owner.parameters():
            p.requires_grad_(False)
    out = {}
    for differentiable in (True, False):
        diag: dict = {}
        out[differentiable] = layer.solve(phi_b, drivers, sources, diagnostics=diag,
                                          differentiable=differentiable, **step,
                                          **TIGHT), diag
    (phi_d, q_d), diag_d = out[True]
    (phi_p, q_p), diag_p = out[False]
    assert torch.equal(phi_d, phi_p) and torch.equal(q_d, q_p)
    assert diag_d["newton_iterations"] == diag_p["newton_iterations"]


# ---------------------------------------------------------------------------- MNA twin
def _mna_aided(exact: bool):
    """A prescribed potential in series with a power law and an aided power-law load: the
    ideal source makes the layer an MNA one."""
    from tests._circuit_laws import Source

    net = _net(["g", "a", "b"], [("a", "g", "source"), ("a", "b", "r"), ("b", "g", "load")])
    aided = _AidedPowerLaw(_t([0.3]), _t(0.8), kind="load", dp_transition=1e-6,
                           learnable=True)
    aided.exact_slope = exact
    layer = KirchhoffLayer(
        net, "e", [Source(_t([1.0])), PowerLaw(_t([0.5]), _t(0.65), kind="r",
                                               dp_transition=1e-6), aided],
        boundary=["g"])
    assert layer.mna
    return layer, aided


def test_mna_a_newton_aid_declared_inexact_keeps_correct_gradients():
    layer, aided = _mna_aided(False)
    sources, phi_b = _t([0.0, 0.2, -0.1], grad=True), _t([0.0], grad=True)
    _gradcheck(layer, phi_b, {}, sources, (aided.C,))


def test_mna_a_newton_aid_declared_exact_corrupts_the_gradient():
    """The gate matters on the MNA path too: claiming `exact_slope` hands the adjoint the
    aid's doubled slope."""
    layer, aided = _mna_aided(True)
    phi_b, sources = _t([0.0]), _t([0.0, 0.2, -0.1])

    def potential(C):
        with torch.no_grad():
            aided.C.copy_(C)
        return layer.solve(phi_b, {}, sources, differentiable=True, **TIGHT)[0][2]

    C0 = aided.C.detach().clone()
    phi = layer.solve(phi_b, {}, sources, differentiable=True, **TIGHT)[0]
    (adjoint,) = torch.autograd.grad(phi[2], aided.C)
    h = 1e-7
    central = (potential(C0 + h) - potential(C0 - h)) / (2 * h)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(adjoint, central.reshape(adjoint.shape), rtol=1e-4,
                                   atol=1e-8)

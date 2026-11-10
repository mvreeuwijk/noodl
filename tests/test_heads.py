"""EdgeHeads: the driver "<layer>.edge_head" a closure writes and a Kirchhoff layer adds to dp."""

from __future__ import annotations

import pytest
import torch

from noodl.elements import Conductance, PowerLaw
from noodl.heads import DriverHead, DuctStack, EdgeHeads, Stack, Wind
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.model import Model
from noodl.topology import Network
from tests._circuit_laws import Cap, Source

F64 = torch.float64


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _net():
    """ambient -> z1 (low and high opening), z1 -> z2, z2 -> ambient: kind airpath (4 edges),
    plus one duct edge z2 -> ambient of a second kind."""
    net = Network(dtype=F64)
    net.add_node("ambient", z_ref=0.0)
    net.add_node("z1", z_ref=0.0)
    net.add_node("z2", z_ref=0.0)
    net.add_edge("ambient", "z1", kind="airpath", z_path=-1.0)
    net.add_edge("ambient", "z1", kind="airpath", z_path=1.0)
    net.add_edge("z1", "z2", kind="airpath", z_path=0.5)
    net.add_edge("z2", "ambient", kind="airpath", z_path=2.0)
    net.add_edge("z2", "ambient", kind="duct", z_path=0.0)
    return net


def _laws():
    return [PowerLaw(torch.tensor([0.01, 0.012, 0.02, 0.01]), 0.65, kind="airpath"),
            PowerLaw(torch.tensor([0.015]), 0.5, kind="duct")]


def _air(net):
    return KirchhoffLayer(net, "air", _laws(), boundary=["ambient"])


def _terms(net):
    return [Stack.from_network(net, "airpath"), DriverHead("duct", "duct_head")]


def _drivers():
    return {"rho": torch.tensor([1.2, 1.15, 1.1]), "duct_head": torch.tensor([3.0])}


PHI_B = torch.zeros(1, dtype=F64)


def test_dp_adds_the_head_driver_to_every_edge():
    net = _net()
    terms = _terms(net)
    layer = _air(net)
    drivers = _drivers()
    head = EdgeHeads(layer, terms).head(drivers)
    assert head.shape == (5,)
    phi = torch.tensor([0.0, 4.0, -2.0])
    assert torch.equal(layer.dp(phi, {**drivers, "air.edge_head": head}),
                       layer._difference(phi) + head)
    assert torch.equal(layer.dp(phi, drivers), layer._difference(phi))   # absent means zero


def test_two_terms_on_one_kind_add_up():
    net = _net()
    terms = [Stack.from_network(net, "airpath"), DriverHead("airpath", "gust")]
    layer = _air(net)
    drivers = {"rho": torch.tensor([1.2, 1.15, 1.1]),
               "gust": torch.tensor([1.0, -2.0, 0.5, 3.0])}
    head = EdgeHeads(layer, terms).head(drivers)
    expected = terms[0](drivers) + drivers["gust"]
    torch.testing.assert_close(head[:4], expected, rtol=1e-14, atol=0)
    assert head[4].item() == 0.0


def _rc():
    # source a -> gnd, resistor a -> b (edge 1), capacitor b -> gnd (dynamic edge 2)
    net = Network(dtype=F64)
    for n in ("gnd", "a", "b"):
        net.add_node(n)
    for u, v, k in (("a", "gnd", "source"), ("a", "b", "r"), ("b", "gnd", "cap")):
        net.add_edge(u, v, kind=k)
    return KirchhoffLayer(
        net, "e", [Source(torch.tensor([1.0])), Conductance(torch.tensor([2.0]), kind="r"),
                   Cap(torch.tensor([1.0]))], boundary=["gnd"])


@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_mna_solve_with_a_head_driver_moves_the_resistor_edge(scheme):
    layer = _rc()
    head = EdgeHeads(layer, [DriverHead("r", "h")]).head({"h": torch.tensor([0.25])})
    assert head.tolist() == [0.0, 0.25, 0.0]
    hist = (torch.zeros(3), torch.zeros(3))
    for kw in ({"mode": "dc"}, {"dt": 0.1, "scheme": scheme, "history": hist}):
        a = layer.solve(PHI_B, {}, differentiable=False, **kw)
        b = layer.solve(PHI_B, {"e.edge_head": head}, differentiable=False, **kw)
        assert not torch.equal(a[0], b[0])


def test_a_head_on_a_dynamic_edge_is_refused_by_name():
    layer = _rc()
    hist = (torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match=r"'e'.*'e.edge_head'.*not zero.*cap"):
        layer.solve(PHI_B, {"e.edge_head": torch.tensor([0.0, 0.0, 0.3])}, dt=0.1, history=hist,
                    differentiable=False)
    # a zero entry on the dynamic edge is fine
    layer.solve(PHI_B, {"e.edge_head": torch.tensor([0.0, 0.3, 0.0])}, dt=0.1, history=hist,
                differentiable=False)


def test_gradcheck_through_a_closure_only_parameter():
    net = _net()
    layer = KirchhoffLayer(net, "air", _laws(), boundary=["ambient"])
    heads = EdgeHeads(layer, [DriverHead("airpath", "h"), DriverHead("duct", "d")])
    base = torch.tensor([2.0, -1.0, 3.0, 1.5])
    k = torch.tensor([1.0, 0.5], dtype=F64, requires_grad=True)   # reaches dp only via head

    def f(k_):
        head = heads.head({"h": k_[0] * base, "d": k_[1:2] * 4.0})
        phi, q = layer.solve(PHI_B, {"air.edge_head": head}, differentiable=True,
                             atol=1e-12, rtol=1e-12)
        return torch.cat([phi[1:], q])

    assert torch.autograd.gradcheck(f, (k,), eps=1e-6, atol=1e-5)


def test_refusals_name_the_layer_and_the_key():
    net = _net()
    layer = _air(net)
    with pytest.raises(ValueError, match=r"'air'.*'air.edge_head'.*\(5,.*width 4"):
        layer.dp(torch.zeros(3), {"air.edge_head": torch.zeros(4)})
    with pytest.raises(ValueError, match=r"term kind 'nope'.*'air'"):
        EdgeHeads(layer, [DriverHead("nope", "h")])
    heads = EdgeHeads(layer, [DriverHead("airpath", "h")])
    with pytest.raises(ValueError, match=r"kind 'airpath' returned width 3.*4 edges"):
        heads.head({"h": torch.zeros(3)})

    class OldStyle:
        kind = "airpath"

        def __call__(self, phi, drivers):
            return drivers["h"]

    with pytest.raises(TypeError, match=r"OldStyle"):
        EdgeHeads(layer, [OldStyle()])
    with pytest.raises(ValueError, match="no head terms"):
        EdgeHeads(layer, [])


def test_a_term_on_a_dynamic_kind_is_refused():
    layer = _rc()
    with pytest.raises(ValueError, match=r"layer 'e': kind 'cap' has a dynamic law"):
        EdgeHeads(layer, [DriverHead("cap", "h")])


def test_inputs_are_the_union_of_the_terms_keys_or_unknown():
    net = _net()
    layer = _air(net)
    heads = EdgeHeads(layer, [Stack.from_network(net, "airpath", rho_key="rho"),
                              DriverHead("duct", "duct_head")])
    assert heads.inputs == ("duct_head", "rho") and heads.outputs == ("air.edge_head",)
    duct = DuctStack("duct", src=[2], tgt=[0], z_ref=[0.0, 0.0, 3.0], rho_key="rho_d")
    assert duct.inputs == ("rho_d",)
    wind = Wind("airpath", sign=[1.0], envelope=[1.0], azimuth=[0.0], ch=[1.0],
                cp_const=[0.5], profile_index=[0])
    assert wind.inputs == ("rho_amb", "V_met")

    class Bare:
        kind = "duct"

        def __call__(self, drivers):
            return drivers["x"]

    assert EdgeHeads(layer, [Bare()]).inputs is None


class Density:
    inputs = ("T",)
    outputs = ("rho",)

    def __call__(self, state, drivers):
        return {"rho": 101325.0 / (287.05 * drivers["T"])}


def _model():
    net = _net()
    layer = KirchhoffLayer(net, "air", _laws(), boundary=["ambient"],
                           quantity="pressure", unit="Pa")
    closures = [Density(), EdgeHeads(layer, _terms(net))]
    return Model(net, {"air": layer}, closures=closures), closures


def test_model_with_a_head_closure_steps_with_the_head_in_the_pass():
    drivers = {"air.phi_boundary": PHI_B, "T": torch.tensor([268.0, 294.0, 296.0]),
               "duct_head": torch.tensor([3.0])}
    model, _ = _model()
    state = {}
    for _ in range(3):
        state = model.step(state, drivers, 60.0)
        assert torch.isfinite(state["air.phi"]).all()
    assert "air.edge_head" in model._apply_closures(state, drivers)


def test_model_to_moves_the_term_tensors():
    model, closures = _model()
    stack = closures[1].terms[0]
    assert stack.z_path.dtype == torch.float64
    model.to(dtype=torch.float32)
    assert stack.z_path.dtype == torch.float32 and stack.z_ref.dtype == torch.float32


def test_gradcheck_through_a_state_dependent_stack_head_classic():
    """The head is a Stack whose density depends on a temperature (the way a Density closure
    feeds it): the gradient reaches the temperature through the head."""
    net = _net()
    layer = KirchhoffLayer(net, "air", _laws(), boundary=["ambient"])
    heads = EdgeHeads(layer, [Stack.from_network(net, "airpath"),
                              DriverHead("duct", "duct_head")])
    T = torch.tensor([268.0, 294.0, 296.0], requires_grad=True)

    def f(T_):
        head = heads.head({"rho": 101325.0 / (287.05 * T_), "duct_head": torch.tensor([3.0])})
        phi, q = layer.solve(PHI_B, {"air.edge_head": head}, differentiable=True,
                             atol=1e-12, rtol=1e-12)
        return torch.cat([phi[1:], q])

    assert torch.autograd.gradcheck(f, (T,), eps=1e-5, atol=1e-5, rtol=1e-4)


def test_gradcheck_through_a_state_dependent_stack_head_mna():
    layer = _rc()
    heads = EdgeHeads(layer, [Stack("r", src=[1], tgt=[2], z_path=[0.5],
                                    z_ref=[0.0, 0.0, 3.0])])
    T = torch.tensor([0.9, 1.1, 1.3], requires_grad=True)
    hist = (torch.zeros(3), torch.zeros(3))

    def f(T_):
        head = heads.head({"rho": 1.0 / T_})
        phi, q = layer.solve(PHI_B, {"e.edge_head": head}, dt=0.1, scheme="backward_euler",
                             history=hist, differentiable=True, atol=1e-12, rtol=1e-12)
        return torch.cat([phi[1:], q])

    assert torch.autograd.gradcheck(f, (T,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_power_residual_covers_the_head():
    net = _net()
    layer = _air(net)
    drivers = _drivers()
    head = EdgeHeads(layer, _terms(net)).head(drivers)
    full = {**drivers, "air.edge_head": head}
    phi, q = layer.solve(PHI_B, full, differentiable=False)
    assert abs(layer.power_residual(phi, q, full).item()) < 1e-10
    # the test bites: the head does work on the flows, and the identity subtracts it
    assert abs((head * q).sum().item()) > 1e-6

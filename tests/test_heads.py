"""EdgeHeads: the driver "<layer>.head" a closure writes and a Kirchhoff layer adds to dp."""

from __future__ import annotations

import pytest
import torch

from noodl.drives import ConstantDrive, DuctStack, Stack, Wind
from noodl.elements import Conductance, PowerLaw
from noodl.heads import EdgeHeads
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


def _pair(net, terms):
    """The same layer built with drives=terms and without them (for the head driver)."""
    old = KirchhoffLayer(net, "air", _laws(), drives=terms, boundary=["ambient"])
    new = KirchhoffLayer(net, "air", _laws(), boundary=["ambient"])
    return old, new


def _terms(net):
    return [Stack.from_network(net, "airpath"), ConstantDrive("duct", "duct_head")]


def _drivers():
    return {"rho": torch.tensor([1.2, 1.15, 1.1]), "duct_head": torch.tensor([3.0])}


PHI_B = torch.zeros(1, dtype=F64)


def test_dp_with_one_term_per_kind_is_bitwise_the_drive_version():
    net = _net()
    terms = _terms(net)
    old, new = _pair(net, terms)
    drivers = _drivers()
    head = EdgeHeads(new, terms).head(drivers)
    assert head.shape == (5,)
    phi = torch.tensor([0.0, 4.0, -2.0])
    assert torch.equal(old.dp(phi, drivers), new.dp(phi, {**drivers, "air.head": head}))


@pytest.mark.parametrize("differentiable", [False, True])
def test_solve_with_one_term_per_kind_is_bitwise_the_drive_version(differentiable):
    net = _net()
    terms = _terms(net)
    old, new = _pair(net, terms)
    drivers = _drivers()
    head = EdgeHeads(new, terms).head(drivers)
    a = old.solve(PHI_B, drivers, differentiable=differentiable)
    b = new.solve(PHI_B, {"air.head": head}, differentiable=differentiable)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_two_terms_on_one_kind_agree_to_rounding():
    net = _net()
    terms = [Stack.from_network(net, "airpath"), ConstantDrive("airpath", "gust")]
    old, new = _pair(net, terms)
    drivers = {"rho": torch.tensor([1.2, 1.15, 1.1]),
               "gust": torch.tensor([1.0, -2.0, 0.5, 3.0])}
    head = EdgeHeads(new, terms).head(drivers)
    a = old.solve(PHI_B, drivers, differentiable=False)
    b = new.solve(PHI_B, {"air.head": head}, differentiable=False)
    torch.testing.assert_close(b[0], a[0], rtol=1e-14, atol=0)
    torch.testing.assert_close(b[1], a[1], rtol=1e-14, atol=0)


def _rc(drives=()):
    # source a -> gnd, resistor a -> b (edge 1), capacitor b -> gnd (dynamic edge 2)
    net = Network(dtype=F64)
    for n in ("gnd", "a", "b"):
        net.add_node(n)
    for u, v, k in (("a", "gnd", "source"), ("a", "b", "r"), ("b", "gnd", "cap")):
        net.add_edge(u, v, kind=k)
    return KirchhoffLayer(
        net, "e", [Source(torch.tensor([1.0])), Conductance(torch.tensor([2.0]), kind="r"),
                   Cap(torch.tensor([1.0]))], drives=drives, boundary=["gnd"])


@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_mna_solve_with_a_head_is_bitwise_the_drive_version(scheme):
    term = ConstantDrive("r", "h")
    old, new = _rc([term]), _rc()
    drivers = {"h": torch.tensor([0.25])}
    head = EdgeHeads(new, [term]).head(drivers)
    assert head.tolist() == [0.0, 0.25, 0.0]
    hist = (torch.zeros(3), torch.zeros(3))
    for kw in ({"mode": "dc"}, {"dt": 0.1, "scheme": scheme, "history": hist}):
        a = old.solve(PHI_B, drivers, differentiable=False, **kw)
        b = new.solve(PHI_B, {"e.head": head}, differentiable=False, **kw)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_a_head_on_a_dynamic_edge_is_refused_by_name():
    layer = _rc()
    hist = (torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match=r"'e'.*'e.head'.*not zero.*cap"):
        layer.solve(PHI_B, {"e.head": torch.tensor([0.0, 0.0, 0.3])}, dt=0.1, history=hist,
                    differentiable=False)
    # a zero entry on the dynamic edge is fine
    layer.solve(PHI_B, {"e.head": torch.tensor([0.0, 0.3, 0.0])}, dt=0.1, history=hist,
                differentiable=False)


def test_gradcheck_through_a_closure_only_parameter():
    net = _net()
    layer = KirchhoffLayer(net, "air", _laws(), boundary=["ambient"])
    heads = EdgeHeads(layer, [ConstantDrive("airpath", "h"), ConstantDrive("duct", "d")])
    base = torch.tensor([2.0, -1.0, 3.0, 1.5])
    k = torch.tensor([1.0, 0.5], dtype=F64, requires_grad=True)   # reaches dp only via head

    def f(k_):
        head = heads.head({"h": k_[0] * base, "d": k_[1:2] * 4.0})
        phi, q = layer.solve(PHI_B, {"air.head": head}, differentiable=True,
                             atol=1e-12, rtol=1e-12)
        return torch.cat([phi[1:], q])

    assert torch.autograd.gradcheck(f, (k,), eps=1e-6, atol=1e-5)


def test_refusals_name_the_layer_and_the_key():
    net = _net()
    _, layer = _pair(net, [])
    with pytest.raises(ValueError, match=r"'air'.*'air.head'.*\(5,.*width 4"):
        layer.dp(torch.zeros(3), {"air.head": torch.zeros(4)})
    with pytest.raises(ValueError, match=r"term kind 'nope'.*'air'"):
        EdgeHeads(layer, [ConstantDrive("nope", "h")])
    heads = EdgeHeads(layer, [ConstantDrive("airpath", "h")])
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
        EdgeHeads(layer, [ConstantDrive("cap", "h")])


def test_inputs_are_the_union_of_the_terms_keys_or_unknown():
    net = _net()
    _, layer = _pair(net, [])
    heads = EdgeHeads(layer, [Stack.from_network(net, "airpath", rho_key="rho"),
                              ConstantDrive("duct", "duct_head")])
    assert heads.inputs == ("duct_head", "rho") and heads.outputs == ("air.head",)
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


def _model(use_heads):
    net = _net()
    terms = _terms(net)
    if use_heads:
        layer = KirchhoffLayer(net, "air", _laws(), boundary=["ambient"],
                               quantity="pressure", unit="Pa")
        closures = [Density(), EdgeHeads(layer, terms)]
    else:
        layer = KirchhoffLayer(net, "air", _laws(), drives=terms, boundary=["ambient"],
                               quantity="pressure", unit="Pa")
        closures = [Density()]
    return Model(net, {"air": layer}, closures=closures), closures


def test_model_with_a_head_closure_equals_the_drive_version_over_three_steps():
    drivers = {"air.phi_boundary": PHI_B, "T": torch.tensor([268.0, 294.0, 296.0]),
               "duct_head": torch.tensor([3.0])}
    (a, _), (b, _) = _model(False), _model(True)
    sa = sb = {}
    for _ in range(3):
        sa, sb = a.step(sa, drivers, 60.0), b.step(sb, drivers, 60.0)
        assert set(sa) == set(sb)
        for key in sa:
            assert torch.equal(sa[key], sb[key]), key


def test_model_to_moves_the_term_tensors():
    model, closures = _model(True)
    stack = closures[1].terms[0]
    assert stack.z_path.dtype == torch.float64
    model.to(dtype=torch.float32)
    assert stack.z_path.dtype == torch.float32 and stack.z_ref.dtype == torch.float32

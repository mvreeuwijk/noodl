"""`noodl._model_setup`: the construction checks of `Model`, called directly."""

from types import SimpleNamespace

import pytest
import torch

from noodl._model_setup import (
    CouplingOptions,
    flow_owners,
    register_layers,
    validate_modes,
    validate_reactions,
    validate_substeps,
)
from noodl.elements import PowerLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.reaction import FirstOrderDecay
from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64


def _net():
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name, z_ref=0.0)
    net.add_edge("ambient", "z1", kind="airpath", z_path=0.0)
    net.add_edge("z1", "z2", kind="airpath", z_path=0.0)
    net.add_edge("z2", "ambient", kind="airpath", z_path=0.0)
    return net


def _air(net, name="air"):
    return KirchhoffLayer(net, name, [PowerLaw(torch.full((3,), 0.01, dtype=F64), 0.65)],
                          boundary=["ambient"])


def _species(net, name="species"):
    return TransportLayer(net, name, capacity=torch.tensor([50.0, 80.0], dtype=F64),
                          flow_kind="airpath", boundary=["ambient"], scheme="implicit")


def _options(**kw):
    base = dict(coupling="pingpong", iterate_tol=None, iterate_max=20, iterate_relaxation=0.5,
                iterate_acceleration=None, iterate_memory=5, iterate_on_limit="raise",
                adjoint_rtol=1e-10, iterate_flow_tol=None,
                kirchhoff={"air": None}, transport={"species": None})
    base.update(kw)
    return CouplingOptions.validate(**base)


def test_validate_modes():
    validate_modes("pingpong", "after_transport")
    validate_modes("iterate", "before_transport")
    with pytest.raises(ValueError, match="coupling must be 'pingpong' or 'iterate', got 'x'"):
        validate_modes("x", "after_transport")
    with pytest.raises(ValueError, match="reaction_order must be"):
        validate_modes("pingpong", "sideways")


def test_register_layers_splits_by_kind_in_order():
    net = _net()
    air, species = _air(net), _species(net)
    reg = register_layers(net, [species, air])
    assert list(reg.layers) == ["species", "air"]
    assert reg.kirchhoff == {"air": air} and reg.transport == {"species": species}
    assert reg.allocation == {}
    assert register_layers(net, {"air": air}).layers == {"air": air}


def test_register_layers_refusals():
    net = _net()
    air = _air(net)
    with pytest.raises(ValueError, match="two layers named 'air'"):
        register_layers(net, [air, _air(net)])
    with pytest.raises(ValueError, match="registered as 'wind' but is named 'air'"):
        register_layers(net, {"wind": air})
    with pytest.raises(TypeError, match="layer 'oops' is a str, not a KirchhoffLayer"):
        register_layers(net, {"oops": "oops"})
    with pytest.raises(ValueError, match="different Network"):
        register_layers(_net(), [air])


def test_flow_owners():
    net = _net()
    air, species = _air(net), _species(net)
    owner, driver = flow_owners({"air": air}, {"species": species}, {})
    assert owner == {"species": "air"} and driver == {"species": "species.q"}
    owner, _ = flow_owners({}, {"species": species}, {})
    assert owner == {"species": None}
    with pytest.raises(ValueError, match="2 Kirchhoff layers of this model provide"):
        flow_owners({"air": air, "air2": _air(net, "air2")}, {"species": species}, {})


def test_validate_reactions():
    decay = FirstOrderDecay(torch.tensor(1e-3, dtype=F64))
    assert validate_reactions({"species": None}, (("species", decay),)) == [("species", decay)]
    with pytest.raises(KeyError, match="reaction targets 'nope'"):
        validate_reactions({"species": None}, [("nope", decay)])
    with pytest.raises(TypeError, match="reaction for layer 'species' is a str"):
        validate_reactions({"species": None}, [("species", "decay")])


def test_validate_substeps():
    assert validate_substeps({"species": None}, None) == {"species": 1}
    assert validate_substeps({"species": None}, {"species": 3.0}) == {"species": 3}
    with pytest.raises(KeyError, match="substeps names 'nope'"):
        validate_substeps({"species": None}, {"nope": 2})
    with pytest.raises(TypeError, match=r"substeps\['species'\] must be an integer"):
        validate_substeps({"species": None}, {"species": "many"})
    with pytest.raises(ValueError, match=r"substeps\['species'\] must be >= 1"):
        validate_substeps({"species": None}, {"species": 0})


def test_coupling_options_values():
    opt = _options(coupling="iterate", iterate_tol={"species": 1e-9, "air": 1e-6},
                   iterate_max=7.0, iterate_relaxation=1, iterate_flow_tol={"air": 1})
    assert opt.iterate_tol == {"species": 1e-9, "air": 1e-6}
    assert opt.iterate_max == 7 and isinstance(opt.iterate_max, int)
    assert opt.iterate_relaxation == 1.0 and isinstance(opt.iterate_relaxation, float)
    assert opt.iterate_flow_tol == {"air": 1.0}
    assert _options().iterate_tol == {} and _options().iterate_flow_tol == {}


@pytest.mark.parametrize("kw, error, match", [
    ({"iterate_relaxation": 0.0}, ValueError, r"iterate_relaxation must lie in \(0, 1\]"),
    ({"iterate_relaxation": 1.5}, ValueError, "iterate_relaxation"),
    ({"iterate_acceleration": "broyden"}, ValueError, "iterate_acceleration must be None"),
    ({"iterate_memory": 0}, ValueError, "iterate_memory must be an integer >= 1"),
    ({"iterate_memory": True}, ValueError, "iterate_memory"),
    ({"iterate_on_limit": "ignore"}, ValueError, "iterate_on_limit must be 'raise'"),
    ({"adjoint_rtol": 0.0}, ValueError, "adjoint_rtol must be positive"),
    ({"coupling": "iterate"}, ValueError, "coupling='iterate' requires iterate_tol"),
    ({"coupling": "iterate", "iterate_tol": {"nope": 1.0}}, KeyError, "'nope'"),
    ({"coupling": "iterate", "iterate_tol": {"air": 1.0}, "iterate_max": 1}, ValueError,
     "iterate_max >= 2, got 1"),
    ({"iterate_tol": {"air": 1.0}, "iterate_flow_tol": {"nope": 1e-9}}, KeyError,
     "iterate_flow_tol names"),
    ({"iterate_tol": {"air": 1.0}, "iterate_flow_tol": {"air": 0.0}}, ValueError,
     "iterate_flow_tol must be positive"),
])
def test_coupling_options_refusals(kw, error, match):
    with pytest.raises(error, match=match):
        _options(**kw)


def test_check_acceleration_refusals():
    kirchhoff = {"air": SimpleNamespace(phasor=False), "ac": SimpleNamespace(phasor=True)}
    ok = _options(coupling="iterate", iterate_tol={"air": 1.0}, kirchhoff=kirchhoff,
                  iterate_acceleration="anderson", iterate_flow_tol={"air": 1.0})
    ok.check_acceleration(kirchhoff)
    _options().check_acceleration(kirchhoff)                      # no acceleration
    with pytest.raises(ValueError, match="needs coupling='iterate', got coupling='pingpong'"):
        _options(iterate_acceleration="anderson").check_acceleration(kirchhoff)
    with pytest.raises(ValueError, match="iterate_tol names none"):
        _options(coupling="iterate", iterate_tol={"species": 1.0}, kirchhoff=kirchhoff,
                 iterate_acceleration="anderson").check_acceleration(kirchhoff)
    with pytest.raises(ValueError, match=r"phasor layer\(s\) \['ac'\]"):
        _options(coupling="iterate", iterate_tol={"ac": 1.0}, kirchhoff=kirchhoff,
                 iterate_acceleration="anderson").check_acceleration(kirchhoff)
    with pytest.raises(ValueError, match=r"iterate_flow_tol for the .* \['air'\]"):
        _options(coupling="iterate", iterate_tol={"air": 1.0}, kirchhoff=kirchhoff,
                 iterate_acceleration="anderson").check_acceleration(kirchhoff)

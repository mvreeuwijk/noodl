"""The mutation, device and serialization contract stated in docs/concepts/ownership.md.

Each test pins one sentence of that page. What the page calls unsupported is not tested here
(a test of it would pin a limitation as if it were a promise).
"""

from __future__ import annotations

import copy
import dataclasses
import io
import pickle

import pytest
import torch

from noodl.apps.building_physics.elements import orifice_law_from_edges
from noodl.apps.electrical.circuit import Circuit, Resistor
from noodl.apps.sewer.network import Manhole, Outfall, Pipe, SewerNetwork
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network
from tests.test_model import _build

F64 = torch.float64


def _step(model, state, drivers):
    return model.step(state, drivers, 10.0)


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ---- Network: growing it after layers exist is refused ----------------------------------


def test_growing_the_network_makes_the_model_refuse_to_step():
    net, model, state, drivers, _, _ = _build()
    _step(model, state, drivers)
    net.add_node("extra")
    with pytest.raises(RuntimeError, match="topology.*changed"):
        _step(model, state, drivers)


def test_adding_an_edge_bumps_the_version_and_re_adding_a_node_does_not():
    net = Network(dtype=F64)
    net.add_node("a")
    net.add_node("b")
    v = net.topology_version
    net.add_node("a", volume=3.0)
    assert net.topology_version == v
    net.add_edge("a", "b", kind="airpath")
    assert net.topology_version == v + 1


def test_layers_built_after_the_network_is_complete_are_current():
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("z", "ambient", kind="airpath", Cd=0.6, area=0.01)
    law = orifice_law_from_edges(net, "airpath")
    layer = KirchhoffLayer(net, "air", [law], boundary=["ambient"])
    layer._require_current_topology()  # does not raise


# ---- Network attributes are copied at construction --------------------------------------


def test_edge_attributes_are_read_when_the_law_is_built_not_afterwards():
    net = Network(dtype=F64)
    net.add_node("ambient")
    net.add_node("z")
    net.add_edge("z", "ambient", kind="airpath", Cd=0.6, area=0.01)
    law = orifice_law_from_edges(net, "airpath")
    before = law.C.detach().clone()
    net.graph["z"]["ambient"][0]["area"] = 1.0  # attribute edit: no version bump
    assert torch.equal(law.C.detach(), before)
    assert not torch.equal(orifice_law_from_edges(net, "airpath").C.detach(), before)


# ---- Laws are live modules: parameters may change between steps -------------------------


def test_an_in_place_parameter_update_is_seen_by_the_next_step():
    _, model, state, drivers, law, _ = _build(learnable=True)
    first = _step(model, state, drivers)
    with torch.no_grad():
        law.C.mul_(2.0)
    assert not _same(first, _step(model, state, drivers))


def test_learnable_parameters_are_registered_on_the_law():
    _, _, _, _, law, _ = _build(learnable=True)
    assert {n for n, p in law.named_parameters() if p.requires_grad} == {"C", "n"}
    assert set(law.state_dict()) == {"C", "n"}


# ---- Network.to ---------------------------------------------------------------------------


def test_network_to_sets_dtype_and_device_for_operators_derived_afterwards():
    net = Network(dtype=F64)
    net.add_node("a")
    net.add_node("b")
    net.add_edge("a", "b", kind="airpath")
    assert net.incidence().dtype == F64
    assert net.to(dtype=torch.float32) is net
    assert net.incidence().dtype == torch.float32
    assert net.to(device="cpu").device == torch.device("cpu")



# ---- Copying and serialization -----------------------------------------------------------


def test_deepcopy_of_a_model_is_independent_and_keeps_its_topology_guard():
    net, model, state, drivers, law, _ = _build(learnable=True)
    twin = copy.deepcopy(model)
    assert twin.net is not net and twin.layers["air"].net is twin.net
    assert twin.layers["air"]._laws[0] is not law
    assert _same(_step(twin, state, drivers), _step(model, state, drivers))
    net.add_node("extra")
    with pytest.raises(RuntimeError, match="topology.*changed"):
        _step(model, state, drivers)
    _step(twin, state, drivers)  # the copy has its own network and is still current


def test_pickle_round_trip_of_a_model_with_built_in_parts_reproduces_the_step():
    _, model, state, drivers, _, _ = _build(learnable=True)
    clone = pickle.loads(pickle.dumps(model))
    assert _same(_step(clone, state, drivers), _step(model, state, drivers))
    buf = io.BytesIO()
    torch.save(model, buf)  # torch.save is pickle underneath


def _module_level_closure(state, drivers):
    return {}


def test_a_closure_pickles_exactly_when_the_callable_does():
    _, picklable, *_ = _build(closures=(_module_level_closure,))
    pickle.dumps(picklable)
    _, lam, *_ = _build(closures=(lambda state, drivers: {},))
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(lam)


def test_attributes_attached_to_a_model_travel_with_copy_and_pickle():
    _, model, *_ = _build(learnable=True)
    model.notes = {"routing": "demo"}
    assert copy.deepcopy(model).notes == {"routing": "demo"}
    assert pickle.loads(pickle.dumps(model)).notes == {"routing": "demo"}


def test_state_dict_holds_tensors_only_not_the_law_kind():
    _, _, _, _, law, _ = _build(learnable=True)
    restored = copy.deepcopy(law)
    with torch.no_grad():
        restored.C.zero_()
    restored.load_state_dict(law.state_dict())
    assert torch.equal(restored.C, law.C)
    assert all(isinstance(v, torch.Tensor) for v in law.state_dict().values())


# ---- Frozen is shallow --------------------------------------------------------------------


def test_circuit_devices_are_frozen_and_held_in_a_tuple():
    device = Resistor("R1", "a", "0", 10.0)
    circuit = Circuit([device])
    assert isinstance(circuit.devices, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        device.R = 5.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        circuit.ground = "x"


def test_frozen_sewer_network_still_owns_a_mutable_notes_dict():
    net = SewerNetwork(
        manholes=(Manhole("J1", 10.0),),
        pipes=(Pipe("C1", "J1", "Out", 100.0, 0.45, 0.013, 0.005),),
        outfalls=(Outfall("Out", 9.0),),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        net.routing = "steady"
    net.notes["added"] = "later"  # frozen is shallow: the dict itself can be edited
    assert net.notes == {"added": "later"}


def test_deepcopy_keeps_a_shared_notes_dict_shared():
    _, model, *_ = _build()
    shared = {"routing": "demo"}
    model.notes = shared
    model.layers["air"].node_hints = shared
    twin = copy.deepcopy(model)
    assert twin.notes is twin.layers["air"].node_hints and twin.notes is not shared

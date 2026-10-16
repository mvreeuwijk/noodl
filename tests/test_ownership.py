"""The mutation, device and serialization contract stated in docs/concepts/ownership.md.

Each test pins one sentence of that page. What the page calls unsupported is not tested here
(a test of it would pin a limitation as if it were a promise).
"""

from __future__ import annotations

import copy
import dataclasses
import io
import pickle
from pathlib import Path

import networkx as nx
import pytest
import torch

from noodl.apps.building_physics.elements import orifice_law_from_edges
from noodl.apps.electrical.circuit import Circuit, Resistor
from noodl.apps.sewer.network import Manhole, Outfall, Pipe, SewerNetwork
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network
from tests.test_model import _build

F64 = torch.float64
F32 = torch.float32
CONTAM = Path(__file__).resolve().parent / "data" / "contam"


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


def test_network_to_after_the_layers_exist_makes_them_refuse():
    net, model, state, drivers, _, _ = _build()
    net.to(dtype=F32)
    with pytest.raises(RuntimeError, match=r"moved to device.*Model\.to"):
        _step(model, state, drivers)
    # Model.to repairs it: the network is where it already is, the layers follow it.
    assert model.to(dtype=F32) is model
    assert _step(model, _cast(state), _cast(drivers))["air.q"].dtype == F32


# ---- Model.to: the whole model moves -------------------------------------------------------


def _cast(values):
    """`values` with every floating tensor in float32 and every complex one in complex64."""
    out = {}
    for k, v in values.items():
        if isinstance(v, torch.Tensor) and v.is_complex():
            v = v.to(torch.complex64)
        elif isinstance(v, torch.Tensor) and v.is_floating_point():
            v = v.to(F32)
        out[k] = v
    return out


def _assert_float32_matches(out, ref, rtol, atol):
    assert out.keys() == ref.keys()
    for k, v in ref.items():
        if not (v.is_floating_point() or v.is_complex()):
            continue
        assert out[k].dtype == (torch.complex64 if v.is_complex() else F32), k
        torch.testing.assert_close(out[k].to(v.dtype), v, rtol=rtol, atol=atol, msg=k)


def _held_dtypes(model):
    """The dtypes of every floating or complex tensor the model's layers and laws hold."""
    found = set()
    for layer in model.layers.values():
        for v in vars(layer).values():
            if isinstance(v, torch.Tensor) and (v.is_floating_point() or v.is_complex()):
                found.add(v.dtype)
        for law in getattr(layer, "_laws", []):
            found.update(t.dtype for t in law.state_dict().values()
                         if t.is_floating_point() or t.is_complex())
    return found


def test_model_to_float32_moves_the_network_layers_and_laws():
    net, model, state, drivers, law, species = _build(learnable=True)
    ref = _step(model, state, drivers)
    C = law.C
    assert model.to(dtype=F32) is model
    assert net.dtype == F32 and net.incidence().dtype == F32
    assert _held_dtypes(model) == {F32}
    assert law.C is C and law.C.dtype == F32  # a parameter moves in place
    out = model.step(_cast(state), _cast(drivers), 10.0, atol=1e-6, rtol=1e-6)
    _assert_float32_matches(out, ref, rtol=1e-5, atol=1e-7)
    # And back: the same model in float64 gives the float64 result again.
    model.to(dtype=F64)
    assert _held_dtypes(model) == {F64}
    torch.testing.assert_close(_step(model, state, drivers)["air.q"], ref["air.q"])


def test_a_float32_building_model_matches_float64():
    from noodl.apps.building_physics.prj import project_to_model, read_prj

    model, state, drivers = project_to_model(read_prj(CONTAM / "duct_three_zone.prj"))
    ref, out = state, _cast(state)
    for _ in range(3):
        ref = model.step(ref, drivers, 300.0)
    model.to(dtype=F32)
    for _ in range(3):
        out = model.step(out, _cast(drivers), 300.0, atol=1e-5, rtol=1e-5)
    _assert_float32_matches(out, ref, rtol=1e-4, atol=1e-5)


def test_a_float32_water_model_matches_float64():
    from noodl.apps.water.network import build_model, twoloop, water_steady

    model, state, drivers = build_model(twoloop())
    ref = water_steady(model, state, drivers)
    model.to(dtype=F32)
    out = water_steady(model, _cast(state), _cast(drivers), atol=1e-5, rtol=1e-5)
    _assert_float32_matches(out, ref, rtol=1e-5, atol=1e-6)


def _divider():
    from noodl.apps.electrical.circuit import Circuit, Resistor, VoltageSource

    return Circuit([VoltageSource("V1", "a", "0", 10.0), Resistor("R1", "a", "b", 2.0),
                    Resistor("R2", "b", "0", 3.0)])


def _rlc():
    from noodl.apps.electrical.circuit import (
        Capacitor,
        Circuit,
        Inductor,
        Resistor,
        VoltageSource,
    )

    return Circuit([VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", 2.0),
                    Inductor("L1", "b", "c", 1e-3), Capacitor("C1", "c", "0", 0.5)])


@pytest.mark.parametrize("phasor", [False, True], ids=["dc", "phasor"])
def test_a_float32_electrical_model_matches_float64(phasor):
    from noodl.apps.electrical.analysis import operating_point
    from noodl.apps.electrical.circuit import build_model

    if phasor:
        model, _, drivers = build_model(_rlc(), phasor=True,
                                        frequency=torch.tensor([0.1, 1.0, 50.0]))
    else:
        model, _, drivers = build_model(_divider())
    ref = operating_point(model, drivers)
    model.to(dtype=F32)
    out = operating_point(model, _cast(drivers), atol=1e-6, rtol=1e-6)
    _assert_float32_matches(out, ref, rtol=1e-5, atol=1e-6)
    if phasor:  # complex stays complex, at the new precision
        assert ref["electrical.phi"].dtype == torch.complex128
        assert out["electrical.phi"].dtype == torch.complex64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_a_cuda_move_gives_the_cpu_result():
    _, model, state, drivers, _, _ = _build()
    ref = _step(model, state, drivers)
    model.to(device="cuda")
    out = _step(model, {k: v.cuda() for k, v in state.items()},
                {k: v.cuda() for k, v in drivers.items()})
    for k, v in ref.items():
        assert out[k].device.type == "cuda"
        torch.testing.assert_close(out[k].cpu(), v)


class _Offset:
    """A user closure holding a tensor, with the `to` hook `Model.to` calls."""

    def __init__(self):
        self.extra = torch.tensor([0.0, 1e-6, 0.0], dtype=F64)
        self.moved_to = None

    def __call__(self, state, drivers):
        return {"species.sources": drivers["species.sources"] + self.extra}

    def to(self, device=None, dtype=None):
        self.moved_to = dtype
        self.extra = self.extra.to(device=device, dtype=dtype)
        return self


def test_model_to_calls_a_user_closures_to_hook_and_moves_attached_tensors():
    hooked = _Offset()
    _, model, state, drivers, _, _ = _build(closures=(hooked,))
    model.junction_idx = torch.tensor([0, 1])
    model.initial = {"x": torch.zeros(2, dtype=F64)}
    model.to(dtype=F32)
    assert hooked.moved_to == F32 and hooked.extra.dtype == F32
    assert model.junction_idx.dtype == torch.long      # an index keeps its dtype
    assert model.initial["x"].dtype == F32
    assert _step(model, _cast(state), _cast(drivers))["species.x"].dtype == F32


class _Flow:
    """A flow closure holding its flows, with a `to` hook."""

    def __init__(self, name, n):
        self.key = f"{name}.q"
        self.q = torch.full((n,), -0.5, dtype=F64)

    def __call__(self, state, drivers):
        return {self.key: self.q}

    def to(self, device=None, dtype=None):
        self.q = self.q.to(device=device, dtype=dtype)
        return self


def _vented(name, n):
    from noodl.layers.transport import TransportLayer
    from noodl.model import Model

    net = Network(dtype=F64)
    net.add_node("out")
    for i in range(n):
        net.add_node(f"{name}{i}")
        net.add_edge(f"{name}{i}", "out", kind="vent")
    layer = TransportLayer(net, name, capacity=torch.full((n,), 5.0, dtype=F64),
                           flow_kind="vent", boundary=["out"])
    return (Model(net, {name: layer}, closures=[_Flow(name, n)]),
            {f"{name}.x": torch.ones(n, dtype=F64)},
            {f"{name}.x_boundary": torch.zeros(1, dtype=F64)})


def test_coupled_model_to_moves_every_model():
    from noodl.couple import ValueLink, union

    link = ValueLink(from_model="a", from_key="a.x", from_index=0,
                     to_model="b", to_key="b.x_boundary", to_index=0)
    coupled, state, drivers = union({"a": _vented("a", 2), "b": _vented("b", 1)},
                                    shared=[link])
    ref = coupled.step(state, drivers, dt=1.0)
    assert coupled.to(dtype=F32) is coupled
    assert all(m.net.dtype == F32 for m in coupled.models.values())
    out = coupled.step({t: _cast(s) for t, s in state.items()},
                       {t: _cast(d) for t, d in drivers.items()}, dt=1.0)
    for tag in ref:
        _assert_float32_matches(out[tag], ref[tag], rtol=1e-5, atol=1e-6)


# ---- Direct graph and attribute edits ----------------------------------------------------


def test_building_a_layer_freezes_the_graph_against_direct_structural_edits():
    net, model, state, drivers, _, _ = _build()
    assert net.frozen
    edits = (lambda g: g.add_node("x"), lambda g: g.add_edge("x", "ambient"),
             lambda g: g.remove_node("ambient"), lambda g: g.clear())
    for edit in edits:
        with pytest.raises(nx.NetworkXError, match="Network.add_node"):
            edit(net.graph)
    _step(model, state, drivers)  # nothing changed
    net.add_node("extra")         # the Network's own path still works ...
    assert net.frozen and "extra" in net.graph
    with pytest.raises(RuntimeError, match="topology.*changed"):
        _step(model, state, drivers)  # ... and the layers refuse the grown network


def test_a_network_without_layers_is_not_frozen_and_copies_stay_frozen():
    net = Network(dtype=F64)
    net.add_node("a")
    net.graph.add_node("b")  # allowed: no layer has read the graph yet
    assert not net.frozen
    _, model, *_ = _build()
    for twin in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert twin.net.frozen
        with pytest.raises(nx.NetworkXError):
            twin.net.graph.add_node("x")
        twin.net.add_node("x")


def test_an_attribute_edit_after_the_layers_is_detected_by_check_and_to():
    net, model, *_ = _build()
    assert not model.check().errors
    net.graph.nodes[net.nodes[1]]["volume"] = 123.0
    assert [i.code for i in model.check().errors] == ["attributes-changed"] * 2  # both layers
    with pytest.raises(RuntimeError, match="attribute.*edited"):
        model.to(dtype=F32)


def test_an_attribute_edit_through_add_node_is_detected_too():
    net, model, *_ = _build()
    net.add_node(net.nodes[1], volume=7.0)  # an existing node: not a topology change
    assert [i.code for i in model.check().errors] == ["attributes-changed"] * 2


def test_the_attribute_fingerprint_survives_copies_and_sees_tensor_values():
    net, model, *_ = _build()
    twin = pickle.loads(pickle.dumps(model))
    assert not twin.check().errors
    assert twin.net.attribute_fingerprint() == net.attribute_fingerprint()
    other = Network(dtype=F64)
    other.add_node("a", T=torch.tensor([1.0, 2.0]))
    before = other.attribute_fingerprint()
    other.graph.nodes["a"]["T"][0] = 3.0
    assert other.attribute_fingerprint() != before


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


class _ClockClosure:
    """An integrating closure (module level, so it pickles) that adds the step's dt."""

    integrates = True
    state_keys = ("clock",)

    def __call__(self, state, drivers, ctx=None):
        dt = 0.0 if ctx is None or ctx.dt is None else ctx.dt
        return {"clock": state["clock"] + dt}


@pytest.mark.parametrize("copier", [copy.deepcopy, lambda m: pickle.loads(pickle.dumps(m))],
                         ids=["deepcopy", "pickle"])
def test_a_copied_model_still_advances_an_integrating_closure(copier):
    _, model, state, drivers, _, _ = _build(closures=(_ClockClosure(),))
    state = {**state, "clock": torch.zeros((), dtype=F64)}
    clone = copier(model)
    assert model.step(state, drivers, 2.0)["clock"].item() == 2.0
    assert clone.step(state, drivers, 2.0)["clock"].item() == 2.0
    assert clone._integrating == {id(c) for c in clone.closures}

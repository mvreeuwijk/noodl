"""One step context for both solve paths: the same refusal from a layer whose every flow
follows from the potentials and from one that solves for flow unknowns."""

import pytest
import torch

from noodl import LinearStorage
from noodl.elements import Conductance
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network
from tests._circuit_laws import Source

F64 = torch.float64
GROUND = torch.zeros(1, dtype=F64)


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _layer(path: str, storage: bool) -> KirchhoffLayer:
    """gnd - a - b: a resistor a-b, and on the general path an ideal source a-gnd; with
    `storage`, a capacity at b."""
    net = Network(dtype=F64)
    for n in ("gnd", "a", "b"):
        net.add_node(n)
    laws = [Conductance(torch.tensor([1.0]), kind="r")]
    if path == "general":
        net.add_edge("a", "gnd", kind="source")
        laws.insert(0, Source(torch.tensor([1.0])))
    else:
        net.add_edge("a", "gnd", kind="g")
        laws.insert(0, Conductance(torch.tensor([1.0]), kind="g"))
    net.add_edge("a", "b", kind="r")
    stores = [LinearStorage([2], torch.tensor([1.0]))] if storage else []
    layer = KirchhoffLayer(net, "e", laws, boundary=["gnd"], storage=stores)
    assert layer.mna == (path == "general")
    return layer


def _histories(layer):
    n, b = layer._n_nodes, len(layer.cols)
    return torch.zeros(n, dtype=F64), torch.zeros(b, dtype=F64)


PATHS = ["fast", "general"]


def _solve(layer, **kwargs):
    return layer.solve(GROUND, differentiable=False, **kwargs)


@pytest.mark.parametrize("storage", [False, True])
@pytest.mark.parametrize("path", PATHS)
class TestEveryLayer:
    def test_unknown_mode_is_refused(self, path, storage):
        with pytest.raises(ValueError, match="mode='bogus'; expected 'dc', 'initial' or None"):
            _solve(_layer(path, storage), mode="bogus")

    def test_phasor_mode_needs_a_phasor_layer(self, path, storage):
        with pytest.raises(ValueError, match="needs a layer built with phasor=True"):
            _solve(_layer(path, storage), mode="phasor")

    @pytest.mark.parametrize("mode", ["dc", "initial"])
    def test_a_steady_mode_takes_no_dt(self, path, storage, mode):
        with pytest.raises(ValueError, match=f"mode='{mode}'.*takes no step; drop dt"):
            _solve(_layer(path, storage), mode=mode, dt=0.1)

    def test_storage_flow_needs_storage_or_is_refused_by_name(self, path, storage):
        layer = _layer(path, storage)
        flow = torch.zeros(1, dtype=F64)
        if storage:
            # a steady solve takes no step, so it takes no flow into storage either
            with pytest.raises(ValueError, match=r"\['storage_flow'\] given without dt"):
                _solve(layer, storage_flow=flow)
        else:
            with pytest.raises(ValueError, match="has no storage; drop storage_flow"):
                _solve(layer, storage_flow=flow)


class TestLayerThatTakesNoStep:
    """A layer with no storage, no dynamic law and no law that reads the context. Every
    implicit law reads the context, so only a layer whose flows all follow from the
    potentials is such a layer in practice."""

    @pytest.mark.parametrize("keyword", ["dt", "history"])
    def test_dt_and_history_are_refused(self, keyword):
        path = "fast"
        layer = _layer(path, storage=False)
        value = 0.1 if keyword == "dt" else _histories(layer)
        with pytest.raises(ValueError, match=rf"takes no step; drop \['{keyword}'\]"):
            _solve(layer, **{keyword: value})


@pytest.mark.parametrize("path", PATHS)
class TestLayerWithStorage:
    def test_dt_must_be_positive(self, path):
        layer = _layer(path, storage=True)
        for dt in (0.0, -1.0):
            with pytest.raises(ValueError, match="dt must be positive"):
                _solve(layer, dt=dt, history=_histories(layer))

    def test_a_step_needs_history(self, path):
        with pytest.raises(ValueError, match="needs history=.phi, q. at the step start"):
            _solve(_layer(path, storage=True), dt=0.1, scheme="backward_euler")

    def test_history_is_a_pair(self, path):
        layer = _layer(path, storage=True)
        with pytest.raises(ValueError, match="history must be the pair"):
            _solve(layer, dt=0.1, history=_histories(layer)[0], scheme="backward_euler")

    def test_history_alone_is_a_step_without_dt(self, path):
        layer = _layer(path, storage=True)
        with pytest.raises(ValueError, match=r"\['history'\] given without dt"):
            _solve(layer, history=_histories(layer))

    def test_the_trapezoidal_scheme_needs_the_flow_into_storage(self, path):
        layer = _layer(path, storage=True)
        with pytest.raises(ValueError, match="trapezoidal scheme needs storage_flow"):
            _solve(layer, dt=0.1, history=_histories(layer), scheme="trapezoidal")

    def test_storage_flow_has_one_value_per_entry(self, path):
        layer = _layer(path, storage=True)
        with pytest.raises(ValueError, match="one flow per storage entry"):
            _solve(layer, dt=0.1, history=_histories(layer), scheme="trapezoidal",
                   storage_flow=torch.zeros(3, dtype=F64))

    def test_the_step_runs_with_every_keyword_right(self, path):
        layer = _layer(path, storage=True)
        phi, q = _solve(layer, dt=0.1, history=_histories(layer), scheme="trapezoidal",
                        storage_flow=torch.zeros(1, dtype=F64))
        assert phi.shape[-1] == layer._n_nodes and q.shape[-1] == len(layer.cols)

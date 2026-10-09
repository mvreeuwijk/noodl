"""`Model.check`: configuration, missing inputs, typos, custom closure keys, orderings."""

from __future__ import annotations

import pytest
import torch

from benchmarks.natural_ventilation import build as build_ventilation
from noodl.apps.building_physics import initial_drivers, initial_state
from noodl.model import Model

F64 = torch.float64


def _setup():
    net, model, _, _ = build_ventilation("pingpong")
    th = model.refs.thermal
    drivers = initial_drivers(model, values={th.x_boundary: {"ambient": 285.0},
                                             th.sources: {"A": 1000.0}})
    return net, model, initial_state(model), drivers


def _codes(report):
    return {(i.code, i.key) for i in report.issues}


def test_a_complete_setup_is_clean_with_and_without_the_probe():
    _, model, state, drivers = _setup()
    assert model.check(state, drivers).ok
    report = model.check(state, drivers, dt=600.0, probe=True)
    assert report.ok and not report.warnings, str(report)
    assert {"rho", "rho_amb"} <= report.closure_outputs
    assert "thermal.sources" in report.driver_reads


def test_the_probe_leaves_the_inputs_untouched():
    _, model, state, drivers = _setup()
    before = {k: v.clone() for k, v in {**state, **drivers}.items()}
    model.check(state, drivers, dt=600.0, probe=True)
    for k, v in {**state, **drivers}.items():
        assert torch.equal(v, before[k])


def test_a_misspelt_optional_source_is_reported_not_silently_dropped():
    _, model, state, drivers = _setup()
    drivers["thermal.source"] = drivers.pop("thermal.sources")
    report = model.check(state, drivers)
    (issue,) = [i for i in report.issues if i.key == "thermal.source"]
    assert issue.code == "unknown-layer-key" and "did you mean 'thermal.sources'" in issue.message
    with pytest.raises(ValueError, match="thermal.source"):
        report.raise_for_errors(strict=True)


def test_a_misspelt_layer_prefix_is_reported():
    _, model, state, drivers = _setup()
    drivers["therml.sources"] = drivers.pop("thermal.sources")
    report = model.check(state, drivers)
    assert ("unknown-layer-prefix", "therml.sources") in _codes(report)


def test_custom_closure_keys_are_accepted_and_unread_ones_flagged_by_the_probe():
    net, base, _, _ = build_ventilation("pingpong")

    class Gain:
        """A custom closure: reads its own driver 'gain_W' and writes the thermal sources."""

        def __call__(self, state, drivers):
            s = torch.zeros(net.n, dtype=F64)
            s[net.node_index("A")] = drivers["gain_W"]
            return {"thermal.sources": s}

    model = Model(net, base.layers, closures=[*base.closures, Gain()])
    state = initial_state(model)
    drivers = initial_drivers(model, values={"thermal.x_boundary": {"ambient": 285.0}})
    drivers["gain_W"] = torch.tensor(800.0, dtype=F64)
    drivers["note"] = torch.tensor(1.0, dtype=F64)
    static = model.check(state, drivers)
    assert static.ok and not static.warnings, str(static)   # custom keys are not rejected
    probed = model.check(state, drivers, dt=600.0, probe=True)
    assert probed.ok
    assert ("unused-key", "note") in _codes(probed)
    assert ("unused-key", "gain_W") not in _codes(probed)
    assert "thermal.sources" in probed.closure_outputs


def test_a_missing_required_input_and_a_failing_probe_are_errors():
    _, model, state, drivers = _setup()
    del drivers["thermal.x_boundary"]
    report = model.check(state, drivers)
    assert ("missing-input", "thermal.x_boundary") in _codes(report)
    assert not report.ok
    probed = model.check(state, drivers, dt=600.0, probe=True)
    assert any(i.code == "probe-failed" for i in probed.errors)


def test_a_missing_transport_state_is_an_error_unless_steady():
    _, model, state, drivers = _setup()
    assert ("missing-input", "thermal.x") in _codes(model.check({}, drivers))
    assert model.check({}, drivers, steady=True).ok


def test_wrong_shapes_and_batches_are_reported():
    net, model, state, drivers = _setup()
    drivers["thermal.sources"] = torch.zeros(net.n - 1, dtype=F64)
    assert ("shape", "thermal.sources") in _codes(model.check(state, drivers))
    _, model, state, drivers = _setup()
    drivers["thermal.sources"] = drivers["thermal.sources"].expand(3, -1)
    drivers["thermal.x_boundary"] = drivers["thermal.x_boundary"].expand(2, -1)
    assert any(i.code == "batch-shape" for i in model.check(state, drivers).errors)


def test_a_source_on_a_boundary_node_of_a_potential_layer_is_an_error():
    """`PotentialFlowLayer` drops boundary rows of `sources` without a word; the check
    names them."""
    net, model, state, drivers = _setup()
    s = torch.zeros(net.n, dtype=F64)
    s[net.node_index("ambient")] = 0.1
    drivers["air.sources"] = s
    assert ("value-on-fixed-node", "air.sources") in _codes(model.check(state, drivers))


def test_a_state_key_given_as_a_driver_is_an_error():
    _, model, state, drivers = _setup()
    drivers["thermal.x"] = state["thermal.x"]
    assert ("wrong-dictionary", "thermal.x") in _codes(model.check(state, drivers))


def test_an_edge_kind_no_layer_uses_is_reported_with_a_suggestion():
    net, model, state, drivers = _setup()
    net.add_edge("A", "ambient", kind="airpth", z_path=1.0, Cd=0.6, area=0.01)
    report = model.check(state, drivers)
    (issue,) = [i for i in report.issues if i.code == "unused-edge-kind"]
    assert "'airpth'" in issue.message and "'airpath'" in issue.message


def test_the_report_prints_a_table_of_every_known_key():
    _, model, state, drivers = _setup()
    text = str(model.check(state, drivers))
    for key in ("air.phi_boundary", "thermal.x", "thermal.sources", "boundary order"):
        assert key in text


# ---------------------------------------------------------- second review regressions
def test_a_known_input_that_is_not_a_tensor_is_an_error():
    _, model, state, drivers = _setup()
    drivers["air.phi_boundary"] = [0.0]
    report = model.check(state, drivers)
    assert ("type", "air.phi_boundary") in _codes(report) and not report.ok


def test_a_closure_output_of_the_wrong_shape_is_an_error_naming_the_closure():
    net, base, _, _ = build_ventilation("pingpong")

    class BadSources:
        def __call__(self, state, drivers):
            return {"thermal.sources": torch.zeros(99, dtype=F64)}

    model = Model(net, base.layers, closures=[*base.closures, BadSources()])
    state = initial_state(model)
    drivers = initial_drivers(model, values={"thermal.x_boundary": {"ambient": 285.0}})
    report = model.check(state, drivers)
    (issue,) = [i for i in report.errors if i.key == "thermal.sources"]
    assert issue.code == "shape" and "written by BadSources" in issue.message
    with pytest.raises(ValueError):
        model.step(state, drivers, 600.0)


def test_a_closure_output_on_a_boundary_node_is_an_error():
    net, base, _, _ = build_ventilation("pingpong")

    class AmbientSource:
        def __call__(self, state, drivers):
            s = torch.zeros(net.n, dtype=F64)
            s[net.node_index("ambient")] = 1.0
            return {"thermal.sources": s}

    model = Model(net, base.layers, closures=[*base.closures, AmbientSource()])
    drivers = initial_drivers(model, values={"thermal.x_boundary": {"ambient": 285.0}})
    report = model.check(initial_state(model), drivers)
    assert ("value-on-fixed-node", "thermal.sources") in _codes(report)


def test_the_unused_edge_kind_report_counts_edges():
    net, model, state, drivers = _setup()
    for _ in range(3):
        net.add_edge("A", "ambient", kind="airpth", z_path=1.0, Cd=0.6, area=0.01)
    (issue,) = [i for i in model.check(state, drivers).issues if i.code == "unused-edge-kind"]
    assert "has 3 edge(s)" in issue.message

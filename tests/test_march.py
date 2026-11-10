"""`Model.march`: the time-march loop with backward-Euler restarts."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.electrical import (
    LAYER,
    Capacitor,
    Circuit,
    Resistor,
    VoltageSource,
    build_model,
    initial_state,
)
from noodl.elements import Conductance, PowerLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.topology import Network

F64 = torch.float64
PHI, Q = f"{LAYER}.phi", f"{LAYER}.q"


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _rc():
    model, _, drivers = build_model(Circuit([
        VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", 1.0),
        Capacitor("C1", "b", "0", 1.0)]))
    return model, initial_state(model, drivers), drivers


GRID = [0.0, 0.1, 0.25, 0.4, 0.7, 1.0]


def test_march_equals_hand_rolled_steps_bitwise():
    model, state, drivers = _rc()
    out = model.march(state, drivers, GRID, restarts={0, 3})
    cur, rows = state, [state]
    for k in range(len(GRID) - 1):
        rule = "backward_euler" if k in (0, 3) else "trapezoidal"
        cur = model.step(cur, drivers, dt=GRID[k + 1] - GRID[k], t=GRID[k], scheme=rule)
        rows.append(cur)
    for key in (PHI, Q):
        assert torch.equal(out[key], torch.stack([r[key] for r in rows]))
    assert torch.equal(out["time"], torch.tensor(GRID, dtype=F64))


def test_diagnostics_list_the_scheme_of_each_step():
    model, state, drivers = _rc()
    diag: dict = {}
    model.march(state, drivers, GRID, restarts={1}, diagnostics=diag)
    assert diag["schemes"] == [None, "backward_euler", None, None, None]
    diag = {}
    model.march(state, drivers, GRID, restarts={1}, scheme="trapezoidal", diagnostics=diag)
    assert diag["schemes"] == ["trapezoidal", "backward_euler", "trapezoidal", "trapezoidal",
                               "trapezoidal"]


def test_record_subsets_the_rows():
    model, state, drivers = _rc()
    full = model.march(state, drivers, GRID, restarts={0})
    part = model.march(state, drivers, GRID, restarts={0}, record=[0, 2, 5])
    assert part[PHI].shape[0] == 3
    assert torch.equal(part[PHI], full[PHI][[0, 2, 5]])
    assert torch.equal(part["time"], torch.tensor([0.0, 0.25, 1.0], dtype=F64))
    late = model.march(state, drivers, GRID, restarts={0}, record=[3])
    assert torch.equal(late[PHI][0], full[PHI][3])


def test_drivers_at_is_called_with_each_interval():
    model, state, drivers = _rc()
    calls = []

    def at(t0, t1):
        calls.append((t0, t1))
        return drivers

    model.march(state, drivers, GRID, drivers_at=at)
    assert calls == list(zip(GRID[:-1], GRID[1:], strict=True))


def test_bad_arguments_are_named():
    model, state, drivers = _rc()
    with pytest.raises(ValueError, match="increasing"):
        model.march(state, drivers, [0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="record"):
        model.march(state, drivers, GRID, record=[2, 1])


def test_a_batched_state_broadcasts_the_start_row():
    model, state, drivers = _rc()
    batched = {k: v.expand(3, *v.shape).clone() for k, v in state.items()}
    out = model.march(batched, drivers, GRID[:3], restarts={0})
    assert out[PHI].shape == (3, 3, state[PHI].shape[-1])
    single = model.march(state, drivers, GRID[:3], restarts={0})
    torch.testing.assert_close(out[PHI][1], single[PHI][1].expand(3, -1))


def _transport_model(scheme):
    net = Network(dtype=F64)
    for name in ("ambient", "z1", "z2"):
        net.add_node(name, z_ref=0.0)
    net.add_edge("ambient", "z1", kind="airpath", z_path=0.0)
    net.add_edge("z1", "z2", kind="airpath", z_path=0.0)
    net.add_edge("z2", "ambient", kind="airpath", z_path=0.0)
    el = PowerLaw(torch.tensor([0.01, 0.02, 0.01], dtype=F64), 0.65)
    air = KirchhoffLayer(net, "air", [el], boundary=["ambient"], quantity="pressure", unit="Pa")
    species = TransportLayer(net, "species", capacity=torch.tensor([50.0, 80.0], dtype=F64),
                             flow_kind="airpath", boundary=["ambient"], scheme=scheme,
                             quantity="mass_fraction", unit="kg/kg")
    model = Model(net, {"air": air, "species": species})
    drivers = {"air.phi_boundary": torch.zeros(1, dtype=F64),
               "air.edge_head": torch.tensor([5.0, 0.0, 0.0], dtype=F64),
               "species.x_boundary": torch.tensor([1e-3], dtype=F64),
               "species.sources": torch.zeros(3, dtype=F64)}
    return model, species, {"species.x": torch.zeros(2, dtype=F64)}, drivers


@pytest.mark.parametrize(("scheme", "expected"), [
    ("trapezoidal", ["backward_euler", None, None]),
    ("exact", [None, None, None]),
    ("backward_euler", [None, None, None]),
])
def test_a_restart_step_takes_a_trapezoidal_transport_layer_by_backward_euler(
        monkeypatch, scheme, expected):
    model, species, state, drivers = _transport_model(scheme)
    seen = []
    original = species.step

    def spy(*args, **kwargs):
        seen.append(kwargs.get("scheme"))
        return original(*args, **kwargs)

    monkeypatch.setattr(species, "step", spy)
    model.march(state, drivers, [0.0, 60.0, 120.0, 180.0], restarts={0})
    assert seen == expected


def test_transient_equals_the_old_loop_bitwise():
    from noodl.apps.electrical import Pulse, transient

    model, _, drivers = build_model(Circuit([
        VoltageSource("V1", "a", "0", 1.0, waveform=Pulse(width=50.0, period=0.5, start_time=0.25)),
        Resistor("R1", "a", "b", 1.0), Capacitor("C1", "b", "0", 1.0)]))
    state = initial_state(model, drivers)
    grid = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2]
    out = transient(model, state, drivers, grid)
    sources = [el for el in (model.kirchhoff[LAYER].law_for(k)[0]
                             for k in model.kirchhoff[LAYER].kinds) if hasattr(el, "breakpoints")]
    cur, phis, qs, rules, evaluated = dict(state), [state[PHI]], [state[Q]], [], grid[0]
    for k in range(len(grid) - 1):
        t0, end = grid[k], grid[k + 1]
        jump = any(src.breakpoints(evaluated, end) for src in sources)
        rule = "backward_euler" if k == 0 or jump else "trapezoidal"
        cur = model.step(cur, drivers, end - t0, t=t0, scheme=rule)
        evaluated = end
        phis.append(cur[PHI])
        qs.append(cur[Q])
        rules.append(rule)
    assert "backward_euler" in rules[1:] and "trapezoidal" in rules
    assert out["scheme"] == rules
    assert torch.equal(out[PHI], torch.stack(torch.broadcast_tensors(*phis)))
    assert torch.equal(out[Q], torch.stack(torch.broadcast_tensors(*qs)))


def test_simulate_circuit_equals_the_old_loop_bitwise():
    from pathlib import Path

    from noodl.apps.electrical.modelica import read_modelica_circuit, simulate_circuit
    from noodl.modelica.timeline import grid_index, step_drivers

    data = Path(__file__).parent / "data" / "modelica_electrical"
    model, state, drivers, names = read_modelica_circuit(
        data / "Analog.Examples.HeatingRectifier.json")
    out = simulate_circuit(model, state, drivers, names, times=names.times[:6])
    grid = names.grid
    want = grid_index(grid, names.times[:6], caller="test")
    rows, cur = [], state
    for k in range(want[-1]):
        t0, t1 = float(grid[k]), float(grid[k + 1])
        rule = "backward_euler" if k in names.restarts else "trapezoidal"
        cur = model.step(cur, step_drivers(drivers, grid, t1), dt=t1 - t0, t=t0, scheme=rule,
                         layer_options={"thermal": {"scheme": rule}})
        if k + 1 in set(want):
            rows.append(cur)
    if want[0] == 0:
        rows.insert(0, state)
    keys = [key for key in rows[0] if key.endswith((".phi", ".q", ".x", ".storage_flow"))
            or key in model.closure_state_keys]
    assert set(out) == {*keys, "time"}
    for key in keys:
        assert torch.equal(out[key], torch.stack([row[key] for row in rows]))


def test_negative_restart_is_refused():
    model, state, drivers = _rc()
    with pytest.raises(ValueError, match="restarts"):
        model.march(state, drivers, GRID, restarts={-1})


def test_restart_past_the_last_step_is_ignored():
    model, state, drivers = _rc()
    a = model.march(state, drivers, GRID, restarts={0, 99})
    b = model.march(state, drivers, GRID, restarts={0})
    assert torch.equal(a[PHI], b[PHI])


def test_layer_options_merge_over_the_callers_on_a_restart_step(monkeypatch):
    model, species, state, drivers = _transport_model("trapezoidal")
    seen = []
    original = model.step

    def spy(*args, **kwargs):
        seen.append(kwargs["layer_options"])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "step", spy)
    model.march(state, drivers, [0.0, 60.0, 120.0], restarts={0},
                layer_options={"air": {"atol": 1e-10}, "species": {"scheme": "trapezoidal"}})
    assert seen[0] == {"air": {"atol": 1e-10}, "species": {"scheme": "backward_euler"}}
    assert seen[1] == {"air": {"atol": 1e-10}, "species": {"scheme": "trapezoidal"}}


def test_a_trapezoidal_transport_layer_matches_explicit_steps():
    model, species, state, drivers = _transport_model("trapezoidal")
    grid = [0.0, 60.0, 120.0, 180.0]
    diag: dict = {}
    out = model.march(state, drivers, grid, restarts={0}, diagnostics=diag)
    assert diag["schemes"] == ["backward_euler", None, None]
    cur, rows = state, [state["species.x"]]
    for k, rule in enumerate(("backward_euler", "trapezoidal", "trapezoidal")):
        cur = model.step(cur, drivers, grid[k + 1] - grid[k], t=grid[k],
                         layer_options={"species": {"scheme": rule}})
        rows.append(cur["species.x"])
    assert torch.equal(out["species.x"], torch.stack(rows))


def _tank_model():
    """A reservoir feeding a tank through one junction: storage at a boundary node, stepped
    by forward Euler."""
    from noodl.storage import LinearStorage

    net = Network(dtype=F64)
    for n in ("res", "j", "tank"):
        net.add_node(n)
    net.add_edge("res", "j", kind="pipe")
    net.add_edge("j", "tank", kind="pipe")
    law = Conductance(torch.tensor([1.0, 1.0], dtype=F64), kind="pipe")
    layer = KirchhoffLayer(net, "water", [law], boundary=["res", "tank"],
                           storage=[LinearStorage([2], 10.0)], scheme="forward_euler")
    model = Model(net, [layer])
    drivers = {"water.phi_boundary": torch.tensor([10.0], dtype=F64)}
    state = {"water.phi": torch.tensor([10.0, 5.0, 1.0], dtype=F64),
             "water.q": torch.zeros(2, dtype=F64)}
    return model, layer, state, drivers


def test_a_restart_keeps_forward_euler_on_a_layer_with_boundary_storage():
    model, _, state, drivers = _tank_model()
    times = [0.0, 1.0, 2.0, 3.0]
    plain = model.march(state, drivers, times)
    diag: dict = {}
    restarted = model.march(state, drivers, times, restarts={0, 2}, diagnostics=diag)
    assert diag["schemes"] == ["backward_euler", None, "backward_euler"]
    for key in plain:
        assert torch.equal(restarted[key], plain[key]), key


def test_a_restart_gives_backward_euler_to_the_layers_that_step_in_their_solve(monkeypatch):
    model, state, drivers = _rc()
    seen = []
    original = model.step

    def spy(*args, **kwargs):
        seen.append((kwargs.get("scheme"), kwargs["layer_options"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "step", spy)
    model.march(state, drivers, GRID[:3], restarts={0}, scheme="trapezoidal")
    assert seen == [("trapezoidal", {LAYER: {"scheme": "backward_euler"}}),
                    ("trapezoidal", None)]


def test_the_history_carries_storage_flow_the_start_state_did_not():
    model, layer, state, drivers = _tank_model()
    out = model.march(state, drivers, [0.0, 1.0, 2.0])
    assert "water.storage_flow" not in state
    step = model.step(state, drivers, dt=1.0, t=0.0)
    assert torch.equal(out["water.storage_flow"][1], step["water.storage_flow"])
    # row 0: what the start flows deliver to the tank
    assert torch.equal(out["water.storage_flow"][0],
                       layer.storage_flows(state["water.phi"], {}, dt=None, q=state["water.q"]))
    rc, rc_state, rc_drivers = _rc()
    rc_state = {k: v for k, v in rc_state.items() if not k.endswith(".storage_flow")}
    hist = rc.march(rc_state, rc_drivers, GRID[:3], restarts={0})
    assert torch.equal(hist[f"{LAYER}.storage_flow"][0],
                       torch.zeros_like(hist[f"{LAYER}.storage_flow"][1]))


class _Clock:
    """An integrating closure carrying "clock.t" in the state."""

    integrates = True
    state_keys = ("clock.t",)

    def __call__(self, state, drivers, ctx):
        dt = 0.0 if ctx is None else ctx.dt
        return {"clock.t": state["clock.t"] + dt}


def test_the_history_records_the_closure_carried_state():
    model, _, drivers = build_model(Circuit([
        VoltageSource("V1", "a", "0", 1.0), Resistor("R1", "a", "b", 1.0),
        Capacitor("C1", "b", "0", 1.0)]))
    clocked = Model(model.net, model.kirchhoff, closures=[_Clock()])
    state = {**initial_state(model, drivers), "clock.t": torch.tensor(0.0, dtype=F64)}
    out = clocked.march(state, drivers, GRID, restarts={0})
    torch.testing.assert_close(out["clock.t"], torch.tensor(GRID, dtype=F64), rtol=0,
                               atol=1e-15)

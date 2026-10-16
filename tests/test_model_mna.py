"""Model driving layers solved by Modified Nodal Analysis: step history from the step start,
the end-of-step time, warm starts, steady with a time, residuals and the iterate coupling."""

import pytest
import torch

from noodl.elements import Conductance, ConstitutiveLaw
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.layers.transport import TransportLayer
from noodl.model import Model
from noodl.refs import required_state_keys
from noodl.topology import Network
from noodl.validation import check_setup
from tests._circuit_laws import Cap, DrivenSource, Ind, Source

F64 = torch.float64
GROUND = torch.zeros(1, dtype=F64)


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _net(nodes, edges):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v, kind in edges:
        net.add_edge(u, v, kind=kind)
    return net


class Clock(ConstitutiveLaw):
    """q = g dp, recording the time every call sees."""

    contextual = True

    def __init__(self, kind="r"):
        super().__init__(kind)
        self.times = []

    def flow(self, dp, drivers=None, ctx=None):
        self.times.append(ctx.t)
        return dp


def _rc(source=None, resistor=None):
    # source a -> gnd holds a at +V, resistor a -> b, capacitor b -> gnd (R C = 1)
    net = _net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "r"),
                                   ("b", "gnd", "cap")])
    source = Source(torch.tensor([1.0])) if source is None else source
    resistor = Conductance(torch.tensor([1.0]), kind="r") if resistor is None else resistor
    layer = KirchhoffLayer(net, "e", [source, resistor, Cap(torch.tensor([1.0]))],
                           boundary=["gnd"])
    return net, layer


def _initial(layer, drivers=None):
    phi, q = layer.solve(GROUND, drivers, mode="initial", differentiable=False)
    return {"e.phi": phi, "e.q": q}


def test_model_steps_rc_like_the_layer():
    net, layer = _rc()
    b = net.nodes.index("b")
    model = Model(net, [layer])
    drivers = {"e.phi_boundary": GROUND}
    state = _initial(layer)
    dt, v = 0.1, 0.0
    for _ in range(10):
        state = model.step(state, drivers, dt=dt, integration="backward_euler")
        v = (v + dt) / (1.0 + dt)
        torch.testing.assert_close(state["e.phi"][b], torch.tensor(v), rtol=0, atol=1e-14)
    torch.testing.assert_close(state["e.q"][2], state["e.q"][1], rtol=0, atol=1e-14)


def test_model_steady_is_dc_operating_point():
    clock = Clock()
    net, layer = _rc(resistor=clock)
    model = Model(net, [layer])
    state = model.steady({}, {"e.phi_boundary": GROUND}, t=0.0)
    # the capacitor is open: no current, b sits at the source potential
    torch.testing.assert_close(state["e.phi"][net.nodes.index("b")], torch.tensor(1.0),
                               rtol=0, atol=1e-14)
    torch.testing.assert_close(state["e.q"], torch.zeros(3), rtol=0, atol=1e-14)
    assert clock.times and set(clock.times) == {0.0}


def test_step_passes_end_time():
    clock = Clock()
    net, layer = _rc(resistor=clock)
    model = Model(net, [layer])
    drivers = {"e.phi_boundary": GROUND}
    state = _initial(layer)
    clock.times.clear()
    model.step(state, drivers, dt=0.5, t=1.0)
    assert clock.times and set(clock.times) == {1.5}
    clock.times.clear()
    model.step(state, drivers, dt=0.5)
    assert clock.times and set(clock.times) == {None}


def test_step_hands_the_layer_warm_start_history_and_end_time(monkeypatch):
    net, layer = _rc()
    model = Model(net, [layer])
    drivers = {"e.phi_boundary": GROUND}
    start = _initial(layer)
    fed = {"e.phi": start["e.phi"] + 0.25, "e.q": start["e.q"] + 0.5}
    seen = []
    solve = layer.solve

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return solve(*args, **kwargs)

    monkeypatch.setattr(layer, "solve", spy)
    model._pass(fed, drivers, 0.1, {}, step_from=start, t=2.0)
    (kw,) = seen
    assert torch.equal(kw["q0"], fed["e.q"][..., layer.implicit_edges])
    assert torch.equal(kw["phi0"], fed["e.phi"][..., layer.interior])
    assert kw["history"][0] is start["e.phi"] and kw["history"][1] is start["e.q"]
    assert kw["dt"] == 0.1 and kw["t"] == pytest.approx(2.1)


def test_step_without_warm_start_drops_the_flow_warm_start_but_keeps_the_history(monkeypatch):
    net, layer = _rc()
    model = Model(net, [layer])
    drivers = {"e.phi_boundary": GROUND}
    start = _initial(layer)
    fed = {"e.phi": start["e.phi"] + 0.25, "e.q": start["e.q"] + 0.5}
    seen = []
    solve = layer.solve

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return solve(*args, **kwargs)

    monkeypatch.setattr(layer, "solve", spy)
    model._pass(fed, drivers, 0.1, {}, step_from=start, t=2.0, warm_start=False)
    (kw,) = seen
    assert "q0" not in kw
    assert kw["phi0"] is None
    assert kw["history"][0] is start["e.phi"] and kw["history"][1] is start["e.q"]
    assert kw["dt"] == 0.1 and kw["t"] == pytest.approx(2.1)


def test_classic_layer_gets_no_circuit_keywords(monkeypatch):
    net = _net(["g", "a"], [("a", "g", "r")])
    layer = KirchhoffLayer(net, "c", [Conductance(torch.tensor([1.0]), kind="r")],
                           boundary=["g"])
    seen = []
    solve = layer.solve

    def spy(*args, **kwargs):
        seen.append(set(kwargs))
        return solve(*args, **kwargs)

    monkeypatch.setattr(layer, "solve", spy)
    state = {"c.phi": torch.zeros(2), "c.q": torch.zeros(1)}
    Model(net, [layer]).step(state, {"c.phi_boundary": torch.ones(1)}, dt=0.1, t=0.0)
    assert seen == [{"phi0", "diagnostics"}]


def test_missing_flow_state_on_dynamic_step_is_named():
    net, layer = _rc()
    model = Model(net, [layer])
    state = _initial(layer)
    del state["e.q"]
    with pytest.raises(KeyError, match=r"e\.q"):
        model.step(state, {"e.phi_boundary": GROUND}, dt=0.1)


def _feedback_model(coupling, **kwargs):
    net, layer = _rc(source=DrivenSource())
    b = net.nodes.index("b")

    def feedback(state, drivers):
        # weak feedback: the source follows the capacitor voltage
        return {"V": 1.0 + 0.1 * state["e.phi"][..., b:b + 1]}

    return net, layer, Model(net, [layer], closures=[feedback], coupling=coupling, **kwargs)


def test_iterate_restarts_dynamic_layer_from_step_start():
    net, layer, model = _feedback_model("iterate", iterate_tol={"e": 1e-12})
    _, _, pingpong = _feedback_model("pingpong")
    b = net.nodes.index("b")
    drivers = {"e.phi_boundary": GROUND, "V": torch.ones(1)}
    state = _initial(layer, drivers)
    dt = 0.1
    manual = dict(state)
    for _ in range(2):
        # Newton's default tolerance (sqrt(eps)) would accept a warm start within ~1e-8 of
        # the answer unchanged and stop the passes there; tighten it to compare at 1e-10
        state = model.step(state, drivers, dt=dt, integration="backward_euler", atol=1e-13)
        # the fixed point by hand: a backward Euler step from v_old with source V gives
        # v = (v_old + dt V)/(1 + dt), and V = 1 + 0.1 v
        v_old = float(manual["e.phi"][b])
        v = (v_old + dt) / (1.0 + dt - 0.1 * dt)
        # one plain step at the converged source value reproduces it
        fixed = {"e.phi_boundary": GROUND, "V": torch.tensor([1.0 + 0.1 * v])}
        manual = Model(net, [layer]).step(manual, fixed, dt=dt, integration="backward_euler")
        torch.testing.assert_close(manual["e.phi"][b], torch.tensor(v), rtol=0, atol=1e-13)
        torch.testing.assert_close(state["e.phi"], manual["e.phi"], rtol=0, atol=1e-10)
        torch.testing.assert_close(state["e.q"], manual["e.q"], rtol=0, atol=1e-10)
    # ping-pong reads the step-start potential and so differs from the fixed point
    once = pingpong.step(_initial(layer, drivers), drivers, dt=dt,
                         integration="backward_euler")
    assert abs(float(once["e.phi"][b]) - float(model.step(
        _initial(layer, drivers), drivers, dt=dt, integration="backward_euler")["e.phi"][b])
    ) > 1e-6


def test_residuals_use_balance_for_mna():
    net, layer = _rc()
    model = Model(net, [layer])
    drivers = {"e.phi_boundary": GROUND}
    state = model.step(_initial(layer), drivers, dt=0.1)
    r = model.residuals(state, drivers)["e"]
    assert r.shape == (2,)
    assert float(r.abs().max()) < 1e-12
    # and it is the layer's balance of the state's own flows
    torch.testing.assert_close(r, layer.balance(state["e.phi"], state["e.q"]), rtol=0, atol=0)


def test_balance_matches_classic_residual():
    net = _net(["gnd", "a", "b"], [("a", "gnd", "r"), ("a", "b", "r"), ("b", "gnd", "r")])
    layer = KirchhoffLayer(net, "c", [Conductance(torch.tensor([1.0, 2.0, 3.0]), kind="r")],
                           boundary=["gnd"])
    phi = torch.tensor([0.0, 0.3, -0.2])
    sources = torch.tensor([0.0, 0.5, 0.1])
    q = layer.flows(phi, {})
    expected = layer.residual(phi[layer.interior], phi[layer.bound], {}, sources)
    torch.testing.assert_close(layer.balance(phi, q, sources), expected, rtol=0, atol=1e-15)


def test_current_flows_reads_the_state_of_an_mna_owner():
    net, layer = _rc()
    heat = TransportLayer(net, "heat", capacity=torch.ones(1), flow_kind="r",
                          boundary=["a"])
    model = Model(net, [layer, heat])
    drivers = {"e.phi_boundary": GROUND, "heat.x_boundary": torch.zeros(1)}
    state = {**_initial(layer), "heat.x": torch.zeros(1)}
    q = model.current_flows("heat", state, drivers)
    torch.testing.assert_close(q, state["e.q"][1:2], rtol=0, atol=0)
    del state["e.q"]
    with pytest.raises(KeyError, match=r"e\.q"):
        model.current_flows("heat", state, drivers)


# ----------------------------------------------------------------------------- phasor


def _rlc_phasor():
    net = _net(["gnd", "a", "b", "c"], [("a", "gnd", "source"), ("a", "b", "r"),
                                        ("b", "c", "ind"), ("c", "gnd", "cap")])
    layer = KirchhoffLayer(
        net, "e", [DrivenSource(), Conductance(torch.tensor([0.5]), kind="r"),
                   Ind(torch.tensor([1e-3])), Cap(torch.tensor([1e-6]))],
        boundary=["gnd"], phasor=True)
    return net, layer


def test_anderson_refuses_phasor_layer():
    net, layer = _rlc_phasor()
    with pytest.raises(ValueError, match=r"phasor layer\(s\) \['e'\]"):
        Model(net, [layer], coupling="iterate", iterate_tol={"e": 1e-9},
              iterate_acceleration="anderson", iterate_flow_tol={"e": 1e-9})


def test_iterate_converges_on_complex_phasor_state():
    net, layer = _rlc_phasor()
    c = net.nodes.index("c")

    def feedback(state, drivers):
        return {"V": (1.0 + 0.01 * state["e.phi"][..., c:c + 1]).to(torch.complex128)}

    model = Model(net, [layer], closures=[feedback], coupling="iterate",
                  iterate_tol={"e": 1e-12}, iterate_max=60)
    drivers = {"e.phi_boundary": GROUND, "e.frequency": torch.tensor(1000.0),
               "V": torch.ones(1, dtype=torch.complex128)}
    phi0 = layer.solve(GROUND, drivers, differentiable=False)[0]
    diag: dict = {}
    state = model.steady({"e.phi": phi0}, drivers, diagnostics=diag, atol=1e-13)
    assert state["e.phi"].dtype == torch.complex128
    assert bool(diag["converged"].all())
    # the hand fixed point: v_c is linear in V, v_c = H V, V = 1 + 0.01 v_c
    H = phi0[c]
    v_c = H / (1.0 - 0.01 * H)
    torch.testing.assert_close(state["e.phi"][c], v_c, rtol=0, atol=1e-10)
    assert float(model.residuals(state, drivers)["e"].abs().max()) < 1e-10


class RealCurrent(ConstitutiveLaw):
    """A phasor-capable current source returning a real tensor (I from u to v)."""

    phasor_capable = True

    def __init__(self, current, kind="load"):
        super().__init__(kind)
        self.current = self._param(current, False)

    def flow(self, dp, drivers=None):
        return self.current.expand(dp.real.shape)


@pytest.mark.parametrize("differentiable", [False, True])
def test_phasor_law_returning_real_tensor(differentiable):
    net = _net(["gnd", "a", "b"], [("a", "gnd", "source"), ("a", "b", "r"),
                                   ("b", "gnd", "load")])
    layer = KirchhoffLayer(
        net, "e", [Source(torch.tensor([1.0])), Conductance(torch.tensor([1.0]), kind="r"),
                   RealCurrent(torch.tensor([0.5]))], boundary=["gnd"], phasor=True)
    phi, q = layer.solve(GROUND, {"e.frequency": torch.tensor(50.0)},
                         differentiable=differentiable)
    assert phi.dtype == torch.complex128
    torch.testing.assert_close(phi[net.nodes.index("b")],
                               torch.tensor(0.5, dtype=torch.complex128), rtol=0, atol=1e-14)
    torch.testing.assert_close(q[2], torch.tensor(0.5, dtype=torch.complex128), rtol=0,
                               atol=1e-14)


def test_frequency_driver_must_be_one_per_instance():
    net, layer = _rlc_phasor()
    batch_boundary = torch.zeros(3, 1)
    drivers = {"V": torch.ones(1, dtype=torch.complex128)}
    ok = layer.solve(batch_boundary, {**drivers, "e.frequency": torch.tensor([50.0, 60.0, 70.0])},
                     differentiable=False)[0]
    assert ok.shape == (3, 4)
    with pytest.raises(ValueError, match=r"e\.frequency"):
        layer.solve(batch_boundary, {**drivers, "e.frequency": torch.full((3, 1), 50.0)},
                    differentiable=False)
    with pytest.raises(ValueError, match=r"e\.frequency"):
        layer.solve(batch_boundary, {**drivers, "e.frequency": torch.full((2,), 50.0)},
                    differentiable=False)


def test_frequency_key_and_required_inputs():
    net, layer = _rlc_phasor()
    assert layer.frequency_key == "e.frequency"
    assert _rc()[1].frequency_key is None
    classic = KirchhoffLayer(
        _net(["g", "a"], [("a", "g", "r")]), "c", [Conductance(torch.tensor([1.0]), kind="r")],
        boundary=["g"])
    assert classic.frequency_key is None
    model = Model(net, [layer])
    field = model.refs["e"].frequency
    assert str(field) == "e.frequency" and field.unit == "Hz"
    assert field.role == "driver" and field.required == "always"
    report = check_setup(model, {}, {"e.phi_boundary": GROUND,
                                     "V": torch.ones(1, dtype=torch.complex128)})
    assert any(i.code == "missing-input" and i.key == "e.frequency" for i in report.errors)
    report = check_setup(model, {}, {"e.phi_boundary": GROUND, "e.frequency": torch.tensor(50.0),
                                     "V": torch.ones(1, dtype=torch.complex128)})
    assert not any(i.key == "e.frequency" for i in report.issues)


def test_dynamic_layer_requires_its_state_to_step():
    net, layer = _rc()
    model = Model(net, [layer])
    assert {"e.phi", "e.q"} <= set(required_state_keys(model))
    assert model.refs["e"].q.required == "step"
    report = check_setup(model, {"e.phi": torch.zeros(3)}, {"e.phi_boundary": GROUND})
    assert any(i.code == "missing-input" and i.key == "e.q" for i in report.errors)
    # a steady solve needs no state
    report = check_setup(model, {}, {"e.phi_boundary": GROUND}, steady=True)
    assert not report.errors


def _phasor_feedback_model(conductance):
    net = _net(["gnd", "a", "b", "c"], [("a", "gnd", "source"), ("a", "b", "r"),
                                        ("b", "c", "ind"), ("c", "gnd", "cap")])
    layer = KirchhoffLayer(
        net, "e", [DrivenSource(), conductance, Ind(torch.tensor([1e-3])),
                   Cap(torch.tensor([1e-6]))], boundary=["gnd"], phasor=True)

    def feedback(state, drivers):
        return {"V": (1.0 + 0.01 * state["e.phi"][..., 3:4]).to(torch.complex128)}

    return Model(net, [layer], closures=[feedback], coupling="iterate",
                 iterate_tol={"e": 1e-12}, iterate_max=60)


def test_iterate_gradient_on_complex_phasor_state():
    g = Conductance(torch.tensor([0.5]), kind="r", learnable=True)
    (param,) = list(g.parameters())
    model = _phasor_feedback_model(g)
    drivers = {"e.phi_boundary": GROUND, "e.frequency": torch.tensor(1000.0),
               "V": torch.ones(1, dtype=torch.complex128)}
    state = {"e.phi": torch.zeros(4, dtype=torch.complex128)}
    diag: dict = {}
    out = model.steady(state, drivers, diagnostics=diag, atol=1e-13)
    assert diag["adjoint"] == "implicit"
    (grad,) = torch.autograd.grad(out["e.phi"][3].abs(), [param])

    def value(g_value):
        with torch.no_grad():
            param.copy_(torch.tensor([g_value]))
            return float(model.steady(state, drivers, atol=1e-13)["e.phi"][3].abs())

    h = 1e-6
    fd = (value(0.5 + h) - value(0.5 - h)) / (2 * h)
    torch.testing.assert_close(grad, torch.tensor([fd]), rtol=1e-6, atol=0)


def test_model_owned_solve_keywords_are_refused_by_name():
    net, layer = _rc()
    model = Model(net, [layer])
    drivers = {"e.phi_boundary": GROUND}
    state = _initial(layer)
    with pytest.raises(ValueError, match=r"\['history', 'q0'\].*set by the Model"):
        model.step(state, drivers, 0.1, q0=state["e.q"][..., :1],
                   history=(state["e.phi"], state["e.q"]))
    with pytest.raises(ValueError, match=r"\['dt'\].*set by the Model"):
        model.steady(state, drivers, t=0.0, dt=0.1)
    with pytest.raises(ValueError, match=r"\['phi0'\].*set by the Model"):
        model.steady(state, drivers, t=0.0, phi0=torch.zeros(2))


def _two_layers():
    net = _net(["g", "a", "b"], [("a", "g", "r"), ("b", "g", "s")])
    layers = [KirchhoffLayer(net, name, [Conductance(torch.tensor([1.0]), kind=kind)],
                             boundary=["g"]) for name, kind in (("c", "r"), ("d", "s"))]
    state = {f"{n}.{k}": torch.zeros(3 if k == "phi" else 1)
             for n in ("c", "d") for k in ("phi", "q")}
    drivers = {"c.phi_boundary": torch.ones(1), "d.phi_boundary": torch.ones(1)}
    return net, layers, state, drivers


def test_layer_options_give_one_layer_its_own_solve_keywords(monkeypatch):
    net, layers, state, drivers = _two_layers()
    seen = {}
    for layer in layers:
        def spy(*args, _solve=layer.solve, _name=layer.name, **kwargs):
            seen[_name] = (kwargs.get("atol"), kwargs.get("rtol"))
            return _solve(*args, **kwargs)
        monkeypatch.setattr(layer, "solve", spy)
    model = Model(net, layers)
    model.steady(state, drivers, atol=1e-12, rtol=1e-14, layer_options={"d": {"atol": 1e-11}})
    assert seen == {"c": (1e-12, 1e-14), "d": (1e-11, 1e-14)}
    model.step(state, drivers, 0.1, layer_options={"c": {"rtol": 1e-9}})
    assert seen == {"c": (None, 1e-9), "d": (None, None)}


def test_layer_options_refuse_unknown_layers_and_model_owned_keywords():
    net, layers, state, drivers = _two_layers()
    model = Model(net, layers)
    with pytest.raises(ValueError, match=r"layer_options names \['x'\].*has \['c', 'd'\]"):
        model.steady(state, drivers, layer_options={"x": {"atol": 1.0}})
    with pytest.raises(ValueError, match=r"\['phi0'\].*set by the Model"):
        model.steady(state, drivers, layer_options={"c": {"phi0": torch.zeros(2)}})

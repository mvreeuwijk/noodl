"""Node storage on a layer solved by Modified Nodal Analysis, and the held initial point."""

import pytest
import torch

from noodl import LinearStorage, Model
from noodl.apps.electrical.laws import Capacitance, PrescribedVoltage, Resistance
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network

F64 = torch.float64
V, R, C, V0 = 5.0, 2.0, 0.3, 1.5
PB = torch.zeros(1, dtype=F64)


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _net(with_cap: bool) -> Network:
    net = Network(dtype=F64)
    for n in ("gnd", "s", "a"):
        net.add_node(n)
    net.add_edge("s", "gnd", kind="voltage_source")
    net.add_edge("s", "a", kind="resistor")
    if with_cap:
        net.add_edge("a", "gnd", kind="capacitor")
    return net


def _laws(learnable=False):
    return [PrescribedVoltage(V, kind="voltage_source"),
            Resistance(R, kind="resistor", learnable=learnable)]


def _cap_layer():
    return KirchhoffLayer(_net(True), "el", [*_laws(), Capacitance(C, v0=V0)],
                          boundary=["gnd"])


def _storage_layer(learnable=False, initial=V0):
    storage = LinearStorage([2], C, initial=initial, learnable=learnable)
    layer = KirchhoffLayer(_net(False), "el", _laws(), boundary=["gnd"], storage=[storage])
    return layer, storage


def test_storage_layer_is_mna_and_dynamic():
    layer, storage = _storage_layer()
    assert layer.mna and layer.dynamic and layer.linear
    assert layer.storage == (storage,)


def _compare(cap_out, st_out, flows):
    (phi_c, q_c), (phi_s, _) = cap_out, st_out
    torch.testing.assert_close(phi_s, phi_c, rtol=0, atol=1e-12)
    torch.testing.assert_close(flows, q_c[..., 2:3], rtol=0, atol=1e-12)


@pytest.mark.parametrize("differentiable", [False, True])
def test_rc_storage_matches_a_capacitance_edge(differentiable):
    cap = _cap_layer()
    layer, _ = _storage_layer()
    kw = {"differentiable": differentiable}

    # the steady operating point: no current, storage absent
    cap_dc = cap.solve(PB, mode="dc", **kw)
    st_dc = layer.solve(PB, mode="dc", **kw)
    _compare(cap_dc, st_dc, layer.storage_flows(st_dc[0], {}, dt=None))
    torch.testing.assert_close(st_dc[0][2], torch.tensor(V))

    # the consistent initial point: the node held at v0, the flow into storage what
    # balances it, (V - v0) / R
    cap_0 = cap.solve(PB, mode="initial", **kw)
    st_0 = layer.solve(PB, mode="initial", **kw)
    w0 = layer.storage_flows(st_0[0], {}, dt=None, mode="initial", q=st_0[1])
    _compare(cap_0, st_0, w0)
    torch.testing.assert_close(st_0[0][2], torch.tensor(V0), rtol=0, atol=1e-12)
    torch.testing.assert_close(w0, torch.tensor([(V - V0) / R]), rtol=0, atol=1e-12)

    # one backward Euler step, then three trapezoidal steps
    cap_prev, st_prev, w = cap_0, st_0, w0
    for scheme in ["backward_euler"] + ["trapezoidal"] * 3:
        dt = 0.2
        cap_new = cap.solve(PB, dt=dt, scheme=scheme, history=cap_prev, **kw)
        st_new = layer.solve(PB, dt=dt, scheme=scheme, history=st_prev, storage_flow=w,
                             **kw)
        w = layer.storage_flows(st_new[0], {}, dt=dt, scheme=scheme, history=st_prev,
                                storage_flow=w)
        _compare(cap_new, st_new, w)
        cap_prev, st_prev = cap_new, st_new


def test_backward_euler_step_needs_no_storage_flow_and_trapezoidal_does():
    layer, _ = _storage_layer()
    st_0 = layer.solve(PB, mode="initial")
    layer.solve(PB, dt=0.1, scheme="backward_euler", history=st_0)
    with pytest.raises(ValueError, match="storage_flow"):
        layer.solve(PB, dt=0.1, scheme="trapezoidal", history=st_0)
    with pytest.raises(ValueError, match="history"):
        layer.solve(PB, dt=0.1, scheme="backward_euler")
    with pytest.raises(ValueError, match="takes no step"):
        layer.solve(PB, mode="initial", storage_flow=torch.zeros(1))


def test_without_an_initial_value_the_initial_point_is_the_steady_point():
    layer, _ = _storage_layer(initial=None)
    phi, q = layer.solve(PB, mode="initial")
    torch.testing.assert_close(phi[2], torch.tensor(V))
    assert torch.equal(layer.storage_flows(phi, {}, dt=None, mode="initial", q=q),
                       torch.zeros(1))


def test_initial_flows_split_by_capacity_among_a_nodes_entries():
    split = [LinearStorage([2], 0.1, initial=V0), LinearStorage([2, 2], [0.2, 0.0])]
    layer = KirchhoffLayer(_net(False), "el", _laws(), boundary=["gnd"], storage=split)
    phi, q = layer.solve(PB, mode="initial")
    torch.testing.assert_close(phi[2], torch.tensor(V0), rtol=0, atol=1e-12)
    flows = layer.storage_flows(phi, {}, dt=None, mode="initial", q=q)
    total = (V - V0) / R
    torch.testing.assert_close(flows, torch.tensor([total / 3, 2 * total / 3, 0.0]),
                               rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="needs q"):
        layer.storage_flows(phi, {}, dt=None, mode="initial")


def test_conflicting_initial_values_are_refused():
    two = [LinearStorage([2], C, initial=1.0), LinearStorage([2], C, initial=2.0)]
    layer = KirchhoffLayer(_net(False), "el", _laws(), boundary=["gnd"], storage=two)
    with pytest.raises(ValueError, match="'a' hold different initial potentials"):
        layer.solve(PB, mode="initial")
    repeated = [LinearStorage([2, 2], C, initial=[1.0, 2.0])]
    layer = KirchhoffLayer(_net(False), "el", _laws(), boundary=["gnd"], storage=repeated)
    with pytest.raises(ValueError, match="different initial potentials"):
        layer.solve(PB, mode="initial")
    # equal values at one node are one held potential
    same = [LinearStorage([2], C, initial=1.0), LinearStorage([2, 2], C, initial=1.0)]
    layer = KirchhoffLayer(_net(False), "el", _laws(), boundary=["gnd"], storage=same)
    phi, _ = layer.solve(PB, mode="initial")
    torch.testing.assert_close(phi[2], torch.tensor(1.0), rtol=0, atol=1e-12)


def test_a_phasor_layer_refuses_storage():
    with pytest.raises(ValueError, match="phasor layer takes no storage"):
        KirchhoffLayer(_net(False), "el", _laws(), boundary=["gnd"], phasor=True,
                       storage=[LinearStorage([2], C)])


def test_gradcheck_through_two_steps():
    layer, storage = _storage_layer(learnable=True)
    resistor = layer._laws[1]
    phi_start = torch.tensor([0.0, V, V0], requires_grad=True)
    w_start = torch.tensor([0.7], requires_grad=True)

    def two_steps(phi0, w0, cap, res):
        history = (phi0, torch.zeros(2))
        w = w0
        out = []
        for _ in range(2):
            phi, q = layer.solve(PB, dt=0.25, scheme="trapezoidal", history=history,
                                 storage_flow=w)
            w = layer.storage_flows(phi, {}, dt=0.25, scheme="trapezoidal",
                                    history=history, storage_flow=w)
            history = (phi, q)
            out += [phi, q, w]
        return torch.cat(out)

    assert torch.autograd.gradcheck(two_steps, (phi_start, w_start, storage.C, resistor.R))


def test_model_initial_point_then_steps():
    layer, _ = _storage_layer()
    cap = _cap_layer()
    model = Model(layer.net, [layer])
    cap_model = Model(cap.net, [cap])
    drivers = {"el.phi_boundary": PB}
    state = model.steady({}, drivers, mode="initial")
    cap_state = cap_model.steady({}, drivers, mode="initial")
    torch.testing.assert_close(state["el.storage_flow"], cap_state["el.q"][..., 2:3],
                               rtol=0, atol=1e-12)
    for scheme in ("backward_euler", "trapezoidal", "trapezoidal"):
        state = model.step(state, drivers, 0.2, scheme=scheme)
        cap_state = cap_model.step(cap_state, drivers, 0.2, scheme=scheme)
        torch.testing.assert_close(state["el.phi"], cap_state["el.phi"], rtol=0, atol=1e-12)
        torch.testing.assert_close(state["el.storage_flow"], cap_state["el.q"][..., 2:3],
                                   rtol=0, atol=1e-12)


# ------------------------------------------------------------------ fix round 1


class _ScaledSlope(LinearStorage):
    """A storage whose `rate_slope` is 3x the true slope (a Newton aid), declared inexact."""

    exact_slope = False

    def rate_slope(self, phi_nodes, phi_start, previous, ctx, drivers=None):
        return 3.0 * super().rate_slope(phi_nodes, phi_start, previous, ctx, drivers)


def test_an_inexact_rate_slope_does_not_reach_the_gradients():
    storage = _ScaledSlope([2], C, learnable=True)
    layer = KirchhoffLayer(_net(False), "el", _laws(), boundary=["gnd"], storage=[storage])
    exact, _ = _storage_layer(learnable=True, initial=None)
    phi_start = torch.tensor([0.0, V, V0], requires_grad=True)

    def step(lay, phi0, cap):
        phi, _ = lay.solve(PB, dt=0.25, scheme="backward_euler",
                           history=(phi0, torch.zeros(2)))
        return phi

    # the primal is the same root; the gradients are exact
    torch.testing.assert_close(step(layer, phi_start, storage.C),
                               step(exact, phi_start, exact.storage[0].C),
                               rtol=0, atol=1e-12)
    assert torch.autograd.gradcheck(lambda p, c: step(layer, p, c),
                                    (phi_start, storage.C))
    g = torch.autograd.grad(step(layer, phi_start, storage.C)[2], (phi_start, storage.C))
    g_exact = torch.autograd.grad(step(exact, phi_start, exact.storage[0].C)[2],
                                  (phi_start, exact.storage[0].C))
    for a, b in zip(g, g_exact, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


class _LearnedInitial(LinearStorage):
    """A storage whose initial potential is a learnable parameter."""

    def __init__(self, nodes, capacity, v0):
        super().__init__(nodes, capacity, learnable=True)
        self.v0 = torch.nn.Parameter(torch.as_tensor(v0, dtype=F64))

    def initial(self, drivers=None):
        return self.v0 * torch.ones(len(self.nodes), dtype=F64)


def test_gradcheck_through_the_initial_point_and_a_step():
    storage = _LearnedInitial([2], C, V0)
    layer = KirchhoffLayer(_net(False), "el", _laws(), boundary=["gnd"], storage=[storage])
    src = torch.tensor([0.0, 0.0, 0.4], requires_grad=True)

    def initial_then_step(s, v0, cap):
        phi0, q0 = layer.solve(PB, sources=s, mode="initial")
        w0 = layer.storage_flows(phi0, {}, dt=None, mode="initial", q=q0, sources=s)
        phi1, q1 = layer.solve(PB, sources=s, dt=0.2, scheme="trapezoidal",
                               history=(phi0, q0), storage_flow=w0)
        w1 = layer.storage_flows(phi1, {}, dt=0.2, scheme="trapezoidal",
                                 history=(phi0, q0), storage_flow=w0)
        return torch.cat([phi0, w0, phi1, w1])

    out = initial_then_step(src, storage.v0, storage.C)
    torch.testing.assert_close(out[2], torch.tensor(V0), rtol=0, atol=1e-12)
    # the source at the held node enters its storage: (V - v0) / R + 0.4
    torch.testing.assert_close(out[3], torch.tensor((V - V0) / R + 0.4), rtol=0, atol=1e-12)
    assert torch.autograd.gradcheck(initial_then_step, (src, storage.v0, storage.C))


def _mixed_net(cap_at_b: bool) -> Network:
    net = Network(dtype=F64)
    for n in ("gnd", "s", "a", "b"):
        net.add_node(n)
    net.add_edge("s", "gnd", kind="voltage_source")
    net.add_edge("s", "a", kind="resistor")
    net.add_edge("a", "b", kind="resistor")
    net.add_edge("a", "gnd", kind="capacitor")
    if cap_at_b:
        net.add_edge("b", "gnd", kind="capacitor")
    return net


@pytest.mark.parametrize("differentiable", [False, True])
def test_storage_beside_a_capacitance_edge_on_one_layer(differentiable):
    C_a, C_b = 0.4, 0.3
    laws = lambda: [PrescribedVoltage(V, kind="voltage_source"),  # noqa: E731
                    Resistance(torch.tensor([R, 1.5]), kind="resistor")]
    ref = KirchhoffLayer(_mixed_net(True), "el",
                         [*laws(), Capacitance(torch.tensor([C_a, C_b]))], boundary=["gnd"])
    mixed = KirchhoffLayer(_mixed_net(False), "el", [*laws(), Capacitance(C_a)],
                           boundary=["gnd"], storage=[LinearStorage([3], C_b)])
    caps_ref, caps_mixed = ref.kind_slice("capacitor"), mixed.kind_slice("capacitor")
    phi_prev = torch.tensor([0.0, V, 1.0, 2.0])
    q_ref = torch.zeros(len(ref.cols))
    q_ref[caps_ref] = torch.tensor([0.3, -0.2])
    q_mixed = torch.zeros(len(mixed.cols))
    q_mixed[caps_mixed] = 0.3
    w = torch.tensor([-0.2])
    h_ref, h_mixed = (phi_prev, q_ref), (phi_prev, q_mixed)
    for scheme in ("backward_euler", "trapezoidal", "trapezoidal"):
        out_ref = ref.solve(PB, dt=0.2, scheme=scheme, history=h_ref,
                            differentiable=differentiable)
        out_mixed = mixed.solve(PB, dt=0.2, scheme=scheme, history=h_mixed,
                                storage_flow=w, differentiable=differentiable)
        w = mixed.storage_flows(out_mixed[0], {}, dt=0.2, scheme=scheme, history=h_mixed,
                                storage_flow=w)
        torch.testing.assert_close(out_mixed[0], out_ref[0], rtol=0, atol=1e-12)
        torch.testing.assert_close(out_mixed[1][caps_mixed], out_ref[1][caps_ref][:1],
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(w, out_ref[1][caps_ref][1:], rtol=0, atol=1e-12)
        h_ref, h_mixed = out_ref, out_mixed

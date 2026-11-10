"""Node storage on the classic nodal path: `step_rate`, `NodeStorage`, `LinearStorage`."""

import pytest
import torch

from noodl import LinearStorage, Model, NodeStorage
from noodl.apps.electrical.laws import Capacitance, Inductance, MutualInductance
from noodl.elements import Conductance, LawContext, step_rate, step_rate_factor
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.operators.shifted import ShiftedOperator
from noodl.refs import required_state_keys
from noodl.topology import Network

F64 = torch.float64


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _ctx(scheme, dt=0.3):
    return LawContext(mode="transient", dt=dt, scheme=scheme)


# ------------------------------------------------------------------ step_rate


def test_step_rate_backward_euler_and_trapezoidal():
    change = torch.tensor([0.6, -1.2])
    previous = torch.tensor([0.5, 0.25])
    assert torch.equal(step_rate(change, _ctx("backward_euler")), change / 0.3)
    assert torch.equal(step_rate(change, _ctx("trapezoidal"), previous),
                       2.0 * change / 0.3 - previous)
    assert step_rate_factor(_ctx("backward_euler")) == 1.0 / 0.3
    assert step_rate_factor(_ctx("trapezoidal")) == 2.0 / 0.3


def test_step_rate_refuses_trapezoidal_without_previous_and_unknown_schemes():
    with pytest.raises(ValueError, match="previous"):
        step_rate(torch.ones(1), _ctx("trapezoidal"))
    with pytest.raises(ValueError, match="exact"):
        step_rate(torch.ones(1), _ctx("exact"))
    with pytest.raises(ValueError, match="euler"):
        step_rate_factor(_ctx("euler"))


def _ulp_close(new, old, *terms):
    """`new` equals `old` to a few units in the last place of the largest term."""
    scale = torch.stack([t.abs() for t in terms]).amax(0)
    assert torch.all((new - old).abs() <= 4 * torch.finfo(F64).eps * scale)


@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_capacitance_and_inductance_residuals_are_unchanged(scheme):
    gen = torch.Generator().manual_seed(7)
    r = lambda *s: torch.rand(*s, generator=gen, dtype=F64) * 4 - 2  # noqa: E731
    dp, q, dp_prev, q_prev = r(50), r(50), r(50), r(50)
    C, L, dt = r(50).abs() + 0.1, r(50).abs() + 0.1, 0.37
    ctx = LawContext(mode="transient", dt=dt, scheme=scheme, dp_prev=dp_prev, q_prev=q_prev)
    cap = Capacitance(C).residual(dp, q, {}, ctx)
    ind = Inductance(L).residual(dp, q, {}, ctx)
    if scheme == "backward_euler":
        # the same operations in the same order: bit for bit
        assert torch.equal(cap, q - C * (dp - dp_prev) / dt)
        assert torch.equal(ind, dp - L * (q - q_prev) / dt)
    else:
        old_cap = q + q_prev - 2.0 * C / dt * (dp - dp_prev)
        _ulp_close(cap, old_cap, q, q_prev, 2.0 * C / dt * (dp - dp_prev))
        old_ind = dp + dp_prev - 2.0 * L / dt * (q - q_prev)
        _ulp_close(ind, old_ind, dp, dp_prev, 2.0 * L / dt * (q - q_prev))


@pytest.mark.parametrize("scheme", ["backward_euler", "trapezoidal"])
def test_mutual_inductance_residual_is_unchanged(scheme):
    gen = torch.Generator().manual_seed(11)
    r = lambda *s: torch.rand(*s, generator=gen, dtype=F64) * 4 - 2  # noqa: E731
    dp, q, dp_prev, q_prev = r(40), r(40), r(40), r(40)
    law = MutualInductance(1.3, 0.7, 0.4)
    dt = 0.21
    ctx = LawContext(mode="transient", dt=dt, scheme=scheme, dp_prev=dp_prev, q_prev=q_prev)
    new = law.residual(dp, q, {}, ctx)
    flux = law._flux(q - q_prev)
    if scheme == "backward_euler":
        assert torch.equal(new, dp - flux / dt)
    else:
        _ulp_close(new, dp + dp_prev - 2.0 / dt * flux, dp, dp_prev, 2.0 / dt * flux)


# ------------------------------------------------------------------ an RC network

NODES = ["gnd", "a", "b", "c"]
G = torch.tensor([2.0, 1.5, 0.5], dtype=F64)        # gnd-a, a-b, b-c
CAP = torch.tensor([0.4, 1.1, 0.7], dtype=F64)      # at a, b, c
PB = torch.tensor([1.25], dtype=F64)                # the boundary potential at gnd
SRC = torch.tensor([0.0, 0.3, -0.2, 0.5], dtype=F64)  # full-node injections


def _net(nodes=NODES, edges=(("gnd", "a"), ("a", "b"), ("b", "c"))):
    net = Network(dtype=F64)
    for n in nodes:
        net.add_node(n)
    for u, v in edges:
        net.add_edge(u, v, kind="g")
    return net


def _rc(scheme="backward_euler", *, learnable=False, capacity=CAP):
    net = _net()
    storage = LinearStorage([1, 2, 3], capacity.clone(), learnable=learnable)
    layer = KirchhoffLayer(net, "heat", [Conductance(G.clone(), kind="g")], boundary=["gnd"],
                           storage=[storage], scheme=scheme)
    return layer, storage


def _laplacian(layer):
    A = layer.A
    L = A @ torch.diag(G) @ A.T
    i, b = layer.interior, layer.bound
    return L[i][:, i], L[i][:, b]


def _phi_prev(layer):
    return torch.tensor([PB.item(), 0.2, -0.4, 0.9])


def test_rc_layer_is_classic_dynamic_and_linear():
    layer, storage = _rc()
    assert layer.implicit_edges.numel() == 0
    assert layer.dynamic
    assert layer.linear
    assert layer.storage == (storage,)


def test_rc_operator_is_shifted_and_certified():
    layer, _ = _rc()
    phi = _phi_prev(layer)
    step = layer._step(None, 0.5, None, "backward_euler", (phi, None), None, {})
    slopes = layer._node_slopes(phi, {}, step)
    torch.testing.assert_close(slopes, CAP / 0.5, rtol=0, atol=0)
    op = layer._operator_at(layer.dflows(phi, {}), slopes)
    assert isinstance(op, ShiftedOperator)
    assert bool(op.spd_certificate())
    # no step: the bare Laplacian, as for a layer without storage
    assert layer._node_slopes(phi, {}, None) is None


def test_rc_backward_euler_step_matches_the_closed_form():
    layer, _ = _rc()
    phi_prev, dt = _phi_prev(layer), 0.5
    L_II, L_IB = _laplacian(layer)
    rhs = CAP * phi_prev[1:] / dt + SRC[1:] - L_IB @ PB
    want = torch.linalg.solve(L_II + torch.diag(CAP / dt), rhs)
    for differentiable in (False, True):
        phi, _ = layer.solve(PB, {}, SRC, dt=dt, history=(phi_prev, None),
                             differentiable=differentiable)
        torch.testing.assert_close(phi[1:], want, rtol=1e-12, atol=1e-12)
    flows = layer.storage_flows(phi, {}, dt=dt, history=(phi_prev, None))
    torch.testing.assert_close(flows, CAP * (phi[1:] - phi_prev[1:]) / dt)


def test_rc_trapezoidal_step_matches_the_closed_form_with_storage_flow():
    layer, _ = _rc("trapezoidal")
    phi_prev, dt = _phi_prev(layer), 0.5
    w_prev = torch.tensor([0.1, -0.3, 0.2])
    L_II, L_IB = _laplacian(layer)
    k = 2.0 * CAP / dt
    rhs = k * phi_prev[1:] + w_prev + SRC[1:] - L_IB @ PB
    want = torch.linalg.solve(L_II + torch.diag(k), rhs)
    phi, q = layer.solve(PB, {}, SRC, dt=dt, history=(phi_prev, None), storage_flow=w_prev)
    torch.testing.assert_close(phi[1:], want, rtol=1e-12, atol=1e-12)
    flows = layer.storage_flows(phi, {}, dt=dt, history=(phi_prev, None),
                                storage_flow=w_prev)
    torch.testing.assert_close(flows, k * (phi[1:] - phi_prev[1:]) - w_prev)
    # the balance of the solved state with its storage flows closes
    torch.testing.assert_close(layer.balance(phi, q, SRC, {}, storage_flow=flows),
                               torch.zeros(3), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="storage_flow"):
        layer.solve(PB, {}, SRC, dt=dt, history=(phi_prev, None))


def test_steady_solve_has_no_storage():
    layer, _ = _rc()
    plain = KirchhoffLayer(_net(), "heat", [Conductance(G.clone(), kind="g")],
                           boundary=["gnd"])
    phi, _ = layer.solve(PB, {}, SRC)
    want, _ = plain.solve(PB, {}, SRC)
    torch.testing.assert_close(phi, want, rtol=0, atol=0)
    assert torch.equal(layer.storage_flows(phi, {}, dt=None), torch.zeros(3))


def test_node_grounded_only_by_storage():
    # "d" and "e" reach no boundary: their storage grounds them during a step only
    net2 = Network(dtype=F64)
    for n in ["gnd", "a", "d", "e"]:
        net2.add_node(n)
    net2.add_edge("gnd", "a", kind="g")
    net2.add_edge("d", "e", kind="g")
    layer = KirchhoffLayer(net2, "heat", [Conductance(torch.tensor([1.0, 1.0]), kind="g")],
                           boundary=["gnd"], storage=[LinearStorage([2, 3], 0.5)])
    phi_prev = torch.tensor([0.0, 0.0, 1.0, 3.0])
    phi, _ = layer.solve(torch.zeros(1), dt=0.1, history=(phi_prev, None),
                         scheme="backward_euler")
    # the d-e pair conserves its stored amount: (phi_d + phi_e) stays 4
    torch.testing.assert_close(phi[2] + phi[3], torch.tensor(4.0))
    with pytest.raises(RuntimeError, match="floating"):
        layer.solve(torch.zeros(1))


def test_repeated_entries_at_one_node_add_up():
    net = _net()
    one = KirchhoffLayer(net, "heat", [Conductance(G.clone(), kind="g")], boundary=["gnd"],
                         storage=[LinearStorage([1, 2, 3], CAP.clone())],
                         scheme="backward_euler")
    split = KirchhoffLayer(
        net, "heat", [Conductance(G.clone(), kind="g")], boundary=["gnd"],
        storage=[LinearStorage([1, 2, 3, 2], torch.tensor([0.4, 0.6, 0.7, 0.5])),
                 LinearStorage([1], 0.0)],
        scheme="backward_euler")
    phi_prev = _phi_prev(one)
    want, _ = one.solve(PB, {}, SRC, dt=0.5, history=(phi_prev, None))
    got, _ = split.solve(PB, {}, SRC, dt=0.5, history=(phi_prev, None))
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    flows = split.storage_flows(got, {}, dt=0.5, history=(phi_prev, None))
    assert flows.shape == (5,)
    total_b = flows[1] + flows[3]
    torch.testing.assert_close(total_b, one.storage_flows(want, {}, dt=0.5,
                                                          history=(phi_prev, None))[1])


def test_construction_and_argument_refusals():
    net = _net()
    with pytest.raises(ValueError, match="not interior unknowns"):
        KirchhoffLayer(net, "heat", [Conductance(G.clone(), kind="g")], boundary=["gnd"],
                       storage=[LinearStorage([0], 1.0)])
    with pytest.raises(TypeError, match="NodeStorage"):
        KirchhoffLayer(net, "heat", [Conductance(G.clone(), kind="g")], boundary=["gnd"],
                       storage=[object()])
    # a layer solved by Modified Nodal Analysis takes storage (tests/layers/test_mna_storage.py)
    assert KirchhoffLayer(net, "heat", [Capacitance(torch.ones(3), kind="g")],
                          boundary=["gnd"],
                          storage=[LinearStorage([1], 1.0)]).implicit_edges.numel() > 0
    plain = KirchhoffLayer(net, "heat", [Conductance(G.clone(), kind="g")], boundary=["gnd"])
    with pytest.raises(ValueError, match="no storage"):
        plain.solve(PB, dt=0.1)
    with pytest.raises(ValueError, match="storage_flow"):
        plain.solve(PB, storage_flow=torch.zeros(3))
    with pytest.raises(ValueError, match="exact"):
        plain.solve(PB, scheme="exact")
    with pytest.raises(ValueError, match="expected one of"):
        plain.solve(PB, scheme="euler")
    plain.solve(PB, scheme="backward_euler")   # a valid scheme is accepted and ignored
    layer, _ = _rc()
    with pytest.raises(ValueError, match="history"):
        layer.solve(PB, dt=0.1)
    with pytest.raises(ValueError, match="positive"):
        layer.solve(PB, dt=0.0, history=(_phi_prev(layer), None))
    with pytest.raises(ValueError, match="takes no step"):
        layer.solve(PB, dt=0.1, history=(_phi_prev(layer), None), mode="dc")


def test_initial_mode_holds_the_storage_nodes_on_the_classic_path():
    """Every interior node is held at 0.5: the flows follow from the potentials, and the
    flow into storage balances each node."""
    net = _net()
    layer = KirchhoffLayer(net, "heat", [Conductance(G.clone(), kind="g")], boundary=["gnd"],
                           storage=[LinearStorage([1, 2, 3], CAP.clone(), initial=0.5)])
    assert layer.implicit_edges.numel() == 0
    phi, q = layer.solve(PB, mode="initial")
    torch.testing.assert_close(phi, torch.tensor([1.25, 0.5, 0.5, 0.5], dtype=F64),
                               rtol=0, atol=1e-14)
    torch.testing.assert_close(q, G * torch.tensor([0.75, 0.0, 0.0], dtype=F64),
                               rtol=1e-14, atol=1e-14)
    w = layer.storage_flows(phi, {}, dt=None, mode="initial", q=q)
    torch.testing.assert_close(w, torch.tensor([1.5, 0.0, 0.0], dtype=F64), rtol=1e-14,
                               atol=1e-14)
    # without an initial value, "initial" is the steady point
    plain, _ = _rc()
    plain.solve(PB, mode="initial")


def test_a_custom_storage_overrides_amount_only():
    class Cubic(NodeStorage):
        def __init__(self, nodes, c):
            super().__init__(nodes)
            self.c = torch.nn.Parameter(torch.as_tensor(c), requires_grad=False)

        def amount(self, phi_nodes, drivers=None):
            return self.c * (phi_nodes + phi_nodes ** 3 / 3)

    net = _net()
    layer = KirchhoffLayer(net, "heat", [Conductance(G.clone(), kind="g")], boundary=["gnd"],
                           storage=[Cubic([1, 2, 3], 0.8)], scheme="backward_euler")
    assert not layer.linear
    phi_prev, dt = _phi_prev(layer), 0.4
    phi, q = layer.solve(PB, {}, SRC, dt=dt, history=(phi_prev, None), atol=1e-13)
    flows = layer.storage_flows(phi, {}, dt=dt, history=(phi_prev, None))
    torch.testing.assert_close(layer.balance(phi, q, SRC, {}, storage_flow=flows),
                               torch.zeros(3), rtol=0, atol=1e-11)
    # capacity by autograd: c (1 + phi^2)
    st = layer.storage[0]
    torch.testing.assert_close(st.capacity(phi[1:], phi_prev[1:]), 0.8 * (1 + phi[1:] ** 2))


# ------------------------------------------------------------------ Model


def _model(scheme="trapezoidal", learnable=False):
    layer, storage = _rc(scheme, learnable=learnable)
    return Model(layer.net, [layer]), layer, storage


def _drivers(src=SRC):
    return {"heat.phi_boundary": PB, "heat.sources": src}


def test_model_steps_carry_storage_flow():
    model, layer, _ = _model()
    drivers = _drivers()
    state = model.steady({"heat.phi": _phi_prev(layer)}, drivers)
    assert torch.equal(state["heat.storage_flow"], torch.zeros(3))
    phi, w = state["heat.phi"], state["heat.storage_flow"]
    for _ in range(3):
        new = model.step(state, drivers, 0.25)
        want_phi, _ = layer.solve(PB, {}, SRC, dt=0.25, history=(phi, None), storage_flow=w)
        torch.testing.assert_close(new["heat.phi"], want_phi, rtol=1e-12, atol=1e-12)
        want_w = layer.storage_flows(want_phi, {}, dt=0.25, history=(phi, None),
                                     storage_flow=w)
        torch.testing.assert_close(new["heat.storage_flow"], want_w, rtol=1e-12, atol=1e-12)
        state, phi, w = new, want_phi, want_w


def test_model_takes_a_missing_storage_flow_as_zero_under_trapezoidal():
    model, layer, _ = _model()
    state = model.steady({"heat.phi": _phi_prev(layer)}, _drivers())
    want = model.step(state, _drivers(), 0.25)              # zeros after the steady solve
    del state["heat.storage_flow"]
    got = model.step(state, _drivers(), 0.25)
    assert torch.equal(got["heat.phi"], want["heat.phi"])
    assert torch.equal(got["heat.storage_flow"], want["heat.storage_flow"])
    # backward Euler reads no step-start storage flow
    model.step(state, _drivers(), 0.25, scheme="backward_euler")
    with pytest.raises(ValueError, match="storage_flow"):
        model.step(state, _drivers(), 0.25, storage_flow=torch.zeros(3))


def test_model_refs_and_validation_know_storage_flow():
    from noodl.validation import check_setup

    model, layer, _ = _model()
    field = model.refs["heat"].storage_flow
    assert field == "heat.storage_flow" and field.required == "optional"
    assert field.labels == ("a", "b", "c")
    state = model.steady({"heat.phi": _phi_prev(layer)}, _drivers())
    report = check_setup(model, state, _drivers())
    assert not [i for i in report.errors + report.warnings if "storage_flow" in str(i)]
    del state["heat.storage_flow"]                          # zero when missing
    report = check_setup(model, state, _drivers())
    assert not [i for i in report.errors if "storage_flow" in str(i)]


def test_gradcheck_through_two_model_steps():
    model, layer, storage = _model(learnable=True)
    phi_start = _phi_prev(layer).clone().requires_grad_(True)
    src = SRC.clone().requires_grad_(True)

    def two_steps(phi0, C, s):
        drivers = _drivers(s)
        state = {"heat.phi": phi0, "heat.q": torch.zeros(3),
                 "heat.storage_flow": torch.zeros(3)}
        for _ in range(2):
            state = model.step(state, drivers, 0.3)
        return torch.cat([state["heat.phi"], state["heat.storage_flow"]])

    assert torch.autograd.gradcheck(two_steps, (phi_start, storage.C, src))


def test_residual_jacobian_and_adjoint_include_the_step_storage():
    layer, _ = _rc()
    phi_prev, dt = _phi_prev(layer), 0.5
    step = dict(dt=dt, history=(phi_prev, None))
    phi, _ = layer.solve(PB, {}, SRC, **step)
    # the residual solve drives to zero includes the storage withdrawal of the step
    r = layer.residual(phi[1:], PB, {}, SRC, **step)
    torch.testing.assert_close(r, torch.zeros(3), rtol=0, atol=1e-12)
    # without dt it is the steady residual: off by exactly the storage withdrawal
    flows = layer.storage_flows(phi, {}, **step)
    torch.testing.assert_close(layer.residual(phi[1:], PB, {}, SRC), -flows,
                               rtol=0, atol=1e-12)
    L_II, _ = _laplacian(layer)
    want = L_II + torch.diag(CAP / dt)
    torch.testing.assert_close(layer.jacobian(phi[1:], PB, {}, **step), want)
    torch.testing.assert_close(layer.jacobian(phi[1:], PB, {}), L_II)
    g = torch.tensor([1.0, -2.0, 0.5])
    lam = layer.adjoint(phi[1:], PB, {}, g, **step)
    torch.testing.assert_close(lam, torch.linalg.solve(want.T, g), rtol=1e-10, atol=1e-12)
    plain = KirchhoffLayer(_net(), "heat", [Conductance(G.clone(), kind="g")],
                           boundary=["gnd"])
    with pytest.raises(ValueError, match="no storage"):
        plain.residual(phi[1:], PB, {}, SRC, dt=dt)


def test_model_step_of_a_storage_layer_does_not_need_q():
    model, layer, _ = _model("backward_euler")
    state = {"heat.phi": _phi_prev(layer)}
    new = model.step(state, _drivers(), 0.25)
    want, _ = layer.solve(PB, {}, SRC, dt=0.25, history=(_phi_prev(layer), None))
    torch.testing.assert_close(new["heat.phi"], want, rtol=1e-12, atol=1e-12)
    assert model.refs["heat"].q.required == "optional"


# ------------------------------------------------------------------ boundary storage
# Under scheme="forward_euler" a storage sits at a boundary node (an EPANET tank): the solve
# holds it at its step-start potential, and the step then moves it by dt * inflow / C.

TANK_C = torch.tensor([0.8], dtype=F64)


def _tank_net():
    return _net(["gnd", "a", "b", "t"], (("gnd", "a"), ("a", "b"), ("b", "t")))


def _tank_layer(storage=None, scheme="forward_euler"):
    storage = LinearStorage([3], TANK_C.clone()) if storage is None else storage
    return KirchhoffLayer(_tank_net(), "w", [Conductance(G.clone(), kind="g")],
                          boundary=["gnd", "t"], storage=[storage], scheme=scheme)


def test_boundary_storage_advances_by_dt_inflow_over_capacity_exactly():
    layer = _tank_layer()
    assert not layer.dynamic and len(layer.storage) == 1
    model = Model(layer.net, [layer])
    # the driver gives the other boundary node only: the pass holds the tank at the state's 2.0
    drivers = {"w.phi_boundary": torch.tensor([1.25]),
               "w.sources": torch.tensor([0.0, 0.3, -0.2, 0.0])}
    state = {"w.phi": torch.tensor([1.25, 0.0, 0.0, 2.0])}
    dt = 0.7
    new = model.step(state, drivers, dt)
    held, _ = layer.solve(torch.tensor([1.25]), {}, drivers["w.sources"],
                          history=(state["w.phi"], None))
    torch.testing.assert_close(new["w.phi"][:3], held[:3], rtol=0, atol=1e-14)
    # the flow into the tank is what the step's own flows deliver to it, and the tank
    # moves by exactly dt * inflow / C
    inflow = -layer.net_outflow(new["w.q"])[..., [3]]
    torch.testing.assert_close(new["w.storage_flow"], inflow, rtol=0, atol=0)
    assert torch.equal(new["w.phi"][3], 2.0 + dt * inflow[0] / TANK_C[0])
    assert float(inflow[0]) != 0.0
    # a steady solve holds the tank and reports the same inflow, moving nothing
    steady = model.steady(state, drivers)
    assert float(steady["w.phi"][3]) == 2.0
    torch.testing.assert_close(steady["w.storage_flow"], inflow, rtol=0, atol=1e-14)
    # the layer's own step from a solve, as `noodl.apps.water.water_run` takes it
    solved = dict(new, **{"w.phi": new["w.phi"].clone()})
    solved["w.phi"][3] = 2.0
    torch.testing.assert_close(layer.advance_storage(solved["w.phi"], inflow, dt),
                               new["w.phi"], rtol=0, atol=0)
    assert model.refs["w"].phi.required == "always"
    assert "w.phi" in required_state_keys(model)


def test_boundary_storage_gradient_reaches_the_capacity():
    storage = LinearStorage([3], TANK_C.clone(), learnable=True)
    layer = _tank_layer(storage)
    phi, q = layer.solve(torch.tensor([1.25]), {}, torch.tensor([0.0, 0.3, -0.2, 0.0]),
                         history=(torch.tensor([1.25, 0.0, 0.0, 2.0]), None))
    inflow = layer.storage_flows(phi, {}, dt=None, q=q)
    layer.advance_storage(phi, inflow, 0.7)[3].backward()
    torch.testing.assert_close(storage.C.grad, -0.7 * inflow.detach() / TANK_C ** 2)


def test_a_nonlinear_boundary_storage_without_an_inverse_is_refused():
    class Cubic(NodeStorage):
        def amount(self, phi_nodes, drivers=None):
            return phi_nodes + phi_nodes ** 3 / 3

    assert Cubic([3]).potential_of(torch.ones(1)) is None
    with pytest.raises(ValueError, match="Cubic at boundary nodes is not linear and"):
        _tank_layer(Cubic([3]))


def test_linear_storage_advance_is_unchanged():
    """`LinearStorage` moves a boundary node by `dt * flow / C`, the arithmetic it had before
    `potential_of` existed, bit for bit (not `(C phi + dt flow) / C`)."""
    storage = LinearStorage([3], torch.tensor([0.37], dtype=F64))
    assert storage.potential_of(torch.ones(1)) is None
    layer = _tank_layer(storage)
    phi = torch.tensor([1.25, 0.1, 0.2, 2.0 / 3.0], dtype=F64)
    flow = torch.tensor([0.123456789], dtype=F64)
    for dt in (0.7, 3600.0, 1e-3):
        out = layer.advance_storage(phi, flow, dt)
        assert torch.equal(out[:3], phi[:3])
        assert torch.equal(out[3], phi[3] + dt * flow[0] / storage.C[0])


class _SquareRoot(NodeStorage):
    """S = phi^2 (phi > 0), with its inverse: a storage whose capacity changes."""

    def amount(self, phi_nodes, drivers=None):
        return phi_nodes ** 2

    def potential_of(self, amount, drivers=None):
        return amount.sqrt()


def test_a_potential_of_that_returns_none_is_refused_by_name():
    class Declines(_SquareRoot):
        def potential_of(self, amount, drivers=None):
            return None

    layer = _tank_layer(Declines([3]))
    with pytest.raises(ValueError, match=r"Declines at boundary nodes \[3\] is not linear "
                                         "and its potential_of returned None"):
        layer.advance_storage(torch.tensor([1.25, 0.1, 0.2, 2.0], dtype=F64),
                              torch.tensor([0.3], dtype=F64), 0.5)


def test_a_storage_with_an_inverse_advances_by_its_amount():
    """A boundary storage that gives `potential_of` moves its node to
    `potential_of(amount(start) + dt * flow)`: the amount changes by exactly dt * flow."""
    layer = _tank_layer(_SquareRoot([3]))
    phi = torch.tensor([1.25, 0.1, 0.2, 2.0], dtype=F64)
    flow = torch.tensor([0.3], dtype=F64)
    out = layer.advance_storage(phi, flow, 0.5)
    assert float(out[3]) == pytest.approx((4.0 + 0.15) ** 0.5, rel=1e-15)
    assert torch.equal(out[:3], phi[:3])
    # differentiable in the flow: d phi / d flow = dt / (2 phi_new)
    flow.requires_grad_(True)
    layer.advance_storage(phi, flow, 0.5)[3].backward()
    assert float(flow.grad[0]) == pytest.approx(0.5 / (2 * (4.15) ** 0.5), rel=1e-12)


def test_forward_euler_refusals():
    # storage at an interior node refuses forward_euler
    with pytest.raises(ValueError, match=r"\['a'\], which are not boundary nodes"):
        _tank_layer(LinearStorage([1], TANK_C.clone()))
    # a storage at a boundary node refuses the implicit schemes
    with pytest.raises(ValueError, match="unless the layer steps it with scheme='forward_euler'"):
        _tank_layer(scheme="backward_euler")
    # one entry per node
    with pytest.raises(ValueError, match="given twice"):
        KirchhoffLayer(_tank_net(), "w", [Conductance(G.clone(), kind="g")],
                       boundary=["gnd", "t"], scheme="forward_euler",
                       storage=[LinearStorage([3], TANK_C), LinearStorage([3], TANK_C)])
    # a per-call scheme other than forward_euler on boundary storage
    layer = _tank_layer()
    with pytest.raises(ValueError, match="step by scheme='forward_euler' only"):
        layer.solve(torch.tensor([1.25]), {}, None, scheme="backward_euler")
    # forward_euler on a layer that steps interior storage
    rc, _ = _rc()
    with pytest.raises(ValueError, match="boundary nodes only"):
        rc.solve(PB, {}, SRC, dt=0.3, history=(_phi_prev(rc), None), scheme="forward_euler")
    with pytest.raises(ValueError, match="no storage at boundary nodes"):
        rc.advance_storage(_phi_prev(rc), torch.zeros(3), 0.3)
    # a transport layer takes no forward_euler
    from noodl.layers.transport import TransportLayer
    with pytest.raises(ValueError, match="unknown scheme 'forward_euler'"):
        TransportLayer(_net(), "x", capacity=torch.ones(3), flow_kind="g", boundary=["gnd"],
                       scheme="forward_euler")
    # the pass needs the step-start potentials it holds the storage at
    model = Model(layer.net, [layer])
    with pytest.raises(KeyError, match="held at its potential at the step start"):
        model.steady({}, {"w.phi_boundary": torch.tensor([1.25])})


def test_the_layer_holds_boundary_storage_on_every_model_path():
    """`steady`, `step`, `residuals` and `current_flows` all hold the tank at the state's
    potential: the driver gives the other boundary node only."""
    layer = _tank_layer()
    model = Model(layer.net, [layer])
    drivers = {"w.phi_boundary": torch.tensor([1.25]),
               "w.sources": torch.tensor([0.0, 0.3, -0.2, 0.0])}
    steady = model.steady({"w.phi": torch.tensor([1.25, 0.0, 0.0, 2.0])}, drivers)
    assert layer.holds_storage and float(steady["w.phi"][3]) == 2.0
    residual = model.residuals(steady, drivers)["w"]
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=1e-12)
    report = model.check(steady, drivers, steady=True)
    assert report.ok, str(report)


def test_a_direct_call_without_history_is_refused_by_name():
    """The layer reads the held potentials from `history` only: a direct `solve`,
    `residual` or `linear_init` without it is refused, and so is a `phi_boundary` that
    still gives the held node."""
    layer = _tank_layer()
    src = torch.tensor([0.0, 0.3, -0.2, 0.0])
    start = torch.tensor([1.25, 0.0, 0.0, 2.0])
    with pytest.raises(ValueError, match=r"holds its storage nodes \['t'\].*pass history"):
        layer.solve(torch.tensor([1.25]), {}, src)
    with pytest.raises(ValueError, match="pass history"):
        layer.residual(torch.zeros(2), torch.tensor([1.25]), {}, src)
    with pytest.raises(ValueError, match="pass history"):
        layer.linear_init(torch.tensor([1.25]), {}, src)
    with pytest.raises(ValueError, match=r"boundary nodes \['gnd'\] only \(1\)"):
        layer.solve(torch.tensor([1.25, 2.0]), {}, src, history=(start, None))
    phi, _ = layer.solve(torch.tensor([1.25]), {}, src, history=(start, None))
    assert float(phi[3]) == 2.0
    r = layer.residual(phi[layer.interior], torch.tensor([1.25]), {}, src,
                       history=(start, None))
    torch.testing.assert_close(r, torch.zeros_like(r), rtol=0, atol=1e-12)


def test_forward_euler_without_storage_is_refused():
    with pytest.raises(ValueError, match="this layer has none"):
        KirchhoffLayer(_tank_net(), "w", [Conductance(G.clone(), kind="g")],
                       boundary=["gnd", "t"], scheme="forward_euler")

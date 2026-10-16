"""Water: ensemble equivalence, warm-start independence and finite-difference gradients.

The constant-tracer invariant (every device, pda) lives in `test_network.py`.
"""

import pytest
import torch

from noodl.apps.water.network import build_model, twoloop, water_steady

F64 = torch.float64
# twoloop: 7 nodes, 6 interior (junction) nodes, 8 pipes: every count is a batch size below.
BATCH = [1, 6, 7, 8]


def _case(**kw):
    model, state, drivers = build_model(twoloop(), quality=0.5, **kw)
    drivers["quality.x_boundary"] = torch.tensor([1.0], dtype=F64)
    return model, state, drivers


def _batched(drivers, b, what):
    d = dict(drivers)
    scale = torch.linspace(0.6, 1.4, b, dtype=F64)
    if what in ("demand", "both"):
        d["water.sources"] = drivers["water.sources"] * scale[:, None]
    if what in ("head", "both"):
        d["water.phi_boundary"] = drivers["water.phi_boundary"] * (1.0 + 0.1 * scale[:, None])
    return d


@pytest.mark.parametrize("b", BATCH)
@pytest.mark.parametrize("what", ["demand", "head", "both"])
def test_batched_steady_equals_stack_of_single_solves(b, what):
    """Batched demands/reservoir heads with an UNBATCHED initial state, hydraulics and quality."""
    model, state, drivers = _case()
    d = _batched(drivers, b, what)
    out = water_steady(model, state, d)
    for i in range(b):
        di = {k: (v[i] if v.dim() > drivers[k].dim() else v) for k, v in d.items()}
        single = water_steady(model, state, di)
        for key in ("water.phi", "water.q", "quality.x"):
            full = out[key].expand(b, -1) if out[key].dim() == 1 else out[key]
            torch.testing.assert_close(full[i], single[key], rtol=1e-7, atol=1e-9,
                                       msg=f"{key}[{i}]")


def test_pressure_driven_batch_equals_stack_of_single_solves():
    model, state, drivers = _case(pda=True, p_req=60.0)
    b = 3
    d = dict(drivers, **{"water.phi_boundary": drivers["water.phi_boundary"]
                         * torch.tensor([[0.9], [1.0], [1.1]], dtype=F64)})
    out = water_steady(model, state, d)
    for i in range(b):
        di = dict(d, **{"water.phi_boundary": d["water.phi_boundary"][i]})
        single = water_steady(model, state, di)
        torch.testing.assert_close(out["water.phi"][i], single["water.phi"], rtol=1e-7,
                                   atol=1e-8)


@pytest.mark.parametrize("guess", ["zeros", "random", "high"])
def test_steady_is_independent_of_the_initial_guess(guess):
    model, state, drivers = _case()
    ref = water_steady(model, state, drivers)
    g = {"zeros": torch.zeros(7, dtype=F64),
         "random": 30.0 * torch.rand(7, dtype=F64, generator=torch.Generator().manual_seed(1)),
         "high": torch.full((7,), 200.0, dtype=F64)}[guess]
    out = water_steady(model, dict(state, **{"water.phi": g}), drivers)
    torch.testing.assert_close(out["water.phi"], ref["water.phi"], rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(out["quality.x"], ref["quality.x"], rtol=1e-7, atol=1e-9)


def test_head_gradient_wrt_a_demand_matches_central_differences():
    """d(head at J4)/d(demand at J2) by autograd, from a cold and a warm start, equals the
    central difference (Hazen-Williams is smooth away from zero flow)."""
    model, state, drivers = build_model(twoloop())
    j2 = 1  # twoloop node order: J1..J6, R1
    j4 = 3

    def head(demand_scale, st):
        src = drivers["water.sources"].clone()
        src = torch.cat([src[:j2], (src[j2:j2 + 1] * demand_scale), src[j2 + 1:]])
        return water_steady(model, st, dict(drivers, **{"water.sources": src}))["water.phi"][j4]

    ref = water_steady(model, state, drivers)
    for st in (state, dict(state, **{"water.phi": ref["water.phi"].detach()})):
        s = torch.tensor(1.0, dtype=F64, requires_grad=True)
        head(s, st).backward()
        h = 1e-5
        with torch.no_grad():
            fd = (head(torch.tensor(1 + h, dtype=F64), st)
                  - head(torch.tensor(1 - h, dtype=F64), st)) / (2 * h)
        assert s.grad.item() == pytest.approx(fd.item(), rel=1e-5)


@pytest.mark.parametrize("pda", [False, True])
def test_constant_tracer_stays_constant_when_the_withdrawals_change(pda):
    """A conserved tracer fed at 100 reads 100 everywhere whatever the demands are: the
    quality sink follows the CURRENT withdrawals, not the build-time ones (including a
    scaled `water.sources` and a batch of different demand patterns)."""
    kw = {"pda": True, "p_req": 100.0} if pda else {}
    model, state, drivers = build_model(twoloop(), quality=0.0, **kw)
    drivers["quality.x_boundary"] = torch.tensor([100.0], dtype=torch.float64)
    if pda:
        out = water_steady(model, state, drivers)["quality.x"]
        assert torch.allclose(out, torch.full_like(out, 100.0), atol=1e-6)
        return
    for scale in (0.5, 1.0, 1.7):
        d = dict(drivers)
        d["water.sources"] = drivers["water.sources"] * scale
        out = water_steady(model, state, d)["quality.x"]
        assert torch.allclose(out, torch.full_like(out, 100.0), atol=1e-6), (scale, out)
    pattern = torch.tensor([0.3, 1.0, 1.4], dtype=torch.float64)
    d = dict(drivers)
    d["water.sources"] = pattern[:, None] * drivers["water.sources"][None, :]
    out = water_steady(model, state, d)["quality.x"]
    assert torch.allclose(out, torch.full_like(out, 100.0), atol=1e-6)


def _dead_end(pda):
    from noodl.apps.water.network import Junction, Reservoir, WaterNetwork, WaterPipe

    net = WaterNetwork(
        junctions=(Junction("A", 0.0, 0.005), Junction("B", 80.0, 0.002)),
        reservoirs=(Reservoir("R", 50.0),),
        pipes=(WaterPipe("P1", "R", "A", 500.0, 0.3, 130.0),
               WaterPipe("P2", "A", "B", 500.0, 0.3, 130.0)))
    return net, ({"pda": True, "p_req": 20.0} if pda else {})


def test_a_stagnant_dead_end_without_decay_is_refused_by_name_and_with_decay_reads_zero():
    """A node with no inflow, no outflow and no decay has no steady concentration: refused
    naming the node and the cause (as the layer does for any stagnant node). With decay the
    stagnant node's steady concentration is 0."""
    from noodl.layers.transport import UndefinedSteadyState

    net, kw = _dead_end(pda=True)
    model, state, drivers = build_model(net, quality=0.0, **kw)
    drivers["quality.x_boundary"] = torch.tensor([100.0], dtype=torch.float64)
    with pytest.raises(UndefinedSteadyState, match="STAGNANT") as info:
        water_steady(model, state, drivers)
    assert info.value.nodes == ["B"]
    model, state, drivers = build_model(net, quality=0.5, **kw)
    drivers["quality.x_boundary"] = torch.tensor([100.0], dtype=torch.float64)
    out = water_steady(model, state, drivers)["quality.x"]
    assert float(out[1]) == pytest.approx(0.0, abs=1e-9)
    assert float(out[0]) > 50.0


def _injected_line():
    """R -> A -> B: A has no demand and takes a water injection, B withdraws 0.01 m3/s."""
    from noodl.apps.water.network import Junction, Reservoir, WaterNetwork, WaterPipe

    return WaterNetwork(
        junctions=(Junction("A", 0.0, 0.0), Junction("B", 0.0, 0.01)),
        reservoirs=(Reservoir("R", 50.0),),
        pipes=(WaterPipe("P1", "R", "A", 100.0, 0.3, 130.0),
               WaterPipe("P2", "A", "B", 100.0, 0.3, 130.0)))


def _tracer_balance(model, drivers, out):
    """`(withdrawn, supplied)` tracer mass rates (kg/s): the withdrawals times their
    concentrations against the reservoir's outflow at its concentration plus the injected
    tracer."""
    names = model.node_names
    junctions = [names.index(j.name) for j in model.water_network.junctions]
    sources = drivers["water.sources"]
    withdrawal = torch.where(sources < 0, -sources, torch.zeros_like(sources))[..., junctions]
    withdrawn = (withdrawal * out["quality.x"]).sum(-1)
    supplied = out["water.q"][..., 0] * drivers["quality.x_boundary"][..., 0]
    supplied = supplied + drivers["quality.sources"].sum(-1)
    return withdrawn, supplied


@pytest.mark.parametrize(("tracer", "expected"), [(0.1, 60.0), (0.0, 50.0)])
def test_a_water_injection_dilutes_and_the_injected_tracer_is_conserved(tracer, expected):
    """A water injection is not a negative withdrawal: it carries no tracer of its own (only
    `quality.sources` does) and so dilutes. R at 100 supplies 0.005 m3/s, A injects 0.005
    m3/s and `tracer` kg/s, B withdraws 0.01: both read (0.005 * 100 + tracer) / 0.01, and
    the tracer withdrawn equals the tracer supplied."""
    from noodl.apps.water.network import initial_drivers

    model, state, _ = build_model(_injected_line(), quality=0.0)
    drivers = initial_drivers(model, values={
        "water.sources": {"A": 0.005},
        "quality.x_boundary": {"R": 100.0},
        "quality.sources": {"A": tracer},
    })
    out = water_steady(model, state, drivers)
    assert torch.allclose(out["water.q"], torch.tensor([0.005, 0.01], dtype=F64), atol=1e-9)
    assert torch.allclose(out["quality.x"], torch.full((2,), expected, dtype=F64), atol=1e-6)
    withdrawn, supplied = _tracer_balance(model, drivers, out)
    assert float(withdrawn) == pytest.approx(float(supplied), rel=1e-9)
    assert float(supplied) == pytest.approx(0.01 * expected, rel=1e-9)


def test_a_batch_of_mixed_injections_conserves_the_tracer_member_by_member():
    """The same balance over a batch of water injections (some of them withdrawals) and
    injected tracers, each member equal to its own single solve."""
    from noodl.apps.water.network import initial_drivers

    model, state, _ = build_model(_injected_line(), quality=0.0)
    base = initial_drivers(model, values={
        "quality.x_boundary": {"R": 100.0}, "quality.sources": {"A": 0.0}})
    names = model.node_names
    a = names.index("A")
    inject = torch.tensor([0.005, 0.0, -0.002, 0.008], dtype=F64)
    tracer = torch.tensor([0.1, 0.0, 0.05, 0.2], dtype=F64)
    d = dict(base)
    d["water.sources"] = base["water.sources"].repeat(4, 1)
    d["water.sources"][:, a] = inject
    d["quality.sources"] = base["quality.sources"].repeat(4, 1)
    d["quality.sources"][:, a] = tracer
    d["quality.x_boundary"] = base["quality.x_boundary"].repeat(4, 1)
    out = water_steady(model, state, d)
    withdrawn, supplied = _tracer_balance(model, d, out)
    assert torch.allclose(withdrawn, supplied, rtol=1e-9, atol=1e-12)
    for k in range(4):
        dk = {key: (v[k] if key in ("water.sources", "quality.sources", "quality.x_boundary")
                    else v) for key, v in d.items()}
        single = water_steady(model, state, dk)
        assert torch.allclose(out["quality.x"][k], single["quality.x"], atol=1e-9)

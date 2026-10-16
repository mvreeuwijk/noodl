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

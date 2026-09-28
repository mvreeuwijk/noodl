"""The MBL laws' analytic slopes (`dflow` under `no_grad`) against the autograd default.

`door._power_law_slope_at` and `_MBLDoorCompartmentBase.stream_slope` apply the chain rule in
the order reverse-mode autograd applies to the laws' forward code, so the Newton Jacobian is
the same bit for bit as with the autograd `dflow` (and the airflow solves, and every result
downstream, are unchanged). Checked here with `torch.equal` over the band, the band edges,
zero and the sharp law, batched and unbatched; plus the memo's invalidation rules.
"""

from __future__ import annotations

import pytest
import torch

from noodl.elements.base import Element, memo
from noodl.elements.door import MBLDoorOpen
from noodl.elements.door_discretized import (
    MBLDoorPortStream,
    mbl_discretized_door,
    mbl_discretized_operable_door,
)
from noodl.elements.media import medium
from noodl.elements.powerlaw_mbl import MBLPowerLaw, mbl_orifice

F64 = torch.float64
MED = medium("Buildings.Media.Air")


def _samples(dp_turbulent: float, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    parts = [torch.randn(2000, n, generator=g, dtype=F64) * s * dp_turbulent
             for s in (0.3, 3.0, 300.0)]
    edges = torch.tensor([0.0, -0.0, dp_turbulent, -dp_turbulent, 1e-300], dtype=F64)
    parts.append(edges.unsqueeze(-1).expand(-1, n))
    return torch.cat(parts)


@pytest.mark.parametrize("element", [
    mbl_orifice(0.37, rho_default=1.2),
    mbl_orifice(0.02, m=0.65, rho_default=1.19, dp_turbulent=0.01),
    MBLPowerLaw(C=1.3e-3, m=0.66, dp_turbulent=0.1, form="mass", rho_default=1.2),
    MBLDoorOpen(direction="ab", src=[0], tgt=[1], medium=MED),
    MBLDoorOpen(direction="ba", src=[0, 1], tgt=[1, 2], medium=MED, wOpe=[0.9, 1.3]),
])
def test_power_law_slopes_equal_autograd_bit_for_bit(element):
    drivers = {"T": torch.tensor([293.15, 295.0, 290.0], dtype=F64)}
    n = element.src.numel() if hasattr(element, "src") else 1
    dp = _samples(element.dp_turbulent, n, 0)
    with torch.no_grad():
        exact = element.dflow(dp, drivers)
        reference = Element.dflow(element, dp, drivers)
    assert torch.equal(exact, reference)


@pytest.mark.parametrize("operable", [False, True])
@pytest.mark.parametrize("y", [0.0, 0.3, 1.0])
def test_discretised_door_stream_slope_equals_autograd_bit_for_bit(operable, y):
    if operable:
        comp, head = mbl_discretized_operable_door(src=0, tgt=1, medium=MED, y_key="y",
                                                   LClo=0.01)
    else:
        comp, head = mbl_discretized_door(src=0, tgt=1, medium=MED, nCom=6)
    stream = MBLDoorPortStream(comp, head, "ab", "door")
    drivers = {"T": torch.tensor([293.15, 296.0], dtype=F64),
               "p_abs": torch.tensor([101325.0, 101321.0], dtype=F64),
               "X_w": torch.full((2,), 0.01, dtype=F64), "y": torch.tensor(y, dtype=F64)}
    for dp in (_samples(0.01, 1, 1), _samples(0.01, 1, 2)[0]):
        with torch.no_grad():
            exact = stream.dflow(dp, drivers)
            assert comp.stream_slope(dp + head(drivers), drivers) is not None
        x = dp.clone().requires_grad_(True)
        mAB, mBA = stream._streams(x, drivers)
        (grad,) = torch.autograd.grad((mAB - mBA).sum(), x)
        assert torch.equal(exact, 0.5 * grad)


@pytest.mark.parametrize("operable", [False, True])
def test_stream_slope_at_the_smooth_heaviside_band_edges(operable):
    """Compartments whose `smoothHeaviside` argument sits on its clamp (u exactly 0 or 1:
    autograd's clamp gradient is 0 there), found by scanning `dp` finely
    across the band edges `dV = +-VZerCom_flow`, with the two sides at different densities."""
    if operable:
        comp, _ = mbl_discretized_operable_door(src=0, tgt=1, medium=MED, y_key="y",
                                                LClo=0.01)
    else:
        comp, _ = mbl_discretized_door(src=0, tgt=1, medium=MED)
    drivers = {"T": torch.tensor([293.15, 297.0], dtype=F64),
               "p_abs": torch.tensor([101325.0, 101325.0], dtype=F64),
               "X_w": torch.full((2,), 0.01, dtype=F64), "y": torch.tensor(1.0, dtype=F64)}
    VZ = float(comp._law(drivers)[2].reshape(-1)[0])
    n = comp.src.numel()

    def dV_at(dp: float) -> float:
        return float(comp._volume_flow(torch.full((n,), dp, dtype=F64), drivers)[0][0])

    lo, hi = 0.0, 10.0
    for _ in range(200):  # dV = VZerCom_flow (u = 1) by bisection; dV is increasing in dp
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if dV_at(mid) < VZ else (lo, mid)
    dp_edge = hi
    steps = torch.linspace(-3e-5, 3e-5, 8001, dtype=F64)  # 1 - u ~ (dx - 1/2)^3: u = 1 over ~1e-5
    hits = 0
    for sign in (1.0, -1.0):
        dpi = (sign * dp_edge * (1.0 + steps)).unsqueeze(-1).expand(-1, comp.src.numel())
        with torch.no_grad():
            exact = comp.stream_slope(dpi.contiguous(), drivers)
        x = dpi.contiguous().clone().requires_grad_(True)
        mAB, mBA = comp.port_flows(x, drivers)
        (grad,) = torch.autograd.grad((mAB - mBA).sum(), x)
        assert torch.equal(exact, grad)
        dV = comp._volume_flow(dpi, drivers)[0]
        u = 0.5 + 0.5 * dV / VZ * (1.875 + (0.5 * dV / VZ) ** 2 * (-5 + 6 * (0.5 * dV / VZ) ** 2))
        hits += int(((u == 0.0) | (u == 1.0)).sum())
    assert hits > 0  # the scan reaches the clamp's bounds


def test_slopes_under_grad_mode_are_autograd_and_differentiable():
    orifice = mbl_orifice(0.37, rho_default=1.2, learnable=True)
    dp = torch.linspace(-2.0, 2.0, 9, dtype=F64)
    slope = orifice.dflow(dp)
    (g,) = torch.autograd.grad(slope.sum(), orifice.C)
    assert torch.isfinite(g) and g != 0


class _Owner:
    pass


def test_memo_recomputes_on_an_in_place_change_or_a_new_object():
    owner, calls = _Owner(), []
    t = torch.ones(3, dtype=F64)

    def fn():
        calls.append(1)
        return t * 2

    first = memo(owner, "x", (t,), fn)
    assert memo(owner, "x", (t,), fn) is first and len(calls) == 1
    t.add_(1.0)  # in place: a new version
    assert torch.equal(memo(owner, "x", (t,), fn), t * 2) and len(calls) == 2
    memo(owner, "x", (t.clone(),), fn)  # an equal but different object
    assert len(calls) == 3


def test_memo_never_keeps_a_graph():
    owner, calls = _Owner(), []
    p = torch.ones(2, dtype=F64, requires_grad=True)

    def fn():
        calls.append(1)
        return p * 3

    memo(owner, "x", (p,), fn)
    memo(owner, "x", (p,), fn)
    assert len(calls) == 2

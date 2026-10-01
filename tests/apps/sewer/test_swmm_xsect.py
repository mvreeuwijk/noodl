"""SWMM 5.2's circular cross-section (`swmm_xsect`) and its `geometry="swmm"` wiring.

Parity against the engine itself is in `tests/verification/test_sewer_parity.py`; these
tests pin the transcription's own properties without pyswmm.
"""

from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import swmm_xsect as sx
from noodl.apps.sewer.inp import read_swmm_inp
from noodl.apps.sewer.network import build_model, tree_steady

F64 = torch.float64
DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"


def test_tables_are_swmm_5_2_verbatim():
    for table in (sx.A_CIRC, sx.Y_CIRC, sx.S_CIRC):
        assert len(table) == 51
        assert table[0] == 0.0 and table[-1] == 1.0
    # Spot values from `xsect.dat`, including the section-factor maximum third from last.
    assert sx.A_CIRC[25] == 0.5 and sx.Y_CIRC[25] == 0.5
    assert sx.S_CIRC[48] == 1.08208 and max(sx.S_CIRC) == sx.S_CIRC[48]


def test_lookup_hits_the_nodes_and_inverts_on_the_increasing_range():
    x = torch.arange(2, 49, dtype=F64) / 50.0
    table = torch.tensor(sx.S_CIRC, dtype=F64)
    y = sx.lookup(x, table)
    assert torch.allclose(y, table[2:49], rtol=0, atol=1e-15)
    # Up to S/Sfull < 1: `circ_getAofS` returns aFull at psi >= 1 before ever inverting.
    between = torch.linspace(0.041, 0.859, 97, dtype=F64)
    back = sx.inv_lookup(sx.lookup(between, table), table)
    assert torch.allclose(back, between, rtol=0, atol=1e-14)


def test_lookup_quadratic_correction_on_the_first_two_segments():
    # SWMM adds (x - x0)(x - x1)/delta^2 (t0/2 - t1 + t2/2) below x = 0.04 (`lookup`).
    table = torch.tensor(sx.A_CIRC, dtype=F64)
    x = torch.tensor([0.01], dtype=F64)
    linear = 0.5 * table[1]
    quad = linear + (0.01) * (0.01 - 0.02) / 0.02**2 * (table[0] / 2 - table[1] + table[2] / 2)
    assert float(sx.lookup(x, table)) == pytest.approx(float(quad), rel=1e-15)


def test_near_invert_closed_forms_are_inverses_only_to_newton_accuracy():
    """`getAcircular` and `getScircular` solve the same relation by two different Newton
    iterations, each stopping after the first step with |d| <= 1e-4 in theta. The next
    error is at most K d^2 with K = |f''/2f'| = O(1/theta), and theta > 0.1 for
    psi >= 1e-5, so the round trip holds to ~1e-6 relative and no better -- which is why SWMM's
    inlet and outlet areas differ near the invert and `kinwave_steady` carries both."""
    psi = torch.tensor([1e-5, 1e-4, 1e-3, 5e-3, 0.01], dtype=F64)
    assert torch.allclose(sx.s_circular(sx.a_circular(psi)), psi, rtol=1e-6, atol=0)


def _pipe(**kw):
    base = {"diameter": 0.45, "roughness": 0.013, "slope": 0.005}
    base.update(kw)
    return {k: torch.tensor([v], dtype=F64, requires_grad=True) for k, v in base.items()}


@pytest.mark.parametrize("fill", [0.004, 0.37])   # near-invert closed form, and the table
def test_kinwave_steady_gradients_against_central_differences(fill):
    p = _pipe()
    q_full = sx.full_flow(p["diameter"], p["roughness"], p["slope"]).detach()
    q = (fill * q_full).clone().requires_grad_(True)
    inputs = [q, p["diameter"], p["roughness"], p["slope"]]
    for output in range(3):
        value = sx.kinwave_steady(*inputs)[output]
        grads = torch.autograd.grad(value.sum(), inputs)
        for i, x in enumerate(inputs):
            h = 1e-6 * float(x.detach())
            plus = [y.detach().clone() for y in inputs]
            minus = [y.detach().clone() for y in inputs]
            plus[i] += h
            minus[i] -= h
            fd = (float(sx.kinwave_steady(*plus)[output]) -
                  float(sx.kinwave_steady(*minus)[output])) / (2 * h)
            assert float(grads[i]) == pytest.approx(fd, rel=1e-4, abs=1e-12), (output, i)


def test_dry_pipe_is_zero_with_finite_gradients():
    p = _pipe()
    q = torch.zeros(1, dtype=F64, requires_grad=True)
    h, a, v = sx.kinwave_steady(q, p["diameter"], p["roughness"], p["slope"])
    assert float(h.detach()) == 0.0 and float(a.detach()) == 0.0 and float(v.detach()) == 0.0
    grads = torch.autograd.grad((h + a + v).sum(), [q, *p.values()])
    assert all(torch.isfinite(g).all() for g in grads)


def test_velocity_is_zero_at_or_below_a_hundredth_of_a_foot():
    # `link_getVelocity` returns 0 for depth <= 0.01 ft; just above it, q / A(depth).
    p = _pipe(diameter=1.05)
    q_full = sx.full_flow(p["diameter"], p["roughness"], p["slope"]).detach()
    q = torch.logspace(-7, -3, 200, dtype=F64) * q_full
    h, _, v = sx.kinwave_steady(q, p["diameter"].detach(), p["roughness"].detach(),
                                p["slope"].detach())
    shallow = h <= 0.01 * sx.LCF
    assert shallow.any() and (~shallow).any()
    assert torch.all(v[shallow] == 0.0) and torch.all(v[~shallow] > 0.0)


def test_read_swmm_inp_defaults_to_swmm_geometry():
    assert read_swmm_inp(DATA / "tree_kinwave.inp")[0].geometry == "tabulated"
    assert read_swmm_inp(DATA / "tree_kinwave.inp", geometry="analytic")[0].geometry == (
        "analytic"
    )
    assert tree_steady().geometry == "analytic"


def test_swmm_geometry_changes_depth_by_the_table_difference_only():
    """The two geometries share the flows exactly and differ in depth by SWMM's table
    error, which is below 1e-2 relative (Ref. Man. Vol. II section 5.1.3)."""
    out = {}
    for g in ("analytic", "swmm"):
        net, _, _ = read_swmm_inp(DATA / "tree_kinwave.inp", geometry=g)
        model, state, drivers = build_model(net, air=False, quality=False)
        out[g] = model._apply_closures(state, drivers)
    assert torch.equal(out["analytic"]["sewer.q"], out["swmm"]["sewer.q"])
    rel = (out["swmm"]["sewer.h"] - out["analytic"]["sewer.h"]).abs() / out["analytic"][
        "sewer.h"
    ]
    assert float(rel.max()) < 1e-2


def test_swmm_geometry_refuses_storage_by_name():
    net, _, _ = read_swmm_inp(DATA / "tree_kinwave.inp")
    with pytest.raises(ValueError, match="geometry='tabulated'.*storage"):
        build_model(net, storage=True, air=False, quality=False)


def test_unknown_geometry_is_refused_by_name():
    with pytest.raises(ValueError, match="geometry must be 'analytic' or 'tabulated'"):
        read_swmm_inp(DATA / "tree_kinwave.inp", geometry="circle")


def test_swmm_geometry_refuses_flow_above_qfull_by_name():
    net, _, _ = read_swmm_inp(DATA / "tree_kinwave.inp")
    model, state, drivers = build_model(net, air=False, quality=False)
    drivers = dict(drivers)
    drivers["inflow"] = drivers["inflow"] * 10.0
    with pytest.raises(ValueError, match="surcharge at pipe"):
        model._apply_closures(state, drivers)


def test_swmm_geometry_is_batched():
    net, _, _ = read_swmm_inp(DATA / "tree_kinwave.inp")
    model, state, drivers = build_model(net, air=False, quality=False)
    single = model._apply_closures(state, drivers)
    drivers = dict(drivers)
    drivers["inflow"] = torch.stack([drivers["inflow"], 0.5 * drivers["inflow"]])
    batched = model._apply_closures(state, drivers)
    assert torch.equal(batched["sewer.h"][0], single["sewer.h"])
    assert torch.all(batched["sewer.h"][1] < batched["sewer.h"][0])

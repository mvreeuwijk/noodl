import math

import pytest
import torch

from noodl.apps.sewer import xsect_tables as xt

F64 = torch.float64
D_FT = torch.tensor(1.5, dtype=F64)          # 1.5 ft pipe
A_FULL = math.pi / 4 * 1.5**2


def t(x):
    return torch.as_tensor(x, dtype=F64)


def test_tables_have_51_entries_and_swmm_end_values():
    assert len(xt.W_CIRC) == 51 and len(xt.R_CIRC) == 51
    assert xt.W_CIRC[25] == 1.0 and xt.W_CIRC[-1] == 0.0
    assert xt.R_CIRC[0] == 0.01 and xt.R_CIRC[-1] == 1.0 and max(xt.R_CIRC) == 1.2172


def test_geometry_at_table_nodes_is_the_table():
    y = t(0.5) * D_FT                                  # node 25 exactly
    assert xt.width_of_depth(y, D_FT).item() == pytest.approx(1.5, rel=1e-15)
    assert xt.radius_of_depth(y, D_FT).item() == pytest.approx(1.5 / 4, rel=1e-15)
    assert xt.area_of_depth(y, D_FT, t(A_FULL)).item() == pytest.approx(0.5 * A_FULL, rel=1e-15)
    assert xt.area_of_depth(t(0.0), D_FT, t(A_FULL)).item() == 0.0
    assert xt.area_of_depth(t(-1.0), D_FT, t(A_FULL)).item() == 0.0


def test_froude_is_zero_when_dry_or_full():
    v = t(3.0)
    assert xt.froude_number(v, t(1e-5), D_FT, t(A_FULL)).item() == 0.0
    assert xt.froude_number(v, D_FT - 5e-5, D_FT, t(A_FULL)).item() == 0.0
    y = t(0.75)
    hd = (xt.area_of_depth(y, D_FT, t(A_FULL)) / xt.width_of_depth(y, D_FT)).item()
    assert xt.froude_number(v, y, D_FT, t(A_FULL)).item() == pytest.approx(
        3.0 / math.sqrt(32.2 * hd), rel=1e-14)


def test_critical_depth_brackets_and_inverts_the_critical_flow():
    q = t([0.0, 0.05, 1.0, 4.0])
    yc = xt.critical_depth(q, D_FT, t(A_FULL))
    assert yc[0].item() == 0.0
    assert torch.all(yc[1:] > 0) and torch.all(yc <= D_FT)
    assert torch.all(yc[1:].diff() > 0)                # monotone in q
    # SWMM interpolates Qc linearly between 25 depth stations. Qc ~ y^2.5 is strongly convex
    # in the first interval (y < 0.04 D), where linear interpolation puts Qc(yc) 8.8 % below q
    # (q = 0.05 cfs); above it the error is below 0.3 %. Hence 10 % for the first interval,
    # 2 % elsewhere (the python scalar transcription of getYcritEnum gives identical yc).
    for qi, yi in zip(q[1:], yc[1:], strict=True):
        a = xt.area_of_depth(yi, D_FT, t(A_FULL))
        w = xt.width_of_depth(yi, D_FT)
        qc = a * torch.sqrt(32.2 * a / w)
        assert qc.item() == pytest.approx(qi.item(), rel=0.1 if qi.item() < 0.1 else 2e-2)


def test_normal_depth_saturates_at_full():
    y_full, a_full, s_full, beta = xt.conduit_constants(t(0.4572), t(0.013), t(0.005))
    q_full = (s_full * beta).item()
    yn = xt.normal_depth_ft(t([0.0, 0.5 * q_full, q_full, 2 * q_full]),
                            y_full, a_full, s_full, beta)
    assert yn[0].item() == 0.0
    assert 0 < yn[1].item() < y_full.item()
    assert yn[2].item() == pytest.approx(y_full.item()) and yn[3].item() == pytest.approx(
        y_full.item())

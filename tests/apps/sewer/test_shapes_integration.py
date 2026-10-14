"""Non-circular sections, losses, transects and culverts through the reader, the network
validation and the dynamic-wave terms. The match with SWMM is
`tests/verification/test_sewer_shapes_parity.py`."""

import dataclasses
import math
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import dynwave_terms as dt_
from noodl.apps.sewer import read_swmm_inp
from noodl.apps.sewer import xsect_tables as xt
from noodl.apps.sewer.culvert import BIG, CulvertSet
from noodl.apps.sewer.inp_xsect import read_transects
from noodl.apps.sewer.network import Pipe
from noodl.apps.sewer.xsect_spec import Xsect

F64 = torch.float64
C1_LINE = "C1               CIRCULAR     0.30             0          0          0          1"
DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"


def t(x):
    return torch.as_tensor(x, dtype=F64)


def tree(tmp_path, old, new, name="x.inp", source="tree_dynwave.inp"):
    text = (DATA / source).read_text().replace(old, new)
    assert text != (DATA / source).read_text(), "the replacement changed nothing"
    path = tmp_path / name
    path.write_text(text)
    return path


# ------------------------------------------------------------------------- the reader
def test_every_shape_keyword_is_read_into_an_xsect(tmp_path):
    cases = {
        "RECT_CLOSED     0.3 0.5 0 0": ("RECT_CLOSED", (0.3, 0.5, 0, 0)),
        "TRAPEZOIDAL     0.3 0.5 1.5 2.5": ("TRAPEZOIDAL", (0.3, 0.5, 1.5, 2.5)),
        "EGG             0.3 0 0 0": ("EGG", (0.3, 0, 0, 0)),
        "FORCE_MAIN      0.3 120 0 0": ("FORCE_MAIN", (0.3, 120.0, 0, 0)),
        "ARCH            3 0 0 0": ("ARCH", (3.0, 0, 3.0, 0)),         # Geom1 is the size code
        "RECT_ROUND      0.3 0.4 0.3 0": ("RECT_ROUND", (0.3, 0.4, 0.3, 0)),
    }
    for line, (kind, geom) in cases.items():
        path = tree(tmp_path, C1_LINE, f"C1               {line} 1")
        pipe = {p.name: p for p in read_swmm_inp(path)[0].pipes}["C1"]
        assert pipe.xsect.kind == kind and pipe.xsect.geom == pytest.approx(geom)
        assert pipe.diameter == pytest.approx(pipe.xsect.height)


def test_us_units_scale_lengths_but_not_slopes_or_codes(tmp_path):
    path = tmp_path / "us.inp"
    text = (DATA / "shapes_us_dynwave.inp").read_text()
    path.write_text(text)
    net = read_swmm_inp(path)[0]
    pipes = {p.name: p for p in net.pipes}
    assert pipes["CTZ"].xsect.geom == pytest.approx((3 * 0.3048, 4 * 0.3048, 2.0, 3.0))
    assert pipes["CAC"].xsect.geom[2] == 11.0              # Geom1 alone: the size code
    assert pipes["CHE"].diameter == pytest.approx(pipes["CHE"].xsect.height)


def test_refusals_name_the_shape_and_the_line(tmp_path):
    old = C1_LINE
    for line, match in (
        ("C1 STREET 0.3 0 0 0", "STREET"),
        ("C1 DUMMY 0.3 0 0 0", "DUMMY"),
        ("C1 BOX 0.3 0 0 0", "unknown cross-section shape"),
        ("C1 RECT_CLOSED 0.3 0 0 0", "xsect_setParams"),
        ("C1 FORCE_MAIN 0.3 0 0 0", "xsect_setParams|positive roughness"),
        ("C1 IRREGULAR NOPE", "transect"),
        ("C1 CUSTOM 0.3 NOPE 0 0", "SHAPE curve"),
    ):
        path = tree(tmp_path, old, line + " 1")
        with pytest.raises(ValueError, match=match):
            read_swmm_inp(path)


def test_non_circular_sections_need_dynamic_wave(tmp_path):
    text = (DATA / "tree_kinwave.inp").read_text().replace(
        "C1               CIRCULAR", "C1               RECT_CLOSED")
    path = tmp_path / "k.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match="shape other than CIRCULAR"):
        read_swmm_inp(path)


def test_losses_are_read_swapped_with_a_reversed_conduit_and_seepage_is_refused(tmp_path):
    base = (DATA / "tree_dynwave.inp").read_text()
    path = tmp_path / "l.inp"
    path.write_text(base + "\n[LOSSES]\nC1 0.5 0.25 0.1 NO 0\n")
    pipe = {p.name: p for p in read_swmm_inp(path)[0].pipes}["C1"]
    assert (pipe.loss_inlet, pipe.loss_outlet, pipe.loss_average) == (0.5, 0.25, 0.1)
    path.write_text(base + "\n[LOSSES]\nC1 0.5 0.25 0.1 NO 3.0\n")
    with pytest.raises(ValueError, match="seepage"):
        read_swmm_inp(path)
    path.write_text(base + "\n[LOSSES]\nC1 -0.5 0 0 NO 0\n")
    with pytest.raises(ValueError, match="negative"):
        read_swmm_inp(path)
    kin = (DATA / "tree_kinwave.inp").read_text() + "\n[LOSSES]\nC1 0.5 0 0 NO 0\n"
    path.write_text(kin)
    with pytest.raises(ValueError, match="dynamic-wave routing only"):
        read_swmm_inp(path)


def test_the_culvert_code_is_read_and_refused_outside_dynamic_wave(tmp_path):
    path = tree(tmp_path, C1_LINE, C1_LINE + "          7")
    assert {p.name: p.culvert for p in read_swmm_inp(path)[0].pipes}["C1"] == 7
    text = (DATA / "tree_kinwave.inp").read_text()
    kin = tmp_path / "k.inp"
    kin.write_text(text.replace(C1_LINE, C1_LINE + "          7"))
    with pytest.raises(ValueError, match="inlet control is modelled by dynamic-wave"):
        read_swmm_inp(kin)


def test_force_main_equation_option_and_darcy_roughness_units(tmp_path):
    net = read_swmm_inp(DATA / "force_main_dw_dynwave.inp")[0]
    assert net.dynamic_wave.force_main_equation == "D-W"
    pipe = net.pipes[0]
    assert pipe.xsect.geom[1] == pytest.approx(0.5e-3)       # 0.5 mm in m (SI file)
    assert read_swmm_inp(DATA / "force_main_hw_dynwave.inp")[0].pipes[0].xsect.geom[1] == 130.0


def test_transect_reader_follows_the_nc_state_machine():
    class L:
        def __init__(self, n, *f):
            self.number, self.fields = n, tuple(str(x) for x in f)

    gr = ("GR", 2.0, 0.0, 0.0, 3.0, 2.0, 6.0)
    lines = [L(1, "NC", 0.1, 0.0, 0.03), L(2, "X1", "A", 3, 0.0, 6.0, 0, 0, 4.0, 0, 0)]
    lines.append(L(3, *gr))
    # the second transect repeats no channel n: it inherits SWMM's static value, already
    # multiplied by sqrt(Lfactor) = 2 when the first was validated
    lines += [L(4, "NC", 0.0, 0.0, 0.0), L(5, "X1", "B", 3, 0.0, 6.0, 0, 0, 0, 0, 0), L(6, *gr)]
    out = read_transects(lines, "t.inp", us=False)
    assert out["A"].n_left == 0.1 and out["A"].n_right == 0.03     # an unset n takes the channel
    assert out["A"].n_channel == 0.03 and out["A"].length_factor == 4.0
    assert out["B"].n_channel == pytest.approx(0.03 * math.sqrt(4.0))
    assert out["B"].n_left == 0.1
    with pytest.raises(ValueError, match="not NC, X1 or GR"):
        read_transects([L(1, "ZZ")], "t.inp", us=False)
    with pytest.raises(ValueError, match="must not decrease"):
        read_transects([L(1, "NC", 0.1, 0.1, 0.1), L(2, "X1", "A", 2, 0, 5, 0, 0, 0, 0, 0),
                        L(3, "GR", 1.0, 5.0, 0.0, 1.0)], "t.inp", us=False)


def test_an_irregular_conduit_takes_its_length_and_roughness_from_the_transect():
    net = read_swmm_inp(DATA / "irregular_dynwave.inp")[0]
    t2 = {p.name: p for p in net.pipes}["CT2"]
    assert t2.transect_length_factor == 1.25
    assert t2.length == pytest.approx(200.0 / 1.25)
    assert t2.n == pytest.approx(0.03 * math.sqrt(1.25))


# -------------------------------------------------------------- network validation
def _net_with(pipe_changes):
    net = read_swmm_inp(DATA / "tree_dynwave.inp")[0]
    pipes = list(net.pipes)
    pipes[0] = dataclasses.replace(pipes[0], **pipe_changes)
    return dataclasses.replace(net, pipes=tuple(pipes))


def test_pipe_validation_refuses_by_name():
    with pytest.raises(ValueError, match="full height"):
        _net_with({"xsect": Xsect("RECT_CLOSED", (0.5, 1.0))}).validate()
    with pytest.raises(ValueError, match="STREET"):
        _net_with({"xsect": Xsect("STREET", (0.5,))}).validate()
    with pytest.raises(ValueError, match="loss_inlet"):
        _net_with({"loss_inlet": -1.0}).validate()
    with pytest.raises(ValueError, match="culvert code"):
        _net_with({"culvert": -2}).validate()
    ok = Xsect("RECT_CLOSED", (0.3, 0.5))
    _net_with({"xsect": ok, "diameter": ok.height}).validate()
    net = _net_with({"xsect": ok, "diameter": ok.height})
    with pytest.raises(ValueError, match="tabulated"):
        dataclasses.replace(net, geometry="analytic").validate()
    kin = dataclasses.replace(net, routing="kinematic_wave")
    with pytest.raises(ValueError, match="xsect"):
        kin.validate()


def test_quality_and_air_are_refused_for_a_non_circular_conduit():
    from noodl.apps.sewer import build_model
    ok = Xsect("RECT_CLOSED", (0.3, 0.5))
    net = _net_with({"xsect": ok, "diameter": ok.height})
    with pytest.raises(ValueError, match="non-circular cross"):
        build_model(net, quality=True)


# ------------------------------------------------------- terms: circle path unchanged
def _geo(sections=None, **kw):
    args = dict(diameter=t([0.45, 0.45]), roughness=t([0.013, 0.013]),
                slope=t([0.005, 0.005]), length=t([200.0, 200.0]), invert1=t([10.0, 10.0]),
                invert2=t([9.0, 9.0]), offset1=t([0.0, 0.0]), offset2=t([0.0, 0.0]),
                outfall1=t([False, False]).bool(), outfall2=t([False, False]).bool())
    args.update(kw)
    return dt_.ConduitGeometry.from_si(**args, sections=sections)


def test_a_circle_next_to_another_shape_gets_exactly_the_circle_terms():
    plain = _geo()
    rect = Xsect("RECT_CLOSED", (0.5, 0.9))
    mixed = _geo(sections=[None, rect], diameter=t([0.45, rect.height]))
    args = (t([[0.7, 0.7]]), t([[0.5, 0.5]]), t([[2.0, 2.0]]), t([[1.8, 1.8]]),
            xt.area_of_depth(t([[0.55, 0.55]]), plain.y_full, plain.a_full), 5.0)
    a = dt_.conduit_terms(plain, *args)
    b = dt_.conduit_terms(mixed, *args)
    for name in ("flow_offset", "conveyance", "momentum_flow", "surface_area1",
                 "surface_area2", "mid_area", "volume", "froude"):
        assert getattr(a, name)[0, 0].item() == getattr(b, name)[0, 0].item(), name


def test_local_losses_lengthen_the_denominator_of_the_momentum_update():
    """dq5 = (cIn q/a1 + cOut q/a2 + cAvg q/aMid) / 2 / length * dt joins ``1 + dq1`` in
    the denominator (dwflow.c:228-239); on a level surface a1 = a2 = aMid."""
    base = _geo()
    lossy = _geo(losses=(t([0.5, 0.0]), t([0.0, 1.0]), t([0.1, 0.2])))
    y = t([[0.6, 0.6]])
    q, dt = 1.5, 5.0
    args = (y, y, t([[q, q]]), t([[q, q]]),
            xt.area_of_depth(t([[0.5, 0.5]]), base.y_full, base.a_full), dt)
    a, b = dt_.conduit_terms(base, *args), dt_.conduit_terms(lossy, *args)
    area = b.mid_area[0]
    length = base.mod_length

    def denominator(r):                       # k = g dt aWtd / (L denom), aWtd = aMid here
        return xt.GRAVITY_FT * dt * area / (length * r.conveyance[0])

    got = denominator(b) - denominator(a)
    want = t([(0.5 + 0.1) * q / area[0].item(), (1.0 + 0.2) * q / area[1].item()])
    want = want / 2.0 / length * dt
    torch.testing.assert_close(got, want, rtol=1e-9, atol=0)
    assert bool((b.conveyance < a.conveyance).all())


# ------------------------------------------------------------------------------ culverts
def _culvert(code, d=0.6, slope=0.004):
    y = t([d / xt.LCF])
    a = math.pi / 4 * y**2
    xs = dt_.XsectSet(["CIRCULAR"], {"y_full": y, "a_full": a, "r_full": 0.25 * y,
                                     "s_full": t([1.0]), "s_max": t([1.08]), "w_max": y})
    return CulvertSet(torch.tensor([0]), torch.tensor([code]), t([slope]), y, a, xs)


def test_culvert_flow_rises_with_head_and_is_continuous_across_the_zones():
    heads = torch.linspace(0.05, 2.5, 80, dtype=F64).unsqueeze(-1)
    for code in (12, 49):                                   # form 2: a power law, monotone
        cv = _culvert(code)
        q, d = cv.inlet_flow(cv.y_full * heads)
        assert bool((q[1:] >= q[:-1]).all()) and bool((d >= 0).all()), code
        assert float((q[1:] - q[:-1]).abs().max()) < 0.2 * float(q.max()), code
    for code in (1, 10):        # form 1: Ridder's last evaluation, SWMM's value, not monotone
        cv = _culvert(code)
        q, d = cv.inlet_flow(cv.y_full * heads)
        assert bool((d >= 0).all()) and bool((q > 0).all()) and bool((q < BIG).all()), code
        assert q[-1].item() > 4 * q[0].item()
    cv = _culvert(1)
    q_lo, _ = cv.inlet_flow(cv.y_low.reshape(1))
    q_hi, _ = cv.inlet_flow(cv.y_high.reshape(1))
    assert q_lo.item() == pytest.approx(cv.q_low.item(), rel=1e-12)
    assert q_hi.item() == pytest.approx(cv.q_high.item(), rel=1e-12)


def test_submerged_culvert_follows_the_orifice_equation():
    cv = _culvert(1)
    y = cv.y_full * 1.8
    q, _ = cv.inlet_flow(y.reshape(1))
    arg = (1.8 - 0.67 + cv.scf.item()) / 0.0398
    assert q.item() == pytest.approx(math.sqrt(arg) * cv.ad.item(), rel=1e-12)


def test_mitered_inlets_use_the_larger_slope_correction():
    assert _culvert(5, slope=0.01).scf.item() == pytest.approx(-0.07)
    assert _culvert(1, slope=0.01).scf.item() == pytest.approx(0.005)


def test_a_culvert_pipe_is_limited_instead_of_normal_flow_checked():
    geo = _geo(culverts=torch.tensor([1, 0]))
    assert geo.culvert is not None and geo.culvert_mask.tolist() == [True, False]
    y = t([[1.5, 1.5]])
    args = (y, t([[0.5, 0.5]]), t([[3.0, 3.0]]), t([[3.0, 3.0]]),
            xt.area_of_depth(t([[0.5, 0.5]]), geo.y_full, geo.a_full), 5.0)
    r = dt_.conduit_terms(geo, *args)
    assert r.cap_flow is not None and bool(torch.isfinite(r.cap_flow[0, 0]))
    assert not bool(torch.isfinite(r.cap_flow[0, 1]))         # not a culvert
    assert not bool(r.limited[0, 0])                          # the normal-flow check is off
    assert dt_.conduit_terms(_geo(), *args).cap_flow is None


def test_culvert_codes_above_the_table_are_ignored_as_swmm_does():
    assert _geo(culverts=torch.tensor([58, 0])).culvert is None


def test_pipe_default_fields_leave_the_circle_untouched():
    p = Pipe("c", "a", "b", 10.0, 0.3, 0.013, 0.01)
    assert p.xsect is None and p.culvert == 0 and p.loss_inlet == 0.0

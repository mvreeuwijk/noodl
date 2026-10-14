"""SWMM `FLOW_UNITS`: the reader converts CFS, GPM, MGD, CMS, LPS and MLD files to SI.

The committed fixtures are all CMS. `twin` rewrites one into another unit system (every
length, flow, area and coefficient the reader converts, by the inverse of the reader's own
rule), so a unit-system error that the reader and the twin share is caught by the live-SWMM
check at the end, which runs the twin in SWMM itself.
"""

import dataclasses
import math
import re
from pathlib import Path

import pytest
import torch

from noodl.apps.sewer import build_model, read_swmm_inp, simulate_dynamic_wave
from noodl.apps.sewer.inp import unit_factors
from noodl.apps.sewer.network import SewerNetwork
from noodl.apps.sewer.structures import M3_PER_FT3

DATA = Path(__file__).resolve().parents[2] / "data" / "sewer"
UNITS = ("CFS", "GPM", "MGD", "CMS", "LPS", "MLD")

# column indices (0-based fields) of each quantity kind, per section
_LENGTH = {"JUNCTIONS": (1, 2, 3, 4), "OUTFALLS": (1,), "CONDUITS": (3, 5, 6),
           "ORIFICES": (4,), "WEIRS": (4,), "OUTLETS": (3,)}
_STORAGE_SHAPE_LENGTHS = {"CYLINDRICAL": (5, 6), "CONICAL": (5, 6), "PYRAMIDAL": (5, 6),
                          "PARABOLOID": (5, 6, 7), "PARABOLIC": (5, 6, 7)}


def _fmt(x: float) -> str:
    return repr(float(x))


def length(f, lf, i, p=1):
    """Field `i` of `f` from SI metres (or m^p) to the file's length unit."""
    f[i] = _fmt(float(f[i]) / lf**p)


def twin(text: str, units: str) -> str:
    """The CMS file `text` rewritten in `units` (see the module docstring)."""
    lf, qf = unit_factors(units)
    out, section, curve_kind = [], None, {}
    for raw in text.splitlines():
        body = raw.split(";", 1)[0].strip()
        if not body or body.startswith(";"):
            out.append(raw)
            continue
        if body.startswith("["):
            section = body.strip("[]").upper()
            out.append(raw)
            continue
        f = body.split()
        up = [x.upper() for x in f]

        if section == "OPTIONS":
            if up[0] == "FLOW_UNITS":
                f[1] = units
            elif up[0] == "HEAD_TOLERANCE":
                length(f, lf, 1)
            elif up[0] == "MIN_SURFAREA":
                length(f, lf, 1, 2)
        elif section in _LENGTH:
            for i in _LENGTH[section]:
                if i < len(f):
                    length(f, lf, i)
            if section == "CONDUITS":
                for i in (7, 8):
                    if i < len(f):
                        f[i] = _fmt(float(f[i]) / qf)
            elif section == "OUTFALLS" and len(f) > 3 and up[2] == "FIXED":
                length(f, lf, 3)
            elif section == "WEIRS":
                if lf != 1.0:               # Cd: m^0.5/s -> ft^0.5/s (SWMM's M3perFT3 path)
                    f[5] = _fmt(float(f[5]) * lf**2.5 / M3_PER_FT3)
                    if len(f) > 8 and f[8] != "*":
                        f[8] = _fmt(float(f[8]) * lf**2.5 / M3_PER_FT3)
            elif section == "OUTLETS" and up[4].startswith("FUNCTIONAL"):
                b = float(f[6])
                f[5] = _fmt(float(f[5]) * lf**b / qf)               # q = a h^b
        elif section == "XSECTIONS":
            length(f, lf, 2)
            if up[1] != "CIRCULAR":
                length(f, lf, 3)
        elif section == "STORAGE":
            for i in (1, 2, 3):
                length(f, lf, i)
            shape = up[4]
            n = 6
            if shape == "FUNCTIONAL":
                a1, a2 = float(f[5]), float(f[6])
                f[5] = _fmt(a1 / lf ** (2.0 - a2))
                f[7] = _fmt(float(f[7]) / lf**2)
                n = 8
            elif shape != "TABULAR":
                for i in _STORAGE_SHAPE_LENGTHS[shape]:
                    length(f, lf, i)
                n = 8
            if len(f) > n:
                length(f, lf, n)
        elif section == "CURVES":
            first = 1
            if f[1].upper() in ("STORAGE", "TIDAL", "RATING"):
                curve_kind[f[0]] = f[1].upper()
                first = 2
            kind = curve_kind[f[0]]
            for i in range(first, len(f), 2):
                if kind == "STORAGE":
                    length(f, lf, i)
                    length(f, lf, i + 1, 2)
                elif kind == "RATING":
                    length(f, lf, i)
                    f[i + 1] = _fmt(float(f[i + 1]) / qf)
                else:                                                # TIDAL: hours, stage
                    length(f, lf, i + 1)
        elif section == "INFLOWS" and up[1] == "FLOW":
            f[6] = _fmt(float(f[6]) / qf)
        out.append(" ".join(f))
    return "\n".join(out) + "\n"


def _close(a, b, rel=1e-12, abs_=1e-14) -> bool:
    """Equal up to round-off, through dataclasses, tuples and floats."""
    if dataclasses.is_dataclass(a):
        return type(a) is type(b) and all(
            _close(getattr(a, f.name), getattr(b, f.name), rel, abs_)
            for f in dataclasses.fields(a))
    if isinstance(a, tuple | list):
        return (len(a) == len(b)
                and all(_close(x, y, rel, abs_) for x, y in zip(a, b, strict=True)))
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)
    return a == b


def test_unit_factors_are_swmms():
    """SWMM 5.2.4's own constants (units.c: 448.831 gpm, 0.64632 mgd, 0.02832 cms, 28.317 lps
    and 2.4466 mld per cfs; read from the engine), against noodl's CMS flow (0.02832 per
    cfs): a flow converts by ``0.02832 / Qcf``, a US length by 0.3048."""
    ft, cfs = 0.3048, 0.02832
    assert unit_factors("CMS") == (1.0, 1.0)
    assert unit_factors("lps") == (1.0, pytest.approx(cfs / 28.317, rel=1e-15))
    assert unit_factors("MLD") == (1.0, pytest.approx(cfs / 2.4466, rel=1e-15))
    assert unit_factors("CFS") == (ft, cfs)
    assert unit_factors("GPM") == (ft, pytest.approx(cfs / 448.831, rel=1e-15))
    assert unit_factors("MGD") == (ft, pytest.approx(cfs / 0.64632, rel=1e-15))
    with pytest.raises(ValueError, match="FLOW_UNITS must be one of"):
        unit_factors("GALLONS")


def test_unknown_flow_units_are_refused_with_the_line(tmp_path):
    text = (DATA / "tree_steady.inp").read_text().replace(
        "FLOW_UNITS           CMS", "FLOW_UNITS           MGAL")
    path = tmp_path / "bad.inp"
    path.write_text(text)
    with pytest.raises(ValueError, match=r"line \d+: FLOW_UNITS must be one of"):
        read_swmm_inp(path)


def test_absent_flow_units_is_swmms_default_cfs(tmp_path):
    text = re.sub(r"(?m)^FLOW_UNITS.*\n", "", (DATA / "tree_steady.inp").read_text())
    path = tmp_path / "default.inp"
    path.write_text(text)
    cfs = read_swmm_inp(path)[0]
    cms = read_swmm_inp(DATA / "tree_steady.inp")[0]
    assert cfs.pipes[0].length == pytest.approx(cms.pipes[0].length * 0.3048, rel=1e-14)
    assert cfs.manholes[0].inflow == pytest.approx(cms.manholes[0].inflow * 0.02832, rel=1e-14)


def test_a_us_file_is_read_in_si(tmp_path):
    """A US-units file with hand-checked numbers: ft lengths and elevations, cfs flows."""
    text = (DATA / "tree_steady.inp").read_text().replace(
        "FLOW_UNITS           CMS", "FLOW_UNITS           CFS")
    path = tmp_path / "us.inp"
    path.write_text(text)
    net, _, _ = read_swmm_inp(path)
    cms, _, _ = read_swmm_inp(DATA / "tree_steady.inp")
    j1 = next(m for m in net.manholes if m.name == "J1")
    assert j1.invert == pytest.approx(12.0 * 0.3048, rel=1e-15)        # ft -> m
    assert j1.inflow == pytest.approx(0.05 * 0.02832, rel=1e-15)         # cfs -> m3/s
    c1, c1_cms = net.pipes[0], cms.pipes[0]
    assert c1.length == pytest.approx(200.0 * 0.3048, rel=1e-15)
    assert c1.diameter == pytest.approx(c1_cms.diameter * 0.3048, rel=1e-15)
    assert c1.n == c1_cms.n                              # Manning n: as is


def test_functional_outlet_and_weir_coefficients_convert_with_the_length_system(tmp_path):
    """``q = a h^b`` becomes ``a flow / length^b``; a weir's ``Cd`` (ft^0.5/s in a US file)
    becomes SWMM's SI one by ``M3perFT3 / ft^2.5`` and is untouched in an SI file whatever its
    flow unit."""
    base = (DATA / "structures_dynwave.inp").read_text().replace(
        "TABULAR/DEPTH   Rate1                       NO",
        "FUNCTIONAL/HEAD 0.5              1.5        NO")
    for units in ("CFS", "GPM", "LPS"):
        path = tmp_path / f"{units}.inp"
        path.write_text(twin(base, units))
        net, _, _ = read_swmm_inp(path)
        assert net.outlets[0].rating == ("functional", pytest.approx(0.5, rel=1e-12), 1.5)
        assert net.weirs[0].discharge_coefficient == pytest.approx(1.84, rel=1e-12)
    ft = 0.3048
    us = base.replace("FLOW_UNITS           CMS", "FLOW_UNITS           CFS")
    path = tmp_path / "raw.inp"
    path.write_text(us)
    net, _, _ = read_swmm_inp(path)
    assert net.outlets[0].rating[1] == pytest.approx(0.5 * 0.02832 / ft**1.5, rel=1e-13)
    assert net.weirs[0].discharge_coefficient == pytest.approx(1.84 * 0.028317 / ft**2.5,
                                                               rel=1e-13)
    assert net.orifices[0].discharge_coefficient == 0.65                 # dimensionless


FIXTURES = ["tree_steady.inp", "tree_kinwave.inp", "tree_dynwave.inp", "options_dynwave.inp",
            "backwater_dynwave.inp", "storage_dynwave.inp", "structures_dynwave.inp",
            "tidal_dynwave.inp", "flooding_outfall_dynwave.inp"]


@pytest.mark.parametrize("units", [u for u in UNITS if u != "CMS"])
@pytest.mark.parametrize("name", FIXTURES)
def test_unit_twins_read_the_same_network(tmp_path, name, units):
    """The CFS, GPM, MGD, LPS and MLD twins of each CMS fixture read to the same SI network
    (and the same dynamic-wave options) up to round-off."""
    text = (DATA / name).read_text()
    path = tmp_path / f"twin_{units}.inp"
    path.write_text(twin(text, units))
    got = read_swmm_inp(path)
    want = read_swmm_inp(DATA / name)
    assert isinstance(got[0], SewerNetwork)
    assert _close(got[0], want[0]), (name, units)
    assert got[1:] == want[1:]


def test_storage_shape_twins_read_the_same_network(tmp_path):
    base = (DATA / "storage_dynwave.inp").read_text()
    for shape, params in (("CYLINDRICAL", "4 2 0"), ("CONICAL", "4 2 0.5"),
                          ("PARABOLIC", "4 2 3"), ("PYRAMIDAL", "4 2 0.5")):
        text = base.replace("FUNCTIONAL 20         1.5        5", f"{shape} {params}")
        assert text != base
        path = tmp_path / f"{shape}.inp"
        path.write_text(twin(text, "CFS"))
        want = tmp_path / f"{shape}_cms.inp"
        want.write_text(text)
        assert _close(read_swmm_inp(path)[0], read_swmm_inp(want)[0]), shape


def test_empty_network_is_refused_by_build_model():
    net = SewerNetwork(manholes=(), pipes=(), outfalls=())
    with pytest.raises(ValueError, match="no links"):
        build_model(net, air=False, quality=False)


# ---- live SWMM on the converted twin ----------------------------------------------------

pyswmm = pytest.importorskip("pyswmm")
WINDOW_S = 120
#: depth (m) and flow targets as in `tests/verification/test_sewer_dynwave_parity.py`
TOL_DEPTH, TOL_FLOW_REL, TOL_FLOW_ABS = 1e-7, 1e-6, 1e-9


def _swmm_final(tmp_path, text, units):
    """SWMM's depths (m) and flows (m3/s, file orientation) after `WINDOW_S` s, with the
    connections."""
    lf, qf = unit_factors(units)
    path = tmp_path / "model.inp"
    path.write_text(re.sub(r"(?m)^END_TIME .*$", f"END_TIME             00:0{WINDOW_S // 60}:00",
                           text))
    with pyswmm.Simulation(str(path), str(tmp_path / "m.rpt"), str(tmp_path / "m.out")) as sim:
        conn = {k.linkid: tuple(k.connections) for k in pyswmm.Links(sim)}
        for _ in sim:
            pass
        depths = {n.nodeid: n.depth * lf for n in pyswmm.Nodes(sim)}
        flows = {k.linkid: k.flow * qf for k in pyswmm.Links(sim)}
    return depths, flows, conn


_FUNCTIONAL_OUTLET = ("TABULAR/DEPTH   Rate1                       NO",
                      "FUNCTIONAL/HEAD 0.5              1.5        NO")
_SHAPE = "FUNCTIONAL 20         1.5        5"
_CASES = [
    ("structures_dynwave.inp", "CFS", ()), ("structures_dynwave.inp", "MLD", ()),
    ("structures_dynwave.inp", "GPM", (_FUNCTIONAL_OUTLET,)),
    ("structures_dynwave.inp", "LPS", (_FUNCTIONAL_OUTLET,)),
    ("storage_dynwave.inp", "CFS", ()), ("tidal_dynwave.inp", "LPS", ()),
    ("tree_dynwave.inp", "GPM", ()), ("tree_dynwave.inp", "MGD", ()),
    *[("storage_dynwave.inp", "CFS", ((_SHAPE, f"{shape} {params}"),))
      for shape, params in (("CYLINDRICAL", "4 2 0"), ("PARABOLIC", "4 2 3"),
                            ("PYRAMIDAL", "4 2 0.5"))],
]


@pytest.mark.parametrize("name,units,edits", _CASES)
def test_noodl_matches_swmm_on_the_converted_file(tmp_path, name, units, edits):
    """The twin file, in SWMM and in noodl: depths and flows agree after two minutes at the
    targets of the CMS parity tests -- a wrong factor in the reader (and in `twin`) would
    show as a mismatch with SWMM's own conversion."""
    text = (DATA / name).read_text()
    for old, new in edits:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    text = twin(text, units)
    path = tmp_path / "twin.inp"
    path.write_text(text)
    depths, flows, conn = _swmm_final(tmp_path, text, units)
    net, _, _ = read_swmm_inp(path)
    model, state, drivers = build_model(net, iterate_tol=1e-13, iterate_max=2000)
    _, hist = simulate_dynamic_wave(model, state, drivers, duration=float(WINDOW_S),
                                    record=("water.phi", "water.q"))
    hyd = model.dynamic_wave
    nodes = list(hyd.junction_names) + list(hyd.outfall_names)
    idx = torch.cat([model.junction_idx, hyd.outfall_idx])
    y = hist["water.phi"][-1, idx] - hyd.node_invert[idx]
    for j, n in enumerate(nodes):
        assert abs(y[j].item() - depths[n]) <= TOL_DEPTH, (n, y[j].item(), depths[n])
    names = list(model.net.nodes)
    for i, link in enumerate(model.link_names):
        pair = (names[int(hyd.edge_u[i])], names[int(hyd.edge_v[i])])
        sign = 1.0 if conn[link] == pair else -1.0
        q, ref = hist["water.q"][-1, i].item(), sign * flows[link]
        assert abs(q - ref) <= TOL_FLOW_REL * abs(ref) + TOL_FLOW_ABS, (link, q, ref)

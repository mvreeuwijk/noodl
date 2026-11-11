"""The water-quality sections of an EPANET `.inp`: `[QUALITY] [SOURCES] [REACTIONS] [MIXING]`,
`[OPTIONS] QUALITY|DIFFUSIVITY|TOLERANCE` and `[TIMES] QUALITY TIMESTEP`, read into
`WaterNetwork.quality` in SI (EPANET 2.2 `input3.c`, `input1.c`)."""

import dataclasses
from pathlib import Path

import pytest

from noodl.apps.water import QualitySource, WaterQuality, read_epanet_inp
from noodl.apps.water.units import DIFFUSIVITY_M2

wntr = pytest.importorskip("wntr")

DATA = Path(__file__).resolve().parents[2] / "data" / "water"
WNTR = Path(wntr.__file__).parent / "library" / "networks"
FOOT = 0.3048

_TANKS = "[TANKS]\nT1 0 2 0 4 10 0\n"
_PIPES = "[PIPES]\nP1 R1 J1 100 100 100\nP2 J1 J2 100 100 100\nP3 J2 T1 100 100 100\n"
_BASE = """[JUNCTIONS]
{junctions}
[RESERVOIRS]
R1 10
{tanks}{pipes}[PUMPS]
PU1 R1 J2 POWER 10
[PATTERNS]
PAT 1 2
[OPTIONS]
Units {units}
{options}
[TIMES]
Duration 1
{times}
{sections}
[END]
"""


def _write(tmp_path, sections="", *, units="LPS", options="", times="", junctions=None,
           tanks=_TANKS, pipes=_PIPES):
    path = tmp_path / "q.inp"
    path.write_text(_BASE.format(
        junctions=junctions or "J1 0 0\nJ2 0 0", units=units, options=options, times=times,
        sections=sections, tanks=tanks, pipes=pipes))
    return path


def _quality(tmp_path, sections="", **kw):
    return read_epanet_inp(_write(tmp_path, sections, **kw)).quality


# ------------------------------------------------------------------ committed networks
def test_net1_reads_chlorine_reactions_and_initial_quality():
    quality = read_epanet_inp(DATA / "Net1.inp").quality
    assert (quality.mode, quality.chemical, quality.units) == ("CHEMICAL", "Chlorine", "mg/L")
    assert quality.global_bulk == -0.5 / 86400
    assert quality.global_wall == -FOOT / 86400
    assert quality.initial["10"] == 0.5
    assert quality.initial["9"] == quality.initial["2"] == 1.0
    assert quality.quality_timestep == 300
    assert quality.diffusivity == DIFFUSIVITY_M2
    assert quality.tolerance == 0.01


def test_net2_net3_ky10_modes():
    net2 = read_epanet_inp(WNTR / "Net2.inp").quality
    assert net2.sources == (QualitySource("1", "CONCEN", 1.0, "3"),)
    net3 = read_epanet_inp(WNTR / "Net3.inp").quality
    assert (net3.mode, net3.trace_node, net3.units) == ("TRACE", "Lake", "%")
    assert net3.chemical == "Trace Lake"
    ky10 = read_epanet_inp(WNTR / "ky10.inp").quality
    assert (ky10.mode, ky10.units, ky10.chemical) == ("AGE", "hrs", "Age")


def test_a_file_without_quality_reads_mode_none(tmp_path):
    net = read_epanet_inp(_write(tmp_path))
    assert net.quality == WaterQuality(
        node_order=("J1", "J2", "R1", "T1"), link_order=("P1", "P2", "P3", "PU1"))
    assert "ignored_quality" not in net.notes


# ------------------------------------------------------------------ options
def test_the_chemical_keyword_keeps_epanets_default_units(tmp_path):
    quality = _quality(tmp_path, options="Quality CHEMICAL ug/L")
    assert (quality.mode, quality.chemical, quality.units) == ("CHEMICAL", "Chemical", "mg/L")
    assert _quality(tmp_path, options="Quality chem").mode == "CHEMICAL"
    named = _quality(tmp_path, options="Quality Fluoride ug/L")
    assert (named.mode, named.chemical, named.units) == ("CHEMICAL", "Fluoride", "ug/L")
    assert _quality(tmp_path, options="Quality Fluoride").units == "mg/L"


def test_the_quality_options_are_not_unrecognised(tmp_path):
    net = read_epanet_inp(_write(
        tmp_path, options="Quality AGE\nTolerance 0.05\nDiffusivity 2\nTrials 40",
        times="Quality Timestep 0:05\nStatistic NONE"))
    assert net.notes["unrecognised_options"] == "Trials 40"
    assert net.notes["unrecognised_times"] == "Statistic NONE"
    assert net.quality.tolerance == 0.05
    assert net.quality.quality_timestep == 300.0


def test_diffusivity_multiplier_actual_and_zero(tmp_path):
    assert _quality(tmp_path, options="Diffusivity 1.0").diffusivity == DIFFUSIVITY_M2
    assert _quality(tmp_path, options="Diffusivity 2").diffusivity == 2 * DIFFUSIVITY_M2
    assert _quality(tmp_path, options="Diffusivity 2e-9").diffusivity == 2e-9
    gpm = _quality(tmp_path, options="Diffusivity 1e-5", units="GPM").diffusivity
    assert gpm == pytest.approx(1e-5 * FOOT**2, rel=1e-15)
    assert _quality(tmp_path, options="Diffusivity 0").diffusivity == 0.0


def test_epanets_short_keywords_are_read(tmp_path):
    """`match` is a prefix test of `text.h`'s strings (`QUAL`, `DIFF`, `TOLER`, `GLOB`, `LIMIT`,
    `ROUG`), not of the full words."""
    net = read_epanet_inp(_write(
        tmp_path, "[REACTIONS]\nGLOB BULK -1\nGLOB WALL -2\nLIMIT POTENTIAL 0.5\n"
        "ROUG CORRELATION 0.5\nORDER BULK 2",
        options="QUAL AGE\nDIFF 2\nTOLER 0.5", times="QUAL TIMESTEP 0:10"))
    quality = net.quality
    assert quality.mode == "AGE" and quality.tolerance == 0.5
    assert quality.diffusivity == 2 * DIFFUSIVITY_M2
    assert quality.quality_timestep == 600.0
    assert quality.global_bulk == -1 / 86400 and quality.global_wall == -2 / 86400
    assert quality.limiting_potential == 0.5
    assert quality.roughness_correlation == 0.5 / 86400
    assert quality.bulk_order == 2.0
    assert "unrecognised_options" not in net.notes
    assert "unrecognised_times" not in net.notes
    assert _quality(tmp_path, times="QUAL 0:10").quality_timestep == 600.0


def test_name_and_units_are_overridden_field_by_field(tmp_path):
    """EPANET's `optionchoice`: NONE and CHEM keep the name and units, another word without a
    units token keeps the units, AGE and TRACE set both."""
    kept = _quality(tmp_path, options="Quality Chlorine ug/L\nQuality CHEMICAL")
    assert (kept.chemical, kept.units) == ("Chlorine", "ug/L")
    renamed = _quality(tmp_path, options="Quality Chlorine ug/L\nQuality Fluoride")
    assert (renamed.chemical, renamed.units) == ("Fluoride", "ug/L")
    aged = _quality(tmp_path, options="Quality Chlorine ug/L\nQuality AGE\nQuality CHEM")
    assert (aged.mode, aged.chemical, aged.units) == ("CHEMICAL", "Age", "hrs")
    assert _quality(tmp_path, options="Quality TRACE J1\nQuality NONE").trace_node is None


# ------------------------------------------------------------------ ranges
_NUMERIC_NODES = "J1 0 0\nJ2 0 0\n10 0 0\n11 0 0\n12 0 0\n110 0 0"
_ALPHA_NODES = "J1 0 0\nJ2 0 0\nA 0 0\nAB 0 0\nB 0 0\nBA 0 0"


def test_range_forms_are_numeric_or_lexicographic(tmp_path):
    quality = _quality(tmp_path, "[QUALITY]\n10 12 0.5", junctions=_NUMERIC_NODES)
    assert quality.initial == {"10": 0.5, "11": 0.5, "12": 0.5}
    quality = _quality(tmp_path, "[QUALITY]\nA B 0.3", junctions=_ALPHA_NODES)
    assert quality.initial == {"A": 0.3, "AB": 0.3, "B": 0.3}
    quality = _quality(tmp_path, "[QUALITY]\n10x 12 0.5", junctions=_NUMERIC_NODES)
    assert quality.initial == {"10": 0.5, "11": 0.5, "12": 0.5}
    # a single node is exact, whatever its name
    assert _quality(tmp_path, "[QUALITY]\nJ1 0.7").initial == {"J1": 0.7}


def test_range_forms_for_links(tmp_path):
    pipes = ("[PIPES]\n10 R1 J1 100 100 100\n11 J1 J2 100 100 100\n12 J2 T1 100 100 100\n"
             "110 R1 J2 100 100 100\n")
    quality = _quality(tmp_path, "[REACTIONS]\nBULK 10 12 -1\nWALL 10x 11 -2", pipes=pipes)
    assert quality.pipe_bulk == {"10": -1 / 86400, "11": -1 / 86400, "12": -1 / 86400}
    assert quality.pipe_wall == {"10": -2 / 86400, "11": -2 / 86400}
    pipes = ("[PIPES]\nA R1 J1 100 100 100\nAB J1 J2 100 100 100\nB J2 T1 100 100 100\n"
             "BA R1 J2 100 100 100\n")
    quality = _quality(tmp_path, "[REACTIONS]\nBULK A B -1", pipes=pipes)
    assert set(quality.pipe_bulk) == {"A", "AB", "B"}


def test_range_forms_for_tanks(tmp_path):
    tanks = "[TANKS]\n10 0 2 0 4 10 0\n11 0 2 0 4 10 0\n110 0 2 0 4 10 0\n"
    pipes = _PIPES.replace("T1", "10")
    quality = _quality(tmp_path, "[REACTIONS]\nTANK 10 12 -3\n[MIXING]\n11 FIFO",
                       tanks=tanks, pipes=pipes)
    assert quality.tank_bulk == {"10": -3 / 86400, "11": -3 / 86400}
    assert quality.mixing == {"11": ("FIFO", 1.0)}
    tanks = "[TANKS]\nA 0 2 0 4 10 0\nAB 0 2 0 4 10 0\nBA 0 2 0 4 10 0\n"
    quality = _quality(tmp_path, "[REACTIONS]\nTANK A B -3", tanks=tanks,
                       pipes=_PIPES.replace("T1", "A"))
    assert set(quality.tank_bulk) == {"A", "AB"}


# ------------------------------------------------------------------ sources
def test_legacy_and_mass_sources(tmp_path):
    quality = _quality(tmp_path, "[SOURCES]\nJ1 2.5 PAT\nJ2 MASS 6 *")
    assert quality.sources == (
        QualitySource("J1", "CONCEN", 2.5, "PAT"), QualitySource("J2", "MASS", 0.1, None))
    assert _quality(tmp_path, "[SOURCES]\nJ1 2.5").sources == (
        QualitySource("J1", "CONCEN", 2.5, None),)
    typed = _quality(tmp_path, "[SOURCES]\nJ1 setpoint 3 PAT\nR1 FLOWPACED 1")
    assert typed.sources == (QualitySource("J1", "SETPOINT", 3.0, "PAT"),
                             QualitySource("R1", "FLOWPACED", 1.0, None))


# ------------------------------------------------------------------ reactions
def test_wall_units_follow_the_order_and_the_unit_system(tmp_path):
    order1 = _quality(tmp_path, "[REACTIONS]\nGLOBAL WALL 2\nORDER WALL 1", units="GPM")
    assert order1.global_wall == pytest.approx(2 * FOOT / 86400, rel=1e-15)
    order0 = _quality(tmp_path, "[REACTIONS]\nORDER WALL 0\nGLOBAL WALL 2", units="GPM")
    assert order0.wall_order == 0.0
    assert order0.global_wall == pytest.approx(2 / 86400 / FOOT**2, rel=1e-15)
    late = _quality(tmp_path, "[REACTIONS]\nGLOBAL WALL 2\nORDER WALL 0", units="GPM")
    assert late.global_wall == order0.global_wall
    for order in (0, 1):
        si = _quality(tmp_path, f"[REACTIONS]\nGLOBAL WALL 2\nORDER WALL {order}", units="LPS")
        assert si.global_wall == 2 / 86400
    pipe = _quality(tmp_path, "[REACTIONS]\nWALL P1 4\nROUGHNESS CORRELATION 0.5\n"
                    "LIMITING POTENTIAL 0.2\nORDER BULK 2\nORDER TANK 0", units="GPM")
    assert pipe.pipe_wall == {"P1": pytest.approx(4 * FOOT / 86400, rel=1e-15)}
    assert pipe.roughness_correlation == pytest.approx(0.5 * FOOT / 86400, rel=1e-15)
    assert (pipe.limiting_potential, pipe.bulk_order, pipe.tank_order) == (0.2, 2.0, 0.0)
    bulk = _quality(tmp_path, "[REACTIONS]\nGLOBAL BULK -0.3\nBULK P2 -1\nTANK T1 -2")
    assert bulk.global_bulk == -0.3 / 86400
    assert bulk.pipe_bulk == {"P2": -1 / 86400}
    assert bulk.tank_bulk == {"T1": -2 / 86400}


def test_later_lines_mixing_fractions_and_epanet_orders(tmp_path):
    quality = _quality(tmp_path, "[REACTIONS]\nGLOBAL BULK -1\nGLOBAL BULK -2")
    assert quality.global_bulk == -2 / 86400
    assert _quality(tmp_path, "[MIXING]\nT1 2COMP 0").mixing == {"T1": ("2COMP", 1.0)}
    assert _quality(tmp_path, "[MIXING]\nT1 2comp 0.3").mixing == {"T1": ("2COMP", 0.3)}
    assert _quality(tmp_path, "[MIXING]\nT1 FIFO\nT1 LIFO").mixing == {"T1": ("LIFO", 1.0)}
    text = (
        "[JUNCTIONS]\nJ1 0 0\nJ2 0 0\n[TANKS]\nT1 0 2 0 4 10 0\n[RESERVOIRS]\nR1 10\n"
        "[VALVES]\nV1 J1 J2 100 TCV 5\n[PIPES]\nP1 R1 J1 100 100 100\nP2 J1 J2 100 100 100\n"
        "P3 J2 T1 100 100 100\n[PUMPS]\nPU1 R1 J2 POWER 10\n[END]\n"
    )
    path = tmp_path / "order.inp"
    path.write_text(text)
    quality = read_epanet_inp(path).quality
    assert quality.node_order == ("J1", "J2", "T1", "R1")
    assert quality.link_order == ("V1", "P1", "P2", "P3", "PU1")


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize(
    ("sections", "options", "match"),
    [
        ("[REACTIONS]\nORDER WALL 2", "", "ORDER WALL"),
        ("[REACTIONS]\nGLOBAL TANK -1", "", "GLOBAL TANK"),
        ("", "Quality TRACE", "QUALITY TRACE needs a node.*error 201"),
        ("", "Quality TRACE NOPE", "QUALITY TRACE names the node 'NOPE'.*error 212"),
        ("[QUALITY]\nJ1 -1", "", "negative initial quality"),
        ("[MIXING]\nT1 PLUG", "", "mixing model 'PLUG'.*error 213"),
        ("[QUALITY]\nNOPE 1", "", "node 'NOPE'"),
        ("[REACTIONS]\nBULK NOPE -1", "", "link 'NOPE'"),
        ("[REACTIONS]\nTANK NOPE -1", "", "node 'NOPE'"),
        ("[SOURCES]\nJ1 CONCEN 1 NOPAT", "", "pattern 'NOPAT'"),
        ("[SOURCES]\nJ1 FOO 1", "", "source type"),
        ("[SOURCES]\nNOPE 1", "", "node 'NOPE'"),
        ("[REACTIONS]\nFOO P1 1", "", "FOO.*error 213"),
    ],
)
def test_refusals_name_the_feature(tmp_path, sections, options, match):
    with pytest.raises(ValueError, match=match):
        read_epanet_inp(_write(tmp_path, sections, options=options))


def test_a_refusal_names_the_file_and_line(tmp_path):
    with pytest.raises(ValueError, match=r"q\.inp: line \d+"):
        read_epanet_inp(_write(tmp_path, "[MIXING]\nT1 PLUG"))


def test_lines_epanet_ignores_are_recorded(tmp_path):
    base = read_epanet_inp(_write(tmp_path))
    net = read_epanet_inp(_write(
        tmp_path, "[MIXING]\nJ1 FIFO\n[REACTIONS]\nTANK J1 -1\nBULK PU1 -1\n"))
    assert net.quality == base.quality
    recorded = net.notes["ignored_quality"]
    assert "J1 FIFO" in recorded and "TANK J1 -1" in recorded and "BULK PU1 -1" in recorded
    # a junction is skipped before the model is read, so even an unknown one is ignored
    junction = read_epanet_inp(_write(tmp_path, "[MIXING]\nJ1 PLUG"))
    assert junction.quality.mixing == {} and "J1 PLUG" in junction.notes["ignored_quality"]
    reservoir = read_epanet_inp(_write(tmp_path, "[MIXING]\nR1 FIFO"))
    assert reservoir.quality.mixing == {}
    assert "R1 FIFO" in reservoir.notes["ignored_quality"]


# ------------------------------------------------------------------ the network check
def test_check_refuses_a_hand_built_quality_by_name(tmp_path):
    net = read_epanet_inp(_write(tmp_path))

    def refused(match, **kw):
        with pytest.raises(ValueError, match=match):
            dataclasses.replace(net, quality=WaterQuality(**kw)).check()

    refused("mode 'DYE'", mode="DYE")
    refused("wall order 2", wall_order=2.0)
    refused("TRACE", mode="TRACE", trace_node="NOPE")
    refused("node 'NOPE'", initial={"NOPE": 1.0})
    refused("pipe 'NOPE'", pipe_bulk={"NOPE": 1.0})
    refused("tank 'NOPE'", tank_bulk={"NOPE": 1.0})
    refused("pattern 'NOPAT'", sources=(QualitySource("J1", "CONCEN", 1.0, "NOPAT"),))
    refused("source type 'FOO'", sources=(QualitySource("J1", "FOO", 1.0),))
    refused("mixing model 'PLUG'", mixing={"T1": ("PLUG", 1.0)})
    dataclasses.replace(net, quality=WaterQuality(initial={"J1": 1.0})).check()


@pytest.mark.parametrize("line, seconds", [
    ("QUAL FOO 10", 36000.0), ("Quality Timestep 5 MIN", 300.0), ("QUAL 0:10", 600.0),
    ("Quality Timestep 0:10", 600.0), ("QUAL X Y 90 SEC", 90.0), ("QUAL 2", 7200.0)])
def test_the_quality_step_is_the_last_token_or_the_last_two(tmp_path, line, seconds):
    """`timedata` (`input3.c`) reads the value from the last token, or from the last two when
    the last is a unit, whatever words come between the keyword and it."""
    assert _quality(tmp_path, times=line).quality_timestep == seconds

"""The water-quality data of an EPANET 2.2 network, in SI: what `[QUALITY] [SOURCES]
[REACTIONS] [MIXING]` and the quality keys of `[OPTIONS]` and `[TIMES]` say.

A leaf module (it imports only `units`): `network.WaterNetwork` carries one as `quality`, the
reader fills it (`inp_quality.read_quality`) and the quality run reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from noodl.apps.water.units import DIFFUSIVITY_M2

#: The modes of `[OPTIONS] QUALITY` (`input3.c`, `optionchoice`).
QUALITY_MODES = ("NONE", "CHEMICAL", "AGE", "TRACE")
#: The source types of `[SOURCES]` (`input3.c`, `sourcedata`; `SourceType` in `text.h`).
SOURCE_KINDS = ("CONCEN", "MASS", "SETPOINT", "FLOWPACED")
#: The tank mixing models of `[MIXING]` (`MixTxt` in `text.h`).
MIXING_MODELS = ("MIXED", "2COMP", "FIFO", "LIFO")


@dataclass(frozen=True)
class QualitySource:
    """A water-quality source at a node: `kind` one of `SOURCE_KINDS`, `strength` in quality
    units (a MASS source in mass per second: the file's mass per minute over 60), scaled by
    the time `pattern` when there is one."""

    node: str
    kind: str
    strength: float
    pattern: str | None = None


@dataclass(frozen=True)
class WaterQuality:
    """EPANET's water-quality data, in SI.

    `mode` NONE, CHEMICAL, AGE or TRACE; `chemical` and `units` name the quantity (AGE: "Age",
    "hrs"; TRACE: "Trace <node>", "%"). `quality_timestep` is the FILE's
    quality step (s), None when it gives none (or 0): the consumer applies EPANET's default, a tenth of the hydraulic step (integer seconds, after the
    pattern and report step caps), and the cap Qstep = min(Qstep, Hstep) (`input1.c`,
    `adjustdata`); `tolerance` EPANET's `Ctol` in quality
    units. `diffusivity` is m2/s, 0 switching the wall mass transfer off. The coefficients
    are per second: `global_bulk`, `tank_bulk` and the per-pipe `pipe_bulk` in (quality
    unit)^(1-n)/s for the order n; wall coefficients in m/s for order 1 and quality unit
    m^-2 s^-1 for order 0 (`global_wall`, `pipe_wall`). `roughness_correlation` is the file's
    factor times the wall conversion. NOTE for its consumer: `adjustdata` computes a pipe's
    wall coefficient from the factor BEFORE `convertunits`, so the Darcy-Weisbach form
    Rfactor / |log(roughness / diameter)| uses the file's own units (roughness in millifeet over
    the diameter in inches in a US file) and Hazen-Williams Rfactor / C, Chezy-Manning
    Rfactor * n; the result is then converted as any wall coefficient. `initial` holds the
    node qualities the file sets; `mixing` maps a tank to (model, volume fraction of the mixing zone). `node_order` and
    `link_order` are EPANET's own indices (nodes: junctions, then reservoirs and tanks in file
    order; links: file order across pipes, pumps and valves), () meaning the builder's order.
    """

    mode: str = "NONE"
    chemical: str = "Chemical"
    units: str = "mg/L"
    trace_node: str | None = None
    quality_timestep: float | None = None
    tolerance: float = 0.01
    diffusivity: float = DIFFUSIVITY_M2
    bulk_order: float = 1.0
    wall_order: float = 1.0
    tank_order: float = 1.0
    global_bulk: float = 0.0
    global_wall: float = 0.0
    pipe_bulk: dict[str, float] = field(default_factory=dict)
    pipe_wall: dict[str, float] = field(default_factory=dict)
    tank_bulk: dict[str, float] = field(default_factory=dict)
    limiting_potential: float = 0.0
    roughness_correlation: float = 0.0
    initial: dict[str, float] = field(default_factory=dict)
    sources: tuple[QualitySource, ...] = ()
    mixing: dict[str, tuple[str, float]] = field(default_factory=dict)
    node_order: tuple[str, ...] = ()
    link_order: tuple[str, ...] = ()

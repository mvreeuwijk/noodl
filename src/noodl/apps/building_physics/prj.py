"""CONTAM 3.4 project (.prj) reader -- the documented subset of TN 1887r1 Appendix A.

What is read: run control (ambient conditions, g, the steady simulation's day and time, the
transient run's dates, time step and day type, the weather and ambient-concentration file
names, the density options), the contaminants list and the species records, kinetic
reactions and the zones that carry them, levels, day and week schedules, wind pressure
profiles, airflow elements (power-law family, quadratic family, doorways, dampers,
constant-flow fans, fan curves and cubic splines), zones and their temperature schedules,
initial zone concentrations, filter elements (constant-efficiency, particle and super
filters) and the filters on airflow paths, airflow paths with their schedules (flow
multipliers) and constant wind pressures, source/sink elements and source/sinks of the
constant, cutoff, decaying and burst types, duct networks (`prj_hvac`): duct elements,
junctions and terminals, their initial concentrations, and segments, and simple air-handling
systems (`prj_hvac`, `air_handling`), and control nodes (`prj_controls`): when a record
references one, the nodes an actuator (a zone's, path's, source's, segment's, terminal's or
AHS path's control reference) or a named report node needs become a sampled
`ControlNetwork`, first among the model's closures, and a control reference acts on what a
schedule on the same record acts on. An air handler's plenums (its zones flagged 10) are
zero-gauge nodes of the airflow solve (`Project.flow_balanced`) and storage-free species
nodes when their volume is 0 (`Project.storage_free`); its supply and return points carry
their design flows, scaled by their schedules; the recirculation path carries the
outdoor-air fraction schedule and the minimum outdoor-air flow, as ContamX reads them; the
flows are `AirHandlerFlows` closures (`Project.closures`). Everything else is skipped to its
-999 terminator. Refused by name (ValueError): a control node of an unsupported type that an
actuator or a named report node depends on, a control on a plain duct junction, on a
triggered source or on an AHS outdoor-air or exhaust path, a non-trace species, a gaseous
(loading-dependent) or UVGI filter element (referenced or not), a CFD or 1-D zone, a zone
held at constant pressure (other than an air handler's plenum), an unsupported element type,
a constant wind pressure on an interior path, duct leakage, a value file on a junction or
segment, a schedule or a control on a Darcy-Colebrook duct segment, a duct network with
densZP = 1 or dodMdt = 1, an air handler with densZP = 1, and the AHS paths ContamX
mis-assigns or whose fields it ignores (see `prj_hvac`). A schedule number that no week
schedule carries raises KeyError naming the zone, path, junction or segment.

Control nodes that nothing uses are not built: NIST's own sample projects carry dozens of
sensor and report nodes (and integrations, which this reader does not build) that no zone and
no path points at, and those projects must load.

Conventions: mass flow; orifice/leak/crack turbulence coefficient
`turb` is C_d A sqrt(2) so C = mult turb sqrt(rho); fcn/test/conn/stair/shaft C = mult turb;
qcn C = mult rho turb; laminar `lam` gives F = lam (rho/mu) dp, so the laminar/turbulent
crossing is dp_t = (C_mass mu / (lam rho))^(1/(1-n)). One element KIND per CONTAM element
(`pl_<nr>`, `qf_<nr>`, `door_<nr>`, `bd_<nr>`, `fan_<nr>`), so every element keeps its own
transition and the transport layers advect over all of them.

ONE DOCUMENTED EXCEPTION to that crossing rule: the DOORWAY kinds (`dor_door`, `dor_pl2`) do
NOT derive `dp_transition` from their record's `lam`. Each doorway becomes two half-area
openings built directly from `ht`/`wd`/`cd`, and that branch never reaches the derivation
above, so both openings take `PowerLaw`'s generic 1e-3 Pa default. What the formula WOULD
give, for the two records this repository carries (`tests/data/contam/doorway_damper_fan.prj`,
at mult = 1, n = 0.5): 2.7e-4 Pa for the `dor_pl2` record and 2.0e-8 Pa for the `dor_door`
one -- both BELOW the default, so on both the regularised laminar blend is entered earlier
(at a larger |dp|) than ContamX would enter it. The error this can cause is bounded twice
over: it is confined to |dp| < 1e-3 Pa, where the two laws differ by a fraction of the flow
that a 1e-3 Pa pressure difference drives at all; and NO doorway flow is compared against ContamX
anywhere (the three-zone and one-zone stack parity projects carry no doorway at all, and the
one parity test that loads `doorway_damper_fan.prj` compares path 5, its fan, alone). Deriving
it properly would move test numbers and is not done here.

The `rho` in those coefficients is the density of the air ENTERING the path, not a fixed
reference: ContamX re-evaluates it per path and per flow direction. Every power-law kind is
therefore built as an `UpstreamDensityPowerLaw`, whose `C` is the coefficient at RHO_0 (as
above) and which multiplies it by `(rho_upstream / RHO_0)**m` at solve time, m being 0, 1/2
or 1 for the three CONTAM families. Freezing it at RHO_0 instead costs about 0.086% per
kelvin of zone-to-ambient temperature difference -- 1.7% in mass flow on a 20 K stack. The
quadratic, damper and fan families keep their reference-density coefficients (they are not
part of the ContamX comparison); `dp_t` above likewise stays at RHO_0.

Dtype: `torch.get_default_dtype()` is float32 in this repository, and a CONTAM
parity comparison is a float64 exercise. The `Network`, every element parameter and every
tensor `Project` carries are built with an EXPLICIT float64 dtype -- never by letting a
Python float fall through to the default.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from noodl.apps.building_physics.air_handling import (
    EXHAUST,
    OUTDOOR_AIR,
    RECIRCULATION,
    RETURN,
    SUPPLY,
    AirHandlerFlows,
    air_handler_from_edges,
)
from noodl.apps.building_physics.controls import (
    PRESSURE_DIFFERENCE,
    ControlNetwork,
    DriverReading,
    StateReading,
    node_mass_fraction,
)
from noodl.apps.building_physics.filters import (
    ConstantEfficiencyFilter,
    FilterSeries,
    ParticleFilter,
    filter_transmission,
)
from noodl.apps.building_physics.inputs import Input, apply_inputs, bind_inputs
from noodl.apps.building_physics.occupants import (
    Occupancy,
    OccupancyDay,
    Occupant,
    OccupantGeneration,
    occupant_inputs,
)
from noodl.apps.building_physics.prj_controls import (
    INPUT_KEY,
    Actuation,
    ControlPlan,
    PrjControlNode,
    control_network,
    plan_controls,
    read_control_nodes,
)
from noodl.apps.building_physics.prj_hvac import (
    AirHandlerRecord,
    BuildContext,
    assign_air_handler_roles,
    build_ducts,
    read_air_handlers,
    read_duct_elements,
    read_duct_segments,
    read_junctions,
)
from noodl.apps.building_physics.schedules import Calendar, DaySchedule, WeekSchedule
from noodl.apps.building_physics.species import KineticReaction, Species, reaction_kinetics
from noodl.apps.building_physics.thermal import (
    R_AIR,
    RHO_0,
    check_flow_balanced,
    species_layer,
)
from noodl.apps.building_physics.wth import _DAYS_BEFORE_MONTH
from noodl.drives import ConstantDrive, Stack, Wind, WindProfile
from noodl.elements import (
    AffinityFan,
    CubicSplineFlow,
    Damper,
    FanCurve,
    FixedFlow,
    Quadratic,
    Scaled,
    UpstreamDensityPowerLaw,
)
from noodl.layers.potential import PotentialFlowLayer
from noodl.layers.transport import active_interior
from noodl.model import Model
from noodl.nodesources import NodeSource
from noodl.topology import Network

F64 = torch.float64
MU_0 = 1.81625e-5

# --------------------------------------------------------------------------------------
# RECORD LAYOUTS. Every field position this reader depends on lives here and nowhere else;
# all of it is verified against NIST's `contamxpy` 0.0.9 sample projects unless a comment
# says otherwise.
#
#   header line 1   `ContamW <version> <flag>`, then the project's own file name.
#   run control     labelled comment lines each followed by one or more data lines, to -999.
#                   `! Ta Pb Ws Wd rh day u..` heads SEVERAL data lines, one per simulation
#                   case, each labelled by its OWN trailing comment; the ambient state is
#                   the one labelled `steady simulation`, and its field 5 (`daytyp`) is the
#                   steady simulation's day type. The lines `<name> ! weather file` and
#                   `<name> ! contaminant file` name the weather and ambient-concentration
#                   files, `null` meaning none. `!sim_af` heads one line whose field 0 is 1
#                   for a transient airflow. `!sim_mf` heads four lines: field 0 of the first
#                   is the contaminant mode (0 none, 1 steady, 2 transient, 3 cyclic), and the
#                   line labelled `! (trace)` carries the integration weight `gamma` at field 6
#                   (1 = implicit Euler). `!mf_solver` field 0 is the integration method
#                   (0 trapezoid, 1 short time step, 2 CVODE). `!dens grav` heads `rho g`.
#                   `!tsdens relax tsmaxi cnvgSS densZP stackD dodMdt` heads the density
#                   options; `densZP` (field 4), `stackD` (field 5) and `dodMdt` (field 6) are
#                   read, see `_read_density_options`. `!date_st` heads
#                   `date_st time_st date_0 time_0 date_1 time_1 t_step t_list t_scrn`: the
#                   steady simulation's date and time, the transient run's start and end, its
#                   time step and output step; dates are `MmmDD` (`Jan01`), times `hh:mm:ss`.
#   section header  `N ! <name>:`; the section's N records then run to a `-999` line.
#                   `! contaminants:` is the exception: N index lines and NO terminator.
#   contaminants    `N ! contaminants:` then the N species NUMBERS that are simulated, in the
#                   order ContamX reports them (one line, no -999).
#   species         `# s t molwt mdiam edens decay Dm CCdef Cp Kuv u[5] name`, then a
#                   description line: `s` 1 simulated / 0 not, `t` 1 non-trace / 0 trace;
#                   `CCdef` (field 8) is the default ambient mass fraction (kg/kg); `decay`
#                   is ContamW-only data that ContamX does not apply (measured).
#   kinetic reaction `nr nkrd name`, a description line, then `nkrd` lines
#                   `product reactant coefficient` (1/s); dx_product/dt += coefficient
#                   x_reactant (measured).
#   filter element  `nr type area depth dens u_A u_D name`, a description line (may be
#                   blank), then by type: `cef` a line `nspcs`, then `nspcs` lines
#                   `species efficiency`; `pf0` a line `npts u_size`, then `npts` lines
#                   `diameter(m) efficiency`; `spf` a line `nsub`, then the `nsub` element
#                   numbers in flow order. ContamX's type table is `cef pf0 gf0 uv1 spf`;
#                   `gf0` (gaseous, loading-dependent) and `uv1` (UVGI, layout undocumented)
#                   are refused when read.
#   filter          `nr element nsub`, then per sub-element a line `load nsload` and `nsload`
#                   species-loading lines (measured: for two sub-elements ContamX refuses the
#                   lines `0`, `0` and loads `0 0`, `0 0`; the split layout is refused). Loadings
#                   matter only to gaseous filters and are skipped.
#   day schedule    `# npts shap utyp ucnv name`, a description line (may be blank), then
#                   `npts` rows `hh:mm:ss value` from 00:00:00 to 24:00:00. `shap` 0 is
#                   rectangular, 1 trapezoidal; values are SI (K for a temperature schedule),
#                   `ucnv` is the display unit only.
#   week schedule   `# utyp ucnv name`, a description line, then one line of 12 day-schedule
#                   NUMBERS, one per day type (`j1 .. j12`).
#   levels          `# refHt delHt ni u name`, then `ni` icon lines (comments interleaved).
#   wind profile    `# npts type name`, one description line (may be blank), `npts` rows
#                   `angle cp`; the last angle is 360 and repeats the Cp at 0.
#   flow element    `# icon dtype name`, ONE description line (often blank -- it must be
#                   CONSUMED, not skipped), then the data line(s).
#     plr_orfc      `lam turb expt area dia coef Re u_A u_D`
#     plr_leak3     `lam turb expt coef pres area1 area2 area3 u u u u`
#     fan_cmf       `Flow u_F`
#     fan_fan       line 1 `lam turb expt rdens fdf sop off`, line 2 `a0 a1 a2 a3 npts
#                   Sarea u`, then `npts` measured-point lines (unused). The
#                   cubic P(F) (Pa, F in kg/s) is rated at density `rdens` and holds up to
#                   flow `fdf`; below speed ratio `off` the fan is the shut-off power law
#                   `lam turb expt`. A path schedule on it is the speed ratio and the path
#                   `mult` is fans in parallel (both measured against ContamX 3.4.1.7).
#     csf_*         `npts u_x u_y`, then `npts` rows `x y` (fsp: dp Pa, kg/s; qsp: dp, m3/s;
#                   psf: kg/s, dp; psq: m3/s, dp); fitted as `noodl.elements.curve` says.
#     dor_door      10 fields: `lam turb expt dTmin ht wd cd u u u`. VERIFIED against
#                   `reg_solverContTrace-mz-MH-trans-3day.prj` lines 704-706, the one NIST
#                   sample that carries a doorway:
#                       18 27 dor_door IntDoor-open
#                       open interior door
#                        0.148966 2.54558 0.5 0.01 2 0.9 1 0 0 0
#                   Reading that as ht = 2, wd = 0.9, cd = 1 reproduces the record's OWN
#                   turbulent coefficient through CONTAM's identity turb = cd A sqrt(2):
#                   1 * (2 * 0.9) * sqrt(2) = 2.545584, against the file's 2.54558 -- six
#                   significant figures, so fields 4, 5 and 6 are pinned by NIST's numbers.
#     dor_pl2       `lam turb expt dH    ht wd cd u u`
#                   -- the two doorway types differ ONLY in field 3 (a minimum temperature
#                   difference versus the half-separation of the two openings); `ht wd cd`
#                   sit at 4, 5, 6 in both, which is why the code below has ONE assignment
#                   for them. The `dor_pl2` half is documented from TN 1887r1
#                   Appendix A only: no NIST sample carries a dor_pl2, so only `dor_door`
#                   is file-verified.
#     plr_bdq/bdf   `lam Cp xp Cn xn ...`
#   zone            19 fields: `# f s# c# k# l# relHt Vol T0 P0 name clr uH uT uP uV axs
#                   cdvf cfd`. `clr` is not in ContamW's own header comment but IS written.
#                   `k#` (field 4) is the zone's kinetic reaction number, 0 for none.
#                   `s#` is a week schedule of the zone's temperature (K). `f` is a bit set;
#                   value 1 is the variable-pressure bit: without it ContamX holds the zone at
#                   constant pressure (measured on ContamX 3.4.1.7: flags 0 and 2 give no flow
#                   through the zone, 1 and 3 the same flows). Value 16 leaves the zone out
#                   of the building volume (measured on ContamX 3.4.1.7: flags 19 leave it
#                   out; 1, 2, 7, 11, 35 and 67 do not); the other bits are not read.
#   path            30 fields: `# f n# m# e# f# w# a# s# c# l# X Y relHt mult wPset wPmod
#                   wazm Fahs Xmax Xmin icn dir clr u[4] cdvf cfd`. `f` (field 1) is a bit
#                   set whose values 8, 16, 32 and 64 mark the paths of a simple air-handling
#                   system, `a#` (field 7) its number and `Fahs` (field 18) its design flow
#                   (`prj_hvac` lists their meaning). `f#` (field 5) is the
#                   path's filter number, 0 for none; the filter acts on the air crossing the
#                   path in either direction (measured). Same story: the header
#                   comment omits `clr` (always -1 in every sample), so the trailing
#                   `cdvf cfd` sit at 28 and 29, not at 27 and 28. A nonzero cdvf or cfd
#                   appends a name / four data fields, which is why the field COUNT is
#                   checked. `n# == -1` is ambient; `wPmod` is Ch; `w#` is the 1-based wind
#                   pressure profile NUMBER. `s#` is a week schedule multiplying the path's
#                   flow. With `w# == 0`, a nonzero `wPset` is a constant wind pressure (Pa)
#                   on the path's ambient side: positive pushes air from ambient into the zone
#                   (measured on ContamX 3.4.1.7: wPset +5 / -5 Pa on the three-zone sample's
#                   ambient -> zone path 1 gives +0.1812 / -0.1654 kg/s through it).
#   source/sink element `nr species type name`, a description line (may be blank), then
#                   the data line(s) by type (TN 1887r1 Appendix A section 9):
#     ccf           `G R u_G u_R` (kg/s; kg/s per unit mass fraction)
#     cut           `G x_cut u_G u_C` (kg/s; kg/kg)
#     eds           `G0 k u_G u_k` (kg/s; 1/s, the decay rate)
#     brs           `M u_M` (kg)
#     prs           `G x u_G` (kg/(s Pa^x); -)
#     dvs           `v_d A u_V u_A` (m/s; m2)
#     drs           `k_d u_k` (1/s)
#     plm           `a b t_p u_a u_tp` (kg/s; -; s); ContamX raises the time to the power
#                   in HOURS (measured)
#     pkm           `a b t_p u_a u_tp` (kg/s; -; s)
#     sup           `n`, then a line of the n sub-element numbers
#     bls, dvr      one data line each; refused when referenced
#   source/sink     `# z# e# s# c# mult CC0 (X,Y,H)min (X,Y,H)max clr u[1] cdvf cfd`.
#   occupancy schedule `# npts u_XYZ name`, a description line (may be blank), then `npts`
#                   lines `hh:mm:ss zone X Y relHt`; zone 0 is out of the building. X, Y and
#                   relHt only place an occupant in a 1-D zone and are ignored. The last time
#                   must be 24:00:00: without it ContamX 3.4.1.7 stops during the run.
#   exposure        `# gen ncg cgmlt` and one more integer: ContamX 3.4.1.7 refuses the
#                   record without the fifth field (TN 1887 lists four), whose value has no
#                   effect. Then a description line, a line of 12 occupancy-schedule numbers
#                   (one per day type) and `ncg` lines `species ps cgmax u_cg vf_type
#                   [vf_node]`: `ps` a week schedule (0 none), `cgmax` in kg/s per person,
#                   multiplied by `cgmlt` persons. With `gen` 0 nothing is generated.
# --------------------------------------------------------------------------------------

_ZONE_FIELDS = 19
_PATH_FIELDS = 30
_STEADY_LABEL = "steady simulation"

_POWERLAW_SQRT_RHO = {"plr_orfc", "plr_leak1", "plr_leak2", "plr_leak3", "plr_crack"}
_POWERLAW_MASS = {"plr_fcn", "plr_test1", "plr_test2", "plr_conn", "plr_stair", "plr_shaft"}
_POWERLAW_VOLUME = {"plr_qcn"}
_POWERLAW = _POWERLAW_SQRT_RHO | _POWERLAW_MASS | _POWERLAW_VOLUME
_QUADRATIC = {"qfr_qab", "qfr_fab", "qfr_crack", "qfr_test2"}
_DOOR = {"dor_door", "dor_pl2"}
_DAMPER = {"plr_bdq", "plr_bdf"}
_FAN_CONST = {"fan_cmf", "fan_cvf"}
_SOURCE_TYPES = {"ccf", "cut", "eds", "brs", "prs", "dvs", "drs", "plm", "pkm"}
_REFUSED_SOURCES = {
    "bls": "a boundary-layer diffusion sink, which needs the adsorbent's own storage state",
    "dvr": "deposition with resuspension, which needs the surface loading as state",
    "sup": "a super source/sink element (several elements as one)",
}
# A schedule TRIGGERS these (once, at its first nonzero value); it is not a multiplier.
_TRIGGERED_SOURCES = {"eds", "brs", "plm", "pkm"}
_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_CONTAMINANT_MODES = {0: "none", 1: "steady", 2: "transient", 3: "cyclic"}
_INTEGRATION = {0: "trapezoid", 1: "short_time_step", 2: "cvode"}
_SHAPES = {0: "rectangular", 1: "trapezoidal"}
_VARIABLE_PRESSURE = 1                  # the zone-flags bit of a zone whose pressure is solved
_VARIABLE_CONTAMINANTS = 2              # ... and of a zone whose concentrations are solved
_OUTSIDE_BUILDING_VOLUME = 16           # ... and of a zone left out of the building volume

@dataclass
class PrjElement:
    nr: int
    dtype: str
    name: str
    data: list[list[float]]          # data lines, tokens as floats


@dataclass
class PrjPath:
    nr: int
    from_zone: int                   # -1 = ambient
    to_zone: int
    element_nr: int
    z: float
    mult: float
    wPset: float
    Ch: float
    azimuth: float
    profile_nr: int
    edge_columns: list[int] = field(default_factory=list)   # columns of q for this path
    schedule_nr: int = 0             # week schedule multiplying the flow; 0 = none
    filter_nr: int = 0               # filter on the path (either flow direction); 0 = none
    flags: int = 0                   # the path's flag bits (8..64: an air handler's paths)
    ahs_nr: int = 0                  # the AHS of a supply or return point; 0 = none
    design_flow: float = 0.0         # an AHS path's design flow (kg/s)
    control_nr: int = 0              # control node acting as its schedule would; 0 = none


@dataclass
class PrjTerminal:
    junction_nr: int
    zone_nr: int                     # the zone the terminal opens into; -1 = ambient
    edge_columns: list[int]          # its column of q


@dataclass
class PrjSegment:
    nr: int
    edge_columns: list[int]          # its column of q


@dataclass
class PrjSource:
    nr: int
    zone_nr: int
    element_nr: int
    source_type: str                 # ccf | cut | eds | brs | prs | dvs | drs | plm | pkm
    params: list[float]
    mult: float
    species: int = 0                 # 0-based index of the source element's species
    schedule_nr: int = 0             # week schedule multiplying the rate; 0 = none
    control_nr: int = 0              # control node acting as its schedule would; 0 = none


@dataclass(frozen=True)
class RunControl:
    """A project's run control: what to simulate and when.

    Airflow transient or steady; contaminants none/steady/transient/cyclic; the integration
    method and its weight (1 = implicit Euler); the steady simulation's day of the year and
    time of day (s); the transient run's start and end day of the year and time of day (s);
    time step and output step (s); the steady weather's day type (1-12); the weather and
    ambient-concentration file names (None when the project names none); whether a zone's
    air-mass change enters its airflow balance (CONTAM's dodMdt option).
    """

    transient_airflow: bool
    contaminants: str
    integration: str
    integration_weight: float
    steady_day: int
    steady_time: float
    start_day: int
    start_time: float
    end_day: int
    end_time: float
    time_step: float
    output_step: float
    day_type: int
    weather_file: str | None
    ambient_file: str | None
    zone_mass_change: bool

    @property
    def duration(self) -> float:
        """Length of the transient run (s)."""
        return (self.end_day - self.start_day) * 86400.0 + self.end_time - self.start_time

    def steady_calendar(self) -> Calendar:
        """The calendar of the steady simulation: its day, time and day type."""
        return Calendar(start_day=self.steady_day, start_time=self.steady_time % 86400.0,
                        start_day_type=self.day_type)


def _clock(hms: str) -> float:
    h, m, s = (int(v) for v in hms.split(":"))
    return 3600.0 * h + 60.0 * m + s


def _code(table: dict[int, str], token: str, what: str) -> str:
    try:
        return table[int(token)]
    except ValueError as exc:
        raise ValueError(f"prj: {what} {token!r} is not an integer code") from exc
    except KeyError as exc:
        raise ValueError(f"prj: {what} {token} is not one of {sorted(table)}") from exc


def _date(token: str) -> int:
    try:
        month = _MONTHS.index(token[:3]) + 1
        day = int(token[3:])
    except ValueError as exc:
        raise ValueError(f"prj: {token!r} is not a CONTAM date (MmmDD)") from exc
    if not 1 <= day <= _DAYS_IN_MONTH[month - 1]:
        raise ValueError(
            f"prj: {token!r} is not a CONTAM date: {_MONTHS[month - 1]} has days "
            f"1-{_DAYS_IN_MONTH[month - 1]}"
        )
    return _DAYS_BEFORE_MONTH[month - 1] + day


@dataclass
class Project:
    """A CONTAM project as read by `read_prj`: the network, its airflow elements and
    drives, zones and their temperatures and volumes, species, initial mass fractions,
    sources and ambient conditions, the run control, and the schedules as `inputs`.
    `project_to_model` builds the model from it."""

    net: Network
    elements: list
    drives: list
    zones: list[str]
    ambient: str
    paths: list[PrjPath]
    species: list[str]
    T_zone: torch.Tensor
    zone_volumes: torch.Tensor
    g: float
    ambient_conditions: dict[str, float]
    profiles: dict[int, WindProfile]
    x0: torch.Tensor                  # (n_zones, K)
    sources: list[PrjSource]
    kinds: list[str]
    # CONTAM numbers zones; it does not promise they are 1..N in file order, and
    # `sources_from_project` resolves a source's `z#` through this map rather than
    # assuming `zones[nr - 1]`.
    zone_nr_to_name: dict[int, str]
    # The file's `densZP` run-control flag (see `_read_density_options`).
    density_uses_zone_pressure: bool = False
    run_control: RunControl | None = None
    # Week schedules by CONTAM number, and the zone and path schedules as inputs: zone
    # temperatures on "T", path flow multipliers on "flow_multiplier.<kind>", fan-curve
    # speed ratios on "fan_speed.<kind>".
    schedules: dict[int, WeekSchedule] = field(default_factory=dict)
    inputs: list[Input] = field(default_factory=list)
    # Each species' default ambient mass fraction (K,), kg/kg.
    ambient_mass_fractions: Tensor | None = None
    # Element kinds wrapped in `Scaled` (a scheduled path among their edges): kind -> edges.
    multiplier_edges: dict[str, int] = field(default_factory=dict)
    # Fan-curve kinds with a speed schedule ("fan_speed.<kind>"): kind -> edges.
    speed_edges: dict[str, int] = field(default_factory=dict)
    # Constant wind pressure per edge of a kind (Pa, on the ambient side), kind -> (n_edges,).
    constant_wind_pressure: dict[str, Tensor] = field(default_factory=dict)
    # Each zone's flag bits as the file gives them, by zone name.
    zone_flags: dict[str, int] = field(default_factory=dict)
    source_path: Path | None = None
    # The simulated species' records, in the order of `species`.
    species_records: list[Species] = field(default_factory=list)
    # Each zone's kinetic reaction, by zone name.
    zone_reactions: dict[str, KineticReaction] = field(default_factory=dict)
    # The filter on each filtered edge, by the edge's `(source, target, key)`.
    edge_filters: dict = field(default_factory=dict)
    # Duct junction nodes in junction-number order (their species rows follow the zones'),
    # their initial mass fractions (n_junctions, K), the terminals and segments with their
    # columns of q, each week schedule's ContamW unit type, and the number of terminals
    # (one opening each, on "terminal_opening.duct_terminal") when one is scheduled.
    junctions: list[str] = field(default_factory=list)
    junction_x0: torch.Tensor | None = None
    terminals: list[PrjTerminal] = field(default_factory=list)
    segments: list[PrjSegment] = field(default_factory=list)
    schedule_units: dict[int, int] = field(default_factory=dict)
    terminal_openings: int = 0
    # Simple air-handling systems: their plenums (zero gauge pressure in the airflow solve),
    # the zero-volume ones (no air held), their `AirHandlerFlows` closures, and each one's
    # kind with its number of edges ("flow_multiplier.<kind>", "outdoor_air_fraction.<kind>").
    flow_balanced: list[str] = field(default_factory=list)
    storage_free: list[str] = field(default_factory=list)
    closures: list = field(default_factory=list)
    air_handler_edges: dict[str, int] = field(default_factory=dict)
    # Duct junction node names by CONTAM junction number.
    junction_nr_to_name: dict[int, str] = field(default_factory=dict)
    # The control nodes as read (by number), and the plan of the ones the model builds
    # (None: no actuator and no named report node); `project_to_model` builds a new
    # `ControlNetwork` from the plan for every model.
    control_nodes: dict[int, PrjControlNode] = field(default_factory=dict)
    control_plan: ControlPlan | None = None
    # The occupants (CONTAM's exposures), in exposure-number order: "occupant <nr>".
    occupants: list = field(default_factory=list)

    @property
    def building_zones(self) -> list[str]:
        """The nodes of the building's conditioned space, as ContamX counts it for its
        building air change rate (measured): every zone without the "outside the building
        volume" flag (air handlers' plenums included), in zone order, then every duct
        junction."""
        inside = [z for z in self.zones if not self.zone_flags.get(z, 0)
                  & _OUTSIDE_BUILDING_VOLUME]
        return inside + list(self.junctions)

    def inputs_for(self, model) -> list[Input]:
        """The project's inputs that `model` can take: a model without a species layer (a run
        control that simulates no contaminants) has no source multipliers to write and no
        occupants' generation (their presence and schedule multipliers) to spread."""
        if "species" in model.transport:
            return list(self.inputs)
        occupant_keys = {k for o in self.occupants for k in (o.presence_key, o.multiplier_key)}
        return [i for i in self.inputs
                if not i.key.startswith("species.") and i.key not in occupant_keys]

    def path_flows(self, q: torch.Tensor) -> torch.Tensor:
        """Net mass flow per path in path-number order (a doorway sums its two edges).

        `q` is the air layer's flow vector, which `PotentialFlowLayer` lays out in element-
        KIND blocks, in `self.elements` order -- NOT in path order. Each path's
        `edge_columns` already carries that translation.
        """
        cols = [torch.as_tensor(p.edge_columns, dtype=torch.long) for p in self.paths]
        return torch.stack([q[..., c].sum(-1) for c in cols], dim=-1)

    def terminal_flows(self, q: torch.Tensor) -> torch.Tensor:
        """Flow out of each duct terminal into its zone (kg/s), in junction-number order
        (ContamX's `getDuctTerminalFlow` order and sign)."""
        cols = torch.tensor([t.edge_columns[0] for t in self.terminals], dtype=torch.long)
        return q[..., cols]

    def segment_flows(self, q: torch.Tensor) -> torch.Tensor:
        """Flow along each duct segment, from its first junction to its second (kg/s), in
        segment-number order."""
        cols = torch.tensor([s.edge_columns[0] for s in self.segments], dtype=torch.long)
        return q[..., cols]

    def zone_mass_fractions(self, x: torch.Tensor) -> torch.Tensor:
        """The zones' rows of a species state (the junctions' rows follow them)."""
        return x[..., : len(self.zones), :]

    def sensed_edge(self, kind: int, nr: int) -> tuple[int, float]:
        """The `"air.q"` column and sign of what a sensor of `kind` (2 path, 4 duct segment,
        6 terminal) `nr` measures, as `path_flows`, `segment_flows` and `terminal_flows`
        read them; refuses a two-way path (two edges)."""
        if kind == 2:
            path = next((p for p in self.paths if p.nr == nr), None)
            if path is None:
                raise ValueError(f"prj: no path {nr}")
            if len(path.edge_columns) != 1:
                raise ValueError(f"prj: path {nr} is a two-way opening (two edges)")
            return int(path.edge_columns[0]), 1.0
        if kind == 4:
            seg = next((s for s in self.segments if s.nr == nr), None)
            if seg is None:
                raise ValueError(f"prj: no duct segment {nr}")
            return int(seg.edge_columns[0]), 1.0
        term = next((t for t in self.terminals if t.junction_nr == nr), None)
        if term is None:
            raise ValueError(f"prj: no terminal at junction {nr}")
        return int(term.edge_columns[0]), 1.0

    def control_outputs(self, history: Tensor) -> Tensor:
        """The named report nodes' values (ContamX's API outputs, in node order) from a
        history of the network's outputs `(..., n_blocks)`."""
        if self.control_plan is None:
            raise ValueError("prj: the project has no control plan (no actuator and no named "
                             "report node)")
        positions = [pos for _, pos in self.control_plan.outputs]
        return history[..., positions]


class _Lines:
    """Cursor over the file's lines with the two ways CONTAM records are laid out."""

    def __init__(self, text: str) -> None:
        self.lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
        self.i = 0

    def raw(self) -> str:
        if self.i >= len(self.lines):
            raise ValueError("prj: file ended in the middle of a record (no '* end project')")
        line = self.lines[self.i]
        self.i += 1
        return line

    def record(self) -> list[str]:
        """Next non-comment, non-blank line, split on whitespace, trailing `!` comment cut."""
        while True:
            line = self.raw()
            body = line.split("!", 1)[0].strip()
            if body:
                return body.split()

    def skip_section(self) -> None:
        while self.raw().strip() != "-999":
            pass

    def expect_end(self, where: str) -> None:
        tok = self.record()
        if tok != ["-999"]:
            raise ValueError(f"prj: expected -999 after {where}, found {tok}")


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _section_header(lines: _Lines) -> tuple[int, str]:
    line = lines.raw()
    while not line.strip():
        line = lines.raw()
    count, sep, name = line.partition("!")
    if not sep or not count.split():
        raise ValueError(f"prj: expected a 'N ! <section>:' header, found {line.strip()!r}")
    return int(count.split()[0]), name.strip().rstrip(":").strip()


def _read_ambient_block(lines: _Lines) -> tuple[dict[str, float], int] | None:
    """Consume the data lines under one `! Ta Pb Ws Wd` header; return the steady one and
    its day type.

    CONTAM writes one such line per simulation case (steady, wind pressure test, ...), each
    labelled by its own trailing comment. Only the `steady simulation` line is the project's
    ambient state -- taking the first or the last of them is wrong.
    """
    found: tuple[dict[str, float], int] | None = None
    while lines.i < len(lines.lines):
        body, _, comment = lines.lines[lines.i].partition("!")
        tok = body.split()
        if len(tok) < 4 or not all(_is_number(t) for t in tok[:4]):
            break                                   # a comment line, or the next block
        lines.i += 1
        if _STEADY_LABEL in comment:
            if len(tok) < 6:
                raise ValueError(
                    f"prj: the {_STEADY_LABEL!r} line has no day type (field 5): "
                    f"{lines.lines[lines.i - 1].strip()!r}"
                )
            found = ({
                "Ta": float(tok[0]), "Pb": float(tok[1]),
                "Ws": float(tok[2]), "Wd": float(tok[3]),
            }, int(tok[5]))
    return found


def _read_density_options(line: str) -> tuple[bool, bool]:
    """`densZP` and `dodMdt` from the `!tsdens relax tsmaxi cnvgSS densZP stackD dodMdt`
    data line.

    `densZP = 1` makes ContamX evaluate each zone's air density at the zone's ABSOLUTE
    pressure, `Pb + P_zone`, instead of at the barometric pressure `Pb` alone. The
    correction is tiny in the density itself (|P_zone| / Pb, ~6e-6 for a 20 K stack over
    1.5 m) but a stack flow is driven by a density DIFFERENCE, which amplifies it by
    rho / |delta rho| (~14 at 20 K): ignoring it left a flat 4.1-4.4e-5 relative gap to
    ContamX over the whole +-20 K sweep of `test_OneZoneWthCtmStack-UseApi.prj`, the one
    fixture that sets it. `project_to_model` honours it (see `ZonePressureDensity`).

    `stackD = 1` asks ContamX for a variable-density stack-pressure calculation (density
    varying with height rather than one value per zone); noodl physics' `Stack` holds one density
    per node, so a project asking for it is REFUSED rather than silently simulated with
    uniform densities. `tsdens`, `relax`, `tsmaxi` and `cnvgSS` govern ContamX's own
    iteration. `dodMdt = 1` puts each zone's air-mass change dM/dt into its airflow balance,
    which matters when a zone temperature changes (a schedule); it is returned for the run
    control, and `simulate`'s `air_mass_change` implements it.
    """
    tok = line.split()
    if len(tok) < 7:
        raise ValueError(
            f"prj: the '!tsdens ... densZP stackD dodMdt' data line has {len(tok)} fields, "
            f"expected at least 7: {line.strip()!r}"
        )
    dens_zp, stack_d = int(tok[4]), int(tok[5])
    if stack_d != 0:
        raise ValueError(
            f"prj: stackD = {stack_d} asks for a variable-density stack calculation "
            f"(density varying with height); noodl physics holds one density per node, so "
            f"this is unsupported"
        )
    return dens_zp != 0, int(tok[6]) != 0


def _read_run_control(lines: _Lines) -> tuple[dict[str, float], float, bool, RunControl]:
    ambient: dict[str, float] | None = None
    day_type: int | None = None
    g: float | None = None
    dens_zp = False
    fields: dict = {}
    files: dict[str, str | None] = {}
    while True:
        line = lines.raw()
        stripped = line.strip()
        if stripped == "-999":
            break
        body, _, comment = line.partition("!")
        if not stripped.startswith("!") and ("weather file" in comment
                                             or "contaminant file" in comment):
            name = body.strip()
            files["weather" if "weather file" in comment else "ambient"] = (
                None if name.lower() == "null" else name)
            continue
        if not stripped.startswith("!"):
            continue
        if stripped[1:].split()[:1] == ["Ta"]:
            block = _read_ambient_block(lines)
            if block is not None:
                ambient, day_type = block
        elif stripped.startswith("!tsdens"):
            dens_zp, fields["zone_mass_change"] = _read_density_options(lines.raw())
        elif stripped.startswith("!dens"):
            g = float(lines.raw().split()[1])
        elif stripped.startswith("!sim_af"):
            mode = int(lines.raw().split()[0])
            if mode not in (0, 1):
                raise ValueError(
                    f"prj: the run control's airflow mode is {mode} (duct balancing, a "
                    f"building airflow test or a pressurization test), which is unsupported")
            fields["transient_airflow"] = mode == 1
        elif stripped.startswith("!sim_mf"):
            fields["contaminants"] = _code(_CONTAMINANT_MODES, lines.raw().split()[0],
                                           "the contaminant simulation mode (sim_mf)")
            for _ in range(3):
                row = lines.raw()
                if "(trace)" in row:
                    fields["integration_weight"] = float(row.split()[6])
        elif stripped.startswith("!mf_solver"):
            fields["integration"] = _code(_INTEGRATION, lines.raw().split()[0],
                                          "the contaminant integration method (mf_solver)")
        elif stripped.startswith("!date_st"):
            d = lines.raw().split()
            fields.update(steady_day=_date(d[0]), steady_time=_clock(d[1]),
                          start_day=_date(d[2]), start_time=_clock(d[3]),
                          end_day=_date(d[4]), end_time=_clock(d[5]),
                          time_step=_clock(d[6]), output_step=_clock(d[7]))
    if ambient is None:
        raise ValueError(
            "prj: the run-control section has no 'Ta Pb Ws Wd' data line labelled "
            f"{_STEADY_LABEL!r}; that line IS the project's ambient state, and the other "
            "cases (a wind pressure test, say) are not interchangeable with it"
        )
    if g is None:
        raise ValueError("prj: the run-control section lacks the '!dens grav' line")
    fields["day_type"] = day_type
    for key in ("weather", "ambient"):
        if key in files:
            fields[f"{key}_file"] = files[key]
    for f in dataclasses.fields(RunControl):
        if f.name not in fields:
            raise ValueError(f"prj: the run-control section lacks {f.name}")
    return ambient, g, dens_zp, RunControl(**fields)


def _read_contaminants(lines: _Lines, count: int) -> list[int]:
    """The `N ! contaminants:` section: N species NUMBERS, then no terminator.

    The indices are whitespace-separated and CONTAM writes them all on ONE line (`   1 2 3`),
    so this counts TOKENS, not lines. A per-line loop happens to work for a single-species
    project and then eats the `N ! species:` header of every other one -- verified on
    `reg_solverContTrace-mz-MH-trans-3day.prj` (2 contaminants, `   1 2`) and
    `test_OneFloorWpcAddMf.prj` (3, `   1 2 3`). The section has no -999 of its own: the
    species section follows it directly.
    """
    numbers: list[int] = []
    while len(numbers) < count:
        numbers += [int(v) for v in lines.record()]
    if len(numbers) != count:
        raise ValueError(
            f"prj: the 'contaminants' section promises {count} species indices but its "
            f"lines carry {len(numbers)}"
        )
    return numbers


class ContamCompatibilityWarning(UserWarning):
    """The reader follows ContamX where ContamX ignores data the file carries.

    Filter it with `warnings.filterwarnings("ignore", category=ContamCompatibilityWarning)`.
    """


def _warn_ignored_decay(species: list[Species]) -> None:
    """One warning per simulated species whose record carries a nonzero decay constant."""
    for sp in species:
        if sp.decay != 0.0:
            warnings.warn(
                f"prj: species {sp.name!r} has a decay constant of {sp.decay:g} 1/s. ContamX "
                f"ignores a species' decay, so noodl does too. To model the decay, add a "
                f"DepositionRateSink(node, rate) in each zone, or a KineticReaction with a "
                f"negative coefficient of the species on itself.",
                ContamCompatibilityWarning, stacklevel=3)


def _read_species(lines: _Lines, count: int, out: list[tuple[int, bool, Species]]) -> None:
    """`(number, simulated, Species)` per record; the name is the last field."""
    for _ in range(count):
        tok = lines.record()
        description = lines.raw().strip()           # the species' description line
        out.append((int(tok[0]), tok[1] == "1", Species(
            tok[-1], molar_mass=float(tok[3]), mean_diameter=float(tok[4]),
            effective_density=float(tok[5]), decay=float(tok[6]),
            diffusivity=float(tok[7]), default_mass_fraction=float(tok[8]),
            non_trace=tok[2] == "1", uv_susceptibility=float(tok[10]),
            description=description)))
    lines.expect_end("species")


def _simulated_species(rows: list[tuple[int, bool, Species]],
                       contaminants: list[int]) -> list[Species]:
    """The simulated species in file order, which must be the contaminants list."""
    simulated = [nr for nr, sim, _ in rows if sim]
    if contaminants != simulated:
        raise ValueError(
            f"prj: the 'contaminants' section lists species {contaminants} but the species "
            f"section marks {simulated} as simulated; a contaminant list that differs from "
            f"the simulated species in file order is unsupported")
    out = []
    for nr, sim, sp in rows:
        if sim and sp.non_trace:
            raise ValueError(
                f"prj: species {nr} ({sp.name!r}) is non-trace: it changes the air density, "
                f"which noodl physics does not model (unsupported)")
        if sim:
            out.append(sp)
    return out


def _read_reactions(lines: _Lines, count: int, out: dict[int, KineticReaction]) -> None:
    """`nr nkrd name`, a description line, then `nkrd` lines `product reactant k`; a pair
    given twice is refused."""
    for _ in range(count):
        tok = lines.record()
        nr, n_pairs = int(tok[0]), int(tok[1])
        name = tok[2] if len(tok) > 2 else ""
        lines.raw()                                 # description (may be blank)
        coefficients: dict[tuple[str, str], float] = {}
        for _ in range(n_pairs):
            product, reactant, coef = lines.record()[:3]
            pair = (product, reactant)
            if pair in coefficients:
                raise ValueError(
                    f"prj: kinetic reaction {nr} ({name!r}) gives the pair (product "
                    f"{product!r}, reactant {reactant!r}) twice; a repeated pair is "
                    f"unsupported")
            coefficients[pair] = float(coef)
        out[nr] = KineticReaction(coefficients, name=name)
    lines.expect_end("kinetic reactions")


_REFUSED_FILTERS = {
    "gf0": "a gaseous filter whose efficiency depends on its loading",
    "uv1": "an ultraviolet germicidal irradiation (UVGI) filter",
}


def _read_filter_elements(lines: _Lines, count: int,
                          out: dict[int, tuple[str, str, list]]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, ftype = int(tok[0]), tok[1]
        name = tok[7] if len(tok) > 7 else ""
        lines.raw()                                 # description (may be blank)
        if ftype in ("cef", "pf0"):
            rows = [lines.record() for _ in range(int(lines.record()[0]))]
            data = ([(r[0], float(r[1])) for r in rows] if ftype == "cef"
                    else [(float(r[0]), float(r[1])) for r in rows])
            out[nr] = (ftype, name, data)
        elif ftype == "spf":
            n = int(lines.record()[0])
            subs: list[int] = []
            while len(subs) < n:
                subs += [int(v) for v in lines.record()]
            out[nr] = (ftype, name, subs)
        else:
            why = _REFUSED_FILTERS.get(ftype, "an unknown filter type")
            raise ValueError(f"prj: filter element {nr} ({name!r}) has type {ftype!r}, {why} "
                             f"(unsupported)")
    lines.expect_end("filter elements")


def _read_filters(lines: _Lines, count: int, out: dict[int, int]) -> None:
    """Filter number -> filter element number; initial loadings are skipped."""
    for _ in range(count):
        tok = lines.record()
        nr, element, n_sub = int(tok[0]), int(tok[1]), int(tok[2])
        for _ in range(n_sub):
            row = lines.record()                    # `load nsload`
            if len(row) < 2:
                raise ValueError(f"prj: filter {nr} needs the pair `load nsload` on one line "
                                 f"for each sub-element, got {' '.join(row)!r} (ContamX "
                                 f"refuses the split layout)")
            n_loads = int(row[1])
            for _ in range(n_loads):
                lines.record()
        out[nr] = element
    lines.expect_end("filters")


def _filter_object(nr: int, elements: dict[int, tuple[str, str, list]],
                   species: list[str], chain: tuple[int, ...] = ()):
    """Filter element `nr` as a filter object (a super filter's stages recursively)."""
    try:
        ftype, name, data = elements[nr]
    except KeyError as exc:
        raise KeyError(f"prj: filter element {nr} is not defined in the 'filter elements' "
                       f"section") from exc
    if ftype == "cef":
        unknown = sorted(s for s, _ in data if s not in species)
        if unknown:
            raise KeyError(f"prj: filter element {nr} ({name!r}) names species {unknown}, "
                           f"which the project does not simulate")
        return ConstantEfficiencyFilter(dict(data), name=name)
    if ftype == "pf0":
        try:
            return ParticleFilter([d for d, _ in data], [e for _, e in data], name=name)
        except ValueError as exc:
            raise ValueError(f"prj: filter element {nr}: {exc}") from exc
    if nr in chain:
        raise ValueError(f"prj: filter element {nr} ({name!r}) contains itself")
    return FilterSeries([_filter_object(k, elements, species, (*chain, nr)) for k in data],
                        name=name)


def _read_day_schedules(lines: _Lines, count: int, out: dict[int, DaySchedule]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, npts, shape = int(tok[0]), int(tok[1]), int(tok[2])
        if shape not in _SHAPES:
            raise ValueError(f"prj: day schedule {nr} has shape {shape}; 0 or 1 expected")
        lines.raw()                                 # description (may be blank)
        pts = [lines.record() for _ in range(npts)]
        out[nr] = DaySchedule([_clock(p[0]) for p in pts], [float(p[1]) for p in pts],
                              shape=_SHAPES[shape], name=tok[5] if len(tok) > 5 else "")
    lines.expect_end("day-schedules")


def _read_week_schedules(lines: _Lines, count: int, days: dict[int, DaySchedule],
                         out: dict[int, WeekSchedule], units: dict[int, int]) -> None:
    """Week schedules by number, and each one's ContamW unit type into `units`."""
    order = sorted(days)
    for _ in range(count):
        tok = lines.record()
        nr = int(tok[0])
        units[nr] = int(tok[1])
        lines.raw()                                 # description (may be blank)
        js = [int(v) for v in lines.record()]
        missing = sorted({j for j in js if j not in days})
        if missing:
            raise KeyError(f"prj: week schedule {nr} names day schedule(s) {missing}, which "
                           f"the 'day-schedules' section does not define")
        out[nr] = WeekSchedule([days[k] for k in order], [order.index(j) + 1 for j in js],
                               name=tok[3] if len(tok) > 3 else "")
    lines.expect_end("week-schedules")


@dataclass
class PrjExposure:
    """An exposure record: one occupant (or `multiplier` people) as the file gives it."""

    nr: int
    generates: bool
    multiplier: float
    description: str
    day_schedules: list[int]                       # occupancy schedule per day type 1-12
    generation: list[tuple[str, int, float]]       # (species, week schedule, kg/s per person)


def _read_occupancy_schedules(lines: _Lines, count: int,
                              out: dict[int, list[tuple[float, int]]]) -> None:
    """Occupancy schedules by number: (time of day in s, zone number; 0 out) per point."""
    for _ in range(count):
        tok = lines.record()
        nr, npts = int(tok[0]), int(tok[1])
        lines.raw()                                   # description (may be blank)
        pts = [lines.record() for _ in range(npts)]
        out[nr] = [(_clock(pt[0]), int(pt[1])) for pt in pts]
    lines.expect_end("occupancy schedules")


def _read_exposures(lines: _Lines, count: int, out: list[PrjExposure]) -> None:
    """Exposure records in file order; a generation read from a value file is refused."""
    for _ in range(count):
        tok = lines.record()
        if len(tok) < 5:
            raise ValueError(f"prj: exposure record {' '.join(tok)!r} has {len(tok)} fields; "
                             f"ContamX reads five fields (nr gen ncg cgmlt and one more) and "
                             f"refuses this record")
        nr, gen, ncg, mult = int(tok[0]), int(tok[1]), int(tok[2]), float(tok[3])
        description = lines.raw().strip()
        days = [int(v) for v in lines.record()]
        if len(days) != 12:
            raise ValueError(f"prj: exposure {nr} lists {len(days)} occupancy schedules; "
                             f"12 (one per day type) expected")
        generation = []
        for _ in range(ncg):
            g = lines.record()
            vf = int(g[4]) if len(g) > 4 else 0
            if vf:
                raise ValueError(f"prj: exposure {nr}'s generation of {g[0]!r} reads a value "
                                 f"file (vf_type {vf}); value files are unsupported")
            generation.append((g[0], int(g[1]), float(g[2])))
        out.append(PrjExposure(nr, gen != 0, mult, description, days, generation))
    lines.expect_end("exposures")


def _read_initial_concentrations(
    lines: _Lines, count: int, x0_rows: dict[int, list[float]], *,
    what: str = "initial zone concentrations",
) -> None:
    """One row per zone (or junction), `Z# x1 .. xK`; the section's count is VALUES, not
    rows.

    Every other section's header count is a record count, but this one's is zones times
    species: `test_OneFloorWpcAddMf.prj:171` heads TWO zone rows with `6` (2 zones, 3
    species). So read rows to the terminator and use the count as the integrity check it
    is. A single-species project hides the difference, which is why this was invisible on
    the shipped fixtures.
    """
    values = 0
    while True:
        tok = lines.record()
        if tok == ["-999"]:
            break
        x0_rows[int(tok[0])] = [float(v) for v in tok[1:]]
        values += len(tok) - 1
    if values != count:
        raise ValueError(
            f"prj: the {what!r} section promises {count} values (rows x species) but its "
            f"rows carry {values}"
        )


def _read_levels(lines: _Lines, count: int, levels: dict[int, float]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, refht, ni = int(tok[0]), float(tok[1]), int(tok[3])
        levels[nr] = refht
        skipped = 0
        while skipped < ni:                         # icon lines, with comments interleaved
            if lines.raw().split("!", 1)[0].strip():
                skipped += 1
    lines.expect_end("levels")


_PROFILE_CURVES = {1: "linear", 2: "spline"}


def _read_profiles(lines: _Lines, count: int, profiles: dict[int, WindProfile]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, npts, fit = int(tok[0]), int(tok[1]), int(tok[2])
        lines.raw()                                 # description (may be blank)
        pts = [lines.record() for _ in range(npts)]
        # `type`: 1 linear, 2 cubic spline, 3 trigonometric (Walker and Wilson's fit to the
        # 0/90/180/270 points).
        if fit == 3:
            raise ValueError(
                f"prj: wind pressure profile {nr} ({tok[3] if len(tok) > 3 else 'unnamed'}) "
                f"uses the trigonometric curve fit, which is unsupported; use the linear or "
                f"cubic-spline fit")
        if fit not in _PROFILE_CURVES:
            raise ValueError(
                f"prj: wind pressure profile {nr} has curve-fit type {fit}; CONTAM defines "
                f"1 (linear), 2 (cubic spline) and 3 (trigonometric)")
        profiles[nr] = WindProfile([float(p[0]) for p in pts], [float(p[1]) for p in pts],
                                   curve=_PROFILE_CURVES[fit])
    lines.expect_end("wind pressure profiles")


def _read_source_elements(
    lines: _Lines, count: int, source_elements: dict[int, tuple[str, str, list[float]]]
) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, species_name, dtype = int(tok[0]), tok[1], tok[2]
        lines.raw()                                 # description (may be blank)
        data = [float(v) for v in lines.record()]
        if dtype == "sup":                          # `n`, then n sub-element numbers
            subs: list[str] = []
            while len(subs) < int(data[0]):
                subs += lines.record()
        source_elements[nr] = (species_name, dtype, data)
    lines.expect_end("source/sink elements")


def _read_flow_elements(lines: _Lines, count: int, elements: dict[int, PrjElement]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, dtype, ename = int(tok[0]), tok[2], tok[3] if len(tok) > 3 else ""
        lines.raw()                                 # description (may be blank)
        data = [[float(v) for v in lines.record()]]
        if dtype == "fan_fan":
            # Cubic fan curve: field 4 of the second data line is the number of measured
            # points, each of which is a line of its own (TN 1887r1 Appendix A; verified
            # by loading `fan_curve_three_zone.prj` in ContamX 3.4.1.7).
            second = [float(v) for v in lines.record()]
            data.append(second)
            for _ in range(int(second[4])):
                data.append([float(v) for v in lines.record()])
        elif dtype.startswith("csf_"):
            for _ in range(int(data[0][0])):        # `npts u_x u_y`, then npts `x y` rows
                data.append([float(v) for v in lines.record()])
        elif dtype == "sup_afe":
            raise ValueError(
                f"prj: element {nr} has type 'sup_afe' (a super airflow element), which "
                f"ContamX 3.4.1.7 itself cannot simulate ('Invalid AFE type'); unsupported")
        elements[nr] = PrjElement(nr, dtype, ename, data)
    lines.expect_end("flow elements")


def _read_zones(lines: _Lines, count: int, zones: list[tuple],
                zone_reactions: dict[int, int]) -> None:
    for _ in range(count):
        tok = lines.record()
        nr = int(tok[0])
        if len(tok) != _ZONE_FIELDS:
            raise ValueError(
                f"prj: zone {nr} line has {len(tok)} fields, not the {_ZONE_FIELDS} of a "
                f"plain zone; zone names with spaces, and the extra fields a value file or "
                f"a CFD zone writes, are unsupported: {' '.join(tok)}"
            )
        flags, ps, pc, pk, pl = (int(v) for v in tok[1:6])
        if pk:
            zone_reactions[nr] = pk
        rel_ht, vol, T0 = float(tok[6]), float(tok[7]), float(tok[8])
        zname = tok[10]
        rest = tok[11:]
        axs, cdvf, cfd = int(rest[5]), int(rest[6]), int(rest[7])
        if axs:
            raise ValueError(f"prj: zone {nr} is a 1-D convection/diffusion zone (unsupported)")
        if cfd:
            raise ValueError(f"prj: zone {nr} is a CFD zone (unsupported)")
        if cdvf:
            raise ValueError(f"prj: zone {nr} reads a continuous values file (unsupported)")
        zones.append((nr, zname, rel_ht, vol, T0, pl, ps, pc, flags))
    lines.expect_end("zones")


def _read_paths(
    lines: _Lines, count: int, paths: list[PrjPath], levels: dict[int, float]
) -> None:
    for _ in range(count):
        tok = lines.record()
        nr = int(tok[0])
        if len(tok) != _PATH_FIELDS:
            raise ValueError(
                f"prj: path {nr} line has {len(tok)} fields, not the {_PATH_FIELDS} of a "
                f"plain path; a continuous values file or a CFD path writes extra name and "
                f"data fields, and neither is supported: {' '.join(tok)}"
            )
        pzn, pzm, pe, pf, pw, pa, ps, pc, pld = (int(v) for v in tok[2:11])
        rel_ht, mult, wPset, wPmod, wazm, design_flow = (float(v) for v in tok[13:19])
        cdvf, cfd = int(float(tok[28])), int(float(tok[29]))
        if cdvf or cfd:
            raise ValueError(
                f"prj: path {nr} uses a values file (cdvf {cdvf}) or CFD (cfd {cfd}) "
                f"(unsupported)"
            )
        if pld not in levels:
            raise KeyError(f"prj: path {nr} sits on level {pld}, which the file never defines")
        paths.append(
            PrjPath(nr, pzn, pzm, pe, levels[pld] + rel_ht, mult, wPset, wPmod, wazm, pw,
                    schedule_nr=ps, filter_nr=pf, flags=int(tok[1]), ahs_nr=pa,
                    design_flow=design_flow, control_nr=pc)
        )
    lines.expect_end("flow paths")


def _read_sources(
    lines: _Lines,
    count: int,
    sources: list[PrjSource],
    source_elements: dict[int, tuple[str, str, list[float]]],
    species: list[str],
) -> None:
    for _ in range(count):
        tok = lines.record()
        nr, pz, pe, ps, pc = (int(v) for v in tok[:5])
        try:
            sname, dtype, params = source_elements[pe]
        except KeyError as exc:
            raise KeyError(
                f"prj: source {nr} references source/sink element {pe}, which the "
                f"'source/sink elements' section does not define"
            ) from exc
        stype = dtype.split("_")[-1] if "_" in dtype else dtype
        if stype in _REFUSED_SOURCES:
            raise ValueError(f"prj: source {nr} uses element {pe} of type {stype!r}, "
                             f"{_REFUSED_SOURCES[stype]} (unsupported)")
        if stype not in _SOURCE_TYPES:
            raise ValueError(f"prj: source {nr} uses element type {dtype!r} (unsupported)")
        if ps and stype in _TRIGGERED_SOURCES:
            raise ValueError(
                f"prj: source {nr} is a {stype} source on schedule {ps}; schedule-triggered "
                f"decaying, burst, power-law and peak sources are unsupported")
        if pc and stype in _TRIGGERED_SOURCES:
            raise ValueError(f"prj: source {nr} references control node {pc}: a control on a "
                             f"triggered source is not supported")
        try:
            k = species.index(sname)
        except ValueError as exc:
            raise KeyError(
                f"prj: source {nr}'s element {pe} names species {sname!r}, which the project "
                f"does not simulate") from exc
        sources.append(PrjSource(nr, pz, pe, stype, params, float(tok[5]), k, ps, pc))
    lines.expect_end("source/sinks")


def _laminar_transition(C: float, lam: float, n: float) -> float:
    """dp_t (Pa) where CONTAM's laminar law F = lam (rho / mu) dp meets the power law
    F = C dp^n, at RHO_0 and MU_0; 1e-3 Pa when the element has no laminar coefficient."""
    if lam > 0 and n < 1:
        return (C * MU_0 / (lam * RHO_0)) ** (1.0 / (1.0 - n))
    return 1e-3


def _check_edge_order(net: Network, kind: str, cols_in_append_order: list[int]) -> None:
    """Refuse per-edge data appended in an order other than the network's edge order of
    `kind`, which `endpoints(kind)` and the flow layer's dp block follow: a mismatch would
    pair each edge with another edge's coefficients and upstream node, silently."""
    cols_in_network_order = net.edge_index(kind).tolist()
    if cols_in_append_order != cols_in_network_order:
        raise RuntimeError(
            f"prj: element kind {kind!r} built its per-edge coefficients in edge order "
            f"{cols_in_append_order} but the network orders that kind's edges "
            f"{cols_in_network_order}; the upstream-density endpoints would be misaligned"
        )


def fan_curve_element(data: list[list[float]], *, kind: str, src, tgt, multiplier,
                      speed_key: str | None) -> AffinityFan:
    """CONTAM's performance-curve fan by the fan laws. `data` holds the element's lines:
    `lam turb expt rdens fdf sop off`, then the cubic's four coefficients (Pa, in kg/s) and
    the measured points (unused). The shut-off orifice is the element's own power law, in
    the sqrt(rho) family with the upstream density."""
    lam, turb, expt, rdens, fdf, _sop, off = data[0][:7]
    curve = FanCurve(torch.tensor(data[1][:4], dtype=F64), torch.tensor(fdf, dtype=F64),
                     kind=kind)
    C = turb * math.sqrt(RHO_0)
    shutoff = UpstreamDensityPowerLaw(
        torch.tensor(C, dtype=F64), torch.tensor(expt, dtype=F64), src=src, tgt=tgt,
        m=torch.tensor(0.5, dtype=F64), rho_ref=RHO_0,
        dp_transition=_laminar_transition(C, lam, expt), kind=kind)
    return AffinityFan(curve, src=src, tgt=tgt, rho_ref=rdens, speed_key=speed_key, off=off,
                       multiplier=torch.tensor(multiplier, dtype=F64), shutoff=shutoff,
                       kind=kind)


_SPLINE_FORMS = {"fsp": ("pressure", False), "qsp": ("pressure", True),
                 "psf": ("flow", False), "psq": ("flow", True)}
_SPLINE = {f"csf_{form}" for form in _SPLINE_FORMS}


def spline_element(dtype: str, points: list[list[float]], *, kind: str, src, tgt,
                   multiplier) -> CubicSplineFlow:
    """CONTAM's cubic-spline element (`csf_*`, or the duct `dct_fsp/qsp/psf/psq`) from its
    `x y` points: F(dp), Q(dp), dp(F) or dp(Q)."""
    independent, volume = _SPLINE_FORMS[dtype[-3:]]
    return CubicSplineFlow(torch.tensor([p[0] for p in points], dtype=F64),
                           torch.tensor([p[1] for p in points], dtype=F64),
                           independent=independent, volume=volume,
                           multiplier=torch.tensor(multiplier, dtype=F64), src=src, tgt=tgt,
                           kind=kind)


def _kind_of(nr: int, dtype: str) -> str:
    """One element KIND per CONTAM element, so each keeps its own laminar transition."""
    if dtype in _POWERLAW:
        return f"pl_{nr}"
    if dtype in _QUADRATIC:
        return f"qf_{nr}"
    if dtype in _DOOR:
        return f"door_{nr}"
    if dtype in _DAMPER:
        return f"bd_{nr}"
    if dtype in _FAN_CONST or dtype == "fan_fan":
        return f"fan_{nr}"
    if dtype in _SPLINE:
        return f"sp_{nr}"
    raise ValueError(f"prj: element {nr} has unsupported type {dtype!r}")


def read_prj(path) -> Project:
    """A CONTAM `.prj` project as a `Project`: zones, airflow paths and their elements,
    species and sources; `project_to_model` turns it into a model."""
    source_path = Path(path).resolve()
    text = source_path.read_text()
    lines = _Lines(text)
    header = lines.raw().split()
    if not header or header[0] != "ContamW":
        raise ValueError(f"prj: not a ContamW project file (first line {header!r})")
    lines.raw()                                   # the project's own file name line
    ambient_conditions, g, dens_zp, run_control = _read_run_control(lines)

    species: list[str] = []
    contaminants: list[int] = []
    species_rows: list[tuple[int, bool, Species]] = []
    species_records: list[Species] = []
    reactions: dict[int, KineticReaction] = {}
    zone_reactions: dict[int, int] = {}
    day_schedules: dict[int, DaySchedule] = {}
    schedules: dict[int, WeekSchedule] = {}
    levels: dict[int, float] = {}
    profiles: dict[int, WindProfile] = {}
    elements: dict[int, PrjElement] = {}
    source_elements: dict[int, tuple[str, str, list[float]]] = {}
    filter_elements: dict[int, tuple[str, str, list]] = {}
    filters: dict[int, int] = {}
    # nr, name, relHt, vol, T0, level, week schedule, control node, flags
    zones: list[tuple[int, str, float, float, float, int, int, int, int]] = []
    x0_rows: dict[int, list[float]] = {}
    paths: list[PrjPath] = []
    sources: list[PrjSource] = []
    schedule_units: dict[int, int] = {}
    duct_elements: dict = {}
    junctions: list = []
    junction_x0_rows: dict[int, list[float]] = {}
    segments: list = []
    handlers: list[AirHandlerRecord] = []
    control_nodes: dict[int, PrjControlNode] = {}
    occupancy: dict[int, list[tuple[float, int]]] = {}
    exposures: list[PrjExposure] = []

    while lines.i < len(lines.lines):
        line = lines.lines[lines.i].strip()
        if line.startswith("*"):
            break
        if not line:
            lines.i += 1
            continue
        count, name = _section_header(lines)
        if name == "contaminants":
            contaminants = _read_contaminants(lines, count)
        elif name == "species":
            # Resolved at once: the source/sinks section later in the file looks its species
            # names up in `species`. The contaminants section always precedes this one.
            _read_species(lines, count, species_rows)
            species_records = _simulated_species(species_rows, contaminants)
            _warn_ignored_decay(species_records)
            species[:] = [s.name for s in species_records]
        elif name == "kinetic reactions":
            _read_reactions(lines, count, reactions)
        elif name == "day-schedules":
            _read_day_schedules(lines, count, day_schedules)
        elif name == "week-schedules":
            _read_week_schedules(lines, count, day_schedules, schedules, schedule_units)
        elif name == "levels plus icon data":
            _read_levels(lines, count, levels)
        elif name == "wind pressure profiles":
            _read_profiles(lines, count, profiles)
        elif name == "source/sink elements":
            _read_source_elements(lines, count, source_elements)
        elif name == "filter elements":
            _read_filter_elements(lines, count, filter_elements)
        elif name == "filters":
            _read_filters(lines, count, filters)
        elif name == "flow elements":
            _read_flow_elements(lines, count, elements)
        elif name == "zones":
            _read_zones(lines, count, zones, zone_reactions)
        elif name == "initial zone concentrations":
            _read_initial_concentrations(lines, count, x0_rows)
        elif name == "flow paths":
            _read_paths(lines, count, paths, levels)
        elif name == "source/sinks":
            _read_sources(lines, count, sources, source_elements, species)
        elif name == "duct elements":
            read_duct_elements(lines, count, duct_elements)
        elif name == "duct junctions":
            read_junctions(lines, count, junctions, levels)
        elif name == "initial junction concentrations":
            _read_initial_concentrations(lines, count, junction_x0_rows,
                                         what="initial junction concentrations")
        elif name == "duct segments":
            read_duct_segments(lines, count, segments)
        elif name == "simple AHS":
            read_air_handlers(lines, count, handlers)
        elif name == "control nodes":
            control_nodes = read_control_nodes(lines, count)
        elif name == "occupancy schedules":
            _read_occupancy_schedules(lines, count, occupancy)
        elif name == "exposures":
            _read_exposures(lines, count, exposures)
        else:
            lines.skip_section()

    return _build(
        ambient_conditions=ambient_conditions, g=g, species=species, levels=levels,
        profiles=profiles, elements=elements, zones=zones, x0_rows=x0_rows, paths=paths,
        sources=sources, dens_zp=dens_zp, schedules=schedules,
        species_records=species_records, reactions=reactions, zone_reactions=zone_reactions,
        run_control=run_control, source_path=source_path,
        filter_elements=filter_elements, filters=filters, schedule_units=schedule_units,
        duct_elements=duct_elements, junctions=junctions, junction_x0_rows=junction_x0_rows,
        segments=segments, handlers=handlers, control_nodes=control_nodes,
        occupancy=occupancy, exposures=exposures,
    )


def _build(*, ambient_conditions, g, species, levels, profiles, elements, zones, x0_rows,
           paths, sources, dens_zp, schedules, species_records, reactions, zone_reactions,
           run_control, source_path, filter_elements, filters, schedule_units, duct_elements,
           junctions, junction_x0_rows, segments, handlers, control_nodes, occupancy=None,
           exposures=None) -> Project:
    """Turn the parsed records into a `Network`, elements, drives and a `Project`.

    Order: the zones, the path edges and their elements, the path filters, constant wind
    pressures and drives, the `Scaled` wrap of scheduled path kinds, then the duct network
    (junctions, segments, terminals, with their own elements, drives, filters and inputs).
    The air handlers' paths become one prescribed-flow kind per handler, built after the
    `Scaled` wrap (their closure applies the schedules). Only then are the element kinds and
    the flow layout read, so both cover the ducts and the air handlers.

    A record's control reference becomes an `Actuation` wherever its schedule becomes an
    `Input`, with the same driver key and positions (and the same side effects: a path kind
    wrapped in `Scaled`, a fan driven by its speed); where a schedule is refused, so is a
    control. The control plan and its schedule nodes' inputs follow the records, then the
    occupants (the exposure records, each with its occupancy schedules) and their inputs."""
    net = Network(dtype=F64)
    ambient = "ambient"
    net.add_node(ambient, z_ref=0.0, volume=0.0, T0=ambient_conditions["Ta"])
    zone_names: list[str] = []
    zone_nr_to_name: dict[int, str] = {}
    T_list, V_list = [], []
    zone_flags: dict[str, int] = {}
    plenums = {z for h in handlers for z in (h.return_zone, h.supply_zone)}
    if handlers and dens_zp:
        raise ValueError("prj: densZP = 1 in a project with a simple air-handling system is "
                         "unsupported (its plenums' densities at their absolute pressure are "
                         "not modelled)")
    for nr, zname, rel_ht, vol, T0, pl, _ps, _pc, flags in zones:
        zone_flags[zname] = flags
        if not flags & _VARIABLE_PRESSURE and nr not in plenums:
            raise ValueError(
                f"prj: zone {nr} ({zname!r}) has flags {flags}, without the "
                f"variable-pressure bit (1): ContamX holds such a zone at constant pressure "
                f"(unsupported; only an air handler's plenum zones may lack it)"
            )
        if species and not flags & _VARIABLE_CONTAMINANTS:
            raise ValueError(
                f"prj: zone {nr} ({zname!r}) has flags {flags}, without the "
                f"variable-contaminants bit (2): ContamX holds such a zone at its initial "
                f"concentrations and ignores its sources (unsupported)")
        if pl not in levels:
            raise KeyError(f"prj: zone {nr} sits on level {pl}, which the file never defines")
        net.add_node(zname, z_ref=levels[pl] + rel_ht, volume=vol, T0=T0, heat_capacity=0.0,
                     in_building_volume=float(not flags & _OUTSIDE_BUILDING_VOLUME))
        zone_names.append(zname)
        zone_nr_to_name[nr] = zname
        T_list.append(T0)
        V_list.append(vol)

    by_zone: dict[str, KineticReaction] = {}
    for znr, k in zone_reactions.items():
        try:
            reaction = reactions[k]
        except KeyError as exc:
            raise KeyError(f"prj: zone {znr} references kinetic reaction {k}, which the "
                           f"'kinetic reactions' section does not define") from exc
        try:
            reaction.matrix(species)
        except KeyError as exc:
            raise KeyError(f"prj: kinetic reaction {k} (zone {znr}): {exc.args[0]}") from exc
        by_zone[zone_nr_to_name[znr]] = reaction

    def node_of(z: int) -> str:
        if z == -1:
            return ambient
        try:
            return zone_nr_to_name[z]
        except KeyError as exc:
            raise KeyError(
                f"prj: a path references zone {z}, which the file never defines"
            ) from exc

    def week(nr: int, what: str) -> WeekSchedule:
        try:
            return schedules[nr]
        except KeyError as exc:
            raise KeyError(f"prj: {what} references schedule {nr}, which the "
                           f"'week-schedules' section does not define") from exc

    inputs: list[Input] = []
    actuations: list[Actuation] = []
    for nr, zname, *_rest, ps, pc, _flags in zones:
        if ps:
            inputs.append(Input(week(ps, f"zone {nr}"), "T", index=net.node_index(zname),
                                name=f"zone {nr} temperature schedule {ps}"))
        if pc:
            actuations.append(Actuation(pc, "T", (net.node_index(zname),), f"zone {nr}"))

    ahs_roles = (assign_air_handler_roles(handlers, paths)
                 if handlers or any(p.ahs_nr for p in paths) else {})
    for p in paths:
        if p.nr not in ahs_roles:
            continue
        _, role = ahs_roles[p.nr]
        if p.element_nr:
            raise ValueError(f"prj: path {p.nr} is an AHS path but names flow element "
                             f"{p.element_nr}; its flow is the air handler's (unsupported)")
        if p.profile_nr or p.wPset:
            raise ValueError(f"prj: path {p.nr} is an AHS path with a wind pressure "
                             f"(profile {p.profile_nr}, wPset {p.wPset}); its flow is the air "
                             f"handler's (unsupported)")
        if role in (OUTDOOR_AIR, EXHAUST) and (p.schedule_nr or p.design_flow):
            raise ValueError(
                f"prj: path {p.nr} (an AHS outdoor-air or exhaust path) has a schedule or a "
                f"flow; ContamX ignores both there (the outdoor-air fraction and minimum are "
                f"the recirculation path's), so they are refused")
        if role in (OUTDOOR_AIR, EXHAUST) and p.control_nr:
            raise ValueError(
                f"prj: path {p.nr} (an AHS outdoor-air or exhaust path) references control "
                f"node {p.control_nr}; ContamX ignores a schedule there (the outdoor-air "
                f"fraction is the recirculation path's), so a control is refused too")
        if role in (SUPPLY, RETURN) and p.mult != 1.0:
            raise ValueError(f"prj: path {p.nr} (an AHS supply or return point) has mult "
                             f"{p.mult}; ContamX ignores it, so it is refused")

    kind_of: dict[int, str] = {}
    for nr in sorted({p.element_nr for p in paths if p.nr not in ahs_roles}):
        try:
            el = elements[nr]
        except KeyError as exc:
            raise KeyError(
                f"prj: a path references flow element {nr}, which the 'flow elements' "
                f"section does not define"
            ) from exc
        kind_of[nr] = _kind_of(nr, el.dtype)

    for j, src in enumerate(sources):
        if src.schedule_nr:
            inputs.append(Input(week(src.schedule_nr, f"source {src.nr}"),
                                "species.source_multiplier", index=j,
                                name=f"source {src.nr} schedule {src.schedule_nr}"))
        if src.control_nr:
            actuations.append(Actuation(src.control_nr, "species.source_multiplier", (j,),
                                        f"source {src.nr}"))

    built: list = []
    per_kind_C: dict[str, list[float]] = {}
    per_kind_n: dict[str, list[float]] = {}
    # Density exponent m of the upstream-density correction, per edge: 0 for the mass-flow
    # family, 1/2 for the sqrt(rho) family (orifice/leak/crack and both doorway openings),
    # 1 for the volumetric one. See `UpstreamDensityPowerLaw`.
    per_kind_m: dict[str, list[float]] = {}
    per_kind_dpt: dict[str, list[float]] = {}
    # The NETWORK edge column of every per-edge coefficient appended above, in append order,
    # so the ordering `endpoints(kind)` assumes can be ASSERTED rather than trusted below.
    per_kind_cols: dict[str, list[int]] = {}
    per_kind_quad: dict[str, list[tuple[float, float]]] = {}
    per_kind_fan: dict[str, list[PrjPath]] = {}
    per_kind_spline: dict[str, list[PrjPath]] = {}
    # A damper or constant-flow fan element on several paths is one element of its kind
    # with per-edge coefficients, built where its first path puts it in `built`
    # (kind -> position, one row of coefficients per edge).
    per_kind_slot: dict[str, tuple[int, list[tuple[float, ...]]]] = {}

    def slot(kind: str, row: tuple[float, ...]) -> None:
        if kind not in per_kind_slot:
            per_kind_slot[kind] = (len(built), [])
            built.append(None)
        per_kind_slot[kind][1].append(row)
        per_kind_cols.setdefault(kind, []).append(column - 1)
    path_edges: list[list[int]] = []          # per path, its NETWORK edge columns
    column = 0
    for p in paths:
        if p.nr in ahs_roles:
            ahs, role = ahs_roles[p.nr]
            attrs = dict(role=float(role),
                         design_flow=p.design_flow if role in (SUPPLY, RETURN) else 0.0)
            if role == RECIRCULATION:
                attrs.update(min_outdoor_air=p.design_flow, outdoor_air_fraction=1.0)
            net.add_edge(node_of(p.from_zone), node_of(p.to_zone), kind=f"ahs_{ahs}",
                         z_path=p.z, **attrs)
            path_edges.append([column])
            column += 1
            continue
        el = elements[p.element_nr]
        kind = kind_of[p.element_nr]
        d = el.dtype
        src, tgt = node_of(p.from_zone), node_of(p.to_zone)
        wind_attrs = {}
        if p.profile_nr:
            wind_attrs = {"Ch": p.Ch, "azimuth": p.azimuth, "profile": float(p.profile_nr)}
        cols: list[int] = []
        first = el.data[0]
        if d in _DOOR:
            # `ht wd cd` sit at fields 4, 5, 6 for BOTH doorway types; only field 3 differs
            # (dor_door's dTmin versus dor_pl2's half-separation dH). Verified for dor_door
            # against NIST's own record and its turb = cd A sqrt(2) identity; see the layout
            # block at the top of this module.
            ht, wd, cd = first[4], first[5], first[6]
            dh = (2.0 * ht / 9.0) if d == "dor_door" else first[3]
            area = wd * ht / 2.0
            for off in (-dh, dh):
                net.add_edge(src, tgt, kind=kind, z_path=p.z + off, area=area, Cd=cd,
                             **wind_attrs)
                per_kind_C.setdefault(kind, []).append(
                    p.mult * cd * area * math.sqrt(2.0 * RHO_0)
                )
                per_kind_n.setdefault(kind, []).append(0.5)
                per_kind_m.setdefault(kind, []).append(0.5)
                per_kind_cols.setdefault(kind, []).append(column)
                cols.append(column)
                column += 1
            path_edges.append(cols)
            continue
        net.add_edge(src, tgt, kind=kind, z_path=p.z, **wind_attrs)
        cols.append(column)
        column += 1
        path_edges.append(cols)
        if d in _POWERLAW:
            lam, turb, expt = first[0], first[1], first[2]
            if d in _POWERLAW_SQRT_RHO:
                C, m = turb * math.sqrt(RHO_0), 0.5
            elif d in _POWERLAW_VOLUME:
                C, m = RHO_0 * turb, 1.0
            else:
                C, m = turb, 0.0
            per_kind_C.setdefault(kind, []).append(p.mult * C)
            per_kind_n.setdefault(kind, []).append(expt)
            per_kind_m.setdefault(kind, []).append(m)
            per_kind_cols.setdefault(kind, []).append(column - 1)
            dpt = _laminar_transition(C, lam, expt)
            per_kind_dpt.setdefault(kind, []).append(dpt)
        elif d in _QUADRATIC:
            a, b = first[0], first[1]
            if d == "qfr_qab":
                a, b = a / RHO_0, b / RHO_0**2
            per_kind_quad.setdefault(kind, []).append((a / p.mult, b / p.mult**2))
        elif d in _DAMPER:
            lam, Cp, xp, Cn, xn = first[:5]
            scale = RHO_0 if d == "plr_bdq" else 1.0
            slot(kind, (p.mult * scale * Cp, xp, p.mult * scale * Cn, xn))
        elif d in _FAN_CONST:
            flow = first[0] * (RHO_0 if d == "fan_cvf" else 1.0)
            slot(kind, (p.mult * flow,))
        elif d == "fan_fan":
            per_kind_fan.setdefault(kind, []).append(p)
            per_kind_cols.setdefault(kind, []).append(column - 1)
        elif d in _SPLINE:
            per_kind_spline.setdefault(kind, []).append(p)
            per_kind_cols.setdefault(kind, []).append(column - 1)

    for kind, (at, rows) in per_kind_slot.items():
        _check_edge_order(net, kind, per_kind_cols[kind])
        columns = torch.tensor(rows, dtype=F64).T
        built[at] = (FixedFlow(columns[0], kind=kind) if len(rows[0]) == 1
                     else Damper(*columns, kind=kind))
    # Power-law kinds: one `UpstreamDensityPowerLaw` per kind with per-edge C, n, m and the
    # kind's (shared) transition. The transition is the element's own, so it is the same for
    # every edge of the kind; edges only differ by `mult`.
    #
    # C is still the coefficient at the REFERENCE density RHO_0; the element multiplies it
    # by (rho_upstream / RHO_0)**m at solve time, which is what ContamX does (TN 1887r1
    # section 3.2) and what the isothermal-only reference value gets wrong by about 0.086%
    # per kelvin of zone-to-ambient temperature difference. `dp_transition` deliberately
    # stays at reference conditions -- see the element's docstring.
    for kind, Cs in per_kind_C.items():
        ns = per_kind_n[kind]
        ms = per_kind_m[kind]
        dpt = per_kind_dpt.get(kind, [1e-3])[0]
        # `endpoints(kind)` returns (src, tgt) in `edge_index(kind)` order, which is also the
        # order `PotentialFlowLayer` slices this element's dp block in. The per-edge C/n/m
        # lists above were appended in PATH order, and a doorway contributes TWO edges per
        # path -- so the two orders agree only because every edge of a kind is added to the
        # network at the moment its coefficient is appended. Assert it: a silent mismatch
        # would pair each edge with another edge's endpoints and give the correction the
        # wrong upstream node, with no exception anywhere.
        _check_edge_order(net, kind, per_kind_cols[kind])
        src, tgt = net.endpoints(kind)
        built.append(
            UpstreamDensityPowerLaw(
                torch.tensor(Cs, dtype=F64), torch.tensor(ns, dtype=F64),
                src=src, tgt=tgt, m=torch.tensor(ms, dtype=F64), rho_ref=RHO_0,
                dp_transition=dpt, kind=kind,
            )
        )
    # A fan curve's path schedule is its speed ratio, read from "fan_speed.<kind>".
    speed_kinds = {kind_of[p.element_nr] for p in paths
                   if (p.schedule_nr or p.control_nr) and p.nr not in ahs_roles
                   and elements[p.element_nr].dtype == "fan_fan"}
    for kind, members in per_kind_fan.items():
        # Per-edge mult in path order against the network's edge order, as for power laws.
        _check_edge_order(net, kind, per_kind_cols[kind])
        src, tgt = net.endpoints(kind)
        built.append(fan_curve_element(
            elements[members[0].element_nr].data, kind=kind, src=src, tgt=tgt,
            multiplier=[p.mult for p in members],
            speed_key=f"fan_speed.{kind}" if kind in speed_kinds else None))
    for kind, members in per_kind_spline.items():
        _check_edge_order(net, kind, per_kind_cols[kind])
        el = elements[members[0].element_nr]
        src, tgt = net.endpoints(kind)
        try:
            built.append(spline_element(el.dtype, el.data[1:], kind=kind, src=src, tgt=tgt,
                                        multiplier=[p.mult for p in members]))
        except ValueError as exc:
            raise ValueError(f"prj: flow element {el.nr}: {exc}") from exc
    for kind, ab in per_kind_quad.items():
        built.append(
            Quadratic(
                torch.tensor([x[0] for x in ab], dtype=F64),
                torch.tensor([x[1] for x in ab], dtype=F64),
                kind=kind,
            )
        )
    def filter_of(nr: int, what: str):
        try:
            element = filters[nr]
        except KeyError as exc:
            raise KeyError(f"prj: {what} references filter {nr}, which the "
                           f"'filters' section does not define") from exc
        return _filter_object(element, filter_elements, species)

    edge_filters: dict = {}
    for p, cols in zip(paths, path_edges, strict=True):
        if not p.filter_nr:
            continue
        flt = filter_of(p.filter_nr, f"path {p.nr}")
        for c in cols:
            edge_filters[net.edges[c]] = flt

    # A constant wind pressure (`wPset` on a path with no profile) acts on the path's ambient
    # side: +wPset on an edge from ambient, -wPset on one into it (dp = phi_src - phi_tgt +
    # drive). A positive value pushes air from ambient into the zone.
    wind_pressure: dict[str, list[float]] = {}
    for p, cols in zip(paths, path_edges, strict=True):
        if p.profile_nr == 0 and p.wPset != 0.0:
            if -1 not in (p.from_zone, p.to_zone):
                raise ValueError(f"prj: path {p.nr} has a constant wind pressure {p.wPset} Pa "
                                 f"but is not an envelope path (unsupported)")
            kind = kind_of[p.element_nr]
            kind_cols = net.edge_index(kind).tolist()
            values = wind_pressure.setdefault(kind, [0.0] * len(kind_cols))
            for c in cols:
                values[kind_cols.index(c)] = p.wPset if p.from_zone == -1 else -p.wPset

    drives: list = []
    for el in built:
        if el.dp_independent:
            continue                              # a fixed flow ignores dp; a drive is moot
        drives.append(Stack.from_network(net, el.kind, g=g))
        # Profiles are keyed by CONTAM's own profile NUMBER, and the lookup is
        # deliberately strict -- an edge naming a number no profile carries raises inside
        # `Wind.from_network`, naming the edge and the number. That is why the drive is
        # built whenever an edge names a profile, even when `profiles` is empty: a missing
        # profiles section must be loud, not a silently vanishing wind pressure.
        if bool((net.edge_attr("profile", el.kind, default=0.0) > 0).any()):
            drives.append(Wind.from_network(net, el.kind, ambient=ambient, profiles=profiles))
    for kind in wind_pressure:
        drives.append(ConstantDrive(kind, f"wind_pressure.{kind}"))

    # A path schedule multiplies the path's flow: its kind is wrapped in `Scaled`, which reads
    # one multiplier per edge of the kind from "flow_multiplier.<kind>", and the schedule
    # becomes an input on the path's own edges. Wrapping comes after the drives above, which
    # read only `el.kind` and `el.dp_independent` (both of which `Scaled` keeps), and before
    # the duct build, the kinds and the layout below, which see the wrapped elements.
    # On a fan curve the schedule is the speed ratio instead ("fan_speed.<kind>", read by
    # the `AffinityFan` itself), so that kind is not wrapped.
    multiplier_edges: dict[str, int] = {}
    speed_edges: dict[str, int] = {}
    for p, cols in zip(paths, path_edges, strict=True):
        if not (p.schedule_nr or p.control_nr) or p.nr in ahs_roles:
            continue                  # an air handler's schedules are its closure's inputs
        kind = kind_of[p.element_nr]
        kind_cols = net.edge_index(kind).tolist()
        index = [kind_cols.index(c) for c in cols]
        if elements[p.element_nr].dtype == "fan_fan":
            speed_edges[kind] = len(kind_cols)
            key, what = f"fan_speed.{kind}", "fan speed schedule"
        else:
            multiplier_edges[kind] = len(kind_cols)
            key, what = f"flow_multiplier.{kind}", "schedule"
        if p.schedule_nr:
            inputs.append(Input(week(p.schedule_nr, f"path {p.nr}"), key, index=index,
                                name=f"path {p.nr} {what} {p.schedule_nr}"))
        if p.control_nr:
            actuations.append(Actuation(p.control_nr, key, tuple(index), f"path {p.nr}"))
    built = [Scaled(el, f"flow_multiplier.{el.kind}") if el.kind in multiplier_edges else el
             for el in built]
    wind_pressure_tensors = {k: torch.tensor(v, dtype=F64) for k, v in wind_pressure.items()}

    # Each air handler is one `PrescribedFlow` kind whose `AirHandlerFlows` closure applies
    # the supply and return points' schedules ("flow_multiplier.<kind>", per edge) and the
    # recirculation path's outdoor-air fraction ("outdoor_air_fraction.<kind>"). The kind is
    # not wrapped in `Scaled`, which would apply the multipliers a second time.
    air_handler_edges: dict[str, int] = {}
    ahs_closures: list = []
    flow_balanced: list[str] = []
    storage_free: list[str] = []
    zone_volume = {z[0]: z[3] for z in zones}
    for h in handlers:
        kind = f"ahs_{h.nr}"
        element, closure = air_handler_from_edges(net, kind)
        built.append(element)
        ahs_closures.append(closure)
        kind_cols = net.edge_index(kind).tolist()
        air_handler_edges[kind] = len(kind_cols)
        for z in (h.return_zone, h.supply_zone):
            flow_balanced.append(node_of(z))
            if zone_volume[z] == 0.0:
                if node_of(z) in by_zone:
                    raise ValueError(f"prj: zone {z} ({node_of(z)!r}) is a zero-volume AHS "
                                     f"plenum with a kinetic reaction; it holds no air to "
                                     f"react (unsupported)")
                storage_free.append(node_of(z))
        for p, cols in zip(paths, path_edges, strict=True):
            if ahs_roles.get(p.nr, (None,))[0] != h.nr or not (p.schedule_nr or p.control_nr):
                continue
            if ahs_roles[p.nr][1] == RECIRCULATION:
                key, index = f"outdoor_air_fraction.{kind}", None
                name = f"AHS {h.nr} outdoor-air fraction schedule {p.schedule_nr}"
            else:
                key, index = f"flow_multiplier.{kind}", [kind_cols.index(c) for c in cols]
                name = f"path {p.nr} schedule {p.schedule_nr}"
            if p.schedule_nr:
                inputs.append(Input(week(p.schedule_nr, f"path {p.nr}"), key, index=index,
                                    name=name))
            if p.control_nr:
                actuations.append(Actuation(p.control_nr, key,
                                            None if index is None else tuple(index),
                                            f"path {p.nr}"))

    # The duct network adds its own kinds (already wrapped where scheduled), drives, filters
    # and inputs; the kinds and the layout below are read after it, so they cover the ducts.
    duct = None
    if junctions or segments:
        if dens_zp:
            raise ValueError("prj: densZP = 1 in a project with a duct network is "
                             "unsupported (junction densities at their absolute pressure "
                             "are not modelled)")
        if run_control.zone_mass_change:
            raise ValueError("prj: dodMdt = 1 in a project with a duct network is "
                             "unsupported (the junctions' air mass under it is not modelled)")
        ctx = BuildContext(net=net, ambient=ambient, zone_name=node_of, week=week,
                           schedule_units=schedule_units, filter_of=filter_of, g=g,
                           profiles=profiles, inputs=inputs, edge_filters=edge_filters,
                           built=built, drives=drives,
                           constant_wind_pressure=wind_pressure_tensors,
                           actuations=actuations)
        duct = build_ducts(ctx, duct_elements, junctions, segments)
        multiplier_edges.update(duct.multiplier_edges)
        speed_edges.update(duct.speed_edges)

    # The control nodes the actuators and the named report nodes need; each schedule node
    # is an input on the network's input, at its slot.
    plan = plan_controls(control_nodes, actuations)
    if plan is not None:
        for nr, slot in plan.schedule_slots.items():
            node = control_nodes[nr]
            ps = int(node.data[0])
            inputs.append(Input(week(ps, node.label), INPUT_KEY, index=slot,
                                name=f"{node.label} schedule {ps}"))

    occupants = _occupants(exposures or [], occupancy or {}, zone_nr_to_name, species, week)
    inputs.extend(occupant_inputs(occupants, net))

    kinds = [el.kind for el in built]

    # `PotentialFlowLayer` concatenates `q` in element-KIND blocks, in
    # `built` order -- NOT in path order, which is the network's own edge order. The two
    # coincide only for a project with a single element kind, so translate here, once.
    layout: dict[int, int] = {}
    offset = 0
    for el in built:
        for i, col in enumerate(net.edge_index(el.kind).tolist()):
            layout[col] = offset + i
        offset += net.edge_index(el.kind).shape[0]
    for p, cols in zip(paths, path_edges, strict=True):
        p.edge_columns = [layout[c] for c in cols]
    terminals = [PrjTerminal(j_nr, z_nr, [layout[col]])
                 for j_nr, z_nr, col in (duct.terminal_columns if duct else [])]
    duct_segments = [PrjSegment(nr, [layout[col]])
                     for nr, col in (duct.segment_columns if duct else [])]

    for section, rows, known in (
            ("initial zone concentrations", x0_rows, {z[0] for z in zones}),
            ("initial junction concentrations", junction_x0_rows, {j.nr for j in junctions})):
        unknown = sorted(set(rows) - known)
        if unknown:
            what = "zone" if section.endswith("zone concentrations") else "junction"
            raise KeyError(f"prj: the {section!r} section has rows for {what}(s) {unknown}, "
                           f"which the file never defines")
    K = max(1, len(species))
    x0 = torch.zeros(len(zone_names), K, dtype=F64)
    for i, (nr, *_rest) in enumerate(zones):
        if nr in x0_rows:
            row = x0_rows[nr]
            x0[i, : len(row)] = torch.tensor(row, dtype=F64)
    junction_x0 = torch.zeros(len(junctions), K, dtype=F64)
    for i, j in enumerate(junctions):
        if j.nr in junction_x0_rows:
            row = junction_x0_rows[j.nr]
            junction_x0[i, : len(row)] = torch.tensor(row, dtype=F64)

    return Project(
        net=net, elements=built, drives=drives, zones=zone_names, ambient=ambient, paths=paths,
        species=species, T_zone=torch.tensor(T_list, dtype=F64),
        zone_volumes=torch.tensor(V_list, dtype=F64), g=g,
        ambient_conditions=ambient_conditions, profiles=profiles, x0=x0, sources=sources,
        kinds=kinds, zone_nr_to_name=zone_nr_to_name, density_uses_zone_pressure=dens_zp,
        run_control=run_control, schedules=schedules, inputs=inputs,
        ambient_mass_fractions=torch.tensor(
            [s.default_mass_fraction for s in species_records], dtype=F64),
        multiplier_edges=multiplier_edges, speed_edges=speed_edges,
        constant_wind_pressure=wind_pressure_tensors,
        zone_flags=zone_flags, source_path=source_path, species_records=species_records,
        zone_reactions=by_zone, edge_filters=edge_filters,
        junctions=duct.junction_names if duct else [], junction_x0=junction_x0,
        terminals=terminals, segments=duct_segments, schedule_units=schedule_units,
        terminal_openings=duct.terminal_openings if duct else 0,
        flow_balanced=flow_balanced, storage_free=storage_free, closures=ahs_closures,
        air_handler_edges=air_handler_edges,
        junction_nr_to_name=duct.junction_nr_to_name if duct else {},
        control_nodes=control_nodes, control_plan=plan, occupants=occupants,
    )


def _occupants(exposures: list[PrjExposure], occupancy: dict[int, list[tuple[float, int]]],
               zone_nr_to_name: dict[int, str], species: list[str], week) -> list[Occupant]:
    """Each exposure record as an `Occupant` named "occupant <nr>": its occupancy schedules
    (one per day type, zone numbers resolved to zone names, 0 out of the building) and,
    when it generates, its rates (kg/s per person) and week schedules. Refused by name: an
    occupancy schedule the file does not define, a zone it never defines, a schedule not
    running from 00:00:00 to 24:00:00, and a generated species the project does not
    simulate."""
    occupants: list[Occupant] = []
    for e in exposures:
        missing = sorted({j for j in e.day_schedules if j not in occupancy})
        if missing:
            raise KeyError(f"prj: exposure {e.nr} names occupancy schedule(s) {missing}, "
                           f"which the 'occupancy schedules' section does not define")
        order = sorted(set(e.day_schedules))
        days = []
        for j in order:
            zones: list[str | None] = []
            for _t, z in occupancy[j]:
                if z == 0:
                    zones.append(None)
                elif z in zone_nr_to_name:
                    zones.append(zone_nr_to_name[z])
                else:
                    raise KeyError(f"prj: occupancy schedule {j} places an occupant in zone "
                                   f"{z}, which the file never defines")
            try:
                days.append(OccupancyDay([t for t, _z in occupancy[j]], zones,
                                         name=f"occupancy schedule {j}"))
            except ValueError as exc:
                raise ValueError(f"prj: occupancy schedule {j}: {exc}") from exc
        generation, schedules = {}, {}
        if e.generates:
            for sname, ps, rate in e.generation:
                if sname not in species:
                    raise KeyError(f"prj: exposure {e.nr} generates species {sname!r}, which "
                                   f"the project does not simulate")
                generation[sname] = rate
                if ps:
                    schedules[sname] = week(ps, f"exposure {e.nr}")
        name = f"occupant {e.nr}"
        occupants.append(Occupant(
            Occupancy(days, [order.index(j) + 1 for j in e.day_schedules], name=name),
            generation=generation, multiplier=e.multiplier, schedules=schedules or None,
            name=name, description=e.description))
    return occupants


class PrescribedTemperatureDensity:
    """Air density at every node from prescribed node temperatures, the CONTAM route's
    density closure: rho = P_ref / (R T) (ideal gas, TN 1887r1 section 3.18).

    Reads `"T"` (K, full node order: every zone's temperature and the ambient's) and
    `"P_ref"` (Pa, the barometric pressure; scalar or batched). Writes `"rho"` (full node
    order), `"rho_amb"` and, when the model has a species layer, `"<layer>.capacity"`: each
    zone's air mass rho V (kg) on that layer's interior, as ContamX holds it. A scheduled zone
    temperature therefore changes the zone's air mass, and the species step's amount form
    (`TransportLayer.step(capacity_prev=...)`) conserves the contaminant across that change.
    """

    inputs = ("T", "P_ref")

    def __init__(self, *, ambient_index: int, volumes: Tensor, capacity_nodes: Tensor | None,
                 R: float = R_AIR, layer: str = "species") -> None:
        self.ambient_index = int(ambient_index)
        self.volumes = torch.as_tensor(volumes, dtype=F64)
        self.capacity_nodes = capacity_nodes
        self.R = float(R)
        self.capacity_key = f"{layer}.capacity"
        self.outputs = ("rho", "rho_amb") + (
            (self.capacity_key,) if capacity_nodes is not None else ())

    @property
    def input_specs(self) -> dict[str, dict]:
        """The node temperatures and barometric pressure this closure reads."""
        return {
            "T": dict(description="air temperature at every node (zones and ambient)",
                      unit="K", over="nodes"),
            "P_ref": dict(description="barometric pressure (not corrected to sea level)",
                          unit="Pa"),
        }

    def __call__(self, state, drivers):
        T = drivers["T"]
        P = torch.as_tensor(drivers["P_ref"], dtype=T.dtype)
        rho = P.unsqueeze(-1) / (self.R * T)
        out = {"rho": rho, "rho_amb": rho[..., self.ambient_index]}
        if self.capacity_nodes is not None:
            out[self.capacity_key] = (rho * self.volumes)[..., self.capacity_nodes]
        return out


class ZonePressureDensity:
    """ContamX's `densZP = 1`: zone densities at the zone's absolute pressure.

    Writes `rho = rho_Pb * (1 + phi / P_ref)` over the full-node `drivers["rho"]`, which
    `PrescribedTemperatureDensity` evaluates at the barometric pressure alone
    (`P_ref / (R T)`), so the result is `(P_ref + phi) / (R T)`. `phi` is the air layer's
    gauge pressure at each node's reference height, read from `state["air.phi"]` -- the
    PREVIOUS solve's, because a closure runs before the solve it feeds. The ambient node is
    the zero-gauge boundary, so `rho_amb` is unchanged. With no `"air.phi"` in the state (the
    initial state `project_to_model` returns) the correction is zero. `P_ref` is read from the
    drivers when present (a weather file's pressure is then used), else `Pb` stands.

    The zone air mass (the species layer's capacity, when `capacity_key` is given) carries the
    same `(1 + phi / P_ref)` factor, as ContamX's zone mass does under `densZP`.
    `densities=False` applies the factor to that zone air mass ONLY and leaves `rho` at the
    barometric pressure: ContamX's zone mass under `dodMdt = 1` with `densZP = 0` (measured;
    `project_to_model` adds it for such a project).

    Consequently a single `model.steady` is one fixed-point pass; `steady` below repeats it
    to convergence. A transient `model.step` uses the pressures of the step before, a lag of
    one step on a term of relative size |P_zone| / Pb.
    """

    def __init__(self, Pb: float, *, layer: str = "air", capacity_key: str | None = None,
                 capacity_nodes: Tensor | None = None, densities: bool = True) -> None:
        self.Pb = float(Pb)
        self.densities = bool(densities)
        self.key = f"{layer}.phi"
        self.capacity_key = capacity_key
        self.capacity_nodes = capacity_nodes

    def __call__(self, state, drivers):
        phi = state.get(self.key)
        if phi is None:
            return {}
        P = torch.as_tensor(drivers.get("P_ref", self.Pb), dtype=phi.dtype)
        factor = 1.0 + phi / P.unsqueeze(-1)
        out = {"rho": drivers["rho"] * factor} if self.densities else {}
        if self.capacity_key is not None:
            if self.capacity_key not in drivers:
                raise KeyError(
                    f"ZonePressureDensity: driver {self.capacity_key!r} (the zone air mass "
                    f"this closure rescales to the zone's absolute pressure) is missing; a "
                    f"density closure that writes it must run first"
                )
            out[self.capacity_key] = drivers[self.capacity_key] * factor[..., self.capacity_nodes]
        return out


class ZoneAirMassChange(NodeSource):
    """The zone air-mass change of a step as a withdrawal from the airflow network (CONTAM's
    dodMdt option; `simulate` switches it on for a model that carries it, unless
    `air_mass_change=False`):

        w = ((P_ref + phi) V / (R T) - M_start) / dt,

    the air mass at the zone's ABSOLUTE pressure at the step's end, minus the mass at its
    start, over the step. ContamX takes the mass at the absolute pressure whether or not the
    densities the flows see include the zone pressure (measured: a zone depressurised by an
    exhaust switching on loses M dP_zone / (Pb dt) in that step). The withdrawal depends on
    the zone's own pressure with slope V / (R T dt), which the airflow solve's Jacobian
    carries, so the step converges at any length (the slope outgrows the envelope's
    conductance for short steps).

    Reads `"T"` and `"P_ref"`, `"time_step"` (s) and `"<layer>.mass_start"` (kg, at this
    source's nodes, in their order: `mass` at the step's start). Without
    `"<layer>.mass_start"` it withdraws nothing and adds nothing to the Jacobian: the steady
    and quasi-steady solves are unchanged. The withdrawal is a difference of masses of order
    M, so the solve cannot resolve it below about 1e-16 M / dt (kg/s).
    """

    def __init__(self, nodes, volumes: Tensor, *, layer: str = "air", R: float = R_AIR) -> None:
        super().__init__(nodes)
        self.register_buffer("V", torch.as_tensor(volumes, dtype=F64))
        self.R = float(R)
        self.mass_key = f"{layer}.mass_start"

    def mass(self, phi_nodes: Tensor, drivers) -> Tensor:
        """(P_ref + phi) V / (R T) at this source's nodes (kg)."""
        P = torch.as_tensor(drivers["P_ref"], dtype=phi_nodes.dtype).unsqueeze(-1)
        return (P + phi_nodes) * self.V / (self.R * drivers["T"][..., self.nodes])

    def flow(self, phi_nodes: Tensor, drivers=None) -> Tensor:
        if drivers is None or self.mass_key not in drivers:
            return phi_nodes * 0.0
        return (self.mass(phi_nodes, drivers) - drivers[self.mass_key]) / drivers["time_step"]

    def dflow(self, phi_nodes: Tensor, drivers=None) -> Tensor:
        if drivers is None or self.mass_key not in drivers:
            return torch.zeros_like(phi_nodes)
        slope = self.V / (self.R * drivers["T"][..., self.nodes] * drivers["time_step"])
        return slope.expand(torch.broadcast_shapes(slope.shape, phi_nodes.shape))


def _initial_mass_fractions(project: Project) -> Tensor:
    """The species state's initial rows in the species layer's node order (the network's:
    the zones, then the junctions). Every zone and junction must be an active node of that
    layer; one that no flow edge reaches is refused by name rather than dropped."""
    net = project.net
    interior, _ = active_interior(net, tuple(project.kinds), [project.ambient])
    rows = {net.node_index(z): project.x0[i] for i, z in enumerate(project.zones)}
    for i, j in enumerate(project.junctions):
        rows[net.node_index(j)] = project.junction_x0[i]
    active = set(interior.tolist())
    inactive = [net.nodes[n] for n in sorted(rows) if n not in active]
    if inactive:
        raise ValueError(f"prj: {inactive} carry no airflow path, duct segment or terminal, "
                         f"so they have no species state (unsupported)")
    return torch.stack([rows[n] for n in interior.tolist()])


def project_to_model(project: Project, *, ambient: dict | None = None, species: bool = True,
                     scheme: str = "implicit"):
    """Air layer + species layer (no thermal layer: a .prj carries no thermal data), the
    initial state and the drivers for the project's ambient conditions (or `ambient`).

    Node densities are computed every step from `"T"` and `"P_ref"` by
    `PrescribedTemperatureDensity` (plus the zone-pressure term of `ZonePressureDensity` when
    the file sets `densZP`), and the species layer's capacity is each zone's air mass at THAT
    density, `rho_zone V` -- as ContamX holds it -- not at the reference RHO_0: 1.2041 is
    `Pb / (R T)` only at 293.15 K and 101325 Pa to five figures (2.2e-6 off even there, which
    alone put 6.5e-6 into the three-zone transient concentrations against ContamX), and about
    3.4 % off per 10 K away from it. The returned drivers carry the initial `"rho"` and
    `"rho_amb"` for code that reads them outside a step; every step recomputes them.

    The drivers are at the run control's STEADY-SIMULATION instant (its day, time and day
    type): the project's schedules (`project.inputs`) are applied there, so a `steady()` on
    them is CONTAM's steady simulation. Each species' boundary value is its default ambient
    mass fraction; a scheduled kind gets `"flow_multiplier.<kind>"` (ones before the
    schedules apply) and a constant wind pressure `"wind_pressure.<kind>"`.

    A project with a control plan gets a new sampled `ControlNetwork` (`prj_controls`),
    first among the closures, its sensors read from this model's layers; it writes nothing
    until `simulate` evaluates it.
    """
    amb = dict(project.ambient_conditions)
    if ambient:
        amb.update({k: v for k, v in ambient.items() if k in ("Ta", "Pb", "Ws", "Wd")})
    net = project.net
    T = net.node_attr("T0")
    T[net.node_index(project.ambient)] = amb["Ta"]
    rho = amb["Pb"] / (R_AIR * T)
    volumes = net.node_attr("volume", default=0.0)
    check_flow_balanced(net, project.elements, project.flow_balanced, where="prj")
    # The zones only: a duct junction's air-mass change is not part of the option, and an
    # air handler's plenum is not a node of the airflow solve.
    zones = set(project.zones) - set(project.flow_balanced)
    zone_nodes = [i for i, name in enumerate(net.nodes)
                  if name in zones and volumes[i].item() > 0.0]
    zone_mass_change = project.run_control is not None and project.run_control.zone_mass_change
    # The zone air-mass change source exists only when the project asks for it and there is a
    # zone to carry it (an empty node set has no air mass to change).
    node_sources = (
        [ZoneAirMassChange(zone_nodes, volumes[zone_nodes])]
        if zone_mass_change and zone_nodes else []
    )
    air = PotentialFlowLayer(net, "air", project.elements, drives=project.drives,
                             boundary=[project.ambient, *project.flow_balanced],
                             quantity="pressure", unit="Pa", flow_unit="kg/s",
                             node_sources=node_sources)
    layers: dict = {"air": air}
    no_contaminants = project.run_control is not None and project.run_control.contaminants == "none"
    if species and project.species and not no_contaminants:
        kinetics = None
        if project.zone_reactions:
            kinetics = reaction_kinetics(net, species=project.species,
                                         reactions=project.zone_reactions,
                                         ambient=project.ambient, flow_kinds=project.kinds)
        transmission = None
        if project.edge_filters:
            transmission = filter_transmission(net, project.kinds, project.species_records,
                                               project.edge_filters)
        x_initial = _initial_mass_fractions(project)
        # A duct junction without volume (its segments all of zero length, a fan's say) and a
        # zero-volume air-handler plenum hold no air: storage-free nodes, whose mass fraction
        # is the mix of what flows in.
        storage_free = [j for j in project.junctions
                        if volumes[net.node_index(j)].item() == 0.0] + project.storage_free
        layers["species"] = species_layer(
            net, ambient=project.ambient, flow_kinds=tuple(project.kinds),
            n_species=len(project.species), scheme=scheme, rho=rho,
            species_names=project.species, kinetics=kinetics, transmission=transmission,
            storage_free=storage_free,
        )
    cap_nodes = layers["species"].interior_idx if "species" in layers else None
    density = PrescribedTemperatureDensity(
        ambient_index=net.node_index(project.ambient), volumes=volumes,
        capacity_nodes=cap_nodes)
    closures: list = [density]
    if project.density_uses_zone_pressure or (zone_mass_change and cap_nodes is not None):
        # With `dodMdt` alone ContamX holds the contaminants' zone air MASS at the zone's
        # absolute pressure but the densities the flows see at Pb.
        closures.append(ZonePressureDensity(
            amb["Pb"], capacity_key="species.capacity" if cap_nodes is not None else None,
            capacity_nodes=cap_nodes, densities=project.density_uses_zone_pressure))
    if "species" in layers and project.sources:
        from noodl.apps.building_physics.sources import SpeciesSources, sources_from_project
        dt = project.run_control.time_step if project.run_control else 60.0
        closures.append(SpeciesSources(
            sources_from_project(project, dt=dt), species=[s.species for s in project.sources],
            layer=layers["species"], names=[f"source {s.nr}" for s in project.sources],
            heights=net.node_attr("z_ref"), g=project.g))
    if "species" in layers and any(o.generation for o in project.occupants):
        # Added to the species sources like `SpeciesSources`; the presence and schedule
        # multipliers it reads are among the project's inputs.
        closures.append(OccupantGeneration(project.occupants, layer=layers["species"]))
    closures.extend(project.closures)
    drivers = {
        "air.phi_boundary": torch.zeros(1 + len(project.flow_balanced), dtype=F64),
        "T": T.clone(),
        "P_ref": torch.tensor(amb["Pb"], dtype=F64),
        # Initial values, for code that reads them outside a step (`noodl.couple`'s
        # mass-fraction conversions); every step recomputes them from "T" and "P_ref".
        "rho": rho,
        "rho_amb": rho[net.node_index(project.ambient)],
        "V_met": torch.tensor(amb["Ws"], dtype=F64),
        "theta_w": torch.tensor(amb["Wd"], dtype=F64),
        "time": torch.tensor(0.0, dtype=F64),
    }
    if "species" in layers and project.sources:
        drivers["species.source_multiplier"] = torch.ones(len(project.sources), dtype=F64)
    if project.control_plan is not None:
        # A new network for every model (one network serves one model), first among the
        # closures: what it writes ("T", multipliers) is what the others read. A model without
        # a species layer has no source multipliers to write, as for the source schedules.
        plan = project.control_plan
        actuations = [a for a in plan.actuations
                      if "species" in layers or not a.key.startswith("species.")]
        names = tuple(drivers)
        closures.insert(0, control_network(
            plan, project.control_nodes, actuations=actuations,
            reading=lambda node: _sensor_reading(project, layers, names, node)))
    for closure in closures:
        if isinstance(closure, ControlNetwork):
            closure.resolve(net, layers, tuple(drivers))
    model = Model(net, layers, closures=closures)
    state: dict = {}
    if "species" in layers:
        K = len(project.species)
        if project.ambient_mass_fractions is not None:
            drivers["species.x_boundary"] = (
                project.ambient_mass_fractions.reshape(1, -1).clone())
        else:
            drivers["species.x_boundary"] = torch.zeros(1, K, dtype=F64)
        state["species.x"] = x_initial
    for closure in closures:
        if isinstance(closure, ControlNetwork):
            # Not evaluated yet: it writes nothing until `simulate` evaluates it, so the
            # capacities below are at the drivers as the schedules leave them.
            state.update(closure.initial_state())
            drivers.update(closure.initial_drivers())
    # The inputs the airflow elements and drives read themselves, declared for
    # `model.refs.inputs` and `Model.check`.
    model.input_specs = {
        "V_met": dict(description="wind speed at the meteorological station",
                      unit="m/s"),
        "theta_w": dict(description="wind direction, as CONTAM's weather file gives it",
                        unit="deg"),
    }
    for kind, n_e in project.multiplier_edges.items():
        drivers[f"flow_multiplier.{kind}"] = torch.ones(n_e, dtype=F64)
        model.input_specs[f"flow_multiplier.{kind}"] = dict(
            description="flow multiplier of each edge (a schedule or control)", unit="-",
            over=("edges", kind))
    for kind, n_e in project.air_handler_edges.items():
        # Declared by the air handler's closure.
        drivers[f"flow_multiplier.{kind}"] = torch.ones(n_e, dtype=F64)
        drivers[f"outdoor_air_fraction.{kind}"] = torch.tensor(1.0, dtype=F64)
    for kind, n_e in project.speed_edges.items():
        drivers[f"fan_speed.{kind}"] = torch.ones(n_e, dtype=F64)
        model.input_specs[f"fan_speed.{kind}"] = dict(
            description="speed of each fan relative to its rated speed (a schedule or "
                        "control); below the fan's off ratio it is stopped",
            unit="-", over=("edges", kind))
    if project.terminal_openings:
        key = "terminal_opening.duct_terminal"
        drivers[key] = torch.ones(project.terminal_openings, dtype=F64)
        model.input_specs[key] = dict(
            description="opening of each duct terminal (1 open, 0 closed: its loss "
                        "coefficient rises by (1 - S) / S)", unit="-",
            over=("edges", "duct_terminal"))
    for kind, value in project.constant_wind_pressure.items():
        drivers[f"wind_pressure.{kind}"] = value.clone()
        model.input_specs[f"wind_pressure.{kind}"] = dict(
            description="constant wind pressure on each edge, on its ambient side", unit="Pa",
            over=("edges", kind))
    if project.inputs and project.run_control is not None:
        drivers = apply_inputs(drivers, bind_inputs(model, project.inputs_for(model)), 0.0,
                               project.run_control.steady_calendar())
        rho_now = density(state, drivers)
        drivers["rho"], drivers["rho_amb"] = rho_now["rho"], rho_now["rho_amb"]
    state.update(model.initial_capacities(state, drivers))
    return model, state, drivers


_SENSED_EDGES = {2: "path", 4: "duct segment", 6: "terminal"}


def _sensor_reading(project: Project, layers: dict, drivers: tuple[str, ...],
                    node: PrjControlNode):
    """A sensor node's reading (fields `offset scale tau oldsig source type measure X Y relHt
    units species`). Type 1 zone, 3 junction: measure 0 mass fraction (kg/kg), 1 temperature
    (K), 4 gauge pressure (Pa); type 2 path, 4 duct segment, 6 terminal: measure 2 flow
    (kg/s), 3 pressure difference (Pa). Source -1 is the ambient."""
    d = node.data
    source, kind, measure, species = int(d[4]), int(d[5]), int(d[6]), d[11]
    where = f"prj: {node.label}"
    if kind in (1, 3):
        what = "zone" if kind == 1 else "junction"
        if source == -1:
            name = project.ambient
        elif kind == 1:
            name = project.zone_nr_to_name.get(source)
        else:
            name = project.junction_nr_to_name.get(source)
        if name is None:
            raise ValueError(f"{where} senses {what} {source}, which the project does not "
                             f"have")
        idx = project.net.node_index(name)
        if measure == 0:
            if "species" not in layers or species not in project.species:
                raise ValueError(f"{where} senses species {species!r}, which the model does "
                                 f"not carry")
            k = project.species.index(species)
            if name == project.ambient:
                return DriverReading("species.x_boundary", (0, k))
            reading = node_mass_fraction(name, species)
            try:
                reading.resolve(project.net, layers, drivers)
            except (KeyError, IndexError) as exc:
                raise ValueError(f"{where} senses the mass fraction at {what} {source}: "
                                 f"{exc.args[0]}") from exc
            return reading.resolved
        if measure == 1:
            return DriverReading("T", (idx,))
        if measure == 4:
            return StateReading("air.phi", (idx,))
        detail = "occupancy (it needs occupants)" if measure == 5 else f"measure {measure}"
        raise ValueError(f"{where}: a {what} sensor of {detail} is not supported")
    if kind in _SENSED_EDGES:
        what = _SENSED_EDGES[kind]
        try:
            column, sign = project.sensed_edge(kind, source)
        except ValueError as exc:
            raise ValueError(f"{where} senses {what} {source}: {exc}") from exc
        key = {2: "air.q", 3: PRESSURE_DIFFERENCE}.get(measure)
        if key is None:
            raise ValueError(f"{where}: a {what} sensor of measure {measure} is not supported")
        return StateReading(key, (column,), factor=sign)
    what = "an exposure" if kind == 5 else "unknown"
    raise ValueError(f"{where}: sensor type {kind} ({what}) is not supported")


def _pressure_coupling(model: Model, where: str) -> tuple[str | None, str]:
    """The potential a closure needs at the solve's own value, and the couplings that need
    it: `ZonePressureDensity` (its `key`) and every closure with `reads_solved_pressure`
    (its `pressure_key`, default "air.phi"). (None, "") when there is none. Closures that
    read different potentials are refused by name: one fixed point feeds back one key."""
    keys: dict[str, list[str]] = {}
    for c in model.closures:
        if isinstance(c, ZonePressureDensity):
            what = "the zone densities" if c.densities else "the contaminants' zone air mass"
            keys.setdefault(c.key, []).append(what)
        elif getattr(c, "reads_solved_pressure", False):
            key = getattr(c, "pressure_key", "air.phi")
            keys.setdefault(key, []).append(f"the pressure reader {type(c).__name__}")
    if not keys:
        return None, ""
    if len(keys) > 1:
        raise ValueError(
            f"{where}: the closures that need the solved pressure read different potentials: "
            + "; ".join(f"{', '.join(v)} read {k!r}" for k, v in keys.items())
            + "; one fixed point feeds back one potential"
        )
    ((key, couplings),) = keys.items()
    return key, " and ".join(couplings)


def _refuse_idle_air_handlers(model: Model, state, drivers) -> None:
    """A steady contaminant solve has no value for an air-handler plenum that nothing passes
    through (a transient run holds it at its last value): refuse it, naming the handler."""
    for closure in model.closures:
        if not isinstance(closure, AirHandlerFlows):
            continue
        flows = closure(state, drivers)[closure.flow_key]
        for role, what in ((SUPPLY, "supply"), (RETURN, "return")):
            total = (flows * (closure.roles == role).to(flows.dtype)).sum(-1)
            if bool((total.detach() == 0).any()):
                raise ValueError(
                    f"prj.steady: air handler {closure.name!r}: nothing passes through its "
                    f"{what} plenum at the steady instant, so a steady contaminant solve "
                    f"has no value for that plenum (unsupported)")


def steady(model: Model, state, drivers, *, rtol: float = 1e-12, max_iter: int = 10,
           **solve_kwargs):
    """`model.steady`, repeated to a fixed point when the model carries a
    `ZonePressureDensity` closure (a `.prj` with `densZP = 1`, or with `dodMdt = 1`, whose
    zone air mass alone carries the zone pressure) or a closure that reads the solved
    pressure (`reads_solved_pressure`, e.g. a pressure-driven source, on the potential its
    `pressure_key` names; closures reading different potentials are refused); one call
    otherwise.
    Such a reader does not feed back into the airflow, so without `ZonePressureDensity` one
    extra pass suffices.

    Each pass feeds the previous pass's `"air.phi"` back to the closure. The loop stops when
    no node pressure moves by more than `rtol * Pb` -- a relative change in density of
    `rtol` -- and raises if that has not happened within `max_iter` passes. The map
    contracts by about `g dz rho / Pb` per pass (~1e-4 for a storey-high opening), so two or
    three passes reach float64 precision. The passes are left on the autograd graph: the
    returned state is differentiable, through the unrolled iteration.
    """
    if max_iter < 1:
        raise ValueError(f"prj.steady: max_iter must be at least 1, got {max_iter!r}")
    if "species" in model.transport:
        _refuse_idle_air_handlers(model, state, drivers)
    zp = [c for c in model.closures if isinstance(c, ZonePressureDensity)]
    key, cause = _pressure_coupling(model, "prj.steady")
    out = model.steady(state, drivers, **solve_kwargs)
    if key is None:
        return out
    Pb = zp[0].Pb if zp else 101325.0
    for _ in range(max_iter):
        new = model.steady({**state, key: out[key]}, drivers, **solve_kwargs)
        change = (new[key] - out[key]).abs().max().item()
        out = new
        if change <= rtol * Pb:
            return out
    raise RuntimeError(
        f"prj.steady: the zone pressures fed back to {cause} did not converge in "
        f"{max_iter} passes "
        f"(last pressure change {change:.3e} Pa > rtol * Pb = {rtol * Pb:.3e} Pa)"
    )

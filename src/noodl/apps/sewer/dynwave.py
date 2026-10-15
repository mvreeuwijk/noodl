"""SWMM 5.2.4 dynamic-wave routing for the sewer application.

`DynamicWaveOptions` holds the `[OPTIONS]` that change the physics. The model is a
`PotentialFlowLayer` ``"water"`` (quantity ``"head"``, m) whose unknowns are the junction
heads: every conduit is a `DynamicWaveConduit`, q = c0 + k (H1 - H2) with the coefficients of
SWMM's momentum update (`dynwave_terms.conduit_terms`) frozen for one pass, and every junction
carries a `JunctionStorage`, SWMM's trapezoidal continuity
``A_s (y - y_start) = 0.5 dt (N_start + N)``. Orifices, weirs and outlets are edges of their
own kinds, `StructureLink`, their flow SWMM's (`structures`) linearised about the previous
pass. `DynamicWaveHydraulics` (a closure) refreshes
the coefficients, surface areas and outfall heads from the previous pass, and
``Model(coupling="iterate")`` repeats the pass until the heads stop changing: the fixed point
is SWMM's own converged trial.

This module must not import `network.py` at module level: `network.py` imports
`DynamicWaveOptions` and `build_dynamic_wave_model` from here.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from noodl._broadcast import broadcast_shapes
from noodl.apps.sewer import geometry as geom
from noodl.apps.sewer.air import (
    F_AIR_DEFAULT,
    F_I_DEFAULT,
    add_air_edges,
    air_layer,
    check_air_topology,
)
from noodl.apps.sewer.controls import start_serial
from noodl.apps.sewer.dynwave_terms import (
    DAMPING,
    FUDGE,
    NORMAL_FLOW_LIMITED,
    SURCHARGE_METHODS,
    ConduitGeometry,
    ConduitTerms,
    LossInputs,
    conduit_terms,
    lengthening_factor,
)
from noodl.apps.sewer.hydraulics import CAPACITY_FLOOR, air_density_nodes
from noodl.apps.sewer.linkcontrol import LinkControl
from noodl.apps.sewer.losses import (
    EVAP_DRIVER,
    HYDCON_DRIVER,
    STORAGE_DRIVER,
    ClimateClock,
    StorageLosses,
    seepage_depth_limit,
)
from noodl.apps.sewer.pumps import (
    IDEAL_PUMP,
    TYPE3_PUMP,
    TYPE5_PUMP,
    PumpSet,
    PumpStation,
    mod_pump_flow,
    pump_curve_flow,
)
from noodl.apps.sewer.quality import H2STransfer, LateralLoads, SulfideGeneration
from noodl.apps.sewer.runoff import RunoffModel, evaporation_can_be_positive
from noodl.apps.sewer.storage import StorageCurves
from noodl.apps.sewer.structures import (
    CIRCULAR,
    DN_CRITICAL,
    RECT_CLOSED,
    RECT_OPEN,
    TINY,
    TRAPEZOIDAL,
    TRIANGULAR,
    UP_CRITICAL,
    Opening,
    OrificeSet,
    OutletSet,
    StructureFlow,
    WeirSet,
    orifice_flow,
    outlet_dqdh,
    outlet_flow,
    weir_flow,
)
from noodl.apps.sewer.xsect_tables import (
    LCF,
    QCF,
    critical_depth,
    normal_depth_ft,
)
from noodl.elements.base import ConstitutiveLaw
from noodl.layers.potential import PotentialFlowLayer
from noodl.layers.reaction import Reaction
from noodl.layers.transport import TransportLayer, active_interior
from noodl.model import Model, StepContext
from noodl.nodesources import NodeSource
from noodl.topology import Network

if TYPE_CHECKING:
    from noodl.apps.sewer.network import SewerNetwork

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

#: The carried keys of the link settings (regulators and pumps; `LinkControl`): each link's
#: setting (SWMM's ``Link.setting``: a pump's speed, a regulator's fraction open), its target
#: setting, the DateTime (days) it was last opened or closed and the end areas a closed orifice
#: keeps giving its junctions (m2, SI area; first ends, then second ends).
CONTROL_KEYS = ("sewer.setting", "sewer.target_setting", "sewer.time_last_set",
                "sewer.orifice_hold")

#: SWMM's dry-node flow cap (`dwflow.c:273-276`): 1e-4 cfs, in m3/s.
FUDGE_FLOW = 1.0e-4 * QCF
#: Bed stiffening factor of `JunctionStorage` (docs: "Bed stiffening and lagged pieces"): below
#: its invert a junction's storage slope gains ``(2 A_bed/dt) K_BED``, standing in for SWMM's
#: ``yNew < 0 -> 0``.
K_BED = 1.0e6
#: Stiffness factor of an EXTRAN junction's hold band at its crown (the tie rule in
#: `JunctionStorage`; docs: "Surcharge: SLOT and EXTRAN"): the band rises with ``(2 A_bed/dt)
#: K_HOLD``. It trades the band's width ``|w_c| dt / (2 A_bed K_HOLD)`` (0.6 mm at w_c = 0.26 m3/s,
#: SWMM's default surface area floor and a 5 s step) against the head solve's round-off floor on it,
#: ``(2 A_bed/dt) K_HOLD eps |H| / 2`` (3e-13 m3/s at 12 m), which must stay below the 1e-12 m3/s a
#: surcharged run's parity needs; the bed's K_BED would put that floor at 6e-10 m3/s.
K_HOLD = 1.0e3
#: The band is also at least `K_HOLD_STORAGE` times the junction's own storage slope
#: ``2 A_s/dt``, so that above the crown it always rises faster than the storage line it
#: starts from (a large A_s, e.g. a trunk junction or a large user MIN_SURFAREA, would
#: otherwise outgrow ``K_HOLD A_bed``). Its round-off floor then scales with A_s: 6e-12
#: m3/s at A_s = 1200 m2, 5 s and 12 m, above a 1e-12 m3/s head-solve atol there.
K_HOLD_STORAGE = 10.0
#: A surcharged junction's anchor slope floor (EXTRAN surcharge): ``K_ANCHOR (2 A_s/dt)`` (m2/s) is
#: added to the conduit conveyance its pass lags (`JunctionStorage`), so the junction's row
#: of the head solve stays nonsingular even where every attached conduit's dependence on its
#: head is lagged or limited. The anchor vanishes at the fixed point, so its size only
#: shapes the iteration: 1e-6 of the storage slope is far below any full conduit's k. Where
#: the floor ALONE grounds the junction (every attached conduit pinned at the far end with
#: no k left, dry or normal-flow limited) nothing bounds the pass's step but the floor:
#: ``dH = dQ / (K_ANCHOR 2 A_s/dt)``, a large jump. Such a pass is still a valid iterate (the
#: fixed point is unchanged), its height is capped by the flood wall at y_max, and if the
#: jumps do not settle the step is refused by name for non-convergence. The
#: common pinned case (an outflow conduit dropping to a DN_CRITICAL end) keeps its lagged k
#: in the anchor, so the step is a Newton step, not the floor's.
K_ANCHOR = 1.0e-6
#: Cap on the slope of the normal-flow slope switch's wall (`DynamicWaveConduit`), as a
#: multiple of the smaller end junction's grounding (its storage slope ``2 A_s/dt`` plus its
#: conduits' conveyance, m2/s). The wall's fixed points do not depend on its slope. A wall
#: steeper than the junction it acts on makes the conduit's flow an S-curve in the head
#: (flat at q_norm, the wall, the momentum line) that the damped Newton head solve cycles
#: on: at 10 x the grounding the tree's drain-down failed the head solve (50 iterations at t
#: = 520 s); at 1 x the slopes a Newton step sees differ by at most 2, which its 0.75
#: damping contracts, and a conduit between two junctions settles on its switch by about
#: half per pass. Its round-off floor, ``S eps |H| / 2``, is then that of the junction's own
#: storage slope. The drop-end area switch's wall (`JunctionStorage`) uses the same factor on
#: its own junction's grounding above the threshold, for the same reason.
K_SWITCH = 1.0
#: Newton's absolute residual tolerance for the head solve (m3/s): a junction balance error
#: of `SOLVE_ATOL` dt m3 at most, far below the head tolerance times any surface area. It is
#: raised to the bed's round-off floor where that is larger (`solve_atol`).
SOLVE_ATOL = 1.0e-12
#: `solve_atol`'s margin over the bed's round-off floor (see there).
_BED_FLOOR_MARGIN = 4.0
#: The head solve's Newton keywords (`PotentialFlowLayer(newton_defaults=...)`; docs:
#: "Iteration and convergence"). Every law of the pass is non-decreasing in its own head
#: difference (conduits, regulators) or head (junction withdrawals), so the head solve's
#: residual is the gradient of a convex function, and ``line_search="convex"`` cuts a step
#: that overshoots its minimum back to it: the kinked laws (the stiff bed below an invert,
#: the normal-flow switch's S-curve, the hold band and flood wall, a regulator's lagged
#: slope) otherwise send the damped Newton into a 2-cycle it never leaves. ``step_floor``
#: ends a solve whose Newton step is within 16 ULPs of the heads: its residual is then at the
#: round-off of its own terms (a stiff wall or a large storage area at a high head can put
#: that above `solve_atol`).
HEAD_SOLVE_NEWTON = {"line_search": "convex", "step_floor": 16.0}
#: The head solve's inner linear solver is the dense LU (``linear_solver="direct"``) up to
#: this many junctions, and the layer's ``"auto"`` (SciPy's sparse LU) above it: on a small
#: network the dense factorisation costs a handful of tensor operations against SuperLU's
#: per-call conversion (measured on `benchmarks/sewer_dynwave_speed.py`'s binary trees, ms
#: per pass dense / sparse: 17.6 / 18.3 at 63 junctions, 18.5 / 19.3 at 127, 21.4 / 20.2 at
#: 255).
DENSE_SOLVE_MAX_JUNCTIONS = 128
#: SWMM's default HEAD_TOLERANCE, 0.005 ft (`dynwave.c:63`, ``DEFAULT_HEADTOL``), in m.
DEFAULT_HEAD_TOLERANCE = 0.005 * LCF
#: SWMM's default MAX_TRIALS (`dynwave.c:66`, ``DEFAULT_MAXTRIALS``).
DEFAULT_MAX_TRIALS = 8
#: The floor of the derived flow tolerance (m3/s, `iterate_tolerances`): the flow residual
#: the head solve resolves at its `SOLVE_ATOL` with margin, and the tolerance the parity
#: tests' 1e-13 m heads always ran with.
FLOW_TOL_FLOOR = 1.0e-10
#: SWMM's default junction surface-area floor (`dynwave.c:62`), ft2.
DEFAULT_MIN_SURFACE_AREA_FT2 = 12.566
#: SI area per SWMM ft2 (docs: "Units"): makes ``A_s dy = 0.5 dt N`` exact in m, m3/s.
_AREA_SI = QCF / LCF

#: The deepest fill (depth over diameter) the headspace geometry sees. A conduit flowing full
#: has no headspace (A_air = 0) and the headspace law's resistance is infinite there; the air
#: side reads the depth capped at this fraction of the diameter instead, so a full or
#: surcharged conduit keeps a finite, very small air area (about 6e-5 D**2) and a very large
#: resistance, with finite derivatives. It is nearly, not exactly, a closed path: the
#: interfacial drag does not depend on the headspace's area, so a small, nonzero flow of air
#: still moves through a capped full conduit (`DynamicWaveHydraulics`, "Air"; docs:
#: "Headspace air on the dynamic wave").
AIR_DEPTH_RATIO_MAX = 0.999

#: A manhole's shaft surface area (m2) when `Manhole.surface_area` is not set: SWMM's
#: MIN_SURFAREA of 12.566 ft2 (the 4 ft shaft), as the kinematic path's air capacity uses.
SHAFT_AREA_DEFAULT = 1.167
#: SWMM's link velocity is 0 at or below this link depth (`link.c:834`), ft.
_VELOCITY_MIN_DEPTH = 0.01
#: SWMM's shortest time step, MINTIMESTEP (`dynwave.c:60`), s.
MIN_TIME_STEP = 0.001
_OUTFALL_CODES = {"free": 0, "normal": 1, "fixed": 2, "tidal": 3, "timeseries": 4}
#: Outfall kinds whose stage varies in time: it is the ``"outfall_stage"`` driver's.
_STAGED_KINDS = ("tidal", "timeseries")
MSEC_PER_DAY = 86400.0 * 1000.0                                          # consts.h MSECperDAY


def tidal_stage(tide: tuple[tuple[float, float], ...], t: float) -> float:
    """A TIDAL outfall's stage elevation (m) `t` s into the run (`node.c::
    outfall_setOutletDepth`, node.c:1445-1452): the curve `tide` of ``(hour, stage)`` pairs
    is read at ``x = x_first + 24 frac(t / 1 day)`` (the elapsed time, not the clock time:
    the simulation's start is not added, so the tide begins at the curve's first hour and
    repeats every 24 h), by `table.c::table_lookup`: the first stage at or below the first
    hour, linear between entries, the last stage beyond the last hour (the curve is not
    wrapped, so one that stops short of 24 h holds its last stage)."""
    days = 1000.0 * t / MSEC_PER_DAY                                     # node.c:1449
    x = tide[0][0] + (days - math.floor(days)) * 24.0                    # node.c:1450
    x1, y1 = tide[0]
    if x <= x1:                                                          # table.c:420
        return y1
    for x2, y2 in tide[1:]:
        if x <= x2:                                                      # table.c:421
            return _table_interpolate(x, x1, y1, x2, y2)
        x1, y1 = x2, y2
    return y1                                                            # table.c:426


def _table_interpolate(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """`table.c::table_interpolate` (table.c:51-63)."""
    dx = x2 - x1
    if abs(dx) < 1.0e-20:
        return (y1 + y2) / 2.0
    return y1 + (x - x1) * (y2 - y1) / dx
#: The flow regulators' edge kinds in the ``"water"`` layer, in the order their columns
#: follow the pipes' in ``"water.q"`` (only the kinds a network has are present).
STRUCTURE_KINDS = ("orifice", "weir", "outlet", "pump")
_WEIR_TYPES = {"transverse": 0, "sideflow": 1, "v_notch": 2, "trapezoidal": 3}
_WEIR_SHAPES = {"transverse": RECT_OPEN, "sideflow": RECT_OPEN, "v_notch": TRIANGULAR,
                "trapezoidal": TRAPEZOIDAL}


@dataclass(frozen=True)
class DynamicWaveOptions:
    """SWMM 5.2.4's dynamic-wave [OPTIONS] that change the physics, with SWMM's defaults
    (`project.c:841-880`). `head_tolerance` and `max_trials` are SWMM's own iteration
    controls, and the defaults of `build_dynamic_wave_model`'s (`iterate_tolerances`; docs:
    "Iteration and convergence").

    `routing_step` is the fixed routing step (s); `inertial_damping` is ``"none"``,
    ``"partial"`` or ``"full"``; `normal_flow_limited` is ``"slope"``, ``"froude"``,
    ``"both"`` or ``"none"``; `surcharge_method` is ``"extran"`` (end depths capped at the
    crown, a junction above its crown holds continuity without storage) or ``"slot"``
    (Preissmann slot);
    `min_surface_area` is the junction surface-area floor (m2), ``None`` for SWMM's
    12.566 ft2; `head_tolerance` is SWMM's head convergence tolerance (the file's length
    unit, m for the CMS files read today; ``None`` for SWMM's 0.005 ft,
    `DEFAULT_HEAD_TOLERANCE`); `max_trials` is SWMM's trial limit (-, ``None`` for SWMM's 8,
    `DEFAULT_MAX_TRIALS`).

    `variable_step` is SWMM's VARIABLE_STEP, the Courant factor of the variable routing step
    (-, 0 to 2): 0 routes with the fixed `routing_step`, above 0 each step is the smallest
    of `routing_step`, the conduit Courant step and the junction depth-change step
    (`routing_step_size`; SWMM's own default is 0.75, this dataclass's is 0, so a fixed
    step is the default). `minimum_step` is SWMM's MINIMUM_STEP (s), the shortest variable
    step and the very first one.

    `lengthening_step` is SWMM's LENGTHENING_STEP (s, >= 0, 0 for none): a conduit shorter
    than the Courant length ``(sqrt(g yFull) + vFull) t`` is lengthened to it for the
    momentum terms, with ``t`` the smaller of `lengthening_step` and the routing step
    (`conduit_getLengthFactor`, link.c:1217-1254; the volume keeps the true length).
    SWMM takes the routing step before it is lowered to WET_STEP (`project.c:238-250`
    validates the conduits first), so `lengthening_routing_step` (s) is the ROUTING_STEP of
    the file where that differs from `routing_step`; ``None`` means `routing_step`.

    `force_main_equation` is SWMM's FORCE_MAIN_EQUATION, ``"H-W"`` (Hazen-Williams) or
    ``"D-W"`` (Darcy-Weisbach): the friction law of a full FORCE_MAIN conduit
    (`forcmain.c`; a partly full one uses the Manning n equivalent to it).

    `allow_ponding` is SWMM's ALLOW_PONDING: a junction that floods and has a positive
    `Manhole.ponded_area` keeps the water above its full depth, over that area, instead of
    losing it (`JunctionStorage`, "Ponding").
    """

    routing_step: float = 20.0
    inertial_damping: str = "partial"
    normal_flow_limited: str = "both"
    surcharge_method: str = "extran"
    min_surface_area: float | None = None
    head_tolerance: float | None = None
    max_trials: int | None = None
    variable_step: float = 0.0
    minimum_step: float = 0.5
    lengthening_step: float = 0.0
    lengthening_routing_step: float | None = None
    force_main_equation: str = "H-W"
    allow_ponding: bool = False

    @property
    def head_tol(self) -> float:
        """SWMM's ``HeadTol`` (m): `head_tolerance`, or SWMM's default 0.005 ft
        (`DEFAULT_HEAD_TOLERANCE`, dynwave.c:63) where the file sets none."""
        return DEFAULT_HEAD_TOLERANCE if self.head_tolerance is None else self.head_tolerance

    @property
    def trials(self) -> int:
        """SWMM's ``MaxTrials`` (-): `max_trials`, or SWMM's default 8
        (`DEFAULT_MAX_TRIALS`, dynwave.c:66) where the file sets none."""
        return DEFAULT_MAX_TRIALS if self.max_trials is None else int(self.max_trials)

    @property
    def min_route_step(self) -> float:
        """SWMM's `MinRouteStep` (s): `minimum_step` clipped to [MINTIMESTEP, routing_step]
        (`dynwave.c:184-185`)."""
        step = min(self.minimum_step, self.routing_step)                      # dynwave.c:184
        return max(step, MIN_TIME_STEP)                                       # dynwave.c:185

    def validate(self) -> None:
        """Refuse an unknown keyword or a non-positive step or area, BY NAME."""
        if not self.routing_step > 0:
            raise ValueError(
                f"DynamicWaveOptions: routing_step must be > 0 s, got {self.routing_step!r}"
            )
        for name, allowed in (
            ("inertial_damping", DAMPING),
            ("normal_flow_limited", NORMAL_FLOW_LIMITED),
            ("surcharge_method", SURCHARGE_METHODS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"DynamicWaveOptions: {name} must be one of {allowed}, got "
                    f"{getattr(self, name)!r}"
                )
        if not 0.0 <= self.variable_step <= 2.0:                              # project.c:698-703
            raise ValueError(
                f"DynamicWaveOptions: variable_step must be in [0, 2], got "
                f"{self.variable_step!r}")
        if not self.lengthening_step >= 0.0:
            raise ValueError(
                f"DynamicWaveOptions: lengthening_step must be >= 0 s, got "
                f"{self.lengthening_step!r}")
        if self.lengthening_routing_step is not None and not self.lengthening_routing_step > 0:
            raise ValueError(
                f"DynamicWaveOptions: lengthening_routing_step must be > 0 s or None, got "
                f"{self.lengthening_routing_step!r}")
        if not self.minimum_step >= 0.0:
            raise ValueError(
                f"DynamicWaveOptions: minimum_step must be >= 0 s, got {self.minimum_step!r}")
        if self.force_main_equation not in ("H-W", "D-W"):
            raise ValueError(
                f"DynamicWaveOptions: force_main_equation must be 'H-W' or 'D-W', got "
                f"{self.force_main_equation!r}")
        if self.min_surface_area is not None and not self.min_surface_area > 0:
            raise ValueError("DynamicWaveOptions: min_surface_area must be > 0 m2")
        if self.head_tolerance is not None and not self.head_tolerance > 0:
            raise ValueError(
                f"DynamicWaveOptions: head_tolerance must be > 0 or None, got "
                f"{self.head_tolerance!r}")
        if self.max_trials is not None and not self.max_trials >= 1:
            raise ValueError(
                f"DynamicWaveOptions: max_trials must be >= 1 or None, got {self.max_trials!r}")


class DynamicWaveConduit(ConstitutiveLaw):
    """SWMM's dynamic-wave conduit (`dwflow.c`) with its coefficients frozen for one pass.

    ``q = c0 + k dp`` (m3/s, dp = H1 - H2 in m), then ``min(q, q_norm)`` where SWMM's
    normal-flow check passed (`dwflow.c:255-258`), then the user flow limit, the flap gate and
    the dry-node caps in SWMM's order (below; `dwflow.c:270-281`): at most +`FUDGE_FLOW`
    out of a dry upstream node, at least -`FUDGE_FLOW` out of a dry downstream node (each
    times the conduit's barrels).

    The normal-flow slope switch (the tie rule). The check's flag is lagged (docs: "Bed
    stiffening and lagged pieces"): evaluated by the closure on the fed heads and flow. Where
    it is SWMM's slope test ``y1 < y2`` alone, with both end depths inside their clamps, the
    test is ``dp < dz`` (dz = z1 - z2, the conduit-end inverts), a switch on this element's
    own argument, and it is applied within the pass instead, as a continuous, non-decreasing
    curve::

        q = min(lin, max(q_norm, q_last + S (dp - dz)))

    with ``lin = c0 + k dp``, ``q_last`` the fed flow and ``S > 0`` a wall slope. At a
    fixed point (``q = q_last``) this is SWMM's unlimited flow ``lin`` where ``dp > dz``,
    its limited ``min(lin, q_norm)`` where ``dp < dz`` -- every SWMM fixed point is one of
    this curve, unchanged, whatever S -- or a point ON the switching surface ``dp = dz``
    (y1 = y2) with q between q_norm and lin, which is a fixed point here only. That last
    case is where SWMM has none: limited, the solution lies above the switch; unlimited,
    below it. A lagged flag then chatters forever, and SWMM's own trials alternate around
    the surface without converging (`tests/apps/sewer/test_dynwave.py::
    test_ramp_down_converges`). The surface is where SWMM's relaxed trials straddle; noodl
    puts the solution exactly on it, deterministically. The cost against SWMM is confined
    to those steps, where SWMM has no converged answer to compare with: its last trial
    lies within its own oscillation of the surface (~6e-4 m of J4's depth on the tree's
    drain-down). S only shapes the passes (`DynamicWaveHydraulics._switch_slope`).

    Reads the per-pipe drivers ``"water.flow_offset"`` c0 (m3/s), ``"water.conveyance"`` k
    (m2/s), ``"water.normal_flow"`` q_norm (m3/s, +inf where neither the limit nor the
    switch applies), the 0/1 flags ``"water.limited"`` (the lagged limit, where no switch
    applies), ``"water.slope_switch"``, ``"water.dry_upstream"``,
    ``"water.dry_downstream"``, and the switch's ``"water.switch_drop"`` dz (m),
    ``"water.switch_slope"`` S (m2/s) and ``"water.flow_last"`` q_last (m3/s), all written
    by `DynamicWaveHydraulics`.

    The element limits that follow the normal-flow limit, in SWMM's order (`dwflow.c:269-281`;
    SWMM's under-relaxation, :259-267, is not modelled): the user flow limit ``|q| <=``
    ``"water.max_flow"`` (m3/s, absent or +inf for none, the default; `dwflow.c:269-273`),
    the flap gate zeroing flow against it (`link.c::link_setFlapGate`:
    ``"water.flap_block_reverse"`` 1 zeroes
    ``q < 0``, ``"water.flap_block_forward"`` 1 zeroes ``q > 0``; `dwflow.c:276`), then the
    dry-node caps at ``"water.dry_flow_cap"`` (m3/s, `FUDGE_FLOW` times the barrels;
    `dwflow.c:280-281`). Each is a
    clamp of ``q`` against a constant, so the flow stays continuous and non-decreasing in ``dp``
    (the head solve's Newton step needs both) with ``dflow`` 0 wherever one acts.
    """

    def __init__(self, kind: str = "pipe") -> None:
        super().__init__(kind)

    def _parts(self, dp: Tensor, drivers: Mapping
               ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """The momentum line, the flow, the switch's wall (NaN where no wall acts) and the
        culvert cap's line (None without culverts)."""
        lin = drivers["water.flow_offset"] + drivers["water.conveyance"] * dp
        q_norm = drivers["water.normal_flow"]
        limited = drivers["water.limited"] > 0.5
        switch = drivers["water.slope_switch"] > 0.5
        wall = (drivers["water.flow_last"]
                + drivers["water.switch_slope"] * (dp - drivers["water.switch_drop"]))
        slid = torch.minimum(lin, torch.maximum(q_norm, wall))
        q = torch.where(limited, torch.minimum(lin, q_norm),
                        torch.where(switch, slid, lin))                    # dwflow.c:255-258
        inlet = None
        cap_flow = drivers.get("water.cap_flow")
        if cap_flow is not None:
            # inlet control (dwflow.c:250-252): the culvert's inlet flow at the upstream
            # head, linearised about the fed head difference with SWMM's dQ/dH
            line = cap_flow + drivers["water.cap_slope"] * (dp - drivers["water.cap_drop"])
            inlet = torch.where(torch.isfinite(cap_flow), line, torch.inf)
            q = torch.minimum(q, inlet)
        # each limit below is optional (absent: none), a clamp against a constant
        limit = drivers.get("water.max_flow")
        if limit is not None:
            q = torch.minimum(torch.maximum(q, -limit), limit)             # dwflow.c:269-273
        block_reverse = drivers.get("water.flap_block_reverse")
        if block_reverse is not None:
            q = torch.where(block_reverse > 0.5, torch.clamp(q, min=0.0), q)  # dwflow.c:276
        block_forward = drivers.get("water.flap_block_forward")
        if block_forward is not None:
            q = torch.where(block_forward > 0.5, torch.clamp(q, max=0.0), q)
        cap = drivers.get("water.dry_flow_cap", FUDGE_FLOW)
        up = drivers["water.dry_upstream"] > 0.5
        dn = drivers["water.dry_downstream"] > 0.5
        q = torch.where(up, torch.clamp(q, max=cap), q)                      # dwflow.c:280
        q = torch.where(dn, torch.clamp(q, min=-cap), q)                     # dwflow.c:281
        return lin, q, torch.where(switch & ~limited, wall, torch.nan), inlet

    def flow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Conduit flow (m3/s, positive from the pipe's first node to its second)."""
        return self._parts(dp, drivers)[1]

    def dflow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """dq/d(dp) (m2/s): k where the flow is the momentum line, the wall slope S where it
        is on the switch's wall, else 0 (every other limit is a constant, so ``q == lin``
        and ``q == wall`` are exact tests)."""
        lin, q, wall, inlet = self._parts(dp, drivers)
        out = torch.where(q == lin, drivers["water.conveyance"],
                          torch.where(q == wall, drivers["water.switch_slope"], 0.0))
        if inlet is not None:
            out = torch.where((q == inlet) & (q != lin) & (q != wall),
                              drivers["water.cap_slope"], out)
        return out


class StructureLink(ConstitutiveLaw):
    """A flow regulator (orifice, weir or outlet) or pump with its flow linearised about the
    previous pass: ``q = c0 + g (dp - dp_fed)`` (m3/s, dp = H1 - H2 in m), then the flap
    gate's clamp (a pump's, which never flows backwards).

    ``c0`` is SWMM's own regulator flow at the previous pass's heads (`link.c`
    ``orifice_getInflow`` / ``weir_getInflow`` / ``outlet_getInflow`` / ``pump_getInflow``,
    `noodl.apps.sewer.structures`, `noodl.apps.sewer.pumps`), ``dp_fed`` that pass's head
    difference, so at a fixed point (dp = dp_fed) the flow is SWMM's exactly, whatever ``g``.
    ``g >= 0`` (m2/s) is SWMM's ``dqdh`` for orifices and weirs, a central difference in the
    upstream head for outlets, the slope of the curve in the head delivered for a PUMP3 or
    PUMP5 pump (none for the other pumps, whose flow does not depend on the head difference):
    it only makes each pass a Newton-like step in the device's heads. The flow is
    non-decreasing in dp (``g >= 0``, the clamps are constants), as the head solve needs.

    Reads the per-device drivers ``"water.<kind>.flow_offset"`` c0 (m3/s),
    ``"water.<kind>.slope"`` g (m2/s), ``"water.<kind>.dp_fed"`` (m) and the 0/1 flags
    ``"water.<kind>.flap_block_reverse"`` (its own flap gate, or a gated outfall at its
    second node: ``q >= 0``) and ``"water.<kind>.flap_block_forward"`` (a gated outfall at
    its first node: ``q <= 0``; `link.c::link_setFlapGate`), all written by
    `DynamicWaveHydraulics`. At a fixed point the clamps never act: SWMM's flow already
    respects the gate."""

    def __init__(self, kind: str) -> None:
        if kind not in STRUCTURE_KINDS:
            raise ValueError(f"StructureLink: kind must be one of {STRUCTURE_KINDS}, got "
                             f"{kind!r}")
        super().__init__(kind)
        self.prefix = f"water.{kind}."

    def _parts(self, dp: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor]:
        p = self.prefix
        lin = drivers[p + "flow_offset"] + drivers[p + "slope"] * (dp - drivers[p + "dp_fed"])
        zero = torch.zeros_like(lin)
        inf = torch.full_like(lin, torch.inf)
        reverse = drivers.get(p + "flap_block_reverse", zero) > 0.5
        forward = drivers.get(p + "flap_block_forward", zero) > 0.5
        q = torch.minimum(torch.maximum(lin, torch.where(reverse, zero, -inf)),
                          torch.where(forward, zero, inf))                  # link.c:643-670
        return lin, q

    def flow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Regulator flow (m3/s, positive from its first node to its second)."""
        return self._parts(dp, drivers)[1]

    def dflow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """dq/d(dp) (m2/s): g where the gate's clamp does not act, else 0."""
        lin, q = self._parts(dp, drivers)
        g = drivers[self.prefix + "slope"].expand_as(q)
        return torch.where(q == lin, g, torch.zeros_like(g))


@dataclass(frozen=True)
class _Structures:
    """One kind of flow regulator in the closure: ``names``; end node indices ``u``, ``v``;
    ``offset1`` (ft, the crest above node 1's invert after SWMM raises a crest below node
    2's invert to it, link.c:425-433) and ``offset2`` (ft, the crest height as given, which
    SWMM keeps at node 2, link.c:366-388); ``y_full`` (ft, the opening height; an outlet's
    is SWMM's DUMMY ``TINY``); ``params`` (`OrificeSet` / `WeirSet` / `OutletSet`);
    ``block_forward``/``block_reverse`` (bool, `structures._gate`); ``extends`` (bool: the
    device extends node 1's full depth to its crown, all but bottom orifices,
    link.c:448-455)."""

    kind: str
    names: list
    u: Tensor
    v: Tensor
    offset1: Tensor
    offset2: Tensor
    y_full: Tensor
    params: object
    block_forward: Tensor
    block_reverse: Tensor
    extends: Tensor
    order: Tensor | None = None
    first: int = 0


def _build_structures(kind: str, devices, nodes: list, invert: Tensor, gated_outfalls: set,
                      route_step: float) -> _Structures:
    """The closure's view of `devices` of one `kind` (`_Structures`); `route_step` (s) sets
    the orifice and weir equivalent lengths."""
    u = torch.tensor([nodes.index(s.u) for s in devices], dtype=torch.long)
    v = torch.tensor([nodes.index(s.v) for s in devices], dtype=torch.long)
    order = torch.tensor([s.order for s in devices], dtype=torch.long)
    if kind == "pump":
        # a pump has no crest, opening or flap gate and never flows backwards; link_validate
        # skips it (link.c:443-445)
        zero = torch.zeros(len(devices), dtype=F64)
        return _Structures(kind, [s.name for s in devices], u, v, zero, zero, zero,
                           PumpSet.build(devices),
                           torch.zeros(len(devices), dtype=torch.bool),
                           torch.ones(len(devices), dtype=torch.bool),
                           torch.zeros(len(devices), dtype=torch.bool), order)
    crest = torch.tensor([s.crest_height for s in devices], dtype=F64)
    inv1, inv2 = invert[u], invert[v]
    # link.c:425-433: under DW a crest below node 2's invert is raised to it (offset1 only)
    offset1 = torch.where(inv1 + crest < inv2, inv2 - inv1, crest) / LCF
    own_gate = torch.tensor([s.flap_gate for s in devices], dtype=torch.bool)
    block_reverse = own_gate | torch.tensor([s.v in gated_outfalls for s in devices],
                                            dtype=torch.bool)
    block_forward = torch.tensor([s.u in gated_outfalls for s in devices], dtype=torch.bool)
    extends = torch.ones(len(devices), dtype=torch.bool)
    if kind == "orifice":
        opening = Opening.from_si(
            [CIRCULAR if s.shape == "circular" else RECT_CLOSED for s in devices],
            [s.height for s in devices],
            [s.height if s.width is None else s.width for s in devices])
        bottom = torch.tensor([s.orientation == "bottom" for s in devices], dtype=torch.bool)
        params = OrificeSet.build(opening, bottom,
                                  [s.discharge_coefficient for s in devices], own_gate,
                                  route_step)
        extends = ~bottom                                                    # link.c:442-444
        y_full = opening.y_full
    elif kind == "weir":
        opening = Opening.from_si([_WEIR_SHAPES[s.kind] for s in devices],
                                  [s.height for s in devices], [s.width for s in devices],
                                  [tuple(s.side_slopes) for s in devices])
        params = WeirSet.build(opening, [_WEIR_TYPES[s.kind] for s in devices],
                               [s.discharge_coefficient for s in devices],
                               [s.end_discharge_coefficient for s in devices],
                               [s.end_contractions for s in devices],
                               [s.can_surcharge for s in devices], own_gate)
        y_full = opening.y_full
    else:
        params = OutletSet.build([s.rating for s in devices],
                                 [s.head_difference for s in devices], own_gate)
        y_full = torch.full((len(devices),), TINY, dtype=F64)                # xsect.c:233
    return _Structures(kind, [s.name for s in devices], u, v, offset1, crest / LCF, y_full,
                       params, block_forward, block_reverse, extends, order)


class JunctionStorage(NodeSource):
    """SWMM's junction continuity as a node withdrawal: trapezoidal storage over the step,
    with a stiff bed below the invert, or no storage at all where the junction is
    EXTRAN-surcharged, capped by flooding at the junction's top.

    ``w(H) = (2 A_s/dt)(H - H_start) - N_start + (2 A_bed/dt) K_BED min(H - z_bed, 0)``
    (m3/s, positive out of the network), so that with the conduit flows and the lateral
    inflow the node balance reads ``A_s (H - H_start) = 0.5 dt (N_start + N)``
    (`dynwave.c:694-711`). The bed term stands in for SWMM's ``yNew < 0 -> 0``
    (`dynwave.c:743`); the slope is at least ``2 A_s/dt > 0``, so every junction grounds
    itself. The bed stiffness uses the fixed `bed_area` A_bed (m2; the model passes SWMM's
    surface-area floor, which no A_s is below), not A_s, so that the head solve's round-off
    floor on the bed is known before the step (`solve_atol`). Relative to the junction's own
    storage slope the bed is ``K_BED A_bed / A_s`` stiffer: 1e6 at the floor, but only ~1e3
    where A_s is a thousand times a tiny user MIN_SURFAREA.

    EXTRAN surcharge (`crown` given, the junction crown heads in m). Above its
    crown a junction stores nothing: SWMM's surcharged update ``y_new = y_last + corr
    dQ/denom`` (`dynwave.c:716-741`) is at its fixed point exactly when the net inflow dQ is
    0. With ``w_c`` the free storage withdrawal at the crown (``(2 A_s/dt)(H_crown - H_start)
    - N_start``, or the drop-end area switch's value there where one applies), ``stiff =
    (2/dt) max(K_HOLD A_bed, K_HOLD_STORAGE A_s)`` and the band ``b(H) = w_c + stiff (H -
    H_crown)``:

    - a SURCHARGED junction (driver ``"water.surcharged"`` 1, set by `surcharged` from the
      fed head ``H_last``: above the crown and ``b(H_last) > 0``, i.e. above the band where
      ``w_c < 0``) withdraws the anchor ``s (H - H_last)``, zero at the
      fixed point, with ``s`` the driver ``"water.surcharge_slope"`` (m2/s): the
      conveyance of the attached conduits that the pass lags into their c0 (a conduit whose
      other end is pinned), so that the pass is a Newton step in this junction's head, plus
      a floor `K_ANCHOR` (2 A_s/dt) that keeps the head solve nonsingular;
    - any other junction withdraws ``storage(H)`` at and below its crown and ``max(storage(H),
      b(H))`` above it (the band, which is steeper): continuous at the crown, a convex curve
      (one Newton-safe solution per pass). The band never acts below the crown.

    The fixed points are SWMM's: a free-surface solution below the crown; a surcharged one
    (net inflow 0) above it -- the flag is lagged on the fed head as SWMM's test
    ``y_last > y_crown`` (`dynwave.c:688`) is on the previous trial, and a surcharged
    solution below the crown drops it on the next pass, as SWMM's floor ``y_new < y_crown ->
    y_crown - FUDGE`` (`dynwave.c:735-736`) does. Where SWMM has NO fixed point (the storage
    equation ends above the crown, but at the crown the junction already loses water, so the
    surcharged solution lies below it) SWMM's trials alternate between its two branches and
    a flag lagged on the depth alone chatters forever. Here the band takes that case: the
    junction is held at its crown (within the band, ``|w_c|/stiff``, below a millimetre),
    releasing what the balance needs, ``w`` between ``w_c`` and 0 -- the deterministic tie
    rule. The band also moves a surcharged solution that lies within ``|w_c|/stiff`` above
    the crown onto the band; SWMM's surcharged depth lies ``N(crown)/sum(k)`` above it, so
    that needs a net inflow at the crown below ``|w_c| sum(k)/stiff``, a thin sliver.

    Flooding (`top` given, the junction heads at y_max = full depth + SurDepth,
    m). SWMM holds a junction whose update would end above y_max at y_max and lets the
    excess leave the system (`dynwave.c:743-795`, no ponding). With ``w`` the withdrawal
    above, ``S`` the wall slope (the band's ``stiff``, at least the anchor's ``s`` where
    surcharged) and the lagged excess ``E = max(N_last - w(top), 0)`` (`flood_excess`;
    driver ``"water.flood_excess"``, m3/s, from the fed net inflow N_last), every junction
    withdraws ``max(w(H), w(top) + E + S (H - top))``:

    - the wall caps the junction at the top within the pass (it rises when the head reaches
      y_max, no lag), and the max of two increasing lines stays convex (Newton-safe);
    - at a flooded fixed point the fed net inflow is the solved one, ``N_last = N``, so the
      balance ``w = N`` puts H EXACTLY at the top whatever S is, and the withdrawal ``N`` is
      the flow over the top; S only sets how fast the passes contract, by ``sum(k)/(S +
      sum(k))`` with ``sum(k)`` the junction's conduit conveyance. That is why S is the hold
      band's stiffness (a round-off floor of ~3e-13 m3/s at 12 m, inside a 1e-12 m3/s head
      solve) and not a 1e6 penalty, whose round-off floor would be ~6e-10 m3/s and whose
      offset ``N/S`` would leave the head ~1e-7 m above y_max;
    - E > 0 exactly where SWMM floods: SWMM's update ends above y_max iff the net inflow at
      the top exceeds what the non-flooded form would withdraw there (the storage form's
      ``0.5 dt (N_start + N) > A_s (y_max - y_start)``, the surcharged form's ``dQ > 0``).
      E falls to 0 when that excess would be negative, and the junction leaves the top on
      the next pass: there is no flag to chatter, only a lagged excess that contracts.
      Below the top the wall reaches down only by ``E/(S - slope)``, a transient within the
      passes.

    The drop-end area switch (the tie rule of `DynamicWaveHydraulics`; drivers
    ``"water.area_switch_head"`` H* (m, +inf where none applies), ``"water.area_below"`` and
    ``"water.area_above"`` (m2), ``"water.area_switch_slope"`` S (m2/s) and the fed net
    inflow ``"water.net_inflow_last"`` N_last, all optional). Where H* is finite the
    storage term is, within the pass, ``max(lo(H), min(hi(H), N_last + S (H - H*)))`` with
    lo/hi the storage lines with the area below/above the threshold: continuous and
    non-decreasing. At a fixed point (withdrawal = N = N_last) it is the lower-area storage
    below H* and the upper-area storage above it -- SWMM's own fixed points, unchanged --
    or the threshold itself, H = H*, with the withdrawal inside the jump: the solution where
    SWMM has none.

    Ponding (`pond_head` given, SWMM's ALLOW_PONDING with a positive Aponded). Such a junction
    does not flood: it keeps its depth above the full depth (`dynwave.c:661, 738-748, 753,
    766-795`) over the area ``A_p`` plus its conduits' (``"water.pond_area"``, m2; a trial's
    area is the ponded one when the previous trial's depth is above the full depth,
    `node_getPondedArea`, `dynwave.c:309-312`), so SWMM's update is one of two lines about the
    full head H* (``pond_head``): the unponded one below it (storage, the band, or the
    surcharged update ``wu``) and the ponded one above it (``wp``, storage over the larger
    area, no surcharge: `dynwave.c:679-680`). A surcharged update that ends above H* is cut at
    H* + FUDGE and the next trial is ponded (`dynwave.c:738-740`); a ponded update that ends
    below H* is raised to H* - FUDGE (`dynwave.c:708-710`) and the next is unponded. Where
    ``wu(H*) <= wp(H*)`` (the junction rises through H*) those trials alternate when the
    unponded update ends above H* and the ponded one below it, and the junction is held at H*,
    the drop-end area switch's tie rule (``"water.pond_switch_slope"`` the wall's slope);
    otherwise the line of the fed head's side (``"water.ponded"``, 0/1: the fed head above H*)
    is the one SWMM's trials settle on. `_pond_free` has the formula. A ponding junction has
    no flood wall, and `surchargeable` is False where its full depth is its crown (nothing is
    surcharged between).

    Reads the per-junction drivers ``"water.storage_area"`` A_s (m2, `_AREA_SI`),
    ``"water.head_start"`` (m), ``"water.net_inflow_start"`` (m3/s) and
    ``"water.storage_rate"`` 1/dt (1/s); with `crown` also ``"water.surcharged"`` (0/1)
    and ``"water.surcharge_slope"`` (m2/s), and `surcharged` reads ``"water.head_last"``
    (m); with `top` also ``"water.flood_excess"`` (m3/s), and `flood_excess` reads
    ``"water.net_inflow_last"`` (m3/s). `bed` holds the junction inverts (m); without
    `crown` (SLOT) no junction surcharges, without `top` none floods. `surchargeable`
    (bool per junction, default all) marks the junctions that can surcharge at all: a SWMM
    storage unit only with a SurDepth, and then above its full depth, which is its `crown`
    here (`dynwave.c:683-687`); a plain junction only with a positive crown depth
    (`dynwave.c:690`, ``yCrown > 0``; `DynamicWaveHydraulics` passes both rules). Where it is
    False the band and the flag never act.
    """

    def __init__(self, nodes: Tensor, bed: Tensor, bed_area: float,
                 crown: Tensor | None = None, top: Tensor | None = None,
                 surchargeable: Tensor | None = None,
                 pond_head: Tensor | None = None) -> None:
        super().__init__(nodes)
        self.register_buffer("bed", torch.as_tensor(bed, dtype=F64))
        self.bed_area = float(bed_area)
        self.extran = crown is not None
        if self.extran:
            self.register_buffer("crown", torch.as_tensor(crown, dtype=F64))
            mask = (torch.ones_like(self.crown, dtype=torch.bool) if surchargeable is None
                    else torch.as_tensor(surchargeable, dtype=torch.bool))
            self.register_buffer("surchargeable", mask)
        self.floods = top is not None
        if self.floods:
            self.register_buffer("top", torch.as_tensor(top, dtype=F64))
        #: whether any junction ponds; `pond_head` is H* per junction (+inf where none)
        self.ponds = pond_head is not None and bool(torch.isfinite(
            torch.as_tensor(pond_head, dtype=F64)).any())
        if self.ponds:
            self.register_buffer("pond_head", torch.as_tensor(pond_head, dtype=F64))

    def _storage(self, phi: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor, Tensor]:
        """The free-surface storage withdrawal (m3/s) and slope (m2/s) at `phi`, and the band's
        stiffness (m2/s)."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        area = drivers["water.storage_area"]
        bed = rate2 * self.bed_area * K_BED
        dry = torch.clamp(phi - self.bed, max=0.0)
        rise = phi - drivers["water.head_start"]
        n_start = drivers["water.net_inflow_start"]
        storage = rate2 * area * rise - n_start                              # dynwave.c:694-711
        slope = (rate2 * area).expand_as(storage)
        if "water.area_switch_head" in drivers:
            storage, slope = self._area_switch(phi, rise, storage, slope, drivers)
        storage = storage + bed * dry
        slope = slope + bed * (phi < self.bed).to(phi.dtype)
        return storage, slope, self._stiff(drivers)

    @staticmethod
    def _area_switch(phi: Tensor, rise: Tensor, storage: Tensor, slope: Tensor,
                     drivers: Mapping) -> tuple[Tensor, Tensor]:
        """The drop-end area switch within the pass (class docstring): ``max(lo, min(hi,
        N_last + S (H - H*)))`` where a threshold applies (finite ``H*``), the lagged storage
        and slope elsewhere."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        h_star = drivers["water.area_switch_head"]
        on = torch.isfinite(h_star)
        if not bool(on.any()):
            return storage, slope
        h_safe = torch.where(on, h_star, phi)
        n_start = drivers["water.net_inflow_start"]
        lo = rate2 * drivers["water.area_below"] * rise - n_start
        hi = rate2 * drivers["water.area_above"] * rise - n_start
        s = drivers["water.area_switch_slope"]
        wall = drivers["water.net_inflow_last"] + s * (phi - h_safe)
        capped = torch.minimum(hi, wall)
        switched = torch.maximum(lo, capped)
        slope_sw = torch.where(
            switched == lo, (rate2 * drivers["water.area_below"]).expand_as(lo),
            torch.where(capped == hi, (rate2 * drivers["water.area_above"]).expand_as(hi),
                        s.expand_as(wall)))
        return torch.where(on, switched, storage), torch.where(on, slope_sw, slope)

    def ponded(self, drivers: Mapping) -> Tensor:
        """The ponded state (0/1, float64, per junction): a ponding junction whose fed head
        ``drivers["water.head_last"]`` is above its full head (SWMM's ``isPonded``,
        `dynwave.c:662`); 0 elsewhere."""
        head_last = drivers["water.head_last"]
        if not self.ponds:
            return torch.zeros_like(head_last)
        return (head_last > self.pond_head).to(F64)

    def _pond_free(self, phi: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor]:
        """The ponding junctions' withdrawal and slope at `phi` (class docstring, "Ponding"),
        the other junctions' `_unponded` one. ``wu`` is the withdrawal below the full head H*
        (storage, the hold band, or the surcharged anchor taken as SWMM's next trial would, from
        the fed head capped at H*), ``wp`` the line with the ponded area added. Where ``wu(H*)
        <= wp(H*)`` the junction rises through H*: ``max(wu, min(wp, N_last + S (H - H*)))``,
        the drop-end area switch with its hold at H* (SWMM's trials alternate about the full
        depth where the unponded update ends above it and the ponded one below it). Otherwise
        the previous trial's side decides (``"water.ponded"``): ``wp`` above, ``wu`` below."""
        base, base_slope = self._unponded(phi, drivers)
        on = torch.isfinite(self.pond_head)
        h_star = torch.where(on, self.pond_head, phi)
        rate2 = 2.0 * drivers["water.storage_rate"]
        a_hi = drivers["water.pond_area"]
        head_start, n_start = drivers["water.head_start"], drivers["water.net_inflow_start"]
        # the surcharged form as the next trial would take it, also at a fed head above H*
        anchor_slope = drivers["water.surcharge_slope"]
        eligible = self._eligible(drivers)
        head_c = torch.minimum(drivers["water.head_last"], h_star)

        def unponded(x: Tensor, free: Tensor, free_slope: Tensor) -> tuple[Tensor, Tensor]:
            if not self.extran:
                return free, free_slope
            return (torch.where(eligible, anchor_slope * (x - head_c), free),
                    torch.where(eligible, anchor_slope.expand_as(free_slope), free_slope))

        def ponded_line(x: Tensor) -> Tensor:
            return rate2 * a_hi * (x - head_start) - n_start

        wu, wu_slope = unponded(phi, base, base_slope)
        wp = ponded_line(phi)
        wp_slope = (rate2 * a_hi).expand_as(wp)
        at_star, _ = self._unponded(h_star, drivers)
        at_star, _ = unponded(h_star, at_star, at_star)
        rises = at_star <= ponded_line(h_star)
        s_wall = drivers["water.pond_switch_slope"]
        wall = drivers["water.net_inflow_last"] + s_wall * (phi - h_star)
        capped = torch.minimum(wp, wall)
        tie = torch.maximum(wu, capped)
        tie_slope = torch.where(tie == wu, wu_slope,
                                torch.where(capped == wp, wp_slope, s_wall.expand_as(wall)))
        ponded = drivers["water.ponded"] > 0.5
        lag = torch.where(ponded, wp, wu)
        lag_slope = torch.where(ponded, wp_slope, wu_slope.expand_as(wp))
        chosen = torch.where(rises, tie, lag)
        chosen_slope = torch.where(rises, tie_slope, lag_slope)
        return torch.where(on, chosen, base), torch.where(on, chosen_slope, base_slope)

    def _stiff(self, drivers: Mapping) -> Tensor:
        """The band's slope (m2/s, per junction): ``(2/dt) max(K_HOLD A_bed,
        K_HOLD_STORAGE A_s)``; the driver ``"water.hold_stiffness"`` where the closure has
        written it for the pass."""
        stiff = drivers.get("water.hold_stiffness")
        if stiff is not None:
            return stiff
        rate2 = 2.0 * drivers["water.storage_rate"]
        return rate2 * torch.clamp(K_HOLD_STORAGE * drivers["water.storage_area"],
                                   min=K_HOLD * self.bed_area)

    def crown_withdrawal(self, drivers: Mapping) -> Tensor:
        """``w_c`` (m3/s, per junction): the free storage withdrawal at the crown, the
        drop-end area switch included (so that the withdrawal is continuous there). It
        depends on the drivers only: the closure writes it once per pass as the driver
        ``"water.crown_withdrawal"``, which `flow` and `dflow` then read."""
        return self._storage(self.crown, drivers)[0]

    def _band(self, phi: Tensor, drivers: Mapping) -> Tensor:
        """The band ``w_c + stiff (H - H_crown)`` (m3/s) at `phi` (m) (`crown_withdrawal`)."""
        w_c = drivers.get("water.crown_withdrawal")
        if w_c is None:
            w_c = self.crown_withdrawal(drivers)
        return w_c + self._stiff(drivers) * (phi - self.crown)

    def surcharged(self, drivers: Mapping) -> Tensor:
        """The EXTRAN surcharge flag (0/1, float64) from the fed heads
        ``drivers["water.head_last"]`` (m): the band positive there (class docstring). With
        a `top` the fed head is taken at most at the top, as SWMM's previous-trial depth
        never exceeds y_max (`dynwave.c:751-754`): a junction whose top IS its crown
        (MaxDepth 0, SurDepth 0) never surcharges."""
        head_last = drivers["water.head_last"]
        if not self.extran:
            return torch.zeros_like(head_last)
        return (self._eligible(drivers) & ~(self.ponded(drivers) > 0.5)).to(F64)  # :679-691

    def _eligible(self, drivers: Mapping) -> Tensor:
        """Where the fed head makes a junction surcharged but for ponding (bool): above the
        crown, surchargeable, and the band positive there (`surcharged`); a ponding junction's
        fed head counts at most at its full head, where SWMM's next trial would start."""
        head_last = drivers["water.head_last"]
        if not self.extran:
            return torch.zeros_like(head_last, dtype=torch.bool)
        if self.floods:
            head_last = torch.minimum(head_last, self.top)
        return ((head_last > self.crown) & self.surchargeable
                & (self._band(head_last, drivers) > 0.0))                # dynwave.c:683-691

    def _free(self, phi: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor]:
        """The non-flooded withdrawal (m3/s) and its slope (m2/s) at `phi` (m): storage,
        the hold band above the crown, or the surcharged anchor (class docstring), and for a
        ponding junction `_pond_free`."""
        if self.ponds and "water.pond_area" in drivers:
            return self._pond_free(phi, drivers)
        return self._unponded(phi, drivers)

    def _unponded(self, phi: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor]:
        """`_free` without ponding: the withdrawal and slope of the junction below its full
        head (and of every junction that does not pond)."""
        storage, slope, stiff = self._storage(phi, drivers)
        if not self.extran:
            return storage, slope
        band = self._band(phi, drivers)
        above = (phi > self.crown) & self.surchargeable
        if self.ponds and "water.ponded" in drivers:
            above = above & (drivers["water.ponded"] < 0.5)             # dynwave.c:679-680
        free = torch.where(above, torch.maximum(storage, band), storage)
        slope = torch.where(above & (band > storage), stiff.expand_as(band), slope)
        surcharged = drivers["water.surcharged"] > 0.5
        anchor_slope = drivers["water.surcharge_slope"]
        anchor = anchor_slope * (phi - drivers["water.head_last"])
        return (torch.where(surcharged, anchor, free),                      # dynwave.c:716
                torch.where(surcharged, anchor_slope.expand_as(slope), slope))

    def _wall_slope(self, drivers: Mapping) -> Tensor:
        """The flood wall's slope (m2/s): the hold band's stiffness, and at least the
        anchor's slope where surcharged, so that above the top the wall always rises
        faster than the withdrawal it caps. The driver ``"water.flood_wall_slope"`` where the
        closure has written it for the pass."""
        given = drivers.get("water.flood_wall_slope")
        if given is not None:
            return given
        stiff = self._stiff(drivers)
        if not self.extran:
            return stiff
        return torch.where(drivers["water.surcharged"] > 0.5,
                           torch.maximum(stiff, drivers["water.surcharge_slope"]), stiff)

    def top_withdrawal(self, drivers: Mapping) -> Tensor:
        """``w(top)`` (m3/s, per junction): the non-flooded withdrawal at the top (shaped as
        ``drivers["water.net_inflow_last"]``). It depends on the drivers only (the
        surcharge flag included): the closure writes it once per pass as the driver
        ``"water.top_withdrawal"``, which `flood_excess`, `flow` and `dflow` then read."""
        w_top = drivers.get("water.top_withdrawal")
        if w_top is None:
            top = self.top.expand(drivers["water.net_inflow_last"].shape)
            w_top = self._free(top, drivers)[0]
        return w_top

    def _wall_on(self, drivers: Mapping) -> Tensor | None:
        """Where the flood wall acts (bool per junction), ``None`` for everywhere: at the
        junctions that do not pond (a ponding junction's ``_pond_free`` is its own wall at the
        full head)."""
        if not self.ponds:
            return None
        return ~torch.isfinite(self.pond_head)

    def flood_excess(self, drivers: Mapping) -> Tensor:
        """The lagged flood excess E (m3/s, >= 0): the fed net inflow
        ``drivers["water.net_inflow_last"]`` over the non-flooded withdrawal at the top,
        ``max(N_last - w(top), 0)``; > 0 exactly where SWMM's update would end above y_max
        (`dynwave.c:752`), i.e. where the junction floods."""
        excess = torch.clamp(drivers["water.net_inflow_last"] - self.top_withdrawal(drivers),
                             min=0.0)
        on = self._wall_on(drivers)
        return excess if on is None else torch.where(on, excess, torch.zeros_like(excess))

    def _wall(self, phi: Tensor, drivers: Mapping) -> Tensor:
        """The flood wall ``w(top) + E + S (H - top)`` (m3/s) at `phi` (m)."""
        w_top = drivers.get("water.top_withdrawal")
        if w_top is None:
            w_top, _ = self._free(self.top.expand(phi.shape), drivers)
        return (w_top + drivers["water.flood_excess"]
                + self._wall_slope(drivers) * (phi - self.top))

    def flow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Withdrawal (m3/s) at junction heads `phi_nodes` (m)."""
        free, _ = self._free(phi_nodes, drivers)
        if not self.floods:
            return free
        walled = torch.maximum(free, self._wall(phi_nodes, drivers))       # dynwave.c:752
        on = self._wall_on(drivers)
        return walled if on is None else torch.where(on, walled, free)

    def dflow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """d(withdrawal)/dH (m2/s), analytic and strictly positive."""
        free, slope = self._free(phi_nodes, drivers)
        if self.floods:
            wall = self._wall(phi_nodes, drivers)
            wall_slope = self._wall_slope(drivers).expand_as(wall)
            hold = wall >= free
            on = self._wall_on(drivers)
            if on is not None:
                hold = hold & on
            slope = torch.where(hold, wall_slope, slope)
        return slope.expand(broadcast_shapes(slope.shape, phi_nodes.shape))


class DynamicWaveHydraulics:
    """The dynamic-wave closure: the conduit coefficients, junction surface areas and
    outfall heads of one pass, from the previous pass's heads and flows.

    Reads the fed ``"water.phi"`` (m, full-node) and ``"water.q"`` (m3/s, per link: the
    pipes, then the regulators), the
    full-node driver ``"inflow"`` (m3/s, lateral inflow, zero at outfalls) and its own
    carried state, pinned by `Model` to the step start: ``"sewer.junction_depth"`` (m),
    ``"sewer.pipe_flow"`` (m3/s), ``"sewer.mid_area"`` (SWMM's stored mid-depth area, SI
    area m2, `_AREA_SI`) and ``"sewer.net_inflow"`` (m3/s, per junction). With a
    `StepContext` it writes ``"water.sources"``, ``"water.phi_boundary"`` (outfall heads, m),
    the `DynamicWaveConduit` drivers and the `JunctionStorage` drivers, and the
    reporting key ``"sewer.overflow"`` (below); always the carried keys and the reporting
    keys ``"sewer.q"`` (m3/s), ``"sewer.h"`` (link depth, m), ``"sewer.v"`` (m/s),
    ``"sewer.V_wet"`` (m3), ``"sewer.junction_depth"`` (m) and ``"sewer.surcharged"``
    (per junction, 0/1).

    EXTRAN surcharge: `crown_depth` is each junction's highest attached conduit
    top (`dynwave.c:137-153`). The closure writes the drivers ``"water.surcharged"`` (0/1,
    from the fed heads: `JunctionStorage.surcharged`), ``"water.surcharge_slope"`` (m2/s)
    and ``"water.head_last"`` (the fed junction heads, m); a surcharged junction holds
    instantaneous continuity, net inflow 0 (see `JunctionStorage` for the tie rule at the
    crown). ``"sewer.surcharged"`` reports SWMM's own test, depth above the crown
    (`dynwave.c:688`): a junction held on the band at its crown (the tie rule) reports 1
    although it is not holding dQ = 0 -- the solve's own flag is the driver
    ``"water.surcharged"``. Under SLOT no junction is ever surcharged (`dynwave.c:677`).

    Flooding (no ponding): `flood_depth` is SWMM's y_max, the full depth
    `full_depth` (MaxDepth extended to the crown, `link.c:440-463`) plus the junction's
    SurDepth (`Manhole.surcharge_depth`). The closure writes ``"water.net_inflow_last"``
    (the fed net inflow, m3/s), ``"water.flood_excess"`` (m3/s, `JunctionStorage`) and the
    lagged flag ``"water.flooded"`` (0/1, the excess positive); a flooded junction is held
    at y_max. ``"sewer.overflow"`` (m3/s per junction, >= 0) is SWMM's flooding loss,
    ``0.5 (N_start + N)`` at a flooded junction (`dynwave.c:673-674, 783-791`), 0 below
    SWMM's FUDGE flow (1e-4 cfs) and elsewhere. It is lost from the system and is not part
    of the carried ``"sewer.net_inflow"``, which, like SWMM's ``oldNetInflow``
    (`node.c:303`), is the conduit and lateral net inflow alone.

    The drop-end area switch (the tie rule; docs: "Drop-end area switch"). A conduit end
    with an offset gives its junction no surface area while UP_/DN_CRITICAL or dry, and a
    quarter of the conduit above that LOWER threshold (FUDGE, or min(yN, yC) where the flow
    leaves the conduit there); downstream, fasnh takes it back to 0 at max(yN, yC), the
    UPPER threshold, above which it is a quarter again (dwflow.c:357-411, 472-499;
    `conduit_terms` reports each end's threshold nearest its depth). A junction FILLING
    through such a jump can have no fixed point (lower area: the storage equation lands
    above the threshold; upper area: below it), and SWMM's trials do not converge there.
    Per junction the closure takes the attached drop end whose threshold head H* is nearest
    the fed head and, where H* is above the step-start head (the jump in the withdrawal is
    then upward), writes H* and the storage area just below and just above it; the
    junction's storage then switches within the pass (`JunctionStorage`). Where the fed net
    inflow lies inside the jump at H*, at the fraction theta of it, the junction is HELD on
    the threshold: that end is classified as SWMM classifies it AT the threshold (the upper
    side, its tests being strict), and at a lower threshold the conduit's other end gets
    ``(2 - theta)`` quarters of the conduit, sliding between the critical side's half and
    the upper side's quarter (`conduit_terms`' ``hold1/2``), so that its class does not
    chatter through the other end's area either. Only the nearest threshold of each
    junction is switched within the pass.

    Orifices, weirs and outlets (`net.orifices`, `.weirs`, `.outlets`): each
    kind present is a `StructureLink` edge kind whose ``"water.q"`` columns follow the
    pipes' (`link_names`, `edge_kinds`). The closure writes, per kind, SWMM's regulator flow
    at the fed heads ``"water.<kind>.flow_offset"`` (m3/s), its slope
    ``"water.<kind>.slope"`` (m2/s: SWMM's ``dqdh``, an outlet's a central difference),
    the fed head difference ``"water.<kind>.dp_fed"`` (m) and the flap-gate flags; an
    orifice's end areas join its junctions' storage areas (`_structure_terms`,
    dynwave.c:503-524), the regulators raise crowns and full depths (dynwave.c:137-153,
    link.c:425-463), an outfall on a regulator takes yN = yC = 0 (link.c:753-765), and every
    link's flow enters the junction net inflows.

    Tidal and time-varying outfalls (design C5): a TIDAL or TIMESERIES outfall follows the
    FIXED outfall's stage rule with the stage of the full-outfall driver ``"outfall_stage"``
    (m, one per outfall; only the TIDAL and TIMESERIES entries are read), the stage of the
    END of the step (`outfall_stage` builds it; `simulate_dynamic_wave` sets it each step).
    The initial heads use its value at t = 0.

    Quality (docs: "Quality on the dynamic wave"): with `quality_layer` set, the
    closure carries one more key, ``"sewer.quality_volume"`` (m3, per junction): the water
    the junction's transport control volume holds. It starts at half the summed wetted
    volume of the attached conduits plus a storage unit's own volume (`quality_capacity`)
    and then follows the junction's own continuity on the flows the transport sees,
    ``V_end = V_start + dt (N - overflow)`` with N the net inflow of the pass
    (``"sewer.net_inflow"``: lateral inflow and every link's flow) and the overflow SWMM's
    flooding loss (``"sewer.overflow"``), floored at `CAPACITY_FLOOR`. The driver
    ``"<quality_layer>.capacity"`` is V_end, so the amount form of the transport step,
    ``V_end x_end - V_start x_start = dt F``, holds the water balance exactly: a uniform
    inflow concentration stays uniform while the junctions fill and drain. (The geometric
    half-volumes do not: a conduit filling from its upstream end gives SWMM's surface area,
    and its water, to the upstream junction, dwflow.c:357-411, while the halves split it;
    on the tree's filling a uniform load of 1 came out between 0 and 5.) The driver
    ``"<quality_layer>.overflow_loss"`` (per junction, 0..1) is the fraction
    ``dt overflow / V_start`` of the step-start contents that floods out (`OverflowLoss`).
    The closure also writes the reporting keys ``"sewer.R_h"`` (hydraulic radius, m) and
    ``"sewer.d_m"`` (hydraulic mean depth A/T, m; 0 for a conduit flowing full) of one
    barrel at the link depth, on the exact circle (as the kinematic path does at SWMM's
    depth), which sulfide generation reads. It never writes the quality layer's flows:
    those are the head layer's ``"water.q"``.

    Air (docs: "Headspace air on the dynamic wave"): the closure always writes the
    headspace reporting keys of each conduit at its link depth y (`Link.newDepth =
    min(y_mid, y_full)`, one barrel's, capped at `AIR_DEPTH_RATIO_MAX` of the diameter D):
    ``"sewer.A_air"`` (m2, the TOTAL over the barrels, ``barrels (pi D2/4 - A(y))``),
    ``"sewer.D_h"`` (m, one barrel's air hydraulic diameter), ``"sewer.T"`` (m, the total free-
    surface width over the barrels, so that ``T / A_air``, which the drag reads, is one
    barrel's) and ``"sewer.V_air"`` (m3, total, ``A_air L``). They are the exact circle
    (`geometry.air_geometry`) at the link depth, on any `net.geometry`, as the kinematic path
    evaluates the circle at SWMM's depth with ``geometry="tabulated"``: the tabulated section
    is the water side's, the air has no SWMM reference. With `air` (the headspace layer
    ``"air"`` built on the conduits, `air.add_air_edges`) it also writes ``"rho_air_nodes"``
    (kg/m3, from the drivers ``"T_head"``, ``"T_amb"``) and
    ``"air_quality.capacity"`` (m3, per junction): the headspace volume of the junction's
    outgoing conduit plus the air of its shaft above the water, ``A_s (full depth - junction
    depth)`` for a junction (A_s `Manhole.surface_area` or `SHAFT_AREA_DEFAULT`) and the
    storage curve's volume between the water and the full depth for a storage unit; the shaft
    term needs `ground` on every manhole, as on the kinematic path.

    The arithmetic is SWMM's own, in feet (docs: "Units"); `layer` names the head layer.
    """

    integrates = True
    state_keys = ("sewer.junction_depth", "sewer.pipe_flow", "sewer.mid_area",
                  "sewer.net_inflow")

    def __init__(self, graph: Network, net: SewerNetwork, *, layer: str = "water",
                 quality_layer: str | None = None, air: bool = False,
                 air_quality_layer: str = "air_quality") -> None:
        self.net = graph
        self.layer = layer
        self.quality_layer = quality_layer
        self.air = air
        self.air_quality_layer = air_quality_layer
        if quality_layer is not None:
            self.state_keys = (*type(self).state_keys, "sewer.quality_volume")
        self.options = net.dynamic_wave
        self.notes: dict[str, str] = {}
        nodes = list(graph.nodes)
        self.pipe_names = [p.name for p in net.pipes]
        self.junction_names = [m.name for m in net.manholes]
        self.outfall_names = [o.name for o in net.outfalls]
        self.key_labels = {
            "sewer.junction_depth": list(self.junction_names),
            "sewer.pipe_flow": list(self.pipe_names),
            "sewer.mid_area": list(self.pipe_names),
            "sewer.net_inflow": list(self.junction_names),
            "sewer.quality_volume": list(self.junction_names),
        }
        # the headspace air's node ``"ambient"`` carries no invert (it is on no conduit)
        self.node_invert = torch.tensor(
            [float(graph.graph.nodes[n].get("invert", 0.0)) for n in nodes], dtype=F64)
        self.junction_idx = torch.tensor(
            [nodes.index(n) for n in self.junction_names], dtype=torch.long)
        self.outfall_idx = torch.tensor(
            [nodes.index(n) for n in self.outfall_names], dtype=torch.long)
        is_outfall = torch.zeros(len(nodes), dtype=torch.bool)
        is_outfall[self.outfall_idx] = True
        self.is_outfall = is_outfall
        self.u = torch.tensor([nodes.index(p.u) for p in net.pipes], dtype=torch.long)
        self.v = torch.tensor([nodes.index(p.v) for p in net.pipes], dtype=torch.long)
        self.diameter = torch.tensor([p.diameter for p in net.pipes], dtype=F64)
        self.roughness = torch.tensor([p.n for p in net.pipes], dtype=F64)
        self.slope = torch.tensor([p.slope for p in net.pipes], dtype=F64)
        self.length = torch.tensor([p.length for p in net.pipes], dtype=F64)
        #: per pipe: its `Xsect` (None for the circle of `diameter`); ``self.sections`` is None
        #: when every pipe is a plain circle (the tabulated circle's own path)
        sections = [p.xsect for p in net.pipes]
        self.sections = sections if any(x is not None for x in sections) else None
        #: per pipe: whether its section is the circle (exact reporting geometry applies)
        self.is_circle = torch.tensor([x is None or x.kind == "CIRCULAR" for x in sections],
                                      dtype=torch.bool)
        # a FILLED_CIRCULAR's offsets rise by its fill (link.c:1069-1074)
        fill = torch.tensor([x.params()["y_bot"] * LCF if x is not None
                             and x.kind == "FILLED_CIRCULAR" else 0.0 for x in sections],
                            dtype=F64)
        self.offset1 = torch.tensor([p.inlet_offset for p in net.pipes], dtype=F64) + fill
        self.offset2 = torch.tensor([p.outlet_offset for p in net.pipes], dtype=F64) + fill
        #: per pipe: SWMM's entry, exit and average loss coefficients (None: none set)
        losses = tuple(torch.tensor([getattr(p, f"loss_{k}") for p in net.pipes], dtype=F64)
                       for k in ("inlet", "outlet", "average"))
        self.losses = losses if any(bool((c != 0).any()) for c in losses) else None
        self._init_wet_losses(net)
        self.transect_factor = torch.tensor([p.transect_length_factor for p in net.pipes],
                                            dtype=F64)
        #: per pipe: its culvert code (0 for none)
        self.culverts = torch.tensor([p.culvert for p in net.pipes], dtype=torch.long)
        self.barrels = torch.tensor([float(p.barrels) for p in net.pipes], dtype=F64)
        #: per-pipe flow limit (m3/s, total over the barrels: SWMM limits one barrel's flow
        #: to MaxFlow, dwflow.c:270-273; +inf for none)
        self.max_flow = self.barrels * torch.tensor(
            [torch.inf if p.max_flow is None else p.max_flow for p in net.pipes], dtype=F64)
        #: per-pipe InitFlow (m3/s, total, positive from u to v)
        self.initial_flow = torch.tensor([p.initial_flow for p in net.pipes], dtype=F64)
        # link_setFlapGate (link.c:658-667): a conduit's own gate closes on flow against its
        # direction (``q * direction < 0``); an outfall's gate closes on flow that would
        # enter the conduit at the outfall's end.
        outfall_gated = {o.name for o in net.outfalls if o.flap_gate}
        own = [p.flap_gate for p in net.pipes]
        rev = [p.flap_gate_reversed for p in net.pipes]
        self.flap_block_reverse = torch.tensor(
            [(g and not r) or p.v in outfall_gated
             for p, g, r in zip(net.pipes, own, rev, strict=True)], dtype=torch.bool)
        self.flap_block_forward = torch.tensor(
            [(g and r) or p.u in outfall_gated
             for p, g, r in zip(net.pipes, own, rev, strict=True)], dtype=torch.bool)
        if not net.pipes:
            raise ValueError("DynamicWaveHydraulics: the network has no conduit; dynamic-wave "
                             "routing needs at least one")
        self._rebuild_geometry()
        #: the number of pipes: the first columns of ``"water.q"``
        self.n_pipes = len(net.pipes)
        # The flow regulators, one `_Structures` per kind present, their columns after the
        # pipes' in STRUCTURE_KINDS order (the layer's element order).
        route = (self.options.routing_step if self.options.lengthening_routing_step is None
                 else self.options.lengthening_routing_step)                # link.c:1721, 2149
        self.structures: dict[str, _Structures] = {
            kind: _build_structures(kind, net.structures[kind], nodes, self.node_invert,
                                    outfall_gated, route)
            for kind in STRUCTURE_KINDS if net.structures[kind]}
        # the link settings (regulators and pumps) are one vector in this order
        first = 0
        for kind, st in list(self.structures.items()):
            self.structures[kind] = replace(st, first=first)
            first += len(st.names)
        #: the number of regulators and pumps: the length of ``"sewer.setting"``
        self.n_settings = first
        #: the name of each regulator and pump, in ``"sewer.setting"`` order
        self.setting_names = [name for st in self.structures.values() for name in st.names]
        #: whether the network has pumps or control rules: then the closure carries each
        #: regulator's and pump's setting (``"sewer.setting"`` ...; `LinkControl` changes it)
        self.controlled = bool(net.pumps) or net.controls is not None
        self.controls = net.controls
        if self.controlled:
            self.state_keys = (*self.state_keys, *CONTROL_KEYS)
            self.key_labels.update({
                "sewer.setting": list(self.setting_names),
                "sewer.target_setting": list(self.setting_names),
                "sewer.time_last_set": list(self.setting_names),
            })
        #: the edge kinds of the ``"water"`` layer, in column order
        self.edge_kinds = ("pipe", *self.structures)
        #: every ``"water.q"`` column's link name: the pipes, then the regulators by kind
        self.link_names = list(self.pipe_names) + [
            name for s in self.structures.values() for name in s.names]
        self.edge_u = torch.cat([self.u, *(s.u for s in self.structures.values())])
        self.edge_v = torch.cat([self.v, *(s.v for s in self.structures.values())])

        # Each outfall's own link and the end it sits at (link.c:740-751: node2 first): a
        # conduit, or a regulator, whose outfall depth SWMM sets from yNorm = yCrit = 0
        # (link.c:753-765: only a conduit computes them).
        pipe_of, at_v, via, reg_offset, link_of = [], [], [], [], []
        for name in self.outfall_names:
            ends = [(i, p.v == name) for i, p in enumerate(net.pipes) if name in (p.u, p.v)]
            regs = [(s, j) for s in self.structures.values() for j in range(len(s.names))
                    if name in (nodes[int(s.u[j])], nodes[int(s.v[j])])]
            if len(ends) + len(regs) != 1:
                raise ValueError(
                    f"DynamicWaveHydraulics: outfall {name!r} is attached to "
                    f"{len(ends) + len(regs)} links; dynamic-wave routing takes exactly one")
            if regs:
                s, j = regs[0]
                if {nodes[int(s.u[j])], nodes[int(s.v[j])]} <= set(self.outfall_names):
                    raise ValueError(
                        f"DynamicWaveHydraulics: {s.kind} {s.names[j]!r} joins two outfalls; "
                        f"SWMM sets the depth of only one of them")
                downstream = nodes[int(s.v[j])] == name
                pipe_of.append(0)                    # a placeholder, masked by `via`
                link_of.append(self.link_names.index(s.names[j]))
                at_v.append(downstream)
                via.append(True)
                reg_offset.append(float(s.offset2[j] if downstream else s.offset1[j]))
                continue
            i, downstream = ends[0]
            pipe = net.pipes[i]
            if pipe.u in self.outfall_names and pipe.v in self.outfall_names:
                raise ValueError(
                    f"DynamicWaveHydraulics: conduit {pipe.name!r} joins two outfalls "
                    f"({pipe.u!r}, {pipe.v!r}); SWMM sets the depth of only one of them")
            pipe_of.append(i)
            link_of.append(i)
            at_v.append(downstream)
            via.append(False)
            reg_offset.append(0.0)
        # per outfall: its conduit's pipe index, internal only (0, a placeholder masked by
        # `outfall_on_structure`, for an outfall on a regulator)
        self._outfall_pipe = torch.tensor(pipe_of, dtype=torch.long)
        #: per outfall: its link's column in ``"water.q"`` (a conduit's, or a regulator's
        #: after the pipes'), positive away from the link's first node
        self.outfall_link = torch.tensor(link_of, dtype=torch.long)
        at_v_t = torch.tensor(at_v, dtype=torch.bool)
        #: per outfall: whether it is its link's downstream (second) node
        self.outfall_at_v = at_v_t
        #: per outfall: whether its link is a regulator (its depth then has yN = yC = 0)
        self.outfall_on_structure = torch.tensor(via, dtype=torch.bool)
        self.outfall_offset_ft = torch.where(
            self.outfall_on_structure, torch.tensor(reg_offset, dtype=F64),
            torch.where(at_v_t, self.offset2[self._outfall_pipe],
                        self.offset1[self._outfall_pipe]) / LCF)
        self.outfall_kind = torch.tensor(
            [_OUTFALL_CODES[o.kind] for o in net.outfalls], dtype=torch.long)
        self.outfall_stage_ft = torch.tensor(
            [(o.stage if o.stage is not None else o.invert) / LCF for o in net.outfalls],
            dtype=F64)
        #: per outfall: whether its stage is the ``"outfall_stage"`` driver's (TIDAL and
        #: TIMESERIES)
        self.outfall_staged = torch.tensor(
            [o.kind in _STAGED_KINDS for o in net.outfalls], dtype=torch.bool)
        self._outfalls = tuple(net.outfalls)
        self.min_area_ft2 = (
            DEFAULT_MIN_SURFACE_AREA_FT2 if self.options.min_surface_area is None
            else self.options.min_surface_area / LCF**2)            # dynwave.c:186-187
        # Each junction's crown: the highest conduit top over the conduit ends it carries,
        # crownElev - invertElev (dynwave.c:137-153); 0 for a junction with none.
        geo = self.geo
        crown = torch.zeros(len(nodes), dtype=F64)
        crown = crown.scatter_reduce(0, self.u, (geo.offset1 + geo.y_full).detach(), "amax")
        crown = crown.scatter_reduce(0, self.v, (geo.offset2 + geo.y_full).detach(), "amax")
        # the full-depth extension (link.c:440-463): both ends of a conduit, node 1 only of
        # a regulator other than a bottom orifice
        reach = crown.clone()
        for s in self.structures.values():
            crown = crown.scatter_reduce(0, s.u, s.offset1 + s.y_full, "amax")  # dynwave.c:148
            crown = crown.scatter_reduce(0, s.v, s.offset2 + s.y_full, "amax")  # dynwave.c:152
            top = torch.where(s.extends, s.offset1 + s.y_full, torch.zeros_like(s.y_full))
            reach = reach.scatter_reduce(0, s.u, top, "amax")                 # link.c:452
        self.crown_ft = crown[self.junction_idx]
        #: per-junction crown depth (m), in junction order
        self.crown_depth = self.crown_ft * LCF
        # The full depth: MaxDepth, extended to every attached conduit's top and to the
        # crown of a regulator leaving it (link.c:440-463) -- at a storage unit only when it
        # has a SurDepth (link.c:450, 458).
        max_depth = torch.tensor([0.0 if m.ground is None else m.ground - m.invert
                                  for m in net.manholes], dtype=F64)
        sur_depth = torch.tensor([m.surcharge_depth for m in net.manholes], dtype=F64)
        #: per junction: a SWMM storage unit (`Manhole.storage`)
        self.is_storage = torch.tensor([m.storage is not None for m in net.manholes],
                                       dtype=torch.bool)
        node_storage = torch.zeros(len(nodes), dtype=torch.bool)
        node_storage[self.junction_idx] = self.is_storage
        #: per node: a storage unit (a regulator's end area is zeroed there, dynwave.c:520)
        self.node_storage = node_storage
        extend = ~self.is_storage | (sur_depth > 0.0)                         # link.c:450
        #: per-junction full depth (m): MaxDepth extended to the crown
        self.full_depth = torch.where(
            extend, torch.maximum(max_depth, reach[self.junction_idx] * LCF), max_depth)
        #: per-junction flood depth y_max (m): the full depth plus SurDepth, above which
        #: the junction floods (no ponding, dynwave.c:746-748)
        self.flood_depth = self.full_depth + sur_depth
        #: per-junction EXTRAN surcharge depth (m): the crown of a junction, the full depth
        #: of a storage unit (dynwave.c:683-690); `surchargeable` is False where SWMM never
        #: surcharges (a storage unit without SurDepth, a junction with no crown)
        self.surcharge_at = torch.where(self.is_storage, self.full_depth, self.crown_depth)
        self.surchargeable = torch.where(self.is_storage, sur_depth > 0.0,
                                         self.crown_depth > 0.0)
        #: the storage units' area and volume curves (`storage.StorageCurves`), per junction
        self.storage_curves = (
            StorageCurves([m.storage for m in net.manholes], self.full_depth,
                          names=self.junction_names)
            if bool(self.is_storage.any()) else None)
        if self._storage_inputs is not None:
            self.storage_losses = StorageLosses(
                self._storage_inputs, self.storage_curves,
                us_units=self.climate is not None and self.climate.us_units)
        if air:
            outgoing = {p.u: i for i, p in enumerate(net.pipes)}
            #: per junction: its outgoing conduit's position among the pipes (air.py has
            #: refused every other topology)
            self.out_pipe = torch.tensor([outgoing[n] for n in self.junction_names],
                                         dtype=torch.long)
            has_ground = all(m.ground is not None for m in net.manholes)
            #: per junction: the shaft's plan area (m2), 0 where no shaft air is counted
            self.shaft_area = torch.tensor(
                [SHAFT_AREA_DEFAULT if m.surface_area is None else m.surface_area
                 for m in net.manholes], dtype=F64) if has_ground else None
        #: per-junction InitDepth (m)
        self.initial_junction_depth = torch.tensor(
            [m.initial_depth for m in net.manholes], dtype=F64)
        too_deep = self.initial_junction_depth > self.flood_depth                 # node.c:216
        if bool(too_deep.any()):
            bad = [self.junction_names[i] for i in too_deep.nonzero().flatten().tolist()]
            raise ValueError(
                f"DynamicWaveHydraulics: junction(s) {bad} start deeper (initial_depth) than "
                f"their full depth plus surcharge depth (SWMM: initial depth exceeds max. "
                f"depth)")
        jinv = self.node_invert[self.junction_idx]
        # Ponding (ALLOW_PONDING, a junction with a positive Aponded; dynwave.c:661): such a
        # junction's flood depth is its full depth, SurDepth plays no part (dynwave.c:748), and
        # it surcharges only between its crown and its full depth (dynwave.c:679-680, 690).
        pond_ft2 = torch.tensor([m.ponded_area for m in net.manholes], dtype=F64) / LCF**2
        #: per junction: whether it ponds (`DynamicWaveOptions.allow_ponding`, a positive
        #: `Manhole.ponded_area`)
        self.can_pond = (pond_ft2 > 0.0) & ~self.is_storage & self.options.allow_ponding
        self.pond_area_ft2 = torch.where(self.can_pond, pond_ft2, torch.zeros_like(pond_ft2))
        top = jinv + torch.where(self.can_pond, self.full_depth + FUDGE * LCF, self.flood_depth)
        if bool(self.can_pond.any()) and (quality_layer is not None or air):
            raise ValueError(
                "DynamicWaveHydraulics: ponding (allow_ponding with a ponded area) is not "
                "modelled together with water quality or air; the ponded water would have to "
                "stay in the junction's transport volume")
        pond_head = torch.where(self.can_pond, jinv + self.full_depth,
                                torch.full_like(jinv, torch.inf))
        #: the junctions' `JunctionStorage` (the model's node source; `junction_residual`)
        self.storage = JunctionStorage(
            self.junction_idx, jinv, bed_area=self.min_area_ft2 * _AREA_SI,
            crown=jinv + self.surcharge_at if self.options.surcharge_method == "extran"
            else None, top=top,
            surchargeable=self.surchargeable & ~(self.can_pond
                                                 & ~(self.full_depth > self.crown_depth)),
            pond_head=pond_head)
        if self.controlled:
            self._init_controls(net)

    def _init_wet_losses(self, net: SewerNetwork) -> None:
        """The evaporation and seepage of the conduits and the evaporation and exfiltration of
        the storage units (`losses.py`): which conduits lose water, which storage units do,
        and the climate source. Sets `pipe_losses`, `storage_losses` (``None`` when there
        are none) and `climate`; a network with neither is untouched by any of it."""
        source = net.climate_source
        self.climate = source
        self.pipe_losses = False
        self.storage_losses: StorageLosses | None = None
        seep = torch.tensor([p.seepage for p in net.pipes], dtype=F64)
        evaporates = torch.tensor(
            [source is not None and evaporation_can_be_positive(source)
             and p.xsect is not None and p.xsect.open for p in net.pipes], dtype=torch.bool)
        self._seep_rate = seep / LCF if bool((seep > 0.0).any()) else None      # ft/s
        self._loss_evap = evaporates if bool(evaporates.any()) else None
        self.pipe_losses = self._seep_rate is not None or self._loss_evap is not None
        if self.pipe_losses:
            self.state_keys = (*self.state_keys, "sewer.pipe_depth")
            self.key_labels["sewer.pipe_depth"] = list(self.pipe_names)
        if (any(m.exfiltration is not None for m in net.manholes)
                or (source is not None and any(m.evaporation_factor != 0.0
                                               for m in net.manholes))):
            self._storage_inputs = list(net.manholes)
        else:
            self._storage_inputs = None
        #: whether the closure needs the climate drivers (`losses.EVAP_DRIVER` ...)
        self.has_wet_losses = self.pipe_losses or self._storage_inputs is not None

    def _init_controls(self, net: SewerNetwork) -> None:
        """The data of the link settings and pumps: the initial settings, each orifice's
        opening rate, the pumps' startup and shutoff depths and the node facts a pump's flow
        reads (`pumps.PumpStation`, the junctions' wet-well volumes)."""
        n_nodes = self.node_invert.shape[0]
        jpos = torch.full((n_nodes,), -1, dtype=torch.long)
        jpos[self.junction_idx] = torch.arange(self.junction_idx.shape[0])
        #: per node: its position among the junctions (-1 for an outfall)
        self.junction_pos = jpos
        nodes = list(self.net.nodes)
        #: node name -> index over all nodes; link name -> its ``"water.q"`` column; link name
        #: -> SWMM's link kind (``"conduit"``, ``"orifice"``, ...); regulator or pump name ->
        #: its position in ``"sewer.setting"``; per column, SWMM's ``Link.direction`` (-1 for
        #: a conduit the reader reversed): what the control rules read link flows through
        self.node_index = {name: i for i, name in enumerate(nodes)}
        self.link_column = {name: i for i, name in enumerate(self.link_names)}
        self.link_direction = [-1.0 if p.flap_gate_reversed else 1.0 for p in net.pipes] + [
            1.0] * (len(self.link_names) - len(net.pipes))
        self.link_kind = {p.name: "conduit" for p in net.pipes}
        for kind, st in self.structures.items():
            self.link_kind.update({name: kind for name in st.names})
        self.setting_index = {name: i for i, name in enumerate(self.setting_names)}
        initial = torch.ones(self.n_settings, dtype=F64)
        for kind, st in self.structures.items():
            if kind == "pump":
                on = torch.tensor([p.initial_on for p in net.pumps], dtype=F64)
                initial[st.first:st.first + len(st.names)] = on             # link.c:1542
        #: the settings a run starts with (pumps at their Status, the rest fully open)
        self.initial_setting = initial
        orifices = self.structures.get("orifice")
        #: per orifice: its opening rate (s per unit of setting; 0: instantaneous), SWMM's
        #: ``orate`` (CloseTime in hours, link.c:369)
        self.orifice_rate = (torch.tensor([o.close_time * 3600.0 for o in net.orifices],
                                          dtype=F64) if orifices is not None
                             else torch.zeros(0, dtype=F64))
        pumps = self.structures.get("pump")
        #: the junctions (positions) an ideal pump draws from
        self.ideal_inlet = torch.zeros(0, dtype=torch.long)
        if pumps is None:
            return
        n_pipes, n_edges = self.n_pipes, self.edge_u.shape[0]
        column = torch.arange(n_edges)
        order = torch.zeros(n_edges, dtype=torch.long)
        for st in self.structures.values():
            lo = n_pipes + st.first
            # SWMM's link index: the file's order, the column order where it gives none
            order[lo:lo + len(st.names)] = st.order * (n_edges + 1) + column[lo:lo + len(st.names)]
        pump_cols = n_pipes + pumps.first + torch.arange(len(pumps.names))
        orifice_cols = (n_pipes + orifices.first + torch.arange(len(orifices.names))
                        if orifices is not None else torch.zeros(0, dtype=torch.long))
        self.pump_station = PumpStation(pumps.u, pump_cols, self.edge_u, self.edge_v, n_pipes,
                                        order, orifice_cols)
        #: per pump: the inlet node's position among the junctions
        self.pump_inlet = jpos[pumps.u]
        #: per pump: the inlet node is a storage unit
        self.pump_storage = self.is_storage[self.pump_inlet]
        # A PUMP1's x values are a wet-well volume: a plain junction inlet is given that
        # volume at its full depth, linear in the depth (link.c:1521-1527, node.c:251, 376)
        wells = torch.zeros(self.junction_idx.shape[0], dtype=F64)
        for p, j in zip(net.pumps, self.pump_inlet.tolist(), strict=True):
            if p.kind == "type1" and not bool(self.is_storage[j]):
                wells[j] = max(float(wells[j]), float(p.curve[0][-1]) / QCF)
        #: per junction: the full volume (ft3) a PUMP1 gives a non-storage inlet, else 0
        self.well_volume_ft3 = wells
        full = wells[self.pump_inlet]
        if self.storage_curves is not None:
            stored = self.storage_curves.volume(self.full_depth)[self.pump_inlet] / LCF**3
            full = torch.where(self.pump_storage, stored, full)
        #: per pump: its inlet node's full volume (ft3; node.c:251)
        self.pump_full_volume_ft3 = full
        ideal = [i for i, p in enumerate(net.pumps) if p.kind == "ideal"]
        self.ideal_inlet = self.pump_inlet[ideal]
        #: per pump: the startup and shutoff depths of its inlet (ft; 0: none)
        self.pump_on_ft = torch.tensor([p.startup_depth for p in net.pumps], dtype=F64) / LCF
        self.pump_off_ft = torch.tensor([p.shutoff_depth for p in net.pumps], dtype=F64) / LCF

    def check_pump_inlets(self, state: Mapping) -> None:
        """Refuse, BY NAME, an ideal pump whose inlet junction floods. SWMM's ideal pump takes
        the node's inflow PLUS its overflow (`pump_getInflow`, link.c:1572-1573), a coupling
        of the flooding loss to the pump this implementation does not model (the pump takes
        the inflow alone), so the run stops rather than report a different flood."""
        if self.ideal_inlet.numel() == 0:
            return
        depth = state["sewer.junction_depth"][..., self.ideal_inlet]
        if bool((depth >= self.flood_depth[self.ideal_inlet] - 1e-9).any()):
            bad = sorted({self.junction_names[int(j)] for j in
                          self.ideal_inlet[(depth >= self.flood_depth[self.ideal_inlet]
                                            - 1e-9).reshape(-1, self.ideal_inlet.numel())
                                           .any(0)]})
            raise RuntimeError(
                f"DynamicWaveHydraulics: the inlet junction(s) {bad} of an ideal pump flood; "
                f"SWMM's ideal pump also pumps the node's overflow, which is not modelled")

    def _node_volume_ft3(self, heads: Tensor) -> Tensor:
        """Each junction's volume (ft3) at full-node `heads` (m), SWMM's ``Node.newVolume``: a
        storage unit's curve, a plain junction's linear share of its PUMP1 wet-well volume
        (`node_getVolume`, node.c:362-379), else 0."""
        depth_m = torch.clamp(heads[..., self.junction_idx] - self.node_invert[self.junction_idx],
                              min=0.0)
        safe = torch.where(self.full_depth > 0.0, self.full_depth, torch.ones_like(depth_m))
        well = torch.where(depth_m >= self.flood_depth, self.well_volume_ft3,
                           self.well_volume_ft3 * depth_m / safe)       # dynwave.c:783
        well = torch.where(self.full_depth > 0.0, well, torch.zeros_like(well))
        if self.storage_curves is None:
            return well
        return torch.where(self.is_storage, self._storage_volume(heads) / LCF**3, well)

    # ------------------------------------------------------------------ geometry
    def _rebuild_geometry(self) -> None:
        opts = self.options
        factor = None
        if opts.lengthening_step > 0.0:                                       # link.c:1105-1108
            route = (opts.routing_step if opts.lengthening_routing_step is None
                     else opts.lengthening_routing_step)
            step = min(route, opts.lengthening_step)                          # link.c:1248
            factor = lengthening_factor(self.diameter, self.roughness, self.slope,
                                        self.length, step, sections=self.sections,
                                        force_main_equation=opts.force_main_equation)
        self.geo = ConduitGeometry.from_si(
            self.diameter, self.roughness, self.slope, self.length,
            self.node_invert[self.u], self.node_invert[self.v], self.offset1, self.offset2,
            self.is_outfall[self.u], self.is_outfall[self.v], barrels=self.barrels,
            length_factor=factor, sections=self.sections,
            force_main_equation=opts.force_main_equation, losses=self.losses,
            transect_factor=self.transect_factor, culverts=self.culverts)
        if self.pipe_losses:                   # link.c:1359-1378, `losses.py`
            fields: dict = {}
            if self._loss_evap is not None:
                fields["loss_evap"] = self._loss_evap
            if self._seep_rate is not None:
                geo = self.geo
                kinds = [] if geo.xsect is None else geo.xsect.kinds
                rect = torch.tensor([k == "RECT_CLOSED" for k in kinds], dtype=torch.bool)
                fields.update(seep_rate=self._seep_rate, seep_depth=seepage_depth_limit(geo),
                              seep_rect=rect if bool(rect.any()) else None)
            self.geo = replace(self.geo, **fields)

    def set_roughness(self, i: int, value) -> None:
        """Set Manning's n (s m^(-1/3)) of pipe `i` to `value`, which may be a tensor that
        requires grad; the geometry is rebuilt so the gradient reaches it. The other pipes'
        n are detached, so only the latest call's `value` carries a gradient."""
        value = torch.as_tensor(value, dtype=F64).reshape(1)
        self.roughness = torch.cat([self.roughness[:i].detach(), value,
                                    self.roughness[i + 1:].detach()])
        self._rebuild_geometry()

    # ------------------------------------------------------------------- helpers
    def _inflow(self, drivers: Mapping) -> Tensor:
        try:
            inflow = drivers["inflow"]
        except KeyError as exc:
            raise KeyError("DynamicWaveHydraulics: driver 'inflow' (m3/s, full-node lateral "
                           "inflow) is required") from exc
        if not bool(torch.isfinite(inflow).all()):
            bad = ~torch.isfinite(inflow).reshape(-1, inflow.shape[-1]).all(0)
            names = [self.net.nodes[i] for i in bad.nonzero().flatten().tolist()]
            raise ValueError(f"DynamicWaveHydraulics: non-finite inflow at {names}")
        at_outfalls = inflow[..., self.outfall_idx]
        if bool((at_outfalls != 0).any()):
            bad = (at_outfalls != 0).reshape(-1, len(self.outfall_names)).any(0)
            names = [self.outfall_names[i] for i in bad.nonzero().flatten().tolist()]
            raise ValueError(
                f"DynamicWaveHydraulics: inflow at outfall(s) {names} must be 0; an outfall "
                f"head is set by its conduit, not by a node balance")
        return inflow

    def _loss_inputs(self, state: Mapping, drivers: Mapping | None) -> LossInputs | None:
        """What the conduits' evaporation and seepage read (`conduit_terms`): the climate
        drivers (`losses.EVAP_DRIVER`, `HYDCON_DRIVER`) and the carried link depths; ``None``
        for a network without conduit losses or a query given no drivers."""
        if not self.pipe_losses or drivers is None:
            return None
        for key in (EVAP_DRIVER, HYDCON_DRIVER):
            if key not in drivers:
                raise KeyError(
                    f"DynamicWaveHydraulics: driver {key!r} is required: the network has "
                    f"conduits that evaporate or seep (`simulate_dynamic_wave` sets it each "
                    f"step from the file's climate)")
        evap = torch.as_tensor(drivers[EVAP_DRIVER], dtype=F64) / LCF
        hydcon = torch.as_tensor(drivers[HYDCON_DRIVER], dtype=F64)
        return LossInputs(evap, hydcon, state["sewer.pipe_depth"] / LCF)

    def _storage_loss(self, drivers: Mapping | None) -> Tensor | None:
        """The storage units' loss (m3/s) at every node from the driver
        `losses.STORAGE_DRIVER` (one per junction); ``None`` for a network without."""
        if self.storage_losses is None or drivers is None:
            return None
        if STORAGE_DRIVER not in drivers:
            raise KeyError(
                f"DynamicWaveHydraulics: driver {STORAGE_DRIVER!r} (m3/s per junction) is "
                f"required: the network has storage units that evaporate or exfiltrate "
                f"(`simulate_dynamic_wave` sets it each step; see `storage_loss`)")
        loss = torch.as_tensor(drivers[STORAGE_DRIVER], dtype=F64)
        full = torch.zeros(loss.shape[:-1] + self.node_invert.shape, dtype=F64)
        return full.index_copy(-1, self.junction_idx, loss)

    def _conduit_loss_nodes(self, terms: ConduitTerms) -> Tensor | None:
        """The rate (m3/s) the conduits' evaporation and seepage take from every node
        (`dynwave.c::updateNodeFlows`, dynwave.c:555-572): the conduit's rate times its
        barrels, half from each end, or all from the end that is not an outfall; ``None``
        where the conduits lose nothing."""
        if terms.loss_rate is None:
            return None
        rate = terms.loss_rate * QCF
        u_out, v_out = self.is_outfall[self.u], self.is_outfall[self.v]
        half = torch.where(u_out | v_out, 1.0, 0.5)
        at_u = torch.where(u_out, 0.0, half)
        at_v = torch.where(v_out, 0.0, half)
        full = torch.zeros(rate.shape[:-1] + self.node_invert.shape, dtype=F64)
        return full.index_add(-1, self.u, rate * at_u).index_add(-1, self.v, rate * at_v)

    def reset_losses(self) -> None:
        """Return the storage units' Green-Ampt exfiltration states to the start of a run
        (`simulate_dynamic_wave` does at its start; a continuation call restarts them)."""
        if self.storage_losses is not None:
            self.storage_losses.reset()

    def storage_loss(self, state: Mapping, dt: float, climate) -> Tensor | None:
        """The loss rate (m3/s, one per junction) of the storage units that evaporate or
        exfiltrate over a routing step of `dt` s that starts at `state` (``None``: the network
        has none), for the driver `losses.STORAGE_DRIVER`. `climate` is a `losses.ClimateValues`.
        SWMM computes it once per routing step from the step-start depth and volume
        (routing.c:363-365, `StorageLosses.rates`); the rate carries gradients through that
        depth. A unit with a positive IMDmax advances its Green-Ampt state by this call, so call
        it once per step, in order (`simulate_dynamic_wave` does)."""
        if self.storage_losses is None:
            return None
        depth_ft = state["sewer.junction_depth"] / LCF
        return self.storage_losses.rates(depth_ft, dt, climate) * QCF

    def _terms(self, state: Mapping, dt: float,
               hold: tuple[Tensor, Tensor] | None = None,
               flow_depths: tuple[Tensor, Tensor] | None = None,
               drivers: Mapping | None = None) -> ConduitTerms:
        heads = state[f"{self.layer}.phi"]
        q = state[f"{self.layer}.q"][..., :self.n_pipes]
        # SWMM's node depth is never negative (dynwave.c:743); the stiff bed lets a dry
        # junction's head sit a hair below its invert.
        depth_ft = torch.clamp(heads - self.node_invert, min=0.0) / LCF
        return conduit_terms(
            self.geo, depth_ft[..., self.u], depth_ft[..., self.v], q / QCF,
            state["sewer.pipe_flow"] / QCF, state["sewer.mid_area"] / _AREA_SI, dt,
            damping=self.options.inertial_damping,
            normal_flow_limited=self.options.normal_flow_limited,
            surcharge_method=self.options.surcharge_method,
            hold1=None if hold is None else hold[0], hold2=None if hold is None else hold[1],
            flow_depths=flow_depths, loss=self._loss_inputs(state, drivers))

    def _depth_constants(self) -> tuple[Tensor, ...]:
        """The section constants of `_flow_depths`' joint evaluation: every pipe's, then
        each outfall pipe's again, ``(y_full, a_full, s_full, beta, critical_flows)``; cached
        on the geometry (rebuilt with it)."""
        cached = getattr(self, "_depth_cache", None)
        if cached is not None and cached[0] is self.geo:
            return cached[1]
        geo, op = self.geo, self._outfall_pipe
        if geo.xsect is None:
            consts = tuple(torch.cat([x, x[op]]) for x in (
                geo.y_full, geo.a_full, geo.s_full, geo.beta, geo.critical_flows))
        else:
            # Only a conduit with an offset reads its yN, yC for the flow class (zr > 0,
            # dwflow.c:327-411), and an outfall conduit's are the outfall head's. So the
            # sections of those conduits, then of each outfall pipe again (its bumped flow),
            # and their betas.
            wanted = (self.offset1 > 0.0) | (self.offset2 > 0.0)
            wanted[op] = True
            need = wanted.nonzero().flatten()
            pick = torch.cat([need, op])
            consts = (geo.xsect.take(pick), geo.beta[pick], need)
        self._depth_cache = (geo, consts)
        return consts

    def _flow_depths(self, q: Tensor) -> dict[str, Tensor]:
        """SWMM's normal and critical depths (ft, one barrel's, `normal_depth_ft`,
        `critical_depth`) for one pass, in one evaluation: ``"normal"``/``"critical"`` at
        each pipe's fed flow `q` (m3/s, total; what `conduit_terms` classifies on, and the
        outfall depth's yN/yC, link.c:759-764), ``"normal_bumped"``/``"critical_bumped"``
        at each outfall pipe's flow bumped by ``"step"`` (m3/s; `_switch_slope`'s forward
        difference)."""
        n, op = self.n_pipes, self._outfall_pipe
        q_abs = q.abs()
        step = 1e-6 * torch.clamp(q_abs[..., op], min=1e-3)
        per_barrel = (q / QCF / self.geo.barrels).abs()                     # dwflow.c:102
        bumped = ((q_abs[..., op] + step) / QCF / self.barrels[op]).abs()    # link.c:759
        flows = torch.cat([per_barrel, bumped], -1)
        if self.geo.xsect is None:
            y_full, a_full, s_full, beta, stations = self._depth_constants()
            y_norm = normal_depth_ft(flows, y_full, a_full, s_full, beta)
            y_crit = critical_depth(flows, y_full, a_full, stations)
        else:
            xsect, beta, need = self._depth_constants()
            pick = torch.cat([need, torch.arange(n, n + op.shape[0])])
            f_pick = flows[..., pick]
            y_norm = torch.zeros_like(flows).index_copy(-1, pick, xsect.normal_depth(f_pick, beta))
            y_crit = torch.zeros_like(flows).index_copy(-1, pick, xsect.critical_depth(f_pick))
        return {"normal": y_norm[..., :n], "critical": y_crit[..., :n],
                "normal_bumped": y_norm[..., n:], "critical_bumped": y_crit[..., n:],
                "step": step}

    def _junction_depth_ft(self, heads: Tensor) -> Tensor:
        """The junction depths (ft, >= 0) at full-node `heads` (m), SWMM's (dynwave.c:743)."""
        jidx = self.junction_idx
        return torch.clamp(heads[..., jidx] - self.node_invert[jidx], min=0.0) / LCF

    def _storage_area_ft2(self, heads: Tensor) -> Tensor:
        """Each junction's own storage area (ft2; 0 at a plain junction) at `heads` (m):
        `node_getSurfArea` (node.c:383-396), the curve in m and m2 (node.c:997-1017)."""
        depth_ft = self._junction_depth_ft(heads)
        if self.storage_curves is None:
            return torch.zeros_like(depth_ft)
        return self.storage_curves.area(depth_ft * LCF) / LCF**2              # node.c:1017

    def _storage_volume(self, heads: Tensor) -> Tensor:
        """Each junction's stored volume (m3; 0 at a plain junction, as SWMM's junction
        volume, node.c:362-379) at `heads` (m): `node_getVolume`, capped at the full volume
        at or above the full depth (node.c:943-945)."""
        depth_ft = self._junction_depth_ft(heads)
        if self.storage_curves is None:
            return torch.zeros_like(depth_ft)
        return self.storage_curves.volume(depth_ft * LCF)

    def _drop_switch(self, state: Mapping, terms: ConduitTerms, net_inflow: Tensor,
                     dt: float, storage_ft2: Tensor) -> dict[str, Tensor]:
        """The drop-end area switch at each junction (`JunctionStorage`, the tie rule), from
        the fed state and one pass's `terms`. Per junction: the threshold head ``"head"`` H*
        (m) of the attached drop end whose threshold is nearest the fed head; the storage
        areas just ``"below"`` and just ``"above"`` it (m2, SWMM's area floor applied);
        ``"applies"``, the threshold lies above the step-start head (the jump in the
        withdrawal is then upward), a storage unit's own area `storage_ft2` (ft2) being on
        both sides; ``"held"``, the fed net inflow `net_inflow` (m3/s) lies
        inside that jump at H*, the junction being on the threshold. Per pipe end:
        ``"hold1"``/``"hold2"``, at the chosen end of a held junction the fraction theta of
        the jump at which the net inflow lies (0 at its bottom, 1 at its top), -1 elsewhere
        (`conduit_terms`)."""
        geo = self.geo
        nodes = torch.cat([self.u, self.v])
        n = self.node_invert.shape[-1]
        heads = state[f"{self.layer}.phi"]
        z = torch.cat([geo.invert1 + geo.offset1, geo.invert2 + geo.offset2])
        depth = torch.cat([terms.drop_depth1, terms.drop_depth2], -1)
        finite = torch.isfinite(depth) & ~self.is_outfall[nodes]
        jidx = self.junction_idx
        if not bool(finite.any()):
            # no drop end has a threshold: no junction switches (the general path's values)
            batch = depth.shape[:-1]
            surf = torch.zeros(batch + (n,), dtype=F64)
            surf = surf.index_add(-1, self.u, terms.surface_area1)
            surf = surf.index_add(-1, self.v, terms.surface_area2)[..., jidx] + storage_ft2
            area = torch.clamp(surf, min=self.min_area_ft2) * _AREA_SI
            start = (self.node_invert[jidx] + state["sewer.junction_depth"]).expand(area.shape)
            no = torch.zeros(area.shape, dtype=torch.bool)
            hold = torch.full(depth.shape, -1.0, dtype=F64)
            p = self.u.shape[-1]
            return {"head": start, "below": area, "above": area, "applies": no, "held": no,
                    "hold1": hold[..., :p], "hold2": hold[..., p:]}
        head_thr = torch.where(finite, (z + torch.where(finite, depth, torch.zeros_like(depth)))
                               * LCF, torch.zeros_like(depth))
        batch = head_thr.shape[:-1]
        index = nodes.expand(batch + nodes.shape)
        inf = torch.full(batch + (n,), torch.inf, dtype=F64)
        dist = torch.where(finite, (heads[..., nodes] - head_thr).abs(),
                           torch.full_like(head_thr, torch.inf))
        nearest = inf.scatter_reduce(-1, index, dist, "amin")
        chosen = finite & (dist == nearest[..., nodes])
        # an exact distance tie (two ends with one threshold head) takes the lowest end index
        order = torch.arange(nodes.shape[-1]).expand(index.shape)
        ranked = torch.where(chosen, order, torch.full_like(order, nodes.shape[-1]))
        first = torch.full(batch + (n,), nodes.shape[-1], dtype=torch.long)
        first = first.scatter_reduce(-1, index, ranked, "amin")
        chosen = chosen & (order == first[..., nodes])

        def pick(values: Tensor, empty: float) -> Tensor:
            base = torch.full(batch + (n,), empty, dtype=F64)
            masked = torch.where(chosen, values, torch.full_like(values, empty))
            return base.scatter_reduce(-1, index, masked, "amax")

        jidx = self.junction_idx
        h_star = pick(head_thr, -torch.inf)[..., jidx]
        jump = pick(torch.cat([terms.drop_area1, terms.drop_area2], -1), 0.0)[..., jidx]
        own = pick(torch.cat([terms.surface_area1, terms.surface_area2], -1), 0.0)[..., jidx]
        above = pick(torch.cat([terms.drop_above1, terms.drop_above2], -1).to(F64),
                     0.0)[..., jidx] > 0.5
        surf = torch.zeros(batch + (n,), dtype=F64)
        surf = surf.index_add(-1, self.u, terms.surface_area1)
        surf = surf.index_add(-1, self.v, terms.surface_area2)[..., jidx] + storage_ft2
        rest = surf - own
        lo = torch.clamp(torch.where(above, rest, surf), min=self.min_area_ft2) * _AREA_SI
        hi = torch.clamp(torch.where(above, surf, rest + jump), min=self.min_area_ft2) * _AREA_SI
        has = torch.isfinite(h_star)
        head_start = self.node_invert[jidx] + state["sewer.junction_depth"]
        safe_star = torch.where(has, h_star, head_start)
        applies = has & (safe_star > head_start) & (hi > lo)
        rate2 = 2.0 / dt
        st_lo = rate2 * lo * (safe_star - head_start) - state["sewer.net_inflow"]
        st_hi = rate2 * hi * (safe_star - head_start) - state["sewer.net_inflow"]
        held = applies & (st_lo < net_inflow) & (net_inflow < st_hi)
        gap = torch.where(held, st_hi - st_lo, torch.ones_like(st_hi))
        theta = torch.where(held, (net_inflow - st_lo) / gap, torch.full_like(gap, -1.0))
        theta_node = torch.full(batch + (n,), -1.0, dtype=F64)
        theta_node[..., jidx] = theta
        hold = torch.where(chosen, theta_node[..., nodes], torch.full_like(head_thr, -1.0))
        p = self.u.shape[-1]
        return {"head": safe_star, "below": lo, "above": hi, "applies": applies,
                "held": held, "hold1": hold[..., :p], "hold2": hold[..., p:]}

    def _structure_terms(self, heads: Tensor, state: Mapping | None = None,
                         inflow: Tensor | None = None, dt: float | None = None,
                         terms: ConduitTerms | None = None,
                         storage_ft2: Tensor | None = None) -> tuple[dict, Tensor]:
        """The regulators and pumps at full-node `heads` (m): per kind ``(StructureFlow,
        slope)`` (ft units: SWMM's flow and its ``dqdh``, an outlet's a central difference,
        `structures.outlet_dqdh`), and every node's regulator surface area (ft2):
        `findNonConduitSurfArea` (dynwave.c:503-524), an orifice's ``surfArea / 2`` at each
        end (weirs, outlets and pumps none), zeroed at an UP_CRITICAL first end, a DN_CRITICAL
        second end and at a storage unit.

        With link settings (`state` carries ``"sewer.setting"``) each device is at its
        setting, a closed orifice keeping the end areas it last had
        (``"sewer.orifice_hold"``: SWMM's ``Orifice.surfArea`` and flow class are not touched
        while ``link_getInflow`` returns 0, link.c:550). A pump (`pumps.pump_curve_flow`)
        also reads the step start in `state`, the lateral `inflow` (m3/s), the step `dt`, the
        pass's conduit `terms` and the junctions' own `storage_ft2`: its limit by the water
        at its inlet (`pumps.mod_pump_flow`) and an ideal pump's flow are those of the
        inlet's partial sums in the trial (`pumps.PumpStation`)."""
        inv = self.node_invert
        h_ft = (inv + torch.clamp(heads - inv, min=0.0)) / LCF               # dynwave.c:743
        z_ft = inv / LCF
        area = torch.zeros(heads.shape[:-1] + inv.shape[-1:], dtype=F64)
        found: dict = {}
        setting = None if state is None else state.get("sewer.setting")
        ends = None
        for kind, s in self.structures.items():
            h1, h2 = h_ft[..., s.u], h_ft[..., s.v]
            z1, z2 = z_ft[s.u], z_ft[s.v]
            crest = z1 + s.offset1                                            # link.c:1865
            gates = (s.block_forward, s.block_reverse)
            k = None if setting is None else setting[..., s.first:s.first + len(s.names)]
            if kind == "orifice":
                params = s.params if k is None else s.params.at_setting(k)
                flow = orifice_flow(params, h1, h2, z1, z2, crest, *gates)
                slope = flow.dqdh
                half = flow.surf_area / 2.0                                   # dynwave.c:511
                end1 = torch.where((flow.flow_class == UP_CRITICAL) | self.node_storage[s.u],
                                   torch.zeros_like(half), half)              # dynwave.c:518-520
                end2 = torch.where((flow.flow_class == DN_CRITICAL) | self.node_storage[s.v],
                                   torch.zeros_like(half), half)              # dynwave.c:521-523
                if k is not None:
                    hold = state["sewer.orifice_hold"] / _AREA_SI
                    n = len(s.names)
                    shut = k <= 0.0
                    end1 = torch.where(shut, hold[..., :n], end1)
                    end2 = torch.where(shut, hold[..., n:], end2)
                area = area.index_add(-1, s.u, end1).index_add(-1, s.v, end2)
                ends = (end1, end2)
            elif kind == "weir":
                params = s.params if k is None else s.params.at_setting(k)
                flow = weir_flow(params, h1, h2, z1, z2, crest, *gates)
                slope = flow.dqdh
            elif kind == "outlet":
                flow = outlet_flow(s.params, h1, h2, z1, z2, crest, *gates, setting=k)
                slope = outlet_dqdh(s.params, h1, h2, z1, z2, crest, *gates, setting=k)
            else:
                flow, slope = self._pump_flow(s, k, heads, h1, h2, state, inflow, dt, terms,
                                              storage_ft2, ends)
            found[kind] = (flow, slope)
        return found, area

    def _pump_flow(self, s: _Structures, k: Tensor, heads: Tensor, h1: Tensor, h2: Tensor,
                   state: Mapping, inflow: Tensor, dt: float, terms: ConduitTerms,
                   storage_ft2: Tensor, ends: tuple | None):
        """The pumps' `StructureFlow` and slope (cfs, cfs/ft) at `heads`, the pumps' settings
        `k`, the previous pass's flows in `state` and the conduit `terms` of the pass:
        ``pump_getInflow`` then ``getModPumpFlow`` (`pumps`), the node sums those of the
        inlet before the pump's own turn in the trial (`pumps.PumpStation`)."""
        ps, st = s.params, self.pump_station
        q_all = state[f"{self.layer}.q"] / QCF
        in_p, out_p = st.inflow_outflow(q_all, inflow / QCF)
        n_nodes = self.node_invert.shape[0]
        conduit_area = torch.zeros(heads.shape[:-1] + (n_nodes,), dtype=F64)
        conduit_area = (conduit_area.index_add(-1, self.u, terms.surface_area1)
                        .index_add(-1, self.v, terms.surface_area2))           # dynwave.c:545-546
        own = storage_ft2[..., self.pump_inlet]
        first, second = (None, None) if ends is None else ends
        surf = st.surface_area(own, conduit_area[..., s.u], first, second)
        jinv = self.node_invert[self.junction_idx]
        depth_ft = torch.clamp(heads[..., self.junction_idx] - jinv, min=0.0) / LCF
        volume_ft3 = self._node_volume_ft3(heads)
        start_depth = state["sewer.junction_depth"]
        start_heads = jinv + start_depth
        q_curve, dqdh = pump_curve_flow(ps, k, depth_ft[..., self.pump_inlet],
                                        volume_ft3[..., self.pump_inlet], h1, h2)
        q = torch.where(ps.kind == IDEAL_PUMP, in_p * k, q_curve)              # link.c:1572
        q = mod_pump_flow(
            ps.kind, q, self.pump_storage, self.pump_full_volume_ft3, in_p, out_p,
            state["sewer.net_inflow"][..., self.pump_inlet] / QCF,
            start_depth[..., self.pump_inlet] / LCF,
            self._node_volume_ft3(start_heads)[..., self.pump_inlet], surf, dt)
        q = torch.where(k == 0.0, torch.zeros_like(q), q)                      # link.c:550
        by_head = (ps.kind == TYPE3_PUMP) | (ps.kind == TYPE5_PUMP)
        slope = torch.where(by_head & (k != 0.0), torch.clamp(dqdh, min=0.0),
                            torch.zeros_like(dqdh))
        zero = torch.zeros_like(q)
        return StructureFlow(q, slope, zero.long(), zero, zero), slope

    def orifice_end_areas(self, state: Mapping) -> tuple[Tensor, Tensor]:
        """The end areas (ft2) the orifices give their first and second nodes at `state`'s
        heads and settings, as `_structure_terms` sums them (zero at a UP_ or DN_CRITICAL
        end and at a storage unit): the values a closing orifice keeps."""
        s = self.structures["orifice"]
        inv = self.node_invert
        heads = state[f"{self.layer}.phi"]
        h_ft = (inv + torch.clamp(heads - inv, min=0.0)) / LCF
        z_ft = inv / LCF
        k = state["sewer.setting"][..., s.first:s.first + len(s.names)]
        z1, z2 = z_ft[s.u], z_ft[s.v]
        flow = orifice_flow(s.params.at_setting(k), h_ft[..., s.u], h_ft[..., s.v], z1, z2,
                            z1 + s.offset1, s.block_forward, s.block_reverse)
        half = flow.surf_area / 2.0
        end1 = torch.where((flow.flow_class == UP_CRITICAL) | self.node_storage[s.u],
                           torch.zeros_like(half), half)
        end2 = torch.where((flow.flow_class == DN_CRITICAL) | self.node_storage[s.v],
                           torch.zeros_like(half), half)
        return end1, end2

    def _surcharged(self, heads: Tensor) -> Tensor:
        """EXTRAN's surcharge test on the junction heads `heads` (m, full-node): depth above
        a positive crown (`dynwave.c:690`), at a storage unit with a SurDepth depth above its
        full depth (`dynwave.c:683-687`), as a bool per junction."""
        depth_ft = self._junction_depth_ft(heads)
        if self.options.surcharge_method != "extran":
            return torch.zeros_like(depth_ft, dtype=torch.bool)
        at_ft = torch.where(self.is_storage, self.full_depth / LCF, self.crown_ft)
        return self.surchargeable & (depth_ft > at_ft)

    def _carried(self, state: Mapping, terms: ConduitTerms, inflow: Tensor,
                 storage_loss: Tensor | None = None) -> State:
        heads = state[f"{self.layer}.phi"]
        q = state[f"{self.layer}.q"]
        jidx = self.junction_idx
        n = self.node_invert.shape[-1]
        batch = broadcast_shapes(q.shape[:-1], inflow.shape[:-1])
        through = torch.zeros(batch + (n,), dtype=F64)
        q_b = q.expand(batch + q.shape[-1:])
        # every link's flow, conduits and regulators (dynwave.c:528-541, 382-412)
        through = through.index_add(-1, self.edge_v, q_b).index_add(-1, self.edge_u, -q_b)
        net_inflow = inflow + through                                  # dynwave.c:528-541
        conduit_loss = self._conduit_loss_nodes(terms)
        if conduit_loss is not None:                                   # dynwave.c:555-572
            net_inflow = net_inflow - conduit_loss
        if storage_loss is not None:                                   # dynwave.c:320
            net_inflow = net_inflow - storage_loss
        carried = {
            "sewer.junction_depth": torch.clamp(heads[..., jidx] - self.node_invert[jidx],
                                                min=0.0),
            "sewer.pipe_flow": q[..., :self.n_pipes],
            "sewer.mid_area": terms.mid_area * _AREA_SI,
            "sewer.net_inflow": net_inflow[..., jidx],
        }
        if self.pipe_losses:              # the next step's Link.oldDepth
            carried["sewer.pipe_depth"] = terms.depth * LCF
        if self.controlled:               # the link settings only change between steps
            for key in CONTROL_KEYS:
                carried[key] = state[key]
        return carried

    def outfall_stage(self, t: float, series: Mapping | None = None) -> Tensor:
        """Every outfall's stage elevation (m, shape ``(n_outfalls,)``) `t` s into the run:
        a TIDAL one's from its curve (`tidal_stage`), a TIMESERIES one's from `series[name]`
        (a stage in m, or a function of `t` returning one), a FIXED one's its own stage, the
        others' their invert (unused). This is the ``"outfall_stage"`` driver of a step
        whose END is at `t` (SWMM sets the outfall depth with the routing clock already
        advanced: `routing.c:301-302` before `routeFlow`, node.c:1449, 1456)."""
        series = {} if series is None else series
        stage = []
        for o in self._outfalls:
            if o.kind == "tidal":
                value = tidal_stage(o.tide, t)
            elif o.kind == "timeseries":
                source = series[o.name]
                value = float(source(t) if callable(source) else source)
                if not math.isfinite(value):
                    raise ValueError(
                        f"DynamicWaveHydraulics: outfall_stage of timeseries outfall "
                        f"{o.name!r} is {value!r} at t={t} s; it must be finite (m elevation)")
            else:
                value = o.stage if o.stage is not None else o.invert
            stage.append(value)
        return torch.tensor(stage, dtype=F64)

    def check_outfall_stage(self, series: Mapping | None) -> dict:
        """`series` as `outfall_stage` takes it: ``{timeseries outfall name: stage in m or
        function of t}``, covering every TIMESERIES outfall and nothing else (a tidal
        outfall reads its curve, the others their own stage). Returns it as a dict."""
        series = {} if series is None else dict(series)
        wanted = {o.name for o in self._outfalls if o.kind == "timeseries"}
        kinds = {o.name: o.kind for o in self._outfalls}
        for name in series:
            if name not in wanted:
                what = ("is not an outfall" if name not in kinds else
                        f"is a {kinds[name]} outfall, whose stage is "
                        + ("its tide curve's" if kinds[name] == "tidal" else "not time-varying"))
                raise ValueError(
                    f"DynamicWaveHydraulics: outfall_stage names {name!r}, which {what}; it "
                    f"takes the timeseries outfalls {sorted(wanted)}")
        missing = sorted(wanted - set(series))
        if missing:
            raise ValueError(
                f"DynamicWaveHydraulics: timeseries outfall(s) {missing} need an "
                f"outfall_stage (m elevation, or a function of t in s) each")
        return series

    def _stage_ft(self, drivers: Mapping | None) -> Tensor:
        """Each outfall's stage (ft): its own stage, a TIDAL or TIMESERIES outfall's the
        ``"outfall_stage"`` driver's (m, ``(..., n_outfalls)``)."""
        if not bool(self.outfall_staged.any()):
            return self.outfall_stage_ft
        if drivers is None or "outfall_stage" not in drivers:
            raise KeyError(
                f"DynamicWaveHydraulics: driver 'outfall_stage' (m elevation, one per "
                f"outfall) is required by outfall(s) "
                f"{[o.name for o in self._outfalls if o.kind in _STAGED_KINDS]}; "
                f"simulate_dynamic_wave and build_model set it")
        given = torch.as_tensor(drivers["outfall_stage"], dtype=F64)
        if given.shape[-1] != len(self._outfalls):
            raise ValueError(
                f"DynamicWaveHydraulics: outfall_stage must have one value per outfall "
                f"({len(self._outfalls)}), got shape {tuple(given.shape)}")
        if not bool(torch.isfinite(given[..., self.outfall_staged]).all()):
            bad = [o.name for o, k in zip(self._outfalls, self.outfall_staged.tolist(),
                                           strict=True) if k]
            raise ValueError(f"DynamicWaveHydraulics: non-finite outfall_stage at {bad}")
        return torch.where(self.outfall_staged, given / LCF, self.outfall_stage_ft)

    def _outfall_heads(self, q: Tensor | None, stage_ft: Tensor | None = None,
                       depths: tuple[Tensor, Tensor] | None = None) -> Tensor:
        """`link.c::link_setOutfallDepth` + `node.c::outfall_setOutletDepth` (link.c:728-766,
        node.c:1413-1492), from each outfall conduit's flow `q` (m3/s, the pipes' columns)
        and the outfalls' stages `stage_ft` (ft, `_stage_ft`; default the static ones):
        heads (m). `depths` is ``(yN, yC)`` (ft, per outfall) of those flows when the caller
        has them (`_flow_depths`; `q` is then not read). An outfall on a regulator takes
        yN = yC = 0. FIXED, TIDAL and TIMESERIES outfalls share the stage rule
        (node.c:1469-1490); they differ only in the stage."""
        op = self._outfall_pipe
        geo = self.geo
        if depths is None:
            if geo.xsect is None:
                q_ft = (q[..., op] / QCF / self.barrels[op]).abs()        # link.c:759
                yf, af = geo.y_full[op], geo.a_full[op]
                depths = (normal_depth_ft(q_ft, yf, af, geo.s_full[op], geo.beta[op]),
                          critical_depth(q_ft, yf, af, geo.critical_flows[op]))
            else:
                q_ft = (q[..., :self.n_pipes] / QCF / self.barrels).abs()
                depths = (geo.normal_depth(q_ft)[..., op], geo.critical_depth(q_ft)[..., op])
        y_norm, y_crit = depths
        # a regulator's outfall: yNorm = yCrit = 0 (link.c:753-765)
        reg = self.outfall_on_structure
        y_norm = torch.where(reg, torch.zeros_like(y_norm), y_norm)
        y_crit = torch.where(reg, torch.zeros_like(y_crit), y_crit)
        z = self.outfall_offset_ft
        zero = torch.zeros_like(y_norm)
        free = torch.where(z > 0.0, zero, torch.minimum(y_norm, y_crit))      # node.c:1431
        normal = torch.where(z > 0.0, zero, y_norm)                           # node.c:1436
        invert = self.node_invert[self.outfall_idx] / LCF
        stage = self.outfall_stage_ft if stage_ft is None else stage_ft
        yc = torch.minimum(y_crit, y_norm)                                    # node.c:1469
        fixed = torch.where(
            yc + z + invert < stage, (stage - invert).expand_as(yc),         # node.c:1473
            torch.where(
                z > 0.0,
                torch.where(stage < invert + z,
                            torch.clamp(stage - invert, min=0.0).expand_as(yc),
                            z + yc),                                          # node.c:1481-1485
                yc))                                                          # node.c:1490
        kind = self.outfall_kind
        depth = torch.where(kind == 0, free,
                            torch.where(kind == 1, normal, fixed))    # kinds >= 2 are staged
        return self.node_invert[self.outfall_idx] + depth * LCF

    def _wall_cap(self, terms: ConduitTerms, area_ft2: Tensor, surf: Tensor, dt: float
                  ) -> Tensor:
        """``S_max`` (m2/s, per pipe): `K_SWITCH` times the smaller end junction's grounding
        (storage slope ``2 A_s/dt`` plus its conduits' conveyance), the cap on a wall's slope
        (`_switch_slope`) and on the culvert cap's."""
        k_si = terms.conveyance * (QCF / LCF)
        ground = torch.zeros_like(surf).index_add(-1, self.u, k_si).index_add(-1, self.v, k_si)
        ground = ground.index_add(-1, self.junction_idx, 2.0 / dt * area_ft2 * _AREA_SI)
        ground = torch.where(self.is_outfall, torch.full_like(ground, torch.inf), ground)
        return K_SWITCH * torch.minimum(ground[..., self.u], ground[..., self.v])

    def _switch_slope(self, terms: ConduitTerms, area_ft2: Tensor, surf: Tensor, dt: float,
                      stage_ft: Tensor, depths: Mapping) -> tuple[Tensor, Tensor]:
        """``(S, heads)``: the outfall heads at the fed flows (m, `_outfall_heads`, from the
        pass's `_flow_depths` `depths`, in one call with the bumped flows below) and the
        slope switch's wall slope S (m2/s, per pipe; `DynamicWaveConduit`): the
        linearised switching surface ``1/(c + 1/S_max)``. ``c`` (m per m3/s) is how far the
        lagged outfall depth at the conduit's downstream end moves per unit of its flow
        (`_outfall_heads`, a forward difference; 0 for other conduits): on the surface
        ``y1 = y2(q)`` the pass then sees the outfall depth follow the flow, as it does at
        the fixed point. An outfall at the conduit's UPSTREAM end gets no such term (there
        the surface would fall with the flow), so that conduit's S is ``S_max``: its
        switch stays exact at the fixed point but may take more passes to settle. ``S_max``
        is `K_SWITCH` times the smaller end junction's grounding (storage slope ``2 A_s/dt``
        plus its conduits' conveyance)."""
        s_max = self._wall_cap(terms, area_ft2, surf, dt)
        op = self._outfall_pipe
        step = depths["step"]
        # the heads at the fed and the bumped flows, in one evaluation
        heads = self._outfall_heads(None, stage_ft, (
            torch.stack([depths["normal"][..., op], depths["normal_bumped"]]),
            torch.stack([depths["critical"][..., op], depths["critical_bumped"]])))
        rise = (heads[1] - heads[0]) / step
        rise = torch.where(self.outfall_at_v & ~self.outfall_on_structure,
                           torch.clamp(rise, min=0.0), torch.zeros_like(rise))
        compliance = torch.zeros_like(s_max).index_add(-1, op, rise.expand(
            s_max.shape[:-1] + rise.shape[-1:]))
        return 1.0 / (compliance + 1.0 / s_max), heads[0]

    # ------------------------------------------------------------------ protocol
    def __call__(self, state: Mapping, drivers: Mapping,
                 ctx: StepContext | None = None) -> dict[str, Tensor]:
        """One pass's drivers (see the class docstring); a query (no `ctx`, or
        ``ctx.dt is None``) returns the carried and reporting keys only."""
        dt = None if ctx is None else ctx.dt
        inflow = self._inflow(drivers)
        storage_loss = self._storage_loss(drivers)
        heads = state[f"{self.layer}.phi"]
        drop = None
        regulators: dict = {}
        if dt is None:
            terms = self._terms(state, self.options.routing_step, drivers=drivers)
            out = self._carried(state, terms, inflow, storage_loss)
        else:
            terms, out, drop, regulators, storage_ft2, depths = self._pass_terms(
                state, inflow, dt, drivers, storage_loss)
        q_all = state[f"{self.layer}.q"]
        q = q_all[..., :self.n_pipes]
        depth_ft = terms.depth
        a_link = self.geo.area(depth_ft)
        moving = (depth_ft > _VELOCITY_MIN_DEPTH) & (a_link > FUDGE)          # link.c:834-840
        v_ft = (q / QCF / self.barrels) / torch.where(moving, a_link, torch.ones_like(a_link))
        out.update({
            "sewer.q": q,
            "sewer.h": depth_ft * LCF,
            "sewer.v": torch.where(moving, v_ft, torch.zeros_like(v_ft)) * LCF,
            "sewer.V_wet": terms.volume * QCF,
            "sewer.surcharged": self._surcharged(heads).to(F64),
            "sewer.storage_volume": self._storage_volume(heads),
            **self._hydraulic_radius_depth(depth_ft),
        })
        out.update(self._headspace(depth_ft * LCF, heads, drivers))
        if self.quality_layer is not None and dt is None:
            # a query: the transport control volumes at the state's own time
            volume = state["sewer.quality_volume"]
            out.update({"sewer.quality_volume": volume,
                        f"{self.quality_layer}.capacity": volume})
        if dt is None:
            return out
        jidx = self.junction_idx
        n = self.node_invert.shape[-1]
        sources = inflow            # the lateral inflow less the conduits' and storage losses
        for loss in (self._conduit_loss_nodes(terms), storage_loss):
            if loss is not None:
                sources = sources - loss
        surf = torch.zeros(terms.surface_area1.shape[:-1] + (n,), dtype=F64)
        surf = surf.index_add(-1, self.u, terms.surface_area1)                # dynwave.c:582-583
        surf = surf.index_add(-1, self.v, terms.surface_area2)
        # the storage unit's own area plus its conduit ends' (not zeroed at a storage node,
        # dwflow.c:546-547; only a regulator's are, dynwave.c:520-523) and its regulators'
        # ends' (in `storage_ft2`, `_pass_terms`), then the floor
        area_ft2 = torch.clamp(surf[..., jidx] + storage_ft2,
                               min=self.min_area_ft2)                         # dynwave.c:670
        # conveyance the pass lags into c0 (other end pinned), per junction (dynwave.c:578)
        lagged = torch.zeros_like(surf).index_add(-1, self.u, terms.lagged_conveyance1)
        lagged = lagged.index_add(-1, self.v, terms.lagged_conveyance2)[..., jidx]
        anchor = lagged * (QCF / LCF) + K_ANCHOR * 2.0 / dt * area_ft2 * _AREA_SI
        switch = terms.slope_switch
        stage_ft = self._stage_ft(drivers)
        switch_slope, outfall_heads = self._switch_slope(terms, area_ft2, surf, dt, stage_ft,
                                                         depths)
        # the drop switch's wall slope: K_SWITCH times the junction's grounding above it
        k_si = terms.conveyance * (QCF / LCF)
        sum_k = torch.zeros_like(surf).index_add(-1, self.u, k_si).index_add(-1, self.v, k_si)
        drop_slope = K_SWITCH * (2.0 / dt * drop["above"] + sum_k[..., jidx])
        drop_head = torch.where(drop["applies"], drop["head"],
                                torch.full_like(drop["head"], torch.inf))
        if self.storage.ponds:                    # the ponding junctions' second line
            raw_ft2 = surf[..., jidx] + storage_ft2
            pond_area = torch.where(
                self.can_pond, torch.clamp(raw_ft2 + self.pond_area_ft2, min=self.min_area_ft2),
                area_ft2) * _AREA_SI                                        # dynwave.c:309-312
            out_pond = {"water.pond_area": pond_area,
                        "water.pond_switch_slope":
                            K_SWITCH * (2.0 / dt * pond_area + sum_k[..., jidx])}
        else:
            out_pond = {}
        out.update({
            f"{self.layer}.sources": sources,
            f"{self.layer}.phi_boundary": outfall_heads,
            "water.flow_offset": terms.flow_offset * QCF,
            "water.conveyance": terms.conveyance * (QCF / LCF),
            "water.normal_flow": torch.where(switch, terms.switch_normal_flow,
                                             terms.normal_flow) * QCF,
            "water.limited": (terms.limited & ~switch).to(F64),
            "water.slope_switch": switch.to(F64),
            "water.switch_drop": terms.switch_drop * LCF,
            "water.switch_slope": torch.where(switch, switch_slope,
                                              torch.zeros_like(switch_slope)),
            "water.flow_last": q,
            "water.dry_flow_cap": FUDGE_FLOW * self.barrels,
            "water.dry_upstream": terms.dry_upstream.to(F64),
            "water.dry_downstream": terms.dry_downstream.to(F64),
            "water.storage_area": area_ft2 * _AREA_SI,
            "water.storage_rate": torch.tensor(1.0 / dt, dtype=F64),
            "water.head_start": self.node_invert[jidx] + state["sewer.junction_depth"],
            "water.net_inflow_start": state["sewer.net_inflow"],
            "water.surcharge_slope": anchor,
            "water.head_last": state[f"{self.layer}.phi"][..., jidx],
            "water.net_inflow_last": out["sewer.net_inflow"],
            "water.area_switch_head": drop_head,
            "water.area_below": drop["below"],
            "water.area_above": drop["above"],
            "water.area_switch_slope": drop_slope,
            **out_pond,
        })
        if self.storage.ponds:
            out["water.ponded"] = self.storage.ponded(out)
        if terms.cap_flow is not None:        # the culverts' inlet control (`culvert.py`)
            cap_slope = terms.cap_slope * (QCF / LCF)
            out.update({
                "water.cap_flow": terms.cap_flow * QCF,
                "water.cap_drop": terms.cap_drop * LCF,
                "water.cap_slope": torch.minimum(
                    cap_slope, self._wall_cap(terms, area_ft2, surf, dt))})
        # the conduit limits a network has at all (`DynamicWaveConduit`: absent is none)
        if bool(torch.isfinite(self.max_flow).any()):
            out["water.max_flow"] = self.max_flow
        if bool(self.flap_block_reverse.any()):
            out["water.flap_block_reverse"] = self.flap_block_reverse.to(F64)
        if bool(self.flap_block_forward.any()):
            out["water.flap_block_forward"] = self.flap_block_forward.to(F64)
        for kind, (flow, slope) in regulators.items():
            s = self.structures[kind]
            out.update({
                f"water.{kind}.flow_offset": flow.flow * QCF,
                f"water.{kind}.slope": slope * (QCF / LCF),
                f"water.{kind}.dp_fed": heads[..., s.u] - heads[..., s.v],
                f"water.{kind}.flap_block_reverse": s.block_reverse.to(F64),
                f"water.{kind}.flap_block_forward": s.block_forward.to(F64),
            })
        # the pass constants of the junction withdrawal, evaluated once (`JunctionStorage`)
        out["water.hold_stiffness"] = self.storage._stiff(out)
        if self.storage.extran:
            out["water.crown_withdrawal"] = self.storage.crown_withdrawal(out)
        out["water.surcharged"] = self.storage.surcharged(out)
        out["water.top_withdrawal"] = self.storage.top_withdrawal(out)
        out["water.flood_wall_slope"] = self.storage._wall_slope(out)
        excess = self.storage.flood_excess(out)
        flooded = excess > 0.0
        # SWMM's overflow is the step's trapezoid net inflow dV/dt, zeroed below FUDGE
        # (dynwave.c:673-674, 783-791); it never enters the next step's oldNetInflow
        # (node.c:303 takes inflow - outflow, which exclude it).
        overflow = 0.5 * (state["sewer.net_inflow"] + out["sewer.net_inflow"])
        overflow = torch.where(flooded & (overflow >= FUDGE_FLOW), overflow,
                               torch.zeros_like(overflow))
        if self.storage.ponds:
            # a ponding junction floods whenever it ends above its full depth, and the
            # volume it gains then is its flooding (getFloodedDepth, dynwave.c:787-792: the
            # ponded volume rises by dV and falls back without a negative overflow)
            ponded = out["water.ponded"] > 0.5
            trapezoid = 0.5 * (state["sewer.net_inflow"] + out["sewer.net_inflow"])
            pond_overflow = torch.where(ponded & (trapezoid >= FUDGE_FLOW), trapezoid,
                                        torch.zeros_like(trapezoid))
            flooded = torch.where(self.can_pond, ponded, flooded)
            overflow = torch.where(self.can_pond, pond_overflow, overflow)
        out.update({"water.flood_excess": excess, "water.flooded": flooded.to(F64),
                    "sewer.overflow": overflow})
        if self.quality_layer is not None:
            out.update(self._quality_volume(state, out["sewer.net_inflow"], overflow, dt))
        return out

    def _hydraulic_radius_depth(self, depth_ft: Tensor) -> dict[str, Tensor]:
        """``"sewer.R_h"`` (hydraulic radius, m) and ``"sewer.d_m"`` (hydraulic mean depth
        A/T, m, 0 for a conduit flowing full) at the link depths `depth_ft`: the exact
        circle's for a circular pipe, SWMM's section tables' (`XsectSet`) otherwise."""
        out = {"sewer.R_h": geom.hydraulic_radius(depth_ft * LCF, self.diameter),
               "sewer.d_m": geom.hydraulic_mean_depth(depth_ft * LCF, self.diameter)}
        if self.geo.xsect is None:
            return out
        xs, geo = self.geo.xsect, self.geo
        dead = (depth_ft <= FUDGE) | (~geo.is_open & (geo.y_full - depth_ft <= FUDGE))
        y = torch.where(dead, 0.5 * geo.y_full, depth_ft)
        area, width = xs.area_width(y)
        radius = torch.where(depth_ft >= geo.y_full, geo.r_full, xs.radius(y))
        d_m = torch.where(dead, torch.zeros_like(y), area / torch.where(
            width > 0.0, width, torch.ones_like(width))) * LCF
        circle = self.is_circle
        return {"sewer.R_h": torch.where(circle, out["sewer.R_h"], radius * LCF),
                "sewer.d_m": torch.where(circle, out["sewer.d_m"], d_m)}

    def _headspace(self, depth: Tensor, heads: Tensor, drivers: Mapping) -> dict[str, Tensor]:
        """The headspace keys at the link depths `depth` (m, one barrel's, per pipe) and the
        full-node `heads` (m): the class docstring's "Air". Every key is finite at a full
        conduit (the depth is capped at `AIR_DEPTH_RATIO_MAX` of the diameter) and the
        derivatives are safe at both ends of the circle (`geometry.air_geometry`)."""
        capped = torch.minimum(depth, AIR_DEPTH_RATIO_MAX * self.diameter)
        a_air, _p_air, d_h = geom.air_geometry(capped, self.diameter)
        top = geom.top_width(capped, self.diameter)
        if self.geo.xsect is not None:
            # the section tables' headspace: the free area aFull - A(y), the free-surface
            # width W(y) and the air perimeter P - A/R + W (P = aFull/rFull the full wetted
            # perimeter), over the non-circular pipes
            geo = self.geo
            y = capped / LCF
            area, width = geo.xsect.area_width(y)
            radius = geo.xsect.radius(y)
            free = (geo.a_full - area) * LCF**2
            wetted = area / torch.where(radius > 0.0, radius, torch.ones_like(radius))
            p_air = (geo.a_full / geo.r_full - wetted + width) * LCF
            d_gen = 4.0 * free / torch.where(p_air > 0.0, p_air, torch.ones_like(p_air))
            circle = self.is_circle
            a_air = torch.where(circle, a_air, free)
            d_h = torch.where(circle, d_h, d_gen)
            top = torch.where(circle, top, width * LCF)
        a_total = self.barrels * a_air
        volume = a_total * self.length
        out = {"sewer.A_air": a_total, "sewer.D_h": d_h, "sewer.T": self.barrels * top,
               "sewer.V_air": volume}
        if not self.air:
            return out
        out["rho_air_nodes"] = air_density_nodes(
            drivers, self.junction_idx, self.node_invert.shape[-1], name="DynamicWaveHydraulics")
        capacity = volume.index_select(-1, self.out_pipe) + self._shaft_air(heads)
        out[f"{self.air_quality_layer}.capacity"] = capacity
        return out

    def _shaft_air(self, heads: Tensor) -> Tensor:
        """The air of each junction above its water (m3): ``A_s (full depth - depth)`` for
        a junction, the storage curve's volume between the depth and the full depth for a
        storage unit; 0 without `ground` on every manhole (as the kinematic path)."""
        zero = torch.zeros_like(self._junction_depth_ft(heads))
        if self.shaft_area is None:
            return zero
        depth = self._junction_depth_ft(heads) * LCF
        plain = self.shaft_area * torch.clamp(self.full_depth - depth, min=0.0)
        if self.storage_curves is None:
            return plain
        stored = torch.clamp(self.storage_curves.full_volume - self.storage_curves.volume(depth),
                             min=0.0)
        return torch.where(self.is_storage, stored, plain)

    def _quality_volume(self, state: Mapping, net_inflow: Tensor, overflow: Tensor,
                        dt: float) -> dict[str, Tensor]:
        """The transport control volumes at the end of the step and the flooded fraction
        (class docstring, "Quality"): ``V_end = max(V_start + dt (N - overflow),
        CAPACITY_FLOOR)`` from the pinned step-start ``"sewer.quality_volume"``, and
        ``dt overflow / V_start`` capped at 1. The cap binds only where one step's overflow
        exceeds the junction's whole step-start contents: all of them leave, and the excess
        overflow is taken from the step's inflow without its mass, so the water balance (and
        a uniform concentration) holds there only to that excess."""
        start = state["sewer.quality_volume"]
        end = torch.clamp(start + dt * (net_inflow - overflow), min=CAPACITY_FLOOR)
        lost = torch.clamp(dt * overflow / start, max=1.0)
        return {"sewer.quality_volume": end, f"{self.quality_layer}.capacity": end,
                f"{self.quality_layer}.overflow_loss": lost}

    def quality_capacity(self, wetted: Tensor, stored: Tensor) -> Tensor:
        """The geometric quality capacity (m3, per junction) from the conduits'
        wetted volumes `wetted` (m3, totals over barrels) and the junctions' own stored
        volumes `stored` (m3, 0 but at a storage unit): half of every attached conduit's
        volume plus its own, floored at `CAPACITY_FLOOR` so a dry junction keeps a
        well-posed transport row. Half of an outfall conduit's volume belongs to no
        junction. It is the initial ``"sewer.quality_volume"`` (`initial_state`)."""
        half = 0.5 * wetted
        batch = half.shape[:-1]
        nodes = torch.zeros(batch + self.node_invert.shape, dtype=F64)
        nodes = nodes.index_add(-1, self.u, half).index_add(-1, self.v, half)
        return torch.clamp(nodes[..., self.junction_idx] + stored, min=CAPACITY_FLOOR)

    def carried(self, state: Mapping, drivers: Mapping, *, dt: float | None = None,
                start: Mapping | None = None) -> State:
        """The four carried keys evaluated AT `state` (``water.phi``, ``water.q``), for
        re-synchronising after a step of `dt` s (default: the routing step): the pass's own
        copy is its input's, which equals its output only to the iteration tolerance.

        With `start` (the step-start state) the conduit terms are a pass's, the drop-end
        area switch's hold included: the hold is decided against the step-start depths and
        net inflows, as within the step, so that the carried ``"sewer.mid_area"`` is the
        held class's at a junction held on its threshold. Without it, the terms take no
        hold."""
        dt = self.options.routing_step if dt is None else dt
        inflow = self._inflow(drivers)
        storage_loss = self._storage_loss(drivers)
        if start is None:
            return self._carried(state, self._terms(state, dt, drivers=drivers), inflow,
                                 storage_loss)
        pinned = dict(state, **{k: start[k] for k in self.state_keys})
        terms = self._pass_terms(pinned, inflow, dt, drivers, storage_loss)[0]
        return self._carried(state, terms, inflow, storage_loss)

    def _pass_terms(self, state: Mapping, inflow: Tensor, dt: float,
                    drivers: Mapping | None = None, storage_loss: Tensor | None = None
                    ) -> tuple[ConduitTerms, State, dict, dict, Tensor, dict]:
        """One pass's conduit terms at `state` (its carried keys the step start's), with
        the drop-end area switch's hold: ``(terms, carried, drop, regulators,
        storage_ft2, depths)``, `drop` from `_drop_switch`, the regulators from
        `_structure_terms`, each junction's own storage area (ft2, its regulators' ends
        included) and the pass's `_flow_depths`."""
        depths = self._flow_depths(state[f"{self.layer}.q"][..., :self.n_pipes])
        flow_depths = (depths["normal"], depths["critical"])
        terms = self._terms(state, dt, flow_depths=flow_depths, drivers=drivers)
        out = self._carried(state, terms, inflow, storage_loss)
        heads = state[f"{self.layer}.phi"]
        # a storage unit's own area at the previous pass's depth (dynwave.c:315)
        storage_ft2 = self._storage_area_ft2(heads)
        regulators: dict = {}
        if self.structures:
            # the regulators at the previous pass's heads (dynwave.c:423-454): their flows,
            # slopes and end areas, which join the storage units' own (dynwave.c:503-524)
            regulators, reg_ft2 = self._structure_terms(heads, state, inflow, dt, terms,
                                                        storage_ft2)
            storage_ft2 = storage_ft2 + reg_ft2[..., self.junction_idx]
        # the drop-end area switch (JunctionStorage): a junction held on its threshold
        # has its chosen drop end classified on the upper side, SWMM's class AT it. The
        # hold set is decided once per pass, from the first evaluation's areas and the
        # fed net inflow (which does not depend on the class), and fixed for the pass:
        # the re-evaluation only supplies the held class's areas, so re-deciding on it
        # would let the class it sets feed back into its own decision within one pass.
        drop = self._drop_switch(state, terms, out["sewer.net_inflow"], dt, storage_ft2)
        if bool(drop["held"].any()):
            terms = self._terms(state, dt, (drop["hold1"], drop["hold2"]), flow_depths,
                                drivers)
            out = self._carried(state, terms, inflow, storage_loss)
            drop = self._drop_switch(state, terms, out["sewer.net_inflow"], dt, storage_ft2)
        return terms, out, drop, regulators, storage_ft2, depths

    def _initial_stage_ft(self, drivers: Mapping | None) -> Tensor:
        """The outfalls' stages (ft) at t = 0 for the initial heads: the driver's if it is
        given, else the tide curves' at t = 0 (a TIMESERIES outfall has no stage without
        its driver)."""
        if drivers is not None and "outfall_stage" in drivers:
            return self._stage_ft(drivers)
        if any(o.kind == "timeseries" for o in self._outfalls):
            raise KeyError("DynamicWaveHydraulics: the initial state needs the "
                           "'outfall_stage' driver for a timeseries outfall")
        return self._stage_ft({"outfall_stage": self.outfall_stage(0.0)})

    def initial_state(self, drivers: Mapping | None = None) -> State:
        """SWMM's initial state (flowrout.c:75-102, 336-507), cold and dry unless the network
        sets InitDepth (`Manhole.initial_depth`) or InitFlow (`Pipe.initial_flow`).

        - Each conduit with an InitFlow ``q0`` (a total, positive from `u` to `v`) starts at
          the normal depth of one barrel's ``q0 / barrels``, capped at the depth of ``qMax``
          (`link.c::link_getYnorm`, link.c:783-804; `conduit_initState`, link.c:1312).
        - A junction with no InitDepth starts at the mean over its attached conduits of that
          depth plus the conduit's FIRST-end offset (zero for a conduit without InitFlow,
          which still counts in the mean): `initNodeDepths`, flowrout.c:337-385 (the first
          offset is used at both ends there, as in SWMM). A junction with an InitDepth keeps
          it, and so does a storage unit, whose InitDepth of 0 is not averaged either
          (flowrout.c:375).
        - Outfalls start at the depth `link_setOutfallDepth` gives from the conduit's initial
          flow (a FIXED outfall at its stage, a TIDAL or TIMESERIES one at its stage at
          t = 0, the others at their invert at zero flow; flowrout.c:383).
        - A conduit with no InitFlow starts at the mean of its end-node depths above its
          offsets, clamped to [0, yFull] and at least FUDGE (`initLinkDepths`,
          flowrout.c:389-420); the area at that depth is its stored area (`initLinks`,
          flowrout.c:497).
        - A junction's net inflow is that of the initial flows through it (`initNodes`,
          flowrout.c:458-466 and node.c:303): 0 with no InitFlow. SWMM zeroes the lateral
          inflow before routing starts (node.c:264, flowrout.c:437), so the first step's
          ``oldNetInflow`` has no lateral part (routing.c:236-242).

        `drivers`, when given, has its ``"inflow"`` validated and supplies the
        ``"outfall_stage"`` at t = 0."""
        if drivers is not None:
            self._inflow(drivers)
        geo = self.geo
        q0 = self.initial_flow.clone()
        # link.c:783-804 / 1311: normal depth of one barrel's initial flow (0 where none)
        y_link = geo.normal_depth(q0.abs() / QCF / self.barrels)
        # flowrout.c:353-363: depth + the first offset, at both ends, where depth > FUDGE
        y_end = torch.where(y_link > FUDGE, y_link + geo.offset1, torch.zeros_like(y_link))
        total = torch.zeros(self.node_invert.shape, dtype=F64)
        count = torch.zeros_like(total)
        total = total.index_add(0, self.u, y_end).index_add(0, self.v, y_end)
        # every link counts, a regulator with depth 0 (link.c:518, flowrout.c:353-363)
        count = count.index_add(0, self.edge_u, torch.ones(self.edge_u.shape, dtype=F64))
        count = count.index_add(0, self.edge_v, torch.ones(self.edge_v.shape, dtype=F64))
        jidx = self.junction_idx
        given = self.initial_junction_depth / LCF
        average = torch.where(count[jidx] > 0.0, total[jidx] / count[jidx].clamp(min=1.0),
                              torch.zeros_like(given))                        # flowrout.c:378
        # a storage unit keeps its own InitDepth, 0 included (flowrout.c:375-376)
        depth_j = torch.where((given > 0.0) | self.is_storage, given, average)
        phi = self.node_invert.clone()
        phi[jidx] = self.node_invert[jidx] + depth_j * LCF
        phi[self.outfall_idx] = self._outfall_heads(
            q0, self._initial_stage_ft(drivers))                              # flowrout.c:383
        depth_ft = (phi - self.node_invert) / LCF
        y1 = torch.clamp(depth_ft[self.u] - geo.offset1, min=0.0)
        y2 = torch.clamp(depth_ft[self.v] - geo.offset2, min=0.0)
        y_mean = torch.clamp(0.5 * (torch.minimum(y1, geo.y_full)
                                    + torch.minimum(y2, geo.y_full)), min=FUDGE)  # :402-416
        y = torch.where(q0 == 0.0, y_mean, y_link)                            # flowrout.c:407, :418
        through = torch.zeros_like(total).index_add(0, self.v, q0).index_add(0, self.u, -q0)
        n_reg = self.edge_u.shape[0] - self.n_pipes
        state = {
            f"{self.layer}.phi": phi,
            f"{self.layer}.q": torch.cat([q0, torch.zeros(n_reg, dtype=F64)]),  # link.c:518
            "sewer.junction_depth": depth_j * LCF,
            "sewer.pipe_flow": q0.clone(),
            "sewer.mid_area": geo.area(y) * _AREA_SI,                         # :497
            "sewer.net_inflow": through[jidx],                                # flowrout.c:458
        }
        if self.pipe_losses:                    # Link.newDepth (flowrout.c:389-420)
            state["sewer.pipe_depth"] = y * LCF
        if self.controlled:
            start = (control_start(self.controls) if self.controls is not None else 0.0)
            state.update({
                "sewer.setting": self.initial_setting.clone(),               # link.c:1542-1543
                "sewer.target_setting": self.initial_setting.clone(),
                "sewer.time_last_set": torch.full((self.n_settings,), start, dtype=F64),
                "sewer.orifice_hold": torch.zeros(2 * self.orifice_rate.shape[0], dtype=F64),
            })
        if self.quality_layer is not None:
            # the half-volumes at the initial depths: the conduit volumes SWMM starts
            # from (as "sewer.V_wet" at this state) and a storage unit's own
            terms = self._terms(state, self.options.routing_step)
            state["sewer.quality_volume"] = self.quality_capacity(
                terms.volume * QCF, self._storage_volume(phi))
        return state

    def overflow(self, before: Mapping, after: Mapping, drivers: Mapping, *,
                 dt: float | None = None) -> Tensor:
        """SWMM's junction overflow (m3/s, >= 0) over the step from `before` to `after`
        (``dt`` s, default the routing step): ``0.5 (N_before + N_after)`` at each junction
        flooded at `after` (its lagged flood excess positive), 0 below SWMM's FUDGE flow
        and elsewhere (`dynwave.c:783-791`); the pass's ``"sewer.overflow"`` fed `after`."""
        dt = self.options.routing_step if dt is None else float(dt)
        drv = self(after | {k: before[k] for k in self.state_keys}, drivers,
                   StepContext(dt=dt))
        return drv["sewer.overflow"]

    def junction_residual(self, before: Mapping, after: Mapping, drivers: Mapping, *,
                          dt: float | None = None) -> Tensor:
        """``0.5 dt (w(H_after) - N_after)`` per junction (m3): the `JunctionStorage`
        withdrawal w (m3/s) against the net inflow N_after (m3/s), both at `after`, with
        the drivers of a pass fed `after` from the step start `before`; ``dt`` (s) defaults
        to the routing step. For a free-surface junction that is ``A_s (y_after -
        y_before) - 0.5 dt (N_before + N_after)``, for an EXTRAN-surcharged one
        ``-0.5 dt N_after``; at a flooded junction w is the flow over its top, so the
        residual is its balance with that outflow (the overflow), ``0.5 dt S (H_after -
        H_top)``. Zero to the iteration tolerance wherever the junction is wet."""
        dt = self.options.routing_step if dt is None else float(dt)
        drv = self(after | {k: before[k] for k in self.state_keys}, drivers,
                   StepContext(dt=dt))
        now = self.carried(after, drivers, dt=dt)
        heads = after[f"{self.layer}.phi"][..., self.junction_idx]
        withdrawal = self.storage.flow(heads, drv)
        return 0.5 * dt * (withdrawal - now["sewer.net_inflow"])


def control_start(controls) -> float:
    """SWMM's ``StartDate`` (link.c:526): the DateTime (days) of the midnight of the day a run
    starts, which a link's time since last set counts from until its first change."""
    return float(int(start_serial(controls.start)))


def iterate_tolerances(options: DynamicWaveOptions, *, iterate_tol: float | None = None,
                       iterate_max: int | None = None,
                       iterate_flow_tol: float | None = None) -> dict:
    """The iteration controls `build_dynamic_wave_model` builds with: ``{"iterate_tol",
    "iterate_flow_tol", "iterate_max", "on_limit"}``. Each argument given is used as is;
    each left ``None`` follows the file (`options`) as SWMM does:

    - `iterate_tol` (m) is SWMM's ``HeadTol`` (`DynamicWaveOptions.head_tol`: HEAD_TOLERANCE,
      or 0.005 ft). SWMM's trial test is the depth change of every node between trials
      (`dynwave.c::findNodeDepths`, ``|yNew - yOld| > HeadTol``); noodl's is the head
      change between passes, the same test.
    - `iterate_flow_tol` (m3/s): SWMM judges the heads alone, but on noodl's normal-flow
      slope switch the heads can stop moving while the flows the closure reads still do
      (`build_dynamic_wave_model`), so noodl judges the flows too, to the flow residual that
      matches the head tolerance at the smallest junction: a flow residual ``dQ`` changes a
      junction's next depth by ``dt dQ / (2 A_s)`` (SWMM's trapezoid, dynwave.c:694-711),
      so ``dQ = 2 A_min iterate_tol / dt`` moves the head by `iterate_tol` at the
      surface-area floor A_min (MIN_SURFAREA, dynwave.c:186-187), the largest move any
      junction makes for it, ``dt`` the routing step (the longest). It is floored at
      `FLOW_TOL_FLOOR`.
    - `iterate_max` is SWMM's ``MaxTrials`` (`DynamicWaveOptions.trials`: MAX_TRIALS, or 8;
      at least 2, `Model`'s minimum). Then ``"on_limit"`` is ``"accept"``: as SWMM, a step
      that uses up its trials is taken with its last pass and counted as not converging
      (`dynwave.c::dynwave_execute`, ``NonConvergeCount``; `simulate_dynamic_wave`'s
      ``"converged"``), not refused. An explicit `iterate_max` keeps ``"raise"``, `Model`'s default.
    """
    tol = options.head_tol if iterate_tol is None else float(iterate_tol)
    if iterate_flow_tol is None:
        min_area = (DEFAULT_MIN_SURFACE_AREA_FT2 * _AREA_SI if options.min_surface_area is None
                    else options.min_surface_area)
        iterate_flow_tol = max(FLOW_TOL_FLOOR, 2.0 * min_area * tol / options.routing_step)
    return {"iterate_tol": tol, "iterate_flow_tol": float(iterate_flow_tol),
            "iterate_max": max(2, options.trials) if iterate_max is None else int(iterate_max),
            "on_limit": "accept" if iterate_max is None else "raise"}


def build_dynamic_wave_model(net: SewerNetwork, *, iterate_tol: float | None = None,
                             iterate_max: int | None = None, relaxation: float = 0.5,
                             acceleration: str | None = "anderson", memory: int = 5,
                             iterate_flow_tol: float | None = None,
                             outfall_stage: Mapping | None = None,
                             quality: bool = False,
                             species: tuple[str, ...] = ("bod", "sulfide"),
                             scheme: str = "implicit",
                             air: bool = False, f_air: float = F_AIR_DEFAULT,
                             f_i: float = F_I_DEFAULT, c_s: float = 1.0,
                             leak_area: float = 8e-4, leak_cd: float = 0.6,
                             fans: tuple = (),
                             ) -> tuple[Model, State, Drivers]:
    """Assemble a dynamic-wave sewer and return ``(model, state, drivers)``.

    One `PotentialFlowLayer` ``"water"`` (quantity ``"head"``, m; flows m3/s) over the pipes,
    with the outfalls as its boundary and a `JunctionStorage` at every junction, and the
    `DynamicWaveHydraulics` closure; ``coupling="iterate"`` converges on the junction heads
    to `iterate_tol` (m) and the conduit flows to `iterate_flow_tol` (m3/s) within
    `iterate_max` passes, by default the file's own HEAD_TOLERANCE and MAX_TRIALS and the
    flow tolerance matching them (`iterate_tolerances`; as SWMM, a step that uses up the
    file's trials is then accepted and reported, not refused). The passes are Anderson-accelerated
    over the last `memory` of them by default (``acceleration="anderson"``, `Model`'s
    `iterate_acceleration`), with `relaxation` the damping of the steps it falls back to;
    ``acceleration=None`` is plain relaxation, which reaches the same fixed point in two to
    three times the passes. `outfall_stage` gives each TIMESERIES outfall its stage
    elevation: ``{outfall name: stage (m) or a function of the elapsed time t (s) returning
    it}``, refused for any outfall that is not a TIMESERIES one (a TIDAL outfall reads its
    tide curve); a TIMESERIES outfall read from a SWMM file defaults to the file's series
    (`Outfall.stage_series`, held at its ends as SWMM does) and this overrides it, and one
    built by hand needs it (`DynamicWaveHydraulics.outfall_stage`); it sets the initial heads (at
    t = 0) and is the default of `simulate_dynamic_wave`'s own `outfall_stage`.
    `iterate_flow_tol` (m3/s) is the tolerance on the conduit
    flows' own fixed-point residual, judged alongside the heads (`Model`'s
    `iterate_flow_tol`); the heads are never judged alone. On the normal-flow slope switch
    (`DynamicWaveConduit`) the heads
    bound the flow residual only through the wall slope S over the junction's grounding,
    so a heads-only test can stop a pass short of the fixed point (a flow residual of
    3e-5 m3/s was measured on the tree's drain-down). The state is
    SWMM's initial state (`DynamicWaveHydraulics.initial_state`: cold and dry, or its
    InitDepth and InitFlow). Attached to the model: `junction_idx`, `junction_invert` (m),
    `outfall_link_idx` (each outfall's link's ``"water.q"`` column), `outfall_stage` (the
    TIMESERIES outfalls' stage functions), `dynamic_wave` (the
    closure), `routing_step` (s), `pipe_names`,
    `link_names` (every ``"water.q"`` column: the pipes, then the orifices, weirs and
    outlets, each a `StructureLink` edge of its own kind), `pipe_columns`, `notes`,
    `input_specs`, `driver_template`.

    ``quality=True`` adds the `TransportLayer` ``"water_quality"`` of `species` (`_quality_parts`;
    `scheme` is its time scheme), judged by the iteration alongside the heads, and attaches
    `manhole_idx` (the junctions' node indices) and, with sulfide, `out_pipe` (each junction's
    outgoing conduit).

    ``air=True`` adds the headspace air of the kinematic path on the solved conduits (docs:
    "Headspace air on the dynamic wave"): the ``"air"`` layer (`f_air`, `f_i`,
    `c_s`, `leak_area` (m2), `leak_cd`, `fans` as in `build_model`) reading the conduits'
    headspace drivers from their link depth, and, with quality and sulfide, the
    ``"air_quality"`` layer of H2S and the two-film transfer `H2STransfer`. It needs a
    dendritic network: a loop, a junction with several outgoing conduits or none (an adverse
    conduit, a junction drained by a regulator) is refused by name
    (`air.check_air_topology`).
    """
    net.validate()
    if net.routing != "dynamic_wave":
        raise ValueError(
            f"build_dynamic_wave_model: network routing is {net.routing!r}, not "
            f"'dynamic_wave'")
    if net.geometry != "tabulated":
        raise ValueError(
            f"build_dynamic_wave_model: network geometry is {net.geometry!r}; dynamic wave "
            f"always routes on SWMM's tabulated circle (xsect_tables.py), so build the "
            f"network with geometry='tabulated' (read_swmm_inp's default)")
    controls = iterate_tolerances(net.dynamic_wave, iterate_tol=iterate_tol,
                                  iterate_max=iterate_max, iterate_flow_tol=iterate_flow_tol)
    iterate_tol, iterate_max = controls["iterate_tol"], controls["iterate_max"]
    iterate_flow_tol = controls["iterate_flow_tol"]
    if air:
        check_air_topology(net)
    if air or quality:
        shaped = [p.name for p in net.pipes if p.xsect is not None and p.xsect.kind != "CIRCULAR"]
        if shaped:
            raise ValueError(
                f"build_dynamic_wave_model: conduit(s) {shaped} have a non-circular cross "
                f"section; the water quality and headspace air models are circular-pipe "
                f"geometry (the sulfide and two-film correlations, the headspace law) and "
                f"are refused for them (build with air=False, quality=False)")
    graph = Network(dtype=F64)
    for manhole in net.manholes:
        graph.add_node(manhole.name, invert=manhole.invert, ground=manhole.ground)
    for outfall in net.outfalls:
        graph.add_node(outfall.name, invert=outfall.invert)
    if air:
        graph.add_node("ambient")
    for pipe in net.pipes:
        graph.add_edge(pipe.u, pipe.v, kind="pipe", name=pipe.name)
    # the regulators after the pipes, kind by kind: the layer's column order
    for kind in STRUCTURE_KINDS:
        for device in net.structures[kind]:
            graph.add_edge(device.u, device.v, kind=kind, name=device.name)
    plan = add_air_edges(graph, net, fans) if air else None
    boundary = [o.name for o in net.outfalls]
    sulfide_air = air and quality and "sulfide" in species
    closure = DynamicWaveHydraulics(graph, net,
                                    quality_layer="water_quality" if quality else None,
                                    air=air)
    if quality and closure.has_wet_losses:
        raise ValueError(
            "build_dynamic_wave_model: the network has conduits that evaporate or seep or "
            "storage units that evaporate or exfiltrate; water quality with those losses is "
            "not modelled (the lost water would have to take its dissolved mass or leave the "
            "transport volumes unchanged, and SWMM's choice is not reproduced) -- build with "
            "quality=False")
    junction_idx = closure.junction_idx
    junction_invert = closure.node_invert[junction_idx]

    def laws():
        return [DynamicWaveConduit(), *(StructureLink(k) for k in closure.structures)]

    layer = PotentialFlowLayer(
        graph, "water", laws(), boundary=boundary,
        node_sources=[closure.storage],
        quantity="head", unit="m", flow_unit="m3/s", analytic_primal=True,
        linear_solver="direct" if len(net.manholes) <= DENSE_SOLVE_MAX_JUNCTIONS else "auto",
        newton_defaults=HEAD_SOLVE_NEWTON)
    if layer.bound.tolist() != closure.outfall_idx.tolist():
        raise ValueError("build_dynamic_wave_model: the layer's boundary order differs from "
                         "the outfall order the closure writes 'water.phi_boundary' in")
    layers: dict = {"water": layer}
    closures: list = [closure]
    tolerances = {"water": iterate_tol}
    parts: dict = {"reactions": [], "drivers": {}, "specs": {}}
    if quality:
        parts = _quality_parts(net, graph, closure, tuple(species), scheme)
        layers["water_quality"] = parts["layer"]
        closures.append(parts["loads"])
        tolerances["water_quality"] = QUALITY_ITERATE_TOL
    if air:
        layers["air"] = air_layer(graph, net, plan, f_air=f_air, f_i=f_i, c_s=c_s,
                                  leak_area=leak_area, leak_cd=leak_cd, fans=fans)
        air_parts = _air_parts(graph, closure, scheme, fans, sulfide=sulfide_air)
        parts["drivers"].update(air_parts["drivers"])
        parts["specs"].update(air_parts["specs"])
        if "layer" in air_parts:
            layers["air_quality"] = air_parts["layer"]
            tolerances["air_quality"] = QUALITY_ITERATE_TOL
            closures.append(H2STransfer(graph.n, closure.junction_idx,
                                        sulfide=tuple(species).index("sulfide"),
                                        out_pipe=parts["out_pipe"]))
    # "before_transport": SWMM reacts a conduit's step-start concentration, then mixes it
    # with the inflow (qualrout.c:337-341), as the kinematic path does
    model = Model(graph, layers, closures=closures, reactions=parts["reactions"],
                  coupling="iterate", reaction_order="before_transport",
                  iterate_tol=tolerances, iterate_max=iterate_max,
                  iterate_relaxation=relaxation, iterate_acceleration=acceleration,
                  iterate_memory=memory,
                  iterate_flow_tol={"water": iterate_flow_tol},
                  iterate_on_limit=controls["on_limit"])
    inflow = torch.zeros(graph.n, dtype=F64)
    inflow[junction_idx] = torch.tensor([m.inflow for m in net.manholes], dtype=F64)
    # a TIMESERIES outfall's file series is its stage unless `outfall_stage` overrides it
    stage_functions = closure.check_outfall_stage(
        {**{o.name: o.stage_series.held for o in net.outfalls
            if o.kind == "timeseries" and o.stage_series is not None},
         **({} if outfall_stage is None else outfall_stage)})
    initial = {"inflow": inflow}
    if bool(closure.outfall_staged.any()):
        initial["outfall_stage"] = closure.outfall_stage(0.0, stage_functions)
    state = closure.initial_state(initial)
    drivers: Drivers = {
        "inflow": inflow,
        "water.phi_boundary": state["water.phi"][closure.outfall_idx].clone(),
    }
    if "outfall_stage" in initial:
        drivers["outfall_stage"] = initial["outfall_stage"]
    if closure.has_wet_losses:
        # the climate at the start of the run; `simulate_dynamic_wave` sets it each step
        start = ClimateClock(None if closure.climate is None else
                             dataclasses.replace(closure.climate, subcatchments=(), gages=()))
        first = start.at(0.0, 0.0)
        drivers[EVAP_DRIVER] = torch.tensor(first.evap * LCF, dtype=F64)
        drivers[HYDCON_DRIVER] = torch.tensor(first.hydcon, dtype=F64)
        if closure.storage_losses is not None:
            drivers[STORAGE_DRIVER] = torch.zeros(len(net.manholes), dtype=F64)
    drivers.update(parts["drivers"])
    if air:
        state.update(air_state(model))
    if quality:
        state.update(quality_state(model, state, drivers))
    closure.notes["routing"] = (
        f"dynamic wave (SWMM 5.2.4 fixed point): heads converged to {iterate_tol} m"
        + f" and flows to {iterate_flow_tol} m3/s"
        + f" within {iterate_max} passes"
        + ("" if acceleration is None else f" ({acceleration}-accelerated)")
        + ("; a step that uses them up is accepted, as SWMM does"
           if controls["on_limit"] == "accept" else ""))
    model.notes = closure.notes
    model.junction_idx = junction_idx
    model.junction_invert = junction_invert
    model.outfall_link_idx = closure.outfall_link
    model.dynamic_wave = closure
    model.outfall_stage = stage_functions
    model.routing_step = float(net.dynamic_wave.routing_step)
    model.pipe_names = list(closure.pipe_names)
    model.link_names = list(closure.link_names)
    model.pipe_columns = PIPE_COLUMNS
    model.input_specs = {
        "inflow": dict(description="lateral inflow at each manhole", unit="m3/s",
                       over="nodes"),
    }
    model.input_specs.update(parts["specs"])
    if closure.has_wet_losses:
        model.input_specs.update({
            EVAP_DRIVER: dict(description="potential evaporation rate (SWMM's Evap.rate; "
                                          "simulate_dynamic_wave sets it each step from the "
                                          "file's climate)", unit="m/s", over="scalar"),
            HYDCON_DRIVER: dict(description="monthly conductivity adjustment (SWMM's "
                                            "Adjust.hydconFactor) of conduit seepage and "
                                            "storage exfiltration", unit="-", over="scalar"),
        })
        if closure.storage_losses is not None:
            model.input_specs[STORAGE_DRIVER] = dict(
                description="evaporation plus exfiltration lost from each storage unit over the "
                            "step (DynamicWaveHydraulics.storage_loss)", unit="m3/s",
                over="junctions")
    if quality or air:
        model.manhole_idx = junction_idx
    if quality:
        if "out_pipe" in parts:
            model.out_pipe = parts["out_pipe"]
    if "outfall_stage" in drivers:
        model.input_specs["outfall_stage"] = dict(
            description="stage elevation of each outfall (TIDAL and TIMESERIES outfalls "
                        "read it; simulate_dynamic_wave sets it each step from the tide "
                        "curve or the outfall_stage functions)",
            unit="m", over=list(closure.outfall_names))
    model.driver_template = dict(drivers)
    model.inflow_schedule = net.inflows
    return model, state, drivers


#: The per-pipe reporting columns of `report.pipe_table` for a dynamic-wave model (all the
#: ones the closure publishes, as on the kinematic path).
PIPE_COLUMNS = ("q", "h", "v", "R_h", "d_m", "T", "A_air", "D_h", "V_wet", "V_air")

#: The quality layer's iteration tolerance (kg/m3), judged alongside the heads.
#: Its concentrations follow the flows of the pass, so it binds only where the heads have
#: converged and the concentrations they carry have not. It is ABSOLUTE: 1e-12 kg/m3 is
#: 1e-9 mg/L, about 1e-11 relative at typical BOD (0.1-0.3 kg/m3) but only ~1e-3 relative
#: for a species at 1 ug/L (1e-9 kg/m3); scale such a species' units up to resolve it.
QUALITY_ITERATE_TOL = 1e-12


class OverflowLoss(Reaction):
    """The contents a flooded junction loses with its overflow (a dynamic-wave quality
    reaction).

    Applied to the quality layer `layer` before the transport step: ``x <- x (1 - f)`` with
    ``f`` the driver ``"<layer>.overflow_loss"`` (per junction, 0..1,
    `DynamicWaveHydraulics`: the overflow over the step, ``dt overflow``, over the
    step-start volume). The transport control volume loses the same water
    (``"sewer.quality_volume"``), so the overflow leaves at the junction's concentration and
    a uniform concentration stays uniform. It is 0 wherever nothing floods. `n_species` is
    the layer's: above 1 its state is (junctions, species)."""

    def __init__(self, n_species: int, layer: str = "water_quality") -> None:
        self.n_species = int(n_species)
        self.layer = layer

    def apply(self, x: Tensor, dt: float | None, drivers: Mapping | None = None) -> Tensor:
        if dt is None:
            return x
        lost = (drivers or {}).get(f"{self.layer}.overflow_loss")
        if lost is None:
            return x
        keep = 1.0 - torch.as_tensor(lost, dtype=F64)
        return x * (keep.unsqueeze(-1) if self.n_species > 1 else keep)


def _quality_parts(net: SewerNetwork, graph: Network, closure: DynamicWaveHydraulics,
                   species: tuple[str, ...], scheme: str) -> dict:
    """The quality pieces of a dynamic-wave model: the layer, the lateral-load
    closure, the reactions, their drivers and input specs.

    ``"water_quality"`` is a `TransportLayer` over the junctions (the outfalls its boundary)
    on every edge kind of the head layer, so its flows are the solved ``"water.q"``
    (`Model.flow_layer_of`), upwind by their sign; an orifice, weir or outlet carries water
    between junctions with no volume of its own (SWMM gives a non-conduit link its upstream
    node's quality, qualrout.c:282-288). The closure writes its capacity, each junction's
    transport control volume (`DynamicWaveHydraulics`, "Quality"), and `OverflowLoss` takes
    a flooded junction's overflow out of it. Each species ``s`` gets the lateral load
    ``inflow * <s>_in`` (`LateralLoads`, driver ``"<s>_in"``, kg/m3). ``"bod"`` decays and
    ``"sulfide"`` is generated (`SulfideGeneration`); any other species is conservative.
    Sulfide needs ``"bod"`` and each junction's unique outgoing conduit (its hydraulic radius,
    speed and mean depth), and is refused by name at the junctions with none or several (a
    loop, an adverse conduit, a junction drained by a regulator)."""
    if not species or len(set(species)) != len(species):
        raise ValueError(f"build_model: species must be distinct names, got {species!r}")
    if "sulfide" in species and "bod" not in species:
        raise ValueError(
            f"build_model: species 'sulfide' needs 'bod' (Pomeroy-Parkhurst generation reads "
            f"the BOD), got {species!r}")
    junctions = list(closure.junction_names)
    k = len(species)
    layer = TransportLayer(
        graph, "water_quality", capacity=torch.ones(len(junctions), dtype=F64),
        flow_kind=closure.edge_kinds, boundary=list(closure.outfall_names),
        n_species=k, scheme=scheme, quantity="concentration", unit="kg/m3",
        species_names=species, source_unit="kg/s", capacity_unit="m3", flow_unit="m3/s")
    ordered = [graph.nodes[i] for i in layer.interior_idx.tolist()]
    if ordered != junctions:
        raise ValueError(
            f"build_model: the water_quality layer's active interior came out as {ordered}, "
            f"not the junction order {junctions}; the capacity is written in junction order")
    n_out = len(closure.outfall_names)
    drivers: Drivers = {
        "water_quality.x_boundary": torch.zeros((n_out,) if k == 1 else (n_out, k),
                                                dtype=F64),
        **{f"{name}_in": torch.zeros(graph.n, dtype=F64) for name in species},
    }
    specs = {f"{name}_in": dict(description=f"{name} concentration of the lateral inflow",
                                unit="kg/m3", over="nodes") for name in species}
    parts: dict = {
        "layer": layer, "reactions": [("water_quality", OverflowLoss(k))], "drivers": drivers,
        "specs": specs,
        "loads": LateralLoads(graph.n, columns={i: f"{name}_in"
                                                for i, name in enumerate(species)},
                              n_species=k)}
    if "bod" not in species:
        return parts
    sulfide = species.index("sulfide") if "sulfide" in species else None
    out_pipe = None
    if sulfide is not None:
        outgoing: dict[str, list[int]] = {name: [] for name in junctions}
        for i, pipe in enumerate(net.pipes):
            if pipe.u in outgoing:
                outgoing[pipe.u].append(i)
        bad = [name for name in junctions if len(outgoing[name]) != 1]
        if bad:
            raise ValueError(
                f"build_model: sulfide generation reads each junction's outgoing conduit, "
                f"and junction(s) {bad} have none or several (a loop, an adverse conduit or "
                f"a regulator); build quality without 'sulfide', e.g. species=('bod',)")
        out_pipe = torch.tensor([outgoing[name][0] for name in junctions], dtype=torch.long)
        drivers["sewer.q_slope"] = torch.tensor(
            [net.pipes[i].slope for i in out_pipe.tolist()], dtype=F64)
        specs["sewer.q_slope"] = dict(
            description="slope of each junction's outgoing conduit", unit="m/m",
            over=junctions, ordering="junction order")
        parts["out_pipe"] = out_pipe
    drivers["T_water"] = torch.tensor(18.0, dtype=F64)
    specs["T_water"] = dict(description="wastewater temperature", unit="degC")
    parts["reactions"].insert(0, ("water_quality", SulfideGeneration(
        bod=species.index("bod"), sulfide=sulfide, out_pipe=out_pipe,
        manhole_idx=closure.junction_idx, n_nodes=graph.n)))
    return parts


def _air_parts(graph: Network, closure: DynamicWaveHydraulics, scheme: str, fans: tuple,
               *, sulfide: bool) -> dict:
    """The air pieces of a dynamic-wave model, as the kinematic builder's: the
    drivers (``"air.phi_boundary"``, the temperatures ``"T_head"`` and ``"T_amb"`` (K) and,
    with `sulfide`, the pH ``"pH"`` and the ``"air_quality.x_boundary"`` of H2S), their input
    specs and, with `sulfide`, the ``"air_quality"`` layer. The air layer carries H2S only, so
    it is built where the water carries sulfide (`H2STransfer` couples them)."""
    drivers: Drivers = {"air.phi_boundary": torch.zeros(1, dtype=F64),
                        "T_head": torch.tensor(293.15, dtype=F64),
                        "T_amb": torch.tensor(283.15, dtype=F64)}
    specs = {"T_head": dict(description="headspace air temperature", unit="K"),
             "T_amb": dict(description="ambient air temperature", unit="K")}
    parts: dict = {"drivers": drivers, "specs": specs}
    if not sulfide:
        return parts
    junctions = list(closure.junction_names)
    kinds = ("headspace", "leak", "fan") if fans else ("headspace", "leak")
    boundary = ["ambient"] + list(closure.outfall_names)
    interior, _ = active_interior(graph, kinds, boundary)
    layer = TransportLayer(
        graph, "air_quality", capacity=torch.ones(interior.numel(), dtype=F64),
        flow_kind=kinds, boundary=boundary, n_species=1, scheme=scheme,
        quantity="concentration", unit="kg/m3", species_names=("h2s",),
        source_unit="kg/s", capacity_unit="m3", flow_unit="m3/s")
    ordered = [graph.nodes[i] for i in layer.interior_idx.tolist()]
    if ordered != junctions:
        raise ValueError(
            f"build_model: the air_quality layer's active interior came out as {ordered}, "
            f"not the junction order {junctions}; H2STransfer reads 'air_quality.x' as a "
            f"per-junction array")
    drivers["air_quality.x_boundary"] = torch.zeros(len(boundary), dtype=F64)
    drivers["pH"] = torch.tensor(7.0, dtype=F64)
    specs["pH"] = dict(description="wastewater pH", unit="-")
    parts["layer"] = layer
    return parts


def air_state(model: Model) -> State:
    """The air layer's initial state: zero pressures and flows (the quiescent headspace)."""
    layer = model.potential["air"]
    return {"air.phi": torch.zeros(model.net.n, dtype=F64),
            "air.q": torch.zeros(len(layer.cols), dtype=F64)}


def quality_state(model: Model, state: State, drivers: Drivers) -> State:
    """The transport layers' initial state at the hydraulic `state`: clean water and clean
    air (``"<layer>.x"`` zero, SWMM's default initial concentration) and the capacities at
    `state` (`Model.initial_capacities`), which the amount form of the first step needs."""
    out = {}
    for name, layer in model.transport.items():
        shape = (layer.n_i,) if layer.n_species == 1 else (layer.n_i, layer.n_species)
        out[f"{name}.x"] = torch.zeros(shape, dtype=F64)
    out.update(model.initial_capacities(dict(state, **out), drivers))
    return out


def solve_atol(model: Model, dt: float | None = None) -> float:
    """The head solve's default absolute tolerance (m3/s) for steps of `dt` s (default: the
    routing step): `SOLVE_ATOL`, raised to 4x the round-off floor of a junction on its stiff
    bed where that is larger. There the Jacobian slope is ``(2/dt)(A_s + K_BED A_bed)``,
    S ~ (2/dt) K_BED A_bed, and Newton cannot move the head by less than half an ulp,
    ``<= eps |H| / 2``, so the residual cannot go below about ``S eps |H| / 2`` -- 6e-10 m3/s
    on a 12 m invert at SWMM's default 12.566 ft2 floor and a 5 s step, giving an atol of
    2.5e-9 m3/s. A wet junction's head is then solved to within ``atol dt / (2 A_s)`` m, at
    most 5e-9 m in that case."""
    dt = model.routing_step if dt is None else float(dt)
    storage = next(ns for ns in model.layers["water"].node_sources
                   if isinstance(ns, JunctionStorage))
    slope = 2.0 * storage.bed_area * (1.0 + K_BED) / dt
    height = float(storage.bed.abs().max())
    floor = slope * torch.finfo(F64).eps * height / 2.0
    return max(SOLVE_ATOL, _BED_FLOOR_MARGIN * floor)


def routing_step_size(hyd: DynamicWaveHydraulics, before: Mapping | None, after: Mapping,
                      previous_dt: float | None) -> float:
    """SWMM's next routing step (s): `dynwave.c::dynwave_getRoutingStep` (dynwave.c:195-220).

    `after` is the state at the end of the step just taken (``previous_dt`` s) and `before`
    the state at its start; ``before=None`` asks for the very first step of the run. With
    ``options.variable_step == 0`` (or a routing step under MINTIMESTEP) the step is the
    fixed ``options.routing_step``; otherwise the first step is ``min_route_step``, and the
    others the smallest of

    - the routing step;
    - each conduit's Courant step ``V / |q| * Fr / (1 + Fr) * variable_step``
      (`getLinkStep`, dynwave.c:836-878), skipped where ``|q|`` or the mid-depth area is
      within FUDGE or ``Fr <= 0.01`` -- a dry, still or full closed conduit never limits
      the step. Flow and volume are one barrel's, and a lengthened conduit's step is
      scaled by its modified length over its own, ``modLength / length``;
    - each junction's depth-change step ``0.25 (crown - invert) / (|dy|/dt)``
      (`getNodeStep`, dynwave.c:881-921), skipped at outfalls, dry junctions, junctions
      within FUDGE of their crown or above it, and where ``|dy|/dt`` is within FUDGE;

    floored at `min_route_step` (dynwave.c:828) and truncated to a millisecond
    (dynwave.c:218). The Froude number, flow, volume and mid-depth area are those of the
    converged state `after` -- SWMM's last trial of the step, equal to it at its
    convergence; ``|dy|`` is ``|y_after - y_before|`` with the depths in `before` and
    `after` (``sewer.junction_depth``), SWMM's ``Xnode.dYdT`` (dynwave.c:758).
    Not differentiable (the step is a switch of the state). Single network only (the
    state's tensors are 1-D).
    """
    opts = hyd.options
    fixed = opts.routing_step
    if opts.variable_step == 0.0 or fixed < MIN_TIME_STEP:                   # dynwave.c:204-205
        return fixed
    if before is None:                                                       # dynwave.c:209-212
        step = opts.min_route_step
    else:
        step = _variable_step(hyd, before, after, float(previous_dt), fixed)
    return math.floor(1000.0 * step) / 1000.0                                # dynwave.c:218


def _variable_step(hyd: DynamicWaveHydraulics, before: Mapping, after: Mapping,
                   dt: float, max_step: float) -> float:
    """`dynwave.c::getVariableStep` (dynwave.c:799-833), `max_step` s, `dt` the last step."""
    with torch.no_grad():
        terms = hyd._terms(after, dt)
        barrels = hyd.geo.barrels
        q = after[f"{hyd.layer}.q"][..., :hyd.n_pipes].abs() / QCF / barrels  # dynwave.c:852
        fr = terms.froude
        live = (q > FUDGE) & (terms.mid_area > FUDGE) & (fr > 0.01)          # dynwave.c:853-856
        residence = (terms.volume / barrels
                     / torch.where(live, q, torch.ones_like(q)))             # dynwave.c:859
        residence = residence * hyd.geo.mod_length / hyd.geo.length          # dynwave.c:865
        courant = residence * fr / (1.0 + fr) * hyd.options.variable_step    # dynwave.c:861-862
        t_link = torch.where(live, courant, torch.full_like(courant, max_step))
        t_min = min(max_step, float(t_link.min())) if t_link.numel() else max_step
        depth = after["sewer.junction_depth"] / LCF
        rate = (depth - before["sewer.junction_depth"] / LCF).abs() / dt     # dynwave.c:758
        crown = hyd.crown_ft
        allowed = 0.25 * crown                                               # dynwave.c:904
        live = ((depth > FUDGE) & (depth + FUDGE < crown) & (allowed >= FUDGE)
                & (rate >= FUDGE))                                           # dynwave.c:897-910
        t_node = allowed / torch.where(live, rate, torch.ones_like(rate))    # dynwave.c:913
        t_node = torch.where(live, t_node, torch.full_like(t_node, t_min))
        if t_node.numel():
            t_min = min(t_min, float(t_node.min()))
    return max(t_min, hyd.options.min_route_step)                            # dynwave.c:828


def simulate_dynamic_wave(model: Model, state: State, drivers: Drivers, *,
                          steps: int | None = None, duration: float | None = None,
                          dt: float | None = None,
                          inflow: Tensor | Callable[[float], Tensor] | None = None,
                          outfall_stage: Mapping | None = None,
                          record: tuple[str, ...] = ("water.phi", "water.q"),
                          diagnostics: list | None = None, **solve_kwargs
                          ) -> tuple[State, dict[str, Tensor]]:
    """Run the dynamic wave from `state` for `steps` steps, or for `duration` s.

    Exactly one of `steps` and `duration` is required. Under a variable step one call is one
    run: the first step is the minimum step and `t` starts at 0, so a continuation call
    restarts as a new SWMM run would (the step rule needs the previous step's start state).
    The step is `dt` s (default: the model's routing step) when
    ``options.variable_step == 0``; under ``variable_step > 0``
    each step is SWMM's variable step (`routing_step_size`: the first is the minimum step)
    and `dt` is refused, since it would be a second step rule. `duration` (s) is SWMM's
    run length: steps are taken until the elapsed time (kept in milliseconds, as SWMM does,
    `swmm5.c:530-545`) reaches it, and the last step is cut to the time that remains (at
    least 1 ms), so the run ends exactly at `duration` whatever the step rule.

    `inflow` (m3/s, full-node) overrides ``drivers["inflow"]``; without it a model built from
    a SWMM file with time series, patterns or dry-weather flow (``model.inflow_schedule``,
    `noodl.apps.sewer.inflows`) uses that schedule, and any other model the constant
    ``drivers["inflow"]``. ``(n,)`` is one constant
    inflow; ``(B, n)`` a batch of B constant inflows -- a 2-D inflow is ALWAYS a batch,
    never a series; ``(steps, B, n)`` the inflow of each step for a batch of B (one
    network's series is ``(steps, 1, n)``; needs `steps`); or a callable ``inflow(t)``
    returning ``(n,)`` or ``(B, n)`` for the step that starts at `t` s (a float). SWMM
    takes a step's lateral inflow from its time series at the START of the step
    (`routing.c:223, 368`: the date is read before the routing clock advances; the inflow
    enters the trapezoid with the step's end state, and the previous step's final net
    inflow is its start), so ``inflow(t)`` is called once per step, with the step's start
    time. (SWMM's own clock reads 1 ms late, ``getDateTime`` adds 1 ms: `swmm5.c:1551`, so
    a [TIMESERIES] is sampled at ``t + 0.001`` s; a function standing in for one must add
    that, as the verification test does.) A callable with a true ``step_aware`` attribute
    (the runoff driver of a file with subcatchments, `noodl.apps.sewer.runoff`) is called
    ``inflow(t, step)``, with the step's length in s as well, because SWMM takes the runoff
    steps a routing step needs up to its end time (`execRouting`, swmm5.c:546). A batched
    inflow expands an unbatched `state` to the batch (a batch cannot be combined with a
    variable step: each member would need its own). After every
    step the carried keys are re-evaluated at the returned state. `record` may also name
    ``"sewer.overflow"`` (m3/s per junction, SWMM's flooding loss over each step:
    `DynamicWaveHydraulics.overflow`), which is not a state key, and the closure's
    reporting keys (``"sewer.storage_volume"``, ``"sewer.V_wet"``, ``"sewer.h"``, ...), from
    its query at each step's end state.
    `outfall_stage` ``{outfall name: stage (m) or a function of t (s) returning it}`` gives
    the TIMESERIES outfalls their stage elevation (default: the model's own, the
    `build_model` argument, which its names override one by one); a TIDAL outfall reads its
    tide curve (`tidal_stage`). Both become the driver ``"outfall_stage"`` (m, per outfall) of
    each step, and share the
    FIXED outfall's stage rule. SWMM sets the outfall depth with the routing clock already
    advanced (`routing.c:301-302`; node.c:1449, 1456), so the stage is that at the END of
    the step: a function is called once per step with the step's end time in s -- not the
    start time `inflow` gets, and with no 1 ms offset (`NewRoutingTime`, not `getDateTime`).
    It works under fixed and variable steps alike.
    With pumps or control rules (``hyd.controlled``) every step starts as SWMM's does
    (`linkcontrol.LinkControl`): the pumps' startup and shutoff depths, the rules (at the
    RULE_STEP, which also shortens the routing steps to end at each rule time) and the
    orifice opening rate move the links' settings, carried as ``"sewer.setting"``,
    ``"sewer.target_setting"`` and ``"sewer.time_last_set"`` (recordable like any state
    key; ``record=("water.q", "sewer.setting")`` reports each link's flow and setting).
    Settings are discrete, not differentiable.
    `diagnostics`, when a list, receives each step's `Model.step` diagnostics.
    `solve_kwargs` reach the head solve (Newton); `atol` defaults to `solve_atol` for each
    step's length (m3/s: `SOLVE_ATOL`, or the bed's round-off floor where larger) and
    `rtol` to 0, because Newton's own default (sqrt(eps), 1.5e-8 m3/s) would stop a
    warm-started pass before it moves and leave the junction balance off by that much.
    Returns the final state and the recorded `record` keys stacked over steps (leading
    dimension: the steps), plus ``"t"`` (s, the END time of each step), ``"dt"`` (s, each
    step's length) and ``"converged"`` (bool per step, every batch instance's passes
    converged): under the file's own MAX_TRIALS (`iterate_tolerances`) a step that uses
    them up is taken, as SWMM takes it, and is False here; ``(~out["converged"]).sum()``
    is SWMM's count of non-converging steps.
    """
    hyd = model.dynamic_wave
    if (steps is None) == (duration is None):
        raise ValueError("simulate_dynamic_wave: give exactly one of steps and duration "
                         f"(got steps={steps!r}, duration={duration!r})")
    if duration is not None and not duration > 0:
        raise ValueError(f"simulate_dynamic_wave: duration must be > 0 s, got {duration!r}")
    variable = hyd.options.variable_step > 0
    if variable and dt is not None:
        raise ValueError(
            f"simulate_dynamic_wave: dt={dt!r} s given, but options.variable_step = "
            f"{hyd.options.variable_step} chooses every step (SWMM's variable routing step); "
            f"set variable_step to 0 for a fixed dt")
    fixed_dt = model.routing_step if dt is None else float(dt)
    staged = bool(hyd.outfall_staged.any())
    stage_functions = hyd.check_outfall_stage(
        {**model.outfall_stage, **({} if outfall_stage is None else outfall_stage)})
    atol_given = "atol" in solve_kwargs
    solve_kwargs.setdefault("rtol", 0.0)
    n = model.net.n
    series = None
    schedule = getattr(model, "inflow_schedule", None)
    if inflow is None and schedule is not None:      # the file's [INFLOWS] and [DWF]
        inflow = schedule.driver(list(model.net.nodes))
    if callable(inflow):
        step_inflow = inflow
    elif inflow is None:
        step_inflow = None
    else:
        series = torch.as_tensor(inflow, dtype=F64)
        if series.dim() not in (1, 2, 3) or series.shape[-1] != n:
            raise ValueError(
                f"simulate_dynamic_wave: inflow must be (n,), (B, n) or (steps, B, n) with "
                f"n = {n} nodes, got shape {tuple(series.shape)}")
        if series.dim() == 3:
            if steps is None:
                raise ValueError("simulate_dynamic_wave: an inflow series (steps, B, n) "
                                 "needs steps=, not duration=")
            if series.shape[0] < steps:
                raise ValueError(f"simulate_dynamic_wave: inflow has {series.shape[0]} "
                                 f"steps, {steps} asked")
        step_inflow = None

    step_aware = bool(getattr(step_inflow, "step_aware", False))
    clock = None
    if hyd.has_wet_losses:        # the evaporation and seepage losses read the file's climate
        shared = getattr(step_inflow, "model", None)
        clock = ClimateClock(hyd.climate, shared if isinstance(shared, RunoffModel) else None)
        hyd.reset_losses()

    def inflow_at(k: int, t: float, step: float | None = None) -> Tensor | None:
        if series is not None:
            return series[k] if series.dim() == 3 else series
        if step_inflow is None:
            return None
        value = torch.as_tensor(step_inflow(t, step) if step_aware else step_inflow(t),
                                dtype=F64)
        if value.dim() not in (1, 2) or value.shape[-1] != n:
            raise ValueError(
                f"simulate_dynamic_wave: inflow(t={t}) must return (n,) or (B, n) with "
                f"n = {n} nodes, got shape {tuple(value.shape)}")
        return value

    first_step = None
    if step_aware:          # a runoff driver needs the step's length: the first step's
        first_step = (routing_step_size(hyd, None, state, None) if variable else fixed_dt)
        if duration is not None and 1000.0 * first_step > 1000.0 * duration:
            first_step = max(float(duration), 1.0 / 1000.0)
    first = inflow_at(0, 0.0, first_step)   # step 0's inflow, reused in the loop; it also
    #                                         sets the batch
    batch: tuple[int, ...] = ()
    if first is not None and first.dim() > 1:
        batch = tuple(first.shape[:-1])
    if variable and batch:
        raise ValueError("simulate_dynamic_wave: a batched inflow cannot be combined with "
                         "options.variable_step > 0 (one step for every member)")
    state = dict(state)
    if batch:
        # an unbatched key is 1-D, but a multi-species concentration (junctions, species)
        unbatched = {f"{name}.x": 2 for name, layer in model.transport.items()
                     if layer.n_species > 1}
        state = {k: (v.expand(batch + v.shape).clone() if v.dim() == unbatched.get(k, 1)
                     else v)
                 for k, v in state.items()}
    rows: dict[str, list[Tensor]] = {key: [] for key in record}
    ends: list[float] = []
    lengths: list[float] = []
    converged: list[Tensor] = []
    now_ms = 0.0                                          # swmm5.c: NewRoutingTime, ms
    end_ms = None if duration is None else 1000.0 * float(duration)
    previous: tuple[State, float] | None = None
    control = LinkControl(hyd) if hyd.controlled else None
    lateral = None
    k = 0
    while (k < steps) if steps is not None else (now_ms < end_ms):
        if variable:
            step = routing_step_size(hyd, None if previous is None else previous[0], state,
                                     None if previous is None else previous[1])
        else:
            step = fixed_dt
        if control is not None:                                             # routing.c:189-195
            step = control.rule_step_limit(now_ms, step)
        next_ms = now_ms + 1000.0 * step                                    # swmm5.c:530
        if end_ms is not None and next_ms > end_ms:                         # swmm5.c:531-536
            step = max((end_ms - now_ms) / 1000.0, 1.0 / 1000.0)
            next_ms = end_ms
        t = now_ms / 1000.0
        if inflow is not None:
            drivers = dict(drivers, inflow=first if k == 0 else inflow_at(k, t, step))
        if staged:                                    # node.c:1449, 1456: the step's END
            drivers = dict(drivers,
                           outfall_stage=hyd.outfall_stage(next_ms / 1000.0, stage_functions))
        kwargs = dict(solve_kwargs)
        if not atol_given:
            kwargs["atol"] = solve_atol(model, step)
        diag: dict = {}
        if control is not None:      # the pump limits and rules, on the step-start state
            state = control.apply(state, now_ms, step, lateral)
        before = state
        if clock is not None:                                      # routing.c:363-365
            climate = clock.at(now_ms, next_ms)
            drivers = dict(drivers, **{
                EVAP_DRIVER: torch.tensor(climate.evap * LCF, dtype=F64),
                HYDCON_DRIVER: torch.tensor(climate.hydcon, dtype=F64)})
            lost = hyd.storage_loss(state, step, climate)
            if lost is not None:
                drivers[STORAGE_DRIVER] = lost
        try:
            state = model.step(state, drivers, step, t=t, diagnostics=diag, **kwargs)
        except RuntimeError as err:
            raise RuntimeError(f"dynamic wave step at t={t} s: {err}") from err
        # resync, as modelica/run.py does, with the step's drop-end hold
        state.update(hyd.carried(state, drivers, dt=step, start=before))
        previous = (before, step)
        lateral = drivers["inflow"]
        if control is not None:
            hyd.check_pump_inlets(state)
        now_ms = next_ms
        k += 1
        ends.append(now_ms / 1000.0)
        lengths.append(step)
        converged.append(torch.as_tensor(diag["converged"]).all())
        if diagnostics is not None:
            diagnostics.append(diag)
        report = None
        for key in record:
            if key == "sewer.overflow":
                rows[key].append(hyd.overflow(before, state, drivers, dt=step))
            elif key in state:
                rows[key].append(state[key])
            else:
                if report is None:
                    report = hyd(state, drivers)
                if key not in report:
                    raise KeyError(f"simulate_dynamic_wave: record names {key!r}, which is "
                                   f"neither a state key nor a closure reporting key")
                rows[key].append(report[key])
    out = {key: torch.stack(v) for key, v in rows.items()}
    out["t"] = torch.tensor(ends, dtype=F64)
    out["dt"] = torch.tensor(lengths, dtype=F64)
    out["converged"] = torch.stack(converged)
    return state, out

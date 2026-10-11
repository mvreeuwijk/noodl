"""SWMM 5.2.4 dynamic-wave routing for the sewer application.

`DynamicWaveOptions` holds the `[OPTIONS]` that change the physics. The model is a
`PotentialFlowLayer` ``"water"`` (quantity ``"head"``, m) whose unknowns are the junction
heads: every conduit is a `DynamicWaveConduit`, q = c0 + k (H1 - H2) with the coefficients of
SWMM's momentum update (`dynwave_terms.conduit_terms`) frozen for one pass, and every junction
carries a `JunctionStorage`, SWMM's trapezoidal continuity
``A_s (y - y_start) = 0.5 dt (N_start + N)``. `DynamicWaveHydraulics` (a closure) refreshes
the coefficients, surface areas and outfall heads from the previous pass, and
``Model(coupling="iterate")`` repeats the pass until the heads stop changing: the fixed point
is SWMM's own converged trial.

This module must not import `network.py` at module level: `network.py` imports
`DynamicWaveOptions` and `build_dynamic_wave_model` from here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from noodl.apps.sewer.dynwave_terms import (
    DAMPING,
    FUDGE,
    NORMAL_FLOW_LIMITED,
    SURCHARGE_METHODS,
    ConduitGeometry,
    ConduitTerms,
    conduit_terms,
)
from noodl.apps.sewer.xsect_tables import (
    LCF,
    QCF,
    area_of_depth,
    critical_depth,
    normal_depth_ft,
)
from noodl.elements.base import Element
from noodl.layers.potential import PotentialFlowLayer
from noodl.model import Model, StepContext
from noodl.nodesources import NodeSource
from noodl.topology import Network

if TYPE_CHECKING:
    from noodl.apps.sewer.network import SewerNetwork

Tensor = torch.Tensor
F64 = torch.float64
State = dict[str, Tensor]
Drivers = dict[str, Tensor]

#: SWMM's dry-node flow cap (`dwflow.c:273-276`): 1e-4 cfs, in m3/s.
FUDGE_FLOW = 1.0e-4 * QCF
#: Bed stiffening factor of `JunctionStorage` (design D3): below its invert a junction's
#: storage slope gains ``(2 A_bed/dt) K_BED``, standing in for SWMM's ``yNew < 0 -> 0``.
K_BED = 1.0e6
#: Stiffness factor of an EXTRAN junction's hold band at its crown (design E1, the tie rule
#: in `JunctionStorage`): the band rises with ``(2 A_bed/dt) K_HOLD``. It trades the band's
#: width ``|w_c| dt / (2 A_bed K_HOLD)`` (0.6 mm at w_c = 0.26 m3/s, SWMM's default surface
#: area floor and a 5 s step) against the head solve's round-off floor on it, ``(2 A_bed/dt)
#: K_HOLD eps |H| / 2`` (3e-13 m3/s at 12 m), which must stay below the 1e-12 m3/s a
#: surcharged run's parity needs; the bed's K_BED would put that floor at 6e-10 m3/s.
K_HOLD = 1.0e3
#: The band is also at least `K_HOLD_STORAGE` times the junction's own storage slope
#: ``2 A_s/dt``, so that above the crown it always rises faster than the storage line it
#: starts from (a large A_s, e.g. a trunk junction or a large user MIN_SURFAREA, would
#: otherwise outgrow ``K_HOLD A_bed``). Its round-off floor then scales with A_s: 6e-12
#: m3/s at A_s = 1200 m2, 5 s and 12 m, above a 1e-12 m3/s head-solve atol there.
K_HOLD_STORAGE = 10.0
#: A surcharged junction's anchor slope floor (design E1): ``K_ANCHOR (2 A_s/dt)`` (m2/s) is
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
#: storage slope.
K_SWITCH = 1.0
#: Newton's absolute residual tolerance for the head solve (m3/s): a junction balance error
#: of `SOLVE_ATOL` dt m3 at most, far below the head tolerance times any surface area. It is
#: raised to the bed's round-off floor where that is larger (`solve_atol`).
SOLVE_ATOL = 1.0e-12
#: `solve_atol`'s margin over the bed's round-off floor (see there).
_BED_FLOOR_MARGIN = 4.0
#: SWMM's default junction surface-area floor (`dynwave.c:62`), ft2.
DEFAULT_MIN_SURFACE_AREA_FT2 = 12.566
#: SI area per SWMM ft2 (design D2): makes ``A_s dy = 0.5 dt N`` exact in m, m3/s.
_AREA_SI = QCF / LCF
#: SWMM's link velocity is 0 at or below this link depth (`link.c:834`), ft.
_VELOCITY_MIN_DEPTH = 0.01
_OUTFALL_CODES = {"free": 0, "normal": 1, "fixed": 2}


@dataclass(frozen=True)
class DynamicWaveOptions:
    """SWMM 5.2.4's dynamic-wave [OPTIONS] that change the physics, with SWMM's defaults
    (`project.c:841-880`). `head_tolerance` and `max_trials` are SWMM's own iteration
    controls: recorded so a file round-trips, never used -- noodl solves the same
    equations to `build_model`'s own `iterate_tol` (docs: "Dynamic wave").

    `routing_step` is the fixed routing step (s); `inertial_damping` is ``"none"``,
    ``"partial"`` or ``"full"``; `normal_flow_limited` is ``"slope"``, ``"froude"``,
    ``"both"`` or ``"none"``; `surcharge_method` is ``"extran"`` (end depths capped at the
    crown, a junction above its crown holds continuity without storage) or ``"slot"``
    (Preissmann slot);
    `min_surface_area` is the junction surface-area floor (m2), ``None`` for SWMM's
    12.566 ft2; `head_tolerance` is SWMM's head convergence tolerance (m); `max_trials` is
    SWMM's trial limit (-).
    """

    routing_step: float = 20.0
    inertial_damping: str = "partial"
    normal_flow_limited: str = "both"
    surcharge_method: str = "extran"
    min_surface_area: float | None = None
    head_tolerance: float | None = None
    max_trials: int | None = None

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
        if self.min_surface_area is not None and not self.min_surface_area > 0:
            raise ValueError("DynamicWaveOptions: min_surface_area must be > 0 m2")


class DynamicWaveConduit(Element):
    """SWMM's dynamic-wave conduit (`dwflow.c`) with its coefficients frozen for one pass.

    ``q = c0 + k dp`` (m3/s, dp = H1 - H2 in m), then ``min(q, q_norm)`` where SWMM's
    normal-flow check passed (`dwflow.c:255-258`), then the dry-node caps in SWMM's order
    (`dwflow.c:273-276`): at most +`FUDGE_FLOW` out of a dry upstream node, at least
    -`FUDGE_FLOW` out of a dry downstream node.

    The normal-flow slope switch (the tie rule). The check's flag is lagged (design D4):
    evaluated by the closure on the fed heads and flow. Where it is SWMM's slope test ``y1 <
    y2`` alone, with both end depths inside their clamps, the test is ``dp < dz`` (dz = z1 -
    z2, the conduit-end inverts), a switch on this element's own argument, and it is applied
    within the pass instead, as a continuous, non-decreasing curve::

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
    """

    def __init__(self, kind: str = "pipe") -> None:
        super().__init__(kind)

    def _parts(self, dp: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor, Tensor]:
        """The momentum line, the flow and the switch's wall (NaN where no wall acts)."""
        lin = drivers["water.flow_offset"] + drivers["water.conveyance"] * dp
        q_norm = drivers["water.normal_flow"]
        limited = drivers["water.limited"] > 0.5
        switch = drivers["water.slope_switch"] > 0.5
        wall = (drivers["water.flow_last"]
                + drivers["water.switch_slope"] * (dp - drivers["water.switch_drop"]))
        slid = torch.minimum(lin, torch.maximum(q_norm, wall))
        q = torch.where(limited, torch.minimum(lin, q_norm),
                        torch.where(switch, slid, lin))                    # dwflow.c:255-258
        up = drivers["water.dry_upstream"] > 0.5
        dn = drivers["water.dry_downstream"] > 0.5
        q = torch.where(up, torch.clamp(q, max=FUDGE_FLOW), q)      # dwflow.c:273-274
        q = torch.where(dn, torch.clamp(q, min=-FUDGE_FLOW), q)     # dwflow.c:275-276
        on_wall = switch & ~limited
        return lin, q, torch.where(on_wall, wall, torch.full_like(wall, torch.nan))

    def flow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Conduit flow (m3/s, positive from the pipe's first node to its second)."""
        return self._parts(dp, drivers)[1]

    def dflow(self, dp: Tensor, drivers: Mapping | None = None) -> Tensor:
        """dq/d(dp) (m2/s): k where the flow is the momentum line, the wall slope S where it
        is on the switch's wall, else 0 (every other limit is a constant, so ``q == lin``
        and ``q == wall`` are exact tests)."""
        lin, q, wall = self._parts(dp, drivers)
        k = drivers["water.conveyance"].expand_as(q)
        s = drivers["water.switch_slope"].expand_as(q)
        zero = torch.zeros_like(k)
        return torch.where(q == lin, k, torch.where(q == wall, s, zero))


class JunctionStorage(NodeSource):
    """SWMM's junction continuity as a node withdrawal: trapezoidal storage over the step,
    with a stiff bed below the invert (design D3), or no storage at all where the junction
    is EXTRAN-surcharged (design E1), capped by flooding at the junction's top (design E3).

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

    EXTRAN surcharge (design E1; `crown` given, the junction crown heads in m). Above its
    crown a junction stores nothing: SWMM's surcharged update ``y_new = y_last + corr
    dQ/denom`` (`dynwave.c:716-741`) is at its fixed point exactly when the net inflow dQ is
    0. With ``w_c = (2 A_s/dt)(H_crown - H_start) - N_start`` the storage withdrawal at the
    crown, ``stiff = (2/dt) max(K_HOLD A_bed, K_HOLD_STORAGE A_s)`` and the band
    ``b(H) = w_c + stiff (H - H_crown)``:

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

    Flooding (design E3; `top` given, the junction heads at y_max = full depth + SurDepth,
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

    Reads the per-junction drivers ``"water.storage_area"`` A_s (m2, design D2's SI area),
    ``"water.head_start"`` (m), ``"water.net_inflow_start"`` (m3/s) and
    ``"water.storage_rate"`` 1/dt (1/s); with `crown` also ``"water.surcharged"`` (0/1)
    and ``"water.surcharge_slope"`` (m2/s), and `surcharged` reads ``"water.head_last"``
    (m); with `top` also ``"water.flood_excess"`` (m3/s), and `flood_excess` reads
    ``"water.net_inflow_last"`` (m3/s). `bed` holds the junction inverts (m); without
    `crown` (SLOT) no junction surcharges, without `top` none floods.
    """

    def __init__(self, nodes: Tensor, bed: Tensor, bed_area: float,
                 crown: Tensor | None = None, top: Tensor | None = None) -> None:
        super().__init__(nodes)
        self.register_buffer("bed", torch.as_tensor(bed, dtype=F64))
        self.bed_area = float(bed_area)
        self.extran = crown is not None
        if self.extran:
            self.register_buffer("crown", torch.as_tensor(crown, dtype=F64))
        self.floods = top is not None
        if self.floods:
            self.register_buffer("top", torch.as_tensor(top, dtype=F64))

    def _storage(self, phi: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor, Tensor]:
        """Phase A's storage withdrawal (m3/s) and slope (m2/s) at `phi`, and the band's
        stiffness (m2/s)."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        area = drivers["water.storage_area"]
        bed = rate2 * self.bed_area * K_BED
        dry = torch.clamp(phi - self.bed, max=0.0)
        storage = (rate2 * area * (phi - drivers["water.head_start"])
                   - drivers["water.net_inflow_start"] + bed * dry)          # dynwave.c:694-711
        slope = rate2 * area + bed * (phi < self.bed).to(phi.dtype)
        return storage, slope, self._stiff(drivers)

    def _stiff(self, drivers: Mapping) -> Tensor:
        """The band's slope (m2/s, per junction): ``(2/dt) max(K_HOLD A_bed,
        K_HOLD_STORAGE A_s)``."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        return rate2 * torch.clamp(K_HOLD_STORAGE * drivers["water.storage_area"],
                                   min=K_HOLD * self.bed_area)

    def _band(self, phi: Tensor, drivers: Mapping) -> Tensor:
        """The band ``w_c + stiff (H - H_crown)`` (m3/s) at `phi` (m)."""
        rate2 = 2.0 * drivers["water.storage_rate"]
        w_c = (rate2 * drivers["water.storage_area"] * (self.crown - drivers["water.head_start"])
               - drivers["water.net_inflow_start"])
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
        if self.floods:
            head_last = torch.minimum(head_last, self.top)
        return ((head_last > self.crown)
                & (self._band(head_last, drivers) > 0.0)).to(F64)        # dynwave.c:688

    def _free(self, phi: Tensor, drivers: Mapping) -> tuple[Tensor, Tensor]:
        """The non-flooded withdrawal (m3/s) and its slope (m2/s) at `phi` (m): storage,
        the hold band above the crown, or the surcharged anchor (class docstring)."""
        storage, slope, stiff = self._storage(phi, drivers)
        if not self.extran:
            return storage, slope
        band = self._band(phi, drivers)
        above = phi > self.crown
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
        faster than the withdrawal it caps."""
        stiff = self._stiff(drivers)
        if not self.extran:
            return stiff
        return torch.where(drivers["water.surcharged"] > 0.5,
                           torch.maximum(stiff, drivers["water.surcharge_slope"]), stiff)

    def flood_excess(self, drivers: Mapping) -> Tensor:
        """The lagged flood excess E (m3/s, >= 0): the fed net inflow
        ``drivers["water.net_inflow_last"]`` over the non-flooded withdrawal at the top,
        ``max(N_last - w(top), 0)``; > 0 exactly where SWMM's update would end above y_max
        (`dynwave.c:752`), i.e. where the junction floods."""
        top = self.top.expand(drivers["water.net_inflow_last"].shape)
        w_top, _ = self._free(top, drivers)
        return torch.clamp(drivers["water.net_inflow_last"] - w_top, min=0.0)

    def _wall(self, phi: Tensor, drivers: Mapping) -> Tensor:
        """The flood wall ``w(top) + E + S (H - top)`` (m3/s) at `phi` (m)."""
        w_top, _ = self._free(self.top.expand(phi.shape), drivers)
        return (w_top + drivers["water.flood_excess"]
                + self._wall_slope(drivers) * (phi - self.top))

    def flow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """Withdrawal (m3/s) at junction heads `phi_nodes` (m)."""
        free, _ = self._free(phi_nodes, drivers)
        if not self.floods:
            return free
        return torch.maximum(free, self._wall(phi_nodes, drivers))         # dynwave.c:752

    def dflow(self, phi_nodes: Tensor, drivers: Mapping | None = None) -> Tensor:
        """d(withdrawal)/dH (m2/s), analytic and strictly positive."""
        free, slope = self._free(phi_nodes, drivers)
        if self.floods:
            wall = self._wall(phi_nodes, drivers)
            wall_slope = self._wall_slope(drivers).expand_as(wall)
            slope = torch.where(wall >= free, wall_slope, slope)
        return slope.expand(torch.broadcast_shapes(slope.shape, phi_nodes.shape))


class DynamicWaveHydraulics:
    """The dynamic-wave closure: the conduit coefficients, junction surface areas and
    outfall heads of one pass, from the previous pass's heads and flows.

    Reads the fed ``"water.phi"`` (m, full-node) and ``"water.q"`` (m3/s, per pipe), the
    full-node driver ``"inflow"`` (m3/s, lateral inflow, zero at outfalls) and its own
    carried state, pinned by `Model` to the step start: ``"sewer.junction_depth"`` (m),
    ``"sewer.pipe_flow"`` (m3/s), ``"sewer.mid_area"`` (SWMM's stored mid-depth area, SI
    area m2 per design D2) and ``"sewer.net_inflow"`` (m3/s, per junction). With a
    `StepContext` it writes ``"water.sources"``, ``"water.phi_boundary"`` (outfall heads, m),
    the ten `DynamicWaveConduit` drivers and the `JunctionStorage` drivers, and the
    reporting key ``"sewer.overflow"`` (below); always the carried keys and the reporting
    keys ``"sewer.q"`` (m3/s), ``"sewer.h"`` (link depth, m), ``"sewer.v"`` (m/s),
    ``"sewer.V_wet"`` (m3), ``"sewer.junction_depth"`` (m) and ``"sewer.surcharged"``
    (per junction, 0/1).

    EXTRAN surcharge (design E1): `crown_depth` is each junction's highest attached conduit
    top (`dynwave.c:137-153`). The closure writes the drivers ``"water.surcharged"`` (0/1,
    from the fed heads: `JunctionStorage.surcharged`), ``"water.surcharge_slope"`` (m2/s)
    and ``"water.head_last"`` (the fed junction heads, m); a surcharged junction holds
    instantaneous continuity, net inflow 0 (see `JunctionStorage` for the tie rule at the
    crown). ``"sewer.surcharged"`` reports SWMM's own test, depth above the crown
    (`dynwave.c:688`): a junction held on the band at its crown (the tie rule) reports 1
    although it is not holding dQ = 0 -- the solve's own flag is the driver
    ``"water.surcharged"``. Under SLOT no junction is ever surcharged (`dynwave.c:677`).

    Flooding (design E3, no ponding): `flood_depth` is SWMM's y_max, the full depth
    `full_depth` (MaxDepth extended to the crown, `link.c:440-463`) plus the junction's
    SurDepth (`Manhole.surcharge_depth`). The closure writes ``"water.net_inflow_last"``
    (the fed net inflow, m3/s), ``"water.flood_excess"`` (m3/s, `JunctionStorage`) and the
    lagged flag ``"water.flooded"`` (0/1, the excess positive); a flooded junction is held
    at y_max. ``"sewer.overflow"`` (m3/s per junction, >= 0) is SWMM's flooding loss,
    ``0.5 (N_start + N)`` at a flooded junction (`dynwave.c:673-674, 783-791`), 0 below
    SWMM's FUDGE flow (1e-4 cfs) and elsewhere. It is lost from the system and is not part
    of the carried ``"sewer.net_inflow"``, which, like SWMM's ``oldNetInflow``
    (`node.c:303`), is the conduit and lateral net inflow alone.

    The arithmetic is SWMM's own, in feet (design D2); `layer` names the head layer.
    """

    integrates = True
    state_keys = ("sewer.junction_depth", "sewer.pipe_flow", "sewer.mid_area",
                  "sewer.net_inflow")
    edge_kinds = ("pipe",)

    def __init__(self, graph: Network, net: SewerNetwork, *, layer: str = "water") -> None:
        self.net = graph
        self.layer = layer
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
        }
        self.node_invert = torch.tensor(
            [float(graph.graph.nodes[n]["invert"]) for n in nodes], dtype=F64)
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
        self.offset1 = torch.tensor([p.inlet_offset for p in net.pipes], dtype=F64)
        self.offset2 = torch.tensor([p.outlet_offset for p in net.pipes], dtype=F64)
        self._rebuild_geometry()

        # Each outfall's own conduit and the end it sits at (link.c:740-751: node2 first).
        pipe_of, at_v = [], []
        for name in self.outfall_names:
            ends = [(i, p.v == name) for i, p in enumerate(net.pipes) if name in (p.u, p.v)]
            if len(ends) != 1:
                raise ValueError(
                    f"DynamicWaveHydraulics: outfall {name!r} is attached to {len(ends)} "
                    f"conduits; dynamic-wave routing takes exactly one")
            i, downstream = ends[0]
            pipe = net.pipes[i]
            if pipe.u in self.outfall_names and pipe.v in self.outfall_names:
                raise ValueError(
                    f"DynamicWaveHydraulics: conduit {pipe.name!r} joins two outfalls "
                    f"({pipe.u!r}, {pipe.v!r}); SWMM sets the depth of only one of them")
            pipe_of.append(i)
            at_v.append(downstream)
        self.outfall_pipe = torch.tensor(pipe_of, dtype=torch.long)
        at_v_t = torch.tensor(at_v, dtype=torch.bool)
        #: per outfall: whether it is its conduit's downstream (second) node
        self.outfall_at_v = at_v_t
        self.outfall_offset_ft = torch.where(
            at_v_t, self.offset2[self.outfall_pipe], self.offset1[self.outfall_pipe]) / LCF
        self.outfall_kind = torch.tensor(
            [_OUTFALL_CODES[o.kind] for o in net.outfalls], dtype=torch.long)
        self.outfall_stage_ft = torch.tensor(
            [(o.stage if o.stage is not None else o.invert) / LCF for o in net.outfalls],
            dtype=F64)
        self.min_area_ft2 = (
            DEFAULT_MIN_SURFACE_AREA_FT2 if self.options.min_surface_area is None
            else self.options.min_surface_area / LCF**2)            # dynwave.c:186-187
        # Each junction's crown: the highest conduit top over the conduit ends it carries,
        # crownElev - invertElev (dynwave.c:137-153); 0 for a junction with none.
        geo = self.geo
        crown = torch.zeros(len(nodes), dtype=F64)
        crown = crown.scatter_reduce(0, self.u, (geo.offset1 + geo.y_full).detach(), "amax")
        crown = crown.scatter_reduce(0, self.v, (geo.offset2 + geo.y_full).detach(), "amax")
        self.crown_ft = crown[self.junction_idx]
        #: per-junction crown depth (m), in junction order
        self.crown_depth = self.crown_ft * LCF
        # The full depth: MaxDepth, extended to every attached conduit's top
        # (link.c:440-463, the same conduit ends as the crown).
        max_depth = torch.tensor([0.0 if m.ground is None else m.ground - m.invert
                                  for m in net.manholes], dtype=F64)
        #: per-junction full depth (m): MaxDepth extended to the crown
        self.full_depth = torch.maximum(max_depth, self.crown_depth)
        #: per-junction flood depth y_max (m): the full depth plus SurDepth, above which
        #: the junction floods (no ponding, dynwave.c:746-748)
        self.flood_depth = self.full_depth + torch.tensor(
            [m.surcharge_depth for m in net.manholes], dtype=F64)
        jinv = self.node_invert[self.junction_idx]
        #: the junctions' `JunctionStorage` (the model's node source; `junction_residual`)
        self.storage = JunctionStorage(
            self.junction_idx, jinv, bed_area=self.min_area_ft2 * _AREA_SI,
            crown=jinv + self.crown_depth if self.options.surcharge_method == "extran"
            else None, top=jinv + self.flood_depth)

    # ------------------------------------------------------------------ geometry
    def _rebuild_geometry(self) -> None:
        self.geo = ConduitGeometry.from_si(
            self.diameter, self.roughness, self.slope, self.length,
            self.node_invert[self.u], self.node_invert[self.v], self.offset1, self.offset2,
            self.is_outfall[self.u], self.is_outfall[self.v])

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

    def _terms(self, state: Mapping, dt: float) -> ConduitTerms:
        heads = state[f"{self.layer}.phi"]
        q = state[f"{self.layer}.q"]
        # SWMM's node depth is never negative (dynwave.c:743); the stiff bed lets a dry
        # junction's head sit a hair below its invert.
        depth_ft = torch.clamp(heads - self.node_invert, min=0.0) / LCF
        return conduit_terms(
            self.geo, depth_ft[..., self.u], depth_ft[..., self.v], q / QCF,
            state["sewer.pipe_flow"] / QCF, state["sewer.mid_area"] / _AREA_SI, dt,
            damping=self.options.inertial_damping,
            normal_flow_limited=self.options.normal_flow_limited,
            surcharge_method=self.options.surcharge_method)

    def _surcharged(self, heads: Tensor) -> Tensor:
        """EXTRAN's surcharge test on the junction heads `heads` (m, full-node): depth above
        a positive crown (`dynwave.c:688`), as a bool per junction."""
        jidx = self.junction_idx
        depth_ft = torch.clamp(heads[..., jidx] - self.node_invert[jidx], min=0.0) / LCF
        if self.options.surcharge_method != "extran":
            return torch.zeros_like(depth_ft, dtype=torch.bool)
        return (self.crown_ft > 0.0) & (depth_ft > self.crown_ft)

    def _carried(self, state: Mapping, terms: ConduitTerms, inflow: Tensor) -> State:
        heads = state[f"{self.layer}.phi"]
        q = state[f"{self.layer}.q"]
        jidx = self.junction_idx
        n = self.node_invert.shape[-1]
        batch = torch.broadcast_shapes(q.shape[:-1], inflow.shape[:-1])
        through = torch.zeros(batch + (n,), dtype=F64)
        q_b = q.expand(batch + q.shape[-1:])
        through = through.index_add(-1, self.v, q_b).index_add(-1, self.u, -q_b)
        net_inflow = inflow + through                                  # dynwave.c:528-541
        return {
            "sewer.junction_depth": torch.clamp(heads[..., jidx] - self.node_invert[jidx],
                                                min=0.0),
            "sewer.pipe_flow": q,
            "sewer.mid_area": terms.mid_area * _AREA_SI,
            "sewer.net_inflow": net_inflow[..., jidx],
        }

    def _outfall_heads(self, q: Tensor) -> Tensor:
        """`link.c::link_setOutfallDepth` + `node.c::outfall_setOutletDepth` (link.c:728-766,
        node.c:1413-1492), from each outfall conduit's flow `q` (m3/s): heads (m)."""
        op = self.outfall_pipe
        geo = self.geo
        q_ft = (q[..., op] / QCF).abs()
        yf, af = geo.y_full[op], geo.a_full[op]
        y_norm = normal_depth_ft(q_ft, yf, af, geo.s_full[op], geo.beta[op])
        y_crit = critical_depth(q_ft, yf, af)
        z = self.outfall_offset_ft
        zero = torch.zeros_like(y_norm)
        free = torch.where(z > 0.0, zero, torch.minimum(y_norm, y_crit))      # node.c:1431
        normal = torch.where(z > 0.0, zero, y_norm)                           # node.c:1436
        invert = self.node_invert[self.outfall_idx] / LCF
        stage = self.outfall_stage_ft
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
        depth = torch.where(kind == 0, free, torch.where(kind == 1, normal, fixed))
        return self.node_invert[self.outfall_idx] + depth * LCF

    def _switch_slope(self, q: Tensor, terms: ConduitTerms, area_ft2: Tensor, surf: Tensor,
                      dt: float) -> Tensor:
        """The slope switch's wall slope S (m2/s, per pipe; `DynamicWaveConduit`): the
        linearised switching surface ``1/(c + 1/S_max)``. ``c`` (m per m3/s) is how far the
        lagged outfall depth at the conduit's downstream end moves per unit of its flow
        (`_outfall_heads`, a forward difference; 0 for other conduits): on the surface
        ``y1 = y2(q)`` the pass then sees the outfall depth follow the flow, as it does at
        the fixed point. An outfall at the conduit's UPSTREAM end gets no such term (there
        the surface would fall with the flow), so that conduit's S is ``S_max``: its
        switch stays exact at the fixed point but may take more passes to settle. ``S_max``
        is `K_SWITCH` times the smaller end junction's grounding (storage slope ``2 A_s/dt``
        plus its conduits' conveyance)."""
        k_si = terms.conveyance * (QCF / LCF)
        ground = torch.zeros_like(surf).index_add(-1, self.u, k_si).index_add(-1, self.v, k_si)
        ground = ground.index_add(-1, self.junction_idx, 2.0 / dt * area_ft2 * _AREA_SI)
        ground = torch.where(self.is_outfall, torch.full_like(ground, torch.inf), ground)
        s_max = K_SWITCH * torch.minimum(ground[..., self.u], ground[..., self.v])
        op = self.outfall_pipe
        q_abs = q.abs()
        step = 1e-6 * torch.clamp(q_abs[..., op], min=1e-3)
        bumped = q_abs.clone()
        bumped[..., op] = q_abs[..., op] + step
        rise = (self._outfall_heads(bumped) - self._outfall_heads(q_abs)) / step
        rise = torch.where(self.outfall_at_v, torch.clamp(rise, min=0.0),
                           torch.zeros_like(rise))
        compliance = torch.zeros_like(s_max).index_add(-1, op, rise.expand(
            s_max.shape[:-1] + rise.shape[-1:]))
        return 1.0 / (compliance + 1.0 / s_max)

    # ------------------------------------------------------------------ protocol
    def __call__(self, state: Mapping, drivers: Mapping,
                 ctx: StepContext | None = None) -> dict[str, Tensor]:
        """One pass's drivers (see the class docstring); a query (no `ctx`, or
        ``ctx.dt is None``) returns the carried and reporting keys only."""
        dt = None if ctx is None else ctx.dt
        inflow = self._inflow(drivers)
        terms = self._terms(state, self.options.routing_step if dt is None else dt)
        out = self._carried(state, terms, inflow)
        q = state[f"{self.layer}.q"]
        depth_ft = terms.depth
        a_link = area_of_depth(depth_ft, self.geo.y_full, self.geo.a_full)
        moving = (depth_ft > _VELOCITY_MIN_DEPTH) & (a_link > FUDGE)          # link.c:834-840
        v_ft = (q / QCF) / torch.where(moving, a_link, torch.ones_like(a_link))
        out.update({
            "sewer.q": q,
            "sewer.h": depth_ft * LCF,
            "sewer.v": torch.where(moving, v_ft, torch.zeros_like(v_ft)) * LCF,
            "sewer.V_wet": terms.volume * QCF,
            "sewer.surcharged": self._surcharged(state[f"{self.layer}.phi"]).to(F64),
        })
        if dt is None:
            return out
        jidx = self.junction_idx
        n = self.node_invert.shape[-1]
        surf = torch.zeros(terms.surface_area1.shape[:-1] + (n,), dtype=F64)
        surf = surf.index_add(-1, self.u, terms.surface_area1)                # dynwave.c:582-583
        surf = surf.index_add(-1, self.v, terms.surface_area2)
        area_ft2 = torch.clamp(surf[..., jidx], min=self.min_area_ft2)        # dynwave.c:670
        # conveyance the pass lags into c0 (other end pinned), per junction (dynwave.c:578)
        lagged = torch.zeros_like(surf).index_add(-1, self.u, terms.lagged_conveyance1)
        lagged = lagged.index_add(-1, self.v, terms.lagged_conveyance2)[..., jidx]
        anchor = lagged * (QCF / LCF) + K_ANCHOR * 2.0 / dt * area_ft2 * _AREA_SI
        switch = terms.slope_switch
        switch_slope = self._switch_slope(q, terms, area_ft2, surf, dt)
        out.update({
            f"{self.layer}.sources": inflow,
            f"{self.layer}.phi_boundary": self._outfall_heads(q),
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
            "water.dry_upstream": terms.dry_upstream.to(F64),
            "water.dry_downstream": terms.dry_downstream.to(F64),
            "water.storage_area": area_ft2 * _AREA_SI,
            "water.storage_rate": torch.tensor(1.0 / dt, dtype=F64),
            "water.head_start": self.node_invert[jidx] + state["sewer.junction_depth"],
            "water.net_inflow_start": state["sewer.net_inflow"],
            "water.surcharge_slope": anchor,
            "water.head_last": state[f"{self.layer}.phi"][..., jidx],
            "water.net_inflow_last": out["sewer.net_inflow"],
        })
        out["water.surcharged"] = self.storage.surcharged(out)
        excess = self.storage.flood_excess(out)
        flooded = excess > 0.0
        # SWMM's overflow is the step's trapezoid net inflow dV/dt, zeroed below FUDGE
        # (dynwave.c:673-674, 783-791); it never enters the next step's oldNetInflow
        # (node.c:303 takes inflow - outflow, which exclude it).
        overflow = 0.5 * (state["sewer.net_inflow"] + out["sewer.net_inflow"])
        overflow = torch.where(flooded & (overflow >= FUDGE_FLOW), overflow,
                               torch.zeros_like(overflow))
        out.update({"water.flood_excess": excess, "water.flooded": flooded.to(F64),
                    "sewer.overflow": overflow})
        return out

    def carried(self, state: Mapping, drivers: Mapping, *, dt: float | None = None) -> State:
        """The four carried keys evaluated AT `state` (``water.phi``, ``water.q``), for
        re-synchronising after a step of `dt` s (default: the routing step): the pass's own
        copy is its input's, which equals its output only to the iteration tolerance."""
        terms = self._terms(state, self.options.routing_step if dt is None else dt)
        return self._carried(state, terms, self._inflow(drivers))

    def initial_state(self, drivers: Mapping | None = None) -> State:
        """SWMM's cold, dry start (flowrout.c:75-102, 336-507): no flow, junctions at their
        inverts, outfalls at the depth `link_setOutfallDepth` gives at zero flow (a FIXED
        outfall at its stage, the others at their invert; flowrout.c:383), every conduit at
        the mean of its end-node depths above its offsets, clamped to [0, yFull] and at
        least FUDGE (flowrout.c:389-420), which sets its stored area (flowrout.c:497), and a
        zero junction net inflow. SWMM zeroes the lateral inflow before routing starts
        (node.c:264, flowrout.c:437), so the first step's ``oldNetInflow`` is 0
        (routing.c:236-242, node.c:303) and its trapezoid sees only the end-of-step inflow.
        `drivers`, when given, has its ``"inflow"`` validated."""
        if drivers is not None:
            self._inflow(drivers)
        q0 = torch.zeros(len(self.pipe_names), dtype=F64)
        phi = self.node_invert.clone()
        phi[self.outfall_idx] = self._outfall_heads(q0)
        geo = self.geo
        depth_ft = (phi - self.node_invert) / LCF
        y1 = torch.clamp(depth_ft[self.u] - geo.offset1, min=0.0)
        y2 = torch.clamp(depth_ft[self.v] - geo.offset2, min=0.0)
        y = torch.clamp(0.5 * (torch.minimum(y1, geo.y_full) + torch.minimum(y2, geo.y_full)),
                        min=FUDGE)                                            # :402-416
        return {
            f"{self.layer}.phi": phi,
            f"{self.layer}.q": q0,
            "sewer.junction_depth": torch.zeros(len(self.junction_names), dtype=F64),
            "sewer.pipe_flow": torch.zeros(len(self.pipe_names), dtype=F64),
            "sewer.mid_area": area_of_depth(y, geo.y_full, geo.a_full) * _AREA_SI,  # :497
            "sewer.net_inflow": torch.zeros(len(self.junction_names), dtype=F64),
        }

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


def build_dynamic_wave_model(net: SewerNetwork, *, iterate_tol: float = 1e-11,
                             iterate_max: int = 2000, relaxation: float = 0.5,
                             acceleration: str | None = "anderson", memory: int = 5,
                             iterate_flow_tol: float | None = 1e-10
                             ) -> tuple[Model, State, Drivers]:
    """Assemble a dynamic-wave sewer and return ``(model, state, drivers)``.

    One `PotentialFlowLayer` ``"water"`` (quantity ``"head"``, m; flows m3/s) over the pipes,
    with the outfalls as its boundary and a `JunctionStorage` at every junction, and the
    `DynamicWaveHydraulics` closure; ``coupling="iterate"`` converges on the junction heads
    to `iterate_tol` (m) within `iterate_max` passes. The passes are Anderson-accelerated
    over the last `memory` of them by default (``acceleration="anderson"``, `Model`'s
    `iterate_acceleration`), with `relaxation` the damping of the steps it falls back to;
    ``acceleration=None`` is plain relaxation, which reaches the same fixed point in two to
    three times the passes. `iterate_flow_tol` (m3/s) is the tolerance on the conduit
    flows' own fixed-point residual, judged alongside the heads (`Model`'s
    `iterate_flow_tol`), and is required: ``None`` (heads judged alone, as phase A did) is
    refused by name. On the normal-flow slope switch (`DynamicWaveConduit`) the heads
    bound the flow residual only through the wall slope S over the junction's grounding,
    so a heads-only test can stop a pass short of the fixed point (a flow residual of
    3e-5 m3/s was measured on the tree's drain-down). The state is
    SWMM's cold, dry start. Attached to the model: `junction_idx`, `junction_invert` (m),
    `outfall_pipe_idx`, `dynamic_wave` (the closure), `routing_step` (s), `pipe_names`,
    `pipe_columns`, `notes`, `input_specs`, `driver_template`.
    """
    net.validate()
    if net.routing != "dynamic_wave":
        raise ValueError(
            f"build_dynamic_wave_model: network routing is {net.routing!r}, not "
            f"'dynamic_wave'")
    if iterate_flow_tol is None:
        raise ValueError(
            "build_dynamic_wave_model: iterate_flow_tol (m3/s) is required, got None: on the "
            "normal-flow slope switch a heads-only convergence test can stop short of the "
            "fixed point; plain relaxation is acceleration=None with a flow tolerance")
    graph = Network(dtype=F64)
    for manhole in net.manholes:
        graph.add_node(manhole.name, invert=manhole.invert, ground=manhole.ground)
    for outfall in net.outfalls:
        graph.add_node(outfall.name, invert=outfall.invert)
    for pipe in net.pipes:
        graph.add_edge(pipe.u, pipe.v, kind="pipe", name=pipe.name)
    boundary = [o.name for o in net.outfalls]
    closure = DynamicWaveHydraulics(graph, net)
    junction_idx = closure.junction_idx
    junction_invert = closure.node_invert[junction_idx]
    # Built once without the node source to learn the interior (a NodeSource must sit on
    # interior nodes), as the Modelica assembly does.
    probe = PotentialFlowLayer(graph, "water", [DynamicWaveConduit()], boundary=boundary,
                               quantity="head", unit="m", flow_unit="m3/s")
    interior = set(probe.interior.tolist())
    stranded = [m.name for m, i in zip(net.manholes, junction_idx.tolist(), strict=True)
                if i not in interior]
    if stranded:
        raise ValueError(f"build_dynamic_wave_model: junction(s) {stranded} touch no conduit")
    layer = PotentialFlowLayer(
        graph, "water", [DynamicWaveConduit()], boundary=boundary,
        node_sources=[closure.storage],
        quantity="head", unit="m", flow_unit="m3/s")
    if layer.bound.tolist() != closure.outfall_idx.tolist():
        raise ValueError("build_dynamic_wave_model: the layer's boundary order differs from "
                         "the outfall order the closure writes 'water.phi_boundary' in")
    model = Model(graph, {"water": layer}, closures=[closure], coupling="iterate",
                  iterate_tol={"water": iterate_tol}, iterate_max=iterate_max,
                  iterate_relaxation=relaxation, iterate_acceleration=acceleration,
                  iterate_memory=memory,
                  iterate_flow_tol={"water": iterate_flow_tol})
    inflow = torch.zeros(graph.n, dtype=F64)
    inflow[junction_idx] = torch.tensor([m.inflow for m in net.manholes], dtype=F64)
    state = closure.initial_state({"inflow": inflow})
    drivers: Drivers = {
        "inflow": inflow,
        "water.phi_boundary": state["water.phi"][closure.outfall_idx].clone(),
    }
    closure.notes["routing"] = (
        f"dynamic wave (SWMM 5.2.4 fixed point): heads converged to {iterate_tol} m"
        + f" and flows to {iterate_flow_tol} m3/s"
        + f" within {iterate_max} passes"
        + ("" if acceleration is None else f" ({acceleration}-accelerated)")
        + "; SWMM's HEAD_TOLERANCE and MAX_TRIALS are not used")
    model.notes = closure.notes
    model.junction_idx = junction_idx
    model.junction_invert = junction_invert
    model.outfall_pipe_idx = closure.outfall_pipe
    model.dynamic_wave = closure
    model.routing_step = float(net.dynamic_wave.routing_step)
    model.pipe_names = list(closure.pipe_names)
    model.pipe_columns = ("q", "h", "v", "V_wet")
    model.input_specs = {
        "inflow": dict(description="lateral inflow at each manhole", unit="m3/s",
                       over="nodes"),
    }
    model.driver_template = dict(drivers)
    return model, state, drivers


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


def simulate_dynamic_wave(model: Model, state: State, drivers: Drivers, *, steps: int,
                          dt: float | None = None, inflow: Tensor | None = None,
                          record: tuple[str, ...] = ("water.phi", "water.q"),
                          diagnostics: list | None = None, **solve_kwargs
                          ) -> tuple[State, dict[str, Tensor]]:
    """Run `steps` routing steps of `dt` s (default: the model's routing step) from `state`.

    `inflow` (m3/s, full-node) overrides ``drivers["inflow"]``: ``(n,)`` is one constant
    inflow; ``(B, n)`` a batch of B constant inflows -- a 2-D inflow is ALWAYS a batch,
    never a series; ``(steps, B, n)`` an inflow per step for a batch of B, the lateral
    inflow at the END of each step (one network's series is ``(steps, 1, n)``). Any other
    rank is refused. A batched inflow expands an unbatched `state` to the batch. After every
    step the carried keys are re-evaluated at the returned state. `record` may also name
    ``"sewer.overflow"`` (m3/s per junction, SWMM's flooding loss over each step:
    `DynamicWaveHydraulics.overflow`), which is not a state key.
    `diagnostics`, when a list, receives each step's `Model.step` diagnostics.
    `solve_kwargs` reach the head solve (Newton); `atol` defaults to `solve_atol` (m3/s:
    `SOLVE_ATOL`, or the bed's round-off floor where larger) and `rtol` to 0, because
    Newton's own default (sqrt(eps), 1.5e-8 m3/s) would stop a warm-started pass before
    it moves and leave the junction balance off by that much. Returns the final state and
    the recorded `record` keys stacked over steps (leading dimension `steps`).
    """
    hyd = model.dynamic_wave
    dt = model.routing_step if dt is None else float(dt)
    solve_kwargs.setdefault("atol", solve_atol(model, dt))
    solve_kwargs.setdefault("rtol", 0.0)
    n = model.net.n
    batch: tuple[int, ...] = ()
    if inflow is not None:
        inflow = torch.as_tensor(inflow, dtype=F64)
        if inflow.dim() not in (1, 2, 3) or inflow.shape[-1] != n:
            raise ValueError(
                f"simulate_dynamic_wave: inflow must be (n,), (B, n) or (steps, B, n) with "
                f"n = {n} nodes, got shape {tuple(inflow.shape)}")
        if inflow.dim() == 3 and inflow.shape[0] < steps:
            raise ValueError(
                f"simulate_dynamic_wave: inflow has {inflow.shape[0]} steps, {steps} asked")
        batch = tuple(inflow.shape[:-1]) if inflow.dim() == 2 else tuple(inflow.shape[1:-1])
    state = dict(state)
    if batch:
        state = {k: (v.expand(batch + v.shape[-1:]).clone() if v.dim() == 1 else v)
                 for k, v in state.items()}
    rows: dict[str, list[Tensor]] = {key: [] for key in record}
    t = 0.0
    for k in range(steps):
        if inflow is not None:
            drivers = dict(drivers, inflow=inflow[k] if inflow.dim() == 3 else inflow)
        diag: dict = {}
        before = state
        try:
            state = model.step(state, drivers, dt, t=t, diagnostics=diag, **solve_kwargs)
        except RuntimeError as err:
            raise RuntimeError(f"dynamic wave step at t={t} s: {err}") from err
        state.update(hyd.carried(state, drivers, dt=dt))   # resync, as modelica/run.py does
        t += dt
        if diagnostics is not None:
            diagnostics.append(diag)
        for key in record:
            rows[key].append(hyd.overflow(before, state, drivers, dt=dt)
                             if key == "sewer.overflow" else state[key])
    return state, {key: torch.stack(v) for key, v in rows.items()}

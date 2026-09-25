"""`SewerHydraulics`: the one stateful closure that owns the sewer's water side.

Per pass, from the state at the START of the pass:

1. validate the inflow driver (finite, non-negative, zero on boundary nodes);
2. per-pipe discharge -- ``cycles.particular_flow(net, s, kind="pipe")`` on the tree, which
   is CLOSED FORM (the cycle space of a tree is empty), or, with
   ``storage=True``, the level-synchronous implicit-Euler storage sweep below;
3. normal depth per pipe by the batched Manning inversion;
4. every derived per-pipe quantity the air side and the quality side read;
5. the two transport layers' capacities and the full-node air density.

Because the closure computes the flow ITSELF (rather than reading a potential layer's
solved ``q``), the air layer sees THIS pass's water state, not the previous one's: closures
run before every potential solve inside one `Model._pass`. That is the whole reason the
water side is continuity-first rather than a Newton potential layer.

Loops. The construction builds the pipe index, each manhole's outgoing pipe, the
leaf-to-root LEVEL order, and, per level, the FLAT index tensors the storage sweep gathers
and scatters with (the level's own manhole positions, and a `(source, target-slot)` pair
tensor for every upstream contribution into that level) -- once, exactly as
`cycles._tree_solve` precomputes its own per-level index tensors. Per call there is NO
Python loop over pipes or manholes anywhere, including inside the storage sweep: the sweep
loops over tree LEVELS only (one gather plus one `index_add` per level), whose count is the
tree DEPTH (3 on the committed fixture), so no per-step path loops over pipes or manholes.

The storage sweep's implicit-Euler residual is evaluated at `h = 0` exactly for a dry leaf
manhole (zero lateral inflow, zero initial state), and `solve_monotone`'s backward
differentiates that residual with respect to the pipe diameter, roughness and slope. This is
safe only because `geometry.hydraulic_radius`'s `** (2/3)` carries a both-branches-safe guard
at `h = 0`; this module relies on that guard and does not re-implement it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from noodl.apps.sewer import geometry as geom
from noodl.apps.sewer.air import air_density
from noodl.cycles import particular_flow
from noodl.solvers.scalar import solve_monotone

if TYPE_CHECKING:
    from noodl.model import StepContext

Tensor = torch.Tensor
F64 = torch.float64

#: Volume (m3) substituted for a pipe carrying EXACTLY zero flow, so that the transport
#: layers' per-step capacity stays strictly positive (`TransportLayer` refuses a
#: non-positive capacity by name). Applied nowhere else, and recorded in `notes` when it is applied.
CAPACITY_FLOOR = 1e-9


class SewerHydraulics:
    """The water side of a sewer, as a `Model` closure.

    `state_keys` is `("sewer.H",)` when `storage=True` (the manhole levels, carried by this
    closure across steps as closure-carried state) and empty otherwise. `integrates` is exactly
    `storage`: under `storage=True` this closure integrates the manhole levels over the
    model's own step interval (a `StepContext`, not a fixed interval read off its own
    constructor); under `storage=False` it is a plain algebraic closure.
    """

    @property
    def integrates(self) -> bool:
        return self.storage

    def __init__(
        self,
        net,
        pipes,
        manholes,
        *,
        storage: bool = False,
        dt: float | None = None,
        quality_layer: str = "water_quality",
        air_quality_layer: str = "air_quality",
        shaft_depth=None,
        surface_area=None,
        name: str = "sewer",
    ) -> None:
        self.net = net
        self.name = name
        self.storage = bool(storage)
        self.quality_layer = quality_layer
        self.air_quality_layer = air_quality_layer
        self.notes: dict[str, str] = {}
        self.state_keys: tuple[str, ...] = ("sewer.H",) if self.storage else ()
        # `dt`, when given, is the interval this closure was built for, and a step of any
        # OTHER interval is then refused by name in `__call__`; when omitted (the default,
        # and allowed under `storage=True` too), the step's own interval is used,
        # via the `StepContext` `Model._apply_closures` passes to an integrating closure.
        if dt is not None and not dt > 0:
            raise ValueError(
                f"SewerHydraulics {name!r}: dt, when given, must be positive, got {dt!r}"
            )
        self.dt = float(dt) if dt is not None else None

        self.pipe_names = [p.name for p in pipes]
        self.manhole_names = [m.name for m in manholes]
        self.length = torch.tensor([p.length for p in pipes], dtype=F64)
        self.diameter = torch.tensor([p.diameter for p in pipes], dtype=F64)
        self.roughness = torch.tensor([p.n for p in pipes], dtype=F64)
        self.slope = torch.tensor([p.slope for p in pipes], dtype=F64)

        node_index = {n: i for i, n in enumerate(net.nodes)}
        self.manhole_idx = torch.tensor(
            [node_index[m.name] for m in manholes], dtype=torch.long
        )
        self.ambient_idx = (
            node_index["ambient"] if "ambient" in node_index else None
        )
        self._node_idx = torch.arange(net.n, dtype=torch.long)
        # Each manhole's OUTGOING pipe: a tree guarantees exactly one, and `SewerNetwork`
        # has already refused anything else by name. The map is a position into the per-pipe
        # order, so every per-manhole quantity is a single gather.
        outgoing = {p.u: i for i, p in enumerate(pipes)}
        missing = [m.name for m in manholes if m.name not in outgoing]
        if missing:
            raise ValueError(
                f"SewerHydraulics {name!r}: manhole(s) {missing} have no outgoing pipe; "
                f"every manhole of a dendritic sewer drains through exactly one"
            )
        self.out_pipe = torch.tensor(
            [outgoing[m.name] for m in manholes], dtype=torch.long
        )
        # Multi-outfall forests. `_tree_flow` must subtract each COMPONENT's own
        # lateral total at its OWN outfall, not the whole network's total at every outfall
        # (which silently gave the wrong per-component balance, and an unnamed
        # `index_add_` shape error, for a forest of more than one component). Each
        # manhole's own outfall is found ONCE here by walking `outgoing` forward until a
        # name that is not a manhole -- a tree guarantees this terminates, and
        # `SewerNetwork.validate()` has already refused a cycle -- then mapped to that
        # outfall's SLOT within `self.outfall_node_idx` (the node positions
        # `_outfall_positions` returns, in the same order every call of `_tree_flow` would
        # otherwise recompute).
        self.outfall_node_idx = _outfall_positions(net, self.manhole_idx, self.ambient_idx)
        outfall_slot_by_node = {
            int(idx): slot for slot, idx in enumerate(self.outfall_node_idx.tolist())
        }
        manhole_name_set = set(self.manhole_names)
        limit = len(manholes) + 1
        slots = []
        for m in manholes:
            node = m.name
            steps = 0
            while node in manhole_name_set and steps <= limit:
                node = pipes[outgoing[node]].v
                steps += 1
            slots.append(outfall_slot_by_node[node_index[node]])
        self.manhole_outfall_slot = torch.tensor(slots, dtype=torch.long)
        self.surface_area = (
            torch.as_tensor(surface_area, dtype=F64)
            if surface_area is not None
            else torch.full((len(manholes),), 1.167, dtype=F64)
        )
        if surface_area is None:
            # SWMM's MIN_SURFAREA default, 12.56 ft2 = the area of a 4 ft manhole shaft
            # (SWMM Ref. Man. Vol. II section 3.1). VERIFIED; noted because it is a
            # substitution the builder made, not a value the network carried.
            self.notes["surface_area"] = (
                "manhole surface areas defaulted to SWMM's MIN_SURFAREA, 1.167 m2 "
                "(12.56 ft2, a 4 ft shaft)"
            )
        self.shaft_depth = (
            torch.as_tensor(shaft_depth, dtype=F64) if shaft_depth is not None else None
        )
        if shaft_depth is None:
            self.notes["shaft_volume"] = (
                "no ground elevation given, so the manhole shafts contribute no headspace "
                "volume; the air quality capacity is the conduit headspace alone"
            )
        self.node_names = list(net.nodes)
        # Leaf-to-root LEVEL order, one Python loop HERE, none per call.
        self.levels = _levels(pipes, manholes)
        self.upstream: dict[int, list[int]] = {}
        position = {m.name: i for i, m in enumerate(manholes)}
        for p in pipes:
            if p.v in position:
                self.upstream.setdefault(position[p.v], []).append(position[p.u])
        # Per level, the FLAT index tensors the storage sweep gathers/scatters with, built
        # ONCE here: `level_idx` is that level's own manhole positions; `level_src`/
        # `level_tgt` is a `(source manhole position, target slot WITHIN this level)` pair
        # for every upstream contribution into this level, so the per-call sweep collapses
        # each level's upstream-inflow assembly to one `index_select` gather plus one
        # `index_add` scatter -- no Python loop over manholes at call time (mirrors
        # `cycles._tree_solve`'s own per-level index tensors).
        self.level_idx: list[Tensor] = []
        self.level_src: list[Tensor] = []
        self.level_tgt: list[Tensor] = []
        for level in self.levels:
            self.level_idx.append(torch.tensor(level, dtype=torch.long))
            src: list[int] = []
            tgt: list[int] = []
            for slot, j in enumerate(level):
                for u in self.upstream.get(j, ()):
                    src.append(u)
                    tgt.append(slot)
            self.level_src.append(torch.tensor(src, dtype=torch.long))
            self.level_tgt.append(torch.tensor(tgt, dtype=torch.long))

    # ------------------------------------------------------------------ call
    def __call__(
        self, state: Mapping, drivers: Mapping, ctx: StepContext | None = None
    ) -> dict[str, Tensor]:
        inflow = _require(drivers, "inflow", self.name)
        if inflow.shape[-1] != self.net.n:
            raise ValueError(
                f"SewerHydraulics {self.name!r}: driver 'inflow' must be in FULL node "
                f"order with trailing shape ({self.net.n},), got "
                f"{tuple(inflow.shape)}"
            )
        bad_finite = ~torch.isfinite(inflow)
        if bool(torch.any(bad_finite)):
            flat = bad_finite.reshape(-1, bad_finite.shape[-1])
            batch_idx, node_idx = torch.nonzero(flat, as_tuple=True)
            names = [self.node_names[i] for i in node_idx.tolist()]
            raise ValueError(
                f"SewerHydraulics {self.name!r}: driver 'inflow' must be finite; "
                f"non-finite at node(s) {names} instance(s) {batch_idx.tolist()}"
            )
        lateral = inflow.index_select(-1, self.manhole_idx)
        if bool(torch.any(lateral < 0)):
            bad = (lateral < 0).reshape(-1, lateral.shape[-1]).any(0)
            names = [self.manhole_names[i] for i in bad.nonzero().flatten().tolist()]
            raise ValueError(
                f"SewerHydraulics {self.name!r}: inflow must be non-negative; negative at "
                f"manhole(s) {names}"
            )
        out: dict[str, Tensor] = {}
        if self.storage:
            if ctx is None:
                # A query (`residuals`, `current_flows`, `ports`, the sewer report): evaluate
                # the algebraic outputs at the given state's levels, without advancing them.
                q, levels = self._at_levels(state)
            else:
                # `Model._apply_closures` refuses `ctx.dt is None` (steady()) by
                # name, generically, before an integrating closure is ever called -- so
                # `ctx.dt` is always a real interval here, never `None`.
                if self.dt is not None and not math.isclose(self.dt, ctx.dt, rel_tol=1e-12):
                    raise ValueError(
                        f"SewerHydraulics {self.name!r} was built for dt={self.dt} s and "
                        f"was asked to step {ctx.dt} s; build it without dt to follow the "
                        f"model's step"
                    )
                q, levels = self._storage_sweep(state, lateral, ctx.dt)
            out["sewer.H"] = levels
            # Profiled hot spot (~40% of one diurnal step's wall time): the storage
            # sweep already solved for each manhole's own level via `solve_monotone` (the
            # manhole level equals the outgoing pipe's entrance depth), so calling
            # `geom.normal_depth` on `q` here would run a SECOND, redundant Newton solve to
            # recover a depth `_storage_sweep` already has -- `levels` remapped from
            # manhole order into pipe order (through `out_pipe`, exactly as `_storage_sweep`
            # itself remaps `q_out`) is that same value, bit-for-bit, with no root-find.
            h = torch.zeros_like(q).index_copy(-1, self.out_pipe, levels)
        else:
            q = self._tree_flow(inflow, lateral)
            h = geom.normal_depth(
                q,
                self.diameter,
                self.roughness,
                self.slope,
                names=self.pipe_names,
            )
        area = geom.flow_area(h, self.diameter)
        radius = geom.hydraulic_radius(h, self.diameter)
        width = geom.top_width(h, self.diameter)
        mean_depth = geom.hydraulic_mean_depth(h, self.diameter)
        a_air, _p_air, d_h = geom.air_geometry(h, self.diameter)
        wetted = area * self.length
        air_volume = a_air * self.length
        # A pipe with EXACTLY zero flow has zero wetted volume, and a transport layer
        # refuses a non-positive capacity by name. Floor it -- on both branches
        # of the `where`, and only where the flow is exactly zero -- and say so in `notes`.
        dry = q <= 0
        wetted = torch.where(dry, torch.full_like(wetted, CAPACITY_FLOOR), wetted)
        if bool(dry.any()):
            names = [
                self.pipe_names[i]
                for i in dry.reshape(-1, dry.shape[-1]).any(0).nonzero().flatten().tolist()
            ]
            self.notes["capacity_floor"] = (
                f"pipes {names} carry exactly zero flow; their water-quality capacity is "
                f"floored at {CAPACITY_FLOOR} m3 so the transport layer stays well posed"
            )
        area_safe = torch.where(dry, torch.ones_like(area), area)
        velocity = torch.where(dry, torch.zeros_like(q), q / area_safe)
        out.update(
            {
                "sewer.q": q,
                "sewer.h": h,
                "sewer.v": velocity,
                "sewer.A_air": a_air,
                "sewer.D_h": d_h,
                "sewer.T": width,
                "sewer.d_m": mean_depth,
                "sewer.R_h": radius,
                "sewer.V_wet": wetted,
                "sewer.V_air": air_volume,
            }
        )
        # The quality layers live on the MANHOLES; each manhole's capacity is the volume of
        # its own outgoing pipe (a tree guarantees exactly one), which with the upwind rule
        # is exactly SWMM's tank-in-series conduit model (Ref. Man. Vol. III Eq. 5-4).
        out[f"{self.quality_layer}.q"] = q
        out[f"{self.quality_layer}.capacity"] = wetted.index_select(-1, self.out_pipe)
        air_cap = air_volume.index_select(-1, self.out_pipe)
        if self.shaft_depth is not None:
            shaft = self.surface_area * torch.clamp(
                self.shaft_depth - h.index_select(-1, self.out_pipe), min=0.0
            )
            air_cap = air_cap + shaft
        out[f"{self.air_quality_layer}.capacity"] = air_cap
        if self.ambient_idx is not None:
            out["rho_air_nodes"] = self._densities(drivers)
        return out

    # --------------------------------------------------------------- helpers
    def _tree_flow(self, inflow: Tensor, lateral: Tensor) -> Tensor:
        """Every pipe's discharge from continuity alone.

        `particular_flow` requires the per-component source sum to be zero, so each
        outfall node absorbs its OWN component's total: a forest has one
        outfall per component, and `self.manhole_outfall_slot` (built once at construction)
        says which of `self.outfall_node_idx`'s positions is each manhole's own -- so a
        two-component forest subtracts component A's total at outfall A and component B's
        at outfall B, rather than the WHOLE network's total at every outfall (which balanced
        only a single-outfall tree and otherwise raised an unnamed `index_add_` error from
        the shape mismatch, or silently misbalanced every component but one).
        """
        sources = torch.zeros_like(inflow)
        sources = sources.index_add(-1, self.manhole_idx, lateral)
        n_outfalls = self.outfall_node_idx.numel()
        comp_totals = torch.zeros(
            lateral.shape[:-1] + (n_outfalls,), dtype=lateral.dtype
        )
        comp_totals = comp_totals.index_add(-1, self.manhole_outfall_slot, lateral)
        sources = sources.index_add(-1, self.outfall_node_idx, -comp_totals)
        return particular_flow(self.net, sources, kind="pipe")

    def _storage_sweep(
        self, state: Mapping, lateral: Tensor, dt: float
    ) -> tuple[Tensor, Tensor]:
        """Implicit Euler on each manhole's level, level-synchronous from leaves to outfall,
        over the interval `dt` (the model's own step interval, not a fixed one read off
        this closure's own constructor).

        Each manhole solves the scalar monotone equation

            A_s (H_new - H_old) / dt + Q_out(H_new) = sum of upstream Q_new + lateral

        with `solve_monotone` on `[0, 0.938 D]`, batched over the manholes of one LEVEL and
        over the ensemble. The bracket is justified exactly as `normal_depth`'s: the
        left-hand side is strictly increasing in `H_new` (both terms are), so a sign change
        is bracketed whenever the right-hand side is between the values at the two ends --
        which is why the required discharge is checked against the outgoing pipe's Manning
        capacity and REFUSED BY NAME before `solve_monotone` runs: an unbracketed root there
        would otherwise raise an unnamed batch-index error. Under constant inflow the fixed
        point IS the quasi-steady solution (verification row W7; measured 5.7e-15 relative
        after 200 steps of 60 s).

        Per level, this uses ONLY the flat `level_idx`/`level_src`/`level_tgt` index tensors
        `__init__` precomputed: one gather (`index_select`) plus one scatter (`index_add`)
        assembles the upstream inflow, so the only per-call Python loop anywhere in this
        method is over LEVELS (the tree depth) -- never over pipes or manholes.
        """
        old = self._levels_from_state(state)
        d_out = self.diameter.index_select(-1, self.out_pipe)
        n_out = self.roughness.index_select(-1, self.out_pipe)
        s_out = self.slope.index_select(-1, self.out_pipe)
        levels = old
        q_out = torch.zeros_like(old)
        for idx, src, tgt in zip(
            self.level_idx, self.level_src, self.level_tgt, strict=True
        ):
            # ONE gather plus one scatter assembles this level's upstream inflow; no Python
            # loop over this level's manholes (see the module docstring and
            # construction's `level_src`/`level_tgt`).
            upstream_sum = torch.zeros_like(lateral.index_select(-1, idx))
            if src.numel():
                upstream_sum = upstream_sum.index_add(-1, tgt, q_out.index_select(-1, src))
            target = lateral.index_select(-1, idx) + upstream_sum
            area_s = self.surface_area.index_select(-1, idx)
            h_old = levels.index_select(-1, idx)
            d_j = d_out.index_select(-1, idx)
            n_j = n_out.index_select(-1, idx)
            s_j = s_out.index_select(-1, idx)

            # Refuse a surcharge BY NAME before `solve_monotone` ever sees it: an
            # unbracketed root there raises an unnamed batch-index `RuntimeError`, which is
            # the wrong failure mode for a required discharge this closure can identify and
            # name itself.
            cap = geom.capacity_flow(d_j, n_j, s_j)
            over = target > cap
            if bool(torch.any(over)):
                # `cap` carries no batch dimension (built from the pipe-only tensors); it
                # must be broadcast up to `target`'s full (possibly batched) shape BEFORE
                # the flat reshape below, or indexing a later batch instance out of the
                # unexpanded `(1, k)` `flat_cap` raises an unnamed `IndexError` instead of
                # the named refusal this check exists to give.
                cap_b = cap.expand_as(target)
                flat_target = target.reshape(-1, target.shape[-1])
                flat_cap = cap_b.reshape(-1, target.shape[-1])
                flat_over = over.reshape(-1, over.shape[-1])
                batch_idx, local_idx = torch.nonzero(flat_over, as_tuple=True)
                idx_list = idx.tolist()
                manholes = [self.manhole_names[idx_list[i]] for i in local_idx.tolist()]
                instances = batch_idx.tolist()
                worst = [
                    float(flat_target[b, i])
                    for b, i in zip(instances, local_idx.tolist(), strict=True)
                ]
                cap_vals = [
                    float(flat_cap[b, i])
                    for b, i in zip(instances, local_idx.tolist(), strict=True)
                ]
                raise ValueError(
                    f"sewer.storage: surcharge at manhole(s) {manholes} instance(s) "
                    f"{instances}: required discharge {worst} m3/s exceeds the outgoing "
                    f"pipe's Manning capacity {cap_vals} m3/s"
                )

            def residual(h, target_t, area_t, old_t, d_t, n_t, s_t):
                return (
                    area_t * (h - old_t) / dt
                    + geom.manning_flow(h, d_t, n_t, s_t)
                    - target_t
                )

            lo = torch.zeros_like(target)
            hi = geom.H_MAX_RATIO * d_j * torch.ones_like(target)
            h_new = solve_monotone(
                residual, lo, hi, target, area_s, h_old, d_j, n_j, s_j,
                tol=1e-15, max_iter=200,
            )
            levels = levels.index_copy(-1, idx, h_new)
            q_out = q_out.index_copy(
                -1, idx, geom.manning_flow(h_new, d_j, n_j, s_j)
            )
        return self._scatter_out(q_out), levels

    def _at_levels(self, state: Mapping) -> tuple[Tensor, Tensor]:
        """Query: this closure's algebraic outputs at the given state's manhole
        levels, without advancing them -- used by `residuals`, `current_flows`, `ports` and
        the sewer report, none of which own an interval to integrate over.

        Each manhole's outgoing-pipe discharge is `Q_out(H)` alone (no `solve_monotone`,
        no upstream assembly): as in `__call__`, the outgoing pipe's
        entrance depth already IS the manhole level, so its discharge is a single
        `geom.manning_flow` evaluation, exactly the closed form `_storage_sweep` solves
        for at each level.
        """
        levels = self._levels_from_state(state)
        d_out = self.diameter.index_select(-1, self.out_pipe)
        n_out = self.roughness.index_select(-1, self.out_pipe)
        s_out = self.slope.index_select(-1, self.out_pipe)
        q_out = geom.manning_flow(levels, d_out, n_out, s_out)
        return self._scatter_out(q_out), levels

    def _levels_from_state(self, state: Mapping) -> Tensor:
        """`state["sewer.H"]`, or a named `KeyError` -- shared by `_storage_sweep` and the
        query path `_at_levels`."""
        try:
            return state["sewer.H"]
        except KeyError as exc:
            raise KeyError(
                f"SewerHydraulics {self.name!r}: state 'sewer.H' (the manhole levels) is "
                f"required when storage=True; build it with initial_state(model)"
            ) from exc

    def _scatter_out(self, q_out: Tensor) -> Tensor:
        """Each manhole's own outflow (`q_out`, manhole order) scattered into its outgoing
        pipe's position (pipe order), by construction; every other pipe position is zero."""
        q = torch.zeros_like(self.length).expand(q_out.shape[:-1] + self.length.shape)
        return q.clone().index_copy(-1, self.out_pipe, q_out)

    def _densities(self, drivers: Mapping) -> Tensor:
        """Full-node ideal-gas air density from `T_head` at the manholes and `T_amb` at
        `ambient` -- the driver the existing `Stack` drive reads (the building app's
        `_DensityClosure` pattern).

        Both drivers are resolved through `resolve_nodal_driver`: a 0-d scalar, a
        full-node vector, or a `(..., 1)` trailing singleton all step; a batched leading
        dimension is preserved throughout.
        """
        t_head = _require(drivers, "T_head", self.name)
        t_amb = _require(drivers, "T_amb", self.name)
        _validate_temperature(t_head, "T_head", self.name)
        _validate_temperature(t_amb, "T_amb", self.name)
        # The AMBIENT density fills the whole vector first, so an outfall node (which carries
        # no headspace edge and is inactive for the air layer) still holds a physical value
        # rather than a zero that would silently give a `Stack` drive an infinite buoyancy
        # if the topology ever changed.
        rho = air_density(
            resolve_nodal_driver(
                t_amb, self._node_idx, self.net.n, key="T_amb", name=self.name
            )
        )
        head = air_density(
            resolve_nodal_driver(
                t_head, self.manhole_idx, self.net.n, key="T_head", name=self.name
            )
        )
        batch = torch.broadcast_shapes(rho.shape[:-1], head.shape[:-1])
        rho = rho.expand(batch + (self.net.n,)).clone()
        head = head.expand(batch + (len(self.manhole_names),))
        return rho.index_copy(-1, self.manhole_idx, head)


def resolve_nodal_driver(
    value: Tensor, idx: Tensor, n_nodes: int, *, key: str, name: str
) -> Tensor:
    """Resolve a "full-node K; scalars broadcast" driver (`T_water`, `T_head`,
    `T_amb`, `pH`) to values AT the node positions `idx`, batched leading dims preserved.

    Three accepted layouts, by the driver's OWN trailing dimension:

    * 0-d (a bare scalar): broadcast to every position in `idx`.
    * trailing dimension exactly `n_nodes` (full-node order): gathered through
      `index_select(-1, idx)`.
    * trailing dimension exactly `1`: broadcast over `idx` (a per-instance scalar with a
      batch leading dimension, `(..., 1)`).

    Anything else is refused BY NAME: neither a bare per-manhole vector nor any other
    trailing size is accepted here, because this helper cannot tell a per-manhole vector
    from a driver error without a topology it is never given -- `H2STransfer`/
    `SulfideGeneration`'s own `out_pipe=None` convention is the place a caller hands in an
    already-per-manhole vector directly, bypassing this helper entirely.

    `idx=torch.arange(n_nodes)` resolves a driver to the WHOLE node vector unchanged (used
    for `T_amb`, which underlies every node's density before the manhole values overwrite
    it): a full-node input is returned as itself, a scalar or singleton is broadcast to
    every node, exactly the same three rules.
    """
    n = idx.numel()
    if value.dim() == 0:
        return value.expand(value.shape + (n,))
    trailing = value.shape[-1]
    if trailing == n_nodes:
        return value.index_select(-1, idx)
    if trailing == 1:
        return value.expand(value.shape[:-1] + (n,))
    raise ValueError(
        f"{name}: driver {key!r} has trailing shape {trailing}; accepted layouts are a "
        f"0-d scalar, full-node (trailing {n_nodes}), or a trailing singleton broadcast "
        f"over {n} node(s) -- got {tuple(value.shape)}"
    )


def _require(drivers: Mapping, key: str, name: str) -> Tensor:
    try:
        return torch.as_tensor(drivers[key], dtype=F64)
    except KeyError as exc:
        raise KeyError(
            f"SewerHydraulics {name!r}: driver {key!r} is required and was not given"
        ) from exc


def _validate_temperature(t: Tensor, key: str, name: str) -> None:
    """`T_head`/`T_amb` are absolute (KELVIN) temperatures: non-finite or non-positive is a
    driver error, named by the driver key and the offending instance(s) rather than left to
    surface later as a silently wrong `air_density`."""
    bad = ~torch.isfinite(t) | (t <= 0)
    if bool(torch.any(bad)):
        idx = bad.reshape(-1).nonzero().flatten().tolist()
        raise ValueError(
            f"SewerHydraulics {name!r}: driver {key!r} must be finite and positive "
            f"(Kelvin); bad at instance(s) {idx}"
        )


def _outfall_positions(net, manhole_idx: Tensor, ambient_idx) -> Tensor:
    """Every node that is neither a manhole nor `ambient` -- i.e. the outfalls."""
    mask = torch.ones(net.n, dtype=torch.bool)
    mask[manhole_idx] = False
    if ambient_idx is not None:
        mask[ambient_idx] = False
    return mask.nonzero().flatten()


def _levels(pipes, manholes) -> list[list[int]]:
    """Leaf-to-root level order over the manholes: the same order `particular_flow`'s tree
    elimination uses. One construction-time loop; its LENGTH is the tree depth."""
    position = {m.name: i for i, m in enumerate(manholes)}
    children: dict[int, list[int]] = {i: [] for i in range(len(manholes))}
    for p in pipes:
        if p.v in position and p.u in position:
            children[position[p.v]].append(position[p.u])
    depth = [0] * len(manholes)
    changed = True
    while changed:
        changed = False
        for j, kids in children.items():
            for k in kids:
                if depth[j] < depth[k] + 1:
                    depth[j] = depth[k] + 1
                    changed = True
    order: dict[int, list[int]] = {}
    for j, d in enumerate(depth):
        order.setdefault(d, []).append(j)
    return [order[d] for d in sorted(order)]

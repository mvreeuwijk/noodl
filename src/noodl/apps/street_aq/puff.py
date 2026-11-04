"""SIRANE's plume trajectory in time: the tables of the unsteady above-roof puff model.

`plume_table` resamples the 10 s scan of the plume (`plume._trajectory_nodes`) onto a
distance grid for the steady kernel. A puff reads the same scan by its age: `puff_tables`
returns the nodes `t_j = 10 j` s of `X`, `U`, `sigma_y`, `sigma_z` and the centre height
`z_c`, one table per release class (the centre law `z_c = max(H_R, d + E|N(c - d,
(k sigma_z)^2)|)` takes its own `c`, so an above-roof release at height `z_s` has its own
table), and `PuffTables.at_age` interpolates them linearly in age. The speed is the
scan's per-segment node speed (`X_j = X_{j-1} + U_j dt`), so a puff that advances its
displacement by `U(a) dt` per step follows the table's `X` exactly under constant wind.

The puffs themselves (design Sect. 4.3-4.6): a `PuffField` holds the sources' geometry
(streets as `plume.source_points`, junctions as `plume.junction_sources`, explicit
releases as points), a `PuffState` the puffs in fixed `(B, K)` slots, and `emit`,
`advance`, `roof_concentration` and `concentration_at` are pure functions of them. A puff
of age `a` is a slug along the wind spanning `[x(a - dt/2), x(a + dt/2)]` from its source,
so the puffs of a continuous source tile the axis exactly for any step `dt`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from noodl.apps.street_aq.canyon import KAPPA
from noodl.apps.street_aq.closures import SIGMA_V_MIN, SIGMA_W_MIN
from noodl.apps.street_aq.network import StreetNetwork
from noodl.apps.street_aq.plume import (
    _DTYPE,
    _SQRT_2PI,
    DEFAULT_TEMPERATURE_K,
    DOWNWIND_CUTOFF_M,
    GAUSS_CUTOFF_SIGMAS,
    PLUME_CENTRE_K,
    TIME_STEP_S,
    _f64,
    _step_bound,
    _table_inputs,
    _trajectory_nodes,
    crosswind_profile,
    junction_sources,
    rows_per_chunk,
    source_points,
    street_midpoints,
    table_chunks,
    vertical_profile,
)

_NODE_KEYS = ("x", "u", "sigma_y", "sigma_z", "z_c")


@dataclass(frozen=True)
class PuffTables:
    """The plume trajectory by age (`puff_tables`).

    `x` (m, the distance travelled), `u` (m/s, the speed of the segment ending at the node:
    segments are right-closed;
    node 0 holds the speed at `H_R`), `sigma_y`, `sigma_z` (m) and `z_c` (m, the plume
    centre height; node 0 holds node 1's) are `(n_tab, n_b, n_nodes)` at ages `0, step_s,
    2 step_s, ...`; a row that reaches `x_max` before the others repeats its last node.
    `h_canopy` and `h_abl` (m) are `(n_b,)`; `x_max` is the extent the tables were built to
    (the last node of each row is the first beyond it)."""

    x: Tensor
    u: Tensor
    sigma_y: Tensor
    sigma_z: Tensor
    z_c: Tensor
    h_canopy: Tensor
    h_abl: Tensor
    x_max: float
    step_s: float = TIME_STEP_S

    @property
    def t_max(self) -> float:
        """The age of the last node, s."""
        return (self.x.shape[-1] - 1) * self.step_s

    def at_age(self, age: Tensor, table: Tensor, row: Tensor
               ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """`(x, u, sigma_y, sigma_z, z_c)` at `age` (s) from table `table` and
        meteorology row `row` (long tensors that broadcast with `age`; the result has their
        broadcast shape). Everything but `u` is linear in age between nodes; `u` is the
        node speed of the segment `(t_{j-1}, t_j]` containing `age`, i.e. the speed of the first
        segment ending at or after `age` (not interpolated; age 0 gives `U_1`). An age beyond the
        last node returns the last node's values (a later step retires such a puff); a
        negative age the source node's."""
        _, n_b, n = self.x.shape
        pos = torch.clamp(age / self.step_s, 0.0, float(n - 1))
        floor = torch.floor(pos.detach()).long()
        i0 = torch.clamp(floor, 0, n - 2)
        frac = pos - i0
        base = (table * n_b + row) * n
        lo = base + i0
        seg = base + torch.clamp(torch.ceil(pos.detach()).long(), 1, n - 1)
        out = []
        for node in (self.x, self.u, self.sigma_y, self.sigma_z, self.z_c):
            v = node.reshape(-1)
            if node is self.u:
                out.append(v[seg])
            else:
                out.append(v[lo] * (1.0 - frac) + v[lo + 1] * frac)
        return out[0], out[1], out[2], out[3], out[4]


class _RecomputedNodes(torch.autograd.Function):
    """One chunk of `_trajectory_nodes` that keeps no graph (the `_RecomputedTrajectory`
    pattern of `plume`): the forward runs without one and the backward rebuilds the chunk's
    graph and differentiates it. First derivatives only; `create_graph=True` raises."""

    @staticmethod
    def forward(ctx, keys, kappa, x_end, *values):
        ctx.keys, ctx.kappa, ctx.x_end = keys, kappa, x_end
        ctx.save_for_backward(*values)
        nodes = _trajectory_nodes(dict(zip(keys, values, strict=True)), None, kappa,
                                  x_end=x_end)
        return tuple(nodes[k] for k in _NODE_KEYS)

    @staticmethod
    def backward(ctx, *grads):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "puff_tables: second-order differentiation (create_graph=True) is not "
                "supported; the trajectory tables have first-order gradients only. Detach "
                "the first-order gradient before using it in a further differentiable loss."
            )
        values = ctx.saved_tensors
        leaves = [v.detach().requires_grad_(ctx.needs_input_grad[3 + i])
                  for i, v in enumerate(values)]
        wanted = [v for v in leaves if v.requires_grad]
        with torch.enable_grad():
            nodes = _trajectory_nodes(dict(zip(ctx.keys, leaves, strict=True)), None,
                                      ctx.kappa, x_end=ctx.x_end)
            found = iter(torch.autograd.grad([nodes[k] for k in _NODE_KEYS], wanted, grads,
                                             allow_unused=True))
        return (None, None, None,
                *(next(found) if v.requires_grad else None for v in leaves))


def _pad_nodes(a: Tensor, n: int) -> Tensor:
    """`a` `(m, n_a)` extended to `n` columns by repeating its last column."""
    return a if a.shape[1] == n else torch.cat(
        [a, a[:, -1:].expand(-1, n - a.shape[1])], dim=1)


def puff_tables(
    *,
    centres: Tensor,
    u_star: Tensor | float,
    h_abl: Tensor | float,
    lmo: Tensor | float,
    h_canopy: Tensor | float,
    z0: Tensor | float,
    d: Tensor | float,
    sigma_theta: Tensor | float,
    theta_star: Tensor | float | None = None,
    temperature: Tensor | float = DEFAULT_TEMPERATURE_K,
    sigma_v_min: float = SIGMA_V_MIN,
    sigma_w_min: float = SIGMA_W_MIN,
    tau_y: Tensor | float | None = None,
    tau_z: Tensor | float | None = None,
    centre_k: Tensor | float = PLUME_CENTRE_K,
    x_max: float = DOWNWIND_CUTOFF_M,
    kappa: float = KAPPA,
) -> PuffTables:
    """SIRANE's plume trajectory by age (module docstring), one table per entry of
    `centres`.

    `centres` (m) is `centre_c` of the centre law, `(n_tab,)` or `(n_tab, n_b)`; the
    meteorology (`u_star`, `h_abl`, `lmo`, `sigma_theta`, `theta_star`, `temperature`, and
    the empirical inputs, all as in `plume_table`) is scalar or `(n_b,)` and broadcasts
    to `n_b` hours. `x_max` is the extent: each row's scan ends at the first node beyond it.
    Needs `h_abl > h_canopy > d`, `u* > 0` and `tau_z < TIME_STEP_S`; a wind at `H_R` so
    low that the scan would need more than `MAX_TABLE_STEPS` steps, and the unstable regime,
    are refused by name. Rows are built in chunks under `KERNEL_BYTES_CAP`
    (`table_chunks`)."""
    centres = torch.as_tensor(centres, dtype=_DTYPE)
    if centres.ndim not in (1, 2):
        raise ValueError(
            f"puff_tables: centres must have shape (n_tab,) or (n_tab, n_b), got "
            f"{tuple(centres.shape)}")
    p, batch, u0 = _table_inputs(
        "puff_tables", u_star=u_star, h_abl=h_abl, lmo=lmo, h_canopy=h_canopy, z0=z0, d=d,
        sigma_theta=sigma_theta, theta_star=theta_star, temperature=temperature,
        sigma_v_min=sigma_v_min, sigma_w_min=sigma_w_min, tau_y=tau_y, tau_z=tau_z,
        centre_c=centres if centres.ndim == 2 else centres[:, None], centre_k=centre_k,
        x_max=x_max, kappa=kappa)
    if len(batch) != 2:
        raise ValueError(
            f"puff_tables: the meteorology must be scalar or (n_b,) and centres (n_tab,) "
            f"or (n_tab, n_b); the broadcast shape is {batch}")
    n_tab, n_b = batch
    x_end = float(x_max)
    steps = _step_bound("puff_tables", u0, x_end)
    keys = tuple(p)
    grad = torch.is_grad_enabled() and any(v.requires_grad for v in p.values())
    out = []
    for a_, b_ in table_chunks(steps.tolist()):
        values = [p[k][a_:b_] for k in keys]
        if grad:
            out.append(_RecomputedNodes.apply(keys, kappa, x_end, *values))
        else:
            nodes = _trajectory_nodes(dict(zip(keys, values, strict=True)), None, kappa,
                                      x_end=x_end)
            out.append(tuple(nodes[k] for k in _NODE_KEYS))
    n_nodes = max(o[0].shape[1] for o in out)
    cols = [torch.cat([_pad_nodes(o[i], n_nodes) for o in out]).reshape(n_tab, n_b, n_nodes)
            for i in range(len(_NODE_KEYS))]
    return PuffTables(*cols, h_canopy=p["h_r"].reshape(batch)[0],
                      h_abl=p["h_abl"].reshape(batch)[0], x_max=x_end)


ALONG_WIND = ("sigma_y", "none")
"""The along-wind profile options of a puff (`PuffField`)."""

_STATE_FIELDS = ("mass", "age", "disp", "source", "table", "live", "sxz", "sx2_shear",
                 "mass_emitted", "mass_out", "mass_dropped", "mass_drawn_down")


@dataclass(frozen=True)
class PuffState:
    """The puffs of `B` batch instances in `K` fixed slots, and the mass counters.

    Per slot `(B, K)`: `mass` (kg, signed), `age` (s), `disp` (`(B, K, 2)`, m: the
    displacement of the puff's source geometry since it was emitted), `source` (the
    `PuffField` source index), `table` (the `PuffTables` table index), `live`, and `sxz`,
    `sx2_shear` (the shear covariance and along-wind variance, zero unless shear is on).
    Per instance `(B,)`: `mass_emitted`, `mass_out`, `mass_dropped`, `mass_drawn_down`
    (kg). A free slot has `live` False and zero mass. Functions take a state and return a
    new one; nothing is mutated."""

    mass: Tensor
    age: Tensor
    disp: Tensor
    source: Tensor
    table: Tensor
    live: Tensor
    sxz: Tensor
    sx2_shear: Tensor
    mass_emitted: Tensor
    mass_out: Tensor
    mass_dropped: Tensor
    mass_drawn_down: Tensor

    @classmethod
    def empty(cls, batch: tuple[int, ...], capacity: int) -> PuffState:
        """No puffs: `batch` is `(B,)`, `capacity` the slots `K` per instance."""
        batch = tuple(int(b) for b in batch)
        if len(batch) != 1 or batch[0] < 1:
            raise ValueError(
                f"PuffState.empty: batch must be (B,) with B >= 1, got {batch}; flatten a "
                f"multi-dimensional batch to (B,)")
        if not int(capacity) >= 1:
            raise ValueError(f"PuffState.empty: capacity must be >= 1, got {capacity!r}")
        b, k = batch[0], int(capacity)

        def zeros(*shape: int) -> Tensor:
            return torch.zeros(*shape, dtype=_DTYPE)

        index = torch.zeros(b, k, dtype=torch.long)
        return cls(mass=zeros(b, k), age=zeros(b, k), disp=zeros(b, k, 2), source=index,
                   table=index.clone(), live=torch.zeros(b, k, dtype=torch.bool),
                   sxz=zeros(b, k), sx2_shear=zeros(b, k), mass_emitted=zeros(b),
                   mass_out=zeros(b), mass_dropped=zeros(b), mass_drawn_down=zeros(b))

    def to_state(self, prefix: str = "puff") -> dict[str, Tensor]:
        """The state as `{"<prefix>.mass": ..., ...}`, one key per field."""
        return {f"{prefix}.{name}": getattr(self, name) for name in _STATE_FIELDS}

    @classmethod
    def from_state(cls, state: Mapping[str, Tensor], prefix: str = "puff") -> PuffState:
        """The inverse of `to_state`; a missing key raises `KeyError` naming it."""
        missing = [f"{prefix}.{n}" for n in _STATE_FIELDS if f"{prefix}.{n}" not in state]
        if missing:
            raise KeyError(f"PuffState.from_state: the state has no {missing}")
        return cls(**{name: state[f"{prefix}.{name}"] for name in _STATE_FIELDS})


class PuffField:
    """The sources puffs are emitted from, and the evaluation options.

    Sources are indexed streets `0 .. n_s - 1`, junctions `n_s .. n_s + n_j - 1`, then
    explicit above-roof releases. Each source is a contiguous run of sub-sources: `xy`
    `(P, 2)` (m), `owner` `(P,)` (the source), `weight` `(P,)` (its share of the source's
    mass), `width` `(B_w, P)` (m, the flat-top crosswind width; `B_w` is 1 or the batch)
    and `clamp` `(B_w, P)` (1/m, the roof-level cap on `P_z`). A puff's sub-sources all
    move by the puff's displacement. `source_xy0` `(S, 2)` is each source's reference
    point: a street's midpoint, a junction's position, and the origin for a release, whose
    sub-source sits at the origin so that its emission position is its displacement.

    `along_wind` (one of `ALONG_WIND`) is the along-wind profile: `"sigma_y"` (the default)
    a slug convolved with a Gaussian of `sigma_x = sigma_y(a)`, `"none"` the slug alone
    (design Sect. 4.3). `cutoff_sigma` is the cut-off in standard deviations beyond the
    flat top (crosswind) and beyond the slug ends (along the wind)."""

    def __init__(self, *, xy: Tensor, owner: Tensor, weight: Tensor, width: Tensor,
                 clamp: Tensor, source_xy0: Tensor, n_streets: int, n_junctions: int,
                 n_releases: int, along_wind: str = "sigma_y",
                 cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS) -> None:
        if along_wind not in ALONG_WIND:
            raise ValueError(
                f"PuffField: along_wind must be one of {ALONG_WIND}, got {along_wind!r}")
        if not cutoff_sigma > 0:
            raise ValueError(f"PuffField: cutoff_sigma must be positive, got {cutoff_sigma!r}")
        n_sources = n_streets + n_junctions + n_releases
        n_sub = xy.shape[0]
        self.xy, self.owner, self.weight = _f64(xy), owner.long(), _f64(weight)
        self.width = _f64(width).reshape(-1, n_sub)
        self.clamp = _f64(clamp).reshape(-1, n_sub)
        self.source_xy0 = _f64(source_xy0)
        self.n_streets, self.n_junctions, self.n_releases = n_streets, n_junctions, n_releases
        self.along_wind, self.cutoff_sigma = along_wind, float(cutoff_sigma)
        count = torch.bincount(self.owner, minlength=n_sources)
        if bool((self.owner[1:] < self.owner[:-1]).any()) or bool((count == 0).any()) \
                or count.numel() != n_sources:
            raise ValueError("PuffField: every source needs a contiguous run of sub-sources "
                             "in source order")
        self._count = count
        self._start = torch.cumsum(count, 0) - count
        # detached geometry of the coarse mask: each source's sub-source bounding box and
        # its widest flat top
        xy_d = self.xy.detach()
        index = self.owner[:, None].expand(-1, 2)
        self._box_lo = torch.full((n_sources, 2), math.inf, dtype=_DTYPE).scatter_reduce(
            0, index, xy_d, "amin")
        self._box_hi = torch.full((n_sources, 2), -math.inf, dtype=_DTYPE).scatter_reduce(
            0, index, xy_d, "amax")
        w = self.width.detach()
        self._half_width = torch.zeros(w.shape[0], n_sources, dtype=_DTYPE).scatter_reduce(
            1, self.owner.expand_as(w), w, "amax") / 2.0

    @property
    def n_sources(self) -> int:
        """The number of sources `S`."""
        return self.n_streets + self.n_junctions + self.n_releases

    @classmethod
    def for_network(cls, net: StreetNetwork, *, theta_w: Tensor | float,
                    n_releases: int = 0, along_wind: str = "sigma_y",
                    cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS) -> PuffField:
        """The streets of `net` as `plume.source_points` (whose widths and clamps depend
        on the wind `theta_w`, scalar or `(B,)`), its junctions as `plume.junction_sources`,
        then `n_releases` explicit above-roof releases (points of zero width and no clamp)."""
        src = source_points(net, theta_w=theta_w)
        j_xy, j_width, j_clamp = junction_sources(net)
        n_s, n_j, n_r = len(net.streets), j_xy.shape[0], int(n_releases)
        n_p = src.xy.shape[0]
        width = src.width.reshape(-1, n_p)
        n_w = width.shape[0]
        return cls(
            xy=torch.cat([src.xy, j_xy, torch.zeros(n_r, 2, dtype=_DTYPE)]),
            owner=torch.cat([src.owner, n_s + torch.arange(n_j + n_r)]),
            weight=torch.cat([src.weight, torch.ones(n_j + n_r, dtype=_DTYPE)]),
            width=torch.cat([width, j_width.expand(n_w, n_j),
                             torch.zeros(n_w, n_r, dtype=_DTYPE)], dim=1),
            clamp=torch.cat([src.clamp.reshape(n_w, n_p), j_clamp.expand(n_w, n_j),
                             torch.full((n_w, n_r), math.inf, dtype=_DTYPE)], dim=1),
            source_xy0=torch.cat([street_midpoints(net), j_xy,
                                  torch.zeros(n_r, 2, dtype=_DTYPE)]),
            n_streets=n_s, n_junctions=n_j, n_releases=n_r, along_wind=along_wind,
            cutoff_sigma=cutoff_sigma)

    @classmethod
    def points(cls, n_releases: int, *, along_wind: str = "sigma_y",
               cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS) -> PuffField:
        """`n_releases` explicit releases and no network: the standalone puff model."""
        n = int(n_releases)
        if n < 1:
            raise ValueError(f"PuffField.points: n_releases must be >= 1, got {n_releases!r}")
        return cls(xy=torch.zeros(n, 2, dtype=_DTYPE), owner=torch.arange(n),
                   weight=torch.ones(n, dtype=_DTYPE), width=torch.zeros(1, n, dtype=_DTYPE),
                   clamp=torch.full((1, n), math.inf, dtype=_DTYPE),
                   source_xy0=torch.zeros(n, 2, dtype=_DTYPE), n_streets=0, n_junctions=0,
                   n_releases=n, along_wind=along_wind, cutoff_sigma=cutoff_sigma)


def _table_rows(tables: PuffTables, n_batch: int, where: str) -> Tensor:
    """`(B,)`: each batch instance's meteorology row of `tables` (its own when the tables
    have `B` rows, row 0 when they have one)."""
    n_b = tables.x.shape[1]
    if n_b == n_batch:
        return torch.arange(n_batch)
    if n_b == 1:
        return torch.zeros(n_batch, dtype=torch.long)
    raise ValueError(f"{where}: the tables have {n_b} meteorology rows; the puffs need 1 or "
                     f"their batch size {n_batch}")


def _wind(theta_w: Tensor | float, n_batch: int, where: str) -> tuple[Tensor, Tensor]:
    theta = _f64(theta_w)
    if theta.numel() not in (1, n_batch):
        raise ValueError(f"{where}: theta_w must be scalar or ({n_batch},), got shape "
                         f"{tuple(theta.shape)}")
    theta = theta.reshape(-1).expand(n_batch)
    return torch.cos(theta), torch.sin(theta)


def _field_rows(field: PuffField, n_batch: int, where: str) -> tuple[Tensor, Tensor]:
    """`(width, clamp)` of the field expanded to `(B, P)`; a batch of 1 or `B` only."""
    for name in ("width", "clamp"):
        n_w = getattr(field, name).shape[0]
        if n_w not in (1, n_batch):
            raise ValueError(f"{where}: the field's {name} has {n_w} batch rows; the puffs "
                             f"need 1 or their batch size {n_batch}")
    return field.width.expand(n_batch, -1), field.clamp.expand(n_batch, -1)


def puff_mass(state: PuffState) -> Tensor:
    """`(B,)`: the summed mass of the live puffs, kg."""
    return torch.where(state.live, state.mass, torch.zeros_like(state.mass)).sum(1)


def _peak_bound(field: PuffField, state: PuffState, tables: PuffTables) -> Tensor:
    """`(B, K)`, detached: a cheap upper bound of each live puff's peak roof-level
    contribution, `|m| max_i w_i min(P_z,max, clamp_i) / (ell (2 y_c,i + sqrt(2 pi)
    sigma_y))` (design Sect. 4.8), with `P_z,max` the vertical profile at `H_R`. The
    slug length `ell` is the node speed (the step `dt` is common to all puffs and cancels
    in the comparisons the bound serves). Zero for a free slot."""
    n_batch, capacity = state.mass.shape
    width, clamp = _field_rows(field, n_batch, "housekeep")
    width, clamp = width.detach(), clamp.detach()
    row = _table_rows(tables, n_batch, "housekeep")[:, None]
    with torch.no_grad():
        _, u, s_y, s_z, z_c = tables.at_age(state.age, state.table, row)
        h_c, h_abl = tables.h_canopy[row[:, 0]], tables.h_abl[row[:, 0]]
        p_z = vertical_profile(h_c[:, None].expand_as(s_y), z_c, s_z, h_c[:, None],
                               h_abl[:, None])
        b, k = torch.nonzero(state.live & (state.mass != 0), as_tuple=True)
        s = state.source[b, k]
        n_sub = field._count[s]
        rep = torch.repeat_interleave(torch.arange(b.numel()), n_sub)
        offset = torch.arange(rep.numel()) - torch.repeat_interleave(
            torch.cumsum(n_sub, 0) - n_sub, n_sub)
        p = field._start[s][rep] + offset
        b_, k_ = b[rep], k[rep]
        sy = s_y[b_, k_]
        w = width[b_, p]
        value = field.weight[p].detach() * torch.minimum(p_z[b_, k_], clamp[b_, p]) / (
            2.0 * torch.clamp((w - _SQRT_2PI * sy) / 2.0, min=0.0) + _SQRT_2PI * sy)
        best = torch.zeros(b.numel(), dtype=_DTYPE).scatter_reduce(0, rep, value, "amax",
                                                                   include_self=False)
        peak = torch.zeros(n_batch, capacity, dtype=_DTYPE)
        peak[b, k] = state.mass[b, k].detach().abs() * best / torch.clamp(u[b, k], min=1e-12)
    return peak


def _retire(state: PuffState, gone: Tensor, counter: str) -> PuffState:
    """The puffs `gone` (B, K) retired into the per-instance `counter`: their mass moves
    with `torch.where`, so the counter keeps the graph."""
    zero = torch.zeros_like(state.mass)
    moved = torch.where(gone, state.mass, zero).sum(1)
    return replace(state, mass=torch.where(gone, zero, state.mass), live=state.live & ~gone,
                   **{counter: getattr(state, counter) + moved})


def housekeep(field: PuffField, state: PuffState, tables: PuffTables, *,
              drop_below: float = 1e-6) -> PuffState:
    """Retire puffs (design Sect. 4.8): those whose age passed the table's last node or
    whose distance `x(age)` passed `tables.x_max` go to `mass_out`; of the others, those
    whose peak bound (`_peak_bound`) is below `drop_below` times the largest among the
    instance's live puffs go to `mass_dropped`. An instance with no live puff is left as
    it is. Decisions are made on detached values; the masses keep their graph."""
    if not 0.0 <= drop_below < 1.0:
        raise ValueError(f"housekeep: drop_below must be in [0, 1), got {drop_below!r}")
    n_batch = state.mass.shape[0]
    row = _table_rows(tables, n_batch, "housekeep")[:, None]
    with torch.no_grad():
        x = tables.at_age(state.age, state.table, row)[0]
        out = state.live & ((state.age > tables.t_max) | (x > tables.x_max))
    state = _retire(state, out, "mass_out")
    peak = _peak_bound(field, state, tables)
    with torch.no_grad():
        faint = state.live & (peak < drop_below * peak.max(1, keepdim=True).values)
    return _retire(state, faint, "mass_dropped")


def emit(field: PuffField, state: PuffState, masses: Tensor, table_index: Tensor,
         release_xy: Tensor | None = None, *, tables: PuffTables | None = None
         ) -> tuple[PuffState, Tensor]:
    """One new puff per source with non-zero mass in `masses` `(B, S)` (kg, signed),
    written into the first free slots of its instance in source order: age 0, table
    `table_index[s]` (`(S,)`), displacement zero, except a release's, which is its
    position in `release_xy` `(B, R, 2)` (m) relative to its `source_xy0`. `mass_emitted`
    accumulates the masses. An instance without enough free slots evicts its faintest live
    puffs into `mass_dropped` (by the peak bound of `housekeep` when `tables` is given,
    else by `|m|`); the state and the number evicted `(B,)` are returned. More new puffs
    than the capacity is refused by name."""
    n_batch, capacity = state.mass.shape
    n_src, n_rel = field.n_sources, field.n_releases
    masses = _f64(masses)
    if masses.shape != (n_batch, n_src):
        raise ValueError(f"emit: masses must be (B, S) = {(n_batch, n_src)}, got "
                         f"{tuple(masses.shape)}")
    table_index = torch.as_tensor(table_index, dtype=torch.long)
    if table_index.shape != (n_src,):
        raise ValueError(f"emit: table_index must be (S,) = ({n_src},), got "
                         f"{tuple(table_index.shape)}")
    disp0 = torch.zeros(n_batch, n_src - n_rel, 2, dtype=_DTYPE)
    if n_rel:
        if release_xy is None:
            raise ValueError(f"emit: the field has {n_rel} releases; pass release_xy "
                             f"({n_batch}, {n_rel}, 2), their positions in m")
        release_xy = _f64(release_xy)
        if release_xy.shape != (n_batch, n_rel, 2):
            raise ValueError(f"emit: release_xy must be (B, R, 2) = {(n_batch, n_rel, 2)}, "
                             f"got {tuple(release_xy.shape)}")
        disp0 = torch.cat([disp0, release_xy - field.source_xy0[n_src - n_rel:]], dim=1)
    new = masses.detach() != 0
    n_new = new.sum(1)
    if bool((n_new > capacity).any()):
        b = int(torch.nonzero(n_new > capacity)[0, 0])
        raise ValueError(
            f"emit: batch instance {b} emits {int(n_new[b])} puffs, more than the capacity "
            f"{capacity}; raise the capacity")
    n_evict = torch.clamp(n_new - (~state.live).sum(1), min=0)
    if bool((n_evict > 0).any()):
        with torch.no_grad():
            weak = (_peak_bound(field, state, tables) if tables is not None
                    else state.mass.abs())
            key = torch.where(state.live, weak, torch.full_like(weak, math.inf))
            rank = torch.argsort(torch.argsort(key, dim=1, stable=True), dim=1)
            evict = state.live & (rank < n_evict[:, None])
        state = _retire(state, evict, "mass_dropped")
    free = ~state.live
    b, s = torch.nonzero(new, as_tuple=True)
    rank = (torch.cumsum(new, 1) - 1)[b, s]
    fb, fk = torch.nonzero(free, as_tuple=True)
    slot = torch.zeros(n_batch, capacity, dtype=torch.long).index_put(
        (fb, (torch.cumsum(free, 1) - 1)[fb, fk]), fk)
    at = (b, slot[b, rank])
    zero = torch.zeros(b.numel(), dtype=_DTYPE)
    state = replace(
        state,
        mass=state.mass.index_put(at, masses[b, s]),
        age=state.age.index_put(at, zero),
        disp=state.disp.index_put(at, disp0[b, s]),
        source=state.source.index_put(at, s),
        table=state.table.index_put(at, table_index[s]),
        live=state.live.index_put(at, torch.ones_like(b, dtype=torch.bool)),
        sxz=state.sxz.index_put(at, zero),
        sx2_shear=state.sx2_shear.index_put(at, zero),
        mass_emitted=state.mass_emitted + masses.sum(1),
    )
    return state, n_evict


def advance(state: PuffState, tables: PuffTables, dt: float,
            theta_w: Tensor | float) -> PuffState:
    """Every live puff `dt` (s) older and moved along the wind `theta_w` (scalar or `(B,)`,
    radians counter-clockwise from east, toward) by its table's distance increment
    `x(a + dt) - x(a)`: exact for any `dt`, since `x` is the integral of the node speed,
    so under constant wind the displacement is `x(a)`. A wind that turns between steps
    turns the puff's further path (design Sect. 4.4)."""
    if not dt > 0:
        raise ValueError(f"advance: dt must be positive, got {dt!r}")
    n_batch = state.mass.shape[0]
    cos, sin = _wind(theta_w, n_batch, "advance")
    row = _table_rows(tables, n_batch, "advance")[:, None]
    age = state.age + dt
    step = (tables.at_age(age, state.table, row)[0]
            - tables.at_age(state.age, state.table, row)[0])
    disp = state.disp + step[..., None] * torch.stack([cos, sin], dim=-1)[:, None, :]
    return replace(state, age=torch.where(state.live, age, state.age),
                   disp=torch.where(state.live[..., None], disp, state.disp))


def _evaluate(field: PuffField, state: PuffState, tables: PuffTables, rec_xy: Tensor,
              rec_z: Tensor | None, theta_w: Tensor | float, dt: float,
              receptor_owner: Tensor | None, where: str) -> Tensor:
    """`(B, R)`: the puffs at receptors `rec_xy` `(R, 2)` and heights `rec_z` `(R,)`, or at
    roof level `H_R` with each sub-source's clamp when `rec_z` is None.

    Two levels, as `plume._pair_values`: a coarse mask of (puff, receptor) pairs by the
    source's displaced bounding box grown by the slug ends, `cutoff_sigma` spreads and its
    widest flat top, then the surviving pairs expanded to the source's sub-sources and cut
    exactly. Masks are detached; the kept values carry the graph. A puff of age 0 (not yet
    advanced: no spread) or whose slug has zero length (beyond the table's last node)
    contributes nothing."""
    if not dt > 0:
        raise ValueError(f"{where}: dt must be positive, got {dt!r}")
    n_batch, capacity = state.mass.shape
    cos, sin = _wind(theta_w, n_batch, where)
    row = _table_rows(tables, n_batch, where)
    rk = row[:, None]
    age = state.age
    x_a, _, s_y, s_z, z_c = tables.at_age(age, state.table, rk)
    e_lo = tables.at_age(torch.clamp(age - dt / 2.0, min=0.0), state.table, rk)[0] - x_a
    e_hi = tables.at_age(age + dt / 2.0, state.table, rk)[0] - x_a
    slug = field.along_wind == "sigma_y"
    cut = field.cutoff_sigma
    width, clamp = _field_rows(field, n_batch, where)
    h_c, h_abl = tables.h_canopy[row], tables.h_abl[row]
    src = state.source
    with torch.no_grad():
        active = state.live & (state.mass != 0) & (age > 0) & (e_hi > e_lo)
        reach = (torch.maximum(-e_lo, e_hi) + (cut * s_y if slug else 0.0)
                 + field._half_width.expand(n_batch, -1).gather(1, src) + cut * s_y)
        reach = torch.where(active, reach, torch.zeros_like(reach))
        box_lo = field._box_lo[src] + state.disp - reach[..., None]
        box_hi = field._box_hi[src] + state.disp + reach[..., None]
    rec_xy = _f64(rec_xy)
    n_rec = rec_xy.shape[0]
    rec_d = rec_xy.detach()
    grad = torch.is_grad_enabled() and any(
        v.requires_grad for v in (state.mass, state.disp, tables.x, tables.sigma_y,
                                  tables.z_c, field.width, rec_xy, cos))

    def block(a_: int, b_: int) -> Tensor:
        r = b_ - a_
        pts = rec_d[a_:b_]
        mask = (active[..., None]
                & (pts[:, 0] >= box_lo[..., 0, None]) & (pts[:, 0] <= box_hi[..., 0, None])
                & (pts[:, 1] >= box_lo[..., 1, None]) & (pts[:, 1] <= box_hi[..., 1, None]))
        if receptor_owner is not None:
            mask &= src[..., None] != receptor_owner[a_:b_]
        b, k, i = torch.nonzero(mask, as_tuple=True)
        del mask
        s = src[b, k]
        n_sub = field._count[s]
        rep = torch.repeat_interleave(torch.arange(b.numel()), n_sub)
        offset = torch.arange(rep.numel()) - torch.repeat_interleave(
            torch.cumsum(n_sub, 0) - n_sub, n_sub)
        p = field._start[s][rep] + offset
        b, k, i = b[rep], k[rep], i[rep]
        d = rec_xy[a_:b_][i] - (field.xy[p] + state.disp[b, k])
        x = d[:, 0] * cos[b] + d[:, 1] * sin[b]
        y = d[:, 1] * cos[b] - d[:, 0] * sin[b]
        sy, lo, hi, w = s_y[b, k], e_lo[b, k], e_hi[b, k], width[b, p]
        with torch.no_grad():
            y_c = torch.clamp((w - _SQRT_2PI * sy) / 2.0, min=0.0)
            keep = y.abs() - y_c <= cut * sy
            if slug:
                keep &= (x >= lo - cut * sy) & (x <= hi + cut * sy)
            else:
                keep &= (x >= lo) & (x < hi)
        b, k, i, p, x, y, sy, lo, hi, w = (
            v[keep] for v in (b, k, i, p, x, y, sy, lo, hi, w))
        if slug:
            root2 = math.sqrt(2.0) * sy
            g_x = (torch.erf((x - lo) / root2) - torch.erf((x - hi) / root2)) / (
                2.0 * (hi - lo))
        else:
            g_x = 1.0 / (hi - lo)
        if rec_z is None:
            z = h_c[b]
        else:
            z = rec_z[a_:b_][i]
        p_z = vertical_profile(z, z_c[b, k], s_z[b, k], h_c[b], h_abl[b])
        if rec_z is None:
            p_z = torch.minimum(p_z, clamp[b, p])
        c = state.mass[b, k] * field.weight[p] * g_x * crosswind_profile(y, sy, w) * p_z
        return torch.zeros(n_batch * r, dtype=_DTYPE).index_add(0, b * r + i, c).reshape(
            n_batch, r)

    rows = rows_per_chunk(capacity * int(field._count.max()), n_batch, backward=grad)
    if n_rec == 0:
        return torch.zeros(n_batch, 0, dtype=_DTYPE)
    return torch.cat([block(a_, min(n_rec, a_ + rows)) for a_ in range(0, n_rec, rows)],
                     dim=1)


def roof_concentration(field: PuffField, state: PuffState, tables: PuffTables,
                       receptors_xy: Tensor, theta_w: Tensor | float, dt: float, *,
                       receptor_owner: Tensor | None = None) -> Tensor:
    """`(B, R)`: the puffs' concentration (kg/m3) at roof level `H_R` above
    `receptors_xy` `(R, 2)`, design Sect. 4.3: per sub-source `m w_i g_x f_y min(P_z,
    clamp_i)`, with `g_x` the slug of the puff's age `a`, `[x(a - dt/2), x(a + dt/2)]` along
    the wind `theta_w` from its source (`dt`, s, the step: consecutive puffs of a
    continuous source tile the axis), and `f_y`, `P_z` read at age `a`. A puff whose source
    is `receptor_owner[r]` (`(R,)`, long; -1 for none) contributes nothing to receptor
    `r`: a street's own `C_ext` excludes its own puffs, as `street_kernel` does."""
    if receptor_owner is not None:
        receptor_owner = torch.as_tensor(receptor_owner, dtype=torch.long)
        if receptor_owner.shape != (receptors_xy.shape[0],):
            raise ValueError(f"roof_concentration: receptor_owner must be (R,) = "
                             f"({receptors_xy.shape[0]},), got "
                             f"{tuple(receptor_owner.shape)}")
    return _evaluate(field, state, tables, receptors_xy, None, theta_w, dt, receptor_owner,
                     "roof_concentration")


def concentration_at(field: PuffField, state: PuffState, tables: PuffTables,
                     points: Tensor, theta_w: Tensor | float, dt: float) -> Tensor:
    """`(B, R)`: the puffs' concentration (kg/m3) at `points` `(R, 3)` (x, y, z in m), as
    `roof_concentration` but at each point's own height and without the clamp (SIRANE's
    roof-level device), and with no receptor excluded."""
    points = _f64(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"concentration_at: points must be (R, 3), got "
                         f"{tuple(points.shape)}")
    return _evaluate(field, state, tables, points[:, :2], points[:, 2], theta_w, dt, None,
                     "concentration_at")


__all__ = ["ALONG_WIND", "PuffField", "PuffState", "PuffTables", "advance",
           "concentration_at", "emit", "housekeep", "puff_mass", "puff_tables",
           "roof_concentration"]

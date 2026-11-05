"""SIRANE's plume trajectory in time: the tables of the unsteady above-roof puff model.

`plume_table` resamples the 10 s scan of the plume (`plume._trajectory_nodes`) onto a
10 m distance grid for the steady kernel. The puffs read that same grid: `puff_tables`
returns the kernel's `u`, `sigma_y` and roof-level `p_z` with the scan's `sigma_z` and
centre height `z_c` resampled the same way, one table per release class (the centre law
`z_c = max(H_R, d + E|N(c - d, (k sigma_z)^2)|)` takes its own `c`, so an above-roof
release at height `z_s` has its own table). A puff's age maps to its distance by the exact
travel time of the grid's piecewise-linear speed `u(x)` (`PuffTables.at_age`).

The puffs themselves (design Sect. 4.3-4.6): a `PuffField` holds the sources' geometry
(streets as `plume.source_points`, junctions as `plume.junction_sources`, explicit
releases as points), a `PuffState` the puffs in fixed `(B, K)` slots, and `emit`,
`advance`, `roof_concentration` and `concentration_at` are pure functions of them. A puff
of age `a` is a slug along the wind spanning `[x(a - dt/2), x(a + dt/2)]` from its source,
`dt` the emission interval (the street step, or `PuffOptions.puff_dt` when the closure
accumulates several steps into one puff, its age then counted from the interval's middle),
so the puffs of a continuous source tile the axis exactly for any interval `dt`. A receptor
reads the table at its own distance from the source; with `along_wind="none"` the slug is
a segment of the steady plume (density `m / (dt u(x))`, not a uniform top-hat), so the
steady limit of a continuous source is the steady kernel itself for any `dt`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from noodl.apps.street_aq.above_roof import roof_fluxes, roof_wiring
from noodl.apps.street_aq.canyon import KAPPA
from noodl.apps.street_aq.closures import SIGMA_V_MIN, SIGMA_W_MIN
from noodl.apps.street_aq.network import StreetNetwork
from noodl.apps.street_aq.plume import (
    _DTYPE,
    _SQRT_2PI,
    DEFAULT_TEMPERATURE_K,
    DOWNWIND_CUTOFF_M,
    GAUSS_CUTOFF_SIGMAS,
    PLUME_CENTRE_C_M,
    PLUME_CENTRE_K,
    TABLE_STEP_M,
    _f64,
    _grid_size,
    _RecomputedTrajectory,
    _step_bound,
    _table_inputs,
    _trajectory,
    crosswind_profile,
    junction_sources,
    rows_per_chunk,
    source_points,
    street_midpoints,
    table_chunks,
    vertical_profile,
)

_PUFF_COLS = ("u", "sigma_y", "sigma_z", "z_c", "p_z")

_CHECKPOINT = True
"""Whether, under autograd, `PuffClosure` checkpoints each step's puffs (emit, advance,
housekeep, evaluate) and `_evaluate` its receptor chunks (design Sect. 4.5). Values and
gradients are the same bit for bit either way; the tests switch it off to show that and
to measure the memory it saves."""


_SERIES_BELOW = 1e-5
"""Below this `|r|` or `|z|` the travel-time ratios use their Taylor series (truncation
error below 3e-16 relative)."""


def _travel_times(u: Tensor, step: float) -> Tensor:
    """`(..., n)`: the travel time (s) from the source to each grid point at the speed
    `u` `(..., n)` (m/s at grid spacing `step`), linear between grid points: cell `k` takes
    `log(u_{k+1} / u_k) / g_k` with `g_k = (u_{k+1} - u_k) / step`, i.e. `step / u_k` times
    `log1p(r) / r` with `r = (u_{k+1} - u_k) / u_k`, by its series `1 - r/2 + r^2/3` for
    `|r| < _SERIES_BELOW` (value and derivative right on a flat cell)."""
    r = (u[..., 1:] - u[..., :-1]) / u[..., :-1]
    small = r.detach().abs() < _SERIES_BELOW
    r_safe = torch.where(small, torch.ones_like(r), r)
    ratio = torch.where(small, 1.0 - r / 2.0 + r * r / 3.0, torch.log1p(r_safe) / r_safe)
    cell = step * ratio / u[..., :-1]
    return torch.cat([torch.zeros_like(u[..., :1]), torch.cumsum(cell, -1)], dim=-1)


@dataclass(frozen=True)
class PuffTables:
    """The plume trajectory on the steady kernel's distance grid, read by age
    (`puff_tables`).

    `u` (m/s), `sigma_y`, `sigma_z` (m), `z_c` (m, the plume centre height) and `p_z`
    (1/m, the vertical profile at `H_R`) are `(n_tab, n_b, n_grid)` at distances `0, step,
    2 step, ...` (`distance`); `t` (s, same shape, derived from `u`) is the travel time to
    each grid point at the speed `u(x)`, linear between grid points. `h_canopy` and `h_abl`
    (m) are `(n_b,)`; `x_max` is the domain's extent (a puff beyond it has left). The
    detached search key of `at_age` and the running maximum of `sigma_y` (the reach of
    `_evaluate`'s coarse mask) are built once, with `t`."""

    u: Tensor
    sigma_y: Tensor
    sigma_z: Tensor
    z_c: Tensor
    p_z: Tensor
    h_canopy: Tensor
    h_abl: Tensor
    x_max: float
    step: float = TABLE_STEP_M
    t: Tensor = field(init=False)
    _span: float = field(init=False, repr=False, compare=False)
    _shifted: Tensor = field(init=False, repr=False, compare=False)
    _sigma_y_reach: Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        t = _travel_times(self.u, self.step)
        object.__setattr__(self, "t", t)
        n_tab, n_b, n = self.u.shape
        with torch.no_grad():
            # every row's travel times shifted by its row number times a span above every
            # travel time, so one search finds the cell of any (row, age) (`at_age`)
            t_d = t.detach().reshape(-1)
            span = 2.0 ** math.ceil(math.log2(float(t_d.max()) + 1.0) + 1.0)
            shifted = t_d + torch.arange(n_tab * n_b).repeat_interleave(n).to(_DTYPE) * span
            reach = torch.cummax(self.sigma_y.detach(), dim=-1).values.reshape(-1)
        object.__setattr__(self, "_span", span)
        object.__setattr__(self, "_shifted", shifted)
        object.__setattr__(self, "_sigma_y_reach", reach)

    @property
    def distance(self) -> Tensor:
        """`(n_grid,)`: the grid distances, m."""
        return torch.arange(self.u.shape[-1], dtype=_DTYPE) * self.step

    def at_age(self, age: Tensor, table: Tensor, row: Tensor
               ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """`(x, u, sigma_y, sigma_z, z_c, p_z)` at `age` (s) from table `table` and
        meteorology row `row` (long tensors that broadcast with `age`; the result has their
        broadcast shape).

        `x` is the distance travelled in `age` at the speed `u(x)`: within grid cell `k`,
        `dx/dt = u_k + g_k (x - x_k)` with `g_k = (u_{k+1} - u_k) / step`, so `x = x_k +
        (u_k / g_k)(exp(g_k (age - t_k)) - 1)` (`x_k + u_k (age - t_k)` when `g_k = 0`).
        The other columns are linear in `x` between grid points (so `u` is `u(x)`). Beyond
        the grid's end the puff moves on at the last speed and the other columns hold their
        last values; a negative age is the source's."""
        _, n_b, n = self.u.shape
        age, base = torch.broadcast_tensors(_f64(age), (table * n_b + row) * n)
        a = torch.clamp(age, min=0.0)
        t = self.t.reshape(-1)
        with torch.no_grad():
            # cell k: the last grid point with t_k <= a (n - 1 beyond the end), found by
            # one search over the shifted rows (`__post_init__`), then corrected by one
            # cell for the shift's rounding
            t_d, a_d, span = t.detach(), a.detach(), self._span
            r = base // n
            key = torch.clamp(a_d, max=span / 2.0) + r.to(_DTYPE) * span
            k = torch.searchsorted(self._shifted, key.reshape(-1), right=True).reshape(
                key.shape)
            k = torch.clamp(k - 1 - r * n, 0, n - 1)
            k = torch.where((k > 0) & (a_d < t_d[base + k]), k - 1, k)
            up = torch.clamp(k + 1, max=n - 1)
            k = torch.where((k < n - 1) & (a_d >= t_d[base + up]), up, k)
        lo = base + k
        hi = base + torch.clamp(k + 1, max=n - 1)
        u_flat = self.u.reshape(-1)
        u_k = u_flat[lo]
        s = a - t[lo]
        z = (u_flat[hi] - u_k) / self.step * s
        # expm1(z) / z, by its series near z = 0 (value and derivative right there)
        small = z.detach().abs() < _SERIES_BELOW
        z_safe = torch.where(small, torch.ones_like(z), z)
        phi = torch.where(small, 1.0 + z / 2.0 + z * z / 6.0, torch.expm1(z_safe) / z_safe)
        x_k = k.to(_DTYPE) * self.step
        x = x_k + u_k * s * phi
        frac = (x - x_k) / self.step
        out = [x]
        for col in (self.u, self.sigma_y, self.sigma_z, self.z_c, self.p_z):
            v = col.reshape(-1)
            out.append(v[lo] + (v[hi] - v[lo]) * frac)
        return out[0], out[1], out[2], out[3], out[4], out[5]

    def at_distance(self, x: Tensor, table: Tensor, row: Tensor
                    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """`(u, sigma_y, sigma_z, z_c, p_z)` at distance `x` (m) from table `table` and
        meteorology row `row` (broadcasting as in `at_age`), linear between grid points;
        a distance outside the grid reads its end values."""
        n_tab, n_b, n = self.u.shape
        x, base = torch.broadcast_tensors(_f64(x), (table * n_b + row) * n)
        pos = torch.clamp(x / self.step, 0.0, float(n - 1))
        i0 = torch.clamp(torch.floor(pos.detach()).long(), 0, n - 2)
        frac = pos - i0
        lo = base + i0
        out = []
        for col in (self.u, self.sigma_y, self.sigma_z, self.z_c, self.p_z):
            v = col.reshape(-1)
            out.append(v[lo] + (v[lo + 1] - v[lo]) * frac)
        return out[0], out[1], out[2], out[3], out[4]


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
    """SIRANE's plume trajectory on the steady kernel's distance grid (module docstring),
    one table per entry of `centres`.

    `centres` (m) is `centre_c` of the centre law, `(n_tab,)` or `(n_tab, n_b)`; the
    meteorology (`u_star`, `h_abl`, `lmo`, `sigma_theta`, `theta_star`, `temperature`, and
    the empirical inputs, all as in `plume_table`) is scalar or `(n_b,)` and broadcasts
    to `n_b` hours. The grid is `plume_table`'s, out to the first grid point at or beyond
    `x_max`; for the same centre its `sigma_y`, `p_z` and `u` are `plume_table`'s bit for
    bit, and `sigma_z`, `z_c` are the scan's nodes resampled the same way (`_trajectory`).
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
    n_grid = _grid_size(x_max)
    steps = _step_bound("puff_tables", u0, (n_grid - 1) * TABLE_STEP_M)
    keys = tuple(p)
    grad = torch.is_grad_enabled() and any(v.requires_grad for v in p.values())
    out = []
    for a_, b_ in table_chunks(steps.tolist()):
        values = [p[k][a_:b_] for k in keys]
        if grad:
            out.append(_RecomputedTrajectory.apply("puff_tables", _PUFF_COLS, keys, n_grid,
                                                   kappa, *values))
        else:
            out.append(_trajectory(dict(zip(keys, values, strict=True)), n_grid, kappa,
                                   _PUFF_COLS))
    cols = [torch.cat([o[i] for o in out]).reshape(n_tab, n_b, n_grid)
            for i in range(len(_PUFF_COLS))]
    return PuffTables(*cols, h_canopy=p["h_r"].reshape(batch)[0],
                      h_abl=p["h_abl"].reshape(batch)[0], x_max=float(x_max))


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
    a uniform slug convolved with a Gaussian of `sigma_x = sigma_y(x(a))`, `"none"` the slug
    alone as a segment of the steady plume (design Sect. 4.3; `_evaluate`). `cutoff_sigma`
    is the cut-off in standard deviations beyond the flat top (crosswind) and beyond the
    slug ends (along the wind)."""

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
    n_b = tables.u.shape[1]
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
    """`(B, K)`, detached: a cheap estimate of each live puff's peak roof-level
    contribution, `|m| max_i w_i min(P_z, clamp_i) / (ell (2 y_c,i + sqrt(2 pi)
    sigma_y))` (design Sect. 4.8), with the profiles at the puff's distance and `ell` the
    speed there (the step `dt` is common to all puffs and cancels). It is not a bound:
    receptors read the table at their own distance, which can give up to about twice
    this; it only ranks puffs against a relative threshold (`housekeep`, `emit`'s
    eviction). Zero for a free slot."""
    n_batch, capacity = state.mass.shape
    width, clamp = _field_rows(field, n_batch, "housekeep")
    width, clamp = width.detach(), clamp.detach()
    row = _table_rows(tables, n_batch, "housekeep")[:, None]
    with torch.no_grad():
        _, u, s_y, _, _, p_z = tables.at_age(state.age, state.table, row)
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
            torch.clamp(2.0 * torch.clamp((w - _SQRT_2PI * sy) / 2.0, min=0.0) + _SQRT_2PI * sy,
                        min=1e-12))
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
    """Retire puffs (design Sect. 4.8): those whose distance `x(age)` passed
    `tables.x_max` go to `mass_out`; of the others, those
    whose peak estimate (`_peak_bound`) is below `drop_below` times the largest among the
    instance's live puffs go to `mass_dropped`. An instance with no live puff is left as
    it is. Decisions are made on detached values; the masses keep their graph."""
    if not 0.0 <= drop_below < 1.0:
        raise ValueError(f"housekeep: drop_below must be in [0, 1), got {drop_below!r}")
    n_batch = state.mass.shape[0]
    row = _table_rows(tables, n_batch, "housekeep")[:, None]
    with torch.no_grad():
        x = tables.at_age(state.age, state.table, row)[0]
        out = state.live & (x > tables.x_max)
    state = _retire(state, out, "mass_out")
    peak = _peak_bound(field, state, tables)
    with torch.no_grad():
        faint = state.live & (peak < drop_below * peak.max(1, keepdim=True).values)
    return _retire(state, faint, "mass_dropped")


def emit(field: PuffField, state: PuffState, masses: Tensor, table_index: Tensor,
         release_xy: Tensor | None = None, *, tables: PuffTables | None = None,
         age: float = 0.0, theta_w: Tensor | float | None = None
         ) -> tuple[PuffState, Tensor]:
    """One new puff per source with non-zero mass in `masses` `(B, S)` (kg, signed),
    written into the first free slots of its instance in source order: age `age` (s,
    default 0), table `table_index[s]` (`(S,)`), displacement zero, except a release's,
    which is its position in `release_xy` `(B, R, 2)` (m) relative to its `source_xy0`. A
    positive `age` (a puff accumulated over an emission interval, `PuffOptions.puff_dt`)
    adds the distance `x(age)` travelled along the wind `theta_w`, which it then needs with
    `tables`. `mass_emitted` accumulates the masses. An instance without enough free slots
    evicts its faintest live puffs into `mass_dropped` (by the peak estimate of `housekeep`
    when `tables` is given, else by `|m|`); the state and the number evicted `(B,)` are
    returned. More new puffs than the capacity is refused by name."""
    n_batch, capacity = state.mass.shape
    if not age >= 0.0:
        raise ValueError(f"emit: age must be >= 0, got {age!r}")
    if age > 0.0 and (tables is None or theta_w is None):
        raise ValueError("emit: a positive age needs tables= and theta_w= (the puffs start "
                         "x(age) downwind)")
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
    if age > 0.0:
        cos, sin = _wind(theta_w, n_batch, "emit")
        row = _table_rows(tables, n_batch, "emit")[:, None]
        x0 = tables.at_age(torch.full((n_batch, n_src), age, dtype=_DTYPE), table_index,
                           row)[0]
        disp0 = disp0 + x0[..., None] * torch.stack([cos, sin], dim=-1)[:, None, :]
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
        age=state.age.index_put(at, zero + age),
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
    `x(a + dt) - x(a)`: exact for any `dt`, since `x` is the exact travel distance at the
    table's speed, so under constant wind the displacement is `x(a)`. A wind that turns
    between steps turns the puff's further path (design Sect. 4.4)."""
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
    advanced) or whose slug has zero length contributes nothing.

    A receptor at along-wind offset `s` from the puff reads the table at its OWN distance
    from the source, `x(a) + s` (design Sect. 4.3, amended): `sigma_y` (crosswind profile
    and cut-off), and at roof level `min(p_z, clamp)`, at a receptor height
    `vertical_profile` of `sigma_z` and `z_c`. Under `"none"` the slug is a segment of the
    steady plume: it spans the distances travelled in the ages `[a - dt/2, a + dt/2]`
    (clamped at 0), a time `tau`, and its along-wind density is `m / (tau u(x(a) + s))`,
    which integrates to `m` over the slug because `int dx / u = tau`; the steady limit of
    a continuous source is then the steady kernel itself. Under `"sigma_y"` the read is
    at `x(a) + s` clamped to the slug's ends, and the along-wind factor is the slug
    convolved with a Gaussian of `sigma_x = sigma_y(x(a))` (density `m / ell`)."""
    if not dt > 0:
        raise ValueError(f"{where}: dt must be positive, got {dt!r}")
    n_batch, capacity = state.mass.shape
    cos, sin = _wind(theta_w, n_batch, where)
    row = _table_rows(tables, n_batch, where)
    rk = row[:, None]
    age = state.age
    x_a, _, s_y, _, _, _ = tables.at_age(age, state.table, rk)
    a_lo = torch.clamp(age - dt / 2.0, min=0.0)
    e_lo = tables.at_age(a_lo, state.table, rk)[0] - x_a
    e_hi = tables.at_age(age + dt / 2.0, state.table, rk)[0] - x_a
    tau = age + dt / 2.0 - a_lo
    slug = field.along_wind == "sigma_y"
    cut = field.cutoff_sigma
    width, clamp = _field_rows(field, n_batch, where)
    h_c, h_abl = tables.h_canopy[row], tables.h_abl[row]
    src = state.source
    with torch.no_grad():
        active = state.live & (state.mass != 0) & (age > 0) & (e_hi > e_lo)
        # the receptors read sigma_y anywhere up to the slug's far end: bound it by the
        # running maximum of the table's sigma_y there
        n_b, n_grid = tables.u.shape[1:]
        far = torch.clamp(torch.ceil((x_a + e_hi).detach() / tables.step).long(), 0,
                          n_grid - 1)
        s_y_far = tables._sigma_y_reach[(state.table * n_b + rk) * n_grid + far]
        reach = (torch.maximum(-e_lo, e_hi) + (cut * s_y if slug else 0.0)
                 + field._half_width.expand(n_batch, -1).gather(1, src) + cut * s_y_far)
        reach = torch.where(active, reach, torch.zeros_like(reach))
        box_lo = field._box_lo[src] + state.disp - reach[..., None]
        box_hi = field._box_hi[src] + state.disp + reach[..., None]
    rec_xy = _f64(rec_xy)
    n_rec = rec_xy.shape[0]
    rec_d = rec_xy.detach()
    # every tensor that reaches the values below
    grad = torch.is_grad_enabled() and any(
        v is not None and v.requires_grad
        for v in (state.mass, state.age, state.disp, tables.u, tables.t, tables.sigma_y,
                  tables.sigma_z, tables.z_c, tables.p_z, tables.h_canopy, tables.h_abl,
                  field.xy, field.weight, field.width, field.clamp, rec_xy, rec_z, cos, sin))

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
        sx, lo, hi, w = s_y[b, k], e_lo[b, k], e_hi[b, k], width[b, p]
        with torch.no_grad():
            if slug:
                keep = (x >= lo - cut * sx) & (x <= hi + cut * sx)
            else:
                keep = (x >= lo) & (x < hi)
        b, k, i, p, x, y, sx, lo, hi, w = (
            v[keep] for v in (b, k, i, p, x, y, sx, lo, hi, w))
        # the table at the receptor's own distance (clamped to the slug under "sigma_y")
        x_r = x_a[b, k] + (torch.minimum(torch.maximum(x, lo), hi) if slug else x)
        u_r, sy, s_z, z_c, p_z_r = tables.at_distance(x_r, state.table[b, k], row[b])
        with torch.no_grad():
            y_c = torch.clamp((w - _SQRT_2PI * sy) / 2.0, min=0.0)
            keep = y.abs() - y_c <= cut * sy
        b, k, i, p, x, y, sx, lo, hi, w, u_r, sy, s_z, z_c, p_z_r = (
            v[keep] for v in (b, k, i, p, x, y, sx, lo, hi, w, u_r, sy, s_z, z_c, p_z_r))
        if slug:
            root2 = math.sqrt(2.0) * sx
            g_x = (torch.erf((x - lo) / root2) - torch.erf((x - hi) / root2)) / (
                2.0 * (hi - lo))
        else:
            g_x = 1.0 / (tau[b, k] * u_r)
        if rec_z is None:
            p_z = torch.minimum(p_z_r, clamp[b, p])
        else:
            p_z = vertical_profile(rec_z[a_:b_][i], z_c, s_z, h_c[b], h_abl[b])
        c = state.mass[b, k] * field.weight[p] * g_x * crosswind_profile(y, sy, w) * p_z
        return torch.zeros(n_batch * r, dtype=_DTYPE).index_add(0, b * r + i, c).reshape(
            n_batch, r)

    def evaluate(a_: int, b_: int) -> Tensor:
        # under autograd each chunk is checkpointed (design Sect. 4.5): the graph keeps
        # the puff state, the backward recomputes the pair products
        if grad and _CHECKPOINT:
            return checkpoint(block, a_, b_, use_reentrant=False)
        return block(a_, b_)

    rows = rows_per_chunk(capacity * int(field._count.max()), n_batch, backward=grad)
    if n_rec == 0:
        return torch.zeros(n_batch, 0, dtype=_DTYPE)
    return torch.cat([evaluate(a_, min(n_rec, a_ + rows)) for a_ in range(0, n_rec, rows)],
                     dim=1)


def roof_concentration(field: PuffField, state: PuffState, tables: PuffTables,
                       receptors_xy: Tensor, theta_w: Tensor | float, dt: float, *,
                       receptor_owner: Tensor | None = None) -> Tensor:
    """`(B, R)`: the puffs' concentration (kg/m3) at roof level `H_R` above
    `receptors_xy` `(R, 2)`, design Sect. 4.3: per sub-source `m w_i g_x f_y min(P_z,
    clamp_i)`, with `g_x` the slug of the puff's age `a`, `[x(a - dt/2), x(a + dt/2)]` along
    the wind `theta_w` from its source (`dt`, s, the emission interval: the step, or
    `PuffOptions.puff_dt`; consecutive puffs of a continuous source tile the axis),
    and `f_y` and the table's `p_z` read at the receptor's own distance from the source
    (`_evaluate`). A puff whose source
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


def pulse_mass(t: float, dt: float, start: Tensor, duration: Tensor, rate: Tensor) -> Tensor:
    """Mass (kg) of a top-hat release inside the step `[t, t + dt]`.

    The release runs at `rate` (kg/s) over `[start, start + duration]`; the mass is `rate`
    times the overlap of that window with the step, `rate * clamp(min(t + dt, start +
    duration) - max(t, start), min=0)`, broadcasting. It is piecewise linear in `start`,
    `duration` and `rate`, so autograd gives the exact derivative (the overlap's slope)
    everywhere but at the kinks, and a step the window misses gets a zero value and a
    finite zero gradient.
    """
    start = torch.as_tensor(start, dtype=_DTYPE)
    duration = torch.as_tensor(duration, dtype=_DTYPE)
    rate = torch.as_tensor(rate, dtype=_DTYPE)
    t_in = torch.clamp(start, min=float(t))
    t_out = torch.clamp(start + duration, max=float(t + dt))
    return rate * torch.clamp(t_out - t_in, min=0.0)


class PulseSource:
    """An explicit pulse release into the street canopy (design Sect. 4.2).

    An integrating closure, called `closure(state, drivers, ctx)`. Drivers: `"pulse.weights"`
    `(B, n_s)` (the share of the release in each street of `street_nodes`, usually one-hot),
    and `"pulse.rate"` (kg/s), `"pulse.start"` (s), `"pulse.duration"` (s), each `(B,)`. Over
    the step `[ctx.t, ctx.t + ctx.dt]` it ADDS `pulse_mass / dt * weights` (kg/s) to
    `"<layer>.sources"` at the street nodes, in full node order, to whatever the drivers
    already carry there (a two-way building link writes it too), creating zeros when it is
    absent. It does not apply `Street.emission_scale`: the transport layer built by
    `build_model` scales `"<layer>.sources"` inside the model, so scaling here would apply
    it twice. A query (`ctx is None`) adds nothing; a step without `Model.step(t=...)` is
    refused by name.
    """

    integrates = True

    def __init__(self, net, street_nodes: Sequence[str], *, layer_name: str = "street"):
        self.layer_name = layer_name
        self.street_nodes = tuple(street_nodes)
        if not self.street_nodes:
            raise ValueError("PulseSource: street_nodes is empty")
        self.n_nodes = len(net.nodes)
        self.index = torch.tensor(
            [net.node_index(name) for name in self.street_nodes], dtype=torch.long
        )
        self.inputs = ("pulse.weights", "pulse.rate", "pulse.start", "pulse.duration")
        self.outputs = (f"{layer_name}.sources",)

    def __repr__(self) -> str:
        return f"PulseSource(layer_name={self.layer_name!r}, n_streets={len(self.street_nodes)})"

    def __call__(self, state, drivers, ctx=None):
        key = f"{self.layer_name}.sources"
        if ctx is None:
            return {}
        if ctx.dt is None:
            raise ValueError("PulseSource: steady() has no step to release over")
        if ctx.t is None:
            raise ValueError("PulseSource: the step time is unknown; call Model.step(t=...)")
        for name in self.inputs:
            if name not in drivers:
                raise KeyError(f"PulseSource: driver {name!r} is missing")
        weights = _f64(drivers["pulse.weights"])
        if weights.shape[-1] != len(self.street_nodes):
            raise ValueError(
                f"PulseSource: pulse.weights has {weights.shape[-1]} columns, one per street "
                f"expected ({len(self.street_nodes)})"
            )
        mass = pulse_mass(
            ctx.t, ctx.dt, _f64(drivers["pulse.start"]), _f64(drivers["pulse.duration"]),
            _f64(drivers["pulse.rate"]),
        )
        add = weights * (mass / ctx.dt).unsqueeze(-1)          # (B, n_s), kg/s
        full = torch.zeros((*add.shape[:-1], self.n_nodes), dtype=_DTYPE)
        full = full.index_add(-1, self.index, add)
        base = drivers.get(key)
        if base is None:
            return {key: full}
        base = _f64(base)
        if base.dim() >= 2 and base.shape[-1] == 1 and base.shape[-2] == self.n_nodes:
            full = full.unsqueeze(-1)
        elif base.shape[-1] != self.n_nodes:
            raise ValueError(
                f"PulseSource: {key} has shape {tuple(base.shape)}; expected (..., "
                f"{self.n_nodes}) or (..., {self.n_nodes}, 1)"
            )
        return {key: base + full}


JUNCTION_SOURCES = ("upward", "net")
"""The junction source options of `PuffClosure` (as `street_steady_with_plume`'s)."""

_RELEASE_KEYS = ("puff.release_xy", "puff.release_z", "puff.release_rate",
                 "puff.release_start", "puff.release_duration")
_CARRIED = ("puff.q", "puff.c_ext", "puff.time", "puff.pending", "puff.pending_time")
_PUFF_DT_RTOL = 1e-9
"""Relative tolerance of `puff_dt` checks: a street step divides `puff_dt`, and the steps
since the last emission fill it."""


@dataclass(frozen=True)
class PuffOptions:
    """The puff model of `build_model(..., puffs=PuffOptions(...))` (design Sect. 4-5).

    `h_canopy`, `z0`, `d` (m) are the tables' canopy height `H_R`, roughness and
    displacement; `sigma_theta` (rad) the wind-direction spread, `None` for the
    `"sigma_theta"` driver. `junction_source` is one of `JUNCTION_SOURCES`; `along_wind` one
    of `ALONG_WIND`; `capacity` the puff slots per batch instance; `puff_dt` (s) the
    emission interval: each source's mass is accumulated over the street steps of one
    interval and emitted as one puff, whose slug spans the interval (`None`: one puff per
    street step); every street step must divide it. `drop_below` the faint threshold of
    `housekeep`; `cutoff_sigma` the profiles' cut-off; `x_max` (m) the extent
    of the tables (a puff beyond it leaves the domain); `n_releases` the explicit
    above-roof releases. `table_kwargs` go to `puff_tables` (the empirical inputs:
    `theta_star`, `temperature`, `tau_y`, `tau_z`, `centre_k`, the floors); a `"centre_c"`
    entry there is the centre of the canopy sources' table (default
    `plume.PLUME_CENTRE_C_M`). `shear` (wind-shear stretching, design Sect. 4.7) is not
    available yet and refused by name."""

    h_canopy: float = 20.0
    z0: float = 1.0
    d: float = 13.0
    sigma_theta: float | None = None
    junction_source: str = "upward"
    along_wind: str = "sigma_y"
    capacity: int = 4096
    puff_dt: float | None = 30.0
    drop_below: float = 1e-3
    shear: bool = False
    cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS
    x_max: float = DOWNWIND_CUTOFF_M
    n_releases: int = 0
    table_kwargs: Mapping = field(default_factory=dict)


class PuffClosure:
    """The unsteady above-roof model as an integrating closure of the street model
    (design Sect. 4.5): it writes `"<layer>.x_boundary"` = `C_ext` per street, the caller's
    `"<layer>.x_boundary"` driver (`C_bg`, read afresh every step) plus the puffs at each
    street midpoint at roof level (a street's own puffs excluded, as `street_kernel` does).

    Each step, from the state at its start: (1) emit the puffs of the previous step, the
    masses backward Euler moved over it: `roof_fluxes` of the stored flows `"puff.q"` and
    boundary `"puff.c_ext"` with the state's `"<layer>.x"` and this step's `C_bg`, times the
    stored step length (street roof fluxes; junction vertical fluxes `F_up`, or `F_up -
    F_down` under `junction_source="net"`; under `"upward"` `F_down dt` is added to
    `mass_drawn_down`), and each explicit release's `pulse_mass` over that step; under
    `options.puff_dt` these masses are added to `"puff.pending"` instead and emitted, one
    puff per source, once the steps since the last emission fill `puff_dt`, at the age
    `puff_dt/2 - dt/2` (displaced `x` of it along the current wind) so that after (2) the
    puff's age is counted from the middle of its interval and its slug, of age span
    `puff_dt`, covers the material of the whole interval; (2) `advance` every puff by
    `dt` on the current wind; (3) `housekeep`; (4) evaluate `C_ext`
    and store this step's flows (`"<layer>.q"`, written by the flow closure, which must
    run first), `C_ext` and `"puff.time"` = `(t, dt)` of this step (`t` is 0 when the step
    has none). The puffs live in the state as `PuffState.to_state()` keys.

    Drivers: `"u_star"`, `"theta_w"`, `"h_abl"` (required), `"lmo"` (optional, neutral
    without it) and `"sigma_theta"` when `options.sigma_theta` is None, each one value per
    instance (under `meteo="per_street"` with a trailing street axis, or junction axis for
    `"sigma_theta"`, whose values must all be equal). With `n_releases = R`:
    `"puff.release_xy"` `(B, R, 2)` (m), `"puff.release_z"` (m, the centre of release
    `r`'s table `1 + r`), `"puff.release_rate"` (kg/s), `"puff.release_start"` and
    `"puff.release_duration"` (s), each `(B, R)` (the batch dim optional), and
    `Model.step(t=...)`. The batch is `()` or `(B,)`. Tables and source field are rebuilt
    when one of their drivers is a different tensor, or has a different value, than at the
    last build (a value-only cache would reuse an earlier run's graph).

    A query (`ctx is None`) returns `C_ext` from the state's puffs without advancing.
    `last` holds the latest step's `{"live": (B,) live puffs, "evicted": (B,) puffs evicted
    for capacity, "rebuilt": whether the tables or the field were rebuilt}`.

    The closure reads the flow wiring of the model it is bound to (`bind`, which
    `build_model` calls) and refuses, at every step, a bound model whose coupling is not
    ping-pong, which sub-steps the layer, steps it other than by backward Euler or carries a
    reaction, and a removal on the layer (design Sect. 3: the mass account is written for
    one backward-Euler step without sinks)."""

    integrates = True

    def __init__(self, net: StreetNetwork, options: PuffOptions, *, meteo: str = "uniform",
                 layer_name: str = "street") -> None:
        if options.junction_source not in JUNCTION_SOURCES:
            raise ValueError(f"PuffOptions: junction_source must be one of "
                             f"{JUNCTION_SOURCES}, got {options.junction_source!r}")
        if options.along_wind not in ALONG_WIND:
            raise ValueError(f"PuffOptions: along_wind must be one of {ALONG_WIND}, got "
                             f"{options.along_wind!r}")
        if options.shear:
            raise NotImplementedError("PuffOptions: shear=True (wind-shear stretching) is not "
                                      "available yet; pass shear=False")
        if int(options.capacity) < 1:
            raise ValueError(f"PuffOptions: capacity must be >= 1, got {options.capacity!r}")
        if not 0.0 <= options.drop_below < 1.0:
            raise ValueError(f"PuffOptions: drop_below must be in [0, 1), got "
                             f"{options.drop_below!r}")
        if options.puff_dt is not None and not 0.0 < options.puff_dt < math.inf:
            raise ValueError(f"PuffOptions: puff_dt must be None or a positive number of "
                             f"seconds, got {options.puff_dt!r}")
        if int(options.n_releases) < 0:
            raise ValueError(f"PuffOptions: n_releases must be >= 0, got "
                             f"{options.n_releases!r}")
        if meteo not in ("uniform", "per_street"):
            raise ValueError(f"PuffClosure: meteo must be 'uniform' or 'per_street', got "
                             f"{meteo!r}")
        kwargs = dict(options.table_kwargs)
        self.centre_c = float(kwargs.pop("centre_c", PLUME_CENTRE_C_M))
        self.table_kwargs = kwargs
        self.net, self.options, self.meteo, self.layer_name = net, options, meteo, layer_name
        self.n_streets = len(net.streets)
        self.n_releases = int(options.n_releases)
        self.receptors = street_midpoints(net)
        self.receptor_owner = torch.arange(self.n_streets)
        self.state_keys = (*(f"puff.{n}" for n in _STATE_FIELDS), *_CARRIED)
        self.model = None
        self.wiring = None
        self.tables: PuffTables | None = None
        self.field: PuffField | None = None
        self._table_key: tuple | None = None
        self._field_key: tuple | None = None
        self.last: dict = {}

    def __repr__(self) -> str:
        return (f"PuffClosure(layer_name={self.layer_name!r}, n_streets={self.n_streets}, "
                f"n_releases={self.n_releases})")

    def bind(self, model) -> None:
        """Read the flow wiring of `model` (`above_roof.roof_wiring`), whose coupling,
        substeps, scheme, reactions and removal every step then checks; `build_model` binds
        the model it builds. A `Model` built by hand around this closure must bind it."""
        wiring = roof_wiring(model, layer_name=self.layer_name)
        if wiring.n_streets != self.n_streets:
            raise ValueError(f"PuffClosure: the model has {wiring.n_streets} streets, the "
                             f"network {self.n_streets}")
        self.model, self.wiring = model, wiring

    def initial_state(self) -> dict[str, Tensor]:
        """The puff keys of a state before the first step: no puffs (one unbatched
        instance, expanded to the drivers' batch on the first step), zero flows and
        boundary, no pending step (`"puff.time"` = `(0, 0)`) and nothing accumulated
        towards the next emission (`"puff.pending"` `(1, S)` kg per source,
        `"puff.pending_time"` `(1,)` s)."""
        self._require_bound()
        n_q = sum(int(self.model.net.edge_index(kind).numel())
                  for kind in self.model.transport[self.layer_name].flow_kinds)
        n_src = self.n_streets + self.wiring.n_junctions + self.n_releases
        return {**PuffState.empty((1,), self.options.capacity).to_state(),
                "puff.q": torch.zeros(n_q, dtype=_DTYPE),
                "puff.c_ext": torch.zeros(self.n_streets, dtype=_DTYPE),
                "puff.time": torch.zeros(2, dtype=_DTYPE),
                "puff.pending": torch.zeros(1, n_src, dtype=_DTYPE),
                "puff.pending_time": torch.zeros(1, dtype=_DTYPE)}

    # ---------------------------------------------------------------- checks
    def _require_bound(self) -> None:
        if self.model is None:
            raise ValueError("PuffClosure: not bound to a model; call closure.bind(model)")

    def _check_model(self, drivers) -> None:
        self._require_bound()
        model, name = self.model, self.layer_name
        if model.coupling != "pingpong":
            raise ValueError(
                f"PuffClosure: the street model has coupling={model.coupling!r}; the puff "
                f"emission is the mass of one backward-Euler step from the step-start state, "
                f"which needs coupling='pingpong' (a two-way link in noodl.couple.union "
                f"re-steps the whole street model and is unaffected)")
        if model.substeps.get(name, 1) != 1:
            raise ValueError(
                f"PuffClosure: transport layer {name!r} takes substeps="
                f"{model.substeps[name]}; the puff emission is the mass of one "
                f"backward-Euler step, so give it no substeps")
        if model.reactions:
            raise ValueError("PuffClosure: the street model carries a reaction; the puff "
                             "mass account has no chemistry, build it with chemistry=None")
        layer = model.transport[name]
        if f"{name}.removal" in drivers or layer.removal is not None:
            raise ValueError(
                f"PuffClosure: transport layer {name!r} has a removal (the driver "
                f"{name + '.removal'!r} or the layer's own); the puff mass account has no "
                f"deposition, drop it")
        if layer.scheme != "implicit":
            raise ValueError(f"PuffClosure: transport layer {name!r} has scheme="
                             f"{layer.scheme!r}; the puff emission needs scheme='implicit'")

    # ---------------------------------------------------------------- drivers
    @staticmethod
    def _instance(drivers, key: str, *, required: bool = True, axis: int = 0):
        """`(raw, value)`: the driver as given and as one value per instance (a trailing
        axis of length `axis`, when `axis`, checked uniform and dropped)."""
        raw = drivers.get(key)
        if raw is None:
            if required:
                hint = (" or pass PuffOptions(sigma_theta=...)" if key == "sigma_theta"
                        else "")
                raise KeyError(f"PuffClosure: the driver {key!r} is missing; the puff "
                               f"tables need it{hint}")
            return None, None
        value = _f64(raw)
        if axis:
            if value.dim() == 0 or value.shape[-1] != axis:
                raise ValueError(f"PuffClosure: under meteo='per_street' {key!r} needs a "
                                 f"trailing axis of length {axis}, got {tuple(value.shape)}")
            with torch.no_grad():
                uniform = bool((value == value[..., :1]).all())
            if not uniform:
                raise ValueError(
                    f"PuffClosure: {key!r} differs along its street (or junction) axis; the "
                    f"puff tables take one meteorology per instance, so pass equal values "
                    f"or build with meteo='uniform'")
            value = value[..., 0]
        return raw, value

    def _drivers(self, drivers, need_releases: bool) -> tuple[dict, dict]:
        per_street = self.meteo == "per_street"
        n_s = self.n_streets if per_street else 0
        n_j = self.wiring.n_junctions if per_street else 0
        met = {key: self._instance(drivers, key, axis=n_s)
               for key in ("u_star", "theta_w", "h_abl")}
        met["lmo"] = self._instance(drivers, "lmo", required=False, axis=n_s)
        if self.options.sigma_theta is None:
            met["sigma_theta"] = self._instance(drivers, "sigma_theta", axis=n_j)
        keys = _RELEASE_KEYS if need_releases else ("puff.release_z",)
        rel = {}
        for key in keys if self.n_releases else ():
            if key not in drivers:
                raise KeyError(f"PuffClosure: the driver {key!r} is missing; the model has "
                               f"{self.n_releases} above-roof release(s)")
            rel[key] = _f64(drivers[key])
            trailing = ((self.n_releases, 2) if key == "puff.release_xy"
                        else (self.n_releases,))
            if tuple(rel[key].shape[rel[key].dim() - len(trailing):]) != trailing:
                raise ValueError(f"PuffClosure: {key!r} must end in {trailing}, got "
                                 f"{tuple(rel[key].shape)}")
        return met, rel

    # ---------------------------------------------------------------- caches
    @staticmethod
    def _same(key: tuple | None, raws: tuple) -> bool:
        """Whether every input is the same object, with the same value, as at the build."""
        if key is None:
            return False
        for (old, value), raw in zip(key, raws, strict=True):
            if old is not raw:
                return False
            if isinstance(raw, Tensor) and not torch.equal(value, raw.detach()):
                return False
        return True

    @staticmethod
    def _key(raws: tuple) -> tuple:
        return tuple((raw, raw.detach().clone() if isinstance(raw, Tensor) else raw)
                     for raw in raws)

    def _build(self, met: dict, rel: dict, *, new_run: bool) -> bool:
        """Rebuild the tables and the field when their inputs changed; whether any was.

        The cache holds the autograd graph of the run that built it, so it is also dropped
        at the start of every run (`new_run`: a step from a state with no pending step) and
        keyed on the grad mode: an earlier run's tables would carry a freed graph, or none
        (built under `torch.no_grad`), and give a wrong or failing gradient."""
        if new_run:
            self._table_key = self._field_key = None
        grad = torch.is_grad_enabled()
        o = self.options
        z_raw = rel.get("puff.release_z")
        sigma = met.get("sigma_theta", (None, o.sigma_theta))
        raws = (met["u_star"][0], met["h_abl"][0], met["lmo"][0], sigma[0], z_raw, grad)
        rebuilt = False
        if not self._same(self._table_key, raws):
            centres = torch.tensor([self.centre_c], dtype=_DTYPE)
            if self.n_releases:
                z = rel["puff.release_z"]
                z = z.reshape(-1, self.n_releases) if z.dim() > 1 else z[None]
                centres = torch.cat([centres[:, None].expand(1, z.shape[0]), z.T])
            lmo = met["lmo"][1] if met["lmo"][1] is not None else math.inf
            self.tables = puff_tables(
                centres=centres, u_star=met["u_star"][1], h_abl=met["h_abl"][1], lmo=lmo,
                h_canopy=o.h_canopy, z0=o.z0, d=o.d, sigma_theta=sigma[1], x_max=o.x_max,
                **self.table_kwargs)
            self._table_key = self._key(raws)
            rebuilt = True
        theta = (met["theta_w"][0], grad)
        if not self._same(self._field_key, theta):
            self.field = PuffField.for_network(
                self.net, theta_w=met["theta_w"][1], n_releases=self.n_releases,
                along_wind=o.along_wind, cutoff_sigma=o.cutoff_sigma)
            self._field_key = self._key(theta)
            rebuilt = True
        return rebuilt

    # ---------------------------------------------------------------- the step
    @staticmethod
    def _batch(shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
        try:
            batch = tuple(torch.broadcast_shapes(*shapes))
        except RuntimeError as exc:
            raise ValueError(f"PuffClosure: the batch shapes of the state and drivers "
                             f"{shapes} do not broadcast") from exc
        if len(batch) > 1:
            raise ValueError(f"PuffClosure: the batch must be () or (B,), got {batch}; "
                             f"flatten a multi-dimensional batch to (B,)")
        return batch

    @staticmethod
    def _puffs(state, n_batch: int) -> PuffState:
        """The state's puffs with `n_batch` instances: an unbatched seed with nothing live
        is expanded (design ruling: one leading batch dim)."""
        puff = PuffState.from_state(state)
        have = puff.mass.shape[0]
        if have == n_batch:
            return puff
        if have != 1:
            raise ValueError(f"PuffClosure: the state's puffs have batch {have}, the drivers "
                             f"{n_batch}")
        if bool(puff.live.any()):
            raise ValueError(f"PuffClosure: the state's puffs are unbatched and some are "
                             f"live, but the drivers have batch {n_batch}; start a batched "
                             f"run from initial_state")
        return PuffState(**{n: getattr(puff, n).expand(n_batch, *getattr(puff, n).shape[1:])
                            .clone() for n in _STATE_FIELDS})

    def __call__(self, state, drivers, ctx=None):
        name = self.layer_name
        xb_key, q_key = f"{name}.x_boundary", f"{name}.q"
        if ctx is None:
            self._require_bound()
        else:
            self._check_model(drivers)
        if xb_key not in drivers:
            raise KeyError(f"PuffClosure: the driver {xb_key!r} (the background C_bg, one "
                           f"per street) is missing")
        c_bg = _f64(drivers[xb_key])
        if c_bg.dim() == 0 or c_bg.shape[-1] != self.n_streets:
            raise ValueError(f"PuffClosure: {xb_key!r} must be (..., {self.n_streets}), one "
                             f"background per street, got {tuple(c_bg.shape)}")
        met, rel = self._drivers(drivers, need_releases=ctx is not None)
        x = _f64(state[f"{name}.x"])
        shapes = [tuple(c_bg.shape[:-1]), tuple(x.shape[:-1])]
        shapes += [tuple(v.shape) for _, v in met.values() if v is not None]
        shapes += [tuple(v.shape[:v.dim() - (2 if k == "puff.release_xy" else 1)])
                   for k, v in rel.items()]
        batch = self._batch(shapes)
        stored = state["puff.mass"].shape[0]
        if not batch and stored > 1:
            batch = (stored,)
        n_batch = batch[0] if batch else 1
        puff = self._puffs(state, n_batch)
        rebuilt = self._build(met, rel, new_run=ctx is not None
                              and float(state["puff.time"][1]) == 0.0)
        theta = met["theta_w"][1]

        def per_instance(t: Tensor, tail: tuple[int, ...]) -> Tensor:
            return t.expand((*batch, *tail)).reshape(n_batch, *tail)

        def c_ext_of(puff: PuffState, dt: float) -> Tensor:
            c = roof_concentration(self.field, puff, self.tables, self.receptors, theta, dt,
                                   receptor_owner=self.receptor_owner)
            return c_bg + (c if batch else c[0])

        puff_dt = self.options.puff_dt
        if ctx is None:
            dt_prev = float(state["puff.time"][1])
            if dt_prev <= 0 or not bool(puff.live.any()):
                return {xb_key: c_bg.expand((*batch, self.n_streets))}
            return {xb_key: c_ext_of(puff, dt_prev if puff_dt is None else puff_dt)}
        if self.n_releases and ctx.t is None:
            raise ValueError("PuffClosure: the releases need the step time; call "
                             "Model.step(t=...)")
        if q_key not in drivers:
            raise KeyError(f"PuffClosure: the driver {q_key!r} is missing; the flow closure "
                           f"must run before the puff closure")
        dt = float(ctx.dt)
        if puff_dt is not None:
            n_steps = puff_dt / dt
            if abs(n_steps - round(n_steps)) > _PUFF_DT_RTOL * n_steps or round(n_steps) < 1:
                raise ValueError(f"PuffClosure: the street step dt = {dt!r} s does not divide "
                                 f"PuffOptions.puff_dt = {puff_dt!r} s; pick a step that does "
                                 f"or puff_dt=None (one puff per step)")
        t_prev, dt_prev = (float(v) for v in state["puff.time"])
        # (1) the previous step's fluxes, the masses backward Euler moved over it
        f_s, up, down = roof_fluxes(self.wiring, _f64(state["puff.q"]), x.unsqueeze(-1),
                                    _f64(state["puff.c_ext"]).unsqueeze(-1),
                                    c_bg.unsqueeze(-1))
        upward = self.options.junction_source == "upward"
        junction = up if upward else up - down
        n_j = self.wiring.n_junctions
        parts = [per_instance(f_s[..., 0] * dt_prev, (self.n_streets,)),
                 per_instance(junction[..., 0] * dt_prev, (n_j,))]
        release_xy = None
        if self.n_releases:
            r = self.n_releases
            parts.append(per_instance(
                pulse_mass(t_prev, dt_prev, rel["puff.release_start"],
                           rel["puff.release_duration"], rel["puff.release_rate"]), (r,)))
            release_xy = per_instance(rel["puff.release_xy"], (r, 2))
        table_index = torch.cat([torch.zeros(self.n_streets + n_j, dtype=torch.long),
                                 1 + torch.arange(self.n_releases)])
        masses = torch.cat(parts, dim=1)
        drawn = per_instance(down[..., 0].sum(-1) * dt_prev, ()) if upward else None
        # the emission interval: under `puff_dt` the step's masses join those accumulated
        # since the last emission, emitted as one puff per source once the steps fill
        # `puff_dt`, at the age of the interval's midpoint less this step's advance (its
        # slug then spans the interval, `[x(a - puff_dt/2), x(a + puff_dt/2)]`)
        pending = _f64(state["puff.pending"])
        if pending.shape[0] != n_batch:
            pending = pending.expand(n_batch, -1)
        pending_time = _f64(state["puff.pending_time"])
        age0, span = 0.0, dt
        if puff_dt is None:
            emitted = masses
        else:
            filled = float(pending_time[0]) + dt_prev
            if filled > puff_dt * (1.0 + _PUFF_DT_RTOL):
                raise ValueError(f"PuffClosure: the street steps since the last emission "
                                 f"span {filled!r} s, past PuffOptions.puff_dt = {puff_dt!r} "
                                 f"s; keep a step that divides puff_dt within an interval")
            pending = pending + masses
            age0, span = puff_dt / 2.0 - dt / 2.0, puff_dt
            if filled >= puff_dt * (1.0 - _PUFF_DT_RTOL):
                emitted, pending = pending, torch.zeros_like(pending)
                pending_time = torch.zeros_like(pending_time)
            else:
                emitted, pending_time = None, pending_time + dt_prev
        # the step's own field and tables: a checkpoint recomputes `puff_step` in the
        # backward, after later steps may have rebuilt them
        field, tables, drop_below = self.field, self.tables, self.options.drop_below

        def puff_step(puff: PuffState) -> tuple[PuffState, Tensor, Tensor]:
            if emitted is None:
                evicted = torch.zeros(n_batch, dtype=torch.long)
            else:
                puff, evicted = emit(field, puff, emitted, table_index, release_xy,
                                     tables=tables, age=age0, theta_w=theta)
            if drawn is not None:
                puff = replace(puff, mass_drawn_down=puff.mass_drawn_down + drawn)
            # (2) advance, (3) housekeep, (4) evaluate
            puff = advance(puff, tables, dt, theta)
            puff = housekeep(field, puff, tables, drop_below=drop_below)
            c = roof_concentration(field, puff, tables, self.receptors, theta, span,
                                   receptor_owner=self.receptor_owner)
            return puff, c_bg + (c if batch else c[0]), evicted

        # under autograd the step is checkpointed (design Sect. 4.5): the graph keeps the
        # puff state of every step and the backward recomputes the step's puffs and pairs
        # (only when something reaching the step requires grad: a forward-only run outside
        # `torch.no_grad` records nothing and pays no checkpoint)
        reaching = [*(getattr(puff, f.name) for f in fields(PuffState)),
                    *(getattr(tables, f.name) for f in fields(PuffTables)),
                    field.xy, field.weight, field.width, field.clamp,
                    emitted, drawn, release_xy, theta, c_bg]
        recorded = torch.is_grad_enabled() and any(
            isinstance(v, Tensor) and v.requires_grad for v in reaching)
        if recorded and _CHECKPOINT:
            puff, c_ext, evicted = checkpoint(puff_step, puff, use_reentrant=False)
        else:
            puff, c_ext, evicted = puff_step(puff)
        self.last = {"live": puff.live.sum(1), "evicted": evicted, "rebuilt": rebuilt}
        t_now = 0.0 if ctx.t is None else float(ctx.t)
        return {xb_key: c_ext, **puff.to_state(), "puff.q": drivers[q_key],
                "puff.c_ext": c_ext, "puff.time": torch.tensor([t_now, dt], dtype=_DTYPE),
                "puff.pending": pending, "puff.pending_time": pending_time}


def mass_in_streets(model, state, c_bg: Tensor, *, layer_name: str = "street") -> Tensor:
    """`(B,)`: the street network's mass in excess of the background, kg: `sum_s V_s (x_s
    - C_bg,s)` from the state's `"<layer>.x"` `(..., n_streets)` and `c_bg` (the
    background, broadcasting with it), with `V_s` the layer's capacity (the street
    volumes). An unbatched state gives `B = 1`. The streets' term of the budget identity
    (design Sect. 4.8); `pending_roof_mass` is the step not yet emitted."""
    layer = model.transport[layer_name]
    x = _f64(state[f"{layer_name}.x"])
    excess = layer.capacity * (x - _f64(c_bg))
    return excess.reshape(-1, layer.n_i).sum(-1)


def pending_roof_mass(model, state, c_bg: Tensor, *, junction_source: str = "upward",
                      layer_name: str = "street") -> tuple[Tensor, Tensor]:
    """`(emitted, drawn_down)`, each `(B,)`, kg: what the puff closure's next step will add
    to `mass_emitted` from the roof and junction fluxes of the last step, and to
    `mass_drawn_down`, computed as the closure does (`roof_fluxes` of `"puff.q"`,
    `"puff.c_ext"` and the state's `"<layer>.x"` against the background `c_bg`, times the
    last step's length `"puff.time"[1]`). The emission lags one step (design Sect. 4.2), so
    the budget identity after a step counts these, and under `PuffOptions.puff_dt` the mass
    accumulated towards the next emission (`"puff.pending"`, releases included). The
    explicit releases' mass of the last step is not included (it is `pulse_mass` over the
    last step). `drawn_down` is zero under `junction_source="net"`."""
    if junction_source not in JUNCTION_SOURCES:
        raise ValueError(f"pending_roof_mass: junction_source must be one of "
                         f"{JUNCTION_SOURCES}, got {junction_source!r}")
    wiring = roof_wiring(model, layer_name=layer_name)
    dt_prev = float(state["puff.time"][1])
    x = _f64(state[f"{layer_name}.x"])
    f_s, up, down = roof_fluxes(wiring, _f64(state["puff.q"]), x.unsqueeze(-1),
                                _f64(state["puff.c_ext"]).unsqueeze(-1),
                                _f64(c_bg).unsqueeze(-1))
    upward = junction_source == "upward"
    junction = up if upward else up - down
    emitted = (f_s.sum((-2, -1)) + junction.sum((-2, -1))) * dt_prev
    drawn = down.sum((-2, -1)) * dt_prev if upward else torch.zeros_like(emitted)
    emitted = emitted.reshape(-1) + _f64(state["puff.pending"]).sum(-1)
    return emitted, drawn.reshape(-1)


__all__ = ["ALONG_WIND", "JUNCTION_SOURCES", "PuffClosure", "PuffField", "PuffOptions",
           "PuffState", "PuffTables", "PulseSource", "advance", "concentration_at", "emit",
           "housekeep", "mass_in_streets", "pending_roof_mass", "puff_mass", "puff_tables",
           "pulse_mass", "roof_concentration"]

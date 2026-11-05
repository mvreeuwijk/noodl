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
`dt` the emission interval (the street step by default, or `PuffOptions.puff_dt` when the
closure is asked to accumulate several steps into one puff, its age then counted from the
interval's middle),
so the puffs of a continuous source tile the axis exactly for any interval `dt`. A receptor
reads the table at its own distance from the source; with `along_wind="none"` the slug is
a segment of the steady plume (density `m / (dt u(x))`, not a uniform top-hat), so the
steady limit of a continuous source is the steady kernel itself for any `dt`. Under
`shear=True` (`along_wind="sigma_y"` only) the puffs also carry the Sykes-Henn wind-shear
moments `sxz`, `sx2_shear` (`advance`, `_evaluate`; design Sect. 4.7).

"design Sect. N" below refers to the sections of the design note
`docs/design/street_puffs.md`; the user's account is the section "Unsteady above-roof
transport (puffs)" of `docs/applications/street_aq.md`.
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


def _phi_m(zeta: Tensor) -> Tensor:
    """The dimensionless shear `1 - zeta dpsi_m/dzeta` of `plume._psi_m` (Businger):
    `(1 - 16 zeta)^(-1/4)` for `zeta < 0`, `1 + 5 zeta` otherwise (1 when neutral)."""
    unstable = zeta < 0
    safe = torch.where(unstable, zeta, torch.zeros_like(zeta))
    return torch.where(unstable, (1.0 - 16.0 * safe) ** -0.25, 1.0 + 5.0 * zeta)


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
    `_evaluate`'s coarse mask) are built once, with `t`.

    `u_star` (m/s), `lmo`, `z0` and `d` (m), each `(n_b,)`, and `kappa` are each
    meteorology row's wind profile (`plume.wind_speed`), whose shear the stretching reads
    (`shear_rate`); `puff_tables` sets them, a hand-made table may leave them None (it
    then cannot be advanced with `shear=True`)."""

    u: Tensor
    sigma_y: Tensor
    sigma_z: Tensor
    z_c: Tensor
    p_z: Tensor
    h_canopy: Tensor
    h_abl: Tensor
    x_max: float
    step: float = TABLE_STEP_M
    u_star: Tensor | None = None
    lmo: Tensor | None = None
    z0: Tensor | None = None
    d: Tensor | None = None
    kappa: float = KAPPA
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
        x, lo, hi, frac = self._travel(age, table, row)
        out = [x]
        for col in (self.u, self.sigma_y, self.sigma_z, self.z_c, self.p_z):
            v = col.reshape(-1)
            out.append(v[lo] + (v[hi] - v[lo]) * frac)
        return out[0], out[1], out[2], out[3], out[4], out[5]

    def _travel(self, age: Tensor, table: Tensor, row: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """`(x, lo, hi, frac)` of `at_age`: the distance, the flat indices of the grid
        points around it and the fraction of the cell beyond `lo`; a column `v` (flat)
        reads `v[lo] + (v[hi] - v[lo]) * frac`."""
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
        return x, lo, hi, (x - x_k) / self.step

    def at_distance(self, x: Tensor, table: Tensor, row: Tensor
                    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """`(u, sigma_y, sigma_z, z_c, p_z)` at distance `x` (m) from table `table` and
        meteorology row `row` (broadcasting as in `at_age`), linear between grid points;
        a distance outside the grid reads its end values."""
        cell = self._cell(x, self._base(table, row))
        return tuple(self._read(col, *cell)
                     for col in (self.u, self.sigma_y, self.sigma_z, self.z_c, self.p_z))

    def shear_rate(self, z: Tensor, row: Tensor) -> Tensor:
        """`dU/dz` (1/s) of meteorology row `row`'s wind profile `plume.wind_speed` at
        height `z` (m above ground; `row` long, broadcasting with `z`): the derivative of
        `u = (u*/kappa) [ln(z'/z0) - psi_m(z'/L) + psi_m(z0/L)]`, `z' = z - d + z0`, in
        closed form, `u* phi_m(z'/L) / (kappa z')` (`_phi_m`). Refused by name on tables
        without a wind profile."""
        if self.u_star is None or self.lmo is None or self.z0 is None or self.d is None:
            raise ValueError("PuffTables.shear_rate: these tables carry no wind profile "
                             "(u_star, lmo, z0, d); build them with puff_tables, or advance "
                             "with shear=False")
        u_star, lmo, z0, d = (v[row] for v in (self.u_star, self.lmo, self.z0, self.d))
        height = _f64(z) - d + z0
        return u_star * _phi_m(height / lmo) / (self.kappa * height)

    def _base(self, table: Tensor, row: Tensor) -> Tensor:
        """The flat index of grid point 0 of table `table`, meteorology row `row`."""
        _, n_b, n = self.u.shape
        return (table * n_b + row) * n

    def _cell(self, x: Tensor, base: Tensor) -> tuple[Tensor, Tensor]:
        """`(lo, frac)` of `at_distance` at distance `x` on the rows starting at `base`
        (`_base`): the flat index of the grid point at or below `x` and the fraction of
        the cell beyond it, for `_read`."""
        n = self.u.shape[-1]
        x, base = torch.broadcast_tensors(_f64(x), base)
        pos = torch.clamp(x / self.step, 0.0, float(n - 1))
        i0 = torch.clamp(torch.floor(pos.detach()).long(), 0, n - 2)
        return base + i0, pos - i0

    @staticmethod
    def _read(col: Tensor, lo: Tensor, frac: Tensor) -> Tensor:
        """Column `col` (`(n_tab, n_b, n_grid)`) at the cells `_cell` found."""
        v = col.reshape(-1)
        return v[lo] + (v[lo + 1] - v[lo]) * frac


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
    low that the scan would need more than `MAX_TABLE_STEPS` steps, and the convective
    regime (`h_abl / |lmo| > CONVECTIVE_H_OVER_L`; SIRANE's neutral class is tabulated as in
    `plume_table`), are refused by name. Rows are built in chunks under `KERNEL_BYTES_CAP`
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
    row = {k: p[k].reshape(batch)[0] for k in ("h_r", "h_abl", "u_star", "lmo", "z0", "d")}
    return PuffTables(*cols, h_canopy=row["h_r"], h_abl=row["h_abl"], x_max=float(x_max),
                      u_star=row["u_star"], lmo=row["lmo"], z0=row["z0"], d=row["d"],
                      kappa=kappa)


ALONG_WIND = ("sigma_y", "none")
"""The along-wind profile options of a puff (`PuffField`)."""

_SHEAR_NEEDS_SIGMA_Y = (
    "{where}: shear=True needs along_wind='sigma_y', got along_wind={along_wind!r}: the "
    "'none' slug is a segment of the steady plume, with no along-wind Gaussian for the "
    "shear to stretch or to correlate with the height; pass along_wind='sigma_y' or "
    "shear=False")

_SHEAR_Z_FACTOR = 0.65
"""The shear rate of a puff is `dU/dz` at `max(z_c, 0.65 sigma_z)` (design Sect. 4.7)."""

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
    slug ends (along the wind). `shear=True` evaluates the puffs' wind-shear moments
    `sxz`, `sx2_shear` (design Sect. 4.7), which `advance(..., shear=True)` integrates: it
    needs `along_wind="sigma_y"`. The two must agree (`PuffClosure` passes
    `PuffOptions.shear` to both); a field with `shear=False` ignores the moments, and
    `advance` without shear leaves them as they are."""

    def __init__(self, *, xy: Tensor, owner: Tensor, weight: Tensor, width: Tensor,
                 clamp: Tensor, source_xy0: Tensor, n_streets: int, n_junctions: int,
                 n_releases: int, along_wind: str = "sigma_y",
                 cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS, shear: bool = False) -> None:
        if along_wind not in ALONG_WIND:
            raise ValueError(
                f"PuffField: along_wind must be one of {ALONG_WIND}, got {along_wind!r}")
        if shear and along_wind != "sigma_y":
            raise ValueError(_SHEAR_NEEDS_SIGMA_Y.format(where="PuffField",
                                                         along_wind=along_wind))
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
        self.shear = bool(shear)
        count = torch.bincount(self.owner, minlength=n_sources)
        if bool((self.owner[1:] < self.owner[:-1]).any()) or bool((count == 0).any()) \
                or count.numel() != n_sources:
            raise ValueError("PuffField: every source needs a contiguous run of sub-sources "
                             "in source order")
        self._count = count
        self._start = torch.cumsum(count, 0) - count
        # detached geometry of the coarse mask: each source's widest flat top
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
                    cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS,
                    shear: bool = False) -> PuffField:
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
            cutoff_sigma=cutoff_sigma, shear=shear)

    @classmethod
    def points(cls, n_releases: int, *, along_wind: str = "sigma_y",
               cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS, shear: bool = False) -> PuffField:
        """`n_releases` explicit releases and no network: the standalone puff model."""
        n = int(n_releases)
        if n < 1:
            raise ValueError(f"PuffField.points: n_releases must be >= 1, got {n_releases!r}")
        return cls(xy=torch.zeros(n, 2, dtype=_DTYPE), owner=torch.arange(n),
                   weight=torch.ones(n, dtype=_DTYPE), width=torch.zeros(1, n, dtype=_DTYPE),
                   clamp=torch.full((1, n), math.inf, dtype=_DTYPE),
                   source_xy0=torch.zeros(n, 2, dtype=_DTYPE), n_streets=0, n_junctions=0,
                   n_releases=n, along_wind=along_wind, cutoff_sigma=cutoff_sigma,
                   shear=shear)


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
    it is. Decisions are made on detached values; the masses keep their graph. The
    default `drop_below` here is 1e-6; `PuffClosure` passes `PuffOptions.drop_below`
    (1e-3 by default)."""
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
            theta_w: Tensor | float, *, shear: bool = False) -> PuffState:
    """Every live puff `dt` (s) older and moved along the wind `theta_w` (scalar or `(B,)`,
    radians counter-clockwise from east, toward) by its table's distance increment
    `x(a + dt) - x(a)`: exact for any `dt`, since `x` is the exact travel distance at the
    table's speed, so under constant wind the displacement is `x(a)`. A wind that turns
    between steps turns the puff's further path (design Sect. 4.4).

    `shear=True` also integrates the wind-shear moments (Sykes and Henn, 1995; design
    Sect. 4.7), `d(sxz)/dt = S sigma_z^2` and `d(sx2_shear)/dt = 2 S sxz`, with `S` the
    tables' `shear_rate` at `z_s = max(z_c, 0.65 sigma_z)` at the step's middle age
    `a + dt/2`, held over the step, and `sigma_z^2` over the step the quadratic in age
    through its values at `a`, `a + dt/2`, `a + dt` (`q0`, `qm`, `q1`):

        sxz      += S dt (q0 + 4 qm + q1) / 6
        sx2_shear += 2 S (sxz dt + S dt^2 (q0 + 2 qm) / 6)

    (`sxz` the value at the step's start), which is exact for a constant `S` and a
    `sigma_z^2` quadratic in age. Tables without a wind profile are refused by name.
    Evaluate with a `PuffField(shear=True)`: a field without shear ignores the moments.
    A puff born at a positive age (`PuffOptions.puff_dt`) starts with zero moments,
    missing their growth over that age (cubic and quartic in it, small for intervals of
    tens of seconds)."""
    if not dt > 0:
        raise ValueError(f"advance: dt must be positive, got {dt!r}")
    n_batch = state.mass.shape[0]
    cos, sin = _wind(theta_w, n_batch, "advance")
    row = _table_rows(tables, n_batch, "advance")[:, None]
    age = state.age + dt
    step = (tables.at_age(age, state.table, row)[0]
            - tables.at_age(state.age, state.table, row)[0])
    disp = state.disp + step[..., None] * torch.stack([cos, sin], dim=-1)[:, None, :]
    moments = {}
    if shear:
        _, _, _, s_z, z_c, _ = tables.at_age(
            torch.stack([state.age, state.age + dt / 2.0, age]), state.table, row)
        q0, qm, q1 = s_z * s_z
        rate = tables.shear_rate(torch.maximum(z_c[1], _SHEAR_Z_FACTOR * s_z[1]),
                                 row.expand_as(state.age))
        sxz = state.sxz + rate * dt * (q0 + 4.0 * qm + q1) / 6.0
        sx2 = state.sx2_shear + 2.0 * rate * (state.sxz * dt
                                              + rate * dt * dt * (q0 + 2.0 * qm) / 6.0)
        moments = {"sxz": torch.where(state.live, sxz, state.sxz),
                   "sx2_shear": torch.where(state.live, sx2, state.sx2_shear)}
    return replace(state, age=torch.where(state.live, age, state.age),
                   disp=torch.where(state.live[..., None], disp, state.disp), **moments)


_COARSE_MARGIN_M = 1e-3
"""Slack (m) added to every bound of `_evaluate`'s coarse mask, so that rounding in the
wind-frame projection never drops a pair the exact cut keeps."""


def _evaluate(field: PuffField, state: PuffState, tables: PuffTables, rec_xy: Tensor,
              rec_z: Tensor | None, theta_w: Tensor | float, dt: float,
              receptor_owner: Tensor | None, where: str) -> Tensor:
    """`(B, R)`: the puffs at receptors `rec_xy` `(R, 2)` and heights `rec_z` `(R,)`, or at
    roof level `H_R` with each sub-source's clamp when `rec_z` is None.

    Two levels, as `plume._pair_values`, on the active puffs only (live, non-zero mass,
    age > 0, a slug of positive length; compacted once): a coarse mask of (puff, receptor)
    pairs in each instance's wind frame (along the wind, the source's sub-source extent
    shifted by the puff's displacement and grown by the slug's ends, plus `cutoff_sigma`
    spreads under `"sigma_y"`; across it, the same extent grown by the source's widest
    flat top and `cutoff_sigma` times the largest `sigma_y` up to the slug's far end),
    then the surviving pairs expanded to the source's sub-sources and cut exactly. Masks
    are detached; the kept values carry the graph. A puff of age 0 (not yet advanced) or
    whose slug has zero length contributes nothing. Each receptor sums its contributions
    in slot order, then sub-source order.

    A receptor at along-wind offset `s` from the puff reads the table at its OWN distance
    from the source, `x(a) + s` (design Sect. 4.3, amended): `sigma_y` (crosswind profile
    and cut-off), and at roof level `min(p_z, clamp)`, at a receptor height
    `vertical_profile` of `sigma_z` and `z_c`. Under `"none"` the slug is a segment of the
    steady plume: it spans the distances travelled in the ages `[a - dt/2, a + dt/2]`
    (clamped at 0), a time `tau`, and its along-wind density is `m / (tau u(x(a) + s))`,
    which integrates to `m` over the slug because `int dx / u = tau`; the steady limit of
    a continuous source is then the steady kernel itself. Under `"sigma_y"` the read is
    at `x(a) + s` clamped to the slug's ends, and the along-wind factor is the slug
    convolved with a Gaussian of `sigma_x = sigma_y(x(a))` (density `m / ell`).

    With `field.shear` (design Sect. 4.7) the puff is a correlated `(x, z)` Gaussian of
    along-wind variance `sigma_x^2 = sigma_y(x(a))^2 + sx2_shear` and covariance `sxz`,
    convolved along the wind with the slug, written as the vertical profile times the
    conditional along-wind profile: each term of the vertical profile, centred at `z_k`
    (`z_c`, and the images `2 H_R - z_c`, `2 h - z_c`), multiplies the slug convolved with
    a Gaussian of the conditional variance `sigma_x^2 - sxz^2 / sigma_z(x(a))^2` (shear
    part floored at 0), shifted along the wind by `beta (z - z_k)` with `beta = sxz /
    sigma_z(x(a))^2` for the plume and `-beta` for its images (a reflection in `z` flips
    the sign of `sxz`). `beta` and the conditional variance are the puff's (at `x(a)`), the
    vertical terms the receptor's, as without shear; each term's along-wind integral is
    one, so the line integral along the wind at any height is the unsheared one: the
    shear moves mass along the wind and keeps it exactly. At roof level the plume and its
    canopy image have the same shift, and the factor is `min(p_z, clamp)` times the
    profile's terms' weighted mean of their along-wind factors. The along-wind cut-off
    takes `sigma_x`."""
    if not dt > 0:
        raise ValueError(f"{where}: dt must be positive, got {dt!r}")
    n_batch = state.mass.shape[0]
    cos, sin = _wind(theta_w, n_batch, where)
    row = _table_rows(tables, n_batch, where)
    slug = field.along_wind == "sigma_y"
    cut = field.cutoff_sigma
    width, clamp = _field_rows(field, n_batch, where)
    h_c, h_abl = tables.h_canopy[row], tables.h_abl[row]
    rec_xy = _f64(rec_xy)
    n_rec = rec_xy.shape[0]
    if n_rec == 0:
        return torch.zeros(n_batch, 0, dtype=_DTYPE)
    # every tensor that reaches the values below
    grad = torch.is_grad_enabled() and any(
        v is not None and v.requires_grad
        for v in (state.mass, state.age, state.disp, tables.u, tables.t, tables.sigma_y,
                  tables.sigma_z, tables.z_c, tables.p_z, tables.h_canopy, tables.h_abl,
                  field.xy, field.weight, field.width, field.clamp, rec_xy, rec_z, cos, sin,
                  state.sxz, state.sx2_shear))
    # the active puffs, compacted once: puff n sits in instance bn[n], slot kn[n]
    with torch.no_grad():
        bn, kn = torch.nonzero(state.live & (state.mass != 0) & (state.age > 0),
                               as_tuple=True)
    tab_n, row_n, age = state.table[bn, kn], row[bn], state.age[bn, kn]
    a_lo = torch.clamp(age - dt / 2.0, min=0.0)
    # the puff and its slug's ends, in one lookup
    x3, lo, hi, frac = tables._travel(torch.cat([age, a_lo, age + dt / 2.0]),
                                      tab_n.repeat(3), row_n.repeat(3))
    x_a, x_lo, x_hi = x3.chunk(3)
    lo, hi, frac = lo[:age.numel()], hi[:age.numel()], frac[:age.numel()]
    sigma_y = tables.sigma_y.reshape(-1)
    s_y = sigma_y[lo] + (sigma_y[hi] - sigma_y[lo]) * frac
    e_lo, e_hi = x_lo - x_a, x_hi - x_a
    with torch.no_grad():
        on = torch.nonzero(e_hi > e_lo, as_tuple=True)[0]
    bn, kn, tab_n, row_n, age, a_lo, x_a, s_y, e_lo, e_hi = (
        v[on] for v in (bn, kn, tab_n, row_n, age, a_lo, x_a, s_y, e_lo, e_hi))
    tau = age + dt / 2.0 - a_lo
    src, mass, disp = state.source[bn, kn], state.mass[bn, kn], state.disp[bn, kn]
    s_x = s_y
    if field.shear:
        # the puff's shear moments: the along-wind spread of the cut-off, the regression
        # of x on z and the conditional along-wind spread (docstring)
        sigma_z = tables.sigma_z.reshape(-1)
        lo_on, hi_on, frac_on = lo[on], hi[on], frac[on]
        s_za = sigma_z[lo_on] + (sigma_z[hi_on] - sigma_z[lo_on]) * frac_on
        sxz, sx2 = state.sxz[bn, kn], state.sx2_shear[bn, kn]
        s_x = torch.sqrt(s_y * s_y + sx2)
        var_z = s_za * s_za
        with torch.no_grad():
            spread_z = var_z > 0
        beta = torch.where(spread_z, sxz / torch.where(spread_z, var_z,
                                                       torch.ones_like(var_z)), 0.0)
        s_c = torch.sqrt(s_y * s_y + torch.clamp(sx2 - sxz * beta, min=0.0))
    with torch.no_grad():
        pair_n, pair_i = _coarse_pairs(field, tables, bn, src, disp, tab_n, row_n, x_a,
                                       s_x, e_lo, e_hi, cos, sin, rec_xy, receptor_owner,
                                       slug)
        # chunks of whole pairs with at most `rows_per_chunk(1, 1)` sub-source pairs
        # (the cap's float64 temporaries per pair), each chunk accumulated into `out` in
        # pair order
        ends = torch.cumsum(field._count[src[pair_n]], 0)
        per_chunk = rows_per_chunk(1, 1, backward=grad)
        total = int(ends[-1]) if ends.numel() else 0
        marks = torch.arange(per_chunk, max(total, per_chunk), per_chunk, dtype=ends.dtype)
        bounds = sorted({0, pair_n.numel(),
                         *torch.searchsorted(ends, marks, right=True).tolist()})
        if len(bounds) == 1:
            bounds = [0, 0]

    # per active puff: the along-wind keep bounds and the table rows (with one instance
    # the per-pair instance gathers are skipped)
    with torch.no_grad():
        if slug:
            keep_lo, keep_hi = e_lo - cut * s_x, e_hi + cut * s_x
        else:
            keep_lo, keep_hi = e_lo, e_hi
    base = tables._base(tab_n, row_n)
    one = n_batch == 1
    # coordinates as contiguous columns
    rec_x, rec_y = rec_xy[:, 0].contiguous(), rec_xy[:, 1].contiguous()
    src_x, src_y = field.xy[:, 0].contiguous(), field.xy[:, 1].contiguous()
    disp_x, disp_y = disp[:, 0].contiguous(), disp[:, 1].contiguous()

    def block(out: Tensor, j0: int, j1: int) -> Tensor:
        n, i = pair_n[j0:j1], pair_i[j0:j1]
        s = src[n]
        n_sub = field._count[s]
        rep = torch.repeat_interleave(n_sub)        # pair of each sub-source pair
        p = (field._start[s] - (torch.cumsum(n_sub, 0) - n_sub))[rep] + torch.arange(
            rep.numel())
        n, i = n[rep], i[rep]
        b = None if one else bn[n]
        c_b, s_b = (cos, sin) if one else (cos[b], sin[b])
        d_x = rec_x[i] - (src_x[p] + disp_x[n])
        d_y = rec_y[i] - (src_y[p] + disp_y[n])
        x = d_x * c_b + d_y * s_b
        y = d_y * c_b - d_x * s_b
        with torch.no_grad():
            if slug:
                keep = (x >= keep_lo[n]) & (x <= keep_hi[n])
            else:
                keep = (x >= keep_lo[n]) & (x < keep_hi[n])
            j = torch.nonzero(keep, as_tuple=True)[0]
        n, i, p, x, y = (v[j] for v in (n, i, p, x, y))
        b = None if one else b[j]
        lo, hi = e_lo[n], e_hi[n]
        # the table at the receptor's own distance (clamped to the slug under "sigma_y")
        x_r = x_a[n] + (torch.minimum(torch.maximum(x, lo), hi) if slug else x)
        cell, frac = tables._cell(x_r, base[n])
        sy = tables._read(tables.sigma_y, cell, frac)
        w = width[0, p] if one else width[b, p]
        with torch.no_grad():
            y_c = torch.clamp((w - _SQRT_2PI * sy) / 2.0, min=0.0)
            j = torch.nonzero(y.abs() - y_c <= cut * sy, as_tuple=True)[0]
        n, i, p, x, y, lo, hi, w, sy, cell, frac = (
            v[j] for v in (n, i, p, x, y, lo, hi, w, sy, cell, frac))
        b = None if one else b[j]
        if field.shear:
            c = mass[n] * field.weight[p] * crosswind_profile(y, sy, w) * _sheared_factor(
                tables, cell, frac, x, lo, hi, s_c[n], beta[n],
                h_c if one else h_c[b], h_abl if one else h_abl[b],
                None if rec_z is None else rec_z[i],
                clamp[0, p] if one else clamp[b, p])
            return out.index_add(0, i if one else b * n_rec + i, c)
        if slug:
            root2 = math.sqrt(2.0) * s_y[n]
            g_x = (torch.erf((x - lo) / root2) - torch.erf((x - hi) / root2)) / (
                2.0 * (hi - lo))
        else:
            g_x = 1.0 / (tau[n] * tables._read(tables.u, cell, frac))
        if rec_z is None:
            p_z = torch.minimum(tables._read(tables.p_z, cell, frac),
                                clamp[0, p] if one else clamp[b, p])
        else:
            p_z = vertical_profile(rec_z[i], tables._read(tables.z_c, cell, frac),
                                   tables._read(tables.sigma_z, cell, frac),
                                   h_c if one else h_c[b], h_abl if one else h_abl[b])
        c = mass[n] * field.weight[p] * g_x * crosswind_profile(y, sy, w) * p_z
        return out.index_add(0, i if one else b * n_rec + i, c)

    out = torch.zeros(n_batch * n_rec, dtype=_DTYPE)
    for j0, j1 in zip(bounds[:-1], bounds[1:], strict=True):
        # under autograd each chunk is checkpointed (design Sect. 4.5): the graph keeps
        # the puff state, the backward recomputes the pair products
        if grad and _CHECKPOINT:
            out = checkpoint(block, out, j0, j1, use_reentrant=False)
        else:
            out = block(out, j0, j1)
    return out.reshape(n_batch, n_rec)


def _sheared_factor(tables: PuffTables, cell: Tensor, frac: Tensor, x: Tensor, lo: Tensor,
                    hi: Tensor, s_c: Tensor, beta: Tensor, h_c: Tensor, h_abl: Tensor,
                    z: Tensor | None, clamp: Tensor) -> Tensor:
    """The along-wind times vertical factor (1/m^2) of a sheared puff at the pairs of
    `_evaluate` (its docstring): the receptor at along-wind offset `x` from the puff, the
    slug `[lo, hi]`, the conditional spread `s_c` and regression `beta` of the puff, the
    table read at the receptor's (clamped) distance (`cell`, `frac`), the receptor height
    `z` or roof level (None, with `clamp`)."""
    root2 = math.sqrt(2.0) * s_c

    def g_x(shift: Tensor) -> Tensor:
        u = x - shift
        return (torch.erf((u - lo) / root2) - torch.erf((u - hi) / root2)) / (2.0 * (hi - lo))

    z_c = tables._read(tables.z_c, cell, frac)
    s_z = tables._read(tables.sigma_z, cell, frac)
    top = 2.0 * h_abl - z_c
    if z is None:
        # the inversion image's share of the profile at H_R (the plume and its canopy
        # image have equal shares and shifts), as a logistic of the exponents' difference
        var = 2.0 * torch.clamp(s_z, min=1e-100) ** 2
        share = torch.sigmoid(((h_c - z_c) ** 2 - (top - h_c) ** 2) / var - math.log(2.0))
        p_z = torch.minimum(tables._read(tables.p_z, cell, frac), clamp)
        return p_z * ((1.0 - share) * g_x(beta * (h_c - z_c))
                      + share * g_x(beta * (top - h_c)))

    def g(s: Tensor) -> Tensor:
        return torch.exp(-0.5 * (s / s_z) ** 2)

    image = 2.0 * h_c - z_c
    return (g(z - z_c) * g_x(beta * (z - z_c)) + g(z - image) * g_x(-beta * (z - image))
            + g(top - z) * g_x(beta * (top - z))) / (_SQRT_2PI * s_z)


def _coarse_pairs(field: PuffField, tables: PuffTables, bn: Tensor, src: Tensor,
                  disp: Tensor, tab_n: Tensor, row_n: Tensor, x_a: Tensor, s_x: Tensor,
                  e_lo: Tensor, e_hi: Tensor, cos: Tensor, sin: Tensor, rec_xy: Tensor,
                  receptor_owner: Tensor | None, slug: bool) -> tuple[Tensor, Tensor]:
    """`(n, i)`, detached: the (active puff, receptor) pairs `_evaluate`'s coarse mask
    keeps, ordered by puff. The bounds are taken in each instance's wind frame (along `e =
    (cos, sin)`, across `(-sin, cos)`) and widened by `_COARSE_MARGIN_M`, so they hold every
    pair the exact cut keeps. The receptors are sorted along the wind once; each puff's
    along-wind band is then a run of them (`searchsorted`), and only that run is tested
    across the wind, in batches of at most `rows_per_chunk(1, 1)` candidates. `s_x` is each
    puff's along-wind spread (`sigma_y(x(a))`, with the shear part under `field.shear`)."""
    n_batch, n_src, cut = cos.shape[0], field.n_sources, field.cutoff_sigma
    c_d, s_d = cos.detach(), sin.detach()
    xy = field.xy.detach()
    along_p = xy[:, 0] * c_d[:, None] + xy[:, 1] * s_d[:, None]            # (B, P)
    across_p = xy[:, 1] * c_d[:, None] - xy[:, 0] * s_d[:, None]
    owner = field.owner.expand(n_batch, -1)

    def extent(v: Tensor, how: str, fill: float) -> Tensor:
        """`(B, S)`: the `how` of `v` `(B, P)` over each source's sub-sources."""
        return torch.full((n_batch, n_src), fill, dtype=_DTYPE).scatter_reduce(
            1, owner, v, how)

    d = disp.detach()
    cb, sb = c_d[bn], s_d[bn]
    d_along = d[:, 0] * cb + d[:, 1] * sb
    d_across = d[:, 1] * cb - d[:, 0] * sb
    # the receptors read sigma_y anywhere up to the slug's far end: bound it by the
    # running maximum of the table's sigma_y there
    n_b, n_grid = tables.u.shape[1:]
    far = torch.clamp(torch.ceil((x_a + e_hi).detach() / tables.step).long(), 0,
                      n_grid - 1)
    s_y_far = tables._sigma_y_reach[(tab_n * n_b + row_n) * n_grid + far]
    spread = cut * s_x.detach() if slug else 0.0
    m = _COARSE_MARGIN_M
    lo_a = extent(along_p, "amin", math.inf)[bn, src] + d_along + e_lo.detach() - spread - m
    hi_a = extent(along_p, "amax", -math.inf)[bn, src] + d_along + e_hi.detach() + spread + m
    reach = field._half_width.expand(n_batch, -1)[bn, src] + cut * s_y_far + m
    lo_c = extent(across_p, "amin", math.inf)[bn, src] + d_across - reach
    hi_c = extent(across_p, "amax", -math.inf)[bn, src] + d_across + reach
    rec = rec_xy.detach()
    rec_along = rec[:, 0] * c_d[:, None] + rec[:, 1] * s_d[:, None]          # (B, R)
    rec_across = rec[:, 1] * c_d[:, None] - rec[:, 0] * s_d[:, None]
    n_rec, n_on = rec.shape[0], bn.numel()
    if n_on == 0:
        return bn, bn
    # the receptors sorted by their along-wind coordinate, instance after instance (each
    # instance's keys shifted by a span above every coordinate and bound), so that a
    # puff's along-wind band is one run of the sorted receptors
    big = float(torch.cat([rec_along.abs().reshape(-1), lo_a.abs(), hi_a.abs()]).max())
    span = 2.0 ** math.ceil(math.log2(2.0 * big + 1.0) + 1.0)
    shift = torch.arange(n_batch, dtype=_DTYPE) * span
    key, order = torch.sort((rec_along + shift[:, None]).reshape(-1))
    first = torch.searchsorted(key, lo_a + shift[bn])
    count = torch.clamp(torch.searchsorted(key, hi_a + shift[bn], right=True) - first, min=0)
    ends = torch.cumsum(count, 0)
    budget = rows_per_chunk(1, 1)
    across = rec_across.reshape(-1)
    pair_n, pair_i = [], []
    a_ = 0
    while a_ < n_on:
        # puffs [a_, b_): at most `budget` candidates, at least one puff
        done = int(ends[a_ - 1]) if a_ else 0
        b_ = max(a_ + 1, int(torch.searchsorted(ends, done + budget, right=True)))
        cnt = count[a_:b_]
        rep = torch.repeat_interleave(cnt)          # puff of each candidate, from a_
        flat = order[(first[a_:b_] - (torch.cumsum(cnt, 0) - cnt))[rep]
                     + torch.arange(rep.numel())]
        rep += a_
        i = flat % n_rec if n_batch > 1 else flat
        rc = across[flat]
        keep = (rc >= lo_c[rep]) & (rc <= hi_c[rep])
        if receptor_owner is not None:
            keep &= src[rep] != receptor_owner[i]
        keep = torch.nonzero(keep, as_tuple=True)[0]
        pair_n.append(rep[keep])
        pair_i.append(i[keep])
        a_ = b_
    return torch.cat(pair_n), torch.cat(pair_i)


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
    """An explicit pulse release into the street canopy: `"pulse.rate"` (kg/s) from
    `"pulse.start"` for `"pulse.duration"` (s), added to the streets' sources (design
    Sect. 4.2).

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
    """The options of the unsteady above-roof puff model, `build_model(...,
    puffs=PuffOptions(...))`: canopy height `h_canopy`, `z0`, `d` (m), puff slots
    `capacity` per instance, emission interval `puff_dt` (s), faint threshold `drop_below`
    (design Sect. 4.2-4.8).

    `h_canopy`, `z0`, `d` (m) are the tables' canopy height `H_R`, roughness and
    displacement; `sigma_theta` (rad) the wind-direction spread, `None` for the
    `"sigma_theta"` driver. `junction_source` is one of `JUNCTION_SOURCES`; `along_wind` one
    of `ALONG_WIND`; `capacity` the puff slots per batch instance; `puff_dt` (s) the
    emission interval, `None` (the default) for one puff per source per street step. A
    number is an opt-in that saves capacity, not time: each source's mass is accumulated
    over the street steps of one interval and emitted as one puff whose slug spans the
    interval, and every street step must divide it. Within an interval the newest material
    is pending, not yet a puff, so `C_ext` near the sources lags by up to one interval and
    short releases are smeared over it (measured on `twelve_street_grid` at 30 s / 5 s
    steps: a steady `C_ext` flickering by up to 0.74 of its peak each interval, street
    concentrations within 3 %; a 17 s release's street-concentration peak off by 0.38, its
    dose by 2.5 %); the pair work of `C_ext` does not fall, since a longer slug covers
    proportionally more receptors. `drop_below` the faint threshold of `housekeep`;
    `cutoff_sigma` the profiles' cut-off; `x_max` (m) the extent
    of the tables (a puff beyond it leaves the domain); `n_releases` the explicit
    above-roof releases. `table_kwargs` go to `puff_tables` (the empirical inputs:
    `theta_star`, `temperature`, `tau_y`, `tau_z`, `centre_k`, the floors); a `"centre_c"`
    entry there is the centre of the canopy sources' table (default
    `plume.PLUME_CENTRE_C_M`). `shear=True` turns on the wind-shear stretching of the
    puffs along the wind (Sykes and Henn, 1995; design Sect. 4.7: `advance(...,
    shear=True)` and `PuffField(shear=True)`); it needs `along_wind="sigma_y"`."""

    h_canopy: float = 20.0
    z0: float = 1.0
    d: float = 13.0
    sigma_theta: float | None = None
    junction_source: str = "upward"
    along_wind: str = "sigma_y"
    capacity: int = 4096
    puff_dt: float | None = None
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
    `mass_drawn_down`), and each explicit release's `pulse_mass` over that step (one
    puff per source and step, the default); under a numeric `options.puff_dt` these
    masses are added to `"puff.pending"` instead and emitted, one
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
    last build (a value-only cache would reuse an earlier run's graph). Under
    `meteo="per_street"` the tables read the first street's (or junction's) entry of each
    meteorology driver, so a gradient with respect to those drivers reaches that entry only.

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
        if options.shear and options.along_wind != "sigma_y":
            raise ValueError(_SHEAR_NEEDS_SIGMA_Y.format(where="PuffOptions",
                                                         along_wind=options.along_wind))
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
                along_wind=o.along_wind, cutoff_sigma=o.cutoff_sigma, shear=o.shear)
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
        shear = self.options.shear

        def puff_step(puff: PuffState) -> tuple[PuffState, Tensor, Tensor]:
            if emitted is None:
                evicted = torch.zeros(n_batch, dtype=torch.long)
            else:
                puff, evicted = emit(field, puff, emitted, table_index, release_xy,
                                     tables=tables, age=age0, theta_w=theta)
            if drawn is not None:
                puff = replace(puff, mass_drawn_down=puff.mass_drawn_down + drawn)
            # (2) advance, (3) housekeep, (4) evaluate
            puff = advance(puff, tables, dt, theta, shear=shear)
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

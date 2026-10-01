"""Above-roof street-to-street transport: SIRANE's street-plume kernel.

Each street's roof-level flux and each junction's vertical flux is a set of passive
sources at roof level; the concentration above a receiving street, `C_ext`, is the
background plus the superposed plumes of every upwind source. This module builds the
linear map from source rates to `C_ext`, in s/m3 (concentration per unit source rate), as a
kernel matrix.

The kernel is SIRANE v2.1's own mechanism, read off SIRANE reference data for
single-street cases (one isolated street, imposed meteorology, the concentration grid; 38
cases across wind angle, street width, height and length, the reflection height `H_R`, the
site's displacement height and a neutral and stable meteorology scan). The physics it
rests on is Soulhac, Salizzoni, Cierco and Perkins (2011), "The model SIRANE for atmospheric urban
pollutant dispersion; part I, presentation of the model", Atmos. Environ. 45:7379-7395
(equation numbers below are that paper's). SIRANE does not evaluate a closed-form Gaussian
for each source-receptor pair; it tabulates one plume trajectory per hour and reads every
pair off that table.

The trajectory table (`plume_table`), per hour:

- Meteorology at `z = H_R` (`h_canopy` here): the Monin-Obukhov wind `U(z)` of Eqs.
  (10)-(11) (`wind_speed`), `sigma_v`, `sigma_w` of Eqs. (15)-(16) floored at
  `sigma_v_min`, `sigma_w_min` (SIRANE's `SIGMA_V_MIN`, `SIGMA_W_MIN`), the Brunt-Vaisala
  frequency `N^2 = (g/T)(theta*/kappa)(1/z + 5/L)` and the `sigma_y` coefficient
  `a = 2.5 u*/h` (neutral) or `2.5 u* L/h^2` (stable). There is no floor on the plume's
  advection speed (SIRANE's `U_MIN` does not act on it).
- Nodes every `TIME_STEP_S` = 10 s: `t_j = 10 j`, `X_j = X_{j-1} + 10 U_j`. The plume
  centre height is `z_c,j = max(H_R, d + E|N(c - d, (k sigma_z,j-1)^2)|)` from the PREVIOUS
  node's `sigma_z` (`d` the displacement height, `c`, `k` below), and the node's speed is
  `U_j = U(max(floor(z_c,j), H_R))` -- the speed steps through integer heights.
- `sigma_z,j` is Eq. (29) at `t_j - tau_z`; `sigma_y,j = sqrt(sigma_v^2 tau^2 / (1 + a tau)
  + (sigma_theta (X_j - U_0 tau_y))^2)` with `tau = t_j - tau_y` (Eq. 28 with time offsets),
  both offsets clamped at zero time.
- `P_z,j`: a Gaussian of `sigma_z,j` centred at `z_c,j`, evaluated at `H_R` with its image
  in the canopy top and its image in the inversion `h`: `(2 g(z_c - H_R) + g(2h - z_c -
  H_R)) / (sqrt(2 pi) sigma_z)`.
- Node 0 is the source: `X = 0`, `U = U(H_R)`, `sigma_y = SIGMA_Y0_M`, `P_z = P_z,1`.
- The nodes are resampled onto a `TABLE_STEP_M` = 10 m distance grid by linear
  interpolation; every pair is read from that grid by linear interpolation again.

A source-receptor pair at downwind distance `x` and crosswind offset `y` (the receptor's
height is ignored) then contributes, per unit rate,

    C = f_y(y; sigma_y(x), w) * min(P_z(x), clamp) / U(x),

    f_y = [|y| <= y_c ? 1 : exp(-(|y| - y_c)^2 / 2 sigma_y^2)] / (2 y_c + sqrt(2 pi) sigma_y),
    y_c = max(0, (w - sqrt(2 pi) sigma_y) / 2),

a flat top of the source's width `w` spliced into a Gaussian tail, zero upwind (`x <= 0`),
beyond the downwind cut-off and more than `cutoff_sigma` standard deviations beyond the
flat top crosswind.

Sources:

- A street of coordinate length `L` is `n = floor(L / 10) + 1` equal sub-sources at the
  centres of `n` equal segments (`source_points`), each of width `w = (L/n)|sin phi|`
  (`phi` the angle between street and wind) and vertical clamp
  `min(10/W, 1/(H (1 - |sin phi|)))` with the street's width `W` and height `H`.
- A junction is one point of width `W_j` and clamp `1/H_j`, the mean width and height of
  the streets that meet there (`junction_sources`).
- `C_ext` of a street is taken at its segment midpoint, without the street's own
  sub-sources (`street_kernel(self_contribution=False)`).

Empirical inputs. Four elements of the mechanism are fitted to SIRANE's output rather than
derived; each is a named parameter with a default:

- `tau_y`, `tau_z` (s), the time offsets of `sigma_y` and `sigma_z`: by default the
  piecewise-linear tables `TAU_Y_BY_SIGMA_V` and `TAU_Z_BY_SIGMA_W` of the hour's floored
  `sigma_v` and `sigma_w` (see those constants for the fitted values behind each knot).
- `centre_c` (m) and `centre_k`, the plume-centre law: defaults `PLUME_CENTRE_C_M` = 10 m
  and `PLUME_CENTRE_K` = 0.675.
- `theta_star` (K), the temperature scale in `N`: by default `u*^2 T / (kappa g L)`, which
  makes `N^2 = u*^2 / (kappa^2 L) (1/z + 5/L)`. SIRANE's own preprocessor can print a
  `theta*` that is not this value (it clamps `L` separately); pass the value that goes with
  the hour when it is known. On the one real stable SIRANE hour available (South
  Kensington, where SIRANE prints 0.072 K and the probe cases of that hour fit 0.060 K) the
  default gives `C_ext` about 5 % off at the median, against 0.2-0.3 % with 0.060 K.
- the downwind cut-off (m): `downwind_cutoff(theta_w, meteo_cell_dx) = meteo_cell_dx /
  |cos theta_w|` when the kernels are given the x-size of SIRANE's meteo-grid cell (valid
  for cells of about 700 m and larger; smaller cells switch SIRANE to a cell-based far
  field, which this kernel does not model), and the table's `x_max`, by default
  `DOWNWIND_CUTOFF_M` = 700 m (700 m cells, wind along x), when it is not given. The
  probe cases' grids end between 688 and 708 m with 700 m cells (708 m in one of the 38
  runs; 686-700 m in the ten committed as test fixtures).

Not implemented, by name: the unstable regime (`L < 0`; SIRANE's bi-Gaussian, Eq. 26, and
the unstable Lagrangian time scale of Eq. 18, which the paper prints with inconsistent
units): `plume_table` raises `NotImplementedError`. Plume rise (roof sources are passive)
and wet-deposition depletion. SIRANE's `B_RUE_DECOUP = 1` path (streets cut on the
retro-trajectory grid), which uses another, cell-based mechanism.

Coupling: `above_roof.street_steady_with_plume` iterates the per-street background
`C_ext = C_bg + K F`; there is no `build_model` option because the kernels depend on the
hour's wind.

Differentiability: the table and the kernel are differentiable in the fluxes (the kernel
is linear in them), `u*`, `L`, `h`, `sigma_theta`, `theta*`, the temperature, the
roughness and displacement, the reflection height `h_canopy`, explicit `tau_y`, `tau_z`,
`centre_c`, `centre_k` and the wind direction. The streets' widths and heights (which set
the sub-source widths and clamps) are plain floats of the network and carry no gradient.
The table is built without a recorded graph and rebuilt in the backward pass: first
derivatives only, and a second-order request (`create_graph=True`) raises by name. These
elements are piecewise constant or piecewise linear, each exact to SIRANE's mechanism:

- `floor(z_c)` in the node speed: the speed steps through integer heights, so it has no
  derivative with respect to `z_c` (the derivative through `u*`, `L` and `d` inside
  `U(z)` is kept); the same holds for `z_c = H_R` while the `max` binds.
- the 10 m table and the linear reads from it: derivatives in `x` (and so in the wind
  direction) are piecewise constant, with kinks every 10 m.
- the default `tau_y`, `tau_z` tables are piecewise linear in `sigma_v`, `sigma_w`: kinks
  at the knots, constant beyond the end knots; below a floor (`sigma_v_min`,
  `sigma_w_min`) the floored value has no derivative.
- `min(P_z, clamp)`, the flat top's `max(0, ...)` and `|sin phi|` (at a street exactly
  along the wind) have kinks.
- the cut-offs are hard: pairs upwind, beyond the downwind cut-off or beyond
  `cutoff_sigma` jump to
  exactly zero; the gradient is that of whichever side a pair sits on and does not see
  the jump. `cutoff_sigma=math.inf` removes the crosswind cut-off.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch.utils.checkpoint import checkpoint

from noodl._broadcast import broadcast_shapes
from noodl.apps.street_aq.canyon import KAPPA
from noodl.apps.street_aq.closures import SIGMA_V_MIN, SIGMA_W_MIN
from noodl.apps.street_aq.network import StreetNetwork

Tensor = torch.Tensor
_DTYPE = torch.float64

GRAVITY = 9.81
"""m/s2, as in SIRANE's `N`."""

GAUSS_CUTOFF_SIGMAS = 4.0
"""The Gaussian cut-off, in lateral standard deviations: pairs more than this many
`sigma_y` beyond the flat top contribute exactly zero. SIRANE's keyword `SEUIL_GAUSS`
("Sigma threshold to neglect a puff", default 4.0)."""

SEUIL_GAUSS = GAUSS_CUTOFF_SIGMAS
"""Earlier name of `GAUSS_CUTOFF_SIGMAS`."""


TIME_STEP_S = 10.0
"""The trajectory's time step, s: the node spacing of SIRANE's plume is exactly `10 U`
(31.192 m at `U(H_R)` = 3.119 m/s, 43.357 m at 4.336 m/s)."""

TABLE_STEP_M = 10.0
"""The distance grid the trajectory is resampled onto, m: the concentration along a
plume's axis has a kink every 10 m, and the output grid spacing (2 m or 4 m) does not
change it."""

DOWNWIND_CUTOFF_M = 700.0
"""The default table extent and downwind cut-off `x_max`, m, used when the meteo-grid cell
is not given (an empirical input: with SIRANE's 700 m meteo cells and the wind along x the
plumes end between 688 and 708 m downwind of their source -- 708 m in one of the 38
runs, 686-700 m in the ten committed as test fixtures; see `downwind_cutoff`)."""

SIGMA_Y0_M = 0.1
"""`sigma_y` at the source node, m."""

SUBSOURCE_LENGTH_M = 10.0
"""A street of length `L` is `floor(L / SUBSOURCE_LENGTH_M) + 1` sub-sources (confirmed at
L = 101, 150, 199 and 200 m)."""

CLAMP_LENGTH_M = 10.0
"""A street sub-source's `P_z` is capped at `CLAMP_LENGTH_M / W` (a plateau of 0.500 at
W = 20 m and 0.250 at W = 40 m, independent of `H` and `H_R`)."""

PLUME_CENTRE_C_M = 10.0
"""`c` of the plume-centre law `z_c = max(H_R, d + E|N(c - d, (k sigma_z)^2)|)`, m (an
empirical input, fitted to the integer-height speed steps of the neutral runs: the speed
steps land on `U(21)`, `U(22)`, `U(24)`, ... identically at `H_R` = 20 and 30 m; one step
in 33 still differs)."""

PLUME_CENTRE_K = 0.675
"""`k` of the plume-centre law (see `PLUME_CENTRE_C_M`); for large `sigma_z` the law tends
to `d + 0.54 sigma_z`."""

TAU_Y_BY_SIGMA_V = ((0.26, 1.03), (0.295, 1.28), (0.50, 1.86), (0.59, 2.03), (1.18, 2.72),
                    (2.36, 3.26))
"""The default `tau_y` (s) as a piecewise-linear function of the hour's floored `sigma_v`
at `H_R` (m/s), constant beyond the end knots (an empirical input). Each knot is the
offset that best reproduces SIRANE's concentration grids (median plus 90th percentile of
the relative error over the cells above 1 % of the maximum) for one meteorology, with the
plume-centre law at its defaults: `u*` = 0.138 and 0.14 m/s stable (`sigma_v` 0.258 and
0.261: 0.98 and 1.07 s), 0.15 neutral (0.295: 1.28), 0.14 stable with SIRANE's floors
(0.50: 1.86), 0.30 stable and neutral (0.577 and 0.590: 2.00 and 2.06), 0.60 neutral
(1.18: 2.72), 1.20 neutral (2.36: 3.26). The offsets grow with `sigma_v` in one monotone
sequence across the neutral and stable runs."""

TAU_Z_BY_SIGMA_W = ((0.169, -0.89), (0.192, 0.32), (0.30, 0.56), (0.38, 0.64), (0.768, 1.53),
                    (1.535, 2.20))
"""The default `tau_z` (s) as a piecewise-linear function of the hour's floored `sigma_w`
at `H_R` (m/s), constant beyond the end knots (an empirical input), from the same fits as
`TAU_Y_BY_SIGMA_V`: `sigma_w` 0.168 and 0.170 (-1.00 and -0.78 s), 0.192 (0.32), 0.30
(0.56), 0.375 and 0.384 (0.55 and 0.74), 0.768 (1.53), 1.535 (2.20). A negative offset
widens `sigma_z` at the first nodes; it stands in for the stable `sigma_z` law, which the
runs do not pin down exactly."""

DEFAULT_TEMPERATURE_K = 288.15
"""The air temperature used with an explicit `theta_star`, K (the default `theta*` does not
need one)."""

MAX_TABLE_STEPS = 10_000
"""The largest number of 10 s trajectory steps `plume_table` takes before refusing."""

_TABLE_BYTES_PER_STEP = 2048
"""The bytes one hour's 10 s step may hold at most in `plume_table`, its graph under
autograd included (about 60 float64 scalars per step, rounded up); the table is built in
chunks of hours whose summed step counts keep this under `KERNEL_BYTES_CAP`."""

KERNEL_BYTES_CAP = 64 * 2**20
"""The largest (receptors x sources) block any function here materialises, in bytes."""

_LIVE_TEMPORARIES = 16
"""How many (batch, receptors, sources) float64 blocks one forward chunk may hold at its
peak; the chunk size divides the cap by it, so the whole evaluation of a chunk, not only
its result, stays under the cap."""

_BACKWARD_TEMPORARIES = 128
"""The same, when a gradient is being recorded: the checkpointed backward of one chunk
recomputes it and holds the compacted pairs' gradients too. Measured as the rise of the
process's peak working set (Windows `K32GetProcessMemoryInfo`) on the 577-street lattice
with 5770 sub-sources: `street_kernel` forward raises it by 2 MB, forward plus the backward
in `u*` by 29 MB."""

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_TABLE_KEYS = ("sigma_y", "p_z", "u")


def _f64(value: Tensor | float) -> Tensor:
    return torch.as_tensor(value, dtype=_DTYPE)


def _regimes(lmo: Tensor, neutral_abs_lmo: float) -> tuple[Tensor, Tensor, Tensor]:
    """`(unstable, neutral, stable)` masks. `L = 0` and NaN are refused by name."""
    if bool(torch.isnan(lmo).any()) or bool((lmo == 0).any()):
        raise ValueError(
            "plume: the Monin-Obukhov length lmo must be non-zero and not NaN; pass "
            "math.inf for a neutral boundary layer (SIRANE's 'L_MO -> infinity')"
        )
    neutral = lmo.abs() >= neutral_abs_lmo
    return (lmo < 0) & ~neutral, neutral, (lmo > 0) & ~neutral


def turbulence(
    z: Tensor | float,
    *,
    u_star: Tensor | float,
    h_abl: Tensor | float,
    lmo: Tensor | float,
    theta_star: Tensor | float | None = None,
    temperature: Tensor | float = DEFAULT_TEMPERATURE_K,
    kappa: float = KAPPA,
    neutral_abs_lmo: float = math.inf,
) -> tuple[Tensor, Tensor, Tensor]:
    """`(sigma_v, sigma_w, N)` at height `z` above ground, Eqs. (15)-(17), UNFLOORED.
    Broadcasts over all arguments.

    Neutral is `L -> infinity`, i.e. `lmo = math.inf` (or `|lmo| >= neutral_abs_lmo`).
    `w* = u* (h / (kappa |L|))^(1/3)` (Eq. 17, written with `|L|`: Eq. 17 prints `L`, which
    is negative wherever `w*` is used). `N` is zero outside the stable regime; in it,
    `N^2 = (g/T)(theta*/kappa)(1/z + 5/L)` from the stable potential-temperature profile of
    Eqs. (12)-(13) (`psi_h = -5 zeta`, `z_T` dropped). `theta_star=None` takes
    `theta* = u*^2 T / (kappa g L)` (the definition of `L` under Eq. 10), for which
    `N^2 = u*^2 / (kappa^2 L) (1/z + 5/L)` whatever the temperature.
    """
    z, u_star, h, lmo = (_f64(v) for v in (z, u_star, h_abl, lmo))
    unstable, neutral, stable = _regimes(lmo, neutral_abs_lmo)
    ratio = z / h
    sv_neutral = 2.0 * u_star * (1.0 - 0.8 * ratio)
    sw_neutral = 1.3 * u_star * (1.0 - 0.8 * ratio)
    sv_stable = 2.0 * u_star * (1.0 - 0.5 * ratio) ** 0.75
    sw_stable = 1.3 * u_star * (1.0 - 0.5 * ratio) ** 0.75
    safe_abs = torch.where(unstable, lmo.abs(), torch.ones_like(lmo))
    w_star = u_star * (h / (kappa * safe_abs)) ** (1.0 / 3.0)
    sv_unstable = torch.sqrt(0.3 * w_star**2 + sv_neutral**2)
    sigma_wc = math.sqrt(0.4) * w_star * 2.1 * ratio ** (1.0 / 3.0) * (1.0 - 0.8 * ratio)
    sw_unstable = torch.sqrt(sigma_wc**2 + sw_neutral**2)
    safe_l = torch.where(stable, lmo, torch.ones_like(lmo))
    if theta_star is None:
        n2 = u_star**2 / (kappa**2 * safe_l) * (1.0 / z + 5.0 / safe_l)
    else:
        n2 = (GRAVITY / _f64(temperature)) * (_f64(theta_star) / kappa) * (
            1.0 / z + 5.0 / safe_l)
    n_bv = torch.where(stable, torch.sqrt(torch.where(stable, n2, torch.ones_like(n2))),
                       torch.zeros_like(n2))
    sigma_v = torch.where(unstable, sv_unstable, torch.where(stable, sv_stable, sv_neutral))
    sigma_w = torch.where(unstable, sw_unstable, torch.where(stable, sw_stable, sw_neutral))
    return sigma_v, sigma_w, n_bv


def _psi_m(zeta: Tensor) -> Tensor:
    """Eq. (11): Businger's momentum stability function; `zeta = 0` is neutral."""
    unstable = zeta < 0
    safe = torch.where(unstable, zeta, torch.zeros_like(zeta))
    x = (1.0 - 16.0 * safe) ** 0.25
    psi_u = (2.0 * torch.log((1.0 + x) / 2.0) + torch.log((1.0 + x * x) / 2.0)
             - 2.0 * torch.atan(x) + math.pi / 2.0)
    return torch.where(unstable, psi_u, -5.0 * zeta)


def wind_speed(
    z: Tensor | float,
    *,
    u_star: Tensor | float,
    lmo: Tensor | float,
    z0: Tensor | float,
    d: Tensor | float,
    kappa: float = KAPPA,
) -> Tensor:
    """The mean wind above the canopy, Eqs. (10)-(11):
    `u = (u*/kappa) [ln((z - d + z0)/z0) - psi_m((z - d + z0)/L) + psi_m(z0/L)]`,
    with `z0` and `d` the district roughness and displacement (`Z0D`, `ZDISPL`).
    `lmo = math.inf` is neutral (`zeta = 0`)."""
    z, u_star, lmo, z0, d = (_f64(v) for v in (z, u_star, lmo, z0, d))
    zeta_top = (z - d + z0) / lmo
    zeta_bottom = z0 / lmo
    return (u_star / kappa) * (
        torch.log((z - d + z0) / z0) - _psi_m(zeta_top) + _psi_m(zeta_bottom)
    )


def _piecewise_linear(x: Tensor, knots: tuple[tuple[float, float], ...]) -> Tensor:
    """Linear interpolation through `knots` `((x, y), ...)`, constant beyond the ends."""
    xs = torch.tensor([k[0] for k in knots], dtype=_DTYPE)
    ys = torch.tensor([k[1] for k in knots], dtype=_DTYPE)
    xc = torch.clamp(x, min=float(xs[0]), max=float(xs[-1]))
    i = torch.clamp(torch.searchsorted(xs, xc.detach(), right=True), 1, len(knots) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (xc - x0) / (x1 - x0)


def default_tau(sigma_v: Tensor | float, sigma_w: Tensor | float) -> tuple[Tensor, Tensor]:
    """`(tau_y, tau_z)`, s: the default time offsets for floored `sigma_v`, `sigma_w` at
    `H_R`, from `TAU_Y_BY_SIGMA_V` and `TAU_Z_BY_SIGMA_W`."""
    return (_piecewise_linear(_f64(sigma_v), TAU_Y_BY_SIGMA_V),
            _piecewise_linear(_f64(sigma_w), TAU_Z_BY_SIGMA_W))


@dataclass(frozen=True)
class PlumeTable:
    """One plume trajectory per hour, resampled onto a distance grid (`plume_table`).

    `sigma_y`, `p_z` and `u` are `(n_batch, n_grid)`: the crosswind standard deviation (m),
    the vertical profile at `H_R` (1/m) and the advection speed (m/s) at grid distances
    `0, step, 2 step, ...`; `batch` is the meteorology's broadcast shape (`n_batch` its
    product, at least 1). Pairs further downwind than `x_max` are zero."""

    batch: tuple[int, ...]
    sigma_y: Tensor
    p_z: Tensor
    u: Tensor
    step: float
    x_max: float

    @property
    def distance(self) -> Tensor:
        """`(n_grid,)`: the grid distances, m."""
        return torch.arange(self.sigma_y.shape[-1], dtype=_DTYPE) * self.step

    def lookup(self, x: Tensor, batch_index: Tensor | None = None
               ) -> tuple[Tensor, Tensor, Tensor]:
        """`(sigma_y, p_z, u)` at downwind distances `x` (linear between grid points;
        distances outside the grid read its end values), for the flat batch rows
        `batch_index` (default row 0)."""
        return _lookup(dict(zip(_TABLE_KEYS, (self.sigma_y, self.p_z, self.u),
                                strict=True)), x, batch_index, self.step)


def _lookup(rows: dict, x: Tensor, b: Tensor | None, step: float
            ) -> tuple[Tensor, Tensor, Tensor]:
    n_grid = rows["u"].shape[-1]
    pos = torch.clamp(x / step, 0.0, float(n_grid - 1))
    i0 = torch.clamp(torch.floor(pos.detach()).long(), 0, n_grid - 2)
    frac = pos - i0
    flat = i0 if b is None else b * n_grid + i0
    out = []
    for key in _TABLE_KEYS:
        v = rows[key].reshape(-1)
        out.append(v[flat] + (v[flat + 1] - v[flat]) * frac)
    return out[0], out[1], out[2]


def _expected_abs(mean: Tensor, sd: Tensor) -> Tensor:
    """`E|X|` for `X ~ N(mean, sd^2)`."""
    return (sd * _SQRT_2_OVER_PI * torch.exp(-0.5 * (mean / sd) ** 2)
            + mean * torch.erf(mean / (sd * math.sqrt(2.0))))


def plume_table(
    *,
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
    centre_c: Tensor | float = PLUME_CENTRE_C_M,
    centre_k: Tensor | float = PLUME_CENTRE_K,
    x_max: float = DOWNWIND_CUTOFF_M,
    kappa: float = KAPPA,
    neutral_abs_lmo: float = math.inf,
) -> PlumeTable:
    """SIRANE's plume trajectory for each hour, resampled onto a 10 m grid (module
    docstring: the table).

    `u_star` (m/s), `h_abl` (m), `lmo` (m, `math.inf` neutral), `sigma_theta` (the wind-
    direction standard deviation, radians), `h_canopy` (SIRANE's reflection height `H_R`,
    m), `z0` and `d` (the district roughness and displacement height, m: SIRANE's
    dispersion site `Z0D`, `ZDISPL`), `theta_star` and `temperature` broadcast to one batch
    shape (hours). `sigma_v_min`, `sigma_w_min` are the turbulence floors (SIRANE's
    defaults; 0 switches a floor off). `tau_y`, `tau_z` (s) default to `default_tau` of the
    floored `sigma_v`, `sigma_w`; `centre_c`, `centre_k` and `theta_star` are the other
    empirical inputs (module docstring). `x_max` is the table's extent and the downwind
    cut-off of a kernel that is not given `meteo_cell_dx`; with it, build the table to
    `downwind_cutoff(theta_w, meteo_cell_dx).max()`. The hours are built in chunks under
    `KERNEL_BYTES_CAP` (`table_chunks`), each step advancing only the hours still short of
    `x_max`. Needs `h_abl > h_canopy > d`, `u* > 0` and
    `tau_z < TIME_STEP_S`; the unstable regime raises `NotImplementedError`.
    """
    u_star, h_abl, lmo, h_r, z0, d, sphi = (
        _f64(v) for v in (u_star, h_abl, lmo, h_canopy, z0, d, sigma_theta)
    )
    if not bool((u_star > 0).all()):
        raise ValueError("plume_table: u_star must be strictly positive")
    if not bool((h_abl > h_r).all()):
        raise ValueError(
            f"plume_table: the boundary-layer height h_abl must exceed the reflection "
            f"height h_canopy (the profile reflects at both); got h_abl down to "
            f"{float(h_abl.min())} m against h_canopy {float(h_r.max())} m"
        )
    if not bool((h_r > d).all()):
        raise ValueError(
            f"plume_table: the reflection height h_canopy must exceed the displacement "
            f"height d (the wind profile starts at d); got {float(h_r.min())} m against "
            f"{float(d.max())} m"
        )
    if not x_max > 0:
        raise ValueError(f"plume_table: x_max must be positive, got {x_max!r}")
    unstable, _, stable = _regimes(lmo, neutral_abs_lmo)
    if bool(unstable.any()):
        raise NotImplementedError(
            "plume_table: the unstable regime (lmo < 0) is not implemented -- its vertical "
            "profile is the bi-Gaussian of Soulhac et al. (2011) Eq. (26) with the "
            "unstable T_L of Eq. (18); see noodl.apps.street_aq.plume"
        )
    sv, sw, n_bv = turbulence(h_r, u_star=u_star, h_abl=h_abl, lmo=lmo,
                              theta_star=theta_star, temperature=temperature, kappa=kappa,
                              neutral_abs_lmo=neutral_abs_lmo)
    sv = torch.clamp(sv, min=float(sigma_v_min))
    sw = torch.clamp(sw, min=float(sigma_w_min))
    safe_l = torch.where(stable, lmo, torch.ones_like(lmo))
    a = torch.where(stable, 2.5 * u_star * safe_l / h_abl**2, 2.5 * u_star / h_abl)
    ty_default, tz_default = default_tau(sv, sw)
    ty = ty_default if tau_y is None else _f64(tau_y)
    tz = tz_default if tau_z is None else _f64(tau_z)
    if not bool((tz < TIME_STEP_S).all()):
        raise ValueError(
            f"plume_table: tau_z must be below the {TIME_STEP_S} s time step (sigma_z at "
            f"the first node would be zero); got {float(tz.max())} s"
        )
    parts = {"u_star": u_star, "h_abl": h_abl, "lmo": lmo, "h_r": h_r, "z0": z0, "d": d,
             "sphi": sphi, "sv": sv, "sw": sw, "n_bv": n_bv, "a": a, "ty": ty, "tz": tz,
             "c": _f64(centre_c), "k": _f64(centre_k)}
    batch = tuple(broadcast_shapes(*(v.shape for v in parts.values())))
    p = {key: v.expand(batch).reshape(-1) for key, v in parts.items()}
    n_grid = int(math.floor(x_max / TABLE_STEP_M + 1e-9)) + 1
    if (n_grid - 1) * TABLE_STEP_M < x_max:
        n_grid += 1
    n_grid = max(n_grid, 2)
    grid_end = (n_grid - 1) * TABLE_STEP_M
    u0 = wind_speed(p["h_r"], u_star=p["u_star"], lmo=p["lmo"], z0=p["z0"], d=p["d"],
                    kappa=kappa)
    if not bool((u0 > 0).all()):
        raise ValueError("plume_table: the wind speed at h_canopy must be positive")
    # The speed never falls below U(H_R), so this bounds each hour's step count.
    steps = torch.ceil((grid_end + TIME_STEP_S * u0.detach()) / (TIME_STEP_S * u0.detach()))
    if bool((steps > MAX_TABLE_STEPS).any()):
        raise ValueError(
            f"plume_table: the trajectory needs up to {int(steps.max())} steps of "
            f"{TIME_STEP_S} s to reach {grid_end} m, above MAX_TABLE_STEPS = "
            f"{MAX_TABLE_STEPS} (wind speed at h_canopy down to {float(u0.min()):.3g} m/s)"
        )
    keys = tuple(p)
    grad = torch.is_grad_enabled() and any(v.requires_grad for v in p.values())
    out = []
    for a_, b_ in table_chunks(steps.tolist()):
        values = [p[k][a_:b_] for k in keys]
        if grad:
            out.append(_RecomputedTrajectory.apply(keys, n_grid, kappa, *values))
        else:
            out.append(_trajectory(dict(zip(keys, values, strict=True)), n_grid, kappa))
    s_y, p_z, u = (torch.cat([o[i] for o in out]) for i in range(3))
    return PlumeTable(batch=batch, sigma_y=s_y, p_z=p_z, u=u, step=TABLE_STEP_M,
                      x_max=float(x_max))


def table_chunks(steps: list[float]) -> list[tuple[int, int]]:
    """`[(start, stop), ...]`: consecutive hours whose summed step counts stay under
    `KERNEL_BYTES_CAP / _TABLE_BYTES_PER_STEP` (one hour per chunk at least), so that one
    chunk's trajectory -- and, under autograd, its checkpointed graph -- stays under the
    cap."""
    budget = max(1.0, KERNEL_BYTES_CAP / _TABLE_BYTES_PER_STEP)
    chunks, start, used = [], 0, 0.0
    for i, s in enumerate(steps):
        if i > start and used + s > budget:
            chunks.append((start, i))
            start, used = i, 0.0
        used += s
    chunks.append((start, len(steps)))
    return chunks


class _RecomputedTrajectory(torch.autograd.Function):
    """One chunk of `_trajectory` that keeps no graph: the forward runs without one and
    the backward rebuilds the chunk's graph and differentiates it. The graph of a 10 s
    scan is thousands of small nodes per chunk whatever its row count, so recording it
    for every chunk would not stay under `KERNEL_BYTES_CAP`. First derivatives only: the
    backward builds no graph over its own result, and `once_differentiable` does not catch a
    `create_graph=True` request (the convention of `noodl.solvers.implicit`), so `backward`
    checks `torch.is_grad_enabled()` at its entry -- off in every ordinary backward pass, on
    only under `create_graph=True` -- and raises by name."""

    @staticmethod
    def forward(ctx, keys, n_grid, kappa, *values):
        ctx.keys, ctx.n_grid, ctx.kappa = keys, n_grid, kappa
        ctx.save_for_backward(*values)
        return _trajectory(dict(zip(keys, values, strict=True)), n_grid, kappa)

    @staticmethod
    def backward(ctx, *grads):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "plume_table: second-order differentiation (create_graph=True) is not "
                "supported; the trajectory table has first-order gradients only. Detach "
                "the first-order gradient before using it in a further differentiable loss."
            )
        values = ctx.saved_tensors
        leaves = [v.detach().requires_grad_(ctx.needs_input_grad[3 + i])
                  for i, v in enumerate(values)]
        wanted = [v for v in leaves if v.requires_grad]
        with torch.enable_grad():
            out = _trajectory(dict(zip(ctx.keys, leaves, strict=True)), ctx.n_grid,
                              ctx.kappa)
            found = iter(torch.autograd.grad(out, wanted, grads, allow_unused=True))
        return (None, None, None,
                *(next(found) if v.requires_grad else None for v in leaves))


def _trajectory(p: dict, n_grid: int, kappa: float) -> tuple[Tensor, Tensor, Tensor]:
    """`(sigma_y, p_z, u)`, each `(m, n_grid)`, for the `m` hours in `p`. Each 10 s step
    advances only the hours whose trajectory has not yet passed the grid's end, and every
    segment between two nodes fills the grid points it covers (`X_lo <= x < X_hi`) by
    linear interpolation as it is made."""
    m = p["u_star"].shape[0]
    grid_end = (n_grid - 1) * TABLE_STEP_M

    def speed(z: Tensor, q: dict) -> Tensor:
        return wind_speed(z, u_star=q["u_star"], lmo=q["lmo"], z0=q["z0"], d=q["d"],
                          kappa=kappa)

    rows = torch.arange(m)
    q = dict(p)
    u0 = speed(q["h_r"], q)
    q["u0"] = u0
    x, u, s_y, sz_prev = torch.zeros_like(u0), u0, torch.full_like(u0, SIGMA_Y0_M), \
        torch.zeros_like(u0)
    p_z = None
    index, values = [], ([], [], [])
    j = 0
    while rows.numel():
        j += 1
        t = TIME_STEP_S * j
        t_y = torch.clamp(t - q["ty"], min=0.0)
        t_z = torch.clamp(t - q["tz"], min=0.0)
        nt = q["n_bv"] * t_z
        s_z = q["sw"] * t_z / torch.sqrt(6.25 + nt * nt / (1.0 + 2.0 * nt))
        spread = torch.clamp(q["k"] * sz_prev, min=1e-9)
        z_c = torch.maximum(q["h_r"], q["d"] + _expected_abs(q["c"] - q["d"], spread))
        u_j = speed(torch.maximum(torch.floor(z_c), q["h_r"]), q)
        x_j = x + u_j * TIME_STEP_S
        s_yj = torch.sqrt((q["sv"] * t_y) ** 2 / (1.0 + q["a"] * t_y)
                          + (q["sphi"] * (x_j - q["u0"] * q["ty"])) ** 2)
        p_zj = (2.0 * torch.exp(-0.5 * ((z_c - q["h_r"]) / s_z) ** 2)
                + torch.exp(-0.5 * ((2.0 * q["h_abl"] - z_c - q["h_r"]) / s_z) ** 2)
                ) / (_SQRT_2PI * s_z)
        if p_z is None:
            p_z = p_zj                               # the source node: P_z0 = P_z1
        # grid points k * step with x <= k * step < x_j
        k_lo = torch.ceil(x.detach() / TABLE_STEP_M).long()
        k_hi = torch.clamp(torch.ceil(x_j.detach() / TABLE_STEP_M).long() - 1, max=n_grid - 1)
        width = int((k_hi - k_lo).max()) + 1 if k_hi.numel() else 0
        if width > 0:
            k = k_lo[:, None] + torch.arange(width)
            r, c = torch.nonzero(k <= k_hi[:, None], as_tuple=True)
            k = k[r, c]
            w = (k * TABLE_STEP_M - x[r]) / (x_j[r] - x[r])
            index.append(rows[r] * n_grid + k)
            for out, lo, hi in zip(values, (s_y, p_z, u), (s_yj, p_zj, u_j), strict=True):
                out.append(lo[r] + (hi[r] - lo[r]) * w)
        live = x_j.detach() <= grid_end
        rows = rows[live]
        x, u, s_y, p_z, sz_prev = (v[live] for v in (x_j, u_j, s_yj, p_zj, s_z))
        q = {key: v[live] for key, v in q.items()}
    flat = torch.cat(index)
    return tuple(torch.zeros(m * n_grid, dtype=_DTYPE).index_put((flat,), torch.cat(v))
                 .reshape(m, n_grid) for v in values)


class StreetSources(NamedTuple):
    """Every street's roof flux as sub-sources (`source_points`)."""

    xy: Tensor
    """`(P, 2)`: sub-source coordinates, m."""
    owner: Tensor
    """`(P,)`: the street each sub-source belongs to."""
    weight: Tensor
    """`(P,)`: each sub-source's share of its street's flux, `1/n`."""
    width: Tensor
    """`batch + (P,)`: the flat-top width `(L/n)|sin phi|`, m."""
    clamp: Tensor
    """`batch + (P,)`: the cap on `P_z`, `min(10/W, 1/(H (1 - |sin phi|)))`, 1/m."""


def source_points(net: StreetNetwork, *, theta_w: Tensor | float) -> StreetSources:
    """Each street's roof flux as SIRANE's sub-sources: a street `u -> v` of coordinate
    length `L` is cut into `n = floor(L / SUBSOURCE_LENGTH_M) + 1` equal segments with one
    sub-source at the centre of each, weight `1/n`, width `(L/n)|sin phi|` and clamp
    `min(CLAMP_LENGTH_M / W, 1/(H (1 - |sin phi|)))`, where `phi` is the angle between the
    street and the wind `theta_w` (radians counter-clockwise from east, the direction the
    wind blows TOWARD; any batch shape) and `W`, `H` the street's width and height. The
    coordinate length is used, not `Street.length`, because the points lie on the drawn
    segment."""
    theta = _f64(theta_w)
    xy, owner, weight, sin_part, cos_part, seg, inv_w, height = ([] for _ in range(8))
    for i, s in enumerate(net.streets):
        x0, y0, x1, y1 = net.x[s.u], net.y[s.u], net.x[s.v], net.y[s.v]
        length = math.hypot(x1 - x0, y1 - y0)
        if not length > 0:
            raise ValueError(f"source_points: street {s.name!r} has zero coordinate length")
        n = int(math.floor(length / SUBSOURCE_LENGTH_M + 1e-9)) + 1
        f = (torch.arange(n, dtype=_DTYPE) + 0.5) / n
        xy.append(torch.stack([x0 + f * (x1 - x0), y0 + f * (y1 - y0)], dim=-1))
        owner.append(torch.full((n,), i, dtype=torch.long))
        weight.append(torch.full((n,), 1.0 / n, dtype=_DTYPE))
        for values, value in ((sin_part, (y1 - y0) / length), (cos_part, (x1 - x0) / length),
                              (seg, length / n), (inv_w, CLAMP_LENGTH_M / s.width),
                              (height, s.height)):
            values.append(torch.full((n,), value, dtype=_DTYPE))
    sin_a, cos_a, seg_l, cap_w, h = (torch.cat(v) for v in (sin_part, cos_part, seg, inv_w,
                                                            height))
    th = theta.unsqueeze(-1)
    sin_phi = (sin_a * torch.cos(th) - cos_a * torch.sin(th)).abs()
    width = seg_l * sin_phi
    clamp = torch.minimum(cap_w.expand_as(sin_phi),
                          1.0 / (h * torch.clamp(1.0 - sin_phi, min=1e-12)))
    return StreetSources(torch.cat(xy), torch.cat(owner), torch.cat(weight), width, clamp)


def street_midpoints(net: StreetNetwork) -> Tensor:
    """`(n_streets, 2)`: the receptor of each street's `C_ext`, the segment midpoint."""
    return torch.tensor(
        [[(net.x[s.u] + net.x[s.v]) / 2.0, (net.y[s.u] + net.y[s.v]) / 2.0]
         for s in net.streets], dtype=_DTYPE,
    )


def junction_points(net: StreetNetwork) -> Tensor:
    """`(n_junctions, 2)`: every junction's coordinates, in `StreetNetwork.junctions` order
    -- the junction index `build_model` writes onto the `vent` edges."""
    return torch.tensor([[net.x[j], net.y[j]] for j in net.junctions], dtype=_DTYPE)


def junction_sources(net: StreetNetwork) -> tuple[Tensor, Tensor, Tensor]:
    """`(xy (J, 2), width (J,), clamp (J,))`: every junction as one point source of width
    `W_j` and clamp `1/H_j`, with `W_j`, `H_j` the mean width and height of the streets
    that meet at the junction. The mean is a choice: on SIRANE's South Kensington fluxes,
    at the deck's downwind cut-off, it gives SIRANE's `C_ext` to a median 0.32 % and p90
    1.38 %, against 0.35 % and 1.38 % with the largest width and height, and 0.53 % and
    1.48 % with the smallest."""
    index = {j: k for k, j in enumerate(net.junctions)}
    widths: list[list[float]] = [[] for _ in index]
    heights: list[list[float]] = [[] for _ in index]
    for s in net.streets:
        for end in (s.u, s.v):
            widths[index[end]].append(s.width)
            heights[index[end]].append(s.height)
    width = torch.tensor([sum(w) / len(w) for w in widths], dtype=_DTYPE)
    height = torch.tensor([sum(h) / len(h) for h in heights], dtype=_DTYPE)
    return junction_points(net), width, 1.0 / height


def _pair_values(receptor_xy: Tensor, source_xy: Tensor, rows: dict, cos: Tensor,
                 sin: Tensor, width: Tensor, clamp: Tensor, cutoff_sigma: float,
                 step: float, x_cut: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The kernel for receptors `(r, 2)` x sources `(p, 2)`, evaluated ONLY on the pairs
    that survive the cut-offs: returns `(b, i, p, c)`, the flat batch, receptor and source
    index of each kept pair and its value. Every other pair is exactly zero. `rows` holds
    the table's `(n_b, n_grid)` tensors; `cos`, `sin` and the downwind cut-off `x_cut` are
    `(n_b,)`; `width`, `clamp` are `(n_b, p)`."""
    dx = receptor_xy[:, None, 0] - source_xy[None, :, 0]
    dy = receptor_xy[:, None, 1] - source_xy[None, :, 1]
    x_all = dx * cos[:, None, None] + dy * sin[:, None, None]
    b, i, p = torch.nonzero((x_all > 0) & (x_all <= x_cut[:, None, None]), as_tuple=True)
    x = x_all[b, i, p]
    del x_all
    y = dy[i, p] * cos[b] - dx[i, p] * sin[b]
    s_y, p_z, u = _lookup(rows, x, b, step)
    y_c = torch.clamp((width[b, p] - _SQRT_2PI * s_y) / 2.0, min=0.0)
    off = y.abs() - y_c
    keep = off <= cutoff_sigma * s_y
    b, i, p, off, s_y, p_z, u, y_c = (v[keep] for v in (b, i, p, off, s_y, p_z, u, y_c))
    tail = torch.exp(-0.5 * (torch.clamp(off, min=0.0) / s_y) ** 2)
    f_y = torch.where(off <= 0, torch.ones_like(tail), tail) / (2.0 * y_c + _SQRT_2PI * s_y)
    c = f_y * torch.minimum(p_z, clamp[b, p]) / u
    return b, i, p, c


def rows_per_chunk(n_sources: int, batch_numel: int, *, backward: bool = False) -> int:
    """Receptor rows per chunk so that one chunk's evaluation stays under
    `KERNEL_BYTES_CAP` (`_LIVE_TEMPORARIES`, or `_BACKWARD_TEMPORARIES` when a gradient is
    recorded, float64 blocks of `batch x rows x sources`)."""
    blocks = _BACKWARD_TEMPORARIES if backward else _LIVE_TEMPORARIES
    per_row = 8 * blocks * max(1, n_sources) * max(1, batch_numel)
    return max(1, KERNEL_BYTES_CAP // per_row)


def _chunked(n_receptors: int, n_sources: int, batch_numel: int,
             evaluate: Callable[[int, int], object], backward: bool = False) -> list:
    rows = rows_per_chunk(n_sources, batch_numel, backward=backward)
    return [evaluate(a, min(n_receptors, a + rows)) for a in range(0, n_receptors, rows)]


def downwind_cutoff(theta_w: Tensor | float, meteo_cell_dx: float) -> Tensor:
    """SIRANE's downwind cut-off, m: `meteo_cell_dx / |cos theta_w|`, the x-size of the
    meteo-grid cell over the cosine of the wind's angle to the x axis (an empirical rule:
    900 m cells give 910 m, 700 m cells 700 m whatever the cells' y-size, and a wind at 45
    degrees to the axis ~1020 m; `|cos|` is floored at 1e-6). It holds for cells of about
    700 m and larger: with smaller cells SIRANE stops the Gaussian plume much earlier
    (about 168 m for 350 m cells) and continues with a cell-based far field, which this
    kernel does not model."""
    if not meteo_cell_dx > 0:
        raise ValueError(f"downwind_cutoff: meteo_cell_dx must be positive, got "
                         f"{meteo_cell_dx!r}")
    theta = _f64(theta_w).detach()
    return meteo_cell_dx / torch.clamp(torch.cos(theta).abs(), min=1e-6)


def _setup(table: PlumeTable, theta_w: Tensor | float, width: Tensor | float,
           clamp: Tensor | float, n_src: int, meteo_cell_dx: float | None,
           where: str) -> tuple[tuple[int, ...], dict]:
    """The common batch of the table, the wind direction and the per-source width and
    clamp, and the flat `(n_b, ...)` tensors `_pair_values` takes. The cut-off is
    `downwind_cutoff` when `meteo_cell_dx` is given (refused beyond the table's extent),
    `table.x_max` otherwise."""
    theta = _f64(theta_w)
    width, clamp = _f64(width), _f64(clamp)
    src_shapes = [v.shape[:-1] if v.dim() else () for v in (width, clamp)]
    batch = tuple(broadcast_shapes(table.batch, theta.shape, *src_shapes))
    n_b = max(1, math.prod(batch))
    n_grid = table.u.shape[-1]
    flat = {key: getattr(table, key).reshape(*table.batch, n_grid).expand(*batch, n_grid)
            .reshape(n_b, n_grid) for key in _TABLE_KEYS}
    flat["cos"] = torch.cos(theta).expand(batch).reshape(n_b)
    flat["sin"] = torch.sin(theta).expand(batch).reshape(n_b)
    flat["width"] = width.expand(*batch, n_src).reshape(n_b, n_src)
    flat["clamp"] = clamp.expand(*batch, n_src).reshape(n_b, n_src)
    if meteo_cell_dx is None:
        x_cut = torch.full((n_b,), table.x_max, dtype=_DTYPE)
    else:
        x_cut = downwind_cutoff(theta, meteo_cell_dx).expand(batch).reshape(n_b)
        if float(x_cut.max()) > table.x_max + 1e-9:
            raise ValueError(
                f"{where}: the downwind cut-off meteo_cell_dx / |cos theta_w| reaches "
                f"{float(x_cut.max()):.1f} m, beyond the table's x_max = {table.x_max} m; "
                f"build the table with plume_table(..., x_max=float(downwind_cutoff("
                f"theta_w, meteo_cell_dx).max()))"
            )
    flat["x_cut"] = x_cut
    return batch, flat


def _evaluate(receptor_xy: Tensor, source_xy: Tensor, flat: dict, table: PlumeTable,
              cutoff_sigma: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _pair_values(receptor_xy, source_xy, {k: flat[k] for k in _TABLE_KEYS},
                        flat["cos"], flat["sin"], flat["width"], flat["clamp"],
                        cutoff_sigma, table.step, flat["x_cut"])


def plume_kernel(
    receptor_xy: Tensor,
    source_xy: Tensor,
    table: PlumeTable,
    *,
    theta_w: Tensor | float,
    width: Tensor | float = 0.0,
    clamp: Tensor | float = math.inf,
    cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS,
    meteo_cell_dx: float | None = None,
) -> Tensor:
    """Concentration at each receptor per unit rate at each source, s/m3: `batch + (R, P)`.

    `C = f_y(y; sigma_y(x), w) min(P_z(x), clamp) / U(x)` read from `table` (module
    docstring), with `x`, `y` the downwind and crosswind offsets of the receptor from the
    source for the wind `theta_w` (radians counter-clockwise from east, the direction the
    wind blows TOWARD -- `StreetFlows`' driver of that name). `width` (m) and `clamp`
    (1/m) are each source's flat-top width and `P_z` cap, scalars or `(P,)` or
    `batch + (P,)`; the defaults are a point source without a cap. The batch is the
    broadcast of the table's, `theta_w`'s and the sources'. Upwind pairs (`x <= 0`), pairs
    beyond the downwind cut-off and pairs more than `cutoff_sigma` standard deviations
    beyond the flat top are exactly zero. The cut-off is `downwind_cutoff(theta_w,
    meteo_cell_dx)` when the x-size of SIRANE's meteo-grid cell is given (SIRANE's rule;
    the table must reach it), and `table.x_max` (by default `DOWNWIND_CUTOFF_M`, the
    cut-off of 700 m cells along the wind) when it is not.

    The full result is materialised chunk by chunk (the peak is the result plus one
    chunk's temporaries) and refused above `KERNEL_BYTES_CAP`; use `street_kernel` for a
    network, which reduces over sources chunk by chunk. When a gradient is being recorded
    each chunk is checkpointed, as in `street_kernel`.
    """
    receptor_xy, source_xy = _f64(receptor_xy), _f64(source_xy)
    n_rec, n_src = receptor_xy.shape[0], source_xy.shape[0]
    batch, flat = _setup(table, theta_w, width, clamp, n_src, meteo_cell_dx, "plume_kernel")
    n_b = max(1, math.prod(batch))
    size = 8 * n_rec * n_src * n_b
    if size > KERNEL_BYTES_CAP:
        raise ValueError(
            f"plume_kernel: the {n_rec} x {n_src} kernel over batch {batch} is "
            f"{size / 2**20:.1f} MB, above the {KERNEL_BYTES_CAP / 2**20:.0f} MB cap; use "
            f"street_kernel, which sums over each street's points chunk by chunk"
        )
    keys = list(flat)
    grad = torch.is_grad_enabled() and any(v.requires_grad for v in flat.values())

    def block(rec: Tensor, *values: Tensor) -> Tensor:
        b, i, p, c = _evaluate(rec, source_xy, dict(zip(keys, values, strict=True)), table,
                               cutoff_sigma)
        return torch.zeros(n_b, rec.shape[0], n_src, dtype=_DTYPE).index_put((b, i, p), c)

    def evaluate(a: int, b_: int) -> Tensor:
        values = [flat[k] for k in keys]
        if grad:
            return checkpoint(block, receptor_xy[a:b_], *values, use_reentrant=False)
        return block(receptor_xy[a:b_], *values)

    out = torch.cat(_chunked(n_rec, n_src, n_b, evaluate, backward=grad), dim=-2)
    return out.reshape(*batch, n_rec, n_src)


def street_kernel(
    net: StreetNetwork,
    table: PlumeTable,
    *,
    theta_w: Tensor | float,
    self_contribution: bool = False,
    cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS,
    meteo_cell_dx: float | None = None,
) -> Tensor:
    """`batch + (n_streets, n_streets)`: `K[..., i, j]` is `C_ext` above street `i` (at its
    midpoint) per unit roof-flux rate of street `j`, s/m3.

    Street `j`'s flux is spread over its `source_points` with their weights, widths and
    clamps; `table`, `theta_w`, `cutoff_sigma` and `meteo_cell_dx` are `plume_kernel`'s.
    Evaluated in receptor chunks under `KERNEL_BYTES_CAP`, each reduced to streets before
    the next; when a gradient is being recorded each chunk is checkpointed, so the backward
    pass recomputes it rather than holding every chunk's temporaries.

    `self_contribution=False` (the default, SIRANE's) excludes a street's own sub-sources
    from its own receptor, so `K[..., i, i] == 0`: `C_ext` sums the contributions of the
    other streets and the junctions. `True` keeps the own upstream sub-sources.
    """
    src = source_points(net, theta_w=theta_w)
    receptors = street_midpoints(net)
    n = len(net.streets)
    batch, flat = _setup(table, theta_w, src.width, src.clamp, src.xy.shape[0],
                         meteo_cell_dx, "street_kernel")
    n_b = max(1, math.prod(batch))
    keys = list(flat)
    grad = torch.is_grad_enabled() and any(v.requires_grad for v in flat.values())

    def reduce(rec: Tensor, first: int, *values: Tensor) -> Tensor:
        b, i, p, c = _evaluate(rec, src.xy, dict(zip(keys, values, strict=True)), table,
                               cutoff_sigma)
        if not self_contribution:
            other = src.owner[p] != i + first
            b, i, p, c = b[other], i[other], p[other], c[other]
        r = rec.shape[0]
        index = (b * r + i) * n + src.owner[p]
        out = torch.zeros(n_b * r * n, dtype=_DTYPE).index_add(0, index, c * src.weight[p])
        return out.reshape(n_b, r, n)

    def evaluate(a: int, b_: int) -> Tensor:
        values = [flat[k] for k in keys]
        if grad:
            return checkpoint(reduce, receptors[a:b_], a, *values, use_reentrant=False)
        return reduce(receptors[a:b_], a, *values)

    out = torch.cat(_chunked(n, src.xy.shape[0], n_b, evaluate, backward=grad), dim=-2)
    return out.reshape(*batch, n, n)


def junction_kernel(
    net: StreetNetwork,
    table: PlumeTable,
    *,
    theta_w: Tensor | float,
    cutoff_sigma: float = GAUSS_CUTOFF_SIGMAS,
    meteo_cell_dx: float | None = None,
) -> Tensor:
    """`batch + (n_streets, n_junctions)`: `C_ext` above street `i` (at its midpoint) per
    unit vertical-flux rate out of junction `j`, s/m3 -- SIRANE's intersection plume
    (Soulhac et al. 2011, Sect. 5.2: `C_ext` sums the contributions of upwind streets AND
    intersections), each junction one point source of `junction_sources`' width and clamp.
    `table`, `theta_w`, `cutoff_sigma` and `meteo_cell_dx` are `plume_kernel`'s.

    Nothing is excluded: a street's own end junctions contribute to its `C_ext` when they
    are upwind of its midpoint. The kernel is materialised whole, so it is refused above
    `KERNEL_BYTES_CAP` like `plume_kernel` (577 streets by a few hundred junctions is about
    2 MB per hour).
    """
    xy, width, clamp = junction_sources(net)
    return plume_kernel(street_midpoints(net), xy, table, theta_w=theta_w, width=width,
                        clamp=clamp, cutoff_sigma=cutoff_sigma,
                        meteo_cell_dx=meteo_cell_dx)

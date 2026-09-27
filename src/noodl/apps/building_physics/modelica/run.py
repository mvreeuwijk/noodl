"""Run a model `read_modelica` built over its driver grid.

`simulate(model, state, drivers, times)` steps the model from `times[0]` over every later
driver-grid time (`assemble.driver_grid`: the output grid, optionally split into substeps,
plus the signals' event times) and returns, at `times`, the time histories MBL's reference
CSV holds: every air-layer edge flow, every node's absolute pressure, temperature, water mass
fraction and trace-substance mass fractions. Time-varying drivers are taken at the step's END
time, except sources, which are the MEAN of their value over the step (`assemble.py`,
"Sources"). Two schemes: `"implicit"` (`Model.step`, flows at the step's end, first order)
and `"midpoint"` (`_Midpoint`, symmetric, second order; `extrapolate` combines two of its
runs into a fourth-order one). A model without zones (no transport layer) is algebraic in its
boundary values and is solved with `model.steady` at every time asked for instead. Row 0 is
the initial state with its quasi-steady flows.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.apps.building_physics.modelica.assemble import _MBLClosure
from noodl.model import Drivers, Model, State

Tensor = torch.Tensor
F64 = torch.float64
_SERIES = "series:"
# Newton tolerances for the airflow solves (kg/s). The layer default, sqrt(eps) = 1.5e-8 in
# absolute and relative terms, is 1e-4 of a small crack flow; the reference simulation's
# declared tolerance is 1e-6 relative, so the airflow is solved to round-off instead.
# Newton converges quadratically, so this costs about one more iteration.
AIR_ATOL = 1e-13
AIR_RTOL = 1e-12


def step_drivers(drivers: Mapping[str, Tensor], grid: Tensor, t: float) -> Drivers:
    """`drivers` at grid time `t`: every `"series:<key>"` entry sliced into `<key>`, and the
    series entries themselves dropped. Raises `ValueError` if `t` is not on the grid."""
    out: Drivers = {k: v for k, v in drivers.items() if not k.startswith(_SERIES)}
    series = {k[len(_SERIES):]: v for k, v in drivers.items()
              if k.startswith(_SERIES) and k != "series:time"}
    if not series:
        return out
    grid = torch.as_tensor(grid, dtype=F64)
    match = torch.isclose(grid, torch.tensor(float(t), dtype=F64), rtol=0.0,
                          atol=1e-9 * max(1.0, abs(float(t))))
    if not bool(match.any()):
        raise ValueError(
            f"simulate: time {t!r} is not on the experiment grid the driver series were "
            f"evaluated on ({float(grid[0])} to {float(grid[-1])}, {grid.numel()} points)"
        )
    k = int(match.nonzero()[0])
    for key, value in series.items():
        out[key] = value[k]
    return out


def _grid_index(grid: Tensor, times: Tensor) -> list[int]:
    """The position of every entry of `times` on the driver grid; raises `ValueError` for a
    time off the grid or a non-increasing one."""
    tol = 1e-9 * max(1.0, float(grid.abs().max()))
    diff = (times.unsqueeze(1) - grid.unsqueeze(0)).abs()
    idx = diff.argmin(dim=1)
    off = diff.gather(1, idx.unsqueeze(1)).squeeze(1) > tol
    if bool(off.any()):
        k = int(torch.nonzero(off)[0])
        raise ValueError(
            f"simulate: times[{k}] = {float(times[k])!r} is not a grid time of the drivers "
            f"({float(grid[0])} to {float(grid[-1])}, {grid.numel()} points)"
        )
    out = idx.tolist()
    if any(b <= a for a, b in zip(out, out[1:], strict=False)):
        raise ValueError("simulate: times must be increasing")
    return out


def _closure(model: Model) -> _MBLClosure:
    for c in model.closures:
        if isinstance(c, _MBLClosure):
            return c
    raise TypeError("simulate: the model was not built by read_modelica (no MBL closure)")


def _solve_air(model: Model, state: State, drivers: Drivers, **solve_kwargs) -> State:
    """The potential layers alone at `state` (closures first): the quasi-steady flows of the
    initial state, without advancing any transport layer.

    Newton starts from the layer's own linear initial guess (`phi0=None`,
    `PotentialFlowLayer.linear_init`), not from the state's `p_start` seed: every zone at
    `p_start` is far from a stack's hydrostatic solution, and from there Newton cycles on
    MBL's discretised doors (`Validation/ThreeRoomsContamDiscretizedDoor.mo`: residual
    0.084 kg/s after 50 iterations), while the linear guess converges in three."""
    drv = dict(drivers)
    for c in model.closures:
        drv.update(c(state, drv))
    new = dict(state)
    for name, layer in model.potential.items():
        phi, q = layer.solve(drv[f"{name}.phi_boundary"], drv, drv.get(f"{name}.sources"),
                             phi0=None, **solve_kwargs)
        new[f"{name}.phi"], new[f"{name}.q"] = phi, q
    return new


def simulate(model: Model, state: State, drivers: Drivers, times, *,
             scheme: str = "implicit", **step_kwargs) -> dict[str, Tensor]:
    """Time histories at `times` (increasing times of the driver grid `"series:time"`).

    Returns `"time"` `(N,)`, `"air.q"` `(N, b)` in the air layer's edge order (see
    `ModelicaNames.edges`), `"air.phi"` `(N, n)` gauge pressure (relative to
    `ModelicaNames.p_ref`), `"p"` `(N, n)` absolute
    pressure (Pa), `"T"` `(N, n)` (K), `"X_w"` `(N, n)`, and `"C"` `(N, n, K)` for the species
    layer's mass fractions (water last when carried) when the model has one. `step_kwargs`
    reach `Model.step`/`Model.steady` (and so the airflow solves); the airflow Newton
    tolerances default to `atol=AIR_ATOL`, `rtol=AIR_RTOL`.

    The model is stepped over EVERY driver-grid time from `times[0]` to `times[-1]` (the
    grid `read_modelica(..., substeps=r)` builds: `r` steps per output interval plus the
    signals' event times), and the rows at `times` are returned. Each source driver is its
    mean over the one grid interval ending at its row (`assemble` module docstring,
    "Sources"), so the injected amounts are exact.

    `scheme`: `"implicit"` (the default) is `Model.step` with `coupling="iterate"`: the
    transport layers advance with the flows held at their END-of-step values, first order
    in the step. `"midpoint"` (`_Midpoint`) advances them with the MEAN of the start- and
    end-of-step flows, boundary values and state-dependent heat sources: symmetric and
    second order in the step, so two runs at `r` and `2 r` substeps combine by `extrapolate`
    to fourth order. It is forward-only (no gradient) and for unbatched models.
    """
    if scheme not in ("implicit", "midpoint"):
        raise ValueError(f"simulate: unknown scheme {scheme!r}; 'implicit' or 'midpoint'")
    step_kwargs = {"atol": AIR_ATOL, "rtol": AIR_RTOL, **step_kwargs}
    times = torch.as_tensor(times, dtype=F64)
    grid = torch.as_tensor(drivers.get("series:time", times), dtype=F64)
    idx = _grid_index(grid, times)
    closure = _closure(model)
    dynamic = bool(model.transport)
    rows: dict[str, list[Tensor]] = {"air.q": [], "air.phi": [], "p": [], "T": [], "X_w": []}
    if closure.sp_interior is not None:
        rows["C"] = []

    def record(state: State, d: Drivers) -> None:
        extra = closure(state, d)
        rows["air.q"].append(state["air.q"])
        rows["air.phi"].append(state["air.phi"])
        rows["p"].append(closure.pressures(state, d))
        rows["T"].append(extra["T"])
        rows["X_w"].append(extra["X_w"] if "X_w" in extra else d["X_w"])
        if "C" in rows:
            rows["C"].append(closure.species(state, d))

    if not dynamic:  # algebraic: the steady solve at each time asked for
        for k in idx:
            d = step_drivers(drivers, grid, float(grid[k]))
            state = model.steady(state, d, **step_kwargs)
            record(state, d)
    else:
        d = step_drivers(drivers, grid, float(grid[idx[0]]))
        state = _solve_air(model, state, d, **step_kwargs)
        record(state, d)
        mid = _Midpoint(model, closure, step_kwargs) if scheme == "midpoint" else None
        for a, b in zip(idx, idx[1:], strict=False):
            for j in range(a + 1, b + 1):
                t0, t1 = float(grid[j - 1]), float(grid[j])
                d1 = step_drivers(drivers, grid, t1)
                if mid is None:
                    state = model.step(state, d1, t1 - t0, t=t0, **step_kwargs)
                else:
                    state = mid.step(state, step_drivers(drivers, grid, t0), d1, t1 - t0)
                d = d1
            record(state, d)
    out = {key: torch.stack(v) for key, v in rows.items()}
    out["time"] = times.clone()
    return out


def extrapolate(coarse: Mapping[str, Tensor], fine: Mapping[str, Tensor],
                order: int = 2) -> dict[str, Tensor]:
    """Richardson extrapolation of two `simulate` histories on the same `times`, `fine`
    with half the step of `coarse`, of a method of order `order`:
    `fine + (fine - coarse)/(2**order - 1)`. For `scheme="midpoint"` (`order=2`, symmetric,
    so its error expands in even powers of the step) the result is fourth order."""
    if not torch.equal(coarse["time"], fine["time"]):
        raise ValueError("extrapolate: the two histories are not on the same times")
    f = 1.0 / (2 ** order - 1)
    return {k: v if k == "time" else v + (v - coarse[k]) * f for k, v in fine.items()}


# Convergence of `_Midpoint`'s step iteration, per unknown: |change| <= ATOL + RTOL |value|.
MIDPOINT_ATOL = {"thermal": 1e-10, "species": 1e-16, "air": 1e-9}  # K, kg/kg, Pa
MIDPOINT_RTOL = 1e-13
MIDPOINT_MAX_ITER = 100
_ANDERSON_DEPTH = 5


class _Midpoint:
    """The symmetric step of `simulate(scheme="midpoint")`.

    Over a step of length `h` from state 0 to state 1, each transport layer advances with
    its own exact scheme (`TransportLayer.step`, the flows held constant over the step) at
    the MEAN flows `(q0 + q1)/2`, the mean boundary values `(x_b0 + x_b1)/2` and its step-mean
    source driver plus the mean of the closure's state-dependent sources (the moist-air heat
    carrier, `_MBLClosure.cp_correction`). State 1 is the fixed point of that map together
    with the airflow solve at state 1 (and the interior pressures `p_abs`, which the closure
    feeds back): an Anderson-accelerated fixed-point iteration (depth `_ANDERSON_DEPTH`) on
    the transport states and the air layer's interior potentials, started from the linear
    extrapolation of the previous step and converged to `MIDPOINT_ATOL`/`MIDPOINT_RTOL`.
    The step map is symmetric in states 0 and 1, so the scheme is second order and its
    global error expands in even powers of the step (`extrapolate`)."""

    def __init__(self, model: Model, closure: _MBLClosure, solve_kwargs: dict) -> None:
        if len(model.potential) != 1:
            raise ValueError("simulate(scheme='midpoint'): one potential layer expected")
        ((self.air_name, self.air),) = model.potential.items()
        self.layers = dict(model.transport)
        self.closure = closure
        self.kw = dict(solve_kwargs, differentiable=False)
        self.last: tuple[float, Tensor, Tensor] | None = None  # (h, z at start, z at end)

    # z = [each transport layer's x, flattened; the air layer's interior potentials]
    def _pack(self, state: State) -> Tensor:
        parts = [state[f"{n}.x"].reshape(-1) for n in self.layers]
        parts.append(state[f"{self.air_name}.phi"][..., self.air.interior].reshape(-1))
        return torch.cat(parts)

    def _unpack(self, z: Tensor, like: State) -> State:
        out, i = dict(like), 0
        for n in self.layers:
            x = like[f"{n}.x"]
            out[f"{n}.x"] = z[i:i + x.numel()].reshape(x.shape)
            i += x.numel()
        phi = like[f"{self.air_name}.phi"].clone()
        phi[..., self.air.interior] = z[i:]
        out[f"{self.air_name}.phi"] = phi
        return out

    def _tolerance(self, z: Tensor, like: State) -> Tensor:
        parts = [torch.full((like[f"{n}.x"].numel(),), MIDPOINT_ATOL.get(n, 1e-12), dtype=F64)
                 for n in self.layers]
        parts.append(torch.full((self.air.interior.numel(),), MIDPOINT_ATOL["air"],
                                dtype=F64))
        return torch.cat(parts) + MIDPOINT_RTOL * z.abs()

    def _extra_sources(self, state: State, d: Drivers) -> dict[str, Tensor]:
        """The closure's state-dependent sources: what it adds to `"<layer>.sources"`."""
        written = self.closure(state, d)
        return {n: written[f"{n}.sources"] - d.get(f"{n}.sources", 0.0)
                for n in self.layers if f"{n}.sources" in written}

    def _zero_sources(self, n: str, x: Tensor) -> Tensor:
        layer = self.layers[n]
        n_nodes = len(self.closure.T0)
        shape = (n_nodes,) if x.ndim == 1 else (n_nodes, layer.n_species)
        return torch.zeros(shape, dtype=F64)

    def step(self, s0: State, d0: Drivers, d1: Drivers, h: float) -> State:
        air, name = self.air, self.air_name
        q0 = s0[f"{name}.q"]
        extra0 = self._extra_sources(s0, d0)
        z0 = self._pack(s0)

        def G(z: Tensor) -> tuple[Tensor, State]:
            s = self._unpack(z, s0)
            drv = dict(d1)
            drv.update(self.closure(s, drv))
            phi, q = air.solve(drv[f"{name}.phi_boundary"], drv, drv.get(f"{name}.sources"),
                               phi0=s[f"{name}.phi"][..., air.interior], **self.kw)
            s[f"{name}.phi"], s[f"{name}.q"] = phi, q
            extra1 = self._extra_sources(s, d1)
            qm = 0.5 * (q0 + q)
            for n, layer in self.layers.items():
                x0 = s0[f"{n}.x"]
                src = d1.get(f"{n}.sources")
                if src is None:
                    src = self._zero_sources(n, x0)
                if n in extra0 or n in extra1:
                    src = src + 0.5 * (extra0.get(n, 0.0) + extra1.get(n, 0.0))
                xb = 0.5 * (d0[f"{n}.x_boundary"] + d1[f"{n}.x_boundary"])
                s[f"{n}.x"] = layer.step(x0, air.flows_of_kind(qm, layer.flow_kinds), src, xb, h)
            return self._pack(s), s

        z = z0.clone()
        if self.last is not None and torch.equal(self.last[2], z0):
            h_prev, z_prev, _ = self.last  # linear extrapolation of the previous step
            z = z0 + (z0 - z_prev) * (h / h_prev)
        dF: list[Tensor] = []
        dG: list[Tensor] = []
        f_old = g_old = None
        for _ in range(MIDPOINT_MAX_ITER):
            g, s1 = G(z)
            f = g - z
            tol = self._tolerance(g, s0)
            if bool((f.abs() <= tol).all()):
                self.last = (h, z0, self._pack(s1))
                return s1
            w = 1.0 / tol  # weights of the least-squares mixing
            if f_old is not None:
                dF = [*dF, (f - f_old) * w][-_ANDERSON_DEPTH:]
                dG = [*dG, g - g_old][-_ANDERSON_DEPTH:]
            f_old, g_old = f, g
            if dF:
                A = torch.stack(dF, dim=1)
                gamma = torch.linalg.lstsq(A, (f * w).unsqueeze(1)).solution.squeeze(1)
                z = g - torch.stack(dG, dim=1) @ gamma
            else:
                z = g
        raise RuntimeError(
            f"simulate(scheme='midpoint'): a step of {h!r} s did not converge in "
            f"{MIDPOINT_MAX_ITER} iterations (max |change|/tolerance "
            f"{float((f.abs() / tol).max()):.3g})"
        )

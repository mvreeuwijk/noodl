"""The coupled street / above-roof steady state: SIRANE's `C_ext` from upwind plumes.

With `build_model(..., background="per_street")` every street has its own atmosphere node,
so its above-roof concentration `C_ext,i` is simply its entry of `"<layer>.x_boundary"`.
This module makes that value an unknown instead of a driver: `C_ext` is the background plus
SIRANE's street plumes (`noodl.apps.street_aq.plume`) of every upwind street's roof flux and
every junction's vertical flux (Soulhac et al. 2011, Atmos. Environ. 45:7379, Sect. 5.2 and
7, Eqs. 6 and 37),

    C_ext = C_bg + K_s F_s + K_j F_j,

and solves the street network and that relation together by fixed-point iteration on the
per-street background, not through a `build_model` option.

The fluxes are noodl physics' own flows, read off the model's closure output
(`Model.current_flows`), so they conserve mass exactly with the street solve:

- Roof flux of street `i` (Eq. 6/37): `F_s,i = u_d,i W_i L_i (C_i - C_ext,i)`, the two
  `exchange` edges of street `i` (they carry the same flow `u_d W L`, one each way). It is
  SIGNED: a street cleaner than the air above it imports (`F < 0`), and the negative source
  enters the plume sum as it is -- the map stays linear, nothing is clipped.
- Vertical flux of junction `j`: the EXCESS OVER BACKGROUND carried from the canopy to the
  atmosphere through that junction's OUTGOING `vent` edges,
  `F_up,j = sum_out q (C_street - C_bg,street)` (each carries the concentration of the
  street it leaves, less that street's background), a point source at the junction
  (SIRANE's intersection plume, `plume.junction_kernel`).

Why excess over background. `C_ext = C_bg + K F` must be NEUTRAL to the background: with
no emission anywhere and a uniform `C_bg`, every street sits at `C_bg` and `C_ext` must
stay at `C_bg`. The roof flux is neutral by construction (`C - C_ext = 0`). A junction
source of `sum_out q C_street` is not: it re-emits background air that the plume sum then
adds on top of the background it already contains -- measured on a 7 x 7 lattice
(L = 100 m, W = H = 20 m, u* = 0.5 m/s, 0.3 rad) as `max C_ext / C_bg = 1.095` with
nothing emitted. The background is already in `C_bg`; only what the canopy adds to it may
enter the plume sum. The subtraction uses the background of the street the air leaves,
which is what that street would hold with no emissions under a uniform background.

- The downward flow through the INCOMING vents carries the `C_ext` of the street it enters
  -- exactly what the per-street background model already transports -- and by default is
  NOT subtracted from the plume sum (`junction_source="upward"`, which therefore has no
  direct `C_ext` self-sink). The reason is measured, not aesthetic: noodl physics has no
  above-intersection concentration (SIRANE's `C_ext-int,j`, Eq. 8), so the air drawn down
  at junction `j` into street `i` is taken at `C_ext,i`, and a sink of that air at the
  junction, half a street upwind of street `i`'s own midpoint, feeds straight back into
  `C_ext,i`: `G[i, i] = -K_j[i, j] q_in`, a loop that exists only because `C_ext-int` is
  missing. On `munich_idealised` (L = 100 m, W = H = 20 m, u* = 0.5 m/s) the spectral
  radius of the coupling is 0.20 at a 0.3 rad wind and 0.31 at 0 rad (the junction
  plume's vertical profile is capped at `1/H`, which keeps that loop gain below one),
  while without the sink it is exactly 0 in both.
  `junction_source="net"` keeps the sink, again as an excess over background,
  `F_j = F_up - F_down` with `F_down,j = sum_in q (C_ext,street - C_bg,street)` (the
  street the air enters), for comparison -- on SIRANE's archived `C_ext` the two modes are
  close, median error 0.32 % (upward) and 0.29 % (net) at the deck's downwind cut-off
  (`tests/verification/test_sirane.py`). The street roof flux keeps its sign in both
  modes: its sink term never reaches its own street
  (`street_kernel(self_contribution=False)`).

Canopy mass balance: with a UNIFORM `C_bg` the network's vent and exchange flows to and
from the atmosphere balance, so the background terms cancel over the network and the
emissions leave as `sum Q_S = sum F_s + sum F_up - sum F_down` (both junction terms are
returned so that a caller can check it). With a non-uniform `C_bg` the background terms
do not cancel and that sum differs from the emissions by the background carried between
streets.

The iteration. Start from `C_ext = C_bg`; solve the street network with
`x_boundary = C_ext`; recompute `C_ext` from the fluxes; repeat until the largest change,
relative to the largest `|C_ext|`, is at most `tol` in every instance. For fixed
meteorology the flows are fixed and every step is linear, so the map is affine,
`C_ext <- a + G C_ext`, and the error contracts by the spectral radius of `G` per pass;
`diagnostics["contraction"]` reports the measured ratio of the last two changes. With the
upward junction source every plume link points downwind (upwind pairs are zero, a
street's own points are excluded), so `G` is nilpotent whenever that dependency graph has
no cycle -- measured spectral radius exactly 0 on `munich_idealised` -- and the iteration
ends after as many passes as the network has downwind levels; streets that see each
other's points both ways, and canyon routing between them, give a small non-zero
radius. `relaxation < 1`
under-relaxes the update, `C_ext <- C_ext + relaxation (image - C_ext)`: it does not move
the fixed point, and it makes a coupling whose eigenvalues are real and negative (a sink
feeding back on its source, as `junction_source="net"` does) contract once
`relaxation < 2 / (1 - lambda_min)`; it cannot help an eigenvalue at or above +1. A map
that does not converge within `max_iter` passes raises, by name.

Differentiability: the gradient is that of the CONVERGED fixed point, by the implicit
adjoint `noodl.solvers.fixed_point.differentiate_fixed_point` -- one differentiable pass at
the converged `C_ext`, whose backward solves `(I - G^T) v = ...` by GMRES. The forward
iterations run without a graph, so memory is one pass whatever the pass count, and the
derivative does not depend on where the iteration started or how many passes it took.
Instances (hours) never couple, so the adjoint solves one small system per instance.

Chemistry: not handled. `Model.steady` applies no reaction (see `chemistry.street_steady`),
and a photostationary relaxation would make the map nonlinear; combining the two outer
iterations is left for when it is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from noodl.model import Drivers, Model, State
from noodl.solvers.fixed_point import differentiate_fixed_point

Tensor = torch.Tensor
_DTYPE = torch.float64


@dataclass(frozen=True)
class RoofWiring:
    """Which columns of the layer's flow vector `q` are each street's roof exchange and each
    junction's vents, read from the edge attributes `build_model` writes (the one junction
    enumeration of this application). Street and junction indices follow the street order
    and `StreetNetwork.junctions`."""

    n_streets: int
    n_junctions: int
    exchange_col: Tensor
    """`(n_streets,)`: the column of street `i`'s outgoing `exchange` edge."""
    vent_out_col: Tensor
    vent_out_street: Tensor
    vent_out_junction: Tensor
    vent_in_col: Tensor
    vent_in_street: Tensor
    vent_in_junction: Tensor


def roof_wiring(model: Model, *, layer_name: str = "street") -> RoofWiring:
    """The flow-column wiring of a `background="per_street"` street model.

    Refuses, by name, a model whose atmosphere is not one node per street in street order
    (a `background="uniform"` model): its single `x_boundary` cannot carry a per-street
    `C_ext`.
    """
    if layer_name not in model.transport:
        raise KeyError(f"roof_wiring: the model has no transport layer {layer_name!r}")
    layer = model.transport[layer_name]
    net = model.net
    n = layer.n_i
    boundary = list(layer.boundary)
    if len(boundary) != n:
        raise ValueError(
            f"above_roof: the model has {len(boundary)} atmosphere node(s) for {n} streets; "
            f"the coupled above-roof solve needs one per street -- build it with "
            f"build_model(..., background='per_street')"
        )
    streets = [net.nodes[i] for i in layer.interior_idx.tolist()]
    offset = 0
    cols: dict[str, Tensor] = {}
    for kind in layer.flow_kinds:
        idx = net.edge_index(kind)
        cols[kind] = torch.arange(offset, offset + idx.numel())
        offset += idx.numel()
    for kind in ("vent", "exchange"):
        if kind not in cols:
            raise ValueError(
                f"above_roof: transport layer {layer_name!r} has no {kind!r} edges; it was "
                f"not built by noodl.apps.street_aq.build_model"
            )

    def edges(kind: str):
        for pos, col in enumerate(net.edge_index(kind).tolist()):
            a, b, key = net.edges[col]
            yield int(cols[kind][pos]), a, b, net.graph.edges[a, b, key]

    exchange = [-1] * n
    for col, a, b, data in edges("exchange"):
        i, out = int(data["street"]), data["direction"] == "out"
        street, atm = (a, b) if out else (b, a)
        if street != streets[i] or atm != boundary[i]:
            raise ValueError(
                f"above_roof: street {streets[i]!r} exchanges with atmosphere node "
                f"{atm!r}, not its own {boundary[i]!r}; build the model with "
                f"build_model(..., background='per_street')"
            )
        if out:
            exchange[i] = col
    vents = {True: ([], [], []), False: ([], [], [])}
    for col, a, b, data in edges("vent"):
        i, out = int(data["street"]), data["direction"] == "out"
        atm = b if out else a
        if atm != boundary[i]:
            raise ValueError(
                f"above_roof: street {streets[i]!r} vents to atmosphere node {atm!r}, not "
                f"its own {boundary[i]!r}; build the model with "
                f"build_model(..., background='per_street')"
            )
        c, s, j = vents[out]
        c.append(col)
        s.append(i)
        j.append(int(data["junction"]))
    # `build_model` derives its junctions from the streets' ends and gives every street end
    # a vent pair, so every junction index 0..n_j-1 appears on some vent edge and this IS
    # the network's junction count (the topology `Network` does not keep the list itself).
    n_j = 1 + max(vents[True][2] + vents[False][2])

    def long(values) -> Tensor:
        return torch.tensor(values, dtype=torch.long)

    return RoofWiring(
        n_streets=n, n_junctions=n_j, exchange_col=long(exchange),
        vent_out_col=long(vents[True][0]), vent_out_street=long(vents[True][1]),
        vent_out_junction=long(vents[True][2]),
        vent_in_col=long(vents[False][0]), vent_in_street=long(vents[False][1]),
        vent_in_junction=long(vents[False][2]),
    )


def roof_fluxes(wiring: RoofWiring, q: Tensor, x: Tensor, c_ext: Tensor, c_bg: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
    """`(F_s, F_up, F_down)`, kg/s: every street's signed roof flux and every junction's
    upward and downward vent flux in excess of the background (module docstring), from the
    flows `q (..., n_edges)` and the street, above-roof and background concentrations `x`,
    `c_ext`, `c_bg`, all `(..., n_streets, n_species)` (a trailing species axis always).
    Returns `(..., n_streets, n_species)` and twice `(..., n_junctions, n_species)`."""
    q = q.unsqueeze(-1)
    f_s = q[..., wiring.exchange_col, :] * (x - c_ext)

    def per_junction(values: Tensor, junction: Tensor) -> Tensor:
        out = torch.zeros(*values.shape[:-2], wiring.n_junctions, values.shape[-1],
                          dtype=_DTYPE)
        return out.index_add(-2, junction, values)

    so, si = wiring.vent_out_street, wiring.vent_in_street
    up = per_junction(q[..., wiring.vent_out_col, :] * (x[..., so, :] - c_bg[..., so, :]),
                      wiring.vent_out_junction)
    down = per_junction(
        q[..., wiring.vent_in_col, :] * (c_ext[..., si, :] - c_bg[..., si, :]),
        wiring.vent_in_junction,
    )
    return f_s, up, down


def street_steady_with_plume(
    model: Model,
    state: State,
    drivers: Drivers,
    *,
    kernel: Tensor,
    junction_kernel: Tensor | None,
    junction_source: str = "upward",
    background: Tensor | None = None,
    tol: float = 1e-12,
    max_iter: int = 100,
    relaxation: float = 1.0,
    adjoint_rtol: float = 1e-12,
    adjoint_restart: int = 20,
    diagnostics: dict | None = None,
    layer_name: str = "street",
    **solve_kwargs,
) -> dict[str, Tensor]:
    """The steady state of the street network with `C_ext` from the above-roof plumes.

    `model` must be built with `background="per_street"` (refused by name otherwise).
    `kernel` is `plume.street_kernel`'s `batch + (n_streets, n_streets)` and
    `junction_kernel` is `plume.junction_kernel`'s `batch + (n_streets, n_junctions)`, or
    `None` to leave the junction vertical fluxes out of `C_ext` (a stated choice: it is a
    required keyword so that nobody drops them by accident). `junction_source` is
    `"upward"` (the default: the junction plume source is `F_up`) or `"net"`
    (`F_up - F_down`); the module docstring gives the measured reason for the default.
    `background` is `C_bg`, in the
    shape of `"<layer>.x_boundary"` (`(n_streets,)` or `(n_streets, n_species)`, plus any
    batch dims); `None` takes `drivers["<layer>.x_boundary"]`. The kernels, the drivers and
    the background broadcast over the leading batch dims (hours), as `Model.steady` does.

    `tol` is RELATIVE and PER INSTANCE: the iteration stops once, in every instance (hour),
    the largest change of `C_ext` over one pass is at most `tol` times that instance's
    largest `|C_ext|` (an all-zero instance counts as converged).
    `relaxation` in `(0, 1]` under-relaxes the update. `adjoint_rtol` and
    `adjoint_restart` go to the adjoint GMRES (`differentiate_fixed_point`): the framework's
    GMRES always completes a restart cycle before it tests the true residual, and every
    matvec is a VJP through one street solve, so the cycle is kept short (20: the coupling
    of a street lattice contracts fast, so a cycle of 20 meets 1e-12 in one or two cycles,
    where the default `min(m, 100)` costs more VJPs).
    `**solve_kwargs` go to
    `Model.steady`. `diagnostics`, when given, is updated with `passes` (street solves in the
    forward iteration, not counting the final differentiable pass), `changes` (the relative
    change of every pass, the largest over the instances), `contraction` (the ratio of the
    last two changes: the measured spectral radius of the coupling) and
    `batched_adjoint`.

    Returns `{"<layer>.x": street concentrations, "<layer>.c_ext": above-roof
    concentrations, "<layer>.roof_flux": F_s, "<layer>.junction_flux": the junction plume
    source used, "<layer>.junction_downflux": F_down}`, the first two in the state's
    shapes, the fluxes `(..., n_streets[, n_species])` and `(..., n_junctions[, n_species])`
    in kg/s. `c_ext` is the pass's image of the converged
    iterate, `C_bg + K F`, and `x` the street solve at that iterate: they satisfy the
    relation to `tol` (times `1 + contraction`). Gradients are those of the converged fixed
    point (implicit adjoint; module docstring).
    """
    if not 0.0 < relaxation <= 1.0:
        raise ValueError(
            f"street_steady_with_plume: relaxation must be in (0, 1], got {relaxation!r}"
        )
    if int(max_iter) < 1:
        raise ValueError(f"street_steady_with_plume: max_iter must be >= 1, got {max_iter!r}")
    if junction_source not in ("upward", "net"):
        raise ValueError(
            f"street_steady_with_plume: junction_source must be 'upward' or 'net', got "
            f"{junction_source!r}"
        )
    wiring = roof_wiring(model, layer_name=layer_name)
    layer = model.transport[layer_name]
    single = layer.n_species == 1
    x_key, xb_key = f"{layer_name}.x", f"{layer_name}.x_boundary"
    n = wiring.n_streets
    c_bg = drivers.get(xb_key) if background is None else background
    if c_bg is None:
        raise KeyError(
            f"street_steady_with_plume: no background -- pass background= or the driver "
            f"{xb_key!r}"
        )
    c_bg = torch.as_tensor(c_bg, dtype=_DTYPE)
    trailing = (n,) if single else (n, layer.n_species)
    if tuple(c_bg.shape[len(c_bg.shape) - len(trailing):]) != trailing:
        raise ValueError(
            f"street_steady_with_plume: the background must end in {trailing} (one value "
            f"per street{'' if single else ' and species'}), got {tuple(c_bg.shape)}"
        )
    k_s = torch.as_tensor(kernel, dtype=_DTYPE)
    if tuple(k_s.shape[-2:]) != (n, n):
        raise ValueError(
            f"street_steady_with_plume: kernel must end in ({n}, {n}), got "
            f"{tuple(k_s.shape)}"
        )
    k_j = None if junction_kernel is None else torch.as_tensor(junction_kernel, dtype=_DTYPE)
    if k_j is not None and tuple(k_j.shape[-2:]) != (n, wiring.n_junctions):
        raise ValueError(
            f"street_steady_with_plume: junction_kernel must end in "
            f"({n}, {wiring.n_junctions}), got {tuple(k_j.shape)}"
        )
    q = model.current_flows(layer_name, state, drivers)

    def species_axis(t: Tensor) -> Tensor:
        return t.unsqueeze(-1) if single else t

    def unspecies(t: Tensor) -> Tensor:
        return t.squeeze(-1) if single else t

    def plume_pass(c_ext: Tensor) -> list[Tensor]:
        solved = model.steady(state, {**drivers, xb_key: c_ext}, **solve_kwargs)
        x = solved[x_key]
        f_s, up, down = roof_fluxes(wiring, q, species_axis(x), species_axis(c_ext),
                                    species_axis(c_bg))
        f_j = up if junction_source == "upward" else up - down
        c_new = species_axis(c_bg) + k_s @ f_s
        if k_j is not None:
            c_new = c_new + k_j @ f_j
        return [x, unspecies(c_new), unspecies(f_s), unspecies(f_j), unspecies(down)]

    changes: list[float] = []
    with torch.no_grad():
        c_ext = c_bg.detach()
        converged = False
        for _pass in range(int(max_iter)):
            c_new = plume_pass(c_ext)[1]
            c_ext = torch.broadcast_to(c_ext, c_new.shape)
            step = c_new - c_ext
            # Per instance (hour): the largest change over its streets (and species)
            # relative to its own largest |C_ext|, so a clean hour cannot hide behind a
            # polluted one's scale; the pass converges when every hour has.
            flat = (-1,) if single else (-2, -1)
            scale = c_new.abs().amax(dim=flat)
            rel = torch.where(scale > 0, step.abs().amax(dim=flat)
                              / torch.where(scale > 0, scale, torch.ones_like(scale)),
                              torch.zeros_like(scale))
            change = float(rel.max())
            changes.append(change)
            c_ext = c_ext + relaxation * step
            if change <= tol:
                converged = True
                break
    if not converged:
        raise RuntimeError(
            f"street_steady_with_plume: the above-roof fixed point did not converge within "
            f"{max_iter} passes; the relative change on the last pass was {changes[-1]:.3e} "
            f"against tol {tol:.1e} (measured contraction "
            f"{_contraction(changes):.3g}; pass relaxation < 1 if it is >= 1)"
        )
    batch = tuple(c_ext.shape[:-1] if single else c_ext.shape[:-2])
    report: dict = {}

    def pass_fn(z: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        outputs = plume_pass(z[0])
        return outputs, [outputs[1]]

    # The final pass at the converged iterate: under the implicit adjoint when anything
    # the answer depends on requires grad, plainly (no graph) otherwise.
    if torch.is_grad_enabled() and _requires_grad(model, state, drivers,
                                                  [c_bg, k_s, k_j]):
        outputs = differentiate_fixed_point(
            [c_ext], pass_fn, rtol=adjoint_rtol, restart=adjoint_restart,
            where="street_steady_with_plume", batch_shape=batch or None, report=report,
        )
    else:
        outputs = plume_pass(c_ext)
    x, c_out, f_s, f_j, down = outputs
    if diagnostics is not None:
        diagnostics.update({
            "passes": len(changes), "changes": changes,
            "contraction": _contraction(changes),
            "batched_adjoint": bool(report.get("batched", False)),
        })
    return {x_key: x, f"{layer_name}.c_ext": c_out, f"{layer_name}.roof_flux": f_s,
            f"{layer_name}.junction_flux": f_j, f"{layer_name}.junction_downflux": down}


def _requires_grad(model: Model, state: State, drivers: Drivers,
                   extra: list[Tensor | None]) -> bool:
    """Whether any input of the pass requires grad: the state, the drivers, the kernels and
    background, and every tensor attribute of the model's transport layers and closures
    (a capacity or geometry tensor made a parameter)."""
    tensors = [*state.values(), *drivers.values(), *extra]
    for part in [*model.transport.values(), *model.closures]:
        for value in vars(part).values():
            tensors.append(value)
            if hasattr(value, "__dict__") and not isinstance(value, torch.Tensor):
                tensors.extend(vars(value).values())
    return any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors)


def _contraction(changes: list[float]) -> float:
    """The ratio of the last two relative changes (NaN before two passes, 0 once a change
    is exactly zero)."""
    if len(changes) < 2:
        return float("nan")
    return changes[-1] / changes[-2] if changes[-2] > 0 else 0.0

"""Intersection classification, routing, closure, direction averaging, and the closure
that turns the wind into every prescribed edge flow.

Every rule here is MUNICH's `ComputeIntersectionFlux`
(`include/models/StreetNetworkTransport.cxx:2851-3040`), `ComputeAlpha` (`:3620-3648`) and
`ComputeWindDirectionFluctuation` (`:3562-3616`), transcribed from the MUNICH source.
Nothing loops over junctions at
call time: the junction layout is padded to the greatest degree once, at construction, and
every step below is a batched tensor operation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from noodl.apps.street_aq.canyon import (
    KAPPA,
    KAPPA_MUNICH,
    Z0_S_DEFAULT,
    BoundaryLayer,
    boundary_layer,
    canyon_velocity,
    exchange_velocity,
    roof_wind,
)

Tensor = torch.Tensor
TWO_PI = 2.0 * math.pi

MAX_SIGMA_THETA = math.pi / 18.0
"""MUNICH's cap on the wind-direction fluctuation, 10 degrees (`:3566`; the comment cites
Ben Salem et al. 2015, "maximum fluctuation of +/-20 deg (2 sigma_theta)")."""

MAX_N_THETA = 10

_FLOW_KINDS = ("route", "vent", "exchange")
"""The edge kinds `StreetFlows` writes its concatenated `q` in, and the order
`build_model` must build the transport layer with. Checked at construction."""

_SAFE_RECIPROCAL = 1e-300
"""The smallest magnitude `_safe_reciprocal` divides by: `1 / 1e-300` is finite, and even a
junction's worth of such terms summed stays far below the float64 maximum."""


def _safe_reciprocal(x: Tensor) -> Tensor:
    """`1 / x` with `|x| < _SAFE_RECIPROCAL` replaced by `+-_SAFE_RECIPROCAL` keeping `x`'s
    sign (`+` at an exact zero): finite forward value and finite gradient everywhere. Equal
    to `1 / x` bit for bit wherever `|x| >= _SAFE_RECIPROCAL`."""
    small = x.abs() < _SAFE_RECIPROCAL
    signed = torch.where(x < 0, torch.full_like(x, -_SAFE_RECIPROCAL),
                         torch.full_like(x, _SAFE_RECIPROCAL))
    return 1.0 / torch.where(small, signed, x)


def _safe_atan2(s: Tensor, c: Tensor) -> Tensor:
    """`atan2(s, c)`, with the origin `(0, 0)` -- where the forward value is already 0 but
    the gradient is `0 / 0` -- replaced by the safe input `(0, 1)`: same forward value,
    zero gradient there instead of NaN."""
    origin = (s == 0) & (c == 0)
    return torch.atan2(torch.where(origin, torch.zeros_like(s), s),
                       torch.where(origin, torch.ones_like(c), c))


def sigma_theta_munich(sigma_v: Tensor, u_ref: Tensor) -> Tensor:
    """`sigma_theta = min(sigma_v / U, 10 deg)` (SRC `:3567`; Blackadar 1997, Soulhac 2009).

    NOT a free parameter: a turbulence-intensity estimate capped at 10 degrees. For the
    neutral branch `sigma_v = 1.2 u*` exactly, so `sigma_theta` is independent of the wind
    speed whenever `u*` is proportional to it -- which is why the published idealised case
    has the same sample count at 5 and at 10 m/s.
    """
    return torch.clamp(sigma_v / u_ref, max=MAX_SIGMA_THETA)


def n_theta_munich(sigma_theta: Tensor) -> Tensor:
    """`ntheta = floor(sigma_theta in DEGREES)` (SRC `:3568`), as a long tensor, clamped to
    `[1, 10]`. MUNICH skips the averaging entirely at 1, i.e. below two degrees."""
    degrees = sigma_theta * 180.0 / math.pi
    return torch.clamp(torch.floor(degrees).long(), min=1, max=MAX_N_THETA)


def direction_offsets(
    scheme: str, sigma_theta: Tensor, *, n_theta: int | None = None
) -> tuple[Tensor, Tensor]:
    """Wind-direction samples and their weights, `(offsets (..., m), weights (..., m))`.

    `"none"`: one sample at offset 0 with weight 1 -- no direction averaging at all.

    `"munich"`: MUNICH's own scheme, reproduced including its artefacts. Uniform
    (rectangle-rule) sampling on `[-2 sigma, +2 sigma]` with both endpoints at full weight,
    `step = 4 sigma/(n-1)`, `w_k = step N(theta_k; 0, sigma)`, and the weights are NOT
    normalised: they sum to 0.974953 at `n = 10` and to 0.431928 at `n = 2` (K22 Eq. B16;
    SRC `:3562-3616`). That scales every intersection flux, and reproducing it is the whole
    point -- "fixing" it puts a uniform few-percent bias between this model and MUNICH.
    `n` is per instance, `floor(sigma_theta in degrees)`; an instance needing fewer samples
    than the widest one in the batch gets exactly zero weight in its surplus slots, which
    is the same sum MUNICH computes for it.

    `"gauss"`: noodl's own, `n_theta`-point Gauss-Hermite with NORMALISED weights, for a
    smooth derivative in the mean direction. Used in no parity test.
    """
    sigma_theta = torch.as_tensor(sigma_theta, dtype=torch.float64)
    if scheme == "none":
        shape = sigma_theta.shape + (1,)
        return (torch.zeros(shape, dtype=torch.float64),
                torch.ones(shape, dtype=torch.float64))
    if scheme == "munich":
        counts = n_theta_munich(sigma_theta)
        m = int(counts.max())
        if m <= 1:
            shape = sigma_theta.shape + (1,)
            return (torch.zeros(shape, dtype=torch.float64),
                    torch.ones(shape, dtype=torch.float64))
        k = torch.arange(m, dtype=torch.float64)
        n = counts.unsqueeze(-1).to(torch.float64)
        active = (k < n) & (n > 1)
        step = torch.where(n > 1, 4.0 * sigma_theta.unsqueeze(-1) / (n - 1.0),
                           torch.ones_like(n))
        offsets = -2.0 * sigma_theta.unsqueeze(-1) + k * step
        sigma = sigma_theta.unsqueeze(-1)
        safe_sigma = torch.where(sigma > 0, sigma, torch.ones_like(sigma))
        weights = step * torch.exp(-0.5 * (offsets / safe_sigma) ** 2) / (
            safe_sigma * math.sqrt(2.0 * math.pi)
        )
        single = (counts <= 1).unsqueeze(-1)
        first = (k == 0).expand_as(weights).to(weights.dtype)
        offsets = torch.where(active, offsets, torch.zeros_like(offsets))
        weights = torch.where(active, weights, torch.zeros_like(weights))
        offsets = torch.where(single, torch.zeros_like(offsets), offsets)
        weights = torch.where(single, first, weights)
        return offsets, weights
    if scheme == "gauss":
        if n_theta is None or n_theta < 1:
            raise ValueError(
                f"direction_offsets: scheme='gauss' needs n_theta >= 1, got {n_theta!r}"
            )
        nodes, raw = np.polynomial.hermite_e.hermegauss(int(n_theta))
        nodes_t = torch.as_tensor(nodes, dtype=torch.float64)
        weights_t = torch.as_tensor(raw, dtype=torch.float64) / math.sqrt(2.0 * math.pi)
        offsets = sigma_theta.unsqueeze(-1) * nodes_t
        return offsets, weights_t.expand(offsets.shape).clone()
    raise ValueError(
        f"direction_offsets: scheme must be 'none', 'munich' or 'gauss', got {scheme!r}"
    )


def node_closure(flux: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """MUNICH's roof closure at an intersection, before any routing (SRC `:2980-3014`).

    `flux` is the signed volume flux per slot, `(..., d)`, positive INTO the junction and
    exactly zero on padding. With `P0 = sum(inflow) - sum(outflow)`:

    * `P0 > 0` (the node exports): `alpha0 = P0/sum(inflow)`; every INFLOW gives
      `alpha0 P_in` to the atmosphere and keeps `(1 - alpha0) P_in` for the streets.
    * `P0 < 0` (the node imports): `alpha0 = |P0|/sum(outflow)`; every OUTFLOW receives
      `alpha0 P_out` from the atmosphere and takes `(1 - alpha0) P_out` from the streets.

    The correction lands on ONE side only, never on both -- that asymmetry is MUNICH's, and
    it is what makes a degree-one junction (a dead end) a one-way exchange with the
    background rather than a wall. Returns
    `(p_in, p_out, to_atmosphere, from_atmosphere)`, all `(..., d)` and all `>= 0`.
    """
    p_in = torch.clamp(flux, min=0.0)
    p_out = torch.clamp(-flux, min=0.0)
    sum_in = p_in.sum(-1, keepdim=True)
    sum_out = p_out.sum(-1, keepdim=True)
    imbalance = sum_in - sum_out
    exports = imbalance > 0
    imports = imbalance < 0
    safe_in = torch.where(sum_in > 0, sum_in, torch.ones_like(sum_in))
    safe_out = torch.where(sum_out > 0, sum_out, torch.ones_like(sum_out))
    alpha_export = torch.where(exports & (sum_in > 0), imbalance / safe_in,
                               torch.zeros_like(sum_in))
    alpha_import = torch.where(imports & (sum_out > 0), -imbalance / safe_out,
                               torch.zeros_like(sum_out))
    to_atmosphere = alpha_export * p_in
    from_atmosphere = alpha_import * p_out
    return p_in - to_atmosphere, p_out - from_atmosphere, to_atmosphere, from_atmosphere


def routing_matrix(flux_in: Tensor, flux_out: Tensor, *, model: str) -> Tensor:
    """The street-to-street flux matrix `F[..., p, r]` from ORDERED, closed marginals.

    `"mixing"`: perfect mixing, `F = P_in outer (P_out / sum P_out)`.

    `"sirane"`: the non-crossing-streamline fill. MUNICH's `ComputeAlpha` (`:3620-3648`)
    walks the inflows counter-clockwise and fills the outflows clockwise with
    `min(remaining_in, remaining_out)`. That greedy sweep IS the north-west-corner rule on
    the two ordered marginals, so it is written here in closed form -- with `A` and `B` the
    two cumulative sums,

        F[p, r] = max(0, min(A_p, B_r) - max(A_{p-1}, B_{r-1}))

    -- which is exact, batched, and piecewise linear in the marginals, so the gradient runs
    through the fluxes while the combinatorics live entirely in the ORDERING. Checked
    against a worked example traced through MUNICH's own code: inflows
    `[10, 4]` against outflows `[6, 8]` give `[[6, 4], [0, 4]]`, where perfect mixing would
    give `[[4.286, 5.714], [1.714, 2.286]]`.
    """
    if model == "mixing":
        total = flux_out.sum(-1, keepdim=True)
        safe = torch.where(total > 0, total, torch.ones_like(total))
        share = torch.where(total > 0, flux_out / safe, torch.zeros_like(flux_out))
        return flux_in.unsqueeze(-1) * share.unsqueeze(-2)
    if model != "sirane":
        raise ValueError(
            f"routing_matrix: model must be 'mixing' or 'sirane', got {model!r}"
        )
    a = torch.cumsum(flux_in, dim=-1)
    b = torch.cumsum(flux_out, dim=-1)
    a_prev = a - flux_in
    b_prev = b - flux_out
    upper = torch.minimum(a.unsqueeze(-1), b.unsqueeze(-2))
    lower = torch.maximum(a_prev.unsqueeze(-1), b_prev.unsqueeze(-2))
    return torch.clamp(upper - lower, min=0.0)


def order_slots(angle: Tensor, active: Tensor, *, descending: bool) -> Tensor:
    """The slot order MUNICH walks, `(..., d)`, with inactive slots pushed to the end.

    Inflows are taken in DECREASING angle (counter-clockwise) and outflows in INCREASING
    angle (clockwise) -- SRC `:2907-2972`. Both lists are then rotated by one position when
    the leading gap exceeds pi, which is MUNICH's `while` loop: the cyclic gaps of a set of
    angles sum to 2 pi, so AT MOST ONE of them can exceed pi and a single rotation always
    settles it. Where the INACTIVE slots land does not matter -- their flux is zero, and a
    zero does not move a cumulative sum, so `routing_matrix` gives them zero rows and
    columns wherever they sit.
    """
    filler = -1e30 if descending else 1e30
    big = torch.where(active, angle, torch.full_like(angle, filler))
    order = torch.argsort(big, dim=-1, descending=descending)
    ordered_angle = torch.gather(angle, -1, order)
    ordered_active = torch.gather(active, -1, order)
    count = ordered_active.sum(-1, keepdim=True)
    d = angle.shape[-1]
    index = torch.arange(d, device=angle.device).expand_as(order)
    two_or_more = count >= 2
    if descending:
        second = torch.gather(
            ordered_angle, -1, torch.clamp(torch.ones_like(count), max=d - 1)
        )
        gap = ordered_angle[..., :1] - second
        shifted = torch.where(index < count, (index + 1) % torch.clamp(count, min=1),
                              index)
    else:
        last = torch.gather(ordered_angle, -1, torch.clamp(count - 1, min=0))
        prev = torch.gather(ordered_angle, -1, torch.clamp(count - 2, min=0))
        gap = last - prev
        shifted = torch.where(index < count, (index - 1) % torch.clamp(count, min=1),
                              index)
    rotate = (gap > math.pi) & two_or_more
    return torch.gather(order, -1, torch.where(rotate, shifted, index))


@dataclass(frozen=True)
class StreetGeometry:
    """Everything `StreetFlows` needs about the streets, as plain tensors.

    Kept here rather than in `network.py` so that `routing` never imports `network`:
    `network` builds one of these from its `StreetNetwork` and hands it over. `azimuth` is
    radians counter-clockwise from east, pointing `u -> v`.
    """

    names: list[str]
    u: list[str]
    v: list[str]
    length: Tensor
    width: Tensor
    height: Tensor
    z0_b: Tensor
    azimuth: Tensor


class StreetFlows:
    """The closure that turns the wind drivers into every prescribed edge flow.

    Writes `"<layer>.q"` (the concatenated `route`, `vent` and `exchange` flows, all
    non-negative, in the layer's own `flow_kinds` order) and `"<layer>.u_canyon"` (the
    SIGNED along-canyon velocity per street, for reporting). Reads `"U_ref"`, `"theta_w"`,
    `"h_abl"` and -- for `stability="munich"` -- `"lmo"`, each shaped `(...,)`.

    `"u_star"`, when present, is the friction velocity itself and replaces the log law
    `u* = kappa U_ref / ln((z_ref - d)/z0)` (MUNICH takes it as an input on its
    `With_local_data` path). `U_ref` is then needed only for MUNICH's direction spread
    `sigma_theta = sigma_v / U`, i.e. only with `direction_averaging="munich"`.

    `meteo="per_street"` gives every one of those drivers a trailing STREET axis,
    `(..., n_streets)`: each street's canyon wind and roof exchange come from its own
    values. Junction routing then uses per-junction values `(..., n_junctions)`, in
    `StreetNetwork.junctions` order, read from `"<name>_junction"` drivers when given
    (MUNICH's `WindDirectionInter`, `WindSpeedInter`, `USTInter`, `PBLHInter`,
    `LMOInter`) and otherwise the mean over the streets meeting there -- circular for the
    direction, through the reciprocal for the Obukhov length (`junction_values`).

    Order of operations, which is MUNICH's (`ComputeIntersectionFlux`):
    the canyon velocities are computed ONCE from the MEAN wind direction and are NOT
    recomputed for the perturbed directions; only the in/out classification and the angular
    ordering change from sample to sample, and the weighted sum is taken over the resulting
    flux matrix. Reproducing that is what makes the direction averaging comparable with
    MUNICH at all.

    `kappa=None` (the default) resolves to MUNICH's 0.41 whenever any
    MUNICH-style form is selected (`canyon_wind="exponential"`, `exchange="schulte"` or
    `roof_wind_form="macdonald"`), and to the neutral form's 0.4 otherwise; an explicit
    float always wins over that resolution.
    """

    def __init__(
        self,
        net,
        layer,
        geometry: StreetGeometry,
        *,
        canyon_wind: str = "soulhac",
        exchange: str = "sirane",
        routing: str = "sirane",
        direction_averaging: str = "none",
        n_theta: int | None = None,
        sigma_theta: float | None = None,
        kappa: float | None = None,
        canyon_wind_min: float = 0.0,
        u_d_min: float = 0.0,
        stability: str = "neutral",
        roof_wind_form: str = "sirane",
        z0_s: float = Z0_S_DEFAULT,
        z_ref: float = 30.0,
        pblh_floor: bool = True,
        meteo: str = "uniform",
        layer_name: str = "street",
    ) -> None:
        if meteo not in ("uniform", "per_street"):
            raise ValueError(
                f"StreetFlows: meteo must be 'uniform' or 'per_street', got {meteo!r}"
            )
        if canyon_wind not in ("soulhac", "exponential"):
            raise ValueError(
                f"StreetFlows: canyon_wind must be 'soulhac' or 'exponential', got "
                f"{canyon_wind!r}"
            )
        if direction_averaging not in ("none", "munich", "gauss"):
            raise ValueError(
                f"StreetFlows: direction_averaging must be 'none', 'munich' or 'gauss', "
                f"got {direction_averaging!r}"
            )
        # `q` is written as one concatenated block, so the layer's own `flow_kinds` order
        # IS the slot layout this closure assumes; a layer built with the kinds in any
        # other order would take the route flows for vent flows with no error anywhere.
        # `layer=None` stays legal: the routing tests drive `_flows` without a Model.
        if layer is not None:
            kinds = tuple(layer.flow_kinds)
            if kinds != _FLOW_KINDS:
                raise ValueError(
                    f"StreetFlows: transport layer {getattr(layer, 'name', layer)!r} has "
                    f"flow_kinds {kinds}, and this closure writes its 'q' in the order "
                    f"{_FLOW_KINDS}; the two must agree exactly"
                )
        self.net = net
        self.layer = layer
        self.geometry = geometry
        self.canyon_wind = canyon_wind
        self.exchange = exchange
        self.routing = routing
        self.direction_averaging = direction_averaging
        self.n_theta = n_theta
        self.sigma_theta = sigma_theta
        if kappa is None:
            munich_form = (
                canyon_wind == "exponential"
                or exchange == "schulte"
                or roof_wind_form == "macdonald"
            )
            kappa = KAPPA_MUNICH if munich_form else KAPPA
        self.kappa = float(kappa)
        self.canyon_wind_min = float(canyon_wind_min)
        self.u_d_min = float(u_d_min)
        self.stability = stability
        self.roof_wind_form = roof_wind_form
        self.z0_s = float(z0_s)
        self.z_ref = float(z_ref)
        self.pblh_floor = bool(pblh_floor)
        self.meteo = meteo
        self.layer_name = layer_name
        self.h_mean = geometry.height.mean()
        self.w_mean = geometry.width.mean()
        self.h_max = float(geometry.height.max())
        self._read_edges(net)

    def _read_edges(self, net) -> None:
        """The WHOLE slot layout and every index tensor, from the edges' own attributes.

        There is exactly one junction enumeration in this application and it lives in
        `build_model`, which writes it onto the edges. Nothing here re-derives it:
        two enumerations by different rules (`u + v` over all streets against `(u, v)` per
        street) would put the flows on the wrong edges and move the answer by 34 % with no
        error anywhere.
        """
        edges = net.edges
        vent_rows: list[tuple[int, int, int, int]] = []
        vent_flat: list[int] = []
        vent_is_out: list[bool] = []
        for col in net.edge_index("vent").tolist():
            a, b, key = edges[col]
            data = net.graph.edges[a, b, key]
            junction, slot = int(data["junction"]), int(data["slot"])
            sign = 1 if data["end"] == "u" else -1
            vent_rows.append((junction, slot, int(data["street"]), sign))
            vent_is_out.append(data["direction"] == "out")
        n_j = max(row[0] for row in vent_rows) + 1
        d = max(row[1] for row in vent_rows) + 1
        self.n_junctions, self.d_max = n_j, d
        slot_street = torch.zeros(n_j, d, dtype=torch.long)
        slot_sign = torch.zeros(n_j, d, dtype=torch.long)
        slot_active = torch.zeros(n_j, d, dtype=torch.bool)
        for junction, slot, street, sign in vent_rows:
            slot_street[junction, slot] = street
            slot_sign[junction, slot] = sign
            slot_active[junction, slot] = True
        # The angle points AWAY from the junction -- the street's own azimuth at its `u`
        # end and the reverse at its `v` end. That is the angle MUNICH's in/out test
        # compares against the wind (`ComputeIntersection`, `:1885-1895`).
        turn = torch.where(
            slot_sign > 0,
            torch.zeros(n_j, d, dtype=torch.float64),
            torch.full((n_j, d), math.pi, dtype=torch.float64),
        )
        angle = torch.remainder(self.geometry.azimuth[slot_street] + turn, TWO_PI)
        self.slot_street = slot_street
        self.slot_active = slot_active
        self.slot_angle = torch.where(slot_active, angle, torch.zeros_like(angle))
        for junction, slot, _street, _sign in vent_rows:
            vent_flat.append(junction * d + slot)
        self.vent_flat = torch.tensor(vent_flat, dtype=torch.long)
        self.vent_is_out = torch.tensor(vent_is_out, dtype=torch.bool)
        route_flat: list[int] = []
        for col in net.edge_index("route").tolist():
            a, b, key = edges[col]
            data = net.graph.edges[a, b, key]
            route_flat.append(
                int(data["junction"]) * d * d + int(data["slot_a"]) * d
                + int(data["slot_b"])
            )
        self.route_flat = torch.tensor(route_flat, dtype=torch.long)
        exchange_street: list[int] = []
        for col in net.edge_index("exchange").tolist():
            a, b, key = edges[col]
            exchange_street.append(int(net.graph.edges[a, b, key]["street"]))
        self.exchange_street = torch.tensor(exchange_street, dtype=torch.long)

    # ---------------------------------------------------------------- the drivers

    def _driver(self, drivers, key: str, *, required: bool = True) -> Tensor | None:
        value = drivers.get(key)
        if value is None:
            if required:
                raise KeyError(
                    f"StreetFlows: the driver {key!r} is missing; it is needed here "
                    f"(canyon_wind={self.canyon_wind!r}, "
                    f"direction_averaging={self.direction_averaging!r}, "
                    f"meteo={self.meteo!r})"
                )
            return None
        return torch.as_tensor(value, dtype=torch.float64)

    def _on_streets(self, value: Tensor | None, key: str) -> Tensor | None:
        """A driver as a per-street tensor `(..., n_streets)`: an instance value gains a
        trailing street axis (`meteo="uniform"`), a per-street one is checked."""
        if value is None:
            return None
        n = len(self.geometry.names)
        if self.meteo == "uniform":
            return value.unsqueeze(-1)
        if value.dim() == 0 or value.shape[-1] != n:
            raise ValueError(
                f"StreetFlows: meteo='per_street' needs {key!r} with a trailing street "
                f"axis of length {n}, got shape {tuple(value.shape)}"
            )
        return value

    def _boundary_layer(self, u_ref: Tensor | None, u_star: Tensor | None,
                        h_abl: Tensor) -> BoundaryLayer:
        """The boundary layer at whatever shape its inputs have: from `u_star` when the
        driver gives it, else from the log law on `U_ref`."""
        floor = self.h_max if self.pblh_floor else None
        if u_star is None:
            if u_ref is None:
                raise KeyError(
                    "StreetFlows: neither 'U_ref' nor 'u_star' is among the drivers; one "
                    "of them sets the friction velocity"
                )
            return boundary_layer(self.h_mean, u_ref, h_abl, z_ref=self.z_ref,
                                  kappa=self.kappa, pblh_floor=floor)
        template = boundary_layer(self.h_mean, torch.ones((), dtype=torch.float64),
                                  torch.ones((), dtype=torch.float64), z_ref=self.z_ref,
                                  kappa=self.kappa)
        if floor is not None:
            h_abl = torch.maximum(h_abl, torch.as_tensor(floor, dtype=torch.float64))
        return BoundaryLayer(u_star=u_star, h_abl=h_abl, z_ref=template.z_ref,
                             d=template.d, z0=template.z0, kappa=self.kappa)

    def velocities(self, drivers) -> tuple[BoundaryLayer, Tensor, Tensor]:
        """`(per-street boundary layer, signed canyon velocity per street, exchange
        velocity)`, every tensor `(..., n_streets)`."""
        g = self.geometry
        u_ref = self._on_streets(self._driver(drivers, "U_ref", required=False), "U_ref")
        u_star = self._on_streets(self._driver(drivers, "u_star", required=False), "u_star")
        theta_w = self._on_streets(self._driver(drivers, "theta_w"), "theta_w")
        h_abl = self._on_streets(self._driver(drivers, "h_abl"), "h_abl")
        lmo = self._on_streets(self._driver(drivers, "lmo", required=False), "lmo")
        bl_s = self._boundary_layer(u_ref, u_star, h_abl)
        phi = theta_w - g.azimuth
        if self.canyon_wind == "soulhac":
            u_street = canyon_velocity(
                g.width, g.height, phi, u_star=bl_s.u_star, form="soulhac",
                z0_b=g.z0_b, kappa=self.kappa, canyon_wind_min=self.canyon_wind_min,
            )
        else:
            u_h = roof_wind(
                bl_s.u_star, g.height, g.width, form=self.roof_wind_form, z0_s=self.z0_s,
                kappa=self.kappa, h_mean=self.h_mean, w_mean=self.w_mean,
            )
            u_street = canyon_velocity(
                g.width, g.height, phi, u_h=u_h, form="exponential", z0_s=self.z0_s,
                canyon_wind_min=self.canyon_wind_min,
            )
        sigma_w = bl_s.sigma_w(g.height, lmo=lmo, stability=self.stability)
        u_d = exchange_velocity(sigma_w, g.height, g.width, form=self.exchange,
                                u_d_min=self.u_d_min)
        return bl_s, u_street, u_d

    def _street_mean(self, values: Tensor, how: str) -> Tensor:
        """Per-street `(..., n_streets)` -> per-junction `(..., n_junctions)`, the mean over
        the streets meeting at each junction.

        Padded (inactive) slots index street 0, so every non-linear step masks them BEFORE
        it is applied (a `* active` afterwards would turn a non-finite street-0 value into
        `inf * 0 = NaN`). The three singular points -- `1/L` at `L = 0`, `1/mean(1/L)` where
        the reciprocals cancel, and `atan2` at the origin (exactly opposing directions) --
        go through safe inputs selected by `torch.where`, so neither the forward value nor
        the gradient is ever NaN; the forward value moves only where it would otherwise be
        infinite (to a finite magnitude of at least `1 / _SAFE_RECIPROCAL`, the same side of
        every stability threshold) or undefined."""
        per_slot = values[..., self.slot_street]                     # (..., n_j, d)
        active = self.slot_active.to(values.dtype)
        count = active.sum(-1)
        if how == "circular":
            s = (torch.sin(per_slot) * active).sum(-1)
            c = (torch.cos(per_slot) * active).sum(-1)
            return torch.remainder(_safe_atan2(s, c), TWO_PI)
        if how == "reciprocal":
            reciprocal = torch.where(self.slot_active, _safe_reciprocal(per_slot),
                                     torch.zeros_like(per_slot))
            return _safe_reciprocal(reciprocal.sum(-1) / count)
        return (per_slot * active).sum(-1) / count

    def junction_values(self, drivers) -> dict[str, Tensor | None]:
        """The meteorology each junction routes with, `(..., n_junctions)` per key:
        `theta_w`, `U_ref`, `u_star`, `h_abl`, `lmo` (`None` where not available).

        `meteo="uniform"`: the instance values, the same at every junction.
        `meteo="per_street"`: the `"<key>_junction"` driver when given, else the mean over
        the streets meeting at the junction (circular for `theta_w`, through `1/L` for
        `lmo`); `u_star` falls back to the streets' friction velocities, whether driven or
        from the log law."""
        n_j = self.n_junctions
        how = {"theta_w": "circular", "lmo": "reciprocal"}
        out: dict[str, Tensor | None] = {}
        for key in ("theta_w", "U_ref", "u_star", "h_abl", "lmo"):
            value = self._driver(drivers, key, required=False)
            if value is None:
                out[key] = None
                continue
            if self.meteo == "uniform":
                out[key] = value.unsqueeze(-1).expand(*value.shape, n_j)
                continue
            given = self._driver(drivers, f"{key}_junction", required=False)
            if given is not None:
                if given.dim() == 0 or given.shape[-1] != n_j:
                    raise ValueError(
                        f"StreetFlows: {key + '_junction'!r} needs a trailing junction "
                        f"axis of length {n_j}, got shape {tuple(given.shape)}"
                    )
                out[key] = given
            else:
                out[key] = self._street_mean(
                    self._on_streets(value, key), how.get(key, "arithmetic")
                )
        if self.meteo == "per_street":
            for key in ("theta_w", "U_ref", "u_star", "h_abl", "lmo"):
                if out[key] is None:
                    given = self._driver(drivers, f"{key}_junction", required=False)
                    if given is not None:
                        out[key] = given
        if out["u_star"] is None and self.direction_averaging == "munich":
            if self.meteo == "uniform":
                if out["U_ref"] is None:
                    raise KeyError("StreetFlows: 'U_ref' is missing")
                out["u_star"] = self._boundary_layer(
                    out["U_ref"], None, out["h_abl"]
                ).u_star
            else:
                bl_s, _, _ = self.velocities(drivers)
                out["u_star"] = self._street_mean(bl_s.u_star, "arithmetic")
        return out

    def _samples(self, drivers) -> tuple[Tensor, Tensor, Tensor]:
        """`(junction direction (..., n_j), offsets (..., n_j, m), weights (..., n_j, m))`."""
        j = self.junction_values(drivers)
        theta_j = j["theta_w"]
        if theta_j is None:
            raise KeyError("StreetFlows: the driver 'theta_w' is missing")
        if self.direction_averaging == "munich":
            if j["U_ref"] is None:
                raise KeyError(
                    "StreetFlows: direction_averaging='munich' needs the driver 'U_ref' "
                    "(or 'U_ref_junction'): MUNICH's direction spread is "
                    "sigma_theta = sigma_v / U, even when 'u_star' is given"
                )
            h_abl_j = j["h_abl"]
            if self.pblh_floor:
                h_abl_j = torch.maximum(h_abl_j, torch.as_tensor(self.h_max,
                                                                  dtype=torch.float64))
            bl_j = BoundaryLayer(u_star=j["u_star"], h_abl=h_abl_j,
                                 z_ref=torch.as_tensor(self.z_ref, dtype=torch.float64),
                                 d=torch.zeros((), dtype=torch.float64),
                                 z0=torch.ones((), dtype=torch.float64), kappa=self.kappa)
            sigma_v = bl_j.sigma_v(lmo=j["lmo"], stability=self.stability)
            sigma = sigma_theta_munich(sigma_v, j["U_ref"])
        elif self.sigma_theta is not None:
            sigma = torch.full_like(theta_j, float(self.sigma_theta))
        else:
            sigma = torch.zeros_like(theta_j)
        offsets, weights = direction_offsets(self.direction_averaging, sigma,
                                             n_theta=self.n_theta)
        return theta_j, offsets, weights

    def __call__(self, state, drivers) -> dict[str, Tensor]:
        g = self.geometry
        _bl, u_street, u_d = self.velocities(drivers)
        theta_j, offsets, weights = self._samples(drivers)
        # Samples first, junctions second: (..., m, n_j).
        theta_k = (theta_j.unsqueeze(-1) + offsets).transpose(-1, -2)
        weights = weights.transpose(-1, -2)
        d, n_j = self.d_max, self.n_junctions
        with torch.no_grad():
            # `cos(theta - street angle) < 0` is MUNICH's `pi/2 < dangle < 3 pi/2` test
            # (`:2871-2882`), with the STRICT inequality that makes a street exactly
            # perpendicular to the wind an OUTFLOW.
            d_angle = theta_k[..., None] - self.slot_angle           # (..., m, n_j, d)
            is_in = (torch.cos(d_angle) < 0) & self.slot_active
            is_out = (~is_in) & self.slot_active
            order_in = order_slots(
                self.slot_angle.expand(is_in.shape), is_in, descending=True
            )
            order_out = order_slots(
                self.slot_angle.expand(is_out.shape), is_out, descending=False
            )
        magnitude = (u_street * g.width * g.height).abs()
        mag = (magnitude[..., self.slot_street] * self.slot_active).unsqueeze(-3)
        flux = torch.where(is_in, mag, -mag)
        p_in, p_out, to_atm, from_atm = node_closure(flux)
        p_in_ord = torch.gather(p_in.expand(order_in.shape), -1, order_in)
        p_out_ord = torch.gather(p_out.expand(order_out.shape), -1, order_out)
        f_ord = routing_matrix(p_in_ord, p_out_ord, model=self.routing)
        flat_idx = order_in.unsqueeze(-1) * d + order_out.unsqueeze(-2)
        flat_idx = flat_idx.expand(f_ord.shape).reshape(*f_ord.shape[:-2], d * d)
        f_slot = torch.zeros(*f_ord.shape[:-2], d * d, dtype=f_ord.dtype)
        f_slot = f_slot.scatter_add(
            -1, flat_idx, f_ord.reshape(*f_ord.shape[:-2], d * d)
        )
        w = weights[..., None]                                       # (..., m, n_j, 1)
        f_slot = (f_slot * w).sum(-3).reshape(*f_slot.shape[:-3], n_j * d * d)
        to_atm = (to_atm * w).sum(-3).reshape(*to_atm.shape[:-3], n_j * d)
        from_atm = (from_atm * w).sum(-3).reshape(*from_atm.shape[:-3], n_j * d)
        q_route = f_slot.index_select(-1, self.route_flat)
        q_vent = torch.where(
            self.vent_is_out,
            to_atm.index_select(-1, self.vent_flat),
            from_atm.index_select(-1, self.vent_flat),
        )
        q_exchange = (u_d * g.width * g.length).index_select(-1, self.exchange_street)
        q = torch.cat([q_route, q_vent, q_exchange], dim=-1)
        if bool((q < 0).any()):
            bad = int((q < 0).sum())
            raise ValueError(
                f"StreetFlows: {bad} prescribed flows came out negative (minimum "
                f"{float(q.min())}); every route, vent and exchange flow carries its "
                f"direction in the topology, so all of them must be non-negative"
            )
        return {f"{self.layer_name}.q": q, f"{self.layer_name}.u_canyon": u_street}

"""Drive protocol: additive potential-difference terms added to an element's dp.

A drive is a function of the `drivers` mapping ONLY: it never
sees the potential `phi`. The Newton Jacobian `A_I diag(g') A_I^T` is therefore exact and stays
symmetric, which is what the SPD certificate and the conjugate-gradient path rely on. Anything a
drive needs that depends on the state -- zone densities for a stack term -- is computed by a
`Model` closure before the solve and written into `drivers`. CONTAM does the same: densities
from barometric pressure and temperature before the iteration, no density-pressure term in the
Jacobian (TN 1887r1 section 3.18, eq. 8-9).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class Drive(Protocol):
    """A drive must read every differentiable quantity from `drivers`, never own one.

    `PotentialFlowLayer.solve(differentiable=True)` only threads gradients through
    `Function.apply` for values reachable via `phi_boundary`, `sources`, `drivers`, and each
    Element's own registered `nn.Parameter`s. A Drive instance is captured by closure inside
    the differentiable solve, so any tensor it holds with `requires_grad=True` is invisible to
    the backward pass; `PotentialFlowLayer._check_no_unreachable_differentiable_tensors`
    raises on that. Geometry and index tensors (no grad) may be held as attributes.
    """

    kind: str

    def __call__(self, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor: ...


def check_drive_signature(drive, *, where: str) -> None:
    """Raise `TypeError` unless `drive` is callable as `drive(drivers)`.

    Loud guard against the older `(phi, drivers)` form: a two-argument drive
    would otherwise fail deep inside `PotentialFlowLayer.dp` with an unattributed TypeError.
    """
    try:
        sig = inspect.signature(drive.__call__)
    except (TypeError, ValueError):  # builtins without a signature: nothing to check
        return
    positional = [
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) != 1:
        raise TypeError(
            f"{where}: drive {type(drive).__name__} (kind {getattr(drive, 'kind', '?')!r}) "
            f"must be callable as drive(drivers) -- a Drive is a function of the drivers "
            f"mapping only; got parameters {positional}. A "
            f"drive that used to take (phi, drivers) must drop phi."
        )


class ConstantDrive:
    """A drive that reads a pre-computed, already batched value from `drivers`."""

    def __init__(self, kind: str, key: str) -> None:
        self.kind = kind
        self.key = key

    def __call__(self, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        try:
            return drivers[self.key]
        except KeyError as exc:
            raise KeyError(f"driver {self.key!r} not found") from exc


class Stack:
    """Hydrostatic stack term (CONTAM TN 1887r1 eq. 17 with eq. 64-65 at the path elevation).

    For edge e from node i to node j at elevation z_path, with node reference heights z_ref
    and densities rho (a full-node driver, `drivers[rho_key]`, shape (..., n)):

        value_e = g * ( rho_i (z_ref_i - z_path_e) - rho_j (z_ref_j - z_path_e) )

    so that dp_e = phi_i - phi_j + value_e is the pressure difference AT the opening.
    Geometry is read once at construction and is not differentiable; rho is.
    """

    def __init__(
        self, kind: str, *, src, tgt, z_path, z_ref, rho_key: str = "rho", g: float = 9.80665
    ) -> None:
        self.kind = kind
        self.src = torch.as_tensor(src, dtype=torch.long)
        self.tgt = torch.as_tensor(tgt, dtype=torch.long)
        self.z_path = torch.as_tensor(z_path)
        self.z_ref = torch.as_tensor(z_ref)
        if self.z_path.shape != self.src.shape:
            raise ValueError(
                f"Stack (kind {kind!r}): z_path has shape {tuple(self.z_path.shape)}, "
                f"expected {tuple(self.src.shape)} (one value per edge of the kind)"
            )
        self.rho_key = rho_key
        self.g = float(g)

    @classmethod
    def from_network(
        cls,
        net,
        kind: str,
        *,
        z_path: str = "z_path",
        z_ref: str = "z_ref",
        rho_key: str = "rho",
        g: float = 9.80665,
    ) -> Stack:
        src, tgt = net.endpoints(kind)
        return cls(
            kind,
            src=src,
            tgt=tgt,
            z_path=net.edge_attr(z_path, kind),
            z_ref=net.node_attr(z_ref, default=0.0),
            rho_key=rho_key,
            g=g,
        )

    def __call__(self, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        try:
            rho = drivers[self.rho_key]
        except KeyError as exc:
            raise KeyError(
                f"Stack drive (kind {self.kind!r}): driver {self.rho_key!r} not found"
            ) from exc
        z_path = self.z_path.to(rho.dtype)
        z_ref = self.z_ref.to(rho.dtype)
        head_src = rho[..., self.src] * (z_ref[self.src] - z_path)
        head_tgt = rho[..., self.tgt] * (z_ref[self.tgt] - z_path)
        return self.g * (head_src - head_tgt)


class WindProfile:
    """CONTAM wind pressure profile: Cp versus relative wind angle, periodic piecewise-linear.

    `angles_deg` starts at 0, is strictly increasing and stays below 360; a trailing 360 row
    (as CONTAM writes) is accepted if its Cp equals the first. Differentiable in theta away
    from the knots, which is what a calibration of wind direction needs.
    """

    def __init__(self, angles_deg, cp) -> None:
        ang = torch.as_tensor([float(a) for a in angles_deg], dtype=torch.float64)
        val = torch.as_tensor([float(c) for c in cp], dtype=torch.float64)
        if ang.numel() != val.numel() or ang.numel() < 1:
            raise ValueError("WindProfile: angles and cp must have the same nonzero length")
        if ang.numel() >= 2 and ang[-1].item() == 360.0:
            if abs(val[-1].item() - val[0].item()) > 1e-12:
                raise ValueError(
                    "WindProfile: the closing 360 row must repeat the Cp at 0 "
                    f"({val[0].item()} != {val[-1].item()})"
                )
            ang, val = ang[:-1], val[:-1]
        if ang[0].item() != 0.0:
            raise ValueError("WindProfile: angles must start at 0 degrees")
        if ang.numel() > 1 and not bool((ang[1:] > ang[:-1]).all()):
            raise ValueError("WindProfile: angles must be strictly increasing")
        if bool((ang >= 360.0).any()):
            raise ValueError("WindProfile: angles must be below 360 (the table is periodic)")
        self.angles = torch.cat([ang, ang.new_tensor([360.0])])
        self.cp = torch.cat([val, val[:1]])

    def __call__(self, theta_deg: torch.Tensor) -> torch.Tensor:
        t = torch.remainder(torch.as_tensor(theta_deg), 360.0)
        ang = self.angles.to(t.dtype)
        cp = self.cp.to(t.dtype)
        idx = (torch.searchsorted(ang, t.detach().contiguous(), right=True) - 1).clamp(
            0, ang.numel() - 2
        )
        a0, a1 = ang[idx], ang[idx + 1]
        c0, c1 = cp[idx], cp[idx + 1]
        return c0 + (c1 - c0) * (t - a0) / (a1 - a0)


class Wind:
    """Wind pressure on envelope paths (CONTAM TN 1887r1 section 3.15):

        value_e = sign_e * envelope_e * 0.5 * rho_amb * V_met^2 * Ch_e * Cp_e(theta_w - az_e)

    sign_e is +1 when the edge's source is the ambient node and -1 when its target is, so the
    wind pressure always acts from the ambient side. Cp comes from a `WindProfile` (per-edge
    1-based `profile` index into `profiles`) or, for edges with no profile, the constant edge
    attribute `Cp`. Drivers `rho_amb`, `V_met`, `theta_w` are one value per batch instance.
    """

    def __init__(
        self,
        kind: str,
        *,
        sign,
        envelope,
        azimuth,
        ch,
        cp_const,
        profile_index,
        profiles=(),
        rho_key: str = "rho_amb",
        speed_key: str = "V_met",
        direction_key: str = "theta_w",
    ) -> None:
        self.kind = kind
        self.sign = torch.as_tensor(sign)
        self.envelope = torch.as_tensor(envelope)
        self.azimuth = torch.as_tensor(azimuth)
        self.ch = torch.as_tensor(ch)
        self.cp_const = torch.as_tensor(cp_const)
        self.profile_index = torch.as_tensor(profile_index).to(torch.long)
        self.profiles = list(profiles)
        b = self.sign.shape
        named = (
            ("envelope", self.envelope), ("azimuth", self.azimuth), ("ch", self.ch),
            ("cp_const", self.cp_const), ("profile_index", self.profile_index),
        )
        for name, t in named:
            if t.shape != b:
                raise ValueError(
                    f"Wind (kind {kind!r}): {name} has shape {tuple(t.shape)}, expected "
                    f"{tuple(b)}"
                )
        if self.profile_index.numel() and int(self.profile_index.max()) > len(self.profiles):
            raise ValueError(
                f"Wind (kind {kind!r}): an edge references profile "
                f"{int(self.profile_index.max())} but only {len(self.profiles)} were given"
            )
        # The valid range is 0 (the constant Cp, no profile) to len(self.profiles); a NEGATIVE
        # index matches no `profile_index == i` in `__call__`, where `i` runs from 1, so it
        # would silently fall through to `cp_const` instead of being refused.
        if self.profile_index.numel() and int(self.profile_index.min()) < 0:
            bad = int(self.profile_index.min())
            where = (self.profile_index == bad).nonzero().flatten().tolist()
            raise ValueError(
                f"Wind (kind {kind!r}): edges {where} reference profile {bad}; a profile "
                f"index must be 0 (the constant Cp) or 1..{len(self.profiles)}"
            )
        self.rho_key, self.speed_key, self.direction_key = rho_key, speed_key, direction_key

    @classmethod
    def from_network(
        cls,
        net,
        kind: str,
        *,
        ambient,
        profiles: Sequence[WindProfile] | Mapping[int, WindProfile] = (),
        profile: WindProfile | None = None,
        azimuth: str = "azimuth",
        cp: str = "Cp",
        ch: str = "Ch",
        ch_default: float = 1.0,
        profile_attr: str = "profile",
        rho_key: str = "rho_amb",
        speed_key: str = "V_met",
        direction_key: str = "theta_w",
    ) -> Wind:
        """Build from edge attributes.

        `profiles` is either a plain sequence -- numbered 1..N by position -- or a
        `Mapping[int, WindProfile]` keyed by CONTAM's own profile NUMBER. The
        CONTAM reader (`apps.building_physics.prj`) stores the file's profile number verbatim in
        each edge's `profile` attribute, and the two forms agree only when a project numbers its
        profiles contiguously from 1; a project numbering them, say, 2 and 5 needs the mapping form
        to resolve correctly. Numbers are remapped to dense positions once, here, so the hot path
        (`__call__`) stays a plain index lookup. `profile=` is shorthand: one profile for
        every envelope edge that carries no `profile` attribute at all. `profile = 0` is the
        only value that means "no profile, use the constant `Cp` attribute" -- it is a
        configured, non-silent choice, not a fallback. Every OTHER value an edge's `profile`
        attribute names must resolve in `profiles`, including when `profiles` is empty
        entirely: raises `KeyError` naming the edge and the number. A caller
        who sets `profile=k` on edges and forgets to pass `profiles` must get a loud error,
        not a silently vanishing wind pressure from the constant `Cp` attribute defaulting
        to zero.
        """
        amb = net.node_index(ambient)  # KeyError names the node
        src, tgt = net.endpoints(kind)
        is_src, is_tgt = src == amb, tgt == amb
        one = torch.ones(src.shape, dtype=net.dtype)
        envelope = (is_src | is_tgt).to(net.dtype)
        sign = torch.where(is_src, one, -one)

        numbers: dict[int, WindProfile] = (
            dict(profiles)
            if isinstance(profiles, Mapping)
            else {i + 1: p for i, p in enumerate(profiles)}
        )
        default_number = 0
        if profile is not None:
            default_number = max(numbers, default=0) + 1
            numbers[default_number] = profile

        raw = net.edge_attr(profile_attr, kind, default=float(default_number))
        position = {number: i + 1 for i, number in enumerate(sorted(numbers))}
        profiles_list = [numbers[number] for number in sorted(numbers)]

        cols = net.edge_index(kind).tolist()
        edges = net.edges
        profile_index = []
        for col, value in zip(cols, raw.tolist(), strict=True):
            number = round(value)
            if number == 0:
                profile_index.append(0)
            elif number in position:
                profile_index.append(position[number])
            else:
                raise KeyError(
                    f"Wind.from_network (kind {kind!r}): edge {edges[col]} references "
                    f"profile number {number}, which is not in profiles"
                )

        return cls(
            kind,
            sign=sign,
            envelope=envelope,
            azimuth=net.edge_attr(azimuth, kind, default=0.0),
            ch=net.edge_attr(ch, kind, default=ch_default),
            cp_const=net.edge_attr(cp, kind, default=0.0),
            profile_index=torch.tensor(profile_index, dtype=torch.long),
            profiles=profiles_list,
            rho_key=rho_key,
            speed_key=speed_key,
            direction_key=direction_key,
        )

    def _driver(self, drivers, key):
        try:
            return torch.as_tensor(drivers[key])
        except KeyError as exc:
            raise KeyError(f"Wind drive (kind {self.kind!r}): driver {key!r} not found") from exc

    def __call__(self, drivers: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rho = self._driver(drivers, self.rho_key)[..., None]
        V = self._driver(drivers, self.speed_key)[..., None]
        dtype = V.dtype
        Cp = self.cp_const.to(dtype)
        if self.profiles:
            theta = self._driver(drivers, self.direction_key)[..., None]
            rel = theta - self.azimuth.to(dtype)
            for i, prof in enumerate(self.profiles, start=1):
                Cp = torch.where(self.profile_index == i, prof(rel), Cp)
        return (
            self.sign.to(dtype) * self.envelope.to(dtype) * 0.5 * rho * V**2
            * self.ch.to(dtype) * Cp
        )

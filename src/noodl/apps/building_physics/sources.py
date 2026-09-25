"""CONTAM contaminant source/sink models (TN 1887r1 section 8.2.4) as (t, x) -> kg/s per node.

Each source returns a FULL-node vector with its contribution at its node; `assemble_sources`
sums them into the "species.sources" driver. Units: kg/s (mass fraction transport with zone
air mass as capacity).
"""

from __future__ import annotations

import math

import torch


class _NodeSource:
    def __init__(self, node: int) -> None:
        self.node = int(node)

    def _vector(self, x: torch.Tensor, value) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[..., self.node] = value
        return out


class ConstantSource(_NodeSource):
    """S = G - R x  (constant coefficient model, eq. 15)."""

    def __init__(self, node: int, G: float, R: float = 0.0) -> None:
        super().__init__(node)
        self.G, self.R = float(G), float(R)

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self._vector(x, self.G - self.R * x[..., self.node])


class CutoffSource(_NodeSource):
    """S = max(G (1 - x / x_cut), 0)  (eq. 17), clamped at the cutoff concentration.

    CONTAM's cutoff-concentration source models generation that shuts OFF once the zone
    concentration reaches ``x_cut``; it does not model a sink. The bare eq. 17 formula
    turns negative for ``x > x_cut``, which would silently remove mass never generated --
    wrong physics for anything that runs a cutoff source past its threshold -- so the rate
    is clamped at zero above the cutoff.

    The clamp uses this project's guarded ``torch.where`` idiom (see
    ``elements/powerlaw.py``'s kink): the branch that is *not* selected still gets
    evaluated by ``torch.where``, so it is fed a substituted, safe input (the cutoff value
    itself) rather than the true above-cutoff concentration. That keeps the discarded
    branch's contribution to the backward pass exactly zero everywhere, so no stray
    gradient (let alone a ``nan``) crosses from the clamped region, and the gradient
    below the cutoff -- ``-G / x_cut`` -- is exactly unchanged from the unclamped formula.
    """

    def __init__(self, node: int, G: float, x_cut: float) -> None:
        super().__init__(node)
        self.G, self.x_cut = float(G), float(x_cut)

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        xn = x[..., self.node]
        below = xn <= self.x_cut
        # xn is only trusted where the formula is actually selected; above the cutoff it
        # is replaced by the constant x_cut, so torch.where's unselected branch there
        # carries no gradient dependence on xn at all (see class docstring).
        x_safe = torch.where(below, xn, torch.full_like(xn, self.x_cut))
        rate = self.G * (1.0 - x_safe / self.x_cut)
        return self._vector(x, torch.where(below, rate, torch.zeros_like(xn)))


class DecayingSource(_NodeSource):
    """S = G0 exp(-(t - t0) / tau) for t >= t0, else 0."""

    def __init__(self, node: int, G0: float, tau: float, t0: float = 0.0) -> None:
        super().__init__(node)
        self.G0, self.tau, self.t0 = float(G0), float(tau), float(t0)

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        value = self.G0 * math.exp(-(t - self.t0) / self.tau) if t >= self.t0 else 0.0
        return self._vector(x, value)


class BurstSource(_NodeSource):
    """`mass` delivered uniformly over the step [t_burst, t_burst + dt)."""

    def __init__(self, node: int, mass: float, t_burst: float, dt: float) -> None:
        super().__init__(node)
        self.mass, self.t_burst, self.dt = float(mass), float(t_burst), float(dt)

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        active = self.t_burst <= t < self.t_burst + self.dt
        return self._vector(x, self.mass / self.dt if active else 0.0)


def assemble_sources(sources, t: float, x_full: torch.Tensor) -> torch.Tensor:
    """Sum of every source's contribution, full-node order, shape like `x_full`.

    The output shape comes from `x_full`; there is no separate node-count argument.
    """
    out = torch.zeros_like(x_full)
    for s in sources:
        out = out + s(t, x_full)
    return out


def sources_from_project(project, *, dt: float = 60.0) -> list:
    """The project's `PrjSource` records (from `read_prj`) -> source objects.

    CONTAM source element data lines: ccf: G R ...; cut: G x_cut ...; eds: G0 k ... (k the
    decay rate, tau = 1/k); brs: M ... (verify against TN 1887r1 Appendix A section 9 when
    a project with sources is loaded).

    A source's `z#` is resolved through `Project.zone_nr_to_name` -- NOT an unchecked
    `project.zones[s.zone_nr - 1]` -- raising `KeyError` naming
    both the source and the zone number when the project never defines that zone.

    The source-type dispatch ends in `else: raise ValueError`, naming both the
    source and its type, rather than a silent fallthrough that would drop an
    unrecognised source from the model with no error at all.
    """
    out = []
    for s in project.sources:
        try:
            zone_name = project.zone_nr_to_name[s.zone_nr]
        except KeyError as exc:
            raise KeyError(
                f"sources: source {s.nr} references zone {s.zone_nr}, which the project "
                f"does not define"
            ) from exc
        node = project.net.node_index(zone_name)
        p = s.params
        if s.source_type == "ccf":
            out.append(ConstantSource(node, s.mult * p[0], s.mult * p[1]))
        elif s.source_type == "cut":
            out.append(CutoffSource(node, s.mult * p[0], p[1]))
        elif s.source_type == "eds":
            out.append(DecayingSource(node, s.mult * p[0], 1.0 / p[1] if p[1] > 0 else math.inf))
        elif s.source_type == "brs":
            out.append(BurstSource(node, s.mult * p[0], 0.0, dt))
        else:
            raise ValueError(
                f"sources: source {s.nr} uses unsupported source type {s.source_type!r}"
            )
    return out

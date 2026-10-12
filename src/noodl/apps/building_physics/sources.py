"""CONTAM's contaminant source/sink models as source objects for `SpeciesSources`.

A source is either a plain callable `source(t, x_full) -> kg/s per node` (evaluated on the
step-start mass fractions), or an object with `generation(t, x_full, pressure_difference)
-> kg/s` and, optionally, `removal(t, x_full, air_mass, density) -> kg/s per unit mass
fraction`. The second form is CONTAM's S = G - R C: `SpeciesSources` divides R by each
zone's air mass and hands it to the species step as the driver `"<layer>.removal"`, where
it is implicit, as ContamX holds it (measured against ContamX 3.4.1.7). Every vector is
in full node order. Units: kg/s, mass fractions kg/kg, air mass kg, density kg/m3.
"""

from __future__ import annotations

import math

import torch

F64 = torch.float64


class _NodeSource:
    def __init__(self, node: int) -> None:
        self.node = int(node)

    def _vector(self, x: torch.Tensor, value) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[..., self.node] = value
        return out


class _GenerationRemoval(_NodeSource):
    """A source whose removal does not depend on the zone's air mass. Called directly it is
    S = G - R x at the x given, derived from its `generation` and `removal`."""

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.generation(t, x) - self.removal(t, x, None, None) * x


class ConstantSource(_GenerationRemoval):
    """S = G - R x (constant coefficient model): G kg/s, R kg/s per unit mass fraction
    (CONTAM's kg_air/s). Under `SpeciesSources` the term R x is implicit in the species
    step, as ContamX holds it; called directly the source is evaluated at the x given."""

    def __init__(self, node: int, G: float, R: float = 0.0) -> None:
        super().__init__(node)
        self.G, self.R = float(G), float(R)

    def generation(self, t: float, x: torch.Tensor, pressure_difference=None) -> torch.Tensor:
        """G at the node (kg/s)."""
        return self._vector(x, self.G)

    def removal(self, t: float, x: torch.Tensor, air_mass, density) -> torch.Tensor:
        """R at the node (kg/s per unit mass fraction)."""
        return self._vector(x, self.R)


class CutoffSource(_GenerationRemoval):
    """S = G (1 - x / x_cut) (cutoff concentration model): G kg/s, x_cut kg/kg.

    Above the cutoff the source removes: ContamX keeps the linear law there (measured: a
    sealed zone at twice the cutoff loses the contaminant to it), and so does this class.
    Under `SpeciesSources` the term (G / x_cut) x is implicit in the species step, as in
    ContamX.
    """

    def __init__(self, node: int, G: float, x_cut: float) -> None:
        super().__init__(node)
        self.G, self.x_cut = float(G), float(x_cut)

    def generation(self, t: float, x: torch.Tensor, pressure_difference=None) -> torch.Tensor:
        """G at the node (kg/s)."""
        return self._vector(x, self.G)

    def removal(self, t: float, x: torch.Tensor, air_mass, density) -> torch.Tensor:
        """G / x_cut at the node (kg/s per unit mass fraction)."""
        return self._vector(x, self.G / self.x_cut)


class DecayingSource(_NodeSource):
    """S = G0 exp(-(t - t0) / tau) for t >= t0, else 0."""

    def __init__(self, node: int, G0: float, tau: float, t0: float = 0.0) -> None:
        super().__init__(node)
        self.G0, self.tau, self.t0 = float(G0), float(tau), float(t0)

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        value = self.G0 * math.exp(-(t - self.t0) / self.tau) if t >= self.t0 else 0.0
        return self._vector(x, value)


class BurstSource(_NodeSource):
    """`mass` delivered uniformly over `[t_burst, t_burst + dt)`.

    `dt` is the burst's duration; `sources_from_project` sets it to the run control's time
    step. Called with the length of the step that ends at `t` (`step`, as `SpeciesSources`
    passes it from `drivers["time_step"]`), it returns the step-averaged rate: the mass
    released over the part of the step that overlaps the burst, divided by the step length, so
    the mass is delivered in full whatever the step length (a coarser `simulate` step spreads
    the same mass over its own length) and is never missed by sampling the step's end. Without
    `step` it returns the instantaneous rate at `t`.
    """

    step_integrated = True

    def __init__(self, node: int, mass: float, t_burst: float, dt: float) -> None:
        super().__init__(node)
        self.mass, self.t_burst, self.dt = float(mass), float(t_burst), float(dt)

    def __call__(self, t: float, x: torch.Tensor, step: float | None = None) -> torch.Tensor:
        if step is None:
            active = self.t_burst <= t < self.t_burst + self.dt
            return self._vector(x, self.mass / self.dt if active else 0.0)
        if not step > 0.0:
            raise ValueError(f"BurstSource: the step length must be positive, got {step!r} s")
        overlap = min(t, self.t_burst + self.dt) - max(t - step, self.t_burst)
        rate = self.mass / self.dt * max(overlap, 0.0) / step
        return self._vector(x, rate)


class DepositionVelocitySink(_NodeSource):
    """Deposition onto a surface at a deposition velocity (CONTAM's deposition velocity
    sink): R = v_d A rho, a first-order rate v_d A / V on the zone's mass fraction.
    `velocity` v_d in m/s, `area` A in m2 (floats or tensors)."""

    def __init__(self, node: int, velocity, area) -> None:
        super().__init__(node)
        self.velocity = torch.as_tensor(velocity, dtype=F64)
        self.area = torch.as_tensor(area, dtype=F64)

    def generation(self, t: float, x: torch.Tensor, pressure_difference=None) -> torch.Tensor:
        """Nothing (kg/s)."""
        return torch.zeros_like(x)

    def removal(self, t: float, x: torch.Tensor, air_mass, density) -> torch.Tensor:
        """v_d A rho at the node (kg/s per unit mass fraction)."""
        return self._vector(x, self.velocity * self.area * density[..., self.node])


class DepositionRateSink(_NodeSource):
    """First-order removal at a deposition rate (CONTAM's deposition rate sink): R = k_d
    rho V, the rate k_d (1/s) on the zone's mass fraction; `rate` a float or tensor. It is
    also how to give a species a first-order decay in a zone."""

    def __init__(self, node: int, rate) -> None:
        super().__init__(node)
        self.rate = torch.as_tensor(rate, dtype=F64)

    def generation(self, t: float, x: torch.Tensor, pressure_difference=None) -> torch.Tensor:
        """Nothing (kg/s)."""
        return torch.zeros_like(x)

    def removal(self, t: float, x: torch.Tensor, air_mass, density) -> torch.Tensor:
        """k_d times the zone's air mass (kg/s per unit mass fraction)."""
        return self._vector(x, self.rate * air_mass[..., self.node])


class PowerLawDecaySource(_NodeSource):
    """Emission decaying as a power of time (the NRCC power-law model): S = a t_h^(-b) after
    t_p and a t_ph^(-b) before it, t_h being the time since the start in HOURS (measured:
    ContamX takes the power of the time in hours, although its file stores t_p in s).
    `coefficient` a kg/s, `exponent` b, `t_p` s (> 0: the power has no value at t = 0)."""

    def __init__(self, node: int, coefficient, exponent, t_p: float) -> None:
        super().__init__(node)
        if not float(t_p) > 0.0:
            raise ValueError(f"PowerLawDecaySource: t_p must be positive (the power law has "
                             f"no value at t = 0), got {t_p}")
        self.coefficient = torch.as_tensor(coefficient, dtype=F64)
        self.exponent = torch.as_tensor(exponent, dtype=F64)
        self.t_p = float(t_p)

    def generation(self, t: float, x: torch.Tensor, pressure_difference=None) -> torch.Tensor:
        """a (max(t, t_p) / 3600 s)^(-b) at the node (kg/s)."""
        hours = max(float(t), self.t_p) / 3600.0
        return self._vector(x, self.coefficient * hours ** (-self.exponent))

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.generation(t, x)


class PeakEmissionSource(_NodeSource):
    """Emission rising to a peak and decaying (the NRCC peak model): S = a exp(-(ln(t /
    t_p) / b)^2 / 2), t the time since the start (s), nothing at t <= 0. `peak` a kg/s,
    `width` b, `t_peak` t_p s."""

    def __init__(self, node: int, peak, width, t_peak: float) -> None:
        super().__init__(node)
        self.peak = torch.as_tensor(peak, dtype=F64)
        self.width = torch.as_tensor(width, dtype=F64)
        self.t_peak = float(t_peak)

    def generation(self, t: float, x: torch.Tensor, pressure_difference=None) -> torch.Tensor:
        """The peak curve at t, at the node (kg/s)."""
        if t <= 0.0:
            return torch.zeros_like(x)
        z = math.log(t / self.t_peak) / self.width
        return self._vector(x, self.peak * torch.exp(-0.5 * z * z))

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.generation(t, x)


class SpeciesSources:
    """Closure: every source's generation summed into `"<layer>.sources"` (kg/s, full node
    order) and every removal coefficient into `"<layer>.removal"` (1/s, active interior
    order), each times its entry of `"<layer>.source_multiplier"`, on its own species.

    `sources` follow the protocol of this module: plain callables `source(t, x_full)`, or
    objects with `generation` and optionally `removal`. `species[i]` is source i's 0-based
    species, `names[i]` its label (default "source 1", ...). They are evaluated at
    `drivers["time"]` (s since the start; `simulate` sets the step's end, which is when
    ContamX evaluates them) on the step-start mass fractions. A `BurstSource` also gets the
    step length, `drivers["time_step"]` (s, set by `simulate`), and releases its mass over
    the steps that overlap its burst. A removal coefficient R (kg/s per unit mass fraction)
    is divided by the zone's air mass, the layer's capacity for the step
    (`"<layer>.capacity"` when a closure writes it), and the species step holds it
    implicitly; the density a source sees is that air mass over the node's volume. The
    multiplier (one per source, batched like any driver) is where a schedule or a control
    acts: CONTAM's S = mult ctrl (G - R C), a sink included.
    """

    def __init__(self, sources, *, species, layer, names=None) -> None:
        self.sources, self.species = list(sources), [int(k) for k in species]
        if len(self.species) != len(self.sources):
            raise ValueError("SpeciesSources: one species index per source")
        self.layer = layer
        self.names = list(names) if names is not None else [
            f"source {i + 1}" for i in range(len(self.sources))]
        self.key = f"{layer.name}.source_multiplier"
        self.capacity_key = f"{layer.name}.capacity"
        self.removes = any(hasattr(s, "removal") for s in self.sources)
        self.volumes = torch.as_tensor(layer.net.node_attr("volume", default=0.0), dtype=F64)
        self.inputs = ("time", self.key)
        self.outputs = (f"{layer.name}.sources",) + (
            (f"{layer.name}.removal",) if self.removes else ())

    @property
    def input_specs(self) -> dict[str, dict]:
        """The time and the per-source multipliers this closure reads."""
        return {
            "time": dict(description="time since the simulation start (end of the step)",
                         unit="s"),
            "time_step": dict(description="length of the step that ends at `time` (read by "
                                          "burst sources, which without it sample the step's "
                                          "end; `simulate` sets it)", unit="s",
                              required=False),
            self.key: dict(description="multiplier of each source (a schedule or control)",
                           unit="-", over=self.names),
        }

    def __call__(self, state, drivers):
        lay, name = self.layer, self.layer.name
        try:
            t = float(drivers["time"])
        except KeyError as exc:
            raise KeyError("SpeciesSources: driver 'time' (s since the start) is required; "
                           "simulate sets it, a hand-driven model.step must too") from exc
        step = drivers.get("time_step")
        step = None if step is None else float(step)
        K, n = lay.n_species, lay.net.n
        x_i = state[f"{name}.x"]
        axis = x_i.dim() >= 2 and x_i.shape[-2] == lay.n_i and x_i.shape[-1] == K
        x_i = x_i if axis else x_i.unsqueeze(-1)
        xb = torch.as_tensor(drivers[f"{name}.x_boundary"], dtype=x_i.dtype)
        xb = xb if xb.dim() >= 2 and xb.shape[-1] == K else xb.unsqueeze(-1)
        mult = torch.as_tensor(drivers[self.key], dtype=x_i.dtype)
        cap = torch.as_tensor(drivers.get(self.capacity_key, lay.capacity), dtype=x_i.dtype)
        batch = torch.broadcast_shapes(x_i.shape[:-2], xb.shape[:-2], mult.shape[:-1],
                                       cap.shape[:-1])
        full = torch.zeros(batch + (n, K), dtype=x_i.dtype)
        full = full.index_copy(-2, lay.interior_idx, x_i.expand(batch + x_i.shape[-2:]))
        full = full.index_copy(-2, lay.boundary_idx, xb.expand(batch + xb.shape[-2:]))
        cap = cap.expand(batch + (lay.n_i,))
        mass = torch.zeros(batch + (n,), dtype=x_i.dtype).index_copy(-1, lay.interior_idx, cap)
        volume = torch.where(self.volumes > 0, self.volumes, torch.ones_like(self.volumes))
        density = mass / volume
        gen = [torch.zeros(batch + (n,), dtype=x_i.dtype) for _ in range(K)]
        rem = [torch.zeros(batch + (n,), dtype=x_i.dtype) for _ in range(K)]
        for j, (src, k) in enumerate(zip(self.sources, self.species, strict=True)):
            m, xk = mult[..., j:j + 1], full[..., k]
            if hasattr(src, "generation"):
                gen[k] = gen[k] + m * src.generation(t, xk, None)
            else:
                kw = {"step": step} if getattr(src, "step_integrated", False) else {}
                gen[k] = gen[k] + m * src(t, xk, **kw)
            if hasattr(src, "removal"):
                rem[k] = rem[k] + m * src.removal(t, xk, mass, density)
        sources = torch.stack(gen, dim=-1)
        out = {f"{name}.sources": sources if axis else sources[..., 0]}
        if self.removes:
            R = torch.stack(rem, dim=-1).index_select(-2, lay.interior_idx)
            out[f"{name}.removal"] = R / cap.unsqueeze(-1)
        return out


def assemble_sources(sources, t: float, x_full: torch.Tensor) -> torch.Tensor:
    """Sum of every source's explicit rate `source(t, x_full)` (kg/s), full-node order,
    shape like `x_full`; there is no separate node-count argument.

    A source with a concentration-proportional part (`ConstantSource`, `CutoffSource`) is
    evaluated at `x_full`. A sink, whose removal needs the zone's air mass, has no such rate
    and is refused by name: use `SpeciesSources` in a model.
    """
    out = torch.zeros_like(x_full)
    for i, s in enumerate(sources):
        if not callable(s):
            raise TypeError(f"assemble_sources: source {i + 1} ({type(s).__name__}) has no "
                            f"explicit rate source(t, x): its removal needs the zone's air "
                            f"mass; use SpeciesSources in a model")
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

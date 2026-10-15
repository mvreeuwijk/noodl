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
        if not float(x_cut) > 0.0:
            raise ValueError(f"CutoffSource: x_cut must be positive (kg/kg), got {x_cut}")
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
    """`mass` delivered over the step(s) covering `[t_burst, t_burst + dt)`.

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
    `velocity` v_d in m/s, `area` A in m2 (floats or tensors). Its node needs a positive
    volume: the density is the zone's air mass over it."""

    needs_volume = True

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


class PressureDrivenSource(_NodeSource):
    """Entry driven by the pressure difference across the zone's envelope (CONTAM's
    pressure-driven model: radon, soil gas): S = G max(dP, 0)^n, dP = P_ambient - P_zone at
    the zone's reference height (Pa). `coefficient` G in kg/(s Pa^n), `exponent` n. Nothing
    enters while the zone is at or above the ambient pressure (measured against ContamX
    3.4.1.7, which evaluates it at the step's own pressures)."""

    needs_pressure_difference = True

    def __init__(self, node: int, coefficient, exponent) -> None:
        super().__init__(node)
        self.coefficient = torch.as_tensor(coefficient, dtype=F64)
        self.exponent = torch.as_tensor(exponent, dtype=F64)

    def generation(self, t: float, x: torch.Tensor, pressure_difference) -> torch.Tensor:
        """G max(dP, 0)^n at the node (kg/s)."""
        dp = pressure_difference[..., self.node]
        entering = dp > 0
        safe = torch.where(entering, dp, torch.ones_like(dp))
        rate = torch.where(entering, self.coefficient * safe ** self.exponent,
                           torch.zeros_like(dp))
        return self._vector(x, rate)


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
        if not self.t_peak > 0.0:
            raise ValueError(f"PeakEmissionSource: t_peak must be positive (s), got {t_peak}")
        if not bool((self.width.detach() > 0.0).all()):
            raise ValueError(f"PeakEmissionSource: width must be positive, got "
                             f"{self.width.detach().tolist()}")

    def generation(self, t: float, x: torch.Tensor, pressure_difference=None) -> torch.Tensor:
        """The peak curve at t, at the node (kg/s)."""
        if t <= 0.0:
            return torch.zeros_like(x)
        z = math.log(t / self.t_peak) / self.width
        return self._vector(x, self.peak * torch.exp(-0.5 * z * z))

    def __call__(self, t: float, x: torch.Tensor) -> torch.Tensor:
        return self.generation(t, x)


def _check_emitting_node(layer, node: int, what: str,
                         use: str = "emit into or remove from") -> None:
    """Refuse `node` (an index into `layer.net.nodes`) as a place to `use` the zone air
    (emit into or remove from, breathe): the layer's boundary (the ambient), a node no flow
    edge of the layer touches, or one of its storage-free nodes. `what` names the emitter or
    the occupant in the message."""
    nodes = layer.net.nodes
    name = nodes[node]
    if node in set(layer.boundary_idx.tolist()):
        raise ValueError(f"{what} is on the boundary node {name!r}: a boundary (the "
                         f"ambient) holds no zone air to {use}")
    if node in set(layer.inactive_idx.tolist()):
        raise ValueError(f"{what} is on node {name!r}, which no flow edge of the layer "
                         f"{layer.name!r} touches: the layer holds no concentration there")
    if node in set(layer.interior_idx[layer.storage_free_mask].tolist()):
        raise ValueError(f"{what} is on the storage-free node {name!r} of the layer "
                         f"{layer.name!r}: it holds no air to {use}, so put it in a zone "
                         f"that stores air")


def _add_sources(base, own: torch.Tensor, *, own_axis: bool, n_species: int) -> torch.Tensor:
    """`own` (kg/s, full node order) added to `base`, what `"<layer>.sources"` already
    holds (None: nothing). With one species either may lack the species axis; the sum keeps
    `base`'s layout."""
    if base is None:
        return own
    base = torch.as_tensor(base, dtype=own.dtype)
    if n_species == 1:
        base_axis = base.shape[-1] == 1         # a network has at least two nodes
        if base_axis and not own_axis:
            own = own.unsqueeze(-1)
        elif own_axis and not base_axis:
            own = own[..., 0]
    return base + own


class SpeciesSources:
    """Closure: every source's generation summed into `"<layer>.sources"` (kg/s, full node
    order), added to what an earlier closure or a driver put there, and every removal
    coefficient into `"<layer>.removal"` (1/s, active interior order), each times its entry
    of `"<layer>.source_multiplier"`, on its own species.

    `sources` follow the protocol of this module: plain callables `source(t, x_full)`, or
    objects with `generation` and optionally `removal`; a source with `step_integrated = True`
    also receives `step=` (its callable, or its `generation`). `species[i]` is source i's 0-based
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

    The generation is added, not written over: an `OccupantGeneration` on the same layer
    may run before or after this closure. The per-step `"<layer>.removal"` replaces the
    layer's construction-time `removal`, so a layer with a nonzero one is refused when a
    source has a removal part. Refused by name at construction too: an object that is
    neither a plain callable nor has `generation`; a source on a boundary node (the ambient
    holds no zone air), on a node no flow edge of the layer touches, or on a storage-free
    node of the layer (a zero-volume plenum holds no air); a deposition velocity sink on a
    node without a positive volume.

    A source with `needs_pressure_difference = True` (the pressure-driven source) gets
    P_ambient - P at each node's reference height (Pa), from `state[pressure_key]` with
    `heights` (m, one per node), `g` (m/s2) and `drivers["rho_amb"]` (kg/m3), the ambient
    being the layer's one boundary node (refused otherwise), and zero before the first
    airflow solve. Such a closure
    reports `reads_solved_pressure`, and `simulate` and `prj.steady` repeat a step until the
    pressure it read is the step's own.

    The closure reads `"<layer>.capacity"` and `"rho_amb"`, which the density closures
    write: place it after the closures that write density and capacity, or it uses the
    construction-time capacity and a stale density.
    """

    def __init__(self, sources, *, species, layer, names=None, heights=None,
                 g: float = 9.80665, pressure_key: str = "air.phi") -> None:
        self.sources, self.species = list(sources), [int(k) for k in species]
        if len(self.species) != len(self.sources):
            raise ValueError("SpeciesSources: one species index per source")
        self.layer = layer
        self.names = list(names) if names is not None else [
            f"source {i + 1}" for i in range(len(self.sources))]
        self.key = f"{layer.name}.source_multiplier"
        self.capacity_key = f"{layer.name}.capacity"
        self.removes = any(self._can_remove(s) for s in self.sources)
        self.reads_solved_pressure = any(
            getattr(s, "needs_pressure_difference", False) for s in self.sources)
        if self.reads_solved_pressure and heights is None:
            raise ValueError("SpeciesSources: a pressure-driven source needs `heights`, every "
                             "node's reference height (m), to place the ambient pressure at "
                             "its zone")
        self.heights = None if heights is None else torch.as_tensor(heights, dtype=F64)
        self.g, self.pressure_key = float(g), pressure_key
        if self.reads_solved_pressure:
            self._check_pressure_reading()
        self.volumes = torch.as_tensor(layer.net.node_attr("volume", default=0.0), dtype=F64)
        self._check_sources()
        self.inputs = ("time", self.key, self.capacity_key, f"{layer.name}.sources") + (
            ("rho_amb",) if self.reads_solved_pressure else ())
        self.outputs = (f"{layer.name}.sources",) + (
            (f"{layer.name}.removal",) if self.removes else ())

    @staticmethod
    def _can_remove(src) -> bool:
        """Whether the source has a removal part that can act: a sink, or a constant
        coefficient that is not zero."""
        if not hasattr(src, "removal"):
            return False
        return not (isinstance(src, ConstantSource) and src.R == 0.0)

    def _check_pressure_reading(self) -> None:
        """The ambient is the layer's one boundary node, and `heights` has one entry per
        node: the pressure difference is measured from that node's static pressure."""
        lay = self.layer
        if int(lay.boundary_idx.numel()) != 1:
            names = [lay.net.nodes[int(i)] for i in lay.boundary_idx.tolist()]
            raise ValueError(f"SpeciesSources: a pressure-driven source measures its pressure "
                             f"difference from the ambient, the layer {lay.name!r}'s one "
                             f"boundary node; the layer has the boundary nodes {names}")
        if tuple(self.heights.shape) != (lay.net.n,):
            raise ValueError(f"SpeciesSources: `heights` needs one reference height (m) per "
                             f"node, {lay.net.n}, got shape {tuple(self.heights.shape)}")

    def _check_sources(self) -> None:
        lay = self.layer
        nodes = lay.net.nodes
        for label, src in zip(self.names, self.sources, strict=True):
            what = f"SpeciesSources: {label} ({type(src).__name__})"
            if not (hasattr(src, "generation") or callable(src)):
                raise TypeError(f"{what} is neither a callable source(t, x_full) nor an object "
                                f"with generation(t, x, pressure_difference)")
            if not hasattr(src, "node"):
                continue
            node = int(src.node)
            if not 0 <= node < len(nodes):
                raise ValueError(f"{what}: node {node} is not in the network")
            _check_emitting_node(lay, node, what)
            name = nodes[node]
            if getattr(src, "needs_volume", False) and not self.volumes[node] > 0:
                raise ValueError(f"{what} is on node {name!r}, which has no positive volume: "
                                 f"the density the sink needs is the air mass over it")
        own = lay.removal
        if self.removes and own is not None and bool((own != 0).any()):
            raise ValueError(f"SpeciesSources: the layer {lay.name!r} has a nonzero "
                             f"construction-time removal, which the per-step "
                             f"'{lay.name}.removal' the sources write would replace; put that "
                             f"removal in a DepositionRateSink instead")

    @property
    def input_specs(self) -> dict[str, dict]:
        """The time and the per-source multipliers this closure reads."""
        extra = {} if not self.reads_solved_pressure else {
            "rho_amb": dict(description="ambient air density, written by a density closure "
                                        "(places the ambient pressure at each zone's height)",
                            unit="kg/m3", required=False)}
        return {
            **extra,
            "time": dict(description="time since the simulation start (end of the step)",
                         unit="s"),
            "time_step": dict(description="length of the step that ends at `time` (read by "
                                          "burst sources, which without it sample the step's "
                                          "end; `simulate` sets it)", unit="s",
                              required=False),
            self.key: dict(description="multiplier of each source (a schedule or control)",
                           unit="-", over=self.names),
        }

    def _pressure_difference(self, state, drivers) -> torch.Tensor | None:
        """P_ambient(z) - P(z) at every node (Pa): the ambient's static pressure at the
        node's reference height minus the node's own, from `state[pressure_key]` (the
        potential at each node's reference height). None before the first airflow solve."""
        phi = state.get(self.pressure_key)
        if phi is None:
            return None
        if "rho_amb" not in drivers:
            raise KeyError("SpeciesSources: driver 'rho_amb' (the ambient air density, kg/m3) "
                           "is required by a pressure-driven source, to place the ambient's "
                           "static pressure at each zone's height; a density closure writes it")
        amb = int(self.layer.boundary_idx[0])
        rho_amb = torch.as_tensor(drivers["rho_amb"], dtype=F64).unsqueeze(-1)
        z = self.heights
        return phi[..., amb:amb + 1] - rho_amb * self.g * (z - z[amb]) - phi

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
        dp = self._pressure_difference(state, drivers) if self.reads_solved_pressure else None
        batch = torch.broadcast_shapes(x_i.shape[:-2], xb.shape[:-2], mult.shape[:-1],
                                       cap.shape[:-1], *(() if dp is None else (dp.shape[:-1],)))
        if self.reads_solved_pressure and dp is None:
            dp = torch.zeros(batch + (n,), dtype=x_i.dtype)   # no entry before the first solve
        full = torch.zeros(batch + (n, K), dtype=x_i.dtype)
        full = full.index_copy(-2, lay.interior_idx, x_i.expand(batch + x_i.shape[-2:]))
        full = full.index_copy(-2, lay.boundary_idx, xb.expand(batch + xb.shape[-2:]))
        cap = cap.expand(batch + (lay.n_i,))
        mass = torch.zeros(batch + (n,), dtype=x_i.dtype).index_copy(-1, lay.interior_idx, cap)
        # Every node a density-reading sink sits on has a positive volume (checked at
        # construction); the 1 elsewhere only keeps 0 / 0 out of the unread entries.
        volume = torch.where(self.volumes > 0, self.volumes, torch.ones_like(self.volumes))
        density = mass / volume
        gen = [torch.zeros(batch + (n,), dtype=x_i.dtype) for _ in range(K)]
        rem = [torch.zeros(batch + (n,), dtype=x_i.dtype) for _ in range(K)]
        for j, (src, k) in enumerate(zip(self.sources, self.species, strict=True)):
            m, xk = mult[..., j:j + 1], full[..., k]
            kw = {"step": step} if getattr(src, "step_integrated", False) else {}
            if hasattr(src, "generation"):
                gen[k] = gen[k] + m * src.generation(t, xk, dp, **kw)
            else:
                gen[k] = gen[k] + m * src(t, xk, **kw)
            if hasattr(src, "removal"):
                rem[k] = rem[k] + m * src.removal(t, xk, mass, density)
        sources = torch.stack(gen, dim=-1)
        own = sources if axis else sources[..., 0]
        out = {f"{name}.sources": _add_sources(drivers.get(f"{name}.sources"), own,
                                               own_axis=axis, n_species=K)}
        if self.removes:
            R = torch.stack(rem, dim=-1).index_select(-2, lay.interior_idx)
            # A storage-free node may have zero capacity, and no source sits on one (refused
            # at construction): the 1 there only keeps 0 / 0 out of its removal rate.
            held = torch.where(cap > 0, cap, torch.ones_like(cap))
            out[f"{name}.removal"] = R / held.unsqueeze(-1)
        return out


def assemble_sources(sources, t: float, x_full: torch.Tensor) -> torch.Tensor:
    """Sum of every source's explicit rate `source(t, x_full)` (kg/s), full-node order,
    shape like `x_full`; there is no separate node-count argument.

    A source with a concentration-proportional part (`ConstantSource`, `CutoffSource`) is
    evaluated at `x_full`. A sink or a pressure-driven source has no such rate and is
    refused by name: use `SpeciesSources` in a model.
    """
    out = torch.zeros_like(x_full)
    for i, s in enumerate(sources):
        if not callable(s):
            raise TypeError(f"assemble_sources: source {i + 1} ({type(s).__name__}) has no "
                            f"explicit rate source(t, x) (a sink or a pressure-driven "
                            f"source); use SpeciesSources in a model")
        out = out + s(t, x_full)
    return out


def sources_from_project(project, *, dt: float = 60.0) -> list:
    """The project's `PrjSource` records (from `read_prj`) -> source objects.

    CONTAM source/sink element data lines (TN 1887r1 Appendix A section 9): ccf: G R;
    cut: G x_cut; eds: G0 k (tau = 1/k); brs: M; prs: G n; dvs: v_d A; drs: k_d; plm:
    a b t_p; pkm: a b t_p. The record's multiplier scales the first coefficient (G, G0, M,
    v_d, k_d, a), and a constant-coefficient source's removal R too, which scales the source
    or the sink as CONTAM's `mult` does.

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
        elif s.source_type == "prs":
            out.append(PressureDrivenSource(node, s.mult * p[0], p[1]))
        elif s.source_type == "dvs":
            out.append(DepositionVelocitySink(node, s.mult * p[0], p[1]))
        elif s.source_type == "drs":
            out.append(DepositionRateSink(node, s.mult * p[0]))
        elif s.source_type == "plm":
            out.append(PowerLawDecaySource(node, s.mult * p[0], p[1], p[2]))
        elif s.source_type == "pkm":
            out.append(PeakEmissionSource(node, s.mult * p[0], p[1], p[2]))
        else:
            raise ValueError(
                f"sources: source {s.nr} uses unsupported source type {s.source_type!r}"
            )
    return out

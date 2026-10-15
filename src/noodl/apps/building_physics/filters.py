"""Contaminant filters on airflow paths.

A filter removes the fraction `efficiency` of each species from the air passing through
its edge. The species layer holds it as its `transmission`, the pass fraction 1 - efficiency
per edge and species. CONTAM's constant-efficiency, simple particle and super filters (TN 1887
Appendix A section 8a) remove a fixed fraction in whichever direction the air flows
(measured against ContamX 3.4.1.7); `filter_transmission` builds the fixed transmission from
them. CONTAM's Penn State UVGI filter (`UVGIFilter`) removes a fraction that depends on the
air passing (its mass flow, temperature and density) and on the lamp's age, so its
transmission changes with every airflow solve: `FilterTransmission` is the closure that
writes it as the species layer's per-step `"<layer>.transmission"` driver. Filters whose
efficiency depends on their loading are not modelled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from noodl.apps.building_physics.species import Species
from noodl.elements.curve import SplineCurve
from noodl.topology import EdgeKey, Network

Tensor = torch.Tensor
F64 = torch.float64

#: The standard air density (kg/m3) with which a UVGI filter's design velocity and design mass
#: flow give its cross-sectional flow area (TN 1887r1: A = F_design / (rho_std u_design)).
STANDARD_AIR_DENSITY = 1.2041
#: The UV dose (J/m2) ContamX gives a UVGI filter when the air crosses its path against the
#: path's direction, whatever the filter's data (measured): eta = 1 - exp(-k).
REVERSE_FLOW_DOSE = 1.0


@dataclass(frozen=True)
class AirThroughFilter:
    """The air passing a filter: what a flow-dependent filter's efficiency reads.

    `mass_flow` (kg/s, signed: positive in the direction of the edge the filter sits on,
    from its source node to its target), `temperature` (K) and `density` (kg/m3) of the air
    at the edge's source node, and `time` (s since the start of the run). Tensors of a
    common batch shape `(...)`; `time` may be a scalar."""

    mass_flow: Tensor
    temperature: Tensor
    density: Tensor
    time: Tensor | float = 0.0


class ConstantEfficiencyFilter:
    """Removes a fixed fraction of each species it lists, nothing of the others.

    `efficiencies` maps species names to fractions in [0, 1] (floats or tensors)."""

    def __init__(self, efficiencies: Mapping[str, float | Tensor], *, name: str = "") -> None:
        self.efficiencies = {s: torch.as_tensor(e, dtype=F64) for s, e in efficiencies.items()}
        bad = {s: e.item() for s, e in self.efficiencies.items()
               if not 0.0 <= e.detach().item() <= 1.0}
        if bad:
            raise ValueError(f"ConstantEfficiencyFilter {name!r}: efficiencies must be "
                             f"between 0 and 1, got {bad}")
        self.name = name

    #: The efficiency does not depend on the air passing.
    depends_on_flow = False

    def efficiency(self, species: Sequence[Species], air: AirThroughFilter | None = None
                   ) -> Tensor:
        """`(K,)`: each species' removed fraction, in `species` order. `air` is not read."""
        names = [s.name for s in species]
        unknown = sorted(set(self.efficiencies) - set(names))
        if unknown:
            raise KeyError(f"ConstantEfficiencyFilter {self.name!r}: species {unknown} are not "
                           f"among the model's species {names}")
        zero = torch.zeros((), dtype=F64)
        return torch.stack([self.efficiencies.get(n, zero) for n in names])


def _points(values, label: str) -> Tensor:
    """A curve's values as one float64 vector; a list of tensors is stacked, so their
    gradients survive (`torch.as_tensor` would copy them out of the graph)."""
    if isinstance(values, (list, tuple)) and any(isinstance(v, Tensor) for v in values):
        parts = [torch.as_tensor(v, dtype=F64) for v in values]
        if any(p.dim() != 0 for p in parts):
            raise ValueError(f"{label}: a list of curve values must hold scalars, got shapes "
                             f"{[tuple(p.shape) for p in parts]}")
        return torch.stack(parts)
    return torch.as_tensor(values, dtype=F64).reshape(-1)


class ParticleFilter:
    """Efficiency as a function of particle diameter (CONTAM's simple particle filter, e.g.
    a MERV curve); zero for a gas.

    `diameters` (m, increasing) and `efficiencies` (0..1) are the curve's points. As in
    ContamX 3.4.1.7 (measured): ContamX's curve fit (`noodl.elements.SplineCurve`) in
    log10(diameter); ContamX refuses a curve whose first two or last two efficiencies
    differ, and so does this class. Efficiencies may be tensors that carry gradients; they
    flow through the shared fit's slopes as well as its values (the values are unchanged).
    """

    def __init__(self, diameters, efficiencies, *, name: str = "") -> None:
        label = f"ParticleFilter {name!r}"
        d, e = _points(diameters, label), _points(efficiencies, label)
        if d.shape != e.shape or d.numel() < 2:
            raise ValueError(f"{label}: diameters and efficiencies must be two lists of "
                             f"equal length, at least 2 points")
        dd, ee = d.detach(), e.detach()
        if bool((dd <= 0).any()) or bool((dd[1:] <= dd[:-1]).any()):
            raise ValueError(f"{label}: diameters must be positive and increasing, "
                             f"got {dd.tolist()}")
        if bool((ee < 0).any()) or bool((ee > 1).any()):
            raise ValueError(f"{label}: efficiencies must be between 0 and 1, "
                             f"got {ee.tolist()}")
        if ee[0] != ee[1] or ee[-2] != ee[-1]:
            raise ValueError(f"{label}: the first two and the last two efficiencies must be "
                             f"equal (the curve is flat at both ends; ContamX refuses this "
                             f"curve), got {ee.tolist()}")
        self.diameters, self.efficiencies, self.name = d, e, name
        self._curve = SplineCurve(torch.log10(d), e, name=name)

    def at(self, diameter: float | Tensor) -> Tensor:
        """The efficiency at a particle diameter (m): ContamX's curve fit in log10 of the
        diameter; with flat end segments it is constant beyond the curve."""
        d = torch.as_tensor(diameter, dtype=F64)
        return self._curve(torch.log10(d))

    #: The efficiency does not depend on the air passing.
    depends_on_flow = False

    def efficiency(self, species: Sequence[Species], air: AirThroughFilter | None = None
                   ) -> Tensor:
        """`(K,)`: the curve at each particle species' mean diameter, 0 for a gas. `air` is
        not read."""
        zero = torch.zeros((), dtype=F64)
        return torch.stack([self.at(s.mean_diameter) if s.is_particle else zero
                            for s in species])


def _scalar(value, label: str) -> Tensor:
    v = torch.as_tensor(value, dtype=F64)
    if v.dim() != 0:
        raise ValueError(f"{label} must be a scalar, got shape {tuple(v.shape)}")
    return v


class UVGIFilter:
    """Ultraviolet germicidal irradiation (CONTAM's Penn State UVGI filter element, TN 1887r1
    equations uv-1 to uv-9): each species is inactivated with the single-pass efficiency

        eta_k = 1 - exp(-k_k I_t),

    `k_k` the species' UV susceptibility (`Species.uv_susceptibility`, m2/J) and `I_t` the
    UV dose (J/m2) the air receives in the lamp's flow region, scaled from the design dose by
    the lamp's output and the air's residence time:

        I_t = I_t,design OUT (rho / rho_std) F_design / F,
        I_t,design = ln(S_design) / -k_design,
        OUT = OUT_TU OUT_age,
        OUT_TU = clamp(C0 + C1 T + C2 U + C3 T^2 + C4 U^2, 0, 100) / 100,
        U = F / (rho A),  A = F_design / (rho_std u_design),

    with F the mass flow through the filter (kg/s), T (deg C) and rho the temperature and
    density of the air arriving, and rho_std = `STANDARD_AIR_DENSITY`. With an age model
    (`age_coefficients=(K0, K1)`) the output also changes with the lamp's age t (hours since
    the start of the run):

        a = (K0 + K1 ln t) / 100,  OUT_age = min(a, 1) for a >= 0, 0.1 for a < 0.

    So OUT_age stays at 1 until t_100 = exp((100 - K0) / K1) when K1 < 0, as TN 1887r1 says,
    and with K1 = 0 it is K0 / 100 from the start. The clamps and the 0.1 for a negative age
    output are ContamX 3.4.1.7's (measured: a negative OUT_TU gives no inactivation, a
    negative age output 10 %).

    Air crossing the filter against its edge's direction (`mass_flow < 0`) receives, as in
    ContamX (measured), the fixed dose `REVERSE_FLOW_DOSE` = 1 J/m2, whatever the filter's
    data: eta_k = 1 - exp(-k_k). That is ContamX's fallback, not the Penn State model.

    Parameters: `design_survival` S_design in (0, 1), `design_velocity` u_design (m/s, > 0),
    `design_mass_flow` F_design (kg/s, > 0), `design_rate_constant` k_design (m2/J, > 0),
    `output_coefficients` (C0..C4, in percent, T in deg C and U in m/s) and
    `age_coefficients` (K0, K1, in percent, t in hours; None for no age model). Each may be a
    tensor that carries a gradient; the efficiency is differentiable in all of them and in the
    air's mass flow, temperature and density. At zero flow the efficiency is 1 (the limit of
    a dose growing as 1 / F) with a finite gradient.
    """

    #: The efficiency depends on the air passing.
    depends_on_flow = True

    def __init__(self, *, design_survival, design_velocity, design_mass_flow,
                 design_rate_constant, output_coefficients, age_coefficients=None,
                 name: str = "") -> None:
        label = f"UVGIFilter {name!r}"
        self.design_survival = _scalar(design_survival, f"{label}: design_survival")
        self.design_velocity = _scalar(design_velocity, f"{label}: design_velocity")
        self.design_mass_flow = _scalar(design_mass_flow, f"{label}: design_mass_flow")
        self.design_rate_constant = _scalar(design_rate_constant,
                                            f"{label}: design_rate_constant")
        s = self.design_survival.detach().item()
        if not 0.0 < s < 1.0:
            raise ValueError(f"{label}: design_survival must be between 0 and 1 (exclusive), "
                             f"got {s}")
        for what, v in (("design_velocity", self.design_velocity),
                        ("design_mass_flow", self.design_mass_flow),
                        ("design_rate_constant", self.design_rate_constant)):
            if not v.detach().item() > 0.0:
                raise ValueError(f"{label}: {what} must be positive, got {v.detach().item()}")
        c = _points(output_coefficients, label)
        if c.numel() != 5:
            raise ValueError(f"{label}: output_coefficients must be the five C0..C4, got "
                             f"{c.numel()}")
        self.output_coefficients = c
        self.age_coefficients = None
        if age_coefficients is not None:
            k = _points(age_coefficients, label)
            if k.numel() != 2:
                raise ValueError(f"{label}: age_coefficients must be the pair (K0, K1), got "
                                 f"{k.numel()} values")
            self.age_coefficients = k
        self.name = name

    @property
    def design_dose(self) -> Tensor:
        """I_t,design = ln(S_design) / -k_design (J/m2)."""
        return torch.log(self.design_survival) / -self.design_rate_constant

    def output(self, air: AirThroughFilter) -> Tensor:
        """`(...)`: the lamp's fractional output OUT = OUT_TU OUT_age for air moving along
        the edge (its flow positive); the class docstring gives the formulae."""
        F = torch.as_tensor(air.mass_flow, dtype=F64)
        rho = torch.as_tensor(air.density, dtype=F64)
        flow = torch.where(F > 0, F, torch.ones_like(F))
        velocity = (flow * STANDARD_AIR_DENSITY * self.design_velocity
                    / (rho * self.design_mass_flow))
        T = torch.as_tensor(air.temperature, dtype=F64) - 273.15
        c0, c1, c2, c3, c4 = self.output_coefficients.unbind()
        tu = c0 + c1 * T + c2 * velocity + c3 * T * T + c4 * velocity * velocity
        out = tu.clamp(0.0, 100.0) / 100.0
        if self.age_coefficients is not None:
            out = out * self.age_factor(air.time)
        return out

    def age_factor(self, time) -> Tensor:
        """OUT_age at `time` (s since the start of the run); 1 without an age model. At
        t = 0 it is the formula's limit (1 for K1 < 0, 0.1 for K1 > 0, K0 / 100 mapped as
        above for K1 = 0)."""
        if self.age_coefficients is None:
            return torch.ones((), dtype=F64)
        k0, k1 = self.age_coefficients.unbind()
        hours = torch.as_tensor(time, dtype=F64) / 3600.0
        later = hours > 0
        log_t = torch.log(torch.where(later, hours, torch.ones_like(hours)))
        a = (k0 + k1 * log_t) / 100.0
        slope = k1.detach().item()
        limit = (torch.ones((), dtype=F64) if slope < 0 else
                 -torch.ones((), dtype=F64) if slope > 0 else k0 / 100.0)
        a = torch.where(later, a, limit)
        return torch.where(a < 0, torch.full_like(a, 0.1), a.clamp(max=1.0))

    def dose(self, air: AirThroughFilter) -> Tensor:
        """`(...)`: the UV dose I_t (J/m2): the class docstring's for air moving along the
        edge, `REVERSE_FLOW_DOSE` for air crossing against it, and +inf at zero flow."""
        F = torch.as_tensor(air.mass_flow, dtype=F64)
        rho = torch.as_tensor(air.density, dtype=F64)
        forward = F > 0
        flow = torch.where(forward, F, torch.ones_like(F))
        along = (self.design_dose * self.output(air) * (rho / STANDARD_AIR_DENSITY)
                 * self.design_mass_flow / flow)
        other = torch.where(F < 0, torch.full_like(along, REVERSE_FLOW_DOSE),
                            torch.full_like(along, math.inf))
        return torch.where(forward, along, other)

    def efficiency(self, species: Sequence[Species], air: AirThroughFilter | None = None
                   ) -> Tensor:
        """`(..., K)`: 1 - exp(-k_k I_t) for each species, in `species` order (0 for a
        species whose UV susceptibility is 0)."""
        if air is None:
            raise ValueError(f"UVGIFilter {self.name!r}: its efficiency depends on the air "
                             f"passing (mass flow, temperature, density, time); pass `air`")
        k = torch.tensor([s.uv_susceptibility for s in species], dtype=F64)
        dose = self.dose(air).unsqueeze(-1)
        # exp(-k inf) = 0 for k > 0; a species with k = 0 passes untouched (k inf would be
        # NaN), and so does the gradient.
        exposed = k > 0
        exponent = torch.where(exposed, k * torch.where(exposed, dose, 0.0), 0.0)
        return 1.0 - torch.exp(-exponent)


class FilterSeries:
    """Filters in series (CONTAM's super filter): the air passes each stage in turn, so the
    pass fraction is the product of the stages' pass fractions."""

    def __init__(self, stages: Sequence, *, name: str = "") -> None:
        if not stages:
            raise ValueError(f"FilterSeries {name!r}: needs at least one stage")
        self.stages, self.name = list(stages), name
        #: True when a stage's efficiency depends on the air passing (a `UVGIFilter`).
        self.depends_on_flow = any(s.depends_on_flow for s in self.stages)

    def efficiency(self, species: Sequence[Species], air: AirThroughFilter | None = None
                   ) -> Tensor:
        """`(..., K)`: 1 minus the product of the stages' pass fractions; `air` is handed to
        every stage (a flow-dependent stage needs it)."""
        passed = torch.ones(len(species), dtype=F64)
        for stage in self.stages:
            passed = passed * (1.0 - stage.efficiency(species, air))
        return 1.0 - passed

    def flow_dependent_stages(self) -> list:
        """The stages, recursively, whose efficiency depends on the air passing."""
        out = []
        for s in self.stages:
            if isinstance(s, FilterSeries):
                out += s.flow_dependent_stages()
            elif s.depends_on_flow:
                out.append(s)
        return out


def _filtered_rows(net: Network, flow_kinds: Sequence[str], filters: Mapping[EdgeKey, object],
                   where: str) -> tuple[int, list[tuple[int, int, object]]]:
    """`(b_flow, [(row, column, filter)])`: each filtered edge's row in the species layer's
    flow order and its network column."""
    order: list[int] = []
    for kind in flow_kinds:
        order += net.edge_index(kind).tolist()
    position = {col: i for i, col in enumerate(order)}
    edges, kinds = net.edges, net.edge_kinds()
    found = []
    for edge, flt in filters.items():
        try:
            col = edges.index(tuple(edge))
        except ValueError as exc:
            raise KeyError(f"{where}: edge {edge!r} is not in the network") from exc
        if col not in position:
            raise KeyError(f"{where}: edge {edge!r} is of kind {kinds[col]!r}, "
                           f"not one of the species layer's flow kinds {tuple(flow_kinds)}")
        found.append((position[col], col, flt))
    return len(order), found


def filter_transmission(net: Network, flow_kinds: Sequence[str], species: Sequence[Species],
                        filters: Mapping[EdgeKey, object]) -> Tensor:
    """The species layer's `transmission`, `(b_flow, K)`: 1 - efficiency on each filtered
    edge, 1 elsewhere. Rows follow the layer's flow order: the edges of each kind in
    `flow_kinds`, kind by kind, each kind in network order; pass the layer's own
    `flow_kinds`. An edge is named by the `(source, target, key)` `add_edge` returned. A
    filter whose efficiency depends on the air passing (`UVGIFilter`, or a `FilterSeries`
    holding one) has no fixed transmission and is refused by name: `FilterTransmission`
    writes it every step."""
    n, found = _filtered_rows(net, flow_kinds, filters, "filter_transmission")
    rows = [torch.ones(len(species), dtype=F64) for _ in range(n)]
    for row, _, flt in found:
        if flt.depends_on_flow:
            raise ValueError(f"filter_transmission: filter {flt.name!r} has an efficiency "
                             f"that depends on the air passing, so no fixed transmission; "
                             f"use the FilterTransmission closure")
        rows[row] = 1.0 - flt.efficiency(species)
    return torch.stack(rows) if rows else torch.ones(0, len(species), dtype=F64)


class FilterTransmission:
    """Closure: the species layer's per-step transmission `"<layer>.transmission"`,
    `(..., b_flow, K)`, for filters that include flow-dependent ones (`UVGIFilter`).

    Each call reads the airflow layer's solved flows (`state["<airflow>.q"]`) and, at each
    filtered edge's SOURCE node (the path's declared from-node, which TN 1887r1 names as the
    upstream zone for the filter), the drivers `"T"` (K) and `"rho"` (kg/m3, written by the
    model's density closure, which must run first), and `"time"` (s, the lamp's age for an
    age model). It writes 1 - efficiency on every filtered edge and 1 elsewhere. With no flows
    in the state yet it writes nothing (the layer's construction-time transmission stands).

    A closure runs before the airflow solve, so the flows it reads are the ones handed in:
    run the model with `coupling="iterate"` (the species layer named in `iterate_tol`), where
    each pass is fed the previous pass's flows and the converged step uses its own.
    `project_to_model` builds a project with a UVGI filter that way.

    `on_reverse(filter)` is called once per flow-dependent filter the first time air
    crosses its edge against the edge's direction (default: nothing).
    """

    def __init__(self, net: Network, flow_kinds: Sequence[str], species: Sequence[Species],
                 filters: Mapping[EdgeKey, object], *, airflow, layer: str = "species",
                 on_reverse=None) -> None:
        self.flow_kinds = tuple(flow_kinds)
        self.species = list(species)
        self.airflow = airflow
        self.flow_key = f"{airflow.name}.q"
        self.key = f"{layer}.transmission"
        self.outputs = (self.key,)
        self.inputs = ("T", "rho", "time")
        n, found = _filtered_rows(net, flow_kinds, filters, "FilterTransmission")
        K = len(self.species)
        fixed = [torch.ones(K, dtype=F64) for _ in range(n)]
        self.varying: list[tuple[int, int, object]] = []
        for row, col, flt in found:
            if flt.depends_on_flow:
                self.varying.append((row, net.node_index(net.edges[col][0]), flt))
            else:
                fixed[row] = 1.0 - flt.efficiency(self.species)
        self.fixed = fixed
        self.on_reverse = on_reverse
        self._reversed: set[int] = set()

    @property
    def input_specs(self) -> dict[str, dict]:
        """The temperatures, densities and time the flow-dependent filters read."""
        return {
            "T": dict(description="air temperature at every node", unit="K", over="nodes"),
            "rho": dict(description="air density at every node", unit="kg/m3",
                        over="nodes"),
            "time": dict(description="time since the start of the run (a UVGI lamp's age)",
                         unit="s"),
        }

    def __call__(self, state, drivers):
        q = state.get(self.flow_key)
        if q is None:
            return {}
        f = self.airflow.flows_of_kind(q, self.flow_kinds)
        T, rho = drivers["T"], drivers["rho"]
        time = drivers.get("time", 0.0)
        rows = list(self.fixed)
        for row, src, flt in self.varying:
            F = f[..., row]
            if (self.on_reverse is not None and id(flt) not in self._reversed
                    and bool((F.detach() < 0).any())):
                self._reversed.add(id(flt))
                self.on_reverse(flt)
            air = AirThroughFilter(F, T[..., src], rho[..., src], time)
            rows[row] = 1.0 - flt.efficiency(self.species, air)
        shape = torch.broadcast_shapes(*(r.shape for r in rows))
        return {self.key: torch.stack([r.expand(shape) for r in rows], dim=-2)}

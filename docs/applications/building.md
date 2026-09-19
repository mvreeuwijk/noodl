# Buildings

Multi-zone airflow, heat balance and contaminant transport — the CONTAM class of problem, in a
differentiable and batched form.

Rooms are nodes; leaks, doorways, fans and dampers are edges. Air moves under buoyancy, wind
pressure and mechanical pressure; heat rides on that airflow and conducts through walls with
thermal mass; contaminants ride on it too, with CONTAM's four source models. Because air density
depends on temperature and the stack effect depends on density, the airflow and heat problems
are genuinely two-way coupled.

**Units.** Mass flow kg/s, temperature K, heat W, mass fraction kg/kg, pressure Pa. The whole
application works in `float64` regardless of the network's default dtype.

```python
from noodl.apps.building import (
    Zone, WallMass, add_zone, build_model, initial_state,
    add_large_opening, orifice_elements_from_edges, read_prj, project_to_model, read_wth,
)
```

![Two zones joined by a doorway, with air, thermal and species layers on one network](../assets/app-building.svg)

## Two ways in

**From scratch**, for research and verification cases:

```python
import torch
from noodl.apps.building import Zone, add_zone, add_large_opening, orifice_elements_from_edges
from noodl.apps.building import build_model, initial_state
from noodl.drives import Stack
from noodl.topology import Network

F64 = torch.float64

net = Network(dtype=F64)
net.add_node("ambient", z_ref=0.0)
add_zone(net, Zone("A", volume=60.0, T0=288.15))
add_zone(net, Zone("B", volume=60.0, T0=285.15))

add_large_opening(net, "A", "B", H=2.0, W=0.9, z_mid=1.0, Cd=0.78)
net.add_edge("ambient", "B", kind="airpath", z_path=0.2, Cd=0.6, area=0.05)
net.add_edge("ambient", "B", kind="airpath", z_path=1.8, Cd=0.6, area=0.05)

el = orifice_elements_from_edges(net, "airpath")
model = build_model(
    net,
    air_elements=[el],
    drives=[Stack.from_network(net, "airpath")],
    coupling="iterate",
    iterate_tol={"thermal": 0.01},
    iterate_max=50,
)

state = initial_state(model)
sources = torch.zeros(net.n, dtype=F64)
sources[net.node_index("A")] = 1000.0      # a 1 kW heat source in room A

drivers = {
    "air.phi_boundary": torch.zeros(1, dtype=F64),
    "thermal.x_boundary": torch.tensor([283.15], dtype=F64),
    "thermal.sources": sources,
}
state = model.step(state, drivers, dt=60.0)

print(state["thermal.x"])   # [T_A, T_B] in K
print(state["air.q"][0])    # doorway low-opening flow, kg/s
```

*(Adapted from `benchmarks/natural_ventilation.py`, the golden reference case.)*

**From a CONTAM project file:**

```python
from noodl.apps.building import read_prj, project_to_model

project = read_prj("valThreeZonesWthCtm-UseApi.prj")
model, state, drivers = project_to_model(project)
solved = model.steady(state, drivers)

flows = project.path_flows(solved["air.q"])   # in CONTAM path-number order
```

## The API

### Zones and walls

| Object | Purpose |
|---|---|
| `Zone(name, volume, T0=293.15, z_ref=0.0, wall=None)` | A room. `volume` in m³, `T0` the initial temperature, `z_ref` its reference height. |
| `WallMass(name, capacity, ua_zone, ua_ambient)` | A lumped wall node: heat capacity J/K, conductances W/K to the zone and to ambient. |
| `add_zone(net, zone, ambient="ambient")` | Adds the zone's air node and, if `zone.wall` is set, the wall node plus `kind="wall"` edges `zone → wall → ambient`. |

`WallMass` refuses a non-positive `ua` at construction. A negative conductance makes the
conduction operator anti-diffusive — it violates the maximum principle — and *every solve still
reports success*, because nothing downstream looks at the sign. Zero is refused as "a wall that
conducts nothing".

### Layers and model assembly

| Function | Returns |
|---|---|
| `thermal_layer(net, *, ambient, name="thermal", flow_kinds=("airpath",), conduction_kind="wall", c_p=1005.0, rho=1.2041, scheme="exact", fixed_temperature=())` | The heat `TransportLayer`. Capacity is $\rho c_p V + C_{\text{node}}$ per active interior node. Raises naming any node with zero heat capacity. |
| `species_layer(net, *, ambient, name="species", flow_kinds=("airpath",), rho=1.2041, n_species=1, scheme="implicit")` | The contaminant layer, as mass fractions, capacity = zone air mass $\rho V$ (CONTAM's convention). |
| `build_model(net, *, air_elements, drives, ambient="ambient", thermal=True, species=0, density="ideal_gas", density_kwargs=None, coupling="pingpong", iterate_tol=None, iterate_max=20, thermal_scheme="exact", species_scheme="implicit", flow_kinds=None)` | The assembled `Model`. |
| `initial_state(model)` | Temperatures from each node's `T0`, zeros for species. |

`density` selects the closure relating temperature to air density:

- `"ideal_gas"` → `IdealGasDensity`: $\rho = P_{\text{ref}} / (R T)$, with `P_ref` overridable per
  call through `drivers["P_ref"]`.
- `"linear"` → `LinearDensity`: the Boussinesq form
  $\rho = \rho_0\,(1 - (T - T_0)/T_0)$, which is what the Li and Delsante analytical solutions
  assume.

`flow_kinds` narrows which edge kinds advect heat and species; it defaults to every air element's
kind. Naming a kind no air element provides raises — a kind the air layer does not solve carries
no flow, so heat and species would silently not advect on it.

### Elements

| Function | Law |
|---|---|
| `mass_orifice(Cd, A, *, rho=1.2041, dp_transition=1e-3, kind="airpath")` | $F = C_d A \sqrt{2 \rho \Delta p}$ kg/s, built as a `PowerLaw` at $n = 0.5$. |
| `orifice_elements_from_edges(net, kind, *, rho=1.2041, ...)` | One `PowerLaw` over every edge of a kind, reading `Cd` and `area` off each edge. |
| `add_large_opening(net, a, b, *, H, W, z_mid, Cd=0.78, kind="airpath")` | CONTAM's two-opening doorway (`DR_PL2`, TN 1887r1 eq. 69–70): two orifices of area $WH/2$ at $z_{\text{mid}} \mp 2H/9$. Exact for a mid-height neutral plane. Returns the two edge keys. |

### Contaminant sources

CONTAM's four source types, all in `noodl.apps.building.sources`:

| Source | Model |
|---|---|
| `ConstantSource(node, G, R=0.0)` | $S = G - R x$ (TN 1887r1 eq. 15) |
| `CutoffSource(node, G, x_cut)` | $S = \max(G(1 - x/x_{\text{cut}}), 0)$ (eq. 17) — clamped at zero rather than going negative, because CONTAM's cutoff source models generation that *shuts off*, not a sink |
| `DecayingSource(node, G0, tau, t0=0.0)` | $S = G_0 e^{-(t-t_0)/\tau}$ for $t \ge t_0$ |
| `BurstSource(node, mass, t_burst, dt)` | `mass` delivered uniformly over $[t_{\text{burst}}, t_{\text{burst}} + dt)$ |

`assemble_sources(sources, t, x_full)` sums them; `sources_from_project(project, dt=60.0)` builds
them from a `.prj`'s own source records.

## File readers

### CONTAM `.prj`

`read_prj(path) -> Project` parses a documented subset of TN 1887r1 Appendix A into a `Project`,
building a float64 `Network`, its elements and its drives. `project_to_model(project, *,
ambient=None, species=True, scheme="implicit")` returns `(model, state, drivers)`.

**Supported:** run control and ambient conditions; species; wind pressure profiles; zones and
initial concentrations; airflow paths; the power-law element family
(`plr_orfc/leak1/leak2/leak3/crack/fcn/test1/test2/conn/stair/shaft/qcn`), the quadratic family
(`qfr_*`), doorways (`dor_door`, `dor_pl2`), dampers (`plr_bdq`, `plr_bdf`), constant-flow fans
(`fan_cmf`, `fan_cvf`) and the cubic fan curve (`fan_fan`); source elements `ccf`, `cut`, `eds`,
`brs`.

**Refused by name**, rather than silently dropped: schedules, control nodes, filters, kinetic
reactions and air-handling systems referenced by nonzero index; CFD and 1-D convection/diffusion
zones; continuous-values-file zones and paths; duct networks; a constant wind pressure with no
profile; a fan curve on a path with `mult != 1`; `csf_*` and `sup_afe` elements.

A **filter** is refused emphatically, because a filter is invisible to airflow but *not* to the
species layer — loading one would silently corrupt contaminant results while the airflow looked
perfect.

Control nodes that no zone and no path references are silently skipped rather than refused: NIST's
own sample projects carry dozens of unreferenced sensor and logger nodes, and those projects must
load.

**`project_to_model` never builds a thermal layer**, because a `.prj` carries no thermal data.

Every power-law element is built as `UpstreamDensityPowerLaw` — re-evaluating the density of the
air *entering* the path, per direction, matching ContamX section 3.2. The quadratic, damper and
fan families keep reference-density coefficients, because they are not part of this application's
parity evidence.

### CONTAM `.wth`

`read_wth(path) -> Weather` reads the `WeatherFile ContamW 2.0` format. Only `Date, Time, Ta, Pb,
Ws, Wd` are used; humidity, radiation and ground-temperature columns are ignored, as is the
per-day header block.

```python
weather = read_wth("year.wth")
drivers.update(weather.drivers_at(t, thermal="thermal"))
```

`at(t)` interpolates linearly between listed times, with wind direction interpolated on its
shortest arc. `drivers_at(t)` returns the driver keys a `build_model`-built model reads:
`"<thermal>.x_boundary"`, `"P_ref"`, `"V_met"`, `"theta_w"`. It handles the single-boundary-node
case; a model with several prescribed temperatures must build its own boundary vector.

## Validation

### Against ContamX

The reference is NIST's ContamX 3.4.1.7, driven through `contamxpy` (the `contam` extra, Windows
x86-64 only — the wheel bundles the engine). `noodl.apps.building.contamx` provides `run_steady`
and `run_transient`, which run the engine on a scratch copy of the project and never mutate your
fixture directory.

| Check | Tolerance | Measured |
|---|---|---|
| Stack project, flow directions | exact signs | match |
| Stack project, flow magnitudes, over a ±20 K ambient sweep | rel 1e-3 | **4.1e-5 – 4.4e-5** |
| Residual flatness across the sweep (max/min ratio) | < 1.5 | holds |
| Constant-mass-flow fan delivers its rating (0.200683 kg/s) | rel 1e-5 | holds |
| Three-zone project, steady flows | rel 1e-3, abs 1e-6 | holds |
| Three-zone project, transient concentrations, 24 steps at 300 s | rel 1e-3 | **6.5e-6** |

The magnitude row has history worth knowing. It was previously an `xfail` at a measured
**1.7e-2** relative error — 17x over tolerance — because the reader froze orifice coefficients at
reference density. Adding the upstream-density correction brought it to 4.1e-5, about 25x inside
tolerance. The residual-flatness test exists to guard the fix: the old error scaled with $\Delta
T$ (1.72e-2 at 20 K, 8.61e-3 at 10 K, 0 at 0 K), so a re-introduced density bug would show up as
a residual proportional to $\Delta T$ even if the absolute number stayed small.

The path-flow sign convention was verified independently against two cases rather than assumed,
since `contamxpy`'s own documentation does not pin the sign of `getPathFlow`.

### Against analytical solutions

Separately from ContamX parity, `tests/verification/test_natural_ventilation.py` checks the
coupled airflow-heat physics against the closed-form solutions of Li and Delsante (2001) for a
single ventilated zone — buoyancy-only, envelope-loss, assisting-wind and opposing-wind cubics —
plus an independent `scipy.optimize.fsolve` oracle for a two-zone doorway case and a golden
regression at rtol 1e-8.

## Limitations

- **The coupling default is a trap.** `build_model`'s `coupling` defaults to `"pingpong"`, one
  pass. A `.steady()` call at the defaults solves the airflow at the *initial* temperatures and
  never re-converges. On the linear-density single-zone case it silently returns
  $T_z = T_o + S / (c_p F(293.15\,\mathrm{K}))$ rather than the coupled answer. Pass
  `coupling="iterate"` with an `iterate_tol` whenever the airflow depends on the temperatures it
  carries.
- **`iterate_tol` has a residual floor** set by the potential solve's own Newton tolerance
  propagated through $dT/dF$. Too tight a value stalls, and `steady` raises naming the layer, the
  change and the tolerance.
- **The three-root case does not converge under `"iterate"`.** For a documented Li and Delsante
  opposing-wind case with three roots, the hard-coded 0.5 relaxation in `Model._iterate` cannot
  reach the wind-driven-upward stable root — it diverges from it even when started exactly on it.
  The test is `xfail(strict=True)`. The physics is sound: ping-pong **time stepping** resolves
  all three roots correctly, and a companion test passes. The blocker is the fixed relaxation,
  not successive substitution as a method.
- **Wall conductance is not learnable** through this application's API — `WallMass` takes a plain
  float.
- **Doorway `dp_transition` is approximated.** Doorway openings do not derive `dp_transition`
  from the record's `lam`; both openings get `PowerLaw`'s generic 1e-3 Pa default instead of the
  smaller value the formula would give. The error is bounded twice over — confined to
  $\lvert \Delta p \rvert < 10^{-3}$ Pa, and no doorway flow is compared against ContamX anywhere
  — and it is recorded as a follow-up rather than quietly fixed.

## Install

The application needs nothing beyond the base dependencies. The `contam` extra is required only
to run ContamX itself for a side-by-side comparison:

```bash
pip install "noodl[contam]"    # Windows x86-64 only
```

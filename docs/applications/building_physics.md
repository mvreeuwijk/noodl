# Building physics

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
from noodl.apps.building_physics import (
    Zone, WallMass, add_zone, build_model, initial_state,
    add_large_opening, orifice_elements_from_edges, read_prj, project_to_model, read_wth,
)
```

![Two zones joined by a doorway, with air, thermal and species layers on one network](../assets/app-building.svg)

## Two ways in

**From scratch**, for research and verification cases:

```python
import torch
from noodl.apps.building_physics import Zone, add_zone, add_large_opening, orifice_elements_from_edges
from noodl.apps.building_physics import build_model, initial_state
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

Models can also be read from CONTAM `.prj`/`.wth` files or from a Modelica Buildings Library export — see [File formats](../formats/contam.md) and [Modelica Buildings Library](../formats/modelica.md).

**From a CONTAM project file:**

```python
from noodl.apps.building_physics import read_prj, project_to_model

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

CONTAM's four source types, all in `noodl.apps.building_physics.sources`:

| Source | Model |
|---|---|
| `ConstantSource(node, G, R=0.0)` | $S = G - R x$ (TN 1887r1 eq. 15) |
| `CutoffSource(node, G, x_cut)` | $S = \max(G(1 - x/x_{\text{cut}}), 0)$ (eq. 17) — clamped at zero rather than going negative, because CONTAM's cutoff source models generation that *shuts off*, not a sink |
| `DecayingSource(node, G0, tau, t0=0.0)` | $S = G_0 e^{-(t-t_0)/\tau}$ for $t \ge t_0$ |
| `BurstSource(node, mass, t_burst, dt)` | `mass` delivered uniformly over $[t_{\text{burst}}, t_{\text{burst}} + dt)$ |

`assemble_sources(sources, t, x_full)` sums them; `sources_from_project(project, dt=60.0)` builds
them from a `.prj`'s own source records.

## Verification

### Against ContamX

The reference is NIST's ContamX 3.4.1.7, driven through `contamxpy` (the `contam` extra, Windows
x86-64 only — the wheel bundles the engine). `noodl.apps.building_physics.contamx` provides `run_steady`
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

### Against OpenModelica (Modelica Buildings Library)

**Algebraic models (6, no volumes).** Every CSV row and column against OpenModelica's own
simulation, `|noodl - omc| <= 1e-6 |omc| + 1e-9` kg/s:

| Model | Worst relative error | Rows compared |
|---|---|---|
| OneWayFlow | 2.4e-16 | 501 |
| DoorOpenClosed | 1.2e-16 | 501 |
| OpenDoorPressure | 2.4e-16 | 49 |
| OpenDoorTemperature | 1.8e-11 | 49 |
| Orifice | 4.0e-16 | 501 |
| PowerLaw | 6.1e-16 | 501 |

All at round-off except `OpenDoorTemperature`, whose discretised-door port flows sit at 1.8e-11
relative — still five orders of magnitude inside the 1e-6 bound, and attributed to an
OpenModelica nonlinear-solver residual on the door's inflow-density loop, not a formula
difference (likely, not diagnosed: not shown by a tighter `omc` tolerance).

The **CONTAM cross-check** on `OneWayFlow` (13 pressure-difference knots × 8 elements, from
CONTAM's own validation table, `OneWayFlow.mo`'s `contamData`): noodl differs from CONTAM by at
most 0.74 % relative (7.68e-4 kg/s absolute) — exactly the amount MBL itself differs from CONTAM.
The assertion is `|noodl - contam| <= |omc - contam| + 1e-6 |omc| + 1e-9` at every entry: noodl
adds nothing to MBL's own departure from the table (41 of 104 entries miss the table's 3
significant figures, in MBL as much as in noodl).

**Dynamic models (12, with volumes).** The error metric is `|noodl - omc| / max(|omc|, floor)`
over every row after `t = StartTime`; the table below instead reports, per model, the worst
ABSOLUTE error in temperature (K) and pressure (Pa), and the worst flow error as a percentage of
the model's own largest flow — the more informative view, since T and p in kelvin and pascal are
insensitive to relative error and a flow that reverses sign makes a relative error explode near
the crossing.

*Parity* — agrees with OpenModelica to its own discretisation/solver tolerance, except
`ZonalFlow`'s T (explained below):

| Model | T, abs (K) | p, abs (Pa) | flow, abs (kg/s) | flow, % of model's largest flow |
|---|---|---|---|---|
| ThreeRoomsContam | 2.1e-6 | 3.7e-5 | 2.3e-7 | 5.4e-5 % |
| ThreeRoomsContamDiscretizedDoor | 2.1e-6 | 3.7e-5 | 1.1e-7 | 2.6e-5 % |
| OneRoom | 5.8e-11 | 1.5e-11 | 1.2e-13 | 1.6e-9 % |
| ZonalFlow | 1.0e-2 | 1.5e-11 | 0 | 0 % |
| CO2TransportStep | 2.1e-6 | 5.0e-5 | 2.6e-7 | 6.2e-5 % |

`CO2TransportStep`'s trace-gas mass fraction `C` is excluded from this group: the row just after
its 3.6 s CO2 pulse differs from OpenModelica by up to 170 % relative (6.0e-8 kg/kg absolute) —
noodl injects the pulse's exact mass but spreads it over its 172.8 s step, while OpenModelica has
only just begun to receive it. An independent DOP853 integration of the same species equations
(reusing noodl's flows) shows OpenModelica's own error dominates from about t > 5000 s: up to
3.8 % of the peak concentration, against noodl's 0.6 %.

`ZonalFlow`'s T, 1.0e-2 K (`rooB.T`, 3.5e-5 relative, peaking at t = 36 s), is not solver
tolerance. `rooA` and `rooB` start 10 K and 0.005 kg/kg water apart (`ZonalFlow.json`), and
noodl carries heat between zones with one common `cp` instead of MBL's per-zone `cp(X)`
(`assemble.py`'s "Capacities" derivation: `|cp(X_in)/cp(X) - 1| <= 0.84 |dX_w|`, here
`0.84 x 0.005 = 4.2e-3` relative). Applied to the zones' 10 K starting gap, that bounds the
resulting error at about 0.04 K — the same order of magnitude as the measured 1.0e-2 K (about
4x tighter, plausibly because `rooB`'s 1 m3 is 1 % of `rooA`'s 100 m3 and the gap decays as
they mix). This is the most likely cause; it has not been confirmed by rerunning with a
per-zone `cp`.

*Step-limited* — first order in noodl's time step; halving the step halves the error:

| Model | T, abs (K) | p, abs (Pa) | flow, abs (kg/s) | flow, % of model's largest flow |
|---|---|---|---|---|
| OpenDoorBuoyancyDynamic | 1.0e-2 | 2.1e-4 | 2.0e-3 | 1.2 % |
| OpenDoorBuoyancyPressureDynamic | 1.1e-2 | 2.0e-4 | 1.9e-3 | 1.1 % |
| NaturalVentilation | 1.1e-3 | 0.12 | 5.3e-5 | 0.15 % |
| ReverseBuoyancy3Zones | 2.0e-2 | 1.6e-3 | 1.3e-3 | 0.46 % |

Confirmed directly: halving `OpenDoorBuoyancyDynamic`'s step scales its worst door-flow and
boundary-temperature error by 1.97 (`test_step_limited_error_halves_with_the_step`).
`OpenDoorBuoyancyPressureDynamic` shows a comparable 2.04 (measured the same way during
development, but not independently asserted by a test).

*Storage-dominated* — MBL's volumes compress and expand; noodl's airflow is quasi-steady, like
CONTAM's, so it does not:

| Model | T, abs (K) | p, abs (Pa) | flow, abs (kg/s) | flow, % of model's largest flow |
|---|---|---|---|---|
| ClosedDoors | 0.31 | 243 | 8.3e-5 | 73 % |
| OneOpenDoor | 0.30 | 366 | 8.2e-4 | 0.9 % |
| ReverseBuoyancy | 0.90 | 566 | 0.20 | 53 % |

Each test asserts the physical mechanism, not just a bound. `ClosedDoors` and `OneOpenDoor` are
closed, ideal-gas rooms heated by a sinusoidal source: MBL's rooms heat at constant volume, while
noodl's zone capacity is the constant-pressure `m cp`, so the ratio of MBL's to noodl's
temperature rise should be `cp/cv` — measured 1.4016 and 1.3995 against `cp/cv` = 1.398 and
1.400. `ReverseBuoyancy`'s zones start 1325 Pa above the boundary; MBL releases the excess through
mass storage and cools by close to the flow-work-minus-latent-heat prediction (0.83 K measured
against 0.78 K predicted, within the ruled 10 % tolerance), while noodl starts already balanced
and does not cool.

**The `t = StartTime` row** is excluded from every bound above, and reported separately. At that
row OpenModelica holds MBL's own pressure initialisation — up to 35 Pa off balance in the
`ThreeRooms*`/`CO2TransportStep`/`ReverseBuoyancy3Zones` stack, 1325 Pa in `ReverseBuoyancy` —
which noodl's quasi-steady solve starts already balanced against. This is an initial-transient
difference from how the two solvers reach their first row, not a parity failure, and the test
still prints it.

Every column's numbers (not just the worst) are committed at
`tests/data/modelica/parity-algebraic.json` and `parity-dynamic.json` (regenerated only with
`NOODL_RECORD_PARITY=1`; see [Reproducing the export](../formats/modelica.md) on the Modelica
format page). Only the *storage-dominated* group
above (`ClosedDoors`, `OneOpenDoor`, `ReverseBuoyancy`) is caused by MBL's volume mass storage,
which noodl's quasi-steady airflow does not model. The *step-limited* group's numbers are
noodl's first-order time step instead (confirmed by halving it, above); `ZonalFlow`'s T is the
single-`cp` carrier (above); and `CO2TransportStep`'s excluded 170 % is its pulse spread over
a step, not storage. Adding volume mass storage would close the remaining gap in the
storage-dominated group; it is a possible extension, not implemented in this release.

### Against analytical solutions

Separately from ContamX parity, `tests/verification/test_natural_ventilation.py` checks the
coupled airflow-heat physics against the closed-form solutions of Li and Delsante (2001) for a
single ventilated zone — buoyancy-only, envelope-loss, assisting-wind and opposing-wind cubics —
plus an independent `scipy.optimize.fsolve` reference for a two-zone doorway case and a golden
regression at rtol 1e-8.

## Limitations

- **Airflow is quasi-steady.** On both the CONTAM and the Modelica route a zone's air mass is
  held fixed within a step, as in CONTAM: pressures and flows balance instantly and the air
  itself does not compress or expand. Where that storage matters — a closed, heated room
  expanding through its leakage, or a model that starts from unbalanced pressures — noodl's
  results differ from a model that resolves it, such as the Modelica Buildings Library. The
  [storage-dominated parity group](#against-openmodelica-modelica-buildings-library) shows by
  how much.
- **The default coupling is a single pass.** `build_model`'s `coupling` defaults to
  `"pingpong"`: a `.steady()` call then solves the airflow at the *initial* temperatures and does
  not re-converge. On the linear-density single-zone case it returns
  $T_z = T_o + S / (c_p F(293.15\,\mathrm{K}))$ rather than the coupled answer. Whenever the
  airflow depends on the temperatures it carries, pass `coupling="iterate"` with an
  `iterate_tol`.
- **`iterate_tol` cannot be tighter than the airflow solve itself.** Its floor is the potential
  solve's own Newton tolerance, propagated through the temperature response. A tolerance below
  that floor stalls, and `steady` raises an error naming the layer, the change reached and the
  tolerance asked for. Loosen `iterate_tol`, or tighten the airflow solve by passing Newton's
  `atol`/`rtol` as keyword arguments to `step` or `steady` (they reach the potential solve).
- **Multiple steady states: `"iterate"` may not find the one you want.** On Li and Delsante's
  opposing-wind single-zone case, which has three steady states, `coupling="iterate"` (a fixed
  0.5 relaxation) cannot hold the wind-driven, upward-flow state: it moves away from it even when
  started on it. Time stepping with the default `"pingpong"` coupling resolves all three states
  correctly, so for a case like this, step to steady state instead of calling `.steady()`.
- **Wall conductance is not learnable** through this application's API: `WallMass` takes plain
  floats for `ua_zone` and `ua_ambient`, so gradients do not reach them.
- **Doorway laminar transition is approximated.** Doorway openings use the generic laminar
  transition of `PowerLaw` (1e-3 Pa) rather than one derived from the door's own record. The
  effect is confined to pressure differences below $10^{-3}$ Pa across the doorway.
- **The Modelica import accepts a stated subset.** Wind pressure, weather data, controllers,
  dynamic medium columns and components that need compressible volume storage are refused with a
  named error; see [the Modelica import's refused content](../formats/modelica.md).

## Install

The application needs nothing beyond the base dependencies. The `contam` extra is required only
to run ContamX itself for a side-by-side comparison:

```bash
pip install "noodl[contam]"    # Windows x86-64 only
```

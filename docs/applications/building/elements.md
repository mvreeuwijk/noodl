# Building physics: airflow elements

Every edge of the air layer carries an element: a law giving the flow $F$ through the edge from the pressure difference $\Delta p$ (Pa) across it, together with its derivative for Newton's method. This page explains the laws the building application uses and where they differ. The classes are listed with their parameters in the [catalogue](../../reference/catalogue/building_physics.md) and the [API reference](../../reference/api/index.md); the element classes live in `noodl.elements`, and the builders `mass_orifice` and `orifice_law_from_edges` in `noodl.apps.building_physics`.

Two groups of elements appear. The CONTAM-form elements follow CONTAM's equations (NIST TN 1887r1). `RegularizedPowerLaw`, `SplineFlowTable`, the door elements and `AirMedium` follow the Modelica Buildings Library (MBL) v13.0.0. Where two elements describe the same physics they differ in two respects that matter in practice: how the law is regularised near $\Delta p = 0$, where a power law with $n < 1$ has infinite slope, and which air density enters the coefficient.

## Power-law openings

Cracks, leaks, orifices and other small openings: $F = C\,\operatorname{sign}(\Delta p)\,\lvert\Delta p\rvert^{n}$.

- `PowerLaw(C, n)`: flow coefficient $C$ and exponent $n$; $F$ is in whatever flow units $C$ carries.
- `Orifice(Cd, A)`: `PowerLaw` with $C = C_d A \sqrt{2/\rho}$, $n = 0.5$, the sharp-edged orifice as a volume flow in m³/s ($A$ in m²).
- `mass_orifice(Cd, A)`: the same orifice as a mass flow in kg/s, $C = C_d A \sqrt{2\rho}$.
- `orifice_law_from_edges(net, kind)`: one mass-flow `PowerLaw` over every edge of a kind, $C_e = C_{d,e} A_e \sqrt{2\rho}$, reading `Cd` and `area` off each edge.
- `RegularizedPowerLaw(C, m, form=...)`: MBL's power law with exponent $m$, always returning mass flow in kg/s. With `form="volume"` $C$ is a volume-flow coefficient and the result is multiplied by `rho_default`; with `form="mass"` $C$ is a mass-flow coefficient, used as it is. Constructors: `regularized_orifice` (volume form, $C = C_D A \sqrt{2/\rho_{\text{default}}}$), `effective_leakage_area` (volume form from an effective leakage area $L$ at the reference pressure difference `dpRat`: $C = L\,C_{D,\text{Rat}} \sqrt{2/\rho_{\text{default}}}\;\text{dpRat}^{\,0.5 - m}$), `power_law_from_point` (mass form fitted to one measured pair, $k = \dot m_{\text{Mea}} / \Delta p_{\text{Mea}}^{\,m}$), `power_law_from_points` (two pairs: $m = \ln(\dot m_0/\dot m_1) / \ln(\Delta p_0/\Delta p_1)$, then $k = \dot m_0 / \Delta p_0^{\,m}$) and `power_law_coefficient` ($C$ or $k$ and $m$ given directly).

The two families evaluate the same law but differ physically in two places.

- **Near zero pressure difference.** `PowerLaw` replaces the law below `dp_transition` (default $10^{-3}$ Pa) by the straight line $F = k\,\Delta p$ with $k = C\,\Delta p_t^{\,n-1}$, chosen so that the flow is continuous in value at the switch; this is the laminar blend CONTAM applies below a transition Reynolds number. Its slope changes there by the factor $n$. With `regularised=eps` it instead uses $F = C\,\Delta p\,(\Delta p^2 + \varepsilon^2)^{(n-1)/2}$, smooth everywhere. `RegularizedPowerLaw` replaces the law below `dp_turbulent` (default 0.1 Pa, a hundred times wider) by a seventh-order odd polynomial in $\pi = \Delta p / \Delta p_{\text{turb}}$, $F = C\,\Delta p_{\text{turb}}^{\,m}\,\pi\,(a + \pi^2(b + \pi^2(c + \pi^2 d)))$, whose coefficients depend only on $m$ and make the law twice continuously differentiable at the switch.
- **Density.** `Orifice` and `mass_orifice` fold one fixed density into $C$ when they build it (both produce a `PowerLaw`, which itself takes $C$ and $n$ as given). MBL's volume form multiplies by the medium's `rho_default`, its density at the medium's own fixed default state, which does not follow the zone temperatures; the mass form involves no density at all. Neither uses the density of the air actually passing through the opening; `UpstreamDensityPowerLaw`, below, does.

**Source:** `PowerLaw` and `Orifice`: CONTAM's power-law element (TN 1887r1). `RegularizedPowerLaw` and its constructors: MBL's `Buildings.Airflow.Multizone` power-law elements (`Orifice`, `EffectiveAirLeakageArea`, `Point_m_flow`, `Points_m_flow`, `Coefficient_V_flow`, `Coefficient_m_flow`) and their shared regularisation.

## Density upwinding

`UpstreamDensityPowerLaw(C, n, src=, tgt=, m=)` is a `PowerLaw` whose coefficient tracks the density of the air entering the path:

$$
F = \left(\frac{\rho_{\text{up}}}{\rho_{\text{ref}}}\right)^{m} C\,\operatorname{sign}(\Delta p)\,\lvert\Delta p\rvert^{n},
\qquad
\rho_{\text{up}} = \begin{cases} \rho_{\text{src}} & \Delta p \ge 0 \\ \rho_{\text{tgt}} & \Delta p < 0 \end{cases}
$$

$C$ is the coefficient at the reference density $\rho_{\text{ref}}$ (kg/m³, default 1.2041), and the node densities are read from the full-node driver `drivers["rho"]`, the same one the `Stack` head reads. The exponent $m$ says how much density is folded into the coefficient: $m = 0$ for a mass-flow rating, $m = 1/2$ for the orifice family ($C = C_d A \sqrt{2\rho}$, including both openings of a two-orifice doorway), $m = 1$ for a volumetric rating. Air density changes by about 0.086 % per kelvin, so on a 20 K stack a coefficient held at reference density is about 1.7 % wrong in mass flow. The flow stays continuous at $\Delta p = 0$, where the upstream node switches; only its slope jumps there. `dp_transition` stays at reference conditions.

**Source:** CONTAM's evaluation of every power-law element with the density of the entering air (TN 1887r1 section 3.2). The CONTAM reader builds every power-law path as this element.

## Quadratic and tabulated laws

- `Quadratic(a, b)`: the quadratic resistance law $\Delta p = a F + b \lvert F\rvert F$, inverted in closed form: $F = \operatorname{sign}(\Delta p)\, 2\lvert\Delta p\rvert / (\sqrt{a^2 + 4 b \lvert\Delta p\rvert} + a)$. It is linear with slope $1/a$ at small $\Delta p$ and quadratic at large, so it needs no separate regularisation. Requires $a > 0$ and $b > 0$. **Source:** CONTAM's quadratic element family.
- `CubicSplineFlow(x, y, independent="pressure", volume=False, multiplier=1.0)`: a law given by measured points (at least four, every segment rising), fitted as ContamX fits them: straight first and last segments, a cubic spline clamped to them in between, extended straight beyond both ends at the signed argument. `independent="flow"` takes the points as $\Delta p$ against flow and inverts them; `volume=True` turns a volume flow into mass flow with the entering air's density. `SplineCurve(x, y)` is the fit itself. A fitted curve that is not increasing (the cubic part can overshoot even when every segment rises) is refused, since the inversion would be ambiguous. Both return float64 whatever the input dtype. **Source:** CONTAM's cubic-spline elements, measured against ContamX 3.4.1.7.
- `SplineFlowTable(dp_points, flow_points, form=...)`: a measured pressure-flow curve, a monotone cubic Hermite spline through the $(\Delta p, F)$ knots, linearly extrapolated outside them, returning mass flow in kg/s. `form="volume"` scales volume-flow knots by `rho_default` before fitting; `form="mass"` takes mass-flow knots as they are. **Source:** MBL's `Table_m_flow` and `Table_V_flow`. This is a different fit from `CubicSplineFlow`'s.

The same measured points as flow against $\Delta p$ (the forward form) and as $\Delta p$ against flow (`independent="flow"`, inverted) give the same law; `SplineCurve` evaluates and inverts the fit directly.

```python
import torch
from noodl.elements import CubicSplineFlow, SplineCurve

F64 = torch.float64
dp = [0.0, 10.0, 25.0, 50.0, 100.0]                 # Pa
flow = [0.0, 0.02, 0.035, 0.05, 0.07]               # kg/s
forward = CubicSplineFlow(dp, flow, src=[0], tgt=[1])
inverted = CubicSplineFlow(flow, dp, independent="flow", src=[0], tgt=[1])
at = torch.tensor([25.0, 75.0], dtype=F64)
print(forward.flow(at))
print(inverted.flow(at))
curve = SplineCurve(dp, flow)
print(f"{curve.inverse(curve(torch.tensor(40.0, dtype=F64))).item():.6f} Pa")
```

```text
tensor([0.0350, 0.0600], dtype=torch.float64)
tensor([0.0350, 0.0600], dtype=torch.float64)
40.000000 Pa
```

## Large openings and doors

A doorway between two rooms at different temperatures carries flow both ways at once: warm air out through the top, cool air in through the bottom. Three representations are available, at increasing resolution of the pressure profile over the door's height.

**Two lumped orifices.** `add_large_opening(net, a, b, H=, W=, z_mid=, Cd=0.78, kind="airpath")` adds two edges, both oriented `a → b`, each an orifice of area $WH/2$ (m²) at height $z_{\text{mid}} \mp 2H/9$ (m); build their element with `orifice_law_from_edges`. The `Stack` head evaluates the hydrostatic pressure difference at each opening's own height, so a temperature difference drives opposite flows through the low and the high opening. The representation is exact for a neutral plane at mid-height. It uses `PowerLaw`'s laminar blend at $10^{-3}$ Pa and the density convention of the element it is built with. **Source:** CONTAM's two-opening doorway (`DR_PL2`, TN 1887r1 eq. 69–70).

**A closed-form door pair.** `OpenDoor` and `OperableDoor` (built with `open_door_pair` and `operable_door_pair`) represent one door as two edges between the same two zones, both oriented from side A to side B with the same $\Delta p = p_A - p_B$. Each edge carries half of a regularised power-law flow on $\Delta p$, plus or minus a buoyancy exchange term $\dot m_{ABt}$: a regularised square root of the temperature difference $T_A - T_B$ of the two zones, read from `drivers["T"]`. At $\Delta p = 0$ the two edges carry $+\dot m_{ABt}$ and $-\dot m_{ABt}$: pure exchange, with zero net flow. The flows depend on $(\Delta p, T_A, T_B)$ alone, and both terms use the medium's fixed `rho_default`. Parameters, with MBL's defaults: door width `wOpe=0.9` m, height `hOpe=2.1` m, discharge coefficient `CD=0.65`, exponent `m=0.5`, `dp_turbulent=0.01` Pa. `OperableDoor` blends the open door with the closed door's crack by an opening signal $y$ from `drivers[y_key]`: the volume flow is $y V_{\text{open}} + (1 - y) V_{\text{closed}}$, the buoyancy term is scaled by $y$ (the closed crack has none), and the closed crack is an effective-leakage-area law with area `LClo` (m², no default). **Source:** MBL's `DoorOpen` and `DoorOperable`.

**Discretised compartments.** `DoorCompartment` and `OperableDoorCompartment`, with the `DoorCompartmentHead` head term, cut the opening into `nCom` compartments of height $\mathrm{d}h = h_{\text{Ope}} / n_{\text{Com}}$, one edge each; `discretized_door(src=, tgt=, medium=, nCom=10, wOpe=0.9, hOpe=2.1, hA=1.35, hB=1.35, CD=0.65, ...)` and `discretized_operable_door` build a whole door. Compartment $i$ sees the pressure difference at its own height,

$$
\Delta p_i = p_A - p_B + \rho_A g \left(h_A - (i - \tfrac12)\,\mathrm{d}h\right) - \rho_B g \left(h_B - (i - \tfrac12)\,\mathrm{d}h\right),
$$

with `hA` and `hB` the heights (m) of the zones' reference-pressure points above the bottom of the door, so the neutral plane falls wherever the two pressure profiles cross rather than being assumed at mid-height. Each compartment's volume flow is a power law with coefficient $C_D\,\mathrm{d}A \sqrt{2/\rho_{\text{default}}}$, $\mathrm{d}A$ the compartment's area, and it is converted to mass flow with the density of the zone the air comes from, selected per compartment by a smoothed step of width $v_{\text{Zer}}\,\mathrm{d}A$ ($v_{\text{Zer}} = 0.001$ m/s). Unlike the door pair, the densities $\rho_A$, $\rho_B$ are evaluated at the actual port pressure, temperature and water content, read from `drivers["p_abs"]`, `drivers["T"]` and (for a moist medium) `drivers["X_w"]`. The operable variant blends the open and closed coefficients, exponents and areas by $y$.

`DoorPortStream(comp, head, direction, kind)` wraps a whole discretised door as MBL's two port streams instead: one edge carrying `mAB_flow`, positive from A to B (`direction="ab"`), and one carrying `-mBA_flow` (`"ba"`), with the compartments and their heads evaluated inside the element. This is how the Modelica reader builds a discretised door. MBL moves the door's enthalpy, moisture and trace substances on these two port flows, each upwinded on its own sign. A transport over compartment edges would instead upwind each compartment's net flow, which differs inside the smoothed step, where a compartment carries air both ways. The node mass balance is the same either way, since the two streams sum to the compartments' net flow. Each stream's `dflow` reports half the slope of the net flow, so that the pair's summed slope, the only one the airflow Jacobian sees, is exact and each edge's slope stays nonnegative. **Source:** MBL's `DoorDiscretizedOpen` and `DoorDiscretizedOperable`.

## Ducts, dampers, fans, fixed flows and conductances

- `Duct(L, D, eps, sum_C=0.0)`: flow through a duct of length `L`, diameter `D` and roughness `eps` (m) with dynamic loss coefficients `sum_C`: $F = \sqrt{2 \rho A^2 \lvert\Delta p\rvert / (f L/D + \sum C)}$, the friction factor $f$ from the Colebrook equation iterated a fixed number of times, and a straight line through the origin below the transition Reynolds number `Re_t` (2000). The density and viscosity are fixed at construction. **Source:** CONTAM TN 1887r1 section 8.3.3, eq. 50–52.
- `Damper(C_pos, n_pos, C_neg, n_neg)`: a backdraught damper, a separate power law with `PowerLaw`'s laminar blend for each flow direction. With `src`, `tgt` and `m` each side's coefficient follows the entering air's density, $(\rho_{up}/\rho_{ref})^m$, as `UpstreamDensityPowerLaw`. **Source:** CONTAM's `PL_BDF` and `PL_BDQ` elements.
- `FanCurve(coeffs, q_max)`: a fan with cubic pressure rise $P(q) = a_0 + a_1 q + a_2 q^2 + a_3 q^3$, decreasing on $0 \le q \le q_{\max}$; the branch law $\Delta p = -P(q)$ is inverted for $q$. The flow is 0 when the back-pressure exceeds the shutoff pressure $P(0)$, and $q_{\max}$ when it is below $P(q_{\max})$. **Source:** CONTAM's cubic fan element.
- `AffinityFan(curve, src=, tgt=, rho_ref=, speed_key=None, off=0.0, multiplier=1.0, shutoff=None)`: a `FanCurve` at speed ratio $s$ (`drivers[speed_key]`) and density ratio $r = \rho_{up}/\rho_{ref}$ by the fan laws, $F = s\,r\,F_0(\Delta p/(s^2 r))$, times `multiplier` (fans in parallel); below `off` the fan is stopped and the air passes `shutoff`. **Source:** CONTAM's performance-curve fan, measured against ContamX 3.4.1.7.
- `FixedFlow(q0)`: a prescribed supply or exhaust, $F = q_0$ whatever $\Delta p$. A zone connected to the rest of the network only through fixed flows makes the Jacobian singular; `solvers.grounding.spd_certificate` reports that case. **Source:** CONTAM's constant-flow fans.
- `PrescribedFlow(key, kind=)`: $F$ = `drivers[key]` whatever $\Delta p$ (kg/s, one value per edge of the kind), a flow a closure or an input sets every step, such as an air handler's supply, return, recirculation, outdoor-air and exhaust flows (`AirHandlerFlows` writes them).
- `Conductance(g)`: the linear law $F = g\,\Delta p$, used for thermal conduction and passive exchange.

`Scaled(law, key)` multiplies any explicit law's flow (and slope) by a per-edge driver, `drivers[key]`: one value per edge of the law's kind, dimensionless. It is how a schedule or a control acts on a path, in the way CONTAM's path schedule multiplies the path's flow. The inner law's parameters are named `inner.<name>`, and a multiplier of zero closes the edge: a zone left with no open path raises an error naming it.

```python
import torch
from noodl.elements import FixedFlow, Scaled

fan = Scaled(FixedFlow(torch.tensor([0.1], dtype=torch.float64), kind="fan"),
             "flow_multiplier.fan")
print(fan.flow(torch.zeros(1, dtype=torch.float64),
               {"flow_multiplier.fan": torch.tensor([0.5], dtype=torch.float64)}))
```

```text
tensor([0.0500], dtype=torch.float64)
```

A fan on a speed schedule follows the fan laws: at speed ratio 0.8, against 0.8² of the pressure, it delivers 0.8 of the flow. A `PrescribedFlow` passes whatever its driver holds.

```python
import torch
from noodl.elements import AffinityFan, FanCurve, PrescribedFlow

F64 = torch.float64
curve = FanCurve(torch.tensor([400.0, 0.0, -400.0, 0.0], dtype=F64),
                 torch.tensor(1.0, dtype=F64), kind="fan")
fan = AffinityFan(curve, src=[0], tgt=[1], rho_ref=1.2041, speed_key="fan_speed")
drivers = {"rho": torch.full((2,), 1.2041, dtype=F64)}
rise = torch.tensor([-100.0], dtype=F64)            # the fan works against 100 Pa
full = fan.flow(rise, {**drivers, "fan_speed": torch.tensor([1.0], dtype=F64)})
slow = fan.flow(rise * 0.8**2, {**drivers, "fan_speed": torch.tensor([0.8], dtype=F64)})
print(f"{full.item():.4f} kg/s at full speed, flow ratio {(slow / full).item():.3f} at 0.8")
supply = PrescribedFlow("prescribed_flow.supply", kind="supply")
print(supply.flow(torch.zeros(2, dtype=F64),
                  {"prescribed_flow.supply": torch.tensor([0.1, 0.05], dtype=F64)}))
```

```text
0.8660 kg/s at full speed, flow ratio 0.800 at 0.8
tensor([0.1000, 0.0500], dtype=torch.float64)
```

## Duct elements

The laws ContamX evaluates for a duct system, measured against ContamX 3.4.1.7. Each takes the density and the viscosity of the air entering it.

- `AirDuct(length, diameter, roughness, loss=0.0, src=, tgt=)`: a duct segment, $\Delta p = (f L/D + \sum C)\,F^2/(2\rho A^2)$ with $f = 64/Re$ up to Re 2300, the Colebrook equation $1/\sqrt f = -2\log_{10}(\varepsilon/3.7D + 2.51/(Re\sqrt f))$ from Re 3500, and $f$ linear in Re between them; inverted for $F$. `pressure_drop(F, drivers)` evaluates the law itself.
- `DuctTerminal(area, loss, src=, tgt=)`: the zero-length duct between a junction and its zone, $\Delta p = C_e F^2/(2\rho A^2)$, linear below $F_t = 30\,\mu\sqrt{A}$; $C_e = $ `loss` $+ (1-S)/S$ with the opening $S$ = `drivers[opening_key]` clamped to $[10^{-6}, 1]$.
- `FixedVolumeFlow(volume_flow, src=, tgt=)`: a constant volume flow (m³/s) delivered at the density of the node it draws from.
- `air_viscosity(T)`: ContamX's air viscosity, $1.81625\times10^{-5} + 4.92859\times10^{-8}\,(T - 293.15)$ Pa s.

`DuctStack(kind, ...)` (in `noodl.heads`) is the hydrostatic term of a duct segment: the air column between its two junctions at the density of their mean temperature. `Damper(..., src=, tgt=, m=)` follows the entering air's density on each side, as ContamX's duct dampers do. `Duct` remains the TN 1887 form of the Colebrook law (used by the water application); it differs from ContamX's by about 0.1 %.

```python
import torch
from noodl.elements import AirDuct

F64 = torch.float64
duct = AirDuct(torch.tensor([10.0], dtype=F64), torch.tensor([0.2], dtype=F64),
               torch.tensor([9e-5], dtype=F64), src=[0], tgt=[1])
drivers = {"rho": torch.full((2,), 1.204097, dtype=F64)}
print(f"{duct.flow(torch.tensor([1.4536846], dtype=F64), drivers).item():.5f} kg/s")
```

```text
0.05000 kg/s
```

A terminal half open, a constant-volume fan drawing 0.05 m³/s of 20 °C air, and the hydrostatic term of a 3 m vertical duct between junctions at 20 °C and 30 °C:

```python
import torch
from noodl.heads import DuctStack
from noodl.elements import DuctTerminal, FixedVolumeFlow

F64 = torch.float64
rho = torch.tensor([1.2041, 1.1630], dtype=F64)       # 20 C and 30 C at node 0 and 1
drivers = {"rho": rho, "terminal_opening": torch.tensor([0.5], dtype=F64)}
terminal = DuctTerminal(0.01, 1.0, src=[0], tgt=[1], opening_key="terminal_opening")
print(f"{terminal.flow(torch.tensor([5.0], dtype=F64), drivers).item():.5f} kg/s")
fan = FixedVolumeFlow(0.05, src=[0], tgt=[1])
print(f"{fan.flow(torch.zeros(1, dtype=F64), drivers).item():.6f} kg/s")
stack = DuctStack("duct", src=[0], tgt=[1], z_ref=torch.tensor([3.0, 0.0], dtype=F64))
print(f"{stack(drivers).item():.4f} Pa")
```

```text
0.02454 kg/s
0.060205 kg/s
34.8095 Pa
```

## Air properties

The CONTAM-form elements carry one fixed reference density in their coefficients: `RHO_0 = 1.2041` kg/m³ in the building application (`mass_orifice`, `orifice_law_from_edges`, the default of `Duct`, and the reference density of `UpstreamDensityPowerLaw`); the core `Orifice` defaults to 1.2 kg/m³. A temperature-dependent density reaches the airflow through the `density` closure of `build_model` (see [Physics](physics.md#density)), which feeds the `Stack` head and `UpstreamDensityPowerLaw`.

The MBL-derived elements take their densities from a medium. `medium(name)` returns an `AirMedium` for one of `"moist_air"` (`Buildings.Media.Air`), `"moist_air_ideal_gas"` (`Buildings.Media.Specialized.Air.PerfectGas`) and `"dry_air_ideal_gas"` (`Modelica.Media.Air.SimpleAir`); the Modelica class path is accepted too, and is what the medium's `name` reports. Each carries:

- `p_default`, `T_default`, `X_default`: the medium's fixed default state;
- `rho_default`: its density at that state, a plain number computed once and used in the coefficients of every MBL-derived power-law, table and door element. For `Buildings.Media.Air` it is about 1.2 kg/m³ whatever the temperature, because that medium's density depends on pressure only;
- `buoyancy_density(T, X_w)`: the density at `p_default` but at the actual temperature and water content;
- `density(p, T, X_w)`: the medium's own density function at pressure `p`;
- `specific_heat_cp(X_w)`: in J/(kg K).

**Source:** the media models of MBL v13.0.0 and the Modelica Standard Library v4.1.0.

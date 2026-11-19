# Building physics: equations and modelling

The equations the building application solves, the assumptions behind them and how the layers fit together. For the airflow laws of single elements see [Airflow elements](elements.md); for what is not modelled see [Limitations](limitations.md).

Rooms are nodes and leaks, doorways, fans, ducts and dampers are edges. Three layers sit on that one network: an **air layer** that solves for the zone pressures so that mass is conserved at every zone, a **thermal layer** for heat and a **species layer** for contaminants. Heat and contaminants ride on the airflow. Because air density depends on temperature and the stack effect depends on density, the airflow and heat problems are two-way coupled.

## Heat is a transport layer

For a well-mixed zone $i$ of volume $V_i$ at temperature $T_i$, with air mass flows $F_e$ on the paths incident to it, an envelope conductance $UA$ to its neighbours and a heat source $S_i$, the heat balance is

$$
\rho_i c_p V_i \frac{dT_i}{dt} = c_p \sum_e F_e T_{\text{up}(e)} + \sum_j UA_{ij} (T_j - T_i) + S_i
$$

This is the species transport equation $dx/dt = M x + N x_b + \text{sources}/\text{capacity}$ with capacity $\rho c_p V$ (J/K rather than the species layer's kg) and carrier $c_p$ on the advective term, and with the same sign-aware upwind operator: $T_{\text{up}(e)}$ is the temperature of the upwind end of path $e$, chosen by the sign of $F_e$. So heat is not a second physics in the code. It is a second transport layer over the same network, differing from a species layer only in its capacity, its carrier, its units and in having conduction edges.

**Conduction is the Laplacian term.** Edges of the layer's conduction kind, carrying a conductance $UA$, contribute $-UA$ to the diagonal and $+UA$ off the diagonal: the weighted graph Laplacian of that edge set, symmetric and negative semi-definite. It is added to the advective matrix $M$. Advection alone is a Metzler matrix and so preserves positivity (here, the maximum principle) under the time schemes described under [Layers](../../concepts/layers.md); adding a Laplacian keeps that structure, since it too has non-negative off-diagonals. A wall-mass node carries conduction edges and no airpath edge: it is an unknown of the thermal layer and not of a species layer over the same network, which is why a layer's active interior is per layer rather than global.

## The airflow solve and heads

The airflow solve is a nodal problem $A f(A^\top \phi + d) = s$ whose Jacobian is $A \operatorname{diag}(f') A^\top$. The branch head $d$ (the stack and wind terms) does not depend on the pressures $\phi$, and that buys this form and its symmetry. Positive definiteness then follows from the element slopes $f' > 0$ and from the network being grounded (a boundary node, so $A^\top$ has trivial null space), two separate requirements the layer checks separately.

Making $d$ a function of the thermal state would not disturb any of that, since $\phi$ is still the only unknown of the solve. Making it a function of $\phi$ or of the flow $f$ would. A head that closed over the flow it produces (an upstream density evaluated from the flow direction, say) is exactly such a term. A head term is therefore a function of the drivers only: the node densities entering a stack head are computed from the temperatures held fixed for the duration of the pass, and any flow-direction dependence lives inside the element, where it enters $f'$ and keeps the Jacobian symmetric. This is what `UpstreamDensityPowerLaw` does, and why a closure-based reference-density correction could not work: a closure runs before the solve and cannot see the flow direction. CONTAM makes the same choice: its air densities are held at the values from the previous thermal update within an airflow solve. Matching CONTAM's numbers is partly a consequence of matching this convention, not only of matching its element laws.

## Coupling: ping-pong and onion

Since heads are frozen for the pass, airflow and heat exchange information only between passes, and there are two ways to spend passes within one time step (Hensen 1995; Dols and Emmerich 2016).

**Ping-pong** takes exactly one: solve the airflow at the temperatures at the start of the step, advance the temperatures on those flows, and move on. Its error is the lag between the two, a first-order splitting error in the step, cheap and usually harmless when the coupling is weak. **Onion** repeats the pass within the step until the exchanged temperatures stop moving, so the flows and the temperatures at the end of the step are mutually consistent and the splitting error is removed; each pass re-advances the same step from the state at its start. Both are available through `coupling=` (`"pingpong"`, the default, and `"iterate"`); the onion is successive substitution under 0.5 relaxation with a per-instance convergence test. The difference between them is only the within-step lag. It vanishes as $\Delta t \to 0$, so a fine-step ping-pong run and a fine-step onion run agree while a coarse-step ping-pong run is the one that drifts.

Two cautions. Successive substitution converges to a fixed point only where the quasi-steady map is a contraction, and stability of the dynamics is a different question from contractivity of that map. In Li and Delsante's three-root opposing-wind case the upward root is dynamically stable and yet repelling under substitution at 0.5 relaxation (measured slope $-7.756$; any relaxation below 0.228 would contract). Time stepping resolves all three roots because ping-pong stepping has the coupled steady states as its fixed points. And an onion tolerance is absolute in the layer's own units, so it cannot be set below the airflow solve's own residual floor propagated through $dT/dF$. The coupled steady states are checked against closed forms, as listed on the [verification page](../../reference/verification/building.md#against-analytical-solutions).

## Assembling the layers

`build_model(net, air_laws=, heads=, ...)` assembles the model: the air layer, optionally a thermal layer and species layers, the density closure and the head closure. The Python signature is in the [catalogue](../../reference/catalogue/building_physics.md). What the options mean:

- **Heads.** `heads` are the stack and wind terms of the air edges (`Stack`, `DuctStack`, `Wind`, `DriverHead`; empty for none). An `EdgeHeads` closure adds them up once per pass after the density closure and writes the driver `"air.edge_head"`, which the layer adds to every edge's pressure difference once. `closures` follow the head closure, and one that writes a driver a term reads (`"rho"`, say) is refused by name.
- **Thermal layer.** Capacity is $\rho c_p V + C_{\text{node}}$ per active interior node. A node with zero heat capacity that is not storage-free is refused, by name. The scheme defaults to `"exact"`, or to `"backward_euler"` when a storage-free node is touched by a flow edge (a plenum: the only scheme that holds it).
- **Species layer.** The contaminant layer works in mass fractions with capacity equal to the zone air mass $\rho V$ (CONTAM's convention). Each flow edge's filter pass fraction is the layer's `transmission`, and each zone's first-order reactions its `kinetics`.
- **Flow kinds.** `flow_kinds` narrows which edge kinds advect heat and species; it defaults to every air element's kind. Naming a kind no air element provides is refused: a kind the air layer does not solve carries no flow, so heat and species would silently not advect on it.
- **Storage-free and flow-balanced nodes.** Zero-volume nodes that hold no air, such as an air-handler plenum, are listed in `storage_free`; `flow_balanced` nodes join the air layer's boundary (see [Air-handling systems](assembling.md#air-handling-systems)). A contaminant source or sink on a storage-free node is refused by name: it holds no air to emit into.
- **Coupling.** `coupling` is `"pingpong"` or `"iterate"` (see above), with `iterate_tol` and `iterate_max` for the latter.

### Density

`density` selects the closure relating temperature to air density.

- `"ideal_gas"` (the default) uses `IdealGasDensity`: $\rho = P_{\text{ref}} / (R T)$, with `P_ref` overridable per call through `drivers["P_ref"]`.
- `"linear"` uses `LinearDensity`: the Boussinesq form $\rho = \rho_0\,(1 - (T - T_0)/T_0)$, which is what the Li and Delsante analytical solutions assume.

### Conduction, surface films and radiant star nodes

The thermal layer's `conduction_kind` is one edge kind or a tuple, for example `("wall", "film", "radiation")`: wall conduction, a surface (film) coefficient and a radiative coefficient as separate edges, each with its own `ua` (W/K), concatenated in the order given into the layer's conductance. By default it is `"wall"` when the network has wall edges; naming a kind with no edge is an error that names it. A closure that recomputes a coefficient from the temperatures, such as a radiative one, writes the per-step driver `"thermal.conductance"` in that order.

A wall surface has no heat capacity of its own: it sits between the zone air (a film coefficient) and the first wall layer (conduction), and in a radiant exchange it joins the zone's star node (a radiative coefficient). Name such nodes in `storage_free` together with the `("wall", "film", "radiation")` edges and the layer folds them into their neighbours. Each surface film in series with the wall behind it becomes one combined surface coefficient, and the star node with its surfaces becomes equivalent conductances between the surfaces' neighbours (the star-delta transform). Absorbed sunlight and the radiant share of gains, given as `sources` on the surface or star nodes, are handed on to their neighbours the same way. The layer keeps the exact scheme, and returns the surface and star temperatures in the state, computed from their neighbours after each step. A radiative coefficient that a closure re-evaluates each pass is folded with its current value.

### Walls

`WallMass(name, capacity, ua_zone, ua_ambient)` is a lumped wall node (see [Assembling a building](assembling.md#zones-and-walls)). It refuses a non-positive conductance at construction: a negative one makes the conduction operator anti-diffusive, violating the maximum principle, and every solve would still report success because nothing downstream looks at the sign. Zero is refused as a wall that conducts nothing. The conductances are plain floats, so gradients do not reach them (see [Limitations](limitations.md)).

## Units and precision

Mass flow is in kg/s, temperature in K, heat in W, mass fraction in kg/kg and pressure in Pa. The whole application works in `float64` regardless of the network's default dtype.

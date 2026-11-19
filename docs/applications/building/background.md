# Building physics: theoretical background

Background for the [building physics application](index.md), moved from the framework theory page ([theory](../../concepts/theory.md)).

## 7. Heat as a transport layer, and coupling modes

**The heat balance IS the transport equation.** For a well-mixed zone $i$ of volume $V_i$
at temperature $T_i$, with air mass flows $F_e$ on the paths incident to it, an envelope
conductance $UA$ to its neighbours and a heat source $S_i$,

$$
\rho_i c_p V_i \frac{dT_i}{dt} = c_p \sum_e F_e T_{\text{up}(e)} + \sum_j UA_{ij} (T_j - T_i) + S_i
$$

which is exactly the species transport equation $dx/dt = M x + N x_b + \text{sources}/\text{capacity}$
with `capacity` $= \rho c_p V$ (J/K rather than the species layer's m³ or kg), `carrier` $= c_p$
on the advective term, and the SAME sign-aware upwind operator: $T_{\text{up}(e)}$ is the temperature
of the upwind end of path $e$, chosen by the sign of $F_e$, which is what
`AdvectionOperator` already does for species. So heat is not a second physics in the code, it
is a second `TransportLayer` over the same typed graph, differing from a species layer only
in its capacity, its carrier, its `quantity`/`unit` tags, and in having conduction edges.

**Conduction is the Laplacian term.** Edges of the layer's `conduction_kind`, carrying a
conductance $UA$, contribute $-UA$ to the diagonal and $+UA$ off-diagonal -- the weighted
graph Laplacian of that edge set, symmetric and negative semi-definite -- which is added to
the advective $M$. Advection alone is a Metzler matrix and so positivity- (here,
maximum-principle-) preserving under the schemes of section 6; adding a Laplacian keeps that
structure, since it too has non-negative off-diagonals. A wall-mass node carries conduction
edges and NO airpath edge: it is an unknown of the thermal layer and not of a species layer
over the same network, which is why a layer's active interior is per layer rather than
global.

**Why a head is a function of the drivers alone.** The airflow solve is a nodal problem
$A f(A^\top \phi + d) = s$ whose Jacobian is $A \operatorname{diag}(f') A^\top$. It is the branch head $d$ (the
stack and wind terms) NOT depending on $\phi$ that buys this form, and with it the symmetry;
positive definiteness then follows from the element slopes $f' > 0$ and from the network
being grounded (a boundary node, so $A^\top$ has trivial null space), which are separate
requirements the layer checks separately. Making $d$ a function of the thermal STATE would
not disturb any of that -- $\phi$ is still the only unknown of the solve -- but making it a function of $\phi$ or of $f$ would, and
a head that closed over the flow it produces (say, an upstream density evaluated from the
flow direction) is exactly such a term. The framework therefore defines a head term as a
function of the DRIVERS only: the node densities entering a stack head are computed from the
temperatures held fixed for the duration of the pass, and any flow-direction dependence lives
inside the ELEMENT, where it enters $f'$ and keeps the Jacobian symmetric (this is what
`UpstreamDensityPowerLaw` does, and why a closure-based reference-density correction could
not work: a closure runs before the solve and cannot see the flow direction). CONTAM makes the same choice -- its air densities are held at the values from the
previous thermal update within an airflow solve -- so matching CONTAM's numbers is partly a
consequence of matching this convention, not only of matching its element laws.

**Ping-pong and onion (Hensen 1995; Dols and Emmerich 2016).** Since heads are frozen for
the pass, airflow and heat exchange information only BETWEEN passes, and there are two ways
to spend passes within one time step. **Ping-pong** takes exactly one: solve the airflow at
the temperatures at the start of the step, advance the temperatures on those flows, and move
on. Its error is the lag between the two -- a first-order-in-$\Delta t$ splitting error, cheap and
usually harmless when the coupling is weak. **Onion** repeats the pass within the step until
the exchanged temperatures stop moving, so the flows and the temperatures at the end of the
step are mutually consistent and the splitting error is removed; each pass re-advances the
same step from the state at its start. `noodl` implements both (`Model(coupling=...)`),
the onion as successive substitution under 0.5 relaxation with a per-instance convergence
test. The difference between them is therefore ONLY the within-step lag: it vanishes as
$\Delta t \to 0$, which is why a fine-step ping-pong run and a fine-step onion run agree while a
coarse-step ping-pong run is the one that drifts -- the table
`tests/verification/test_natural_ventilation.py` prints.

Two warnings. Successive substitution converges to a fixed point
only where the quasi-steady map is a contraction, and stability of the DYNAMICS is a
different question from contractivity of that map: in Li and Delsante's three-root opposing
wind case the upward root is dynamically stable and yet repelling under substitution at 0.5
relaxation (measured slope −7.756; any relaxation below 0.228 would contract). Time stepping
resolves all three roots because ping-pong stepping has the coupled steady states as its
fixed points. And an onion tolerance is absolute in the layer's own units, so it cannot be
set below the airflow solve's own residual floor propagated through $dT/dF$.

**Verification.** The closed forms of Li and Delsante (2001, *Building and Environment*
36:59-71) -- buoyancy alone, buoyancy against an envelope loss, assisting wind, and the
three-root opposing-wind case -- are the reference for the coupled steady states, together
with Brown and Solvason's doorway exchange, an $m c / UA$ wall RC time constant, and a
two-zone doorway against an independent `scipy` root find; ContamX itself is the reference
for the transient and for the stack. See [Building physics](index.md#verification).

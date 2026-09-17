# Theoretical background for `tellegen`

> Status (11 Sep 2026): research survey drafted by an AI agent from web sources; claims are
> to be verified against the cited sources. The design spec
> (`docs/superpowers/specs/2026-09-11-tellegen-framework-design.md`) supersedes this note
> where they differ; in particular the framework is nodal-primary with the cycle space
> retained for latent flows, not loop-primary as section 3 states.

A general-purpose, differentiable network solver for conservative transport — pressures
and buoyancy in multizone buildings, heat, contaminant species, sewer flow, and urban
street-canyon exchange — needs one topology layer and one small set of constitutive
patterns that specialise per physical domain. This note gives the graph-theoretic and
physical foundations, surveys existing multi-physics network solvers for comparison, and
covers the differentiable-programming and co-simulation questions that follow from
building this in PyTorch. Uncertain or unverified claims are flagged explicitly.

## 1. Graph fundamentals: incidence, cycle, cutset, and Tellegen's theorem

Represent the network as a directed graph with `n` nodes and `b` branches. The
**incidence matrix** `A` (`n × b`) has `A[i,e] = +1` if branch `e` leaves node `i`, `-1`
if it enters, `0` otherwise (this is exactly `Network.incidence()` in
`src/tellegen/topology.py`). For a connected graph `A` has rank `n − 1`; one row is
redundant because every column sums to zero (each branch leaves one node and enters
another).

**Kirchhoff's current law (KCL)** is conservation at every node:

```
A f = s
```

where `f ∈ R^b` are branch flows and `s ∈ R^n` are external sources/sinks (zero at
interior nodes). **Kirchhoff's voltage law (KVL)** says branch potential differences are
consistent with a single-valued nodal potential `φ ∈ R^n`:

```
e = A^T φ        (branch effort = "target minus source", i.e. gradient)
```

A **fundamental cycle matrix** `B` (`l × b`, `l = b − n + c` for `c` connected
components) is built from a spanning forest: each non-tree ("chord") branch closes a
cycle through the tree, giving one row of ±1/0 entries (`Network.cycle_basis()`
implements this directly, walking the tree path in `networkx`). Rows of `B` span the
**cycle space**, the null space of `A`: `A B^T = 0`. Dually, a **fundamental cutset
matrix** `Q` — one row per tree branch, listing the tree branch plus the chords whose
fundamental cycle uses it — spans the **cutset space**, and `A` itself is a (redundant)
generator of that same space, since removing all branches crossing a node cut is a
special cutset. The cycle space and cutset space are orthogonal complements of `R^b`, and
`A B^T = 0` is the algebraic expression of that fact ([cutset/cycle-space duality,
Wikipedia](https://en.wikipedia.org/wiki/Tellegen%27s_theorem)).

**Tellegen's theorem** (Tellegen 1952; see also Penfield, Spence & Duinker,
*Tellegen's Theorem and Electrical Networks*, MIT Press 1970): for *any* `f` satisfying
`A f = 0` and *any* `e` satisfying `e = A^T φ` for some `φ` — on the same graph, but not
necessarily from the same physical system or even the same instant in time —

```
e^T f = φ^T (A f) = 0.
```

The proof is one line, but the content is large: `f` need only be any element of the
cycle space (no constitutive law required), and `e` need only be a gradient.
Consequences used throughout `tellegen`:

- **Power balance / energy conservation**: if `e, f` are the *same* system's effort and
  flow, `Σ e_k f_k = 0` is the statement that stored power equals dissipated/supplied
  power (`Network.power_residual` is exactly this residual, and is a very cheap
  correctness check on any solver state — it should vanish to numerical precision
  regardless of the constitutive laws used).
- **Adjoint sensitivities / reciprocity**: if `e` is the *adjoint* solution of a network
  with the same topology but sources placed at the point where a sensitivity is wanted,
  Tellegen's theorem gives the sensitivity of any output to any parameter as a single
  extra solve — this is precisely the classical circuit-theory reciprocity theorem, and
  it is the physical reading of the implicit-function-theorem adjoint used for autograd
  in §4.
- **Cross-checks between different times/systems**: because `f` and `e` need not come
  from the same evaluation, Tellegen's theorem underlies mixed potential/Brayton–Moser
  formulations (the origin of the 2019 `legacy/Tellegen` package) and time-domain
  reciprocity identities.

### Two dual formulations

**Nodal analysis.** Unknowns are the `n − 1` independent node potentials `φ`. Given a
branch constitutive law `f = g(e) = g(A^T φ)`, KCL becomes

```
A g(A^T φ) = s.
```

For a linear conductance law `f = G A^T φ` (`G = diag` of conductances) this is

```
(A G A^T) φ = s,    L := A G A^T
```

a weighted graph **Laplacian**, symmetric positive semi-definite (PD after grounding one
node) — the sparse SPD system familiar from resistive-network and finite-volume
diffusion solves, amenable to Cholesky or CG. Nonlinear `g` (turbulent orifices,
power-law cracks, quadratic pipe friction) is solved by Newton's method with Jacobian
`A G'(A^T φ) A^T` — the same congruence structure at every iteration, i.e. a **Schur
complement** of the branch Jacobian onto the nodal potentials. Nodal analysis handles
flow sources trivially (add to `s`) but a branch whose *natural* law fixes a flow or
potential directly (an ideal pump, a fixed-flow fan, a prescribed pressure boundary)
cannot be written as `f = g(e)`; this is handled by **Modified Nodal Analysis (MNA)**,
which promotes that branch's flow to an extra unknown and adds one extra row enforcing
its constitutive constraint, at the cost of a larger, indefinite (saddle-point) system
([MNA overview](https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA1.html)).

**Loop (mesh) analysis.** Unknowns are the `l = b − n + c` cycle amplitudes `λ`. Set

```
f = B^T λ + f_p ,     A f_p = s
```

which satisfies KCL *by construction*, for any `λ` (this is exactly
`physics/flows.py: branch_flows`, which builds `f = λ @ B` so mass conservation is exact
to floating-point precision independent of solver convergence — a deliberate design
choice in `tellegen`). What remains is KVL on the branch effort law `e = h(f)`:

```
B h(B^T λ + f_p) = 0.
```

Newton's method gives Jacobian `B H'(f) B^T`, the loop-impedance analogue of the nodal
Laplacian, again a Schur complement (`H' = dh/df`). Loop analysis needs only `l`
unknowns rather than `n − 1`, favourable for tree-like networks with a handful of loops
(most building airflow networks), and flow sources become part of the particular
solution rather than extra unknowns; a branch with a *prescribed potential drop* needs
the dual augmentation instead. Neither formulation is unconditionally better-conditioned
— both reduce to a congruence transform of the same branch Jacobian — so the practical
choice is driven by `n` vs `l` and by which sources are naturally flow- or
potential-type. `Network.assert_forward_oriented` encodes a related implementation
constraint: only an entrywise non-negative cycle basis guarantees non-negative `λ` maps
to non-negative `f`, needed for the constant upwind operator of §2.

## 2. Port-Hamiltonian systems, bond graphs, and advected quantities

Bond graphs (Paynter 1961; Breedveld) generalise circuit theory to any energy domain via
conjugate **effort/flow** pairs whose product is power:

| Domain | Effort `e` | Flow `f` |
|---|---|---|
| Electrical | voltage `V` | current `I` |
| Hydraulic/pneumatic | pressure `p` | volumetric flow `Q` |
| Thermal (true bond graph) | temperature `T` | entropy flow `dS/dt` |
| Thermal (pseudo bond graph, common engineering use) | temperature `T` | heat flow `Φ = dQ/dt` |
| Chemical | chemical potential `μ` | molar flow `ṅ` |
| Species (pseudo bond graph) | concentration `c` | mass flow `ṁ` |

(For the true bond graph pair `T·(dS/dt)` has units of power; the pseudo-bond-graph
`T·Φ` pairing is a convenient engineering approximation, not literally power — this
distinction matters if `tellegen`'s thermal module is ever audited for exact energy
balance, and should be flagged as a modelling choice, not swept under Tellegen's
theorem.) Van der Schaft & Maschke, ["Port-Hamiltonian Systems on
Graphs"](https://pure.rug.nl/ws/files/2340945/2013SIAMJContOptimvdSchaft.pdf) (SIAM J.
Control Optim. 51:906–937, 2013; [arXiv:1107.2006](https://arxiv.org/pdf/1107.2006))
give the general framework: the incidence matrix defines a **Dirac structure** relating
edge flows/efforts to vertex flows/efforts, `f_v = -A f_e`, `e_e = A^T e_v`, so that
storage, dissipation and ports can be attached at either edges or vertices while power
balance (`dH/dt` = power in through ports − dissipation) holds identically — the
continuous-time, energy-storing generalisation of Tellegen's theorem, and the natural
home for `tellegen`'s "storage at nodes, typed edges" design.

**The subtlety for heat and species.** A resistive bond-graph element has `f = g(e)`: the
flow is driven by the effort *difference alone* (Fourier conduction, `q = k(T_i − T_j)`).
But heat and species carried by an airflow are **advected**: the branch flux is
`ṁ_e · c_upstream(e)`, the *product* of a flow variable that itself solves a different
(air-pressure) network with the scalar value *at whichever node is upstream* for the
current flow direction. This is not a function of a potential difference at all — it is
a directed, flow-weighted selection. Represent it with the **upwind operator**
`U(f) ∈ R^{b×n}`, `U[e,i] = 1` iff node `i` is upstream of branch `e` given the sign of
`f_e` (`Network.upwind`), and build the **advection matrix**

```
M(f) = A diag(f) U(f)     (n × n)
```

so the node balance for a scalar `c` (temperature, concentration) with nodal storage
`V dc/dt = -M(f) c + s` reproduces exactly `physics/species.py`. Because `f ≥ 0` on a
forward-oriented graph, `-M(f)` is a (singular) **M-matrix** — off-diagonal entries
non-negative, columns summing to `≤ 0` — precisely the generator structure of a
continuous-time Markov chain / compartmental system, which guarantees the flow
**preserves positivity** of `c` for any time step (see
[Wikipedia: M-matrix](https://en.wikipedia.org/wiki/M-matrix) and the positive-systems
literature on Metzler matrices). Genuine conduction/diffusion (envelope U-values, pipe
wall conduction, turbulent diffusivity between street-canyon cells) adds a *symmetric*
Laplacian term `A K A^T` (§1) on top, giving the standard convection–diffusion operator
on a graph, `V dc/dt = -(A K A^T) c - M(f) c + s`.

## 3. Existing multi-physics network solvers

| Solver | Formulation | Nonlinearity | Thermal/species coupling |
|---|---|---|---|
| NIST **CONTAM**/AIRNET (Walton 1989) | Nodal (zone pressures) | Newton on power-law/quadratic orifice flows | Contaminant transport: a second nodal solve on well-mixed zones, using AIRNET's converged flows |
| **COMIS** | Nodal, same family | Newton | Similar; largely superseded, merged into EnergyPlus AFN |
| **EnergyPlus AirflowNetwork** | Nodal (zones = nodes) | Newton on Bernoulli/power-law components ([Engineering Reference](https://bigladdersoftware.com/epx/docs/9-4/engineering-reference/airflownetwork-model.html)) | Airflow solved first each timestep; resulting flows feed the zone heat balance (sequential/operator-split) |
| **Modelica Buildings** (Wetter et al.) | Acausal DAE; **stream connectors** carry `p, m_flow, h_outflow, Xi_outflow, C_outflow` together ([spec](https://specification.modelica.org/master/stream-connectors.html)) | Tool-dependent Newton after index reduction | Heat and species ride along with mass flow *by connector design* — closest existing analogue to `tellegen`'s intent |
| **pandapipes**/**pandapower** | Nodal Newton (Y-bus-like for power; pressure/temperature nodes for pipes) | Newton, linearised pipe friction each step | Sequential heat-network extension in pandapipes |
| **EPANET** (Rossman) | Nodal heads via the **Global Gradient Algorithm** (Todini & Pilati 1988) | One sparse linear solve for heads + a scalar flow-update per link, each Newton iteration ([EPANET 2.2 docs](https://epanet22.readthedocs.io/en/latest/12_analysis_algorithms.html)) | Water-quality solved as a second, decoupled transport step on converged flows |
| **WNTR** | Python wrapper around EPANET's engine | as EPANET | Adds resilience/failure scenario analysis |
| CityLearn-type digital twins | Wraps EnergyPlus/others for RL | n/a (black box) | n/a |

None of these is differentiable in the autograd sense (uncertain: no published
"differentiable EPANET/CONTAM" was found; the closest is differentiable AC/DC
[power-flow optimisation](https://arxiv.org/pdf/2603.28203)). Two patterns matter for
`tellegen`: every mature tool is **nodal** in the pressure/head variable (cheap, robust
SPD systems), whereas `tellegen` deliberately chose the **loop** formulation for air
networks to get exact conservation by construction (§1) — an intentional trade-off; and
Modelica's stream connectors are the strongest existing precedent for carrying enthalpy
and species *with* the flow, matching the advection-matrix construction in §2.

For sewer networks, EPA's **SWMM** solves the full 1-D Saint-Venant equations (dynamic
wave routing) plus a nodal continuity equation at junctions, supporting surcharging and
reverse flow — a genuinely dynamic (not quasi-steady) network, more demanding than the
quasi-steady airflow case. For street-canyon networks, Soulhac et al.'s **SIRANE** is
explicitly built on a street/intersection graph with a nodal mass balance at
intersections and parametrised exchange with a background compartment at roof level —
structurally very close to the multizone-airflow case, just with different branch and
node constitutive laws.

## 4. Differentiable physics

Backpropagating through an iterative Newton solve `F(x, θ) = 0 → x*(θ)` need not
differentiate the iterations themselves. By the **implicit function theorem**,

```
dx*/dθ = -(∂F/∂x)^{-1} (∂F/∂θ),
```

so the backward pass is one linear solve with the (transposed) converged Jacobian —
`O(1)` extra memory regardless of iteration count, and (§1) *exactly* the classical
adjoint-network / reciprocity computation Tellegen's theorem licenses: run the network
once forward, once "backward" with sources moved to where the gradient is wanted, and
read sensitivities off a single extra solve. `torch.linalg.solve`'s built-in backward
implements this adjoint rule directly, so building the branch Newton step and the
loop/nodal linear solve on top of it (rather than a custom non-differentiable
factorisation) gives calibration/data-assimilation gradients "for free." Related
libraries: `torchdiffeq` (adjoint-sensitivity Neural ODEs, Chen et al. 2018); on the JAX
side, `jaxopt`'s implicit-differentiation layer (Blondel et al., ["Efficient and Modular
Implicit Differentiation"](https://arxiv.org/pdf/2105.15183), NeurIPS 2021) and
Optimistix/Diffrax; Theseus (Meta) and OptNet (Amos & Kolter 2017) are differentiable
nonlinear/QP solver layers in the same family; Deep Equilibrium Models (Bai et al. 2019)
popularised implicit differentiation for fixed-point layers generally. **Caveat**:
PyTorch's native sparse-tensor autograd coverage is limited, so a practical `tellegen`
sparse solver likely needs a custom `autograd.Function` wrapping a sparse solve with an
*analytic* adjoint via the rule above rather than autograd tracing the factorisation;
recent work (`torch-sla`, [arXiv:2601.13994](https://arxiv.org/html/2601.13994v2))
targets this gap but is new and unverified for production use here.

## 5. Co-simulation and coupling for digital twins

**FMI** (Functional Mock-up Interface, Modelica Association) defines two interface
kinds: **Model Exchange (ME)**, where the importer owns the integrator and needs direct
access to state derivatives, and **Co-Simulation (CS)**, where each FMU carries its own
internal solver and only exchanges signals at communication points — the natural fit for
coupling independently developed tools. FMI 3.0 (2022) adds clocks/scheduled execution,
binary and array variables, and layered standards for hybrid/embedded use
([FMI 3.0 release notes](https://fmi-standard.org/news/2022-05-10-fmi-3.0-release/)).
Within a coupling step, **Jacobi** coupling advances all subsystems in parallel from the
previous step's exchanged values (parallelisable, may need iteration/smaller steps for
stiff couplings), while **Gauss-Seidel** coupling advances subsystems sequentially,
letting later ones see already-updated values from earlier ones in the same step (faster
convergence, serial). Dols & Emmerich's CONTAM–EnergyPlus co-simulation work (*Building
Simulation* 2016, [10.1007/s12273-016-0279-2](https://link.springer.com/article/10.1007/s12273-016-0279-2))
uses the terms **ping-pong** (one exchange per macro-step, no within-step iteration —
loose coupling, cheap but can lag/mis-converge for stiff pressure–thermal interaction)
and **onion** (iterate the exchange within a macro-step until convergence before
advancing — tight coupling, more accurate, more expensive). **Waveform relaxation**
generalises Gauss-Seidel/Jacobi from single time points to whole time-history "waveforms"
over a window, iterating until the exchanged trajectories converge before accepting them
(originated in VLSI circuit simulation, Lelarasmee, Ruehli & Sangiovanni-Vincentelli
1982); it is the natural generalisation when coupling windows are long relative to the
fastest coupled timescale.

For a Python/PyTorch solver whose entire point is end-to-end autograd, I would recommend
*not* defaulting to opaque FMI co-simulation for internal coupling: expose typed ports
(pressure/flow, temperature/heat-flow, concentration/species-flow — directly analogous to
Modelica's stream connectors) and couple modules with a differentiable Jacobi or Picard
iteration whose fixed point is itself differentiated via the implicit function theorem
(§4), reserving FMI (likely CS, since ME would require exposing PyTorch's internal state
derivatives through a C API) purely as the boundary adapter to legacy, non-differentiable
tools such as EnergyPlus, where gradients across that boundary are either unavailable or
must be obtained by a surrogate/finite-difference approximation. This is a design
recommendation, not an established result — I did not find published work coupling a
PyTorch network solver to FMI, so treat it as a hypothesis to validate.

On topology and metadata standards for auto-generating the graph: **Brick schema**
(Balaji et al., BuildSys 2016) and the W3C **Building Topology Ontology (BOT)** are
complementary — BOT for spatial containment/adjacency (sites, storeys, spaces, elements),
Brick for equipment/point semantics — and both can plausibly seed a `tellegen.Network`'s
nodes and edge `kind`s from a BAS point list or **IFC/BIM** model (`IfcSpace` adjacency
for airflow-path topology, `IfcDistributionElement` for duct/pipe branches). **Project
Haystack** is a competing/overlapping industry tagging convention for the same purpose.
**ISO 23247** ("digital twin framework for manufacturing") is explicitly a
manufacturing-domain standard; any building-digital-twin use is by analogy, not direct
applicability — flagged as uncertain and likely needs a building-specific standard (none
as mature as ISO 23247 currently exists for buildings, to my knowledge).

## 6. Time integration for stiff advection-reaction on networks

For a linear, time-invariant step `dc/dt = A_c c + b` (flows frozen over the step — valid
whenever the pressure/flow network is quasi-steady relative to thermal/species
timescales, the standard multizone assumption also made by CONTAM and EnergyPlus AFN),
the **exact** solution is `c(t+Δt) = e^{A_c Δt} c(t) + A_c^{-1}(e^{A_c Δt} - I) b`. When
`A_c` is singular (e.g. a node with zero net flow), Van Loan's **augmented-matrix trick**
(["Computing Integrals Involving the Matrix
Exponential"](https://www.olemartin.no/artikler/vanloan.pdf), IEEE TAC 23:395–404, 1978)
avoids inverting `A_c` by exponentiating the block matrix

```
Z = [[A_c Δt, b Δt], [0, 0]],   e^Z = [[e^{A_c Δt}, Φ], [0, I]]
```

with `Φ` the exact integrated forcing term — exactly what `physics/species.py`
implements for its two-block augmented system. Because this step is *exact* for the
linear operator, it inherits whatever positivity/conservation structure `A_c` has: since
`A_c` is (negative) Metzler/M-matrix (§2), `e^{A_c Δt}` is entrywise non-negative for
*any* `Δt > 0` — unconditional positivity, with no CFL-type step restriction, which is
the chief argument for exponential integration over explicit upwind finite-volume
stepping (Hochbruck & Ostermann, ["Exponential
Integrators"](https://na.math.kit.edu/download/papers/acta-final.pdf), Acta Numerica
19:209–286, 2010). **Implicit Euler**, `(I - A_c Δt) c^{n+1} = c^n + b Δt`, is
unconditionally stable and (since `(I - A_c Δt)^{-1}` is the inverse of an M-matrix, hence
non-negative) also positivity-preserving, but only first-order accurate, so it
systematically over-diffuses compared with the exact exponential step. **Crank-Nicolson**,
`(I - A_c Δt/2) c^{n+1} = (I + A_c Δt/2) c^n + b Δt`, is second-order but is *not*
unconditionally positivity-preserving — for `Δt` large relative to the fastest mode of
`A_c` it can produce spurious oscillation/negative concentrations, a well-known defect
inherited from its use on diffusion equations. For the coupled system, **operator
splitting** is the natural fit: solve the (generally nonlinear, algebraic) pressure/flow
network first on the slow control step (§1, Newton on the loop or nodal system), freeze
the resulting `f`, then advance heat/species exactly (or via implicit Euler for stiff
reaction terms) over that step — a Lie/Strang splitting between the "instantaneous"
flow-equilibration timescale and the finite thermal/species timescale, matching the
quasi-steady-state assumption already built into CONTAM, COMIS and EnergyPlus AFN.

## 7. Heat as a transport layer, and coupling modes

*(Written from the implementation, milestone 2, not from the survey: this section describes
what `tellegen` does and why, and its verification cases are in the repository.)*

**The heat balance IS the transport equation.** For a well-mixed zone `i` of volume `V_i`
at temperature `T_i`, with air mass flows `F_e` on the paths incident to it, an envelope
conductance `UA` to its neighbours and a heat source `S_i`,

```
rho_i c_p V_i dT_i/dt = c_p sum_e F_e T_up(e) + sum_j UA_ij (T_j - T_i) + S_i
```

which is exactly the species transport equation `dx/dt = M x + N x_b + sources / capacity`
with `capacity = rho c_p V` (J/K rather than the species layer's m^3 or kg), `carrier = c_p`
on the advective term, and the SAME sign-aware upwind operator: `T_up(e)` is the temperature
of the upwind end of path `e`, chosen by the sign of `F_e`, which is what
`AdvectionOperator` already does for species. So heat is not a second physics in the code, it
is a second `TransportLayer` over the same typed graph, differing from a species layer only
in its capacity, its carrier, its `quantity`/`unit` tags, and in having conduction edges.

**Conduction is the Laplacian term.** Edges of the layer's `conduction_kind`, carrying a
conductance `UA`, contribute `-UA` to the diagonal and `+UA` off-diagonal -- the weighted
graph Laplacian of that edge set, symmetric and negative semi-definite -- which is added to
the advective `M`. Advection alone is a Metzler matrix and so positivity- (here,
maximum-principle-) preserving under the schemes of section 6; adding a Laplacian keeps that
structure, since it too has non-negative off-diagonals. A wall-mass node carries conduction
edges and NO airpath edge: it is an unknown of the thermal layer and not of a species layer
over the same network, which is why a layer's active interior is per layer rather than
global.

**Why a drive is a function of the drivers alone.** The airflow solve is a nodal problem
`A f(A^T phi + d) = s` whose Jacobian is `A diag(f') A^T`. It is the branch head `d` (the
stack and wind terms) NOT depending on `phi` that buys this form, and with it the symmetry;
positive definiteness then follows from the element slopes `f' > 0` and from the network
being grounded (a boundary node, so `A^T` has trivial null space), which are separate
requirements the layer checks separately. Making `d` a function of the thermal STATE would
not disturb any of that -- `phi` is still the only unknown of the solve -- but making it a function of `phi` or of `f` would, and
a drive that closed over the flow it produces (say, an upstream density evaluated from the
flow direction) is exactly such a term. The framework therefore defines a `Drive` as a
function of the DRIVERS only: the node densities entering a stack head are computed from the
temperatures held fixed for the duration of the pass, and any flow-direction dependence lives
inside the ELEMENT, where it enters `f'` and keeps the Jacobian symmetric (this is what
`UpstreamDensityPowerLaw` does, and why the spec's proposed closure-based
`ReferenceCorrection` could not work: a closure runs before the solve and cannot see the flow
direction). CONTAM makes the same choice -- its air densities are held at the values from the
previous thermal update within an airflow solve -- so matching CONTAM's numbers is partly a
consequence of matching this convention, not only of matching its element laws.

**Ping-pong and onion (Hensen 1995; Dols and Emmerich 2016).** Since drives are frozen for
the pass, airflow and heat exchange information only BETWEEN passes, and there are two ways
to spend passes within one time step. **Ping-pong** takes exactly one: solve the airflow at
the temperatures at the start of the step, advance the temperatures on those flows, and move
on. Its error is the lag between the two -- a first-order-in-`dt` splitting error, cheap and
usually harmless when the coupling is weak. **Onion** repeats the pass within the step until
the exchanged temperatures stop moving, so the flows and the temperatures at the end of the
step are mutually consistent and the splitting error is removed; each pass re-advances the
same step from the state at its start. `tellegen` implements both (`Model(coupling=...)`),
the onion as successive substitution under 0.5 relaxation with a per-instance convergence
test. The difference between them is therefore ONLY the within-step lag: it vanishes as
`dt -> 0`, which is why a fine-step ping-pong run and a fine-step onion run agree while a
coarse-step ping-pong run is the one that drifts -- the table
`tests/verification/test_natural_ventilation.py` prints.

Two warnings the implementation earned. Successive substitution converges to a fixed point
only where the quasi-steady map is a contraction, and stability of the DYNAMICS is a
different question from contractivity of that map: in Li and Delsante's three-root opposing
wind case the upward root is dynamically stable and yet repelling under substitution at 0.5
relaxation (measured slope -7.756; any relaxation below 0.228 would contract). Time stepping
resolves all three roots because ping-pong stepping has the coupled steady states as its
fixed points. And an onion tolerance is absolute in the layer's own units, so it cannot be
set below the airflow solve's own residual floor propagated through `dT/dF`.

**Verification.** The closed forms of Li and Delsante (2001, *Building and Environment*
36:59-71) -- buoyancy alone, buoyancy against an envelope loss, assisting wind, and the
three-root opposing-wind case -- are the reference for the coupled steady states, together
with Brown and Solvason's doorway exchange, an `m c / (UA)` wall RC time constant, and a
two-zone doorway against an independent `scipy` root find; ContamX itself is the reference
for the transient and for the stack. All live in `tests/verification/`.

## 8. Street canyons as a transport layer

*(Written from the implementation, milestone 3: this section describes what `apps/street`
does and why, and its verification cases are in the repository.)*

**The street balance IS the framework's transport equation, again.** For a canyon street
`i` of length `L_i`, width `W_i` and height `H_i`, holding a well-mixed pollutant mass
concentration `x_i` (kg/m3),

```
L_i W_i H_i dx_i/dt = sum_r Q_r x_up(r) - u_d,i (L_i W_i) x_i + u_d,i (L_i W_i) x_b + S_i
```

is exactly `dx/dt = M x + N x_b + sources / capacity` (section 6 above), with `capacity =
L W H`, `carrier = 1` (the transported quantity IS the concentration, not a scaled energy
or momentum density as in section 7's heat layer), the routed inter-street terms `Q_r
x_up(r)` playing the advective role, and the roof exchange `u_d (L W)(x_b - x_i)` playing
the same role a conduction edge plays for heat. So the street network is not a third
physics either: it is a third `TransportLayer`, over a graph whose edges are `route`
(street-to-street, through an eliminated junction), `vent` (a street's own flux leaving or
entering at a junction end, before routing) and `exchange` (roof exchange with the
atmosphere boundary node), with `carrier = 1`.

**Why the intersections are ELIMINATED rather than modelled.** A junction with its own
storage would need a potential (a pressure-like quantity) to drive flow between the streets
meeting there, but the only physical constraint at a street intersection is that the
*flows* balance -- there is no independent state for a junction to hold, and the
prescribed-flow edges (`route`, `vent`) that would carry that balance are FIXED-FLOW, not
potential-driven, edges. A potential layer built entirely from fixed-flow edges has no
element relating a node's potential to anything, so its would-be conductance matrix is
identically zero -- singular by construction (framework spec section 9). Eliminating the
junction sidesteps that non-problem entirely: what a junction actually does is redistribute
the fixed street-end fluxes among each other and the atmosphere, and that redistribution is
computed once per solve, outside any potential layer, as a routed flux (`routing_matrix`,
the non-crossing fill below) plus a roof closure (`node_closure`, MUNICH's roof-exchange
correction, which lands asymmetrically on the side needed to keep every one of a junction's
in/out totals matched, per `StreetNetworkTransport.cxx:2980-3014`). The result is written
straight onto the `route`/`vent`/`exchange` edges of the `TransportLayer` as this step's
flow, exactly like a `Drive`, before that layer's own solve runs at all.

**Why every `route`, `vent` and `exchange` flow is non-negative, with direction living in
the topology.** Each of these three edge kinds is added to the graph in BOTH directions
(`direction="in"`/`"out"`, or the ordered pair of distinct street ends at a junction), so a
physical flow that reverses under a wind-direction change does not need its own sign: it is
a different, non-negative amount on the OTHER edge of the pair, exactly as
`assert_forward_oriented` already requires of every fixed-flow edge in the framework. This
reproduces, for free, MUNICH's one-way dead ends (spec section 4.5b): a street whose one
open end is unambiguously downwind gets zero flux on the edge that would carry flow the
other way, not a negative one, because that edge's non-negative closure output is zero
there -- no special-cased dead-end branch is needed anywhere in `StreetFlows`.

**The two canyon-wind closures and the two exchange closures.** `canyon_velocity` (Soulhac,
Perkins and Salizzoni 2008; Soulhac et al. 2011, the SIRANE/IMPAQ form) and the exponential
in-canyon profile (Kim et al. 2018, 2022; the MUNICH form) both turn a reference wind into
an along-street canyon velocity, differing in their treatment of the boundary-layer profile
above the canyon and in their von Karman constant (`KAPPA_IMPAQ = 0.4` against
`KAPPA_MUNICH = 0.41`). `exchange_velocity` similarly has a SIRANE branch (`u_d =
sigma_w/(sqrt(2) pi)`, S11 Eq. (5), K18 Eq. (3), K22 Eq. (B10), `StreetNetworkTransport.cxx
:3273` -- see the issue-C retraction in the README) and a Schulte mixing-length branch
(`SCHULTE_BETA = 2/(sqrt(2) pi)`, fixed by matching the SIRANE form at `a_r = 1`, K18 p.
613). Both pairs are selectable independently (`canyon_wind=`, `exchange=`) because the two
source codebases do not always pair them the same way, and the parity tests exercise both
pairings.

**The non-crossing routing as the north-west-corner rule.** MUNICH's `ComputeAlpha` walks a
junction's inflows counter-clockwise and its outflows clockwise, greedily filling each
inflow/outflow cell with `min(remaining_in, remaining_out)` -- which is exactly the
transportation-problem north-west-corner rule applied to the two ORDERED marginals, and so
has the closed form `F[p, r] = max(0, min(A_p, B_r) - max(A_{p-1}, B_{r-1}))` on their
cumulative sums `A`, `B` (`routing_matrix`, spec section 4.4). Writing it this way rather
than as MUNICH's literal loop makes the fill BATCHED (every junction, every wind direction,
every forcing step in one call) and DIFFERENTIABLE (piecewise linear in the marginals, so
`torch.autograd` sees a gradient through the fluxes), while the one genuinely
combinatorial part -- WHICH order the inflows and outflows are visited in -- is confined to
`order_slots`'s argsort and run under `no_grad`: the ordering is a discrete choice (it can
only change at a measure-zero set of exactly-tied angles), so no gradient through it is
either needed or well-defined, and keeping it out of the traced graph is what lets the
fluxes it feeds stay differentiable.

**Verification.** `tests/verification/test_munich.py` pins the thirteen exact formula pairs
against MUNICH's own source (each at the precision its source publishes) and the idealised
12-street case against Kim et al. (2022) Fig. 1; `tests/verification/test_street_parity.py`
checks the ported IMPAQ oracle against both a four-node hand network and the real
`leiden_small` domain. Conservation, junction elimination against hand algebra, and
gradients through the whole model against central differences are in
`tests/apps/street/test_conservation.py`.

## Summary of flags/uncertainties

- Pseudo-bond-graph `(T, Φ)` vs true bond-graph `(T, dS/dt)`: only the latter is
  literally power-conjugate; check which convention any energy-balance check assumes.
- No published "differentiable EPANET"/"differentiable CONTAM" was found; the closest
  precedent is differentiable AC/DC power flow.
- `torch-sla` and similar differentiable-sparse-linear-algebra packages are recent
  (2026) and unverified for production use here.
- ISO 23247 is a manufacturing standard; its fit to buildings is an analogy, not a
  citation of direct applicability.
- FMI-vs-native-port coupling recommendation in §5 is a design opinion, not a result
  drawn from existing published systems.

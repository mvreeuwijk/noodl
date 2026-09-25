# Theoretical background for `noodl`

A general-purpose, differentiable network solver for conservative transport — pressures
and buoyancy in multizone buildings, heat, contaminant species, sewer flow, and urban
street-canyon exchange — needs one topology layer and one small set of constitutive
patterns that specialise per physical domain. This page gives the graph-theoretic and
physical foundations, compares existing multi-physics network solvers, and covers the
differentiable-programming and co-simulation questions that follow from building this in
PyTorch. Sections 7 to 10 describe how each application maps onto the framework; the
[application pages](applications/index.md) give the details and the verification.

## 1. Graph fundamentals: incidence, cycle, cutset, and Tellegen's theorem

Represent the network as a directed graph with $n$ nodes and $b$ branches. The
**incidence matrix** $A$ ($n \times b$) has $A_{ie} = +1$ if branch $e$ leaves node $i$, $-1$
if it enters, $0$ otherwise (this is exactly `Network.incidence()` in
`src/noodl/topology.py`). For a connected graph $A$ has rank $n - 1$; one row is
redundant because every column sums to zero (each branch leaves one node and enters
another).

**Kirchhoff's current law (KCL)** is conservation at every node:

$$
A f = s
$$

where $f \in \mathbb{R}^b$ are branch flows and $s \in \mathbb{R}^n$ are external sources/sinks (zero at
interior nodes). **Kirchhoff's voltage law (KVL)** says branch potential differences are
consistent with a single-valued nodal potential $\phi \in \mathbb{R}^n$:

$$
e = A^\top \phi \qquad \text{(branch effort = "target minus source", i.e. gradient)}
$$

A **fundamental cycle matrix** $B$ ($l \times b$, $l = b - n + c$ for $c$ connected
components) is built from a spanning forest: each non-tree ("chord") branch closes a
cycle through the tree, giving one row of ±1/0 entries (`Network.cycle_basis()`
implements this directly, walking the tree path in `networkx`). Rows of $B$ span the
**cycle space**, the null space of $A$: $A B^\top = 0$. Dually, a **fundamental cutset
matrix** $Q$ — one row per tree branch, listing the tree branch plus the chords whose
fundamental cycle uses it — spans the **cutset space**, and $A$ itself is a (redundant)
generator of that same space, since removing all branches crossing a node cut is a
special cutset. The cycle space and cutset space are orthogonal complements of $\mathbb{R}^b$, and
$A B^\top = 0$ is the algebraic expression of that fact ([cutset/cycle-space duality,
Wikipedia](https://en.wikipedia.org/wiki/Tellegen%27s_theorem)).

**Tellegen's theorem** (Tellegen 1952; see also Penfield, Spence & Duinker,
*Tellegen's Theorem and Electrical Networks*, MIT Press 1970): for *any* $f$ satisfying
$A f = 0$ and *any* $e$ satisfying $e = A^\top \phi$ for some $\phi$ — on the same graph, but not
necessarily from the same physical system or even the same instant in time —

$$
e^\top f = \phi^\top (A f) = 0.
$$

The proof is one line, but the content is large: $f$ need only be any element of the
cycle space (no constitutive law required), and $e$ need only be a gradient.
Consequences used throughout `noodl`:

- **Power balance / energy conservation**: if $e, f$ are the *same* system's effort and
  flow, $\sum_k e_k f_k = 0$ is the statement that stored power equals dissipated/supplied
  power (`Network.power_residual` is exactly this residual, and is a very cheap
  correctness check on any solver state — it should vanish to numerical precision
  regardless of the constitutive laws used).
- **Adjoint sensitivities / reciprocity**: if $e$ is the *adjoint* solution of a network
  with the same topology but sources placed at the point where a sensitivity is wanted,
  Tellegen's theorem gives the sensitivity of any output to any parameter as a single
  extra solve — this is precisely the classical circuit-theory reciprocity theorem, and
  it is the physical reading of the implicit-function-theorem adjoint used for autograd
  in §4.
- **Cross-checks between different times/systems**: because $f$ and $e$ need not come
  from the same evaluation, Tellegen's theorem underlies mixed potential/Brayton–Moser
  formulations and time-domain reciprocity identities.

### Two dual formulations

**Nodal analysis.** Unknowns are the $n - 1$ independent node potentials $\phi$. Given a
branch constitutive law $f = g(e) = g(A^\top \phi)$, KCL becomes

$$
A\, g(A^\top \phi) = s.
$$

For a linear conductance law $f = G A^\top \phi$ ($G = \operatorname{diag}$ of conductances) this is

$$
(A G A^\top)\, \phi = s, \qquad L := A G A^\top
$$

a weighted graph **Laplacian**, symmetric positive semi-definite (PD after grounding one
node) — the sparse SPD system familiar from resistive-network and finite-volume
diffusion solves, amenable to Cholesky or CG. Nonlinear $g$ (turbulent orifices,
power-law cracks, quadratic pipe friction) is solved by Newton's method with Jacobian
$A\, G'(A^\top \phi)\, A^\top$ — the same congruence structure at every iteration, i.e. a **Schur
complement** of the branch Jacobian onto the nodal potentials (`solvers/newton.py`'s own
implementation is a damped, batched iteration whose relaxation factor `omega` per instance
switches to a full step once the residual has shrunk enough, and falls back to the damped
step for any instance whose residual did NOT shrink under a full step — a dead-end
square-root-law edge otherwise cycles $\Delta p \to -\Delta p$ forever). Nodal analysis handles
flow sources trivially (add to $s$) but a branch whose *natural* law fixes a flow or
potential directly (an ideal pump, a fixed-flow fan, a prescribed pressure boundary)
cannot be written as $f = g(e)$; this is handled by **Modified Nodal Analysis (MNA)**,
which promotes that branch's flow to an extra unknown and adds one extra row enforcing
its constitutive constraint, at the cost of a larger, indefinite (saddle-point) system
([MNA overview](https://lpsa.swarthmore.edu/Systems/Electrical/mna/MNA1.html)).

**Loop (mesh) analysis.** Unknowns are the $l = b - n + c$ cycle amplitudes $\lambda$. Set

$$
f = B^\top \lambda + f_p, \qquad A f_p = s
$$

which satisfies KCL *by construction*, for any $\lambda$ (`cycles.branch_flows` builds
`f = λ @ B` this way, so mass conservation is exact to floating-point precision
independent of solver convergence). What remains is KVL on the branch effort law
$e = h(f)$:

$$
B\, h(B^\top \lambda + f_p) = 0.
$$

Newton's method gives Jacobian $B H'(f) B^\top$, the loop-impedance analogue of the nodal
Laplacian, again a Schur complement ($H' = dh/df$). Loop analysis needs only $l$
unknowns rather than $n - 1$, favourable for tree-like networks with a handful of loops
(most building airflow networks), and flow sources become part of the particular
solution rather than extra unknowns; a branch with a *prescribed potential drop* needs
the dual augmentation instead. Neither formulation is unconditionally better-conditioned
— both reduce to a congruence transform of the same branch Jacobian — so the practical
choice is driven by $n$ vs $l$ and by which sources are naturally flow- or
potential-type. `noodl`'s potential layers are nodal: Newton on the nodal residual above.
The cycle space is kept for flows that are not potential-driven — prescribed or
closure-computed flows, and the closed-form continuity solve on a tree
(`cycles.particular_flow`, §9). `cycles.assert_forward_oriented` encodes a related
constraint: only an entrywise non-negative cycle basis guarantees non-negative $\lambda$
maps to non-negative $f$, needed for the constant upwind operator of §2.

## 2. Port-Hamiltonian systems, bond graphs, and advected quantities

Bond graphs (Paynter 1961; Breedveld) generalise circuit theory to any energy domain via
conjugate **effort/flow** pairs whose product is power:

| Domain | Effort $e$ | Flow $f$ |
|---|---|---|
| Electrical | voltage $V$ | current $I$ |
| Hydraulic/pneumatic | pressure $p$ | volumetric flow $Q$ |
| Thermal (true bond graph) | temperature $T$ | entropy flow $dS/dt$ |
| Thermal (pseudo bond graph, common engineering use) | temperature $T$ | heat flow $\Phi = dQ/dt$ |
| Chemical | chemical potential $\mu$ | molar flow $\dot n$ |
| Species (pseudo bond graph) | concentration $c$ | mass flow $\dot m$ |

(For the true bond graph pair $T \cdot dS/dt$ has units of power; the pseudo-bond-graph
$T \cdot \Phi$ pairing is a convenient engineering approximation, not literally power — this
distinction matters if `noodl`'s thermal module is ever audited for exact energy
balance, and should be flagged as a modelling choice, not swept under Tellegen's
theorem.) Van der Schaft & Maschke, ["Port-Hamiltonian Systems on
Graphs"](https://pure.rug.nl/ws/files/2340945/2013SIAMJContOptimvdSchaft.pdf) (SIAM J.
Control Optim. 51:906–937, 2013; [arXiv:1107.2006](https://arxiv.org/pdf/1107.2006))
give the general framework: the incidence matrix defines a **Dirac structure** relating
edge flows/efforts to vertex flows/efforts, $f_v = -A f_e$, $e_e = A^\top e_v$, so that
storage, dissipation and ports can be attached at either edges or vertices while power
balance ($dH/dt$ = power in through ports − dissipation) holds identically — the
continuous-time, energy-storing generalisation of Tellegen's theorem, and the natural
home for `noodl`'s "storage at nodes, typed edges" design.

**The subtlety for heat and species.** A resistive bond-graph element has $f = g(e)$: the
flow is driven by the effort *difference alone* (Fourier conduction, $q = k(T_i - T_j)$).
But heat and species carried by an airflow are **advected**: the branch flux is
$\dot m_e \cdot c_{\text{upstream}(e)}$, the *product* of a flow variable that itself solves a different
(air-pressure) network with the scalar value *at whichever node is upstream* for the
current flow direction. This is not a function of a potential difference at all — it is
a directed, flow-weighted selection. Represent it with the **upwind operator**
$U(f) \in \mathbb{R}^{b \times n}$, $U_{ei} = 1$ iff node $i$ is upstream of branch $e$ given the sign of
$f_e$ (`Network.upwind`), and build the **advection matrix**

$$
M(f) = A \operatorname{diag}(f)\, U(f) \qquad (n \times n)
$$

so the node balance for a scalar $c$ (temperature, concentration) with nodal storage
$V\, dc/dt = -M(f)\, c + s$ reproduces exactly `layers/transport.py`. Because $f \ge 0$ on a
forward-oriented graph, $-M(f)$ is a (singular) **M-matrix** — off-diagonal entries
non-negative, columns summing to $\le 0$ — precisely the generator structure of a
continuous-time Markov chain / compartmental system, which guarantees the flow
**preserves positivity** of $c$ for any time step (see
[Wikipedia: M-matrix](https://en.wikipedia.org/wiki/M-matrix) and the positive-systems
literature on Metzler matrices). Genuine conduction/diffusion (envelope U-values, pipe
wall conduction, turbulent diffusivity between street-canyon cells) adds a *symmetric*
Laplacian term $A K A^\top$ (§1) on top, giving the standard convection–diffusion operator
on a graph, $V\, dc/dt = -(A K A^\top)\, c - M(f)\, c + s$.

## 3. Existing multi-physics network solvers

| Solver | Formulation | Nonlinearity | Thermal/species coupling |
|---|---|---|---|
| NIST **CONTAM**/AIRNET (Walton 1989) | Nodal (zone pressures) | Newton on power-law/quadratic orifice flows | Contaminant transport: a second nodal solve on well-mixed zones, using AIRNET's converged flows |
| **COMIS** | Nodal, same family | Newton | Similar; largely superseded, merged into EnergyPlus AFN |
| **EnergyPlus AirflowNetwork** | Nodal (zones = nodes) | Newton on Bernoulli/power-law components ([Engineering Reference](https://bigladdersoftware.com/epx/docs/9-4/engineering-reference/airflownetwork-model.html)) | Airflow solved first each timestep; resulting flows feed the zone heat balance (sequential/operator-split) |
| **Modelica Buildings** (Wetter et al.) | Acausal DAE; **stream connectors** carry `p, m_flow, h_outflow, Xi_outflow, C_outflow` together ([spec](https://specification.modelica.org/master/stream-connectors.html)) | Tool-dependent Newton after index reduction | Heat and species ride along with mass flow *by connector design* — closest existing analogue to `noodl`'s intent |
| **pandapipes**/**pandapower** | Nodal Newton (Y-bus-like for power; pressure/temperature nodes for pipes) | Newton, linearised pipe friction each step | Sequential heat-network extension in pandapipes |
| **EPANET** (Rossman) | Nodal heads via the **Global Gradient Algorithm** (Todini & Pilati 1988) | One sparse linear solve for heads + a scalar flow-update per link, each Newton iteration ([EPANET 2.2 docs](https://epanet22.readthedocs.io/en/latest/12_analysis_algorithms.html)) | Water-quality solved as a second, decoupled transport step on converged flows |
| **WNTR** | Python wrapper around EPANET's engine | as EPANET | Adds resilience/failure scenario analysis |
| CityLearn-type digital twins | Wraps EnergyPlus/others for RL | n/a (black box) | n/a |

None of these is differentiable in the autograd sense (uncertain: no published
"differentiable EPANET/CONTAM" was found; the closest is differentiable AC/DC
[power-flow optimisation](https://arxiv.org/pdf/2603.28203)). Two patterns matter for
`noodl`: every mature tool is **nodal** in the pressure/head variable (cheap, robust
SPD systems), and `noodl`'s potential layers are nodal too (§1); and Modelica's stream
connectors are the strongest existing precedent for carrying enthalpy
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

Backpropagating through an iterative Newton solve $F(x, \theta) = 0 \to x^*(\theta)$ need not
differentiate the iterations themselves. By the **implicit function theorem**,

$$
\frac{dx^*}{d\theta} = -\left(\frac{\partial F}{\partial x}\right)^{-1} \frac{\partial F}{\partial \theta},
$$

so the backward pass is one linear solve with the (transposed) converged Jacobian —
$O(1)$ extra memory regardless of iteration count, and (§1) *exactly* the classical
adjoint-network / reciprocity computation Tellegen's theorem licenses: run the network
once forward, once "backward" with sources moved to where the gradient is wanted, and
read sensitivities off a single extra solve. `noodl` applies this rule itself
(`noodl.solvers.implicit`): the backward pass of a potential solve, and of a coupled fixed
point, is one adjoint linear solve at the converged state, rather than autograd traced
through the Newton iterations. Related
libraries: `torchdiffeq` (adjoint-sensitivity Neural ODEs, Chen et al. 2018); on the JAX
side, `jaxopt`'s implicit-differentiation layer (Blondel et al., ["Efficient and Modular
Implicit Differentiation"](https://arxiv.org/pdf/2105.15183), NeurIPS 2021) and
Optimistix/Diffrax; Theseus (Meta) and OptNet (Amos & Kolter 2017) are differentiable
nonlinear/QP solver layers in the same family; Deep Equilibrium Models (Bai et al. 2019)
popularised implicit differentiation for fixed-point layers generally. PyTorch's native
sparse-tensor autograd coverage is limited, which is a further reason to differentiate a
sparse solve through an *analytic* adjoint rather than by tracing the factorisation;
recent work (`torch-sla`, [arXiv:2601.13994](https://arxiv.org/html/2601.13994v2))
targets the same gap.

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

For a PyTorch solver whose point is end-to-end autograd, opaque FMI co-simulation is a
poor fit for internal coupling: gradients cannot cross an FMU boundary. `noodl` instead
couples models through named values — directly analogous to Modelica's stream connectors
— iterated to a fixed point by Gauss-Seidel substitution, and differentiates that fixed
point through the implicit function theorem (§4; see [Coupling](applications/coupling.md)).
FMI (most naturally CS, since ME would require exposing PyTorch's internal state
derivatives through a C API) remains the natural boundary adapter to non-differentiable
tools such as EnergyPlus, where gradients across the boundary are unavailable or must come
from a surrogate or finite differences; `noodl` does not implement it.

On topology and metadata standards for auto-generating the graph: **Brick schema**
(Balaji et al., BuildSys 2016) and the W3C **Building Topology Ontology (BOT)** are
complementary — BOT for spatial containment/adjacency (sites, storeys, spaces, elements),
Brick for equipment/point semantics — and both can plausibly seed a `noodl.Network`'s
nodes and edge `kind`s from a BAS point list or **IFC/BIM** model (`IfcSpace` adjacency
for airflow-path topology, `IfcDistributionElement` for duct/pipe branches). **Project
Haystack** is a competing/overlapping industry tagging convention for the same purpose.
**ISO 23247** ("digital twin framework for manufacturing") is explicitly a
manufacturing-domain standard; any building-digital-twin use is by analogy, not direct
applicability, and no building-specific standard of comparable maturity is known.

## 6. Time integration for stiff advection-reaction on networks

For a linear, time-invariant step $dc/dt = A_c c + b$ (flows frozen over the step — valid
whenever the pressure/flow network is quasi-steady relative to thermal/species
timescales, the standard multizone assumption also made by CONTAM and EnergyPlus AFN),
the **exact** solution is $c(t+\Delta t) = e^{A_c \Delta t} c(t) + A_c^{-1}(e^{A_c \Delta t} - I)\, b$. When
$A_c$ is singular (e.g. a node with zero net flow), Van Loan's **augmented-matrix trick**
(["Computing Integrals Involving the Matrix
Exponential"](https://www.olemartin.no/artikler/vanloan.pdf), IEEE TAC 23:395–404, 1978)
avoids inverting $A_c$ by exponentiating the block matrix

$$
Z = \begin{pmatrix} A_c \Delta t & b\, \Delta t \\ 0 & 0 \end{pmatrix}, \qquad
e^Z = \begin{pmatrix} e^{A_c \Delta t} & \Phi \\ 0 & I \end{pmatrix}
$$

with $\Phi$ the exact integrated forcing term. `layers/transport.py`'s `exact` scheme
applies this augmented exponential as a matrix-exponential action (a Taylor series on the
augmented system), never forming a dense exponential. Because this step is *exact* for the
linear operator, it inherits whatever positivity/conservation structure $A_c$ has: since
$A_c$ is (negative) Metzler/M-matrix (§2), $e^{A_c \Delta t}$ is entrywise non-negative for
*any* $\Delta t > 0$ — unconditional positivity, with no CFL-type step restriction, which is
the chief argument for exponential integration over explicit upwind finite-volume
stepping (Hochbruck & Ostermann, ["Exponential
Integrators"](https://na.math.kit.edu/download/papers/acta-final.pdf), Acta Numerica
19:209–286, 2010). **Implicit Euler**, $(I - A_c \Delta t)\, c^{n+1} = c^n + b\, \Delta t$, is
unconditionally stable and (since $(I - A_c \Delta t)^{-1}$ is the inverse of an M-matrix, hence
non-negative) also positivity-preserving, but only first-order accurate, so it
systematically over-diffuses compared with the exact exponential step. **Crank-Nicolson**,
$(I - A_c \Delta t/2)\, c^{n+1} = (I + A_c \Delta t/2)\, c^n + b\, \Delta t$, is second-order but is *not*
unconditionally positivity-preserving — for $\Delta t$ large relative to the fastest mode of
$A_c$ it can produce spurious oscillation/negative concentrations, a well-known defect
inherited from its use on diffusion equations. For the coupled system, **operator
splitting** is the natural fit: solve the (generally nonlinear, algebraic) pressure/flow
network first on the slow control step (§1, Newton on the loop or nodal system), freeze
the resulting $f$, then advance heat/species exactly (or via implicit Euler for stiff
reaction terms) over that step — a Lie/Strang splitting between the "instantaneous"
flow-equilibration timescale and the finite thermal/species timescale, matching the
quasi-steady-state assumption already built into CONTAM, COMIS and EnergyPlus AFN.

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

**Why a drive is a function of the drivers alone.** The airflow solve is a nodal problem
$A f(A^\top \phi + d) = s$ whose Jacobian is $A \operatorname{diag}(f') A^\top$. It is the branch head $d$ (the
stack and wind terms) NOT depending on $\phi$ that buys this form, and with it the symmetry;
positive definiteness then follows from the element slopes $f' > 0$ and from the network
being grounded (a boundary node, so $A^\top$ has trivial null space), which are separate
requirements the layer checks separately. Making $d$ a function of the thermal STATE would
not disturb any of that -- $\phi$ is still the only unknown of the solve -- but making it a function of $\phi$ or of $f$ would, and
a drive that closed over the flow it produces (say, an upstream density evaluated from the
flow direction) is exactly such a term. The framework therefore defines a `Drive` as a
function of the DRIVERS only: the node densities entering a stack head are computed from the
temperatures held fixed for the duration of the pass, and any flow-direction dependence lives
inside the ELEMENT, where it enters $f'$ and keeps the Jacobian symmetric (this is what
`UpstreamDensityPowerLaw` does, and why a closure-based reference-density correction could
not work: a closure runs before the solve and cannot see the flow direction). CONTAM makes the same choice -- its air densities are held at the values from the
previous thermal update within an airflow solve -- so matching CONTAM's numbers is partly a
consequence of matching this convention, not only of matching its element laws.

**Ping-pong and onion (Hensen 1995; Dols and Emmerich 2016).** Since drives are frozen for
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
for the transient and for the stack. See [Building physics](applications/building_physics.md#verification).

## 8. Street canyons as a transport layer

**The street balance IS the framework's transport equation, again.** For a canyon street
$i$ of length $L_i$, width $W_i$ and height $H_i$, holding a well-mixed pollutant mass
concentration $x_i$ (kg/m³),

$$
L_i W_i H_i \frac{dx_i}{dt} = \sum_r Q_r\, x_{\text{up}(r)} - u_{d,i} (L_i W_i)\, x_i + u_{d,i} (L_i W_i)\, x_b + S_i
$$

is exactly $dx/dt = M x + N x_b + \text{sources}/\text{capacity}$ (section 6 above), with
`capacity` $= L W H$, `carrier` $= 1$ (the transported quantity IS the concentration, not a scaled energy
or momentum density as in section 7's heat layer), the routed inter-street terms
$Q_r\, x_{\text{up}(r)}$ playing the advective role, and the roof exchange $u_d (L W)(x_b - x_i)$ playing
the same role a conduction edge plays for heat. So the street network is not a third
physics either: it is a third `TransportLayer`, over a graph whose edges are `route`
(street-to-street, through an eliminated junction), `vent` (ONLY the junction's closure
share -- the `to_atmosphere`/`from_atmosphere` imbalance `node_closure` computes for that
junction end; a street's own flux rides the `route` edges) and `exchange` (roof exchange
with the atmosphere boundary node), with `carrier` $= 1$. The `exchange` pair is written as
two ADVECTIVE edges rather than as the framework's conduction kind because a conductance is
fixed at construction, whereas $u_d$ varies with the forcing at every step; the two agree
to 1.2e-16, pinned by `test_the_exchange_edge_pair_is_exactly_the_conduction_term`.

**Why the intersections are ELIMINATED rather than modelled.** A junction with its own
storage would need a potential (a pressure-like quantity) to drive flow between the streets
meeting there, but the only physical constraint at a street intersection is that the
*flows* balance -- there is no independent state for a junction to hold, and the
prescribed-flow edges (`route`, `vent`) that would carry that balance are FIXED-FLOW, not
potential-driven, edges. A potential layer built entirely from fixed-flow edges has no
element relating a node's potential to anything, so its would-be conductance matrix is
identically zero -- singular by construction. Eliminating the
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
reproduces, for free, MUNICH's one-way dead ends: a street whose one
open end is unambiguously downwind gets zero flux on the edge that would carry flow the
other way, not a negative one, because that edge's non-negative closure output is zero
there -- no special-cased dead-end branch is needed anywhere in `StreetFlows`.

**The two canyon-wind closures and the two exchange closures.** `canyon_velocity` (Soulhac,
Perkins and Salizzoni 2008; Soulhac et al. 2011, the SIRANE form) and the exponential
in-canyon profile (Kim et al. 2018, 2022; the MUNICH form) both turn a reference wind into
an along-street canyon velocity, differing in their treatment of the boundary-layer profile
above the canyon and in their von Karman constant (0.40 against MUNICH's 0.41). `exchange_velocity` similarly has a SIRANE branch
($u_d = \sigma_w/(\sqrt{2}\,\pi)$, S11 Eq. (5), K18 Eq. (3), K22 Eq. (B10), `StreetNetworkTransport.cxx
:3273`; see the street page's [note on this constant](applications/street_aq.md#limitations-and-caveats))
and a Schulte mixing-length branch
(`SCHULTE_BETA` $= 2/(\sqrt{2}\,\pi)$, fixed by matching the SIRANE form at $a_r = 1$, K18 p.
613). Both pairs are selectable independently (`canyon_wind=`, `exchange=`) because the two
source codebases do not always pair them the same way, and the parity tests exercise both
pairings.

**The Soulhac form, written out.** With $d_i = \min(W/2, H)$ and the in-canyon roughness
$z_{0,b}$, the shape parameter $c$ is the root of

$$
\frac{1}{2} \frac{z_{0,b}}{d_i}\, c = \exp\left(\frac{\pi}{2} \frac{Y_1(c)}{J_1(c)} - \gamma_E\right)
$$

(`soulhac_residual`, solved by `solve_monotone` on $[10^{-4}, 3]$ rather than by MUNICH's
0.01-wide brute-force grid), and the roof-level wind that the in-canyon integral is written
on is

$$
u_h = u_* \sqrt{\frac{\pi}{\sqrt{2}\, \kappa^2 c} \left[Y_0(c) - J_0(c) \frac{Y_1(c)}{J_1(c)}\right]}
$$

The four Bessel functions are the reason `apps/street_aq/canyon.py` wraps
`torch.special.bessel_j0/j1/y0/y1` in `torch.autograd.Function`s: the torch kernels carry
no `grad_fn` at all, and `solve_monotone` differentiates its own residual.

**MUNICH's direction averaging.** The wind direction is not a single number: MUNICH spreads
it with $\sigma_\theta = \min(\sigma_v/U,\ 10^\circ)$ and takes $n_\theta = \lfloor \sigma_\theta \text{ in degrees} \rfloor$
samples, clamped to $[1, 10]$, uniformly on $\pm 2 \sigma_\theta$ with UNNORMALISED weights
(`StreetNetworkTransport.cxx:3567`, `:3568`, `:3562-3616`; `sigma_theta_munich`,
`n_theta_munich`, `direction_offsets`). The canyon velocities are computed ONCE from the
MEAN direction and are NOT recomputed per sample -- only the in/out classification and the
angular ordering at each junction change from sample to sample -- which is what makes the
averaged flux matrix comparable with MUNICH's at all.

**The non-crossing routing as the north-west-corner rule.** MUNICH's `ComputeAlpha` walks a
junction's inflows counter-clockwise and its outflows clockwise, greedily filling each
inflow/outflow cell with $\min(\text{remaining}_{\text{in}}, \text{remaining}_{\text{out}})$ -- which is exactly the
transportation-problem north-west-corner rule applied to the two ORDERED marginals, and so
has the closed form $F_{pr} = \max\bigl(0, \min(A_p, B_r) - \max(A_{p-1}, B_{r-1})\bigr)$ on their
cumulative sums $A$, $B$ (`routing_matrix`). Writing it this way rather
than as MUNICH's literal loop makes the fill BATCHED (every junction, every wind direction,
every forcing step in one call) and DIFFERENTIABLE (piecewise linear in the marginals, so
`torch.autograd` sees a gradient through the fluxes), while the one genuinely
combinatorial part -- WHICH order the inflows and outflows are visited in -- is confined to
`order_slots`'s argsort and run under `no_grad`: the ordering is a discrete choice (it can
only change at a measure-zero set of exactly-tied angles), so no gradient through it is
either needed or well-defined, and keeping it out of the traced graph is what lets the
fluxes it feeds stay differentiable.

**Verification.** Thirteen exact formula pairs are checked against MUNICH's own source
(each at the precision its source publishes), and the idealised 12-street case against Kim
et al. (2022) Fig. 1; conservation, junction elimination against hand algebra, and
gradients through the whole model against central differences are checked as well. See
[Street air quality](applications/street_aq.md#verification).

## 9. Sewers and water distribution

**Why the sewer's water side is CONTINUITY-first and the water-distribution side is a
potential layer, on the same framework.** A gravity sewer runs free-surface: on the
dendritic tree of section 1's own analysis, normal flow at every pipe is set by continuity
alone -- the net inflow upstream of it (`cycles.particular_flow`, closed form on a tree,
no Newton solve at all) -- and depth follows from Manning's law given that flow, not the
other way round; a head difference does not drive the water, so a Newton potential layer
built on head is the wrong shape for it. A pressurised water-distribution
network is the opposite case, and it is exactly the framework's own: every pipe runs full,
and head loss is a monotone function of the head difference at each pipe, pump and valve,
so `noodl.apps.water` is an ordinary `PotentialFlowLayer` (§1, §3 above) with
Hazen-Williams or Darcy-Weisbach pipes, `PumpCurve` and minor-loss elements, solved by the
same batched, damped Newton iteration as every other potential layer in this codebase.
Section 3's own table already says why: SWMM solves the sewer's full 1-D Saint-Venant
equations (dynamic wave routing) with surcharging and reverse flow, "a genuinely dynamic
(not quasi-steady) network" -- exactly why surcharge, backwater and dynamic-wave routing
are OUT of scope here, and SWMM's KINWAVE (kinematic wave) routing, which shares the
continuity-first assumption, is the parity target for rows W1-W4 rather than SWMM's own
dynamic-wave engine.

**The manhole-level storage idealisation, and what it is not.** With `storage=True`,
each manhole gets a surface area and a level advanced by implicit Euler,
$A_s\, dH/dt = \sum \text{inflow} - Q_{\text{out}}(H)$, where $Q_{\text{out}}$ is read off the SAME Manning law as the
quasi-steady case at depth $H$. This is a lumped storage-node approximation -- "the
manhole's level equals its own outgoing pipe's entrance depth" -- not a hydraulic profile
along the pipe and not SWMM's dynamic wave, which carries one head per junction and a
backwater-coupled momentum equation between them; under constant inflow the storage sweep's
fixed point is the quasi-steady solution (measured 5.7e-15 relative on flows, 4.4e-15 on
depths, row W7), which is the only property this idealisation is required to
have. The sweep advances every manhole LEVEL-SYNCHRONOUSLY, leaves to the outfall, in the
same order the tree's own closed-form flow solve uses: it precomputes, once, the leaf-to-
root level order and, per level, flat gather/scatter index tensors, so the per-call sweep is
one batched `solve_monotone` call plus one `index_add` per tree LEVEL rather than a Python
loop over manholes -- the loop count is the tree depth, not the manhole count, satisfying
the rule that no per-step path loops in Python. The pipe carrying each
manhole's outflow is found by which pipe drains INTO that manhole's own outfall, never
assumed to be the last pipe read from the file, because the reader's pipe order is whatever
order the `.inp` lists conduits in, not necessarily the order the tree drains in;
and the headspace `Stack` buoyancy drive's `z_path` on a `headspace` edge is the pipe's
CROWN elevation -- the mean of its two end inverts plus its own diameter -- not the mean
invert alone, which is only the pipe's invert at its midpoint. A forest of more
than one tree is supported on both the water and air sides: each component gets
its own outfall and, on the air side, its own outfall-to-ambient headspace edge, so `air.q`
and `sewer.q` conserve mass within each component independently rather than at one shared
sink.

**The `air.phi` datum.** Every `leak` edge's `Stack` carries a manhole term that cancels
against the manhole's own ground level (`z_ref - z_path = 0` there) but an AMBIENT term that
does not, because `ambient` has no `ground` attribute of its own and so falls back to a
`0.0` datum rather than to the manhole's; the same `0.0` is used identically at the outfall's
open-air edge and at `ambient`'s own fixed boundary condition (`air.phi_boundary = 0`), so
every air pressure in the solve is shifted by the same constant -- `air.phi` is
DATUM-REFERENCED to this convention, not to a physical zero, and every drive, refusal and
verification row reads only DIFFERENCES between manholes, which the convention leaves
unaffected.

**The headspace momentum balance, and the `Drive`-cannot-see-`phi` rule.** Air in the
headspace above the flow is driven by three terms per pipe: Darcy-Weisbach wall
friction on the air itself, buoyancy from the density difference between a warm manhole
shaft and ambient (the SAME `Stack` drive the building application's stack effect uses, one
full-node air-density driver, `rho_air_nodes`), and drag from the moving water surface
dragging the air along above it. That last term is written as a `Drive` --
$D = \tfrac{1}{2} f_i\, \rho_{\text{air}} U_s \lvert U_s \rvert\, T L / A_{\text{air}}$ in the ABSOLUTE water-surface velocity $U_s$, not the
physically more complete relative form $(U_s - V_{\text{air}})$ -- because a `Drive` is a function of
the drivers alone and is never allowed to read the layer's own solved potential $\phi$
(§1, `drives.py`): letting the drag depend on the air's own flow would put a term into the
Newton residual that the Jacobian $A \operatorname{diag}(g') A^\top$ does not account for, silently breaking
the solver's own convergence theory. The absolute form is therefore not a simplification of
convenience but the one shape the framework's `Drive` protocol admits; the relative form is
recorded as a follow-up requiring an element-side formulation (a batched monotone root on
$R \lvert Q \rvert Q - \tfrac{1}{2} f_i\, \rho\, (U_s - Q/A)^2\, T L / A_{\text{air}} = \Delta p$) rather than a drive.

**The two-film coupling, and its lagged pass.** Total sulfide in the water and H₂S in the
headspace air are two separate `TransportLayer` species, coupled by one closure,
`H2STransfer`, computing a flux $J = K_L a\, V_{\text{wet}} (f C_S - C_G/H)$ per manhole from the free-
sulfide fraction $f$, Henry's constant $H(T)$ and a two-film transfer coefficient $K_L a$,
and writing $+J$ onto the air side and $-J$ onto the water side (equal in moles of S by
construction, checked node by node, row C2). The lateral inflow-concentration
drivers, `bod_in`/`sulfide_in`, enter the water-quality layer's source term the same way
every other nodal load does -- $\text{inflow} \times \text{concentration}$ (m³/s times kg/m³, kg/s per species)
-- written by a separate closure, `LateralLoads`, registered before `H2STransfer` so its
load and the two-film transfer's own source term add rather than one overwriting the other.
Both correlations read per-PIPE quantities --
hydraulic radius, slope, wetted velocity, mean depth -- at each manhole's own SINGLE
outgoing pipe (a tree guarantees exactly one), gathered once at construction into a
per-manhole index (`out_pipe`) rather than looked up per call; the same gather
underlies the Pomeroy-Parkhurst sulfide-generation reaction. The closure reads the PREVIOUS
pass's concentrations -- the framework's ordinary lagged coupling (§5, §7 above) -- so
`coupling="pingpong"` carries a one-pass lag on the cross-phase source terms exactly as the
building application's heat/species coupling does, and `coupling="iterate"` removes it by
Hensen's-onion successive substitution when the two quality layers' own `iterate_tol` is set.

**Verification.** The sewer's water flows, depths, velocities and a first-order-decay
tracer are checked against pyswmm/SWMM 5.2.4 under KINWAVE routing (rows W1-W4); the air
side against Pescod and Price's laboratory ratios and Tyneside field range (rows A1-A3) and
the H₂S two-film transfer's Henry-equilibrium fixed point (row H3); the water-distribution
application's heads, flows, pump gain, tank trajectory, pressure-driven demand, loop
consistency, gradients and a trace-quality row against EPANET 2.2 through wntr (rows D1-D8,
G2). $f_i$ and $f_{\text{air}}$ are CALIBRATED to a single source (Pescod and Price Test 8),
not literature-pinned, so rows A1 and A2 are consistency checks against the calibration
source rather than independent validation. See [Sewers](applications/sewer.md) and
[Water distribution](applications/water.md#verification).

## 10. A fourth flow-determination mode: clip-and-allocate capacitated transfer

**Why this is a fourth way of determining flows, not a variant of the potential layer.**
Sections 1 and 9 above both describe networks where flow at an edge is either the closed-
form output of a Newton potential solve or of continuity alone on a tree. WSIMOD's own
`Arc`/`Node` model (Dobson, Liu and Mijic, JOSS 2023; GMD 2024) is neither: an edge is asked
to carry a REQUESTED flow (typically emitted by some upstream process closure, not a
pressure difference) and simply clips it against two independent bounds -- its own arc
capacity and the downstream node's remaining storage headroom -- with no potential variable
anywhere in the calculation. This is the fourth way an edge's flow can be determined (alongside a potential-flow Newton solve, a driver-prescribed
flow, and a closure-computed flow), and `CapacitatedTransferLayer`
(`src/noodl/layers/capacitated.py`) is its implementation: a `Model` layer type that owns
one or more edge kinds exactly like `PotentialFlowLayer` does, but whose per-step output is
an explicit clip-and-allocate computation rather than a solve.

**The clip/allocation math, condensed.** Given the previous
per-node storage $s$, a per-edge capacity $c_{\text{arc}}$ and a per-edge REQUEST driver $r$:
receiver headroom is $h = (s_{\max} - s)/\Delta t$ at each edge's downstream node (a boundary node
has $s_{\max} = \infty$ and never constrains) -- a RATE, in the same units as $r$ and $f$, which is
what makes the clip bound the actual storage INCREASE $\Delta t\, h = s_{\max} - s$ at any $\Delta t$ and
not only at $\Delta t = 1$; the realised flow in hard-clip mode is
$f = \min(r, c_{\text{arc}}, h)$, vectorised over every edge via the incidence matrix in one pass, no
Python loop over nodes or edges. Where more than one edge converges on a node whose combined
tentative demand exceeds its free headroom, each edge's share is instead a preference-
weighted fraction of that headroom, resolved by a small FIXED number of vectorised passes
(`n_passes`, default 5, matching WSIMOD's own `constants.MAXITER`) rather than WSIMOD's own
per-node bounded `while`-with-early-exit -- each pass is strictly non-expansive, so a fixed
count no worse than WSIMOD's own cap is a safe over-approximation of the same fixed point,
not an approximation of a different algorithm. This sharing is deliberately
RECEIVER-LOCAL: it triggers only at the node actually oversubscribed and never propagates a
downstream bottleneck back to an earlier edge in the same step, mirroring WSIMOD's own
per-arc semantics (a node's accept decision is against its own headroom, never its future
ability to forward flow onward) and matching what storage itself is for in a discrete-time
capacitated network. The storage update is $s' = \min(s + A_{\text{in}} f - A_{\text{out}} f,\ s_{\max})$ with
`overflow = relu(...)` reported as a diagnostic, never fed back -- and deliberately has NO
floor at zero: $f$ is already clipped against the RECEIVER's headroom, never the sender's
own available storage, so a caller requesting more than is actually available upstream is
a modelling responsibility upstream of this layer (a closure sizing requests correctly),
exactly as a `PotentialFlowLayer`'s boundary sources are trusted as given rather than
second-guessed.

**Two differentiability modes, and why the sharing site needed a real solve, not just a
softened `min`.** `"smooth"` replaces every `min`/`clamp`/`relu` above with its
softmin/softplus counterpart at a construction-time temperature $\tau$ -- ordinary autograd
through the vectorised passes, no new solver machinery, exactly the same style as the
building application's smoothed switches elsewhere in this codebase. `"projection"` poses
the allocation at an oversubscribed node as a genuine per-node QP -- minimise
$\sum_i \text{pref}_i\, (f_i - \text{remaining}_i)^2$ subject to $0 \le f_i \le \text{avail}_i$ per edge and
$\sum_i f_i \le \text{free headroom}$ jointly -- because the FIRST implementation of this mode
reused hard-clip's own preference-proportional share formula, which depends only on
preference weights and total headroom, never on any individual edge's own request; being
provably independent of a competitor's request, it could not carry a gradient between
competing edges no matter how it was wrapped, defeating the purpose of this mode, which
is that gradients flow through which arc absorbs a constraint. The real QP's
Lagrangian stationarity, with the box bound applied, reduces to
$f_i = \operatorname{clamp}(\text{remaining}_i - \lambda/\text{pref}_i,\ 0,\ \text{avail}_i)$ for one scalar $\lambda$ SHARED by every edge competing at
that node -- so $\partial f_i / \partial r_j$ for a competing edge $j \ne i$ is genuinely nonzero, flowing
entirely through $\lambda$. $\lambda$'s root is monotone in $\lambda$ by construction (every
edge's contribution shrinks as $\lambda$ grows), which is exactly
`noodl.solvers.scalar.solve_monotone`'s own contract (a batched, monotone,
implicit-function-differentiable scalar root) -- the same primitive the sewer application
uses for Manning-depth inversion (section 9 above) -- rather than new solver machinery or a
Fischer-Burmeister-smoothed complementarity condition fed through
`noodl.solvers.implicit.implicit_solve`, which was investigated and rejected as strictly
harder to verify for no accuracy benefit. The resulting cross-gradient,
$\partial f_{BD} / \partial r_{CD} = -0.5$ on this specific SYMMETRIC-preference four-node diamond test
fixture, was hand-derived and is checked directly (sign and magnitude) by
`test_projection_mode_sharing_has_nonzero_cross_gradient`. The general mechanism behind
it is the KKT derivation
$\partial f_i / \partial r_j = -(1/\text{pref}_i) / \sum_k (1/\text{pref}_k)$, of which −0.5 is the
symmetric-weight special case; it was also hand-checked against asymmetric preferences
(`pref=[1,3]` and `pref=[1,2,5]`), where the value differs (e.g. `pref=[1,3]` gives
−0.25).

**Verification, and what it does and does not show.** `tests/verification/
test_wsimod_parity.py` replays WSIMOD's own captured per-arc requests from its packaged
`quickstart_demo` and `oxford_demo` scripts through `CapacitatedTransferLayer` and compares
against WSIMOD's own realised flows -- WSIMOD is the reference implementation here, the same
relationship pyswmm and EPANET have with the sewer and water applications above, and it is
NOT an independent measurement. This comparison has a real, specific limitation: of
`quickstart_demo`'s 6 arcs and `oxford_demo`'s 21 arcs, all but one sit at WSIMOD's own
unbounded capacity for the whole run, and the one finite-capacity arc never sees its request
approach its own capacity either -- so neither demo's numbers ever exercise the branch where
$c_{\text{arc}}$ actually binds, only the identity path $\min(x, c_{\text{arc}}) = x$. The same holds, for a
different reason, of the clip's other bound: both fixtures set $s_{\max} = \infty$ at every node
(the harness captures per-arc capacity only), so the receiver-headroom clip is the identity
everywhere too and the proportional-sharing branch never runs against WSIMOD's numbers. Both
mechanisms are separately and rigorously covered by synthetic unit fixtures
with deliberately tight bounds; what remains unvalidated is specifically WSIMOD's own
numbers at a binding point, not the mechanism (see
[what the WSIMOD parity does not show](applications/capacitated.md#what-the-wsimod-parity-does-not-show)).

## Caveats

- Pseudo-bond-graph $(T, \Phi)$ vs true bond-graph $(T, dS/dt)$: only the latter is
  literally power-conjugate; an energy-balance check must state which convention it
  assumes.
- No published "differentiable EPANET" or "differentiable CONTAM" is known; the closest
  precedent is differentiable AC/DC power flow.
- `torch-sla` and similar differentiable sparse linear-algebra packages are recent (2026).
- ISO 23247 is a manufacturing standard; its fit to buildings is an analogy, not a
  citation of direct applicability.
- §5's preference for native, differentiable coupling over FMI is a design choice, not a
  result drawn from existing published systems.

"""ConstitutiveLayer: the loop formulation for arbitrary branch laws.

``PotentialFlowLayer`` solves one specific shape of problem well -- a nodal conservation
residual driven by potential-controlled branch laws ``q = g(dp)``. Not every branch law has
that shape. A branch law may be FLOW-controlled (``dp = f(q)``, e.g. a prescribed potential
drop as a function of current, not the other way round), may mix the two conventions edge by
edge (a pressure source on one branch, a resistor on another), or may be a DYNAMIC element --
an inductor or capacitor -- whose branch relation is a differential equation stepped
implicitly, coupling ``p`` and ``q`` through a previous-step state rather than through either
alone. None of that fits a nodal solve, because a nodal solve's unknown is ``phi`` and its
residual is written in terms of ``q(dp)`` specifically.

The 2019 ``Circuit`` (``legacy/Tellegen/circuit.py``) handled all of this uniformly by
choosing different unknowns: cycle amplitudes ``m`` and reduced nodal potentials ``phi_red``,
related to ``p`` and ``q`` by ``q = J.T @ m`` (``J`` a basis of the cycle space, so
``A q = 0`` identically -- Kirchhoff's current law) and ``p = A_red.T @ phi_red`` (``A_red``
the incidence matrix with one grounded row dropped, so ``B p = 0`` identically -- Kirchhoff's
voltage law, ``B`` any basis of the cycle space). Both conservation laws hold BY CONSTRUCTION,
for any ``m`` and ``phi_red`` whatsoever, leaving only the branch law itself,
``law(p, q, theta) = 0``, to solve: ``b`` equations in ``l + (n - 1) = b`` unknowns for a
connected graph of ``b`` branches, ``n`` nodes and cycle rank ``l``. This is the ONE
capability of the legacy ``Circuit`` that a purely nodal solver structurally cannot express,
and this layer exists to carry it forward, unchanged in spirit, on noodl's own primitives.

This is a DENSE solve, deliberately: the residual's Jacobian is built by
``torch.func.jacrev`` as a plain ``(b, b)`` tensor at every Newton iterate, with no sparse or
matrix-free path. That is the right trade for the loop formulation's natural habitat -- a
handful of branches carrying a genuinely mixed or dynamic law -- and the wrong one for a
network of any size: ``PotentialFlowLayer`` remains the tool for nodal balances over a large
airflow, thermal or transport network, where the unknown is one potential per node, the
Jacobian is sparse by construction, and the whole apparatus (linear-operator eligibility,
sparse-direct/PCG/GMRES selection, batched instances) is built for that shape.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from noodl.solvers.implicit import implicit_solve
from noodl.topology import Network

Tensor = torch.Tensor

# The cycle-amplitude block of the default initial guess is nudged off exactly zero (see
# `ConstitutiveLayer.__init__`): a law with a q^2-type nonlinearity (the tutorial's quadratic
# loop, `p = a |q| q`) has a zero q-derivative at q = 0, which makes every cycle-amplitude
# column of the residual's Jacobian vanish there -- the Newton step from `z0 = 0` is singular
# before the first iteration even starts. `phi_red`'s block is left at exactly zero: the laws
# this layer is built for (see the module docstring) are linear in `p`, so the corresponding
# Jacobian columns do not have this problem.
_DEFAULT_AMPLITUDE = 1e-2


class ConstitutiveLayer:
    """Solves an arbitrary branch law ``law(p, q, theta) = 0`` over all edges of one `kind`,
    in the loop formulation (see the module docstring for why).

    ``law(p, q, theta) -> Tensor`` takes the branch potential differences ``p`` and flows
    ``q`` (both ``(b,)``, in the network's own edge order for `kind`) and this layer's own
    parameters `theta`, and returns `b` residuals -- one law per branch, in any mix of
    potential- and flow-controlled form. ``solve`` finds `p`, `q` satisfying it, together
    with the two Kirchhoff laws, and is differentiable in `theta` through
    ``solvers.implicit.implicit_solve``.
    """

    def __init__(
        self,
        net: Network,
        name: str,
        *,
        kind: str,
        law: Callable[[Tensor, Tensor, Tensor], Tensor],
    ) -> None:
        self.net = net
        self.name = name
        self.kind = kind
        self.law = law

        try:
            n_components = net.n_components_of(kind)
        except KeyError as exc:
            raise ValueError(
                f"ConstitutiveLayer {name!r}: unknown kind {kind!r}; the network has no "
                f"edges of that kind"
            ) from exc
        if n_components != 1:
            raise ValueError(
                f"ConstitutiveLayer {name!r}: disconnected network under kind {kind!r} "
                f"({n_components} components); the loop formulation needs every node "
                f"reachable from a single ground, so cycle amplitudes and reduced nodal "
                f"potentials together account for every branch"
            )

        # Built once, at construction: neither depends on theta, so nothing here re-derives
        # per solve() call. `J` (l, b) and `A_red` (n - 1, b) are exactly the cycle- and
        # cut-space bases the module docstring describes; dropping node 0's row of the
        # incidence matrix is the "ground" (`net.incidence` orders rows by `net.nodes`,
        # i.e. insertion order, so this is always the first node ever added to the network).
        self.J = net.cycle_basis(kind)
        self.A_red = net.incidence(kind)[1:]
        self.l = self.J.shape[0]
        self.b = self.J.shape[1]
        self.n_red = self.A_red.shape[0]
        # A connected graph always satisfies this (l = b - n + 1); an `assert` alone would
        # both disappear under `python -O` and give no diagnostic, so this is raised
        # explicitly -- it can fire for a node carrying no edge of `kind`.
        if self.l + self.n_red != self.b:
            raise ValueError(
                f"ConstitutiveLayer {name!r}: cycle rank l={self.l} and reduced-node count "
                f"n_red={self.n_red} do not add up to branch count b={self.b}; this signals "
                f"a node carrying no edge of kind {kind!r}"
            )

        self._law_checked = False

    def _split(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return z[..., : self.l], z[..., self.l :]

    def _pq(self, z: Tensor) -> tuple[Tensor, Tensor]:
        m, phi_red = self._split(z)
        q = self.J.T @ m
        p = self.A_red.T @ phi_red
        return p, q

    def _residual(self, z: Tensor, theta: Tensor) -> Tensor:
        p, q = self._pq(z)
        return self.law(p, q, theta)

    def _operator(self, z: Tensor, theta: Tensor) -> Tensor:
        return torch.func.jacrev(self._residual, argnums=0)(z, theta)

    def _default_z0(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        z0 = torch.zeros(self.b, dtype=dtype, device=device)
        z0[: self.l] = _DEFAULT_AMPLITUDE
        return z0

    def solve(
        self,
        theta: Tensor,
        *,
        z0: Tensor | None = None,
        diagnostics: dict | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Solve ``law(p, q, theta) = 0`` subject to both Kirchhoff laws; returns ``(p, q)``.

        ``z0`` is the starting guess for the underlying ``(m, phi_red)`` unknowns (``(b,)``),
        defaulting to zero nodal potentials and a small nonzero cycle amplitude (see
        ``_DEFAULT_AMPLITUDE``). Pass the previous step's own converged ``z`` -- available
        from ``diagnostics`` is not needed for this; the caller already has ``p``, ``q`` --
        as `z0` on the next call for a dynamic branch law stepped implicitly over many steps,
        exactly as the legacy ``Circuit`` reused its stored state between calls.

        `diagnostics`, when a dict is given, receives the forward Newton solve's own
        `newton_iterations`, `linear_iterations`, `backend`, `converged` and `residual_norm`
        (see ``solvers.implicit.implicit_solve``).
        """
        if z0 is None:
            z0 = self._default_z0(theta.dtype, theta.device)

        if not self._law_checked:
            r0 = self._residual(z0, theta)
            if r0.shape[-1] != self.b:
                raise ValueError(
                    f"ConstitutiveLayer {self.name!r}: law returned {r0.shape[-1]} "
                    f"residuals but this network has {self.b} branches of kind "
                    f"{self.kind!r}; law(p, q, theta) must return one residual per branch"
                )
            self._law_checked = True

        z = implicit_solve(
            self._residual,
            self._operator,
            z0,
            (theta,),
            diagnostics=diagnostics,
            where=f"ConstitutiveLayer {self.name!r} solve",
        )
        return self._pq(z)

"""Population-weighted exposure on a street network and the exposure-reduction vector of
Li, Fellini and van Reeuwijk (2023, Atmos. Environ. 292, 119432).

Their eq. 3: e_i = p_i C_i q (q the inhalation rate); eq. 4: e = q p o (A Q); eq. 12: the
reduction R_j from an emission cut delta_Q in street j is delta_Q * sum_i E_ij, the
out-degree of node j in the exposure network. They build A by one dispersion run per street
per wind direction. Here R is ONE backward pass of the total exposure through the model:
R_j = delta_Q * d e_tot / d Q_j. The derivative itself is exact for the model as solved,
whatever its nonlinearity; R is its first-order (linearised) extrapolation over the finite
cut delta_Q, and is identical to the finite cut e_tot(Q) - e_tot(Q - delta_Q e_j) -- what
`exposure_reduction_forward` below computes directly -- ONLY for a passive (linear) tracer,
where the model's response to a source is linear in the source (`tests/test_exposure.py`,
"passive scalar: linear in Q").

Single instance, single species: every function here takes an UNBATCHED `"street.sources"`
(shape `(n,)`) and an unbatched `"street.x"` (shape `(n_i,)`); a batch axis or more than one
species is rejected loudly rather than silently misaddressed.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from noodl.apps.street.network import StreetNetwork, street_index

Q_INHALATION = 0.571 / 3600.0   # m3/s per person: 0.571 m3/h (Li et al. 2023, section 2.3)


def street_population(net: StreetNetwork, *, n_total: float) -> torch.Tensor:
    """p_i = n_total * H_i L_i / sum_j H_j L_j (their eq. 6 with a common building depth
    and storey height, which cancel)."""
    hl = torch.tensor([s.height * s.length for s in net.streets], dtype=torch.float64)
    return float(n_total) * hl / hl.sum()


def total_exposure(concentration: torch.Tensor, population: torch.Tensor,
                    *, q: float = Q_INHALATION) -> torch.Tensor:
    return (q * population * concentration).sum()


def _steady_exposure(model, state, drivers, population, streets, sources) -> torch.Tensor:
    d = dict(drivers)
    d["street.sources"] = sources
    out = model.steady(state, d)
    idx = street_index(model)
    rows = torch.tensor([idx[name] for name in streets], dtype=torch.long)
    x = out["street.x"]
    if x.dim() != 1:
        raise ValueError(
            f"_steady_exposure: 'street.x' has shape {tuple(x.shape)}; these functions "
            f"take a single-instance, single-species state (shape (n_i,)), not a batch or "
            f"multiple species"
        )
    return total_exposure(x[rows], population)


def _emissions(model, drivers, streets):
    base = drivers["street.sources"]
    if base.dim() != 1:
        raise ValueError(
            f"_emissions: 'street.sources' has shape {tuple(base.shape)}; these functions "
            f"take a single-instance, single-species emissions vector (shape (n,)), not a "
            f"batch or multiple species"
        )
    cols = torch.tensor([model.net.node_index(n) for n in streets], dtype=torch.long)
    return cols, base


def exposure_reduction_adjoint(model, state, drivers, *, population, streets: Sequence[str],
                                delta_fraction: float = 0.1):
    cols, base = _emissions(model, drivers, streets)
    q = base[cols].detach().clone().requires_grad_(True)
    sources = base.detach().clone()
    sources = sources.scatter(0, cols, q)          # keeps the graph to q
    e_tot = _steady_exposure(model, state, drivers, population, streets, sources)
    (grad,) = torch.autograd.grad(e_tot, q)
    delta_q = delta_fraction * q.detach().mean()
    return delta_q * grad, {"solves": 1, "backward": 1}


def exposure_reduction_forward(model, state, drivers, *, population, streets: Sequence[str],
                                delta_fraction: float = 0.1):
    cols, base = _emissions(model, drivers, streets)
    with torch.no_grad():
        delta_q = delta_fraction * base[cols].mean()
        e0 = _steady_exposure(model, state, drivers, population, streets, base)
        r = torch.zeros(len(streets), dtype=torch.float64)
        for j, col in enumerate(cols.tolist()):
            cut = base.clone()
            cut[col] = cut[col] - delta_q
            r[j] = e0 - _steady_exposure(model, state, drivers, population, streets, cut)
    return r, {"solves": len(streets) + 1}

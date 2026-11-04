"""SIRANE's plume trajectory in time: the tables of the unsteady above-roof puff model.

`plume_table` resamples the 10 s scan of the plume (`plume._trajectory_nodes`) onto a
distance grid for the steady kernel. A puff reads the same scan by its age: `puff_tables`
returns the nodes `t_j = 10 j` s of `X`, `U`, `sigma_y`, `sigma_z` and the centre height
`z_c`, one table per release class (the centre law `z_c = max(H_R, d + E|N(c - d,
(k sigma_z)^2)|)` takes its own `c`, so an above-roof release at height `z_s` has its own
table), and `PuffTables.at_age` interpolates them linearly in age. The speed is the
scan's per-segment node speed (`X_j = X_{j-1} + U_j dt`), so a puff that advances its
displacement by `U(a) dt` per step follows the table's `X` exactly under constant wind.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from noodl.apps.street_aq.canyon import KAPPA
from noodl.apps.street_aq.closures import SIGMA_V_MIN, SIGMA_W_MIN
from noodl.apps.street_aq.plume import (
    _DTYPE,
    DEFAULT_TEMPERATURE_K,
    DOWNWIND_CUTOFF_M,
    PLUME_CENTRE_K,
    TIME_STEP_S,
    _step_bound,
    _table_inputs,
    _trajectory_nodes,
    table_chunks,
)

_NODE_KEYS = ("x", "u", "sigma_y", "sigma_z", "z_c")


@dataclass(frozen=True)
class PuffTables:
    """The plume trajectory by age (`puff_tables`).

    `x` (m, the distance travelled), `u` (m/s, the speed of the segment ending at the node;
    node 0 holds the speed at `H_R`), `sigma_y`, `sigma_z` (m) and `z_c` (m, the plume
    centre height; node 0 holds node 1's) are `(n_tab, n_b, n_nodes)` at ages `0, step_s,
    2 step_s, ...`; a row that reaches `x_max` before the others repeats its last node.
    `h_canopy` and `h_abl` (m) are `(n_b,)`; `x_max` is the extent the tables were built to
    (the last node of each row is the first beyond it)."""

    x: Tensor
    u: Tensor
    sigma_y: Tensor
    sigma_z: Tensor
    z_c: Tensor
    h_canopy: Tensor
    h_abl: Tensor
    x_max: float
    step_s: float = TIME_STEP_S

    @property
    def t_max(self) -> float:
        """The age of the last node, s."""
        return (self.x.shape[-1] - 1) * self.step_s

    def at_age(self, age: Tensor, table: Tensor, row: Tensor
               ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """`(x, u, sigma_y, sigma_z, z_c)` at `age` (s) from table `table` and
        meteorology row `row` (long tensors that broadcast with `age`; the result has their
        broadcast shape). Everything but `u` is linear in age between nodes; `u` is the
        node speed of the segment containing `age` (not interpolated). An age beyond the
        last node returns the last node's values (a later step retires such a puff); a
        negative age the source node's."""
        _, n_b, n = self.x.shape
        pos = torch.clamp(age / self.step_s, 0.0, float(n - 1))
        floor = torch.floor(pos.detach()).long()
        i0 = torch.clamp(floor, 0, n - 2)
        frac = pos - i0
        base = (table * n_b + row) * n
        lo = base + i0
        seg = base + torch.clamp(floor + 1, 1, n - 1)
        out = []
        for node in (self.x, self.u, self.sigma_y, self.sigma_z, self.z_c):
            v = node.reshape(-1)
            if node is self.u:
                out.append(v[seg])
            else:
                out.append(v[lo] * (1.0 - frac) + v[lo + 1] * frac)
        return out[0], out[1], out[2], out[3], out[4]


class _RecomputedNodes(torch.autograd.Function):
    """One chunk of `_trajectory_nodes` that keeps no graph (the `_RecomputedTrajectory`
    pattern of `plume`): the forward runs without one and the backward rebuilds the chunk's
    graph and differentiates it. First derivatives only; `create_graph=True` raises."""

    @staticmethod
    def forward(ctx, keys, kappa, x_end, *values):
        ctx.keys, ctx.kappa, ctx.x_end = keys, kappa, x_end
        ctx.save_for_backward(*values)
        nodes = _trajectory_nodes(dict(zip(keys, values, strict=True)), None, kappa,
                                  x_end=x_end)
        return tuple(nodes[k] for k in _NODE_KEYS)

    @staticmethod
    def backward(ctx, *grads):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "puff_tables: second-order differentiation (create_graph=True) is not "
                "supported; the trajectory tables have first-order gradients only. Detach "
                "the first-order gradient before using it in a further differentiable loss."
            )
        values = ctx.saved_tensors
        leaves = [v.detach().requires_grad_(ctx.needs_input_grad[3 + i])
                  for i, v in enumerate(values)]
        wanted = [v for v in leaves if v.requires_grad]
        with torch.enable_grad():
            nodes = _trajectory_nodes(dict(zip(ctx.keys, leaves, strict=True)), None,
                                      ctx.kappa, x_end=ctx.x_end)
            found = iter(torch.autograd.grad([nodes[k] for k in _NODE_KEYS], wanted, grads,
                                             allow_unused=True))
        return (None, None, None,
                *(next(found) if v.requires_grad else None for v in leaves))


def _pad_nodes(a: Tensor, n: int) -> Tensor:
    """`a` `(m, n_a)` extended to `n` columns by repeating its last column."""
    return a if a.shape[1] == n else torch.cat(
        [a, a[:, -1:].expand(-1, n - a.shape[1])], dim=1)


def puff_tables(
    *,
    centres: Tensor,
    u_star: Tensor | float,
    h_abl: Tensor | float,
    lmo: Tensor | float,
    h_canopy: Tensor | float,
    z0: Tensor | float,
    d: Tensor | float,
    sigma_theta: Tensor | float,
    theta_star: Tensor | float | None = None,
    temperature: Tensor | float = DEFAULT_TEMPERATURE_K,
    sigma_v_min: float = SIGMA_V_MIN,
    sigma_w_min: float = SIGMA_W_MIN,
    tau_y: Tensor | float | None = None,
    tau_z: Tensor | float | None = None,
    centre_k: Tensor | float = PLUME_CENTRE_K,
    x_max: float = DOWNWIND_CUTOFF_M,
    kappa: float = KAPPA,
) -> PuffTables:
    """SIRANE's plume trajectory by age (module docstring), one table per entry of
    `centres`.

    `centres` (m) is `centre_c` of the centre law, `(n_tab,)` or `(n_tab, n_b)`; the
    meteorology (`u_star`, `h_abl`, `lmo`, `sigma_theta`, `theta_star`, `temperature`, and
    the empirical inputs, all as in `plume_table`) is scalar or `(n_b,)` and broadcasts
    to `n_b` hours. `x_max` is the extent: each row's scan ends at the first node beyond it.
    Needs `h_abl > h_canopy > d`, `u* > 0` and `tau_z < TIME_STEP_S`; a wind at `H_R` so
    low that the scan would need more than `MAX_TABLE_STEPS` steps, and the unstable regime,
    are refused by name. Rows are built in chunks under `KERNEL_BYTES_CAP`
    (`table_chunks`)."""
    centres = torch.as_tensor(centres, dtype=_DTYPE)
    if centres.ndim not in (1, 2):
        raise ValueError(
            f"puff_tables: centres must have shape (n_tab,) or (n_tab, n_b), got "
            f"{tuple(centres.shape)}")
    p, batch, u0 = _table_inputs(
        "puff_tables", u_star=u_star, h_abl=h_abl, lmo=lmo, h_canopy=h_canopy, z0=z0, d=d,
        sigma_theta=sigma_theta, theta_star=theta_star, temperature=temperature,
        sigma_v_min=sigma_v_min, sigma_w_min=sigma_w_min, tau_y=tau_y, tau_z=tau_z,
        centre_c=centres if centres.ndim == 2 else centres[:, None], centre_k=centre_k,
        x_max=x_max, kappa=kappa)
    if len(batch) != 2:
        raise ValueError(
            f"puff_tables: the meteorology must be scalar or (n_b,) and centres (n_tab,) "
            f"or (n_tab, n_b); the broadcast shape is {batch}")
    n_tab, n_b = batch
    x_end = float(x_max)
    steps = _step_bound("puff_tables", u0, x_end)
    keys = tuple(p)
    grad = torch.is_grad_enabled() and any(v.requires_grad for v in p.values())
    out = []
    for a_, b_ in table_chunks(steps.tolist()):
        values = [p[k][a_:b_] for k in keys]
        if grad:
            out.append(_RecomputedNodes.apply(keys, kappa, x_end, *values))
        else:
            nodes = _trajectory_nodes(dict(zip(keys, values, strict=True)), None, kappa,
                                      x_end=x_end)
            out.append(tuple(nodes[k] for k in _NODE_KEYS))
    n_nodes = max(o[0].shape[1] for o in out)
    cols = [torch.cat([_pad_nodes(o[i], n_nodes) for o in out]).reshape(n_tab, n_b, n_nodes)
            for i in range(len(_NODE_KEYS))]
    return PuffTables(*cols, h_canopy=p["h_r"].reshape(batch)[0],
                      h_abl=p["h_abl"].reshape(batch)[0], x_max=x_end)


__all__ = ["PuffTables", "puff_tables"]

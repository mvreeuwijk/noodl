"""The `Reaction` protocol: a local per-node map applied around a transport step by operator
splitting.

First-order decay is not a reaction here: it is the transport layer's `removal`. The
applications supply their own reactions (the street model's `Photostationary`, the sewer's
`SulfideGeneration`)."""

from __future__ import annotations

from collections.abc import Mapping

import torch


class Reaction:
    """A local, per-node nonlinear map applied to the state after a transport step (or before
    it, under `Model(reaction_order="before_transport")`)."""

    def apply(
        self, x: torch.Tensor, dt: float, drivers: Mapping[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

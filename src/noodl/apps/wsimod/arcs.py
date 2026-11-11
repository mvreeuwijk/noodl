"""WSIMOD 0.8.1's plain arcs (``wsimod/arcs/arcs.py:18-270``, ``:808-905``), batched.

An arc clips a push or a pull to ``capacity - flow_in`` and to the far node's check, passes the
rest on, and accumulates what it moved in ``flow_in`` and ``vqip_in`` until ``end_timestep``,
so capacity is served first come, first served (spec, Protocol). Every call takes the caller's
``active`` mask: the accumulators are written only where it holds (D3).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from noodl.apps.wsimod.engine import Ctx, dmax, dmin
from noodl.apps.wsimod.vqip import (
    Pollutants,
    empty,
    extract_vqip,
    sum_vqip,
    v_change_vqip,
    write,
)

if TYPE_CHECKING:  # pragma: no cover
    from noodl.apps.wsimod.config import ArcSpec
    from noodl.apps.wsimod.node import Node, Tag

__all__ = ["ARC_CLASSES", "Arc", "PullArc", "PushArc", "SewerArc", "WeirArc"]


class Arc:
    """``Arc`` (``arcs.py:18-270``): one capacity-limited link from ``in_port`` to ``out_port``.

    ``capacity`` and ``preference`` start at the configured values; :meth:`bind` replaces them
    with the day's (batched) values. ``in_type`` and ``out_type`` are the ends' WSIMOD class
    names after the renames, the keys of the nodes' ``in_arcs_type``/``out_arcs_type``.
    """

    ctx: Ctx

    def __init__(self, spec: ArcSpec, in_port: Node, out_port: Node, pollutants: Pollutants):
        self.spec = spec
        self.name = spec.name
        self.p = pollutants
        self.in_port = in_port
        self.out_port = out_port
        self.in_type = in_port.wsimod_type
        self.out_type = out_port.wsimod_type
        self.capacity = torch.tensor(float(spec.capacity), dtype=torch.float64)
        self.preference = torch.tensor(float(spec.preference), dtype=torch.float64)
        self.end_timestep()
        # arcs.py:70-80: the ports learn of the arc in construction order
        in_port.connect(self, "out")
        out_port.connect(self, "in")

    def bind(self, capacity: Tensor, preference: Tensor) -> None:
        """This day's ``capacity`` and ``preference`` (batched)."""
        self.capacity = torch.as_tensor(capacity, dtype=torch.float64)
        self.preference = torch.as_tensor(preference, dtype=torch.float64)

    def stores(self) -> dict[str, Tensor]:
        """Carried state: none for a plain arc (its accumulators reset every day)."""
        return {}

    def load_stores(self, stores: Mapping[str, Tensor]) -> None:
        if stores:
            raise KeyError(f"arc {self.name!r} carries no state, got {sorted(stores)}")

    def end_timestep(self) -> None:
        """``arcs.py:259-265``: reset the day's accumulators."""
        self.flow_in = torch.zeros((), dtype=torch.float64)
        self.flow_out = self.flow_in
        self.vqip_in = empty(self.p, ())
        self.vqip_out = self.vqip_in

    def _accumulate(self, moved: Tensor, active: Tensor) -> None:
        """``arcs.py:152-156``, ``:191-195``."""
        self.flow_in = write(active, self.flow_in + moved[..., 0], self.flow_in)
        self.flow_out = self.flow_in
        self.vqip_in = write(active, sum_vqip(self.p, self.vqip_in, moved), self.vqip_in)
        self.vqip_out = self.vqip_in

    def send_push_request(
        self, vqip: Tensor, *, active: Tensor, tag: Tag = "default", force: bool = False
    ) -> Tensor:
        """``arcs.py:112-158``: push ``vqip`` to ``out_port``; returns what was not pushed.

        ``force`` skips the clip but still adds to ``flow_in`` (``:131-152``), so the pipe
        excess can go negative afterwards; that is transcribed, not clamped.
        """
        p = self.p
        if force:
            not_pushed = torch.zeros_like(vqip)
        else:
            excess_in = self.get_excess("push", vqip, active=active, tag=tag)
            clip = dmax(self.ctx, f"{self.name}.push_clip", vqip[..., 0] - excess_in[..., 0], 0.0)
            not_pushed = v_change_vqip(p, vqip, clip)
        vqip = extract_vqip(p, vqip, not_pushed)
        reply = self.out_port.push_set(vqip, tag, active=active)
        vqip = extract_vqip(p, vqip, reply)
        reply = sum_vqip(p, reply, not_pushed)
        self._accumulate(vqip, active)
        return reply

    def send_pull_request(self, vqip: Tensor, *, active: Tensor, tag: Tag = "default") -> Tensor:
        """``arcs.py:160-197``: pull ``vqip``'s volume from ``in_port``; returns what was obtained.

        The request's additive entries are not rescaled (D4): no in-scope ``pull_set`` reads
        them (spec, Protocol, Request mutation); the request passed on carries the clipped
        volume and the caller's other entries.
        """
        volume = vqip[..., 0]
        excess_in = self.get_excess("pull", vqip, active=active, tag=tag)[..., 0]
        not_pulled = dmax(self.ctx, f"{self.name}.pull_clip", volume - excess_in, 0.0)
        volume = volume - not_pulled
        request = torch.cat(
            [volume[..., None], vqip[..., 1:].expand(volume.shape + (self.p.K,))], -1
        )
        reply = self.in_port.pull_set(request, tag, active=active)
        self._accumulate(reply, active)
        return reply

    def send_push_check(
        self, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``arcs.py:199-212``."""
        return self.get_excess("push", vqip, active=active, tag=tag)

    def send_pull_check(
        self, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``arcs.py:214-228``."""
        return self.get_excess("pull", vqip, active=active, tag=tag)

    def get_excess(
        self, direction: str, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``arcs.py:230-257``: ``min(capacity - flow_in, node check volume)`` at the node
        check's concentration. ``vqip=None`` asks the node for its unclipped check."""
        pipe_excess = self.capacity - self.flow_in
        if direction == "push":
            node_excess = self.out_port.push_check(vqip, tag, active=active)
        elif direction == "pull":
            node_excess = self.in_port.pull_check(vqip, tag, active=active)
        else:
            raise ValueError(f"arc {self.name!r}: direction {direction!r} is not push or pull")
        excess = dmin(self.ctx, f"{self.name}.excess", pipe_excess, node_excess[..., 0])
        return v_change_vqip(self.p, node_excess, excess)


class PullArc(Arc):
    """``PullArc`` (``arcs.py:808-849``): pushes are refused."""

    def send_push_request(
        self, vqip: Tensor, *, active: Tensor, tag: Tag = "default", force: bool = False
    ) -> Tensor:
        """``send_push_deny`` (``:821-835``): returns the VQIP."""
        return vqip

    def send_push_check(
        self, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``send_push_check_deny`` (``:837-849``): empty."""
        return self.ctx.empty()


class PushArc(Arc):
    """``PushArc`` (``arcs.py:852-893``): pulls are refused."""

    def send_pull_request(self, vqip: Tensor, *, active: Tensor, tag: Tag = "default") -> Tensor:
        """``send_pull_deny`` (``:865-879``): empty."""
        return self.ctx.empty()

    def send_pull_check(
        self, vqip: Tensor | None = None, *, active: Tensor, tag: Tag = "default"
    ) -> Tensor:
        """``send_pull_check_deny`` (``:881-893``): empty."""
        return self.ctx.empty()


class SewerArc(Arc):
    """``SewerArc`` (``arcs.py:896-899``): a plain arc."""


class WeirArc(SewerArc):
    """``WeirArc`` (``arcs.py:902-905``): a plain arc."""


ARC_CLASSES: dict[str, type[Arc]] = {
    "Arc": Arc,
    "PullArc": PullArc,
    "PushArc": PushArc,
    "SewerArc": SewerArc,
    "WeirArc": WeirArc,
}

"""WSIMOD 0.8.1's temperature-dependent decay (``DecayObj``, ``wsimod/core/core.py:432-521``).

``generic_temperature_decay`` (``:494-521``) walks the ``decays`` dict and, per pollutant,
computes ``diff = t * min(constant * exponent ** (T - 20), 1)`` and ``t -= diff``. The factor
depends only on the day's temperature and the element's parameters, and WSIMOD calls
``make_decay`` several times per element and day with the same ``T`` (``arcs.py:798-805``), so
:class:`Decay` evaluates it once per element and day, when the parameters are bound (D5), and
every :meth:`Decay.make_decay` applies the stored factor. The values are bitwise WSIMOD's: the
inputs are identical, and the power is :func:`~noodl.apps.wsimod.vqip.pow_python` (X18).

The ``decays`` key set is structure (X9): which pollutants decay fixes ``present``. Driver
entries at absent keys are inert ``(0, 1)`` and refused otherwise (:func:`check_inert`).
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from noodl.apps.wsimod.constants import DECAY_REFERENCE_TEMPERATURE
from noodl.apps.wsimod.engine import Ctx
from noodl.apps.wsimod.vqip import Pollutants, pow_python

__all__ = ["Decay", "DecaySpec", "check_inert", "read_decays"]


@dataclass(frozen=True)
class DecaySpec:
    """One element's configured decays: the keys in configuration order (X9)."""

    keys: tuple[str, ...]
    constant: dict[str, float]
    exponent: dict[str, float]


def read_decays(where: str, decays: Mapping | None, pollutants: Pollutants) -> DecaySpec | None:
    """The configured ``decays`` of ``where``; ``None`` for ``decays: None`` (D6).

    Refuses by name a key that is not a configured pollutant (``volume`` included), an entry
    whose keys are not exactly ``constant`` and ``exponent``, a ``constant`` or ``exponent``
    that is not a number, and ``exponent <= 0``
    (``core.py:513-515`` raises or returns a ``complex`` there)."""
    if decays is None:
        return None
    if not isinstance(decays, Mapping):
        raise ValueError(f"{where}: decays must be a mapping, got {type(decays).__name__}")
    constant: dict[str, float] = {}
    exponent: dict[str, float] = {}
    for key, pars in decays.items():
        if key not in pollutants.names:
            raise ValueError(f"{where}: decays key {key!r} is not a configured pollutant")
        if not isinstance(pars, Mapping) or set(pars) != {"constant", "exponent"}:
            got = sorted(pars) if isinstance(pars, Mapping) else pars
            raise ValueError(
                f"{where}: decays[{key!r}] must have exactly the keys 'constant' and "
                f"'exponent', got {got}"
            )
        for field in ("constant", "exponent"):
            if isinstance(pars[field], str) or not isinstance(pars[field], numbers.Real):
                raise ValueError(
                    f"{where}: decays[{key!r}] {field} {pars[field]!r} is not a number "
                    "(WSIMOD multiplies it, core.py:513-515)"
                )
        if not float(pars["exponent"]) > 0:
            raise ValueError(
                f"{where}: decays[{key!r}] exponent {pars['exponent']!r} must be > 0 "
                "(WSIMOD raises or returns a complex number there)"
            )
        constant[key] = float(pars["constant"])
        exponent[key] = float(pars["exponent"])
    return DecaySpec(tuple(decays), constant, exponent)


def check_inert(
    where: str,
    key: str,
    present: Tensor,
    constant: Tensor,
    exponent: Tensor,
    names: Sequence[str] | None = None,
) -> None:
    """Refuse an absent column whose ``(constant, exponent)`` is not ``(0, 1)`` (X9, D15).

    ``present`` is ``(1 + K,)`` (column 0 the volume) or ``(K,)``; ``constant`` and
    ``exponent`` are ``(..., K)``. The refusal names ``where``, the driver ``key`` and the
    pollutant (``names``, else its column)."""
    k = constant.shape[-1]
    absent = ~present[-k:]
    non_inert = ((constant != 0) | (exponent != 1)).reshape(-1, k).any(0)
    bad = absent & non_inert
    if bool(bad.any()):
        i = int(bad.nonzero()[0])
        name = names[i] if names is not None else f"column {i}"
        raise ValueError(
            f"{where}: {key} has a non-inert entry at {name!r}, which the configuration's "
            "decays does not list; an absent key's (constant, exponent) must be (0, 1) "
            "(the key set is structure)"
        )


class Decay:
    """The decay of one element (``DecayObj``): ``factor`` is bound once per day (D5)."""

    def __init__(self, pollutants: Pollutants, spec: DecaySpec, *, site: str) -> None:
        self.p = pollutants
        self.spec = spec
        self.site = site
        self.present = torch.tensor(
            [False] + [n in spec.keys for n in pollutants.names], dtype=torch.bool
        )
        self.constant = torch.tensor(
            [spec.constant.get(n, 0.0) for n in pollutants.names], dtype=torch.float64
        ).reshape(pollutants.K)
        self.exponent = torch.tensor(
            [spec.exponent.get(n, 1.0) for n in pollutants.names], dtype=torch.float64
        ).reshape(pollutants.K)
        self.factor: Tensor | None = None

    def bind(self, ctx: Ctx, constant: Tensor, exponent: Tensor, temperature: Tensor) -> None:
        """Evaluate the day's ``min(constant * exponent ** (T - 20), 1)`` (``core.py:513-515``).

        One decision per present pollutant, ``<site>/make_decay/cap:<pollutant>``, true where
        the factor exceeds 1 (Python's ``min(factor, 1)`` keeps the factor unless ``1 <
        factor``); the cap gives exactly 1, so ``t - t * 1`` is ``+0.0``."""
        constant = torch.as_tensor(constant, dtype=torch.float64)
        exponent = torch.as_tensor(exponent, dtype=torch.float64)
        temperature = torch.as_tensor(temperature, dtype=torch.float64)
        check_inert(
            self.site, "decays.constant/exponent", self.present, constant, exponent, self.p.names
        )
        present = self.present[1:]
        nonpositive = ((exponent <= 0) & present).reshape(-1, self.p.K).any(0)
        if bool(nonpositive.any()):
            i = int(nonpositive.nonzero()[0])
            raise ValueError(
                f"{self.site}: decays exponent of {self.p.names[i]!r} must be > 0 "
                "(WSIMOD raises or returns a complex number there)"
            )
        self.constant, self.exponent = constant, exponent
        raw = constant * pow_python(exponent, temperature[..., None] - DECAY_REFERENCE_TEMPERATURE)
        # every column at the day's batch shape, whatever the drivers' own (a batched capacity
        # leaves the decays unbatched)
        shape = torch.broadcast_shapes(raw.shape[:-1], ctx.shape)
        raw = torch.broadcast_to(raw, (*shape, self.p.K))
        all_true = torch.ones(ctx.shape, dtype=torch.bool)
        columns = []
        for i, name in enumerate(self.p.names):
            if bool(present[i]):
                cap = ctx.decide(
                    f"{self.site}/make_decay/cap:{name}",
                    raw[..., i],
                    ">",
                    1.0,
                    active=all_true,
                    call=0,
                )
                columns.append(torch.where(cap, torch.ones_like(raw[..., i]), raw[..., i]))
            else:
                columns.append(torch.zeros_like(raw[..., i]))
        factor = torch.stack(columns, dim=-1) if columns else torch.zeros_like(raw)
        self.factor = torch.cat([torch.zeros_like(factor[..., :1]), factor], dim=-1)

    def make_decay(self, vqip: Tensor) -> tuple[Tensor, Tensor]:
        """``(decayed, diff)`` of ``generic_temperature_decay`` (``core.py:507-519``): ``diff``
        is ``t * factor`` at the present columns and 0 elsewhere (the volume included)."""
        if self.factor is None:
            raise RuntimeError(f"{self.site}: Decay.bind has not run")
        diff = torch.where(self.present, vqip * self.factor, torch.zeros_like(vqip))
        decayed = torch.where(self.present, vqip - diff, vqip)
        return decayed, diff

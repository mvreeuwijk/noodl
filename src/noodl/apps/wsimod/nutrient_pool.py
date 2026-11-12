"""WSIMOD 0.8.1's soil nutrient pools (``wsimod/nodes/nutrient_pool.py``), batched.

A growing surface holds one :class:`NutrientPool` (``np.py:11-564``, ``land.py:1273``).
WSIMOD's five ``NutrientStore``s, each a dict ``{"N": ..., "P": ...}``, are one dense
``storage`` tensor ``(..., 5, 2)``: the pools in WSIMOD's ``self.pools`` order (:data:`POOLS`,
``np.py:130-136``) and the nutrients ``N``, then ``P`` (``constants.NUTRIENTS``). WSIMOD's
in-place ``+=`` and ``-=`` become functional writes under the caller's ``active`` mask, in
WSIMOD's statement order, and every request is computed before its pool changes.

The arithmetic is ``+ - * /`` and ``min`` only, in WSIMOD's operand order, so every value is
bitwise WSIMOD's; no per-instance Python math is needed. Python's integer zeros (the empty
nutrient ``{"N": 0, "P": 0}``, ``np.py:185``, and ``allocate_dry_deposition``'s ``"P": 0``) are
float zeros here; ``0 + x`` and ``x - 0`` give the same float in both, up to the sign of a zero,
which ``torch.equal`` ignores (rule 6).

Each ``NutrientStore.extract`` (``np.py:551-556``) is, per nutrient in the request's order,
``reply = min(storage, amount)`` then ``storage -= reply``: the decision
``"<site>.<pool>/extract/clip:<N|P>"``, whose ``call`` is that pool's extract count of the day.
``site`` is the owner's attribute path, ``"node:<land>/surfaces.<s>.nutrient_pool"``.

Not ported: ``allocate_residue`` (``np.py:292-312``; it would raise ``TypeError``, and residue
data are refused, Y36) and ``apply_overrides`` (``np.py:155-181``; nothing calls it, L20).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from noodl.apps.wsimod.engine import Ctx, dmin

__all__ = [
    "ADS",
    "DI",
    "DO",
    "FAST",
    "HUMUS",
    "NUTRIENTS",
    "NUTRIENT_POOL_PAIRS",
    "NUTRIENT_POOL_SCALARS",
    "POOLS",
    "NutrientPool",
]

# np.py:130-136, WSIMOD's self.pools order
POOLS = (
    "fast_pool",
    "humus_pool",
    "dissolved_inorganic_pool",
    "dissolved_organic_pool",
    "adsorbed_inorganic_pool",
)
FAST, HUMUS, DI, DO, ADS = range(5)
NUTRIENTS = ("N", "P")  # constants.NUTRIENTS

# np.py:16-23, WSIMOD's defaults as written: (N, P)
NUTRIENT_POOL_PAIRS: dict[str, tuple[float, float]] = {
    "degrhpar": (7 * 1e-5, 7 * 1e-6),
    "dishpar": (7 * 1e-5, 7 * 1e-6),
    "minfpar": (0.00013, 0.000003),
    "disfpar": (0.000003, 0.0000001),
    "immobdpar": (0.0056, 0.2866),
    "fraction_manure_to_dissolved_inorganic": (0.5, 0.1),
}
NUTRIENT_POOL_SCALARS: dict[str, float] = {"fraction_dry_n_to_dissolved_inorganic": 0.9}


def _f64(x) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


class NutrientPool:
    """``NutrientPool`` (``np.py:11-564``): five ``(N, P)`` stores and their transformations.

    ``storage`` is ``(..., 5, 2)``, zeros at construction (``NutrientStore.init_store``,
    ``np.py:187-190``). ``temperature_dependence_factor`` and ``soil_moisture_dependence_factor``
    are the day's factors ``(...)``, 0 until the owner sets them (``np.py:102-103``). The
    parameters start at WSIMOD's defaults; :meth:`bind` replaces them.

    Every method takes the caller's ``active`` mask and writes ``storage`` only where it holds.
    A method's replies are computed on every instance; only the active ones are WSIMOD's.
    """

    def __init__(self, *, site: str) -> None:
        self.site = site
        element, _, path = site.partition("/")
        self._element, self._path = element, path
        self.storage = torch.zeros(len(POOLS), len(NUTRIENTS), dtype=torch.float64)
        self.temperature_dependence_factor = _f64(0.0)
        self.soil_moisture_dependence_factor = _f64(0.0)
        self._ctx: Ctx | None = None
        self.bind(
            {
                **{name: _f64(pair) for name, pair in NUTRIENT_POOL_PAIRS.items()},
                **{name: _f64(value) for name, value in NUTRIENT_POOL_SCALARS.items()},
            }
        )

    # --- the day's context ------------------------------------------------------------------
    @property
    def ctx(self) -> Ctx:
        if self._ctx is None:
            raise RuntimeError(f"{self.site}: no day context; the owner sets nutrient_pool.ctx")
        return self._ctx

    @ctx.setter
    def ctx(self, ctx: Ctx) -> None:
        self._ctx = ctx

    # --- parameters -------------------------------------------------------------------------
    def bind(self, parameters: Mapping[str, Tensor]) -> None:
        """Set all seven parameters: ``(..., 2)`` for each of the :data:`NUTRIENT_POOL_PAIRS`
        and ``(...)`` for ``fraction_dry_n_to_dissolved_inorganic``. A missing or unknown name
        is refused by name, and so is a ``fraction_dry_n_to_dissolved_inorganic`` that is
        neither 0-d nor of the pairs' batch shape (it would broadcast against the nutrient
        axis). Then the derived fractions as ``calculate_fraction_parameters``
        (``np.py:146-153``): ``fraction_manure_to_fast = 1 -
        fraction_manure_to_dissolved_inorganic`` per nutrient, and ``fraction_dry_n_to_fast =
        1 - fraction_dry_n_to_dissolved_inorganic``."""
        known = (*NUTRIENT_POOL_PAIRS, *NUTRIENT_POOL_SCALARS)
        unknown = [name for name in parameters if name not in known]
        if unknown:
            raise ValueError(
                f"{self.site}: unknown NutrientPool parameter(s) {unknown}; known: {list(known)}"
            )
        missing = [name for name in known if name not in parameters]
        if missing:
            raise ValueError(f"{self.site}: NutrientPool parameter(s) {missing} not given")
        values = {name: _f64(parameters[name]) for name in known}
        for name in NUTRIENT_POOL_PAIRS:
            value = values[name]
            if value.ndim < 1 or value.shape[-1] != 2:
                raise ValueError(
                    f"{self.site}: {name} of shape {tuple(value.shape)}; expected (..., 2), "
                    "N then P"
                )
        try:
            batch = torch.broadcast_shapes(*(values[n].shape[:-1] for n in NUTRIENT_POOL_PAIRS))
        except RuntimeError as error:
            raise ValueError(
                f"{self.site}: the NutrientPool pair parameters' batch shapes do not broadcast"
            ) from error
        for name in NUTRIENT_POOL_SCALARS:
            shape = values[name].shape
            if values[name].ndim != 0 and shape != batch:
                raise ValueError(
                    f"{self.site}: {name} of shape {tuple(shape)}; expected () or the pairs' "
                    f"batch shape {tuple(batch)}"
                )
        for name, value in values.items():
            setattr(self, name, value)
        self.fraction_manure_to_fast = 1 - self.fraction_manure_to_dissolved_inorganic
        self.fraction_dry_n_to_fast = 1 - self.fraction_dry_n_to_dissolved_inorganic

    # --- the stores -------------------------------------------------------------------------
    def _put(self, pool: int, nutrient: int | None, value: Tensor, active: Tensor) -> None:
        """Write ``value`` into ``storage[..., pool, nutrient]`` (both nutrients for ``None``,
        ``value`` then ``(..., 2)``) where ``active``."""
        hot = torch.zeros(len(POOLS), len(NUTRIENTS), dtype=torch.bool)
        if nutrient is None:
            hot[pool] = True
            new = torch.where(hot, value[..., None, :], self.storage)
        else:
            hot[pool, nutrient] = True
            new = torch.where(hot, value[..., None, None], self.storage)
        self.storage = torch.where(active[..., None, None], new, self.storage)

    def receive(self, pool: int, amount: Tensor, *, active: Tensor) -> None:
        """``NutrientStore.receive`` (``np.py:503-511``): ``storage[n] += amount[n]``."""
        active = torch.as_tensor(active, dtype=torch.bool)
        self._put(pool, None, self.storage[..., pool, :] + _f64(amount), active)

    def extract(
        self,
        pool: int,
        amount: Tensor,
        *,
        active: Tensor,
        order: tuple[str, ...] = NUTRIENTS,
    ) -> Tensor:
        """``NutrientStore.extract`` (``np.py:542-556``): per nutrient in ``order`` (the request
        dict's insertion order), ``reply = min(storage, amount)``, then ``storage -= reply``.
        Returns the reply ``(..., 2)``, N then P."""
        active = torch.as_tensor(active, dtype=torch.bool)
        amount = _f64(amount)
        if any(name not in NUTRIENTS for name in order) or len(set(order)) != len(order):
            raise ValueError(
                f"{self.site}.{POOLS[pool]}/extract: order {order!r}; expected distinct names "
                f"from {NUTRIENTS} (a dict request holds each nutrient at most once)"
            )
        call = self.ctx.call(self._element, f"{self._path}.{POOLS[pool]}/extract")
        reply = [_f64(0.0), _f64(0.0)]  # the empty nutrient (np.py:551)
        for name in order:
            j = NUTRIENTS.index(name)
            stored = self.storage[..., pool, j]
            value = dmin(
                self.ctx,
                f"{self.site}.{POOLS[pool]}/extract/clip:{name}",
                stored,
                amount[..., j],
                active=active,
                call=call,
                round=None,
            )
            reply[j] = value
            self._put(pool, j, stored - value, active)
        return torch.stack(torch.broadcast_tensors(*reply), dim=-1)

    # --- inputs -----------------------------------------------------------------------------
    def allocate_inorganic_irrigation(self, amount: Tensor, *, active: Tensor) -> Tensor:
        """``np.py:192-207``: into the dissolved inorganic pool; returns ``amount``."""
        self.receive(DI, amount, active=active)
        return _f64(amount)

    def allocate_organic_irrigation(self, amount: Tensor, *, active: Tensor) -> Tensor:
        """``np.py:209-224``: into the dissolved organic pool; returns ``amount``."""
        self.receive(DO, amount, active=active)
        return _f64(amount)

    def allocate_dry_deposition(self, dep: Tensor, *, active: Tensor) -> Tensor:
        """``np.py:226-248``: ``fast.N += dep.N * fraction_dry_n_to_fast``, then ``DI.N +=
        dep.N * fraction_dry_n_to_dissolved_inorganic``, then ``ads.P += dep.P``. Returns
        ``(dep.N * fraction_dry_n_to_dissolved_inorganic, 0)``."""
        active = torch.as_tensor(active, dtype=torch.bool)
        dep = _f64(dep)
        n, p = dep[..., 0], dep[..., 1]
        self._put(FAST, 0, self.storage[..., FAST, 0] + n * self.fraction_dry_n_to_fast, active)
        fddi = self.fraction_dry_n_to_dissolved_inorganic
        self._put(DI, 0, self.storage[..., DI, 0] + n * fddi, active)
        self._put(ADS, 1, self.storage[..., ADS, 1] + p, active)
        n_in = n * fddi
        return torch.stack([n_in, torch.zeros_like(n_in)], dim=-1)

    def allocate_wet_deposition(self, dep: Tensor, *, active: Tensor) -> Tensor:
        """``np.py:250-265``: into the dissolved inorganic pool; returns ``dep``."""
        self.receive(DI, dep, active=active)
        return _f64(dep)

    def allocate_manure(self, dep: Tensor, *, active: Tensor) -> Tensor:
        """``np.py:267-290``: ``DI.receive(dep * fmdi)``, then ``fast.receive(dep * fmf)``
        (``multiply_nutrients``, ``np.py:501``); returns ``dep * fmdi``, recomputed."""
        dep = _f64(dep)
        self.receive(DI, dep * self.fraction_manure_to_dissolved_inorganic, active=active)
        self.receive(FAST, dep * self.fraction_manure_to_fast, active=active)
        return dep * self.fraction_manure_to_dissolved_inorganic

    def allocate_fertiliser(self, dep: Tensor, *, active: Tensor) -> Tensor:
        """``np.py:314-328``: into the dissolved inorganic pool; returns ``dep``."""
        self.receive(DI, dep, active=active)
        return _f64(dep)

    # --- outputs ----------------------------------------------------------------------------
    def extract_dissolved(self, proportion: Tensor, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``np.py:330-356``: ``DI.extract(DI.storage * proportion)``, then
        ``DO.extract(DO.storage * proportion)``. Returns ``(inorganic, organic)``."""
        proportion = _f64(proportion)[..., None]
        inorganic = self.extract(DI, self.storage[..., DI, :] * proportion, active=active)
        organic = self.extract(DO, self.storage[..., DO, :] * proportion, active=active)
        return inorganic, organic

    def get_erodable_P(self) -> Tensor:
        """``np.py:358-365``: ``ads.P + humus.P``."""
        return self.storage[..., ADS, 1] + self.storage[..., HUMUS, 1]

    def erode_P(self, amount: Tensor, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``np.py:367-394``: ``fa = ads.P / (ads.P + humus.P)``; ``ads.extract({N: 0, P:
        amount * fa})``, then ``humus.extract({N: 0, P: amount * (1 - fa)})``. Returns
        ``(humus removed, adsorbed removed)``, the P replies.

        Every zero denominator, active or not, is replaced by 1, so values and gradients stay
        finite: such an instance then removes nothing and its stores are unchanged. On an
        active one WSIMOD would raise ``ZeroDivisionError``; its caller excludes it, because
        the guard ``eff_erodedP > 0`` (``land.py:2048``) implies ``ads.P + humus.P != 0``
        (``land.py:2001-2013``), so the caller's ``active`` holds no zero denominator."""
        active = torch.as_tensor(active, dtype=torch.bool)
        amount = _f64(amount)
        ads_p = self.storage[..., ADS, 1]
        denominator = ads_p + self.storage[..., HUMUS, 1]
        nonzero = (denominator != 0).detach()
        fraction_adsorbed = ads_p / torch.where(nonzero, denominator, torch.ones_like(denominator))
        zero = torch.zeros_like(fraction_adsorbed * amount)
        request = torch.stack([zero, amount * fraction_adsorbed], dim=-1)
        reply_adsorbed = self.extract(ADS, request, active=active)
        request = torch.stack([zero, amount * (1 - fraction_adsorbed)], dim=-1)
        reply_humus = self.extract(HUMUS, request, active=active)
        return reply_humus[..., 1], reply_adsorbed[..., 1]

    # --- transformations --------------------------------------------------------------------
    def temp_soil_process(
        self, parameter: Tensor, source: int, target: int, *, active: Tensor
    ) -> Tensor:
        """``np.py:452-481``: ``x[n] = parameter[n] * tdf * smdf * source.storage[n]``, then
        ``x = source.extract(x)`` and ``target.receive(x)``. Returns ``x``."""
        active = torch.as_tensor(active, dtype=torch.bool)
        tdf = _f64(self.temperature_dependence_factor)[..., None]
        smdf = _f64(self.soil_moisture_dependence_factor)[..., None]
        to_extract = _f64(parameter) * tdf * smdf * self.storage[..., source, :]
        to_extract = self.extract(source, to_extract, active=active)
        self.receive(target, to_extract, active=active)
        return to_extract

    def soil_pool_transformation(self, *, active: Tensor) -> tuple[Tensor, Tensor]:
        """``np.py:396-450``: the five processes in WSIMOD's order. Returns ``(increase in
        dissolved inorganic, increase in dissolved organic)``, each summed from the empty
        nutrient (``sum_nutrients``, ``subtract_nutrients``, ``np.py:513-540``)."""
        empty = torch.zeros(2, dtype=torch.float64)
        # humus -> fast, untracked (np.py:413)
        self.temp_soil_process(self.degrhpar, HUMUS, FAST, active=active)
        # humus -> dissolved organic (np.py:418-423)
        amount = self.temp_soil_process(self.dishpar, HUMUS, DO, active=active)
        increase_in_dissolved_organic = empty + amount
        # fast -> dissolved inorganic (np.py:426-431)
        amount = self.temp_soil_process(self.minfpar, FAST, DI, active=active)
        increase_in_dissolved_inorganic = empty + amount
        # fast -> dissolved organic (np.py:434-439)
        amount = self.temp_soil_process(self.disfpar, FAST, DO, active=active)
        increase_in_dissolved_organic = increase_in_dissolved_organic + amount
        # immobilisation, dissolved inorganic -> fast (np.py:442-447)
        amount = self.temp_soil_process(self.immobdpar, DI, FAST, active=active)
        increase_in_dissolved_inorganic = increase_in_dissolved_inorganic - amount
        return increase_in_dissolved_inorganic, increase_in_dissolved_organic

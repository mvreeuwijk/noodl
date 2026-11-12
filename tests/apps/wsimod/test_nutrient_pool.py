"""The batched ``NutrientPool`` (``wsimod/nodes/nutrient_pool.py``, ``np.py:11-564``).

Every method is ``+ - *`` and ``min`` in WSIMOD's operand order, so the random-sequence test
compares every store and every reply against a live WSIMOD ``NutrientPool`` bitwise
(``abs(a - b) <= 0``, under which ``-0.0 == 0.0``, rule 6).
"""

from __future__ import annotations

import random

import pytest
import torch

from noodl.apps.wsimod.engine import Ctx
from noodl.apps.wsimod.nutrient_pool import (
    ADS,
    DI,
    DO,
    FAST,
    HUMUS,
    NUTRIENT_POOL_PAIRS,
    NUTRIENT_POOL_SCALARS,
    NUTRIENTS,
    POOLS,
    NutrientPool,
)
from noodl.apps.wsimod.vqip import QUANTITY_ONLY

F64 = torch.float64
SITE = "node:L/surfaces.g.nutrient_pool"
TRUE = torch.tensor(True)
PARAMETERS = ("degrhpar", "dishpar", "minfpar", "disfpar", "immobdpar")
ALLOCATIONS = (
    "allocate_inorganic_irrigation",
    "allocate_organic_irrigation",
    "allocate_dry_deposition",
    "allocate_wet_deposition",
    "allocate_manure",
    "allocate_fertiliser",
)
METHODS = (
    *ALLOCATIONS,
    "extract_dissolved",
    "get_erodable_P",
    "erode_P",
    "soil_pool_transformation",
    "temp_soil_process",
    "receive",
    "extract",
)


def make_pool(shape=(), record=False) -> tuple[NutrientPool, Ctx]:
    pool = NutrientPool(site=SITE)
    ctx = Ctx(
        QUANTITY_ONLY, 1e-11, torch.Size(shape), day=0, node_names=[], record_decisions=record
    )
    pool.ctx = ctx
    return pool, ctx


def t(*values) -> torch.Tensor:
    return torch.tensor(values, dtype=F64)


def set_storage(pool: NutrientPool, rows: dict[int, tuple[float, float]]) -> None:
    storage = torch.zeros(5, 2, dtype=F64)
    for index, row in rows.items():
        storage[index] = t(*row)
    pool.storage = storage


# --- live WSIMOD ----------------------------------------------------------------------------
def _wsimod_pool():
    pytest.importorskip("wsimod")
    from wsimod.nodes.nutrient_pool import NutrientPool as WsimodPool

    from tests.verification._wsimod_reference import reset_wsimod_globals

    reset_wsimod_globals()
    return WsimodPool()


def _amount(rng: random.Random) -> float:
    r = rng.random()
    if r < 0.15:
        return 0.0
    return rng.uniform(0.0, 10.0) * 10.0 ** rng.randint(-3, 2)


def _factor(rng: random.Random) -> float:
    r = rng.random()
    if r < 0.1:
        return 0.0
    if r < 0.2:
        return 1.0
    return rng.uniform(0.0, 1.5)


def _request(rng: random.Random, storage: float) -> float:
    """A request at, below or above the store (the ``min`` cap and its tie)."""
    r = rng.random()
    if r < 0.2:
        return storage
    if r < 0.3:
        return 0.0
    return storage * rng.uniform(0.0, 2.0)


def _same(got, ref) -> bool:
    return abs(float(got) - ref) <= 0


def _assert_stores(wp, pool: NutrientPool, where: str) -> None:
    for i, store in enumerate(wp.pools):
        for j, n in enumerate(NUTRIENTS):
            ref, got = store.storage[n], pool.storage[i, j]
            assert _same(got, ref), f"{where}: {POOLS[i]}.{n} noodl {float(got)!r} wsimod {ref!r}"


def _assert_nutrient(got, ref: dict, where: str) -> None:
    for j, n in enumerate(NUTRIENTS):
        assert _same(got[j], ref[n]), f"{where}: {n} noodl {float(got[j])!r} wsimod {ref[n]!r}"


def _drive(wp, pool: NutrientPool, rng: random.Random, calls: int) -> None:
    for k in range(calls):
        name = rng.choice(METHODS)
        where = f"call {k} {name}"
        if name in ALLOCATIONS:
            a = {"N": _amount(rng), "P": _amount(rng)}
            ref = getattr(wp, name)(dict(a))
            got = getattr(pool, name)(t(a["N"], a["P"]), active=TRUE)
            _assert_nutrient(got, ref, where)
        elif name == "extract_dissolved":
            proportion = rng.choice((0.0, 1.0, rng.random()))
            ref = wp.extract_dissolved(proportion)
            inorganic, organic = pool.extract_dissolved(t(proportion)[0], active=TRUE)
            _assert_nutrient(inorganic, ref["inorganic"], where)
            _assert_nutrient(organic, ref["organic"], where)
        elif name == "get_erodable_P":
            assert _same(pool.get_erodable_P(), wp.get_erodable_P()), where
        elif name == "erode_P":
            erodable = wp.get_erodable_P()
            if not erodable > 0:  # WSIMOD's caller guards it (land.py:2049); 0/0 otherwise
                continue
            amount = erodable * rng.uniform(0.0, 1.2)
            ref = wp.erode_P(amount)
            got = pool.erode_P(t(amount)[0], active=TRUE)
            assert _same(got[0], ref[0]) and _same(got[1], ref[1]), where
        elif name in ("soil_pool_transformation", "temp_soil_process"):
            tdf, smdf = _factor(rng), _factor(rng)
            wp.temperature_dependence_factor = tdf
            wp.soil_moisture_dependence_factor = smdf
            pool.temperature_dependence_factor = t(tdf)[0]
            pool.soil_moisture_dependence_factor = t(smdf)[0]
            if name == "soil_pool_transformation":
                ref = wp.soil_pool_transformation()
                got = pool.soil_pool_transformation(active=TRUE)
                _assert_nutrient(got[0], ref[0], where)
                _assert_nutrient(got[1], ref[1], where)
            else:
                parameter = rng.choice(PARAMETERS)
                source, target = rng.sample(range(5), 2)
                ref = wp.temp_soil_process(
                    getattr(wp, parameter), wp.pools[source], wp.pools[target]
                )
                got = pool.temp_soil_process(getattr(pool, parameter), source, target, active=TRUE)
                _assert_nutrient(got, ref, where)
        elif name == "receive":
            index = rng.randrange(5)
            a = {"N": _amount(rng), "P": _amount(rng)}
            wp.pools[index].receive(dict(a))
            pool.receive(index, t(a["N"], a["P"]), active=TRUE)
        else:
            index = rng.randrange(5)
            store = wp.pools[index].storage
            a = {n: _request(rng, store[n]) for n in NUTRIENTS}
            ref = wp.pools[index].extract(dict(a))
            got = pool.extract(index, t(a["N"], a["P"]), active=TRUE)
            _assert_nutrient(got, ref, where)
        _assert_stores(wp, pool, where)


@pytest.mark.parametrize("parameters", ["default", "random"])
def test_every_method_equals_wsimods_on_random_sequences(parameters):
    wp = _wsimod_pool()
    pool, _ = make_pool()
    rng = random.Random(20261112 + (parameters == "random"))
    if parameters == "random":
        bound = {}
        for name in NUTRIENT_POOL_PAIRS:
            values = {"N": rng.uniform(0.0, 0.5), "P": rng.uniform(0.0, 0.5)}
            setattr(wp, name, values)  # a fresh dict: WSIMOD's defaults are shared (L20)
            bound[name] = t(values["N"], values["P"])
        fddi = rng.random()
        wp.fraction_dry_n_to_dissolved_inorganic = fddi
        wp.calculate_fraction_parameters()
        bound["fraction_dry_n_to_dissolved_inorganic"] = t(fddi)[0]
        pool.bind(bound)
    _assert_stores(wp, pool, "start")
    _drive(wp, pool, rng, 2000)


def test_the_defaults_are_wsimods():
    wp = _wsimod_pool()
    pool, _ = make_pool()
    for name in NUTRIENT_POOL_PAIRS:
        _assert_nutrient(getattr(pool, name), getattr(wp, name), name)
    _assert_nutrient(pool.fraction_manure_to_fast, wp.fraction_manure_to_fast, "fmf")
    for name, value in NUTRIENT_POOL_SCALARS.items():
        assert _same(getattr(pool, name), getattr(wp, name)) and value == getattr(wp, name)
    assert _same(pool.fraction_dry_n_to_fast, wp.fraction_dry_n_to_fast)


def test_bind_refuses_an_unknown_parameter():
    pool, _ = make_pool()
    with pytest.raises(ValueError, match="fraction_residue_to_fast"):
        pool.bind({"fraction_residue_to_fast": t(0.1, 0.1)})


# --- the methods by hand --------------------------------------------------------------------
def test_extract_takes_pythons_min_per_nutrient_in_request_order():
    pool, ctx = make_pool(record=True)
    set_storage(pool, {DI: (2.0, 1.0)})
    reply = pool.extract(DI, t(3.0, 1.0), active=TRUE, order=("P", "N"))
    assert torch.equal(reply, t(2.0, 1.0))
    site = f"{SITE}.dissolved_inorganic_pool/extract/clip"
    assert [d.site for d in ctx.decisions] == [f"{site}:P", f"{site}:N"]
    assert [d.call for d in ctx.decisions] == [0, 0]
    # the tie min(1.0, 1.0) keeps the storage (b < a is false); 3.0 > 2.0 keeps it too
    assert not bool(ctx.decisions[0].outcome) and not bool(ctx.decisions[1].outcome)
    assert torch.equal(pool.storage[DI], t(0.0, 0.0))
    # a second call of the same pool's extract is call 1
    pool.extract(DI, t(0.0, 0.0), active=TRUE)
    assert [d.call for d in ctx.decisions[2:]] == [1, 1]


def test_the_transformation_by_hand():
    pool, _ = make_pool()
    tdf, smdf = 0.5, 0.8
    pool.temperature_dependence_factor = t(tdf)[0]
    pool.soil_moisture_dependence_factor = t(smdf)[0]
    set_storage(pool, {FAST: (10.0, 1.0), HUMUS: (100.0, 10.0), DI: (5.0, 0.5), DO: (2.0, 0.2)})
    fast, humus, di, do = [10.0, 1.0], [100.0, 10.0], [5.0, 0.5], [2.0, 0.2]
    pars = NUTRIENT_POOL_PAIRS
    inc_di, inc_do = [0, 0], [0, 0]
    for j in range(2):
        a1 = pars["degrhpar"][j] * tdf * smdf * humus[j]  # np.py:413
        humus[j] -= a1
        fast[j] += a1
        a2 = pars["dishpar"][j] * tdf * smdf * humus[j]  # np.py:418
        humus[j] -= a2
        do[j] += a2
        inc_do[j] = 0 + a2
        a3 = pars["minfpar"][j] * tdf * smdf * fast[j]  # np.py:426
        fast[j] -= a3
        di[j] += a3
        inc_di[j] = 0 + a3
        a4 = pars["disfpar"][j] * tdf * smdf * fast[j]  # np.py:434
        fast[j] -= a4
        do[j] += a4
        inc_do[j] = inc_do[j] + a4
        a5 = pars["immobdpar"][j] * tdf * smdf * di[j]  # np.py:442
        di[j] -= a5
        fast[j] += a5
        inc_di[j] = inc_di[j] - a5
    got_di, got_do = pool.soil_pool_transformation(active=TRUE)
    assert torch.equal(got_di, t(*inc_di)) and torch.equal(got_do, t(*inc_do))
    expected = torch.stack([t(*fast), t(*humus), t(*di), t(*do), t(0.0, 0.0)])
    assert torch.equal(pool.storage, expected)


def test_erode_P_splits_by_the_adsorbed_fraction():
    pool, ctx = make_pool(record=True)
    set_storage(pool, {ADS: (4.0, 3.0), HUMUS: (7.0, 1.0)})
    org, inorg = pool.erode_P(t(2.0)[0], active=TRUE)
    fa = 3.0 / (3.0 + 1.0)
    assert fa == 0.75
    assert float(inorg) == 2.0 * fa and float(org) == 2.0 * (1 - fa)
    assert torch.equal(pool.storage[ADS], t(4.0, 3.0 - 2.0 * fa))
    assert torch.equal(pool.storage[HUMUS], t(7.0, 1.0 - 2.0 * (1 - fa)))
    # the request dict holds N = 0 too: each pool's N is clipped against 0 (np.py:384-392)
    sites = [d.site for d in ctx.decisions]
    assert sites == [
        f"{SITE}.adsorbed_inorganic_pool/extract/clip:N",
        f"{SITE}.adsorbed_inorganic_pool/extract/clip:P",
        f"{SITE}.humus_pool/extract/clip:N",
        f"{SITE}.humus_pool/extract/clip:P",
    ]


def test_dry_deposition_goes_to_fast_di_and_adsorbed():
    pool, _ = make_pool()
    fddi = NUTRIENT_POOL_SCALARS["fraction_dry_n_to_dissolved_inorganic"]
    reply = pool.allocate_dry_deposition(t(2.0, 0.3), active=TRUE)
    assert torch.equal(reply, t(2.0 * fddi, 0.0))
    assert float(pool.storage[FAST, 0]) == 0 + 2.0 * (1 - fddi)
    assert float(pool.storage[DI, 0]) == 0 + 2.0 * fddi
    assert float(pool.storage[ADS, 1]) == 0.3
    assert float(pool.storage[FAST, 1]) == 0 and float(pool.storage[DI, 1]) == 0


def test_sums_from_the_empty_nutrient_equal_pythons_under_torch_equal():
    """Rule 6: a never-touched pool's increase is ``0 + 0.0 - 0.0``; its negation, ``-0.0`` in
    both Python and torch, compares equal to Python's under ``torch.equal``."""
    pool, _ = make_pool()
    pool.temperature_dependence_factor = t(1.0)[0]
    pool.soil_moisture_dependence_factor = t(1.0)[0]
    inc_di, inc_do = pool.soil_pool_transformation(active=TRUE)
    python = 0 + 0.0 - 0.0
    assert torch.equal(-inc_di, t(-python, -python))
    assert torch.equal(-inc_do, t(-(0 + 0.0 + 0.0), -(0 + 0.0 + 0.0)))
    assert torch.equal(pool.storage, torch.zeros(5, 2, dtype=F64))


# --- batches, masks, gradients --------------------------------------------------------------
def _batch_values(rng: random.Random, b: int, draw) -> torch.Tensor:
    return torch.tensor([[draw(rng), draw(rng)] for _ in range(b)], dtype=F64)


def _call(pool: NutrientPool, name: str, a, s, act, *, parameter, index, source, target):
    """One method of the batch test, on ``pool``."""
    if name in ALLOCATIONS:
        return getattr(pool, name)(a, active=act)
    if name == "extract_dissolved":
        return pool.extract_dissolved(s, active=act)
    if name == "erode_P":
        return pool.erode_P(s, active=act)
    if name == "soil_pool_transformation":
        return pool.soil_pool_transformation(active=act)
    if name == "temp_soil_process":
        return pool.temp_soil_process(getattr(pool, parameter), source, target, active=act)
    if name == "receive":
        return pool.receive(index, a, active=act)
    return pool.extract(index, a, active=act)


def test_a_batch_equals_its_single_pools_and_masked_instances_are_unchanged():
    b = 5
    rng = random.Random(7)
    batch, _ = make_pool((b,))
    singles = [make_pool()[0] for _ in range(b)]
    bound = {
        name: _batch_values(rng, b, lambda r: r.uniform(0.0, 0.5)) for name in NUTRIENT_POOL_PAIRS
    }
    bound["fraction_dry_n_to_dissolved_inorganic"] = torch.tensor(
        [rng.random() for _ in range(b)], dtype=F64
    )
    batch.bind(bound)
    for i, single in enumerate(singles):
        single.bind({name: value[i] for name, value in bound.items()})
    for k in range(600):
        name = rng.choice([m for m in METHODS if m != "get_erodable_P"])
        active = torch.tensor([rng.random() < 0.7 for _ in range(b)])
        before = torch.broadcast_to(batch.storage, (b, 5, 2)).clone()
        tdf = torch.tensor([_factor(rng) for _ in range(b)], dtype=F64)
        smdf = torch.tensor([_factor(rng) for _ in range(b)], dtype=F64)
        batch.temperature_dependence_factor, batch.soil_moisture_dependence_factor = tdf, smdf
        index, source, target = rng.randrange(5), *rng.sample(range(5), 2)
        parameter = rng.choice(PARAMETERS)
        amount = _batch_values(rng, b, _amount)
        scalar = torch.tensor([rng.uniform(0.0, 1.2) for _ in range(b)], dtype=F64)
        if name == "erode_P":
            erodable = batch.get_erodable_P()
            active = active & (erodable > 0)
            scalar = erodable * scalar
        elif name == "extract":
            amount = torch.broadcast_to(batch.storage[..., index, :], (b, 2)) * scalar[:, None]

        if name == "extract_dissolved":
            scalar = scalar.clamp(max=1.0)
        where = dict(parameter=parameter, index=index, source=source, target=target)
        got = _call(batch, name, amount, scalar, active, **where)
        got = got if isinstance(got, tuple) else (got,)
        for i, single in enumerate(singles):
            stored = torch.broadcast_to(batch.storage, (b, 5, 2))[i]
            if not bool(active[i]):
                assert torch.equal(stored, before[i]), f"call {k} {name}: masked {i}"
                continue
            single.temperature_dependence_factor = tdf[i]
            single.soil_moisture_dependence_factor = smdf[i]
            ref = _call(single, name, amount[i], scalar[i], TRUE, **where)
            ref = ref if isinstance(ref, tuple) else (ref,)
            for g, r in zip(got, ref, strict=True):
                if g is not None:
                    assert torch.equal(torch.broadcast_to(g, (b, *r.shape))[i], r), (
                        f"call {k} {name}"
                    )
            assert torch.equal(stored, single.storage), f"call {k} {name}: instance {i}"


def test_gradcheck_on_the_transformation_and_extract_dissolved():
    storage0 = torch.tensor(
        [[10.0, 1.0], [100.0, 10.0], [5.0, 0.5], [2.0, 0.2], [0.0, 3.0]], dtype=F64
    )
    pars0 = torch.tensor([NUTRIENT_POOL_PAIRS[n] for n in PARAMETERS], dtype=F64)
    inputs = (
        storage0.clone().requires_grad_(),
        pars0.clone().requires_grad_(),
        torch.tensor(0.7, dtype=F64, requires_grad=True),
        torch.tensor(0.9, dtype=F64, requires_grad=True),
        torch.tensor(0.3, dtype=F64, requires_grad=True),
    )

    def run(storage, pars, tdf, smdf, proportion):
        pool, _ = make_pool()
        pool.storage = storage
        pool.bind({n: pars[i] for i, n in enumerate(PARAMETERS)})
        pool.temperature_dependence_factor, pool.soil_moisture_dependence_factor = tdf, smdf
        inc_di, inc_do = pool.soil_pool_transformation(active=TRUE)
        inorganic, organic = pool.extract_dissolved(proportion, active=TRUE)
        return inc_di, inc_do, inorganic, organic, pool.storage

    assert torch.autograd.gradcheck(run, inputs)

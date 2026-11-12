"""The temperature-dependent decay (``core.py:432-521``) and ``DecayTank`` (``tanks.py:403-457``).

Decay arithmetic is ``* - **`` in WSIMOD's order with the power from Python's ``float.__pow__``,
so every comparison with live WSIMOD is bitwise.
"""

from __future__ import annotations

import pytest
import torch

from noodl.apps.wsimod.config import NodeSpec
from noodl.apps.wsimod.decay import Decay, DecaySpec, check_inert, read_decays
from noodl.apps.wsimod.engine import Ctx
from noodl.apps.wsimod.node import Node
from noodl.apps.wsimod.tanks import DecayTank
from noodl.apps.wsimod.vqip import Pollutants

P = Pollutants(
    ("phosphate", "do", "nitrate"), ("phosphate", "nitrate"), ("do",)
)  # nitrate is absent from every decays below
SPEC = DecaySpec(
    ("phosphate", "do"),
    {"phosphate": 0.1, "do": 0.05},
    {"phosphate": 1.05, "do": 1.02},
)
FA = 1e-11
SITE = "node:N/tank"


def ctx_for(shape=(), record=True):
    return Ctx(P, FA, torch.Size(shape), day=0, node_names=[], record_decisions=record)


def make(spec=SPEC, shape=(), constant=None, exponent=None, temperature=20.0, record=True):
    ctx = ctx_for(shape, record)
    decay = Decay(P, spec, site=SITE)
    constant = decay.constant if constant is None else constant
    exponent = decay.exponent if exponent is None else exponent
    decay.bind(ctx, constant, exponent, torch.as_tensor(temperature, dtype=torch.float64))
    return ctx, decay


def vq(volume, phosphate, do=0.0, nitrate=0.0):
    return P.to_vqip({"volume": volume, "phosphate": phosphate, "do": do, "nitrate": nitrate})


class Dummy:
    """What ``DecayObj`` reads the temperature from (``core.py:464-470``, ``:485-487``)."""

    def __init__(self) -> None:
        self.t = 0
        self.data_input_dict = {}


@pytest.fixture
def wsimod_core(monkeypatch):
    pytest.importorskip("wsimod")
    from wsimod.core import constants, core

    monkeypatch.setattr(constants, "POLLUTANTS", list(P.names))
    monkeypatch.setattr(constants, "ADDITIVE_POLLUTANTS", list(P.additive))
    monkeypatch.setattr(constants, "NON_ADDITIVE_POLLUTANTS", list(P.non_additive))
    monkeypatch.setattr(constants, "FLOAT_ACCURACY", FA)
    return core


def as_dict(row):
    return P.to_dict(row)


def test_generic_temperature_decay_equals_decayobj_bitwise(wsimod_core):
    g = torch.Generator().manual_seed(11)
    n = 1000
    u = lambda *s: torch.rand(*s, generator=g, dtype=torch.float64)  # noqa: E731
    vqips = torch.stack([u(n) * 5, u(n), u(n) * 9, u(n)], dim=-1) * 1.0
    zero_volume = u(n) < 0.1
    vqips[zero_volume, 0] = 0.0
    constant = torch.stack([0.001 + 0.199 * u(n), 0.001 + 0.199 * u(n), torch.zeros(n)], dim=-1)
    constant[u(n) < 0.05, 0] = 1.5
    constant[u(n) < 0.05, 1] = 1.5
    exponent = torch.stack([1 + 0.1 * u(n), 1 + 0.1 * u(n), torch.ones(n, dtype=torch.float64)], -1)
    exponent[u(n) < 0.1, 0] = 1.0
    exponent[u(n) < 0.1, 1] = 1.0
    temperature = -5 + 40 * u(n)
    ctx = ctx_for((n,))
    decay = Decay(P, SPEC, site=SITE)
    decay.bind(ctx, constant, exponent, temperature)
    decayed, diff = decay.make_decay(vqips)
    capped = 0
    for i in range(n):
        dummy = Dummy()
        decays = {
            "phosphate": {"constant": float(constant[i, 0]), "exponent": float(exponent[i, 0])},
            "do": {"constant": float(constant[i, 1]), "exponent": float(exponent[i, 1])},
        }
        obj = wsimod_core.DecayObj.__new__(wsimod_core.DecayObj)
        obj.parent = dummy
        wsimod_core.DecayObj.__init__(obj, decays)
        dummy.data_input_dict[("temperature", 0)] = float(temperature[i])
        vq_in = {"volume": float(vqips[i, 0])}
        vq_in.update({n_: float(vqips[i, 1 + j]) for j, n_ in enumerate(P.names)})
        out = obj.make_decay(dict(vq_in))
        want = [out[k] for k in ("volume", *P.names)]
        # total_decayed keeps only the additive columns of diff (core.py:104-114); the
        # non-additive diff is read from generic_temperature_decay itself
        _, d = obj.generic_temperature_decay(dict(vq_in), decays, float(temperature[i]))
        diff_want = [d[k] for k in ("volume", *P.names)]
        assert decayed[i].tolist() == want
        assert diff[i].tolist() == diff_want
        for k in ("volume", *P.additive):
            assert diff[i, 0 if k == "volume" else P.column(k)].item() == obj.total_decayed[k]
        capped += float(constant[i, 0]) * float(exponent[i, 0]) ** (float(temperature[i]) - 20) > 1
    assert capped > 10  # the cap binds in the sample


def test_the_cap_leaves_exactly_zero():
    ctx, decay = make(
        constant=torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
        exponent=torch.tensor([1.05, 1.0, 1.0], dtype=torch.float64),
        temperature=25.0,
    )
    decayed, diff = decay.make_decay(vq(1.0, 0.7))
    assert decayed[1].item() == 0.0 and torch.signbit(decayed[1]).item() is False
    assert diff[1].item() == 0.7
    (rec,) = [d for d in ctx.decisions if d.site == f"{SITE}/make_decay/cap:phosphate"]
    assert bool(rec.outcome) and rec.margin.item() == 2 * 1.05**5 - 1
    # factor exactly 1 without the cap
    _, decay = make(
        constant=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        exponent=torch.tensor([1.05, 1.0, 1.0], dtype=torch.float64),
        temperature=20.0,
    )
    t = vq(1.0, 0.3)
    decayed, _ = decay.make_decay(t)
    assert decayed[1].item() == t[1].item() - t[1].item() * 1.0 == 0.0


def test_decay_continues_below_zero_degrees():
    _, decay = make(
        constant=torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
        exponent=torch.tensor([1.05, 1.0, 1.0], dtype=torch.float64),
        temperature=-3.0,
    )
    factor = 0.1 * 1.05 ** (-3 - 20)
    assert decay.factor[1].item() == factor
    decayed, diff = decay.make_decay(vq(1.0, 4.0))
    assert diff[1].item() == 4.0 * factor
    assert decayed[1].item() == 4.0 - 4.0 * factor


def test_a_zero_volume_vqip_with_mass_decays():
    _, decay = make(temperature=10.0)
    decayed, diff = decay.make_decay(vq(0.0, 2.0))
    factor = 0.1 * 1.05 ** (10.0 - 20)
    assert decayed[1].item() == 2.0 - 2.0 * factor
    assert diff[0].item() == 0.0 and decayed[0].item() == 0.0


def test_an_absent_pollutant_is_untouched():
    _, decay = make(temperature=15.0)
    t = vq(1.5, 0.5, do=3.0, nitrate=0.37)
    decayed, diff = decay.make_decay(t)
    assert decayed[3].item() == t[3].item() and diff[3].item() == 0.0
    assert decayed[0].item() == t[0].item() and diff[0].item() == 0.0
    present = decay.present
    with pytest.raises(ValueError, match="nitrate"):
        check_inert(
            "N",
            "k",
            present,
            torch.tensor([0.1, 0.05, 0.3], dtype=torch.float64),
            torch.tensor([1.05, 1.02, 1.0], dtype=torch.float64),
            P.names,
        )
    with pytest.raises(ValueError, match="nitrate"):
        decay.bind(
            ctx_for(),
            torch.tensor([0.1, 0.05, 0.3], dtype=torch.float64),
            torch.tensor([1.05, 1.02, 1.0], dtype=torch.float64),
            torch.tensor(15.0),
        )
    check_inert("N", "k", present, decay.constant, decay.exponent, P.names)  # inert: no error


@pytest.mark.parametrize(
    ("decays", "match"),
    [
        ({"volume": {"constant": 1, "exponent": 1}}, "volume"),
        ({"nitrite": {"constant": 1, "exponent": 1}}, "nitrite"),
        ({"phosphate": {"constant": 1}}, "exponent"),
        ({"phosphate": {"constant": 1, "exponent": 1, "rate": 2}}, "rate"),
        ({"phosphate": {"constant": 1, "exponent": 0}}, "exponent"),
        ({"phosphate": {"constant": 1, "exponent": -1}}, "exponent"),
        # a non-number names where and the key, not a bare TypeError
        (
            {"phosphate": {"constant": "x", "exponent": 1}},
            r"node N: decays\['phosphate'\] constant",
        ),
        (
            {"phosphate": {"constant": None, "exponent": 1}},
            r"node N.*constant None is not a number",
        ),
        (
            {"phosphate": {"constant": 1, "exponent": "1.1"}},
            r"node N.*exponent '1.1' is not a number",
        ),
        (
            {"phosphate": {"constant": 1, "exponent": [1]}},
            r"node N.*exponent \[1\] is not a number",
        ),
    ],
)
def test_read_decays_refusals(decays, match):
    with pytest.raises(ValueError, match=match):
        read_decays("node N", decays, P)


def test_read_decays_reads_the_configuration_order():
    assert read_decays("N", None, P) is None
    spec = read_decays(
        "N",
        {"do": {"constant": 0.2, "exponent": 1.1}, "phosphate": {"constant": 1, "exponent": 1}},
        P,
    )
    assert spec.keys == ("do", "phosphate")
    assert spec.constant == {"do": 0.2, "phosphate": 1.0}
    assert read_decays("N", {}, P).keys == ()


def test_a_batched_nonpositive_exponent_is_refused_at_bind():
    decay = Decay(P, SPEC, site=SITE)
    exponent = torch.tensor([[1.05, 1.02, 1.0], [1.05, 0.0, 1.0]])
    with pytest.raises(ValueError, match="do"):
        decay.bind(
            ctx_for((2,)),
            torch.tensor([0.1, 0.05, 0.0], dtype=torch.float64).expand(2, 3),
            exponent.double(),
            torch.tensor([10.0, 10.0], dtype=torch.float64),
        )
    with pytest.raises(RuntimeError, match="bind"):
        decay.make_decay(vq(1.0, 1.0))


def test_a_decay_with_no_keys_changes_nothing():
    spec = read_decays("N", {}, P)
    ctx = ctx_for()
    decay = Decay(P, spec, site=SITE)
    decay.bind(ctx, decay.constant, decay.exponent, torch.tensor(5.0))
    t = vq(1.0, 2.0, 3.0, 4.0)
    decayed, diff = decay.make_decay(t)
    assert decayed.tolist() == t.tolist() and diff.tolist() == [0.0] * 4
    assert not ctx.decisions


def decay_tank(decay, initial, parent=None):
    parent = parent or Node(NodeSpec("N", "Node", "Node", {}), P)
    return DecayTank(
        P,
        parent=parent,
        decay=decay,
        capacity=100.0,
        area=1.0,
        datum=0.0,
        initial_storage=initial,
        owner="N",
    )


def test_decay_tank_with_a_decay_stores_decays_and_records_the_term():
    ctx, decay = make(temperature=10.0)
    tank = decay_tank(decay, vq(2.0, 1.0, 5.0, 0.4))
    tank.ctx = ctx
    tank.end_timestep()
    factor_p = 0.1 * 1.05 ** (10.0 - 20)
    factor_d = 0.05 * 1.02 ** (10.0 - 20)
    assert tank.storage[1].item() == 1.0 - 1.0 * factor_p
    assert tank.storage[2].item() == 5.0 - 5.0 * factor_d
    assert tank.storage[3].item() == 0.4
    assert tank.total_decayed.tolist()[:2] == [0.0, 1.0 * factor_p]
    assert ctx.terms["DecayTank.decayed"][1].item() == 1.0 * factor_p
    n = len(ctx.decisions)
    tank.end_timestep()  # the decay of the second end_timestep takes no decision (D5)
    assert len(ctx.decisions) == n and not ctx.sequence
    tank2 = decay_tank(None, vq(1.0, 1.0))
    tank2.ctx = ctx
    with pytest.raises(KeyError, match="temperature"):
        tank2.end_timestep()


def test_decay_tank_equals_wsimods_on_random_days(wsimod_core):
    pytest.importorskip("wsimod")
    from wsimod.nodes import tanks as wtanks

    g = torch.Generator().manual_seed(23)
    u = lambda: float(torch.rand((), generator=g, dtype=torch.float64))  # noqa: E731
    decays = {
        "phosphate": {"constant": 0.12, "exponent": 1.07},
        "do": {"constant": 0.4, "exponent": 1.03},
    }
    dummy = Dummy()
    theirs = wtanks.DecayTank(
        decays=decays,
        parent=dummy,
        capacity=20.0,
        area=1.0,
        datum=0.0,
        initial_storage={"volume": 3.0, "phosphate": 1.0, "do": 8.0, "nitrate": 0.2},
    )
    spec = read_decays("N", decays, P)
    decay = Decay(P, spec, site=SITE)
    ours = decay_tank(
        decay, P.to_vqip({"volume": 3.0, "phosphate": 1.0, "do": 8.0, "nitrate": 0.2})
    )
    for day in range(60):
        ctx = Ctx(P, FA, torch.Size(()), day=day, node_names=[], record_decisions=True)
        ours.ctx = ctx
        temp = -5 + 40 * u()
        decay.bind(ctx, decay.constant, decay.exponent, torch.tensor(temp, dtype=torch.float64))
        dummy.t = day
        dummy.data_input_dict[("temperature", day)] = temp
        push = {"volume": u() * 4, "phosphate": u(), "do": u() * 9, "nitrate": u()}
        theirs.push_storage(dict(push), force=True)
        ours.push_storage(P.to_vqip(push), active=torch.tensor(True), force=True)
        want_pull = {"volume": u() * 3, "phosphate": 0.0, "do": 0.0, "nitrate": 0.0}
        theirs.pull_storage(dict(want_pull))
        ours.pull_storage(P.to_vqip(want_pull), active=torch.tensor(True))
        theirs.end_timestep()
        ours.end_timestep()
        for k in ("volume", *P.names):
            col = 0 if k == "volume" else P.column(k)
            assert ours.storage[col].item() == theirs.storage[k], (day, k)
            assert ours.total_decayed[col].item() == theirs.total_decayed[k], (day, k)


def test_gradcheck_on_the_decay_function():
    ctx = ctx_for(record=False)
    decay = Decay(P, SPEC, site=SITE)
    c = torch.tensor([0.1, 0.05], dtype=torch.float64, requires_grad=True)
    e = torch.tensor([1.05, 1.02], dtype=torch.float64, requires_grad=True)
    temperature = torch.tensor(12.0, dtype=torch.float64, requires_grad=True)
    t = vq(1.0, 2.0, 3.0, 4.0).clone().requires_grad_(True)

    def f(c, e, temperature, t):
        # the absent key's entries are inert (0, 1), not variables
        decay.bind(ctx, torch.cat([c, c.new_zeros(1)]), torch.cat([e, e.new_ones(1)]), temperature)
        decayed, diff = decay.make_decay(t)
        return decayed, diff

    assert torch.autograd.gradcheck(f, (c, e, temperature, t), atol=1e-6)

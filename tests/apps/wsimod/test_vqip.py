"""VQIP algebra and decision primitives against WSIMOD's WSIObj (core.py)."""

import pytest
import torch

from noodl.apps.wsimod.vqip import (
    QUANTITY_ONLY,
    Pollutants,
    ds_vqip,
    empty,
    extract_vqip,
    safe_den,
    safe_sqrt,
    sum_vqip,
    tmax,
    tmin,
    v_change_vqip,
    v_distill_vqip,
    write,
)

P = Pollutants(("phosphate", "temperature"), ("phosphate",), ("temperature",))


def f64(x):
    return torch.tensor(x, dtype=torch.float64)


def vq(v, ph, t):
    return P.to_vqip({"volume": v, "phosphate": ph, "temperature": t})


def test_vqip_examples_from_wsimods_docstrings():
    t1, t2 = vq(100, 0.25, 10), vq(10, 0.25, 15)
    s = P.to_dict(sum_vqip(P, t1, t2))
    assert s == {"volume": 110.0, "phosphate": 0.5, "temperature": (15 * 10 + 10 * 100) / 110}
    assert P.to_dict(extract_vqip(P, t1, t2)) == {
        "volume": 90.0,
        "phosphate": 0.0,
        "temperature": 10.0,
    }
    c = P.to_dict(v_change_vqip(P, vq(100, 0.25, 0), f64(10.0)))
    assert c["volume"] == 10.0 and c["phosphate"] == pytest.approx(0.025)
    d = P.to_dict(ds_vqip(P, t1, vq(90, 0.2, 9)))
    assert d == {"volume": 10.0, "phosphate": 0.25 - 0.2, "temperature": 0.0}


def test_v_change_scales_by_the_ratio_not_to_the_target():
    out = v_change_vqip(P, vq(49, 1.0, 5), f64(1.0))
    assert out[0].item() == 49 * (1 / 49) == 0.9999999999999999
    out = v_change_vqip(P, vq(0, 1.0, 5), f64(3.0))
    assert P.to_dict(out) == {"volume": 3.0, "phosphate": 1.0, "temperature": 5.0}


@pytest.fixture
def wsiobj(monkeypatch):
    pytest.importorskip("wsimod")
    from wsimod.core import constants
    from wsimod.core.core import WSIObj

    names = ["phosphate", "nitrate", "temperature", "do"]
    monkeypatch.setattr(constants, "POLLUTANTS", names)
    monkeypatch.setattr(constants, "ADDITIVE_POLLUTANTS", ["phosphate", "nitrate"])
    monkeypatch.setattr(constants, "NON_ADDITIVE_POLLUTANTS", ["temperature", "do"])
    return WSIObj(), names


def test_vqip_operations_equal_wsiobj_bitwise_on_random_inputs(wsiobj):
    obj, names = wsiobj
    p = Pollutants(tuple(names), ("phosphate", "nitrate"), ("temperature", "do"))
    g = torch.Generator().manual_seed(0)
    n = 1000
    a = torch.rand(n, 5, generator=g, dtype=torch.float64) * 100
    b = torch.rand(n, 5, generator=g, dtype=torch.float64) * 100
    a[torch.rand(n, generator=g) < 0.10, 0] = 0.0
    b[torch.rand(n, generator=g) < 0.10, 0] = 0.0
    both = torch.rand(n, generator=g) < 0.05
    a[both, 0] = 0.0
    b[both, 0] = 0.0
    # negative volumes: "V > 0 is false" keeps t1 (core.py:107, :270)
    a = torch.cat([a, -torch.rand(20, 5, generator=g, dtype=torch.float64) * 10])
    b = torch.cat([b, torch.rand(20, 5, generator=g, dtype=torch.float64) * 5])
    a[-10:, 0] = a[-10:, 0].abs()
    b[-10:, 0] = -b[-10:, 0]
    n = len(a)
    assert (a[:, 0] == 0).any() and ((a[:, 0] + b[:, 0]) == 0).any()
    assert (a[:, 0] < 0).any() and ((a[:, 0] + b[:, 0]) < 0).any()
    v = torch.rand(n, generator=g, dtype=torch.float64) * 50

    def d(row):
        return p.to_dict(row)

    def stack(dicts):
        return torch.stack([p.to_vqip(x) for x in dicts])

    ops = {
        "sum": (sum_vqip(p, a, b), [obj.sum_vqip(d(x), d(y)) for x, y in zip(a, b, strict=True)]),
        "extract": (
            extract_vqip(p, a, b),
            [obj.extract_vqip(d(x), d(y)) for x, y in zip(a, b, strict=True)],
        ),
        "v_change": (
            v_change_vqip(p, a, v),
            [obj.v_change_vqip(d(x), float(w)) for x, w in zip(a, v, strict=True)],
        ),
        "v_distill": (
            v_distill_vqip(a, v),
            [obj.v_distill_vqip(d(x), float(w)) for x, w in zip(a, v, strict=True)],
        ),
        "ds": (ds_vqip(p, a, b), [obj.ds_vqip(d(x), d(y)) for x, y in zip(a, b, strict=True)]),
    }
    for name, (got, ref) in ops.items():
        assert torch.equal(got, stack(ref)), name


def test_quantity_only_vqips_have_one_column():
    a = torch.tensor([[3.0], [0.0]], dtype=torch.float64)
    b = torch.tensor([[2.0], [0.0]], dtype=torch.float64)
    one = torch.tensor([1.0, 4.0], dtype=torch.float64)
    assert sum_vqip(QUANTITY_ONLY, a, b).tolist() == [[5.0], [0.0]]
    assert extract_vqip(QUANTITY_ONLY, a, b).tolist() == [[1.0], [0.0]]
    assert ds_vqip(QUANTITY_ONLY, a, b).tolist() == [[1.0], [0.0]]
    assert v_change_vqip(QUANTITY_ONLY, a, one).tolist() == [[1.0], [4.0]]
    assert v_distill_vqip(a, one).tolist() == [[2.0], [-4.0]]
    assert empty(QUANTITY_ONLY, (2,)).shape == (2, 1)


@pytest.mark.parametrize(
    ("args", "name"),
    [
        ((("phosphate",), ("phosphate",), ("phosphate",)), "phosphate"),
        ((("phosphate",), (), ()), "phosphate"),
        ((("phosphate",), ("phosphate", "nitrate"), ()), "nitrate"),
        ((("phosphate",), ("phosphate",), ("nitrate",)), "nitrate"),
    ],
)
def test_inconsistent_pollutant_lists_are_refused_by_name(args, name):
    with pytest.raises(ValueError, match=name):
        Pollutants(*args)


def grad_pair(f, a, b):
    a = f64(a).requires_grad_()
    b = f64(b).requires_grad_()
    out = f(a, b)
    out.backward()
    return out.item(), a.grad.item(), b.grad.item()


def test_ties_follow_pythons_min_and_max():
    assert grad_pair(tmin, 2.0, 2.0) == (2.0, 1.0, 0.0)
    assert grad_pair(tmax, 2.0, 2.0) == (2.0, 1.0, 0.0)
    assert grad_pair(tmin, 3.0, 2.0) == (2.0, 0.0, 1.0)
    assert grad_pair(tmax, 1.0, 2.0) == (2.0, 0.0, 1.0)


def test_guarded_division_and_root_keep_gradients_finite():
    n = f64(3.0).requires_grad_()
    d = f64(0.0).requires_grad_()
    out = torch.where(d > 0, n / safe_den(d), torch.zeros_like(n))
    out.backward()
    assert out.item() == 0.0 and torch.isfinite(n.grad) and torch.isfinite(d.grad)
    x = f64(0.0).requires_grad_()
    safe_sqrt(x).backward()
    assert x.grad.item() == 0.0
    x = f64(4.0).requires_grad_()
    y = safe_sqrt(x)
    y.backward()
    assert y.item() == 2.0 and x.grad.item() == 0.25


def test_write_broadcasts_over_the_column_axis():
    active = torch.tensor([True, False])
    new, old = torch.ones(2, 3, dtype=torch.float64), torch.zeros(2, 3, dtype=torch.float64)
    assert write(active, new, old).tolist() == [[1.0] * 3, [0.0] * 3]
    assert write(active, new[:, 0], old[:, 0]).tolist() == [1.0, 0.0]


def test_vqip_operations_pass_gradcheck():
    g = torch.Generator().manual_seed(1)

    def mk(*shape):
        return (torch.rand(*shape, generator=g, dtype=torch.float64) + 0.5).requires_grad_()

    a, b, v = mk(4, 3), mk(4, 3), mk(4)
    assert torch.autograd.gradcheck(lambda x, y: sum_vqip(P, x, y), (a, b))
    assert torch.autograd.gradcheck(lambda x, y: extract_vqip(P, x, y), (a, b))
    assert torch.autograd.gradcheck(lambda x, y: ds_vqip(P, x, y), (a, b))
    assert torch.autograd.gradcheck(lambda x, w: v_change_vqip(P, x, w), (a, v))
    assert torch.autograd.gradcheck(v_distill_vqip, (a, v))


def test_zero_volume_gradients_are_finite():
    t1, t2 = vq(0, 0.5, 4), vq(0, 0.2, 6)
    t1.requires_grad_()
    t2.requires_grad_()
    v = f64(2.0).requires_grad_()
    (sum_vqip(P, t1, t2).sum() + v_change_vqip(P, t1, v).sum()).backward()
    for x in (t1.grad, t2.grad, v.grad):
        assert torch.isfinite(x).all()


def test_pollutants_accepts_lists():
    p = Pollutants(["phosphate", "temperature"], ["phosphate"], ["temperature"])
    assert p.names == ("phosphate", "temperature") and p.additive == ("phosphate",)
    assert p == P and p.volume_or_additive_mask.tolist() == [True, True, False]

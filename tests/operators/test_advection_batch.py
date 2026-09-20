"""P2-5: an ensemble batched in exactly ONE coefficient family must step like the
per-instance unbatched layers, under every scheme."""

from __future__ import annotations

import pytest
import torch

from noodl.layers.transport import TransportLayer
from noodl.topology import Network

F64 = torch.float64
FAMILIES = ("conductance", "removal", "kinetics", "transmission", "capacity", "flow")


def _net():
    net = Network(dtype=F64)
    for n in ("ambient", "a", "b", "c"):
        net.add_node(n)
    for u, v in (("ambient", "a"), ("a", "b"), ("b", "c"), ("c", "ambient")):
        net.add_edge(u, v, kind="flow")
    net.add_edge("a", "c", kind="wall")
    return net


def _base(seed=0):
    g = torch.Generator().manual_seed(seed)
    K = 2
    return {
        "conductance": torch.rand(1, generator=g, dtype=F64) + 0.1,           # (n_c,)
        "removal": torch.rand(3, K, generator=g, dtype=F64),                   # (n_i, K)
        "kinetics": 0.3 * torch.randn(3, K, K, generator=g, dtype=F64),        # (n_i, K, K)
        "transmission": torch.rand(4, K, generator=g, dtype=F64) + 0.5,        # (b, K)
        "capacity": torch.rand(3, generator=g, dtype=F64) + 0.5,               # (n_i,)
        "flow": torch.randn(4, generator=g, dtype=F64),                        # (b,)
    }


def _layer(coef, scheme):
    return TransportLayer(
        _net(), "c", capacity=coef["capacity"], flow_kind="flow", boundary=["ambient"],
        n_species=2, transmission=coef["transmission"], conduction_kind="wall",
        conductance=coef["conductance"], removal=coef["removal"], kinetics=coef["kinetics"],
        scheme=scheme,
    )


def _step(layer, q):
    x0 = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=F64)
    xb = torch.tensor([[0.2, 0.1]], dtype=F64)
    sources = torch.zeros(4, 2, dtype=F64)
    return layer.step(x0, q, sources, xb, 0.7)


# Run against 852c65d before marking (see the module docstring's cross-reference): per
# (family, scheme), NOT per family as first assumed -- `removal` fails only under
# "implicit" (exact and trapezoidal already pass) and `transmission` fails only under
# "exact" (implicit and trapezoidal already pass), alongside `conductance` and `kinetics`
# which fail under all three schemes. Mark exactly these eight (family, scheme) pairs.
_FAILING = {
    ("conductance", "exact"), ("conductance", "implicit"), ("conductance", "trapezoidal"),
    ("removal", "implicit"),
    ("kinetics", "exact"), ("kinetics", "implicit"), ("kinetics", "trapezoidal"),
    ("transmission", "exact"),
}


def _case(family, scheme):
    if (family, scheme) in _FAILING:
        return pytest.param(family, scheme, id=f"{family}-{scheme}", marks=pytest.mark.xfail(
            strict=True,
            reason=f"P2-5: {family} is missing from the operator batch shape under "
            f"scheme={scheme!r}",
        ))
    return pytest.param(family, scheme, id=f"{family}-{scheme}")


@pytest.mark.parametrize(
    "family, scheme",
    [_case(family, scheme) for family in FAMILIES
     for scheme in ("exact", "implicit", "trapezoidal")],
)
def test_an_ensemble_batched_in_one_family_matches_the_per_instance_layers(family, scheme):
    base = _base()
    alt = _base(seed=1)
    batched = dict(base)
    batched[family] = torch.stack([base[family], alt[family]])       # (2, ...)
    per_instance = [dict(base), dict(base)]
    per_instance[1][family] = alt[family]
    q_b = batched.pop("flow") if family == "flow" else base["flow"]
    got = _step(_layer(batched if family != "flow" else base, scheme), q_b)
    assert got.shape == (2, 3, 2)
    for i in range(2):
        coef = per_instance[i]
        q_i = coef.pop("flow")
        want = _step(_layer(coef, scheme), q_i)
        torch.testing.assert_close(got[i], want, rtol=1e-10, atol=1e-12)

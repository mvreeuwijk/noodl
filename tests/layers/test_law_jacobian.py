"""`ConstitutiveLaw.jacobian`: the default blocks, and the operator's use of an override."""

import torch

from noodl.elements import Conductance, ConstitutiveLaw
from noodl.elements._slopes import _colour_blocks, complex_slope_block
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.topology import Network
from tests._circuit_laws import Source
from tests.layers.test_mna import Ratio

F64 = torch.float64
C128 = torch.complex128


class Cubic(ConstitutiveLaw):
    """q = k dp^3, a one-port nonlinear law."""

    def __init__(self, k, kind="r"):
        super().__init__(kind)
        self.k = self._param(k, False)

    def flow(self, dp, drivers=None):
        return self.k * dp ** 3


class ConstantPower(ConstitutiveLaw):
    """q = conj(S / dp): a phasor law, not holomorphic in dp."""

    phasor_capable = True

    def __init__(self, S, kind="load"):
        super().__init__(kind)
        self.S = self._param(S, False)

    def flow(self, dp, drivers=None):
        return torch.conj(self.S / dp)


def test_one_port_default_matches_autograd_reference():
    k = torch.tensor([1.5, 2.0, 0.5], dtype=F64)
    law = Cubic(k)
    dp = torch.tensor([0.3, -1.2, 2.0], dtype=F64)
    block, = law.jacobian(dp, None, {}, None)
    ref = torch.stack([torch.autograd.functional.jacobian(
        lambda d, i=i: law(d, {}, None)[i], dp)[i] for i in range(3)])
    assert block.shape == (3, 1, 1, 1, 1)
    assert torch.equal(block.reshape(3), ref)


def test_analytic_dflow_matches_colouring_to_rounding():
    law = Conductance(torch.tensor([1.5, 2.0, 0.5], dtype=F64))
    dp = torch.tensor([0.3, -1.2, 2.0], dtype=F64)
    block, = law.jacobian(dp, None, {}, None)
    ref, = _colour_blocks(lambda d: law(d, {}, None), (dp,), 1, 1)
    torch.testing.assert_close(block, ref, rtol=1e-14, atol=0)


class DrivenCubic(ConstitutiveLaw):
    """q = drivers["k"] dp^3."""

    def __init__(self, kind="r"):
        super().__init__(kind)

    def flow(self, dp, drivers=None):
        return drivers["k"] * dp ** 3


def test_driver_batch_wider_than_dp_gives_per_instance_blocks():
    law = DrivenCubic()
    dp = torch.tensor([0.5, 1.0], dtype=F64)
    drivers = {"k": torch.tensor([[1.0], [2.0], [3.0]], dtype=F64)}
    block, = law.jacobian(dp, None, drivers, None)
    ref, = _colour_blocks(lambda d: law(d, drivers, None), (dp,), 1, 1)
    assert block.shape == (3, 2, 1, 1, 1, 1)
    assert torch.equal(block, ref)
    expected = drivers["k"] * 3 * dp ** 2
    torch.testing.assert_close(block.reshape(3, 2), expected)


def test_two_port_implicit_default_equals_colouring():
    law = Ratio(torch.tensor(2.0, dtype=F64))
    dp = torch.tensor([1.0, 0.4, -0.7, 2.0], dtype=F64)
    q = torch.tensor([0.2, -0.1, 0.9, 0.3], dtype=F64)
    blocks = law.jacobian(dp, q, {}, None)
    ref = _colour_blocks(lambda d, qq: law(d, qq, {}, None), (dp, q), 2, 1)
    assert len(blocks) == 2 and blocks[0].shape == (2, 2, 2, 1, 1)
    for b, r in zip(blocks, ref, strict=True):
        assert torch.equal(b, r)


def test_phasor_default_equals_colouring():
    law = ConstantPower(torch.tensor([0.5 + 0.2j, 0.1 - 0.3j], dtype=C128))
    dp = torch.tensor([1.0 + 0.1j, 0.9 - 0.2j], dtype=C128)
    block, = law.jacobian(dp, None, {}, None)
    ref, = _colour_blocks(lambda d: law(d, {}, None), (dp,), 1, 2, C128)
    assert block.shape == (2, 1, 1, 2, 2)
    assert torch.equal(block, ref)


def test_complex_slope_block_is_the_real_view_of_multiplication():
    c = torch.tensor([2.0 + 3.0j, -1.0 + 0.5j], dtype=C128)
    block = complex_slope_block(c)
    z = torch.tensor([0.7 - 0.2j, 1.1 + 0.4j], dtype=C128)
    zr = torch.view_as_real(z)
    got = (block @ zr.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(torch.view_as_complex(got.contiguous()), c * z)
    assert torch.equal(block[0], torch.tensor([[2.0, -3.0], [3.0, 2.0]], dtype=F64))


def test_operator_uses_a_law_override():
    calls = []

    class Spied(Conductance):
        def jacobian(self, dp, q, drivers, ctx):
            calls.append(dp.shape)
            return super().jacobian(dp, q, drivers, ctx)

    net = Network(dtype=F64)
    for n in ("gnd", "a"):
        net.add_node(n)
    net.add_edge("a", "gnd", kind="source")
    net.add_edge("a", "gnd", kind="r")
    layer = KirchhoffLayer(net, "e", [Source(torch.tensor([3.0])),
                                      Spied(torch.tensor([2.0]), kind="r")],
                           boundary=["gnd"])
    assert layer.mna
    phi, q = layer.solve(torch.zeros(1), differentiable=False)
    assert calls
    torch.testing.assert_close(q[1].to(F64), torch.tensor(6.0, dtype=F64))

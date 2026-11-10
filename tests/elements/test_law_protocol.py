"""The constitutive-law protocol: forms, ports, context, implicit laws."""

import dataclasses

import pytest
import torch

from noodl.elements import (
    Conductance,
    FixedFlow,
    ImplicitLaw,
    LawContext,
    RegularizedPowerLaw,
    SplineFlowTable,
)
from noodl.elements.base import MODES, SCHEMES, ConstitutiveLaw


@pytest.fixture(autouse=True)
def _float64_default():
    """float64 for these tests, restored afterwards so other modules see the default dtype."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_defaults_declare_a_static_potential_controlled_law():
    g = Conductance(2.0)
    assert (isinstance(g, ImplicitLaw), g.ports, g.dynamic, g._takes_ctx) == (
        False, 1, False, False)
    assert g.phasor_capable is True
    assert FixedFlow(1.0).phasor_capable is True
    assert ConstitutiveLaw.phasor_capable is False


def test_forward_passes_context_only_to_laws_whose_flow_takes_ctx():
    class Seen(ConstitutiveLaw):
        def flow(self, dp, drivers=None, ctx=None):
            return dp * (0.0 if ctx is None else ctx.dt)

    ctx = LawContext(mode="transient", dt=0.5)
    assert Seen._takes_ctx
    assert torch.equal(Seen("k")(torch.tensor([2.0]), {}, ctx), torch.tensor([1.0]))
    # a law whose flow has no ctx keeps its two-argument flow
    assert torch.equal(Conductance(3.0)(torch.tensor([2.0]), {}, ctx), torch.tensor([6.0]))


def test_a_subclass_overriding_flow_without_ctx_stops_receiving_it():
    class Seen(ConstitutiveLaw):
        def flow(self, dp, drivers=None, ctx=None):
            return dp * ctx.dt

    class Plain(Seen):
        def flow(self, dp, drivers=None):
            return 2.0 * dp

    ctx = LawContext(mode="transient", dt=0.5)
    assert Seen._takes_ctx and not Plain._takes_ctx
    assert torch.equal(Plain("k")(torch.tensor([2.0]), {}, ctx), torch.tensor([4.0]))


def test_functional_call_round_trip_with_and_without_ctx():
    class Seen(ConstitutiveLaw):
        def __init__(self):
            super().__init__("k")
            self.g = torch.nn.Parameter(torch.tensor(2.0))

        def flow(self, dp, drivers=None, ctx=None):
            return self.g * dp * ctx.dt

    ctx = LawContext(mode="transient", dt=0.5)
    dp = torch.tensor([3.0])
    out = torch.func.functional_call(Seen(), {"g": torch.tensor(4.0)}, (dp, {}, ctx))
    assert torch.equal(out, torch.tensor([6.0]))
    plain = Conductance(2.0)
    out = torch.func.functional_call(plain, dict(plain.named_parameters()), (dp, {}, ctx))
    assert torch.equal(out, plain(dp, {}, ctx))


def test_residual_is_q_minus_flow_on_every_law():
    from noodl.apps.electrical import Resistance
    from noodl.elements import PowerLaw
    dp = torch.tensor([0.4, -1.3, 2.0])
    q = torch.tensor([0.1, 0.2, -0.3])
    for law in (PowerLaw(1.0, 0.5), Resistance(2.0)):
        assert torch.equal(law.residual(dp, q, {}, LawContext(mode="dc")), q - law(dp, {}))


def test_existing_form_attributes_do_not_make_a_relation():
    """Laws with their own `form` (volume or mass) stay potential-controlled."""
    power = RegularizedPowerLaw(1.0, 0.5, form="volume", rho_default=1.2)
    table = SplineFlowTable(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), form="mass")
    assert (isinstance(power, ImplicitLaw), power.form) == (False, "volume")
    assert (isinstance(table, ImplicitLaw), table.form) == (False, "mass")


def test_implicit_law_routes_forward_to_residual_and_has_no_flow():
    class Source(ImplicitLaw):
        def residual(self, dp, q, drivers, ctx):
            return dp - 5.0

    law = Source("vs")
    r = law(torch.tensor([5.0]), torch.tensor([0.3]), {}, LawContext(mode="dc"))
    assert torch.equal(r, torch.tensor([0.0]))
    assert law._takes_ctx
    with pytest.raises(TypeError, match="relation"):
        law.flow(torch.tensor([1.0]))


def test_context_is_frozen_and_modes_are_listed():
    ctx = LawContext(mode="dc")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.dt = 1.0
    assert MODES == ("dc", "initial", "transient", "phasor")
    assert SCHEMES == ("exact", "backward_euler", "trapezoidal", "forward_euler")


def test_removed_law_form_attribute_is_refused_naming_the_replacement():
    with pytest.raises(TypeError, match=r"law_form.*subclasses ImplicitLaw"):
        class Odd(Conductance):
            law_form = "implicit"


def test_removed_contextual_attribute_is_refused_naming_the_replacement():
    with pytest.raises(TypeError, match=r"contextual.*`ctx` parameter in `flow`"):
        class Odd(ConstitutiveLaw):
            contextual = True

            def flow(self, dp, drivers=None, ctx=None):
                return dp

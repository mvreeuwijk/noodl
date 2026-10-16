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
from noodl.elements.base import INTEGRATIONS, MODES, ConstitutiveLaw


@pytest.fixture(autouse=True)
def _float64_default():
    """float64 for these tests, restored afterwards so other modules see the default dtype."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_defaults_declare_a_static_potential_controlled_law():
    g = Conductance(2.0)
    assert (g.law_form, g.ports, g.dynamic, g.contextual) == ("potential", 1, False, False)
    assert g.phasor_capable is True
    assert FixedFlow(1.0).phasor_capable is True
    assert ConstitutiveLaw.phasor_capable is False


def test_forward_passes_context_only_to_contextual_laws():
    class Seen(ConstitutiveLaw):
        contextual = True

        def flow(self, dp, drivers=None, ctx=None):
            return dp * (0.0 if ctx is None else ctx.dt)

    ctx = LawContext(mode="transient", dt=0.5)
    assert torch.equal(Seen("k")(torch.tensor([2.0]), {}, ctx), torch.tensor([1.0]))
    # a non-contextual law ignores ctx and keeps its two-argument flow
    assert torch.equal(Conductance(3.0)(torch.tensor([2.0]), {}, ctx), torch.tensor([6.0]))


def test_existing_form_attributes_are_not_the_law_form():
    """Laws with their own `form` (volume or mass) stay potential-controlled."""
    power = RegularizedPowerLaw(1.0, 0.5, form="volume", rho_default=1.2)
    table = SplineFlowTable(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), form="mass")
    assert (power.law_form, power.form) == ("potential", "volume")
    assert (table.law_form, table.form) == ("potential", "mass")


def test_implicit_law_routes_forward_to_residual_and_has_no_flow():
    class Source(ImplicitLaw):
        def residual(self, dp, q, drivers, ctx):
            return dp - 5.0

    law = Source("vs")
    r = law(torch.tensor([5.0]), torch.tensor([0.3]), {}, LawContext(mode="dc"))
    assert torch.equal(r, torch.tensor([0.0]))
    assert law.law_form == "implicit" and law.contextual
    with pytest.raises(TypeError, match="implicit"):
        law.flow(torch.tensor([1.0]))


def test_context_is_frozen_and_modes_are_listed():
    ctx = LawContext(mode="dc")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.dt = 1.0
    assert MODES == ("dc", "initial", "transient", "phasor")
    assert INTEGRATIONS == ("trapezoidal", "backward_euler")

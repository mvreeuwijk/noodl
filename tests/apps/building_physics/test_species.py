"""Species records and first-order kinetic reactions."""

from __future__ import annotations

import pytest
import torch

from noodl.apps.building_physics.species import KineticReaction, Species, reaction_kinetics
from noodl.apps.building_physics.thermal import species_layer
from noodl.topology import Network

F64 = torch.float64


def _net():
    net = Network(dtype=F64)
    net.add_node("ambient", volume=0.0)
    net.add_node("one", volume=300.0)
    net.add_node("two", volume=600.0)
    net.add_node("store", volume=10.0)                 # no airflow edge: inactive
    net.add_edge("ambient", "one", kind="airpath")
    net.add_edge("one", "two", kind="airpath")
    return net


FAST = KineticReaction({("sarin", "sarin"): -1e-3, ("product", "sarin"): 1e-3,
                        ("product", "product"): -5e-4}, name="fast")


def test_a_species_record_says_whether_it_is_a_particle():
    assert not Species("sarin", molar_mass=140.09).is_particle
    assert Species("pm", mean_diameter=2e-6, effective_density=1000.0).is_particle


def test_the_reaction_matrix_is_product_by_reactant():
    m = FAST.matrix(["sarin", "product"])
    torch.testing.assert_close(m, torch.tensor([[-1e-3, 0.0], [1e-3, -5e-4]], dtype=F64))
    assert m.dtype == F64
    m3 = FAST.matrix(["other", "product", "sarin"])
    assert m3[1, 2].item() == 1e-3 and m3[0].abs().sum().item() == 0.0


def test_a_reaction_naming_an_unknown_species_is_refused():
    with pytest.raises(KeyError, match=r"'fast'.*\['product'\]"):
        FAST.matrix(["sarin"])


def test_reaction_coefficients_carry_gradients():
    k = torch.tensor(-1e-3, dtype=F64, requires_grad=True)
    m = KineticReaction({("sarin", "sarin"): k}).matrix(["sarin"])
    (g,) = torch.autograd.grad(m.sum(), k)
    assert g.item() == 1.0


def test_reaction_kinetics_places_each_zones_matrix_in_interior_order():
    kin = reaction_kinetics(_net(), species=["sarin", "product"], reactions={"two": FAST})
    assert kin.shape == (2, 2, 2)                      # interior: one, two
    assert kin[0].abs().sum().item() == 0.0
    torch.testing.assert_close(kin[1], FAST.matrix(["sarin", "product"]))


def test_a_reaction_on_a_zone_the_layer_does_not_solve_is_refused():
    with pytest.raises(KeyError, match=r"'store'.*not an active interior node"):
        reaction_kinetics(_net(), species=["sarin", "product"], reactions={"store": FAST})


def test_the_species_layer_integrates_a_reaction_implicitly():
    """A sealed zone (no flow), one 300 s implicit step from (1e-3, 0): x1 = (I - dt K)^-1
    x0 = (1e-3 / 1.3, 0.3 (1e-3 / 1.3) / 1.15). ContamX gives (7.692307973e-4,
    2.006689119e-4) for the same reaction, from float32 inputs."""
    net = _net()
    kin = reaction_kinetics(net, species=["sarin", "product"], reactions={"one": FAST})
    layer = species_layer(net, n_species=2, species_names=["sarin", "product"],
                          kinetics=kin)
    x0 = torch.tensor([[1e-3, 0.0], [0.0, 0.0]], dtype=F64)
    x1 = layer.step(x0, torch.zeros(2, dtype=F64), torch.zeros(4, 2, dtype=F64),
                    torch.zeros(1, 2, dtype=F64), 300.0)
    assert x1[0, 0].item() == pytest.approx(1e-3 / 1.3, rel=1e-13)
    assert x1[0, 1].item() == pytest.approx(0.3 * 1e-3 / 1.3 / 1.15, rel=1e-13)
    assert x1[0, 1].item() == pytest.approx(2.006689119e-4, rel=1e-7)

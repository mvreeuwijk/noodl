"""CONTAM contaminant source types (TN 1887r1 section 8.2.4) as source-term callables."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from tellegen.apps.building.prj import PrjSource
from tellegen.apps.building.sources import (
    BurstSource,
    ConstantSource,
    CutoffSource,
    DecayingSource,
    assemble_sources,
    sources_from_project,
)
from tellegen.topology import Network

F64 = torch.float64


def test_constant_source_is_generation_minus_removal_times_concentration():
    s = ConstantSource(node=1, G=2e-6, R=1e-3)
    x = torch.tensor([0.0, 5e-4, 0.0], dtype=F64)
    out = s(0.0, x)
    assert out.shape == (3,)
    assert out[1].item() == pytest.approx(2e-6 - 1e-3 * 5e-4)
    assert out[0].item() == 0.0 and out[2].item() == 0.0


def test_cutoff_source_vanishes_at_the_cutoff_concentration():
    s = CutoffSource(node=2, G=1e-5, x_cut=1e-3)
    x = torch.tensor([0.0, 0.0, 1e-3], dtype=F64)
    assert s(0.0, x)[2].item() == pytest.approx(0.0)
    x[2] = 5e-4
    assert s(0.0, x)[2].item() == pytest.approx(5e-6)


def test_cutoff_source_clamps_at_zero_above_the_cutoff_instead_of_reversing_sign():
    # Controller ruling R31: CONTAM's cutoff source shuts generation off past x_cut; the
    # bare eq. 17 formula would go negative (a sink) there, which is not the model.
    s = CutoffSource(node=2, G=1e-5, x_cut=1e-3)
    x = torch.tensor([0.0, 0.0, 2e-3], dtype=F64)  # 2x the cutoff
    assert s(0.0, x)[2].item() == 0.0


def test_cutoff_source_below_cutoff_still_matches_the_unclamped_formula():
    s = CutoffSource(node=2, G=1e-5, x_cut=1e-3)
    x = torch.tensor([0.0, 0.0, 9e-4], dtype=F64)  # just below x_cut
    expected = 1e-5 * (1.0 - 9e-4 / 1e-3)
    assert s(0.0, x)[2].item() == pytest.approx(expected)


def test_cutoff_source_gradient_is_unchanged_below_cutoff_and_zero_above():
    s = CutoffSource(node=0, G=1e-5, x_cut=1e-3)

    x_below = torch.tensor([9e-4], dtype=F64, requires_grad=True)
    s(0.0, x_below)[0].backward()
    assert x_below.grad.item() == pytest.approx(-1e-5 / 1e-3)

    x_above = torch.tensor([2e-3], dtype=F64, requires_grad=True)
    s(0.0, x_above)[0].backward()
    assert x_above.grad.item() == 0.0


def test_decaying_source_follows_the_exponential_from_its_start():
    s = DecayingSource(node=0, G0=1e-5, tau=600.0, t0=100.0)
    x = torch.zeros(2, dtype=F64)
    assert s(50.0, x)[0].item() == 0.0
    assert s(700.0, x)[0].item() == pytest.approx(1e-5 * math.exp(-1.0))


def test_burst_source_delivers_its_mass_over_one_step():
    s = BurstSource(node=1, mass=0.5, t_burst=300.0, dt=60.0)
    x = torch.zeros(2, dtype=F64)
    assert s(299.0, x)[1].item() == 0.0
    assert s(300.0, x)[1].item() == pytest.approx(0.5 / 60.0)
    assert s(359.0, x)[1].item() == pytest.approx(0.5 / 60.0)
    assert s(360.0, x)[1].item() == 0.0


def test_assemble_sources_sums_contributions_in_full_node_order():
    srcs = [
        ConstantSource(node=1, G=1e-6),
        ConstantSource(node=1, G=2e-6),
        CutoffSource(node=2, G=1e-6, x_cut=1.0),
    ]
    x = torch.zeros(3, dtype=F64)
    # Ruling R11: `n` is dropped from the signature (it was never read; the shape comes
    # from `x_full`), so this is a 3-argument call, not the brief's dictated 4-argument one.
    out = assemble_sources(srcs, 0.0, x)
    torch.testing.assert_close(out, torch.tensor([0.0, 3e-6, 1e-6], dtype=F64))


# --------------------------------------------------------------------------------------
# `sources_from_project` -- untested by the brief's Step 1 fixtures. Rulings R5 and R10
# both change its error behaviour (a checked `KeyError` for an unresolvable zone, a
# checked `ValueError` for an unrecognised source type), so both paths get their own test
# here, plus one exercising the normal four-type translation.
# --------------------------------------------------------------------------------------


@dataclass
class _FakeProject:
    net: Network
    sources: list
    zone_nr_to_name: dict


def _network_with_zones(*names: str) -> Network:
    net = Network(dtype=F64)
    net.add_node("ambient")
    for name in names:
        net.add_node(name)
    return net


def test_sources_from_project_builds_each_type_at_its_resolved_zone():
    net = _network_with_zones("z1", "z2")
    sources = [
        PrjSource(
            nr=1, zone_nr=5, element_nr=1, source_type="ccf",
            params=[2e-6, 1e-3], mult=1.0,
        ),
        PrjSource(
            nr=2, zone_nr=7, element_nr=2, source_type="cut",
            params=[1e-5, 1e-3], mult=1.0,
        ),
        PrjSource(
            nr=3, zone_nr=5, element_nr=3, source_type="eds",
            params=[1e-5, 1.0 / 600.0], mult=1.0,
        ),
        PrjSource(nr=4, zone_nr=7, element_nr=4, source_type="brs", params=[0.5], mult=1.0),
    ]
    project = _FakeProject(net=net, sources=sources, zone_nr_to_name={5: "z1", 7: "z2"})
    out = sources_from_project(project, dt=60.0)
    assert [type(s) for s in out] == [ConstantSource, CutoffSource, DecayingSource, BurstSource]
    n_z1, n_z2 = net.node_index("z1"), net.node_index("z2")
    assert out[0].node == n_z1
    assert out[0].G == pytest.approx(2e-6) and out[0].R == pytest.approx(1e-3)
    assert out[1].node == n_z2 and out[1].x_cut == pytest.approx(1e-3)
    assert out[2].node == n_z1 and out[2].tau == pytest.approx(600.0)
    assert out[3].node == n_z2
    assert out[3].mass == pytest.approx(0.5) and out[3].dt == pytest.approx(60.0)


def test_sources_from_project_raises_keyerror_naming_source_and_zone_when_zone_is_absent():
    net = _network_with_zones("z1")
    sources = [
        PrjSource(
            nr=9, zone_nr=42, element_nr=1, source_type="ccf",
            params=[1e-6, 0.0], mult=1.0,
        ),
    ]
    project = _FakeProject(net=net, sources=sources, zone_nr_to_name={5: "z1"})
    with pytest.raises(KeyError, match="9") as exc:
        sources_from_project(project)
    assert "42" in str(exc.value)


def test_sources_from_project_raises_valueerror_naming_source_and_type_for_an_unknown_type():
    net = _network_with_zones("z1")
    sources = [PrjSource(nr=3, zone_nr=5, element_nr=1, source_type="wpc", params=[1e-6], mult=1.0)]
    project = _FakeProject(net=net, sources=sources, zone_nr_to_name={5: "z1"})
    with pytest.raises(ValueError, match="3") as exc:
        sources_from_project(project)
    assert "wpc" in str(exc.value)

"""Electrical: ensemble equivalence, warm-start independence and finite-difference gradients
on a nonlinear (diode) circuit with storage."""

import pytest
import torch

from noodl.apps.electrical import (
    LAYER,
    Capacitor,
    Circuit,
    Diode,
    Resistor,
    Signal,
    VoltageSource,
    build_model,
    operating_point,
)
from noodl.apps.electrical.analysis import initial_state, transient

F64 = torch.float64
PHI = f"{LAYER}.phi"


@pytest.fixture(autouse=True)
def _float64_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(F64)
    yield
    torch.set_default_dtype(previous)


def _circuit(**kw):
    """v -> R1 -> d -> diode -> 0, with a capacitor on d: 4 nodes (incl. ground)."""
    return Circuit([VoltageSource("V1", "a", "0", 1.0, waveform=Signal("v")),
                    Resistor("R1", "a", "d", 100.0), Diode("D1", "d", "0"),
                    Capacitor("C1", "d", "0", 1e-3)], **kw)


# 3 nodes (a, d, ground) and 4 devices: 1, 3 and 4 are the coincidental sizes.
@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_batched_source_with_unbatched_state_equals_stack_of_single_runs(b):
    model, _, drivers = build_model(_circuit())
    v = torch.linspace(0.5, 2.0, b).reshape(b, 1)
    state = initial_state(model, {**drivers, "v": torch.ones(1)})
    out = model.step(state, {**drivers, "v": v}, 0.01, t=0.0)
    assert out[PHI].shape[0] == b
    for i in range(b):
        single = model.step(state, {**drivers, "v": v[i]}, 0.01, t=0.0)
        torch.testing.assert_close(out[PHI][i], single[PHI], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("b", [1, 3])
def test_batched_transient_equals_stack_of_single_transients(b):
    model, _, drivers = build_model(_circuit())
    v = torch.linspace(0.5, 2.0, b).reshape(b, 1)
    d = {**drivers, "v": v}
    grid = [0.0, 0.01, 0.02, 0.03]
    out = transient(model, initial_state(model, d), d, grid)[PHI]
    for i in range(b):
        di = {**drivers, "v": v[i]}
        single = transient(model, initial_state(model, di), di, grid)[PHI]
        torch.testing.assert_close(out[:, i], single, rtol=1e-8, atol=1e-11)


@pytest.mark.parametrize("guess", ["zeros", "distant", "negative"])
def test_operating_point_is_independent_of_the_initial_guess(guess):
    """The diode's exponential law is where a poor warm start fails; every start reaches the
    same solution."""
    model, _, drivers = build_model(_circuit())
    d = {**drivers, "v": torch.tensor([2.0])}
    tight = {"atol": 1e-14, "rtol": 0.0}
    ref = operating_point(model, d, **tight)[PHI]
    n = model.net.n
    phi0 = {"zeros": torch.zeros(n), "distant": torch.full((n,), 5.0),
            "negative": torch.full((n,), -3.0)}[guess]
    phi = model.steady({PHI: phi0}, d, **tight)[PHI]
    torch.testing.assert_close(phi, ref, rtol=1e-9, atol=1e-11)


def test_diode_operating_point_gradient_matches_central_differences():
    """d v(d) / d R1 by autograd equals the central difference (forward-biased diode, far from
    any switching)."""
    def solve(r):
        circuit = Circuit([VoltageSource("V1", "a", "0", 2.0), Resistor("R1", "a", "d", r),
                           Diode("D1", "d", "0")])
        model, _, drivers = build_model(circuit)
        return operating_point(model, drivers, atol=1e-14, rtol=0.0)[PHI][
            model.net.nodes.index("d")]

    # the resistance is a device parameter: build it learnable through the netlist
    model, _, drivers = build_model(Circuit([VoltageSource("V1", "a", "0", 2.0),
                                             Resistor("R1", "a", "d", 100.0),
                                             Diode("D1", "d", "0")]), learnable=True)
    law, _ = model.kirchhoff[LAYER].law_for("resistor")
    v = operating_point(model, drivers, atol=1e-14, rtol=0.0)[PHI][model.net.nodes.index("d")]
    (g,) = torch.autograd.grad(v, law.R)
    h = 1e-3
    with torch.no_grad():
        fd = (solve(100.0 + h) - solve(100.0 - h)) / (2 * h)
    assert g.item() == pytest.approx(fd.item(), rel=1e-5)

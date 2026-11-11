"""The storage ramp of the fallback ladder on a water network in its resistance (flow) form,
where every pipe's flow is an unknown and the nodes' balance rows have no diagonal."""

from pathlib import Path

import pytest
import torch

from noodl.apps.water.elements import FlowControlledPipe, FlowControlledPump
from noodl.apps.water.inp import read_epanet_inp
from noodl.apps.water.network import build_model, water_steady
from noodl.layers._flow_unknowns import split
from noodl.layers.kirchhoff import KirchhoffLayer
from noodl.solvers import ConvergenceError, continuation

NET1 = Path(__file__).resolve().parents[2] / "data" / "water" / "Net1.inp"
TOL = dict(atol=1e-11, rtol=1e-11)


def test_a_failed_warm_start_converges_through_the_storage_ramp(monkeypatch):
    model, state, drivers = build_model(read_epanet_inp(NET1))
    reference = water_steady(model, state, drivers)
    layer = model.kirchhoff["water"]
    n_kcl = len(layer.interior)
    warm = {**state, "water.q": state["water.q"] * 30.0}     # far from the operating point
    with pytest.raises(ConvergenceError):
        model.steady(warm, drivers, max_iter=4, fallback=(), **TOL)

    shifts = []
    real_shift = continuation.shift_diagonal

    def spy(op, g):
        shifts.append((op.diagonal(), g))
        return real_shift(op, g)

    monkeypatch.setattr(continuation, "shift_diagonal", spy)
    diagnostics: dict = {}
    out = model.steady(warm, drivers, max_iter=4, fallback=("storage_ramp",),
                       diagnostics=diagnostics, **TOL)
    assert diagnostics["layers"]["water"]["fallback"] == "storage_ramp"
    torch.testing.assert_close(out["water.phi"], reference["water.phi"], rtol=0, atol=1e-9)
    torch.testing.assert_close(out["water.q"], reference["water.q"], rtol=0, atol=1e-12)
    # The balance rows of a network whose every edge is a flow unknown have no diagonal, so
    # no storage; the rows of the flow unknowns take an inertia with their diagonal's sign,
    # which never cancels it.
    diag, g = shifts[0]
    assert bool((diag[..., :n_kcl] == 0).all()) and bool((g[..., :n_kcl] == 0).all())
    flow_diag, flow_g = diag[..., n_kcl:], g[..., n_kcl:]
    assert bool((flow_g != 0).any())
    assert bool((torch.sign(flow_g) == torch.sign(flow_diag)).all())
    assert bool(((flow_diag + flow_g).abs() >= flow_diag.abs()).all())


def _ramp_starts(monkeypatch, n_kcl):
    """The flow unknowns each storage ramp starts from, recorded as the ladder runs."""
    starts = []
    real_ramp = KirchhoffLayer._ramp

    def spy(self, x0, *args, **kwargs):
        starts.append(x0[..., n_kcl:].detach().clone())
        return real_ramp(self, x0, *args, **kwargs)

    monkeypatch.setattr(KirchhoffLayer, "_ramp", spy)
    return starts


def test_net1s_linear_guess_starts_from_epanets_first_iterate(monkeypatch):
    """The pipes' and the pump's start flows (`start_flow`: 1 ft/s, the design flow) put
    Net1's linear guess within a factor of two of the converged flows; from zero flow, on
    the pipes' `RQtol` floor, it carried the 52 m between the reservoir and the tank as
    1.9e7 m3/s."""
    model, state, drivers = build_model(read_epanet_inp(NET1))
    reference = water_steady(model, state, drivers)
    layer = model.kirchhoff["water"]
    guesses = []
    real_guess = KirchhoffLayer._linear_guess

    def spy(self, *args, **kwargs):
        guesses.append(real_guess(self, *args, **kwargs))
        return guesses[-1]

    monkeypatch.setattr(KirchhoffLayer, "_linear_guess", spy)
    warm = {**state, "water.q": state["water.q"] * 30.0}
    diagnostics: dict = {}
    out = model.steady(warm, drivers, max_iter=4, fallback=("linear_guess",),
                       diagnostics=diagnostics, **TOL)
    _, q_f = split(layer._unknowns, guesses[0])
    ratio = q_f / reference["water.q"][..., layer.implicit_edges]
    assert bool((ratio > 0.5).all()) and bool((ratio < 2.0).all())
    # and the "linear_guess" rung converges from it, where from zero flow it did not
    assert diagnostics["layers"]["water"]["fallback"] == "linear_guess"
    torch.testing.assert_close(out["water.q"], reference["water.q"], rtol=0, atol=1e-12)


def test_a_warm_start_ramp_converges_from_the_linear_guess(monkeypatch):
    """With the start flows the first ramp, from the linear guess, converges: the ramp from
    the warm start is not needed."""
    model, state, drivers = build_model(read_epanet_inp(NET1))
    reference = water_steady(model, state, drivers)
    layer = model.kirchhoff["water"]
    starts = _ramp_starts(monkeypatch, len(layer.interior))
    warm = {**state, "water.q": state["water.q"] * 30.0}
    diagnostics: dict = {}
    out = model.steady(warm, drivers, max_iter=4, fallback=("storage_ramp",),
                       diagnostics=diagnostics, **TOL)
    assert len(starts) == 1
    assert float(starts[0].abs().max()) < 1.0                    # the linear guess's flows
    assert diagnostics["layers"]["water"]["fallback"] == "storage_ramp"
    torch.testing.assert_close(out["water.phi"], reference["water.phi"], rtol=0, atol=1e-9)
    torch.testing.assert_close(out["water.q"], reference["water.q"], rtol=0, atol=1e-12)


def test_the_ramp_runs_again_from_the_warm_start_when_the_linear_guess_settles_nowhere(
        monkeypatch):
    """Net1's laws taken back to zero start flow: the linear guess carries the 52 m between
    the reservoir and the tank over the pipes' `RQtol` floor, a flow of 1e7 m3/s; the
    storage ramp settles nowhere from it, and runs again from the failed solve's own start,
    which converges."""
    for law in (FlowControlledPipe, FlowControlledPump):
        monkeypatch.setattr(law, "start_flow", lambda self, drivers=None, ctx=None: None)
    model, state, drivers = build_model(read_epanet_inp(NET1))
    reference = water_steady(model, state, drivers)
    layer = model.kirchhoff["water"]
    warm = {**state, "water.q": state["water.q"] * 30.0}
    starts = _ramp_starts(monkeypatch, len(layer.interior))
    diagnostics: dict = {}
    out = model.steady(warm, drivers, max_iter=4, fallback=("storage_ramp",),
                       diagnostics=diagnostics, **TOL)
    assert len(starts) == 2
    assert float(starts[0].abs().max()) > 1e6                    # the linear guess's flows
    torch.testing.assert_close(starts[1], warm["water.q"][..., layer.implicit_edges])
    assert diagnostics["layers"]["water"]["fallback"] == "storage_ramp"
    assert diagnostics["layers"]["water"]["fallback_iterations"] > 16   # both ramps' stages
    torch.testing.assert_close(out["water.phi"], reference["water.phi"], rtol=0, atol=1e-9)
    torch.testing.assert_close(out["water.q"], reference["water.q"], rtol=0, atol=1e-12)

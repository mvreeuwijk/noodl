"""Each noodl node replayed on the calls WSIMOD 0.8.1's node received (spec, Verification 1).

WSIMOD runs each purpose-built case (``_wsimod_cases.py``) under ``capture_node_calls``; the
replay builds the noodl node alone on scripted arcs and compares every arc call it makes,
every reply and every store to ``1e-12 * scale``, in both modes: with the case's pollutants,
and quantity-only on the volumes.
"""

from __future__ import annotations

import copy
import functools

import pytest

pytest.importorskip("wsimod")

from noodl.apps.wsimod.config import read_wsimod_config  # noqa: E402
from noodl.apps.wsimod.vqip import QUANTITY_ONLY  # noqa: E402
from tests.apps.wsimod._replay import BOUND, Mismatch, replay  # noqa: E402
from tests.verification._wsimod_cases import CASES  # noqa: E402
from tests.verification._wsimod_reference import (  # noqa: E402
    capture_node_calls,
    record_wsimod_run,
    wsimod_model,
)

PAIRS = [(case, name) for case, spec in CASES.items() if spec.wsimod for name in spec.replay]


@functools.cache
def captured(case: str):
    spec = CASES[case]
    model = wsimod_model(spec.build())
    with capture_node_calls(model, spec.replay) as calls:
        record_wsimod_run(model, days=spec.days)
    return calls


@pytest.mark.parametrize(("case", "name"), PAIRS, ids=[f"{c}-{n}" for c, n in PAIRS])
def test_replay_matches_wsimod(case, name):
    spec = CASES[case]
    calls = captured(case)[name]
    assert calls, f"WSIMOD made no call into {name!r}"
    network = read_wsimod_config(spec.build())
    for mode in spec.modes:
        pollutants = network.pollutants if mode == "pollutants" else QUANTITY_ONLY
        worst = replay(network, name, calls, scale=spec.scale, pollutants=pollutants)
        assert worst <= BOUND, (mode, worst)


def test_the_cases_reach_the_branches_they_are_built_for():
    """three_out_arcs over-subscribes the junction on some days and not others, and polls
    again after a round; node_chain's pushes ask the checks down the chain."""
    calls = captured("three_out_arcs")["junction"]
    pushes = [c for c in calls if c.method == "push_set"]
    rounds = [sum(a.method == "send_push_request" for a in c.arc_calls) for c in pushes]
    assert max(rounds) > 3 and min(rounds) == 3
    oversubscribed = [
        c.vqip["volume"] > sum(a.reply["volume"] for a in c.arc_calls[:3]) for c in pushes
    ]
    assert any(oversubscribed) and not all(oversubscribed)
    chain = captured("node_chain")
    checks = [c for c in chain["reach_2"] if c.method == "push_check"]
    assert checks and all([a.method for a in c.arc_calls] == ["send_push_check"] for c in checks)
    routes = [c for c in chain["headwater"] if c.method == "route"]
    unrouted = [c.post["unrouted_water"]["volume"] > 0 for c in routes]
    assert len(routes) == CASES["node_chain"].days and any(unrouted) and not all(unrouted)


def test_the_replay_notices_a_difference():
    """A reply moved by 1e-11 x scale, or a reordered arc script, fails the replay."""
    spec = CASES["three_out_arcs"]
    network = read_wsimod_config(spec.build())
    calls = copy.deepcopy(captured("three_out_arcs")["junction"])
    call = next(c for c in calls if c.method == "push_set" and len(c.arc_calls) > 6)
    push = next(a for a in call.arc_calls if a.method == "send_push_request")
    push.reply["volume"] += 1e-11 * spec.scale
    with pytest.raises(Mismatch):
        replay(network, "junction", calls, scale=spec.scale)
    calls = copy.deepcopy(captured("three_out_arcs")["junction"])
    call = next(c for c in calls if c.method == "push_set")
    call.arc_calls[3], call.arc_calls[4] = call.arc_calls[4], call.arc_calls[3]
    with pytest.raises(Mismatch):
        replay(network, "junction", calls, scale=spec.scale)

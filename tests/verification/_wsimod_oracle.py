"""Instruments a live WSIMOD run to capture per-arc requests and realised flows.

Test-only. WSIMOD is never reimplemented (framework spec 4.2b, milestone 4b design spec
section 1): this module runs WSIMOD's OWN engine on WSIMOD's OWN demo models and
records what it actually did, so `CapacitatedTransferLayer` can be validated by replay
rather than by re-deriving WSIMOD's node science (design spec section 4.4).

This module is a plain helper, not a test file, so it does not itself guard the
`wsimod` import -- every test module that imports it must call
`pytest.importorskip("wsimod")` before doing so, matching the pyswmm/wntr pattern.

Hook point (design spec amendment A1): `Arc.send_push_request`/`send_pull_request` is
the one place both the per-arc REQUEST (pre-clip, the `vqip` argument) and the REALISED
transfer (`requested - reply`, or `reply` itself for a pull -- see each wrapper below)
are visible together, at exactly the per-edge granularity `CapacitatedTransferLayer`'s
incidence structure needs. `Arc.get_excess` clips against an ACCUMULATING per-timestep
`flow_in`, reset only by `end_timestep`, so if an arc sees MORE THAN ONE push/pull event
within one WSIMOD timestep, each event is captured here as a SEPARATE row -- this
function does not sum them, deliberately: aggregation is the caller's decision, made
where the replay semantics are known, not silently here.

**Arcs DO see more than one event per timestep, in a demo this repo actually ships.**
`oxford_demo` has four such arcs (`abstraction_to_farmoor`, `evenlode_to_thames`,
`thames_to_thames`, `thames_to_farmoor`), each with both a push and a pull event in the
same timestep -- 5824 of its 28056 distinct `(arc, timestep)` pairs carry two rows.
`quickstart_demo` happens to have none (7464 events, 7464 pairs), but that is a property
of that one demo, not a general guarantee, and earlier versions of this docstring claimed
it held for both. Anything replaying these events through one
`CapacitatedTransferLayer.step` per timestep must therefore ACCUMULATE a `(arc, t)`
group's rows rather than assign them one at a time -- `tests/verification/
test_wsimod_parity.py` (Tasks 6/7) does exactly that, and the fixture-writing script
`scripts/regenerate_wsimod_fixtures.py` aggregates only over (arc, DIRECTION, timestep),
leaving push and pull as separate rows for the replay to sum.

Timestep index: `wsimod.orchestration.model.Model` has no `.t` or other current-
timestep attribute of its own (confirmed by inspecting a constructed `Model` directly,
milestone 4b Task 5's smoke check). `Model.run` instead sets `node.t = date` on every
node in `model.nodelist` at the START of each timestep, before that timestep's push/pull
calls are dispatched -- so `arc.in_port.t` holds the date active during any call this
harness intercepts. `t` in each captured event is that date's 0-based position in
`model.dates` (looked up once per event against a `{date: index}` table built when
`capture_events` is entered), which requires `model.dates` to already be set -- true for
every model built by `create_oxford_model` or the quickstart inline build (design spec
amendment A3) before `.run()` is ever called.
"""

from __future__ import annotations

from contextlib import contextmanager

__all__ = ["capture_events", "extract_topology"]


@contextmanager
def capture_events(model):
    """Yields a list that fills with one dict per push/pull event during `model.run()`.

    Each dict: `{"arc": str, "direction": "push" | "pull", "t": int, "requested":
    float, "realised": float}`. `requested`/`realised` are WSIMOD's raw per-timestep
    volumes (WSIMOD's native unit, not m^3/s -- callers convert at the boundary, design
    spec section 2). See the module docstring for what `t` means and for why a caller
    must aggregate per `(arc, timestep)` itself -- an arc CAN see several events in one
    timestep, and this function never sums them.

    `model.arcs` is a plain `dict[str, Arc]` on `wsimod.orchestration.model.Model`
    (confirmed directly, not guarded with `hasattr`: a permanent fallback on a fact this
    cheap to check is exactly what this project's conventions refuse). Every original
    `send_push_request`/`send_pull_request` is restored on exit, including on an
    exception raised inside the `with` block.
    """
    if not model.arcs:
        raise ValueError(f"capture_events: model {model!r} has no arcs to instrument")
    dates = getattr(model, "dates", None)
    if dates is None:
        # `Model.__init__` never sets `self.dates` at all (confirmed directly, not
        # assumed) -- it is only assigned after construction, by `create_oxford_model`
        # or the quickstart inline build (design spec amendment A3). `getattr` here is
        # a genuine "has this been set yet" check, not a guess at an alternative
        # attribute name.
        raise ValueError(
            "capture_events: model.dates is not set -- build the model (which sets "
            "`.dates`, e.g. create_oxford_model) before capturing, not after"
        )

    events: list[dict] = []
    date_index = {date: i for i, date in enumerate(dates)}

    def _t(arc) -> int:
        date = arc.in_port.t
        try:
            return date_index[date]
        except KeyError as exc:
            raise ValueError(
                f"capture_events: arc {arc.name!r}'s in_port {arc.in_port.name!r} has "
                f"t={date!r}, not one of model.dates -- was the model run with a "
                "different `dates` than the one `capture_events` was entered with?"
            ) from exc

    originals = []
    for arc in model.arcs.values():
        orig_push = arc.send_push_request
        orig_pull = arc.send_pull_request

        def make_push(arc=arc, orig=orig_push):
            def wrapped(vqip, tag="default", force=False):
                requested = vqip["volume"]
                reply = orig(vqip, tag=tag, force=force)
                events.append(
                    {
                        "arc": arc.name,
                        "direction": "push",
                        "t": _t(arc),
                        "requested": requested,
                        "realised": requested - reply["volume"],
                    }
                )
                return reply

            return wrapped

        def make_pull(arc=arc, orig=orig_pull):
            def wrapped(vqip, tag="default"):
                requested = vqip["volume"]
                reply = orig(vqip, tag=tag)
                events.append(
                    {
                        "arc": arc.name,
                        "direction": "pull",
                        "t": _t(arc),
                        "requested": requested,
                        "realised": reply["volume"],
                    }
                )
                return reply

            return wrapped

        arc.send_push_request = make_push()
        arc.send_pull_request = make_pull()
        originals.append((arc, orig_push, orig_pull))
    try:
        yield events
    finally:
        for arc, orig_push, orig_pull in originals:
            arc.send_push_request = orig_push
            arc.send_pull_request = orig_pull


def extract_topology(model) -> dict:
    """Node names/types and arc `(name, source, target, capacity)`, in the model's own
    iteration order.

    Reads `model.nodes`/`model.arcs` (plain `dict`s, see `capture_events`) and each
    arc's own `.in_port`/`.out_port`/`.capacity` attributes directly off the constructed
    WSIMOD `Model` -- never hand-transcribed, closing off the class of bug milestone 3
    hit with the hand-ported IMPAQ prototype.
    """
    if not model.arcs:
        raise ValueError(f"extract_topology: model {model!r} has no arcs")
    return {
        "nodes": [{"name": n.name, "type": type(n).__name__} for n in model.nodes.values()],
        "arcs": [
            {
                "name": a.name,
                "source": a.in_port.name,
                "target": a.out_port.name,
                "capacity": float(a.capacity),
            }
            for a in model.arcs.values()
        ],
    }

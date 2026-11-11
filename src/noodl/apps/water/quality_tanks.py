"""EPANET 2.2's tank mixing models, transcribed (`qualreact.c`, tag v2.2: `mixtank`,
`tankmix1` .. `tankmix4`; `qualroute.c`, `initsegs`).

A tank's chain in the `SegmentPool` is its water: MIXED (`tankmix1`) one segment; 2COMP
(`tankmix2`) two, the stagnant zone `FirstSeg` and the mixing zone `LastSeg` (`initsegs`'s
own comments say the reverse); FIFO (`tankmix3`) and LIFO (`tankmix4`) as many as the water
entering at different qualities makes. The functions work on the pool's scalars
(`SegmentPool.open`: `cs`, `vs`), take the volumes and the mass of the step (`vin`, `win`,
`vnet = vin - vout`) as EPANET does, and return the tank's quality (`Stank.C`). Every decision
(a branch, a segment count) is taken on a detached value; volumes and concentrations keep
their graph (Ruling Q5). EPANET's quirks are kept: a used-up segment without a segment
behind it stays at its volume (FIFO, LIFO) and the last segment gives whatever is asked
of it; a 2COMP tank without net flow loses its stagnant zone's volume (`vt == 0` sets it to
0).
"""

from __future__ import annotations

import torch


def detached(x) -> float:
    """A decision's operand: `x` as a Python float, detached."""
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def initial_segments(pool, chain: int, model: str, volume, c, vmz: float) -> None:
    """`initsegs` (`qualroute.c`) for one tank: a segment of `volume` at quality `c`; for
    2COMP, that first (stagnant) segment holds `max(0, volume - V1max)` and a second (the
    mixing zone) the rest, `min(volume, V1max)`."""
    if model != "2COMP":
        pool.add(chain, volume, c)
        return
    stagnant = volume - vmz
    if detached(stagnant) <= 0.0:                    # MAX(0, v - V1max)
        stagnant = 0.0
    pool.add(chain, stagnant, c)
    pool.add(chain, volume - stagnant, c)


def mix_tank(pool, chain: int, model: str, vmz: float, ctol: float, vin, win, vnet,
             c_tank=0.0):
    """`mixtank` (`qualreact.c`): tank chain `chain` mixed by `model` ("MIXED", "2COMP",
    "FIFO", "LIFO") with inflow volume `vin`, inflow mass `win` and `vnet = vin - vout` over
    the step; `vmz` is the 2COMP mixing zone's full volume `V1max` and `ctol` the quality
    tolerance (FIFO and LIFO merge an inflow within it of the last segment). Returns the
    tank's quality; `c_tank` is returned unchanged when a model returns early (no segment).
    The pool's scalars must be open."""
    if model == "MIXED":
        return _tankmix1(pool, chain, vin, win, vnet, c_tank)
    if model == "2COMP":
        return _tankmix2(pool, chain, vmz, vin, win, vnet, c_tank)
    if model == "FIFO":
        return _tankmix3(pool, chain, ctol, vin, win, vnet, c_tank)
    return _tankmix4(pool, chain, ctol, vin, win, vnet, c_tank)


def _tankmix1(pool, chain, vin, win, vnet, c_tank):
    """`tankmix1`: a completely mixed tank; its quality is its segment's."""
    seg = pool.first[chain]
    if seg < 0:
        return c_tank
    cs, vs = pool.cs, pool.vs
    vnew = vs[seg] + vin
    if detached(vnew) > 0.0:
        cs[seg] = (cs[seg] * vs[seg] + win) / vnew
    vs[seg] = vs[seg] + vnet
    if detached(vs[seg]) < 0.0:                       # MAX(0.0, seg->v)
        vs[seg] = 0.0
    return cs[seg]


def _tankmix2(pool, chain, vmz, vin, win, vnet, c_tank):
    """`tankmix2`: two compartments. The mixing zone (`LastSeg`) holds up to `vmz`; filling
    beyond that moves `vt` of it to the stagnant zone (`FirstSeg`); emptying draws `vt` from
    the stagnant zone into the mixing zone. The tank's quality is the mixing zone's."""
    mix, stag = pool.last[chain], pool.first[chain]
    if mix < 0 or stag < 0:
        return c_tank
    cs, vs = pool.cs, pool.vs
    vt = 0.0
    net = detached(vnet)
    if net > 0.0:                                     # filling
        vt = vs[mix] + vnet - vmz
        if detached(vt) <= 0.0:                       # MAX(0.0, mix->v + vnet - vmz)
            vt = 0.0
        if detached(vin) > 0.0:
            cs[mix] = (cs[mix] * vs[mix] + win) / (vs[mix] + vin)
        if detached(vt) > 0.0:
            cs[stag] = (cs[stag] * vs[stag] + cs[mix] * vt) / (vs[stag] + vt)
    elif net < 0.0:                                   # emptying
        if detached(vs[stag]) > 0.0:
            vt = vs[stag] if detached(vs[stag]) < detached(-vnet) else -vnet   # MIN
        if detached(vin + vt) > 0.0:
            cs[mix] = (cs[mix] * vs[mix] + win + cs[stag] * vt) / (vs[mix] + vin + vt)
    if detached(vt) > 0.0:
        vs[mix] = vmz
        if net > 0.0:
            vs[stag] = vs[stag] + vt
        else:
            left = vs[stag] - vt
            vs[stag] = left if detached(left) > 0.0 else 0.0              # MAX(0, .)
    else:
        v = vs[mix] + vnet
        if detached(v) > detached(vmz):               # MIN(v, vmz)
            v = vmz
        if detached(v) < 0.0:                         # MAX(0.0, v)
            v = 0.0
        vs[mix] = v
        vs[stag] = 0.0
    return cs[mix]


def _withdraw(pool, chain, volume):
    """The loop of `tankmix3` and `tankmix4`: `volume` taken from the chain's leading
    segment onwards (the last segment gives whatever is asked of it); a used-up segment is
    freed only when another is behind it. Returns the volume and mass taken."""
    cs, vs = pool.cs, pool.vs
    vsum, wsum = 0.0, 0.0
    while detached(volume) > 0.0:
        seg = pool.first[chain]
        if seg < 0:
            break
        vseg = vs[seg]
        if not detached(vseg) < detached(volume):     # MIN(vseg, volume)
            vseg = volume
        if seg == pool.last[chain]:
            vseg = volume
        vsum = vsum + vseg
        wsum = wsum + cs[seg] * vseg
        volume = volume - vseg
        if detached(volume) >= 0.0 and detached(vseg) >= detached(vs[seg]):
            if pool.prev[seg] >= 0:
                pool.remove_first(chain)
        else:
            vs[seg] = vs[seg] - vseg
    return vsum, wsum


def _tankmix3(pool, chain, ctol, vin, win, vnet, c_tank):
    """`tankmix3`, first in first out: the inflow joins the last segment when within `ctol`
    of it, else becomes a new last segment; then `vin - vnet` is withdrawn from the first
    segment onwards. The tank's quality is the mean of the water withdrawn, else its first
    segment's (0 with none)."""
    if pool.last[chain] < 0 or pool.first[chain] < 0:
        return c_tank
    cs, vs = pool.cs, pool.vs
    if detached(vin) > 0.0:
        cin = win / vin
        seg = pool.last[chain]
        if abs(detached(cs[seg]) - detached(cin)) < ctol:
            vs[seg] = vs[seg] + vin
        else:
            pool.add(chain, vin, cin)
    vsum, wsum = _withdraw(pool, chain, vin - vnet)
    if detached(vsum) > 0.0:
        return wsum / vsum
    if pool.first[chain] < 0:
        return 0.0
    return cs[pool.first[chain]]


def _tankmix4(pool, chain, ctol, vin, win, vnet, c_tank):
    """`tankmix4`, last in first out: a net inflow `vnet` joins the last segment when its
    quality is within `ctol` of the inflow's `win / vin`, else becomes a new last segment; a
    net outflow is withdrawn from the last segment backwards (the chain reversed for it and
    back after). The tank's quality is the last segment's after a net inflow (the new one
    if added), the mixture of the water released and the inflow (`(wsum + win) / (vsum +
    vin)`) after a net outflow (`qualreact.c`, last line of `tankmix4`)."""
    if pool.last[chain] < 0 or pool.first[chain] < 0:
        return c_tank
    cs, vs = pool.cs, pool.vs
    cin = win / vin if detached(vin) > 0.0 else 0.0
    seg = pool.last[chain]
    c_tank = cs[seg]
    net = detached(vnet)
    if net > 0.0:
        if abs(detached(cs[seg]) - detached(cin)) < ctol:
            vs[seg] = vs[seg] + vnet
        else:
            pool.add(chain, vnet, cin)
        c_tank = cs[pool.last[chain]]
    elif net < 0.0:
        pool.reverse(chain)
        vsum, wsum = _withdraw(pool, chain, -vnet)
        pool.reverse(chain)
        c_tank = (wsum + win) / (vsum + vin)       # the water released and any inflow
    return c_tank

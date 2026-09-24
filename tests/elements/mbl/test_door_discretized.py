"""Tests for `noodl.elements.mbl.door_discretized`: MBL's `DoorDiscretizedOpen` and
`DoorDiscretizedOperable` as `nCom` compartment edges plus a hydrostatic-head drive.

The NumPy reference below is transcribed directly from the Modelica Buildings Library (MBL)
v13.0.0 (commit 55abf579598ca81cae0a82f337350375958e6722) and the Modelica Standard Library
(MSL) v4.1.0 sources, never by importing `noodl.elements.mbl.door_discretized`:

* `Airflow/Multizone/BaseClasses/DoorDiscretized.mo:21,34-40,52,60-75` (dh, hAg/hBg,
  VZerCom_flow, dA, the compartment pressures, the smoothed directional split, VAB/VBA),
* `Airflow/Multizone/BaseClasses/TwoWayFlowElement.mo:72-92` (inflow densities at the actual
  port pressure via `density_pTX`; `VZer_flow = vZer*A`; `port_a1.m_flow = rho_a1_inflow *
  VAB_flow`, `port_a2.m_flow = rho_a2_inflow * VBA_flow`),
* `Airflow/Multizone/DoorDiscretizedOpen.mo:10-36` (mFixed = 0.5, CVal, powerLaw05),
* `Airflow/Multizone/DoorDiscretizedOperable.mo:35-63` (the y-blend of m, A and CVal and the
  variable-m `powerLaw`),
* `Airflow/Multizone/BaseClasses/powerLaw.mo:15-29`, `powerLaw05.mo:19-29`,
* `Utilities/Math/Functions/smoothHeaviside.mo:9-12`,
* `Utilities/Psychrometrics/Functions/density_pTX.mo:11-16`,
* MSL `Modelica/Constants.mo:38` (g_n), `Media/IdealGases/Common/SingleGasesData.mo:5,49,59,
  9187,9197` (R_NASA_2002, Air.MM, Air.R_s, H2O.MM, H2O.R_s).

Wiring (design section 6): side A holds `port_a1` (and `port_b2`), side B holds `port_a2`
(and `port_b1`); every compartment edge runs from A (`src`) to B (`tgt`).

The one rearrangement: MBL writes `dpAB[i] = (p_a1 + rho_A hAg[i]) - (p_a2 + rho_B hBg[i])`
(`DoorDiscretized.mo:64-66`). The transcription evaluates the algebraically identical
`(p_a1 - p_a2) + (rho_A hAg[i] - rho_B hBg[i])`: forming `p + rho g h` at 1e5 Pa first and
subtracting loses ~1e-11 Pa to cancellation, which for an in-band compartment dp of 1e-5 Pa
would be a 1e-6 relative error -- an artefact of the grouping, not of either model.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from noodl.elements.mbl import (
    DoorCompartmentHead,
    MBLDoorCompartment,
    MBLDoorCompartmentOperable,
    mbl_discretized_door,
    mbl_discretized_operable_door,
    medium,
)
from noodl.layers.potential import PotentialFlowLayer
from noodl.topology import Network

F64 = torch.float64
RTOL, ATOL = 1e-12, 1e-15

AIR = medium("Buildings.Media.Air")
SIMPLE = medium("Modelica.Media.Air.SimpleAir")

# ---------------------------------------------------------------------------------------
# NumPy transcription
# ---------------------------------------------------------------------------------------

G_N = 9.80665  # Modelica/Constants.mo:38
R_NASA_2002 = 8.314510  # SingleGasesData.mo:5
R_S_AIR = R_NASA_2002 / 0.0289651159  # SingleGasesData.mo:49,59
R_S_H2O = R_NASA_2002 / 0.01801528  # SingleGasesData.mo:9187,9197
P_DEFAULT = 101325.0  # Modelica/Media/package.mo:3969
GAMMA = 1.5  # DoorDiscretizedOpen.mo:11, powerLaw.mo:15


def density_pTX_np(p, T, X_w):
    """Utilities/Psychrometrics/Functions/density_pTX.mo:11-16."""
    R = R_S_AIR * (1 - X_w) + R_S_H2O * X_w
    return p / (R * T)


def smooth_heaviside_np(x, delta):
    """Utilities/Math/Functions/smoothHeaviside.mo:9-12."""
    dx = 0.5 * x / delta
    xpow2 = dx * dx
    return max(0.0, min(1.0, 0.5 + dx * (1.875 + xpow2 * (-5 + 6 * xpow2))))


def power_law_05_np(C, dp, dp_t):
    """BaseClasses/powerLaw05.mo:24-29 with DoorDiscretizedOpen.mo:13-20's a..d at m=0.5."""
    m = 0.5
    a = GAMMA
    b = 1 / 8 * m**2 - 3 * GAMMA - 3 / 2 * m + 35.0 / 8
    c = -1 / 4 * m**2 + 3 * GAMMA + 5 / 2 * m - 21.0 / 4
    d = 1 / 8 * m**2 - GAMMA - m + 15.0 / 8
    pi = dp / dp_t
    pi2 = pi * pi
    if dp >= dp_t:
        return C * math.sqrt(dp)
    if dp <= -dp_t:
        return -C * math.sqrt(-dp)
    return C * pi * math.sqrt(dp_t) * (a + pi2 * (b + pi2 * (c + pi2 * d)))


def power_law_np(C, dp, m, dp_t):
    """BaseClasses/powerLaw.mo:17-29 (variable m)."""
    pi = dp / dp_t
    pi2 = pi * pi
    if dp >= dp_t:
        return C * dp**m
    if dp <= -dp_t:
        return -C * (-dp) ** m
    return (
        C
        * dp_t**m
        * pi
        * (
            GAMMA
            + pi2
            * (
                (1 / 8 * m**2 - 3 * GAMMA - 3 / 2 * m + 35.0 / 8)
                + pi2
                * (
                    (-1 / 4 * m**2 + 3 * GAMMA + 5 / 2 * m - 21.0 / 4)
                    + pi2 * (1 / 8 * m**2 - GAMMA - m + 15.0 / 8)
                )
            )
        )
    )


def door_discretized_np(
    pA,
    pB,
    TA,
    TB,
    XA,
    XB,
    *,
    nCom,
    wOpe,
    hOpe,
    hA,
    hB,
    dp_t,
    vZer,
    med,
    CD=0.65,
    operable=None,
    dphi=None,
):
    """One DoorDiscretizedOpen (or, with `operable` a dict holding y, LClo, CDOpe, CDClo,
    CDCloRat, dpCloRat, mOpe, mClo, a DoorDiscretizedOperable). pA/pB are the absolute port
    pressures p_a1/p_a2, TA/TB and XA/XB the inflow (zone) temperatures and water fractions.
    `dphi` overrides the port pressure difference p_a1 - p_a2 (default pA - pB) where a caller
    holds it more accurately than the difference of two 1e5 Pa numbers (the layer test).
    Returns a dict of MBL's per-compartment and summed quantities."""
    moist = med.has_moisture
    # TwoWayFlowElement.mo:72-81: density_pTX at the port pressure, X_w = 0 if nXi == 0.
    rhoA = density_pTX_np(pA, TA, XA if moist else 0.0)
    rhoB = density_pTX_np(pB, TB, XB if moist else 0.0)
    rho_default = med.rho_default  # DoorDiscretized.mo:23-29
    dh = hOpe / nCom  # DoorDiscretized.mo:21
    hAg = [G_N * (hA - (i - 0.5) * dh) for i in range(1, nCom + 1)]  # DoorDiscretized.mo:34-36
    hBg = [G_N * (hB - (i - 0.5) * dh) for i in range(1, nCom + 1)]  # DoorDiscretized.mo:38-40
    if operable is None:
        A = wOpe * hOpe  # DoorDiscretizedOpen.mo:23
        dA = A / nCom  # DoorDiscretized.mo:60
        CVal = CD * dA * math.sqrt(2 / rho_default)  # Open.mo:24
        m = 0.5
    else:
        o = operable
        y = o["y"]
        AOpe = wOpe * hOpe  # DoorDiscretizedOperable.mo:35
        AClo = o["CDClo"] / o["CDCloRat"] * o["LClo"] * o["dpCloRat"] ** (0.5 - o["mClo"])  # :43
        CClo = o["CDClo"] * AClo / nCom * math.sqrt(2 / rho_default)  # :46
        COpe = o["CDOpe"] * AOpe / nCom * math.sqrt(2 / rho_default)  # :47
        m = y * o["mOpe"] + (1 - y) * o["mClo"]  # :50
        A = y * AOpe + (1 - y) * AClo  # :52
        CVal = y * COpe + (1 - y) * CClo  # :54
    VZer_flow = vZer * A  # TwoWayFlowElement.mo:83
    VZerCom_flow = VZer_flow / nCom  # DoorDiscretized.mo:52
    head, dpAB, dV, gai, dVAB, dVBA = [], [], [], [], [], []
    for i in range(nCom):
        h_i = rhoA * hAg[i] - rhoB * hBg[i]
        # DoorDiscretized.mo:64-66, regrouped (module docstring)
        dp_i = ((pA - pB) if dphi is None else dphi) + h_i
        if operable is None:
            v = power_law_05_np(CVal, dp_i, dp_t)  # DoorDiscretizedOpen.mo:27-35
        else:
            v = power_law_np(CVal, dp_i, m, dp_t)  # DoorDiscretizedOperable.mo:58-62
        g = smooth_heaviside_np(v, VZerCom_flow)  # DoorDiscretized.mo:69
        head.append(h_i)
        dpAB.append(dp_i)
        dV.append(v)
        gai.append(g)
        dVAB.append(v * g)  # DoorDiscretized.mo:70
        dVBA.append(-v * (1 - g))  # DoorDiscretized.mo:71
    VAB = sum(dVAB)  # DoorDiscretized.mo:74
    VBA = sum(dVBA)  # DoorDiscretized.mo:75
    return dict(
        rhoA=rhoA,
        rhoB=rhoB,
        hAg=np.array(hAg),
        hBg=np.array(hBg),
        head=np.array(head),
        dp=np.array(dpAB),
        dV=np.array(dV),
        dVAB=np.array(dVAB),
        dVBA=np.array(dVBA),
        mAB=rhoA * VAB,  # TwoWayFlowElement.mo:85,91
        mBA=rhoB * VBA,  # TwoWayFlowElement.mo:86,92
        # Per-compartment share of the port flows: rho_A dVAB[i] - rho_B dVBA[i]; its sum is
        # port_a1.m_flow - port_a2.m_flow = mAB_flow - mBA_flow exactly.
        m_com=rhoA * np.array(dVAB) - rhoB * np.array(dVBA),
        VZerCom=VZerCom_flow,
    )


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

NCOM = 10
GEOM = dict(nCom=NCOM, wOpe=1.0, hOpe=2.2, hA=1.5, hB=1.5, dp_t=0.01, vZer=0.001)
OPER = dict(LClo=20e-4, CDOpe=0.78, CDClo=0.78, CDCloRat=1.0, dpCloRat=4.0, mOpe=0.5, mClo=0.65)


def _open(med=AIR, CD=0.65, **kw):
    g = {**GEOM, **kw}
    return mbl_discretized_door(
        nCom=g["nCom"],
        wOpe=g["wOpe"],
        hOpe=g["hOpe"],
        hA=g["hA"],
        hB=g["hB"],
        CD=CD,
        m=0.5,
        dp_turbulent=g["dp_t"],
        vZer=g["vZer"],
        src=0,
        tgt=1,
        medium=med,
        kind="door",
    )


def _operable(med=AIR, **kw):
    g = {**GEOM, **kw}
    return mbl_discretized_operable_door(
        nCom=g["nCom"],
        wOpe=g["wOpe"],
        hOpe=g["hOpe"],
        hA=g["hA"],
        hB=g["hB"],
        y_key="y",
        dp_turbulent=g["dp_t"],
        vZer=g["vZer"],
        src=0,
        tgt=1,
        medium=med,
        kind="door",
        **OPER,
    )


def _drivers(pA, pB, TA, TB, XA=0.01, XB=0.01):
    """Full-node drivers for nodes (A, B); scalars or equal-shape arrays -> (..., 2)."""

    def pair(a, b):
        return torch.stack([torch.as_tensor(a, dtype=F64), torch.as_tensor(b, dtype=F64)], dim=-1)

    return {"p_abs": pair(pA, pB), "T": pair(TA, TB), "X_w": pair(XA, XB)}


def _flows(comp, head, dphi, drv):
    """Compartment flows for a potential difference dphi (per batch row) plus the head."""
    dphi = torch.as_tensor(dphi, dtype=F64)
    dp = dphi[..., None] + head(drv)
    return comp.flow(dp, drv)


# Stratified conditions spanning both flow directions, a neutral plane inside the door and
# pressure-driven one-way flow; moisture differs between the zones.
CASES = [
    # (dphi = p_A - p_B, TA, TB, XA, XB)
    (0.0, 296.15, 293.15, 0.008, 0.012),
    (0.0, 290.15, 295.15, 0.010, 0.010),
    (0.35, 298.15, 293.15, 0.015, 0.005),
    (-0.2, 293.15, 299.15, 0.0, 0.02),
    (3.0, 293.15, 297.15, 0.01, 0.01),
    (-4.0, 300.15, 290.15, 0.01, 0.01),
    (2e-3, 293.15, 293.15, 0.01, 0.01),
]


# ---------------------------------------------------------------------------------------
# 1. Per-compartment flows and heads against the transcription
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("med", [AIR, SIMPLE], ids=["Air", "SimpleAir"])
@pytest.mark.parametrize("CD", [0.65, 0.78])
def test_open_door_heads_and_compartment_flows_match_the_transcription(med, CD):
    comp, head = _open(med, CD=CD)
    for dphi, TA, TB, XA, XB in CASES:
        pA, pB = P_DEFAULT + 7.0 + dphi, P_DEFAULT + 7.0
        ref = door_discretized_np(pA, pB, TA, TB, XA, XB, med=med, CD=CD, **GEOM)
        drv = _drivers(pA, pB, TA, TB, XA, XB)
        np.testing.assert_allclose(head(drv).numpy(), ref["head"], rtol=RTOL, atol=1e-13)
        q = _flows(comp, head, pA - pB, drv).numpy()
        np.testing.assert_allclose(q, ref["m_com"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("y", [0.0, 0.4, 1.0])
def test_operable_door_compartment_flows_match_the_transcribed_blend(y):
    comp, head = _operable()
    for dphi, TA, TB, XA, XB in CASES:
        pA, pB = P_DEFAULT - 3.0 + dphi, P_DEFAULT - 3.0
        ref = door_discretized_np(
            pA, pB, TA, TB, XA, XB, med=AIR, operable={**OPER, "y": y}, **GEOM
        )
        drv = {**_drivers(pA, pB, TA, TB, XA, XB), "y": torch.tensor(y, dtype=F64)}
        q = _flows(comp, head, pA - pB, drv).numpy()
        np.testing.assert_allclose(q, ref["m_com"], rtol=RTOL, atol=ATOL)


def test_operable_door_fully_open_equals_open_door_with_the_same_discharge_coefficient():
    ocomp, ohead = _operable()
    comp, head = _open(CD=OPER["CDOpe"])  # CDOpe == CDClo here, see OPER
    for dphi, TA, TB, XA, XB in CASES:
        drv = _drivers(P_DEFAULT + dphi, P_DEFAULT, TA, TB, XA, XB)
        q1 = _flows(comp, head, dphi, drv)
        q2 = _flows(ocomp, ohead, dphi, {**drv, "y": torch.tensor(1.0, dtype=F64)})
        torch.testing.assert_close(q2, q1, rtol=RTOL, atol=ATOL)


def test_absolute_pressure_enters_through_the_inflow_densities():
    """TwoWayFlowElement.mo:73-81 evaluate density_pTX at port_a1.p/port_a2.p: the head and
    the flows scale with the absolute pressure, not with p_default."""
    comp, head = _open()
    for scale in (0.9, 1.0, 1.1):
        pA = pB = scale * P_DEFAULT
        ref = door_discretized_np(pA, pB, 297.15, 293.15, 0.01, 0.01, med=AIR, **GEOM)
        drv = _drivers(pA, pB, 297.15, 293.15)
        np.testing.assert_allclose(head(drv).numpy(), ref["head"], rtol=RTOL, atol=1e-13)
        np.testing.assert_allclose(
            _flows(comp, head, 0.0, drv).numpy(), ref["m_com"], rtol=RTOL, atol=ATOL
        )
    h09 = head(_drivers(0.9 * P_DEFAULT, 0.9 * P_DEFAULT, 297.15, 293.15))
    h11 = head(_drivers(1.1 * P_DEFAULT, 1.1 * P_DEFAULT, 297.15, 293.15))
    torch.testing.assert_close(h11 / 1.1, h09 / 0.9, rtol=1e-13, atol=1e-13)


def test_moisture_is_read_for_moist_air_and_ignored_for_simple_air():
    drv_dry = _drivers(P_DEFAULT, P_DEFAULT, 297.15, 293.15, 0.0, 0.0)
    drv_wet = _drivers(P_DEFAULT, P_DEFAULT, 297.15, 293.15, 0.02, 0.0)
    _, head_air = _open(AIR)
    assert not torch.equal(head_air(drv_dry), head_air(drv_wet))
    _, head_simple = _open(SIMPLE)
    torch.testing.assert_close(head_simple(drv_dry), head_simple(drv_wet), rtol=0, atol=0)
    # SimpleAir has no moisture driver at all.
    no_x = {k: v for k, v in drv_dry.items() if k != "X_w"}
    torch.testing.assert_close(head_simple(no_x), head_simple(drv_dry), rtol=0, atol=0)


def test_isothermal_door_with_equal_reference_heights_has_zero_head():
    comp, head = _open(hA=1.2, hB=1.2)
    drv = _drivers(P_DEFAULT + 2.0, P_DEFAULT + 2.0, 294.0, 294.0, 0.01, 0.01)
    assert torch.equal(head(drv), torch.zeros(NCOM, dtype=F64))
    # Every compartment then carries the same flow: the plain orifice on dA each.
    q = _flows(comp, head, 1.5, drv)
    torch.testing.assert_close(q, q[:1].expand_as(q), rtol=0, atol=0)


def test_unequal_reference_heights_give_a_uniform_hydrostatic_offset():
    """hA != hB, isothermal: rho g (hA - hB) on every compartment."""
    _, head = _open(hA=1.5, hB=1.0)
    drv = _drivers(P_DEFAULT, P_DEFAULT, 294.0, 294.0, 0.01, 0.01)
    rho = density_pTX_np(P_DEFAULT, 294.0, 0.01)
    np.testing.assert_allclose(head(drv).numpy(), rho * G_N * 0.5, rtol=1e-12)


# ---------------------------------------------------------------------------------------
# 2. Directional sums against MBL's smoothed mAB_flow / mBA_flow
# ---------------------------------------------------------------------------------------


def _c_smooth():
    """max over 0 < u <= 1 of u (1 - smoothHeaviside(u, 1)), from the source's polynomial
    (smoothHeaviside.mo:9-12), on a 2e6-point grid: 0.0705529 at u = 0.3018."""
    u = np.linspace(1e-9, 1.0, 2000001)
    dx = 0.5 * u
    g = np.clip(0.5 + dx * (1.875 + dx * dx * (-5 + 6 * dx * dx)), 0.0, 1.0)
    return float((u * (1 - g)).max())


def test_net_compartment_sum_equals_mbl_net_port_flow_exactly():
    comp, head = _open()
    for dphi, TA, TB, XA, XB in CASES:
        pA, pB = P_DEFAULT + dphi, P_DEFAULT
        ref = door_discretized_np(pA, pB, TA, TB, XA, XB, med=AIR, **GEOM)
        q = _flows(comp, head, pA - pB, _drivers(pA, pB, TA, TB, XA, XB)).numpy()
        scale = abs(ref["mAB"]) + abs(ref["mBA"])
        np.testing.assert_allclose(q.sum(), ref["mAB"] - ref["mBA"], rtol=0, atol=1e-13 * scale)


def test_directional_sums_equal_mbl_outside_the_smoothing_band():
    """No compartment has |dV_flow| < VZerCom_flow: smoothHeaviside is exactly 0 or 1
    (smoothHeaviside.mo:12 clips at |x| >= delta), so the plain signed sums ARE MBL's sums."""
    comp, head = _open()
    for dphi, TA, TB, XA, XB in CASES:
        pA, pB = P_DEFAULT + dphi, P_DEFAULT
        ref = door_discretized_np(pA, pB, TA, TB, XA, XB, med=AIR, **GEOM)
        assert np.all(np.abs(ref["dV"]) >= ref["VZerCom"]), "case must be outside the band"
        q = _flows(comp, head, pA - pB, _drivers(pA, pB, TA, TB, XA, XB)).numpy()
        np.testing.assert_allclose(q[q > 0].sum(), ref["mAB"], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(-q[q < 0].sum(), ref["mBA"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("u_target", [0.3, -0.3, 0.05, -0.8])
def test_directional_sums_differ_from_mbl_only_by_the_smoothing_width(u_target):
    """Place the neutral plane so that one compartment's volume flow is u_target *
    VZerCom_flow, inside MBL's smoothing band.

    MBL splits each compartment's volume flow with gaiFlo = smoothHeaviside(dV, VZerCom_flow)
    (DoorDiscretized.mo:69-71), VZerCom_flow = vZer*A/nCom (:52, TwoWayFlowElement.mo:83),
    and weights the A->B part with rho_A and the B->A part with rho_B
    (TwoWayFlowElement.mo:91-92). noodl's compartment edge carries exactly
    rho_A dVAB[i] - rho_B dVBA[i], so the NET sum is exact (test above), but the plain
    positive/negative sums of the edge flows use the sign of the edge flow, a hard split.
    For a compartment with dV = u delta (|u| < 1) the two differ by delta |u| (1 - g(|u|))
    times rho_B (u > 0) or rho_A (u < 0), since g(-u) = 1 - g(u). Hence

        |sum(q > 0) - mAB_flow| <= n_band * max(rho_A, rho_B) * VZerCom_flow * c,
        c = max_{0<u<=1} u (1 - g(u)) = 0.07055 at u = 0.30   (smoothHeaviside.mo:9-12),

    and likewise for mBA_flow. With the defaults (vZer = 1 mm/s, A = 2.2 m2, nCom = 10)
    delta = 2.2e-4 m3/s, so the bound is 1.86e-5 kg/s per in-band compartment against
    directional flows of 0.15/0.39 kg/s. Measured (one in-band compartment, u_target):
    +0.30 -> 1.858e-5 kg/s (the maximiser, equal to the bound to 4 digits), -0.30 -> 1.839e-5,
    +0.05 -> 6.06e-6, -0.80 -> 1.80e-6; the mBA_flow difference is the same number because
    the net sum is exact. The measured difference is asserted to be nonzero (the band is
    really exercised), to match the per-compartment formula above to 1e-9 and to stay below
    the bound (with a 1e-9 relative allowance for rounding at the maximiser).
    """
    comp, head = _open()
    TA, TB, XA, XB = 296.15, 293.15, 0.01, 0.01
    k = 6  # 7th compartment from the bottom
    pB = P_DEFAULT
    ref0 = door_discretized_np(pB, pB, TA, TB, XA, XB, med=AIR, **GEOM)
    delta = ref0["VZerCom"]
    # Invert the in-band polynomial for the dp that gives dV = u delta at compartment k.
    C = 0.65 * (1.0 * 2.2 / NCOM) * math.sqrt(2 / AIR.rho_default)
    target = u_target * delta
    lo, hi = -0.01, 0.01
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if power_law_05_np(C, mid, 0.01) < target:
            lo = mid
        else:
            hi = mid
    dp_k = 0.5 * (lo + hi)
    dphi = dp_k - ref0["head"][k]
    pA = pB + dphi
    ref = door_discretized_np(pA, pB, TA, TB, XA, XB, med=AIR, **GEOM)
    in_band = np.abs(ref["dV"]) < ref["VZerCom"]
    assert in_band[k] and in_band.sum() >= 1
    q = _flows(comp, head, pA - pB, _drivers(pA, pB, TA, TB, XA, XB)).numpy()
    d_ab = q[q > 0].sum() - ref["mAB"]
    d_ba = -q[q < 0].sum() - ref["mBA"]
    # Per-compartment prediction of the difference (derivation in the docstring).
    u = ref["dV"] / delta
    g = np.array([smooth_heaviside_np(x, delta) for x in ref["dV"]])
    pred_ab = np.where(u > 0, ref["rhoB"] * ref["dV"] * (1 - g), -ref["rhoA"] * ref["dV"] * g)
    pred_ab = np.where(in_band, pred_ab, 0.0).sum()
    np.testing.assert_allclose(d_ab, pred_ab, rtol=1e-9, atol=1e-18)
    np.testing.assert_allclose(d_ba, pred_ab, rtol=1e-9, atol=1e-18)  # net sum is exact
    bound = in_band.sum() * max(ref["rhoA"], ref["rhoB"]) * delta * _c_smooth()
    assert 0 < abs(d_ab) <= bound * (1 + 1e-9)
    assert abs(d_ab) < 1e-4 * (ref["mAB"] + ref["mBA"])


def test_smoothing_constant_is_what_the_docstring_states():
    assert abs(_c_smooth() - 0.0705529) < 1e-7


# ---------------------------------------------------------------------------------------
# 3. Geometry of the builders
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("nCom", [1, 4, 10, 25])
def test_compartment_areas_and_heights_add_up_to_the_door(nCom):
    wOpe, hOpe, hA, hB = 0.9, 2.1, 1.35, 0.8
    comp, head = mbl_discretized_door(
        nCom=nCom, wOpe=wOpe, hOpe=hOpe, hA=hA, hB=hB, src=0, tgt=1, medium=AIR, kind="d"
    )
    assert comp.dA.shape == (nCom,) and head.hAg.shape == (nCom,)
    np.testing.assert_allclose(comp.dA.sum().item(), wOpe * hOpe, rtol=1e-14)
    # Compartment centres z_i = (i - 0.5) dh above the door bottom, from hAg = g (hA - z_i).
    z = hA - head.hAg.numpy() / G_N
    zB = hB - head.hBg.numpy() / G_N
    np.testing.assert_allclose(z, zB, rtol=0, atol=1e-14)
    dh = hOpe / nCom
    np.testing.assert_allclose(np.diff(z), dh, rtol=1e-12)
    np.testing.assert_allclose(z[0] - dh / 2, 0.0, atol=1e-14)
    np.testing.assert_allclose(z[-1] + dh / 2, hOpe, rtol=1e-14)
    np.testing.assert_allclose(nCom * dh, hOpe, rtol=1e-14)
    assert torch.equal(comp.src, torch.zeros(nCom, dtype=torch.long))
    assert torch.equal(comp.tgt, torch.ones(nCom, dtype=torch.long))
    assert head.kind == comp.kind == "d"


def test_operable_builder_areas_add_up_to_the_open_and_closed_door():
    comp, _ = _operable()
    AOpe = 1.0 * 2.2
    AClo = (
        OPER["CDClo"] / OPER["CDCloRat"] * OPER["LClo"] * OPER["dpCloRat"] ** (0.5 - OPER["mClo"])
    )
    # Per-compartment open area AOpe/nCom (DoorDiscretizedOperable.mo:47) sums to the door's.
    np.testing.assert_allclose((comp.AOpe / comp.nCom).sum().item(), AOpe, rtol=1e-14)
    for y in (0.0, 0.3, 1.0):
        A = comp.face_area({"y": torch.tensor(y, dtype=F64)})
        np.testing.assert_allclose(A.numpy(), y * AOpe + (1 - y) * AClo, rtol=1e-14)


def test_builder_accepts_endpoints_from_the_network():
    net = Network(dtype=F64)
    net.add_node("A")
    net.add_node("B")
    for _ in range(3):
        net.add_edge("A", "B", kind="door")
    src, tgt = net.endpoints("door")
    comp, head = mbl_discretized_door(nCom=3, src=src, tgt=tgt, medium=AIR, kind="door")
    assert torch.equal(comp.src, src) and torch.equal(head.tgt, tgt)
    with pytest.raises(ValueError, match="nCom"):
        mbl_discretized_door(nCom=4, src=src, tgt=tgt, medium=AIR, kind="door")


# ---------------------------------------------------------------------------------------
# 4. Batches
# ---------------------------------------------------------------------------------------


def test_head_and_flows_broadcast_over_leading_batch_dims():
    comp, head = _open()
    gen = np.random.default_rng(11)
    shape = (2, 3)
    dphi = gen.uniform(-2, 2, size=shape)
    TA = gen.uniform(288, 302, size=shape)
    TB = gen.uniform(288, 302, size=shape)
    XA = gen.uniform(0, 0.02, size=shape)
    XB = gen.uniform(0, 0.02, size=shape)
    pB = P_DEFAULT + gen.uniform(-50, 50, size=shape)
    pA = pB + dphi
    drv = _drivers(pA, pB, TA, TB, XA, XB)
    h = head(drv)
    q = _flows(comp, head, pA - pB, drv)
    assert h.shape == (2, 3, NCOM) and q.shape == (2, 3, NCOM)
    for idx in np.ndindex(*shape):
        ref = door_discretized_np(
            pA[idx], pB[idx], TA[idx], TB[idx], XA[idx], XB[idx], med=AIR, **GEOM
        )
        np.testing.assert_allclose(h[idx].numpy(), ref["head"], rtol=RTOL, atol=1e-13)
        np.testing.assert_allclose(q[idx].numpy(), ref["m_com"], rtol=RTOL, atol=ATOL)


def test_operable_opening_can_vary_per_batch_row():
    comp, head = _operable()
    y = torch.tensor([[0.0], [0.5], [1.0]], dtype=F64)
    drv = {**_drivers([P_DEFAULT + 0.3] * 3, [P_DEFAULT] * 3, [297.0] * 3, [293.0] * 3), "y": y}
    q = _flows(comp, head, torch.full((3,), (P_DEFAULT + 0.3) - P_DEFAULT, dtype=F64), drv).numpy()
    for i, yi in enumerate([0.0, 0.5, 1.0]):
        ref = door_discretized_np(
            P_DEFAULT + 0.3,
            P_DEFAULT,
            297.0,
            293.0,
            0.01,
            0.01,
            med=AIR,
            operable={**OPER, "y": yi},
            **GEOM,
        )
        np.testing.assert_allclose(q[i], ref["m_com"], rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------------------
# 5. Derivatives
# ---------------------------------------------------------------------------------------


def test_gradcheck_compartment_flow_wrt_dp_across_the_bands():
    comp, head = _open(nCom=6)
    # outside, inside the dp_turbulent band, inside the smoothHeaviside band (|dp| < ~1e-4),
    # zero, both signs.
    dp = torch.tensor([[-2.0, -0.004, -3e-5, 0.0, 2e-5, 0.3]], dtype=F64, requires_grad=True)
    drv = _drivers(P_DEFAULT, P_DEFAULT + 0.1, 297.0, 293.0, 0.012, 0.004)
    assert torch.autograd.gradcheck(lambda x: comp.flow(x, drv), (dp,), eps=1e-9, atol=1e-8)


def test_gradcheck_flow_and_head_wrt_temperature_pressure_and_moisture_drivers():
    comp, head = _open(nCom=4)
    T = torch.tensor([[297.0, 293.0], [293.0, 293.0]], dtype=F64, requires_grad=True)
    p = torch.tensor(
        [[P_DEFAULT + 0.2, P_DEFAULT], [P_DEFAULT, P_DEFAULT]], dtype=F64, requires_grad=True
    )
    X = torch.tensor([[0.012, 0.004], [0.01, 0.01]], dtype=F64, requires_grad=True)
    dphi = torch.tensor([0.2, 1e-3], dtype=F64)

    def f(t, pp, xx):
        drv = {"T": t, "p_abs": pp, "X_w": xx}
        return torch.cat([_flows(comp, head, dphi, drv), head(drv)], dim=-1)

    assert torch.autograd.gradcheck(f, (T, p, X), eps=1e-6, atol=1e-8)


def test_gradcheck_operable_wrt_dp_and_y():
    comp, _ = _operable(nCom=3)
    dp = torch.tensor([[0.5, 0.003, -1e-5]], dtype=F64, requires_grad=True)
    y = torch.tensor([[0.3]], dtype=F64, requires_grad=True)
    drv = _drivers(P_DEFAULT, P_DEFAULT, 297.0, 293.0)
    assert torch.autograd.gradcheck(
        lambda x, yy: comp.flow(x, {**drv, "y": yy}), (dp, y), eps=1e-9, atol=1e-8
    )


def test_dflow_positive_and_linear_init_per_edge():
    comp, head = _open(nCom=5)
    drv = _drivers(P_DEFAULT, P_DEFAULT, 297.0, 293.0)
    c, k = comp.linear_init(drv)
    assert c.shape == (5,) and k.shape == (5,)
    assert torch.equal(c, torch.zeros(5, dtype=F64))
    assert (k > 0).all()
    dp = torch.tensor([-1.0, -2e-5, 0.0, 3e-5, 0.02], dtype=F64)
    assert (comp.dflow(dp, drv) > 0).all()


# ---------------------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------------------


def test_missing_drivers_raise_naming_key_and_kind():
    comp, head = _open()
    drv = _drivers(P_DEFAULT, P_DEFAULT, 297.0, 293.0)
    for key in ("p_abs", "T", "X_w"):
        partial = {k: v for k, v in drv.items() if k != key}
        with pytest.raises(KeyError, match=f"'{key}'.*door|door.*'{key}'"):
            head(partial)
        with pytest.raises(KeyError, match=f"'{key}'"):
            comp.flow(torch.zeros(NCOM, dtype=F64), partial)
    ocomp, _ = _operable()
    with pytest.raises(KeyError, match="'y'"):
        ocomp.flow(torch.zeros(NCOM, dtype=F64), drv)


def test_short_node_driver_and_wrong_width_raise():
    comp = MBLDoorCompartment(src=[0, 0], tgt=[3, 3], medium=AIR, dA=[0.5, 0.5], kind="d")
    drv = _drivers(P_DEFAULT, P_DEFAULT, 297.0, 293.0)
    with pytest.raises(ValueError, match="node"):
        comp.flow(torch.zeros(2, dtype=F64), drv)
    comp2 = MBLDoorCompartment(src=[0, 0], tgt=[1, 1], medium=AIR, dA=[0.5, 0.5], kind="d")
    with pytest.raises(ValueError, match="columns"):
        comp2.flow(torch.zeros(3, dtype=F64), drv)
    with pytest.raises(ValueError, match="hAg"):
        DoorCompartmentHead("d", src=[0, 0], tgt=[1, 1], hAg=[1.0], hBg=[1.0, 2.0], medium=AIR)


def test_head_is_a_drive():
    from noodl.drives import check_drive_signature

    _, head = _open()
    check_drive_signature(head, where="test")
    assert isinstance(MBLDoorCompartmentOperable, type)


# ---------------------------------------------------------------------------------------
# 7. Inside a PotentialFlowLayer
# ---------------------------------------------------------------------------------------


def _two_zone_net(nCom):
    net = Network(dtype=F64)
    net.add_node("A")
    net.add_node("B")
    for _ in range(nCom):
        net.add_edge("A", "B", kind="door")
    return net


def test_head_drive_is_wired_by_kind_in_a_two_boundary_layer():
    """Boundary A and B at given pressures and temperatures: the layer's dp for the door
    kind is phi_A - phi_B plus the head, and its flows are the transcription's."""
    net = _two_zone_net(NCOM)
    src, tgt = net.endpoints("door")
    comp, head = mbl_discretized_door(
        nCom=NCOM,
        wOpe=1.0,
        hOpe=2.2,
        hA=1.5,
        hB=1.5,
        src=src,
        tgt=tgt,
        medium=AIR,
        kind="door",
    )
    layer = PotentialFlowLayer(net, "air", [comp], drives=[head], boundary=["A", "B"])
    iA, iB = net.node_index("A"), net.node_index("B")
    for dphi, TA, TB, XA, XB in CASES:
        phi = torch.zeros(2, dtype=F64)
        phi[iA], phi[iB] = 0.4 + dphi, 0.4
        pA, pB = P_DEFAULT + phi[iA].item(), P_DEFAULT + phi[iB].item()
        drv = {
            "p_abs": P_DEFAULT + phi,
            "T": torch.tensor([TA, TB], dtype=F64)[[iA, iB]],
            "X_w": torch.tensor([XA, XB], dtype=F64)[[iA, iB]],
        }
        dphi_l = (phi[iA] - phi[iB]).item()
        ref = door_discretized_np(pA, pB, TA, TB, XA, XB, med=AIR, dphi=dphi_l, **GEOM)
        sl = layer.kind_slice("door")
        dp = layer.dp(phi, drv)[..., sl]
        np.testing.assert_allclose(dp.numpy(), ref["dp"], rtol=1e-12, atol=1e-13)
        q = layer.flows(phi, drv)[..., sl]
        el, sl2 = layer.element_for("door")
        assert el is comp and sl2 == sl
        np.testing.assert_allclose(q.numpy(), ref["m_com"], rtol=RTOL, atol=ATOL)


def test_discretised_door_solves_inside_a_potential_layer_and_gradients_reach_T():
    """out (boundary) -- orifice -- A (interior, warm) == door (nCom edges) == B (boundary)."""
    from noodl.elements.mbl import mbl_orifice

    net = Network(dtype=F64)
    for n in ("out", "A", "B"):
        net.add_node(n)
    net.add_edge("out", "A", kind="ori")
    for _ in range(NCOM):
        net.add_edge("A", "B", kind="door")
    src, tgt = net.endpoints("door")
    comp, head = mbl_discretized_door(
        nCom=NCOM,
        wOpe=1.0,
        hOpe=2.2,
        hA=1.5,
        hB=1.5,
        src=src,
        tgt=tgt,
        medium=AIR,
        kind="door",
    )
    ori = mbl_orifice(0.05, dp_turbulent=0.01, rho_default=AIR.rho_default, kind="ori")
    layer = PotentialFlowLayer(net, "air", [ori, comp], drives=[head], boundary=["out", "B"])
    idx = {n: net.node_index(n) for n in ("out", "A", "B")}
    phi_b = torch.tensor([0.5, 0.0], dtype=F64)  # out, B

    def drivers(TA):
        T = (torch.zeros(3, dtype=F64) + 293.15).index_put(
            (torch.tensor([idx["A"]]),), TA.reshape(1)
        )
        return {
            "T": T,
            "p_abs": torch.full((3,), P_DEFAULT, dtype=F64),
            "X_w": torch.full((3,), 0.01, dtype=F64),
        }

    TA0 = torch.tensor(299.15, dtype=F64)
    _, q = layer.solve(phi_b, drivers(TA0), differentiable=False, atol=1e-13, rtol=1e-13)
    s_ori, s_door = layer.kind_slice("ori"), layer.kind_slice("door")
    q_door = q[..., s_door]
    # Mass balance at A; two-way exchange through the stratified door.
    torch.testing.assert_close(q[..., s_ori].sum(), q_door.sum(), rtol=1e-10, atol=1e-12)
    assert (q_door > 0).any() and (q_door < 0).any()
    # Warm A: outflow at the top compartments, inflow at the bottom.
    assert q_door[..., -1] > 0 and q_door[..., 0] < 0

    def top_flow(TA):
        _, qq = layer.solve(phi_b, drivers(TA), atol=1e-13, rtol=1e-13)
        return qq[..., s_door][..., -1]

    TA = TA0.clone().requires_grad_(True)
    top_flow(TA).backward()
    h = 1e-5
    with torch.no_grad():
        fd = (top_flow(TA0 + h) - top_flow(TA0 - h)) / (2 * h)
    torch.testing.assert_close(TA.grad, fd, rtol=1e-5, atol=1e-10)

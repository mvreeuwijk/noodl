"""Tests for `noodl.elements.mbl.door`: MBL's `DoorOpen` and `DoorOperable` as two directional
noodl edges.

The NumPy reference functions below are transcribed directly from the Modelica Buildings
Library (MBL) v13.0.0 (commit 55abf579598ca81cae0a82f337350375958e6722) and the Modelica
Standard Library (MSL) v4.1.0 sources, never by importing `noodl.elements.mbl.door` itself:

* `Airflow/Multizone/BaseClasses/Door.mo:43-54,64-65` (conTP, rho_default, the two port flows),
* `Airflow/Multizone/DoorOpen.mo:16-35,39-69` (CVal, kT, m_flow_turbulent, pressure and
  buoyancy terms),
* `Airflow/Multizone/DoorOperable.mo:35-59,68-109` (the open/closed blend),
* `Airflow/Multizone/BaseClasses/powerLaw05.mo`, `powerLawFixedM.mo` (the pressure law),
* `Fluid/BaseClasses/FlowModels/basicFlowFunction_dp.mo:14-23` (the buoyancy law),
* `Airflow/Multizone/EffectiveAirLeakageArea.mo:3-5` (the closed-door reference),
* MSL `Modelica/Constants.mo:38` (g_n), `Media/IdealGases/Common/SingleGasesData.mo:5,49,59`
  (R_NASA_2002, Air.MM, Air.R_s), MBL `Media/Air.mo:45` (dStp).

Wiring: side A holds `port_a1` and `port_b2`, side B holds `port_b1` and
`port_a2`. Edge `ab` is path 1 and carries `port_a1.m_flow`; edge `ba` is path 2 signed from A
to B, i.e. `-port_a2.m_flow = port_b2.m_flow` (`Door.mo:79`).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from noodl.elements.mbl import (
    MBLDoorOpen,
    MBLDoorOperable,
    mbl_door_pair,
    mbl_ela,
    mbl_operable_door_pair,
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
MM_AIR = 0.0289651159  # SingleGasesData.mo:49
R_S_AIR = R_NASA_2002 / MM_AIR  # SingleGasesData.mo:59
D_STP = 1.2  # Buildings/Media/Air.mo:45
CON_TP = D_STP * R_S_AIR  # BaseClasses/Door.mo:43-44
GAMMA = 1.5  # DoorOpen.mo:16, DoorOperable.mo:35


def _coeffs_np(m):
    """DoorOpen.mo:18-25 (DoorOperable.mo:37-44 evaluate the same per exponent)."""
    a = GAMMA
    b = 1 / 8 * m**2 - 3 * GAMMA - 3 / 2 * m + 35.0 / 8
    c = -1 / 4 * m**2 + 3 * GAMMA + 5 / 2 * m - 21.0 / 4
    d = 1 / 8 * m**2 - GAMMA - m + 15.0 / 8
    return a, b, c, d


def _power_law_05_np(C, dp, a, b, c, d, dp_t, sqrt_dp_t):
    """BaseClasses/powerLaw05.mo:20-29."""
    dp = np.asarray(dp, dtype=np.float64)
    pi = dp / dp_t
    pi2 = pi * pi
    out = np.empty_like(dp)
    for i, x in np.ndenumerate(dp):
        if x >= dp_t:
            out[i] = C * math.sqrt(x)
        elif x <= -dp_t:
            out[i] = -C * math.sqrt(-x)
        else:
            p, p2 = pi[i], pi2[i]
            out[i] = C * p * sqrt_dp_t * (a + p2 * (b + p2 * (c + p2 * d)))
    return out


def _power_law_fixed_m_np(C, dp, m, a, b, c, d, dp_t):
    """BaseClasses/powerLawFixedM.mo:21-30."""
    dp = np.asarray(dp, dtype=np.float64)
    out = np.empty_like(dp)
    for i, x in np.ndenumerate(dp):
        p = x / dp_t
        p2 = p * p
        if x >= dp_t:
            out[i] = C * x**m
        elif x <= -dp_t:
            out[i] = -C * (-x) ** m
        else:
            out[i] = C * dp_t**m * p * (a + p2 * (b + p2 * (c + p2 * d)))
    return out


def _pressure_law_np(C, dp, m, dp_t):
    """DoorOpen.mo:39-59: powerLaw05 if isEqual(m, 0.5, 1e-10), else powerLawFixedM."""
    a, b, c, d = _coeffs_np(m)
    if abs(m - 0.5) <= 1e-10:
        return _power_law_05_np(C, dp, a, b, c, d, dp_t, math.sqrt(dp_t))
    return _power_law_fixed_m_np(C, dp, m, a, b, c, d, dp_t)


def _basic_flow_function_dp_np(dp, k, m_flow_turbulent):
    """Fluid/BaseClasses/FlowModels/basicFlowFunction_dp.mo:14-23."""
    dp = np.asarray(dp, dtype=np.float64)
    dp_turbulent = (m_flow_turbulent / k) ** 2
    out = np.empty_like(dp)
    for i, x in np.ndenumerate(dp):
        dpNorm = x / dp_turbulent
        dpNormSq = dpNorm**2
        if abs(x) > dp_turbulent:
            out[i] = np.sign(x) * k * math.sqrt(abs(x))
        else:
            out[i] = (
                (1.40625 + (0.15625 * dpNormSq - 0.5625) * dpNormSq) * m_flow_turbulent * dpNorm
            )
    return out


def _door_open_np(dp, TA, TB, *, wOpe, hOpe, CD, m, dp_t, med):
    """DoorOpen.mo + Door.mo: returns (port_a1.m_flow, port_b2.m_flow)."""
    rho_default = med.rho_default  # Door.mo:49-54
    AOpe = wOpe * hOpe  # Door.mo:41
    CVal = CD * AOpe * math.sqrt(2 / rho_default)  # DoorOpen.mo:27
    kT = rho_default * CD * AOpe / 3 * math.sqrt(G_N / (med.T_default * CON_TP) * hOpe)  # :29-30
    m_flow_turbulent = CVal * rho_default * math.sqrt(dp_t)  # DoorOpen.mo:33-34
    VABp = _pressure_law_np(CVal, dp, m, dp_t)  # DoorOpen.mo:39-59
    dT = np.asarray(TA, dtype=np.float64) - np.asarray(TB, dtype=np.float64)
    mABt = _basic_flow_function_dp_np(CON_TP * dT, kT, m_flow_turbulent)  # DoorOpen.mo:66-69
    a1 = rho_default * VABp / 2 + mABt  # Door.mo:64
    b2 = rho_default * VABp / 2 - mABt  # Door.mo:65
    return a1, b2


def _door_operable_np(
    dp, TA, TB, y, *, wOpe, hOpe, CDOpe, mOpe, LClo, mClo, dpCloRat, CDCloRat, dp_t, med
):
    """DoorOperable.mo + Door.mo: returns (port_a1.m_flow, port_b2.m_flow)."""
    rho_default = med.rho_default
    AOpe = wOpe * hOpe
    AClo = LClo * dpCloRat ** (0.5 - mClo)  # DoorOperable.mo:46-47
    CVal1 = CDOpe * AOpe * math.sqrt(2 / rho_default)  # DoorOperable.mo:48-50
    CVal2 = CDCloRat * AClo * math.sqrt(2 / rho_default)
    kT = rho_default * CDOpe * AOpe / 3 * math.sqrt(G_N / (med.T_default * CON_TP) * hOpe)  # :53-54
    m_flow_turbulent = CVal1 * rho_default * math.sqrt(dp_t)  # DoorOperable.mo:57-58
    V1 = _pressure_law_np(CVal1, dp, mOpe, dp_t)  # DoorOperable.mo:68-88
    a2, b2_, c2, d2 = _coeffs_np(mClo)
    V2 = _power_law_fixed_m_np(CVal2, dp, mClo, a2, b2_, c2, d2, dp_t)  # DoorOperable.mo:91-99
    VABp = y * V1 + (1 - y) * V2  # DoorOperable.mo:100
    dT = np.asarray(TA, dtype=np.float64) - np.asarray(TB, dtype=np.float64)
    mABt = y * _basic_flow_function_dp_np(CON_TP * dT, kT, m_flow_turbulent)  # :106-109
    return rho_default * VABp / 2 + mABt, rho_default * VABp / 2 - mABt


def _ela_np(dp, *, L, dpRat, CDRat, m, dp_t, med):
    """EffectiveAirLeakageArea.mo:3-5 through Coefficient_V_flow (m_flow = rho_default*V_flow)."""
    C = L * CDRat * math.sqrt(2.0 / med.rho_default) * dpRat ** (0.5 - m)
    a, b, c, d = _coeffs_np(m)
    return med.rho_default * _power_law_fixed_m_np(C, dp, m, a, b, c, d, dp_t)


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

DOOR = dict(wOpe=1.0, hOpe=2.2, CD=0.78, m=0.5, dp_t=0.01)  # ThreeRoomsContam.mo's door
OPER = dict(
    wOpe=1.0,
    hOpe=2.2,
    CDOpe=0.78,
    mOpe=0.5,
    LClo=20e-4,
    mClo=0.65,
    dpCloRat=4.0,
    CDCloRat=1.0,
    dp_t=0.01,
)

DP_GRID = np.concatenate(
    [np.linspace(-5.0, 5.0, 41), [-0.01, -0.005, -1e-4, 0.0, 1e-4, 0.005, 0.01]]
)
DT_GRID = np.array([-10.0, -3.0, -1e-3, 0.0, 1e-3, 0.5, 3.0, 10.0])


def _pair(med=AIR, **kw):
    p = {**DOOR, **kw}
    return mbl_door_pair(
        src=[0],
        tgt=[1],
        medium=med,
        wOpe=p["wOpe"],
        hOpe=p["hOpe"],
        CD=p["CD"],
        m=p["m"],
        dp_turbulent=p["dp_t"],
        kind="door",
    )


def _opair(**kw):
    p = {**OPER, **kw}
    return mbl_operable_door_pair(
        src=[0],
        tgt=[1],
        medium=AIR,
        y_key="y",
        LClo=p["LClo"],
        wOpe=p["wOpe"],
        hOpe=p["hOpe"],
        CDOpe=p["CDOpe"],
        mOpe=p["mOpe"],
        mClo=p["mClo"],
        dpCloRat=p["dpCloRat"],
        CDCloRat=p["CDCloRat"],
        dp_turbulent=p["dp_t"],
        kind="door",
    )


def _grid():
    dp, dT = np.meshgrid(DP_GRID, DT_GRID, indexing="ij")
    TB = 293.15
    TA = TB + dT
    return dp.reshape(-1), TA.reshape(-1), np.full(dp.size, TB)


def _torch_inputs(dp, TA, TB):
    dp_t = torch.tensor(dp, dtype=F64).unsqueeze(-1)  # (N, 1 edge)
    T = torch.stack([torch.tensor(TA, dtype=F64), torch.tensor(TB, dtype=F64)], dim=-1)
    return dp_t, T  # T is full-node (N, 2)


# ---------------------------------------------------------------------------------------
# 1. DoorOpen against the NumPy transcription
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("med", [AIR, SIMPLE], ids=["Air", "SimpleAir"])
@pytest.mark.parametrize("m", [0.5, 0.62])
def test_door_open_both_directions_match_the_transcription_on_the_dp_dT_grid(med, m):
    dp, TA, TB = _grid()
    ab, ba = _pair(med, m=m)
    dp_t, T = _torch_inputs(dp, TA, TB)
    a1, b2 = _door_open_np(dp, TA, TB, wOpe=1.0, hOpe=2.2, CD=0.78, m=m, dp_t=0.01, med=med)
    q_ab = ab.flow(dp_t, {"T": T})[..., 0].numpy()
    q_ba = ba.flow(dp_t, {"T": T})[..., 0].numpy()
    np.testing.assert_allclose(q_ab, a1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(q_ba, b2, rtol=RTOL, atol=ATOL)


def test_pure_exchange_at_dp_zero_equal_and_opposite_warm_side_out_on_path_one():
    ab, ba = _pair()
    dp = torch.zeros(3, 1, dtype=F64)
    T = torch.tensor([[303.15, 293.15], [293.15, 293.15], [283.15, 293.15]], dtype=F64)
    q_ab = ab.flow(dp, {"T": T})[..., 0]
    q_ba = ba.flow(dp, {"T": T})[..., 0]
    # Door.mo:64-65 at VABp = 0: port_a1.m_flow = +mABt, port_b2.m_flow = -mABt.
    torch.testing.assert_close(q_ab, -q_ba, rtol=0.0, atol=0.0)
    assert q_ab[0] > 0 and q_ab[2] < 0 and q_ab[1] == 0
    a1, _ = _door_open_np(
        np.zeros(1),
        np.array([303.15]),
        np.array([293.15]),
        wOpe=1.0,
        hOpe=2.2,
        CD=0.78,
        m=0.5,
        dp_t=0.01,
        med=AIR,
    )
    np.testing.assert_allclose(q_ab[0].item(), a1[0], rtol=RTOL, atol=ATOL)


def test_equal_temperatures_split_the_orifice_flow_equally_between_the_two_paths():
    ab, ba = _pair()
    dp = torch.tensor(DP_GRID, dtype=F64).unsqueeze(-1)
    T = torch.full((dp.shape[0], 2), 297.0, dtype=F64)
    q_ab = ab.flow(dp, {"T": T})[..., 0]
    q_ba = ba.flow(dp, {"T": T})[..., 0]
    torch.testing.assert_close(q_ab, q_ba, rtol=0.0, atol=0.0)
    # Net flow is the Orifice law rho_default * CD*A*sqrt(2/rho) * dp^0.5 (DoorOpen.mo:27).
    C = 0.78 * 2.2 * math.sqrt(2 / AIR.rho_default)
    net = AIR.rho_default * _pressure_law_np(C, DP_GRID, 0.5, 0.01)
    np.testing.assert_allclose((q_ab + q_ba).numpy(), net, rtol=RTOL, atol=ATOL)


def test_moisture_and_absolute_pressure_are_not_inputs():
    """DoorOpen.mo:66-69 read only the two inflow TEMPERATURES; Door.mo:64-65 use the fixed
    rho_default. No moisture or pressure driver is needed."""
    ab, _ = _pair()
    dp = torch.tensor([[1.0]], dtype=F64)
    T = torch.tensor([[295.0, 293.0]], dtype=F64)
    q1 = ab.flow(dp, {"T": T})
    q2 = ab.flow(
        dp,
        {
            "T": T,
            "X_w": torch.tensor([[0.5, 0.0]], dtype=F64),
            "p": torch.tensor([[1e5, 2e5]], dtype=F64),
        },
    )
    torch.testing.assert_close(q1, q2, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------------------
# 2. DoorOperable
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("y", [0.0, 0.3, 1.0])
def test_operable_door_matches_the_transcribed_blend(y):
    dp, TA, TB = _grid()
    ab, ba = _opair()
    dp_t, T = _torch_inputs(dp, TA, TB)
    a1, b2 = _door_operable_np(
        dp,
        TA,
        TB,
        y,
        wOpe=1.0,
        hOpe=2.2,
        CDOpe=0.78,
        mOpe=0.5,
        LClo=20e-4,
        mClo=0.65,
        dpCloRat=4.0,
        CDCloRat=1.0,
        dp_t=0.01,
        med=AIR,
    )
    drv = {"T": T, "y": torch.tensor(y, dtype=F64)}
    np.testing.assert_allclose(ab.flow(dp_t, drv)[..., 0].numpy(), a1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ba.flow(dp_t, drv)[..., 0].numpy(), b2, rtol=RTOL, atol=ATOL)


def test_operable_door_closed_is_the_effective_air_leakage_area_law():
    dp, TA, TB = _grid()
    ab, ba = _opair()
    dp_t, T = _torch_inputs(dp, TA, TB)
    drv = {"T": T, "y": torch.tensor(0.0, dtype=F64)}
    q_ab = ab.flow(dp_t, drv)[..., 0]
    q_ba = ba.flow(dp_t, drv)[..., 0]
    torch.testing.assert_close(q_ab, q_ba, rtol=0.0, atol=0.0)  # no buoyancy term at y=0
    ela = _ela_np(dp, L=20e-4, dpRat=4.0, CDRat=1.0, m=0.65, dp_t=0.01, med=AIR)
    np.testing.assert_allclose((q_ab + q_ba).numpy(), ela, rtol=RTOL, atol=ATOL)
    # ... and the noodl ELA element built the way Validation/DoorOpenClosed.mo builds `lea`.
    lea = mbl_ela(20e-4, m=0.65, dp_turbulent=0.01, rho_default=AIR.rho_default)
    torch.testing.assert_close(q_ab + q_ba, lea.flow(dp_t[..., 0]), rtol=RTOL, atol=ATOL)


def test_operable_door_open_equals_door_open():
    dp, TA, TB = _grid()
    oab, oba = _opair()
    ab, ba = _pair()
    dp_t, T = _torch_inputs(dp, TA, TB)
    drv = {"T": T, "y": torch.tensor(1.0, dtype=F64)}
    torch.testing.assert_close(oab.flow(dp_t, drv), ab.flow(dp_t, drv), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(oba.flow(dp_t, drv), ba.flow(dp_t, drv), rtol=RTOL, atol=ATOL)


def test_operable_door_opening_can_vary_per_batch_row():
    ab, _ = _opair()
    dp = torch.tensor([[2.0], [2.0], [2.0]], dtype=F64)
    T = torch.tensor([[296.0, 293.0]] * 3, dtype=F64)
    y = torch.tensor([[0.0], [0.3], [1.0]], dtype=F64)
    q = ab.flow(dp, {"T": T, "y": y})[..., 0].numpy()
    for i, yi in enumerate([0.0, 0.3, 1.0]):
        a1, _ = _door_operable_np(
            np.array([2.0]),
            np.array([296.0]),
            np.array([293.0]),
            yi,
            wOpe=1.0,
            hOpe=2.2,
            CDOpe=0.78,
            mOpe=0.5,
            LClo=20e-4,
            mClo=0.65,
            dpCloRat=4.0,
            CDCloRat=1.0,
            dp_t=0.01,
            med=AIR,
        )
        np.testing.assert_allclose(q[i], a1[0], rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------------------
# 3. Several doors in one element, batched
# ---------------------------------------------------------------------------------------


def test_multi_edge_element_broadcasts_over_leading_batch_dims():
    """Two doors (nodes 0->2 and 1->2) with per-edge geometry, batch shape (2, 3)."""
    wOpe = torch.tensor([0.9, 1.2], dtype=F64)
    hOpe = torch.tensor([2.1, 2.4], dtype=F64)
    ab = MBLDoorOpen(
        direction="ab",
        src=[0, 1],
        tgt=[2, 2],
        medium=AIR,
        wOpe=wOpe,
        hOpe=hOpe,
        CD=0.65,
        m=0.5,
        dp_turbulent=0.01,
        kind="d_ab",
    )
    ba = MBLDoorOpen(
        direction="ba",
        src=[0, 1],
        tgt=[2, 2],
        medium=AIR,
        wOpe=wOpe,
        hOpe=hOpe,
        CD=0.65,
        m=0.5,
        dp_turbulent=0.01,
        kind="d_ba",
    )
    gen = np.random.default_rng(3)
    dp = gen.uniform(-4, 4, size=(2, 3, 2))
    T = gen.uniform(285, 305, size=(2, 3, 3))
    drv = {"T": torch.tensor(T, dtype=F64)}
    q_ab = ab.flow(torch.tensor(dp, dtype=F64), drv).numpy()
    q_ba = ba.flow(torch.tensor(dp, dtype=F64), drv).numpy()
    assert q_ab.shape == (2, 3, 2)
    for e, (s, t) in enumerate([(0, 2), (1, 2)]):
        a1, b2 = _door_open_np(
            dp[..., e],
            T[..., s],
            T[..., t],
            wOpe=[0.9, 1.2][e],
            hOpe=[2.1, 2.4][e],
            CD=0.65,
            m=0.5,
            dp_t=0.01,
            med=AIR,
        )
        np.testing.assert_allclose(q_ab[..., e], a1, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(q_ba[..., e], b2, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------------------
# 4. Derivatives
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["ab", "ba"])
def test_gradcheck_wrt_dp_inside_and_outside_the_band_including_zero(direction):
    ab, ba = _pair()
    el = ab if direction == "ab" else ba
    dp = torch.tensor(
        [[-3.0], [-0.007], [0.0], [0.004], [0.02], [2.5]], dtype=F64, requires_grad=True
    )
    T = torch.tensor([[294.0, 293.15]] * 6, dtype=F64)
    assert torch.autograd.gradcheck(lambda x: el.flow(x, {"T": T}), (dp,), eps=1e-7, atol=1e-8)


@pytest.mark.parametrize("direction", ["ab", "ba"])
def test_gradcheck_wrt_temperature_driver_inside_and_outside_the_buoyancy_band(direction):
    ab, ba = _pair()
    el = ab if direction == "ab" else ba
    dp = torch.tensor([[0.0], [0.3], [-1.0], [0.0]], dtype=F64)
    # dT = 0 (band centre), 2e-4 K (inside the buoyancy regularisation band), +-5 K outside.
    T = torch.tensor(
        [[293.15, 293.15], [293.1502, 293.15], [298.15, 293.15], [288.15, 293.15]],
        dtype=F64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(lambda t: el.flow(dp, {"T": t}), (T,), eps=1e-6, atol=1e-8)


def test_gradcheck_operable_wrt_dp_T_and_y():
    ab, _ = _opair()
    dp = torch.tensor([[0.0], [0.005], [1.5]], dtype=F64, requires_grad=True)
    T = torch.tensor(
        [[295.0, 293.15], [293.15, 293.15], [290.0, 293.15]], dtype=F64, requires_grad=True
    )
    y = torch.tensor([[0.3], [0.7], [0.0]], dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, t, yy: ab.flow(x, {"T": t, "y": yy}), (dp, T, y), eps=1e-7, atol=1e-8
    )


def test_dflow_is_positive_and_linear_init_is_per_edge():
    wOpe = torch.tensor([0.9, 1.2], dtype=F64)
    ab = MBLDoorOpen(direction="ab", src=[0, 1], tgt=[2, 2], medium=AIR, wOpe=wOpe, kind="x")
    ba = MBLDoorOpen(direction="ba", src=[0, 1], tgt=[2, 2], medium=AIR, wOpe=wOpe, kind="y")
    T = torch.tensor([300.0, 290.0, 293.15], dtype=F64)
    for el in (ab, ba):
        c, k = el.linear_init({"T": T})
        assert c.shape == (2,) and k.shape == (2,)
        zero = torch.zeros(2, dtype=F64)
        torch.testing.assert_close(c, el.flow(zero, {"T": T}), rtol=0.0, atol=0.0)
        torch.testing.assert_close(k, el.dflow(zero, {"T": T}), rtol=0.0, atol=0.0)
        assert (k > 0).all()
        dp = torch.tensor([-2.0, 0.003], dtype=F64)
        assert (el.dflow(dp, {"T": T}) > 0).all()
    # Opposite signs of the pure-exchange offset on the two paths.
    torch.testing.assert_close(ab.linear_init({"T": T})[0], -ba.linear_init({"T": T})[0])


# ---------------------------------------------------------------------------------------
# 5. Pair constructors and validation
# ---------------------------------------------------------------------------------------


def test_pair_constructors_share_parameters_and_name_the_kinds():
    ab, ba = _pair()
    assert (ab.direction, ba.direction) == ("ab", "ba")
    assert (ab.kind, ba.kind) == ("door_ab", "door_ba")
    assert isinstance(ab, MBLDoorOpen) and isinstance(ba, MBLDoorOpen)
    oab, oba = _opair()
    assert isinstance(oab, MBLDoorOperable) and isinstance(oba, MBLDoorOperable)
    assert (oab.kind, oba.kind) == ("door_ab", "door_ba")
    assert oab.y_key == oba.y_key == "y"


def test_bad_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        MBLDoorOpen(direction="up", src=[0], tgt=[1], medium=AIR)


def test_mismatched_endpoints_raise():
    with pytest.raises(ValueError, match="src"):
        MBLDoorOpen(direction="ab", src=[0, 1], tgt=[1], medium=AIR)


def test_missing_temperature_driver_raises_naming_key_and_kind():
    ab, _ = _pair()
    with pytest.raises(KeyError, match="'T'.*door_ab|door_ab.*'T'"):
        ab.flow(torch.zeros(1, dtype=F64), {})


def test_missing_opening_driver_raises_naming_key():
    ab, _ = _opair()
    with pytest.raises(KeyError, match="'y'"):
        ab.flow(torch.zeros(1, dtype=F64), {"T": torch.tensor([293.0, 294.0], dtype=F64)})


def test_temperature_driver_too_short_raises():
    ab = MBLDoorOpen(direction="ab", src=[0], tgt=[3], medium=AIR)
    with pytest.raises(ValueError, match="node"):
        ab.flow(torch.zeros(1, dtype=F64), {"T": torch.tensor([293.0, 294.0], dtype=F64)})


def test_dp_of_wrong_width_raises():
    ab = MBLDoorOpen(direction="ab", src=[0, 1], tgt=[2, 2], medium=AIR)
    with pytest.raises(ValueError, match="columns"):
        ab.flow(torch.zeros(1, dtype=F64), {"T": torch.full((3,), 293.0, dtype=F64)})


# ---------------------------------------------------------------------------------------
# 6. Inside a PotentialFlowLayer
# ---------------------------------------------------------------------------------------


def _layer():
    """out (boundary) -- orifice -- A == door pair == B -- orifice -- out."""
    from noodl.elements.mbl import mbl_orifice

    net = Network(dtype=F64)
    for n in ("out", "A", "B"):
        net.add_node(n)
    net.add_edge("out", "A", kind="ori_A")
    net.add_edge("A", "B", kind="door_ab")
    net.add_edge("A", "B", kind="door_ba")
    net.add_edge("B", "out", kind="ori_B")
    src, tgt = net.endpoints("door_ab")
    ab, ba = mbl_door_pair(src=src, tgt=tgt, medium=AIR, wOpe=1.0, hOpe=2.2, CD=0.78, kind="door")
    oA = mbl_orifice(0.01, dp_turbulent=0.01, rho_default=AIR.rho_default, kind="ori_A")
    oB = mbl_orifice(0.02, dp_turbulent=0.01, rho_default=AIR.rho_default, kind="ori_B")
    return net, PotentialFlowLayer(net, "air", [oA, ab, ba, oB], boundary=["out"])


def test_door_pair_solves_inside_a_potential_layer_and_the_gradient_reaches_T():
    net, layer = _layer()
    idx = {n: net.node_index(n) for n in ("out", "A", "B")}
    phi_b = torch.tensor([5.0], dtype=F64)

    def T_of(TA):
        T = torch.zeros(3, dtype=F64) + 293.15
        return T.index_put((torch.tensor([idx["A"]]),), TA.reshape(1))

    TA0 = torch.tensor(298.15, dtype=F64)
    phi, q = layer.solve(phi_b, {"T": T_of(TA0)}, differentiable=False, atol=1e-13, rtol=1e-13)
    s = {k: layer._kind_slices[k] for k in ("ori_A", "door_ab", "door_ba", "ori_B")}
    q_ab = q[..., s["door_ab"][0]]
    q_ba = q[..., s["door_ba"][0]]
    # Mass balance at A and B; the door exchanges air both ways (A is warmer: path 1 out).
    torch.testing.assert_close(q[..., s["ori_A"][0]], q_ab + q_ba, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(q[..., s["ori_B"][0]], q_ab + q_ba, rtol=1e-10, atol=1e-12)
    assert q_ab > 0 and q_ba < 0

    def net_flow(TA):
        _, qq = layer.solve(phi_b, {"T": T_of(TA)}, atol=1e-13, rtol=1e-13)
        return qq[..., s["door_ab"][0]]

    TA = TA0.clone().requires_grad_(True)
    net_flow(TA).backward()
    h = 1e-5
    with torch.no_grad():
        fd = (net_flow(TA0 + h) - net_flow(TA0 - h)) / (2 * h)
    torch.testing.assert_close(TA.grad, fd, rtol=1e-5, atol=1e-10)

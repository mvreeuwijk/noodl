"""MBL's large-opening doors, ``DoorOpen`` and ``DoorOperable``, as two directional noodl edges.

Transcribed from Modelica Buildings Library (MBL) v13.0.0
(commit 55abf579598ca81cae0a82f337350375958e6722), with constants from the Modelica Standard
Library (MSL) v4.1.0.

The MBL model
-------------
Both doors extend ``Airflow/Multizone/BaseClasses/Door.mo`` (NOT ``TwoWayFlowElement.mo``,
which in v13 is the base of ``DoorDiscretized`` only). A door has four ports and two flow
paths: path 1 from ``port_a1`` to ``port_b1`` and path 2 from ``port_a2`` to ``port_b2``. The
two port mass flows are (``Door.mo:64-65``)

    port_a1.m_flow = rho_default * VABp_flow / 2 + mABt_flow
    port_b2.m_flow = rho_default * VABp_flow / 2 - mABt_flow

with ``rho_default = Medium.density(setState_pTX(T_default, p_default, X_default))``
(``Door.mo:49-54``), a fixed number: the directional flows use the DEFAULT density, not the
inflow density ``TwoWayFlowElement.mo:91-92`` uses for the discretised door. The static
pressure term is a regularised power law on ``dp = port_a1.p - port_a2.p``
(``DoorOpen.mo:39-59``) and the buoyancy term a regularised square root of the temperature
difference of the two INFLOWING streams (``DoorOpen.mo:66-69``)

    mABt_flow = basicFlowFunction_dp(k = kT,
                                     dp = conTP * (T(state_a1_inflow) - T(state_a2_inflow)),
                                     m_flow_turbulent = m_flow_turbulent)
    conTP = Buildings.Media.Air.dStp * SingleGasesData.Air.R_s                 (Door.mo:43-44)
    kT = rho_default * CD * AOpe / 3 * sqrt(g_n / (Medium.T_default * conTP) * hOpe)
                                                                          (DoorOpen.mo:29-31)
    m_flow_turbulent = CVal * rho_default * sqrt(dp_turbulent)            (DoorOpen.mo:33-35)
    CVal = CD * AOpe * sqrt(2 / rho_default),   AOpe = wOpe * hOpe        (DoorOpen.mo:27,
                                                                           Door.mo:41)

``conTP`` uses ``Buildings.Media.Air``'s ``dStp`` and dry air's ``R_s`` whatever the door's
medium is (it is a ``constant`` of ``Door.mo``); ``T_default`` and ``rho_default`` are the
door medium's own.

What the flows depend on
------------------------
``state_a1_inflow = setState_phX(port_a1.p, inStream(port_a1.h_outflow), ...)``
(``Fluid/Interfaces/PartialFourPortInterface.mo:76-78``, ``:82-84`` for ``a2``), and
``Medium.temperature`` of that state is independent of pressure for all three supported media
(``Media/Air.mo:823-840``: "the temperature is independent of the pressure"; the two ideal
gases' ``h(T)`` has no pressure term). With ``port_a1`` on side A and ``port_a2`` on side B
(the wiring the reader enforces), ``inStream`` of each is that zone's own
enthalpy, so the two temperatures are the zone temperatures ``T_A`` and ``T_B``. The door
flows are therefore functions of ``(dp, T_A, T_B)`` alone: NOT of moisture (which enters only
through ``h -> T``, and the zone temperature is supplied directly) and NOT of absolute
pressure. No moisture or pressure driver is read, and no layer plumbing was needed: the zone
temperatures arrive as the full-node driver ``drivers[T_key]`` exactly as
``UpstreamDensityPowerLaw`` receives ``drivers[rho_key]``. They are fixed within one potential
solve, so the Newton Jacobian needs only ``d flow / d dp`` -- the buoyancy term does not
depend on ``dp`` at all.

The noodl mapping
-----------------
One door is two noodl edges between the same two nodes, both oriented from side A (``src``)
to side B (``tgt``), with the same ``dp = p_A - p_B``:

* direction ``"ab"`` (path 1) returns ``port_a1.m_flow``, positive from A to B;
* direction ``"ba"`` (path 2) returns ``port_b2.m_flow = -port_a2.m_flow`` (``Door.mo:79``),
  which is ``-mBA_flow`` -- also positive from A to B.

At ``dp = 0`` the two edges carry ``+mABt`` and ``-mABt``: a pure exchange with zero net flow.
With equal temperatures ``mABt = 0`` exactly and each edge carries half the orifice flow.
The unused ``Door.mo`` equations (``VAB_flow``/``VBA_flow`` at ``:68-69`` and the velocities)
only report averaged velocities and do not feed back into the port flows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from noodl.elements.base import Element
from noodl.elements.mbl.media import _R_AIR, MBLMedium

Tensor = torch.Tensor

_GAMMA = 1.5  # DoorOpen.mo:16-17, DoorOperable.mo:35-36
_G_N = 9.80665  # MSL Modelica/Constants.mo:38
_D_STP_AIR = 1.2  # Buildings/Media/Air.mo:45 (dStp)
# BaseClasses/Door.mo:43-44: conTP = Buildings.Media.Air.dStp *
# Modelica.Media.IdealGases.Common.SingleGasesData.Air.R_s (SingleGasesData.mo:59).
_CON_TP = _D_STP_AIR * _R_AIR

_DIRECTIONS = ("ab", "ba")


def _f64(value) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float64)


def _power_law(C: Tensor, dp: Tensor, m: Tensor, dp_turbulent: float) -> Tensor:
    """Volume flow of ``BaseClasses/powerLawFixedM.mo:21-30`` (``powerLaw05.mo:20-29`` for
    ``m = 0.5``, the same expression with ``dp^0.5`` written as ``sqrt(dp)``):
    ``C sign(dp) |dp|^m`` for ``|dp| >= dp_turbulent``, else
    ``C dp_turbulent^m pi (a + pi^2 (b + pi^2 (c + pi^2 d)))`` with ``pi = dp/dp_turbulent``
    and ``a..d`` from ``DoorOpen.mo:18-25`` (``DoorOperable.mo:37-44``).

    ``dp_safe`` keeps the unselected sharp branch away from the singular ``|dp|^m`` at 0.
    """
    a = _GAMMA
    b = 1 / 8 * m**2 - 3 * _GAMMA - 3 / 2 * m + 35.0 / 8
    c = -1 / 4 * m**2 + 3 * _GAMMA + 5 / 2 * m - 21.0 / 4
    d = 1 / 8 * m**2 - _GAMMA - m + 15.0 / 8
    mask = dp.abs() < dp_turbulent
    dp_safe = torch.where(mask, torch.full_like(dp, dp_turbulent), dp.abs())
    sharp = C * torch.sign(dp) * dp_safe**m
    pi = dp / dp_turbulent
    pi2 = pi * pi
    inner = C * dp_turbulent**m * pi * (a + pi2 * (b + pi2 * (c + pi2 * d)))
    return torch.where(mask, inner, sharp)


def _basic_flow_function_dp(dp: Tensor, k: Tensor, m_flow_turbulent: Tensor) -> Tensor:
    """``Fluid/BaseClasses/FlowModels/basicFlowFunction_dp.mo:14-23``:
    ``dp_turbulent = (m_flow_turbulent/k)^2``; ``sign(dp) k sqrt(|dp|)`` for
    ``|dp| > dp_turbulent``, else
    ``(1.40625 + (0.15625 dpNorm^2 - 0.5625) dpNorm^2) m_flow_turbulent dpNorm``.
    """
    dp_turbulent = (m_flow_turbulent / k) ** 2
    dp_norm = dp / dp_turbulent
    dp_norm_sq = dp_norm**2
    sharp_mask = dp.abs() > dp_turbulent
    dp_safe = torch.where(sharp_mask, dp.abs(), dp_turbulent)
    sharp = torch.sign(dp) * k * torch.sqrt(dp_safe)
    inner = (1.40625 + (0.15625 * dp_norm_sq - 0.5625) * dp_norm_sq) * m_flow_turbulent * dp_norm
    return torch.where(sharp_mask, sharp, inner)


class _MBLDoor(Element):
    """Shared plumbing of the two door laws: endpoints, the temperature driver, direction."""

    def __init__(
        self,
        *,
        direction: str,
        src,
        tgt,
        medium: MBLMedium,
        dp_turbulent: float,
        T_key: str,
        kind: str,
    ) -> None:
        super().__init__(kind)
        name = type(self).__name__
        if direction not in _DIRECTIONS:
            raise ValueError(
                f"{name} (kind {kind!r}): direction must be 'ab' or 'ba', got {direction!r}"
            )
        src_t = torch.as_tensor(src).to(torch.long)
        tgt_t = torch.as_tensor(tgt).to(torch.long)
        if src_t.shape != tgt_t.shape or src_t.ndim != 1:
            raise ValueError(
                f"{name} (kind {kind!r}): src and tgt must be 1-D with one node position per "
                f"door, got shapes {tuple(src_t.shape)} and {tuple(tgt_t.shape)}"
            )
        if src_t.numel() and int(torch.minimum(src_t.min(), tgt_t.min())) < 0:
            raise ValueError(
                f"{name} (kind {kind!r}): src/tgt must be non-negative node positions, got "
                f"src={src_t.tolist()} tgt={tgt_t.tolist()}"
            )
        if not float(dp_turbulent) > 0.0:
            raise ValueError(
                f"{name} (kind {kind!r}): dp_turbulent must be strictly positive, got "
                f"{dp_turbulent!r}"
            )
        self.register_buffer("src", src_t)
        self.register_buffer("tgt", tgt_t)
        self.direction = direction
        self.medium = medium
        self.rho_default = float(medium.rho_default)
        self.T_default = float(medium.T_default)
        self.dp_turbulent = float(dp_turbulent)
        self.T_key = str(T_key)

    # ------------------------------------------------------------------ drivers
    def _check_width(self, dp: Tensor) -> None:
        n_edges = self.src.numel()
        width = dp.shape[-1] if dp.ndim else 1
        if n_edges > 1 and width != n_edges:
            raise ValueError(
                f"{type(self).__name__} (kind {self.kind!r}): dp has {width} columns but "
                f"this element covers {n_edges} edges"
            )

    def _delta_T(self, drivers: Mapping[str, Tensor] | None) -> Tensor:
        """``T(state_a1_inflow) - T(state_a2_inflow) = T_A - T_B`` (``DoorOpen.mo:68``),
        gathered from the full-node driver ``drivers[T_key]`` of shape ``(..., n_nodes)``."""
        if drivers is None or self.T_key not in drivers:
            raise KeyError(
                f"{type(self).__name__} (kind {self.kind!r}): driver {self.T_key!r} not "
                f"found; it needs the full-node zone temperature vector"
            )
        T = torch.as_tensor(drivers[self.T_key])
        n_nodes = T.shape[-1] if T.ndim else 0
        needed = int(torch.maximum(self.src.max(), self.tgt.max())) + 1 if self.src.numel() else 0
        if n_nodes < needed:
            raise ValueError(
                f"{type(self).__name__} (kind {self.kind!r}): driver {self.T_key!r} has "
                f"{n_nodes} node values but this element's endpoints reach node index "
                f"{needed - 1}"
            )
        return T[..., self.src] - T[..., self.tgt]

    # ------------------------------------------------------------------ law
    def _terms(self, dp: Tensor, drivers) -> tuple[Tensor, Tensor]:
        """``(VABp_flow, mABt_flow)``; implemented by each door."""
        raise NotImplementedError

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        """``Door.mo:64`` (``"ab"``) or ``Door.mo:65`` (``"ba"``), mass flow in kg/s,
        positive from ``src`` (side A) to ``tgt`` (side B)."""
        self._check_width(dp)
        V_p, m_t = self._terms(dp, drivers)
        half = self.rho_default * V_p / 2
        return half + m_t if self.direction == "ab" else half - m_t

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        """Tangent at ``dp = 0`` per edge: ``c`` is the pure-exchange flow ``+-mABt`` and
        ``k = rho_default/2 dVABp/ddp > 0``. Evaluated on a per-edge zero (not the base
        class's 0-d zero, whose autograd slope would sum over a multi-edge element)."""
        zero = torch.zeros(max(self.src.numel(), 1), dtype=self._dtype())
        return self.flow(zero, drivers), self.dflow(zero, drivers)


class MBLDoorOpen(_MBLDoor):
    """``Buildings.Airflow.Multizone.DoorOpen``: one direction of the always-open door.

    Parameters default to MBL's: ``wOpe = 0.9``, ``hOpe = 2.1``, ``dp_turbulent = 0.01``
    (``BaseClasses/Door.mo:19-28``), ``CD = 0.65``, ``m = 0.5`` (``DoorOpen.mo:8-12``).
    ``wOpe``/``hOpe``/``CD``/``m`` may be scalars or one value per door. See the module
    docstring for the equations and the edge mapping.
    """

    def __init__(
        self,
        *,
        direction: str,
        src,
        tgt,
        medium: MBLMedium,
        wOpe=0.9,
        hOpe=2.1,
        CD=0.65,
        m=0.5,
        dp_turbulent: float = 0.01,
        T_key: str = "T",
        kind: str = "door",
        learnable: bool = False,
    ) -> None:
        super().__init__(
            direction=direction,
            src=src,
            tgt=tgt,
            medium=medium,
            dp_turbulent=dp_turbulent,
            T_key=T_key,
            kind=kind,
        )
        self.wOpe = self._param(_f64(wOpe), learnable)
        self.hOpe = self._param(_f64(hOpe), learnable)
        self.CD = self._param(_f64(CD), learnable)
        self.m = self._param(_f64(m), learnable)

    def _terms(self, dp: Tensor, drivers) -> tuple[Tensor, Tensor]:
        rho = self.rho_default
        AOpe = self.wOpe * self.hOpe  # Door.mo:41
        CVal = self.CD * AOpe * math.sqrt(2 / rho)  # DoorOpen.mo:27
        kT = (
            rho
            * self.CD
            * AOpe
            / 3
            * torch.sqrt(  # DoorOpen.mo:29-31
                _G_N / (self.T_default * _CON_TP) * self.hOpe
            )
        )
        m_flow_turbulent = CVal * rho * math.sqrt(self.dp_turbulent)  # DoorOpen.mo:33-35
        V_p = _power_law(CVal, dp, self.m, self.dp_turbulent)  # DoorOpen.mo:39-59
        m_t = _basic_flow_function_dp(  # DoorOpen.mo:66-69
            _CON_TP * self._delta_T(drivers), kT, m_flow_turbulent
        )
        return V_p, m_t


class MBLDoorOperable(_MBLDoor):
    """``Buildings.Airflow.Multizone.DoorOperable``: the door blended between open and closed
    by the opening signal ``y`` read from ``drivers[y_key]``.

    ``DoorOperable.mo:100``: ``VABp = y VOpe + (1 - y) VClo``; ``:106-109``:
    ``mABt = y basicFlowFunction_dp(...)`` (the closed crack has no buoyancy term). The open
    law is ``DoorOpen``'s with ``CDOpe``/``mOpe`` (``:48-49,53-59,68-88``); the closed law is
    the effective-leakage-area crack ``CClo = CDCloRat LClo dpCloRat^(0.5 - mClo)
    sqrt(2/rho_default)`` with exponent ``mClo`` (``:46-50,91-99``), always through
    ``powerLawFixedM`` and with the door's ``dp_turbulent``. Defaults are MBL's: ``CDOpe =
    0.65``, ``mOpe = 0.5``, ``mClo = 0.65``, ``dpCloRat = 4``, ``CDCloRat = 1``
    (``DoorOperable.mo:8-28``); ``LClo`` has no default.

    ``drivers[y_key]`` is a 0-d scalar or a tensor whose last dimension is 1 or this
    element's door count, broadcasting against ``dp``. MBL declares ``y`` with ``min=0,
    max=1`` (``:30``) and disclaims accuracy for ``0 < y < 1``; values are used as given.
    """

    def __init__(
        self,
        *,
        direction: str,
        src,
        tgt,
        medium: MBLMedium,
        y_key: str,
        LClo,
        wOpe=0.9,
        hOpe=2.1,
        CDOpe=0.65,
        mOpe=0.5,
        mClo=0.65,
        dpCloRat=4.0,
        CDCloRat=1.0,
        dp_turbulent: float = 0.01,
        T_key: str = "T",
        kind: str = "door",
        learnable: bool = False,
    ) -> None:
        super().__init__(
            direction=direction,
            src=src,
            tgt=tgt,
            medium=medium,
            dp_turbulent=dp_turbulent,
            T_key=T_key,
            kind=kind,
        )
        self.y_key = str(y_key)
        self.wOpe = self._param(_f64(wOpe), learnable)
        self.hOpe = self._param(_f64(hOpe), learnable)
        self.CDOpe = self._param(_f64(CDOpe), learnable)
        self.mOpe = self._param(_f64(mOpe), learnable)
        self.LClo = self._param(_f64(LClo), learnable)
        self.mClo = self._param(_f64(mClo), learnable)
        self.dpCloRat = self._param(_f64(dpCloRat), learnable)
        self.CDCloRat = self._param(_f64(CDCloRat), learnable)

    def _y(self, drivers) -> Tensor:
        if drivers is None or self.y_key not in drivers:
            raise KeyError(
                f"MBLDoorOperable (kind {self.kind!r}): driver {self.y_key!r} not found; it "
                f"needs the door opening signal y (0 closed, 1 open)"
            )
        y = torch.as_tensor(drivers[self.y_key])
        n_edges = self.src.numel()
        if y.ndim and y.shape[-1] not in (1, n_edges):
            raise ValueError(
                f"MBLDoorOperable (kind {self.kind!r}): driver {self.y_key!r} has last "
                f"dimension {y.shape[-1]}; expected 1 or the door count {n_edges}"
            )
        return y

    def _terms(self, dp: Tensor, drivers) -> tuple[Tensor, Tensor]:
        rho = self.rho_default
        y = self._y(drivers)
        AOpe = self.wOpe * self.hOpe  # Door.mo:41
        AClo = self.LClo * self.dpCloRat ** (0.5 - self.mClo)  # DoorOperable.mo:46-47
        CVal_ope = self.CDOpe * AOpe * math.sqrt(2 / rho)  # DoorOperable.mo:48-51
        CVal_clo = self.CDCloRat * AClo * math.sqrt(2 / rho)
        kT = (
            rho
            * self.CDOpe
            * AOpe
            / 3
            * torch.sqrt(  # DoorOperable.mo:53-55
                _G_N / (self.T_default * _CON_TP) * self.hOpe
            )
        )
        m_flow_turbulent = CVal_ope * rho * math.sqrt(self.dp_turbulent)  # :57-59
        V_ope = _power_law(CVal_ope, dp, self.mOpe, self.dp_turbulent)  # :68-88
        V_clo = _power_law(CVal_clo, dp, self.mClo, self.dp_turbulent)  # :91-99
        V_p = y * V_ope + (1 - y) * V_clo  # DoorOperable.mo:100
        m_t = y * _basic_flow_function_dp(  # DoorOperable.mo:106-109
            _CON_TP * self._delta_T(drivers), kT, m_flow_turbulent
        )
        return V_p, m_t


def mbl_door_pair(*, kind: str = "door", **kwargs) -> tuple[MBLDoorOpen, MBLDoorOpen]:
    """Both edges of one ``DoorOpen`` (or of several, with per-door ``src``/``tgt``): the
    ``"ab"`` edge (path 1) with kind ``f"{kind}_ab"`` and the ``"ba"`` edge (path 2) with kind
    ``f"{kind}_ba"``, sharing every keyword argument of :class:`MBLDoorOpen`."""
    return (
        MBLDoorOpen(direction="ab", kind=f"{kind}_ab", **kwargs),
        MBLDoorOpen(direction="ba", kind=f"{kind}_ba", **kwargs),
    )


def mbl_operable_door_pair(
    *, kind: str = "door", **kwargs
) -> tuple[MBLDoorOperable, MBLDoorOperable]:
    """Both edges of one ``DoorOperable``; see :func:`mbl_door_pair`."""
    return (
        MBLDoorOperable(direction="ab", kind=f"{kind}_ab", **kwargs),
        MBLDoorOperable(direction="ba", kind=f"{kind}_ba", **kwargs),
    )

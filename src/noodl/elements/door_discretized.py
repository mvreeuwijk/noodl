"""Two-way flow through a large opening, discretised into ``nCom`` height compartments with a
hydrostatic-head drive, transcribed from the Modelica Buildings Library's
``DoorDiscretizedOpen``/``DoorDiscretizedOperable``.

Transcribed from Modelica Buildings Library (MBL) v13.0.0
(commit 55abf579598ca81cae0a82f337350375958e6722), with constants from the Modelica Standard
Library (MSL) v4.1.0.

The MBL model
-------------
Both doors extend ``Airflow/Multizone/BaseClasses/DoorDiscretized.mo``, which extends
``TwoWayFlowElementBuoyancy.mo`` (geometry ``wOpe``, ``hOpe``, ``hA``, ``hB`` only) and
``TwoWayFlowElement.mo``. The opening is cut into ``nCom`` compartments of height
``dh = hOpe/nCom`` (``DoorDiscretized.mo:21``). For compartment ``i = 1..nCom``
(``DoorDiscretized.mo:34-40,62-72``)

    hAg[i] = g_n (hA - (i - 0.5) dh),       hBg[i] = g_n (hB - (i - 0.5) dh)
    pA[i]  = port_a1.p + rho_a1_inflow hAg[i]
    pB[i]  = port_a2.p + rho_a2_inflow hBg[i]
    dpAB[i] = pA[i] - pB[i]
    gaiFlo[i]   = smoothHeaviside(dV_flow[i], VZerCom_flow)
    dVAB_flow[i] =  dV_flow[i] gaiFlo[i]
    dVBA_flow[i] = -dV_flow[i] (1 - gaiFlo[i])

so ``hA``/``hB`` are the heights of the zones' reference-pressure points above the door
BOTTOM (the compartment centres sit at ``(i - 0.5) dh``). The port flows are
(``DoorDiscretized.mo:74-75``, ``TwoWayFlowElement.mo:83,91-92``)

    VAB_flow = sum(dVAB_flow),  VBA_flow = sum(dVBA_flow)
    port_a1.m_flow = rho_a1_inflow VAB_flow,   port_a2.m_flow = rho_a2_inflow VBA_flow
    VZer_flow = vZer A,  VZerCom_flow = VZer_flow / nCom            (vZer = 0.001 m/s)

The compartment law is a VOLUME flow with the fixed ``rho_default`` inside its coefficient:

* ``DoorDiscretizedOpen.mo:10-36``: ``m = 0.5``, ``A = wOpe hOpe``, ``dA = A/nCom``
  (``DoorDiscretized.mo:60``), ``CVal = CD dA sqrt(2/rho_default)``,
  ``dV_flow[i] = powerLaw05(CVal, dpAB[i], a, b, c, d, dp_turbulent)``.
* ``DoorDiscretizedOperable.mo:35-63``: ``AOpe = wOpe hOpe``,
  ``AClo = CDClo/CDCloRat LClo dpCloRat^(0.5 - mClo)``, ``CClo = CDClo AClo/nCom
  sqrt(2/rho_default)``, ``COpe = CDOpe AOpe/nCom sqrt(2/rho_default)``, then with the opening
  signal ``y``: ``m = y mOpe + (1-y) mClo``, ``A = y AOpe + (1-y) AClo``,
  ``CVal = y COpe + (1-y) CClo`` and ``dV_flow[i] = powerLaw(CVal, dpAB[i], m, dp_turbulent)``.
  ``CDClo`` enters ``CClo`` twice (once through ``AClo``); this is MBL's formula and is kept.

The densities (``TwoWayFlowElement.mo:72-81``)
----------------------------------------------
``rho_a1_inflow = density_pTX(p = port_a1.p, T = T(state_a1_inflow), X_w = Xi_a1_inflow[1])``
(``X_w = 0`` when the medium has no moisture, ``nXi == 0``), and likewise for ``a2``. This is
``Buildings.Utilities.Psychrometrics.Functions.density_pTX`` (``density_pTX.mo:11-16``,
``p / ((R_air (1 - X_w) + R_h2o X_w) T)``) for EVERY medium -- including
``Modelica.Media.Air.SimpleAir``, whose own density uses the CODATA gas constant: the
discretised door does not call ``Medium.density``. It is evaluated at the ACTUAL port pressure,
not at ``p_default``. With ``port_a1`` on side A and ``port_a2`` on side B,
``inStream`` at each port is that zone's own state whatever the flow direction, so
``rho_A = density_pTX(p_A, T_A, X_w,A)`` and ``rho_B = density_pTX(p_B, T_B, X_w,B)``.

Elements see only ``dp``, so the absolute port pressures are read from the full-node driver
``drivers[p_key]`` (default ``"p_abs"``, Pa, shape ``(..., n_nodes)``), the temperatures from
``drivers[T_key]`` and, for a medium with moisture only, the water mass fractions from
``drivers[Xw_key]``. The port pressure enters ONLY through these two densities (and so through
the compartment heads and the mass-flow weighting); ``p_A - p_B`` itself is the layer's
potential difference. Nothing else in these models reads moisture.

The noodl mapping
-----------------
Compartment ``i`` of a door is one noodl edge from side A (``src``) to side B (``tgt``):

* :class:`DoorCompartmentHead` is a drive (``noodl.drives`` protocol) returning
  ``rho_A hAg[i] - rho_B hBg[i]``, so that the layer's ``dp = (phi_A - phi_B) + head`` is
  ``dpAB[i]`` (with ``phi = p - p_ref``, any fixed gauge reference -- the Modelica reader uses
  the first boundary's pressure -- the port pressures' difference is the potentials').
* :class:`DoorCompartment` / :class:`OperableDoorCompartment` return that
  compartment's share of the port mass flows, positive from A to B:

      q_i = rho_A dVAB_flow[i] - rho_B dVBA_flow[i]
          = dV_flow[i] (rho_A gaiFlo[i] + rho_B (1 - gaiFlo[i])),

  i.e. MBL's inflow-density selection per compartment, smoothed exactly as MBL smooths it.
  The sum over the compartments is ``port_a1.m_flow - port_a2.m_flow = mAB_flow - mBA_flow``
  EXACTLY. The plain sums of the positive and of the negative edge flows equal MBL's
  ``mAB_flow``/``mBA_flow`` except for compartments with ``|dV_flow[i]| < VZerCom_flow``,
  where they differ by at most ``max(rho_A, rho_B) VZerCom_flow * 0.0706`` per compartment
  (``test_door_discretized.py`` derives the bound from ``smoothHeaviside.mo:9-12``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from noodl.elements.base import Element, band_edges, memo
from noodl.elements.door import (
    _power_law_at,
    _power_law_coeffs,
    _power_law_slope_at,
    _slope_is_exact,
)
from noodl.elements.media import _R_AIR, _R_H2O, AirMedium

Tensor = torch.Tensor

_G_N = 9.80665  # MSL Modelica/Constants.mo:38
_M_FIXED = 0.5  # DoorDiscretizedOpen.mo:10


def _f64(value) -> Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value, dtype=torch.float64)


def _density_pTX(p: Tensor, T: Tensor, X_w) -> Tensor:
    """``Utilities/Psychrometrics/Functions/density_pTX.mo:11-16``:
    ``d = p / ((Air.R_s (1 - X_w) + H2O.R_s X_w) T)``."""
    R = _R_AIR * (1 - X_w) + _R_H2O * X_w
    return p / (R * T)


def _smooth_heaviside(x: Tensor, delta: Tensor) -> Tensor:
    """``Utilities/Math/Functions/smoothHeaviside.mo:9-12``:
    ``max(0, min(1, 0.5 + dx (1.875 + dx^2 (-5 + 6 dx^2))))`` with ``dx = 0.5 x/delta``."""
    dx = 0.5 * x / delta
    xpow2 = dx * dx
    return torch.clamp(0.5 + dx * (1.875 + xpow2 * (-5 + 6 * xpow2)), 0.0, 1.0)


def _endpoints(src, tgt, *, where: str) -> tuple[Tensor, Tensor]:
    src_t = torch.as_tensor(src).to(torch.long).reshape(-1)
    tgt_t = torch.as_tensor(tgt).to(torch.long).reshape(-1)
    if src_t.shape != tgt_t.shape:
        raise ValueError(
            f"{where}: src and tgt must have one node position per compartment edge, got "
            f"shapes {tuple(src_t.shape)} and {tuple(tgt_t.shape)}"
        )
    if src_t.numel() and int(torch.minimum(src_t.min(), tgt_t.min())) < 0:
        raise ValueError(f"{where}: src/tgt must be non-negative node positions")
    return src_t, tgt_t


class _InflowDensities:
    """``rho_A``, ``rho_B`` of ``TwoWayFlowElement.mo:72-81`` from full-node drivers."""

    src: Tensor
    tgt: Tensor
    medium: AirMedium
    kind: str
    T_key: str
    Xw_key: str
    p_key: str

    def _node_driver(self, drivers: Mapping[str, Tensor] | None, key: str, what: str) -> Tensor:
        name = type(self).__name__
        if drivers is None or key not in drivers:
            raise KeyError(
                f"{name} (kind {self.kind!r}): driver {key!r} not found; it needs the "
                f"full-node {what} vector"
            )
        v = torch.as_tensor(drivers[key])
        n_nodes = v.shape[-1] if v.ndim else 0
        needed = int(torch.maximum(self.src.max(), self.tgt.max())) + 1 if self.src.numel() else 0
        if n_nodes < needed:
            raise ValueError(
                f"{name} (kind {self.kind!r}): driver {key!r} has {n_nodes} node values but "
                f"the endpoints reach node index {needed - 1}"
            )
        return v

    def _densities(self, drivers: Mapping[str, Tensor] | None) -> tuple[Tensor, Tensor]:
        if drivers is None:
            return self._densities_of(drivers)
        keys = (self.p_key, self.T_key, *((self.Xw_key,) if self.medium.has_moisture else ()))
        return memo(self, "densities", tuple(drivers.get(k) for k in keys),
                    lambda: self._densities_of(drivers))

    def _densities_of(self, drivers: Mapping[str, Tensor] | None) -> tuple[Tensor, Tensor]:
        p = self._node_driver(drivers, self.p_key, "absolute pressure (Pa)")
        T = self._node_driver(drivers, self.T_key, "zone temperature")
        if self.medium.has_moisture:
            X = self._node_driver(drivers, self.Xw_key, "water mass fraction")
            X_A, X_B = X[..., self.src], X[..., self.tgt]
        else:  # TwoWayFlowElement.mo:76,81: X_w = 0 if Medium.nXi == 0
            X_A = X_B = 0.0
        rho_A = _density_pTX(p[..., self.src], T[..., self.src], X_A)
        rho_B = _density_pTX(p[..., self.tgt], T[..., self.tgt], X_B)
        return rho_A, rho_B


class DoorCompartmentHead(_InflowDensities):
    """The hydrostatic part of ``DoorDiscretized.mo:64-66``, as a noodl drive:
    ``rho_A hAg[i] - rho_B hBg[i]`` per compartment edge, shape ``(..., n_edges)``.

    ``hAg``/``hBg`` are MBL's ``g_n (h - (i - 0.5) dh)`` (``DoorDiscretized.mo:34-40``), one
    per edge of ``kind`` in the layer's edge order; ``rho_A``/``rho_B`` are the inflow
    densities of the module docstring, read from ``drivers[p_key]``, ``drivers[T_key]`` and
    (moist media) ``drivers[Xw_key]``. Geometry is fixed, the drivers are differentiable.
    """

    def __init__(
        self,
        kind: str,
        *,
        src,
        tgt,
        hAg,
        hBg,
        medium: AirMedium,
        T_key: str = "T",
        Xw_key: str = "X_w",
        p_key: str = "p_abs",
    ) -> None:
        self.kind = kind
        where = f"DoorCompartmentHead (kind {kind!r})"
        self.src, self.tgt = _endpoints(src, tgt, where=where)
        self.hAg = torch.as_tensor(hAg, dtype=torch.float64).reshape(-1).detach()
        self.hBg = torch.as_tensor(hBg, dtype=torch.float64).reshape(-1).detach()
        n = self.src.numel()
        if self.hAg.numel() != n or self.hBg.numel() != n:
            raise ValueError(
                f"{where}: hAg and hBg need one value per compartment edge ({n}), got "
                f"{self.hAg.numel()} and {self.hBg.numel()}"
            )
        self.medium = medium
        self.T_key, self.Xw_key, self.p_key = str(T_key), str(Xw_key), str(p_key)

    def __call__(self, drivers: Mapping[str, Tensor]) -> Tensor:
        def head() -> Tensor:
            rho_A, rho_B = self._densities(drivers)
            return rho_A * self.hAg.to(rho_A.dtype) - rho_B * self.hBg.to(rho_B.dtype)

        return memo(self, "head", self._densities(drivers), head)


class _DoorCompartmentBase(_InflowDensities, Element):
    """Endpoints, drivers and the smoothed density weighting shared by both doors."""

    def __init__(
        self,
        *,
        src,
        tgt,
        medium: AirMedium,
        dp_turbulent: float,
        vZer: float,
        T_key: str,
        Xw_key: str,
        p_key: str,
        kind: str,
    ) -> None:
        Element.__init__(self, kind)
        name = type(self).__name__
        src_t, tgt_t = _endpoints(src, tgt, where=f"{name} (kind {kind!r})")
        if not float(dp_turbulent) > 0.0:
            raise ValueError(
                f"{name} (kind {kind!r}): dp_turbulent must be strictly positive, got "
                f"{dp_turbulent!r}"
            )
        if not float(vZer) > 0.0:
            raise ValueError(f"{name} (kind {kind!r}): vZer must be strictly positive")
        self.register_buffer("src", src_t)
        self.register_buffer("tgt", tgt_t)
        self.medium = medium
        self.rho_default = float(medium.rho_default)
        self.dp_turbulent = float(dp_turbulent)
        self.vZer = float(vZer)
        self.T_key, self.Xw_key, self.p_key = str(T_key), str(Xw_key), str(p_key)

    def _check_width(self, dp: Tensor) -> None:
        n_edges = self.src.numel()
        width = dp.shape[-1] if dp.ndim else 1
        if n_edges > 1 and width != n_edges:
            raise ValueError(
                f"{type(self).__name__} (kind {self.kind!r}): dp has {width} columns but "
                f"this element covers {n_edges} compartment edges"
            )

    def _volume_flow(self, dp: Tensor, drivers) -> tuple[Tensor, Tensor]:
        """``(dV_flow, VZerCom_flow)`` per edge; implemented by each door."""
        raise NotImplementedError

    def _law(self, drivers) -> tuple:
        """``(CVal, m, VZerCom_flow, a, b, c, d, CVal dp_turbulent^m)`` of the compartments'
        power law (``door._power_law_coeffs``), a memo; implemented by each door."""
        raise NotImplementedError

    def stream_slope(self, dp: Tensor, drivers=None) -> Tensor | None:
        """``d(mAB_flow - mBA_flow)/d dpAB[i]`` per compartment (:meth:`port_flows`), by the
        chain rule in the order of operations reverse-mode autograd applies to
        ``port_flows`` (``door._power_law_slope_at``), so that it equals the autograd
        derivative bit for bit (checked in the tests). ``None`` where that is not assured
        (grad mode on, or broadcasting beyond ``dp``'s shape): the caller takes autograd."""
        rho_A, rho_B = self._densities(drivers)
        CVal, m, VZerCom, *coeffs = self._law(drivers)
        if not _slope_is_exact(dp, rho_A, rho_B, CVal, m, VZerCom):
            return None
        dV = _power_law_at(CVal, dp, m, self.dp_turbulent, coeffs)
        # _smooth_heaviside(dV, VZerCom), forward.
        dx = 0.5 * dV / VZerCom
        xpow2 = dx * dx
        r2 = -5 + 6 * xpow2
        r1 = 1.875 + xpow2 * r2
        u = 0.5 + dx * r1
        gai = torch.clamp(u, 0.0, 1.0)
        # mAB = sum(rho_A dV gai), mBA = sum(rho_B (-dV) (1 - gai)); adjoints 1 and -1.
        one = torch.ones_like(dV)
        g_e2 = -one
        tb = rho_B * -dV
        g_tb = g_e2 * (1 - gai)
        g_gai = -(g_e2 * tb) + one * (rho_A * dV)
        g_u = torch.where((u > 0.0).logical_and_(u < 1.0), g_gai,  # clamp: 0 at the bounds
                          torch.zeros((), dtype=g_gai.dtype))
        g_r1 = g_u * dx
        g_r2 = g_r1 * xpow2
        c = (g_r1 * r2 + g_r2 * 6) * dx
        g_dx = (g_u * r1 + c) + c
        g_dV = (-(g_tb * rho_B) + (one * gai) * rho_A) + (g_dx / VZerCom) * 0.5
        return _power_law_slope_at(CVal, dp, m, self.dp_turbulent, coeffs, g_dV)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        """``rho_A dVAB_flow[i] - rho_B dVBA_flow[i]`` (``DoorDiscretized.mo:69-71``,
        ``TwoWayFlowElement.mo:91-92``), kg/s, positive from ``src`` (A) to ``tgt`` (B)."""
        self._check_width(dp)
        rho_A, rho_B = self._densities(drivers)
        dV, VZerCom = self._volume_flow(dp, drivers)
        gai = _smooth_heaviside(dV, VZerCom)  # DoorDiscretized.mo:69
        dVAB = dV * gai  # DoorDiscretized.mo:70
        dVBA = -dV * (1 - gai)  # DoorDiscretized.mo:71
        return rho_A * dVAB - rho_B * dVBA

    def port_flows(self, dp: Tensor, drivers=None) -> tuple[Tensor, Tensor]:
        """MBL's two port flows of the whole door, ``(port_a1.m_flow, port_a2.m_flow)`` =
        ``(mAB_flow, mBA_flow)`` (``TwoWayFlowElement.mo:85-86,91-92``: ``rho_a1_inflow
        VAB_flow`` and ``rho_a2_inflow VBA_flow``, with ``VAB_flow = sum(dVAB_flow)``,
        ``VBA_flow = sum(dVBA_flow)``, ``DoorDiscretized.mo:74-75``), kg/s, each summed over
        the last (compartment) dimension of ``dp``.

        These are the reference's ``m1_flow``/``m2_flow`` and ``mAB_flow``/``mBA_flow``
        outputs EXACTLY, including compartments inside the ``smoothHeaviside`` band, where the
        plain sums of the positive and negative edge flows differ from them (module
        docstring); their difference is the sum of :meth:`flow`."""
        self._check_width(dp)
        rho_A, rho_B = self._densities(drivers)
        dV, VZerCom = self._volume_flow(dp, drivers)
        gai = _smooth_heaviside(dV, VZerCom)  # DoorDiscretized.mo:69
        mAB = (rho_A * dV * gai).sum(dim=-1)  # DoorDiscretized.mo:70,74 (rho_A per edge)
        mBA = (rho_B * -dV * (1 - gai)).sum(dim=-1)  # DoorDiscretized.mo:71,75
        return mAB, mBA

    def switching(self, dp: Tensor, drivers=None) -> Tensor:
        """Per compartment, the band edges of the power law, ``dpAB = +-dp_turbulent``, and
        of ``smoothHeaviside``, ``dV_flow = +-VZerCom_flow`` (where it meets 0 and 1): all
        twice continuously differentiable, not three times."""
        self._check_width(dp)
        dV, VZerCom = self._volume_flow(dp, drivers)
        return torch.cat([band_edges(dp, self.dp_turbulent), band_edges(dV, VZerCom)], dim=-1)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        """Tangent at ``dp = 0`` on a per-edge zero (the base class's 0-d zero would sum
        the autograd slope over the compartments)."""
        zero = torch.zeros(max(self.src.numel(), 1), dtype=self._dtype())
        return self.flow(zero, drivers), self.dflow(zero, drivers)


class DoorCompartment(_DoorCompartmentBase):
    """One compartment of ``Buildings.Airflow.Multizone.DoorDiscretizedOpen`` per edge.

    ``dA`` is the compartment area ``A/nCom`` (``DoorDiscretized.mo:60``), scalar or one value
    per edge; ``CVal = CD dA sqrt(2/rho_default)`` (``DoorDiscretizedOpen.mo:24``) and the
    volume flow is ``powerLaw05`` (``DoorDiscretizedOpen.mo:27-35``, ``powerLaw05.mo:24-29``;
    evaluated as the general law at ``m``, identical at MBL's fixed ``m = 0.5``,
    ``DoorDiscretizedOpen.mo:10,22``). The smoothing width is
    ``VZerCom_flow = vZer A/nCom = vZer dA`` (``DoorDiscretized.mo:52``,
    ``TwoWayFlowElement.mo:83``). Defaults are MBL's: ``CD = 0.65``, ``dp_turbulent = 0.01``,
    ``vZer = 0.001``. Build a whole door with :func:`discretized_door`.
    """

    def __init__(
        self,
        *,
        src,
        tgt,
        medium: AirMedium,
        dA,
        CD=0.65,
        m=_M_FIXED,
        dp_turbulent: float = 0.01,
        vZer: float = 0.001,
        T_key: str = "T",
        Xw_key: str = "X_w",
        p_key: str = "p_abs",
        kind: str = "door",
        learnable: bool = False,
    ) -> None:
        super().__init__(
            src=src,
            tgt=tgt,
            medium=medium,
            dp_turbulent=dp_turbulent,
            vZer=vZer,
            T_key=T_key,
            Xw_key=Xw_key,
            p_key=p_key,
            kind=kind,
        )
        self.dA = self._param(_f64(dA), learnable)
        self.CD = self._param(_f64(CD), learnable)
        self.m = self._param(_f64(m), learnable)

    def _law(self, drivers) -> tuple:
        def law():
            CVal = self.CD * self.dA * math.sqrt(2 / self.rho_default)  # :24
            return (CVal, self.m, self.vZer * self.dA,  # DoorDiscretized.mo:52,
                    *_power_law_coeffs(CVal, self.m, self.dp_turbulent))  # TwoWay...mo:83

        return memo(self, "law", (self.CD, self.dA, self.m), law)

    def _volume_flow(self, dp: Tensor, drivers) -> tuple[Tensor, Tensor]:
        CVal, m, VZerCom, *coeffs = self._law(drivers)
        dV = _power_law_at(CVal, dp, m, self.dp_turbulent,
                           coeffs)  # DoorDiscretizedOpen.mo:27-35
        return dV, VZerCom


class OperableDoorCompartment(_DoorCompartmentBase):
    """One compartment of ``Buildings.Airflow.Multizone.DoorDiscretizedOperable`` per edge,
    blended between the open door and the closed crack by ``y = drivers[y_key]``.

    ``AOpe = wOpe hOpe`` is the whole door's open area and ``nCom`` its compartment count
    (scalars or one value per edge); ``LClo`` is the closed door's effective leakage area.
    ``DoorDiscretizedOperable.mo:43-54``: ``AClo = CDClo/CDCloRat LClo dpCloRat^(0.5-mClo)``,
    ``CClo = CDClo AClo/nCom sqrt(2/rho_default)``, ``COpe = CDOpe AOpe/nCom
    sqrt(2/rho_default)``, ``m = y mOpe + (1-y) mClo``, ``A = y AOpe + (1-y) AClo``,
    ``CVal = y COpe + (1-y) CClo``; ``:57-63``: ``dV_flow = powerLaw(CVal, dp, m,
    dp_turbulent)``. The smoothing width is ``VZerCom_flow = vZer A / nCom``
    (``DoorDiscretized.mo:52``, ``TwoWayFlowElement.mo:83``) with the blended ``A``.
    Defaults are MBL's (``DoorDiscretizedOperable.mo:6-27``): ``dpCloRat = 4``,
    ``CDCloRat = 1``, ``CDOpe = CDClo = 0.65``, ``mOpe = 0.5``, ``mClo = 0.65``.

    ``drivers[y_key]`` is a 0-d scalar or a tensor whose last dimension is 1 or the edge
    count, broadcasting against ``dp``; values are used as given (MBL declares ``min=0,
    max=1``, ``:29``).
    """

    def __init__(
        self,
        *,
        src,
        tgt,
        medium: AirMedium,
        y_key: str,
        AOpe,
        LClo,
        nCom,
        CDOpe=0.65,
        CDClo=0.65,
        CDCloRat=1.0,
        dpCloRat=4.0,
        mOpe=0.5,
        mClo=0.65,
        dp_turbulent: float = 0.01,
        vZer: float = 0.001,
        T_key: str = "T",
        Xw_key: str = "X_w",
        p_key: str = "p_abs",
        kind: str = "door",
        learnable: bool = False,
    ) -> None:
        super().__init__(
            src=src,
            tgt=tgt,
            medium=medium,
            dp_turbulent=dp_turbulent,
            vZer=vZer,
            T_key=T_key,
            Xw_key=Xw_key,
            p_key=p_key,
            kind=kind,
        )
        self.y_key = str(y_key)
        self.register_buffer("nCom", torch.as_tensor(nCom, dtype=torch.float64))
        self.AOpe = self._param(_f64(AOpe), learnable)
        self.LClo = self._param(_f64(LClo), learnable)
        self.CDOpe = self._param(_f64(CDOpe), learnable)
        self.CDClo = self._param(_f64(CDClo), learnable)
        self.CDCloRat = self._param(_f64(CDCloRat), learnable)
        self.dpCloRat = self._param(_f64(dpCloRat), learnable)
        self.mOpe = self._param(_f64(mOpe), learnable)
        self.mClo = self._param(_f64(mClo), learnable)

    def _y(self, drivers) -> Tensor:
        if drivers is None or self.y_key not in drivers:
            raise KeyError(
                f"OperableDoorCompartment (kind {self.kind!r}): driver {self.y_key!r} not "
                f"found; it needs the door opening signal y (0 closed, 1 open)"
            )
        y = torch.as_tensor(drivers[self.y_key])
        n_edges = self.src.numel()
        if y.ndim and y.shape[-1] not in (1, n_edges):
            raise ValueError(
                f"OperableDoorCompartment (kind {self.kind!r}): driver {self.y_key!r} has "
                f"last dimension {y.shape[-1]}; expected 1 or the edge count {n_edges}"
            )
        return y

    def _AClo(self) -> Tensor:
        """``DoorDiscretizedOperable.mo:43``."""
        return self.CDClo / self.CDCloRat * self.LClo * self.dpCloRat ** (0.5 - self.mClo)

    def face_area(self, drivers) -> Tensor:
        """``A = y AOpe + (1 - y) AClo`` (``DoorDiscretizedOperable.mo:52``)."""
        y = self._y(drivers)
        return y * self.AOpe + (1 - y) * self._AClo()

    def _law(self, drivers) -> tuple:
        y = self._y(drivers)

        def law():
            s = math.sqrt(2 / self.rho_default)
            CClo = self.CDClo * self._AClo() / self.nCom * s  # DoorDiscretizedOperable.mo:46
            COpe = self.CDOpe * self.AOpe / self.nCom * s  # :47
            m = y * self.mOpe + (1 - y) * self.mClo  # :50
            A = y * self.AOpe + (1 - y) * self._AClo()  # :52
            CVal = y * COpe + (1 - y) * CClo  # :54
            return (CVal, m, self.vZer * A / self.nCom,  # DoorDiscretized.mo:52,
                    *_power_law_coeffs(CVal, m, self.dp_turbulent))  # TwoWayFlowElement.mo:83

        params = (self.nCom, self.AOpe, self.LClo, self.CDOpe, self.CDClo, self.CDCloRat,
                  self.dpCloRat, self.mOpe, self.mClo)
        return memo(self, "law", (y, *params), law)

    def _volume_flow(self, dp: Tensor, drivers) -> tuple[Tensor, Tensor]:
        CVal, m, VZerCom, *coeffs = self._law(drivers)
        dV = _power_law_at(CVal, dp, m, self.dp_turbulent,
                           coeffs)  # :57-63, powerLaw.mo:17-29
        return dV, VZerCom


# ---------------------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------------------


def _door_endpoints(nCom: int, src, tgt) -> tuple[Tensor, Tensor]:
    """A scalar node position repeated ``nCom`` times, or ``nCom`` positions as given (e.g.
    ``net.endpoints(kind)``)."""
    s = torch.as_tensor(src).to(torch.long).reshape(-1)
    t = torch.as_tensor(tgt).to(torch.long).reshape(-1)
    if s.numel() == 1 and t.numel() == 1:
        return s.expand(nCom).clone(), t.expand(nCom).clone()
    if s.numel() != nCom or t.numel() != nCom:
        raise ValueError(
            f"discretised door: src/tgt must be one node each or nCom = {nCom} positions, "
            f"got {s.numel()} and {t.numel()}"
        )
    return s, t


def _heights(nCom: int, hOpe, hA, hB) -> tuple[Tensor, Tensor]:
    """``DoorDiscretized.mo:21,34-40``: ``dh = hOpe/nCom``, ``hAg[i] = g_n (hA - (i-0.5) dh)``,
    ``hBg[i] = g_n (hB - (i-0.5) dh)``."""
    dh = float(hOpe) / nCom
    i = torch.arange(1, nCom + 1, dtype=torch.float64)
    return _G_N * (float(hA) - (i - 0.5) * dh), _G_N * (float(hB) - (i - 0.5) * dh)


def _check_ncom(nCom) -> int:
    n = int(nCom)
    if n != nCom or n < 1:
        raise ValueError(f"discretised door: nCom must be a positive integer, got {nCom!r}")
    return n


def discretized_door(
    *,
    src,
    tgt,
    medium: AirMedium,
    nCom: int = 10,
    wOpe: float = 0.9,
    hOpe: float = 2.1,
    hA: float = 2.7 / 2,
    hB: float = 2.7 / 2,
    CD: float = 0.65,
    m: float = _M_FIXED,
    dp_turbulent: float = 0.01,
    vZer: float = 0.001,
    kind: str = "door",
    T_key: str = "T",
    Xw_key: str = "X_w",
    p_key: str = "p_abs",
) -> tuple[DoorCompartment, DoorCompartmentHead]:
    """One ``DoorDiscretizedOpen`` as ``nCom`` edges of ``kind``: the compartment element and
    its head drive. Defaults are MBL's (``DoorDiscretized.mo:6-11``,
    ``TwoWayFlowElementBuoyancy.mo:6-14``, ``DoorDiscretizedOpen.mo:6``,
    ``TwoWayFlowElement.mo:19``). ``dA = wOpe hOpe / nCom`` on every edge
    (``DoorDiscretizedOpen.mo:23``, ``DoorDiscretized.mo:60``)."""
    n = _check_ncom(nCom)
    s, t = _door_endpoints(n, src, tgt)
    hAg, hBg = _heights(n, hOpe, hA, hB)
    dA = torch.full((n,), float(wOpe) * float(hOpe) / n, dtype=torch.float64)
    comp = DoorCompartment(
        src=s,
        tgt=t,
        medium=medium,
        dA=dA,
        CD=CD,
        m=m,
        dp_turbulent=dp_turbulent,
        vZer=vZer,
        T_key=T_key,
        Xw_key=Xw_key,
        p_key=p_key,
        kind=kind,
    )
    head = DoorCompartmentHead(
        kind,
        src=s,
        tgt=t,
        hAg=hAg,
        hBg=hBg,
        medium=medium,
        T_key=T_key,
        Xw_key=Xw_key,
        p_key=p_key,
    )
    return comp, head


def discretized_operable_door(
    *,
    src,
    tgt,
    medium: AirMedium,
    y_key: str,
    LClo: float,
    nCom: int = 10,
    wOpe: float = 0.9,
    hOpe: float = 2.1,
    hA: float = 2.7 / 2,
    hB: float = 2.7 / 2,
    CDOpe: float = 0.65,
    CDClo: float = 0.65,
    CDCloRat: float = 1.0,
    dpCloRat: float = 4.0,
    mOpe: float = 0.5,
    mClo: float = 0.65,
    dp_turbulent: float = 0.01,
    vZer: float = 0.001,
    kind: str = "door",
    T_key: str = "T",
    Xw_key: str = "X_w",
    p_key: str = "p_abs",
) -> tuple[OperableDoorCompartment, DoorCompartmentHead]:
    """One ``DoorDiscretizedOperable`` as ``nCom`` edges of ``kind``; see
    :func:`discretized_door`. ``AOpe = wOpe hOpe`` (``DoorDiscretizedOperable.mo:35``);
    ``LClo`` has no MBL default."""
    n = _check_ncom(nCom)
    s, t = _door_endpoints(n, src, tgt)
    hAg, hBg = _heights(n, hOpe, hA, hB)
    comp = OperableDoorCompartment(
        src=s,
        tgt=t,
        medium=medium,
        y_key=y_key,
        AOpe=torch.full((n,), float(wOpe) * float(hOpe), dtype=torch.float64),
        LClo=LClo,
        nCom=torch.full((n,), float(n), dtype=torch.float64),
        CDOpe=CDOpe,
        CDClo=CDClo,
        CDCloRat=CDCloRat,
        dpCloRat=dpCloRat,
        mOpe=mOpe,
        mClo=mClo,
        dp_turbulent=dp_turbulent,
        vZer=vZer,
        T_key=T_key,
        Xw_key=Xw_key,
        p_key=p_key,
        kind=kind,
    )
    head = DoorCompartmentHead(
        kind,
        src=s,
        tgt=t,
        hAg=hAg,
        hBg=hBg,
        medium=medium,
        T_key=T_key,
        Xw_key=Xw_key,
        p_key=p_key,
    )
    return comp, head


class DoorPortStream(Element):
    """One of the two port streams of a whole discretised door, as ONE edge from side A to
    side B: ``direction="ab"`` returns ``port_a1.m_flow = mAB_flow``, ``"ba"`` returns
    ``port_b2.m_flow = -port_a2.m_flow = -mBA_flow`` (kg/s, both positive from A to B), from
    the compartment law ``comp`` (:class:`DoorCompartment` or
    :class:`OperableDoorCompartment`, one door) at the compartment pressure differences
    ``dp + head(drivers)`` (:class:`DoorCompartmentHead`). The edge's own ``dp`` is the zones'
    potential difference ``phi_A - phi_B``.

    Why two stream edges and not ``nCom`` compartment edges: MBL moves a discretised door's
    enthalpy, moisture and trace substances on its two PORT flows (``TwoWayFlowElement.mo:
    94-111``: ``port_b1.h_outflow = inStream(port_a1.h_outflow)`` etc.), each upwinded on its
    own sign (``actualStream``), i.e. on the SUM over the compartments of ``dVAB_flow`` resp.
    ``dVBA_flow``. A compartment inside the ``smoothHeaviside`` band (``|dV_flow| <
    VZerCom_flow``) contributes to both paths with opposite signs, which cancel in the
    exchange; a transport over compartment edges would upwind each compartment's NET flow and
    carry all of it with one zone's state. With the two stream edges every transport layer,
    every time scheme and the moist-air heat carrier see MBL's streams, and the edges' flows
    ARE the door's ``m1_flow``/``m2_flow`` (``mAB_flow``, ``-mBA_flow``). The node mass balance
    is the same (``mAB - mBA`` is the compartments' net flow).
    """

    def __init__(self, comp: _DoorCompartmentBase, head: DoorCompartmentHead,
                 direction: str, kind: str) -> None:
        Element.__init__(self, kind)
        if direction not in ("ab", "ba"):
            raise ValueError(
                f"DoorPortStream (kind {kind!r}): direction must be 'ab' or 'ba', got "
                f"{direction!r}"
            )
        pairs = set(zip(comp.src.tolist(), comp.tgt.tolist(), strict=True))
        if len(pairs) != 1:
            raise ValueError(
                f"DoorPortStream (kind {kind!r}): the compartment element must be ONE door "
                f"(every compartment edge between the same two nodes), got {sorted(pairs)}"
            )
        self.comp, self.head, self.direction = comp, head, direction
        self.register_buffer("src", comp.src[:1].clone())
        self.register_buffer("tgt", comp.tgt[:1].clone())

    def _shared(self, name: str, dp: Tensor, drivers, fn):
        """`fn()`, shared between the door's two stream edges (they have the same `comp`):
        both evaluate the door at the same `dp` and drivers within one pass of the layer
        (`flows`, `dflows`), so the second takes the first's value. Keyed on `dp`'s values,
        the compartment head (a memo of the drivers, `DoorCompartmentHead`) and the opening
        signal; nothing that carries an autograd graph is shared."""
        head = self.head(drivers)
        y_key = getattr(self.comp, "y_key", None)
        y = drivers.get(y_key) if (y_key is not None and drivers is not None) else None
        y_ver = y._version if isinstance(y, Tensor) else None
        shared = self.comp.__dict__.setdefault("_stream_shared", {})
        hit = shared.get(name)
        if (hit is not None and not dp.requires_grad and hit[0] is head and hit[1] is y
                and hit[2] == y_ver and hit[3].shape == dp.shape and torch.equal(hit[3], dp)):
            return hit[4]
        value = fn(head)
        parts = value if isinstance(value, tuple) else (value,)
        if dp.requires_grad or head.requires_grad or any(v.requires_grad for v in parts):
            shared.pop(name, None)
        else:
            shared[name] = (head, y, y_ver, dp.clone(), value)
        return value

    def _streams(self, dp: Tensor, drivers, head: Tensor | None = None
                 ) -> tuple[Tensor, Tensor]:
        if head is None:
            head = self.head(drivers)
        dpi = dp + head  # (..., 1) + (..., nCom): the compartments' dpAB
        return self.comp.port_flows(dpi, drivers)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        """``mAB_flow`` (``"ab"``) or ``-mBA_flow`` (``"ba"``), shape ``dp``'s (one edge)."""
        mAB, mBA = self._shared("streams", dp, drivers,
                                lambda head: self._streams(dp, drivers, head))
        out = mAB if self.direction == "ab" else -mBA
        return out.unsqueeze(-1) if dp.ndim else out

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        """HALF the slope of the door's net flow ``mAB_flow - mBA_flow``, for either stream.

        One stream's own slope can be negative: ``d(dV gaiFlo)/d dV = gaiFlo + dV
        gaiFlo'`` dips below 0 for a compartment just below zero inside the
        ``smoothHeaviside`` band (``ClosedDoors``: -5e-5 kg/(s Pa)). The two stream edges join
        the same two nodes at the same ``dp``, so the airflow Jacobian (and every per-edge
        slope a potential layer sums into its Laplacian) only ever sees the SUM of their
        slopes, which is the net flow's, positive; reporting it split evenly keeps that sum
        exact and each edge's slope nonnegative for the layer's grounding certificate."""
        grad_enabled = torch.is_grad_enabled()

        def slope(head: Tensor) -> Tensor:
            exact = self.comp.stream_slope(dp + head, drivers)
            if exact is not None:
                return 0.5 * exact.sum_to_size(dp.shape)
            x = dp.detach().clone()
            x.requires_grad_(True)
            with torch.enable_grad():
                mAB, mBA = self._streams(x, drivers, head)
                (grad,) = torch.autograd.grad((mAB - mBA).sum(), x,
                                              create_graph=grad_enabled)
            return 0.5 * grad

        if grad_enabled:
            return slope(self.head(drivers))
        return self._shared("slope", dp, drivers, slope)

    def switching(self, dp: Tensor, drivers=None) -> Tensor:
        """The compartments' switches (:meth:`_DoorCompartmentBase.switching`) at their
        pressure differences ``dp + head``."""
        return self.comp.switching(dp + self.head(drivers), drivers)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        """Tangent at ``dp = 0`` on a one-edge zero."""
        zero = torch.zeros(1, dtype=self._dtype())
        return self.flow(zero, drivers), self.dflow(zero, drivers)

# Pre-rename names, kept as aliases so existing code keeps working.
MBLDoorCompartment = DoorCompartment  # alias, the pre-rename name
MBLDoorCompartmentOperable = OperableDoorCompartment  # alias, the pre-rename name
MBLDoorPortStream = DoorPortStream  # alias, the pre-rename name
mbl_discretized_door = discretized_door  # alias, the pre-rename name
mbl_discretized_operable_door = discretized_operable_door  # alias, the pre-rename name

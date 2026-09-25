"""Headspace air: the `Headspace` branch element, the `Drag` drive and the air-density
closure that feeds the existing `Stack` buoyancy drive.

Per `headspace` edge the branch law is

    dp = R(h) |Q| Q - D(h, V_w) - B(T_head, T_amb)

with the wall-friction resistance ``R(h) = f_air L rho / (2 D_h A_air^2)`` (Darcy-Weisbach
on the air segment's hydraulic diameter), the interfacial drag ``D`` from the moving water
surface, and the stack term ``B`` from the existing `drives.Stack`.

``A_air`` and ``D_h`` arrive as per-edge DRIVERS written by `SewerHydraulics` in the same
pass, exactly as `UpstreamDensityPowerLaw` reads its density driver: an element may read
`drivers` inside `flow`/`dflow`/`linear_init`, a Drive may read `drivers` and nothing else.

Coefficient status: ``f_air`` default 0.02 is UNVERIFIED (the reported
range for sewer crowns is 0.015-0.045, Edwini-Bonsu and Steffler 2006, paywalled);
``F_I_DEFAULT`` is CALIBRATED here against Pescod and Price's Test 8 as tabulated by
Edwini-Bonsu and Steffler 2004, Table 1 p.337.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from noodl.elements.base import Element

Tensor = torch.Tensor
F64 = torch.float64

#: Air density at 20 C and 101325 Pa (kg/m^3), the reference the elements are written at.
RHO_AIR_REF = 1.2041

#: Molar mass of dry air (kg/mol) and the universal gas constant (J/mol/K).
M_AIR = 0.0289647
R_GAS = 8.314462618

#: Standard atmospheric pressure (Pa).
P_ATM = 101325.0

#: Wall friction factor for the headspace (Darcy). UNVERIFIED; inside the 0.015-0.045 range
#: reported for sewer crowns.
F_AIR_DEFAULT = 0.02

#: Interfacial drag factor, CALIBRATED to Pescod and Price Test 8 (300 mm UPVC, 15 m, both
#: ends open, h/D = 0.6, V_w = 0.8 m/s, measured U_air = 0.20 m/s, i.e. U_air/V_w = 25 %) at
#: ``f_air = 0.02``. For a pipe open at both ends the balance ``R Q^2 = D`` has the closed
#: form ``U_air / V_w = sqrt((f_i / f_air) T D_h / A_air)``, which at that geometry gives
#: ``f_i = 0.02 * 0.25^2 / 1.668281 = 7.492741e-4``, rounded to 7.49e-4. The same value
#: reproduces Tests 7 and 9 at 24.14 % and 25.15 % against measured 35 % and 27.5 % (both
#: inside the 20-40 % acceptance band of verification check A1). MEASURED.
F_I_DEFAULT = 7.49e-4


class Headspace(Element):
    """``dp = R |Q| Q`` on a headspace edge, with ``R`` read from per-edge drivers.

    ``R(h) = f_air L rho / (2 D_h A_air^2)``; the inversion is the closed form
    ``Q = sign(dp) sqrt(|dp| / R)``, laminar-blended below ``dp_transition`` in the
    `PowerLaw` manner, with the safe input substituted on BOTH branches of the ``where``
    (the exponent 1/2 has an infinite derivative at ``dp = 0``).

    ``area_key`` and ``dh_key`` name full per-`pipe`-edge driver tensors; the element holds
    the positions of its own edges within that per-pipe order as a registered buffer, so the
    pairing between a `headspace` edge and the `pipe` edge it parallels is decided ONCE, at
    construction, and never re-derived.
    """

    def __init__(
        self,
        length,
        pipe_positions,
        *,
        f_air=F_AIR_DEFAULT,
        rho_air: float = RHO_AIR_REF,
        area_key: str = "sewer.A_air",
        dh_key: str = "sewer.D_h",
        dp_transition: float = 1e-8,
        kind: str = "headspace",
        learnable: bool = False,
    ) -> None:
        super().__init__(kind)
        self.length = self._param(length, False)
        # `Element._param` casts a plain Python float with `torch.get_default_dtype()`
        # (float32 in this repo, per `elements/duct.py`'s documented gotcha); casting
        # explicitly to `self.length`'s dtype first keeps a float64-constructed element
        # float64 throughout instead of silently losing precision on the default `f_air`.
        self.f_air = self._param(torch.as_tensor(f_air, dtype=self.length.dtype), learnable)
        self.rho_air = float(rho_air)
        self.area_key = area_key
        self.dh_key = dh_key
        self.dp_transition = float(dp_transition)
        self.register_buffer(
            "pipe_positions", torch.as_tensor(pipe_positions, dtype=torch.long)
        )

    def resistance(self, drivers: Mapping | None) -> Tensor:
        # Name only the key(s) that are actually missing, not always both -- a
        # caller who supplied `area_key` but forgot `dh_key` (or the reverse) otherwise
        # gets an error naming a driver it DID give, which reads as if the code were wrong
        # about its own requirement.
        missing = [
            key for key in (self.area_key, self.dh_key)
            if drivers is None or key not in drivers
        ]
        if missing:
            raise KeyError(
                f"Headspace (kind {self.kind!r}): driver(s) {missing} are required and "
                f"were not given; they are written by the SewerHydraulics closure in the "
                f"same pass"
            )
        a_air = drivers[self.area_key].index_select(-1, self.pipe_positions)
        d_h = drivers[self.dh_key].index_select(-1, self.pipe_positions)
        if bool(torch.any(a_air <= 0)):
            bad = (a_air <= 0).reshape(-1, a_air.shape[-1]).any(0).nonzero()
            raise ValueError(
                f"Headspace (kind {self.kind!r}): the headspace area is not positive on "
                f"edge positions {bad.flatten().tolist()}; a full pipe has no headspace "
                f"and no air path"
            )
        return self.f_air * self.length * self.rho_air / (2.0 * d_h * a_air**2)

    def flow(self, dp: Tensor, drivers=None) -> Tensor:
        r = self.resistance(drivers)
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = torch.sign(dp) * torch.sqrt(dp_safe / r)
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=dp.dtype) / r) / dpt
        return torch.where(mask, slope * dp, sharp)

    def dflow(self, dp: Tensor, drivers=None) -> Tensor:
        r = self.resistance(drivers)
        dpt = self.dp_transition
        mask = dp.abs() < dpt
        dp_safe = torch.where(mask, torch.full_like(dp, dpt), dp.abs())
        sharp = 0.5 / torch.sqrt(r * dp_safe)
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=dp.dtype) / r) / dpt
        return torch.where(mask, slope * torch.ones_like(dp), sharp)

    def linear_init(self, drivers=None) -> tuple[Tensor, Tensor]:
        r = self.resistance(drivers)
        dpt = self.dp_transition
        slope = torch.sqrt(torch.as_tensor(dpt, dtype=r.dtype) / r) / dpt
        return torch.zeros_like(slope), slope


class Drag:
    """Interfacial drag as a `Drive`: the shear a moving water surface exerts on the air.

    ``D = (f_i / 2) rho U_s |U_s| T L / A_air`` with ``U_s = c_s V_w`` the surface velocity
    (Edwini-Bonsu and Steffler 2004's rigid-moving-lid framing, p.332), spread over the air
    area.

    SIGN. The layer evaluates ``dp = (phi_src - phi_tgt) + sum(drives)`` and the element's
    law is ``dp = R |Q| Q``, so the momentum balance ``phi_src - phi_tgt = R|Q|Q - D - B``
    means the
    drive returns ``+D``: with both ends at ambient the drag alone then produces
    ``R|Q|Q = D``, i.e. flow from the upstream manhole towards the downstream one, which is
    the direction the wastewater drags the air. (Returning ``-D`` reverses the airflow and
    was caught by the A1 measurement.)

    A `Drive` reads DRIVERS ONLY -- never `phi`, never another layer's state -- which is
    what keeps the Newton Jacobian exactly ``A_I diag(g') A_I^T``. The relative-velocity
    form ``(U_s - V_air)`` is therefore out of scope and not implemented.

    ``zero_positions`` names edge positions within this kind's block whose drive is zeroed:
    the outfall edge, whose "pipe" is the open end and carries no water surface of its own.
    """

    kind = "headspace"

    def __init__(
        self,
        length,
        pipe_positions,
        *,
        f_i=F_I_DEFAULT,
        c_s: float = 1.0,
        rho_air: float = RHO_AIR_REF,
        v_key: str = "sewer.v",
        width_key: str = "sewer.T",
        area_key: str = "sewer.A_air",
        zero_positions=(),
        kind: str = "headspace",
    ) -> None:
        self.kind = kind
        self.length = torch.as_tensor(length, dtype=F64)
        self.pipe_positions = torch.as_tensor(pipe_positions, dtype=torch.long)
        self.f_i = torch.as_tensor(f_i, dtype=F64)
        self.c_s = float(c_s)
        self.rho_air = float(rho_air)
        self.v_key = v_key
        self.width_key = width_key
        self.area_key = area_key
        mask = torch.ones(self.pipe_positions.shape[-1], dtype=F64)
        for p in zero_positions:
            mask[int(p)] = 0.0
        self.mask = mask

    def __call__(self, drivers: Mapping) -> Tensor:
        for key in (self.v_key, self.width_key, self.area_key):
            if key not in drivers:
                raise KeyError(
                    f"Drag (kind {self.kind!r}): driver {key!r} is required and was not "
                    f"given; it is written by the SewerHydraulics closure in the same pass"
                )
        v_w = drivers[self.v_key].index_select(-1, self.pipe_positions) * self.c_s
        width = drivers[self.width_key].index_select(-1, self.pipe_positions)
        a_air = drivers[self.area_key].index_select(-1, self.pipe_positions)
        drag = (
            0.5 * self.f_i * self.rho_air * v_w * v_w.abs() * width * self.length / a_air
        )
        return self.mask * drag


def air_density(temperature: Tensor, pressure: float = P_ATM) -> Tensor:
    """Ideal-gas air density ``rho = p M / (R T)`` (kg/m^3) at absolute temperature T (K)."""
    return pressure * M_AIR / (R_GAS * temperature)

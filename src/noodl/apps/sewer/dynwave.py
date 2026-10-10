"""SWMM 5.2.4 dynamic-wave routing for the sewer application.

This module starts with `DynamicWaveOptions`, the `[OPTIONS]` that change the physics of
dynamic-wave routing. It must not import `network.py` at module level: `network.py` imports
`DynamicWaveOptions` from here.
"""

from __future__ import annotations

from dataclasses import dataclass

from noodl.apps.sewer.dynwave_terms import DAMPING, NORMAL_FLOW_LIMITED

#: ``SURCHARGE_METHOD`` values accepted now. SWMM also has ``SLOT`` (Preissmann slot), which
#: changes surface areas below the crown and is not modelled yet.
SURCHARGE_METHODS = ("extran",)


@dataclass(frozen=True)
class DynamicWaveOptions:
    """SWMM 5.2.4's dynamic-wave [OPTIONS] that change the physics, with SWMM's defaults
    (`project.c:841-880`). `head_tolerance` and `max_trials` are SWMM's own iteration
    controls: recorded so a file round-trips, never used -- noodl solves the same
    equations to `build_model`'s own `iterate_tol` (docs: "Dynamic wave").

    `routing_step` is the fixed routing step (s); `inertial_damping` is ``"none"``,
    ``"partial"`` or ``"full"``; `normal_flow_limited` is ``"slope"``, ``"froude"``,
    ``"both"`` or ``"none"``; `surcharge_method` is ``"extran"`` (``"slot"`` is refused for
    now); `min_surface_area` is the junction surface-area floor (m2), ``None`` for SWMM's
    12.566 ft2; `head_tolerance` is SWMM's head convergence tolerance (m); `max_trials` is
    SWMM's trial limit (-).
    """

    routing_step: float = 20.0
    inertial_damping: str = "partial"
    normal_flow_limited: str = "both"
    surcharge_method: str = "extran"
    min_surface_area: float | None = None
    head_tolerance: float | None = None
    max_trials: int | None = None

    def validate(self) -> None:
        """Refuse an unknown keyword or a non-positive step or area, BY NAME."""
        if not self.routing_step > 0:
            raise ValueError(
                f"DynamicWaveOptions: routing_step must be > 0 s, got {self.routing_step!r}"
            )
        if self.surcharge_method == "slot":
            raise ValueError(
                "DynamicWaveOptions: surcharge_method 'slot' is not modelled yet: SWMM's "
                "Preissmann slot changes surface areas below the crown"
            )
        for name, allowed in (
            ("inertial_damping", DAMPING),
            ("normal_flow_limited", NORMAL_FLOW_LIMITED),
            ("surcharge_method", SURCHARGE_METHODS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(
                    f"DynamicWaveOptions: {name} must be one of {allowed}, got "
                    f"{getattr(self, name)!r}"
                )
        if self.min_surface_area is not None and not self.min_surface_area > 0:
            raise ValueError("DynamicWaveOptions: min_surface_area must be > 0 m2")

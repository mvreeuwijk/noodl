"""MUNICH v2.2 run on the 12-street network, against noodl physics hour by hour.

The case is built here (`_case`), written by `write_case(format="munich")`, and run by
MUNICH v2.2 (`With_stationary_hypothesis: yes`); its inputs and MUNICH's output are kept in
`tests/data/street/munich_twelve_street/` (see its NOTICE.md for how they were produced).
Three option sets are recorded, between them every MUNICH closure `StreetCase.model_options`
maps: the writer's defaults (exponential canyon wind, Schulte exchange, Bessel roof wind,
horizontal fluctuation), the alternatives (Bessel canyon wind, SIRANE exchange, Macdonald
roof wind, no fluctuation), and the closures MUNICH takes from SIRANE (Bessel canyon and roof
wind, SIRANE exchange).

Tolerance. MUNICH saves float32 (about 6e-8 relative) and stops its steady sweep once no
street changes by more than 0.01 ug/m3 (`StreetNetworkTransport.cxx:2515`), so the check is
2e-6 relative, the largest the stopping rule left in these runs being below 1e-6.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from noodl.apps.street_aq import (
    StreetCase,
    build_model,
    drivers_at,
    read_case,
    read_results,
    street_index,
    twelve_street_grid,
    write_case,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "street" / "munich_twelve_street"
SPECIES = "NO2"
TOLERANCE = 2e-6

VARIANTS = {
    "default": {},
    "alternate": {
        "Mean_wind_speed_parameterization": "Sirane",
        "Transfer_parameterization": "Sirane",
        "Building_height_wind_speed_parameterization": "Macdonald",
        "With_horizontal_fluctuation": "no",
    },
    # Every closure MUNICH borrows from SIRANE: no choice here is written with 0.41, so it
    # is the set on which inferring kappa would give 0.40 where MUNICH uses 0.41.
    "sirane_closures": {
        "Mean_wind_speed_parameterization": "Sirane",
        "Transfer_parameterization": "Sirane",
        "Building_height_wind_speed_parameterization": "Sirane",
    },
}


def _case() -> StreetCase:
    """Eight hours on `twelve_street_grid(L=100, W=20, H=20)`: winds across, along and
    oblique to the streets, neutral, stable and unstable, three emitting streets (one
    varying hour by hour) and a uniform background."""
    net, names = twelve_street_grid(L=100.0, W=20.0, H=20.0)
    n_hours = 8
    emissions = np.zeros((n_hours, len(names)))
    emissions[:, names.index("11")] = 1.0e-5
    emissions[:, names.index("4")] = 4.0e-6 * (1.0 + 0.1 * np.arange(n_hours))
    emissions[:, names.index("7")] = 2.0e-6
    return StreetCase.synthetic(
        net, species=[SPECIES], times=[3600.0 * k for k in range(n_hours)],
        meteo={
            "wind_dir_from_deg": np.array([200.0, 270.0, 0.0, 45.0, 135.0, 225.0, 315.0, 90.0]),
            "wind_speed": np.array([3.0, 5.0, 2.0, 4.0, 6.0, 3.0, 2.5, 4.5]),
            "u_star": np.array([0.30, 0.45, 0.20, 0.35, 0.50, 0.25, 0.22, 0.40]),
            "lmo": np.array([1e5, -200.0, 150.0, 1e5, -50.0, 400.0, 80.0, -1000.0]),
            "h_abl": np.array([800.0, 1200.0, 300.0, 900.0, 1500.0, 500.0, 250.0, 1000.0]),
            "temperature": 288.0,
        },
        emissions=emissions, background=1.0e-8, start=datetime(2014, 3, 16, 0),
    )


def _inputs(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.parent.name != "results"}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_recorded_inputs_are_what_write_case_writes(tmp_path, variant):
    """The fixture's MUNICH inputs are byte for byte what `write_case` writes for `_case`,
    so the run below checks the writer as well as the model."""
    written = write_case(tmp_path / variant, _case(), format="munich",
                         options=VARIANTS[variant])
    assert _inputs(written) == _inputs(FIXTURE / variant)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_noodl_reproduces_munichs_street_concentrations(variant, record_property):
    """`read_case` -> `model_options()` -> one steady solve per hour reproduces MUNICH's
    in-street concentration on all 12 streets in all 8 hours."""
    case = read_case(FIXTURE / variant)
    munich = read_results(FIXTURE / variant / "results", case=case).c_in[SPECIES]
    model, state, _ = build_model(case.network, species=(SPECIES,), meteo="per_street",
                                  background="per_street", **case.model_options())
    assert list(street_index(model)) == list(case.street_ids)   # street.x's order
    ours = np.stack([
        model.steady(state, drivers_at(case, model, k))["street.x"].numpy()
        for k in range(len(case.times))
    ])
    worst = float(np.max(np.abs(ours - munich) / munich))
    record_property("check", f"MUNICH 12-street run ({variant}), street concentration")
    record_property("tolerance", f"rel {TOLERANCE}")
    record_property("measured_rel", worst)
    assert worst < TOLERANCE, worst
    # Not trivially the background: every hour has streets well above it.
    assert np.all(munich.max(axis=1) > 1.4 * 1.0e-8)

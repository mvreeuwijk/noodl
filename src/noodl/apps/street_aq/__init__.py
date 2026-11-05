"""The street application: a SIRANE/MUNICH-type street-network dispersion model.

A model built ON TOP of the core package. The core supplies
every solve; this package decides the network, the closures and the units. Nothing under
`src/noodl/` outside `apps/` imports from here.
"""

from noodl.apps.street_aq.above_roof import street_steady_with_plume
from noodl.apps.street_aq.canyon import (
    ASPECT_RATIO_EXCHANGE_BETA,
    EULER_GAMMA_TRUNCATED,
    EXCHANGE_SIGMA_W_RATIO,
    GAMMA_E,  # noqa: F401  (alias)
    SCHULTE_BETA,  # noqa: F401  (alias)
    SIRANE_EXCHANGE,  # noqa: F401  (alias)
    Z0_B_DEFAULT,
    Z0_S_DEFAULT,
    BoundaryLayer,
    bessel_shape_parameter,
    boundary_layer,
    canopy_displacement_roughness,
    canyon_velocity,
    exchange_velocity,
    grid_shape_parameter,
    macdonald_profile,  # noqa: F401  (alias)
    roof_wind,
    shape_parameter,
    soulhac_shape,  # noqa: F401  (alias)
)
from noodl.apps.street_aq.case import (
    StreetCase,
    StreetResults,
    drivers_at,
    read_case,
    read_results,
    write_case,
    write_sweep,
)
from noodl.apps.street_aq.chemistry import (
    j_no2_elevation_cloud,
    j_no2_sirane,  # noqa: F401  (alias)
    j_no2_zenith_table,
    k_no_o3_sirane,  # noqa: F401  (alias)
    k_no_o3_soulhac2011,
    photostationary_for_streets,
    solar_elevation,
    street_steady,
)
from noodl.apps.street_aq.closures import (
    KAPPA_040,
    KAPPA_041,
    KAPPA_MUNICH,  # noqa: F401  (alias)
    OPTIONS,
    PRESETS,
)
from noodl.apps.street_aq.exposure import (
    Q_INHALATION,
    exposure_reduction_adjoint,
    exposure_reduction_forward,
    street_population,
    total_exposure,
)
from noodl.apps.street_aq.network import (
    Street,
    StreetNetwork,
    build_model,
    from_test_network,
    initial_drivers,
    initial_state,
    munich_idealised,  # noqa: F401  (alias)
    street_geometry,
    street_index,
    twelve_street_grid,
)
from noodl.apps.street_aq.plume import (
    GAUSS_CUTOFF_SIGMAS,
    SEUIL_GAUSS,  # noqa: F401  (alias)
    PlumeTable,
    StreetSources,
    downwind_cutoff,
    junction_kernel,
    junction_sources,
    plume_kernel,
    plume_table,
    source_points,
    street_kernel,
)
from noodl.apps.street_aq.puff import (
    ALONG_WIND,
    JUNCTION_SOURCES,
    PuffClosure,
    PuffField,
    PuffOptions,
    PuffState,
    PuffTables,
    PulseSource,
    advance,
    concentration_at,
    emit,
    housekeep,
    mass_in_streets,
    pending_roof_mass,
    puff_mass,
    puff_tables,
    pulse_mass,
    roof_concentration,
)
from noodl.apps.street_aq.reactions import (
    K_NO_O3,
    K_NO_O3_298,
    MOLAR_MASS,
    Photostationary,
    k_no_o3_jpl2003,
    k_no_o3_munich,  # noqa: F401  (alias)
    molar_volume,
)
from noodl.apps.street_aq.report import (
    from_ug_m3,
    to_ug_m3,
    write_network_concentration,
)
from noodl.apps.street_aq.routing import (
    StreetFlows,
    StreetGeometry,
    direction_offsets,
    exact_gaussian_direction_samples,
    n_theta_munich,  # noqa: F401  (alias)
    n_theta_rectangle_rule,
    node_closure,
    routing_matrix,
    sigma_theta_munich,  # noqa: F401  (alias)
    sigma_theta_turbulence_intensity,
    sirane_direction_samples,  # noqa: F401  (alias)
)

# Reference-model aliases (the imports marked `# noqa: F401  (alias)`) are importable from
# here but kept out of `__all__`: an alias is named in its target's docstring, not catalogued.
__all__ = [
    "ALONG_WIND",
    "ASPECT_RATIO_EXCHANGE_BETA",
    "EULER_GAMMA_TRUNCATED",
    "EXCHANGE_SIGMA_W_RATIO",
    "GAUSS_CUTOFF_SIGMAS",
    "JUNCTION_SOURCES",
    "K_NO_O3",
    "K_NO_O3_298",
    "KAPPA_040",
    "KAPPA_041",
    "MOLAR_MASS",
    "OPTIONS",
    "PRESETS",
    "Q_INHALATION",
    "Z0_B_DEFAULT",
    "Z0_S_DEFAULT",
    "BoundaryLayer",
    "Photostationary",
    "PlumeTable",
    "PuffClosure",
    "PuffField",
    "PuffOptions",
    "PuffState",
    "PuffTables",
    "PulseSource",
    "Street",
    "StreetCase",
    "StreetFlows",
    "StreetResults",
    "StreetSources",
    "StreetGeometry",
    "StreetNetwork",
    "advance",
    "bessel_shape_parameter",
    "boundary_layer",
    "build_model",
    "canopy_displacement_roughness",
    "canyon_velocity",
    "concentration_at",
    "direction_offsets",
    "downwind_cutoff",
    "drivers_at",
    "emit",
    "exact_gaussian_direction_samples",
    "exchange_velocity",
    "grid_shape_parameter",
    "exposure_reduction_adjoint",
    "exposure_reduction_forward",
    "from_test_network",
    "from_ug_m3",
    "housekeep",
    "initial_drivers",
    "initial_state",
    "j_no2_elevation_cloud",
    "j_no2_zenith_table",
    "junction_kernel",
    "junction_sources",
    "k_no_o3_jpl2003",
    "k_no_o3_soulhac2011",
    "mass_in_streets",
    "molar_volume",
    "n_theta_rectangle_rule",
    "node_closure",
    "pending_roof_mass",
    "photostationary_for_streets",
    "plume_kernel",
    "plume_table",
    "puff_mass",
    "puff_tables",
    "pulse_mass",
    "read_case",
    "read_results",
    "roof_concentration",
    "roof_wind",
    "routing_matrix",
    "shape_parameter",
    "sigma_theta_turbulence_intensity",
    "solar_elevation",
    "source_points",
    "street_geometry",
    "street_index",
    "street_kernel",
    "street_population",
    "street_steady",
    "street_steady_with_plume",
    "to_ug_m3",
    "total_exposure",
    "twelve_street_grid",
    "write_case",
    "write_network_concentration",
    "write_sweep",
]

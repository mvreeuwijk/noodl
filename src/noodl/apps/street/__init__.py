"""The street application: a SIRANE/MUNICH-type street-network dispersion model.

A model built ON TOP of the core package (milestone 3 spec section 1). The core supplies
every solve; this package decides the network, the closures and the units. Nothing under
`src/noodl/` outside `apps/` imports from here.
"""

from noodl.apps.street.canyon import (
    GAMMA_E,
    KAPPA_IMPAQ,
    KAPPA_MUNICH,
    SCHULTE_BETA,
    SIRANE_EXCHANGE,
    Z0_B_DEFAULT,
    Z0_S_DEFAULT,
    BoundaryLayer,
    boundary_layer,
    canyon_velocity,
    exchange_velocity,
    macdonald_profile,
    roof_wind,
    soulhac_shape,
)
from noodl.apps.street.chemistry import photostationary_for_streets, street_steady
from noodl.apps.street.loader import RHO_AIR, AqdtData, Forcing, read_aqdt
from noodl.apps.street.network import (
    Street,
    StreetNetwork,
    build_street_model,
    from_test_network,
    initial_state,
    munich_idealised,
    street_geometry,
    street_index,
)
from noodl.apps.street.report import (
    from_ug_m3,
    to_ug_m3,
    write_network_concentration,
)
from noodl.apps.street.routing import (
    StreetFlows,
    StreetGeometry,
    direction_offsets,
    n_theta_munich,
    node_closure,
    routing_matrix,
    sigma_theta_munich,
)

__all__ = [
    "GAMMA_E",
    "KAPPA_IMPAQ",
    "KAPPA_MUNICH",
    "RHO_AIR",
    "SCHULTE_BETA",
    "SIRANE_EXCHANGE",
    "Z0_B_DEFAULT",
    "Z0_S_DEFAULT",
    "AqdtData",
    "BoundaryLayer",
    "Forcing",
    "Street",
    "StreetFlows",
    "StreetGeometry",
    "StreetNetwork",
    "boundary_layer",
    "build_street_model",
    "canyon_velocity",
    "direction_offsets",
    "exchange_velocity",
    "from_test_network",
    "from_ug_m3",
    "initial_state",
    "macdonald_profile",
    "munich_idealised",
    "n_theta_munich",
    "node_closure",
    "photostationary_for_streets",
    "read_aqdt",
    "roof_wind",
    "routing_matrix",
    "sigma_theta_munich",
    "soulhac_shape",
    "street_geometry",
    "street_index",
    "street_steady",
    "to_ug_m3",
    "write_network_concentration",
]

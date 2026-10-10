"""The node and edge attributes the building application reads off a `Network`.

A building is parameterised through attributes set with `net.add_node(...)` and
`net.add_edge(...)` (or by `add_zone`, `add_large_opening` and the CONTAM reader), which the
layer builders, elements and drives read by name. This module is the one place that says
what each attribute means and in which unit, and `scripts/gen_catalogue.py` renders it.
Each value is `(meaning with unit, who reads it)`.
"""

NODE_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "volume": ("air volume of the zone (m3); 0 for a wall node", "thermal_layer, species_layer"),
    "T0": ("initial temperature (K), also the boundary temperature of a boundary node",
           "initial_state, initial_drivers, project_to_model"),
    "z_ref": ("height of the node's reference pressure (m)", "Stack.from_network"),
    "heat_capacity": ("extra lumped heat capacity of the node (J/K); a wall node's capacity",
                      "thermal_layer"),
}

EDGE_ATTRIBUTES: dict[str, dict[str, tuple[str, str]]] = {
    "airpath": {
        "z_path": ("height of the opening (m)", "Stack.from_network"),
        "Cd": ("discharge coefficient (dimensionless)", "orifice_elements_from_edges"),
        "area": ("opening area (m2)", "orifice_elements_from_edges"),
        "name": ("label of the edge in model.refs and in named results (optional)",
                 "noodl.refs"),
    },
    "airpath, to or from the ambient node": {
        "azimuth": ("facade direction, in the unit of the theta_w driver (degrees from CONTAM)",
                    "Wind.from_network"),
        "Cp": ("constant wind pressure coefficient when the edge has no profile",
               "Wind.from_network"),
        "Ch": ("wind speed modifier (dimensionless, 1 by default)", "Wind.from_network"),
        "profile": ("number of the WindProfile to use; 0 means the constant Cp",
                    "Wind.from_network"),
    },
    "wall": {
        "ua": ("conductance zone -> wall or wall -> ambient (W/K)", "thermal_layer"),
    },
}

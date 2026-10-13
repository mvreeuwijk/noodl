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
    "position": ("absolute (x, y, z) of the node's component (m), set by Component.flatten; "
                 "for drawing only", "noodl.components"),
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
        "facade": ("key of the building's inner_table('facades', ...); flatten writes the "
                   "facade's azimuth", "Component.flatten"),
        "Ch": ("wind speed modifier (dimensionless, 1 by default)", "Wind.from_network"),
        "profile": ("number of the WindProfile to use; 0 means the constant Cp",
                    "Wind.from_network"),
    },
    "duct, a segment between two junctions (made by DuctNetwork)": {
        "length": ("length of the segment (m)", "duct_elements_from_edges"),
        "diameter": ("hydraulic diameter of the duct (m)", "duct_elements_from_edges"),
        "roughness": ("roughness of the duct wall (m)", "duct_elements_from_edges"),
        "loss": ("sum of the segment's dynamic loss coefficients (dimensionless)",
                 "duct_elements_from_edges"),
        "area": ("cross-section of the duct (m2); the circle of the diameter by default",
                 "duct_elements_from_edges"),
    },
    "duct_terminal, a junction to a zone (made by DuctNetwork)": {
        "area": ("area of the duct at the terminal (m2)", "duct_elements_from_edges"),
        "loss": ("loss coefficient of the terminal (dimensionless)",
                 "duct_elements_from_edges"),
        "z_path": ("height of the terminal, the junction's height (m)", "Stack.from_network"),
        "profile": ("number of the WindProfile on a terminal in the ambient; absent or 0 "
                    "for none", "duct_elements_from_edges, Wind.from_network"),
        "Cp": ("constant wind pressure coefficient of a terminal in the ambient without a "
               "profile", "duct_elements_from_edges, Wind.from_network"),
        "Ch": ("wind speed modifier (dimensionless, 1 by default)", "Wind.from_network"),
        "azimuth": ("wall direction of a terminal with a profile (degrees from CONTAM)",
                    "Wind.from_network"),
    },
    "air_handler, an air handler's edges (made by add_air_handler)": {
        "role": ("what the edge is: 0 supply point (plenum -> zone), 1 return point (zone -> "
                 "plenum), 2 recirculation, 3 outdoor air, 4 exhaust", "air_handler_from_edges"),
        "design_flow": ("design mass flow of a supply or return point (kg/s); 0 on the other "
                        "edges", "air_handler_from_edges"),
        "outdoor_air_fraction": ("on the recirculation edge: the outdoor-air fraction of the "
                                 "supply when no input sets it (dimensionless, 1 by default)",
                                 "air_handler_from_edges"),
        "min_outdoor_air": ("on the recirculation edge: the minimum outdoor-air flow (kg/s, 0 "
                            "by default)", "air_handler_from_edges"),
    },
    "fan, a constant-mass-flow fan (made by the fan component)": {
        "flow": ("mass flow delivered from source to target whatever the pressures (kg/s)",
                 "fan_elements_from_edges"),
    },
    "wall": {
        "ua": ("conductance zone -> wall or wall -> ambient (W/K)", "thermal_layer"),
    },
}

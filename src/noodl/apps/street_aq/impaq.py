"""A faithful numpy/scipy port of the IMPAQ prototype, as the comparison ORACLE.

`aqdt/impaq.py` (483 lines, byte-identical in both AQ_DT copies) is reproduced here so that
the parity tests need no import from the AQ_DT repository, and so that its two remaining
documented issues can be switched on and off one at a time and their effect reported
separately (framework spec section 9). Nothing here is used by the model: this module is
numpy and scipy, it is not differentiable, and nothing else in `apps/street_aq/` imports it.

THE THIRD ISSUE IS RETRACTED. The prototype's docstring calls its ventilation coefficient
`sigma_w(H) W L / (sqrt(2) pi)` an error against `sigma_w(H) W L / sqrt(2 pi)` ("issue C").
It is not an error: Soulhac et al. 2011 Eq. (5), Kim et al. 2018 Eq. (3) and Kim et al.
2022 Eq. (B10) all typeset `sigma_w/(sqrt(2) pi)` -- the radical covers only the 2, checked
at glyph level in all three PDFs -- and MUNICH implements exactly that at
`StreetNetworkTransport.cxx:3273`. There is therefore no `fix_c`, and the retraction is
recorded in the README against both the framework spec and the prototype's docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Intersections:
    x: np.ndarray
    y: np.ndarray
    node_ids: np.ndarray


@dataclass(slots=True)
class Roads:
    s: np.ndarray
    e: np.ndarray
    width_m: np.ndarray
    height_m: np.ndarray
    roughness_m: np.ndarray
    emission_rate: np.ndarray
    length_m: np.ndarray | None = None
    road_angle_rad: np.ndarray | None = None
    center_x: np.ndarray | None = None
    center_y: np.ndarray | None = None
    canyon_velocity_mps: np.ndarray | None = None


@dataclass(slots=True)
class ImpaqNetwork:
    intersections: Intersections
    roads: Roads


@dataclass(slots=True)
class ImpaqBoundaryLayer:
    background_concentration: float
    wind_speed_mps: float
    wind_angle_rad: float
    reference_height_m: float
    abl_height_m: float
    displacement_height_m: float
    roughness_height_m: float
    friction_velocity_mps: float

    def sigma_w(self, z_m: np.ndarray | float) -> np.ndarray | float:
        return 1.3 * self.friction_velocity_mps * (1 - 0.8 * np.asarray(z_m) / self.abl_height_m)


def build_test_network() -> ImpaqNetwork:
    intersections = Intersections(
        x=np.array([0.0, 300.0, 600.0, 300.0]),
        y=np.array([400.0, 300.0, 400.0, 0.0]),
        node_ids=np.array([0, 1, 2, 3]),
    )
    roads = Roads(
        s=np.array([0, 1, 3], dtype=int),
        e=np.array([1, 2, 1], dtype=int),
        width_m=np.array([20.0, 10.0, 20.0]),
        height_m=np.array([20.0, 20.0, 30.0]),
        roughness_m=np.array([0.15, 0.15, 0.15]),
        emission_rate=np.array([1.0, 1.0, 1.0]),
    )
    network = ImpaqNetwork(intersections=intersections, roads=roads)
    compute_road_geometry(network)
    return network


def compute_road_geometry(network: ImpaqNetwork) -> None:
    roads = network.roads
    intersections = network.intersections
    dx = intersections.x[roads.e] - intersections.x[roads.s]
    dy = intersections.y[roads.e] - intersections.y[roads.s]
    roads.length_m = np.sqrt(dx**2 + dy**2)
    roads.road_angle_rad = np.arctan2(dy, dx)
    roads.center_x = 0.5 * (intersections.x[roads.s] + intersections.x[roads.e])
    roads.center_y = 0.5 * (intersections.y[roads.s] + intersections.y[roads.e])


def compute_boundary_layer(
    network: ImpaqNetwork,
    background_concentration: float,
    wind_speed_mps: float,
    wind_angle_rad: float,
    reference_height_m: float = 30.0,
    abl_height_m: float = 1200.0,
) -> ImpaqBoundaryLayer:
    mean_height = float(np.mean(network.roads.height_m))
    displacement_height_m = 2.0 * mean_height / 3.0
    roughness_height_m = mean_height / 10.0
    friction_velocity_mps = 0.4 * wind_speed_mps / math.log(
        (reference_height_m - displacement_height_m) / roughness_height_m
    )
    return ImpaqBoundaryLayer(
        background_concentration=background_concentration,
        wind_speed_mps=wind_speed_mps,
        wind_angle_rad=wind_angle_rad,
        reference_height_m=reference_height_m,
        abl_height_m=abl_height_m,
        displacement_height_m=displacement_height_m,
        roughness_height_m=roughness_height_m,
        friction_velocity_mps=friction_velocity_mps,
    )


def canyon_velocity(
    roads: Roads, friction_velocity_mps: float, wind_angle_rad: float
) -> np.ndarray:
    from scipy.optimize import fsolve
    from scipy.special import jv, yv

    gamma = 0.577
    kappa = 0.4
    velocity = np.zeros(len(roads.s), dtype=float)
    for idx in range(len(roads.s)):
        width = float(roads.width_m[idx])
        height = float(roads.height_m[idx])
        roughness = float(roads.roughness_m[idx])
        phi = float(roads.road_angle_rad[idx] - wind_angle_rad)
        di = min(width / 2.0, height)

        def equation(c_val: np.ndarray, roughness=roughness, di=di) -> np.ndarray:
            c_scalar = float(c_val[0])
            return np.array(
                [
                    0.5 * roughness / di * c_scalar
                    - math.exp(math.pi / 2.0 * yv(1, c_scalar) / jv(1, c_scalar) - gamma)
                ]
            )

        c_solution = fsolve(equation, np.array([1.0]))
        c_val = float(c_solution[0])
        alpha = math.log(di / roughness)
        beta = math.exp(c_val / math.sqrt(2.0) * (1.0 - height / di))
        u_h = friction_velocity_mps * math.sqrt(
            math.pi
            / (math.sqrt(2.0) * kappa**2 * c_val)
            * (yv(0, c_val) - jv(0, c_val) * yv(1, c_val) / jv(1, c_val))
        )
        velocity[idx] = (
            u_h
            * math.cos(phi)
            * di**2
            / width
            / height
            * (
                2.0 * math.sqrt(2.0) / c_val * (1.0 - beta)
                * (1.0 - c_val**2 / 3.0 + c_val**4 / 45.0)
                + beta * (2.0 * alpha - 3.0) / alpha
                + (width / di - 2.0) * (alpha - 1.0) / alpha
            )
        )
    return velocity


def build_physical_topology(network: ImpaqNetwork) -> np.ndarray:
    topology = np.zeros((len(network.intersections.x), len(network.roads.s)), dtype=float)
    for idx in range(len(network.roads.s)):
        topology[network.roads.s[idx], idx] = 1.0
        topology[network.roads.e[idx], idx] = -1.0
    return topology


def flow_route(fluxes: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    fluxes = np.asarray(fluxes, dtype=float).flatten()
    angles_rad = np.asarray(angles_rad, dtype=float).flatten()
    if fluxes.shape != angles_rad.shape:
        raise ValueError("Flux and angle vectors must have the same shape.")

    count = len(fluxes)
    order = np.argsort(angles_rad)
    fluxes = fluxes[order]
    angles_rad = angles_rad[order]

    routing = np.zeros((count + 1, count + 1), dtype=float)
    source_idx = np.where(fluxes > 0)[0]
    sink_idx = np.where(fluxes < 0)[0]

    original_fluxes = fluxes.copy()
    total_in = np.sum(fluxes[source_idx])
    total_out = np.sum(fluxes[sink_idx])
    imbalance = total_in + total_out

    if imbalance > 0:
        for idx in source_idx:
            delta = fluxes[idx] / total_in * abs(imbalance)
            routing[idx, count] = delta
            routing[count, idx] = -delta
            fluxes[idx] -= delta
    elif imbalance < 0:
        for idx in sink_idx:
            delta = fluxes[idx] / total_out * abs(imbalance)
            routing[idx, count] = -delta
            routing[count, idx] = delta
            fluxes[idx] += delta

    iterations = 0
    while np.sum(np.abs(fluxes)) > 1e-10:
        source_idx = np.where(fluxes > 0)[0]
        sink_idx = np.where(fluxes < 0)[0]
        left_sink = np.zeros(len(source_idx), dtype=int)
        left_flux = np.zeros(len(source_idx), dtype=float)
        right_sink = np.zeros(len(source_idx), dtype=int)
        right_flux = np.zeros(len(source_idx), dtype=float)

        for n, source in enumerate(source_idx):
            for offset in range(1, count):
                idx = (source - offset) % count
                left_sink[n] = idx
                if fluxes[idx] > 0:
                    left_flux[n] = 0.0
                    break
                if fluxes[idx] < 0:
                    left_flux[n] = -fluxes[idx]
                    break
            for offset in range(1, count):
                idx = (source + offset) % count
                right_sink[n] = idx
                if fluxes[idx] > 0:
                    right_flux[n] = 0.0
                    break
                if fluxes[idx] < 0:
                    right_flux[n] = -fluxes[idx]
                    break

        for n, source in enumerate(source_idx):
            delta_left = min(left_flux[n], right_flux[n], fluxes[source] / 2.0)
            delta_right = delta_left
            left_full = left_flux[n] - delta_left == 0
            right_full = right_flux[n] - delta_right == 0
            total_delta = delta_left + delta_right
            if left_full and not right_full:
                delta_right += min(fluxes[source] - total_delta, right_flux[n] - delta_right)
            elif right_full and not left_full:
                delta_left += min(fluxes[source] - total_delta, left_flux[n] - delta_left)
            routing[source, left_sink[n]] += delta_left
            routing[left_sink[n], source] -= delta_left
            routing[source, right_sink[n]] += delta_right
            routing[right_sink[n], source] -= delta_right

        fluxes = original_fluxes - np.sum(routing[:count, :], axis=1)

        for sink in sink_idx:
            if fluxes[sink] > 0:
                left_sources = np.where(left_sink == sink)[0]
                right_sources = np.where(right_sink == sink)[0]
                delta = fluxes[sink] / 2.0
                if left_sources.size:
                    routing[sink, source_idx[left_sources[0]]] += delta
                    routing[source_idx[left_sources[0]], sink] -= delta
                if right_sources.size:
                    routing[sink, source_idx[right_sources[0]]] += delta
                    routing[source_idx[right_sources[0]], sink] -= delta

        fluxes = original_fluxes - np.sum(routing[:count, :], axis=1)
        iterations += 1
        if iterations > count:
            raise RuntimeError("No convergence in flow routing.")

    inverse_order = np.append(order, count)
    routing = routing[np.ix_(inverse_order, inverse_order)]
    routing[routing < 0] = 0.0
    routing[np.abs(routing) < 1e-10] = 0.0
    return routing


def compute_intersection_routing(network: ImpaqNetwork) -> tuple[np.ndarray, np.ndarray]:
    topology = build_physical_topology(network)
    if network.roads.canyon_velocity_mps is None:
        raise ValueError("Canyon velocities must be computed before routing.")
    road_fluxes = network.roads.canyon_velocity_mps * network.roads.width_m * network.roads.height_m
    n_roads = len(network.roads.s)
    routing = np.zeros((n_roads + 1, n_roads + 1), dtype=float)

    for node_idx in range(len(network.intersections.x)):
        local_flux = -road_fluxes * topology[node_idx, :]
        road_indices = np.where(np.abs(local_flux) > 0)[0]
        if road_indices.size == 0:
            continue
        phi = (
            network.roads.road_angle_rad[road_indices]
            + (topology[node_idx, road_indices] > 0) * math.pi
            + math.pi
        ) % (2.0 * math.pi) - math.pi
        local_routing = flow_route(local_flux[road_indices], phi)
        full_indices = np.append(road_indices, len(network.roads.s))
        routing[np.ix_(full_indices, full_indices)] += local_routing
    return topology, routing


def build_transport_system(
    network: ImpaqNetwork,
    boundary_layer: ImpaqBoundaryLayer,
    *,
    fix_a: bool = False,
    fix_b: bool = False,
):
    """`impaq.py:298-406`, with the two remaining issues switchable.

    `fix_a=False` reproduces the prototype: the advective edge parameters are the raw +-1
    of the topology matrix, so the canyon velocity, the street cross-section and the
    routing matrix never enter the coupling at all. `fix_a=True` applies the fix the
    prototype's own docstring prescribes -- replace the topology-derived edges by the
    routing matrix `P` from `compute_intersection_routing`, whose entries are ALREADY
    `canyon_velocity * width * height` weighted volume flows. `P` has an environment ROW
    as well as an environment COLUMN, and both are ordinary advective edges: the column is
    what a road sends to the atmosphere at an intersection, the row is what the atmosphere
    sends back into a road. Taking only the column leaves a hole in the mass balance worth
    4.7 % on `build_test_network`; taking both puts this oracle within 4.2e-16 of the
    noodl model.

    `fix_b=False` reproduces the prototype's `n_state = max(n_intersections, n_roads) + 1`,
    which leaves all-zero phantom rows whenever there are more intersections than roads.
    `fix_b=True` uses the correct `n_roads + 1`.
    """
    roads = network.roads
    if roads.length_m is None or roads.canyon_velocity_mps is None:
        raise ValueError(
            "Road geometry and canyon velocity must be computed before system assembly."
        )
    n_roads = len(roads.s)
    n_intersections = len(network.intersections.x)
    if fix_b and not fix_a and n_intersections > n_roads:
        # Without fix_a the advective edges are indexed by INTERSECTION, so they need the
        # oversized state that issue B is about. Shrinking it to n_roads + 1 rows aliases
        # an intersection onto the environment row (4 intersections, 3 roads) or indexes
        # past the end of the matrix (230 intersections, 162 roads on leiden_small). The
        # two fixes are not independent, and saying so beats a corrupted answer.
        raise ValueError(
            f"impaq.build_transport_system: fix_b without fix_a is only defined when "
            f"there are no more intersections than roads; this network has "
            f"{n_intersections} intersections and {n_roads} roads, and issue A's edges "
            f"are indexed by intersection, so they do not fit in the n_roads + 1 = "
            f"{n_roads + 1} rows issue B's fix allows. Fix issue A too, or leave both"
        )
    if fix_b:
        n_state = n_roads + 1
    else:
        n_state = max(n_intersections, n_roads) + 1
    env_idx = n_state - 1

    if fix_a:
        _topology, routing = compute_intersection_routing(network)
        rows, cols = np.nonzero(routing[:n_roads, :n_roads])
        env_rows = np.nonzero(routing[:n_roads, n_roads])[0]
        env_cols = np.nonzero(routing[n_roads, :n_roads])[0]
        advective_start = np.concatenate([
            rows, env_rows, np.full(len(env_cols), env_idx, dtype=int)
        ])
        advective_end = np.concatenate([
            cols, np.full(len(env_rows), env_idx, dtype=int), env_cols
        ])
        advective_param = np.concatenate([
            routing[rows, cols], routing[env_rows, n_roads], routing[n_roads, env_cols]
        ])
    else:
        topology = build_physical_topology(network)
        positive_idx = np.where(topology.flatten() > 0)[0]
        advective_start, advective_end = np.unravel_index(positive_idx, topology.shape)
        advective_param = topology.flatten()[positive_idx]

    n_transport = len(advective_start)
    edge_start = np.zeros(n_transport + 2 * n_roads, dtype=int)
    edge_end = np.zeros(n_transport + 2 * n_roads, dtype=int)
    edge_type = np.zeros(n_transport + 2 * n_roads, dtype=int)
    edge_param = np.zeros(n_transport + 2 * n_roads, dtype=float)
    edge_start[:n_transport] = advective_start
    edge_end[:n_transport] = advective_end
    edge_param[:n_transport] = advective_param
    edge_type[:n_transport] = 1

    for idx in range(n_roads):
        edge_start[n_transport + idx] = idx
        edge_end[n_transport + idx] = env_idx
        # NOT an error -- see the module docstring. This is SIRANE's and MUNICH's own
        # coefficient, and there is no fix_c.
        edge_param[n_transport + idx] = (
            float(boundary_layer.sigma_w(roads.height_m[idx]))
            * roads.width_m[idx] * roads.length_m[idx] / math.sqrt(2.0) / math.pi
        )
        edge_type[n_transport + idx] = 2
    for idx in range(n_roads):
        edge_start[n_transport + n_roads + idx] = env_idx
        edge_end[n_transport + n_roads + idx] = idx
        edge_param[n_transport + n_roads + idx] = roads.emission_rate[idx]
        edge_type[n_transport + n_roads + idx] = 3

    transport = np.zeros((n_state, len(edge_start)), dtype=float)
    for idx in range(len(edge_start)):
        transport[edge_start[idx], idx] = 1.0
        transport[edge_end[idx], idx] = -1.0
    t_pos = transport.copy()
    t_pos[transport < 0] = 0.0
    a = np.zeros((len(edge_start), n_state), dtype=float)
    b = np.zeros(len(edge_start), dtype=float)
    for idx in np.where(edge_type == 1)[0]:
        a[idx, :] = t_pos[:, idx] * edge_param[idx]
    for idx in np.where(edge_type == 2)[0]:
        a[idx, :] = transport[:, idx] * edge_param[idx]
    for idx in np.where(edge_type == 3)[0]:
        b[idx] = edge_param[idx]
    return transport, a, b, edge_type


def solve_steady_state(
    network: ImpaqNetwork,
    boundary_layer: ImpaqBoundaryLayer,
    *,
    fix_a: bool = False,
    fix_b: bool = False,
) -> np.ndarray:
    """`(n_roads + 1,)`: one concentration per road, then the environment's."""
    transport, a, b, _edge_type = build_transport_system(
        network, boundary_layer, fix_a=fix_a, fix_b=fix_b
    )
    matrix = transport @ a
    rhs = -transport @ b
    matrix[-1, :] = 0.0
    matrix[-1, -1] = 1.0
    rhs[-1] = boundary_layer.background_concentration
    try:
        solution = np.linalg.solve(matrix, rhs)
    # Kept verbatim from the prototype (it is the oracle); if this ever fires a parity
    # number silently becomes a least-squares answer -- see the diagnosis tests in
    # tests/verification/test_street_parity.py.
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    n_roads = len(network.roads.s)
    return np.concatenate([solution[:n_roads], solution[-1:]])


def network_from_street_network(net, emission: np.ndarray) -> ImpaqNetwork:
    """A `StreetNetwork` (Task 5) and a per-street emission as an `ImpaqNetwork`.

    The junction order is the `StreetNetwork`'s own (`net.junctions`), so road index `i`
    here is street `i` there and the two models can be compared row by row. `z0_b` becomes
    `roughness_m`, which is what the prototype's `canyon_velocity` reads.
    """
    junctions = net.junctions
    index = {name: j for j, name in enumerate(junctions)}
    intersections = Intersections(
        x=np.array([float(net.x[name]) for name in junctions]),
        y=np.array([float(net.y[name]) for name in junctions]),
        node_ids=np.arange(len(junctions), dtype=int),
    )
    roads = Roads(
        s=np.array([index[s.u] for s in net.streets], dtype=int),
        e=np.array([index[s.v] for s in net.streets], dtype=int),
        width_m=np.array([float(s.width) for s in net.streets]),
        height_m=np.array([float(s.height) for s in net.streets]),
        roughness_m=np.array([float(s.z0_b) for s in net.streets]),
        emission_rate=np.asarray(emission, dtype=float),
    )
    out = ImpaqNetwork(intersections=intersections, roads=roads)
    compute_road_geometry(out)
    return out

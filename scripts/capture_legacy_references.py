"""Capture John Craske's 2019 Tellegen tutorial results as golden references.

Run ONCE, in a throwaway environment with numpy<2, scipy, networkx<3 and autograd, from the
repository root, before legacy/ is deleted:
    <legacy-venv>/python scripts/capture_legacy_references.py
Writes tests/golden/legacy/tutorial.json. The values are the ORIGINAL package's output; the
noodl parity tests compare against them.
"""
import json
import sys

sys.path.insert(0, "legacy")

import autograd.numpy as np  # noqa: I001 -- legacy/ must be on sys.path first

from Tellegen.circuit import Circuit, Constitutive, H
from Tellegen.topology import Graph

out = {}
# 1. quadratic-drag loop
g = Graph.from_edges([(0, 1), (1, 2), (2, 0)])


@Constitutive
def loop(p, q, a):
    return [p[0] - a[0], p[1] - a[1] * abs(q[1]) * q[1], p[2] - a[2] * abs(q[2]) * q[2]]


G = Circuit(g, loop)
for label, a in (("a111", [1.0, 1.0, 1.0]), ("a211", [2.0, 1.0, 1.0])):
    x = G(np.array(a))
    out[f"quadratic_loop_{label}"] = {
        "p": G.p(x).tolist(),
        "q": G.q(x).tolist(),
        "dp_da": G.state.dp.tolist(),
        "dq_da": G.state.dq.tolist(),
    }
out["partitions_triangle"] = int(g.partitions())


# 2. spring-mass-damper, 50 implicit steps of 0.2
@Constitutive
def smd(p, q, a):
    return [q[0] - a[3] - a[5] * p[0], p[1] - a[1] * q[1], p[2] - a[4] - a[5] * q[2]]


G = Circuit(g, smd)
a = np.array([1.0, 0.2, 1.0, 0.0, 1.0, 0.2])
xs = [1.0]
for _ in range(50):
    m = G(a)
    p, q = G.p(m), G.q(m)
    a = np.array([a[0], a[1], a[2], q[0], p[2], a[5]])
    xs.append(float(p[2] / a[2]))
out["spring_mass_damper_displacement"] = xs

# 3. three-zone contaminant exchange, 50 steps of 0.2
g3 = Graph.from_edges([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)])


@Constitutive
def exchange(p, q, a):
    V0, V1, V2, p0, p1, p2, Q, dt = a
    return [
        p[0] - p0 + dt * q[0] / V0,
        p[1] - p1 + dt * q[1] / V1,
        p[2] - p2 + dt * q[2] / V2,
        q[3] - H(Q) * p[0] + H(-Q) * p[1],
        q[4] - H(Q) * p[1] + H(-Q) * p[2],
        q[5] - H(Q) * p[2] + H(-Q) * p[0],
    ]


G = Circuit(g3, exchange)
a = np.array([10.0, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.2])
series = [[2.0, 1.0]]
for _ in range(50):
    x = G(a)
    p = G.p(x)
    a = np.array([a[0], a[1], a[2], p[0], p[1], p[2], a[6], a[7]])
    series.append([float(p[1]), float(p[2])])
out["three_zone_exchange_rooms"] = series
out["partitions_six_edge_graph"] = int(g3.partitions())
json.dump(out, open("tests/golden/legacy/tutorial.json", "w"), indent=1)

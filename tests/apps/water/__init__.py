"""Shared constants of the water tests."""

# A closed link's flow is an unknown of the Newton solve whose row is `kappa q = 0`, so it
# is zero to round-off, not bit-exactly, and the last bit depends on the BLAS. (Flows are
# ~1e-2 m3/s, up to ~0.1 m3/s on Net1's pump; eps ~ 2e-16: well below 1e-15.) "Round-off"
# holds because the row is met exactly by a FULL Newton step (`newton` switches omega to 1
# once the residual ratio is below 0.5, `src/noodl/solvers/newton.py`); a solve that
# converged while still damped could leave up to ~atol/kappa on a closed link.
ZERO_FLOW = 1e-15

`case9.m`, `case14.m`, `case30.m` and `case118.m` are MATPOWER case files, copied unchanged
from MATPOWER 8.1 (`data/`, as distributed in the PyPI package `matpower` 8.1.0.2.3.0). Their
SHA-256 digests (LF line endings) are recorded in the matching `.json` files and checked by
`tests/verification/test_power_flow_parity.py`.

MATPOWER's own licence states: "The code in MATPOWER is distributed under the 3-clause BSD
license below. The MATPOWER case files distributed with MATPOWER are not covered by the BSD
license. In most cases, the data has either been included with permission or has been
converted from data available from a public source." These four files are therefore covered
neither by MATPOWER's BSD licence nor by noodl physics' BSD-3 licence; they are redistributed
here as test data, with their headers (which name their sources) intact, as pandapower and
PYPOWER redistribute the same cases. Their sources, as the headers state:

- `case9.m`: J. H. Chow (ed.), *Time-Scale Modeling of Dynamic Networks with Applications to
  Power Systems*, Springer-Verlag, 1982, p. 70 (from R. P. Schulz, A. E. Turner and
  D. N. Ewart, "Long Term Power System Dynamics", EPRI Report 90-7-0, 1974).
- `case14.m`, `case118.m`: the IEEE 14-bus and 118-bus test cases, IEEE Common Data Format
  files from the University of Washington Power Systems Test Case Archive
  (https://labs.ece.uw.edu/pstca/), converted by MATPOWER's `cdf2matp`.
- `case30.m`: O. Alsac and B. Stott, "Optimal Load Flow with Steady State Security", IEEE
  Trans. PAS 93(3), 745-751, 1974, with generator data from R. W. Ferrero,
  S. M. Shahidehpour and V. C. Ramesh, IEEE Trans. Power Systems 12(3), 1340-1347, 1997.

MATPOWER asks that work using it cite R. D. Zimmerman, C. E. Murillo-Sánchez and
R. J. Thomas, "MATPOWER: Steady-State Operations, Planning and Analysis Tools for Power
Systems Research and Education", IEEE Trans. Power Systems 26(1), 12-19, 2011.

`case9_features.m` is written for this repository from `case9.m`'s data, modified as its
header says. The `.json` files are reference results computed by
`scripts/power_flow_reference.py` (pandapower 3.5.6 checked against PYPOWER 5.1.21) and are
part of noodl physics.

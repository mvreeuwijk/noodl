The `.json` files here are reference power-flow results for MATPOWER's `case9`, `case14`,
`case30` and `case118` and for `case9_features`. They are computed by
`scripts/power_flow_reference.py` (pandapower 3.5.6 checked against PYPOWER 5.1.21) and are
part of noodl physics.

The MATPOWER case files themselves are not in this repository. MATPOWER's own licence states:
"The code in MATPOWER is distributed under the 3-clause BSD license below. The MATPOWER case
files distributed with MATPOWER are not covered by the BSD license. In most cases, the data
has either been included with permission or has been converted from data available from a
public source." noodl physics therefore does not redistribute them: `matpower_case`
(`noodl.apps.electrical.power`) downloads each on first use from MATPOWER's GitHub repository
at release 8.1 (tag `8.1`, commit `1a828c7af590714499284e36ee9c81273388c594`),
`https://raw.githubusercontent.com/MATPOWER/matpower/<commit>/data/<case>.m`, and caches it in
`$NOODL_CACHE_DIR/matpower/8.1/` (default `~/.cache/noodl/matpower/8.1/`), outside the
repository. Each file is checked against the SHA-256 (LF line endings) recorded as
`case_sha256_lf` in its `.json` file and in `MATPOWER_CASES`; these are the digests of the
files as distributed in MATPOWER 8.1 (identical in the PyPI package `matpower` 8.1.0.2.3.0).
The cases' sources, as their headers state:

- `case9`: J. H. Chow (ed.), *Time-Scale Modeling of Dynamic Networks with Applications to
  Power Systems*, Springer-Verlag, 1982, p. 70 (from R. P. Schulz, A. E. Turner and
  D. N. Ewart, "Long Term Power System Dynamics", EPRI Report 90-7-0, 1974).
- `case14`, `case118`: the IEEE 14-bus and 118-bus test cases, IEEE Common Data Format
  files from the University of Washington Power Systems Test Case Archive
  (https://labs.ece.uw.edu/pstca/), converted by MATPOWER's `cdf2matp`.
- `case30`: O. Alsac and B. Stott, "Optimal Load Flow with Steady State Security", IEEE
  Trans. PAS 93(3), 745-751, 1974, with generator data from R. W. Ferrero,
  S. M. Shahidehpour and V. C. Ramesh, IEEE Trans. Power Systems 12(3), 1340-1347, 1997.

MATPOWER asks that work using it cite R. D. Zimmerman, C. E. Murillo-Sánchez and
R. J. Thomas, "MATPOWER: Steady-State Operations, Planning and Analysis Tools for Power
Systems Research and Education", IEEE Trans. Power Systems 26(1), 12-19, 2011.

`case9_features` is `case9` modified for noodl physics (off-nominal taps and phase shifters, a
parallel branch, an out-of-service branch, a bus shunt with conductance, two generators on one
bus, a generator on a PQ bus, an out-of-service generator leaving a PV bus without generation,
and a slack angle of 10°). `matpower_case("case9_features")` derives it from the fetched
`case9` by the row edits in `src/noodl/apps/electrical/power/cases.py` and checks the result
against its own recorded SHA-256.

# MATPOWER `.m`

Reading a MATPOWER case file produces a `PowerNetwork` for the
[power flow](../applications/electrical.md#power-flow) of the electrical application.
`noodl.apps.electrical.power.read_matpower(path)` reads the case format of MATPOWER version 2
(`mpc.version = '2'`), the format of MATPOWER's standard cases, within the limits listed
[below](#what-is-read-ignored-and-refused) (no DC lines, no indexed assignments, no
`define_constants`). A case file is a MATLAB function that fills a struct `mpc`; the reader
parses its text and never runs MATLAB or Octave. The network keeps the file's units (MW, MVAr,
degrees, p.u. impedances on `baseMVA`); the power-flow builders convert to per unit.

A four-bus case written for this page (the network of the
[power-flow example](../applications/electrical.md#power-flow)), saved and read back:

```python
from pathlib import Path

from noodl.apps.electrical.power import build_power_flow, read_matpower, solve_power_flow

Path("four_bus.m").write_text("""function mpc = four_bus
%FOUR_BUS  A 132 kV triangle feeding a 33 kV bus through a tap-changing transformer.
mpc.version = '2';
mpc.baseMVA = 100;
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	3	0	0	0	0	1	1	0	132	1	1.1	0.9;
	2	2	20	5	0	0	1	1	0	132	1	1.1	0.9;
	3	1	60	20	0	0	1	1	0	132	1	1.1	0.9;
	4	1	45	15	0	8	1	1	0	33	1	1.1	0.9;
];
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
mpc.gen = [
	1	0	0	100	-100	1.02	100	1	200	0;
	2	50	0	60	-60	1.01	100	1	80	0;
];
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	2	0.01	0.06	0.03	0	0	0	0	0	1	-360	360;
	1	3	0.02	0.08	0.04	0	0	0	0	0	1	-360	360;
	2	3	0.015	0.07	0.02	0	0	0	0	0	1	-360	360;
	3	4	0	0.1	0	0	0	0	0.975	0	1	-360	360;
];
%	2	startup	shutdown	n	c2	c1	c0
mpc.gencost = [
	2	0	0	3	0.01	20	0;
	2	0	0	3	0.02	25	0;
];
""", encoding="utf-8")

net = read_matpower("four_bus.m")
print(net.name, len(net.buses), len(net.branches), net.notes["ignored_fields"])
print([(s.bus, s.g_mw, s.b_mvar) for s in net.shunts], net.branches[3].ratio)
model, state, drivers = build_power_flow(net)
solved = solve_power_flow(model, state, drivers)
print(model.power_index.bus_ids)
```

```text
four_bus 4 4 ['gencost']
[(4, 0.0, 8.0)] 0.975
(1, 2, 3, 4)
```

## What is read, ignored and refused

| | Fields and statements |
|---|---|
| **Read** | `mpc.version` (must be exactly `'2'`; any other version is refused), `mpc.baseMVA`, `mpc.bus` (at least 13 columns), `mpc.gen` (at least 10 columns), `mpc.branch` (at least 11 columns). Extra columns (solved-case or optimal-power-flow results) are ignored. |
| **Ignored**, because they do not change the power flow; recorded in file order in `notes["ignored_fields"]` | `gencost`, `bus_name`, `gen_name`, `branch_name`, `areas`, `gentype`, `genfuel` |
| **Refused by name**, because skipping them would silently change the network | `mpc.dcline` (DC lines); isolated buses (type 4, every one named); any other bus type; any other `mpc.<field>`; any statement that is neither the `function mpc = <name>` header nor a whole-field assignment `mpc.<field> = ...` (an indexed assignment such as `mpc.bus(:, 8) = 1;`, `define_constants`, `return`, ...); a field assigned twice |

**Syntax.** The file is read as UTF-8 (a byte-order mark is skipped; a file that is not UTF-8
is refused, naming it). `%` starts a comment except inside a single-quoted string; block
comments `%{ ... %}` are not supported (a line holding only `%{` or `%}` is refused with its
number). `...` continues a line (the rest of the physical line is dropped). Matrices `[ ... ]` may span lines; a row ends at
`;` or at the end of a line; values are separated by whitespace or commas; `Inf` and `-Inf`
are numbers. Cell arrays `{ ... }` are accepted only for ignored fields. A row shorter than its
table needs, a row of a different length from the table's first row, or a token that is not a
number, is refused with the line number. Bus ids must be
integers, and `baseMVA` must be positive (`PowerNetwork` refuses `base_mva <= 0`). The
network's `name` is the function's name (`case9`), or the file's stem if the file has no
`function` header.

## Columns

Columns are named as in MATPOWER's `idx_bus`, `idx_gen` and `idx_brch`; a column not listed is
read past.

**`mpc.bus`**, one `Bus` per row, plus a `Load` and a `FixedShunt` where nonzero:

| Column | MATPOWER name | Unit | Becomes |
|---|---|---|---|
| 1 | `BUS_I` | integer | `Bus.id` |
| 2 | `TYPE` | 1, 2, 3 | `Bus.kind`: 1 `"pq"` (load bus), 2 `"pv"` (voltage-controlled), 3 `"slack"` (reference); 4 (isolated) is refused |
| 3, 4 | `PD`, `QD` | MW, MVAr | `Load(bus, p_mw, q_mvar)`, absorbed, when either is nonzero |
| 5, 6 | `GS`, `BS` | MW, MVAr at 1 p.u. | `FixedShunt(bus, g_mw, b_mvar)`, when either is nonzero: `GS` consumed, `BS` injected |
| 8 | `VM` | p.u. | `Bus.vm` (as stated; not used by the solve) |
| 9 | `VA` | degrees | `Bus.va_deg`; the slack bus's angle setpoint |
| 10 | `BASE_KV` | kV | `Bus.base_kv` (informative) |

**`mpc.gen`**, one `Generator` per row:

| Column | MATPOWER name | Unit | Becomes |
|---|---|---|---|
| 1 | `GEN_BUS` | integer | `Generator.bus` |
| 2 | `PG` | MW | `Generator.p_mw` |
| 3 | `QG` | MVAr | `Generator.q_mvar` (used only on a bus solved as PQ) |
| 4, 5 | `QMAX`, `QMIN` | MVAr | `Generator.q_max_mvar`, `q_min_mvar` (kept; limits are not enforced) |
| 6 | `VG` | p.u. | `Generator.vm_set`, the voltage setpoint of its bus |
| 8 | `GEN_STATUS` | | `Generator.in_service` is `GEN_STATUS > 0` |

**`mpc.branch`**, one `Branch` per row:

| Column | MATPOWER name | Unit | Becomes |
|---|---|---|---|
| 1, 2 | `F_BUS`, `T_BUS` | integer | `Branch.from_bus`, `to_bus` |
| 3, 4, 5 | `BR_R`, `BR_X`, `BR_B` | p.u. on `baseMVA` | `Branch.r`, `x`, `b` (series resistance and reactance, total line charging) |
| 9 | `TAP` | ratio | `Branch.ratio`, off-nominal turns ratio at the from end; 0 means 1 (a line) |
| 10 | `SHIFT` | degrees | `Branch.shift_deg`, phase shift at the from end |
| 11 | `BR_STATUS` | | `Branch.in_service` is `BR_STATUS > 0` |

## How the network is solved

The reader keeps every element, out-of-service ones flagged; the power-flow builders apply
MATPOWER's rules (`makeYbus`, `bustypes`, `makeSbus`, `makeBdc`, `dcpf`):

- **Status.** Generators and branches with status `<= 0` are dropped.
- **Generators per bus.** Generators on one bus are summed. A voltage-controlled bus whose
  in-service generators state different `VG` values is refused (MATPOWER silently takes the
  last).
- **Bus types.** A PV bus with no in-service generator is solved as PQ; the `PG`, `QG` of
  generators on a PQ bus are a fixed injection, netted against the bus's load.
- **Slack.** Exactly one slack bus, with at least one in-service generator. Its voltage
  magnitude is its generators' `VG`, its angle the bus `VA` column. A PV bus's magnitude is
  its generators' `VG`.
- **Per unit.** Powers are divided by `baseMVA`, angles converted to radians; `BR_R`, `BR_X`,
  `BR_B` are already p.u. on `baseMVA`, and `VM`, `VG` in p.u.
- **Start.** A flat start: regulated buses at their setpoint, every other bus at 1 p.u., all
  at the slack angle.

Buses not connected to the slack through in-service branches, and branches of zero impedance,
are refused at build time (see
[Refusals and limitations](../applications/electrical.md#refusals-and-limitations)).

## MATPOWER's standard cases

MATPOWER's code is BSD-licensed, but its licence states that the case files distributed with
it are not covered by that licence, so noodl physics does not redistribute them.
`matpower_case(name, *, cache_dir=None)` returns the path of one of MATPOWER 8.1's `case9`,
`case14`, `case30` and `case118`, downloading it on first use from MATPOWER's GitHub
repository at the commit of the 8.1 release (`MATPOWER_COMMIT`), and checks its SHA-256
(`MATPOWER_CASES`, of the bytes with LF line endings) on every call. The file is only ever
parsed as text by `read_matpower`. It is cached in `$NOODL_CACHE_DIR/matpower/8.1/`, by
default `~/.cache/noodl/matpower/8.1/`, or under `cache_dir` when given:

```py
from noodl.apps.electrical.power import matpower_case, read_matpower

net = read_matpower(matpower_case("case118"))
```

Without a network connection `matpower_case` raises `MatpowerCaseUnavailableError` (an
`OSError`) naming the URL, the path to save the file at by hand and its expected SHA-256; a
downloaded or cached file with another digest raises `MatpowerCaseDigestError` (a
`ValueError`) and a download that fails the check is not cached. Any other name is refused.
`matpower_case("case9_features")` is the variant of `case9` the
[verification](../verification.md#power-flow-against-pandapower) uses, derived from the
fetched `case9` and checked the same way. `tests/data/power_flow/` holds only the stored
reference results (`*.json`, part of noodl physics), each recording the SHA-256 of its case,
and `tests/data/power_flow/NOTICE.md` the cases' sources.

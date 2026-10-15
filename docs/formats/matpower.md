# MATPOWER `.m`

Reading a MATPOWER case file produces a `PowerNetwork` for the
[power flow](../applications/electrical.md#power-flow) of the electrical application.
`noodl.apps.electrical.power.read_matpower(path)` reads the case format of MATPOWER version 2
(`mpc.version = '2'`), the format of every case MATPOWER 8.1 distributes. A case file is a
MATLAB function that fills a struct `mpc`; the reader parses its text and never runs MATLAB or
Octave. The network keeps the file's units (MW, MVAr, degrees, p.u. impedances on `baseMVA`);
the power-flow builders convert to per unit.

```python
from noodl.apps.electrical.power import build_power_flow, read_matpower, solve_power_flow

net = read_matpower("tests/data/power_flow/case9.m")
model, state, drivers = build_power_flow(net)
solved = solve_power_flow(model, state, drivers)

# net.notes["ignored_fields"] lists the fields skipped, here ['gencost']
```

## What is read, ignored and refused

| | Fields and statements |
|---|---|
| **Read** | `mpc.version` (must be `'2'`), `mpc.baseMVA`, `mpc.bus` (at least 13 columns), `mpc.gen` (at least 10 columns), `mpc.branch` (at least 11 columns). Extra columns (solved-case or optimal-power-flow results) are ignored. |
| **Ignored**, because they do not change the power flow; recorded in file order in `notes["ignored_fields"]` | `gencost`, `bus_name`, `gen_name`, `branch_name`, `areas`, `gentype`, `genfuel` |
| **Refused by name**, because skipping them would silently change the network | `mpc.dcline` (DC lines); isolated buses (type 4, every one named); any other bus type; any other `mpc.<field>`; any statement that is neither the `function mpc = <name>` header nor a whole-field assignment `mpc.<field> = ...` (an indexed assignment such as `mpc.bus(:, 8) = 1;`, `define_constants`, `return`, ...); a field assigned twice |

**Syntax.** `%` starts a comment except inside a single-quoted string; `...` continues a line
(the rest of the physical line is dropped). Matrices `[ ... ]` may span lines; a row ends at
`;` or at the end of a line; values are separated by whitespace or commas; `Inf` and `-Inf`
are numbers. Cell arrays `{ ... }` are accepted only for ignored fields. A row shorter than its
table needs, or a token that is not a number, is refused with the line number. Bus ids must be
integers. The network's `name` is the function's name (`case9`), or the file's stem if the
file has no `function` header.

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

## Case files in this repository

`tests/data/power_flow/` holds MATPOWER 8.1's `case9.m`, `case14.m`, `case30.m` and
`case118.m`, copied unchanged, with `case9_features.m`, a variant of `case9` written for noodl
physics, and the stored reference results (`*.json`) the
[verification](../verification.md) compares against. MATPOWER's code is BSD-licensed, but its
licence states that the case files distributed with it are not covered by that licence. The
four MATPOWER files are therefore covered by neither MATPOWER's nor noodl physics' BSD-3
licence: they are redistributed as test data, with their headers naming their sources intact.
`tests/data/power_flow/NOTICE.md` gives their provenance, sources and SHA-256 digests;
`case9_features.m` and the `.json` results are part of noodl physics.

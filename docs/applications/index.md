# Applications

An application is a physical system modelled on the shared core: its constitutive laws, heads
and closures, usually with a reader for the file format of the tool it reproduces. Each is
built with `build_model`, initialised by name and run the same way; see the
[user guide](../guide/workflow.md).

## Choosing an application

| Application | Matches | Reads | Maturity | Start |
|---|---|---|---|---|
| Building airflow and heat | CONTAM (ContamX), Modelica Buildings Library | CONTAM `.prj`, `.wth`, `.ctm`; Modelica JSON + CSV | Verified on a stated subset | [Building](building/index.md) |
| Street air quality | MUNICH, SIRANE | MUNICH and SIRANE case files (`read_case`); NetCDF output | Partial | [Street air quality](street/index.md) |
| Sewers | SWMM 5.2 | SWMM `.inp` | Verified on a stated subset | [Sewers](sewer/index.md) |
| Water distribution | EPANET 2.2 | EPANET `.inp` | Verified on a stated subset | [Water distribution](water/index.md) |
| Water resources | WSIMOD 0.8.1 | WSIMOD `config.yml` and CSV forcing | Verified on a stated subset | [Water resources](water-resources/index.md) |
| Electrical circuits and power flow | Modelica Standard Library, pandapower | Modelica JSON + CSV; MATPOWER `.m` | Verified on a stated subset | [Electrical](electrical/index.md) |

Maturity labels:

- **Verified on a stated subset**: compared with the reference tool on the cases and features
  the application page lists as supported, at a stated tolerance. Anything outside that list is
  refused by name or marked unsupported.
- **Partial**: some comparisons pass, others are open or are only regression guards. The
  application page and the verification page say which.
- **Experimental**: no reference comparison yet.

The evidence behind each label, with tolerances and measured agreement, is under
[Verification](../reference/verification/index.md). Verification means agreement with another
code or a closed form; it is not validation against measurements.

## Related

- Each application uses the flow-determination modes that suit its physics: see
  [the four modes](../concepts/four-modes.md).
- Two models can exchange values at a shared boundary: see [Coupling](../guide/coupling.md).
- Every file reader and what it accepts: [File formats](../reference/formats.md).
- Every type an application offers: the [catalogue](../reference/catalogue/index.md).
- Adding a new application: [Extending noodl](../project/extending/index.md).

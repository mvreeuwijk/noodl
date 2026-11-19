# File formats

noodl physics reads each of these formats into the same kind of object, a network with elements
and layers, so a model read from a file can be batched, differentiated and coupled exactly like
one built by hand in Python. Reading a file is one of two ways in to an application; the other is
building the network directly, shown on each application's pages.

## Support matrix

| Format | Application | Read | Write | Reader | Importing guide |
|---|---|---|---|---|---|
| CONTAM `.prj` | Building physics | Yes | No | `read_prj`, `project_to_model`, `simulate_project` | [CONTAM files](../applications/building/contam.md) |
| CONTAM `.wth` weather | Building physics | Yes | No | `read_wth` | [CONTAM files](../applications/building/contam.md) |
| CONTAM `.ctm` ambient concentrations | Building physics | Yes | No | `read_ctm` | [CONTAM files](../applications/building/contam.md) |
| EnergyPlus `.epw` weather | Building physics | Yes | As `.wth` (`write_wth`) | `read_epw` | [CONTAM files](../applications/building/contam.md) |
| Modelica Buildings Library airflow (JSON + CSV export) | Building physics | Yes | No | `read_modelica` | [Modelica files](../applications/building/modelica.md) |
| Modelica circuits (JSON + CSV export) | Electrical circuits | Yes | No | `read_modelica_circuit` | [Modelica circuits](../applications/electrical/modelica.md) |
| MATPOWER `.m` (case format version 2) | Electrical power flow | Yes | No | `read_matpower` | [MATPOWER files](../applications/electrical/matpower.md) |
| SWMM `.inp` | Sewers | Yes | No | `read_swmm_inp` | [SWMM files](../applications/sewer/swmm.md) |
| EPANET `.inp` | Water distribution | Yes | No | `read_epanet_inp` | [EPANET files](../applications/water/epanet.md) |
| WSIMOD `config.yml` + CSV forcing | Water resources | Yes | No | `read_wsimod_config` | [WSIMOD configuration](../applications/water-resources/wsimod-config.md) |
| MUNICH case directory | Street air quality | Yes | Yes | `read_case`, `write_case` | [Reading and writing cases](../applications/street/cases.md) |
| SIRANE case | Street air quality | Yes | Yes | `read_case`, `write_case` | [Reading and writing cases](../applications/street/cases.md) |

A reader accepts a documented subset of its format; what is and is not supported is stated on the
importing guide. Each reader's signature is in the [API reference](api/index.md).

## Shared parsing

The two `.inp` formats share one lexical layer, `noodl.apps.inpfile`: `[SECTION]` headers, `;`
comments, whitespace-separated fields and blank lines. Neither reader's own section names, column
layouts or units live there; the sewer and water packages each own theirs. The two Modelica
readers share the pipeline in `noodl.modelica` (signal blocks, input resolution, the time grid).
See [Shared file helpers](api/shared.md).

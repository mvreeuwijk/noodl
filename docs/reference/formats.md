# File formats

noodl physics reads each of these formats into the same kind of object — a network with elements and
layers — so a model read from a file can be batched, differentiated and coupled exactly like one
built by hand in Python. Reading a file is one of two ways in to an application; the other is
building the network directly, shown on each application page.

| Format | Reader | What it describes | Produces a model for |
|---|---|---|---|
| [CONTAM](../applications/building/contam.md) `.prj` / `.wth` / `.ctm` | `read_prj`, `project_to_model`, `simulate_project`, `read_wth`, `read_ctm` | Multi-zone building airflow, contaminants and occupants; weather; ambient concentrations | [Building physics](../applications/building/index.md) |
| [Modelica Buildings Library](../applications/building/modelica.md) (JSON + CSV export) | `read_modelica` | Multi-zone airflow models built from MBL components | [Building physics](../applications/building/index.md) |
| [SWMM](../applications/sewer/swmm.md) `.inp` | `read_swmm_inp` | Gravity sewer and drainage networks | [Sewers](../applications/sewer/index.md) |
| [EPANET](../applications/water/epanet.md) `.inp` | `read_epanet_inp` | Pressurised water distribution | [Water distribution](../applications/water/index.md) |
| [Modelica circuits](../applications/building/modelica.md#electrical-circuits) (JSON + CSV export) | `read_modelica_circuit` | Lumped circuits from the Modelica Standard Library and Buildings Library | [Electrical circuits](../applications/electrical/index.md) |
| [MATPOWER](../applications/electrical/matpower.md) `.m` (case format version 2) | `read_matpower` | Balanced power networks: buses, lines, transformers, loads, shunts, generators | [Electrical power flow](../applications/electrical/index.md#power-flow) |
| [WSIMOD](../applications/water-resources/wsimod-config.md) `config.yml` + CSV forcing | `read_wsimod_config` | Integrated catchment, sewer, treatment and river models, as WSIMOD describes them | [WSIMOD](../applications/water-resources/wsimod.md) |

The two `.inp` formats share one lexical layer, `noodl.apps.inpfile`: `[SECTION]` headers, `;`
comments, whitespace-separated fields and blank lines. Neither reader's own section names, column
layouts or units live there — `apps/sewer/inp.py` and `apps/water/inp.py` each own theirs.

# File formats

noodl reads each of these formats into the same kind of object — a network with elements and
layers — so a model read from a file can be batched, differentiated and coupled exactly like one
built by hand in Python. Reading a file is one of two ways in to an application; the other is
building the network directly, shown on each application page.

| Format | Reader | What it describes | Produces a model for |
|---|---|---|---|
| [CONTAM](contam.md) `.prj` / `.wth` | `read_prj`, `project_to_model`, `read_wth` | Multi-zone building airflow and contaminants; weather | [Building physics](../applications/building_physics.md) |
| [Modelica Buildings Library](modelica.md) (JSON + CSV export) | `read_modelica` | Multi-zone airflow models built from MBL components | [Building physics](../applications/building_physics.md) |
| [SWMM](swmm.md) `.inp` | `read_swmm_inp` | Gravity sewer and drainage networks | [Sewers](../applications/sewer.md) |
| [EPANET](epanet.md) `.inp` | `read_epanet_inp` | Pressurised water distribution | [Water distribution](../applications/water.md) |

The two `.inp` formats share one lexical layer, `noodl.apps.inpfile`: `[SECTION]` headers, `;`
comments, whitespace-separated fields and blank lines. Neither reader's own section names, column
layouts or units live there — `apps/sewer/inp.py` and `apps/water/inp.py` each own theirs.

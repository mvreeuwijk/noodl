# Water resources

Two things in noodl model water resources: a **flow allocation layer** (rule-based allocation of
requested flows between stores, clipped by arc capacity and free storage) and a **differentiable
port of WSIMOD 0.8.1**, the Water Systems Integrated Modelling framework of Imperial College
London (land, groundwater, rivers, reservoirs, sewers, treatment works and demands, stepped one
day at a time). Both run batched on PyTorch tensors and give exact gradients of any output with
respect to the parameters, forcings and initial stores listed on the [gradients page](wsimod-gradients.md).

| Part | Maturity | Why |
| --- | --- | --- |
| [Flow allocation](allocation.md), `"hard"` mode | Verified on a stated subset | Equals WSIMOD's realised flows on `quickstart_demo` (1e-16) and on 20 of the 21 arcs of `oxford_demo` (1.9e-9; the `sewer_to_wwtw` arc is excluded), and on scripted networks with binding capacities and headroom (4.4e-16); differences from WSIMOD's call-order sharing are stated and tested |
| Flow allocation, `"smooth"` and `"projection"` modes | Experimental | Checked against the hard clip, by gradient checks and by a hand-derived cross-gradient; no external reference |
| [WSIMOD port](wsimod.md) | Verified on a stated subset | Bitwise equal to WSIMOD (worst difference 0) on both demonstration models and on 61 purpose-built cases, in two modes; 81 gradients agree with central differences to 5.3e-7. Subset: every class except `EnfieldFoulSewer`; configurations read from `config.yml` and CSV only |
| [Importing `config.yml`](wsimod-config.md) | Verified on a stated subset | Reads WSIMOD's own saved format; anything it cannot run is refused by name |

The evidence is on the [verification page](../../reference/verification/water-resources.md).

## Supported and not supported

| Supported | Not supported (see [Limitations](limitations.md)) |
| --- | --- |
| Reading WSIMOD's `config.yml` with CSV or CSV.GZ forcing, or the same structure as a Python mapping | Writing a configuration; `extensions` and the parquet `unified_data_file` (refused) |
| All WSIMOD 0.8.1 node and arc classes except `EnfieldFoulSewer`, with the twelve default pollutants | `EnfieldFoulSewer` (undefined in WSIMOD itself); surface `*-residue` data |
| Quantity-only runs and runs with pollutants; decay, treatment, river chemistry, crops, irrigation and gardens | Models that make WSIMOD's checks recurse into each other (refused by name) |
| Batches in which each instance takes its own branch; gradients of every numeric input | Gradients across a branch change (they are those of the branch taken); checkpointing of long gradient runs |
| Allocation by capacity and receiver headroom, with proportional sharing by preference | WSIMOD's call-order sharing, pass-through bottlenecks, queue arcs and pollutants in the allocation layer |
| float64 on the CPU | Other dtypes and devices |

## A short example

Two arcs in series, the first limited to 1.5 m³/s although 5 m³/s is requested:

```python
import torch
from noodl.layers.allocation import AllocatedFlowLayer
from noodl.topology import Network

net = Network()
for name in ("reservoir", "works", "demand"):
    net.add_node(name)
net.add_edge("reservoir", "works", kind="link")
net.add_edge("works", "demand", kind="link")

layer = AllocatedFlowLayer(
    net, "alloc", "link",
    s_max=torch.full((net.n,), 100.0, dtype=torch.float64),
    c_arc=torch.tensor([1.5, 10.0], dtype=torch.float64),
)
requests = torch.tensor([5.0, 1.0], dtype=torch.float64)
storage, flow = layer.step(torch.zeros(net.n, dtype=torch.float64), {"alloc.requests": requests}, dt=1.0)
print(flow.tolist())
print(storage.tolist())
```

```text
[1.5, 1.0]
[-1.5, 0.5, 1.0]
```

To run a WSIMOD model, start from [Running a WSIMOD model](wsimod.md).

## Pages

- [Flow allocation](allocation.md): use the layer, its modes, how it differs from WSIMOD.
- [Flow allocation: the physics](allocation-physics.md): the clip, proportional sharing and the
  three differentiability modes.
- [Running a WSIMOD model](wsimod.md): load, run, results, diagnostics, mass balance, state and
  drivers.
- [WSIMOD: gradients, batches and long runs](wsimod-gradients.md).
- [WSIMOD: what is ported](wsimod-ported.md): classes and processes.
- [Differences from WSIMOD](differences.md).
- [Importing WSIMOD files](wsimod-config.md): the `config.yml` format.
- [Limitations](limitations.md): what is not supported and where results can differ.
- Reference: [WSIMOD catalogue](../../reference/catalogue/wsimod.md),
  [verification](../../reference/verification/water-resources.md),
  [API](../../reference/api/index.md).

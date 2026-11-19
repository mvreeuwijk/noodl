# User guide

How to do things with noodl, one task per page. The pages are cross-application: the same calls
work for a building, a street, a sewer, a water network, WSIMOD and a circuit. For the physics of
one application, see [Applications](../applications/index.md); to learn by following an example,
start with [Get started](../get-started/index.md).

| Page | Purpose |
|---|---|
| [The workflow](workflow.md) | Configure, initialise by name, check and run a model; the lead page, with a complete example. |
| [Time stepping and schedules](time-stepping.md) | Step, solve the steady state, march over a time grid, change drivers in time. |
| [Reading results by name](results.md) | Read any state or flow by label instead of by position. |
| [Batches and ensembles](batches.md) | Run many cases in one call by giving tensors instead of numbers. |
| [Gradients and optimisation](gradients.md) | Differentiate a result and calibrate with `torch.optim`. |
| [Assembling from components](components.md) | Build a large network from rooms, doors and other pieces with ports. |
| [Coupling two models](coupling.md) | Join two models at a shared boundary and iterate them to a fixed point. |
| [Devices, dtype, copying and saving](devices.md) | Move a model to a GPU or another precision; what survives copying and saving. |
| [Choosing solvers and coupling schemes](solvers.md) | Pick a linear solver, rescue a Newton failure, choose ping-pong or iterate. |
| [Troubleshooting](troubleshooting.md) | Non-convergence, the ping-pong default, float32, and what each `check()` message means. |

Keys and the order of each tensor are tabulated in the [keys reference](../reference/keys.md).

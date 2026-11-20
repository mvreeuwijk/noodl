# WSIMOD: gradients, batches and long runs

How to differentiate a WSIMOD run, what a gradient means where the model branches, how batches
behave, and how to keep the autograd graph manageable. The setup is the one of
[Running a WSIMOD model](wsimod.md).

## A gradient

Every driver and every initial store can carry `requires_grad`. The river's storage after three
days of the `small` model, with respect to its velocity:

```python
import torch

from noodl.apps.wsimod import build_model, read_wsimod_config

network = read_wsimod_config(".")  # config.yml, and the forcing file it names
velocity = torch.tensor([17280.0], dtype=torch.float64, requires_grad=True)
model, state, drivers = build_model(network)
drivers = {**drivers, "wsimod.River.velocity": velocity}
for k in range(3):
    state = model.step(state, drivers, dt=86400.0, t=k * 86400.0)
state["wsimod.storage"][0, 0].backward()
print(f"{velocity.grad.item():.3e}")
```

```text
-6.380e-06
```

WSIMOD's branches (`if x > FLOAT_ACCURACY`, `min`, `max`, the distribute loops) are decided on
detached values per instance, so the gradient is that of the branch each instance took; it jumps
where a branch changes.

## What can be learned

Every numeric parameter (`"wsimod.<Class>.<parameter>"`), the arcs' capacities and preferences,
the forcing series and the initial stores may require grad. Central-difference checks cover 81
of them:

- **Hydrology and networks:** arc capacity and preference, groundwater and land residence times,
  impervious and pervious surface coefficients, `ihacres_p`, river `mrf` and `velocity`, a
  sewer's time-area fractions, `Distribution.leakage`, the `WWTW` and `FWTW` throughput
  capacities, `ResidentialDemand.per_capita`, a demand's `pollutant_load`, a catchment's flow
  and concentration, precipitation and an initial storage.
- **Decay, treatment and river chemistry:** the decay `constant` and `exponent` of a `Storage`, a
  `RiverReservoir`, a `QueueGroundwater` and a `DecayArc`; the temperature forcing through a
  decay and through the treatment; the `WWTW`'s `process_parameters` and `liquor_multiplier`; the
  three soil-temperature weights; and, through the River's biochemistry, `denpar_w`,
  `halfsatINwater`, `prodNpar`, `muptNpar`, `T_wdays` and the air temperature.
- **Land and crops:** the growing surface's `ET_depletion_factor`, `rooting_depth`, crop factors
  and covers, `uptake1`, `uptake3`, `denpar`, `hsatINs`, `exppar`, `satact`, `thetapow`, `kfr`,
  `nfr`, `kadsdes`, `erodibility`, `sreroexp` and `srfilt`; the nutrient pool's `minfpar`,
  `immobdpar` and `fraction_manure_to_dissolved_inorganic`; the monthly deposition, fertiliser
  and manure data and the surfaces' `pollutant_load`; `IrrigationSurface.irrigation_coefficient`,
  `VariableAreaSurface.current_surface_area` and `ResidentialDemand.gardening_efficiency`.

The values agree with central differences at a relative step of 1e-6 to within 5.3e-7 relative
(the tolerance is 1e-6). The [evidence page](../../reference/verification/water-resources.md)
has the figures.

## What cannot be learned

- **Structure.** The topology, the classes, the pollutant lists, the orchestration, the travel
  times (whole days: they size the queues) and whether a `Distribution` has leakage at all
  (WSIMOD decorates its handlers only then) are fixed when the model is built.
- **Python's power.** The power is evaluated per instance in Python, as WSIMOD does, so values
  are bitwise WSIMOD's. Its slope in the base is `y x^y / x` wherever the base is not 0, exact
  for a negative base with an integral power. Its slope in the power is 0 for a negative base,
  and both slopes are 0 at a base of 0 (a saturated soil).
- **Jumps and kinks.** The gradient is not defined where a branch changes: at a volume that
  crosses WSIMOD's tolerance, at a capacity that just clips a push, at the day a tank starts to
  spill, at a distribute loop that ends one round earlier. A decision's margin (`lhs - rhs`, in
  `diagnostics["decisions"]`) says how near a run is to such a point. A residence time near one
  day leaves a recession tail of about 1e-9 that a central difference sees as a kink.
- **Specific kinks** in the land and river processes: the soil temperature factor jumps at 5 °C;
  rain crossing 5 mm switches erosion on; the River's 50 % caps, its `T < 5` factor and its
  production and mineralisation switch are kinks and its dry guard is a jump; the adsorption
  gradient is that of the unrolled Newton passes, not the implicit one at the equilibrium, and
  its convergence limit has no gradient. See [What is ported](wsimod-ported.md).

## Batches

Any driver or state key may carry leading batch dimensions. Every instance follows its own
WSIMOD path, so one instance may clip an arc where another does not, run its distribute loop to
its cap where another stops, or pond a sewer, and each equals its own single run bit for bit.
What cannot differ between instances is structure: the sequence of calls (fixed by a tracing day
at build time and checked on every later day), the travel times, the pollutant lists and
whether a `Distribution` has leakage.

Where one instance takes a process out of its domain (a division by zero in a growing surface,
say), Python has no per-instance error, so noodl raises a `ValueError` naming the surface, the function, the
day and the cause, and the whole batch stops.

## Graph size and fitting in windows

Autograd is on only when a state or driver tensor requires grad; otherwise the run is under
`torch.no_grad()`. The graph is large because a day is thousands of small operations: about
4,000 autograd nodes a day for `quickstart` and 15,600 for `oxford`, independent of the batch,
which only widens the tensors. Gradients through the full 1,456 days of `oxford` would build
about 23 million nodes. Fit gradients in windows instead: run some days, take the gradient,
`detach` the state and carry on. noodl does not checkpoint.

A batch of 100 costs little more than a batch of 1, because the batch widens
tensors rather than adding operations. The per-instance Python arithmetic that keeps the result
equal to WSIMOD's (power, exponential and square root in a loop over the batch) costs time
proportional to the batch; on these models it is small beside the many small tensor operations
of a day, and larger where many growing surfaces or rivers are present. A pollutant run costs a little more than a quantity-only run.

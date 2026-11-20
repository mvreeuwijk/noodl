# Differences from WSIMOD

noodl runs WSIMOD 0.8.1's day with the same calls, order and arithmetic, so on the demonstration
models and the tested cases its results are bitwise WSIMOD's. This page lists the situations in
which a noodl run can differ from a WSIMOD run of the same configuration. Each is one of three
kinds:

- **Runs with an event:** in an edge case, such as an empty tank or a zero denominator, noodl
  keeps a finite value, keeps the mass balance closed and records a named event (see
  [the diagnostics](wsimod.md#diagnostics)).
- **Refused:** a configuration noodl cannot run faithfully is rejected by name when the model is
  built.
- **Interpreted:** where a setting admits more than one reading, noodl takes the one stated here.

The [flow allocation layer](allocation.md#where-the-layer-differs-from-wsimod) has its own list
of differences.

## Structure and bookkeeping

| Situation | What noodl does |
|---|---|
| System mass balance | A per-day balance of inflow, outflow and storage change, closing to rounding (see [the mass balance](wsimod.md#the-mass-balance)) |
| Ponded sewer water on a `Land` | Forced into the first impervious surface |
| `EnfieldFoulSewer` | Refused by name |
| `NonResidentialDemand` | Takes its constant demand, as a house does |
| Rivers tied in the routing order | Configuration order (results can differ in the last bit where ties matter) |
| Overrides | Applied per node and per surface |
| Overrides on a `Distribution` | Its handlers are wrapped once |
| `UnlimitedDistribution` with leakage | Refused |
| River arcs without a `Waste`; an arc into a `Catchment` | Refused |
| A zero priority among non-zero ones in a distribute round | Nothing is distributed in that round; event `"zero_priority"` |
| A `QueueArc` push below its tolerance | Dropped, as in WSIMOD, and recorded as dropped water |
| A zero-volume `QueueArc` pull with additive keys | The rescale is skipped, as `Arc` does |
| A `WWTW` discharge rejected at or below the tolerance | Recorded as dropped water |

## Treatment, decay and river chemistry

| Situation | What noodl does |
|---|---|
| Treatment solids below zero (`constant * factor + liquor_multiplier > 1`) | The same values; event `"wtw_negative_solids"` |
| A treatment works with no input | The previous liquor value is kept; event `"wtw_liquor_zero_denominator"` |
| A decay's or treatment's `exponent` of 0 or below | Refused by name |
| `decays` naming `volume` | Refused by name |
| `decays: {}` on the storage family | The temperature forcing is required |
| A decaying `Groundwater` or `QueueGroundwater` | The temperature comes from a `filename` override; without one it is refused by name |
| A River evaluated twice in one day | Both evaluations count in the balance, which closes |
| River mineralisation with no dissolved nitrogen | The nitrogen leaves through the balance's outflow; event `"river_nitrogen_lost"` |

## Land and crops

| Situation | What noodl does |
|---|---|
| A date past the last crop stage | Refused with its name |
| A configured `et0_coefficient` on a growing surface | Listed in `model.unread` (the crop's factor is used) |
| The water-stress coefficient at the wilting point with the default depletion factor | `Ks = 0`, the limit and FAO-56's value; event `"crop_water_stress_zero_denominator"` |
| `wilting_point >= field_capacity` | Refused |
| `total_porosity` not above `field_capacity` or `wilting_point` | Refused |
| A growing surface drawing on a tank with no inorganic nitrogen | The three species get 0; event `"pull_storage_no_din"` |
| A transformation in a tank with no `nitrate` or `ammonia` | The change goes to `ammonia`; event `"transformation_no_din"` |
| Crop uptake from a dry soil | No uptake; event `"crop_uptake_dry_soil"` |
| The nutrient pool and the tank drifting apart where the tank's clamp applies | Event `"nutrient_pool_not_tracking"` |
| A Freundlich adsorption step that adjusts the flow | Event `"freundlich_flow_adjusted"` |
| A float `initial_soil_storage` | Refused |
| Nutrient pool parameters | Drivers with WSIMOD's defaults; refused as configuration |
| An `IrrigationSurface` without `nitrate` among the pollutants | Runs; the nutrient pool is updated only when `nitrate` is a pollutant |
| Two or more `GardenSurface`s on a `Land` | The last one is irrigated, as in WSIMOD; the others are listed in `model.unread` |
| Erosion | Requires `solids` among the pollutants |
| A `pollutant_load` override on a surface | Applied to that surface only; a load the surface does not use is listed in `model.unread` |
| A `data_input_dict` override adding deposition data | Listed as unread; removing `nhx-dry` by override is refused |
| `depth` on a growing surface | Refused; `rooting_depth` is the override |
| Dry deposition without wet deposition data | Refused by name |
| Data named `*-residue` | Refused |

## Rounding

On a land with a growing surface and `nitrate` among the pollutants, WSIMOD's node check can
print a mass-balance line for `org-phosphorus` of the order of 1e-17 kg, when a tank holds a large
organic store and the day's flows are small. The check subtracts two large stores, and whether a
line appears depends on the last bit of the platform's maths library. Values are unaffected, and
noodl's balance closes.

# Differences from WSIMOD

noodl runs WSIMOD 0.8.1's day with the same calls, order and arithmetic, so on every model it
accepts it agrees with WSIMOD exactly. This page lists the places where WSIMOD's source is
defective or silent and says what noodl does instead. There are four outcomes:

- **Reproduced:** noodl does what WSIMOD does, defect included, so results stay comparable.
- **Fixed:** noodl does the evident intent.
- **Recorded:** WSIMOD divides by zero, raises or loses water; noodl runs on with a finite value
  and records an event (see [the diagnostics](wsimod.md#diagnostics)).
- **Refused:** the model is rejected by name when it is built.

Defects in the growing surface, the nutrient pool, irrigation and gardens were reported to the
WSIMOD maintainers
([ImperialCollegeLondon/wsi#260](https://github.com/ImperialCollegeLondon/wsi/issues/260); earlier
ones are in #258 and #259). The [flow allocation layer](allocation.md#where-the-layer-differs-from-wsimod)
has its own list of differences.

## Structure and bookkeeping

| WSIMOD | noodl |
|---|---|
| The system mass balance compares the inflow with itself, so it never reports | Fixed: a real balance, per day, closing to rounding |
| `Land.push_set_sewer` lacks `self.surface`, and `Land` has no `"Sewer"` check | Fixed: ponded sewer water is forced into the first impervious surface |
| `EnfieldFoulSewer.make_discharge` refers to an undefined variable | Refused by name |
| `NonResidentialDemand.get_demand` recurses without end | Fixed: its constant demand, as a house |
| A river's tie order depends on Python's set hashing | Configuration order is used (can differ in the last bit where ties matter) |
| Overrides update shared default dicts, so one node's override reaches others | Per node (and per surface) |
| `Distribution.apply_overrides` decorates the handlers again | Decorated once |
| `UnlimitedDistribution` loses its leakage | Refused |
| River arcs without a `Waste`; an arc into a `Catchment` | Refused |
| A partial zero priority in a distribute round divides by zero | Recorded: nothing is distributed in that round (`"zero_priority"`) |
| `QueueArc` drops a push below its tolerance | Reproduced (real models' delivery depends on it) and recorded as dropped water |
| `QueueArc` zero-volume pull with additive keys | The rescale is skipped, as `Arc` does |
| `WWTW.make_discharge` rejection at or below the tolerance | A tolerance; recorded as dropped water |

## Treatment, decay and river chemistry

| WSIMOD | noodl |
|---|---|
| Treatment solids are the unclamped remainder and go negative where `constant * factor + liquor_multiplier > 1` | Reproduced; event `"wtw_negative_solids"` |
| The treatment liquor blend divides by zero with no input | The previous value is kept; event `"wtw_liquor_zero_denominator"` |
| No check of a decay's or treatment's `exponent`: `0 ** negative` raises or gives a complex number | An `exponent` of 0 or below is refused by name |
| `decays` may name `volume`, so the water itself would decay | Refused by name |
| `decays: {}` on the storage family builds a decaying tank that reads the temperature and decays nothing | Reproduced; the temperature forcing is required |
| WSIMOD's `record_all` skips a decaying tank | Reproduced: the tank drops out of the `tank_*` results; its storage stays in the state |
| A decaying `Groundwater` or `QueueGroundwater` loses its configured forcing and raises `TypeError` at its first end of day | Its forcing is listed as unread; the temperature comes from a `filename` override, and a decaying one without it is refused by name |
| A decaying queue tank defaults to `number_of_timesteps=1`, against its docstring and the plain queue's 0 | Reproduced: `decays` delays a `QueueGroundwater` by one day; the `number_of_timesteps` override sets it |
| `Surface.run` converts the day's decayed mass at every call | Reproduced: a second `Land.run` on one day converts the same mass twice |
| `River.calculate_discharge` assigns its mass-balance terms, so a second call a day drops the first call's terms and WSIMOD prints an error every day | The processes run twice as in WSIMOD; noodl's terms add both calls and its balance closes |
| River mineralisation with no dissolved nitrogen destroys the mineralised nitrogen | Reproduced; the loss is in the balance's outflow and event `"river_nitrogen_lost"` |
| The River's "365-day" phosphorus mean adds 366 entries; every window divides by its full length from day 1 | Reproduced |
| `totalphosfcn` compares the River's phosphorus mass (kg) with concentrations (kg/m³) | Reproduced: the evident fix has no single form |

## Land and crops

| WSIMOD | noodl |
|---|---|
| `GrowingSurface` tests `t.is_leap_year`, which is a method in the loader noodl reads and always true, so the day of year shifts after 28 February in every year | Reproduced: 1 March interpolates at day 59 and 31 December at 364 in every year. A parquet-loaded WSIMOD, where it is a property, shifts only in leap years |
| The sowing day has no crop (`days_after_sow` is 0 on it) | Reproduced: covers start to change the day after; they persist outside the season until the harvest resets them |
| An autumn-sown crop (`sowing_day > harvest_day`, or the default `harvest_day = 365`) has no cover all year | Reproduced |
| The crop interpolation wraps to the last stage for a day at or before the first, and raises past the last | The wrap reproduced; a date that would raise is refused with its name |
| The configured `et0_coefficient` is overwritten by the crop's | Listed in `model.unread` |
| The stress coefficient divides by `(1 - ET_depletion_factor) * TAW`, which is 0 for the default factor at the wilting point | `Ks = 0` (the limit and FAO-56's value); event `"crop_water_stress_zero_denominator"`. WSIMOD raises |
| `TAW < 0` only prints; `TAW = 0` divides by zero | `wilting_point >= field_capacity` refused |
| `pull_storage` divides by the tank's inorganic nitrogen | The three species get 0; event `"pull_storage_no_din"`. WSIMOD raises when a growing surface starts from a tank with no inorganic N and has no input in its first month |
| The transformation divides by the tank's `nitrate + ammonia` | The change goes to `ammonia`; event `"transformation_no_din"` |
| Crop uptake divides by the tank's volume | No uptake; event `"crop_uptake_dry_soil"` |
| Flushing extracts the outflows' nutrients from pools that `pull_storage` already took them from | Reproduced: the pools lose the outflows' nutrients twice, so the tank and the pool drift apart |
| Uptake and denitrification take all their nitrogen as nitrate | Reproduced; where the tank's clamp bites, event `"nutrient_pool_not_tracking"` (WSIMOD's own check compares a value with itself and never prints) |
| The Freundlich "flow adjusted" warning can never fire | The evident test is used; event `"freundlich_flow_adjusted"` |
| The Newton loop's convergence test reads the previous pass's residual | Reproduced |
| The soil temperature factor is `T / 5` below 5 °C, not HYPE's product, so it jumps at 5 °C | Reproduced |
| A float `initial_soil_storage`, which the docstring allows, raises `TypeError` | Refused |
| The nutrient pool's parameters cannot be configured | Drivers with WSIMOD's defaults; refused as configuration |
| `IrrigationSurface.irrigate` reads five pollutant keys whatever the pollutants | The pool is updated only when `nitrate` is a pollutant. WSIMOD raises `KeyError` on the first day with a demand |
| Two or more `GardenSurface`s on a `Land`: the last one's handlers replace the others' | Reproduced; the earlier ones are listed in `model.unread` |
| The garden water is an empty request with a volume, so it is at 0 °C | Reproduced: it cools the soil it enters |
| Erosion reads `solids`, which the docstring's pollutant list omits | Refused without `solids` |
| A `pollutant_load` override on a surface built without one is ignored and mutates a shared default | Per surface; the load is listed in `model.unread` |
| A `data_input_dict` override does not rebuild the deposition inflows | Deposition data added by override are unread; removing `nhx-dry` by override is refused |
| `depth` raises `TypeError` in the configuration and prints an error as an override | Refused in both; `rooting_depth` is the override |
| Dry deposition without wet data raises `KeyError` | Refused by name |
| The harvest branch stores a crop factor that the next statement overwrites | Not ported (no observable effect) |
| `GrowingSurface.residue` is unwired | Not ported; data named `*-residue` is refused |
| `total_porosity` not above `field_capacity` or `wilting_point` | Refused |

## Rounding-level differences

On a land with a growing surface and `nitrate` among the pollutants, WSIMOD's node check can
print round-off lines of the form "mass balance error for org-phosphorus of x in <land>" with
`|x|` near 1e-17 kg, when a tank holds a large organic store and the day's flows are small. The
check subtracts two large stores; noodl's own balance closes. Whether a line appears depends on
the last bit of the platform's maths library. Values are unaffected.

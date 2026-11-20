# WSIMOD: what is ported

Reference for the node and arc classes of WSIMOD 0.8.1 that noodl runs, the processes inside
them, and the drivers and state each process adds. For how to run a model, see
[Running a WSIMOD model](wsimod.md); for the places where noodl differs from WSIMOD, see
[Differences from WSIMOD](differences.md).

**Nodes:** `Node`, `Waste`, `Catchment`, `Storage`, `Groundwater`, `QueueGroundwater`, `River`
(with its nitrate biochemistry), `Reservoir`, `RiverReservoir`, `Demand`,
`NonResidentialDemand`, `ResidentialDemand`, `Distribution` (with leakage),
`UnlimitedDistribution`, `Sewer`, `WTW`, `WWTW`, `FWTW` and `Land` with `Surface`,
`ImperviousSurface`, `PerviousSurface` (IHACRES), `GrowingSurface`, `IrrigationSurface`,
`GardenSurface` and `VariableAreaSurface`.

**Arcs:** `Arc`, `QueueArc`, `AltQueueArc`, `PullArc`, `PushArc`, `SewerArc`, `WeirArc`, and the
user decay arcs `DecayArc` and `DecayArcAlt`.

**Not ported:** `EnfieldFoulSewer`, whose `make_discharge` refers to an undefined variable in
WSIMOD itself, and the surface `residue` handling, which WSIMOD never wires up.

A day is WSIMOD's: the orchestration slot by slot, each slot's nodes in configuration order;
`River.distribute` in the river order, most upstream first; the record; then `end_timestep` on
every node and every arc.

| Class | Process |
|---|---|
| `Storage`, `Reservoir`, `RiverReservoir`, `Groundwater` | first-order decay of the tank at the end of the day, when `decays` is given |
| `QueueGroundwater` | the same, in a decaying queue tank whose internal queue is a `DecayArcAlt` |
| `River` | the decaying tank, and the biochemistry: temperature relaxation, denitrification, nitrogen and phosphorus production and mineralisation, macrophyte uptake |
| `WTW`, `WWTW`, `FWTW` | per-pollutant treatment (`process_parameters`) and liquor multipliers |
| `Land` surfaces | decay at the land's air temperature, and the nitrite carry-over |
| `PerviousSurface` | the soil temperature |
| `GrowingSurface` and subclasses | crop calendar, nutrient pool, soil processes |
| `DecayArc`, `DecayArcAlt` | decay of the requests in transit, at the in-port's temperature |

## Decay

### The storage family

A `Storage`, `Reservoir`, `RiverReservoir` or `River` built with a `decays` dict (an empty one
included) decays its tank at the end of every day, after the orchestration and the record. For
every pollutant `p` in `decays`:

```
diff[p]  = storage[p] * min(constant[p] * exponent[p] ** (T - 20), 1)
storage[p] -= diff[p]
```

with `T` the node's own `temperature` forcing. The pollutant may be non-additive (`do` decays
like any other); the volume and every pollutant not in `decays` are untouched. The decayed mass
is the mass-balance outflow `"DecayTank.decayed"`, so the balance closes.

Where `T` comes from:

| Node | Temperature |
|---|---|
| `Storage`, `Reservoir`, `RiverReservoir`, `River` | the node's own forcing: the constructor's `data_input_dict` or `filename` (relative to the configuration's directory), or a `filename` override (relative to the working directory). A missing date is refused by name when the model is built, whatever `decays` lists |
| `Groundwater`, `QueueGroundwater` | a `filename` **override** only: the constructor's forcing is not used for these classes, so a configured forcing is listed in `model.unread` and a `decays` dict without the override's `temperature` on every date is refused by name. Use an absolute path |
| a node with `decays: None` | none |

The cap `min(factor, 1)` is a decision (`node:<name>/tank/make_decay/cap:<pollutant>`), true
where the factor exceeds 1; its margin is `factor - 1`. The factor is non-differentiable at the
cap, where its gradient jumps from the exponential's to 0.

Drivers: `"wsimod.<Class>.decays.constant"` and `".exponent"`, `(..., n, K)`, for every node of
the class, nodes built with `decays: None` included (all-inert rows). The configured key set is
structure: an entry outside it is inert (`constant` 0, `exponent` 1) and any other value is
refused by name. Both drivers are absent in a quantity-only run, which decays nothing but still
reads the temperature. An `exponent` of 0 or below is refused (WSIMOD raises or returns a
complex number), and so is a `constant` or `exponent` that is not a number.

A `decays` override merges per pollutant: an entry replaces that pollutant's whole dict, so
`{phosphate: {constant: 0.3}}` is refused for the missing `exponent`. An override on a node
built with `decays: None` is refused.

### A decaying `QueueGroundwater`

A `QueueGroundwater` with `decays` decays water in its internal queue on entry and at every end
of day. WSIMOD's class differs from a non-decaying one in four ways, all reproduced:

- **One day longer.** `number_of_timesteps` defaults to 1, so adding `decays` delays all of the
  node's water by one day. The override `number_of_timesteps` sets it (for example to 0); it is
  refused on a node without `decays`.
- **The decay reaches `storage` a day late.** A push adds the undecayed water to `storage`, and
  `storage` loses the queue's decay at the end of the day. The decay of day d's end waits in the
  carried state row `(node, "tank.internal_arc")` of `"wsimod.total_decayed"` until the end of
  day d + 1. Only the pollutant columns lag, and no pull reads them. The balance counts the tank
  by its `storage`, so it closes every day.
- **`active_storage` never decays.** Water decays only while it is in the queue.
- **No flush at the end of the day.** Water that the end of day d moves to slot 0 becomes active
  only at the next push or at `distribute`, and decays again at each end of day it waits.

A quantity-only run keeps the decaying queue's structure (its delay, its end of day and its
`total_decayed` row, all zero) with no `decays.*` driver. Its tank is not in the `tank_*`
results, nor is any `decays` tank's.

### The user decay arcs

A `DecayArc` is a `QueueArc` whose requests decay; a `DecayArcAlt` is an `AltQueueArc` with the
decaying queue's entry and end of day. Both decay by the storage family's rule with their own
`decays`, at the temperature of their **in-port**, for pushes and pulls alike. Drivers are
`"wsimod.DecayArc.decays.constant"`, `".exponent"` and the same for `DecayArcAlt`,
`(..., n_arcs_of_class, K)`, one row per arc in configuration order, inert where `decays` lists
no pollutant. The cap decision is `arc:<name>/make_decay/cap:<pollutant>`.

- **Entry.** The request passes `enter_arc` undecayed, so the arc's inflow counts it as sent,
  then decays, then waits. A pulled request decays after it left the sender, which loses the
  undecayed water. A push's backflow is the decayed water, returned to the sender. With
  `number_of_timesteps: 0` a request decays once and is delivered at once.
- **End of the day.** Every waiting request decays in place, in both directions.
- **Mass balance.** Each decayed mass is the outflow `"DecayArc.decayed"` or
  `"DecayArcAlt.decayed"`. Where a `DecayArc`'s end of day finds due-and-waiting requests in both
  directions, the order of addition of that term may differ from WSIMOD's in the last bit, and
  the event `"decay_arc_order_unknown"` is recorded at the arc; the delivered water and the
  queued requests are WSIMOD's in every case.
- **A catchment in-port.** A catchment that a decay arc leaves has its own
  `"wsimod.temperature"` row, which the arc's decay reads; the outflow reads its
  `"wsimod.concentration"` columns. Where `temperature` is a pollutant the same value is in
  both drivers, and changing one does not change the other.

### Decay on a land's surfaces

A `Surface` with a `decays` dict decays its storage at the end of every day by the same rule,
at the `Land`'s **air** temperature (not the soil temperature a `PerviousSurface` works out). A
surface's `decays` default to `{}`, which decays nothing. Drivers:
`"wsimod.<ImperviousSurface|PerviousSurface>.decays.constant"` and `".exponent"`, `(..., n, K)`,
labelled `(node, surface)`, inert for a pollutant the surface does not list.

**The nitrite carry-over.** With `"nitrite"` among the pollutants, `Surface.run` starts with
yesterday's decayed mass: decayed ammonia is added to the surface's nitrite and decayed nitrite
to its nitrate, and both count as inflows of the land. Decayed nitrate, phosphate or anything
else is lost (the outflow `"DecayTank.decayed"`). Without `"nitrite"` all of the decayed mass is
lost. The block reads `nitrate`, `nitrite` and `ammonia`: a `Land` with `nitrite` but not the
other two is refused by name. Day 0 converts nothing, and `"wsimod.total_decayed"` carries the
mass. A custom orchestration that runs a `Land` twice a day converts the same mass twice, as
WSIMOD does.

WSIMOD's default `decays={}` is one dict shared by every surface built without a `decays` key,
so a `decays` override of one of them reaches all of them. noodl merges per surface: the
override reaches only the surface it names.

## Treatment works

`WTW.treat_current_input` (also `WWTW` and `FWTW`) splits the day's input into treated water,
liquor and solids. Every additive pollutant `k` has its own
`process_parameters[k] = {constant, exponent}` and `liquor_multiplier[k]`; the volume's constant
is derived, `1 - percent_solids - liquor_multiplier["volume"]`. With `T` the influent's
temperature:

```
treated[k] = influent[k] * constant[k] * exponent[k] ** (20 - T)     (volume: no factor)
liquor[k]  = influent[k] * liquor_multiplier[k]                       (overwritten, not blended)
solids[k]  = (influent[k] - treated[k]) - liquor[k]                   (the remainder, unclamped)
```

Three quirks are reproduced. The factor *rises* as the temperature falls when `exponent > 1`.
Nothing caps `constant * factor + liquor_multiplier`: where it exceeds 1 the solids go negative
and the works creates mass, and noodl records the event `"wtw_negative_solids"`. With no
influent the temperature column is whatever the current input holds, 0 after a reset.

The drivers, one row per node of the class:

| Key | Shape | Columns |
|---|---|---|
| `"wsimod.<Class>.process_parameters.constant"`, `"....exponent"` | `(..., n, K_additive)` | the additive pollutants; absent in a quantity-only run |
| `"wsimod.<Class>.liquor_multiplier"` | `(..., n, 1 + K_additive)` | `volume`, then the additive pollutants; `(..., n, 1)` in a quantity-only run |

A configured `process_parameters` or `liquor_multiplier` must name every additive pollutant, and
every `constant`, `exponent` and multiplier must be a real number; an override may be partial
and merges. An `exponent` of 0 or below is refused, also in a batched driver when the day is
bound. Entries for a pollutant that is not additive or not listed, and a `volume` `exponent`,
are refused: WSIMOD never reads them.

## The soil temperature

With a `temperature` pollutant, a `PerviousSurface` moves its storage's temperature every day,
after the IHACRES step:

`T = (T_storage * w_prev + T_air * w_air + T_deep * w_deep) / (w_air + w_deep + w_prev)`

WSIMOD's defaults are `soil_temp_w_prev = 0.1`, `soil_temp_w_air = 0.6`,
`soil_temp_w_deep = 0.1`, `soil_temp_deep = 10`. Any weights are accepted; weights that sum to 0
are refused by name, in the configuration and, for a batched parameter, when the day is bound. A
quantity-only run does not read the temperature.

## River biochemistry

A `River` runs WSIMOD's in-river biochemistry when `nitrate` is among the pollutants, in the
orchestration slot of `calculate_discharge`, before the day's catchment inflow. It needs
`temperature`, `phosphate`, `org-phosphorus`, `ammonia`, `nitrate` and `org-nitrogen` among the
pollutants and a `temperature` forcing on every date. The River reads that forcing in a
quantity-only run too; a quantity-only run ignores the parameters below. The steps, in WSIMOD's
order:

1. the depth `V / area`;
2. the water temperature relaxes towards the air, `T = (1 - 1/T_wdays) T + (1/T_wdays) T_air`,
   on dry and frozen days too;
3. both registers shift in `T` and `phosphate + org-phosphorus`;
4. the guards `V < tolerance` (dry) and `T <= 0` (frozen) end the call with nothing moved;
5. denitrification, `min(denpar_w * area * 2^((T-20)/10) * confcn, 0.5 din)` (times `T/5` below
   5 °C), taken from ammonia and nitrate in proportion;
6. the 10-day and 20-day temperature means and the phosphorus mean;
7. production (`minprodN > 0`: dissolved nitrogen and phosphate into the organic pools) or
   mineralisation (the reverse), each capped at half its pool;
8. macrophyte uptake of dissolved nitrogen and phosphate, capped at half the pools.

The ten parameters are drivers `"wsimod.River.<p>"`, set by overrides only: `uptake_PNratio`,
`denpar_w`, `T_wdays`, `halfsatINwater`, `hsatTP`, `limpppar`, `prodNpar`, `prodPpar`,
`muptNpar` and `muptPpar`; their defaults are WSIMOD's. `max_temp_lag` (20) and
`max_phosphorus_lag` (365) are structural: the registers have one slot more each.
`"River.bio_in"` and `"River.bio_out"` are mass-balance terms that add every call's gains and
losses. The state keys `"wsimod.river_temperature_lag"` and `"wsimod.river_phosphorus_lag"`
hold the two registers, slot 0 the oldest, zero-padded at the front; all zeros is WSIMOD's empty
lists, and a run can start from another run's registers.

As in WSIMOD, each window sum divides by its full length from the first day (a cold start), and
the phosphorus mean holds 366 entries and divides by 365. Mineralisation with no dissolved
nitrogen takes nitrogen from `org-nitrogen`; it leaves through `bio_out` and the event
`"river_nitrogen_lost"` records it. An orchestration that calls `calculate_discharge` twice a
day runs the processes twice, and noodl's balance counts both calls.

Refused by name: a missing pollutant of the six, a missing temperature on a date,
`T_wdays = 0`, `length * width = 0`, `halfsatINwater <= 0`, `max_temp_lag` that is not a
non-negative integer and `max_phosphorus_lag` that is not a positive integer.

Kinks: the four 50 % caps, the `T < 5` factor, the production and mineralisation switch and the
`max(0, .)` of the macrophyte factors are kinks; the dry guard is a jump; `totalphosfcn` is a
singularity as `TP - limpppar + hsatTP` tends to 0 from above.

## Land: growing, irrigation, garden and variable-area surfaces

### The growing surface

A `GrowingSurface` is a `PerviousSurface` whose physical depth is its `rooting_depth`. A
configured `depth` is refused by name; `rooting_depth` is the override. Constructor keywords:
`rooting_depth`, `ET_depletion_factor`, `crop_factor_stages`, `crop_factor_stage_dates`,
`sowing_day`, `harvest_day` and `initial_soil_storage`; its other 25 attributes are reachable
through overrides only.

Each day `calc_crop_cover` runs first. It sows on `sowing_day` and harvests on `harvest_day`,
interpolates the crop factor from `crop_factor_stages` at `crop_factor_stage_dates`,
interpolates the crop and ground cover between sowing and harvest, and sets the day's
`et0_coefficient` to the water-stress coefficient `Ks` times the crop factor. A growing surface
never reads the configured `et0_coefficient` (it is listed in `model.unread`). `Ks` is 1 while
the root-zone depletion is below `RAW = TAW * ET_depletion_factor`, and otherwise
`max(0, (TAW - depletion) / ((1 - ET_depletion_factor) * TAW))`. The calendar is built once,
when the model is built, in WSIMOD's own arithmetic, so it behaves as WSIMOD's does: 1 March
is interpolated at day 59 and 31 December at day 364 in every year, the crop cover starts to
change the day after sowing, and a crop sown after its harvest day (`sowing_day > harvest_day`,
or the default `harvest_day = 365`) has no cover.

Refused by name: `wilting_point >= field_capacity`, `crop_factor_stages` and
`crop_factor_stage_dates` of different lengths, a stage date list on which WSIMOD's
interpolation would raise on one of the run's dates, a `sowing_day` or `harvest_day` that is not
a whole day, and an `adsorption_nr_maxiter` that is not a non-negative integer. As a driver,
`crop_factor_stages` has one row per surface, padded with 0 to the longest list; a padding entry
other than 0 on a shorter surface is refused when the day is bound, naming the surface and the
index. With `nitrate`
among the pollutants a growing surface needs the monthly fertiliser and manure data
(`nhx-fertiliser`, `noy-fertiliser`, `srp-fertiliser`, `nhx-manure`, `noy-manure`,
`srp-manure`, kg/m²/d) in either mode.

**Crop state.** `"wsimod.days_after_sow"`, `"wsimod.crop_cover"` and `"wsimod.ground_cover"`
are carried in both modes, one value per growing surface. The day's crop factor and
`et0_coefficient` are recomputed and not carried; the results record them as
`surface["crop_factor"]` and `surface["et0_coef"]`.

**The nutrient pool.** With `nitrate` among the pollutants a growing surface owns WSIMOD's
nutrient pool: five stores of nitrogen and phosphorus (`fast_pool`, `humus_pool`,
`dissolved_inorganic_pool`, `dissolved_organic_pool`, `adsorbed_inorganic_pool`), carried as
`"wsimod.nutrient_pool"` (kg), ten rows per surface. It starts from the tank: the dissolved
inorganic pool holds `nitrate + ammonia + nitrite` and `phosphate`, the dissolved organic pool
`org-nitrogen` and `org-phosphorus`. A truthy `initial_soil_storage` (a mapping of six keys)
fills the adsorbed and fast pools. The pool is not water, so the mass balance leaves it out.
It is fed by deposition, fertiliser and manure and drained by the surface's outflows and soil
processes. The pool's parameters are drivers with WSIMOD's defaults,
`"wsimod.GrowingSurface.nutrient_pool.<p>"`, `(..., n, 2)` with columns `("N", "P")`, for
`degrhpar`, `dishpar`, `minfpar`, `disfpar`, `immobdpar` and
`fraction_manure_to_dissolved_inorganic`, plus `...fraction_dry_n_to_dissolved_inorganic`
`(..., n)`. WSIMOD cannot configure them, so a surface key or override naming one is refused.
Without `nitrate`, and in a quantity-only run, there is no pool, no pool state and no pool
driver.

**Soil processes**, after the soil temperature, in WSIMOD's order:

- **Temperature factor** of the soil storage's temperature `T`: `2 ** ((T - 20) / 10)` above
  5 °C, `T / 5` on `(0, 5]`, 0 below.
- **Soil-moisture factor** of `smc = V / area`: `satact` at or above field capacity, 0 at or
  below the wilting point, and a minimum of two power laws between.
- **Transformation:** the pool's five transformations (humus to fast, humus and fast to
  dissolved organic, fast to dissolved inorganic, and the immobilisation back), each
  `rate * factors * store`; the net change of each dissolved pool enters or leaves the tank.
- **Crop uptake,** where a crop grows: HYPE's logistic uptake, capped at the dissolved
  inorganic pool's nitrogen and phosphorus above the wilting point; phosphorus first.
- **Erosion** (HYPE's): rain above 5 mm and infiltration excess mobilise sediment, scaled by a
  transport factor; the sediment and its phosphorus go straight into the day's outflows and
  bypass the tank.
- **Denitrification,** on a soil wetter than `limpar` of field capacity; the nitrogen leaves
  the model.
- **Adsorption** (HYPE's Freundlich isotherm): the adsorbed and dissolved inorganic phosphorus
  are split at equilibrium by an unrolled Newton iteration of `adsorption_nr_maxiter` passes
  under a mask, then the adsorbed concentration moves `1 - exp(-kadsdes)` of the way to the
  equilibrium. The convergence limit `adsorption_nr_limit` enters only comparisons and has no
  gradient. A negative `kadsdes` is accepted, as WSIMOD runs it, though it is non-physical (the
  concentration moves away from the equilibrium); below -709.78 WSIMOD's `exp(-kadsdes)`
  overflows and noodl refuses it by name.

**Refused with `nitrate`, in either mode:** a missing one of `ammonia`, `nitrite`, `phosphate`,
`org-nitrogen`, `org-phosphorus`, `solids` and `temperature`; `rooting_depth`, `bulk_density`,
`kfr` or `nfr` at or below 0; `cohesion`, `thetaupp` or `thetalow` of 0; `hsatINs` at or below
0; `limpar` below 0; `field_capacity` at or below 0; and a `VariableAreaSurface`'s
`current_surface_area` below 0. Each is outside the domain of the process equations
(a division by zero or a logarithm of a non-positive number). These checks run at build, and again for every batched instance
when the day is bound, but only with pollutants on. A value inside these domains can still take
the arithmetic out of its domain on an active instance (a very small `nfr`); noodl then raises a `ValueError` naming the surface, the function, the day and
the cause.

### The irrigation surface

An `IrrigationSurface` is a `GrowingSurface` that draws the water its crop lacks from the
`Land`'s in-arcs of type `River`, `Node`, `Groundwater` and `Reservoir`. `Land.apply_irrigation`
runs every one's `irrigate` in surface order; the default orchestration of WSIMOD's own
configurations puts the `Land: apply_irrigation` step after `Land: run`. With a crop (not on the
sowing day) the demand is `max(evaporation - precipitation, 0) * irrigation_coefficient` of the
surface's last IHACRES step, and 0 where the soil is saturated. The supply is forced into the
tank. The irrigation water arrives through the `Land`'s arcs, so the node balance counts it.

`irrigation_coefficient` (default 0.1) is the driver
`"wsimod.IrrigationSurface.irrigation_coefficient"` `(..., n)` and an override; it sets the
volumes, so a quantity-only run reads it. With `nitrate` a pollutant the pool takes the supply's
organic and inorganic nutrients; the irrigated nitrite enters the tank and not the pool.
Without `nitrate`, no pool is updated.

**The climate is carried.** `evaporation` and `precipitation` are the surface's flows from its
last IHACRES step: the same day's when `apply_irrigation` runs after `Land.run`, the previous
day's (0 on day 0) in an orchestration that runs it first. They are state,
`"wsimod.surface_climate"` (m³), two rows per irrigation surface. A crop sown on day 0 is first
irrigated on day 2 when `apply_irrigation` runs before `calc_crop_cover`.

### The garden surface

A `GardenSurface` is a `GrowingSurface` (a grass crop) with two extra methods that its `Land`
binds as the `("Demand", "Garden")` push check and push set. A `ResidentialDemand` asks for them
when it creates its demand: the check gives `max(evaporation - precipitation, 0)` of the
surface's flows, 0 where the soil's moisture deficit is negligible; the demand takes it times
`gardening_efficiency` (default 0.6 x 0.7), pulls it from its supply with the house demand and
pushes it back with the garden tag, and the surface takes it by force. Without a `GardenSurface`
the `Land` answers an empty request and the garden demand is 0. A `NonResidentialDemand` asks
for no garden water. The garden surface has no parameter of its own. The garden water crosses
the `Land`'s in-arcs, so the node balance counts it.

The demand is yesterday's (the climate is carried in `"wsimod.surface_climate"`); the garden
water is at 0 °C; and with two or more garden surfaces on a `Land` only the last is irrigated,
the earlier ones being listed in `model.unread` as `<land>.surfaces.<name>.garden_handlers`. See
[differences](differences.md#land-and-crops).

### The variable-area surface

A `VariableAreaSurface` is a `GrowingSurface` whose rain and evaporation are scaled by the share
of its area in use today: `P` and `E = et0 * et0_coefficient`, each times
`current_surface_area / area`. `current_surface_area` (default 0, m²) is a constructor keyword
and the driver `"wsimod.VariableAreaSurface.current_surface_area"` `(..., n)`; an override
naming it is refused. A surface at the default 0 gets no rain and no evaporation. A
quantity-only run reads it. Erosion still reads the `Land`'s unscaled rain, so an idle surface
erodes on a rainy day, as in WSIMOD.

## Surface pollutant loads and deposition

A surface's `pollutant_load` and its monthly deposition data are live: see
[Running a WSIMOD model](wsimod.md#surface-deposition). The mass-balance parts
`"Surface.atmospheric_deposition"`, `"Surface.precipitation_deposition"`,
`"GrowingSurface.fertiliser"`, `"GrowingSurface.manure"` and `"GrowingSurface.erosion"` record
what enters; `"GrowingSurface.soil_pool_transformation"`, `"GrowingSurface.calc_crop_uptake"`
and `"GrowingSurface.denitrification"` record the removals the tank actually made, so the
balance closes.

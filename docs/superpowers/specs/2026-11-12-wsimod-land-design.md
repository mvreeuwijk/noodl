# WSIMOD land processes in `apps.wsimod` (3c)

Status: designed 2026-11-12 under the user's standing instruction ("do not wait for me and make
sensible decisions"). Rulings Y1–Y8 are the brainstorming rulings, which stand in for the user's
approval. Y9–Y40 are made here (Rulings made in this spec). The Fable spec review approved it
after fixes (2 Important, 12 Minor findings); this revision applies them and the coordinator's
rulings on them as Y41–Y50, and amends Y13, Y14, Y17, Y27, Y34, Y35 and Y38 to match. This document covers sub-project 3c
of 3 of the WSIMOD port (W1), the last one. It builds on 3a (`2026-11-11-wsimod-core-design.md`,
PR #41) and 3b (`2026-11-12-wsimod-processes-design.md`, PR #42, main `13b997d`). Rulings
W1–W95 and X1–X40 still bind and are cited by number.

## Goal

3a carries pollutants, 3b changes them inside water. 3c ports what WSIMOD 0.8.1 does on the
land: pollutants that enter a surface from outside, and the soil nutrient cycle of the growing
surfaces.

- surface deposition: `simple_deposition` (a `pollutant_load`), `atmospheric_deposition` and
  `precipitation_deposition` (monthly `nhx`/`noy`/`srp` data);
- `GrowingSurface`: the crop calendar, crop cover, available water and the crop's evaporation
  factor; the soil nutrient pools (`NutrientPool`) with fertiliser, manure, deposition,
  transformation, crop uptake, erosion, denitrification and phosphorus adsorption;
- its subclasses `IrrigationSurface`, `GardenSurface` (with the `Demand` garden link that 3a
  answers empty) and `VariableAreaSurface`;
- a surface's `data_input_dict` override;
- a decision on `River.pull_check_fp` and the `RiparianBuffer` it names.

After 3c nothing is refused as "3c": every such refusal is ported or refused for good by name
(Y1). The port is faithful, batched, branch-free and differentiable, on the 3a/3b substrate
(Y2), and is judged against live WSIMOD 0.8.1 node by node on replayed inputs, on both demos
running their unmodified `config.yml` with every process live, and on purpose-built cases
(Y3).

Authority: the installed WSIMOD 0.8.1 source, `.venv/Lib/site-packages/wsimod/`. Every
`file:line` is relative to that directory; `land.py` is `nodes/land.py`, `np.py` is
`nodes/nutrient_pool.py`. Where this spec and the source disagree, the source wins and the spec
is corrected (W2). Statements marked "run while writing this spec" were checked by running
WSIMOD 0.8.1 from a script in the session scratchpad (not in the repository), on the
configurations described with each.

## Constraints

The constraints are 3a's and 3b's: branch `wsimod-land` from main `13b997d`, worktree
`C:/Users/mvr/AppData/Local/Temp/noodl-wt/wsimod-land` (Y6); physical names and no aliases;
code, tests and docs change together; no wall-clock assertions; `ruff check .` passes; every
transcribed behaviour cites its WSIMOD `file:line`; WSIMOD is imported only by tests;
`torch.float64` on the CPU only (W94).

## Scope (Y1)

### In: what 3c ports

| WSIMOD class or method (file:lines) | What 3c ports |
|---|---|
| `Surface` deposition (`land.py:375-382`, `:462-581`) | `get_data_input_surface` (`:462-471`), `dry_deposition_to_tank` and `wet_deposition_to_tank` (`:473-503`), `simple_deposition` with any `pollutant_load` (`:505-526`), `atmospheric_deposition` (`:528-554`) and `precipitation_deposition` (`:556-581`) on every surface class |
| `Surface.apply_overrides`'s `data_input_dict` (`land.py:410-421`) | the override as a path or a mapping (Interaction › Overrides) |
| `GrowingSurface` (`land.py:1082-2312`) | `__init__` (`:1085-1324`), `infer_sow_harvest_calendar` (`:1326-1351`), `calculate_available_water` (`:1353-1365`), `apply_overrides` (`:1367-1431`), `pull_storage` (`:1433-1493`), `quick_interp` (`:1495-1512`), `calc_crop_cover` (`:1514-1572`), `adjust_vqip_to_liquid` (`:1574-1603`), `effective_precipitation_flushing` (`:1605-1647`), `fertiliser` (`:1649-1679`), `manure` (`:1681-1710`), `soil_pool_transformation` (`:1744-1811`), `calc_temperature_dependence_factor` (`:1813-1831`), `calc_soil_moisture_dependence_factor` (`:1833-1856`), `calc_crop_uptake` (`:1858-1926`), `erosion` (`:1928-2094`), `denitrification` (`:2096-2152`), `adsorption` (`:2154-2265`), and the `dry_`/`wet_deposition_to_tank` overrides (`:2267-2311`). Its `decays`, soil temperature and nitrite carry-over are 3b's, inherited |
| `NutrientPool` and `NutrientStore` (`np.py:11-564`) | every method a `GrowingSurface` calls: `calculate_fraction_parameters`, `allocate_inorganic_irrigation`, `allocate_organic_irrigation`, `allocate_dry_deposition`, `allocate_wet_deposition`, `allocate_manure`, `allocate_fertiliser`, `extract_dissolved`, `get_erodable_P`, `erode_P`, `soil_pool_transformation`, `temp_soil_process`, `multiply_nutrients`, `receive`, `sum_nutrients`, `subtract_nutrients`, `extract` |
| `IrrigationSurface` (`land.py:2314-2381`) | `irrigate` and its override; `Land.apply_irrigation` (`:192-198`, `:116-118`) |
| `GardenSurface` (`land.py:2384-2437`) | `calculate_irrigation_demand` and `receive_irrigation_demand` as the `Land`'s `("Demand", "Garden")` handlers (`:120-128`), reached from `ResidentialDemand.get_garden_demand` and `create_demand` (`nodes/demand.py:78-117`, `:250-301`) |
| `VariableAreaSurface` (`land.py:2440-2461`) | `get_climate_` |
| `Tank.pull_pollutants` (`nodes/tanks.py:241-266`) | new to noodl's `Tank`; the growing surface's processes call it |

### Out: never ported, with the reason

- `GrowingSurface.residue` (`land.py:1712-1742`): never wired (`:1281` is commented out), and
  `NutrientPool.allocate_residue` would raise `TypeError` (`np.py:312` calls the dict
  `self.empty_nutrient`). Residue data are refused by name (Y36).
- `NutrientPool.apply_overrides` (`np.py:155-181`): nothing calls it; `GrowingSurface` builds its
  pool without arguments (`land.py:1273`). The pool's parameters are drivers with WSIMOD's
  defaults and have no configuration route (Y24).
- `River.pull_check_fp` and the `("RiparianBuffer", "volume")` handler (`storage.py:588-589`,
  `:938-949`): refused for good (Y10). WSIMOD 0.8.1 has no `RiparianBuffer` class (the line is a
  `# TODO`), no class in the package sends that tag (`git grep`-equivalent search of the
  installed package), and the handler reads `self.river_tank`, which no River has
  (`AttributeError`), and returns a tuple, not a VQIP. There is no behaviour to transcribe.
- `Model.change_impervious_area` (`orchestration/model.py:852-891`), which rescales a surface and
  its pools: a scenario utility, not part of a run; out of scope as in 3a (W1).

### Every "3c" refusal, decided

`git grep -n 3c src tests scripts docs/applications docs/formats` on `13b997d` gives the list
below.

| Refusal (noodl file:line) | 3c decision |
|---|---|
| `classes.py:416-420`: `OUT_OF_SCOPE` `GrowingSurface`, `IrrigationSurface`, `GardenSurface`, `VariableAreaSurface` | **ported**; `OUT_OF_SCOPE` keeps only `EnfieldFoulSewer` |
| `config.py:332-339`: a `data_input_dict` override, and a surface `filename` override | a **surface's** `data_input_dict` override is **ported** (`land.py:410-421`). A **node's** `data_input_dict` override is **refused for good**: `Node.apply_overrides` consumes only `filename` (`nodes.py:92-100`) and prints "No override behaviour defined" for anything else. A **surface's** `filename` override is **refused for good**: no surface `apply_overrides` consumes it (`Tank.apply_overrides`, `tanks.py:55-68`, prints) |
| `land.py:15`, `:74`, `:211`, `:313-323`, `:908-909`, `:931-947` (`Land.validate`, `_validate_surface`, `Surface.simple_deposition`): deposition data and a non-zero `pollutant_load` with pollutants on | **ported** (Deposition) |
| `land.py:607`, `:789-791`, `demand.py:253`: the `("Demand", "Garden")` check and push answer empty | **ported** with a `GardenSurface`; without one they stay WSIMOD's empty answer (`land.py:93-94`) |
| `land.py:812-813`: `apply_irrigation` is `lambda: None` | **ported** (`IrrigationSurface.irrigate`) |
| `storage.py:896`, `:1079-1085`: `River.pull_check_fp` | **refused for good** (Y10); the message names the reason instead of 3c |
| `docs/applications/wsimod.md:41`, `:580`, `:594`, `:640`, `:765-768`; `docs/formats/wsimod.md:109`, `:191`, `:203`; `tests/data/wsimod/README.md:7`; `scripts/regenerate_wsimod_fixtures.py:345-361` | rewritten (Documentation); `config_no_deposition.yml` and its writer are retired (Y30) |
| tests: `test_build.py:363`, `test_config.py:415`, `test_land.py:386-405`, `:780-830`, `test_storage.py:920-923`, `test_wsimod_replay.py:658` | rewritten to the new behaviour or the new refusal message |

## Approach: transcribed and batched (Y2)

Everything in 3a's and 3b's Approach sections holds: masks, `MAXITER` rounds, safe denominators,
Python's tie rule for `min`/`max` (first argument unless the second is strictly smaller or
greater), WSIMOD's order of reduction, 1e15 rather than `inf`, the static call structure, and
per-instance Python transcendentals. 3c adds six rules.

1. **Python's transcendentals, per instance.** Every `**`, `exp`, `log`, `log10` and `sin` whose
   argument is a tensor is evaluated per instance by Python:
   - `pow_python` (`vqip.py`) for every power, including `** 2` (`land.py:1893`), and
     `exp_python` (`storage.py`) for every `exp`;
   - three new helpers next to them, `log_python`, `log10_python` and `sin_python` (`math.log`,
     `math.log10`, `math.sin`), with the gradients `1/x`, `1/(x ln 10)` and `cos x`.

   Measured while writing this spec (torch 2.14.1+cpu, Python 3.12.10, 10⁵ samples), the share
   of results where torch differs from Python in the last bit is: `log` 0.03 % (x ∈ [1e-6, 50]),
   `log10` 0.03 %, `sin` 0.16 % on [0, 0.2] and 3.0 % on [−7, 7], `exp` 0.7 % on [−30, 5],
   `x ** (1/2.6)` 1.3 % and `x ** 1.3` 1.5 % (x ∈ [1e-4, 10]). Every other operation is
   `+ − × ÷`, `min`, `max` or `abs` in WSIMOD's operand order, so the replay bound is 0
   (bitwise) for every 3c class.
2. **Date arithmetic at build time.** The day of year, its adjusted value, the sowing and
   harvest tests, the interpolation indices of `quick_interp` and `sin(2π(doy − 70)/365)`
   (`land.py:1948`) depend only on the date and on structural parameters. They are Python
   numbers computed when the model is built, one per day, exactly as WSIMOD computes them
   (`bisect_left` on the same list, the same Python indexing). Only the values interpolated
   (`crop_factor_stages`, `crop_cover_max`, `ground_cover_max`) are tensors (Y14).
3. **Structural switches.** Fixed at build time; Python control flow follows these and nothing
   else:
   - `"nitrate" in POLLUTANTS` (the nutrient processes, `land.py:1276`; also `:1455`, `:1594`);
   - `"nhx-dry"` among the variables of a surface's **constructor** data (deposition,
     `land.py:375-378`), and `len(pollutant_load) > 0` of its **constructor** configuration
     (`:379-380`). Overrides change neither (Y21);
   - each surface's class, and which `GardenSurface` holds the `Land`'s garden handlers (the
     last one, `land.py:120-128`; Y28);
   - `sowing_day`, `harvest_day`, `crop_factor_stage_dates`, `autumn_sow = sowing_day > 181`
     (`:1346`), `adsorption_nr_maxiter` (the Newton loop's length).
4. **The nutrient pool is a dense `(5, 2)` tensor per growing surface** (pools in WSIMOD's
   `self.pools` order, `np.py:130-136`: `fast_pool`, `humus_pool`, `dissolved_inorganic_pool`,
   `dissolved_organic_pool`, `adsorbed_inorganic_pool`; nutrients `N`, `P`,
   `constants.NUTRIENTS`). WSIMOD's in-place `+=` and `-=` become functional writes in WSIMOD's
   statement order. Each `NutrientStore.extract` is `reply = min(storage, amount)` then `storage
   −= reply`, per nutrient in the request dict's insertion order (`np.py:551-556`); the
   nutrients are independent, so order changes nothing numerically, but the decision records
   follow it.
5. **The Newton–Raphson loop is unrolled** (`land.py:2205-2220`). `adsorption_nr_maxiter`
   iterations run on every instance under a `running` mask that the `while` test, the `break`
   and `j < maxiter` clear. An instance whose loop has stopped keeps its `xn`, `xn_1`, `fxn`
   unchanged, so each instance's arithmetic is exactly Python's (Y25).
6. **Python's integer zeros.** WSIMOD assigns the integers `0` and `1` in several branches
   (`land.py:1562`, `:1829`, `:1846`, `:1956`, `:1970`, `:2108`, `:2115`). `0 * x` with a float
   `x` is `0.0` with `x`'s sign rule, exactly as torch's `0.0 * x`, so noodl's float constants
   give the same values. The sign of a zero can differ: WSIMOD's pools start as the integer `0`
   (`np.py:185`), so `-increase` is the integer `0` (giving `+0.0`) on a pool never touched and
   `-0.0` on a float one, where noodl gives `-0.0`. The harness compares with `torch.equal`, for
   which `-0.0 == 0.0`, so "bitwise" in this spec means equal under that comparison (Y50).

## The equations, transcribed

`FA` is `FLOAT_ACCURACY`. A surface's `V` is `storage["volume"]`, `smc = V / area`
(`get_smc`, `land.py:895-903`), `cmd = get_excess()["volume"] / area` (`get_cmd`, `:886-893`,
with `get_excess` = `max(capacity − V, 0)`, `tanks.py:151`). The constants are WSIMOD's
(`core/constants.py`): `M_TO_MM = 1e3`, `M_TO_KM = 1e-3`, `M2_TO_KM2 = 1e-6`,
`G_M2_TO_KG_M2 = 1e-3`, `KG_M2_TO_KG_KM2 = 1e6`, `KG_M3_TO_KG_KM3 = 1e9`,
`KG_M3_TO_MG_L = 1e3`, `KG_TO_MG = 1e6`, `PI = 3.141592653589793`. Every expression is in
WSIMOD's operation order; `a *= b * c` means `a = a * (b * c)`.

### Surface deposition (`land.py:375-382`, `:462-581`)

**Which functions run** (construction, `:375-382`): if `"nhx-dry"` is a variable of the
surface's constructor `data_input_dict`, `inflows = [atmospheric_deposition,
precipitation_deposition]`; then, if the constructor's `pollutant_load` is non-empty,
`simple_deposition` is appended. Subclasses append their own functions after these.

**The data** (`get_data_input_surface`, `:462-471`): `data_input_dict[(var, parent.monthyear)]`,
where `monthyear = date.to_period("M")` (`orchestration/model.py:997`), WSIMOD's `to_datetime`
of `"YYYY-MM"`: the first of the month, compared by `datetime` equality
(`model.py:103-112`, `:164-170`). A data row dated `"2000-01"` or `"2000-01-01"` matches; one
dated `"2000-01-15"` never does. noodl's `parse_date` already normalises dates this way.

**`simple_deposition`** (`:505-526`):

```
pollution = empty
for pol, item in pollutant_load.items():                 # :516-520, the dict's order
    pollution[pol] = item * area if pol in ADDITIVE else item
pollution["volume"] = 0                                    # :521
push_storage(pollution, force=True)                        # :524
return (pollution, empty)
```

A non-additive entry blends into the storage at volume 0: `sum_vqip` gives `(item·0 + t·V)/V`,
which is `t` to rounding, or keeps `t` when `V = 0` (`core.py:104-114`). It is accepted and
transcribed (Y38). A key that is not a configured pollutant is written into a key no sum reads
(`core.py:83-116` iterates the configured lists), so it has no effect; it is listed in
`model.unread` (Y38).

**`atmospheric_deposition`** (`:528-554`) and **`precipitation_deposition`** (`:556-581`):

```
nhx = data(nhx-dry) * area;  noy = data(noy-dry) * area;  srp = data(srp-dry) * area   # :540-542
vqip = empty; vqip["ammonia"] = nhx; vqip["nitrate"] = noy; vqip["phosphate"] = srp      # :545-548
in_ = dry_deposition_to_tank(vqip)                                                       # :551
return (in_, empty)
```

and the same with the `-wet` variables and `wet_deposition_to_tank`. `Surface`'s two
`*_deposition_to_tank` push `vqip` with `force=True` and return it (`:473-503`). The
`GrowingSurface` overrides route it through the pool (below).

- All six variables are read once `nhx-dry` is present (`KeyError` otherwise; run while writing
  this spec: `KeyError: ('nhx-wet', 2001-01)` at `land.py:471`).
- A surface whose data has wet variables but no `nhx-dry` reads none of them: listed in
  `model.unread`.
- If `ammonia`, `nitrate` or `phosphate` is not a configured pollutant, its deposition is read
  and dropped (the key is not summed). noodl reads it (the forcing check applies) and lists it
  in `model.unread`.

### `GrowingSurface`: construction (`land.py:1085-1324`)

`GrowingSurface(PerviousSurface)` passes `depth = rooting_depth` to `PerviousSurface`
(`:1192`, `:1251`), so the physical depth is the rooting depth: `field_capacity_m =
field_capacity · rooting_depth`, `wilting_point_m = wilting_point · rooting_depth`, the tank's
depth `rooting_depth · total_porosity` and `capacity = area · depth` (`:796-816`, 3a). A
configured `depth` raises `TypeError` (multiple values for `depth`; run while writing this
spec) and is refused by name.

**Attributes set only in `__init__`** (`:1186-1250`), reachable through overrides only
(`:1376-1410`), with their defaults as written (Python evaluates the expressions):

| Attribute | Default | Unit (WSIMOD's comment) |
|---|---|---|
| `crop_cover_max`, `ground_cover_max` | 0.9, 0.3 | - |
| `satact`, `thetaupp`, `thetalow`, `thetapow` | 0.6, 0.12, 0.08, 1 | - |
| `uptake1`, `uptake2`, `uptake3` | 15, 1, 0.02 | g/m2/y, -, 1/d |
| `uptake_PNratio` | `1 / 7.2` | - |
| `erodibility`, `sreroexp`, `cohesion`, `slope` | 0.0025, 1.2, 1, 5 | g d/(J mm), -, kPa, % |
| `srfilt`, `macrofilt` | 0.7, 0.01 | - |
| `limpar`, `exppar`, `hsatINs`, `denpar` | 0.7, 2.5, 1, 0.015 | -, -, mg/l, - |
| `adosorption_nr_limit` (WSIMOD's spelling), `adsorption_nr_maxiter` | 0.00001, 20 | -, - |
| `kfr`, `nfr`, `kadsdes` | 153.7, `1 / 2.6`, 0.03 | l/kg, -, 1/d |
| `bulk_density` | 1300 | kg/m3 |

The constructor keywords are `rooting_depth = 1`, `ET_depletion_factor = 1`,
`crop_factor_stages = [1, 1]`, `crop_factor_stage_dates = [0, 365]`, `sowing_day = 1`,
`harvest_day = 365`, `initial_soil_storage = None` (`:1085-1094`), and `PerviousSurface`'s
(3a) without `depth`.

**Derived** (recomputed after overrides, `:1422-1431`, and by noodl's `bind`):

```
harvest_sow_calendar = [0, sowing_day, harvest_day, harvest_day + 1, 365]    # :1335-1341
ground_cover_stages  = [0, 0, ground_cover_max, 0, 0]                          # :1342
crop_cover_stages    = [0, 0, crop_cover_max, 0, 0]                            # :1343
autumn_sow = sowing_day > 181                                                   # :1346-1349
TAW = field_capacity_m - wilting_point_m                                        # :1360
RAW = TAW * ET_depletion_factor                                                 # :1363
```

`TAW < 0` only prints `"warning: TAW < 0..."` (`:1361-1362`), and `TAW = 0` crashes on day 0
(the stress division below); noodl refuses `wilting_point ≥ field_capacity`, that is `TAW ≤ 0`
(Y27, Y44).

**State after construction** (`:1261-1270`): `days_after_sow = None`, `crop_cover = 0`,
`ground_cover = 0`, `crop_factor = 0`, `et0_coefficient = 1`. The configured
`et0_coefficient` of `PerviousSurface` is overwritten here and every day by `calc_crop_cover`,
so a growing surface never reads it (listed in `model.unread`; Y39).

**The pool** (`:1273`, `:1293-1324`, only when `nitrate` is a pollutant): `NutrientPool()` with
its defaults, then

```
DI.P = storage["phosphate"];   DI.N = storage["nitrate"] + storage["ammonia"] + storage["nitrite"]
DO.P = storage["org-phosphorus"];   DO.N = storage["org-nitrogen"]
if initial_soil_storage:                                       # :1308, truthy
    ads.P = iss["phosphate"];  ads.N = iss["ammonia"] + iss["nitrate"] + iss["nitrite"]
    fast.N = iss["org-nitrogen"];  fast.P = iss["org-phosphorus"]
```

Every pool starts at the integer 0 (`np.py:185`, `:189-190`). `initial_soil_storage` must be a
mapping with those six keys (`KeyError` otherwise) or falsy; a float, which the docstring
allows (`:1142`), raises `TypeError` (run while writing this spec: `'float' object is not
subscriptable` at `:1312`) and is refused (Y37). The adsorbed pool's `N` is set here and never
read again.

**Function lists** (`:1275-1291`):

- `inflows.insert(0, calc_crop_cover)`;
- with `nitrate`: `inflows += [effective_precipitation_flushing, fertiliser, manure]`, and
  `processes += [calc_temperature_dependence_factor, calc_soil_moisture_dependence_factor,
  soil_pool_transformation, calc_crop_uptake, erosion, denitrification, adsorption]`.

### `GrowingSurface`: the day's call order

`Land.run` calls each surface's `Surface.run` (`land.py:200-205`, `:425-449`). For a growing
surface the order is:

| # | Function (file:line) | When |
|---|---|---|
| 0 | the nitrite carry-over (3b, `:427-439`) | `nitrite` a pollutant |
| 1 | `calc_crop_cover` (`:1514-1572`) | always |
| 2 | `atmospheric_deposition` → `GrowingSurface.dry_deposition_to_tank` (`:528-554`, `:2267-2288`) | constructor data has `nhx-dry` |
| 3 | `precipitation_deposition` → `wet_deposition_to_tank` (`:556-581`, `:2290-2311`) | same |
| 4 | `simple_deposition` (`:505-526`) | constructor `pollutant_load` non-empty |
| 5 | `ihacres` (3a, `:913-1044`), with `get_climate` (or `VariableAreaSurface.get_climate_`) and `GrowingSurface.pull_storage` twice (`:995-996`) | always |
| 6 | `effective_precipitation_flushing` (`:1605-1647`) | `nitrate` |
| 7 | `fertiliser` (`:1649-1679`) | `nitrate` |
| 8 | `manure` (`:1681-1710`) | `nitrate` |
| 9 | `calculate_soil_temperature` (3b, `:1060-1079`) | always |
| 10 | `calc_temperature_dependence_factor` (`:1813-1831`) | `nitrate` |
| 11 | `calc_soil_moisture_dependence_factor` (`:1833-1856`) | `nitrate` |
| 12 | `soil_pool_transformation` (`:1744-1811`) | `nitrate` |
| 13 | `calc_crop_uptake` (`:1858-1926`) | `nitrate` |
| 14 | `erosion` (`:1928-2094`) | `nitrate` |
| 15 | `denitrification` (`:2096-2152`) | `nitrate` |
| 16 | `adsorption` (`:2154-2265`) | `nitrate` |
| 17 | `route` (3a, `:1046-1058`) | always |

Each function's `(in_, out_)` is summed into the parent's `running_inflow_mb` and
`running_outflow_mb` (`:441-449`). Later in the day, `Land.apply_irrigation` (default
orchestration slot 11, `model.py:202`) calls every `IrrigationSurface.irrigate` in surface
order (`:192-198`, `:116-118`). Earlier, `Demand.create_demand` (slot 2, `model.py:193`) calls
the garden handlers. At the end of the day `Land.end_timestep` ends every surface (3b's decay)
and the residence tanks (`:266-275`).

### `calc_crop_cover` (`land.py:1514-1572`)

```
doy = t.dayofyear                                         # :1524
if t.is_leap_year:                                        # :1526  -- see below: always true
    if doy > 59: doy -= 1                                 # :1528-1529
if days_after_sow is None:                                # :1531
    if t.dayofyear == sowing_day: days_after_sow = 0      # :1532-1534  (raw day of year)
else:
    if t.dayofyear == harvest_day:                        # :1536
        days_after_sow = None; crop_factor = crop_factor_stages[0]
        crop_cover = 0; ground_cover = 0                  # :1537-1541
    else:
        days_after_sow += 1                               # :1544
crop_factor = quick_interp(doy, crop_factor_stage_dates, crop_factor_stages)     # :1547-1549
if days_after_sow:                                        # :1550, None and 0 are false
    crop_cover   = quick_interp(doy, harvest_sow_calendar, crop_cover_stages)    # :1553-1555
    ground_cover = quick_interp(doy, harvest_sow_calendar, ground_cover_stages)  # :1556-1558
root_zone_depletion = max(field_capacity_m - smc, 0)     # :1560
if root_zone_depletion < RAW: Ks = 1                      # :1561-1562
else: Ks = max(0, (TAW - root_zone_depletion) / ((1 - ET_depletion_factor) * TAW))   # :1564-1568
et0_coefficient = Ks * crop_factor                        # :1570
return (empty, empty)
```

`quick_interp(x, xp, yp)` (`:1495-1512`):

```
i = bisect_left(xp, x); x_left = xp[i - 1]; x_right = xp[i]; dif = x - x_left
y = yp[i - 1] + (yp[i] - yp[i - 1]) * dif / (x_right - x_left)      # ((Δy · dif) / Δx), then +
```

Quirks, all transcribed:

- **The leap test is always true on the individual-files loader.** `Model.load` has two data
  loaders. `_load_individual_files`, which reads the `config.yml` + CSV format that noodl reads,
  builds the dates with WSIMOD's own `to_datetime` (`model.py:347-348`, `:39-170`), whose
  `is_leap_year` is a **method** (`:114-121`), not a property. `if t.is_leap_year:` tests the
  bound method, which is truthy. So every year, not only leap years, shifts the day of year down
  by one after 28 February: in a common year 1 March and 28 February both interpolate at 59, and
  31 December at 364 (run while writing this spec: `bool(to_datetime("2001-03-01").is_leap_year)`
  is `True`). The unified-data loader (`_load_unified_data`, `model.py:276-322`, the parquet
  format) sets `self.dates = pd.to_datetime(...)`: pandas timestamps, whose `is_leap_year` is a
  property, so there the shift happens only in leap years, as intended. noodl reproduces the
  individual-files loader, the format it reads, and the test harness builds `model.dates` with
  `to_datetime` too (`_wsimod_reference.py:318-319`). A user comparing with a parquet-loaded
  WSIMOD sees crop factors that differ by a day in common years; the application page's L1 row
  says so (Y13, Y42).
- **Raw and adjusted days.** The sowing and harvest tests use the raw `t.dayofyear`; the
  interpolations use the adjusted `doy`; erosion's `sin` uses the raw one (`:1948`).
- **The sowing day itself has no crop.** `days_after_sow = 0` is false at `:1550` (and at
  `:1872`, `:2350`), so cover, uptake and irrigation start the day after sowing.
- **Cover persists.** Outside the season `crop_cover` and `ground_cover` keep their last values
  until harvest sets them to 0; they are carried state (State).
- **Dead store.** The harvest branch's `crop_factor = crop_factor_stages[0]` (`:1539`) is
  overwritten at `:1547`; it is not ported (no observable effect).
- **The calendar is unsorted** whenever `sowing_day > harvest_day` or `harvest_day = 365`.
  The default (`harvest_day = 365`) gives `[0, 1, 365, 366, 365]`, which is benign: the adjusted
  day of year never exceeds 365, and `bisect_left` returns 2 on every in-season day; the
  default-calendar unit test covers it. With `sowing_day > harvest_day` the calendar is, for
  example, `[0, 280, 200, 201, 365]`. `bisect_left` on it returns indices whose segments all
  have `y = 0`, so an autumn-sown crop has `crop_cover = ground_cover = 0` all year (run while
  writing this spec: the maximum over days 1–365 is 0.0 for sowing 280 and harvest 200, and 0.9
  for sowing 90 and harvest 250). Reproduced exactly by rule 2; documented (Y15).
- **Index wrap.** If `x ≤ xp[0]`, `bisect_left` returns 0 and `xp[-1]`, `yp[-1]` are read
  (Python's negative index); for `xp = [1, 365]`, `yp = [0.3, 1]` at `x = 1` this gives
  `0.30000000000000004` (run while writing this spec). Reproduced. `x > xp[-1]` raises
  `IndexError` and `x_right = x_left` divides by zero: both are known at build time, and noodl
  refuses the configuration by name, naming the first date (Y14). The dates checked are the
  run's `dates`, since WSIMOD only crashes on those (`[0, 300]` with a run ending in October
  runs in WSIMOD). A `model.step` on a date past them is checked when that step is taken, and
  refused by name there (Y47). `len(crop_factor_stages) ≠
  len(crop_factor_stage_dates)` is refused (an `IndexError` on some day, or unread entries).
- **The stress coefficient divides by zero** when `(1 − ET_depletion_factor) · TAW = 0`, and it
  reaches that branch when `root_zone_depletion ≥ RAW`. With the **default**
  `ET_depletion_factor = 1` that is any soil at or below its wilting point. (With `TAW = 0` it
  would be every day, and the guard below would make the soil never evaporate; `wilting_point ≥
  field_capacity` is therefore refused, Y44, and the guard covers `ET_depletion_factor = 1`
  only.) WSIMOD raises `ZeroDivisionError` (run while writing this spec: a default
  growing surface drying under `et0 = 0.01` crashed at `land.py:1566`). noodl does the evident
  intent: `Ks = 0`. That is the value of the formula's limit, and FAO-56's `Ks` at the wilting
  point. Python's `max(0, ·)` gives the same for IEEE's `−inf` and `nan`. noodl records the
  event `"crop_water_stress_zero_denominator"` (Y17).

### `GrowingSurface.pull_storage` (`land.py:1433-1493`)

`ihacres` pulls the subsurface flow and then the percolation through it (`:995-996`):

```
if V == 0: return empty                                         # :1446
reply = min(request["volume"], V)                                # :1450
reply = v_change_vqip(storage, reply)                            # :1453
if nitrate:
    prop = reply["volume"] / V                                   # :1457
    nutrients = pool.extract_dissolved(prop)                     # :1458
    din = storage["nitrate"] + storage["ammonia"] + storage["nitrite"]
    reply["nitrate"] = nutrients.inorganic.N * storage["nitrate"] / din   # :1459-1467
    reply["ammonia"] = nutrients.inorganic.N * storage["ammonia"] / din   # :1468-1476
    reply["nitrite"] = nutrients.inorganic.N * storage["nitrite"] / din   # :1477-1485
    reply["phosphate"] = nutrients.inorganic.P                    # :1486
    reply["org-phosphorus"] = nutrients.organic.P;  reply["org-nitrogen"] = nutrients.organic.N
storage = extract_vqip(storage, reply)                           # :1491
```

`extract_dissolved(prop)` (`np.py:330-356`): it extracts `{N: DI.N·prop, P: DI.P·prop}` from
`DI`, then `{N: DO.N·prop, P: DO.P·prop}` from `DO`. Each request is computed before its pool
changes.

- The volume part is `Tank.pull_storage`'s (`tanks.py:226-239`, 3a), so volumes never depend on
  the pools.
- The tank's nutrient columns leave in the **pool's** proportions, not the tank's. Where the
  pool holds more than the tank, a column can go negative (see the pool–tank rows of the defect
  table).
- **`din = 0`** with `V > 0` raises `ZeroDivisionError` (run while writing this spec: a growing
  surface whose initial storage held no inorganic N, with rain on day 0 and no fertiliser,
  crashed at `land.py:1460`). The trigger is ordinary: any growing surface started from a tank
  with no inorganic N, with zero fertiliser, manure and deposition in its first month (a winter
  start), crashes on the first day with net flow through the soil. Fertiliser, the main N
  source, enters only after `ihacres` (call order 7). Deposition data mask the day-0 crash,
  because dry and wet deposition add N to the tank before `ihacres` (call orders 2-3). WSIMOD
  therefore needs initial inorganic N or a non-zero first-month input; the application page
  says so. noodl does the evident intent: the three species get 0, because the tank has none to
  give. The pool has still lost its share, as WSIMOD's other pool–tank gaps lose it. Event
  `"pull_storage_no_din"` (Y18, Y46).

### Effective precipitation flushing, fertiliser and manure (`land.py:1605-1710`)

**Flushing** (`:1605-1647`), immediately after `ihacres`:

```
out.N = ss.ammonia + ss.nitrite + ss.nitrate + perc.ammonia + perc.nitrite + perc.nitrate
        + ie.ammonia + ie.nitrite + ie.nitrate                                  # :1615-1625
out.P = ss.phosphate + perc.phosphate + ie.phosphate                            # :1626-1630
DI.extract(out)                                                                  # :1631
out.N = ss.org-nitrogen + perc.org-nitrogen + ie.org-nitrogen                   # :1635-1639
out.P = ss.org-phosphorus + perc.org-phosphorus + ie.org-phosphorus             # :1640-1644
DO.extract(out)                                                                  # :1645
return (empty, empty)
```

Here `ss`, `perc` and `ie` are `subsurface_flow`, `percolation` and `infiltration_excess`, as
`ihacres` left them.

- `ie` carries no nutrients at this point, because erosion adds to it only later (call order
  14). Its terms are therefore 0.
- **The pools lose the outflows' nutrients twice.** `pull_storage` has already extracted them
  through `extract_dissolved` (`:1458`), and flushing extracts them again.
- Running WSIMOD while writing this spec, a growing surface after 120 days held
  - tank phosphate 1.158 against the pool's 0.052;
  - tank `org-nitrogen` 1.0007 against 0.00068.

  With flushing removed from `inflows`, the same run gave tank P `0.05309067078560237` and pool
  P `0.0530906707856026`.
- Reproduced (Y16; defect table).

**Fertiliser** (`:1649-1679`):

```
nhx = data(nhx-fertiliser) * area; noy = data(noy-fertiliser) * area; srp = data(srp-fertiliser) * area
vqip = empty; vqip.ammonia = nhx; vqip.nitrate = noy; vqip.phosphate = srp       # :1664-1667
dep.N = vqip.nitrate + vqip.ammonia;  dep.P = vqip.phosphate                       # :1671-1672
in_ = pool.allocate_fertiliser(dep)            # DI.receive(dep) (N, then P); returns dep (np.py:327-328)
vqip = adjust_vqip_to_liquid(vqip, dep, in_)
push_storage(vqip, force=True);  return (vqip, empty)
```

**Manure** (`:1681-1710`): the same with the `-manure` variables and `allocate_manure`
(`np.py:267-290`):

```
DI.receive({N: dep.N * fmdi.N, P: dep.P * fmdi.P})      # fmdi = fraction_manure_to_dissolved_inorganic
fast.receive({N: dep.N * fmf.N, P: dep.P * fmf.P})      # fmf = 1 - fmdi, per nutrient (np.py:146-149)
in_ = {N: dep.N * fmdi.N, P: dep.P * fmdi.P}             # recomputed, the same values
```

**`adjust_vqip_to_liquid(vqip, dep, in_)`** (`:1574-1603`, with `nitrate`):

```
if dep.N > 0: vqip.nitrate *= in_.N / dep.N; vqip.ammonia *= in_.N / dep.N; vqip.org-nitrogen *= in_.N / dep.N
if dep.P > 0: vqip.phosphate *= in_.P / dep.P; vqip.org-phosphorus *= in_.P / dep.P
```

Each line is `x = x * (in_/dep)`, the ratio first. For fertiliser and wet deposition `in_` is
`dep` itself, so the ratio is exactly 1.0 for any finite positive value, and noodl still
evaluates it. Fertiliser and manure data are read whenever `nitrate` is a pollutant: missing
data raise `KeyError` (run while writing this spec: `('nhx-fertiliser', 2001-01)`). The forcing
check refuses them by name.

**Deposition into a growing surface** (`:2267-2311`):

- **Dry.** `dep = {N: nitrate + ammonia, P: phosphate}`, then `allocate_dry_deposition`
  (`np.py:226-248`):

  ```
  fast.N += dep.N * fdf          # fdf = 1 - fraction_dry_n_to_dissolved_inorganic (np.py:153)
  DI.N   += dep.N * fddi         # fddi = fraction_dry_n_to_dissolved_inorganic
  ads.P  += dep.P
  in_ = {N: dep.N * fddi, P: 0}
  ```

  Then `adjust_vqip_to_liquid` and `push_storage`. The dry phosphate goes entirely to the
  adsorbed pool, and the tank gets `phosphate · (0/P) = 0`.
- **Wet.** `allocate_wet_deposition` is `DI.receive(dep)`, and the ratio is 1.
- Without `nitrate` the pool calls still run, on an inert pool, and `adjust_vqip_to_liquid`
  does nothing (`:1594`). The tank then gets the whole deposition. noodl skips the inert pool,
  which has no observable effect.

### `soil_pool_transformation` (`land.py:1744-1811`, `np.py:396-481`)

```
in_ = empty; out_ = empty
nitrate_proportion = storage.nitrate / (storage.nitrate + storage.ammonia)      # :1758-1760
# NutrientPool.soil_pool_transformation (np.py:409-450):
a1 = tsp(degrhpar, humus, fast)                    # humus -> fast, untracked (:413)
a2 = tsp(dishpar, humus, DO);  inc_DO = 0 + a2     # :418-423 (sum_nutrients from the empty nutrient)
a3 = tsp(minfpar, fast, DI);   inc_DI = 0 + a3     # :426-431
a4 = tsp(disfpar, fast, DO);   inc_DO = inc_DO + a4   # :434-439
a5 = tsp(immobdpar, DI, fast); inc_DI = inc_DI - a5   # :442-447
#   tsp(par, X, Y): x[n] = par[n] * tdf * smdf * X[n] for n in (N, P); x = X.extract(x); Y.receive(x)
#   (np.py:468-481; tdf, smdf are the pool's temperature and soil-moisture factors of the day)
if inc_DI.N > 0: in_.nitrate = inc_DI.N * nitrate_proportion; in_.ammonia = inc_DI.N * (1 - nitrate_proportion)
else:           out_.nitrate = -inc_DI.N * nitrate_proportion; out_.ammonia = -inc_DI.N * (1 - nitrate_proportion)
if inc_DO.N > 0: in_.org-nitrogen = inc_DO.N   else: out_.org-nitrogen = -inc_DO.N
if inc_DI.P > 0: in_.phosphate = inc_DI.P      else: out_.phosphate = -inc_DI.P
if inc_DO.P > 0: in_.org-phosphorus = inc_DO.P else: out_.org-phosphorus = -inc_DO.P   # :1770-1802
push_storage(in_, force=True)                                                     # :1805
out_ = pull_pollutants(out_)                                                       # :1806
return (in_, out_)
```

**`Tank.pull_pollutants(vqip)`** (`tanks.py:261-266`): for each `pol` in `ADDITIVE_POLLUTANTS
+ ["volume"]`, `vqip[pol] = min(storage[pol], vqip[pol])`; then `storage =
extract_vqip(storage, vqip)`. It **mutates and returns its argument**. So the returned `out_` is
what the tank actually lost, and WSIMOD's lists close. The check that follows, `if not
compare_vqip(out_, out2_)` (`:1808`), compares an object with itself and can never print
"nutrient pool not tracking soil tank" (run while writing this spec: `pull_pollutants(req) is
req`). noodl transcribes the clamp and records, where the clamp removes more than `FA` from any
column, the event `"nutrient_pool_not_tracking"`. That is the evident intent of the dead check,
at this site and at the three below (Y16).

- `nitrate_proportion` ignores nitrite. When `nitrate + ammonia = 0`, WSIMOD raises
  `ZeroDivisionError` (run while writing this spec: a growing surface holding only nitrite, with
  no fertiliser, crashed at `land.py:1758`; the review reproduced it on **day 0** with a plain
  zero-N tank and zero first-month inputs, no nitrite needed). This, too, is every growing
  surface started from an empty tank in winter. noodl uses proportion 0, which sends the change
  to ammonia, the product of mineralisation, and records `"transformation_no_din"` (Y18, Y46).
- An increase of exactly 0 takes the `else` branches: `out_` gets `−0.0` (or `+0.0`, rule 6),
  and the clamp `min(storage, −0.0)` returns `storage` when that is `+0.0`; the result equals
  Python's under `torch.equal`.

### The soil factors (`land.py:1813-1856`)

```
T = storage["temperature"]                                          # after the soil temperature (call 9)
if T > 5:   tdf = 2 ** ((T - 20) / 10)                               # :1822-1825
elif T > 0: tdf = T / 5                                              # :1826-1827
else:       tdf = 0                                                  # :1828-1829
pool.temperature_dependence_factor = tdf

if smc >= field_capacity_m:   smdf = satact                          # :1843-1844
elif smc <= wilting_point_m:  smdf = 0                               # :1845-1846
else:
    fc_comp = ((field_capacity_m - smc) / (thetaupp * rooting_depth)) ** thetapow
    fc_comp = (1 - satact) * fc_comp + satact                        # :1848-1850
    wp_comp = ((smc - wilting_point_m) / (thetalow * rooting_depth)) ** thetapow   # :1851-1852
    smdf = min(1, wp_comp, fc_comp)                                  # :1853-1855
pool.soil_moisture_dependence_factor = smdf
```

`min(1, a, b)` takes the first of the smallest, in that order. The temperature factor jumps at
5 °C, from `T/5 → 1` below to `2^(−1.5) = 0.35357789789620914` above (run while writing this
spec). It is therefore larger at 4.9 °C than at 15 °C (0.7071). HYPE, and WSIMOD's own River
(`storage.py:775-777`), use the product `2^((T−20)/10) · T/5` below 5 °C. Reproduced and
documented (Y19). Both factors are the day's: they are set before every reader, so they are not
carried.

### `calc_crop_uptake` (`land.py:1858-1926`)

```
N_common = 0; P_common = 0
if days_after_sow:                                                         # :1872
    d = days_after_sow
    if autumn_sow: temp_func = max(0, min(1, (T - 5) / 20)); d -= 25        # :1877-1879
    else:          temp_func = 1                                            # :1880-1881
    uptake_par = (uptake1 - uptake2) * exp(-uptake3 * d)                    # :1884-1886
    if (uptake_par + uptake2) > 0:                                          # :1887
        N_common = uptake1 * uptake2 * uptake3 * uptake_par / ((uptake2 + uptake_par) ** 2)   # :1888-1894
    N_common *= temp_func * G_M2_TO_KG_M2 * area                            # :1895
    P_common = N_common * uptake_PNratio                                    # :1896
    N_max = max(0, V - wilting_point_m * area) / V * DI.N                   # :1898-1902
    P_max = max(0, V - wilting_point_m * area) / V * DI.P                   # :1903-1907
    uptake = {"P": min(P_common, P_max), "N": min(N_common, N_max)}         # :1909-1912
    crop_uptake = DI.extract(uptake)                                        # :1913 (P, then N)
    out_.nitrate = crop_uptake.N;  out_.phosphate = crop_uptake.P           # :1917-1918
    out_ = pull_pollutants(out_)                                            # :1920
    return (empty, out_)
return (empty, empty)
```

- `T` is the soil storage's temperature.
- All the uptake is nitrate (`:1916`). When the tank holds less nitrate than the pool's DIN
  share, the clamp leaves the tank's ammonia and nitrite while the pool has lost them (defect
  table). The event `"nutrient_pool_not_tracking"` applies.
- **`V = 0`** with a crop present raises `ZeroDivisionError` (run while writing this spec: a
  growing surface with `initial_storage = 0`, no rain and crops sown crashed at `land.py:1899`).
  It is reachable only with `ET_depletion_factor < 1`: with the default 1 an empty soil crashes
  first at `:1566` (L3) on day 0 (the review reproduced `:1899` at `p = 0.5`,
  `days_after_sow = 1`).
  noodl does the evident intent: no water, no available uptake, so `N_max = P_max = 0`. Event
  `"crop_uptake_dry_soil"` (Y18).

### `erosion` (`land.py:1928-2094`, `np.py:358-394`)

```
pd = precipitation * M_TO_MM                                                 # :1940 (the Land's forcing)
if pd > 5:                                                                    # :1943
    re = 8.95 + 8.44 * log10(pd * (0.257 + sin(2 * PI * ((t.dayofyear - 70) / 365)) * 0.09) * 2)   # :1944-1952
    re *= pd                                                                  # :1953
    mobilised_rain = re * (1 - crop_cover) * erodibility                      # :1954
else: mobilised_rain = 0
if ie.volume > 0:                                                             # :1959
    mf = (ie.volume / area * M_TO_MM * 365) ** sreroexp                       # :1960-1962
    mf *= (1 - ground_cover) * (1 / (0.5 * cohesion)) * sin(slope / 100) / 365   # :1963-1968
else: mf = 0
total_flows = ie.volume + ss.volume + perc.volume                             # :1974-1978
erodingflow = total_flows / area * M_TO_MM                                    # :1981
transportfactor = min(1, (erodingflow / 4) ** 1.3)                            # :1984
erodedsed = 1000 * (mf + mobilised_rain) * transportfactor                    # :1985-1987
if erodingflow > 4:   enrichment = 1.5                                        # :1992-1993
elif erodingflow > 0: enrichment = 4 - (4 - 1.5) * erodingflow / 4            # :1994-1995
else: return (empty, empty)                                                   # :1996-1997
erodableP = pool.get_erodable_P() / area * KG_M2_TO_KG_KM2                    # :2000-2002 (ads.P + humus.P)
erodedP = erodedsed * (erodableP / (rooting_depth * M_TO_KM * bulk_density * KG_M3_TO_KG_KM3)) * enrichment
erodedP *= area * M2_TO_KM2;  erodedsed *= area * M2_TO_KM2                   # :2018-2019
surface_P = srfilt * ie.volume / total_flows * erodedP;  surface_sed = srfilt * ie.volume / total_flows * erodedsed
sub_P = macrofilt * ss.volume / total_flows * erodedP;   sub_sed = ...         # :2029-2034
perc_P = macrofilt * perc.volume / total_flows * erodedP; perc_sed = ...       # :2036-2041
eff_erodedP = perc_P + surface_P + sub_P                                       # :2047
if eff_erodedP > 0:                                                            # :2048
    org, inorg = pool.erode_P(eff_erodedP)                                     # :2050
    # print "weird nutrients" if abs(inorg + org - eff_erodedP) > FA            # :2051-2054
    ie.org-phosphorus += surface_P * org / eff_erodedP; ss... sub_P; perc... perc_P     # :2057-2065
    ie.phosphate += surface_P * inorg / eff_erodedP; ss... ; perc...                    # :2069-2077
    in_.phosphate = inorg;  in_.org-phosphorus = org                                    # :2082-2083
ie.solids += surface_sed; ss.solids += sub_sed; perc.solids += perc_sed       # :2088-2090
in_.solids = surface_sed + sub_sed + perc_sed                                  # :2092
return (in_, empty)
```

`erode_P(amount)` (`np.py:379-394`): it computes `fa = ads.P / (ads.P + humus.P)`, then
extracts `{N: 0, P: amount · fa}` from `ads` and `{N: 0, P: amount · (1 − fa)}` from `humus`,
and returns `(humus P removed, adsorbed P removed)`. The request dict also extracts `N = 0`
from each pool.

- The day of year in the `sin` is the raw one. That `sin` is a Python float computed at build
  time (rule 2).
- `sin(slope / 100)` and the `log10` use `sin_python` and `log10_python`, and the two powers use
  `pow_python`.
- The eroded masses are added to the day's outflow VQIPs, after `ihacres` and before `route`,
  and listed as inflows. They never pass through the tank.
- `"weird nutrients"` is reachable: the pools may hold less than the requested P. noodl records
  it as the event `"erosion_pool_short"`, one for each WSIMOD print.
- Erosion reads and writes `solids`. Without a `solids` pollutant, WSIMOD raises `KeyError` on
  the first day with `erodingflow > 0`, so noodl refuses that configuration by name (Interaction
  › Validation).
- `eff_erodedP > 0` implies `erodableP > 0`, so `erode_P`'s denominator is positive.

### `denitrification` (`land.py:2096-2152`)

```
if smc > field_capacity_m:                   f = 1                                       # :2107-2108
elif smc / field_capacity_m > limpar:        f = ((smc / field_capacity_m - limpar) / (1 - limpar)) ** exppar
else:                                        return (empty, empty)                       # :2114-2116
din_conc = DI.N / V * KG_M3_TO_MG_L                                                      # :2119-2123
h = din_conc / (din_conc + hsatINs)                                                      # :2124-2126
dN = DI.N * h * f * pool.temperature_dependence_factor * denpar                         # :2129-2135
dN = DI.extract({N: dN, P: 0})                                                           # :2136-2140
out_.nitrate = dN.N;  out_ = pull_pollutants(out_);  return (empty, out_)              # :2144-2152
```

The denitrified nitrogen leaves the model as nitrate (`:2142-2145`). The clamp and its event are
as for the uptake.

### `adsorption` (`land.py:2154-2265`)

```
limit = adosorption_nr_limit
pool_ = (ads.P + DI.P) / (area * M2_TO_KM2)                                  # :2171-2175 [kg/km2]
if pool_ == 0: return (empty, empty)                                          # :2176-2177
smc_mm = smc * M_TO_MM                                                        # :2180-2182
conc_sol = ads.P * KG_TO_MG / (bulk_density * rooting_depth * area)           # :2183-2187
coeff = kfr * bulk_density * rooting_depth                                    # :2188
if conc_sol <= 0:                                                             # :2191
    # prints "Warning: soil partP <=0. Freundlich will give error, take shortcut."   :2193
    xn_1 = pool_ / (smc_mm + coeff);  equi = kfr * xn_1                       # :2194-2195
else:
    x0 = exp((log(conc_sol) - log(kfr)) / nfr)                                # :2198-2200
    fxn = x0 * smc_mm + coeff * (x0 ** nfr) - pool_                           # :2201
    xn = x0; xn_1 = xn; j = 0
    while abs(fxn) > limit and j < adsorption_nr_maxiter:                     # :2205-2207
        fxn = xn * smc_mm + coeff * (xn ** nfr) - pool_                       # :2208
        fprimxn = smc_mm + nfr * coeff * (xn ** (nfr - 1))                    # :2209-2211
        dx = fxn / fprimxn                                                    # :2212
        if abs(dx) < (0.000001 * xn): break                                   # :2213-2215
        xn_1 = xn - dx
        if xn_1 <= 0: xn_1 = 1e-10                                            # :2216-2218
        xn = xn_1; j += 1                                                     # :2219-2220
    equi = kfr * (xn_1 ** nfr)                                                # :2221
if abs(equi - conc_sol) > 1e-6:                                               # :2225
    adsdes = (equi - conc_sol) * (1 - exp(-kadsdes))                          # :2230-2232
    request.P = adsdes * bulk_density * rooting_depth * (area * M2_TO_KM2)    # :2233-2238
    if request.P > 0:                                                         # :2239 adsorption
        adsorbed = DI.extract(request); ads.receive(adsorbed)                 # :2241-2244
        out_.phosphate = adsorbed.P; out_ = pull_pollutants(out_)             # :2247-2250
    else:                                                                     # desorption
        request.P = -request.P; desorbed = ads.extract(request); DI.receive(desorbed)   # :2255-2259
        in_.phosphate = desorbed.P; push_storage(in_, force=True)             # :2262-2263
return (in_, out_)
```

Here `request = {N: 0, P: …}` (`:2226`), so each extract and receive also moves `N = 0`.

- **The convergence test lags one iteration.** The `while` condition reads the residual computed
  at the top of the previous pass, which is that of the iterate before the last update. A loop
  the residual ends therefore takes one more step than the residual needs, and runs two passes
  at least, unless `f(x0)` is already within the limit, when no pass runs and `xn_1 = x0` (a
  soil near equilibrium; corrected while fixing Task 8b, W2). Transcribed.
- **The shortcut is common.** The adsorbed pool starts empty unless `initial_soil_storage` or
  dry phosphorus deposition fills it. The shortcut therefore prints on day 0 of an ordinary
  growing surface (run while writing this spec: it printed on day 0 of a 3-day run, and the
  adsorbed P was 0.059 afterwards). noodl records `"adsorption_shortcut"`, one for each print.
- **Two warnings are unreachable.** `"Warning: freundlich flow adjusted, was larger than pool"`
  is tested as `adsorbed − request > FA` and `desorbed − request > FA` (`:2242`, `:2257`).
  Because `extract` returns `min(storage, request)`, both differences are ≤ 0, and neither
  warning can print. noodl records the evident intent, `request − extracted > FA`, as the event
  `"freundlich_flow_adjusted"` (Y16).
- Parameter domains: `log(kfr)` needs `kfr > 0`, and `/ nfr` needs `nfr ≠ 0`, with `fprimxn`
  positive for `nfr > 0`. The shortcut's denominator `smc_mm + coeff` is positive when `coeff >
  0`. noodl refuses `kfr ≤ 0`, `nfr ≤ 0`, `bulk_density ≤ 0` and `rooting_depth ≤ 0` (Interaction
  › Validation).

### `IrrigationSurface.irrigate` (`land.py:2347-2381`)

```
if days_after_sow:                                                            # :2350
    demand = max(evaporation.volume - precipitation.volume, 0) * irrigation_coefficient   # :2352-2355
    if demand > FA:                                                           # :2356
        if cmd <= FA: demand = 0                                              # :2357-2360
        supplied = parent.pull_distributed({"volume": demand},
                       of_type=["River", "Node", "Groundwater", "Reservoir"])  # :2363-2366
        push_storage(supplied, force=True)                                    # :2369
        organic   = {N: supplied.org-nitrogen, P: supplied.org-phosphorus}    # :2372-2375
        inorganic = {N: supplied.ammonia + supplied.nitrate, P: supplied.phosphate}   # :2376-2379
        DO.receive(organic); DI.receive(inorganic)                            # :2380-2381
```

- `evaporation` and `precipitation` are the surface's flows from its last `ihacres`. In the
  default orchestration that is the same day's `Land.run`; in another orchestration it can be
  the previous day's, so they are carried (State).
- A demand set to 0 by the saturated-soil test still makes a zero-volume pull (`:2363`), which
  3a's distribute transcribes.
- The irrigation water arrives through the `Land`'s in-arcs, so 3a's node balance counts it.
  `irrigate`'s return value is never summed (`:197-198`).
- **The pool update reads five pollutant keys unconditionally.** Without `org-nitrogen` (or any
  of the five), WSIMOD raises `KeyError` on the first day with a demand (run while writing this
  spec: `KeyError: 'org-nitrogen'` at `land.py:2373`, pollutants `phosphate` and `temperature`).
  Every other pool access in `land.py` sits behind the `nitrate` switch. noodl does the evident
  intent and updates the pool only when `nitrate` is a pollutant, that is, when the pool is live
  (Y20).
- The irrigated nitrite enters the tank and not the pool, which is another pool–tank gap.

### `GardenSurface` and the `Demand` garden link (`land.py:2396-2437`, `demand.py:78-117`, `:250-301`)

When a `Land` has `GardenSurface`s, the **last** one's methods become the `Land`'s
`("Demand", "Garden")` push check and push set (`land.py:120-128`, overwriting `:93-94`):

```
calculate_irrigation_demand(ignore_vqip=None):                                 # :2396-2423
    demand = max(evaporation.volume - precipitation.volume, 0)                 # :2411-2413
    if cmd <= FA: demand = 0                                                   # :2415-2418
    reply = empty; reply.volume = demand; return reply
receive_irrigation_demand(vqip):  return push_storage(vqip, force=True)       # :2425-2437
```

`ResidentialDemand.create_demand` (slot 2, before `Land.run`) asks for it:

1. `get_garden_demand` (`demand.py:250-271`) is `get_connected(push, of_type="Land",
   tag=("Demand", "Garden"))["avail"]`, times `gardening_efficiency`;
2. that volume is pulled from the distribution with the house demand (`:85-90`);
3. it is pushed to the `Land` with the garden tag (`:106-113`).

Consequences, all transcribed:

- **The demand is yesterday's.** `create_demand` runs before `Land.run`, so `evaporation` and
  `precipitation` are the previous day's flows, and 0 on day 0 (run while writing this spec: the
  check answered 0 on days 0 and 1, then 40.0, and the push carried 16.8 = 40 × 0.42). They are
  carried state (State).
- **The garden water is at 0 °C.** `apply_gardening_pollutants` builds the push from
  `empty_vqip()` (`demand.py:273-287`), so its temperature is 0. Blended into the soil store, it
  cools it (run while writing this spec: a 0 °C push into a soil at 11.5 °C). The pulled water's
  quality stays at the `Demand` (3a). Reproduced and documented (Y29).
- The garden water carries no nutrients, so `receive_irrigation_demand`, which bypasses the
  pool, opens no gap.
- **Only the last garden surface is irrigated.** With two or more, the earlier ones' handlers
  are replaced (run while writing this spec: with `g1` and `g2` the check was bound to `g2`).
  They still grow and drain as growing surfaces. Reproduced, documented and listed in
  `model.unread` (Y28).
- A `NonResidentialDemand` asks for no garden water (3a).

### `VariableAreaSurface.get_climate_` (`land.py:2443-2461`)

`__init__` replaces `get_climate` with an instance attribute (`:2446`), which `ihacres` calls
(`:934`):

```
P = precipitation;  E = et0 * et0_coefficient                       # :2455-2456 (crop-adjusted)
P *= current_surface_area / area;  E *= current_surface_area / area # :2458-2459
```

- `current_surface_area` is configured in the constructor (`:2443`), default 0, and is a
  driver; it is refused as an override (no `apply_overrides` consumes it). A default
  variable-area surface receives no rain and no evaporation (run while writing this spec: its precipitation
  volume was 0.0).
- Erosion still reads the unscaled precipitation (`:1940`). Transcribed.

## State (Y5)

New carried state, in 3a's dict-of-tensors layout (each class declares its `carried_paths`;
rows in the state layout's order, `Land`s in configuration order and surfaces in order, D7):

| Key | Shape | What it holds (WSIMOD attribute) | Labels, unit |
|---|---|---|---|
| `"wsimod.nutrient_pool"` | (..., 10 · n_growing), flat | each growing surface's five `NutrientStore.storage`s (`np.py:125-136`), present only when `nitrate` is a configured pollutant and pollutants are on; the engine's pool tensor stays `(..., 5, 2)`, and the state holds it as 10 labelled rows (the slot layout of `arc_queue_since` and the River lags) | rows `(land, "surfaces.<name>.nutrient_pool", <pool>, <N\|P>)`, pools in the order `fast_pool`, `humus_pool`, `dissolved_inorganic_pool`, `dissolved_organic_pool`, `adsorbed_inorganic_pool`, each with `N`, then `P`; `kg` |
| `"wsimod.days_after_sow"` | (..., n_growing) | `days_after_sow` (`land.py:1261`, `:1531-1544`), with **−1 for `None`** | rows `(land, "surfaces.<name>")`; `d` |
| `"wsimod.crop_cover"`, `"wsimod.ground_cover"` | (..., n_growing) | `crop_cover`, `ground_cover` (`:1262-1263`), which persist outside the season | as above; `-` |
| `"wsimod.surface_climate"` | (..., 2 · n_climate), flat | the `evaporation` and `precipitation` **volumes** of every `GardenSurface` and `IrrigationSurface` (`:1029-1044`, read by `:2352`, `:2411`) | rows `(land, "surfaces.<name>", "evaporation")` and `(land, "surfaces.<name>", "precipitation")`; `m3` |

- `n_growing` counts every surface of class `GrowingSurface` or a subclass. The crop keys exist
  in quantity-only runs too, because the calendar drives the crop's evaporation factor and so
  the volumes.
- `evaporation` and `precipitation` are `v_change_vqip(empty, volume)` (`:1029-1031`), with
  zero pollutant columns. Carrying the volume loses nothing, and `load_stores` rebuilds the
  VQIPs.
- A growing surface's tank is a surface row of `"wsimod.storage"` (3a), and its
  `total_decayed` is carried under 3b's rule (when `nitrite` is a pollutant).
- **Not carried:**
  - `crop_factor` and `et0_coefficient` are recomputed by `calc_crop_cover` before every
    reader. They are recorded per day: the results' `et0_coef`, as WSIMOD's `record_all`
    records it (`model.py:1150`), and `crop_factor` as a noodl field (Y32);
  - the pool's two factors are the day's;
  - `infiltration_excess`, `subsurface_flow` and `percolation` are written by `ihacres` before
    every reader;
  - the `running_*_mb` stay the day's (3a).
- **Initial state.**
  - The pools come from the constructor rule above: the initial storage, then
    `initial_soil_storage`.
  - `days_after_sow` starts at −1, and both covers start at 0.
  - The climate starts at 0, which gives WSIMOD's zero garden demand on day 0.
  - A run can be warm-started from another run's state.
- The pools and the crop and climate registers are not water held, so the mass balance skips
  them and they join `_FLAT_KEYS` where flat (Y11).
- **3c adds nothing to `end_timestep`.** The engine's rule is that `end_timestep` calls no
  `ctx.decide` or `ctx.call` (`day.py:1216`). Every new register is written during `Land.run`,
  `apply_irrigation` or the garden handlers (the harvest's reset of `days_after_sow` included,
  in `calc_crop_cover`); the only end-of-day work on a growing surface is 3b's decay (Y50).

## Inputs

**Parameters** (3a's convention `"wsimod.<Class>.<parameter>"`; rows are every surface of
exactly that class, in state order, under the name `surfaces.<name>.<parameter>` on its
`Land`, as 3a's surface parameters are bound):

| Driver | Shape | Notes |
|---|---|---|
| `"wsimod.<Class>.<p>"` for the 25 real attributes of the table above (`adsorption_nr_maxiter` excluded), and `ET_depletion_factor`, `rooting_depth` | (..., n) | `Class` is `GrowingSurface`, `IrrigationSurface`, `GardenSurface` or `VariableAreaSurface`. `rooting_depth` re-derives the capacity, `field_capacity_m`, `wilting_point_m`, `TAW` and `RAW` in `bind`, as 3a's surfaces re-derive their capacity |
| `"wsimod.<Class>.crop_factor_stages"` | (..., n, S) | `S` is the class's longest `crop_factor_stages`. A shorter surface's padding is inert: its template is 0, and `bind` refuses any other value (Y33, X9's pattern) |
| `"wsimod.<Class>.nutrient_pool.<p>"` for `degrhpar`, `dishpar`, `minfpar`, `disfpar`, `immobdpar`, `fraction_manure_to_dissolved_inorganic` | (..., n, 2) | columns `N`, `P`; WSIMOD's defaults as written (`np.py:16-23`): `7 * 1e-5`, `7 * 1e-6`, …; no configuration route (Y24). The derived `1 − fraction` values are computed in `bind` as `calculate_fraction_parameters` does (`np.py:146-153`) |
| `"wsimod.<Class>.nutrient_pool.fraction_dry_n_to_dissolved_inorganic"` | (..., n) | default 0.9 |
| `"wsimod.IrrigationSurface.irrigation_coefficient"` | (..., n) | default 0.1 (`land.py:2317`) |
| `"wsimod.VariableAreaSurface.current_surface_area"` | (..., n) | default 0 (`:2443`), `m2` |
| `"wsimod.<Class>.pollutant_load"` (3a) | (..., n, K) | now live for every surface class. Absent in a quantity-only run |

The nutrient-pool drivers are present only with `nitrate` and pollutants on. The other growing
parameters are present in both modes, but a quantity-only run reads only `ET_depletion_factor`,
`rooting_depth`, `crop_factor_stages`, `irrigation_coefficient` and `current_surface_area`; the
rest are listed in `model.ignored_by_quantity_only`.

**Forcing.** A surface's data become daily drivers `"wsimod.<variable>"` `(..., T,
n_readers)`, one per variable, with rows `(land, "surfaces.<name>")`. They are filled from the
monthly table, so every day of a month gets that month's value (Y23). The variables:

- `nhx-dry`, `noy-dry`, `srp-dry`, `nhx-wet`, `noy-wet`, `srp-wet`: read by every surface whose
  constructor data has `nhx-dry`;
- `nhx-fertiliser`, `noy-fertiliser`, `srp-fertiliser`, `nhx-manure`, `noy-manure`,
  `srp-manure`: read by every growing surface when `nitrate` is in the network's pollutant
  list. The check uses the network's list in both modes: a quantity-only run of a network that
  lists `nitrate` must carry complete fertiliser and manure data, because the equivalent WSIMOD
  run's `POLLUTANTS` holds `nitrate` and reads them (Y49).

All are in `kg/m2/d` (WSIMOD's docstrings, `land.py:354-359`, `:1167-1172`, "kg/m2/timestep").
The build-time forcing check refuses a missing `(variable, first of the month)` by name. As in
3a (D9), it applies in quantity-only runs too, because WSIMOD reads these values whatever the
run mode; there the values are listed as ignored. A batched driver may vary within a month,
which WSIMOD cannot express, and the docs say so.

**Learnable.** Every real-valued driver above, and these forcing drivers. Gradients reach:

- the volumes through `crop_factor_stages`, `ET_depletion_factor`, `rooting_depth`,
  `irrigation_coefficient` and `current_surface_area`;
- the pollutants through everything.

`adosorption_nr_limit` is a driver whose only effect is on decisions, so its gradient is 0.

**Structural** (cannot be batched or learnt): the surface classes; `sowing_day`, `harvest_day`
(integers; an integral float is accepted, as `==` with an `int` accepts it);
`crop_factor_stage_dates` and the length of `crop_factor_stages`; `adsorption_nr_maxiter`; the
constructor's data variables and `pollutant_load` keys; `initial_soil_storage` (it sets the
initial state, which the state dict then carries); the pollutant lists.

## Mass balance (Y11)

The boundary is WSIMOD's. The water held is the tanks, surfaces included; the nutrient pools'
solid stores (fast, humus, adsorbed) and the dissolved pools' own bookkeeping are outside it.
WSIMOD's surface lists close on this boundary, because every process lists exactly what the
tank (or an outflow) gained or lost:

- the dissolved part of fertiliser, manure and deposition;
- the transformation's `in_` and `out_`, with `out_` clamped by `pull_pollutants`;
- uptake, denitrification and adsorption `out_`, clamped;
- desorption `in_`;
- erosion's `in_`, added straight to the outflows.

Running WSIMOD while writing this spec, its node check on a growing `Land` printed only
round-off "mass balance error for org-phosphorus/org-nitrogen of ~1e-17" lines.

noodl's balance keeps 3a's aggregate terms `"Land.running_inflow_mb"` (in) and
`"Land.running_outflow_mb"` (out). They now contain every 3c process, and they are WSIMOD's lists
bitwise. noodl also records a **breakdown** of them, `WsimodMassBalance.parts`, which is not
summed into the balance, with one `(..., T, 1 + K_add)` entry per function and direction:

| Part | Dir. | Source |
|---|---|---|
| `"Surface.simple_deposition"` | in | `land.py:526` |
| `"Surface.atmospheric_deposition"`, `"Surface.precipitation_deposition"` | in | `:554`, `:581`, the dissolved part for a growing surface |
| `"GrowingSurface.fertiliser"`, `"GrowingSurface.manure"` | in | `:1679`, `:1710` |
| `"GrowingSurface.soil_pool_transformation"` | in and out | `:1811` |
| `"GrowingSurface.calc_crop_uptake"` | out | `:1924` |
| `"GrowingSurface.erosion"` | in | `:2094` |
| `"GrowingSurface.denitrification"` | out | `:2152` |
| `"GrowingSurface.adsorption"` | in and out | `:2265` |
| `"Surface.nitrite_carry_over"` (3b) | in | `:434-439` |
| `"PerviousSurface.ihacres"`, `"ImperviousSurface.precipitation_evaporation"` (3a) | in, out | precipitation in, evaporation out |

A test asserts that the parts of each direction sum to the aggregate within 1e-12 × scale. The
sums differ only by summation order: WSIMOD adds sequentially (`land.py:444-449`), and the parts
are summed per function first. Irrigation and garden water cross arcs, which 3a already counts.
The 3b term `"DecayTank.decayed"` covers a growing surface's own decay. Every 3c process changes
no volume, except that irrigation and garden water move water between nodes through arcs, so the
volume balance is 3a's.

## Interaction with 3a and 3b

### Classes and modules

- `land.py`:
  - `Surface` gets the live `simple_deposition`, `atmospheric_deposition`,
    `precipitation_deposition` and `*_deposition_to_tank`. Its structural flags come from the
    constructor configuration, not the merged one (Y21).
  - `Land` gets `apply_irrigation` and the garden handlers.
  - `SURFACE_CLASSES` gains the four growing classes.
- A new `growing.py` holds `GrowingSurface(PerviousSurface)`, `IrrigationSurface`,
  `GardenSurface` and `VariableAreaSurface`.
- A new `nutrient_pool.py` holds the batched `NutrientPool`: its `(..., 5, 2)` storage and the
  transcribed methods. Each method takes the caller's `active` mask, as the tanks do.
- `tanks.Tank` gains `pull_pollutants`.
- `vqip.py` gains `log_python`, `log10_python` and `sin_python`.
- `classes.py`:
  - `_CONSTRUCTORS` and `_OVERRIDES` gain the four classes. `GrowingSurface`'s overrides are the
    31 names of `land.py:1376-1408` plus `rooting_depth`; `IrrigationSurface`'s is
    `irrigation_coefficient`.
  - `_NOT_OVERRIDES["GrowingSurface"]` gets `depth`, as for `ImperviousSurface` (`:1412-1417`
    pops it with an "ERROR" print), and `_DERIVED` gets `depth`.
  - `"Surface"`'s overrides gain `data_input_dict`.
  - `OUT_OF_SCOPE` loses the four.
- `config.py`:
  - the surface `data_input_dict` override (Overrides below);
  - the surface data variables accepted (Validation);
  - `SurfaceSpec` keeps the constructor's data variables and `pollutant_load` keys, as
    `constructed_variables` and `constructed_load`.
- `day.py` gets the state keys, the drivers and the forcing rows; `run.py` gets the parts;
  `structure.forcing_readers` gets the surface readers.
- `storage.River.pull_check_fp`'s message (Y10).

### Validation

New refusals, each by name with a test:

- **Pollutants.**
  - With `nitrate` a pollutant and a growing surface present, the run needs all of these, the
    first seven additive: `ammonia`, `nitrite`, `nitrate`, `phosphate`, `org-nitrogen`,
    `org-phosphorus`, `solids` and `temperature`. WSIMOD raises `KeyError` without them
    (`:1294-1307`, `:1465`, `:2088`, `:1822`). Its docstring names five of them and omits
    `solids` (`:1119-1121`).
  - A non-additive `ammonia`, `nitrate` or `phosphate` with deposition configured is **not**
    refused: WSIMOD runs it. The value blends into the tank at volume 0, which changes it only by
    rounding, as Y38's non-additive `pollutant_load` does (`core.py:104-114`), while on a growing
    surface the pool still receives the N and P (a pool–tank gap by construction). Transcribed
    and documented (Y43).
- **Surface data.**
  - The accepted variables are the six deposition ones on every surface class, plus, on a growing
    surface, the six fertiliser and manure ones. Any other variable is refused (3a's D21
    rule). `*-residue` is refused with its own reason (Y36).
  - `nhx-dry` without all five companions is refused.
  - A forcing date missing for a variable that is read is refused.
- **`GrowingSurface`.**
  - Its configuration may not carry `depth` (`TypeError`), a float `initial_soil_storage`
    (`TypeError`), or an `initial_soil_storage` mapping that lacks a key (`KeyError`).
  - `crop_factor_stages` and `crop_factor_stage_dates` must have the same length. Any date whose
    interpolation would raise `IndexError` or divide by zero is refused: the run's dates at
    build, any later date when `model.step` reaches it (Y47).
  - `sowing_day` and `harvest_day` must be integral.
  - `adsorption_nr_maxiter` must be a non-negative integer.
  - `wilting_point ≥ field_capacity` (`TAW ≤ 0`) is refused by name. `TAW < 0` is stricter
    than WSIMOD's print; `TAW = 0` crashes WSIMOD on day 0 (Y27, Y44).
  - The domain refusals below are checked at build and again at `bind`. With `nitrate`, each
    stands for a WSIMOD crash:
    - `rooting_depth ≤ 0`, `bulk_density ≤ 0`, `kfr ≤ 0` and `nfr ≤ 0` (adsorption and erosion
      divide or take logs);
    - `cohesion = 0`, `thetaupp = 0` and `thetalow = 0` (divisions);
    - `hsatINs ≤ 0` (`din_conc + hsatINs` can be 0);
    - `limpar < 0`, with which a dry soil reaches `DI.N / 0`;
    - per instance, `kadsdes < -709.782712893384` (`math.log(sys.float_info.max)`), where
      `math.exp(-kadsdes)` raises `OverflowError` on the first day adsorption moves P
      (`:2230-2232`); a negative `kadsdes` within it runs, as in WSIMOD;
    - `field_capacity ≤ 0` (`smc / field_capacity_m`);
    - per instance, `thetaupp < 0` or `thetalow < 0` where `thetapow` is not integral: the
      soil-moisture factor's base is then negative, Python's `**` is complex and WSIMOD's `min`
      raises `TypeError` (`land.py:1849-1853`). With an integral `thetapow` WSIMOD runs, and
      noodl matches it bitwise, so that case is not refused (Task 7 review, finding 2).
  - In both modes, the `ET_depletion_factor = 1` stress case is not refused: it is the evident
    intent (Y17).
- **`IrrigationSurface`:** none beyond its parent's. The missing-pollutant crash is avoided
  (Y20).
- **Override refusals:**
  - a node's `data_input_dict` override;
  - a surface's `filename` override;
  - a surface `data_input_dict` override that drops `nhx-dry` from a surface constructed with it
    (WSIMOD keeps the deposition functions and raises `KeyError` at the first run);
  - an override that drops fertiliser or manure data from a growing surface;
  - an override naming a `NutrientPool` parameter.

### Loading and overrides

- **Surface overrides** follow WSIMOD's chain:
  - `GrowingSurface.apply_overrides` (`:1376-1431`) sets the 31 attributes and pops
    `rooting_depth`. It sets `overrides["depth"] = rooting_depth` and calls
    `PerviousSurface.apply_overrides`.
  - Unlike a `PerviousSurface`'s, a growing surface's derived values after an override equal
    construction's bitwise. `Surface.apply_overrides` sets the physical depth to
    `rooting_depth` (`:400`), then `field_capacity_m = field_capacity · depth` and `depth *=
    total_porosity` (`:881-884`), the constructor's expressions.
  - The calendar, `TAW` and `RAW` are recomputed (`:1422-1431`); the pool is not.
- **`data_input_dict` override on a surface (Y22).** It **replaces** the surface's data
  (`:413-421`), as a mapping or as a path:
  - a mapping in noodl's inline `(variable, date)` form, refused if it is not one;
  - a path, read by `read_csv` (`model.py:1252-1269`) as written, relative to the process's
    working directory, as X27's `filename` is.

  `None` or another type raises `ValueError` in WSIMOD (`:419-421`), and noodl refuses it.
- **Constructor flags stay (Y21).** The inflow list is fixed at construction (`:375-382`), and an
  override does not rebuild it:
  - a `data_input_dict` override that adds `nhx-dry` to a surface built without it is read and
    never used. Its deposition variables are listed in `model.unread`;
  - a `pollutant_load` override on a surface built without a `pollutant_load` is **ignored**.
    Run while writing this spec: the surface's inflows stayed `['ihacres']`. noodl lists it in
    `model.unread`.
  - WSIMOD's override on such a surface also mutates the shared default `{}`
    (`land.py:312`, `:397`). Every surface built afterwards in the same process without a load
    then gets that load. Run while writing this spec: the default became `{'phosphate': 1.0}`,
    shared by the other surface. noodl's loads are per surface, and the harness's
    `_snapshot_constructor_defaults` and `_restore_constructor_defaults` keep the WSIMOD
    reference clean.
  - A `pollutant_load` override on a surface built with one merges per key (`:397`) and is
    live.
- `irrigation_coefficient` is an `IrrigationSurface` override (`:2342-2345`).
  `current_surface_area` is not an override of anything, so it is refused as one.
- The 3a rule stands: an override key the class does not consume is refused by name.

### Decision vocabulary, events and diagnostics

Sites follow 3a's grammar `"<node|arc>:<name>/<attribute path>/<WSIMOD method>/<branch>"`. The
attribute path of a surface is `surfaces.<s>`, and of its pools
`surfaces.<s>.nutrient_pool.<pool>`. `call` is the static index of the method's call that day.
In the list below, `S` is `node:N/surfaces.<s>`.

- **Pools:** `S.nutrient_pool.<pool>/extract/clip:<N|P>`, each `min(storage, amount)`
  (`np.py:553`).
- **Tank:** `S/pull_pollutants/clip:<volume|pollutant>`, each `min` (`tanks.py:262`).
- **`calc_crop_cover`:**
  - `S/calc_crop_cover/unsown` (`:1531`, `days_after_sow` is −1);
  - `S/calc_crop_cover/crop` (`:1550`, `days_after_sow > 0`);
  - `S/calc_crop_cover/depletion` (`:1560`, the `max`);
  - `S/calc_crop_cover/unstressed` (`:1561`);
  - `S/calc_crop_cover/stress_floor` (`:1564`, the `max(0, ·)`);
  - `S/calc_crop_cover/stress_zero_denominator` (noodl's guard, Y17).

  The leap, sowing and harvest tests are structural and record nothing.
- **`pull_storage`:** `S/pull_storage/empty` (`:1446`), `S/pull_storage/clip` (`:1450`) and
  `S/pull_storage/no_din` (noodl's guard, Y18; event `"pull_storage_no_din"`).
- **`adjust_vqip_to_liquid`:** `S/adjust_vqip_to_liquid/N` and `…/P` (`:1595`, `:1599`), with
  `call` counting the dry deposition, wet deposition, fertiliser and manure calls, in that
  (call) order.
- **`soil_pool_transformation`:**
  - `S/soil_pool_transformation/no_din` (Y18; event `"transformation_no_din"`);
  - `S/soil_pool_transformation/inorganic_N`, `…/organic_N`, `…/inorganic_P` and `…/organic_P`
    (`:1770`, `:1783`, `:1790`, `:1797`).
- **`calc_temperature_dependence_factor`:** `S/calc_temperature_dependence_factor/warm` and
  `…/thawed` (`:1822`, `:1826`).
- **`calc_soil_moisture_dependence_factor`:** `S/calc_soil_moisture_dependence_factor/saturated`
  and `…/wilted` (`:1843`, `:1845`), and `…/min:wp` and `…/min:fc` (`:1853`, the three-way `min`
  as two margins in Python's order).
- **`calc_crop_uptake`:**
  - `S/calc_crop_uptake/crop` (`:1872`);
  - `…/temperature_cap` and `…/temperature_floor` (`:1878`);
  - `…/shape` (`:1887`);
  - `…/available_water` (`:1899`, the `max`) and `…/dry_soil` (Y18);
  - `…/cap:P` and `…/cap:N` (`:1910-1911`).
- **`erosion`:**
  - `S/erosion/rain` (`:1943`) and `…/runoff` (`:1959`);
  - `…/transport` (`:1984`);
  - `…/enrichment_cap` (`:1992`) and `…/flow` (`:1994`);
  - `…/phosphorus` (`:2048`).
- **`denitrification`:** `S/denitrification/saturated` (`:2107`) and `…/moist` (`:2109`).
- **`adsorption`:**
  - `S/adsorption/empty` (`:2176`) and `…/shortcut` (`:2191`);
  - per iteration `j`: `…/newton:<j>/residual` (`:2206`), `…/newton:<j>/step` (`:2213`) and
    `…/newton:<j>/positive` (`:2217`);
  - `…/change` (`:2225`) and `…/adsorb` (`:2239`).
- **`irrigate`:**
  - `S/irrigate/crop` (`:2350`);
  - `…/deficit` (`:2353`, the `max`);
  - `…/demand` (`:2356`) and `…/saturated` (`:2358`).

  The pull is 3a's distribute, with its sites.
- **Garden:** `S/calculate_irrigation_demand/deficit` (`:2411`) and `…/saturated` (`:2416`).

Every branch after a guard is evaluated under the mask that the guard leaves.

**Events** (D13), one per occurrence:

| Event | Where | WSIMOD |
|---|---|---|
| `"adsorption_shortcut"` | `conc_sol ≤ 0` (`:2191-2195`) | prints "Warning: soil partP <=0…" |
| `"erosion_pool_short"` | `abs(inorg + org − eff) > FA` (`:2053`) | prints "weird nutrients" |
| `"nutrient_pool_not_tracking"` | a `pull_pollutants` clamp removes more than `FA` (`:1806`, `:1920`, `:2148`, `:2250`) | its check is dead (defect table) |
| `"freundlich_flow_adjusted"` | `request − extracted > FA` (`:2241`, `:2256`) | its test is reversed, never prints |
| `"crop_water_stress_zero_denominator"` | Y17 | `ZeroDivisionError` |
| `"pull_storage_no_din"` | Y18 (`:1459-1485`): the species set to 0 | `ZeroDivisionError` |
| `"transformation_no_din"` | Y18 (`:1758`): the proportion set to 0 | `ZeroDivisionError` |
| `"crop_uptake_dry_soil"` | Y18 (`:1898-1907`) | `ZeroDivisionError` |

The print check (W90/W91) maps the first two events one to one onto WSIMOD's prints.
"warning: TAW < 0" and the `depth` "ERROR" are never printed, because noodl refuses those
configurations. One named admission is measured and asserted (Y35, Y41): WSIMOD's own node
check prints round-off lines of exactly the form "mass balance error for <additive> of x in
<land>" on lands that hold a growing surface (run while writing this spec: |x| ≈ 1e-17). They
print when a tank holds a large organic store and the day's flows are small, because WSIMOD's
`ds_` is a difference of two large stores while its normaliser `magnitude =
10**int(log10(largest))` (`core.py:405-411`) is the flow scale. `|x| / magnitude` therefore has
no useful bound (the review measured 1.7e-9). The admission is bounded by the store instead:

- per case and per pollutant, `|x| ≤ c · ε · S`, where `ε = 2.220446049250313e-16` and `S` is
  the largest store of that pollutant on the land's surface tanks that day (start or end);
- `c` is measured in the plan and stated in the test. The review measured `|x| ≈ 3e-16 × S`
  (1.6e-16 at 0.5 kg, 1.7e-13 at 500 kg), so `c ≈ 1.4` there; the plan sets `c` to the
  measured maximum over the cases, rounded up to the next integer;
- a case where no such line is measured asserts **zero** such lines (the review's 420-day
  draining case without `initial_soil_storage` printed none);
- every other print stays a failure.

noodl's residual closes within 1e-12 × scale.

### The replay harness

`capture_node_calls` (`tests/verification/_wsimod_reference.py`) also captures:

- each growing surface's five pool storages (`pad`-free, `(5, 2)`), its `days_after_sow`
  (`None` → −1), `crop_cover`, `ground_cover`, `crop_factor` and `et0_coefficient`, and the pool's
  two factors, before and after every `Land.run`;
- `Land.apply_irrigation` and the `("Demand", "Garden")` handler calls, with their replies;
- the `Land`'s `t` and `monthyear`, and the surface data value read on that day.

`_store_paths` and `_wsimod_store` gain the paths `surfaces.<s>.nutrient_pool`,
`.days_after_sow`, `.crop_cover`, `.ground_cover` and `.surface_climate`. The bound is 0
(bitwise) for every 3c class. A replay that cannot be bitwise is measured, its cause written in
the test, and its bound asserted (W19); none is expected.

No case patches WSIMOD. The crash cases (Verification › 3) run WSIMOD as it is, assert its
exception, and assert noodl's result and event.

### The demos (Y3, Y30)

Both demos run their unmodified `config.yml` with pollutants on and every process live:

- quickstart: `simple_deposition` of phosphate on its impervious and pervious surfaces;
- oxford: `simple_deposition` of 11 additive pollutants on its two surfaces.

Neither demo has deposition data, a growing surface, irrigation or a garden, and running WSIMOD
while writing this spec, neither printed anything with the loads live (quickstart in full,
oxford for 365 days). `config_no_deposition.yml` is retired, because nothing remains to
neutralise:

- delete both files;
- drop `_no_deposition_overrides` and `--no-deposition-only` from
  `scripts/regenerate_wsimod_fixtures.py`;
- move `test_wsimod_parity.py`, `test_wsimod_replay.py`, `test_wsimod_fixtures.py` and
  `test_wsimod_gradients.py` to `config.yml`;
- update `test_build.py:355-365` (quickstart with pollutants on now builds);
- update the README and the application page.

The 3a and 3b specs and plans keep the old name as history. The quantity-only run of
`config.yml` stays, and the test that quantity-only volumes equal the live volumes now compares
it with the live `config.yml` run.

## WSIMOD defects found in these files (Y4)

Rule as in 3a and 3b:

- where the intent is evident, noodl does it;
- otherwise noodl refuses by name or, for a state-dependent case, reproduces it and records an
  event.

Each row has a test. Nothing is filed upstream without the user (Y4); #258 and #259 exist, and
none of these is in them. 26 rows.

| # | Defect | WSIMOD | noodl | Test |
|---|---|---|---|---|
| L1 | `to_datetime.is_leap_year` is a method; `if t.is_leap_year:` tests the bound method (`land.py:1526`, `model.py:114-121`) | every year's day of year after 28 February is shifted by one; run while writing this spec | reproduced: the individual-files loader's path (`model.py:347-348`), the format noodl reads; the unified-data (parquet) loader uses pandas dates and shifts only in leap years (Y13, Y42) | `test_every_year_shifts_the_day_after_february`; `growing_spring` spans 2003–2005 |
| L2 | The calendar `[0, sow, harvest, harvest+1, 365]` is unsorted when `sowing_day > harvest_day` or `harvest_day = 365` (`:1335-1341`), and `bisect_left` needs a sorted list | autumn-sown crops have zero crop and ground cover all year; run while writing this spec. The default calendar is unsorted too, but benign | reproduced exactly (rule 2); documented (Y15) | `growing_autumn`; unit test over days 1–365 against `quick_interp` |
| L3 | `(1 − ET_depletion_factor) · TAW` divides (`:1564-1568`); the default `ET_depletion_factor = 1` makes it 0 | `ZeroDivisionError` once the soil reaches its wilting point; run while writing this spec (`:1566`) | evident intent `Ks = 0`; event (Y17) | `growing_default_depletion` (crash case) |
| L4 | `TAW < 0` only prints (`:1361-1362`); `TAW = 0` divides by zero | `Ks` can exceed 1; `ZeroDivisionError` on day 0 | `wilting_point ≥ field_capacity` refused by name (Y27, Y44) | refusal tests (`>` and `=`) |
| L5 | `pull_storage` divides by the tank's DIN (`:1459-1485`) | `ZeroDivisionError` when the tank has no inorganic N and water flows: any growing surface started from a zero-N tank with zero first-month inputs; run while writing this spec (`:1460`) | species 0; event `"pull_storage_no_din"` (Y18) | `growing_no_din` (crash case) |
| L6 | `soil_pool_transformation` divides by `nitrate + ammonia` (`:1758-1760`) | `ZeroDivisionError` with neither, on day 0 for a plain zero-N tank with zero first-month inputs; run while writing this spec (`:1758`) | proportion 0; event `"transformation_no_din"` (Y18) | `growing_zero_n` and `growing_nitrite_only` (crash cases) |
| L7 | `calc_crop_uptake` divides by the tank volume (`:1898-1907`) | `ZeroDivisionError` with crops and an empty soil, reachable only with `ET_depletion_factor < 1` (with 1, L3 crashes first); run while writing this spec (`:1899`) | available uptake 0; event (Y18) | `growing_empty_soil` (crash case) |
| L8 | `effective_precipitation_flushing` extracts the outflows' nutrients from the pools that `pull_storage` already extracted them from (`:1458`, `:1605-1645`) | the dissolved pools lose them twice; after 120 days tank P 1.158 against pool P 0.052, and equal without the flushing; run while writing this spec (the review: 0.2406 against 0.0506, both 0.05560 without the flushing) | reproduced, no switch (Y16, Y45): every growing-surface result depends on it, and the two fix forms differ (Y45); documented | `growing_spring`; `test_the_pools_are_flushed_twice` asserts that noodl's tank–pool gap equals WSIMOD's, not a hard-coded ratio |
| L9 | Uptake and denitrification take all N as nitrate (`:1917`, `:2145`); `simple_deposition`, the tank's decay (3b) and irrigated nitrite bypass the pool | the tank and the pool diverge further. With flushing off, DIN after 120 days was tank 15.5 against pool 0.18; run while writing this spec. Where the pool exceeds the tank, `pull_storage` can make a tank column negative (the mechanism is real; neither the review's 420-day case nor this spec's runs produced one) | reproduced; event `"nutrient_pool_not_tracking"` where the tank clamps | `growing_spring`, `growing_decay`; the test checks the sign of every tank column and states whether a negative one occurred |
| L10 | `Tank.pull_pollutants` mutates and returns its argument (`tanks.py:261-266`), so `compare_vqip(out_, out2_)` compares an object with itself (`:1808`, `:1921`, `:2149`, `:2251`) | "nutrient pool not tracking soil tank" can never print; run while writing this spec | the clamp is transcribed; the check's evident intent is the event (Y16) | `test_a_clamped_pull_records_the_event` |
| L11 | The Freundlich warnings test `extracted − request > FA` (`:2242`, `:2257`), which `min` makes ≤ 0 | never prints | the evident test `request − extracted > FA`; event (Y16) | unit test with a short pool |
| L12 | The soil temperature factor is `T/5` on `(0, 5]` (`:1826-1827`), not HYPE's `2^((T−20)/10) · T/5`, which WSIMOD's River uses (`storage.py:775-777`) | jumps from 1 to 0.354 at 5 °C, and is larger at 4.9 °C than at 15 °C | reproduced; documented (Y19) | unit test either side of 5 °C; `growing_spring` (cold spells) |
| L13 | `irrigate` reads five pollutant keys whatever the pollutants (`:2372-2381`) | `KeyError: 'org-nitrogen'` with a demand; run while writing this spec (`:2373`) | the pool update only with `nitrate` (Y20) | `irrigation_without_nutrients` (crash case) |
| L14 | A `pollutant_load` override on a surface built without one is ignored (the inflows were fixed at `:379-380`), and it mutates the shared default `{}` (`:312`, `:397`) | no load on that surface, and the load on every later surface built without one in the process; run while writing this spec | per surface; the override is listed in `model.unread` (Y21) | `surface_overrides`; `test_a_load_override_without_a_load_is_unread` |
| L15 | A `data_input_dict` override does not rebuild the inflows (`:375-382` against `:410-421`) | deposition data added by override are unread; removing `nhx-dry` by override raises `KeyError` at the first run | unread; the removal is refused (Y21) | `surface_overrides`; refusal test |
| L16 | `initial_soil_storage` as a float, which the docstring allows (`:1142`) | `TypeError` at `:1312`; run while writing this spec | refused (Y37) | refusal test |
| L17 | `depth` on a growing surface | `TypeError` (multiple values for `depth`) in the configuration; run while writing this spec. As an override it is popped with an "ERROR" print (`:1412-1417`) | refused in both (Y27) | refusal tests |
| L18 | Two or more `GardenSurface`s on a `Land` (`:120-128`) | only the last is irrigated; run while writing this spec | reproduced; documented; the earlier ones are listed in `model.unread` (Y28) | `garden` (two garden surfaces) |
| L19 | The garden water is `empty_vqip()` with volume (`demand.py:273-287`) | it enters the soil at 0 °C; run while writing this spec | reproduced; documented (Y29) | `garden` |
| L20 | `NutrientPool`'s parameters are unreachable: built without arguments (`land.py:1273`), and `apply_overrides` is never called (`np.py:155-181`). Its defaults are shared mutable dicts that `apply_overrides` would mutate for every pool (`np.py:17-23`, `:169-179`) | no way to configure the pool | drivers with WSIMOD's defaults, per surface; refused as configuration (Y24) | driver test; refusal test |
| L21 | `allocate_residue` returns `self.empty_nutrient()`, calling a dict (`np.py:312`); residue is unwired (`land.py:1281`) | unreachable `TypeError` | not ported; residue data refused (Y36) | refusal test |
| L22 | A growing surface's `et0_coefficient` is overwritten at construction and daily (`:1265`, `:1570`) | the configured or overridden value is never read | listed in `model.unread` (Y39) | `test_a_growing_et0_coefficient_is_unread` |
| L23 | `quick_interp` reads `xp[-1]` when `x ≤ xp[0]` (Python's negative index, `:1505-1509`), and raises `IndexError` beyond `xp[-1]` | silent wrap (run while writing this spec: 0.30000000000000004), or a crash | the wrap is reproduced; the crash is refused at build (Y14) | unit tests; refusal test |
| L24 | Erosion reads `solids`, which the docstring's pollutant warning omits (`:1119-1121`, `:2088`) | `KeyError` on the first eroding day | refused by name (Validation) | refusal test |
| L25 | Dry deposition without wet data (`:540-569`) | `KeyError: ('nhx-wet', …)`; run while writing this spec | refused by name | refusal test |
| L26 | The harvest branch's `crop_factor = crop_factor_stages[0]` (`:1539`) is overwritten at `:1547` | dead store | not ported; noted | — (no observable effect) |

Behaviours that are transcribed and are not defects:

- the Newton test lags one iteration;
- the sowing day has no crop;
- the eroded masses bypass the tank;
- the raw and adjusted days of year differ;
- a non-additive `pollutant_load` has an effect only by rounding.

They are documented on the application page.

## Differentiability

- **Learnable:** every real-valued driver of Inputs, with the land forcing (precipitation, et0
  and temperature, 3a) through every new process.
- **Detached:** every decision of the vocabulary above.
- **Kinks and jumps.** They are documented on the application page, and the gradient tests
  avoid them.
  - **The calendar** is structural. `crop_factor` is piecewise linear in `crop_factor_stages`,
    with exact gradients, and constant in the dates. The covers are linear in `crop_cover_max`
    and `ground_cover_max` inside the season and frozen outside it.
  - **Sowing and harvest** are jumps in time, with no parameter gradient.
  - **`Ks`.** `max(fc_m − smc, 0)` and `max(0, ·)` are kinks. The `< RAW` switch is continuous,
    because `Ks = 1` at `RAW`. The zero-denominator case is a jump from 1 to 0 at the wilting
    point when `p = 1` (Y17).
  - **The temperature factor** jumps at 5 °C (L12) and has kinks at 0 °C.
  - **The soil-moisture factor** is continuous at `fc_m` and `wp_m`, with the `min` kinks. The
    slope of `x^thetapow` is unbounded at 0 for `thetapow < 1`.
  - **The uptake** has its `min` caps and the `max(0, min(1, ·))` of the autumn temperature as
    kinks. Its `shape` guard is continuous.
  - **The transformation's `in_`/`out_` switches** are continuous kinks, because both sides are 0
    at 0. The pool extracts' `min` caps are kinks.
  - **Erosion.**
    - Rain above 5 mm is a jump, from 0 to `8.95 + …`.
    - `erodingflow > 0` is continuous. `ie.volume > 0` is continuous for `sreroexp > 0` and a
      jump for `sreroexp = 0` (`0 ** 0 = 1`; Task 8a review).
    - The enrichment's switch at 4 is continuous (`4 − 2.5 = 1.5`).
    - `min(1, ·)` of the transport factor is a kink.
  - **Denitrification** is continuous at `fc_m` (`f = 1` on both sides) and at `limpar` (`f →
    0`).
  - **Adsorption.**
    - The `pool_ == 0` and `conc_sol ≤ 0` guards are jumps.
    - `abs(equi − conc_sol) > 1e-6` is a jump of at most 1e-6 in concentration.
    - The adsorb/desorb switch is continuous.
    - The Newton iteration count jumps where a `break` or residual test flips. Within a fixed
      count, the gradient is the exact derivative of the unrolled iterations (Y25), which
      approaches the implicit derivative of the equilibrium as the loop converges.
  - **Irrigation and garden demand.** `max(E − P, 0)` is a kink. `> FA` and `cmd ≤ FA` are jumps
    of `FA` size.
  - **`VariableAreaSurface`** is smooth (linear in `current_surface_area`).
- **Per-instance Python math (Y26).** Approach rule 1 lists it. Every power base is non-negative
  in its branch: the soil-moisture factor's quotients are positive inside their branch, and
  `(erodingflow / 4)` is ≥ 0. A base of 0 has `pow_python`'s existing derivative convention.
- **Cost.**
  - A growing surface makes about 30 per-instance Python calls per day, plus 3 per Newton
    iteration, up to 20 iterations by default.
  - `benchmarks/wsimod_run.py` gains the `growing_spring` case at batch sizes 1, 10 and 100,
    with no asserts (W8).
  - The default suite runs `growing_spring` on its first 540 days. The full three years run under
    `-m slow` (Verification).

## Verification (Y3)

The harness is 3a's: live WSIMOD 0.8.1, `reset_wsimod_globals()` before every build, and the
cases in `tests/verification/_wsimod_cases.py`. The bounds are W19's: 0 (bitwise) expected, and
1e-12 × scale only with a named, measured and asserted admission.

### 1. Per-class replay on captured inputs

- **Unit tests against WSIMOD's own objects.** Random sequences (seeded), asserted bitwise, each
  with a stub parent that supplies `t`, `monthyear`, `get_data_input` and
  `running_*_mb`:
  - **`NutrientPool`:** every method in Scope › In; random pools, factors and requests,
    including requests above the pool (the `min` caps), zero pools, and non-default parameters
    set on the WSIMOD object directly.
  - **`quick_interp` and `calc_crop_cover`,** over every day of 2003–2005, for:
    - the spring calendar (90/250) and the autumn one (280/200);
    - the default (1/365);
    - a wrapping `crop_factor_stage_dates` (L23);
    - stress either side of `RAW`.
  - **`GrowingSurface.pull_storage`,** with random tanks and pools, `V = 0`, a binding volume
    cap, and the `din = 0` guard (noodl only; WSIMOD raises, asserted).
  - **The processes:** `soil_pool_transformation`, the two factors (either side of 0 °C, 5 °C,
    `fc_m` and `wp_m`), `calc_crop_uptake` (spring and autumn, every cap), `erosion` (rain below
    and above 5 mm, `erodingflow` 0, in (0, 4] and above 4, an erodable pool shorter than the
    request), `denitrification` (each branch) and `adsorption`. Adsorption covers:
    - the empty pool and the shortcut;
    - Newton exits by residual, by step and by `maxiter`, with `maxiter` overridden to 2;
    - the `xn_1 ≤ 0` clamp, from a huge first guess (`conc_sol / kfr` large, `nfr` small), whose first step `x0 - dx` is positive in exact arithmetic (`(nfr * A + D) / f'`, `A` and `D` the adsorbed and dissolved P per km²) but cancels to `≤ 0` in rounding; a large `kfr` makes `x0` small and never clamps (corrected while implementing Task 8b, W2);
    - both directions, and a short pool, which raises the event.
  - **The deposition methods** of `Surface` and `GrowingSurface`, with `adjust_vqip_to_liquid`.
  - **`IrrigationSurface.irrigate`** with a scripted `pull_distributed`, and **`GardenSurface`'s
    handlers**.
  - **`VariableAreaSurface.get_climate_`.**
- **Node replays** (`capture_node_calls`, bound 0). `Land` on both demos (live loads) and on every
  purpose-built case below. They cover the pools, crop state, climate, flows and running
  balances before and after each call. The default suite samples days as 3a does (W15); every
  day is replayed under `-m slow`.

### 2. Whole-model parity

- Each demo's `config.yml` runs with pollutants on and is compared bitwise (every arc's flow and
  pollutants, every recorded tank, every surface field) against WSIMOD, in 3a's composition:
  quickstart in full and oxford for 365 days by default, full oxford under `slow`.
- The quantity-only run of `config.yml` stays. A test asserts that the quantity-only volumes
  equal the live volumes exactly, on both demos and on every purpose-built case.
- WSIMOD's prints are checked (W90/W91) with the event mapping and the one named admission
  above (Y35, Y41): bounded by the store per case and pollutant, and zero lines asserted where
  zero is measured.
- noodl's balance closes within 1e-12 × scale on every case and day (X8's contiguous window),
  and the parts sum to the aggregates.

### 3. Purpose-built cases

Each runs in both modes (where pollutants matter) and is compared with WSIMOD as it is, with no
patches. The default dates start on 2003-10-01, so a run crosses 29 February 2004 and a full
season. Each case pins what is listed.

- **`surface_deposition`.** An `ImperviousSurface` and a `PerviousSurface` with monthly `nhx`,
  `noy` and `srp` dry and wet data over 15 months; a third surface with wet data only (unread);
  `pollutant_load`s with an additive entry, a non-additive `temperature` entry and a key that is
  not a pollutant (unread). It pins:
  - `monthyear` lookups across month and year boundaries;
  - the deposition order before `ihacres`;
  - the non-additive blend at volume 0.
- **`growing_spring`.** One `GrowingSurface` (sowing 90, harvest 250), `ET_depletion_factor =
  0.5`, five-point `crop_factor_stages`, fertiliser and manure data, dry and wet deposition,
  `initial_soil_storage`, and the full eight pollutants. The forcing (three years under `slow`,
  540 days by default) has:
  - rain above 5 mm and runoff;
  - dry spells that stress the crop;
  - soil temperatures below 0, in (0, 5] and above 5 °C.

  It reaches every branch of the transformation, uptake, erosion, denitrification and
  adsorption. A test asserts from the decisions that each branch is reached at least once
  (`-m slow` only, as X39). It pins the call order, the pool gap (L8, L9) and the leap shift
  (L1).
- **`growing_autumn`.** Sowing 280 and harvest 200, which pins the unsorted calendar (L2), the
  autumn uptake temperature function and `days_after_sow − 25`, and a season crossing the new
  year.
- **`growing_no_nitrate`.** The same surface with pollutants `phosphate` and `temperature` only (no
  `nitrate`), which pins the structural switch: calendar and IHACRES only, deposition straight
  into the tank, and no pool state. It runs with pollutants on; noodl's quantity-only mode runs
  on every case separately (Y49).
- **`growing_decay`.** The full 12 pollutants and `decays` on ammonia and nitrite. It pins:
  - 3b's decay and nitrite carry-over on a growing tank;
  - the pool that ignores them (L9);
  - `"nutrient_pool_not_tracking"`.
- **`irrigation`.** An `IrrigationSurface` fed by a `River`, a `Groundwater` and a `Reservoir`
  through the `Land`'s in-arcs. It pins:
  - dry days with demand, saturated days (`cmd ≤ FA`), days before sowing, and the sowing day
    itself;
  - the default orchestration's `apply_irrigation`, and a second run with `apply_irrigation`
    before `Land.run`, which uses the previous day's flows (the carried climate);
  - nutrients carried into the pools.
- **`garden`.** A `ResidentialDemand` through a `Distribution` to a `Land` with two
  `GardenSurface`s and a `PerviousSurface`. It pins:
  - the previous day's demand, `gardening_efficiency`, the 0 °C garden water (L19) and the last
    garden surface taking the handlers (L18);
  - the zero answer of a `Land` with no garden surface on the same demand.
- **`variable_area`.** A `VariableAreaSurface` at the default `current_surface_area = 0`, and one
  at half its area. It pins the scaled climate and erosion's unscaled rain.
- **`surface_overrides`.** It pins:
  - `data_input_dict` overrides as a path (written into a temporary directory, absolute, as
    X11's) and as a mapping;
  - a `pollutant_load` override on a surface with a load (live) and on one without (unread,
    L14);
  - every `GrowingSurface` override, including `rooting_depth` and the calendar;
  - `irrigation_coefficient`.
- **Crash cases.** WSIMOD raises; noodl runs, with the evident intent and the event, at value
  parity up to WSIMOD's crash day:
  - `growing_default_depletion` (L3);
  - `growing_no_din` (L5): a zero-N tank, no deposition, rain on day 0;
  - `growing_zero_n` (L6): a plain zero-N tank with zero first-month inputs, crashing on day 0;
  - `growing_nitrite_only` (L6): the nitrite-only variant;
  - `growing_empty_soil` (L7): `ET_depletion_factor = 0.5`, an empty soil and a sown crop, so
    that `:1899` and not `:1566` is reached;
  - `irrigation_without_nutrients` (L13).

Default-suite time is measured as W15 requires, and no case may be among the default suite's
slowest items; if one is, its default run is cut (the full run stays under `slow`).

### 4. Gradients

The protocol is 3a's:

- central differences at relative step 1e-6 and agreement 1e-6;
- the `"decisions"` asserted identical between ±h;
- the kink rule of W92/W93.

The derivatives checked are those of arc pollutant masses, volumes, final storages and the pools
with respect to:

- `crop_factor_stages` and `ET_depletion_factor`, through the volumes; `rooting_depth`;
  `irrigation_coefficient`; `current_surface_area`;
- `uptake1`, `uptake3`, `denpar`, `hsatINs`, `exppar`, `kfr`, `nfr`, `kadsdes`, `erodibility`,
  `sreroexp`, `srfilt`, `satact` and `thetapow`;
- the pool rates `minfpar`, `immobdpar` and `fraction_manure_to_dissolved_inorganic`;
- the fertiliser, manure and deposition forcing, and `pollutant_load`;
- the soil temperature, through the factors (away from 0 °C and 5 °C).

`torch.autograd.gradcheck` also runs on `quick_interp`, one `adsorption` call (fixed iteration
count) and one `erosion` call away from their guards. A test asserts that
`adosorption_nr_limit`'s gradient is 0.

### Refusals and structure

- Each refusal of Interaction › Validation gets a by-name test.
- A batch of 4 instances with different 3c parameters equals the 4 single runs.
- `model.step` day by day equals `wsimod_run` with the new state keys.
- A state round trip through `initial_state`, the pools, the crop registers and the climate is
  exact.
- The no-`wsimod`-import subprocess test is unchanged.

## Delivery and size (Y7, Y31)

**One PR.** The estimate, from 3b's PR #42 (2,146 source lines, 9,160 in all), is:

| Part | Source lines | Test and case lines |
|---|---|---|
| deposition, overrides and validation | 450 | 1,000 |
| `NutrientPool` and `pull_pollutants` | 350 | 700 |
| `GrowingSurface` processes, calendar and Newton | 1,300 | 2,600 |
| `IrrigationSurface`, `GardenSurface`, `VariableAreaSurface` and the handlers | 250 | 900 |
| state, drivers, forcing and parts | 400 | 800 |
| the harness | — | 600 |

That is about 2,750 source lines and 6,600 test lines, with about 700 lines of docs: some 10,000
lines in all, about 10 % more than 3b.

The 3c-ii half of Y7 (Irrigation, Garden, VariableArea and the overrides) is about 15 % of that,
and it depends entirely on 3c-i. A second PR would add a full review cycle for little code, while
the first would leave the garden link and `apply_irrigation` refused for one more merge.

The plan orders 3c-ii's tasks last, behind a clean commit boundary. If the final review finds the
PR too large to review, they can move to a follow-up PR without rework. That is the fallback, not
the plan.

## Documentation

- **`docs/applications/wsimod.md`:**
  - What is ported: the four surface classes, deposition, the pool, irrigation and the garden.
  - New sections for deposition, the growing surface (call order, calendar, pool, processes),
    irrigation, the garden link and the variable-area surface.
  - State and drivers: the new keys, drivers and forcing.
  - Quantity only or with pollutants: no more "until 3c".
  - The demo configurations: `config.yml` live, `config_no_deposition.yml` retired.
  - Refusals.
  - Where WSIMOD is silent or wrong: rows L1–L25, replacing the `residue` row with L21.
  - Differentiability: the kinks and jumps above.
  - Limitations: the 3c item becomes "every WSIMOD 0.8.1 class in `nodes/` and `arcs/` is ported
    except `EnfieldFoulSewer` and the never-ported list".
  - Verification rows with measured values.
  - The pool–tank gap, explained for users (L8, L9) with the event to watch.
- **`docs/formats/wsimod.md`:**
  - the four surface classes and their keys;
  - the surface data variables and their monthly dates;
  - the surface `data_input_dict` override (path relative to the working directory, X27) and
    the refused node and `filename` forms;
  - the `GrowingSurface` and `IrrigationSurface` overrides;
  - `initial_soil_storage`;
  - the 3c refusal lines replaced.
- **`tests/data/wsimod/README.md`:** `config_no_deposition.yml` retired; `config.yml` runs live.
- **`docs/verification.md`:** the new rows.
- **`AGENTS.md` and `.github/scripts/select_tests.py`:** the wsimod row is unchanged, since the
  paths do not change.

## Rulings made in this spec

Each ruling states what was decided, why, and the cost if it is wrong.

- **Y9, scope and the refusal table.** 3c ports Scope › In. Every "3c" refusal is decided in the
  table: ported, or refused for good with a reason.
  - Why: Y1, checked against `git grep 3c`.
  - Cost if wrong: a refusal left behind.
- **Y10, `RiparianBuffer` and `River.pull_check_fp` are refused for good.**
  - Why: no class exists, nothing sends the tag, and the handler cannot run (`AttributeError`)
    nor return a VQIP. There is nothing to transcribe.
  - Cost if wrong: if a later WSIMOD adds the class, noodl ports it then.
- **Y11, WSIMOD's balance boundary, with a breakdown.** The pools are carried and are not held.
  The aggregate running terms stay the balance, and the per-function parts are a diagnostic that
  is checked to sum to them.
  - Why: WSIMOD's lists close on the tank boundary, and the aggregates are bitwise WSIMOD's.
    Moving solids inside the boundary would need terms WSIMOD does not have.
  - Cost if wrong: users read the pools' own mass changes from the state, not from the balance.
- **Y12, the state keys** `"wsimod.nutrient_pool"`, `"wsimod.days_after_sow"` (−1 for `None`),
  `"wsimod.crop_cover"`, `"wsimod.ground_cover"` and `"wsimod.surface_climate"`, as tabled.
  - Why: Y5; one tensor per concept, and −1 keeps `days_after_sow` a single float register.
  - Cost if wrong: a state-format change before release.
- **Y13 (amended by Y42), the leap test is reproduced (L1).**
  - Why: WSIMOD's individual-files loader, which reads the format noodl reads, does it, and
    parity needs it.
  - Cost if wrong: crop factors shifted by one day; documented.
- **Y14, the calendar arithmetic is done at build time.** That covers `bisect_left`, Python
  indexing (the wrap included) and the `sin` of the date. Configurations whose interpolation
  would raise are refused by date: the run's dates at build, later dates when `model.step`
  reaches them (Y47).
  - Why: the inputs are structural, so the result is Python's by construction, and the gradients
    follow the piecewise-linear segments (Y2).
  - Cost if wrong: none for values; a refusal names the first failing date.
- **Y15, the autumn calendar is reproduced (L2).**
  - Why: no crash and no single evident form (wrap the season over the new year, or sort the
    list), and parity needs it.
  - Cost if wrong: autumn crops without cover in noodl as in WSIMOD; documented.
- **Y16, the pool–tank gap is reproduced, and its intended checks become events.** The double
  flush (L8) and the nitrate-only removals (L9) are transcribed.
  `"nutrient_pool_not_tracking"` and `"freundlich_flow_adjusted"` record what the dead checks
  (L10, L11) meant to report.
  - Why: every growing-surface value depends on these behaviours, the fix has two candidate
    forms, and X14's precedent is to reproduce and document. The events make the gap visible.
  - Cost if wrong: noodl inherits a modelling error, which is documented and visible.
- **Y17 (amended by Y44), the stress coefficient's zero denominator gives `Ks = 0` (L3).** It
  applies to `ET_depletion_factor = 1` only, since `TAW ≤ 0` is refused.
  - Why: WSIMOD crashes there; 0 is the formula's limit and FAO's value at the wilting point.
  - Cost if wrong: dry-soil evaporation of default growing surfaces differs from a future WSIMOD
    fix; recorded by the event.
- **Y18, three zero denominators are guarded (L5, L6, L7).** Respectively: species 0 when the
  tank has no DIN, the change to ammonia when it has neither nitrate nor ammonia, and no uptake
  from an empty soil. Each has an event.
  - Why: WSIMOD crashes on states that ordinary runs reach. Each choice is the only one that
    neither creates nor misplaces tank mass, or for the split, the mineralisation product.
  - Cost if wrong: on such days the species split or the uptake differs from a future fix.
- **Y19, the soil temperature factor's jump is reproduced (L12).**
  - Why: parity, X14's precedent; the HYPE form is evident in intent, but changing it changes
    every cold-soil value.
  - Cost if wrong: a documented modelling error.
- **Y20, irrigation updates the pool only when `nitrate` is a pollutant (L13).**
  - Why: WSIMOD crashes otherwise, and every other pool access sits behind that switch.
  - Cost if wrong: none observable, since the pool is inert without `nitrate`.
- **Y21, the constructor's deposition flags are structural (L14, L15).** Overrides that WSIMOD
  ignores are listed in `model.unread`; an override that would crash is refused.
  - Why: WSIMOD fixes the inflow list at construction.
  - Cost if wrong: a user's load override silently does nothing, in noodl as in WSIMOD; it is
    listed.
- **Y22, the surface `data_input_dict` override is ported; the node form and the surface
  `filename` form are refused for good.**
  - Why: only `Surface.apply_overrides` consumes `data_input_dict`; `Node` consumes only
    `filename`, and no surface consumes `filename`.
  - Cost if wrong: a node-data override must go through `filename` (X27).
- **Y23, surface forcing as daily drivers `"wsimod.<variable>"`, filled monthly.** They are
  required in both modes when WSIMOD reads them, and their values are ignored in a
  quantity-only run.
  - Why: 3a's forcing layout; WSIMOD crashes on missing data whatever the run mode.
  - Cost if wrong: a driver-layout change; a batched value may vary within a month, which the
    docs flag.
- **Y24, the `NutrientPool` parameters are drivers with WSIMOD's defaults, with no configuration
  route (L20).**
  - Why: WSIMOD has none. Drivers let users calibrate without inventing a configuration key, and
    the defaults reproduce WSIMOD.
  - Cost if wrong: if WSIMOD later wires `apply_overrides`, noodl adds the override then.
- **Y25, the Newton loop is unrolled to `adsorption_nr_maxiter` under masks.** The gradient flows
  through the unrolled iterations.
  - Why: branch-free and bitwise per instance; the iteration count is structural.
  - Cost if wrong: speed (20 iterations by default), measured in the benchmark; a gradient that
    differs from the implicit one when the loop stops early.
- **Y26, per-instance Python math.** `pow_python` and `exp_python` are used everywhere, with the
  new `log_python`, `log10_python` and `sin_python`, and date-only values are computed at build.
  - Why: the measured last-bit differences (Approach rule 1).
  - Cost if wrong: speed at large batches.
- **Y27 (amended by Y44), stricter refusals.** These are `wilting_point ≥ field_capacity` (L4),
  `depth` (L17), and the parameter-domain zeros listed under Validation.
  - Why: each is a WSIMOD crash, or a print followed by meaningless results.
  - Cost if wrong: a configuration that WSIMOD runs must be edited; the `TAW < 0` one is the
    stricter case.
- **Y28, the last `GardenSurface` takes the handlers (L18).** The rule is reproduced, documented
  and listed.
  - Why: no crash, and the static structure is WSIMOD's.
  - Cost if wrong: users with two gardens on one land see one irrigated, as in WSIMOD.
- **Y29, garden water at 0 °C is reproduced (L19).**
  - Why: parity; the intended temperature (the house mix? the air?) is not evident.
  - Cost if wrong: cooled garden soils; documented.
- **Y30, `config_no_deposition.yml` is retired.**
  - Why: the demos' only 3c process is now live, so there is nothing left to neutralise (Y3).
  - Cost if wrong: fixture and test churn.
- **Y31, one PR (Y7).** See Delivery and size, with the 3c-ii tasks last as the fallback.
  - Why: 3c-ii is small and wholly dependent on 3c-i.
  - Cost if wrong: a PR about 10 % larger than 3b's to review.
- **Y32, `crop_factor` and `et0_coefficient` are recorded, not carried.**
  - Why: they are recomputed before every reader. `et0_coef` is in WSIMOD's surface record
    (`model.py:1150`).
  - Cost if wrong: a state key added later.
- **Y33, `crop_factor_stages` is dense per class with inert padding.**
  - Why: 3a's one-layout-per-class rule (X34) and X9's inert pattern.
  - Cost if wrong: a driver-shape change.
- **Y34 (amended by Y46), the eight events** of the events table.
  - Why: two mirror WSIMOD prints (W90/W91), two give dead checks their intent, and four record
    evident-intent guards (D13).
  - Cost if wrong: more diagnostic keys.
- **Y35 (amended by Y41), one print-check admission:** WSIMOD's round-off "mass balance error"
  lines on growing lands, bounded by the store (`|x| ≤ c · ε · S`, per case and pollutant), with
  zero lines asserted wherever zero is measured.
  - Why: run while writing this spec; they are WSIMOD's own summation rounding, while noodl's
    residual closes.
  - Cost if wrong: a real WSIMOD balance error below the bound would go unflagged; noodl's own
    balance still flags any of its own.
- **Y36, residue is not ported, and residue data are refused.**
  - Why: residue is unwired, and its pool method would crash (L21).
  - Cost if wrong: a configuration carrying residue data, which WSIMOD ignores, must drop them.
- **Y37, `initial_soil_storage` is a mapping with the six keys, or falsy (L16).** It is read only
  with `nitrate`.
  - Why: WSIMOD's code, against its docstring.
  - Cost if wrong: none, since a float crashes WSIMOD.
- **Y38, `pollutant_load` keys.** A non-additive entry is accepted and transcribed, and so is a
  non-additive deposition species (Y43). A key that is
  not a pollutant is listed in `model.unread`.
  - Why: WSIMOD runs both; 3a's refusals were placeholders until 3c.
  - Cost if wrong: a typo goes unnoticed except in `model.unread`.
- **Y39, a growing surface's `et0_coefficient` is listed as unread (L22).**
  - Why: WSIMOD overwrites it.
  - Cost if wrong: none.
- **Y40, a growing surface's `decays` are 3b's, applied to the tank only.**
  - Why: `GrowingSurface` is a `DecayTank` through `Surface` (`land.py:303`, `:366-371`), and
    nothing decays the pool.
  - Cost if wrong: a pool–tank gap, reproduced and visible (L9).

### Rulings after the review (Y41–Y50)

These record the coordinator's rulings on the Fable spec review (findings 1–14).

- **Y41, the Y35 admission is bounded by the store (finding 1).** Per case and per pollutant,
  `|x| ≤ c · ε · S`, with `S` the largest store of that pollutant on the land's surface tanks
  that day and `c` measured and stated in the test. Zero printed lines are asserted wherever
  zero is measured, and the admitted text is exactly "mass balance error for <additive> of x in
  <land>" on lands holding a growing surface.
  - Why: WSIMOD's normaliser is the flow scale, so `|x| / magnitude` is unbounded as the flows
    shrink (the review measured 1.7e-9, seventeen times the first draft's bound), while the
    rounding scales with the store.
  - Cost if wrong: a case with a larger store-relative rounding fails, and its `c` is
    re-measured.
- **Y42, L1 names its loader (finding 2).** The decision stands. The claim is corrected to the
  individual-files loader (`model.py:347-348`), and the unified-data loader's pandas dates (a
  shift in leap years only) are named, here and on the application page.
  - Why: the first draft called the quirk universal, which it is not.
  - Cost if wrong: none for values; a user comparing with a parquet-loaded WSIMOD is told why.
- **Y43, non-additive deposition species are transcribed, not refused (finding 5).**
  - Why: WSIMOD runs them, with no crash and no print, as it runs Y38's non-additive load.
  - Cost if wrong: a configuration error goes unflagged; the effect is rounding in the tank and
    a pool–tank gap on a growing surface, documented.
- **Y44, `wilting_point ≥ field_capacity` is refused by name (finding 6).** Y17's guard then
  covers only `ET_depletion_factor = 1`.
  - Why: `TAW = 0` crashes WSIMOD on day 0, and under Y17 it would run a soil that never
    evaporates; nothing WSIMOD can run is lost.
  - Cost if wrong: none for runnable configurations; `TAW < 0` remains the stricter case (Y27).
- **Y45, L8 is closed: reproduced, no switch (finding 14).** The two fix forms differ. Dropping
  the flushing leaves `pull_storage`'s two sequential extractions, which remove `DI₀·(p₁ + p₂ −
  p₁p₂)`. Dropping the extraction in `pull_storage` and keeping the flushing removes `DI₀·(p₁ +
  p₂)`, and the reply's species split then needs a pool read that WSIMOD does not have. Neither
  is evident. `test_the_pools_are_flushed_twice` asserts that noodl's tank–pool gap on
  `growing_spring` equals WSIMOD's, not a hard-coded ratio. Whether L8 and the other defects go
  upstream is the user's decision, which is pending (Y4).
  - Why: parity, X14's precedent, and no single evident fix.
  - Cost if wrong: noodl inherits a large modelling error, documented and visible in the state.
- **Y46, two events for the two zero-DIN guards (finding 13).** `"pull_storage_no_din"` (species
  set to 0, `:1459-1485`) and `"transformation_no_din"` (proportion set to 0, `:1758`) replace
  `"nutrient_split_zero_denominator"`.
  - Why: the remedies differ, and a reader should tell them apart by name.
  - Cost if wrong: one more event key.
- **Y47, the interpolation refusal checks the run's dates (finding 11).** A `model.step` past
  them is checked, and refused by name, when the step is taken.
  - Why: parity with WSIMOD, which only crashes on dates it runs.
  - Cost if wrong: a warm-started step can fail late, with a named refusal.
- **Y48, the crash triggers are named as ordinary (findings 3, 4).** L5 and L6 hit any growing
  surface started from a zero-N tank with zero first-month inputs (deposition masks the day-0
  `:1460` crash); L7 needs `ET_depletion_factor < 1`. The new case `growing_zero_n` and
  `growing_empty_soil` (at `p = 0.5`) pin them, and the application page says that WSIMOD needs
  initial inorganic N or a non-zero first-month input.
  - Why: the first draft's triggers read as exotic, and `growing_empty_soil` could have
    duplicated L3.
  - Cost if wrong: one more crash case.
- **Y49, `growing_no_nitrate`, and the forcing check uses the network's pollutant list (finding
  9).** The case formerly named `growing_quantity` runs with pollutants on. The fertiliser and
  manure check applies in both modes whenever the network lists `nitrate`.
  - Why: the old name suggested noodl's quantity-only mode, and the forcing rule was ambiguous
    across modes; WSIMOD reads the data whenever its `POLLUTANTS` holds `nitrate`.
  - Cost if wrong: a quantity-only run must carry fertiliser data it does not use.
- **Y50, textual corrections (findings 7, 8, 10, 12).**
  - The sign of zero is qualified to the `torch.equal` comparison (rule 6).
  - 3c adds nothing to `end_timestep` (`day.py:1216`).
  - The default calendar is unsorted too, and benign.
  - `current_surface_area` is a constructor keyword and a driver, refused only as an override.
  - Why: each sentence could be read two ways.
  - Cost if wrong: none.

## Open questions

- **Upstream filing (Y4).** Whether L1–L25 go upstream as a third issue after #258 and #259 is
  the user's decision, which is pending. Nothing is filed until then. The crash and mass-loss
  rows (L3, L5–L10, L13, L14) are the strongest candidates.

# Data assimilation library reference

Standalone reference for `libs/data-assimilation`. Read this alongside
[codebase_guide.md §6](codebase_guide.md#6-data-assimilation-flow), which
covers how the library is wired into the monorepo. Config keys and the run
scripts' artifacts are in [scripts_and_configs.md](scripts_and_configs.md) and
[configs/README.md](../configs/README.md); metrics and figures computed from
the artifacts are in [evaluation.md](evaluation.md).

---

## 1. Purpose and scope

The library implements ensemble data assimilation in JAX, in three flavors:

* **Smoothing** — Ensemble Smoother with Multiple Data Assimilation (ESMDA):
  per assimilation window, the whole window is re-forecast `num_steps` times
  with tempered (`alpha`-weighted) Kalman updates of the window's initial
  condition and/or parameters (§4–§5).
* **Filtering** — the sequential ensemble Kalman filter (EnKF): per cycle, the
  ensemble is forecast one segment and a full-weight analysis updates the
  end-of-segment state and/or parameters, which warm-start the next cycle
  (§8).
* **Filter smoothing** — the hybrid: per window, a parameter-only ESMDA
  followed by a filter for the state (§9).

All are **solver-agnostic**: they take any `BaseEnsembleForwardModel` (see
[base_ensemble_forward_model.py](../src/pyurbanair/base_ensemble_forward_model.py))
so the same classes cover pylbm, pyudales, pypalm, and the neural surrogate.
They share one analysis implementation, one augmentation (flatten/unflatten)
layer, and the localization machinery.

All public entry points accept and return `xarray.Dataset`; internal arrays
are `jax.numpy` (JAX-CPU throughout; no JAX GPU use within this library).

Parameter-bearing smoothers and filters accept `parameter_names_to_estimate`
(`None` for all parameter fields, an explicit list for that subset, or `[]`
for none). Their forecasts and returned parameter artifacts retain the full
Dataset. Only selected fields enter parameter analysis, localization, inflation,
and filtering parameter evolution (such as random walks). Unselected fields
retain their supplied values, including ensemble spread. State-only modes keep
every supplied parameter fixed. The run scripts map
`assimilation.params_to_estimate` to this argument.

Source tree:

```
libs/data-assimilation/src/data_assimilation/
  observation_operator.py   # ObservationOperator, TemporalObservationOperator,
                            #   AggregateObservations, flatten_observations,
                            #   sensor_observation_coords
  observation_error.py      # ObservationErrorSpec, ResolvedObservationError
  interpolation.py          # trilinear grid-to-point interpolation
  reduction.py              # OnlineStateReduction, StreamingStateReduction
  augmentation.py           # ParamAugmentation, StateAugmentation — the
                            #   Dataset <-> flat-array transforms shared by
                            #   smoothing and filtering
  parameter_selection.py    # parameter_names_to_estimate select/merge
  inflation.py              # InflationScheme, MultiplicativeInflation, RTPS, RTPP
  io.py                     # load_dataset, get_sorted_state_files (shared file I/O)
  localization/
    base.py                 # BaseLocalization, taper_inflation, localized_update,
                            #   resolve_row_inflation, active_observations
    correlation.py          # CorrelationLocalization
    distance.py             # DistanceLocalization
  smoothing/
    base.py                 # BaseSmoothing (_forecast_step, _observation_step)
    esmda.py                # all five ESMDA variant classes
  filtering/
    analysis.py             # stochastic_enkf_update (shared with ESMDA),
                            #   AnalysisScheme, StochasticEnKFAnalysis
    etkf.py                 # ETKFAnalysis, LETKFAnalysis, ObservationTSVD
    base.py                 # BaseFilter (cycle loop), EnsembleKalmanFilter,
                            #   FilterResult, CycleDiagnostics, validate_beta
    parameter_evolution.py  # ParameterEvolution, RandomWalkEvolution
  filter_smoothing/
    base.py                 # FilterSmoothing, FilterSmoothingResult,
                            #   trajectory helpers
    tempering.py            # TemperingPolicy, resolve_tempering_policy
```

---

## 2. Observation operator

**File:**
[observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py)

### `ObservationOperator`

Maps one state `xarray.Dataset` (or an ensemble of states) to a flat
NumPy vector of length `num_sensors * len(obs_states)`.

**Two construction modes** (mutually exclusive):

| Mode | Args | How |
|---|---|---|
| Index-based | `obs_ids_x`, `obs_ids_y`, `obs_ids_z` | Direct `isel` with xarray vectorized indexing |
| Coordinate-based | `obs_x`, `obs_y`, `obs_z` | Trilinear interpolation via `interpolation.py` |

The cases use coordinate-based mode: `observation.operator` in
[configs/assimilation.yaml](../configs/assimilation.yaml) passes the case's
`obs.*_points` lists, and `make_observation_operator` in
[scripts/utils/helper_functions.py](../scripts/utils/helper_functions.py)
instantiates it with the truth or assimilation model's `solver_name`.

**Staggered-grid `dim_mapping`.** Each backend uses different dimension names
for the velocity components. The operator holds a `dim_mapping` dict that maps
`{variable -> {z, y, x} -> dim_name}`:

- `"pylbm"` (regular grid): uniform `x/y/z` for `u`, `v`, `w`.
- `"palm"` (post-processed): `u` on `xu`, `v` on `yv`, all on one `z`.
- `"udales"` (staggered C-grid): `u` at `(zt, yt, xm)`, `v` at
  `(zt, ym, xt)`, `w` at `(zm, yt, xt)`.

Adding a new backend requires a new `elif solver_name == "..."` branch in
`ObservationOperator.__init__` (§12).

`__call__(state)` dispatches to `_observation_single` or
`_observation_ensemble` depending on whether an `ensemble` dim is present.
The flattened vector pattern is `[all_sensors_for_var0, all_sensors_for_var1,
...]`, so the **sensor index is the innermost (fastest) axis within each
variable block**.

### `TemporalObservationOperator`

Wraps `ObservationOperator` and applies it to *every frame* of the window's
`time` dimension. It performs **no aggregation**. `__call__` returns an
`xarray.DataArray` — dims `("time", "obs")` for a single state,
`("ensemble", "time", "obs")` for an ensemble — carrying the state's
seconds-valued `time` coordinate, with the base operator's `obs` layout.

### `AggregateObservations`

A standalone callable `AggregateObservations(interval_seconds, mode="mean")`
that maps an observation DataArray (dims `(..., "time", "obs")`) to an
aggregated one. Frames are binned by their `time` coordinate (in seconds)
into contiguous `interval_seconds`-wide bins — frame at time `t` belongs to
bin `floor((t - t0) / interval_seconds)` — and reduced within each bin with
`mode` (`"mean"`, `"median"`, `"max"`, `"min"`). The output keeps the dims,
with `time` re-labelled to the interval start times. An empty interval raises
(a silent gap would misalign the flattened vector). By default the interval
count is fixed by the first call, because `C_D` is sized from the first window;
`make_aggregation` sets `allow_interval_count_change = True` so windows may
differ (e.g. a shorter last one).

It is an optional constructor input to the ESMDA smoothers
(`aggregate_observations=`). They route both the real observations and the
predicted observations `H(x)` through one path: optional aggregation, then
the module-level `flatten_observations` helper — a time-major flatten
(`("time", "obs") → (T·num_obs,)`, ensemble inputs to `(N_e, T·num_obs)`).
With `aggregate_observations=None` the full time-resolved vector is assimilated
as one batch. Aggregating observations rather than states is exactly
equivalent for `"mean"` (the operator is linear in the state).

The **sequential filter does not aggregate at all**: it assimilates every
frame of a segment serially — see [§8 Cycle semantics](#cycle-semantics).

Because the sensor is the innermost axis, observation `j` of the flattened
vector lives at sensor `j % num_sensors` regardless of variable, frame or
interval. `sensor_observation_coords` builds the observation coordinates for
distance localization from this (used by both smoother and filter).

Aggregation is configured as `observation.aggregation` in
`configs/assimilation.yaml` (`interval_seconds`, `mode`); a null
`interval_seconds` or a null block means full-resolution assimilation. The
filter scripts ignore it.

### Physical observation likelihood

The three run scripts share `observation.error` in
[`configs/assimilation.yaml`](../configs/assimilation.yaml): an
`ObservationErrorSpec`
([`observation_error.py`](../libs/data-assimilation/src/data_assimilation/observation_error.py))
with instrument std, representation std and a `propagation` policy (passed to
the spec's `aggregation` field by `make_observation_error` until that field is
renamed).

The immutable spec resolves labelled diagonal variances for each window's
actual times (`resolve(raw_observations, operator, aggregation)` →
`ResolvedObservationError` with `covariance_diag`, `std`,
`raw_instrument_std`). Each standard deviation may be scalar or a mapping with
a required `default` and optional `components`, zero-based `sensors`, and
`height_bands` overrides. Height bands use `{min_z, max_z, std}` and half-open
`[min_z, max_z)` bounds. The flattening order remains time-major, with
component blocks and sensors inside each frame.

`propagation: propagate_mean` supports independent frame errors and mean
aggregation. It propagates each raw diagonal covariance through the exact bin
weights: independent variance `σ²` averaged over `m` equally weighted frames
becomes `σ²/m`. `median`, `min`, and `max` aggregation are rejected because
they need a calibrated product likelihood.

`representation_time_model` (default `independent`) sets how the
representation error is correlated in time:

- `independent`: every raw frame has its own representation error, so mean
  aggregation shrinks it like instrument noise (`σ_r²/m`). This is an explicit
  approximation, not evidence about cross-frame residual correlation.
- `persistent`: fully correlated within an aggregation bin and independent
  across bins. A mean over the bin does not shrink it: its std combines
  linearly with the bin weights, `(Σ w_i σ_r,i)²`, which is `σ_r²` for a mean.
  Instrument noise still shrinks as `σ²/m`. Without aggregation (one frame per
  bin) it equals `independent`, and under `propagation: none` the two are the
  same too.

No other time model is accepted: a partially correlated one would need a
calibrated temporal covariance. The model is part of the
`observation_error_model` provenance string in `run_info.yaml`.

`propagation: none` keeps both configured standard deviations unchanged for
each averaged observation: the likelihood variance is
`instrument_std² + representation_std²`, regardless of the bin's frame count.
This assigns uncertainty directly to the averaged product; it is not covariance
propagation of independent raw noise. Use the string `none`, not YAML `null`.
Temporal averaging itself remains controlled by `observation.aggregation`.

Instrument noise generates the synthetic measurements (drawn on the raw
frames). Representation uncertainty contributes to the likelihood only; it is
not added to synthetic truth. `representation_std` is specified in the
observed variable's units at raw-frame resolution. Persistent forecast bias
belongs in model-discrepancy handling, rather than being hidden in an enlarged
observation covariance.

`observation.error` specifies physical covariance. ESMDA `alpha` and filter
`beta` apply algorithmic tempering separately; they do not change synthetic
noise or the reported physical covariance.

The smoother accepts either a diagonal matrix or a variance vector (the latter
stays a vector internally). Use `set_observation_covariance(...)` between
windows, or pass `observation_covariance=...` to one smoother call; the latter
restores the constructor covariance afterward. The filter's optional
`run(observation_covariances=...)` accepts physical variances shaped
`(cycles, frames, obs)` or `(cycles, obs)` for single-frame cycles. It validates
all entries before forecasts and applies the same frame stride as the data.
The constant constructor covariance remains the default.

---

## 3. Interpolation

**File:**
[interpolation.py](../libs/data-assimilation/src/data_assimilation/interpolation.py)

`interpolate_dataarray_at_points(data_array, *, x_dim, y_dim, z_dim, obs_x,
obs_y, obs_z)` performs trilinear interpolation of a 3D `xarray.DataArray`
at paired sensor points. It:

1. Resolves staggered-grid dimension aliases (`xt/xm → x`, `yt/ym → y`,
   `zt/zm → z`) via `_resolve_axis_dim_name`.
2. Clips sensor positions to a half-cell extrapolation margin so that
   sensors placed slightly outside the grid (as can happen with staggered
   face-centred velocities) still interpolate cleanly.
3. Carries any non-spatial dimensions (e.g. `time`) through to the output
   so the returned `DataArray` has shape `(..., sensor)`.

---

## 4. Smoothing — base classes

**File:**
[smoothing/base.py](../libs/data-assimilation/src/data_assimilation/smoothing/base.py)

`BaseSmoothing` holds `observation_operator` and `forward_model` and provides:

- `_forecast_step(state, params)` — calls
  `forward_model.run_ensemble(state=state, params=params)`, returning an
  `xarray.Dataset` (in-memory) or `None` (on-disk).
- `_observation_step(state, results_dir)` — if a state dataset is provided,
  applies the observation operator directly; otherwise opens `state_*.nc`
  files from `results_dir` sorted by member index, applies the operator per
  member, and stacks the results. Either way the result goes through
  `_get_observations` (optional aggregation + flatten), shape `(N_e, N_d)`.
- `__call__` delegates to `_analysis` (abstract in this class).

---

## 5. ESMDA variants

**File:**
[smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py)

### `_BaseESMDA`

Subclasses `BaseSmoothing`. Holds `C_D` (diagonal observation-error
covariance), `num_steps`, `alpha`, `rng_key`, and an optional `localization`.

**Alpha tempering.** The default `alpha = num_steps` satisfies
`sum_i (1/alpha_i) = 1` for the equal-weight schedule, and that is the only
scalar schedule accepted: the constructor rejects `num_steps / alpha != 1`
(beyond a `1e-6` slack) because any other value silently conditions on
`L^(num_steps/alpha)` — a different inference problem. `self.alpha` is always
this **base** coefficient.

**Likelihood weight.** `likelihood_weight=w` (default `1.0`, every variant)
makes the whole MDA loop condition on `L^w` instead of `L`: every update runs
with `effective_alpha = alpha / w` (a read-only property), so
`sum_i 1/effective_alpha_i = w`. It exists for the hybrid's `shared_budget`
policy (§9), which derives it — it is not a user knob and not in
`smoother.yaml`. Validated finite `0 < w <= 1`, and the effective covariance
`effective_alpha * C_D` is checked finite in the compute dtype. Explicit
`alpha` arguments (including `_compute_kalman_update(alpha=...)`) are always
base values and are divided by `w` exactly once, inside the update. `C_D`
stays physical. `final_time_smoothing=True` with `w != 1` is rejected (its
extra full-weight update has no allocation). At `w = 1` the path — values and
RNG stream — is the untempered one, bit for bit.

**On-disk mode.** When `forward_model.save_on_disk` is True the constructor
creates `step_0/` through `step_{num_steps}/` under `base_results_dir` and
clears stale `state_*.nc` files. `_set_step_results_dir(i)` redirects the
forward model's output before each forecast. `get_state(ensemble_member,
step)` re-opens the NetCDF for a specific member and step. Disk pruning
(`prune_disk_steps=True`, `keep_prior_disk_step`; the run scripts set both)
caps peak storage at ~2× ensemble size by deleting intermediate step
directories as soon as their Kalman update is computed.

**`_compute_kalman_update`**. Implements the standard ESMDA perturbed-
observation update:

```
C_MD = aug_dev @ pred_obs_dev.T / (N_e - 1)
C_DD = pred_obs_dev @ pred_obs_dev.T / (N_e - 1)
augmented += C_MD @ solve(C_DD + alpha * C_D, perturbed_obs - pred_obs)
```

The body is the shared
[`filtering/analysis.py::stochastic_enkf_update`](../libs/data-assimilation/src/data_assimilation/filtering/analysis.py)
(the ESMDA per-step update *is* the stochastic EnKF analysis with a tempered
`alpha`); this method is a thin wrapper that splits `self.rng_key` and passes
the 1-D variance vector. When `self.localization` is set the shared function
forwards to `localization.localized_update(...)` instead (see §6). Accepts
optional `group_ids`, `localize_mask`, `row_coords`, `obs_coords` forwarded
from the variant.

**Augmentation delegation.** The structure transforms
(`_flatten_state`/`_unflatten_state`, `_flatten_time_varying_params`/
`_unflatten_params`, `_state_group_ids`, `_state_row_coords`,
`_time_varying_group_ids`) are thin wrappers over the shared
[`augmentation.py`](../libs/data-assimilation/src/data_assimilation/augmentation.py)
classes `StateAugmentation` / `ParamAugmentation`, which the filtering
package uses too — one flatten order, one pinning semantics.

**`_analysis` loop (iterated joint update).** Runs `num_steps` iterations.
Each iteration:

1. `_set_step_results_dir(i)` — point the forward model at `step_{i}/`.
2. `_forecast_step(initial_state, params)` — run the ensemble.
3. `apply_failure_substitutions_to_params(params)` — clone donor params
   into failed members before the Kalman update.
4. `_one_step(params, obs, state)` — subclass-specific update; returns
   `(updated_state_or_None, updated_params)`.
5. For **state-bearing variants**: feed `updated_state` back as
   `initial_state` for the next iteration, so the analyzed IC actually
   propagates forward. `apply_failure_substitutions_to_state` then repairs
   failed slots in the IC. For **parameter-only variants**: `_one_step`
   returns `None`; `initial_state` stays pinned at the caller's value.

After the loop one final `_forecast_step(initial_state, params)` produces
the posterior forecast (written to `step_{num_steps}/`); `final_forecast=False`
skips it (used by the hybrid, §9). An optional `_final_time_smoothing_step`
follows (no-op in the base; overridden by the state-bearing variants when
`final_time_smoothing=True`).

**Hidden-state replay (opt-in backend protocol).** An ensemble advertising
`forecast_window_replay_enabled` captures its member checkpoints at window
entry, restores them before every forecast, and commits only after the posterior
forecast and postprocessing succeed. An exception or `final_forecast=False`
rolls the backend back to its pre-window inputs. Analyzed initial conditions
and the current parameter vector are injected after restoration. This prevents
intermediate endpoints from becoming hidden inputs to subsequent MDA iterations.
Backends without this capability retain their existing behavior and RNG streams.
With replay enabled, final-forecast donor substitutions also update the returned
parameter ensemble and its final history entry to match the accepted forecast.
The first implementation is uDALES with enabled discrepancy; see
[pyudales §4.2](pyudales.md#42-window-checkpoints-for-repeated-forecasts).
Replay does not checkpoint the smoother's RNG for job recovery.

**Global parameters and SGS coefficients.** `global_parameter_names` is generic
localization metadata on the smoothers and filters: declared static parameter
rows use the global update, including when state rows use correlation or
distance localization. The default empty tuple preserves existing localization
and RNG behavior. Parameter-only ESMDA accepts distance localization when every
parameter is explicitly global. The run scripts declare the uDALES
SGS-discrepancy coefficients `sgs_bias_b0/b1/b2` global when the assimilation
model's `model_discrepancy` is enabled (`parameter_names` in
`helper_functions.py`). The coefficients are ordinary static prior parameters,
estimated by naming them in `assimilation.params_to_estimate`; static
coefficients carry their analyzed values to the next window without process
noise. The config rules are in
[scripts_and_configs.md §1.7](scripts_and_configs.md).

**Observation-space diagnostics (opt-in).** Set `collect_obs_diagnostics =
True` after construction and the smoother records, in `pred_obs_history`, the
`(N_d, N_e)` predicted observations it materializes at every iteration —
`num_steps + 1` entries per `_analysis` call, entry 0 the prior forecast and
entry −1 the posterior forecast (only `num_steps` entries with
`final_forecast=False`). The list is rebound at `_analysis` entry, so a
multi-window caller reads one window's entries per call. With
`final_time_smoothing=True` the last entry is pre-smoothing. Off by default.
`run_smoother.py` and `run_hybrid.py` turn it on and persist the arrays per
window (`window_{w}_obs.nc`).

### Five variants

| Class | Augmented state | Notes |
|---|---|---|
| `ParameterESMDA` | Parameters only (scalar per ensemble member) | Each parameter is its own localization block |
| `TimeVaryingParameterESMDA` | Time-varying params flattened to `{name}_{t}` scalars | Knots of one parameter share a block (`ParamAugmentation.group_ids`); `pin_initial_time_point` fixes `t=0` (the runners set it from window 1 on, for continuity) |
| `StateESMDA` | `time=0 state` | Parameters are supplied to forecasts but held fixed; optional `state_reduction` + `final_time_smoothing` |
| `StateAndParameterESMDA` | `[time=0 state | static params]` | Strategy-aware localization via `localize_mask`; optional `state_reduction` + `final_time_smoothing` |
| `StateAndTimeVaryingParameterESMDA` | `[time=0 state | {name}_{t} scalars]` | MRO combines both parents: state flattening from `StateAndParameterESMDA`, param flattening from `TimeVaryingParameterESMDA` |

The entries of
[`configs/assimilation_settings/smoother.yaml`](../configs/assimilation_settings/smoother.yaml)
map `static`, `dynamic`, `state`, `state_and_parameter`, `state_and_dynamic`
onto these classes in table order (`static` → `ParameterESMDA`,
`dynamic` → `TimeVaryingParameterESMDA`, …).

**State flatten/unflatten.** `_flatten_state` iterates variables in sorted
order, transposes each to `(ensemble, ...)`, and stacks columns.
`_unflatten_state` reverses this. The sorted-variable order is critical —
it must match between flatten and unflatten and between `_flatten_state`
and `_state_row_coords`. `_get_states` selects `time=0` so the augmented
vector holds the window initial condition. `_get_window_states` (no time
selection) feeds the `window_snapshots` basis source for state reduction.

**`_augmented_state_update`.** The shared method for joint state-bearing
variants builds `[states_flat | params_array]`, applies the Kalman update
(global or localized), and splits the result back into updated state and
updated params.

---

## 6. Localization (optional)

**Reference:** Vossepoel et al. (2025, MWR-D-24-0269.1)

**File:**
[localization/base.py](../libs/data-assimilation/src/data_assimilation/localization/base.py)

### `BaseLocalization` contract

Subclasses implement one method:

```python
def inflation_factors(
    self,
    aug_dev: jnp.ndarray,       # (N_aug, N_e) anomalies
    pred_obs_dev: jnp.ndarray,  # (N_d, N_e) anomalies
    row_coords: Optional[jnp.ndarray] = None,   # (N_aug, 3)
    obs_coords: Optional[jnp.ndarray] = None,   # (N_d, 3)
) -> jnp.ndarray:               # (N_aug, N_d)
```

Return convention: `1.0` = keep observation at full weight; `> 1` = taper
(error variance inflated by `E_inf²`); `jnp.inf` = exclude. The class
attribute `requires_coordinates: bool` tells the smoother whether to
compute `row_coords` and `obs_coords`.

### `taper_inflation`

Shared taper (Vossepoel Eqs. 9–10) used by both strategies. Takes an
abstract `distance`, `truncation`, `tapering_beta` (`beta ∈ (0, 1)`: the
fraction of `truncation` left un-tapered), and `max_inflation` (reached at
`distance == truncation`). Observations beyond `truncation` get `inf`.

| Strategy | `distance` | `truncation` |
|---|---|---|
| Correlation | `1 − |ρ|` | `1 − ρ_t` |
| Distance | `‖grid_pt − sensor‖` | `localization_radius` |

### `localized_update`

The shared local-analysis entry point. Calls `inflation_factors`, then
`jax.vmap`s `update_row` over the `N_aug` augmented rows, each solving its
own `N_d × N_d` system with only the active (finite-inflation) observations.
Excluded observations are decoupled by zeroing their rows and columns and
placing 1 on the diagonal of `C_DD_alpha`, keeping the shape stable for
`vmap`. Cost: `O(N_aug · N_d²)` — cheap for parameter variants, expensive
for large state-bearing ones.

**Strategy-aware joint localization.** State rows are always localized.
Correlation localization also localizes parameter rows because it needs no
spatial coordinates. Distance localization sets parameter rows to
`localize_mask=False`; those rows receive all-ones inflation and therefore the
exact global update.

**Grid-block joint analysis** (`block_grouping`, Vossepoel §3b). When
`block_grouping=True` (on the localization instance), `_group_inflation`
takes the per-observation minimum inflation across all rows in a block so
they share one active-observation set and one transition matrix. Parameter
blocks group the time knots of one parameter (unless
`group_parameter_knots=False`, below); state blocks group truly
co-located grid cells (`StateAugmentation.group_ids`: `u/v/w` share blocks on
pylbm's collocated grid but not on the staggered uDALES/PALM grids).
Masked/global rows are excluded from the block minimum, then restored to
all-ones inflation.

**Knot-wise temporal localization** (`group_parameter_knots`, default `true`).
With one block per parameter, every knot of a time-varying parameter takes the
taper of its most strongly correlated knot, so an early knot is updated by
observations only a later knot explains. `CorrelationLocalization(...,
group_parameter_knots=False)` gives each knot its own block (read in
`TimeVaryingParameterESMDA._time_varying_group_ids`), while `block_grouping`
keeps grouping the co-located `u/v/w` state rows. It has no effect without
`block_grouping` or on static parameters. Distance localization keeps parameter
rows global, so it has no such argument.

That mask-then-group-then-restore ordering is `resolve_row_inflation` in the
same module, and the "is this observation active" predicate is
`active_observations` (`isfinite(E_inf) & (E_inf > 0)` — `E_inf` scales an error
*standard deviation*, so `0` is a singular gain, not a localization decision).
Both are called by `localized_update` **and** by `LETKFAnalysis`, so the
stochastic and deterministic local analyses cannot drift apart on what a
localization strategy means.

### `CorrelationLocalization`

**File:**
[localization/correlation.py](../libs/data-assimilation/src/data_assimilation/localization/correlation.py)

`requires_coordinates = False` — needs only ensemble anomalies. For each
`(augmented_row, observation)` pair:

1. Compute sample correlation `ρ` (with `ddof=1`, matching the `N_e−1`
   covariance denominator).
2. Correlation distance `d_c = 1 − |ρ|`.
3. Exclude when `|ρ| < ρ_t` (`d_c > 1 − ρ_t`); taper the rest with
   `taper_inflation`.

`truncation_correlation=None` defaults to `min(3 / √N_e, 0.99)` (Eq. 6).
Works on **any** smoother variant (parameters have no spatial location, but
ensemble correlations still exist).

### `DistanceLocalization`

**File:**
[localization/distance.py](../libs/data-assimilation/src/data_assimilation/localization/distance.py)

`requires_coordinates = True` — needs the physical grid and sensor
coordinates. For each `(state_row, observation)` pair computes the
Euclidean distance between the grid point and the sensor via the
`|a|² + |b|² − 2a·b` identity to avoid the large `(N_aug, N_d, 3)`
broadcast. When `horizontal_only=True`, only `(x, y)` separation is used.

**Only valid with state-bearing smoothers** (`state` / `state_and_parameter` /
`state_and_dynamic`; or parameter-only ESMDA when every parameter is declared
global) and **coordinate-based observations**.

Defaults for both strategies are in
[localization.yaml](../configs/assimilation_settings/localization.yaml).

---

## 7. Reduced SVD/KL state update (optional)

**File:**
[reduction.py](../libs/data-assimilation/src/data_assimilation/reduction.py)

`OnlineStateReduction` replaces the raw state rows in the augmented Kalman
vector with reduced SVD/KL coefficients. In ESMDA the basis is **refitted every
iteration** from the current forecast ensemble.

**API:**
- `fit(snapshots_flat)` — thin SVD of anomaly matrix; retains the smallest
  rank `r` whose cumulative `∑ σ_i²` reaches `energy_fraction`. Always
  capped by the number of nonzero singular values and by `max_rank` (if set).
  Snapshots containing non-finite values are rejected. On this whitened path
  the numerical-rank cut is `eps · max(shape)`, because `encode` divides by
  `σ` — see §8 for the non-whitened variant.
- `encode(states_flat)` — `Σ_r⁻¹ Φ_r^T (u − ū)` — whitened coefficients
  `(r, N_e)`.
- `decode_increment(d_xi)` — `Φ_r Σ_r d_xi` — maps a coefficient increment
  back to `(N_s, N_e)`. Applied as `u += decode_increment(xi_post − xi_prior)`
  so each member's projection residual is preserved.

**`basis_source`:**
- `"initial_condition"` — fits on the flattened `time=0` IC ensemble
  (rank ≤ N_e − 1, exactly whitened).
- `"window_snapshots"` — fits on every output frame of every member (N_e × N_t
  samples; richer basis, approximately whitened; controlled by
  `snapshot_stride`).

**Incompatible with (state) localization** — the state-bearing smoothers
raise if both are set (reduced coefficients are non-local).

**`final_time_smoothing`** (state-bearing variants, in-memory mode, requires a
state reduction). After the main loop, applies one un-tempered (`alpha=1`)
Kalman update of the full window trajectory (all time frames at once) in the
reduced basis. Reuses the final posterior forecast — no extra forward solve.
Parameters are not part of this augmented vector; they are frozen.

---

## 8. Filtering — the sequential EnKF

**Files:**
[filtering/base.py](../libs/data-assimilation/src/data_assimilation/filtering/base.py),
[filtering/analysis.py](../libs/data-assimilation/src/data_assimilation/filtering/analysis.py),
[filtering/etkf.py](../libs/data-assimilation/src/data_assimilation/filtering/etkf.py),
[filtering/parameter_evolution.py](../libs/data-assimilation/src/data_assimilation/filtering/parameter_evolution.py),
[inflation.py](../libs/data-assimilation/src/data_assimilation/inflation.py).
Design record: [plans/implemented/da_filtering_module_plan.md](plans/implemented/da_filtering_module_plan.md).

### Cycle semantics

A filter's unit of work is a **cycle**: forecast the ensemble over one segment
(the forward model's configured horizon), assimilate that segment's
observations into the state *at the end of the segment* and/or the parameters,
and warm-start the next cycle from the analyzed state. Each observation is
consumed exactly once — there is no MDA schedule.

The observation operator is applied to the whole segment, so a cycle carries
`T` time-resolved observation **frames**, and the filter assimilates them
**serially**: one full-weight analysis per frame, in time order — an
asynchronous/serial EnKF. Nothing is aggregated. Per cycle:

- each frame's `(num_sensors × num_states)` vector gets its own analysis of the
  same end-of-segment augmented state, through the ensemble cross-covariances.
  `C_D` is therefore the error covariance of **one frame**, not of a segment;
- the *other* frames' predicted observations are appended as ride-along rows of
  the augmented matrix, so every analysis updates them too: frame `t` is
  assimilated against predictions that already reflect frames `0…t-1`;
- everything that belongs to the cycle rather than to an observation happens
  **once**: the state-reduction basis fit, prior and posterior inflation
  (posterior inflation relaxes toward the pre-sweep prior anomalies), the
  localization plumbing. `obs_prior_rmse` /
  `obs_posterior_rmse` / `innovation_chi2` are measured on the stacked
  `(T·N_obs,)` system. The `transform_*` / `local_*` diagnostics report the
  **last** frame's transform;
- the RNG splits once per cycle; with `T = 1` that subkey is used directly.

The run scripts use `T = 1`: one cycle is `assimilation.assimilate_every_n_step`
output frames, and a `StridedOperator` wrapper (in `helper_functions.py`) keeps
only the last one. The library's own `assimilate_every_n_step` attribute
(thinning analyses within a multi-frame segment) is left at 1.

### `BaseFilter` / `EnsembleKalmanFilter`

`BaseFilter` owns the cycle loop: forecasting, augmentation, inflation,
parameter evolution, failure substitution, on-disk `cycle_{k}/` management
(mirroring the smoother's `step_{i}/` pattern, with `prune_disk_cycles` /
`keep_first_disk_cycle` knobs) and per-cycle diagnostics. The analysis math is
an injected `AnalysisScheme` — a pure function of arrays; `EnsembleKalmanFilter`
is `BaseFilter` composed with the default `StochasticEnKFAnalysis`. Update
flavors are `AnalysisScheme` implementations, not new filter classes.

```python
enkf = EnsembleKalmanFilter(
    observation_operator=obs_op, forward_model=ensemble_model,
    C_D=variance_vector,            # 1-D (N_obs,) per-FRAME variances
    mode="joint",                   # "state" | "parameter" | "joint"
    localization=None, inflation=RTPS(0.6), parameter_evolution=None,
)
result = enkf.run(state=None, params=prior_params,
                  # one ("time", "obs") DataArray per cycle (T frames each,
                  # assimilated serially), or a flat (num_cycles, N_obs) array
                  observations=observations,
                  return_history=True)           # -> FilterResult
```

Mode semantics: `"state"` updates the flattened end-of-segment state only
(params, if any, are carried unmodified); `"parameter"` updates the flattened
params only (applied from the next cycle onward) and **requires spread
maintenance** (`parameter_evolution` or `inflation` — the constructor refuses
silently-collapsing configurations, unless the parameter selection is
explicitly empty); `"joint"` updates `[state | params]`. Correlation
localization applies to both blocks, while distance localization applies to
state rows and keeps parameter rows global.

For SGS-coefficient inference, parameter/joint filters use no parameter
evolution (no process noise); keep an inflation scheme for spread maintenance.
Coefficients are constant during a forecast segment and updated for the next
segment. With `return_history=True` and declared global parameters,
`FilterResult.applied_params_history` records the coefficients actually used by
each accepted forecast, after donor substitution; `params_history` records the
initial ensemble and analyzed values.

Other attribute knobs, set after construction: `collect_pred_obs` (record
`pred_obs_history` / `pred_obs_post_history`) and `collect_forecast_frames`
(fill `FilterResult.forecast_history` with every output frame of every
segment — the free-running forecast between analyses).

**Beta tempering.** `beta` (default `1.0`, config `filtering.beta`) multiplies
the observation-error covariance of **every** analysis: `R_filter = beta R`, so
`K = C_xy (C_yy + beta R)^(-1)` and the likelihood is `L^(1/beta)`. It must be
a finite real `>= 1` (`validate_beta`; booleans, NaN and infinity are
rejected). `C_D_diag` stays the physical per-frame variance;
`effective_C_D_diag = beta * C_D_diag` is the only covariance handed to the
analysis, derived once, so repeated `run()` calls cannot compound it; the same
holds for a covariance replaced with `set_observation_covariance` or passed as
`run(..., observation_covariances=...)`. The stochastic kernel draws
perturbations with std `sqrt(beta) sigma`, and ETKF/LETKF whiten by it
(`R_eff = E_inf**2 * beta * R` under localization) — beta is never also passed
as the kernel's `alpha`. `beta` is read-only after construction. The
innovation chi2 and the saved `obs_std` stay **physical**. Beta is distinct
from ensemble-spread inflation and from localization's `tapering_beta`;
`beta = 1` is the untempered filter, bit for bit.

### Analysis schemes

Selected with `'filtering.analysis=${analysis.<name>}'`
([analysis.yaml](../configs/assimilation_settings/analysis.yaml)):

| Option | Class | Localization | What it changes |
|---|---|---|---|
| `stochastic` (default) | `StochasticEnKFAnalysis` | optional | Perturbed observations |
| `etkf` | `ETKFAnalysis` | **forbidden** | One global deterministic transform per cycle |
| `etkf_tsvd` | `ETKFAnalysis` | **forbidden** | …plus observation-space truncation |
| `letkf` | `LETKFAnalysis` | **required** | One transform per local block |
| `letkf_tsvd` | `LETKFAnalysis` | **required** | …plus per-block truncation |

> **Status.** The ensemble-transform schemes are tested but **not yet
> benchmarked**: no accuracy, memory or speed claim is made. The campaign
> template [plans/rejected/filtering_ensemble_transform_benchmark.md](plans/rejected/filtering_ensemble_transform_benchmark.md)
> was never run and has been dropped.

`StochasticEnKFAnalysis` draws perturbed observations, sharing its
implementation with the ESMDA smoother's per-step update, so the posterior
sample covariance equals the Kalman covariance only *in expectation over the
draw* — at `N_e = 50` that sampling noise is not small. The ensemble-transform
family
([filtering/etkf.py](../libs/data-assimilation/src/data_assimilation/filtering/etkf.py))
removes the draw: it computes one ensemble-space weight matrix and
right-multiplies the forecast anomalies with it, giving the Kalman covariance
exactly, given the sample forecast moments. It is deterministic (`rng_key` is
accepted and ignored).

**The kernel** (`ensemble_transform`, Hunt et al. 2007 in SVD form). With
`N = N_e`, `R_eff = diag(E_inf² · C_D)` and the whitened anomalies/innovation
`Y_w = R_eff^{-1/2} (pred_obs − mean)`, `d_w = R_eff^{-1/2} (obs − mean)`, take
the thin SVD `Y_w = U S Vᵀ`:

```
C    = (N-1) I + Y_wᵀ Y_w
W_a  = sqrt(N-1) · C^{-1/2}  =  I + V diag( sqrt((N-1)/((N-1)+s²)) - 1 ) Vᵀ
wbar = C^{-1} Y_wᵀ d_w       =      V diag( s / ((N-1)+s²) ) Uᵀ d_w
```

and the posterior is `mean + X @ (wbar 1ᵀ + W_a)`, with `X` the raw forecast
anomalies. Three properties are contracts:

* **The square root is the symmetric one.** Any `W_a Q` with orthogonal `Q`
  gives the same covariance, but RTPP blends posterior against *prior*
  anomalies member by member, and a rotated root scrambles member identity.
* **Mean preservation is structural.** `Y_w 1 = 0` gives `Vᵀ 1 = 0` and
  `W_a 1 = 1`, so the anomaly transform never moves the mean, including under
  truncation.
* **The update is pure right-multiplication.** The weights depend only on
  `(pred_obs, obs, C_D)`, which is what lets `BaseFilter` re-run the analysis on
  the reduction's small coordinate array (below) and makes the state reduction
  compose for free.

The transform is stored factored (`ObservationTransform`: mean weights, modes
`V`, anomaly scales) and applied as `X + (X V) diag(scale - 1) Vᵀ`, so the
dense `N_e × N_e` matrix is never a required intermediate.

`ETKFAnalysis` applies one global transform to every augmented row (state,
parameters and appended observation rows alike) and therefore supports all
three modes and the filtering state reduction. `LETKFAnalysis` computes one
transform per *distinct local observation selection*, from the same
`inflation_factors` the stochastic localized update uses, so switching between
the two changes the estimator and nothing about what a localization radius
means. It deduplicates blocks on the canonical per-row **inflation vector**,
not on `group_ids` — on a staggered grid `pres`/`u`/`v`/`w` each carry their
own grid signature, so `group_ids` dedup collapses nothing while
inflation-vector dedup does. Blocks with no active observation are partitioned
out host-side and returned unchanged.

`AnalysisScheme.localization_policy` (`optional` | `forbidden` | `required`) is
validated in `BaseFilter.__init__` (and by `check_config`), so a mismatch fails
before the first forecast instead of silently running a global update under a
localized config name. It also rules out LETKF plus state reduction:
`BaseFilter` refuses a state reduction together with any localization.

**Observation TSVD.** `ObservationTSVD` is nested on the analysis object
(`ETKFAnalysis(tsvd=ObservationTSVD(...))`). It truncates weak *linear
combinations* of the whitened predicted-observation anomalies `Y_w`; it never
modifies the physical observation-error variances. Knobs: `enabled` (off by
default), `energy_fraction`, `max_rank` (rejected with `enabled=false`), and a
`numerical_tolerance` relative singular-value floor that applies even when
`enabled=false`. `energy_fraction` cuts on the **suffix** — retain the smallest
prefix whose discarded tail holds at most `1 - energy_fraction` of the squared
spectrum — because a float32 cumulative prefix sum saturates at 1.0 inside the
traced LETKF block loop. With the TSVD off every thin-SVD direction is kept:
the weights `s/((N-1)+s²)` and `sqrt((N-1)/((N-1)+s²))` are damped as
`s → 0`, never a `1/s` amplification. Truncation is a retention mask over the
fixed rank `min(N_d, N_e)`, which lets the LETKF block loop batch blocks with
different active-observation counts. The order is fixed:

```text
localize -> form R_eff -> whiten Y -> TSVD -> ensemble transform
```

Observation TSVD (observation axis), the filtering state reduction (state
rows) and localization (which observations reach a block) do not substitute
for each other. Both TSVD options stay off by default: with the shipped sensor
network (`N_d ~ 12` globally, fewer per local block) there is little to
regularize.

```bash
# Global deterministic update (ETKF forbids a localization).
python scripts/run_filtering.py params@prior_params=static filtering.mode=state \
  'filtering.analysis=${analysis.etkf}' 'filtering.localization=${localization.none}'
# Localized deterministic update.
python scripts/run_filtering.py params@prior_params=static filtering.mode=state \
  'filtering.analysis=${analysis.letkf}' 'filtering.localization=${localization.distance}'
```

### Filtering state reduction

`BaseFilter` optionally accepts a `state_reduction`. It is an analysis-space
projection, not a reduced forecast: every member still runs through the full
CFD model, and predicted observations still come from the full forecast
segment. At each cycle the basis input is the ensemble of **final forecast
states**, centered across members. In `mode="state"` that state block is
replaced by modal coefficients; in `mode="joint"` scalar parameter rows remain
in their existing full representation beside the coefficients. Reduction is
invalid in `mode="parameter"` and with any localization; both fail at
construction.

`svd_current` reuses `OnlineStateReduction` with `whiten=False`:

```text
a = U_r.T @ (x - forecast_mean)
delta_x = U_r @ delta_a
```

Only the coefficient **increment** is decoded and added to each member's full
physical prior, so a zero-gain update preserves projection residuals. With all
nonzero modes retained, the stochastic global update agrees with the unreduced
filter to float32 tolerance. Current rank cannot exceed `ensemble_size - 1`.

`svd_streaming` (`StreamingStateReduction`) updates an incremental basis from
successive final-state anomaly blocks without storing historical fields. Its
unnormalized accumulator is `C_k = lambda C_{k-1} + B_k B_k.T`; for
`lambda < 1` the old-block half-life is `log(0.5) / log(lambda)` cycles. Bound
it with `max_rank`: with `energy_fraction` alone the retained rank grows every
cycle. `update_every_n_cycles` can reuse the basis between scheduled updates;
the basis is in-memory run state, not a restart checkpoint.

On this non-whitened path, numerical rank is cut relative to `sigma_max` at
`eps * min(N_s, N_samples)` rather than `eps * max(shape)`: in float32, scaling
by a state size of order `1e5` would discard every mode below about one percent
of `sigma_max`. The whitened ESMDA path keeps the `max(shape)` cut because
`encode` divides by the retained singular values. Retained energy is reported
against the *full* spectrum on both paths.

Both strategies accept optional `variable_scales: {variable: positive_scale}`:
each state-variable row is divided by its scale for fitting and encoding, and
the decoded increment is restored to physical units (`null` keeps the
Euclidean flattening). Row expansion is `StateAugmentation.row_scales`.
Inflation and state-spread diagnostics remain in physical state space.

### Cycle diagnostics

`FilterResult` is a plain dataclass (`params`, `state`, optional
`cycle`-concatenated histories, optional `forecast_history`, and
`diagnostics`: one `CycleDiagnostics` per cycle with innovation χ²,
observation-space prior/posterior RMSE, and per-block spreads). The reduction
fields are `None` on the full-space path; reduced cycles also report retained
and available rank, retained energy, projection residual, decoded-increment
norm and discarded fraction, basis/analysis wall time, condition indicator,
whether the basis was rebuilt, and (streaming) subspace drift.
`analysis_time` is recorded on both paths.

The ensemble-transform fields follow the same additive, nullable pattern:

| Group | Filled by | Contents |
|---|---|---|
| `transform_*` | `ETKFAnalysis` | `available_rank`, `retained_rank`, `retained_energy`, `discarded_spectrum_max` for the cycle's global transform |
| `local_*` | `LETKFAnalysis` | `num_blocks` / `num_active_blocks` / `num_updated_rows`, `active_obs_{min,median,max}`, `retained_rank_{min,mean,max}`, `available_rank_max`, `retained_energy_{min,mean}`, `discarded_spectrum_max`, `chunk_size` |

Both are `None` for `StochasticEnKFAnalysis`. `discarded_spectrum_max = 0.0`
means the truncation ran and discarded nothing; `None` means no transform of
that kind ran. With the TSVD off, `available_rank` (bounded by
`min(N_d, N_e - 1)`) is the meaningful rank. `BaseFilter` reads these off the
scheme by attribute name (`last_transform` / `last_diagnostics`), so
`filtering/base.py` never imports `filtering/etkf.py`. The `local_*` summaries
cover active blocks only; `local_num_blocks` counts distinct inflation vectors.
These are the quantities the LETKF resource gate of
[plans/implemented/filtering_state_reduction_and_transforms.md](plans/implemented/filtering_state_reduction_and_transforms.md)
§6 asks for.

`reduction_discarded_increment_fraction` costs nothing extra: the fit already
yields the forecast anomalies' coordinates `C` in the complete basis, and the
EnKF increment is `anomalies @ W`, so running the analysis on the small
`(k, N_e)` array `C` gives `C @ W`, whose rows `[rank:]` are the discarded part
of the update. It is `None` on a streaming cycle whose basis update was
skipped.

The appended predicted-observation rows take a *global, full-space* ride-along
update, so `obs_posterior_rmse_kind` records what `obs_posterior_rmse` means
(`exact` | `unreduced_ride_along` | `unlocalized_ride_along`). Only the global
ETKF without reduction is `exact`; under a reduction or localization (LETKF
included) it is a proxy, **not** `H` applied to the analyzed state. Do not rank
reduced or localized runs on it; score the analyzed state against the truth.

### Spread maintenance

* **Inflation** ([inflation.py](../libs/data-assimilation/src/data_assimilation/inflation.py)):
  `MultiplicativeInflation(factor)` scales forecast anomalies before the
  analysis (the predicted-observation anomalies are scaled consistently);
  `RTPS(alpha)` / `RTPP(alpha)` rescale/blend the posterior anomalies toward
  the prior spread/perturbations after it.
* **Parameter evolution**
  ([filtering/parameter_evolution.py](../libs/data-assimilation/src/data_assimilation/filtering/parameter_evolution.py)):
  the parameters' forecast model between cycles,
  `RandomWalkEvolution(std={name: std})`. The std is per parameter, in its own
  units (e.g. `{inflow_angle: 2.0, velocity_magnitude: 0.05}`); a scalar is
  refused, and names left out get no noise. Without an evolution or inflation,
  an un-inflated parameter ensemble collapses after a few cycles and stops
  learning.

  The evolution runs **right before each forecast except the filter instance's
  first**, not after the analysis. The saved posterior parameters are
  therefore the analyses themselves, and the evolved values are what the next
  forecast uses (recorded in `applied_params_history` when that history is
  on). The "first forecast" is tracked on the instance, so the evolution also
  applies to the first cycle of each later `run()` call: one `run()` and `W`
  windowed calls give the same chain, as do the hybrid's one-cycle calls.

### Run script

[scripts/run_filtering.py](../scripts/run_filtering.py) reads the
`assimilation`, `observation` and `filtering` blocks of
[configs/assimilation.yaml](../configs/assimilation.yaml); `filtering` is the
`EnsembleKalmanFilter` constructor itself, with slots
`analysis|localization|state_reduction|inflation|parameter_evolution`. It needs
a static prior (time-varying priors stay with the smoothers and the hybrid).

The run is **windowed like an ESMDA run** so the two are configured the same
way and produce the same artifact layout: `assimilation.num_windows` windows
of `time.simulation_time` seconds. Cycles are derived: one cycle spans
`assimilation.assimilate_every_n_step` output frames (which must divide the
frames per window) and assimilates the last one. A window is **pure
chunking** — one `run()` call and one set of artifacts per window. The analyzed
state and parameters carry across, `rng_key` is mutated in place (consecutive
`run()` calls continue one stream), and the per-cycle observation noise is
drawn for the whole horizon before the window loop, so 1 window or `W` windows
give identical posteriors. The artifacts are listed in the script's docstring
and in [scripts_and_configs.md §2.1](scripts_and_configs.md).

---

## 9. Filter smoothing — the ESMDA × filter hybrid

[filter_smoothing/base.py](../libs/data-assimilation/src/data_assimilation/filter_smoothing/base.py)
composes the two families above. Per assimilation window:

1. **ESMDA phase** — a *parameter-only* smoother (`ParameterESMDA` or
   `TimeVaryingParameterESMDA`; the state-bearing variants are rejected) runs
   its MDA loop over the whole window as in §5, with `final_forecast=False`:
   the posterior forward pass is skipped and the call returns the updated
   parameter ensemble alone.
2. **Filter phase** — a sequential filter (§8, `mode="state"` or `"joint"`;
   `"parameter"` is rejected) produces the posterior state by filtering the
   window with the ESMDA-estimated parameters. The filter consumes the **raw
   per-frame observations**, never the smoother's aggregated product.

```python
hybrid = FilterSmoothing(smoother=<ParameterESMDA>, filter=<EnsembleKalmanFilter>,
                         tempering=None)  # None = filter_only at filter.beta
result = hybrid.run(state=..., params=prior, observations=[...], return_history=True)
```

`run()` takes one labelled `("time", "obs")` DataArray per filter cycle with
time coordinates on the window clock (seconds); it concatenates them on `time`
for the smoother and hands them to the filter raw. Each collaborator keeps its
own `C_D` (smoother: window-aggregated diagonal; filter: one frame's variances)
and its own tempering weight (below).

How the filter is driven depends on the ESMDA posterior:

* **Static parameters** — one plain `filter.run(state, params=theta,
  observations)` over the window.
* **Time-varying trajectory** — the filter's forward model forecasts one cycle
  at a time and restarts its clock at 0 each cycle, so the hybrid loops the
  segments itself — one single-cycle `filter.run(...)` per segment, boundaries
  from `segment_bounds` — which is numerically identical to one multi-cycle
  call:
  * `mode="state"`: the segment forecast uses `params_for_segment(theta,
    t0, t1)` — the trajectory restricted to the segment, re-based to
    `[0, t1 − t0]`; the parameters ride through the analysis unmodified.
  * `mode="joint"` — *correction on the ESMDA schedule*: the hybrid keeps a
    static correction `c` (initially zero). Segment `k` forecasts with
    `e_k + c` where `e_k = trajectory_values_at(theta, midpoint of segment
    k)`; after the joint update, `c = posterior_k − e_k`. With a static
    `theta` this reduces exactly to standard joint filtering. The forecast
    holds the parameter constant within each segment.

`FilterSmoothingResult` carries `esmda_params` (the MDA posterior — what seeds
the next window's prior), `state` (the filter's analyzed end-of-window frame —
the next window's warm start), `params` (joint mode: the final carried
`e_k + c`; state mode: `None`), the per-cycle `CycleDiagnostics`, and optional
histories (including `applied_params_history`).

Two couplings worth knowing: the ESMDA phase of window `w+1` starts from the
*filtered* state of window `w`; and the joint correction `c` resets at each
window boundary — the ESMDA posterior alone seeds the next prior, and
`window_{w}_filter_params.nc` preserves what the filter had learned.

Entry point: [scripts/run_hybrid.py](../scripts/run_hybrid.py), which reads the
`smoothing`, `filtering` and `hybrid` blocks (smoother restricted to
`static`/`dynamic`) and builds **two** ensemble forward-model stacks — the
smoother's with the window horizon, the filter's with one cycle — so each
collaborator forecasts on its own clock. `assimilation.assimilate_every_n_step`
thins both phases: they share one strided observation operator, so the run
keeps exactly one observation product. `check_config` validates the tempering
settings before the truth is simulated.

An earlier, different filter-smoothing algorithm (an outer ESMDA whose forecast
operator was an inner EnKF pass) was removed; its design record is
[plans/implemented/filter_smoothing_windowed_esmda.md](plans/implemented/filter_smoothing_windowed_esmda.md).

For SGS discrepancy the hybrid uses parameter-only ESMDA and a state-only
filter: each member keeps its ESMDA coefficient vector throughout the window's
filter phase. The uDALES stacks synchronize native carry and clocks at window
entry; see [pyudales §4.2](pyudales.md#42-window-checkpoints-for-repeated-forecasts).
Failure donor substitution across phases is rejected (`failure.policy=raise`
is required). Joint hybrid coefficient updates remain deferred.

### Beta tempering: splitting each observation between the phases

Every raw observation is assimilated **twice** — by the MDA loop and by the
filter — so the hybrid carries a `TemperingPolicy`
([filter_smoothing/tempering.py](../libs/data-assimilation/src/data_assimilation/filter_smoothing/tempering.py)).
Resolve it once with `resolve_tempering_policy(beta, likelihood_allocation)`
(config `filtering.beta`, `hybrid.likelihood_allocation`), build the filter
with `beta=policy.beta` and the smoother with
`likelihood_weight=policy.smoother_weight`, and pass it to
`FilterSmoothing(smoother, filter, tempering=policy)`:

| `likelihood_allocation` | smoother weight `w` | filter | nominal exponent per reused observation |
|---|---|---|---|
| `filter_only` (default) | `1` (full normalized schedule) | `beta R` | `1 + 1/beta` |
| `shared_budget` (opt-in) | `(beta - 1)/beta` | `beta R` | `w + 1/beta = 1` |

`filter_only` at `beta = 1` is the untempered hybrid (and what `tempering=None`
resolves to, at the filter's own beta). It is conservative damping, **not**
likelihood accounting: no finite beta removes the double use. `shared_budget`
splits the unit budget — four steps at beta 2 keep base alpha 4 but run
effective alpha 8 against a `2R` filter (half each); beta 4 gives three quarters
to ESMDA. It requires `beta > 1`, and `w` is computed as `(beta - 1)/beta` to
avoid cancellation near 1. The resolver rejects NaN/inf/`< 1`/bool beta and
checks `beta R`, `effective_alpha` and `effective_alpha R` for overflow (and
`w` for underflow) in the analysis dtype.

The hybrid **validates, never mutates**: at construction and at every `run()`
it requires `filter.beta == policy.beta` and
`smoother.likelihood_weight == policy.smoother_weight`. Under `shared_budget`
it also requires the same observation product in both phases — no smoother
aggregation, `filter.assimilate_every_n_step == 1`, one shared
observation-operator instance (the run script additionally requires
`assimilation.assimilate_every_n_step=1`) — and, before the first forecast,
identical non-time coordinates across batches, strictly increasing frame times,
the smoother's flattened window vector equal to the filter's frames in order,
and the smoother's physical `C_D` diagonal equal to the filter's `C_D_diag`
tiled over those frames. Per-window covariances installed with
`set_observation_covariance` are tempered by the collaborators themselves, so
each window's covariance is scaled exactly once.

**What the policy does not claim.** The allocation is nominal. For the scalar
`x = theta` example with a joint filter, `shared_budget` recovers the
once-conditioned posterior `V^-1 = P^-1 + R^-1` and `filter_only` gives
`V^-1 = P^-1 + (1 + 1/beta) R^-1`
(`tests/data_assimilation/test_hybrid_tempering.py`). But with `mode="state"`
the joint state–parameter covariance is wrong, and parameter-only ESMDA
followed by a state filter is not a factorization of the joint posterior;
nonlinear reforecasting, localization and inflation prevent an exactness claim
in any case. Treat `shared_budget` as a hybrid approximation to be validated
(held-out coverage, proper scores, physical NIS), not as exact Bayesian
accounting.

---

## 10. Configuration

Each component's options live in one file under
[configs/assimilation_settings/](../configs/assimilation_settings/), and the
`smoothing` / `filtering` blocks of
[configs/assimilation.yaml](../configs/assimilation.yaml) point their slots at
one option by interpolation, e.g. `'smoothing.localization=${localization.correlation}'`.
The file/option/slot table is
[scripts_and_configs.md §1.5](scripts_and_configs.md) and the config checks are
§1.6 there; common overrides are in [configs/README.md](../configs/README.md).
Every smoother entry wires `num_steps`, `alpha` and `localization` from the
`smoothing:` block (`alpha` defaults to `${.num_steps}`, §5). The ESMDA
`likelihood_weight` is deliberately absent: only the hybrid passes it.

---

## 11. End-to-end run

A smoother run uses the library as follows (very brief; see
[scripts/run_smoother.py](../scripts/run_smoother.py) and
[scripts/utils/helper_functions.py](../scripts/utils/helper_functions.py)):

```python
truth_operator = make_observation_operator(cfg, cfg.truth_model.solver_name)
error = make_observation_error(cfg)        # cfg.observation.error
aggregation = make_aggregation(cfg)        # None -> every frame
# Per window, before any ensemble forecast: noisy obs + resolved physical error.
obs, obs_clean, resolved = observe(window_truth, truth_operator, error, aggregation, key)
smoother = instantiate(
    cfg.smoothing.smoother,
    observation_operator=make_observation_operator(cfg, cfg.assim_model.solver_name),
    forward_model=ensemble_model, C_D=resolved.covariance_diag, rng_key=key,
    aggregate_observations=aggregation,
    parameter_names_to_estimate=selected, global_parameter_names=global_names,
)
params_history, states = smoother(
    state=state, params=params, observations=obs,
    observation_covariance=resolved.covariance_diag,
    return_params_history=True,
)
```

The next window starts from the posterior's last state and the posterior
parameters (time-varying ones extrapolated with `next_window_params`).
`run_filtering.py` and `run_hybrid.py` instead draw one observation per cycle
with `cycle_observations`. Truth source and validation sensors are in
[codebase_guide.md §6](codebase_guide.md#6-data-assimilation-flow).

> **pylbm results produced before 2026-08-07 do not carry the state update.**
> The window-to-window state handoff (and the filter's cycle-to-cycle warm
> start) reaches a pylbm solver as an LBM *restart file*, and two bugs there
> were fixed only on 2026-08-07 (PRs #112–#114): a restart-filename width
> mismatch made the solver silently reopen its own previous restart, so
> **every pylbm rollout of a state-bearing smoother discarded the Kalman state
> update at every window boundary**; and every pylbm warm start was rebuilt
> from a pure-equilibrium distribution. Both ran to completion and looked
> healthy. No other backend is affected — but re-check any pylbm ESMDA or
> filtering result from before that date. See [pylbm.md](pylbm.md)
> §"Restart / output filename width", §"Restart record layout" and
> §"A truncated run exits 0".

---

## 12. Extension recipes

### Adding a new ESMDA variant

1. Subclass `_BaseESMDA` in
   [smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py).
2. Override `_one_step(params, obs, state)`. Choose what enters the
   augmented vector, call `self._compute_kalman_update(...)`, return
   `(updated_state_or_None, updated_params)`. Return `None` for state if
   the variant should not propagate the IC forward (parameter-only behavior).
3. Add a new entry to
   [configs/assimilation_settings/smoother.yaml](../configs/assimilation_settings/smoother.yaml)
   with `_target_` pointing at your class and wire `num_steps`, `alpha`,
   `localization` via `${smoothing.*}`. No script changes needed —
   `run_smoother.py` instantiates whatever `cfg.smoothing.smoother` resolves
   to. Teach `scripts/utils/inconsistency_check.py` which priors it pairs with.

### Adding a new localization strategy

1. Subclass `BaseLocalization` in
   [localization/](../libs/data-assimilation/src/data_assimilation/localization/).
2. Implement `inflation_factors(aug_dev, pred_obs_dev, row_coords=None,
   obs_coords=None) -> (N_aug, N_d)`. Return `1.0`/`>1`/`jnp.inf`.
   Reuse `taper_inflation` for the Vossepoel taper (Eqs. 9–10).
3. Set `requires_coordinates = True` if the strategy needs grid/sensor
   geometry; it will then only work with state-bearing smoothers and
   coordinate-based observations.
4. Add an entry to
   [configs/assimilation_settings/localization.yaml](../configs/assimilation_settings/localization.yaml)
   (`<name>: {_target_: ..., ...}`). Select it with
   `'smoothing.localization=${localization.<name>}'`. All smoothers already
   forward `localization: ${smoothing.localization}`.

### Adding a new filter analysis scheme

1. Implement the `AnalysisScheme` interface in
   [filtering/analysis.py](../libs/data-assimilation/src/data_assimilation/filtering/analysis.py)
   (or a sibling module): a pure function
   `(augmented, pred_obs, obs, C_D_diag, rng_key, localization?, ...) ->
   updated augmented`. `BaseFilter` handles everything around it.
2. Declare `localization_policy` (`optional` | `forbidden` | `required`) on the
   class if the default `optional` is wrong. `BaseFilter.__init__` validates
   it. A `forbidden` scheme should also reject a non-`None` `localization` in
   `__call__` for direct callers.
3. Add an entry to
   [configs/assimilation_settings/analysis.yaml](../configs/assimilation_settings/analysis.yaml)
   and select it with `'filtering.analysis=${analysis.<name>}'`. State the
   localization requirement in a comment there and in
   `scripts/utils/inconsistency_check.py`. Nested settings objects are nested
   `_target_` blocks (`_convert_: all` propagates from the `filtering` block);
   see the `etkf_tsvd` entry.

### Adding a new solver to the observation operator

Add a new `elif solver_name == "<name>"` branch to
`ObservationOperator.__init__` in
[observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py)
that defines `self.dim_mapping` for each observed velocity component. See the
backend recipe in [codebase_guide.md §8](codebase_guide.md).

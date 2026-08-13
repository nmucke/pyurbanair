# Data assimilation library reference

Standalone reference for `libs/data-assimilation`. Read this alongside
[codebase_guide.md §6](codebase_guide.md#6-data-assimilation-flow), which
covers how the library is wired into the monorepo. The end-to-end run flow
is kept brief here; the guide is the primary source for it.

---

## 1. Purpose and scope

The library implements ensemble data assimilation in JAX, in two flavors:

* **Smoothing** — Ensemble Smoother with Multiple Data Assimilation (ESMDA):
  per assimilation window, the whole window is re-forecast `num_steps` times
  with tempered (`alpha`-weighted) Kalman updates of the window's initial
  condition and/or parameters (§4–§5).
* **Filtering** — the sequential ensemble Kalman filter (EnKF): per cycle, the
  ensemble is forecast one segment and ONE full-weight analysis updates the
  end-of-segment state and/or parameters, which warm-start the next cycle
  (§8).

Both are **solver-agnostic**: they take any `BaseEnsembleForwardModel` (see
[base_ensemble_forward_model.py](../src/pyurbanair/base_ensemble_forward_model.py))
so the same classes cover pylbm, pyudales, pypalm, and the neural surrogate.
They share one analysis implementation, one augmentation (flatten/unflatten)
layer, and the localization machinery.

All public entry points accept and return `xarray.Dataset`; internal arrays
are `jax.numpy` (JAX-CPU throughout; no JAX GPU use within this library).

Source tree:

```
libs/data-assimilation/src/data_assimilation/
  observation_operator.py   # ObservationOperator, TemporalObservationOperator,
                            #   AggregateObservations, flatten_observations,
                            #   sensor_observation_coords
  interpolation.py          # trilinear grid-to-point interpolation
  reduction.py              # Current and streaming SVD/POD state reduction
  augmentation.py           # ParamAugmentation, StateAugmentation — the
                            #   Dataset <-> flat-array transforms shared by
                            #   smoothing and filtering
  inflation.py              # InflationScheme, MultiplicativeInflation, RTPS, RTPP
  io.py                     # load_dataset, get_sorted_state_files (shared file I/O)
  localization/
    base.py                 # BaseLocalization, taper_inflation, localized_update
    correlation.py          # CorrelationLocalization
    distance.py             # DistanceLocalization
  smoothing/
    base.py                 # BaseSmoothing (_forecast_step, _observation_step)
    esmda.py                # all five ESMDA variant classes
  filtering/
    analysis.py             # stochastic_enkf_update (shared with ESMDA),
                            #   AnalysisScheme, StochasticEnKFAnalysis
    base.py                 # BaseFilter (cycle loop), EnsembleKalmanFilter,
                            #   FilterResult, CycleDiagnostics
    parameter_evolution.py  # ParameterEvolution, IdentityEvolution,
                            #   RandomWalkEvolution
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

The case `obs.yaml` configs use coordinate-based mode; `create_observation_operator`
in
[hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py)
constructs the operator from the `obs.*_points` lists.

**Staggered-grid `dim_mapping`.** Each backend uses different dimension names
for the velocity components. The operator holds a `dim_mapping` dict that maps
`{variable -> {z, y, x} -> dim_name}`:

- `"pylbm"` / `"palm"` (post-processed, regular grid): uniform `x/y/z`
  for `u` and `w`; PALM staggers `u` on `xu` and `v` on `yv`.
- `"udales"` (staggered C-grid): `u` at `(zt, yt, xm)`, `v` at
  `(zt, ym, xt)`, `w` at `(zm, yt, xt)`.

Adding a new backend requires a new `elif solver_name == "..."` branch in
`ObservationOperator.__init__`; the guide's §8 recipe lists this step
explicitly.

`__call__(state)` dispatches to `_observation_single` or
`_observation_ensemble` depending on whether an `ensemble` dim is present.
The flattened vector pattern is `[all_sensors_for_var0, all_sensors_for_var1,
...]`, so the **sensor index is the innermost (fastest) axis within each
variable block**.

### `TemporalObservationOperator`

Wraps `ObservationOperator` and applies it to *every frame* of the window's
`time` dimension. Takes only the base operator; it performs **no
aggregation**. `__call__` returns an `xarray.DataArray` — dims
`("time", "obs")` for a single state, `("ensemble", "time", "obs")` for an
ensemble — carrying the state's seconds-valued `time` coordinate. The `obs`
axis has length `num_sensors * len(obs_states)` with the base operator's
layout (`[all_sensors_for_var0, all_sensors_for_var1, ...]`).

Temporal aggregation is *not* the operator's job anymore: it lives in
`AggregateObservations` and is performed inside the DA classes (see below),
so the raw time-resolved observations stay available to whichever method
wants them.

### `AggregateObservations`

A standalone callable `AggregateObservations(interval_seconds, mode="mean")`
that maps an observation DataArray (dims `(..., "time", "obs")`) to an
aggregated one. Frames are binned by their `time` coordinate (in seconds)
into contiguous `interval_seconds`-wide bins — frame at time `t` belongs to
bin `floor((t - t0) / interval_seconds)` — and reduced within each bin with
`mode` (`"mean"`, `"median"`, `"max"`, `"min"`). The output keeps the dims,
with `time` re-labelled to the interval start times. Bins are **absolute**:
an empty interval raises (a silent gap would misalign the flattened vector),
and the interval count is fixed by the first call — a later call producing a
different count raises, because `C_D` is sized from the first window.

It is an **optional constructor input to the DA classes that assimilate one
batch at a time**: `aggregate_observations=` on the ESMDA smoothers. Each
routes both the real observations and the predicted observations
`H(x)` through one path: optional aggregation, then the module-level
`flatten_observations` helper — a time-major flatten (`("time", "obs") →
(T·num_obs,)`, ensemble inputs to `(N_e, T·num_obs)`), which reproduces the
historical `[interval0 block, interval1 block, ...]` vector layout. With
`aggregate_observations=None` the full time-resolved vector is assimilated as
one batch.

The **sequential filter does not aggregate at all** (`BaseFilter` /
`EnsembleKalmanFilter` take no such argument): it assimilates every frame of a
segment serially, one full-weight analysis each — see [§8 Cycle
semantics](#cycle-semantics).

The physical (x, y, z) position of each observation in the flattened
vector is the sensor location, independent of which variable, frame, or
interval it belongs to — because the sensor is the innermost axis. This
ordering is exploited by `_BaseESMDA._observation_coords` when building
coordinates for distance-based localization (observation `j` lives at sensor
`j % num_sensors`).

Aggregating *observations* (this class) instead of *states* (the old
operator modes) is exactly equivalent for `"mean"` — the operator is linear
in the state — and differs slightly for `median`/`max`/`min`; `mean` is the
default everywhere.

Because aggregation is a DA choice rather than an operator argument, its two
knobs live on the run config's algorithm node — `esmda.interval_seconds` /
`esmda.aggregation_mode`, right next to
`obs_error_std` — not in the case's `obs:` block, which carries only
observation-operator arguments. `run_filtering.yaml` has no such keys, because
the filter aggregates nothing. `create_aggregate_observations` (§11) reads
them; a null `interval_seconds` means full-resolution assimilation.

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
  member, and stacks the results. Returns shape `(N_e, num_obs)`.
- `__call__` delegates to `_analysis` (abstract in this class).

---

## 5. ESMDA variants

**File:**
[smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py)

### `_BaseESMDA`

Subclasses `BaseSmoothing`. Holds `C_D` (diagonal observation-error
covariance matrix, shape `(N_d, N_d)`), `C_D_sqrt`, `num_steps`, `alpha`,
`rng_key`, and an optional `localization`.

**Alpha tempering.** The default `alpha = num_steps` satisfies
`sum_i (1/alpha_i) = 1` for the equal-weight schedule. Any scalar override
is also valid.

**On-disk mode.** When `forward_model.save_on_disk` is True the constructor
creates `step_0/` through `step_{num_steps}/` under `base_results_dir` and
clears stale `state_*.nc` files. `_set_step_results_dir(i)` redirects the
forward model's output before each forecast. `get_state(ensemble_member,
step)` re-opens the NetCDF for a specific member and step. Optional disk
pruning (`prune_disk_steps=True`, `keep_prior_disk_step`) caps on-disk
peak storage at ~2× ensemble size by deleting intermediate step directories
as soon as their Kalman update is computed.

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
`jnp.diag(self.C_D)` under the shared function's 1-D variance-vector
contract. When `self.localization` is set the shared function forwards to
`localization.localized_update(...)` instead (see §6). Accepts optional
`group_ids`, `localize_mask`, `row_coords`, `obs_coords` forwarded from the
variant.

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
the posterior forecast (written to `step_{num_steps}/`). An optional
`_final_time_smoothing_step` follows (no-op in the base; overridden by the
state-bearing variants when `final_time_smoothing=True`).

**`_observation_coords`** (used by distance localization). Tiles the
sensor xyz coordinates so that observation index `j` maps to sensor
`j % num_sensors`, matching the flattened observation vector layout.

**Observation-space diagnostics (opt-in).** Set `collect_obs_diagnostics =
True` after construction (the same attribute-plumbing pattern as
`prune_disk_steps`) and the smoother records, in `pred_obs_history`, the
`(N_d, N_e)` predicted observations it materializes at every iteration —
`num_steps + 1` entries per `_analysis` call, entry 0 the prior forecast and
entry −1 the posterior forecast. The list is rebound at `_analysis` entry, so
a multi-window caller reads one window's entries per call. The extra
`_observation_step` for the posterior forecast runs **before**
`_final_time_smoothing_step` (that step is a second Kalman update of the
trajectory, not a forecast, so its predicted observations are excluded); with
`final_time_smoothing=True` the last entry is therefore pre-smoothing. Off by
default: nothing is recorded and no extra operator evaluation happens.
`run_esmda.py` turns it on under `esmda.save_obs_diagnostics` and persists the
arrays per window.

### Five variants

| Class | Augmented state | Notes |
|---|---|---|
| `ParameterESMDA` | Parameters only (scalar per ensemble member) | Groups time knots by `_group_ids_by_base_name` when block grouping is enabled |
| `TimeVaryingParameterESMDA` | Time-varying params flattened to `{name}_{t}` scalars | `_flatten_time_varying_params` / `_unflatten_params`; `pin_initial_time_point` optionally fixes `t=0` across windows |
| `StateESMDA` | `time=0 state` | Parameters are supplied to forecasts but held fixed; optional `state_reduction` + `final_time_smoothing` |
| `StateAndParameterESMDA` | `[time=0 state | static params]` | `_flatten_state` / `_unflatten_state`; strategy-aware localization via `localize_mask`; optional `state_reduction` + `final_time_smoothing` |
| `StateAndTimeVaryingParameterESMDA` | `[time=0 state | {name}_{t} scalars]` | MRO combines both parents: state flattening from `StateAndParameterESMDA`, param flattening from `TimeVaryingParameterESMDA` |

**Config names** (the `esmda/smoother` group filenames):
`static` → `ParameterESMDA`; `state` → `StateESMDA`;
`dynamic` → `TimeVaryingParameterESMDA`;
`state_and_parameter` → `StateAndParameterESMDA`;
`state_and_dynamic` → `StateAndTimeVaryingParameterESMDA`.

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

Mapping per strategy:

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
they share one active-observation set and one transition matrix. For
`ParameterESMDA` blocks group time knots of the same parameter
(`_group_ids_by_base_name`); for `StateAndParameterESMDA` blocks group
co-located `u/v/w` grid cells (`_state_group_ids`). Masked/global rows are
excluded from the block minimum, then restored to all-ones inflation.

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
   covariance denominator, to give the exact sample correlation).
2. Correlation distance `d_c = 1 − |ρ|`.
3. Exclude when `|ρ| < ρ_t` (`d_c > 1 − ρ_t`); taper the rest with
   `taper_inflation`.

`truncation_correlation=None` defaults to `min(3 / √N_e, 0.99)` (Eq. 6).
Config defaults: `truncation_correlation=0.35`, `tapering_beta=0.5`,
`max_inflation=8.0`, `block_grouping=True`.

Works on **any** smoother variant (parameters have no spatial location, but
ensemble correlations still exist). The parameter-only smoothers receive it
without modification.

### `DistanceLocalization`

**File:**
[localization/distance.py](../libs/data-assimilation/src/data_assimilation/localization/distance.py)

`requires_coordinates = True` — needs the physical grid and sensor
coordinates. For each `(state_row, observation)` pair computes the
Euclidean distance between the grid point and the sensor via the
`|a|² + |b|² − 2a·b` identity to avoid the large `(N_aug, N_d, 3)`
broadcast. When `horizontal_only=True`, only `(x, y)` separation is used.

**Only valid with state-bearing smoothers** (`state` / `state_and_parameter` /
`state_and_dynamic`) and **coordinate-based observations** (the operator
must have `obs_x/obs_y/obs_z` to supply `_observation_coords`).

Config defaults: `localization_radius=10.0`, `tapering_beta=0.5`,
`max_inflation=4.0`, `block_grouping=True`, `horizontal_only=False`.

---

## 7. Reduced SVD/KL state update (optional)

**File:**
[reduction.py](../libs/data-assimilation/src/data_assimilation/reduction.py)

`OnlineStateReduction` replaces the raw state rows in the augmented Kalman
vector with reduced SVD/KL coefficients. The basis is **refitted every ESMDA
iteration** from the current forecast ensemble.

**API:**
- `fit(snapshots_flat)` — thin SVD of anomaly matrix; retains the smallest
  rank `r` whose cumulative `∑ σ_i²` reaches `energy_fraction`. Always
  capped by the number of nonzero singular values and by `max_rank` (if set).
  Snapshots containing non-finite values are rejected (previously they
  silently produced a NaN basis); the numerical-rank cut on this whitened path
  is unchanged (`eps · max(shape)`), because `encode` divides by `σ` — see
  §8's filtering notes for the non-whitened variant.
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

**Incompatible with (state) localization** — the constructor of
`StateAndParameterESMDA` raises if both are set (reduced coefficients are
non-local, so neither distance- nor correlation-based state localization
applies).

**`final_time_smoothing`** (`StateAndParameterESMDA` / `StateAndTimeVaryingParameterESMDA`
only, in-memory mode). After the main loop, applies one un-tempered (`alpha=1`)
Kalman update of the full window trajectory (all time frames at once) in the
reduced basis. Reuses the final posterior forecast — no extra forward solve.
Parameters are not part of this augmented vector; they are frozen.

For theory and implementation notes, link: `docs/reduced_state_da.md` (does
not yet exist as a standalone file; the reduction code and the SVD config
comments in
[conf/esmda/state_reduction/svd.yaml](../conf/esmda/state_reduction/svd.yaml)
are the canonical references until that file is written).

---

## 8. Filtering — the sequential EnKF

**Files:**
[filtering/base.py](../libs/data-assimilation/src/data_assimilation/filtering/base.py),
[filtering/analysis.py](../libs/data-assimilation/src/data_assimilation/filtering/analysis.py),
[filtering/parameter_evolution.py](../libs/data-assimilation/src/data_assimilation/filtering/parameter_evolution.py),
[inflation.py](../libs/data-assimilation/src/data_assimilation/inflation.py).
Design record: [docs/temp/da_filtering_module_plan.md](temp/da_filtering_module_plan.md).

### Cycle semantics

A filter's unit of work is a **cycle**: forecast the ensemble over one segment
(the forward model's configured horizon), assimilate that segment's
observations into the state *at the end of the segment* and/or the parameters,
and warm-start the next cycle from the analyzed state. Each observation is
consumed exactly once — there is no MDA schedule.

The observation operator is applied to the whole segment, so a cycle carries
`T` time-resolved observation **frames** (one per output frame), and the filter
assimilates them **serially**: one full-weight (`alpha = 1`) analysis per
frame, in time order — an asynchronous/serial EnKF. Nothing is aggregated
(unlike the smoothers, the filter takes no `aggregate_observations`).

What that means concretely, per cycle:

- each frame's `(num_sensors × num_states)` vector gets its own analysis of the
  same end-of-segment augmented state, through the ensemble cross-covariances.
  `C_D` is therefore the error covariance of **one frame**, not of a segment;
- the *other* frames' predicted observations are appended as ride-along rows of
  the augmented matrix, so every analysis updates them too: frame `t` is
  assimilated against predictions that already reflect frames `0…t-1`, and the
  observation-space posterior diagnostics cover every frame;
- everything that belongs to the cycle rather than to an observation happens
  **once**: the state-reduction basis fit, prior and posterior inflation
  (posterior inflation relaxes toward the pre-sweep prior anomalies), the
  localization plumbing, and the parameter evolution. `obs_prior_rmse` /
  `obs_posterior_rmse` / `innovation_chi2` are measured on the stacked
  `(T·N_obs,)` system with `tile(C_D, T)`; `obs_posterior_rmse_kind` keeps its
  meaning unchanged (the appended rows still ride along). The `transform_*` /
  `local_*` diagnostics report the **last** frame's transform;
- the RNG splits once per cycle; with `T = 1` that subkey is used directly, and
  only a genuine sweep splits it into per-frame keys.

`T = 1` — one frame per cycle, and every flat `(num_cycles, N_d)` observation
array — is the degenerate case: exactly the single full-weight analysis per
cycle the filter has always applied, bit for bit.

### `BaseFilter` / `EnsembleKalmanFilter`

`BaseFilter` owns the cycle loop: forecasting, augmentation, inflation,
parameter evolution, failure substitution, on-disk `cycle_{k}/` management
(mirroring the smoother's `step_{i}/` pattern, with `prune_disk_cycles` /
`keep_first_disk_cycle` knobs) and per-cycle diagnostics. The analysis math is
an injected `AnalysisScheme` — a pure function of arrays; `EnsembleKalmanFilter`
is `BaseFilter` composed with the default `StochasticEnKFAnalysis`. Update
flavors are `AnalysisScheme` implementations, not new filter classes: the
deterministic ETKF/LETKF (see [Analysis schemes](#analysis-schemes)) ship as
schemes, and a particle-style update would be another.

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
silently-collapsing configurations); `"joint"` updates `[state | params]`.
Correlation localization applies to both blocks, while physical-distance
localization applies to state rows and keeps parameter rows global. Localization
strategies are reused from `localization/` unchanged; distance-based strategies
need state rows.

### Analysis schemes

Selected with `filtering/analysis=<name>` (§1.8 of
[scripts_and_configs.md](scripts_and_configs.md)). For a short standalone
orientation to the deterministic family — the kernel, what separates the four
variants, and what is not yet measured — see
[ensemble_transform_filters.md](ensemble_transform_filters.md):

| Option | Class | Localization |
|---|---|---|
| `stochastic` (default) | `StochasticEnKFAnalysis` | `optional` |
| `etkf` / `etkf_tsvd` | `ETKFAnalysis` | `forbidden` |
| `letkf` / `letkf_tsvd` | `LETKFAnalysis` | `required` |

`StochasticEnKFAnalysis`
([filtering/analysis.py](../libs/data-assimilation/src/data_assimilation/filtering/analysis.py))
draws perturbed observations, sharing its implementation with the ESMDA
smoother's per-step update. The ensemble-transform family
([filtering/etkf.py](../libs/data-assimilation/src/data_assimilation/filtering/etkf.py))
instead computes an ensemble-space weight matrix and right-multiplies the
forecast anomalies with it, so the posterior sample covariance is the Kalman
covariance exactly (given the sample forecast moments) rather than in
expectation over the perturbation draw. The transform uses the *symmetric*
square root — the unique mean-preserving root, and the one that keeps member
identity intact so RTPP still blends a member against its own prior
perturbation. It is stored factored, as `(V, scale, wbar)` applied via
`X + (X V) diag(scale - 1) V^T`, so the dense `N_e x N_e` matrix is never a
required intermediate. `ETKFAnalysis` applies one global transform per cycle to
every augmented row and therefore supports all three modes and the filtering
state reduction; `LETKFAnalysis` computes one transform per state block from
that block's locally selected observations.

`LETKFAnalysis` deduplicates those blocks on the canonical per-row **inflation
vector**, not on `group_ids`: the transform is a function of
`(pred_obs, obs, C_D, E_inf_row)` alone, so rows in different `group_ids` blocks
that see the same observation selection share one solve. That distinction is not
cosmetic on a staggered grid — `pres`/`u`/`v`/`w` each carry their own grid
signature there, so `group_ids` dedup collapses nothing while inflation-vector
dedup does. Blocks with no active observation are provably identity and are
partitioned out host-side rather than solved. The per-cycle counts are in
`cycle_diagnostics.yaml` (`local_*`, below).

`AnalysisScheme.localization_policy` (`optional` | `forbidden` | `required`) is
a declarative contract validated in `BaseFilter.__init__`, so a mismatch fails
before the first forecast instead of silently running a global update under a
localized config name. `forbidden` is why a localized deterministic analysis is
the explicit `LETKFAnalysis` class rather than a flag on the ETKF. It also
settles LETKF-plus-state-reduction structurally: `BaseFilter` already refuses a
state reduction together with any localization, and LETKF requires one.

**Observation TSVD.** `ObservationTSVD` is nested on the analysis object
(`ETKFAnalysis(tsvd=ObservationTSVD(...))`), not a separate filter class,
because it regularizes the transform the analysis already computes. It
truncates weak *linear combinations* of the whitened predicted-observation
anomalies `Y_w = R_eff^{-1/2} Y`; it never modifies the physical observation-
error variances. Knobs: `enabled` (off by default), `energy_fraction`,
`max_rank`, and a `numerical_tolerance` relative singular-value floor.

`enabled` gates the **scientific** truncation, which is `energy_fraction` and
`max_rank` together — so `max_rank` with `enabled=false` is rejected at
construction rather than silently ignored (a cap that cannot fire is a config
error, and honouring it would make a block spelled `enabled: false` truncate
anyway). `numerical_tolerance` is *not* gated: it redefines "numerically zero"
rather than making a scientific choice, so it applies in addition to the
scientific cut when that is on and **on its own when it is off**.

`energy_fraction` cuts on the **suffix** — retain the smallest prefix whose
discarded tail holds at most `1 - energy_fraction` of the squared spectrum.
That form is used because the same criterion runs under a trace in the LETKF
block loop, where a float32 cumulative *prefix* sum saturates at 1.0 and would
let dtype decide the rank. `energy_fraction = 1.0` needs no special case in the
suffix form and has none: the tolerated tail is then exactly zero, so the
criterion counts every strictly nonzero direction and the numerical cap reduces
that to the numerically nonzero rank — "retain every numerically nonzero
direction" is what the general path already computes.

Disabled *and* with `numerical_tolerance` unset — which is what both untruncated
groups ship, since their whole `tsvd` node is `null` — the kernel retains
*every* thin-SVD direction rather than truncating at the numerical rank: nothing
in the transform divides by a singular value (`wbar` weights direction `i` by
`s_i / ((N_e-1) + s_i**2)`, `W_a` by `sqrt((N_e-1) / ((N_e-1) + s_i**2))`, both
damped as `s_i -> 0`), so a round-off direction costs essentially nothing while
an over-eager cut discards real information. Truncation is a retention mask over
the fixed rank `min(N_d, N_e)`, never a reshape, which is what lets a LETKF
block loop batch over blocks whose active observation counts differ.

Observation TSVD and the [filtering state reduction](#filtering-state-reduction)
act on **different axes**: the state SVD chooses a basis for the state rows,
the observation TSVD truncates directions of the observation anomalies. The same
holds for TSVD versus localization, which is why their order is fixed:

```text
localize -> form R_eff -> whiten Y -> TSVD -> ensemble transform
```

Localization decides which observations reach a block and how strongly
(`R_eff = diag(E_inf**2 * R)` from `BaseLocalization.inflation_factors`, with
infinite inflation excluding an observation as a zero weight); the TSVD then
decides which directions of that already selected and whitened local matrix to
retain. Neither substitutes for the other. Both TSVD options stay **off by
default**: with the shipped sensor network (`N_d ~ 12` globally, fewer active
per local block) there is little to regularize. Turn them on only when the
logged rank/energy diagnostics show persistent ill-conditioning.

### Filtering state reduction

`BaseFilter` optionally accepts a `state_reduction`. It is an analysis-space
projection, not a reduced forecast: every member still runs through the full
CFD model, and predicted observations still come from the full forecast
segment. At each cycle the basis input is the ensemble of **final forecast
states**, centered across members. In `mode="state"` that state block is
replaced by modal coefficients; in `mode="joint"` scalar parameter rows remain
in their existing full representation beside the coefficients. Reduction is
invalid in `mode="parameter"` and with any localization, because global POD
modes have no physical coordinates; both combinations fail at construction.

The current-cycle strategy reuses `OnlineStateReduction` with `whiten=False`:

```text
a = U_r.T @ (x - forecast_mean)
delta_x = U_r @ delta_a
```

Only the coefficient **increment** is decoded and added to each member's full
physical prior. Consequently, a zero-gain update preserves projection
residuals rather than replacing states by their projections. With all nonzero
current-ensemble modes retained, the stochastic global update agrees with the
unreduced filter to normal JAX float32 tolerance. Current rank cannot exceed
`ensemble_size - 1`.

`StreamingStateReduction` updates an incremental basis from successive finite
final-state anomaly blocks without storing historical fields. Bound it with
`max_rank`: the accumulator keeps absorbing new directions, so an
`energy_fraction` criterion alone lets the retained rank — and the
`(N_s, rank)` basis — grow every cycle until it dwarfs the ensemble it
summarises. The shipped `svd_streaming` group therefore caps the rank at the
ensemble size. Its
unnormalized accumulator is `C_k = lambda C_{k-1} + B_k B_k.T`; the new block
always enters at unit weight, including when `lambda=1`. For `lambda < 1`, the
old-block covariance half-life is `log(0.5) / log(lambda)` cycles. The
accumulator weight is tracked so absolute spectrum diagnostics remain
comparable between cycles. `update_every_n_cycles` can reuse the existing basis
between scheduled updates; the basis itself is in-memory run state and is not
a restart checkpoint.

On this non-whitened path, numerical rank is cut relative to `sigma_max` at
`eps * min(N_s, N_samples)` rather than numpy's `eps * max(shape)`: JAX runs
float32, where scaling by a state size of order `1e5` would discard every mode
below about one percent of `sigma_max` and quietly turn `energy_fraction=1.0`
into a truncated basis. The whitened ESMDA path keeps the conservative
`max(shape)` cut, because `encode` divides by the retained singular values
there and a mode admitted just above the round-off floor is amplified by
`1/sigma`. Retained energy is reported against the *full* spectrum on both
paths, so anything the numerical cut removes shows up as retained energy below
one.

Both strategies accept optional `variable_scales: {variable: positive_scale}`.
Each state-variable row is divided by its scale for fitting and encoding, then
the decoded increment is restored to physical units. Keys must name actual
state variables and values must be finite and positive. `null` preserves the
existing Euclidean flattening (and then skips the row arithmetic entirely);
when conflicting non-empty `units` attributes exist without explicit scales,
the reduction warns rather than guessing a conversion. Row expansion is
`StateAugmentation.row_scales`, so the scale vector and the flattened state
always share one ordering. Inflation and the existing state-spread diagnostics remain
defined in physical state space.

`FilterResult` is a plain dataclass (`params`, `state`, optional
`cycle`-concatenated histories, and `diagnostics`: one `CycleDiagnostics` per
cycle with innovation χ² consistency, observation-space prior/posterior RMSE,
and per-block spreads. Its stable additive reduction fields are `None` on the
full-space path; reduced cycles also report retained and available rank,
retained energy, current-anomaly projection residual, decoded-increment norm
and discarded fraction, basis/total analysis wall time, spectral condition
indicator, normalized spectrum maximum, whether the basis was rebuilt this
cycle, and (for streaming) subspace drift. `analysis_time` is recorded on
*both* paths, so a reduced run's analysis cost is directly comparable with the
unreduced filter's. Physical state spreads keep their original meaning and
never report modal-coefficient spread under the existing names.

The ensemble-transform fields follow the same additive, nullable pattern, so
`cycle_diagnostics.yaml` has one schema for every analysis:

| Group | Filled by | Contents |
|---|---|---|
| `transform_*` | `ETKFAnalysis` | `available_rank`, `retained_rank`, `retained_energy`, `discarded_spectrum_max` for the cycle's one global transform |
| `local_*` | `LETKFAnalysis` | `num_blocks` / `num_active_blocks` / `num_updated_rows`, `active_obs_{min,median,max}`, `retained_rank_{min,mean,max}`, `available_rank_max`, `retained_energy_{min,mean}`, `discarded_spectrum_max`, `chunk_size` |

Both groups are `None` for `StochasticEnKFAnalysis`, which forms no transform.
The per-block energy readouts are computed inside the traced block loop (they
are reductions, not host-side rank queries), so the LETKF reports the same four
mandated TSVD quantities as the global ETKF — available rank, retained rank,
retained energy, discarded spectrum — with the last three summarized over
blocks. The zero/one convention is the same in both groups and is deliberately
distinct from `None`: `transform_discarded_spectrum_max = 0.0` (and
`local_discarded_spectrum_max = 0.0`, with `local_retained_energy_* = 1.0`)
means the truncation ran and discarded nothing, whereas `None` means no
transform of that kind ran at all.

`BaseFilter` reads these off the scheme *by attribute name*
(`last_transform` / `last_diagnostics`), so `filtering/base.py` never imports
`filtering/etkf.py` and a future scheme that publishes neither simply leaves both
groups null. The `local_*` figures are summaries: the per-block arrays behind
them have one entry per block (`N_s`-order in production) and stay on
`LETKFAnalysis.last_diagnostics` rather than being written every cycle. The
rank, energy and active-observation summaries cover the **active** blocks only — a block
with no active observation computes no transform and returns its rows
untouched — with the inactive count recoverable as
`local_num_blocks - local_num_active_blocks`. `local_num_blocks` counts distinct
*inflation vectors*, not distinct `group_ids` (see the deduplication note under
[Analysis schemes](#analysis-schemes)).

These are the per-cycle quantities the LETKF resource gate of
`docs/plans/filtering_state_reduction_and_transforms.md` §6 requires; the
campaign record is
[docs/temp/filtering_ensemble_transform_benchmark.md](temp/filtering_ensemble_transform_benchmark.md),
which is a template with no measurements in it yet.

`reduction_discarded_increment_fraction` costs nothing extra: the fit already
yields the forecast anomalies' coordinates in the complete untruncated basis
(`anomalies = U_full @ C`), and the EnKF increment is `anomalies @ W` for an
ensemble-space weight matrix `W`, so running the analysis on the tiny `(k, N_e)`
array `C` gives `C @ W`, whose rows `[rank:]` are exactly the discarded part of
the update. It is measured in the reduction's own (scaled) norm, and it is
`None` on a streaming cycle whose basis update was skipped by
`update_every_n_cycles` (there is no current coordinate split to report).

Two caveats on the observation-space diagnostics. The appended predicted-
observation rows take a *global, full-space* ride-along update, so under a
reduction `obs_posterior_rmse` is bit-identical to the unreduced filter's and
says nothing about truncation; `obs_posterior_rmse_kind` records this
(`exact` | `unreduced_ride_along` | `unlocalized_ride_along`), and reduced runs
must not be ranked on that value.

The same label covers the analysis schemes, and the LETKF needs no new value:

* global ETKF (`filtering/localization=none`, no reduction) → `exact`. The one
  global transform is applied to every augmented row, observation rows included,
  so the value really is `H` applied to the analyzed state.
* LETKF, like the localized stochastic analysis, → `unlocalized_ride_along`:
  the state rows are updated block-locally while the appended observation rows
  keep the global update. Under that label `obs_posterior_rmse` is a
  global-analysis proxy, **not** `H` applied to the row-wise localized
  posterior. Do not use it to rank ETKF against LETKF; score the analyzed state
  against the truth instead.

### Spread maintenance

* **Inflation** ([inflation.py](../libs/data-assimilation/src/data_assimilation/inflation.py)):
  `MultiplicativeInflation(factor)` scales forecast anomalies before the
  analysis (the predicted-observation anomalies are scaled consistently);
  `RTPS(alpha)` / `RTPP(alpha)` rescale/blend the posterior anomalies toward
  the prior spread/perturbations after it.
* **Parameter evolution**
  ([filtering/parameter_evolution.py](../libs/data-assimilation/src/data_assimilation/filtering/parameter_evolution.py)):
  the parameters' forecast model between cycles — `IdentityEvolution` or
  `RandomWalkEvolution(std | {name: std})`. Without one, an un-inflated
  parameter ensemble collapses after a few cycles and stops learning — so the
  parameter-updating modes (`parameter`/`joint`) refuse to construct without
  an evolution or an inflation.

### Run script

[scripts/filtering/run_filtering.py](../scripts/filtering/run_filtering.py) (config
[conf/run_filtering.yaml](../conf/run_filtering.yaml)) is the entry point:
truth inline or from disk (as run_esmda.py), Hydra groups
`filtering/analysis|localization|state_reduction|inflation|evolution`, static scalar
parameters only (time-varying/AR(2) priors stay with the ESMDA smoothers).

The run is **windowed like an ESMDA run**, and that is the only reason windows
exist here: `filtering.num_assimilation_windows` is configured exactly as
`esmda.num_assimilation_windows` — one window is `time.simulation_time`
seconds and the horizon is `num_assimilation_windows · time.simulation_time` —
so a filtering run and an ESMDA run of the same experiment are configured the
same way and comparable artifact for artifact. **Cycles are derived, not
configured:** the filter forecasts observation to observation, so one cycle is
one observation interval (`time.output_frequency` s) ending in one full-weight
analysis of that single frame (`T = 1` in the serial-sweep machinery of §8.1),
and cycles per window = observation frames per window =
`time.simulation_time / time.output_frequency`. `filtering.assimilate_every_n_step
= n` (default 1) is the one knob over that: every `n`-th observation frame is
assimilated, so a cycle spans `n * time.output_frequency` seconds and a window
holds `n` times fewer of them — the intermediate frames are still simulated and
still written out (the per-cycle `_ensemble_states/cycle_{k}/` segments hold all
`n`), they are just never analyzed, and the analysis is the segment's last
frame; `n` must divide the window's frame count or the run refuses to start.

A window is **pure chunking**: it splits the horizon into one `run()` call and
one set of per-window artifacts each, so RAM and peak disk stay bounded, and it
changes nothing mathematically. The analyzed state and parameters are carried
across a boundary exactly as ESMDA carries its own, `BaseFilter.rng_key` is an
instance attribute mutated in place (so consecutive `run()` calls continue one
stream — `run()` never resets it), and the per-cycle observation noise is
drawn for the whole horizon before the window loop. The same horizon run as 1
window or as `W` windows therefore gives identical posteriors and per-cycle
diagnostics.

See [scripts_and_configs.md](scripts_and_configs.md) §1.8 / §2.1 for the config
groups and the (dual-schema) artifact list.

---

## 9. Filter smoothing — the ESMDA × filter hybrid

[filter_smoothing/base.py](../libs/data-assimilation/src/data_assimilation/filter_smoothing/base.py)
composes the two families above instead of picking one. Per assimilation
window:

1. **ESMDA phase** — a *parameter-only* smoother (`ParameterESMDA` or
   `TimeVaryingParameterESMDA`; the state-bearing variants are rejected) runs
   its MDA loop over the whole window exactly as in §5, but with the new
   `final_forecast=False` seam: the posterior forward pass after the loop is
   skipped, and the call returns the updated parameter ensemble alone. (With
   the default `final_forecast=True` the seam is inert — byte-identical to the
   pre-seam behavior. When skipped, the smoother's `pred_obs_history` holds
   only the `num_steps` pre-update entries — there is **no** posterior entry —
   and no `step_{num_steps}/` directory is written.)
2. **Filter phase** — a sequential filter (§8, `mode="state"` or `"joint"`;
   `"parameter"` is rejected) produces the posterior state by filtering the
   window with the ESMDA-estimated parameters. The filter consumes the **full
   raw per-frame observations**, never the smoother's aggregated product —
   aggregation exists only inside the ESMDA phase, via its own
   `aggregate_observations`, exactly as in a plain ESMDA run.

```python
hybrid = FilterSmoothing(smoother=<ParameterESMDA>, filter=<EnsembleKalmanFilter>)
result = hybrid.run(state=..., params=prior, observations=[...], return_history=True)
```

`run()` takes one labelled `("time", "obs")` DataArray per filter cycle with
time coordinates on the window clock (seconds); it concatenates them on `time`
for the smoother and hands them to the filter raw. Each collaborator keeps its
own `C_D` from construction (smoother: window-aggregated diagonal; filter:
one frame's variances).

How the filter is driven depends on the ESMDA posterior:

* **Static parameters** — one plain `filter.run(state, params=theta,
  observations)` over the window, exactly as `run_filtering.py` uses the
  filter. Nothing hybrid-specific.
* **Time-varying trajectory** — the filter's forward model forecasts one
  observation interval at a time and restarts its clock at 0 each cycle
  (which is also why `run_filtering.py` rejects dynamic priors), so the
  whole-window schedule cannot be handed over as-is. The hybrid loops the
  segments itself — one single-cycle `filter.run(...)` per segment, segment
  boundaries derived from the observation time coordinates — which is
  numerically identical to one multi-cycle call (`run()` never resets
  `rng_key`, and warm starts chain through `result.state`):
  * `mode="state"`: the segment forecast uses `params_for_segment(theta,
    t0, t1)` — the trajectory restricted to the segment (endpoints
    interpolated, interior knots kept, time re-based to segment-local
    `[0, t1 − t0]`); the parameters ride through the analysis unmodified.
  * `mode="joint"` — *correction on the ESMDA schedule*: the hybrid keeps a
    static correction `c` (initially zero). Segment `k` forecasts with
    `e_k + c` where `e_k = trajectory_values_at(theta, midpoint of segment
    k)` (a time-dim-free Dataset, so the filter's static-params contract
    holds); after the joint update, `c = posterior_k − e_k`. ESMDA sets the
    dynamic shape; the filter accumulates a persistent correction. With a
    static `theta` this reduces *exactly* to standard joint filtering. The
    segment **midpoint** is a fixed convention (the best constant
    approximation of the schedule over the segment), not a knob — and note
    the approximation it implies: the forecast holds the parameter constant
    within each segment, which loses within-segment schedule variation when
    `time.seconds_per_knot < time.output_frequency`.

`FilterSmoothingResult` carries `esmda_params` (the MDA posterior — what seeds
the next window's prior), `state` (the filter's analyzed end-of-window frame —
the next window's warm start), `params` (joint mode: the final carried
`e_k + c`; state mode: `None`), the per-cycle `CycleDiagnostics` renumbered
`0..L−1`, and optional histories (`esmda_step`- and `cycle`-concatenated).

Two couplings worth knowing: the ESMDA phase of window `w+1` starts from the
*filtered* state of window `w`, so its model error includes the filter's
increments (intended — that is the point of the hybrid); and the joint
correction `c` deliberately resets at each window boundary — the ESMDA
posterior alone seeds the next prior/extrapolation, and the per-window
`window_{w}_filter_params.nc` artifact preserves what the filter had learned.

Entry point:
[scripts/filter_smoothing/run_filter_smoothing.py](../scripts/filter_smoothing/run_filter_smoothing.py)
(config
[conf/run_filter_smoothing.yaml](../conf/run_filter_smoothing.yaml)), which
composes the existing `esmda/*` and `filtering/*` groups (smoother restricted
to `static`/`dynamic`) plus a small `filter_smoothing:` node, and instantiates
**two** ensemble forward-model stacks — the smoother's with the window
horizon, the filter's with `simulation_time` = one observation interval — so
each collaborator forecasts on its own clock. `filtering.assimilate_every_n_step`
is pinned to 1 here (a stride would give the two phases inconsistent
observation products).

> Historical note: an earlier, different filter-smoothing algorithm (an outer
> ESMDA whose *forecast operator* was an inner EnKF pass, updating the whole
> knot trajectory against stacked per-cycle observations) was removed in
> `0e3291c`; see `docs/plans/filter_smoothing_windowed_esmda.md` for its
> design record. This section describes its replacement.

---

## 10. Configuration

All smoother configuration is via Hydra groups under
[conf/esmda/](../conf/esmda/); the filter's equivalents live under
[conf/filtering/](../conf/filtering/) (see §8 and
[scripts_and_configs.md §1.8](scripts_and_configs.md)).

### `esmda/smoother` group

Five options in
[conf/esmda/smoother/](../conf/esmda/smoother/):

| File | Class | Notes |
|---|---|---|
| `static.yaml` | `ParameterESMDA` | Parameter-only, static scalars |
| `state.yaml` | `StateESMDA` | State-only; static parameters held fixed; wires `state_reduction` / `final_time_smoothing` |
| `dynamic.yaml` | `TimeVaryingParameterESMDA` | Parameter-only, time-varying (AR(2)) |
| `state_and_parameter.yaml` | `StateAndParameterESMDA` | Joint state + static; wires `state_reduction` / `final_time_smoothing` |
| `state_and_dynamic.yaml` | `StateAndTimeVaryingParameterESMDA` | Joint state + time-varying; same reduction knobs |

Every smoother YAML wires shared fields via Hydra interpolation:
```yaml
num_steps: ${esmda.num_steps}
alpha: ${esmda.alpha}
localization: ${esmda.localization}
```
so the `esmda:` block in `run_esmda.yaml` is the single place to change
`num_steps` or `alpha`.

### `esmda/localization` group

Three options in
[conf/esmda/localization/](../conf/esmda/localization/):

| File | Class | Default key params |
|---|---|---|
| `none.yaml` | — (`esmda.localization: null`) | Global update |
| `correlation.yaml` | `CorrelationLocalization` | `truncation_correlation=0.35`, `block_grouping=True` |
| `distance.yaml` | `DistanceLocalization` | `localization_radius=10.0`, `block_grouping=True` |

Select with `esmda/localization=correlation`, or override a field with
`esmda.localization.localization_radius=40`. Force the global update with
`esmda.localization=null`.

### `esmda/state_reduction` group

Two options in
[conf/esmda/state_reduction/](../conf/esmda/state_reduction/):

| File | Class |
|---|---|
| `none.yaml` | — (`state_reduction: null`) |
| `svd.yaml` | `OnlineStateReduction` |

The `state_reduction` key is consumed by `state.yaml`,
`state_and_parameter.yaml`, and `state_and_dynamic.yaml`. Selecting
`esmda/state_reduction=svd` while using a parameter-only smoother is a no-op.

---

## 11. End-to-end run

A run uses the library as follows (very brief; see
[scripts/esmda/run_esmda.py](../scripts/esmda/run_esmda.py),
[codebase_guide.md §6](codebase_guide.md#6-data-assimilation-flow), and
[conf/run_esmda.yaml](../conf/run_esmda.yaml) for the full picture):

```python
# From src/pyurbanair/config/hydra_helpers.py
obs_op  = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)
C_D     = create_C_D(obs_op.num_obs, cfg.esmda.obs_error_std)

# Hydra instantiates the smoother from the esmda/smoother group
esmda = instantiate(cfg.esmda.smoother, observation_operator=obs_op,
                    forward_model=ensemble_model, C_D=C_D, rng_key=rng_key)

# run_esmda.py's window loop (num_assimilation_windows ≥ 1)
for window in range(cfg.esmda.num_assimilation_windows):
    truth_obs = slice_and_noise(truth_state, cfg.esmda.obs_error_std)
    posterior_params, posterior_state = esmda(
        state=prior_state, params=prior_params, observations=truth_obs
    )
    prior_state = posterior_state  # warm-start next window
```

`create_observation_operator` builds a `TemporalObservationOperator`
(time-resolved xarray output) wrapping an `ObservationOperator` using the
case's `obs_x/y/z_points`; `create_aggregate_observations` builds the
optional `AggregateObservations` from the run config's algorithm node
(`esmda.interval_seconds` / `esmda.aggregation_mode` — aggregation is a DA
choice, not an operator argument; `run_filtering.py` has no such keys, its
filter assimilating every frame serially), which the script passes to the DA
class. `create_C_D`
produces the diagonal `σ² I` error covariance, sized from the aggregated
first-window observation vector. The script also constructs validation sensors
(never assimilated; scored as held-out check) and handles inline vs. on-disk
truth; see `codebase_guide.md §6` and the script's docstring.

> **pylbm results produced before 2026-08-07 do not carry the state update.**
> The `prior_state = posterior_state` handoff above (and the filter's
> cycle-to-cycle warm start, §8) reaches a pylbm solver as an LBM *restart
> file*, and two independent bugs there were fixed only on 2026-08-07 (PRs
> #112–#114). Python spelled the restart filename with a 9-digit iteration
> field while the Fortran opened a 6-digit one, so the solver silently reopened
> its own restart from the previous window: for
> `esmda/smoother=state`, `state_and_parameter`, and `state_and_dynamic`, **every pylbm
> rollout discarded the Kalman state update at every window boundary**.
> Separately, the restart *template* read raised into a bare `except`, so every
> pylbm warm start was rebuilt from a pure-equilibrium distribution, discarding
> the non-equilibrium stress. Both ran to completion and looked healthy; a
> truncated member also exited 0 and is only now treated as a failure. Nothing
> in this library changed, and no other backend is affected — but re-check any
> pylbm ESMDA or filtering result from before that date before reading it. See
> [pylbm.md](pylbm.md) §"Restart / output filename width", §"Restart record
> layout" and §"A truncated run exits 0".

---

## 12. Extension recipes

### Adding a new ESMDA variant

1. Subclass `_BaseESMDA` in
   [smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py).
2. Override `_one_step(params, obs, state)`. Choose what enters the
   augmented vector, call `self._compute_kalman_update(...)`, return
   `(updated_state_or_None, updated_params)`. Return `None` for state if
   the variant should not propagate the IC forward (parameter-only behavior).
3. Add a new YAML file to
   [conf/esmda/smoother/](../conf/esmda/smoother/) with `_target_` pointing
   at your class and wire `num_steps`, `alpha`, `localization` via
   `${esmda.*}`. No script changes needed — `run_esmda.py` instantiates
   whatever `cfg.esmda.smoother` resolves to.
4. If the variant needs the flattened field, check
   `isinstance(esmda, StateAndParameterESMDA)` as the script already does
   for all state-bearing branches.

### Adding a new localization strategy

1. Subclass `BaseLocalization` in
   [localization/](../libs/data-assimilation/src/data_assimilation/localization/).
2. Implement `inflation_factors(aug_dev, pred_obs_dev, row_coords=None,
   obs_coords=None) -> (N_aug, N_d)`. Return `1.0`/`>1`/`jnp.inf`.
   Reuse `taper_inflation` for the Vossepoel taper (Eqs. 9–10).
3. Set `requires_coordinates = True` if the strategy needs grid/sensor
   geometry; it will then only work with state-bearing smoothers and
   coordinate-based observations.
4. Add a YAML file to
   [conf/esmda/localization/](../conf/esmda/localization/)
   (`# @package esmda`, setting `localization: {_target_: ..., ...}`).
   Select it with `esmda/localization=<name>`. All smoothers already forward
   `localization: ${esmda.localization}` so no smoother YAML changes are
   needed. `esmda/localization=none` (or `esmda.localization=null`) restores
   the global update.

### Adding a new filter analysis scheme

1. Implement the `AnalysisScheme` interface in
   [filtering/analysis.py](../libs/data-assimilation/src/data_assimilation/filtering/analysis.py)
   (or a sibling module): a pure function
   `(augmented, pred_obs, obs, C_D_diag, rng_key, localization?, ...) ->
   updated augmented`. `BaseFilter` handles everything around it (cycle loop,
   augmentation, inflation, evolution, diagnostics).
2. Declare `localization_policy` (`optional` | `forbidden` | `required`) on the
   class if the default `optional` is wrong — a global-only scheme is
   `forbidden`, a scheme that is meaningless unlocalized is `required`.
   `BaseFilter.__init__` validates it (and the spelling), so an invalid config
   fails at construction. A `forbidden` scheme should also reject a non-`None`
   `localization` in `__call__` for direct callers.
3. Add a YAML option to
   [conf/filtering/analysis/](../conf/filtering/analysis/)
   (`# @package filtering`, setting `analysis: {_target_: ...}`) and select it
   with `filtering/analysis=<name>`. State the localization requirement in the
   file: the option name is the only place a user sees it before construction.
   Nested settings objects are nested `_target_` blocks (recursive
   instantiation is on and `_convert_: all` propagates from
   `conf/run_filtering.yaml`); see `conf/filtering/analysis/etkf_tsvd.yaml`.

### Adding a new solver to the observation operator

Add a new `elif solver_name == "<name>"` branch to
`ObservationOperator.__init__` in
[observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py)
that defines `self.dim_mapping` for each observed velocity component. See
codebase_guide.md §8 step 6.

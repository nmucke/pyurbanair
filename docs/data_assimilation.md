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
                            #   sensor_observation_coords
  interpolation.py          # trilinear grid-to-point interpolation
  reduction.py              # OnlineStateReduction (SVD/KL reduced update)
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
    esmda.py                # all four ESMDA variant classes
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

Wraps `ObservationOperator` with time aggregation over the window's
`time` dimension. Constructor args:

- `mode`: one of `"mean"`, `"median"`, `"max"`, `"min"`, `"full"`,
  `"intervals"`.
- `interval_seconds`: required when `mode="intervals"`.
- `aggregation_mode`: aggregation function applied within each interval
  (`"mean"` default). Only used when `mode="intervals"`.

**`"intervals"` (the config default).** Frames are binned by their `time`
coordinate (in seconds) into contiguous `interval_seconds`-wide windows:
frame at time `t` belongs to bin `floor((t - t0) / interval_seconds)`.
All frames in a bin are aggregated with `aggregation_mode`, then the
per-sensor vectors are concatenated across bins. The total observation
count is `num_intervals * num_sensors * len(obs_states)` and is not known
until the first call (the operator detects `_num_intervals` lazily).

**`"full"` mode.** Each time step contributes its own observation block;
the output length is `T * num_sensors * len(obs_states)`. `_num_time_steps`
is either passed at construction or detected from the first call.

The physical (x, y, z) position of each observation in the flattened
vector is the sensor location, independent of which variable or interval it
belongs to — because the sensor is the innermost axis. This ordering is
exploited by `_BaseESMDA._observation_coords` when building coordinates for
distance-based localization (observation `j` lives at sensor
`j % num_sensors`).

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

### Four variants

| Class | Augmented state | Notes |
|---|---|---|
| `ParameterESMDA` | Parameters only (scalar per ensemble member) | Groups time knots by `_group_ids_by_base_name` when block grouping is enabled |
| `TimeVaryingParameterESMDA` | Time-varying params flattened to `{name}_{t}` scalars | `_flatten_time_varying_params` / `_unflatten_params`; `pin_initial_time_point` optionally fixes `t=0` across windows |
| `StateAndParameterESMDA` | `[time=0 state | static params]` | `_flatten_state` / `_unflatten_state`; state-only localization via `localize_mask`; optional `state_reduction` + `final_time_smoothing` |
| `StateAndTimeVaryingParameterESMDA` | `[time=0 state | {name}_{t} scalars]` | MRO combines both parents: state flattening from `StateAndParameterESMDA`, param flattening from `TimeVaryingParameterESMDA` |

**Config names** (the `esmda/smoother` group filenames):
`static` → `ParameterESMDA`; `dynamic` → `TimeVaryingParameterESMDA`;
`state_and_parameter` → `StateAndParameterESMDA`;
`state_and_dynamic` → `StateAndTimeVaryingParameterESMDA`.

**State flatten/unflatten.** `_flatten_state` iterates variables in sorted
order, transposes each to `(ensemble, ...)`, and stacks columns.
`_unflatten_state` reverses this. The sorted-variable order is critical —
it must match between flatten and unflatten and between `_flatten_state`
and `_state_row_coords`. `_get_states` selects `time=0` so the augmented
vector holds the window initial condition. `_get_window_states` (no time
selection) feeds the `window_snapshots` basis source for state reduction.

**`_augmented_state_update`.** The shared method for both state-bearing
variants: builds `[states_flat | params_array]`, applies the Kalman update
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

**State-only localization.** For the state-bearing smoothers `localize_mask`
is set to `True` for state rows and `False` for parameter rows. Rows where
`localize_mask == False` receive all-ones inflation, which reduces their
per-row solve to the exact global update — so parameter rows are always
globally updated regardless of the localization strategy.

**Grid-block joint analysis** (`block_grouping`, Vossepoel §3b). When
`block_grouping=True` (on the localization instance), `_group_inflation`
takes the per-observation minimum inflation across all rows in a block so
they share one active-observation set and one transition matrix. For
`ParameterESMDA` blocks group time knots of the same parameter
(`_group_ids_by_base_name`); for `StateAndParameterESMDA` blocks group
co-located `u/v/w` grid cells (`_state_group_ids`). Done before grouping,
masked rows use all-ones inflation so they never influence a block's minimum.

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

**Only valid with state-bearing smoothers** (`state_and_parameter` /
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
(the forward model's configured horizon), apply ONE full-weight (`alpha = 1`)
analysis to the state *at the end of the segment* and/or the parameters, and
warm-start the next cycle from the analyzed state. Each observation batch is
consumed exactly once — there is no MDA schedule. The observation operator is
applied to the whole segment, so with the config-default
`TemporalObservationOperator(mode="intervals")` and one interval per segment
the batch is the segment's interval mean — an observation *of the segment*,
assimilated into the end-of-segment state (an operator choice, not an
approximation error).

### `BaseFilter` / `EnsembleKalmanFilter`

`BaseFilter` owns the cycle loop: forecasting, augmentation, inflation,
parameter evolution, failure substitution, on-disk `cycle_{k}/` management
(mirroring the smoother's `step_{i}/` pattern, with `prune_disk_cycles` /
`keep_first_disk_cycle` knobs) and per-cycle diagnostics. The analysis math is
an injected `AnalysisScheme` — a pure function of arrays; `EnsembleKalmanFilter`
is `BaseFilter` composed with the default `StochasticEnKFAnalysis`. New update
flavors (ETKF/LETKF, particle-style) are new `AnalysisScheme` implementations,
not new filter classes.

```python
enkf = EnsembleKalmanFilter(
    observation_operator=obs_op, forward_model=ensemble_model,
    C_D=variance_vector,            # 1-D (N_d,) variances (diag matrix accepted)
    mode="joint",                   # "state" | "parameter" | "joint"
    localization=None, inflation=RTPS(0.6), parameter_evolution=None,
)
result = enkf.run(state=None, params=prior_params,
                  observations=obs_batches,      # (num_cycles, N_d)
                  return_history=True)           # -> FilterResult
```

Mode semantics: `"state"` updates the flattened end-of-segment state only
(params, if any, are carried unmodified); `"parameter"` updates the flattened
params only (applied from the next cycle onward) and **requires spread
maintenance** (`parameter_evolution` or `inflation` — the constructor refuses
silently-collapsing configurations); `"joint"` updates `[state | params]` with
parameter rows always globally updated under localization (`localize_mask`),
exactly as the joint smoothers do. Localization strategies are reused from
`localization/` unchanged; distance-based strategies need state rows.

`FilterResult` is a plain dataclass (`params`, `state`, optional
`cycle`-concatenated histories, and `diagnostics`: one `CycleDiagnostics` per
cycle with innovation χ² consistency, observation-space prior/posterior RMSE
and per-block spreads — the "is the filter diverging/overconfident" signals,
visible at cycle k instead of at the end).

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
truth inline or from disk (as run_esmda.py), one cycle per
`time.simulation_time` segment, Hydra groups
`filtering/analysis|localization|inflation|evolution`, static scalar
parameters only (time-varying/AR(2) priors stay with the ESMDA smoothers).
See [scripts_and_configs.md](scripts_and_configs.md) §1.8 / §2.1.

---

## 9. Configuration

All smoother configuration is via Hydra groups under
[conf/esmda/](../conf/esmda/); the filter's equivalents live under
[conf/filtering/](../conf/filtering/) (see §8 and
[scripts_and_configs.md §1.8](scripts_and_configs.md)).

### `esmda/smoother` group

Four options in
[conf/esmda/smoother/](../conf/esmda/smoother/):

| File | Class | Notes |
|---|---|---|
| `static.yaml` | `ParameterESMDA` | Parameter-only, static scalars |
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

The `state_reduction` key is only consumed by `state_and_parameter.yaml`
and `state_and_dynamic.yaml` (the two smoother YAMLs that list it). Selecting
`esmda/state_reduction=svd` while using a parameter-only smoother is a no-op.

---

## 10. End-to-end run

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

`create_observation_operator` builds a `TemporalObservationOperator` (mode
`"intervals"` by default) wrapping an `ObservationOperator` using the
case's `obs_x/y/z_points`. `create_C_D` produces the diagonal
`σ² I` error covariance. The script also constructs validation sensors
(never assimilated; scored as held-out check) and handles inline vs. on-disk
truth; see `codebase_guide.md §6` and the script's docstring.

> **pylbm results produced before 2026-08-07 do not carry the state update.**
> The `prior_state = posterior_state` handoff above (and the filter's
> cycle-to-cycle warm start, §8) reaches a pylbm solver as an LBM *restart
> file*, and two independent bugs there were fixed only on 2026-08-07 (PRs
> #112–#114). Python spelled the restart filename with a 9-digit iteration
> field while the Fortran opened a 6-digit one, so the solver silently reopened
> its own restart from the previous window: for
> `esmda/smoother=state_and_parameter` and `state_and_dynamic`, **every pylbm
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

## 11. Extension recipes

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
   for both state-bearing branches.

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
2. Add a YAML option to
   [conf/filtering/analysis/](../conf/filtering/analysis/)
   (`# @package filtering`, setting `analysis: {_target_: ...}`) and select it
   with `filtering/analysis=<name>`.

### Adding a new solver to the observation operator

Add a new `elif solver_name == "<name>"` branch to
`ObservationOperator.__init__` in
[observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py)
that defines `self.dim_mapping` for each observed velocity component. See
codebase_guide.md §8 step 6.

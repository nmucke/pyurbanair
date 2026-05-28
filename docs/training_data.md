# Training-data generation for the neural-surrogate library

This describes how `scripts/generate_training_data.py` produces the
`(state, parameter)` datasets that downstream surrogate models are
trained on. It complements the broader [codebase_guide.md](codebase_guide.md);
read that first if you need orientation on the forward-model / ensemble
abstractions referenced below.

## 1. What the script produces

One invocation builds a complete `train` / `val` / `test` split for one
backend at one size:

```
training_data/<model_name>_<size>/
├── config.yaml                  resolved Hydra config used for this run
├── state/
│   ├── train/sample_XXXX.nc     one forward-model output per sample
│   ├── val/sample_XXXX.nc
│   └── test/sample_XXXX.nc
├── param/
│   ├── train/sample_XXXX.nc     matching parameter trajectory per sample
│   ├── val/sample_XXXX.nc       (interpolated onto the state time grid)
│   └── test/sample_XXXX.nc
├── params.nc                    consolidated interpolated trajectories
├── sampled_params.nc            consolidated sampler control points
├── sampled_params.png           every control-point trajectory, colored by split
├── params_interpolated.png      same, after projection onto state time
├── split_examples.png           one mid-window velocity slice per split
├── <stl>.stl                    backend geometry (when applicable)
└── <split>_animation.mp4        velocity-magnitude animation per split
```

The split is by sample index in a single shared ensemble of size
`num_train + num_val + num_test`: indices `[0, num_train)` go to train,
the next `num_val` to val, and the remainder to test. Every per-sample
state file has a sibling under `param/` at the same path, with the
inflow parameters interpolated onto the state's output time grid (one
value per saved state time step).

## 2. Config layout

Five preset sizes live in [conf/training_data/](../conf/training_data/):
`tiny`, `small`, `medium`, `large`, `xlarge`. Each is a Hydra
`# @package _global_` overlay so it can override the top-level `/model`,
`/domain`, and `/time` selections from a single file. Pick a size with
`training_data=<name>` on the CLI:

```bash
python scripts/generate_training_data.py training_data=small
```

Each preset declares:

| Field | Purpose |
|---|---|
| `num_train`, `num_val`, `num_test` | per-split sample counts |
| `output_dir` | resolves to `training_data/${model.name}_<size>/` |
| `simulation_time` / `output_frequency` / `spinup_time` | forward-model horizon, output cadence, spin-up before time-varying inflow kicks in |
| `num_time_points` | control points on the sampler's time grid (the forward model interpolates between these internally; the script also interpolates them onto the state time grid for the saved `param/*` files) |
| `seed` | RNG seed driving every random draw |
| `num_parallel_processes` | ensemble parallelism — see §5 |
| `params_sampler` | Hydra `_target_` block; see §3 |

CLI overrides apply to any field, e.g.:

```bash
python scripts/generate_training_data.py training_data=tiny \
  model=pylbm size=small \
  training_data.num_train=8 \
  training_data.params_sampler.time_series.correlation_length=30
```

## 3. The parameter sampler

The default sampler is
[`pyurbanair.training_data.UniformExternalAR2Sampler`](../src/pyurbanair/training_data/samplers.py),
configured via two blocks:

```yaml
params_sampler:
  _target_: pyurbanair.training_data.UniformExternalAR2Sampler
  _convert_: all
  external:
    inflow_angle:
      mean: {min: -30.0, max: 30.0}   # uniform per sim
      std: 5.0                          # fixed; use {min, max} to sample
    velocity_magnitude:
      mean: {min: 7.0, max: 8.0}
      std: 0.5
  time_series:
    correlation_length: 60.0
  ensemble_size: 1                     # overridden by the script
```

For every simulation member and every parameter:

1. Draw `mean_e` from the param's `mean` spec — either fixed (scalar) or
   `Uniform(min, max)` (dict). Same for `std_e`.
2. Integrate a critically-damped AR(2) anomaly `z(t)` (unit-variance,
   smooth, correlation length set by `time_series.correlation_length`).
3. Return `x(t, e) = mean_e + std_e · z(t, e)`.

The result has shape `(time, ensemble)`. Each individual simulation
gets its own per-sim central value and its own AR(2) trajectory.

**Optional clipping.** AR(2) anomalies are unbounded; if a parameter's
`mean` spec is a `{min, max}` dict, the final trajectory is clipped to
that range so it never punches outside the user-stated bounds (this
prevents solver-unsafe values like sub-physical velocity). For finer
control, pass an explicit `clip: {min?, max?}` block alongside `mean`/
`std`.

The script swaps `ensemble_size` to `num_train + num_val + num_test`
at instantiate time so a single `sample_prior(time_coords, rng_key)`
call yields every member's trajectory in one shot.

## 4. End-to-end script flow

[scripts/generate_training_data.py](../scripts/generate_training_data.py)
runs:

1. **Resolve `output_dir`** (`training_data/<model>_<size>/`), persist
   the resolved Hydra config to `config.yaml`, and wipe any stale
   `_raw_states/` staging dir — see §6.
2. **Instantiate the sampler** with `ensemble_size = n_total`. Draw all
   trajectories. Save the raw control points to `sampled_params.nc`
   and render `sampled_params.png`.
3. **Build the template forward model** (`results_dir=None`), run the
   backend's `prepare` step (compile/preprocess), and clean stale
   solver outputs.
4. **Copy the STL geometry** (if `model.forward_model.stl_path` is set).
5. **Augment params for the backend**: pyudales gets a constant-per-
   member `pressure_gradient_magnitude` array (no time dim).
6. **Build the ensemble model** with `ensemble_size = n_total`,
   `num_parallel_processes` from the config, and `results_dir = output_dir/_raw_states`.
   Failure policy is `raise` — parallel + on-disk does not support
   resample-from-successes.
7. **Run the ensemble once**: `ensemble_model.run_ensemble(params=sampled, sim_name="state")`.
   This writes per-member NetCDFs `state_{i}.nc` into the staging dir,
   in parallel.
8. **Partition into splits**: open each raw state in order, write it to
   `state/{split}/sample_{i:04d}.nc`, and delete the raw file. After
   reading the first state to learn the canonical output time grid,
   linearly interpolate the sampler control points onto that grid and
   save one `param/{split}/sample_{i:04d}.nc` per sample. Also write
   the consolidated `params.nc` at the top level.
9. **Plots and animations**: `params_interpolated.png` (post-interpolation
   trajectories), `split_examples.png` (mid-time velocity-magnitude
   slice per split), and `<split>_animation.mp4` for the first sample
   of each split.

## 5. Parallelism and resources

The script runs *all* `n_total` samples in a single
`ensemble_model.run_ensemble(...)` call — train, val, and test together
— so the underlying `ProcessPoolExecutor` keeps all
`num_parallel_processes` workers saturated until the dataset is done.
Splits are a post-hoc partition, not separate runs.

Sizing defaults:

| Preset | n_total | num_parallel_processes |
|---|---|---|
| `tiny` | 8 | 2 |
| `small` | 24 | 4 |
| `medium` | 48 | 8 |
| `large` | 96 | 8 |
| `xlarge` | 192 | 8 |

The ensemble's worker pool uses `forkserver` (not `fork`) because the
parent imports JAX, and Linux pins each worker to distinct physical
cores via `pyurbanair.utils.cpu_pinning`. See
[ensemble_scaling.md](ensemble_scaling.md) for the DRAM-bandwidth
ceiling on the dev machine — past ~4–8 workers, returns diminish.

## 6. Failure modes and gotchas

- **Stale `_raw_states/` triggers warm-start.** The ensemble model's
  `get_member_state` interprets any pre-existing `state_{i}.nc` in
  `results_dir` as a warm-start initial condition. A partial NetCDF
  from a previous *failed* run will silently switch that member into
  warm-start mode and can crash the solver during restart-file I/O
  (a uDALES SIGILL was the symptom). The script wipes
  `_raw_states/` at the top of every run so the ensemble always
  cold-starts.
- **Parallel + on-disk has no resample.** With `failure: raise` (the
  default for training-data generation), the first per-member failure
  aborts the whole ensemble. The codebase guide flags
  `resample_from_successes` as unsupported on this path; for
  training-data work, prefer fixing the root cause (e.g. an out-of-
  range parameter clip) over trying to skip failed members.
- **`num_time_points` controls trajectory smoothness.** The forward
  model linearly interpolates between sampled control points, so few
  points → smoother / coarser inflow, many points → more dynamic.
  When `correlation_length` is much larger than `simulation_time` AND
  `num_time_points` is small, each member's trajectory degenerates
  toward a straight line — set the correlation length comparable to or
  smaller than the window if you want visible time variation.
- **Output `state_*.nc` files do not embed parameters.** The matching
  trajectory is the file at the same relative path under `param/{split}/`
  (or sliced from `params.nc`). The single source of truth is
  `params.nc`.

## 7. Adding a new sampler

Implement a class exposing the
`sample_prior(time_coords, rng_key) -> xarray.Dataset` contract used by
`ParameterTimeSeries` subclasses. The returned dataset must have
`(time, ensemble)` arrays for every parameter; non-time-varying vars
(e.g. pyudales `pressure_gradient_magnitude`, shape `(ensemble,)`) are
passed through unchanged by the interpolation step. Register it under
`src/pyurbanair/training_data/` and point a new `conf/training_data/<name>.yaml`
at it via a `_target_` block.

## 8. Adding a new size

Mirror an existing preset under [conf/training_data/](../conf/training_data/).
Pick matching `/domain` and `/time` overrides from the existing groups,
and scale `num_*` and `num_parallel_processes` to the host. The
`output_dir` pattern `training_data/${model.name}_<size>/` automatically
keeps backend-specific datasets in separate trees.

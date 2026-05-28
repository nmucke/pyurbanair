# Neural surrogates

End-to-end stack for training a learned one-step surrogate of the CFD
forward models: dataset generation → on-disk layout → PyTorch
`TransitionDataset` → architectures → training loop. Complements the
broader [codebase_guide.md](codebase_guide.md); read that first for
orientation on the forward-model / ensemble abstractions referenced
below.

The stack splits into four pieces that are useful (and runnable) on
their own:

1. **Training-data generation** — drive the CFD ensemble to produce a
   `(state, parameter)` dataset on disk. See §1–§5.
2. **Data loading** — `TransitionDataset` turns the on-disk layout into
   one-step transition pairs ready for PyTorch training. See §6.
3. **Architectures + training loop** — `SimpleConv` baseline,
   `UNetConvNeXt` architecture, and the generic `Trainer`. The trainer
   checkpoints the best-val weights and supports patience-based early
   stopping; resolved config + best weights land under
   `model_weights/<model_name>/`. See §7–§10.
4. **Autoregressive rollout / test** — reload a saved model from its
   config + weights and step it through a full test trajectory,
   producing diagnostic plots and a `truth | pred | |err|`
   animation. See §11.

---

## Part A — Training-data generation

### 1. What the script produces

One invocation of `scripts/generate_training_data.py` builds a complete
`train` / `val` / `test` split for one backend at one size:

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

### 2. Config layout

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

### 3. The parameter sampler

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

### 4. End-to-end script flow

[scripts/generate_training_data.py](../scripts/generate_training_data.py)
runs:

1. **Resolve `output_dir`** (`training_data/<model>_<size>/`), persist
   the resolved Hydra config to `config.yaml`, and wipe any stale
   `_raw_states/` staging dir — see §5.
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

### 5. Parallelism, sizing, and gotchas

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

**Failure modes:**

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
  aborts the whole ensemble. Prefer fixing the root cause (e.g. an
  out-of-range parameter clip) over trying to skip failed members.
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

### Adding a new sampler

Implement a class exposing the
`sample_prior(time_coords, rng_key) -> xarray.Dataset` contract used by
`ParameterTimeSeries` subclasses. The returned dataset must have
`(time, ensemble)` arrays for every parameter; non-time-varying vars
(e.g. pyudales `pressure_gradient_magnitude`, shape `(ensemble,)`) are
passed through unchanged by the interpolation step. Register it under
`src/pyurbanair/training_data/` and point a new
`conf/training_data/<name>.yaml` at it via a `_target_` block.

### Adding a new size

Mirror an existing preset under [conf/training_data/](../conf/training_data/).
Pick matching `/domain` and `/time` overrides from the existing groups,
and scale `num_*` and `num_parallel_processes` to the host. The
`output_dir` pattern `training_data/${model.name}_<size>/` automatically
keeps backend-specific datasets in separate trees.

---

## Part B — Data loading

### 6. `TransitionDataset`

[libs/neural-surrogates/src/neural_surrogates/data.py](../libs/neural-surrogates/src/neural_surrogates/data.py)

A `torch.utils.data.Dataset` that flattens every trajectory in a split
into individual `(state_n, params_n, geometry) -> state_{n+1}` pairs. A
split with `N` trajectories of length `T` produces `N · (T − 1)` samples;
shuffling a `DataLoader` over it samples uniformly across all pairs and
all trajectories. Each item is a dict:

| Key          | Shape       | Notes |
|---|---|---|
| `state_n`    | `(C, *grid)` | velocity channels stacked in `state_vars` order |
| `state_next` | `(C, *grid)` | the next snapshot in the same trajectory |
| `params_n`   | `(P,)`      | inflow params at step `n`; scalar params (e.g. uDALES `pressure_gradient_magnitude`) are broadcast along `time` |
| `geometry`   | `(*grid,)`  | binary mask: `1` = fluid, `0` = obstacle. Same tensor for every item in the split |

The geometry mask is read from the state file's `geometry_var`
(default `"blanking"` — pylbm's per-cell obstacle indicator, inverted to
match the `1`-is-fluid convention). For backends that don't ship one,
the fallback marks fluid cells as those with a non-zero stacked state in
the first trajectory's first snapshot; ground-and-building cells stay 0.

#### Memory model

`__init__` only walks each state file to read `ds.sizes["time"]` (metadata
only) so it can build the flat `(traj, t)` index. Parameters and the
static geometry mask are loaded eagerly (both are small). State
snapshots are read lazily on each `__getitem__` via
`xr.open_dataset(..., cache=cache).isel(time=slice(t, t+2))`.

The `cache` constructor flag (`cache: bool = False`) is threaded straight
into xarray:

- `cache=False` (default) — every `.values` read goes to disk; only the
  two slices for the current pair are materialized; nothing accumulates.
  Use this for large datasets that don't fit in RAM.
- `cache=True` — xarray keeps every read slice in memory, so after one
  epoch all visited trajectories are resident and subsequent epochs are
  disk-free. Use this when the dataset comfortably fits in RAM and you
  want maximum iteration throughput.

State file handles are kept in a per-process `_state_cache` dict; a
`__getstate__` hook drops the cache before pickling so each `DataLoader`
worker rebuilds its own handles (avoids sharing netCDF descriptors across
processes).

#### Smoke script

[scripts/dataloading.py](../scripts/dataloading.py) is the smoke test:
it builds a `TransitionDataset`, wraps it in a `DataLoader`, prints the
shape of the first few batches, and writes three diagnostic plots into
`plot_dir`:

- `states.png` — `|u|` at the mid-z slice for the first 4 batch items, with
  `state_n` on top and `state_next` on the bottom on a shared color scale.
- `params.png` — scatter of the batch's `(inflow_angle, velocity_magnitude)`
  pairs.
- `geometry.png` — one subplot per vertical (z) level, white = fluid,
  black = obstacle.

```bash
pixi run -e dev python scripts/dataloading.py
pixi run -e dev python scripts/dataloading.py \
  data_dir=training_data/pylbm_small cache=true batch_size=16
```

The data-loading config lives in
[conf/neural_surrogate_training/default.yaml](../conf/neural_surrogate_training/default.yaml).

---

## Part C — Architectures and training

### 7. Components

| Piece | File |
|---|---|
| `SimpleConv` baseline | [libs/neural-surrogates/src/neural_surrogates/architectures/simple_conv.py](../libs/neural-surrogates/src/neural_surrogates/architectures/simple_conv.py) |
| `UNetConvNeXt` architecture | [libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py](../libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py) |
| `Trainer` (train/val loop) | [libs/neural-surrogates/src/neural_surrogates/training.py](../libs/neural-surrogates/src/neural_surrogates/training.py) |
| `TransitionDataset` | [libs/neural-surrogates/src/neural_surrogates/data.py](../libs/neural-surrogates/src/neural_surrogates/data.py) |
| Run script | [scripts/train_neural_surrogate.py](../scripts/train_neural_surrogate.py) |
| Config | [conf/neural_surrogate_training/train.yaml](../conf/neural_surrogate_training/train.yaml) |

All architectures share the contract
`forward(state, params, geometry) -> state_next`. The geometry mask is
concatenated to the state along the channel dimension at the stem; how
parameters enter depends on the architecture.

### 8. `SimpleConv` — baseline

Single `Conv3d` layer over `(state ⊕ geometry)` along the channel dim.

- **Input channels**: `n_state_channels + 1` — the state channels stacked
  in `state_vars` order, with the binary geometry mask appended.
- **Output channels**: `n_state_channels` — one channel per state var.
- **Parameter injection**: each inflow parameter is broadcast-added to a
  distinct output channel (param `i` → channel `i`). If
  `n_params < n_state_channels` the extra channels receive zero bias. If
  `n_params > n_state_channels` construction raises.

The model predicts `state_next` directly; there is no residual /
delta-state structure.

### 9. `UNetConvNeXt` — 3D UNet with ConvNeXt blocks

#### `_ConvNeXtBlock3d`

- **Depthwise conv** `Conv3d(C, C, k, groups=C)` — large-kernel spatial
  mixing per channel.
- **GroupNorm(1, C)** — channel-wise normalization (LayerNorm-equivalent
  for conv tensors).
- **Pointwise expand** `Conv3d(C → C·expansion, 1)` → **GELU** →
  **Pointwise project** `Conv3d(C·expansion → C, 1)` — the inverted
  bottleneck MLP.
- **Parameter bias injection**: a `Linear(n_params, C)` projects the
  per-sample inflow vector to one bias per channel; that bias is
  broadcast-added over all spatial positions inside *every* block, so
  params modulate every layer of the network.
- Residual connection wraps the whole block.

#### `UNetConvNeXt`

- **Stem**: `Conv3d(n_state_channels + 1, base_channels, 3)`.
- **Encoder**: for each level `i`, a stage of `depths[i]` ConvNeXt
  blocks at `base_channels · channel_mults[i]`, then a stride-2 `Conv3d`
  to the next stage's channel count. Each pre-downsample activation is
  stashed as a skip.
- **Bottleneck**: one stage at the deepest channel count.
- **Decoder** (mirror): `ConvTranspose3d` upsamples, a 1×1 `Conv3d`
  fuses the upsampled tensor concatenated with its skip, then another
  stage of ConvNeXt blocks.
- **Head**: `Conv3d(base_channels, n_state_channels, 1)` — predicts
  `state_next` directly (no residual / delta-state structure yet).
- **Arbitrary input shapes**: `_pad_to_multiple` pads `(D, H, W)` up to
  a multiple of `2^n_levels` before the stem, then the head output is
  cropped back to the original spatial shape. Lets odd grid sizes
  (e.g. `5×7×11`) round-trip cleanly.

#### Size presets

The config group
[conf/neural_surrogate_architectures/unet_convnext/](../conf/neural_surrogate_architectures/unet_convnext/)
holds five presets that scale `base_channels`, `channel_mults`,
`depths`, `kernel_size`, `expansion`. Each file is a single
`_target_: neural_surrogates.UNetConvNeXt` block:

| Preset | base | mults | depths | kernel | expansion |
|---|---|---|---|---|---|
| tiny | 8 | [1, 2] | [1, 1] | 3 | 2 |
| small | 16 | [1, 2, 4] | [1, 1, 1] | 5 | 4 |
| medium | 24 | [1, 2, 4] | [2, 2, 2] | 7 | 4 |
| large | 32 | [1, 2, 4, 8] | [2, 2, 2, 2] | 7 | 4 |
| xlarge | 48 | [1, 2, 4, 8] | [3, 3, 3, 3] | 7 | 4 |

### 10. `Trainer` and run script

`Trainer` is a generic loop. Its constructor takes `model`,
`train_loader`, `val_loader`, `optimizer`, `loss_fn`, `num_epochs`,
`device`, optional `patience`, and optional `weights_path`. `fit()` runs
the loop; each epoch calls `_train_epoch` then `_validate` and prints
the mean losses. Batch unpacking assumes the `TransitionDataset` dict
layout (`state_n`, `state_next`, `params_n`, `geometry`).

**Best-checkpoint saving.** When `weights_path` is set, the trainer
writes `model.state_dict()` to that path every time the val loss
improves, and reloads it into the model at the end of `fit()` so the
returned model is the best-val checkpoint (not the last epoch). The run
script passes `weights_path=model_weights/<model_name>/weights.pt`, so
nothing needs to be saved by the caller after `fit()`.

**Early stopping.** When `trainer.patience` is set in the config
(default `null`, disabled), training halts after `patience` consecutive
epochs without val-loss improvement. Combine with a generous
`num_epochs` to let the patience criterion choose when to stop.

The model and dataloaders are deliberately **constructed outside** the
trainer and passed in — this keeps `Trainer` agnostic to backend choice,
augmentation, and config structure.

[scripts/train_neural_surrogate.py](../scripts/train_neural_surrogate.py):

1. Pull `dtype` from `cfg.dataset.dtype` (string → `torch.dtype`).
2. `instantiate(cfg.dataset, split="train"|"val", dtype=...)` → two
   `TransitionDataset`s.
3. `instantiate(cfg.dataloader, dataset=...)` for each, forcing
   `shuffle=False` on val.
4. `instantiate(cfg.architecture, n_state_channels=len(cfg.dataset.state_vars),
   n_params=len(train_ds.param_names))` → model.
5. Save the resolved Hydra config to
   `model_weights/<model_name>/config.yaml`. `model_name` is a top-level
   config field (default `unet_convnext_small`); override on the CLI
   with `model_name=...`.
6. `instantiate(cfg.trainer, model=..., train_loader=..., val_loader=...,
   optimizer=instantiate(cfg.optimizer, params=model.parameters()),
   loss_fn=instantiate(cfg.loss),
   weights_path=model_weights/<model_name>/weights.pt)`.
7. `trainer.fit()` — the trainer writes `weights.pt` on every val-loss
   improvement and loads the best checkpoint back into the model before
   returning. Re-instantiating the architecture from the saved
   `config.yaml` and loading `weights.pt` rebuilds the exact trained
   model.

Every runtime object — architecture, dataset, dataloader, optimizer,
loss, trainer — is constructed via `hydra.utils.instantiate` against a
`_target_` block. Only `n_state_channels` and `n_params` stay explicit
because they're derived from the dataset, not the architecture preset.

### Config and CLI

[conf/neural_surrogate_training/train.yaml](../conf/neural_surrogate_training/train.yaml)
is `# @package _global_` and pulls an architecture preset into its
defaults list:

```yaml
defaults:
  - /neural_surrogate_architectures/unet_convnext@architecture: small
  - _self_
```

`@hydra.main` is pointed at the top-level `conf/` so the cross-group
defaults entry resolves. Default preset:

```bash
pixi run -e dev python scripts/train_neural_surrogate.py
```

Swap architecture presets — the override key is the full group path:

```bash
pixi run -e dev python scripts/train_neural_surrogate.py \
    'neural_surrogate_architectures/unet_convnext@architecture=medium'
```

Override individual fields:

```bash
pixi run -e dev python scripts/train_neural_surrogate.py \
    dataset.root_dir=training_data/pylbm_small \
    dataloader.batch_size=16 \
    trainer.num_epochs=20 \
    optimizer.lr=5e-4 \
    architecture.kernel_size=5
```

### 11. Autoregressive rollout on the test split

[scripts/test_neural_surrogate.py](../scripts/test_neural_surrogate.py)
loads `model_weights/<model_name>/config.yaml`, re-instantiates the
architecture and `TransitionDataset` from it, restores `weights.pt`, and
steps the model from `truth[0]` for `T - 1` steps so the predicted
trajectory matches the test trajectory length. At each step the
ground-truth `params_n` for that time index is fed in. The script is
Hydra-driven via
[conf/neural_surrogate_testing/test.yaml](../conf/neural_surrogate_testing/test.yaml)
and takes `model_dir`, `sample_idx`, `device`, and `output_dir` (default
`${model_dir}/rollout_${sample_idx}`).

Outputs in `${output_dir}/`:

| File | Contents |
|---|---|
| `trajectory.pt` | `{"truth": (T, C, *grid), "pred": (T, C, *grid)}` torch tensors |
| `rollout.png` | mid-z `|u|` slices at evenly-spaced times: truth / pred / `|err|` rows |
| `rmse.png` | per-step RMSE vs ground truth across the rollout |
| `rollout.mp4` | three-panel animation (truth, pred, `|err|`) of mid-z `|u|`, all `T` steps. Falls back to `rollout.gif` when ffmpeg is missing. |

All slice plots index the z-axis (first spatial dim of the `(C, nz, ny, nx)`
state tensor), matching the convention used in
[scripts/dataloading.py](../scripts/dataloading.py).

```bash
pixi run -e dev python scripts/test_neural_surrogate.py \
    model_dir=model_weights/unet_convnext_small sample_idx=0
```

### Extending

- **New architecture**: add a module under
  [libs/neural-surrogates/src/neural_surrogates/architectures/](../libs/neural-surrogates/src/neural_surrogates/architectures/),
  re-export from
  [architectures/__init__.py](../libs/neural-surrogates/src/neural_surrogates/architectures/__init__.py)
  (and the top-level
  [neural_surrogates/__init__.py](../libs/neural-surrogates/src/neural_surrogates/__init__.py)
  if you want a flat `_target_`), and add a sibling group under
  [conf/neural_surrogate_architectures/](../conf/neural_surrogate_architectures/)
  with one preset file per size. The `Trainer` does not need to change
  as long as the new model accepts `(state, params, geometry)`.
- **New optimizer / loss / loader**: change the `_target_` (and kwargs)
  in [train.yaml](../conf/neural_surrogate_training/train.yaml). No code
  edits required.
- **New trainer behavior** (schedulers, checkpointing, logging): extend
  `Trainer` and bump the `_target_` in the `trainer:` block.

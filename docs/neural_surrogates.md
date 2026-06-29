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
   `UNetConvNeXt` architecture, `UPT` (Universal Physics Transformer),
   `P3D`, and the generic `Trainer`. The trainer checkpoints the best-val
   weights and supports patience-based early stopping; resolved config +
   best weights land under `model_weights/<model_name>/`. See §7–§10.
4. **Autoregressive rollout / test** — reload a saved model from its
   config + weights and step it through a full test trajectory,
   producing diagnostic plots and a `truth | pred | |err|`
   animation. See §11.

---

## Part A — Training-data generation

### 1. What the script produces

One invocation of `scripts/neural_surrogate/generate_training_data.py` builds a complete
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

**Grid collocation.** pyudales solves on a staggered C-grid (`u@xm`,
`v@ym`, `w@zm`); before saving, each member state is linearly
interpolated to cell centers (`xt`, `yt`, `zt`) via
`pyudales.utils.grid_utils.interpolate_grid` so all state channels land
on a common regular grid. pylbm output is already cell-centered and is
passed through unchanged.

### 2. Config layout

A single config drives data generation:
[conf/neural_surrogate/training_data.yaml](../conf/neural_surrogate/training_data.yaml)
(`config_name="neural_surrogate/training_data"`). It bases off the forward-model
entry point, so the **physical setup — domain (grid + bounds), geometry/STL and
the per-window time horizon — comes from the selected `case`** (default
`xie_and_castro`; switch with `case=barcelona`). The file only adds the dataset
shape + parameter sampler:

```bash
python scripts/neural_surrogate/generate_training_data.py            # default case + model
python scripts/neural_surrogate/generate_training_data.py model=pylbm case=barcelona
```

It declares (under `training_data:`):

| Field | Purpose |
|---|---|
| `num_train`, `num_val`, `num_test` | per-split sample counts |
| `output_dir` | resolves to `training_data/${model.name}_medium/` |
| `simulation_time` / `output_frequency` / `spinup_time` | generation horizon — interpolated from the case's `${time.*}` |
| `seed` | RNG seed driving every random draw |
| `num_parallel_processes` | ensemble parallelism — see §5 |
| `params_sampler` | Hydra `_target_` block (incl. `num_time_points`, the sampler time-grid control points); see §3 |

CLI overrides apply to any field, e.g.:

```bash
python scripts/neural_surrogate/generate_training_data.py \
  model=pylbm \
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

[scripts/neural_surrogate/generate_training_data.py](../scripts/neural_surrogate/generate_training_data.py)
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
[ensemble_scaling.md](temp/ensemble_scaling.md) for the DRAM-bandwidth
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
`src/pyurbanair/training_data/` and point `training_data.params_sampler._target_`
in [conf/neural_surrogate/training_data.yaml](../conf/neural_surrogate/training_data.yaml)
at it.

### Changing the dataset size

There is a single data-generation config (no size group). Edit the
`training_data.*` fields in
[conf/neural_surrogate/training_data.yaml](../conf/neural_surrogate/training_data.yaml)
(or override them on the CLI), and switch the grid/horizon via `case=`. The
`output_dir` pattern `training_data/${model.name}_medium/` keeps backend-specific
datasets in separate trees.

---

## Part B — Data loading

### 6. `TransitionDataset`

[libs/neural-surrogates/src/neural_surrogates/datasets/transition.py](../libs/neural-surrogates/src/neural_surrogates/datasets/transition.py)

A `torch.utils.data.Dataset` that flattens every trajectory in a split
into `K`-step training samples (`K = pushforward_steps`, default `1`).
Each sample anchors at trajectory time `t`, returns `state_n` at `t` and
`state_next` at `t+K`, and provides the `K` parameter vectors needed to
unroll the model forward. A split with `N` trajectories of length `T`
produces `N · (T − K)` samples; shuffling a `DataLoader` over it samples
uniformly across all transition windows and all trajectories. With `K=1`
this reduces to one-step transition pairs (the original behavior); see
§10 for how the trainer uses `K>1`. Each item is a dict:

| Key          | Shape       | Notes |
|---|---|---|
| `state_n`    | `(C, *grid)` | velocity channels stacked in `state_vars` order, at time `t` |
| `state_next` | `(C, *grid)` | snapshot at time `t + K` — the pushforward target |
| `params_n`   | `(K, P)`    | inflow params at steps `t, …, t+K-1`; scalar params (e.g. uDALES `pressure_gradient_magnitude`) are broadcast along `time` |
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
`xr.open_dataset(..., cache=cache).isel(time=[t, t+K])` — only the two
endpoint snapshots ever leave disk per sample, regardless of `K`. The
intermediate ground-truth states are never read because the pushforward
unroll feeds the model its own predictions in their place.

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

[scripts/neural_surrogate/dataloading.py](../scripts/neural_surrogate/dataloading.py) is the smoke test:
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
pixi run -e dev python scripts/neural_surrogate/dataloading.py
pixi run -e dev python scripts/neural_surrogate/dataloading.py \
  --data-dir training_data/pylbm_medium --cache --batch-size 16
```

It is a plain argparse CLI (not Hydra) — run with `--help` to see every flag.

---

## Part C — Architectures and training

### 7. Components

| Piece | File |
|---|---|
| `SimpleConv` baseline | [libs/neural-surrogates/src/neural_surrogates/architectures/simple_conv.py](../libs/neural-surrogates/src/neural_surrogates/architectures/simple_conv.py) |
| `UNetConvNeXt` architecture | [libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py](../libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py) |
| `UPT` architecture | [libs/neural-surrogates/src/neural_surrogates/architectures/upt.py](../libs/neural-surrogates/src/neural_surrogates/architectures/upt.py) |
| `P3D` architecture | [libs/neural-surrogates/src/neural_surrogates/architectures/p3d.py](../libs/neural-surrogates/src/neural_surrogates/architectures/p3d.py) |
| `BaseTraining` (shared machinery) | [libs/neural-surrogates/src/neural_surrogates/training/base.py](../libs/neural-surrogates/src/neural_surrogates/training/base.py) |
| `Trainer` (full-grid train/val loop) | [libs/neural-surrogates/src/neural_surrogates/training/standard.py](../libs/neural-surrogates/src/neural_surrogates/training/standard.py) |
| `TransitionDataset` | [libs/neural-surrogates/src/neural_surrogates/datasets/transition.py](../libs/neural-surrogates/src/neural_surrogates/datasets/transition.py) |
| Run script | [scripts/neural_surrogate/train_neural_surrogate.py](../scripts/neural_surrogate/train_neural_surrogate.py) |
| Config | [conf/neural_surrogate/training.yaml](../conf/neural_surrogate/training.yaml) |

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
[conf/neural_surrogate/architectures/unet_convnext/](../conf/neural_surrogate/architectures/unet_convnext/)
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

### 9a. `UPT` — Universal Physics Transformer

[libs/neural-surrogates/src/neural_surrogates/architectures/upt.py](../libs/neural-surrogates/src/neural_surrogates/architectures/upt.py)

UPT treats the fluid cells as an **unstructured point cloud** and avoids
all convolutions on the full grid. One forward step:

1. **Gather fluid points.** The geometry mask is used to extract fluid
   cell indices and their `(z, y, x)` integer coordinates — obstacle
   cells stay zero throughout and are never fed to the network.
2. **Encode onto supernodes.** `EncoderSupernodes` builds a sparse
   neighbourhood graph (radius `r`, up to `max_degree` neighbours per
   supernode) and pools the `num_supernodes` supernode features via a
   GNN message MLP (`gnn_dim`) into `enc_depth` transformer layers,
   then projects to `num_latent_tokens` latent tokens via a perceiver
   cross-attention tail.
3. **Approximate in latent space.** `Approximator` runs `approx_depth`
   self-attention layers over the latent tokens.
4. **Decode back to fluid cells.** `DecoderPerceiver` uses cross-attention
   from the latent tokens to the query fluid-cell positions, producing
   `n_state_channels` values per fluid cell.
5. **Scatter to grid.** Decoded values are written back at the fluid
   indices; obstacle cells stay 0.

**`_geom_cache`.** The supernode selection, neighbour graph, and fluid
indices are a pure function of the geometry. They are cached in
`self._geom_cache` keyed on `(total_cells, n_fluid, device, dtype)`, so
the expensive `cdist`-based neighbour build runs only on the first step
for a given geometry and is reused every subsequent step.

**Key knobs:**

| Knob | Default | Notes |
|---|---|---|
| `normalize` | `True` | Z-score state channels and inflow params before the encoder, de-normalise the output. **Load-bearing**: raw `inflow_angle` (~50°) swamps the velocity channels (~1 m/s); without normalisation the encoder's input projection ignores the state and the rollout collapses. |
| `predict_residual` | `True` | Predict the *change* `state_{t+1} − state_t` rather than the absolute next state. At the compression ratios affordable on dense ~256k-cell grids, an absolute prediction collapses to a smooth mean; the residual keeps the task well-scaled and makes near-identity the model's natural default. |
| `cond_dim` | `None` | `None`: inflow params are concatenated to every fluid-cell's feature vector. If set, params are projected to a `(B, cond_dim)` DiT condition vector passed to all transformer stacks. |
| `attention_type` | `"dot_product"` | Self-attention implementation for all transformer stacks (perceiver cross-attention tails are unaffected). Options: `"dot_product"` (standard scaled-dot-product), `"dot_product_slow"`, `"efficient"` (linear), `"linformer"`, `"transsolver"`. `"transsolver"` requires `attention_kwargs: {num_slices: N}`. |
| `extra_in_channels` | `0` | Extra input-only channels gathered at fluid cells raw (no normalisation) and concatenated before the params. Used by the DD wrapper to feed per-patch coarse context + positional encodings. Default 0 keeps the state dict byte-identical to a model built without the argument. |

**Normalization stats.** Computed by `_compute_normalization_stats` in
the training script: streamed file-by-file over the training split in
float64, restricted to fluid cells via the geometry mask. Installed via
`model.set_normalization(state_mean, state_std, param_mean, param_std)`;
stored as buffers `state_mean/state_std/param_mean/param_std` and saved
with the checkpoint so rollout/test callers get the correct
standardisation for free.

**Shared-geometry fast path.** Within one forward call, all batch members
voxelise the same STL onto the same grid, so one point set and supernode
graph serves the whole batch. The shared-geometry guard checks that all
batch members have the same number of fluid cells (`O(B)` reductions per
step, not `O(B·N)`) and falls back to a per-sample loop only when they
differ.

#### Size presets

The config group
[conf/neural_surrogate/architectures/upt/](../conf/neural_surrogate/architectures/upt/)
holds five presets (`_target_: neural_surrogates.UPT`). All default to
`normalize: true`, `predict_residual: true`, `attention_type:
dot_product`, `cond_dim: null`.

| Preset | dim | latent tokens | supernodes | gnn_dim | enc/approx/dec depth | heads | radius | max_degree |
|---|---|---|---|---|---|---|---|---|
| tiny | 32 | 16 | 16 | 16 | 1/1/1 | 2 | 2.5 | 8 |
| small | 128 | 64 | 128 | 128 | 2/4/2 | 4 | 4.0 | 24 |
| medium | 192 | 128 | 256 | 192 | 4/4/4 | 3 | 5.0 | 32 |
| large | 384 | 256 | 512 | 256 | 4/6/4 | 6 | 5.0 | 32 |
| xlarge | 768 | 512 | 1024 | 384 | 4/8/4 | 12 | 6.0 | 32 |

### 10. `Trainer` / `BaseTraining` and run script

`Trainer`
([training/standard.py](../libs/neural-surrogates/src/neural_surrogates/training/standard.py))
is a thin subclass of `BaseTraining`
([training/base.py](../libs/neural-surrogates/src/neural_surrogates/training/base.py)).
`BaseTraining` holds all architecture-agnostic machinery; `Trainer`'s
only addition is `_final_loss` — a masked element-wise `loss_fn(pred,
target)` applied to the final rollout step. `PatchTrainer` overrides the
same hook with the four-term Eq (9) loss instead (see §18).

`BaseTraining.__init__` accepts `model`, `train_loader`, `val_loader`,
`optimizer`, `loss_fn`, `num_epochs`, `device`, plus a rich knob set:

| Knob | Purpose |
|---|---|
| `patience` | epochs without val improvement before stopping (default `None`, disabled) |
| `weights_path` | path for best-val `weights.pt` (written on every improvement, reloaded at end) |
| `amp` / `amp_dtype` | mixed-precision autocast (`bfloat16` by default) with grad scaling |
| `compile_model` / `compile_dynamic` | `torch.compile` the model before training |
| `channels_last` | `channels_last_3d` memory layout for faster 3D-conv kernels |
| `cudnn_benchmark` / `tf32` | CUDA backend tuning (autotuned conv, Ampere TF32) |
| `pushforward_epochs_per_step` / `pushforward_start_steps` | pushforward-horizon curriculum (see below) |
| `lr_warmup_epochs` / `lr_warmup_start` / `lr_min` | linear warmup → cosine annealing LR schedule |
| `grad_clip_norm` | gradient clipping (`torch.nn.utils.clip_grad_norm_`) |
| `checkpoint_every` / `resume` | full checkpoint (model + optimizer + scheduler + scaler + curriculum) for resume |

`fit()` runs the loop; each epoch calls `_train_epoch` then `_validate`
and prints the mean losses. Batch unpacking assumes the `TransitionDataset`
dict layout (`state_n`, `state_next`, `params_n`, `geometry`).

**Pushforward trick.** When the dataset is built with
`pushforward_steps=K>1` ([Brandstetter et al., 2022](https://iclr-blogposts.github.io/2023/blog/2023/autoregressive-neural-pde-solver/)),
`params_n` arrives as `(B, K, P)` and the rollout runs the model through
`K-1` steps under `torch.no_grad()` starting at `state_n`, then takes one
gradient-tracked step against `state_next` (the snapshot at `t+K`). This
exposes the network to its own predictions during training — closing the
distribution gap that pure one-step training leaves — without
backpropagating through the unroll. With `K=1` (the default) the inner
loop is skipped and behaviour is identical to one-step training.
Validation uses the same rollout, so with `K>1` the checkpointed best-val
model minimises a `K`-step error rather than a one-step error.

**Pushforward curriculum.** Setting `pushforward_epochs_per_step` enables a
curriculum that starts the rollout horizon at `pushforward_start_steps`
and increments it by one every `pushforward_epochs_per_step` epochs up to
the dataset's `pushforward_steps`. This lets the model first learn one-step
transitions before being exposed to its own compounding errors.

**LR schedule.** When `lr_warmup_epochs` is set the optimizer's LR ramps
linearly from `lr_warmup_start` to its configured peak over the warmup
window, then cosine-anneals down to `lr_min` over the remaining epochs.
`lr_warmup_epochs=None` (default) keeps the LR fixed.

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

**Normalization.** After building the model but before saving the config,
the training script calls `_compute_normalization_stats(train_ds)` if
`hasattr(model, "set_normalization")` — currently only `UPT` exposes this
hook. Stats are streamed file-by-file in float64, restricted to fluid
cells via the geometry mask (so obstacle zeros don't bias the mean), and
passed to `model.set_normalization(state_mean, state_std, param_mean,
param_std)`. They are stored as model buffers, travel with the checkpoint,
and are restored automatically at rollout / test time.

The model and dataloaders are deliberately **constructed outside** the
trainer and passed in — this keeps `Trainer` agnostic to backend choice,
augmentation, and config structure.

[scripts/neural_surrogate/train_neural_surrogate.py](../scripts/neural_surrogate/train_neural_surrogate.py):

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

[conf/neural_surrogate/training.yaml](../conf/neural_surrogate/training.yaml)
is `# @package _global_` and pulls an architecture preset into its
defaults list:

```yaml
defaults:
  - _self_
  - /neural_surrogate/mode@_global_: domain_decomposition
```

`@hydra.main` is pointed at the top-level `conf/` so the cross-group
defaults entries resolve. The trainer is **not** a group — its fields live in an
inline `trainer:` block in `training.yaml`. The `mode` group (see §19) bundles
everything that varies together — the **architecture default**, the trainer
*class*, the *loss* and `model_name`: `mode=standard` → `unet_convnext/medium` +
`Trainer` + `MSELoss`, `mode=domain_decomposition` → `domain_decomposed/medium` +
`PatchTrainer` + `DomainDecompositionLoss`. The `mode` entry sits **after**
`_self_` so it overrides the inline `trainer._target_`, and it supplies both the
architecture default and the whole `loss:` config (there is no separate `loss`
group). The architecture family/size is still swappable on the CLI on top of the
mode default. Default preset:

```bash
pixi run -e dev python scripts/neural_surrogate/train_neural_surrogate.py
```

Swap architecture presets / families — the override value is the nested
`family/preset` path:

```bash
pixi run -e dev python scripts/neural_surrogate/train_neural_surrogate.py \
    'neural_surrogate/architectures@architecture=unet_convnext/large'
```

Override individual fields:

```bash
pixi run -e dev python scripts/neural_surrogate/train_neural_surrogate.py \
    dataset.root_dir=training_data/pylbm_small \
    dataset.pushforward_steps=4 \
    dataloader.batch_size=16 \
    trainer.num_epochs=20 \
    optimizer.lr=5e-4 \
    architecture.kernel_size=5
```

### 11. Autoregressive rollout on the test split

[scripts/neural_surrogate/test_neural_surrogate.py](../scripts/neural_surrogate/test_neural_surrogate.py)
loads `model_weights/<model_name>/config.yaml`, re-instantiates the
architecture and `TransitionDataset` from it, restores `weights.pt`, and
steps the model from `truth[0]` for `T - 1` steps so the predicted
trajectory matches the test trajectory length. At each step the
ground-truth `params_n` for that time index is fed in. The script is
Hydra-driven via
[conf/neural_surrogate/testing.yaml](../conf/neural_surrogate/testing.yaml)
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
[scripts/neural_surrogate/dataloading.py](../scripts/neural_surrogate/dataloading.py).

```bash
pixi run -e dev python scripts/neural_surrogate/test_neural_surrogate.py \
    model_dir=model_weights/unet_convnext_small sample_idx=0
```

---

## Part D — Running the surrogate as a forward model

### 12. `NeuralSurrogateForwardModel`

[libs/neural-surrogates/src/neural_surrogates/forward_model.py](../libs/neural-surrogates/src/neural_surrogates/forward_model.py)

A trained one-step network is wrapped as a
[`BaseForwardModel`](../src/pyurbanair/base_forward_model.py) so it slots
into the ensemble / ESMDA machinery as a fourth backend alongside pylbm,
pyudales and pypalm. `run_single(state, params, sim_name)` rolls the
network autoregressively and returns an `xarray.Dataset` over `time` on a
regular cell-centered grid with coords `(z, y, x)` — so `solver_name:
pylbm` (the regular-grid observation mapping) applies regardless of the
spin-up backend.

Everything describing the trained network is read from a **`model_dir`** —
the folder [scripts/neural_surrogate/train_neural_surrogate.py](../scripts/neural_surrogate/train_neural_surrogate.py)
writes (§10):

| Read from | Supplies |
|---|---|
| `model_dir/config.yaml` → `architecture` | the network to rebuild |
| `model_dir/config.yaml` → `dataset.state_vars` / `param_vars` | channel & parameter ordering (`param_vars: null` → read from the first training param file) |
| `model_dir/weights.pt` | trained parameters |
| `dataset.root_dir/config.yaml` → `domain` | the **trained domain** the requested grid is checked against |
| `dataset.root_dir/config.yaml` → `time.output_frequency` | the **trained step size** (one network step) |

Each can still be overridden explicitly (handy for tests), but the normal
path is to only set `model_dir`.

Key behaviours:

| Concern | Behaviour |
|---|---|
| **Trained step size** | The network always advances at its trained cadence (`trained_output_frequency`). To honour a requested `output_frequency` that differs, the rollout emits a frame at the internal step closest to each requested output time — so the result lands on the requested grid whether or not the two cadences divide evenly. A requested cadence *finer* than the trained step (the surrogate can't emit between steps) raises. |
| **Domain check** | The requested `(nx, ny, nz, bounds)` must equal `trained_domain`; a mismatch raises (the network only applies to its training grid). |
| **Spin-up / collocation** | With `spinup_source: forward_model` a cold start (`state is None`) is bootstrapped by `spinup_forward_model` — the CFD backend that generated the training data — whose final field seeds the rollout. Because the training data is collocated to cell centers (pyudales' staggered C-grid → `xt/yt/zt`; §1), the spin-up field is collocated the same way and renamed to `(z, y, x)` *before* it reaches the network, so the inputs match what it trained on. Warm starts (a `state` is passed) skip spin-up; collocation is idempotent, so the surrogate's own regular-grid output passes through unchanged. `disable_spinup()` propagates to the backend. With `spinup_source: training_data` the surrogate runs **no** spin-up of its own — the assimilation is warm-started from training snapshots loaded by `run_esmda` (see below), so a cold start (`state is None`) raises. |
| **Geometry** | When `stl_path` is set the geometry channel is voxelised from the STL onto the grid ([geometry.py](../libs/neural-surrogates/src/neural_surrogates/geometry.py)); otherwise it falls back to the non-zero-state convention used by `TransitionDataset`. |
| **Parameters** | Time-varying inflow params are interpolated onto the internal step times in the trained `param_vars` order; scalar params are broadcast. |

`NeuralSurrogateEnsembleForwardModel`
([ensemble_forward_model.py](../libs/neural-surrogates/src/neural_surrogates/ensemble_forward_model.py))
clones each member by sharing the (stateless) network and cloning the
spin-up backend into its own experiment directory via that backend's
`create_new_forward_model` helper.

### Config and usage

[conf/model/neural_surrogate.yaml](../conf/model/neural_surrogate.yaml)
mirrors the other `conf/model/*.yaml` files (`name`, `solver_name`,
`forward_model._target_`, `ensemble_model._target_`, `prepare._target_`).
The `forward_model` node points at a `model_dir` (default
`model_weights/unet_convnext_tiny`) and uses `_recursive_: false` so the
surrogate fills in `n_state_channels` / `n_params` and builds its spin-up
backend itself. `prepare` runs `prepare_neural_surrogate`, which
compiles/preprocesses the spin-up backend. `default_params` provides
constant fallbacks for trained parameters a caller omits (e.g. ESMDA only
varies the inflow, but a uDALES-trained net also expects
`pressure_gradient_magnitude`).

Select it as a truth or assimilation model just like any backend:

```bash
python scripts/run_esmda.py \
    model@truth_model=pyudales model@assim_model=neural_surrogate \
    assim_model.forward_model.model_dir=model_weights/unet_convnext_tiny
```

**Training-data warm start (`spinup_source: training_data`).** When the surrogate
is the assimilation model and `forward_model.spinup_source` is `training_data`,
`scripts/run_esmda.py` seeds the **first** assimilation window from pre-computed
training trajectories instead of a CFD spin-up, using the model-level
`training_data_spinup` config node (`root` / `split` /
`initial_param_jitter_scale`) and the helpers in
[`neural_surrogates.training_spinup`](../libs/neural-surrogates/src/neural_surrogates/training_spinup.py):
each ensemble member starts from the **last** frame of a training sample
(streamed one frame at a time to per-member files on disk, so the full ensemble
never sits in RAM — these files are handed to ESMDA as the window-0 initial
state), and its sampled prior inflow is anchored to that sample's final inflow
value (the AR(2) draw's shape is kept; only its level is pinned). The known `t=0`
is then pinned in the smoother for window 0. The surrogate forward model itself
holds **no** training-data logic — it only rolls a provided warm-start state
forward — so the two pieces (loading + anchoring) live entirely in `run_esmda`.

The `pyudales_neural_surrogate` case in
[tests/test_run_esmda.py](../tests/test_run_esmda.py)
builds a throwaway `model_dir` (random weights, trained domain == the test
grid) via the `surrogate_model_dir_factory` fixture, exercising the full
load-from-folder path without needing a real checkpoint.

### Extending

- **New architecture**: add a module under
  [libs/neural-surrogates/src/neural_surrogates/architectures/](../libs/neural-surrogates/src/neural_surrogates/architectures/),
  re-export from
  [architectures/__init__.py](../libs/neural-surrogates/src/neural_surrogates/architectures/__init__.py)
  (and the top-level
  [neural_surrogates/__init__.py](../libs/neural-surrogates/src/neural_surrogates/__init__.py)
  if you want a flat `_target_`), and add a sibling group under
  [conf/neural_surrogate/architectures/](../conf/neural_surrogate/architectures/)
  with one preset file per size. The `Trainer` does not need to change
  as long as the new model accepts `(state, params, geometry)`.
- **New optimizer / loss / loader**: change the `_target_` (and kwargs)
  in [train.yaml](../conf/neural_surrogate/training.yaml). No code
  edits required.
- **New trainer behavior** (schedulers, checkpointing, logging): extend
  `Trainer` and bump the `_target_` in the `trainer:` block.

---

## Part E — Domain decomposition (two-level, Recommendation A)

A learned one-step surrogate whose **spatial decomposition lives inside the
model**. Instead of one network spanning the whole grid, a two-level
overlapping decomposition splits the domain into a uniform batch of
overlapping patches, runs a shared per-patch *fine* net, and stitches the
patch outputs back together with a partition-of-unity (PoU) blend — while a
small *coarse* net supplies global context. The design (companion PDF §2 +
§5, Algorithm 1) is described in
[docs/dd_implementation_plan.md](plans/dd_implementation_plan.md).

The point of the decomposition is **grid flexibility**: because the model
tiles a fixed patch size, one trained instance runs on any global grid that
shares its training cell spacing (§16) — the global grid becomes a free
parameter.

### 13. The two-level update (Algorithm 1)

[libs/neural-surrogates/src/neural_surrogates/architectures/domain_decomposed.py](../libs/neural-surrogates/src/neural_surrogates/architectures/domain_decomposed.py)

`DomainDecomposed.forward(state, params, geometry) -> state_next` is one step
of Algorithm 1, executed entirely on full-grid tensors:

1. **Coarse step (global context).** Average-pool the state
   (`dd.restrict_coarse`, factor `coarsen_factor`) and the geometry mask
   (`dd.restrict_coarse_geom`, `any_fluid` by default — a coarse cell is
   fluid if *any* fine cell in its window is, keeping thin corridors visible),
   run the small dedicated `coarse_net` (a residual `UNetConvNeXt`), and
   trilinearly `prolong` the result back to the fine grid. This is the global
   *context* field `C`.
2. **Fine step.** Tile state, geometry and `C` into a uniform batch of
   overlapping **extended (halo) blocks** of edge `n + 2h`
   (`dd.restrict` — pad each axis to a multiple of `interior_size = n`, then
   add `halo = h` on every side). Append the per-patch positional encoding
   (`dd.positional`). The shared `fine_net` (residual `UNetConvNeXt`) runs
   **once** on the whole patch batch and predicts a residual per patch; the
   context + positional channels enter the stem **raw** through the widened
   `extra_in_channels` path (§14). Each sample's `params` are broadcast to all
   its patches.
3. **Merge.** Crop each patch output to the `(n + 2·taper)` PoU footprint,
   window-blend back to the full grid (`dd.extend_merge`, `Σ wᵢ ≡ 1`), crop
   to the original shape, apply the optional divergence projection
   (`divergence_projection`, default **off** — currently an identity stub
   pending Eq 8), and **zero obstacle cells** with the geometry mask.

The decomposition operators are pure torch in
[decomposition.py](../libs/neural-surrogates/src/neural_surrogates/decomposition.py)
(`DomainDecomposition`): `restrict` / `restrict_coarse` / `restrict_coarse_geom`
/ `prolong` / `positional` / `extend_merge` / `neighbor_indices`. There is **no
I/O** there — everything is tensor bookkeeping so it can sit inside the model.
A small per-shape `_Plan` (tiling counts, padding, PoU window, positional
encoding) is built lazily and cached per `(grid, device, dtype)`, so a change
of grid size simply rebuilds the plan.

**PoU windows.** Strict-`n` interiors are disjoint and give no blend, so the
merge uses a slightly larger `(n + 2·taper)` footprint carrying a separable
Hann taper (overlap `2·taper`), scatter-added into the padded grid and
normalised by the overlap-sum so `Σᵢ wᵢ ≡ 1` everywhere. `taper ≤ halo` so the
PoU band lies inside the extended block.

### 14. The `extra_in_channels` widening of `UNetConvNeXt`

[unet_convnext.py](../libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py)

`UNetConvNeXt` gained `extra_in_channels: int = 0` and a
`forward(..., extra=None)` argument. When set, the stem widens to
`n_state_channels + 1 + extra_in_channels` and `extra` is concatenated
**after** the geometry mask, **raw** — no standardisation, no geometry masking
— so the DD wrapper can feed per-patch context (`C`) and positional encodings
straight in. The fine net is built with
`extra_in_channels = n_state_channels + n_pos`; the coarse net takes none.

The default `0` / `None` keeps the stem (and the whole state dict)
**byte-identical** to a model built without the argument, so pre-existing
`UNetConvNeXt` checkpoints still load. `set_normalization` on the wrapper
forwards the train-split statistics to **both** inner nets (a no-op for a net
built with `normalize=False`).

### 15. The key design property — it's a drop-in architecture

The decomposition is **embedded inside the model**: `DomainDecomposed.forward`
takes and returns **full-grid** tensors with exactly the
`forward(state, params, geometry) -> state_next` contract every other
architecture (§7) obeys. As a consequence:

- it trains with the **existing `Trainer` on the existing `TransitionDataset`**
  under a plain `MSELoss` on the merged prediction (no trainer or dataset
  changes — verified; this is the primary, validated milestone);
- it runs through the existing `NeuralSurrogateForwardModel` / ensemble /
  ESMDA path (§12) unchanged, save for the relaxed domain check of §16.

It is exported from
[architectures/__init__.py](../libs/neural-surrogates/src/neural_surrogates/architectures/__init__.py)
and the top-level
[neural_surrogates/__init__.py](../libs/neural-surrogates/src/neural_surrogates/__init__.py)
(`DomainDecomposed`, plus `DomainDecomposition` and `DomainDecompositionLoss`),
so its `_target_` is the flat `neural_surrogates.DomainDecomposed`.

### 16. Flexible grid at inference

[forward_model.py](../libs/neural-surrogates/src/neural_surrogates/forward_model.py)

`DomainDecomposed` advertises `domain_flexible = True`. The forward model's
`_check_domain` detects this (`getattr(self.model, "domain_flexible", False)`)
and switches to a **cell-spacing** invariant (`_check_domain_flexible`): the
requested and trained `(dx, dy, dz) = (hi − lo) / (nx, ny, nz)` must agree per
axis, but `nx/ny/nz` and the absolute bounds are otherwise free. The trained
global grid is no longer required — only the spacing. Every non-flexible model
keeps the strict `nx/ny/nz` + bounds equality check verbatim. `rollout_batched`
and the rest of the forward-model path are unchanged.

### 17. Periodicity

Global periodicity is configured once, on the decomposition, via
`decomposition.periodic_axes` in `(z, y, x)` order. The lab default is
**y-periodic** `[false, true, false]` (uDALES runs are spanwise-periodic; x
inflow-outflow and z ground/top are not). On a periodic axis:

- the **halo fill** wraps circularly (`F.pad(mode='circular')`) instead of
  using `boundary_mode`;
- the **PoU overlap wraps around** the domain — the `taper` overhang of the
  boundary tiles is wrap-added onto the opposite end before normalisation, so
  `Σᵢ wᵢ ≡ 1` holds *across the seam* and the merge is seamless (C0-continuous)
  there;
- the **positional encoding** uses a periodic signal
  (`sin(2π·wrapped_coord / g)`) instead of the signed wall-distance ramp, so a
  tile against the seam is not told it sits against a wall;
- `interior_size` must **divide** the periodic axis length exactly (periodic
  axes are not padded — padding would corrupt the ring length).

The **inner nets are NOT given `periodic_axes`** (the wrapper forces
`periodic_axes=()` on both): patch interiors are not periodic, and all global
periodicity is handled by the DD halo fill.

### 18. Two training paths

**(a) Drop-in path (primary, validated).** The existing
[`Trainer`](../libs/neural-surrogates/src/neural_surrogates/training/standard.py) +
[`TransitionDataset`](../libs/neural-surrogates/src/neural_surrogates/datasets/transition.py) +
`MSELoss` on the full merged prediction (§15). Nothing about §10 changes; this
is milestone 1.

**(b) Patch-based Eq (9) objective.** The four-term loss of companion PDF §2.7
in
[dd_loss.py](../libs/neural-surrogates/src/neural_surrogates/dd_loss.py)
(`DomainDecompositionLoss`):

| Term | Weight | Computed on |
|---|---|---|
| **one-step** | `1` | MSE of the *merged* next-state vs ground truth on fluid cells |
| **interface** | `λ_if = 0.1` | disagreement of adjacent patches' `(n+2·taper)` PoU blocks on their `2·taper` overlap band (`+z/+y/+x` faces, each shared band once; periodic wrap via `neighbor_indices`) |
| **divergence** | `λ_div = 0.01` | squared `∇·u` (central differences, `dx = 1`) of the merged velocity channels on fluid cells |
| **coarse** | `λ_c = 1.0` | MSE of `coarse_pred` vs `restrict_coarse(target)` |

The loss consumes the `info` dict returned by
`DomainDecomposed.forward(..., return_intermediates=True)` (`coarse_pred`,
`patch_pred` — the per-patch PoU blocks before windowing —, `context`,
`num_patches`, `dd`) alongside the merged prediction.

[`PatchTrainer`](../libs/neural-surrogates/src/neural_surrogates/training/patch.py)
mirrors `Trainer` (device handling, best-checkpoint saving) but trains on
**full-field** `TransitionDataset` batches via `return_intermediates=True`.
**Why full fields rather than per-patch items?** The interface and divergence
terms couple *neighbouring* patches; an isolated patch item cannot supply its
neighbours' predictions (they may land in a different minibatch, or be absent),
so those terms are only well-defined when the whole field — hence every
patch — is present each step.

[`PatchTransitionDataset`](../libs/neural-surrogates/src/neural_surrogates/datasets/patch.py)
is a **new** dataset that reads the **same on-disk `training_data/` layout**
(§1) but yields one sample per spatial patch: it subclasses `TransitionDataset`
to reuse its file walking, parameter loading and lazy two-snapshot reads, and
returns the extended block, its interior delta target, the patch geometry /
positional / neighbour table, and the (global, per-`(traj, t)`) coarse fields.
It is suited to one-step delta-only patch training; the coupled
interface/divergence terms still need `PatchTrainer`'s full-field path.
**`K = 1` is the supported patch path** — a `K > 1` patch pushforward needs the
full PoU merge (a patch's halo at `t+1` depends on its neighbours' interiors),
which only the model owns.

### 19. Config and CLI

The config group
[conf/neural_surrogate/architectures/domain_decomposed/](../conf/neural_surrogate/architectures/domain_decomposed/)
holds three presets. Each is a single
`_target_: neural_surrogates.DomainDecomposed` block with
`_recursive_: false` and `_convert_: all` (so the nested `decomposition` /
`fine_net` / `coarse_net` arrive as plain kwarg dicts — they are **not**
`_target_` nodes — and the wrapper builds `DomainDecomposition` /
`UNetConvNeXt(**...)` itself). `extra_in_channels`, `residual=True` and the
inner-net `periodic_axes=()` are fixed by the wrapper and must not be set in the
config; global periodicity lives under `decomposition.periodic_axes`.

| Preset | interior_size | halo | taper | coarsen | fine-net (base / mults / depths / kernel) |
|---|---|---|---|---|---|
| tiny | 16 | 4 | 2 | 4 | 8 / [1, 2] / [1, 1] / 3 |
| small | 32 | 8 | 4 | 4 | 16 / [1, 2, 4] / [1, 1, 1] / 5 |
| medium | 32 | 8 | 4 | 4 | 24 / [1, 2, 4] / [2, 2, 2] / 7 |

(All three use `periodic_axes: [false, true, false]`, `geometry_coarsen:
any_fluid`, `n_pos: 3`, FiLM conditioning, `normalize: true`, and a small
dedicated coarse net at `base_channels: 8`.)

**The architecture, trainer class and loss are bundled into the `mode` group.**
A single `mode` choice selects all three (plus `model_name`): `mode=standard` →
`p3d/medium` + `neural_surrogates.Trainer` + `MSELoss` (model_name
`p3d_barcelona`), `mode=domain_decomposition` → `domain_decomposed/small` +
[`PatchTrainer`](../libs/neural-surrogates/src/neural_surrogates/training/patch.py)
+ `DomainDecompositionLoss`. The mode entry sits **after** `_self_` in the
defaults list so it overrides the inline `trainer._target_`, and it supplies the
architecture default and the whole `loss:` config inline (there is no separate
`loss` group and no separate `trainer` config group — both are embedded in the
`mode/` files). The trainer's remaining fields stay in the inline `trainer:`
block in `training.yaml`. The architecture **family/size is still swappable on
the CLI** on top of the mode default (the override value is the nested
`family/preset` path) — e.g. to train a DD model as a full-grid drop-in
(generic Trainer + MSELoss):

```bash
# (a) Drop-in path: mode=standard but with a domain_decomposed architecture.
pixi run -e dev python scripts/neural_surrogate/train_neural_surrogate.py \
    neural_surrogate/mode@_global_=standard \
    'neural_surrogate/architectures@architecture=domain_decomposed/small' \
    model_name=domain_decomposed_small init_weights_path=null
```

```bash
# (b) Patch (Eq 9) path: PatchTrainer + DomainDecompositionLoss (the default mode).
pixi run -e dev python scripts/neural_surrogate/train_neural_surrogate.py \
    'neural_surrogate/architectures@architecture=domain_decomposed/small' \
    neural_surrogate/mode@_global_=domain_decomposition \
    model_name=domain_decomposed_small init_weights_path=null
```

Both trainers share the inline `trainer:` config and the same machinery via
`BaseTraining` (AMP, `torch.compile`, the pushforward-rollout curriculum, the
warmup+cosine LR schedule, gradient clipping, early stopping, checkpoint/resume),
so only `_target_` differs between the two paths. `DomainDecompositionLoss` cannot
be driven by the generic `Trainer` (its `forward` signature differs from a plain
element-wise loss), which is why the loss is paired with the trainer inside each
mode rather than chosen independently.

**Periodicity.** The single `architecture.periodic_axes: [y]` knob in
`training.yaml` drives a DD model exactly as it drives a plain `UNetConvNeXt`:
`DomainDecomposed` translates the axis-letter list into its decomposition's
`(z, y, x)` periodicity (overriding the preset's `decomposition.periodic_axes`);
the inner patch nets stay non-periodic. For DD, `interior_size` must divide `Ny`.

### 20. File map

| Piece | File |
|---|---|
| `DomainDecomposition` (operators, PoU, periodic wrap, positional, coarse pool/prolong) | [decomposition.py](../libs/neural-surrogates/src/neural_surrogates/decomposition.py) |
| `DomainDecomposed` (Algorithm 1; `domain_flexible`; `return_intermediates`) | [architectures/domain_decomposed.py](../libs/neural-surrogates/src/neural_surrogates/architectures/domain_decomposed.py) |
| `UNetConvNeXt` `extra_in_channels` widening | [architectures/unet_convnext.py](../libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py) |
| `PatchTransitionDataset` (per-patch dataset, same on-disk layout) | [datasets/patch.py](../libs/neural-surrogates/src/neural_surrogates/datasets/patch.py) |
| `DomainDecompositionLoss` (Eq 9, four terms) | [dd_loss.py](../libs/neural-surrogates/src/neural_surrogates/dd_loss.py) |
| `PatchTrainer` (full-field Eq-9 training) | [training/patch.py](../libs/neural-surrogates/src/neural_surrogates/training/patch.py) |
| Spacing-invariant domain check (`domain_flexible`) | [forward_model.py](../libs/neural-surrogates/src/neural_surrogates/forward_model.py) |
| Architecture presets `tiny` / `small` / `medium` | [conf/neural_surrogate/architectures/domain_decomposed/](../conf/neural_surrogate/architectures/domain_decomposed/) |
| Mode groups `standard` / `domain_decomposition` (bundle trainer class + loss + architecture default) | [conf/neural_surrogate/mode/](../conf/neural_surrogate/mode/) |
| Tests | [test_decomposition.py](../tests/test_decomposition.py), [test_domain_decomposed.py](../tests/test_domain_decomposed.py), [test_unet_convnext_extra_channels.py](../tests/test_unet_convnext_extra_channels.py), [test_patch_transition_dataset.py](../tests/test_patch_transition_dataset.py), [test_dd_loss.py](../tests/test_dd_loss.py), [test_dd_forward_model_flexible.py](../tests/test_dd_forward_model_flexible.py), [test_dd_training_wiring.py](../tests/test_dd_training_wiring.py) |

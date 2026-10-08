# Neural surrogates

Learned surrogates of the CFD forward models: training-data generation →
on-disk layout → PyTorch datasets → architectures → training → evaluation →
use as a forward model in the DA scripts. Read
[codebase_guide.md](codebase_guide.md) first for the forward-model / ensemble
abstractions referenced below. The library is
[libs/neural-surrogates/src/neural_surrogates](../libs/neural-surrogates/src/neural_surrogates/);
the scripts and configs are:

| Step | Script | Config (`configs/`) |
|---|---|---|
| Generate training data | [scripts/surrogate/generate_data.py](../scripts/surrogate/generate_data.py) | `surrogate/generate_data.yaml`, `params/surrogate_training_data.yaml` |
| Train | [scripts/surrogate/train.py](../scripts/surrogate/train.py) `--config-name surrogate/<config>` | `surrogate/{train_stepper,train_autoencoder,train_dft,finetune_stepper,train_latent_generator}.yaml`, each on top of `surrogate/training.yaml`; stepper architectures in `surrogate/architectures.yaml` |
| Evaluate | `scripts/surrogate/evaluate_{stepper,autoencoder,latent_generator}.py` | `surrogate/eval.yaml`, one block per script |
| Assimilate with it | `scripts/run_{smoother,filtering,hybrid}.py model@assim_model=neural_surrogate` | `model/neural_surrogate.yaml` |

Each script's module docstring lists its options and outputs. `train.py` builds
the datasets and model for the config's `task` in
[scripts/utils/tasks.py](../scripts/utils/tasks.py) and writes
`<paths.weights_dir>/<name>/` (`model_weights/` by default); training data
lives under `<paths.training_data_dir>/` (`training_data/`). SLURM wrappers are
in [job_scripts.md](job_scripts.md).

Parts: training data (A), data loading (B), next-step architectures and
training (C), the surrogate as a forward model (D), domain decomposition (E),
LoRA fine-tuning (F), Tadpole autoencoder pre-training (G), AE → time-stepper
(H) and generative spin-up by latent flow matching (I).

---

## Part A — Training-data generation

### 1. What the script produces

[scripts/surrogate/generate_data.py](../scripts/surrogate/generate_data.py)
runs one simulation per sample, each with its own sampled inflow trajectory,
and writes a `train` / `val` / `test` split to `paths.results_dir` =
`<paths.training_data_dir>/<model.name>_<data.geometry.name>/`:

```
training_data/<model>_<name>/
├── config.yaml                              resolved config of the run
├── state/{train,val,test}/sample_XXXX.nc    the field on cell centres + `blanking` (1 = building)
└── param/{train,val,test}/sample_XXXX.nc    the parameters at the state times
```

Every state file has a parameter file at the same relative path under
`param/`: the time-varying parameters linearly interpolated onto the state's
output times, scalar parameters (pyudales `pressure_gradient_magnitude`) kept
as scalars.

**Parameter time convention.** A network step from frame `t` to `t+1` is
driven by the parameters at its start, the time of frame `t`: training pairs
frame `t` with param row `t`, and inference
(`NeuralSurrogateForwardModel._param_schedule`) interpolates the parameters
onto `k · tf` for step `k` (#164). This needs the frames stamped at their
physical times, `tf … T`. Compatible corpora are those whose state `time`
starts at `output_frequency`: pylbm, PALM and spun-up uDALES corpora generated
after #163, and uDALES corpora without spin-up of any version. pylbm, PALM and
spun-up uDALES corpora from before #163 start at `time = 0`; their param row
`t` is one `tf` before the step's start, and models trained on them see a
one-`tf` offset at inference.

State files carry the attrs `geometry` (the STL stem, or the case
name) and `spinup_time_s` (the spin-up actually used). `data.save_vars` lists
the time-varying variables kept; time-independent ones such as `blanking` are
always kept.

**Grid collocation.** pyudales solves on a staggered C-grid (`u@xm`,
`v@ym`, `w@zm`); before saving, each state is linearly interpolated to cell
centres (`xt`, `yt`, `zt`) by
`pyudales.utils.grid_utils.interpolate_grid` so all channels share one
regular grid, and its `blanking` mask is rebuilt from uDALES's `solid_c.txt`
(uDALES leaves small non-zero velocities inside buildings, so the mask cannot
be recovered from the field). pylbm output is already cell-centred.

### 2. Config and geometry modes

[configs/surrogate/generate_data.yaml](../configs/surrogate/generate_data.yaml)
composes like `forward.yaml` (`/common`, `/case`, `/params`, `/model`) with a
`time` block for one trajectory (`simulation_time`, `output_frequency`,
`spinup_time`, `seconds_per_knot`; it overrides the case's window times) and a
`data` block:

| Key | Purpose |
|---|---|
| `num_train` / `num_val` / `num_test` | samples per split |
| `seed` | geometry split, parameter draws and uDALES inlet-turbulence seeds |
| `save_vars` | time-varying variables written (default `[u, v, w]`) |
| `pressure_gradient_magnitude` | constant forcing added for backends whose parameter schema has it (pyudales) |
| `adaptive_spinup` | `enabled`, `fill_times`, `max_spinup_time` (§3) |
| `state_encoding` | NetCDF encoding of the state files: `zlib`/`complevel` on every variable, `least_significant_digit` (lossy) on float variables only |
| `sharding` | `num_shards`, `shard_index` (§3) |
| `geometry` | `mode` (`fixed` \| `random`), the pool `source` with its `stl_dir` / `case_dir`, the output-name suffix `name`, and the random-mode grid: `resolution`, `upstream_padding`, `downstream_padding`, `lateral_padding`, `z_size` (metres) |

- **`mode: fixed`** — every sample runs on the case geometry and grid
  (`case=`); set `data.geometry.name=${case_name}` so the output folder is
  named after the case.
- **`mode: random`** (default) — every sample runs on a building layout from
  the STL pool in `stl_dir` (made by `scripts/tools/rasters_to_stl.py`), with
  the uDALES/PALM templates from `case_dir`. Val and test each get
  `num_val` / `num_test` distinct held-out geometries (one sample each); train
  cycles through the remaining ones when it needs more samples than
  geometries.

```bash
# random layouts from the realistic UrbanTALES pool, uDALES (the defaults)
pixi run -e dev python scripts/surrogate/generate_data.py
# the case geometry for every sample
pixi run -e dev python scripts/surrogate/generate_data.py model=pylbm case=barcelona \
    data.geometry.mode=fixed 'data.geometry.name=${case_name}'
# another pool, resolution and backend
pixi run -e dev python scripts/surrogate/generate_data.py model=pypalm \
    data.geometry.source=idealized data.geometry.resolution=2.0 data.geometry.z_size=64.0
```

### 3. How samples run

- **Per-geometry grid (random mode).** Each layout's physical size comes from
  the pool's `manifest.csv` (mesh bounds as fallback, which under-span the
  domain when the outer buildings sit inset from its edges). At `resolution`,
  the domain adds `upstream_padding` in front of the mesh, `downstream_padding`
  behind it and `lateral_padding` on both sides (each rounded up to whole
  cells), then rounds `nx` and `ny` up to multiples of 16: the x slack goes
  behind the mesh, the y slack is split over both sides (an odd cell at the
  far side). The domain window moves, not the mesh, so lower `bounds` go
  negative. `nz = z_size / resolution` (keep it a multiple of 16); layouts as
  tall as `z_size` are skipped.
- **One prepared model per geometry.** Samples on the same geometry run one
  after the other on one model, built and prepared once in a fresh scratch dir
  `<paths.experiment_dir>/data_<name>` (stale solver output from another grid
  is the classic uDALES fielddump trap). For pyudales the script copies the
  case template with `stl_file` rewritten in every `namoptions.*` and sets
  `precomputed_geom_dir=None`; pylbm and pypalm get `stl_path` (pypalm also
  `case_dir`). Samples on one uDALES model get distinct inlet-turbulence seeds
  (`seed * 100003 + member`).
- **Adaptive spin-up.** `time.spinup_time` is a floor. With
  `adaptive_spinup.enabled` the spin-up of a geometry is
  `fill_times * Lx / u_slow`, where `u_slow = 0.8 * U * max(cos θ, 0.1)` is the
  slowest initial streamwise inflow over that geometry's samples, rounded up
  to a whole `output_frequency` and clamped to
  `[spinup_time, max_spinup_time]`. It matters on the uDALES turbulent-inlet
  path, where the interior starts at rest and fills from the inlet face alone
  (see [pyudales.md](pyudales.md)). In fixed mode all samples share one
  geometry, hence one spin-up.
- **Resumable and shardable.** Both files of a sample are written atomically
  (hidden temp name, then rename; the parameters last) and a sample whose two
  files exist is skipped, so rerunning after a crash or with a larger
  `num_train` only runs what is missing. `data.sharding` splits the geometry
  groups between independent jobs (group `k` runs on shard
  `k % num_shards`); the split and the parameters are deterministic in the
  config, so every shard plans the same corpus. A failing geometry is logged
  and skipped, and the job raises at the end naming the failed geometries. On
  SLURM, submit one `surrogate_generate_data.slurm` per shard
  ([job_scripts.md](job_scripts.md)).
- **`ncpu` must divide every `nx`** for pypalm/pyudales (slab decomposition).
  Pool `nx` are multiples of 16, so `ncpu` ∈ {1, 2, 4, 8, 16} always works.

### 4. The parameter sampler

[configs/params/surrogate_training_data.yaml](../configs/params/surrogate_training_data.yaml)
selects
[`pyurbanair.training_data.UniformExternalAR2Sampler`](../src/pyurbanair/training_data/samplers.py):

```yaml
_target_: pyurbanair.training_data.UniformExternalAR2Sampler
_convert_: all
ensemble_size: 1                       # set to the number of samples by the script
external:
  inflow_angle:
    mean: {min: -45.0, max: 45.0}      # uniform per sample
    std: 15.0                          # fixed; use {min, max} to sample
  velocity_magnitude:
    mean: {min: 5.0, max: 10.0}
    std: 0.5
time_series:
  correlation_length: 300.0
```

For every sample and every parameter:

1. Draw `mean_e` from the `mean` spec — fixed (scalar) or `Uniform(min, max)`
   (dict). Same for `std_e`.
2. Integrate a critically-damped AR(2) anomaly `z(t)` (unit variance, smooth,
   correlation length `time_series.correlation_length`).
3. Return `x(t, e) = mean_e + std_e · z(t, e)`.

The script evaluates this on knots `time.seconds_per_knot` apart over
`[0, simulation_time]`, and the forward model interpolates between knots.
**Clipping:** AR(2) anomalies are unbounded, so a `{min, max}` `mean` spec also
clips the trajectory to that range (no solver-unsafe values); an explicit
`clip: {min?, max?}` block next to `mean`/`std` takes precedence.

### 5. Adding a sampler

A new sampler implements `sample_prior(time_coords, rng_key) ->
xarray.Dataset` with `(time, ensemble)` arrays per parameter (time-independent
ones as `(ensemble,)`) and accepts `ensemble_size`; put it in
`src/pyurbanair/training_data/` and point the `_target_` of a `configs/params/`
file at it.

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
§10 for how the trainer uses `K>1`. `num_history_steps` (`H`, default `1`)
is the mirror-image *backward* window — see below. Each item is a dict:

| Key          | Shape       | Notes |
|---|---|---|
| `state_n`    | `(H·C, *grid)` | velocity channels stacked in `state_vars` order, for the history window `t−H+1 … t` flattened onto the channel axis **oldest first** (`H = num_history_steps`, default `1` → the plain `(C, *grid)` snapshot at `t`) |
| `state_next` | `(C, *grid)` | snapshot at time `t + K` — the pushforward target |
| `params_n`   | `(K, P)`    | inflow params at steps `t, …, t+K-1`; scalar params (e.g. uDALES `pressure_gradient_magnitude`) are broadcast along `time` |
| `geometry`   | `(*grid,)`  | binary mask: `1` = fluid, `0` = obstacle. The item's *trajectory's* mask; equal masks are content-deduped to one shared tensor at init |
| `geom_features` | `(C, *grid)` | SDF / ∇SDF channels — present **only** when built with a non-`none` `sdf_features` mode (default `none`); `C` = 1 (`sdf`), 3 (`grad`) or 4 (`both`); one tensor per unique mask |

The geometry mask is read from each state file's `geometry_var`
(default `"blanking"`, the obstacle mask every generated corpus carries
(§1), inverted to the `1`-is-fluid convention). For data without one,
the fallback marks fluid cells as those with a non-zero stacked state in
that trajectory's first snapshot; ground-and-building cells stay 0.

#### State history (`num_history_steps`)

`num_history_steps` (`H`, default `1`) is the **backward** window — the
mirror image of `pushforward_steps` (`K`, the forward horizon). With
`H>1` each sample carries the `H` consecutive snapshots `t−H+1 … t`
instead of the single snapshot at `t`; `state_next`, `params_n`,
`geometry` and `geom_features` are unchanged.

The window is **pre-flattened onto the channel axis, oldest first**, so
`state_n` is `(H·C, *grid)` with channel order `[u,v,w @ t−H+1, …, u,v,w
@ t]` — the newest frame is always the last `C` channels, `state_n[-C:]`.
That one convention is what the architectures take as their residual base
(§7), what the trainer's ring buffer appends to (§10) and what the
forward model's rollout buffer holds (§12); nothing downstream ever sees
a time axis. `transition_collate` is shape-agnostic and stacks it to
`(B, H·C, *grid)`.

Anchors start at `t = H−1` so every sample has a full window: a
trajectory of length `T` contributes `T − K − (H−1)` samples (the window
trims the front, the horizon still trims the back) and needs at least
`K + H` snapshots — a shorter one raises a `ValueError` naming both
knobs. `set_pushforward_steps` rebuilds the index with the same offset,
so `H` survives the pushforward curriculum (§10).

`H=1` reproduces the historyless item byte for byte (same keys, same
tensors, same `len()`), so existing configs and checkpoints are
unaffected. `PatchTransitionDataset` accepts the key but raises
`NotImplementedError` for `H>1` — its per-patch item schema and the
domain-decomposed model both assume one `C`-channel block (§14) — and
`SnapshotDataset` is unaffected.

#### Multi-geometry splits (`TrajectoryBatchSampler`)

Random-geometry training data (§3) gives every trajectory its own grid
and geometry. The dataset handles this natively: geometry, SDF features
and state shapes are all per-trajectory, with equal masks deduped to one
tensor object — a single-geometry split therefore behaves byte-identically
to the fixed-domain dataset (one mask in memory, plain shuffled batching
works unchanged). What plain shuffling *cannot* do on a multi-geometry
split is stack different grids into one batch, so
[`TrajectoryBatchSampler`](../libs/neural-surrogates/src/neural_surrogates/datasets/sampler.py)
draws every batch from a single trajectory: `default_collate` sees one
shape and `transition_collate` ships that trajectory's mask (it fails
loud on a mixed-geometry batch rather than training against the wrong
mask). Batches are shuffled across trajectories each epoch, and the
per-trajectory batch size is `min(batch_size, max(1, cell_budget // cells))` —
the `cell_budget` (total grid cells per batch) keeps memory flat across
domain sizes (the UrbanTALES realistic pool spans ~25× in cell count).
The sampler re-reads the dataset's flat index at each epoch, so the
pushforward curriculum (`set_pushforward_steps`) propagates without a
rebuild. [training.yaml](../configs/surrogate/training.yaml) ships it in
the `batch_sampler:` block (`batch_sampler: null` falls back to the plain
shuffled DataLoader); the trainer keys its device-side geometry/SDF cache on the batch's mask,
so `state_mean`/`std` stats, the masked loss and the SDF features always
match the current batch's trajectory. The P3D wrapper itself accepts any
grid divisible by 16 (§7), which the random-geometry generator
guarantees. With `torch.compile` P3D requires `dynamic=False`, i.e. one
recompile per distinct grid shape — fine for the idealized pool (14
shapes at 2 m resolution), impractical for the realistic pool (~98
shapes); leave `compile_model: false` there.

#### SDF geometry features (`sdf_features`)

Built with a non-`none` `sdf_features` mode (default `none`), the dataset also
ships a geometry-feature tensor `geom_features` alongside the mask, drawn from the
clamped signed distance field `sdf_n` and its unit gradient `(g_z, g_y, g_x)`, all
bounded in `[-1, 1]` and in **cell** units (see
[`neural_surrogates.sdf.sdf_features`](../libs/neural-surrogates/src/neural_surrogates/sdf.py)).
The `sdf_features` mode selects which channels: `sdf` (the SDF only, `C=1`), `grad`
(the gradient only, `C=3`) or `both` (`C=4`); `true`/`false` are accepted as
aliases for `both`/`none`, so already-trained configs load unchanged. The
sub-modes are exact slices of the full 4-channel stack — the full stack is always
computed (the gradient needs the unclamped SDF) and then sliced, so a mode can
never diverge from the full computation.
`sdf = EDT(mask) − EDT(1 − mask)` is positive in fluid, negative in solid; the
gradient (from the *unclamped* SDF, normalised) points away from the nearest
wall. The single EDT is computed **once at init** from the (shared) mask — never
per `__getitem__` and never inside the training step — and `transition_collate`
ships it once per batch as `(1, C, *grid)`, exactly like the mask. `sdf_clamp_cells`
(default `32`) is the clamp radius `L`: `sdf_n = clip(sdf, −L, L) / L`. The
features enter the model stem **raw** (no z-scoring, no masking) and are excluded
from `_compute_normalization_stats`. Only `P3D` consumes them today (§10, §7);
see the SDF plan in [archive/sdf_features_plan.md](archive/sdf_features_plan.md).

#### Memory model

`__init__` only walks each state file to read `ds.sizes["time"]` (metadata
only) so it can build the flat `(traj, t)` index. Parameters and the
static geometry mask are loaded eagerly (both are small). State
snapshots are read lazily on each `__getitem__` via a single
`xr.open_dataset(..., cache=cache).isel(...)` covering that sample's own
slices (`[t, t+K]`, or the `H` history frames plus `t+K`) — only those
ever leave disk per sample, regardless of `K`. The
intermediate ground-truth states are never read because the pushforward
unroll feeds the model its own predictions in their place.

The `cache` constructor flag (`cache: bool = False`) is threaded straight
into xarray:

- `cache=False` (default) — every `.values` read goes to disk; only the
  current sample's own slices are materialized; nothing accumulates.
  Use this for large datasets that don't fit in RAM.
- `cache=True` — xarray keeps every read slice in memory, so after one
  epoch all visited trajectories are resident and subsequent epochs are
  disk-free. Use this when the dataset comfortably fits in RAM and you
  want maximum iteration throughput.

State file handles are kept in a per-process `_state_cache` dict; a
`__getstate__` hook drops the cache before pickling so each `DataLoader`
worker rebuilds its own handles (avoids sharing netCDF descriptors across
processes).

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
| Run script | [scripts/surrogate/train.py](../scripts/surrogate/train.py) (`task: stepper`, built in [tasks.py](../scripts/utils/tasks.py)) |
| Config | [configs/surrogate/train_stepper.yaml](../configs/surrogate/train_stepper.yaml) + [training.yaml](../configs/surrogate/training.yaml) |

All architectures share the contract
`forward(state, params, geometry) -> state_next`. The geometry mask is
concatenated to the state along the channel dimension at the stem; how
parameters enter depends on the architecture.

**State history (`num_history_steps`).** Every next-step architecture —
`SimpleConv`, `UNetConvNeXt`, `P3D`, `UPT` — takes `num_history_steps`
(`H`, default `1`) as a trailing keyword argument and exposes
`self.num_history_steps` and `self.n_input_state_channels = H ·
n_state_channels`. The `H` past frames arrive **pre-flattened onto the
channel axis, oldest first** (§6), so `state` is `(B, H·C, *grid)` and
the newest frame is `state[:, -C:]`; the architectures never see a time
axis. Only the **input** widens — `SimpleConv.conv` / `UNetConvNeXt.stem`
/ P3D's `in_channels` / UPT's `feat_dim` count `H·C` state channels —
while the output head still emits `n_state_channels`. The z-score buffers
`state_mean` / `state_std` also stay length `C`, so `set_normalization`
and existing checkpoints are untouched and `_normalization_signature`
needs no history key; each architecture tiles them `H`× at its one input
use site. Residual-predicting models add the **newest** frame
(`state[:, -C:] + out`), and the geometry mask broadcasts over the whole
`H·C` input. `H=1` gives an identical state dict and a bit-identical
forward output versus a model built without the argument. The dataset and
the model must agree: `train.py` stamps the dataset's `num_history_steps`
onto the architecture (§10). `DomainDecomposed` (§14) and `TadpoleTimeStepper` (§31)
accept the key but raise `NotImplementedError` for `H>1`.

**SDF geometry features (P3D).** `P3D` accepts an optional `sdf_features` mode
(`none` | `sdf` | `grad` | `both`, default `none`) that widens the stem by the
selected channels, inserted right after the mask: stem order `[state, geometry,
<sdf channels>, params, extra]` where `<sdf channels>` is `sdf_n` (`sdf`), `g_z,
g_y, g_x` (`grad`), or all four (`both`). It advertises `n_geom_feature_channels`
= 1 / 3 / 4 accordingly (`true`/`false` alias `both`/`none`). The dataset and the
model must agree on the mode: `train.py` copies the architecture's mode onto the
dataset (§10).
During **training** the features arrive precomputed via a `geom_features=`
argument (shipped by `TransitionDataset`/`transition_collate` when
`dataset.sdf_features` is non-`none`; the trainer keys off `n_geom_feature_channels`
and fails loud if the batch lacks them). During **inference** callers keep the
`(state, params, geometry)` contract unchanged — the model self-computes the
features from the mask and caches them on the geometry tensor's identity, so the
EDT runs once per rollout (step 1) and every later step hits the cache. The
default (`none`) keeps `in_channels` and the whole state dict byte-identical, so
existing checkpoints load untouched. The EDT is **never** run inside the training
step or a `torch.compile`'d region. Other architectures can adopt the same
`n_geom_feature_channels` + `geom_features=` convention later.
The autoregressive driver (`NeuralSurrogateForwardModel._rollout_chunk`) also
computes `geom_features` **once per rollout** and passes it into every internal
step for models that advertise `n_geom_feature_channels > 0` **and** expose the
`_sdf_features` hook (the Tadpole field-IO mixin), so an SDF-enabled stepper that
lacks its own identity cache (e.g. the Tadpole time-stepper) runs the EDT only
once rather than per step. Every other model keeps the byte-identical
`(state, params, geometry)` call: P3D with SDF features self-computes and caches
on the geometry tensor's identity internally (as above), and `none` / UPT expose
no feature channels.

### 8. `SimpleConv` — baseline

Single `Conv3d` layer over `(state ⊕ geometry)` along the channel dim.

- **Input channels**: `n_state_channels + 1` — the state channels stacked
  in `state_vars` order, with the binary geometry mask appended
  (`num_history_steps · n_state_channels + 1` under a history window, §7).
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

- **Stem**: `Conv3d(n_state_channels + 1, base_channels, 3)` — widened to
  `num_history_steps · n_state_channels + 1 + extra_in_channels` when
  either knob is set (§7, §14).
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

[architectures.yaml](../configs/surrogate/architectures.yaml) holds
`unet_convnext_{tiny,small,medium,large,xlarge}` (all with
`separable_dwconv`, `normalize`, `residual`, FiLM conditioning):

| Preset | base | mults | depths | kernel | expansion |
|---|---|---|---|---|---|
| tiny | 8 | [1, 2] | [1, 1] | 3 | 2 |
| small | 16 | [1, 2, 4] | [1, 1, 1] | 5 | 4 |
| medium | 32 | [1, 2, 4, 8] | [3, 3, 3, 3] | 7 | 2 |
| large | 32 | [1, 2, 4, 8] | [2, 2, 2, 2] | 7 | 4 |
| xlarge | 48 | [1, 2, 4, 8] | [3, 3, 3, 3] | 7 | 4 |

The same file holds `p3d_{tiny,small,medium,large,xlarge}` (`P3D` sizes S/B/L;
`p3d_medium`, the `train_stepper` default, adds `sdf_features: both`).

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

**Normalization stats.** Computed by `get_normalization_stats`
([training/data_utils.py](../libs/neural-surrogates/src/neural_surrogates/training/data_utils.py)):
streamed file-by-file over the training split in
float64, restricted to fluid cells via each trajectory's geometry mask
(per-trajectory on multi-geometry splits). Installed via
`model.set_normalization(state_mean, state_std, param_mean, param_std)`;
stored as buffers `state_mean/state_std/param_mean/param_std` and saved
with the checkpoint so rollout/test callers get the correct
standardisation for free.

The streaming pass is slow on large splits, so it is **cached in the data
folder** at `<root_dir>/normalization_stats/<split>.npz`. The cache is keyed by a signature of every
input the computation reads — dataset class, split name, `state_vars`/`param_names`
order, `geometry_var`, and each state file's `(name, size, mtime)` — so
regenerating or editing the training data (or changing any of those
knobs) invalidates it and forces a recompute; `pushforward_steps` is not
part of the key (the stats span every snapshot regardless of horizon). A
missing/corrupt/stale cache silently recomputes, and a write failure is
non-fatal. This is purely a training-time speedup — inference still reads
the stats from the checkpoint buffers, not this file.

**Shared-geometry fast path.** Within one forward call, all batch members
voxelise the same STL onto the same grid, so one point set and supernode
graph serves the whole batch. The shared-geometry guard checks that all
batch members have the same number of fluid cells (`O(B)` reductions per
step, not `O(B·N)`) and falls back to a per-sample loop only when they
differ.

#### Size presets

[architectures.yaml](../configs/surrogate/architectures.yaml) holds
`upt_{tiny,small,medium,large,xlarge}` with `attention_type: dot_product`;
`normalize`, `predict_residual` and `cond_dim` keep their class defaults.

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

**State history.** `BaseTraining` reads the backward window off the eager
model (`num_history_steps` and `n_state_channels`, both `getattr`-defaulted
so a pre-history architecture stays on the one-step path). There is no new
trainer config key — the window is a property of the architecture. The
pushforward rollout then *rolls* its input instead of replacing it:
`_advance_history(state, pred)` returns `pred` itself at `H=1`
(byte-identical to the old `state = self._model_forward(...)`) and otherwise
`cat([state[:, C:], pred], dim=1)` — drop the oldest frame, append the
prediction, so the newest frame stays last. It is called at both rollout
sites (the `no_grad` prefix steps and the gradient-bearing ones). Nothing
else moves: `state_n` is still 5-D, so the `channels_last_3d` cast and the
geometry `expand` are unaffected, and both `_final_loss` hooks compare a
`(B, C, *grid)` prediction against `state_next` regardless of `H`.
`PatchTrainer` only ever sees `H=1` (§14).

**LR schedule.** When `lr_warmup_epochs` is set the optimizer's LR ramps
linearly from `lr_warmup_start` to its configured peak over the warmup
window, then cosine-anneals down to `lr_min` over the remaining epochs.
`lr_warmup_epochs=None` (default) keeps the LR fixed.

**Best-checkpoint saving.** When `weights_path` is set, the trainer
writes `model.state_dict()` to that path every time the val loss
improves, and reloads it into the model at the end of `fit()` so the
returned model is the best-val checkpoint (not the last epoch). `train.py`
passes `weights_path=<weights_dir>/<name>/weights.pt`, so nothing needs to
be saved by the caller after `fit()`.

Every best-weight save also writes `best_val.json`. On resume this score takes
precedence over a worse score in an older periodic checkpoint, for both plain
training and merged LoRA exports. Plain checkpoints predating the sidecar are
handled by evaluating the saved best weights once, then restoring the resumable
checkpoint model and torch RNG before continuing. Optional `checkpoint_metadata`
lets a training script attach a semantic contract to the checkpoint; a supplied
contract must match saved metadata before any model/optimizer state is loaded.
Increasing `num_epochs` on resume extends the cosine horizon while preserving
the checkpoint's current learning rate. It continues decreasing to `lr_min`;
a run already at `lr_min` stays there instead of cycling back upward.

**Early stopping.** `patience` (class default `None`, disabled; `20` in
`training.yaml`) halts training after that many epochs without val-loss
improvement.

The model and dataloaders are deliberately **constructed outside** the
trainer and passed in — this keeps `Trainer` agnostic to backend choice,
augmentation, and config structure.

### Run script and config

`train.py` with `task: stepper`
([train_stepper.yaml](../configs/surrogate/train_stepper.yaml)) does, via
`_stepper` in [tasks.py](../scripts/utils/tasks.py):

1. Copy `sdf_features` / `sdf_clamp_cells` from the architecture onto the
   dataset, so the dataset ships the geometry features the model takes.
2. `instantiate(cfg.dataset, split="train"|"val", dtype=...)` → two
   `TransitionDataset`s.
3. Stamp the dataset's `num_history_steps` and `param_vars` onto both
   `dataset:` and `architecture:` (the forward model (§12) rebuilds the net
   from `architecture` alone; the evaluation reads `dataset`), then
   `instantiate(cfg.architecture, n_state_channels=..., n_params=...)`.
   `init_weights` (a `weights.pt`) warm-starts it.
4. If the model has `set_normalization`, install the train-split stats from
   `get_normalization_stats` ([training/data_utils.py](../libs/neural-surrogates/src/neural_surrogates/training/data_utils.py),
   fluid cells only, float64, cached as in §9a).
5. Save the config to `<weights_dir>/<name>/config.yaml`, build the loaders
   (`build_loader`: the `TrajectoryBatchSampler` when `batch_sampler` is set,
   val unshuffled), and `instantiate(cfg.trainer, ..., optimizer=..., loss_fn=...,
   weights_path=<weights_dir>/<name>/weights.pt).fit()`. Rebuilding the
   architecture from `config.yaml` and loading `weights.pt` gives the trained
   model.

[training.yaml](../configs/surrogate/training.yaml) holds the shared
defaults (`trainer`, `loss: MSELoss`, `optimizer: AdamW`, `batch_sampler`,
`dataloader`, `dataset`, `paths.data_dir`); each training config overrides
what differs. The stepper architecture is picked from
[architectures.yaml](../configs/surrogate/architectures.yaml) (quote the
interpolation on the CLI):

```bash
pixi run -e dev python scripts/surrogate/train.py --config-name surrogate/train_stepper
pixi run -e dev python scripts/surrogate/train.py --config-name surrogate/train_stepper \
    'architecture=${architectures.unet_convnext_large}' name=unet_large \
    paths.data_dir=training_data/pylbm_barcelona \
    dataset.pushforward_steps=4 dataset.num_history_steps=3 \
    trainer.num_epochs=20 optimizer.lr=5e-4
```

### 11. Evaluating steppers

[scripts/surrogate/evaluate_stepper.py](../scripts/surrogate/evaluate_stepper.py)
(block `stepper` of [eval.yaml](../configs/surrogate/eval.yaml)) rolls one or
several trained steppers out on the **same** test trajectories. Each model is
rebuilt from its `config.yaml` + `weights.pt` and its own training dataset
class on `split` (top-level `data_dir` puts every model on one dataset; `null`
uses each model's own). The dataset is built with `sdf_features=none`: an
SDF-consuming model computes its features from the mask at inference. A model
with `num_history_steps = H` is seeded with the true frames `0 … H-1` (the
first `H` frames of its prediction are the truth) and then steps with the
trajectory's own parameters up to `max_steps` frames (`null`: the whole
trajectory).

```bash
pixi run -e dev python scripts/surrogate/evaluate_stepper.py \
    'stepper.models=[model_weights/p3d_a,model_weights/p3d_b]' 'stepper.sample_indices=[0,1]'
```

Outputs in `stepper.output_dir`: `metrics.csv` (RMSE, MAE, relative L2,
final-step RMSE and rollout time per model and sample, plus their mean),
`rmse.png` / `rmse_mean.png` (RMSE per rollout step), `summary_metrics.png`,
`slices.png` (`|U|` truth / prediction / error), `tke_slices.png` (resolved
TKE, the per-cell time variance), `params.png` (the trajectory's parameters
with the inflow angle and speed recovered at the inlet) and `rollout.mp4`
(`animate: true`).

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
the folder `train.py` writes (§10):

| Read from | Supplies |
|---|---|
| `model_dir/config.yaml` → `architecture` | the network to rebuild |
| `model_dir/config.yaml` → `dataset.state_vars` / `param_vars` | channel & parameter ordering (`param_vars: null` → read from the first training param file) |
| `model_dir/weights.pt` | trained parameters |
| `dataset.root_dir/config.yaml` → `domain` | the **trained domain** the requested grid is checked against (for a random-geometry corpus this is the case template, not any trajectory's grid: pass `trained_domain` explicitly) |
| `dataset.root_dir/config.yaml` → `time.output_frequency` | the **trained step size** (one network step) |

Each can still be overridden explicitly (handy for tests), but the normal
path is to only set `model_dir`.

Key behaviours:

| Concern | Behaviour |
|---|---|
| **Trained step size** | The network always advances at its trained cadence (`trained_output_frequency`). To honour a requested `output_frequency` that differs, the rollout emits a frame at the internal step closest to each requested output time — so the result lands on the requested grid whether or not the two cadences divide evenly. A requested cadence *finer* than the trained step (the surrogate can't emit between steps) raises. |
| **Domain check** | The requested `(nx, ny, nz, bounds)` must equal `trained_domain`; a mismatch raises (the network only applies to its training grid). |
| **Spin-up / collocation** | With `spinup_source: forward_model` a cold start (`state is None`) is bootstrapped by `spinup_forward_model` — the CFD backend that generated the training data — whose final field seeds the rollout. Because the training data is collocated to cell centers (pyudales' staggered C-grid → `xt/yt/zt`; §1), the spin-up field is collocated the same way and renamed to `(z, y, x)` *before* it reaches the network, so the inputs match what it trained on. Warm starts (a `state` is passed) skip spin-up; collocation is idempotent, so the surrogate's own regular-grid output passes through unchanged. `disable_spinup()` propagates to the backend. With `spinup_source: training_data` the surrogate runs **no** spin-up of its own — the caller must supply warm-start states (see below), so a cold start (`state is None`) raises. With `spinup_source: generative` a cold start is **sampled** from a trained latent generator conditioned on the member's current parameters (Part I, §40); the CFD backend is neither built nor run. |
| **Geometry** | When `stl_path` is set the geometry channel is voxelised from the STL onto the grid ([geometry.py](../libs/neural-surrogates/src/neural_surrogates/geometry.py)); otherwise it falls back to the non-zero-state convention used by `TransitionDataset`. |
| **Parameters** | Time-varying inflow params are interpolated onto the start of each internal step, `k · trained_output_frequency` (§1, parameter time convention), in the trained `param_vars` order; scalar params are broadcast. |
| **State history** | `num_history_steps` (`H`) is read **off the built network** — there is no forward-model config knob. The rollout buffer is `(B, H·C, *grid)`, oldest first; each step feeds it to the net, appends the `(B, C, *grid)` prediction, drops the oldest frame, and **emits the prediction** rather than the wider buffer. `_output_schedule()` (`n_internal`, `emit_steps`) is unchanged, so a history rollout emits exactly as many frames as an `H=1` one and substepping stays orthogonal. At `H=1` it is the historic loop, tensor for tensor. Seeding policy below. |

`NeuralSurrogateEnsembleForwardModel`
([ensemble_forward_model.py](../libs/neural-surrogates/src/neural_surrogates/ensemble_forward_model.py))
clones each member by sharing the (stateless) network and cloning the
spin-up backend into its own experiment directory via that backend's
`create_new_forward_model` helper.

**Seeding a history-conditioned rollout.** `_get_template_and_initial_state`
returns a single snapshot at `H=1`, as always; at `H>1` it returns the same
template but with a `time` dimension of length `H` (oldest first), and
`_stack_history` is the only reader of the whole window — `_build_geometry`
and `_assemble_output` already reduce a template with `isel(time=-1)`. What
lands in the window is decided by `_history_window`:

- the field carries `≥ H` time frames → the **last `H`**, oldest first;
- fewer than `H` (or no `time` dim at all) → the **oldest available frame
  is repeated** to fill the buffer, with a `RuntimeWarning` emitted **once
  per process** (an ESMDA run would otherwise print it per member per
  window).

Repeat-seeding is correct but degraded — the first predictions see a
frozen, zero-tendency history — so treat the first `~H` frames of such a
rollout as transient. Two operational cases hit it:

- **A CFD cold start.** A spin-up backend configured with
  `simulation_time = output_frequency` hands over exactly one post-spin-up
  frame, so any `H>1` takes the repeat path. Raise the spin-up backend's
  `simulation_time` to at least `H · output_frequency` and the surrogate
  picks up the last `H` frames of its trajectory automatically.
- **A short ESMDA window.** Windows after the first warm-start from the
  previous window's forecast; a window shorter than `H` output frames
  carries fewer than `H` of them, so the oldest is repeated again. Keep
  `simulation_time / output_frequency ≥ num_history_steps` per
  assimilation window.

`NeuralSurrogateEnsembleForwardModel` needed no functional change: both
`_spinup_templates` and `_warm_start_templates` already hand the forward
model the member's *whole* state, `time` dimension included, and let it
reduce the window — so a per-member `state_{i}.nc` may now legitimately
hold several time steps.

### Config and usage

[configs/model/neural_surrogate.yaml](../configs/model/neural_surrogate.yaml)
mirrors the other `configs/model/*.yaml` files (`name`, `solver_name`,
`forward_model`, `ensemble_model`, `prepare`). `forward_model` points at a
`model_dir` and uses `_recursive_: false` so the surrogate fills in
`n_state_channels` / `n_params` and builds its spin-up backend itself.
`prepare` runs `prepare_neural_surrogate`, which compiles/preprocesses the
spin-up backend (a no-op for `training_data` and `generative`).
`default_params` gives constant fallbacks for trained parameters a caller
omits (DA varies only the inflow, but a uDALES-trained net also expects
`pressure_gradient_magnitude`). `rollout_batch_size` caps the members rolled
through the network in one batched pass.

Select it like any backend:

```bash
pixi run -e dev python scripts/run_smoother.py \
    model@assim_model=neural_surrogate \
    assim_model.forward_model.model_dir=model_weights/p3d_barcelona \
    assim_model.forward_model.spinup_source=forward_model
```

**Spin-up sources.** `forward_model` runs the CFD backend; `generative`
samples the cold start (§40). `training_data` (the config default) has **no**
cold start of its own: the caller must hand every member a warm-start state,
or the run raises. The helpers that build those states (the last `H` frames of
a training sample per member, streamed to disk, and the prior inflow anchored
to that sample's final value) are in
[`neural_surrogates.training_spinup`](../libs/neural-surrogates/src/neural_surrogates/training_spinup.py),
configured by the model-level `training_data_spinup` block, but the current DA
scripts do not call them (the archived `run_esmda.py` did), so use
`forward_model` or `generative` with them.

### Extending

- **New architecture**: add a module under
  [libs/neural-surrogates/src/neural_surrogates/architectures/](../libs/neural-surrogates/src/neural_surrogates/architectures/),
  re-export it from
  [architectures/__init__.py](../libs/neural-surrogates/src/neural_surrogates/architectures/__init__.py)
  and the top-level
  [neural_surrogates/__init__.py](../libs/neural-surrogates/src/neural_surrogates/__init__.py)
  (for a flat `_target_`), and add `<family>_<size>` entries to
  [architectures.yaml](../configs/surrogate/architectures.yaml). The
  `Trainer` does not need to change as long as the model accepts
  `(state, params, geometry)`.
- **New optimizer / loss / loader**: change the `_target_` (and kwargs) in
  [training.yaml](../configs/surrogate/training.yaml) or on the CLI.
- **New trainer behavior** (schedulers, checkpointing, logging): extend
  `Trainer` and point `trainer._target_` at it.

---

## Part E — Domain decomposition (two-level, Recommendation A)

A learned one-step surrogate whose **spatial decomposition lives inside the
model**. Instead of one network spanning the whole grid, a two-level
overlapping decomposition splits the domain into a uniform batch of
overlapping patches, runs a shared per-patch *fine* net, and stitches the
patch outputs back together with a partition-of-unity (PoU) blend — while a
small *coarse* net supplies global context. The design (companion PDF §2 +
§5, Algorithm 1) is described in
[archive/dd_implementation_plan.md](archive/dd_implementation_plan.md).

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

`DomainDecomposed` also accepts `num_history_steps` (§7) — so a Hydra node
carrying the canonical key instantiates, and the trainer / forward model can
read the attribute uniformly — but raises `NotImplementedError` for anything
but `1`: the patch tiling, the coarse average-pooling and the fine-net
chunking all assume `C` state channels per block, and the fine net's
`extra_in_channels = n_state_channels + n_pos` context contract is written
against a single frame. The key is deliberately **not** forwarded to the
sub-nets, so `fine_net.num_history_steps == 1`. For `H>1` use a plain
next-step architecture; `PatchTransitionDataset` rejects
a history window too (§6). `TadpoleTimeStepper` (§31) rejects it for the same
class of reason — its frozen, pre-trained AE encodes exactly `C` state
channels (plus the geometry block) per crop.

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

### 19. Config

No shipped config uses domain decomposition: the `domain_decomposed` presets
and the `mode` group went with the archived setup. A `DomainDecomposed` node
takes `_recursive_: false` and `_convert_: all` (the nested `decomposition` /
`fine_net` / `coarse_net` are plain kwarg dicts, not `_target_` nodes; the
wrapper builds `DomainDecomposition` / `UNetConvNeXt(**...)` itself).
`extra_in_channels`, `residual=True` and the inner-net `periodic_axes=()` are
fixed by the wrapper and must not be set; global periodicity lives under
`decomposition.periodic_axes`, and `interior_size` must divide every periodic
axis. `DomainDecompositionLoss` cannot be driven by the generic `Trainer` (its
`forward` signature differs from an element-wise loss), so the patch path
pairs it with `PatchTrainer`; both trainers share `BaseTraining`.

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
| Tests | [test_decomposition.py](../tests/neural_surrogates/test_decomposition.py), [test_domain_decomposed.py](../tests/neural_surrogates/test_domain_decomposed.py), [test_unet_convnext_extra_channels.py](../tests/neural_surrogates/test_unet_convnext_extra_channels.py), [test_patch_transition_dataset.py](../tests/neural_surrogates/test_patch_transition_dataset.py), [test_dd_loss.py](../tests/neural_surrogates/test_dd_loss.py), [test_dd_forward_model_flexible.py](../tests/neural_surrogates/test_dd_forward_model_flexible.py) |

---

## Part F — Parameter-efficient fine-tuning (LoRA / PEFT)

Take an already-trained next-step surrogate (focus: `P3D`, but
architecture-agnostic), inject LoRA adapters, train **only** the adapter weights
on new data with the existing `Trainer`, and export a fine-tuned `model_dir`
that drops into `NeuralSurrogateForwardModel` unchanged. This is plan 01 of
the [surrogate plans](plans/implemented/neural_surrogates/00_master_plan.md); `method: full` (and the stepper's
`init_weights` warm start) remain for full-weight fine-tuning.

### 21. `neural_surrogates.finetuning`

[libs/neural-surrogates/src/neural_surrogates/finetuning/](../libs/neural-surrogates/src/neural_surrogates/finetuning/)
wraps HF **PEFT** — the single LoRA stack for the whole repo (the Tadpole and
BaLoRA plans reuse it). `peft` is a **lazy import** inside the submodules, so
`import neural_surrogates` stays free of the transformers/accelerate stack; it is
declared in the pixi `neural-surrogates` feature and as the package
`[finetuning]` extra.

| Symbol | Role |
|---|---|
| `inject_lora(model, *, rank, alpha, dropout, target_modules, variant, modules_to_save)` | `get_peft_model(model, LoraConfig(...))`. `variant="balora"` raises `NotImplementedError` (plan 04). Returns a `PeftModel` whose forward preserves our `(state, params, geometry, geom_features)` contract and forwards attribute access (e.g. `n_geom_feature_channels`) to the base model. |
| `merge_to_state_dict(peft_model)` | `merge_and_unload()` on a **deepcopy** → a plain, full base-architecture state dict (no `lora_*` keys), byte-loadable into a fresh architecture. The ESMDA-critical export. |
| `save_adapter` / `load_adapter` | PEFT adapter checkpoint I/O (`adapter_model.safetensors` + `adapter_config.json`). |
| `resolve_target_modules(model, *, preset, target_modules)` | Explicit `target_modules` (regex/list) wins; else the per-architecture `preset`. |
| `all_adaptable_module_names(model)` | Every LoRA-adaptable `nn.Linear`/`nn.Conv3d` leaf, used by the `all` preset. |

**P3D target presets** ([targets.py](../libs/neural-surrogates/src/neural_surrogates/finetuning/targets.py), derived by enumerating `P3D(...).net.named_modules()`):

- `attention` (default) — the transformer-block Linears: `attn.qkv`,
  `mlp.fc1/.fc2`, per-block adaLN `adain_*.linear`, and the outer
  `mlp_scale_bias`.
- `attention+conv` — the above plus the 3×3×3 downsample convs
  (`down*_*.body.*`).
- `all` — every adaptable leaf.

**Conv gotcha (verified, peft 0.19).** LoRA on a **1×1×1** Conv3d crashes at
merge time (`get_delta_weight` reuses the Conv2d squeeze path), and a **grouped**
Conv3d requires `rank % groups == 0` at inject time. Both classes are excluded
from the presets and from `all_adaptable_module_names`, so the auto presets
always inject *and* merge cleanly (P3D's `reduce_chan_level*` 1×1 convs are
dropped; UNetConvNeXt's depthwise convs are dropped). Target one explicitly via
`lora.target_modules` (divisible rank, never merged) only if you must.
`param_to_scalar` (P3D's native-conditioning head) is excluded too — train it
fully via `lora.modules_to_save: [param_to_scalar]` instead.

### 22. `BaseTraining.weights_transform`

The trainer writes best-val `weights.pt` on every improvement. A LoRA run's
in-loop `state_dict()` is the *wrapped* (base + adapter) dict, which a crash
mid-training would leave behind in a format `NeuralSurrogateForwardModel` can't
load. The optional `weights_transform` callable (default `None` → byte-identical
old behavior) is applied when persisting best weights; a LoRA fine-tune
passes `merge_to_state_dict`, so the on-disk `weights.pt` is **always** a plain
merged dict. When it is set, `BaseTraining` also keeps the best-val *wrapped*
state in RAM and restores that (not the on-disk merged form) into the model at
the end of `fit()`.

**Resume-safety.** The best-val wrapped snapshot is also persisted into
`checkpoint.pt` (`best_model_state`) and recovered on resume, so a resumed run
that never beats the pre-resume `best_val` still restores the *true* best at the
end of `fit()` — otherwise the caller's final `save(merge(...))` would clobber the
on-disk best with last-epoch weights. `fit()` exposes `restored_best_weights`;
`_save_lora` in [tasks.py](../scripts/utils/tasks.py) only overwrites `weights.pt`
when it is `True` (else it keeps the trainer's on-disk best). `merge_to_state_dict` returns
**CPU** tensors, so the per-improvement deepcopy+merge retains no GPU memory and
`weights.pt` is device-agnostic.

The disk-best score guard also applies to plain training, including latent flow
matching; it is not limited to `weights_transform` runs.

### 23. Config + script

`train.py --config-name surrogate/finetune_stepper`
([finetune_stepper.yaml](../configs/surrogate/finetune_stepper.yaml),
`task: finetune_stepper`, built by `_finetune_stepper` in
[tasks.py](../scripts/utils/tasks.py)):

1. Load `<pretrained_dir>/config.yaml`; take its `architecture` node and its
   dataset `state_vars` / `sdf_features` / `sdf_clamp_cells` (and
   `param_vars` / `num_history_steps` unless set) as the fine-tune dataset's.
2. Build the fine-tune `TransitionDataset`s (`dataset.root_dir`: the new
   data), instantiate the architecture and load `weights.pt`. With
   `recompute_normalization: true` install the new data's stats; otherwise
   keep the pretrained ones.
3. `method: full` trains every weight from there. `method: lora` freezes the
   model and `inject_lora`s the `lora:` block (`variant`, `rank`, `alpha`,
   `dropout`, `target_preset`, `target_modules`, `modules_to_save`); the
   optimizer gets only the trainable (adapter) parameters and the trainer
   `weights_transform=merge_to_state_dict`.
4. After `fit()` (LoRA): `save_adapter` → `<name>/adapter/`, then overwrite
   `weights.pt` with the merged plain dict (and `best_val.json`) **only when
   `trainer.restored_best_weights`**, else keep the trainer's on-disk best
   (§22).

```bash
pixi run -e dev python scripts/surrogate/train.py --config-name surrogate/finetune_stepper \
    pretrained_dir=model_weights/p3d_xie_and_castro name=p3d_ft_barcelona \
    dataset.root_dir=training_data/pylbm_barcelona lora.rank=32 lora.target_preset=attention
```

### 24. Artifact layout + ESMDA compatibility

```
model_weights/<name>/
  config.yaml           # pretrained architecture + fine-tune dataset + pretrained_dir
  weights.pt            # full MERGED plain state dict — ESMDA loads this, unchanged
  adapter/
    adapter_model.safetensors   # PEFT adapter (small, portable)
    adapter_config.json
  checkpoint.pt, metrics.csv, best_val.json   # training-loop artifacts
```

`weights.pt` is indistinguishable from a fully trained model, so
[§12](#12-neuralsurrogateforwardmodel) works with **zero changes**: the fine-tune
`config.yaml` presents the same top-level keys the loader reads (`architecture`,
`dataset.state_vars/param_vars/root_dir`), with `dataset.root_dir` pointing at the
fine-tune data (that *is* the domain the fine-tuned model targets).

### 25. File map

| Piece | File |
|---|---|
| `inject_lora` / `merge_to_state_dict` / `save_adapter` / `load_adapter` | [finetuning/inject.py](../libs/neural-surrogates/src/neural_surrogates/finetuning/inject.py) |
| `resolve_target_modules` / presets / `all_adaptable_module_names` | [finetuning/targets.py](../libs/neural-surrogates/src/neural_surrogates/finetuning/targets.py) |
| `weights_transform` hook + resume-safe best-val restore | [training/base.py](../libs/neural-surrogates/src/neural_surrogates/training/base.py) |
| Shared loader + normalization helpers | [training/data_utils.py](../libs/neural-surrogates/src/neural_surrogates/training/data_utils.py) |
| Config | [configs/surrogate/finetune_stepper.yaml](../configs/surrogate/finetune_stepper.yaml) |
| Run script | [scripts/surrogate/train.py](../scripts/surrogate/train.py) (`task: finetune_stepper`) |
| Tests | [test_lora_finetuning.py](../tests/neural_surrogates/test_lora_finetuning.py), [test_base_training_weights_transform.py](../tests/neural_surrogates/test_base_training_weights_transform.py) |

---

## Part G — Autoencoder (foundation-model) pre-training (Tadpole)

Pre-train a **Tadpole-style (V)AE** on flow snapshots as pure representation
learning — **no** next-step objective. This is plan 02 of the
[surrogate plans](plans/implemented/neural_surrogates/00_master_plan.md): the autoencoder wrapper, a
single-snapshot dataset and an AE trainer. Part H turns a pre-trained AE into a
next-step time-stepper; the AE itself is **never** a forward model.

### 26. `TadpoleAE` — the wrapper architecture

[architectures/tadpole_ae.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_ae.py)
wraps the **vendored** `TadpoleAutoencoder`
([architectures/_tadpole/](../libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole/),
from [tum-pbs/Tadpole](https://github.com/tum-pbs/Tadpole)) behind this repo's
conventions. We do **not** touch our `P3D` wrapper or `p3d_surrogate` — the AE
uses Tadpole's own P3D encoder/decoder, which ship the KL/VAE head.

Only the autoencoder subtree is vendored (like `_upt/`), **not** an external
`tadpole` dependency: upstream's `requirements.txt` pulls in `torchfsm` (its
online data-generation dep, unused here) → `vape4d`, an unnecessary,
resolution-risky chain the AE path never imports. The vendored files are
byte-for-byte upstream except **two** documented pyurbanair edits: (1) the `GIFt`
import (upstream's integer-rank LoRA library, used only by a branch we never
take; we drive LoRA through PEFT) is made optional, and (2) the optional
**geometry-branch projections** in `architecture/p3d/{conv,core,kl,skip_wrapper}.py`
and `model/{autoencoder,dft}.py` (each of those files' module docstrings states
its own injection points). The projections are created **only** when the caller
passes `geom_in_dims`, so on the default path nothing is added. Preserving the
exact subtree keeps every internal relative import valid **and** the
encoder/decoder `state_dict` keys identical, so the HF `thuerey-group/Tadpole`
weights still load strictly. Runtime deps
(`diffusers`/`timm`/`einops`) are the `neural_surrogates[tadpole]` extra, imported
lazily inside `TadpoleAE.__init__` so `import neural_surrogates` stays light.

What the wrapper adds:

| Concern | Behaviour |
|---|---|
| **Normalization** | `normalize=True` z-scores each state channel with buffered training stats (`set_normalization`, saved with the weights) — the same contract as `P3D`/`UPT`, so `train.py`'s `get_normalization_stats` path just works. Standardising *before* the autoencoder folds channels into the batch makes each folded **state** crop ~`N(0,1)`, matching Tadpole's pre-training statistics (this is what the HF warm start relies on). The geometry-block channels are fed **raw** (see below), so on those few auxiliary channels the HF encoder sees out-of-distribution inputs — acceptable: they carry a bounded, near-constant geometry cue (not primary flow statistics) that the encoder adapts to during continued pre-training. |
| **Geometry** | Input is masked (`state * geometry`, obstacles zeroed) like `P3D`. With `encode_geometry=True` the mask (`{0,1}`) and, if `sdf_features` is on, the SDF channels (`[-1,1]`) are appended **raw** (already bounded; a 0/1 mask has no mean/std to standardise) as **extra folded encoder channels**, and *reconstructed* — on purpose: recon loss on the state alone would let the encoder discard geometry from the latent, so making it reconstruct the geometry block is the supervision that forces geometry *into* the latent, which is what the plan-03 DFT attends over. On a single-geometry corpus this re-encodes a constant per snapshot (intended for the multi-geometry regime; use `encode_geometry=False` for state-only / single-geometry). |
| **SDF features** | `sdf_features` (`none`/`sdf`/`grad`/`both`) appends the clamped-SDF / gradient channels alongside the encoded mask; requires `encode_geometry=True` **or** a `geometry_branch` (in branch mode the same channels feed the branch instead of the encoder), and must match the dataset's mode + `sdf_clamp_cells` (`train_autoencoder.yaml` interpolates them from `dataset`). |
| **Geometry branch** | `geometry_branch: {width: 32}` (default `null`) switches geometry from *content* to *conditioning* — see below. |
| **Spatial processing** | `spatial_mode: local` preserves independent tiles; `global` processes each whole rectangular field; `halo` uses overlapping encoder/decoder tiles and keeps central cores. `encoder_crop_size` accepts a scalar or an anisotropic `[z, y, x]` shape; local/halo pad each axis to its tile size, while global pads only to stride 16. See below. |
| **Params** | Deliberately **not** an AE input — physical params condition dynamics, not single-snapshot appearance (they enter in plan 03). `n_params` is accepted for signature parity and ignored. |
| **Pretrained** | `pretrained`: `none` (random) / `hf` (`thuerey-group/Tadpole` weights for the size, via `huggingface_hub`) / `{encoder, decoder}` local paths. |

**Geometry branch (`geometry_branch`, off by default).** The alternative to
folding. `GeometryBranch`
([architectures/tadpole_geometry_branch.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_geometry_branch.py))
is a small trainable conv net (strided conv → GroupNorm → GELU → conv per level)
that maps the raw geometry block `[mask, (sdf, ∇sdf)]` — on the **padded** grid —
to four feature maps at strides `(1, 2, 4, 16)` with widths
`(w, 2w, 4w, 8w)` by default. Those strides are exactly the resolutions the
vendored P3D stack exposes, and each level is added through a **zero-initialised
`1×1×1` `Conv3d`** at a matching point:

| Level (stride) | Encoder | Decoder |
|---|---|---|
| 0 (1) | after `feature_embed` | just before `decompress` (after the last upsampling) |
| 1 (2) | after `downsampling_layers[0]` | after `upsampling_layers[0]` |
| 2 (4) | after the last `downsampling_layers` | at the conv up-path's input |
| 3 (16 = latent grid) | — | the decoder's **latent input**, before the transformer decoder (`geom_latent_proj`) |

The projections live **inside** the vendored encoder/decoder, so they travel in
`encoder.pt` / `decoder.pt`; the branch itself is a separate module saved as
`geometry_branch.pt` (§29). In this mode geometry is **neither folded nor
reconstructed**: the working space is the state channels only
(`n_geometry_channels == 0`), there is no geometry recon term (the trainer's
`geom` term is identically zero, no config change needed), and latent capacity
stays on the flow — "condition, don't predict". `encode_geometry` must therefore
be `False`: setting both **raises** (they are mutually exclusive). `sdf_features`
stays available and feeds the branch.

Because every projection is zero-init, a freshly built branch-mode AE
reconstructs *exactly* what it would with the branch disconnected, so training
starts from the unconditioned AE. That also makes the warm start usable: with
`geometry_branch` set, `encoder.pt`/`decoder.pt` (or the HF weights) are loaded
**non-strictly**, and the load then fails loud unless the *only* missing keys are
the zero-init `geom_proj` / `geom_latent_proj` ones and nothing is unexpected —
so a genuinely mismatched checkpoint is still rejected.

> **First-step gotcha (harmless).** With a *randomly* initialised AE
> (`pretrained: none`), upstream zero-inits the transformer decoder's
> `final_layer.out_proj`, so at the very first optimizer step everything upstream
> of it — the encoder-side projections and the decoder's latent-input projection
> — receives exactly zero gradient. It resolves as soon as that layer moves; with
> a pretrained warm start it never occurs.

`forward(state, geometry, geom_features=None, *, return_kl_element=False,
working_space=False)`: by default returns the **physical-units** state
reconstruction `(B, C, *grid)` (obstacles zeroed) — the clean contract for plan
03 / notebooks. With `working_space=True` it returns `(recon, target)` of the
**full-channel working-space** reconstruction and its input target, which is what
the trainer needs to split the state / geometry loss without recomputing the
assembled input. `encode(...)` / `decode(...)` passthroughs are exposed for plan
03 and analysis. Sizes `S`/`B`/`L` (8.8M/38.1M/152.1M params; latent compression
16/8/4).

#### Spatial processing (AE and DFT)

Both wrappers expose the same architecture settings, saved with the model:

| Setting | Meaning |
|---|---|
| `spatial_mode: local` | Default and legacy behavior: encode/decode independent tiles. `encoder_crop_size` is either a scalar for cubic tiles or `[z, y, x]` for anisotropic tiles; every entry must be a multiple of 16. |
| `spatial_mode: global` | Encode/decode each state channel over the whole rectangular domain. Pad each axis only to a multiple of 16; `encoder_crop_size` is unused. This uses more memory and retains the backbone's existing windowed attention. |
| `spatial_mode: halo` | Encode overlapping tiles with `halo_size` cells of context on each side, assemble the central latent cores, then decode overlapping latent neighborhoods and retain only central output cores. `encoder_crop_size` sets the per-axis core shape. |
| `halo_size` | Default 16; nonnegative multiple of 16, used only in halo mode. Zero gives tiles without overlap. Boundary halos are clipped at the padded domain edge. |

In **every DFT mode**, the latent time-stepping network runs **once over the
assembled full-domain latent grid**, joining all state variables. Halo mode
uses neighboring **evolved** latents during decoding, together with the
corresponding encoder skips and geometry features. It does not run independent
time-steppers on overlapping patches. Geometry folding and the separate branch
are supported in all three modes; geometry stays static.

Halo processing trades extra computation for spatial context. It is a
finite-context approximation, not an exact reproduction of global processing:
normalization and attention depend on the region being processed. In particular,
larger halos should be evaluated on seam errors and rollout quality rather than
assumed equivalent to global processing. `max_internal_batchsize` limits the
encoder/decoder batch, not the full-grid latent transform or all saved training
activations; in local mode `encode(...)` honours it too.

For example, set these in `train_autoencoder.yaml` or `train_dft.yaml`:

```yaml
architecture:
  spatial_mode: halo  # local | global | halo
  encoder_crop_size: [16, 32, 32]  # (z, y, x); scalar 32 remains valid
  halo_size: 16
```

For a `16 × 64 × 64` field, `[16, 32, 32]` produces `1 × 2 × 2` tiles per
channel with no padding; scalar `32` would pad the z axis from 16 to 32.

AE and DFT may use different spatial modes without changing weight shapes or
re-pretraining the AE. Their geometry configuration must still agree. A DFT
starts from the AE reconstruction **under its selected processing mode**;
changing the mode can change that reconstruction. Old configs without these
keys retain local processing and the same state-dict keys.

### 27. `SnapshotDataset` — single time slices

[datasets/snapshot.py](../libs/neural-surrogates/src/neural_surrogates/datasets/snapshot.py)
is the representation-learning sibling of `TransitionDataset` (§6): it reads the
same on-disk split but every `(trajectory, t)` pair is a sample — a single state
snapshot, **no** next-step target and **no** params. It reuses the file
discovery, lazy `xr.open_dataset`, geometry-mask loading and content-deduped SDF
computation, and exposes empty `param_names` / zero-width `_params` so the shared
`get_normalization_stats` path (`training/data_utils.py`) works unchanged (its
param branch degenerates to empty stats). Items: `state` `(C, *grid)`, `geometry`
`(*grid,)`, optional `geom_features` `(C, *grid)`. `time_stride` sub-samples
snapshots to decorrelate them; `random_crop_size` returns a random spatial crop
per item (the paper's intermediate pre-cropping — more crop diversity, smaller
batches; default `null` = full field). `snapshot_collate` ships a shared
geometry once as `(1, *grid)` (the full-field fast path) and falls back to
stacking per-sample geometry when random-cropping. Each loader worker keeps
**one** trajectory's NetCDF file open: it is reused within a trajectory batch
and closed when the next batch switches trajectory. netCDF4 keeps a chunk cache
per open variable (64 MiB each for u/v/w on the realistic corpus), so keeping
every visited file open, as shuffled multi-geometry batches would, exhausts
worker RAM mid-epoch. Hence `cache: true` only keeps the current trajectory in
memory here, not the whole split.

### 28. `AutoencoderTrainer` — the (V)AE loss

[training/autoencoder.py](../libs/neural-surrogates/src/neural_surrogates/training/autoencoder.py)
reuses all of `BaseTraining`'s machinery (device/AMP, warmup+cosine LR, grad
clip, early stopping, checkpoint/resume, `metrics.csv`, best-weights) but has
**no** pushforward curriculum (a snapshot AE has no time axis). It overrides
`_forward` to compute

```
loss = masked_mse(state_recon, state)                       # fluid cells only
     + geometry_recon_weight * mse(geometry_recon, geometry_block)
     + kl_weight * kl_elem.mean()
```

on a `SnapshotDataset` batch. The state reconstruction MSE is fluid-masked
(obstacle cells carry no signal); the geometry/SDF channels (present only when
`encode_geometry`) get their own small weight so the total stays dominated by
state reconstruction; `kl_weight` (β) defaults tiny (latent-diffusion
convention). `kl_weight=0` + `latent_type="mode"` degrades to a plain
deterministic AE — the "AE core" of the staged scope, one config knob away. The
per-term breakdown (`recon` / `geom` / `kl`) lands in `metrics.csv` via
`_aux_terms`.

**Deterministic validation.** Under `latent_type: "sample"` the training loss
samples the latent (VAE-proper), but the trainer's validation path forces the AE
to use the **mode** latent (it temporarily sets `latent_type="mode"` for the
duration of `_validate`, restoring after). Validation loss is therefore
noise-free, so best-weights selection and patience-based early stopping ride on a
stable signal rather than sampling jitter — the val curve is reproducible across
runs at a fixed checkpoint.

### 28b. `TadpoleDiscriminator` — the optional adversarial (GAN) loss

MSE is minimised by the conditional **mean**, so a pure MSE+KL autoencoder blurs
away exactly the small-scale structure urban flow lives on (shear layers, wakes,
the sharp gradients hugging building faces). The VQGAN / latent-diffusion fix is
a **patch critic**: a small 3-D convolutional discriminator that scores local
patches of the reconstruction and adds an adversarial term rewarding texture the
pixel loss cannot see.
[architectures/tadpole_discriminator.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_discriminator.py)
wraps the vendored `P3DDiscriminator` (a P3D encoder backbone + a `1x1x1` conv to
one channel, so it emits a logit **map** `(B, 1, d, h, w)`, downsampled by 16;
the paper's scalar "belief" is that map's mean). It is **off by default**
(`discriminator: null`) and the default path is byte-identical to the pure-VAE
loss above — no critic, no second optimizer, no extra `metrics.csv` columns.

**The critic is conditioned on geometry.** Its input is the AE's *working-space*
field with the geometry block concatenated as extra channels:
`n_input_channels = n_state_channels + (1 + n_sdf_feature_channels)`. Both the
real and the fake pass are given the **true** geometry block — never the
reconstructed one, which would let the AE hide flow errors behind a distorted
obstacle field. The trainer settles the channel contract once at construction and
fails loudly (naming both counts) on a mismatch; `train.py` (tasks.py) injects
`n_state_channels` / `encode_geometry` / `sdf_features` from the *built model*, so
the critic and the AE cannot disagree. Both real and reconstructed state
channels are zeroed inside solids before either critic pass; geometry/SDF
conditioning remains unmasked, and adversarial gradients act only on fluid
state cells.

> **Deliberate deviation from upstream — do not "fix" it back.** Tadpole
> (Appendix C.1) feeds its critic the same *folded, single-channel* crops the
> encoder sees (`(B, C, X, Y, Z) → (B*C, 1, 64³)`, `in_channels=1`), so its
> discriminator never sees state and geometry — or even two velocity components —
> together. We deliberately do **not** fold: obstacle-adjacent sharpness is the
> artefact an urban-flow AE gets wrong, and scoring each channel in isolation
> makes that structurally invisible. Everything else follows the paper.

Per training step (a standard 1:1 alternating update):

```
loss += coeff * hinge_g_loss(D(fake))            # generator side, in _loss
coeff = ramp * adv_weight * lambda
                                                 # then, after the AE's step:
d_loss = hinge_d_loss(D(real), D(fake))          # _after_optimizer_step, detached pair
```

| Knob | Default | What it does |
|---|---|---|
| `adv_weight` | `1.0e-4` | **Maximum** adversarial scale (paper: "a maximum scale value of 1e-4"). |
| `adv_start_step` | `1000` | Term is exactly zero and the critic is not updated before this **global optimizer step**. |
| `adv_ramp_steps` | `1000` | Linear ramp 0→1 over this many further steps (`0` = step on at full strength). |
| `adaptive_adv_weight` | `true` | Esser et al. (2021) gradient balancing: `lambda = ‖∂recon/∂w_out‖ / ‖∂adv/∂w_out‖`, clamped to `[0, 1]`, so the effective coefficient lives in `[0, adv_weight]` and needs no per-dataset tuning. Costs two `torch.autograd.grad` probes per step. |
| `disc_recon_threshold` | `null` | Optional upstream gate: update the critic only once the masked L2 recon is below this (upstream uses `1e-3`, tied to *their* normalisation — hence off here). Costs a host sync per step. |

Warm-up is counted in **optimizer steps, not epochs** — the paper specifies
iterations, and our snapshot corpora differ in size by more than an order of
magnitude, so an epoch is not a portable unit. The counter is persisted in
`checkpoint.pt`, so a resumed run continues the schedule rather than restarting
the warm-up; the critic, its optimizer and its `GradScaler` are persisted too, and
a checkpoint predating the extension still resumes (the keys are simply absent).
The adversarial term is **never** part of the validation loss: it would make the
val curve jump at the warm-up boundary and stop being a comparable yardstick for
best-weight selection. `adv` / `adv_w` / `d` join `metrics.csv` when the critic is
configured.

`BaseTraining` gained exactly one thing for this: a no-op
`_after_optimizer_step(batch)` hook called once per optimizer step, where the
critic takes its turn. Every other trainer is unaffected.

**AMP caveat.** The adaptive weight probes gradients on the *unscaled* losses,
which is exact under the default `amp_dtype: bfloat16` (where `GradScaler` is
constructed disabled). Under **fp16** loss scaling those probes could underflow
and collapse `lambda` to its fallback — set `adaptive_adv_weight: false` if you
must run the adversarial path under fp16.

### 29. Config + script + artifacts

`train.py --config-name surrogate/train_autoencoder`
([train_autoencoder.yaml](../configs/surrogate/train_autoencoder.yaml),
`task: autoencoder`): datasets → `TadpoleAE` with `n_state_channels` from the
data → `set_normalization` → optional discriminator (§28b) → save
`config.yaml` → `AutoencoderTrainer.fit()`. The architecture's `sdf_features`
/ `sdf_clamp_cells` interpolate the dataset's, so the two always agree; the
KL, geometry and adversarial weights are in `loss_weights`.

```bash
pixi run -e dev python scripts/surrogate/train.py --config-name surrogate/train_autoencoder \
    paths.data_dir=training_data/pylbm_barcelona name=tadpole_ae_s
```

Artifacts in `model_weights/<name>/`: `weights.pt` (full `TadpoleAE` state
dict), plus `encoder.pt` / `decoder.pt` via `save_separate_weights` (the
handoff to the DFT stepper and the latent generator), and `config.yaml` /
`checkpoint.pt` / `metrics.csv` as usual. In geometry-branch mode one more
file sits next to them — `geometry_branch.pt`, a plain `state_dict` of the
branch (the encoder/decoder projections it feeds already travel inside
`encoder.pt`/`decoder.pt`); `TadpoleTimeStepper` loads it (§31). These
exports are cut from `weights.pt` at the start of each run that already has
one, every time the trainer saves new best weights, and once more after training, so they always match
`weights.pt`, also when a job is killed at its time limit. Resubmitting the
same command resumes from `checkpoint.pt`; `num_epochs` is the total.

**Batching.** `training.yaml` ships the `TrajectoryBatchSampler` (§6), which
replaces `dataloader.batch_size` / `shuffle` / `drop_last` (set the batch size
on `batch_sampler`), draws every batch from one trajectory (needed for
multi-geometry corpora) and caps it at `max(1, cell_budget // cells)` cells:
on large grids that silently drops the batch to 1, so size `cell_budget`
from a known-good run (`batch_size * cells_per_sample`).
`train_autoencoder.yaml` sets `batch_size: 8`, `cell_budget: 393216` and
`random_crop_size: 64`. `cell_budget` counts each trajectory's **full** grid,
not the random crop the dataset returns, so it does not bound crop memory: for
crop training on large grids set `cell_budget: null` and tune `batch_size`
directly. Choose each `encoder_crop_size` entry as a multiple of
16 that divides the crop (or grid) along that axis: non-divisible axes are
zero-padded every forward, which wastes compute and inflates the logged `kl`.

**Pre-chunked data (optional).** Snapshot reads of a realistic corpus are
dominated by the NetCDF chunk layout the data was written with. With
`prechunk.output_root` set, `train.py` first writes (or validates) a lossless
re-chunked copy of every `state/<split>/sample_*.nc` there
([datasets/rechunk.py](../libs/neural-surrogates/src/neural_surrogates/datasets/rechunk.py):
time chunk 1, `prechunk.spatial_chunks` (`[16, 64, 64]` here) spatial chunks,
zlib level 1), then trains on it. It writes whole output chunks at a time:
filling a chunk piecemeal recompresses it on every write.
Only the physical storage changes: every split, frame, value, coordinate and
attribute is kept, packed values are copied as stored. A manifest of source and
output file fingerprints makes an interrupted preparation resume, and a changed
source or option fails rather than mixing data (pick a new directory to
rebuild); a lock stops two writers. A complete copy is validated without the
lock and without writing, so runs can share it and it may be read-only. The
lock is node-local on BeeGFS (`tuneUseGlobalFileLocks = false`): never let two
preparations of one directory run at once (submit training with
`--dependency` on the prep job). `config.yaml` keeps the source
`dataset.root_dir`, while the normalization stats are cached under the copy.
The `prechunk` block lives in `training.yaml` (whole frames,
`spatial_chunks: null`) and this config overrides it with tiles; `stepper`,
`finetune_stepper` (so the baselines too) and `dft` read whole frames, so one
whole-frame copy of a corpus serves all of them. They read their params from
`dataset.root_dir` (`TransitionDataset(param_root=...)`), and the steppers
stream their state normalization stats from the source, whose big chunks read
a whole split faster than the copy. The stats cache keys on the dataset family
(`TransitionDataset` / `SnapshotDataset`), so subclasses such as
`RolloutTransitionDataset` share their parent's cache.
`prechunk.prepare_only=true` makes the copy and exits without building a model,
so it can run as a CPU job before the GPU one
(`job_scripts/<machine>/surrogate_prechunk_data.slurm`, see
[job_scripts.md](job_scripts.md)).

### 30. File map

| Piece | File |
|---|---|
| `TadpoleAE` wrapper | [architectures/tadpole_ae.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_ae.py) |
| `TadpoleDiscriminator` + GAN loss helpers | [architectures/tadpole_discriminator.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_discriminator.py) |
| `GeometryBranch` | [architectures/tadpole_geometry_branch.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_geometry_branch.py) |
| Vendored autoencoder subtree | [architectures/_tadpole/](../libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole/) |
| `SnapshotDataset` / `snapshot_collate` | [datasets/snapshot.py](../libs/neural-surrogates/src/neural_surrogates/datasets/snapshot.py) |
| Lossless re-chunked copy (`prechunk`) | [datasets/rechunk.py](../libs/neural-surrogates/src/neural_surrogates/datasets/rechunk.py) |
| `AutoencoderTrainer` | [training/autoencoder.py](../libs/neural-surrogates/src/neural_surrogates/training/autoencoder.py) |
| Config | [configs/surrogate/train_autoencoder.yaml](../configs/surrogate/train_autoencoder.yaml) |
| Run script | [scripts/surrogate/train.py](../scripts/surrogate/train.py) (`task: autoencoder`) |
| Tests | [test_autoencoder_pretraining.py](../tests/neural_surrogates/test_autoencoder_pretraining.py), [test_tadpole_discriminator.py](../tests/neural_surrogates/test_tadpole_discriminator.py), [test_autoencoder_adversarial.py](../tests/neural_surrogates/test_autoencoder_adversarial.py), [test_tadpole_geometry_branch.py](../tests/neural_surrogates/test_tadpole_geometry_branch.py), [test_snapshot_rechunking.py](../tests/neural_surrogates/test_snapshot_rechunking.py) |

---

## Part H — Autoencoder → time-stepper (Tadpole DFT)

Turn a **pre-trained** autoencoder (Part G) into a next-step forward model
with Tadpole's **DFT** ("Dynamic Fine-Tuning") recipe: a *frozen* encoder/decoder
with **zero-initialised** reintroduced skip connections (the paper's γ scales) and
a **zero-initialised** latent sub-network that together act as a trainable
increment *around* the frozen AE reconstruction. This is plan 03 of the
[surrogate plans](plans/implemented/neural_surrogates/00_master_plan.md); it composes
plans 01 (LoRA/PEFT) and 02 (`TadpoleAE`, `encoder.pt`/`decoder.pt` handoff) — the
DFT stage is "just" a plan-01 fine-tune with a different architecture plus a few
extra fully-trained modules.

### 31. `TadpoleTimeStepper` — the wrapper architecture

[architectures/tadpole_stepper.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_stepper.py)
adapts the **vendored** `TadpoleDFT`
([architectures/_tadpole/model/dft.py](../libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole/model/dft.py))
to this repo's forward contract `forward(state, params, geometry,
geom_features=None) -> state_next` (what `Trainer._forward` and
`NeuralSurrogateForwardModel._rollout_chunk` call). Field IO (masking, per-channel
z-scoring, geometry+SDF channel assembly, crop-multiple padding, `set_normalization`)
is shared with `TadpoleAE` through the
[`_TadpoleFieldIO`](../libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole_field_io.py)
mixin, so the two wrappers behave identically on the plumbing.

What the wrapper adds around `TadpoleDFT`:

| Concern | Behaviour |
|---|---|
| **Frozen enc/dec + LoRA** | Built with `encoder_ft_state="frozen"` / `decoder_ft_state="frozen"`; the pre-trained `encoder.pt`/`decoder.pt` are loaded via the DFT's own `weight_encoder`/`weight_decoder` **load-before-skip-wrap** path (so keys line up with the raw `_KLP3DEncoder`/`_P3DDecoder`). LoRA is injected **externally** through plan 01's PEFT stack (not Tadpole's bundled GIFt path), so `lora.variant` and the merged-`weights.pt` export work here too. |
| **Latent sub-network** | `subnetwork="default"` builds a `ParamConditionedSubnetwork` wrapping a vendored `SequentialModel` (`attention_method="naive"` — the dev box has **no** triton, so the upstream `"hyper"` attention is unavailable) over the folded latent tokens, with `init_zero_proj=True` so its output is exactly `0` at init. `None` builds no sub-network (pure frozen AE). |
| **Param conditioning** (our addition; Tadpole has none) | `param_conditioning="film"` (default): the (z-scored) params `(B, P)` drive a small MLP with a **zero-initialised** output layer producing per-channel `(scale, shift)` applied to the latent tokens as `x*(1+scale)+shift`. `"token"`: an additive (adaLN-style) zero-init param embedding. `"none"` or `n_params==0`: **no** conditioning module at all (repo no-op rule) — the module tree is byte-identical to a param-free build. |
| **Normalization** | State stats are **inherited** from the AE (read from `<ae_dir>/weights.pt`) so the frozen encoder sees its pre-training distribution; `set_normalization` installs the fine-tune split's **param** stats (used to z-score params before conditioning). Buffers travel with the checkpoint. **Fail-loud:** if the AE dir's `weights.pt` is missing or lacks `state_mean`/`state_std`, loading the stats now **raises** (repo convention) rather than silently continuing with identity zeros/ones — the only exception is an explicit `recompute_normalization` opt-out, which will overwrite the stats anyway. |
| **Geometry** | Masked like `P3D` (`state * geometry`); with `encode_geometry=True` the mask (+SDF) channels ride through the frozen encoder exactly as in pre-training, so the latent tokens the sub-network attends over carry geometry. Output geometry channels are discarded (geometry is static). Must match the AE. |
| **Geometry branch** | `geometry_branch: {width: 32}` (default `null`): the AE's pre-trained branch is reused, **frozen**, and its features condition the frozen enc/dec and the latent sub-network — see below. Mutually exclusive with `encode_geometry`; must match the AE. |

**Optional state mixing on DFT skips.** Set
`architecture.skip_mixing: {width: 32, levels: [4, 8]}` to add one pointwise
bottleneck adapter at each selected spatial stride (`1`, `2`, `4`, `8`). The
adapter concatenates aligned state-variable features, applies per-voxel
LayerNorm, a `1×1×1` projection to `width`, GELU, and a zero-initialized `1×1×1`
projection back to the original feature count. Its correction is split by state
and added as `decoder + gamma * skip + mixer(all_state_skips)`. It bypasses the
zero-initialized gamma so its output projection receives gradients immediately;
only the final projection starts at zero. Geometry channels are excluded, though
geometry-branch conditioning already present in state features is retained.

The default `null` creates no adapters and preserves the original execution and
checkpoint keys. Enabling mixing requires no AE retraining; adapters train fully
through the `skip_mixing` entry in `trainable_modules`, and are saved in merged
`weights.pt` with their architecture config. Start with strides `[4, 8]`; adding
`[1, 2]` increases activation memory and compute. All three spatial modes are
supported. States are gathered per region before mixing, independent of the
encoder/decoder batch chunk limit. With mixing enabled, local mode also uses
`encode`/`decode`'s full-grid latent plus `SpatialResiduals` contract (otherwise
local mode retains its original crop-folded contract). No extra spatial halo is
needed for these pointwise adapters.

**Subnetwork controls.** `architecture.subnetwork_cfg` exposes overrides for
the latent transformer (for example `use_checkpoint: true`, `n_layers`,
`hidden_size`, `num_heads`, and `film_hidden`). Its defaults retain full-domain
attention. `use_checkpoint` recomputes transformer activations during backward
to reduce memory. Positive `in_context_patches` selects overlapping windows of
flattened spatial tokens and averages their predictions back onto every token;
this changes the attention context and should be evaluated separately from the
default global model.

**Geometry branch.** With `geometry_branch={...}` the stepper reuses the AE's
branch instead of folding geometry: it builds `GeometryBranch(in_channels=1 +
n_sdf, **geometry_branch)`, loads `<pretrained_ae_dir>/geometry_branch.pt` (a
plain `state_dict`; **fail loud** if absent, since a random branch would feed the
frozen AE conditioning it has never seen) and **freezes** it — the branch is part
of the frozen AE, exactly like `encoder.pt`/`decoder.pt`, and `train_dft.yaml`'s
`trainable_modules` never lists it. `skip_pretrained_load` (the deploy
build) is the only sanctioned way past the load, because the merged
`weights.pt` already carries the branch as a submodule. The four features are
injected in two places:

* **enc/dec projections** — the same zero-init `1×1×1` convs as in the AE (§26),
  which arrive already trained inside `encoder.pt`/`decoder.pt`; the folded
  feature pyramid is chunked alongside `x` when `max_internal_batchsize` bites;
* **spatial FiLM on the latent sub-network** — `ParamConditionedSubnetwork`
  gains a zero-init `Conv3d(out_dims[3], 2 * in_dim, 1)` (`geom_cond_dim > 0`;
  `0` builds nothing) that maps the *unfolded* level-3 feature, on the latent
  grid, to a **per-token** `(scale, shift)` applied as `x * (1 + scale) + shift`
  after the existing param FiLM. The stepper checks that this feature's grid is
  the padded grid / 16 before handing it over.

The enc/dec projections are frozen along with the encoder/decoder they live in
(and the `tadpole_encdec` LoRA preset skips them — they are `1×1×1` convs), so
during DFT fine-tuning the **only trainable geometry path is the FiLM**.

Only the **state** crops are folded in branch mode (`n_geometry_channels == 0`),
so the sub-network already reads and writes `C * Cl` channels and only state
crops are decoded — "condition, don't predict". `encode_geometry` must be
`False`; setting both raises.

**Residual convention.** Output is `state_next = dft_state * mask` — the DFT
*directly* predicts the next state (it morphs its own reconstruction toward
`u_{t+Δt}`); there is **no** separate `state + increment` term. `predict_residual`
is kept for API parity but is a no-op framing here (`state + (dft_state - state) ==
dft_state`). The trainable increment is *intrinsic*: the zero-init sub-network / γ
skips are the learned morph around the frozen AE reconstruction.

**Identity-at-init invariant.** At construction the sub-network output is `0`
(`init_zero_proj`), the γ skip `scales` are `0` and `latent_residual_scale == 1.0`,
so the DFT output is **identical to the plain-AE reconstruction** of the same
working-space input. `stepper._ae_reference_recon(state, geometry)` exposes that
pure-AE reconstruction cheaply (same encode/decode with the sub-network bypassed
and skip residuals zeroed), and the unit test asserts `stepper(state) ==
stepper._ae_reference_recon(state, geometry)` **exactly** (0.0 in practice) — the
single most informative test of the DFT wiring (it proves every zero-init and the
skip gating are correct). This holds in branch mode too: the enc/dec projections
belong to the frozen AE, so `_ae_reference_recon` applies the *same* branch
features, and the only new DFT-side addition — the spatial FiLM — is zero-init. This is *not* `state_next == state`: that holds only for
a perfectly-reconstructing AE, a **training** outcome, not a wiring invariant.

### 32. Training path — `task: dft`

`train.py --config-name surrogate/train_dft`
([train_dft.yaml](../configs/surrogate/train_dft.yaml), `_dft` in
[tasks.py](../scripts/utils/tasks.py)). `pretrained_dir` is the **AE** export,
and the architecture is declared inline (`TadpoleTimeStepper` with
`pretrained_ae_dir: ${pretrained_dir}`, `size`, `spatial_mode`,
`param_conditioning`, `latent_type`, `subnetwork`, `skip_mixing`,
geometry settings). The task:

1. takes the AE's dataset `state_vars` / `sdf_features` / `sdf_clamp_cells`
   (and `param_vars`, `num_history_steps` = 1 unless set) for the fine-tune
   `TransitionDataset`s;
2. builds the stepper with `require_ae_state_stats = not
   recompute_normalization`: by default the AE's state stats are kept and only
   the param stats of the fine-tune split are installed (computed from the
   in-memory param tables; the state stats are not streamed);
3. freezes everything, injects LoRA on `dft.encoder` / `dft.decoder` with the
   `lora:` block (`target_preset: tadpole_encdec`), and unfreezes the
   `trainable_modules` (the NEW modules trained *fully*: `subnetwork`,
   `latent_residual_scale`, `skip_mixing`, the γ skip `scales`);
4. stamps `skip_pretrained_load: true` / `pretrained_ae_dir: null` on the
   saved architecture and trains with `weights_transform=merge_to_state_dict`
   and the plan-01 export (§23 step 4).

The stepper's geometry settings (`encode_geometry`, `sdf_features`,
`geometry_branch`, `size`, `normalize`) must match the AE's: nothing
cross-checks them beyond the strict encoder/decoder weight load.

**Pre-chunked data (optional).** The same `prechunk` block as the autoencoder
(§29), but with `spatial_chunks: null`: one chunk per frame and variable,
since every sample reads two whole domains (`t` and `t+K`). Give it its own
`output_root`; the autoencoder's crop-sized tiles make each whole-frame read
many small ones. The copy holds only `state/`, so the task reads the params
from `dataset.root_dir` (`TransitionDataset(param_root=...)`). On the realistic
uDALES corpus (BeeGFS, 40-frame source chunks) a frame read measured ~1.4 s
(decompression-bound) from the source, ~2.5 s (latency-bound) from the
autoencoder's copy and ~0.55 s with ~0.1 s CPU from whole-frame chunks, at
about the source's size.

```bash
pixi run -e dev python scripts/surrogate/train.py --config-name surrogate/train_dft \
    pretrained_dir=model_weights/tadpole_ae_s name=dft_s
```

**`tadpole_encdec` LoRA preset** ([targets.py](../libs/neural-surrogates/src/neural_surrogates/finetuning/targets.py)):
a regex selecting the Linear + 3×3×3 Conv3d leaves inside `dft.encoder`/
`dft.decoder` (`qkv`, `mlp.fc1/.fc2`, the outer adaLN, and the encoder/decoder
convs). The 1×1×1 convs (`to_latent`, `linear_conv`) and the `cpb_mlp`
positional-bias net are excluded for the same merge-safety / dead-capacity reasons
as the P3D presets.

### 33. ESMDA deploy + artifact layout

The export is the standard plan-01 shape, so
[§12](#12-neuralsurrogateforwardmodel) works with **zero** loader changes:

```
model_weights/<name>/
  config.yaml   # inline TadpoleTimeStepper arch (skip_pretrained_load: true,
                #   pretrained_ae_dir: null) + fine-tune dataset + pretrained_dir
  weights.pt    # full MERGED plain state dict (enc/dec + subnetwork + γ + LoRA)
  adapter/      # PEFT adapter (adapter_model.safetensors + adapter_config.json)
  checkpoint.pt, metrics.csv, best_val.json
```

`weights.pt` is the **sole source of truth** for this mode. Unlike a pure-LoRA
next-step fine-tune, the fully-trained NEW modules (`subnetwork`, the γ skip
`scales`, `latent_residual_scale`, and optional `skip_mixing`) live **only** in the merged `weights.pt`, not in
`adapter/` — which holds just the encoder/decoder LoRA deltas. So `adapter/` alone
cannot reconstruct the trained model here; it is **provenance-only** (a portable
record of the LoRA half). When a resumed run does not restore its best weights
(§22), `weights.pt` is the trainer's on-disk best while `adapter/` is the last
epoch's — the two can disagree, and `weights.pt` (what deployment loads) is the
one to trust.

The critical detail: the saved `architecture` node is stamped with
**`skip_pretrained_load: true`** and **`pretrained_ae_dir: null`**, so at deploy
time the stepper is rebuilt with random enc/dec (no AE dir needed) and the merged
`weights.pt` — which already contains every weight — overwrites them. Deployment
therefore never depends on the original AE dir still existing. Fixed-grid like
P3D (no `domain_flexible`); the ensemble `clone_for_member` shallow-share is
unaffected. Deterministic rollouts use `latent_type="mode"` (the default).

### 34. File map

| Piece | File |
|---|---|
| `TadpoleTimeStepper` / `ParamConditionedSubnetwork` | [architectures/tadpole_stepper.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_stepper.py) |
| Vendored `TadpoleDFT` + downstream sub-network | [architectures/_tadpole/model/dft.py](../libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole/model/dft.py), [.../architecture/downstream/](../libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole/architecture/downstream/) |
| Shared field IO mixin | [architectures/_tadpole_field_io.py](../libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole_field_io.py) |
| `tadpole_encdec` LoRA preset | [finetuning/targets.py](../libs/neural-surrogates/src/neural_surrogates/finetuning/targets.py) |
| Config | [configs/surrogate/train_dft.yaml](../configs/surrogate/train_dft.yaml) |
| Run script | [scripts/surrogate/train.py](../scripts/surrogate/train.py) (`task: dft`) |
| `GeometryBranch` (shared with the AE) | [architectures/tadpole_geometry_branch.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_geometry_branch.py) |
| Tests | [test_ae_to_timestepper.py](../tests/neural_surrogates/test_ae_to_timestepper.py), [test_tadpole_stepper_geometry_branch.py](../tests/neural_surrogates/test_tadpole_stepper_geometry_branch.py) |

---

## Part I — Generative spin-up (latent flow matching, plan 07)

Replace the CFD spin-up with a **sample**: a conditional flow-matching model
trained in the latent space of a *frozen* pre-trained `TadpoleAE` (Part G)
generates a statistically developed flow state from the obstacle geometry and a
short history of the inflow parameters, and the surrogate rollout (Part D / H)
starts from it. This is plan 07 of the
[surrogate plans](plans/implemented/neural_surrogates/00_master_plan.md)
([07_latent_flow_matching_spinup.md](plans/implemented/neural_surrogates/07_latent_flow_matching_spinup.md));
it depends on plan 02 (the AE) and, for deployment, on the `TadpoleTimeStepper`
of plan 03. None of the DFT machinery (skips, gates, LoRA, skip mixers) is
involved — a generated latent has no input state whose encoder skips could be
reused, so decoding goes through the plain AE decoder.

The key deployment property (§40): with `spinup_source: generative` the initial
state is **regenerated on every cold forward call from each member's current
parameters**, so during the first ESMDA window every iteration and the final
posterior forecast re-condition the initial state on the *updated* parameters —
the LES cold-start lifecycle, without the LES. Flow time `tau` is an artificial
integration coordinate in `[0, 1]`; it has nothing to do with physical time or
the parameter-history cadence.

### 35. `SnapshotHistoryDataset` — snapshots plus their parameter history

[datasets/snapshot_history.py](../libs/neural-surrogates/src/neural_surrogates/datasets/snapshot_history.py)
subclasses `SnapshotDataset` (§27) — same file discovery, lazy I/O, geometry /
SDF handling, `sample_index`, `grid_shape` and geometry dedup — and adds the
physical conditioning a conditional generator needs. Every item carries, next
to `state` `(C, *grid)`, `geometry` `(*grid,)` and the optional `geom_features`,
a `params_hist` tensor `(Hp, P)`: the parameter rows at the saved times
`t-Hp+1 … t`, **oldest first**, ending at the snapshot's own time, columns in
the given `param_vars` order. Scalar (static) parameters are broadcast across
the history.

Two constructor arguments are **required** (`None` raises): `param_vars` — the
*ordered* conditioning schema, which the generator artifact records so
deployment can never substitute a different convention — and
`param_history_steps` (`Hp >= 1`). `random_crop_size` must stay `None`
(`NotImplementedError` otherwise): plan 07 v1 trains on full snapshots, since a
global latent generator trained on relocated crops needs its own assessment
(§41). Because the conditioning is physical, the dataset is strict where the
transition datasets are lenient:

| Contract | Behaviour |
|---|---|
| **Pairing** | State and parameter files are paired by sample id (the `sample_XXXX` stem), never by sorted position. A state sample without a param partner, or a param sample without a state partner, raises. |
| **Time coordinates** | Both files must carry a `time` coordinate (no alignment by index); the two must agree per sample (`allclose`), be finite and strictly increasing. Parameter values must be finite and the resolved variable set identical across samples. |
| **Cadence** | The median saved `dt` over every trajectory is stored as `history_dt_seconds`; every `dt` must lie within `cadence_rtol` (class default `0.05`, `0.1` in the config) of it, else the error names the sample and the step. Real corpora are slightly non-uniform (`pyudales_idealized` saves at 0, 4.85, 9.92, 15.00, … s), so an exact check would reject valid data while a loose one would let a mixed-cadence corpus train a generator whose `Hp` rows span an ill-defined duration. `Hp` samples span `(Hp-1) * history_dt_seconds` (12 × 5 s → 55 s). |
| **Anchors** | Anchors start at `t = Hp-1` so every history is fully recorded; a trajectory shorter than `Hp` raises. With `constant_prehistory=True` anchors start at `t = 0` and the missing leading rows repeat the first recorded row — valid **only** when the forcing was constant at those values before the first saved time (e.g. a constant-forcing spin-up at least `(Hp-1) * history_dt_seconds` long ending at the first save). Nothing verifies this against the corpus; the flag lives in the training config so the choice is explicit. `time_stride` thins the *anchors* only; histories always use contiguous saved steps. |
| **Shared reader** | The `(T, P)` per-trajectory parameter table is read by `load_param_table(param_path, t_len, param_vars, dtype) -> (Tensor, names)` in [datasets/_params.py](../libs/neural-surrogates/src/neural_surrogates/datasets/_params.py), hoisted from `TransitionDataset._load_params` (which is now a thin wrapper — same broadcasting of scalars, same length check, same error messages; behaviour unchanged and tested). |

The per-trajectory `_params` tables are kept exactly as `TransitionDataset`
keeps them, so `get_normalization_stats` (`training/data_utils.py`) yields
parameter mean/std over all saved times unchanged, and the inherited
`sample_index` / geometry dedup let `TrajectoryBatchSampler` (§6) bucket a
multi-geometry split as before. `snapshot_history_collate` delegates to
`snapshot_collate`: a shared geometry (+ SDF) ships once as `(1, *grid)`, and
`params_hist` goes through the default collate to `(B, Hp, P)`. On the trainer
side, `BaseTraining._prepare_snapshot_batch(batch) -> (state, geometry,
features)` — device upload plus the cached shared-geometry broadcast — was
hoisted out of `AutoencoderTrainer._prepare_ae_batch` (kept as a wrapper) so the
AE and the generator trainers share one implementation.

### 36. `TadpoleLatentGenerator` — the generator architecture

[architectures/tadpole_latent_flow.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_latent_flow.py)
is a plain `nn.Module` (not a `_TadpoleFieldIO` mixin user — it delegates field
I/O to the AE it owns) with three parts: a **frozen** nested `TadpoleAE`
(`self.ae`), a flow-matching **velocity network** (a `ParamConditionedSubnetwork`,
§31) and the **normalisation buffers** (`latent_mean` / `latent_std` /
`latent_stats_installed`, `param_mean` / `param_std`).

**Frozen AE.** Exactly one of `pretrained_ae_dir` (training: the AE export's
`config.yaml` `architecture` node supplies the kwargs, its `weights.pt` is loaded
**strictly**) or `ae_kwargs` (deployment: the resolved kwargs saved with the
generator) must be given. Hydra keys are dropped, unknown keys fail loud, the
export's `dataset.state_vars` count must equal `n_state_channels`, and three
kwargs are pinned so the stored kwargs rebuild the very same AE anywhere:
`n_state_channels`, `pretrained: "none"` (no HF download at deploy) and
`latent_type: "mode"` (deterministic latents, plan 07 v1; the export's own
latent type is kept as `ae_export_latent_type` for the record — pinned on both
the wrapper and the vendored `ae.ae`, which read it independently). Every AE
parameter gets `requires_grad=False`, and the generator's `train(mode)` override
keeps the AE in `eval()` after `BaseTraining` calls `model.train()`. The AE
**must** have a geometry path — `encode_geometry=True` (folded geometry latents)
or a `geometry_branch` — else construction raises: there would be nothing to
condition on. Its `spatial_mode` / `encoder_crop_size` / `halo_size` are
inherited (recorded in `ae_kwargs`) and cannot change at sampling time. The
sha256 of the export's `weights.pt` is stored as `ae_fingerprint`.

**One canonical full-grid latent for all spatial modes.** The velocity network
always sees a single `(B, D, Zl, Yl, Xl)` token grid with `D = C * Cl` (`Cl` =
the AE's latent feature count per folded channel — 256 / 512 / 1024 for S / B /
L — read off `ae.ae.decoder.latent_size`), whatever the AE's spatial mode:

1. the AE's own helpers assemble the masked, z-scored working input and pad it
   with the inherited policy (`global` → stride 16, `local` / `halo` → the crop
   size);
2. `encode_spatial` (the shared `_tadpole_spatial` helper) runs the plain AE
   encoder for **every** mode and gathers one latent per central latent cell —
   local regions simply have a zero halo, so local mode produces exactly the
   latents the AE's own crop fold would, assembled on the full grid instead of
   left folded (tested exactly against the AE path);
3. the `(B*C, Cl, …)` result is reshaped to `(B, C*Cl, …)` and z-scored per
   latent channel with the buffered statistics.

Geometry enters as **conditioning only**, through the sub-network's spatial
FiLM, and is never generated. `_geometry_conditioning` is the single code path
both `encode_latents` and `geometry_condition` go through, so the state-free
conditioning at sampling time is bit-identical to the one seen in training:

| AE geometry path | `geom_cond` (spatial FiLM input, `G` channels) | What the decoder gets |
|---|---|---|
| **Fold** (`encode_geometry: true`) | the geometry block's deterministic latents, encoded separately from the state channels and z-scored with their own buffered stats (`G = D_geom = n_geometry_channels * Cl`) | the **raw** geometry latents re-appended after the denormalised state latents, in working-channel order (state, then geometry) |
| **Branch** (`geometry_branch`) | the branch's stride-16 feature (`feats[3]`, `G = branch.out_dims[3]`) | the full feature pyramid, expanded to the `(B*C, F, …)` full-grid layout `decode_spatial` slices regions from (`_full_grid_feats` — **not** the local `_fold_geom_feats` crop layout) |

`encode_latents(state, geometry, geom_features=None)` returns a
`LatentEncoding` dataclass (`z`, `geom_cond`, `decoder_geom_feats`,
`orig_shape`, `geom_latents`, `mask`); `geometry_condition(geometry,
geom_features=None, batch_size=None, dtype=None, device=None)` returns the same
with `z=None` (a single mask is expanded to `batch_size`);
`decode_latents(z, cond)` checks `z` sits on `latent_grid_for(orig_shape)`
(padded grid / 16 — it would broadcast silently otherwise), denormalises the
**state** latents only, reshapes to `(B*C_work, Cl, …)`, calls `decode_spatial`
with the plain AE decoder and the matching geometry features, crops the padding,
keeps the first `C` channels, denormalises through the AE's state stats and
applies the fluid mask. There are no DFT residuals or mixers anywhere.

**Full-width velocity network — why the DFT width would be a subspace
restriction.** The DFT's fixed `_SUBNET_SIZES` hidden widths are deliberately
**not** inherited. For three variables and AE size S, `D = 768`, while the DFT
default hidden width is 144: its final `Linear(144, 768)` confines every
velocity to a fixed 144-dimensional subspace — for any vector `a` orthogonal to
that projection's columns, `aᵀ v = aᵀ bias` — so the ODE could only *translate*
those components of its Gaussian initialisation, never remove their noise or
learn their conditional distribution (equal input/output widths do not resolve
this). Here `hidden_size: null` resolves to `D`, rounded **up** to a multiple of
`num_heads`, and an explicit value below `D` is rejected. The build is
`ParamConditionedSubnetwork(in_dim=D, n_params=Hp*P + time_embed_dim,
n_layers=4, num_heads=8, hidden_size=h, param_conditioning="film",
geom_cond_dim=G)`; the output projection keeps its zero initialisation — the
hidden activations are nonzero, so it receives gradients on the very first step,
and the conditioning layers (parameter FiLM, geometry FiLM) start receiving
gradients once it has moved (the unit test asserts exactly this order). This
removes a structural blocker, not a guarantee of sample quality; report the
actual `count_parameters()` (also in `extra_repr`), not the paper's figure.

The conditioning vector is `concat(z-scored params_hist.flatten (Hp*P),
sinusoidal embedding of tau (time_embed_dim, even))`, fed to the sub-network's
input FiLM; `tau` is scaled by `_TAU_SCALE = 1000` before the standard
sinusoidal embedding (the SD3 / rectified-flow convention) so small `tau`
differences do not collapse. `velocity(z, tau, params_hist, cond)` validates
shapes (`params_hist` is `(B, Hp, P)` in **raw physical units**; z-scored inside)
and refuses `B * Zl*Yl*Xl > max_latent_tokens` — naive global attention
allocates `B*heads*N*N`, and a physical-cell budget alone does not bound it.

**Flow objective.** `forward(state, params_hist, geometry, geom_features=None,
*, generator=None)` encodes the normalised target `z1`, draws `z0 ~ N(0, I)`
and one `tau ~ U(0, 1)` per sample (from `generator` when given — the fixed
validation draws), forms `z_tau = (1 - tau) z0 + tau z1`, `v_target = z1 - z0`,
and returns `(velocity(z_tau, tau, params_hist, cond), v_target)`; the trainer
forms the MSE.

**Latent statistics.** `compute_latent_normalization(batches, max_batches=None)`
consumes `(state, geometry, geom_features)` device tuples, encodes **raw**
latents in fp32 and accumulates sums / sums of squares in **float64** over the
batch and spatial axes for every working channel (state first, then geometry
in fold mode); standard deviations are floored at `latent_eps` (`1e-6`),
near-constant channels (`std < 10 * latent_eps`) are warned about, non-finite
stats raise, and `set_latent_normalization(mean, std)` installs them and sets
the `latent_stats_installed` flag. `encode_latents`, `geometry_condition` and
`sample` raise until the flag is set (a trained `state_dict` carries it).
`set_normalization(state_mean, state_std, param_mean, param_std)` installs the
**parameter** stats only — the state stats are accepted for call-site parity
but ignored, because the frozen AE owns them (they travel in its `weights.pt`
and are what its encoder was pre-trained on); `normalize=False` skips the param
stats too.

**Sampling.** `sample(params_hist (B, Hp, P), geometry (*grid) | (B, *grid)
fluid mask, geom_features=None, *, initial_noise=None, generator=None,
num_steps=None) -> (B, C, *grid)` in physical units, obstacles zeroed: explicit
Euler from `tau = 0` to `1` in `num_steps` (default `num_sampling_steps`, 50)
steps — `z += dt * velocity(z, i*dt, …)` — then `decode_latents`. Pass
`initial_noise` `(B, D, Zl, Yl, Xl)` for reproducible per-member noise **or** a
`generator`; both at once is rejected. Fixed seeds guarantee identical initial
noise per member, not bitwise-identical kernels across batch sizes / devices;
the tests use tolerances.

**Precision policy.** The frozen encoder, geometry conditioning, latent
statistics and the whole sampling loop always run in **fp32** with
`torch.autocast(enabled=False)` around them; only the velocity network sees the
caller's (bf16) autocast during training. A cast *after* a bf16 encoding would
not recover fp32 latents, which is why autocast is disabled explicitly rather
than left to the caller.

The training task forces `dataset.dtype: float32`. Use `trainer.amp` /
`amp_dtype` for mixed precision of the velocity network; the data dtype never
casts the frozen AE.

**Self-contained artifact.** `ae_kwargs` (a plain, YAML-serialisable dict) and
`ae_fingerprint` are recorded at build time, and the generator's own
`state_dict` carries the frozen `ae.*` weights plus every buffer. Deployment
rebuilds with `skip_pretrained_load=True, ae_kwargs=...` — no AE directory, no
HF download — and loads one `weights.pt` strictly (§38).

### 37. `LatentFlowMatchingTrainer` — the flow-matching loss

[training/flow_matching.py](../libs/neural-surrogates/src/neural_surrogates/training/flow_matching.py)
reuses `BaseTraining`'s machinery (device/AMP, warmup+cosine LR, grad clip,
early stopping, checkpoint/resume, `metrics.csv`, best weights) exactly as
`AutoencoderTrainer` does — no pushforward curriculum, no `_final_loss`, no
discriminator. `_forward` computes one flow-matching draw on a
`SnapshotHistoryDataset` batch:

```
v_pred, v_target = model(state, params_hist, geometry, geom_features)   # under _autocast()
loss = loss_fn(v_pred.float(), v_target.float())                       # fp32, outside autocast
```

**fp32 MSE including padding.** `mask_loss` (the fluid-cell masking every
physical-space trainer here uses) is meaningless for this objective and
ignored: the regression target lives on the AE's latent grid, one token per
16³ block of *padded* cells, so there is no per-voxel fluid indicator. The loss
deliberately covers every state-latent channel at every latent position,
padded ones included — latent attention and halo decoding propagate errors at
padded positions into the retained domain, so leaving them untrained would not
be harmless. Padding sensitivity is an evaluation item (§39).
The `loss` config defaults to `torch.nn.MSELoss`; alternative experimental
losses must return a scalar over the same latent tensors.

**Deterministic validation.** The objective is stochastic (a fresh `z0` and
`tau` per sample), so a validation loss drawn with the training RNG would ride
on sampling noise and make best-weight selection / patience meaningless.
`_validate` therefore (i) requires a loader whose example order is fixed — a
`RandomSampler` or a custom batch sampler with `shuffle=True` is rejected at
construction, since re-seeding the flow draws alone would not fix a reshuffled
order — and (ii) draws its noise and flow times from a dedicated
`torch.Generator` on the model's device, re-seeded from `val_seed` at the start
of every pass. Training keeps the global RNG (`generator=None`), so validation
never perturbs the training stream; two passes on the same weights return
identical losses (tested).

**Construction and resume guards.** At construction the trainer checks the model
is generator-shaped (`ae`, `latent_stats_installed`, `velocity_net`), that no AE
parameter is thawed, that the optimizer holds only `requires_grad` parameters and
none of the AE's (weight decay on frozen tensors would still move them), and
that both loaders yield at least one batch (a `TrajectoryBatchSampler` with
`drop_last=True` and fewer samples per trajectory than its batch size is the
usual way to get an empty loader). The latent mean/std are model buffers, so
`weights.pt` and `checkpoint.pt` both carry them and `BaseTraining.fit` restores
them on resume; they must be installed *before* training and are never
recomputed here. `fit()` refuses to resume from a checkpoint whose
`latent_stats_installed` is missing/`False` (it would silently train on identity
statistics) and re-checks the flag before and after training.
`prepared_batches(loader)` yields the `(state, geometry, geom_features)` tuples
`compute_latent_normalization` consumes, through the very same
`_prepare_snapshot_batch` the training step uses, so the statistics are
estimated on exactly what the objective later encodes.

### 38. Config + script + artifacts

`train.py --config-name surrogate/train_latent_generator`
([train_latent_generator.yaml](../configs/surrogate/train_latent_generator.yaml),
`_latent_generator` in [tasks.py](../scripts/utils/tasks.py)):

```bash
pixi run -e dev python scripts/surrogate/train.py --config-name surrogate/train_latent_generator \
    autoencoder_dir=model_weights/tadpole_ae_s paths.data_dir=training_data/pyudales_idealized \
    'dataset.param_vars=[inflow_angle,velocity_magnitude]' name=latent_generator_s
```

| Block | Contents |
|---|---|
| `autoencoder_dir` | the AE export (needs a geometry path, §36) |
| `architecture` | `TadpoleLatentGenerator`: `param_history_steps: 12` (`Hp`), `hidden_size: null` (→ `D`), `n_layers`, `num_heads`, `time_embed_dim`, `film_hidden`, `mlp_ratio`, `use_checkpoint`, `normalize`, `num_sampling_steps: 50`, `latent_eps`, `max_latent_tokens: 4096` |
| `dataset` | `SnapshotHistoryDataset`: `param_vars` (the ordered conditioning schema — include *every* varying forcing parameter needed to distinguish target states), `param_history_steps: ${architecture.param_history_steps}`, `constant_prehistory: true`, `cadence_rtol: 0.1` |
| `latent_cache` | `output_root: null` (a dir: train on precomputed latents, below), `prepare_only: false` |
| `latent_stats` | `max_batches: 50` train batches, `seed: 0` (without a cache) |
| `physical_metadata` | `units` (one entry per state **and** parameter variable), `coordinate_order: [z, y, x]`, `geometry_mask_convention` (must equal `MASK_CONVENTION`, which deployment checks), `notes` |
| `trainer` | `LatentFlowMatchingTrainer`, `val_seed: 0`; `amp` wraps the velocity net only |
| `optimizer` | AdamW, `weight_decay: 1.0e-2`, handed only the velocity-net parameters |

The task takes `state_vars` / `sdf_features` / `sdf_clamp_cells` from the AE
export (the frozen encoder only understands its own inputs) and forces
`dataset.dtype: float32`; builds the model with `pretrained_ae_dir`; installs
the split's **param** stats and the conditioning schema
(`set_conditioning_schema(param_names, history_dt_seconds)`); stamps the
architecture with `skip_pretrained_load: true`, `pretrained_ae_dir: null`,
the resolved `ae_kwargs`, `hidden_size` and `mlp_ratio`, plus the resolved
`dataset.param_vars`; writes the `generator:` block below; saves `config.yaml`
**before** `fit()`; and, just before fitting, estimates the latent statistics
on `max_batches` seeded-shuffle train batches through
`trainer.prepared_batches`.

**Latent cache (optional).** The frozen encoder is deterministic (latent type
pinned to `mode`, whole frames, no crops), yet without a cache it re-encodes
every frame at every step in fp32: about 8x the velocity net's FLOPs, plus a
whole-frame read of the source data. With `latent_cache.output_root` set the
task first makes (or validates) a cache of the raw latents there
([datasets/latent_cache.py](../libs/neural-surrogates/src/neural_surrogates/datasets/latent_cache.py)
`prepare_latent_cache`): per trajectory and split, the raw state latents of
all frames (float32 `(T, D, Zl, Yl, Xl)`, memory-mapped) and the raw geometry
conditioning, plus the latent statistics over every saved train frame, which
replace the `latent_stats` estimate. It reads each trajectory once in source
time chunks and encodes it one frame at a time through `_encode_raw`, as
training does. A manifest of the AE fingerprint and `ae_kwargs`, the source
files and the dataset settings makes an interrupted run resume and refuses a
changed autoencoder or data (pick a new dir); `LatentCacheDataset` checks the
sources again when training starts. Training then reads `LatentCacheDataset` items (`latent`, `geom`,
`params_hist`) and calls `forward_cached`, which is `forward` minus the
encoder, so the objective is unchanged. `latent_cache.prepare_only=true`
encodes and exits (`job_scripts/delftblue/surrogate_prepare_latents.slurm`, a
GPU job; Snellius's environment is CPU-only, see
[job_scripts.md](job_scripts.md)); `config.yaml` keeps the source data, so
evaluation and deployment still encode from states. A complete cache is
validated without locking or writing, so several runs can share it and it may
be read-only; fingerprints are size and mtime, so permission changes or
`rsync -a` keep it valid. As for `prechunk`, the lock is node-local on BeeGFS:
run one preparation per directory and submit training with `--dependency`.

**Artifact layout** (`model_weights/<name>/`): `config.yaml`, `weights.pt`
(best-val **full** state dict: velocity net + frozen `ae.*` + every buffer),
`checkpoint.pt`, `metrics.csv`, `best_val.json`. The `generator:` block, as
written by `_generator_block` and read by the deploy side:

```yaml
generator:
  physical_schema:
    state_vars: [u, v, w]                 # ordered channel layout of the generated field
    param_vars: [inflow_angle, ...]       # ordered; deployment reads CURRENT values in this order
    param_history_steps: 12
    history_dt_seconds: 5.0               # the split's median saved cadence
    units: {u: m/s, v: m/s, w: m/s, inflow_angle: deg, ...}
    coordinate_order: [z, y, x]
    geometry_mask_convention: "blanking: 1 = obstacle; model fluid mask = 1 - blanking"
    boundary_conditions: "<physical_metadata.notes>"
    constant_forcing_notes: ""
    grid: {nz, ny, nx, dz, dy, dx, bounds: [[x0, x1], [y0, y1], [z0, z1]], dims, first_center}
    # train-split unique geometries; mask_sha256 = sha256 of the uint8 fluid
    # mask in (z, y, x) order, so a relocation of the same obstacles (identical
    # shape AND fluid_cells) is not mistaken for a trained geometry
    supported_geometries: [{shape: [nz, ny, nx], fluid_cells: <int>, mask_sha256: <sha256>, grid: {...}}, ...]
  ae_fingerprint: <sha256 of the AE export's weights.pt>
  ae_dir: <provenance path of the AE export>
  sampling: {num_steps: 50}               # the Euler step count
```

Each supported geometry carries its own `grid`, read from that state file's
coordinates (spacing = median coordinate step, `bounds` = cell edges) rather
than the corpus `config.yaml` `domain` block, which for a random-geometry
corpus is the case template, not the grid any trajectory ran on. The
top-level `grid` is the first one (legacy artifacts have only that).
Deployment rebuilds with `instantiate(cfg.architecture,
n_state_channels=len(state_vars), n_params=len(param_vars))` + a strict
`load_state_dict(weights.pt)` — no AE dir, no data.

### 39. Evaluation and acceptance gate

**Statistical acceptance precedes assimilation** (plan 07 §3): a low flow
loss alone does not establish a useful spin-up distribution, and AE
reconstruction error bounds nothing about generation or rollout error.
[scripts/surrogate/evaluate_latent_generator.py](../scripts/surrogate/evaluate_latent_generator.py)
(block `latent_generator` of [eval.yaml](../configs/surrogate/eval.yaml))
rebuilds the generator from `model_dir`, reinstalls its conditioning schema,
and compares, on up to `max_snapshots` held-out snapshots (`split: test`,
spread over the trajectories) under matched geometry and history:

* `real` — the held-out states;
* `ae_recon` — encode/decode through the frozen AE: the best the decoder can
  do, and the baseline generation error is judged against;
* `generated` — `sample()` with the true history, `num_noise_seeds` draws;
* `generated_const_history` — the **cold-start** case deployment uses (the
  last parameter row repeated `Hp` times);
* the conditioning probes `generated_shuffled_history` (histories permuted
  between snapshots) and `generated_omitted_history` (the training-mean
  parameters), and one `generated_steps<k>` per entry of `num_steps_sweep`.

```bash
pixi run -e dev python scripts/surrogate/evaluate_latent_generator.py \
    latent_generator.model_dir=model_weights/latent_generator_s \
    latent_generator.rollout_stepper_dir=model_weights/dft_s
```

**Metrics** — fluid cells only, with each trajectory's own `dz/dy/dx`, per
source and per grid, merged across trajectories on the same grid: mean and
fluctuation-RMS vertical profiles; pooled per-component histograms and the
1-Wasserstein distance to `real`; 1-D energy spectra along `x`; Reynolds
stresses (diagonal and cross terms); divergence on a **stencil-valid** fluid
mask (cells whose six face neighbours are all fluid); diversity across noise
seeds against the real pairwise spread; and sampling time per step count.
With `rollout_stepper_dir` a trained stepper is rolled out `rollout_steps`
from real, AE and generated fields.

**The gate is declared, not eyeballed** (`aggregate_report` in
[generator_evaluation.py](../libs/neural-surrogates/src/neural_surrogates/generator_evaluation.py)):
each generated-vs-real number must lie within its `acceptance` factor
(`profile_rmse_factor`, `w1_factor`, `divergence_factor`) times
`max(AE baseline, held-out bootstrap variability)` (`bootstrap_resamples`),
and the diversity must reach `diversity_min_ratio` of the real spread (near
zero is mode collapse). A non-finite number is a failure. The verdict is
printed as `PASS` / `FAIL: …` and written to `summary.json`.

Outputs in `latent_generator.output_dir`: `summary.json`, `metrics.csv`,
`states_*.png`, `profiles.png`, `histograms.png`, `spectra.png`,
`divergence.png`, `step_sweep.png` and `rollout_transients.png` (with a
rollout stepper).

The metric functions are pure numpy over `(N, C, nz, ny, nx)` stacks and a
fluid mask — no Hydra, no files, no model — so each is unit-tested on analytic
fields. They deliberately do not import `libs/evaluation`: that library is a
leaf the *scripts* depend on, while `neural_surrogates` declares only numpy /
xarray / torch.

### 40. Deployment: `spinup_source: generative`

[generative_spinup.py](../libs/neural-surrogates/src/neural_surrogates/generative_spinup.py)
provides `GenerativeSpinup`, the reusable loader/sampler both
`NeuralSurrogateForwardModel` (single member) and
`NeuralSurrogateEnsembleForwardModel` (batched) call; it is configured by the
nested `forward_model.generative_spinup` block of
[configs/model/neural_surrogate.yaml](../configs/model/neural_surrogate.yaml):

```yaml
forward_model:
  spinup_source: generative
  generative_spinup:
    model_dir: null          # a train_latent_generator export (config.yaml + weights.pt)
    template_path: null      # NetCDF with canonical coords + an explicit `blanking` mask
    seed: 0                  # base seed; member i's noise is seeded by (seed, i)
    sample_batch_size: 8     # members sampled per generator call (memory bound)
    num_sampling_steps: null # Euler steps; null -> the artifact's validated default
    expected_units: null     # optional {variable: unit} map asserted against the artifact
    save_diagnostics: false  # ignored by the current DA scripts (see below)
```

`model_dir` / `template_path` default to `null` rather than `???` so every
composition that never uses generative mode still resolves; the surrogate
validates them at construction **only** in this mode (`_build_generative_spinup`
also pops `save_diagnostics`, a run-script concern, and passes its own `device`,
`dtype`, `default_params` and `geometry_var` through). `GenerativeSpinup`
itself is a lazy handle — `config.yaml`, weights and template load on the first
`generate` — so constructing one is free and an ensemble shares a single
instance read-only across members.

| Concern | Behaviour |
|---|---|
| **Template requirements** | `template_path` is a NetCDF carrying the canonical coordinates and an explicit obstacle mask (`blanking`); its velocity values are **never** used, obstacles are never inferred from generated zeros, and a training snapshot may serve (for its metadata only). It is canonicalised through `_to_regular_grid`, reduced to its last frame, and validated against the artifact's `generator.physical_schema`: every `state_vars` variable and the mask present on `coordinate_order` dims; grid shape, spacing (within `1e-4` relative) and bounds matching the selected supported geometry's `grid` (legacy artifacts fall back to the top-level `grid`); a binary mask; and the geometry fingerprint — shape, fluid-cell count **and** `geometry_fingerprint(fluid)` (sha256 of the uint8 mask in `(z, y, x)`) — present in `supported_geometries`, so an unseen geometry is rejected rather than sampled blindly, including a relocation of the same obstacles, which matches on shape and cell count alone (initial scope is a validated supported geometry/grid; an artifact whose entries predate `mask_sha256` is refused and must be re-exported). Only the state variables and the mask are kept, so no stale template variable leaks into generated states; the fluid mask (`1 - blanking`) and, for an AE with SDF feature channels, its SDF features are computed once and cached. |
| **Mask polarity and units** | The schema's `geometry_mask_convention` must equal `MASK_CONVENTION` (`"blanking: 1 = obstacle; model fluid mask = 1 - blanking"`) *exactly* — the value `train_latent_generator.yaml` writes — and must name the `geometry_var`, so an artifact can never carry a polarity opposite to the `1 - blanking` the deployment applies. The schema's `units` must cover **every** state and parameter variable (the error lists the missing names), and the optional `expected_units` constructor kwarg / config key states the deployment's own convention: every variable listed there must match the artifact's unit, so a generator trained on `deg` cannot be driven with `rad`. |
| **Conditioning schema on the model** | After the strict `load_state_dict`, the loader calls `model.set_conditioning_schema(param_vars, history_dt_seconds)` from the artifact's `physical_schema`, and every `sample` call restates `param_names` / `history_dt_seconds` for the model to re-check (order-sensitive names, cadence within `1e-6` relative). `params_hist` is a bare `(B, Hp, P)` tensor, so nothing else would catch a reordered conditioning vector or a history saved at another cadence; supplying a claim to a model with **no** schema installed raises rather than passing silently. |
| **Current-first-knot conditioning** | `current_param_vector(params, member)` reads each member's **current** value of every `param_vars` entry in the saved order: the first knot (`isel(time=0)`) of a time-varying schedule, the scalar of a static one, `default_params` for a variable the params omit — else it raises naming the member and the variable; non-finite values raise. The physical rollout itself still uses the full parameter schedule. |
| **Constant history** | The cold start has no history, so `constant_history(values, hp)` repeats the current vector `Hp` times, `(Hp, P)` — the `constant_prehistory` convention of the training dataset (§35). This is the case the acceptance study must cover separately (§39). |
| **Per-member seeded noise / common random numbers** | Member `i`'s base noise `(D, Zl, Yl, Xl)` is drawn on CPU from its own `torch.Generator` seeded `_member_seed(seed, i) = seed * 1_000_003 + i` — a function of the configured seed and the **stable ensemble index** only, never of the batch position or composition. So member `i` sees the same latent noise across `sample_batch_size` settings and across ESMDA iterations, and a parameter update changes its sample *only* through the conditioning, keeping unrelated Monte Carlo noise out of the assimilation map. Distinct members get distinct noise; no member is ever reset to a shared draw. |
| **Regeneration on every cold forecast** | Nothing about a generated state is cached: every `generate(member_params, member_indices)` call samples afresh from the members' current parameters, in chunks of `sample_batch_size`, via `model.sample(params_hist, geometry, geom_features, initial_noise=noise, num_steps=self.num_sampling_steps)` (`num_sampling_steps` = the configured override, else the artifact's `generator.sampling.num_steps`). The output is shape-checked, re-masked (obstacle cells zero is an invariant of the class), finite-checked and written onto a deep copy of the template as one canonical `(z, y, x)` snapshot per member. A generator failure raises with the member indices and their conditioning values — there is **no** CFD fallback. |
| **Single-member path** | `_get_template_and_initial_state(state=None)` with `spinup_source == "generative"` calls `generate([params], [member_index])` (`member_index` defaults to `None`, which falls back to member 0 — right for a genuine single-model run — and warns once per process that an ensemble caller must pass a stable index, since otherwise every member would share one noise draw), runs `_validate_generated_snapshot` (every `state_vars` channel on `(z, y, x)` at the surrogate's `(nz, ny, nx)` with its cell spacing — a generator/stepper mismatch fails here rather than inside the rollout), then continues through the normal `_to_regular_grid` / `_history_window` path, so single forward runs work too. An explicit `state` always takes the warm path. The geometry channel comes from the mask on the (generated) template, or — if the state carries none — from an explicit `stl_path`, voxelised as in every other mode; only the CFD-backend and non-zero-state fallbacks are forbidden here (they need the absent backend, or would infer obstacles from generated zeros) and raise. |
| **Ensemble path** | `run_ensemble(state=None)` takes `_generative_templates` before `_spinup_templates`: one `generate` call over all members with `member_indices = range(ensemble_size)`, in the parent process; each snapshot then takes the **warm** path (same canonicalisation and history handling as the single member) and the batched `rollout_batched` follows. The CFD spin-up ensemble is never constructed and `_last_failure_substitutions` is empty (nothing to resample). |
| **No CFD anywhere** | The constructor accepts `generative`; a config-node `spinup_forward_model` is left **un-instantiated** (`None`) so a generator needs no CFD executable, case dir or preprocessing; `dirs` raises `AttributeError` so the ensemble base falls back to its own `temp_dir`; `clone_for_member` shares the backend (none) and the `GenerativeSpinup` handle; `prepare_neural_surrogate` is a no-op (as for `training_data`). |
| **Diagnostics** | Setting `generator.diagnostics_dir` makes every `generate` call write its snapshots to `<diagnostics_dir>/call_<k>/member_<i>.nc` (`k` restarts at 0 whenever the directory changes). Write-only: nothing ever reads them back as an initial state. |

**Assimilation lifecycle.** Generation happens on every **cold** forward call
(`state=None`) and nowhere else, so in the first window every ESMDA iteration
and the final posterior forecast re-condition the initial state on the
current parameters, while later windows warm-start from the carried-forward
state and generate nothing; a supplied restart state also bypasses
generation. Two things the archived `run_esmda.py` did are not done by the
current DA scripts: it rejected joint-state smoothers (a Kalman-analysed
initial state and the regenerated cold start would compete for window 0 with
no reconciliation policy — use parameter-only smoothers), and it pointed
`diagnostics_dir` at the results dir when `save_diagnostics: true` (the key
is now ignored). `H > 1` steppers get the single generated frame repeated
(the repeat-seeding warning of §12).

Tests: [tests/neural_surrogates/test_generative_spinup.py](../tests/neural_surrogates/test_generative_spinup.py) —
an instrumented stub generator injected through `GenerativeSpinup._load_model`
(noise identical across batch sizes and calls, distinct per member, changed
params rerun the generator, static/default/missing params, first-knot
conditioning, canonical coords + zero obstacles, every template check), the
forward-model and ensemble cold/warm paths, a real ESMDA first window
asserting a generator call per cold forecast, and a real trained (tiny)
`TadpoleLatentGenerator` artifact sampled through the whole path.

### 41. Limitations / deferred

- **Acceptance not yet run on real data.** Phases 1, 2 and 4 are implemented;
  the §39 statistical acceptance on a real corpus has not been performed, so
  `spinup_source: generative` is not yet validated for production assimilation.
- **Deployment histories are constant only.** Only the `Hp`-fold repeated
  current-parameter history is supported at deployment; an arbitrary supplied
  history must match `history_dt_seconds` and would need explicit resampling
  (nothing consumes one today).
- **Supported geometries only.** The template geometry must be one of the
  train-split geometries (shape + fluid-cell fingerprint); unseen geometries
  need a held-out geometry evaluation and an explicit supported-domain policy.
- **Parameter-inference only.** Joint-state smoothers are not reconciled with
  conditional regeneration (and the current DA scripts do not reject them,
  §40).
- **Noise policy.** Common random numbers across ESMDA iterations is the only
  policy; independent noise resampling across iterations is a separate future
  option.
- **Single-frame generation.** `H > 1` steppers get a repeated frame;
  state-history seed trajectories are deferred.
- **Full-snapshot training only.** `random_crop_size` is rejected; training a
  global latent generator on relocated crops needs its own assessment of
  coordinates, global correlations and conditioning.
- **Deterministic AE latents.** `latent_type` is pinned to `mode`; stochastic
  posterior targets are a separate experiment.
- **Attention memory.** Naive global attention is `B*heads*N*N`; the budget
  (`max_latent_tokens`) rejects over-large batches instead of replacing global
  attention with local windows. A separately validated memory-efficient backend
  is future work.
- **Deferred experiments** (plan 07 §6): latent caching after profiling, Heun,
  non-uniform flow-time sampling, classifier-free guidance, per-layer
  time/history conditioning, stochastic AE targets, random-crop generator
  training, state-history seed trajectories, faster encoder precision (after
  measuring representation/parity changes and recording it in the artifact).

### 42. File map

| Piece | File |
|---|---|
| `SnapshotHistoryDataset` / `snapshot_history_collate` | [datasets/snapshot_history.py](../libs/neural-surrogates/src/neural_surrogates/datasets/snapshot_history.py) |
| Latent cache (`prepare_latent_cache`, `LatentCacheDataset`) | [datasets/latent_cache.py](../libs/neural-surrogates/src/neural_surrogates/datasets/latent_cache.py) |
| Shared parameter-table reader (`load_param_table`) | [datasets/_params.py](../libs/neural-surrogates/src/neural_surrogates/datasets/_params.py) |
| `TadpoleLatentGenerator` / `LatentEncoding` | [architectures/tadpole_latent_flow.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_latent_flow.py) |
| Shared spatial helpers (`encode_spatial` / `decode_spatial`) | [architectures/_tadpole_spatial.py](../libs/neural-surrogates/src/neural_surrogates/architectures/_tadpole_spatial.py) |
| `ParamConditionedSubnetwork` (the velocity net) | [architectures/tadpole_stepper.py](../libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_stepper.py) |
| `LatentFlowMatchingTrainer` | [training/flow_matching.py](../libs/neural-surrogates/src/neural_surrogates/training/flow_matching.py) |
| `BaseTraining._prepare_snapshot_batch` | [training/base.py](../libs/neural-surrogates/src/neural_surrogates/training/base.py) |
| Training config | [configs/surrogate/train_latent_generator.yaml](../configs/surrogate/train_latent_generator.yaml) |
| Training script | [scripts/surrogate/train.py](../scripts/surrogate/train.py) (`task: latent_generator`) |
| Acceptance metrics (pure numpy) | [generator_evaluation.py](../libs/neural-surrogates/src/neural_surrogates/generator_evaluation.py) |
| Acceptance config | [configs/surrogate/eval.yaml](../configs/surrogate/eval.yaml) (block `latent_generator`) |
| Acceptance script | [scripts/surrogate/evaluate_latent_generator.py](../scripts/surrogate/evaluate_latent_generator.py) |
| `GenerativeSpinup` (deploy loader/sampler) | [generative_spinup.py](../libs/neural-surrogates/src/neural_surrogates/generative_spinup.py) |
| Forward-model / ensemble integration | [forward_model.py](../libs/neural-surrogates/src/neural_surrogates/forward_model.py), [ensemble_forward_model.py](../libs/neural-surrogates/src/neural_surrogates/ensemble_forward_model.py) |
| Deploy config block | [configs/model/neural_surrogate.yaml](../configs/model/neural_surrogate.yaml) (`forward_model.generative_spinup`) |
| `prepare_neural_surrogate` | [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py) |
| Plan | [07_latent_flow_matching_spinup.md](plans/implemented/neural_surrogates/07_latent_flow_matching_spinup.md) |
| Tests | [test_snapshot_history_dataset.py](../tests/neural_surrogates/test_snapshot_history_dataset.py), [test_tadpole_latent_flow.py](../tests/neural_surrogates/test_tadpole_latent_flow.py), [test_latent_generator_training.py](../tests/neural_surrogates/test_latent_generator_training.py), [test_latent_generator_evaluation.py](../tests/neural_surrogates/test_latent_generator_evaluation.py), [test_generative_spinup.py](../tests/neural_surrogates/test_generative_spinup.py), [test_latent_cache.py](../tests/neural_surrogates/test_latent_cache.py), shared fixtures [_latent_generator_fixtures.py](../tests/neural_surrogates/_latent_generator_fixtures.py) |

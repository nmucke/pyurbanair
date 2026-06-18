# Neural surrogate for urban airflow — model & training note

Source note for presentation material. Everything here reflects the code
as of June 2026: the `UNetConvNeXt` architecture, the `Trainer`, and the
default `train.yaml` configuration. Where a number comes from a specific
config or run it is called out, so slides can quote it safely.

Related docs: [training_data.md](training_data.md) (data generation),
[unet_convnext.md](unet_convnext.md) and
[neural_surrogate_training.md](neural_surrogate_training.md) (older,
partially superseded by this note), [codebase_guide.md](codebase_guide.md)
(orientation).

## 1. Why a surrogate

pyurbanair estimates urban wind fields by assimilating sparse sensor
observations into a CFD model with ESMDA (ensemble smoother with multiple
data assimilation). ESMDA needs **ensembles of forward runs** — typically
~100 members, repeatedly — and a full LES run (uDALES) per member is far
too expensive for that loop. The neural surrogate replaces the CFD time
stepper with a learned one: given the current 3D velocity field, the
inflow parameters, and the building geometry, predict the field one
output step later. Rolled out autoregressively, it produces full
trajectories at a tiny fraction of LES cost, and slots into the existing
ensemble machinery via
[`forward_model.py`](../libs/neural-surrogates/src/neural_surrogates/forward_model.py)
(CFD handles the spin-up transient; the surrogate takes over from there).

## 2. Training data

Produced by `scripts/generate_training_data.py` from uDALES LES runs of
the Xie & Castro urban canopy case (see [training_data.md](training_data.md)).
The current production dataset is `training_data/pyudales_medium`:

| Item | Value |
|---|---|
| Backend | uDALES (LES) |
| Grid | 100 × 40 × 16 cells (x × y × z), domain 100 m × 40 m × 32 m |
| State variables | `u, v, w` (3 channels) |
| Trajectories | 200 train / 8 val / 8 test |
| Snapshots per trajectory | 300, at 1 s output cadence |
| Training pairs (K = 5 horizon) | 59,000 |
| Parameters (per time step) | `inflow_angle` (±70°), `velocity_magnitude` (3–10 m/s), `pressure_gradient_magnitude` (constant in this dataset) |
| Geometry | static binary mask from the uDALES `blanking` field (1 = fluid, 0 = building/ground) |

Each training sample is `(state_t, params_{t..t+K-1}, geometry) →
state_{t+K}`: only the two endpoint snapshots leave disk; the model fills
the gap with its own predictions during training (see §4).
`TransitionDataset` ([data.py](../libs/neural-surrogates/src/neural_surrogates/data.py))
reads samples lazily from netCDF, so the dataset never has to fit in RAM.

A caveat baked into several design choices below: uDALES field dumps
carry **junk (non-zero) values inside obstacles**. The mask is therefore
applied to the model's inputs, its outputs, and the loss.

## 3. Model — 3D ConvNeXt-UNet

File: [unet_convnext.py](../libs/neural-surrogates/src/neural_surrogates/architectures/unet_convnext.py).
Contract: `forward(state, params, geometry) → state_next`. The medium
preset (current default) has **2.38 M parameters**.

Encoder–decoder UNet over the 3D grid; every stage is a stack of
ConvNeXt blocks:

- **Stem**: 3×3×3 conv over `state ⊕ geometry` (4 input channels).
- **Encoder**: per level, a stage of ConvNeXt blocks, then a stride-2
  conv downsample. Pre-downsample activations are kept as skips.
- **Bottleneck**: one stage at the deepest width.
- **Decoder**: transposed-conv upsample, 1×1 conv fusing the skip
  concat, then another ConvNeXt stage. 1×1 conv head.
- Arbitrary grid sizes are handled by padding to a multiple of
  `2^levels` and cropping the output back.

Each **ConvNeXt block** is: depthwise (spatial) conv → GroupNorm →
FiLM conditioning → pointwise expand → GELU → pointwise project →
residual add. Two non-standard choices, both motivated by 3D cost:

- **Separable depthwise convs** (`separable_dwconv: true`): the dense
  k³ stencil that is cheap in 2D ConvNeXt dominates compute in 3D
  (343 taps/voxel/channel at k = 7). It is factorized into three
  axis-aligned k-tap convs — same receptive field, ~16× fewer MACs in
  the depthwise step.
- **GroupNorm(8)** instead of the original LayerNorm-style
  GroupNorm(1).

### Physics-aware wrapping (the parts that matter for rollouts)

- **Input/output masking**: state is multiplied by the fluid mask
  before entering the network (kills obstacle junk) and the final
  output is masked again, so solid cells are exactly zero — including
  during autoregressive rollout, where output feeds back as input.
- **Z-score normalization** (`normalize: true`): per-channel mean/std
  of state and params over the training split's *fluid cells* are
  stored as model buffers (`set_normalization`), so they ship with the
  weights and rollout code needs no extra plumbing. Raw inputs are far
  from unit scale (e.g. inflow angle ~±70, streamwise velocity mean
  ≈ 4.3 m/s); an earlier surrogate (UPT) collapsed to constant rollouts
  without exactly this fix. Zero stds (constant params) are guarded to 1.
- **Residual prediction** (`residual: true`): the head predicts the
  normalized state *increment*, so "no change" is the zero-output
  solution. For small time steps this is a much easier function to
  learn and markedly improves rollout stability.
- **FiLM conditioning** (`conditioning: film`): the (normalized) params
  pass once through a shared 2-layer MLP embedding; every block applies
  a per-channel scale and shift (zero-initialized → blocks start as
  identity modulation) right after its norm. Scale modulation lets
  params change *how* features mix (an inflow angle rotates the flow),
  not just add an offset, replacing the older additive-bias scheme.
- **Periodic boundary in y** (`periodic_axes: [y]`, set in
  `train.yaml` because it is a property of the data, not the
  architecture): the LES runs are spanwise-periodic, so every spatial
  conv wraps circularly along y instead of zero-padding — the last
  cell's neighbour is the first cell, matching the simulation's
  boundary condition exactly (verified by cyclic-shift equivariance of
  the model output). x stays non-periodic (inflow–outflow) and z stays
  non-periodic (ground/top). Weight-compatible with non-periodic
  checkpoints; requires the y size to divide `2^levels` (40 / 8 ✓).

### Presets

Config group: [conf/neural_surrogate_architectures/unet_convnext/](../conf/neural_surrogate_architectures/unet_convnext/)
(tiny / small / medium / large / xlarge scale `base_channels`,
`channel_mults`, `depths`, `kernel_size`, `expansion`). Default in
training: **medium** (base 32, mults [1,2,4,8], depths [3,3,3,3], k = 7,
expansion 2). All presets enable the new options; checkpoints trained
before them still load because evaluation rebuilds the architecture from
the `config.yaml` saved next to each `weights.pt`.

## 4. Training procedure

Files: [training.py](../libs/neural-surrogates/src/neural_surrogates/training.py),
[train.yaml](../conf/neural_surrogate_training/train.yaml),
[train_neural_surrogate.py](../scripts/train_neural_surrogate.py).

**Objective.** MSE on the predicted field at `t + K`, computed **over
fluid cells only** (`mask_loss: true`) — obstacle interiors contain
junk in the targets and are pinned to zero in the predictions anyway.

**Pushforward training (Brandstetter et al.).** A one-step model
trained on ground-truth inputs sees a different input distribution than
it does at rollout time (its own, slightly-off predictions). So each
training sample unrolls the model K steps from `state_t`: the first
K − g steps run without gradients (the model "pollutes" its own input,
on purpose), and the loss is taken after the final step. `g =
grad_unroll_steps = 2` final steps carry gradients — backprop through
the last two model calls rather than only the last one (g = 1 is the
classic pushforward trick).

**Horizon curriculum.** The unroll length K ramps from 1 to the
dataset's target horizon (5), +1 step every 5 epochs. Validation always
runs at the full horizon K = 5, so val loss is a fixed yardstick across
the entire run — early stopping and best-weight selection stay
meaningful while the training task gets harder. Patience gets a fresh
window at each ramp stage so an intermediate plateau can't trip early
stopping.

**Optimization** (defaults from `train.yaml`):

| Knob | Value |
|---|---|
| Optimizer | AdamW, lr 1e-4, weight decay 1e-5 |
| LR schedule | 5-epoch linear warmup (from 1e-6) → cosine to 1e-6 |
| Batch size | 32 |
| Epochs | up to 200, early stop patience 15 (at final horizon only) |
| Gradient clipping | global norm 1.0 |
| Precision | bf16 autocast (fp16 + GradScaler fallback on pre-Ampere GPUs) |

**Best-weight selection.** `weights.pt` is overwritten whenever
validation (full-horizon) loss improves; at the end of `fit()` the best
weights are loaded back.

## 5. Performance engineering

What keeps the GPU busy (all in `Trainer` / `train.yaml`):

- **Async input pipeline**: 4 DataLoader workers overlap per-sample
  netCDF reads with GPU compute; pinned memory + `non_blocking` copies.
  (`persistent_workers` must stay off — the curriculum relies on the
  per-epoch worker re-fork to pick up the new horizon.)
- **Geometry de-duplication**: the mask is identical for every sample,
  so it is moved to the GPU once and broadcast with a zero-copy
  `expand`, instead of shipping B copies per batch (~25% of H2D traffic
  for a 3-channel state).
- **`torch.compile`** (`compile_model: true`): the model is a
  static-shape conv stack, ideal for Inductor fusion. Falls back to
  eager with a warning if triton is unusable. One-off recompile
  (~1 min) at start and at each curriculum horizon bump — the progress
  bar stalling there is expected, not a hang. (The cuda pixi env gets
  triton from PyPI; conda triton is uninstallable next to the gcc 15
  the CFD builds require.)
- **Separable depthwise convs** (§3) — the single biggest FLOP cut.
- **bf16 autocast** — no loss scaling, no underflow risk in the
  no-grad unroll.
- `channels_last: false` for now — the knob exists but has not been
  A/B-benchmarked on the training GPU.

## 6. Observability, checkpointing, resume

Each run writes to `model_weights/<model_name>/`:

| File | Contents |
|---|---|
| `config.yaml` | the resolved Hydra config (used later to rebuild the exact architecture) |
| `weights.pt` | best-validation model state dict (includes normalization buffers) |
| `checkpoint.pt` | full per-epoch training state: model, optimizer, scheduler, scaler, best-val/patience counters, curriculum horizon |
| `metrics.csv` | per epoch: horizon, lr, train loss, val loss, best val, wall-clock seconds |

`trainer.resume=true` continues an interrupted run exactly where it
stopped — including the curriculum horizon and patience window. (It
assumes `num_epochs` is unchanged, since the scheduler state is
restored.)

## 7. Evaluation & deployment

- [`test_neural_surrogate.py`](../scripts/test_neural_surrogate.py)
  loads `config.yaml` + `weights.pt`, rolls the model out
  autoregressively on the held-out test trajectories and reports/plots
  rollout error.
- [`forward_model.py`](../libs/neural-surrogates/src/neural_surrogates/forward_model.py)
  wraps the trained model as a pyurbanair forward model: a CFD backend
  performs the physical spin-up, then the surrogate advances the state;
  `ensemble_forward_model.py` vectorizes this over ESMDA ensemble
  members.

## 8. Suggested slide storyline

1. **Problem**: ESMDA needs ~100-member ensembles of urban-LES runs,
   repeatedly — LES is the bottleneck.
2. **Idea**: learn the LES time stepper; CFD only for spin-up.
3. **Data**: 200 LES trajectories × 300 snapshots, 100×40×16 grid,
   59k training pairs; randomized inflow angle/magnitude.
4. **Model**: 3D ConvNeXt-UNet, 2.38 M params — with three
   physics-aware wrappers: fluid masking everywhere, buffered z-score
   normalization, residual (increment) prediction. FiLM conditioning on
   inflow parameters.
5. **Training**: pushforward unrolling with a 1→5-step horizon
   curriculum; gradient through the last 2 steps; fluid-masked MSE;
   warmup+cosine; early stopping at fixed-horizon validation.
6. **Engineering**: separable 3D convs (~16× depthwise MACs cut),
   torch.compile, bf16, async netCDF pipeline — the GPU stays busy.
7. **Deployment**: drop-in pyurbanair forward model inside the ESMDA
   ensemble.

Caveats worth a footnote: EMA of weights is designed but not yet
implemented; `channels_last` untested; the normalization/residual/FiLM
options are weight-breaking, so old checkpoints run via their own saved
configs (legacy mode).

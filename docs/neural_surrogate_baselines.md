# Neural-surrogate baselines

Re-implementations of published urban-flow surrogates, trained on the same
corpora and at the same Δt as our steppers, so the comparison is fair. The
library is
[libs/neural-surrogate-baselines](../libs/neural-surrogate-baselines/src/neural_surrogate_baselines/)
(import `neural_surrogate_baselines`), installed by the pixi `baselines`
feature. Read [neural_surrogates.md](neural_surrogates.md) first: the baselines
use its datasets, trainer, scripts and forward model unchanged.

| Baseline | Papers | Classes |
|---|---|---|
| **Local-FNO** | Qin et al., *Build. Environ.* 273, 112668 (2025), arXiv:2411.11348 | `LocalFNOStepper`, `LocalFNOTrainer` |
| **SSRollingUrbanNet** (and 3DSwinUrbanNet) | Park et al., *Phys. Fluids* 38, 045136 and Park & Lee, 085167 (2026) | `SSRollingUrbanNet`, `RolloutTrainer`, `RolloutTransitionDataset` |

The plan and the assessments behind these choices are in
[plans/neural_surrogate_baselines.md](plans/neural_surrogate_baselines.md).

## 1. How they plug in

Both models satisfy the stepper contract
`forward(state (B, H·C, nz, ny, nx), params (B, P), geometry, geom_features=None) -> (B, C, nz, ny, nx)`,
expose `num_history_steps`, `n_state_channels`, `set_normalization` and
`domain_flexible = True`. So, with no change to our code:

- [scripts/surrogate/train.py](../scripts/surrogate/train.py) trains them from
  the configs in [configs/surrogate/baselines/](../configs/surrogate/baselines/).
  These compose [configs/surrogate/training.yaml](../configs/surrogate/training.yaml),
  so data, splits, dataloader and paths are the ones our steppers use.
- `eval_common.load_model`, `evaluate_stepper.py` and
  `NeuralSurrogateForwardModel` (forward runs and data assimilation) rebuild and
  roll them out from `config.yaml` + `weights.pt`.

Each paper's training recipe is a trainer subclass of
`neural_surrogates.training.base.BaseTraining`.

**Overrides.** A config's `architecture` is an interpolation
(`${baseline_architectures.<name>}`), so `architecture.<key>=...` fails on the
command line. Change the key at its source,
`baseline_architectures.<name>.<key>=...` (no `+`), or pick another entry with
`'architecture=${baseline_architectures.<other>}'`. Give every arm its own
`name`: the output folder is `weights_dir/<name>`, and with `resume: true`
(training.yaml) a run with a reused name resumes and overwrites the other.

```bash
pixi run -e cuda python scripts/surrogate/train.py --config-name surrogate/baselines/local_fno/train
pixi run -e cuda python scripts/surrogate/train.py --config-name surrogate/baselines/ssrolling/train_roll1
pixi run -e cuda python scripts/surrogate/train.py --config-name surrogate/baselines/ssrolling/finetune_roll3
pixi run -e dev python scripts/surrogate/baselines/compare.py \
    'models=[model_weights/local_fno,model_weights/ssrolling_tl_roll3,model_weights/p3d_idealized]'
```

On the clusters, `job_scripts/<machine>/surrogate_train_{local_fno,ssrolling}.slurm`
run these on a GPU ([job_scripts.md](job_scripts.md)).

**Data loading.** Each sample reads a few whole frames (Local-FNO t−1…t+1,
SSRollingUrbanNet t−1…t+K; Local-FNO's patches are cut on the GPU). From a
corpus stored in large zlib time chunks that is decompression-bound (on
DelftBlue ~1.7 s CPU for 3 frames); a whole-frame copy (`prechunk`, see
[neural_surrogates.md](neural_surrogates.md) §29) costs ~0.1 s CPU per frame
and mostly waits on the file system, which more workers overlap (the configs
use `num_workers: 16`). Pass `prechunk.output_root=<copy>`; the DFT's
whole-frame copy of the same corpus is reused as is.

Both papers' models have no inflow parameters. Ours vary in time, so both
baselines take the z-scored `param_vars`. That is the main deviation, listed
with the others below.

## 2. Local-FNO

[local_fno/](../libs/neural-surrogate-baselines/src/neural_surrogate_baselines/local_fno/):
`spectral.py` (the FNO kernel), `patches.py` (tiling), `model.py`,
`training.py`.

**Model** (their §2.4):
- **Input:** the two latest frames, the SDF of the buildings (`sdf.sdf_features`
  on the whole domain, mode `sdf`), and the parameters as constant channels.
- **Network:** pointwise-MLP lifting, `n_layers` Fourier layers
  `v <- σ(M(K v) + W v + b + v)`, pointwise-MLP projection. Width 36, modes
  (z, y, x) = (8, 16, 16).
- **Patches:** the domain is tiled into horizontal patches spanning the full
  height, each a `patch_core` core plus `patch_overlap` cells on every side.
  One shared network predicts all patches, and the output is stitched from the
  cores, so each overlap cell comes from the neighbour that owns it ("split
  evenly"). The default 64 + 2×6 = 76 cells is their 304 m inference patch at
  our 4 m spacing. **Deviation:** they trained on 2× downsampled data with
  38-cell patches (8 m) and applied the model at full resolution only for
  inference; we train and infer on 76-cell patches at our native 4 m. The
  smallest resolved wavelength is the same (about 19 m).
- **Domain edges:** the domain is first extended to whole cores plus the
  overlap, by wrapping on axes in `periodic_axes` and repeating the edge cell
  otherwise. `PatchGrid.valid` marks repeated cells so the loss skips them.
- **Output:** the next state directly, not an increment, as in the paper.

**Training** (their §2.5), `LocalFNOTrainer`:
- RMSE over every raw patch output, overlap included, on its real fluid cells.
- Adam with lr 2e-3 and weight decay 1e-4; StepLR halves the learning rate
  every epoch.
- Stop at the first validation rise (`patience: 1`), at most 12 epochs; the
  best-validation weights are kept.
- **Batch:** theirs is 42 patches of 38×38×75 = 4.6 M cells. Whole domains are
  loaded and cut into patches on the GPU, so the batch is set in domain cells:
  `cell_budget: 3e6`, `batch_size: 8` gives on average 24 of our 76×76×32
  patches = 4.5 M cells on the realistic corpus (at most 6.7 M).
- **Epoch:** theirs is ~1300 samples × 64 patches / 42 ≈ 2000 steps (~6e9
  cells). `epoch_batches: 2000` matches both with the batch above; the
  realistic corpus (500 × 118 samples) holds ~14× that, so 12 epochs see it
  about once. Validation takes the whole split (`val_batches: null`, 1180
  samples), as theirs (3 minutes of one run).

**Deviations on the realistic corpus** (beyond the parameters above):
- Time step 5 s, the corpus' frame interval and our steppers' step; their best
  were 10–20 s. `TransitionDataset` has no frame stride.
- Native 4 m patches (above), and our vertical extent: nz = 32 (128 m) against
  their 75 cells over 1 km, with the same 8 z modes.
- bf16 autocast (the FFTs stay float32); the paper does not say, likely fp32.
- `periodic_axes: [y]`: pyudales always runs y periodic (`BCym=1`), x is
  inflow/outflow on this corpus.
- 85 M parameters, as theirs: the spectral weights dominate
  (4 corners × 36² × 16·16·8 complex per layer).

**Cost on the realistic corpus** (measured, bf16): ~4 GB per million patch
cells, so the largest batch (6.7 M) peaks at ~29 GB and an A100 40 GB
suffices; forward and backward take ~15 ms per patch on an A100 (~8 ms on an
H100), and the loader keeps up (device 99% busy). An epoch takes ~15 min on
an A100, 12 epochs ~3 h. The first run on a corpus streams the whole training split once for
the normalization stats (~2 h on the realistic corpus) and caches them
in the data folder (`normalization_stats/`).

**Chosen where the paper is silent:** GELU; hidden width 2 × width in the
MLPs; z-score normalisation; Li et al.'s `rfftn` corner-block layout for the
modes; how domain edges are filled.

**Gotcha:** the spectral weights are stored as real `(..., 2)` tensors and
viewed as complex at use. `NeuralSurrogateForwardModel` casts the model with
`.to(dtype)`, which turns complex parameters real and drops their imaginary
part.

## 3. SSRollingUrbanNet

[ssrolling/](../libs/neural-surrogate-baselines/src/neural_surrogate_baselines/ssrolling/):
`aurora_adapter.py` (`UrbanAurora`, the backbone), `ssgen.py`, `model.py`,
`training.py`; plus `RolloutTransitionDataset` in
[datasets.py](../libs/neural-surrogate-baselines/src/neural_surrogate_baselines/datasets.py).

**3DSwinUrbanNet is Microsoft Aurora.** Paper 1 §II B says it is
"constructed by modifying ... Aurora", with a 3D Perceiver encoder/decoder
around a 3D Swin U-Net. Paper 2 §II B gives its size: window (2,6,12), depths
(2,6,2), heads (4,8,16)/(16,8,4), embedding 512, LoRA on all attention
projections, 451 M trainable parameters. Aurora (pinned `microsoft-aurora ==
2.0.1`, used unpatched) with these values has **451,432,016** parameters, LoRA
on every Swin block's qkv and output projections. Patch size 4 (paper 1 §III B)
and two input snapshots (Eq. 1) match too. Neither paper gives
`latent_levels`; we keep Aurora's 4.

Where unpatched Aurora differs from the papers' modified one:
- **Patches.** Aurora embeds 4×4 *horizontal* patches per level, then a
  Perceiver aggregates all levels into `latent_levels - 1` latent levels.
  Paper 2 §II B says "4 × 4 × 4 patches" and paper 1 §III B crops all three
  axes to multiples of 4, so their modification likely patches height too.
  It is not described, so we keep Aurora's.
- **Conditioning.** Paper 1 replaces Aurora's lead-time conditioning of the
  backbone with the 2D building-height map, and feeds the 3D building geometry
  average-pooled over patches and linearly projected to the encoder, backbone
  and decoder (§II B, §IV B). We give Aurora the height map as surface and
  static variable and the 3D solid mask as an input variable. The backbone's
  conditioning slot carries the inflow parameters plus a constant lead time.

**`UrbanAurora`** maps our fields onto Aurora's `Batch` and back:
- **Atmospheric variables:** the z-scored state channels (zero in buildings)
  and the 3D solid mask, one Aurora level per cell layer. The predicted mask
  is dropped.
- **Surface and static variable:** the building-height map, i.e. the solid
  fraction per column.
- **Metadata:** pseudo-degrees at 0.01° per cell (`_DEG_PER_CELL`, chosen to
  stay inside Aurora's position and scale encoding ranges), and a constant
  time.
- **Normalisation tables:** Aurora's module-global tables get identity entries
  under prefixed names (`nsb_*`), added only when missing.
- **Inflow parameters:** an MLP, added to `backbone.time_mlp`'s output (the
  adaptive-LayerNorm conditioning of every Swin block) by a forward hook.
- **Grid:** H and W are padded to a multiple of the patch size and the
  prediction cropped back.
- **Output:** the next state directly; paper 1 found increments made rollouts
  oscillate.

**Lateral boundaries.** Aurora's shifted windows wrap around its W axis (its
`warped=True` is hard-coded):
- If `periodic_axes` has `y`, y goes to W (the fields are transposed).
- With `x`, x goes to W.
- With neither, the wrap is switched off in every block.
- **Gotcha:** Aurora's wrap only joins the two edges of a stage whose patch
  count along W is a multiple of the window width (12). Elsewhere its padding
  separates them, as for a non-periodic axis. With patch 4 the wrap is fully
  effective only for an axis of 48·2^s cells per stage s. Most of our grids
  (multiples of 16) therefore behave as non-periodic in Aurora, whatever
  `periodic_axes` says. When the periodic axis is not a multiple of the patch
  size, its circular padding repeats cells inside the wrapped axis, so the
  period Aurora sees is the padded length.

**SSGen** (paper 2 §II C, Eqs. 1–8) folds height into channels, so it is built for
`n_levels` = the corpus' nz (default 32). It runs on the z-scored, masked
backbone prediction, and the model adds its output (Eq. 8). It matches the
paper module by module:
- 3×3 average-pool high-pass;
- 3×3 conv to C = 64, GN, GELU, dropout 0.1;
- an 8-channel per-level 1×3×3 building conv, averaged over height and
  projected by a 1×1 conv; gate `x' = g⊙x + 0.2·g⊙f_b`;
- three refine blocks with 8-group 3×3 convs;
- a 1×1–GN–GELU–1×1 head.

Chosen where the paper is silent: 8 GroupNorm groups, zero padding, a 1×1
gate conv. The paper's "9.5 M" SSGen parameters (461 M in total) do not follow
from this description: at its V·N_y = 4 × 68 = 272 channels the described
module has 0.21 M. Ours has 0.10 M at 3 × 32 = 96 channels.

**Training** (paper 2 §II E):
- `RolloutTransitionDataset` adds the intermediate targets `state_targets`
  `(K, C, *grid)`.
- `RolloutTrainer` unrolls all K steps with gradients and sums
  `MSE_i + α·L_spec,i` without dividing by K. `L_spec` is the spectral loss
  along y on the masked fields; by Parseval it equals `ny` times the MSE over
  all cells with the buildings zeroed (`ny` × fluid fraction × the fluid-cell
  MSE of the other term), so it only rescales the MSE (see [losses.py](../libs/neural-surrogate-baselines/src/neural_surrogate_baselines/losses.py)).
- `train_roll1.yaml` and `train_roll3.yaml` train from scratch;
  `finetune_roll3.yaml` is TL Roll-3 from a Roll-1 run.
- Paper 2 §III C: Adam (lr 1e-5, β 0.9/0.999) for every model, batch 2 (on
  16 GPUs with FSDP), early stopping with patience 50 restoring the best
  checkpoint. Its Roll-1 stopped after 440 epochs, then TL Roll-3 ran 200 more
  "under the same optimizer settings and early-stopping criterion" (§II E 2);
  Roll-3 from scratch ran 499. No weight decay, clipping, schedule or precision
  is mentioned (we use none, a constant lr and bf16).
- `L_spec` is Eq. 10: `(1/N_s) Σ |Q̂ − Q|²` of the DFT along the spanwise
  (lateral, periodic) axis, our `norm="backward"` mean. α = 1 (Eq. 11), and
  the rollout loss is summed with no 1/N (Eq. 13). Their MSE (Eq. 9) runs over
  every grid point; ours over fluid cells only, since our predictions are
  zero in buildings anyway.

**Note:** Aurora zero-initialises its adaptive-LayerNorm modulation, so at
initialisation the inflow parameters have no effect until training moves those
weights.

**Realistic corpus** (`pyudales_realistic`, the DFT's data, splits, u v w and
all three `param_vars`; (z, y, x) = 32 × 96…320 × 96…352 cells, up to
2.8 Mcell):
- **Unavoidable deviations from the papers:** u v w without p; Δt = 5 s (theirs
  12 s); 4 m grids of 32 levels (theirs 20 × 10 × 16 m, 152 × 68 × 88);
  z-score normalisation (theirs channel-wise min–max, §III C, chosen for
  stabler convergence); the inflow parameters; patches and conditioning as
  above.
- **Batch 2 everywhere:** `cell_budget: 6000000` (> 2 × 2.8 Mcell; training.yaml's
  4.19 M would give batch 1 on 51 of the 500 trajectories).
- **Memory and speed** (H100, bf16, batch 2, 2.8 Mcell / 1.0 Mcell grid):
  Roll-1 39 / 24 GB, 0.31 / 0.14 s per step. Roll-3 does not fit at 2.8 Mcell
  without checkpointing; `finetune_roll3`/`train_roll3` checkpoint the Swin
  blocks and Aurora's Perceiver encoder/decoder (`activation_checkpoint_modules`):
  31 GB, 1.14 s per step (blocks only: 70 GB, 1.02 s). The Perceivers hold
  most activations (they see all 32 levels); block-only checkpointing saves
  little. A100 40 GB: Roll-1 batch 2 does not fit at 2.8 Mcell without
  checkpointing, and runs ~2× slower than an H100, so the Snellius job uses `gpu_h100`.
- **Epoch time** (H100, 15 loader workers): Roll-1 ~1.5–2 h (59,000 samples;
  ~9 samples/s on a smoke over the largest grids, partly loader-bound), TL
  Roll-3 ~5 h (58,000 samples). Training to patience 50 would take months,
  so the launch lines in the job scripts cap the epochs:
  Roll-1 15 epochs, patience 5 (~26 h); TL Roll-3 7 epochs, patience 5
  (~33 h). Learning rate, batch, loss and schedule stay the papers'. At
  the same batch 2 the caps match their number of updates: their ~2,040
  training samples per epoch (7668 × 80 % of 4 s snapshots, every third)
  give ~0.9 M Roll-1 (440 epochs) and ~0.41 M TL Roll-3 samples (200
  epochs); ours give 0.89 M and 0.41 M.
- **Lateral wrap.** uDALES runs y periodic (`BCym=1`), x inflow/outflow, so
  `periodic_axes: [y]`. Per the gotcha above, the wrap is fully effective
  only where ny is a multiple of 48 (145 of 500 train grids at stage 0).
- The first run streams the training split once for the normalization stats
  and caches them in the whole-frame copy (`normalization_stats/`).

## 4. Comparing models

[scripts/surrogate/baselines/compare.py](../scripts/surrogate/baselines/compare.py)
(config [compare.yaml](../configs/surrogate/baselines/compare.yaml)) evaluates
any steppers, ours and the baselines, on the same test trajectories:

- **Same start frame.** Every model starts from the latest frame any of them
  needs as history (`max(H) - 1`), each from its own last `H` true frames.
- **Fluid cells only.** Errors are taken on fluid cells only.
  `evaluate_stepper.py`'s RMSE includes building cells.
- **Persistence reference.** Repeating the start frame is scored as well, and
  every RMSE is also reported relative to it.
- **Same data enforced.** It refuses models trained on different data: the
  training-data folder name, `state_vars`, `param_vars` and, where the corpus'
  `config.yaml` is readable, its output frequency must match. Models are
  labelled by folder name, or by the given path when two names clash.
- **Turbulence statistics** ([diagnostics.py](../libs/neural-surrogate-baselines/src/neural_surrogate_baselines/diagnostics.py)):
  - each component's spatial std relative to the truth against lead time,
    which shows smoothing or laminarisation;
  - lateral spectra of u';
  - profiles of mean u, resolved TKE and -<u'w'>.

## 5. Tests

`tests/neural_surrogate_baselines/` (CI:
`.github/workflows/tests-neural-surrogate-baselines.yml`):
- unit tests per model;
- the diagnostics;
- composition of every config under `configs/surrogate/baselines/`;
- end-to-end runs on the tests' tiny synthetic corpus through `train.py`,
  `evaluate_stepper.py`, `run_forward.py` and `compare.py`.

The tiny overlays are in `tests/configs/surrogate/baselines/<baseline>/test/`.

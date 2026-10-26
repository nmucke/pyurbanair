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
pixi run -e cuda python scripts/surrogate/train.py --config-name surrogate/baselines/local_fno/finetune \
    pretrained_dir=model_weights/local_fno name=local_fno_ft paths.data_dir=<new corpus>
pixi run -e dev bash workflows/compare_surrogates.sh \
    'models=[model_weights/local_fno,model_weights/ssrolling_tl_roll3,model_weights/p3d_idealized]'
```

On the clusters, `job_scripts/<machine>/surrogate_train_{local_fno,ssrolling}.slurm`
run these on a GPU ([job_scripts.md](job_scripts.md)), and
`surrogate_finetune_{local_fno,ssrolling}.slurm pretrained_dir=<model>
name=<new name> paths.data_dir=<new corpus>` fine-tune a trained model on new
data (the SSRolling one with Roll-3 and the activation checkpointing that
realistic-size grids need, see §3).

**Data loading.** Each sample reads a few whole frames (Local-FNO t−1…t+1,
SSRollingUrbanNet t−1…t+K; Local-FNO's patches are cut on the GPU). From a
corpus stored in large zlib time chunks that is decompression-bound (on
DelftBlue ~1.7 s CPU for 3 frames); a whole-frame copy (`prechunk`, see
[neural_surrogates.md](neural_surrogates.md) §29) costs ~0.1 s CPU per frame
and mostly waits on the file system, which more workers overlap (the configs
use `num_workers: 16`). pyudales_idealized is stored uncompressed and
contiguous, so it needs no copy: training there is compute-bound. Pass
`prechunk.output_root=<copy>`; the DFT's
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
- Stop at the first validation rise (`patience: 1`), at most 12 epochs.
- Our corpora have about 10× more samples per epoch than their record, so
  `epoch_batches` / `val_batches` cap an epoch at about theirs.

**Fine-tuning** (`local_fno/finetune.yaml`, `task: finetune_stepper`,
`method: full`): the same schedule (halving per epoch, first validation rise,
at most 12 epochs) from lr 2e-4, 10× below the paper's start, on uncapped
epochs since a fine-tuning corpus is small. The architecture, inputs and
normalisation come from the pretrained model.

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
"constructed by modifying ... Aurora". Aurora (pinned `microsoft-aurora ==
2.0.1`, used unpatched) with `AuroraSmall`'s depths (2,6,2), heads
(4,8,16)/(16,8,4), `embed_dim = 512` and its default LoRA has **451,432,016**
parameters, against the papers' "451 M". Its other defaults match the papers
too: window (2,6,12), patch size 4, two input snapshots.

Two corrections to the papers' descriptions:
- **Tokens are horizontal patches.** Aurora embeds 4×4 *horizontal* patches
  per level, then a Perceiver aggregates all levels into `latent_levels - 1`
  latent levels. Paper 2's "4×4×4 patches" is not what the code does.
- **Lead time is constant.** Aurora's lead-time encoding only accepts 1 min to
  504 h, so the papers' 12 s step cannot have been encoded as such. It is
  constant per model and carries no information.

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

**SSGen** (paper 2 Eqs. 1–8) folds height into channels, so it is built for
`n_levels` = the corpus' nz (16 in `architectures.yaml`, pyudales_idealized's). It runs on the z-scored, masked
backbone prediction, and the model adds its output (Eq. 8). Chosen where the
paper is silent: 8 GroupNorm groups, zero padding. The paper's 9.5 M SSGen
parameters do not follow from its description; ours has 0.1 M at 96 channels.

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
- Adam at a constant 1e-5, batch 2, patience 50, no gradient clipping.
- `epoch_batches` / `val_batches` (default null: the whole split) cap an
  epoch, as for Local-FNO. A full pyudales_idealized epoch is ~25k batches,
  hours per validation.

**On one 24 GB RTX 3090, pyudales_idealized** (bf16, batch 2; mixed grids up
to 16×112×224):

| Run | Peak memory | s/batch |
|---|---|---|
| Local-FNO (85 M) | 17 GB | 0.16 (batch 4) |
| SSRolling 512, Roll-1 | 17 GB | 0.19 |
| SSRolling 512, Roll-3 | out of memory on the largest grid | – |
| SSRolling 512, Roll-3 + `'trainer.activation_checkpoint_modules=[aurora.backbone.*.blocks.*]'` | 19 GB | 0.68 |
| SSRolling 256, Roll-1 | 5 GB | 0.11 |

On realistic-size grids (nz 32, up to 32×288×304), TL Roll-3
(`ssrolling_tl_roll3_realistic`) ran at batch 2 on a 40 GB A100 with the
encoder, the decoder and every Swin block recomputed (~5.3 h per 29k-batch epoch):
`'trainer.activation_checkpoint_modules=[aurora.backbone.encoder_layers.*.blocks.*,aurora.backbone.decoder_layers.*.blocks.*,aurora.encoder,aurora.decoder]'`
(set by `surrogate_finetune_ssrolling.slurm`).

Local-FNO's paper recipe runs its 12 capped epochs in about 70 min. Roll-1 at
512 with `trainer.epoch_batches=2000 trainer.val_batches=200` takes ~6 min per
epoch, and its validation loss was still falling after 120 epochs (12 h).

**Note:** Aurora zero-initialises its adaptive-LayerNorm modulation, so at
initialisation the inflow parameters have no effect until training moves those
weights.

## 4. Comparing models

`workflows/compare_surrogates.sh` (scripts in
[scripts/surrogate_comparison/](../scripts/surrogate_comparison/), config
[comparison.yaml](../configs/comparison.yaml)) compares any steppers, ours and
the baselines, on the same test trajectories. The metrics and figures are in
[scripts_and_configs.md](scripts_and_configs.md) §2.3b. For the baselines:

- **Same start frame.** Every model starts from the latest frame any of them
  needs as history (`max(H) - 1`), each from its own last `H` true frames.
- **Fluid cells only.** `evaluate_stepper.py`'s RMSE includes building cells.
- **Persistence reference.** Repeating the start frame is scored as well.
- **Same data enforced.** It refuses models trained on different data: the
  training-data folder name, `state_vars`, `param_vars` (`null` read as the
  corpus' names) and, where the corpus' `config.yaml` is readable, its output
  frequency must match.

## 5. Tests

`tests/neural_surrogate_baselines/` (CI:
`.github/workflows/tests-neural-surrogate-baselines.yml`):
- unit tests per model;
- the diagnostics;
- composition of every config under `configs/surrogate/baselines/`;
- end-to-end runs on the tests' tiny synthetic corpus through `train.py`,
  `evaluate_stepper.py` and `run_forward.py`; the comparison scripts are
  tested in `tests/scripts/test_comparison.py`.

The tiny overlays are in `tests/configs/surrogate/baselines/<baseline>/test/`.

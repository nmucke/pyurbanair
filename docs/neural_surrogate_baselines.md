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

```bash
pixi run -e cuda python scripts/surrogate/train.py --config-name surrogate/baselines/local_fno/train
pixi run -e cuda python scripts/surrogate/train.py --config-name surrogate/baselines/ssrolling/train_roll1
pixi run -e cuda python scripts/surrogate/train.py --config-name surrogate/baselines/ssrolling/finetune_roll3
pixi run -e dev python scripts/surrogate/baselines/compare.py \
    'models=[model_weights/local_fno,model_weights/ssrolling_tl_roll3,model_weights/p3d_idealized]'
```

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
  our 4 m spacing.
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
  `periodic_axes` says.

**SSGen** (paper 2 Eqs. 1–8) folds height into channels, so it is built for
`n_levels` = the corpus' nz (default 32). It runs on the z-scored, masked
backbone prediction, and the model adds its output (Eq. 8). Chosen where the
paper is silent: 8 GroupNorm groups, zero padding. The paper's 9.5 M SSGen
parameters do not follow from its description; ours has 0.1 M at 96 channels.

**Training** (paper 2 §II E):
- `RolloutTransitionDataset` adds the intermediate targets `state_targets`
  `(K, C, *grid)`.
- `RolloutTrainer` unrolls all K steps with gradients and sums
  `MSE_i + α·L_spec,i` without dividing by K. `L_spec` is the spectral loss
  along y on the masked fields; by Parseval it equals `ny · MSE`, so it only
  rescales the MSE (see [losses.py](../libs/neural-surrogate-baselines/src/neural_surrogate_baselines/losses.py)).
- `train_roll1.yaml` and `train_roll3.yaml` train from scratch;
  `finetune_roll3.yaml` is TL Roll-3 from a Roll-1 run.
- Adam at a constant 1e-5, batch 2, patience 50, no gradient clipping.

**Note:** Aurora zero-initialises its adaptive-LayerNorm modulation, so at
initialisation the inflow parameters have no effect until training moves those
weights.

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
  training-data folder, `state_vars` and `param_vars` must match.
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

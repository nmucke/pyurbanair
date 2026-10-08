# Neural-surrogate baselines: SSRollingUrbanNet and Local-FNO

Plan (2026-10-08). **Status:** phases 1–4 implemented (code, configs, tests on
the tests' synthetic data; reference doc [neural_surrogate_baselines.md](../neural_surrogate_baselines.md)).
Phases 5–7 (GPU benchmark, training runs, comparison) are open. Goal: re-implement two published urban-flow
surrogates and train them on the same corpora, at the same Δt, as our own
steppers, for a fair comparison. Both live in one separate package,
`libs/neural-surrogate-baselines` (import `neural_surrogate_baselines`), with
their own configs. Our surrogate code, configs and default runs stay
byte-identical.

| Baseline | Papers | What it is |
|---|---|---|
| **SSRollingUrbanNet** | Park et al., *Phys. Fluids* 38, 045136 and 085167 (2026) | 3DSwinUrbanNet (Microsoft Aurora) + small-scale generator; Roll-1 → TL Roll-3 training |
| **Local-FNO** | Qin et al., *Build. Environ.* 273, 112668 (2025), arXiv:2411.11348 | 3D FNO stepper trained and applied on overlapping horizontal patches |

Chen et al., *Build. Simul.* 19, 271 (2026), the 2D pedestrian-level FNO, was
assessed and not adopted. It is underspecified and its task (2D speed at
0.1 s steps) is too far from ours. Local-FNO is the same group's 3D,
better-specified method.

## 1. Separation and reuse

`scripts/surrogate/train.py` instantiates `cfg.dataset`, `cfg.architecture`
and `cfg.trainer` from their `_target_`. `scripts/utils/eval_common.load_model`
and `NeuralSurrogateForwardModel` rebuild a model the same way from
`config.yaml` and `weights.pt`. Both baselines satisfy the stepper contract
`forward(state (B, H·C, nz, ny, nx), params (B, P), geometry, geom_features=None) -> (B, C, nz, ny, nx)`,
so they plug into our training, evaluation and data-assimilation paths through
configs alone.

| New, separate | Reused unchanged |
|---|---|
| `libs/neural-surrogate-baselines/` | `scripts/surrogate/train.py`, `tasks.py` (`stepper`, `finetune_stepper`) |
| `configs/surrogate/baselines/{local_fno,ssrolling}/*.yaml` | `configs/surrogate/training.yaml` (data, dataloader, paths), composed as the base |
| `tests/neural_surrogate_baselines/`, `tests/configs/test/baselines_*.yaml` | `TransitionDataset`, `TrajectoryBatchSampler`, `get_normalization_stats`, `sdf.sdf_features` |
| `scripts/surrogate/baselines/compare.py` | `BaseTraining`: AMP, checkpoint/resume, early stopping, metrics.csv, `_aux_terms`, `_after_optimizer_step` |
| `job_scripts/<machine>/baselines_train.slurm` (GPU) | `scripts/surrogate/evaluate_stepper.py` (several models per run) |
| `docs/neural_surrogate_baselines.md` (once implemented) | `evaluation.turbulence` |

Rules:

- No edits under `libs/neural-surrogates/`, `configs/surrogate/*.yaml` (top
  level), `scripts/surrogate/*.py` or `scripts/utils/`. We subclass instead.
- External code is pinned and never patched. That covers Aurora; Local-FNO
  needs no new dependency.
- Pixi: one new feature, `baselines`, with
  `neural-surrogate-baselines = { path = "libs/neural-surrogate-baselines", editable = true }`
  and `microsoft-aurora == 2.0.1`. Add it to the `dev` and `cuda` environments
  so `evaluate_stepper` and `compare.py` can load every model.
  - Aurora's dependencies (torch, einops, timm, huggingface-hub, scipy, numpy)
    are already in `dev`.
  - Acceptance: the lock diff only adds packages. Done: Aurora also brings
    azure-storage-blob and pydantic with their dependencies (8 small
    packages); no existing version changed.

## 2. Package layout (`neural_surrogate_baselines`)

```text
neural_surrogate_baselines/
  datasets.py      RolloutTransitionDataset (adds intermediate targets; SSRollingUrbanNet only)
  losses.py        spectral_loss (Park Eq. 10), masked_rmse (Qin Eq. 8)
  diagnostics.py   lateral spectra, velocity/vorticity PDFs, TI, Reynolds stress, integral scale
  local_fno/       spectral.py, model.py (LocalFNOStepper), patches.py (tile/stitch), training.py
  ssrolling/       aurora_adapter.py (UrbanAurora), ssgen.py, model.py, training.py (RolloutTrainer)
```

Shared conventions:

- **Inflow parameters.** Neither paper has any. Both baselines take our
  z-scored `param_vars`: Local-FNO as constant input channels, UrbanAurora
  through the backbone's global conditioning vector.
- **Normalisation.** z-score, via `set_normalization` with our training-split
  statistics, saved in `weights.pt`.
- **Masking.** The output is multiplied by the fluid mask, and losses use fluid
  cells only, as in all our steppers.
- **Output.** Both papers predict the next state directly, not a residual on
  the input. We keep that.

## 3. Local-FNO

**What the paper specifies** (arXiv:2411.11348, §2.4–2.5):

- **Inputs and output.** Two input steps (t−Δt, t) of the state plus a 3D SDF,
  as channels. One step out, rolled out autoregressively.
- **Layer.** Lifting Q and projection P are pointwise MLPs with one hidden
  layer. Four layers of `v ← σ(M(K v) + W v + b + v)`: K is the spectral
  convolution, M a pointwise FFN, W a pointwise linear map, plus an identity
  skip.
- **Size.** Width 36, modes (16, 16, 8) in (x, y, z), keeping |k| ≤ k_max per
  axis (their Eq. 7).
- **Patches.** Horizontal only, a uniform grid with 20% overlap. Each patch
  predicts itself including its overlap. In the output, each overlap is "split
  evenly" between neighbours: each patch's own border prediction is dropped and
  the patch centres are stitched together. The stitched field is fed back for
  the next step. One shared model serves all patches. There is no spatial
  padding: the overlap handles the patch edges.
- **Training.**
  - RMSE loss.
  - Adam, lr 2e-3, weight decay 1e-4.
  - StepLR halving the learning rate every epoch.
  - Early stopping at the first rise in validation loss, max 12 epochs.
  - Batch = patches (42 for 8×8).
- **Grid.** The full-resolution grid is **4 m** horizontally, the same as
  ours. Inference patches are 76 × 76 cells (304 m: a 250 m core plus 20%
  overlap). They trained on 2× downsampled 38 × 38 patches only to save memory.
- **Δt.** They tested prediction intervals of 4, 10, 20 and 60 s; 10–20 s was
  best. Our 5 s cadence is inside the tested range.

**Unspecified, so ours to choose and document:**

| Detail | Our choice |
|---|---|
| Activation σ | GELU |
| FFN M and lifting/projection hidden widths | 2 × width |
| Spectral layout | Li et al.'s `rfftn` corner blocks: ±16 on x and y, 0..8 on z after the real FFT. Covers the same modes as Eq. 7, using Hermitian symmetry. |
| Normalisation | z-score |
| How patches at the domain edge are filled | see the `LocalFNOStepper` notes below |

**`LocalFNOStepper`** (`local_fno/model.py`):

- Constructor: `n_state_channels`, `n_params`, `num_history_steps = 2`,
  `width = 36`, `modes = (8, 16, 16)` in (z, y, x), `n_layers = 4`,
  `patch_core = 64`, `patch_overlap = 6`, `sdf_clamp_cells`.
  - A 64-cell core plus 6 cells each side gives 76 cells, 19% overlap. That
    matches their 304 m inference patch at our 4 m.
  - Patches span the full height (nz = 32).
- `forward` does tile → batched patch prediction → stitch the cores → mask.
  - Full grid in, full grid out, so `evaluate_stepper`, the forward model and
    ESMDA work unchanged.
  - `domain_flexible = True`: any nx, ny at the same spacing.
- `forward_patches` returns the raw per-patch outputs, for training.
- **Domain edges.** Before tiling, pad the domain to whole cores plus the
  overlap: circular in y if the corpus is laterally periodic, replicate in x.
  Crop back after stitching.
- **Geometry.** A 1-channel 3D SDF from `sdf.sdf_features(mode="sdf")` on the
  whole domain, cropped per patch. This is their Fig. 2, which computes the SDF
  on the whole domain and then splits it. The model computes it itself when
  `geom_features` is None, as P3D does.
- **Inputs per patch:** 2 × 3 velocity channels + 1 SDF + P parameter channels.
  We have no temperature, so 3 state channels instead of their 4.

**`LocalFNOTrainer(BaseTraining)`** (`local_fno/training.py`):

- K = 1 and no pushforward, as in the paper. Uses the plain `TransitionDataset`
  with `num_history_steps = 2`.
- Overrides `_final_loss`: RMSE over each raw patch output, including its
  overlap, on fluid cells, averaged over the batch of patches. This matches
  training each patch on its full output.
- StepLR stepped from `_after_optimizer_step`.
- Our epoch holds about 59k transitions × ~9–25 patches, roughly 10× theirs.
  An `epoch_batches` knob caps an epoch at about their length (~2000 steps), so
  their per-epoch schedule (halve the learning rate, stop at the first
  validation rise, ≤ 12 epochs) carries over. Validation runs on a fixed val
  subset.

**Configs** (`configs/surrogate/baselines/local_fno/`):

- `architectures.yaml`: `local_fno` (paper size).
- `train.yaml`: `task: stepper`, `dataset.num_history_steps: 2`,
  `pushforward_steps: 1`, `trainer.pushforward_epochs_per_step: null`.

**Compute:** small. A width-36 FNO on 76 × 76 × 32 patches takes GPU-hours per
run and fits one GPU.

## 4. SSRollingUrbanNet

**3DSwinUrbanNet is Microsoft Aurora.** Paper 1 §II B says it is "constructed
by modifying ... Aurora". Aurora with `AuroraSmall`'s depths (2,6,2), heads
(4,8,16)/(16,8,4), `embed_dim = 512` and its default `use_lora = True` gives
**451.4 M** parameters, against the papers' 451 M. Its defaults also match the
rest of the description:

- window (2,6,12), patch size 4;
- two input snapshots;
- Fourier embeddings for position, scale, level and lead time.

How Aurora works, and what that means here:

- **Tokens.** Each vertical level is embedded in 4×4 *horizontal* patches. A
  Perceiver then aggregates all levels into `latent_levels − 1` latent levels,
  plus one surface level. Paper 2's "4×4×4 patches" doesn't match the code.
- **Conditioning slot.** The backbone's adaptive LayerNorm takes
  `c = backbone.time_mlp(lead-time expansion)`.
- **Lead time.** Aurora's encoding only covers 1 min to 504 h; it is constant
  per model, so pass any valid value.
- **Wrap-around.** `warped = True` is hard-coded and makes Aurora's W axis
  periodic.

**SSGen** (paper 2 Eqs. 1–8, fully specified, 9.5 M parameters):

1. Fold height into channels.
2. High-pass: the field minus its 3×3 horizontal average.
3. 3×3 conv to 64 channels, GroupNorm, GELU, dropout 0.1.
4. Building-height gate: `x' = g⊙x + 0.2·g⊙f_b`.
5. Three residual grouped-conv blocks.
6. A two-layer 1×1 head.
7. Add the result to the backbone prediction.

**Training recipe** (paper 2):

- **Loss.** MSE + α·L_spec, α = 1. L_spec compares complex Fourier
  coefficients along the lateral axis. By Parseval's theorem it equals c·MSE,
  so its gradient is parallel to the MSE gradient. Implement it anyway, for
  fidelity.
- **Rollout loss.** Backprop through all N steps and sum the per-step losses,
  without dividing by N.
- **Schedule.** Roll-1 to convergence, then TL Roll-3. Comparison arms: Roll-3
  from scratch, and SSGen trained on MSE only.
- **Optimiser.** Adam, lr 1e-5 constant, batch 2, early-stopping patience 50,
  no gradient clipping.

**Modules:**

- **`UrbanAurora`** (`ssrolling/aurora_adapter.py`): 3DSwinUrbanNet behind the
  stepper contract.
  - **Grid to Aurora `Batch`.** Our `(B, H·C, nz, ny, nx)` maps to atmospheric
    variables `(B, T = 2, levels = nz, H = ny, W = nx)`.
    - Levels are cell-centre heights in metres.
    - Latitude and longitude are fixed-scale pseudo-degrees near the equator,
      the same for every sample. Phase 3 checks they stay inside Aurora's
      encoding ranges.
  - **Geometry.**
    - 2D building height (solid fraction per column) as Aurora's static
      variable.
    - The 3D solid mask as an extra atmospheric input; its prediction is
      dropped.
  - **Inflow parameters.** Through an MLP, added to `backbone.time_mlp`'s
    output by a forward hook.
  - **Aurora's statistics table.** Identity entries under prefixed names
    (`nsb_u_<level>` …) so its built-in variables are never touched.
  - **Wrap-around.** Map y to W if the corpus is laterally periodic. Otherwise
    turn it off per block with `partial(..., warped=False)`.
- **`SSGen`** (`ssrolling/ssgen.py`): uses V·nz = 3 × 32 = 96 channels. Its
  2D convs run over (y, x), so it works on any horizontal grid.
- **`SSRollingUrbanNet(UrbanAurora)`** (`ssrolling/model.py`): adds SSGen.
  `ssgen: false` gives the 3DSwinUrbanNet baseline.
- **`RolloutTransitionDataset(TransitionDataset)`** (`datasets.py`): overrides
  `__getitem__` to read frames t−H+1 … t+K in one `isel` and add
  `state_targets (K, C, *grid)`. Keeps `state_next`.
- **`RolloutTrainer(BaseTraining)`** (`ssrolling/training.py`): overrides
  `_forward`: one autocast context, gradients through all K steps, loss
  Σ_i [MSE_i + α·L_spec,i] on fluid cells. Logs `mse` and `spec` via
  `_aux_terms`.

**Configs** (`configs/surrogate/baselines/ssrolling/`):

| Config | What it trains |
|---|---|
| `architectures.yaml` | `urbanaurora_512` (451 M) and `urbanaurora_256` (113 M, size-matched arm), each with `ssgen: true/false` |
| `train_roll1.yaml` | `task: stepper`, N = 1, `grad_clip_norm: null`, no warmup/cosine |
| `finetune_roll3.yaml` | `task: finetune_stepper`, `method: full`, N = 3 |
| `train_roll3.yaml` | N = 3 from scratch |
| ablation | `loss.alpha: 0` (SSGen + MSE only) |

**Compute.** Measured on CPU with the FLOP counter: 3.9–5.1 TFLOP per training
sample. Matching their number of gradient samples is roughly 30–75 A100-hours
per TL Roll-3 model, and the model fits one GPU.

## 5. Fair-comparison protocol

- **Same data.**
  - The same `training_data/<model>_<name>` corpus and its train/val/test
    splits; with random geometries, the test layouts are held out.
  - Same `state_vars` (u, v, w) and `param_vars`.
  - **Same Δt:** the corpus's `output_frequency`, for every model, with no
    subsampling.
  - Native 4 m resolution: Local-FNO's downsampling was only a memory
    workaround.
- **Same evaluation.**
  - One `evaluate_stepper.py` run lists every model: same test samples, same
    initial states, full trajectories.
  - Then `compare.py` adds what is missing for a fair comparison:
    - fluid-masked per-step RMSE/MAE (`evaluate_stepper`'s RMSE includes
      building cells);
    - a **persistence baseline** (copy the last frame);
    - lateral spectra, PDFs, TI, Reynolds stress and integral scale at matched
      lead times.
  - `compare.py` refuses models whose recorded `dataset.root_dir`, splits,
    `state_vars`, `param_vars` or cadence differ.
- **Separate architecture from training recipe.** Each baseline is trained
  with its paper's recipe and with ours (`Trainer`, pushforward curriculum).
  Our best stepper is also trained with each paper's recipe, where it applies.
- **Equal tuning.** A small LR sweep per arm. For Local-FNO, also a 3× budget
  run, to show the paper's schedule isn't what limits it.
- **Seeds.** At least 3 per arm, reported as mean ± std. A size-matched
  SSRollingUrbanNet (113 M) next to the paper-size one.
- **History length.** Their models use H = 2. Report our stepper at its usual
  H, and at H = 2 if it supports it.
- **Label.** "Re-implementation on our data", with every deviation listed:
  - inflow conditioning;
  - no temperature or pressure;
  - Δt;
  - normalisation;
  - geometry inputs;
  - patch edge handling;
  - wrap-around.

## 6. Phases

Each phase is one branch and one PR, and leaves our setup untouched.
Local-FNO goes first: it has no external dependency and builds the shared
pieces.

| # | Phase | Contents | Done when |
|---|---|---|---|
| 0 | Decisions | §7 answered; GPU allocation confirmed | the plan is updated |
| 1 | Package + Local-FNO model | Package skeleton, pixi feature (Aurora pin included), `SpectralConv3d`, tile/stitch, `LocalFNOStepper`, `docs/neural_surrogate_baselines.md`, an AGENTS.md table row | Tests: tile → stitch identity on a random field, for several nx/ny and the padding modes; forward shape and masking; `load_model` round trip; overfits one tiny trajectory on CPU. `tests/neural_surrogates` and `tests/scripts` pass unchanged. |
| 2 | Local-FNO training + comparison tooling | `LocalFNOTrainer`, `masked_rmse`, configs, test overlay, `compare.py` with persistence and masked metrics | A tiny run through `train.py`, then `evaluate_stepper` and `compare.py` (Local-FNO vs one of our steppers vs persistence) |
| 3 | SSRollingUrbanNet model | `UrbanAurora`, `SSGen`, `SSRollingUrbanNet` | Tests: 451.4 M / 113.3 M parameter counts; forward on a tiny grid; SSGen shapes; overfits one tiny trajectory |
| 4 | SSRollingUrbanNet training | `RolloutTransitionDataset`, `spectral_loss`, `RolloutTrainer`, configs | Tests: `state_targets[-1] == state_next`; Parseval identity; summed loss equals a hand-rolled unroll. A tiny Roll-1 → TL Roll-3 run. |
| 5 | GPU smoke and benchmark | GPU SLURM file; a few hundred steps per baseline on the real corpus | Measured s/sample and peak memory, which fix the budgets |
| 6 | Runs | The §5 matrix with seeds | All runs inside budget, each with `config.yaml` and `weights.pt` |
| 7 | Comparison | `compare.py` over everything, a report in `docs/` | The table and figures; durable findings in `docs/neural_surrogate_baselines.md` |

**Effort.** About 3–4 weeks of engineering in total: Local-FNO with the shared
tooling about 1–1.5 weeks (phases 1–2), SSRollingUrbanNet about 1.5–2 weeks
(phases 3–4). Compute is about 0.5–1k A100-hours, nearly all of it
SSRollingUrbanNet.

## 7. Decisions needed before phase 1

1. **GPU machine and partition.** The `snellius` and `delftblue` pixi envs are
   CPU-only, and so are the current training SLURM files. The `cuda` env runs
   on linux-64 GPU nodes.
2. **Corpus.** It must be the one our compared steppers were trained on.
3. **Lateral boundaries.** `docs/pyudales.md` points to periodic y (`BCym = 1`)
   for uDALES corpora; confirm this per corpus.
   - Periodic: map y to Aurora's W axis and pad Local-FNO edges circularly
     in y.
   - Not periodic: turn Aurora's wrap off and replicate-pad.
4. **Our reference stepper(s)** for the comparison, and their H.
5. **Budget unit and size,** e.g. N A100-hours per SSRollingUrbanNet arm.

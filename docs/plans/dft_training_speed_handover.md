# DFT training speed and memory: handover (`feat/dft-grad-accum`)

Status as of 2026-10-12. Work moved from Snellius (H100, 93 GB) to a local
RTX 3090 (24 GB) to save SBUs. This note is the handover: what is on the
branch, what was measured on Snellius, what is still untested, and how to test
it on the 3090. Move it to [implemented/](implemented/) once the PR is merged
and the full DFT run is launched.

## Goal

Train the DFT stepper (`configs/surrogate/train_dft.yaml`, `task: dft`, §31-33
of [neural_surrogates.md](../neural_surrogates.md)) on the realistic uDALES
corpus from the size-B autoencoder `tadpole_ae_b_realistic`:

- effective batch 8;
- pushforward rollout up to 4 steps, gradients through the last **2**
  (`grad_unroll_steps: 2` — keep it, it clearly improves rollouts);
- LoRA rank 32 (keep);
- as fast as possible without running near the GPU memory limit.

## What is on the branch

All new knobs default to off: default runs are unchanged. Commits on top of
`main` (67ee5a2):

| Commit | Knob(s) | What it does | GPU-tested? |
|---|---|---|---|
| 52750a0 | `trainer.grad_accum_steps` (1) | k micro-batches per optimizer step, each loss divided by the group's size. | yes (H100 smoke, see below) |
| 52750a0 | — | Frozen geometry-branch cache now hits: the trainer passes the same broadcast geometry/SDF tensors while geometry and batch size hold (was recomputed on every model call; CPU check: 6 → 1 branch forwards for 2 batches × 3 rollout steps). | yes (part of the H100 smoke) |
| 52750a0 | — | Masked MSE/L1/Huber loss without boolean indexing (`torch.where` + an on-device fluid-cell rescale): no GPU→host sync per step. Geometry uploads `non_blocking`. Exact same value/gradient (tested, incl. NaN junk in solid cells). | yes (part of the H100 smoke) |
| e6f9c05 | `trainer.lr_warmup_per_step` (false) | Warmup advances per optimizer step instead of per epoch. With `lr_warmup_epochs=1`: LR rises linearly over epoch 0 from `lr_warmup_start`, peak from epoch 1, then cosine to `lr_min` per epoch. Fixes "epoch 0 runs entirely at 1e-6". | **no** |
| 6dca519 | `trainer.grad_accum_samples` (null) | Accumulate until ≥ N samples, step on the per-sample mean. For variable micro-batches from `batch_sampler.cell_budget`. Exclusive with `grad_accum_steps > 1`. `_optimizer_steps_per_epoch()` (used by the per-step warmup) replays the sampler's planned batch sizes. | **no** |
| 251fb3a | `trainer.profile_steps` (null) | `[start, count]`: torch.profiler over those epoch-0 micro-batches; prints it/s and GPU-busy %, writes `profile.txt` (top 40 ops) next to `weights.pt`. | **no** |
| f7c9b75 | `trainer.compile_modules` (null) | Regional `nn.Module.compile` (in place, dynamic shapes) of submodules matching globs; state-dict keys / LoRA merge unchanged. CPU parity vs eager with LoRA tested. | **no** |

Also on the branch: `job_scripts/snellius/*` and `docs/job_scripts.md` use the
new Snellius account `tdsei19851` (the old `tdsr72361` is rejected by Slurm).

Activation checkpointing and the step-1 encoder skip are now opt-in trainer
knobs (`activation_checkpoint_modules`, `unroll_no_grad_modules`), see
[Activation memory](#activation-memory-rtx-3090) below.

## Data and artifacts

On Snellius (project space `/projects/prjs2075/urbanair`, 2 TiB quota):

| What | Path | Size |
|---|---|---|
| AE (size B, 200 epochs, best val 0.0052) | `model_weights/tadpole_ae_b_realistic` | 806 MB |
| Corpus (500 train / 10 val / 10 test, 120 frames, u v w + blanking) | `training_data/pyudales_realistic` | 324 GB |
| Whole-frame prechunked copy (DFT) | `training_data/pyudales_realistic_frames` | 323 GB |
| AE-tile prechunked copy | `training_data/pyudales_realistic_rechunked` | 354 GB |

The prechunked copy only holds `state/`; params are read from the corpus
(`paths.data_dir`). A copy validates against the corpus files' size+mtime, so
copying the corpus to another machine invalidates the copy: re-make it there
with `prechunk.output_root=<new dir>` (`prechunk.prepare_only=true` for a
CPU-only pass), or skip `prechunk` (reads are ~2.5× slower per frame).

Grids: (z, y, x) = 32 × 96…272 × 128…320, **0.29M–2.8M cells, 69 distinct
shapes**; per trajectory file ≈ 0.75 GB for a mid-size grid. Cumulative counts
of the 500 train trajectories: ≤ 0.6M cells: 98; ≤ 0.8M: 146; ≤ 1.0M: 262;
≤ 1.2M: 323; ≤ 1.5M: 394. Largest: 16 trajectories at 2.80M, 26 at ~2.5M.
Model step = one stored frame = 5 s.

## Measured on Snellius (H100 93 GB / A100 40 GB, bf16 autocast)

Smoke runs, 40 min each, `name=dft_b_smoke`:

| GPU | Micro-batch | Rollout (grad steps) | Result |
|---|---|---|---|
| A100 | 8 | 1 (1) | OOM step 1 |
| A100 | 4 | 1 (1) | step 1 ≈ 38 GB, OOM step 2 |
| H100 | 8 | 1 (1) | OOM step 1 |
| H100 | 4 | 1 (1) | 3 steps, OOM at step 4 (> 93 GB) |
| H100 | 2 | 1 (1) | ran; peak 81 GB; steady 3–5 it/s (~7 samples/s) |
| H100 | 1 × accum 8 | 5 (2) | ran; **peak 57 GB**; steady **~1.85 samples/s** → ~8.6 h/epoch |
| H100 | 1 × accum 8 | 4 (2) | ran; steady **~2.05 samples/s** → ~7.9 h/epoch; GPU 86 % busy (run 27934039, below) |

Memory model (bf16 autocast, per sample, activations saved for backward):
~9.5 GB per million cells with 1 grad step, **~19 GB/Mcell with 2 grad steps**
(the no-grad prefix of the rollout frees its activations, so 3, 4 or 5 rollout
steps cost the same as 2). Peak for the 2.8M-cell grids at micro-batch 1 ≈
47–57 GB. `grad_accum_*` does not change peak memory; only the micro-batch
does. CPU breakdown of one grad step: decoder ~53 %, encoder ~44 % (exists only
because of encoder LoRA), subnetwork ~3 %.

Compute: > 90 % of FLOPs are 3×3×3 Conv3d at full and /2 resolution
(4.6 TFLOP per Mcell per sample forward). Even micro-batch 1 is compute-bound
by arithmetic; micro-batch 2 was estimated to give < 5 % per sample (not
measured cleanly). LoRA rank 32 adds ~33 % forward FLOPs over the frozen model
(rank 16: ~20 %, rank 8: ~8 %); rank stays 32 by decision.

cuDNN: with `cudnn_benchmark: true` every new (shape, batch) pair is
autotuned on one CPU core while the GPU idles (py-spy: main thread in conv3d,
GPU 0 %). 69 shapes ⇒ ~25 min of slow steps at the start of **every job**
(22 s/it falling to ~0.3–0.5 s/it). Small share of a 48 h job; dominates a
40 min smoke, so judge speed only after ~25 min (or on a fixed-shape subset).

Measurement run 27934039 (commit 52750a0; rollout 4 from the start,
micro-batch 1 × accum 8, fused AdamW, `expandable_segments`; 40 min, no OOM):

- Steady speed (after ~25 min): **~2.0–2.1 samples/s** (steps 1,399→1,899 in
  241 s) → ~7.9 h per epoch (58,000 samples at horizon 4), ~10 % faster than
  horizon 5 (one forward fewer).
- `nvidia-smi dmon` (2 s samples): SM busy **12 %** during the warm-up,
  **86 %** in the last 8 min (92 % of samples ≥ 50 %), memory bandwidth ~60 %.
- py-spy in steady state: the main Python process at ~106 % CPU (one core
  saturated), stacks split between the autograd engine (backward) and Python
  forward code (`_spatial_recon`, the subnetwork `llm.py`). So the remaining
  ~14 % GPU idle is likely CPU-side launch overhead: the case for
  `compile_modules` (fewer, fused kernels) — confirm with `profile_steps`.
- Raw logs (Snellius scratch, not kept): `nvidia-smi dmon` and py-spy dumps;
  the job log is `job_scripts/snellius/out_files/smoke_dft_h100-27934039.out`
  (gitignored).

## Recommended full-run config

Command-line overrides (`train_dft.yaml` untouched; Snellius values shown):

```bash
R=/projects/prjs2075/urbanair
sbatch -p gpu_h100 -c 16 --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  job_scripts/snellius/surrogate_train_dft.slurm \
  paths.weights_dir=$R/model_weights paths.data_dir=$R/training_data/pyudales_realistic \
  pretrained_dir=$R/model_weights/tadpole_ae_b_realistic \
  prechunk.output_root=$R/training_data/pyudales_realistic_frames \
  name=dft_b architecture.size=B \
  batch_sampler.batch_size=1 batch_sampler.cell_budget=null trainer.grad_accum_steps=8 \
  dataset.pushforward_steps=4 trainer.grad_unroll_steps=2 \
  trainer.num_epochs=15 trainer.pushforward_epochs_per_step=2 trainer.patience=3 \
  optimizer.lr=7e-4 +optimizer.fused=true trainer.lr_min=5e-6 \
  trainer.lr_warmup_epochs=1 trainer.lr_warmup_per_step=true \
  dataloader.num_workers=15
```

- `cell_budget: 393216` in `train_dft.yaml` is below every grid, so without
  `cell_budget=null` every batch is 1 regardless of `batch_size`.
- `lr 7e-4` is a cautious bump for the effective batch 8 (5e-4 was effectively
  tuned at batch 1); untested — fall back to 5e-4 if epoch 1 is unstable.
- Curriculum: horizon 1,1,2,2,3,3,4,… → 9 of 15 epochs at 4 steps. Patience
  resets at every horizon change, so it can only stop during the final stage.
- Validation always runs at the target horizon (1,150 samples, ~6 min/epoch).
- Estimated total ≈ 100 GPU-hours (≈ 2–3 × 48 h jobs); resubmit the identical
  line to resume (`checkpoint_every=1`; changed overrides resume silently with
  the new values). Copy `checkpoint.pt` aside between jobs: a time-limit kill
  during its write would corrupt it.
- `surrogate_train_dft.slurm` still asks for `gpu_a100` / 18 CPUs: an A100
  cannot run this (11 % of trajectories exceed 40 GB even at micro-batch 1), so
  pass `-p gpu_h100 -c 16` or change the script's defaults.

Once tested, candidate additions: `trainer.compile_modules='[dft.encoder.conv_encoder.conv_encoder.blocks.*,dft.decoder.conv_decoder.conv_decoder.blocks.*]'`
and/or `batch_sampler.batch_size=4 batch_sampler.cell_budget=3500000
trainer.grad_accum_steps=1 trainer.grad_accum_samples=8` (micro-batches 1–4 by
grid size, peak ≲ 70 GB on the H100).

## Skip the step-1 encoder graph (`unroll_no_grad_modules`, default off)

In the 2-grad-step rollout `x2 →[call 3]→ x3 →[call 4]→ x4`, run call 3's
**encoder** under `no_grad` (latent and skips become constants). Gradients
through call 4 and through call 3's subnetwork/decoder are unchanged (bit-
identical on CPU); encoder LoRA loses only its call-3 term (CPU, random init:
cosine 0.9991 vs full). CPU: −21 % saved activations, −25 % backward time; est.
10–15 % per step at horizon ≥ 2. Needs an explicit "a later grad step follows"
flag (at horizon 1 it must not apply, or encoder LoRA gets no gradient).
Changes training only, not the model or deployment. Where: the grad loop of
`BaseTraining._forward` and `_spatial_recon` in `tadpole_stepper.py`.

## Activation memory (RTX 3090)

Measured on the 3090 with `bench_dft_step.py` (S AE, bf16, horizon 4, 2 grad
steps, 32x144x192 = 0.885 Mcell unless noted; the user's LBM jobs shared the
GPU, so times vary by ~±8 %).

Where the memory goes (saved for backward, one sample, both grad steps,
before fused LoRA): 6.6 GB = decoder 51 % / encoder 47 % / subnetwork 0.6 % /
skip mixing 0.6 %; 30 % of it fp32 (GroupNorm's fp32 input copy and GELU's fp32
input in every conv block, softmax in the window attention). Full-resolution
32-channel tensors dominate the stems (`feature_embed`, `downsampling_layers.0`,
`decompress`) and PEFT kept a second bf16 copy of every LoRA input (its fp32
input cast, now off: bit-identical, peak 7.70 -> 6.69 GB, −13 %). With
`lora.fused_forward` the saved set is 4.5 GB, 43 % of it fp32.
`encode_spatial` also keeps a `torch.cat` copy of the full-res skips (one
chunk), ~0.36 GB here; the halo branch's rewrite already drops it.

`trainer.activation_checkpoint_modules` (fused LoRA on, 32x144x192, b = 1):

| Pattern | Peak | s/it | vs none |
|---|---|---|---|
| none | 5.28 GB | 0.261–0.273 | – |
| conv blocks (`dft.{encoder.conv_encoder.conv_encoder,decoder.conv_decoder.conv_decoder}.blocks.*`) | 3.85 GB | 0.289 | −27 %, +6…11 % |
| transformer stages (`dft.encoder.transformer_encoder`, `dft.decoder.transformer_decoder`) | 3.56 GB | 0.378 | −33 %, +40 % |
| all four stages (`dft.encoder.conv_encoder`, `…transformer_encoder`, `dft.decoder.transformer_decoder`, `…conv_decoder`) | 2.16 GB | 0.411 | −59 %, +50 % |

Same loss; gradients bit-identical for the conv blocks and within rounding for
the stages (max rel. 5e-3 in bf16, 2e-4 in fp32/TF32, cosine 1.0000; cuDNN
picks other kernels during the recompute). Largest corpus grid 32x272x320
(2.79 Mcell): 16.3 GB none, 11.8 GB conv blocks (+6 %), 6.7 GB stages (+29 %).
The subnetwork's `use_checkpoint` is not worth it (its activations are <1 %).
Do not use the vendored skip wrappers' `use_checkpoint`: it calls reentrant
checkpointing, which drops the encoder LoRA gradients.

Micro-batch (fused, 32x144x192): b = 1 3.7–3.8 samples/s, b = 2 4.35 (10.4 GB),
b = 3 4.59 (15.5 GB), b = 4 4.72 (20.6 GB); conv blocks b = 4 4.50 (14.9 GB).
So on the 3090 a micro-batch of 2–4 gains 15–25 % per sample; checkpointing
pays only where it lets a larger micro-batch (or the grid) fit at all.

Size B (random init, `skip_pretrained_load`), 32x144x192, b = 1: PEFT cast on
17.5 GB, off 15.4 GB (0.96 s/it), stages 5.2 GB (1.28 s/it, +34 %);
fused 13.2 GB (0.79 s/it), fused + conv blocks 7.5 GB (0.86 s/it, +9 %),
fused + stages 4.8 GB (1.02 s/it, +30 %). The PEFT-cast number, 19.7 GB/Mcell,
matches the ~19 GB/Mcell measured on the H100, so the per-Mcell figures should
carry over (cuDNN workspaces differ). Per Mcell at micro-batch 1 (2 grad steps):
S fused ≈ 5.9 GB; B fused ≈ 14.3, + conv blocks ≈ 8.0, + stages ≈ 4.8 GB/Mcell.
Projected H100 peaks, B, fused (≈ 0.5 GB + per-Mcell x cells x micro-batch):
2.8 Mcell grids 41 GB at b = 1 (was 47–57), 81 GB at b = 2 (too tight);
with conv blocks 23 GB at b = 1, 45 GB at b = 2, 67 GB at b = 3; ~1 Mcell grids
fused b = 4 ≈ 58 GB, with conv blocks b = 8 ≈ 64 GB.

Step-1 encoder skip (`unroll_no_grad_modules: [dft.encoder]`, real AE,
LoRA at init): every gradient but the encoder LoRA's is unchanged; the encoder
LoRA gradient keeps cosine 0.994 but only 0.61x its norm (the dropped call-3
term is ~40 % of it); whole-vector cosine 0.970. Memory −20 % (5.28 -> 4.25
GB fused), time −3…7 % fused (−14 % unfused). Not recommended: it acts like a
~40 % lower encoder-LoRA learning rate for a few % speed; checkpointing the
conv blocks saves more memory without changing the gradient.

## Testing on the RTX 3090 (24 GB)

What transfers: correctness of every knob; relative speed effects of compile,
the geometry cache, the sync-free loss, `channels_last`, `cudnn_benchmark`;
profiler breakdowns (conv vs GroupNorm/GELU vs LoRA vs gaps); the cuDNN
warm-up behaviour. What does not: absolute it/s (3090 ≈ 3–4× slower than an
H100 in bf16 conv), the memory headroom, and anything needing the large grids.

Memory: at micro-batch 1 with 2 grad steps a sample needs ~19 GB/Mcell plus
~2–3 GB for weights/optimizer/workspace, so **only grids ≤ ~0.8–0.9M cells fit**
(146 of 500 trajectories at ≤ 0.8M). With 1 grad step (rollout 1) ~9.5 GB/Mcell
⇒ grids ≤ ~2M. Ampere supports bf16 and TF32, so the same `amp_dtype` works.

Suggested setup:

1. Copy the AE dir (806 MB) and a subset corpus: val + test (20 files) and the
   train trajectories with ≤ 0.8M cells (or ~40 of them for quick tests),
   keeping the corpus layout (`state/<split>/sample_*.nc`,
   `param/<split>/sample_*.nc`, `normalization_stats/`, `geometries/`,
   `params.nc`, `config.yaml`). The DFT keeps the AE's state normalization
   (`recompute_normalization: false`), so the subset's own stats are not used
   for states; param stats are recomputed from the subset.
2. Make a whole-frame prechunked copy locally (`--config-name surrogate/train_dft
   ... prechunk.output_root=<dir> prechunk.prepare_only=true`).
3. Fixed-shape speed tests: a subset with a single grid shape removes the
   cuDNN warm-up from the comparison (or compare only after all shapes are seen).
4. Compare one knob at a time against the baseline (same subset, same horizon,
   `trainer.pushforward_start_steps=4` to measure the dominant stage),
   with `trainer.profile_steps='[START,50]'` after warm-up:
   - baseline (current branch defaults + recommended overrides);
   - `compile_modules` (watch compile time per shape and recompiles);
   - `channels_last=true` (expected no effect: `_tadpole_spatial.py` folds
     channels into the batch and calls `.contiguous()`);
   - `cudnn_benchmark=false` (shorter warm-up vs slower steady state);
   - `grad_accum_samples` + `cell_budget` (only meaningful where micro-batches > 1 fit);
   - the step-1 encoder skip, if implemented.
5. Check the per-step warmup on a short run: `metrics.csv` logs the LR at the
   start of each epoch (epoch 0 ≈ `lr_warmup_start`, epoch 1 = peak).

## Known environment issues

- `.temp` in the Snellius checkout is a dangling symlink, so 9
  `tests/scripts` tests (assimilation/forward/workflow) fail with
  `FileNotFoundError: '.temp/ensemble_experiments'`; unrelated to this branch.
  `tests/neural_surrogates` passes.
- pixi ≥ 0.72.1 is required. The `bin/test` clash between `coreutils` and
  `tempest-remap` hits every env that has both (dev, snellius, cuda), not only
  dev: `rm -rf .pixi/envs/<env>/bin/test && pixi install -e <env>`.
- Snellius only: import `neural_surrogates` before `torch` in ad-hoc scripts
  (system `libstdc++` vs netCDF4/ICU).

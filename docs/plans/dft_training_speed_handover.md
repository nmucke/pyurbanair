# DFT training speed and memory: handover for the H100 (`feat/dft-grad-accum`)

Status as of 2026-10-13. The 3090 work below (size-S AE, idealized corpus) was
confirmed on a Snellius H100 with the size-B AE `tadpole_ae_b_realistic` and the
realistic corpus: see [H100 results](#h100-results-2026-10-13) for the
measurements, the calls made and the launched command. Move this note to
[implemented/](implemented/) once the PR is merged and the run has finished.

## H100 results (2026-10-13)

Commit 1ebf77d, `gpu_h100` (H100 93 GB), bf16, size-B AE, horizon 4 with 2 grad
steps. `tests/neural_surrogates`: 625 passed.

**Fixed-shape A/B** (`scripts/tools/benchmark_dft_step.py`, 8 warm-up + 30
timed steps, one process per row, back to back on one GPU; b704c0f run from a
worktree of that commit with its own `neural_surrogates` on `PYTHONPATH`).
Repeats of two rows agreed to < 0.3 %.

| 32×144×192 (0.885 Mcell) | Micro-batch | s/it | samples/s | Peak alloc |
|---|---|---|---|---|
| b704c0f (before this work) | 1 | 0.345 | 2.90 | 17.5 GB |
| this branch, defaults | 1 | 0.295 | 3.39 | 14.7 GB |
| + `lora.fused_forward` | 1 | 0.219 | 4.58 | 12.4 GB |
| + `compile_modules` (conv stems) | 1 | 0.195 | **5.14** | 8.3 GB |
| + `compile_modules` | 2 | 0.366 | 5.46 | 16.2 GB |
| + `compile_modules` | 4 | 0.706 | 5.67 | 31.9 GB |

| 32×272×320 (2.8 Mcell) | Micro-batch | s/it | samples/s | Peak alloc |
|---|---|---|---|---|
| b704c0f | 1 | 0.973 | 1.03 | 54.1 GB |
| fused | 1 | 0.644 | 1.55 | 38.1 GB |
| fused + compile | 1 | 0.570 | **1.75** | 25.3 GB |
| fused + compile | 2 | 1.115 | 1.79 | 50.1 GB |
| + conv-block `activation_checkpoint_modules` | 2 | 1.238 | 1.62 | 40.9 GB |
| + conv-block `activation_checkpoint_modules` | 3 | 1.844 | 1.63 | 61.1 GB |

- Fused + compile: **1.7–1.8× the old samples/s at under half the memory**,
  more than projected for B (compile alone cut memory 33 %, not ~20 %).
- Memory, fused + compile: **peak ≈ 0.5 GB + 8.9 GB × Mcell × micro-batch**
  (same slope at both sizes; the projection was ~11).
- Micro-batches > 1 help much less than on the 3090: +6 % (b=2) and +10 % (b=4)
  at 0.885 Mcell, +2 % at 2.8 Mcell — B on the H100 is mostly GPU-bound.
- Conv-block checkpointing: −18 % memory for −10 % samples/s. Not needed.

**Smoke run** (job 27989597, the command below with `trainer.num_epochs=1
trainer.pushforward_start_steps=4`, 90 min, `TORCH_LOGS=recompiles`):

- No OOM, no NaN. Peak GPU 71.3 GB (`nvidia-smi`, incl. reserved/workspace).
- 3 recompiles in total, all at the start: the conv stems' grad-mode variants
  (no-grad rollout step, first grad step, input requiring grad in the second).
  None for the 69 shapes or the 1–4 micro-batch sizes.
- Host memory flat: `/dev/shm` constant at 3.85 GB (one shared copy of the
  geometries), PSS of the main process + 16 loader processes 34.6 GB, +54 MB
  over 10 min (main process anon constant at 3.98 GB).
- cuDNN warm-up ~30 min (15 s → 1 s per micro-batch). Then **~1.18
  micro-batches/s × 3.76 samples per micro-batch ≈ 4.4 samples/s** (minutes
  70–85), vs **2.05 samples/s** before (run 27934039): **2.16×**.
- Epoch at horizon 4 ≈ 58,000 / 4.4 ≈ **3.6 h** (was 7.9 h). Whole run (2 × h1,
  2 × h2, 2 × h3, 9 × h4 epochs, scaled by forward-pass count; plus validation
  and the per-job warm-up) ≈ 49–52 GPU-hours: 2 × 48 h jobs, a 3rd if the
  partial epochs lost at each time-limit kill add up.

**Calls made:** fused LoRA on; compile on; conv-block checkpointing off;
`batch_sampler.batch_size=4 batch_sampler.cell_budget=7500000` (largest
micro-batch ≈ 0.5 + 8.9 × 7.5 ≈ 67 GB: b=4 up to 1.87 Mcell, b=3 at ~2.5 Mcell,
b=2 at 2.8 Mcell) with `grad_accum_samples=8`; `optimizer.lr=7e-4` with the
per-step warmup over epoch 0 (resubmit with 5e-4 if epoch 1 is unstable —
check `metrics.csv`); `cudnn_benchmark` on;
`TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=0`; `unroll_no_grad_modules` off.

Snellius notes: the benchmark tool imports `torch` before `neural_surrogates`,
so outside `pixi run` put `$CONDA_PREFIX/lib` first on `LD_LIBRARY_PATH`
(system `libstdc++` lacks `GLIBCXX_3.4.30`); the `cuda` env's activation needs
`set +u` around `pixi shell-hook` in a `set -u` script; the tool writes to
`.temp/`, which is a dangling symlink in the Snellius checkout, so run it from
a worktree.

## Goal (unchanged)

Train the DFT stepper (`configs/surrogate/train_dft.yaml`, `task: dft`, §31-33
of [neural_surrogates.md](../neural_surrogates.md)) on the realistic uDALES
corpus from `tadpole_ae_b_realistic`: effective batch 8; pushforward up to 4
steps with gradients through the last **2** (`grad_unroll_steps: 2`, keep);
LoRA rank 32 (keep); as fast as possible without running near the memory limit.
The mathematical model must not change.

## TL;DR

- On the 3090 (S AE, 32×144×192 = 0.885 Mcell, horizon 4, 2 grad steps, bf16)
  the branch trains **~2× more samples/s** than the branch before this work (b704c0f) at the
  same peak memory (table below), with matching gradients (parity tests) and
  matching real-data training (see [Accuracy check](#accuracy-check)).
- The wins, in order: `lora.fused_forward=true` (LoRA through the merged weight:
  −36 % step time, −31 % memory), larger micro-batches (the step is close to
  CPU launch-bound at micro-batch 1, more so on an H100), regional compile of
  the two conv stems (−22 % memory, −10 % time at micro-batch 2), and several
  always-on bit-identical fixes (−12 % step time, −18 % memory together).
- **Fix a likely OOM before launching:** DataLoader workers each kept a
  shared-memory copy of every geometry they served (+325 MB/worker on the
  idealized split; est. ~11 GB/worker, ~170 GB for 15 workers on the realistic
  corpus). Fixed on this branch (`share_once`, commit 20a9b39).
- Keep `spatial_mode: global` (halo study below).
- Your job: confirm the speed/memory effects with the B AE on the H100, pick the
  micro-batch / `cell_budget`, run a smoke over all 69 shapes, then launch.

Measured on the 3090 (`scripts/tools/benchmark_dft_step.py`, 30 timed steps on a
fixed batch, back to back, idle GPU):

| Branch / settings, 32×144×192 | Micro-batch | s/it | samples/s | Peak |
|---|---|---|---|---|
| before this work, b704c0f (two runs) | 1 | 0.366–0.385 | 2.6–2.7 | 7.7 GB |
| this branch, defaults (always-on fixes only) | 1 | 0.328 | 3.05 | 6.3 GB |
| + `lora.fused_forward=true` | 1 | 0.232 | 4.32 | 4.9 GB |
| + `compile_modules` (conv stems) | 1 | 0.20–0.24 | 4.2–5.0 | 3.8 GB |
| + `compile_modules` (conv stems) | **2** | 0.379 | **5.27** | 7.5 GB |

## What is on the branch

`feat/dft-grad-accum` = its previous head b704c0f + five work branches
(pipeline, LoRA, memory, halo, compile). `tests/neural_surrogates` and the
`tests/scripts` surrogate/config/docs/import tests pass (CPU). New knobs default
to off; the always-on changes are bit-identical (tested) unless noted.

### Always on

| Commit | Change | Effect |
|---|---|---|
| 20a9b39 | `share_once`: dataset geometry masks + SDF features in one shared-memory buffer before the workers start (`TransitionDataset`, `SnapshotDataset`) | /dev/shm 775 → ~350 MB on the idealized split; prevents the per-worker copies above. Same values. |
| 20a9b39, bff865b | `lr_warmup_per_step`: warmup length counted at each epoch's own curriculum horizon and in that epoch's own seeded shuffle (incl. `cell_budget` variable batches) | Was ending warmup early / cosine one epoch late (e.g. epoch 2 lr 4.47e-4 instead of 5e-4). Only with `lr_warmup_per_step`. |
| 20a9b39 | `TrajectoryBatchSampler.set_epoch` called by `fit()` | A resumed run no longer replays epoch 0, 1, … shuffles. Fresh runs byte-identical. |
| 20a9b39 | `grad_accum_samples` gradient division as one `_foreach_div_` | fewer kernels |
| 0e3252b | Subnetwork positional embedding cached on the device (**vendored** `_tadpole/architecture/downstream/llm.py`) | Removed the last per-call host→device copy/sync: −11 % s/it at 32×144×192 |
| eb65876 | PEFT LoRA layers no longer cast their input to fp32 (it was cast straight back to bf16) | peak −13 %, slightly faster; bit-identical |
| cd3e535 | Frozen window-attention position bias (`cpb_mlp`) computed once and reused (no vendored edit; recomputed if it ever becomes trainable) | ~800 fewer kernels/step, −4 % s/it |
| 9bae22e | Global mode: the spatial runner returns the single whole-grid region directly (no zero-filled grid + copy) | small memory win |
| c5bbf85 | **Vendored** `PixelShuffle3d`: `arange(n)[:-6].tolist()` → `list(range(n-6))` (same axes) | removes a torch.compile graph break in the decoder stem |
| 3db3d50 | `trainer.channels_last=true` no longer crashes on 4-D tensors | still not recommended (eager +25 % slower) |

The two vendored edits are listed in `docs/neural_surrogates.md` with the other
local patches.

### New knobs (default off)

| Knob | What | Recommendation |
|---|---|---|
| `lora.fused_forward` | Each PEFT LoRA Linear/Conv3d computes `base(x; W + s·B@A)` in one call; the merge is rebuilt every call in fp32 (outside autocast), gradients reach A and B through it, W stays frozen. Falls back to PEFT for dropout, variants, LoRA bias, groups, merged/disabled adapters. State-dict keys, adapter save and merged `weights.pt` unchanged (bit-identical export). | **on** |
| `trainer.compile_modules` (existing, now GPU-tested) | Regional compile, dynamic shapes. Use the two conv stems `[dft.encoder.conv_encoder,dft.decoder.conv_decoder]` (the old `blocks.*` globs miss the full-resolution stem convs). | **on**, confirm on H100 |
| `trainer.activation_checkpoint_modules` | Non-reentrant checkpoint of the matched submodules (globs). Cheap pattern: the conv blocks `[dft.encoder.conv_encoder.conv_encoder.blocks.*,dft.decoder.conv_decoder.conv_decoder.blocks.*]`. | only if a micro-batch does not fit otherwise |
| `trainer.unroll_no_grad_modules` | Runs the matched modules under `no_grad` in every grad step but the last. **Changes the gradient** (encoder LoRA: cosine 0.994, 0.61× norm). | **off** |
| `architecture.spatial_checkpoint` | Recompute each spatial region's encoder/decoder call in backward (whole enc/dec in global mode; per tile in halo). Subnetwork and loss stay global. | off (global); needed only for halo |
| `architecture.spatial_batch_tiles` | Stack equal-shape halo/local tiles into one call | off (halo only) |

Tool: `scripts/tools/benchmark_dft_step.py` (run from the repo root; see its
docstring). It builds the model exactly like `train.py` (`task: dft`) and times
`trainer._train_epoch()` on one fixed batch tiled to `--grid`, so cuDNN
autotuning and compilation stay out of the timed part. Pass the corpus with
`paths.data_dir=...` after `--`.

## Findings (3090, S AE)

### LoRA is the largest cost

At size S, rank-32 LoRA costs **~1.5× the frozen convs**, not the ~33 % assumed
before: rank 32 is as wide as most conv layers (32 channels). `decompress`
(32→1 at full resolution) pays 32× its base conv for `lora_A` (a full-res
32→32 3×3×3 conv). `fused_forward` replaces the low-rank branch by one full
weight-gradient conv per layer; it wins on every layer measured, at S and at B
widths (per-layer micro-benchmark, fwd+bwd):

| Layer at 32×144×192 | PEFT | fused |
|---|---|---|
| S decompress / feature_embed / 32-ch block | 31.7 / 17.8 / 4.3 ms | 15.3 / 8.6 / 2.2 ms |
| B feature_embed / down0 / decompress | 28.3 / 32.5 / 56.0 ms | 15.7 / 18.5 / 29.2 ms |
| B 128-ch block (r/C = 0.25) | 24.7 ms | 22.1 ms |

So the gain on B/H100 should be smaller than −36 % but positive; the
full-resolution layers dominate and gain ~2× on any hardware. Parity: fp64
loss rel. 7.5e-16, LoRA grads 2e-11 (same function); fp32 grads 3e-6; in bf16
fused is slightly closer to the fp32 reference than PEFT (it rounds `W + δ` to
bf16 instead of `base + lora`). 200 real training steps: loss curve within
PEFT's own run-to-run bf16 noise.

### Launch-bound step: use micro-batches > 1

With fused LoRA one step at micro-batch 1 is ~8,000 kernels / ~11.5k aten ops;
small idealized grids all take 0.14–0.22 s/it regardless of size (the CPU
floor). The fused step's GPU time at 0.885 Mcell (~220 ms on the 3090) would be
~60–70 ms on an H100, below that floor, so **expect the H100 to be CPU-bound at
micro-batch 1** (Snellius already showed 86 % GPU busy with one core saturated
before these changes). On the 3090, micro-batch 2 / 3 / 4 gave +15 / +21 / +25 %
samples/s (fused, eager) and compile adds −10 % s/it at micro-batch 2.

### Compile

`compile_modules=[dft.encoder.conv_encoder,dft.decoder.conv_decoder]`: one
compile of 45–115 s per process (both grad modes), then **no recompiles across
all 11 idealized shapes** (a new shape costs only cuDNN autotuning, as in
eager), so the 69 realistic shapes should not add compile time — verify.
Memory −22 % at any micro-batch (fewer saved fp32 GroupNorm/GELU tensors);
time neutral at micro-batch 1, −10 % at 2. Parity vs eager: loss rel. 2e-4,
grads global rel. 4e-3 (bf16); 4e-5 / 2e-4 (fp32/TF32). Rejected: compiling
the transformer stages or the whole encoder/decoder (> 14 min compile, killed),
`compile_model`, `channels_last`. If `dft.subnetwork` is ever compiled, the
cached positional embedding in `llm.py` (an `id()`-keyed dict) graph-breaks; a
`functools.lru_cache` version traced as one graph (not on the branch).

### Memory model

Saved activations for one sample, both grad steps, S fused: decoder ~51 %,
encoder ~47 %, subnetwork < 1 %; 43 % of it fp32 (GroupNorm/GELU inputs,
attention softmax). The no-grad rollout prefix and the masked-loss temporaries
are negligible.

Per Mcell at micro-batch 1 (2 grad steps), measured on the 3090; the size-B
model was built randomly initialised (`skip_pretrained_load`) for memory/speed
only. The old B figure (19.7 GB/Mcell) matches the ~19 GB/Mcell measured on the
H100, so these should carry over (cuDNN workspaces differ):

| Setting | S GB/Mcell | B GB/Mcell | B s/it at 0.885 Mcell (3090) |
|---|---|---|---|
| old (PEFT cast) | 8.7 | 19.7 | 1.00 |
| fused | 5.9 | 14.3 | 0.79 |
| fused + compile (S measured −22 %; B assumed) | 4.3 | ~11 | – |
| fused + conv-block checkpointing | 4.3 | 8.0 | 0.86 |
| fused + checkpoint all four stages | 2.4 | 4.8 | 1.02 |

Projected H100 peaks for B (≈ 0.5 GB + per-Mcell × cells × micro-batch):
fused, 2.8 Mcell: ~41 GB at b=1, ~81 GB at b=2 (too tight); fused + conv-block
checkpointing: 23 / 45 / 67 GB at b = 1 / 2 / 3; ~1 Mcell grids fused b=4
≈ 58 GB. Compile should take ~20 % off the fused figures — **measure it**.

Do not use the vendored skip wrappers' `use_checkpoint` (reentrant checkpoint:
silently drops the encoder LoRA gradients). The subnetwork's `use_checkpoint`
is not worth it (< 1 % of activations).

### Data pipeline and cuDNN

- The loader does not starve the GPU: 48–66 batches/s locally with 4–8 workers.
  On Snellius a whole-frame read is ~0.55 s, so one sample ≈ 1.1 s of I/O wait;
  workers needed ≈ 1.1 s × samples/s × 1.5 → ~13 at 8 samples/s. Keep
  `num_workers: 15`; if the GPU starves at the new speed, check this first.
- cuDNN `benchmark: true`: the first step on a new shape costs ~12 s (≈ 43
  forward + 32 backward conv configurations, CPU-side plan building);
  `benchmark: false` costs 4–7 % at steady state. Keep it on for 48 h jobs;
  off for short smokes. `benchmark_limit` does not help. ~100 cached plans per
  grid × 69 grids ≈ 7,000 < the default 10,000 LRU limit; mixing micro-batch
  sizes multiplies the keys, so set
  `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=0` (unlimited) with `cell_budget` batching.
- GPU-tested on real data: `grad_accum_steps` / `grad_accum_samples` equal the
  large-batch gradient (fp32 rel. 2.4e-7), `profile_steps` writes `profile.txt`,
  per-step warmup + resume. Checkpoints are written only at epoch end (no
  mid-epoch resume).

### Halo vs global

Recommendation: **`global`** for training (idealized and realistic) and for
ESMDA deployment. Deploy in exactly the mode/crop/halo the model was trained
with: each mode is a different function.

- The frozen AE behaves like a function of its 16×32×32 training tile (every
  latent cell of a 32-wide tile touches the zero-padded tile border). AE
  reconstruction (z-score NRMSE, 60 val/test frames, = DFT output at init):
  local [16,32,32] 0.139, halo 16 0.230, global 0.288, halo 32 / larger cores
  0.30–0.31. Halo 16 has the worst seams (1.56× error at tile borders); global
  and halo 32 none. After a 400-step fine-tune all modes reached 0.51–0.53
  one-step NRMSE (persistence 0.83): LoRA re-adapts quickly; no clear winner.
- Halo costs 1.8–7.8× the cells (overlap; plus padding to the core multiple)
  and, without `spatial_checkpoint`, *more* training memory than global
  (12.4 vs 3.75 GB at 0.4 Mcell): autograd keeps every tile's activations.
  With `spatial_checkpoint` + `spatial_batch_tiles` halo trains, but at every
  size measured global + `spatial_checkpoint` was faster and as small or
  smaller (2.8 Mcell, S: global 1.23 s / 21.8 GB, global + checkpoint 1.37 s /
  10.1 GB, best halo 4.06 s / 11.6 GB).
- The loss and the latent subnetwork are always global in every mode; only the
  encoder/decoder calls are tiled.
- Deployment (no grad, S, 2.8 Mcell): global 0.16 s, 5.0 GB (2.7 GB with
  `max_internal_batchsize: 1`).
- Halo would only pay for domains far beyond 2.8 Mcell. Then: core z = full z
  (e.g. crop [32,64,64]), `halo_size` ≥ 16 (32 removes seams), both knobs on,
  `max_internal_batchsize` set. Further halo work (not done): recompute skips
  inside the decoder checkpoint; gather geometry features without expanding
  them per tile.

## Accuracy check

Real idealized data, S AE, on the 3090: 400 training steps on the same 400 real
train transitions in the same order (batch 1, horizon 4 with 2 grad steps, lr
5e-4 without warmup, same seed), then the mean horizon-4 loss on 20 fixed val
samples:

| Run | Train loss, mean of last 100 steps | Held-out val loss | Peak |
|---|---|---|---|
| b704c0f (before this work) | 0.595 | 0.723 | 4.13 GB |
| b704c0f, repeat | 0.605 | 0.752 | 4.13 GB |
| this branch, defaults | 0.652 | 0.789 | 3.64 GB |
| + `lora.fused_forward` + `compile_modules` (recommended) | 0.595 | 0.723 | 2.51 GB |
| + conv-block `activation_checkpoint_modules` | 0.613 | 0.727 | 2.28 GB |

Read this against the run-to-run noise: two runs of the *same* code have the
same first-step loss but differ from step 2 on (cuDNN's nondeterministic
backward) and the difference compounds over the 400 steps. To check that the
always-on changes are really identical, the defaults row was repeated with
deterministic kernels (`torch.backends.cudnn.deterministic`,
`use_deterministic_algorithms`, `cudnn_benchmark=false`): over 40 training
steps, including the optimizer updates, **this branch's defaults reproduce the
old branch bit for bit** (max loss difference 0.0; two runs of the old branch
are bit-identical too). The defaults row's gap above is therefore noise. The
recommended settings and conv-block checkpointing stay within that noise, as
expected for changes that only alter rounding (fused LoRA, compile) or nothing
(conv-block checkpointing is bit-identical). Wall time was ~200–215 s for all
runs except checkpointing (251 s): these small idealized grids are CPU-bound at
micro-batch 1, so they do not show the speedups (see the benchmark above).

Scripts and raw losses (local, not in git):
`/export/scratch1/ntm/dft_speed/acc2/` (`train_cmp.py`, `run.sh`,
`run_det.sh`, `results.jsonl`, `loss_*.txt`). On the H100, a cheap check is to
compare the first epochs' train/val loss of the full run with the earlier
`dft_b_smoke` runs.

## H100 test plan (for the Snellius agent)

All numbers above are from the 3090 with the S AE: confirm before relying on
them. Suggested order (each step ≲ 1 h on `gpu_h100`):

1. **Setup.** Check out `feat/dft-grad-accum` (pixi ≥ 0.72.1; see
   [Known environment issues](#known-environment-issues)). Run
   `pytest tests/neural_surrogates`.
2. **Fixed-shape benchmark with the B AE** (`scripts/tools/benchmark_dft_step.py
   --pretrained $R/model_weights/tadpole_ae_b_realistic -- paths.data_dir=...
   architecture.size=B`), horizon 4, at 32×144×192 (0.885 Mcell) and
   32×272×320 (2.8 Mcell). Matrix, back to back:
   - b704c0f (`git worktree add`) vs this branch's defaults vs `lora.fused_forward=true`
     vs fused + `compile_modules` — micro-batch 1;
   - fused + compile at micro-batch 2, 4 (0.885 Mcell) and 2 (2.8 Mcell);
   - fused + compile + conv-block `activation_checkpoint_modules` at the
     largest micro-batch that OOMs without it.
   Record s/it, samples/s and peak; also `nvidia-smi dmon` (SM busy %) and,
   for one setting, `trainer.profile_steps` in a real run.
   **Decide:** fused on (expected), compile on (expected; check compile time
   and memory), and GB per Mcell for the chosen setting.
3. **Pick the batching.** From the measured GB/Mcell, set
   `batch_sampler.cell_budget ≈ (target peak − ~3 GB) / (GB/Mcell) × 1e6`
   with a target peak of ~70 GB on the 93 GB card, `batch_sampler.batch_size`
   = the micro-batch cap (e.g. 4), `trainer.grad_accum_steps=1`,
   `trainer.grad_accum_samples=8` (effective batch stays 8). Grids larger than
   the budget still run at micro-batch 1 — check the 2.8 Mcell peak at b=1
   fits. Add the conv-block checkpointing only if it lets the budget roughly
   double for the common ~1 Mcell grids at ≤ 10 % time.
4. **Smoke run** (60–90 min, real corpus, the full launch command below with
   `trainer.num_epochs=1 trainer.pushforward_start_steps=4`): all 69 shapes
   and micro-batch sizes are seen during the first ~30 min (cuDNN autotune;
   judge speed only after). Check: no OOM, `nvidia-smi` peak, no recompiles
   (`TORCH_LOGS=recompiles`), `/dev/shm` stays ~one copy of the geometries
   (`df -h /dev/shm` over time), host RAM, GPU busy %, steady samples/s.
5. **Launch.** Estimate epoch time from the smoke (58,000 samples at horizon 4).

Open calls for you: `optimizer.lr` for effective batch 8 (7e-4 was a guess;
5e-4 was tuned at batch 1); whether `cudnn_benchmark` stays on; the final
`cell_budget`.

## Launch command (as launched on 2026-10-13)

```bash
R=/projects/prjs2075/urbanair
sbatch -p gpu_h100 -c 16 \
  --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=0 \
  job_scripts/snellius/surrogate_train_dft.slurm \
  paths.weights_dir=$R/model_weights paths.data_dir=$R/training_data/pyudales_realistic \
  pretrained_dir=$R/model_weights/tadpole_ae_b_realistic \
  prechunk.output_root=$R/training_data/pyudales_realistic_frames \
  name=dft_b architecture.size=B \
  lora.fused_forward=true \
  "trainer.compile_modules=[dft.encoder.conv_encoder,dft.decoder.conv_decoder]" \
  batch_sampler.batch_size=4 batch_sampler.cell_budget=7500000 \
  trainer.grad_accum_steps=1 trainer.grad_accum_samples=8 \
  dataset.pushforward_steps=4 trainer.grad_unroll_steps=2 \
  trainer.num_epochs=15 trainer.pushforward_epochs_per_step=2 trainer.patience=3 \
  optimizer.lr=7e-4 +optimizer.fused=true trainer.lr_min=5e-6 \
  trainer.lr_warmup_epochs=1 trainer.lr_warmup_per_step=true \
  dataloader.num_workers=15
```

- `cell_budget: 393216` in `train_dft.yaml` is below every realistic grid, so
  without overriding it every batch is 1.
- Curriculum: horizon 1,1,2,2,3,3,4,… → 9 of 15 epochs at 4 steps; patience
  resets at each horizon change. Validation runs at the target horizon (1,150
  samples, ~6 min/epoch before these changes).
- Resubmit the identical line to resume (`checkpoint_every=1`; changed
  overrides resume silently with the new values). Copy `checkpoint.pt` aside
  between jobs (a kill during its write corrupts it).
- `surrogate_train_dft.slurm` defaults to `gpu_a100` / 18 CPUs: pass
  `-p gpu_h100 -c 16` (an A100 cannot fit the largest grids even at b=1
  without checkpointing).

## Earlier Snellius measurements (before this branch's changes)

H100, B AE, bf16, smoke runs of 40 min: micro-batch 2 at horizon 1 peaked at
81 GB; micro-batch 1 × accum 8 at horizon 4 (2 grad steps): peak ~47–57 GB on
the 2.8 Mcell grids, steady **~2.05 samples/s** (~7.9 h/epoch), GPU 86 % busy
with one CPU core saturated (run 27934039). A100 40 GB: OOM even at
micro-batch 4, horizon 1. ~19 GB/Mcell per sample with 2 grad steps (3, 4 or 5
rollout steps cost the same as 2). cuDNN autotuning of 69 shapes: ~25 min of
slow steps at the start of every job.

## Data and artifacts (Snellius)

Project space `/projects/prjs2075/urbanair` (2 TiB quota):

| What | Path | Size |
|---|---|---|
| AE (size B, 200 epochs, best val 0.0052) | `model_weights/tadpole_ae_b_realistic` | 806 MB |
| Corpus (500 train / 10 val / 10 test, 120 frames, u v w + blanking) | `training_data/pyudales_realistic` | 324 GB |
| Whole-frame prechunked copy (DFT) | `training_data/pyudales_realistic_frames` | 323 GB |
| AE-tile prechunked copy | `training_data/pyudales_realistic_rechunked` | 354 GB |

The prechunked copy holds only `state/`; params are read from the corpus. It
validates against the corpus files' size+mtime (re-make it after copying the
corpus). Grids: (z, y, x) = 32 × 96…272 × 128…320, **0.29M–2.8M cells, 69
shapes**. Train trajectories ≤ 0.6M cells: 98; ≤ 1.0M: 262; ≤ 1.5M: 394;
largest: 16 at 2.80M, 26 at ~2.5M. Model step = one stored frame = 5 s.

## Known environment issues

- pixi ≥ 0.72.1 is required. The `bin/test` clash between `coreutils` and
  `tempest-remap` hits every env that has both (dev, snellius, cuda):
  `rm -rf .pixi/envs/<env>/bin/test && pixi install -e <env>`.
- `.temp` in the Snellius checkout is a dangling symlink, so 9 `tests/scripts`
  tests (assimilation/forward/workflow) fail with `FileNotFoundError:
  '.temp/ensemble_experiments'`; unrelated to this branch.
- Snellius only: import `neural_surrogates` before `torch` in ad-hoc scripts
  (system `libstdc++` vs netCDF4/ICU).
- `trainer.tf32=false` does not switch off PyTorch's default
  `torch.backends.cudnn.allow_tf32=True`; set it by hand for true-fp32 parity
  checks.
- Local 3090 box only: an unsearchable `/usr/local/sbin` on `PATH` makes
  inductor's `nvcc` probe raise `PermissionError`; drop it from `PATH` to
  compile there.

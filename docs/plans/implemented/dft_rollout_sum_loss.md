# Plan: summed rollout loss for the DFT time-stepper (dft_b, realistic)

Status: open, not yet trained. Branch `feat/dft-rollout-sum-loss`. The code is
implemented and unit-tested; no GPU run has finished. DelftBlue attempts
(2026-11-04 to 11-07) ran out of memory; the work moves to Snellius (H100,
dft_b's own hardware and torch). Current experiment: one arm, below.

## Why

SCALES' time-stepper (`dft_b`, `neural_surrogates.TadpoleTimeStepper`)
under-predicts small scales: at canopy top the spanwise energy at 16–32 m
(4–8 cells) is −7.2 dB below uDALES after **one** step, and stays at
−6 dB over a 50-step rollout. The SSRolling baseline loses only ~1 dB.
Diagnosis on the realistic test split (2026-11-03):

- The autoencoder is not the cause at these scales (−0.5 dB, coherence² 0.91).
- The deficit appears in the first step; rollout drift adds nothing.
- The one-step residual is largely predictable (SSRolling explains 77 % of
  the 4–8-cell variance, dft_b 34 %).

Hypothesis: the pushforward loss supervises only the **final** state of a
K-step rollout (K = 4 for dft_b), so the stepper learns the conditional mean
~20 s ahead, which is smooth, even for its one-step output. A loss on every
rollout step should sharpen the one-step prediction.

## What is implemented (on the branch)

- `neural_surrogates.Trainer(rollout_loss="final" | "sum")`, default `final`
  (unchanged training). `sum`: the mean of the masked per-step losses over the
  K steps. The last `g = grad_unroll_steps` steps backpropagate through their
  chain; each earlier step is trained on its own one-step loss from its
  detached input (step 1 from the true state), backpropagated at once so its
  graph is freed. With `grad_unroll_steps = K` every step is in the chain:
  full backpropagation through the rollout with a loss on every step, as
  SSRolling's Roll-3 (which sums instead of averaging and adds a spectral loss
  along y; neither is here). Validation always uses the final-state loss.
  Per-step training losses are logged as `train_step_1..K` in `metrics.csv`.
- `neural_surrogates.RolloutTransitionDataset` (moved from the baselines lib):
  a `TransitionDataset` that also returns every intermediate target.
- `configs/surrogate/continue_dft.yaml`: continue a trained DFT from its merged
  weights with fresh zero LoRA adapters and the same trainable modules (the
  start model is exactly the pretrained one), fixed horizon from epoch 1,
  constant lr after a one-epoch warmup, and dft_b's regional compile of the
  conv stems (`trainer.compile_modules`). Docs: `docs/neural_surrogates.md`
  ("Rollout loss", `continue_dft`).
- `job_scripts/delftblue/surrogate_continue_dft.slurm` (A100 80 GB) and
  `job_scripts/snellius/surrogate_continue_dft.slurm` (H100 94 GB, `cuda` env).

## What is tested

- `tests/neural_surrogates/test_rollout_loss.py` (15 tests, CPU): `final` is
  unchanged; `sum` equals the step-by-step reference in loss and gradients for
  (K, g) = (1, 2), (3, 1), (3, 3) and (4, 2), the last with a two-frame
  history window; step 1 is trained from the true state; early steps are
  backpropagated at once; validation keeps the final loss; the per-step terms
  are logged.
- `tests/scripts/test_surrogate.py -k continue_dft`: the config composes,
  builds from a trained DFT and trains the pretrained parameter set.
- Not tested: any GPU run to the end of an epoch (so `train_step_*` in a real
  `metrics.csv`, validation and checkpoint/resume on the real data), peak
  memory of a run that fits, epoch time.

## DelftBlue attempts (A100 80 GB, torch 2.14.0+cu130)

dft_b trained on Snellius (H100 94 GB, torch 2.12.1+cu126). Inputs checked on
DelftBlue: `/projects/urbanair/model_weights/dft_b`, the corpus
`/projects/urbanair/training_data/pyudales_realistic` (500/10/10 trajectories,
69 train grid shapes up to 304×288×32 = 2.8 Mcell) and a complete frames copy
`/scratch/ntmucke/training_data/pyudales_realistic_frames` (its manifest pins
that corpus). All smoke runs used `pretrained_dir=dft_b`, the frames copy,
`trainer.grad_accum_samples=8`, `lora.fused_forward=true`.

| Job | Setup | Result |
|---|---|---|
| 1200037 / 1200038 | K = 4, g = 2, sum, batch 4, `cell_budget=7.5M`, no compile | OOM at iteration 9 / 4 |
| 1208710 | same, compile on | 64 recompiles of the decoder stem by iteration 62 (then that frame runs eager), OOM at 117 |
| 1211925 | K = 3, g = 3 (full backprop), sum, batch 2, `cell_budget=7.5M`, compile | allocator OOM warnings from iteration 81, recompile limit at ~90, fatal OOM at 229 (78 GB allocated, inside compiled decoder code); ~1.1–1.3 it/s once running |
| 1226650 | as 1211925 with `cell_budget=4M`, 1 epoch | cancelled before it started |

Findings:

- **Memory.** By dft_b's measured model (≈ 0.5 GB + 4.45 GB per Mcell and
  sample per gradient step, compile on, H100), three gradient steps on two
  304×288 samples need ≈ 75–79 GB: `cell_budget=7.5M` does not fit 80 GB.
  `cell_budget=4M` (batches over 4 Mcell, i.e. grids above 2 Mcell, run as one
  sample) is ≈ 54 GB, ≈ 70 GB if compile fell back to eager (+30 %).
- **Recompiles.** On DelftBlue the compiled conv stems recompile per grid size
  (`tensor 'x' size mismatch at index 3/4`) and per `requires_grad` variant
  until dynamo's limit (64, `BaseTraining`), although dft_b's own smoke run
  logged 3 recompiles for all 69 shapes (`plans/dft_training_speed_handover.md`).
  Not reproduced on CPU (dynamo with the `eager` backend, size S, 10 real
  grids, bf16, sum and final rollouts: 4–8 recompiles in both torch 2.12.1 and
  2.14.0), so inductor on the GPU or torch 2.14 there are not ruled out.
  `trainer.compile_dynamic` is not the cause (null means dynamic for regional
  compile). Check with `TORCH_LOGS=recompiles` on Snellius.
- Startup ~18 min before the first iteration (frames-copy validation, loaders)
  and ~25 min of compiling and cuDNN warm-up.

## Experiment

One arm, compared only with `dft_b` (fully trained, so the comparison is
fair): continue dft_b for 3 epochs with K = 3, gradients through all three
steps and the summed loss. Name: `dft_b_sumloss_k3`. Overrides on top of
`continue_dft.yaml`:

```
pretrained_dir=<weights>/dft_b
paths.data_dir=<data>/pyudales_realistic
prechunk.output_root=<data>/pyudales_realistic_frames
dataset.pushforward_steps=3 trainer.grad_unroll_steps=3 trainer.rollout_loss=sum
batch_sampler.batch_size=2 batch_sampler.cell_budget=4000000
trainer.grad_accum_samples=8 lora.fused_forward=true trainer.num_epochs=3
```

Keep `optimizer.lr=1e-4` and `grad_clip_norm=1.0`. No activation
checkpointing. A smoke run first (one epoch, `TORCH_LOGS=recompiles`,
`nvidia-smi` logged): no OOM, recompiles settle, `train_step_1..3` and
`val_loss` in `metrics.csv`. Expected ~7 h/epoch on an A100 (rough; the H100
faster). Resubmit the same line to resume after a time limit (`trainer.resume`,
epoch checkpoints).

## Evaluation

Roll `dft_b` and `dft_b_sumloss_k3` out on the `pyudales_realistic` test split
with `workflows/compare_surrogates.sh` (`split=test max_steps=50
inlet_turbulence.enabled=true periodic_y=true device=cuda`) on a GPU node.
Compare:

- `spectra_db_u_mid` / `spectra_db_u_near_cutoff` (canopy-top spanwise u'
  energy at 4–8 and 2–4 cells, dB vs uDALES; 0 is exact) — the primary metric;
- `tke_rmse`, the TKE and −u'w' profiles;
- RMSE and ACC per lead (`rmse_h1`, `rmse_window`, ACC) — the trade-off;
- `train_step_1..3` and `val_loss` per epoch in `metrics.csv`.

Success: `dft_b_sumloss_k3` clearly reduces the dB deficit at 4–8 and 2–4
cells relative to `dft_b` (reference: −5 to −7 dB at 4–8 cells), with TKE no
worse and at most a small RMSE cost. Without a final-loss control, a gain
cannot be split between the loss and the extra training; run
`trainer.rollout_loss=final` with the same overrides if it is needed. If the
sum does not help, next: a spectral loss along y as SSRolling's, then the
residual small-scale corrector (assessment of 2026-11-03).

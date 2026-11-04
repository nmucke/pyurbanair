# Plan: summed rollout loss for the DFT time-stepper (dft_b, realistic)

Status: open. Branch `feat/dft-rollout-sum-loss`. Runs on DelftBlue (A100).

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
  K steps. The last `grad_unroll_steps` steps backpropagate through their chain
  as before; each earlier step is trained on its own one-step loss from its
  detached input (step 1 from the true state) and its loss is backpropagated at
  once, so peak memory equals `final`'s. Validation always uses the final-state
  loss, so `val_loss` is comparable between arms. Per-step training losses are
  logged as `train_step_1..K` in `metrics.csv`.
- `neural_surrogates.RolloutTransitionDataset` (moved from the baselines lib):
  a `TransitionDataset` that also returns every intermediate target.
- `configs/surrogate/continue_dft.yaml`: continue a trained DFT from its merged
  weights with fresh zero LoRA adapters and the same trainable modules (the
  start model is exactly the pretrained one), fixed horizon from epoch 1,
  constant lr after a one-epoch warmup. Docs: `docs/neural_surrogates.md`
  ("Rollout loss", `continue_dft`).
- `job_scripts/delftblue/surrogate_continue_dft.slurm`.

## Experiment

Two arms, identical except `trainer.rollout_loss`, both continued from
`dft_b` on `pyudales_realistic` with dft_b's batch settings:

| Arm | name | rollout_loss |
|---|---|---|
| summed loss | `dft_b_sumloss` | `sum` |
| control | `dft_b_finalloss_ctrl` | `final` |

The control separates the effect of the loss from that of extra training.
Overrides on top of `continue_dft.yaml` (dft_b's own values, see
`model_weights/dft_b/config.yaml`):

```
pretrained_dir=/projects/urbanair/model_weights/dft_b
paths.data_dir=/projects/urbanair/training_data/pyudales_realistic
dataset.pushforward_steps=4
batch_sampler.batch_size=4 batch_sampler.cell_budget=7500000
trainer.grad_accum_samples=8
lora.fused_forward=true
trainer.num_epochs=3
```

plus `prechunk.output_root=<frames copy>` if a re-chunked copy of
pyudales_realistic exists on DelftBlue (dft_b trained on one; it is much
faster to read). Keep `optimizer.lr=1e-4` (both arms).

Cost estimate: dft_b took ~3.8 h/epoch at K = 4 on an H100. On an A100 expect
~5–6 h/epoch for the control and ~1.6× that for `sum` (all K steps get a
backward), i.e. ~9 h/epoch. Run the arms as two parallel jobs; resubmit the
same line to resume after the 48 h limit (`trainer.resume`, epoch checkpoints).

## Evaluation

Focus: small scales and TKE. Roll all three models (`dft_b` and both arms)
out on the `pyudales_realistic` test split with `workflows/compare_surrogates.sh`
(`inlet_turbulence.enabled=true`, `periodic_y=true`, `max_steps=50`), on a GPU
node. Compare:

- `spectra_db_u_mid` / `spectra_db_u_near_cutoff` (canopy-top spanwise u'
  energy at 4–8 and 2–4 cells, dB vs uDALES; 0 is exact) — the primary metric;
- `tke_rmse`, the TKE and −u'w' profiles;
- RMSE and ACC per lead (`rmse_h1`, `rmse_window`, ACC) — the trade-off;
- the `train_step_1` column of `metrics.csv` (does the one-step loss fall?).

Success: `dft_b_sumloss` clearly reduces the dB deficit at 4–8 and 2–4 cells
relative to both `dft_b` and `dft_b_finalloss_ctrl` (reference: dft_b −5 to
−7 dB at 4–8 cells), with TKE no worse and at most a small RMSE cost. The
per-band coherence analysis is done afterwards locally from the copied
weights. If `sum` does not help, the next step is the residual small-scale
corrector (see the assessment of 2026-11-03).

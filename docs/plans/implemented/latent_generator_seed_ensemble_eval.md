# Seed-ensemble evaluation of the compact latent generators

**Status: implemented (2026-10-28).** The evaluation, comparison, job scripts
and tests are in `docs/neural_surrogates.md` §39a. The seven-model run is done.
Results: [research/latent_generator_final_2026-10.md](../../research/latent_generator_final_2026-10.md).
R3 is now `model_weights/best_latent_flow_matching_realistic`. Follows
[latent_generator_compact_full_reach.md](latent_generator_compact_full_reach.md)
and [archive/research/latent_generator_results_2026-10.md](../../archive/research/latent_generator_results_2026-10.md).

**Goal.** Compare the small latent flow-matching generators against the
solver's own spread at fixed forcing, using the seed ensemble
`training_data/pyudales_realistic_seed_ensemble`.

## Why the current evaluation is not enough

`evaluate_latent_generator.py` compares generated fields with 64 held-out
snapshots taken along 10 test trajectories. Those snapshots mix geometries,
parameter states and time. So the "real" distribution is a mixture, the pooled
W1 is noisy (a 200k random subsample, AE W1 0.008 vs 0.016 between runs), and
nothing tests the *conditional* distribution p(x | geometry, parameters) that
the generator is meant to sample.

The seed ensemble gives that conditional distribution directly. Each of the 10
test geometries (`state/sample_XXXX/member_YYYY.nc`) has 64 spun-up states with
identical geometry, parameters and spin-up; only `irandom` and the inlet seed
differ. The draws are iid (no time correlation). The states are at the
reference's first frame (t = 0 of test trajectory XXXX), whose generator
conditioning is `params_hist_for(traj, 0)`: 12 copies of the first parameter
row (`constant_prehistory`). This is the conditioning the model saw for t = 0
frames in training. The test geometries are held out from training.
sample_0004 lacks member 59 and sample_0006 lacks member 40 (n = 63).

## Models (all on the frozen `tadpole_ae_b_realistic`)

| Tag | Run dir (`/projects/urbanair/model_weights/`) | Params | Best val |
|---|---|---|---|
| R1 | `lg_b_realistic_a_h320_w1024` (`compact_wide_head`) | 27.6M | 1.115 |
| R2 | `lg_b_realistic_b_h256_m512` (`compact_bottleneck`) | 26.8M | 1.127 |
| R3 | `best_latent_flow_matching_realistic`, trained as `lg_b_realistic_h512_l6` (`compact_h512`, default) | 32.0M | 1.157 |
| R4 | `lg_b_realistic_h512_l6_xpred` (R3 + `prediction: data`) | 32.0M | 1.162 |
| R5a | `lg_b_realistic_h512_l6_ts0p5` (R3 + `time_shift: 0.5`) | 32.0M | 1.147 |
| R5b | `lg_b_realistic_h512_l6_ts3` (R3 + `time_shift: 3`) | 32.0M | 1.149 |
| h176 | `latent_generator_b_realistic_adaln_h176` (PR #221) | 5.5M | 1.285 |

Sampling: Heun, 25 steps, the established setup. The 177M/232M models are out of
scope.

## Sources per geometry g (n_R = 63–64 real members)

- `real` (R): the members, loaded and transformed exactly as the corpus loader
  does. Member 0 must reproduce `load_states(data, traj_g, [0])` up to
  round-off. This is a check, not an assumption.
- `ae_recon` (A): each real member through the frozen AE. This is the decoder
  floor, the best a generator in this latent can do.
- `generated` (G): `num_draws` (default 64) draws with g's own t = 0 history.
  Noise seeds depend only on (seed, g, draw), so every model uses the same
  noise: common random numbers for paired model comparisons.
- `generated_shuffled_params` (probe, `probe_draws`, default 16): g's geometry
  with another test geometry's t = 0 history. Does the ensemble respond to the
  parameters?

## Metrics (per geometry, per variable u/v/w, over fluid cells)

Noise floors: with 64 vs 64 samples every distance has a finite-sample size,
so each metric is either an unbiased estimator (expected value 0 or 1 when the
distributions agree) or reported next to a **split-half floor**: the same metric
between disjoint random halves of R (32 vs 32, `num_splits` random splits). The
model's value is then taken at the same sizes, G-32 vs an R-half.

1. **Mean bias.** `bias² = mean_cells (μ_S − μ_R)²`, debiased by subtracting
   `mean_cells (σ_S²/n_S + σ_R²/n_R)`. Reported raw, debiased, and standardised
   by `mean_cells σ_R²`. Per height too.
2. **Spread.** Unbiased pointwise variance. `spread_ratio = sqrt(Σσ_S² / Σσ_R²)`
   (global and per height). The spatial pattern: the Pearson correlation of
   σ_S and σ_R maps over cells, and the RMS of log(σ_S/σ_R). Does the model put
   variance where the solver does (wakes, canyons, shear layer)?
3. **Fair CRPS** at a fixed, deterministic set of `num_points` fluid cells per
   geometry, shared by every source and model. It is scored over every real
   member as the verifying truth: `CRPS(S, x_r)`. Oracle floor: leave-one-out
   `CRPS(R∖r, x_r)`. Report `crps_ratio = CRPS(S)/CRPS_loo(R)` (1 = as sharp
   and calibrated as the solver).
4. **Rank histogram** over the same points, with each real member as truth,
   ranked within S, coarsened to `rank_bins` bins. Flatness: the χ²/bin
   deviation and the outer-bin mass. A U shape is under-dispersion, a hump is
   over-dispersion.
5. **Energy distance**, multivariate, on whole fluid fields (u, v, w jointly,
   L2 over fluid cells, RMS-normalised). Uses the unbiased U-statistic (expected
   0 when S and R come from the same distribution). Its split-half floor gives
   the scale.
6. **Per-geometry W1** of the marginal values at the fixed points, at matched
   sizes against the split-half floor. Deterministic: this answers the noisy
   W1 open issue.
7. **Physics on the ensemble:** the existing `generator_evaluation` profiles,
   spectra, Reynolds stresses, divergence and diversity, compared to R per
   geometry. This is now like-for-like, since every source has the same
   geometry and forcing.
8. **Latent spread (optional):** the per-token latent variance of G against the
   encoded R in R's PCA basis (`latent_spectrum`), separating generator from
   decoder error.

Aggregation: the mean over the 10 geometries, with a geometry-level bootstrap
CI. Model differences are paired per geometry (same R, same noise).

## Implementation

- `scripts/surrogate/evaluate_latent_generator_ensemble.py`: one model per run,
  config block `latent_generator_ensemble` in `configs/surrogate/eval.yaml`.
  The new metric math goes in `neural_surrogates.generator_evaluation`, with
  unit tests on synthetic Gaussians where the answer is known. Helpers shared
  with `evaluate_latent_generator.py` (`_generate`, `_ae_reconstruct`,
  latent moments) move to `scripts/utils/`.
- `scripts/surrogate/compare_latent_generator_ensembles.py`: reads every
  model's output and writes a cross-model table and figures. These are spread
  ratio per height, rank histograms, CRPS ratio and energy distance with
  floors, and paired differences relative to R3.
- `job_scripts/delftblue/surrogate_evaluate_latent_generator_ensemble.slurm`:
  a SLURM array over the model dirs (one A100 each), then the compare step.
- Test overlay and a test in `tests/scripts/`; docs in `docs/neural_surrogates.md`.

## Data analysis (alongside)

QC of the ensemble:

- member 0 against the reference frame;
- NaN/Inf, seed uniqueness, missing members;
- whether the members are independent.

What the spread looks like:

- spread magnitude per variable and height;
- where it sits spatially;
- seed spread against the temporal spread along the reference trajectory;
- the `_ic` and `_ic_inlet` pilots: does the IC seed alone already decorrelate
  the state after the spin-up?
- convergence of each metric's floor with ensemble size (8, 16, 32, 64): is 64
  enough to separate the models?
- the test geometries and parameters against the training corpus.

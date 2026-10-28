# Latent flow-matching generators: final note (2026-10)

The experiments with the latent flow-matching generator of spun-up initial
fields on `pyudales_realistic`, from the first full-width models to the seed-ensemble
evaluation of the compact ones. **Outcome:** `compact_h512` (run R3) is the
default and lives at `model_weights/best_latent_flow_matching_realistic`. It is
the best of the models tested. It is still under-dispersed and about 30x the
autoencoder floor in energy distance.

Method and code: [neural_surrogates.md](../neural_surrogates.md) §36–39a. The
superseded working notes are in `docs/archive/research/`
(`latent_generator_results_2026-10.md`, `latent_generator_seed_ensemble_2026-10.md`).
The plans are in `docs/plans/implemented/`: `neural_surrogates/07_latent_flow_matching_spinup.md`,
`latent_generator_compact_full_reach.md` and
`latent_generator_seed_ensemble_eval.md`.

## The model

- **Latent.** The frozen Tadpole-B autoencoder (`tadpole_ae_b_realistic`) gives
  tokens of D = 1536 channels (3 variables × 512) per 16³-cell block, about
  130–450 tokens per realistic sample.
- **Generator.** Conditional flow matching from N(0, I) to the latent. It is
  conditioned on the geometry (per token) and a 12-step history of inflow
  angle and speed (adaLN-Zero). The sampled latent is decoded by the AE.
- **R3 = `compact_h512`.** A 6-block transformer at width 512, 32.0M
  parameters, with the **Gaussian skip**. The exact flow of a per-token
  N(μ, Σ), fitted to the real latents, is added to the net's output. The net
  starts at zero, so the untrained model already has the right variance in
  every latent direction, and the net learns only the conditional correction.
- **Sampling.** Heun, 25 steps (50 velocity evaluations).

## Experiments

### 1. Full-width models (2026-09 to 10)

| Run | Net | Params | Best val |
|---|---|---|---|
| `latent_generator_b_realistic` | `input_film`, H = D | 232M | 1.051 |
| `latent_generator_b_realistic_restart` | same, restarted | 232M | 1.046 (never evaluated) |
| `latent_generator_b_realistic_adaln` | adaLN, H = D | 177M | 1.074 |
| `latent_generator_b_realistic_adaln_h176` | adaLN, H 176 + channel skip (PR #221) | 5.5M | 1.285 |

- The 177M adaLN model passed the profile, divergence and diversity checks.
  Its W1 was 0.093 against a bound of 0.016 (another machine, Euler).
- The first 232M `input_film` model failed badly in an early evaluation (W1
  0.21, profile RMSE 0.30).
- The 5.5M h176 model plateaus at val 1.29 from epoch 20: it is
  capacity-limited, not undertrained.

### 2. Why a narrow net fails (latent analysis)

- **The latent is high-dimensional.** The top 176 PCA components hold 37 % of
  the variance; 90 % needs 859 and 99 % needs 1140. About 400 directions have
  variance below 0.1.
- **The decoder uses about 1100–1300 directions.** Truncated to the top 176,
  the decode error is 7.9x the AE's. At 1280 directions it is 1.08x.
- **`w` lives in the low-variance directions.** Filling unreached directions
  with unit noise gives about 6x the real divergence.
- **A plain Gaussian generator fails.** Tokens drawn iid from N(μ, Σ) fail
  profile, W1 and divergence, because the Gaussian is unconditional. Yet on `w`
  its W1 (0.045) beats the 177M model's (0.086).
- **Consequence.** The output must reach all ~1300 directions with context,
  while the attention (the H² cost) can stay narrow. That led to the Gaussian
  skip and the two full-reach designs.

### 3. Compact nets (Snellius, 2026-10-24..26)

Setup: batch 32, lr 1e-4, 250 epochs (R5: 100), one MIG A100 slice each.

| Run | Config | Params | Best val | Snapshot W1 | Profile RMSE |
|---|---|---|---|---|---|
| R1 | `compact_wide_head` (H 320 + wide per-token head) | 27.6M | **1.115** | 0.053 | 0.056 |
| R2 | `compact_bottleneck` (D-wide stream, attention at 256) | 26.8M | 1.127 | 0.052 | 0.060 |
| R3 | `compact_h512` (H 512 + Gaussian skip) | 32.0M | 1.157 | 0.042 | **0.047** |
| R4 | R3 + `prediction: data` | 32.0M | 1.162 | 0.048 | 0.057 |
| R5a | R3 + `time_shift: 0.5` | 32.0M | 1.147* | 0.051 | 0.064 |
| R5b | R3 + `time_shift: 3` | 32.0M | 1.149* | **0.041** | 0.059 |

The snapshot W1 comes from `evaluate_latent_generator.py`: Heun 25, 16 seeds,
64 test snapshots, bound 0.031. \* `time_shift` also warps the validation
times, so these val losses are not comparable.

- **The compact nets beat the 177M model on W1 by 45–55 %**, at a sixth of its
  size. Every one still fails W1.
- **Val loss does not predict sample quality.** R1 has the best val loss, R3
  the best samples.
- **Heun 25 beats Euler 50** at equal cost: W1 0.053 → 0.042 on R3.
- **Open issue.** With Heun every model over-disperses the lowest-variance
  latent directions (components 1025–1536 at 2–3.4x).

### 4. Seed-ensemble evaluation (DelftBlue, 2026-10-27..28)

**Why.** The snapshot evaluation mixes geometries, parameter states and time.
Its W1 depends mostly on which snapshots are drawn: two random 64-snapshot
selections differ by about 0.07, more than the gaps between models.

**Data.** `pyudales_realistic_seed_ensemble` has 64 spun-up members for each of
the 10 held-out test geometries, at identical parameters; sample 4 lacks
member 59 and sample 6 lacks member 40. Only the IC and inlet seeds differ, and
the members are independent. They give the conditional distribution directly.

**Evaluation.**
- 64 generator draws per geometry, with the t = 0 history. The noise is the
  same for every model.
- Scores against the members:
  - fair CRPS over the leave-one-out solver CRPS;
  - energy distance (U-statistic);
  - spread ratio and spread-map correlation;
  - rank histograms;
  - debiased mean bias;
  - physics.
- The 95 % CIs come from a bootstrap over geometries. Paired model differences
  use the shared noise.

| Model | CRPS ratio | Energy dist. (rel.) | Spread ratio | Spread-map corr. | Profile RMSE | Spectra (dB) |
|---|---|---|---|---|---|---|
| **R3** | **1.046** | 0.048 | 0.933 | 0.711 | **0.038** | 1.32 |
| R4 | **1.046** | **0.048** | 0.937 | 0.711 | 0.053 | 1.34 |
| R1 | 1.049 | 0.052 | 0.918 | **0.723** | 0.049 | **1.23** |
| R5a | 1.050 | 0.053 | 0.937 | 0.664 | 0.053 | 1.38 |
| R5b | 1.057 | 0.060 | 0.942 | 0.633 | 0.054 | 1.99 |
| R2 | 1.063 | 0.070 | 0.963 | 0.582 | 0.057 | 1.42 |
| h176 | 1.082 | 0.084 | 0.999 | 0.437 | 0.074 | 3.07 |
| AE floor | 1.002 | 0.0016 | 0.983 | 0.996 | 0.008 | 1.10 |

**Model ranking.**
- R3 and R4 tie: R4 − R3 is not significant except for a slightly lower
  divergence. R3 is kept, as it is simpler and had the better snapshot samples.
- R1 is third: it has the best spectra and spread placement but is the most
  under-dispersed.
- The time shift hurts at both values.
- R2 and h176 add spread in the wrong places.

**Shortcomings shared by every model.**
- **Under-dispersed.** All but h176 have U-shaped rank histograms (outer mass
  0.29–0.30 vs 0.25) and spread ratios of 0.92–0.96. h176 has the right total
  spread (0.999, outer mass 0.27), but in the wrong places.
- **The spread drops at 64 m.** It falls to 0.85–0.9 at 62–66 m in u and v,
  in every model but not in the AE. That is the boundary between the two
  vertical token layers.
- **Unseen grid sizes are hardest.** For R1, R3, R4, R5a and R5b, test
  samples 3 and 5 are worst (energy distance 0.08–0.12 vs 0.03–0.05), and they
  are the only grid sizes without a training sample. R2 and h176 are worst on
  sample 7 (0.14–0.15).
- **The generator is the bottleneck, not the decoder.** It is about 30x the AE
  floor in energy distance.
- **The conditioning works.** Another geometry's parameter history raises the
  energy distance by about 3.4 in every model.

**About the data.**
- Member 0 does not reproduce the corpus' reference frame (RMS 0.02–0.72 m/s
  in u). The solver is deterministic, so some setting differs from the corpus
  run.
- The members' seeds follow an older derivation.
- Varying only the IC seed gives 15 % of the full ensemble's spread
  (pilot ensembles on sample 6). The inlet seed drives the rest.

## Open directions

1. **Under-dispersion.** Try `gaussian_eig_floor: 1e-3`, more sampling steps,
   and a look at the upper token layer.
2. **Grid generalisation.** Training data on more grid sizes, or test whether
   that is the bottleneck.
3. **The 232M restart model** has never been evaluated. Run it through the
   seed-ensemble evaluation for a like-for-like reference.
4. **The member-0 mismatch** in `generate_seed_ensemble.yaml` needs resolving
   before new seed ensembles are generated.
5. **Deployment.** `spinup_source: generative` still samples with Euler and has
   no default `model_dir`. Point it at `best_latent_flow_matching_realistic`
   and move it to Heun when the generator is used in assimilation.

## Where things are

- Weights: `/projects/urbanair/model_weights/` on DelftBlue. The job scripts
  on both machines use `best_latent_flow_matching_realistic`. Snellius'
  `/projects/prjs2304/urbanair/model_weights/lg_b_realistic_h512_l6` must be
  renamed to match.
- Seed-ensemble results: `<model dir>/eval_seed_ensemble/`. The comparison is in
  `model_weights/latent_generator_ensemble_comparison/` (`comparison.md`,
  `scores.png`, `spread_ratio_profiles.png`, `rank_histograms.png`,
  `paired_differences.png`).
- Snapshot results: `<model dir>/eval_heun25_s16/`.

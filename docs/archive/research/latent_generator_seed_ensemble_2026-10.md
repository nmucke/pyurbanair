# Latent generators against the seed ensemble (2026-10)

**Superseded** by [research/latent_generator_final_2026-10.md](../../research/latent_generator_final_2026-10.md).

The seven compact latent generators of
[latent_generator_results_2026-10.md](latent_generator_results_2026-10.md),
evaluated against the solver's own spread at fixed geometry and forcing. The
method is in [neural_surrogates.md](../../neural_surrogates.md) §39a, the plan in
[plans/implemented/latent_generator_seed_ensemble_eval.md](../../plans/implemented/latent_generator_seed_ensemble_eval.md).

## Setup

- Data: `pyudales_realistic_seed_ensemble`. The 10 held-out test geometries of
  `pyudales_realistic`, 64 spun-up members each (sample 4 lacks member 59,
  sample 6 lacks member 40). Only `irandom` and the inlet seed differ between
  members. Each member is the reference's first frame.
- Generators: 64 draws per geometry with the t = 0 history, Heun 25 steps. The
  noise depends only on (seed, geometry, draw), so every model gets the same
  noise. The frozen `tadpole_ae_b_realistic` round trip (AE) is the decoder
  floor.
- DelftBlue, one A100 per model: 45–105 min each (jobs 1105907, 1105908).
- Outputs: `<model dir>/eval_seed_ensemble/`. The comparison is in
  `model_weights/latent_generator_ensemble_comparison/` (`comparison.md`,
  `scores.png`, `spread_ratio_profiles.png`, `rank_histograms.png`,
  `paired_differences.png`).
- R3 now lives at `model_weights/best_latent_flow_matching_realistic`. It was
  `lg_b_realistic_h512_l6`; the summary.json written before the rename still
  records the old path.

## Data quality

- **No NaN/Inf.** Grids, masks, spin-up and parameters match the reference.
- **Independent members.** Their mean anomaly correlation is −1/(M−1), with no
  near-duplicates.
- **Member 0 does not reproduce the reference frame.** The fluid RMS
  difference in u is 0.02–0.72 m/s, which is 3–53 % of the distance between two
  members. The solver is deterministic (the pilot members are bitwise equal),
  so some setting differs from the corpus run, and the flow's chaos amplifies
  it. The reference frame is still a typical member.
- **The data predate the current seed derivation.** Member *m* has the same
  seeds in every sample. This is harmless: same-grid pairs are uncorrelated.
- **The inlet seed drives the spread.** The IC seed alone gives 15 % of it.
  Seed spread over temporal spread along the trajectory is 0.94 (u), 0.73 (v)
  and 1.00 (w); the inflow angle drifts along the trajectory.
- **Sampling floors are tiny.** Fields have 0.5–1.8 M fluid cells, so at
  64 vs 64 the null std of the 10-geometry mean is ~0.0015 for the spread ratio
  and ~0.0002 for the CRPS ratio. The differences between geometries, not
  sampling, decide significance.
- **The old snapshot W1 is selection-dominated.** Two random 64-snapshot
  selections differ by W1 ≈ 0.07, more than the 0.041–0.053 gaps between models.

## Results

The mean over the 10 geometries:

| Model | CRPS ratio | Energy dist. (rel.) | Spread ratio | Spread-map corr. | Profile RMSE | Spectra (dB) | Div RMS |
|---|---|---|---|---|---|---|---|
| R3 `compact_h512` | **1.046** | 0.048 | 0.933 | 0.711 | **0.038** | 1.32 | 0.036 |
| R4 R3 + `prediction: data` | **1.046** | **0.048** | 0.937 | 0.711 | 0.053 | 1.34 | 0.034 |
| R1 `compact_wide_head` | 1.049 | 0.052 | 0.918 | **0.723** | 0.049 | **1.23** | 0.035 |
| R5a R3 + `time_shift: 0.5` | 1.050 | 0.053 | 0.937 | 0.664 | 0.053 | 1.38 | 0.037 |
| R5b R3 + `time_shift: 3` | 1.057 | 0.060 | 0.942 | 0.633 | 0.054 | 1.99 | 0.048 |
| R2 `compact_bottleneck` | 1.063 | 0.070 | 0.963 | 0.582 | 0.057 | 1.42 | **0.032** |
| h176 (PR #221) | 1.082 | 0.084 | 0.999 | 0.437 | 0.074 | 3.07 | 0.073 |
| AE (floor) | 1.002 | 0.0016 | 0.983 | 0.996 | 0.008 | 1.10 | 0.020 |

The CRPS ratio is fair CRPS over the leave-one-out solver CRPS. The energy
distance is relative to the mean distance between members. 95 % geometry
bootstrap CIs and the paired differences are in `comparison.md`.

## Findings

- **R3 and R4 tie for best.** R4 − R3 is not significant on any score except a
  slightly lower divergence. R3 stays the default: it gave the better snapshot
  samples and is the plain objective.
- **R1 is third.** It is significantly worse than R3 on CRPS (+0.003) and
  energy distance (+0.0035). It has the best spectra and spread placement, and
  it is the most under-dispersed.
- **The time shift hurts at both values; R2 and h176 are worst.** For R2 and
  h176, more spread comes with worse placement. h176's spread ratio of 1.0 is
  misplaced variance (map corr. 0.44), up to 1.3x near the ground and 1.8x in w
  at the top.
- **The models are under-dispersed.** The rank histograms are U-shaped (outer
  mass 0.29–0.30 vs 0.25), and the spread ratios are 0.92–0.96 (all but h176).
- **The gap to the decoder floor is about 30x** in energy distance. The
  generator, not the AE, limits sample quality.
- **Samples 3 and 5 are hardest for R1, R3–R5b** (energy distance 0.08–0.12
  vs 0.03–0.05). They are the only test geometries whose grid size has no
  training sample.
- **The spread depends on the domain.** It is under-dispersed on the large
  grids (samples 2–5: 0.83–0.92), and matched to over-dispersed on the
  128×128 grids (samples 6, 8: 0.94–1.19).
- **The spread drops at 62–66 m in u and v**, to 0.85–0.9 of the solver's, in
  every model but not in the AE. 64 m is the boundary between the two vertical
  token layers (16 cells × 4 m), so the generator under-disperses the upper
  token layer.
- **All models respond to the parameters.** Another geometry's history raises
  the energy distance by ~3.4. h176 responds slightly less.

## Next

1. Find out why the generators under-disperse: the latent tail
   (`gaussian_eig_floor`), the sampler steps, or the upper token layer.
2. Add training data on the grid sizes of test samples 3 and 5, or test
   whether grid-size generalisation is the bottleneck.
3. Find the setting behind the member-0 mismatch before reusing
   `generate_seed_ensemble.yaml` for new corpora.

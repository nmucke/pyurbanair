# Latent generator results (2026-10)

**Superseded** by [research/latent_generator_final_2026-10.md](../../research/latent_generator_final_2026-10.md).

Status of the flow-matching latent generator of initial fields on
`pyudales_realistic` (B autoencoder, D = 1536 latent channels per token). The
design and the full evidence are in
[plans/latent_generator_compact_full_reach.md](../../plans/implemented/latent_generator_compact_full_reach.md);
the configs in `configs/latent_generator_architecture/`
([neural_surrogates.md](../../neural_surrogates.md) §38).

## Setup

- All compact runs: Snellius, one MIG A100 slice each (~172 s per epoch),
  batch 32, lr 1e-4, 250 epochs (R5: 100), from a copy of the DelftBlue latent
  cache with the latent second moment added.
- Evaluation: `evaluate_latent_generator.py` with Heun at 25 steps, 16 noise
  seeds, 64 test snapshots. W1 bound 0.031 (2 x the AE's own W1).
- Run dirs: `/projects/prjs2304/urbanair/model_weights/lg_b_realistic_*`
  (also copied to DelftBlue).

## Results

| Run | Config | Params | Best val | W1 (u / v / w) | Profile RMSE | Div RMS | Spectra (dB) |
|---|---|---|---|---|---|---|---|
| R1 | `compact_wide_head` | 27.6M | **1.115** | 0.053 (.075/.048/.036) | 0.056 | 0.035 | **1.20** |
| R2 | `compact_bottleneck` | 26.8M | 1.127 | 0.052 (.075/.048/.032) | 0.060 | **0.032** | 1.33 |
| R3 | `compact_h512` (default) | 32.0M | 1.157 | 0.042 (.050/.049/.025) | **0.047** | 0.035 | 1.24 |
| R4 | R3 + `prediction: data` | 32.0M | 1.162 | 0.048 (.074/.044/.025) | 0.057 | 0.034 | 1.27 |
| R5a | R3 + `time_shift: 0.5` | 32.0M | 1.147* | 0.051 (.068/.049/.037) | 0.064 | 0.036 | 1.30 |
| R5b | R3 + `time_shift: 3` | 32.0M | 1.149* | **0.041** (.053/.047/.023) | 0.059 | 0.047 | 1.83 |
| ref. | AE reconstruction | – | – | 0.016 | 0.007 | 0.019 | 1.03 |
| ref. | 177M full-width adaLN† | 177M | 1.074 | 0.093 (.104/.089/.086) | 0.062 | 0.024 | – |

\* `time_shift` also warps the validation flow times: not comparable.
† Stored evaluation from another machine (Euler, batch 40); approximate.

## Findings

- **Every model fails only the W1 check**; profile, divergence and diversity
  (0.77–0.80) pass. The compact nets beat the 177M model on W1 by ~45–55% at
  a sixth of its size, but its val loss is still 4–8% lower.
- **R3 (`compact_h512`) gives the best samples** and is the config default.
  R1 has the best val loss and spectra, R2 the best divergence and Reynolds
  stresses. Val loss does not predict sample quality here.
- **Clean-sample prediction (R4) does not help**; nor does the time shift:
  α = 3 adds high-wavenumber energy and divergence, α = 0.5 is worse on W1.
- **Heun 25 steps beats Euler 50** at equal cost (R1–R3: W1 0.064 → 0.053,
  0.059 → 0.052, 0.053 → 0.042). Use Heun.
- **Spectra are close to the AE's limit** (0.2–0.3 dB above it with Heun);
  the AE itself loses energy at k ≈ 0.1–0.4 rad/m. The halo and global decodes
  show tile-seam spikes (~9 dB) for the AE too: a decoder property.
- **Open issues:**
  - With Heun, every model over-disperses the lowest-variance latent
    directions (components 1025–1536 at 2.0–3.4x the real variance).
  - The W1 bound is noisy: it uses a 200k-value random subsample, and the AE's
    own W1 came out 0.008 vs 0.016 in two evaluation setups.

## Next

1. R3 with `gaussian_eig_floor: 1.0e-3` against the latent-tail
   over-dispersion.
2. A deterministic W1 (all values, or one fixed subsample shared by every
   source) before W1 decides between models.
3. Re-evaluate the 177M and 232M full-width models with the same Heun-25
   setup for a like-for-like comparison.

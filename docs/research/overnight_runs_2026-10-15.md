# Overnight runs, 14–15 October 2026

Runs launched after the cross-model parity work (PRs #201, #202), on the
`xie_and_castro` case with the current defaults. They test the first directions
of `sgs_correction_assessment_2026-10.md` §6. Outputs are in
`/export/scratch1/ntm/overnight_2026-10-14/` (local, not in git).

## 1. Round-2 rerun: uDALES assimilating a PALM truth

The setup is Round 2's: 3 × 180 s windows, 32 members, 4 ESMDA steps, 180 s
window-mean observations at z = 2, 8 and 14 m, and the persistent height-banded
representation error. Both PALM and uDALES now run with the corrected defaults.
Values are the mean over 3 seeds; Round 2 is in brackets.

| | T2: inflow only | TC: inflow + SGS constant |
|---|---|---|
| Desroziers ratio (1 = calibrated) | 3.01 (3.57) | 2.76 (3.59) |
| Validation spread/skill (1 = calibrated) | 0.54 (0.39) | 0.53 (0.37) |
| Validation CRPS, window-mean \|U\| (m/s) | 0.27 (0.45) | 0.25 (0.45) |
| Replica CRPS: the noise floor (m/s) | 0.086 (0.18) | 0.086 (0.18) |
| Posterior inflow angle (true 10°) | 6.8° | 7.4° |
| Posterior speed (true 6.0 m/s) | 6.24 m/s | 6.14 m/s |
| Posterior SGS constant (prior 0.12 ± 0.02) | — | 0.22 ± 0.01 |

- **Held-out error dropped by about 40 %,** because the artefacts in the truth
  and in the model are gone. Most of the remaining error is still model
  difference: the CRPS is about 3× the noise floor.
- **Still miscalibrated.** The Desroziers ratio of about 3 means the
  observation error is underestimated about threefold.
- **Parameters compensate for structural error.** The inflow angle is biased by
  about 3°. TC drives the SGS constant 5 prior standard deviations up, to about
  its old value of 0.24. That buys a 9 % lower CRPS, consistent across seeds,
  but the parameter is acting as a compensator, not as physics.

## 2. Forward comparisons, without DA (`compare_models.sh`)

Ratio = model-pair distance ÷ the models' own replica-to-replica noise; about 1
means the models differ by noise only. These are turbulent-inflow runs unless
noted.

| Run | Mean u, all | Mean u, street level | Sensor means | TKE profile |
|---|---|---|---|---|
| uDALES vs PALM, 2 m (4 replicas, 600 s) | 4.3 | 7.9 (uDALES +0.27 m/s) | 6.0 (0.43 m/s) | 1.3 |
| uDALES vs PALM, periodic, 2 m (4 replicas, 600 s) | 1.4 | 1.9 | 1.5 | 1.1 |
| **uDALES vs PALM, 1 m** (2 replicas, 300 s) | 2.3 | 2.7 (bias +0.03 m/s) | 3.1 (0.24 m/s) | 1.9 |
| uDALES vs pylbm, 2 m (2 replicas, 300 s) | 5.4 | 10 (uDALES +0.62 m/s) | 23 | 9.9 |

- **The near-ground gap is mainly a 2 m resolution effect.** At 1 m, uDALES'
  street-level bias against PALM vanishes, from +0.27 to +0.03 m/s, and the
  sensor-mean distance halves.
- **Periodic runs agree within noise**, at 2 m as well.
- **pylbm is a "far" truth** (no wall model), as expected. Its two replicas
  came out identical because `configs/model_comparison.yaml` gives pylbm no
  replica seed, so its own noise floor is missing (to fix).

**Inflow-turbulence sensitivity** (uDALES at intensity 0.10, 0.15 and 0.20):
- mean flow and sensor means don't respond: ratios 0.7–0.9;
- TKE responds strongly: field ratios 3–6, profile ratios 6–13.

The intensity can therefore be estimated only from turbulence observations,
such as window variances, not from window means.

## 3. Next step: roughness estimation

The question: **can estimating roughness (z0) compensate for the 2 m resolution
error?** z0 is a real-world parameter, and the gap sits next to walls and the
ground, where z0 acts. The proposed order:

1. **Add a z0 knob.** z0 is fixed today in the uDALES facet types
   (`factypes.inp`) and the PALM `_p3d` (`roughness_length`). Expose it,
   per surface type (ground, walls, roofs), through the shared `flow` block.
2. **Sensitivity at 2 m.** Sweep z0 in uDALES and measure how much the sensors
   move compared with the noise floor. Does any z0 bring the street-level means
   of the 2 m model onto the 1 m model's?
3. **Isolate the resolution error first.** Use the same model as the truth,
   uDALES at 1 m, with uDALES at 2 m as the DA model and z0 as the estimated
   parameter. This shows whether z0 can absorb pure resolution error before any
   model difference enters.
4. **Cross-model.** Add a TZ arm (inflow + z0) to the Round-2 setup, against
   the PALM truth, and compare it with T2 and TC on CRPS, Desroziers ratio and
   whether the inflow bias shrinks.

The risk to watch: a fitted z0 that removes the bias may just be another
compensator, like the SGS constant. Step 3 tests that directly: z0 must stay
physically plausible and must transfer to other inflow angles.

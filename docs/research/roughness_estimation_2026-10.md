# Estimating wall roughness (z0) in the DA

October 2026 · Follows `sgs_correction_assessment_2026-10.md` §6 (directions 3
and 4). Code: PR #208 (`roughness_length` for uDALES and PALM, new sensor
layout) and branch `feat/bounded-param-transform` (log and bounded parameter
transforms). Runs: `/export/scratch2/ntm/z0_experiments/` (local, not in git).

**Result.** Both twins recover z0 and are calibrated. Cross-model (PALM truth,
uDALES DA model), z0 settles at about 1.7× the truth. It halves the
validation-sensor error of the window-mean wind, but leaves the field-wide wind
and turbulence errors unchanged. At 2 m resolution z0 acts as a sensor-level
compensator for structural error, mostly at street level, not as a physical
estimate.

## Setup

- **Case:** `xie_and_castro`, 2 m grid.
  - 18 assimilation sensors (z = 2, 8, 20 m) and 10 held-out validation
    sensors (z = 2 to 26 m), spread over the whole array.
  - The truth is one continuous run at 10°, 6.0 m/s and z0 = 0.1 m, plus a
    replica with another seed.
- **DA:** parameter-only ESMDA, 32 members, 4 steps, 3 windows of 180 s,
  180 s window-mean observations of u and v.
  - Estimated parameters: inflow angle, speed and z0.
  - z0 prior: log-normal with median 0.05 m, log-std 0.7, bounded to
    0.005–0.3 m. The DA updates a bounded transform of log z0, so no member
    leaves the range.
- **Observation error:**
  - instrument error 0.25 m/s, plus a persistent height-banded
    representation error;
  - twins: the truth model's own realisation noise (0.05–0.09 m/s);
  - cross-model: inflated to 0.53 / 0.43 / 0.33 m/s at 2 / 8 / 20 m. The
    inflation comes from the Desroziers statistics of a first inflow-only run;
    it is a data-only estimate, also available with real observations.

## Results (final-window posteriors; 3 seeds unless noted)

| Experiment | z0 (m), truth 0.1 | Angle (°), truth 10 | Speed (m/s), truth 6.0 | Desroziers | Validation CRPS, mean \|U\| (replica) | Field RMSE u | Field TKE RMSE (replica) | Validation CRPS, variance (replica) |
|---|---|---|---|---|---|---|---|---|
| uDALES twin | 0.088–0.096 | 9.8–10.2 | 5.98–6.02 | 0.7–1.0 | 0.05 (0.089) | 0.10–0.12 | 0.040 (0.052) | 0.020 (0.031) |
| PALM twin | 0.095–0.107 | 9.4–9.7 | 5.97–5.99 | 0.73–0.78 | 0.04 (0.093) | 0.10 | 0.034 (0.049) | 0.016–0.017 (0.039) |
| Cross-model, inflow only | — | 9.0–9.6 | 6.01–6.10 | 0.99 | 0.17–0.18 (0.093) | 0.43–0.45 | 0.078–0.081 (0.047) | 0.019–0.020 (0.039) |
| Cross-model, inflow + z0 (2 seeds) | 0.17 ± 0.035 | 8.8–9.2 | 5.99–6.03 | 0.88–0.89 | 0.09–0.10 (0.093) | 0.42–0.44 | 0.077 (0.047) | 0.022–0.024 (0.039) |

Units: m/s for wind, m²/s² for TKE and variance. The posterior z0 spread is
about ±0.005 m in the twins. "Replica" is one other realisation of the truth
model scored the same way.

**Error at the assimilated sensors, by height** (ensemble-mean RMSE of the
window-mean u and v against the noise-free truth, mean over seeds, m/s):

| Height | Inflow only | Inflow + z0 | Change |
|---|---|---|---|
| 2 m (street) | 0.56 | 0.48 | −14 % |
| 8 m (mid-canopy) | 0.48 | 0.43 | −10 % |
| 20 m (above the roofs) | 0.21 | 0.20 | −3 % |

In both twins the posterior error is 0.06–0.09 m/s at every height. The
held-out sensors can't be broken down by height: the window states of the
z0 arm's runs were deleted after scoring.

## What this shows

1. **z0 is identifiable when the model is right.** Both twins recover it to
   within about 0.01 m, with the correct inflow, and are calibrated
   (Desroziers ≈ 1, spread/skill ≈ 1). The PALM twin's angle is 0.3–0.6° low,
   possibly from PALM zeroing w at every warm start.
2. **The error model decides whether cross-model DA is usable.** With only the
   realisation noise in the observation error (Desroziers ≈ 5.2), the first
   update pushed z0 to 1–3 m and uDALES crashed. With the data-derived inflated
   error, the run is calibrated (≈ 0.9–1.0) and stable.
3. **Cross-model, z0 compensates, mostly at street level.**
   - It doubles the inflow-only run's skill at the held-out sensors, down to
     the noise floor.
   - Its gain at the assimilated sensors is largest at 2 m and vanishes above
     the roofs.
   - But the field-wide wind and TKE errors don't change, and the angle drifts
     slightly further from the truth.
   - z0 at 1.7× the truth is consistent with the earlier finding that uDALES
     at 2 m runs too fast at street level against PALM, a resolution effect
     (`overnight_runs_2026-10-15.md`): extra roughness buys the missing drag
     near the sensors.
4. **Turbulence is not helped.** Field TKE error stays at about 1.7× the
   replica level with or without z0, and the validation variance error gets
   slightly worse with z0.

## Caveats

- The cross-model z0 arm has only 2 seeds. Seed 1 failed in window 1, when one
  member's time step collapsed (diffusion-limited) under
  `failure.policy=raise`.
- The inflated error comes from the same experiment's inflow-only run.
  With real data it would come from earlier windows or a separate period.
- One truth realisation per model, so seed spread understates truth
  variability.

## Next steps

- **Rerun the failed seed** with `failure.policy=resample_from_successes` and
  keep its states, so the held-out sensors can be broken down by height.
- **Test transfer.** Estimate z0 at 10°, then forecast another inflow angle
  with that z0. A physical z0 should transfer; a compensator may not.
- **The 1 m test** (§6 direction 3 of the assessment note): uDALES at 1 m as
  truth, uDALES at 2 m as the DA model. This isolates how much of z0's 1.7× is
  pure resolution error.
- **Per-surface z0** (ground vs walls vs roofs) only if a single z0 proves
  physical. uDALES supports it; PALM would need its urban surface model.

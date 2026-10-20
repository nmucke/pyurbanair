# Estimating wall roughness (z0) in the DA

October 2026 · Follows `sgs_correction_assessment_2026-10.md` §6 (directions 3
and 4). Code: PR #208 (`roughness_length` for uDALES and PALM, new sensor
layout) and PR #211 (log and bounded parameter
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

In both twins the posterior error is 0.06–0.09 m/s at every height.

**Error at the held-out sensors, by height** (seed 1, whose states were kept;
CRPS of the window-mean |U|, m/s; the "Replica" column is the noise floor):

| Height | Inflow only | Inflow + z0 | Replica |
|---|---|---|---|
| 2 m | 0.31 | 0.085 | 0.056 |
| 5 m | 0.20 | 0.071 | 0.046 |
| 8 m | 0.069 | 0.090 | 0.19 |
| 14 m | 0.098 | 0.11 | 0.11 |
| 26 m | 0.088 | 0.076 | 0.099 |

The held-out gain is almost entirely below 5 m. Seed 1, rerun with failed
members replaced, matches seeds 2 and 3: z0 0.166 ± 0.035, Desroziers 0.88,
validation CRPS 0.095, and field u and TKE errors unchanged.

## Transfer to another inflow angle

Forward runs only. uDALES runs at the true inflow (6.0 m/s) with z0 = 0.05
(prior median), 0.1 (the truth) and 0.17 (the cross-model estimate), 2
replicas each. Each is scored against a PALM truth (z0 0.1) at 10°, where z0
was estimated, and at 40°. Ensemble-mean RMSE in m/s:

| Score | 10°: z0 0.05 / 0.1 / 0.17 | 40°: z0 0.05 / 0.1 / 0.17 | PALM noise (10° / 40°) |
|---|---|---|---|
| Assimilation sensors | 0.42 / 0.40 / **0.37** | 0.75 / 0.77 / 0.74 | 0.11 / 0.18 |
| … at 2 m | 0.47 / 0.45 / **0.41** | 0.53 / 0.58 / 0.54 | 0.10 / 0.19 |
| Validation sensors | 0.35 / 0.31 / **0.27** | 0.36 / 0.33 / 0.34 | 0.11 / 0.13 |
| … at 2 m | 0.33 / 0.29 / **0.18** | 0.41 / 0.35 / 0.38 | 0.09 / 0.11 |
| Old layout, at 2 m | 0.41 / 0.33 / **0.24** | 0.59 / 0.63 / 0.68 | 0.15 / 0.29 |
| Field u | 0.43 / 0.41 / 0.43 | 0.46 / 0.47 / 0.47 | 0.09 / 0.13 |

- **At 10°** z0 = 0.17 is best at the sensors, most of all at street level.
  The true z0 of 0.1 is not.
- **At 40° the advantage is gone.** The three values are within about 0.03 m/s
  of each other overall, and at street level 0.17 is no better and sometimes
  the worst (old layout, 2 m).
- **Field u doesn't depend on z0 at either angle.**
- At 40° every uDALES run is about twice as far from PALM as at 10°.

So the 0.17 estimate is a 10°-specific fit, not a transferable correction.

**TKE outliers are an outflow artefact.** Two transfer runs had high field-TKE
errors (10°, z0 0.1, replica 0: 0.31; 40°, z0 0.05: 0.15–0.20 m²/s²). Both are
concentrated at x = 73–79 m, just upstream of the 20 m outflow buffer, near
the ground. The worst one has |U| bursts up to 29 m/s over 43 frames, i.e.
backflow instabilities that get through uDALES' 20 m buffer at c_vreman 0.12.
They are not a z0 effect; excluding them, field TKE is the same for all z0.

## What this shows

1. **z0 is identifiable when the model is right.** Both twins recover it to
   within about 0.01 m, with the correct inflow, and are calibrated
   (Desroziers ≈ 1, spread/skill ≈ 1). The PALM twin's angle is 0.3–0.6° low,
   possibly from PALM zeroing w at every warm start.
2. **The error model decides whether cross-model DA is usable.** With only the
   realisation noise in the observation error (Desroziers ≈ 5.2), the first
   update pushed z0 to 1–3 m and uDALES crashed. With the data-derived inflated
   error, the run is calibrated (≈ 0.9–1.0) and stable.
3. **Cross-model, z0 compensates, mostly at street level, and doesn't
   transfer.**
   - It doubles the inflow-only run's skill at the held-out sensors, down to
     the noise floor.
   - Both at the assimilated and the held-out sensors, the gain is almost all
     below 5 m.
   - The fitted 0.17 helps at the inflow angle it was estimated at (10°), but
     not at 40°.
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

- Cross-model seed 1 first failed when one member diverged in window 1 under
  `failure.policy=raise`. It was rerun with `resample_from_successes`.
- The 20 m outflow buffer doesn't always stop uDALES' outflow backflow at
  c_vreman 0.12 (transfer TKE outliers). A 40 m buffer, which stabilised a case
  the 20 m one didn't in the SGS study, is worth checking as the default.
- The inflated error comes from the same experiment's inflow-only run.
  With real data it would come from earlier windows or a separate period.
- One truth realisation per model, so seed spread understates truth
  variability.

## Next steps

- **The 1 m test** (running) (§6 direction 3 of the assessment note): uDALES at 1 m as
  truth, uDALES at 2 m as the DA model. This isolates how much of z0's 1.7× is
  pure resolution error.
- **Per-surface z0** (ground vs walls vs roofs) only if a single z0 proves
  physical. uDALES supports it; PALM would need its urban surface model.

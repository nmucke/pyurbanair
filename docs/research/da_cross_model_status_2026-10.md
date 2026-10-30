# Cross-model data assimilation: status and next experiments

Status as of 30 October 2026. This is the single entry point for the
data-assimilation / cross-model (uDALES vs PALM) / model-discrepancy line of
work. It replaces the October notes and the September review, now in
[docs/archive/](../archive/) (map at the end). The z0 study's detailed results
stay in [roughness_estimation_2026-10.md](roughness_estimation_2026-10.md).
Newest results (21–30 Oct) come from the session that ran them; several aren't
in a repo note yet and are marked *(session)*.

## 1. Aim and constraints

- **The end goal is DA against real observations.** PALM is only a truth from
  a different source than the DA model (uDALES). Neither the true state nor
  the true parameters will be known.
- **Don't learn or fit the model–truth difference**: no learned SGS
  correction, no bias or additive forcing fields built from the PALM–uDALES
  difference. Methods must work with sensor data alone.
- **What the cross-model work is for:** finding *which parameters* to
  estimate so one simulation gets closer to another, and *what must go into
  the observation error*.
- **No case-specific model alignment** (e.g. snapping the STL to the grid):
  real geometries won't align with the grid.
- **Resolution matters, fetch length doesn't** (the upstream fetch is not a
  question to pursue).

## 2. Standard setup

`xie_and_castro`, 2 m grid, `inflow_outflow`, digital-filter inlet turbulence,
40 m outflow buffer (default since #218). Truth: one continuous PALM run at
10°, 6.0 m/s, z0 0.1 m, plus a replica (other seed) as the noise floor. DA:
parameter-only ESMDA smoother, 32 members, 4 steps, 180 s windows with 180 s
window-mean u, v observations at z = 2, 8, 20 m (18 sensors), 10 held-out
validation sensors (2–26 m). z0 prior log-normal (median 0.05, log-std 0.7),
bounded 0.005–0.3 m, updated in bounded log space (#211). Detail:
[roughness note, Setup](roughness_estimation_2026-10.md#setup).

## 3. What has been established

### Model parity and the remaining gap

1. **The Round-2 PALM truth carried artefacts, now fixed** (#201, #202):
   pw-scheme streaks (now ws-scheme), outflow damping over the array's wake
   (now an empty outflow buffer), inflow kicks inside the array (now the
   shared digital-filter generator), uDALES c_vreman 0.24 (now 0.12; 0.24 hid
   outflow backflow), PALM warm-start inflow zeroing (fixed). Table:
   [sgs_correction_assessment §5](../archive/research/sgs_correction_assessment_2026-10.md#5-what-the-parity-work-changed-pr-201-pr-202).
2. **Periodic uDALES and PALM agree within realisation noise** (distance /
   noise ratios 1.1–1.4). The gap is in the mean flow of inflow_outflow runs,
   near the ground: uDALES +0.24–0.27 m/s at street level, −0.12 m/s above
   the canopy; sensor-mean distance about 0.43 m/s (6× noise). [overnight §2](../archive/research/overnight_runs_2026-10-15.md#2-forward-comparisons-without-da-compare_modelssh)
3. **At 1 m the street-level bias vanishes** (+0.27 → +0.03 m/s, sensor
   distance halves). But a 2 m DA model's resolved TKE stays about 3× the
   replica level from a 1 m truth whatever z0: a resolution limit no
   parameter fixes. [overnight §2](../archive/research/overnight_runs_2026-10-15.md),
   [roughness, Resolution only](roughness_estimation_2026-10.md#resolution-only-udales-truth-at-1-m-udales-da-model-at-2-m)
4. **The 20 m outflow buffer was not enough**: backflow bursts reached the
   bounds in 3 of 40 uDALES runs and inflated street-level TKE 2–5× at 40°;
   40 m and 60 m agree within noise, so 40 m is the default (nx 50 → 70,
   `ncpu` must divide 70; PALM 1/5/7). Runs before #218 used 20 m. [docs/pyudales.md §7](../pyudales.md#7-instability-watchdog)
5. **The wall-law distance is the located cause of the near-wall gap.** Both
   codes put the first fluid point 1 m from a blocked face, but uDALES
   evaluates its log law at the true facet distance, 2 m for every cube wall
   in this case (walls sit mid-cell). On the same flow the uDALES law gives
   PALM's ground stress, 0.65× its roof stress and 0.47× its wall stress.
   uDALES on a grid-snapped STL matches PALM's canopy mean u at 1 m (0.85 vs
   0.94 m/s; real STL 1.41). Exact wall mapping: z0_uDALES = 2 z0_PALM.
   After matching walls, a street-level residual of opposite sign remains
   (lane u 0.16 vs 0.35 m/s, TKE too high): SGS or numerics, not z0.
   [roughness, Follow-up](roughness_estimation_2026-10.md#follow-up-2123-oct-seed-1-unless-noted-several-runs-lost-to-oom-and-a-crash);
   [docs/pypalm.md "Wall treatment"](../pypalm.md). The snapped STL was a
   diagnostic only (see §1).
6. **Inflow turbulence intensity shows only in turbulence statistics**: 0.10
   → 0.20 leaves means and sensor means within noise (ratios 0.7–0.9) but
   moves TKE strongly (ratios 3–13). Estimating it needs variance-type
   observations. [overnight §2](../archive/research/overnight_runs_2026-10-15.md)

### SGS (the earlier study, closed)

7. **Learned / parametrised SGS correction is a dead end for cross-model DA.**
   Round 1 (uDALES truth): the 3-coefficient correction is recovered only with
   inlet turbulence off (held-out errors −40 to −90 %); with turbulent inflow
   nothing is recovered. Round 2 (PALM truth): no value, coefficients flip
   sign or saturate. The mean flow barely responds to the SGS term, and the
   error sits where the correction doesn't act (walls, street level).
   [sgs_correction_assessment §1–3](../archive/research/sgs_correction_assessment_2026-10.md),
   [SGS twin report](../archive/research/sgs_twin_tests_2026-10/report.md).
8. **A single SGS constant is a compensator.** Round-2 rerun with fixed
   defaults: TC drives c_vreman from 0.12 to 0.22 ± 0.01 (5 prior std), for a
   consistent but small gain (~9 % CRPS here, ~4 % in Round 2).
   [overnight §1](../archive/research/overnight_runs_2026-10-15.md#1-round-2-rerun-udales-assimilating-a-palm-truth)
9. **Observation design lessons from Round 1b:** with turbulent inflow the
   realisation noise (≈ 0.15 m/s per 20 s mean) must be in R, and the
   number of observations must be small compared with members: 180 s window
   means work, 20 s bins and raw frames collapse the ensemble.
   [report, Round 1b verdict](../archive/research/sgs_twin_tests_2026-10/report.md#round-1b-verdict)

### Roughness (z0) and the observation error

10. **z0 is identifiable when the model is right.** uDALES twin, PALM twin and
    uDALES 1 m truth → 2 m DA all recover z0 ≈ 0.1 (0.088–0.113 across arms
    and seeds; the facts sheet rounds this to 0.09–0.11), with Desroziers ≈ 1.
    Grid resolution doesn't bias z0. [roughness, Results](roughness_estimation_2026-10.md#results-final-window-posteriors-3-seeds-unless-noted)
11. **Cross-model, z0 is a compensator, not a site property.** PALM truth (z0
    0.1) → uDALES: 0.17 [0.11, 0.22] at 10°, 0.021 [0.011, 0.043] at 40°
    (seed 1, inflated R per regime): an 8× swing, a sensor-weighted mix of
    the ground (1×) and wall (~2.5×) mismatch. At 10° it halves the held-out
    sensor error (almost all below 5 m) but leaves field u and TKE errors
    unchanged, and the 10° fit doesn't help at 40°. A forward sweep puts the
    best uDALES z0 at 0.15–0.17, the DA estimate.
12. **The observation error decides whether cross-model DA is usable.** With
    realisation-noise-only R (Desroziers ≈ 5): z0 runs to the 0.3 bound with
    collapsed spread and the angle is pulled to 8.3° (truth 10); in an earlier
    run the first update pushed z0 to 1–3 m and uDALES crashed. With a
    Desroziers-inflated R (about 5× noise: 0.53 / 0.45 / 0.35 m/s at 2 / 8 /
    20 m) runs are calibrated and stable.
13. **Adaptive observation error, complete study** (24 Oct, on dev with the
    40 m buffer and `save_states=prior`; PALM truth 10°, 6.0 m/s, z0 0.1 →
    uDALES; 6 windows, 2 seeds per arm; `runs_adaptive_r2`). Final-window
    posterior median [5–95 %]:

    | Arm | Angle (°) | Speed (m/s) | z0 (m) | Inflation 2 / 8 / 20 m |
    |---|---|---|---|---|
    | A: noise-only R, inflow + z0 | 7.9 [7.7, 8.1] / 8.2 [8.0, 8.3] | 5.91 / 6.11 | 0.29 / 0.30 (at the 0.3 bound, no spread) | – |
    | B: adaptive R, inflow + z0 | 8.1 [7.4, 9.3] / 9.0 [7.3, 10.1] | 6.04 / 6.02 | 0.20 [0.16, 0.22] / 0.19 [0.15, 0.22] | ×4.8 / 4.7 / 4.2–4.7 |
    | C: adaptive R, inflow only | 9.7 [8.8, 10.7] / 10.1 [8.6, 11.2] | 6.22 / 6.06 | – | ×5.5 / 5.1–5.4 / 4.0–5.2 |

    - The adaptive inflation is reproducible: ×4–6 the realisation noise in all
      runs, matching the offline calibration (item 12).
    - With the adaptive error, z0 stays off the bound with real spread, but
      at about 2× the truth, and it still pulls the angle 1–2° low. The inflow
      is recovered best without z0 (C).
    - Held-out sensors, all 10 (CRPS of window-mean u; replica 0.096): A
      0.21–0.22, **B 0.15–0.17**, C 0.21–0.24. B is best on every sensor
      score in both seeds.
    - By height (seed 1; CRPS of window-mean |U|; replica 0.063 / 0.079 /
      0.046 / 0.129 / 0.083 at 2 / 5 / 8 / 14 / 26 m): A 0.26 / 0.09 / 0.08 /
      0.34 / 0.13, **B 0.12 / 0.05 / 0.07** / 0.17 / 0.13, C 0.31 / 0.27 / 0.07
      / **0.14** / 0.13. The z0 gain is at street level; without z0 the 2–5 m
      wind is too fast; z0 pinned at the bound (A) overcorrects at 14 m; B is
      still ~2× the noise at 2 m (the structural residual of item 5).
    - All arms are overconfident at the held-out sensors (z-score std 3.5–4.9,
      expected ~1): the adaptive error widens the uncertainty of the assimilated
      observations, but a parameter-only ensemble has too little spread at
      unobserved locations.
14. **Summary verdict:** inflow angle and speed are robustly estimable
    cross-model. A single z0 buys street-level accuracy at the cost of a biased
    inflow angle, and only with the adaptive error does it not hurt higher
    sensors; it is not a transferable parameter. Street-level structural error
    belongs in the observation error.
15. **Separate wall z0 is possible only in uDALES.** uDALES takes z0 per
    facet type (`factypes.inp`). Since #227 pyudales takes either
    `roughness_length` (all facets) or `roughness_length_wall` (vertical
    facets, |n_z| < 0.5, retyped to their own facet type) +
    `roughness_length_ground` (ground and roofs) ([pyudales.md](../pyudales.md)).
    PALM without USM/LSM has one z0 for all surfaces and rejects the split;
    only the DA model (uDALES) needs it.
16. **Wall/ground z0 study: the pair doesn't transfer either** *(session,
    30 Oct; `runs_wallz0`)*. Same setup as item 13 (adaptive error, 6
    windows, 2 seeds), arms S = inflow + single z0, W = inflow + wall z0 +
    ground z0 (prior medians 0.05, bounds 0.005–0.6 wall, 0.005–0.3 ground).
    PALM truths at 10° and 40° (6.0 m/s, z0 0.1); the 40° prior angle is
    centred on truth − 10° like the 10° runs. Final-window posterior mean ± std:

    | Run | Angle (°) | Speed (m/s) | z0 single | z0 wall | z0 ground |
    |---|---|---|---|---|---|
    | Twin (uDALES truth 10°, wall 0.2, ground 0.05), W | 9.9 / 10.1 | 6.03 / 6.05 | – | 0.12 / 0.15 (±0.01) | 0.043 / 0.046 |
    | PALM 10°, S | 8.2 / 8.8 (±0.8) | 6.04 / 6.00 | 0.19 / 0.18 (±0.02) | – | – |
    | PALM 10°, W | 8.1 / 9.2 (±0.7) | 6.03 / 5.99 | – | 0.04 / 0.07 (±0.02) | 0.18 / 0.20 (±0.02) |
    | PALM 40°, S | 39.6 / 41.9 (±1) | 5.53 / 5.45 | 0.027 / 0.016 (±0.006) | – | – |
    | PALM 40°, W | 40.1 / 42.4 (±1) | 5.48 / 5.55 | – | 0.13 / 0.05 (±0.03) | 0.011 / 0.023 (±0.005) |

    - **Twin:** inflow and ground z0 are recovered; wall z0 is
      underestimated (0.12–0.15 vs 0.2) but the held-out error is at the
      noise floor (CRPS of window-mean |U| over the 10 sensors 0.075–0.078,
      replica 0.092). With a perfect model the pair is identifiable enough.
    - **Cross-model, the ground z0 carries the single z0's swing:**
      0.18–0.20 at 10° and 0.011–0.023 at 40° (~10×), while the wall z0 stays
      small at 10° (0.04–0.07) and is undetermined at 40° (0.05 and 0.13
      between seeds). The pair is no more transferable than the single z0
      (item 11).
    - **W adds nothing over S** on the held-out sensors: CRPS |U| over all
      10, 10° S 0.13–0.14, W 0.12–0.15 (replica 0.091); 40° S 0.245–0.251,
      W 0.242–0.244 (replica 0.134). By height the arms differ by less than
      the seed spread at both angles.
    - **40° is much harder than 10°.** The inflow is recovered (angle within
      2°), but at 5.5 m/s, and the error is 2–6× the noise floor at every
      height except 14 m (CRPS |U| at 2 / 5 / 8 / 14 / 26 m: 0.22–0.27 /
      0.23–0.27 / 0.39–0.50 / 0.12–0.14 / 0.47–0.56; replica 0.095 / 0.104 /
      0.088 / 0.183 / 0.130). Updates barely improve on each window's
      forecast. The adaptive error saturates: ×4.2–4.4 at 2 m, ×10 (the cap)
      at 8 m, ×8–8.7 at 20 m, vs ×4.6–5.0 at all heights at 10°.
    - **Verdict:** splitting z0 into wall and ground doesn't give a
      transferable roughness. The angle dependence is in the structural
      mismatch (item 11), not in which surfaces carry the roughness; it
      belongs in the observation error, which at 8–20 m is about twice as
      large at 40° as at 10°.

## 4. Ruled out, and why

| Idea | Why not |
|---|---|
| Learning or fitting the model–truth difference (learned SGS/bias correction, additive forcing from the PALM–uDALES difference) | Needs the truth field; doesn't exist with real data; doesn't transfer (§1, item 7) |
| The 3-coefficient SGS correction as a DA parameter | Mean flow insensitive; error not where it acts (item 7) |
| Tuning parameters without a real-world counterpart until they absorb the mismatch (e.g. c_vreman) | Compensation, not physics; at most a nuisance parameter (item 8) |
| Case-specific alignment (grid-snapped STL) | Real geometries won't align (§1); used only as a diagnostic |
| A single z0 as a transferable site property | Swings 8× with the inflow angle (item 11) |
| A (wall z0, ground z0) pair as a transferable parameterisation | Ground z0 swings ~10× with the angle, wall z0 undetermined, no held-out gain (item 16) |
| Realisation-noise-only R in cross-model runs | Parameters hit bounds, inflow biased, crashes (item 12) |
| 20 s bins / raw frames as observations with turbulent inflow | Ensemble collapse (item 9) |
| Upstream fetch length as a study axis | Not of interest to the aim (user decision) |

**Accepted model differences** (need vendored-source changes, so documented,
not patched): wall function (uDALES) vs surface layer (PALM); Vreman vs
Deardorff SGS; central-difference vs ws-scheme advection; PALM zeroing w at
warm starts (generate a truth as one continuous run); pylbm's missing wall
model and ~2 % density drift.
[sgs_correction_assessment §6](../archive/research/sgs_correction_assessment_2026-10.md#6-directions-for-real-world-cross-model-da)

## 5. Open questions

- What is the structural residual left (SGS, numerics), and why is it about
  2× larger at 40° than at 10° (item 16)? Is the SGS constant a sensible
  nuisance parameter for it?
- Should the structural error get a spatial correlation? The height-banded
  adaptive inflation does NOT make the held-out predictions calibrated
  (z-score std 3.5–4.9, item 13): how should uncertainty reach unobserved
  locations? This is next (§6).
- With real data, the inflation must come from earlier windows or a separate
  period, not from the same experiment's calibration run (the adaptive error
  does this; the offline calibration doesn't).
- Seed coverage: most follow-up DA results are single seeds, and there is one
  truth realisation per model, so seed spread understates truth variability.

## 6. Immediate experiment plan (priority order)

Done: separate wall/ground roughness (#227), its twin check and the
cross-model test at 10° and 40° (item 16; the pair doesn't transfer).

1. **The held-out overconfidence** (item 13, §5): how uncertainty reaches
   unobserved sensors. Every run in `runs_wallz0` and the seed-1 runs in
   `runs_adaptive_r2` keep `sensor_window_means.nc` (per-member window means
   of u, v, |U| at all 28 sensors, prior and posterior, plus truth and
   replica), so this can start offline without rerunning.
2. **Later**: the SGS constant as a candidate for the street-level residual
   (low priority; ~4–9 % so far); a real-data dry run (deferred by the user).

**Parked** (older review items, not on the current path; revisit if they
become relevant):
- Inflow-turbulence intensity (and length scales) as a parameter: needs
  window-variance observations (item 6).
- Bracketing the structural error with pylbm as a "far" truth (no wall model,
  bounce-back acts like z0 ≈ 0.08–0.09): a calibrated setup should stay
  calibrated against both truths. Before using pylbm's noise floor, check its
  replicas differ: the overnight runs got identical replicas, while
  `configs/model_comparison.yaml` now says pylbm seeds from the clock.
- A 1 m DA model in the region of interest (surrogates for cost), if
  resolution dominates.
- Sensor placement by sensitivity to the estimated parameters relative to the
  structural error.
- From the September review: minimum-image (periodic) distances in
  `DistanceLocalization` (not implemented), the periodic case's forcing
  observability (inflow parameters only nudge the slab above 16 m), restart
  and increment-consistency audits, and adaptive ESMDA step schedules.
  [summary](../archive/research/da_review_2026-09/summary.md),
  [recommendations](../archive/research/data_assimilation_recommendations.md).
  The review's likelihood, hybrid-tempering, diagnostics and code fixes are
  done ([plans/implemented/](../plans/implemented/)).

## 7. Running experiments: lessons

- **Memory.** Before #223, ESMDA in memory mode with `save_prior_state` kept
  all 5 step states, stacked them, and held the previous window's history:
  30–50 GB per uDALES run, and 4 of 5 concurrent runs were OOM-killed. With
  `save_states=prior` (#223) a 32-member run peaks at ~9 GB (memory mode) or
  ~5 GB (disk mode), and nothing carries over between windows: 4 concurrent
  runs fit easily.
- **Cores.** pyurbanair's `cpu_pinning` round-robins from the process
  affinity, so concurrent runs all land on cores 0, 4, 8, 12; PALM's prterun
  binds its ranks to cores 0–4. Give each run its own `taskset` slot and set
  `OMPI_MCA_hwloc_base_binding_policy=none
  PRTE_MCA_hwloc_default_binding_policy=none`. No proper fix yet.
- **Failures.** One diverged member ends a run under `failure.policy=raise`;
  the z0 runs use `resample_from_successes`.
- **Outflow.** Check member max speed near the outflow for backflow bursts;
  results before #218 used the 20 m buffer.
- **Machine.** squamish crashed on 22, 23 (twice) and 24 Oct (GPU drivers).
  IT fixed it on 30 Oct (Fedora 44, kernel 7.2.13, new driver); no crash
  since. Keep runs restartable.
- **Prior.** Centre the inflow-angle prior on the truth case's angle − 10°
  (the 10° setup's offset) for every truth angle; a prior left at the 10°
  case's N(0, 10) for a 40° truth biases the posterior angle low.
- **Disk.** A 32-member, 6-window run with `save_states=prior` writes ~33 GB
  of window states. Extract the sensor means (and by-height scores) first,
  then delete them.

## 8. Where the data lives (local, not in git)

| What | Where | State (30 Oct) |
|---|---|---|
| Wall/ground z0 study (item 16) | `/export/scratch2/ntm/runs_wallz0/` | all 10 runs complete; truths palm40, udales10 (palm10 links to `runs_adaptive_r2`); window states deleted, `sensor_window_means.nc` + `valid_by_height.yaml` per run; `invalid_prior0/` = first 40° runs with the wrong prior |
| Adaptive-error study, complete rerun on dev (40 m buffer) | `/export/scratch2/ntm/runs_adaptive_r2/` | all 6 runs complete (24 Oct); window states deleted; `sensor_window_means.nc` and `valid_by_height.yaml` per seed-1 run (seed 2 kept no states) |
| Adaptive-error study, first attempt (20 m buffer, superseded) | `/export/scratch2/ntm/runs_adaptive_r/` | partial (OOM, crash) |
| z0 regimes (calibrations `C_*`, finals `Z_*` inflow + z0, `T2_*` inflow only) | `/export/scratch2/ntm/runs_z0_regimes/` | Calibrations 10°, 40°, 10° at 4 m/s done; 70° lost. Finals done: Z_a10_s1, Z_a40_s1, T2_a40_s1, T2_a10v4_s1; lost: Z_a10v4_s1, T2_a10_s1; seeds 2–3 pending |
| First z0 study (small results) | `/export/scratch2/ntm/z0_results_archive/` | complete |
| Overnight runs, 14–15 Oct | `/export/scratch1/ntm/overnight_2026-10-14/` (per the note) | not found on 23 Oct |

## Archived notes (detail behind the items above)

| Doc | What it holds |
|---|---|
| [archive/research/sgs_correction_assessment_2026-10.md](../archive/research/sgs_correction_assessment_2026-10.md) | Why SGS correction fails; parity changes; original directions; references |
| [archive/research/overnight_runs_2026-10-15.md](../archive/research/overnight_runs_2026-10-15.md) | Round-2 rerun, forward comparisons (1 m, periodic, pylbm, inflow intensity) |
| [archive/research/sgs_twin_tests_2026-10/](../archive/research/sgs_twin_tests_2026-10/report.md) | SGS Rounds 1, 1b, 2: report, appendices, LaTeX |
| [archive/plans/](../archive/plans/) `sgs_discrepancy_twin_tests`, `sgs_twin_round1b_error_model`, `sgs_round2_palm_truth`, `model_discrepancy_validation` | The executed SGS plans |
| [archive/research/da_review_2026-09/](../archive/research/da_review_2026-09/summary.md), [archive/research/data_assimilation_recommendations.md](../archive/research/data_assimilation_recommendations.md) | September review of the ISDA campaign and the code |

# Cross-model data assimilation: summary of the experiments (Sept–Oct 2026)

All experiments of the cross-model DA line (uDALES DA model, PALM or uDALES
truth) up to 1 November 2026: what was run, what came out, and what follows.
This is the entry point. The working notes it replaces are in
[docs/archive/](../archive/) (map at the end); run data is local, not in git
(§9).

**Bottom line.**
- **Estimate:** the inflow angle and speed. They are recovered across models,
  within about 2° and 0.1–0.5 m/s, at 10° and at 40°.
- **Don't estimate as site properties:**
  - roughness: a single z0, or a separate wall and ground z0;
  - an SGS correction or constant.
  Across models they act as compensators. They help at the conditions they
  were fitted for, but swing up to 10× with the inflow angle.
- **Put the structural error in the observation error.** Estimate it online
  from the innovations (adaptive Desroziers inflation). It is about 5× the
  realisation noise at 10° and saturates at 8–20 m at 40°.
- **Open:** the predictions at held-out sensors are overconfident (z-score
  std 3.5–4.9).

## 1. Aim and constraints

- **The end goal is DA against real observations.** PALM is only a truth from
  a different source than the DA model (uDALES). Neither the true state nor
  the true parameters will be known.
- **Don't learn or fit the model–truth difference**: no learned SGS
  correction, no bias or additive forcing fields built from the PALM–uDALES
  difference. Methods must work with sensor data alone.
- **The cross-model work identifies** *which parameters* to estimate and
  *what must go into the observation error*.
- **No case-specific model alignment** (e.g. snapping the STL to the grid):
  real geometries won't align with the grid.
- **Resolution matters, fetch length doesn't.**

## 2. Standard setup

- **Case:** `xie_and_castro`, 2 m grid, `inflow_outflow`, digital-filter inlet
  turbulence, 40 m outflow buffer (the default since #218; earlier runs used
  20 m).
- **Truth:** one continuous PALM run at 10° (or 40°), 6.0 m/s, z0 0.1 m, plus a
  replica with another seed, which sets the noise floor.
- **DA:** parameter-only ESMDA smoother, 32 members, 4 steps.
  - Windows: 3 of 180 s (z0 study) or 6 of 180 s (adaptive-error and
    wall/ground studies).
  - Observations: 180 s window means of u and v at 18 sensors, z = 2, 8 and
    20 m.
  - Held out: 10 validation sensors at 2–26 m.
- **z0 prior:** log-normal, median 0.05 m, log-std 0.7, bounded to 0.005–0.3 m.
  It is updated in a bounded log space (#211).
- **Inflow-angle prior:** centred 10° off the truth.
- **Scores:**
  - CRPS of the window-mean |U| at the held-out sensors, overall and by height;
  - ensemble-mean RMSE of u and v;
  - field RMSE of u and TKE;
  - the Desroziers ratio.

## 3. Studies at a glance

| # | Study | When | Question | Answer | Code |
|---|---|---|---|---|---|
| A | Model parity | early Oct | Are the two codes comparable at all? | Yes after fixes; gap = near-ground mean flow | #201, #202, #218 |
| B | SGS correction, Rounds 1/1b/2 | Sept–mid Oct | Can a parametrised SGS correction absorb the model difference? | No (closed) | SGS discrepancy branch, archived |
| C | z0 identifiability and cross-model z0 | mid Oct | Is z0 estimable, and what does it do across models? | Yes in twins; cross-model 1.7× truth, a compensator | #208, #211, #210 |
| D | Resolution only (1 m truth) | mid Oct | Does resolution bias z0? | No | – |
| E | z0 transfer and wall-law cause | 21–23 Oct | Does the 10° z0 transfer to 40°? Why 1.7×? | No (8× swing); wall-law distance | – |
| F | Adaptive observation error | 22–24 Oct | Can R be estimated online from data? | Yes, ×4–6 noise, reproducible | #220 |
| G | Wall/ground z0 split | 26–30 Oct | Does a (wall, ground) pair transfer across angles? | No | #227 |
| – | Infrastructure | throughout | Memory, disk, metadata | `save_states`, member metadata | #223, #225 |

## 4. Model parity and the remaining gap (A)

1. **The Round-2 PALM truth carried artefacts, now fixed** (#201, #202):
   - pw-scheme streaks: PALM now uses the ws-scheme;
   - outflow damping over the array's wake: now an empty outflow buffer;
   - inflow kicks inside the array: now the shared digital-filter generator;
   - uDALES c_vreman 0.24, which hid outflow backflow: now 0.12;
   - PALM zeroing the inflow at warm starts: fixed.
2. **Periodic uDALES and PALM agree within realisation noise** (distance /
   noise ratios 1.1–1.4).
   - In `inflow_outflow` runs the gap is in the mean flow near the ground:
     uDALES is 0.24–0.27 m/s faster at street level and 0.12 m/s slower above
     the canopy.
   - The sensor-mean distance is about 0.43 m/s, 6× the noise.
3. **At 1 m the street-level bias vanishes** (+0.27 → +0.03 m/s). But a 2 m
   model's resolved TKE stays about 3× the replica level from a 1 m truth: a
   resolution limit no parameter fixes.
4. **The 20 m outflow buffer was not enough.**
   - Backflow bursts reached the domain in 3 of 40 uDALES runs and inflated
     street-level TKE 2–5× at 40°.
   - 40 m and 60 m agree within noise, so 40 m is the default (nx 50 → 70,
     `ncpu` must divide 70) (#218).
   - A 40° forward check with 20 and 40 m buffers confirmed that uDALES and
     PALM both turn 40° inflow to about 33° aloft. The flow turning isn't a
     model difference.
5. **Inflow turbulence intensity shows only in turbulence statistics.**
   0.10 → 0.20 leaves the means within noise but moves TKE 3–13×. Estimating
   it needs variance-type observations.

## 5. SGS correction (B, closed)

- **Round 1 (uDALES truth).** The 3-coefficient SGS correction is recovered
  only with inlet turbulence off. With turbulent inflow nothing is recovered.
- **Round 1b (observation design).** With turbulent inflow, the realisation
  noise (≈ 0.15 m/s per 20 s mean) must be in R, and there must be few
  observations compared with members. 180 s window means work; 20 s bins and
  raw frames collapse the ensemble.
- **Round 2 (PALM truth).** The correction adds no value: the coefficients
  flip sign or saturate. The mean flow barely responds to the SGS term, and
  the error sits where it doesn't act (walls, street level).
- **A single SGS constant compensates.** In the Round-2 rerun, c_vreman went
  from 0.12 to 0.22 ± 0.01 for a small, consistent gain (~4–9 % CRPS). At
  most it is a nuisance parameter.

## 6. Roughness (C, D, E)

**Identifiable when the model is right** (3 seeds, truth z0 0.1, 10°,
6.0 m/s):

| Experiment | z0 (m) | Angle (°) | Speed (m/s) | Desroziers | Held-out CRPS \|U\| (replica) |
|---|---|---|---|---|---|
| uDALES twin | 0.088–0.096 | 9.8–10.2 | 5.98–6.02 | 0.7–1.0 | 0.05 (0.089) |
| PALM twin | 0.095–0.107 | 9.4–9.7 | 5.97–5.99 | 0.73–0.78 | 0.04 (0.093) |
| uDALES 1 m truth → 2 m DA | 0.091–0.113 | 9.7–10.3 | 6.02–6.09 | 0.92–0.96 | 0.13–0.14 (0.11) |
| PALM → uDALES, inflow only | – | 9.0–9.6 | 6.01–6.10 | 0.99 | 0.17–0.18 (0.093) |
| PALM → uDALES, inflow + z0 | 0.17 ± 0.035 | 8.8–9.2 | 5.99–6.03 | 0.88–0.89 | 0.09–0.10 (0.093) |

The cross-model rows used an inflated R (0.53 / 0.43 / 0.33 m/s at 2 / 8 /
20 m) from an inflow-only calibration run.

- **Resolution doesn't bias z0** (D). The 2 m model recovers the true z0
  against a 1 m truth.
- **Across models, z0 settles at 1.7× the truth.**
  - It halves the held-out error, almost all of it below 5 m (2 m CRPS 0.31 →
    0.085, replica 0.056).
  - Field u and TKE errors don't change.
- **It doesn't transfer** (E).
  - In forward runs, z0 0.17 is best at 10° but no better than 0.05 or 0.1 at
    40°.
  - DA at 40° gives z0 0.021 [0.011, 0.043] against 0.17 [0.11, 0.22] at 10°:
    an 8× swing.
- **Cause: the wall-law distance.**
  - Both codes put the first fluid point 1 m from a wall, but uDALES evaluates
    its log law at the true facet distance, 2 m for every cube wall here.
  - On the same flow, uDALES' law gives 0.47× PALM's wall stress and 0.65×
    its roof stress.
  - An STL snapped to the grid closes the canopy gap. That was a diagnostic
    only (§1).
  - With the walls matched, a street-level residual of the opposite sign
    remains (SGS or numerics).
- **Without inflation, R breaks the run.** With only the realisation noise in
  R (Desroziers ≈ 5), the first update pushed z0 to 1–3 m and uDALES crashed.

## 7. Adaptive observation error (F, #220)

**Method.**
- Per height group g, the variance inflation is
  λ_g² = mean over windows of d_a·d_f / R_conf (Desroziers et al. 2005; Li,
  Kalnay & Miyoshi 2009).
  - d_f and d_a are the innovations before and after the update.
  - It is pooled over the previous windows; window 0 is dropped once later
    windows exist.
  - It is clipped to [1, 10] and applied after the interval aggregation.
- The output stores both the configured std (`obs_std`) and the inflated one
  (`obs_std_assimilated`).
- Config: `adaptive_error.desroziers.*` ([data_assimilation.md](../data_assimilation.md)).

**Study** (PALM 10° → uDALES, 6 windows, 2 seeds, 40 m buffer; final-window
posterior medians):

| Arm | Angle (°) | Speed (m/s) | z0 (m) | Inflation 2 / 8 / 20 m |
|---|---|---|---|---|
| A: noise-only R, inflow + z0 | 7.9 / 8.2 | 5.91 / 6.11 | 0.29 / 0.30 (at the bound) | – |
| B: adaptive R, inflow + z0 | 8.1 / 9.0 | 6.04 / 6.02 | 0.20 / 0.19 | ×4.8 / 4.7 / 4.2–4.7 |
| C: adaptive R, inflow only | 9.7 / 10.1 | 6.22 / 6.06 | – | ×5.5 / 5.1–5.4 / 4.0–5.2 |

- **The inflation is reproducible:** ×4–6 the realisation noise in every run,
  matching the offline calibration. It is learned from the data alone, so it
  applies to real observations.
- **The inflow is recovered best without z0 (C).** z0 pulls the angle 1–2° low.
- **B is best on the held-out sensors** (CRPS of window-mean u 0.15–0.17
  against 0.21–0.24 for A and C). The gain is at 2–5 m; at 2 m B is still
  about 2× the noise level.
- **All arms are overconfident at the held-out sensors** (z-score std
  3.5–4.9). The inflation widens the uncertainty of the assimilated
  observations, but a parameter-only ensemble has too little spread at
  unobserved locations.

## 8. Separate wall and ground roughness (G, #227)

**Code (#227).** pyudales takes either:
- `roughness_length` (all facets), or
- `roughness_length_wall` (vertical facets, |n_z| < 0.5, retyped to their own
  facet type) + `roughness_length_ground` (ground and roofs).

PALM without its urban surface model has one z0 and rejects the split. `check_config` validates the combinations
([pyudales.md](../pyudales.md)).

**Study** (adaptive R, 6 windows, 2 seeds; S = inflow + single z0, W =
inflow + wall z0 + ground z0; the 40° prior angle is centred on 30°;
final-window posterior means):

| Run | Angle (°) | Speed (m/s) | z0 single | z0 wall | z0 ground |
|---|---|---|---|---|---|
| Twin (uDALES truth 10°, wall 0.2, ground 0.05), W | 9.9 / 10.1 | 6.03 / 6.05 | – | 0.12 / 0.15 | 0.043 / 0.046 |
| PALM 10°, S | 8.2 / 8.8 | 6.04 / 6.00 | 0.19 / 0.18 | – | – |
| PALM 10°, W | 8.1 / 9.2 | 6.03 / 5.99 | – | 0.04 / 0.07 | 0.18 / 0.20 |
| PALM 40°, S | 39.6 / 41.9 | 5.53 / 5.45 | 0.027 / 0.016 | – | – |
| PALM 40°, W | 40.1 / 42.4 | 5.48 / 5.55 | – | 0.13 / 0.05 | 0.011 / 0.023 |

- **Twin:** the inflow and the ground z0 are recovered. The wall z0 comes out
  low (0.12–0.15 against 0.2), but the held-out error is at the noise floor
  (CRPS 0.075–0.078, replica 0.092).
- **Cross-model, the ground z0 carries the single z0's swing** (~10× between
  angles). The wall z0 is small at 10° and undetermined at 40°.
- **No held-out gain from the split:**

  | | Single z0 | Wall/ground | Replica |
  |---|---|---|---|
  | 10° | 0.13–0.14 | 0.12–0.15 | 0.091 |
  | 40° | 0.245–0.251 | 0.242–0.244 | 0.134 |

- **40° is much harder.**
  - The speed comes out about 0.5 m/s low.
  - The error is 2–6× the noise floor at every height except 14 m.
  - The inflation reaches ×4.3 at 2 m, ×10 (the cap) at 8 m and ×8.5 at 20 m,
    against about ×4.8 everywhere at 10°.
- **The first 40° runs were invalid.** Their angle prior was the 10° case's
  N(0, 10), which biased the posterior angle to 34–37°. They were rerun with
  the prior centred on 30°.

## 9. Infrastructure and running lessons

- **`assimilation.save_states: posterior | prior | all`** (#223) replaces
  `save_prior_state`.
  - Memory and disk mode now behave the same.
  - A 32-member uDALES run peaks at ~9 GB in memory mode and ~5 GB in disk
    mode, with nothing carried over between windows. Before #223 it was
    30–50 GB, and 4 of 5 concurrent runs were killed for running out of
    memory.
- **Disk-mode state files keep per-member metadata** (#225:
  `model_discrepancy_by_member`).
- **Concurrency.**
  - At most 3 DA runs at a time, through the memory-gated queue
    (`/export/scratch2/ntm/shared/memqueue.sh`).
  - Give each run its own `taskset` core slot: pyurbanair's `cpu_pinning`
    and PALM's rank binding otherwise put every run on the same cores.
- **Disk.**
  - A 6-window run with `save_states=prior` writes ~33 GB of window states.
  - Extract per-member sensor window means first
    (`/export/scratch2/ntm/shared/extract_sensor_means.py` →
    `sensor_window_means.nc`), then delete the states.
- **Failures.**
  - One diverged member ends a run under `failure.policy=raise`; the studies
    used `resample_from_successes`.
  - squamish crashed 4 times on 22–24 Oct from GPU drivers. IT fixed it on
    30 Oct, with no crash since.
- **CI.** The `integration (macos-15)` job fails in about 1 of 4–5 runs: the
  vendored `IBM_preproc.exe` segfaults (the IBM top-cell out-of-bounds read).
  It isn't caused by code changes; re-running the failed job clears it.

## 10. Ruled out

| Idea | Why not |
|---|---|
| Learning or fitting the model–truth difference | Needs the truth field; doesn't exist with real data (§1) |
| SGS correction as a DA parameter | Mean flow insensitive; error not where it acts (§5) |
| Tuning parameters with no real-world counterpart until they absorb the mismatch (e.g. c_vreman) | Compensation, at most a nuisance parameter (§5) |
| Case-specific alignment (grid-snapped STL) | Real geometries won't align; diagnostic only (§6) |
| A single z0 as a site property | Swings 8× with the inflow angle (§6) |
| A (wall z0, ground z0) pair as a transferable parameterisation | Ground z0 swings ~10×, wall z0 undetermined, no held-out gain (§8) |
| Realisation-noise-only R across models | Parameters hit bounds, inflow biased, crashes (§6, §7) |
| 20 s bins / raw frames as observations with turbulent inflow | Ensemble collapse (§5) |
| Upstream fetch length as a study axis | Not of interest to the aim |

**Accepted model differences** (fixing them needs changes to vendored code):
- wall function (uDALES) vs surface layer (PALM);
- Vreman vs Deardorff SGS;
- central-difference vs ws-scheme advection;
- PALM zeroing w at warm starts (generate a truth as one continuous run);
- pylbm's missing wall model and ~2 % density drift.

## 11. Open questions and next steps

1. **Held-out overconfidence** (next).
   - How should uncertainty reach unobserved sensors? Options are a spatially
     correlated structural error, state or forcing perturbations, or a
     structural-error term in the predictions.
   - It can start offline: every wall/ground run and the seed-1
     adaptive-error runs keep `sensor_window_means.nc`.
2. **The structural residual.** Why is it about 2× larger at 40° than at 10°?
   Is the SGS constant a sensible nuisance parameter for it (low priority)?
3. **With real data,** the inflation must come from earlier windows or a
   separate period. The adaptive error already works this way.
4. **Seed coverage.** There are 2 seeds per arm and one truth realisation per
   model, so the seed spread understates truth variability.
5. **Parked:**
   - inflow-turbulence intensity as a parameter (needs variance observations);
   - pylbm as a "far" truth;
   - a 1 m DA model in the region of interest;
   - sensitivity-based sensor placement;
   - minimum-image distances in `DistanceLocalization`;
   - a real-data dry run (deferred).

## 12. Where the data lives (local, not in git)

| What | Where |
|---|---|
| Wall/ground study (G): 10 runs, truths palm40 and udales10 (palm10 links to F) | `/export/scratch2/ntm/runs_wallz0/` |
| Adaptive-error study (F), 6 runs, and the PALM 10° truth | `/export/scratch2/ntm/runs_adaptive_r2/` |
| z0 regimes (E): inflow-only calibrations, inflow + z0 finals at 10° and 40° (truths deleted) | `/export/scratch2/ntm/runs_z0_regimes/` |
| First z0 study (C, D), small results | `/export/scratch2/ntm/z0_results_archive/` |
| Shared tools: `px.sh`, `memqueue.sh`, `extract_sensor_means.py` | `/export/scratch2/ntm/shared/` |

Each run keeps its config, parameters, observations, metrics, figures, logs,
`valid_by_height.yaml` and (G, F seed 1) `sensor_window_means.nc`. The window
states were deleted after extraction. Superseded data (the 20 m-buffer truths, the
first adaptive-error attempt, the invalid-prior 40° runs, the parity and
buffer check runs) was deleted on 1 Nov.

## Archived notes (detail behind this summary)

| Doc | What it holds |
|---|---|
| [archive/research/da_cross_model_status_2026-10.md](../archive/research/da_cross_model_status_2026-10.md) | The running status doc this summary replaces (items 1–16) |
| [archive/research/roughness_estimation_2026-10.md](../archive/research/roughness_estimation_2026-10.md) | z0 study C–E in full: tables by height, transfer runs, resolution-only |
| [archive/research/sgs_correction_assessment_2026-10.md](../archive/research/sgs_correction_assessment_2026-10.md) | Why the SGS correction fails; the parity changes |
| [archive/research/overnight_runs_2026-10-15.md](../archive/research/overnight_runs_2026-10-15.md) | Round-2 rerun, forward comparisons (1 m, periodic, pylbm, inflow intensity) |
| [archive/research/sgs_twin_tests_2026-10/](../archive/research/sgs_twin_tests_2026-10/report.md) | SGS Rounds 1, 1b, 2 |
| [archive/plans/](../archive/plans/) | The executed SGS plans |
| [archive/research/da_review_2026-09/](../archive/research/da_review_2026-09/summary.md) | September review of the DA campaign and code |

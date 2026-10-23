# Cross-model data assimilation: status and next experiments

Status as of 23 October 2026. This is the single entry point for the
data-assimilation / cross-model (uDALES vs PALM) / model-discrepancy line of
work. It replaces the October notes and the September review, now in
[docs/archive/](../archive/) (map at the end). The z0 study's detailed results
stay in [roughness_estimation_2026-10.md](roughness_estimation_2026-10.md).
Newest results (21–23 Oct) come from the session that ran them; several aren't
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
13. **Adaptive (online) observation error works for the inflow** *(session,
    preliminary)*: inflow only, 2 seeds: angle 9.5 / 9.7°, speed 6.18 / 6.08
    m/s (truth 10°, 6.0), spread covers the truth; the online inflation
    settles at the offline calibration values. Inflow + z0 with the adaptive
    error (arm B): no completed run yet.
14. **Summary verdict:** inflow angle and speed are robustly estimable
    cross-model; roughness isn't (as a single z0). Street-level structural
    error belongs in the observation error.
15. **Separate wall z0 is possible only in uDALES** *(session)*. uDALES
    supports z0 per facet type (`factypes.inp`), but in `xie_and_castro` all
    facets are type 1 and pyudales writes one z0 to all; splitting walls /
    roofs / ground needs a small pyudales change. PALM without USM/LSM has one
    z0 for all surfaces; its urban surface model has per-surface z0 but needs
    radiation and an energy balance, so it isn't used. Only the DA model
    (uDALES) needs the split.

## 4. Ruled out, and why

| Idea | Why not |
|---|---|
| Learning or fitting the model–truth difference (learned SGS/bias correction, additive forcing from the PALM–uDALES difference) | Needs the truth field; doesn't exist with real data; doesn't transfer (§1, item 7) |
| The 3-coefficient SGS correction as a DA parameter | Mean flow insensitive; error not where it acts (item 7) |
| Tuning parameters without a real-world counterpart until they absorb the mismatch (e.g. c_vreman) | Compensation, not physics; at most a nuisance parameter (item 8) |
| Case-specific alignment (grid-snapped STL) | Real geometries won't align (§1); used only as a diagnostic |
| A single z0 as a transferable site property | Swings 8× with the inflow angle (item 11) |
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

- Does z0 stay off the bounds with honest spread when the observation error is
  adaptive (arm B)?
- Is a (wall z0, ground/roof z0) pair consistent across inflow angles, i.e. a
  transferable parameterisation where the single z0 isn't?
- What is the street-level residual left after the walls are matched (SGS,
  numerics)? Is the SGS constant a sensible nuisance parameter for it?
- Should the structural error get a spatial correlation, and does the
  height-banded adaptive inflation stay calibrated at held-out sensors (not
  just Desroziers at assimilated ones)?
- With real data, the inflation must come from earlier windows or a separate
  period, not from the same experiment's calibration run (the adaptive error
  does this; the offline calibration doesn't).
- Seed coverage: most follow-up DA results are single seeds, and there is one
  truth realisation per model, so seed spread understates truth variability.

## 6. Immediate experiment plan (priority order)

1. **Merge the infrastructure**: `assimilation.save_states:
   posterior|prior|all` (replaces `save_prior_state`; fixes ESMDA memory
   growth, same behaviour in memory and disk mode; in progress) and the
   adaptive observation error (PR #220: online per-height Desroziers
   inflation, smoother only, off by default, settings in
   `configs/assimilation_settings/adaptive_error.yaml`). Relaunch with
   `save_states=prior` (or disk mode); up to ~4 concurrent runs after the fix.
2. **Finish the adaptive-error study**: B_s1, B_s2, A_s1, and salvage C_s2
   (all windows saved, final assembly lost) and A_s2 (5/6 windows). Question:
   does z0 stay off the bounds with honest spread under the adaptive error?
3. **Estimate wall z0 separately from ground/roof z0** in uDALES (PALM truth
   keeps one z0), at 10° and 40°. Needs a small pyudales change: a wall facet
   type and a `roughness_length_wall` parameter. If the pair agrees across
   angles it is a transferable parameterisation; the single z0 is not.
4. **Trim the z0-regime runs** to seed 2 at 10° and 40° plus the 70°
   calibration (lost, rerun); drop third seeds.
5. **Later**: the SGS constant as a candidate for the street-level residual
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

- **Memory.** ESMDA in memory mode with `save_prior_state` keeps all 5 step
  states, stacks them, and holds the previous window's history: 30–50 GB per
  uDALES run. Five concurrent runs on the 125 GB box got 4 OOM-killed. Until
  `save_states` lands, run at most 2–3 at a time.
- **Cores.** pyurbanair's `cpu_pinning` round-robins from the process
  affinity, so concurrent runs all land on cores 0, 4, 8, 12; PALM's prterun
  binds its ranks to cores 0–4. Give each run its own `taskset` slot and set
  `OMPI_MCA_hwloc_base_binding_policy=none
  PRTE_MCA_hwloc_default_binding_policy=none`. No proper fix yet.
- **Failures.** One diverged member ends a run under `failure.policy=raise`;
  the z0 runs use `resample_from_successes`.
- **Outflow.** Check member max speed near the outflow for backflow bursts;
  results before #218 used the 20 m buffer.
- **Machine.** squamish crashed on 22 Oct (system SSD I/O errors) and
  rebooted; several runs were lost (§8).

## 8. Where the data lives (local, not in git)

| What | Where | State (23 Oct) |
|---|---|---|
| Adaptive-error study (A: z0 + noise-only R; B: z0 + adaptive R; C: inflow + adaptive R) | `/export/scratch2/ntm/runs_adaptive_r/` | C_s1 complete; C_s2 windows saved, assembly lost; A_s2 5/6 windows; A_s1, B_s1, B_s2 lost (OOM) |
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

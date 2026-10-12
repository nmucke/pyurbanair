# SGS correction, Round 2: does it improve uDALES predictions of a PALM flow?

Status: proposed, not run. Written 2026-10-12 against `3b279f3` (main). It
follows Round 1 and Round 1b
([sgs_discrepancy_twin_tests.md](sgs_discrepancy_twin_tests.md),
[sgs_twin_round1b_error_model.md](sgs_twin_round1b_error_model.md); results in
`docs/research/sgs_twin_tests_2026-10/` on `exp/sgs-twin-tests`).

## Question

> **When uDALES assimilates sensor data from a PALM simulation, does
> estimating the SGS correction `b0, b1, b2` (together with the inflow) give
> better estimates of the flow state and of quantities of interest than
> estimating the inflow alone?**

There is no true `b`. The mismatch comes only from PALM and uDALES being
different models: different SGS closures, numerics, grids, wall treatment and
inflow generation. The coefficients are calibration knobs. **Their values are
not scored**: they are judged only by the predictions they produce, on data
the assimilation never saw. A correction that improves the predictions with
"implausible" values still counts as a success. One that leaves them unchanged
is "no added value", however its coefficients look.

### What Rounds 1 and 1b imply for the design

1. **Inflow turbulence decides everything.**
   - With a smooth, deterministic truth (Round 1, stage A), estimating `b`
     improved the held-out errors by 40–90 %.
   - With turbulent inflow (stage B) it gave nothing, because the
     realisation noise (≈ 0.17 m/s per 20 s mean) swamps the SGS signal.
   - Round 2 therefore runs **both regimes** (see Truth regimes).
2. **The error model has to be honest, and window means are the setting that
   works.** Only 180 s window means with a persistent representation error
   avoided ensemble collapse with turbulent inflow (Round 1b). Raw frames and
   20 s bins collapsed even when their Desroziers ratio and spread/skill were
   ≈ 1.
3. **The smoother only.** The hybrid ran the coefficients out of range and the
   joint filter failed or collapsed in Round 1. They are not used here.
4. **Extra knobs are not free.** Any added parameter can absorb some model
   mismatch. A fair test therefore compares the SGS correction with equally
   cheap alternatives, not only with inflow-only estimation.

## Truth regimes

| Regime | PALM truth | uDALES model inflow | Like |
|---|---|---|---|
| **P-smooth** | `inlet_turbulence.enabled=false`, `initial_seed=false` (the pypalm defaults). Deterministic, so the replica floor is ≈ 0. | inlet turbulence off | Round 1 stage A |
| **P-turb** | `inlet_turbulence.enabled=true` (`dt_disturb=5`, `amplitude=0.25`, the committed defaults), with `initial_seed=true` | uDALES synthetic inlet on | Round 1 stage B |

In P-turb the two codes' inflow turbulence is generated differently (PALM's
random disturbances vs uDALES' digital-filter planes). Matching their
statistics is part of the pilot (§1), not tuned afterwards.

## Comparisons (the "arms")

All arms use the uDALES assimilation model, the same priors for the inflow
(angle N(0, 10°), speed N(5, 1)), and the same error model and sensors.

| Arm | Estimated | Purpose |
|---|---|---|
| **T2** | inflow angle, speed | Baseline: no model correction |
| **T3** | inflow + `b0, b1, b2` (prior N(0, 0.2²)) | The SGS correction |
| **TC** | inflow + `sgs_constant` (c_vreman, prior N(0.24, 0.03²), min 0.18) | The simplest closure correction |
| **Tα** | inflow + `vertical_inflow_exponent` (prior N(0.25, 0.05²)) | An equally cheap *non-SGS* knob: is any extra degree of freedom enough? |
| *T3C (optional)* | inflow + `sgs_constant` + `b` | Only if both T3 and TC add value |
| *Prior (no DA)* | none: the open-loop prior ensemble | Context; it comes free as each run's window-0 prior |

Per arm, 3 seeds (`assimilation.seed`, `prior_params.seed`). The PALM truth is
the same for all seeds of a regime (see the caveat in §4).

## Scores: states and quantities of interest only

Everything is scored on what the assimilation did not see:

- the **held-out validation sensors**;
- the **full field**;
- the **forecasts**: the window-1/2 prior, which is a forecast from the
  previous window's posterior made before seeing the window's data.

Every score is reported for both the posterior and the forecast, and against
the noise floor (§4).

**State (existing `compute_metrics.py` blocks):**

- `sensor_statistics` on the validation sensors: CRPS of window means and
  window variances of u, v and |U|. These are the headline scores.
- `sensor_distributions`: W2 at the validation sensors.
- `field_statistics`: RMSE of the time-mean u, v and w, the TKE and
  ⟨u′w′⟩ against PALM over the fluid cells, with PALM on the uDALES cell
  centres. **Check this regridding in the smoke test (§1.3).**
- `canopy_profiles`: horizontally averaged profiles of u, TKE and ⟨u′w′⟩.
- `spectra`: the large-scale and mid bands only. The near-cutoff band mostly
  measures the two codes' different numerical dissipation, which no inflow or
  SGS knob should be expected to fix. Report it, but don't score it.

**Quantities of interest.** These are computed offline from the saved window
states by a new analysis script, `qoi.py`, beside `compute_metrics.py`'s
outputs:

1. **Pedestrian-level wind:** the RMSE of the time-mean |U| map at z ≈ 2 m
   (the first fluid cells) against PALM, over the street cells.
2. **Wind-comfort exceedance:** per street cell, the fraction of time that
   |U| > 1.5 × the median street |U|, compared with PALM. Report the RMSE
   and the bias of the exceedance map. The threshold is relative so it
   travels between regimes.
3. **Roof-level exchange:** the turbulent momentum flux ⟨u′w′⟩ and the mean
   |w| through the horizontal plane at z = H = 10 m, averaged over the array:
   the canopy's ventilation rate.
4. **Street-canyon mean flow:** the time-mean u and v averaged over each N–S
   lane (x = 10/20/30) and each E–W gap, as a small vector of channel flows.
   It is the "what a city model would report" quantity.

Each QoI is scored as the error of the ensemble mean, and as the CRPS where a
distribution makes sense.

**The coefficients are reported as diagnostics only:**

- the posterior of `b`, `sgs_constant` and α;
- their consistency across windows and seeds;
- whether they pile up at the prior edge or saturate the multiplier.

A drifting or seed-flipping coefficient that still improves the predictions is
fitting noise in a way that happens to help. Flag it, because it would not
transfer.

## Runbook

### 0. Code prerequisites (small; separate PRs)

1. **pypalm seed.** Add an optional `random_seed` (or `ensemble_member_nr`)
   key that writes `&initialization_parameters ensemble_member_nr`. This is
   the only lever on PALM's random generator
   (`docs/scripts_and_configs.md`). It must be a no-op when unset, with a
   unit test. It is needed for the P-turb replica and for a second P-turb
   truth realisation.
2. **`qoi.py`** in the experiment's tools (not committed unless it proves
   generally useful): the four QoIs above, from `windows/window_*_state.nc`
   and the PALM truth regridded the same way `compute_metrics.py` does.
   Unit-test it on a synthetic field where the answer is known.

### 1. Pilot (forward runs only, about half a day)

1. **The PALM truths.**
   - Run `scripts/run_forward.py model=pypalm case=xie_and_castro
     params=static_truth time.spinup_time=30 time.simulation_time=540`,
     with inflow 10° and 6.0 m/s, for both regimes. Use the `ncpu` that
     divides nx = 30.
   - Remove the `sgs_bias_*` and `vertical_inflow_exponent` entries from
     `static_truth` with `~`, unless PALM provably ignores them.
   - Keep PALM's own closure (`sgs_constant: null`).
   - Record the wall time. PALM diverges on this case for 13–23 % of seeds
     in earlier ESMDA runs, so a truth may need a second
     attempt with another seed.
   - **Replica (P-turb only):** the same run with another `random_seed`.
     For P-smooth, confirm determinism with a plain rerun; the replica floor
     is then ≈ 0.
2. **uDALES against PALM, at the true inflow.**
   - Run uDALES forward with b = 0 and with the Round 1 b* = (0.3, −0.3,
     0.2), at the true inflow, for both regimes.
   - Compare each with the PALM truth on every score above.
   - This gives the **size of the structural mismatch** at the sensors and
     in the QoIs, and whether a nonzero `b` moves uDALES towards or away from
     PALM.
   - It also exposes gross definitional mismatches before any DA: the inflow
     reference height (uDALES `z_ref = 64 m` vs PALM's profile definition),
     the time origin, the spinup and the staggering. Fix definitional
     mismatches in the configs (command line) now, and document them. If the
     mean flows differ by more than the inflow prior can absorb, stop and
     report.
3. **Smoke test the pipeline.** Run one window of T3 with
   `assimilation.truth_dir=<P-smooth run>` and `assimilation.truth_start_time=30`,
   then `compute_metrics.py` and `qoi.py`. Check that the sensor sampling uses
   PALM's operator for the truth (`truth_model=pypalm`), and that
   `field_statistics` really compares PALM interpolated onto the uDALES cell
   centres. Inspect the fields visually.
4. **Noise and error model (as E0 in Round 1b).**
   - Per regime, measure the noise of 180 s window means at the L2 and VAL
     sensors from PALM truth vs replica (P-turb), and from a uDALES ensemble
     at fixed inflow (the member spread).
   - Set the representation error to the measured *realisation* noise only,
     never to the PALM–uDALES mismatch: that mismatch is exactly what the
     arms try to explain. Use the same error for every arm.
   - P-smooth has ≈ 0 realisation noise. Use σ_r = 0.03 m/s there, a
     deliberate floor that keeps the error finite. Check in the pilot that
     it does not collapse the ensemble.
5. **Freeze** before any DA run, and record in the report: the error
   model, the aggregation, the sensor layout and the priors.

### 2. DA runs

The fixed settings follow Round 1:

- **Case and model:** `case=xie_and_castro`, `model@assim_model=pyudales`,
  c_vreman = 0.24 (except where TC estimates it), the SGS feature settings
  (H = 10 m, band [0.5, 2] H, cap log 1.5).
- **Windows:** 3 of 180 s after a 30 s spinup.
- **Smoother:** ESMDA with 4 steps, 32 members and no localisation;
  `failure.policy=raise`, `save_prior_state=true`.
- **Sensors:** layout L2 (18 sensors) plus the 4 validation sensors.
- **Truth:** `assimilation.truth_dir` points at the frozen PALM truth, with
  `assimilation.truth_start_time` set so that it aligns with the uDALES
  windows (from the pilot), and `assimilation.replica_dir` for P-turb.

**Aggregation:** 180 s window means with the frozen persistent error, in both
regimes. For P-smooth also run 20 s bins, because Round 1 stage A showed the
value at 20 s, and in a deterministic flow more bins are not
over-correlated.

| Priority | Runs | Count | Cumulative time (2 side by side, ~1.5 h each) |
|---|---|---|---|
| 1 | P-smooth, 180 s: T2, T3, TC, Tα × 3 seeds | 12 | 9 h |
| 2 | P-turb, 180 s: T2, T3, TC, Tα × 3 seeds | 12 | 18 h |
| 3 | P-smooth, 20 s: T2, T3 × 3 seeds | 6 | 22.5 h |
| 4 | P-turb, second PALM truth realisation (new `random_seed`): T2, T3 × 3 seeds | 6 | 27 h |
| 5 | T3C, or a dense-sensor reference (below), if warranted | 6 | 31.5 h |

Plus the pilot (about half a day). In total that is about 1.5–2 days of
machine time.

- **Disk.** Keep the window states for seed 1 of each arm; QoIs need them.
  For seeds 2–3, compute the metrics and QoIs, then delete the states (about
  11 GB per run otherwise).
- **Dense-sensor reference (optional).** T3 and T2 with many sensors (e.g. a
  `mode: grid` layout at z = 2, 8 and 14 m) give roughly the ceiling of what
  each knob set can achieve. This separates "the correction cannot mimic
  PALM" from "the sparse sensors cannot pin it down".

### 3. What to record per run

- **Run facts:** the posterior and forecast scores of §Scores, the QoIs, the
  calibration (Desroziers ratio, spread/skill, innovation χ², collapse
  check: posterior std ≥ 0.1 prior std), `sgs_health` for T3, solver
  failures and wall time.
- **Coefficient diagnostics:** as described in §Scores.

## 4. Acceptance criteria (frozen before running)

**Noise rules** (as in Round 1). A score difference between two arms counts
only if it meets all of these:

- it is larger than the seed spread;
- it is larger than the replica floor (P-turb);
- it has the same sign in all 3 seeds;
- for P-turb, it has the same sign for both PALM truth realisations
  (priority 4). Seeds vary only the DA, so they understate the truth's
  variability.

**Calibration first.** An arm that collapses, or has a Desroziers ratio
outside 0.7–1.4 or a validation spread/skill ratio outside 0.6–1.5, is
reported as miscalibrated. Its value verdict is then not trusted. These
limits are wider than Round 1b's, because the structural mismatch inflates
both ratios for every arm alike.

**Value verdict per regime, for T3 against each of T2, TC and Tα:**

- **Adds value:** T3 is better by ≥ 10 % on the validation `sensor_statistics`
  CRPS (window means *and* variances), for the posterior *and* the forecast,
  and better or equal on at least 3 of the 4 QoIs, under the noise rules.
  Its calibration must be no worse than the comparison arm's.
- **No added value:** T3 is within noise of the comparison arm.
- **Hurts:** the comparison arm is better under the noise rules.

**Overall reading:**

| T3 vs T2 | T3 vs TC / Tα | Conclusion |
|---|---|---|
| Adds value | Adds value against both | The SGS correction is worth its cost against PALM in this regime |
| Adds value | No added value against TC or Tα | Some model correction helps, but the 3-coefficient SGS form is not special; prefer the simpler knob |
| No added value | n/a | Inflow estimation alone is enough to match PALM at these sensors |
| Hurts | n/a | Check compensation between `b` and the inflow in the posterior correlations; report it |

Expected, given Rounds 1 and 1b: a positive result is plausible in
P-smooth, and unlikely in P-turb at 180 s. Both outcomes are worth
reporting. Do not tune the error model, the sensors or the priors after
seeing the held-out scores; any change is made on the pilot and the
calibration diagnostics only, and is recorded.

## Deliverables

On a new branch `exp/sgs-palm-round2`:

- `docs/research/sgs_palm_round2_<date>/report.md` (plus the LaTeX version,
  reusing `make_figures.py`), containing:
  - the pilot (the mismatch tables, the frozen error model);
  - the run table with the exact commands;
  - the calibration table;
  - the per-regime value tables (T3 vs T2, TC and Tα) for states and QoIs;
  - the coefficient diagnostics;
  - the verdict.
- The two code prerequisites as separate, tested PRs.

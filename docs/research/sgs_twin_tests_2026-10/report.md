# SGS discrepancy twin tests, Round 1 (uDALES truth, uDALES model)

Status: **done**. The pilot ran on 2026-10-08, the DA runs from
2026-10-08 22:00 to 2026-10-10 01:07, and the analysis on 2026-10-10.
Plan: [sgs_discrepancy_twin_tests.md](../../plans/sgs_discrepancy_twin_tests.md).

## Setup

- Machine: squamish (Ryzen 9 3950X, 16 cores / 32 threads, 125 GB RAM).
- Commit: `06249b0` (main), fresh clone, `pixi run setup-dev` with pixi 0.72.1
  (the repo now requires `>=0.72.1`; the installed 0.63.2 cannot parse the
  `linux-64-cuda` platform table).
- Native SGS tests (`tests/pyudales -k "discrepancy or replay" -m integration`):
  5 passed.
- Every setting is on the command line (plan runbook arrays); no config edits.
  Run scripts and the queue live under `.temp/sgs_twin/` in the clone (not
  committed): `tools/lib.sh` (override arrays), `tools/queue_pilot.sh`,
  `tools/queue_da.sh`, `tools/pilot_analysis.py`.
- Sensor check: all 18 L2 points and the 4 validation sensors are fluid at every
  trilinear stencil node (STL voxelised on each variable's staggered grid).
- σ_eff of one 20 s aggregated observation (from `make_observation_error`):
  **0.0603 m/s**.

## Pilot

All pilot runs: `case=xie_and_castro`, `params=static_truth`, inflow 10° / 6.0
m/s, CV=0.24, the plan's SGS feature settings, 3 × 180 s windows after a 30 s
spinup. Wall time ≈ 3 min per run (16 concurrent single-member runs), so
`t_1` ≈ 1 min per member-window.

### P1: stability (CV = 0.24, inlet on)

| b (others 0) | finished | multiplier min / max | saturated | min dt [s] |
|---|---|---|---|---|
| 0 | yes | 1 / 1 | 0 | 0.076 |
| b0 = −0.6 | yes | 0.694 / 0.694 | 0 | 0.068 |
| b0 = −0.3 | yes | 0.775 / 0.775 | 0 | 0.070 |
| b0 = +0.3 | yes | 1.291 / 1.291 | 0 | 0.082 |
| b0 = +0.6 | yes | 1.441 / 1.441 | 0 | 0.085 |
| b1 = −0.6 | yes | 0.695 / 1 | 0 | 0.069 |
| b1 = +0.6 | yes | 1 / 1.439 | 0 | 0.084 |
| b2 = −0.6 | yes | 0.694 / 1.441 | 0 | 0.083 |
| b2 = +0.6 | yes | 0.694 / 1.441 | 0 | 0.069 |

Every value is stable at CV = 0.24, so CV stays 0.24 and the CV = 0.30 repeat
was not needed. ±0.6 already reaches 0.69–1.44, close to the multiplier's
[0.67, 1.5] range. min dt is the smallest uDALES dt seen during the run
(sampled from the run log every 5 s, since output cleanup deletes the log).

### P2: signal to noise

D = RMS(difference of 20 s means over sensors × bins × {u, v}) / σ_eff, pooled
over the 3 windows (27 bins). Same inlet seed within a pair unless the pair is
the seed pair.

| Pair | inlet | L1 | L2 | VAL |
|---|---|---|---|---|
| b = 0 vs b* = [0.3, −0.3, 0.2] (SGS) | on | 0.60 | 0.51 | 0.53 |
| b = 0, seed A vs B (realisation) | on | 3.25 | 4.01 | 3.80 |
| b = 0 vs +5°, +0.5 m/s (inflow) | on | 2.42 | 4.27 | 4.49 |
| b = 0 vs b* (SGS) | off | 0.66 | 0.58 | 0.46 |
| b = 0, irandom A vs B (realisation) | off | 0.01 | 0.01 | 0.01 |
| b = 0 vs +5°, +0.5 m/s (inflow) | off | 2.52 | 4.60 | 4.50 |

Single coefficients at their largest stable values (inlet on, vs b = 0):

| b | L1 | L2 | VAL |
|---|---|---|---|
| b0 = ±0.3 | 1.11 / 1.01 | 1.04 / 0.93 | 0.61 / 0.57 |
| b0 = ±0.6 | 1.60 / 1.41 | 1.46 / 1.24 | 0.95 / 0.85 |
| b1 = ±0.6 | 0.64 / 0.56 | 0.85 / 0.70 | 0.53 / 0.50 |
| b2 = ±0.6 | 1.05 / 1.20 | 0.94 / 1.01 | 0.58 / 0.82 |

**Gate: failed.** No layout reaches D_sgs ≥ 2, with inlet on or off. The
plan's fixes do not rescue it:

1. A larger b* cannot: b0 = ±0.6 already sits near the multiplier cap and
   gives D ≤ 1.6. Under the cap, b* has at most about 3× its current effect.
2. Inlet off removes the realisation noise (D_seed ≈ 0.01, so
   D_sgs ≥ 2·D_seed holds). But D_sgs stays at 0.5–0.7.
3. By the frozen criterion, the correction is **not observable per
   observation** on this case.

Also note that with inlet on, the realisation noise (≈ 0.2 m/s per 20 s mean)
is more than three times σ_eff. Truth and members are different turbulent
realisations, so the assimilated misfit is dominated by noise that the
observation error does not represent.

**Frozen for the DA runs:** CV = 0.24, σ_b = 0.2 (the static prior's
default; ±3σ_b = ±0.6 is inside the stable range), b* = [0.3, −0.3, 0.2]
(the plan's starting value, unchanged), Stage A = inlet off, Stage B = inlet on.

### Why the DA runs went ahead anyway (exploratory)

The gate is a per-observation signal-to-noise ratio. A DA window assimilates
9 bins × 6–18 sensors × 2 components jointly. With inlet off the noise is
≈ 0, so the information across all of them (≈ D·√N ≈ 0.6·√324 ≈ 10 per
window for L2) may still constrain `b`. The machine was idle overnight, so the
full matrix was run as a **post-gate exploratory set**, with b* left
untouched. If T1 recovers `b` despite D < 2, that is a finding about the gate
itself. If it does not, the gate is confirmed. Read every result below in
that light.

## Truth replicas

One continuous 540 s forward run per truth configuration, with only the seed
changed (`.temp/sgs_twin/replicas/`):

- `rep_off_{b0,bstar}`: inlet off, `per_member_irandom=true`, experiment
  `"997"`. It differs from the truth by 4e-4 m/s field RMS after 10 s and by
  0.07 m/s at 540 s: a near-deterministic twin.
- `rep_on_{b0,bstar}`: inlet on, `inlet_turbulence.seed=4242`. It differs by
  0.5–0.8 m/s field RMS throughout.

## DA runs

Queued in plan priority order, two runs side by side at
`num_parallel_processes=8`, `ncpu=1`. Seeds 2 and 3 set `assimilation.seed`,
`prior_params.seed` and the truth inlet seed to 2 and 3 together. Member inlet
seeds come from member names, so they are the same in every run. T2 is also
run for seeds 2 and 3, because the T3-vs-T2 value verdict needs the seed
spread for both.

### Run table

27 of 28 runs finished. The full table, with test, stage, layout, seed,
status, wall time and estimated parameters, is in
[appendix_params.md §1](appendix_params.md). The exact commands are in
[appendix_commands.md](appendix_commands.md).

- **Smoother runs:** 85–112 min with two side by side, plus metrics and
  figures. Two runs took about 200 min (B_T1_L2_s3, B_T3_L2_s2) while the
  machine was under outside load on 2026-10-09.
- **Hybrid:** about 215 min. **Joint filter:** 165 min.
- **Failure:** `A_T3_L2_s1_filtering`, the joint filter with inlet off.
  1 s into filter cycle 2, just after the first joint state+parameter update,
  5 members' dt fell below 1e-4 s and the watchdog killed them. With
  `failure.policy=raise`, that ended the run. Divergence stayed at machine
  precision and the Courant number at 0.07. The diffusion number was pinned
  at 0.255, so dt was diffusion-limited: the eddy viscosity blew up on the
  analysed state. The plan asked for the joint filter only to document that it
  doesn't work, so it was not rerun.

## Results

### Execution

Every smoother run finished without solver failures. Two runs did not
(details in the run table): the inlet-off joint filter failed, and the
inlet-off hybrid ran away (see Method comparison).

Saturation was **not measured correctly**. In every smoother run,
`window_*_posterior_state.nc` carries the *prior* members'
`model_discrepancy_by_member` attribute: its coefficients are the prior b
values, not the posterior ones. So `sgs_health` reports the prior's
multiplier range under the "posterior" label. The posterior and prior lists
are identical in every `metrics.yaml`.

- The reported window-0 saturation of 14–41 % therefore belongs to the prior
  ensemble. Its N(0, 0.2) draws reach the cap in their tails.
- The posterior means stay well inside the unsaturated range. For example
  A_T1_L2_s1 ends at b ≈ (0.34, −0.40, 0.26), whose largest argument is
  about 0.5 against the 0.74 saturation threshold.
- The state data itself is genuine: the posterior and prior fields differ,
  and the posterior spread is 7× smaller.
- The bug is fixed on a separate branch (see Bugs found).

### Recovery

Per-run tables are in [appendix_params.md §2](appendix_params.md). Here are
the final-window posteriors for L2, as the mean over seeds 1–3 with the
seed-to-seed std and the mean posterior std:

| Stage | Test | b0 (truth 0.3) | b1 (−0.3) | b2 (0.2) | inflow angle (10) | speed (6.0) |
|---|---|---|---|---|---|---|
| A | T1 | 0.301, seed sd 0.052, post sd 0.018 | −0.276, 0.121, 0.034 | 0.256, 0.024, 0.023 | pinned | pinned |
| A | T3 | 0.288, 0.054, 0.018 | −0.316, 0.136, 0.031 | 0.193, 0.051, 0.022 | 10.06, 0.15, 0.06 | 6.06, 0.01, 0.00 |
| B | T1 | 0.094, 0.137, 0.000 | −0.094, 0.120, 0.000 | 0.006, 0.064, 0.000 | pinned | pinned |
| B | T3 | 0.094, 0.025, 0.000 | 0.029, 0.175, 0.000 | 0.037, 0.198, 0.000 | 12.50, 1.87, 0.00 | 5.96, 0.06, 0.00 |

Verdicts against the frozen criteria
([appendix_params.md §5](appendix_params.md)):

**Stage A (inlet off): recovered on average, but overconfident.**

- **T1:** passes only for L1, seed 1, where all three b are recovered. It
  fails for L2 in all 3 seeds: b0 is recovered in 2 of them, but each has at
  least one b "Wrong".
- **T0:** passes. Every b is within 0.1 of 0 and within 2·std.
- **T3:** fails "no b wrong" in every run.
- What the numbers show:
  - The seed means are close to the truth: 0.288 / −0.316 / 0.193 for T3.
  - The misses are 0.1–0.6 prior std.
  - The posterior std is 3–4× smaller than the seed-to-seed scatter. That is
    why the 2·std rule calls the misses "Wrong".
- The speed collapses to std ≈ 0.003 with a bias of 0.05–0.08 m/s. The
  inflow angle collapses to ±0.06–0.08° with a 0.1–0.3° bias.
- Window 0 alone is best. A_T1_L2_s1, for example, reaches
  (0.298, −0.339, 0.162) after window 0. Windows 1–2 then drift: b2 rises in
  every run (0.16 → 0.26 in s1), and b1 drifts in s1. All the while the
  spread shrinks.
- χ² is 1.2–1.7 (final up to 2.0) and the Desroziers ratio 1.0–1.14. But the
  validation spread/skill ratio is only 0.1–0.3: the ensemble is far too
  narrow at held-out sensors.
- Posterior correlations (T3): b0–b1 is −0.58 to −0.67 and b0–b2 is +0.43
  to +0.62. The sensors constrain combinations of the coefficients better
  than each one alone.

**Stage B (inlet on): collapsed, nothing recovered.**

- Every ensemble collapses to std ≈ 0.001 within window 0, across all tests,
  layouts and seeds.
- The estimates scatter by seed, at 0.8–3.4 prior std from the truth with
  wrong signs.
- T0 invents a correction: b1 = +0.22 and b2 = −0.18 against a truth of 0.
- In two of three seeds the inflow angle sticks at 13.4–13.7° (truth 10°).
- **Mechanism.** The members' predicted-observation spread is about 0.125 m/s
  at every ESMDA step: inlet-turbulence realisation noise, since each member
  has its own seed. Two things combine:
  - Each window has 324 observations, 32 members and no localisation, so the
    31 parameter anomalies are always an exact linear function of the
    observation anomalies. The update fits noise as if it were signal.
  - The observation error (0.06 m/s) does not include the realisation
    noise (0.13–0.16 m/s RMS between the ensemble mean and the clean truth).
- The Desroziers ratio of 2.3–3.1 and the underfit flag on every B run say
  the same: the observation error should be roughly 3× larger.

### Benefit: T3 vs T2

The full tables, with per-seed signs, are in
[appendix_value.md](appendix_value.md). Scores are on the held-out
validation sensors, as the mean over seeds 1–3.

- "Posterior" is the mean over all 3 windows.
- "Forecast" is the window-1/2 prior: a forecast from the previous window's
  posterior, made before seeing the window's data.
- A row counts only if |T3 − T2| exceeds both the replica floor and the
  seed std, with the same sign in all 3 seeds.

| Score | Stage | T2 | T3 | T3 − T2 | Replica floor | Verdict |
|---|---|---|---|---|---|---|
| CRPS, window-mean magnitude, posterior | A | 0.0327 | 0.0109 | −67 % | 6e-5 | T3 better |
| CRPS, window-mean magnitude, forecast | A | 0.0441 | 0.0235 | −47 % | 8e-5 | T3 better |
| CRPS, window-variance magnitude, posterior | A | 6.0e-5 | 7.5e-6 | −88 % | 2e-6 | T3 better |
| CRPS, window-variance magnitude, forecast | A | 1.47e-5 | 1.52e-5 | +3 % | 2e-8 | T2 better (small) |
| W2 magnitude (member median), posterior | A | 0.0476 | 0.0186 | −61 % | 1e-4 | T3 better |
| Field RMSE, TKE, posterior | A | 0.0048 | 0.0019 | −60 % | 2e-4 | T3 better |
| Canopy-profile RMSE, TKE, posterior | A | 4.6e-4 | 1.3e-4 | −72 % | 1e-5 | T3 better |
| Near-cutoff spectrum, u in canopy, \|dB\|, posterior | A | 8.2 | 6.4 | −23 % | 0.17 | T3 better |
| CRPS, window-mean magnitude, posterior | B | 0.0846 | 0.0852 | +1 % | 0.064 | within noise |
| CRPS, window-mean magnitude, forecast | B | 0.0883 | 0.0872 | −1 % | 0.052 | within noise |
| CRPS, window-variance magnitude, posterior | B | 0.0131 | 0.0126 | −4 % | 0.020 | within noise |
| Field RMSE, TKE, posterior | B | 0.0359 | 0.0358 | 0 % | 0.054 | within noise |

**Stage A: adds value.**

- T3 is clearly better on 10 of the 12 validation CRPS rows, and all 10 clear
  the ≥ 10 % threshold.
- It is also clearly better on 27 of the 34 W2, field, canopy and spectra
  rows, and clearly worse on none.
- The plan's T3-vs-T2 pass requires ≥ 10 % on both posterior and forecast
  for means *and* variances. That is **not fully met**:
  - the forecast `variance_u` is +1.8 %;
  - the forecast `variance_magnitude` is +3.4 %, the same sign in all seeds
    but a tiny effect.
- The T3 inflow error is not above T2's: angle 10.06 against
  10.01–10.33 for T2, speed 6.06 against 6.06–6.08.
- The best reference is the T0 control (correct model, b = 0 truth): it
  scores 0.0108 against T3's 0.0105. Estimating b removes essentially the
  whole SGS-misfit penalty that T2 pays, which costs a factor of 3 on this
  score.
- Both T0 and T3 stay about 170× above the deterministic replica floor. The
  likely cause is the biased, collapsed inflow speed.

**Stage B: no added value.**

- Every CRPS, W2, field and canopy row is within noise, and so are the
  spectra rows.
- The differences between T3 and T2 (−4 % to +2 %) are far smaller than the
  seed std, and both sit within about 1.3× the replica floor.
- T0, T2 and T3 score alike. With inlet turbulence on, the held-out scores
  are dominated by realisation noise that no parameter can reduce on this
  case.

### Layout (T3, seed 1: L1 vs L2)

- **Stage A:** L2 is better on the sensor scores, field RMSE and the
  above-canopy w spectrum (2.1 against 11.3 dB). L1 is better on the
  canopy-u profile.
- **Stage B:** L2 is better on canopy u and on the sensor and forecast
  scores. The two layouts are otherwise alike.
- T1 L1 recovers all three b in stage A even though b1 is only seen
  indirectly there. Its posterior is wider than L2's (b1 std 0.08 against
  0.03), so the strict rule is easier to pass.

### Method comparison (T3, L2, seed 1)

| Score (posterior) | Stage | Smoother | Hybrid | Joint filter | Replica |
|---|---|---|---|---|---|
| CRPS, window-mean magnitude | A | 0.0105 | 0.124 | failed | 6e-5 |
| CRPS, window-mean magnitude | B | 0.0686 | 0.129 | 0.0898 | 0.064 |
| Field RMSE, TKE | B | 0.034 | 0.303 | 0.291 | 0.046 |

- **Hybrid, stage A: runs away.** Its window-0 posterior is reasonable
  (b ≈ 0.23 / −0.16 / 0.14). Window 1 then jumps to b0 = 0.97 and
  b2 = 1.35, and window 2 reaches b0 = 1.13, b2 = 1.57 and a speed of 6.50.
  These values are far outside the prior and in the saturated range, and
  every score is 10× worse than the smoother's.
- **Hybrid, stage B:** collapsed and wrong, like the smoothers.
- **Joint filter, stage B:** all 32 members identical by the end. It shows no
  learning, and its TKE error is 6× the smoother's. This confirms the earlier
  standalone result, as the plan asked.
- The estimated coefficients do **not** survive the state filter. Only the
  parameter-only smoother is usable here.

## Bugs found

- **Stale SGS diagnostics in posterior state files.** Fixed on branch
  `fix/posterior-sgs-diagnostics` (bfaf737), with a regression test in
  `tests/data_assimilation/test_esmda_replay.py`.
  - **Cause.** `ParameterESMDA._analysis_window` stacks the per-step states
    with `xarray.concat(..., join="override")`, and its default
    `combine_attrs="override"` stamps step 0's (the prior's) attributes on
    the whole stack. `run_smoother.py` writes the posterior from
    `esmda_step=-1`, but only with `save_prior_state=true`.
  - **Fix.** The stack now uses `combine_attrs="drop_conflicts"`, the
    smoother keeps each step's own attributes, and `run_smoother.py`
    restores them on the prior and posterior files.
  - **Existing runs.** Their posterior `sgs_health` cannot be recovered: the
    posterior forecasts' native diagnostics were never saved, so getting it
    needs a rerun. The prior diagnostics and every posterior parameter are
    correct.
- **Related, not fixed.**
  - The hybrid and filter posterior states are Kalman-analysed states with
    no attributes, so their `sgs_health` is null. That is arguably honest:
    no single solver run produced them.
  - Their `window_*_forecast_state.nc` concatenates along time with the same
    `override` default, so it keeps only the first cycle's attributes.
  - With `ensemble_save_on_disk=true`, `concat_member_files` copies member
    0's attributes onto the whole ensemble. This was found by reading the
    code and is not checked against data.

## Verdict

**Execution.** The parameter-only smoother ran cleanly in all 24 runs. The
state-updating methods did not: the inlet-off joint filter failed with
diffusion-limited dt collapse after its first analysis, and the inlet-off
hybrid ran the coefficients far outside the prior. Saturation could not be
judged, because of the stale-diagnostics bug.

**Recovery.** With inlet turbulence off (a near-deterministic twin), the
smoother recovers b on average: the seed means are within 0.03 of truth for
T3. T0 does not invent a correction. This holds even though the frozen
per-observation gate (D_sgs ≈ 0.6) predicted the correction would be
unobservable, so the gate is too conservative when the noise is near zero.
The posteriors are 3–4× overconfident, though. By the frozen 2·std rule, T1
on L2 fails in all three seeds and T3 always has a "Wrong" coefficient. With
inlet turbulence on (the realistic setting), nothing is recovered: the
ensemble collapses onto noise within the first window. The cause is that the
observation error omits the realisation noise (≈ 0.15 m/s, against
σ_eff = 0.06) while 324 observations face 32 members without localisation.

**Benefit.** Stage A: **adds value**. T3 cuts the held-out errors by 40–90 %
against T2, on both posterior and forecast, consistently across seeds, and
matches the correctly specified T0. Only the forecast variance scores miss
the plan's 10 % bar. Stage B: **no added value**. T3 and T2 are
indistinguishable and both sit at the replica floor. On this case, whether
estimating SGS pays off depends entirely on whether turbulence realisation
noise is small compared with the SGS signal. With realistic inlet turbulence,
it is not.

With 1–3 seeds this is evidence, not a statistical claim.

## Before Round 2

Round 2 should not start as planned. Stage B, the realistic analogue, fails
on the error model rather than on the correction itself. Changes worth
testing first:

1. **Represent realisation noise in the observation error.**
   `representation_std` ≈ 0.15 m/s, which the Desroziers ratio of about 3
   also suggests. Or assimilate statistics that average the noise down:
   longer aggregation intervals, or window means and variances as the
   observations.
2. **Fewer, more informative observations, or localisation / a larger
   ensemble.** That keeps the parameter-observation regression from being
   rank-deficient.
3. **Fix and rerun saturation** with the corrected diagnostics.

Recheck the gate itself against Stage A. D·√N (≈ 10 per window for L2)
predicts the inlet-off recoverability better than per-observation D.

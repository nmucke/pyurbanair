# SGS twin tests, Round 1b: an error model for turbulent inflow

Status: proposed, not run. Written 2026-10-10 against `3b279f3` (main, with the
#183 attribute fix). It follows Round 1
([sgs_discrepancy_twin_tests.md](sgs_discrepancy_twin_tests.md); report in
`docs/research/sgs_twin_tests_2026-10/` on `exp/sgs-twin-tests`), and comes
before Round 2 (PALM truth).

## Why

Round 1 found:

- **Inlet turbulence off (stage A):** ESMDA recovers the SGS coefficients `b`
  on average, and estimating them improves the held-out scores by 40–90 %
  over inflow-only (T3 vs T2). The posteriors are 3–4× overconfident.
- **Inlet turbulence on (stage B, the realistic one):** every ensemble
  collapses to std ≈ 0.001 within window 0, the `b` estimates are wrong, and
  T3 ≈ T2. The collapse comes from the error model, not from the correction:
  - **The observation error leaves out the realisation noise.** σ_eff is
    0.060 m/s, but truth and members differ by about 0.15 m/s per 20 s
    mean. The Desroziers ratio is about 3. With
    `propagation=propagate_mean`, the configured representation error
    (0.1 m/s) is also divided by √20, because the 20 one-second frames in a
    bin are treated as independent. Turbulence is correlated over seconds,
    so it is not.
  - **Too many observations for the ensemble.** 324 observations per window
    against 32 members, with no localisation, so 31 parameter anomalies
    always regress exactly onto the observation anomalies.

Round 2 only makes sense once stage B works: if uDALES cannot use the
correction against its own turbulent truth, a PALM truth will not tell us
anything. This round asks one question:

> **With an honest error model, can the SGS correction add value when the
> truth is turbulent?**

All three levers already exist; no new code is needed.

## Fixed choices (unchanged from Round 1 unless listed)

Everything in the Round 1 runbook stays the same: case, CV=0.24, the SGS
feature settings, b* = [0.3, −0.3, 0.2], inflow 10° / 6.0 m/s, the priors
with σ_b = 0.2, ESMDA with 4 steps, 32 members, 3 windows of 180 s,
`failure.policy=raise`, `save_prior_state=true`, and layout L2 with the 4
validation sensors. Also:

- **Stage B only** (`INLET=true`), except for the saturation reruns in E5.
- **Replica:** reuse Round 1's `rep_on_bstar` (and `rep_on_b0` for T0). They
  match the truth configuration exactly, and every lever here is on the DA
  side only.
- **Seeds:** 1–3, defined as in Round 1.
- **Overrides:** reuse `.temp/sgs_twin/tools/lib.sh` from the experiment
  clone; the levers are extra overrides on top:

```bash
# Representation error that is not averaged away inside a bin.
ERR=(observation.error.representation_std=$SIGMA_R
     +observation.error.representation_time_model=persistent)
AGG=(observation.aggregation.interval_seconds=$INTERVAL)   # 20 (Round 1), 60, 180
LOC=('smoothing.localization=${localization.correlation}')  # rho_t = 0.35
```

With `persistent`, the error of one aggregated observation is
σ_eff = √(0.25²/n + σ_r²), where n is the number of frames per bin. So
σ_eff ≈ σ_r once σ_r ≳ 0.1.

## Experiments

### E0. Measure the realisation noise (offline, ~1 h, no runs)

σ_r must be the realisation noise at the chosen interval: averaging over
longer bins reduces it. Measure it before any run, and freeze it.

- **Source 1, members.** The collapsed Round 1 stage B smoother ensembles,
  e.g. `B_T3_L2_s{1,2,3}`, window 1–2 prior states. Their members share
  (near) identical parameters but have different inlet seeds, so the
  across-member std of the sensor means is pure realisation noise.
- **Source 2, truths.** Truth vs replica (`true_state.nc` vs
  `rep_on_bstar/state.nc`): RMS difference / √2.
- **Output.** Sample u and v at L2 and VAL through the DA observation operator
  (reuse `tools/pilot_analysis.py`), bin at 20, 60 and 180 s, and report the
  noise std per interval, per height (2/8/14/20 m) and per component.
- **Freeze** σ_r(interval) as the mean over sensors, per interval. If it
  differs by more than 1.5× between heights, use the `height_bands`
  mapping of `representation_std` instead of a scalar.

Expect about 0.15 m/s at 20 s; the pilot's seed pair gives 0.14–0.17. Also
recompute the per-observation gate with the new σ_eff, and the aggregate
D·√N, so the expected signal is on record before the runs.

### E1. Honest error, 20 s bins

`ERR` with σ_r(20 s), and Round 1's aggregation (324 observations per
window).

### E2. Honest error, 180 s bins (window means)

`ERR` with σ_r(180 s), and `INTERVAL=180`. That gives 18 sensors × 2
components = 36 observations per window: about the size of the ensemble, so
the regression is no longer rank-deficient. Run 60 s
(σ_r(60 s), 108 observations) only if E2 is clearly better than E1.

### E3. Honest error and localisation, 20 s bins

`ERR` with σ_r(20 s), plus `LOC`. This tests whether localisation alone
rescues the many-observation setup.

### E4. Larger ensemble (only if E1–E3 still collapse)

The best of E1–E3 with `ensemble.ensemble_size=64`, which costs 2×. It is
only worth it if the parameter spread still collapses below 0.1 prior std.

### E5. Stage A saturation reruns (with the #183 fix)

`A_T1_L2_s1` and `A_T3_L2_s1`, unchanged from Round 1. They give the real
posterior `sgs_health`, the one execution check Round 1 could not make. Also
run A_T3_L2_s1 with E1's error model, to see whether it widens Stage A's
overconfident posterior without losing its value.

### Run matrix and priority

Each variant runs T3 and T2 for seeds 1–3; T1 and T0 are added for seed 1
only (T1 tests identifiability, T0 tests invented corrections). One run takes
about 1.5 h, and two run side by side.

| Priority | Variant | Runs | Cumulative wall time |
|---|---|---|---|
| 1 | E0 (offline) | none | 1 h |
| 2 | E5 saturation reruns | 2 | 2.5 h |
| 3 | E1: T3, T2 × 3 seeds | 6 | 7 h |
| 4 | E2 (180 s): T3, T2 × 3 seeds | 6 | 11.5 h |
| 5 | E3 (localisation): T3, T2 × 3 seeds | 6 | 16 h |
| 6 | Best variant: T1 and T0, seed 1 | 2 | 17.5 h |
| 7 | E5 Stage A with E1's error model | 1 | 18 h |
| 8 | E2 at 60 s, or E4 (64 members), if needed | 6 | 22.5 h |

That is about one day of machine time. Queue it with the Round 1 runner
(`tools/runner.sh`, MAX_SLOTS=16, NPP=8). Add a `queue_round1b.sh` next to
`queue_da.sh`, with the levers as extra arguments to `mkda.sh`.

## Acceptance criteria (frozen before running)

Use the final window, and the mean over seeds where there are 3.

**Calibration (does the error model fit?)**

- No collapse: every estimated parameter keeps a posterior std
  ≥ 0.1 × prior std in all windows.
- The Desroziers ratio is 0.8–1.25.
- The validation spread/skill ratio of the window means is 0.7–1.3.
- Calibration is a precondition. A variant that fails it is reported, but its
  recovery and value verdicts are not trusted.

**Recovery:** the Round 1 rule per coefficient (Recovered / Unconstrained /
Wrong). With an honest error, *Unconstrained* is an acceptable outcome;
*Wrong* is not.

**Benefit (T3 vs T2):** the Round 1 rule. A score row counts only if |T3−T2|
exceeds both the replica floor and the seed std, with the same sign in all
3 seeds. Adds value means the validation `sensor_statistics` CRPS is
≥ 10 % lower for posterior and forecast, and no `b` is Wrong. Otherwise the
verdict is no added value, or hurts.

**Execution:** no solver failures, and E5 shows a posterior saturated
band-cell fraction under 5 %.

## Decision after Round 1b

| Outcome | Next step |
|---|---|
| A calibrated variant where T3 adds value | Round 2 (PALM truth) with that error model, interval and localisation frozen. |
| Calibrated, but no added value in every variant | The sensors cannot see the correction through realistic turbulence on this case. Round 2 then tests only T2 against an estimated `sgs_constant`. Report the 3-coefficient correction as not worth it here. |
| Still collapsed or miscalibrated after E4 | Stop. Report the error model as the open problem (statistic observations, i.e. window variances as observations, would need new code) before any PALM work. |

## Deliverables

On `exp/sgs-twin-tests`:

- the E0 noise table;
- a run table with the exact commands;
- calibration, recovery and value tables per variant, and the decision;
- the existing report and the LaTeX version extended with a "Round 1b"
  section, reusing `make_figures.py`.

Keep the run directories until the decision is made. After that, the Round 1
runs (~260 GB) can go; keep the replicas and the pilot.

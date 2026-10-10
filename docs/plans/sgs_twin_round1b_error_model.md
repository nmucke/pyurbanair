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

The levers (the representation error model, the averaging interval, including none, localisation and the ensemble size) already exist; no new code is needed.

## Fixed choices (unchanged from Round 1 unless listed)

Everything in the Round 1 runbook stays the same: case, CV=0.24, the SGS
feature settings, b* = [0.3, −0.3, 0.2], inflow 10° / 6.0 m/s, the priors
with σ_b = 0.2, ESMDA with 4 steps, 32 members, 3 windows of 180 s,
`failure.policy=raise`, `save_prior_state=true`, and layout L2 with the 4
validation sensors. Also:

- **Stage B** (`INLET=true`) for E1–E3; stage A (`INLET=false`) for E4.
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
NOAGG=(observation.aggregation=null)                       # every 1 s frame
LOC=('smoothing.localization=${localization.correlation}')  # rho_t = 0.35
```

With `persistent`, the error of one aggregated observation is
σ_eff = √(0.25²/n + σ_r²), where n is the number of frames per bin. So
σ_eff ≈ σ_r once σ_r ≳ 0.1. Without aggregation, n = 1 and
σ_eff = √(0.25² + σ_r(1 s)²).

`observation.aggregation=null` composes, and the observation-error code
handles it (`scripts/utils/helper_functions.py`). It gives
180 frames × 18 sensors × 2 = 6480 observations per window. Before queueing,
check that one window runs at that size (memory, ESMDA time) with a single
tiny run.

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
  (reuse `tools/pilot_analysis.py`), bin at 1 s (no averaging), 20, 60 and
  180 s, and report the noise std per interval, per height
  (2/8/14/20 m) and per component.
- **Decorrelation time.** Also report the noise's autocorrelation time τ at
  each sensor. ESMDA assumes uncorrelated observation errors. Frames closer
  than τ are therefore counted as independent information they do not carry:
  unaveraged data is over-weighted by about τ/Δt. This is the case for
  averaging, and E1 tests it. If τ ≳ 20 s, even the 20 s bins are
  correlated, which favours 60 or 180 s.
- **Freeze** σ_r(interval) as the mean over sensors, per interval. If it
  differs by more than 1.5× between heights, use the `height_bands`
  mapping of `representation_std` instead of a scalar.

Expect about 0.15 m/s at 20 s; the pilot's seed pair gives 0.14–0.17. Also
recompute the per-observation gate with the new σ_eff, and the aggregate
D·√N, so the expected signal is on record before the runs.

### E1. Does averaging help? (stage B, honest error)

The main experiment. With the honest error `ERR` at each interval's own
σ_r from E0, compare:

| Arm | Averaging | Observations per window | σ_r |
|---|---|---|---|
| E1-none | none (`NOAGG`, every 1 s frame) | 6480 | σ_r(1 s) |
| E1-20 | 20 s bins (Round 1) | 324 | σ_r(20 s) |
| E1-60 | 60 s bins | 108 | σ_r(60 s) |
| E1-180 | 180 s bins (window means) | 36 | σ_r(180 s) |

- **What it separates.** Averaging trades information for honesty.
  - More frames carry more signal, but ESMDA treats their correlated errors
    as independent.
  - Fewer, longer means have nearly independent errors, but carry less
    detail.
  - At 180 s, 36 observations is about the ensemble size, so the regression
    is no longer rank-deficient.
- **The scale question.** Does the benefit of averaging grow with the
  interval, or is it flat once the error is honest?
- **Control (E1-none-R1).** No averaging with Round 1's error
  (`representation_std=0.1`, `independent`). This shows the cost of using
  raw frames *and* an optimistic error together, and checks that Round 1's
  collapse is not specific to 20 s bins.
- **Reading the result.** Averaging "helps" if an averaged arm beats E1-none
  on calibration first, and then on the held-out T3 scores, under the same
  noise rules as Round 1 (larger than the replica floor and the seed std,
  the same sign in all 3 seeds). If E1-none is as calibrated and as good,
  averaging is unnecessary and the raw frames can be used.

### E2. Honest error and localisation

The averaging interval from E1 that came out best, with `ERR` and `LOC`.
Also run localisation with no averaging (E2-none): localisation is the other
way to cope with many observations, so it may make raw frames usable where
averaging alone does not.

### E3. Larger ensemble (only if E1–E2 still collapse)

The best of E1–E2 with `ensemble.ensemble_size=64`, which costs 2×. It is
only worth it if the parameter spread still collapses below 0.1 prior std.

### E4. Stage A (inlet off): averaging and saturation

Stage A has almost no realisation noise, so it isolates what averaging does
to the information content, apart from noise.

- **E4-sat (with the #183 fix).** `A_T1_L2_s1` and `A_T3_L2_s1`, unchanged
  from Round 1. They give the real posterior `sgs_health`, the one execution
  check Round 1 could not make.
- **E4-none.** `A_T3_L2_s1` and `A_T2_L2_s1` without averaging and with
  Round 1's error (n = 1, so σ_eff = √(0.25² + 0.1²) ≈ 0.27). Against Round
  1's 20 s result, this asks whether averaging loses or keeps Stage A's
  recovery and its value of estimating b.
- **E4-180.** The same with 180 s bins: does Stage A keep its 40–90 % value
  of estimating b with only window means?
- **E4-err.** `A_T3_L2_s1` with E1's honest error at 20 s, to see whether it
  widens Stage A's overconfident posterior without losing the value.

### Run matrix and priority

Each variant runs T3 and T2 for seeds 1–3; T1 and T0 are added for seed 1
only (T1 tests identifiability, T0 tests invented corrections). One run takes
about 1.5 h, and two run side by side.

| Priority | Variant | Runs | Cumulative wall time |
|---|---|---|---|
| 1 | E0 (offline) + one-window check of no averaging | none | 1.5 h |
| 2 | E4-sat saturation reruns | 2 | 3 h |
| 3 | E1-none, E1-20, E1-180: T3, T2 × 3 seeds | 18 | 16.5 h |
| 4 | E1-60 and E1-none-R1: T3, T2 × 3 seeds | 12 | 25.5 h |
| 5 | E4-none, E4-180 (T3, T2), E4-err (T3), seed 1 | 5 | 29.5 h |
| 6 | E2 (best interval) and E2-none: T3, T2 × 3 seeds | 12 | 38.5 h |
| 7 | Best variant: T1 and T0, seed 1 | 2 | 40 h |
| 8 | E3 (64 members), only if needed | 6 | 49 h (64 members cost 2×) |

That is about 1.5–2 days of machine time; the E1-none runs may be slower in
the analysis step, so time the first one. After priority 3, check E1's
outcome before spending the rest: if every arm still collapses, skip ahead
to E2 and E3. Queue it with the Round 1 runner
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

**Averaging (E1, E4):** report, per interval (none, 20, 60, 180 s):
calibration, the T3 parameter errors, and the T3 held-out scores and value
over T2. The verdict is one of:

- **Averaging helps:** an averaged arm is calibrated where E1-none is not, or
  it scores clearly better (under the same noise rules), in all 3 seeds.
- **Averaging is neutral:** E1-none is calibrated and within noise of the
  best averaged arm. The raw frames are then usable, and the interval is a
  free choice.
- **Averaging hurts:** E1-none is calibrated and clearly better. Long means
  throw away information the correction needs.

Report the Stage A arms (E4-none, E4-180) the same way. Without realisation
noise, they show what averaging costs in pure information.

**Execution:** no solver failures, and E4-sat shows a posterior saturated
band-cell fraction under 5 %.

## Decision after Round 1b

| Outcome | Next step |
|---|---|
| A calibrated variant where T3 adds value | Round 2 (PALM truth) with that error model, interval and localisation frozen. |
| Calibrated, but no added value in every variant | The sensors cannot see the correction through realistic turbulence on this case. Round 2 then tests only T2 against an estimated `sgs_constant`. Report the 3-coefficient correction as not worth it here. |
| Still collapsed or miscalibrated after E3 | Stop. Report the error model as the open problem (statistic observations, i.e. window variances as observations, would need new code) before any PALM work. |

## Deliverables

On `exp/sgs-twin-tests`:

- the E0 noise table;
- a run table with the exact commands;
- calibration, recovery and value tables per variant, and the decision;
- the existing report and the LaTeX version extended with a "Round 1b"
  section, reusing `make_figures.py`.

Keep the run directories until the decision is made. After that, the Round 1
runs (~260 GB) can go; keep the replicas and the pilot.

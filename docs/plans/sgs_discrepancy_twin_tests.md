# SGS discrepancy: twin tests of joint inflow + SGS-bias estimation

Status: proposed, not run. Written 2026-10-08 against `5eba7bc` (main). It is
the first concrete slice of [model_discrepancy_validation.md](model_discrepancy_validation.md)
(its steps 1, 3 and 4, same-model only). That plan's command lines predate the
lean refactor; use the ones below.

## Goal

The two rounds ask different questions:

- **Round 1 (this plan, overnight on squamish): uDALES truth, uDALES
  assimilation model.** Does the machinery work? The true coefficients are
  known, so check whether the strain/rotation SGS correction
  (`sgs_bias_b0/b1/b2`,
  [pyudales §4.1](../pyudales.md#41-strainrotation-discrepancy)) can be
  estimated together with the inflow parameters from sparse velocity sensors.
- **Round 2 (later, outline at the end): PALM truth, uDALES assimilation
  model.** Does the correction help? This mimics the real-world case, where no
  "true" uDALES coefficients exist. The coefficients mean different things in
  the two solvers, so they are calibration knobs. They should be whatever
  makes uDALES reproduce the PALM output most closely, and they are not judged
  against a target value.

For Round 1, report three distinct questions separately. Mixing them up is the
most likely way to misread the results:

1. **Execution:** do the runs finish without solver failures or saturation?
2. **Recovery:** are the true coefficients and inflow recovered, within the
   posterior spread?
3. **Benefit:** does estimating SGS improve predictions at held-out sensors and
   in the next window's forecast, compared with estimating inflow alone?

A negative or inconclusive result is a valid outcome. Report it as such. Do not
tune until something passes.

## Fixed choices

**Case.** `case=xie_and_castro` as committed: 30×40×16 cells over
60×80×32 m (2 m grid), staggered cubes 2.8 to 17.2 m tall (mean about 10 m).
This is small enough for a 32-member ensemble on one machine. It is also big
enough for the test to mean something: several rows of buildings, and a
resolved canopy layer in which SGS dissipation matters at 5 cells per building
height. Windows are `time.simulation_time=180`, `time.spinup_time=30`,
`time.output_frequency=1` and `assimilation.num_windows=3`.

**SGS feature settings.** These are pinned explicitly, so the run does not
depend on the current `configs/model/pyudales.yaml`. The settings are
`canopy_height=10`, `height_band_over_H=[0.5,2.0]` (a band from 5 to 20 m),
`gradient_regularization=0.01` and `log_multiplier_cap=log(1.5)`. The
multiplier is `exp(L·tanh((b0 + b1·phi + b2·q)/L))`. Here `phi` is in [0, 1]
and nonzero only inside the band, and `q` is in [-1, 1]. A cell counts as
saturated when the argument exceeds about 0.74. The multiplier always stays
within [0.67, 1.5].

**Parameters.** Truth uses `params@truth_params=static_truth`. Inflow is
`inflow_angle=10` and `velocity_magnitude=6.0`, one prior standard deviation
from the prior mean for both. The prior is `params@prior_params=static` with
inflow N(0, 10°) and speed N(5, 1). The prior's `sgs_constant` and
`vertical_inflow_exponent` are removed, so truth and model share the backend
`sgs_constant` and the default shear exponent. That leaves `sgs_constant` out
as a confounder of `b0` in this round. SGS priors are N(0, σ_b), with σ_b set
by the pilot (default 0.2). The truth coefficients `b*` are also set by the
pilot.

**Smoother.** Parameter-only ESMDA (`${smoother.static}`), `num_steps=4`,
no localization. With 5 parameters and 32 members it is not needed and would
only add a confounder. Use `ensemble.failure.policy=raise`: a diverging member
is information, not something to resample away.

**Observations.** These are the committed defaults, pinned: u and v, 20 s mean
aggregation, `instrument_std=0.25`, `representation_std=0.1` and
`propagation=propagate_mean`. Held-out scoring uses the case's 4 validation
sensors, which include one above the canopy at z=20.

**Sensor layouts:**

- **L1 (street):** the committed 6 sensors, all at z=2 m. This is *below* the
  height band, so `b1` is only seen indirectly.
- **L2 (street + band):** the same 6 (x, y) positions at z=2, 8 and 14 m, which
  gives 18 sensors. The N–S lanes at x=10/20/30 are open at every height, but
  check that the 12 new points are fluid cells against the voxelized geometry
  (the building mask in a forward `state.nc`) before using them.

**Forecast skill without new code.** With `assimilation.save_prior_state=true`,
the window `w ≥ 1` *prior* is a forecast from window `w−1`'s posterior
parameters and end state, made before seeing window `w`'s data. Its scores in
`metrics.yaml` are the assimilation-off forecast skill.

## Runbook

### 0. Setup on squamish (about 30 min)

1. Make a fresh clone of main and run `pixi run setup-dev`. Do not edit
   `configs/`: every setting goes on the command line, because the committed
   configs drift between runs. Results go under `.temp/sgs_twin/` and are never
   committed.
2. Record `nproc`, memory and `git rev-parse HEAD`. `ncpu` must divide `nx=30`
   (1, 2, 3, 5 or 6). Keep each run at 8 workers or fewer
   (`num_parallel_processes`): the machine is DRAM-bound beyond that. If you
   need throughput, run several DA runs side by side, each with its own scratch
   (`paths.scratch.local=$PWD/.temp/scratch/<run>`). They would otherwise
   collide on the uDALES experiment directories.
3. Run the native SGS tests first. They build the discrepancy solver variant
   into `.cache/pyudales`:
   `pixi run -e dev python -m pytest tests/pyudales -k "discrepancy or replay" -m integration`
   If they fail, stop and report. Nothing below is meaningful without them.
4. **Keep every run directory, including the window state files.** The
   turbulence metrics in [da_turbulence_metrics.md](da_turbulence_metrics.md)
   are not implemented yet; they will be recomputed from these files later.
   Budget disk accordingly. One frame is 30·40·16 cells × 4 variables ×
   4 B ≈ 0.3 MB. A window file is then 180 frames × 32 members ≈ 1.8 GB, so a
   smoother run with prior states is about 11 GB. Check `df` before starting.

Shared overrides (bash arrays; `$RUN`, `$NCPU`, `$NPP`, `$CV` and `$INLET` are
set per run):

```bash
M="forward_model.model_discrepancy"
COMMON=(
  case=xie_and_castro model@truth_model=pyudales model@assim_model=pyudales
  params@truth_params=static_truth params@prior_params=static
  'smoothing.smoother=${smoother.static}' 'smoothing.localization=${localization.none}'
  smoothing.num_steps=4
  time.simulation_time=180 time.spinup_time=30 time.output_frequency=1
  assimilation.num_windows=3 assimilation.save_prior_state=true
  ensemble.ensemble_size=32 ensemble.num_parallel_processes=$NPP ensemble.failure.policy=raise
  observation.aggregation.interval_seconds=20 observation.error.instrument_std=0.25
  observation.error.representation_std=0.1
  paths.results_dir=.temp/sgs_twin/$RUN paths.scratch.local=$PWD/.temp/scratch/$RUN
  '~prior_params.parameters.sgs_constant' '~prior_params.parameters.vertical_inflow_exponent'
  truth_params.parameters.inflow_angle.value=10 truth_params.parameters.velocity_magnitude.value=6.0
)
for role in truth_model assim_model; do COMMON+=(
  $role.forward_model.ncpu=$NCPU $role.forward_model.sgs_constant=$CV
  $role.forward_model.closure=vreman $role.$M.enabled=true
  $role.$M.canopy_height=10.0 "$role.$M.height_band_over_H=[0.5,2.0]"
  $role.$M.gradient_regularization=0.01 $role.$M.log_multiplier_cap=0.4054651081
  $role.forward_model.inlet_turbulence.enabled=$INLET
); done
# Single quotes keep ${smoother...} away from bash; quote any [..] list.
L2=('obs.x_points=[10,10,20,20,30,30,10,10,20,20,30,30,10,10,20,20,30,30]'
    'obs.y_points=[20,60,10,50,30,70,20,60,10,50,30,70,20,60,10,50,30,70]'
    'obs.z_points=[2,2,2,2,2,2,8,8,8,8,8,8,14,14,14,14,14,14]')
# Truth coefficients:  truth_params.parameters.sgs_bias_b{0,1,2}.value=...
# SGS prior std:       prior_params.parameters.sgs_bias_b{0,1,2}.std=$SIGMA_B
# Run + metrics + figures:
pixi run -e dev bash workflows/assimilation_workflow.sh smoother "${COMMON[@]}" <extra>
```

All four variants below were composed and checked with `check_config` on
2026-10-08. To pin an inflow parameter in the prior, delete it and add it back
as a Constant. A plain assignment cannot replace the Normal node:

```bash
PIN_INFLOW=('~prior_params.parameters.inflow_angle'
  '+prior_params.parameters.inflow_angle={_target_:pyurbanair.static_parameters.Constant,value:10.0}'
  '~prior_params.parameters.velocity_magnitude'
  '+prior_params.parameters.velocity_magnitude={_target_:pyurbanair.static_parameters.Constant,value:6.0}')
# No SGS correction in the model (coefficients fall back to zero; the extension stays enabled):
NO_SGS=('~prior_params.parameters.sgs_bias_b0' '~prior_params.parameters.sgs_bias_b1'
        '~prior_params.parameters.sgs_bias_b2')
EST_ALL='assimilation.params_to_estimate=[inflow_angle,velocity_magnitude,sgs_bias_b0,sgs_bias_b1,sgs_bias_b2]'
EST_SGS='assimilation.params_to_estimate=[sgs_bias_b0,sgs_bias_b1,sgs_bias_b2]'
EST_INFLOW='assimilation.params_to_estimate=[inflow_angle,velocity_magnitude]'
```

### 1. Pilot: stability and signal-to-noise (forward runs only, 1–2 h)

Use `scripts/run_forward.py` with `case=xie_and_castro model=pyudales
params=static_truth`, the same time, ncpu and SGS feature overrides, and
`params.parameters.sgs_bias_b*.value=...`. Write a small analysis script under
`.temp/sgs_twin/` (not committed) that reads each `state.nc`, samples the L2
and validation sensors (reuse `scripts/utils/helper_functions.py` /
`libs/evaluation` sensor helpers), and bins into 20 s means.

**P1, stability.** Use `CV=0.24`, the committed `c_vreman` and close to its
stability floor on this domain. Sweep `b0 ∈ {−0.6, −0.3, 0, 0.3, 0.6}` and
`b1, b2 ∈ {−0.6, 0.6}` (others zero), which is 9 runs. Record for each run:
whether it finished, the minimum dt, and the multiplier extrema and saturation
fraction (the `model_discrepancy` attribute of `state.nc`). Negative
coefficients lower the viscosity, and 0.67 × 0.24 ≈ 0.16 is in the range where
the uncorrected closure diverges. If any negative value diverges, repeat P1 at
`CV=0.30` and use that `CV` for truth and model in everything below. Choose
`σ_b` so that ±3σ_b stays inside the stable range.

**P2, signal-to-noise.** Do this with inlet turbulence on (`INLET=true`, the
realistic setting), and also off if the on-case fails the gate. Compare 20 s
sensor means between pairs of runs:

| Pair | Measures |
|---|---|
| b = 0 vs b = candidate `b*` | SGS signal |
| b = 0, truth inlet seed A vs seed B (`model.forward_model.inlet_turbulence.seed`) | realisation noise |
| b = 0 vs inflow +5° and +0.5 m/s | inflow signal, for scale |

Report `D = RMS(difference) / σ_eff` per pair, per layout (L1, L2, validation).
`σ_eff` is the effective std of one aggregated observation. Take it from
`make_observation_error` / the run's `C_D`, not from a hand estimate (expect
roughly 0.06 m/s).

Start from `b* = [0.3, −0.3, 0.2]` with the signs chosen inside the stable
range. Keep `|b0|+|b1|+|b2| ≲ 0.7` so that most cells stay out of saturation.

**Gate.** Proceed when, for at least one layout, `D_sgs ≥ 2` and
`D_sgs ≥ 2·D_seed`. If neither holds, fix it in this order:

1. Larger `b*` within the stable, unsaturated range.
2. Inlet turbulence off. This also turns interior nudging on above 16 m, which
   damps the top of the band; note that.
3. Report that the correction is not observable on this case and stop.

Freeze `CV`, `σ_b`, `b*` and the inlet setting in the report **before** any DA
run.

**Truth replicas (the noise floor).** For each truth configuration the DA runs
use, make one forward run with the frozen truth parameters and only the inlet
seed changed: `model.forward_model.inlet_turbulence.seed=<other>`,
`time.simulation_time=540` (3 windows × 180 s), the same spinup. Its scores
against the truth are the best a perfect model can do. With inlet turbulence
off, only uDALES' initial perturbation differs between runs. Set
`model.forward_model.per_member_irandom=true` and another
`model.forward_model.experiment_name` (e.g. `"997"`, not the driver's `998`).
The seed derives from that name, and the inline truth uses `"999"`. Verify
that the two runs actually differ. These runs are cheap; make them
alongside the DA runs.

### 2. Same-model DA runs (smoother)

Stage A is the deterministic twin (`INLET=false`) if P2 needed it; otherwise
skip straight to Stage B (`INLET=true`). Within a stage, the tests are:

| Test | Truth b | Prior SGS | Prior inflow | Estimate | Question |
|---|---|---|---|---|---|
| T1 SGS only | `b*` | N(0, σ_b) | `PIN_INFLOW` | `EST_SGS` | Is SGS identifiable at all? |
| T3 joint | `b*` | N(0, σ_b) | prior | `EST_ALL` | The main test |
| T2 inflow only | `b*` | `NO_SGS` | prior | `EST_INFLOW` | Uncorrected baseline for T3 |
| T0 zero control | 0 | N(0, σ_b) | prior | `EST_ALL` | Does the fit invent a correction? |

Run in this priority order, and stop adding runs when the night runs out:

1. T1-L2 (pilot seed). **Gate:** if T1 fails to recover, stop the DA runs and
   spend the remaining time on a diagnosis. Look at the per-step
   `window_*_obs.nc` misfit, prior-ensemble sensitivity (the correlation of each
   `b` with the predicted observations) and saturation. A joint run cannot
   succeed where T1 fails.
2. T3-L2, T2-L2, T0-L2.
3. T1-L1 and T3-L1, which give the layout effect.
4. Seeds 2 and 3 for T1-L2 and T3-L2. Vary `assimilation.seed`,
   `prior_params.seed` and the truth inlet seed
   (`truth_model.forward_model.inlet_turbulence.seed`) together. Member inlet
   seeds are derived from member names, so they are the same in every run;
   record that.
5. Hybrid T3-L2 (`scripts/run_hybrid.py`; `filtering.mode=state` is required
   and is the default). It checks that the estimated coefficients survive the
   state filter.
6. Optional: joint filter T3-L2 (`filtering.mode=joint`,
   `filtering.parameter_evolution=null`). The earlier standalone filter did not
   recover the coefficients. Run it only to document that, not to tune it.

**Cost.** One smoother run makes about `3 windows × 5 forecasts × 32 members =
480` member-window forecasts. Time one P1 run (`t_1`) and estimate
`480 · t_1 / NPP`. Plan the night from that, and report the actual wall time per
run (from `run_info.yaml` or the log).

### 3. What to record per run

From `metrics.yaml` (`compute_metrics.py` as it is today), per window:

- Parameter posterior RMSE, CRPS and reduction against the prior, for every
  parameter.
- **`sensor_statistics`:** the CRPS of window means and variances at the
  assimilated and validation sensors, posterior and prior. The prior for
  windows 1 and 2 is the forecast. These are the scores to use.
- The instantaneous `sensors` RMSE and `spread_skill` blocks, as sanity
  checks only. With different turbulent realisations they mostly measure
  unpredictable eddies.
- Innovation χ² and the Desroziers ratio.

From the window parameter files: the posterior mean ± std of each parameter
against the truth, and the posterior correlation matrix. The correlations show
which combinations are constrained. Also record solver failures, multiplier
saturation per member (`model_discrepancy_by_member`) and wall time.

## Acceptance criteria (frozen before running)

For each coefficient, use the final window:

- **Recovered:** `|mean − truth| ≤ 2·std` and `std ≤ 0.5·prior std`.
- **Unconstrained:** `std > 0.5·prior std`. This means not identifiable with
  this layout; it is not a bug.
- **Wrong:** constrained but more than 2·std from the truth. This is the result
  to investigate first.

The tests pass when:

| Test | Passes when |
|---|---|
| T1 | `b0` and at least one of `b1`, `b2` are recovered; none is wrong |
| T0 | Every `b` is within 2·std of 0; inflow errors are comparable to T3's |
| T3 vs T2 | T3's inflow error ≤ T2's. T3's validation-sensor CRPS of window means and variances (`sensor_statistics`) is lower than T2's by at least 10%, both for the posterior and for the window-1/2 forecasts. No `b` is wrong. |
| All | No solver failures; saturated band cells under about 5% |

When the metrics plan lands, also judge T3 vs T2 on field statistics, canopy
profiles, sensor distributions (W2) and spectra. A difference counts only if it
is clearly larger than the gap between the replica and the truth. With 1–3
seeds this is evidence, not a statistical claim. Say so in the report.

## Deliverables

On the branch `exp/sgs-twin-tests` (never main), commit
`docs/research/sgs_twin_tests_2026-10/report.md` containing:

- Machine, commit, and the frozen pilot choices with the P1/P2 tables.
- A run table: test, layout, seed, status, wall time, and the exact command.
- A per-run parameter table (truth / prior / posterior mean ± std / verdict)
  and the sensor and forecast scores.
- Failures, and anything that surprised you.
- A one-paragraph verdict on execution, recovery and benefit.

Commit compact figures only if they are small. Push the branch; do not open a
PR unless asked. If an actual bug turns up, fix it on a separate branch: test
it, run `pre-commit`, and open a PR.

## Round 2 outline: PALM truth, uDALES model (not tonight)

Run this only if Round 1 shows at least recovery in T1/T3.

**The objective changes.** The aim is the best possible uDALES prediction of
the PALM flow, not the "correct" uDALES coefficients: none exist. The SGS
coefficients mean different things in the two solvers. The DA should find
whatever values make uDALES resemble PALM most closely, as in a real
deployment where the best coefficients for reality are unknown. Consequences:

- **Judge predictions only.** The primary scores are the ones the DA never
  sees: held-out validation sensors, the window-1/2 forecasts after the
  coefficients are frozen, and the full-field state RMSE against PALM where
  the grids allow it. Check how `compute_metrics.py` samples a PALM truth on
  the uDALES grid before trusting the state RMSE. Calibration (spread–skill,
  χ²) is secondary.
- **The coefficients are diagnostics, not targets.** There is no "recovered"
  or "wrong" verdict for `b`. Report their posteriors, and check that they are
  consistent across windows, seeds and sensor layouts. Coefficients that drift
  from window to window, or flip with the layout, are fitting noise rather than
  a model correction. Also report saturation, and whether the posterior piles
  up at the edge of the prior or at the cap. If it does, the correction wants
  more than the prior allows: widen σ_b within the Round 1 stable range, but
  decide that on the calibration windows only.
- **Inflow is partly effective too.** PALM's inflow values are known, but the
  inlet profiles and turbulence differ between the solvers, so the uDALES
  inflow estimates are not exact targets either. Report their error against
  the PALM values as a diagnostic. Also report whether estimating SGS moves the
  inflow estimates toward or away from them, because SGS and inflow can
  compensate for each other.

**Comparisons:**

- **T2:** inflow only, `b = 0`. This is the uncorrected baseline.
- **T3:** inflow and SGS.
- **Estimated `sgs_constant`, `b = 0`:** the simpler closure correction, per
  step 4 of the validation plan. The 3-coefficient correction is only worth it
  if it beats this.
- **Optional, a best-achievable reference:** a parameter-only ESMDA with dense
  observations (many sensors or full fields) over the calibration windows.
  Its predictive skill is roughly the ceiling for this correction. The gap
  between the sparse-sensor T3 and this reference separates "the correction
  cannot mimic PALM" from "the sensors cannot pin it down".

**Passes when:** T3 beats both T2 and the `sgs_constant` baseline on held-out
sensors and on forecasts after assimilation stops. Use the same ≥10% threshold
as Round 1, and calibration must not get worse. The coefficients must also be
stable across seeds and windows.

**Setup:**

- **Generate the truth once:** `scripts/run_forward.py model=pypalm
  case=xie_and_castro params=static_truth` with
  `time.simulation_time = spinup + 3·180`. PALM's `ncpu` must divide `nx=30`.
  PALM keeps its own SGS closure (`sgs_constant: null`). Check that PALM ignores
  the `sgs_bias_*` entries in `static_truth` (remove them with `~` if it
  doesn't). Then point the DA at it with `assimilation.truth_dir=<run dir>` and
  `assimilation.truth_start_time=<spinup>`.
- **Align the solvers:** inlet turbulence differs (PALM disturbances vs the
  uDALES driver planes), and so do the spinup, the grid staggering of the
  sensor sampling, and the time origin. Read [pypalm.md](../pypalm.md) first.
- **Freeze before scoring:** start from the Round 1 feature settings,
  observation errors and σ_b. Any change, such as a wider σ_b, is made on the
  calibration windows and assimilated sensors only, and is frozen before the
  held-out sensors and forecasts are scored.

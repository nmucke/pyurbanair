# SGS model discrepancy — validation and transfer plan

Status: proposed next work; no experiments or implementation are performed by
this document. Written 2026-09-29 against commit `53ac02f` on
`feat/sgs-discrepancy-esmda` (PR #149).

This continues [the original implementation plan](model_discrepancy_implementation.md).
That plan's opening status and some configuration restrictions are historical.
Use the maintained [uDALES](../pyudales.md),
[data assimilation](../data_assimilation.md), and
[scripts/configuration](../scripts_and_configs.md) documentation for current
contracts, and verify them against the checkout before implementing.

## Objective and next milestone

Establish whether the existing three-coefficient correction gives reproducible
predictive improvement, with identifiable parameters and acceptable numerical
and physical behavior. Complete controlled same-model validation before testing
transfer to PALM-generated truth.

The immediate deliverable is a reproducible validation suite and a report
covering solver continuation, sensitivity, known-error recovery, and forecasts
after assimilation stops. A negative result is a valid scientific outcome;
completion of the implementation must not be presented as evidence of benefit.

## Current starting point

Implemented:

- Automatic preparation and selection of the extended uDALES Vreman solver;
  fixed `sgs_bias_b0`, `sgs_bias_b1`, and `sgs_bias_b2` in forward runs.
- Native viscosity refresh on enabled startup, window-start checkpoints,
  deterministic replay, and accepted posterior forecast carry.
- Static parameter ESMDA, parameter/joint filtering, and static ESMDA followed
  by state-only filtering. Hybrid likelihood allocation and native carry
  handoff are implemented.
- Joint estimation of SGS and other scalar parameters. Explicit priors live
  in the regular parameter YAML files. `params_to_estimate: null` includes all
  configured parameters and the three SGS coefficients; `Constant` entries
  still have no ensemble spread. `prior_std` is only a missing-prior fallback.
- Three small recipes, evaluation support, and native workflow tests in both
  memory and disk modes, including mixed physical/SGS parameter estimation.

Existing same-model examples use a 32×24×16 cropped Xie–Castro domain, 12
members, two workers, 10 s spinup, and a 20 s assimilation window. Injected
coefficients are `[0.10, -0.12, 0.08]`. The initial ESMDA example recovered
approximately `[0.090, -0.103, 0.063]` and reduced held-out component velocity
RMSE from 0.0075 to 0.0031 m/s. The hybrid preserved its estimated coefficients
through state filtering. The standalone filter completed but did **not** recover
the coefficients. These are single-seed demonstrations, not acceptance results
for the work below.

Still unverified or incomplete:

- Continuous versus segmented native forecasts and their restart error.
- Discrete energy and wall-flux budgets, including nonzero scalar top flux.
- Robust recovery across seeds, sensor layouts, and useful forecast horizons.
- Identifiability when SGS and physical parameters are estimated together.
- Benefit over simpler corrections at comparable computational cost.
- Transfer to unseen forcing and another solver's truth.

## Scope to preserve

Keep the existing correction formula and persistent coefficients. Do not add
time-varying SGS coefficients, OU evolution, `b3`, another closure, or another
assimilation backend as part of this milestone. Dynamic forcing with SGS
inference remains a separate integration task; placing coefficients under
`static_parameters` in a dynamic YAML does not make that runner mode supported.

Keep mixed parameter estimation available. Fixed-forcing experiments below are
experimental controls, not new validation restrictions on user configurations.
In particular, do not reintroduce an exactly-three-parameters requirement.

Keep the regular prior/truth YAMLs. Put experimental values and parameter
selection in experiment overrides; do not recreate `sgs_bias_truth.yaml`.
Reuse generic assimilation and evaluation code wherever possible.

Hybrid state-only filtering and `ensemble.failure.policy=raise` remain the
supported discrepancy configuration. Joint hybrid coefficient updates,
cross-stack failure donors, and resuming interrupted assimilation jobs are
separate work.

## 1. Establish a reproducible validation protocol

Start from the existing small recipes, preserving them as quick examples:

```bash
pixi run -e dev python scripts/esmda/run_esmda.py experiment=esmda/sgs_bias_small
pixi run -e dev python scripts/filtering/run_filtering.py experiment=filtering/sgs_bias_small
pixi run -e dev python scripts/filter_smoothing/run_filter_smoothing.py experiment=filter_smoothing/sgs_bias_small
```

Add validation-specific experiment overrides and a small batch/report workflow
only where the current scripts cannot already do the job. Any new executable
follows `run(cfg)` plus a thin Hydra wrapper. Keep outputs in ignored result
directories, not in Git.

Use one inexpensive pilot seed to choose duration, sampling cadence, stable
prior ranges, and a feasible compute budget. Then freeze these choices before
evaluation. Start with five independent evaluation seeds and two sensor layouts
(the existing layout and a sparser/height-restricted layout). Expand only if the
uncertainty in the comparison warrants the extra cost. A seed must identify
prior sampling, observation noise, assimilation perturbations, and stochastic
inlet realizations separately; record the actual random sources used.

Separate calibration observations, spatially withheld sensors, and future
times without assimilation. Do not tune priors, `L`, height band, regularization,
observation errors, or inflation on evaluation seeds or withheld outcomes.
The existing 10 s spinup and 20 s window are demonstration settings: check
stationarity and relevant flow timescales before adopting them for turbulence
statistics or longer-term recovery claims.

Record each run's resolved config, source/build provenance, seed identifiers,
parameter selection, observation split, solver failures, number of member
forecasts, simulated member-seconds, and wall-clock cost. Record ESMDA replays,
spinup, hybrid phases, and posterior/free forecasts in that cost accounting.

**Acceptance:** a saved protocol specifies the run matrix, primary scores,
numerical tolerances, practically meaningful improvement threshold, and allowed
physical-budget degradation before evaluation starts. Publish failed runs and
inconclusive results along with successes. Do not quietly retune the protocol
after seeing evaluation results.

## 2. Close the native continuation and physics gaps

Extend the existing native tests instead of replacing their established checks.

### Continuous versus segmented forecasts

From a common initialized native state, compare a continuous forecast over
`T1 + T2` with two forecasts over `T1` and `T2`, with no analysis, identical
forcing, and identical coefficients. Exercise zero and nonzero coefficients,
one and two MPI ranks, and relevant periodic/inflow boundary configurations.
Align physical output times and compare all velocity components, available
pressure/scalar fields, native clocks, and timestep/viscosity diagnostics.

Use stock and enabled-zero controls to distinguish native restart error from
extension-specific error. Enabled startup refresh can change adaptive timesteps;
do not demand bitwise equality with stock or independently initialized MPI
decompositions. Derive tolerances from controlled numerical checks, freeze
them, and report both absolute and normalized errors. Do not merely loosen
tolerances until a failing continuation test passes.

### Stability and discrete budgets

Exercise coefficient changes and analyzed-state changes separately and together
at warm starts. Include the chosen calibration range and deliberate approach to
the configured multiplier cap. Check finite positive viscosity, bounded
multiplier, timestep collapse, saturation, and the first-step refresh.

Measure discrete momentum and kinetic-energy balances using native fluxes and
stencils, with pressure work, forcing, transport, molecular/SGS dissipation, and
boundary contributions accounted for as appropriate. Report budget residuals
and their timestep/grid sensitivity. The correction relative to native SGS may
reduce dissipation; continuum dissipativity does not prove a discrete budget.
Do not impose zero net momentum change in cases with wall or boundary stresses.

Add a focused nonzero scalar-top-flux check. Startup scalar gradients may depend
on diffusivity, so verify refresh ordering and boundary consistency explicitly.
Distinguish preservation of scalar/molecular closure at the **same input state**
from indirect scalar changes caused by an evolving corrected velocity field.
If extra native diagnostics are necessary, keep them opt-in and preserve the
disabled path; update extension resources, hashes, and build provenance together.

**Acceptance:** continuation errors meet the recorded tolerances, instability
and saturation behavior are understood over the declared range, and budget
residuals have a defensible numerical explanation. Any failing boundary regime
must be fixed or explicitly excluded from the validated scope before transfer.

## 3. Measure sensitivity and joint identifiability

At a fixed replay checkpoint, perturb each selected parameter while holding
initial state, inlet realization, and observation sampling fixed. Use central
finite differences at multiple step sizes to distinguish actual sensitivity
from numerical noise. Respect parameter bounds; report any one-sided estimates.

For predicted observations `h(theta)`, form the prior-scaled,
observation-whitened sensitivity matrix:

```text
J[:, j] = derivative of h with respect to theta[j]
A = R^(-1/2) J D_prior
```

`R` is the actual observation-error covariance in the assimilated observation
space, including aggregation/correlation where applicable. `D_prior` contains
the recorded prior scales for the independent priors used here. Report physical
likelihood sensitivity separately from any algorithmic tempering.

Save singular values, parameter combinations in weakly constrained directions,
column norms, and posterior correlations. Examine these selections in order:

1. Three SGS coefficients with physical parameters fixed.
2. SGS coefficients plus inflow angle and speed.
3. All scientifically active configured scalar parameters, explicitly including
   `sgs_constant` alongside `sgs_bias_b0` as a confounding experiment.

Audit whether each physical parameter actually controls the chosen case.
For example, a pressure-gradient parameter can have little or no effect under
an inlet-driven configuration. Do not equate inclusion in a Dataset with
identifiability. Give fitted parameters nonzero prior spread.

**Acceptance:** report which coefficients or combinations the observations can
constrain and how that changes with sensor layout. Weakly identified directions
should motivate reduced-fit comparison cases or additional observations, while
leaving the general joint-estimation capability intact.

## 4. Validate known-error recovery and predictive benefit

Use the frozen protocol to compare the following correction choices within
each applicable workflow:

| Choice | Purpose |
|---|---|
| Native closure, fixed SGS constant | Uncorrected reference |
| Estimated native SGS constant, zero discrepancy | Simpler scalar closure correction |
| Native closure with calibrated inflation only | Existing uncertainty treatment |
| Three SGS coefficients, fixed native SGS constant | Proposed correction |
| SGS coefficients plus selected physical parameters | Joint estimation sensitivity |

Use a stock/disabled forward control where useful for native equivalence. For
an assimilation baseline that needs common replay semantics, enabled-zero
coefficients are an explicit control; label that choice in the report. Configure
baselines through already supported modes rather than weakening mode checks
just to force an empty parameter estimation problem.

Hold truth, observation realization, likelihood, and common parameter priors
fixed within paired comparisons. Inflation-only means changing the supported
ensemble inflation mechanism, not changing the observation-error model.
Calibrate any inflation settings only on the pilot/calibration data.

Compare at matched member-forecast/simulated-time budgets where possible and
report actual costs. Different algorithms consume observations and forecasts
differently: compare corrections within an algorithm first, then show cost/skill
across algorithms. Preserve the hybrid's shared likelihood budget.

Run truth scenarios in this order: injected SGS error only; physical-forcing
error only; then combined error. Include a zero-injected-discrepancy control to
check whether the fit invents a correction or degrades an already correct model.
For same-model truth, do not let shared spinup or inlet randomness accidentally
remove the uncertainty a particular experiment claims to test.

Evaluate parameter error and interval coverage where truth parameters exist,
withheld velocity RMSE and a proper ensemble score such as CRPS, posterior
correlations, ensemble spread/coverage, mean flow, and Reynolds stresses where
the averaging duration is adequate. Track dissipation, saturation, and solver
failures. Interpret statistical uncertainty at the independent-seed level;
time samples and ensemble members are not independent experimental replicates.

After the last analysis, continue each member from its accepted native endpoint
with its final parameter vector and **no further observation updates**, for at
least an assimilation-window duration in the initial protocol. Save forecast
lead time explicitly. This tests the complete posterior forecast, including its
state estimate. If attribution to coefficients is needed, add a separate
common-initial-state comparison and label its different question.

Investigate the existing standalone filter's poor coefficient recovery using
the forecast-used versus analyzed histories, sensitivity results, spread, and
window length. Change tuning only on calibration data. Do not claim filter
recovery merely because it runs successfully or because ESMDA succeeds.

**Acceptance:** the report distinguishes execution, parameter recovery, and
predictive skill. A claim of benefit requires the preregistered practical
improvement on withheld and assimilation-off forecasts across evaluation seeds,
acceptable calibration, and no unacceptable physics/stability degradation.
Failure to meet that gate triggers a documented diagnosis or narrower model;
it does not justify automatically adding more coefficients.

## 5. Controlled transfer to PALM-generated truth

Begin only after the preceding results support proceeding. First test held-out
forcing conditions with same-model truth; then change the truth solver to PALM
while retaining uDALES as the assimilation model. PALM needs no SGS discrepancy
implementation for this experiment.

Read the PALM backend documentation before preparing these cases. Align geometry,
physical units, time origins, forcing conventions, sensor coordinates, and
observation extraction across the solvers. Quantify grid/interpolation and
spinup differences. Keep observation-error assumptions fixed unless a separately
documented calibration exercise establishes a new comparison protocol.

Freeze priors, feature definitions, height band, regularization, cap, and
algorithm tuning before opening transfer evaluation results. Coefficients may
be **estimated from the designated transfer calibration observations**; they
need not equal same-model fitted values. Evaluate distinct held-out sensors
and future assimilation-off periods. If testing literal reuse of fitted
coefficients without recalibration, make that a separate, explicitly labeled
experiment.

Reuse the simpler baselines and cost accounting from step 4. Cross-model truth
has no ground-truth SGS bias coefficients, so omit coefficient-accuracy scores.
Judge predictions, ensemble calibration, stability, and budgets instead.

**Acceptance:** demonstrate improvement on held-out transfer predictions at
frozen tuning and comparable cost, without unacceptable calibration or physical
degradation. If this fails, retain the negative result and state the supported
same-model scope; no universal coefficient interpretation follows.

## Delivery sequence and ownership

Deliver the work in reviewable increments:

| PR | Deliverable | Completion evidence |
|---|---|---|
| A — native validation | Continuation tests and missing boundary/budget diagnostics or fixes | Native test results, tolerances, budget report, updated validated scope |
| B — sensitivity and recovery | Reusable sensitivity/report tooling, validation overrides, seeded same-model comparisons and free forecasts | Frozen protocol, machine-readable metrics, figures, explicit pass/fail/inconclusive conclusions |
| C — transfer | Held-out forcing and PALM-truth experiments using frozen tuning | Comparable baseline results and a transfer report |

Protocol design and non-native evaluation tooling can proceed alongside PR A;
expensive validation runs and transfer should wait for their prerequisite gates.
Do not make an independent framework for each assimilation algorithm.

If a team is requested for implementation, a suitable split is a strong agent
for native continuation/physics, a strong agent for sensitivity and experimental
design, and a smaller agent for configuration composition, artifact checks,
and documentation. Keep a primary integrator responsible for the shared
protocol, review, and final claims; separate file ownership where possible.

Store compact experiment manifests, commands, summary tables, and interpretation
in a versioned report. Keep large NetCDF/solver outputs ignored and reference
their paths/provenance. Reuse `libs/evaluation` for metrics/figures; do not pull
solver dependencies into that leaf library. Update maintained documentation
when behavior or validated scope changes.

## Next-session checklist

1. Check whether PR #149 has merged and select the appropriate base before
   branching. Preserve the user's local case/model/parameter edits and submodule
   changes; do not treat them as validation defaults or commit them incidentally.
2. Read this plan, the original formula/solver contract, and the relevant
   maintained docs. Audit the existing tests before declaring a gap still open.
3. Start PR A with continuous-versus-segmented forecasts and the nonzero
   scalar-top-flux/refresh check. Reproduce existing native baselines first.
4. Draft and save the step-1 protocol; implement only missing diagnostics and
   reporting support. Freeze evaluation choices after the pilot.
5. Run focused fast tests and explicitly selected `integration` tests. Preserve
   `forkserver`, conservative worker counts, automatic solver preparation, and
   disabled-path behavior. Run pre-commit before committing.
6. Review the evidence before moving from native validation to recovery claims
   and from same-model recovery to transfer.

Relevant existing tests include `test_udales_discrepancy_physics.py`,
`test_udales_discrepancy_native.py`, `test_udales_discrepancy_warmstart.py`,
`test_udales_window_replay.py`, `test_esmda_replay.py`,
`test_discrepancy_workflows.py`, `test_discrepancy_inference_config.py`, and
`test_sgs_bias_recipe_configs.py`, all under `tests/`. The shared demonstration
config is `conf/experiment/sgs_bias_small_common.yaml`; workflow overrides are
under the `esmda`, `filtering`, and `filter_smoothing` experiment groups.

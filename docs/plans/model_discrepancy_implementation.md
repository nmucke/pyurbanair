# Plan 4 — Explicit dynamical model discrepancy

Status: proposed implementation; no algorithm or solver changes made.
Implements recommendation 4 in [the review](../data_assimilation_recommendations.md).
Related plans: [observation likelihood](observation_likelihood_implementation.md)
and [hybrid beta](hybrid_beta_tempering.md).

## Outcome and first deliverable

Estimate a small, physically interpretable momentum correction separately from
physical forcing and SGS parameters. Start with **uDALES assimilation**, using
same-model truth with known injected discrepancy; transfer to PALM truth after
that works. Add a PALM assimilation implementation later through the same
physical contract. Unsupported backends must reject enabled discrepancy.

The first source model is:

```text
du/dt = F_u(x, θ) + Σ φ_j(z) b_u,j(t)
dv/dt = F_v(x, θ) + Σ φ_j(z) b_v,j(t)
```

Use one or two smooth vertical basis functions per horizontal component;
start with no vertical forcing. Basis functions are dimensionless, evaluated
on the native staggered grid, with recorded normalization and physical-height
support. Coefficients have units **m s⁻²**. Evaluate the source during solver
integration, not as a velocity adjustment after a complete forecast window.

Begin with persistent coefficients. Add an OU process for time-varying error in
a later milestone, then permit persistent mean plus OU fluctuations only if
sensor sensitivities distinguish them. Do not simultaneously fit an unrestricted
bias field, forcing trajectory, closure constants and process hyperparameters.

## A. Define the shared contract and deterministic uDALES source

Add `discrepancy.py` under `libs/data-assimilation/src/data_assimilation/` for
basis metadata, coefficient priors and evolution. Solver-independent schedule
validation/serialization belongs in `src/pyurbanair/` or backend utilities so
CFD backends do not acquire a dependency on the DA library. Store coefficients
in the parameter Dataset under explicit names such as `model_bias_u_0`.

Proposed opt-in configuration:

```yaml
model_discrepancy:
  enabled: false
  basis: vertical_smooth
  modes_per_component: 1
  components: [u, v]
  coefficient_model: persistent  # later: ou
  prior_std: null               # required calibration, m/s²
  tau_seconds: null             # required only for OU
```

Basis support, normalization and coefficient limits must be explicit in the
resolved configuration. No universal amplitude or timescale is prescribed.
Missing/disabled configuration must preserve existing results and RNG streams.

Implementation steps:

1. Add a dedicated schedule extractor/writer beside uDALES's
   `utils/nudging_utils.py`. Existing `utils/params_utils.py` filters parameters
   through `INFLOW_PARAM_NAMES` and detects time variation only for inflow
   angle/speed. Handle discrepancy independently so its variables are neither
   dropped nor mistaken for inlet controls.
2. Add a Fortran module that reads a versioned member-local schedule once,
   interpolates coefficient values at the integration clock and adds
   acceleration to the horizontal momentum tendencies. The proposed call site
   is after `nudge` and before immersed-boundary enforcement in
   `libs/pyudales/u-dales/src/program.f90`. Verify RK-stage timing, staggering,
   MPI ownership and boundary handling against native forcing routines.
3. Do not multiply tendencies by the timestep; the integrator does that.
   Preserve immersed-boundary and pressure corrections. Check the realized
   momentum budget, since wall/flux corrections can modify the applied effect.
4. Add finite/range validation, explicit enable flags and stale-file cleanup.
   Disabled runs must never consume a previous member's source file. Do not
   repurpose `pressure_gradient_magnitude` or change nudging targets to hide
   this source: those have different physical meanings and existing conventions.
5. Deliver the external Fortran change in a maintained solver revision/patch and
   pin its provenance and build inputs in the Python repository. A local edit to
   downloaded solver code is not a reproducible implementation.

## B. Make repeated forecasts reproducible

Before ESMDA iteration, capture each member's window-start native restart,
hidden SGS fields, clocks and any stochastic-source state. Restore that
checkpoint before every replay, then inject that iteration's exposed initial
state if it is being estimated. Commit the accepted endpoint once. uDALES's
latest-run carry cannot automatically serve as every iteration's initial state.

A hybrid needs equivalent initial checkpoint copies for its separate smoother
and filter model stacks. Include coefficient paths, process RNG and checkpoint
identity in failure substitution: a cloned member must be consistent across
state, parameters and hidden fields. Preserve `forkserver` execution.

First verify identical-input replays and compare continuous versus segmented
no-analysis runs. Quantify restart transients before attributing improvements
to discrepancy estimation.

## C. Integrate filtering, then ESMDA

**Filtering.** Add discrepancy rows to the joint augmented vector, preserving
member identity. Support a parameter-update mask so physical forcing can be
held fixed while discrepancy is estimated. Otherwise current joint mode updates
every included parameter. Keep physical-parameter and discrepancy evolution
separate, with a composable evolution policy.

For OU coefficients, use the exact transition:

```text
rho = exp(-Δt/tau)
b_next = mean + rho (b_analysis - mean)
         + stationary_std sqrt(1 - rho²) ξ
```

Pass actual elapsed time into `ParameterEvolution.evolve`; the current API has
no duration. Preserve existing random-walk semantics via a compatibility
adapter. Analyze the coefficient used during the current segment, then evolve
for the next; save analyzed and next-forecast coefficients separately. Initial
filter support holds coefficients constant within a segment, so test timestep
refinement relative to `tau`. OU transition statistics are exact, but this
piecewise-constant forcing approximation is not an exact continuous OU path.

**ESMDA.** Sample a correlated discrepancy trajectory once per member/window
and augment its coefficients or knots using `TimeVaryingParameterESMDA` and
`ParamAugmentation`. Replay each proposed trajectory deterministically; never
redraw hidden process noise at every MDA iteration. Persist prior paths and RNG
provenance. Update prior/schema resolution so discrepancy can be estimated even
when the supplied truth has no corresponding parameter; do not force these
nuisance variables into `params_to_estimate`'s shared truth/prior contract.

Start with coarse knots and an OU-correlated prior. Inspect posterior roughness
and forcing/discrepancy sensitivity overlap. If needed, add an explicit
transition penalty or non-centred innovation controls; correlated prior samples
alone do not guarantee an exact posterior OU law after nonlinear ensemble
updates. Discrepancy modes should have explicit parameter/localization metadata,
not inherit one arbitrary grid-cell location.

## D. Connect the hybrid and transfer to another solver

First support an ESMDA-estimated discrepancy schedule followed by state-only
filtering. Extend `params_for_segment` schedule/rebasing tests to discrepancy
variables and carry the full native checkpoint between phases correctly.
Apply the beta plan's policy to the same error covariance; discrepancy process
uncertainty and likelihood tempering are independent controls.

Defer joint hybrid discrepancy updates until their meaning is explicit. Model
an evolving residual around the ESMDA baseline schedule, rather than applying
OU decay to baseline plus correction. Specify how the residual carries across
windows; current hybrid joint corrections reset. Do not both propagate a
residual and add it again when constructing the next prior. Retaining variation
within a segment needs a schedule adapter beyond the current midpoint-value
joint approximation.

After uDALES passes the gates below, implement an equivalent acceleration hook
for PALM. Transfer the basis definition and inference procedure; fitted bias
coefficients can depend on solver, resolution and closure. Keep each backend's
native closure as a baseline and use backend-specific SGS priors.

## Tests, experiments and delivery gates

| Milestone | Acceptance |
|---|---|
| Native source | Disabled baseline equivalence; correct acceleration sign/units; integrated momentum check |
| Schedule/restart | Correct clocks at segment/window boundaries; deterministic replay; consistent MPI decomposition results within numerical tolerance |
| OU evolution | Correct mean, stationary variance and lag covariance at unequal durations; distinct analysis/forecast artifacts |
| Known-error assimilation | Recover a low-dimensional injected source without compensating through incorrect forcing |
| Transfer | Improve held-out and assimilation-off forecasts without degrading calibration or energy/momentum budgets |

Add focused discrepancy unit tests and backend source/restart integration tests;
extend `test_model_error_parameters.py`, filtering, ESMDA, hybrid and runner
tests. Include failure-donor synchronization and disk/memory parity. Update
`docs/data_assimilation.md`, `docs/pyudales.md`, `docs/scripts_and_configs.md`
and relevant configs; update PALM documentation when that implementation lands.

Compare no discrepancy, existing parameter compensation, inflation-only,
persistent discrepancy and OU discrepancy, with fixed observation-error settings.
Use forcing-only, discrepancy-only and combined synthetic perturbations before
cross-model truth. Assess sensitivity rank and posterior parameter correlations;
reduce basis size if forcing and discrepancy cannot be distinguished. Tune on
calibration runs and freeze hyperparameters for held-out solver/forcing tests.

Deliver A–C first; D follows after demonstrated benefit. The established
[IEnKF-Q formulation](https://arxiv.org/abs/1711.06110) is a useful later
model-error-aware benchmark, not a prerequisite for this initial implementation.

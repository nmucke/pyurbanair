# Plan 4 — Strain/rotation-dependent SGS model discrepancy

Status: proposed implementation; no algorithm or solver changes made.
Implements recommendation 4 in [the review](../../research/data_assimilation_recommendations.md).
Related plans: [observation likelihood](observation_likelihood_implementation.md)
and [hybrid beta](hybrid_beta_tempering.md).

## Outcome and first deliverable

Estimate **three dimensionless coefficients** that correct uDALES's native
Vreman SGS eddy viscosity using local strain/rotation and height. This replaces
the earlier additive-acceleration proposal. Begin with same-model truth with a
known injected correction, then assess transfer to PALM-generated truth.
Keep physical forcing and the native global SGS constant fixed initially.

The first implementation uses only `b0`, `b1`, and `b2`. Coefficients are constant
during each solver forecast, while flow-dependent features are recomputed at
every native SGS evaluation. There is **no `b3*s` term**, strain-strength feature,
reference timescale, OU process, or within-forecast coefficient schedule in this
deliverable. These are later extensions, not dormant initial configuration.

**Deployment requirement:** a fresh clone must support the normal documented
Pixi setup and launch workflow without manual Fortran edits, patch commands,
solver-fork checkouts or executable-path changes. The full local correction
requires a native extension; pyudales must ship, prepare and select it
automatically. Existing scalar `cs`/`c_vreman` inputs cannot express it faithfully.
This assumes the documented platform/toolchain prerequisites.

## A. Mathematical and parameter contract

For resolved velocity `u`, define the native cell-centred gradient and invariants:

```text
G_ij = ∂u_i/∂x_j
S = (G + Gᵀ)/2,       Ω = (G − Gᵀ)/2
q = (Ω:Ω − S:S) / (Ω:Ω + S:S + ε_g²)
```

`A:A = Σ_ij A_ij²`. Choose a fixed, positive `ε_g` in s⁻¹, record it in run
metadata, and test sensitivity to its scale. Then `−1 < q < 1`: strain dominates
for negative values, rotation for positive values, and simple shear gives zero.
Zero gradient gives zero. Use nonnegative squared norms and finite validation.

Use one dimensionless height feature:

```text
φ(z) = sin²[π(z − z_a)/(z_b − z_a)]   for z_a < z < z_b
       0                            otherwise
```

The band targets rooftop exchange. The urban canopy is the building-occupied
layer, and `H` is an explicitly chosen representative building height. Configure
`z_a/H` and `z_b/H`, validate `z_b > z_a`, and evaluate at native viscosity
locations (`zf(k)` in uDALES). Record heights relative to the case's vertical
datum; do not infer `H` from the evolving flow.

The corrected SGS viscosity is

```text
g(x,t) = b0 + b1 φ(z) + b2 q(x,t)
ν_t^b(x,t) = ν_t^0(x,t) exp[L tanh(g(x,t)/L)],   L > 0
```

`ν_t^0` denotes the native SGS viscosity evaluated from the **current member's
current velocity field**, including native buoyancy adjustment where enabled.
It is not an initial-time field and excludes molecular viscosity. Both
viscosities have units m²/s; `b0`, `b1`, `b2`, `g`, and `L` are dimensionless.
The coefficients control global level, rooftop-band level, and strain/rotation
dependence, respectively.

The exponential keeps the multiplier positive; `tanh` smoothly bounds it between
`exp(−L)` and `exp(L)`. Near zero, the multiplier is approximately `1 + g`.
For illustration, `L = log(3)` permits factors from 1/3 to 3; this is not a
prescribed default. Select and freeze `L` using calibration/stability experiments.
Monitor saturation because it reduces parameter sensitivity. Zero coefficients
recover the native closure; zero native SGS viscosity remains zero.

The continuum momentum correction implied by this change is

```text
d_i = ∂_j[2(ν_t^b − ν_t^0) S_ij]
τ_ij^dev = −2ν_t^b S_ij
Π = −τ^dev:S = 2ν_t^b S:S ≥ 0
```

Apply the viscosity inside the existing stress-flux calculation for **all three
momentum components**. Multiplying an already computed diffusion tendency would
omit spatial derivatives of the multiplier. Leave molecular viscosity, scalar
diffusivity, pressure treatment and wall/immersed-boundary algorithms unchanged.
Scalar transport can still change indirectly as the velocity evolves.
The total SGS model is dissipative in the continuum; the correction relative to
baseline may reduce dissipation. It cannot model SGS backscatter or independently
rotate the stress. Discrete energy behavior and boundary stress fluxes still
require tests; do not assume zero domain-integrated momentum change at walls.

Proposed opt-in configuration (feature settings are required when enabled;
prior scales are required only for estimation):

```yaml
model_discrepancy:
  enabled: false
  kind: sgs_strain_rotation
  coefficient_model: persistent
  canopy_height: null             # H, m
  height_band_over_H: null         # [z_a/H, z_b/H]
  gradient_regularization: null    # ε_g, s^-1
  log_multiplier_cap: null         # L, dimensionless
  prior_std: null                  # three dimensionless standard deviations
```

Use explicit Dataset names `sgs_bias_b0`, `sgs_bias_b1`, `sgs_bias_b2` and a
zero-centred Gaussian prior with positive, calibrated scales. Start with a
diagonal prior covariance. Fix `c_vreman`: estimating it alongside `b0` creates
strong confounding. Fixed feature settings belong to configuration, not the
estimated parameter vector. Missing/disabled configuration preserves inputs,
RNG streams and numerical behavior. Enabled unsupported closures/backends must
fail clearly before CFD; first support Vreman only.

## B. Implement the native hook and Python wiring

The review inspected pristine uDALES commit
`b84916ac60cecd1da54dd09df76c15e30dcaabe9` (v2.2.0). Verify this revision in the
extension manifest. Relevant integration points are `src/modsubgrid.f90`
(`initsubgrid`, `closure`, `diffu/v/w`), `src/modboundary.f90` (`closurebc`),
`src/program.f90`, and `src/tstep.f90`.

1. Add an explicit enable flag, three coefficients and fixed feature settings
   to the strict `NAMSUBGRID` namelist, with matching MPI broadcasts and finite/
   range checks, including positive `H` and a cap whose exponentials are safe in
   the solver's real precision. Ship definitions and helper routines as project-owned source
   resources. No new arithmetic runs on the disabled path.
2. Reuse Vreman's nine cell-centred derivatives. Its `a_ij = ∂u_j/∂x_i` is the
   transpose of `G`, but produces identical squared strain/rotation norms.
   Compute the multiplier in that loop and retain it in optional scratch
   storage, or justify a second gradient pass. Allocate only when enabled;
   recompute from each member's flow at every closure call, including RK stages.
3. Apply the multiplier to interior turbulent `ekm` **after native buoyancy
   correction and `ekh = ekm*prandtli`, before `ekm += numol`**. This exact order
   preserves the scalar closure and molecular viscosity. Then retain native
   `closurebc` halo exchange/boundary filling and the existing stress divergence.
   Do not multiply ghost-cell boundary values independently or add a second
   source in `program.f90`. Check wall budgets rather than asserting that wall
   stresses themselves remain unchanged.
4. Test the native zero-gradient limit explicitly. Vreman currently evaluates
   `bb/aa` without a zero-denominator guard; regularizing `q` does not fix it.
   If reproduced, add a documented, tested zero-gradient limit for the enabled
   extension, with zero turbulent viscosity there. Keep any broader upstream
   robustness fix separate from the default-preserving implementation.
5. Add a dedicated pyudales discrepancy extractor/validator and namelist writer.
   The current `INFLOW_PARAM_NAMES` whitelist would otherwise drop these
   parameters. Carry discrepancy separately from inlet controls through merge,
   member creation and per-call application, and extend parameter-schema/prior
   resolution. Write all three coefficients per enabled call, including zeros,
   after preprocessing so old values cannot leak into a subsequent member/run.
   Disabling a previously enabled case must remove extension-only namelist keys
   before selecting the stock executable; an untouched default case is a no-op.
   Reject time-array coefficients initially instead of silently averaging them.
6. Keep backend serialization independent of `data-assimilation`. DA code owns
   inference/prior handling; pyudales owns native configuration. Support fixed
   coefficients in a forward run before adding estimation. Record coefficients,
   feature settings, multiplier extrema and saturation diagnostics in artifacts.

### Automatic source preparation and build

1. **Ship the extension.** Store the minimal checked patch, optional Fortran
   module and manifest under `pyudales/solver_extensions/discrepancy/`, included
   as package resources. The manifest identifies the exact upstream commit and
   resource hashes. No user-managed fork or edited submodule is required.
2. **Prepare an isolated source copy.** A helper such as `utils/solver_build.py`
   materializes the pinned pristine source in a writable, gitignored cache.
   Verify input hashes, apply the patch, and verify outputs. Leave the upstream
   checkout and developer changes untouched; do not copy uncommitted edits into
   the reproducible build. Unsupported revisions/conflicts fail before launch.
3. **Verify builds.** Key reuse by upstream/extension hashes, compiler, MPI,
   dependencies, flags and platform. Require a successful executable and matching
   capability manifest, not merely `CMakeCache.txt`. Check configure/build exit
   codes, including commands piped through `tee`. Lock concurrent preparation,
   publish completed builds atomically, and recover from corrupt/partial caches.
4. **Select automatically.** Refactor eager import-time building into normal
   preparation before ensemble workers start. Pass the chosen executable through
   `DirectoryPaths`, member copying and `DA_BUILD`; remove the fixed-path
   assumption in `utils/config_utils.py`. Build scripts accept source/build
   paths; preprocessing tool paths stay explicit. Enabled discrepancy must never
   silently launch a stock executable. Disabled runs retain stock behavior.
5. **Prove fresh-clone operation.** Submodule and direct-clone fallback resolve
   the same pinned commit. No later tag checkout may override it. Test normal
   Pixi setup/launch on supported Linux and macOS environments, empty caches,
   read-only package resources and simultaneous stock-truth/extended-assimilation
   variants. Record build provenance; subsequent launches reuse the matching
   build without recompiling for each coefficient sample.

## C. Restart safety and deterministic replay

Native restart files already store `ekm`; no new prognostic field or restart
format is needed for this algebraic correction. However, `tstep_update` runs
before `subgrid` and uses stored `ekm`/`ekh` to choose the timestep. A changed
coefficient or analyzed velocity can make that first estimate stale.

Before the first enabled forecast step, refresh viscosity from the final staged
state and coefficients, with valid velocity halos/boundaries, without advancing
time or applying momentum tendencies. Verify that calling/refactoring the
closure for this purpose has no unwanted side effects. If a safe refresh is
infeasible, implement and justify a conservative first-step bound; the bounded
multiplier alone is not sufficient when the velocity field also changes.
Test cold starts, warm starts, large allowed coefficient changes, and diffusion
stability throughout the forecast. Existing adaptive timestep logic remains
responsible for subsequent steps.

Before each ESMDA window, capture member-specific native restart, hidden fields
and clocks. Restore the same window-start checkpoint before every replay, then
inject any analyzed initial state and the current coefficients and refresh
viscosity. Accept the endpoint once. Latest-run carry must not become the next
ESMDA iteration's initial condition. Preserve member identity, checkpoint and
coefficients during failure-donor substitution; preserve `forkserver` execution.
Check identical-input replays and continuous versus segmented no-analysis runs.

## D. Integrate ESMDA, filtering and the hybrid

**ESMDA:** augment the existing parameter vector with the three coefficients,
constant over the entire assimilation window. Replay each proposed vector
deterministically. No coefficient process noise or trajectory knots are needed.
Support prior-only nuisance parameters even when cross-model truth has no
corresponding fields; do not require truth/prior parameter schemas to match.
Give these global coefficients explicit localization metadata rather than an
arbitrary grid-cell location. Hold forcing and the native SGS constant fixed in
the first recovery experiment, then assess sensitivity before joint estimation.

**Filtering:** add three parameter rows to parameter/joint augmentation and use
identity evolution initially. Each forecast segment holds coefficients fixed;
analysis may update them for the next segment. This is distinct from ESMDA's
one vector per full window. Support a parameter-update mask to hold forcing
fixed, and record forecast-used versus analyzed coefficients separately.

**Hybrid:** first estimate the three coefficients with ESMDA and use that same
vector through the window's state-only filtering phase. Ensure segment slicing
preserves static coefficients and both phases replay the intended checkpoint.
Apply the [beta plan](hybrid_beta_tempering.md) independently: discrepancy changes
the forecast model and does not solve repeated observation use. Defer joint
hybrid coefficient updates until their carry/reset semantics are specified.

## Tests, experiments and delivery gates

| Milestone | Acceptance |
|---|---|
| Mathematical kernel | Zero, pure-strain, solid-rotation and simple-shear gradients; correct units and height support; bounded finite multiplier; zero coefficients give unity |
| Native closure | Disabled baseline equivalence; enabled zero-coefficient comparison on finite native cases; molecular and scalar closure preserved at the same input state; spatial multiplier enters stress divergence correctly |
| Solver robustness | Zero-gradient Vreman handling; single-/multi-rank agreement within tolerance; periodic and upper-forced cases; wall/energy budgets and diffusive stability |
| Fresh clone/build | Normal setup/launch succeeds without manual edits; pristine source remains unchanged; repeat reuse, invalidation, concurrent preparation and interrupted-build recovery |
| Restart/replay | Coefficient/state changes refresh first-step stability inputs; deterministic window replay; continuous/segmented comparison; synchronized failure donors |
| Known-error assimilation | Recover identifiable injected coefficients and improve withheld predictions with physical forcing fixed |
| Transfer | Frozen tuning improves held-out and assimilation-off forecasts against another solver, without unacceptable loss of calibration or physical budget fidelity |

Use fast tests for feature algebra, parameter routing, package resources, patch
verification, cache invalidation and executable selection with subprocess
doubles. Mark fresh-clone compilation, MPI, native stress and restart runs as
`integration`. Extend existing model-parameter, ESMDA, filtering, hybrid and
runner tests, including disk/memory parity. When implemented, update maintained
`docs/pyudales.md`, `docs/data_assimilation.md`, `docs/scripts_and_configs.md` and
relevant configs; this plan alone does not change current runtime contracts.

Compare the native closure, an estimated global SGS constant, inflation-only,
and the three-coefficient correction at matched ensemble/forward-run budgets
and fixed observation-error settings. Use same-model injected discrepancy first,
then forcing-only and combined perturbations, followed by cross-model truth.
Inspect prior-scaled, observation-whitened sensitivity singular values and
posterior correlations; reduce fitted coefficients if observations cannot
distinguish them. Evaluate held-out sensors, free forecasts, mean flow, Reynolds
stresses, dissipation and ensemble coverage. Freeze priors, feature settings and
`L` before held-out forcing/solver tests; training-observation fit alone is not
evidence of transferable improvement.

## Later extensions

Only after the three-coefficient Vreman implementation passes these gates:

- Add Smagorinsky support with consistent full-gradient stencils; its current
  strain calculation alone does not supply rotation.
- Add `b3*s`, with `s = T_ref² S:S / (1 + T_ref² S:S)` and a fixed, documented
  `T_ref`, if it contributes identifiable predictive information. Add its prior,
  configuration and tests at that time; none are required initially.
- Introduce time-varying coefficients/OU priors only with explicit elapsed-time,
  replay, schedule and hybrid residual semantics. Do not emulate continuous
  variation through undocumented extra restarts.
- Implement an equivalent SGS correction in another assimilation backend using
  the same automatic delivery principle. PALM-generated truth does not require
  modifying PALM. Transfer feature definitions and inference methodology, not an
  assumption that fitted coefficients are universal across closures or grids.

Delivery order: automatic build plus fixed-coefficient Vreman forward runs;
restart/physics verification; three-parameter ESMDA and filtering; state-only
hybrid integration; controlled transfer experiments. Scientific benefit remains
to be demonstrated even though the source review found a feasible solver hook.

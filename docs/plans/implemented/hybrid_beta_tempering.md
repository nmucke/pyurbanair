# Plan 3 — Beta tempering for the ESMDA–EnKF hybrid

Status: proposed implementation; no algorithm changes made.
Companion plans: [observation likelihood](observation_likelihood_implementation.md)
and [model discrepancy](model_discrepancy_implementation.md).

## Outcome and mathematical definition

Introduce **beta as a multiplier of the state-filter observation covariance**,
analogous to ESMDA's alpha:

```text
R_filter = beta R,              beta >= 1
L_filter ∝ L^(1/beta)
K = C_xy (C_yy + beta R)^(-1)
```

Larger beta weakens each filter analysis. Stochastic observation perturbations
must have covariance `beta R` and standard deviation `sqrt(beta) σ`; multiplying
their standard deviation by beta would be incorrect. ETKF/LETKF must use the
same effective covariance. This does not alter the observed data or physical
instrument noise, and is distinct from ensemble-spread inflation and
localization's existing `tapering_beta`.

**Limitation:** beta controls observation influence; it cannot guarantee absence
of overfitting or repair all state–parameter dependence in the current hybrid.
Introduce two explicitly named policies to separate those purposes.

## Policy 1: filter-only tempering, preserving current behavior

```yaml
filter_smoothing:
  beta: 1.0
  likelihood_allocation: filter_only
```

ESMDA keeps its normalized schedule `Σ 1/alpha_i = 1`; the filter uses `beta R`.
The default `beta=1` reproduces current runs. For repeated identical likelihood
factors, the nominal combined exponent is `1 + 1/beta`, so no finite beta makes
this a single full likelihood. This is useful conservative damping, not exact
likelihood accounting. Existing aggregate-ESMDA/raw-filter runs remain supported
under this explicitly heuristic policy.

## Policy 2: shared likelihood budget, opt-in research mode

To reserve part of the observation influence for filtering, reduce ESMDA's share
as well. Define a normalized base ESMDA schedule and derive:

```text
w_filter = 1/beta
w_esmda  = 1 - 1/beta
alpha_effective_i = alpha_base_i / w_esmda

Σ 1/alpha_effective_i + 1/beta = 1
```

```yaml
filter_smoothing:
  beta: 2.0
  likelihood_allocation: shared_budget
esmda:
  interval_seconds: null
```

With four steps and beta 2, the base alpha remains 4, effective alpha becomes 8,
and the filter uses `2R`: half the nominal budget goes to each phase. Beta 4
assigns three quarters to ESMDA and one quarter to filtering. This accounting
is **per reused observation**, not divided by the number of distinct filter
cycles. Repeated windows must not accidentally consume boundary observations
again.

Initial shared-budget support requires finite `beta > 1`, no smoother
aggregation, the same selected raw observations/operator, and compatible base
covariances in both phases. Validate identities and timestamps, not merely
vector lengths. The current default aggregation must be explicitly disabled.
Reject correlated temporal errors until both phases implement the same joint
or conditional likelihood. Do not silently change the observation product.

Reject beta 1 in shared mode: it assigns zero budget to the entire parameter
stage. Explicit phase-skipping endpoints, including a filter-disabled limit,
can be a later extension; do not pass infinite covariances through solvers.
Reject NaN, infinity and beta below 1 in both policies.
Compute the shared smoother weight as `(beta - 1)/beta` near beta 1 and
validate effective alpha/covariance in the actual computation dtype: finite
configuration values can still overflow or round to a zero weight.

Even with identical data, parameter-only ESMDA followed by state-only filtering
is not a general factorization of the joint posterior. In joint mode, nonlinear
reforecasting, localization and inflation also prevent a universal exactness
claim. Shared-budget mode enforces a nominal allocation and must be validated
as a hybrid approximation. A conditional-dual/joint-smoother redesign remains
separate work; see [Ait-El-Fquih et al.](https://hess.copernicus.org/articles/20/3289/2016/).

## Implementation sequence

### A. Add common weighting seams without changing analysis kernels

1. Add `beta=1.0` to `BaseFilter`/`EnsembleKalmanFilter`. Keep `C_D_diag` as
   physical covariance; derive the effective analysis vector separately.
   Pass it through `_assimilate_frames` to every analysis scheme, including
   multi-frame sweeps and the replay used by reduction diagnostics.
2. Scale covariance exactly once. The stochastic kernel already draws from the
   supplied covariance, and ETKF/LETKF already whiten by it. Do not additionally
   pass beta as the stochastic kernel's alpha after scaling `R`.
3. Add `likelihood_weight=1.0` to ESMDA constructors, threaded through all
   variants. Preserve validation of the base alpha schedule; use
   `alpha_base / likelihood_weight` in the update. Do not weaken or remove the
   existing `Σ 1/alpha_base = 1` check. Keep physical `C_D` unchanged.
   Validate finite `0 < likelihood_weight <= 1`. Explicit alpha overrides
   remain base-schedule values, never already-scaled coefficients. Reject
   `final_time_smoothing=true` with a non-unit likelihood weight: its extra
   update has no allocation in this policy. Preserve the legacy unit-weight
   path and its existing double-conditioning warning.
4. Use a shared, pure policy resolver to obtain filter beta and smoother weight
   before constructing either collaborator. The hybrid validates that their
   configured weights match its policy. Apply the same validation to direct
   library use and Hydra entry points; avoid temporary mutation of collaborators.

In `joint` filtering, beta applies to the entire joint update, including
parameters. Scaling only state rows while updating parameter rows at full
weight would define a different method. Inflation remains independently
configured and is disabled or identity-valued in exact-posterior tests.

### B. Wire configuration, artifacts and diagnostics

Use `filter_smoothing.beta` as the hybrid's single source, forwarded into
`filtering.filter`; the standalone filter may expose `filtering.beta=1.0`.
Shared-budget ESMDA weight is derived, not another tunable override. Reject
conflicts. Update `conf/run_filter_smoothing.yaml`, `conf/run_filtering.yaml`,
the smoother constructor wiring and `scripts/filter_smoothing/run_filter_smoothing.py`.
Both static and dynamic hybrid paths must use the same resolved policy.

Persist beta, allocation policy, base/effective alphas, both phase weights,
physical-error-model version and observation-product identity. Preserve
physical `obs_error_std` and the existing physical-covariance NIS. Any
effective-covariance diagnostic needs a separate name. Record actual analyzed
state predictions for comparisons; do not rank localized filters by ride-along
posterior RMSE. Ensure repeated windows do not repeatedly multiply covariance.

### C. Verify before CFD tuning

| Test | Expected behavior |
|---|---|
| Default policy, beta 1 | Seeded legacy equivalence, including RNG stream |
| Fixed linear prior, beta sweep | Mean/covariance match Kalman analysis with `beta R` |
| Stochastic versus ETKF | Same target moments; sampling tolerances only for stochastic updates |
| LETKF and TSVD variants | Effective covariance used in local selection/whitening as appropriate |
| Static/dynamic, multi-frame/window, disk/memory paths | Scaling once per analysis; no covariance drift |
| Invalid beta/shared-product combinations | Rejected before any CFD forecast |

Add `tests/test_hybrid_tempering.py`; extend `test_filtering.py`,
`test_filtering_etkf.py`, `test_filtering_letkf.py`, `test_filter_smoothing.py`,
`test_run_filter_smoothing.py`, `test_esmda_smoother.py` and Hydra/diagnostic tests.

Include the analytical `x=θ`, Gaussian-prior example from the review. For a
compatible joint update with no inflation and identical observations, shared
weights recover the scalar once-conditioned covariance; filter-only weights
give `V⁻¹ = P⁻¹ + (1 + 1/beta) R⁻¹`. Also test the state-only case: correct
scalar state variance does not imply a correct joint state–parameter covariance.
Those equalities describe ideal updates; use exact-transform or analytical
oracles and statistical tolerances for finite-ensemble stochastic ESMDA.
General joint-Gaussian examples should quantify that discrepancy, not encode
the assertion that beta fixes it.

## Experiments and completion criteria

After plan 1's diagonal contract, compare filter-only beta `1, 2, 4, 8` and
shared-budget beta `2, 4, 8` on a cheap model, then periodic upper-forcing and
cross-model CFD. Freeze calibrated `R`, inflation and solver budgets during
the beta comparison. Select beta using calibration seeds; report held-out
coverage, proper scores, forcing bias, physical NIS and assimilation-off forecast
skill on new seeds. Do not select beta by assimilated RMSE alone.

Ship the control when the mathematical tests and compatibility checks pass;
recommend a non-default policy/value only after independent validation. Update
`docs/data_assimilation.md` and `docs/scripts_and_configs.md`, including their
currently stale statement that arbitrary scalar alpha overrides are valid.

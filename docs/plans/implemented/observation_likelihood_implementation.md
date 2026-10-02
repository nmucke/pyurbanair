# Plan 1 — Consistent observation likelihood

Status: proposed implementation; no algorithm changes made.
Implements recommendation 1 in [the review](../../research/data_assimilation_recommendations.md).
Related plans: [model discrepancy](model_discrepancy_implementation.md) and
[hybrid beta](hybrid_beta_tempering.md).

## Outcome and scope

Give ESMDA, filtering and the hybrid one explicit observation-error contract:
what noise generates synthetic measurements, what uncertainty the likelihood
assumes, and how aggregation changes that uncertainty. Deliver a correct
diagonal implementation first, then extend to correlated errors.

Keep physical covariance `R` separate from tempering. In the beta plan, **beta
is a covariance multiplier**, so the filter uses `beta R`. It is not an error
standard deviation or a likelihood exponent. Synthetic measurements must not
change when alpha or beta changes.

## Design decisions

- Add `observation_error.py` under `libs/data-assimilation/src/data_assimilation/`
  and a `create_observation_error(...)` helper in
  `src/pyurbanair/config/hydra_helpers.py`. Use labelled sensor/component/time
  metadata and the existing time-major flattening order.
- Keep `observation_error: null` as the exact legacy path, including existing
  random draws and the current conservative aggregation behavior. New runs opt
  into corrected behavior explicitly; do not silently reinterpret old results.
- Separate zero-mean instrument covariance from calibrated representation-error
  covariance. Do not add fictitious representation noise to synthetic truth
  merely because that uncertainty is included in the assimilation likelihood.
  Persistent forecast bias belongs in plan 4, not an arbitrarily large `R`.
- Allow instrument standard deviations by component and sensor, with a scalar
  fallback. Resolve height bands through sensor metadata. Reject missing labels,
  duplicates, non-finite values and non-positive total variances.

Proposed opt-in configuration, shared by all three entry points:

```yaml
observation_error:
  instrument_std: 0.25
  representation_std: 0.0
  representation_time_model: independent
  aggregation: propagate_mean
```

`representation_std` is a calibrated raw-frame marginal uncertainty, in the
observed variable's units. The initial `independent` model is an explicit
approximation, not a claim about cross-model residuals. Existing scalar error
keys supply defaults only in legacy mode. Reject conflicting explicit overrides
of old and new settings; document precedence during Hydra resolution.

## Implementation sequence

### A. Centralize diagonal covariance and aggregation

1. Introduce an immutable error specification and resolved covariance product,
   carrying observation labels, physical times, instrument/representation
   contributions and provenance. Keep diagonal vectors as the fast path.
2. Refactor `AggregateObservations` in `observation_operator.py` to expose its
   actual bin membership and averaging weights. Data and covariance must use
   the same bins, including partial bins, unequal counts and analysis strides.
3. For averaging matrix `A`, implement `R_aggregate = A R_raw Aᵀ`. With diagonal
   raw errors, compute weighted variance sums without allocating dense `A` or
   `R`. Equal independent variances give `σ²/m`, not `σ²`.
4. Preserve empty-bin checks. Resolve covariance for each window's actual times;
   equal vector length does not guarantee equal bin counts or covariance.
5. Initially reject `median`, `min` and `max` in corrected mode. They require
   a separately calibrated, potentially biased and signal-dependent error
   distribution. Keep their current behavior available in legacy mode.

A persistent representation-error floor must have an explicit temporal model
or a separately calibrated aggregate-product likelihood. Do not apply the
independent `1/m` reduction to persistent error, or silently reuse one floor at
every aggregation duration. Product-specific approximations must be labelled
and cannot establish exact cross-phase likelihood accounting.

### B. Integrate every runner and preserve diagnostics

Replace duplicated covariance construction in `scripts/esmda/run_esmda.py`,
`scripts/filtering/run_filtering.py` and
`scripts/filter_smoothing/run_filter_smoothing.py` with the shared resolver.
Generate noisy raw measurements once; filter on selected frames and smooth on
the consistently aggregated product. Preserve frame identities in artifacts.

Extend the DA constructors/call contracts to accept resolved per-window or
per-frame covariance where it changes, avoiding stale first-window covariance.
Preserve existing constant-array APIs as adapters. Validate all covariance
products before costly forecasts; do not mutate shared covariance in place.

Keep `obs_error_std` artifacts as square roots of physical marginal variances.
Add error-model version, components, aggregation weights/counts and separate
analysis multipliers. For correlated errors, store the structured covariance
or a reproducible reference; diagonal marginals alone are insufficient.

Save signed innovations and physical-covariance NIS alongside bias,
autocorrelation, cross-sensor residuals and coverage. Reapply `H` to actual
analyzed states at matching times; localized/reduced ride-along predictions
remain explicitly labelled proxies. Never make increasing beta appear to fix
calibration by changing the denominator of the existing physical NIS.

### C. Add structured correlations behind capability checks

Extend `filtering/analysis.py`, `filtering/etkf.py`, `filtering/base.py` and
`smoothing/esmda.py` with SPD covariance factor/solve/sample operations, retaining
the unchanged diagonal path. Start with same-frame component/sensor blocks.
Stochastic perturbations need a matrix factor, not element-wise square roots.

For localized analyses, select and taper physical observation subsets before
local whitening; first reject unsupported full-covariance/localization
combinations. Global whitening must not silently change sensor locality.
Batch ESMDA may then accept temporal covariance. Sequential filtering requires
conditional likelihoods or a colored-error state before accepting temporal
correlations; silently treating those frames as independent is not supported.

## Verification and delivery gates

| Check | Required result |
|---|---|
| Mean aggregation, including unequal/partial bins | Matches explicit `A R Aᵀ` |
| Label/flatten order and strided frames | Data and covariance remain aligned |
| Linear-Gaussian oracle | Correct posterior mean/covariance for supported error models |
| Legacy seeded runs | Unchanged observations, updates and RNG consumption |
| Alpha/beta sweep | Measurements and physical covariance unchanged |
| Unsupported nonlinear/correlated cases | Fail before forecast, with a useful message |

Add `tests/test_observation_error.py`; extend observation-operator, ESMDA,
filtering/ETKF, runner, Hydra and observation-diagnostics tests. Use analytical
cases first, then small end-to-end runs. Update `docs/data_assimilation.md`,
`docs/scripts_and_configs.md` and the three run configurations.

Release A–B before C. Calibrate on designated training runs and freeze settings
before same-model/cross-model tests. Acceptance is improved held-out uncertainty
calibration for each observation product, without spurious gains from
miscounting averaged noise. Aggregation can legitimately discard information;
require raw/aggregated posterior equivalence only in an oracle where the
aggregate is a sufficient statistic. Residual-based calibration should respect
the assumptions discussed by
[Desroziers et al.](https://rmets.onlinelibrary.wiley.com/doi/10.1256/qj.05.108).

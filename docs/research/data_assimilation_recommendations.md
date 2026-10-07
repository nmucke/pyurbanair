# Improving data assimilation for transferable urban-flow predictions

Review date: 28 September 2026 · Code baseline: `a0e64d4`.

**Recommendation:** build a statistically consistent assimilation method that
separates uncertain forcing from model discrepancy and accounts for the delayed
response of canopy flow to upper-layer forcing. This offers a stronger research
direction than increasing ensemble size or tuning ESMDA iterations alone.

This report synthesizes three specialist reviews—ESMDA, filtering/hybrid
inference, and CFD/cross-model transfer—with an evaluation and literature review.
Findings below come from the implementation; proposed explanations and benefits
remain hypotheses. No new CFD experiments were run.

## What is already strong

The repository already provides five ESMDA variants, dynamic AR(2) forcing,
stochastic EnKF and ETKF/LETKF, localization, state reduction, inflation and
parameter evolution. The hybrid runs parameter ESMDA followed by state or joint
filtering over the same window. Evaluation already includes held-out sensors,
proper ensemble scores, turbulence statistics and optional probe spectra.
These are foundations to retain, not missing features to implement.

## Prioritized recommendations

### 1. Establish a consistent observation likelihood — first

**Finding.** ESMDA and filtering currently require diagonal observation-error
covariance. ESMDA adds independent noise to raw frames, then averages them while
deliberately retaining the raw-frame variance. Thus aggregation changes the
degree of conservatism. See [ESMDA covariance validation](../../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py)
and [observation construction](../../archive/scripts/esmda/run_esmda.py), lines 948–956 (archived runner; now `scripts/utils/helper_functions.py`).

**Action.** Separate instrument noise, representation error and dynamical model
error. For linear averaging, propagate instrument covariance as
`R_mean = A R_raw Aᵀ`: averaging `m` independent, equally noisy frames gives
`σ²/m`. Add an explicitly calibrated representation-error floor rather than
implicitly using the discarded variance. Start with component/height-dependent
variances; introduce temporal or spatial correlations only where residuals
justify them. Correlated errors require changes to both covariance solves and
localization; global whitening can mix distant sensors.

Record signed innovation bias, autocorrelation and cross-sensor correlation,
alongside existing scores. Estimate error settings on calibration runs, then
freeze them for transfer tests. Residual diagnostics have an established basis,
but biased/nonlinear/localized analyses require care:
[Desroziers et al. (2005)](https://rmets.onlinelibrary.wiley.com/doi/10.1256/qj.05.108).

**Acceptance:** calibrated predictive intervals and stable conclusions across
aggregation intervals. A lower assimilated RMSE alone is insufficient.

### 2. Make the hybrid's state–parameter posterior coherent — first

**Finding.** [FilterSmoothing.run](../../libs/data-assimilation/src/data_assimilation/filter_smoothing/base.py)
explicitly uses observations in parameter ESMDA and again in the subsequent
filter (lines 564–639). This is a useful heuristic, but ordinary filtering of
the propagated parameter posterior does not generally produce the correct joint
posterior. Reusing observations is legitimate only with the appropriate
conditional construction.

A simple acceptance case exposes the issue: let `x = θ`, a Gaussian prior with
variance `P`, and independent additive Gaussian observation noise with variance
`R`. Exact conditioning gives both variables variance
`V = PR/(P+R)`. Filtering the resulting `x` ensemble again with the same datum
gives `PR/(2P+R)`. State-only filtering additionally breaks `x = θ`; joint
filtering contracts both twice. This is an analytic counterexample, not a
measurement of current CFD runs.

**Action.** First test means, joint covariance and coverage against exact
linear-Gaussian posteriors. Then implement a consistent joint fixed-lag smoother
or conditional dual scheme, retaining state–parameter dependence. A relevant
precedent is the [Bayesian-consistent dual EnKF of Ait-El-Fquih et al. (2016)](https://hess.copernicus.org/articles/20/3289/2016/).
Simply splitting scalar likelihood weights between the existing two stages is
not a demonstrated repair. Keep optional ESMDA `final_time_smoothing` off for
uncertainty claims: its source already warns about extra conditioning.

**Acceptance:** exact-posterior tests, nonlinear toy tests, then CFD coverage.
Label hybrid within-window estimates as retrospective: their parameters already
depend on observations later in that window. Test shifted window boundaries:
the current hybrid resets its joint parameter correction at each boundary.

### 3. Address periodic geometry and forcing observability — highest physics priority

**Finding.** [DistanceLocalization](../../libs/data-assimilation/src/data_assimilation/localization/distance.py)
uses ordinary Euclidean distance, so neighboring points across periodic faces
are treated as distant. Dynamic parameter knots also share one localization
block; an observation selected by one knot can be admitted for all knots
([augmentation](../../libs/data-assimilation/src/data_assimilation/augmentation.py),
`group_ids`; [localization](../../libs/data-assimilation/src/data_assimilation/localization/base.py),
`_group_inflation`).

**Action.** Add minimum-image distances on configured periodic axes, followed
by separate horizontal/vertical scales. Test local turbulent corrections
alongside global mean/profile modes. For dynamic forcing, compare knot-local or
short-time blocks with current grouping.

The physical hypothesis is that upper forcing influences canopy sensors through
delayed momentum transport, with forcing and mixing effects partly confounded.
Measure step/impulse responses and ensemble sensitivity singular values before
choosing knot spacing, assimilation windows or adding parameters. Retain
observation times and design lag-aware updates from these responses. Prior
trajectory correlations can legitimately connect different times; do not
replace this with a blanket causal mask.

Compare canopy-only sensors with a fixed-budget canopy-plus-upper-layer network.
Use common physical controls—reference wind components/profile, reference
height, nudging cutoff and relaxation time—across solvers. Current defaults
even use different nudging cutoffs: 16 m in [uDALES](../../configs/model/pyudales.yaml)
and 4 m in [PALM](../../configs/model/pypalm.yaml). Match them for controlled transfer
experiments, then vary them deliberately.

**Acceptance:** periodic translation invariance, identifiable forcing directions,
and better held-out canopy forecasts across forcing timescales.

### 4. Estimate model discrepancy explicitly — central research contribution

**Finding.** Vertical-profile and SGS compensation parameters already exist,
but the shared DA workflow has no explicit within-window dynamical discrepancy
trajectory. Inflation maintains spread without estimating a persistent bias.
Solver parameters are not necessarily comparable: PALM's `sgs_constant` selects
constant diffusivity and disables its constant-flux layer, whereas other
backends use closure coefficients. See [PALM `_apply_sgs_setting`](../../libs/pypalm/src/pypalm/forward_model.py).

**Action.** Introduce a small, regularized basis of height-dependent momentum
tendencies, with persistent bias and stochastic variability modeled separately:
`x[k+1] = M(x[k], θ[k]) + B b[k] + η[k]`. Give `b` a duration-aware persistence
model and constrain `B` to physically admissible modes. Use backend-specific
closure priors; do not score unrelated SGS constants as the same true parameter.
Forcing and discrepancy may remain confounded, so limit discrepancy dimension
using sensitivity tests and withheld forecasts.

Implement the same discrepancy concept in ESMDA trajectories and filter process
noise. Scale evolution with elapsed physical time; current parameter random
walks use a per-cycle standard deviation. Compare against inflation-only and
existing parameter compensation. [IEnKF-Q](https://arxiv.org/abs/1711.06110)
provides a model-error-aware iterative benchmark; its low-order experiments
motivate testing, not an assumed CFD benefit.

**Acceptance:** gains persist after assimilation stops, without forcing estimates
systematically absorbing solver bias or discrepancy fitting only sensor sites.

### 5. Preserve physical consistency and reproducible restarts

**Finding.** The shared DA layer unflattens analyzed fields directly; it does
not explicitly enforce discrete divergence or wall constraints. Local updates
and row-dependent inflation can disturb constraints that a global ensemble
transform preserves. Restart state is also backend-dependent: uDALES carries
unexposed SGS fields from the previous run, while other backends reconstruct or
retain different internal fields. See [filter analysis](../../libs/data-assimilation/src/data_assimilation/filtering/base.py)
and [uDALES `run_single`](../../libs/pyudales/src/pyudales/forward_model.py), lines 1019–1032.

**Action.** Measure divergence, wall flux, momentum/energy jumps and the fraction
of each increment lost during the next forecast. If needed, apply a
solver-consistent constrained increment/projection; test gradual increment
application only where restart shocks dominate. PALM already retains an
initialization pressure solve for non-flat topography; measure its adjustment
before adding another projection. Use log/logit coordinates for
positive/bounded parameters and wind components or suitable circular coordinates
for direction; sampler bounds do not constrain subsequent Kalman updates.

Before expensive experiments, audit identical-input reruns from identical full
checkpoints. In ESMDA, restore the correct window-start hidden state before each
iteration; carrying a previous iteration's endpoint could otherwise change the
forecast operator. Also verify that the first forecast output used as an ESMDA
initial state has the intended physical timestamp. These are audit targets,
not confirmed failures.

**Acceptance:** reproducible reruns and assimilation improvements that survive
restart adjustment without artificial energy injection.

### 6. Improve nonlinear iteration after the structural work

Compare equal-weight ESMDA with geometric/adaptive schedules or a regularized
iterative smoother, preserving `Σ 1/αᵢ = 1`. The scalar consistency check already
exists. Use equal forward-solve budgets and report prior displacement, failures,
spread and withheld forecast skill. [Emerick's geometric schedules](https://arxiv.org/abs/1812.00924)
are an established comparison. ETKF/LETKF should remain strong existing
baselines. Treat additional iteration tuning as an ablation, not the primary
publication claim.

## A focused publication experiment

**Proposed question:** Can separating forcing from structured model discrepancy,
with response-aware state–parameter inference, improve calibrated urban-flow
forecasts across solvers and boundary conditions?

| Stage | Comparison | Purpose |
|---|---|---|
| Validity | Exact Gaussian cases; periodic translation; identical-checkpoint reruns | Establish statistical and numerical correctness |
| Controlled physics | Same solver: inflow/outflow versus periodic upper forcing; forcing bias versus closure/grid mismatch | Identify which mechanism each improvement addresses |
| Transfer | uDALES → PALM and reverse; add LBM only under comparable supported forcing | Separate generalization from a favorable transfer direction |
| Stress test | Unseen forcing timescales, sensor heights/layouts, and one geometry or resolution | Test robustness beyond tuning conditions |

Use paired truth/noise/ensemble seeds across methods and separate calibration
from final test regimes. Start with a small pilot, then choose replication from
effect-size uncertainty; bootstrap independent runs or time blocks, not
correlated frames as independent samples. Report computational cost including
failed/resampled members. Compare ESMDA, filtering, the current heuristic hybrid,
and the proposed consistent method, with component ablations at matched cost.

Primary outcomes should be held-out velocity-vector error, proper ensemble
scores/coverage, and assimilation-off forecasts at common lead times and
information cutoffs. Distinguish causal filtering from retrospective smoothing.
Extend the existing [evaluation pipeline](../../scripts/compute_metrics.py)
with these forecast comparisons; retain mean profiles, resolved TKE, Reynolds
stress and spectra. Score turbulence per member before ensemble reduction.
Reapply `H` to actual analyzed states at matching analysis times: localized/reduced filter
`obs_posterior_rmse` is explicitly a proxy. Speed-only RMSE and a fixed
normalized-misfit target cannot establish directional accuracy or calibration.

**Implementation order:** likelihood and hybrid validity → periodic/response
tests and restart audit → discrepancy-aware method → locked transfer benchmark.
Existing urban LES research already uses ESMDA for meteorological forcing and
reports structural-error and sensor-placement limitations
([Lumet et al., 2026](https://eliott.lumet.me/publications/2026_lumet-etal_bae_esmda/)).
A recent [Bombardi et al. preprint](https://arxiv.org/abs/2607.03571) also studies
regularized EnKF inference and transfer of urban RANS closure parameters;
distinguish that setting from transient LES forcing and state assimilation.
The defensible contribution is therefore demonstrated transfer with calibrated
uncertainty and a diagnosed physical mechanism. Novelty and publishability
require those results and a fuller related-work comparison.

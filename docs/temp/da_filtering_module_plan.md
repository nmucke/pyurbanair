# Plan: A `filtering` Submodule for the Data-Assimilation Library

Goal: add sequential (filtering) data assimilation alongside the existing `smoothing/`
package. Main focus: the **ensemble Kalman filter and its extensions** (stochastic/
perturbed-obs EnKF, deterministic square-root transforms ETKF/LETKF, localized variants),
supporting **state estimation alone, parameter estimation alone, and joint
state+parameter estimation**, with an architecture that later admits non-EnKF methods
(particle filters, hybrid/variational schemes) without rework.

Everything below is scoped to `libs/data-assimilation`; run-script and Hydra wiring are
listed at the end as a separate stage.

---

## 1. What filtering changes conceptually vs. the existing smoother

The ESMDA smoother's unit of work is a **window**: forecast the whole window, update the
window's IC + parameters `num_steps` times with tempered weight, re-forecast. A filter's
unit of work is a **cycle**: forecast the ensemble only to the next observation time,
apply **one** full-weight analysis to the state *at that time*, continue from the
analyzed state. Consequences that drive the design:

* **The forecast/analysis loop inverts.** The smoother calls the forward model
  `num_steps + 1` times per window over the same interval; the filter calls it once per
  cycle over successive intervals. The cycle loop is new code; the analysis math is not
  (see §3).
* **The analyzed field is always integrated.** In the smoother, only the IC of the next
  MDA iteration is integrated; in a filter *every* analysis becomes the model state.
  Attractor/realizability handling (post-analysis fixups, §6) matters more.
* **Parameters need an evolution model.** In a smoother, parameters are static unknowns
  of the window. In a filter, a parameter that receives no forecast noise collapses to
  (near-)zero spread after a few cycles and stops learning. Parameter-only and joint
  filtering therefore need explicit spread maintenance (random-walk evolution, kernel
  shrinkage, or inflation — §5).
* **Observation semantics change.** The current operators aggregate a window
  (`intervals` mode). A filter assimilates the observation *at the analysis time* — for
  this codebase that is naturally "the interval mean of the cycle that just ended",
  which the existing operators already produce if one cycle = one interval.

## 2. Package layout

```
src/data_assimilation/
    augmentation.py          # NEW (extracted): flatten/unflatten state & params,
                             #   group ids, row coords — shared by smoothing & filtering
    inflation.py             # NEW: multiplicative, RTPS, RTPP (+ later adaptive)
    filtering/
        __init__.py          # public API: EnKF, ETKF, LETKF, filter configs
        base.py              # BaseFilter: cycle loop, forecast-to-obs-time,
                             #   augmentation handling, history/persistence
        analysis.py          # AnalysisScheme ABC + StochasticEnKFAnalysis
        etkf.py              # ETKFAnalysis (square-root transform), LETKF driver
        parameter_evolution.py  # identity / random-walk / kernel-shrinkage models
smoothing/  localization/    # unchanged (localization is reused as-is)
```

Three deliberate splits:

1. **`BaseFilter` (the cycler) vs. `AnalysisScheme` (the update math).** The filter owns
   time management, forecasting, augmentation, parameter evolution, inflation, and
   diagnostics; the analysis scheme is a pure function
   `(augmented, aug_dev, pred_obs, obs, C_D, rng) -> updated augmented`. This is the
   extension point for "more methods later": a particle-filter or rank-histogram update
   is a new `AnalysisScheme` (plus, for weight-based methods, a resampling hook — see
   §8), not a new filter class.
2. **Augmentation extracted, not duplicated.** `_flatten_state`, `_unflatten_state`,
   `_flatten_time_varying_params`, `_unflatten_params`, `_state_row_coords`,
   `_state_group_ids`, `_group_ids_by_base_name` currently live as methods on the ESMDA
   class hierarchy but are pure structure transformations. Extract them into
   `augmentation.py` as two small classes used by both packages:
   * `ParamAugmentation(pin_initial_time_point=...)` — Dataset ⟷ `(N_p, N_e)` array +
     names + group ids;
   * `StateAugmentation()` — Dataset ⟷ `(N_s, N_e)` array + row coords + cell group ids.
   The smoother keeps its behavior (its methods become thin delegating wrappers, or are
   replaced outright in the same PR); the filter gets the identical flatten order and
   pinning semantics for free. This also resolves the code-review finding that the
   smoother's 5-class MRO diamond exists mainly to mix augmentation variants: the filter
   should NOT reproduce that hierarchy — one `EnsembleKalmanFilter` class with
   `mode="state" | "parameter" | "joint"` selecting which augmentations are active.
3. **Inflation as its own module**, because smoothing will want it too (math-review
   §3.5) and because every EnKF variant needs it independent of the analysis flavor.

## 3. Reuse: the stochastic EnKF analysis already exists

The key implementation observation: **`_BaseESMDA._compute_kalman_update` with
`alpha = 1` *is* the stochastic (perturbed-observation) EnKF analysis**, including the
localized local-analysis path (`BaseLocalization.localized_update` takes `alpha`
explicitly and is already smoother-agnostic — it lives in `localization/`, not
`smoothing/`). Plan:

* Move the body of `_compute_kalman_update` (global path) into
  `filtering/analysis.py::StochasticEnKFAnalysis` — a pure function of arrays plus an
  explicit `rng_key` (no `self.rng_key` mutation) and an `alpha: float = 1.0` parameter.
* `_BaseESMDA` calls this shared implementation with its tempered `alpha` — the smoother
  and the filter then share one tested implementation of the perturbed-obs update, one
  C_D contract, and the localization machinery (`group_ids`, `localize_mask`,
  `row_coords`/`obs_coords` plumbing) verbatim.
* Fix the two known defects at the shared site while moving it (dead
  `jnp.linalg.LinAlgError` handler; element-wise `sqrt(C_D)` → validated 1-D variance
  vector), so both packages inherit the fixes.

`CorrelationLocalization` and `DistanceLocalization` work for the filter **unchanged**:
they only consume anomalies/coordinates and were designed against the local-analysis
update, which is standard for EnKFs.

## 4. The classes

### 4.1 `BaseFilter` (`filtering/base.py`)

```python
class BaseFilter:
    def __init__(self, observation_operator, forward_model, C_D,
                 analysis: AnalysisScheme,
                 mode: Literal["state", "parameter", "joint"] = "joint",
                 localization: BaseLocalization | None = None,
                 inflation: InflationScheme | None = None,
                 parameter_evolution: ParameterEvolution | None = None,
                 rng_key: jax.Array | None = None): ...

    def run(self, state, params, observations, *,
            return_history: bool = False) -> FilterResult: ...
```

* `observations`: sequence of per-cycle observation vectors (shape
  `(num_cycles, N_d)`), one per forecast segment — the filter analog of the smoother's
  single window vector. The cycle loop: for each cycle, (1) forecast the ensemble over
  one segment (`forward_model.run_ensemble(state=analysis_state, params=params)` — the
  segment length is the forward model's configured horizon, exactly as the smoother's
  window is today), (2) apply the observation operator to the segment (final frame, or
  interval mean — the existing operators' behavior), (3) build the augmented vector per
  `mode`, (4) inflate forecast anomalies (prior inflation) if configured, (5) call the
  analysis scheme, (6) post-analysis inflation (RTPS uses both prior and posterior
  spread), (7) split the augmented vector back, apply parameter evolution noise, (8) the
  analyzed final frame warm-starts the next cycle.
* `FilterResult`: small dataclass — `params`, `state`, optional per-cycle histories,
  and per-cycle diagnostics (spread, innovation statistics, effective inflation). This
  avoids from day one the return-type polymorphism the smoother has.
* Failure handling mirrors the smoother:
  `apply_failure_substitutions_to_state/params` after each forecast.
* On-disk mode: reuse the smoother's `step_{i}` pattern as `cycle_{k}` directories with
  the same pruning knobs; `_get_sorted_state_files` and the streaming helpers are
  already in `smoothing/base.py` — move them to a shared `io.py` (or `augmentation.py`)
  rather than importing filtering from smoothing.

Mode semantics:

* `mode="state"`: augmented vector = flattened final-frame state only; params (if any)
  are carried unmodified through cycles.
* `mode="parameter"`: augmented = flattened params only; the analyzed params apply from
  the next cycle onward. Requires `parameter_evolution` (or inflation on the param
  block) to avoid collapse — validate at construction and error with guidance if both
  are absent.
* `mode="joint"`: `[state | params]`, with `localize_mask` marking parameter rows for
  the global update exactly as the smoother does today.

### 4.2 `StochasticEnKFAnalysis` (`filtering/analysis.py`)

The shared perturbed-obs update of §3. Options: `alpha` (fixed 1 for filtering; the
smoother passes its own), centered perturbations (subtract the sample mean of the
perturbation draw — math-review §2.3 — on by default here since it is new code with no
backward-compat concern).

### 4.3 `ETKFAnalysis` + LETKF (`filtering/etkf.py`)

The deterministic square-root family, second priority after the stochastic filter:

* **ETKF** (Bishop et al. 2001 / Hunt et al. 2007 formulation): compute weights in
  ensemble space — `P̃ = [(N_e−1)I + Yᵀ R⁻¹ Y]⁻¹`, mean weight
  `w̄ = P̃ Yᵀ R⁻¹ (y − ȳ)`, transform `W = [(N_e−1) P̃]^{1/2}` via symmetric square
  root — and apply `x_a = x̄_f + X_f (w̄ + W)`. No perturbed observations, no sampling
  noise in the analysis, exact posterior covariance in the linear-Gaussian limit; all
  dense algebra is `N_e × N_e`, so it is *cheaper* than the stochastic filter when
  `N_d > N_e` (always true here). Diagonal-R assumption matches the library's C_D
  contract.
* **LETKF**: the local analysis wrapper — every augmented row (or block, reusing
  `group_ids`) gets its own ETKF weight matrix computed from its own
  inflation-weighted observations. Crucially, the existing
  `BaseLocalization.inflation_factors` output is exactly what LETKF needs: per-(row,
  obs) error-variance inflation `E_inf²` multiplies R (equivalently divides the R⁻¹ in
  `Yᵀ R⁻¹ Y`), with `inf` → exclusion. So both existing localization strategies drive
  LETKF with **no changes to `localization/`** — implement one
  `local_etkf_update(...)` alongside `localized_update(...)` in `localization/base.py`
  (or in `etkf.py` consuming `inflation_factors`). The per-row cost is `O(N_d N_e²)`
  with `N_e × N_e` eigendecompositions — vmap-able, and the same
  unique-row/blocking dedup recommended in the code review applies.

Rationale for having both families: the stochastic filter maximizes reuse and gives a
like-for-like comparison with ESMDA; ETKF/LETKF is the field-standard choice for small
ensembles because it removes perturbation sampling noise — plausibly significant at
N_e = 50 with N_d = O(10–100) per cycle.

### 4.4 `parameter_evolution.py`

Small strategies applied to the parameter block after each analysis:

* `Identity` — for short experiments / joint runs relying on inflation;
* `RandomWalk(std | per-param stds)` — `θ_{k+1} = θ_k + ξ`, the standard augmented-state
  approach;
* `KernelShrinkage(a≈0.99)` (Liu & West 2001) — shrink toward the ensemble mean and add
  compensating noise so the parameter ensemble keeps its variance without inflating it:
  `θ ← a·θ + (1−a)·θ̄ + √(1−a²)·σ_θ·ξ`. Preferred default for parameter-only mode.

These also give the natural hook for the AR(2) time-varying inflow parameters: in
filtering, "time-varying parameter estimation" becomes "the parameter value *now*",
evolved by the prior's own AR model between cycles — the filter counterpart of the
dynamic smoother. Phase 2+ (see §7), but the interface should anticipate it: give
`ParameterEvolution.evolve(params, rng_key)` the whole Dataset, not a bare array.

### 4.5 `inflation.py`

* `Multiplicative(factor)` — anomalies × λ before analysis;
* `RTPS(alpha)` (Whitaker & Hamill 2012) — posterior anomalies rescaled toward prior
  *spread*: needs prior and posterior std, applied after analysis;
* `RTPP(alpha)` — blend of prior and posterior *anomalies*;
* interface: `inflate_prior(dev) -> dev` and `inflate_posterior(dev_prior,
  dev_post) -> dev_post`, both optional; applied per augmented block (state and params
  can have different inflation — important: parameters usually need more).
* Later: adaptive multiplicative inflation from innovation statistics
  (Desroziers-based), which the diagnostics (§6) already compute.

## 5. Mathematical contracts to enforce from day one

* **Full-weight single analysis** (`alpha = 1`) per observation batch; never re-use an
  observation batch across cycles (the filter has no MDA schedule — assert observation
  batches are consumed exactly once).
* **Diagonal C_D as a 1-D variance vector** in the new API (the honest contract; both
  ETKF's `R⁻¹` and the localization inflation assume it).
* **Parameter spread floor**: refuse silently-collapsing configurations
  (parameter modes without evolution/inflation) at construction.
* **Analysis-time semantics of aggregated observations**: if a cycle's observation is a
  30 s interval *mean* but the analysis updates the final instantaneous frame, the
  observation operator must compute the predicted obs the same way (interval mean of
  the forecast segment) — H and y must agree, which the existing
  `TemporalObservationOperator(mode="intervals")` on the segment already guarantees.
  Document that the filter treats the interval-mean as an observation *of the segment*,
  assimilated into the end-of-segment state; this is standard (it is an
  observation-operator choice, not an approximation error).

## 6. Diagnostics & realizability (small but essential)

Per cycle, compute and return: ensemble spread (state & param blocks), innovation
`d = y − H(x̄_f)`, innovation consistency `dᵀ(HP_fHᵀ + R)⁻¹d / N_d` (χ²), observation-
space prior/posterior RMSE, count of active observations per row (localized runs), and
the applied inflation factors. These make "the filter is diverging / overconfident"
visible at cycle k instead of at the end — the exact gap the state-estimation review
identified for the smoother.

Post-analysis state fixups (hook on `BaseFilter`, default no-op): clamping/physical
projection supplied by the forward-model side (e.g. re-imposing solid-cell zeros for
LBM). The analyzed field is integrated every cycle, so unphysical increments hurt
filters more than smoothers; keep the hook in the library and the physics in the
forward-model packages.

## 7. Phased implementation

**Phase 0 — extraction (no behavior change).** `augmentation.py` + shared stochastic
update + shared file/IO helpers; smoothing delegates to them; existing tests must pass
unchanged. Fix the two shared-site bugs (dead except; C_D contract) here.
Estimated size: one focused PR.

**Phase 1 — stochastic EnKF, all three modes.** `BaseFilter`, `StochasticEnKFAnalysis`,
`Identity`/`RandomWalk` evolution, `Multiplicative`/`RTPS` inflation, `FilterResult`,
diagnostics. Tests: (a) linear-Gaussian cycle vs. exact Kalman filter (mean and
covariance, N_e → large); (b) scalar parameter convergence on a toy forward model;
(c) joint mode with `localize_mask` equivalence checks mirroring
`tests/test_localization.py`; (d) collapse guard (parameter mode without evolution
raises).

**Phase 2 — localized EnKF + ETKF/LETKF.** Localized stochastic filter comes free from
Phase 0 reuse (test it); then `ETKFAnalysis` (global) with the standard
symmetric-square-root identities tested against the stochastic filter in the large-N_e
limit and against exact KF in the linear case; then LETKF driven by
`inflation_factors`. Add `KernelShrinkage`.

**Phase 3 — integration.** `run_enkf.py` (or a `da=filter|smoother` axis in
`run_esmda.py` — decide then; the truth-generation, sensor, and metrics plumbing is
shared), Hydra groups `filtering/analysis=stochastic|etkf|letkf`,
`filtering/inflation=...`, case wiring where one cycle = one 30 s observation interval
(segment horizon = `interval_seconds` instead of `simulation_time`). e2e test in the
style of `test_run_esmda.py` (serial, per the existing e2e constraint).

**Phase 4 (later, as requested) — beyond EnKF.** The `AnalysisScheme` seam plus one
addition — an optional per-member weight vector in `FilterResult` and a
`resample(ensemble, weights, rng)` hook on `BaseFilter` — is sufficient for: bootstrap
particle filter with resampling, ensemble-transform particle filter (ETPF), Gaussian
mixture / rank-based updates, and hybrid EnKF–PF schemes. Nothing in Phases 0–3 needs
to change shape for these; do not build the weight machinery until a first non-Gaussian
method is actually scheduled.

## 8. Open decisions (flagged, with a default)

* **Cycle length vs. observation interval**: default one cycle per observation interval
  (30 s here). Longer cycles with multiple intervals per analysis re-introduce a
  window — at that point the smoother is the right tool; don't blur the two.
* **Where LETKF's local loop lives**: default `etkf.py` consuming
  `BaseLocalization.inflation_factors`, keeping `localization/` strategy-only.
* **Filter + state reduction**: excluded from the plan. The reduction exists to make the
  *smoother's* giant IC update tractable; a filter analysis in a per-cycle POD basis is
  a research topic (and the state-estimation report covers better-suited alternatives).
  Keep `state_reduction` out of `BaseFilter`'s signature.
* **Naming**: `filtering/` package, `EnsembleKalmanFilter` as the user-facing composed
  class (`BaseFilter` + default stochastic analysis), `ETKF`/`LETKF` as named presets.

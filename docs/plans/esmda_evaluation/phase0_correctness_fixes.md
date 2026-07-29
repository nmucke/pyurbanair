# Phase 0 — correctness fixes in existing metric code

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Rationale: §6.1 and §5 of
> [../esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md).
> No new features; land before everything else. One PR.
>
> **Implementer: when this phase lands, update the status table in
> master_plan.md and record any deviations in the "Deviations" section at the
> bottom of this file.**

## Why first

The pairwise-spread term of the CRPS/energy-score estimators currently uses
the biased `1/M²` form whose optimum is an under-dispersed ensemble — the
scores reward ensemble collapse, which is the failure mode the whole
evaluation effort is designed to detect. The spread summary averages standard
deviations (Jensen-biased low → fakes under-dispersion). Everything in later
phases builds on these numbers being unbiased.

## WP0.1 Fair pairwise estimators

Three sites compute the same biased term; fix all three identically
(divide the pairwise sum by `M(M−1)`, excluding the zero diagonal —
Ferro 2014 "fair" estimator):

1. `src/pyurbanair/utils/da_metrics.py::per_knot_crps` (~line 37):

   ```python
   # before
   term2 = 0.5 * diffs.mean(axis=(0, 1))
   # after
   term2 = diffs.sum(axis=(0, 1)) / (2.0 * n * (n - 1))
   ```

   The `n < 2` early-return already guards the degenerate case. Update the
   docstring (it currently documents the mean-over-all-pairs behavior).

2. `src/pyurbanair/plotting.py::_crps_ensemble` (~lines 124–136): same
   change. This propagates to `compute_parameter_metrics` and
   `compute_sensor_metrics`, i.e. to the ESMDA metric stage *and* the sweep
   pipeline (`scripts/figure_creation/compute_sweep_metrics.py`).

3. `scripts/esmda/_esmda_common.py::_energy_score` (~lines 334–366): the
   pairwise term `d_pair.mean(axis=(0, 1))` → `d_pair.sum(axis=(0, 1)) /
   (E * (E - 1))` (then the existing `0.5 *` weighting). Guard `E < 2`.

Also add the prior-skill companion:

4. `src/pyurbanair/plotting.py::compute_parameter_metrics` already computes
   `prior_rmse` when the prior x-grid matches; compute `prior_crps` the same
   way, and in `_esmda_common.py::parameter_metric_summary` (~line 446) emit
   `crps_reduction_vs_prior = 1 − mean(post_crps)/mean(prior_crps)` next to
   the existing `rmse_reduction_vs_prior`.

**Breaking-number note for the PR description:** all historical CRPS /
energy-score values shift by ~O(1/M) (≈2 % at M=50). Comparisons against
`run_summary.yaml` / `sweep_metrics/` files produced before this PR are
invalid. Key *names* do not change, so downstream readers
(`make_figures_summary.py`, `compare_sweep_results.py`, …) keep working.

**Machine-readable version marker (this WP):** a PR note is not visible to
tooling or to anyone diffing sweep dirs later, so emit `metrics_version: 2`
as a top-level `run_summary.yaml` key (absent or `1` = pre-phase-0
semantics; this WP and WP0.2 are the version-2 value changes). Teach
`scripts/figure_creation/compare_sweep_results.py` and
`compare_state_runs.py` to warn when comparing summaries with mismatched
versions. Later phases do **not** bump the version for purely additive keys
— bump only when an existing key's values or semantics change again.

## WP0.2 Spread–skill formulation

1. `src/pyurbanair/utils/da_metrics.py::summary_scalars` (~line 61):
   `"time_avg_spread": float(np.mean(spr))` →
   `float(np.sqrt(np.mean(spr**2)))` (RMS of per-knot stds = root of the
   average variance). Key name unchanged; value semantics change — note in
   the PR.
2. `scripts/figspec/metrics.py::spread_skill` (~line 100): currently
   `mean(spread_ts)/mean(rmse_ts)`. Change to

   ```python
   def spread_skill(spread_ts, rmse_ts, n_members: int) -> float:
       num = float(np.sqrt(np.nanmean(np.asarray(spread_ts) ** 2)))
       den = float(np.sqrt(np.nanmean(np.asarray(rmse_ts) ** 2)))
       return float(np.sqrt((n_members + 1) / n_members)) * num / den
   ```

   `n_members` is **required** — a `None` escape hatch would silently
   preserve the uncorrected value for every existing caller, contradicting
   master-plan invariant 4. Grep `figspec` and `scripts/figure_creation/`
   for callers and update each of them in this WP; the member count is
   available in `run_summary.yaml` / `config.yaml` for every run.

## WP0.3 Degenerate-member guard

New helper `src/pyurbanair/utils/da_metrics.py::ensemble_uniqueness`:

```python
def ensemble_uniqueness(members: np.ndarray) -> dict:
    """members: (M, K) flattened parameter vectors. Returns n_unique,
    n_members, min/median pairwise L2 distance. n_unique uses EXACT row
    matching: divergence-resample clones are bit-identical copies, and any
    rounding would turn legitimately near-collapsed members into false
    clone positives. Near-duplicates are the min/median pairwise
    distance's job, not n_unique's."""
```

Wire into `compute_esmda_metrics.py` (runs on the always-available parameter
files, so **before** the `skip_viz` early-return): flatten all parameter
variables of `posterior_params.nc` into `(M, K)`; also loop
`windows/window_{w}_posterior_params.nc` for per-window counts. Emit:

```yaml
ensemble_health:
  n_members: 50
  n_unique: 48
  n_unique_per_window: [50, 48]
  min_over_median_pairwise: 0.0
```

Log a warning when `n_unique < n_members`. Later phases use `n_unique` as
the effective `M` in fair corrections; for this phase, reporting is enough.

## Tests (`tests/test_da_metrics.py`, extend or create)

- Fair CRPS: for `ens ~ N(0,1)`, `M=10⁴`, truth `y`, compare against the
  analytic Gaussian CRPS `σ[z(2Φ(z)−1) + 2φ(z) − 1/√π]` to ~1e-2.
- Unbiasedness direction: for small `M` drawn from the same distribution as
  truth, the fair estimator's expectation (averaged over many draws) matches
  the large-`M` value; the old estimator overshoots. (Seeded, tolerance-based.)
- `spread_skill`: synthetic exchangeable truth/members → SSR ≈ 1 with the
  factor, ≈ `√(M/(M+1))` without.
- `ensemble_uniqueness`: constructed ensemble with two cloned rows →
  `n_unique = M−1`, `min_pairwise = 0`.
- Run the existing suite; per memory, ~28 pre-existing failures are baseline —
  stash-verify before attributing failures to this change.

## Acceptance

- All three biased sites fixed and covered by tests; `crps_reduction_vs_prior`,
  `ensemble_health`, and `metrics_version: 2` present in a smoke-run
  `run_summary.yaml`; the comparison scripts warn on version mixing; every
  `spread_skill` caller passes `n_members`; pre-commit clean; PR description
  carries the breaking-number note.

## Deviations

- The two scalar CRPS sites use the algebraically equivalent sorted-sample
  pairwise-sum identity instead of materializing an `M × M` difference tensor.
  This keeps the specified `M=10⁴` Gaussian test linear-memory.
- `metrics_version: 2` is also emitted by the filtering metric stage because it
  reuses the corrected parameter CRPS and energy-score code. Sweep metrics are
  always marked version 2 when recomputed by the updated stage, even if the raw
  run's older summary had no marker. When such a run lacks
  `truth_access.yaml`, its unrecomputed version-1 sensor scores are omitted from
  the sweep artifact rather than copied under the version-2 marker.
- No existing production caller of `figspec.metrics.spread_skill` was present,
  so there were no call sites to update when making `n_members` required.

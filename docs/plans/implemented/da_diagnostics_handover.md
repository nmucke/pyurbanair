# Handover: DA diagnostics from the September review

The September DA review (`docs/research/da_review_2026-09/summary.md`, audits
A–E) found good fits at the assimilated sensors but poor skill at held-out ones,
and recommends measuring before tuning. This PR adds the missing diagnostics
to `scripts/compute_metrics.py`. **Runs don't change**: only `metrics.yaml`
gains keys. Read `AGENTS.md`, `docs/evaluation.md` and
`docs/data_assimilation.md` first.

Branch from `main`, open the PR into `main`, don't merge it. The review was
written against the archived setup; the references below were checked against
`main` at 02b582c, so re-check each one before relying on it.

## What exists

- `compute_metrics.py` writes the blocks `parameters`, `state`, `sensors`,
  `sensor_statistics`. Prior sensor scores appear only when `prior_state` exists,
  and only the smoother can save it (`assimilation.save_prior_state`, default
  false); `run_filtering.py` and `run_hybrid.py` never write a prior state.
- Every run writes `window_*_obs.nc` with `obs`, `obs_clean`, `obs_std` and the
  forecast/analysis `pred_obs` (`scripts/utils/helper_functions.py` ~222–245,
  `scripts/run_filtering.py` ~141–149).
- Unused helpers in `libs/evaluation/src/evaluation/scores.py`: `spread_skill`
  (~322), `_skill_score` (~870), `data_mismatch` (`O_N`, ~1397; only a figure
  helper calls it).
- `CycleDiagnostics` (`obs_prior_rmse`, `obs_posterior_rmse`, `innovation_chi2`)
  is computed in `data_assimilation/filtering/base.py` (~117–148) but no longer
  written; the archived `cycle_diagnostics.yaml` is gone.

## Add (one block each in `metrics.yaml`, reusing the helpers above)

1. **Observation-space fit per window/cycle** (`observation` block): RMSE of
   forecast and analysis `pred_obs` against `obs` from `window_*_obs.nc`, their
   ratio (the forecast/analysis sawtooth), the normalised innovation χ²
   (≈ 1 when R and the spread are honest), and `O_N` for the smoother.
2. **Desroziers check:** from the same files, the estimated observation error
   std per sensor set, next to the `obs_std` the run used. This is the input for
   calibrating R in the next PR's experiments.
3. **Spread–skill ratio over time** per sensor set: add the ensemble spread
   series to `vector_sensor_metrics` (it returns only `rmse` and
   `energy_score`) and call `spread_skill`. ≈ 1 for a calibrated ensemble.
4. **Climatology baseline** per sensor set: the RMSE of predicting each
   sensor's time mean, from the truth alone, so the posterior RMSE has a
   reference.
5. **Prior baseline for filter and hybrid:** let `run_filtering.py` and
   `run_hybrid.py` write the prior state when `save_prior_state` is true, as
   the smoother does. It defaults to false, so runs stay byte-identical. Then
   `sensor_statistics` scores the prior for every method.

Skip what the review ranks as later work (lead-time sawtooth from
`window_*_forecast_state.nc`, reference-run scoring with a `--reference` run
directory) unless it falls out in a few lines; list it in the PR as next steps.

## Tests

- `tests/evaluation/`: spread–skill ≈ 1 on a synthetic calibrated ensemble;
  χ² ≈ 1 and Desroziers recovering a known σ on synthetic `window_obs` data.
- `tests/scripts/test_assimilation.py`: the new `metrics.yaml` keys exist for
  smoother, filter and hybrid (tiny overlays, no compiled solver), and runs
  without `save_prior_state` are unchanged.
- Update `docs/evaluation.md` (the `metrics.yaml` blocks and "Who calls it").

## Done when

Outcome: the hybrid's ESMDA prior state was not added (it needs ESMDA to keep
the step-0 state when `final_forecast=False`); the filter and hybrid forecasts
are scored as `forecast`, separate from the free-run `prior`.

- [x] `metrics.yaml` has the five additions for all three methods (item 5
      as revised: `forecast` for filter and hybrid, the hybrid `prior` left).
- [x] Default runs byte-identical (only the metrics output grows).
- [ ] `tests/evaluation`, `tests/data_assimilation`, `tests/scripts`,
      `pre-commit` pass; CI green on Linux and macOS.
- [x] The PR shows the new blocks for one tiny run of each method.
- [x] This file moved to `docs/plans/implemented/`.

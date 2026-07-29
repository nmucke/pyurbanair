# Phase 2 — persist observation-space arrays + the diagnostics they unlock

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Rationale: §5, §6.2 of
> [../esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md).
> Requires phases 0–1. Two PRs (WP2.1 persistence; WP2.2+2.3 diagnostics
> and figures).
>
> **Implementer: update the master_plan.md status table per WP; record
> deviations at the bottom of this file. WP2.1 changes stage-1 artifacts —
> update `docs/data_assimilation.md` in the same PR (CLAUDE.md rule).**

## Context: what exists and where

- Observations are built per window in `scripts/esmda/run_esmda.py:683–691`
  (`window_obs = truth_obs_op(window_true_state)` + `√C_D·noise`) and passed
  to the smoother; never saved.
- Predicted observations `pred_obs` (shape `(N_d, N_e)` after the transpose)
  are computed inside each `_one_step` in
  `libs/data-assimilation/src/data_assimilation/smoothing/esmda.py` —
  sites: `ParameterESMDA._one_step` (~line 399),
  `StateAndParameterESMDA._one_step` (~line 827),
  `StateAndTimeVaryingParameterESMDA._one_step` (~line 946); check whether
  `TimeVaryingParameterESMDA._one_step` (~line 512) computes its own or
  delegates — instrument wherever `pred_obs` is actually materialized, once
  per ESMDA iteration. The `_final_time_smoothing_step` pred_obs (~line 888)
  is a different operation (α=1 trajectory smoothing) — **exclude it**.
- The `_analysis` loop (lines 274–382) calls `_one_step` `num_steps` times;
  step `i`'s `pred_obs` comes from the forecast under the params of iteration
  `i`, so **step 0's pred_obs are the prior predicted observations** — exactly
  what the identifiability/contraction diagnostics need.
- Per-iteration parameter ensembles are *already* returned
  (`return_params_history=True`, concat over `esmda_step`,
  run_esmda.py:707–726) and then discarded via `.isel(esmda_step=-1)` —
  saving them is one line.
- The posterior forecast's own predicted obs are **not** computed inside the
  loop today (the final forecast has no update step). WP2.1 adds them at the
  source: one extra application of the *existing* observation operator to
  the already-computed final forecast (no extra solver run, KB-scale
  output). Stage 2 must **not** re-implement the operator's interval
  reduction — a mirrored implementation would drift from the real one and
  has nothing to be cross-checked against.
- Precedent for diagnostics plumbing: the filtering side's
  `CycleDiagnostics` (`libs/data-assimilation/.../filtering/base.py:56–92,
  634–680`).

## WP2.1 Persistence

### Config

`conf/run_esmda.yaml`, `esmda:` block:

```yaml
esmda:
  save_obs_diagnostics: true   # false reproduces the pre-phase-2 artifact set
```

Files are KB-scale (`N_d ≈ 180`/window, `M ≤ 100`, `num_steps ≈ 3`), so
default-on is fine; the flag exists for byte-identical reproduction of old
runs.

### Smoother change (minimally invasive; no return-signature changes)

In `_BaseESMDA.__init__`: `self.collect_obs_diagnostics: bool = False` and
`self.pred_obs_history: list = []`. In `_analysis`, reset the list at entry.
In each `_one_step` that materializes `pred_obs`, immediately after the
transpose:

```python
if self.collect_obs_diagnostics:
    self.pred_obs_history.append(np.asarray(pred_obs))  # (N_d, N_e)
```

(`np.asarray` to detach from JAX/device memory.) The runner reads the
attribute after the call — same pattern as the existing
`esmda.prune_disk_steps` / `pin_initial_time_point` attribute plumbing
(run_esmda.py:647, 675).

Additionally, in `_analysis` right after the final posterior forecast
(~line 356) and **before** `_final_time_smoothing_step`, when the flag is
set: compute the posterior forecast's predicted obs with the same
`_observation_step` call the `_one_step` implementations use (disk mode:
pass the final step's results dir) and append it to the history — the
history then holds `num_steps + 1` entries, entry 0 = prior, entry −1 =
posterior forecast. When `final_time_smoothing` is active the last entry
reflects the pre-smoothing forecast; record that in the file attrs. The
`_final_time_smoothing_step`-internal pred_obs (~line 888) stays excluded.

### Runner change (`run_esmda.py`, inside the window loop, after line ~726)

Gated on the flag; per window `w` write into `windows/`:

1. `window_{w}_obs.nc` — `obs (obs,)` (the noisy vector actually
   assimilated), `obs_clean (obs,)` (pre-noise, available at line 686),
   `obs_error_std (obs,)` (from `C_D`'s diagonal). Attach whatever metadata
   the truth observation operator exposes as coordinates (sensor index,
   x/y/z, component label, interval index/time — inspect
   `TemporalObservationOperator` for what is recoverable; at minimum store
   the flat obs index and document the ordering convention in the file's
   attrs, because every downstream diagnostic groups by sensor/component).
2. `window_{w}_pred_obs.nc` — `pred_obs (esmda_step, obs, ensemble)` stacked
   from `esmda.pred_obs_history` (length = `num_steps + 1`; entry 0 =
   prior, entry −1 = posterior forecast — see the smoother change above).
3. `window_{w}_params_steps.nc` — `result_params` saved **before** the
   `.isel(esmda_step=-1)` at line 726 (carries the `esmda_step` dim;
   `esmda_step=0` is the prior because `params_history` seeds with the
   input params, smoother line 310).

Set `esmda.collect_obs_diagnostics = True` next to the other attribute
assignments. Record the flag in `run_info.yaml` so stage 2 knows whether to
expect the files.

### Docs

Update the artifact table in `docs/data_assimilation.md` and the layout
comment block at the top of `run_esmda.py`.

## WP2.2 Diagnostics (stage 2)

All in `compute_esmda_metrics.py` via `ensemble_scores` /
`da_metrics`; **every block no-ops with a log line when the phase-2 files
are absent** (old run dirs must still process). Notation: `d = obs`,
`g_l (N_d, M)` = pred_obs at step `l`, `σ_D` = obs error std, `M` from the
file, un-inflated `C_D` throughout.

1. **Normalized data mismatch** (metrics doc §5): per member, per step,
   `O_N = mean_j[ ((d_j − g_j)/σ_D,j)² ] / 2`. Emit per-step
   `{median, iqr, min}` across members, the target band `0.5 ± 3/√(2N_d)`,
   and flags `underfit_final` (`median ≫` band) / `overfit_final`
   (`median ≪` band) / `collapsed` (across-member IQR of `O_N` → 0 while
   median off-target). The χ² target assumes `C_D` includes
   representativeness error, which it does not until WP3.3 lands — so the
   flags are **advisory** until then: whenever
   `esmda.obs_error_representativeness` is absent or null, emit
   `caveat: no_representativeness_error` in this block *and* in
   `innovations`. The caveat must live in the artifact, not only in this
   plan; the D3/D4 figures annotate it when present.
2. **Innovations** (prior = step 0): per obs
   `z_j = (d_j − mean_m g_0)/√((1+1/M)·var_m(g_0) + σ_D,j²)`; emit pooled
   `{mean, std}`, `chi2_norm = mean(z²)`, and per-sensor `{mean, std}`
   (grouping via the obs metadata from WP2.1). Sub-sample intervals if the
   per-sensor series are strongly autocorrelated before quoting stds.
3. **Contraction vs achievable** (metrics doc §3): from step-0 params
   (`window_{w}_params_steps.nc`, `esmda_step=0`) and `g_0`: anomaly
   matrices `/(√(M−1))` → `C_θθ, C_θd, C_dd`;
   `C_pred = C_θθ − C_θd (C_dd + C_D)⁻¹ C_dθ` via Cholesky solve
   (`C_dd + C_D` is full-rank thanks to diagonal `C_D`);
   `r_i = σ_post,i/√(C_pred,ii)` per parameter → `{value, spurious:
   r < 0.5}`; report `N_d` and `M` alongside (the estimate is noisy when
   `N_d > M−1`).
4. **Identifiability table** (metrics doc §3): `S_ij = cov(θ_i, g_0,j)/
   (σ_θi σ_D,j)`; `SNR_j = std_m(g_0,j)/σ_D,j`; DFS from the whitened
   anomaly SVD `Y = diag(1/σ_D) G_anom`, `DFS = Σ λ_k²/(1+λ_k²)`; emit
   `{dfs, dfs_cap: min(N_d, M−1), n_params, snr: per-sensor summary,
   n_obs_snr_below_1}`.
5. **Obs-space verification** of the posterior: consume the persisted
   posterior entry `pred_obs[esmda_step=-1]` from WP2.1 — stage 2 performs
   no observation-operator re-derivation. Fair CRPS + CRPSS vs the prior
   `g_0`, variogram score
   `VS_0.5` over sensor pairs (weights `1/distance` from sensor positions),
   rank histogram + PIT of `d` in the prior and posterior ensembles
   (held-out sensors are phase 3; until then label these
   `assimilated_obs` and treat as consistency, not skill), and the binned
   spread–skill (10 equipopulated spread bins; include the same-M
   calibrated synthetic reference: resample "truth" from the ensemble to get
   the attenuated reference slope).

Schema (all new top-level keys, additive):

```yaml
esmda_diagnostics:
  data_mismatch: {per_step_median: [...], per_step_iqr: [...],
                  target: 0.5, target_band: ..., overfit_final: false,
                  underfit_final: false, collapsed: false,
                  caveat: no_representativeness_error}
  innovations: {mean: ..., std: ..., chi2_norm: ...,
                per_sensor: {...}, caveat: no_representativeness_error}
  contraction_vs_achievable: {inflow_angle: {value: ..., spurious: false}, ...}
  identifiability: {dfs: ..., dfs_cap: ..., n_obs_snr_below_1: ...,
                    snr_min: ..., snr_median: ...}
  obs_space: {crps: ..., crpss_vs_prior: ..., variogram_score: ...,
              pit_counts: [...], spread_skill_binned: {...}}
```

## WP2.3 Figures, second wave

New plot functions + calls in `make_esmda_figures.py`, each no-op when
inputs are missing:

| ID | Function | Content |
|---|---|---|
| D3 | `plot_data_mismatch_decay` | per-member `O_N` boxes vs step (0 = prior), horizontal target band at ½, log-y when the drop spans decades; annotate the representativeness caveat when the YAML carries it |
| D4 | `plot_innovation_consistency` | z-histogram + N(0,1) overlay, annotated `chi2_norm`; companion normal Q–Q panel; annotate the representativeness caveat when the YAML carries it |
| D1 | `plot_rank_histograms` | prior \| posterior PIT bars, uniform line + binomial consistency band; caption states U/dome/slope reading |
| D2 | `plot_spread_vs_error` | time-series RMSE vs corrected spread, plus the binned-reliability panel with the same-M reference line |
| P2 | `plot_parameter_iterations` | full per-iteration member trajectories from `window_{w}_params_steps.nc` (x = esmda_step, 0 = prior, truth line); per-iteration boxes instead of spaghetti if resampling broke member identity that window (`ensemble_health.n_unique_per_window`) |

## Tests

- Unit: `O_N` on a constructed linear-Gaussian toy (posterior samples →
  mean `O_N ≈ ½`); innovation z on synthetic exchangeable ensembles →
  std ≈ 1; `C_pred` against the analytic Kalman posterior covariance on the
  same toy; DFS caps (`≤ min(N_d, M−1)`) on random anomalies; variogram
  score zero for identical fields.
- Integration: smoke run with the flag on → files exist with expected dims
  (`esmda_step == num_steps + 1` for both pred_obs and params_steps — both
  now seed with the prior and end with the posterior; verify against the
  seeded history at smoother line 310 and the WP2.1 posterior append);
  `esmda_diagnostics` keys present and finite. Smoke run with the flag off →
  artifact set identical to pre-phase-2 (directory listing comparison).
  Stage 2 on a phase-1-era run dir → no `esmda_diagnostics` keys, no crash.
- The smoke ESMDA config runs `num_steps=3`, `M=2`-ish — several diagnostics
  degenerate at M=2 (variances with ddof=1, binned spread–skill). Guard the
  degenerate paths (emit `null` + log) rather than special-casing tests.

## Acceptance

- Flag-off byte-compatibility demonstrated; all diagnostics no-op cleanly on
  old run dirs; docs updated; D1–D4 + P2 render on a smoke run;
  `run_summary.yaml` schema exactly as above (additive).

## Deviations

_(record here as they occur)_

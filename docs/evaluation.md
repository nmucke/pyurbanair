# Evaluation metrics and figures

The `evaluation` library (`libs/evaluation`, import `evaluation`) holds the
metric and figure code for ensemble data-assimilation runs. It is a **leaf
library**: plain functions, NumPy arrays / `xarray` objects in, floats, dicts
and figures out. It never imports `pyurbanair`, `data_assimilation`, JAX or a
backend, and knows nothing about Hydra or the run-directory layout; the scripts
own I/O and orchestration and call in here. Its only dependencies are numpy,
scipy, xarray, netcdf4 and matplotlib. It is an editable dependency of the
`dev` Pixi environment (feature `evaluation`).

Nothing is re-exported from the package root (so `scores` never pulls in
matplotlib). Import from the module you need:

```python
from evaluation.scores import crps_ensemble, window_statistics_summary
from evaluation.sensors import window_statistics
from evaluation.turbulence import colocate_components
from evaluation.figures import plot_rank_histogram
```

Design rules (from the package docstring): five flat modules, no base classes,
no registries. The only class with state is
`turbulence.MomentAccumulator`.

## Modules

### `evaluation.scores`: probabilistic ensemble scores

- **Estimators**: `crps_ensemble` (fair CRPS, pairwise term over the
  `M(M-1)` off-diagonal pairs), the energy score (multivariate CRPS, inside
  `vector_sensor_metrics`), `ensemble_rank` (rank of the truth in `0..M`, ties
  broken at random with a fixed seed), `spread_skill`, `z_score_stats` /
  `calibrated_z_std`, `hit_rate` (VDI 3783/9 `q`), `data_mismatch` and
  `data_mismatch_summary` (normalized data mismatch `O_N` against the ½ target
  band), `observation_fit` (forecast/analysis RMSE against the noisy
  observations, the diagonal innovation χ² `innovation_chi2_diag`, and the
  Desroziers estimate `sqrt(mean(d_a·d_f))` of the observation error std, for
  one update).
- **Field / parameter metrics**: `field_rmse`, `field_rmse_timeseries`,
  `normalized_field_rmse`, `param_metrics`, `compute_parameter_metrics`
  (per-parameter RMSE and CRPS series of posterior and prior),
  `parameter_bundle` / `compute_parameter_bundles` / `parameter_metric_summary`
  (z-score, normalized error, contraction ratio per knot).
- **Sensor metrics**: `vector_sensor_metrics` (RMSE, energy score and
  ensemble spread of the `(u, v, w)` vector per time step; the spread uses the
  same vector norm as the RMSE, so `spread_skill` compares like with like),
  `compute_sensor_metrics`, `sensor_rmse`.
- **Window statistics**: `window_statistics_summary` scores every
  `statistic x quantity` pair (`mean_u`, `variance_magnitude`, ...) with CRPS,
  z-score and rank, posterior and (when given) a reference ensemble, and
  returns the `sensor_statistics` block including the `rank_counts` the rank
  histogram reads. `reference` names the reference block and its skill keys
  (`prior` by default, `forecast` for a filter's forecasts).
- **Distributions**: `wasserstein2` (1-D W2 from `N_QUANTILES = 99` quantiles,
  split exactly into `location` (Δμ)², `scale` (Δσ)² and `shape`, the rest),
  `kl_divergence` (KL(truth ‖ pred) in nats on `N_BINS = 30` shared bins with
  0.5-count smoothing; it depends on the binning, grows without bound as the
  supports separate and is asymmetric, so read it beside W2),
  `distribution_scores` (both, pooled over sensors and members, per sensor
  normalised by its truth std and averaged, and the median over members) and
  `shared_histograms` (densities on shared bins and quantiles, for a figure).
  `sensor_distribution_summary` builds one sensor set's
  `sensor_distributions` block and figure arrays from them.
- `series_stats` reduces a 1-D series to `{mean, final, max, min}`;
  `window_series_stats` adds the `per_window` values.
- `member_correlation`: the parameters' correlation matrix over the members,
  per window.

`METRICS_VERSION = 2` marks the switch to the fair (`M(M-1)`) estimators;
scores from before it are ~O(1/M) larger and not comparable.

### `evaluation.sensors`: window statistics of sensor series

Consumes sensor series that the caller has already extracted:
`(component, [ensemble,] time, sensor)` `DataArray`s with a global `time`
coordinate. Extraction needs the observation operator, so it lives in
`scripts/utils/helper_functions.py` (`sensor_series`), not here.

- `sensor_magnitude`: `|U|` from the `component` dim.
- `window_masks`: bins frames into assimilation windows by time coordinate,
  so truth and ensemble may have different output cadences. Window `w` holds
  the frames in `(w·sim_time, (w+1)·sim_time]`: every backend stamps its
  output on `(0, sim_time]`, and the scripts put each window file on the
  global axis by its end (`global_time` in `scripts/utils/helper_functions.py`).
- `window_statistics`: per-window mean and variance (`ddof=1`) of `u`, `v`, `w`
  and `|U|`, each `(window, quantity, [ensemble,] sensor)`.
- `window_sampling_std`: block-bootstrap sampling std of each of those
  statistics (the identifiability floor).

The scored object is the window statistic, not the instantaneous time series:
members decorrelate within an eddy turnover, so pointwise errors mostly measure
phase.

### `evaluation.turbulence`: flow statistics over state fields

Everything streams; window state files `(ensemble, time, z, y, x)` are never
loaded whole.

- `select_z_plane` selects one level of a state; `on_grid` interpolates a
  `(z, y, x)` field linearly onto another grid's cell centres, only along the
  axes that differ (a truth from another solver).
- `stl_solid_mask(stl_path, z, y, x)`: the building cells of the case STL
  (`read_binary_stl`) on a grid of cell centres, in the STL's frame (every
  backend writes its state in it). A cell is solid when its centre is at or
  below the highest point where the vertical ray through its column meets the
  mesh, or within `SURFACE_TOLERANCE` (1e-3 of the smallest cell size) of it,
  so a centre on a wall or a roof is solid. That is pylbm's voxelisation and
  PALM's topography, and on the Xie & Castro grid it equals pylbm's mask cell
  for cell. The geometry is taken as 2.5-D: the space under an overhang or a
  bridge counts as solid, as in pylbm and PALM. Inside a building the backends
  write different things, which is why the mask cannot come from the data:
  PALM exact zeros, uDALES near-zero leftovers, pylbm arbitrary values on its
  solid nodes (up to ~1 m/s next to walls).
- `colocate_components(ds, solver_name)`: interpolates staggered `u, v, w`
  onto cell centres per backend, so one-point moments (Reynolds stresses, TKE)
  are formed at one point. `extrapolated_centre_dims` names the dims whose last
  index is extrapolated.
- `MomentAccumulator`: chunk-wise mean and second moments `<u_i'u_j'>`.
- Field statistics, per member: `field_statistics` (time-mean u, v, w, TKE and
  resolved u′w′ of colocated `(time, z, y, x)` fields), `fluid_mask`
  (`stl_solid_mask` dilated by one cell, so centres whose colocation reads a
  solid face drop out), `intrinsic_profile` (per-level mean over the fluid
  cells) and `statistic_rmse` (RMSE over the fluid cells, total and per
  level). `member_field_reductions` applies them (and the spectra below) to
  one member; `score_window_fields` scores one window's sources against the
  truth on the posterior's grid, and `field_metric_blocks` turns the windows
  into the `field_statistics`, `canopy_profiles` and `spectra` blocks and
  their `diagnostics.nc` arrays.
- Spanwise spectra: `spanwise_spectra` FFTs a component on its **native**
  grid (interpolation would low-pass the tail) along the periodic y, on every
  fully fluid `(x, z)` line from the first building to two cells before the
  outflow, grouped `above_canopy` (levels without solid, below the top two)
  and `in_canopy` (open streets along y; a case without them, likely
  Barcelona, has none). `band_energy_ratio` gives the prediction/truth energy
  in dB in `SPECTRAL_BANDS`: large `λ > 8Δ`, mid `4Δ–8Δ`, near cutoff `2Δ–4Δ`.
- `rolling_tke`, `sensor_tke_evolution`: rolling resolved TKE at sensors,
  members kept separate.
- `integral_time_scale`, `block_bootstrap_std`: moving-block bootstrap
  standard errors with the block length from the integral time scale.
- Spectra: `welch_spectrum`, `probe_spectra` (matched truth / member spectra
  at the probes), `median_spectrum`, `log_spectral_distance`,
  `spectral_metric_summary`, plus `spectral_band_bins` /
  `minimum_spectral_samples` for record-length checks.

### `evaluation.figures`: figure builders

Each function takes arrays or Datasets plus an `output_path` and writes the
file.

- General plots: `plot_rollout_time_evolution` (parameter trajectories over
  windows), `plot_parameter_error`, `plot_sensor_timeseries`,
  `plot_final_state_with_obs`.
- Assimilation statistics (from `diagnostics.nc`): `plot_parameter_pairs`,
  `plot_canopy_profiles`, `plot_sensor_distributions`, `plot_spanwise_spectra`.
- The evaluation figure set (IDs from
  [esmda_turbulence_evaluation.md](research/esmda_turbulence_evaluation.md) §7):
  `plot_parameter_marginals` (P1), `plot_station_profiles` (S1),
  `plot_sensor_fans` (S5), `plot_mean_slices` (F1), `plot_tke_slices`,
  `plot_rank_histogram` (D1), `plot_spectra` (S4), `plot_data_mismatch_decay`
  (D3), `plot_tke_time_evolution`, `plot_tke_error_evolution`.

The rank-histogram pipeline (window statistic -> rank per knot -> pooled
counts -> histogram) is derived step by step in
[rank_histogram_math.md](research/rank_histogram_math.md).

### `evaluation.style`: figure conventions

Shared colours and labels (`COLORS`, `MODEL_*`, `METHOD_*`, `PARAM_LABELS`,
`PARAM_UNITS`, colormaps), `apply_style`, window shading (`shade_windows`,
`mark_windows`), bands (`band`, `nested_bands`), `finite_limits`,
`save_pdf` / `save_png` and `write_table` (CSV plus a booktabs `.tex`).

## Who calls it

| Script | Uses |
|---|---|
| [scripts/compute_metrics.py](../scripts/compute_metrics.py) | `compute_parameter_metrics`, `series_stats`, `vector_sensor_metrics`, `spread_skill`, `_skill_score`, `window_statistics_summary`, `window_series_stats`, `member_correlation`, `sensor_distribution_summary`, `window_statistics`, `window_sampling_std`, `quantity_series`, `observation_fit`, `data_mismatch`, `data_mismatch_summary`, `member_field_reductions`, `score_window_fields`, `field_metric_blocks` |
| [scripts/visualize_assimilation.py](../scripts/visualize_assimilation.py) | `plot_rollout_time_evolution`, `plot_final_state_with_obs`, `plot_sensor_timeseries`, `plot_tke_time_evolution`, `plot_rank_histogram`, `plot_parameter_pairs`, `plot_canopy_profiles`, `plot_sensor_distributions`, `plot_spanwise_spectra`, `sensor_magnitude`, `colocate_components`, `select_z_plane`, `sensor_tke_evolution` |
| [scripts/visualize_forward.py](../scripts/visualize_forward.py) | `colocate_components` |

Both assimilation scripts take a finished run directory (`config.yaml`,
`run_info.yaml`, `true_params.nc`, the truth state and
`windows/window_{w}_{prior,posterior}_{params,state}.nc`, `window_{w}_obs.nc`).
`compute_metrics.py <run dir> [key=value ...]` (overrides of the run's
`config.yaml`, e.g. `assimilation.replica_dir=<dir>`) writes `metrics.yaml`.

**Which numbers are the scores.** The truth and every member are driven by
different turbulent realisations, which decorrelate within seconds, so an
instantaneous comparison measures eddy phase, whatever the parameters are.
The scores are the **statistics**: `field_statistics`, `canopy_profiles`,
`sensor_statistics`, `sensor_distributions` and `spectra`. The instantaneous
`sensors` and `spread_skill` blocks are sanity checks (the spread–skill ratio
reads about 1 even with wrong parameters). Read every score against the
**replica floor** (`replica`, from `assimilation.replica_dir`: the truth rerun
with another turbulence seed, scored as a one-member prediction) and, for the
distributions, `truth_halves`: a posterior at the floor is as good as a
perfect model can be.

"Sources" below are the posterior, `prior` (`assimilation.save_prior_state`,
smoother), `forecast` (`assimilation.save_forecast_history`, filter and
hybrid) and `replica`, each when present; the prior, forecast and replica are
read on the posterior's time stamps. Each series is summarised as `{mean,
final, max, min}`; the window-indexed ones also list `per_window` values (a
time-varying parameter's knots averaged within each window):

| Block | Holds |
|---|---|
| `parameters` | per parameter: posterior and prior RMSE and CRPS against the truth, and the reduction |
| `parameter_correlation` | `posterior` and `prior`: the correlation matrix of the estimated parameters over the members, final window, as a nested mapping (a time-varying parameter enters as its window mean) |
| `sgs_health` | only with `assim_model.forward_model.model_discrepancy.enabled`: per window, `posterior` (and `prior`), the SGS multiplier min and max and the largest saturation fraction over the members, from each state file's `model_discrepancy_by_member` attribute |
| `field_statistics` | per statistic (`u`, `v`, `w` time means, `tke`, `uw`; per member and window on the cell centres, averaged over the members, never the statistic of the mean field) and source: `rmse` against the truth's over the fluid cells (`fluid_mask` of `geometry.stl_path`) per window, and `level_rmse` per height (`z` listed once), RMS over the windows. A truth on another grid is interpolated onto the ensemble's centres |
| `canopy_profiles` | per profile (`u`, `tke`, `uw`, intrinsic averages over each level's fluid cells) and source: `profile_rmse`, the RMSE over z of the ensemble-mean profile against the truth's, per window |
| `sensor_distributions` | per sensor set, quantity (`u`, `v`, `w`, `magnitude`) and source: `distribution_scores` per window of its sensor values (frames × sensors × members) against the truth's: `w2` with `location`/`scale`/`shape`, `kl`, `*_per_sensor`, `*_member_median`; `truth_halves`, the truth's first half-window against its second (assumes stationarity) |
| `spectra` | per component, height group and source: the `large`, `mid` and `near_cutoff` band energy ratios (dB) and the `log_spectral_distance` of the members' median spanwise spectrum against the truth's, per window. A truth spectrum on another grid is interpolated onto the ensemble's wavenumbers (NaN beyond its own cutoff). A group without fully fluid lines is null, and a log line says so |
| `sensors` | sanity check, per sensor set (`assimilation`, `validation`): RMSE and energy score of the instantaneous `(u, v, w)` vector |
| `spread_skill` | sanity check, per sensor set: the spread on the same vector norm and its `ratio` (≈ 1 when calibrated); `prior_ratio` when the prior states were saved |
| `climatology` | per sensor set: RMSE of predicting each sensor's time mean of the clean truth, and `rmse_skill_vs_climatology` of the posterior |
| `sensor_statistics` | per sensor set: per-window mean and variance of u/v/w/\|U\| scored with CRPS, z-score and rank, per source |
| `observation` | per stage (`smoother`: one value per window; `filter`: one per cycle): `forecast_rmse`, `analysis_rmse`, `rmse_ratio`, `innovation_chi2_diag`; the smoother's `data_mismatch` (O_N) |
| `desroziers` | per stage with an analysis: `obs_std_estimated`, the `obs_std_used` (RMS, after aggregation) and their `ratio` |

A hybrid has both stages, the smoother's from `window_{w}_obs.nc` and the
filter's from `window_{w}_filter_obs.nc`. Its smoother stops before the
posterior forecast, so its `smoother` stage has `data_mismatch` only, over the
forecasts before each ESMDA update.

Read these with their limits:

- **`prior` and `forecast` are different baselines.** The prior is the free run
  with the prior parameters; a filter forecast already contains every earlier
  analysis, so it does not compare across methods the way the prior does.
- **The climatology is in-sample.** The time mean comes from the record it is
  scored on, so it is a reference level, not a forecast any method could have
  issued.
- **`innovation_chi2_diag` ignores correlations.** It is the diagonal of the
  normalised innovation χ² (`d_f² / (var_ens + σ²)`, `ddof=1`), not the
  full-matrix χ² of `CycleDiagnostics`.
- **Desroziers needs both residuals from the same update.** For an ESMDA
  smoother the "analysis" is the posterior forecast after all steps.
  `obs_std_estimated` is `null` when `mean(d_a·d_f) ≤ 0`.
- **Desroziers and χ² assume consistent error statistics.** With a collapsed
  spread the gain is about 0, so `d_a ≈ d_f` and Desroziers returns the
  innovation RMS, which says nothing about R. Read both only where
  `spread_skill.ratio` and `innovation_chi2_diag` are near 1.
- **The held-out sensors have no observations**, so `observation` and
  `desroziers` cover the assimilated sensors only.
- **Resolved stresses only.** TKE and u′w′ are the resolved parts; the outputs
  carry no SGS stress. Fair within one solver; across solvers (Round 2 of
  [sgs_discrepancy_twin_tests.md](plans/sgs_discrepancy_twin_tests.md)) the
  two codes resolve different fractions.
- **The near-cutoff band partly measures numerics.** Across solvers the two
  codes' numerical dissipation differs near 2Δ, so a mismatch there is not
  only physics; the large band is where inflow errors show.
- **Distribution sample sizes are the realisation's**, not the frame count:
  frames are autocorrelated. The KL value depends on its binning.
- **The domain is small**: 80 m in y gives 20 wavenumbers on Xie–Castro; the
  averaging over lines, frames and members is what makes the spectra usable.

All of it comes from one pass over the window files, one member at a time:
memory is one member's window plus running totals.

It also writes `diagnostics.nc` next to `metrics.yaml`, the arrays the figures
need: `{posterior,prior}_parameter_members` (window, ensemble, parameter),
`{posterior,prior}_parameter_correlation` (window, parameter, parameter_j),
`true_parameter` (parameter; NaN where the truth varies in time),
`profile_<source>` (window, [ensemble,] profile_quantity, z),
`spectrum_<source>` (window, [ensemble,] component, group, k; `k` in
cycles/m, `spectrum_dy` the spacing), `building_height` (lowest and highest
top), and per sensor set `sensor_bin_edges_<set>`, `sensor_density_<set>` and
`sensor_quantiles_<set>` (pooled over sensors, windows and members). The
truth and the replica have no ensemble dim.

`visualize_assimilation.py`
writes PNGs into `<run dir>/figures/` and reads `rank_counts` from
`metrics.yaml` for `rank_histogram.png` and `diagnostics.nc` for
`parameter_pairs.png` (the final-window posterior members over the prior,
truth marked when static), `canopy_profiles.png` (final-window ⟨ū⟩, ⟨TKE⟩,
⟨u′w′⟩ against z: truth, replica, member bands and means, building heights
shaded), `sensor_distributions_<set>.png` (per quantity, densities on shared
bins and a Q–Q plot against the truth) and `spectra.png` (u, v, w above and in
the canopy: truth, replica, posterior median with its 10–90 % band, prior
median; the three bands shaded and a k^(-5/3) guide), so run
`compute_metrics.py` first; without it those figures are skipped with a
message.
[workflows/assimilation_workflow.sh](../workflows/assimilation_workflow.sh)
runs both after the assimilation run.

The other functions (`hit_rate`, the probe spectra,
the P1/S1/S5/F1/S4/D3 figures) have no caller in `scripts/`
today; only the tests exercise them.

## Tests

```bash
pixi run -e dev python -m pytest tests/evaluation
```

`tests/evaluation/test_evaluation_library.py` guards the leaf-library rule
(no forbidden imports) alongside the numerical tests.

Background and formulas: [esmda_turbulence_evaluation.md](research/esmda_turbulence_evaluation.md)
(§3–§7); the original build plan is in
[plans/implemented/esmda_evaluation/](plans/implemented/esmda_evaluation/master_plan.md).

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
  band).
- **Field / parameter metrics**: `field_rmse`, `field_rmse_timeseries`,
  `normalized_field_rmse`, `param_metrics`, `compute_parameter_metrics`
  (per-parameter RMSE and CRPS series of posterior and prior),
  `parameter_bundle` / `compute_parameter_bundles` / `parameter_metric_summary`
  (z-score, normalized error, contraction ratio per knot).
- **Sensor metrics**: `vector_sensor_metrics` (RMSE and energy score of the
  `(u, v, w)` vector per time step), `compute_sensor_metrics`, `sensor_rmse`.
- **Window statistics**: `window_statistics_summary` scores every
  `statistic x quantity` pair (`mean_u`, `variance_magnitude`, ...) with CRPS,
  z-score and rank, posterior and (when given) prior, and returns the
  `sensor_statistics` block including the `rank_counts` the rank histogram reads.
- `series_stats` reduces a 1-D series to `{mean, final, max, min}`.

`METRICS_VERSION = 2` marks the switch to the fair (`M(M-1)`) estimators;
scores from before it are ~O(1/M) larger and not comparable.

### `evaluation.sensors`: window statistics of sensor series

Consumes sensor series that the caller has already extracted:
`(component, [ensemble,] time, sensor)` `DataArray`s with a global `time`
coordinate. Extraction needs the observation operator, so it lives in
`scripts/utils/helper_functions.py` (`sensor_series`), not here.

- `sensor_magnitude`: `|U|` from the `component` dim.
- `window_masks`: bins frames into assimilation windows by time coordinate,
  so truth and ensemble may have different output cadences.
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

- `streaming_state_rmse`: per-time RMSE of `|U|` between truth and an
  ensemble-mean state on a few z-levels; `select_z_plane`,
  `evenly_spaced_levels` pick the levels.
- `colocate_components(ds, solver_name)`: interpolates staggered `u, v, w`
  onto cell centres per backend, so one-point moments (Reynolds stresses, TKE)
  are formed at one point. `extrapolated_centre_dims` names the dims whose last
  index is extrapolated.
- `MomentAccumulator`: chunk-wise mean and second moments `<u_i'u_j'>`.
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

- General plots: `plot_rollout_time_evolution` (parameter trajectories and
  `|U|` RMSE over windows), `plot_parameter_error`, `plot_sensor_timeseries`,
  `plot_final_state_with_obs`.
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
`save_pdf` / `save_png`, `write_table` (CSV plus a booktabs `.tex`), and the
geometry helpers `read_binary_stl` / `stl_solid_mask`.

`stl_solid_mask` skips columns with fewer than two z-crossings, so on an STL
without ground triangles under the buildings (Xie & Castro) it marks no solid
cells. Check the mask before relying on it.

## Who calls it

| Script | Uses |
|---|---|
| [scripts/compute_metrics.py](../scripts/compute_metrics.py) | `compute_parameter_metrics`, `series_stats`, `vector_sensor_metrics`, `window_statistics_summary`, `window_statistics`, `window_sampling_std`, `streaming_state_rmse` |
| [scripts/visualize_assimilation.py](../scripts/visualize_assimilation.py) | `plot_rollout_time_evolution`, `plot_final_state_with_obs`, `plot_sensor_timeseries`, `plot_tke_time_evolution`, `plot_rank_histogram`, `sensor_magnitude`, `colocate_components`, `select_z_plane`, `sensor_tke_evolution`, `streaming_state_rmse` |
| [scripts/visualize_forward.py](../scripts/visualize_forward.py) | `colocate_components` |

Both assimilation scripts take a finished run directory (`config.yaml`,
`true_params.nc`, the truth state and
`windows/window_{w}_{prior,posterior}_{params,state}.nc`).
`compute_metrics.py` writes `metrics.yaml` with the blocks `parameters`,
`state`, `sensors` and `sensor_statistics`; `visualize_assimilation.py`
writes PNGs into `<run dir>/figures/` and reads `rank_counts` from
`metrics.yaml` for `rank_histogram.png`, so run `compute_metrics.py` first.
[workflows/assimilation_workflow.sh](../workflows/assimilation_workflow.sh)
runs both after the assimilation run.

The other functions (`hit_rate`, `data_mismatch*`, spectra, `MomentAccumulator`,
the P1/S1/S5/F1/S4/D3 figures, `stl_solid_mask`) have no caller in `scripts/`
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

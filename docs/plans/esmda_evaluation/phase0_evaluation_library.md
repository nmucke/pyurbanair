# Phase 0 — the `libs/evaluation` library: setup and code move

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Two PRs (WP0.1 skeleton; WP0.2 move).
> No behavior changes anywhere in this phase — WP0.2 is a pure refactor and
> must leave every emitted number and figure byte-identical.
>
> **Implementer: update the master_plan.md status table per WP; record
> deviations at the bottom of this file. Follow the master plan's
> Implementation process: Opus 5 agent team, tests in the same PR, two
> adversarial review rounds, CI green before merge.**

## WP0.1 Library skeleton + wiring — size XS

Create the lib with the structure from the master plan (five flat modules,
empty or docstring-only bodies is fine at this point):

- `libs/evaluation/pyproject.toml`: copy the shape of
  `libs/data-assimilation/pyproject.toml` (hatchling + pixi-build sections);
  `name = "evaluation"`, packages `["src/evaluation"]`, dependencies
  `numpy`, `scipy`, `xarray`, `netcdf4`, `matplotlib` — **not** jax, and
  never `pyurbanair` or a backend (leaf-library invariant 5).
- Root `pyproject.toml`: add
  `[tool.pixi.feature.evaluation.pypi-dependencies] evaluation = { path =
  "libs/evaluation", editable = true }` next to the existing
  `data-assimilation` block (~line 100), and add `"evaluation"` to the same
  environment feature lists that carry `data-assimilation` (`dev`,
  `delftblue`, `snellius`, `cuda`, ~lines 294–299).
- Smoke check: `pixi run -e dev python -c "import evaluation"`.
- Docs: add the lib to the repo map in `docs/codebase_guide.md` (one row)
  and note it in `docs/scripts_and_configs.md` where the metric/figure
  stages are described. In WP0.2, also fix the existing rows that point at
  moved files (`docs/codebase_guide.md` references
  `src/pyurbanair/plotting.py`; `docs/scripts_and_configs.md` references
  `figspec/metrics.py`) — the acceptance grep below will catch them.

Design rules, restated because they are the point of this library
(master-plan invariant 5): plain functions, arrays/`xr.Dataset`s in →
floats/dicts/`Figure`s out; no Hydra config objects, no run-dir layout
knowledge, no registries or base classes; the only class is the streaming
moment accumulator (WP1.4). Scripts own I/O and orchestration.

## WP0.2 Move existing metric and plotting code — size M

Pure refactor: move, update imports at every call site, delete the old
locations. **No compatibility shims / re-export stubs** — the caller list
is small enough to update in one PR (grep hits as of 2026-08-03:
`scripts/_common.py`, `scripts/esmda/*`, `scripts/filtering/*`,
`scripts/figure_creation/*`, `scripts/figspec/*`,
`tests/test_model_error_parameters.py`).

What moves where:

| From | To |
|---|---|
| `src/pyurbanair/utils/da_metrics.py` (whole file) | `evaluation.scores` |
| `src/pyurbanair/plotting.py`: `_crps_ensemble`, `compute_parameter_metrics`, `compute_sensor_metrics` | `evaluation.scores` |
| `src/pyurbanair/plotting.py`: all `plot_*` + their private helpers (`_save`, `_shade_windows`, …) | `evaluation.figures` |
| `scripts/figspec/metrics.py` (whole file) | `evaluation.scores` |
| `scripts/figspec/style.py`; from `mask.py` only the pure geometry (`read_binary_stl`, `stl_solid_mask`), with the STL path and grid as **mandatory arguments** | `evaluation.style` |
| `scripts/esmda/_esmda_common.py`: `_energy_score`, `vector_sensor_metrics`, `parameter_metric_summary`, `series_stats` | `evaluation.scores` |
| `scripts/esmda/_esmda_common.py`: `sensor_magnitude` + other pure series reductions | `evaluation.sensors` |
| `scripts/esmda/_esmda_common.py`: `select_z_plane`, `_horizontal_coord`, `_vel_field_4z`, `streaming_state_rmse` | `evaluation.turbulence` |

Two moved `plot_*` functions import `pyurbanair.utils` helpers
(`add_velocity_magnitude` in `plotting.py:8`,
`get_velocity_magnitude_field` in `plot_rollout_time_evolution`). Those
helpers also serve non-evaluation flows and stay put — duplicate the few
lines into `evaluation.figures` (or take a precomputed magnitude array as
argument); do not import `pyurbanair` from the lib.

What deliberately stays in `scripts/`:

- `_esmda_common.py` orchestration: `load_run_config`, `read_yaml` /
  `write_yaml`, `_to_native`, `build_sensor_sets`, `open_truth`,
  `truth_x_min` — these know run-dir layout / Hydra config, which the lib
  must not.
- **Sensor-series extraction** (`_sensor_component_timeseries`,
  `_concat_sensor_pieces`, `ensemble_sensor_series`,
  `truth_sensor_series`): these import `data_assimilation`'s
  `ObservationOperator` / interpolation helpers (which pull in jax) and
  call `open_truth` — moving them would violate the leaf-library rule.
  Extraction stays in `_esmda_common.py`; `evaluation.sensors` consumes
  the extracted `(ensemble, time, sensor)` arrays.
- The `dataio`-bound wrappers of `mask.py` (`truth_solid_mask`,
  `mask_for_slice`, `nearest_z_index`) — they hard-code repo data
  locations via `figspec.dataio`, which stays.
- `scripts/figspec/dataio.py`, `figcommon.py` and everything in
  `scripts/figure_creation/` — sweep-figure orchestration; they now import
  from `evaluation` instead of `figspec.metrics` / `figspec.style` /
  `pyurbanair.plotting`.
- `src/pyurbanair/animation.py` and `utils/animation_utils.py` — animation
  is forward-model visualization, not evaluation; out of scope.

Deduplication in this WP is limited to the one verified-identical pair:
`da_metrics.per_knot_crps` and `plotting._crps_ensemble` implement the
same biased estimator formula-for-formula — collapse them to one function.
The `figspec.metrics` vs `plotting.py` parameter/sensor helpers look
similar but are **different estimators** (mean-trajectory RMSE with ddof=0
vs per-knot member-RMSE/CRPS) with different callers — move them as
distinct functions and defer any unification to phase 1, where numbers may
move under `metrics_version: 2`.

Typing: pre-commit runs strict mypy on staged files, and much of the moved
code is unannotated (`_esmda_common.py` carries a file-level waiver
today). Give each moved module a `# mypy: ignore-errors` header in this WP
— dropping the waivers is cleanup for later, not part of the move.

Verification that the refactor is inert:

- Run `compute_esmda_metrics.py` + `make_esmda_figures.py` on an existing
  smoke run dir before and after; diff `run_summary.yaml` (byte-identical)
  and eyeball one figure.
- Full test suite via CI (local collection crashes on this Mac — see
  auto-memory); failure set identical to baseline.
- `grep -rn "figspec.metrics\|pyurbanair.plotting\|da_metrics"` returns no
  hits outside `libs/evaluation` and git history.

## Deviations

_(record here as they occur)_

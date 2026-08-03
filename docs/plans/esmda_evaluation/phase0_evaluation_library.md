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

**WP0.1**

- The lib's own `[tool.pixi.pypi-dependencies]` self-reference uses
  `path = "."` (the `libs/pylbm` / `libs/pypalm` shape), not
  `libs/data-assimilation`'s `path = "libs/data-assimilation"` — that value
  is root-relative and does not resolve from inside the lib directory. The
  root `pyproject.toml` feature block is root-relative as the plan specifies.
- `libs/data-assimilation/pyproject.toml` also lists `pyurbanair` as a pixi
  pypi-dependency; `libs/evaluation` deliberately does not (invariant 5).
- `src/evaluation/__init__.py` re-exports nothing (unlike
  `data_assimilation`'s root re-export). `style`/`figures` import matplotlib,
  so a root re-export would pull it into every consumer of a metric function.
  Callers import from the module: `from evaluation.scores import ...`.
- `[tool.pixi.workspace] platforms` lists `osx-arm64` **and** `linux-64`
  (`libs/data-assimilation` lists only the former). This is a pure-Python lib
  and `delftblue` / `snellius` / CI are Linux; it only affects standalone
  `pixi` use inside the lib directory, but there is no reason to exclude them.
- The plan's smoke check is a shell one-liner; it landed as
  `tests/test_evaluation_library.py` instead, which additionally asserts
  invariant 5 (no jax / `pyurbanair` / backend / Hydra behind any module) and
  the matplotlib split, each in a subprocess so a shared `sys.modules` cannot
  mask a violation. The one-liner remains valid.

Carried into WP0.2 (decide there, recorded here so it is not lost):
`scripts/figspec/style.py:14` calls `matplotlib.use("Agg")` at module import.
Moved verbatim, that makes importing `evaluation.style` mutate global
matplotlib state as a side effect — a leaf library reaching out into the
process. Moving the backend choice to the scripts is the alternative, and is
*not* inert. Nothing in the suite currently catches either way.

**WP0.2**

- **CRPS merge, dtype.** `da_metrics.per_knot_crps` and
  `plotting._crps_ensemble` are the same formula but were *not* numerically
  equivalent as invoked: `_crps_ensemble` cast its inputs to float64 first,
  `per_knot_crps` did not, and the parameter artifacts on disk are float32
  (measured difference ~4.4e-07 on the pairwise term). They are merged into
  one `evaluation.scores.crps_ensemble` carrying `per_knot_crps`'s body — no
  internal cast, `n < 2` early return kept — and the one call site that
  relied on the implicit upcast (`compute_parameter_metrics`) now casts
  explicitly. Dtype is documented as the caller's policy at the function.
- **`matplotlib.use("Agg")`** (the WP0.1 carry-over above) moved **verbatim**
  into `evaluation/style.py`. Removing it is not inert, so it stays; it
  remains a known wart of a leaf library mutating global process state.
- **`stl_solid_mask`** takes the STL path and grid as mandatory arguments and
  loses its `@lru_cache` (numpy grids are unhashable). Its shape check now
  compares against the grid argument instead of `dataio.truth_grid()`, with
  the same `None`-returning behaviour. Inert: the only caller,
  `figspec.mask.truth_solid_mask`, keeps its own `@lru_cache(maxsize=1)`.
- **`pyurbanair` helpers inlined.** `figures.py` carries private copies of
  `run_utils.add_velocity_magnitude` and
  `state_utils.get_velocity_magnitude_field` (identical arithmetic, `_`
  prefix); the originals stay put for their non-evaluation callers.
  Duplicates drift silently, so `tests/test_evaluation_library.py` compares
  each copy's AST body against its original and fails if they diverge.
- **`figures.py` imports two underscore-private names from `scores.py`**
  (`_param_members_and_x`, `_plotted_param_names`). They were intra-module
  helpers in `plotting.py`; the split makes the leading underscore
  understate their scope. Accepted rather than renamed, because promoting
  them is API surface this WP has no mandate to design — phase 1 should
  decide when it touches the parameter bundle.
- **`figspec.metrics` was renamed at the call sites**, not aliased: the block
  drivers now `from evaluation import scores` and call `scores.field_rmse(…)`
  rather than keeping a `metrics` alias for a module that no longer exists.
  `figspec.style` keeps its `S` alias (`from evaluation import style as S`).
- **`_filtering_common.py`'s re-export block** lost the six moved names
  (`parameter_metric_summary`, `select_z_plane`, `sensor_magnitude`,
  `series_stats`, `streaming_state_rmse`, `vector_sensor_metrics`); its two
  consumers import them from `evaluation.*` directly (no-shims rule).
- **Black reformatted the moved code.** `scripts/figspec/style.py` and
  `metrics.py` were never black-clean, and pre-commit formats staged files.
  Every purely-moved function was verified byte-identical to `black`-formatted
  HEAD source, so the reformat is the only whitespace delta.
- **mypy still fails on the touched scripts.** `scripts/figure_creation/*`,
  `figspec/figcommon.py`, `figspec/dataio.py` and
  `tests/test_model_error_parameters.py` are unannotated and only get checked
  when staged. The failure is pre-existing and this branch strictly reduces
  it. Reproduce with pre-commit's own mypy hook over the files the move
  commit modifies:

  ```
  git diff --cached --name-only --diff-filter=M | grep '\.py$' > /tmp/f.txt
  pre-commit run mypy --files $(tr '\n' ' ' < /tmp/f.txt)
  ```

  On this branch: **98 errors in 10 files** (26 checked). Running the same
  file list against an `esmda-evaluation` worktree: **112 errors in 11
  files** (21 checked — the five `libs/evaluation` modules do not exist
  there, and contribute 0 here because of the waivers). A subset, not a
  different set: `figspec/style.py`'s 11 errors and three `no-any-return`s
  are gone because that code now sits behind a module waiver. Absolute
  counts differ under a bare `mypy --ignore-missing-imports` in the dev env
  (more type info available, so more errors) — the pre-commit hook is the
  gate, so it is the number quoted. Annotating these files is out of scope
  for a move; black and isort pass, and CI runs pytest, not mypy.
- **The library now does file I/O and prints to stdout**, which sits badly
  with invariant 5 ("arrays/datasets in → numbers/dicts/figures out") and
  with "scripts own I/O". `evaluation.style`'s `save_pdf` / `save_png`
  write a figure and `print()` the path; `write_table` writes a `.csv` *and*
  a `.tex`. `evaluation.figures` is the same story — every `plot_*` takes an
  `output_path` and saves through `_save` rather than returning a `Figure`.
  Both are plan-sanctioned (the table moves `figspec/style.py` and the
  `plot_*` set wholesale) and changing either is not inert, so WP0.2 keeps
  them. Recording it so the tension is a known debt rather than a
  rediscovery: `write_table` in particular is table *output*, not a figure
  convention, and belongs back in `scripts/figure_creation/`.
- **`scripts/figspec/mask.py` gains an import-time `matplotlib.use("Agg")`**
  it never had, via `from evaluation.style import stl_solid_mask`. This is
  the concrete blast radius of keeping the `Agg` call in the library (see
  the WP0.1 note above). Inert today — all six modules that import `mask`
  already set the backend themselves before importing pyplot — but it is a
  library reaching out and changing global process state, and the set of
  importers is no longer the set that opted in.
- Inertness verified by re-running `compute_esmda_metrics.py` +
  `make_esmda_figures.py` on a smoke run dir: `run_summary.yaml` and all six
  figure artifacts byte-identical to a baseline captured before the move.
  Stronger check in review: every moved top-level definition is AST-identical
  (docstrings stripped) to its pre-move original, with exactly six intended
  deltas — the `crps_ensemble` rename, the explicit cast in
  `compute_parameter_metrics`, the two inlined `pyurbanair.utils` helpers,
  `stl_solid_mask`'s signature, and the `xarray.` → `xr.` annotation rewrite
  in `scores.py` (8 annotation nodes; inert under lazy annotations, and
  `xr is xarray` regardless).
- **Acceptance grep: no live references remain**, but it is not literally
  empty. Three stale anchors outside `libs/evaluation` were updated rather
  than waived, because two are instructions aimed at the next work package:
  the `per_knot_crps` and `figspec/style.py` references in
  `esmda_turbulence_evaluation.md` §6/§7 (WP1.1's first task is that very
  function) and `_PLOTTED_PARAMS` in `srst_sgs_parameterization.md`. What
  still matches the grep is provenance — this plan's own move table and
  deviations, the "formerly …" note left at the §6 anchor, and the history
  paragraph in `tests/test_evaluation_scores.py`. Those are deliberate: the
  grep's purpose is to catch code still pointing at deleted modules, and no
  import, call or path reference does.
- **`scores.py` collapsed the duplicate xarray import.** The moved sources
  disagreed on the alias (`xr` vs `xarray`); keeping both was recorded as a
  deferred cleanup, but `xr is xarray` and the module carries
  `from __future__ import annotations`, so unifying on `xr` is provably
  inert and was done here instead.

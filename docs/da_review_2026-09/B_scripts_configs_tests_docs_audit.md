# Audit B — scripts / Hydra configs / tests / docs (`pyurbanair`)

Read-only audit, 2026-09-04, branch `isda_experiments` @ `7ab1c6d`.
Backends and DA-lib internals were audited by sibling agents; §5 summarises the
backend findings, `libs/data-assimilation` internals are out of scope here.

---

## 0. What the ISDA campaign actually exercises (the calibration for everything below)

The plan in `docs/plans/isda2026_talk_experiments.md` (written 2026-08-10) is
**not** what was run. The real campaign is recorded as one `.args` file per run
under `presentations/isda_new/experiments/{esmda,filtering,filter_smoothing}/_logs/`
(61 runs), plus the two forward-only scripts
`presentations/isda_crps/latex/scripts/run_{bias,barcelona}_cases.sh`.

Every distinct override across all 61 `.args` files:

| Axis | Values actually used | Never used |
|---|---|---|
| `case` | `xie_and_castro` (61/61); `barcelona` only in the two forward-run `.sh` | — |
| `model@truth_model` | `pyudales` (35), `pypalm` (26) | `pylbm`, `neural_surrogate` |
| `model@assim_model` | `pyudales` (61) | `pylbm`, `pypalm`, `neural_surrogate` |
| `params@truth_params` | `dynamic_sine` (61) | `static_truth`, `dynamic_truth`, `dynamic_cosine` |
| `params@prior_params` | `dynamic` (40), `static` (21) | — |
| `esmda/smoother` | `dynamic` (40) | `static`, `state`, `state_and_parameter`, `state_and_dynamic` |
| `esmda/localization` | `correlation` (28), `none` (12) | `distance` |
| `esmda/state_reduction` | (default `none`) | `svd` |
| `esmda.interval_seconds` | 7.5 / 15 / 30 | — |
| `esmda.num_steps` | 3 | — |
| `filtering.mode` | `state` (29), `joint` (12) | `parameter` |
| `filtering/analysis` | `stochastic` (41) | `etkf`, `etkf_tsvd`, `letkf`, `letkf_tsvd` |
| `filtering/inflation` | `rtps` (38), `none` (3) | `multiplicative`, `rtpp` |
| `filtering/evolution` | `none` (29), `random_walk` (12) | — |
| `filtering/localization` | `correlation` (23), `none` (18) | `distance` |
| `filtering/state_reduction` | `none` (41) | `svd_current`, `svd_streaming` |
| `filtering.assimilate_every_n_step` | 1 (41) | any n>1 |
| ensemble | `ensemble_size=50`, `num_parallel_processes=10` | — |

So: **the entire ETKF/LETKF/TSVD family, both filtering state-reduction options,
the ESMDA state-reduction option, `distance` localization, the four
state-bearing/static ESMDA smoothers, and `filtering.mode=parameter` were built
but never used by the campaign that motivated them.** `docs/temp/filtering_
ensemble_transform_benchmark.md` and `docs/temp/filtering_state_reduction_
benchmark.md` are still labelled "campaign template; not run" — honest, and still
true.

---

## 1. Entry points and pipelines

| Script | Lines | Purpose | Produces | Campaign use |
|---|---|---|---|---|
| `scripts/run_forward_model.py` | 232 | single/ensemble/rollout forward run; dynamic params write a truth artifact | `<out>/<model>[_ensemble][_rollout][_time_varying]/{state.nc,params.nc}` + viz | **YES** — the only entry point in `run_bias_cases.sh` / `run_barcelona_cases.sh` |
| `scripts/esmda/run_esmda.py` | 1073 | ESMDA smoother, 5 smoother variants × static/dynamic prior × W windows | `windows/window_{w}_*`, `posterior_params.nc`, `prior_params.nc`, `posterior_state_mean.nc`, `true_{state,params}.nc`, `truth_access.yaml`, `run_info.yaml` | **YES** (20 runs, `smoother=dynamic` only) |
| `scripts/esmda/compute_esmda_metrics.py` | 1616 | metric stage → `run_summary.yaml`, `eval_fields.nc` | as above | YES |
| `scripts/esmda/make_esmda_figures.py` | 569 | figure stage (P1/S1/S5/F1/D1/D3, animations) | PNG/MP4 in the run dir | YES |
| `scripts/esmda/_esmda_common.py` | 758 | shared truth-access / sensor-series / yaml helpers | — | YES |
| `scripts/esmda/run_probe_series.py` | 1282 | **pylbm-only** high-rate probe re-run of one finished window (WP3.2a) | `truth_probes.nc`, `windows/window_{w}_probes*.nc` | **NO** — refuses non-pylbm assim models; campaign is uDALES/PALM |
| `scripts/filtering/run_filtering.py` | 1051 | sequential EnKF, 3 modes × 5 analyses × inflation/evolution/localization/reduction | ESMDA-schema window files **and** filtering-native (`cycle_diagnostics.yaml`, `params_history.nc`, `state_history.nc`) | **YES** (21 runs) |
| `scripts/filtering/compute_filtering_metrics.py` | 542 | filtering metric stage | `run_summary.yaml` | YES |
| `scripts/filtering/make_filtering_figures.py` | 521 | filtering figure stage | PNGs | YES |
| `scripts/filtering/_filtering_common.py` | 711 | cycle↔frame bookkeeping; re-exports `_esmda_common` | — | YES |
| `scripts/filter_smoothing/run_filter_smoothing.py` | 1088 | hybrid: ESMDA parameter MDA (`final_forecast=False`) + EnKF state filter per window | both schemas + `window_{w}_filter_params.nc`, `window_{w}_esmda_pred_obs.nc` | **YES** (20 runs, `smoother=dynamic` × `mode=state`) |
| `scripts/compare_models.py` | 2538 | N-way backend comparison over shared scenarios; ~20 figure families | `results/compare_models/...` | **NO** — zero references outside its own docstring/doc; **zero tests** |
| `scripts/_common.py` | 370 | forward-run viz glue | — | partially (see §1.3) |
| `scripts/run_esmda_pipeline.sh` | 44 | run → metrics → figures | — | not directly (campaign drives the stages via its own launcher) |
| `scripts/run_filtering_pipeline.sh` | 123 | run → metrics → figures **+ `esmda_view/` symlink farm → ESMDA stages** | — | " |
| `scripts/run_filter_smoothing_pipeline.sh` | 138 | identical to the above + copies 3 PNGs back to the run root | — | " |
| `scripts/setup_dev_env.sh` | 52 | pixi bootstrap (`pixi run setup-dev`) | — | n/a |

### 1.1 Duplication between the three pipelines — measured

`run()` bodies, comment/blank-stripped, compared line-for-line (`difflib`):

| Pair | identical lines | ratio | longest verbatim runs |
|---|---|---|---|
| `run_esmda` vs `run_filtering` | 87 / 293 | 0.26 | 16, 11, 9 |
| `run_esmda` vs `run_filter_smoothing` | 98 / 293 | 0.25 | 32, 9 |
| **`run_filtering` vs `run_filter_smoothing`** | **252 / 366** | **0.59** | **37, 25, 16, 12, 12, 10** |

The 37-line verbatim run starts at `run_filtering.py:558` / `run_filter_smoothing.py:571`
(dynamic-truth detection → `params_to_estimate` filtering → out/windows dirs →
`config.yaml` save → inline-vs-disk truth branch → x-offset alignment). The
25-line run is the `run_info.yaml` `configuration:` block. This is the single
largest consolidation opportunity in `scripts/`: a shared
`_da_common.setup_truth_and_dirs(cfg)` would remove ~150 lines from
`run_filter_smoothing.py` alone and remove a real drift risk (the two already
differ on which keys land in `run_info.yaml`).

`scripts/run_filtering_pipeline.sh` and `scripts/run_filter_smoothing_pipeline.sh`
are ~95 % identical (only the entry-point script name, the config name, and a
3-file `cp` at the end differ). One parameterised script would do.

### 1.2 Duplication between the metric / figure stages

- `compute_filtering_metrics.py:100-106` **does** import `MeanFieldCollector`,
  `_flatten_parameter_members`, `_mean_field_block`, `_sensor_statistics`,
  `station_columns` from `compute_esmda_metrics.py`. Good reuse. `_ensemble_health`
  is genuinely reimplemented (different artifact layout: `windows/*_posterior_params.nc`
  vs `params_history.nc`) — justified.
- `make_filtering_figures.py:101-168` vs `make_esmda_figures.py:111-176`:
  `_note_skipped`, `_reference_velocity`, `_rank_counts` are **byte-identical
  code** with only docstring wording differing. ~66 lines of pure copy-paste.
  `make_esmda_figures.py` additionally has `_d3_step_label` (177-196) which is
  filtering-aware — so the copy is the wrong direction.
- Both figure stages import the same 9 `evaluation.figures` symbols and the same
  3 `evaluation.{scores,sensors,turbulence}` symbols; the `make_figures()` bodies
  differ mainly by which `_filtering_common` vs `_esmda_common` reader is called.
- No `compute_filter_smoothing_metrics.py` / `make_filter_smoothing_figures.py`
  exists — the hybrid reuses the filtering stages (good), but **nothing tests
  that path** (§6).

### 1.3 Dead code inside the shared script modules

| Location | What | Evidence |
|---|---|---|
| `scripts/_common.py:198-370` | `plot_time_varying_params`, `compute_time_varying_metrics`, `write_metrics_csv`, `print_metrics_summary`, `plot_time_varying_metrics` — **zero callers anywhere** (`grep` over `scripts/ tests/ src/ libs/`) | 173 of 370 lines dead |
| `scripts/esmda/_esmda_common.py:170-256` | `observation_noise_key`, `perturb_observations`, `_frame_noise` — **zero callers**; the only hits are their own cross-references. Observation noise is now drawn inline (`run_filtering.py:732`, `run_filter_smoothing.py:625`) and inside the DA lib | 87 lines dead + the inline draws are themselves duplicated |
| `scripts/run_forward_model.py:92` | `import pdb` inside `run()`, unused | leftover debugging |
| `src/pyurbanair/config/hydra_helpers.py:177-186` | `obs.mode == "grid"` branch — no case config uses it | dead branch |
| `src/pyurbanair/config/hydra_helpers.py:158,259,263` | `create_initial_state_ensemble`, `create_C_D`, `make_time_coords` — no callers repo-wide | 3 dead public helpers |

### 1.4 Stale docstrings in live scripts

- `scripts/run_forward_model.py:9-16, 22-24` documents `run.num_steps` and
  `run.time_varying` — **neither knob exists**; the config has `run.rollout_steps`
  and dynamic-ness is inferred from `"time" in params.coords` (line 106).
- `scripts/esmda/run_esmda.py:73-74` tells the user to produce a truth with
  `run_forward_model.py run.time_varying=true` — same non-existent knob. The
  correct incantation is `params=dynamic_sine`.

---

## 2. Hydra config tree — every group and option, with usage

Legend: **campaign** = appears in an ISDA `.args`/`.sh`; **tests** = selected by
name in `tests/`; **docs** = named in a `docs/` recipe.

### 2.1 `case/`
| Option | campaign | tests | Notes |
|---|---|---|---|
| `xie_and_castro` | ✅ (61/61 DA runs + bias forward runs) | ✅ | pinned by `conftest._ESMDA_OVERRIDES` |
| `barcelona` | ✅ forward runs only (`run_barcelona_cases.sh`) | ❌ never composed | has **no** `validation_*_points`; handled gracefully by `build_sensor_sets` (`_esmda_common.py:61-72`) |

### 2.2 `esmda/`
| Group/option | campaign | tests | Verdict |
|---|---|---|---|
| `smoother/dynamic` | ✅ 40 runs | e2e | live |
| `smoother/static` | ❌ | e2e + unit | live but campaign-unused |
| `smoother/state` | ❌ | e2e + unit | campaign-unused |
| `smoother/state_and_parameter` | ❌ | e2e + unit | campaign-unused |
| `smoother/state_and_dynamic` | ❌ | e2e only, **no unit test** | campaign-unused |
| `localization/correlation` | ✅ 28 | unit + 1 e2e | live |
| `localization/none` | ✅ 12 | ✅ | live |
| `localization/distance` | ❌ | unit + e2e | campaign-unused |
| `state_reduction/none` | ✅ (default) | ✅ | live |
| `state_reduction/svd` | ❌ | unit + e2e | campaign-unused; **`svd.yaml` and `filtering/state_reduction/svd_current.yaml` target the same class `OnlineStateReduction` with different key sets** (`basis_source`/`snapshot_stride` vs `whiten`/`variable_scales`) — the two groups have silently diverged |

### 2.3 `filtering/`
| Group/option | campaign | tests | Verdict |
|---|---|---|---|
| `analysis/stochastic` | ✅ 41 | unit + e2e | live |
| `analysis/etkf` | ❌ | unit + e2e | **built, tested, never used** |
| `analysis/etkf_tsvd` | ❌ | unit + e2e | " |
| `analysis/letkf` | ❌ | unit + e2e | " |
| `analysis/letkf_tsvd` | ❌ | unit + e2e | " |
| `inflation/rtps` | ✅ 38 | unit | live |
| `inflation/none` | ✅ 3 | config-only | live |
| `inflation/rtpp` | ❌ | class unit-tested; group **never selected** | dead option |
| `inflation/multiplicative` | ❌ | class unit-tested; group **never selected** | dead option |
| `evolution/none` | ✅ 29 | ✅ | live |
| `evolution/random_walk` | ✅ 12 | ✅ | live |
| `localization/none` | ✅ 18 | ✅ | live |
| `localization/correlation` | ✅ 23 | **group never selected in any test** | live-but-untested-as-config |
| `localization/distance` | ❌ | unit + e2e | campaign-unused |
| `state_reduction/none` | ✅ 41 | ✅ | live |
| `state_reduction/svd_current` | ❌ | unit + e2e | campaign-unused |
| `state_reduction/svd_streaming` | ❌ | unit + **config-compose only, no solver run** | campaign-unused, weakest coverage |

### 2.4 `params/`
| Option | campaign | tests | Notes |
|---|---|---|---|
| `dynamic` (AR(2)) | ✅ 40 (prior) | ✅ | live |
| `static` | ✅ 21 (prior) | ✅ | samples `sgs_constant` and `pressure_gradient_magnitude`, **neither of which is in any `params_to_estimate`**; `pressure_gradient_magnitude` is inert on every backend (§5) |
| `dynamic_sine` | ✅ 61 (truth) + bias runs | ❌ never composed in tests | live |
| `dynamic_cosine` | ❌ | ❌ | only reachable via `conf/compare_models.yaml` (itself unused) → **effectively dead** |
| `dynamic_truth` | ❌ | ✅ | test-only |
| `static_truth` | ❌ | ✅ | test-only |

### 2.5 `model/`
`pyudales` ✅ (truth+assim), `pypalm` ✅ (truth only, 26 runs), `pylbm` ❌ in the
campaign but the workhorse of the e2e test suite, `neural_surrogate` ❌ in both —
never mounted as `truth_model`/`assim_model` in any test or campaign run.

### 2.6 `neural_surrogate/`
All 7 primary configs have exactly one owning script (`compare_surrogate_models.py`
→ `comparison`, `finetune_neural_surrogate.py` → `finetuning`, `pretrain_autoencoder.py`,
`test_autoencoder.py` → `testing_autoencoder`, `generate_training_data.py` +
`generate_random_geometries_training_data.py` → `training_data`,
`test_neural_surrogate.py` → `testing`, `train_neural_surrogate.py` → `training`).
Groups `mode/{standard,domain_decomposition}` and `finetune_mode/{lora_nextstep,dft}`
are selected from those files' `defaults:` (`training.yaml:16`, `finetuning.yaml:8`).
17 `architectures/*` options exist; only `unet_convnext/{tiny,small,medium}` are
referenced by another config (`domain_decomposed/*`), the rest are CLI-only.
This subtree is orthogonal to the DA campaign.

### 2.7 Knobs read by no code

| Key | Defined at | Evidence |
|---|---|---|
| `run.ground_truth_dir` | `conf/run_forward_model.yaml:66` | zero readers; `tests/test_run_esmda.py:298` explicitly notes "the code reads `run.truth_dir` (not `run.ground_truth_dir`)" |
| `run.ensemble_save_on_disk` | `conf/run_forward_model.yaml:71` | only `run_esmda.py:720` / `run_filtering.py:674` read it; `run_forward_model.py` never does |
| `obs.mode: grid` support | `hydra_helpers.py:177-186` | no case config selects it |
| `esmda.state_reduction` (scalar) under a parameter-only smoother | `run_esmda.yaml:126` | `static.yaml`/`dynamic.yaml` don't interpolate it — documented as a no-op in `docs/data_assimilation.md:1041` |

---

## 3. Drift between the three inlined bases

`conf/run_esmda.yaml` / `run_filtering.yaml` / `run_filter_smoothing.yaml` were
built in sequence by copy-paste. Differences that are **not** obviously intentional:

| Key | run_esmda | run_filtering | run_filter_smoothing | Comment |
|---|---|---|---|---|
| `ensemble.ensemble_size` | 50 | 50 | **40** | filter_smoothing alone; a like-for-like hybrid-vs-filter comparison is broken by default |
| `esmda/localization` default | **correlation** | n/a | **none** | the hybrid's smoother half defaults unlocalized while the pure smoother defaults localized |
| `run.ensemble_save_on_disk` | **true** | false | false | |
| `run.save_prior_state` | false (`:160`) | absent (code default `True` at `run_esmda.py:728`) | absent, hardcoded `False` (`run_filter_smoothing.py` writes `save_prior_state: false` into `run_info.yaml`) | three different mechanisms for one concept |
| `run.save_history` | absent | true | true | |
| `run.save_forecast_history` | absent | **true** | absent | see §3.1 |
| `esmda.num_assimilation_windows` | 3 | n/a | **absent** — the hybrid uses `filter_smoothing.num_assimilation_windows` | three different window-count keys for the same unit |
| `esmda.seed` / `obs_error_std` | present | n/a | **absent** — hybrid uses `filter_smoothing.*` | the `esmda:` node in `run_filter_smoothing.yaml` is a *partial* copy; `docs/scripts_and_configs.md:83` claims "run_filter_smoothing reuses the node", which is only half true |
| `esmda.save_obs_diagnostics` | true | n/a | absent (hardcoded `True` at `run_filtering.py:967`) | |
| `filtering.num_assimilation_windows` | n/a | 3 | absent | |
| `filtering.mode` default | n/a | `joint` | `state` | |
| `time.seconds_per_knot` | 30.0 | 30.0 | 30.0 (fwd: 20.0) | |

### 3.1 A live inconsistency worth fixing first

`conf/run_filtering.yaml:189-192`:

```yaml
  # Off by default: the files are `assimilate_every_n_step` times the size of
  # the window's analyzed states, and a window's worth is held in memory while
  # it runs.
  save_forecast_history: true
```

The comment says off; the value is `true`, flipped by `7ab1c6d "config: ISDA 2026
experiment settings"`. `tests/test_run_filtering.py:554` asserts
`default_cfg.run.save_forecast_history is False`, and `_SMOKE_OVERRIDES`
(`tests/conftest.py:47-72`) does not touch the key — **so that test fails on the
current tree.** Same commit also baked campaign values into the shared defaults
(`esmda.num_steps 4→2`, `obs_error_std 0.25→0.1`, `interval_seconds 30→15`,
`case.xie_and_castro` `nz 24→16`, `simulation_time 60→120`, `spinup_time 30→5`,
`params/dynamic_sine` amplitudes/frequencies, `params/static.velocity_magnitude`
`N(5,1)→N(7.5,0.5)`). Per the repo's own convention (auto-memory: "conf/*.yaml
stays dirty — live run tuning, never commit"), these should have stayed
uncommitted or moved into a campaign overlay config.

---

## 4. Figure / analysis tooling

No script in this area imports the removed top-level `scripts.{run_esmda,
compute_esmda_metrics,make_esmda_figures,_esmda_common}` modules — the July move
into `scripts/esmda/` did not leave broken Python imports. The breakage is
elsewhere: **hard-coded HPC roots, pre-move `python scripts/<x>.py` paths in the
SLURM wrappers, and one script whose imports/config keys no longer exist.**

Also load-bearing: `run_esmda.py` no longer writes `run_summary.yaml` — that is
`compute_esmda_metrics.py`'s output (`scripts/esmda/run_esmda.py:532-534`). Any
wrapper that runs the runner and then reads `run_summary.yaml` is a no-op.

### 4.1 `scripts/figure_creation/` and `scripts/figspec/`

| Path | Status | Evidence |
|---|---|---|
| `figure_creation/plot_state_slices.py` | **LIVE (generic)** | no internal imports; SLURM wrappers use the new path |
| `figure_creation/visualize_ground_truth.py` | **LIVE (generic)** | `scripts._common.plot_derived_*` exist; usage strings pre-move (`:20-22`) |
| `figure_creation/visualize_run.py` | **LIVE (generic)** | reads `run_summary.yaml`/`*_params.nc`/`posterior_state_mean.nc`/`truth_access.yaml`, all still written; called correctly by `job_scripts/delftblue/visualize_run.slurm:50` |
| `figure_creation/compare_localization.sh` | **STALE-BROKEN** | runs only the runner (`:159`) then reads `.temp/loc_*/run_summary.yaml` (`:227`) — never invokes `compute_esmda_metrics.py`, so the table always prints `n/a`. Hydra tokens themselves are valid. |
| `figure_creation/compute_sweep_metrics.py` | **STALE-BROKEN as invoked** | imports all resolve, but `job_scripts/local/eval_sweep.sh:85`, `job_scripts/delftblue/eval_sweep.slurm:69`, `job_scripts/snellius/eval_sweep.slurm:66` all call `scripts/compute_sweep_metrics.py` (pre-move) → file not found |
| `figure_creation/compare_sweep_results.py` | **STALE-UNUSED** | reads `pyurbanair/sweep_metrics/`, which does not exist |
| `figure_creation/compare_state_runs.py` | **STALE-UNUSED** | `_DEFAULT_ROOT = /projects/prjs2075/urbanair/assim_with_state` (`:63`, used `:364`); metric key paths are still *correct*, just pointed at a dead campaign |
| `figure_creation/compare_param_vs_state.py` | **STALE-UNUSED** | same retired Snellius campaign |
| `figure_creation/visualize_state_run.py` | **STALE-UNUSED** | scoped to the `_ic`/`_all` suffix convention of the retired `assim_with_state` campaign |
| `figure_creation/make_all_figures.py`, `make_animations.py`, `make_figures_block_{a,b,c}.py`, `make_figures_summary.py`, `make_notes.py` | **STALE-UNUSED (one dead pipeline)** | all data comes through `figspec.dataio.DATA_ROOT = /projects/prjs2075/urbanair` (`scripts/figspec/dataio.py:31-34`), which does not exist; spec doc moved to `docs/archive/figure_specs.md` |
| `figspec/{__init__,_selftest,dataio,figcommon,mask}.py` | **STALE-UNUSED** | `dataio._RUN_RE` (`:47`) matches the old `job_scripts/local/*/rollout_esmda_from_truth.sh:57` run-tag scheme, **not** the ISDA dir names (`pypalm_to_pyudales_w3_loccorrelation_obs15_inflow`) — it cannot discover the current campaign at all |

### 4.2 `scripts/adjust_simulations/` and `scripts/tools/`

| Path | Status | Evidence |
|---|---|---|
| `adjust_simulations/regenerate_ground_truth_params.py` | **STALE-BROKEN (4 independent breakages)** | `:17` imports `create_time_varying_true_params` — **does not exist anywhere**; `:19` `REPO_ROOT = parent.parent` → `scripts/`, so `initialize_config_dir(scripts/conf)` fails (`:39`); `:31` reads `time_varying.num_time_points` (no such group); `:45-46` reads `cfg.params.true` / `cfg.params.external` (no such keys); `:33` reads `esmda.seed` from `run_forward_model.yaml` |
| `adjust_simulations/convert_ground_truth_to_32bit.py` | **STALE-BROKEN** | `:18-19` `Path(__file__).parent.parent / "ground_truth"` resolves to `scripts/ground_truth` after the directory move; also expects a `64_bit/`+`32_bit/` split that no longer exists |
| `adjust_simulations/make_state_small.py` | **STALE-BROKEN as invoked** | hardcoded `/projects/prjs2075/...` (`:13`); `job_scripts/{snellius,delftblue}/make_state_small.slurm:34` call the pre-move path |
| `adjust_simulations/trim_spinup.py` | **LIVE (generic)** but callers broken | pure netCDF4; output contract still valid for `run.truth_dir`. `job_scripts/{delftblue,snellius}/trim_and_visualize.slurm:41` call the pre-move path |
| `tools/prepare_case_stl.py` | **LIVE** | standalone trimesh utility; produces the Barcelona STLs |
| `tools/preprocess_udales_geometry.py` | **LIVE** | `save_precomputed_geometry` exists (`libs/pyudales/.../forward_model.py:146`); wired to `conf/case/barcelona.yaml`. The only in-scope file with uncommitted changes (alongside the modified `examples/udales/barcelona/*`). Its docstring says `conf/case/<case>/geometry.yaml` (`:14`) — cases are single files now. |

### 4.3 `presentations/` and `experiments_report/`

Both trees are **untracked** (not in git).

| Path | Status | Evidence |
|---|---|---|
| `isda_crps/latex/scripts/run_bias_cases.sh` | **LIVE — campaign driver** | `params.profiles.*.offset` exist (`conf/params/dynamic_sine.yaml:11,18`); all 6 output dirs present |
| `isda_crps/latex/scripts/run_barcelona_cases.sh` | **LIVE — in progress** | only 2 of 6 result dirs present; matching `.temp_bcn_*` scratch dirs untracked in the working tree |
| `isda_crps/latex/scripts/make_{bias,barcelona,cases,truths}_animation.py`, `make_figures.py`, `make_geom_figures.py` | **LIVE** | inputs all present; `make_figures.py` covers 18 run IDs |
| `isda_final/latex/scripts/*` | **LIVE** | trimmed to the 12 final-deck runs |
| `isda_final_crps/latex/scripts/make_figures.py` | **ORPHAN — actively dangerous** | **byte-identical (md5 `8c08464e…`) to `isda_final`'s copy, including `OUT = .../isda_final/latex/figures` (`:38`)** — running it silently overwrites the *other* deck's figures |
| `isda_final_crps/latex/scripts/make_cases_animation.py` | **ORPHAN (older fork)** | 349 lines vs `isda_final`'s 382; missing the `ASSIM_XY`/`VALID_XY` sensor network added at `isda_final:64-67` |
| `presentations/isda/scripts/*` | **STALE-UNUSED** | superseded by `isda_crps`; `diff -r` against `presentations/isda_new/latex/scripts/` is **empty** — but only the `isda/scripts` copy resolves `parents[3]` correctly; the `isda_new/latex/scripts` copy is off by one |
| `experiments_report/scripts/figlib.py` | **LIVE (hub)** | `EXP = presentations/isda_new/experiments` (`:68`); 92 `run_summary.yaml` + 911 PNGs there; imported by 8 scripts including both live decks |
| `experiments_report/scripts/extract_{esmda,filtering,filter_smoothing,comparison}.py` | **LIVE** | every artifact they read is still written today |
| `experiments_report/scripts/make_{esmda,filtering,filter_smoothing,comparison,conclusion}_figures.py` | **LIVE** | — |
| `experiments_report/scripts/figlib_demo.py` | **LIVE (dev-only)** | writes `figures/_demo/`, not referenced by any `.tex` |

### 4.4 Duplication in the figure tooling

- `make_figures.py` × 3 decks: `isda_final` and `isda_final_crps` are **byte-identical**; `isda_crps`'s differs by ~15 lines (output dir, 18 vs 12 run IDs, sensor indices).
- `make_cases_animation.py` × 3: `isda_crps` vs `isda_final` differ in **2 lines**; `isda_final_crps`'s is a stale earlier revision.
- `numbers.json` × 3: identical (md5 `61f5a25f…`).
- `scripts/figspec/figcommon.py` reimplements `libs/evaluation/figures.py`
  (`figcommon.plot_sensor_timeseries:256` ↔ `evaluation.figures.plot_sensor_timeseries:425`;
  `plot_param_trajectories:125` ↔ `plot_parameter_error:363`;
  `plot_field_error_grid:301` ↔ `plot_mean_slices:1448`) — duplication with **zero
  live consumers**.
- `experiments_report/scripts/figlib.py` is *not* a duplicate: it is post-hoc PIL
  cropping of already-rendered PNGs, a different layer from the matplotlib renderers.

---

## 5. Backends from the DA point of view (summary of the sibling audit)

**Schema mechanism.** `resolve_parameter_schema` (`src/pyurbanair/config/hydra_helpers.py:103-119`)
is a hard-coded tuple keyed on the config `name` string, **not** a backend
declaration, and **no assimilation entry point imports it** — its only consumers
are the three neural-surrogate training-data generators. The real `params_to_estimate`
mechanism is `filter_parameter_config` (`hydra_helpers.py:129-155`), which deletes
non-selected keys from `_PARAM_CONFIG_BLOCKS = ("parameters", "external_parameters",
"static_parameters")`. Consequences: a name not present in the sampler config is a
**silent no-op**, and `HarmonicParameterModel`'s `profiles:` block is not in
`_PARAM_CONFIG_BLOCKS`, so `params_to_estimate` cannot filter the `dynamic_sine`
truth at all.

**What the campaign estimates:** `[inflow_angle, velocity_magnitude]` in all three
run configs (`run_esmda.yaml:89`, `run_filtering.yaml:97`, `run_filter_smoothing.yaml:124`);
no job script overrides it.

**Parameter × backend reachability:**

| Parameter | pyudales | pypalm | pylbm | neural_surrogate |
|---|---|---|---|---|
| `inflow_angle` | ✅ | ✅ | ✅ | ✅ (if trained) |
| `velocity_magnitude` | ✅ | ✅ | ✅ (also sets the lattice `C_u`) | ✅ |
| `vertical_inflow_exponent` | ✅ | ✅ | ✅ | only if trained |
| `sgs_constant` | ✅ → `&NAMSUBGRID c_vreman` (`forward_model.py:850,860`) — **the "cs is inert" memory is FIXED in this tree** | ✅ → `km_constant`, but m²/s and forces `constant_flux_layer=.false.`; `conf/model/pypalm.yaml:34` deliberately `null` | ✅ → `ivreman` | only if trained |
| `pressure_gradient_magnitude` | whitelisted but **INERT** — `_apply_inflow_settings` hard-assigns `use_nudging = True` (`forward_model.py:732`), so the only branch that would write `dpdx/dpdy` is dead; the live paths write zeros (`nudging_utils.py:372-390`, `inlet_turbulence_utils.py:974-1000`) | inert | inert | filler only |

So `conf/params/static.yaml:42-44` samples a parameter with **zero observation
sensitivity on every backend**, and `resolve_parameter_schema` advertises it as
uDALES-specific — both are wrong today.

**Could-be-but-isn't DA parameters:**
- Already fully wired, only absent from `params_to_estimate`: uDALES `sgs_constant`
  (in the whitelist, params wins over config), pylbm `sgs_constant`, PALM
  `km_constant`.
- One small change away: uDALES synthetic-eddy `intensity` / `length_scale_{x,y,z}`
  — `apply_inlet_turbulence` is already called per `run_single` with the params
  Dataset in hand (`forward_model.py:747-750`); needs a `_resolve_inlet_turbulence(params)`
  mirroring `_resolve_nudging_config` plus one whitelist entry. Campaign values are
  static (`intensity=0.05`, `length_scale_*=6.0`), pinned identically in 23 `.args`.
- Needs real new code: nudging `tnudge`/`nnudge_meters` (uDALES and PALM), surface
  roughness `z0` (hard-coded per facet class in `python_udgeom/preprocessing.py:303-519`;
  also `&BC z0` is dead under IBM), PALM `disturbance_amplitude`, pylbm inlet-turbulence
  amplitude (written at compile time, not per member).

**Legacy code confirmed:**
1. `src/pyurbanair/base_rollout_forward_model.py` — **zero importers** repo-wide;
   documented as legacy at `docs/codebase_guide.md:225-229`, `README.md:458`.
   Note `tests/conftest.py:44-45` still cites it in a comment.
2. `libs/pyudales/src/pyudales/utils/rollout_utils.py::collect_rollout_results` —
   only the definition exists.
3. Orphan `.pyc` for deleted sources: `libs/pylbm/.../rollout_forward_model.pyc`,
   `libs/pylbm/utils/.../rollout_utils.pyc`, `libs/pyudales/.../rollout_forward_model.pyc`.
4. `hydra_helpers.{create_C_D, make_time_coords, create_initial_state_ensemble}` —
   no callers.
5. uDALES static/periodic inflow branch `forward_model.py:776-782` +
   `params_utils.apply_inflow_settings` — unreachable (see `pressure_gradient_magnitude`).
6. 50 commented-out lines `nudging_utils.py:391-440`.
7. `training_data/samplers.py::UniformParameterSampler` — no `_target_` in any config.

`conf/model/neural_surrogate.yaml:71-76`'s `nudging_config: {tnudge, nnudge, ...}`
is **not** stale: `nnudge` is a real uDALES key (`forward_model.py:216-219`,
`nudging_utils.py:233`), just the levels-form of `nnudge_meters`. (It *would* be
silently dropped on pypalm, which reads only `tnudge`/`nnudge_meters`.) That whole
block is inert anyway because `spinup_source: training_data` short-circuits
`prepare_neural_surrogate` (`hydra_helpers.py:69-70`).

---

## 6. Tests

`pyproject.toml:184-185` sets only `addopts = "-v"` — **no markers, no
`python_files` override, no way to deselect the solver-running tests**. There is
no skip guard on any DA test: `test_run_esmda`, `test_run_filtering`,
`test_run_filter_smoothing` build and run pylbm/uDALES unconditionally.

Smoke shape (`tests/conftest.py:47-72`): 20×20×4 domain, `simulation_time=3.0`,
`output_frequency=1.0` (→4 frames/window), `ensemble_size=2`,
`seconds_per_knot=1.5`, one pinned validation sensor. Windows are never forced —
each test sets 1 or 2.

### 6.1 Coverage headlines
- Every ESMDA smoother, every filtering analysis (incl. all four ETKF/LETKF
  variants), `svd_current`, `distance` localization, the stride, and all four live
  hybrid combinations **do** have smoke-e2e runs. The DA math is well unit-tested
  (`test_filtering_etkf.py`, `test_filtering_letkf.py`, `test_state_reduction.py`,
  `test_esmda_smoother.py`, `test_filter_smoothing.py`).
- But every DA e2e runs at `N_e=2` (once 4), `esmda.num_steps=1`, `interval_seconds
  = simulation_time` (⇒ exactly one aggregation bin, `mode=mean` only), and ≤2
  windows. The shipped defaults (`N_e=50`, `num_steps=2/3`, `interval_seconds=15`,
  3 windows) are **never** exercised.

### 6.2 Zero coverage
- `run.truth_start_time` — defined in all three entry points, zero references in `tests/`.
- `filtering/localization=correlation` as a config group (the campaign's most-used
  filtering localization, 23 runs).
- `filtering/inflation={rtpp,multiplicative}` as config groups.
- `filtering/state_reduction=svd_streaming` end to end.
- filter_smoothing `static` × `state`.
- **The whole stage-2/3 half of `run_filter_smoothing_pipeline.sh`** — no test
  runs `compute_filtering_metrics` / `make_filtering_figures` / the `esmda_view/`
  symlink farm on a hybrid run dir.
- All three `run_*_pipeline.sh` scripts.
- `neural_surrogate` and `pypalm` as DA models (pypalm composes only).
- `case=barcelona`.
- `params@truth_params=dynamic_sine` — the shipped default of `run_esmda.yaml:50`
  and `run_filtering.yaml:70`, used by 61/61 campaign runs, never composed in a test.
- `ensemble.failure` policy inside a real DA run (14 unit tests on the internals only).
- `scripts/compare_models.py` (2538 lines) — no test mentions it.

### 6.3 Dead / broken tests
1. **`tests/_nudging_utils.py` — 805 lines, 24 test methods in 8 `Test*` classes,
   NEVER COLLECTED.** The name doesn't match pytest's `test_*.py` pattern, there
   is no `python_files` override, and no module imports it
   (`grep -r _nudging_utils` → 0 non-`__pycache__` hits). Added with that name in
   `5f4c2c3` (2026-04-06) and dark ever since. Directly covers the uDALES nudging
   path the whole campaign relies on.
2. `tests/test_run_filtering.py:554` — currently **failing** against
   `conf/run_filtering.yaml:192` (§3.1).
3. Four near-identical e2e override blocks: `test_run_esmda.py:22-77`,
   `test_esmda_obs_diagnostics.py:854-872`, `test_run_probe_series.py:25-42`,
   `test_localization.py:415-441` — already drifted.
4. `test_run_esmda.py:370-372` duplicates the assertion at `:367-369` verbatim.
5. `test_run_filtering.py:629` asserts an identity between two `xarray.isel`
   expressions written in the test itself — exercises no product code.
6. `test_hydra_config.py` bypasses `compose_test_cfg` (`:14-16`) and asserts
   production values (`:64`, `ensemble_size == 64`) — it breaks every time the
   entry points are retuned for a run, which is exactly what `7ab1c6d` did.
7. `test_prepare_case_stl.py:511` always skips (env-var gate).

---

## 7. Docs — current vs stale

| Doc | Verdict |
|---|---|
| `docs/data_assimilation.md` | **Current and good.** §8 (filter cycle semantics), §9 (the hybrid), §10 (config groups) match the code. Only §10 omits the `filtering/*` group tables (it defers to `scripts_and_configs.md`) and the `esmda/state_reduction` no-op note at `:1041` is accurate. |
| `docs/ensemble_transform_filters.md` | **Current and unusually honest** — the four variants, the localization pre/proscriptions, and the "not benchmarked" status all check out against `conf/filtering/analysis/*` and the code. |
| `docs/scripts_and_configs.md` | **Structurally current, numerically stale.** All five entry points and the hybrid are documented (§1.1, §1.8, §2.1, §2.3-2.4). But nearly every default in the tables predates `7ab1c6d`: `paths.results_dir` esmda documented as `/export/scratch2/ntm/...` (actual `.temp/...`); `seconds_per_knot` 30/60 (actual 20/30); `esmda.num_steps` 3 (actual 2); `obs_error_std` 0.25 everywhere (actual 0.1); `esmda.interval_seconds` 30 (actual 15); `filtering.num_assimilation_windows` 2 (actual 3); `run.save_forecast_history` "false (filtering only)" (actual `true`); `filter_smoothing` `num_assimilation_windows` 2 / `obs_error_std` 0.25 (actual 3 / 0.1); no mention of `ensemble_size=40`. Case section: xie_and_castro documented `nx=30, simulation_time=300, spinup_time=50, 2 validation sensors` — actual `nx=40, 120, 5, 4 sensors`; barcelona documented `simulation_time=1200` — actual `300`. Model notes: pylbm `cuda: true` (actual `auto`), pyudales `ncpu: 25` (actual 1), pypalm `ncpu: 14` (actual 1). |
| `docs/codebase_guide.md` | **Stale on the biggest structural fact.** §6 is still titled "**The two** assimilation entry points" (`:485`) and lists only ESMDA + filtering; `filter_smoothing` appears **nowhere in the file** (grep → 0 hits), even though it is a merged third entry point with its own config, script and pipeline. Its last commit is `refactor(da): erase filter smoothing` (2026-08-13) — it was updated to remove the *old* design and never updated when the *new* `FilterSmoothing` landed (PR #125). The `BaseRolloutForwardModel` legacy note (`:225-229`) is correct. |
| `conf/README.md` | **Stale** (last touched 2026-07-08). Claims "exactly **three** run entry points" (`:25`); omits `run_filter_smoothing.yaml`, `compare_models.yaml`, `run_probe_series.yaml`. The `filtering/` group list (`:50-52`) omits `state_reduction/`. The worked example at `:73` uses **`filtering.num_cycles=4`, a key that does not exist** — it would die under Hydra struct mode. |
| `README.md` | **Stale.** `:295` uses the same non-existent `filtering.num_cycles=4`. The repo tree (`:534-559`) omits `conf/run_filter_smoothing.yaml`, `conf/compare_models.yaml`, `conf/run_probe_series.yaml`, `conf/filtering/state_reduction/`, `scripts/filter_smoothing/`, `scripts/run_filter_smoothing_pipeline.sh`, `scripts/compare_models.py`, `scripts/esmda/run_probe_series.py`. |
| `docs/plans/isda2026_talk_experiments.md` | **Superseded, not implemented as written.** Prerequisite §2.3 asks for three frozen case configs (`case=xie_and_castro_turbulent`, `case=xie_and_castro_periodic`) that were never created — the campaign instead overrides `*_model.forward_model.boundary_condition` / `.inlet_turbulence.enabled` per run. E1/E7/E8/E9 (ETKF, LETKF, reduction ladder) were **never run** (§0). The experiment IDs it defines (E1–E11) don't match the E*/F*/H* IDs in `experiments_report/`. It also uses `filtering.num_cycles`. Keep as history; do not use as a runbook. |
| `docs/plans/udales_inlet_turbulence.md` | Marked IMPLEMENTED 2026-07-29; matches the code (`inlet_turbulence_utils.py`). Correctly points at `docs/pyudales.md §6.1` as the maintained ref. |
| `docs/plans/filtering_state_reduction_and_transforms.md` | Implemented (PR 1 + PR 2 shipped: `svd_current`/`svd_streaming`, ETKF/LETKF/TSVD all exist and are tested) but **its §6 resource gate was never passed** — the two benchmark records in `docs/temp/` are still "not run". |
| `docs/plans/filter_smoothing_windowed_esmda.md` | Implemented (PR #125) — the script and `data_assimilation.FilterSmoothing` match. |
| `docs/plans/esmda_evaluation/*` (5 files) | Implemented and merged (PR #116/#117); pure history now. |
| `docs/plans/srst_sgs_parameterization.md` | **Abandoned + one stale claim.** Nothing in the tree implements a two-channel closure parameterization. Its "incidental finding" that uDALES `cs` is inert under `lvreman` is **already fixed** (`forward_model.py:850` selects `c_vreman`). |
| `docs/plans/palm_nudging_driver_plan.md` | Implemented (`pypalm/utils/nudging_utils.py`, `conf/model/pypalm.yaml:66-72`). |
| `docs/plans/esmda_turbulence_evaluation.md` | Research doc; its metric set is what `compute_esmda_metrics.py` implements. History. |
| `docs/temp/da_filtering_module_plan.md`, `da_review_state_estimation.md` | Implemented / superseded history. |
| `docs/temp/filtering_{ensemble_transform,state_reduction}_benchmark.md` | Templates, unpopulated, correctly labelled. |
| `docs/temp/rank_histogram_math.md` | Uncommitted working note; consistent with `evaluation/scores.py`. |
| `docs/archive/*` (21 files) | Archive; not checked. |

---

## 8. Ranked candidates for deletion / consolidation

**Correctness first (do these regardless of any cleanup):**

1. **`conf/run_filtering.yaml:192` `save_forecast_history: true` vs its own comment
   and `tests/test_run_filtering.py:554`.** The suite is red on the current tree.
   Decide which is right; if the campaign needs `true`, move it to an overlay, not
   the shared default.
2. **Back out `7ab1c6d`'s campaign tuning from the shared entry-point configs**
   (or move it to a `conf/campaign/isda2026.yaml` overlay). It silently changed
   `case/xie_and_castro`, `params/static`, `params/dynamic_sine` and all three run
   configs for every future user, and it is what desynchronised
   `docs/scripts_and_configs.md`.
3. **`conf/params/static.yaml:42-44` `pressure_gradient_magnitude`** — remove or
   mark inert; it has zero observation sensitivity on every backend and
   `resolve_parameter_schema:117` advertises it as a uDALES parameter.
4. **`tests/_nudging_utils.py` → `tests/test_nudging_utils.py`** — 805 lines / 24
   tests of the uDALES nudging path have never run. Rename, then fix whatever
   breaks.
5. **`presentations/isda_final_crps/latex/scripts/make_figures.py` writes into
   `isda_final/latex/figures`** (`:38`, byte-identical fork). Running it silently
   overwrites the live deck's figures. Delete the fork or fix the path.
6. **Six `job_scripts/*` entries invoke pre-move `scripts/*.py` paths** and fail
   immediately: `job_scripts/local/eval_sweep.sh:85`,
   `job_scripts/{delftblue,snellius}/eval_sweep.slurm:{69,66}`,
   `job_scripts/{delftblue,snellius}/trim_and_visualize.slurm:41`,
   `job_scripts/{delftblue,snellius}/make_state_small.slurm:34`.

**Deletion candidates (ranked by lines removed × confidence):**

| # | Target | Lines | Rationale |
|---|---|---|---|
| 1 | `scripts/compare_models.py` + `conf/compare_models.yaml` | ~2600 | Zero callers, zero tests, not used by the campaign; the only reachable use of `params/dynamic_cosine`. If kept, it needs at least one compose test. |
| 2 | `scripts/esmda/run_probe_series.py` + `conf/run_probe_series.yaml` + the `probe_*` helpers in `_esmda_common.py` | ~1400 | pylbm-only by construction; the campaign is uDALES/PALM, so it can never run on a campaign artifact. Well tested — keep only if pylbm work resumes. |
| 3 | `scripts/_common.py:198-370` | 173 | Five functions, zero callers. |
| 4 | `scripts/esmda/_esmda_common.py:170-256` | 87 | `observation_noise_key` / `perturb_observations` / `_frame_noise`, zero callers; superseded by inline draws + the DA lib. |
| 5 | `src/pyurbanair/base_rollout_forward_model.py` + `libs/pyudales/.../utils/rollout_utils.py` + the three orphan `rollout_*.pyc` | ~120 | Zero importers; already documented as legacy in two places. |
| 6 | `hydra_helpers.{create_C_D, make_time_coords, create_initial_state_ensemble}` + the `obs.mode: grid` branch | ~35 | No callers / no config selects it. |
| 7 | `conf/filtering/inflation/{rtpp,multiplicative}.yaml` | 6 | Groups never selected by any script, test or campaign run (the classes stay unit-tested). Cheap to keep — low priority. |
| 8 | `conf/params/dynamic_cosine.yaml` | 22 | Only reachable through `compare_models.yaml` (#1). |
| 9 | `nudging_utils.py:391-440` (commented-out block) + the unreachable `apply_inflow_settings` branch | ~120 | Dead source retained as comments; the branch is provably unreachable (`forward_model.py:732`). |
| 10 | `scripts/run_forward_model.py:92` `import pdb` | 1 | Leftover. |

**Figure/analysis tooling deletion candidates (ranked separately):**

| # | Target | Lines | Rationale |
|---|---|---|---|
| 1 | `scripts/adjust_simulations/regenerate_ground_truth_params.py` | 68 | Four independent breakages (missing `create_time_varying_true_params`, wrong `conf/` dir, dead `time_varying`/`params.true` keys). Unfixable without a rewrite; no caller. |
| 2 | `presentations/isda_final_crps/latex/scripts/` (3 files) | ~600 | Byte-identical fork that writes into `isda_final`'s figure dir; its animation script is an older revision. |
| 3 | `scripts/figspec/` (5 files) + `scripts/figure_creation/make_{all_figures,animations,figures_block_a,figures_block_b,figures_block_c,figures_summary,notes}.py` | ~3500 | One coherent pipeline bolted to `/projects/prjs2075/urbanair` and a run-naming scheme the ISDA campaign does not use; its spec already lives in `docs/archive/figure_specs.md`. `figcommon.py` also duplicates `evaluation/figures.py` with zero live consumers. |
| 4 | `scripts/adjust_simulations/convert_ground_truth_to_32bit.py` | 88 | Path broken by the directory move; targets a `64_bit/32_bit` layout that no longer exists. |
| 5 | `scripts/figure_creation/{compute_sweep_metrics,compare_sweep_results}.py` | 1502 | Sweep store (`pyurbanair/sweep_metrics/`) does not exist; all three SLURM callers use the pre-move path. |
| 6 | `scripts/figure_creation/compare_localization.sh` | 269 | Its reporting stage can never populate the table (missing metrics stage). Fix or delete. |
| 7 | `scripts/figure_creation/{compare_state_runs,compare_param_vs_state,visualize_state_run}.py` | 1201 | Schema-correct but scoped to the retired `assim_with_state` `_ic`/`_all` campaign. |
| 8 | `presentations/isda/scripts/` **or** `presentations/isda_new/latex/scripts/` | ~700 | `diff -r` empty; keep the `isda/scripts` copy (correct `parents[3]`), drop the other. |
| 9 | `scripts/adjust_simulations/make_state_small.py` | 95 | Hardcoded Snellius path; caller uses the pre-move path. |

**Explicitly keep:** all of `experiments_report/scripts/*`,
`presentations/isda_crps/latex/scripts/*`, `presentations/isda_final/latex/scripts/*`,
`scripts/tools/*`, and the generic single-run viewers `visualize_run.py`,
`visualize_ground_truth.py`, `plot_state_slices.py`, `trim_spinup.py`.

**Consolidation candidates (ranked by risk reduced):**

| # | Target | Rationale |
|---|---|---|
| 1 | Extract a `scripts/_da_common.py` for the truth/dirs/`run_info.yaml` block shared by `run_filtering.py` and `run_filter_smoothing.py` | 252 identical lines, 59 % of the smaller body, incl. a 37-line and a 25-line verbatim run. They have **already** drifted on `run_info.yaml` keys and on the inline noise draw. |
| 2 | Merge `run_filtering_pipeline.sh` and `run_filter_smoothing_pipeline.sh` into one parameterised script | ~95 % identical; the `esmda_view/` symlink farm is duplicated verbatim and is untested in both. |
| 3 | Move `_note_skipped` / `_reference_velocity` / `_rank_counts` into `_esmda_common.py` (or a `scripts/_figcommon.py`) | 66 byte-identical lines across `make_esmda_figures.py` and `make_filtering_figures.py`. |
| 4 | Unify the window-count / seed / obs-error keys across the three entry points (`esmda.num_assimilation_windows` / `filtering.num_assimilation_windows` / `filter_smoothing.num_assimilation_windows`) | Three names for one unit; the hybrid's `esmda:` node is a partial copy, which `docs/scripts_and_configs.md:83` misdescribes. |
| 5 | Reconcile `conf/esmda/state_reduction/svd.yaml` with `conf/filtering/state_reduction/svd_current.yaml` | Same `_target_` (`OnlineStateReduction`), disjoint key sets (`basis_source`/`snapshot_stride` vs `whiten`/`variable_scales`) — one of the two is passing keys the class no longer takes, or the class has two personalities. |
| 6 | Collapse the four e2e-override blocks in `tests/` into one shared fixture | Already drifted (`ensemble_size`, `num_parallel_processes`, backend). |
| 7 | Add a `slow`/`e2e` pytest marker | `pyproject.toml:184-185` has none, so there is no way to run the ~40 unit-only DA tests without building Fortran. |

**Doc fixes (cheap, high value):**

| # | Target |
|---|---|
| 1 | `docs/codebase_guide.md:485-508` — "the two assimilation entry points" → three; add `run_filter_smoothing.py` / `run_filter_smoothing_pipeline.sh`. |
| 2 | `conf/README.md:25` "exactly three run entry points" → five; add `filtering/state_reduction/` at `:50-52`; fix the `filtering.num_cycles=4` example at `:73`. |
| 3 | `README.md:295` — same `filtering.num_cycles` fix; add the missing tree entries at `:534-559`. |
| 4 | `docs/scripts_and_configs.md` §1.1-§1.3 — refresh every default from the current YAML (see §7). |
| 5 | `scripts/run_forward_model.py:9-24` and `scripts/esmda/run_esmda.py:73-74` — drop `run.num_steps` / `run.time_varying`. |
| 6 | `docs/plans/srst_sgs_parameterization.md` — strike the "cs is inert" finding (fixed in `6a7ca2f`). |
| 7 | `docs/plans/isda2026_talk_experiments.md` — add a "superseded by `presentations/isda_new/experiments/`" banner. |

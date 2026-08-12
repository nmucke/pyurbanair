# Filtering ensemble-transform benchmark record

> **Status: campaign template; not run.** No scientific, accuracy, memory, or
> performance benefit is claimed here, and the LETKF resource gate of
> `docs/plans/filtering_state_reduction_and_transforms.md` §6 is **not** passed
> until every `TBD` below is replaced by a measurement from an executed run.
> Correctness tests in CI establish the transform identities only; they say
> nothing about cost, and smoke-shaped runs are never held-out acceptance.

## Purpose and acceptance order

This campaign covers PR 2 of
[`docs/plans/filtering_state_reduction_and_transforms.md`](../plans/filtering_state_reduction_and_transforms.md):
the global ETKF (step 4), the reusable observation-space TSVD (step 5), and the
LETKF (step 6). It answers two separate questions that must not be merged into
one ranking.

1. **Deterministic vs stochastic, globally.** Global stochastic EnKF against
   global ETKF, and against global ETKF + observation TSVD. All three run
   unlocalized (`filtering/localization=none`); `ETKFAnalysis` declares
   `localization_policy="forbidden"`, so a localized ETKF is a construction-time
   error, not a run.
2. **The LETKF resource gate.** The localized stochastic EnKF (the incumbent
   baseline) against LETKF retaining every local thin-SVD direction and against LETKF
   with configured TSVD, on at least the canonical Xie-and-Castro grid. This is
   the gate the plan mandates: it is a *resource* measurement first and a skill
   measurement second.

Acceptance order:

1. The CI correctness gates pass first — all-ones localization equals global
   ETKF, TSVD-disabled equals the full-rank local transform, chunked equals
   unchunked, the exact linear-KF mean/covariance comparison. "TSVD-disabled"
   there means what the `etkf`/`letkf` groups ship — the whole `tsvd` node
   `null`, so `enabled=false` *and* no `numerical_tolerance`. A
   `numerical_tolerance` is not gated by `enabled` (it redefines round-off
   rather than making a scientific choice) and would cut the rank on its own,
   so a run that sets one is not the untruncated reference. Do not interpret
   any number here before those pass; a fast wrong transform is not a result.
2. Then the resource table. **The LETKF passes the gate only if peak memory
   remains bounded by the documented chunking design and the runtime is usable
   for the intended experiment.** If it does not, LETKF stays explicitly
   EXPERIMENTAL and the limitation is recorded in this file and in
   `docs/data_assimilation.md`. The plan forbids compensating by changing
   defaults or by weakening localization to make the numbers fit — a larger
   localization radius or a smaller grid is a different experiment, not a pass.
3. Then skill. Held-out sensor skill and innovation consistency are read
   together; a posterior is not better because its spread collapsed.

Defaults do **not** change in this PR under any outcome:
`filtering/analysis=stochastic` and `filtering/state_reduction=none` remain the
shipped defaults, and observation TSVD ships disabled. Promotion of any new
default requires results on more than this one case.

## Provenance (required)

| Item | Recorded value |
|---|---|
| Campaign date/time and timezone | TBD |
| Git commit and dirty-tree diff/archive | TBD |
| Python/Pixi lock or environment identifier (`dev` and `cuda`; see below) | TBD |
| Host, OS, kernel | TBD |
| CPU model / physical and logical cores | TBD |
| RAM and NUMA layout | TBD |
| Accelerator and driver, if used | TBD |
| BLAS/JAX backend, platform (`JAX_PLATFORMS`) and precision (`jax_enable_x64`) | TBD |
| CFD backend/build/compiler and revision | `pyudales`; TBD |
| Case and resolved domain shape (`nx, ny, nz`, bounds) | `xie_and_castro`; TBD — read from each run's `config.yaml` |
| Flattened state dimension `N_s` and state variables | TBD |
| Ensemble size `N_e` / parallel workers / CPUs per worker | `50 / TBD / TBD` |
| Observation dimension `N_d` per cycle and sensor definitions | TBD (about 12: 6 sensors x `obs.states=[u, v]`); archive resolved `obs` subtree |
| Localization strategy and resolved parameters (radius, beta, `max_inflation`, `block_grouping`) | TBD per run — read from `run_info.yaml.configuration.localization` |
| Unique local block count and block-grouping mode | TBD — `local_num_blocks` / `local_num_active_blocks` per cycle in `cycle_diagnostics.yaml` |
| LETKF chunk size (derived element budget; not a config knob) | TBD — `local_chunk_size` per cycle in `cycle_diagnostics.yaml` |
| Observation TSVD settings (`energy_fraction`, `max_rank`, numerical tolerance) | TBD — read from `run_info.yaml.configuration.analysis.tsvd` |
| State reduction | `none` in every run of this campaign |
| Ensemble save mode (`run.ensemble_save_on_disk`) | TBD |
| Number of cycles and segment length (`time.simulation_time`, `obs.interval_seconds`) | TBD |
| Shared truth artifact path and checksum | TBD |
| Shared seeds (filter, truth, prior, failures) | TBD |
| Run root | `.temp/filtering_ensemble_transform_benchmark/` (or record replacement) |
| Failures, donor substitutions, retries | TBD, including member/cycle |

Timings without hardware, state shape, `N_e`, `N_d`, retained rank,
localization, save mode, and solver are not comparable (plan §8). All of those
fields are in the table above; leaving one `TBD` while filling a timing cell
invalidates that timing.

`run_info.yaml.configuration` records the fully resolved `analysis` **and**
`localization` subtrees, which is the pair every gate number depends on;
`cycle_diagnostics.yaml` records the transform diagnostics per cycle (see
[Resource gate](#resource-gate-unpopulated)). Neither needs re-deriving from
`config.yaml`.

Each run directory must retain `config.yaml`, `run_info.yaml`,
`cycle_diagnostics.yaml`, `run_summary.yaml`, and the normal pipeline outputs.
Record paths and checksums for any shared truth/prior artifacts. Large NetCDF
states, figures, and solver logs stay in the gitignored results tree
(`.temp/...`), not in Git; this file records paths, checksums, and numbers only.

### Which config baseline this assumes

**This campaign assumes the COMMITTED `conf/` defaults at the recorded commit,
not the working tree.** Several `conf/*.yaml` files carry uncommitted
live-tuning edits, and they change exactly the quantities this benchmark
reports. The table below is a **snapshot taken 2026-08-11**, not a maintained
list — the live tuning moves, and two of the rows below already changed within
a day of the snapshot being taken. Re-derive it with `git diff -- conf/` in the
checkout you are about to run, and treat the snapshot only as the reminder that
the difference exists:

| Key | Committed | Working tree |
|---|---|---|
| `conf/run_filtering.yaml` `filtering/localization` | `none` | `distance` |
| `conf/run_filtering.yaml` `filtering/evolution` | `none` | `random_walk` |
| `conf/run_filtering.yaml` `filtering.mode` | `parameter` | `joint` |
| `conf/run_filtering.yaml` `filtering.num_cycles` | `250` | `60` |
| `conf/run_filtering.yaml` `params@truth_params` | `dynamic_truth` | `dynamic_sine` |
| `conf/case/xie_and_castro.yaml` `domain.nx, ny, nz` | `50, 40, 16` | `40, 60, 24` |
| `conf/case/xie_and_castro.yaml` `domain.bounds[0]` | `[-20.0, 80.0]` | `[-20.0, 40.0]` |
| `conf/case/xie_and_castro.yaml` `obs.interval_seconds` | `60.0` | `30.0` |
| `conf/case/xie_and_castro.yaml` `time.simulation_time` | `300.0` | `150.0` |
| `conf/filtering/localization/distance.yaml` `localization_radius` | `10.0` | `7.5` |

The canonical grid for this campaign is therefore the committed
`50 x 40 x 16` domain over `x in [-20, 80]`, and the canonical localization
radius is the committed `10.0`. Two consequences:

- The commands below pin every axis the campaign varies on the CLI, so group
  selection is independent of the defaults list. They do **not** pin the domain
  shape or the localization radius, which come from the case/localization YAML.
  Run the campaign from a clean checkout (or a git worktree) of the recorded
  commit so those resolve to the committed values.
- Whatever actually resolved is authoritative. Read `domain`, `obs`, and the
  `localization` subtree back out of each run's saved `config.yaml`, record them
  in the provenance table, and archive `git diff -- conf/` if the tree was not
  clean. A run whose grid or radius differs from its siblings is not part of
  this campaign.

## Reproducible commands

First enter the pinned environment and record the exact revision and machine:

```bash
pixi shell -e dev
git rev-parse HEAD
git status --short
git diff -- conf/ > conf_worktree_diff.patch   # must be empty for a clean campaign
uname -a
lscpu
free -h
```

On macOS substitute `sysctl -n machdep.cpu.brand_string`, `sysctl hw.memsize`,
and `/usr/bin/time -l` for the Linux hardware and peak-memory commands below.

`scripts/run_filtering_pipeline.sh` runs stage 1 (`run_filtering.py`) under
`pixi run -e cuda` and stages 2–3 (`compute_filtering_metrics.py`,
`make_filtering_figures.py`) under `pixi run -e dev`. Record the lock/revision
of **both** environments, and record whether the `cuda` environment actually
resolved to a GPU on this host (`JAX_PLATFORMS`, device list) — an ETKF/LETKF
timing on GPU and one on CPU are not comparable.

Create or select one production truth artifact once, then set `TRUTH_DIR` to the
directory containing its `state.nc` and `params.nc`. Record checksums of both
files. The commands below deliberately run the full three-stage filtering
pipeline so held-out scores come from `build_sensor_sets(cfg)` and saved
artifacts rather than from a validation operator in `run_filtering.py`.

```bash
export TRUTH_DIR=/absolute/path/to/shared_truth
export BENCH_ROOT=.temp/filtering_ensemble_transform_benchmark

COMMON="case=xie_and_castro model@truth_model=pyudales model@assim_model=pyudales params@truth_params=static_truth params@prior_params=static run.truth_dir=${TRUTH_DIR} filtering.mode=joint filtering.num_cycles=60 filtering/state_reduction=none filtering/inflation=rtps filtering/evolution=random_walk filtering.seed=42 ensemble.ensemble_size=50 ensemble.num_parallel_processes=8 ensemble.num_cpus_per_process=1 ensemble.failure.seed=0"

# --- Global half: unlocalized stochastic EnKF vs ETKF vs ETKF + observation TSVD ---
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/localization=none filtering/analysis=stochastic paths.results_dir=${BENCH_ROOT}/enkf_global
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/localization=none filtering/analysis=etkf      paths.results_dir=${BENCH_ROOT}/etkf_global
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/localization=none filtering/analysis=etkf_tsvd paths.results_dir=${BENCH_ROOT}/etkf_global_tsvd

# --- LETKF resource gate: localized stochastic EnKF vs LETKF vs LETKF + TSVD ---
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/localization=distance filtering/analysis=stochastic paths.results_dir=${BENCH_ROOT}/enkf_local
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/localization=distance filtering/analysis=letkf      paths.results_dir=${BENCH_ROOT}/letkf_full
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/localization=distance filtering/analysis=letkf_tsvd paths.results_dir=${BENCH_ROOT}/letkf_tsvd
```

**Verified against the merged PR 2 configs:** all four analysis groups
`filtering/analysis={etkf,etkf_tsvd,letkf,letkf_tsvd}` exist in
`conf/filtering/analysis/`, and each is exercised end to end by
`tests/test_run_filtering.py::test_run_filtering_ensemble_transform` (smoke
shape, real solver). The nested TSVD knobs are reachable as
`filtering.analysis.tsvd.{enabled,energy_fraction,max_rank,numerical_tolerance}`
on the `*_tsvd` variants; on `etkf`/`letkf` the whole `tsvd` node is `null`, so
turning truncation on there means selecting the `*_tsvd` group, not overriding a
leaf under a null node. Two knobs are asymmetric if a sweep toggles them:
`max_rank` is part of the scientific truncation and is **rejected at
construction** together with `enabled=false`, while `numerical_tolerance` is not
gated by `enabled` at all and takes effect either way.

The LETKF chunk size is **not** a config knob. `_resolve_chunk_size`
(`filtering/etkf.py`) derives it from a fixed element budget, because the
footprint scales with `N_e * min(N_d, N_e)` and no fixed block *count* covers
the supported backends. The constructor's `block_chunk_size` override exists for
tests and for a deliberate sweep; reach it with
`+filtering.analysis.block_chunk_size=<n>` (a `+` add, since the group files do
not set the key) and record that you did. Otherwise record the value the run
resolved, which every cycle reports as `local_chunk_size` in
`cycle_diagnostics.yaml`.

Confirm each command composes with
`python scripts/filtering/run_filtering.py --cfg job <overrides>` before
committing compute to it.

Verified: `scripts/run_filtering_pipeline.sh` forwards all extra arguments to
`run_filtering.py` and re-composes `conf/run_filtering.yaml` with the same
overrides to resolve the run dir, so `paths.results_dir=...` is honored by all
three stages and no post-hoc moving of results is needed.

Shell variables are shown for readability. Before execution, save the expanded
command for every run (for example with `set -x`) and confirm that Hydra's
resolved `config.yaml` contains identical non-analysis, non-localization
settings across all six runs. Capture `/usr/bin/time` stderr per run so maximum
resident set size is preserved.

### Seed replicates (required for the stochastic comparison)

The stochastic EnKF is a Monte-Carlo scheme and the ETKF/LETKF transforms are
deterministic. A single-seed difference between them is not a result. Repeat the
two stochastic baselines over at least the seed set below, holding everything
else fixed, and report the spread across seeds beside the deterministic point
value:

```bash
for SEED in 42 43 44 45 46; do
  /usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON/filtering.seed=42/filtering.seed=${SEED}} filtering/localization=none     filtering/analysis=stochastic paths.results_dir=${BENCH_ROOT}/enkf_global_seed${SEED}
  /usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON/filtering.seed=42/filtering.seed=${SEED}} filtering/localization=distance filtering/analysis=stochastic paths.results_dir=${BENCH_ROOT}/enkf_local_seed${SEED}
done
```

**Unverified:** the `${COMMON/.../...}` bash substitution is shown for brevity;
expand and log the literal override string per run rather than trusting it.

## Run inventory

| ID | Analysis | Localization | Obs TSVD | State reduction | Run directory | Exit/failures |
|---|---|---|---|---|---|---|
| enkf_global | stochastic | none | off | none | TBD | TBD |
| etkf_global | etkf | none | off | none | TBD | TBD |
| etkf_global_tsvd | etkf | none | on (`etkf_tsvd`: energy_fraction 0.99, max_rank null, numerical_tolerance null) | none | TBD | TBD |
| enkf_local | stochastic | distance | off | none | TBD | TBD |
| letkf_full | letkf | distance | off (`tsvd: null` — every thin-SVD direction retained) | none | TBD | TBD |
| letkf_tsvd | letkf | distance | on (`letkf_tsvd`: energy_fraction 0.99, max_rank null, numerical_tolerance null) | none | TBD | TBD |
| enkf_global_seed{43..46} | stochastic | none | off | none | TBD | TBD |
| enkf_local_seed{43..46} | stochastic | distance | off | none | TBD | TBD |

`letkf_*` runs must reject `filtering/state_reduction != none` at construction
(plan §6, step 6): LETKF plus PR 1's global state reduction is deliberately
unsupported. If such a run starts instead of raising, that is a defect, not a
benchmark configuration.

## Resource gate (unpopulated)

These six columns are the gate criteria named by the plan. Fill every cell from
a measured run; leave none inferred from another row.

| ID | Wall time (end-to-end) | Peak memory | Active observations per block (min / median / max) | Unique block count | Chunk size | Time per cycle (analysis / total) |
|---|---:|---:|---:|---:|---:|---:|
| enkf_global | TBD | TBD | n/a (global) | n/a | n/a | TBD / TBD |
| etkf_global | TBD | TBD | n/a (global) | n/a | n/a | TBD / TBD |
| etkf_global_tsvd | TBD | TBD | n/a (global) | n/a | n/a | TBD / TBD |
| enkf_local | TBD | TBD | TBD (see note) | n/a (no dedup) | n/a | TBD / TBD |
| letkf_full | TBD | TBD | TBD | TBD | TBD | TBD / TBD |
| letkf_tsvd | TBD | TBD | TBD | TBD | TBD | TBD / TBD |

Sources for each column, verified against the merged PR 2 implementation. Every
per-block quantity the gate names is now recorded per cycle in
`cycle_diagnostics.yaml`; none of it requires re-instrumenting a run:

- **Wall time** — `/usr/bin/time` elapsed for the whole pipeline invocation, and
  separately the stage-1 (`run_filtering.py`) portion, since stages 2–3 are
  identical work across runs and would otherwise dilute the comparison.
- **Peak memory** — `/usr/bin/time -v` maximum resident set size (`-l` on
  macOS). Note that this is a per-process maximum over the invocation and its
  waited-for children, **not** a sum across the `ensemble.num_parallel_processes`
  forecast workers. The gate is about the analysis, whose allocation is what the
  chunking design bounds, so also sample RSS during the analysis phase directly
  (for example a sampler on the stage-1 PID) and record both numbers with their
  method. State which number the gate verdict is based on.
- **Active observations per block** — the per-block `N_d_active` after
  localization and infinite-inflation exclusion. Every cycle records the summary
  directly as `local_active_obs_{min,median,max}` in `cycle_diagnostics.yaml`;
  aggregate those over cycles. Record the distribution, not only a mean: the
  gate is about the worst block as much as the typical one. The summary covers
  the **active** blocks only — blocks with zero active observations are counted
  separately as `local_num_blocks - local_num_active_blocks`, and would
  otherwise pin every minimum at zero.
- **Unique block count** — `local_num_blocks` in `cycle_diagnostics.yaml`: the
  number of distinct local analyses the LETKF actually solves per cycle. This is
  **not** the number of distinct `group_ids` blocks. The implementation
  deduplicates on the canonical per-row **inflation vector**, because the
  transform is a function of `(pred_obs, obs, C_D, E_inf_row)` alone, so rows in
  different `group_ids` blocks that see the same observation selection share one
  transform. (Plan §6 correction #1: on the staggered uDALES grid, `group_ids`
  dedup collapses nothing — `pres`/`u`/`v`/`w` each carry a distinct grid
  signature, so unique `group_ids` blocks equal `N_s` exactly — 230,400 on the
  working-tree grid the correction was measured on. Whether inflation-vector
  dedup collapses anything, and by how much, is one of this campaign's
  measurements; the plan records no number for it, so do not carry one in.)
  Record it against both the resolved
  `nx * ny * nz` and `N_s`, plus `local_num_active_blocks` — at
  `localization_radius = 7.5` only about 5% of blocks had any active
  observation, and the inactive remainder is partitioned out rather than solved.
- **Chunk size** — `local_chunk_size` in `cycle_diagnostics.yaml`. It is a
  derived element budget, not a config knob (see the note under the commands);
  record the resolved value, and if a sweep overrode it, record the override.
- **Time per cycle** — mean and median `analysis_time` from
  `cycle_diagnostics.yaml`, plus mean end-to-end seconds per cycle. Report both:
  the CFD forecast dominates the total and would otherwise mask an analysis
  regression entirely. Run 1's baseline is already measured on this hardware at
  full size (plan §6 correction #4: localized stochastic EnKF, `analysis_time`
  mean 1.289 s/cycle over 60 cycles) — reproduce it in this campaign rather than
  importing the number, so it shares the row's provenance.

`n/a` cells above are genuine non-applicability (a global analysis has no
blocks), not unmeasured values. Do not replace them with `TBD`.

The `enkf_local` row needs two clarifications, because the `local_*` diagnostics
belong to the LETKF analysis and the localized *stochastic* update publishes
none:

- **Active observations per block.** The selection comes from
  `BaseLocalization.inflation_factors`, which both analyses call with the same
  strategy and radius, so the per-row active-observation distribution of
  `enkf_local` is by construction the one `letkf_full` reports. Record it from
  the `letkf_full` row and say so; do not present it as an independent
  measurement of the stochastic run.
- **Unique block count.** This one is *not* shared. The stochastic
  `localized_update` solves every augmented row independently (`jax.vmap` over
  rows); `group_ids` only harmonizes the selection co-located rows see, and
  nothing deduplicates identical selections. It therefore has no comparable
  count and the cell is `n/a` rather than `TBD`. That asymmetry is part of the result: it is one of the reasons the
  two runs' analysis costs are not a like-for-like ratio of block counts.

## Results (unpopulated)

Report per-run point estimates and, where available, variation over cycles. For
the stochastic rows, report the across-seed spread from the replicate runs. Link
every value to its output artifact/key.

| ID | State RMSE/statistic | Assimilated sensor score | Held-out sensor score | Innovation chi-square | State spread | Joint parameter recovery | Retained rank (mean / max) | Retained obs-space energy | Analysis time | End-to-end time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enkf_global | TBD | TBD | TBD | TBD | TBD | TBD | n/a (no transform) | n/a (no transform) | TBD | TBD |
| etkf_global | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| etkf_global_tsvd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| enkf_local | TBD | TBD | TBD | TBD | TBD | TBD | n/a (no transform) | n/a (no transform) | TBD | TBD |
| letkf_full | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| letkf_tsvd | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Also summarize, per run and per cycle, the observation-space TSVD diagnostics the
plan requires: available rank, retained rank, retained energy, and the discarded
spectrum. `cycle_diagnostics.yaml` carries them for every run, `null` where they
do not apply, so no run needs re-instrumenting:

| Quantity | Key (global `etkf*`) | Key (local `letkf*`) |
|---|---|---|
| Available rank | `transform_available_rank` | `local_available_rank_max` |
| Retained rank | `transform_retained_rank` | `local_retained_rank_{min,mean,max}` |
| Retained energy | `transform_retained_energy` | `local_retained_energy_{min,mean}` |
| Discarded spectrum | `transform_discarded_spectrum_max` | `local_discarded_spectrum_max` |

All four quantities are recorded per cycle on **both** the global and the local
path; nothing in this campaign has to be inferred from the ranks alone, and no
run needs re-instrumenting. The per-block energy readouts are computed inside
the traced block loop as plain reductions over each block's spectrum, so they
cost no host synchronization.

Both groups are `null` on the `stochastic` rows, which is the correct value: a
perturbed-observation update forms no transform. Zero is not null in either
group: `transform_discarded_spectrum_max = 0.0` (and, locally,
`local_discarded_spectrum_max = 0.0` with `local_retained_energy_* = 1.0`)
means the truncation ran and discarded nothing, whereas `null` means no
transform of that kind ran.

For the LETKF rows the ranks and energies are distributions over local blocks.
The per-cycle record keeps the min/mean/max summaries (and, for the energies,
min/mean); the full per-block arrays live on `LETKFAnalysis.last_diagnostics`
for interactive inspection and are deliberately not written every cycle (one
entry per block). The summaries cover the **active** blocks only.

Report LETKF TSVD activity from both readouts, and say which one the conclusion
rests on. From the ranks: with the TSVD disabled every block retains the full
fixed rank `min(N_d, N_e)`, so `local_retained_rank_max < min(N_d, N_e)` means
every block truncated and `local_retained_rank_min < min(N_d, N_e)` means at
least one did. From the energies: `local_discarded_spectrum_max` is the largest
singular value any block dropped, and `local_retained_energy_min` is the
worst-case retained fraction, which is what distinguishes "truncated a
round-off direction" from "truncated a direction that carried information". If
both say truncation removed essentially nothing, TSVD is inactive and any
difference between `letkf_full` and `letkf_tsvd` is noise — itself a reportable
result.

### `obs_posterior_rmse` must not rank ETKF against LETKF

The appended predicted-observation rows take a **global, full-space ride-along
update** in every scheme. Under LETKF and under localized stochastic analyses,
`obs_posterior_rmse` is therefore a global-analysis proxy — it is not `H`
applied to the row-wise localized posterior that those runs actually produce.
Comparing `etkf_global`'s value against `letkf_full`'s compares two quantities
with different definitions and will produce a confident, meaningless ordering.

PR 1 added the `obs_posterior_rmse_kind` provenance label to
`cycle_diagnostics.yaml` (`data_assimilation/filtering/base.py`) precisely so
this is visible per cycle. Record that label for every run in this campaign and
refuse to tabulate `obs_posterior_rmse` across runs whose labels differ. Rank
schemes on the held-out and assimilated sensor scores from the pipeline's metric
stage instead.

### Stochastic vs deterministic is a statistical comparison, and belongs here

Comparing the stochastic EnKF against the ETKF/LETKF is a comparison between a
Monte-Carlo estimator and a deterministic one. At `N_e=50` the stochastic
scheme's sampling noise is the same order as the effect being measured. That
comparison belongs in this offline record — pinned seeds, seed replicates,
deliberately generous tolerances — and **not** in CI, where it would be flaky
and would eventually be silenced by widening a tolerance until it always passes.
CI keeps the exact linear-Kalman-filter mean/covariance comparison and the
identity tests (all-ones localization equals global ETKF, TSVD-disabled with no
`numerical_tolerance` equals the full-rank transform, chunked equals unchunked),
which are deterministic.

### The pass condition, and what failure means

The LETKF passes the resource gate only if **both** hold: peak memory remains
bounded by the documented chunking design (no `(n_blocks, N_e, N_e)` transform
tensor is ever materialized for the full domain, and observed peak memory is
consistent with the documented bound), and the runtime is usable for the
intended experiment. "Usable" must be stated as a number before the campaign
runs — record the intended experiment's cycle count and its acceptable
wall-clock budget in the provenance table, then compare.

If it fails, LETKF stays explicitly **EXPERIMENTAL**, the limitation is recorded
here and in `docs/data_assimilation.md`, and that is the end of the matter. The
plan forbids compensating by changing defaults or weakening localization: do not
enlarge the localization radius, coarsen the grid, drop the ensemble size, or
promote a different analysis to default in order to convert a failure into a
pass. Those are separate experiments with their own records.

### Held-out scores come from the pipeline, not from the filter

Held-out sensor scores come from the filtering post-processing pipeline —
`build_sensor_sets(cfg)` plus `compute_filtering_metrics.py` reading the saved
state/config artifacts — and **not** from a validation operator inside
`run_filtering.py`, which has none. That is why every command above runs the
full three-stage `run_filtering_pipeline.sh` rather than stage 1 alone.

Smoke-shaped configurations can place the case's validation sensors outside
their reduced domain, so smoke runs verify wiring and correctness only and are
never held-out acceptance. No number in the results table may come from a
smoke-shaped run.

### Expect TSVD to do little here, and record that honestly

With about 12 global observations, and fewer active observations per local
block, the plan's stated expectation is that observation TSVD provides **little
benefit inside LETKF**. A result of "no measurable difference between
`letkf_full` and `letkf_tsvd`" is a complete, publishable outcome for this
campaign and must be recorded as such — not softened, not re-run with different
settings until a difference appears, and not padded with a speculative
suggestion that a larger sensor network would help unless that too is measured.

Observation TSVD must remain **off by default** until the diagnostics show
persistent local ill-conditioning. The evidence for that would be the retained
local condition indicator and the discarded-spectrum diagnostics across cycles,
not a single favorable skill number.

## Required conclusions (leave blank until measured)

- Global ETKF vs global stochastic EnKF, skill and calibration, with the
  across-seed spread of the stochastic baseline: TBD.
- Whether global observation TSVD changes anything at `N_d ≈ 12`: TBD.
- LETKF vs localized stochastic EnKF, held-out skill and innovation
  consistency: TBD.
- Whether observation TSVD changes the LETKF result, read from both the
  per-block rank summaries and the per-block energy summaries: TBD. (A literal
  *fraction of blocks that truncated* is not in `cycle_diagnostics.yaml` — the
  min/mean/max summaries bracket it, and the exact fraction needs the per-block
  arrays on `LETKFAnalysis.last_diagnostics`.)
- Unique block count, active-observation distribution, and chunk size actually
  used on the canonical grid: TBD.
- LETKF peak memory against the documented chunking bound, with the measurement
  method stated: TBD.
- LETKF wall time and time per cycle against the localized stochastic baseline,
  and against the stated wall-clock budget for the intended experiment: TBD.
- **Resource-gate verdict (pass / fail, with the numbers it rests on): TBD.**
  If fail: LETKF is marked EXPERIMENTAL and the limitation recorded; defaults
  and localization are unchanged.
- Whether joint-mode parameter recovery differs between the transform family and
  the stochastic baseline: TBD.
- Default analysis decision: **unchanged — `stochastic`, with
  `state_reduction=none` and observation TSVD off.** Any change requires results
  on more than this one case.

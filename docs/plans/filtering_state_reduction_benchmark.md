# Filtering state-reduction benchmark record

> **Status: campaign template; not run.** No scientific, accuracy, memory, or
> performance benefit is claimed here. Fill every provenance field and replace
> every `TBD` only after executing the production/offline campaign below. Smoke
> tests establish wiring and correctness only; their reduced domain does not
> provide a valid held-out-sensor benchmark.

## Purpose and acceptance order

This campaign compares the existing full stochastic EnKF against current-cycle
SVD projection at full statistical rank, three retained-energy thresholds, and
two hard caps below the default observation dimension (about 12). All runs must
use the same truth, prior sampler, ensemble size, observation noise, and seeds.
Interpret truncated runs only after the full-rank equivalence control passes.

A proposed default must preserve held-out sensor skill and innovation
consistency without producing an artificially collapsed posterior spread.
`filtering/state_reduction=none` remains the shipped default regardless of this
first campaign. Since the CFD forecast is unchanged and dominates runtime, do
not infer a speedup from smaller analysis arrays; measure end-to-end time and
peak memory.

## Provenance (required)

| Item | Recorded value |
|---|---|
| Campaign date/time and timezone | TBD |
| Git commit and dirty-tree diff/archive | TBD |
| Python/Pixi lock or environment identifier | TBD |
| Host, OS, kernel | TBD |
| CPU model / physical and logical cores | TBD |
| RAM and NUMA layout | TBD |
| Accelerator and driver, if used | TBD |
| BLAS/JAX backend and precision settings | TBD |
| CFD backend/build/compiler and revision | TBD |
| Case and resolved domain shape | `xie_and_castro`; verify from each `config.yaml` |
| Ensemble size / parallel workers / CPUs per worker | `50 / TBD / TBD` |
| Observation count and sensor definitions | TBD; archive resolved `obs` subtree |
| Shared truth artifact path and checksum | TBD |
| Shared seeds (filter, truth, prior, failures) | TBD |
| Run root | `.temp/filtering_state_reduction_benchmark/` (or record replacement) |
| Failures, donor substitutions, retries | TBD, including member/cycle |

Each run directory must retain `config.yaml`, `run_info.yaml`,
`cycle_diagnostics.yaml`, `run_summary.yaml`, and the normal pipeline outputs.
Record paths and checksums for any shared truth/prior artifacts. Large NetCDF
states and solver logs stay in the gitignored results tree, not in Git.

## Reproducible commands

First enter the pinned environment and record the exact revision and machine:

```bash
pixi shell -e dev
git rev-parse HEAD
git status --short
uname -a
lscpu
free -h
```

On macOS substitute `sysctl -n machdep.cpu.brand_string`, `sysctl hw.memsize`,
and `/usr/bin/time -l` for the Linux hardware and peak-memory commands below.
Create or select one production truth artifact once, then set `TRUTH_DIR` to
the directory containing its `state.nc` and `params.nc`. Record checksums of
both files. The commands below deliberately run the full three-stage filtering
pipeline so held-out scores come from `build_sensor_sets(cfg)` and saved
artifacts rather than from a validation operator in `run_filtering.py`.

```bash
export TRUTH_DIR=/absolute/path/to/shared_truth
export BENCH_ROOT=.temp/filtering_state_reduction_benchmark

COMMON="case=xie_and_castro model@truth_model=pyudales model@assim_model=pyudales params@truth_params=static_truth params@prior_params=static run.truth_dir=${TRUTH_DIR} filtering.mode=joint filtering.num_cycles=60 filtering/analysis=stochastic filtering/localization=none filtering/inflation=rtps filtering/evolution=random_walk filtering.seed=42 ensemble.ensemble_size=50 ensemble.num_parallel_processes=8 ensemble.num_cpus_per_process=1"

/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/state_reduction=none paths.results_dir=${BENCH_ROOT}/full
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=1.0 filtering.state_reduction.max_rank=null paths.results_dir=${BENCH_ROOT}/svd_full_rank
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=0.90 filtering.state_reduction.max_rank=null paths.results_dir=${BENCH_ROOT}/svd_energy_090
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=0.95 filtering.state_reduction.max_rank=null paths.results_dir=${BENCH_ROOT}/svd_energy_095
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=0.99 filtering.state_reduction.max_rank=null paths.results_dir=${BENCH_ROOT}/svd_energy_099
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=1.0 filtering.state_reduction.max_rank=4 paths.results_dir=${BENCH_ROOT}/svd_rank_04
/usr/bin/time -v scripts/run_filtering_pipeline.sh ${COMMON} filtering/state_reduction=svd_current filtering.state_reduction.energy_fraction=1.0 filtering.state_reduction.max_rank=8 paths.results_dir=${BENCH_ROOT}/svd_rank_08
```

Shell variables are shown for readability. Before execution, save the expanded
command for every run (for example with `set -x`) and confirm that Hydra's
resolved `config.yaml` contains identical non-reduction settings. Capture
`/usr/bin/time` stderr per run so maximum resident set size is preserved. If
the platform's pipeline wrapper does not forward `paths.results_dir`, record the
Hydra-selected run directory printed by the command instead of moving results.

## Run inventory

| ID | Reduction | Energy | Max rank | Run directory | Exit/failures | Peak RSS | Wall time |
|---|---|---:|---:|---|---|---:|---:|
| full | none | — | — | TBD | TBD | TBD | TBD |
| svd_full_rank | current | 1.00 | none | TBD | TBD | TBD | TBD |
| svd_energy_090 | current | 0.90 | none | TBD | TBD | TBD | TBD |
| svd_energy_095 | current | 0.95 | none | TBD | TBD | TBD | TBD |
| svd_energy_099 | current | 0.99 | none | TBD | TBD | TBD | TBD |
| svd_rank_04 | current | 1.00 | 4 | TBD | TBD | TBD | TBD |
| svd_rank_08 | current | 1.00 | 8 | TBD | TBD | TBD | TBD |

## Results (unpopulated)

Report per-run point estimates and, where available, variation over cycles.
Link every value to its output artifact/key.

| ID | State RMSE/statistic | Assimilated sensor score | Held-out sensor score | Innovation chi-square | State spread | Joint parameter recovery | Mean retained rank/energy | Analysis time | End-to-end time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | TBD | TBD | TBD | TBD | TBD | TBD | n/a | TBD | TBD |
| svd_full_rank | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| svd_energy_090 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| svd_energy_095 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| svd_energy_099 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| svd_rank_04 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| svd_rank_08 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Do **not** rank reduced against unreduced runs on `obs_posterior_rmse`: the
appended predicted-observation rows take a global, full-space ride-along
update, so that number is insensitive to truncation by construction (it comes
back bit-identical). `cycle_diagnostics.yaml` records
`obs_posterior_rmse_kind` so this is visible per cycle. Use the held-out and
assimilated sensor scores from the pipeline's metric stage instead.

Analysis cost is comparable across runs: `analysis_time` is recorded on every
cycle of every run, reduced or not, and `reduction_basis_time` isolates the SVD
inside it.

Also summarize projection residual, decoded-increment norm/discarded fraction,
retained energy (measured against the full spectrum, so a numerical-rank drop
shows up as a value below one), spectral condition indicator, and any reduction
warning per cycle from `cycle_diagnostics.yaml`. A separate streaming campaign should add forgetting
factor, covariance half-life, update cadence, and subspace drift; it must retain
the same full/current controls.

## Required conclusions (leave blank until measured)

- Full-rank equivalence result and tolerance: TBD.
- Whether any truncated setting degrades held-out skill: TBD.
- Whether innovation consistency and physical spread remain calibrated: TBD.
- Whether joint-mode parameter recovery changes: TBD.
- Whether batch SVD is slower than the existing cross-covariance at the default
  `N_e=50`, `N_d≈12` shape: TBD.
- End-to-end wall-time and peak-memory comparison: TBD.
- Candidate default decision: **none pending multi-case evidence**.

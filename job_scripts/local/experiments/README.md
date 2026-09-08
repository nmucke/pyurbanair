# DA-method experiment campaigns

Three campaign drivers — one per assimilation method — that sweep the **same
axes** with the **same constants**, so their run dirs are directly comparable:

```
job_scripts/local/experiments/
├── settings.sh                            # THE knob file (constants + axes)
├── campaign_lib.sh                        # queue, lanes, markers, guards
├── run_esmda_experiments.sh               # -> scripts/run_esmda_pipeline.sh
├── run_filtering_experiments.sh           # -> scripts/run_filtering_pipeline.sh
└── run_filter_smoothing_experiments.sh    # -> scripts/run_filter_smoothing_pipeline.sh
```

Each driver expands the axes into one run per combination and hands each run to
the corresponding **pipeline** script, so every run is assimilated *and*
scored *and* plotted (metrics + figures, including the ESMDA-schema view for the
filtering and hybrid pipelines).

The assimilation model is always `pyudales`. The **truth** model is an axis:
with `pyudales` the truth and the assimilation model are the same solver, so a
run measures the DA method alone; with `pypalm` it is a genuine cross-model
experiment that additionally carries model error (see [Cross-model
runs](#cross-model-runs-pypalm-truth)).

## Usage

```bash
# see the plan without running anything (prints every resolved command)
DRY_RUN=1 bash job_scripts/local/experiments/run_esmda_experiments.sh

# run a campaign
bash job_scripts/local/experiments/run_filtering_experiments.sh

# detached, so it outlives the shell
setsid nohup bash job_scripts/local/experiments/run_esmda_experiments.sh \
  > /dev/null 2>&1 < /dev/null &
tail -f /export/scratch2/ntm/experiments/esmda/_logs/*.log
```

Any knob can be set by editing [`settings.sh`](settings.sh) or exported for one
campaign:

```bash
NUM_WINDOWS_LIST="2 4 8" INFLOW_LIST=inflow_turb \
  bash job_scripts/local/experiments/run_filter_smoothing_experiments.sh
```

## The axes

`settings.sh` holds every knob. The `*_LIST` ones are swept (full cross
product); everything else is constant across all three campaigns.

| Axis | Values (default) | ESMDA | Filtering | Filter smoothing |
|---|---|---|---|---|
| `TRUTH_MODEL_LIST` | `pyudales` (also `pypalm`) | `model@truth_model` + that mount's forcing | same | same |
| `NUM_WINDOWS_LIST` | `3` | `esmda.num_assimilation_windows` | `filtering.num_assimilation_windows` | `filter_smoothing.num_assimilation_windows` |
| `LOCALIZATION_LIST` | `none correlation` | `esmda/localization` | `filtering/localization` | both mounts (see `LOCALIZATION_SCOPE`) |
| `OBS_INTERVAL_LIST` | `15.0` | `esmda.interval_seconds` | **n/a** | `esmda.interval_seconds` (smoother half only) |
| `INFLOW_LIST` | `inflow inflow_turb periodic` | both model mounts | both model mounts | both model mounts |

`SIMULATION_TIME` (per-window horizon) is deliberately a **constant**, not an
axis; the num-windows axis is what varies the total horizon
(`SIMULATION_TIME * windows`).

**Spin-up is per inflow setting**, the one non-axis knob that is not a single
value: `SPINUP_TIME_INFLOW=50` s for `inflow` / `inflow_turb`, and
`SPINUP_TIME_PERIODIC=150` s for `periodic`, which has to build its whole
momentum field from rest through the nudging relaxation instead of being fed by
an inlet.

**Where the constants come from.** Every non-axis value in `settings.sh` mirrors
the entry-point configs on the `isda_experiments` branch (120 s windows, 2 s
output, 30 s knots, nz=16, `obs_error_std=0.1`, hybrid `filtering.mode=state`),
so a campaign reproduces the branch's intent from a clean checkout. They are
**pinned, not inherited** — edit a config on the branch and you must mirror it
here too. Two deliberate deviations:

- `ENSEMBLE_SIZE=50` for all three. The branch sets 50 in `run_esmda.yaml` /
  `run_filtering.yaml` but 40 in `run_filter_smoothing.yaml`; the campaigns need
  one value to stay comparable across methods.
- `NUM_ESMDA_STEPS=3` (the branch configs say 2), applied to both the ESMDA and
  the hybrid campaign so the two smoother halves run the same MDA schedule.

The per-method DA modes are: ESMDA = `esmda/smoother=dynamic` with
`params@prior_params=dynamic` (time-varying parameters, no state update);
filtering = `filtering.mode=joint` (state + parameters, static prior, the only
kind the filter supports); hybrid = the same dynamic MDA parameter loop with
`filtering.mode=state`, so the filter owns the state and the smoother owns the
parameters.

**Observation interval and the filter.** The EnKF assimilates individual frames
— one analysis per observation interval (`OUTPUT_FREQUENCY`) — and never
aggregates, so `OBS_INTERVAL_LIST` is a smoother-side axis and
`run_filtering_experiments.sh` ignores it. Its analogue there is
`ASSIMILATE_EVERY_N_STEP` (the analysis stride), held constant.

**Inflow settings**, applied to the truth *and* the assimilation mount. The
three settings mean the same thing physically in both backends, but the
machinery behind them does not, so the mapping is per backend:

| Value | uDALES | PALM |
|---|---|---|
| `inflow` | `inflow_outflow`, no synthetic inlet turbulence (nudged inlet + interior relaxation) | `inflow_outflow`, inflow disturbances off |
| `inflow_turb` | `inflow_outflow` + the digital-filter synthetic inlet (BCxm=3 / idriver=2); the backend turns volume nudging off in this mode by design | `inflow_outflow` + random inflow disturbances (`dt_disturb` / `disturbance_amplitude`), plus the cold-start kick |
| `periodic` | periodic x BCs; interior nudging is the only momentum source, so it stays on, and no inlet turbulence is possible | periodic BCs + PALM's nudging driver (same relaxation physics and parameter meaning); PALM *rejects* inlet turbulence under cyclic BCs |

The uDALES `inflow_turb` knobs (`INLET_INTENSITY`,
`UDALES_INLET_LENGTH_SCALE_{X,Y,Z}`, `UDALES_INLET_TIME_STEP`) mirror
`conf/model/pyudales.yaml`, and the PALM ones are matched to them. PALM's
`initial_seed` (the cold-start symmetry-breaking kick) is a *separate*
mechanism from inlet turbulence: `inflow_turb` forces it on, the other two
settings leave it at `PALM_INITIAL_SEED` (default off).

### How close are the two `inflow_turb` setups?

The mean inflow is identical in construction on both backends: the same
power-law profile (`alpha=0.25`) built from the same DA-estimated
`velocity_magnitude` / `inflow_angle`. Only the *fluctuations* differ, and the
campaign matches everything the two mechanisms have in common:

| | uDALES | PALM | Aligned by |
|---|---|---|---|
| Nominal rms | `intensity * |U(z)|` | uniform noise on `[-1.5A, 1.5A]`, rms `A*sqrt(3)/2` | `INLET_INTENSITY` — PALM's `A` is derived as `intensity * INLET_REFERENCE_SPEED / (sqrt(3)/2)` (0.05 × 7.5 → **0.433 m/s**, vs PALM's own default 0.25) |
| Time scale | AR(1), `T = L_x / U_ref` = 0.8 s | white in time, one kick per `dt_disturb` | **not matched, deliberately.** PALM *adds* each kick to the field while uDALES *prescribes* a boundary value, so the realised variance grows like `tau_decay/dt_disturb`; pacing kicks at 0.8 s would land far above the 5% target. `dt_disturb` stays at the config's **5 s** |
| Streamwise extent | inlet plane only, advected in | a strip `[begin, end]` grid points | `begin=2, end=14` → x = −17…1 m, upstream of the array. PALM's auto values (10, 29) put the kicks at x = −5…23 m, i.e. *inside* the building array |
| Vertical extent | full inlet plane | `[level_b, level_t]` | `level_t=26.0` m (PALM caps at `zu(nzt-2)`=27 m on the nz=16 grid). PALM's auto range is 5…9 m — below the 17 m rooftops |

What cannot be matched, and is worth remembering when reading the results:

- **Correlated eddies vs white kicks.** uDALES injects a spatially filtered,
  time-correlated field that then advects through the domain; PALM re-randomises
  a strip in place. Only the amplitude and the spatial extent can be equated,
  not the structure or the time scale.
- **PALM's realised intensity is not analytically predictable.** Its kicks
  accumulate, so the equilibrium rms depends on `dt_disturb` against the decay
  time as well as on the amplitude. Measure `u'_rms` upstream of the array in a
  run and retune `PALM_DISTURBANCE_AMPLITUDE` / `PALM_DT_DISTURB` if it misses
  the `INLET_INTENSITY` target — do not assume the nominal value.
- **Realised rms is below nominal in both, by different amounts.** PALM smooths
  the perturbation field twice before adding it; uDALES subtracts the plane mean
  (and loses more the closer `L_y`/`L_z` get to the plane's dimensions). Treat
  the alignment above as equal *nominal* forcing.
- **Profile shape.** uDALES' rms follows `|U(z)|`, so it is sheared with height;
  PALM's is uniform inside the perturbed band.
- **Components.** Both perturb `u` and `v`; uDALES also perturbs `w` (at
  `0.7 ×` the streamwise intensity).
- **Band limiting / self limiting.** uDALES' driver signal is band-limited to
  `1/(2*UDALES_INLET_TIME_STEP)`; PALM's initial kick (not the in-run
  perturbations) stops once resolved TKE passes `disturbance_energy_limit`.

Set the `PALM_DISTURBANCE_*` knobs explicitly to override any of the derived
values, or all four of `AMPLITUDE`/`DT_DISTURB`/`BEGIN`/`END` (plus `LEVEL_T`)
empty to fall back to PALM's own auto behaviour.

## Outputs

```
/export/scratch2/ntm/experiments/<method>/   # RESULTS_ROOT/<method>/
├── _logs/
│   ├── <run id>.log      # the whole pipeline's stdout/stderr
│   ├── <run id>.args     # the exact override set that produced the run
│   ├── <run id>.ok       # completion marker (a re-run skips it)
│   ├── progress.tsv      # id, lane, exit code, start/end, elapsed
│   └── driver.lock       # refuses a second concurrent driver
├── _scratch/lane<N>/     # per-lane paths.experiment_dir (uDALES scratch)
└── <run id>/             # paths.results_dir — a normal pipeline run dir
```

**The filtering campaign prunes `state_history.nc`** from each run dir once that
run's pipeline has finished with it (assimilation, metrics, figures and the
ESMDA-schema view all run first). The filter writes the same analyzed frames
twice — cycle-indexed in `state_history.nc`, window-indexed in
`windows/window_{w}_posterior_state.nc` — which is ~7 GB of duplication per run
here. The window files are kept: they are the only source the ESMDA-schema
stages have, and they are what makes a filtering run comparable to an ESMDA one.
`state_history.nc` rebuilds from them with

```python
xarray.concat([...window_{w}_posterior_state.nc...], "time").rename(time="cycle")
```

The pruning is guarded (it refuses unless those window files exist and their
bytes account for the history's) and only fires on a run that exited 0, so a
failed run keeps everything for the post-mortem. It costs re-analysis only: the
filtering-native stages then fall back to `posterior_state.nc` and their
per-cycle state RMSE covers the last cycle alone. Set `PRUNE_STATE_HISTORY=false`
to keep it. The hybrid duplicates identically and does **not** prune — set
`POST_RUN_HOOK=prune_state_history` on that campaign to opt in.

Run ids encode the axes:
`<truth>_to_<assim>_w<windows>_loc<localization>[_obs<interval>]_<inflow>`, e.g.
`pyudales_to_pyudales_w4_loccorrelation_obs30_inflow_turb` and
`pypalm_to_pyudales_w4_loccorrelation_obs30_inflow_turb`. The `_obs<interval>`
part is absent from the filtering campaign (no aggregation axis).

Method knobs set by env (`ESMDA_SMOOTHER`, `FILTERING_MODE`,
`ASSIMILATE_EVERY_N_STEP`, `PARAMS_TO_ESTIMATE`, …) are **not** in the id, so
two campaigns differing only there would write into the same dirs — give each
variant its own `RESULTS_ROOT`.

`compare_sweep_results.py` globs `*/run_summary.yaml`, so
`${RESULTS_ROOT}/<method>/` is directly usable as a sweep folder.

Outputs live on **scratch2**, not under the repo's `.temp/`: a campaign is
~225 GB (~450 GB with a pypalm truth as well) and the repo's filesystem does not
have the room. `RESULTS_ROOT` moves everything — run dirs, figures, logs and the
per-lane solver scratch — in one place.

## Resumability and failures

A finished run writes `_logs/<id>.ok` and is skipped on re-run; a failed one
records its exit code in `progress.tsv` and the campaign continues. Unlike the
per-stage benchmark drivers, a run here is **one** call to a three-stage
pipeline, so an interrupted run restarts from its assimilation stage.

Restrict a re-run to specific points with `ONLY="w2_locnone_obs30_inflow ..."`.

## Concurrency

`NUM_LANES` (default 1) concurrent runs, each with its **own**
`paths.experiment_dir` — the uDALES scratch tree is the one piece of shared
mutable state a run has, and two runs sharing it corrupt each other's
namoptions and fielddumps. Each run additionally fans out `WORKERS` (default 8)
single-core ensemble members.

This box is DRAM-bandwidth-bound past ~4–8 concurrent members, so
`NUM_LANES * WORKERS` is the number to watch — and with `NUM_LANES > 1` wall
times are contended and **not** comparable between runs (`progress.tsv` records
start/end so co-residency can be reconstructed).

`JAX_PLATFORMS=cpu` is exported by `settings.sh`: the pipelines invoke the
runners under `pixi run -e cuda`, and without it every forkserver worker creates
its own CUDA context and a shared card runs out of device memory hours in.

## Cross-model runs (`pypalm` truth)

```bash
TRUTH_MODEL_LIST="pyudales pypalm" \
  bash job_scripts/local/experiments/run_esmda_experiments.sh
```

The truth becomes a PALM simulation while the ensemble stays uDALES, so the run
now contains model error as well as parameter/state error. Both mounts read the
same `domain.*` and the same STL, so the grid and geometry still match. Things
to know before launching one:

- **PALM must be built.** `conf/model/pypalm.yaml` ships `compile: false`; the
  binary comes from `install_palm.sh` at install time and does *not* need
  rebuilding when the grid changes.
- **`PALM_NCPU` must divide `domain.nx`** (40 for `xie_and_castro` → 1, 2, 4, 5,
  8, 10, 20, 40): PALM uses a slab decomposition and the inflow/outflow
  multigrid solver needs uniform subdomains. It applies to pypalm mounts only,
  and the truth is a single simulation — cores spent there do not compete with
  the ensemble's `WORKERS`.
- **PALM's grid constraints**: `nz >= 16`, even `nx`/`ny`. True for this case at
  its default grid; check if you override `domain.*` via `EXTRA_ARGS`.
- **Model-error parameters.** `PARAMS_TO_ESTIMATE` is the knob for letting the
  DA absorb the model discrepancy — `vertical_inflow_exponent` is the safe
  addition. Read the `sgs_constant` warning in `settings.sh` first: it is not
  the same physical quantity across backends, and on PALM setting it at all
  switches the closure.
- **PALM members diverge ~15% of the time** intrinsically (docs/pypalm.md §8).
  Here PALM only runs the truth — a single simulation with no resampling to
  hide behind — so a diverged truth fails the whole run. It is recorded in
  `progress.tsv` and the campaign continues; re-run to retry just that point.

## A shared truth artifact

By default every run simulates its own truth inline. Point `TRUTH_DIR` at a dir
holding `state.nc` + `params.nc` (as written by `run_forward_model.py`) to
assimilate a pre-simulated truth instead — with `TRUTH_START_TIME` to skip a
spin-up. A truth artifact carries the forcing *and* the backend it was
simulated with, so the campaigns refuse to start when `TRUTH_DIR` is set and
either `INFLOW_LIST` or `TRUTH_MODEL_LIST` has more than one value: run one
inflow setting and one truth model per truth artifact.

# pylbm — Reference

`pylbm` is the Python wrapper around Geir Evensen's Lattice Boltzmann Fortran
CFD solver. It implements the `BaseForwardModel` / `BaseEnsembleForwardModel`
interface so it is interchangeable with `pyudales` and `pypalm` inside ESMDA
and data-generation runs. The state output is cell-centred `(x, y, z)` with
velocity variables `u, v, w`; all coordinates are in physical units (metres,
m/s). No staggered grid.

Read [codebase_guide.md](codebase_guide.md) §3 for the base-class contracts and
§7 for cross-backend gotchas; this document focuses on pylbm-specific behaviour.

---

## 1. Fortran code — submodule bootstrap

[`libs/pylbm/src/pylbm/__init__.py`](../libs/pylbm/src/pylbm/__init__.py)
runs at import time and ensures the Fortran sources are present:

1. It reads `.gitmodules` to find the LBM submodule path (`libs/pylbm/LBM/`)
   and URL.
2. If the directory is empty or missing, it runs
   `git submodule update --init --recursive libs/pylbm/LBM`, falling back to
   `git clone` from the URL if the submodule command fails.
3. No network access at all → the submodule init silently fails and
   `LBM_PATH` stays as the (empty) on-disk path. Nothing raises; later
   steps fail when the binary is missing.

**Per-job isolation.** The Fortran build mutates its own source tree
(`mod_dimensions.F90`, generated Fortran, object files, the `boltzmann` binary).
Two concurrent builds on the shared submodule corrupt each other. For HPC / CI
use, set `PYLBM_LBM_PATH` to a private copy of the tree and `__init__.py` uses
that path directly, skipping all submodule logic.

The compiled binary lands at `LBM/bin/boltzmann` (not in the shared pixi
`bin/`), so isolating the LBM tree per run also isolates the binary.

---

## 2. Class structure

### `ForwardModel`

[`libs/pylbm/src/pylbm/forward_model.py`](../libs/pylbm/src/pylbm/forward_model.py)
— subclasses `BaseForwardModel`.

**Constructor** (`__init__`) does the following immediately, not deferred to
`compile()`:

- Calls [`stl_to_lbm_geometry`](#5-geometry) to voxelise the STL into a
  bathymetry file.
- Calls `set_experiment` ([`utils/mod_dimensions_utils.py`](../libs/pylbm/src/pylbm/utils/mod_dimensions_utils.py))
  to write `nx/ny/nz` into `mod_dimensions.F90`.
- Writes `uvel_shear.dat` if `profile_config` requests a non-uniform vertical
  profile (or deletes a stale one).

Key constructor parameters:

| Parameter | Default | Notes |
|---|---|---|
| `stl_path` | required | STL geometry file |
| `nx, ny, nz` | 120, 120, 8 | Grid cells; wired from `${domain.*}` in YAML |
| `bounds` | `((0,160),(0,160),(0,40))` | Physical domain extents in metres |
| `simulation_time` | 53.8 | Seconds of output to collect after spin-up |
| `output_frequency` | 0.0538 | Seconds between output snapshots |
| `spinup_time` | 0.0 | Warm-up seconds prepended (outputs discarded) |
| `cuda` | False | Use NVFORTRAN/CUDA build |
| `verbose` | True | `False` → `stderr=DEVNULL` (see §7) |
| `boundary_condition` | `"periodic"` | `"inflow_outflow"` for real cases |
| `profile_config` | None | Vertical shear profile dict, e.g. `{"type":"power_law","alpha":0.25}` |
| `results_dir` | None | `None` → in-memory mode; path → on-disk mode |

The default in [`conf/model/pylbm.yaml`](../conf/model/pylbm.yaml) sets
`cuda: true`, `verbose: false`, and `boundary_condition: inflow_outflow`.

#### `compile(compile=True)`

Triggered via the Hydra `prepare` step (see §8). Does:

1. Calls `compile_lbm` ([`utils/compile_utils.py`](../libs/pylbm/src/pylbm/utils/compile_utils.py))
   — GPU arch detection, makefile patching, `make`.
2. Wipes stale `seed_*.dat` / `seed_*.orig` files from `experiment_dir` so a
   rebuilt binary (which may use a different `RANDOM_SEED` size) does not crash
   on an old restart file format.
3. Runs the binary once with no arguments to generate `infile.in` if it does
   not already exist (`create_infile`).
4. Sets boundary condition, `tecout`, and `experiment` name in `infile.in`.

#### `run_single(state, params, sim_name)`

The public entry point (called by `BaseForwardModel.__call__`):

1. **Warm start** — if `state` is not `None`, spin-up is suppressed for this
   call, `_set_scaling_factors(params)` is called first to fix `C_u` for
   velocity-unit conversion, then `_prepare_warmstart(state)` writes a restart
   `.uf` file and records `nt0` for the run.
2. **Scaling factors** — `_set_scaling_factors(params)` writes `C_l`
   (= `min_cell_size`) and `C_u` (= `velocity_magnitude * 15`, or 75 for no
   params) into `infile.in`, then derives `nt0/nt1/iout` (timestep range and
   output interval). Guards against `nt1 > MAX_ITERATION` (see §7).
3. **Inflow settings** — `_apply_inflow_settings(params)` (see below).
4. **Output cleanup** — `_clean_output()` deletes all `out_*.nc` files in
   `output_dir` to prevent stale files from a prior run being collected.
5. **Run** — `self.run()` executes the `boltzmann` binary via `subprocess.run`
   with `check=True` (non-zero exit raises `CalledProcessError`). Stack size is
   raised to `unlimited` / `hard` before launch to handle large
   `nx*ny*nz` automatic arrays.
6. **Collect** — globs `out_0000_F<iter>.nc` in `(nt0, nt1]`, concatenates
   with `xarray.concat`, assigns physical coordinates, scales velocity from
   lattice units (`* C_u`), trims spin-up outputs, trims to `simulation_time /
   output_frequency` outputs, and assigns a seconds-based `time` coordinate.
7. **Prune restarts** — `remove_old_restart_files` keeps only the latest
   restart, preventing unbounded accumulation.

#### `_apply_inflow_settings(params)`

- Calls `resolve_profile_config(params, self.profile_config)` — if `params`
  contains `vertical_inflow_exponent`, overrides `alpha` in the profile and
  rewrites `uvel_shear.dat` per member.
- Calls `apply_sgs_setting(params, self.dirs)` — if `params` contains
  `sgs_constant`, writes `ivreman <smagorinsky>` to `infile.in` (the Fortran
  sets `const = 2.5 * smagorinsky**2` in `m_vreman.F90`).
- **Time-varying branch** (`is_time_varying_params` is True): writes
  `uvel_time.dat` (shifted onto the LBM's absolute clock via `nt0 * dt`),
  then calls `apply_inflow_settings` with the first-frame values as static
  fallback for `infile.in`.
- **Static branch**: removes any stale `uvel_time.dat`, calls
  `apply_inflow_settings` directly.

#### `save_results` / `_clean_output`

`save_results` delegates to `BaseForwardModel._save_results`.
`_clean_output` deletes all `out_*.nc` files from `output_dir`.

#### `disable_spinup()`

Sets `self.spinup_time = 0.0`. Called by `BaseRolloutForwardModel` after
window 0 when `spinup_first_step_only=True`.

### `EnsembleForwardModel`

[`libs/pylbm/src/pylbm/ensemble_forward_model.py`](../libs/pylbm/src/pylbm/ensemble_forward_model.py)
— subclasses `BaseEnsembleForwardModel`. The only override is
`_create_new_forward_model`, which delegates to
[`utils/forward_model_utils.create_new_forward_model`](../libs/pylbm/src/pylbm/utils/forward_model_utils.py):
it deep-copies the template `ForwardModel`, copies all files from the template's
`experiment_dir` into a new per-member directory (so `infile.in`,
`bathymetry_*.uf`, `uvel_shear.dat`, and restart files are cloned), and updates
`self.dirs` on the copy.

Parallel dispatch, failure policy, CPU pinning, and forkserver context are all
in `BaseEnsembleForwardModel` (see codebase_guide.md §3).

---

## 3. Compilation flow

[`utils/compile_utils.py`](../libs/pylbm/src/pylbm/utils/compile_utils.py)
handles the full build chain:

1. **Environment resolution** (`_resolve_build_environment`) — prefers the
   active Pixi environment if it has `include/netcdf.mod`; falls back to
   `delftblue`/`dev`/`default` envs in `.pixi/envs/` if not.
2. **GPU arch detection** (`_detect_gpu_compute_capability`) — queries
   `nvidia-smi` for compute capability (e.g. `"86"` for sm_86), patching the
   makefile's `-gpu=cc<N>` so the binary matches the host GPU. Override with
   `PYLBM_GPU_ARCH=<cc>`.
3. **CUDA + NetCDF** — NVFORTRAN cannot consume the conda-forge `netcdf.mod`
   (built with `gfortran`). `_ensure_cuda_netcdf_fortran` builds or reuses an
   NVHPC-compatible netcdf-fortran installation under
   `.pixi/envs/<env>/.nvhpc/netcdf-fortran/`. Override root with
   `NETCDF_FORTRAN_ROOT`.
4. **Make invocation** — always `make -B` (full rebuild); passes
   `CUDA=1` or `GFORTRAN=1`, `NETCDF=1`, `NCFDIR`, `BINDIR=LBM/bin`,
   `LIBDIR`. Compilation failure raises `RuntimeError`.

Compilation is gated by `cfg.model.compile` (a bool). The `prepare` step
is `pyurbanair.config.hydra_helpers.prepare_compile`, which calls
`forward_model.compile(compile=cfg.model.compile)`.

---

## 4. Runtime configuration — `infile.in` and the `Infile` editor

The Fortran binary reads all runtime settings from `infile.in`. The Python side
edits it with the [`Infile`](../libs/pylbm/src/pylbm/utils/infile_utils.py)
class, which parses the `"value(s)  ! key : description"` format line-by-line.
API: `Infile(path)`, `.get_value(key)`, `.set_value(key, value)`, `.write()`.
Keys are the first word after `!`; multi-token values (e.g. `uini, udir`) use
the comma-inclusive first token as the key (`uini,`).

Key fields set by the wrapper:

| `infile.in` key | Meaning |
|---|---|
| `experiment` | Experiment name (selects geometry case in `m_solid_objects_init.F90`) |
| `C_l` | Lattice length scale = `min_cell_size` (metres) |
| `C_u` | Lattice velocity scale = `velocity_magnitude * 15` (m/s) |
| `nt0`, `nt1` | Start and end timestep indices (encode the run window) |
| `iout` | Output every N timesteps |
| `iprt1` | Disables the every-iteration NetCDF dump (`0 <nt1+1> 1`); without this, warm-start runs (where the iteration counter already exceeds the trigger) produce ~20× more output files |
| `uini, udir` | Initial inflow speed (m/s) and direction (degrees) |
| `ibnd`, `jbnd` | x/y boundary conditions (0 = periodic, 1 = inflow-outflow) |
| `tecout` | Output format (3 = NetCDF) |
| `ivreman` | SGS model; second token is the Smagorinsky constant (see §7) |

**Output filenames.** Each saved snapshot is `output/out_0000_F<iter:09d>.nc`
where `iter` is the absolute LBM timestep. Files are collected by
`_get_output_files_for_current_run` which globs the output directory and filters
to `(nt0, nt1]`.

---

## 5. Geometry

### STL → bathymetry file

[`stl_to_lbm.py`](../libs/pylbm/src/pylbm/stl_to_lbm.py) is called from
`ForwardModel.__init__` on every instantiation (including per-member clones,
where the file is already present in the cloned directory).

The conversion pipeline (`stl_to_lbm_geometry`):

1. **Voxelise** (`compute_solid_occupancy`) — loads the STL with `trimesh`,
   handles `Scene` (multi-part) files by concatenating all meshes. Casts one
   vertical ray downward per `(i, j)` grid column; the highest z-intersection
   is that column's roof height. Cells whose centre is at or below the roof are
   marked solid. This handles open-bottomed extrusions and concave footprints
   correctly (no `mesh.contains`, which is unreliable for non-watertight meshes).
2. **Write bathymetry** (`write_bathymetry_file`) — writes a Fortran
   unformatted binary `bathymetry_<experiment>.uf` in `experiment_dir`:
   record 1 is `(nx, ny, nz)` as `int32`; record 2 is the `(nx+2, ny+2, nz+2)`
   padded blanking array (with ghost shell) in Fortran column-major order.
3. **Wire the Fortran** (`update_solid_objects_init`) — edits
   `m_solid_objects_init.F90` in-place to add a
   `case('<experiment>')` branch that calls
   `read_bathymetry(blanking_global, '<experiment>')` at run time. Removes any
   stale `use m_<experiment>` from the old generated-module approach.
4. **Purge stale generated module** — deletes `m_<experiment>.F90` (the legacy
   per-gridpoint Fortran that grew too large to compile for city-block domains).

> **Caveat.** The STL-to-LBM conversion is functional but not fully validated.
> For production runs the Barcelona geometry was confirmed correct by visual
> inspection and cross-checking against the pyudales run (which uses the same
> STL), but STL voxelisation edge cases (thin walls, boundary straddling) have
> not been systematically tested.

### Bathymetry vs. generated F90

The current code path is always the `.uf` binary bathymetry (written at init
time and loaded at run time by `m_read_bathymetry`). The old generated-module
approach (`generate_fortran_code`, `process_stl_to_fortran`) is still present in
`stl_to_lbm.py` but is not called by `stl_to_lbm_geometry`.

---

## 6. The `utils/` subpackage

[`libs/pylbm/src/pylbm/utils/`](../libs/pylbm/src/pylbm/utils/)

| Module | Purpose |
|---|---|
| [`dir_utils.py`](../libs/pylbm/src/pylbm/utils/dir_utils.py) | `DirectoryPaths` dataclass; `get_lbm_directory_paths` factory that reads `LBM_PATH` and constructs all experiment paths |
| [`infile_utils.py`](../libs/pylbm/src/pylbm/utils/infile_utils.py) | `Infile` editor; `create_infile` (runs binary to generate the file); `_augment_runtime_library_paths` (prepends NVHPC/conda libs to `LD_LIBRARY_PATH` at launch) |
| [`params_utils.py`](../libs/pylbm/src/pylbm/utils/params_utils.py) | `is_time_varying_params`, `apply_inflow_settings` (writes `uini, udir` to `infile.in`), `write_uvel_time_file`, `write_uvel_shear_file`, `apply_sgs_setting`, `resolve_profile_config`, and `remove_*` helpers |
| [`compile_utils.py`](../libs/pylbm/src/pylbm/utils/compile_utils.py) | `compile_lbm`; GPU arch detection; CUDA netcdf-fortran bootstrap |
| [`mod_dimensions_utils.py`](../libs/pylbm/src/pylbm/utils/mod_dimensions_utils.py) | Parses and writes `mod_dimensions.F90`; `set_experiment` adds/updates the `nx/ny/nz` parameter block for the named experiment |
| [`forward_model_utils.py`](../libs/pylbm/src/pylbm/utils/forward_model_utils.py) | `create_new_forward_model` — deep-copies template model into a per-member directory (used by `EnsembleForwardModel._create_new_forward_model`) |
| [`warm_start_utils.py`](../libs/pylbm/src/pylbm/utils/warm_start_utils.py) | `write_restart_file_from_xarray` (writes `.uf` restart from xarray state); `identify_latest_restart_iteration` (mtime-based, not max-iter); `remove_old_restart_files` |
| [`state_utils.py`](../libs/pylbm/src/pylbm/utils/state_utils.py) | `scale_velocity_to_physical` / `scale_velocity_to_lattice` (multiply by `C_u`) |
| [`vertical_profile.py`](../libs/pylbm/src/pylbm/utils/vertical_profile.py) | `build_profile_shape` — `uniform` and `power_law` profiles; mirrors `pyudales.utils.vertical_profile` for cross-backend consistency |
| [`makefile_utils.py`](../libs/pylbm/src/pylbm/utils/makefile_utils.py) | `Makefile` — edits `HOME`, `NCFDIR`, and `-gpu=cc<N>` in the LBM makefile |
| [`environment_utils.py`](../libs/pylbm/src/pylbm/utils/environment_utils.py) | `identify_environment` — resolves active Pixi/conda environment path |

---

## 7. Parameters

### Inflow parameters

`inflow_angle` and `velocity_magnitude` are the two primary inflow parameters.
They are consumed in `apply_inflow_settings`:

```python
# infile.in "uini, udir" line:
infile.set_value("uini,", f"{velocity_magnitude:.1f} {inflow_angle:.1f}")
```

The LBM's `udir` convention matches `pyudales`' `inflow_angle` (degrees
measured from +x, counter-clockwise), so no negation is applied.

For **time-varying params** (`is_time_varying_params` is True),
`write_uvel_time_file` writes `uvel_time.dat` — one row per control point:
`time_seconds  velocity_m_s  direction_degrees`. Times are window-relative,
then shifted by `nt0 * dt` onto the LBM's absolute clock so the schedule
survives warm-start rollouts. The static `infile.in` values are set to the
first frame as a fallback.

For **static params**, any stale `uvel_time.dat` is removed.

### `uvel_shear.dat` — vertical inflow profile

Written by `write_uvel_shear_file` when `profile_config["type"] != "uniform"`.
Format: `k  z_metres  shape_value` per level. The Fortran renormalises by the
top cell value so `uini` is the speed at the top cell centre. The default
config uses `power_law` with `alpha=0.25`.

The shear profile is per-member when `params` contains `vertical_inflow_exponent`
(the ESMDA model-error knob): `resolve_profile_config` overrides `alpha` and
`write_uvel_shear_file` is called inside `_apply_inflow_settings` for each
member run.

### SGS constant — `ivreman smagor` in `infile.in`

`apply_sgs_setting` writes the `ivreman` key as `"1 <smagorinsky>"`. The
Fortran computes `const = 2.5 * smagorinsky**2` in `m_vreman.F90`. This is a
no-op when `sgs_constant` is absent from `params`. Note: the LBM Smagorinsky
constant is dimensionless and physically distinct from pypalm's `km_constant`
(m²/s) — prior ranges in YAML should not be shared between backends.

### `C_u` scaling

`C_u = int(velocity_magnitude * 15)`. All output velocities in the NetCDF are
in lattice units; `scale_velocity_to_physical` multiplies by `C_u` to recover
m/s. When `params` is `None`, `C_u` defaults to 75.

---

## 8. Configuration

[`conf/model/pylbm.yaml`](../conf/model/pylbm.yaml):

```yaml
name: pylbm
solver_name: pylbm
compile: true

forward_model:
  _target_: pylbm.forward_model.ForwardModel
  stl_path: ${geometry.stl_path}
  temp_dir: ${paths.experiment_dir}
  experiment_name: runcase
  cuda: true
  verbose: false
  boundary_condition: inflow_outflow
  profile_config: {type: power_law, alpha: 0.25}
  nx: ${domain.nx}
  ny: ${domain.ny}
  nz: ${domain.nz}
  bounds: ${domain.bounds}
  simulation_time: ${time.simulation_time}
  output_frequency: ${time.output_frequency}
  spinup_time: ${time.spinup_time}

prepare:
  _target_: pyurbanair.config.hydra_helpers.prepare_compile
  compile: ${..compile}

ensemble_model:
  _target_: pylbm.ensemble_forward_model.EnsembleForwardModel
  ensemble_size: ${ensemble.ensemble_size}
  num_parallel_processes: ${ensemble.num_parallel_processes}
  num_cpus_per_process: ${ensemble.num_cpus_per_process}
  failure: ${ensemble.failure}
```

`solver_name: pylbm` selects the regular-grid dim mapping in
`ObservationOperator` (no staggered axes — `x, y, z` for all velocity
components). Mount with `model@assim_model=pylbm` or
`model@truth_model=pylbm`.

CLI overrides that matter in practice:

```bash
# Surface CUDA failures
model.forward_model.verbose=true

# Disable CUDA (e.g. for CPU-only debugging)
model.forward_model.cuda=false

# Skip recompile (binary already up to date)
model.compile=false
```

---

## 9. Known gotchas

### Silent CUDA failures (`verbose=false`)

The default config has `verbose: false`, which routes both `stdout` and `stderr`
to `subprocess.DEVNULL`. A crashed or misconfigured binary produces no visible
error; the ensemble runner sees a `CalledProcessError` (non-zero exit) but the
cause is invisible. **To see Fortran error messages, add
`model.forward_model.verbose=true` to the CLI.** This is the first thing to try
when LBM produces no output or all members fail.

### Iteration filename field width — i9.9 overflow

LBM output and restart files encode the iteration in a 9-digit fixed-width
Fortran format field. `_set_scaling_factors` guards against this:

```python
MAX_ITERATION = 999_999_999  # forward_model.py top-level constant
if nt1 > MAX_ITERATION:
    raise ValueError(...)
```

Rollout runs accumulate large `nt0` values across windows. If the total
`nt0 + total_timesteps` exceeds 999,999,999 the output filenames overflow to
`out_0000_F*********.nc` (literal asterisks), the glob does not match them, and
`_get_output_files_for_current_run` raises `FileNotFoundError` or the concat
fails with an `AlignmentError` across members. Reduce `simulation_time`,
`output_frequency`, or the number of rollout windows, or widen the field in the
Fortran sources.

### Domain-height SIGFPE

The Boltzmann solver diverges to SIGFPE when the STL building heights exceed the
domain's z extent. Verify that `bounds[2][1]` (domain top in metres) is greater
than the tallest structure in the STL. For Barcelona, the merged-ground
elevation from the STL adds to building heights, so the effective height is
higher than the building-only measurement.

### In-memory ensemble OOM at large grid sizes

`EnsembleForwardModel` concatenates all member states in memory by default.
For ensembles of ~96 members at grid sizes ≥ 75³ cells, this exhausts DRAM.
Fix: set `run.ensemble_save_on_disk=true` (or `results_dir` on the ensemble
model) so per-member files are written and read back individually. At 100³ the
run remains disk-bound — the per-member file I/O becomes the bottleneck.

### Stale seed files after recompile

After a `make -B` rebuild, `compile()` wipes all `seed_*.dat` and
`seed_*.orig` files from `experiment_dir`. If these are present from a binary
built with a different `RANDOM_SEED` size, the new binary fails on startup with
a Fortran I/O read-past-end error. The wipe in `compile()` prevents this, but
it only runs when `cfg.model.compile=true` — a manual binary swap without
recompiling through the Python layer can leave stale seeds.

### Warm-start C_u sequencing

In `run_single`, when a warm-start `state` is provided, `_set_scaling_factors`
is called *before* `_prepare_warmstart` to fix the current window's `C_u`. The
restart file is then written in the correct lattice units for this window's
velocity scale. If this order is reversed, the velocity field jumps when
`velocity_magnitude` changes between windows.

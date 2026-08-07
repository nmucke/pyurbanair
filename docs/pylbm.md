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
3. `_verify_submodule_pin` then compares the submodule's `HEAD` against the
   commit the parent repo records, and **logs which commit is about to be
   built**. On mismatch:
   - working copy clean → it is checked out onto the recorded commit;
   - working copy dirty → left completely alone (it may hold real WIP) and
     reported as a loud `WARNING` with the exact remediation command.
   This exists because a bare `git clone` fallback lands on the default-branch
   tip rather than the pin, and a manual `git checkout` inside the submodule is
   invisible to the parent repo. See §9 for why a wrong commit surfaces as an
   unrelated-looking `make` error.
4. No network access at all → the submodule init silently fails and
   `LBM_PATH` stays as the (empty) on-disk path. Nothing raises; later
   steps fail when the binary is missing.

**The submodule is read-only; every run builds in its own tree.** The Fortran
build mutates its source tree — `mod_dimensions.F90` bakes the grid in at compile
time, `m_solid_objects_init.F90` gets the geometry case wired in, and the makefile
regenerates `depends.file` / `source.files` — and all of those are files *tracked*
by the submodule. So `dir_utils` no longer points the build at the submodule:
[`utils/build_tree_utils.py`](../libs/pylbm/src/pylbm/utils/build_tree_utils.py)
mirrors `src/` and the `bin/` helper scripts into
`<paths.experiment_dir>/lbm_build/` and the build happens there.

- The submodule stays byte-identical to its checked-out commit, so `git status`
  no longer shows a permanently dirty `M libs/pylbm/LBM` gitlink (which had been
  swept into unrelated commits and broke CI).
- The mirror is incremental — only files differing in size or mtime are copied —
  and prunes sources that vanished upstream, which matters because the makefile
  rebuilds `source.files` from `ls *.F90` and would otherwise compile a stale
  module back in.
- Override the location with `PYLBM_BUILD_ROOT` (e.g. node-local scratch).

**Per-job isolation.** `PYLBM_LBM_PATH` keeps its original meaning: point it at a
private copy of the whole LBM tree and `__init__.py` uses that path directly,
skipping submodule discovery *and* the mirroring above (the caller owns that tree
and it is built in place).

The compiled binary lands at `<build tree>/bin/boltzmann` (not in the shared pixi
`bin/`), alongside a `.pylbm_build_stamp.json` recording the experiment, the
netcdf/cuda mode, and hashes of the two compiled-in sources. With
`model.compile=false`, `ForwardModel._verify_prebuilt_binary` checks that stamp
and raises rather than reusing a binary built for a different grid or geometry —
which does not fail loudly, it just produces wrong-shaped or all-NaN output.

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
| `cuda` | `"auto"` | `"auto"` → CUDA if NVHPC is installed, else gfortran; `True` requires CUDA; `False` forces gfortran |
| `verbose` | True | `False` → `stderr=DEVNULL` (see §7) |
| `boundary_condition` | `"periodic"` | `"inflow_outflow"` for real cases |
| `profile_config` | None | Vertical shear profile dict, e.g. `{"type":"power_law","alpha":0.25}` |
| `inlet_turbulence` | None | Inflow-turbulence forcing dict, e.g. `{"enabled":True,"amplitude":5e-5,"update_interval":100}` (see §7) |
| `results_dir` | None | `None` → in-memory mode; path → on-disk mode |

The default in [`conf/model/pylbm.yaml`](../conf/model/pylbm.yaml) sets
`cuda: auto`, `verbose: false`, and `boundary_condition: inflow_outflow`.

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
5. Applies `inlet_turbulence` (`apply_inlet_turbulence`, §7) — a no-op unless
   the knob is present *and* enabled.

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

> **One external caller drives these steps itself.**
> [`scripts/esmda/run_probe_series.py`](../scripts/esmda/run_probe_series.py)
> (the high-rate probe re-runs behind the Welch spectrum / figure S4) repeats
> `run_single`'s launch sequence — `_set_scaling_factors` → `_prepare_warmstart`
> → `_set_scaling_factors` → `_apply_inflow_settings` → `_clean_output` →
> `run()` — and replaces only its *collection* step: at a ~1 s cadence one
> window's snapshots are tens of GB, so each file is reduced to the probe points
> and unlinked instead of being concatenated into one Dataset. It also keeps
> `spinup_time` on a warm start (which `run_single` zeroes) to trim the restart's
> transient. Keep that sequence and the `out_0000_F<iter>.nc` layout in mind when
> refactoring `run_single`.

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

0. **CUDA mode resolution** (`resolve_cuda`) — turns the configured `cuda`
   setting into a concrete toolchain. `"auto"` (the shipped default) builds with
   CUDA where `find_nvfortran` locates an NVHPC install under `<env>/.nvhpc` and
   falls back to gfortran otherwise, logging which it picked; `true` *requires*
   CUDA and raises without it (a GPU batch job should fail rather than silently
   drop to a ~100× slower CPU build); `false` forces gfortran. A bad string is
   rejected in `ForwardModel.__init__` via `validate_cuda_setting`.
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
4. **Dependency priming** — a throwaway `make depends.file` runs *before* the
   real build. The makefile derives `source.files` (`ls *.F90`) and
   `depends.file` (a `use`-statement crawl in `bin/mkdepend.pl`) by scanning
   `src/`, and its `depends.file` rule deliberately **fails** whenever the
   result changed (`>>> Dependencies updated — please rerun make`). Since this
   wrapper rewrites `m_solid_objects_init.F90`'s `use` statements per experiment,
   that fires on the first build in any fresh tree; priming absorbs the intended
   failure so the real build starts with both files up to date.
5. **Make invocation** — always `make -B` (full rebuild); passes
   `CUDA=1` or `GFORTRAN=1`, `NETCDF=1`, `NCFDIR`, `BINDIR=<build tree>/bin`,
   `LIBDIR`. Compilation failure raises `RuntimeError`.
6. **Build stamp** — on success, `write_build_stamp` records the experiment, the
   cuda/netcdf mode, and hashes of the compiled-in sources next to the binary
   (see §1).

`Makefile.set_path` (`makefile_utils.py`) is idempotent: it scans the whole file
rather than stopping at the first blank line, consumes the line's own newline
when substituting, and collapses duplicate assignments. Before that it appended a
fresh `NCFDIR` line and an extra blank line on *every* construction — the
checked-out submodule makefile had grown from 240 to 1518 lines that way.

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

**The file is positional.** `m_readinfile.F90` reads the value lines strictly in
order, so a line that is dropped, split, or reordered shifts every subsequent
read and is misparsed silently rather than raising. Several lines are consumed by
a *single* list-directed `read` and therefore carry several values that must be
written together (`iprt1 iprt2 x`, `ivreman smagor`, `lturb amp nrtu`). Use
`.get_value_tokens(key)` / `.set_value_tokens(key, [...])` for those — they split
and rejoin the whole value field, leaving the `! key : description` comment and
the surrounding lines untouched.

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
| `lturb` | Inflow turbulence: `"<on> <amplitude> <update_interval>"` — all three read at once (see §7) |

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
| [`inlet_turbulence_utils.py`](../libs/pylbm/src/pylbm/utils/inlet_turbulence_utils.py) | `validate_inlet_turbulence` (config checks, incl. the periodic guard), `apply_inlet_turbulence` (writes the 3-value `lturb` line) |
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

**Where `sgs_constant` comes from.** Two sources, in precedence order:

1. `sgs_constant` in the params Dataset (from the `conf/params/*.yaml` sampler) —
   used when ESMDA estimates or pins it.
2. `forward_model.sgs_constant` in the backend's own `conf/model/*.yaml` — the
   per-backend default.

Absent from both is a strict no-op: the solver's own closure/template value
stands. The per-backend default exists because the three backends' `sgs_constant`
are **different physical quantities** (uDALES/pylbm take a dimensionless
Smagorinsky-family constant; PALM takes an eddy diffusivity in m²/s), so a single
value in the shared params sampler cannot be correct for all three at once.


### Inflow turbulence — `lturb amp nrtu` in `infile.in`

The Fortran ships a full inflow-turbulence subsystem
(`m_inflow_turbulence_{init,compute,apply,update,forcing}.F90`) that
superimposes smooth pseudo-random perturbations on the inlet plane. It is not a
per-member ESMDA parameter — it is a static model setting, applied once in
`compile()` and inherited by ensemble members when `experiment_dir` is cloned.

```yaml
inlet_turbulence:
  enabled: false
  amplitude: 5.0e-05      # turbulence_ampl
  update_interval: 100    # nrturb — timesteps between turbulence updates
```

The three values are read by one Fortran statement
(`read(10,*) inflowturbulence, turbulence_ampl, nrturb`), so
`apply_inlet_turbulence` rewrites the whole `lturb` line via
`Infile.set_value_tokens`. Semantics:

- **Absent (`None`) or `enabled: false` → strict no-op.** The line the solver
  generated for itself is left byte-identical, so default runs are unchanged.
- Values omitted while enabled fall back to whatever is already in `infile.in`
  rather than to hardcoded defaults. The solver's own template (`m_mkinfile.F90`)
  writes ` F 0.00005  100`, which is where the YAML defaults come from.
- `update_interval` must be ≥ 1: `inflow_turbulence_init` aborts on `nrturb <= 0`
  and `main.F90` uses it as `mod(it, nrturb)`. It also sizes the precomputed
  `uu/vv/ww/rr(ny,nz,0:nrturb)` buffers, so large values cost memory.
- **Requires `boundary_condition: inflow_outflow`.** `main.F90` gates the
  turbulence refresh on `ibnd == 1`, so `enabled: true` with `periodic` raises a
  `ValueError` at construction instead of silently doing nothing.

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
  cuda: auto
  verbose: false
  boundary_condition: inflow_outflow
  profile_config: {type: power_law, alpha: 0.25}
  inlet_turbulence: {enabled: false, amplitude: 5.0e-05, update_interval: 100}
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

# Turn on inflow turbulence (needs boundary_condition=inflow_outflow)
model.forward_model.inlet_turbulence.enabled=true
model.forward_model.inlet_turbulence.amplitude=2.0e-04
model.forward_model.inlet_turbulence.update_interval=50
```

---

## 9. Known gotchas

### `No rule to make target 'm_read_bathymetry.o'` — a submodule at the wrong commit

The makefile **regenerates `depends.file` at build time** from `bin/mkdepend.pl`,
which scans `src/*.F90` for `use` statements — the checked-in `depends.file` is
not authoritative. `stl_to_lbm.update_solid_objects_init` injects
`use m_read_bathymetry` into `m_solid_objects_init.F90` on every
`ForwardModel` construction, so the generated dependency
`m_solid_objects_init.o: m_read_bathymetry.o` always exists. If the submodule is
checked out at a commit *predating* `m_read_bathymetry.F90` (added upstream in
`2635d44`), make has no rule for that object and dies — with an error that names
a file nobody edited and that `git grep` cannot even find at that commit.

The tell is that the error is about a *missing source*, not a compile failure.
Check `git -C libs/pylbm/LBM rev-parse HEAD` against
`git rev-parse HEAD:libs/pylbm/LBM`; `_verify_submodule_pin` now logs both at
import time and reconciles them when it safely can (§1).

### Stale `boltzmann` binaries produce silent garbage

`mod_dimensions.F90` is compiled in, so a binary built for another grid does not
error — it returns wrong-shaped or all-NaN output. Builds are per-run and out of
tree (§1) and `make -B` always rebuilds, so this cannot happen with
`model.compile=true`; with `model.compile=false`, `_verify_prebuilt_binary`
compares the build stamp and raises instead of running.

### Silent CUDA failures (`verbose=false`)

The default config has `verbose: false`, which routes both `stdout` and `stderr`
to `subprocess.DEVNULL`. A crashed or misconfigured binary produces no visible
error; the ensemble runner sees a `CalledProcessError` (non-zero exit) but the
cause is invisible. **To see Fortran error messages, add
`model.forward_model.verbose=true` to the CLI.** This is the first thing to try
when LBM produces no output or all members fail.

### Restart / output filename width: i6.6, on both sides

Every LBM filename encodes its iteration in a fixed-width field. The Fortran
declares `character(len=6) cit` and formats it with `i6.6`
(`m_readrestart.F90`, `m_saverestart.F90`, `m_save_uvw.F90`, `m_diag.F90`), builds
the name it opens from that, and calls `stop` when the file is absent. Python
writes those same files, so it has to spell them identically. One constant owns
this:

```python
# pylbm/utils/warm_start_utils.py
ITERATION_FIELD_WIDTH = 6
MAX_ITERATION = 10**ITERATION_FIELD_WIDTH - 1          # 999_999
restart_file_name(iteration, prefix="restart", tile="0000")
```

`tests/test_pylbm_restart_filenames.py` parses the width out of the Fortran
sources and fails if the two ever disagree, so a submodule bump that widens the
field is caught there rather than in a silently wrong run.

**This was broken until 2026-08-07 and is worth understanding.** Python carried a
9-digit field (`MAX_ITERATION = 999_999_999`, `:09d`) introduced for a newer LBM;
commit `68d3aa4` moved the submodule pin back to a commit that never had it, and
the two sides drifted apart with nothing to catch it. The consequences were all
silent:

- **The warm start was discarded.** Python wrote
  `restart_0000_000000696.uf`; the solver opened `restart_0000_000696.uf` — its
  own restart from the previous window, sitting in the same directory at the same
  iteration. The run continued and looked healthy. For a state-bearing smoother
  that means the Kalman state update never reached the solver.
- **When no same-iteration file existed**, `readrestart` printed
  `restart file does not exist: ...` and called `stop`, which exits **0** — so the
  wrapper saw no `CalledProcessError`, only an empty output dir or a truncated
  member. (That is the other half of the "truncated member exits 0" failure.)
- **The template lookup missed too**, so the restart Python wrote was built from
  a pure-equilibrium distribution — see the next gotcha.

Output *collection* was never affected: `_get_output_files_for_current_run` globs
`out_0000_F(\d+)` and is width-agnostic. Only the exact-name fallback used the
wrong width.

The ceiling still matters after the fix. `nt0` accumulates across warm starts, so
a long rollout reaches 999,999 even when no single window is near it;
`_set_scaling_factors` raises rather than letting the name overflow to
`out_0000_F******.nc`, which matches no glob and no restart pattern (so it could
never be read *or* pruned). Start from a clean experiment dir, shorten the run,
or widen `i6.6` in the Fortran **and** `ITERATION_FIELD_WIDTH` together.

### Warm starts are built from a pure-equilibrium distribution

**Open defect, verified 2026-08-07.** `_try_load_restart_distribution` reads the
restart template with
`FortranFile.read_record(np.int32, np.int32, np.int32, np.int32, np.float32)`.
Given a list of scalar dtypes, scipy treats them as one repeating 20-byte
compound and requires the record length to be a multiple of it. An LBM restart is
4 `int32` followed by `27*(nx+2)*(ny+2)*(nz+2)` `float32`, which is not — so the
call always raises:

```
ValueError: Size obtained (313648) is not a multiple of the dtypes given (20).
```

The exception used to hit a bare `except Exception: return None`, so the template
was silently treated as absent on **every** call and
`write_restart_file_from_xarray` always fell back to a pure-equilibrium restart —
losing the non-equilibrium and ghost-cell content the template exists to carry.
The macroscopic fields (rho, u, v, w) it writes are still correct, which is why
this was invisible; the filename-width bug above hid it further, because the
template lookup was also looking under the wrong name.

The fallback is now logged rather than swallowed. Reading it correctly needs
shaped dtypes sized from the grid, which would enable a code path that has never
executed in production and changes the field every pylbm warm start begins from —
so it wants its own change with its own stability check, not a drive-by fix.

### A truncated run exits 0, and is now caught

The LBM's error paths call Fortran `stop`, which exits **0**, so
`subprocess.run(check=True)` returns cleanly and the wrapper is left holding a
partial run. The collector still finds files and the trims in `run_single` only
ever *shorten*, so a short member used to pass straight through and surface
windows later as a broadcast/`AlignmentError` at ensemble assembly — nowhere near
the member that caused it.

`run_single` now verifies the frame count *before* loading or trimming and raises
`pyurbanair.base_ensemble_forward_model.ForwardModelRunFailure`, which the
ensemble runner treats exactly like a `CalledProcessError`: under
`resample_from_successes` the member is cloned from a survivor, under `raise` it
aborts. A single non-ensemble run still fails loudly.

The expected count mirrors `m_diag.F90`'s dump rule rather than
`simulation_time / output_frequency` — those agree only on a cold start:

```
expected = (nt1 // iout - nt0 // iout) + (0 if nt1 % iout == 0 else 1)
```

The trailing term is the **warm-start** case. `nt1 - nt0` is always a multiple of
`iout`, but `nt0` is the previous window's final iteration and
`iout = C_l/C_u/output_frequency` moves with each member's inflow speed, so
`nt0 % iout != 0` is normal from window 1 onwards and the run legitimately ends
one off-grid frame long. A rule derived from the window length alone would flag
every healthy warm start as truncated.

Only a shortfall is an error; a surplus is trimmed as before. When a member does
trip this, rerun with `model.forward_model.verbose=true` to see the solver's own
`stop` message — it is the only place the reason appears.

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

### `infile.in` edits fail silently, not loudly

`infile.in` is positional, not a namelist: `m_readinfile.F90` walks the value
lines in order. If an edit inserts, removes, or splits a line, every later read
lands on the wrong line and the solver runs happily with garbage settings — the
`err=100` branch only fires when a line is not even type-compatible. When adding
a knob, always write a whole line at once (`Infile.set_value_tokens` for the
multi-value ones) and verify against the solver's own echo: `m_readinfile.F90`
prints every parsed value at startup, so a run with
`model.forward_model.verbose=true` shows exactly what the Fortran understood
(e.g. `inflowturbulence  =        T    0.20000E-03     50`). Unit tests on the
generated text are not sufficient on their own.

Note also that `_set_scaling_factors` writes `uini` while the generated file's
key is `uini,` (the comma is part of the first token after `!`). `Infile`
therefore appends a separate trailing `uini` line instead of editing the inflow
line; it lands past everything `m_readinfile.F90` reads, so it is harmless, but
do not copy the pattern — `apply_inflow_settings` uses the correct `uini,` key.

### Warm-start C_u sequencing

In `run_single`, when a warm-start `state` is provided, `_set_scaling_factors`
is called *before* `_prepare_warmstart` to fix the current window's `C_u`. The
restart file is then written in the correct lattice units for this window's
velocity scale. If this order is reversed, the velocity field jumps when
`velocity_magnitude` changes between windows.

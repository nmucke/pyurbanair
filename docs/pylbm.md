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
netcdf/cuda mode, `openmp: true` for an OpenMP build (the key is absent for a
serial one), and hashes of the two compiled-in sources. With
`model.compile=false`, `ForwardModel._verify_prebuilt_binary` checks that stamp
and raises rather than reusing a binary built for a different grid or geometry —
which does not fail loudly, it just produces wrong-shaped or all-NaN output. It
also raises when the build's OpenMP mode does not match `ncpu` (serial for
`ncpu: 1`, OpenMP for `ncpu > 1`; a CUDA build is never OpenMP).

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
| `ncpu` | 1 | OpenMP threads. `> 1` builds the gfortran binary with `MP=1` and runs it with `OMP_NUM_THREADS=ncpu`; `1` is the serial build, launched exactly as before. A CUDA build ignores it with a warning (§3) |
| `verbose` | True | `False` → `stderr=DEVNULL` (see §7) |
| `boundary_condition` | `"periodic"` | `"inflow_outflow"` for real cases |
| `outflow_buffer` | None | `inflow_outflow` only: metres of empty ground added past `bounds`' xmax; states are cropped back to `bounds`. See §5 "Outflow buffer" |
| `ground_boundary` | None | `"no_slip"` / `"free_slip"`: the domain bottom (`kbnd` 12 / 22, the lid free-slip either way); None leaves the solver's 22. See §5 "Ground and lid" |
| `profile_config` | None | Vertical shear profile dict, e.g. `{"type":"power_law","alpha":0.25}` |
| `inlet_turbulence` | None | Inflow-turbulence dict, e.g. `{"enabled":True,"method":"digital_filter","seed":1}` or `{"enabled":True,"amplitude":5e-5,"update_interval":100}` (`random_forcing`, the default method); see §7 |
| `results_dir` | None | `None` → in-memory mode; path → on-disk mode |

The default in [`configs/model/pylbm.yaml`](../configs/model/pylbm.yaml) sets
`cuda: auto`, `verbose: false`, `boundary_condition: inflow_outflow` and
`ground_boundary: no_slip`.

#### `compile(compile=True)`

Triggered via the Hydra `prepare` step (see §8). Does:

1. Calls `compile_lbm` ([`utils/compile_utils.py`](../libs/pylbm/src/pylbm/utils/compile_utils.py))
   — GPU arch detection, makefile patching, `make`.
2. Wipes stale `seed_*.dat` / `seed_*.orig` files from `experiment_dir` so a
   rebuilt binary (which may use a different `RANDOM_SEED` size) does not crash
   on an old restart file format.
3. Runs the binary once with no arguments to generate `infile.in` if it does
   not already exist (`create_infile`).
4. Sets the boundary conditions (`ibnd`, `jbnd`, and `kbnd` from
   `ground_boundary`), `tecout`, and `experiment` name in `infile.in`.
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
3. **Inflow settings** — `_apply_inflow_settings(params)` (see below). With
   `inlet_turbulence.method: digital_filter` the window is also turned into a
   restart run on the `nrturb` grid and its forcing buffers are written
   (`_prepare_restart_cold_start`, `_apply_digital_filter`; §7), and the
   member's clock advances after a successful run. A laminar cold start with
   `ground_boundary: no_slip` is staged the same way (restart from the state
   the solver's cold start builds), because only a restart sets the ground's
   ghost corners (§5 "Ground and lid").
4. **Output cleanup** — `_clean_output()` deletes all `out_*.nc` files in
   `output_dir` to prevent stale files from a prior run being collected.
5. **Run** — `self.run()` executes the `boltzmann` binary through the shared
   `pyurbanair.utils.solver_process.run_solver` (non-zero exit raises
   `CalledProcessError`; the process dies with its Python owner). Stack size is
   raised to `unlimited` / `hard` before launch to handle large
   `nx*ny*nz` automatic arrays.
6. **Collect** — globs `out_0000_F<iter>.nc` in `(nt0, nt1]`, concatenates
   with `xarray.concat`, assigns physical coordinates, crops the outflow
   buffer (§5), scales velocity from
   lattice units (`* C_u`), trims spin-up outputs, trims to `simulation_time /
   output_frequency` outputs, and assigns a seconds-based `time` coordinate
   (`dt, 2·dt, …, simulation_time`).
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

Sets `self.spinup_time = 0.0`. Its only caller is the neural surrogate's
`disable_spinup`, which forwards to its `spinup_forward_model`; warm-start
windows already skip spin-up inside `run_single`.

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
in `BaseEnsembleForwardModel` (see codebase_guide.md §3). With `ncpu > 1` each
member runs `ncpu` OpenMP threads, so set `ensemble.num_cpus_per_process` to at
least `ncpu`. That is advice, not checked anywhere: on Linux each worker is pinned
to `num_cpus_per_process` cores, and more threads than pinned cores just
oversubscribe them. macOS has no CPU pinning, so there the OS schedules all
`num_parallel_processes * ncpu` threads freely.

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
   rejected in `ForwardModel.__init__` via `validate_cuda_setting`. The SDK is
   installed with `pixi run -e cuda install-nvhpc`, which then runs
   `prune-nvhpc` to delete the bundled CUDA toolkits nvfortran does not pick for
   the host's GPU driver (~12 GB; a CUDA 12 driver keeps 12.9, R580+ keeps 13.1).
   Set `NVHPC_PRUNE=0` for an install shared across hosts with different drivers.
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
   `LIBDIR`, and `MP=1` (`-fopenmp -DOPEN_MP`) when `ncpu > 1` — on both the
   priming and the real invocation. OpenMP is CPU-only: when the build resolves
   to CUDA (e.g. `cuda: auto` on a GPU node), `MP=1` is dropped with a warning,
   so one config runs on both kinds of host. The thread count is a run-time
   setting (`OMP_NUM_THREADS`), so changing `ncpu` between two values above 1
   needs no rebuild; going between 1 and >1 does. Compilation failure raises
   `RuntimeError` with the end of the build output.
6. **Build stamp** — on success, `write_build_stamp` records the experiment, the
   cuda/netcdf mode, `openmp: true` when built with `MP=1`, and hashes of the
   compiled-in sources next to the binary (see §1). A failed build writes no stamp, so the next compile retries it.

**Platforms.** The gfortran build runs on Linux and macOS (osx-arm64) with the
pixi env's compilers, FFTW and NetCDF. On macOS, `LIBDIR` also carries
`-B/usr/bin/` from the shared `pyurbanair.utils.toolchain.apple_linker_flags`,
so the link uses Apple's ld: conda's ld64 cannot read a current SDK's
`libSystem.tbd` (`unknown architecture arm64e.x1`, then missing `expf`,
`memcpy`). The CUDA build is Linux-only.

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
| `kbnd` | z boundary: first digit the ground (k = 1), second the lid (k = nz); 1 no-slip bounce-back, 2 free-slip (`ground_boundary`, §5) |
| `tecout` | Output format (3 = NetCDF) |
| `ivreman` | SGS model; second token is the Smagorinsky constant (see §7) |
| `lturb` | Inflow turbulence: `"<on> <amplitude> <nrturb>"` — all three read at once (see §7; `digital_filter` writes `T 1.0 <window steps>` per window) |

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

### Ground and lid

`ground_boundary` (`configs/model/pylbm.yaml`: `no_slip`) picks the domain
bottom; the lid is free-slip either way. It writes `kbnd` in `infile.in`:

| `ground_boundary` | `kbnd` | k = 1 (ground, z = 0) | k = nz (lid) |
|---|---|---|---|
| `no_slip` (default) | 12 | half-way bounce-back | free-slip |
| `free_slip` | 22 (the solver's own default) | free-slip | free-slip |
| `null` | infile unchanged | — | — |

What the solver does (`m_boundarycond.F90:175-205`,
`m_boundary_k_closed_kernel.F90`): every population leaving through the ghost
plane k = 0 comes back as its mirror, with all of `c` reversed for `opt = -1`
(bounce-back: no-slip, wall at z = 0) or only `c_z` for `opt = +1` (specular:
free-slip). The value returned is the one stored the step before, so the
reflection lags by one step; in steady flow that is plain bounce-back. The
y-edges are not involved (`jbnd = 0`), and the i-k edge routine only runs for
closed x boundaries.

**The ground under the STL.** The xie_and_castro STL carries a ground plate at
z = 0 under x = 0..160 m. The ray-cast voxelisation (above) only fills cells
whose centre is at or below a roof, so a plate at z = 0 fills nothing: the
first cell row (centre 1 m) is fluid everywhere except under the blocks, on the
fetch (x < 0, where the STL has no ground) and in the array alike. The ground
is therefore the `kbnd` boundary everywhere, as uDALES (STL ground plus facets
over the fetch) and PALM (MOST wall) have a wall everywhere. Until 2026-10
pylbm left `kbnd = 22`, so its ground was free-slip everywhere while its
buildings (full-way bounce-back, `m_solids.F90`) were no-slip.

**No wall model.** The LBM has no wall function or roughness length.
`iablvisc = 1` is not one: it adds a fixed mixing-length eddy viscosity
`kappa u* z (1 - z/h)^2`, with `u*` from `uini` at a hard-coded z = 100 m and
z0 = 0.1 m (`m_mechanical_ablvisc.F90`), zero at the ground, so it sets no
wall stress. The no-slip ground is plain bounce-back on the 2 m grid.

**Frozen ground corners.** No kernel writes the ghost cells where the k = 0
plane meets the inflow (i = 0) and outflow (i = nx + 1) ghost planes, but
their populations stream into the cells (1, j, 1) and (nx, j, 1) every step,
so they keep whatever the initial state put there. The solver's cold start
(`fillghosts`) copies the flow next to them, which for a no-slip ground is a
wall moving at the inflow speed: on the empty domain below it lifted the flow
at the inflow (w = 0.6 m/s at x = -19 m, a +14 % jet at z = 5 m) and drained
4.2 % of the domain's mass within ~20 s. Nothing restores the mean density
(the Krüger inflow fixes the momentum flux `rho0 u`, the zero-gradient outflow
anchors nothing), so every velocity `u = j / rho` stayed 4.4 % high. pylbm
therefore sets both corner columns to the resting wall (rho = 1, u = 0) in
every restart it writes (`warm_start_utils._hold_ground_corners_at_rest`), and
stages a no-slip cold start as a restart from the state `inipert` builds
(`_prepare_restart_cold_start`, as for the digital filter). The density then
stays at 1.001-1.002 on the empty domain and the jet is gone; Fortran-saved
restarts carry the corners on to warm starts. Not covered:
`inlet_turbulence.method: random_forcing` with `no_slip` still cold-starts
through the solver (its restart run would need the solver's own turbulence
buffers), so pylbm warns. With buildings the mean density still drifts, by
about +2 % with either ground (the blocks' start-up transient), which is the
same unanchored level, not the corners.

**Measured** — empty domain x = -20..140 m (80x40x16, 2 m), 10°, 6 m/s, α 0.25,
z_ref 64 m, 600 s after 60 s spin-up, means from t = 100 s; PALM on the same
domain (docs/pypalm.md §8). Time- and y-mean horizontal speed at z = 1 m,
relative to the first plane (x = -19 m), laminar / digital-filter inflow:

| fetch | PALM | pylbm `free_slip` | pylbm `no_slip` |
|---|---|---|---|
| 10 m (x = -9) | -4 / -5 % | +2 / +2 % | -12 / -12 % |
| 20 m (x = 1) | -9 / -9 % | +3 / +3 % | -19 / -19 % |
| 40 m (x = 21) | -17 / -15 % | +3 / +4 % | -30 / -29 % |
| 80 m (x = 61) | -27 / -19 % | +4 / +7 % | -44 / -43 % |
| 120 m (x = 101) | -32 / -17 % | +5 / +11 % | -52 / -50 % |

The wall stress, from the streamwise momentum budget of the time-mean flow
(`-d/dx` of the z-integrated `u u + p`, free-slip lid, cyclic y), is 0.045 /
0.028 m²/s² (u* 0.21 / 0.17 m/s) over x = 1..41 / 41..101 m with `no_slip`,
and zero within ±0.005 with `free_slip`. PALM's MOST stress at its own z = 1 m
speed is 0.056 / 0.041-0.052 m²/s² (u* 0.24 / 0.20-0.23). Against its own
z = 1 m speed the bounce-back wall acts like a log-law wall with z0 ≈ 0.08-0.09
m instead of 0.05 m: it is draggier, so the near-ground flow decelerates about
twice as much as PALM's (its stress is lower only because its speed is). This
is the residual difference of the ground. Higher up the profile follows PALM's:
rms difference over z ≤ 25 m 0.12-0.16 m/s laminar and 0.12-0.25 m/s turbulent
(z = 3 m -0.3..+0.2 m/s, z = 5 m up to +0.17 m/s, above 9 m within 0.1 m/s),
against 0.06-0.27 / 0.06-0.20 m/s with `free_slip`, whose ground accelerates
the flow where PALM's slows it. The ground does not add resolved turbulence:
TI_u at 9 m 0.061 / 0.087 / 0.107 at 2 / 22 / 102 m past the slab (`free_slip`
0.062 / 0.091 / 0.123).

**On xie_and_castro** (`workflows/compare_models.sh`: uDALES, PALM and pylbm
with both grounds, 2 replicas each, 10°, 5 m/s, 420 s, statistics from 150 s;
pylbm on x = -20..100 m with the STL cut at x = 80 m, i.e. the other backends'
20 m flat outflow buffer, which `outflow_buffer` now builds), the distance/noise
ratio of the time-mean u against PALM (against uDALES in brackets):

| region | turbulent `no_slip` | turbulent `free_slip` | laminar `no_slip` | laminar `free_slip` |
|---|---|---|---|---|
| all | 3.9 (4.9) | 3.6 (4.2) | 10.7 (113) | 9.4 (24) |
| street level, z < 4.5 m | 5.3 (15) | 3.6 (8.9) | 12.7 (81) | 8.9 (31) |
| canopy | 4.6 (5.5) | 4.4 (5.0) | 9.7 (119) | 9.0 (21) |
| above the canopy | 2.4 (3.2) | 2.3 (3.0) | 13.8 (389) | 12.5 (39) |

(PALM vs uDALES: 2.7 turbulent, 9.0 laminar. The laminar `no_slip` replicas are
bit-identical, so the ratios against the steady uDALES run blow up.) pylbm is
too slow near the ground in the array with either setting: plane-mean speed at
z = 1 m 0.98 m/s with a free-slip ground and 0.57 m/s with `no_slip`, against
1.54 (uDALES) and 1.45 m/s (PALM), laminar and turbulent alike. uDALES's
laminar run has no turbulence either, so the gap is the LBM's walls
(bounce-back blocks and ground on a 2 m grid, no wall model), not missing
mixing. A free-slip ground offsets part of the draggy blocks; `no_slip` is the
boundary the other backends have, and leaves the wall-model difference visible
instead of cancelling it, at the cost of street-level agreement (time-mean u
distance to PALM 0.72 against 0.50 m/s, turbulent).
`ground_boundary: free_slip` gives the earlier behaviour.

### Outflow buffer

`outflow_buffer` (`${flow.outflow_buffer}`: 20 m on xie_and_castro, null in
`common.yaml` and the periodic case) mirrors pyudales and pypalm (docs/pypalm.md
§8): `_extend_for_outflow_buffer` adds `ceil(buffer / dx)` cells past
`bounds`' xmax, so `mod_dimensions.F90`, the bathymetry and the restarts are
built for the extended `nx` (50 -> 60 on the case). The buffer is empty:
`stl_to_lbm_geometry(x_max=xmax)` leaves every column past xmax fluid (the
xie_and_castro STL has a fifth block column at x = 85-95 m), and the ground
there is the same `kbnd` boundary as everywhere else. `run_single` crops every
state back to `bounds` (`_crop_outflow_buffer`), so callers never see the
buffer; a warm start pads the cropped state by repeating its last column, with
`blanking` 0 (`_pad_outflow_buffer`). The inflow, its profile and the
digital-filter forcing are per (y, z) and do not see `nx`. `null` changes
nothing; a buffer with `boundary_condition: periodic` or ≤ 0 m raises.

**Measured** on xie_and_castro through the normal config path (x = -20..80 m
plus 20 m, `no_slip`, 10°, 5 m/s, 420 s, laminar and digital-filter seed 1):
the states are on the case grid (50 x-cells, x = -19..79 m) and bit-identical
to the same runs on the hand-extended domain with the STL cut at 80 m (the
comparison in "Ground and lid"). Canopy TKE stays at 0.04-0.06 m²/s² across
the array up to x = 77 m, |u| ≤ 7.1 m/s, |w| ≤ 3.1 m/s. The mean density
settles 2.0-2.4 % above 1 within the first flow-through (fluid mean 1.0037 at
t = 5 s, 1.016 at 20 s, 1.024 at 40 s, then steady) and stays there; it does
the same with a free-slip ground (1.016-1.020), so it is neither the buffer nor
the ground. As in "Frozen ground corners", nothing anchors the LBM's density
level under a velocity inflow and a zero-gradient outflow: while the wake
first reaches the outflow, less mass leaves than enters, and every velocity
`u = j / rho` then reads ~2 % low.

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
| [`inlet_turbulence_utils.py`](../libs/pylbm/src/pylbm/utils/inlet_turbulence_utils.py) | `validate_inlet_turbulence` (config checks, incl. the periodic guard), `apply_inlet_turbulence` (the `random_forcing` `lturb` line), and the `digital_filter` path: `build_digital_filter_fluctuations`, `build_forcing_buffers`, `write_turbulence_restart`, `write_digital_filter_forcing`, `aligned_start_iteration` |
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

**`profile_config.z_ref`.** pyudales and pypalm read `velocity_magnitude` as the
speed at `z_ref`. Because of the top-cell renormalisation `z_ref` used to have
no effect on the LBM at all (with `z_ref: 64` on a 32 m domain the LBM inflow
was 20 % faster than PALM's). When `z_ref` is set, `reference_speed_scale`
writes `uini = velocity_magnitude * s(z_top)` (to 4 decimals; also into
`uvel_time.dat`), so the three backends impose the same profile. Without
`z_ref` nothing changes: `uini` is the speed at the top cell centre, half a
cell below the other backends' default `z_ref = zsize`. `pylbm.yaml` reads it,
with the profile type and alpha, from the shared `flow.profile` block
(`configs/common.yaml`, `z_ref` 64 m), as pyudales and pypalm do; set
`flow.profile.z_ref=null` to get the old top-cell behaviour. `C_u` is still
derived from `velocity_magnitude`. The digital-filter keys (intensity, length
scales, time step, seed) also come from `flow.inlet_turbulence`; `enabled` and
`method` stay pylbm's own (off, `random_forcing`).

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

1. `sgs_constant` in the params Dataset (from the `configs/params/*.yaml` sampler) —
   used when ESMDA estimates or pins it.
2. `forward_model.sgs_constant` in the backend's own `configs/model/*.yaml` — the
   per-backend default.

Absent from both is a strict no-op: the solver's own closure/template value
stands. The per-backend default exists because the three backends' `sgs_constant`
are **different physical quantities** (uDALES/pylbm take a dimensionless
Smagorinsky-family constant; PALM takes an eddy diffusivity in m²/s), so a single
value in the shared params sampler cannot be correct for all three at once.


### Inflow turbulence — `inlet_turbulence`

`inlet_turbulence.enabled` is the cross-backend inlet-turbulence switch;
`method` picks the mechanism. Both use the Fortran's inflow-turbulence subsystem
(`m_inflow_turbulence_{init,compute,apply,update,forcing}.F90`), whose three
settings are read by one statement (`read(10,*) inflowturbulence,
turbulence_ampl, nrturb`, the `lturb` line), so they are always rewritten
together with `Infile.set_value_tokens`. Common rules:

- **Absent (`None`) or `enabled: false` → strict no-op.** The line the solver
  generated for itself is left byte-identical, and no file is written.
- **Requires `boundary_condition: inflow_outflow`.** `main.F90` gates the
  turbulence refresh on `ibnd == 1`, so `enabled: true` with `periodic` raises a
  `ValueError` at construction instead of silently doing nothing.
- Both inject the turbulence through a **forcing slab at `i = iturb_pos = 10`**
  (`m_inflow_turbulence_init.F90:3`), i.e. 9.5 cells downstream of the inlet
  face (x = -1 m on xie_and_castro at 2 m), not on the inlet face itself.
  Upstream of the slab the flow is the laminar Krüger inflow.

#### `method: random_forcing` (default) — the LBM's own perturbations

```yaml
inlet_turbulence:
  enabled: false
  method: random_forcing
  amplitude: 5.0e-05      # turbulence_ampl
  update_interval: 100    # nrturb — timesteps between turbulence updates
```

Applied once in `compile()` and inherited by ensemble members when
`experiment_dir` is cloned. Values omitted while enabled fall back to the
infile's (the solver template `m_mkinfile.F90` writes ` F 0.00005  100`).

**What the solver does.** Every `nrturb` steps (`main.F90:426-428`, and once at
a cold start, `:270`) `inflow_turbulence_compute` draws `nrturb + 1` fresh
random planes `uu/vv/ww/rr(ny, nz, 0:nrturb)` with `m_pseudo2D` (Gaussian
covariance, decorrelation length `10/sqrt(3)` ≈ 5.8 cells, FFT-periodic in y
**and z**, so the ground and the lid are correlated), chains them by an AR(1)
of 0.98 per step (e-folding ~50 steps) and normalises every plane to zero mean
and unit variance (`m_inflow_turbulence_compute.F90:19-60`). Each refresh starts
from an independent plane, so the forcing decorrelates abruptly every `nrturb`
steps. Every step, `inflow_turbulence_forcing` adds `amplitude * uu(:, :, lit)`
(`lit = mod(it, nrturb)`, 0 → `nrturb`) to the slab's velocity as a Kupershtokh
exact-difference forcing, `f += feq(rho, u + du) - feq(rho, u)`
(`m_inflow_turbulence_forcing.F90:57-127`, applied post-collision at
`main.F90:327`); `rr` is never used. `amplitude` is therefore a **lattice
velocity increment per step**, not an intensity: the fluid crosses the slab in
`1/U_lattice` ≈ 15 steps (`C_u = 15 U`), so the injected fluctuation is roughly
`15 * amplitude * C_u` m/s and does not scale with the flow. `update_interval`
must be ≥ 1 (`inflow_turbulence_init` aborts on `nrturb <= 0`) and sizes the
buffers. Restart runs read the buffers back from
`restart/turbulence_0000_<nt0>.uf` (`m_readrestart.F90:66-101`).

**Seeds.** `m_seedmanagement.F90` seeds `RANDOM_NUMBER` only on a cold start
(`nt0 = 0`): from `seed_0000.orig` when it exists, otherwise from the wall
clock (`m_set_random_seed3.F90:33-37`, `DATE_AND_TIME` + `system_clock` + rank)
and saves it there. `compile()` wipes the seed files, so every cold start draws
a new clock seed and two runs of one config differ (see §9). On a warm start it
looks for `seed.dat` — a name nothing writes — so it never reseeds, and the
refreshes of every warm window start from the compiler's default RNG state.

**Measured** (empty domain x = -20..140 m, 2 m grid, `ny` 40, `nz` 16, 10°,
6 m/s, α 0.25, `z_ref` = top cell, 600 s after 60 s spinup, frames from
t = 100 s; z = 9 m, local mean ≈ 4.4 m/s):

| `amplitude` | plane | TI u / v / w at 9 m | T_int | L_y |
|---|---|---|---|---|
| 5e-5 | inflow face (x = -19) | 0.000 / 0.001 / 0.000 | — | — |
| 5e-5 | 2 m past the slab (x = 1) | 0.009 / 0.016 / 0.015 | 1.5 s | 6.5 m |
| 5e-5 | x = 101 | 0.017 / 0.015 / 0.006 | 3.9 s | 5.4 m |
| 5e-4 | inflow face (x = -19) | 0.002 / 0.007 / 0.004 | — | — |
| 5e-4 | 2 m past the slab (x = 1) | 0.093 / 0.156 / 0.157 | 1.5 s | 6.4 m |
| 5e-4 | x = 21 | 0.130 / 0.150 / 0.125 | 1.9 s | 5.8 m |
| 5e-4 | x = 101 | 0.143 / 0.118 / 0.100 | 4.3 s | 5.4 m |

`5e-5` (the template default) is laminar for practical purposes; `5e-4` gives
a usable intensity but with short scales (L_y ≈ 6 m ≈ 3 cells, T_int ≈ 1.5 s)
and no handle on the spectrum, the length scales or the realisation.

#### `method: digital_filter` — the inflow uDALES and PALM use

```yaml
inlet_turbulence:
  enabled: true
  method: digital_filter
  intensity: 0.15         # u'_rms / |U_mean(z)|, constant in z; v', w' 0.7x
  length_scale_y: 12.0    # m
  length_scale_z: 12.0    # m
  length_scale_x: 24.0    # m -> AR(1) time scale via Taylor
  time_step: 0.5          # s between planes; must divide the window
  seed: null              # null -> derived from the member's experiment name
```

Same keys, defaults and generator as pyudales (§6.1 of docs/pyudales.md) and
pypalm (§8 of docs/pypalm.md):
[`pyurbanair/utils/digital_filter.py`](../src/pyurbanair/utils/digital_filter.py).
`build_digital_filter_fluctuations` returns the fluctuation part of uDALES'
driver planes value for value for the same seed, grid and record times
(`test_matches_the_udales_inlet_planes`). The seed is `seed`, else
`derive_seed(experiment_name)`: ensemble members (`000`, `001`, … on every
backend) get the same realisations as their uDALES/PALM counterparts; a single
run is named `runcase` here, so set `seed` explicitly to drive pylbm and
another backend with one realisation, and change it for a replica. No solver RNG
is involved, so a digital-filter run is bit-for-bit reproducible.

**Why it goes through the forcing slab.** The LBM cannot read a time-varying
inflow plane: the Krüger inflow (`m_boundary_i_inflow_kernel.F90:58-70`) takes
one `uvel(nz)` profile and one `udir`, uniform in y and without w, and the only
time input is the scalar speed/direction series `uvel_time.dat`
(`m_inflow.F90:44-86`). The only per-(y, z, step) input is the forcing buffer,
which `m_readrestart` loads from `restart/turbulence_0000_<nt0>.uf` on every
run with `nt0 > 0`. So, without editing the vendored code, each window is run
as a restart run whose buffers pylbm writes:

- **Iterations.** `nrturb` = the window length `L` in steps and `nt0` a
  multiple of `L` (`aligned_start_iteration`). The forcing reads slot
  `mod(it, L)` (0 → `L`), so step `nt0 + l` reads slot `l`: the buffer is
  played once, front to back, and the solver's own refresh at
  `mod(it, nrturb) == 0` falls on `nt1`, after the last forcing. A cold start
  writes an equilibrium restart at `nt0 = L` holding the state `inipert` would
  build (the inflow profile everywhere, rho = 1; without its 1e-5 density
  noise) and removes old restarts first; a warm start writes its restart at the
  next multiple of `L` after the previous `nt1`. The iteration counter
  therefore advances by up to `2L` per window — mind `MAX_ITERATION` (§9) on
  long rollouts.
- **Lattice units.** In the mean flow a slab increment `du` per step leaves a
  jump `du / U_x` downstream (the fluid spends `1/U_x` steps in the one-cell
  slab). pylbm writes `du = U_x(z, t) * u'/C_u` (same for v', w'),
  `amplitude = 1`, where `U_x` is the x-velocity the inflow kernel imposes at
  that step and height (`uini * uvel_shear * cos(udir)` in lattice units,
  following `uvel_time.dat` when the params vary in time). The
  exact-difference forcing leaves rho unchanged, so mass is conserved exactly;
  the plane mean of u' is removed per record, as on the other backends, so the
  downstream mass flux carries no mean perturbation.
- **Time.** Planes every `time_step` s, interpolated linearly onto the steps
  (as uDALES and PALM interpolate theirs); step `l` sits at window time
  `(l - 1) * dt`, the clock `m_inflow.F90` uses for the mean inflow.
- **Grid.** The planes are built on the LBM's cell centres, which are the case
  grid's (u, v at `(k + 0.5) dz`, w on the faces averaged to the centres).
- **Windows.** The member's elapsed physical time is persisted to
  `<experiment_dir>/inlet_turbulence_clock.json` (reset when the model is
  built, advanced only after a successful window) and selects the slice of the
  member's turbulence history, so window *n* continues window *n-1*, as on
  uDALES/PALM. As on PALM, a member substituted after a failure keeps its own
  clock (statistically harmless).
- **Memory.** The four buffers `(ny, nz, L + 1)` float32 stay on the device
  for the whole run, and the restart read/write holds a host copy: 4 × 640 ×
  29 701 × 4 B ≈ 300 MB for the 660 s empty-domain window below. pylbm warns
  above 512 MB; long windows on large planes need shorter windows.

**What the flow does with it.** For v' and w' the slab jump is the whole
story. A streamwise force with zero plane mean is not divergence free, so half
of the u-impulse goes into a pressure field that decays over about one
transverse wavelength on both sides of the slab: u' is about half its target
just downstream of the slab, shows up weakly upstream of it, and recovers
downstream as the pressure field decays — the same projection the
uDALES/PALM pressure solvers apply to their face planes, but applied here in
the interior.

**Measured** (same empty domain; 10°, 6 m/s, α 0.25, `z_ref` 64 m as in the
pypalm/pyudales configs; 600 s after 60 s spinup, frames from t = 100 s;
integral scales at z = 9 m from u'; the plane's own TI at 9 m is 0.128 / 0.111
for u / v; free-slip ground, `ground_boundary: free_slip` — the default
`no_slip` gives the same turbulence and the near-ground profile of §5 "Ground
and lid"):

| plane (distance past the slab) | mean-profile error (rms, z ≤ 25 m) | TI u / v / w at 9 m | T_int | L_y | spectral slope 0.08-0.4 Hz |
|---|---|---|---|---|---|
| inflow face, x = -19 (upstream) | 1.1 % | 0.002 / 0.002 / 0.002 | — | — | — |
| x = -3 (2 m upstream) | 1.4 % | 0.037 / 0.039 / 0.036 | 3.5 s | 10.5 m | -3.4 |
| x = 1 (2 m) | 1.6 % | 0.062 / 0.091 / 0.080 | 3.7 s | 9.3 m | -3.2 |
| x = 21 (22 m) | 2.2 % | 0.091 / 0.088 / 0.069 | 4.7 s | 9.3 m | -3.7 |
| x = 61 (62 m) | 3.0 % | 0.106 / 0.096 / 0.057 | 5.9 s | 8.4 m | -4.8 |
| x = 101 (102 m) | 4.1 % | 0.123 / 0.096 / 0.057 | 6.6 s | 8.2 m | -5.0 |

PALM on the same domain and inflow (docs/pypalm.md §8), by distance past its
inflow face: 0 m 0.131 / 0.102 / 0.091, 2.7 s, 10.6 m; 20 m 0.115 / 0.108 /
0.076, 4.1 s, 9.9 m; 80 m 0.132 / 0.118 / 0.066, 5.2 s, 7.6 m; 120 m 0.148 /
0.117 / 0.072, 6.0 s, 6.8 m; mean-profile error 0.5-5.6 %.

- The flow carries the written realisation: u' and v' at 9 m, 2 m past the
  slab, correlate 0.78 / 0.79 with the generator's planes for that seed (1 s
  lag). v' and w' arrive at 80-90 % of the planes' rms; u' at about half, as
  explained above, and it recovers over the fetch (0.091 at 22 m, 0.123 at
  102 m) while TI_w decays (0.080 → 0.057). The fluctuations neither die nor
  blow up.
- Relative to PALM the LBM's turbulence ends up 15-25 % weaker at the same
  distance and its spectrum steeper (-3 to -5 against -1.7 to -3.4): the slab
  halves u' at the injection point, and the LBM's ground makes no turbulence
  as PALM's does: with the bounce-back ground (`no_slip`) TI_u at 9 m is
  0.061 / 0.087 / 0.107 at 2 / 22 / 102 m, no more than over the free-slip one.
  Integral scales match the other backends (L_y 8-10 m against 7-11 m; T_int
  4-7 s against 3-6 s).
- The mean profile stays within 1-4 % of PALM's target `6 (z/64)^0.25`; the
  laminar LBM run (`enabled: false`) drifts the same way (1.7-2.3 %), so that
  is the profile adapting to the free-slip ground, not the forcing. With
  `no_slip` the error is 8-16 %, the near-ground deficit of §5.
- A second seed gives the same statistics (TI_u 0.063 / 0.090 / 0.112 at 2 /
  22 / 102 m, L_y within 10 %) and an uncorrelated realisation (u' correlation
  with the first |r| < 0.04 at every plane); the 500 s means of the two differ
  by 0.05-0.09 m/s rms, the realisation noise.
- On xie_and_castro with the inflow moved upstream (STL cropped to the blocks
  at x ≤ 35 m, domain x = -60..120 m, 90x40x16, two 300 s windows, the second
  warm-started): 20 m past the slab (x = -21, 26 m in front of the array) TI
  u / v / w at 9 m is 0.091 / 0.089 / 0.068 in window 1 and 0.093 / 0.091 /
  0.070 in window 2, TI_u 0.08 at 17-23 m, and in front of the first block
  (x = -1) TI_u rises to 0.17. The warm window starts at `nt0` = 40 500 (the
  next multiple of its 13 500 steps after 32 400) and continues the clock at
  record 720 (360 s); the frame-to-frame change of u across the window
  boundary (0.13 m/s) is within the ordinary range (median 0.10 m/s). PALM's
  run on the same layout gave 0.17 / 0.16 / 0.10 25 m in front of the array,
  with its inflow face 40 m upstream of that plane.
- Not verified: the CUDA build (all numbers above are the gfortran/OpenMP
  build; the buffers are read through the same `m_readrestart` CUDA branch that
  `random_forcing` restarts already use), MPI builds (pylbm never builds with
  MPI), and time-varying params in a full run (unit tested only).

### `C_u` scaling

`C_u = int(velocity_magnitude * 15)`. All output velocities in the NetCDF are
in lattice units; `scale_velocity_to_physical` multiplies by `C_u` to recover
m/s. When `params` is `None`, `C_u` defaults to 75.

---

## 8. Configuration

[`configs/model/pylbm.yaml`](../configs/model/pylbm.yaml):

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
  ncpu: 1                    # OpenMP threads; >1 builds with MP=1
  verbose: false
  boundary_condition: inflow_outflow
  ground_boundary: no_slip     # kbnd 12; free_slip: 22 (§5)
  outflow_buffer: ${flow.outflow_buffer}   # xie_and_castro: 20 m (§5)
  profile_config: {type: power_law, alpha: 0.25}
  inlet_turbulence: {enabled: false, method: random_forcing, amplitude: 5.0e-05,
                     update_interval: 100, intensity: 0.15, length_scale_x: 24.0,
                     length_scale_y: 12.0, length_scale_z: 12.0, time_step: 0.5,
                     seed: null}
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

# Four OpenMP threads per run (with an ensemble, also ensemble.num_cpus_per_process=4)
model.forward_model.ncpu=4

# Skip recompile (binary already up to date)
model.compile=false

# Turn on inflow turbulence (needs boundary_condition=inflow_outflow)
model.forward_model.inlet_turbulence.enabled=true
model.forward_model.inlet_turbulence.amplitude=2.0e-04
model.forward_model.inlet_turbulence.update_interval=50

# The digital-filter inflow uDALES and PALM use (same flow.* settings)
model.forward_model.inlet_turbulence.enabled=true
model.forward_model.inlet_turbulence.method=digital_filter
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

### OpenMP thread scaling, and why two runs never match

With the same random seed (below), `ncpu > 1` reproduces the serial run bit
for bit on macOS arm64 (checked at 1/2/4/8 threads, inflow turbulence on). On
x86-64 Linux it does too without inflow turbulence, but with it the OpenMP build
differs from the serial build by up to 2e-5 m/s on the tiny case. That
difference is the same at 1, 2 and 4 threads (with `OMP_NUM_THREADS=1` too) and
reproducible run to run, so it is code generation: the inflow-turbulence loops
compiled with `-fopenmp` under `-Ofast -march=native` round differently, not
thread scheduling, the RNG (drawn outside parallel regions) or reductions (none).
The thread count never changes the result. Seeds matter far more: on a cold start
`m_seedmanagement.F90` seeds `RANDOM_NUMBER` from the clock unless
`seed_0000.orig` exists in the experiment dir, and `compile()` wipes the seed
files, so two serial runs of one config already differ (~4e-5 m/s on the tiny
case, ~0.1 m/s with inflow turbulence). Copy one run's `seed_0000.orig` into the
other's experiment dir before comparing them
(`tests/pylbm/test_pylbm_openmp.py`). The `digital_filter` inflow (§7) runs
every window as a restart and never touches the solver's RNG, so its runs are
bit-for-bit reproducible without that step.

Speed-up on an Apple M-series laptop (4 performance + 6 efficiency cores):
~3.2x at 4 threads (3.3x on 120x120x24, 3.1x on 400x400x32), only 3.9x at 8 —
little gain beyond the physical performance cores, as the solver is
DRAM-bandwidth-bound. `ulimit -s` in `run()` raises only the main thread's
stack; the OpenMP worker threads get the runtime's default (`OMP_STACKSIZE`).
That default sufficed up to 400x400x32 on macOS (not measured on Linux), so no
`OMP_STACKSIZE` is set; if large grids segfault only with `ncpu > 1`, export a
bigger one. For an ensemble, weigh `ncpu` against
`ensemble.num_parallel_processes` on the same budget of cores.

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

`tests/pylbm/test_pylbm_restart_filenames.py` parses the width out of the Fortran
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
  a pure-equilibrium distribution. That had a second, independent cause and is
  fixed separately — see "Restart record layout" below.

Output *collection* was never affected: `_get_output_files_for_current_run` globs
`out_0000_F(\d+)` and is width-agnostic. Only the exact-name fallback used the
wrong width.

The ceiling still matters after the fix. `nt0` accumulates across warm starts, so
a long rollout reaches 999,999 even when no single window is near it;
`_set_scaling_factors` raises rather than letting the name overflow to
`out_0000_F******.nc`, which matches no glob and no restart pattern (so it could
never be read *or* pruned). Start from a clean experiment dir, shorten the run,
or widen `i6.6` in the Fortran **and** `ITERATION_FIELD_WIDTH` together.

### Restart record layout, and why the template read needs shaped dtypes

`m_saverestart.F90` writes one unformatted record — `write(iunit) nx,ny,nz,nl,f`
with `f(nl,0:nx+1,0:ny+1,0:nz+1)` — so on disk it is 4 default `int32` followed
by `27*(nx+2)*(ny+2)*(nz+2)` `float32`, wrapped in gfortran's 4-byte length
marker at each end. The reals are 4-byte **only because this repo's build never
passes the makefile's `DP=1`** (`-fdefault-real-8`); if that ever changes, the
reader and `write_restart_file_from_xarray` will silently disagree with the
solver about every value in the file.

Reading it needs **shaped** dtypes:

```python
_, f_flat = f.read_record(
    np.dtype((np.int32, (4,))),                # header
    np.dtype((np.float32, (expected_size,))),  # payload
)
```

Given a list of *scalar* dtypes, scipy treats them as one repeating 20-byte
compound and demands the record be a multiple of it — which an LBM restart never
is, so `read_record(np.int32, np.int32, np.int32, np.int32, np.float32)` raises:

```
ValueError: Size obtained (313648) is not a multiple of the dtypes given (20).
```

The payload length is derived from the record's **own** leading marker rather
than from the expected grid (mirroring `m_readrestart`'s read-header-then-rewind).
Sized from an assumed grid instead, a wrong-grid restart and a truncated one both
fail inside scipy with the same arithmetic complaint and lose their specific
diagnostics; sized from the file, each reaches the guard that names it.

**This was broken until 2026-08-07**, and the consequence outlived the filename
bug above. The exception hit a bare `except Exception: return None`, so the
template was treated as absent on *every* call and every pylbm warm start was
built from a pure-equilibrium distribution — discarding the non-equilibrium
stress. The macroscopic fields were still correct, which is why nothing looked
wrong; the solver simply had to re-establish the stress each time, as a startup
transient. Measured on a tiny case: the first warm frame sat **3.17 %** (RMS)
from the state handed in, with a visible `max|u|` excursion at frame 0; with the
template it is **0.12 %** and there is no excursion.

Two things this turned on that outlive the fix:

- **The blanking mask is live for the first time.** `fluid_mask_zyx` is computed
  unconditionally but consulted *only* inside `if template_f is not None:`, so
  the "non-zero == solid" convention had never executed. It is correct
  (`stl_to_lbm.py` writes 1 for solid; `m_solid_objects_init` sets `.true.` for
  solids; `m_netcdfout` writes 1.0 where `lblanking`) — fluid cells take the new
  state, solid cells keep the template. An inverted mask would swap the whole
  domain, so this is the thing to eyeball if a warm start ever looks wrong.
- **Peak memory per member roughly doubles.** The template branch holds
  `template_f`, `feq`, `feq_template` and `f_new` at once — four full
  27-direction fields where the equilibrium path held one. Negligible at
  45x60x24 (~8 MB each), ~890 MB each at 200³, i.e. ~3.5 GB per member times
  `ensemble.num_parallel_processes`. See the in-memory ensemble OOM gotcha below.

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

### Outflow in the wake

Without `outflow_buffer` the xie_and_castro outflow (zero-gradient, x = 80 m)
sits 5 m behind the last block column, in its wake, and the flow there breaks
down with either ground (420 s, 10°, 5 m/s): time-mean canopy TKE grows from
0.4 m²/s² at x = 37 m to 10-22 m²/s² at x = 61-77 m (PALM 0.2-0.4), |u|
reaches 25 m/s and the mean density drifts by 2-6 %. The case therefore sets
`flow.outflow_buffer: 20.0`, which pylbm honours like the other backends
(§5 "Outflow buffer"); set it whenever the bounds end in a wake.

### Domain-height SIGFPE

The Boltzmann solver diverges to SIGFPE when the STL building heights exceed the
domain's z extent. Verify that `bounds[2][1]` (domain top in metres) is greater
than the tallest structure in the STL. For Barcelona, the merged-ground
elevation from the STL adds to building heights, so the effective height is
higher than the building-only measurement.

### In-memory ensemble OOM at large grid sizes

`EnsembleForwardModel` concatenates all member states in memory by default.
For ensembles of ~96 members at grid sizes ≥ 75³ cells, this exhausts DRAM.
Fix: set `assimilation.ensemble_save_on_disk=true` (or `results_dir` on the ensemble
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

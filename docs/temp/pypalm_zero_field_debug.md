# pypalm forward run produces an all-zero velocity field

## RESOLVED (2026-06-03)

**Root cause: `combine_plot_fields.x` never ran, so pypalm read a zero-filled
netCDF skeleton.** For a `__parallel` PALM build with `netcdf_data_format < 5`
(our case template uses `=2`), `data_output_3d.f90` does **not** write the
velocity arrays into the per-PE `_3d.NNN.nc` netCDF — it streams them to a
Fortran binary file and relies on `combine_plot_fields.x` to merge them into the
final `_3d.nc`. PALM's internal field is correct (RUN_CONTROL shows
`UMAX=6.36, VMAX=3.23`); only the netCDF write depends on combine.

On macOS the default **palmrun** path's combine step dies with
`dyld: Library not loaded: rrtmg.so` (signal 9): `combine_plot_fields.x` is
linked against a *bare-leaf* `rrtmg.so` (vs `palm`, which has an absolute path),
and palmbuild relinks it fresh each run so no static patch survives; `mpirun`
also strips `DYLD_*`. With combine dead, no merged `_3d.nc` is produced and
`_locate_3d_output` silently falls back to the per-PE skeleton (all 0.0, **no
NaN** — exactly the sentinel `docs/palm_overhead_plan.md` predicted). This is
why hypotheses 1–3 below were all dead ends: the domain/p3d were unchanged, and
ncpu=1 does *not* make combine optional for a parallel build.

**Fix (three parts):**
1. `direct_palm._link_binaries` now also symlinks a top-level `rrtmg.so` into the
   run tempdir so combine's leaf dependency resolves via CWD on macOS (Linux was
   already covered by the `rrtmg/` subdir symlink).
2. `ForwardModel.run()` defaults to the direct path (`PYPALM_USE_DIRECT_RUN`
   defaults to `1`; set `=0` for the palmrun fallback). The slurm scripts already
   defaulted to direct — this is the M4 flip for local runs. direct_palm runs
   combine itself (no `mpirun`, with the symlinks), so the merged field is
   produced.
3. `ForwardModel._assert_combine_succeeded` raises if any of u/v/w is finite
   everywhere and identically zero — turning a future silent combine failure
   into a loud error instead of a dead field.

The original `model=pypalm run.ensemble=false run.rollout_steps=0` now yields a
real field (u mean ≈5.4 m/s, proper topography NaNs). Historical investigation
notes below.

## Symptom
`python scripts/run_forward_model.py model=pypalm run.ensemble=false run.rollout_steps=0`
(static **or** dynamic params) runs to completion but the returned/saved state has
`u, v, w` all **exactly 0** (`vel_magnitude = 0`). pyudales and pylbm work.
Reported to have worked before a config/scripts refactor.

## Key measurement (the decisive clue)
Raw PALM `*_3d.000.nc` output, **before** the `fillna(0)` in
`_load_and_postprocess_state` ([libs/pypalm/src/pypalm/forward_model.py](../libs/pypalm/src/pypalm/forward_model.py)):
- PALM **integrates cleanly**: 60 timesteps, valid time axis `[1.1, 2.0, 3.1, …]`,
  no divergence, no NaN.
- Raw `u/v/w` are **exactly 0 (nan%=0, finite-nonzero%=0)** at every timestep —
  not NaN, and not even the initial-condition value.
- So PALM itself writes a zero velocity field; `fillna(0)` is not the cause.

## Ruled out (with evidence)
- **`temp_dir` config change is NOT the cause.** `temp_dir: ${paths.experiment_dir}`
  resolves to `${hydra:runtime.cwd}/.temp`, which from the repo root is the *identical
  absolute path* as the old default `get_project_root()/.temp`. pyudales/pylbm use the
  same interpolation and work.
- **`ncpu` (16→1) is NOT the cause.** All-zero with both `ncpu=1` and `ncpu=4`.
- **Inflow path is NOT the cause.** All-zero for both `params=static` (no
  turbulent_inflow) and `params=dynamic` (turbulent_inflow `read_from_file`).
- **Setup is correct.** Dynamic driver has proper inflow (`inflow_plane_u: 4.83–5.48`);
  p3d has `u_profile=5.48`, `ug_surface=5.48`, `initializing_actions='set_constant_profiles'`,
  `turbulent_inflow_method='read_from_file'`, `switch_off_module=.false.`.
- **Topography is fine** (`urban_run_topo`: 40×60, heights 0–12.5 m, 83% open, domain
  top 40 m — not masking the domain).
- **pypalm library code + PALM case template are unchanged by the refactor** (git: only
  `conf/model/pypalm.yaml` changed — temp_dir added, ncpu 16→1). So the refactor did not
  change pypalm physics.

## Leading hypotheses for next session
1. **Domain/case difference.** The refactor moved domain into
   `conf/case/xie_and_castro/domain.yaml` (`60×40×16`, `dz=2.5`, bounds
   `[[-20,40],[0,40],[0,40]]`). Compare to the pre-refactor `conf/domain.yaml` PALM used.
   PALM may handle this grid differently than pyudales/pylbm (e.g. multigrid solver,
   anisotropic dz, inflow develop length).
2. **Per-PE output not combined.** `_locate_3d_output` reads `urban_run_3d.000.nc`
   (a per-PE file) rather than a combined `urban_run_3d.nc`. For ncpu>1 this would read a
   partial field — but ncpu=1 (single PE = full domain) is *also* zero, so this is at most
   secondary.
3. **Environment.** Confirm pypalm ever produced non-zero output *locally* (vs on a
   cluster with parallel netCDF writing a combined `urban_run_3d.nc`).

## How to reproduce / inspect
```bash
# clean stale experiment dir first
rm -rf .temp/palm_experiment
pixi run -e dev python scripts/run_forward_model.py \
  model=pypalm params=static run.ensemble=false run.rollout_steps=0 \
  model.forward_model.ncpu=4 model.forward_model.verbose=true run.skip_viz=true
```
To see the *raw* field before `fillna(0)`, temporarily log
`state[var].values` stats (nan% / finite-nonzero% / absmax) and the `output_file`
path inside `_load_and_postprocess_state` just before the `fillna` loop. The raw
`urban_run_3d.000.nc` is removed by `_clean_output` after the run, so inspect it during
the load (or disable the cleanup).

## Status of related fixes (already done this session, not pypalm)
- pyudales / pylbm rollout works again after reverting the dynamic-param time shift back
  to **window-local** `[0, simulation_time]` in `scripts/run_forward_model.py`.
- pylbm keeps a `nt0*dt` offset in `write_uvel_time_file` to map the local schedule onto
  its absolute warm-start clock (fixes the pylbm window-boundary jump).
- pyudales warm start has no warm-start support changes; rollout uses cold restarts per
  window (acceptable for now).

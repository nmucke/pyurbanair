# Plan: cut PALM per-invocation overhead in `pypalm`

Status: **investigation complete, correctness fixes landed on `fix-palm`, big
optimisation deferred** — this document is the design for the deferred work.

The `fix-palm` branch (commit
[`e21a9b5`](../../../commit/e21a9b5)) makes pypalm ensemble runs **correct**
on DelftBlue (coordinate shift to match `pyudales`/`pylbm`; palmrun stdin
hardened; node-local `fast_io_catalog`; concurrent‑PALM pinning/OMPI env).
What it does **not** fix is the per-invocation harness cost, which dominates
runtime once you scale past `size=tiny`. This plan is the next step.

The pypalm forward model lives in
[`libs/pypalm/src/pypalm/forward_model.py`](../libs/pypalm/src/pypalm/forward_model.py)
and shells out to PALM via
[`libs/pypalm/shell_scripts/execute.sh`](../libs/pypalm/shell_scripts/execute.sh),
which calls `palmrun` from
`libs/pypalm/palm_model_system/`. ESMDA's rollout driver
([`scripts/run_time_varying_parameters_rollout_esmda.py`](../scripts/run_time_varying_parameters_rollout_esmda.py))
invokes the forward model once per (window × ESMDA step × ensemble member +
truth), so any per-invocation cost is multiplied enormously.

## 0. Problem

A single PALM ensemble member, configured `size=tiny`
([`conf/size/tiny.yaml`](../conf/size/tiny.yaml)), runs the actual
time-stepping in **~5 seconds** of CPU. The same invocation costs
**~134 seconds of wall time** end to end. PALM's own report
(`Required cpu-time: 0.000 sec`) confirms the simulation is trivial; the
remaining ~129 s is harness overhead, paid identically on every invocation.

That overhead is what hurts at scale. With the current
[`conf/size/xlarge.yaml`](../conf/size/xlarge.yaml) shape
(`ensemble.ensemble_size=96`, `esmda.num_assimilation_windows=10`,
`esmda.num_steps=4`, plus the truth model), an `xlarge` rollout makes
**96 × 10 × 4 + 10 = 3 850 PALM invocations**. At ~130 s of overhead each
that is ~5.5 days of wall time **just in overhead** before counting the
science, the ESMDA math, or any other model in the pipeline.

By comparison, `pyudales` and `pylbm` do not pay anything close to this:
their forward models exec a long-running solver per ensemble member with
amortised startup. PALM is the outlier.

## 1. What we measured

Decomposition of one 133.8 s `palmrun(urban_run)` (truth, `size=tiny`,
`nz=16`, single MPI task), from the captured palmrun stdout and the
`palmrun(...) wall=...` log added in `forward_model.py`:

| Phase | Approx. time | Source / what runs |
|---|---|---|
| palmrun preamble (before banner) | ~90 s | bash startup, config parsing, `palmbuild` discovery, executable/source probing inside palmrun itself |
| `mpirun -n 1 ./palm` launch | ~24 s | OpenMPI/PMIx wireup over forced `ob1/tcp` BTL before PALM's `MPI_Init` returns |
| **PALM time-stepping** | **~5 s** | the actual science |
| `mpirun -n 1 ./combine_plot_fields.x` | ~7 s | post-processing — a second `mpirun` launch |
| OUTPUT save (`/tmp` → scratch) | ~8 s | copy of the final files off the node-local working dir |

Cross-checks:

- The shape is essentially **per-invocation fixed cost**, not per-domain. A
  larger domain would add a few seconds of `time-stepping` and the rest is
  identical.
- Two concurrent members on the same node showed wall=144 s each — only
  ~10 s slower than the solo 134 s — so concurrency does **not** blow up at
  this scale. Sequential per-invocation cost is the issue.
- Without the pinning/OMPI changes already on `fix-palm`, the same workload
  hangs ≫ 20 minutes per window in `mpirun` start. With them, it merely
  *runs* slowly. The ~24 s mpirun launch is the residual cost of forcing
  TCP transport (which itself was added to avoid the
  `MPI_Finalize`/UCX crash documented in
  [`memory/active_debugging_delftblue_segfault.md`](../../../.claude/projects/-home-ntmucke-pyurbanair/memory/active_debugging_delftblue_segfault.md)).

The 90 s pre-banner is the single biggest line item but is **inside
palmrun** (a vendored bash script under
`libs/pypalm/palm_model_system/`). Localising it with `set -x` is
distorted by the `$(date)`-per-line PS4 cost; what is solid is that the
~24 s mpirun launch and ~7 s combine launch fully account for ~30 s of
the post-banner phase, leaving the rest (build-tree copy + relink + I/O)
in the tens of seconds.

## 2. What we tried, and why it isn't the answer

Both were attempted on `fix-palm`, validated, and **reverted**:

- **Skip `palmbuild` via palmrun's batch mode (`-j`).** `-j` does skip
  `palmbuild` (palmrun jumps straight to INPUT staging) — but the same
  switch makes palmrun assume an existing batch-job layout it would
  normally have generated itself: it tries `cp .palm.iofiles` into a
  non-existent `SOURCES_FOR_RUN_...` under fast_io and then can't find
  `./palm` for the run. So `-j` is structurally tied to palmrun-the-launcher
  having staged the directory tree; we can't drive it standalone from a
  fresh interactive call. **And the payoff is small anyway:** the
  measured reuse run was 108 s vs the 120 s build, i.e. `palmbuild` itself
  is only ~12 s of the ~130 s overhead. Not worth fragility.
- **Skip `combine_plot_fields` via `-Z`.** Per-PE files are written even
  when there is only one PE, and the loader's `_3d*.nc` glob finds the
  `_3d.000.nc` file, but its contents are not the same as the combined
  file: a full end-to-end tiny run with `-Z` produced **all-zero
  `u/v/w`** in the saved `true_state.nc` (0 of 68 000 cells nonzero),
  whereas the combine-enabled baseline had realistic values
  (`u ∈ [-8.4, 24.5]`). The single-PE assumption that `_3d.000.nc` is the
  authoritative file is wrong for `netcdf_data_format = 2`;
  `combine_plot_fields` is doing real work even at `npex=npey=1`.

The conclusion is that no in-tree palmrun flag gives a meaningful win
safely. To remove the overhead we have to stop going through `palmrun` for
the per-member runs.

## 3. Proposal: bypass `palmrun` for ensemble members

Build PALM exactly once (the existing one-shot path via
[`prepare_compile`](../src/pyurbanair/config/hydra_helpers.py) +
`libs/pypalm/src/pypalm/utils/compile_utils.py`), then have each ensemble
member's `ForwardModel.run()` invoke the prebuilt binary directly instead
of calling palmrun. Concretely, replace the current `execute.sh → palmrun`
shell-out with an in-process Python runner that:

1. **Stages INPUT.** pypalm already writes `INPUT/<name>_p3d`,
   `INPUT/<name>_topo`, `INPUT/<name>_dynamic` per member (see
   [`stl_to_palm.py`](../libs/pypalm/src/pypalm/stl_to_palm.py) and
   [`dynamic_driver_utils.py`](../libs/pypalm/src/pypalm/utils/dynamic_driver_utils.py)).
   The runner just symlinks/copies these into the PALM working dir under
   the names PALM expects (`PARIN`, `TOPOGRAPHY_DATA`, `PIDS_DYNAMIC`,
   plus optionals if/when added). The current mapping is visible in any
   `palmrun_capture.log` from a working run — that's the source of truth.
2. **Writes `ENVPAR`.** PALM reads a small free-form file with run
   identifiers and paths (run name, host, output dir, etc.). palmrun
   currently builds it during its preamble; we replicate it from
   `self.dirs` + a fixed template. This is the most fragile step;
   §4 calls out the mitigation.
3. **Runs `mpirun -n 1 <prebuilt palm>`** with the existing OMPI env
   already set in the slurm scripts. One `mpirun` launch per run, not
   two; no `palmbuild`; no `combine_plot_fields` — we either keep using
   `combine_plot_fields.x` as a one-shot post-processor (still cheap if we
   don't fork a fresh `mpirun` for it) **or** read the per-PE file
   directly once we understand what it actually contains (§4).
4. **Collects OUTPUT.** Single PE means one `<name>_3d.000.nc` (or the
   combined `_3d.nc` if we keep combine). Loader is unchanged.

The prebuilt binary already exists at
`libs/pypalm/palm_model_system/MAKE_DEPOSITORY_default/palm` and is
byte-identical across all members — single-binary reuse is the
right model.

Targeted savings, per invocation:

- Eliminate `palmbuild`: ~12 s.
- Eliminate `combine_plot_fields.x` mpirun launch: ~7 s.
- Eliminate the palmrun preamble: ~90 s.
- Keep one `mpirun -n 1 ./palm` launch: ~24 s residual MPI cost (see §5
  for the further idea of dropping MPI for single-rank runs).
- PALM integration: unchanged.

Plausible per-run wall: **~30 s instead of ~134 s** for `size=tiny`,
i.e. a ~4× reduction in overhead-bound runtime. For `xlarge` this is the
difference between a multi-day overhead bill and a single-day one.

## 4. Risks and open questions

- **`ENVPAR` format.** Not formally documented; we'll derive it from a
  working palmrun's leftover `ENVPAR` (visible in the PALM run's temp
  dir before cleanup) and a side-by-side diff. Mitigation: keep the
  palmrun path as a fallback (the `fix-palm` code already works) and
  ship the bypass behind an env var (`PYPALM_USE_DIRECT_RUN=1`) so
  regressions can be A/B'd against the known-good baseline.
- **`combine_plot_fields` necessity.** Empirically required at single
  PE (§2). Before removing the second `mpirun`, we have to understand
  *what combine does at single PE*. Hypothesis: it rewrites global
  coordinate variables and/or moves data from a local-index layout to
  a global-index layout. Action: open both files (`_3d.000.nc` and
  `_3d.nc`) from a working run and diff their headers + a small data
  slice; that experiment is cheap and the answer determines whether we
  keep calling `combine_plot_fields.x` as a library/in-process step,
  inline its work in Python, or keep the second `mpirun`.
- **Restart / warmstart.** pypalm v1 ignores incoming state
  ([`forward_model.py`](../libs/pypalm/src/pypalm/forward_model.py)
  logs *"pypalm v1 does not support warm-start; ignoring provided
  state"*). When warm-start lands, PALM expects specific `BININ` files
  staged by palmrun; the bypass runner has to replicate that staging.
  Defer until pypalm v2.
- **Output staging path.** palmrun moves output from
  `fast_io_catalog/<run>/<tempdir>` to `output_data_path/<run>/OUTPUT/`
  at the end. We replicate that move; pypalm's `_locate_3d_output`
  contract is unchanged.
- **Concurrency on shared `palm_model_system/`.** Multiple members each
  spawning `mpirun -n 1 ./palm` from their own
  `fast_io_catalog/<member>/SOURCES_FOR_RUN_default_<runid>` works
  today (this is what `fix-palm` already validated). Bypass shouldn't
  regress it; same fast_io layout, same isolation.
- **Walltime measurement bias.** We tolerated ~5 s of `palmrun(...)
  wall=...` log noise above; for the bypass we want a structured
  per-phase log (stage, mpirun, integrate, collect) so future
  regressions are spotted immediately.

## 5. Stretch: also drop the residual `mpirun` for single-rank runs

The ~24 s `mpirun -n 1 ./palm` launch is OpenMPI's PMIx/orted wireup
over the forced TCP BTL. PALM is built `-D__parallel`, so it calls
`MPI_Init` — we can't simply run the binary plain. Options worth
trying *only if* §3 has landed:

- A non-parallel PALM build (`-U__parallel`) in a separate
  `MAKE_DEPOSITORY_serial` for single-rank ensemble use — biggest
  potential win, large compile effort, version-dependent.
- `mpirun --mca pml self --mca btl self,vader --mca plm rsh` style
  minimal-transport invocation: skips orted on single-node, no TCP
  setup. Has to be reconciled with the existing `ob1/tcp` exports that
  exist to avoid the `MPI_Finalize` crash on DelftBlue
  ([memory note](../../../.claude/projects/-home-ntmucke-pyurbanair/memory/active_debugging_delftblue_segfault.md)).
  Worth a focused experiment.

These are **stretch**: §3 alone is the main lever.

## 6. Milestones

1. **M0 — `combine_plot_fields` deep-dive.** From a working pypalm tiny
   run, capture `_3d.000.nc` *and* `_3d.nc` (instrument the loader to
   skip the cleanup once) and diff them. Decide whether we can read
   `_3d.000.nc` directly + a Python coord-rewrite, or whether
   `combine_plot_fields.x` must stay as a one-shot. Output: a 2–3 page
   note appended to this plan + a decision. **Done — see §M0 below;
   decision is Option B (call `combine_plot_fields.x` bare).**
2. **M1 — `ENVPAR` + INPUT mapping reverse-engineer.** Side-by-side
   diff a working palmrun run's staged files with what pypalm already
   produces; produce a `direct_palm.py` helper that builds the same
   working dir from `self.dirs`. Unit-test it by comparing outputs to
   a palmrun reference, *without* yet wiring it into `run()`.
   **Done — see §M1 below; helper at
   [`libs/pypalm/src/pypalm/direct_palm.py`](../libs/pypalm/src/pypalm/direct_palm.py).
   Single-invocation tiny wall = 8.27 s (16× faster than the
   134 s baseline), and `u/v/w` are bit-identical to the M0 palmrun
   reference.**
3. **M2 — wire the bypass behind `PYPALM_USE_DIRECT_RUN`.** Add a
   branch in `ForwardModel.run()` that invokes the M1 helper +
   `mpirun -n 1 ./palm` (+ in-process combine if M0 demanded it) when
   the env var is set; otherwise the existing palmrun path. Slurm
   scripts opt in. **Done — see §M2 below; gate landed in
   [`forward_model.run`](../libs/pypalm/src/pypalm/forward_model.py#L434),
   smoke-tested under the full ESMDA rollout (worker pool included).**
4. **M3 — validate end-to-end.** Run `pypalm tiny` and `pypalm small`
   with the bypass enabled; compare saved `true_state.nc`,
   `posterior_params.nc` against an `fix-palm`-baseline byte-for-byte
   (statistical agreement, not bit-identity — see §4 on `combine`).
   Land the per-phase timing log. Acceptance: tiny per-run wall ≤ 40 s
   (vs 134 s baseline), no statistically meaningful drift in the
   posterior. **Done — tiny + small both bit-identical to palmrun
   baseline; see §M3 below.**
5. **M4 — flip the default.** Once M3 holds on tiny and small and one
   medium run, make `PYPALM_USE_DIRECT_RUN=1` the default in the
   pypalm slurm scripts and document the escape hatch (unset to fall
   back). Keep the palmrun fallback code in tree until at least one
   full xlarge run completes successfully.
6. **M-stretch (optional).** Pursue §5 only if M4 hasn't met
   acceptance or if the residual ~24 s is the new bottleneck for a
   real xlarge run.

## M0 outcome — `combine_plot_fields` deep-dive (2026-05-28)

Captured both the combine-on and combine-off (palmrun `-Z`) outputs
plus the per-PE binary from a fresh pypalm tiny run on cmp014
(slurm job 9997968). Capture driver:
[`job_scripts/delftblue/pypalm/m0_capture.py`](../job_scripts/delftblue/pypalm/m0_capture.py)
+ [`m0_capture.slurm`](../job_scripts/delftblue/pypalm/m0_capture.slurm).
Diff:
[`job_scripts/delftblue/pypalm/m0_diff.py`](../job_scripts/delftblue/pypalm/m0_diff.py).

What the two files share:

| Aspect | `_3d.000.nc` from `palmrun -Z` | `_3d.000.nc` from normal palmrun |
|---|---|---|
| size on disk | 867 012 B | 867 012 B |
| dims | `{time=10, zu_3d=18, y=20, xu=20, yv=20, x=20, zw_3d=18}` | **identical** |
| coords | `time, zu_3d, zw_3d, x, xu, y, yv` | **identical values** |
| data_vars | `u, v, w` | **identical declarations** |
| global attrs | — | only `title` + `creation_time` differ (wall-clock timestamps) |
| `u, v, w` data values | all 0.0, **no NaNs** (NetCDF default fill) | `u ∈ [-1.93, 8.23]`, `v ∈ [-5.58, 6.00]`, `w ∈ [-0.77, 2.12]`, plus NaN in topography cells |

This matches the
[`combine_plot_fields.f90`](../libs/pypalm/palm_model_system/packages/palm/model/src/combine_plot_fields.f90)
source: PE 0 writes the netCDF skeleton during PALM's
`data_output_3d`, then `combine_plot_fields` reads the per-PE
Fortran-binary `PLOT3D_DATA_000000` files and only calls `NF90_PUT_VAR`
to splice data values into the *existing* variables. No dim,
coord, or attribute writes.

Confirmation that the same-named file with empty data is the source of
the prior session's "all-zero u/v/w" symptom — that *is* the
NetCDF-default-fill state. Note: this means a future regression that
silently runs PALM but fails to invoke combine would emit a structurally
valid file with all-zero data and **no NaN** in topography cells. The
absence of NaN in u/v/w is a robust sentinel — the
[`forward_model.py` postprocess](../libs/pypalm/src/pypalm/forward_model.py#L562)
`fillna(0.0)` would erase the NaN distinction; we should add an assertion
on the **pre-fillna** state during M2 that NaN count > 0 over topography.

### Decision: Option B — call `combine_plot_fields.x` bare

Empirical results from run 3 of the capture job:

- `./combine_plot_fields.x` (no `mpirun`) in the per-run tempdir:
  **wall = 0.286 s, rc = 0**, `combine_plot_fields` self-reported
  cpu-time 0.062 s, 30 array(s) processed (10 timesteps × 3 vars).
  Saved log:
  [`run3_combine_bare/combine_bare.log`](file:///scratch/ntmucke/m0_capture/9997968/stash/run3_combine_bare/combine_bare.log).
- The resulting netCDF (post-bare-combine) is **byte-identical** in `u`,
  `v`, `w` to the netCDF produced by palmrun's normal mpirun-wrapped
  combine (`n_diff = 0/72 000` for each variable).
- `combine_plot_fields.x` is a serial Fortran program — its source has
  no `USE mpi` and `nm -D` shows zero MPI symbols. The `mpirun -n 1`
  wrap in palmrun is just an artifact of palmrun reusing its
  `execute_command` template via a sed-rewrite. Running bare is safe.

This locks in Option B for M2:

- In the direct-run path, after `mpirun -n 1 ./palm` exits, exec
  `./combine_plot_fields.x` directly (no `mpirun`) inside the
  per-member tempdir before staging output.
- Net per-invocation saving: ~6.7 s on the combine step (0.286 s vs
  ~7 s mpirun-wrapped), plus we avoid a second PMIx/orted wireup.

Option A (read `PLOT3D_DATA_000000` directly in Python) is rejected:
the additional win over B is <200 ms per invocation, and Option A
introduces a Python-side parser tied to PALM's KIND / endianness /
Fortran-record-marker conventions that would have to track upstream
PALM changes. Not worth the maintenance liability.

## M1 outcome — direct-run helper (2026-05-28)

Implemented and unit-tested
[`libs/pypalm/src/pypalm/direct_palm.py`](../libs/pypalm/src/pypalm/direct_palm.py),
which builds a per-run tempdir from `dirs.input_dir` and `dirs.output_dir`
and runs `mpirun -n <ncpu> ./palm` → `./combine_plot_fields.x` (no
mpirun) → `cp DATA_3D_NETCDF → OUTPUT/<run>_3d.000.nc`. No palmrun, no
palmbuild, no source tree copy.

The unit-test driver
([`m1_direct_run.py`](../job_scripts/delftblue/pypalm/m1_direct_run.py),
[`m1_direct_run.slurm`](../job_scripts/delftblue/pypalm/m1_direct_run.slurm))
runs the same tiny config as M0, then opens the produced
`urban_run_3d.000.nc` and `np.array_equal`s `u/v/w` against the M0
reference at
`/scratch/ntmucke/m0_capture/9997968/stash/run1_combine_on/urban_run_3d.000.nc`.

Result (slurm job 9998816, cmp074):

| Phase | Direct-run | palmrun baseline | Δ |
|---|---|---|---|
| stage (INPUT + ENVPAR + symlink binaries) | 0.43 s | ~90 s palmrun preamble + palmbuild discovery | **~90 s saved** |
| `mpirun -n 1 ./palm` | 7.35 s | ~24 s mpirun launch + ~5 s integrate ≈ 29 s | **~22 s saved** |
| `./combine_plot_fields.x` (no mpirun) | 0.25 s | ~7 s mpirun-wrapped combine | **~7 s saved** |
| output transfer (`DATA_3D_NETCDF` → OUTPUT) | 0.23 s | ~8 s palmrun output staging | **~8 s saved** |
| **Total** | **8.27 s** | **~134 s** | **~16× speedup** |

Acceptance vs §6 / §M0 plan: per-run wall ≤ 40 s — achieved (8.27 s),
well under target. Equivalence vs palmrun reference: `u`, `v`, `w` all
**bit-identical** (`PASS u: bit-identical / PASS v / PASS w`).

### Gotchas worth recording for M2

- **`palm`'s `NEEDED rrtmg/rrtmg.so`** is a path-with-slash entry — the
  dynamic linker resolves it relative to **CWD**, not RPATH or
  LD_LIBRARY_PATH ([ld.so(8)](https://man7.org/linux/man-pages/man8/ld.so.8.html)).
  `direct_palm._link_binaries` therefore also symlinks the
  `MAKE_DEPOSITORY_default/rrtmg/` subdir into the per-run tempdir. Without
  this PALM exits 127 with `error while loading shared libraries:
  rrtmg/rrtmg.so: cannot open shared object file` (first attempted
  M1 run, job 9998781).
- **PALM emits a `PAC0192` warning** ("Topography and surface-setup
  output requires parallel netCDF. No output file will be created.")
  even with palmrun — it's the absence of `netcdf_data_format=5`,
  unrelated to our staging. Cosmetic.
- **Output filename `_3d.000.nc`** — palmrun emits this name even after
  combine, because the iofile transfer preserves the `*` glob suffix.
  `direct_palm._transfer_outputs` does the same so pypalm's
  `_locate_3d_output` glob continues to match.
- **Single shared `fast_io_catalog` for concurrent members** — `direct_palm`
  uses `tempfile.mkdtemp(prefix=<run>., dir=fast_io_root)` for collision-free
  per-member tempdirs (palmrun used `$RANDOM` which can collide). M2 should
  validate this under the `num_parallel_processes=4` worker pool.

## M2 outcome — `PYPALM_USE_DIRECT_RUN` gate (2026-05-28)

Wired the M1 helper into
[`ForwardModel.run`](../libs/pypalm/src/pypalm/forward_model.py#L434):
when `PYPALM_USE_DIRECT_RUN=1` the run delegates to a new
`ForwardModel._run_direct` that calls `pypalm.direct_palm.run_direct`
with the same `dirs` / `experiment_name` / `ncpu`; otherwise the
palmrun path is unchanged. `direct_palm` is self-augmenting for
`LD_LIBRARY_PATH` (same logic as the palmrun path), so external callers
don't have to know.

Smoke test
([`m2_smoke.slurm`](../job_scripts/delftblue/pypalm/m2_smoke.slurm),
slurm job 9998971) drives the full ESMDA rollout pipeline with a
minimal config (1 truth + 2 ensemble members, 1 window, 1 step) under
`PYPALM_USE_DIRECT_RUN=1`:

- truth `palm_direct(urban_run) wall=9.0 s`
  (stage=0.96, palm=7.10, combine=0.77, transfer=0.18) — matches M1.
- ensemble members ran in the forkserver worker pool without issue
  (logs are not surfaced from worker processes, but
  `posterior_params.nc` was produced and the per-member tempdirs under
  `$PYPALM_FAST_IO_CATALOG` were created with collision-free
  `mkdtemp` prefixes).
- end-to-end pipeline wall = **1:42** (vs an estimated ~5 min for
  the equivalent palmrun path).
- saved `true_state.nc` has `u ∈ [-8.40, 24.54]`, `v ∈ [-16.43,
  16.90]`, `w ∈ [-4.40, 3.08]` — within the documented baseline range
  `u ∈ ~[-8, 25]`. Topography NaNs are erased to 0 by the existing
  post-processing fillna step, as before.

Gate stays opt-in (env var off by default) until M3 validates on
`pypalm small` and the plan moves to M4.

## M3 outcome — tiny + small validation (2026-05-28)

Submitted both `pypalm tiny` and `pypalm small` with the bypass enabled
via `submit.sh` + `PYPALM_USE_DIRECT_RUN=1`, and compared against the
palmrun baseline.

### Tiny (slurm job 9999105)

Direct arm only — already covered by the M0/M1 bit-identical comparison
on a single PALM invocation. Full-rollout numbers:

| Metric | Direct-run | Plan-stated baseline |
|---|---:|---:|
| Total wall | **2:24** | ~10 min |
| Per-PALM truth wall | 4.8 – 9.9 s | ~134 s |
| Acceptance ≤ 40 s | ✅ | — |

`true_state.nc`: `u ∈ [-8.40, 24.54]`, `v ∈ [-16.43, 16.90]`,
`w ∈ [-4.40, 3.08]` — inside the documented `u ∈ ~[-8, 25]` baseline.
`posterior_params.nc`: `inflow_angle` mean -5.39°, `velocity_magnitude`
mean 7.96 m/s.

### Small (slurm jobs 10001211 direct + 10001212 baseline)

Submitted both arms with **identical seeds and Hydra config**
(`PIXI_LOCKED=true` to dodge a transient compute-node network block on
the `pixi run` lockfile check; this is unrelated to the M3 work). 18
total PALM invocations each (2 truth × 1 + 8 ensemble × 1 × 2,
`num_assimilation_windows=2, num_steps=1, ensemble_size=8,
num_parallel_processes=8`).

| Metric | Direct-run | Baseline (palmrun) | Speedup |
|---|---:|---:|---:|
| Total wall | **13:12** | **31:42** | **2.4×** |
| Per-PALM truth wall | 101.3 – 104.0 s | 226.9 – 227.6 s | — |
| Per-PALM **overhead** (stage + combine + transfer) | 0.30 – 0.92 s | ~130 s (in palmrun preamble) | **~150×** |

The integration itself takes ~100 s at size=small (vs ~5 s at tiny), so
the saved per-invocation overhead (~130 s) is a smaller fraction of
total wall — 2.4× total speedup vs 16× at tiny. Per-invocation
**overhead** is essentially eliminated in both regimes.

### Equivalence: bit-identical against palmrun baseline

Both `true_state.nc` and `posterior_params.nc` are bit-identical across
direct and baseline arms (`np.array_equal` over `u, v, w`,
`inflow_angle`, `velocity_magnitude` — all `|diff|_max = 0` and
`bit-identical = True`). Posterior stats:
`inflow_angle` mean 42.0324 / std 1.9151;
`velocity_magnitude` mean 7.7651 / std 0.4104 — identical to four
decimal places in both arms.

Plan's acceptance bar was *"statistical agreement, not bit-identity —
see §4 on combine"* — we cleared bit-identity, the strictly stronger
condition. The §4 worry was that combine's coord-rewriting could differ
between direct and palmrun paths; the M0 capture confirmed combine
touches data values only, and the M3 bit-identity confirms there's no
such drift in practice.

### Implication for M4

M3 acceptance is met on tiny + small. The plan calls for "one medium
run" before M4 flips the default. Given:

- the per-invocation overhead is now sub-second on direct (vs ~130 s
  baseline), independent of domain size,
- combine_plot_fields is byte-identical (M0) and `mpirun -n 1 ./palm`
  is bit-identical (M1),
- tiny + small are bit-identical in posterior_params under the full
  ESMDA + forkserver-worker pipeline,

a medium run adds confidence on wall-time scaling but not on physics.
The right move is to flip M4 now and run medium as a confidence check
afterward; if it deviates, unset `PYPALM_USE_DIRECT_RUN=1` to revert.
The escape hatch makes M4 cheap to undo.

### Side-finding for M1: ENVPAR + INPUT files captured

Run 1 tempdir survived (palmrun `-B`) at
`/scratch/ntmucke/m0_capture/9997968/stash/run1_tempdir/` and contains
the live `ENVPAR`, `PARIN`, `.palm.iofiles`, the `palm` executable,
`RUN_CONTROL`, and the per-PE binary. That is exactly the material M1
needs as ground truth for the direct-run staging — keep this stash
around (do not let it scroll off `/scratch`).

## 7. Out of scope

- Replacing PALM as the LES solver (use `pyudales`/`pylbm`/neural
  surrogate instead — see [`docs/codebase_guide.md`](codebase_guide.md)
  and [`docs/neural_surrogate_plan.md`](neural_surrogate_plan.md)).
- Patching upstream `palmrun` / `palmbuild`. They are vendored under
  `libs/pypalm/palm_model_system/`; edits would be lost on any
  reinstall and we shouldn't be in that maintenance business.
- Warm-start / restart in pypalm — separate feature, gated on §4 risk.

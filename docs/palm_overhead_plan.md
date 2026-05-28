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
   note appended to this plan + a decision.
2. **M1 — `ENVPAR` + INPUT mapping reverse-engineer.** Side-by-side
   diff a working palmrun run's staged files with what pypalm already
   produces; produce a `direct_palm.py` helper that builds the same
   working dir from `self.dirs`. Unit-test it by comparing outputs to
   a palmrun reference, *without* yet wiring it into `run()`.
3. **M2 — wire the bypass behind `PYPALM_USE_DIRECT_RUN`.** Add a
   branch in `ForwardModel.run()` that invokes the M1 helper +
   `mpirun -n 1 ./palm` (+ in-process combine if M0 demanded it) when
   the env var is set; otherwise the existing palmrun path. Slurm
   scripts opt in.
4. **M3 — validate end-to-end.** Run `pypalm tiny` and `pypalm small`
   with the bypass enabled; compare saved `true_state.nc`,
   `posterior_params.nc` against an `fix-palm`-baseline byte-for-byte
   (statistical agreement, not bit-identity — see §4 on `combine`).
   Land the per-phase timing log. Acceptance: tiny per-run wall ≤ 40 s
   (vs 134 s baseline), no statistically meaningful drift in the
   posterior.
5. **M4 — flip the default.** Once M3 holds on tiny and small and one
   medium run, make `PYPALM_USE_DIRECT_RUN=1` the default in the
   pypalm slurm scripts and document the escape hatch (unset to fall
   back). Keep the palmrun fallback code in tree until at least one
   full xlarge run completes successfully.
6. **M-stretch (optional).** Pursue §5 only if M4 hasn't met
   acceptance or if the residual ~24 s is the new bottleneck for a
   real xlarge run.

## 7. Out of scope

- Replacing PALM as the LES solver (use `pyudales`/`pylbm`/neural
  surrogate instead — see [`docs/codebase_guide.md`](codebase_guide.md)
  and [`docs/neural_surrogate_plan.md`](neural_surrogate_plan.md)).
- Patching upstream `palmrun` / `palmbuild`. They are vendored under
  `libs/pypalm/palm_model_system/`; edits would be lost on any
  reinstall and we shouldn't be in that maintenance business.
- Warm-start / restart in pypalm — separate feature, gated on §4 risk.

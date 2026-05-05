# Session handoff: `parallel-multinode-srun` branch

## Status (as of 2026-05-05 ~12:00)

Implementing two changes (**A** and **C**) to the pyudales `EnsembleForwardModel`
parallelism, then benchmarking the speedup against the original code on
DelftBlue. All code is written and uncommitted on `parallel-multinode-srun`.
**Three benchmark attempts have failed**, each unmasking a deeper layer of an
mpiexec/PRTE/slurm interaction problem in the conda-forge OpenMPI shipped via
pixi. A small **diagnostic job 9840575** (5-min wall, 2 nodes) is now queued to
empirically determine which mpiexec invocation actually launches inside this
allocation, before resubmitting the real benchmark.

The decision to commit changes A and C depends on a clean benchmark run.

## The two changes under test

### A — `srun --exact` fan-out in [local_execute.sh](libs/pyudales/shell_scripts/local_execute.sh)

Inside a slurm allocation, replace
```sh
mpiexec -n $NCPU --oversubscribe $DA_BUILD namoptions.$exp …
```
with
```sh
srun --exact -N1 -n1 --cpus-per-task=$NCPU mpiexec -n $NCPU $DA_BUILD …
```
gated by `SLURM_JOB_ID` and `PYURBANAIR_DISABLE_SRUN`. Each ensemble member's
MPI step is scheduled by slurm into a free $NCPU-core slot in the parent
allocation; concurrent members spread across nodes instead of all crowding onto
the coordinator's node and oversubscribing.

Per-member runs stay intra-node (`-N1`) — required because the warm-start file
is per-member.

### C — Persistent `ProcessPoolExecutor` in [base_ensemble_forward_model.py](src/pyurbanair/base_ensemble_forward_model.py)

Lazily-created pool stored on the instance, reused across `run_ensemble` calls
(instead of being torn down and rebuilt each call). New methods:
`_get_or_create_executor()`, `close()`, `__del__`. Catches `BrokenProcessPool`
and rebuilds once. Gated by `PYURBANAIR_PERSIST_POOL` (default `1` =
persistent).

Note: the `BrokenProcessPool` catch is wrapped around `executor.submit` calls,
but that exception more commonly surfaces from `future.result()` later in the
loop. The retry path will not fire in typical worker-death cases. Not a
blocker for the benchmark; worth fixing later.

## Diagnostic journey — what's been ruled out

All four jobs use the same workload: 2 nodes × 16 tasks × 4 cpus on
`compute-p2`, ensemble_size=20, num_parallel=20, ncpu=4 (80 task-cpus → 1.25×
oversubscribed on a single 64-core node, fits on 2 nodes).

| Job | Outcome | What it ruled out |
|---|---|---|
| 9837671 (2026-05-04) | Failed at 81s, baseline scenario, member 007 exit 1 | Bug exists in baseline path, not just changes A/C |
| 9837881 (overnight) | Same failure, member 004 | First-fix theory (`OMPI_MCA_plm=^slurm` in slurm script) was a no-op |
| 9839842 (today AM) | Same failure at 1m14s, member 016 | Second-fix theory (`PRTE_MCA_plm=^slurm` + PATH for srun) addressed *symptoms* not cause |
| 9839786 (today AM) | Cancelled before run | Replaced by smaller-workload 9839842 |
| 9839930 (today midday) | **TIMEOUT** at 20:25, exit 0:0, all `run.*.log` files **0 bytes** | `--host localhost:$NCPU` doesn't crash but **silently hangs** |

### Layer 1 (ruled out): "srun missing from PATH"

The error in `.temp/outputs/<member>/run.<member>.log` says:
```
The SLURM process starter for OpenMPI was unable to locate a usable "srun"
[…] FORCE-TERMINATE AT (null):1 - error orte/mca/plm/slurm/plm_slurm_module.c(475)
```
Pixi env activation strips `/cm/shared/apps/slurm/current/bin` from PATH inside
the python subprocess that invokes [local_execute.sh](libs/pyudales/shell_scripts/local_execute.sh).
**Fix applied**: re-prepend that dir at the top of `local_execute.sh` if
`SLURM_JOB_ID` is set and `srun` isn't found. (Also needed for the
`command -v srun` gate on the change-A path to actually fire — without it,
change A would silently fall back to baseline.) **Necessary but not sufficient**:
fixing PATH alone still failed because PRTE then takes a different unhappy path
(see Layer 3).

### Layer 2 (ruled out): "wrong env-var prefix"

This pixi env runs **OpenMPI 5.0.8** with **PRTE 3.0.11** as the runtime
(verified via `mpiexec --version` and `prte_info`). The runtime-layer MCA env
prefix is **`PRTE_MCA_*`**, not `OMPI_MCA_*` (which only governs the MPI
layer). The original yesterday-evening fix `OMPI_MCA_plm=^slurm` was silently
ignored. **Fix applied**: replaced with `PRTE_MCA_plm=^slurm` in both
[local_execute.sh](libs/pyudales/shell_scripts/local_execute.sh) and
[benchmark_parallelism.slurm](job_scripts/delftblue/benchmark_parallelism.slurm).
Necessary but not sufficient — see Layer 3.

### Layer 3 (current state): "PRTE auto-spreads ranks across the slurm nodelist"

The error message *looks* like "srun missing", but the real reason PRTE is
calling srun in the first place is that it auto-detects
`SLURM_JOB_NODELIST=cmp[…]` and tries to distribute the 4 per-member ranks
across 2 nodes — needing some PLM (slurm or ssh) to launch the remote half.
With slurm PLM excluded, ssh PLM was tried; it fails because passwordless ssh
between compute nodes isn't set up.

**Attempted fix (failed)**: add `--host localhost:$NCPU` to mpiexec calls in
[local_execute.sh](libs/pyudales/shell_scripts/local_execute.sh) so PRTE knows
it's a single-host launch with $NCPU local slots. This *should* have made it a
local-fork-only invocation. Result (job 9839930): mpiexec **hung silently** —
all 20 `run.*.log` files are 0 bytes, slurm timed it out cleanly with exit 0:0
after 20 minutes. So `--host localhost:N` is wrong for this PRTE+slurm combo
(possibly: PRTE tries to launch a daemon on "localhost" and wedges on the TCP
handshake, or `--oversubscribe` interacts badly with `--host`).

### Misleading clue: stale `tee -a run.<exp>.log` content

The per-member log files are written with `tee -a` (append). Across multiple
jobs they accumulate output from each run. During Layer 3 diagnosis I briefly
chased an "ssh to cmp272" trail in `run.000.log` that turned out to be from
yesterday's job (cmp[235,272]) appended to today's job's allocation
(cmp[304,308]). **Cleaned `.temp/outputs/*/run.*.log` before resubmitting
9839930** so the next failure mode would be unambiguous — that's what revealed
the silent-hang behavior cleanly.

## What's queued now

**Diagnostic job 9840575** ([job_scripts/delftblue/diag_mpi_launch.slurm](job_scripts/delftblue/diag_mpi_launch.slurm)):
2 nodes × 16 tasks × 4 cpus, **5-min wall** (much more backfill-eligible than
the 20-min benchmark). Runs 8 different `mpiexec hostname` invocations through
`pixi run -e delftblue --`, each with a 30s timeout, to find which combination
actually launches 4 ranks intra-node:

| Test | Command shape |
|---|---|
| T1_plain | `mpiexec -n 4 hostname` |
| T2_oversubscribe | `mpiexec -n 4 --oversubscribe hostname` |
| T3_host_localhost | `mpiexec --host localhost:4 -n 4 hostname` *(known to hang from job 9839930)* |
| T4_host_hostname | `mpiexec --host $(hostname):4 -n 4 hostname` |
| T5_unset_slurm_nodelist | unset SLURM_JOB_NODELIST/NODELIST/NNODES/etc, then plain mpiexec |
| T6_prte_ras_simulator | `PRTE_MCA_ras=simulator mpiexec -n 4 hostname` |
| T7_srun_exact_then_mpiexec_plain | `srun --exact -N1 -n1 --cpus-per-task=4 mpiexec -n 4 hostname` |
| T8_srun_exact_then_unset_then_mpiexec | T7 + unset SLURM_* inside the step |

A test passes if its block prints 4 hostname lines and the trailing `[Tn] exit=0`.
Whichever test passes is the invocation to use in `local_execute.sh`.

ETA reported by slurm: `2026-05-05T16:30` (conservative; backfills sooner for
short jobs). Output:
- `job_scripts/delftblue/out_files/slurm-diag_mpi_launch-9840575.out`
- `job_scripts/delftblue/out_files/slurm-diag_mpi_launch-9840575.err`

## Benchmark slurm script (currently sized for diagnosis, not full validation)

[benchmark_parallelism.slurm](job_scripts/delftblue/benchmark_parallelism.slurm):
20-min wall, 2 nodes, runs `scripts/benchmark_ensemble_parallelism.py` four
times back-to-back on the same allocation, varying only the env-var gating:

| label | DISABLE_SRUN | PERSIST_POOL |
|---|---|---|
| `baseline_no_srun_no_persist` | 1 | 0 |
| `C_only_no_srun_persist` | 1 | 1 |
| `A_only_srun_no_persist` | (unset) | 0 |
| `A_plus_C_srun_persist` | (unset) | 1 |

Sizing: **ensemble_size=80, num_parallel=80, ncpu=1**, sim_time=3, spinup_time=2,
num_calls=2 (1 cold + 1 warm). The `ncpu=1` matches the user's production
usage (warmstart does not work for ncpu>1); the bumped ensemble keeps
single-node oversubscription at 1.25× (80 task-cpus on 64-core compute-p2)
so change A still has a measurable signal. This depends on the user
actually running ensembles >64 in production — at ensemble≤64 with ncpu=1,
a single 64-core node fits without oversub and change A is unnecessary.

The benchmark script auto-snapshots the case namoptions to set
`nprocx=1, nprocy=1` for the run, then restores it. This is required
because [validate_and_sync_ncpu](libs/pyudales/src/pyudales/utils/ncpu_utils.py#L75-L80)
silently overrides `UDALES_ARGS["ncpu"]` to match `nprocx*nprocy` from
namoptions; without the snapshot, `ncpu=1` would be bumped back to 4. Each scenario prints:
```
[label] SUMMARY total=Xs first=Ys steady_mean=Zs n_steady=1
```
`first` is the cold call (pays pool fork + JAX init). `steady_mean` is the
average of warm calls — the per-call cost the production ESMDA loop pays.

## Predicted outcomes (when benchmark eventually runs)

- `baseline` `steady_mean` ≈ X (with 1.25× oversub thrash on coordinator node)
- `A_only` `steady_mean` ≈ X / 1.25 (fan-out across 2 nodes, no oversub) — **headline win for A**
- `C_only` `steady_mean` ≈ baseline `steady_mean` BUT `first` ≈ `steady_mean` (no rebuild), unlike baseline where `first > steady_mean` — **signal for C**
- `A_plus_C` ≈ `A_only` `steady_mean` with smaller `first` gap

## How to resume

```sh
cd /home/ntmucke/pyurbanair
git status                           # confirm branch parallel-multinode-srun, uncommitted

# Diagnostic — when 9840575 completes, identify a passing test:
sacct -j 9840575 --format=JobID,State,Elapsed,ExitCode -X
cat job_scripts/delftblue/out_files/slurm-diag_mpi_launch-9840575.out
# Look for "[Tn] exit=0" preceded by 4 lines of hostname output. Apply that
# invocation pattern to BOTH branches in local_execute.sh, clean stale
# run.*.log files, then submit the real benchmark:
find .temp/outputs -name "run.*.log" -delete
sbatch job_scripts/delftblue/benchmark_parallelism.slurm

# Real benchmark — when it completes:
sacct -j <new_job_id> --format=JobID,State,Elapsed,ExitCode -X
tail -200 job_scripts/delftblue/out_files/slurm-bench_parallelism-<id>.out
tail -50  job_scripts/delftblue/out_files/slurm-bench_parallelism-<id>.err
```

### Decision tree on the benchmark result

- **A_only `steady_mean` < baseline `steady_mean` AND C_only `first` ≈ `steady_mean`**:
  both work → commit. Stage just `local_execute.sh`,
  `base_ensemble_forward_model.py`, the benchmark files, and the diagnostic
  script; **leave `scripts/config.py` and the namoptions modifications alone**
  (those are the user's working state).
- **A works, C doesn't (or marginal)**: commit A. Investigate why C is silent
  (maybe fork is so fast there's no win to capture; check `first - steady_mean`
  per scenario for evidence).
- **C works, A doesn't**: commit C. Diagnose A — most likely `srun --exact`
  failed to actually fan out (look in `.err` for srun warnings; verify
  `SLURM_JOB_NODELIST` shows two nodes; check uDALES output paths land on
  different hostnames).
- **Neither works**: tail `.err` and per-member `run.*.log`; consider the
  fallback approaches in "If it fails again" below.

### If a benchmark FAILS outright

**Always look at `.temp/outputs/<member>/run.<member>.log` first** — Python
`subprocess.run(..., stdout=self.stdout, stderr=self.stderr)` swallows the
actual MPI/uDALES output, so the slurm `.err` only shows the Python
`CalledProcessError` traceback, not the underlying error. The
`tee -a run.$exp.log` line in `local_execute.sh` is the authoritative source.

**Always clean stale `run.*.log` files between attempts** — `tee -a` appends
across jobs and creates ambiguity about which line came from which run.

If the diagnostic shows that **none of T1–T8 pass**, fall back paths in
descending order of preference:
1. Use slurm's PMI directly: `srun --exact -N1 -n$NCPU --cpus-per-task=1 $DA_BUILD ...`
   (skips mpiexec entirely). Not done historically because of suspected
   slurm-MPI PMI handshake issues with conda-forge OpenMPI; the diagnostic
   doesn't cover this but it's the obvious next thing to try.
2. Disable change A entirely and only validate change C — confirms the
   benchmark machinery works end-to-end while we untangle A separately.
3. Wrap each per-member mpiexec in `bash -c` that explicitly unsets all
   `SLURM_*` env vars before invoking mpiexec (effectively T5 / T8 from the
   diagnostic, but applied permanently in `local_execute.sh`).

## Things validated earlier (don't redo)

- Pixi `delftblue` env loads fine on compute-p2 (smoke job 9837254 got past
  imports + prepare + a truth uDALES run + parallel fork + mpiexec + integration
  loop before being cancelled for taking too long at sim_time=60s). The
  multi-second simulation output in `.temp/outputs/000/run.000.log` (lines
  ~1–716) is from this smoke run, not from any benchmark job.
- `PYPALM_SKIP_AUTOINSTALL=1` correctly disables the PALM compile attempt on
  compute nodes (which have no internet AND a broken `kpp4palm` tree on this
  checkout). Any new slurm script that touches pyudales must set this.
- `BaseEnsembleForwardModel` has no `__call__` — must use `.run_ensemble(...)`.
  Benchmark already uses the right method.
- The full-run config in `scripts/config.py` (`ensemble=64, num_parallel=64,
  ncpu=4` → 256 cores) doesn't fit a single compute-p2 node. With A+C, the
  right slurm allocation for the production run is **4 nodes × 16 tasks × 4
  cpus** on compute-p2 with `account=research-ceg-gse` (NOT `innovation`,
  which caps at 64 CPUs/24h/4 jobs).
- compute-p2 has been heavily booked all session (~84/90 nodes allocated).
  Backfill into short walls (≤20 min) is the only reliable way to get a 2-node
  job to run within an hour or two.

## Files modified or added on this branch (uncommitted)

```
M  libs/pyudales/shell_scripts/local_execute.sh         # change A + Layer-1/2/3 fixes
M  src/pyurbanair/base_ensemble_forward_model.py        # change C
A  scripts/benchmark_ensemble_parallelism.py            # benchmark driver
A  job_scripts/delftblue/benchmark_parallelism.slurm    # 4-scenario benchmark, 20-min sizing
A  job_scripts/delftblue/diag_mpi_launch.slurm          # mpiexec invocation diagnostic (job 9840575)
A  scripts/run_smoke_rollout_esmda.py                   # smoke launcher (from earlier)
A  job_scripts/delftblue/smoke_rollout_esmda.slurm
```
plus `M scripts/config.py` (the user's full-run sizing — leave alone, it's
their working state).

Current state of `local_execute.sh` cumulative edits (vs. main):
1. Top of script: prepend `/cm/shared/apps/slurm/current/bin` to PATH if
   `SLURM_JOB_ID` is set and `srun` isn't on PATH.
2. Top of script: `export PRTE_MCA_plm=^slurm`.
3. Both mpiexec invocations: `--host localhost:$NCPU` *(currently in place but
   confirmed to hang — will be replaced based on 9840575 diagnostic results).*
4. Change-A branch: wrap mpiexec in `srun --exact -N1 -n1 --cpus-per-task=$NCPU`.

## Open issues unrelated to A+C

- The `kpp4palm` subtree under `libs/pypalm/palm_model_system/packages/chemistry/kpp4palm/`
  is in a broken state (missing `src/`); a clean re-extract or `git clean -fdx`
  on that subdir would fix the recursive-make hang. Currently masked by
  `PYPALM_SKIP_AUTOINSTALL=1`.
- `pypalm/__init__.py` runs install at import time — even when the user has
  `PALM_ARGS["compile"]=False` in `scripts/config.py`. The two flags should be
  unified, OR the import-time installer should respect a "no compile" hint.
  Filed mentally; not on this branch.
- The `BrokenProcessPool` catch in
  [base_ensemble_forward_model.py:459](src/pyurbanair/base_ensemble_forward_model.py#L459)
  is in the wrong place — wraps `executor.submit` calls, but that exception
  more commonly surfaces from `future.result()`. The retry path won't fire in
  typical worker-death cases. Not a benchmark blocker.

## What to ask the user when you resume

- After reading the benchmark numbers, summarize them for the user before
  committing. They want to be in the loop on whether the speedup is real.
- The full-run intent (`ensemble=64`) still needs a 4-node compute-p2 slurm
  script — once A+C are validated, draft that and confirm with the user before
  submitting.

# Handover: refactor and simplify `job_scripts/`

**For:** the agent doing this work. **Branch:** create one from
`feat/simplified-configs-and-scripts` and open the PR back into that branch
(not into `main`). Read `AGENTS.md` first; its workflow rules apply.

## Why

The repo was refactored into a lean layout: `configs/` (flat Hydra tree),
`scripts/` (the scripts you run, each a `run(cfg)` + thin `@hydra.main`, with
shared helpers in `scripts/utils/`), `workflows/` (shell pipelines) and `tests/`.
The old setup is in `archive/`. `job_scripts/` was not touched. It still calls
the archived runners (`scripts/esmda/run_esmda.py`,
`scripts/run_forward_model.py`, `scripts/neural_surrogate/*`,
`scripts/figure_creation/*`, ...) and `conf/`, so **every job script is broken
right now**. The docs flag this as "not ported yet".

## What the user asked for

- **One job script per script the user runs** in `scripts/`: no per-backend
  or per-experiment copies.
- **Simple.** Experiment settings belong in configs and overrides, not in
  shell.
- **Proposal first:** write the proposal (a tree plus a short rationale, see
  below) and get the user's OK before porting.

## Current state

106 tracked files, about 9.3k lines:

| Where | What |
|---|---|
| `job_scripts/{snellius,delftblue}/` | SLURM scripts per task (`ground_truth.slurm`, `generate_training_data.slurm`, `make_state_small.slurm`, `trim_and_visualize.slurm`, `visualize_run*.slurm`, `figures_*.slurm`, `eval_sweep.slurm`, `run_esmda_test.slurm`, ...), a `common.sh` with shared experiment settings (results/scratch roots, ground-truth dir, domain, windows, dynamic parameters), `submit.sh`, `templates/esmda.slurm`, and per-backend folders `pylbm/`, `pyudales/`, `pypalm/` (`rollout_esmda_from_truth.slurm`, sweep launchers). DelftBlue also has PALM debugging scripts (`pypalm/m0_capture.py`, `m1_direct_run.py`, `m2_smoke.slurm`). |
| `job_scripts/local/` | the same pattern as shell scripts, plus `experiments/` (ESMDA / filter-smoothing experiment drivers and `settings.sh`), filtering benchmarks and `make_*_truth.py` helpers |
| `docs/job_scripts.md`, `job_scripts/**/README.md` | docs (currently marked "not ported yet") |

Read a few to see the conventions, e.g. `job_scripts/snellius/common.sh`, one
`pyudales/rollout_esmda_from_truth.slurm` and `submit.sh`. Note what is
genuinely machine-specific: SLURM headers, `pixi` environment, scratch vs
results storage, MPI/CPU counts, module loads. Note what is experiment config
dressed up as shell: domain, windows, ensemble size, ground-truth paths,
parameter settings.

The current setup gives you:

- **Runnable scripts** (everything in `scripts/` except `scripts/utils/`):
  - `run_forward.py`, `run_smoother.py`, `run_filtering.py`, `run_hybrid.py`;
  - `compute_metrics.py <run dir>`, `visualize_forward.py <run dir>`,
    `visualize_assimilation.py <run dir>`;
  - `surrogate/generate_data.py`,
    `surrogate/train.py --config-name surrogate/<config>`,
    `surrogate/evaluate_{stepper,autoencoder,latent_generator}.py`;
  - `tools/prepare_case_stl.py`, `tools/preprocess_udales_geometry.py`.

  Each module docstring lists its options and outputs. `setup_dev_env.sh` and
  `start_mcp` are infrastructure, not jobs.
- **Machines in the config:** `configs/common.yaml` already has
  `paths.machine: local | snellius | delftblue`, which selects the scratch dir
  (`paths.scratch.*`). `paths.results_root` sets where results go, and
  `ensemble.{ensemble_size, num_parallel_processes, num_cpus_per_process}`
  sets the budget.
- **Pixi environments:** `snellius`, `delftblue`, `cuda` and `dev`, with
  activation scripts in `activation_scripts/`.
- **Shell pipelines:** `workflows/forward_workflow.sh` and
  `workflows/assimilation_workflow.sh <method>` (a run followed by its
  post-processing). A job can call a workflow instead of a single script if
  that keeps things simpler; justify it.

## Proposal first

Propose, with a short rationale, before porting:

- **Layout.** A natural fit is one SLURM script per runnable script and per
  cluster, e.g. `job_scripts/snellius/run_smoother.slurm`. Each holds the
  `#SBATCH` header for that machine, the env setup, and `pixi run -e <env>
  python scripts/<script>.py paths.machine=<machine> "$@"`. Hydra overrides
  then go straight through `sbatch`:

  ```bash
  sbatch job_scripts/snellius/run_smoother.slurm model@assim_model=pylbm ensemble.ensemble_size=64
  ```

  Consider factoring the machine part into one sourced file per machine, so
  each job script is a few lines. Say whether `local/` is needed at all:
  locally you can run the script or a workflow directly.
- **Where the experiment settings go.** The shared `common.sh` settings should
  become overrides or a small config (e.g. a case or params file in
  `configs/`, or a documented override line), not shell variables. Don't add a
  new config mechanism without the user's OK.
- **Sweeps.** The current per-backend sweep launchers and
  `local/experiments/` drivers. Say whether Hydra multirun or a tiny submit
  loop covers them, or whether they can go. Ask before keeping any.
- **What gets dropped.** Figure scripts for archived figure code, the
  `make_state_small` / `trim_spinup` / 32-bit conversion jobs (their scripts
  were not ported), and the PALM debugging scripts. List them and whether each
  is dropped or needs a script first. Don't port a job whose script doesn't
  exist in `scripts/`. Ask the user if one is still needed.

## Apply

- **Archive the old ones:** move the old `job_scripts/` content to
  `archive/job_scripts/` with `git mv`, in the same way the old configs,
  scripts and tests were archived. Pre-commit and pytest already ignore
  `archive/`.
- **Write the new job scripts** per the approved proposal. Keep them minimal:
  header, environment, one command. Comments only where the machine needs a
  non-obvious setting (e.g. the Snellius `/scratch-shared` note in the
  current `common.sh`: pyudales fails under `$TMPDIR`).
- **Guard test:** add `tests/scripts/test_job_scripts.py`. For every runnable
  script in `scripts/` (excluding `scripts/utils/`, `setup_dev_env.sh` and
  `start_mcp`), each cluster has a job script that calls it, and every job
  script passes `bash -n`. This keeps job scripts and scripts from drifting
  apart.
- **Docs:** rewrite `docs/job_scripts.md` (short: layout, how to submit, how
  to pass overrides, machine notes) and replace or remove the per-folder
  READMEs. Update the job-scripts mentions in `README.md`, `AGENTS.md` /
  `AGENTS.md` and `docs/codebase_guide.md`, and drop the "not ported yet"
  notes.
- **Other references:** `grep -rn "job_scripts"` outside `archive/`, e.g.
  `.github/`, `pyproject.toml` and `activation_scripts/`.

## Coordination

Two other PRs run in parallel off the same base:

- the MCP port (`docs/plans/mcp_server_refactor_handover.md`);
- `examples/` → `geometries/` (`docs/plans/geometries_cleanup_handover.md`).

The geometries PR changes case paths. Job scripts shouldn't hard-code
geometry paths at all, because the case config holds them. All three PRs may
touch `README.md`, `AGENTS.md` and `docs/codebase_guide.md`; keep those edits
small so the later merges rebase easily.

## Constraints and gotchas

- **Repo rules (see `AGENTS.md`):**
  - pixi;
  - run `pixi run -e dev pre-commit` before committing;
  - parallel ensembles use `forkserver`;
  - don't scale parallelism blindly (the hardware is DRAM-bandwidth-bound past
    about 4–8 workers);
  - keep docs in sync.
- **Untouchable files:** `configs/assimilation.yaml` and `archive/conf/*.yaml`
  may carry the user's live, uncommitted tuning. Never commit, stash or reset
  them.
- **You cannot run SLURM here.** Verify with `bash -n`, the guard test, and a
  local dry run of the command each job runs (e.g. on the tiny test overlays:
  `python scripts/run_forward.py --config-dir tests/configs +test=forward`).
  State in the PR that the cluster runs are untested.

## Done when

- [ ] The user approved the proposal.
- [ ] Every runnable script has its job script(s). The old ones are in
  `archive/job_scripts/`.
- [ ] No job script hard-codes experiment settings that belong in configs or
  overrides.
- [ ] `tests/scripts/test_job_scripts.py` passes. `pixi run -e dev py.test` and
  pre-commit pass.
- [ ] `docs/job_scripts.md` and the other mentions are updated, and the "not
  ported yet" notes are gone.
- [ ] The PR description shows the final tree, what was dropped and why, and
  that SLURM submission was not tested.

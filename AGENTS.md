# AGENTS.md

Instructions for AI coding agents working in `pyurbanair`. Keep this file lean:
it loads into every session. It holds the stable conventions and a map to the
docs; detail lives in `docs/`.

## Principles

- **Simplicity first.** Prefer the smallest change that solves the problem.
  Delete dead code rather than deprecate it; no compatibility shims, no new
  abstractions, configs knobs or files unless they remove real duplication.
  One obvious place for each thing.
- **Reuse before adding.** Extend the existing scripts, configs, helpers and
  tests in their own style before creating new ones.
- **Match the surrounding code**: comment density, naming, idioms.
- **Ask before** anything hard to reverse or outside the task: deleting data or
  results, force-pushing, editing vendored code, large restructures.

## Layout

```text
configs/    Hydra configs: forward.yaml, assimilation.yaml (entry points);
            groups case/, model/, params/, assimilation_settings/; surrogate/
scripts/    the scripts you run (run_*, compute_metrics, visualize_*,
            surrogate/, tools/); shared helpers only in scripts/utils/
workflows/  shell pipelines: a run followed by its post-processing
geometries/ case inputs: one folder per case (STL, namoptions, _p3d), urbantales/
src/        pyurbanair: base classes every backend inherits
libs/       pylbm, pyudales, pypalm, neural-surrogates, data-assimilation,
            evaluation, visualization, mcp-server (each an editable package)
tests/      one folder per package; tests/scripts/ for scripts, configs, workflows
docs/       reference docs; plans/, research/, archive/ are working notes
archive/    the retired setup: dead code, never edit, import or run it
```

All public I/O is `xarray.Dataset`, NetCDF on disk. Data assimilation (ESMDA,
EnKF, hybrid) is in JAX and never depends on a specific solver: backends plug
in through `BaseForwardModel` / `BaseEnsembleForwardModel`.

Vendored code (`libs/pyudales/u-dales/`, `libs/pypalm/palm_model_system/`,
`libs/pylbm/LBM/`, `neural_surrogates/architectures/_tadpole/` and `_upt/`) is
upstream: don't edit it.

## Read the right doc first

Before editing, read the doc for the area you touch, and only that one.

| If the task touches… | Read first |
|---|---|
| Anything non-trivial (orientation, "add a new X" recipes) | [docs/codebase_guide.md](docs/codebase_guide.md) |
| `configs/`, `scripts/`, `workflows/`, `geometries/` | [docs/scripts_and_configs.md](docs/scripts_and_configs.md), [configs/README.md](configs/README.md), [geometries/README.md](geometries/README.md) |
| LBM (`libs/pylbm`) | [docs/pylbm.md](docs/pylbm.md) |
| uDALES (`libs/pyudales`) | [docs/pyudales.md](docs/pyudales.md) |
| PALM (`libs/pypalm`) | [docs/pypalm.md](docs/pypalm.md) |
| Data assimilation (`libs/data-assimilation`) | [docs/data_assimilation.md](docs/data_assimilation.md) |
| Neural surrogates (`libs/neural-surrogates`, `scripts/surrogate/`) | [docs/neural_surrogates.md](docs/neural_surrogates.md) |
| Metrics and figures (`libs/evaluation`) | [docs/evaluation.md](docs/evaluation.md) |
| MCP server (`libs/mcp-server`) | [docs/mcp.md](docs/mcp.md) |
| HTML forward-run viewer (`libs/visualization`) | [docs/visualization.md](docs/visualization.md) |
| HPC jobs (`job_scripts/`) | [docs/job_scripts.md](docs/job_scripts.md) |
| Tests | [tests/README.md](tests/README.md) |

Only the top-level `docs/*.md` are maintained references. `docs/plans/` holds
open plans (finished ones move to `plans/implemented/` or `plans/rejected/`);
`docs/research/` and `docs/archive/` are notes and history. Verify any of these
against the code before relying on them.

## Commands

Everything runs through [Pixi](https://pixi.sh) in the `dev` environment.
Always go through `pixi run` (or `pixi shell`): calling `.pixi/envs/*/bin/python`
directly skips the activation that sets up the compilers for the solver builds.

```bash
pixi run setup-dev                                   # one-time install
pixi run -e dev py.test                              # all tests without compiled solvers
pixi run -e dev python -m pytest tests/<package>     # one package
pixi run -e dev test-integration                     # tiny real-solver runs
pixi run -e dev pre-commit                           # black, isort, mypy on staged files

pixi run -e dev python scripts/run_forward.py model=pylbm forward.ensemble=true
bash workflows/assimilation_workflow.sh smoother     # run + metrics + figures
# a tiny run for quick manual checks (the test overlays):
pixi run -e dev python scripts/run_forward.py --config-dir tests/configs +test=forward
```

Each script's module docstring lists its options and outputs.

## Rules

- **Branch first.** Never commit to `main`; branch, commit, open a PR.
  `main` is unprotected, so be deliberate.
- **Run `pixi run -e dev pre-commit` before committing.** Use
  `# type: ignore[...]` only for checks you deliberately don't satisfy.
- **Scripts:** `def run(cfg)` plus a thin `@hydra.main` wrapper (so tests can
  call `run`), `check_config(cfg, ...)` from
  `scripts/utils/inconsistency_check.py` first, outputs under
  `cfg.paths.results_dir`. Shared helpers go in `scripts/utils/`.
- **No-op when absent.** A new parameter or knob must leave default runs
  byte-identical: read it, and skip its write site when it isn't set (recipe in
  `docs/codebase_guide.md`).
- **Tests use their own configs.** Compose the real configs made tiny by an
  overlay from `tests/configs/` (`compose("forward", "+test=forward",
  root=tmp_path)`), and set every value a test depends on there. Never rely on
  the current values in `configs/`, which change between runs. Mark tests that
  run a compiled solver `integration`.
- **Parallel ensembles use `forkserver`, not `fork`** (JAX threads + fork
  deadlock). Don't change the multiprocessing context.
- **Don't scale parallelism blindly.** The hardware is DRAM-bandwidth-bound
  past ~4–8 workers; benchmark before raising `ensemble.num_parallel_processes`.
- **No large artifacts in git**: `.temp/`, ground truths, model weights, SLURM
  logs, generated figures stay out.
- **Keep docs in sync.** If a change moves files, renames configs or alters a
  documented contract, update the doc in the same PR. Record durable,
  non-obvious findings in the matching doc, not in this file.

## Before you finish

- Run the tests of every package you touched, plus `tests/scripts` if you
  touched `scripts/`, `configs/` or `workflows/`. Run `pre-commit`.
- Run the relevant `integration` tests if you changed solver orchestration;
  otherwise say they weren't run.
- Report what you changed, what you verified, and what you couldn't verify.

## Environment notes

- macOS: `import torch` can abort on an OpenMP clash; set
  `KMP_DUPLICATE_LIB_OK=TRUE`.
- `tests/pyudales/test_udales_discrepancy_native.py` fails to compile its
  kernel with the macOS gfortran; CI (Linux) runs it.

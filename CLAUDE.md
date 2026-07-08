# CLAUDE.md

Instructions for AI coding assistants working in `pyurbanair`. Keep this file
lean — it loads into every session. Deep detail lives in `docs/` (see the
routing table below); evolving gotchas live in auto-memory. This file holds the
stable conventions and the map to everything else.

## Read the right doc before you implement

This repo is large. **Before editing, identify the area you're touching and read
that doc first.** Do *not* read docs for areas you aren't working on (e.g. skip
the surrogate docs for a pure LBM change).

| If the task touches… | Read first |
|---|---|
| Anything non-trivial — start here for orientation | [docs/codebase_guide.md](docs/codebase_guide.md) |
| The LBM backend (`libs/pylbm`, `conf/model/pylbm.yaml`) | [docs/pylbm.md](docs/pylbm.md) |
| The uDALES backend (`libs/pyudales`) | [docs/pyudales.md](docs/pyudales.md) |
| The PALM backend (`libs/pypalm`) | [docs/pypalm.md](docs/pypalm.md) |
| ESMDA / observation operator / localization (`libs/data-assimilation`) | [docs/data_assimilation.md](docs/data_assimilation.md) |
| Neural surrogates (`libs/neural-surrogates`) | [docs/neural_surrogates.md](docs/neural_surrogates.md) |
| Hydra configs (`conf/`) or scripts (`scripts/`) | [docs/scripts_and_configs.md](docs/scripts_and_configs.md) |
| Running jobs on HPC (Snellius / DelftBlue / local) | [docs/job_scripts.md](docs/job_scripts.md) |

`docs/codebase_guide.md` is the entrypoint and has its own finer-grained
documentation map plus the "adding a new X" recipes. When in doubt, start there.

> `docs/plans/`, `docs/temp/`, and `docs/domain_decomposition_surrogate/` are
> working notes / design records, **not** maintained references. Read them for
> theory or history, but verify against the code before relying on them, and
> don't update them as if they were current docs.

## What this repo is (one screen)

A Python monorepo for urban-airflow CFD ensembles and ensemble data assimilation
(ESMDA). It wraps Fortran/learned CFD solvers behind one Python interface and
runs them in ensembles for parameter / state estimation.

- **Backends** (each an editable lib under `libs/`): `pylbm` (Lattice
  Boltzmann), `pyudales` (uDALES), `pypalm` (PALM), and `neural-surrogates`
  (learned drop-in).
- **`pyurbanair`** (`src/`) holds the base classes every backend inherits
  (`BaseForwardModel`, `BaseEnsembleForwardModel`). ESMDA never depends on a
  specific solver — polymorphism is through these base classes.
- **`data-assimilation`** implements ESMDA in JAX.
- All public I/O is `xarray.Dataset`; on-disk format is NetCDF.
- Run-time config is a Hydra tree under `conf/` with two self-contained entry
  points: `run_forward_model.yaml` and `run_esmda.yaml`.

## Commands

Everything runs through [Pixi](https://pixi.sh); the `dev` environment carries
all backends and dev tools.

```bash
pixi run setup-dev              # one-time bootstrap (handles a known bin/test clobber)
pixi shell -e dev               # activate the dev env
pixi run -e dev py.test         # run the test suite (--exitfirst; smoke-shaped, fast)
pixi run -e dev pre-commit      # black + isort + mypy on staged files
```

- Tests compose Hydra configs and call each script's `run(cfg)` directly; a
  tiny "smoke shape" (small domain / short window / 2-member ensemble) keeps
  them fast. See `tests/conftest.py` (`compose_test_cfg`, `_SMOKE_OVERRIDES`).
- Forward runs: `python scripts/run_forward_model.py model=pylbm ...`
- Assimilation: `python scripts/esmda/run_esmda.py ...` (the single ESMDA entry point;
  mode = `esmda/smoother` × `params@prior_params` × `esmda.num_assimilation_windows`).
- Sequential filtering (EnKF): `python scripts/filtering/run_filtering.py ...`
  (mode = `filtering.mode=state|parameter|joint` × the `filtering/*` groups).

## Workflow rules

- **Branch first.** Never commit directly to `main`. Create a branch, commit,
  open a PR. (Main is unprotected and pre-commit is not enforced server-side —
  be deliberate.)
- **Run `pixi run -e dev pre-commit` before committing.** It formats and
  type-checks staged files. Use `# type: ignore[...]` for the rare check you
  don't want to satisfy.
- **Match the surrounding code** — comment density, naming, idioms. New scripts
  follow the `def run(cfg)` + thin `@hydra.main` wrapper shape so they stay
  testable; resolve output paths via `resolve_output_dir(...)`.
- **No-op when a param/field is absent.** Backends must stay byte-identical on
  single-model / default runs when you add a new parameter or knob — read it,
  and skip the write site if it isn't present (see the "adding a new parameter"
  recipe in `docs/codebase_guide.md`).
- **Parallel ensembles use `forkserver`, not `fork`** (JAX threads + bare fork =
  deadlock). Don't change the mp context.
- **Don't blindly scale parallelism.** This hardware is DRAM-bandwidth-bound past
  ~4–8 workers; re-benchmark before raising `ensemble.num_parallel_processes`.
- **Don't commit large artifacts.** `.temp/`, `ground_truth*`, model weights,
  and SLURM logs are scratch/gitignored — keep them out of commits.
- **Keep docs in sync.** If a change moves files, renames configs, or alters a
  contract documented in `docs/`, update the relevant doc in the same PR.

## Memory & conventions

This repo also has **auto-memory** (debugging insights, scaling findings,
backend gotchas) that loads automatically — this file deliberately does *not*
duplicate it. If you discover a durable, non-obvious fact while working, prefer
recording it in memory (or the matching `docs/` file) over inflating this file.
Reserve CLAUDE.md for stable, team-wide conventions and the doc map above.

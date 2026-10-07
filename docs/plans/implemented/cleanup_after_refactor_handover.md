# Handover: cleanup after the lean refactor

Small leftovers from the lean refactor (PR #153). Nothing here may change a
result: default runs stay byte-identical. Read `AGENTS.md` first.

Branch from `main`, open the PR into `main`, don't merge it.

## 1. pixi deprecations

pixi 0.81 prints two warnings on every command:

- **`[tool.pixi.feature.cuda.system-requirements]`** (`pyproject.toml`, the
  `cuda = "12.6"` table) is deprecated "in favor of virtual packages on
  `platforms`". The `cuda` feature already has its own
  `platforms = ["linux-64"]`; move the CUDA requirement onto that entry in the
  form the pixi docs give (e.g. `{ platform = "linux-64", cuda = "12.6" }`), and
  keep the comment explaining why it is needed (the `__cuda` virtual package for
  the CUDA pytorch builds).
- **Lock format v6:** pixi offers to upgrade `pixi.lock` to v7.

Both may need a newer pixi than the current floor, `requires-pixi = ">=0.63"`
in `[tool.pixi.workspace]`. Find the first pixi release that supports each
(pixi changelog), and raise the floor to that version if needed. The floor
exists so that a fresh `pixi install` never rewrites `pixi.lock`; keep that
true. If the floor rises, say so in `README.md` (`pixi self-update`) and check
that `job_scripts/*/env.sh` and `activation_scripts/` don't pin an older pixi.

Verify:
- no deprecation or lock-format warning from `pixi run -e dev python -c 1`;
- `pixi lock` is a no-op afterwards, and `pixi install` on a fresh clone leaves
  `pixi.lock` unchanged;
- every environment still solves for its platforms (`pixi lock` does that):
  `dev`, `cuda`, `snellius`, `delftblue`, `mcp`, `rendering`;
- the `cuda` env solves to the same pytorch / jax builds as before (compare
  `pixi list -e cuda` before and after).

## 2. Stale references to the archived setup

Comments and error messages still name the archived scripts and configs:
`scripts/esmda/run_esmda.py`, `run_esmda.yaml`, `scripts/esmda/_esmda_common.py`,
`scripts/filtering/run_filtering.py`, `run_forward_model`, `conf/`. Find them:

```bash
git grep -nE "scripts/(esmda|filtering|hybrid)/|run_esmda|_esmda_common|run_forward_model\b|conf/" \
  -- configs scripts src libs/*/src tests ':!archive'
```

Known places: `configs/assimilation.yaml`, `configs/case/xie_and_castro.yaml`,
`configs/params/{dynamic,dynamic_truth,static,static_truth}.yaml`; in `libs/`:
`data_assimilation/filter_smoothing/base.py`, `data_assimilation/smoothing/esmda.py`,
`evaluation/{figures,scores,sensors,turbulence}.py`,
`pylbm/utils/build_tree_utils.py`.

Point each at what exists now (`scripts/run_smoother.py`, `run_filtering.py`,
`run_hybrid.py`, `configs/assimilation.yaml`, `params_to_estimate` in
`configs/assimilation.yaml`), or delete the sentence when it is only history
("moved in WP0.2 from ..."). Comments and messages only, no code changes.

Leave alone:
- `configs/model/neural_surrogate.yaml` and `neural_surrogates/{forward_model,ensemble_forward_model}.py`:
  their `run_esmda` mentions belong to the `training_data` spin-up, which has
  its own plan (`docs/plans/surrogate_training_data_spinup.md`);
- `docs/*.md` lines that say "the archived `run_esmda.py`": they are correct;
- `archive/`, `docs/archive/`, `docs/research/`, `docs/plans/implemented/`.

## 3. `quiet_jax` before `jax`

`src/pyurbanair/quiet_jax.py` silences the CUDA-fallback noise in the `cuda`
env, but only when imported before `jax`. Today it is imported only in
`scripts/utils/helper_functions.py` (line 29), after `data_assimilation`, which
already imports `jax`; the entry points (`scripts/run_*.py`,
`compute_metrics.py`, `visualize_*.py`, `scripts/surrogate/*.py`) import `jax`
or the libraries first. So it silences nothing.

Fix: make `import pyurbanair.quiet_jax  # noqa: F401` the first import in each
entry point that imports JAX directly or indirectly, and drop it from
`helper_functions.py`. isort would move it below `jax`; add
`force_to_top = ["pyurbanair.quiet_jax"]` to `[tool.isort]` in `pyproject.toml`
rather than `# isort: skip` on every line.

Test (in `tests/scripts/`): for each entry point, a subprocess imports the
script module and checks that `pyurbanair.quiet_jax` was imported before `jax`
(e.g. `sys.modules` key order). Keep it one parametrized test.

## 4. Filtering-benchmark plans on the old configs

`docs/plans/filtering_ensemble_transform_benchmark.md` and
`docs/plans/filtering_state_reduction_benchmark.md` describe their runs as
`conf/run_filtering.yaml` overrides. **Ask the user** whether each is still
wanted. If yes, rewrite the overrides for `configs/assimilation.yaml` and
`scripts/run_filtering.py` (the mapping is in `docs/scripts_and_configs.md`) and
check each override composes (`--cfg job`). If not, move it to
`docs/plans/rejected/`.

## 5. Dead file in `libs/pylbm`

`libs/pylbm/makefile.macos` is not referenced anywhere (it holds a commented
user path). Confirm with `git grep makefile.macos`, then delete it.

## Done when

- [ ] No pixi warnings; `pixi.lock` stable on a fresh clone; all envs solve.
- [ ] The `git grep` in item 2 only hits the files listed under "Leave alone".
- [ ] Every entry point imports `quiet_jax` first; the test passes.
- [ ] The two benchmark plans are updated or moved, per the user.
- [ ] `libs/pylbm/makefile.macos` is gone.
- [ ] `pixi run -e dev py.test` and `pre-commit` pass; CI green on Linux and macOS.
- [ ] This file moved to `docs/plans/implemented/`.

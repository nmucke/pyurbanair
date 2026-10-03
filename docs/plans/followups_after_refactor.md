# Follow-ups after the lean refactor

What is left after PR #153 (the lean refactor and its follow-up PRs #154–#159).
Read `AGENTS.md` first. Keep every change small: no new abstractions, files or
config knobs unless they remove real duplication.

The items split into four PRs, because they differ in risk:

| PR | Items | Changes results? | Who |
|---|---|---|---|
| A. Cleanup | 1–4 | no: default runs stay byte-identical | an agent |
| B. Surrogate `training_data` spin-up | 5 | adds a missing feature | an agent, after a decision |
| C. `stl_solid_mask` fix | 6 | yes: every evaluation metric and figure | an agent, then a re-check of published numbers |
| D. DA review fixes | 7 | yes: assimilation results | research work, led by the user |

A and C are independent and can run in parallel. B touches the DA scripts, so
run it after A to avoid conflicts. D is not a code cleanup; it is listed so
nothing is lost.

## PR A: cleanup (no behaviour change)

### 1. pixi 0.81 deprecations

`pixi` ≥ 0.81 warns on every command:

- `[tool.pixi.feature.cuda.system-requirements]` (`pyproject.toml`) is
  deprecated. Move `cuda = "12.6"` to its replacement in the current pixi
  manifest docs.
- `pixi.lock` is lock format v6 and pixi offers v7. Upgrade only if v7 still
  reads with the `requires-pixi` floor (`>=0.63`); otherwise raise the floor and
  say so in `README.md`. The lock must not change on a fresh `pixi install`
  (this is what `requires-pixi` guards).

Verify: no warnings from `pixi run -e dev py.test --co -q`; `pixi install` on a
fresh clone leaves `pixi.lock` unchanged; CI green on both platforms.

### 2. Stale references to the archived scripts

Comments and error messages still name `scripts/esmda/run_esmda.py`,
`run_esmda.yaml`, `scripts/filtering/run_filtering.py`,
`scripts/esmda/_esmda_common.py` and `conf/`. Point them at the current scripts
(`scripts/run_smoother.py`, `run_filtering.py`, `run_hybrid.py`) and configs
(`configs/assimilation.yaml`), or drop the history. Find them with:

```bash
git grep -nE "scripts/(esmda|filtering|hybrid)/|run_esmda|_esmda_common|conf/" -- configs scripts src libs/*/src ':!archive'
```

Known places: `configs/assimilation.yaml`, `configs/case/xie_and_castro.yaml`,
`configs/model/neural_surrogate.yaml`, `configs/params/{dynamic,dynamic_truth,static,static_truth}.yaml`,
and in `libs/`: `data_assimilation/filter_smoothing/base.py`,
`data_assimilation/smoothing/esmda.py`, `evaluation/{figures,scores,sensors,turbulence}.py`,
`neural_surrogates/{forward_model,ensemble_forward_model}.py`,
`pylbm/utils/build_tree_utils.py`. Comments and messages only; no code changes
in `libs/`. The `docs/*.md` hits that say "the archived `run_esmda.py`" are
correct; leave them. Leave the `training_data` comments in
`configs/model/neural_surrogate.yaml` to PR B.

### 3. `quiet_jax` import order

`src/pyurbanair/quiet_jax.py` must be imported before `jax`.
`scripts/utils/helper_functions.py` imports it after `data_assimilation`,
which already imported `jax`, so the CUDA-fallback noise is not silenced in the
`cuda` environment. Import it first in every entry point (each script in
`scripts/` that imports JAX directly or through `scripts/utils/`), as the module
docstring says. A test in `tests/scripts/` can check that importing each script
loads `pyurbanair.quiet_jax` before `jax` (via `sys.modules` order in a
subprocess).

### 4. Filtering-benchmark plans on the old configs

`docs/plans/filtering_ensemble_transform_benchmark.md` and
`docs/plans/filtering_state_reduction_benchmark.md` describe their runs as
`conf/run_filtering.yaml` overrides. Translate the overrides to
`configs/assimilation.yaml` + `scripts/run_filtering.py` (use the mapping in
`docs/scripts_and_configs.md`), or move a plan to `docs/plans/rejected/` if it
no longer applies. Ask the user which is still wanted.

## PR B: surrogate `training_data` spin-up

`configs/model/neural_surrogate.yaml` defaults to `spinup_source: training_data`
and documents a `training_data_spinup:` block "consumed by run_esmda". No
current script reads that block, so an assimilation with `model=neural_surrogate`
and the defaults fails at the cold start (`ensemble_forward_model.py` raises).
The helpers exist in `neural_surrogates/training_spinup.py`
(`resolve_training_root`, `list_split_samples`, `write_initial_state_files`,
`anchor_prior_params`); the archived `archive/scripts/esmda/run_esmda.py` shows
how they were called. The MCP server already requires an explicit
`initial_state` for this mode (`mcp_server/jobs/preparation.py`).

Decide with the user first:

- **Wire it up (recommended if surrogate DA runs are planned):** one helper in
  `scripts/utils/` that, when `spinup_source == "training_data"`, writes the
  initial states and anchors the prior; call it from `run_smoother`,
  `run_filtering` and `run_hybrid`. Test with the tiny surrogate overlay in
  `tests/configs/`.
- **Or change the default** to `spinup_source: forward_model` and delete the
  `training_data_spinup` block, plus the `training_data` branch if nothing else
  uses it.

Then make `check_config` reject `training_data` without a loader, so it can't
fail late again.

## PR C: `stl_solid_mask` on Xie & Castro

`evaluation.style.stl_solid_mask` (`libs/evaluation/src/evaluation/style.py`,
the `if cz.size < 2: continue` guard) masks nothing on the Xie & Castro STL:
columns inside a building have one z-crossing (the roof; there are no ground
triangles under the cubes), so they are skipped. Measured: 0 instead of 392
solid cells at z = 2 m. Metrics and figures built on it silently include solid
cells.

Fix: apply the parity rule (`count(z_crossings > z) % 2 == 1` per column, no
size guard), plus a test on the Xie & Castro STL. This changes every evaluation
metric and figure on that geometry, so list in the PR which published numbers
(`docs/archive/experiments_report`, decks in `latex/`) need regenerating.

## PR D: DA review fixes

`docs/research/da_review_2026-09/summary.md` lists defects behind good fits
at the assimilated sensors but poor skill at held-out ones. It was written
against the archived setup, so re-check each item against `configs/` first. In
order:
1. observation error R (instrument 0.25 + representation 0.1 now; check it is
   adequate);
2. filter random-walk parameter evolution (now `parameter_evolution: null`);
3. `block_grouping: true` losing knot-wise temporal localization;
4. uDALES `irandom` shared by the truth and all members;
5. periodic case: nudging only above `nnudge_meters`, sensors at z = 2 m;
6. missing metrics: skill vs reference runs, spread–skill over time,
   forecast/analysis sawtooth.

Fix 1–3 before tuning localization, inflation or state reduction.

## Done means

- PR A: default runs byte-identical, no pixi warnings, all CI green on Linux
  and macOS, this file's PR A section removed or the file moved to
  `implemented/` once B–D have their own plans.
- Each PR runs the tests of the packages it touches plus `tests/scripts`, and
  `pre-commit`.

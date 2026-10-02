# Handover: rename `examples/` to `geometries/` and give it a lean structure

**For:** the agent doing this work. **Branch:** create one from
`feat/simplified-configs-and-scripts` and open the PR back into that branch
(not into `main`). Read `AGENTS.md` first; its workflow rules apply.

## Why

The repo was just refactored into a lean layout: `configs/` (flat Hydra
tree), `scripts/` (short `run(cfg)` scripts), `workflows/`, and `tests/` (one
folder per package). The old setup is in `archive/`. `examples/` was left as it
was. It mixes case geometries, per-backend case templates, duplicated STLs,
generator code and gitignored datasets. It should become `geometries/`, with a
structure as simple as the rest.

## What the user asked for

1. **Rename `examples/` to `geometries/`.**
2. **Propose a simpler, leaner structure,** consistent with the other
   refactorings. Write the proposal (a tree plus a short rationale, see
   "Proposal first") and get the user's OK before moving files.
3. **Apply it completely.** Every config, script, library default, test,
   CI file and doc that uses a path in here must use the new structure. No
   reference to `examples/` may remain in live code (outside `archive/`).

## Current state

31 tracked files, about 72 MB, plus gitignored datasets.

| Path | What | Used by |
|---|---|---|
| `examples/xie_and_castro/xie_castro_2008_STL.stl` | the Xie & Castro case STL (domain frame) | `configs/case/xie_and_castro.yaml` `geometry.stl_path` (pylbm, pypalm) |
| `examples/udales/xie_and_castro/{namoptions.300, xie_castro_2008_STL.stl}` | uDALES case template, with a **second copy** of the STL (uDALES reads the STL from its case dir) | `geometry.udales_case_dir` |
| `examples/palm/xie_and_castro/_p3d` | PALM case template | `geometry.palm_case_dir` |
| `examples/lbm/xie_and_castro/geom.STL` | LBM geometry | `libs/pylbm/src/pylbm/forward_model.py:123` **hard-codes** `case_dir=pathlib.Path("examples/lbm")` |
| `examples/barcelona/buildings.stl` (17 MB) | Barcelona case STL | `configs/case/barcelona.yaml` `stl_path` |
| `examples/udales/barcelona/` (15 files, ~55 MB) | uDALES template + **second copy** of `buildings.stl` + precomputed IBM geometry (`facet_sections_*`, `fluid_boundary_*`, `solid_*`, `geom_meta.json`) | `udales_case_dir`, `udales_precomputed_geom_dir` |
| `examples/palm/barcelona/_p3d` | PALM template | `palm_case_dir` |
| `examples/udales/{idealized,realistic}/namoptions.300`, `examples/palm/{idealized,realistic}/_p3d` | templates for the random-geometry training data (one per UrbanTALES source) | `configs/surrogate/generate_data.yaml` `data.geometry.{udales,palm}_case_dir` |
| `examples/geometries/{download_urbantales_geometries.py, rasters_to_stl.py, README.md}` | **code**: download UrbanTALES and convert rasters to STLs | run by hand |
| `examples/geometries/{raw,processed}/` | **gitignored** datasets those scripts produce; `processed/<source>/` is the STL pool | `generate_data.yaml` `data.geometry.stl_dir` |
| `examples/benchmark_geometry/{benchmark_geometry.py, benchmark_geometry_utils.py}` | **code**: the Xie & Castro geometry generator (CLI) | pixi feature `benchmark_geometry` (check `pyproject.toml`) |

Smells to fix: duplicate STLs, backend-first nesting (`examples/<backend>/<case>`)
next to case-first nesting (`examples/<case>`), code living among data, a library
default path into this folder, and stray `.DS_Store` files.

## Proposal first

Before moving anything, write the proposal for the user: the target tree, one
line per folder, and what moves where and why. Things to decide and justify:

- **One folder per case,** e.g. `geometries/<case>/` holding the case STL and
  each backend's template (`udales/`, `palm/`, `lbm/`). A case would then be one
  folder, and `configs/case/<case>.yaml` would point at it with one root path
  plus fixed sub-paths.
- **One copy of each STL.** Check how uDALES reads its STL (`stl_file` in
  `namoptions`, see `scripts/surrogate/generate_data.py` `_udales_case`, which
  already copies an STL into a case copy). Check whether the pyudales forward
  model can copy or link the case STL into its working dir, so the template
  doesn't need its own copy. Don't change solver numerics: runs must stay
  byte-identical.
- **Code out of the data folder.** The UrbanTALES download/convert scripts and
  the benchmark geometry generator probably belong in `scripts/tools/` (where
  `prepare_case_stl.py` and `preprocess_udales_geometry.py` already are), or in
  a lib if they are imported.
- **Datasets:** where the gitignored UrbanTALES `raw/` and `processed/` pools
  and the random-geometry templates (`idealized`, `realistic`) live, e.g.
  `geometries/urbantales/`.
- **The LBM default:** `pylbm`'s hard-coded `examples/lbm` should become a
  constructor argument fed from the case config, like the other backends' case
  dirs, unless the LBM template turns out to be unused.
- **Large files:** say whether the ~70 MB of Barcelona files should stay
  committed as is. Don't add Git LFS without the user's OK.

Keep it as simple as the rest of the refactor: fewer folders, one obvious place
for each thing.

## Apply it everywhere

Use `git mv` so history follows. Then find every reference:

```bash
grep -rn "examples" --include='*.py' --include='*.yaml' --include='*.yml' \
  --include='*.sh' --include='*.slurm' --include='*.toml' --include='*.md' \
  --include='*.json' --include='.gitignore' . \
  | grep -v "^./archive/\|^./.pixi/\|^./libs/pyudales/u-dales/\|^./libs/pypalm/palm_model_system/"
```

The excluded paths are the archive and the vendored solver sources (their own
`examples/` are unrelated). Known places:

- **Configs:** `configs/case/xie_and_castro.yaml` and `configs/case/barcelona.yaml`
  (the `geometry` block and its comments); `configs/surrogate/generate_data.yaml`
  (`data.geometry.stl_dir`, `udales_case_dir`, `palm_case_dir`).
- **Test configs:** `tests/legacy/conf/case/*.yaml`,
  `tests/legacy/conf/neural_surrogate/training_data.yaml`,
  `tests/legacy/conf/sensor_layout.yaml`; check `tests/configs/` too.
- **Scripts:** `scripts/tools/prepare_case_stl.py`,
  `scripts/tools/preprocess_udales_geometry.py` (defaults, help texts,
  docstrings). `preprocess_udales_geometry.py` still composes the archived
  `conf/` (`config_name="run_forward_model"`, lines ~43 and ~70), so it fails
  as is. Port it to `configs/forward.yaml` in this PR: it produces the
  precomputed uDALES geometry that lives in `geometries/`.
- **Libraries:** `libs/pylbm/src/pylbm/forward_model.py:123`; grep the other
  `libs/*/src`.
- **Tests:** `tests/pyurbanair/test_model_error_parameters.py`,
  `tests/pyurbanair/test_forward_preparation.py`,
  `tests/pyurbanair/test_forward_input_safety.py`,
  `tests/pypalm/test_palm_inlet_turbulence.py`,
  `tests/pypalm/test_pypalm_nudging_driver.py`,
  `tests/pylbm/test_pylbm_build_tree.py`.
- **Other:** `.gitignore` (lines ~35–43); `pyproject.toml` (the
  `benchmark_geometry` feature, if it names a path);
  `job_scripts/delftblue/pypalm/{m0_capture,m1_direct_run}.py`.
- **Docs:** `README.md` (layout section), `AGENTS.md` / `AGENTS.md` if they
  mention it, `docs/codebase_guide.md`, the backend docs, and the moved
  `geometries/` README.

Add a guard test in `tests/scripts/test_configs.py`: for every case, every
geometry path the composed config points at exists. This keeps a future move
from breaking the configs silently.

## Coordination

The MCP port (`docs/plans/mcp_server_refactor_handover.md`) runs in parallel on
another branch off the same base. Both PRs may touch `docs/codebase_guide.md`,
`README.md` and `tests/pyurbanair/`. Keep your edits there small and focused,
so whichever merges second rebases easily.

## Constraints and gotchas

- **Repo rules (see `AGENTS.md`):**
  - pixi `dev` env;
  - run `pixi run -e dev pre-commit` before committing;
  - backends stay byte-identical on default runs;
  - keep docs in sync in the same PR;
  - never commit large generated artifacts (the gitignored `raw/` and
    `processed/` pools stay gitignored at their new path).
- **Untouchable files:** `configs/assimilation.yaml` and `archive/conf/*.yaml`
  may carry the user's live, uncommitted tuning. Never commit, stash or reset
  them. Don't edit `archive/` at all.
- **Known local failures on the user's Mac:**
  - `import torch` aborts unless `KMP_DUPLICATE_LIB_OK=TRUE`;
  - LBM compilation aborts, so pylbm runs only on CI;
  - `tests/pyudales/test_udales_discrepancy_native.py` fails to compile its
    kernel.
- **MCP tests are skipped:** some test modules carry `pytest.mark.skip("MCP
  port pending ...")`; leave the markers, but update their paths.
- **Local integration runs:** uDALES works locally. Check at least
  `pixi run -e dev test-integration` for the uDALES runs on the moved case
  templates, and one uDALES forward run on `case=barcelona` (it uses the
  precomputed geometry).

## Done when

- [ ] The user approved the proposed structure, and it is applied with `git mv`.
- [ ] `examples/` is gone. The grep above finds no live reference to
  `examples` (outside `archive/` and vendored solver sources).
- [ ] No duplicated geometry files remain, unless the proposal justified one.
- [ ] The new geometry-path test passes. `pixi run -e dev py.test` passes,
  and so do the uDALES integration runs. Pre-commit passes.
- [ ] The docs describe the new structure, including the `geometries/`
  README and the README's layout section.
- [ ] The PR description shows the final tree, what moved where, and what
  wasn't run (e.g. LBM and PALM on real geometries).

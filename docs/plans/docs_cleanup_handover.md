# Handover: clean up `docs/`

**For:** the agent doing this work. **Branch:** create one from
`feat/simplified-configs-and-scripts` and open the PR back into that branch
(not into `main`). Read `AGENTS.md` first; its workflow rules apply.

## Why

The code was refactored into a lean layout: `configs/`, `scripts/` (with
helpers in `scripts/utils/`), `workflows/`, `tests/` and `libs/`. The old setup
is in `archive/`. `docs/` still mixes maintained references with plans,
research notes, dated reviews and stale material, spread over five
"not maintained" folders. Several docs describe archived code, and links
point at moved files. The docs should get the same lean treatment as the code.

## Target structure (from the user)

```text
docs/
  <main docs>   maintained reference for implemented code: one per lib, plus
                scripts/configs, workflows and other things directly tied to
                code. Accurate to the current code.
  plans/        open plans only: work not done yet (including the handovers
                for PRs in flight)
  research/     research notes, method/math write-ups, reviews and paper
                sources: useful context, not a code reference
  archive/      finished plans and anything superseded or historical
```

**Rule of thumb:** a doc that describes code as it is goes in the main docs.
Work still to do goes in `plans/`. Ideas, maths and analysis not tied to the
code's current behaviour go in `research/`. Everything done or outdated goes in
`archive/`. After this PR there are no other folders: `temp/`,
`neural_surrogate_plans/`, `da_review_2026-09/` and
`domain_decomposition_surrogate/` are gone, their content filed into the four
places above.

## Proposal first

Before moving anything, write the classification for the user: one table row
per doc, giving its current path, its target (main / plans / research /
archive), the action (keep, move, merge into X, rewrite section Y) and a
one-line reason. Get the user's OK, then apply. Keep the main set small. Merge
short or overlapping docs into the doc of the code they describe, rather than
keeping many small files.

### Current inventory and first-guess classification (verify each)

**Main docs (maintained):**

| Doc | Notes |
|---|---|
| `codebase_guide.md` (857 lines) | orientation and "add a new X" recipes. Stale: `BaseRolloutForwardModel` (deleted), the `hydra_helpers` function list (most deleted), `extract_2d_slice` (deleted), links into `temp/`. |
| `scripts_and_configs.md` (284) | current. It overlaps `configs/README.md` on config keys: make one the source of truth (suggest `configs/README.md` for keys and overrides, linked from here). There is no workflows doc yet: add `workflows/` here or as a short section. |
| `data_assimilation.md` (1441) | Stale mentions: `create_observation_operator` (deleted), old `filtering.*` key names in §8, `conf/` paths. Consider merging in `ensemble_transform_filters.md` (161, reference material). |
| `neural_surrogates.md` (2886) | Part A (data generation, including the training-data sharding section around line 220, which still describes the old plan/simulate/finalize stages) and the CLI/config sections (§1–5, §6, §10, §11, §11b, §19, §23, §29, §32, §38, §39, Part D, §40) describe the archived `scripts/neural_surrogate/*` and `conf/neural_surrogate/*`. Rewrite them against `scripts/surrogate/*` + `configs/surrogate/*`. Keep the library reference parts. |
| `pylbm.md` (753), `pyudales.md` (939), `pypalm.md` (661) | Mostly current. Stale: "Called by `BaseRolloutForwardModel`" (pylbm.md:171, pyudales.md:898), archived paths, `temp/` links. |
| *(missing)* | **`evaluation`** (`libs/evaluation`) has no doc: add a short one. |
| `mcp.md` (269), `visualization.md` (213), `job_scripts.md` (108) | freshly rewritten by the merged MCP (#154) and job-scripts (#155) PRs: current. Keep them; only fix links and consistency. |
| `geometries/README.md` (outside docs/) | written by the geometries PR (#156): current. Link it from the doc map. |
| `config_setup_spec.md` (151) | the old-to-new config mapping. The refactor and the PRs that used it are merged → **archive/**. |

**Plans, research and archive candidates:**

| Doc | First guess |
|---|---|
| `plans/{mcp_server_refactor,job_scripts_refactor,geometries_cleanup}_handover.md` | done (#154, #155, #156 merged) → **archive/** |
| `plans/docs_cleanup_handover.md` (this one) | stays in **plans/** while you work; move it to **archive/** as the PR's last step |
| `plans/local_forward_mcp.md`, `plans/local_forward_visualization.md` | the original MCP/viewer plans, implemented → **archive/** |
| other `plans/*.md` and `plans/esmda_evaluation/*` | check each: implemented → **archive/**; still open → stay. E.g. `udales_inlet_turbulence.md` says "IMPLEMENTED"; `config_structure_proposal.md` is implemented. |
| `plans/references/forward_viewer.original.html.txt` | reference for the implemented viewer → **archive/** |
| `neural_surrogate_plans/00–07` | 01–03 and 07 implemented → **archive/**; the open proposals (04, 05, 06) → **plans/**; `06_implementation_review.md` → research or archive |
| `temp/*` (5 files) | the benchmarks and reviews → **research/** or **archive/**; `rank_histogram_math.md` → research (or fold into the evaluation doc) |
| `da_review_2026-09.md` + `da_review_2026-09/` (5) | dated review → **research/** (keep as one folder) or **archive/** |
| `data_assimilation_recommendations.md` (221) | → **research/**, or merge the still-relevant parts into `data_assimilation.md` |
| `multi_geometry_surrogate_research.md` (386) | → **research/** |
| `domain_decomposition_surrogate/` (`main.tex`, `references.bib`) | paper source → **research/** (domain decomposition is discontinued in the configs) |
| `config_setup.md` (86) | the user's original refactor brief → **archive/** |
| `archive/*` (48 files, incl. `experiments_report/`) | stays archived; move any true research write-up to `research/` |
| untracked: `sgs_discrepancy_mathematical_note.pdf`, `filter_smoothing/` (a PDF), PNGs under `archive/experiments_report/` | not in git; leave them and mention them |

## Apply

- **Move with `git mv`,** so history follows.
- **Fix every link:** relative links in all `*.md`, plus the routing tables in
  `AGENTS.md`, `README.md` ("Documentation" table) and `codebase_guide.md`
  (doc map).
  - Code comments that point at moved or deleted docs need fixing too. There
    are about 30, e.g. `docs/esmda_model_error_parameters.md` (18 refs),
    `docs/training_data.md`, `docs/reduced_state_da.md`,
    `docs/figure_specs.md`, `docs/model_zoo.md` (doesn't exist). They're in
    `configs/params`, `configs/assimilation_settings`, `src/pyurbanair`,
    `scripts/tools`, `tests/` and `libs/*`.
  - In `libs/` change comments only: no code refactoring there.
- **Link check:** add a small test (e.g. in `tests/scripts/test_configs.py`, or
  `tests/test_docs.py` if it reads better) that resolves every relative
  Markdown link under `docs/`, `README.md`, `AGENTS.md`, `configs/` and
  `tests/` and fails on a missing target. Skip `docs/archive/`. Keep it to a
  few lines, so links can't silently rot again.
- **Rewrite the stale sections** listed above against the current code. Read
  the code and the scripts' docstrings; don't copy from the archived docs.
  Keep each main doc's style. Shorter is better: delete what is no longer true
  rather than annotating it.
- **Add short READMEs** for `plans/`, `research/` and `archive/` (2–3 lines
  each, stating the folder's rule). Update `AGENTS.md`'s note on which folders
  are not maintained references.

## Coordination

This is the last PR of the refactor: the MCP port (#154), the job scripts
(#155) and `examples/` → `geometries/` (#156) are merged, so no other PR touches
docs now and every doc is yours. The docs those PRs wrote (`mcp.md`,
`visualization.md`, `job_scripts.md`, `geometries/README.md`, and their edits to
`README.md`, `AGENTS.md`, `codebase_guide.md`) describe the current code:
build on them rather than rewriting them.

## Constraints

- **Repo rules (see `AGENTS.md`):** run `pixi run -e dev pre-commit` before
  committing; no large artifacts (keep the untracked PDFs and PNGs out of git).
- **Off-limits:** don't edit `archive/` at the repo root (the archived code).
  `docs/archive/` is yours. Never commit, stash or reset
  `configs/assimilation.yaml` or `archive/conf/*.yaml` (the user's live,
  uncommitted tuning).
- **Lean:** fewer, accurate docs beat many partial ones. If a doc's content is
  fully covered elsewhere, archive it rather than keeping it in sync.

## Done when

- [ ] The user approved the classification.
- [ ] `docs/` has only the main docs plus `plans/`, `research/` and
  `archive/`, each sub-folder with its README.
- [ ] The main docs match the current code: no references to deleted or
  archived code outside clearly marked historical notes. There is one doc per
  lib, including a new evaluation doc, plus scripts/configs/workflows.
- [ ] The link-check test passes. `pixi run -e dev py.test` and pre-commit pass.
- [ ] `AGENTS.md`, `README.md` and `codebase_guide.md` route to the right docs.
- [ ] The PR description contains the final classification table and lists
  the sections that were rewritten.

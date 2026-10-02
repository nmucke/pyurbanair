# Handover: port the MCP server to the refactored setup

**For:** the agent doing this work. **Branch:** create one from
`feat/simplified-configs-and-scripts` and open the PR back into that branch
(not into `main`). Read `CLAUDE.md` first; its workflow rules apply.

## Why

The repo is mid-refactor. The old `conf/`, `scripts/` and `tests/` are being
replaced by `configs_new/`, `scripts_new/` (with the shell workflows in
`workflows/`) and `tests_new/`. The old ones are retired once the refactor is
done. The MCP server still runs on the old setup and has to move over.

## Ground rules (from the user)

- **Only the new setup.** The MCP server and everything it uses read
  `configs_new/` and run `scripts_new/`. Nothing in the result may read
  `conf/`, import `scripts/`, or depend on `tests/`.
- **Reuse, don't reinvent.** Use the existing new scripts and workflows:
  - `scripts_new/run_forward.py` for the run;
  - `scripts_new/visualize_forward.py` where its figures fit;
  - `scripts_new/inconsistency_check.py` for config checks;
  - `workflows/forward_workflow.sh` as the reference for what a forward run
    plus post-processing is.

  Define a new script or workflow only if one is genuinely needed, and say why
  in the PR. When something is missing, extend the existing script in its own
  style instead.
- **Forward runs only.** The MCP server handles forward-model runs.
  Assimilation (smoother, filter, hybrid) comes in a later PR: don't add
  tools, configs or code paths for it.
- **New tests go in `tests_new/`.** Never add to or rely on `tests/`.
- **Same functionality.** Keep the tools, run lifecycle and rendering. Change
  behaviour only for something clearly better, and list it in the PR.

## Scope

1. Port the MCP server to `configs_new/` and `scripts_new/` (forward runs).
2. Move the HTML visualization into a new lib, `libs/visualization`.
3. Rename the lib to `mcp-server` and move `src/pyurbanair/jobs/` into it.
4. Update the tests, in `tests_new/`.
5. Add a `pixi run -e mcp register-claude` task that registers the server
   with Claude Code.

## Current state (read these first)

| Where | What | Size |
|---|---|---|
| `libs/mcp_server/src/pyurbanair_mcp/` | thin MCP adapter: `server.py` (tool registration, SDK v2), `tools.py`, `schemas.py`, `__main__.py`. Dist name `pyurbanair-mcp`. | 461 lines |
| `src/pyurbanair/jobs/` | everything behind the tools: `preparation.py` (plan validation, code identity / fingerprints), `supervisor.py` + `registry.py` (SQLite queue, Unix socket), `worker.py`, `native.py`, `results.py`, `paths.py`, `processes.py`, `rendering_environment.py` | ~2.3k lines |
| `src/pyurbanair/workflows/forward.py` | the forward engine the MCP worker runs today: per-window persistence, `artifact_index.json`, run record, initial state `{path, member, time_index}`, solver-input snapshots, retained-bytes limit | 710 lines |
| `src/pyurbanair/config/composition.py` | composes `conf/run_forward_model.yaml`, lists options, validates `_target_`s / resolvers / overrides | ~200 lines |
| `src/pyurbanair/visualization/` | `render.py`, `render_3d.py`, `data.py` (reads `artifact_index.json`, falls back to a single `state.nc`), `assets.py` (loopback asset server), `web/` (`index.html`, `viewer.js`, `probe_charts.js`, `viewer.css`) | ~1.5k lines |
| `scripts/start_mcp` | launcher clients register (absolute path); `--check` is a read-only readiness check | |
| `conf/visualization/` | render presets `quicklook.yaml`, `flow_3d.yaml` | |
| `pyproject.toml` | feature `mcp` (pypi dep `pyurbanair-mcp`, task `start-mcp`); env `mcp = [mcp, dev, py312, cpu]`; env `rendering` (PyVista/VTK, ffmpeg) | |
| `docs/mcp.md`, `docs/forward_visualization.md` | user docs for both | |

The new setup the server must use:

| Where | What |
|---|---|
| `configs_new/forward.yaml` | forward entry config (`forward.ensemble`, `forward.rollout_steps`, `forward.initial_state`), plus `common.yaml` and the `case/`, `model/` and `params/` groups |
| `scripts_new/run_forward.py` | `run(cfg)`: runs `1 + forward.rollout_steps` windows; writes `config.yaml`, `state.nc` and `params.nc` into `paths.results_dir` |
| `scripts_new/visualize_forward.py` | `run(run_dir)`: the matplotlib figures of a forward run |
| `scripts_new/inconsistency_check.py` | `check_config(cfg, "forward")`: fast config validation |
| `workflows/forward_workflow.sh` | run, then visualize, from the same overrides |
| `docs/config_setup_spec.md` | old to new config mapping (e.g. `run.rollout_steps` → `forward.rollout_steps`) |

Old-setup couplings to remove:

- `tools.py`: requires `conf/run_forward_model.yaml` to accept `--repo-root`.
- `composition.py`: composes `conf/` (`config_name="run_forward_model"`),
  trusts `_target_`s from `conf/**`, and reads `conf/trusted_forward_targets.txt`.
- `preparation.py` `code_identity`: fingerprints `src/pyurbanair`, `conf/`,
  `scripts/_common.py` and `scripts/run_forward_model.py`.
- `jobs/worker.py`: runs `pyurbanair.workflows.forward.run`, which imports
  from `scripts._common`.

## Tasks in detail

### 1. Port to configs_new / scripts_new

- **Engine:** the worker runs `scripts_new/run_forward.py`'s `run(cfg)`, not
  `src/pyurbanair/workflows/forward.py`.
  - List what the MCP needs that `run_forward.py` doesn't do yet. Likely
    candidates:
    - writing each window as it finishes, so a cancelled or failed run keeps
      partial results;
    - the `{path, member, time_index}` initial state. `configs_new/forward.yaml`
      already promises it, but `run_forward._initial_state` only takes a path;
    - an index/record of what ran.
  - Add only those, in `run_forward.py`'s style. `CLAUDE.md`'s "no-op when
    absent" rule applies: a plain `run_forward.py` run must stay unchanged.
  - `visualization/data.py` already reads a plain `state.nc` when there is no
    `artifact_index.json`, so prefer dropping the index if nothing else needs it.
- **Old engine:** leave `src/pyurbanair/workflows/forward.py` and
  `scripts/run_forward_model.py` in place. The old `scripts/` still uses them,
  and they're deleted when the old setup is retired. The MCP just stops using
  them.
- **Composition:** compose `configs_new/forward.yaml`.
  - Option lists come from `configs_new/{case,model,params}`.
  - Trusted `_target_`s come from `configs_new/**`.
  - Move the render presets (`conf/visualization/*.yaml`) to
    `configs_new/visualization/`. Move `conf/trusted_forward_targets.txt` to
    `configs_new/` too, if it is still needed.
- **Config check:** call `check_config(cfg, "forward")` during preparation, so
  the MCP rejects what the CLI rejects.
- **Code identity:** fingerprint `configs_new/` and the `scripts_new/` files
  the run imports.
- **Tool parameters and docs:** use the new key names. Update tool
  descriptions and `get_capabilities` (e.g. `total_windows: "1 +
  forward.rollout_steps"`).
- **Backends:** keep all four.
  - The surrogate's `spinup_source: training_data` has no support in
    `scripts_new` (nothing loads the training snapshots as the initial state).
    Under MCP it needs an explicit initial state, as today.
  - `spinup_source: generative` works without caller support.

### 2. `libs/visualization`

- **Package:** follow the other libs' convention. That's `libs/visualization`,
  dist name `visualization`, import `visualization`, hatchling, `src/` layout
  (see `libs/evaluation/pyproject.toml`). Add it as an editable pypi
  dependency in `pyproject.toml` where used (at least `dev`).
- **What moves:** `src/pyurbanair/visualization/` (the renderer, 3D renderer,
  data reader, asset server and `web/` assets). Ship the `web/` files as
  package data.
- **What stays:** matplotlib-only code like `src/pyurbanair/animation.py` and
  the `scripts_new/visualize_*.py` figures.
- **Dependencies:** the lib must not depend on `mcp-server`. MCP depends on it.
- **Optional 3D:** keep PyVista/VTK optional (the `rendering` env).

### 3. Rename to `mcp-server`; move the jobs into it

- **Rename:** `git mv libs/mcp_server libs/mcp-server`, dist name `mcp-server`,
  import package `mcp_server`, matching the convention. The SDK is imported
  as `mcp`; keep that import explicit. If this naming causes a real conflict,
  stop and ask the user.
- **Move the jobs:** move `src/pyurbanair/jobs/` into `mcp_server`
  (e.g. `mcp_server/jobs/`). It's MCP-only infrastructure. Also move
  `src/pyurbanair/config/composition.py` if only the MCP uses it.
- **Update every reference:**
  - `pyproject.toml` feature `mcp`, its tasks and `pixi.lock`;
  - `scripts/start_mcp`'s `python -m`;
  - `[project.scripts]`;
  - `.github/workflows/tests-mcp.yml` (its `paths:` list `libs/mcp_server/**`);
  - `docs/mcp.md`, `docs/codebase_guide.md` and `CLAUDE.md`'s doc table.
- **Server name:** keep the registered server name `pyurbanair`, so existing
  client registrations keep working.

### 4. Tests (all in `tests_new/`)

- **Layout:** `tests_new/` has one folder per package plus `scripts/` (see
  `tests_new/README.md`).
  - The MCP tests are in `tests_new/mcp/` (`test_mcp_protocol.py`, plus
    `test_mcp_forward_integration.py`, which is `integration`).
  - The tests of the code you move are in `tests_new/pyurbanair/`:
    `test_forward_preparation*.py`, `test_forward_plan_identity.py`,
    `test_forward_input_safety.py`, `test_local_jobs.py` and
    `test_forward_visualization*.py`.
- **Moving tests:** move tests with their code.
  - Jobs tests go to `tests_new/mcp/`.
  - Visualization tests go to a new `tests_new/visualization/`, with a CI
    workflow copied from one of `.github/workflows/tests-*.yml`.
  - Update the path filters of every workflow you touch.
- **Port to the new configs:** these tests came from the old suite and use the
  legacy fixtures and configs in `tests_new/legacy/`. Port the ones you touch
  to `configs_new` with overlays from `tests_new/configs/` (`compose("forward",
  "+test=forward", root=tmp_path)` in `tests_new/conftest.py`). Drop what tests
  removed behaviour. Add any overlay you need under `tests_new/configs/`.
- **New features of `run_forward.py`:** test them in
  `tests_new/scripts/test_forward.py`.
- **Old `tests/`:** don't edit it, with one exception. `ci.yml` still runs it,
  and its copies of the tests for the code you move (`tests/test_local_jobs.py`,
  `tests/test_forward_preparation*.py`, `tests/test_forward_plan_identity.py`,
  `tests/test_forward_input_safety.py`, `tests/test_forward_visualization*.py`,
  `tests/test_mcp_*.py`) will break on the moved imports. Delete those old
  copies: the `tests_new/` ones are the maintained versions. Keep the MCP
  protocol step in `ci.yml` working or remove it, since `tests-mcp.yml` covers
  it.
- **mypy:** `.pre-commit-config.yaml` excludes the carried-over library-test
  folders from mypy. Folders you rewrite should pass mypy; remove them from
  that exclude.
- **Running the MCP tests:** they run in the `mcp` env: `pixi run --locked -e mcp
  python -m pytest tests_new/mcp`.

### 5. `pixi run -e mcp register-claude`

- **The task:** add `register-claude` to the `mcp` feature's tasks in
  `pyproject.toml`, backed by a short script (e.g.
  `libs/mcp-server/scripts/register_claude.sh`, or a `mcp_server` entry point).
- **What it does:**
  1. If the `claude` CLI isn't on `PATH`, print how to install it and exit 0.
  2. If `claude mcp get pyurbanair` already succeeds, say so and exit (idempotent).
  3. Ask in the terminal: "Add the pyurbanair MCP server to Claude Code? [y/N]".
  4. On yes, run `claude mcp add --transport stdio --scope user pyurbanair --
     <absolute repo>/scripts/start_mcp`. Resolve the absolute path from the
     script's own location, not the cwd.
- **Non-interactive shells:** without a terminal (`[ -t 0 ]` false), don't
  prompt. Print the command instead.
- **Docs:** put it in `docs/mcp.md`'s install steps: `pixi install -e mcp`,
  then `pixi run -e mcp register-claude`. Keep the Codex / Claude Desktop
  instructions as the manual path.

## Constraints and gotchas

- **Repo rules (see `CLAUDE.md`):**
  - pixi `dev` env;
  - run `pixi run -e dev pre-commit` before committing;
  - parallel ensembles use `forkserver`;
  - backends stay byte-identical when a new knob is absent;
  - keep docs in sync in the same PR.
- **Untouchable files:** `conf/*.yaml` and `configs_new/assimilation.yaml`
  carry the user's live, uncommitted tuning. Never commit, stash or reset them.
- **Known local failures on the user's Mac:**
  - `import torch` aborts unless `KMP_DUPLICATE_LIB_OK=TRUE`;
  - LBM compilation (`prepare_compile`) aborts, so pylbm integration runs only
    on CI;
  - `tests_new/pyudales/test_udales_discrepancy_native.py` fails to compile its
    kernel (fails on a clean HEAD too).
- **Integration tests aren't in CI:** CI runs the default suite only (no
  compiled solver). Run the relevant `integration` tests locally
  (`pixi run -e dev test-new-integration`, uDALES works locally) or note what
  you couldn't run.
- **Sandboxes:** the supervisor uses Unix sockets and loopback; sandboxes may
  block them (see `docs/mcp.md`, "Validation").

## Done when

- [ ] `scripts/start_mcp --check` passes. Over the protocol: list options,
  prepare and launch a forward run on `configs_new`, poll it, inspect
  results, render the HTML view, and cancel a run.
- [ ] The MCP worker runs `scripts_new/run_forward.py`. No MCP, jobs or
  visualization code reads `conf/`, imports `scripts/` or uses
  `pyurbanair.workflows.forward`. Any new script or workflow is justified in
  the PR.
- [ ] `libs/visualization` and `libs/mcp-server` exist with the conventions
  above. `src/pyurbanair/jobs/` and `src/pyurbanair/visualization/` are gone,
  and no reference to `libs/mcp_server` / `pyurbanair_mcp` remains.
- [ ] `pixi run -e mcp register-claude` registers the server after a "y",
  does nothing on a second run, and never prompts without a terminal.
- [ ] All new and moved tests are in `tests_new/` and pass, with their CI
  workflows. `ci.yml` (old suite) still passes. Pre-commit passes.
- [ ] `docs/mcp.md`, `docs/forward_visualization.md` (or a new
  `docs/visualization.md`), `docs/codebase_guide.md`, `CLAUDE.md`'s doc table
  and `tests_new/README.md` are updated.
- [ ] The PR description lists behaviour changes, what `run_forward.py` gained
  and why, and anything not run (e.g. integration tests on backends
  unavailable locally).

# Handover: port the MCP server to the refactored setup

**For:** the agent doing this work. **Branch:** create one from
`feat/simplified-configs-and-scripts` and open the PR back into that branch
(not into `main`). Read `AGENTS.md` first; its workflow rules apply.

## Why

The repo's configs, scripts and tests were refactored. The current setup is
`configs/`, `scripts/`, `workflows/` and `tests/`. The old setup (`conf/`, the
old `scripts/` and the old `tests/`) now sits in `archive/`: dead code, not run
or tested. The MCP server still uses the old setup, so it's broken until this
port. Its test modules are skipped with the reason "MCP port pending".

## Ground rules (from the user)

- **Only the current setup.** The MCP server and everything it uses read
  `configs/` and run `scripts/`. Nothing may read, import or depend on
  `archive/`.
- **Reuse, don't reinvent.** Use the existing scripts and workflows:
  - `scripts/run_forward.py` for the run;
  - `scripts/visualize_forward.py` where its figures fit;
  - `scripts/utils/inconsistency_check.py` for config checks;
  - `workflows/forward_workflow.sh` as the reference for what a forward run
    plus post-processing is.

  Define a new script or workflow only if one is genuinely needed, and say why
  in the PR. When something is missing, extend the existing script in its own
  style instead.
- **Forward runs only.** The MCP server handles forward-model runs.
  Assimilation (smoother, filter, hybrid) comes in a later PR: don't add
  tools, configs or code paths for it.
- **Tests go in `tests/`** (the current suite). Never add to `archive/tests/`.
- **Same functionality.** Keep the tools, run lifecycle and rendering. Change
  behaviour only for something clearly better, and list it in the PR.

## Scope

1. Port the MCP server to `configs/` and `scripts/` (forward runs).
2. Move the HTML visualization into a new lib, `libs/visualization`.
3. Rename the lib to `mcp-server` and move `src/pyurbanair/jobs/` into it.
4. Update the tests, in `tests/`, and un-skip the MCP test modules.
5. Add a `pixi run -e mcp register-claude` task that registers the server
   with Claude Code.

## Current state (read these first)

| Where | What | Size |
|---|---|---|
| `libs/mcp_server/src/pyurbanair_mcp/` | thin MCP adapter: `server.py` (tool registration, SDK v2), `tools.py`, `schemas.py`, `__main__.py`. Dist name `pyurbanair-mcp`. | 461 lines |
| `src/pyurbanair/jobs/` | everything behind the tools: `preparation.py` (plan validation, code identity / fingerprints), `supervisor.py` + `registry.py` (SQLite queue, Unix socket), `worker.py`, `native.py`, `results.py`, `paths.py`, `processes.py`, `rendering_environment.py` | ~2.3k lines |
| `src/pyurbanair/workflows/forward.py` | the forward engine the MCP worker runs today: per-window persistence, `artifact_index.json`, run record, initial state `{path, member, time_index}`, solver-input snapshots, retained-bytes limit. Its only other user, the old `run_forward_model.py`, is archived. | 710 lines |
| `src/pyurbanair/config/composition.py` | composes the old `conf/run_forward_model.yaml`, lists options, validates `_target_`s / resolvers / overrides | ~200 lines |
| `src/pyurbanair/visualization/` | `render.py`, `render_3d.py`, `data.py` (reads `artifact_index.json`, falls back to a single `state.nc`), `assets.py` (loopback asset server), `web/` (`index.html`, `viewer.js`, `probe_charts.js`, `viewer.css`) | ~1.5k lines |
| `scripts/start_mcp` | launcher clients register (absolute path); `--check` is a read-only readiness check | |
| `archive/conf/visualization/` | the render presets `quicklook.yaml`, `flow_3d.yaml` (archived with `conf/`; bring them back under `configs/`) | |
| `pyproject.toml` | feature `mcp` (pypi dep `pyurbanair-mcp`, task `start-mcp`); env `mcp = [mcp, dev, py312, cpu]`; env `rendering` (PyVista/VTK, ffmpeg) | |
| `docs/mcp.md`, `docs/forward_visualization.md` | user docs for both | |

The setup the server must use:

| Where | What |
|---|---|
| `configs/forward.yaml` | forward entry config (`forward.ensemble`, `forward.rollout_steps`, `forward.initial_state`), plus `common.yaml` and the `case/`, `model/` and `params/` groups |
| `scripts/run_forward.py` | `run(cfg)`: runs `1 + forward.rollout_steps` windows; writes `config.yaml`, `state.nc` and `params.nc` into `paths.results_dir` |
| `scripts/visualize_forward.py` | `run(run_dir)`: the matplotlib figures of a forward run |
| `scripts/utils/inconsistency_check.py` | `check_config(cfg, "forward")`: fast config validation |
| `workflows/forward_workflow.sh` | run, then visualize, from the same overrides |
| `docs/config_setup_spec.md` | old to new config mapping (e.g. `run.rollout_steps` → `forward.rollout_steps`) |

Couplings to the archived setup to remove:

- `tools.py`: requires `conf/run_forward_model.yaml` to accept `--repo-root`.
- `composition.py`: composes `conf/` (`config_name="run_forward_model"`),
  trusts `_target_`s from `conf/**`, and reads `conf/trusted_forward_targets.txt`.
- `preparation.py` `code_identity`: fingerprints `src/pyurbanair`, `conf/`,
  `scripts/_common.py` and `scripts/run_forward_model.py` (all archived now).
- `jobs/worker.py`: runs `pyurbanair.workflows.forward.run`, which imports
  from the old `scripts._common`.

## Tasks in detail

### 1. Port to configs / scripts

- **Engine:** the worker runs `scripts/run_forward.py`'s `run(cfg)`.
  - List what the MCP needs that `run_forward.py` doesn't do yet. Likely
    candidates:
    - writing each window as it finishes, so a cancelled or failed run keeps
      partial results;
    - the `{path, member, time_index}` initial state. `configs/forward.yaml`
      already promises it, but `run_forward._initial_state` only takes a path;
    - an index/record of what ran.
  - Add only those, in `run_forward.py`'s style. `AGENTS.md`'s "no-op when
    absent" rule applies: a plain `run_forward.py` run must stay unchanged.
  - `visualization/data.py` already reads a plain `state.nc` when there is no
    `artifact_index.json`, so prefer dropping the index if nothing else needs it.
- **Old engine:** delete `src/pyurbanair/workflows/forward.py` once nothing
  uses it. Its only other user, the old `run_forward_model.py`, is archived.
  Remove the then-empty `src/pyurbanair/workflows/` package as well, so it no
  longer clashes in name with the top-level `workflows/` (shell pipelines).
- **One config validator:** `src/pyurbanair/config/run_record.py`
  (`validate_run_config`, `write_run_record`, `append_constructor_override`;
  ~250 lines) validates the old `conf/` keys and duplicates
  `scripts/utils/inconsistency_check.py`. Only `workflows/forward.py` and
  `jobs/preparation.py` use it, plus one test
  (`tests/pyudales/test_udales_discrepancy_wiring.py`). Use `check_config`
  instead and delete `run_record.py`. Then delete the functions in
  `src/pyurbanair/config/discrepancy.py` that nothing uses any more:
  `validate_sgs_discrepancy_inference`, `augment_sgs_discrepancy_prior` and
  `validate_parameter_selection`. Keep `SGS_BIAS_PARAMETER_NAMES` and
  `validate_sgs_discrepancy_settings`; the scripts use them.
- **Unused config keys:** nothing in `configs/` or `scripts/` reads `run.name`,
  `run.skip_viz` (both in `configs/common.yaml`) or `paths.base_results_dir`.
  The last exists only for `resolve_output_dir` in
  `src/pyurbanair/config/hydra_helpers.py`, which only the old engine and
  `job_scripts/` call. If the port doesn't need them, remove the keys and
  their documentation (`configs/README.md`). Also remove `resolve_output_dir`
  if the job-scripts PR has dropped its last caller by then; otherwise leave
  it to that PR.
- **Composition:** compose `configs/forward.yaml`.
  - Option lists come from `configs/{case,model,params}`.
  - Trusted `_target_`s come from `configs/**`.
  - Bring the render presets back from `archive/conf/visualization/*.yaml` to
    `configs/visualization/`. Bring back `trusted_forward_targets.txt` too, if
    it is still needed.
- **Config check:** call `check_config(cfg, "forward")` during preparation, so
  the MCP rejects what the CLI rejects.
- **Code identity:** fingerprint `configs/` and the `scripts/` files the run
  imports.
- **Tool parameters and docs:** use the new key names. Update tool
  descriptions and `get_capabilities` (e.g. `total_windows: "1 +
  forward.rollout_steps"`).
- **Backends:** keep all four.
  - The surrogate's `spinup_source: training_data` has no support in
    `scripts/` (nothing loads the training snapshots as the initial state).
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
- **What stays:** matplotlib-only code: `src/pyurbanair/utils/animation_utils.py`
  (where the old `animation.py` was merged) and the `scripts/visualize_*.py`
  figures.
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
  - `docs/mcp.md`, `docs/codebase_guide.md` and `AGENTS.md`'s doc table.
- **Server name:** keep the registered server name `pyurbanair`, so existing
  client registrations keep working.

### 4. Tests (all in `tests/`)

- **Layout:** `tests/` has one folder per package, plus `scripts/` (tests of
  `scripts/`, `configs/` and `workflows/`), `configs/` (test overlays) and
  `legacy/` (frozen configs and fixtures of the carried-over library tests).
  See `tests/README.md`.
  - The MCP tests are in `tests/mcp/` (`test_mcp_protocol.py`, plus
    `test_mcp_forward_integration.py`, which is `integration`).
  - The tests of the code you move are in `tests/pyurbanair/`:
    `test_forward_preparation*.py`, `test_forward_plan_identity.py`,
    `test_forward_input_safety.py`, `test_local_jobs.py` and
    `test_forward_visualization*.py`.
- **Un-skip:** the modules that broke when the old setup was archived carry
  `pytestmark = pytest.mark.skip(reason="MCP port pending: ...")`:
  - `tests/mcp/test_mcp_protocol.py` and `test_mcp_forward_integration.py`
    (`Tools()` requires `conf/run_forward_model.yaml`);
  - `tests/pyurbanair/test_forward_preparation.py`,
    `test_forward_preparation_paths.py`, `test_forward_plan_identity.py` and
    `test_forward_input_safety.py` (their `checkout` fixture copies the old
    `tests/conf`, now archived; the frozen copy is in `tests/legacy/conf`).

  Remove every such marker (`grep -rn "MCP port pending" tests`) and make the
  tests pass. `test_local_jobs.py` and `test_forward_visualization*.py` still
  pass and are not skipped.
- **Moving tests:** move tests with their code.
  - Jobs tests go to `tests/mcp/`: `test_forward_preparation*.py`,
    `test_forward_plan_identity.py`, `test_forward_input_safety.py` and
    `test_local_jobs.py`, from `tests/pyurbanair/`.
  - Visualization tests go to a new `tests/visualization/`, with a CI workflow
    copied from one of `.github/workflows/tests-*.yml`:
    `test_forward_visualization*.py`, from `tests/pyurbanair/`.
  - Afterwards `tests/pyurbanair/` holds only the tests of what stays in
    `src/pyurbanair` (base classes, samplers).
  - Update the path filters of every workflow you touch.
- **Port to the current configs:** these tests came from the old suite and use
  the legacy fixtures and configs in `tests/legacy/`. Port the ones you touch
  to `configs/` with overlays from `tests/configs/` (`compose("forward",
  "+test=forward", root=tmp_path)` in `tests/conftest.py`). Drop what tests
  removed behaviour. Add any overlay you need under `tests/configs/`.
- **New features of `run_forward.py`:** test them in
  `tests/scripts/test_forward.py`.
- **mypy:** `.pre-commit-config.yaml` excludes the carried-over library-test
  folders from mypy. Folders you rewrite should pass mypy; remove them from
  that exclude.
- **Running the MCP tests:** they run in the `mcp` env: `pixi run --locked -e mcp
  python -m pytest tests/mcp`.

### 5. `pixi run -e mcp register-claude`

- **The task:** add `register-claude` to the `mcp` feature's tasks in
  `pyproject.toml`, backed by a short script (e.g.
  `scripts/register_claude.sh` next to `scripts/start_mcp`, or a `mcp_server`
  entry point).
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

- **Repo rules (see `AGENTS.md`):**
  - pixi `dev` env;
  - run `pixi run -e dev pre-commit` before committing;
  - parallel ensembles use `forkserver`;
  - backends stay byte-identical when a new knob is absent;
  - keep docs in sync in the same PR.
- **Untouchable files:** `configs/assimilation.yaml` and `archive/conf/*.yaml`
  may carry the user's live, uncommitted tuning. Never commit, stash or reset
  them.
- **Known local failures on the user's Mac:**
  - `import torch` aborts unless `KMP_DUPLICATE_LIB_OK=TRUE`;
  - LBM compilation (`prepare_compile`) aborts, so pylbm integration runs only
    on CI;
  - `tests/pyudales/test_udales_discrepancy_native.py` fails to compile its
    kernel (fails on a clean checkout too).
- **Integration tests aren't in CI:** CI runs the default suite only (no
  compiled solver). Run the relevant `integration` tests locally
  (`pixi run -e dev test-integration`, uDALES works locally) or note what you
  couldn't run.
- **Sandboxes:** the supervisor uses Unix sockets and loopback; sandboxes may
  block them (see `docs/mcp.md`, "Validation").

## Done when

- [ ] `scripts/start_mcp --check` passes. Over the protocol: list options,
  prepare and launch a forward run on `configs/`, poll it, inspect results,
  render the HTML view, and cancel a run.
- [ ] The MCP worker runs `scripts/run_forward.py`. No MCP, jobs or
  visualization code touches `archive/`. `src/pyurbanair/workflows/` and
  `config/run_record.py` are gone, and `check_config` is the only config
  validator. Any new script or workflow is justified in the PR.
- [ ] `libs/visualization` and `libs/mcp-server` exist with the conventions
  above. `src/pyurbanair/jobs/` and `src/pyurbanair/visualization/` are gone,
  and no reference to `libs/mcp_server` / `pyurbanair_mcp` remains.
- [ ] `pixi run -e mcp register-claude` registers the server after a "y",
  does nothing on a second run, and never prompts without a terminal.
- [ ] No "MCP port pending" skips remain. All new and moved tests are in
  `tests/` and pass, with their CI workflows. Pre-commit passes.
- [ ] `docs/mcp.md`, `docs/forward_visualization.md` (or a new
  `docs/visualization.md`), `docs/codebase_guide.md`, `AGENTS.md`'s doc table
  and `tests/README.md` are updated.
- [ ] The PR description lists behaviour changes, what `run_forward.py` gained
  and why, and anything not run (e.g. integration tests on backends
  unavailable locally).

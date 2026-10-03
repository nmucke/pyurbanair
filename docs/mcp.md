# Local forward simulations through MCP

The optional `mcp-server` library (`libs/mcp-server`, import `mcp_server`)
exposes local forward simulations through stdio. A run is
`scripts/run_forward.py` on `configs/forward.yaml`, the same as on the command
line; assimilation is not exposed. `mcp_server/server.py` registers the tools
with the MCP SDK (imported as `mcp`); `mcp_server.jobs` holds composition,
preparation, the job queue and the workers and works without the SDK. Rendering
is the `visualization` library ([docs/visualization.md](visualization.md)).
Workers use a selected Pixi environment; the protocol process never imports
solver runtimes for discovery.

## Install and connect

From an existing checkout:

```bash
pixi run setup-dev
pixi install --locked -e mcp
scripts/start_mcp --check
pixi run -e mcp register-claude     # Claude Code: asks, then registers (user scope)
```

`register-claude` (`scripts/register_claude.sh`) does nothing if `claude` is not
installed or a `pyurbanair` server is already registered, and only prints the
command when there is no terminal to ask in.

Prepare the selected backend using its maintained reference:
[LBM](pylbm.md), [uDALES](pyudales.md), [PALM](pypalm.md), or
[neural surrogates](neural_surrogates.md). `--check` is read-only: it does not
compile, fetch source, or prove that a solver runs. A trained surrogate export
and its referenced metadata are supplied separately. No LLM API key is needed.

uDALES readiness checks require its initialized source submodule and the worker
build tools. Workers prepare the selected stock or discrepancy solver in the
managed uDALES cache; the legacy `build/release/u-dales` binary is not required.

CPU workers use `dev`; GPU workers use the installed `cuda` environment. The MCP
server runs in the separate `mcp` environment. Install optional 3D support with
`pixi install --locked -e rendering`. It includes PyVista/VTK and ffmpeg.
The root lockfile pins the tested MCP SDK v2 and renderer packages.

Other clients, or Claude Code by hand, register an **absolute** launcher path:

```bash
codex mcp add pyurbanair -- /absolute/repo/scripts/start_mcp
claude mcp add --transport stdio --scope user pyurbanair -- /absolute/repo/scripts/start_mcp
```

For Codex CLI, IDE and ChatGPT desktop on the same Codex host, the equivalent
`~/.codex/config.toml` entry is:

```toml
[mcp_servers.pyurbanair]
command = "/absolute/repo/scripts/start_mcp"
args = []
```

ChatGPT desktop also offers Settings → MCP servers → Add server → STDIO;
select the absolute launcher, save, and restart the server. These clients share
configuration on the host; ChatGPT web does not read this local file. See the
[current OpenAI MCP configuration reference](https://learn.chatgpt.com/docs/extend/mcp).
Client UI availability depends on the installed version.

Claude Desktop uses:

```json
{
  "mcpServers": {
    "pyurbanair": {
      "command": "/absolute/repo/scripts/start_mcp",
      "args": []
    }
  }
}
```

See [Claude Code configuration](https://code.claude.com/docs/en/mcp) and
[local desktop setup](https://modelcontextprotocol.io/docs/develop/connect-local-servers).
The [official Python SDK](https://py.sdk.modelcontextprotocol.io/) supplies the
protocol implementation; this library uses its `MCPServer` v2 API.

The launcher accepts `--store-root /absolute/local/storage` to relocate the
registry, plans, jobs and results. Clients sharing a checkout should use the
same dedicated storage root. Existing stores must be owned by the current user
and private (mode `0700`); the service does not change existing permissions.
`PYURBANAIR_PIXI` can select an absolute Pixi executable when
a desktop app has a restricted PATH. Provision environments before connecting;
startup does not intentionally install packages and all diagnostics use stderr.

## Tool workflow

1. `get_capabilities` and `list_config_options` inspect available backends,
   prerequisites, native-setting coverage and the options of `configs/case`,
   `configs/model`, `configs/params` and `configs/visualization` (render presets).
2. `inspect_config` composes `configs/forward.yaml` with ordered Hydra overrides
   and an optional subtree. Examples include `model=pylbm`,
   `case=xie_and_castro`, `params=static`, `forward.ensemble=true`,
   `forward.rollout_steps=2`, additions using `+`/`++`, and deletions using `~`.
3. `prepare_forward_run` returns a plan ID/digest, resolved values, source files,
   differences from defaults, resource estimates and field-specific validation.
   The config check is the CLI's own, `check_config(cfg, "forward")`
   (`scripts/utils/inconsistency_check.py`); on top come this machine's resource
   limits and the input and backend prerequisites. Configuration validity,
   prerequisite presence and solver smoke testing are separate. Preparation
   does not run or compile a solver.
4. `launch_forward_run(plan_id, idempotency_key)` queues the snapshot and returns
   a run ID promptly. Reuse the same key when retrying the same launch. Reusing
   it for different inputs is an error. Optional `post_render` settings enqueue
   a render job automatically after a successful simulation.
5. `get_run_status`, `get_run_logs` and `list_runs` work after reconnecting.
   Logs use byte cursors, default to 32 KiB, and accept at most 128 KiB per read.
   `cancel_run` requests teardown; poll until `cancelled` before assuming the
   resource slot has been released.
6. `inspect_run_results` lists the run's NetCDF files (`artifact_id` is a
   file's position in that list), inspects one's dimensions and variables, or
   returns a numeric slice of at most 4096 values. Selection uses dimension
   names and integer indices or `[start, stop, stride]` lists.
7. `render_simulation` queues a separate visualization of a successful run.
   `get_visualization` returns a PNG (up to 2 MiB), bundle metadata, available
   previews and an on-demand loopback viewer URL.

Preparation is a technical validation step. It does not add an extra user
confirmation when the run has already been requested.

## Configuration and artifacts

Scientific configuration is the `configs/` tree. Trusted `_target_`s are the
ones declared in `configs/**`. Requests cannot add arbitrary executable
`_target_` values, resolver expressions, Hydra plugins, sweepers or search paths.

Native settings are separately classified. Wrapper-managed grid, time, inflow
and turbulence fields must use their canonical Hydra paths. Currently writable
independent fields are uDALES `run.dtmax`/`run.courant` and PALM
`runtime_parameters.dt_max`/`initialization_parameters.cfl_factor`. They are
patched in private staged templates using the existing backend editors and
survive constructor/member copying. Unknown native fields are rejected clearly.
LBM's positional input is not treated as a generic namelist.

Plans snapshot selected templates and fingerprint input files, relevant source
(`configs/`, `scripts/run_forward.py`, `scripts/utils/`, `src/pyurbanair`, the
MCP server and the selected backends), backend identities and environment
information. Launch refuses a changed plan,
code, or input. It consumes the saved resolved configuration. Job-owned paths
are applied last and recorded in `launch.json`; source templates remain intact.
For uDALES, source fingerprints include the managed builder's shell scripts and
discrepancy extension resources, including its manifest, patch and Fortran code.
Plans also record compiler, linker, and macOS SDK/deployment overrides. The
supervisor restores those saved settings and the client's search path, clearing
stale overrides before starting a worker. Pixi then activates the selected
worker environment. The worker checks build settings against the plan, including
the expected NVHPC library-path addition for CUDA activation. Changes to the preparing client's
`PATH` or `CONDA_PREFIX` also require a new plan.
Large input hashing can take time. Do not edit the checkout between preparing
and launching a run; prepare a new plan after changes.

Case directories must be separate from managed job storage. Directory symlinks
and non-regular files are rejected. Regular-file symlinks within the selected
case or checkout, such as the supplied cases' linked STL geometry, are copied
as regular files; targets in managed storage are rejected. Preparation bounds
case staging to 2 GiB, 10,000 entries and 32 directory levels by default. The
machine-local `max_case_input_bytes` and `max_case_input_entries` limits can be configured;
requests may only tighten them. These limits apply to copied case templates,
not surrogate weights or other inputs that are fingerprinted in place.

An initial state is a NetCDF path or `{"path": "/path/state.nc", "member": 3,
"time_index": -1}` (the `initial_state` argument, or `forward.initial_state`).
`member` selects an ensemble member by its coordinate label and `time_index` a
frame by position (default: the last); `run_forward.py` starts from that one
frame. The surrogate's `training_data` spin-up has no loader in `scripts/`, so
under MCP it needs an explicit initial state (its CFD spin-up model is then
dropped from the plan); `generative` spin-up needs none. CPU hosts should set
`model.forward_model.device=cpu`. MCP preparation rejects uninitialized weights.

`forward.rollout_steps` counts **additional** windows. A value of 2 means three
windows. The worker runs `run(cfg)` of `scripts/run_forward.py` with
`paths.results_dir` bound to the run's `artifacts/` and
`forward.save_windows=true`:

```text
<store>/plans/<plan_id>/plan.json
<store>/runs/<run_id>/
  job.json, launch.json, worker.log, worker_status.json, completion.json
  artifacts/config.yaml, state.nc, params.nc       # run_forward.py's outputs
  artifacts/windows/state_<window>.nc, params_<window>.nc
  scratch/, build/, fast_io/, tmp/, cache/, work/
```

Each window is written as it finishes, so a failed or cancelled job keeps the
finished windows and its logs; `state.nc` and `params.nc` appear when the run
completes. `run_forward.py` keeps every window in memory until the end; the
plan's `estimated_output_bytes` is checked against the machine's
`max_output_bytes`.

## Lifecycle and isolation

A detached supervisor owns a transactional SQLite queue and a private Unix
socket. It admits one active simulation or render job at a time. States are
`queued`, `preparing`, `running`, `finalizing`, `succeeded`, `failed`, `cancelling`,
`cancelled`, and `interrupted`. Closing MCP does not stop the supervisor.

Workers capture native stdout/stderr into their job log and select `dev`,
`cuda`, or `rendering` independently. Cancellation signals verified process
identities and inherited job tokens, including detached MPI/forkserver children,
then escalates TERM to KILL after five seconds. The slot stays occupied until
owned descendants exit. Forward workers hold a checkout-wide backend lock for
shared source/cache operations, including PALM's first-use build; imports
cannot auto-sync LBM sources. Recovery reconciles recorded completion with verified
PID creation times; reused PIDs are never signalled. Unfinished jobs whose
workers disappeared are marked interrupted after supervisor recovery.
There is no automatic numerical resume after reboot.

The asset server serves only registered bundle files, validates Host/Origin,
and supports byte ranges for video seeking. It binds to loopback and expires
when idle. Retrieving a visualization starts a fresh server when needed.
Opening the returned URL is a separate browser action. A remote client may
not reach this localhost URL; PNG content is delivered through MCP directly.

## Visualization

Render options include `member` or `reduction` (`mean_velocity`/`mean_speed`),
`variable` (`u`, `v`, `w`, `speed`, `horizontal_speed`), `slices`, `probes`, time
selection, color limits, FPS, dimensions, and bounded frame counts. `render_3d`
adds the optional offscreen renderer with explicit geometry, seeds and camera.
The default 2D dashboard displays two horizontal maps and a vertical `w` section
at once, alongside matching A–C virtual-probe traces at both heights. Per-slice
`variable` and physical `position` or domain `fraction` are configurable. Explicit
probe coordinates replace automatic samples; `probes: []` omits them. The viewer's
2D/3D buttons and shared timeline keep every visible panel at matching physical
time, including PNG-sequence playback when a movie is unavailable.
Inspect the `configs/visualization/` presets for examples and use
[the visualization reference](visualization.md) for scientific details.
Rendering reads the run's `artifacts/state.nc`.

Regular and staggered components are collocated by physical coordinates before
computing speed. Horizontal speed and full magnitude are distinct. Solids use
explicit masks, never zero velocity. Labels retain actual selected slice/probe
coordinates. Movies map presentation time to saved simulation times and do not
invent intermediate solver output. Re-rendering another height or camera uses
the same numerical artifacts.

A bundle contains local HTML/CSS/JS, its versioned manifest, probes in JSON/CSV,
PNG previews, provenance, effective settings, and available MP4s. Missing ffmpeg
leaves stills and probes usable. Missing optional 3D prerequisites leave standard
views usable. Serve exported bundles with a local static HTTP server; direct
`file://` opening is not a supported JSON-fetch workflow.

## Validation and troubleshooting

Run the core and protocol checks with local Unix/loopback sockets available:

```bash
pixi run --locked -e mcp python -m pytest tests/mcp
pixi run --locked -e mcp python -m pytest tests/mcp/test_mcp_forward_integration.py -m integration
pixi run --locked -e dev python -m pytest tests/visualization
```

The integration tests run each backend over the protocol on the tiny grid of
`tests/configs/test/tiny.yaml`, passed as overrides of `configs/forward.yaml`.
PALM's installed solver requires at least 14 vertical cells, so its run uses 16.
A generated untrained surrogate export tests deployment and makes no claim
about predictive accuracy.

If a job fails, read its validation issues, then its bounded worker log. Missing
compilers/binaries require backend setup; a valid YAML configuration alone is
insufficient. MPI may fail in sandboxes that disallow local sockets. Offscreen
VTK must be verified on the actual host. Supervisor startup errors are retained
in `<store>/supervisor.log`. Native desktop UI testing and macOS backend testing
must be performed on those hosts; protocol tests are not a substitute.

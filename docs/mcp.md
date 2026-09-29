# Local forward simulations through MCP

The optional `pyurbanair-mcp` library exposes local forward simulations through
stdio. Configuration, execution, job storage and rendering live in `pyurbanair`
and work without the MCP SDK. Workers use a selected Pixi environment; the
protocol process never imports solver runtimes for discovery.

## Install and connect

From an existing checkout:

```bash
pixi run setup-dev
pixi install --locked -e mcp
scripts/start_mcp --check
```

Prepare the selected backend using its maintained reference:
[LBM](pylbm.md), [uDALES](pyudales.md), [PALM](pypalm.md), or
[neural surrogates](neural_surrogates.md). `--check` is read-only: it does not
compile, fetch source, or prove that a solver runs. A trained surrogate export
and its referenced metadata are supplied separately. No LLM API key is needed.

CPU workers use `dev`; GPU workers use the installed `cuda` environment. The MCP
server runs in the separate `mcp` environment. Install optional 3D support with
`pixi install --locked -e rendering`. Its PyVista/VTK and ffmpeg dependencies
are independent of the older `viz` environment and `les-render` library.
The root lockfile pins the tested MCP SDK v2 and renderer packages.

Register an **absolute** launcher path:

```bash
codex mcp add pyurbanair -- /absolute/repo/scripts/start_mcp
claude mcp add --transport stdio pyurbanair -- /absolute/repo/scripts/start_mcp
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
   prerequisites, execution presets, native-setting coverage and render presets.
2. `inspect_config` accepts ordered Hydra overrides and an optional subtree.
   Examples include `model@model=pylbm`, `case=xie_and_castro`, `params=static`,
   `run.ensemble=true`, additions using `+`/`++`, and deletions using `~`.
3. `prepare_forward_run` returns a plan ID/digest, resolved values, source files,
   differences from defaults, resource estimates and field-specific validation.
   Configuration validity, prerequisite presence and solver smoke testing are
   separate. Preparation does not run or compile a solver.
4. `launch_forward_run(plan_id, idempotency_key)` queues the snapshot and returns
   a run ID promptly. Reuse the same key when retrying the same launch. Reusing
   it for different inputs is an error. Optional `post_render` settings enqueue
   a render job automatically after a successful simulation.
5. `get_run_status`, `get_run_logs` and `list_runs` work after reconnecting.
   Logs use byte cursors, default to 32 KiB, and accept at most 128 KiB per read.
   `cancel_run` requests teardown; poll until `cancelled` before assuming the
   resource slot has been released.
6. `inspect_run_results` lists indexed artifacts, inspects NetCDF dimensions and
   variables, or returns a numeric slice of at most 4096 values. Selection uses
   dimension names and integer indices or `[start, stop, stride]` lists.
7. `render_simulation` queues a separate visualization of a successful run.
   `get_visualization` returns a PNG (up to 2 MiB), bundle metadata, available
   previews and an on-demand loopback viewer URL.

Preparation is a technical validation step. It does not add an extra user
confirmation when the run has already been requested.

## Configuration and artifacts

Scientific configuration remains the Hydra tree. Trusted targets come from
repository component YAML and the optional owner-maintained
`conf/trusted_forward_targets.txt`. Requests cannot add arbitrary executable
`_target_` values, resolver expressions, Hydra plugins, sweepers or search paths.

Native settings are separately classified. Wrapper-managed grid, time, inflow
and turbulence fields must use their canonical Hydra paths. Currently writable
independent fields are uDALES `run.dtmax`/`run.courant` and PALM
`runtime_parameters.dt_max`/`initialization_parameters.cfl_factor`. They are
patched in private staged templates using the existing backend editors and
survive constructor/member copying. Unknown native fields are rejected clearly.
LBM's positional input is not treated as a generic namelist.

Plans snapshot selected templates and fingerprint input files, relevant source,
backend identities and environment information. Launch refuses a changed plan,
code, or input. It consumes the saved resolved configuration. Job-owned paths
are applied last and recorded in `launch.json`; source templates remain intact.
Large input hashing can take time. Do not edit the checkout between preparing
and launching a run; prepare a new plan after changes.

An initial state is `{"path":"/path/state.nc","time_index":-1}` with optional
`member`. The time index selects the end of the retained history. Ensemble
states retain their member dimension. Surrogate inputs must match trained
coordinates, variables, cadence and required history; the `training_data`
spin-up mode requires explicit state selection. CPU hosts should set
`model.forward_model.device=cpu`. MCP preparation rejects uninitialized weights.

`run.rollout_steps` counts **additional** windows. A value of 2 means three
windows. MCP selects complete numerical persistence, independent of plotting:

```text
<store>/plans/<plan_id>/plan.json
<store>/runs/<run_id>/
  job.json, launch.json, worker.log, worker_status.json, completion.json
  run_manifest.yaml, config.yaml, config.resolved.yaml, forward_status.json, artifact_index.json
  sampled_params.nc, state.nc, params.nc
  windows/<window>/state_<member>.nc, params_<member>.nc
  scratch/, build/, fast_io/, tmp/, cache/, work/
```

`artifact_index.json` records every completed member/window, physical times,
checksums and ensemble failure substitutions. It also saves bounded copies of
the actual generated solver inputs for each completed window, with source paths
and hashes. Solver-input artifacts support bounded text inspection. A failed or cancelled job keeps
partial artifacts and logs. Completed numerical results survive optional plot
failures. `run.ensemble_save_on_disk=true` is rejected: the workflow retains
rollout history in memory and does not promise constant-memory streaming.
Complete mode enforces `run.max_retained_bytes` (2 GiB by default).

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
shared source/cache operations; imports cannot auto-sync LBM sources or
auto-install PALM. Recovery reconciles recorded completion with verified
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
Inspect `conf/visualization/` presets for examples and use
[the visualization reference](forward_visualization.md) for scientific details.

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
pixi run -e dev python -m pytest tests/test_forward_preparation.py tests/test_forward_workflow.py tests/test_local_jobs.py tests/test_forward_visualization.py
pixi run -e mcp python -m pytest tests/test_mcp_protocol.py
pixi run -e mcp python -m pytest tests/test_mcp_forward_integration.py -m integration
pixi run -e dev python -m pytest tests/test_forward_workflow_integration.py -m integration
```

The integration tests use independent small test configurations. PALM's installed
solver requires at least 14 vertical cells, so its complete-artifact fixture uses
16. A generated compatible surrogate checkpoint tests deployment/history
handling and makes no claim about trained predictive accuracy.

If a job fails, read its validation issues, then its bounded worker log. Missing
compilers/binaries require backend setup; a valid YAML configuration alone is
insufficient. MPI may fail in sandboxes that disallow local sockets. Offscreen
VTK must be verified on the actual host. Supervisor startup errors are retained
in `<store>/supervisor.log`. Native desktop UI testing and macOS backend testing
must be performed on those hosts; protocol tests are not a substitute.

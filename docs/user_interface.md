# Browser app for city simulations

`libs/user-interface` (import `user_interface`) is a local web app: draw an
area on a map, build the case, pick a model and settings, launch the run,
watch it and open the results viewer. No LLM is involved. It runs in the `ui`
environment (mcp-server's tools and jobs, without the MCP SDK) and builds cases as jobs in the `city4cfd` environment, which
compiles City4CFD on its first install (~2 min, see
[pycity4cfd.md](pycity4cfd.md)).

```bash
pixi install -e ui -e city4cfd
pixi run -e ui start-ui             # = python -m user_interface --repo-root .
```

`python -m user_interface [--repo-root .] [--port 0] [--no-browser]` prints the
URL (`http://127.0.0.1:<port>/<token>/`) and opens it unless `--no-browser`.
Ctrl-C stops the server; launched runs keep going in the job supervisor
([mcp.md](mcp.md)). The case build needs internet access (Overpass), and so
does the page itself (map tiles and Leaflet).

## What it reuses

Every action is a call on one `mcp_server.tools.Tools(repo_root)`:
`prepare_city_case`, `get_city_case`, `inspect_config`, `prepare_forward_run`,
`launch_forward_run`, `get_run_status`, `get_run_logs`, `cancel_run`,
`inspect_run_results`, `render_simulation` and `visualization`. Runs, plans,
renders and case builds therefore land in the same store as MCP runs
(`.temp/local-jobs`), and city cases in `.temp/city_cases/`
([pycity4cfd.md](pycity4cfd.md)). A case build is a `city_case` supervisor job
in the `city4cfd` environment ([mcp.md](mcp.md)), so this process never
imports pycity4cfd; a case built before is ready at once.

`user_interface/area_selector.py` is the map selector of the MCP tools
`open_area_selector`/`get_selected_area` (its page is
`web/area_selector.html`) and holds the one bbox check, `validate_bbox`, and
`MAX_AREA_KM2`, which this app uses too. The app's own page draws the area
itself (a different UI); only the area formula is shared, mirrored in
`app.js`.

`user_interface/app.py` holds the server and the API; `web/` the static page
(`index.html`, `app.js`, `app.css`, no build step). The server mirrors
`user_interface/area_selector.py`: a stdlib `ThreadingHTTPServer` on 127.0.0.1,
exact Host and Origin checks, an unguessable token as the first path segment,
JSON-only POST bodies of at most 64 KiB (an empty body counts as `{}`).
Cases, runs and renders are kept in memory for the session; a restart forgets
them (the runs themselves stay in the store).

## JSON API

All routes are under `/<token>/`: the page at `/<token>/`, static files at
`/<token>/static/<file>`, the API at `/<token>/api/...`. Errors are HTTP 4xx/5xx
with `{"error": "<message>"}`.

| Method, path | Body | Response |
|---|---|---|
| `GET api/options` | | `{models, fields, case_defaults: {resolution, default_height}, max_area_km2}` |
| `POST api/cases` | `{bbox: {west, south, east, north}, resolution?, default_height?}` | 202 `{case_id}`; the build runs as a supervisor job |
| `GET api/cases/<id>` | | `{case_id, status: building\|ready\|failed, error?, summary?, buildings?, sensors?}` |
| `POST api/runs` | `{case_id, model, settings: {<field key>: value}, extra_overrides?}` | `{run_id}` (prepare + launch) |
| `GET api/runs` | | `{runs: [{run_id, case_id, model, state, created}]}`, newest first |
| `GET api/runs/<id>` | | `{run_id, state, phase?, message?, log_tail, results?: {files, variables, dimensions, times}}` |
| `POST api/runs/<id>/cancel` | | `{run_id, state}` |
| `POST api/runs/<id>/visualization` | | `{visualization_id}` (409 until the run succeeded) |
| `GET api/visualizations/<id>` | | `{state, viewer_url?, message?}` |

- **Runs** compose `["model=<m>", "params=dynamic", *case overrides,
  *settings, *extra_overrides]`. `params=dynamic` is pinned because the inflow
  fields live in that sampler. A plan with configuration or prerequisite issues
  is answered with a 400 listing them, and nothing is launched.
- **Fields** (`FIELDS` in `app.py`) are curated Hydra override paths. Each
  default is read from the composed forward config of the first model in the
  field's `models` (all models when absent), unless the field sets its own.
  A choice whose value is `null` sends no override: the boundary condition's
  "Model default" keeps each model's own value. Unknown keys are rejected, and
  so are fields for another model; `extra_overrides` is the escape hatch.
  The inflow angle is the flow direction, counter-clockwise from +x (east).
- **Case geometry**: `buildings` is `case_dir/buildings.geojson` reprojected to
  EPSG:4326 (properties `height`, `height_source`). `sensors` are the
  assimilated `obs.x/y_points` as `[lon, lat]`, converted from the domain
  frame with case.json's `crs` and `origin`. The held-out validation sensors
  are not returned.
- **Log tail**: the last 40 lines of the last 16 KiB of `worker.log`.
- **Viewer**: `visualization` passes the app's own origin as `frame_ancestor`,
  so the viewer served on the supervisor's asset server may be embedded in the
  page (any other page still may not). A render that succeeded without a
  `viewer_url` (browser serving unavailable) says why in `message`.

## Tests

`tests/user_interface/` starts the app on a fake `Tools`, which keeps the real
config composition and fakes everything else: no network, no solver, no
supervisor. It checks the token, Host and Origin checks, the body rules, that
every field key resolves for each model it applies to, and the case and run
lifecycles. `test_area_selector.py` tests the map selector's server directly;
its MCP tools are tested in `tests/mcp/test_city_tools.py`.

```bash
pixi run --locked -e ui python -m pytest tests/user_interface
```

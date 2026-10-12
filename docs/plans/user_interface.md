# Plan: a browser interface for city simulations (no agent)

`pixi run -e mcp start-ui` starts a local web app and opens the browser. The
user draws an area, reviews the generated case, picks a model and settings,
launches the run, watches it, and sees the results. No LLM anywhere.

## Layout

```text
libs/user-interface/src/user_interface/
  app.py        loopback HTTP server + JSON API, wraps mcp_server.tools.Tools
  __main__.py   python -m user_interface [--repo-root .] [--port 0] [--no-browser]
  web/          index.html, app.js, app.css (static, no build step)
tests/user-interface/
```

Reuse, don't duplicate: every action goes through `mcp_server.tools.Tools`
(`prepare_city_case`, `inspect_config`, `prepare_forward_run`,
`launch_forward_run`, `get_run_status`, `get_run_logs`, `cancel_run`,
`render_simulation`, `visualization`, `inspect_run_results`; only `server.py`
imports the MCP SDK). Bbox validation is `mcp_server.area_selector.validate_bbox`.
The server mirrors `mcp_server/area_selector.py` (stdlib `ThreadingHTTPServer`
on 127.0.0.1, Host/Origin checks, unguessable token in the URL path, JSON-only
POSTs with a size cap).

## JSON API

All routes under `/<token>/`; static files at `/<token>/` (index.html) and
`/<token>/static/<file>`. Errors: HTTP 4xx/5xx with `{"error": "<message for
the user>"}`.

| Method, path | Body | Response |
|---|---|---|
| `GET api/options` | | `{models: [{id, label, description}], fields: [{key, label, type: "number"\|"int"\|"bool"\|"choice", default, min?, max?, step?, unit?, choices?, models?: [ids], help}], case_defaults: {resolution, default_height}, max_area_km2}` |
| `POST api/cases` | `{bbox: {west, south, east, north}, resolution, default_height}` | `{case_id}` (202; the build runs in a background thread) |
| `GET api/cases/<case_id>` | | `{case_id, status: "building"\|"ready"\|"failed", error?, summary?: {n_buildings, height_sources, max_height, domain: {nx, ny, nz, lengths_m}}, buildings?: GeoJSON FeatureCollection in EPSG:4326 with properties {height, height_source}, sensors?: [[lon, lat], ...]}` |
| `POST api/runs` | `{case_id, model, settings: {<field key>: value}, extra_overrides?: [str]}` | `{run_id}` (prepare + launch) |
| `GET api/runs` | | `{runs: [{run_id, case_id, model, state, created}]}` (this app's runs) |
| `GET api/runs/<run_id>` | | `{run_id, state, phase?, message?, log_tail: [str], results?: {variables, times, ...}}` (`state` as `get_run_status` reports it) |
| `POST api/runs/<run_id>/cancel` | | `{run_id, state}` |
| `POST api/runs/<run_id>/visualization` | | `{visualization_id}` (starts rendering once the run succeeded) |
| `GET api/visualizations/<id>` | | `{state, viewer_url?}` (`viewer_url` is the loopback viewer `Tools.visualization` serves) |

`fields` are curated, model-aware run settings, each `key` a Hydra override
path read from the composed config for its default (`inspect_config`): run
time, output frequency, spin-up, inflow speed and angle, ensemble on/off and
size, and anything else worth a knob. Unknown keys are rejected; free-form
`extra_overrides` are the escape hatch.

## Interface

One page, a left panel stepping through 1 Area -> 2 Case -> 3 Model & settings
-> 4 Run -> 5 Results, a full-height map on the right. Area: draw a rectangle
(live km², over-limit in red), place search. Case: progress while building,
then buildings drawn on the map coloured by height (legend; outline style by
`height_source`), sensors as dots, summary cards (buildings, heights from tags
vs default, grid, estimated cells). Model: cards for pylbm / PALM / uDALES
with one-line trade-offs, then the settings form generated from `fields`.
Run: state, phase, elapsed time, log tail, cancel. Results: the viewer
embedded (iframe) with an "open in new tab" link. Previous runs listed for
reopening. Clean, modern, light and dark mode, works at laptop widths.

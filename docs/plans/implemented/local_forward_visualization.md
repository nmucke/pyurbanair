# Forward simulation visualization and HTML viewer

**Overview.** Keep the supplied HTML's appearance and playback interface, but
replace its fixed cases and filenames with a generated manifest. Add a new
postprocessing service that reads completed simulation artifacts and produces
2D slices, virtual-probe series, and optional 3D flow movies. Agents receive
rendered PNGs they can inspect; users get a local browser viewer with synchronized
playback and downloads.

This supplements the [local MCP plan](local_forward_mcp.md). It is a design,
not an implemented viewer. The supplied source is preserved in
[forward_viewer.original.html.txt](references/forward_viewer.original.html.txt)
as a reference, not a runnable application.

**1. What the supplied HTML already does**

The HTML is a player for previously rendered MP4s, with posters, a 2D/3D toggle,
play/pause, restart, seeking, playback speed, fullscreen and downloads. Its dark
theme, responsive layout and canvas probe panels are worth preserving. It does
not load NetCDF, calculate flow lines, generate movies or provide an orbitable
3D scene. Changing a camera or field requires a new render; playback and view
switching work directly in the browser.

The attachment references `sensor_data.js`, `sensor_charts.js`, posters and six
movies that were not supplied and were not found in the checkout. Reimplement
the chart controller against generated probe JSON; the attachment alone cannot
reproduce the original films. Keep the design recognizable without promising
identical rendering to unavailable source media.

| Current code | Planned adaptation |
| --- | --- |
| Fixed `train/val/test` cases and geometry names | Populate selections from bundle entries with run IDs, case labels and member/reduction labels. Start with one run; allow collections of rendered entries. |
| Filename patterns such as `flow3d_test.mp4` | Use per-view media/poster references from the manifest. Only enable available views. |
| Hard-coded 4 m grid, heights 2/26 m and three sensors | Generate domain/grid descriptions, actual slice heights and any configured probe series; make panels dynamic. |
| Fixed `max=20` and `/ 0:20` clock | Set duration after media metadata loads; format hours/minutes/seconds and handle zero/unknown duration. |
| `window.LESCharts` from missing scripts | Replace with a packaged chart module taking probe data, current simulation time and display ranges. |
| `${[d.id](http://d.id/)}` | Repair paste corruption; create label nodes with `textContent` rather than interpolating metadata into `innerHTML`. |
| Markdown embedded in the README link | Replace with an actual relative provenance/report link. |
| Lecture navigation, `lecture:*` messages and wildcard parent messaging | Remove lecture behavior; keep an optional compact layout parameter for future embedding. |
| Statements that all results are uDALES LES | Generate captions from backend, field, member and processing metadata; label neural results as surrogate predictions. |
| Always-visible video download | Offer only generated media and correctly label MIME/extension; support a still image when movies are unavailable. |

Separate the template into `index.html`, `viewer.css`, `viewer.js` and
`probe_charts.js`, packaged under `pyurbanair.visualization.web`. Keep vanilla
HTML/CSS/JS and locally bundled assets; no frontend framework or CDN is needed.
Preserve accessible buttons, labels and keyboard navigation. Keep native video
controls until the custom controller initializes successfully. Add loading,
missing-media and playback-error states, and resize canvases for display density.

**2. Rendering and data services**

Create `src/pyurbanair/visualization/` with a small public service for validating
requests, reading artifacts, rendering and assembling bundles. It contains the
HTML assets and remains usable from Python/CLI without MCP. The separate
`libs/mcp_server` package contains only tool adapters and protocol response
conversion. Rendering runs in isolated workers managed by `pyurbanair.jobs`.

Use completed-run artifact IDs to identify state, sampled parameters, geometry
and masks. Support consolidated files and indexed per-member/per-window files.
Read only the selected members, times and needed spatial regions, with bounded
frame caches. Do not load the entire ensemble into RAM or send full volume data
to the browser. Leave source artifacts untouched.

Normalize vector components before computing derived fields:

| Output grid | Adaptation |
| --- | --- |
| LBM / neural regular `x,y,z` | Read named coordinates and transpose explicitly; preserve actual physical positions. |
| uDALES staggered `xt/xm`, `yt/ym`, `zt/zm` | Collocate components using their physical coordinates and a documented boundary policy. |
| PALM `u` on `xu`, `v` on `yv` | Collocate horizontal components before combining them with `w`. |

Make this a dependency-light xarray/numpy layer. Importing backend grid utilities
can trigger backend setup, so extract any reusable pure logic rather than import
whole solvers or assimilation scripts. Equal array shapes do not prove equal
coordinates. The current magnitude helper combines `.values` before collocation;
it must not be used blindly for quantitative visualization.

Use explicit `blanking` conventions or a mask derived from the matching geometry.
Do not infer solid cells from zero velocity. Keep missing fluid data distinct
from solids and valid zeros. Render gaps honestly and record a missing-mask
warning when geometry is unavailable. Validate units, coordinate orientation,
time monotonicity and STL alignment. Slice/sampling requests outside the domain
or inside solids need clear errors or invalid flags, not silent relocation.

Define `horizontal_speed = sqrt(u²+v²)` and `speed = sqrt(u²+v²+w²)` separately.
Compute them after collocation/interpolation. For ensemble statistics, record
whether the quantity is mean speed or speed of mean velocity; they differ.
Require an explicit member or reduction for an ensemble. Use real coordinates
and the actual selected height in labels, even when choosing the nearest cell.

**3. Two rendering backends**

For 2D, use Matplotlib's noninteractive Agg backend and xarray. Existing
[`visualize_forward_state`](../../scripts/_common.py) and
[`animate_height_panels`](../../../src/pyurbanair/utils/animation_utils.py) provide
plotting precedents, but their fixed layouts, eager loading and grid assumptions
need adaptation. Extract useful pure helpers instead of importing all script
dependencies. Recreate the supplied movie's layout with configurable horizontal
speed planes, a vertical `w` section, geometry outlines and probe markers.
Also support a single requested field/plane snapshot.

For 3D, implement a new focused PyVista/VTK renderer. Load matching STL geometry,
place a speed-colored slice and draw instantaneous streamlines seeded at
configured fluid locations. Use a fixed or prescribed camera and record the
geometry scale; default to no vertical exaggeration. Integrate each streamline
through one instantaneous vector field and stop it at obstacles, invalid regions
and domain boundaries. Reuse deterministic seeds across frames. Label the lines
as instantaneous streamlines, not particle trajectories or streaklines.
PyVista exposes seeded streamline integration and image capture, making it a
suitable building block for this design.
[Streamline API](https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.streamlines_from_source),
[image capture](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.screenshot).

Use a rectilinear grid for nonuniform axes and explicit data ordering; do not
assume cubic uniform voxels. Define the conversion from cell centers to VTK
sampling points and test it with a known vector field. Respect obstacle masks
during interpolation and integration, rather than hiding erroneous lines behind
the building mesh. Bound seed count, integration steps, spatial resolution and
frame count; cache static geometry and close rendering contexts on cancellation.

Add PyVista/VTK as a new optional `visualization-3d` feature with a tested version
and dedicated local rendering environment. Do a small offscreen readiness probe
in a child process. Headless operation depends on the installed VTK build and
graphics libraries; verify actual rendering on the target platform instead of
assuming package installation suffices.
[PyVista installation and offscreen requirements](https://docs.pyvista.org/getting-started/installation).
The 2D path and MCP startup must work without this feature. This design introduces
no dependency on the retiring renderer or its environment.

Encode both renderers' frames using one new ffmpeg wrapper with fixed arguments,
H.264 MP4, browser-compatible pixel format and progressive playback metadata.
Report encoder availability before starting a movie job. Always produce PNG
posters/selected snapshots and probe data independently of video encoding. Do
not silently serve a GIF under an MP4 filename when ffmpeg is absent. A single
stored state produces a still view rather than an invented evolving movie.

**4. Render configuration and probe synchronization**

Expose renderer presets and all supported settings through the MCP configuration
inspection tools, using new `conf/visualization/` files. Define typed scientific
settings rather than accepting arbitrary Python callbacks or encoder commands.
Requests include member/reduction, variable, time range/stride, slice axes and
physical positions, probe XYZ positions, color maps/ranges, FPS, simulation
seconds per video second, dimensions, camera and streamline seed/integration
settings. Record the effective config and report estimated frame count/cost
before queuing. A bounded `quicklook` preset is the default.

Color ranges are fixed for a clip. Derive an automatic range from selected fluid
data or accept an explicit range; record clipping and percentile policies. For
comparisons, use shared limits when requested and show when runs use different
ones. Use symmetric limits for signed fields when appropriate. Retain a clear
legend describing field and units.

Virtual probes are sampled simulation values, not field observations. Assign
stable IDs and colors, show markers at matching positions in rendered planes,
and record requested/actual coordinates and nearest-cell or interpolated
sampling. Sample vector components first and then derive speed. For multiple
heights, preserve requested horizontal positions; never move a probe into fluid
without reporting that change. Store invalid samples as gaps, not zeros.

Save probe values at actual solver output times in JSON and CSV. Video may
interpolate components between snapshots for smooth playback, but labels must
disclose that and must not imply additional simulated frames. Record a mapping
of video presentation times to simulation times for each view, including
nonuniform samples and rollout boundaries. Validate/deduplicate shared window
endpoints according to the run artifact contract; do not bridge missing windows
with fabricated flow.

The chart cursor uses mapped simulation time. Switching views preserves physical
time where their intervals overlap, rather than blindly copying video seconds.
Playback rate changes affect how quickly the movie advances, not its physical
time mapping. Update the cursor with `requestVideoFrameCallback` where available,
with a fallback and explicit seek/pause updates.
[Browser video frame callbacks](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback).
Use an update token to ignore stale media-load events when users switch views
quickly. Clamp seeks only after a finite duration becomes available.

**5. Versioned viewer bundle**

```text
<run>/visualizations/<visualization_id>/
  index.html, viewer.css, viewer.js, probe_charts.js
  viewer_manifest.json
  render_config.resolved.yaml, provenance.json
  probes.json, probes.csv
  previews/                      # PNG snapshots, plots and contact sheet
  media/                         # generated MP4s and posters
```

`viewer_manifest.json` owns versioned entries for source run/backend/case,
artifact identities, member/reduction, coordinate frame and units, domain/grid,
geometry/mask provenance, selected time interval, processing methods, warnings,
and available views. Each view records its media/poster paths, dimensions,
field/units/color range, slice/camera settings and frame-to-simulation-time map.
Probe metadata includes IDs, coordinates, sampling method and references to
series. Reference the exact geometry and state fingerprints used by the renderer.

Paths are relative to the bundle; all browser dependencies are packaged locally.
Persist renderer version, source hashes, effective settings and sampling choices.
Cache by those identities and options, including the viewer template version.
Publish a complete manifest atomically; distinguish partial media after failure.
Never overwrite a previously complete bundle just because a new render starts.

Bundles can be copied/exported for sharing and served offline with a local static
server. Avoid promising that direct `file://` opening will support JSON fetches.
The original source attachment remains a design reference and is not copied into
production bundles; only the repaired, tested template is shipped.

**6. MCP delivery and local browser access**

`render_simulation` validates inputs and queues a render job for a completed run.
It returns `job_id` and `visualization_id` promptly. Existing job tools cover
status, logs and cancellation. `get_visualization` returns bounded PNG content,
structured metadata and artifact links when ready. It can return a selected
snapshot or contact sheet so the agent can inspect the flow and request another
view. Expensive missing views are new render requests, not hidden synchronous
work inside retrieval. Original numerical results remain the source for precise
quantitative analysis.

MCP tools can return images and resource links. Use image content for immediate
visual inspection and registered artifact resources for larger media; a local
HTML path alone is insufficient for model inspection.
[MCP tool result types](https://modelcontextprotocol.io/specification/2025-06-18/server/tools).
Respect response-size limits and provide metadata even in clients without image
support. No full MP4 or volume arrays should be embedded in a tool result.

An on-demand, read-only asset server exposes completed bundles at a loopback URL
such as `http://127.0.0.1:<port>/view/<opaque-id>/`. It serves only registered
bundle files with validated paths, correct MIME types and HTTP Range support
for seeking. It has no launch/edit endpoints or directory listings. Validate
Host/Origin where applicable, use unguessable access URLs, and keep assets
same-origin with no external scripts. Do not expose the repository as a static
root. The viewer has no dependency on the MCP stdio connection staying open.

The supervisor manages this server's lifetime separately from render workers;
keep it alive while the viewer is in use with a documented idle timeout.
`get_visualization` restarts serving and returns a fresh URL after restart.
Return a clickable URL; opening a browser is an explicit local action, not an
automatic side effect of rendering. A remote/cloud client cannot be assumed to
reach localhost; MCP image/resource retrieval remains separate from that URL.

Inline embedding can later adapt the same frontend to the MCP Apps extension,
which uses UI resources and host communication. It requires client support and
media-access/CSP integration; it is not implied by returning an HTML resource.
[MCP Apps overview](https://apps.extensions.modelcontextprotocol.io/api/documents/overview.html).
Do not make this a requirement for the local browser workflow.

**7. Delivery and verification**

1. Build synthetic regular/staggered-grid fixtures and the normalized artifact
   reader. Test known vector fields, units, masks, axis orientation, boundary
   sampling, windows and ensemble reduction order without running CFD.
2. Implement PNG/slice rendering, probe extraction, timeline mapping, bounded
   video encoding and bundle manifests. Verify saved samples against xarray
   reference values and confirm source files are unchanged.
3. Rebuild the supplied viewer against generated manifests and probes. Browser
   tests cover changing views, seek/pause/restart, non-20-second clips, physical
   time synchronization, missing media, responsive layout and downloads.
4. Add new PyVista/VTK rendering with an offscreen smoke test and a tiny known
   flow around a masked obstacle. Verify STL alignment, seed reproducibility,
   streamline termination and output frames on supported machines.
5. Wire render jobs and tool results into MCP. Test cancellation and reconnect,
   cache invalidation, HTML metadata escaping, HTTP Range/containment and PNG
   responses. A render failure must leave the simulation status successful.
6. Exercise completed outputs from all four backends: ask the agent for a movie
   and probe chart, inspect a returned PNG, then change the height and camera.
   Confirm this creates only render jobs and the viewer describes the actual
   backend, selected member, interpolation and physical timeline.

Run browser tests against the new template, not the intentionally preserved
broken source. Keep dependency-heavy offscreen/video tests separately marked;
core normalization, manifest and probe tests run without PyVista or ffmpeg.

# Forward simulation visualization

The `visualization` library (`libs/visualization`, import `visualization`)
renders a completed forward run into a standalone HTML viewer bundle without
importing or running a CFD backend. It depends on neither `pyurbanair` nor the
MCP server, only on `evaluation` for the buildings' solid mask; MCP queues the same service as a separate visualization job, and a
failed render never changes the simulation's numerical artifacts. It is an
editable dependency of the `dev`, `mcp` and `rendering` Pixi environments.

```python
from visualization import BundleAssetServer, render

manifest = render(
    "/absolute/run_dir",  # holds state.nc from scripts/run_forward.py
    "/absolute/new/bundle",
    {
        "member": 0,  # omit for a single simulation
        "variable": "horizontal_speed",
        # Default: two horizontal maps, a vertical section, and matched probes.
        "stride": 2,
    },
)
server = BundleAssetServer()
url = server.register("/absolute/new/bundle")
print(url)
# Keep the process/server alive while browsing. Close explicitly when finished.
# server.close()
```

From the command line, `python -m visualization <run dir> <bundle dir>`
renders with the default options, the ensemble mean velocity for an ensemble
and the smallest stride that fits both `max_frames` and the 3D scene's size
limit, plus the interactive 3D view whenever every axis has at least two cells
(otherwise it renders 2D only);
`workflows/forward_workflow.sh` runs it in the `rendering` environment (for
ffmpeg) into `<run dir>/viewer`.
`python -m visualization --serve <bundle dir>` serves a bundle through
`BundleAssetServer` and prints its URL. Don't use `python -m http.server`: it
ignores HTTP byte ranges, so the viewer cannot seek in the movies and every
panel stays on its first frame.

The output directory must be new. `viewer_manifest.json` appears atomically only
after completion. Interrupted/failed directories contain partial products and
must not be served as completed bundles. Source hashes, viewer asset hashes,
options and a cache identity are recorded in `provenance.json`. Input changes
during rendering prevent publication. The service does not automatically reuse
existing bundles; callers can compare cache identities.

## Data and scientific conventions

The input is a run directory written by `scripts/run_forward.py`: one
`state.nc` holding every window on a single `time` axis (numeric simulation
seconds, strictly increasing), with an `ensemble` dimension only for ensembles.
Frames are read one at a time. The run's `config.yaml`, when present, supplies
the manifest's backend (`model.name`) and case (`case_name`). The SHA-256 of
`state.nc` is recorded and rechecked before publication. Coordinates must remain
unchanged across a rendered clip.

Velocity components are collocated using physical coordinates before computing
quantities. Supported aliases are regular `x/y/z`, uDALES `xt/xm`, `yt/ym`,
`zt/zm`, and PALM `xu/yv/zw` staggering. Descending axes are sorted. Linear
interpolation has no boundary extrapolation: unsupported boundaries stay missing.
Coordinates are metres and velocity is metres/second; absent coordinate units
use the repository convention, absent velocity units produce warnings, and
incompatible explicit units are rejected. `time` must be numeric seconds.

`horizontal_speed = sqrt(u²+v²)` and `speed = sqrt(u²+v²+w²)` are distinct.
Ensembles require an explicit coordinate-valued `member`, or `reduction`:

- `mean_velocity`: average components, then calculate their magnitude.
- `mean_speed`: calculate magnitudes per member, then average those magnitudes.

Both reductions preserve invalid samples rather than silently dropping members.
Components shown under `mean_speed` are still mean components; the derived speed
fields are the mean of member magnitudes. Spatially changing grids are rejected.

`blanking != 0` means solid; no buildings are inferred from velocity zeros.
No backend writes `blanking`, so without it the reader takes the solid cells
from the buildings' STL, `geometry` or else the run config's
`geometry.stl_path` (resolved from the run dir upwards, then the working
directory), through `evaluation.turbulence.stl_solid_mask`: the mask the state
metrics use. Without either the reader warns. Slices and virtual probes use the nearest cell
centre and retain requested and actual coordinates. Requests outside the
cell-centre domain are errors. Solid/missing probe samples are JSON `null` and
CSV empty cells. Valid zero velocity remains valid data. Probes are simulated
values, not observations. 2D obstacle outlines come from the explicit mask.

## Options and products

Discoverable presets live in `configs/visualization/quicklook.yaml` and
`configs/visualization/flow_3d.yaml`. The matplotlib figures of a run come from
`scripts/visualize_forward.py <run dir>` instead. `RenderOptions` /
`validate_options` define the supported settings; arbitrary callbacks, Python expressions and encoder command
arguments are not accepted.

The default recreates the supplied viewer's multi-panel arrangement: two
horizontal-speed maps at the lowest cell centre and 65% of the cell-centre
height range, a central y-normal vertical-velocity (`w`) section underneath,
and probe traces grouped by height alongside. All three planes appear together.
Cyan lines locate the section on the maps; A–C markers match the chart colors.
The same horizontal sample locations are used at each map height. Automatic
locations are deterministic grid samples, not observed sensors, and solid samples
remain gaps. Set `probes: []` to omit charts or supply explicit XYZ probes.

Each slice accepts `axis`, a physical `position` or a `fraction` of the axis's
cell-centre range, and an optional `variable`. Explicit slices replace the
default layout, so a single requested slice remains supported. For example:

```yaml
slices:
  - {axis: z, position: 2, variable: horizontal_speed}
  - {axis: z, position: 26, variable: horizontal_speed}
  - {axis: y, fraction: 0.5, variable: w}
```

Choose positions inside the saved domain; labels report actual nearest-cell
coordinates. Colors are fixed across a clip and shared by slices of the same
field. Signed velocity components use symmetric diverging colors; the side
section shows rising air in red and sinking air in blue. `color_limits` applies
to `variable`, while other slice fields retain their own scales. Limits clip
displayed colors only, not stored probe values. Products include PNG snapshots,
probe JSON/CSV and plots, optional H.264 MP4, the optional 3D scene, and the
HTML/CSS/JavaScript viewer.
Vertical-section images use a shorter, wide canvas within the requested image
dimensions so the dashboard does not shrink a portrait-sized image into its
wide lower panel. Axes retain equal physical scale without vertical stretching.

The bounds are 300 rendered frames, six slices, 32 probes, 1920×1080 pixels,
60 FPS, 10,000 probe sample times and eight million selected cells per frame
(including members for reductions). Defaults are smaller. Oversized requests
fail with a message to increase stride or narrow the time interval; frames are
never silently dropped to fit. A frame is read at a time and color estimation
uses a separate pass. This is bounded postprocessing, not a streaming CFD writer.

Probe files retain every saved output time in the selected interval even when
movie frames use a stride. Movies hold saved frames; they do not interpolate
new simulated states. Each view records every presentation timestamp and its
source simulation time, including nonuniform output intervals. Optionally,
`simulation_seconds_per_video_second` controls snapshot hold durations, rounded
to integer video frames and subject to the frame budget. Otherwise each selected
snapshot takes one frame at the requested FPS. A single saved state remains a
still. Physical-time cursors use this mapping; view changes preserve physical
time to the closest preceding available sample. Playback rate does not change
the mapping.

Without ffmpeg, the viewer plays the saved PNG sequence with the same timeline.
Every view records `snapshots` containing actual sample times and bundle paths;
`duration` describes presentation pacing even when there is no MP4. Older bundles
without snapshots retain an explicitly labeled still-image fallback. Encoding failures
also preserve those products and become manifest warnings. The encoder uses
fixed H.264/yuv420p/faststart arguments and a five-minute timeout.

The viewer uses only local assets. Its **2D slices / 3D flow** buttons switch
between the simultaneous slice dashboard and the interactive 3D view, preserving
physical time. The 3D button is disabled with an explanation when no 3D view
was requested or available. One play/pause, restart, seek and speed controller
synchronizes every visible panel and the probe cursor through each view's own
time mapping. Panels show their actual held sample times. Downloads are offered
per panel, with responsive charts and fullscreen support.
Metadata is inserted as text, never HTML. Native controls remain until the
custom controller initializes. Serve an exported bundle with a server that
supports HTTP byte ranges (`python -m visualization --serve`); playback seeks
in the movies, and direct `file://` JSON fetches are not supported.

## Optional interactive 3D

Set `render_3d: true` (the `flow_3d` preset) to add an interactive 3D panel:
the browser draws the saved 3D velocity with WebGL, so rendering needs no 3D
library, GPU or offscreen context; numpy writes a scene export. three.js r170
(MIT) is vendored as `web/vendor/three.module.min.js` with its licence beside
it; there are no other JavaScript dependencies and no CDN.

The export is `media/flow-3d/`, described in `render_3d.py`: `scene.json`
(cell-centre `x/y/z` in metres, saved `times`, limits, camera), `fluid.bin`
(Uint8 mask, 1 = fluid: `blanking == 0` and finite velocity in the first frame),
one `velocity/NNNNN.bin` per saved time and `buildings.bin`. Velocity is Int16
little-endian, planar u, v, w each (nz, ny, nx) with x fastest, with a per-frame
scale of max |component| / 32767: the quantization error is at most half that
scale. Solid and invalid cells are stored as 0. The selected saved times and
their pacing are those of the 2D movies (a saved time starts where its 2D frames
start). Before anything is written, the scene (mask plus 6 bytes per cell per
frame) must fit in 512 MiB; a larger request fails with a message to increase
stride or shorten the interval. A grid with fewer than two cells on an axis is
also an error.

`speed_limits` are [0, 99.5th percentile] of the full speed `sqrt(u²+v²+w²)`
over the fluid cells of all exported frames. `level_speed_limits` give each z
level [0, 99th percentile] over its fluid cells (null for a level without
fluid), so the height-slice colorbar stays fixed per level. Explicit
`color_limits` with `variable: speed` replace both: `speed_limits` and every
non-null level entry; the browser uses them exactly as exported (no rounding,
lower bound kept). Both percentiles come from a 4,096-bin histogram over the
written frames: at most one bin (peak speed / 4096) above the exact value.

Buildings come from the STL (`geometry`, or the run config's), in the same metre
coordinates without scaling or vertical exaggeration. Triangles are cut to the
saved field's cell-face bounds, so a larger source city cannot pull the camera
away from the flow data; the source hash, displayed bounds and triangle count
are kept in provenance. An STL that does not overlap the domain, a non-STL
geometry or an unreadable file drops only the 3D view: the 2D bundle is published
with a `3D view unavailable` warning and no partial scene. Overlapping bounds
cannot prove precise alignment, so the STL must match the run.
`mean_speed` is rejected for 3D (it needs velocity components).

The 3D panel has two displays, both coloured by full speed on one palette
(dark, indigo/violet, orange, yellow):

- **Height slice**: a glowing horizontal speed sheet at one z level, faded over
  solid cells, with short additive tracer streaks drifting over it and lit grey
  buildings casting shadows. The height slider picks the level (`z = … m`); it
  starts about 9 m above the ground, or at the first level that is at least 60%
  fluid when that is higher.
- **Full 3D**: a raymarched soft glow of the Q criterion (Q > 0, vortices) coloured
  by speed, with sparse streaks seeded mostly in those vortices and in the canopy.
  Its colour range is the largest exported `level_speed_limits` entry below
  twice the canopy height, used as exported, so the faster air above saturates
  (only a scene without level limits falls back to the 99th percentile of that
  speed in the first decoded state). Grids over 1.5 million cells are averaged over columns first.

Drag to rotate, scroll or pinch to zoom, right- or shift-drag (or two fingers)
to pan, double-click to reset the camera. `camera` (`[position, focal_point,
up_vector]`) sets the start view; by default it looks from upstream-left and
above, so +x flow runs left to right. **Save image** downloads the current view
as a PNG with its time, subtitle and colorbar. Without WebGL the panel says so
and the 2D slices remain.

Scientific caveats. The display blends linearly between the two saved states
around the current time; it shows no solver output between them. Streaks are
display tracers integrated (RK2) through that blended field, restarted on seeks
and when a display is switched on; in the height slice they move in the plane
and are capped in wall-clock speed at fast playback. They show direction and
relative speed, not exact particle paths. Seeking, restarting and view changes
snap to saved states; only playback passes between them. The Q glow is computed
in the browser from the quantized field, damped next to walls and zeroed at the
open lateral boundaries.

## Asset serving and validation

`BundleAssetServer` binds only `127.0.0.1` on an ephemeral port and serves only
registered completed bundle files through unguessable URL tokens. It validates
Host/Origin, applies a same-origin content security policy, resolves symlinks,
rejects traversal and directory listings, and implements GET/HEAD and one HTTP
byte range. File reads use 64 KiB chunks. No page may frame a bundle
(`frame-ancestors 'none'`) unless it was registered with
`register(bundle, frame_ancestor="http://127.0.0.1:<port>")`: then that one
loopback origin may (the browser app embeds its viewer so). There are no
mutation endpoints and it does not open a browser. The default idle timeout is 30 minutes; clients must
retrieve/register again after server expiry. The supervisor should own its
lifetime independently from the MCP client and render workers. A remote client
cannot assume it can access these loopback URLs.

Tests live in `tests/visualization/` (CI: `tests-visualization.yml`). Fast
tests in `test_forward_visualization.py` cover normalization, masks, ensemble
reduction order, the `state.nc` reader, probe values, PNG bundles and HTTP
range/containment. `test_forward_visualization_optional.py` is marked
`integration` and exercises real ffmpeg and optional Playwright/Chromium playback,
including the WebGL 3D panel; `test_forward_visualization_3d.py` covers the
scene export (round trip, mask, limits, STL clipping, size bound, 2D fallback)
and the asset allowlist for scene files and older bundles.
Run the integration tests in the `rendering` env (`pixi run -e rendering python -m pytest
tests/visualization -m integration`). Install Playwright and its Chromium
browser separately to run browser checks; those tests skip when the dependency
is absent.

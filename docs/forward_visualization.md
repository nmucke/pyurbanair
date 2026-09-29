# Forward simulation visualization

`pyurbanair.visualization` renders completed forward artifacts without importing
or running a CFD backend. MCP queues the same service as a separate visualization
job; a failed render never changes the simulation's numerical artifacts.

```python
from pyurbanair.visualization import BundleAssetServer, render

manifest = render(
    "/absolute/completed/run",
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

The output directory must be new. `viewer_manifest.json` appears atomically only
after completion. Interrupted/failed directories contain partial products and
must not be served as completed bundles. Source hashes, viewer asset hashes,
options and a cache identity are recorded in `provenance.json`. Input changes
during rendering prevent publication. The service does not automatically reuse
existing bundles; callers can compare cache identities.

## Data and scientific conventions

The reader prefers a complete `artifact_index.json` from the forward workflow,
including per-member/per-window files and global simulation seconds. It also
accepts a consolidated `state.nc`. Indexed SHA-256 values must match. Missing
windows, missing selected ensemble members, nonmonotone times, overlapping
windows beyond a common endpoint, and conflicting values at shared endpoints
are errors. Equal shared endpoints are deduplicated. Coordinates must remain
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
Missing masks produce a warning. Slices and virtual probes use the nearest cell
centre and retain requested and actual coordinates. Requests outside the
cell-centre domain are errors. Solid/missing probe samples are JSON `null` and
CSV empty cells. Valid zero velocity remains valid data. Probes are simulated
values, not observations. 2D obstacle outlines come from the explicit mask.

## Options and products

Discoverable presets live in `conf/visualization/quicklook.yaml` and
`conf/visualization/flow_3d.yaml`. `RenderOptions`/`validate_options` define the
supported settings; arbitrary callbacks, Python expressions and encoder command
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
probe JSON/CSV and plots, optional H.264 MP4, and the HTML/CSS/JavaScript viewer.
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
between the simultaneous slice dashboard and available 3D renders, preserving
physical time. The 3D button is disabled with an explanation when no 3D product
was requested or available. One play/pause, restart, seek and speed controller
synchronizes every visible panel and the probe cursor through each view's own
time mapping. Panels show their actual held sample times. Downloads are offered
per panel, with responsive charts and fullscreen support.
Metadata is inserted as text, never HTML. Native controls remain until the
custom controller initializes. Serve an exported bundle using a local static
server; direct `file://` JSON fetches are not supported.

## Optional 3D

The separate `rendering` Pixi environment supplies PyVista/VTK and ffmpeg.
MCP selects it for requested movies when installed, including 2D movies;
otherwise standard rendering falls back to `dev` and its available encoder.
Explicit 3D requests always select `rendering`.
Nothing in the standard rendering or MCP startup path imports VTK. Validate an
actual offscreen context in a child process:

```bash
pixi run -e rendering python -c \
  'from pyurbanair.visualization.render_3d import probe_offscreen; print(probe_offscreen())'
```

Set `render_3d: true` for a speed-colored slice with instantaneous streamlines,
optionally displaying a matching STL through `geometry`. Geometry uses the same
metre coordinates, unit scale and no vertical exaggeration. Nonoverlapping STL
bounds are rejected; overlapping bounds alone cannot establish precise alignment,
so users must supply the matching geometry. With STL geometry an explicit state
mask is required to prevent lines crossing buildings. The regular 2D products
remain available if the optional 3D render raises a recoverable error.
Displayed STL surfaces are clipped to the saved field's cell-face bounds; a
larger source city cannot force the camera to frame an area without flow data.
The original geometry hash and displayed bounds are retained in provenance.

Physical cell centres become rectilinear VTK sampling points, with x fastest
in flattened data. Nonuniform axes are supported. Every interpolation cell
touching a masked or invalid point is removed, so streamline integration stops
at the conservative fluid mesh boundary. Seeds are deterministic; explicit
seeds outside fluid interpolation cells are rejected. The default inlet grid
filters invalid seeds. Limits are 128 seeds and 2,000 integration steps.
`camera` is `[position, focal_point, up_vector]`. These are instantaneous
streamlines, not particle trajectories. Contexts close in `finally` blocks.

The 3D movie currently uses one video frame per selected saved time; the 2D
snapshot-hold pacing option does not alter 3D pacing. Each view's independent
mapping keeps browser synchronization correct. 3D color limits describe full
speed; 2D limits describe its selected variable. Optional 3D is verified locally
with PyVista 0.46.5 / VTK 9.5.2; offscreen support remains machine-dependent.

## Asset serving and validation

`BundleAssetServer` binds only `127.0.0.1` on an ephemeral port and serves only
registered completed bundle files through unguessable URL tokens. It validates
Host/Origin, applies a same-origin content security policy, resolves symlinks,
rejects traversal and directory listings, and implements GET/HEAD and one HTTP
byte range. File reads use 64 KiB chunks. There are no mutation endpoints and
it does not open a browser. The default idle timeout is 30 minutes; clients must
retrieve/register again after server expiry. The supervisor should own its
lifetime independently from the MCP client and render workers. A remote client
cannot assume it can access these loopback URLs.

Fast tests in `tests/test_forward_visualization.py` cover normalization, masks,
ensemble reduction order, windows/fingerprints, probe values, PNG bundles and
HTTP range/containment. `tests/test_forward_visualization_optional.py` is marked
`integration` and exercises real ffmpeg, optional Playwright/Chromium playback,
and offscreen VTK including vector ordering and masked streamline termination.
Install Playwright and its Chromium browser separately to run browser checks;
those tests skip when the dependency is absent.

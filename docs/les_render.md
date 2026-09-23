# les-render — cinematic LES animations (Unreal Engine + Blender)

`libs/les-render` (package `les_render`) turns an LES state file into a
**render bundle**: a folder of engine-neutral, render-ready assets plus a
`manifest.json` that describes them. Two renderers consume the same bundle:

* **Unreal Engine 5** (production look) — `les_render/unreal/` holds Unreal
  Editor Python scripts that import the bundle, build the level, lighting,
  materials and a Level Sequence, and render it with Movie Render Queue.
* **Blender** (headless preview, and the Alembic writer for UE) —
  `les_render/blender/` builds the same scene with `blender -b` and renders an
  mp4. This is the path that is tested on Linux.

It lives in its own pixi environment (`pixi run -e viz ...`) because
`openvdb` (conda-forge only) drags in boost/tbb.

## Quick start

```bash
# one-time: the viz env (linux-64 only; openvdb 11, numba, scikit-image)
pixi install -e viz

# whole pipeline: bundle + Unreal scripts + Blender preview mp4
pixi run -e viz python scripts/visualization/render_les.py \
    input=training_data/pyudales_idealized/state/train/sample_0000.nc

# a quick 8 s look at 960x540 from sim t=300 s
... time.t_start=300 time.duration=8 render.width=960 render.height=540

# other presets / looks
... render_preset=smoke_tunnel | vortex | comfort_map
... render.look=daylight

# Unreal only: bundle + Alembic caches, no Blender preview
... stages.blender=false stages.video=false stages.alembic=true

# re-render / re-encode an existing bundle without re-exporting
... output_dir=<bundle> stages.export=false stages.hud=false
```

Assemble a self-contained case folder from a training sample (symlinks the
state, STL and params, optionally a `render.yaml` template):

```bash
pixi run -e viz python scripts/visualization/make_render_case.py \
    training_data/pyudales_idealized/state/train/sample_0000.nc cases/va53 --render-yaml
pixi run -e viz python scripts/visualization/render_les.py input=cases/va53
```

Tests: `pixi run -e viz test-render` (the `libs/les-render/tests` suite; the
Blender and ffmpeg tests skip when those binaries are absent; CI runs it in
the `render-tests` job).

The bundle goes to `output_dir`, else `results/les_render/<case>/<preset>/`.
The Blender preview ends up in `<bundle>/preview/<case>.mp4`; the Unreal
workflow is in `<bundle>/unreal/README.md` (generated from
`les_render/unreal/README.md`).

### Presets (`conf/render_preset/`)

| Preset | Look | Layers | Camera |
|---|---|---|---|
| `cinematic` (default) | dark | streaklines, 20k pathline trails, speed-deviation glow volume, Q-criterion vortex cores, z = 2 m LIC wind map | establishing -> plan -> street -> wake -> establishing |
| `smoke_tunnel` | dark | dense inlet-rake + ground-line streaklines | establishing -> street -> establishing |
| `vortex` | dark | vortex cores (every frame) + sparse streaklines | establishing -> wake orbit |
| `comfort_map` | daylight | z = 2 m wind map (viridis + LIC) + low trails | plan -> street -> plan |

Design choices behind them (from a flow-vis literature pass): the flow is
unsteady, so lines are **streaklines** (smoke-wire dye) and **pathline
trails**, never instantaneous streamlines; colour is perceptually uniform
(`inferno` for speed, emissive on the dark look); the glow volume shows
`|speed - median speed|` so wakes and jets light up rather than the free
stream; vortex cores use a Q level held constant over the clip so surfaces
don't flicker; everything is masked by `blanking` so nothing is drawn inside
buildings; the HUD keeps a clock, an inflow compass and colorbars so the
stylised shots stay quantitative.

### Cost (96x48x16 case, loaded 32-core box)

Per video frame: streaklines ~0.3 s, 20k trails ~0.25 s, glow volume ~0.9 s
(every 4th frame), vortex isosurfaces ~2 s (every 2nd), LIC slice ~1 s
(4 workers in parallel), HUD ~15 ms. Disk: trails dominate at ~7.7 MB per
frame (20k x 24 points), so a 50 s clip at 30 fps is ~12 GB; the other layers
together are ~1 MB per frame.

## Blender preview

`les_render.blender_runner.run_blender(bundle, ...)` (the `blender` and
`alembic` stages, in one Blender launch when both are on) runs `blender -b -P les_render/blender/build_scene.py -- --bundle DIR ...`;
the scripts use only `bpy` + numpy, never `les_render`. Useful flags (also via
`blender.*` in the Hydra config): `--engine eevee|cycles`, `--samples`,
`--frames a:b`, `--look dark|daylight`, `--layers a,b`, `--export-alembic`,
`--save-blend` (writes `preview/scene.blend` for interactive tweaking).
Frames go to `<bundle>/preview/frames/%04d.png` with file `0000` = video
frame 0; `encode_preview` (the `video` stage) encodes exactly the frames
`blender.frames` selected (a contiguous `a:b` range), overlaying the HUD and
scaling it if the preview resolution differs. The camera is baked per frame from `cameras.sample_camera` with a
36 mm horizontal sensor (UE needs the matching 36 x 36*H/W mm filmback).

* **EEVEE** (default): ~2.4 s/frame at 1280x720 on the 3090 plus a ~15-50 s
  first-frame shader compile. Blender 4.2's EEVEE Next draws every hair curve
  with a fixed 8 points, so particle runs are chunked into <= 8-point curves
  for EEVEE (without this, streaklines render as zig-zag polylines); strands
  are hairlines (EEVEE ignores per-point radius).
* **Cycles**: true per-point radius, ~12 s/frame on the GPU. Fedora's Blender
  ships no CUDA kernels and nvcc 12.5 won't build against the system gcc 14:
  point `LES_CUDA_CCBIN` at a gcc 13 `g++` (e.g. from
  `pixi exec --spec gxx_linux-64=13.*`) for a one-time ~30 min kernel build
  (cached in `~/.cache/cycles/kernels`; keep the same `LES_CUDA_CCBIN` later,
  it is part of the cache key). Without it Cycles falls back to CPU.
* Alembic for UE (`--export-alembic`): particle layers as constant-topology
  curves (position + width only; no speed attribute survives Blender 4.2's
  curve writer, so UE colours strands along their length), isosurfaces as
  varying-topology meshes with an sRGB face-varying `Cd`. Trails dominate the
  size (~16 B per point per frame); `--alembic-max-lines N` subsamples.

## Unreal Engine

UE is not scriptable from Linux without an Epic-account install, so the UE
side is **untested here** (the Python maths is unit-tested against a fake
`unreal` module; every API name was checked against Epic's 5.5 Python docs).
What the pipeline hands UE:

| Bundle asset | UE asset / actor |
|---|---|
| `volumes/*/*.vdb` (OpenVDB 11, format 224, one float grid, cubic voxels) | animated Sparse Volume Texture -> Heterogeneous Volume + volume material |
| `alembic/<particles>.abc` (Blender stage, `stages.alembic=true`) | Groom + Groom Cache (on failure the build logs the manual re-import steps) |
| `alembic/<isosurface>.abc` | Geometry Cache (vertex colours) |
| `slices/*/*.png` | Img Media Source -> Media Texture on a plane (`unreal/meshes/*.glb`) |
| `geometry/*.glb` | static meshes (clay / dark matte materials) |
| `shots` (+ `unreal/camera_bake.json`) | CineCameraActors keyed per frame (36 mm filmback), Camera Cut track |
| `hud/` | composited after rendering with ffmpeg |

Build and render: `UnrealEditor-Cmd <proj> -ExecutePythonScript="<bundle>/unreal/build_scene.py --bundle <bundle> --quit"`,
then `python <bundle>/unreal/render.py --bundle <bundle> --project <proj>`
(Movie Render Queue, Path Tracer). `ue_commands.sh/.bat` chain build ->
render -> HUD composite. The generated README lists required plugins and
project settings, placement knobs if a volume comes in mirrored, and the
unverified API points to check on first use.

## Input: the case folder

Point the script at a folder (or directly at a state file):

| File | Required | Resolved from |
|---|---|---|
| state NetCDF (`u, v, w` on `(time, zt, yt, xt)`; optional `pres`, `blanking`) | yes | `state.nc`, else the only `*.nc` with `u, v, w` |
| building geometry STL (metres, same frame as `xt/yt/zt`) | no | `*.stl` in the folder, else `attrs["geometry_stl"]` in the folder or `<dataset>/geometries/`, else boxes from `blanking` |
| inflow parameters NetCDF (`inflow_angle`, `velocity_magnitude` vs `time`) | no | `params.nc`, else a `*.nc` with `inflow_angle`, else `<dataset>/param/<split>/<state name>` |
| `render.yaml` | no | per-case overrides (e.g. hand-tuned shots, `render_preset: {look: daylight}`). Precedence: preset/config < `render.yaml` < command line. It can override preset keys but not switch presets by name. |

`blanking` (1 = solid) is strongly recommended: raw solver output holds junk
velocity inside buildings, and the mask is used to zero it, to keep particles
out of buildings and to cut isosurfaces/slices at walls. For the
`training_data/<dataset>/state/<split>/sample_XXXX.nc` layout all of the above
resolves automatically.

## Coordinate frame and units

Everything in the bundle is in the **simulation frame**: metres, right-handed,
z-up, the same frame as the STL and the state file's `xt, yt, zt`. Engine
conversions happen in the renderer, never in the bundle:

* Blender: identity (Blender is right-handed, z-up, metres).
* Unreal: `(x, y, z)_m -> (100 x, -100 y, 100 z)_cm` (left-handed, z-up).
  glTF files are y-up per the glTF spec; UE's glTF importer converts them.

Internal arrays are indexed `(x, y, z)` (the NetCDF is `(z, y, x)`), matching
OpenVDB `[i, j, k]`.

## Timeline

Video frame `i` (0-based) shows sim time `t_start + i * dt` with
`dt = playback_speed / fps`; `playback_speed` is sim seconds per video
second. Fields between stored snapshots are interpolated with Catmull-Rom in
time (`FieldSeries(time_interpolation="cubic")`; linear blending makes the
time derivative jump at every snapshot, which kinks streaklines every
`U * dt_snapshot`) and upsampled tricubically in space (velocity first,
derived quantities after).

## Bundle layout

```
bundle/
  manifest.json
  geometry/buildings.{glb,obj}  geometry/ground.{glb,obj}
  volumes/<name>/<name>.<FFFF>.vdb        one float grid per file, grid name = <name>
  particles/<name>/<name>.<FFFF>.npz      polylines, see below
  isosurfaces/<name>/<name>.<FFFF>.ply    triangle mesh + per-vertex colour & value
  slices/<name>/<name>.<FFFF>.png         RGBA texture of a plane
  hud/hud.<FFFF>.png                      RGBA overlay at render resolution
  alembic/                                written by the Blender stage for UE
  unreal/                                 UE scripts copied here for convenience
  preview/                                Blender preview frames + mp4
```

`<FFFF>` is the zero-padded *file* index. Layers may be exported every
`frame_step` video frames; the renderer shows file `floor(i / frame_step)`
for video frame `i` (volumes/isosurfaces are heavier than particles).

### manifest.json (version 1)

```jsonc
{
  "version": 1,
  "case": {"name": "...", "state": "...", "geometry": "...|null", "params": "...|null"},
  "frame": {"units": "m", "handedness": "right", "up": "z"},
  "domain": {"lower": [x, y, z], "upper": [x, y, z], "spacing": [dx, dy, dz], "shape": [nx, ny, nz]},
  "timeline": {"fps": 30, "playback_speed": 20, "t_start": 0, "t_end": ..., "dt": ..., "n_frames": N,
               "frame_times": [...]},
  "render": {"width": 1920, "height": 1080, "look": "dark|daylight", "preset": "..."},
  "geometry": {"buildings": {"obj": "...", "glb": "..."}, "ground": {...},
               "buildings_bounds": [[...], [...]], "max_building_height": h, "footprints": [{"min", "max"}]},
  "inflow": {"time": [...], "angle_deg": [...], "speed": [...]} | null,
  "layers": [ <layer>, ... ],
  "shots": [ <shot>, ... ],
  "hud": {"pattern": "hud/hud.{frame:04d}.png", "frame_step": 1} | null
}
```

Every layer has `name`, `type`, `pattern` (a `str.format` template with
`frame`), `frame_step`, `n_files`, and the colour keys (merged flat into the
layer dict) `variable`, `range: [vmin, vmax]`, `colormap`,
`lut_linear_rgb: [[r, g, b] x 256]`. Type-specific keys:

* `type: "volume"` — `grid` (VDB grid name), `voxel_size` (m), `origin`
  (world position of voxel (0,0,0)'s centre), `shape`, `density_range`
  (values mapped to zero/full density), `emission_strength`,
  `density_scale`, `transform` (`linear` | `abs_excess` | `gamma_norm`: how
  `variable` was mapped to the stored grid). Values below `density_range[0]` are inactive voxels, so files
  stay sparse.
* `type: "particles"` — `kind` (`"streaklines"` | `"trails"`), `n_lines`,
  `points_per_line`, `radius` (m), `emission_strength`. Each npz holds
  `points (n_lines, points_per_line, 3) float32` (world metres, index 0 = head),
  `speed (n_lines, points_per_line) float16`, and
  `alpha (n_lines, points_per_line) float16` in [0, 1] (0 = hidden: dead,
  not yet born, or inside a building). Topology (`n_lines`,
  `points_per_line`) is constant over all files, so Alembic/Groom caches stay
  valid; visibility is carried by `alpha`. **Segment rule**: draw segment
  `k -> k+1` with opacity `min(alpha[k], alpha[k+1])`. Every invalid segment
  (death, wrap-around, entering a building) has alpha 0 at both ends, and
  hidden points are collapsed onto the nearest live point, so scaling the tube
  radius by alpha is a safe fallback. For streaklines index `k` is "the k-th
  newest release from this emitter", not a fixed particle, so don't derive
  per-point velocity/motion blur from index continuity.
* `type: "isosurface"` — `iso_variable`, `level`. PLY vertices carry
  `red, green, blue` (sRGB uint8 from the colour block) and a float
  `value` property (the colour variable).
* `type: "slice"` — `axis` (`"x" | "y" | "z"`), `position` (m), `extent`
  `[[u0, v0], [u1, v1]]` in world metres of the two in-plane axes (for
  `axis: "z"`: x then y), texture `resolution [w, h]`. PNG row 0 is the
  *max-v* edge (standard image orientation); alpha 0 inside solids.

### Shots

```jsonc
{"name": "establishing", "start": 0, "end": 239,          // video frames, inclusive
 "keys": [{"frame": 0, "location": [x, y, z], "target": [x, y, z],
           "focal_length_mm": 35, "fstop": 8.0}],
 "layers": ["streaklines", "speed_glow"]}                 // optional: visible layers (default all)
```

Camera keys are world-space (sim frame); renderers interpolate location and
target with Catmull-Rom through the keys and aim the camera at `target`. Easing
is applied once per shot (slow in/out at the shot's first and last frame only),
so the camera keeps its speed through interior keys
(`les_render.cameras.sample_camera` is the reference implementation). Shots tile the
timeline without gaps.

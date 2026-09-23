"""Write a small contract-conformant render bundle for the Blender renderer tests.

Stand-in for the full ``les_render.export`` pipeline: it only uses the core
modules (case / fields / geometry / colormaps / timeline) plus deliberately
simple layer exporters (explicit-Euler particle trails, a vorticity VDB, a
Q-criterion marching-cubes isosurface, a colormapped speed slice near the
ground, a text HUD and three hand-placed shots). The output follows
``docs/les_render.md`` so the Blender stage can be developed and tested
without the real layer exporters.

Run in the ``viz`` environment::

    pixi run -e viz python libs/les-render/tests/blender_fixture.py OUT_DIR \
        [--n-frames 60] [--n-lines 20000] [--look dark] [--width 960 --height 540]
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, cast

import numpy as np
from les_render import colormaps
from les_render.case import discover_case
from les_render.fields import FieldSeries, robust_range, scalar_field
from les_render.geometry import export_geometry
from les_render.timeline import make_timeline

REPO = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_STATE = REPO / "training_data/pyudales_idealized/state/train/sample_0000.nc"


# -- layers ---------------------------------------------------------------------


def _particles(
    fields: FieldSeries,
    frame_times: np.ndarray,
    out: pathlib.Path,
    n_lines: int,
    points_per_line: int,
    name: str = "trails",
    preroll: float = 40.0,
) -> dict[str, Any]:
    """Trails: each line is one particle's recent history (index 0 = head)."""
    rng = np.random.default_rng(0)
    g = fields.grid
    lo, hi = g.lower, g.upper
    zmax = min(hi[2], 40.0)

    def spawn(n: int) -> np.ndarray:
        p = rng.uniform((lo[0], lo[1], 0.5), (hi[0], hi[1], zmax), size=(n, 3))
        # bias spawns upstream so the domain fills from the inlet
        p[:, 0] = lo[0] + (hi[0] - lo[0]) * rng.power(0.6, n) * 0.999
        return p

    pos = spawn(n_lines)
    bad = fields.is_solid(pos)
    while bad.any():
        pos[bad] = spawn(int(bad.sum()))
        bad = fields.is_solid(pos)
    age = np.zeros(n_lines, dtype=np.int64)  # frames since (re)spawn
    dt = float(frame_times[1] - frame_times[0])
    hist = np.repeat(pos[None], points_per_line, axis=0)  # (P, N, 3) ring, 0 = newest
    hist_speed = np.zeros((points_per_line, n_lines), dtype=np.float32)
    n_pre = int(np.ceil(preroll / dt))
    times = np.concatenate([frame_times[0] - dt * np.arange(n_pre, 0, -1), frame_times])
    (out / name).mkdir(parents=True, exist_ok=True)
    taper = (1.0 - np.arange(points_per_line) / (points_per_line - 1)) ** 1.2
    all_speed = []
    for step, t in enumerate(times):
        if step > 0:
            for sub in range(2):
                vel = fields.sample_velocity(pos, t - dt + sub * dt / 2)
                pos = pos + vel * dt / 2
            age += 1
            dead = (
                ~fields.in_domain(pos)
                | fields.is_solid(pos)
                | (rng.random(n_lines) < 0.004)
            )
            if dead.any():
                k = int(dead.sum())
                fresh = spawn(k)
                fresh[:, 0] = lo[0] + rng.uniform(0, 8.0, k)
                pos[dead] = fresh
                age[dead] = 0
        spd = np.linalg.norm(fields.sample_velocity(pos, t), axis=1)
        hist = np.roll(hist, 1, axis=0)
        hist_speed = np.roll(hist_speed, 1, axis=0)
        hist[0] = pos
        hist_speed[0] = spd
        i = step - n_pre
        if i < 0:
            continue
        alive = np.arange(points_per_line)[:, None] <= age[None, :]  # (P, N)
        alpha = (taper[:, None] * alive).T.astype(np.float16)
        pts = np.transpose(hist, (1, 0, 2)).astype(np.float32).copy()
        pts[alpha == 0] = 0.0  # garbage for hidden points: the renderer must cope
        np.savez_compressed(
            out / name / f"{name}.{i:04d}.npz",
            points=pts,
            speed=hist_speed.T.astype(np.float16),
            alpha=alpha,
        )
        all_speed.append(spd)
    vmin, vmax = robust_range(np.concatenate(all_speed), 1, 99)
    return {
        "name": name,
        "type": "particles",
        "kind": "trails",
        "pattern": f"particles/{name}/{name}.{{frame:04d}}.npz",
        "frame_step": 1,
        "n_files": len(frame_times),
        "n_lines": n_lines,
        "points_per_line": points_per_line,
        "radius": 0.12,
        "emission_strength": 4.0,
        **colormaps.layer_color_spec("inferno", 0.0, vmax, "speed"),
    }


def _write_vdb(
    path: pathlib.Path,
    name: str,
    values: np.ndarray,
    origin: np.ndarray,
    voxel: float,
    threshold: float,
) -> None:
    try:
        import pyopenvdb as openvdb
    except ImportError:
        import openvdb

    arr = np.where(values >= threshold, values, 0.0).astype(np.float32)
    grid = openvdb.FloatGrid(0.0)
    grid.copyFromArray(arr, ijk=(0, 0, 0), tolerance=0.0)
    grid.prune()
    m = [[voxel, 0, 0, 0], [0, voxel, 0, 0], [0, 0, voxel, 0], [*origin.tolist(), 1.0]]
    grid.transform = openvdb.createLinearTransform(m)
    grid.name = name
    grid.gridClass = openvdb.GridClass.FOG_VOLUME
    openvdb.write(str(path), grids=[grid])


def _volume(
    fields: FieldSeries,
    frame_times: np.ndarray,
    out: pathlib.Path,
    step: int,
    name: str = "vorticity",
) -> dict[str, Any]:
    files = frame_times[::step]
    (out / name).mkdir(parents=True, exist_ok=True)
    vals0, grid = scalar_field(
        fields, "vorticity_magnitude", float(files[0]), upsample=2
    )
    lo_hi = robust_range(vals0, 70, 99.7)
    for i, t in enumerate(files):
        vals, grid = scalar_field(fields, "vorticity_magnitude", float(t), upsample=2)
        _write_vdb(
            out / name / f"{name}.{i:04d}.vdb",
            name,
            vals,
            grid.origin,
            float(grid.spacing[0]),
            lo_hi[0],
        )
    return {
        "name": name,
        "type": "volume",
        "pattern": f"volumes/{name}/{name}.{{frame:04d}}.vdb",
        "frame_step": step,
        "n_files": len(files),
        "grid": name,
        "voxel_size": float(grid.spacing[0]),
        "origin": grid.origin.tolist(),
        "shape": list(grid.shape),
        "density_range": list(lo_hi),
        "density_scale": 0.08,
        "emission_strength": 1.5,
        **colormaps.layer_color_spec("magma", *lo_hi, "vorticity_magnitude"),
    }


def _write_ply(
    path: pathlib.Path,
    verts: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    value: np.ndarray,
) -> None:
    vdt = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("value", "<f4"),
        ]
    )
    v = np.empty(len(verts), vdt)
    v["x"], v["y"], v["z"] = verts.T
    v["red"], v["green"], v["blue"] = rgb.T
    v["value"] = value
    fdt = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    f = np.empty(len(faces), fdt)
    f["n"] = 3
    f["i"] = faces
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float value\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(v.tobytes())
        fh.write(f.tobytes())


def _isosurface(
    fields: FieldSeries,
    frame_times: np.ndarray,
    out: pathlib.Path,
    step: int,
    name: str = "q_iso",
) -> dict[str, Any]:
    from les_render.fields import trilinear
    from skimage.measure import marching_cubes

    files = frame_times[::step]
    (out / name).mkdir(parents=True, exist_ok=True)
    q0, _ = scalar_field(fields, "q_criterion", float(files[0]), upsample=2)
    level = float(np.percentile(q0[q0 > 0], 90))
    vrange = (0.0, 9.0)
    for i, t in enumerate(files):
        q, grid = scalar_field(fields, "q_criterion", float(t), upsample=2)
        spd, _ = scalar_field(fields, "speed", float(t), upsample=2)
        verts, faces, _, _ = marching_cubes(q, level)
        world = grid.origin + verts * grid.spacing
        val = trilinear(spd, verts)
        rgb = np.rint(colormaps.apply(val, *vrange, "viridis") * 255).astype(np.uint8)
        _write_ply(
            out / name / f"{name}.{i:04d}.ply",
            world.astype(np.float32),
            faces[:, ::-1].astype(np.int32),
            rgb,
            val.astype(np.float32),
        )
    return {
        "name": name,
        "type": "isosurface",
        "pattern": f"isosurfaces/{name}/{name}.{{frame:04d}}.ply",
        "frame_step": step,
        "n_files": len(files),
        "iso_variable": "q_criterion",
        "level": level,
        **colormaps.layer_color_spec("viridis", *vrange, "speed"),
    }


def _slice(
    fields: FieldSeries,
    frame_times: np.ndarray,
    out: pathlib.Path,
    name: str = "ground_speed",
    position: float = 2.0,
) -> dict[str, Any]:
    import matplotlib.image

    (out / name).mkdir(parents=True, exist_ok=True)
    vrange = (0.0, 7.0)
    for i, t in enumerate(frame_times):
        spd, grid = scalar_field(fields, "speed", float(t), upsample=4)
        k = int(np.argmin(np.abs(grid.z - position)))
        s = spd[:, :, k]  # (x, y)
        solid = s == 0.0
        rgba = np.zeros(s.shape + (4,), dtype=np.float32)
        rgba[..., :3] = colormaps.apply(s, *vrange, "turbo")
        rgba[..., 3] = np.where(solid, 0.0, 1.0)
        img = np.transpose(rgba, (1, 0, 2))[::-1]  # rows = y (row 0 = max y), cols = x
        matplotlib.image.imsave(out / name / f"{name}.{i:04d}.png", img)
    lo, hi = grid.lower, grid.upper
    return {
        "name": name,
        "type": "slice",
        "pattern": f"slices/{name}/{name}.{{frame:04d}}.png",
        "frame_step": 1,
        "n_files": len(frame_times),
        "axis": "z",
        "position": position,
        "extent": [[lo[0], lo[1]], [hi[0], hi[1]]],
        "resolution": [int(s.shape[0]), int(s.shape[1])],
        **colormaps.layer_color_spec("turbo", *vrange, "speed"),
    }


def _hud(
    frame_times: np.ndarray, out: pathlib.Path, width: int, height: int
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (out / "hud").mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_alpha(0.0)
    txt = fig.text(
        0.03, 0.05, "", color="white", fontsize=height / 40, family="monospace"
    )
    for i, t in enumerate(frame_times):
        txt.set_text(f"t = {t:7.1f} s")
        fig.savefig(out / "hud" / f"hud.{i:04d}.png", transparent=True, dpi=100)
    plt.close(fig)
    return {"pattern": "hud/hud.{frame:04d}.png", "frame_step": 1}


def _shots(
    n: int, geo: dict[str, Any], domain: dict[str, Any], layer_names: list[str]
) -> list:
    lo, hi = np.array(domain["lower"]), np.array(domain["upper"])
    (bx0, by0, _), (bx1, by1, bz1) = geo["buildings_bounds"]
    c = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2, bz1 * 0.4])
    a, b = n // 3, 2 * n // 3
    return [
        {
            "name": "establishing",
            "start": 0,
            "end": a - 1,
            "keys": [
                {
                    "frame": 0,
                    "location": [lo[0] - 90, lo[1] - 110, 150],
                    "target": c.tolist(),
                    "focal_length_mm": 30,
                    "fstop": 11.0,
                },
                {
                    "frame": a - 1,
                    "location": [lo[0] - 40, lo[1] - 140, 120],
                    "target": (c + [20, 0, 0]).tolist(),
                    "focal_length_mm": 32,
                    "fstop": 11.0,
                },
            ],
        },
        {
            "name": "street",
            "start": a,
            "end": b - 1,
            "keys": [
                {
                    "frame": a,
                    "location": [bx1 + 40, c[1] - 55, 9],
                    "target": [bx1 - 10, c[1], 12],
                    "focal_length_mm": 24,
                    "fstop": 2.8,
                },
                {
                    "frame": b - 1,
                    "location": [bx1 + 55, c[1] - 40, 11],
                    "target": [bx1 - 5, c[1] + 5, 12],
                    "focal_length_mm": 24,
                    "fstop": 2.8,
                },
            ],
        },
        {
            "name": "overview",
            "start": b,
            "end": n - 1,
            "keys": [
                {
                    "frame": b,
                    "location": [c[0] + 60, c[1] + 180, 170],
                    "target": (c + [40, 0, 0]).tolist(),
                    "focal_length_mm": 28,
                    "fstop": 11.0,
                }
            ],
            "layers": [x for x in layer_names if x != "vorticity"],
        },
    ]


def make_fixture_bundle(
    out: pathlib.Path,
    state: pathlib.Path = DEFAULT_STATE,
    n_frames: int = 60,
    n_lines: int = 20000,
    points_per_line: int = 16,
    look: str = "dark",
    width: int = 960,
    height: int = 540,
    heavy_step: int = 2,
    t_start: float = 300.0,
    layers: tuple[str, ...] = ("particles", "volume", "isosurface", "slice"),
) -> dict[str, Any]:
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    case = discover_case(state)
    fields = FieldSeries(case.open_state())
    tl = make_timeline(
        fields.times,
        fps=30,
        playback_speed=20,
        t_start=t_start,
        duration=(n_frames - 0.5) / 30,
    )
    g = fields.grid
    manifest: dict[str, Any] = {
        "version": 1,
        "case": {
            "name": case.name,
            "state": str(case.state_path),
            "geometry": str(case.geometry_path) if case.geometry_path else None,
            "params": None,
        },
        "frame": {"units": "m", "handedness": "right", "up": "z"},
        "domain": {
            "lower": g.lower.tolist(),
            "upper": g.upper.tolist(),
            "spacing": g.spacing.tolist(),
            "shape": list(g.shape),
        },
        "timeline": tl.to_dict(),
        "render": {"width": width, "height": height, "look": look, "preset": "fixture"},
    }
    manifest["geometry"] = export_geometry(
        case.buildings(), g.lower, g.upper, out, ground_margin=150.0
    )
    manifest["inflow"] = None
    ft = tl.frame_times
    lay = []
    if "particles" in layers:
        lay.append(_particles(fields, ft, out / "particles", n_lines, points_per_line))
    if "volume" in layers:
        lay.append(_volume(fields, ft, out / "volumes", heavy_step))
    if "isosurface" in layers:
        lay.append(_isosurface(fields, ft, out / "isosurfaces", heavy_step))
    if "slice" in layers:
        lay.append(_slice(fields, ft, out / "slices"))
    manifest["layers"] = lay
    manifest["shots"] = _shots(
        tl.n_frames, manifest["geometry"], manifest["domain"], [x["name"] for x in lay]
    )
    manifest["hud"] = _hud(ft, out, width, height)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def make_real_bundle(
    out: pathlib.Path,
    state: pathlib.Path = DEFAULT_STATE,
    n_frames: int = 60,
    look: str = "dark",
    width: int = 960,
    height: int = 540,
    t_start: float = 300.0,
    trail_count: int = 50000,
    workers: int = 4,
) -> dict[str, Any]:
    """Same bundle shape, but built with the teammates' real layer exporters,
    camera rig and HUD (canonical layer names: streaklines, trails,
    speed_glow, vortices, ground_speed)."""
    import logging
    import time

    from les_render.cameras import make_shots
    from les_render.hud import inflow_block, render_hud
    from les_render.isosurfaces import export_isosurfaces
    from les_render.particles import export_particles
    from les_render.slices import export_slices
    from les_render.volumes import export_volumes

    logging.basicConfig(level=logging.INFO)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    case = discover_case(state)
    fields = FieldSeries(case.open_state())
    tl = make_timeline(
        fields.times,
        fps=30,
        playback_speed=20,
        t_start=t_start,
        duration=(n_frames - 0.5) / 30,
    )
    g = fields.grid
    manifest: dict[str, Any] = {
        "version": 1,
        "case": {
            "name": case.name,
            "state": str(case.state_path),
            "geometry": str(case.geometry_path) if case.geometry_path else None,
            "params": str(case.params_path) if case.params_path else None,
        },
        "frame": {"units": "m", "handedness": "right", "up": "z"},
        "domain": {
            "lower": g.lower.tolist(),
            "upper": g.upper.tolist(),
            "spacing": g.spacing.tolist(),
            "shape": list(g.shape),
        },
        "timeline": tl.to_dict(),
        "render": {
            "width": width,
            "height": height,
            "look": look,
            "preset": "fixture-real",
        },
    }
    manifest["geometry"] = export_geometry(
        case.buildings(), g.lower, g.upper, out, ground_margin=0.0
    )
    params = case.open_params()
    manifest["inflow"] = inflow_block(params) if params is not None else None
    fp = manifest["geometry"]["footprints"]
    jobs = [
        (export_particles, {"name": "streaklines", "kind": "streaklines"}),
        (export_particles, {"name": "trails", "kind": "trails", "counts": trail_count}),
        (export_volumes, {"name": "speed_glow"}),
        (export_isosurfaces, {"name": "vortices"}),
        (export_slices, {"name": "ground_speed"}),
    ]
    manifest["layers"] = []
    cache = out / "_layer_cache"  # resume: skip layers exported by an earlier run
    cache.mkdir(exist_ok=True)
    for fn, spec in jobs:
        t0 = time.perf_counter()
        spec = cast(dict[str, Any], spec)
        cached = cache / f"{spec['name']}.json"
        if cached.is_file():
            manifest["layers"].append(json.loads(cached.read_text()))
            continue
        try:
            layer = fn(fields, tl, {**spec, "workers": workers, "footprints": fp}, out)
        except Exception as exc:  # teammate module mid-edit: fall back to the stand-in
            print(
                f"layer {spec['name']} failed ({exc!r}); using the stand-in exporter",
                flush=True,
            )
            ft = tl.frame_times
            stand_in = {
                "particles": lambda: _particles(
                    fields, ft, out / "particles", 20000, 16, spec["name"]
                ),
                "volume": lambda: _volume(fields, ft, out / "volumes", 2, spec["name"]),
                "isosurface": lambda: _isosurface(
                    fields, ft, out / "isosurfaces", 2, spec["name"]
                ),
                "slice": lambda: _slice(fields, ft, out / "slices", spec["name"]),
            }
            kind = {
                "export_particles": "particles",
                "export_volumes": "volume",
                "export_isosurfaces": "isosurface",
                "export_slices": "slice",
            }[fn.__name__]
            layer = stand_in[kind]()
        cached.write_text(json.dumps(layer))
        manifest["layers"].append(layer)
        print(f"layer {spec['name']}: {time.perf_counter() - t0:.1f} s", flush=True)
    manifest["shots"] = make_shots(manifest["geometry"], manifest["domain"], tl)
    manifest["hud"] = None
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    manifest["hud"] = render_hud(manifest, {"workers": workers}, out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--state", type=pathlib.Path, default=DEFAULT_STATE)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--n-lines", type=int, default=20000)
    ap.add_argument("--points-per-line", type=int, default=16)
    ap.add_argument("--look", default="dark")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--layers", default="particles,volume,isosurface,slice")
    ap.add_argument("--real", action="store_true", help="use the real layer exporters")
    a = ap.parse_args()
    if a.real:
        m = make_real_bundle(
            a.out, a.state, a.n_frames, a.look, a.width, a.height, trail_count=a.n_lines
        )
    else:
        m = make_fixture_bundle(
            a.out,
            a.state,
            a.n_frames,
            a.n_lines,
            a.points_per_line,
            a.look,
            a.width,
            a.height,
            layers=tuple(a.layers.split(",")),
        )
    print(json.dumps({k: m[k] for k in ("domain",)}, default=str)[:400])
    print("layers:", [(x["name"], x["n_files"]) for x in m["layers"]])

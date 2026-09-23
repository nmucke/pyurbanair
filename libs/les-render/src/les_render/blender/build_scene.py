"""Build (and optionally render / export) a Blender scene from a render bundle.

Run with Blender's own Python (bpy + numpy + stdlib only; never imports
``les_render``)::

    blender -b --factory-startup -P build_scene.py -- --bundle DIR \
        [--render] [--frames a:b | --frames 0,40,80] [--engine eevee|cycles] \
        [--samples N] [--width W --height H] [--look dark|daylight] \
        [--export-alembic] [--save-blend] [--out DIR] [--radius-scale S] \
        [--emission-scale S] [--glare S] [--motion-blur]

Outputs (all inside the bundle unless ``--out``):
    preview/frames/<FFFF>.png   one per rendered video frame (index = video frame)
    preview/timings.json        per-frame render wall times
    alembic/<layer>.abc         particle (curves) / isosurface (mesh) caches + buildings.abc
    preview/scene.blend         with ``--save-blend``; re-registers its frame handler on load

Scene frame ``f`` == video frame ``f`` (starting at 0), fps from the manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bpy  # noqa: E402
import bundle as bmod  # noqa: E402
import layers  # noqa: E402
import look as lookmod  # noqa: E402
import materials  # noqa: E402
import numpy as np  # noqa: E402


def log(msg: str) -> None:
    print(f"[les] {msg}", flush=True)


# -- args -----------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="build_scene.py")
    ap.add_argument("--bundle", required=True, type=pathlib.Path)
    ap.add_argument("--render", action="store_true")
    ap.add_argument(
        "--frames", default=None, help="'a:b' (inclusive), 'a:b:step' or 'f1,f2,...'"
    )
    ap.add_argument("--engine", default="eevee", choices=("eevee", "cycles"))
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--look", default=None, choices=(None, "dark", "daylight"))
    ap.add_argument("--export-alembic", action="store_true")
    ap.add_argument("--save-blend", action="store_true")
    ap.add_argument(
        "--alembic-max-lines",
        type=int,
        default=0,
        help="subsample particle lines in the groom caches (0 = all; caches grow ~16 B/point/frame)",
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="frames dir (default <bundle>/preview/frames)",
    )
    ap.add_argument("--radius-scale", type=float, default=1.0)
    ap.add_argument(
        "--max-strand-px",
        type=float,
        default=1.6,
        help="cap particle strand width to this many pixels (0 = off)",
    )
    ap.add_argument("--emission-scale", type=float, default=1.0)
    ap.add_argument("--volume-density-scale", type=float, default=1.0)
    ap.add_argument(
        "--glare", type=float, default=1.0, help="compositor glow amount (0 = off)"
    )
    ap.add_argument(
        "--haze",
        type=float,
        default=0.85,
        help="distance haze via the mist pass (0 = off)",
    )
    ap.add_argument("--motion-blur", action="store_true")
    ap.add_argument(
        "--layers", default=None, help="comma list: only build these layers"
    )
    ap.add_argument("--no-dof", action="store_true")
    ap.add_argument(
        "--device", default="auto", choices=("auto", "gpu", "cpu"), help="cycles device"
    )
    return ap.parse_args(argv)


def parse_frames(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    if "," in spec or ":" not in spec:
        return [int(f) for f in spec.split(",") if f.strip()]
    parts = [int(p) if p else None for p in spec.split(":")]
    a = parts[0] or 0
    b = parts[1] if len(parts) > 1 and parts[1] is not None else n - 1
    step = parts[2] if len(parts) > 2 and parts[2] else 1
    return list(range(max(a, 0), min(b, n - 1) + 1, step))


# -- scene ----------------------------------------------------------------------------


def reset_scene() -> bpy.types.Scene:
    # keeps user prefs (shader-compile subprocesses), unlike read_factory_settings
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    scene.name = "LES"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


def new_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def import_obj(
    path: pathlib.Path, coll: bpy.types.Collection, name: str
) -> bpy.types.Object:
    before = set(bpy.data.objects)
    # identity import: the bundle is already z-up metres
    bpy.ops.wm.obj_import(
        filepath=str(path), forward_axis="Y", up_axis="Z", global_scale=1.0
    )
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        for c in o.users_collection:
            c.objects.unlink(o)
        coll.objects.link(o)
    if len(new) > 1:
        with bpy.context.temp_override(
            active_object=new[0], selected_editable_objects=new, object=new[0]
        ):
            bpy.ops.object.join()
    obj = new[0]
    obj.name = name
    obj.data.name = name
    return obj


def build_context(b: bmod.Bundle, look: str, coll) -> dict:
    m = b.manifest
    geo = m.get("geometry") or {}
    out = {}
    if geo.get("buildings", {}).get("obj"):
        bld = import_obj(b.path(geo["buildings"]["obj"]), coll, "buildings")
        bld.data.materials.clear()
        bld.data.materials.append(materials.building_material(look))
        for p in bld.data.polygons:
            p.use_smooth = False
        # crisp but not razor edges: catches a highlight like a physical model
        bev = bld.modifiers.new("bevel", "BEVEL")
        bev.width = 0.12
        bev.segments = 2
        bev.limit_method = "ANGLE"
        bev.angle_limit = math.radians(40)
        bev.harden_normals = False
        bev.use_clamp_overlap = True
        lo, hi = np.array([v.co[:] for v in bld.data.vertices]).min(0), np.array(
            [v.co[:] for v in bld.data.vertices]
        ).max(0)
        ref = np.asarray(geo.get("buildings_bounds", [lo, hi]), float)
        if not np.allclose([lo, hi], ref, atol=1e-2):
            log(
                f"WARNING buildings bounds {lo}..{hi} != manifest {ref.tolist()} (axis conversion?)"
            )
        out["buildings"] = bld
    gmat = materials.ground_material(look, m["domain"])
    if geo.get("ground", {}).get("obj"):
        gnd = import_obj(b.path(geo["ground"]["obj"]), coll, "ground")
        gnd.data.materials.clear()
        gnd.data.materials.append(gmat)
        out["ground"] = gnd
    # infinite floor under the domain ground so frames never show its edge
    lo = np.asarray(m["domain"]["lower"], float)
    hi = np.asarray(m["domain"]["upper"], float)
    c = 0.5 * (lo + hi)
    R = 12.0 * float(np.linalg.norm(hi[:2] - lo[:2]))
    fl = bpy.data.meshes.new("floor")
    fl.from_pydata(
        [
            (c[0] - R, c[1] - R, -0.02),
            (c[0] + R, c[1] - R, -0.02),
            (c[0] + R, c[1] + R, -0.02),
            (c[0] - R, c[1] + R, -0.02),
        ],
        [],
        [(0, 1, 2, 3)],
    )
    fl.materials.append(gmat)
    floor = bpy.data.objects.new("floor", fl)
    coll.objects.link(floor)
    out["floor"] = floor
    return out


# -- camera ---------------------------------------------------------------------------


def build_camera(b: bmod.Bundle, frames_n: int, use_dof: bool) -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = (
        36.0  # full-frame; UE CineCamera filmback must match (36 x 20.25)
    )
    cam_data.clip_start = 0.5
    cam_data.clip_end = 20000.0
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"
    cam_data.dof.use_dof = use_dof
    shots = b.shots or bmod.default_shots(b.manifest)
    prev_q = None
    cut_frames = {s["end"] for s in shots}
    for f in range(frames_n):
        s = bmod.sample_camera(shots, f)
        q = np.array(bmod.camera_quaternion(s["location"], s["target"]))
        if prev_q is not None and float(np.dot(q, prev_q)) < 0:
            q = -q  # keep quaternion keys on one hemisphere
        prev_q = q
        cam.location = s["location"].tolist()
        cam.rotation_quaternion = q.tolist()
        cam_data.lens = s["focal_length_mm"]
        cam_data.dof.focus_distance = float(np.linalg.norm(s["target"] - s["location"]))
        cam_data.dof.aperture_fstop = s["fstop"]
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        cam_data.keyframe_insert("lens", frame=f)
        cam_data.keyframe_insert("dof.focus_distance", frame=f)
        cam_data.keyframe_insert("dof.aperture_fstop", frame=f)
    for ad in (cam.animation_data, cam_data.animation_data):
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                # linear inside shots, hard cut at shot ends (matters for motion blur only)
                kp.interpolation = (
                    "CONSTANT" if int(kp.co[0]) in cut_frames else "LINEAR"
                )
    return cam


def key_visibility(b: bmod.Bundle, objs: dict[str, bpy.types.Object]) -> None:
    shots = b.shots
    if not shots:
        return
    for name, obj in objs.items():
        for s in shots:
            vis = s.get("layers") is None or name in s["layers"]
            obj.hide_render = not vis
            obj.hide_viewport = not vis
            obj.keyframe_insert("hide_render", frame=s["start"])
            obj.keyframe_insert("hide_viewport", frame=s["start"])
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"


# -- persistence ------------------------------------------------------------------------


REGISTER_TEXT = """# Auto-run on load (needs "Auto Run Python Scripts"): re-wires the per-frame
# bundle loaders that feed particles / isosurfaces / volumes / slices.
import sys, bpy
sys.path.insert(0, {here!r})
import build_scene
build_scene.rebuild_updaters(bpy.context.scene)
"""


def rebuild_updaters(scene) -> None:
    """Re-register frame updaters for a saved .blend (layers already exist)."""
    b = bmod.Bundle(scene["les_bundle"])
    opts = json.loads(scene.get("les_opts", "{}"))
    layers.UPDATERS.clear()
    for layer in b.layers:
        obj = bpy.data.objects.get(layer["name"])
        if obj is None:
            continue
        _attach_updater(b, layer, obj, opts)
    layers.VISIBLE[:] = [b.visible_layers]
    layers.install_handler()
    layers.on_frame_change(scene)


def _attach_updater(b, layer, obj, opts) -> None:
    """Recreate the updater closure for an existing object (used on .blend load)."""
    kind = layer["type"]
    if kind == "particles":
        coll = obj.users_collection[0]
        bpy.data.objects.remove(obj)
        layers.build_particles(b, layer, coll, b.look, opts)
    elif kind == "isosurface":
        mesh = obj.data
        state = {"file": None}

        def update(frame, mesh=mesh, state=state):
            path = b.layer_file(layer, frame)
            if state["file"] != path:
                layers.fill_mesh_from_ply(mesh, path)
                state["file"] = path

        layers.UPDATERS[layer["name"]] = update
    elif kind == "volume":
        vol = obj.data

        def update(frame, vol=vol):
            p = str(b.layer_file(layer, frame))
            if vol.filepath != p:
                vol.filepath = p

        layers.UPDATERS[layer["name"]] = update
    elif kind == "slice":
        img = bpy.data.images.get(f"{layer['name']}_tex")

        def update(frame, img=img):
            p = str(b.layer_file(layer, frame))
            if img is not None and img.filepath != p:
                img.filepath = p
                img.reload()

        layers.UPDATERS[layer["name"]] = update


# -- alembic ----------------------------------------------------------------------------


def export_alembic(
    b: bmod.Bundle,
    scene,
    coll,
    objs: dict,
    ctx: dict,
    frames_n: int,
    max_lines: int = 0,
) -> dict:
    out_dir = b.root / "alembic"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    scene["les_update_hidden"] = True
    layers.VISIBLE.clear()  # export every layer on every frame, whatever the shot
    jobs = []
    for layer in b.layers:
        if layer["type"] == "particles" and layer["name"] in objs:
            groom = layers.build_particles_groom(
                b, layer, coll, {"max_lines": max_lines}
            )
            jobs.append((layer["name"], groom, "curves"))
        elif layer["type"] == "isosurface" and layer["name"] in objs:
            obj = objs[layer["name"]]
            layers.mark_animated(obj.data, frames_n)
            layers.add_passthrough_modifier(obj)
            jobs.append((layer["name"], obj, "mesh"))
    if "buildings" in ctx:
        jobs.append(("buildings", ctx["buildings"], "static"))
    for name, obj, kind in jobs:
        path = out_dir / f"{name}.abc"
        t0 = time.perf_counter()
        # unhide for export (visibility keys would otherwise write invisible samples)
        # evaluation_mode RENDER skips render-hidden objects; the groom object is
        # hidden from the preview render, and shot visibility keys may hide others
        saved_anim = obj.animation_data.action if obj.animation_data else None
        if saved_anim is not None:
            obj.animation_data.action = None
        saved_hide = obj.hide_render
        obj.hide_render = False
        bpy.ops.object.select_all(action="DESELECT")
        obj.hide_viewport = False
        obj.hide_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        static = kind == "static"
        bpy.ops.wm.alembic_export(
            filepath=str(path),
            start=0,
            end=0 if static else frames_n - 1,
            selected=True,
            visible_objects_only=False,
            flatten=True,
            uvs=True,
            normals=kind != "curves",
            vcolors=True,
            face_sets=False,
            export_hair=False,
            export_particles=False,
            export_custom_properties=False,
            evaluation_mode="RENDER",
            triangulate=False,
            global_scale=1.0,
            as_background_job=False,
            init_scene_frame_range=False,
        )
        obj.hide_render = saved_hide
        if saved_anim is not None:
            obj.animation_data.action = saved_anim
        dt = time.perf_counter() - t0
        report[name] = {
            "path": str(path),
            "kind": kind,
            "seconds": round(dt, 2),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        log(f"alembic {name}: {report[name]}")
    scene["les_update_hidden"] = False
    layers.VISIBLE[:] = [b.visible_layers]
    for name in list(layers.UPDATERS):
        if name.endswith("_groom"):
            del layers.UPDATERS[name]
            layers.ALWAYS_UPDATE.discard(name)
    report.update(verify_alembic(report, b, frames_n))
    (out_dir / "alembic_report.json").write_text(json.dumps(report, indent=1))
    return report


def verify_alembic(report: dict, b: bmod.Bundle, frames_n: int) -> dict:
    """Re-import each cache into the scene (frame handlers off) and record
    per-sample topology, attributes and a position checksum."""
    scene = bpy.context.scene
    out = {}
    saved_handlers = list(bpy.app.handlers.frame_change_pre)
    bpy.app.handlers.frame_change_pre.clear()
    try:
        for name, info in list(report.items()):
            if not info.get("bytes"):
                continue
            before = set(bpy.data.objects)
            bpy.ops.wm.alembic_import(filepath=info["path"], set_frame_range=False)
            new = [o for o in bpy.data.objects if o not in before]
            checks: dict = {}
            probe = (
                sorted({0, frames_n // 2, frames_n - 1})
                if info["kind"] != "static"
                else [0]
            )
            for f in probe:
                scene.frame_set(f)
                dg = bpy.context.evaluated_depsgraph_get()
                for o in new:
                    e = o.evaluated_get(dg)
                    rec: dict = {"frame": f}
                    if o.type == "MESH":
                        me = e.to_mesh()
                        pos = np.zeros(len(me.vertices) * 3, np.float32)
                        me.vertices.foreach_get("co", pos)
                        rec.update(
                            verts=len(me.vertices),
                            faces=len(me.polygons),
                            colors=[
                                f"{a.name}:{a.domain}:{a.data_type}"
                                for a in me.color_attributes
                            ],
                            attrs=sorted(
                                a.name
                                for a in me.attributes
                                if not a.name.startswith(".")
                            ),
                        )
                        e.to_mesh_clear()
                    elif o.type == "CURVES":
                        d = e.data
                        pos = np.zeros(len(d.points) * 3, np.float32)
                        d.position_data.foreach_get("vector", pos)
                        r = np.zeros(len(d.points), np.float32)
                        if "radius" in d.attributes:
                            d.attributes["radius"].data.foreach_get("value", r)
                        rec.update(
                            curves=len(d.curves),
                            points=len(d.points),
                            attrs=sorted(a.name for a in d.attributes),
                            radius_max=float(r.max()) if len(r) else None,
                            radius_zero_frac=(
                                round(float((r == 0).mean()), 4) if len(r) else None
                            ),
                        )
                    else:
                        continue
                    p3 = pos.reshape(-1, 3)
                    rec["bbox"] = (
                        [p3.min(0).round(2).tolist(), p3.max(0).round(2).tolist()]
                        if len(p3)
                        else None
                    )
                    rec["pos_hash"] = round(float(np.abs(pos).sum()), 2)
                    checks.setdefault(o.name, []).append(rec)
            for o in new:
                bpy.data.objects.remove(o, do_unlink=True)
            out[f"{name}_verify"] = checks
            log(f"alembic verify {name}: {json.dumps(checks)[:700]}")
    finally:
        bpy.app.handlers.frame_change_pre.extend(saved_handlers)
    return out


# -- main -------------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    t_start = time.perf_counter()
    b = bmod.Bundle(args.bundle)
    m = b.manifest
    look = args.look or b.look
    width = args.width or int(m["render"]["width"])
    height = args.height or int(m["render"]["height"])
    n = b.n_frames
    opts = {
        # metres per pixel per metre of distance, times focal (mm): sensor_w / width_px
        "pixel_angle": 36.0 / width,
        "max_strand_px": args.max_strand_px,
        "max_curve_points": 8 if args.engine == "eevee" else 0,
        "radius_scale": args.radius_scale,
        "emission_scale": args.emission_scale,
        "volume_density_scale": args.volume_density_scale,
    }

    scene = reset_scene()
    scene.render.fps = int(round(b.fps))
    scene.render.fps_base = float(round(b.fps)) / b.fps if b.fps else 1.0
    scene.frame_start = 0
    scene.frame_end = n - 1
    scene["les_bundle"] = str(b.root)
    scene["les_opts"] = json.dumps(opts)

    center, span, sun_dir = lookmod.scene_frame(m)
    ctx_coll = new_collection("context")
    flow_coll = new_collection("flow")
    light_coll = new_collection("lights")
    ctx = build_context(b, look, ctx_coll)
    lookmod.setup_world(look, sun_dir)
    lookmod.setup_lights(look, sun_dir, center, span, light_coll)

    only = set(args.layers.split(",")) if args.layers else None
    objs: dict[str, bpy.types.Object] = {}
    for layer in b.layers:
        if only is not None and layer["name"] not in only:
            continue
        builder = layers.BUILDERS.get(layer["type"])
        if builder is None:
            log(f"skipping layer {layer['name']}: unknown type {layer['type']!r}")
            continue
        objs[layer["name"]] = builder(b, layer, flow_coll, look, opts)
        log(
            f"layer {layer['name']} ({layer['type']}): {layer.get('n_files')} files, step {layer.get('frame_step', 1)}"
        )
    layers.VISIBLE[:] = [b.visible_layers]
    layers.install_handler()
    key_visibility(b, objs)
    build_camera(b, n, use_dof=not args.no_dof)

    lookmod.setup_render(
        scene,
        look,
        args.engine,
        args.samples,
        width,
        height,
        args.motion_blur,
        args.device,
    )
    lookmod.setup_mist(scene, look, span)
    lookmod.setup_compositor(scene, look, args.glare, args.haze)
    log(
        f"scene built in {time.perf_counter() - t_start:.1f} s (look={look}, engine={args.engine}, {width}x{height})"
    )

    if args.export_alembic:
        export_alembic(b, scene, flow_coll, objs, ctx, n, args.alembic_max_lines)

    if args.save_blend:
        txt = bpy.data.texts.new("les_register.py")
        txt.write(REGISTER_TEXT.format(here=str(HERE)))
        txt.use_module = True
        path = b.root / "preview" / "scene.blend"
        path.parent.mkdir(parents=True, exist_ok=True)
        scene.frame_set(0)
        bpy.ops.wm.save_as_mainfile(filepath=str(path), compress=True)
        log(f"saved {path}")

    if args.render:
        frames = parse_frames(args.frames, n)
        out_dir = args.out or (b.root / "preview" / "frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        timings = []
        for f in frames:
            t0 = time.perf_counter()
            scene.frame_set(f)
            t1 = time.perf_counter()
            scene.render.filepath = str(out_dir / f"{f:04d}.png")
            bpy.ops.render.render(write_still=True)
            t2 = time.perf_counter()
            timings.append(
                {
                    "frame": f,
                    "update_s": round(t1 - t0, 3),
                    "render_s": round(t2 - t1, 3),
                }
            )
            log(f"frame {f}: update {t1 - t0:.2f} s, render {t2 - t1:.2f} s")
        rs = [t["render_s"] for t in timings]
        summary = {
            "engine": args.engine,
            "look": look,
            "width": width,
            "height": height,
            "samples": args.samples,
            "frames": timings,
            "mean_render_s": float(np.mean(rs)) if rs else None,
            "mean_render_s_excl_first": float(np.mean(rs[1:])) if len(rs) > 1 else None,
        }
        (out_dir.parent / f"timings_{args.engine}.json").write_text(
            json.dumps(summary, indent=1)
        )
        log(
            f"rendered {len(frames)} frames -> {out_dir}; mean {summary['mean_render_s']:.2f} s/frame"
        )
    log(f"done in {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    try:
        main(argv)
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)

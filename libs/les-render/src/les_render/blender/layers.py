"""Bundle layers -> Blender objects, kept in sync with the frame by one handler.

Every per-frame layer registers an *updater* ``f(frame)`` here; a single
persistent ``frame_change_pre`` handler calls them. That handler fires on
``scene.frame_set`` (our render loop), during ``render(animation=True)`` and
inside ``wm.alembic_export`` (verified), so the same code path feeds preview
renders and the UE Alembic caches. Updaters remember the file they loaded
last and skip the reload when ``floor(frame / frame_step)`` hasn't changed.

Layer objects
-------------
particles   Hair ``Curves`` object rebuilt per file from the visible runs
            (segment k->k+1 drawn iff both alphas > 0). Point attributes
            ``radius`` (tapered), ``speed``, ``alpha`` feed renderer + shader.
volume      Volume object whose ``filepath`` is swapped per file.
isosurface  Mesh rebuilt from the PLY (numpy reader); corner colour attribute
            ``Cd`` (sRGB bytes from the PLY) + point attribute ``value``.
slice       Plane with UVs over ``extent``; its image's filepath is swapped.
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable

import bpy
import bundle as bmod
import materials
import numpy as np
from bpy.app.handlers import persistent

UPDATERS: dict[str, Callable[[int], None]] = {}
# frame -> names of layers visible in that frame's shot (None = all). Hidden
# layers are not reloaded; ``hide_render`` itself can't be used because the
# handler runs before the new frame's animation is evaluated.
VISIBLE: list[Callable[[int], set]] = []
ALWAYS_UPDATE: set[str] = set()


@persistent
def on_frame_change(scene, depsgraph=None):  # noqa: ARG001 - handler signature
    frame = scene.frame_current
    visible = VISIBLE[0](frame) if VISIBLE else None
    for name, fn in UPDATERS.items():
        if visible is not None and name not in visible and name not in ALWAYS_UPDATE:
            continue
        fn(frame)


def install_handler() -> None:
    hs = bpy.app.handlers.frame_change_pre
    for h in list(hs):
        if getattr(h, "__name__", "") == "on_frame_change":
            hs.remove(h)
    hs.append(on_frame_change)


def _link(obj: bpy.types.Object, collection: bpy.types.Collection) -> bpy.types.Object:
    collection.objects.link(obj)
    return obj


def mark_animated(id_data: bpy.types.ID, n_frames: int) -> None:
    """Alembic writes data once unless it thinks the data is animated; a keyed
    dummy property on the datablock makes ``check_is_animated`` true (works
    for Curves; meshes additionally need :func:`add_passthrough_modifier`)."""
    id_data["les_anim"] = 0.0
    id_data.keyframe_insert('["les_anim"]', frame=0)
    id_data["les_anim"] = 1.0
    id_data.keyframe_insert('["les_anim"]', frame=max(n_frames - 1, 1))


def add_passthrough_modifier(obj: bpy.types.Object) -> None:
    """An identity Geometry Nodes modifier: any non-subsurf modifier makes the
    Alembic writer treat the object as animated (one sample per frame)."""
    ng = bpy.data.node_groups.get("les_passthrough")
    if ng is None:
        ng = bpy.data.node_groups.new("les_passthrough", "GeometryNodeTree")
        ng.interface.new_socket(
            "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        ng.interface.new_socket(
            "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )
        gin, gout = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
        ng.links.new(gin.outputs[0], gout.inputs[0])
    if "les_anim" not in obj.modifiers:
        obj.modifiers.new("les_anim", "NODES").node_group = ng


# -- particles ------------------------------------------------------------------


def build_particles(
    b: bmod.Bundle, layer: dict[str, Any], coll, look: str, opts: dict
) -> bpy.types.Object:
    """Hair ``Curves`` object holding only the visible runs of each polyline.

    Blender only renders curves as hair on a ``Curves`` object (GN curves on a
    mesh object are silently dropped by EEVEE/Cycles), and ``Curves`` has no
    python API to change topology in place, so each file builds a fresh
    datablock and swaps it in (cheap: one ``add_curves`` + ``foreach_set``).
    A run is a maximal chain of points with alpha > 0, so segment k->k+1 is
    drawn iff both ends are visible (the contract's segment rule); within a
    run alpha interpolates along the segment, and the radius tapers with it.

    The radius is also capped to ``max_px`` pixels at the point's distance
    from the (baked) camera, so strands sweeping past the lens stay thin
    lines instead of screen-filling ribbons; recomputed every frame.
    """
    name = layer["name"]
    mat = materials.particle_material(layer, look, opts)
    obj = _link(bpy.data.objects.new(name, _particle_curves(name, None, mat)), coll)
    radius = (
        float(layer.get("radius", 0.1))
        * float(opts.get("radius_scale", 1.0))
        * materials.LOOKS.get(look, materials.LOOKS["dark"])["particle_radius"]
    )
    shots = b.shots or bmod.default_shots(b.manifest)
    px_angle = float(
        opts.get("pixel_angle", 0.0)
    )  # rad per pixel at 1 mm focal / set by build_scene
    max_px = float(opts.get("max_strand_px", 1.6))
    # streaklines chain releases that are metres apart: smooth them (trails
    # are sampled densely enough already)
    subdiv = int(opts.get("subdiv") or (3 if layer.get("kind") == "streaklines" else 1))
    # EEVEE Next draws only 8 samples per hair curve (see bundle.chunk_runs)
    max_curve_points = int(opts.get("max_curve_points", 0))
    # fade segments stretched beyond ~1-2 grid cells (turbulent dispersion
    # zig-zags); lengths scale with the LES grid spacing
    dx = float(min(b.manifest["domain"].get("spacing", [4.0])))
    fade_len = None if opts.get("no_stretch_fade") else (1.0 * dx, 2.5 * dx)
    state: dict[str, Any] = {"file": None, "pts": None, "alpha": None}

    def update(frame: int) -> None:
        path = b.layer_file(layer, frame)
        if state["file"] != path:
            pts, spd, alpha = bmod.load_particles(path)
            sizes, idx = bmod.visible_runs(alpha)
            P, S, A = (
                pts.reshape(-1, 3)[idx],
                spd.reshape(-1)[idx],
                alpha.reshape(-1)[idx],
            )
            if fade_len is not None and len(P) > 1:
                # hair transparency is not honoured by EEVEE, so segments that
                # fade out completely are cut rather than made transparent
                seg_len = np.linalg.norm(P[1:] - P[:-1], axis=1)
                keep = np.concatenate([seg_len < fade_len[1], [True]])
                A = A * bmod.stretch_fade(P, sizes, *fade_len)
                P, (S, A), sizes = bmod.cut_runs(P, [S, A], sizes, keep)
            P, (S, A), sizes = bmod.smooth_runs(P, [S, A], sizes, subdiv)
            if max_curve_points:
                P, (S, A), sizes = bmod.chunk_runs(P, [S, A], sizes, max_curve_points)
            old = obj.data
            obj.data = _particle_curves(name, (P, S, A, sizes), mat)
            if old.users == 0:
                bpy.data.hair_curves.remove(old)
            # same-run neighbour flags: segment j -> j+1 exists iff both are in one run
            has_next = np.ones(len(P), bool)
            has_next[np.cumsum(sizes) - 1] = False
            state.update(file=path, pts=P, alpha=A, has_next=has_next)
        if state["pts"] is None or len(state["pts"]) == 0:
            return
        r = radius * (0.35 + 0.65 * np.sqrt(state["alpha"]))
        if px_angle > 0:
            cam = bmod.sample_camera(shots, frame)
            dist = bmod.strand_camera_distance(
                state["pts"], state["has_next"], cam["location"]
            )
            # metres per pixel at distance d: d * sensor_w / (focal * width_px)
            r = np.minimum(r, 0.5 * max_px * dist * px_angle / cam["focal_length_mm"])
        att = obj.data.attributes["radius"]
        att.data.foreach_set("value", np.ascontiguousarray(r, dtype=np.float32))
        obj.data.update_tag()

    UPDATERS[name] = update
    return obj


def _particle_curves(name: str, data, mat) -> bpy.types.Curves:
    """Fresh hair datablock from concatenated runs ``(pts, speed, alpha, sizes)``."""
    cv = bpy.data.hair_curves.new(name)
    cv.materials.append(mat)
    if data is None:
        return cv
    pts, spd, alpha, sizes = data
    if len(sizes) == 0:
        return cv
    cv.add_curves(sizes.tolist())
    # POLY, not the default uniform Catmull-Rom: that overshoots into
    # metre-scale loops on unevenly spaced streak points (smoothing is done
    # beforehand with a centripetal spline, see bundle.smooth_runs)
    ct = cv.attributes.get("curve_type") or cv.attributes.new(
        "curve_type", "INT8", "CURVE"
    )
    ct.data.foreach_set("value", np.ones(len(sizes), np.int8))
    cv.position_data.foreach_set(
        "vector", np.ascontiguousarray(pts, dtype=np.float32).reshape(-1)
    )
    for key, val in (
        ("radius", np.zeros(len(pts), np.float32)),
        ("speed", spd),
        ("alpha", alpha),
    ):
        att = cv.attributes.get(key) or cv.attributes.new(key, "FLOAT", "POINT")
        att.data.foreach_set("value", np.ascontiguousarray(val, dtype=np.float32))
    return cv


def build_particles_groom(
    b: bmod.Bundle, layer: dict[str, Any], coll, opts: dict
) -> bpy.types.Object:
    """Constant-topology Curves object for Alembic -> UE Groom (not rendered)."""
    name = f"{layer['name']}_groom"
    L, P = int(layer["n_lines"]), int(layer["points_per_line"])
    max_lines = int(opts.get("max_lines") or 0)
    keep = np.arange(L)
    if 0 < max_lines < L:
        keep = (
            np.linspace(0, L - 1, max_lines).round().astype(np.int64)
        )  # deterministic, stable ids
        L = len(keep)
    cv = bpy.data.hair_curves.new(name)
    cv.add_curves([P] * L)
    obj = _link(bpy.data.objects.new(name, cv), coll)
    obj.hide_render = True
    radius = float(layer.get("radius", 0.1))
    rad = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    spd_att = cv.attributes.new("speed", "FLOAT", "POINT")
    alpha_att = cv.attributes.new("alpha", "FLOAT", "POINT")
    state = {"file": None}

    def update(frame: int) -> None:
        path = b.layer_file(layer, frame)
        if state["file"] == path:
            return
        pts, spd, alpha = bmod.load_particles(path)
        pts, spd, alpha = pts[keep], spd[keep], alpha[keep]
        pts = bmod.collapse_hidden(pts, alpha)
        cv.position_data.foreach_set("vector", pts.reshape(-1))
        rad.data.foreach_set(
            "value",
            (radius * (alpha > 0) * (0.35 + 0.65 * np.sqrt(alpha)))
            .astype(np.float32)
            .reshape(-1),
        )
        spd_att.data.foreach_set("value", spd.astype(np.float32).reshape(-1))
        alpha_att.data.foreach_set("value", alpha.astype(np.float32).reshape(-1))
        cv.update_tag()
        state["file"] = path

    UPDATERS[name] = update
    ALWAYS_UPDATE.add(name)
    mark_animated(cv, b.n_frames)
    return obj


# -- volumes --------------------------------------------------------------------


def build_volume(
    b: bmod.Bundle, layer: dict[str, Any], coll, look: str, opts: dict
) -> bpy.types.Object:
    name = layer["name"]
    vol = bpy.data.volumes.new(name)
    vol.filepath = str(b.layer_file(layer, 0))
    obj = _link(bpy.data.objects.new(name, vol), coll)
    vol.materials.append(materials.volume_material(layer, look, opts))
    try:
        vol.display.density = 0.2
        vol.render.space = "WORLD"
        vol.render.step_size = 0.0
    except AttributeError:
        pass
    state = {"file": str(b.layer_file(layer, 0))}

    def update(frame: int) -> None:
        path = str(b.layer_file(layer, frame))
        if state["file"] == path:
            return
        vol.filepath = path
        state["file"] = path

    UPDATERS[name] = update
    return obj


# -- isosurfaces --------------------------------------------------------------------


def fill_mesh_from_ply(mesh: bpy.types.Mesh, path: pathlib.Path) -> None:
    verts, tris, props = bmod.read_ply(path)
    mesh.clear_geometry()
    if len(verts) == 0 or len(tris) == 0:
        mesh.update()
        return
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.reshape(-1))
    n = len(tris)
    mesh.loops.add(3 * n)
    mesh.loops.foreach_set("vertex_index", tris.reshape(-1))
    mesh.polygons.add(n)
    mesh.polygons.foreach_set("loop_start", np.arange(0, 3 * n, 3, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(n, dtype=bool))
    if all(k in props for k in ("red", "green", "blue")):
        # Corner-domain byte colour named "Cd": the only colour layout Blender's
        # Alembic writer exports (as a face-varying C4f "Cd" param, the
        # Houdini/UE convention); shaders read it the same way.
        rgb = np.stack([props["red"], props["green"], props["blue"]], axis=1).astype(
            np.float32
        )
        rgb = rgb / (255.0 if rgb.max() > 1.0 else 1.0)
        rgba = np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], axis=1)
        col = mesh.color_attributes.new("Cd", "BYTE_COLOR", "CORNER")
        col.data.foreach_set("color_srgb", rgba[tris.reshape(-1)].reshape(-1))
        mesh.color_attributes.active_color = col
        mesh.color_attributes.render_color_index = (
            mesh.color_attributes.active_color_index
        )
    if "value" in props:
        att = mesh.attributes.new("value", "FLOAT", "POINT")
        att.data.foreach_set("value", props["value"].astype(np.float32))
    mesh.update(calc_edges=True)
    mesh.validate(clean_customdata=False)


def build_isosurface(
    b: bmod.Bundle, layer: dict[str, Any], coll, look: str, opts: dict
) -> bpy.types.Object:
    name = layer["name"]
    mesh = bpy.data.meshes.new(name)
    obj = _link(bpy.data.objects.new(name, mesh), coll)
    mat = materials.isosurface_material(layer, look, opts)
    state = {"file": None}

    def update(frame: int) -> None:
        path = b.layer_file(layer, frame)
        if state["file"] == path:
            return
        fill_mesh_from_ply(mesh, path)
        if not mesh.materials:
            mesh.materials.append(mat)
        state["file"] = path

    mesh.materials.append(mat)
    UPDATERS[name] = update
    return obj


# -- slices -----------------------------------------------------------------------


def build_slice(
    b: bmod.Bundle, layer: dict[str, Any], coll, look: str, opts: dict
) -> bpy.types.Object:
    name = layer["name"]
    (u0, v0), (u1, v1) = layer["extent"]
    axis = layer.get("axis", "z")
    pos = float(layer["position"])
    if axis == "z":
        pos += float(opts.get("slice_lift", 0.05))
        corners = [(u0, v0, pos), (u1, v0, pos), (u1, v1, pos), (u0, v1, pos)]
    elif axis == "y":
        corners = [(u0, pos, v0), (u1, pos, v0), (u1, pos, v1), (u0, pos, v1)]
    else:
        corners = [(pos, u0, v0), (pos, u1, v0), (pos, u1, v1), (pos, u0, v1)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    uv = mesh.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", np.array([0, 0, 1, 0, 1, 1, 0, 1], dtype=np.float32))
    obj = _link(bpy.data.objects.new(name, mesh), coll)
    first = b.layer_file(layer, 0)
    img = bpy.data.images.load(str(first), check_existing=False)
    img.name = f"{name}_tex"
    img.alpha_mode = "STRAIGHT"
    mesh.materials.append(materials.slice_material(layer, img, look, opts))
    state = {"file": first}

    def update(frame: int) -> None:
        path = b.layer_file(layer, frame)
        if state["file"] == path:
            return
        img.filepath = str(path)
        img.reload()
        state["file"] = path

    UPDATERS[name] = update
    return obj


BUILDERS = {
    "particles": build_particles,
    "volume": build_volume,
    "isosurface": build_isosurface,
    "slice": build_slice,
}

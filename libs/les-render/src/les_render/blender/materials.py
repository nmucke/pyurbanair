"""Shader node trees for the two looks (``dark`` / ``daylight``).

Colour always comes from the layer's manifest LUT (256 linear-RGB triples),
baked into a 256x1 float image and sampled with ``value -> Map Range(range)
-> (u, 0.5)``, so Blender reproduces the exporter's colours exactly (no
matplotlib inside Blender, no 32-stop ColorRamp limit).

Look tuning knobs are collected in ``LOOKS`` so the preview can be adjusted in
one place.
"""

from __future__ import annotations

from typing import Any

import bpy
import numpy as np

LOOKS: dict[str, dict[str, float]] = {
    "dark": {
        "particle_emission": 2.2,  # x layer emission_strength (x opacity, see particle_material)
        "particle_diffuse": 0.0,
        "particle_opacity": 0.5,  # per-strand coverage; overlaps add up like glow
        "particle_radius": 0.4,  # x manifest radius: thin filaments, the glare supplies the width
        "lut_floor": 0.15,  # skip the near-black LUT start (volumes)
        "particle_lut_floor": 0.3,  # particles start 30% up the LUT: slow canyon flow still reads
        "particle_min_emission": 0.3,  # emission floor (x opacity scale) for faint/slow strands
        "volume_emission": 1.0,
        "volume_density": 1.0,
        "iso_emission": 0.35,
        "slice_emission": 0.7,
        "slice_alpha": 0.8,
    },
    "daylight": {
        "particle_emission": 0.12,
        "particle_diffuse": 1.0,
        "particle_opacity": 0.85,
        "particle_radius": 0.8,
        "lut_floor": 0.0,
        "particle_lut_floor": 0.0,
        "particle_min_emission": 0.0,
        "volume_emission": 0.6,
        "volume_density": 0.5,
        "iso_emission": 0.0,
        "slice_emission": 0.12,
        "slice_alpha": 0.9,
    },
}


VOLUME_SIGMA = 0.02  # extinction 1/m at full density and density_scale 1
VOLUME_EMIT = (
    0.004  # emitted radiance per metre at full density and emission_strength 1
)
NEAR_FADE = (
    1.5,
    10.0,
)  # m: strands closer than this to the camera fade out (no giant ribbons)


def _look(look: str) -> dict[str, float]:
    return LOOKS.get(look, LOOKS["dark"])


def lut_image(layer: dict[str, Any]) -> bpy.types.Image:
    name = f"LUT_{layer['name']}"
    img = bpy.data.images.get(name)
    if img is not None:
        return img
    lut = np.asarray(
        layer.get("lut_linear_rgb") or np.linspace(0, 1, 256)[:, None].repeat(3, 1),
        dtype=np.float32,
    )
    n = len(lut)
    img = bpy.data.images.new(name, width=n, height=1, alpha=False, float_buffer=True)
    try:
        img.colorspace_settings.name = "Linear Rec.709"
    except TypeError:
        img.colorspace_settings.is_data = True
    px = np.concatenate([lut, np.ones((n, 1), np.float32)], axis=1)
    img.pixels.foreach_set(px.reshape(-1))
    img.pack()
    return img


class _Tree:
    """Tiny helper for building node trees without repetition."""

    def __init__(self, mat: bpy.types.Material):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.x = 0

    def node(self, kind: str, **inputs):
        n = self.nt.nodes.new(kind)
        n.location = (self.x, 0)
        self.x += 200
        for k, v in inputs.items():
            if k.startswith("_"):
                setattr(n, k[1:], v)
            else:
                n.inputs[k].default_value = v
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op: str, a, b=None, clamp=False):
        m = self.node("ShaderNodeMath", _operation=op, _use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = float(v)
            else:
                self.link(v, m.inputs[i])
        return m.outputs[0]


def _lut_color(t: _Tree, layer: dict[str, Any], value_socket, floor: float = 0.0):
    """(color socket, normalized [0,1] value socket) for a scalar socket.

    ``floor`` > 0 starts the LUT lookup at that fraction (colour only; the
    returned normalized value is unaffected)."""
    vmin, vmax = (layer.get("range") or [0.0, 1.0])[:2]
    mr = t.node("ShaderNodeMapRange", _clamp=True)
    mr.inputs["From Min"].default_value = float(vmin)
    mr.inputs["From Max"].default_value = (
        float(vmax) if vmax != vmin else float(vmin) + 1.0
    )
    t.link(value_socket, mr.inputs["Value"])
    lookup = mr.outputs["Result"]
    if floor > 0:
        lookup = t.math("MULTIPLY_ADD", lookup, 1.0 - floor)
        t.nt.nodes[-1].inputs[2].default_value = floor
    comb = t.node("ShaderNodeCombineXYZ", Y=0.5)
    t.link(lookup, comb.inputs["X"])
    tex = t.node("ShaderNodeTexImage", _interpolation="Linear", _extension="EXTEND")
    tex.image = lut_image(layer)
    t.link(comb.outputs["Vector"], tex.inputs["Vector"])
    return tex.outputs["Color"], mr.outputs["Result"]


def _lut_ramp(
    t: _Tree, layer: dict[str, Any], value_socket, floor: float = 0.0, stops: int = 32
):
    """LUT as a 32-stop ColorRamp. Used in volume shaders, where EEVEE Next
    does not evaluate image textures (the LUT image comes out grey)."""
    vmin, vmax = (layer.get("range") or [0.0, 1.0])[:2]
    mr = t.node("ShaderNodeMapRange", _clamp=True)
    mr.inputs["From Min"].default_value = float(vmin)
    mr.inputs["From Max"].default_value = (
        float(vmax) if vmax != vmin else float(vmin) + 1.0
    )
    mr.inputs["To Min"].default_value = float(floor)
    t.link(value_socket, mr.inputs["Value"])
    ramp = t.node("ShaderNodeValToRGB")
    lut = np.asarray(
        layer.get("lut_linear_rgb") or np.linspace(0, 1, 256)[:, None].repeat(3, 1),
        float,
    )
    xs = np.linspace(0.0, 1.0, stops)
    cols = np.stack(
        [np.interp(xs, np.linspace(0, 1, len(lut)), lut[:, c]) for c in range(3)],
        axis=1,
    )
    el = ramp.color_ramp.elements
    el[0].position, el[-1].position = 0.0, 1.0
    for x in xs[1:-1]:  # new() inserts in sorted order; never move existing stops
        el.new(float(x))
    for e, c in zip(sorted(el, key=lambda e: e.position), cols):
        e.color = (*map(float, c), 1.0)
    ramp.color_ramp.interpolation = "LINEAR"
    t.link(mr.outputs["Result"], ramp.inputs["Fac"])
    return ramp.outputs["Color"]


def _attr(t: _Tree, name: str):
    a = t.node("ShaderNodeAttribute", _attribute_type="GEOMETRY", _attribute_name=name)
    return a.outputs["Fac"]


def _output(t: _Tree, shader, volume=False):
    out = t.node("ShaderNodeOutputMaterial")
    t.link(shader, out.inputs["Volume" if volume else "Surface"])


# -- flow layers -------------------------------------------------------------------


def particle_material(
    layer: dict[str, Any], look: str, opts: dict
) -> bpy.types.Material:
    L = _look(look)
    mat = bpy.data.materials.new(f"M_{layer['name']}")
    t = _Tree(mat)
    color, tnorm = _lut_color(t, layer, _attr(t, "speed"), L["particle_lut_floor"])
    alpha = _attr(t, "alpha")
    # near-camera fade: strands passing right by the lens
    cam = t.node("ShaderNodeCameraData")
    near = t.node("ShaderNodeMapRange", _clamp=True, _interpolation_type="SMOOTHSTEP")
    near.inputs["From Min"].default_value, near.inputs["From Max"].default_value = (
        NEAR_FADE
    )
    t.link(cam.outputs["View Distance"], near.inputs["Value"])
    # comet profile: sharpen the exporter's alpha taper (trails: bright head,
    # quickly fading tail); dense layers get proportionally fainter strands so
    # 50k trails read as a flow texture instead of a solid mat
    gamma = 2.5 if layer.get("kind") == "trails" else 1.3
    density = min(1.0, (8000.0 / max(float(layer.get("n_lines", 8000)), 1.0)) ** 0.5)
    op = t.math(
        "MULTIPLY",
        t.math("MULTIPLY", t.math("POWER", alpha, gamma), near.outputs["Result"]),
        L["particle_opacity"] * max(density, 0.45),
    )
    # fast particles glow brighter: strength * (0.35 + 0.65 t). EEVEE hair
    # ignores transparency, so the opacity also scales the emission (which
    # alone reads as fading on the dark look); Cycles uses the mix as well.
    strength = (
        float(layer.get("emission_strength", 3.0))
        * L["particle_emission"]
        * float(opts.get("emission_scale", 1.0))
    )
    boost = t.math("MULTIPLY_ADD", tnorm, 0.45)
    t.nt.nodes[-1].inputs[2].default_value = 0.55
    op_emit = op
    if L["particle_min_emission"] > 0:
        # EEVEE strands are hairlines, so faint tails/slow flow fall below
        # visibility: floor the emission (still faded by alpha and near-camera)
        floor = t.math(
            "MULTIPLY",
            t.math("MULTIPLY", t.math("POWER", alpha, 0.5), near.outputs["Result"]),
            L["particle_min_emission"],
        )
        op_emit = t.math("MAXIMUM", op, floor)
    s = t.math("MULTIPLY", t.math("MULTIPLY", boost, strength), op_emit)
    if L["particle_diffuse"] > 0:
        bsdf = t.node("ShaderNodeBsdfPrincipled", Roughness=0.45)
        t.link(color, bsdf.inputs["Base Color"])
        t.link(color, bsdf.inputs["Emission Color"])
        t.link(s, bsdf.inputs["Emission Strength"])
        shader = bsdf.outputs[0]
    else:
        em = t.node("ShaderNodeEmission")
        t.link(color, em.inputs["Color"])
        t.link(s, em.inputs["Strength"])
        shader = em.outputs[0]
    tr = t.node("ShaderNodeBsdfTransparent")
    mix = t.node("ShaderNodeMixShader")
    t.link(op, mix.inputs["Fac"])
    t.link(tr.outputs[0], mix.inputs[1])
    t.link(shader, mix.inputs[2])
    _output(t, mix.outputs[0])
    _eevee_blend(mat, blended=False)
    return mat


def volume_material(layer: dict[str, Any], look: str, opts: dict) -> bpy.types.Material:
    """Glow volume: LUT-coloured emission + grey absorption (dark), or LUT
    scattering + absorption with a little emission (daylight). Density and
    emission are per metre (see VOLUME_SIGMA / VOLUME_EMIT), so the look does
    not depend on the domain size or the camera being inside the volume."""
    L = _look(look)
    mat = bpy.data.materials.new(f"M_{layer['name']}")
    t = _Tree(mat)
    grid = layer.get("grid", layer["name"])
    a = t.node(
        "ShaderNodeAttribute", _attribute_type="GEOMETRY", _attribute_name=grid
    ).outputs["Fac"]
    lo, hi = (layer.get("density_range") or layer.get("range") or [0.0, 1.0])[:2]
    dens = t.node("ShaderNodeMapRange", _clamp=True)
    dens.inputs["From Min"].default_value = float(lo)
    dens.inputs["From Max"].default_value = float(hi) if hi != lo else float(lo) + 1.0
    t.link(a, dens.inputs["Value"])
    dnorm = dens.outputs["Result"]
    color = _lut_ramp(t, layer, a, L["lut_floor"])
    sigma = (
        float(layer.get("density_scale", 1.0))
        * VOLUME_SIGMA
        * L["volume_density"]
        * float(opts.get("volume_density_scale", 1.0))
    )
    em = (
        float(layer.get("emission_strength", 1.0))
        * VOLUME_EMIT
        * L["volume_emission"]
        * float(opts.get("emission_scale", 1.0))
    )
    d = t.math("POWER", dnorm, 2.0)  # soft edges: only strong structures glow
    absorb = t.node("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
    t.link(t.math("MULTIPLY", d, sigma), absorb.inputs["Density"])
    emit = t.node("ShaderNodeEmission")
    t.link(color, emit.inputs["Color"])
    t.link(t.math("MULTIPLY", d, em), emit.inputs["Strength"])
    add = t.node("ShaderNodeAddShader")
    t.link(absorb.outputs[0], add.inputs[0])
    t.link(emit.outputs[0], add.inputs[1])
    out = add.outputs[0]
    if look == "daylight":
        sc = t.node("ShaderNodeVolumeScatter", Anisotropy=0.2)
        t.link(color, sc.inputs["Color"])
        t.link(t.math("MULTIPLY", d, sigma), sc.inputs["Density"])
        add2 = t.node("ShaderNodeAddShader")
        t.link(out, add2.inputs[0])
        t.link(sc.outputs[0], add2.inputs[1])
        out = add2.outputs[0]
    _output(t, out, volume=True)
    return mat


def isosurface_material(
    layer: dict[str, Any], look: str, opts: dict
) -> bpy.types.Material:
    L = _look(look)
    mat = bpy.data.materials.new(f"M_{layer['name']}")
    t = _Tree(mat)
    ca = t.node("ShaderNodeVertexColor", _layer_name="Cd")
    color = ca.outputs["Color"]
    bsdf = t.node("ShaderNodeBsdfPrincipled", Roughness=0.32)
    bsdf.inputs["Coat Weight"].default_value = 0.3 if look == "daylight" else 0.0
    bsdf.inputs["Sheen Weight"].default_value = 0.35 if look == "daylight" else 0.0
    t.link(color, bsdf.inputs["Base Color"])
    if L["iso_emission"] > 0:
        # fresnel rim glow: edges of the vortex tubes read as luminous outlines
        lw = t.node("ShaderNodeLayerWeight", Blend=0.35)
        rim = t.math("POWER", lw.outputs["Facing"], 2.0)
        s = t.math("MULTIPLY_ADD", rim, 3.0)
        t.nt.nodes[-1].inputs[2].default_value = 0.25
        t.link(color, bsdf.inputs["Emission Color"])
        t.link(
            t.math(
                "MULTIPLY",
                s,
                L["iso_emission"] * float(opts.get("emission_scale", 1.0)),
            ),
            bsdf.inputs["Emission Strength"],
        )
    _output(t, bsdf.outputs[0])
    return mat


def slice_material(
    layer: dict[str, Any], image: bpy.types.Image, look: str, opts: dict
) -> bpy.types.Material:
    L = _look(look)
    mat = bpy.data.materials.new(f"M_{layer['name']}")
    t = _Tree(mat)
    uv = t.node("ShaderNodeUVMap", _uv_map="UVMap")
    tex = t.node("ShaderNodeTexImage", _interpolation="Cubic", _extension="CLIP")
    tex.image = image
    t.link(uv.outputs["UV"], tex.inputs["Vector"])
    bsdf = t.node("ShaderNodeBsdfPrincipled", Roughness=0.7)
    t.link(tex.outputs["Color"], bsdf.inputs["Base Color"])
    t.link(tex.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = L["slice_emission"] * float(
        opts.get("emission_scale", 1.0)
    )
    t.link(
        t.math("MULTIPLY", tex.outputs["Alpha"], L["slice_alpha"]), bsdf.inputs["Alpha"]
    )
    _output(t, bsdf.outputs[0])
    _eevee_blend(mat, blended=True)
    return mat


def _eevee_blend(mat: bpy.types.Material, blended: bool) -> None:
    for attr, val in (
        ("surface_render_method", "BLENDED" if blended else "DITHERED"),
        ("blend_method", "BLEND" if blended else "HASHED"),
        ("use_transparency_overlap", False),
    ):
        try:
            setattr(mat, attr, val)
        except (AttributeError, TypeError):
            pass


# -- context -------------------------------------------------------------------------


def building_material(look: str) -> bpy.types.Material:
    mat = bpy.data.materials.new("M_buildings")
    t = _Tree(mat)
    if look == "daylight":
        base, rough = (0.78, 0.77, 0.75), 0.62  # warm white clay
    else:
        base, rough = (
            0.13,
            0.138,
            0.155,
        ), 0.4  # graphite; glossy enough to catch the glow
    bsdf = t.node("ShaderNodeBsdfPrincipled", Roughness=rough)
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Specular IOR Level"].default_value = (
        0.35 if look == "daylight" else 0.5
    )
    # subtle vertical AO-like darkening towards street level for depth
    geo = t.node("ShaderNodeNewGeometry")
    sep = t.node("ShaderNodeSeparateXYZ")
    t.link(geo.outputs["Position"], sep.inputs[0])
    mr = t.node("ShaderNodeMapRange", _clamp=True)
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = 18.0
    mr.inputs["To Min"].default_value = 0.72 if look == "daylight" else 0.6
    mr.inputs["To Max"].default_value = 1.0
    t.link(sep.outputs["Z"], mr.inputs["Value"])
    mix = t.node("ShaderNodeMix", _data_type="RGBA", _blend_type="MULTIPLY")
    mix.inputs["Factor"].default_value = 1.0
    mix.inputs[6].default_value = (*base, 1.0)
    comb = t.node("ShaderNodeCombineColor")
    for k in ("Red", "Green", "Blue"):
        t.link(mr.outputs["Result"], comb.inputs[k])
    t.link(comb.outputs[0], mix.inputs[7])
    t.link(mix.outputs[2], bsdf.inputs["Base Color"])
    _output(t, bsdf.outputs[0])
    return mat


def ground_material(look: str, domain: dict[str, Any]) -> bpy.types.Material:
    """Ground: faint 20 m survey grid + radial falloff so the infinite floor
    fades into the world instead of ending at a hard edge."""
    mat = bpy.data.materials.new("M_ground")
    t = _Tree(mat)
    lo = np.asarray(domain["lower"], float)
    hi = np.asarray(domain["upper"], float)
    c = 0.5 * (lo + hi)
    r = 0.5 * float(np.linalg.norm(hi[:2] - lo[:2]))
    if look == "daylight":
        base, line, far, rough = (
            (0.42, 0.43, 0.44),
            (0.36, 0.37, 0.385),
            (0.55, 0.56, 0.58),
            0.85,
        )
    else:
        base, line, far, rough = (
            (0.010, 0.011, 0.014),
            (0.028, 0.034, 0.048),
            (0.0, 0.0, 0.0),
            0.42,
        )
    geo = t.node("ShaderNodeNewGeometry")
    # grid lines every 20 m
    sep = t.node("ShaderNodeSeparateXYZ")
    t.link(geo.outputs["Position"], sep.inputs[0])

    def gridline(axis):
        w = t.math("PINGPONG", t.math("DIVIDE", sep.outputs[axis], 20.0), 0.5)
        return t.math("LESS_THAN", w, 0.012)

    lines = t.math("MAXIMUM", gridline("X"), gridline("Y"))
    mix = t.node("ShaderNodeMix", _data_type="RGBA")
    mix.inputs[6].default_value = (*base, 1.0)
    mix.inputs[7].default_value = (*line, 1.0)
    t.link(lines, mix.inputs["Factor"])
    # radial fade to the horizon colour
    vec = t.node("ShaderNodeVectorMath", _operation="DISTANCE")
    vec.inputs[1].default_value = (float(c[0]), float(c[1]), 0.0)
    t.link(geo.outputs["Position"], vec.inputs[0])
    fade = t.node("ShaderNodeMapRange", _clamp=True)
    fade.inputs["From Min"].default_value = 1.1 * r
    fade.inputs["From Max"].default_value = 3.5 * r
    t.link(vec.outputs["Value"], fade.inputs["Value"])
    mix2 = t.node("ShaderNodeMix", _data_type="RGBA")
    t.link(fade.outputs["Result"], mix2.inputs["Factor"])
    t.link(mix.outputs[2], mix2.inputs[6])
    mix2.inputs[7].default_value = (*far, 1.0)
    bsdf = t.node("ShaderNodeBsdfPrincipled", Roughness=rough)
    bsdf.inputs["Specular IOR Level"].default_value = 0.6 if look == "dark" else 0.2
    t.link(mix2.outputs[2], bsdf.inputs["Base Color"])
    _output(t, bsdf.outputs[0])
    return mat

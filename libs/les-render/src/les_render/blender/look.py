"""World, lights, colour management, engine settings and compositor per look."""

from __future__ import annotations

import math
from typing import Any

import bpy
import numpy as np


def _set(obj, attr: str, value) -> bool:
    """setattr that tolerates API drift between Blender versions."""
    try:
        setattr(obj, attr, value)
        return True
    except (AttributeError, TypeError, ValueError):
        return False


# -- world + lights -----------------------------------------------------------------


DARK_SKY_AMBIENT = (0.09, 0.12, 0.2)  # dark-look ambient from above (lighting only)
DARK_BOUNCE = (0.16, 0.09, 0.13)  # ...and from below: warm-magenta bounce of the glow


def setup_world(look: str, sun_dir: np.ndarray) -> None:
    world = bpy.data.worlds.new(f"World_{look}")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    if look == "daylight":
        sky = nt.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_disc = False
        sky.sun_elevation = math.asin(float(sun_dir[2]))
        sky.sun_rotation = math.atan2(float(sun_dir[0]), float(sun_dir[1]))
        sky.altitude = 50.0
        sky.air_density = 1.0
        sky.dust_density = 2.0
        sky.ozone_density = 1.0
        # soften: blend the physical sky towards a pale overcast tint
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.inputs["Factor"].default_value = 0.35
        nt.links.new(sky.outputs["Color"], mix.inputs[6])
        mix.inputs[7].default_value = (0.75, 0.8, 0.88, 1.0)
        nt.links.new(mix.outputs[2], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 0.22
    else:
        # vertical gradient: deep blue-black zenith, faintly lighter horizon haze
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Generated"], sep.inputs[0])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        cr = ramp.color_ramp
        cr.elements[0].position = 0.0
        cr.elements[0].color = (0.016, 0.024, 0.045, 1.0)
        cr.elements[1].position = 0.35
        cr.elements[1].color = (0.0012, 0.0016, 0.0035, 1.0)
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.0
        mr.inputs["From Max"].default_value = 1.0
        nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 1.0
        # Lighting-only ambient (camera rays still see the near-black sky):
        # a cool sky dome from above plus a warm "ground bounce" from below,
        # standing in for the glow of the flow layers lighting the streets.
        # Gives facades a dark-graphite read in canyons without lifting the
        # background or competing with the emissive elements.
        amb = nt.nodes.new("ShaderNodeValToRGB")
        ac = amb.color_ramp
        ac.elements[0].position = 0.0  # straight down (lower hemisphere)
        ac.elements[0].color = (*DARK_BOUNCE, 1.0)
        ac.elements[1].position = 1.0  # straight up
        ac.elements[1].color = (*DARK_SKY_AMBIENT, 1.0)
        mr2 = nt.nodes.new("ShaderNodeMapRange")
        mr2.inputs["From Min"].default_value = -1.0
        mr2.inputs["From Max"].default_value = 1.0
        nt.links.new(sep.outputs["Z"], mr2.inputs["Value"])
        nt.links.new(mr2.outputs["Result"], amb.inputs["Fac"])
        bg2 = nt.nodes.new("ShaderNodeBackground")
        nt.links.new(amb.outputs["Color"], bg2.inputs["Color"])
        bg2.inputs["Strength"].default_value = 1.0
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg2.outputs[0], mix.inputs[1])
        nt.links.new(bg.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])


def _look_at_rotation(direction: np.ndarray):
    from mathutils import Vector

    return Vector((-direction).tolist()).to_track_quat("-Z", "Y")


def setup_lights(
    look: str, sun_dir: np.ndarray, center: np.ndarray, span: float, coll
) -> None:
    def add(name, ldata, rot=None, loc=None):
        obj = bpy.data.objects.new(name, ldata)
        coll.objects.link(obj)
        if rot is not None:
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = rot
        if loc is not None:
            obj.location = loc
        return obj

    if look == "daylight":
        sun = bpy.data.lights.new("Sun", "SUN")
        sun.energy = 3.2
        sun.angle = math.radians(4.0)  # soft penumbrae
        sun.color = (1.0, 0.96, 0.9)
        _set(sun, "shadow_soft_size", 0.5)
        add("Sun", sun, _look_at_rotation(sun_dir))
        fill = bpy.data.lights.new("Fill", "SUN")
        fill.energy = 0.35
        fill.angle = math.radians(30.0)
        fill.color = (0.75, 0.83, 1.0)
        _set(fill, "use_shadow", False)
        add("Fill", fill, _look_at_rotation(np.array([-sun_dir[0], -sun_dir[1], 0.8])))
    else:
        # cool moonlight key, very low: gives the buildings form without
        # competing with the emissive flow
        key = bpy.data.lights.new("Moon", "SUN")
        key.energy = 0.8
        key.angle = math.radians(2.0)
        key.color = (0.55, 0.68, 1.0)
        _set(key, "specular_factor", 0.3)
        add("Moon", key, _look_at_rotation(sun_dir))
        rim = bpy.data.lights.new("Rim", "SUN")
        rim.energy = 0.12
        rim.angle = math.radians(10.0)
        rim.color = (0.9, 0.55, 0.35)
        _set(rim, "use_shadow", False)
        _set(rim, "specular_factor", 0.0)  # else a brown hotspot on the glossy floor
        add("Rim", rim, _look_at_rotation(np.array([-sun_dir[0], -sun_dir[1], 0.35])))
        # shadowless skylight from straight above: street canyons and facades
        # keep a hint of form instead of going pure black
        fill = bpy.data.lights.new("SkyFill", "SUN")
        fill.energy = 0.18
        fill.angle = math.radians(60.0)
        fill.color = (0.5, 0.62, 0.9)
        _set(fill, "use_shadow", False)
        _set(fill, "specular_factor", 0.0)
        add("SkyFill", fill, _look_at_rotation(np.array([0.15, 0.1, 1.0])))


# -- render settings -------------------------------------------------------------


def setup_render(
    scene,
    look: str,
    engine: str,
    samples: int | None,
    width: int,
    height: int,
    motion_blur: bool = False,
    device: str = "auto",
) -> None:
    r = scene.render
    r.resolution_x, r.resolution_y = int(width), int(height)
    r.resolution_percentage = 100
    r.film_transparent = False
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.use_motion_blur = bool(motion_blur)
    r.motion_blur_shutter = 0.5
    r.use_compositing = True
    r.use_sequencer = False

    vs = scene.view_settings
    vs.view_transform = "AgX"
    for candidate in (
        ("AgX - Medium High Contrast" if look == "dark" else "AgX - Base Contrast"),
        "None",
    ):
        if _set(vs, "look", candidate):
            break
    vs.exposure = 0.0 if look == "dark" else 0.25
    vs.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    scene.sequencer_colorspace_settings.name = "sRGB"

    if engine == "cycles":
        r.engine = "CYCLES"
        _setup_cycles(scene, samples or 128, device)
    else:
        r.engine = "BLENDER_EEVEE_NEXT"
        _setup_eevee(scene, look, samples or 48)
    # EEVEE Next (4.2) STRIP hair applies the *first* curve's radius profile
    # (by local point index) to every curve, so per-strand radii and the
    # near-camera width cap are lost and random strands become fat tubes.
    # STRAND draws every curve as an anti-aliased hairline instead, which is
    # the thin-filament look we want anyway. Cycles honours per-point radii.
    r.hair_type = "STRAND" if engine != "cycles" else "STRIP"
    r.hair_subdiv = (
        0  # polylines as exported (Catmull-Rom refinement overshoots on uneven spacing)
    )


def _setup_eevee(scene, look: str, samples: int) -> None:
    ee = scene.eevee
    ee.taa_render_samples = int(samples)
    for attr, val in (
        ("use_gtao", True),
        ("gtao_distance", 6.0),
        ("use_shadows", True),
        ("shadow_ray_count", 2),
        ("shadow_step_count", 8),
        ("use_raytracing", True),
        ("ray_tracing_method", "SCREEN"),
        ("use_volumetric_shadows", look == "daylight"),
        ("volumetric_tile_size", "4"),
        ("volumetric_samples", 64),
        ("volumetric_start", 1.0),
        ("volumetric_end", 3000.0),
        ("volumetric_light_clamp", 0.0),
        ("volumetric_ray_depth", 8),
        ("use_bloom", False),
        ("fast_gi_method", "GLOBAL_ILLUMINATION"),
        ("use_fast_gi", True),
        ("horizon_quality", 0.5),
        ("clamp_surface_indirect", 10.0),
    ):
        _set(ee, attr, val)
    rto = getattr(ee, "ray_tracing_options", None)
    if rto is not None:
        _set(rto, "resolution_scale", "2")
        _set(rto, "trace_max_roughness", 0.6)
        _set(rto, "use_denoise", True)


def _cuda_kernel_available() -> bool:
    """Distro Blender builds may ship no CUDA kernels; compiling one on the fly
    needs a working nvcc + host compiler. Only pick CUDA when kernels are
    bundled, or when the user configured the compile via
    CYCLES_CUDA_EXTRA_CFLAGS (e.g. ``-ccbin g++-13``; blender_runner sets it
    from $LES_CUDA_CCBIN). The compiled kernel is cached under a hash of the
    source *and* those flags, so later runs must pass the same flags."""
    import glob
    import os

    cands = glob.glob(
        os.path.join(
            bpy.utils.system_resource("SCRIPTS"),
            "addons*",
            "cycles",
            "lib",
            "kernel_sm_*",
        )
    )
    return bool(cands) or bool(os.environ.get("CYCLES_CUDA_EXTRA_CFLAGS"))


def _setup_cycles(scene, samples: int, device: str = "auto") -> None:
    cy = scene.cycles
    prefs = bpy.context.preferences.addons["cycles"].preferences
    device_type = None
    kinds = [] if device == "cpu" else ["OPTIX", "CUDA"]
    for kind in kinds:
        if kind == "CUDA" and device == "auto" and not _cuda_kernel_available():
            print(
                "[les] cycles: no CUDA kernel available (set LES_CUDA_CCBIN to compile one); using CPU"
            )
            continue
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == kind
                device_type = kind
                break
        except TypeError:
            continue
    if device_type is None:
        prefs.compute_device_type = "NONE"
    cy.device = "GPU" if device_type else "CPU"
    print(f"[les] cycles device: {device_type or 'CPU'}")
    cy.samples = int(samples)
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02
    cy.use_denoising = True
    _set(cy, "denoiser", "OPTIX" if device_type == "OPTIX" else "OPENIMAGEDENOISE")
    cy.max_bounces = 6
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 3
    cy.transparent_max_bounces = 32
    cy.volume_bounces = 0
    cy.volume_step_rate = 1.0
    cy.volume_preview_step_rate = 1.0
    cy.sample_clamp_indirect = 8.0
    _set(cy, "use_fast_gi", False)
    scene.cycles_curves.shape = "RIBBONS"
    scene.cycles_curves.subdivisions = 1
    scene.render.use_persistent_data = True


# -- compositor ------------------------------------------------------------------


HAZE_COLOR = {"dark": (0.012, 0.018, 0.034), "daylight": (0.62, 0.68, 0.76)}


def setup_mist(scene, look: str, span: float) -> None:
    """Mist pass (distance fog factor) used by the compositor haze."""
    scene.view_layers[0].use_pass_mist = True
    ms = scene.world.mist_settings
    ms.start = 0.6 * span
    ms.depth = 4.0 * span
    ms.falloff = "QUADRATIC"


def setup_compositor(scene, look: str, glare: float = 1.0, haze: float = 0.85) -> None:
    if look == "daylight":
        haze *= 0.55  # the pale sky already reads as distance; keep far geometry crisp
    scene.use_nodes = True
    nt = scene.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    cur = rl.outputs["Image"]
    if haze > 0 and "Mist" in rl.outputs:
        # aerial perspective: distant geometry sinks into the horizon colour
        mix = nt.nodes.new("CompositorNodeMixRGB")
        mix.blend_type = "MIX"
        fac = nt.nodes.new("CompositorNodeMath")
        fac.operation = "MULTIPLY"
        fac.inputs[1].default_value = float(haze)
        nt.links.new(rl.outputs["Mist"], fac.inputs[0])
        nt.links.new(fac.outputs[0], mix.inputs["Fac"])
        nt.links.new(cur, mix.inputs[1])
        mix.inputs[2].default_value = (*HAZE_COLOR[look], 1.0)
        cur = mix.outputs["Image"]
    if glare > 0:
        g = nt.nodes.new("CompositorNodeGlare")
        g.glare_type = "FOG_GLOW"
        g.quality = "HIGH"
        g.size = 8 if look == "dark" else 7
        g.threshold = 0.6 if look == "dark" else 1.2
        g.mix = -0.55 if look == "dark" else -0.85
        # mix: -1 = image only, +1 = glare only; scale the glare share by `glare`
        g.mix = float(np.clip(2.0 * (0.5 * (g.mix + 1.0) * glare) - 1.0, -1.0, 1.0))
        nt.links.new(cur, g.inputs["Image"])
        cur = g.outputs["Image"]
        if look == "dark":
            # a second, tight streak-free glow for the hot cores
            g2 = nt.nodes.new("CompositorNodeGlare")
            g2.glare_type = "FOG_GLOW"
            g2.quality = "HIGH"
            g2.size = 6
            g2.threshold = 1.5
            g2.mix = -0.75
            nt.links.new(cur, g2.inputs["Image"])
            cur = g2.outputs["Image"]
    # vignette: soft elliptical mask multiplied in
    mask = nt.nodes.new("CompositorNodeEllipseMask")
    mask.width, mask.height = 1.05, 1.0
    blur = nt.nodes.new("CompositorNodeBlur")
    blur.filter_type = "GAUSS"
    blur.use_relative = True
    blur.factor_x = blur.factor_y = 30.0
    blur.size_x = blur.size_y = 300
    nt.links.new(mask.outputs["Mask"], blur.inputs["Image"])
    vig = nt.nodes.new("CompositorNodeMixRGB")
    vig.blend_type = "MULTIPLY"
    vig.inputs["Fac"].default_value = 1.0
    ramp = nt.nodes.new("CompositorNodeMapRange")
    ramp.inputs["To Min"].default_value = 0.72 if look == "dark" else 0.86
    ramp.inputs["To Max"].default_value = 1.0
    nt.links.new(blur.outputs["Image"], ramp.inputs["Value"])
    nt.links.new(cur, vig.inputs[1])
    nt.links.new(ramp.outputs["Value"], vig.inputs[2])
    nt.links.new(vig.outputs["Image"], comp.inputs["Image"])


def scene_frame(manifest: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    """(centre of the building cluster, horizontal span, sun direction)."""
    lo = np.asarray(manifest["domain"]["lower"], float)
    hi = np.asarray(manifest["domain"]["upper"], float)
    center = 0.5 * (lo + hi)
    center[2] = 0.0
    span = float(np.linalg.norm(hi[:2] - lo[:2]))
    # sun from the south-west-ish, 38 deg up: long raking shadows along streets
    az, el = math.radians(215.0), math.radians(38.0)
    sun_dir = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
    )
    return center, span, sun_dir  # sun_dir points *towards* the sun

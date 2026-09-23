"""Case folder -> render bundle: the pipeline driver.

``build_bundle(cfg, out_dir)`` runs the stages in order; each stage is
skippable via ``cfg["stages"]`` so an expensive export can be reused while
iterating on the look:

1. ``export``  -- geometry, per-layer assets (particles / volumes / isosurfaces
   / slices), camera shots and ``manifest.json``.
2. ``hud``     -- transparent HUD overlays (clock, inflow compass, colorbars).
3. ``unreal``  -- copy the Unreal Editor scripts + LUT textures into the bundle.
4. ``alembic`` -- Blender writes ``alembic/*.abc`` for UE Geometry Cache/Groom.
5. ``blender`` -- headless Blender preview render into ``preview/frames``.
6. ``video``   -- ffmpeg: preview frames + HUD -> ``preview/<case>.mp4``.

``cfg`` is a plain dict (the Hydra config resolved to a container); the keys
are documented in ``conf/render_les.yaml``.
"""

from __future__ import annotations

import json
import logging
import pathlib
import shutil
import time
from typing import Any, Callable, Optional

from les_render.case import Case, discover_case
from les_render.fields import FieldSeries
from les_render.geometry import export_geometry
from les_render.timeline import Timeline, make_timeline

log = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def _layer_exporter(layer_type: str) -> Callable[..., dict]:
    # Imported lazily so a missing optional dependency only breaks its layer.
    if layer_type == "particles":
        from les_render.particles import export_particles

        return export_particles
    if layer_type == "volume":
        from les_render.volumes import export_volumes

        return export_volumes
    if layer_type == "isosurface":
        from les_render.isosurfaces import export_isosurfaces

        return export_isosurfaces
    if layer_type == "slice":
        from les_render.slices import export_slices

        return export_slices
    raise KeyError(f"unknown layer type {layer_type!r}")


def resolve_case(cfg: dict[str, Any]) -> Case:
    return discover_case(
        cfg["input"],
        state=cfg.get("state"),
        geometry=cfg.get("geometry"),
        params=cfg.get("params"),
    )


def build_timeline(fields: FieldSeries, time_cfg: dict[str, Any]) -> Timeline:
    return make_timeline(
        fields.times,
        fps=time_cfg.get("fps", 30),
        playback_speed=time_cfg.get("playback_speed", 20),
        t_start=time_cfg.get("t_start"),
        t_end=time_cfg.get("t_end"),
        duration=time_cfg.get("duration"),
    )


def enabled_layers(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Preset layers that are switched on, with ``name`` filled in from the key."""
    layers = cfg["render_preset"].get("layers") or {}
    out = {}
    for name, spec in layers.items():
        if spec is None or not spec.get("enabled", True):
            continue
        spec = dict(spec)
        spec.setdefault("name", name)
        spec.setdefault("workers", cfg.get("workers", 4))
        out[name] = spec
    return out


def export_stage(
    case: Case, cfg: dict[str, Any], out_dir: pathlib.Path
) -> dict[str, Any]:
    """Write geometry, layers and shots; return (and save) the manifest."""
    from les_render.cameras import make_shots
    from les_render.hud import inflow_block

    ds = case.open_state()
    fields = FieldSeries(ds)
    timeline = build_timeline(fields, cfg.get("time", {}))
    grid = fields.grid
    preset = cfg["render_preset"]
    log.info(
        "case %s: grid %s, %d video frames (%.1f s of video, sim %.1f-%.1f s)",
        case.name,
        grid.shape,
        timeline.n_frames,
        timeline.n_frames / timeline.fps,
        timeline.t_start,
        timeline.t_end,
    )

    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "case": {
            "name": case.name,
            "state": str(case.state_path),
            "geometry": str(case.geometry_path) if case.geometry_path else None,
            "params": str(case.params_path) if case.params_path else None,
        },
        "frame": {"units": "m", "handedness": "right", "up": "z"},
        "domain": {
            "lower": grid.lower.tolist(),
            "upper": grid.upper.tolist(),
            "spacing": grid.spacing.tolist(),
            "shape": list(grid.shape),
        },
        "timeline": timeline.to_dict(),
        "render": {
            "width": int(cfg["render"]["width"]),
            "height": int(cfg["render"]["height"]),
            "look": cfg["render"].get("look") or preset.get("look", "dark"),
            "preset": preset.get("name", "custom"),
        },
    }

    t0 = time.perf_counter()
    manifest["geometry"] = export_geometry(
        case.buildings(),
        grid.lower,
        grid.upper,
        out_dir,
        ground_margin=float(preset.get("ground_margin", 0.0)),
    )
    params = case.open_params()
    manifest["inflow"] = inflow_block(params) if params is not None else None
    log.info("geometry + inflow: %.1f s", time.perf_counter() - t0)

    manifest["layers"] = []
    for name, spec in enabled_layers(cfg).items():
        t0 = time.perf_counter()
        exporter = _layer_exporter(spec["type"])
        spec = {**spec, "footprints": manifest["geometry"]["footprints"]}
        layer = exporter(fields, timeline, spec, out_dir)
        manifest["layers"].append(layer)
        log.info("layer %s (%s): %.1f s", name, spec["type"], time.perf_counter() - t0)

    manifest["shots"] = make_shots(
        manifest["geometry"], manifest["domain"], timeline, preset.get("cameras") or {}
    )
    manifest["hud"] = None
    write_manifest(manifest, out_dir)
    ds.close()
    return manifest


def write_manifest(manifest: dict[str, Any], out_dir: pathlib.Path) -> pathlib.Path:
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path


def read_manifest(out_dir: pathlib.Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((out_dir / "manifest.json").read_text())
    return manifest


def build_bundle(cfg: dict[str, Any], out_dir: pathlib.Path) -> dict[str, Any]:
    """Run the enabled stages; returns the final manifest."""
    stages = cfg.get("stages", {})
    out_dir.mkdir(parents=True, exist_ok=True)
    case = resolve_case(cfg)
    log.info("state: %s", case.state_path)
    log.info("geometry: %s", case.geometry_path or "(from blanking)")
    log.info("params: %s", case.params_path or "(none: no inflow compass)")

    if stages.get("export", True):
        if cfg.get("overwrite", False):
            for sub in (
                "geometry",
                "particles",
                "volumes",
                "isosurfaces",
                "slices",
                "hud",
            ):
                shutil.rmtree(out_dir / sub, ignore_errors=True)
        manifest = export_stage(case, cfg, out_dir)
    else:
        manifest = read_manifest(out_dir)

    if stages.get("hud", True):
        from les_render.hud import render_hud

        t0 = time.perf_counter()
        hud_spec = {
            **(cfg["render_preset"].get("hud") or {}),
            "workers": cfg.get("workers", 4),
        }
        manifest["hud"] = render_hud(manifest, hud_spec, out_dir)
        write_manifest(manifest, out_dir)
        log.info("hud: %.1f s", time.perf_counter() - t0)

    if stages.get("unreal", True):
        from les_render.unreal.prepare import prepare_unreal

        prepare_unreal(out_dir)
        log.info("unreal scripts + LUTs -> %s", out_dir / "unreal")

    blender_cfg = cfg.get("blender", {})
    if stages.get("alembic", False):
        from les_render.blender_runner import run_blender

        run_blender(
            out_dir, export_alembic=True, render=False, **_blender_opts(blender_cfg)
        )

    frames_dir: Optional[pathlib.Path] = None
    if stages.get("blender", True):
        from les_render.blender_runner import run_blender

        t0 = time.perf_counter()
        frames_dir = run_blender(out_dir, render=True, **_blender_opts(blender_cfg))
        log.info("blender render: %.1f s", time.perf_counter() - t0)

    if stages.get("video", True):
        from les_render.compose import compose_video

        frames_dir = frames_dir or out_dir / "preview" / "frames"
        mp4 = out_dir / "preview" / f"{manifest['case']['name']}.mp4"
        hud = manifest.get("hud")
        compose_video(
            str(frames_dir / "%04d.png"),
            mp4,
            fps=manifest["timeline"]["fps"],
            hud_pattern=str(out_dir / "hud" / "hud.%04d.png") if hud else None,
        )
        log.info("video: %s", mp4)
    return manifest


def _blender_opts(blender_cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in blender_cfg.items() if v is not None}

"""Stage a render bundle for Unreal Engine (runs in the ``viz`` env, no Unreal).

``prepare_unreal(bundle_dir)`` fills ``<bundle>/unreal/`` with everything the
Unreal side needs:

* ``build_scene.py`` / ``render.py`` -- the Unreal Editor Python scripts,
  with the absolute bundle path baked in (``DEFAULT_BUNDLE``) so they run from
  *Tools > Execute Python Script* without arguments;
* ``luts/<layer>_lut.png`` -- each layer's 256-entry colour LUT as a 256x1
  RGBA8 sRGB texture (Unreal decodes sRGB -> linear, reproducing the
  manifest's ``lut_linear_rgb``);
* ``meshes/<slice>_plane.glb`` -- one textured quad per slice layer in the
  simulation frame (glTF y-up, like ``geometry/*.glb``), so slices go through
  the same importer and axis conversion as the buildings;
* ``camera_bake.json`` -- per-video-frame camera samples from
  ``les_render.cameras.sample_camera`` (the Blender preview's reference), so
  the Unreal cameras match the preview exactly;
* ``README.md`` + ``ue_commands.sh`` / ``ue_commands.bat`` -- exact commands.

The coordinate helpers (``sim_to_ue``, ``look_at_rotator``, ...) are
re-exported from ``build_scene`` (single source of truth; that module imports
cleanly without ``unreal``).
"""

from __future__ import annotations

import json
import logging
import pathlib
import struct
import zlib
from typing import Any, Callable, Optional, Sequence

from . import build_scene as _bs
from .build_scene import (  # noqa: F401  (re-exported pure helpers)
    bake_camera,
    file_index,
    fps_to_fraction,
    geometry_fix,
    layer_visibility_keys,
    look_at_rotator,
    parse_manifest,
    rotator_forward,
    sample_camera,
    sim_dir_to_ue,
    sim_to_ue,
    slice_quad,
    volume_frame_keys,
    volume_transform,
)

log = logging.getLogger(__name__)
HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = ("build_scene.py", "render.py")
README_TEMPLATE = HERE / "README.md"

# glTF is y-up: z-up (x, y, z) -> y-up (x, z, -y); same as les_render.geometry.
_ZUP_TO_YUP = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))


# -- colour / PNG ---------------------------------------------------------------


def linear_to_srgb(c: float) -> float:
    c = min(max(float(c), 0.0), 1.0)
    return float(12.92 * c if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055)


def png_bytes(width: int, height: int, rgba_rows: Sequence[bytes]) -> bytes:
    """Minimal RGBA8 PNG encoder (no Pillow dependency)."""
    if len(rgba_rows) != height or any(len(r) != 4 * width for r in rgba_rows):
        raise ValueError("rows must be height x (4 * width) bytes")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(r) for r in rgba_rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    # sRGB chunk (rendering intent 0) so tools know the encoding.
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"sRGB", b"\x00")
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def lut_png(lut_linear_rgb: Sequence[Sequence[float]]) -> bytes:
    """256x1 (len x 1) RGBA8 sRGB-encoded PNG of a linear-RGB LUT."""
    row = bytearray()
    for r, g, b in lut_linear_rgb:
        row += bytes(int(round(255.0 * linear_to_srgb(v))) for v in (r, g, b)) + b"\xff"
    return png_bytes(len(lut_linear_rgb), 1, [bytes(row)])


def write_luts(manifest: dict[str, Any], out_dir: pathlib.Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for layer in manifest.get("layers") or []:
        lut = layer.get("lut_linear_rgb")
        if not lut:
            continue
        path = out_dir / f"{layer['name']}_lut.png"
        path.write_bytes(lut_png(lut))
        written[layer["name"]] = str(path)
    return written


def vdb_file_version(path: pathlib.Path | str) -> Optional[int]:
    """OpenVDB file-format version from a .vdb header (int64 magic
    0x56444220, then uint32 version), or None if it isn't a VDB."""
    with open(path, "rb") as fh:
        head = fh.read(12)
    if len(head) < 12 or struct.unpack("<q", head[:8])[0] != 0x56444220:
        return None
    return int(struct.unpack("<I", head[8:12])[0])


def check_volumes(manifest: dict[str, Any], bundle: pathlib.Path) -> list[str]:
    """Warnings for volume layers Unreal's SVT importer is likely to reject."""
    warnings: list[str] = []
    for layer in manifest.get("layers") or []:
        if layer.get("type") != "volume":
            continue
        first = bundle / layer["pattern"].format(frame=0)
        version = vdb_file_version(first) if first.exists() else None
        warnings += _bs.volume_warnings(layer, version)
    return warnings


# -- slice planes -----------------------------------------------------------------


def write_slice_planes(
    manifest: dict[str, Any], out_dir: pathlib.Path
) -> dict[str, str]:
    """One two-triangle glTF quad per slice layer, UVs mapping the slice PNG
    (row 0 = max-v edge) onto the plane. trimesh stores UVs with a bottom-left
    origin and flips V when writing glTF, so the glTF (top-left) UVs from
    ``slice_quad`` are converted back first."""
    import numpy as np
    import trimesh

    slices = [
        layer for layer in manifest.get("layers") or [] if layer.get("type") == "slice"
    ]
    if not slices:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    rot = np.array(_ZUP_TO_YUP)
    written = {}
    for layer in slices:
        corners, uvs_gltf = slice_quad(layer)
        verts = np.asarray(corners, dtype=np.float64) @ rot.T
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(verts, faces, process=False)
        uv = np.array([[u, 1.0 - v] for u, v in uvs_gltf], dtype=np.float64)
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
        path = out_dir / f"{layer['name']}_plane.glb"
        mesh.export(path)
        written[layer["name"]] = str(path)
    return written


# -- cameras ------------------------------------------------------------------------


def camera_samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-video-frame camera samples (sim frame). Location/target/focal come
    from ``les_render.cameras.sample_camera`` when importable (the Blender
    preview's reference), f-stop and the shot name from the pure port."""
    shots = manifest["shots"]
    ref_sample: Optional[Callable[..., Any]]
    try:
        from les_render.cameras import sample_camera as ref_sample
    except Exception:  # noqa: BLE001
        ref_sample = None
    out = []
    for f in range(int(manifest["timeline"]["n_frames"])):
        s = sample_camera(shots, f)
        if ref_sample is not None:
            loc, tgt, focal = ref_sample(shots, f)
            s = {
                **s,
                "location": [float(v) for v in loc],
                "target": [float(v) for v in tgt],
                "focal_length_mm": float(focal),
            }
        out.append(
            {
                "frame": f,
                **{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()},
            }
        )
    return out


def write_camera_bake(manifest: dict[str, Any], path: pathlib.Path) -> pathlib.Path:
    frames = camera_samples(manifest)
    path.write_text(
        json.dumps({"frame": "sim (m, right-handed, z-up)", "frames": frames}, indent=0)
    )
    return path


# -- scripts / docs -------------------------------------------------------------------


def _bake_default_bundle(src: str, bundle: pathlib.Path) -> str:
    line = "DEFAULT_BUNDLE = None"
    if line not in src:
        raise ValueError("script lacks the DEFAULT_BUNDLE placeholder")
    return src.replace(line, f"DEFAULT_BUNDLE = {str(bundle)!r}", 1)


def copy_scripts(bundle: pathlib.Path, out_dir: pathlib.Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in SCRIPTS:
        src = (HERE / name).read_text()
        dst = out_dir / name
        dst.write_text(_bake_default_bundle(src, bundle))
        written[name] = str(dst)
    return written


def _layer_table(manifest: dict[str, Any]) -> str:
    rows = ["| layer | type | Unreal asset / actor | source |", "|---|---|---|---|"]
    kinds = {
        "volume": "Sparse Volume Texture + Heterogeneous Volume (Frame track)",
        "particles": "Groom + Groom Cache (fallback: Geometry Cache from `<layer>_mesh.abc`)",
        "isosurface": "Geometry Cache (Geometry Cache track)",
        "slice": "Img Media Source + Media Texture on a plane (Media track)",
    }
    for layer in manifest.get("layers") or []:
        t = layer.get("type", "?")
        src = {
            "volume": layer.get("pattern", "").replace("{frame:04d}", "0000"),
            "particles": f"alembic/{layer['name']}.abc",
            "isosurface": f"alembic/{layer['name']}.abc",
            "slice": layer.get("pattern", "").rsplit("/", 1)[0] + "/",
        }.get(t, "")
        rows.append(
            f"| `{layer['name']}` | {t} | {kinds.get(t, '(not handled)')} | `{src}` |"
        )
    return "\n".join(rows)


def _shot_table(manifest: dict[str, Any]) -> str:
    rows = ["| shot | frames | visible layers |", "|---|---|---|"]
    for s in manifest.get("shots") or []:
        layers = ", ".join(s["layers"]) if s.get("layers") is not None else "all"
        rows.append(f"| `{s['name']}` | {s['start']}-{s['end']} | {layers} |")
    return "\n".join(rows)


def render_readme(manifest: dict[str, Any], bundle: pathlib.Path) -> str:
    plan = _bs.build_plan(manifest, bundle)
    tl, rd = manifest["timeline"], manifest["render"]
    num, den = plan["fps"]
    hud = manifest.get("hud")
    values = {
        "CASE": plan["case"],
        "BUNDLE": str(bundle),
        "MAP": plan["map"],
        "SEQUENCE": plan["sequence"],
        "MRQ_CONFIG": plan["mrq_config"],
        "OUTPUT_DIR": plan["output_dir"],
        "WIDTH": str(rd.get("width", 1920)),
        "HEIGHT": str(rd.get("height", 1080)),
        "FPS": f"{tl['fps']:g}" + ("" if den == 1 else f" ({num}/{den})"),
        "FPS_FFMPEG": f"{num}" if den == 1 else f"{num}/{den}",
        "N_FRAMES": str(tl["n_frames"]),
        "LOOK": plan["look"],
        "LAYER_TABLE": _layer_table(manifest),
        "SHOT_TABLE": _shot_table(manifest),
        "HUD_PATTERN": "hud/hud.%04d.png" if hud else "(no HUD in this bundle)",
        "SEQ_NAME": plan["sequence"].rsplit("/", 1)[1],
    }
    text = README_TEMPLATE.read_text()
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    return text


def render_commands(manifest: dict[str, Any], bundle: pathlib.Path) -> tuple[str, str]:
    """(bash, bat) scripts: build the scene, render, composite the HUD."""
    plan = _bs.build_plan(manifest, bundle)
    num, den = plan["fps"]
    fps_s = f"{num}" if den == 1 else f"{num}/{den}"
    seq = plan["sequence"].rsplit("/", 1)[1]
    hud = bool(manifest.get("hud"))
    overlay = (
        f'-framerate {fps_s} -i "$BUNDLE/hud/hud.%04d.png" -filter_complex "[0:v][1:v]overlay=0:0:format=auto,format=yuv420p" '
        if hud
        else "-vf format=yuv420p "
    )
    sh = f"""#!/usr/bin/env bash
# Unreal Engine commands for bundle {plan['case']} (generated by les_render.unreal.prepare).
# Set UE_EDITOR_CMD (path to UnrealEditor-Cmd) and UE_PROJECT (your .uproject) first.
set -euo pipefail
BUNDLE={json.dumps(str(bundle))}
: "${{UE_EDITOR_CMD:=UnrealEditor-Cmd}}"
: "${{UE_PROJECT:?set UE_PROJECT=/path/to/Project.uproject}}"

step="${{1:-all}}"

if [[ "$step" == all || "$step" == build ]]; then
  # 1. build the level, sequence and MRQ config (full editor, headless; quits when done)
  "$UE_EDITOR_CMD" "$UE_PROJECT" -ExecutePythonScript="$BUNDLE/unreal/build_scene.py --bundle $BUNDLE --quit" \\
      -unattended -nosplash -log
fi

if [[ "$step" == all || "$step" == render ]]; then
  # 2. render with Movie Render Queue (prints the command, then the output directory)
  python3 "$BUNDLE/unreal/render.py" --bundle "$BUNDLE" --project "$UE_PROJECT" --editor "$UE_EDITOR_CMD"
fi

if [[ "$step" == all || "$step" == video ]]; then
  # 3. composite the HUD and encode
  ffmpeg -y -framerate {fps_s} -i "$BUNDLE/unreal/render/{seq}.%04d.png" {overlay}\\
      -c:v libx264 -crf 16 -preset slow -movflags +faststart "$BUNDLE/unreal/{plan['case']}_ue.mp4"
fi
"""
    overlay_bat = (
        f'-framerate {fps_s} -i "%BUNDLE%\\hud\\hud.%%04d.png" -filter_complex "[0:v][1:v]overlay=0:0:format=auto,format=yuv420p" '
        if hud
        else "-vf format=yuv420p "
    )
    bat = f"""@echo off
rem Unreal Engine commands for bundle {plan['case']} (generated by les_render.unreal.prepare).
rem Edit UE_EDITOR_CMD / UE_PROJECT, or set them before calling. Usage: ue_commands.bat [all^|build^|render^|video]
setlocal
set "BUNDLE={bundle}"
if "%UE_EDITOR_CMD%"=="" set "UE_EDITOR_CMD=C:\\Program Files\\Epic Games\\UE_5.5\\Engine\\Binaries\\Win64\\UnrealEditor-Cmd.exe"
if "%UE_PROJECT%"=="" (echo set UE_PROJECT=C:\\path\\to\\Project.uproject & exit /b 1)
set "STEP=%~1"
if "%STEP%"=="" set "STEP=all"

if /i "%STEP%"=="all" goto build
if /i "%STEP%"=="build" goto build
if /i "%STEP%"=="render" goto render
if /i "%STEP%"=="video" goto video
echo unknown step %STEP% & exit /b 1

:build
"%UE_EDITOR_CMD%" "%UE_PROJECT%" -ExecutePythonScript="%BUNDLE%\\unreal\\build_scene.py --bundle %BUNDLE% --quit" -unattended -nosplash -log
if /i not "%STEP%"=="all" goto :eof

:render
python "%BUNDLE%\\unreal\\render.py" --bundle "%BUNDLE%" --project "%UE_PROJECT%" --editor "%UE_EDITOR_CMD%"
if /i not "%STEP%"=="all" goto :eof

:video
ffmpeg -y -framerate {fps_s} -i "%BUNDLE%\\unreal\\render\\{seq}.%%04d.png" {overlay_bat}-c:v libx264 -crf 16 -preset slow -movflags +faststart "%BUNDLE%\\unreal\\{plan['case']}_ue.mp4"
endlocal
"""
    return sh, bat.replace("\n", "\r\n")


# -- entry point ---------------------------------------------------------------------


def prepare_unreal(
    bundle_dir: pathlib.Path | str, camera_bake: bool = True
) -> dict[str, Any]:
    """Populate ``<bundle>/unreal/``; returns the written paths."""
    bundle = pathlib.Path(bundle_dir).expanduser().resolve()
    manifest = parse_manifest(json.loads((bundle / "manifest.json").read_text()))
    out = bundle / "unreal"
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"dir": str(out)}
    result["warnings"] = check_volumes(manifest, bundle)
    for w in result["warnings"]:
        log.warning("unreal: %s", w)
    result["scripts"] = copy_scripts(bundle, out)
    result["luts"] = write_luts(manifest, out / "luts")
    result["planes"] = write_slice_planes(manifest, out / "meshes")
    result["camera_bake"] = (
        str(write_camera_bake(manifest, out / "camera_bake.json"))
        if camera_bake
        else None
    )
    readme = out / "README.md"
    readme.write_text(render_readme(manifest, bundle))
    result["readme"] = str(readme)
    sh, bat = render_commands(manifest, bundle)
    sh_path, bat_path = out / "ue_commands.sh", out / "ue_commands.bat"
    sh_path.write_text(sh)
    sh_path.chmod(0o755)
    bat_path.write_bytes(bat.encode())
    result["commands"] = [str(sh_path), str(bat_path)]
    return result


def _cli(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Stage an les-render bundle for Unreal Engine"
    )
    ap.add_argument("bundle")
    ap.add_argument("--no-camera-bake", action="store_true")
    args = ap.parse_args(argv)
    res = prepare_unreal(args.bundle, camera_bake=not args.no_camera_bake)
    print(json.dumps(res, indent=2))


__all__ = [
    "prepare_unreal",
    "lut_png",
    "png_bytes",
    "linear_to_srgb",
    "write_luts",
    "write_slice_planes",
    "camera_samples",
    "sim_to_ue",
    "sim_dir_to_ue",
    "look_at_rotator",
    "rotator_forward",
    "fps_to_fraction",
    "file_index",
    "volume_transform",
    "geometry_fix",
    "slice_quad",
]


if __name__ == "__main__":
    _cli()

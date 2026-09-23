"""Launch the headless Blender stage (``les_render/blender/build_scene.py``).

``run_blender(bundle_dir, render=True, ...)`` runs ``blender -b -P
build_scene.py -- --bundle DIR ...`` in a subprocess, streams its log (and
raises ``BlenderError`` with the log tail on failure), and returns the preview
frames directory (``render=True``) or the alembic directory (otherwise).

Blender is found via ``$BLENDER`` or ``PATH``. The Blender side only needs its
own bundled Python (bpy + numpy), never this package.

Two environment details matter for speed and are handled here:

* EEVEE Next compiles its shaders on the first frame; with the default
  ``max_shader_compilation_subprocesses = 0`` that takes ~2 min, with 8
  subprocesses ~30 s. The preference only applies at start-up and
  ``--factory-startup`` ignores user prefs, so we point
  ``BLENDER_USER_CONFIG`` at a private config dir holding a prepared
  ``userpref.blend`` (created on first use; the user's own Blender config is
  never touched).
* Cycles on a distro Blender without precompiled CUDA kernels compiles them
  with the system ``nvcc`` on first use. If the host gcc is too new for that
  nvcc, set ``$LES_CUDA_CCBIN`` to an older ``g++`` (it is forwarded as
  ``CYCLES_CUDA_EXTRA_CFLAGS=-ccbin ...``); the kernel is cached afterwards
  in ``~/.cache/cycles/kernels``. Without a GPU kernel Cycles falls back to CPU.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
import time
from collections import deque
from typing import Any, Optional, Sequence

log = logging.getLogger(__name__)

BUILD_SCENE = pathlib.Path(__file__).resolve().parent / "blender" / "build_scene.py"
CONFIG_DIR = pathlib.Path(
    os.environ.get(
        "LES_BLENDER_CONFIG",
        pathlib.Path.home() / ".cache" / "les_render" / "blender_config",
    )
)


class BlenderError(RuntimeError):
    pass


def find_blender(blender: Optional[str] = None) -> str:
    exe = blender or os.environ.get("BLENDER") or shutil.which("blender")
    if not exe or not (pathlib.Path(exe).exists() or shutil.which(exe)):
        raise FileNotFoundError(
            "Blender not found: install it, put it on PATH or set $BLENDER"
        )
    return str(exe)


def _ensure_config(blender: str, subprocesses: int) -> pathlib.Path:
    """Private Blender config with parallel shader compilation enabled."""
    cfg = CONFIG_DIR / "config"
    marker = cfg / f"userpref.{subprocesses}.ok"
    if marker.exists():
        return cfg
    cfg.mkdir(parents=True, exist_ok=True)
    expr = (
        "import bpy; p = bpy.context.preferences; "
        f"p.system.max_shader_compilation_subprocesses = {int(subprocesses)}; "
        "bpy.ops.wm.save_userpref()"
    )
    env = {**os.environ, "BLENDER_USER_CONFIG": str(cfg)}
    proc = subprocess.run(
        [blender, "-b", "--factory-startup", "--python-expr", expr],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode == 0 and (cfg / "userpref.blend").exists():
        for old in cfg.glob("userpref.*.ok"):
            old.unlink()
        marker.touch()
    else:
        log.warning(
            "could not prepare Blender prefs (%s); using factory settings",
            proc.stderr[-500:],
        )
    return cfg


def blender_command(
    bundle_dir: pathlib.Path | str,
    render: bool = True,
    export_alembic: bool = False,
    engine: str = "eevee",
    samples: Optional[int] = None,
    frames: Optional[str | Sequence[int]] = None,
    look: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    out: Optional[pathlib.Path | str] = None,
    save_blend: bool = False,
    extra_args: Sequence[str] = (),
    blender: Optional[str] = None,
    use_factory_startup: bool = False,
) -> list[str]:
    cmd = [find_blender(blender), "-b"]
    if use_factory_startup:
        cmd.append("--factory-startup")
    cmd += [
        "--python-exit-code",
        "1",
        "-P",
        str(BUILD_SCENE),
        "--",
        "--bundle",
        str(bundle_dir),
    ]
    if render:
        cmd.append("--render")
    if export_alembic:
        cmd.append("--export-alembic")
    if save_blend:
        cmd.append("--save-blend")
    cmd += ["--engine", engine]
    if samples is not None:
        cmd += ["--samples", str(int(samples))]
    if frames is not None:
        cmd += [
            "--frames",
            (
                frames
                if isinstance(frames, str)
                else ",".join(str(int(f)) for f in frames)
            ),
        ]
    if look:
        cmd += ["--look", look]
    if width:
        cmd += ["--width", str(int(width))]
    if height:
        cmd += ["--height", str(int(height))]
    if out:
        cmd += ["--out", str(out)]
    cmd += list(extra_args)
    return cmd


def run_blender(
    bundle_dir: pathlib.Path | str,
    render: bool = True,
    export_alembic: bool = False,
    engine: str = "eevee",
    samples: Optional[int] = None,
    frames: Optional[str | Sequence[int]] = None,
    look: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    out: Optional[pathlib.Path | str] = None,
    save_blend: bool = False,
    extra_args: Sequence[str] = (),
    blender: Optional[str] = None,
    shader_subprocesses: int = 8,
    timeout: Optional[float] = None,
    stream: bool = True,
    **unused: Any,
) -> pathlib.Path:
    """Build the Blender scene for ``bundle_dir`` and render and/or export Alembic.

    Returns ``out`` (default ``<bundle>/preview/frames``) when rendering,
    else ``<bundle>/alembic``. Raises ``BlenderError`` if Blender fails.
    """
    if unused:
        log.debug("run_blender: ignoring options %s", sorted(unused))
    bundle_dir = pathlib.Path(bundle_dir).resolve()
    if not (bundle_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"{bundle_dir} has no manifest.json")
    exe = find_blender(blender)
    env = dict(os.environ)
    use_factory = True
    if shader_subprocesses > 0:
        try:
            env["BLENDER_USER_CONFIG"] = str(_ensure_config(exe, shader_subprocesses))
            use_factory = False
        except Exception as exc:  # never fail a render over a preference
            log.warning("blender prefs setup failed: %s", exc)
    ccbin = os.environ.get("LES_CUDA_CCBIN")
    if ccbin and "CYCLES_CUDA_EXTRA_CFLAGS" not in env:
        env["CYCLES_CUDA_EXTRA_CFLAGS"] = f"-ccbin {ccbin}"
    env.setdefault(
        "PYTHONNOUSERSITE", "1"
    )  # keep the host python's site-packages out of Blender
    cmd = blender_command(
        bundle_dir,
        render=render,
        export_alembic=export_alembic,
        engine=engine,
        samples=samples,
        frames=frames,
        look=look,
        width=width,
        height=height,
        out=out,
        save_blend=save_blend,
        extra_args=extra_args,
        blender=exe,
        use_factory_startup=use_factory,
    )
    log.info("blender: %s", " ".join(cmd))
    tail: deque[str] = deque(maxlen=60)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            # Blender's per-tile progress spam ("Fra:12 Mem:...") is noise
            if stream and not line.startswith("Fra:"):
                (
                    log.info("[blender] %s", line)
                    if line.startswith("[les]")
                    else log.debug("[blender] %s", line)
                )
        rc = proc.wait(timeout=timeout)
    except BaseException:
        proc.kill()
        raise
    if rc != 0 or any(line.startswith("Traceback") for line in tail):
        raise BlenderError(
            f"blender exited with {rc}; last output:\n" + "\n".join(tail)
        )
    log.info("blender finished in %.1f s", time.perf_counter() - t0)
    if render:
        return pathlib.Path(out).resolve() if out else bundle_dir / "preview" / "frames"
    return bundle_dir / "alembic"


def render_preview(
    bundle_dir: pathlib.Path | str,
    mp4: Optional[pathlib.Path | str] = None,
    crf: int = 18,
    **blender_opts: Any,
) -> pathlib.Path:
    """Render the preview frames with Blender and encode ``preview/<case>.mp4``
    (with the HUD overlay when the bundle has one)."""
    import json

    bundle_dir = pathlib.Path(bundle_dir).resolve()
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    frames_dir = run_blender(bundle_dir, render=True, **blender_opts)
    mp4 = (
        pathlib.Path(mp4)
        if mp4
        else bundle_dir / "preview" / f"{manifest['case']['name']}.mp4"
    )
    hud = manifest.get("hud")
    hud_pattern = None
    if hud and int(hud.get("frame_step", 1)) == 1:
        hud_pattern = str(bundle_dir / hud["pattern"].replace("{frame:04d}", "%04d"))
    frames = sorted(frames_dir.glob("[0-9][0-9][0-9][0-9].png"))
    if not frames:
        raise FileNotFoundError(f"no rendered frames in {frames_dir}")
    first = int(frames[0].stem)
    fps = float(manifest["timeline"]["fps"])
    size = _png_size(frames[0])
    hud_first = pathlib.Path(hud_pattern % first) if hud_pattern else None
    same_size = hud_first is None or (
        hud_first.is_file() and _png_size(hud_first) == size
    )
    if same_size:
        try:
            from les_render.compose import compose_video

            return compose_video(
                str(frames_dir / "%04d.png"),
                mp4,
                fps=fps,
                hud_pattern=hud_pattern,
                crf=crf,
                start_number=first,
            )
        except ImportError:
            pass
    # preview at a different resolution than the HUD: scale the overlay
    return _ffmpeg(frames_dir, mp4, fps, hud_pattern, crf, first, size)


def _png_size(path: pathlib.Path) -> tuple[int, int]:
    import struct

    with open(path, "rb") as fh:
        head = fh.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def _ffmpeg(
    frames_dir: pathlib.Path,
    mp4: pathlib.Path,
    fps: float,
    hud_pattern: Optional[str],
    crf: int,
    first: int,
    size: Optional[tuple[int, int]] = None,
) -> pathlib.Path:
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise FileNotFoundError("ffmpeg not found on PATH")
    mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ff,
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(first),
        "-i",
        str(frames_dir / "%04d.png"),
    ]
    if hud_pattern:
        cmd += [
            "-framerate",
            str(fps),
            "-start_number",
            str(first),
            "-i",
            hud_pattern,
            "-filter_complex",
            (
                f"[1:v]scale={size[0]}:{size[1]}:flags=lanczos[h];"
                if size
                else "[1:v]null[h];"
            )
            + "[0:v][h]overlay=0:0:format=auto,format=yuv420p[v]",
            "-map",
            "[v]",
        ]
    else:
        cmd += ["-pix_fmt", "yuv420p"]
    cmd += [
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "slow",
        "-movflags",
        "+faststart",
        str(mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return mp4

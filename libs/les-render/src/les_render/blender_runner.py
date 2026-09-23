"""Launch the headless Blender stage (``les_render/blender/build_scene.py``).

``run_blender(bundle_dir, render=True, ...)`` runs ``blender -b -P
build_scene.py -- --bundle DIR ...`` in a subprocess, streams its log (and
raises ``BlenderError`` with the log tail on failure), and returns the preview
frames directory (``render=True``) or the alembic directory (otherwise); one
run can render and export Alembic together. ``encode_preview`` turns the
rendered frames (+ HUD) into ``preview/<case>.mp4``.

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
import threading
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
    # The read loop below blocks until Blender closes stdout, so the timeout
    # has to kill the process from a timer rather than via proc.wait().
    timer = threading.Timer(timeout, proc.kill) if timeout else None
    if timer is not None:
        timer.start()
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
        rc = proc.wait()
    except BaseException:
        proc.kill()
        raise
    finally:
        if timer is not None:
            timer.cancel()
    if timeout and time.perf_counter() - t0 >= timeout and rc != 0:
        raise BlenderError(f"blender timed out after {timeout:.0f} s")
    if rc != 0 or any(line.startswith("Traceback") for line in tail):
        raise BlenderError(
            f"blender exited with {rc}; last output:\n" + "\n".join(tail)
        )
    log.info("blender finished in %.1f s", time.perf_counter() - t0)
    if render:
        return pathlib.Path(out).resolve() if out else bundle_dir / "preview" / "frames"
    return bundle_dir / "alembic"


def preview_frames(frames: Optional[str | Sequence[int]], n_frames: int) -> list[int]:
    """Video frames a ``--frames`` spec selects (same rules as the Blender side:
    ``a:b`` and ``a:b:s`` are inclusive, ``f1,f2`` is a list, None is all)."""
    if frames is None:
        return list(range(n_frames))
    if not isinstance(frames, str):
        return [int(f) for f in frames]
    if "," in frames or ":" not in frames:
        return [int(f) for f in frames.split(",") if f.strip()]
    parts = [int(p) if p else None for p in frames.split(":")]
    a = parts[0] or 0
    b = parts[1] if len(parts) > 1 and parts[1] is not None else n_frames - 1
    step = parts[2] if len(parts) > 2 and parts[2] else 1
    return list(range(max(a, 0), min(b, n_frames - 1) + 1, step))


def encode_preview(
    bundle_dir: pathlib.Path | str,
    frames: Optional[str | Sequence[int]] = None,
    frames_dir: Optional[pathlib.Path | str] = None,
    mp4: Optional[pathlib.Path | str] = None,
    crf: int = 18,
) -> pathlib.Path:
    """Encode rendered preview frames (+ HUD overlay) to ``preview/<case>.mp4``.

    Only the frames ``frames`` selects are encoded (a contiguous range is
    required), starting at the first of them, so stale files from an earlier,
    longer render never leak in. The HUD is scaled to the frame size when the
    preview was rendered at a different resolution, and skipped with a warning
    when it was exported with ``frame_step > 1`` (ffmpeg pairs files 1:1).
    """
    import json

    from les_render.compose import compose_video

    bundle_dir = pathlib.Path(bundle_dir).resolve()
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    frames_dir = (
        pathlib.Path(frames_dir) if frames_dir else bundle_dir / "preview" / "frames"
    )
    wanted = preview_frames(frames, int(manifest["timeline"]["n_frames"]))
    if not wanted:
        raise ValueError(f"frame selection {frames!r} is empty")
    first, count = wanted[0], len(wanted)
    if wanted != list(range(first, first + count)):
        raise ValueError(
            f"video needs a contiguous frame range, got {frames!r}; "
            "render with a:b or skip the video stage"
        )
    missing = [f for f in wanted if not (frames_dir / f"{f:04d}.png").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} preview frames missing in {frames_dir} "
            f"(first: {missing[0]:04d}.png)"
        )
    mp4 = (
        pathlib.Path(mp4)
        if mp4
        else bundle_dir / "preview" / f"{manifest['case']['name']}.mp4"
    )
    hud = manifest.get("hud")
    hud_pattern: Optional[str] = None
    hud_size: Optional[tuple[int, int]] = None
    if hud:
        if int(hud.get("frame_step", 1)) != 1:
            log.warning("HUD exported with frame_step > 1: encoding without it")
        else:
            hud_pattern = str(
                bundle_dir / hud["pattern"].replace("{frame:04d}", "%04d")
            )
            size = _png_size(frames_dir / f"{first:04d}.png")
            if _png_size(pathlib.Path(hud_pattern % first)) != size:
                hud_size = size
    return compose_video(
        str(frames_dir / "%04d.png"),
        mp4,
        fps=float(manifest["timeline"]["fps"]),
        hud_pattern=hud_pattern,
        crf=crf,
        start_number=first,
        frame_count=count,
        hud_size=hud_size,
    )


def _png_size(path: pathlib.Path) -> tuple[int, int]:
    import struct

    with open(path, "rb") as fh:
        head = fh.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)

"""Compose rendered frames (+ optional HUD overlay) into a delivery mp4, via ffmpeg.

Kept as thin ``subprocess`` wrappers around ``ffmpeg`` -- no Python video
library dependency. Both functions raise :class:`FFmpegNotFoundError` (a
clear, actionable error) rather than a raw ``FileNotFoundError`` when
``ffmpeg`` isn't on ``PATH``.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
from typing import Optional, Sequence


class FFmpegNotFoundError(RuntimeError):
    """``ffmpeg`` is not on ``PATH`` (or the configured binary is missing)."""


def _ffmpeg_bin(ffmpeg: Optional[str] = None) -> str:
    candidate = ffmpeg or "ffmpeg"
    found = shutil.which(candidate) or (
        candidate if pathlib.Path(candidate).is_file() else None
    )
    if found is None:
        raise FFmpegNotFoundError(
            f"ffmpeg not found ({candidate!r} is not on PATH and is not an existing file). "
            "Install ffmpeg or pass compose_video(..., ffmpeg='/path/to/ffmpeg')."
        )
    return found


def _run(cmd: Sequence[str]) -> None:
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed (exit {proc.returncode}): {' '.join(cmd)}\n\n{proc.stdout}"
        )


def compose_video(
    frames_pattern: str,
    out_mp4: str | pathlib.Path,
    fps: float,
    hud_pattern: Optional[str] = None,
    crf: int = 18,
    start_number: int = 0,
    ffmpeg: Optional[str] = None,
    extra_input_args: Optional[Sequence[str]] = None,
    frame_count: Optional[int] = None,
    hud_size: Optional[tuple[int, int]] = None,
) -> pathlib.Path:
    """Encode a frame sequence (optionally with a transparent HUD overlay) to
    H.264 mp4.

    ``start_number`` is the index of the first frame file (the HUD sequence is
    read from the same index); ``frame_count`` stops after that many frames, so
    stale files beyond a partial render are never encoded. ``hud_size`` scales
    the HUD to ``(width, height)`` when it was drawn at a different resolution
    than the frames.

    ``frames_pattern`` / ``hud_pattern`` are ffmpeg ``-i`` printf patterns
    (e.g. ``"render/frame.%04d.png"``); PNG or any format ffmpeg reads.
    Output is ``yuv420p`` (broadly compatible, e.g. QuickTime) with
    ``-movflags +faststart`` (moov atom moved to the front for progressive/
    web playback). Raises :class:`FFmpegNotFoundError` if ``ffmpeg`` is
    missing, and :class:`RuntimeError` (with ffmpeg's own output) on any
    encode failure.
    """
    binary = _ffmpeg_bin(ffmpeg)
    out_mp4 = pathlib.Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [binary, "-y", "-loglevel", "error"]
    cmd += ["-framerate", str(fps), "-start_number", str(start_number)]
    cmd += list(extra_input_args or [])
    cmd += ["-i", str(frames_pattern)]

    if hud_pattern is not None:
        cmd += [
            "-framerate",
            str(fps),
            "-start_number",
            str(start_number),
            "-i",
            str(hud_pattern),
        ]
        # HUD (straight-alpha RGBA) composited over the render; straight alpha
        # is what ffmpeg's overlay filter expects by default.
        scale = (
            f"[1:v]scale={hud_size[0]}:{hud_size[1]}:flags=lanczos[h];"
            if hud_size
            else "[1:v]null[h];"
        )
        cmd += [
            "-filter_complex",
            scale + "[0:v][h]overlay=format=auto[v]",
            "-map",
            "[v]",
        ]
    if frame_count is not None:
        cmd += ["-frames:v", str(int(frame_count))]

    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    _run(cmd)
    return out_mp4


def make_contact_sheet(
    frames_pattern: str,
    out_png: str | pathlib.Path,
    columns: int = 5,
    rows: int = 4,
    tile_width: int = 320,
    start_number: int = 0,
    step: int = 1,
    ffmpeg: Optional[str] = None,
) -> pathlib.Path:
    """A quick ``columns`` x ``rows`` thumbnail grid of the sequence, for a
    fast visual sanity check without opening the video."""
    binary = _ffmpeg_bin(ffmpeg)
    out_png = pathlib.Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = columns * rows
    tile_h = round(tile_width * 9 / 16)
    vf = f"select='not(mod(n\\,{step}))',scale={tile_width}:{tile_h},tile={columns}x{rows}"
    cmd = [
        binary,
        "-y",
        "-loglevel",
        "error",
        "-start_number",
        str(start_number),
        "-i",
        str(frames_pattern),
        "-frames:v",
        "1",
        "-vf",
        vf,
        "-vframes",
        "1",
        str(out_png),
    ]
    _run(cmd)
    return out_png


__all__ = ["compose_video", "make_contact_sheet", "FFmpegNotFoundError"]

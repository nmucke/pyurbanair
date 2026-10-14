"""Side-by-side movies of horizontal slices, for the comparison scripts.

`animate_slices` lays out a grid of panels (rows, e.g. heights; columns, e.g.
models then their differences), each a (time, y, x) stack, and writes one
frame per time: an MP4 when ffmpeg is available (it is a pixi dependency),
else a GIF. Colour limits are fixed per panel over the whole movie, so frames
compare, and the background is white so the movie reads in a dark viewer.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


@dataclass
class Panel:
    """One panel: `frames` (time, y, x), shown with `cmap` between vmin and vmax.

    Panels of a row with the same `group` share one colour bar, drawn after
    the group's last panel; give them the same limits.
    """

    title: str
    frames: np.ndarray
    cmap: str
    vmin: float
    vmax: float
    group: str = ""


def frame_indices(n_time: int, max_frames: int) -> np.ndarray:
    """At most `max_frames` evenly spaced indices into `n_time` frames."""
    return np.unique(np.linspace(0, n_time - 1, min(n_time, max_frames)).astype(int))


def animate_slices(
    rows: Sequence[Sequence[Panel]],
    times: Sequence[float],
    path: pathlib.Path,
    *,
    title: str,
    fps: int,
    dpi: int,
    extent: Sequence[float] | None = None,
    labels: tuple[str, str] = ("x [m]", "y [m]"),
) -> pathlib.Path:
    """Write the movie of `rows` to `path` (suffix replaced by .mp4 or .gif).

    `times` labels each frame (`title` gets ", t = <time> s"); every panel
    holds one slice per entry of `times`. Returns the written path.
    """
    n_cols = max(len(row) for row in rows)
    fig, axes = plt.subplots(
        len(rows),
        n_cols,
        figsize=(3.0 * n_cols, 2.8 * len(rows)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.set_facecolor("white")
    images = []
    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            ax = axes[r, c]
            image = ax.imshow(
                panel.frames[0],
                origin="lower",
                extent=extent,
                cmap=panel.cmap,
                vmin=panel.vmin,
                vmax=panel.vmax,
                aspect="equal",
            )
            images.append((image, panel.frames))
            ax.set_title(panel.title, fontsize=9)
            last = c == len(row) - 1 or row[c + 1].group != panel.group
            if last:  # one bar per group keeps the panels the same size
                group = [axes[r, j] for j, p in enumerate(row) if p.group == panel.group]
                fig.colorbar(image, ax=group, shrink=0.8)
        for c in range(len(row), n_cols):
            axes[r, c].set_visible(False)
        axes[r, 0].set_ylabel(labels[1])
    for ax in axes[-1]:
        ax.set_xlabel(labels[0])
    heading = fig.suptitle("")

    def draw(t: int) -> list:
        for image, frames in images:
            image.set_data(frames[t])
        heading.set_text(f"{title}, t = {float(times[t]):.0f} s")
        return [image for image, _ in images]

    movie = animation.FuncAnimation(fig, draw, frames=len(times), blit=False)
    if animation.writers.is_available("ffmpeg"):
        path, writer = pathlib.Path(path).with_suffix(".mp4"), "ffmpeg"
    else:
        path, writer = pathlib.Path(path).with_suffix(".gif"), "pillow"
    movie.save(
        path, writer=writer, fps=fps, dpi=dpi, savefig_kwargs={"facecolor": "white"}
    )
    plt.close(fig)
    print(f"  + {path}")
    return path

"""Render the HTML viewer of a finished forward run, or serve one.

    python -m visualization <run dir> <bundle dir>
    python -m visualization --serve <bundle dir>

Rendering uses the default options (the `quicklook` preset) plus the
interactive 3D view. An ensemble shows its mean velocity. The stride is the
smallest that fits every saved time in `max_frames` and the 3D scene in its
size limit. The bundle dir must be new.

`--serve` prints a local URL and serves the bundle until Ctrl-C or 30 idle
minutes. Use it rather than a plain static server: the viewer seeks in its
movies, which needs HTTP byte ranges (`python -m http.server` has none, so
the movies stay on their first frame).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import xarray as xr

from .assets import BundleAssetServer
from .data import _coordinates
from .render import RenderOptions, render
from .render_3d import scene_frames


def main(run_dir: Path, bundle: Path) -> None:
    with xr.open_dataset(run_dir / "state.nc", decode_times=False) as ds:
        frames = ds.sizes["time"]
        ensemble = ds.sizes.get("ensemble", 1) > 1
        sizes = {axis: len(values) for axis, values in _coordinates(ds).items()}
    try:
        budget = min(RenderOptions.max_frames, scene_frames(sizes))
        render_3d = True
    except ValueError:  # fewer than two cells on an axis: no 3D view
        budget, render_3d = RenderOptions.max_frames, False
    options = RenderOptions(
        reduction="mean_velocity" if ensemble else None,
        stride=max(1, math.ceil(frames / max(1, budget))),
        render_3d=render_3d,
    )
    manifest = render(run_dir, bundle, options)
    for warning in manifest.get("warnings", []):
        print(f"Warning: {warning}")
    print(f"Saved viewer in {bundle}")


def serve(bundle: Path) -> None:
    with BundleAssetServer() as server:
        print(f"Viewer: {server.register(bundle)}  (Ctrl-C to stop)", flush=True)
        try:
            server.thread.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    if sys.argv[1] == "--serve":
        serve(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))

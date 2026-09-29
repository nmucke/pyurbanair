"""Optional bounded PyVista renderer; imported only for an explicit 3D request."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    from .data import ArtifactReader
    from .render import RenderOptions

import numpy as np

from .data import fingerprint


def rectilinear_field(ds: xr.Dataset) -> Any:
    """Use physical cell centres as VTK sampling points, with x varying fastest."""
    import pyvista as pv

    if any(ds.sizes[axis] < 2 for axis in "xyz"):
        raise ValueError("3D rendering needs at least two points on each axis")
    grid = pv.RectilinearGrid(ds.x.values, ds.y.values, ds.z.values)
    components = [ds[name].transpose("z", "y", "x").values for name in ("u", "v", "w")]
    vectors = np.column_stack([values.ravel(order="C") for values in components])
    grid.point_data["velocity"] = vectors
    grid.point_data["speed"] = np.linalg.norm(vectors, axis=1)
    valid = np.logical_and.reduce([np.isfinite(values) for values in components])
    # Remove every interpolation cell touching a solid/invalid sample. Streamline
    # integration then terminates at the resulting mesh boundary, conservatively.
    cell_valid = np.logical_and.reduce(
        [
            valid[
                z : z + valid.shape[0] - 1,
                y : y + valid.shape[1] - 1,
                x : x + valid.shape[2] - 1,
            ]
            for z in (0, 1)
            for y in (0, 1)
            for x in (0, 1)
        ]
    )
    return grid.extract_cells(np.flatnonzero(cell_valid.ravel(order="C")))


def render_3d(
    reader: ArtifactReader,
    output: Path,
    times: list[float],
    opts: RenderOptions,
    limits: list[float],
) -> dict[str, Any]:
    import pyvista as pv

    from .render import _slice, encode_movie

    frame_dir = output / "media" / "flow-3d"
    frame_dir.mkdir()
    geometry = None
    geometry_metadata = None
    if opts.geometry:
        path = Path(opts.geometry).resolve()
        if path.suffix.lower() != ".stl" or not path.is_file():
            raise ValueError(
                "3D geometry must be an existing STL in the state coordinate frame"
            )
        geometry = pv.read(path)
        geometry_metadata = {
            "path": str(path),
            "sha256": fingerprint(path),
            "scale": 1.0,
        }
    first = reader.frame(times[0])
    if any(first.sizes[axis] < 2 for axis in "xyz"):
        raise ValueError("3D rendering needs at least two points on each axis")
    if geometry is not None:
        bounds = geometry.bounds
        if any(
            bounds[2 * i + 1] < float(first[axis].min())
            or bounds[2 * i] > float(first[axis].max())
            for i, axis in enumerate("xyz")
        ):
            raise ValueError(
                "STL does not overlap the state domain; check coordinate alignment"
            )
        if "blanking" not in first:
            raise ValueError(
                "Geometry rendering requires an explicit blanking mask to prevent streamlines crossing solids"
            )
        # An STL can cover a much larger city than this saved simulation. Show
        # only its intersection with the cell-face domain, in physical metres.
        display_bounds = []
        for axis in "xyz":
            coordinates = first[axis].values
            display_bounds.extend(
                [
                    float(coordinates[0] - (coordinates[1] - coordinates[0]) / 2),
                    float(coordinates[-1] + (coordinates[-1] - coordinates[-2]) / 2),
                ]
            )
        geometry = geometry.clip_box(display_bounds, invert=False)
        assert geometry_metadata is not None
        geometry_metadata["display_bounds"] = display_bounds
        geometry_metadata["clipping"] = "saved field cell-face domain; no rescaling"
    seeds = np.asarray(opts.seeds, dtype=float)
    if not len(seeds):
        # Deterministic inlet seeds; invalid ones are explicitly removed below.
        seeds = np.array(
            [
                [float(first.x.values[0]), float(y), float(z)]
                for y in np.linspace(float(first.y.min()), float(first.y.max()), 5)
                for z in np.linspace(float(first.z.min()), float(first.z.max()), 4)
            ]
        )
    seeds_source = pv.PolyData(seeds)
    mapping = []
    for index, time in enumerate(times):
        ds = reader.frame(time)
        field = rectilinear_field(ds)
        if not field.n_cells:
            raise ValueError("No valid interpolation cells remain for 3D rendering")
        sampled = seeds_source.sample(field)
        valid_seeds = np.asarray(sampled["vtkValidPointMask"], dtype=bool)
        if opts.seeds and not valid_seeds.all():
            raise ValueError(
                "A requested streamline seed is outside fluid interpolation cells"
            )
        lines = None
        if valid_seeds.any():
            lines = field.streamlines_from_source(
                pv.PolyData(seeds[valid_seeds]),
                vectors="velocity",
                max_steps=opts.max_steps,
                integration_direction="forward",
                compute_vorticity=False,
            )
        spec = opts.slices[0]
        axis = spec["axis"]
        origin = [float(ds[value].mean()) for value in "xyz"]
        _, slice_metadata = _slice(ds, spec)
        origin["xyz".index(axis)] = slice_metadata["actual"]
        plane = field.slice(normal=axis, origin=origin)
        axis_index = "xyz".index(axis)
        if not plane.n_points and origin[axis_index] in (
            field.bounds[2 * axis_index],
            field.bounds[2 * axis_index + 1],
        ):
            # VTK's cutting plane can omit a coincident outer face. Extract
            # those exact faces instead of moving the requested physical plane.
            surface = field.extract_surface()
            centres = surface.cell_centers().points[:, axis_index]
            tolerance = max(1.0, abs(origin[axis_index])) * 1e-12
            plane = surface.extract_cells(
                np.flatnonzero(abs(centres - origin[axis_index]) <= tolerance)
            )
        slice_metadata["rendered_points"] = plane.n_points
        plotter = pv.Plotter(off_screen=True, window_size=(opts.width, opts.height))
        try:
            plotter.set_background("#071c29")
            if geometry is not None and geometry.n_points:
                plotter.add_mesh(geometry, color="#adc6cd")
            if plane.n_points:
                plotter.add_mesh(
                    plane,
                    scalars="speed",
                    clim=limits,
                    cmap=opts.cmap,
                    scalar_bar_args={"color": "white", "fmt": "%.3g"},
                )
            if lines is not None and lines.n_points:
                plotter.add_mesh(
                    lines,
                    scalars="speed",
                    clim=limits,
                    cmap=opts.cmap,
                    line_width=2,
                    scalar_bar_args={"color": "white", "fmt": "%.3g"},
                )
            plotter.add_text(
                f"Instantaneous streamlines · t={time:g} s", font_size=12, color="white"
            )
            if opts.camera is not None:
                plotter.camera_position = opts.camera
            else:
                plotter.view_isometric()
            plotter.show(auto_close=False)
            plotter.screenshot(frame_dir / f"{index:05d}.png")
        finally:
            plotter.close()
        mapping.append({"video_time": index / opts.fps, "simulation_time": time})
    poster = "previews/flow-3d.png"
    shutil.copyfile(frame_dir / "00000.png", output / poster)
    movie = None
    if (
        opts.movie
        and len(times) > 1
        and encode_movie(frame_dir, output / "media" / "flow-3d.mp4", opts.fps)
    ):
        movie = "media/flow-3d.mp4"
    return {
        "id": "flow-3d",
        "label": "3D instantaneous streamlines",
        "kind": "3d",
        "poster": poster,
        "media": movie,
        "mime_type": "video/mp4" if movie else "image/png",
        "width": opts.width,
        "height": opts.height,
        "field": "speed",
        "units": "m/s",
        "color_limits": limits,
        "camera": opts.camera,
        "slice": slice_metadata,
        "geometry": geometry_metadata,
        "seeds": seeds.tolist(),
        "vertical_exaggeration": 1,
        "frames": mapping,
        "snapshots": [
            {
                "simulation_time": time,
                "path": str((frame_dir / f"{index:05d}.png").relative_to(output)),
            }
            for index, time in enumerate(times)
        ],
        "duration": len(times) / opts.fps,
    }


def probe_offscreen() -> dict:
    """Run this in an isolated readiness child: graphics initialization may fail."""
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=(128, 128))
    try:
        plotter.add_mesh(pv.Sphere())
        plotter.show(auto_close=False)
        image = plotter.screenshot(return_img=True)
    finally:
        plotter.close()
    if image is None or image.shape[:2] != (128, 128):
        raise RuntimeError("VTK did not produce an offscreen image")
    return {"available": True, "pyvista": pv.__version__, "vtk": pv.vtk_version_info}

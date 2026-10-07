"""Bounded postprocessing of persisted forward states into standalone bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from .data import ArtifactReader, fingerprint


@dataclass
class RenderOptions:
    member: Any = None
    reduction: str | None = None
    variable: str = "horizontal_speed"
    slices: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"axis": "z", "fraction": 0.0},
            {"axis": "z", "fraction": 0.65},
            {"axis": "y", "fraction": 0.5, "variable": "w"},
        ]
    )
    probes: list[dict[str, Any]] | None = None
    time_start: float | None = None
    time_end: float | None = None
    stride: int = 1
    max_frames: int = 60
    max_cells: int = 8_000_000
    fps: int = 12
    width: int = 960
    height: int = 640
    cmap: str = "viridis"
    color_limits: list[float] | None = None
    movie: bool = True
    render_3d: bool = False
    geometry: str | None = None
    camera: list[list[float]] | None = None
    seeds: list[list[float]] = field(default_factory=list)
    max_steps: int = 500
    simulation_seconds_per_video_second: float | None = None


def validate_options(
    options: dict[str, Any] | RenderOptions | None = None
) -> RenderOptions:
    opts = (
        options
        if isinstance(options, RenderOptions)
        else RenderOptions(**(options or {}))
    )
    if opts.member is not None and opts.reduction is not None:
        raise ValueError("Select a member or reduction, not both")
    if opts.reduction not in (None, "mean_velocity", "mean_speed"):
        raise ValueError("Reduction must be mean_velocity or mean_speed")
    if opts.variable not in ("u", "v", "w", "speed", "horizontal_speed"):
        raise ValueError("Unsupported visualization variable")
    for name, lower, upper in (
        ("stride", 1, 10000),
        ("max_frames", 1, 300),
        ("max_cells", 1, 8_000_000),
        ("fps", 1, 60),
        ("width", 128, 1920),
        ("height", 128, 1080),
        ("max_steps", 1, 2000),
    ):
        value = getattr(opts, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not lower <= value <= upper
        ):
            raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    if opts.width % 2 or opts.height % 2:
        raise ValueError("Image dimensions must be even for browser MP4 compatibility")
    if (
        not 1 <= len(opts.slices) <= 6
        or len(opts.probes or []) > 32
        or len(opts.seeds) > 128
    ):
        raise ValueError("Limits: 1–6 slices, 32 probes and 128 streamline seeds")
    for item in opts.slices:
        if set(item) - {"axis", "position", "fraction", "variable"} or item.get(
            "axis"
        ) not in ("x", "y", "z"):
            raise ValueError("Each slice has an axis (x/y/z) and physical position")
        if item.get("variable", opts.variable) not in (
            "u",
            "v",
            "w",
            "speed",
            "horizontal_speed",
        ):
            raise ValueError("Unsupported slice variable")
        if "fraction" in item and (
            not np.isfinite(item["fraction"])
            or not 0 <= item["fraction"] <= 1
            or item.get("position") is not None
        ):
            raise ValueError(
                "Slice fraction must be between 0 and 1, without a position"
            )
        if item.get("position") is not None and not np.isfinite(item["position"]):
            raise ValueError("Slice position must be finite")
    ids = set()
    for index, probe in enumerate(opts.probes or []):
        if set(probe) - {"id", "x", "y", "z"} or any(
            axis not in probe or not np.isfinite(probe[axis]) for axis in "xyz"
        ):
            raise ValueError(
                "Each probe needs finite x/y/z coordinates and an optional id"
            )
        identifier = str(probe.get("id", f"probe-{index + 1}"))
        if identifier in ids or len(identifier) > 80:
            raise ValueError("Probe IDs must be unique and at most 80 characters")
        ids.add(identifier)
    if opts.color_limits is not None and (
        len(opts.color_limits) != 2
        or not np.isfinite(opts.color_limits).all()
        or opts.color_limits[0] >= opts.color_limits[1]
    ):
        raise ValueError("color_limits must contain two increasing finite values")
    if (
        opts.time_start is not None
        and not np.isfinite(opts.time_start)
        or opts.time_end is not None
        and not np.isfinite(opts.time_end)
    ):
        raise ValueError("Time bounds must be finite")
    if (
        opts.time_start is not None
        and opts.time_end is not None
        and opts.time_start > opts.time_end
    ):
        raise ValueError("time_start must not exceed time_end")
    if opts.simulation_seconds_per_video_second is not None and (
        not np.isfinite(opts.simulation_seconds_per_video_second)
        or opts.simulation_seconds_per_video_second <= 0
    ):
        raise ValueError("simulation_seconds_per_video_second must be positive")
    if opts.camera is not None and (
        np.shape(opts.camera) != (3, 3) or not np.isfinite(opts.camera).all()
    ):
        raise ValueError("Camera must contain position, focal point and up vector")
    if any(len(seed) != 3 or not np.isfinite(seed).all() for seed in opts.seeds):
        raise ValueError("Streamline seeds must be finite XYZ positions")
    if opts.render_3d and opts.reduction == "mean_speed":
        raise ValueError("3D streamlines require a member or mean_velocity reduction")
    return opts


def _slice(ds: xr.Dataset, spec: dict[str, Any]) -> tuple[xr.Dataset, dict[str, Any]]:
    axis = spec["axis"]
    requested = spec.get("position")
    coord = ds[axis].values
    if requested is None:
        requested = (
            float(coord.min() + spec["fraction"] * (coord.max() - coord.min()))
            if "fraction" in spec
            else float(coord[len(coord) // 2])
        )
    if requested < coord.min() or requested > coord.max():
        raise ValueError(
            f"Slice {axis}={requested} lies outside the cell-centre domain"
        )
    position = int(np.argmin(np.abs(coord - requested)))
    return ds.isel({axis: position}), {
        "axis": axis,
        "requested": requested,
        "actual": float(coord[position]),
        "sampling": "nearest cell centre",
    }


def _default_probes(
    ds: xr.Dataset, slices: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare the same three horizontal cells at every displayed map height.

    These are deterministic virtual samples, not observations or relocated fluid
    points. A selected solid cell deliberately remains a gap in its trace.
    """
    heights = sorted(
        {_slice(ds, spec)[1]["actual"] for spec in slices if spec["axis"] == "z"}
    )
    positions = []
    for fraction in (0.2, 0.5, 0.8):
        requested = float(ds.x.min() + fraction * (ds.x.max() - ds.x.min()))
        x = float(ds.x.sel(x=requested, method="nearest"))
        side = next(
            (spec for spec in slices if spec["axis"] == "y"),
            {"axis": "y", "fraction": 0.5},
        )
        y = _slice(ds, side)[1]["actual"]
        if (x, y) not in positions:
            positions.append((x, y))
    return [
        {"id": f"{chr(65 + index)} · z={z:g} m", "x": x, "y": y, "z": z}
        for z in heights
        for index, (x, y) in enumerate(positions)
    ]


def _probe(
    ds: xr.Dataset, spec: dict[str, Any], variable: str
) -> tuple[float | None, dict[str, float]]:
    requested = {axis: float(spec[axis]) for axis in "xyz"}
    if any(
        value < float(ds[axis].min()) or value > float(ds[axis].max())
        for axis, value in requested.items()
    ):
        raise ValueError(
            f"Probe {spec.get('id', '')} lies outside the cell-centre domain"
        )
    # Nearest selection is explicit and never relocates samples out of solids.
    point = ds.sel(requested, method="nearest")
    value = float(point[variable])
    return value if np.isfinite(value) else None, {
        axis: float(point[axis]) for axis in "xyz"
    }


def encode_movie(frames: Path, destination: Path, fps: int) -> bool:
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        return False
    subprocess.run(
        [
            encoder,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
        timeout=300,
    )
    return True


def _write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False))


def render(
    run_root: str | Path,
    output_root: str | Path,
    options: dict[str, Any] | RenderOptions | None = None,
) -> dict[str, Any]:
    """Render saved fields, publishing the complete manifest only on success.

    Source files are read-only. Each frame is loaded independently; ensemble
    reductions have an explicit combined cell budget. Output must be a new path.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    opts = validate_options(options)
    if opts.cmap not in colormaps:
        raise ValueError(f"Unknown matplotlib color map {opts.cmap!r}")
    reader = ArtifactReader(
        run_root,
        member=opts.member,
        reduction=opts.reduction,
        max_cells=opts.max_cells,
        geometry=opts.geometry,
    )
    probe_times = [
        time
        for time in reader.times
        if (opts.time_start is None or time >= opts.time_start)
        and (opts.time_end is None or time <= opts.time_end)
    ]
    if not probe_times:
        raise ValueError("No saved frames in requested interval")
    if len(probe_times) > 10000:
        raise ValueError("Requested interval exceeds 10000 probe sample times")
    times = probe_times[:: opts.stride]
    if len(times) > opts.max_frames:
        raise ValueError(
            f"Request selects {len(times)} frames; increase stride (max_frames={opts.max_frames})"
        )
    repeats = [1] * len(times)
    if opts.simulation_seconds_per_video_second and len(times) > 1:
        repeats = [
            max(1, round((b - a) * opts.fps / opts.simulation_seconds_per_video_second))
            for a, b in zip(times, times[1:])
        ] + [1]
        if sum(repeats) > opts.max_frames:
            raise ValueError("Requested playback rate exceeds max_frames budget")
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Render output already exists: {output}")
    output.mkdir(parents=True)
    (output / "previews").mkdir()
    (output / "media").mkdir()
    automatic_probes = opts.probes is None
    probe_specs = (
        _default_probes(reader.frame(times[0]), opts.slices)
        if automatic_probes
        else opts.probes or []
    )
    effective_options = {
        **asdict(opts),
        "probes": probe_specs,
        "automatic_probes": automatic_probes,
    }
    (output / "render_config.resolved.yaml").write_text(
        yaml.safe_dump(effective_options)
    )
    warnings = set()
    if opts.movie and shutil.which("ffmpeg") is None:
        warnings.add("ffmpeg unavailable; PNG previews and probes remain available")
    probes: list[dict[str, Any]] = [
        {
            "id": str(spec.get("id", f"probe-{i + 1}")),
            "label": (
                str(spec["id"]).split(" · ")[0]
                if automatic_probes
                else str(spec.get("id", f"probe-{i + 1}"))
            ),
            "requested": {axis: spec[axis] for axis in "xyz"},
            "sampling": "nearest cell centre; solid/missing samples are null",
            "color": ["#73dbd4", "#ffbe73", "#a9a6ff", "#f17c9b"][
                (ord(str(spec["id"])[0]) - 65 if automatic_probes else i) % 4
            ],
            "values": [],
        }
        for i, spec in enumerate(probe_specs)
    ]
    variables = {spec.get("variable", opts.variable) for spec in opts.slices}
    extrema = {variable: [float("inf"), float("-inf")] for variable in variables}
    speed_extrema = [float("inf"), float("-inf")]
    reference_coords: dict[str, np.ndarray] = {}
    slice_metadata: list[dict[str, Any]] = []
    domain: dict[str, Any] = {}
    for time in probe_times:
        ds = reader.frame(time)
        warnings.update(ds.attrs["warnings"])
        if reference_coords and any(
            not np.array_equal(ds[axis].values, reference_coords[axis])
            for axis in "xyz"
        ):
            raise ValueError(
                "State coordinates change across frames; render each grid separately"
            )
        if not domain:
            reference_coords = {axis: ds[axis].values.copy() for axis in "xyz"}
            domain = {
                axis: {
                    "min": float(ds[axis].min()),
                    "max": float(ds[axis].max()),
                    "size": ds.sizes[axis],
                    "units": "m",
                }
                for axis in "xyz"
            }
        for probe, spec in zip(probes, probe_specs):
            value, actual = _probe(ds, spec, opts.variable)
            probe["actual"] = actual
            probe["values"].append(value)
        if time not in times:
            continue
        if opts.render_3d:
            speed_values = ds.speed.values
            finite_speed = speed_values[np.isfinite(speed_values)]
            if finite_speed.size:
                speed_extrema[0] = min(speed_extrema[0], float(finite_speed.min()))
                speed_extrema[1] = max(speed_extrema[1], float(finite_speed.max()))
        slice_metadata = []
        for spec in opts.slices:
            plane, metadata = _slice(ds, spec)
            slice_metadata.append(metadata)
            variable = spec.get("variable", opts.variable)
            values = plane[variable].values
            valid = values[np.isfinite(values)]
            if valid.size:
                extrema[variable][0] = min(extrema[variable][0], float(valid.min()))
                extrema[variable][1] = max(extrema[variable][1], float(valid.max()))
    field_limits = {}
    for variable, values in extrema.items():
        limits = values
        if opts.color_limits and variable == opts.variable:
            limits = opts.color_limits
        elif not np.isfinite(limits).all():
            limits = [-1.0, 1.0] if variable in ("u", "v", "w") else [0.0, 1.0]
            warnings.add(f"Selected {variable} slices have no finite fluid data")
        elif variable in ("u", "v", "w"):
            magnitude = max(abs(limits[0]), abs(limits[1]), 1e-9)
            limits = [-magnitude, magnitude]
        elif limits[0] == limits[1]:
            limits = [limits[0], limits[1] + max(abs(limits[1]) * 0.01, 1e-9)]
        field_limits[variable] = limits
    probe_data = {
        "version": 1,
        "field": opts.variable,
        "units": "m/s",
        "times": probe_times,
        "probes": probes,
    }
    _write_json(output / "probes.json", probe_data)
    with (output / "probes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["simulation_time_seconds"] + [p["id"] for p in probes])
        for i, time in enumerate(probe_times):
            writer.writerow([time] + [p["values"][i] for p in probes])
    views = []
    for view_index, metadata in enumerate(slice_metadata):
        view_id = f"slice-{view_index}"
        frame_dir = output / "media" / view_id
        frame_dir.mkdir()
        mapping = []
        snapshots = []
        frame_number = 0
        axis = metadata["axis"]
        view_height = (
            opts.height
            if axis == "z"
            else min(opts.height, max(128, 2 * round(opts.width / 6.4)))
        )
        text_scale = max(1.0, opts.width / (640 if axis == "z" else 960))
        variable = opts.slices[view_index].get("variable", opts.variable)
        limits = field_limits[variable]
        cmap = "RdBu_r" if variable in ("u", "v", "w") else opts.cmap
        plot_axes = [value for value in "xyz" if value != axis]
        for time, repeat in zip(times, repeats):
            ds = reader.frame(time)
            plane, _ = _slice(ds, opts.slices[view_index])
            fig = Figure(
                figsize=(opts.width / 100, view_height / 100),
                dpi=100,
                layout="constrained",
                facecolor="#091e2b",
            )
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            ax.set_facecolor("#142936")
            ax.tick_params(colors="#adc6cd", labelsize=10 * text_scale)
            for spine in ax.spines.values():
                spine.set_color("#31505d")
            plotted_values = plane[variable].transpose(plot_axes[1], plot_axes[0])
            artist = ax.pcolormesh(
                plane[plot_axes[0]],
                plane[plot_axes[1]],
                plotted_values,
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                shading="nearest",
            )
            if "blanking" in plane:
                solid = plane.blanking.transpose(plot_axes[1], plot_axes[0])
                if np.any(solid.values != 0) and np.any(solid.values == 0):
                    ax.contour(
                        plane[plot_axes[0]],
                        plane[plot_axes[1]],
                        solid,
                        levels=[0.5],
                        colors=["black"],
                        linewidths=0.8,
                    )
            for probe in probes:
                if np.isclose(probe["actual"][axis], metadata["actual"]):
                    ax.scatter(
                        probe["actual"][plot_axes[0]],
                        probe["actual"][plot_axes[1]],
                        c=probe["color"],
                        edgecolors="black",
                    )
                    ax.annotate(
                        probe["label"],
                        (probe["actual"][plot_axes[0]], probe["actual"][plot_axes[1]]),
                        xytext=(6, 6),
                        textcoords="offset points",
                        color="#ffffff",
                        fontsize=12 * text_scale,
                        fontweight="bold",
                        bbox={
                            "facecolor": "#071c29",
                            "alpha": 0.8,
                            "edgecolor": "none",
                            "pad": 2,
                        },
                    )
            if axis == "z":
                for section in slice_metadata:
                    if section["axis"] in ("x", "y"):
                        draw_line = ax.axvline if section["axis"] == "x" else ax.axhline
                        draw_line(
                            section["actual"],
                            color="#00c9df",
                            linewidth=1.2,
                            linestyle="--",
                        )
            ax.set(
                xlabel=f"{plot_axes[0]} [m]",
                ylabel=f"{plot_axes[1]} [m]",
                title=f"{variable} · {axis}={metadata['actual']:g} m · t={time:g} s",
            )
            ax.set_aspect("equal")
            ax.xaxis.label.set_color("#adc6cd")
            ax.yaxis.label.set_color("#adc6cd")
            ax.title.set_color("#eef4ee")
            ax.xaxis.label.set_fontsize(11 * text_scale)
            ax.yaxis.label.set_fontsize(11 * text_scale)
            ax.title.set_fontsize(12 * text_scale)
            colorbar = fig.colorbar(artist, ax=ax, label=f"{variable} [m/s]")
            colorbar.ax.tick_params(colors="#adc6cd", labelsize=10 * text_scale)
            colorbar.ax.yaxis.label.set_color("#adc6cd")
            colorbar.ax.yaxis.label.set_fontsize(11 * text_scale)
            colorbar.outline.set_edgecolor("#31505d")
            frame_path = frame_dir / f"{frame_number:05d}.png"
            fig.savefig(frame_path)
            fig.clear()
            snapshots.append(
                {"simulation_time": time, "path": str(frame_path.relative_to(output))}
            )
            for _ in range(repeat):
                target = frame_dir / f"{frame_number:05d}.png"
                if target != frame_path:
                    shutil.copyfile(frame_path, target)
                mapping.append(
                    {"video_time": frame_number / opts.fps, "simulation_time": time}
                )
                frame_number += 1
        poster = f"previews/{view_id}.png"
        shutil.copyfile(frame_dir / "00000.png", output / poster)
        movie = None
        if opts.movie and len(times) > 1:
            try:
                if encode_movie(
                    frame_dir, output / "media" / f"{view_id}.mp4", opts.fps
                ):
                    movie = f"media/{view_id}.mp4"
            except (subprocess.SubprocessError, OSError) as exc:
                warnings.add(f"Movie encoding failed for {view_id}: {exc}")
        views.append(
            {
                "id": view_id,
                "label": f"{'Horizontal slice' if axis == 'z' else 'Side section'} · {axis}={metadata['actual']:g} m",
                "kind": "2d",
                "poster": poster,
                "media": movie,
                "mime_type": "video/mp4" if movie else "image/png",
                "width": opts.width,
                "height": view_height,
                "field": variable,
                "cmap": cmap,
                "units": "m/s",
                "color_limits": limits,
                "slice": metadata,
                "frames": mapping,
                "snapshots": snapshots,
                "duration": frame_number / opts.fps,
            }
        )
    if opts.render_3d:
        try:
            from .render_3d import render_3d

            speed_limits = (
                opts.color_limits
                if opts.variable == "speed" and opts.color_limits
                else speed_extrema
            )
            if not np.isfinite(speed_limits).all():
                speed_limits = [0.0, 1.0]
            elif speed_limits[0] == speed_limits[1]:
                speed_limits = [
                    speed_limits[0],
                    speed_limits[1] + max(abs(speed_limits[1]) * 0.01, 1e-9),
                ]
            views.append(render_3d(reader, output, times, opts, speed_limits, warnings))
        except (
            ImportError,
            RuntimeError,
            ValueError,
            OSError,
            subprocess.SubprocessError,
        ) as exc:
            shutil.rmtree(output / "media" / "flow-3d", ignore_errors=True)
            for path in (
                output / "media" / "flow-3d.mp4",
                output / "previews" / "flow-3d.png",
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            warnings.add(f"Optional 3D rendering unavailable: {exc}")
    if probes:
        fig = Figure(
            figsize=(opts.width / 100, opts.height / 100), dpi=100, layout="constrained"
        )
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        for probe in probes:
            ax.plot(
                probe_times,
                [np.nan if value is None else value for value in probe["values"]],
                label=probe["id"],
                color=probe["color"],
            )
        ax.set(
            xlabel="Simulation time [s]",
            ylabel=f"{opts.variable} [m/s]",
            title="Virtual probes (saved solver samples)",
        )
        ax.legend()
        fig.savefig(output / "previews" / "probes.png")
        fig.clear()
    if fingerprint(reader.path) != reader.sources[0]["sha256"]:
        raise ValueError("state.nc changed during rendering; bundle not published")
    backend = reader.backend
    provenance = {
        "renderer_version": 2,
        "viewer_version": 2,
        "source_root": str(reader.root),
        "sources": reader.sources,
        "options": effective_options,
    }
    template_hashes = {}
    for name in ("index.html", "viewer.css", "viewer.js", "probe_charts.js"):
        asset_text = files("visualization.web").joinpath(name).read_text()
        (output / name).write_text(asset_text)
        template_hashes[name] = hashlib.sha256(asset_text.encode()).hexdigest()
    provenance["template_hashes"] = template_hashes
    provenance["geometry"] = next(
        (view.get("geometry") for view in views if view.get("geometry")), None
    )
    cache_key = hashlib.sha256(
        json.dumps(provenance, sort_keys=True).encode()
    ).hexdigest()
    provenance["cache_key"] = cache_key
    _write_json(output / "provenance.json", provenance)
    manifest = {
        "version": 1,
        "status": "complete",
        "cache_key": cache_key,
        "run_id": reader.root.name,
        "backend": backend,
        "case": reader.case,
        "prediction_type": (
            "surrogate prediction" if "surrogate" in str(backend) else "simulation"
        ),
        "selection": {"member": opts.member, "reduction": opts.reduction},
        "domain": domain,
        "time_range": [times[0], times[-1]],
        "sources": reader.sources,
        "processing": {
            "collocation": "linear interpolation at physical centre coordinates; no boundary extrapolation",
            "mask": "blanking != 0 indicates solid; no velocity-derived mask",
            "temporal": "saved snapshots held until next output; no temporal interpolation",
            "ensemble": opts.reduction,
            "color_limits": "fixed across clip; all selected slice fluid values unless explicitly configured",
        },
        "warnings": sorted(warnings),
        "views": views,
        "probes": "probes.json",
        "probe_csv": "probes.csv",
        "provenance": "provenance.json",
    }
    _write_json(output / "viewer_manifest.json.tmp", manifest)
    (output / "viewer_manifest.json.tmp").replace(output / "viewer_manifest.json")
    return manifest

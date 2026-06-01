#!/usr/bin/env python
"""Merge a raw (buildings + ground) STL pair into one domain-frame STL.

This is a **standalone utility**, deliberately kept out of the Hydra workflow in
``scripts/``. Run it by hand whenever you receive a new geometry export in the
"two STL" format (a buildings mesh + an extruded ground/terrain mesh, in
real-world metre coordinates) to produce a single STL that the forward models
can consume.

Why it is needed
----------------
uDALES' Python preprocessing (``libs/pyudales/.../python_udgeom/write_inputs.py``)
loads exactly **one** ``stl_file`` and runs it *directly* against the simulation
grid, which spans ``0 -> xlen / ylen / zsize`` with the floor at ``z = 0`` and no
translation or scaling applied. Every triangle becomes a uDALES facet
(``nfcts = TR.faces.shape[0]``). pylbm and pypalm follow the same
single-STL-in-domain-frame contract. Raw exports violate all of this: two files,
real-world origin (e.g. ``(-473, -511)``), kilometre-scale footprints, millions
of triangles, and a buildings mesh with inconsistent winding / many open bodies.

What this tool does
-------------------
1. Loads the buildings and ground meshes (binary or ASCII STL).
2. Crops both to a square/rectangular window (region of interest).
3. Repairs the buildings mesh (merge vertices, drop degenerate/duplicate faces,
   make winding/normals consistent).
4. Merges cropped buildings + ground into one mesh.
5. Translates it into the domain frame: ``(xmin, ymin) -> (0, 0)`` and the
   chosen vertical datum -> ``z = 0`` (so all solid geometry is at ``z >= 0``).
6. Writes **one** binary STL, shared by all three backends, at
   ``examples/<case>/<output-name>``. pylbm/pypalm reference it directly through
   their ``stl_path`` config; uDALES needs the STL inside its case dir, so a
   relative symlink ``examples/udales/<case>/<output-name>`` is created pointing
   at the shared file (uDALES dereferences it into the run dir when it copies
   the case dir, so the shared file is never mutated by the in-run STL shift).

It then prints the domain extents you should set in the uDALES ``namoptions``
(``xlen``/``ylen``/``zsize``) and a reminder to regenerate the ``&WALLS`` facet
counts. This tool only produces the STL; it does not edit namoptions.

Example
-------
    pixi run python tools/prepare_case_stl.py buildings.stl groundpatched_extrude.stl
    # smaller, cheaper window centred on a point:
    pixi run python tools/prepare_case_stl.py buildings.stl ground.stl \
        --size 250 --center -39 -53 --output-name buildings.stl
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import trimesh
from trimesh import repair


def _load(path: pathlib.Path) -> trimesh.Trimesh:
    print(f"  loading {path} ...", flush=True)
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
        raise ValueError(f"{path} did not load as a non-empty triangle mesh")
    print(f"    faces={len(mesh.faces):,}  bounds={np.round(mesh.bounds, 1).tolist()}")
    return mesh


def _crop_xy(mesh: trimesh.Trimesh, lo: np.ndarray, hi: np.ndarray) -> trimesh.Trimesh:
    """Keep faces whose centroid lies in the [lo, hi] xy window.

    Centroid filtering leaves the cut edges ragged, which is fine here: the
    forward models classify solid cells with a *vertical* odd/even ray test, so
    only the top/bottom surfaces matter, not closed side walls at the crop
    boundary.
    """
    centroids = mesh.triangles.mean(axis=1)
    mask = (
        (centroids[:, 0] >= lo[0])
        & (centroids[:, 0] <= hi[0])
        & (centroids[:, 1] >= lo[1])
        & (centroids[:, 1] <= hi[1])
    )
    out = mesh.copy()
    out.update_faces(mask)
    out.remove_unreferenced_vertices()
    return out


def _repair_buildings(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean a messy buildings mesh so facet normals are usable."""
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    # fix_normals makes per-component winding consistent and outward-facing.
    try:
        repair.fix_winding(m)
        repair.fix_normals(m)
    except Exception as exc:  # noqa: BLE001 - never let cosmetic repair abort the run
        print(f"    (warning) normal repair skipped: {exc}")
    return m


def _maybe_decimate(mesh: trimesh.Trimesh, max_faces: int | None) -> trimesh.Trimesh:
    if max_faces is None or len(mesh.faces) <= max_faces:
        return mesh
    print(f"  decimating {len(mesh.faces):,} -> ~{max_faces:,} faces ...", flush=True)
    try:
        return mesh.simplify_quadric_decimation(max_faces)
    except Exception as exc:  # noqa: BLE001
        print(
            f"    (warning) decimation unavailable ({exc}); keeping full resolution. "
            "Install fast-simplification/open3d or reduce --size instead."
        )
        return mesh


def prepare(
    buildings_path: pathlib.Path,
    ground_path: pathlib.Path,
    *,
    center: tuple[float, float] | None,
    size: tuple[float, float] | None,
    margin: float,
    z_datum: str,
    max_faces: int | None,
    repair_buildings: bool,
) -> trimesh.Trimesh:
    print("Loading meshes:")
    buildings = _load(buildings_path)
    ground = _load(ground_path)

    # --- choose the crop window ------------------------------------------------
    # Default centre is the buildings' bounding-box centre, NOT ``.centroid``:
    # ``.centroid`` is the area/volume-weighted centre of mass, which on the real
    # (messy, denser-on-one-side) buildings mesh sits tens of metres off the
    # geometric centre. Combined with the full-footprint default size below, an
    # off-centre window slides past the far edges and clips whole buildings.
    bcx, bcy = buildings.bounds.mean(axis=0)[:2]
    cx, cy = (bcx, bcy) if center is None else center
    if size is None:
        # Default: the full buildings footprint (+ margin), so every building is
        # kept and the ground is cropped to match.
        half = buildings.extents[:2] / 2.0 + margin
    else:
        half = np.array([size[0] / 2.0, size[1] / 2.0]) + margin
    lo = np.array([cx - half[0], cy - half[1]])
    hi = np.array([cx + half[0], cy + half[1]])
    print(
        f"Crop window: center=({cx:.1f}, {cy:.1f})  "
        f"x=[{lo[0]:.1f}, {hi[0]:.1f}]  y=[{lo[1]:.1f}, {hi[1]:.1f}]"
    )

    # --- crop, repair, merge ---------------------------------------------------
    print("Cropping buildings ...", flush=True)
    buildings = _crop_xy(buildings, lo, hi)
    print(f"  -> {len(buildings.faces):,} faces")
    if repair_buildings:
        print("Repairing buildings mesh ...", flush=True)
        buildings = _repair_buildings(buildings)
        print(f"  -> {len(buildings.faces):,} faces (watertight={buildings.is_watertight})")

    print("Cropping ground ...", flush=True)
    ground = _crop_xy(ground, lo, hi)
    print(f"  -> {len(ground.faces):,} faces")

    if len(buildings.faces) == 0:
        raise ValueError("No building faces left after cropping - check --center/--size.")
    if len(ground.faces) == 0:
        raise ValueError("No ground faces left after cropping - check --center/--size.")

    merged = trimesh.util.concatenate([buildings, ground])
    merged = _maybe_decimate(merged, max_faces)

    # --- move into the domain frame -------------------------------------------
    # Translate by the merged mesh's actual min bounds (not the window corner):
    # centroid-cropping leaves triangles whose vertices overhang the window, so
    # the true minimum is what must map to 0 to keep every vertex >= 0.
    x0, y0 = merged.bounds[0, 0], merged.bounds[0, 1]
    if z_datum == "min":
        z0 = merged.bounds[0, 2]
    elif z_datum == "ground":
        z0 = float(ground.triangles.mean(axis=1)[:, 2].mean())
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"unknown z_datum {z_datum!r}")
    merged.apply_translation([-x0, -y0, -z0])
    # Guard against any solid dipping below the floor (always true for 'ground',
    # possible for terrain roughness): push the lowest point exactly onto z=0.
    if merged.bounds[0, 2] < 0:
        merged.apply_translation([0.0, 0.0, -merged.bounds[0, 2]])

    return merged


def write_outputs(
    mesh: trimesh.Trimesh,
    *,
    examples_root: pathlib.Path,
    case: str,
    output_name: str,
    dry_run: bool,
) -> list[pathlib.Path]:
    """Write one shared STL and a relative uDALES symlink to it.

    Returns ``[shared_file, udales_symlink]``. pylbm/pypalm point their
    ``stl_path`` at ``shared_file``; the symlink lets uDALES find the STL inside
    its case dir without a second copy.
    """
    shared = examples_root / case / output_name
    udales_link = examples_root / "udales" / case / output_name
    rel_target = os.path.relpath(shared, udales_link.parent)

    if dry_run:
        print(f"[dry-run] would write {shared}")
        print(f"[dry-run] would symlink {udales_link} -> {rel_target}")
        return [shared, udales_link]

    shared.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(shared)  # .stl extension -> binary STL
    print(f"wrote {shared}  ({shared.stat().st_size / 1e6:.1f} MB)")

    udales_link.parent.mkdir(parents=True, exist_ok=True)
    if udales_link.is_symlink() or udales_link.exists():
        udales_link.unlink()
    udales_link.symlink_to(rel_target)
    print(f"linked {udales_link} -> {rel_target}")
    return [shared, udales_link]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Merge a (buildings, ground) STL pair into one domain-frame STL "
        "for the urban-airflow forward models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("buildings", type=pathlib.Path, help="Buildings STL (obstacles).")
    p.add_argument("ground", type=pathlib.Path, help="Ground/terrain (extruded) STL.")
    p.add_argument("--case", default="barcelona", help="Case name -> examples/<backend>/<case>/.")
    p.add_argument("--output-name", default="buildings.stl", help="Output STL filename.")
    p.add_argument(
        "--examples-root",
        type=pathlib.Path,
        default=pathlib.Path("examples"),
        help="Root of the per-backend example folders.",
    )
    p.add_argument(
        "--center",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Crop-window centre in input coords (default: buildings centroid).",
    )
    p.add_argument(
        "--size",
        type=float,
        nargs="+",
        metavar="L",
        default=None,
        help="Crop-window size in metres: one value (square) or two (Lx Ly). "
        "Default: full buildings footprint.",
    )
    p.add_argument("--margin", type=float, default=0.0, help="Extra metres added around the window.")
    p.add_argument(
        "--z-datum",
        choices=("min", "ground"),
        default="min",
        help="Which level maps to z=0: lowest solid point ('min') or mean ground level.",
    )
    p.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="Optional triangle budget; decimates the merged mesh if exceeded.",
    )
    p.add_argument("--no-repair", action="store_true", help="Skip buildings-mesh repair.")
    p.add_argument("--dry-run", action="store_true", help="Report only; do not write files.")
    args = p.parse_args(argv)

    for f in (args.buildings, args.ground):
        if not f.exists():
            p.error(f"input not found: {f}")

    size: tuple[float, float] | None
    if args.size is None:
        size = None
    elif len(args.size) == 1:
        size = (args.size[0], args.size[0])
    elif len(args.size) == 2:
        size = (args.size[0], args.size[1])
    else:
        p.error("--size takes one (square) or two (Lx Ly) values")

    center = tuple(args.center) if args.center is not None else None

    mesh = prepare(
        args.buildings,
        args.ground,
        center=center,
        size=size,
        margin=args.margin,
        z_datum=args.z_datum,
        max_faces=args.max_faces,
        repair_buildings=not args.no_repair,
    )

    lo = mesh.bounds[0]
    hi = mesh.bounds[1]
    print("\nFinal merged mesh (domain frame):")
    print(f"  faces        = {len(mesh.faces):,}")
    print(f"  bounds min   = {np.round(lo, 2).tolist()}")
    print(f"  bounds max   = {np.round(hi, 2).tolist()}")
    print("\nSuggested uDALES namoptions domain (round up zsize above building height):")
    print(f"  xlen = {hi[0]:.1f}    ylen = {hi[1]:.1f}    zsize >= {hi[2]:.1f}")
    print(f"  (set itot/jtot/ktot for your target dx; then regenerate the &WALLS facet counts)\n")

    write_outputs(
        mesh,
        examples_root=args.examples_root,
        case=args.case,
        output_name=args.output_name,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
4. Flattens the ground to a flat plane at its mean level (``--flatten-ground``,
   on by default) so it lands at exactly ``z = 0`` after seating -- one shared
   STL then works in every backend (open streets in PALM, a strippable z=0
   sheet for pylbm, a standard flat floor for uDALES). ``--no-flatten-ground``
   keeps the raw terrain.
5. Merges cropped buildings + ground into one mesh.
6. Translates it into the domain frame: ``(xmin, ymin) -> (0, 0)`` and the
   chosen vertical datum -> ``z = 0`` (so all solid geometry is at ``z >= 0``),
   clipping any sub-floor geometry (building basements) flush to the floor.
7. Writes **one** binary STL, shared by all backends, at
   ``examples/<case>/<output-name>``. All backends reference this single file
   through their ``stl_path`` config; nothing is written under the per-backend
   case dirs.

It then prints the domain extents you should set in the uDALES ``namoptions``
(``xlen``/``ylen``/``zsize``) and a reminder to regenerate the ``&WALLS`` facet
counts. This tool only produces the STL; it does not edit namoptions.

Example
-------
    pixi run python tools/prepare_case_stl.py buildings.stl groundpatched_extrude.stl
    # smaller, cheaper window centred on a point:
    pixi run python tools/prepare_case_stl.py buildings.stl ground.stl \
        --size 250 --center -39 -53 --output-name buildings.stl
    # rotate the buildings 30 deg relative to the (unrotated) ground:
    pixi run python tools/prepare_case_stl.py buildings.stl ground.stl \
        --rotate-buildings 30
"""

from __future__ import annotations

import argparse
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


def _rotate_about_center(mesh: trimesh.Trimesh, deg: float) -> trimesh.Trimesh:
    """Yaw ``mesh`` by ``deg`` degrees about its own xy bounding-box centre.

    The rotation axis is +z, so heights are untouched and the z-datum logic
    stays valid. Each mesh is rotated about its own centre *before* the merge,
    so the relative yaw between buildings and ground is exactly the difference
    of the two angles. Because both meshes are cropped to the *same* xy window
    first, their bbox centres nearly coincide, so a shared angle keeps them
    registered while differing angles spin one relative to the other in place.
    """
    if deg % 360.0 == 0.0:
        return mesh
    pivot = mesh.bounds.mean(axis=0)
    pivot[2] = 0.0
    R = trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1], pivot)
    out = mesh.copy()
    out.apply_transform(R)
    return out


def _seat_floor(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Seat geometry onto the ``z = 0`` floor by CLIPPING, not by lifting.

    After the vertical datum is applied, building basements and below-datum
    terrain dips sit at ``z < 0``. The old behaviour lifted the *entire* mesh so
    the lowest point reached ``z = 0`` -- which dragged the near-zero ground
    sheet up into an elevated plateau and made PALM's top-down height map read
    "one big building" with no open streets (see
    docs/palm_ground_topography_issue.md).

    Instead we drop faces lying entirely below the floor (basements, deep dips)
    and clamp the remaining straddling vertices up to ``z = 0``. The forward
    models classify solid cells with a vertical ray test and put their own floor
    at ``z = 0``, so the ragged clamped underside is harmless. When nothing sits
    below the floor (e.g. ``z_datum='min'``) this is a no-op.
    """
    face_max_z = mesh.vertices[mesh.faces][:, :, 2].max(axis=1)
    out = mesh.copy()
    out.update_faces(face_max_z >= 0.0)
    out.remove_unreferenced_vertices()
    verts = out.vertices.copy()
    verts[:, 2] = np.maximum(verts[:, 2], 0.0)
    out.vertices = verts
    # Clamping flattens sub-floor triangles onto z=0; drop the resulting slivers.
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    return out


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
    rotate_buildings: float = 0.0,
    rotate_ground: float = 0.0,
    flatten_ground: bool = True,
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

    # --- rotate each mesh independently, BEFORE the merge ----------------------
    # Yaw is applied per-mesh about its own xy centre so the relative rotation
    # between buildings and ground is exactly (rotate_buildings - rotate_ground).
    # A shared angle keeps them registered (their cropped centres ~coincide); a
    # difference spins the buildings relative to the ground (e.g. to align blocks
    # to the wind/grid). Rotation is about +z, so heights / the z datum are
    # untouched. The min-bounds translation below re-seats the rotated footprint
    # to the (0, 0) corner.
    if rotate_buildings % 360.0 != 0.0:
        print(f"Rotating buildings by {rotate_buildings:.1f} deg ...", flush=True)
        buildings = _rotate_about_center(buildings, rotate_buildings)
    if rotate_ground % 360.0 != 0.0:
        print(f"Rotating ground by {rotate_ground:.1f} deg ...", flush=True)
        ground = _rotate_about_center(ground, rotate_ground)

    ground_mean = float(ground.triangles.mean(axis=1)[:, 2].mean())

    # --- flatten the ground ----------------------------------------------------
    # Collapse the ground to a flat plane at its mean level. After the datum
    # translation below this lands at exactly z=0, which serves all three
    # backends from one STL:
    #   * PALM   -- street cells read 0 -> a fully open street grid;
    #   * pylbm  -- its ground filter strips faces whose vertices are *exactly*
    #               z==0, so a flat z=0 sheet is removed reliably. Real terrain
    #               (z != 0) is NOT caught and gets mis-read as one giant
    #               building, especially once buildings/ground are rotated by
    #               different angles (the ground then spans <90% of one axis and
    #               also escapes pylbm's coverage filter);
    #   * uDALES -- a flat z=0 floor is the standard case.
    # Buildings keep their real heights; their deep basements (well below the
    # mean) are clipped flush to z=0 by _seat_floor, so they sit on the floor
    # rather than floating.
    #
    # TODO(ntm): reintroduce real terrain relief later. Doing it correctly needs
    # per-building re-seating (subtract the local ground height from each
    # building) so blocks don't float/clip on slopes, plus a pylbm ground filter
    # that no longer assumes a flat z==0 sheet. Use --no-flatten-ground to keep
    # the raw terrain in the meantime.
    if flatten_ground:
        print(f"Flattening ground to a flat plane (z={ground_mean:.2f}) ...", flush=True)
        gv = ground.vertices.copy()
        gv[:, 2] = ground_mean
        ground.vertices = gv

    merged = trimesh.util.concatenate([buildings, ground])
    merged = _maybe_decimate(merged, max_faces)

    # --- vertical datum --------------------------------------------------------
    # 'ground' seats the mean ground level at z=0 so the ground sheet sits on the
    # floor and PALM's top-down height map reads open streets. 'min' (legacy)
    # seats the lowest point -- on meshes with building basements this lifts the
    # ground into an elevated plateau (the PALM "one big building" bug), so it is
    # kept only for back-compat. See docs/palm_ground_topography_issue.md.
    if z_datum == "min":
        z0 = merged.bounds[0, 2]
    elif z_datum == "ground":
        z0 = ground_mean
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"unknown z_datum {z_datum!r}")
    merged.apply_translation([0.0, 0.0, -z0])

    # Seat onto z=0 by clipping sub-floor geometry (basements, terrain dips)
    # rather than lifting everything -- lifting is what re-creates the plateau.
    merged = _seat_floor(merged)

    # --- move into the domain frame -------------------------------------------
    # Map the (rotated, seated) min corner to (0, 0, 0). Translate by the actual
    # min bounds (not the window corner): centroid cropping leaves triangles
    # overhanging the window, so the true minimum is what must map to 0 to keep
    # every vertex >= 0. The z shift here is always downward by the small gap
    # between the floor and the lowest surviving surface -- _seat_floor has
    # already dropped sub-floor basements, so this can never lift the ground.
    merged.apply_translation(-merged.bounds[0])

    return merged


def write_outputs(
    mesh: trimesh.Trimesh,
    *,
    examples_root: pathlib.Path,
    case: str,
    output_name: str,
    dry_run: bool,
) -> list[pathlib.Path]:
    """Write the single shared domain-frame STL at ``examples/<case>/<name>``.

    Returns ``[shared_file]``. All backends reference this one file via their
    ``stl_path`` config; nothing is written under the per-backend case dirs.
    """
    shared = examples_root / case / output_name

    if dry_run:
        print(f"[dry-run] would write {shared}")
        return [shared]

    shared.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(shared)  # .stl extension -> binary STL
    print(f"wrote {shared}  ({shared.stat().st_size / 1e6:.1f} MB)")
    return [shared]


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
        "--rotate",
        type=float,
        default=0.0,
        help="Shared yaw (deg, counter-clockwise) applied to BOTH meshes about the "
        "vertical axis before seating them in the domain frame. Used as the default "
        "for --rotate-buildings / --rotate-ground when those are not given.",
    )
    p.add_argument(
        "--rotate-buildings",
        type=float,
        default=None,
        help="Yaw the buildings mesh by this many degrees about its own xy centre, "
        "before the merge (overrides --rotate for the buildings).",
    )
    p.add_argument(
        "--rotate-ground",
        type=float,
        default=None,
        help="Yaw the ground mesh by this many degrees about its own xy centre, before "
        "the merge (overrides --rotate for the ground). The relative building/ground "
        "yaw is exactly (--rotate-buildings minus --rotate-ground).",
    )
    p.add_argument(
        "--z-datum",
        choices=("min", "ground"),
        default="ground",
        help="Which level maps to z=0: mean ground level ('ground', default -- keeps "
        "streets open in PALM) or the lowest solid point ('min', legacy -- lifts the "
        "ground into a plateau on basement-bearing meshes).",
    )
    p.add_argument(
        "--flatten-ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse the ground to a flat z=0 plane (default). Required for one "
        "shared STL to work in all backends: PALM gets open streets and pylbm can "
        "strip the ground. Use --no-flatten-ground to keep the raw terrain relief "
        "(terrain reintroduction is future work; see the TODO in prepare()).",
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

    # --rotate is the shared default; --rotate-buildings/-ground override per mesh.
    rotate_buildings = args.rotate if args.rotate_buildings is None else args.rotate_buildings
    rotate_ground = args.rotate if args.rotate_ground is None else args.rotate_ground

    mesh = prepare(
        args.buildings,
        args.ground,
        center=center,
        size=size,
        margin=args.margin,
        z_datum=args.z_datum,
        max_faces=args.max_faces,
        repair_buildings=not args.no_repair,
        rotate_buildings=rotate_buildings,
        rotate_ground=rotate_ground,
        flatten_ground=args.flatten_ground,
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

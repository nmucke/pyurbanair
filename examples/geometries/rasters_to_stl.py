#!/usr/bin/env python3
"""Convert UrbanTALES ``*_topo`` height rasters into domain-frame binary STLs.

Reads the raw ASCII building-height rasters downloaded by
``download_urbantales_geometries.py`` (under ``raw/{idealized,realistic}``) and
writes one binary ``.stl`` per case to ``processed/{idealized,realistic}``.

Each raster is a 2-D grid of building heights in metres at 1 m resolution
(0 = ground / street, positive = building height). It is a top-down 2.5-D
height map, so the mesh is built by extruding footprints straight up from
z = 0 -- exactly the shared single-STL contract every backend consumes
(``docs/plans/geometry_generation_plan.md`` Stage B):

  * geometry lives in the domain frame: it spans (0,0,0) -> (Lx, Ly, z_max) m;
  * a flat ground sheet sits at exactly z = 0 over the whole footprint;
  * footprints are 2.5-D extrusions (no overhangs); courtyards -- holes in the
    footprint -- fall out for free;
  * output is a binary STL with consistent (outward) winding.

Meshing: for each distinct building height the footprint mask is decomposed
into maximal axis-aligned rectangles (greedy meshing) and each rectangle is
extruded to a closed box from z = 0 to its height. By default the boxes are
boolean-unioned (via ``manifold3d``) into a single **watertight manifold** per
city: the union deletes the coincident interior walls between abutting
buildings, so ray-based ``trimesh.mesh.contains`` -- and hence
``neural_surrogates.geometry.stl_to_fluid_mask`` -- voxelises the geometry
exactly. Stepped / variable-height / courtyard footprints are represented
exactly. No separate ground sheet is added (an open sheet would spoil the
manifold; the floor is the z=0 boundary condition, and building undersides
already sit at z=0). Pass ``--multibody`` for the older fast box-soup mesh
(no union, like the repo's Barcelona STL) if you don't need exact contains.
For these height-map geometries the exact obstacle mask is also available
analytically, straight from the raster: cell (i, j, k) is solid iff
``raster[i, j] > z_k``.

Axis convention: raster axis 0 (rows) -> y, axis 1 (cols) -> x, so
Lx = ncols * dx and Ly = nrows * dx. The physically correct spacing is 1 m
(UrbanTALES resolution); ``--dx`` is exposed only if you deliberately want to
rescale the whole domain.

Usage:
    pixi run -e dev python examples/geometries/rasters_to_stl.py
    pixi run -e dev python examples/geometries/rasters_to_stl.py --set idealized
    pixi run -e dev python examples/geometries/rasters_to_stl.py --workers 8 --force

Requires numpy + trimesh (both root deps of the repo); run inside the ``dev``
Pixi env.
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
SETS = ("idealized", "realistic")


def greedy_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover a boolean mask with maximal ``(row, col, n_rows, n_cols)`` rects."""
    m = mask.copy()
    h_rows, w_cols = m.shape
    rects: list[tuple[int, int, int, int]] = []
    for i in range(h_rows):
        j = 0
        row = m[i]
        while j < w_cols:
            if not row[j]:
                j += 1
                continue
            w = 1
            while j + w < w_cols and m[i, j + w]:
                w += 1
            h = 1
            while i + h < h_rows and m[i + h, j : j + w].all():
                h += 1
            rects.append((i, j, h, w))
            m[i : i + h, j : j + w] = False
            j += w
    return rects


# Triangles of a unit box over its 8 corners (bottom 0-3 at z=0, top 4-7 at
# z=h), wound outward; ``fix_normals`` re-checks on export.
_BOX_FACES = np.array(
    [
        (0, 2, 1),
        (0, 3, 2),  # bottom  (-z)
        (4, 5, 6),
        (4, 6, 7),  # top     (+z)
        (0, 1, 5),
        (0, 5, 4),  # -y
        (3, 7, 6),
        (3, 6, 2),  # +y
        (0, 4, 7),
        (0, 7, 3),  # -x
        (1, 2, 6),
        (1, 6, 5),  # +x
    ],
    dtype=np.int64,
)


def _building_rects(z: np.ndarray) -> list[tuple[float, int, int, int, int]]:
    """Greedy rectangles per height level: ``(height, row, col, n_rows, n_cols)``."""
    out: list[tuple[float, int, int, int, int]] = []
    for hv in np.unique(z[z > 0]):
        for i, j, dh, dw in greedy_rectangles(z == hv):
            out.append((float(hv), i, j, dh, dw))
    return out


def raster_to_mesh(
    z: np.ndarray, dx: float, watertight: bool = False
) -> trimesh.Trimesh:
    """Extrude a height raster into a domain-frame binary-STL mesh.

    ``z`` is a 2-D array of building heights in metres (0 = ground). Cell
    ``(i, j)`` occupies ``x in [j, j+1]*dx``, ``y in [i, i+1]*dx``. Each maximal
    same-height rectangle becomes a closed box from z=0 to its height; a flat
    ground sheet is added at z=0. With ``watertight=True`` the building boxes
    are boolean-unioned into a single manifold (needs a boolean engine).
    """
    z = z.astype(np.float64)
    nrows, ncols = z.shape
    lx, ly = ncols * dx, nrows * dx
    rects = _building_rects(z)

    if watertight:
        # Boolean-union the per-height boxes into a single closed manifold. The
        # union deletes the coincident interior walls that make the multi-body
        # mesh mis-voxelise under ray-based ``mesh.contains``, so the result is
        # exact for point-in-solid tests. No separate ground sheet is added: an
        # open sheet reintroduces ray ambiguity at its free edges, and the
        # building undersides already sit at z=0 (pylbm strips the sheet, PALM
        # ignores it, and the floor is imposed as a z=0 boundary condition).
        boxes = []
        for h, i, j, dh, dw in rects:
            b = trimesh.creation.box(extents=(dw * dx, dh * dx, h))
            b.apply_translation(((j + dw / 2) * dx, (i + dh / 2) * dx, h / 2))
            boxes.append(b)
        mesh = trimesh.boolean.union(boxes) if boxes else trimesh.Trimesh()
        mesh.fix_normals()
        return mesh

    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for h, i, j, dh, dw in rects:
        x0, x1 = j * dx, (j + dw) * dx
        y0, y1 = i * dx, (i + dh) * dx
        verts.append(
            np.array(
                [
                    (x0, y0, 0.0),
                    (x1, y0, 0.0),
                    (x1, y1, 0.0),
                    (x0, y1, 0.0),
                    (x0, y0, h),
                    (x1, y0, h),
                    (x1, y1, h),
                    (x0, y1, h),
                ],
                dtype=np.float64,
            )
        )
        faces.append(_BOX_FACES + offset)
        offset += 8

    # Flat ground sheet at z = 0 over the whole domain footprint (the repo's
    # convention: strippable in pylbm, a flat floor in uDALES).
    verts.append(
        np.array(
            [(0.0, 0.0, 0.0), (lx, 0.0, 0.0), (lx, ly, 0.0), (0.0, ly, 0.0)],
            dtype=np.float64,
        )
    )
    faces.append(np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int64) + offset)

    mesh = trimesh.Trimesh(
        vertices=np.concatenate(verts),
        faces=np.concatenate(faces),
        process=False,  # keep boxes as independent bodies (no vertex weld)
    )
    mesh.fix_normals()
    return mesh


def process_one(args: tuple[str, str, str, float, bool, bool]) -> dict:
    """Worker: convert one raster file. ``args`` is picklable for the pool."""
    src, dst, name, dx, force, watertight = args
    src_p, dst_p = Path(src), Path(dst)
    if dst_p.exists() and dst_p.stat().st_size > 0 and not force:
        return {"name": name, "status": "skip"}
    z = np.loadtxt(src_p, dtype=np.float64)
    if z.ndim != 2:
        return {"name": name, "status": "bad-shape"}
    mesh = raster_to_mesh(z, dx, watertight=watertight)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst_p, file_type="stl")  # binary STL
    nrows, ncols = z.shape
    return {
        "name": name,
        "status": "ok",
        "nx": ncols,
        "ny": nrows,
        "Lx_m": round(ncols * dx, 3),
        "Ly_m": round(nrows * dx, 3),
        "z_max_m": round(float(z.max()), 3),
        "lambda_p": round(float((z > 0).mean()), 4),
        "n_faces": int(len(mesh.faces)),
    }


def run_set(
    kind: str,
    root: Path,
    dx: float,
    workers: int,
    force: bool,
    limit: int | None,
    watertight: bool,
) -> None:
    raw_dir = root / "raw" / kind
    out_dir = root / "processed" / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    topo = sorted(raw_dir.glob("*_topo"))
    if limit:
        topo = topo[:limit]
    if not topo:
        print(
            f"[{kind}] no *_topo rasters in {raw_dir} -- run the downloader first",
            file=sys.stderr,
        )
        return
    print(f"[{kind}] {len(topo)} rasters -> {out_dir}", flush=True)

    jobs = []
    for src in topo:
        name = src.name[:-5] if src.name.endswith("_topo") else src.name
        jobs.append(
            (str(src), str(out_dir / f"{name}.stl"), name, dx, force, watertight)
        )

    rows: list[dict] = []
    counts: dict[str, int] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process_one, j): j[2] for j in jobs}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                r = {"name": name, "status": "error"}
                print(f"  ! {name}: {exc}", file=sys.stderr, flush=True)
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] == "ok":
                rows.append(r)
            done += 1
            if done % 25 == 0:
                print(f"  [{kind}] {done}/{len(jobs)} ...", flush=True)

    # Per-set manifest of the geometry stats -- handy for downstream indexing.
    if rows:
        rows.sort(key=lambda r: r["name"])
        cols = ["name", "nx", "ny", "Lx_m", "Ly_m", "z_max_m", "lambda_p", "n_faces"]
        with (out_dir / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in cols})
    print(f"[{kind}] done: {counts}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--set", choices=[*SETS, "all"], default="all")
    ap.add_argument(
        "--root",
        type=Path,
        default=HERE,
        help="geometries root holding raw/ and processed/ (default: script dir)",
    )
    ap.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="grid spacing in metres (default 1.0 = UrbanTALES resolution)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="overwrite existing STLs")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N cases (debugging)",
    )
    ap.add_argument(
        "--multibody",
        action="store_true",
        help="emit a fast multi-body box mesh instead of boolean-"
        "unioning into one watertight manifold (skips manifold3d)",
    )
    args = ap.parse_args()

    kinds = list(SETS) if args.set == "all" else [args.set]
    for kind in kinds:
        run_set(
            kind,
            args.root,
            args.dx,
            args.workers,
            args.force,
            args.limit,
            watertight=not args.multibody,
        )
    print("All requested rasters converted to STL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

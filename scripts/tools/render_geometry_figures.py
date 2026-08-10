#!/usr/bin/env python
"""Render building-geometry STLs as shaded 3D figures (paper-quality PNGs).

This is a **standalone utility**, deliberately kept out of the Hydra workflow
in ``scripts/``. It renders the STL meshes under ``examples/`` as 3/4-aerial
3D views — buildings coloured by height on a single-hue blue ramp, standing on
a light ground plane with a shadow-casting sun — and assembles the geometry
figures of the surrogate-model manuscript.

Environment
-----------
The pixi environments do **not** carry the rendering deps. Rendering is done
headlessly on the GPU via NVIDIA EGL (no X server needed); the script sets
``PYOPENGL_PLATFORM=egl`` itself. Build a throwaway venv with exactly these
pins (newer PyOpenGL breaks glGenTextures under EGL, numpy 2 breaks pyrender,
newer scipy requires numpy 2):

    python3 -m venv .renderenv
    .renderenv/bin/pip install pyrender==0.1.45 PyOpenGL==3.1.7 "numpy<2" \\
        scipy==1.13.1 trimesh matplotlib

Usage
-----
Render a single STL to a PNG (any camera/colour option can be overridden)::

    .renderenv/bin/python scripts/tools/render_geometry_figures.py stl \\
        examples/barcelona/buildings.stl /tmp/barcelona.png \\
        --azim 235 --elev 38 --ground-pad 0.05 --z-max p97

Regenerate the three manuscript geometry figures (the main-text Xie & Castro +
Barcelona pair, and the two appendix grids of randomly sampled pretraining
geometries)::

    .renderenv/bin/python scripts/tools/render_geometry_figures.py manuscript

The manuscript mode caches per-panel renders in ``--panel-dir`` (default
``.temp/geometry_panels``, scratch — delete to force re-rendering) and writes
``fig_geom_targets.png``, ``fig_geoms_idealized.png`` and
``fig_geoms_realistic.png`` into the manuscript ``figures/`` directory. The
appendix samples are drawn with a fixed ``--seed`` (default 7), stratified
over the UA/US/VA/VS families (idealized pool) and over distinct cities
(realistic pool), keeping only idealized cases whose tallest building stays
below the 32 m domain top (the pretraining-usable subset).

Notes on the rendering choices
------------------------------
- Colours are assigned **per face** from the face's highest vertex, so walls
  match their roof and blocks read as solid extrusions (per-vertex colouring
  produces a translucent, foggy look).
- ``--z-max`` sets the height that maps to the darkest colour: a number gives
  an absolute scale (shared across panels when passed the same value), the
  string ``pNN`` uses the NN-th percentile of face heights of that mesh (good
  for realistic districts with a few tall towers).
- Shadows use ``SHADOWS_DIRECTIONAL`` with an 8192 px shadow map; there is no
  offscreen MSAA, so frames are rendered at ``--supersample``x resolution and
  LANCZOS-downscaled.
"""
import argparse
import csv
import os
import random
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image

import pyrender
import pyrender.constants
from pyrender import RenderFlags

# Raise the shadow-map resolution before any renderer is created.
pyrender.constants.SHADOW_TEX_SZ = 8192

REPO = Path(__file__).resolve().parents[2]
IDEAL = REPO / "examples/geometries/processed/idealized"
REAL = REPO / "examples/geometries/processed/realistic"
FIGDIR = REPO / "manuscripts/surrogate_model_manuscript/figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "cm",
    }
)


# --------------------------------------------------------------------------
# Single-STL rendering
# --------------------------------------------------------------------------


def _height_colors(mesh, z_max=None, cmap_name="Blues", lo=0.35, hi=0.85):
    """Per-face colors: each face takes the color of its highest vertex, so the
    walls of a building match its roof and blocks read as solid extrusions."""
    z_face = mesh.vertices[mesh.faces][:, :, 2].max(axis=1)
    if isinstance(z_max, str) and z_max.startswith("p"):
        zm = max(np.percentile(z_face, float(z_max[1:])), 1e-6)
    else:
        zm = z_max if z_max is not None else max(z_face.max(), 1e-6)
    t = np.clip(z_face / zm, 0.0, 1.0)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(lo + (hi - lo) * t)
    return (rgba * 255).astype(np.uint8)


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = map(np.asarray, (eye, target, up))
    fwd = eye - target  # camera looks down -z
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(up, fwd)
    right = right / np.linalg.norm(right)
    true_up = np.cross(fwd, right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, true_up, fwd, eye
    return pose


def _fit_distance(corners, target, azim, elev, yfov, aspect, margin=1.06):
    """Smallest camera distance along the view ray that keeps all bbox corners
    inside the frustum."""
    xfov = 2 * np.arctan(np.tan(yfov / 2) * aspect)
    a, e = np.radians(azim), np.radians(elev)
    view_dir = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    dist = 1.0
    for _ in range(60):
        eye = target - view_dir * dist
        pose = _look_at(eye, target)
        R, t = pose[:3, :3], pose[:3, 3]
        pc = (corners - t) @ R  # world -> camera
        depth = -pc[:, 2]
        if (depth <= 0).any():
            dist *= 1.5
            continue
        need = max(
            (np.abs(pc[:, 0]) / (depth * np.tan(xfov / 2))).max(),
            (np.abs(pc[:, 1]) / (depth * np.tan(yfov / 2))).max(),
        )
        if abs(need - 1.0) < 1e-3:
            break
        dist *= need
    return dist * margin


def render_stl(
    stl_path,
    out_png,
    azim=225.0,
    elev=32.0,
    width=1600,
    height=1000,
    supersample=3,
    z_max=None,
    yfov_deg=32.0,
    ground_pad=0.18,
    sun_azim=250.0,
    sun_elev=48.0,
    cmap_name="Blues",
):
    """Render one STL to a trimmed, white-background PNG.

    ``azim``/``elev`` set the camera direction (degrees; azimuth is measured
    in the ground plane from +x), ``z_max`` the colour normalisation (see
    module docstring), ``ground_pad`` the ground-plane margin as a fraction of
    the larger horizontal extent.
    """
    mesh = trimesh.load(stl_path, force="mesh")
    mesh.visual.face_colors = _height_colors(mesh, z_max=z_max, cmap_name=cmap_name)

    lo, hi_b = mesh.bounds
    center = (lo + hi_b) / 2
    size = hi_b - lo

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.30] * 3)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    # Ground plane, slightly below z=0 to avoid z-fighting with building bases.
    pad = ground_pad * max(size[0], size[1])
    gx0, gx1 = lo[0] - pad, hi_b[0] + pad
    gy0, gy1 = lo[1] - pad, hi_b[1] + pad
    gz = -0.02 * max(size[2], 1.0) - 0.05
    ground = trimesh.Trimesh(
        vertices=[[gx0, gy0, gz], [gx1, gy0, gz], [gx1, gy1, gz], [gx0, gy1, gz]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    ground.visual.vertex_colors = np.tile(
        np.array([236, 238, 240, 255], dtype=np.uint8), (4, 1)
    )
    scene.add(pyrender.Mesh.from_trimesh(ground, smooth=False))

    # Sun (shadow-casting) + weak fill from the opposite side.
    def _dir_pose(az, el):
        a, e = np.radians(az), np.radians(el)
        d = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        return _look_at(center - d * max(size), center)

    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.4),
        pose=_dir_pose(sun_azim, sun_elev),
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=0.7),
        pose=_dir_pose(sun_azim + 160.0, 35.0),
    )

    # Camera: 3/4 aerial view fitted to the building bounds (not the ground).
    aspect = width / height
    yfov = np.radians(yfov_deg)
    target = center.copy()
    target[2] = lo[2] + 0.22 * size[2]
    corners = np.array(
        [
            [x, y, z]
            for x in (lo[0], hi_b[0])
            for y in (lo[1], hi_b[1])
            for z in (lo[2], hi_b[2])
        ]
    )
    dist = _fit_distance(corners, target, azim, elev, yfov, aspect)
    a, e = np.radians(azim), np.radians(elev)
    view_dir = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    cam_pose = _look_at(target - view_dir * dist, target)
    scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect), pose=cam_pose)

    W, H = width * supersample, height * supersample
    r = pyrender.OffscreenRenderer(W, H)
    try:
        color, _ = r.render(scene, flags=RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.RGBA)
    finally:
        r.delete()

    img = Image.fromarray(color).resize((width, height), Image.LANCZOS)
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])

    # Trim surrounding whitespace so panel assembly controls the margins.
    arr = np.asarray(bg)
    content = (arr < 250).any(axis=2)
    rows, cols = np.where(content.any(axis=1))[0], np.where(content.any(axis=0))[0]
    if rows.size and cols.size:
        pad_px = int(0.015 * max(bg.size))
        r0, r1 = max(rows[0] - pad_px, 0), min(rows[-1] + pad_px, arr.shape[0])
        c0, c1 = max(cols[0] - pad_px, 0), min(cols[-1] + pad_px, arr.shape[1])
        bg = bg.crop((c0, r0, c1, r1))
    bg.save(out_png, optimize=True)
    return out_png


# --------------------------------------------------------------------------
# Manuscript figures: sampling, panel rendering, grid assembly
# --------------------------------------------------------------------------


def load_manifest(pool_dir):
    with open(pool_dir / "manifest.csv") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def sample_cases(seed):
    """Stratified random draw: 2 cases per idealized family (UA/US/VA/VS,
    pretraining-usable only) and 8 realistic cases from distinct cities."""
    rng = random.Random(seed)
    man_i = load_manifest(IDEAL)
    # Only cases usable for pretraining: tallest building below the 32 m domain top.
    usable = [n for n, r in man_i.items() if float(r["z_max_m"]) < 32.0]
    fams = {"UA": [], "US": [], "VA": [], "VS": []}
    for n in usable:
        for f in fams:
            if n.startswith(f):
                fams[f].append(n)
    idealized = []
    for f in ("UA", "US", "VA", "VS"):
        idealized += rng.sample(sorted(fams[f]), 2)

    man_r = load_manifest(REAL)
    d00 = sorted(n for n in man_r if n.endswith("_d00"))
    by_city = {}
    for n in d00:
        city = "-".join(n.split("-")[:2])
        by_city.setdefault(city, []).append(n)
    cities = rng.sample(sorted(by_city), 8)
    realistic = [rng.choice(by_city[c]) for c in cities]
    return idealized, man_i, realistic, man_r


def _label_ideal(n, man):
    r = man[n]
    fam = n[:2]
    arr = "aligned" if fam[1] == "A" else "staggered"
    hts = "uniform" if fam[0] == "U" else "variable"
    ang = n.rsplit("_d", 1)[1]
    lam = float(r["lambda_p"])
    return f"{arr}, {hts} height, $\\lambda_p = {lam:.2f}$, ${int(ang)}^\\circ$"


def _label_real(n, man):
    r = man[n]
    lam = float(r["lambda_p"])
    zm = float(r["z_max_m"])
    return f"$\\lambda_p = {lam:.2f}$, $h_\\mathrm{{max}} = {zm:.0f}\\,$m"


def assemble_grid(panel_files, labels, out_png, ncols=2, panel_aspect=0.60):
    """Lay out pre-rendered panels in a grid with name + description labels."""
    n = len(panel_files)
    nrows = (n + ncols - 1) // ncols
    fig_w = 12.0
    cell_w = fig_w / ncols
    cell_h = cell_w * panel_aspect + 0.55
    fig_h = cell_h * nrows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=260)
    for i, (pf, (name, desc)) in enumerate(zip(panel_files, labels)):
        row, col = divmod(i, ncols)
        ax = fig.add_axes(
            [
                col / ncols + 0.012,
                1 - (row + 1) * cell_h / fig_h + 0.55 / fig_h,
                1 / ncols - 0.024,
                (cell_h - 0.62) / fig_h,
            ]
        )
        ax.imshow(mpimg.imread(pf))
        ax.set_axis_off()
        ax.set_anchor("S")
        ax.text(0.5, -0.045, name, transform=ax.transAxes, ha="center", va="top",
                fontsize=13, fontfamily="monospace")
        ax.text(0.5, -0.19, desc, transform=ax.transAxes, ha="center", va="top",
                fontsize=12, color="0.25")
    fig.savefig(out_png, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", out_png)


def make_manuscript_figures(seed, panel_dir, figdir):
    panel_dir.mkdir(parents=True, exist_ok=True)
    idealized, man_i, realistic, man_r = sample_cases(seed)
    print("idealized:", idealized)
    print("realistic:", realistic)

    # Appendix panels. Idealized cases share one colour scale; realistic
    # districts are normalised per panel (a few tall towers would otherwise
    # wash out the rest).
    z_shared = max(float(man_i[n]["z_max_m"]) for n in idealized) * 1.05
    for n in idealized:
        out = panel_dir / f"ideal_{n}.png"
        if not out.exists():
            render_stl(str(IDEAL / f"{n}.stl"), str(out), azim=230, elev=30,
                       width=1500, height=950, z_max=z_shared, ground_pad=0.10)
            print("rendered", n)
    for n in realistic:
        out = panel_dir / f"real_{n}.png"
        if not out.exists():
            render_stl(str(REAL / f"{n}.stl"), str(out), azim=230, elev=36,
                       width=1500, height=1000, z_max="p97", ground_pad=0.06)
            print("rendered", n)

    assemble_grid(
        [panel_dir / f"ideal_{n}.png" for n in idealized],
        [(n, _label_ideal(n, man_i)) for n in idealized],
        figdir / "fig_geoms_idealized.png",
        panel_aspect=0.55,
    )
    assemble_grid(
        [panel_dir / f"real_{n}.png" for n in realistic],
        [(n, _label_real(n, man_r)) for n in realistic],
        figdir / "fig_geoms_realistic.png",
        panel_aspect=0.62,
    )

    # Main-text figure: the two target configurations side by side.
    targets = [
        (REPO / "examples/xie_and_castro/xie_castro_2008_STL.stl",
         panel_dir / "target_xie.png",
         dict(azim=230, elev=30, width=1500, height=1150, ground_pad=0.14),
         "(a) Xie & Castro staggered array"),
        (REPO / "examples/barcelona/buildings.stl",
         panel_dir / "target_bcn.png",
         dict(azim=235, elev=38, width=1500, height=1150, ground_pad=0.05,
              z_max="p97"),
         "(b) Barcelona district"),
    ]
    for stl, out, kw, _ in targets:
        if not out.exists():
            render_stl(str(stl), str(out), **kw)
            print("rendered", out.name)
    fig = plt.figure(figsize=(12.0, 4.6), dpi=260)
    for i, (_, out, _, label) in enumerate(targets):
        ax = fig.add_axes([i * 0.5 + 0.012, 0.10, 0.5 - 0.024, 0.88])
        ax.imshow(mpimg.imread(out))
        ax.set_axis_off()
        ax.set_anchor("S")
        ax.text(0.5, -0.03, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=15)
    fig.savefig(figdir / "fig_geom_targets.png", facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", figdir / "fig_geom_targets.png")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_stl = sub.add_parser("stl", help="render a single STL to a PNG")
    p_stl.add_argument("stl", type=Path)
    p_stl.add_argument("out", type=Path)
    p_stl.add_argument("--azim", type=float, default=225.0)
    p_stl.add_argument("--elev", type=float, default=32.0)
    p_stl.add_argument("--width", type=int, default=1600)
    p_stl.add_argument("--height", type=int, default=1000)
    p_stl.add_argument("--supersample", type=int, default=3)
    p_stl.add_argument("--yfov-deg", type=float, default=32.0)
    p_stl.add_argument("--ground-pad", type=float, default=0.18)
    p_stl.add_argument("--sun-azim", type=float, default=250.0)
    p_stl.add_argument("--sun-elev", type=float, default=48.0)
    p_stl.add_argument("--cmap-name", default="Blues")
    p_stl.add_argument(
        "--z-max", default=None,
        help="height mapped to the darkest colour: a number (absolute scale), "
        "'pNN' (NN-th percentile of this mesh), or omit for the mesh maximum",
    )

    p_man = sub.add_parser(
        "manuscript", help="regenerate the three manuscript geometry figures"
    )
    p_man.add_argument("--seed", type=int, default=7,
                       help="RNG seed for the appendix samples (default: 7)")
    p_man.add_argument("--panel-dir", type=Path,
                       default=REPO / ".temp/geometry_panels",
                       help="per-panel render cache (delete to force re-render)")
    p_man.add_argument("--figdir", type=Path, default=FIGDIR,
                       help="output directory for the assembled figures")

    args = parser.parse_args()
    if args.command == "stl":
        z_max = args.z_max
        if z_max is not None and not str(z_max).startswith("p"):
            z_max = float(z_max)
        render_stl(
            str(args.stl), str(args.out), azim=args.azim, elev=args.elev,
            width=args.width, height=args.height, supersample=args.supersample,
            z_max=z_max, yfov_deg=args.yfov_deg, ground_pad=args.ground_pad,
            sun_azim=args.sun_azim, sun_elev=args.sun_elev,
            cmap_name=args.cmap_name,
        )
        print(args.out)
    else:
        make_manuscript_figures(args.seed, args.panel_dir, args.figdir)


if __name__ == "__main__":
    main()

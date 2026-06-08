"""Plot horizontal (z-constant) slices of a uDALES state.nc and animate one.

Loads lazily (xarray, one slice at a time), renders a grid of z-slices at one
time step, and builds a time animation of a single z level reading one frame
at a time so memory stays flat regardless of the number of time steps.

If the requested variable is ``vel_magnitude`` and it is not stored in the
file, it is computed on the fly as sqrt(u^2 + v^2 + w^2). Components live on
staggered grids; for visualisation we treat them as index-aligned (ignoring the
half-cell stagger), which is accurate enough for slice plots.

Usage:
    python scripts/plot_state_slices.py STATE_NC OUT_DIR [VAR] [Z_INDEX]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless compute node
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation

# Use the ffmpeg binary provided by the loaded system module, if any.
_ffmpeg = os.environ.get("FFMPEG_BIN")
if _ffmpeg:
    plt.rcParams["animation.ffmpeg_path"] = _ffmpeg

state_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
var_name = sys.argv[3] if len(sys.argv) > 3 else "vel_magnitude"
out_dir.mkdir(parents=True, exist_ok=True)

# Lazy open: the netCDF backend reads a slice only when we .values it, so we
# never hold the full array in RAM (no dask needed).
ds = xr.open_dataset(state_path)

tdim = "time"
compute_mag = var_name == "vel_magnitude" and "vel_magnitude" not in ds

if compute_mag:
    u, v, w = ds["u"], ds["v"], ds["w"]
    zdim = "zt"
    zmdim = next(d for d in w.dims if d.startswith("z"))
    # Horizontal coords taken from the cell-centre grid.
    xc = ds["xt"].values if "xt" in ds else ds[u.dims[-1]].values
    yc = ds["yt"].values if "yt" in ds else ds[u.dims[-2]].values
    nz = ds.sizes[zdim]
    label = "|U| = sqrt(u^2+v^2+w^2)  [m/s]"

    def volume(t):
        uu = u.isel({tdim: t}).values
        vv = v.isel({tdim: t}).values
        ww = w.isel({tdim: t}).values
        return np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)  # (z, y, x)

    def frame(t, k):
        uu = u.isel({tdim: t, zdim: k}).values
        vv = v.isel({tdim: t, zdim: k}).values
        ww = w.isel({tdim: t, zmdim: k}).values
        return np.sqrt(uu ** 2 + vv ** 2 + ww ** 2)  # (y, x)
else:
    if var_name not in ds:
        var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    zdim = next(d for d in da.dims if d.startswith("z"))
    xdim, ydim = [d for d in da.dims if d not in (zdim, tdim)][::-1]
    xc, yc = da[xdim].values, da[ydim].values
    nz = da.sizes[zdim]
    label = var_name

    def _orient(arr):
        # ensure (y, x) with x as the last axis
        return arr.T if arr.shape[0] == len(xc) else arr

    def volume(t):
        return da.isel({tdim: t}).values

    def frame(t, k):
        return _orient(da.isel({tdim: t, zdim: k}).values)

nt = ds.sizes[tdim]
zvals = ds[zdim].values
tvals = ds[tdim].values
print(f"Variable '{var_name}'  nt={nt} nz={nz}  computed={compute_mag}", flush=True)

# Shared colour scale from a single mid-time 3D field (one time step in RAM).
mid_t = nt // 2
vol_mid = volume(mid_t)  # (z, y, x)
vmin, vmax = float(np.nanmin(vol_mid)), float(np.nanmax(vol_mid))
extent = [float(xc.min()), float(xc.max()), float(yc.min()), float(yc.max())]


def orient_zk(vol, k):
    s = vol[k]
    return s.T if s.shape[0] == len(xc) else s


# --- 1) grid of z slices at the mid time step ------------------------------
z_idx = np.linspace(0, nz - 1, min(nz, 9), dtype=int)
ncol = 3
nrow = int(np.ceil(len(z_idx) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for ax, k in zip(axes.flat, z_idx):
    ax.axis("on")
    im = ax.imshow(
        orient_zk(vol_mid, k), origin="lower", extent=extent, aspect="auto",
        vmin=vmin, vmax=vmax, cmap="viridis",
    )
    ax.set_title(f"{zdim}={float(zvals[k]):.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
fig.suptitle(f"{label}  z-slices @ t={float(tvals[mid_t]):.1f} s")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=label)
slices_png = out_dir / f"{var_name}_zslices.png"
fig.savefig(slices_png, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {slices_png}", flush=True)

# --- 2) animation of a single z level over time ----------------------------
z_anim = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "" else nz // 2
figa, axa = plt.subplots(figsize=(7, 6))
ima = axa.imshow(
    frame(0, z_anim), origin="lower", extent=extent, aspect="auto",
    vmin=vmin, vmax=vmax, cmap="viridis",
)
axa.set_xlabel("x [m]")
axa.set_ylabel("y [m]")
figa.colorbar(ima, ax=axa, label=label)
title = axa.set_title("")


def update(t):
    ima.set_data(frame(t, z_anim))  # one frame in RAM
    title.set_text(
        f"{label.split(' ')[0]}  {zdim}={float(zvals[z_anim]):.2f} m  "
        f"t={float(tvals[t]):.1f} s"
    )
    return ima, title


# Use every time step (no subsampling); a lower fps makes each frame linger
# on screen so the animation plays back a little slower.
frames = range(nt)
anim = FuncAnimation(figa, update, frames=frames, blit=False)
mp4_path = out_dir / f"{var_name}_z{z_anim}_animation.mp4"
fps = int(sys.argv[5]) if len(sys.argv) > 5 else 15  # lower fps -> slower playback
writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=4000,
                      extra_args=["-pix_fmt", "yuv420p"])
anim.save(mp4_path, writer=writer, dpi=120)
plt.close(figa)
print(f"Wrote {mp4_path}  ({nt} frames @ {fps} fps -> {nt / fps:.0f}s)", flush=True)
ds.close()
print("Done.", flush=True)

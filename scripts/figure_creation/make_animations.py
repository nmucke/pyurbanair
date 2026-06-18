"""Talk animations (mp4) from ESMDA results (spec §4 A-anim, §5 B-anim, §6 anim_C).

Produces, under ``<out>/``:
  A_param_only/anim_A_<model>_<res>.mp4   -- per-model state-evolution triptych
                                             over the full Block-A horizon.
  B_state_estimation/anim_B_methods.mp4   -- 4-panel method comparison over the
                                             last 1-2 Block-B windows.
  C_full_joint/anim_C_full.mp4            -- full-joint icstate_red triptych over
                                             the full short horizon.

Each animation renders a |U| pedestrian-height slice (figcommon.PED_Z) on the
common truth grid. Triptychs show ``truth | posterior-mean estimate | error``;
the field panels share a viridis ``[0, vmax]`` scale and the error panel uses a
symmetric ``RdBu_r`` diverging scale. Window boundaries / the current window
index are annotated each frame.

These are heavy and need ffmpeg, which is NOT in the pixi env -- it comes from a
system module exposed via the ``FFMPEG_BIN`` env var (set by the SLURM script).
The module is import-clean and per-frame rendering works without ffmpeg; only the
final ``mp4`` write needs it (a missing ffmpeg prints a clear message).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from figspec import dataio, figcommon as FC, mask
from figspec import style as S

HORIZON_B_S = 300.0  # Block B overlapping horizon [s] (10 windows x 30 s)


# ---------------------------------------------------------------------------
# ffmpeg wiring (the binary comes from a system module, not the pixi env)
# ---------------------------------------------------------------------------
def configure_ffmpeg() -> str | None:
    """Point matplotlib at the system ffmpeg (``FFMPEG_BIN`` or ``PATH``)."""
    ff = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
    if ff:
        matplotlib.rcParams["animation.ffmpeg_path"] = ff
    return ff


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------
def finest_complete_by_model(runs) -> dict:
    """Finest (largest nx) complete run per model -> {model: Run}."""
    out: dict = {}
    for r in runs:
        if not r.complete:
            continue
        if r.model not in out or r.nx > out[r.model].nx:
            out[r.model] = r
    return out


def block_b_by_method(runs) -> dict:
    """{method: Run} for the complete Block-B state-estimation runs."""
    return {r.method: r for r in runs if r.method is not None and r.complete}


def param_only_baseline():
    """Finest complete uDALES Block-A run (param-only baseline), or None."""
    best = None
    for r in dataio.discover_block_a():
        if r.model != "udales" or not r.complete:
            continue
        if best is None or r.nx > best.nx:
            best = r
    return best


# ---------------------------------------------------------------------------
# Field building
# ---------------------------------------------------------------------------
def _slice_stack(run, zi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior-mean |U| of a run on the truth grid at the pedestrian level.

    Returns ``(model[time,y,x], truth[time,y,x], time[s])`` with the model and
    truth sharing the model's time axis (truth interpolated onto it).
    """
    sm = dataio.load_state_mean(run)
    if sm is None:
        raise FileNotFoundError(f"no posterior_state_mean.nc for {run.name}")
    model_f = dataio.interp_to_truth(dataio.velmag_field(sm))      # (t,z,y,x)
    truth_f = dataio.align_truth_time(dataio.truth_velmag(), model_f)
    model = np.asarray(model_f.isel(z=zi).values, dtype=float)     # (t,y,x)
    truth = np.asarray(truth_f.isel(z=zi).values, dtype=float)
    t = np.asarray(model_f["time"].values, dtype=float)
    sm.close()
    return model, truth, t


def _baseline_slice_on_axis(run, zi: int, time_axis: np.ndarray) -> np.ndarray:
    """param-only |U| slice (truth grid, pedestrian level) interpolated onto a
    given time axis -- used for the Block-B param-only panel."""
    sm = dataio.load_state_mean(run)
    if sm is None:
        raise FileNotFoundError(f"no posterior_state_mean.nc for {run.name}")
    field = dataio.interp_to_truth(dataio.velmag_field(sm))        # (t,z,y,x)
    field = field.interp(time=time_axis, kwargs={"fill_value": "extrapolate"})
    out = np.asarray(field.isel(z=zi).values, dtype=float)
    sm.close()
    return out


def _apply_mask(stack: np.ndarray, solid2d: np.ndarray | None) -> np.ndarray:
    """Set solid (building) cells to NaN across all frames of a (t,y,x) stack."""
    if solid2d is None:
        return stack
    if stack.shape[1:] != solid2d.shape:
        return stack
    out = np.array(stack, dtype=float)
    out[:, solid2d] = np.nan
    return out


def _subsample(n: int, max_frames: int) -> np.ndarray:
    """Evenly-spaced frame indices into ``range(n)`` (<= ``max_frames``)."""
    if n <= max_frames:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_frames).round().astype(int))


def _window_index(t: float, edges: np.ndarray | None) -> int | None:
    """Index of the assimilation window containing time ``t`` (or None)."""
    if edges is None:
        return None
    k = int(np.searchsorted(edges, t, side="right") - 1)
    return max(0, min(k, len(edges) - 2))


# ---------------------------------------------------------------------------
# Even-pixel sizing (yuv420p requires even width & height)
# ---------------------------------------------------------------------------
def _even(n: int) -> int:
    n = int(round(n))
    return n if n % 2 == 0 else n + 1


def _figsize_even(width_px: int, height_px: int, dpi: int) -> tuple[float, float]:
    """Figure (w, h) in inches whose ``size*dpi`` is even in both dimensions."""
    return _even(width_px) / dpi, _even(height_px) / dpi


# ---------------------------------------------------------------------------
# Heatmap panel helpers
# ---------------------------------------------------------------------------
def _hatch(ax, solid2d, extent) -> None:
    if solid2d is None:
        return
    xs = np.linspace(extent[0], extent[1], solid2d.shape[1])
    ys = np.linspace(extent[2], extent[3], solid2d.shape[0])
    ax.contourf(xs, ys, solid2d.astype(float), levels=[0.5, 1.5],
                colors="none", hatches=["////"])
    ax.contour(xs, ys, solid2d.astype(float), levels=[0.5], colors="0.3",
               linewidths=0.5)


def _imshow(ax, frame, extent, *, cmap, vmin, vmax):
    return ax.imshow(frame, origin="lower", extent=extent, aspect="equal",
                     cmap=cmap, vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Animation writing
# ---------------------------------------------------------------------------
def _write_animation(fig, update, frames, out_path: pathlib.Path, fps: int,
                     ff: str | None, *, frame_indices=None,
                     dump_frames: bool = False) -> bool:
    """Drive ``FuncAnimation`` and encode H.264/yuv420p; return success.

    Renders all panels once via ``update`` per frame. When ``dump_frames`` is
    set, also writes ``frames/frame_%05d.png`` (from 0) next to the mp4. If no
    ffmpeg is available, the mp4 step is skipped with a clear message (frames are
    still dumped when requested).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dump_frames:
        fdir = out_path.parent / "frames" / out_path.stem
        fdir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frame_indices):
            update(fr)
            fig.savefig(fdir / f"frame_{i:05d}.png", dpi=fig.dpi, transparent=False)
        print(f"  + {fdir}/frame_*.png ({len(frame_indices)} frames)")

    if not ff:
        print(f"  ! ffmpeg unavailable (set FFMPEG_BIN); SKIP mp4 {out_path.name}")
        plt.close(fig)
        return False

    anim = FuncAnimation(fig, update, frames=list(frame_indices), blit=False)
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out_path), writer=writer, dpi=fig.dpi)
    plt.close(fig)
    print(f"  + {out_path} ({len(frame_indices)} frames @ {fps} fps)")
    return True


# ---------------------------------------------------------------------------
# Triptych animation (A-anim, anim_C): truth | estimate | error
# ---------------------------------------------------------------------------
def build_triptych(truth: np.ndarray, model: np.ndarray, t: np.ndarray,
                   *, extent, solid2d, edges, title_prefix: str, dpi: int = 100):
    """Build a 3-panel ``[truth | estimate | error]`` figure + per-frame update.

    Returns ``(fig, update)``. Color scales: fields share ``[0, vmax]`` viridis;
    the error panel is symmetric ``RdBu_r``.
    """
    err = model - truth
    vmax = float(np.nanmax([np.nanmax(truth), np.nanmax(model)]))
    emax = float(np.nanmax(np.abs(err))) or 1.0

    # One panel is ~3.0 in wide; 3 panels + colorbars -> ~10 in wide, even px.
    fig_w, fig_h = _figsize_even(2000, 760, dpi)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    titles = ["truth", "posterior-mean estimate", "error (estimate - truth)"]
    ims = []
    for j, ax in enumerate(axes):
        cmap = S.CMAP_DIFF if j == 2 else S.CMAP_FIELD
        lo, hi = (-emax, emax) if j == 2 else (0.0, vmax)
        data = err[0] if j == 2 else (model[0] if j == 1 else truth[0])
        ims.append(_imshow(ax, data, extent, cmap=cmap, vmin=lo, vmax=hi))
        _hatch(ax, solid2d, extent)
        ax.set_title(titles[j], fontsize=10)
        ax.set_xlabel("x [m]")
        if j == 0:
            ax.set_ylabel("y [m]")
    fig.colorbar(ims[0], ax=[axes[0], axes[1]], shrink=0.85, label="|U| [m/s]")
    fig.colorbar(ims[2], ax=axes[2], shrink=0.85, label="error [m/s]")
    suptitle = fig.suptitle("", fontsize=12)

    def update(i: int):
        ims[0].set_data(truth[i])
        ims[1].set_data(model[i])
        ims[2].set_data(err[i])
        w = _window_index(t[i], edges)
        wtxt = "" if w is None else f"  |  window W{w}"
        suptitle.set_text(f"{title_prefix}    t = {t[i]:6.1f} s{wtxt}")
        return ims

    return fig, update


# ---------------------------------------------------------------------------
# 4-panel method animation (B-anim)
# ---------------------------------------------------------------------------
def build_methods_panel(panels, truth: np.ndarray, t: np.ndarray,
                        *, extent, solid2d, edges, title_prefix: str,
                        dpi: int = 100):
    """Build a 4-panel ``{truth, param-only, ic, icstate}`` figure + update.

    ``panels``: list of ``(label, stack[time,y,x])``; the first is conventionally
    the truth. All panels share a single viridis ``[0, vmax]`` scale.
    """
    stacks = [truth] + [s for _, s in panels[1:]]
    labels = [lbl for lbl, _ in panels]
    vmax = float(np.nanmax([np.nanmax(s) for s in stacks]))

    n = len(stacks)
    fig_w, fig_h = _figsize_even(2000, 720, dpi)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    ims = []
    for j, ax in enumerate(axes):
        ims.append(_imshow(ax, stacks[j][0], extent, cmap=S.CMAP_FIELD,
                           vmin=0.0, vmax=vmax))
        _hatch(ax, solid2d, extent)
        ax.set_title(labels[j], fontsize=10)
        ax.set_xlabel("x [m]")
        if j == 0:
            ax.set_ylabel("y [m]")
    fig.colorbar(ims[-1], ax=list(axes), shrink=0.85, label="|U| [m/s]")
    suptitle = fig.suptitle("", fontsize=12)

    def update(i: int):
        for im, s in zip(ims, stacks):
            im.set_data(s[i])
        w = _window_index(t[i], edges)
        wtxt = "" if w is None else f"  |  window W{w}"
        suptitle.set_text(f"{title_prefix}    t = {t[i]:6.1f} s{wtxt}")
        return ims

    return fig, update


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def animate_block_a(out, fps, max_frames, ff, *, solid2d, zi, extent, edges_of,
                    dump_frames=False, sample_only=False):
    """anim_A_<model>_<res>.mp4 for each model's finest complete run.

    With ``sample_only`` no mp4 is written -- a single mid-horizon PNG frame is
    saved to /tmp for a visual sanity check (used by the dry run).
    """
    runs = dataio.discover_block_a()
    rep = finest_complete_by_model(runs)
    odir = out / "A_param_only"
    if not rep:
        print("  (anim A skipped: no complete Block-A runs)")
        return
    for model in ("udales", "lbm", "palm"):
        if model not in rep:
            continue
        run = rep[model]
        model_s, truth_s, t = _slice_stack(run, zi)
        model_s = _apply_mask(model_s, solid2d)
        truth_s = _apply_mask(truth_s, solid2d)
        idx = _subsample(len(t), max_frames)
        edges = edges_of(run)
        fig, update = build_triptych(
            truth_s, model_s, t, extent=extent, solid2d=solid2d, edges=edges,
            title_prefix=f"Block A {S.MODEL_LABELS[model]} ({run.res_label})")
        res = f"nx{run.nx}"
        if sample_only:
            mid = int(idx[len(idx) // 2])
            update(mid)
            p = pathlib.Path("/tmp") / f"sample_anim_A_{model}_{res}.png"
            fig.savefig(p, dpi=fig.dpi, transparent=False)
            plt.close(fig)
            print(f"  sample frame -> {p} (t={t[mid]:.1f}s, {len(idx)} frames @ {fps} fps)")
            return p  # one sample is enough for the check
        _write_animation(fig, update, len(idx),
                         odir / f"anim_A_{model}_{res}.mp4", fps, ff,
                         frame_indices=idx, dump_frames=dump_frames)


def animate_block_b(out, fps, max_frames, ff, *, solid2d, zi, extent,
                    n_windows=2, dump_frames=False):
    """anim_B_methods.mp4: 4 panels over the last ``n_windows`` Block-B windows."""
    runs = dataio.discover_block_b()
    by_method = block_b_by_method(runs)
    odir = out / "B_state_estimation"
    if not by_method:
        print("  (anim B skipped: no complete Block-B runs)")
        return
    ref = next(iter(by_method.values()))
    edges = dataio.window_edges(ref)
    if edges is not None:
        edges = edges[edges <= HORIZON_B_S + 1e-6]
    # last n_windows windows -> [t_lo, horizon]
    t_lo = float(edges[max(0, len(edges) - 1 - n_windows)]) if edges is not None else 0.0

    # best-localization method for the IC+param panel: prefer ic_corr, else ic_dist
    ic_key = "ic_corr" if "ic_corr" in by_method else (
        "ic_dist" if "ic_dist" in by_method else None)
    base = param_only_baseline()

    icstate = by_method.get("icstate_red")
    if icstate is None:
        print("  (anim B skipped: no icstate_red run)")
        return
    model_ics, truth_s, t = _slice_stack(icstate, zi)

    # restrict to the last windows
    keep = t >= t_lo - 1e-6
    t = t[keep]
    truth_s = _apply_mask(truth_s[keep], solid2d)
    model_ics = _apply_mask(model_ics[keep], solid2d)

    panels = [("truth", truth_s)]
    if base is not None:
        base_s = _apply_mask(_baseline_slice_on_axis(base, zi, t), solid2d)
        panels.append(("param-only", base_s))
    if ic_key is not None:
        ic_s, _, _ = _slice_stack(by_method[ic_key], zi)
        ic_s = _apply_mask(ic_s[keep], solid2d)
        panels.append((S.METHOD_LABELS[ic_key], ic_s))
    panels.append((S.METHOD_LABELS["icstate_red"], model_ics))

    idx = _subsample(len(t), max_frames)
    fig, update = build_methods_panel(
        panels, truth_s, t, extent=extent, solid2d=solid2d, edges=edges,
        title_prefix="Block B methods")
    _write_animation(fig, update, len(idx), odir / "anim_B_methods.mp4", fps, ff,
                     frame_indices=idx, dump_frames=dump_frames)


def animate_block_c(out, fps, max_frames, ff, *, solid2d, zi, extent,
                    dump_frames=False):
    """anim_C_full.mp4: full-joint icstate_red triptych over the short horizon."""
    runs = dataio.discover_block_b()
    by_method = block_b_by_method(runs)
    odir = out / "C_full_joint"
    run = by_method.get("icstate_red")
    if run is None:
        print("  (anim C skipped: no icstate_red run)")
        return
    model_s, truth_s, t = _slice_stack(run, zi)
    model_s = _apply_mask(model_s, solid2d)
    truth_s = _apply_mask(truth_s, solid2d)
    edges = dataio.window_edges(run)
    if edges is not None:
        edges = edges[edges <= HORIZON_B_S + 1e-6]
    idx = _subsample(len(t), max_frames)
    fig, update = build_triptych(
        truth_s, model_s, t, extent=extent, solid2d=solid2d, edges=edges,
        title_prefix="Block C full-joint IC+state+param (reduction)")
    _write_animation(fig, update, len(idx), odir / "anim_C_full.mp4", fps, ff,
                     frame_indices=idx, dump_frames=dump_frames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent / "figures")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=150,
                    help="cap on frames per animation (time subsampled evenly)")
    ap.add_argument("--which", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--no-mask", action="store_true",
                    help="do not mask building cells")
    ap.add_argument("--frames", action="store_true",
                    help="also dump frames/<name>/frame_*.png per animation")
    ap.add_argument("--b-windows", type=int, default=2,
                    help="number of trailing Block-B windows for anim_B (1 or 2)")
    ap.add_argument("--dry-check", action="store_true",
                    help="render one Block-A sample frame to /tmp (no ffmpeg/mp4)")
    args = ap.parse_args()

    S.apply_style()
    # opaque background reads better for talk video frames
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "savefig.transparent": False})

    ff = configure_ffmpeg()
    print(f"ffmpeg: {ff or 'NOT FOUND (mp4 writing will be skipped)'}")

    solid = None if args.no_mask else mask.truth_solid_mask()
    zi = mask.nearest_z_index(FC.PED_Z)
    solid2d = None if solid is None else solid[zi]
    extent = [float(v) for v in (
        dataio.truth_grid()["x"].min(), dataio.truth_grid()["x"].max(),
        dataio.truth_grid()["y"].min(), dataio.truth_grid()["y"].max())]

    if args.dry_check:
        print("Dry check: rendering one Block-A sample frame (no ffmpeg).")
        animate_block_a(args.out, args.fps, args.max_frames, None,
                        solid2d=solid2d, zi=zi, extent=extent,
                        edges_of=dataio.window_edges, sample_only=True)
        return

    if args.which in ("A", "all"):
        print("Animation A (per-model state evolution):")
        animate_block_a(args.out, args.fps, args.max_frames, ff,
                        solid2d=solid2d, zi=zi, extent=extent,
                        edges_of=dataio.window_edges, dump_frames=args.frames)
    if args.which in ("B", "all"):
        print("Animation B (methods side by side):")
        animate_block_b(args.out, args.fps, args.max_frames, ff,
                        solid2d=solid2d, zi=zi, extent=extent,
                        n_windows=args.b_windows, dump_frames=args.frames)
    if args.which in ("C", "all"):
        print("Animation C (full-joint triptych):")
        animate_block_c(args.out, args.fps, args.max_frames, ff,
                        solid2d=solid2d, zi=zi, extent=extent,
                        dump_frames=args.frames)
    print("Done.")


if __name__ == "__main__":
    main()

"""Assemble ``state_history.nc`` for ESMDA runs from their per-window state files.

ESMDA runs (``scripts/esmda/run_esmda.py``) keep the full posterior ensemble
trajectory only as one file per window, ``windows/window_{w}_posterior_state.nc``
(dims ``(ensemble, time, z, y, x)``, window-local ``time`` starting at 0), and
reduce it to the ensemble mean in ``posterior_state_mean.nc``. The filtering
pipelines instead ship one ``state_history.nc`` holding the whole horizon.

This script gives ESMDA runs the same artifact: it concatenates the window
files along ``time`` onto ONE global monotonic axis (window ``w`` starts at
``w * simulation_time_per_window``, matching ``posterior_state_mean.nc``'s
axis) and writes ``<run_dir>/state_history.nc``. The result is the posterior
ensemble trajectory of every window, i.e. the rollout with the final
(post-ESMDA) parameters of that window -- NOT one analyzed frame per cycle as
in the filtering ``state_history.nc``; the dims stay ``(ensemble, time, ...)``.

The ensemble is never held in memory: the output is created with ``netCDF4``
and filled one window / one variable at a time (~0.5 GB slabs).

Usage (from the repo root, inside the dev env)::

    python job_scripts/local/make_esmda_state_history.py \
        /export/scratch2/ntm/experiments/esmda --glob 'pyudales_to_pyudales_*'
    python job_scripts/local/make_esmda_state_history.py <run_dir> [<run_dir> ...]

Existing ``state_history.nc`` files are skipped unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time as _time

import netCDF4
import numpy as np
import xarray as xr
import yaml

STATE_VARS = ("u", "v", "w", "pres")


def _run_dirs(paths: list[str], pattern: str) -> list[pathlib.Path]:
    """Expand CLI paths: a run dir (has ``run_info.yaml``) is taken as-is, any
    other directory is globbed for run dirs matching ``pattern``."""
    runs: list[pathlib.Path] = []
    for p in paths:
        path = pathlib.Path(p)
        if (path / "run_info.yaml").exists():
            runs.append(path)
        else:
            runs.extend(
                sorted(d for d in path.glob(pattern) if (d / "run_info.yaml").exists())
            )
    return runs


def _window_paths(run_dir: pathlib.Path) -> tuple[list[pathlib.Path], float]:
    info = yaml.safe_load((run_dir / "run_info.yaml").read_text())["configuration"]
    num_windows = int(info["num_assimilation_windows"])
    sim_time = float(info["simulation_time_per_window"])
    paths = [run_dir / "windows" / f"window_{w}_posterior_state.nc" for w in range(num_windows)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing window files {missing}")
    return paths, sim_time


def assemble(run_dir: pathlib.Path, force: bool = False) -> pathlib.Path | None:
    out = run_dir / "state_history.nc"
    if out.exists() and not force:
        print(f"[skip] {out} exists (use --force to rebuild)")
        return None
    paths, sim_time = _window_paths(run_dir)
    tmp = out.with_suffix(".nc.tmp")
    t0 = _time.time()

    # Lazily open every window to build the global time axis and check shapes.
    windows = [xr.open_dataset(p) for p in paths]
    template = windows[0]
    var_names = [v for v in STATE_VARS if v in template.data_vars]
    global_time = np.concatenate(
        [
            (np.asarray(ds["time"].values, dtype=float) - float(ds["time"].values[0]))
            + w * sim_time
            for w, ds in enumerate(windows)
        ]
    )
    for ds in windows[1:]:
        for v in var_names:
            if ds[v].dims != template[v].dims:
                raise ValueError(f"{run_dir}: dim mismatch for {v}: {ds[v].dims} vs {template[v].dims}")

    with netCDF4.Dataset(tmp, "w", format="NETCDF4") as nc:
        # Dimensions: time is unlimited so windows can be appended slab by slab.
        for dim, size in template.sizes.items():
            nc.createDimension(dim, None if dim == "time" else size)
        # Spatial coordinates copied verbatim; time gets the global axis.
        for name, coord in template.coords.items():
            if name == "time":
                continue
            var = nc.createVariable(name, coord.dtype, coord.dims)
            var[:] = coord.values
            var.setncatts({k: v for k, v in coord.attrs.items()})
        tvar = nc.createVariable("time", "f8", ("time",))
        tvar[:] = global_time
        tvar.setncatts({"long_name": "global simulation time", "units": "s"})
        # Data variables, chunked one member-time-slab at a time; no compression
        # so reads stay fast (the source files are uncompressed too).
        for v in var_names:
            dims = template[v].dims
            chunks = tuple(1 if d in ("ensemble", "time") else template.sizes[d] for d in dims)
            var = nc.createVariable(v, template[v].dtype, dims, chunksizes=chunks)
            var.setncatts({k: val for k, val in template[v].attrs.items()})
        nc.setncatts({k: str(val) for k, val in template.attrs.items()})
        nc.setncattr(
            "history",
            "state_history.nc assembled from windows/window_{w}_posterior_state.nc "
            "by job_scripts/local/make_esmda_state_history.py; time rebased so "
            f"window w starts at w*{sim_time}",
        )

        # Fill: one window, one variable at a time (~0.5 GB slabs).
        t_off = 0
        for w, ds in enumerate(windows):
            nt = ds.sizes["time"]
            for v in var_names:
                tax = ds[v].dims.index("time")
                sl = [slice(None)] * ds[v].ndim
                sl[tax] = slice(t_off, t_off + nt)
                nc.variables[v][tuple(sl)] = ds[v].values
            t_off += nt
            print(f"  {run_dir.name}: window {w} written ({t_off}/{len(global_time)} frames, {_time.time()-t0:.0f}s)")
    for ds in windows:
        ds.close()
    tmp.replace(out)
    print(f"[done] {out} ({out.stat().st_size/2**30:.1f} GiB, {_time.time()-t0:.0f}s)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="ESMDA run dirs, or parent dirs to glob")
    ap.add_argument("--glob", default="*", help="run-dir pattern used for parent dirs (default: *)")
    ap.add_argument("--force", action="store_true", help="rebuild existing state_history.nc")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be built")
    args = ap.parse_args(argv)

    runs = _run_dirs(args.paths, args.glob)
    if not runs:
        print("no run dirs found", file=sys.stderr)
        return 1
    print(f"{len(runs)} run dir(s):")
    for r in runs:
        print(f"  {r}")
    if args.dry_run:
        return 0
    for r in runs:
        assemble(r, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

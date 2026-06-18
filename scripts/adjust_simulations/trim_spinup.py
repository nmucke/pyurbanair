"""Trim the spin-up transient from a saved ground-truth state + params pair.

Reads a ``state.nc`` / ``params.nc`` artifact (as written by
``run_forward_model.py``), drops every frame whose ``time`` is before
``--spinup-time`` seconds, rebases the remaining time axis to start at 0, and
writes the trimmed artifacts to ``--output-dir`` (as ``state.nc`` / ``params.nc``
so the directory can be fed straight to ``run_esmda.py``'s ``run.truth_dir``).

The state is streamed in time-chunks via the netCDF4 backend, so a multi-GB
truth is never held in memory in full. Parameters are small and handled with
xarray. Only frames with ``time >= spinup_time`` are kept and the kept times are
shifted down by ``spinup_time``, so the output time axis always starts at >= 0.

Usage::

    python scripts/trim_spinup.py \
        --state ground_truth/state.nc \
        --params ground_truth/params.nc \
        --spinup-time 25 \
        --output-dir ground_truth_spunup
"""

import argparse
import pathlib
import shutil

import numpy as np
import xarray
from netCDF4 import Dataset

TIME_NAME = "time"


def _copy_trimmed_state(
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    spinup_time: float,
    time_chunk: int,
) -> tuple[int, int]:
    """Copy ``in_path`` to ``out_path`` dropping frames with time < spinup_time.

    Streams the time-dependent variables in blocks of ``time_chunk`` frames so
    peak memory stays a small multiple of one block, regardless of the total
    file size. Returns ``(n_dropped, n_kept)``.
    """
    with Dataset(in_path, "r") as src, Dataset(out_path, "w", format=src.data_model) as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

        times = np.asarray(src.variables[TIME_NAME][:], dtype=float)
        start = int(np.searchsorted(times, spinup_time, side="left"))
        n_keep = times.size - start
        if n_keep <= 0:
            raise ValueError(
                f"spinup_time={spinup_time} drops all {times.size} frames "
                f"(max time = {times.max():g})"
            )

        for name, dim in src.dimensions.items():
            if name == TIME_NAME:
                dst.createDimension(name, n_keep)
            else:
                dst.createDimension(name, None if dim.isunlimited() else len(dim))

        for name, var in src.variables.items():
            kwargs: dict = {}
            filters = var.filters() or {}
            if filters.get("zlib"):
                kwargs["zlib"] = True
                kwargs["complevel"] = filters.get("complevel", 4)
                kwargs["shuffle"] = filters.get("shuffle", False)
            chunking = var.chunking()
            if isinstance(chunking, (list, tuple)):
                cs = list(chunking)
                if TIME_NAME in var.dimensions:
                    cs[var.dimensions.index(TIME_NAME)] = min(
                        cs[var.dimensions.index(TIME_NAME)], n_keep
                    )
                kwargs["chunksizes"] = cs
            fill = getattr(var, "_FillValue", None)

            out = dst.createVariable(
                name, var.dtype, var.dimensions, fill_value=fill, **kwargs
            )
            out.setncatts(
                {k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"}
            )

            if TIME_NAME not in var.dimensions:
                out[:] = var[:]
                continue
            if name == TIME_NAME:
                out[:] = times[start:] - spinup_time
                continue

            # Time-dependent field: copy one block of frames at a time.
            t_axis = var.dimensions.index(TIME_NAME)
            for c0 in range(0, n_keep, time_chunk):
                c1 = min(c0 + time_chunk, n_keep)
                src_idx = [slice(None)] * var.ndim
                dst_idx = [slice(None)] * var.ndim
                src_idx[t_axis] = slice(start + c0, start + c1)
                dst_idx[t_axis] = slice(c0, c1)
                out[tuple(dst_idx)] = var[tuple(src_idx)]

    return start, n_keep


def _copy_trimmed_params(
    in_path: pathlib.Path, out_path: pathlib.Path, spinup_time: float
) -> None:
    """Trim + rebase the (small) parameter time axis with xarray, then write."""
    params = xarray.load_dataset(in_path)
    if TIME_NAME in params.dims:
        params = params.sel({TIME_NAME: params[TIME_NAME] >= spinup_time})
        params = params.assign_coords({TIME_NAME: params[TIME_NAME] - spinup_time})
    params.to_netcdf(out_path)
    params.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", required=True, type=pathlib.Path,
                        help="Path to the input state.nc")
    parser.add_argument("--params", required=True, type=pathlib.Path,
                        help="Path to the input params.nc")
    parser.add_argument("--spinup-time", required=True, type=float,
                        help="Seconds of leading spin-up to drop")
    parser.add_argument("--output-dir", required=True, type=pathlib.Path,
                        help="Directory to write the trimmed state.nc / params.nc")
    parser.add_argument("--time-chunk", type=int, default=50,
                        help="Number of frames to stream per block (default: 50)")
    parser.add_argument("--state-only", action="store_true",
                        help="Only trim the state; copy params.nc through unchanged "
                             "instead of trimming + rebasing its time axis")
    args = parser.parse_args()

    if args.spinup_time < 0:
        parser.error("--spinup-time must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_state = args.output_dir / "state.nc"
    out_params = args.output_dir / "params.nc"
    if out_state.resolve() == args.state.resolve():
        parser.error("output state would overwrite the input state")

    n_dropped, n_kept = _copy_trimmed_state(
        args.state, out_state, args.spinup_time, args.time_chunk
    )
    print(
        f"State:  dropped {n_dropped} spin-up frames (< t={args.spinup_time:g}s), "
        f"kept {n_kept} -> {out_state}"
    )

    if args.state_only:
        if out_params.resolve() != args.params.resolve():
            shutil.copyfile(args.params, out_params)
        print(f"Params: copied through unchanged -> {out_params}")
    else:
        _copy_trimmed_params(args.params, out_params, args.spinup_time)
        print(f"Params: trimmed + rebased -> {out_params}")


if __name__ == "__main__":
    main()

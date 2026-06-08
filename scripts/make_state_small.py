"""Create a reduced copy of state.nc.

Keeps only the u, v, w state variables and the time range [0, 1800],
writing to state_small.nc in the same folder. The original state.nc is
opened read-only and left untouched.

Data are streamed in time batches so peak memory stays small (~1 GB),
rather than loading the full ~40 GB selection into RAM at once.
"""

import os

import netCDF4 as nc
import numpy as np

FOLDER = "/projects/prjs2075/urbanair/ground_truth/pyudales_time_varying"
SRC = os.path.join(FOLDER, "state_full.nc")
DST = os.path.join(FOLDER, "state.nc")

KEEP_VARS = ["u", "v", "w"]
DROP_VARS = {"u", "v", "w", "pres"}  # state vars; only KEEP_VARS are written
T_MIN, T_MAX = 0.0, 1000.0
BATCH = 100  # time steps written per chunk


def copy_attrs(src_var, dst_var):
    for attr in src_var.ncattrs():
        if attr == "_FillValue":
            continue  # set at creation time, cannot be copied as an attr
        dst_var.setncattr(attr, src_var.getncattr(attr))


def main():
    if os.path.exists(DST):
        raise SystemExit(f"refusing to overwrite existing {DST}")

    src = nc.Dataset(SRC, "r")
    try:
        time = src.variables["time"][:]
        # time is monotonically increasing; keep the contiguous prefix in range.
        nt = int(np.count_nonzero((time >= T_MIN) & (time <= T_MAX)))
        print(f"Source time steps: {time.size}")
        print(f"Selected time steps: {nt} ({float(time[0])} .. {float(time[nt-1])})")
        print(f"Variables kept: {KEEP_VARS}")

        dst = nc.Dataset(DST, "w", format=src.file_format)
        try:
            # Dimensions (time becomes the truncated, unlimited dimension).
            for name, dim in src.dimensions.items():
                if name == "time":
                    dst.createDimension("time", None)
                else:
                    dst.createDimension(name, len(dim))

            # Coordinate / auxiliary variables: everything except the state vars.
            for name, var in src.variables.items():
                if name in DROP_VARS:
                    continue
                fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
                out = dst.createVariable(
                    name, var.datatype, var.dimensions, fill_value=fill,
                )
                copy_attrs(var, out)
                if "time" in var.dimensions:
                    out[:] = var[:nt]
                else:
                    out[:] = var[:]

            # State variables, streamed in time batches.
            for name in KEEP_VARS:
                var = src.variables[name]
                fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
                chunks = var.chunking()
                out = dst.createVariable(
                    name, var.datatype, var.dimensions, fill_value=fill,
                    chunksizes=chunks if chunks not in (None, "contiguous") else None,
                )
                copy_attrs(var, out)
                for i in range(0, nt, BATCH):
                    j = min(i + BATCH, nt)
                    out[i:j] = var[i:j]
                    print(f"  {name}: wrote time {i}:{j}/{nt}", flush=True)

            # Global attributes.
            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
        finally:
            dst.close()
    finally:
        src.close()

    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()

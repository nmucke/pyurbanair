#!/usr/bin/env python
"""Copy ground_truth NetCDF files to 32-bit float.

Reads every *.nc file in ``ground_truth/64_bit`` and writes a copy into
``ground_truth/32_bit`` where any 64-bit ``double`` variable is downcast to
32-bit ``float``. All other dtypes, dimensions, and attributes are preserved.

The source files are opened read-only and are never modified. Large variables
are copied slice-by-slice along their first (unlimited) dimension to keep memory
bounded even for multi-GB files.
"""

from pathlib import Path

import netCDF4
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "ground_truth" / "64_bit"
DST_DIR = Path(__file__).resolve().parent.parent / "ground_truth" / "32_bit"

# Chunk size (in elements along the leading dim) used when streaming big vars.
SLICE_LIMIT = 64


def out_dtype(dtype: np.dtype) -> np.dtype:
    """Map f8 -> f4, leave everything else untouched."""
    return np.dtype("f4") if dtype == np.dtype("f8") else dtype


def convert_file(src_path: Path, dst_path: Path) -> None:
    with netCDF4.Dataset(src_path, "r") as src, netCDF4.Dataset(
        dst_path, "w", format=src.data_model
    ) as dst:
        # Global attributes.
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

        # Dimensions (preserve unlimited).
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))

        for name, var in src.variables.items():
            new_dtype = out_dtype(var.dtype)

            # _FillValue must be passed at creation time.
            fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None

            out = dst.createVariable(
                name,
                new_dtype,
                var.dimensions,
                fill_value=fill,
                zlib=var.filters().get("zlib", False) if var.filters() else False,
            )

            # Copy attributes except _FillValue (already handled).
            out.setncatts(
                {k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"}
            )

            if var.ndim == 0:
                out[...] = var[...]
                continue

            n = var.shape[0]
            for start in range(0, n, SLICE_LIMIT):
                stop = min(start + SLICE_LIMIT, n)
                out[start:stop] = var[start:stop]

            print(f"  {name}: {var.dtype} -> {new_dtype}  shape={var.shape}")


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(SRC_DIR.glob("*.nc"))
    if not nc_files:
        raise SystemExit(f"No .nc files found in {SRC_DIR}")

    for src_path in nc_files:
        dst_path = DST_DIR / src_path.name
        print(f"{src_path}  ->  {dst_path}")
        convert_file(src_path, dst_path)

    print("Done.")


if __name__ == "__main__":
    main()

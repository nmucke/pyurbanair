"""M0 offline diff: compare the combine-on vs combine-off NetCDF outputs and
the pre/post bare-combine snapshots. Run on the login node with the delftblue
pixi env.
"""

import sys
from pathlib import Path

import numpy as np
import xarray


STASH = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/ntmucke/m0_capture/9997968/stash")


def summary(label: str, path: Path) -> xarray.Dataset:
    print(f"\n=== {label}: {path} ===")
    ds = xarray.open_dataset(path, engine="netcdf4", decode_timedelta=False)
    print(f"  size on disk: {path.stat().st_size:,} bytes")
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  coords: {list(ds.coords)}")
    print(f"  data_vars: {list(ds.data_vars)}")
    for v in ("u", "v", "w"):
        if v in ds.data_vars:
            arr = ds[v].values
            nonnan = arr[~np.isnan(arr)]
            print(f"  {v}: shape={arr.shape}  dtype={arr.dtype}  nan_count={np.isnan(arr).sum()}  "
                  f"nonnan_min={(nonnan.min() if nonnan.size else None)!r}  "
                  f"nonnan_max={(nonnan.max() if nonnan.size else None)!r}  "
                  f"nonzero_count={int((arr != 0).sum())}/{arr.size}")
    return ds


def diff_structure(label_a: str, ds_a: xarray.Dataset, label_b: str, ds_b: xarray.Dataset) -> None:
    print(f"\n--- structural diff: {label_a} vs {label_b} ---")
    only_a = set(ds_a.variables) - set(ds_b.variables)
    only_b = set(ds_b.variables) - set(ds_a.variables)
    print(f"  variables only in {label_a}: {sorted(only_a)}")
    print(f"  variables only in {label_b}: {sorted(only_b)}")

    dims_diff = {k: (ds_a.sizes.get(k), ds_b.sizes.get(k)) for k in set(ds_a.sizes) | set(ds_b.sizes) if ds_a.sizes.get(k) != ds_b.sizes.get(k)}
    print(f"  differing dim sizes: {dims_diff or 'NONE'}")

    coord_diffs = []
    for name in sorted(set(ds_a.coords) & set(ds_b.coords)):
        a = ds_a.coords[name].values
        b = ds_b.coords[name].values
        if a.shape != b.shape:
            coord_diffs.append(f"  coord '{name}' shape diff: {a.shape} vs {b.shape}")
        else:
            if np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
                same = np.allclose(a, b, equal_nan=True)
            else:
                same = np.array_equal(a, b)
            if not same:
                coord_diffs.append(f"  coord '{name}' VALUES DIFFER")
    print(f"  coord differences: {coord_diffs if coord_diffs else 'NONE'}")

    attr_diffs = []
    for k in set(ds_a.attrs) | set(ds_b.attrs):
        if ds_a.attrs.get(k) != ds_b.attrs.get(k):
            attr_diffs.append((k, ds_a.attrs.get(k), ds_b.attrs.get(k)))
    print(f"  global-attr diffs: {len(attr_diffs)}")
    for k, va, vb in attr_diffs[:5]:
        print(f"    {k!r}: A={va!r}  B={vb!r}")


def diff_data(ds_a: xarray.Dataset, ds_b: xarray.Dataset) -> None:
    print(f"\n--- data-value diff (combine-on vs combine-off) ---")
    for v in ("u", "v", "w"):
        if v not in ds_a.data_vars or v not in ds_b.data_vars:
            continue
        a = ds_a[v].values
        b = ds_b[v].values
        if a.shape != b.shape:
            print(f"  {v}: SHAPES DIFFER {a.shape} vs {b.shape}")
            continue
        # Treat NaN as equal-to-NaN so the off file's masked cells don't poison the diff.
        diff = np.where(np.isnan(a) & np.isnan(b), 0.0, a - b)
        n_diff = int(np.sum(diff != 0))
        print(f"  {v}: n_diff={n_diff}/{a.size}  diff_min={diff.min():.4g}  diff_max={diff.max():.4g}  "
              f"|diff|_mean={np.abs(diff).mean():.4g}")


def main() -> None:
    combine_on = STASH / "run1_combine_on" / "urban_run_3d.000.nc"
    combine_off = STASH / "run2_combine_off" / "urban_run_3d.000.nc"
    pre_bare = STASH / "run3_combine_bare" / "DATA_3D_NETCDF.pre_bare_combine"
    post_bare = STASH / "run3_combine_bare" / "DATA_3D_NETCDF.post_bare_combine"

    ds_on = summary("combine ON (run1)", combine_on)
    ds_off = summary("combine OFF (run2, -Z)", combine_off)
    ds_pre = summary("pre bare-combine (= run2's tempdir, pre)", pre_bare)
    ds_post = summary("post bare-combine (= run2's tempdir, post-bare combine_plot_fields.x)", post_bare)

    diff_structure("combine_ON", ds_on, "combine_OFF", ds_off)
    diff_data(ds_on, ds_off)

    diff_structure("pre_bare", ds_pre, "post_bare", ds_post)
    diff_data(ds_post, ds_on)


if __name__ == "__main__":
    main()

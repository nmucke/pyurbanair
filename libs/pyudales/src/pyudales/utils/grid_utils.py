"""Grid utilities for uDALES staggered grid interpolation."""

import xarray


def interpolate_grid(ds: xarray.Dataset) -> xarray.Dataset:
    """
    Interpolate all data variables to center positions (xt, yt, zt).

    The uDALES staggered grid has variables at different positions:
    - w: at (zm, yt, xt) - needs interpolation from zm to zt
    - v: at (zt, ym, xt) - needs interpolation from ym to yt
    - u: at (zt, yt, xm) - needs interpolation from xm to xt
    - pres: at (zt, yt, xt) - already at centers

    This function ensures all variables are interpolated to center positions
    (xt, yt, zt) using linear interpolation.

    Args:
        ds: xarray Dataset with uDALES state variables on staggered grid

    Returns:
        xarray Dataset with all variables interpolated to center positions (xt, yt, zt)
    """
    result_vars = {}

    # Process each variable
    for var_name, var_data in ds.data_vars.items():
        dims = var_data.dims

        # Check which dimensions need interpolation
        needs_z_interp = "zm" in dims
        needs_y_interp = "ym" in dims
        needs_x_interp = "xm" in dims

        # Start with the original data
        interpolated = var_data

        # Interpolate z dimension: zm -> zt
        if needs_z_interp:
            interpolated = interpolated.interp(
                zm=ds.zt,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            # Rename dimension from zm to zt to match coordinate name

            interpolated = interpolated.drop_vars("zm")

        # Interpolate y dimension: ym -> yt
        if needs_y_interp:
            interpolated = interpolated.interp(
                ym=ds.yt,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            # Rename dimension from ym to yt to match coordinate name
            interpolated = interpolated.drop_vars("ym")

        # Interpolate x dimension: xm -> xt
        if needs_x_interp:
            interpolated = interpolated.interp(
                xm=ds.xt,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            # Rename dimension from xm to xt to match coordinate name
            interpolated = interpolated.drop_vars("xm")

        result_vars[var_name] = interpolated

    # Create new dataset with interpolated variables
    # Keep only center coordinates (xt, yt, zt) and time
    result_ds = xarray.Dataset(
        data_vars=result_vars,
        coords={
            "time": ds.time,
            "xt": ds.xt,
            "yt": ds.yt,
            "zt": ds.zt,
        },
        attrs=ds.attrs,
    )

    return result_ds

"""Utilities for handling parameter extraction and application for ForwardModel."""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray

if TYPE_CHECKING:
    from .dir_utils import DirectoryPaths

from .infile_utils import Infile
from .vertical_profile import build_profile_shape

logger = logging.getLogger(__name__)


def is_time_varying_params(params: Optional[xarray.Dataset]) -> bool:
    """Check if parameters contain time-varying inflow_angle or velocity_magnitude.

    Args:
        params: Optional xarray.Dataset that may contain inflow parameters.

    Returns:
        True if any inflow parameter has a ``time`` dimension.
    """
    if params is None:
        return False
    for var_name in ("inflow_angle", "velocity_magnitude"):
        if var_name in params and "time" in params[var_name].dims:
            return True
    return False


def get_param_value(params: xarray.Dataset, param_name: str) -> Optional[float]:
    """
    Extract a parameter value from an xarray.Dataset.

    Args:
        params: xarray.Dataset containing parameters.
        param_name: Name of the parameter to extract.

    Returns:
        The parameter value as a float, or None if not found.
    """
    if param_name not in params:
        return None

    return params[param_name].item()  # type: ignore[no-any-return]


def apply_inflow_settings(
    params: xarray.Dataset,
    dirs: "DirectoryPaths",
) -> None:
    """
    Apply the inflow settings to infile.in.

    Args:
        params: xarray.Dataset containing inflow parameters. Must contain
               inflow_angle and velocity_magnitude.
        dirs: DirectoryPaths instance containing infile_path.

    Raises:
        ValueError: If required parameters are missing or uini key is not found.
    """
    # Extract parameter values from params Dataset
    inflow_angle = get_param_value(params, "inflow_angle")
    velocity_magnitude = get_param_value(params, "velocity_magnitude")

    # Check if required parameters are present
    if inflow_angle is None:
        raise ValueError("inflow_angle is required but not found in params")

    if velocity_magnitude is None:
        raise ValueError("velocity_magnitude is required but not found in params")

    # LBM's udir uses the same convention as pyudales' inflow_angle (measured
    # from +x, CCW), so write it as-is (no negation). Verified by matching
    # uDALES and LBM observations at identical parameters.
    inflow_angle_for_lbm = inflow_angle

    # Update infile.in using Infile class
    infile = Infile(dirs.infile_path)

    # Key is first word after "!" on the uini, udir line (e.g. "uini,")
    uini_key = next((k for k in infile.get_keys() if k.startswith("uini")), None)
    if uini_key is None:
        raise ValueError(
            f"Could not find 'uini, udir' in infile.in at {dirs.infile_path}"
        )

    infile.set_value(uini_key, f"{velocity_magnitude:.1f} {inflow_angle_for_lbm:.1f}")
    infile.write()

    logger.info(
        f"Updated inflow settings: angle={inflow_angle:.1f}° "
        f"(written to LBM as {inflow_angle_for_lbm:.1f}°), "
        f"velocity={velocity_magnitude:.1f} m/s"
    )


def extract_initial_params(params: xarray.Dataset) -> xarray.Dataset:
    """Extract scalar initial-time values from a time-varying params Dataset.

    Takes the first time point of each time-varying variable and returns a
    scalar Dataset suitable for ``apply_inflow_settings``.
    """
    data_vars: dict = {}
    for name in ("inflow_angle", "velocity_magnitude"):
        if name not in params:
            continue
        var = params[name]
        if "time" in var.dims:
            data_vars[name] = float(var.isel(time=0).item())
        else:
            data_vars[name] = float(var.item())
    return xarray.Dataset(data_vars=data_vars)


def write_uvel_time_file(
    params: xarray.Dataset,
    dirs: "DirectoryPaths",
    spinup_time: float = 0.0,
    time_offset: float = 0.0,
) -> None:
    """Write ``uvel_time.dat`` for the Fortran LBM code.

    The file format expected by ``m_inflow.F90`` is one row per data point::

        time_seconds  velocity_m_s  direction_degrees

    Times are in physical seconds, velocity in m/s, direction in degrees.
    The Fortran code handles non-dimensionalization internally.

    Args:
        params: xarray.Dataset with ``velocity_magnitude`` and ``inflow_angle``
                having a ``time`` dimension.
        dirs: DirectoryPaths with ``experiment_dir``.
        spinup_time: If > 0, prepend a constant row at t=0 with initial values
                     and shift user-provided times by this offset.
        time_offset: Shift applied to the (window-relative) schedule so it lands
                     on the LBM's absolute clock. ``params["time"]`` is treated as
                     global simulation time and first re-based to the window start
                     (``time - time[0]``); ``time_offset`` (= ``nt0 * dt``) then
                     places it at the warm-start clock position. 0 for a cold
                     start (``nt0 = 0``).
    """
    times = params["time"].values.astype(float)
    # Re-base global window time to window-relative, then onto the run clock.
    times = times - times[0] + time_offset
    velocities = params["velocity_magnitude"].values.astype(float)

    # Broadcast scalar inflow_angle to match time dimension if needed
    if "time" in params["inflow_angle"].dims:
        angles = params["inflow_angle"].values.astype(float)
    else:
        angles = np.full_like(times, float(params["inflow_angle"].item()))

    # LBM's udir uses the same convention as pyudales' inflow_angle (measured
    # from +x, CCW): the inflow kernel applies cos(udir)/sin(udir) directly to
    # the x/y momentum, matching angle_to_velocity. Write the angle as-is, so
    # this path is consistent with the static apply_inflow_settings path. A
    # previous negation here flipped the inflow direction ONLY in the
    # time-varying path, which drove cross-model ESMDA (pyudales truth) to
    # negative angles.
    angles_for_lbm = angles

    if spinup_time > 0:
        # Prepend constant plateau at initial values for spinup period
        times = np.concatenate([[0.0], times + spinup_time])
        velocities = np.concatenate([[velocities[0]], velocities])
        angles_for_lbm = np.concatenate([[angles_for_lbm[0]], angles_for_lbm])

    file_path = dirs.experiment_dir / "uvel_time.dat"
    with open(file_path, "w") as f:
        for t, vel, ang in zip(times, velocities, angles_for_lbm):
            f.write(f"{t:.6f}  {vel:.6f}  {ang:.6f}\n")

    logger.info("Wrote %d time-varying inflow entries to %s", len(times), file_path)


def remove_uvel_time_file(dirs: "DirectoryPaths") -> None:
    """Remove ``uvel_time.dat`` if it exists, preventing stale data."""
    file_path = dirs.experiment_dir / "uvel_time.dat"
    if file_path.exists():
        file_path.unlink()
        logger.info("Removed stale %s", file_path)


def write_uvel_shear_file(
    dirs: "DirectoryPaths",
    heights: np.ndarray,
    zsize: float,
    profile_config: dict,
) -> None:
    """Write ``uvel_shear.dat`` describing the vertical inflow shear profile.

    The file format expected by ``m_inflow.F90`` is one row per vertical level::

        k  z_meters  shape_value

    The Fortran code reads the third column into ``uvel_h(k)`` and renormalizes
    it by the top-cell value (``uvel_shear(k) = uvel_h(k)/uvel_h(nz)``), so the
    inlet velocity becomes ``uini * uvel_shear(k)``.  We write the dimensionless
    shape ``s(z) = (z/z_ref)**alpha`` produced by :func:`build_profile_shape`,
    using the same ``power_law``/``z_ref`` convention as pyudales so that, for a
    given ``velocity_magnitude``, both backends impose the same inflow shear.

    Note: because the Fortran normalizes by the top cell, LBM's ``uini`` is the
    speed at the top *cell center*, whereas pyudales' ``velocity_magnitude`` is
    the speed at ``z_ref = zsize`` (domain top).  The profile *shape* matches
    exactly; the reference height differs by half a cell (<1% for typical nz).

    Args:
        dirs: DirectoryPaths with ``experiment_dir``.
        heights: Cell-center heights from the domain bottom, shape (nz,).
        zsize: Domain height in meters (default ``z_ref`` for the power law).
        profile_config: ``{"type": ..., ...}`` passed to ``build_profile_shape``.
    """
    shape = build_profile_shape(profile_config, heights, zsize)

    file_path = dirs.experiment_dir / "uvel_shear.dat"
    with open(file_path, "w") as f:
        for k, (z, s) in enumerate(zip(heights, shape), start=1):
            f.write(f"{k:6d}  {z:.6f}  {s:.6f}\n")

    logger.info(
        "Wrote vertical inflow shear (%s) with %d levels to %s",
        profile_config.get("type", "uniform"),
        len(heights),
        file_path,
    )


def remove_uvel_shear_file(dirs: "DirectoryPaths") -> None:
    """Remove ``uvel_shear.dat`` if it exists, preventing stale data.

    Without the file ``m_inflow.F90`` defaults to a uniform inflow profile.
    """
    file_path = dirs.experiment_dir / "uvel_shear.dat"
    if file_path.exists():
        file_path.unlink()
        logger.info("Removed stale %s", file_path)

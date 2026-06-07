"""Utilities for generating time-dependent nudging files for uDALES.

uDALES supports time-dependent velocity nudging via:
- A `timedepnudge.inp.{expnr}` file containing vertical profiles at discrete time snapshots
- Namoptions flags: lnudge, ltimedepnudge, ntimedepnudge, nnudge, tnudge

The Fortran code linearly interpolates between snapshots at runtime and relaxes
horizontal-average velocities toward the target profiles.
"""

import logging
import pathlib
from typing import Any, Optional

import numpy as np
import xarray

from .dir_utils import DirectoryPaths
from .file_update_utils import update_lscale_file_profile, update_prof_file_profile
from .inflow_utils import angle_to_velocity
from .namoptions_utils import NamoptionsFile
from .vertical_profile import build_profile_shape

logger = logging.getLogger(__name__)


def compute_nudging_profiles(
    time_seconds: np.ndarray,
    inflow_angle: np.ndarray,
    velocity_magnitude: np.ndarray,
    heights: np.ndarray,
    thl0: float = 288.0,
    qt0: float = 0.0,
    profile_shape: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute nudging profiles for each time snapshot.

    At each time step, the velocity components u(t) and v(t) are computed from
    the angle and magnitude, then multiplied by ``profile_shape`` along the
    vertical axis.  When ``profile_shape`` is None, the profiles are uniform.

    Args:
        time_seconds: Time values in seconds, shape (n_times,).
        inflow_angle: Inflow angle in degrees at each time, shape (n_times,).
        velocity_magnitude: Velocity magnitude in m/s at each time, shape (n_times,).
        heights: Vertical grid cell centers in meters, shape (ktot,).
        thl0: Constant potential temperature (K) for all profiles.
        qt0: Constant specific humidity (kg/kg) for all profiles.
        profile_shape: Dimensionless shape s(z), length ktot.  If None, uniform.

    Returns:
        Tuple of (thl_profiles, qt_profiles, u_profiles, v_profiles),
        each with shape (n_times, ktot).
    """
    n_times = len(time_seconds)
    ktot = len(heights)

    u_arr, v_arr = angle_to_velocity(inflow_angle, velocity_magnitude)

    shape = profile_shape if profile_shape is not None else np.ones(ktot)
    u_profiles = u_arr[:, np.newaxis] * shape[np.newaxis, :]
    v_profiles = v_arr[:, np.newaxis] * shape[np.newaxis, :]
    thl_profiles = np.full((n_times, ktot), thl0)
    qt_profiles = np.full((n_times, ktot), qt0)

    return thl_profiles, qt_profiles, u_profiles, v_profiles


def write_timedepnudge_file(
    file_path: pathlib.Path,
    time_seconds: np.ndarray,
    heights: np.ndarray,
    thl_profiles: np.ndarray,
    qt_profiles: np.ndarray,
    u_profiles: np.ndarray,
    v_profiles: np.ndarray,
) -> None:
    """Write a timedepnudge.inp file in the format expected by uDALES.

    File format per time block::

        height    thl        qt       u        v
        #    {time_seconds}
        {z}  {thl}  {qt}  {u}  {v}     (one row per vertical level)
        --------------------------------------------

    Args:
        file_path: Output file path.
        time_seconds: Time values in seconds, shape (n_times,).
        heights: Vertical grid cell centers, shape (ktot,).
        thl_profiles: Potential temperature profiles, shape (n_times, ktot).
        qt_profiles: Specific humidity profiles, shape (n_times, ktot).
        u_profiles: U-velocity profiles, shape (n_times, ktot).
        v_profiles: V-velocity profiles, shape (n_times, ktot).
    """
    with open(file_path, "w") as f:
        for t_idx, t_sec in enumerate(time_seconds):
            f.write("height    thl        qt       u        v\n")
            f.write(f"#    {t_sec:.3f}\n")
            for k in range(len(heights)):
                f.write(
                    f"  {heights[k]:11.6f}"
                    f"  {thl_profiles[t_idx, k]:11.6f}"
                    f"    {qt_profiles[t_idx, k]:11.6f}"
                    f"    {u_profiles[t_idx, k]:11.6f}"
                    f"    {v_profiles[t_idx, k]:11.6f}\n"
                )
            f.write("--------------------------------------------\n")

    logger.info(
        "Wrote timedepnudge file %s with %d time snapshots and %d levels",
        file_path.name,
        len(time_seconds),
        len(heights),
    )


def enable_nudging_in_namoptions(
    namoptions_path: pathlib.Path,
    n_time_snapshots: int,
    nnudge: int,
    tnudge: float = 10.0,
) -> None:
    """Enable time-dependent velocity nudging in namoptions.

    Sets the &PHYSICS section flags required for uDALES to read the
    timedepnudge.inp file and apply nudging.

    Args:
        namoptions_path: Path to the namoptions file.
        n_time_snapshots: Number of time-dependent nudging profiles.
        nnudge: Number of vertical levels from the bottom that are NOT nudged.
        tnudge: Nudging relaxation timescale in seconds.
    """
    namoptions = NamoptionsFile(namoptions_path)
    namoptions.set_value("PHYSICS", "lnudge", ".true.")
    namoptions.set_value("PHYSICS", "nnudge", nnudge)
    namoptions.set_value("PHYSICS", "tnudge", f"{tnudge:.1f}")
    namoptions.set_value("PHYSICS", "ltimedepnudge", ".true.")
    namoptions.set_value("PHYSICS", "ntimedepnudge", n_time_snapshots)
    namoptions.write()

    logger.info(
        "Enabled nudging in %s: nnudge=%d, tnudge=%.1f, ntimedepnudge=%d",
        namoptions_path.name,
        nnudge,
        tnudge,
        n_time_snapshots,
    )


def apply_time_varying_inflow(
    params: xarray.Dataset,
    dirs: DirectoryPaths,
    tnudge: float = 10.0,
    nnudge: Optional[int] = None,
    nnudge_meters: Optional[float] = None,
    spinup_time: float = 0.0,
    simulation_time: float = 0.0,
    boundary_condition: str = "periodic",
    profile_config: Optional[dict[str, Any]] = None,
) -> None:
    """Apply inflow settings via nudging files.

    Generates a ``timedepnudge.inp.{expnr}`` file, enables nudging in
    namoptions, and applies the initial (t=0) values as static inflow
    settings for prof.inp/lscale.inp consistency.

    Supports two modes:

    * **Time-varying params** (Dataset with ``time`` dimension): nudging
      profiles follow the time-varying schedule.
    * **Scalar / constant params** (no ``time`` dimension): a synthetic
      2-snapshot constant schedule spanning ``[0, runtime]`` is created so
      that nudging holds the values fixed — this is needed to stabilize
      inflow/outflow boundary conditions.

    When ``spinup_time > 0``, a constant plateau at the initial parameter
    values is prepended so that the flow reaches quasi-steady state before
    the time-varying schedule begins.

    Args:
        params: xarray.Dataset with ``inflow_angle`` and
            ``velocity_magnitude``.  May include a ``time`` coordinate
            (seconds relative to simulation start, after spinup) for
            time-varying mode, or be scalar for constant nudging.
        dirs: DirectoryPaths for the experiment.
        tnudge: Nudging relaxation timescale in seconds.
        nnudge: Number of vertical levels from the bottom that are NOT nudged.
            Defaults to 0 (nudge entire domain).
        nnudge_meters: Height (in meters above the domain floor) below which
            nudging is NOT applied; nudging is applied above this height. When
            given, it is converted to a grid-level count and overrides
            ``nnudge``.  Cells whose center lies below ``nnudge_meters`` are
            excluded from nudging.
        spinup_time: Duration of the spinup period in seconds.  During spinup
            the nudging holds the initial parameter values constant.
    """
    from .params_utils import apply_inflow_settings

    namoptions_path = dirs.experiment_dir / f"namoptions.{dirs.experiment_name}"
    namoptions = NamoptionsFile(namoptions_path)

    # Read grid parameters
    ktot = namoptions.get_value_as_int("DOMAIN", "ktot")
    zsize = namoptions.get_value_as_float("INPS", "zsize")
    if ktot is None or zsize is None:
        raise ValueError(
            "Cannot read ktot or zsize from namoptions; "
            "these are required for time-varying inflow."
        )
    dz = zsize / ktot
    heights = np.arange(0.5 * dz, zsize, dz)  # cell centers

    profile_shape = build_profile_shape(profile_config, heights, zsize)

    # Resolve the number of un-nudged bottom levels. ``nnudge_meters`` (a
    # physical height) takes precedence over ``nnudge`` (a raw level count):
    # count the cells whose center lies below the requested height so that
    # nudging is applied only above it.
    if nnudge_meters is not None:
        nnudge = int(np.count_nonzero(heights < nnudge_meters))
        logger.info(
            "Converted nnudge_meters=%.2f m to nnudge=%d levels (dz=%.3f m).",
            nnudge_meters,
            nnudge,
            dz,
        )
    elif nnudge is None:
        nnudge = 0

    if nnudge >= ktot:
        raise ValueError(
            f"nnudge={nnudge} leaves no nudged levels (ktot={ktot}). "
            + (
                f"nnudge_meters={nnudge_meters} m exceeds the domain height "
                f"({zsize} m)."
                if nnudge_meters is not None
                else "Reduce nnudge below ktot."
            )
        )

    # Extract arrays from params.  When params has no ``time`` dimension
    # (scalar / constant params), create a synthetic constant schedule
    # spanning the full runtime so that nudging holds constant values.
    if "time" not in params.dims:
        time_seconds = np.linspace(0.0, simulation_time, 2)

        inflow_angle = np.linspace(
            params["inflow_angle"].values, params["inflow_angle"].values, 2
        )
        velocity_mag = np.linspace(
            params["velocity_magnitude"].values, params["velocity_magnitude"].values, 2
        )

        logger.info(
            "Scalar-params nudging schedule: t=%s, angle=%.2f, vel=%.2f",
            time_seconds,
            params["inflow_angle"].values,
            params["velocity_magnitude"].values,
        )
    else:
        time_seconds = params["time"].values.astype(float)

        inflow_angle = params["inflow_angle"].values
        velocity_mag = params["velocity_magnitude"].values

        # Handle mixed scalar + time-varying: broadcast to same shape
        if inflow_angle.ndim == 0:
            inflow_angle = np.full_like(time_seconds, float(inflow_angle))
        if velocity_mag.ndim == 0:
            velocity_mag = np.full_like(time_seconds, float(velocity_mag))

    # Handle spinup: prepend a constant plateau at initial values and shift
    # user times so that the time-varying schedule starts after spinup.
    if spinup_time > 0:
        # Prepend t=0 with initial values (constant during spinup)
        time_seconds = np.concatenate([[0.0], time_seconds + spinup_time])
        inflow_angle = np.concatenate([[inflow_angle[0]], inflow_angle])
        velocity_mag = np.concatenate([[velocity_mag[0]], velocity_mag])

    # Compute profiles
    thl_profs, qt_profs, u_profs, v_profs = compute_nudging_profiles(
        time_seconds,
        inflow_angle,
        velocity_mag,
        heights,
        profile_shape=profile_shape,
    )

    # Write the timedepnudge file
    nudge_file_path = dirs.experiment_dir / f"timedepnudge.inp.{dirs.experiment_name}"
    write_timedepnudge_file(
        nudge_file_path,
        time_seconds,
        heights,
        thl_profs,
        qt_profs,
        u_profs,
        v_profs,
    )

    # Enable nudging in namoptions
    enable_nudging_in_namoptions(
        namoptions_path,
        len(time_seconds),
        nnudge,
        tnudge,
    )

    # Under inflow_outflow BCs the west inlet face (plus nudging) drives the
    # flow, so the constant body-force pressure gradient (INPS dpdx/dpdy)
    # inherited from the case namoptions is redundant.  Left non-zero it adds
    # a direction-frozen momentum source that LBM (inlet-driven, no body force)
    # has no equivalent of, biasing cross-model ESMDA.  Zero it so that
    # velocity_magnitude is the only streamwise driver, matching LBM.
    if boundary_condition == "inflow_outflow":
        body_force_namoptions = NamoptionsFile(namoptions_path)
        body_force_namoptions.set_value("INPS", "dpdx", "0.0")
        body_force_namoptions.set_value("INPS", "dpdy", "0.0")
        body_force_namoptions.write()

    zeros = np.zeros(ktot)
    update_prof_file_profile(
        dirs.experiment_dir / f"prof.inp.{dirs.experiment_name}",
        u_profile=zeros,
        v_profile=zeros,
    )
    update_lscale_file_profile(
        dirs.experiment_dir / f"lscale.inp.{dirs.experiment_name}",
        u_profile=zeros,
        v_profile=zeros,
        dpdx_profile=zeros,
        dpdy_profile=zeros,
    )
    # if boundary_condition == "inflow_outflow":
    #     # Match the reference expnr=400 convention: start the flow from
    #     # rest and let nudging plus the west-face inflow BC ramp toward
    #     # the t=0 target during spinup.  Writing t=0 velocities into
    #     # prof.inp stagnates the flow against building walls before the
    #     # pressure solver has settled.
    #     zeros = np.zeros(ktot)
    #     update_prof_file_profile(
    #         dirs.experiment_dir / f"prof.inp.{dirs.experiment_name}",
    #         u_profile=zeros,
    #         v_profile=zeros,
    #     )
    #     update_lscale_file_profile(
    #         dirs.experiment_dir / f"lscale.inp.{dirs.experiment_name}",
    #         u_profile=zeros,
    #         v_profile=zeros,
    #         dpdx_profile=zeros,
    #         dpdy_profile=zeros,
    #     )

    #     # Still write scalar u0, v0 to namoptions INPS (uDALES uses
    #     # these as reference scalars independent of IC).  dpdx/dpdy are
    #     # zeroed because under inflow_outflow the west-face BC drives
    #     # the flow.
    #     u0, v0 = angle_to_velocity(float(inflow_angle[0]), float(velocity_mag[0]))
    #     namoptions.set_value("INPS", "u0", f"{u0:.7f}")
    #     namoptions.set_value("INPS", "v0", f"{v0:.7f}")
    #     namoptions.set_value("INPS", "dpdx", "0.0")
    #     namoptions.set_value("INPS", "dpdy", "0.0")
    #     namoptions.write()
    # else:
    #     # Periodic BC: keep the t=0 profile in prof.inp and lscale.inp,
    #     # because there is no inflow face to drive the flow.
    #     initial_params_vars: dict = {
    #         "inflow_angle": float(inflow_angle[0]),
    #         "velocity_magnitude": float(velocity_mag[0]),
    #     }
    #     if "pressure_gradient_magnitude" in params:
    #         pg = params["pressure_gradient_magnitude"].values
    #         initial_params_vars["pressure_gradient_magnitude"] = (
    #             float(pg) if pg.ndim == 0 else float(pg[0])
    #         )

    #     initial_params = xarray.Dataset(data_vars=initial_params_vars)
    #     apply_inflow_settings(
    #         initial_params,
    #         dirs,
    #         boundary_condition=boundary_condition,
    #         profile_shape=profile_shape,
    #     )

    # Verify critical files exist
    nudge_exists = nudge_file_path.exists()
    prof_path = dirs.experiment_dir / f"prof.inp.{dirs.experiment_name}"
    lscale_path = dirs.experiment_dir / f"lscale.inp.{dirs.experiment_name}"
    logger.info(
        "Nudging setup complete: timedepnudge=%s, prof=%s, lscale=%s, "
        "n_snapshots=%d, initial_angle=%.2f, initial_vel=%.2f",
        nudge_exists,
        prof_path.exists(),
        lscale_path.exists(),
        len(time_seconds),
        inflow_angle[0],
        velocity_mag[0],
    )

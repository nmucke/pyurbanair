"""
Utility functions for converting flow angle to pressure gradient and velocity components in u-dales.

Based on analysis of u-dales source code:
- In modforces.f90, pressure gradients are subtracted from velocity tendencies:
    up = up - dpdxl
    vp = vp - dpdyl
- For steady-state flow, the pressure gradient should be proportional to velocity
- The relationship: dpdx/dpdy = u0/v0 (for the same magnitude scaling factor)
- The magnitude of the pressure gradient vector should be constant: |dp| = sqrt(dpdx² + dpdy²)

Key insights:
1. The pressure gradient magnitude should remain constant regardless of angle.
   For example, if dpdx = 0.0041912 Pa/m at 0°, then at -45°:
       dpdx = 0.0041912 * cos(-45°) ≈ 0.002960 Pa/m
       dpdy = 0.0041912 * sin(-45°) ≈ -0.002960 Pa/m
       |dp| = 0.0041912 Pa/m (constant)

2. Velocity components follow the same pattern:
       u = wind_speed * cos(angle)
       v = wind_speed * sin(angle)
       |U| = sqrt(u² + v²) = wind_speed (constant)
"""

import numpy as np


def angle_to_pressure_gradient(
    angle_deg: float,
    dp_magnitude: float,
) -> tuple[float, float]:
    """
    Convert flow angle to pressure gradient components (dpdx, dpdy).

    In u-dales, the pressure gradient force is applied as:
        du/dt = -dpdx/ρ
        dv/dt = -dpdy/ρ

    The pressure gradient components are calculated to maintain a constant
    magnitude regardless of angle:
        dpdx = dp_magnitude * cos(angle)
        dpdy = dp_magnitude * sin(angle)

    This ensures that |dp| = sqrt(dpdx² + dpdy²) = dp_magnitude for all angles.

    Args:
        angle_deg: Flow angle in degrees, measured from positive x-axis.
                   Positive angles are counterclockwise (meteorological convention).
                   -45° means flow from northeast to southwest.
        dp_magnitude: Magnitude of the pressure gradient vector (Pa/m).
                     This is the total magnitude: |dp| = sqrt(dpdx² + dpdy²)

    Returns:
        tuple: (dpdx, dpdy) in Pa/m

    Examples:
        >>> # For -45 degree flow with magnitude 0.0041912 Pa/m
        >>> dpdx, dpdy = angle_to_pressure_gradient(-45, dp_magnitude=0.0041912)
        >>> print(f"dpdx={dpdx:.7f}, dpdy={dpdy:.7f}")
        >>> # Verify: sqrt(dpdx² + dpdy²) = 0.0041912

        >>> # For 0 degree flow (west to east)
        >>> dpdx, dpdy = angle_to_pressure_gradient(0, dp_magnitude=0.0041912)
        >>> # dpdx = 0.0041912, dpdy = 0.0
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)

    # Calculate pressure gradient components
    # The magnitude is constant, components follow the angle
    dpdx = dp_magnitude * np.cos(angle_rad)
    dpdy = dp_magnitude * np.sin(angle_rad)

    return dpdx, dpdy


def angle_to_velocity(
    angle_deg: float,
    wind_speed: float,
) -> tuple[float, float]:
    """
    Convert flow angle to velocity components (u, v).

    The velocity components are calculated to maintain a constant
    magnitude regardless of angle:
        u = wind_speed * cos(angle)
        v = wind_speed * sin(angle)

    This ensures that |U| = sqrt(u² + v²) = wind_speed for all angles.

    Args:
        angle_deg: Flow angle in degrees, measured from positive x-axis.
                   Positive angles are counterclockwise (meteorological convention).
                   -45° means flow from northeast to southwest.
        wind_speed: Magnitude of the wind speed vector (m/s).
                   This is the total magnitude: |U| = sqrt(u² + v²)

    Returns:
        tuple: (u, v) in m/s

    Examples:
        >>> # For -45 degree flow with wind speed 3.0 m/s
        >>> u, v = angle_to_velocity(-45, wind_speed=3.0)
        >>> print(f"u={u:.4f}, v={v:.4f}")
        >>> # Verify: sqrt(u² + v²) = 3.0

        >>> # For 0 degree flow (west to east)
        >>> u, v = angle_to_velocity(0, wind_speed=3.0)
        >>> # u = 3.0, v = 0.0
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)

    # Calculate velocity components
    # The magnitude is constant, components follow the angle
    u = wind_speed * np.cos(angle_rad)
    v = wind_speed * np.sin(angle_rad)

    return u, v


def velocity_magnitude(u: float, v: float) -> np.ndarray:
    """
    Calculate the magnitude of the velocity vector from components.

    Args:
        u: Velocity component in x-direction (m/s)
        v: Velocity component in y-direction (m/s)

    Returns:
        Magnitude of velocity vector: sqrt(u² + v²) in m/s
    """
    return np.sqrt(u**2 + v**2)


def velocity_to_angle(u: float, v: float) -> np.ndarray:
    """
    Convert velocity components to flow angle.

    This is the inverse of angle_to_velocity.

    Args:
        u: Velocity component in x-direction (m/s)
        v: Velocity component in y-direction (m/s)

    Returns:
        Flow angle in degrees, measured from positive x-axis
    """
    angle_rad = np.arctan2(v, u)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


def pressure_gradient_magnitude(dpdx: float, dpdy: float) -> np.ndarray:
    """
    Calculate the magnitude of the pressure gradient vector from components.

    Args:
        dpdx: Pressure gradient in x-direction (Pa/m)
        dpdy: Pressure gradient in y-direction (Pa/m)

    Returns:
        Magnitude of pressure gradient vector: sqrt(dpdx² + dpdy²) in Pa/m
    """
    return np.sqrt(dpdx**2 + dpdy**2)


def pressure_gradient_to_angle(dpdx: float, dpdy: float) -> np.ndarray:
    """
    Convert pressure gradient components to flow angle.

    This is the inverse of angle_to_pressure_gradient.

    Args:
        dpdx: Pressure gradient in x-direction (Pa/m)
        dpdy: Pressure gradient in y-direction (Pa/m)

    Returns:
        Flow angle in degrees, measured from positive x-axis
    """
    angle_rad = np.arctan2(dpdy, dpdx)
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg


def verify_pressure_gradient(
    angle_deg: float,
    u0: float,
    v0: float,
    dpdx: float,
    dpdy: float,
    tolerance: float = 0.01,
) -> bool:
    """
    Verify that pressure gradient components are consistent with flow angle and velocity.

    Args:
        angle_deg: Flow angle in degrees
        u0: Initial u-velocity component (m/s)
        v0: Initial v-velocity component (m/s)
        dpdx: Pressure gradient in x-direction (Pa/m)
        dpdy: Pressure gradient in y-direction (Pa/m)
        tolerance: Relative tolerance for verification

    Returns:
        True if consistent, False otherwise
    """
    # Check angle consistency
    angle_from_vel = np.rad2deg(np.arctan2(v0, u0))
    angle_from_dp = pressure_gradient_to_angle(dpdx, dpdy)

    # Check ratio consistency: dpdx/dpdy should equal u0/v0 (if both non-zero)
    if abs(v0) > 1e-10 and abs(dpdy) > 1e-10:
        ratio_vel = u0 / v0
        ratio_dp = dpdx / dpdy
        ratio_consistent = (
            abs(ratio_vel - ratio_dp) / max(abs(ratio_vel), abs(ratio_dp)) < tolerance
        )
    else:
        ratio_consistent = True  # Can't check ratio if denominator is zero

    angle_consistent = abs(angle_from_vel - angle_deg) < tolerance

    return ratio_consistent and angle_consistent


##### Example usage #####
if __name__ == "__main__":
    # Example: Verify your -45 degree case
    print("Example: -45 degree flow")
    print("=" * 50)

    angle = -45.0
    u0_example = 2.1213
    v0_example = -2.1213
    dpdx_example = 0.0029343
    dpdy_example = -0.0029343

    # Calculate the magnitude from the -45 degree case
    dp_magnitude_from_example = pressure_gradient_magnitude(dpdx_example, dpdy_example)
    print(f"Magnitude from -45° example: {dp_magnitude_from_example:.7f} Pa/m")
    print()

    # Use the magnitude from 0 degrees (as provided by user)
    dp_magnitude = 0.0041912
    print(f"Using magnitude: {dp_magnitude} Pa/m (from 0° case)")
    print(f"Note: This should match the magnitude at all angles")
    print()

    print(f"Input angle: {angle} degrees")
    print(f"u0: {u0_example}, v0: {v0_example}")
    print(f"dpdx: {dpdx_example}, dpdy: {dpdy_example}")
    print()

    # Calculate what dpdx, dpdy should be with constant magnitude
    dpdx_calc, dpdy_calc = angle_to_pressure_gradient(angle, dp_magnitude)
    print(f"Calculated dpdx: {dpdx_calc:.7f}, dpdy: {dpdy_calc:.7f}")
    print(f"Calculated magnitude: {np.sqrt(dpdx_calc**2 + dpdy_calc**2):.7f} Pa/m")
    print()

    # Verify consistency
    is_consistent = verify_pressure_gradient(
        angle, u0_example, v0_example, dpdx_example, dpdy_example
    )
    print(f"Consistency check: {'PASS' if is_consistent else 'FAIL'}")
    print()

    # Show relationship
    print("Relationship:")
    print(f"  dpdx/dpdy = {dpdx_example/dpdy_example:.6f}")
    print(f"  u0/v0 = {u0_example/v0_example:.6f}")
    print(
        f"  Ratio match: {abs(dpdx_example/dpdy_example - u0_example/v0_example) < 0.001}"
    )
    print()

    # Test other angles with constant magnitude
    print("\nOther examples (constant magnitude):")
    print("=" * 50)
    print(f"Using dp_magnitude = {dp_magnitude} Pa/m for all angles")
    print()
    for test_angle in [0, 30, 60, 90, 120, 150, 180, -90, -60, -30, -45]:
        dpdx_test, dpdy_test = angle_to_pressure_gradient(test_angle, dp_magnitude)
        magnitude_test = np.sqrt(dpdx_test**2 + dpdy_test**2)
        print(
            f"Angle {test_angle:4d}°: dpdx={dpdx_test:9.7f}, dpdy={dpdy_test:9.7f}, |dp|={magnitude_test:.7f}"
        )

    # Example: Velocity components from angle
    print("\n" + "=" * 50)
    print("Example: Velocity components from angle")
    print("=" * 50)

    wind_speed = velocity_magnitude(u0_example, v0_example)
    print(f"Wind speed magnitude from -45° example: {wind_speed:.4f} m/s")
    print()

    # Test velocity function for various angles
    print("Velocity components for different angles (wind_speed = 3.0 m/s):")
    print()
    test_wind_speed = 3.0
    for test_angle in [0, 30, 60, 90, 120, 150, 180, -90, -60, -30, -45]:
        u_test, v_test = angle_to_velocity(test_angle, test_wind_speed)
        magnitude_test = velocity_magnitude(u_test, v_test)
        angle_check = velocity_to_angle(u_test, v_test)
        print(
            f"Angle {test_angle:4d}°: u={u_test:8.4f}, v={v_test:8.4f}, |U|={magnitude_test:.4f}, angle_check={angle_check:6.1f}°"
        )

"""Convert (angle, magnitude) to (u, v) wind components.

Mirrors pyudales/utils/inflow_utils.py:angle_to_velocity — a ten-line function
duplicated across wrappers deliberately (three-copy tolerance) so pypalm does
not take a runtime dependency on pyudales.
"""

import numpy as np


def angle_to_velocity(
    angle_deg: float | np.ndarray,
    wind_speed: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert flow angle (degrees from +x axis, CCW) and speed to (u, v).

    ``sqrt(u**2 + v**2) == wind_speed`` for all angles.
    """
    angle_rad = np.deg2rad(angle_deg)
    u = wind_speed * np.cos(angle_rad)
    v = wind_speed * np.sin(angle_rad)
    return u, v

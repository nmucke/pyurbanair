"""Colormap lookup tables shared by every renderer.

The exporter writes each layer's LUT into the manifest (256 linear-RGB
triples) so Blender and Unreal reproduce identical colours without needing
matplotlib. Values are *linear* RGB (sRGB-decoded), which is what both
engines' material colour ramps and emissive inputs expect.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps as _mpl_colormaps


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lut(name: str, n: int = 256, linear: bool = True) -> np.ndarray:
    """(n, 3) RGB table for a matplotlib colormap name."""
    rgb = _mpl_colormaps[name](np.linspace(0.0, 1.0, n))[:, :3]
    return srgb_to_linear(rgb) if linear else rgb


def lut_list(name: str, n: int = 256) -> list[list[float]]:
    return [[round(float(v), 5) for v in row] for row in lut(name, n)]


def apply(
    values: np.ndarray, vmin: float, vmax: float, name: str, linear: bool = False
) -> np.ndarray:
    """Map scalars to RGB (..., 3) in [0, 1]; sRGB by default (for PNG output)."""
    table = lut(name, 256, linear=linear)
    x = np.clip((np.asarray(values, dtype=np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
    return table[np.rint(x * 255).astype(np.int64)]


def layer_color_spec(name: str, vmin: float, vmax: float, variable: str) -> dict:
    """The manifest block describing how a layer is coloured."""
    return {
        "variable": variable,
        "range": [float(vmin), float(vmax)],
        "colormap": name,
        "lut_linear_rgb": lut_list(name),
    }

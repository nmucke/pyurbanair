"""Canonical handling of scalar and anisotropic Tadpole tile sizes."""

from __future__ import annotations

from collections.abc import Sequence

CropSize = int | Sequence[int]
CropShape = tuple[int, int, int]


def normalize_crop_size(value: object) -> CropShape:
    """Return ``encoder_crop_size`` as a validated ``(z, y, x)`` tuple."""
    if type(value) is int:
        sizes = (int(value),) * 3
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
        if len(raw) != 3:
            raise ValueError(
                "encoder_crop_size must be an int or a 3-item (z, y, x) "
                f"sequence, got {value!r}"
            )
        if any(type(size) is not int for size in raw):
            raise ValueError(
                "encoder_crop_size entries must be integers, got " f"{value!r}"
            )
        sizes = (int(raw[0]), int(raw[1]), int(raw[2]))
    else:
        raise ValueError(
            "encoder_crop_size must be an int or a 3-item (z, y, x) sequence, "
            f"got {value!r}"
        )

    if any(size < 16 or size % 16 != 0 for size in sizes):
        raise ValueError(
            "each encoder_crop_size entry must be a positive multiple of 16, got "
            f"{value!r}"
        )
    return sizes


def crop_size_config_value(shape: CropShape) -> int | list[int]:
    """Keep legacy isotropic configs scalar; spell anisotropic shapes as lists."""
    return shape[0] if shape[0] == shape[1] == shape[2] else list(shape)

"""Unit tests for :func:`neural_surrogates.sdf.sdf_features`.

Correctness of the SDF + ∇SDF geometry-feature helper on hand-built masks in a
small grid: sign inside/outside, exact axis distances, unit-norm gradients
pointing away from the obstacle, clamp behaviour, batched == looped, and the
dtype/device round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neural_surrogates import sdf_features
from neural_surrogates.sdf import N_SDF_FEATURE_CHANNELS

NZ = NY = NX = 9


def _point_obstacle_mask() -> torch.Tensor:
    """All-fluid grid with a single solid voxel at the centre (4, 4, 4)."""
    mask = torch.ones(NZ, NY, NX)
    mask[4, 4, 4] = 0.0
    return mask


def _block_obstacle_mask() -> torch.Tensor:
    """All-fluid grid with a solid block spanning x in [3, 5] at mid z/y."""
    mask = torch.ones(NZ, NY, NX)
    mask[3:6, 3:6, 3:6] = 0.0
    return mask


def test_output_shape_and_channel_count() -> None:
    feat = sdf_features(_point_obstacle_mask())
    assert feat.shape == (N_SDF_FEATURE_CHANNELS, NZ, NY, NX)
    assert N_SDF_FEATURE_CHANNELS == 4


def test_sign_inside_and_outside() -> None:
    feat = sdf_features(_point_obstacle_mask(), clamp_cells=8.0)
    sdf_n = feat[0]
    # Solid cell: sdf < 0.
    assert sdf_n[4, 4, 4] < 0.0
    # Fluid cells: sdf > 0.
    assert sdf_n[4, 4, 5] > 0.0
    assert sdf_n[0, 0, 0] > 0.0


def test_exact_axis_distances() -> None:
    """Along an axis from the single solid voxel, the (unclamped) SDF equals the
    integer cell distance; with clamp_cells=L that is distance / L."""
    L = 8.0
    feat = sdf_features(_point_obstacle_mask(), clamp_cells=L)
    sdf_n = feat[0]
    for k in (1, 2, 3):
        assert sdf_n[4, 4, 4 + k].item() == pytest.approx(k / L, abs=1e-6)
        assert sdf_n[4, 4 + k, 4].item() == pytest.approx(k / L, abs=1e-6)
        assert sdf_n[4 + k, 4, 4].item() == pytest.approx(k / L, abs=1e-6)


def test_gradient_unit_norm_and_direction() -> None:
    """Gradient is unit-length and points away from the obstacle."""
    feat = sdf_features(_block_obstacle_mask(), clamp_cells=32.0)
    grad = feat[1:]  # (3, *grid) = (g_z, g_y, g_x)
    norm = torch.linalg.vector_norm(grad, dim=0)
    # On the +x side of the block, the nearest wall is the x=5 face, so the
    # gradient of the positive-in-fluid SDF points +x (away from the block).
    gz, gy, gx = grad[:, 4, 4, 7]
    assert gx.item() == pytest.approx(1.0, abs=1e-6)
    assert gy.item() == pytest.approx(0.0, abs=1e-6)
    assert gz.item() == pytest.approx(0.0, abs=1e-6)
    # Unit-norm at fluid cells a little away from the interface.
    assert norm[4, 4, 7].item() == pytest.approx(1.0, abs=1e-5)
    assert norm[4, 4, 8].item() == pytest.approx(1.0, abs=1e-5)


def test_clamp_behaviour() -> None:
    """A cell far from the obstacle saturates the clamped channel to 1."""
    L = 2.0
    feat = sdf_features(_point_obstacle_mask(), clamp_cells=L)
    sdf_n = feat[0]
    # Corner is many cells from the centre voxel -> clipped to +1.
    assert sdf_n[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert sdf_n.max().item() <= 1.0 + 1e-6
    assert sdf_n.min().item() >= -1.0 - 1e-6


def test_batched_matches_looped() -> None:
    m0 = _point_obstacle_mask()
    m1 = _block_obstacle_mask()
    batched = sdf_features(torch.stack([m0, m1], dim=0), clamp_cells=6.0)
    assert batched.shape == (2, 4, NZ, NY, NX)
    torch.testing.assert_close(batched[0], sdf_features(m0, clamp_cells=6.0))
    torch.testing.assert_close(batched[1], sdf_features(m1, clamp_cells=6.0))


def test_dtype_and_device_round_trip() -> None:
    mask64 = _point_obstacle_mask().to(torch.float64)
    feat = sdf_features(mask64)
    assert feat.dtype == torch.float64
    assert feat.device == mask64.device


def test_rejects_bad_ndim() -> None:
    with pytest.raises(ValueError):
        sdf_features(torch.ones(NY, NX))  # 2-D grid unsupported

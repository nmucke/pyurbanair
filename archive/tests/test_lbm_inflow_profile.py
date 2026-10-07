"""Tests for the LBM vertical inflow shear profile (uvel_shear.dat).

LBM previously imposed a uniform inflow, while pyudales applies a power-law
shear at the inlet.  These tests cover the ``uvel_shear.dat`` writer and assert
that pylbm and pyudales share the same profile convention, so that a given
``velocity_magnitude`` imposes the same inflow shear in both backends.
"""

import pathlib
import types

import numpy as np
from pylbm.utils.params_utils import remove_uvel_shear_file, write_uvel_shear_file
from pylbm.utils.vertical_profile import build_profile_shape


def _dirs(tmp_path: pathlib.Path) -> types.SimpleNamespace:
    """Minimal stand-in exposing the only attribute the writer uses."""
    return types.SimpleNamespace(experiment_dir=tmp_path)


def _heights(nz: int, zsize: float) -> np.ndarray:
    dz = zsize / nz
    return (np.arange(nz) + 0.5) * dz


class TestBuildProfileShape:
    def test_power_law_matches_formula(self) -> None:
        zsize = 40.0
        heights = _heights(8, zsize)
        alpha = 0.25
        shape = build_profile_shape({"type": "power_law", "alpha": alpha}, heights, zsize)
        np.testing.assert_allclose(shape, (heights / zsize) ** alpha)

    def test_uniform_is_ones(self) -> None:
        heights = _heights(8, 40.0)
        np.testing.assert_allclose(
            build_profile_shape({"type": "uniform"}, heights, 40.0), 1.0
        )

    def test_none_is_uniform(self) -> None:
        heights = _heights(8, 40.0)
        np.testing.assert_allclose(build_profile_shape(None, heights, 40.0), 1.0)

    def test_matches_pyudales_convention(self) -> None:
        """pylbm and pyudales must produce identical shapes for the same input."""
        from pyudales.utils.vertical_profile import build_profile_shape as udales_build

        zsize = 40.0
        heights = _heights(16, zsize)
        cfg = {"type": "power_law", "alpha": 0.25}
        np.testing.assert_allclose(
            build_profile_shape(cfg, heights, zsize),
            udales_build(cfg, heights, zsize),
        )


class TestWriteUvelShearFile:
    def test_writes_one_row_per_level(self, tmp_path: pathlib.Path) -> None:
        nz, zsize = 8, 40.0
        heights = _heights(nz, zsize)
        write_uvel_shear_file(
            _dirs(tmp_path), heights, zsize, {"type": "power_law", "alpha": 0.25}
        )

        out = tmp_path / "uvel_shear.dat"
        assert out.exists()
        rows = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert len(rows) == nz

        # Columns are: index, height, shape value.
        last = rows[-1].split()
        assert int(last[0]) == nz
        np.testing.assert_allclose(float(last[1]), heights[-1], atol=1e-6)
        np.testing.assert_allclose(
            float(last[2]), (heights[-1] / zsize) ** 0.25, atol=1e-6
        )

    def test_shear_is_monotonic_increasing(self, tmp_path: pathlib.Path) -> None:
        nz, zsize = 8, 40.0
        heights = _heights(nz, zsize)
        write_uvel_shear_file(
            _dirs(tmp_path), heights, zsize, {"type": "power_law", "alpha": 0.25}
        )
        values = [
            float(ln.split()[2])
            for ln in (tmp_path / "uvel_shear.dat").read_text().splitlines()
            if ln.strip()
        ]
        assert all(b > a for a, b in zip(values, values[1:]))


class TestRemoveUvelShearFile:
    def test_removes_existing_file(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "uvel_shear.dat"
        path.write_text("1  0.5  1.0\n")
        remove_uvel_shear_file(_dirs(tmp_path))
        assert not path.exists()

    def test_no_error_when_missing(self, tmp_path: pathlib.Path) -> None:
        # Should be a no-op rather than raising.
        remove_uvel_shear_file(_dirs(tmp_path))

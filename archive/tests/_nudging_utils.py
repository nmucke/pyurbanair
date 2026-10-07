"""Unit tests for pyudales nudging utilities."""

import pathlib
import tempfile

import numpy as np
import pytest
import xarray
from pyudales.utils.inflow_utils import angle_to_velocity
from pyudales.utils.nudging_utils import (
    compute_nudging_profiles,
    enable_nudging_in_namoptions,
    write_timedepnudge_file,
)
from pyudales.utils.params_utils import is_time_varying_params


class TestIsTimeVaryingParams:
    def test_none_params(self) -> None:
        assert is_time_varying_params(None) is False

    def test_scalar_params(self) -> None:
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": 30.0,
                "velocity_magnitude": 5.0,
            }
        )
        assert is_time_varying_params(params) is False

    def test_time_varying_angle(self) -> None:
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": ("time", [30.0, 45.0, 60.0]),
                "velocity_magnitude": 5.0,
            },
            coords={"time": [0.0, 50.0, 100.0]},
        )
        assert is_time_varying_params(params) is True

    def test_time_varying_velocity(self) -> None:
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": 30.0,
                "velocity_magnitude": ("time", [3.0, 4.0, 5.0]),
            },
            coords={"time": [0.0, 50.0, 100.0]},
        )
        assert is_time_varying_params(params) is True

    def test_both_time_varying(self) -> None:
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": ("time", [30.0, 45.0, 60.0]),
                "velocity_magnitude": ("time", [3.0, 4.0, 5.0]),
            },
            coords={"time": [0.0, 50.0, 100.0]},
        )
        assert is_time_varying_params(params) is True

    def test_ensemble_dim_not_time(self) -> None:
        """Ensemble dimension should not trigger time-varying detection."""
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": ("ensemble", [30.0, 45.0]),
                "velocity_magnitude": ("ensemble", [3.0, 4.0]),
            },
            coords={"ensemble": [0, 1]},
        )
        assert is_time_varying_params(params) is False


class TestComputeNudgingProfiles:
    def test_uniform_profiles_shape(self) -> None:
        time_s = np.array([0.0, 100.0, 200.0])
        angles = np.array([30.0, 45.0, 60.0])
        velocities = np.array([3.0, 4.0, 5.0])
        heights = np.linspace(0.5, 9.5, 10)

        thl, qt, u, v = compute_nudging_profiles(
            time_s,
            angles,
            velocities,
            heights,
        )

        assert thl.shape == (3, 10)
        assert qt.shape == (3, 10)
        assert u.shape == (3, 10)
        assert v.shape == (3, 10)

    def test_uniform_profiles_values(self) -> None:
        """Profiles should be uniform at all heights."""
        time_s = np.array([0.0, 100.0])
        angles = np.array([0.0, 90.0])
        velocities = np.array([5.0, 5.0])
        heights = np.linspace(0.5, 9.5, 10)

        thl, qt, u, v = compute_nudging_profiles(
            time_s,
            angles,
            velocities,
            heights,
        )

        # At t=0: angle=0 => u=5, v=0
        np.testing.assert_allclose(u[0, :], 5.0, atol=1e-10)
        np.testing.assert_allclose(v[0, :], 0.0, atol=1e-10)

        # At t=100: angle=90 => u=0, v=5
        np.testing.assert_allclose(u[1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(v[1, :], 5.0, atol=1e-10)

    def test_constant_thl_qt(self) -> None:
        time_s = np.array([0.0])
        angles = np.array([45.0])
        velocities = np.array([3.0])
        heights = np.linspace(0.5, 9.5, 5)

        thl, qt, u, v = compute_nudging_profiles(
            time_s,
            angles,
            velocities,
            heights,
            thl0=300.0,
            qt0=0.01,
        )

        np.testing.assert_allclose(thl[0, :], 300.0)
        np.testing.assert_allclose(qt[0, :], 0.01)

    def test_velocity_magnitude_preserved(self) -> None:
        """Total wind speed should be preserved at all times."""
        time_s = np.linspace(0, 100, 11)
        angles = np.linspace(0, 180, 11)
        velocities = np.full(11, 4.0)
        heights = np.array([5.0])

        _, _, u, v = compute_nudging_profiles(
            time_s,
            angles,
            velocities,
            heights,
        )

        magnitudes = np.sqrt(u[:, 0] ** 2 + v[:, 0] ** 2)
        np.testing.assert_allclose(magnitudes, 4.0, atol=1e-10)


class TestWriteTimedepnudgeFile:
    def test_file_format(self, tmp_path: pathlib.Path) -> None:
        file_path = tmp_path / "timedepnudge.inp.999"
        time_s = np.array([0.0, 100.0])
        heights = np.array([0.5, 1.5, 2.5])
        thl = np.full((2, 3), 288.0)
        qt = np.full((2, 3), 0.0)
        u = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        v = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        write_timedepnudge_file(file_path, time_s, heights, thl, qt, u, v)

        content = file_path.read_text()
        lines = content.split("\n")

        # First block: header, time marker, 3 data lines, separator
        assert "height" in lines[0]
        assert "#    0" in lines[1]
        assert "----" in lines[5]

        # Second block
        assert "height" in lines[6]
        assert "#    100" in lines[7]
        assert "----" in lines[11]

    def test_file_has_correct_number_of_blocks(self, tmp_path: pathlib.Path) -> None:
        file_path = tmp_path / "timedepnudge.inp.999"
        n_times = 5
        n_levels = 4
        time_s = np.arange(n_times, dtype=float) * 10.0
        heights = np.arange(n_levels, dtype=float) + 0.5
        thl = np.full((n_times, n_levels), 288.0)
        qt = np.zeros((n_times, n_levels))
        u = np.ones((n_times, n_levels))
        v = np.ones((n_times, n_levels))

        write_timedepnudge_file(file_path, time_s, heights, thl, qt, u, v)

        lines = file_path.read_text().strip().split("\n")
        # Count full separator lines (lines starting with dashes)
        separator_count = sum(1 for line in lines if line.startswith("----"))
        assert separator_count == n_times


class TestEnableNudgingInNamoptions:
    def test_sets_physics_flags(self, tmp_path: pathlib.Path) -> None:
        namoptions_path = tmp_path / "namoptions.999"
        namoptions_path.write_text(
            "&RUN\n"
            "iexpnr      = 999\n"
            "runtime     = 100.\n"
            "/\n"
            "\n"
            "&DOMAIN\n"
            "itot        = 40\n"
            "jtot        = 40\n"
            "ktot        = 4\n"
            "/\n"
        )

        enable_nudging_in_namoptions(
            namoptions_path, n_time_snapshots=11, nnudge=0, tnudge=10.0
        )

        from pyudales.utils.namoptions_utils import NamoptionsFile

        nf = NamoptionsFile(namoptions_path)
        assert nf.get_value("PHYSICS", "lnudge") == ".true."
        assert nf.get_value("PHYSICS", "ltimedepnudge") == ".true."
        assert nf.get_value_as_int("PHYSICS", "ntimedepnudge") == 11
        assert nf.get_value_as_int("PHYSICS", "nnudge") == 0
        assert nf.get_value_as_float("PHYSICS", "tnudge") == 10.0


class TestSpinupPlateau:
    def test_no_spinup_leaves_times_unchanged(self, tmp_path: pathlib.Path) -> None:
        """With spinup_time=0, times should be unchanged."""
        namoptions_path = tmp_path / "namoptions.999"
        namoptions_path.write_text(
            "&RUN\niexpnr = 999\n/\n"
            "&DOMAIN\nitot = 4\njtot = 4\nktot = 4\n/\n"
            "&INPS\nzsize = 10.\n/\n"
        )
        # Create a minimal experiment dir
        exp_dir = tmp_path
        from pyudales.utils.dir_utils import DirectoryPaths

        dirs = DirectoryPaths(
            case_dir=tmp_path,
            experiment_dir=tmp_path,
            experiment_name="999",
            experiment_base_dir=tmp_path,
            temp_dir=tmp_path,
            output_dir=tmp_path,
            udales_root_path=tmp_path,
            cwd=tmp_path,
            results_dir=None,
        )

        params = xarray.Dataset(
            data_vars={
                "inflow_angle": ("time", [30.0, 45.0, 60.0]),
                "velocity_magnitude": ("time", [3.0, 4.0, 5.0]),
            },
            coords={"time": [0.0, 50.0, 100.0]},
        )

        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        apply_time_varying_inflow(params, dirs, spinup_time=0.0)

        # Read back the file and check times
        nudge_file = tmp_path / "timedepnudge.inp.999"
        content = nudge_file.read_text()
        # Should have 3 time blocks: t=0, t=50, t=100
        assert "#    0\n" in content
        assert "#    50\n" in content
        assert "#    100\n" in content
        separator_count = sum(
            1 for line in content.strip().split("\n") if line.startswith("----")
        )
        assert separator_count == 3

    def test_spinup_prepends_constant_plateau(self, tmp_path: pathlib.Path) -> None:
        """With spinup_time>0, a t=0 plateau should be prepended and times shifted."""
        namoptions_path = tmp_path / "namoptions.999"
        namoptions_path.write_text(
            "&RUN\niexpnr = 999\n/\n"
            "&DOMAIN\nitot = 4\njtot = 4\nktot = 4\n/\n"
            "&INPS\nzsize = 10.\n/\n"
        )
        from pyudales.utils.dir_utils import DirectoryPaths

        dirs = DirectoryPaths(
            case_dir=tmp_path,
            experiment_dir=tmp_path,
            experiment_name="999",
            experiment_base_dir=tmp_path,
            temp_dir=tmp_path,
            output_dir=tmp_path,
            udales_root_path=tmp_path,
            cwd=tmp_path,
            results_dir=None,
        )

        params = xarray.Dataset(
            data_vars={
                "inflow_angle": ("time", [30.0, 45.0, 60.0]),
                "velocity_magnitude": ("time", [3.0, 4.0, 5.0]),
            },
            coords={"time": [0.0, 50.0, 100.0]},
        )

        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        apply_time_varying_inflow(params, dirs, spinup_time=20.0)

        # Read back the file
        nudge_file = tmp_path / "timedepnudge.inp.999"
        content = nudge_file.read_text()

        # Should have 4 time blocks: t=0 (spinup plateau), t=20, t=70, t=120
        assert "#    0\n" in content
        assert "#    20\n" in content
        assert "#    70\n" in content
        assert "#    120\n" in content
        separator_count = sum(
            1 for line in content.strip().split("\n") if line.startswith("----")
        )
        assert separator_count == 4

        # The t=0 and t=20 blocks should have the same u,v values
        # (both use the initial angle=30, velocity=3)
        lines = content.strip().split("\n")
        # Find data lines after t=0 and t=20 markers
        t0_data = []
        t20_data = []
        current_block = None
        for line in lines:
            if "#    0" in line:
                current_block = "t0"
            elif "#    20" in line:
                current_block = "t20"
            elif "#    70" in line:
                current_block = None
            elif line.startswith("----"):
                current_block = None
            elif line.startswith("height"):
                pass
            elif current_block == "t0":
                t0_data.append(line.strip())
            elif current_block == "t20":
                t20_data.append(line.strip())

        # Both should have the same velocity profiles (initial values)
        assert len(t0_data) == len(t20_data) == 4  # ktot=4
        for t0_line, t20_line in zip(t0_data, t20_data):
            # u and v columns (last 2) should be identical
            t0_vals = t0_line.split()
            t20_vals = t20_line.split()
            assert t0_vals[3] == t20_vals[3]  # u
            assert t0_vals[4] == t20_vals[4]  # v

    def test_spinup_ntimedepnudge_includes_plateau(
        self, tmp_path: pathlib.Path
    ) -> None:
        """ntimedepnudge in namoptions should count the prepended plateau entry."""
        namoptions_path = tmp_path / "namoptions.999"
        namoptions_path.write_text(
            "&RUN\niexpnr = 999\n/\n"
            "&DOMAIN\nitot = 4\njtot = 4\nktot = 4\n/\n"
            "&INPS\nzsize = 10.\n/\n"
        )
        from pyudales.utils.dir_utils import DirectoryPaths

        dirs = DirectoryPaths(
            case_dir=tmp_path,
            experiment_dir=tmp_path,
            experiment_name="999",
            experiment_base_dir=tmp_path,
            temp_dir=tmp_path,
            output_dir=tmp_path,
            udales_root_path=tmp_path,
            cwd=tmp_path,
            results_dir=None,
        )

        params = xarray.Dataset(
            data_vars={
                "inflow_angle": ("time", [30.0, 60.0]),
                "velocity_magnitude": ("time", [3.0, 5.0]),
            },
            coords={"time": [0.0, 100.0]},
        )

        from pyudales.utils.namoptions_utils import NamoptionsFile
        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        apply_time_varying_inflow(params, dirs, spinup_time=10.0)

        nf = NamoptionsFile(namoptions_path)
        # 2 user snapshots + 1 prepended = 3 total
        assert nf.get_value_as_int("PHYSICS", "ntimedepnudge") == 3


class TestScalarParamsNudging:
    """Tests for the scalar/constant params path through apply_time_varying_inflow."""

    def _make_dirs_and_namoptions(self, tmp_path: pathlib.Path) -> "DirectoryPaths":  # type: ignore[name-defined]
        """Create minimal dirs, namoptions, prof.inp, and lscale.inp for testing."""
        from pyudales.utils.dir_utils import DirectoryPaths

        namoptions_path = tmp_path / "namoptions.999"
        namoptions_path.write_text(
            "&RUN\niexpnr = 999\nruntime = 100.\n/\n"
            "&DOMAIN\nitot = 4\njtot = 4\nktot = 4\n/\n"
            "&INPS\nzsize = 10.\nu0 = 2.0\nv0 = -2.0\n"
            "dpdx = 0.003\ndpdy = -0.003\n/\n"
        )
        # Create minimal prof.inp and lscale.inp files
        # (needed by apply_inflow_settings which is called at end of apply_time_varying_inflow)
        dz = 10.0 / 4
        heights = [0.5 * dz + k * dz for k in range(4)]
        prof_path = tmp_path / "prof.inp.999"
        with open(prof_path, "w") as f:
            f.write("# SDBL flow \n")
            f.write("# z thl qt u v tke\n")
            for z in heights:
                f.write(
                    f"{z:<20.15f} 288.000000   0.000000     2.000000     -2.000000    0.000000\n"
                )

        lscale_path = tmp_path / "lscale.inp.999"
        with open(lscale_path, "w") as f:
            f.write("# SDBL flow \n")
            f.write("# z uq vq pqx pqy wfls dqtdxls dqtdyls dqtdtls dthlrad\n")
            for z in heights:
                f.write(
                    f"{z:<20.15f} 0.000000     0.000000     0.003000000  -0.003000000 "
                    f"0.000000000     0.000000     0.000000     0.000000     0.000000000000\n"
                )

        return DirectoryPaths(
            case_dir=tmp_path,
            experiment_dir=tmp_path,
            experiment_name="999",
            experiment_base_dir=tmp_path,
            temp_dir=tmp_path,
            output_dir=tmp_path,
            udales_root_path=tmp_path,
            cwd=tmp_path,
            results_dir=None,
        )

    def test_scalar_params_creates_nudge_file(self, tmp_path: pathlib.Path) -> None:
        """Scalar params should create a timedepnudge file with 3 snapshots."""
        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        dirs = self._make_dirs_and_namoptions(tmp_path)
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": -15.0,
                "velocity_magnitude": 5.0,
                "pressure_gradient_magnitude": 0.0041912,
            }
        )

        apply_time_varying_inflow(params, dirs, spinup_time=0.0, tnudge=10.0, nnudge=0)

        nudge_file = tmp_path / "timedepnudge.inp.999"
        assert nudge_file.exists()
        content = nudge_file.read_text()
        # Should have 3 time blocks (start, mid, end+buffer)
        separator_count = sum(
            1 for line in content.strip().split("\n") if line.startswith("----")
        )
        assert separator_count == 3

    def test_scalar_params_enables_nudging_flags(self, tmp_path: pathlib.Path) -> None:
        """Scalar params should enable nudging in namoptions."""
        from pyudales.utils.namoptions_utils import NamoptionsFile
        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        dirs = self._make_dirs_and_namoptions(tmp_path)
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": -15.0,
                "velocity_magnitude": 5.0,
            }
        )

        apply_time_varying_inflow(params, dirs, spinup_time=0.0, tnudge=10.0, nnudge=0)

        nf = NamoptionsFile(tmp_path / "namoptions.999")
        assert nf.get_value("PHYSICS", "lnudge") == ".true."
        assert nf.get_value("PHYSICS", "ltimedepnudge") == ".true."
        assert nf.get_value_as_int("PHYSICS", "ntimedepnudge") == 3
        assert nf.get_value_as_int("PHYSICS", "nnudge") == 0
        assert nf.get_value_as_float("PHYSICS", "tnudge") == 10.0

    def test_scalar_params_updates_namoptions_inps(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Scalar params should update u0/v0/dpdx/dpdy in namoptions INPS."""
        from pyudales.utils.namoptions_utils import NamoptionsFile
        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        dirs = self._make_dirs_and_namoptions(tmp_path)
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": 0.0,  # u=5, v=0
                "velocity_magnitude": 5.0,
                "pressure_gradient_magnitude": 0.004,
            }
        )

        apply_time_varying_inflow(params, dirs, spinup_time=0.0, tnudge=10.0, nnudge=0)

        nf = NamoptionsFile(tmp_path / "namoptions.999")
        u0 = nf.get_value_as_float("INPS", "u0")
        v0 = nf.get_value_as_float("INPS", "v0")
        assert u0 is not None
        assert v0 is not None
        np.testing.assert_allclose(u0, 5.0, atol=1e-5)
        np.testing.assert_allclose(v0, 0.0, atol=1e-5)

    def test_scalar_params_nudge_schedule_extends_beyond_runtime(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The nudging schedule should extend slightly beyond runtime."""
        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        dirs = self._make_dirs_and_namoptions(tmp_path)
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": -15.0,
                "velocity_magnitude": 3.0,
            }
        )

        apply_time_varying_inflow(params, dirs, spinup_time=0.0)

        content = (tmp_path / "timedepnudge.inp.999").read_text()
        # The last time marker should be 101 (runtime=100 + 1 buffer)
        assert "#    101\n" in content
        # The mid-point should be 50 (runtime/2)
        assert "#    50\n" in content
        # Start at 0
        assert "#    0\n" in content


class TestScalarAndTimeVaryingEquivalence:
    """Verify that scalar and time-varying paths produce equivalent nudging setups."""

    def _make_dirs_and_namoptions(self, tmp_path: pathlib.Path) -> "DirectoryPaths":  # type: ignore[name-defined]
        """Create minimal dirs, namoptions, prof.inp, and lscale.inp for testing."""
        from pyudales.utils.dir_utils import DirectoryPaths

        namoptions_path = tmp_path / "namoptions.999"
        namoptions_path.write_text(
            "&RUN\niexpnr = 999\nruntime = 100.\n/\n"
            "&DOMAIN\nitot = 4\njtot = 4\nktot = 4\n/\n"
            "&INPS\nzsize = 10.\nu0 = 2.0\nv0 = -2.0\n"
            "dpdx = 0.003\ndpdy = -0.003\n/\n"
        )
        dz = 10.0 / 4
        heights = [0.5 * dz + k * dz for k in range(4)]
        prof_path = tmp_path / "prof.inp.999"
        with open(prof_path, "w") as f:
            f.write("# SDBL flow \n")
            f.write("# z thl qt u v tke\n")
            for z in heights:
                f.write(
                    f"{z:<20.15f} 288.000000   0.000000     2.000000     -2.000000    0.000000\n"
                )

        lscale_path = tmp_path / "lscale.inp.999"
        with open(lscale_path, "w") as f:
            f.write("# SDBL flow \n")
            f.write("# z uq vq pqx pqy wfls dqtdxls dqtdyls dqtdtls dthlrad\n")
            for z in heights:
                f.write(
                    f"{z:<20.15f} 0.000000     0.000000     0.003000000  -0.003000000 "
                    f"0.000000000     0.000000     0.000000     0.000000     0.000000000000\n"
                )

        return DirectoryPaths(
            case_dir=tmp_path,
            experiment_dir=tmp_path,
            experiment_name="999",
            experiment_base_dir=tmp_path,
            temp_dir=tmp_path,
            output_dir=tmp_path,
            udales_root_path=tmp_path,
            cwd=tmp_path,
            results_dir=None,
        )

    def test_scalar_and_time_varying_produce_same_nudging_profiles(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Given the same constant values, scalar and time-varying paths should
        produce identical u/v nudging profiles."""
        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        angle = -15.0
        vel = 5.0
        pg = 0.0041912

        # --- Scalar path ---
        scalar_dir = tmp_path / "scalar"
        scalar_dir.mkdir()
        scalar_dirs = self._make_dirs_and_namoptions(scalar_dir)
        scalar_params = xarray.Dataset(
            data_vars={
                "inflow_angle": angle,
                "velocity_magnitude": vel,
                "pressure_gradient_magnitude": pg,
            }
        )
        apply_time_varying_inflow(
            scalar_params, scalar_dirs, spinup_time=0.0, tnudge=10.0, nnudge=0
        )

        # --- Time-varying path (constant-valued) ---
        tv_dir = tmp_path / "timevarying"
        tv_dir.mkdir()
        tv_dirs = self._make_dirs_and_namoptions(tv_dir)
        n_snaps = 20
        tv_params = xarray.Dataset(
            data_vars={
                "inflow_angle": ("time", np.full(n_snaps, angle)),
                "velocity_magnitude": ("time", np.full(n_snaps, vel)),
                "pressure_gradient_magnitude": pg,
            },
            coords={"time": np.linspace(0, 100, n_snaps)},
        )
        apply_time_varying_inflow(
            tv_params, tv_dirs, spinup_time=0.0, tnudge=10.0, nnudge=0
        )

        # Both should produce nudging files
        scalar_nudge = scalar_dir / "timedepnudge.inp.999"
        tv_nudge = tv_dir / "timedepnudge.inp.999"
        assert scalar_nudge.exists()
        assert tv_nudge.exists()

        # Extract u/v values from data rows (skipping headers/separators/time markers)
        def extract_uv_from_nudge_file(path: pathlib.Path) -> list[tuple[float, float]]:
            """Extract (u, v) from all data rows in a timedepnudge file."""
            uv_pairs = []
            for line in path.read_text().strip().split("\n"):
                line = line.strip()
                if (
                    line.startswith("height")
                    or line.startswith("#")
                    or line.startswith("----")
                    or not line
                ):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    uv_pairs.append((float(parts[3]), float(parts[4])))
            return uv_pairs

        scalar_uv = extract_uv_from_nudge_file(scalar_nudge)
        tv_uv = extract_uv_from_nudge_file(tv_nudge)

        # All u/v values should be constant and match between both files
        expected_u, expected_v = angle_to_velocity(angle, vel)
        for u, v in scalar_uv:
            np.testing.assert_allclose(u, expected_u, atol=1e-4)
            np.testing.assert_allclose(v, expected_v, atol=1e-4)
        for u, v in tv_uv:
            np.testing.assert_allclose(u, expected_u, atol=1e-4)
            np.testing.assert_allclose(v, expected_v, atol=1e-4)

    def test_scalar_and_time_varying_produce_same_namoptions(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Both paths should enable the same nudging flags in namoptions."""
        from pyudales.utils.namoptions_utils import NamoptionsFile
        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        angle = -15.0
        vel = 5.0

        # --- Scalar path ---
        scalar_dir = tmp_path / "scalar"
        scalar_dir.mkdir()
        scalar_dirs = self._make_dirs_and_namoptions(scalar_dir)
        scalar_params = xarray.Dataset(
            data_vars={"inflow_angle": angle, "velocity_magnitude": vel}
        )
        apply_time_varying_inflow(
            scalar_params, scalar_dirs, spinup_time=0.0, tnudge=10.0, nnudge=0
        )

        # --- Time-varying path ---
        tv_dir = tmp_path / "timevarying"
        tv_dir.mkdir()
        tv_dirs = self._make_dirs_and_namoptions(tv_dir)
        tv_params = xarray.Dataset(
            data_vars={
                "inflow_angle": ("time", np.full(5, angle)),
                "velocity_magnitude": ("time", np.full(5, vel)),
            },
            coords={"time": np.linspace(0, 100, 5)},
        )
        apply_time_varying_inflow(
            tv_params, tv_dirs, spinup_time=0.0, tnudge=10.0, nnudge=0
        )

        scalar_nf = NamoptionsFile(scalar_dir / "namoptions.999")
        tv_nf = NamoptionsFile(tv_dir / "namoptions.999")

        # Both should enable nudging
        assert scalar_nf.get_value("PHYSICS", "lnudge") == ".true."
        assert tv_nf.get_value("PHYSICS", "lnudge") == ".true."
        assert scalar_nf.get_value("PHYSICS", "ltimedepnudge") == ".true."
        assert tv_nf.get_value("PHYSICS", "ltimedepnudge") == ".true."
        assert scalar_nf.get_value_as_float(
            "PHYSICS", "tnudge"
        ) == tv_nf.get_value_as_float("PHYSICS", "tnudge")
        assert scalar_nf.get_value_as_int(
            "PHYSICS", "nnudge"
        ) == tv_nf.get_value_as_int("PHYSICS", "nnudge")

        # Both should set the same INPS u0/v0
        scalar_u0 = scalar_nf.get_value_as_float("INPS", "u0")
        tv_u0 = tv_nf.get_value_as_float("INPS", "u0")
        scalar_v0 = scalar_nf.get_value_as_float("INPS", "v0")
        tv_v0 = tv_nf.get_value_as_float("INPS", "v0")
        np.testing.assert_allclose(scalar_u0, tv_u0, atol=1e-5)
        np.testing.assert_allclose(scalar_v0, tv_v0, atol=1e-5)

    def test_scalar_and_time_varying_produce_same_prof_and_lscale(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Both paths should produce identical prof.inp and lscale.inp files."""
        from pyudales.utils.nudging_utils import apply_time_varying_inflow

        angle = -25.0
        vel = 3.0
        pg = 0.0041912

        # --- Scalar path ---
        scalar_dir = tmp_path / "scalar"
        scalar_dir.mkdir()
        scalar_dirs = self._make_dirs_and_namoptions(scalar_dir)
        scalar_params = xarray.Dataset(
            data_vars={
                "inflow_angle": angle,
                "velocity_magnitude": vel,
                "pressure_gradient_magnitude": pg,
            }
        )
        apply_time_varying_inflow(
            scalar_params, scalar_dirs, spinup_time=0.0, tnudge=10.0, nnudge=0
        )

        # --- Time-varying path ---
        tv_dir = tmp_path / "timevarying"
        tv_dir.mkdir()
        tv_dirs = self._make_dirs_and_namoptions(tv_dir)
        tv_params = xarray.Dataset(
            data_vars={
                "inflow_angle": ("time", np.full(10, angle)),
                "velocity_magnitude": ("time", np.full(10, vel)),
                "pressure_gradient_magnitude": pg,
            },
            coords={"time": np.linspace(0, 100, 10)},
        )
        apply_time_varying_inflow(
            tv_params, tv_dirs, spinup_time=0.0, tnudge=10.0, nnudge=0
        )

        # Compare prof.inp files
        scalar_prof = (scalar_dir / "prof.inp.999").read_text()
        tv_prof = (tv_dir / "prof.inp.999").read_text()
        assert scalar_prof == tv_prof, "prof.inp files should be identical"

        # Compare lscale.inp files
        scalar_lscale = (scalar_dir / "lscale.inp.999").read_text()
        tv_lscale = (tv_dir / "lscale.inp.999").read_text()
        assert scalar_lscale == tv_lscale, "lscale.inp files should be identical"


class TestMergeParamsWithTimeDimension:
    def test_merge_time_varying_with_scalar_defaults(self) -> None:
        from pyudales.utils.params_utils import merge_params

        defaults = xarray.Dataset(
            data_vars={
                "inflow_angle": 45.0,
                "velocity_magnitude": 3.0,
                "pressure_gradient_magnitude": 0.004,
            }
        )
        new = xarray.Dataset(
            data_vars={
                "inflow_angle": ("time", [30.0, 60.0]),
                "velocity_magnitude": ("time", [3.0, 5.0]),
            },
            coords={"time": [0.0, 100.0]},
        )

        merged = merge_params(defaults, new)
        assert merged is not None
        assert "time" in merged["inflow_angle"].dims
        assert "time" in merged["velocity_magnitude"].dims
        # pressure_gradient_magnitude should remain scalar
        assert merged["pressure_gradient_magnitude"].dims == ()

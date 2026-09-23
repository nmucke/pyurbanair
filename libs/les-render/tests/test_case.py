"""Tests for les_render.case discovery and case handling."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pytest
import trimesh
import xarray as xr
import yaml
from les_render.case import Case, discover_case, mesh_from_blanking


def _make_tiny_state_nc(
    path: pathlib.Path,
    geometry_stl: str | None = None,
    include_blanking: bool = True,
) -> None:
    """Write a tiny state.nc with u, v, w, optional blanking and metadata."""
    time = np.arange(2)
    zt = np.arange(4, dtype=float)
    yt = np.arange(8, dtype=float)
    xt = np.arange(16, dtype=float)

    ds = xr.Dataset(
        {
            "u": (("time", "zt", "yt", "xt"), np.random.randn(2, 4, 8, 16)),
            "v": (("time", "zt", "yt", "xt"), np.random.randn(2, 4, 8, 16)),
            "w": (("time", "zt", "yt", "xt"), np.random.randn(2, 4, 8, 16)),
        },
        coords={"time": time, "zt": zt, "yt": yt, "xt": xt},
    )

    if include_blanking:
        # 1 = solid; create a small box in the middle (3D, no time dimension)
        blanking = np.zeros((4, 8, 16), dtype=np.int8)
        blanking[1:3, 3:5, 5:7] = 1
        ds["blanking"] = (("zt", "yt", "xt"), blanking)

    if geometry_stl is not None:
        ds.attrs["geometry_stl"] = geometry_stl

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _make_tiny_stl(path: pathlib.Path, name: str = "geometry") -> None:
    """Write a tiny STL file."""
    mesh = trimesh.creation.box(extents=(2, 2, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def _make_tiny_params_nc(path: pathlib.Path) -> None:
    """Write a tiny params.nc with inflow_angle."""
    ds = xr.Dataset(
        {
            "inflow_angle": (("time",), np.array([0.0, 45.0])),
            "velocity_magnitude": (("time",), np.array([5.0, 6.0])),
        },
        coords={"time": np.array([0.0, 100.0])},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


class TestDiscoverCase:
    """discover_case() resolution tests."""

    def test_folder_with_state_nc_and_stl_and_params(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Folder with state.nc + one .stl + params.nc resolves all three."""
        _make_tiny_state_nc(tmp_path / "state.nc")
        _make_tiny_stl(tmp_path / "building.stl")
        _make_tiny_params_nc(tmp_path / "params.nc")

        case = discover_case(tmp_path)

        assert case.state_path == tmp_path / "state.nc"
        assert case.geometry_path == tmp_path / "building.stl"
        assert case.params_path == tmp_path / "params.nc"

    def test_single_sample_nc_no_state_nc(self, tmp_path: pathlib.Path) -> None:
        """Folder with a single sample_0001.nc (no state.nc) resolves it."""
        _make_tiny_state_nc(tmp_path / "sample_0001.nc")

        case = discover_case(tmp_path)

        assert case.state_path == tmp_path / "sample_0001.nc"

    def test_multiple_state_files_raises(self, tmp_path: pathlib.Path) -> None:
        """Two state-like nc files raises FileNotFoundError."""
        _make_tiny_state_nc(tmp_path / "sample_0001.nc")
        _make_tiny_state_nc(tmp_path / "sample_0002.nc")

        with pytest.raises(FileNotFoundError, match="exactly one state file"):
            discover_case(tmp_path)

    def test_training_layout_resolves_all(self, tmp_path: pathlib.Path) -> None:
        """Training data layout: state/<split>/sample_*.nc resolves all three."""
        dataset_root = tmp_path
        state_file = dataset_root / "state" / "train" / "sample_0000.nc"
        _make_tiny_state_nc(state_file, geometry_stl="geometry.stl")

        geom_file = dataset_root / "geometries" / "geometry.stl"
        _make_tiny_stl(geom_file)

        params_file = dataset_root / "param" / "train" / "sample_0000.nc"
        _make_tiny_params_nc(params_file)

        case = discover_case(state_file)

        assert case.state_path == state_file
        assert case.geometry_path == geom_file
        assert case.params_path == params_file

    def test_no_geometry_returns_none(self, tmp_path: pathlib.Path) -> None:
        """No STL or geometry_stl attr → geometry_path is None."""
        _make_tiny_state_nc(tmp_path / "state.nc")

        case = discover_case(tmp_path)

        assert case.geometry_path is None

    def test_no_params_returns_none(self, tmp_path: pathlib.Path) -> None:
        """No params.nc or inflow_angle nc → params_path is None."""
        _make_tiny_state_nc(tmp_path / "state.nc")

        case = discover_case(tmp_path)

        assert case.params_path is None

    def test_case_name_from_state_stem(self, tmp_path: pathlib.Path) -> None:
        """Case name taken from state file stem (unless 'state' then folder name)."""
        _make_tiny_state_nc(tmp_path / "sample_0001.nc")
        case = discover_case(tmp_path)
        assert case.name == "sample_0001"

        # state.nc → use folder name
        _make_tiny_state_nc(tmp_path / "state.nc")
        case = discover_case(tmp_path)
        assert case.name == tmp_path.name

    def test_render_yaml_overrides_loaded(self, tmp_path: pathlib.Path) -> None:
        """render.yaml overrides are loaded into case.overrides."""
        _make_tiny_state_nc(tmp_path / "state.nc")
        render_yaml = tmp_path / "render.yaml"
        render_yaml.write_text(yaml.dump({"look": "daylight", "duration": 20}))

        case = discover_case(tmp_path)

        assert case.overrides == {"look": "daylight", "duration": 20}

    def test_render_yaml_missing_or_empty(self, tmp_path: pathlib.Path) -> None:
        """Missing or empty render.yaml → empty overrides."""
        _make_tiny_state_nc(tmp_path / "state.nc")
        case = discover_case(tmp_path)
        assert case.overrides == {}

        # Empty file
        render_yaml = tmp_path / "render.yaml"
        render_yaml.write_text("")
        case = discover_case(tmp_path)
        assert case.overrides == {}


class TestMeshFromBlanking:
    """mesh_from_blanking() tests."""

    def test_blanking_mesh_bounds(self, tmp_path: pathlib.Path) -> None:
        """mesh_from_blanking builds from solid voxels; bounds match solid cells."""
        # Create state with a small blanking block at a known location.
        time = np.arange(1)
        zt = np.arange(4, dtype=float)
        yt = np.arange(8, dtype=float)
        xt = np.arange(16, dtype=float)

        ds = xr.Dataset(
            {
                "u": (("time", "zt", "yt", "xt"), np.zeros((1, 4, 8, 16))),
                "v": (("time", "zt", "yt", "xt"), np.zeros((1, 4, 8, 16))),
                "w": (("time", "zt", "yt", "xt"), np.zeros((1, 4, 8, 16))),
                "blanking": (
                    ("zt", "yt", "xt"),
                    _blanking_with_box(
                        4, 8, 16, box_z=(1, 3), box_y=(2, 4), box_x=(4, 6)
                    ),
                ),
            },
            coords={"time": time, "zt": zt, "yt": yt, "xt": xt},
        )

        mesh = mesh_from_blanking(ds)
        bounds = mesh.bounds

        # Mesh should cover the solid voxels (Python slicing: box_x[0]:box_x[1] sets indices box_x[0] to box_x[1]-1).
        # Each voxel coordinate is an integer (0, 1, 2, ...) and the box has extents 1 unit,
        # so box at index i goes from (i-0.5) to (i+0.5).
        # box_x=(4, 6) -> indices [4, 5] -> x bounds [3.5, 5.5]
        # box_y=(2, 4) -> indices [2, 3] -> y bounds [1.5, 3.5]
        # box_z=(1, 3) -> indices [1, 2] -> z bounds [0.5, 2.5]
        assert bounds[0, 0] == pytest.approx(3.5, abs=0.1)  # x_min
        assert bounds[1, 0] == pytest.approx(5.5, abs=0.1)  # x_max
        assert bounds[0, 1] == pytest.approx(1.5, abs=0.1)  # y_min
        assert bounds[1, 1] == pytest.approx(3.5, abs=0.1)  # y_max
        assert bounds[0, 2] == pytest.approx(0.5, abs=0.1)  # z_min
        assert bounds[1, 2] == pytest.approx(2.5, abs=0.1)  # z_max


def _blanking_with_box(
    nz: int,
    ny: int,
    nx: int,
    box_z: tuple[int, int],
    box_y: tuple[int, int],
    box_x: tuple[int, int],
) -> np.ndarray:
    """Create a blanking array (1 = solid) with a box of given extent."""
    blanking = np.zeros((nz, ny, nx), dtype=np.int8)
    blanking[box_z[0] : box_z[1], box_y[0] : box_y[1], box_x[0] : box_x[1]] = 1
    return blanking


class TestCaseBuildings:
    """case.buildings() tests."""

    def test_buildings_from_stl(self, tmp_path: pathlib.Path) -> None:
        """buildings() loads from geometry_path STL."""
        _make_tiny_state_nc(tmp_path / "state.nc")
        _make_tiny_stl(tmp_path / "building.stl")

        case = discover_case(tmp_path)
        mesh = case.buildings()

        assert isinstance(mesh, trimesh.Trimesh)
        # Loaded mesh should have the box shape (is_watertight may be False for a simple box).
        assert mesh.vertices.size > 0

    def test_buildings_from_blanking(self, tmp_path: pathlib.Path) -> None:
        """buildings() uses blanking when no STL."""
        _make_tiny_state_nc(tmp_path / "state.nc", include_blanking=True)

        case = discover_case(tmp_path)
        assert case.geometry_path is None

        mesh = case.buildings()
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.vertices.size > 0


class TestMakeRenderCaseScript:
    """Integration test for make_render_case.py script."""

    def test_end_to_end_script_creates_symlinks(self, tmp_path: pathlib.Path) -> None:
        """Script creates symlinked render case folder."""
        # Set up input case.
        input_dir = tmp_path / "input_case"
        _make_tiny_state_nc(input_dir / "state.nc")
        _make_tiny_stl(input_dir / "building.stl")
        _make_tiny_params_nc(input_dir / "params.nc")

        # Run script.
        output_dir = tmp_path / "output_case"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
import sys
sys.path.insert(0, '{pathlib.Path(__file__).parent.parent.parent / "src"}')
sys.path.insert(0, '{pathlib.Path(__file__).parent.parent.parent.parent / "pyurbanair" / "src"}')

from scripts.visualization.make_render_case import main
import sys

sys.argv = ['make_render_case.py', '{input_dir / "state.nc"}', '{output_dir}']
main()
""",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(f"Script failed with return code {result.returncode}")

        # Check output folder.
        assert (output_dir / "state.nc").is_symlink()
        assert (output_dir / "building.stl").is_symlink()
        assert (output_dir / "params.nc").is_symlink()

        # Discover case from output folder.
        case = discover_case(output_dir)
        assert case.state_path == output_dir / "state.nc"
        assert case.geometry_path == output_dir / "building.stl"
        assert case.params_path == output_dir / "params.nc"

    def test_script_with_render_yaml(self, tmp_path: pathlib.Path) -> None:
        """Script writes render.yaml template with --render-yaml."""
        input_dir = tmp_path / "input_case"
        _make_tiny_state_nc(input_dir / "state.nc")

        output_dir = tmp_path / "output_case"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"""
import sys
sys.path.insert(0, '{pathlib.Path(__file__).parent.parent.parent / "src"}')
sys.path.insert(0, '{pathlib.Path(__file__).parent.parent.parent.parent / "pyurbanair" / "src"}')

from scripts.visualization.make_render_case import main
import sys

sys.argv = ['make_render_case.py', '{input_dir / "state.nc"}', '{output_dir}', '--render-yaml']
main()
""",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            raise RuntimeError(f"Script failed with return code {result.returncode}")

        render_yaml = output_dir / "render.yaml"
        assert render_yaml.exists()
        content = render_yaml.read_text()
        assert "render_preset" in content
        assert "duration" in content

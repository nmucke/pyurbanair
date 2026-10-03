"""The IBM geometry compiler uses the same macOS linker as the solver build."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh
from pyudales.python_udgeom import ibm


def test_ibm_compile_receives_native_linker_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = tmp_path / "tools"
    (tools / "IBM/IBM_preproc_fortran").mkdir(parents=True)
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    grid = np.array([0.0, 1.0])
    native_env = {"LDFLAGS": "-B/usr/bin/", "PATH": "/usr/bin"}
    monkeypatch.setattr(ibm, "_build_environment", lambda compiler: native_env)
    monkeypatch.setattr(ibm.os, "chdir", lambda path: None)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def capture(command: list[str], **kwargs: Any) -> None:
        calls.append((command, kwargs))
        raise RuntimeError("compile intercepted")

    monkeypatch.setattr(ibm.subprocess, "run", capture)
    with pytest.raises(RuntimeError, match="compile intercepted"):
        ibm.write_ibm_files_using_fortran(
            TR=mesh,
            xgrid_u=grid,
            ygrid_u=grid,
            zgrid_u=grid,
            xgrid_v=grid,
            ygrid_v=grid,
            zgrid_v=grid,
            xgrid_w=grid,
            ygrid_w=grid,
            zgrid_w=grid,
            xgrid_c=grid,
            ygrid_c=grid,
            zgrid_c=grid,
            fpath=str(experiment),
            dx=1.0,
            dy=1.0,
            itot=2,
            jtot=2,
            ktot=2,
            stl_ground=False,
            diag_neighbs=False,
            periodic_x=False,
            periodic_y=False,
            toolsdir=str(tools),
        )

    command, kwargs = calls[0]
    assert command[0] == "gfortran"
    assert "-B/usr/bin/" in command
    assert kwargs["env"] == native_env
    assert kwargs["cwd"] == str(experiment)


@pytest.mark.parametrize(  # type: ignore[misc]
    "nz, refused", [(4, True), (5, True), (6, False)]
)
def test_ibm_refuses_roofs_in_the_top_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nz: int, refused: bool
) -> None:
    """A 10 m roof under a lid of nz 2.5 m cells: upstream reads past its arrays
    unless the roof stays 1.5 cells under the lid (checked with -fcheck=bounds)."""
    tools = tmp_path / "tools"
    (tools / "IBM/IBM_preproc_fortran").mkdir(parents=True)
    roof = trimesh.Trimesh(
        vertices=[[0, 0, 10], [1, 0, 10], [0, 1, 10]], faces=[[0, 1, 2]], process=False
    )
    zh = np.arange(nz) * 2.5
    monkeypatch.setattr(ibm.os, "chdir", lambda path: None)

    def compile_intercepted(command: list[str], **kwargs: Any) -> None:
        raise RuntimeError("compile intercepted")

    monkeypatch.setattr(ibm.subprocess, "run", compile_intercepted)
    grid = np.array([0.5, 1.5])
    expected = (ValueError, "1.5 cells") if refused else (RuntimeError, "intercepted")
    with pytest.raises(expected[0], match=expected[1]):
        ibm.write_ibm_files_using_fortran(
            TR=roof,
            xgrid_u=grid,
            ygrid_u=grid,
            zgrid_u=zh + 1.25,
            xgrid_v=grid,
            ygrid_v=grid,
            zgrid_v=zh + 1.25,
            xgrid_w=grid,
            ygrid_w=grid,
            zgrid_w=zh,
            xgrid_c=grid,
            ygrid_c=grid,
            zgrid_c=zh + 1.25,
            fpath=str(tmp_path),
            dx=1.0,
            dy=1.0,
            itot=2,
            jtot=2,
            ktot=nz,
            stl_ground=False,
            diag_neighbs=False,
            periodic_x=False,
            periodic_y=False,
            toolsdir=str(tools),
        )

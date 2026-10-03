"""The environment PALM's installer runs in: pixi libraries, local HOME, Apple ld."""

from pathlib import Path

import pypalm
import pytest


def _fake_mpif90(directory: Path) -> None:
    """An mpif90 whose --showme output mimics conda's Open MPI on macOS."""
    script = directory / "mpif90"
    script.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  --showme:compile) echo '-I/env/include -pthread' ;;\n"
        "  --showme:link) echo '-I/env/include -pthread -L/env/lib "
        "-Wl,-rpath,/env/lib -lmpi_mpifh -lmpi' ;;\n"
        "esac\n"
    )
    script.chmod(0o755)


def test_macos_links_keep_the_wrapper_flags_and_use_apple_ld(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_mpif90(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setenv("CONDA_PREFIX", "/env")
    monkeypatch.setenv("LDFLAGS", "-Wl,-dead_strip")
    monkeypatch.setattr(pypalm, "apple_linker_flags", lambda *args: ["-B/usr/bin/"])

    env = pypalm._install_environment()

    pad = "-Wl,-headerpad_max_install_names"
    assert env["LDFLAGS"] == f"-Wl,-dead_strip -B/usr/bin/ {pad}"
    # OMPI_LDFLAGS replaces the wrapper's linker flags: -L and rpath must stay.
    assert env["OMPI_LDFLAGS"] == f"-L/env/lib -Wl,-rpath,/env/lib -B/usr/bin/ {pad}"


def test_pixi_env_first_and_home_inside_the_palm_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONDA_PREFIX", "/env")
    monkeypatch.setenv("CMAKE_PREFIX_PATH", "/opt/other")
    monkeypatch.delenv("LDFLAGS", raising=False)
    monkeypatch.delenv("OMPI_LDFLAGS", raising=False)
    monkeypatch.setattr(pypalm, "apple_linker_flags", lambda *args: [])

    env = pypalm._install_environment()

    assert env["CMAKE_PREFIX_PATH"].split(":") == ["/env", "/opt/other"]
    assert env["HOME"] == str(pypalm.PALM_MODEL_SYSTEM_PATH)
    assert "LDFLAGS" not in env and "OMPI_LDFLAGS" not in env

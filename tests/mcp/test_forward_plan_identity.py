"""Prepared uDALES plans bind builder resources and caller toolchain choices."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from mcp_server.jobs.preparation import (
    PreparationService,
    code_identity,
    verify_worker_toolchain_environment,
)
from mcp_server.jobs.supervisor import Supervisor, _forward_worker_environment


@pytest.mark.parametrize(  # type: ignore[misc]
    "relative",
    [
        "libs/pyudales/shell_scripts/build_udales_macos.sh",
        "activation_scripts/cuda_activation.sh",
        "libs/pyudales/src/pyudales/solver_extensions/discrepancy/manifest.json",
        "libs/pyudales/src/pyudales/solver_extensions/discrepancy/discrepancy.patch",
        "libs/pyudales/src/pyudales/solver_extensions/discrepancy/modsgsdiscrepancy.f90",
    ],
)
def test_udales_executable_resources_change_code_identity(
    checkout: Path, relative: str
) -> None:
    resource = checkout / relative
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_bytes(b"original")
    before = code_identity(checkout, "pyudales")
    resource.write_bytes(b"changed")
    assert (
        code_identity(checkout, "pyudales")["source_digest"] != before["source_digest"]
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "relative",
    [
        "libs/pyudales/shell_scripts/build_preprocessing_macos.sh",
        "libs/pyudales/src/pyudales/solver_extensions/discrepancy/discrepancy.patch",
    ],
)
def test_udales_resource_change_invalidates_prepared_plan(
    checkout: Path, tmp_path: Path, relative: str
) -> None:
    resource = checkout / relative
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_bytes(b"original")
    service = PreparationService(checkout, tmp_path / "store")
    plan = service.prepare()
    resource.write_bytes(b"changed")
    with pytest.raises(ValueError, match="code or configuration changed"):
        service.verify(plan["plan_id"])


def test_udales_build_environment_invalidates_prepared_plan(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    choices = {
        "FC": "mpif90",
        "FFLAGS": "-O2",
        "CC": "cc",
        "CMAKE_PREFIX_PATH": "/chosen/dependencies",
    }
    for key, value in choices.items():
        monkeypatch.setenv(key, value)
    service = PreparationService(checkout, tmp_path / "store")
    plan = service.prepare()
    assert all(
        plan["provenance"]["environment"][key] == value
        for key, value in choices.items()
    )
    for key, value in choices.items():
        monkeypatch.setenv(key, value + "-changed")
        with pytest.raises(ValueError, match="execution environment changed"):
            service.verify(plan["plan_id"])
        monkeypatch.setenv(key, value)
    assert service.verify(plan["plan_id"])["digest"] == plan["digest"]


def test_supervisor_replays_saved_overrides_and_clears_stale_values() -> None:
    plan = {
        "backend": "pyudales",
        "provenance": {
            "environment": {
                "FC": "chosen-mpif90",
                "FFLAGS": "-O2",
                "PATH": "/client/path",
            }
        },
    }
    inherited = {
        "FC": "stale-mpif90",
        "CC": "stale-cc",
        "PATH": "/supervisor/path",
        "CONDA_PREFIX": "/supervisor/prefix",
    }
    worker = _forward_worker_environment(inherited, plan)
    assert worker["FC"] == "chosen-mpif90"
    assert worker["FFLAGS"] == "-O2"
    assert "CC" not in worker
    assert worker["PATH"] == "/client/path"
    assert worker["CONDA_PREFIX"] == inherited["CONDA_PREFIX"]
    verify_worker_toolchain_environment(plan, worker)
    worker["FC"] = "pixi-overrode-compiler"
    with pytest.raises(ValueError, match="uDALES build environment changed.*FC"):
        verify_worker_toolchain_environment(plan, worker)


@pytest.mark.parametrize(  # type: ignore[misc]
    "key",
    ["FC", "FFLAGS", "CMAKE_TOOLCHAIN_FILE", "SDKROOT", "LD_LIBRARY_PATH"],
)
def test_worker_rejects_build_values_added_by_activation(key: str) -> None:
    plan = {"backend": "pyudales", "provenance": {"environment": {}}}
    with pytest.raises(ValueError, match=f"uDALES build environment changed.*{key}"):
        verify_worker_toolchain_environment(plan, {key: "activation-value"})


def test_worker_accepts_only_recorded_cuda_activation_library(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NVHPC_INSTALL_BASE", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    compiler = (
        checkout / ".pixi/envs/cuda/.nvhpc/Linux_x86_64/24.5/compilers/bin/nvfortran"
    )
    compiler.parent.mkdir(parents=True)
    compiler.write_text("fake compiler")
    compiler.chmod(0o700)
    library = compiler.parent.parent / "lib"
    library.mkdir()
    plan = PreparationService(checkout, tmp_path / "store").prepare(environment="cuda")
    assert plan["provenance"]["cuda_activation_library"] == str(library)
    worker = dict(plan["provenance"]["environment"])
    worker["LD_LIBRARY_PATH"] = str(library)
    verify_worker_toolchain_environment(plan, worker)
    worker["LD_LIBRARY_PATH"] = f"/external/lib:{library}"
    with pytest.raises(
        ValueError, match="uDALES build environment changed.*LD_LIBRARY_PATH"
    ):
        verify_worker_toolchain_environment(plan, worker)


def test_supervisor_launches_child_with_saved_build_environment(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FFLAGS", "-O2")
    monkeypatch.delenv("CC", raising=False)
    store = tmp_path / "plans"
    plan = PreparationService(checkout, store).prepare()
    monkeypatch.setenv("FFLAGS", "-O0")
    monkeypatch.setenv("CC", "stale-cc")
    monkeypatch.setenv("PATH", "/stale/supervisor/path")
    output = tmp_path / "child-environment.json"
    script = (
        "import json,os,sys; "
        "json.dump({key:os.environ.get(key) for key in ('FFLAGS','CC','PATH')}, "
        "open(sys.argv[1],'w'))"
    )
    supervisor = Supervisor(
        checkout,
        tmp_path / "queue",
        command_factory=lambda _job: [sys.executable, "-c", script, str(output)],
    )
    payload = {
        "kind": "forward",
        "plan_id": plan["plan_id"],
        "plan_digest": plan["digest"],
        "store_root": str(store),
        "environment": "dev",
    }
    job = supervisor.registry.submit(payload, "saved-environment")
    supervisor._start(job)
    child = supervisor.children[job["id"]]
    assert child.wait(timeout=10) == 0
    assert json.loads(output.read_text()) == {
        "FFLAGS": "-O2",
        "CC": None,
        "PATH": plan["provenance"]["environment"]["PATH"],
    }


def test_preparation_records_caller_path_but_worker_uses_pixi_path(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = PreparationService(checkout, tmp_path / "store")
    plan = service.prepare()
    assert plan["provenance"]["environment"]["PATH"] == os.environ["PATH"]
    monkeypatch.setenv("PATH", os.environ["PATH"] + ":/different/path")
    with pytest.raises(ValueError, match="execution environment changed"):
        service.verify(plan["plan_id"])

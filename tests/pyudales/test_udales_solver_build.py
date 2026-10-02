"""Reproducible solver preparation, exercised without Fortran or network access."""

from __future__ import annotations

import importlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pyudales.utils import solver_build as build


@pytest.fixture  # type: ignore[misc]
def fake_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        build, "_environment_identity", lambda env: {"compiler": "test"}
    )
    monkeypatch.setattr(build, "_source_repository", lambda *_: tmp_path)

    def export(_repository: Path, destination: Path) -> None:
        (destination / "tools").mkdir(parents=True)
        (destination / "src").mkdir()
        (destination / "src" / "native.f90").write_text("pristine")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(command)
        if command[0] == "bash" and command[1].endswith("build_udales_macos.sh"):
            executable = Path(command[-1]) / "u-dales"
            executable.parent.mkdir()
            executable.write_bytes(b"native executable")
            executable.chmod(0o755)
        if command[0] == "bash" and command[1].endswith("build_preprocessing_macos.sh"):
            executable = Path(command[-1]) / "tools/View3D/build/src/view3d"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"preprocessor")
            executable.chmod(0o755)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(build, "_export_source", export)
    monkeypatch.setattr(build, "_run", run)
    return calls


def test_import_has_no_subprocess_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Import attempted a build/download")

    monkeypatch.setattr(subprocess, "run", forbidden)
    import pyudales

    importlib.reload(pyudales)


def test_reuse_and_executable_verification(
    tmp_path: Path, fake_build: list[list[str]]
) -> None:
    executable = build.prepare_solver(cache_dir=tmp_path)
    assert build.solver_source_dir(executable).joinpath("tools").is_dir()
    assert build.prepare_solver(cache_dir=tmp_path) == executable
    assert len(fake_build) == 1
    executable.write_bytes(b"corrupt executable")
    assert build.prepare_solver(cache_dir=tmp_path) == executable
    assert len(fake_build) == 2
    executable.unlink()
    build.prepare_solver(cache_dir=tmp_path)
    assert len(fake_build) == 3


def test_environment_invalidates_cache(
    tmp_path: Path, fake_build: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = build.prepare_solver(cache_dir=tmp_path)
    monkeypatch.setattr(
        build, "_environment_identity", lambda env: {"compiler": "changed"}
    )
    assert build.prepare_solver(cache_dir=tmp_path) != first
    assert len(fake_build) == 2


def test_macos_conda_build_uses_system_linker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "env"
    compiler = prefix / "bin/mpif90"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build.shutil, "which", lambda *args, **kwargs: str(compiler))
    monkeypatch.setenv("CONDA_PREFIX", str(prefix))
    monkeypatch.setenv("LDFLAGS", "-Wl,-dead_strip")

    assert build._build_environment()["LDFLAGS"] == ("-Wl,-dead_strip -B/usr/bin/")
    monkeypatch.setenv("LDFLAGS", "-B/custom/linker -Wl,-dead_strip")
    assert build._build_environment()["LDFLAGS"] == ("-B/custom/linker -Wl,-dead_strip")


def test_macos_pixi_build_without_conda_prefix_uses_system_linker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler = tmp_path / ".pixi/envs/dev/bin/mpif90"
    compiler.parent.mkdir(parents=True)
    compiler.touch()
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build.shutil, "which", lambda *args, **kwargs: str(compiler))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("LDFLAGS", raising=False)

    assert build._build_environment()["LDFLAGS"] == "-B/usr/bin/"


def test_linux_build_does_not_change_linker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    monkeypatch.setenv("LDFLAGS", "-Wl,--as-needed")
    assert build._build_environment()["LDFLAGS"] == "-Wl,--as-needed"


def test_both_native_builds_receive_the_linker_environment(
    tmp_path: Path, fake_build: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = build._run
    environments: list[dict[str, str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        environments.append(kwargs["env"])
        return original(command, **kwargs)

    monkeypatch.setattr(build, "_build_environment", lambda: {"LDFLAGS": "-B/usr/bin/"})
    monkeypatch.setattr(build, "_run", run)
    build.prepare_solver(cache_dir=tmp_path, prepare_tools=True)
    assert len(fake_build) == 2
    assert environments == [{"LDFLAGS": "-B/usr/bin/"}] * 2


def test_macos_sdk_contents_enter_the_build_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = tmp_path / "MacOSX.sdk"
    libsystem = sdk / "usr/lib/libSystem.tbd"
    libsystem.parent.mkdir(parents=True)
    libsystem.write_bytes(b"first")
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    env = {"SDKROOT": str(sdk), "PATH": ""}
    first = build._environment_identity(env)
    libsystem.write_bytes(b"second")
    second = build._environment_identity(env)
    assert first["macos_sdk"]["path"] == str(sdk)
    assert (
        first["macos_sdk"]["libsystem_sha256"]
        != second["macos_sdk"]["libsystem_sha256"]
    )


def test_concurrent_builds_publish_once(
    tmp_path: Path, fake_build: list[list[str]]
) -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(
            pool.map(lambda _: build.prepare_solver(cache_dir=tmp_path), range(4))
        )
    assert len(set(paths)) == 1
    assert len(fake_build) == 1
    assert not list(tmp_path.glob(".*-*"))


def test_failure_never_publishes_and_retry_recovers(
    tmp_path: Path, fake_build: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = build._run

    def fail(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        original(command, **kwargs)
        raise RuntimeError("compile failed")

    monkeypatch.setattr(build, "_run", fail)
    with pytest.raises(RuntimeError, match="compile failed"):
        build.prepare_solver(cache_dir=tmp_path)
    assert not list(tmp_path.glob("*/capability.json"))
    monkeypatch.setattr(build, "_run", original)
    assert build.prepare_solver(cache_dir=tmp_path).is_file()


def test_capability_manifest_is_required(
    tmp_path: Path, fake_build: list[list[str]]
) -> None:
    executable = build.prepare_solver(cache_dir=tmp_path)
    manifest = executable.parent.parent / "capability.json"
    manifest.write_text("{}")
    build.prepare_solver(cache_dir=tmp_path)
    assert len(fake_build) == 2


def test_preprocessing_is_explicit_and_verified(
    tmp_path: Path, fake_build: list[list[str]]
) -> None:
    executable = build.prepare_solver(cache_dir=tmp_path, prepare_tools=True)
    tool = build.solver_source_dir(executable) / "tools/View3D/build/src/view3d"
    assert tool.is_file()
    assert len(fake_build) == 2
    tool.unlink()
    build.prepare_solver(cache_dir=tmp_path, prepare_tools=True)
    assert len(fake_build) == 4


def test_resource_hash_tampering_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = tmp_path / "solver_extensions/discrepancy"
    resource.mkdir(parents=True)
    (resource / "patch").write_bytes(b"unexpected")
    (resource / "manifest.json").write_text(
        json.dumps(
            {
                "upstream_commit": build.UPSTREAM_COMMIT,
                "capability": build.CAPABILITY,
                "resources": {"patch": build._sha(b"expected")},
            }
        )
    )
    monkeypatch.setattr(build.resources, "files", lambda _: tmp_path)
    with pytest.raises(ValueError, match="resource hash mismatch"):
        build._extension()


def test_enabled_cannot_reuse_stock(
    tmp_path: Path, fake_build: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    stock = build.prepare_solver(cache_dir=tmp_path)
    source_hash = build._sha(b"pristine")
    manifest = {
        "capability": build.CAPABILITY,
        "outputs": {"src/native.f90": source_hash},
    }
    monkeypatch.setattr(build, "_extension", lambda: (manifest, {}))
    monkeypatch.setattr(build, "_apply_extension", lambda *args: None)
    extended = build.prepare_solver(True, cache_dir=tmp_path)
    assert stock != extended
    capability = json.loads((extended.parent.parent / "capability.json").read_text())
    assert capability["capabilities"] == [build.CAPABILITY]
    assert len(fake_build) == 2


def test_subprocess_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, command, stderr=b"compiler failed")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="compiler failed"):
        build._run(["compile"])


def test_export_uses_committed_source_and_pinned_nested_dependencies(
    tmp_path: Path,
) -> None:
    """The archive follows gitlinks even when nested working trees are dirty."""
    import io
    import tarfile

    upstream = Path(__file__).resolve().parents[2] / "libs/pyudales/u-dales"
    if not all(
        (upstream / name / ".git").exists()
        for name in (".", "2decomp-fft", "tools/View3D")
    ):
        pytest.skip("Local pinned source is not initialized")
    destination = tmp_path / "source"
    build._export_source(upstream, destination)
    for relative in (
        "src/modsubgrid.f90",
        "2decomp-fft/src/decomp_2d.f90",
        "tools/View3D/src/view3d.c",
    ):
        assert (destination / relative).is_file()
    archive = build._run(["git", "archive", build.UPSTREAM_COMMIT], cwd=upstream).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        member = tar.extractfile("src/modsubgrid.f90")
        assert member is not None
        assert (destination / "src/modsubgrid.f90").read_bytes() == member.read()
    assert (
        build.FINDFFTW_COMMIT in (destination / "downloadFindFFTW.cmake.in").read_text()
    )


def test_missing_source_uses_private_pinned_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(command)
        if command[:3] == ["git", "clone", "--bare"]:
            Path(command[-1]).mkdir()
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(build, "_run", run)
    original = tmp_path / "uninitialized-submodule"
    mirror = build._source_repository(original, tmp_path)
    assert mirror == tmp_path / "upstream.git"
    assert not original.exists()
    assert any(f"{build.UPSTREAM_COMMIT}^{{commit}}" in command for command in calls)
    assert calls[0][:3] == ["git", "clone", "--bare"]


def test_extension_input_and_output_hashes_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "original").write_bytes(b"wrong")
    manifest = {
        "inputs": {"original": build._sha(b"original")},
        "outputs": {"original": build._sha(b"patched")},
        "patch": "patch",
    }
    with pytest.raises(ValueError, match="source hash mismatch"):
        build._apply_extension(source, manifest, {"patch": b"patch"})
    (source / "original").write_bytes(b"original")
    monkeypatch.setattr(build, "_run", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="source hash mismatch"):
        build._apply_extension(source, manifest, {"patch": b"patch"})


def test_validate_solver_rejects_wrong_variant_and_corrupt_binary(
    tmp_path: Path,
    fake_build: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stock = build.prepare_solver(cache_dir=tmp_path)
    assert build.validate_solver(stock)
    assert not build.validate_solver(stock, discrepancy_enabled=True)
    manifest = {
        "capability": build.CAPABILITY,
        "outputs": {"src/native.f90": build._sha(b"pristine")},
    }
    monkeypatch.setattr(build, "_extension", lambda: (manifest, {}))
    monkeypatch.setattr(build, "_apply_extension", lambda *args: None)
    extended = build.prepare_solver(True, cache_dir=tmp_path)
    assert build.validate_solver(extended, discrepancy_enabled=True)
    assert not build.validate_solver(extended)
    extended.write_bytes(b"corrupt")
    assert not build.validate_solver(extended, discrepancy_enabled=True)
    assert not build.validate_solver(tmp_path / "missing")


def test_validate_solver_rejects_corrupt_source_or_capability(
    tmp_path: Path,
    fake_build: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        "capability": build.CAPABILITY,
        "outputs": {"src/native.f90": build._sha(b"pristine")},
    }
    monkeypatch.setattr(build, "_extension", lambda: (manifest, {}))
    monkeypatch.setattr(build, "_apply_extension", lambda *args: None)
    extended = build.prepare_solver(True, cache_dir=tmp_path)
    (build.solver_source_dir(extended) / "src/native.f90").write_bytes(b"edited")
    assert not build.validate_solver(extended, discrepancy_enabled=True)
    extended = build.prepare_solver(True, cache_dir=tmp_path)
    capability_path = extended.parent.parent / "capability.json"
    capability = json.loads(capability_path.read_text())
    capability["capabilities"] = []
    capability_path.write_text(json.dumps(capability))
    assert not build.validate_solver(extended, discrepancy_enabled=True)

"""Build immutable, capability-checked solvers without editing the submodule.

Preparation belongs in the parent process before starting ensemble workers. The
cache lock also serializes independent processes preparing the same build.
"""

from __future__ import annotations

import configparser
import contextlib
import fcntl
import hashlib
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Iterator, Mapping

UPSTREAM_COMMIT = "b84916ac60cecd1da54dd09df76c15e30dcaabe9"
UPSTREAM_URL = "https://github.com/uDALES/u-dales.git"
# Upstream's CMake downloader otherwise tracks the mutable findFFTW HEAD.
FINDFFTW_COMMIT = "ac788529392825067c9118f9f099aaa1efb77589"
CAPABILITY = "sgs_strain_rotation_v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, check=True, capture_output=True, **kwargs)
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or b"") + (exc.stderr or b"")
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        raise RuntimeError(f"uDALES command failed: {command!r}\n{output}") from exc


def _scripts() -> Path:
    from pyudales import LOCAL_EXECUTE_SCRIPT

    return LOCAL_EXECUTE_SCRIPT.parent


def _default_cache() -> Path:
    if os.environ.get("PYUDALES_CACHE_DIR"):
        return Path(os.environ["PYUDALES_CACHE_DIR"]).expanduser()
    for parent in Path(__file__).resolve().parents:
        if (parent / ".gitmodules").is_file() and os.access(parent, os.W_OK):
            return parent / ".cache" / "pyudales"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pyudales"


@contextlib.contextmanager
def _lock(path: Path) -> Iterator[None]:
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _relative(root: Path, name: str) -> Path:
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Unsafe path in solver manifest: {name!r}")
    return path


def _extension() -> tuple[dict[str, Any], dict[str, bytes]]:
    resource = resources.files("pyudales").joinpath("solver_extensions/discrepancy")
    manifest = json.loads(resource.joinpath("manifest.json").read_bytes())
    if manifest["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError(
            "Discrepancy manifest requires an unsupported upstream revision"
        )
    if manifest["capability"] != CAPABILITY:
        raise ValueError("Unsupported discrepancy capability")
    blobs = {}
    for name, expected in manifest["resources"].items():
        _relative(Path("."), name)
        blob = resource.joinpath(name).read_bytes()
        if _sha(blob) != expected:
            raise ValueError(f"Discrepancy resource hash mismatch: {name}")
        blobs[name] = blob
    return manifest, blobs


def _verify_files(root: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = _relative(root, name)
        if not path.is_file() or _sha(path.read_bytes()) != expected:
            raise ValueError(f"uDALES source hash mismatch: {name}")


def _apply_extension(
    source: Path, manifest: dict[str, Any], blobs: dict[str, bytes]
) -> None:
    _verify_files(source, manifest["inputs"])
    patch = blobs[manifest["patch"]]
    # A tar export has no .git, so apply is deliberately run outside the host repo.
    _run(
        ["git", "apply", "--check", "-"],
        cwd=source,
        input=patch,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(source.parent)},
    )
    _run(
        ["git", "apply", "-"],
        cwd=source,
        input=patch,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(source.parent)},
    )
    for name, target in manifest.get("copies", {}).items():
        destination = _relative(source, target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blobs[name])
    _verify_files(source, manifest["outputs"])


def _source_repository(source_dir: Path | None, cache: Path) -> Path:
    if source_dir is None:
        from pyudales import UDALES_PATH

        source_dir = UDALES_PATH
    if (source_dir / ".git").exists():
        # Export the pinned object, never the developer's working tree or HEAD.
        _run(["git", "cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}"], cwd=source_dir)
        return source_dir
    mirror = cache / "upstream.git"
    with _lock(cache / "upstream.lock"):
        if not mirror.is_dir():
            staging = Path(tempfile.mkdtemp(prefix="upstream-", dir=cache))
            try:
                _run(["git", "clone", "--bare", UPSTREAM_URL, str(staging / "repo")])
                _run(
                    [
                        "git",
                        "--git-dir",
                        str(staging / "repo"),
                        "cat-file",
                        "-e",
                        f"{UPSTREAM_COMMIT}^{{commit}}",
                    ]
                )
                os.replace(staging / "repo", mirror)
            finally:
                shutil.rmtree(staging)
        _run(
            [
                "git",
                "--git-dir",
                str(mirror),
                "cat-file",
                "-e",
                f"{UPSTREAM_COMMIT}^{{commit}}",
            ]
        )
    return mirror


def _export_tree(repository: Path, destination: Path, commit: str) -> None:
    archive = _run(["git", "archive", "--format=tar", commit], cwd=repository).stdout
    destination.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter="data")
    gitmodules = destination / ".gitmodules"
    if not gitmodules.is_file():
        return
    modules = configparser.ConfigParser()
    modules.read(gitmodules)
    for section in modules.sections():
        relative = modules[section]["path"]
        child_destination = _relative(destination, relative)
        entry = (
            _run(["git", "ls-tree", commit, "--", relative], cwd=repository)
            .stdout.decode()
            .split()
        )
        if len(entry) < 3 or entry[0] != "160000":
            raise ValueError(f"Missing pinned gitlink for uDALES submodule {relative}")
        child_commit = entry[2]
        child_repository = repository / relative
        if (child_repository / ".git").exists():
            _export_tree(child_repository, child_destination, child_commit)
        else:
            with tempfile.TemporaryDirectory(
                prefix="dependency-", dir=destination.parent
            ) as temporary:
                mirror = Path(temporary) / "repo.git"
                _run(["git", "clone", "--bare", modules[section]["url"], str(mirror)])
                _export_tree(mirror, child_destination, child_commit)


def _export_source(repository: Path, destination: Path) -> None:
    _export_tree(repository, destination, UPSTREAM_COMMIT)
    # Pin and modernize only this disposable source export. The latter is needed
    # by CMake 4 for the nested findFFTW project (policy CLI flags don't propagate).
    downloader = destination / "downloadFindFFTW.cmake.in"
    text = downloader.read_text()
    text = text.replace("VERSION 2.8.2", "VERSION 3.5")
    text = text.replace(
        'GIT_REPOSITORY    "https://github.com/egpbos/findfftw.git"',
        'GIT_REPOSITORY    "https://github.com/egpbos/findfftw.git"\n'
        f'    GIT_TAG "{FINDFFTW_COMMIT}"',
    )
    downloader.write_text(text)


def _build_environment(compiler_command: str | None = None) -> dict[str, str]:
    """Use Apple's linker with a Conda Fortran compiler and the active SDK.

    Conda's ld64 can lag a new macOS SDK's .tbd syntax. The compiler still
    compiles normally; ``-B/usr/bin/`` makes its link driver select Apple's ld.
    Respect an explicit linker search prefix supplied by the caller.
    """
    env = os.environ.copy()
    if platform.system() != "Darwin" or not os.access("/usr/bin/ld", os.X_OK):
        return env
    compiler_tokens = shlex.split(compiler_command or env.get("FC", "mpif90"))
    compiler = (
        shutil.which(compiler_tokens[0], path=env.get("PATH"))
        if compiler_tokens
        else None
    )
    if compiler is None:
        return env
    compiler_path = Path(compiler)
    prefix = env.get("CONDA_PREFIX")
    in_conda = prefix is not None and compiler_path.resolve().is_relative_to(
        Path(prefix).resolve()
    )
    in_pixi = ".pixi" in compiler_path.parts and "envs" in compiler_path.parts
    if not (in_conda or in_pixi):
        return env
    flags = shlex.split(env.get("LDFLAGS", ""))
    if not any(flag.startswith("-B") for flag in flags):
        env["LDFLAGS"] = f"{env.get('LDFLAGS', '').strip()} -B/usr/bin/".strip()
    return env


def _environment_identity(env: Mapping[str, str]) -> dict[str, Any]:
    keys = (
        "FC",
        "CC",
        "CXX",
        "FFLAGS",
        "FCFLAGS",
        "CFLAGS",
        "LDFLAGS",
        "CMAKE_PREFIX_PATH",
        "CMAKE_GENERATOR",
        "CMAKE_TOOLCHAIN_FILE",
        "CMAKE_OSX_SYSROOT",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
        "DEVELOPER_DIR",
        "NETCDF_DIR",
        "NETCDF_FORTRAN_DIR",
        "FFTW_DOUBLE_LIB",
        "FFTW_FLOAT_LIB",
        "CONDA_PREFIX",
        "PATH",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "CPATH",
        "LIBRARY_PATH",
        "UDALES_BUILD_JOBS",
        "NVHPC_INSTALL_BASE",
    )
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment": {key: env.get(key, "") for key in keys},
    }
    commands = {
        "compiler": shlex.split(env.get("FC", "mpif90")) + ["--version"],
        "mpi": ["mpirun", "--version"],
        "cmake": ["cmake", "--version"],
        "netcdf": ["nc-config", "--all"],
        "netcdf_fortran": ["nf-config", "--all"],
        "fftw": ["pkg-config", "--modversion", "fftw3", "fftw3f"],
    }
    for name, command in commands.items():
        executable = shutil.which(command[0], path=env.get("PATH"))
        if executable:
            completed = subprocess.run(
                command, capture_output=True, check=False, env=dict(env)
            )
            result[name] = {
                "path": str(Path(executable).resolve()),
                "sha256": _sha(Path(executable).read_bytes()),
                "output": (completed.stdout + completed.stderr).decode(
                    errors="replace"
                ),
                "returncode": completed.returncode,
            }
        else:
            result[name] = None
    # Pixi/Conda dependency build IDs cover in-place library upgrades too.
    if platform.system() == "Darwin":
        sdk = env.get("CMAKE_OSX_SYSROOT") or env.get("SDKROOT")
        if not sdk and shutil.which("xcrun", path=env.get("PATH")):
            completed = subprocess.run(
                ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                capture_output=True,
                check=False,
                env=dict(env),
            )
            if completed.returncode == 0:
                sdk = completed.stdout.decode(errors="replace").strip()
        if sdk:
            sdk_path = Path(sdk).resolve()
            libsystem = sdk_path / "usr/lib/libSystem.tbd"
            result["macos_sdk"] = {
                "path": str(sdk_path),
                "libsystem_sha256": (
                    _sha(libsystem.read_bytes()) if libsystem.is_file() else None
                ),
            }
    prefix = env.get("CONDA_PREFIX")
    if prefix:
        result["packages"] = {
            p.name: _sha(p.read_bytes())
            for p in sorted((Path(prefix) / "conda-meta").glob("*.json"))
        }
    return result


def solver_source_dir(executable: Path) -> Path:
    """Return the explicit source/tools tree associated with a prepared solver."""
    return Path(executable).parent.parent / "source"


def _valid_build(root: Path, identity: dict[str, Any]) -> bool:
    try:
        manifest = json.loads((root / "capability.json").read_text())
        executable = root / "build" / "u-dales"
        if manifest["capabilities"] != (
            [CAPABILITY] if identity["discrepancy_enabled"] else []
        ):
            return False
        if manifest["identity"] != identity or not os.access(executable, os.X_OK):
            return False
        if manifest["executable_sha256"] != _sha(executable.read_bytes()):
            return False
        _verify_files(root, manifest["tool_hashes"])
        if any(not os.access(root / name, os.X_OK) for name in manifest["tool_hashes"]):
            return False
        if identity["discrepancy_enabled"]:
            _verify_files(root / "source", identity["extension"]["outputs"])
        return (root / "source" / "tools").is_dir()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_solver(executable: Path, discrepancy_enabled: bool = False) -> bool:
    """Check a prepared executable and capability without probing the toolchain.

    Forward models use this for cheap validation before reusing a selected
    solver. A failed check must go through ``prepare_solver`` again.
    """
    executable = Path(executable)
    root = executable.parent.parent
    if executable != root / "build" / "u-dales":
        return False
    try:
        manifest = json.loads((root / "capability.json").read_text())
        identity = manifest["identity"]
        if identity["upstream_commit"] != UPSTREAM_COMMIT:
            return False
        if identity["discrepancy_enabled"] is not discrepancy_enabled:
            return False
        if discrepancy_enabled and identity["extension"]["capability"] != CAPABILITY:
            return False
        return _valid_build(root, identity)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_solver(
    discrepancy_enabled: bool = False,
    *,
    source_dir: Path | None = None,
    cache_dir: Path | None = None,
    build_type: str = "release",
    prepare_tools: bool = False,
) -> Path:
    """Return a verified stock or discrepancy-capable executable.

    The pinned source is exported from an existing checkout or fetched into a
    private bare mirror. Cache entries publish atomically only after successful
    builds. An interrupted/invalid entry is rebuilt while holding its key lock.
    """
    build_type = build_type.lower()
    if build_type not in {"release", "debug"}:
        raise ValueError("build_type must be release or debug")
    cache = (Path(cache_dir) if cache_dir is not None else _default_cache()).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    manifest, blobs = _extension() if discrepancy_enabled else ({}, {})
    scripts = _scripts()
    script_names = ["build_udales_macos.sh", "build_preprocessing_macos.sh"]
    build_env = _build_environment()
    identity = {
        "upstream_commit": UPSTREAM_COMMIT,
        "findfftw_commit": FINDFFTW_COMMIT,
        "discrepancy_enabled": discrepancy_enabled,
        "extension": manifest,
        "build_type": build_type,
        "prepare_tools": prepare_tools,
        "environment": _environment_identity(build_env),
        "builder": _sha(Path(__file__).read_bytes()),
        "scripts": {name: _sha((scripts / name).read_bytes()) for name in script_names},
    }
    key = _sha(_json(identity))
    target = cache / key
    with _lock(cache / f"{key}.lock"):
        if _valid_build(target, identity):
            return target / "build" / "u-dales"
        if target.exists():
            shutil.rmtree(target)
        for abandoned in cache.glob(f".{key}-*"):
            shutil.rmtree(abandoned)
        repository = _source_repository(
            Path(source_dir) if source_dir is not None else None, cache
        )
        staging = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=cache))
        try:
            source = staging / "source"
            _export_source(repository, source)
            if discrepancy_enabled:
                _apply_extension(source, manifest, blobs)
            build = staging / "build"
            _run(
                [
                    "bash",
                    str(scripts / script_names[0]),
                    build_type,
                    str(source),
                    str(build),
                ],
                env=build_env,
            )
            tool_hashes = {}
            if prepare_tools:
                _run(
                    ["bash", str(scripts / script_names[1]), str(source)],
                    env=build_env,
                )
                tool = source / "tools" / "View3D" / "build" / "src" / "view3d"
                if not tool.is_file() or not os.access(tool, os.X_OK):
                    raise RuntimeError(
                        f"Preprocessing build did not produce an executable: {tool}"
                    )
                tool_hashes[str(tool.relative_to(staging))] = _sha(tool.read_bytes())
            executable = build / "u-dales"
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise RuntimeError(
                    f"uDALES build did not produce an executable: {executable}"
                )
            capabilities = {
                "identity": identity,
                "executable_sha256": _sha(executable.read_bytes()),
                "capabilities": [CAPABILITY] if discrepancy_enabled else [],
                "tool_hashes": tool_hashes,
            }
            (staging / "capability.json").write_bytes(_json(capabilities))
            os.replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return target / "build" / "u-dales"

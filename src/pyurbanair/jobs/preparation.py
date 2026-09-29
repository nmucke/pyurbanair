"""Immutable forward plans, input identities and conservative local validation.

This module must stay usable without importing a numerical runtime or solver.
Readiness checks inspect files; they never build or download backend sources.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast

from omegaconf import OmegaConf

from pyurbanair.config.composition import (
    compose_forward_config,
    validate_resolvers,
    validate_targets,
)
from pyurbanair.config.run_record import validate_run_config
from pyurbanair.jobs.native import apply_native_overrides, native_coverage
from pyurbanair.jobs.paths import bind_job_paths, ensure_private_directory, owned_path

_INPUT_KEYS = {
    "stl_path",
    "case_dir",
    "precomputed_geom_dir",
    "model_dir",
    "weights_path",
    "template_path",
}
_RECORDED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PALM_ROOT",
    "PALM_BIN",
    "PYPALM_PALM_VERSION",
    "PYPALM_MPIRUN_EXTRA_ARGS",
    "PYPALM_USE_DIRECT_RUN",
    "PYLBM_GPU_ARCH",
    "PYUDALES_CACHE_DIR",
    "NETCDF_FORTRAN_ROOT",
    "PYURBANAIR_DISABLE_CPU_PINNING",
)
_UDALES_BUILD_ENV = (
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
    "NETCDF_DIR",
    "NETCDF_FORTRAN_DIR",
    "FFTW_DOUBLE_LIB",
    "FFTW_FLOAT_LIB",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "CPATH",
    "LIBRARY_PATH",
    "UDALES_BUILD_JOBS",
    "NVHPC_INSTALL_BASE",
)


def _relevant_environment(backend: str) -> tuple[str, ...]:
    if backend in {"pyudales", "neural_surrogate"}:
        return (*_RECORDED_ENV, *_UDALES_BUILD_ENV, "PATH", "CONDA_PREFIX")
    return _RECORDED_ENV


def _environment_snapshot(backend: str) -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in _relevant_environment(backend)
        if name in os.environ
    }


def _cuda_activation_library(
    repo_root: Path, selected_environment: str, recorded: dict[str, str]
) -> str | None:
    """Mirror the pinned CUDA activation script's optional NVHPC lib prefix."""
    if selected_environment != "cuda":
        return None
    base = Path(
        recorded.get("NVHPC_INSTALL_BASE", str(repo_root / ".pixi/envs/cuda/.nvhpc"))
    ).expanduser()
    if not base.is_absolute():
        base = repo_root / base
    compilers = sorted(base.glob("Linux_x86_64/*/compilers/bin/nvfortran"))
    available = [path for path in compilers if os.access(path, os.X_OK)]
    if not available:
        return None
    library = available[-1].parent.parent / "lib"
    return str(library) if library.is_dir() else None


def verify_worker_toolchain_environment(
    plan: dict[str, Any], environment: dict[str, str]
) -> None:
    if plan["backend"] not in {"pyudales", "neural_surrogate"}:
        return
    recorded = plan["provenance"]["environment"]
    activation_library = plan["provenance"].get("cuda_activation_library")
    expected = {key: recorded[key] for key in _UDALES_BUILD_ENV if key in recorded}
    if activation_library:
        recorded_library = recorded.get("LD_LIBRARY_PATH")
        expected["LD_LIBRARY_PATH"] = activation_library + (
            f":{recorded_library}" if recorded_library else ""
        )
    changed = [
        key
        for key in _UDALES_BUILD_ENV
        if (key in environment) != (key in expected)
        or (key in expected and environment.get(key) != expected[key])
    ]
    if changed:
        raise ValueError(
            "uDALES build environment changed after Pixi activation: "
            + ", ".join(changed)
            + "; prepare again"
        )


_DEFAULT_LIMITS = {
    "max_members": 256,
    "max_windows": 32,
    "max_workers": 8,
    "max_cpu_threads": 128,
    "max_output_bytes": 16 * 1024**3,
    "max_case_input_bytes": 2 * 1024**3,
    "max_case_input_entries": 10_000,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _artifact_architecture(raw: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Resolve artifact references before checking every executable target."""
    validate_resolvers(raw)
    document = OmegaConf.to_yaml(OmegaConf.create(raw)).replace(
        "${oc.env:PWD}", str(repo_root)
    )
    exported = OmegaConf.create(document)
    architecture = OmegaConf.select(exported, "architecture")
    if not OmegaConf.is_dict(architecture):
        raise ValueError(
            "exported architecture must resolve to a mapping with trusted repository targets"
        )
    resolved = OmegaConf.to_container(architecture, resolve=True, throw_on_missing=True)
    if not isinstance(resolved, dict):
        raise ValueError("exported architecture must resolve to a mapping")
    result = cast(dict[str, Any], resolved)
    validate_targets(result, repo_root)
    return result


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    if source.is_file():
        return {
            "path": str(source),
            "kind": "file",
            "sha256": _hash_file(source),
            "bytes": source.stat().st_size,
        }
    files = {}
    size = 0
    for child in sorted(source.rglob("*")):
        if child.is_file():
            files[str(child.relative_to(source))] = _hash_file(child)
            size += child.stat().st_size
    return {
        "path": str(source),
        "kind": "directory",
        "sha256": hashlib.sha256(_canonical(files)).hexdigest(),
        "bytes": size,
    }


def _check_case_source(source: Path, store_root: Path) -> None:
    # Resolve aliases before walking: the default store may live below the checkout.
    if source.is_relative_to(store_root) or store_root.is_relative_to(source):
        raise ValueError(f"case_dir overlaps managed job storage: {source}")
    if not source.is_dir():
        raise ValueError(f"case_dir must be a directory: {source}")


def _case_entries(
    source: Path, repo_root: Path, store_root: Path, limits: dict[str, int]
) -> list[tuple[Path, bool, int, Path]]:
    """Collect bounded entries and resolve safe regular-file links before copying."""
    entries: list[tuple[Path, bool, int, Path]] = []
    pending = [(source, 0)]
    size = 0
    while pending:
        directory, depth = pending.pop()
        with os.scandir(directory) as scan:
            for child in scan:
                relative = Path(child.path).relative_to(source)
                metadata = child.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    target = Path(child.path).resolve(strict=True)
                    if not (
                        target.is_relative_to(source)
                        or target.is_relative_to(repo_root)
                    ) or target.is_relative_to(store_root):
                        raise ValueError(
                            f"case_dir contains an unsafe symlink: {relative}"
                        )
                    metadata = target.stat(follow_symlinks=False)
                    if not stat.S_ISREG(metadata.st_mode):
                        raise ValueError(f"case_dir contains a symlink: {relative}")
                else:
                    target = Path(child.path)
                is_directory = stat.S_ISDIR(metadata.st_mode)
                if not (is_directory or stat.S_ISREG(metadata.st_mode)):
                    raise ValueError(
                        f"case_dir contains a non-regular entry: {relative}"
                    )
                entries.append((relative, is_directory, metadata.st_size, target))
                if len(entries) > limits["max_case_input_entries"]:
                    raise ValueError("case_dir exceeds max_case_input_entries")
                if is_directory:
                    if depth >= 32:
                        raise ValueError("case_dir exceeds maximum directory depth 32")
                    pending.append((Path(child.path), depth + 1))
                else:
                    size += metadata.st_size
                    if size > limits["max_case_input_bytes"]:
                        raise ValueError("case_dir exceeds max_case_input_bytes")
    return entries


def _open_case_file(path: Path, relative: Path) -> int:
    """Open a scanned file without following a replaced directory component."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(path.anchor, directory_flags)
    try:
        for component in path.parts[1:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        # A swapped FIFO must not block the open before the regular-file check.
        return os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise ValueError(
            f"case_dir entry changed or contains a symlink: {relative}"
        ) from error
    finally:
        os.close(directory_fd)


def _case_fingerprint(
    path: str | Path, repo_root: Path, store_root: Path, limits: dict[str, int]
) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    _check_case_source(source, store_root)
    files: dict[str, str] = {}
    size = 0
    for relative, is_directory, expected_size, input_path in _case_entries(
        source, repo_root, store_root, limits
    ):
        if is_directory:
            continue
        digest = hashlib.sha256()
        descriptor = _open_case_file(input_path, relative)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f"case_dir entry changed: {relative}")
            file_size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_size += len(block)
                size += len(block)
                if size > limits["max_case_input_bytes"]:
                    raise ValueError("case_dir exceeds max_case_input_bytes")
                digest.update(block)
            if file_size != expected_size:
                raise ValueError(f"case_dir entry changed: {relative}")
        files[str(relative)] = digest.hexdigest()
    return {
        "path": str(source),
        "kind": "directory",
        "sha256": hashlib.sha256(_canonical(files)).hexdigest(),
        "bytes": size,
        "case_input": True,
    }


def _stage_case_directory(
    source: Path,
    destination: Path,
    repo_root: Path,
    store_root: Path,
    limits: dict[str, int],
) -> dict[str, Any]:
    """Copy a bounded case tree without following directory or file symlinks."""
    _check_case_source(source, store_root)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError(f"case_dir overlaps staged input destination: {source}")
    entries = _case_entries(source, repo_root, store_root, limits)

    destination.mkdir(parents=True)
    files: dict[str, str] = {}
    copied = 0
    for relative, is_directory, expected_size, input_path in entries:
        target = destination / relative
        if is_directory:
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        descriptor = _open_case_file(input_path, relative)
        with (
            os.fdopen(descriptor, "rb") as input_stream,
            target.open("xb") as output_stream,
        ):
            metadata = os.fstat(input_stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"case_dir entry changed during staging: {relative}")
            os.fchmod(output_stream.fileno(), stat.S_IMODE(metadata.st_mode))
            file_size = 0
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                file_size += len(block)
                copied += len(block)
                if copied > limits["max_case_input_bytes"]:
                    raise ValueError("case_dir exceeds max_case_input_bytes")
                digest.update(block)
                output_stream.write(block)
            if file_size != expected_size:
                raise ValueError(f"case_dir entry changed during staging: {relative}")
        files[str(relative)] = digest.hexdigest()
    return {
        "path": str(source),
        "kind": "directory",
        "sha256": hashlib.sha256(_canonical(files)).hexdigest(),
        "bytes": copied,
        "case_input": True,
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def code_identity(repo_root: str | Path, backend: str) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    files = {}
    directories = [root / "src" / "pyurbanair", root / "conf"]
    # Nested surrogate spin-up and auxiliary generator dependencies are relevant.
    names = (
        ("pylbm", "pyudales", "pypalm", "neural-surrogates")
        if backend == "neural_surrogate"
        else (backend,)
    )
    directories.extend(root / "libs" / name / "src" for name in names)
    for directory in directories:
        for source in sorted(directory.rglob("*")):
            if source.is_file() and source.suffix in {".py", ".yaml", ".toml", ".txt"}:
                files[str(source.relative_to(root))] = _hash_file(source)
    if "pyudales" in names:
        for relative, suffixes in (
            ("libs/pyudales/shell_scripts", {".sh"}),
            ("libs/pyudales/src/pyudales/shell_scripts", {".sh"}),
            (
                "libs/pyudales/src/pyudales/solver_extensions/discrepancy",
                {".json", ".patch", ".f90"},
            ),
        ):
            for source in sorted((root / relative).glob("*")):
                if source.is_file() and source.suffix in suffixes:
                    files[str(source.relative_to(root))] = _hash_file(source)
        activation = root / "activation_scripts/cuda_activation.sh"
        if activation.is_file():
            files[str(activation.relative_to(root))] = _hash_file(activation)
    for filename in (
        "pyproject.toml",
        "pixi.lock",
        "scripts/_common.py",
        "scripts/run_forward_model.py",
    ):
        source = root / filename
        if source.is_file():
            files[filename] = _hash_file(source)
    native = {}
    for name, relative_native in (
        ("pylbm", "libs/pylbm/LBM"),
        ("pyudales", "libs/pyudales/u-dales"),
        ("pypalm", "libs/pypalm/palm_model_system"),
    ):
        if name in names:
            location = root / relative_native
            native[name] = {
                "revision": _git(location, "rev-parse", "HEAD"),
                "changes": hashlib.sha256(
                    _git(location, "diff", "HEAD", "--", "src", "source").encode()
                ).hexdigest(),
            }
    return {
        "revision": _git(root, "rev-parse", "HEAD"),
        "source_digest": hashlib.sha256(_canonical(files)).hexdigest(),
        "native": native,
    }


def _active_models(config: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    def visit(path: str, model: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
        yield path, model
        if model.get("spinup_source") == "forward_model" and isinstance(
            model.get("spinup_forward_model"), dict
        ):
            yield from visit(
                f"{path}.spinup_forward_model", model["spinup_forward_model"]
            )
        if model.get("spinup_source") == "generative" and isinstance(
            model.get("generative_spinup"), dict
        ):
            yield f"{path}.generative_spinup", model["generative_spinup"]

    yield from visit("model.forward_model", config["model"]["forward_model"])


def _issue(
    field: str, message: str, fix: str, kind: str = "configuration"
) -> dict[str, str]:
    return {"field": field, "message": message, "suggested_fix": fix, "kind": kind}


def _resources(config: dict[str, Any]) -> dict[str, Any]:
    ensemble = config.get("ensemble", {})
    active = bool(config.get("run", {}).get("ensemble", False))
    members = int(ensemble.get("ensemble_size", 1)) if active else 1
    workers = (
        min(members, int(ensemble.get("num_parallel_processes", 1))) if active else 1
    )
    ranks = max(int(model.get("ncpu", 1)) for _, model in _active_models(config))
    threads = int(ensemble.get("num_cpus_per_process", 1)) if active else 1
    windows = 1 + int(config.get("run", {}).get("rollout_steps", 0))
    domain, time = config["domain"], config["time"]
    cells = math.prod(int(domain[key]) for key in ("nx", "ny", "nz"))
    frequency = float(time["output_frequency"])
    frames = (
        max(1, math.ceil(float(time["simulation_time"]) / frequency))
        if frequency > 0
        else 0
    )
    return {
        "members": members,
        "windows": windows,
        "workers": workers,
        "mpi_ranks_per_worker": ranks,
        "cpu_threads": workers * max(ranks, threads),
        "grid": {key: domain[key] for key in ("nx", "ny", "nz")},
        "frames_per_window": frames,
        "estimated_output_bytes": cells * frames * members * windows * 4 * 8,
        "estimate_note": "Uncompressed four-field float64 estimate; execution retains rollout states in memory.",
        "surrogate_batch_size": config["model"]["forward_model"].get(
            "rollout_batch_size"
        ),
        "device": config["model"]["forward_model"].get("device", "cpu"),
    }


def _validate(
    config: dict[str, Any], limits: dict[str, int]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    issues = []
    try:
        validate_run_config(OmegaConf.create(config), "forward")
    except (ValueError, TypeError) as error:
        issues.append(
            _issue("config", str(error), "Select a compatible forward experiment.")
        )
    for section, keys in (
        ("domain", ("nx", "ny", "nz")),
        ("time", ("simulation_time", "output_frequency", "seconds_per_knot")),
        (
            "ensemble",
            ("ensemble_size", "num_parallel_processes", "num_cpus_per_process"),
        ),
    ):
        for key in keys:
            value = config.get(section, {}).get(key)
            if value is None and key == "seconds_per_knot":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or (section in {"domain", "ensemble"} and int(value) != value)
            ):
                issues.append(
                    _issue(
                        f"{section}.{key}",
                        "must be a finite positive value (integer for counts)",
                        "Set a positive value.",
                    )
                )
    for field, value in (
        ("run.rollout_steps", config.get("run", {}).get("rollout_steps", 0)),
        ("time.spinup_time", config.get("time", {}).get("spinup_time", 0)),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (field.endswith("rollout_steps") and int(value) != value)
        ):
            issues.append(
                _issue(
                    field,
                    "must be nonnegative (integer for windows)",
                    "Use zero or a positive value.",
                )
            )
    bounds = config.get("domain", {}).get("bounds", [])
    if len(bounds) != 3 or any(
        not isinstance(axis, list)
        or len(axis) != 2
        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in axis)
        or axis[1] <= axis[0]
        for axis in bounds
    ):
        issues.append(
            _issue(
                "domain.bounds",
                "requires three increasing finite [lower, upper] metre pairs",
                "Correct the physical domain bounds.",
            )
        )
    if config.get("run", {}).get("ensemble_save_on_disk"):
        issues.append(
            _issue(
                "run.ensemble_save_on_disk",
                "streaming is not implemented in the forward workflow",
                "Set false; complete artifacts are still saved.",
            )
        )
    try:
        resources = _resources(config)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError):
        return issues, {}
    for resource, limit in (
        ("members", "max_members"),
        ("windows", "max_windows"),
        ("workers", "max_workers"),
        ("cpu_threads", "max_cpu_threads"),
        ("estimated_output_bytes", "max_output_bytes"),
    ):
        if resources[resource] > limits[limit]:
            issues.append(
                _issue(
                    resource,
                    f"requested {resources[resource]} exceeds local {limit}={limits[limit]}",
                    "Reduce the request or have the local owner raise the configured limit.",
                )
            )
    retained_limit = config.get("run", {}).get("max_retained_bytes", 2 * 1024**3)
    if (
        not isinstance(retained_limit, int)
        or isinstance(retained_limit, bool)
        or retained_limit <= 0
    ):
        issues.append(
            _issue(
                "run.max_retained_bytes",
                "must be a positive integer for managed complete-artifact runs",
                "Set an explicit positive retained-memory budget.",
            )
        )
    elif resources.get("estimated_output_bytes", 0) > retained_limit:
        issues.append(
            _issue(
                "run.max_retained_bytes",
                "estimated retained rollout data exceeds the configured memory budget",
                "Reduce the grid, members or windows, or explicitly raise run.max_retained_bytes within machine limits.",
            )
        )
    ensemble_model = config.get("model", {}).get("ensemble_model", {})
    for key in ("ensemble_size", "num_parallel_processes", "num_cpus_per_process"):
        if ensemble_model.get(key) != config.get("ensemble", {}).get(key):
            issues.append(
                _issue(
                    f"model.ensemble_model.{key}",
                    "contradicts the canonical ensemble resource budget",
                    f"Set ensemble.{key} instead.",
                )
            )
    for path, model in _active_models(config):
        backend = str(model.get("_target_", "")).split(".")[0]
        if path == "model.forward_model":
            expected_backend = (
                "neural_surrogate" if backend == "neural_surrogates" else backend
            )
            if config["model"].get("name") != expected_backend:
                issues.append(
                    _issue(
                        "model.name",
                        "does not match the selected constructor",
                        "Select the model group rather than renaming its backend.",
                    )
                )
            for key in ("nx", "ny", "nz", "bounds"):
                if model.get(key) != config["domain"].get(key):
                    issues.append(
                        _issue(
                            f"{path}.{key}",
                            "contradicts the canonical domain and resource estimate",
                            f"Set domain.{key} instead.",
                        )
                    )
            for key in ("simulation_time", "output_frequency", "spinup_time"):
                if model.get(key) != config["time"].get(key):
                    issues.append(
                        _issue(
                            f"{path}.{key}",
                            "contradicts the canonical simulation timing",
                            f"Set time.{key} instead.",
                        )
                    )
        ncpu = model.get("ncpu", 1)
        if not isinstance(ncpu, int) or isinstance(ncpu, bool) or ncpu <= 0:
            issues.append(
                _issue(
                    f"{path}.ncpu",
                    "must be a positive integer",
                    "Set a positive MPI rank count.",
                )
            )
        elif backend in {"pyudales", "pypalm"} and int(model.get("nx", 1)) % ncpu:
            issues.append(
                _issue(
                    f"{path}.ncpu",
                    "MPI ranks must divide nx",
                    "Choose a divisor of domain.nx.",
                )
            )
        if (
            backend == "pypalm"
            and model.get("boundary_condition") == "periodic"
            and any(int(model.get(key, 0)) % 2 for key in ("nx", "ny"))
        ):
            issues.append(
                _issue(
                    "domain",
                    "PALM periodic horizontal grid counts must be even",
                    "Choose even nx and ny.",
                )
            )
        name = str(model.get("experiment_name", "run"))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
            issues.append(
                _issue(
                    f"{path}.experiment_name",
                    "must be a plain name without path separators",
                    "Use a simple name or experiment number.",
                )
            )
    params = config.get("params", {})
    for section in ("parameters", "external_parameters", "static_parameters"):
        for name, sampler in params.get(section, {}).items():
            if not isinstance(sampler, dict):
                continue
            if (
                "min" in sampler
                and "max" in sampler
                and sampler["min"] is not None
                and sampler["max"] is not None
                and sampler["min"] > sampler["max"]
            ):
                issues.append(
                    _issue(
                        f"params.{section}.{name}",
                        "sampler minimum exceeds maximum",
                        "Correct the parameter bounds.",
                    )
                )
            std = sampler.get("std", 0)
            if any(
                not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
                for v in (std if isinstance(std, list) else [std])
            ):
                issues.append(
                    _issue(
                        f"params.{section}.{name}.std",
                        "standard deviations must be finite and nonnegative",
                        "Correct the prior standard deviation.",
                    )
                )
    return issues, resources


class PreparationService:
    """Create and verify resolved, content-addressed execution snapshots."""

    def __init__(
        self,
        repo_root: str | Path,
        store_root: str | Path,
        limits: dict[str, int] | None = None,
    ):
        self.repo_root = Path(repo_root).resolve(strict=True)
        self.store_root = Path(store_root).resolve()
        self.limits = dict(_DEFAULT_LIMITS)
        if limits:
            if set(limits) - set(self.limits):
                raise ValueError("unknown machine-local execution limit")
            self.limits.update(limits)
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in self.limits.values()
        ):
            raise ValueError("machine-local limits must be positive integers")

    def prepare(
        self,
        overrides: Iterable[str] = (),
        native_overrides: dict[str, Any] | None = None,
        initial_state: dict[str, Any] | str | None = None,
        execution_limits: dict[str, int] | None = None,
        environment: str = "dev",
    ) -> dict[str, Any]:
        if environment not in {"dev", "cuda"}:
            raise ValueError("simulation environment must be dev or cuda")
        limits = dict(self.limits)
        for key, requested_limit in (execution_limits or {}).items():
            if (
                key not in limits
                or isinstance(requested_limit, bool)
                or not isinstance(requested_limit, int)
                or requested_limit <= 0
                or requested_limit > limits[key]
            ):
                raise ValueError(
                    f"execution_limits.{key}: requests may only tighten configured positive limits"
                )
            limits[key] = requested_limit
        composition = compose_forward_config(self.repo_root, overrides)
        config = copy.deepcopy(composition["config"])
        defaults = compose_forward_config(self.repo_root)["config"]
        issues, resources = _validate(config, limits)
        plan_id = uuid.uuid4().hex
        ensure_private_directory(self.store_root)
        plan_dir = owned_path(self.store_root, Path("plans") / plan_id)
        plan_dir.mkdir(parents=True, mode=0o700)
        identities = []
        normalized = {}
        for prefix, model in _active_models(config):
            for key in _INPUT_KEYS:
                value = model.get(key)
                if value is None:
                    continue
                source = (self.repo_root / str(value)).resolve()
                model[key] = str(source)
                normalized[f"{prefix}.{key}"] = str(source)
                if not source.exists():
                    issues.append(
                        _issue(
                            f"{prefix}.{key}",
                            f"input does not exist: {source}",
                            "Supply an existing local input.",
                            "prerequisite",
                        )
                    )
                    continue
                if key == "case_dir":
                    staged = plan_dir / "inputs" / prefix.replace(".", "_") / "case"
                    identities.append(
                        _stage_case_directory(
                            source, staged, self.repo_root, self.store_root, limits
                        )
                    )
                    model[key] = str(staged)
                else:
                    identities.append(fingerprint(source))
        if initial_state is None:
            initial_state = config.get("run", {}).get("initial_state")
        if initial_state is not None:
            initial_state = (
                {"path": initial_state}
                if isinstance(initial_state, str)
                else copy.deepcopy(initial_state)
            )
            if not isinstance(initial_state, dict) or not initial_state.get("path"):
                raise ValueError("initial_state requires a NetCDF path")
            if set(initial_state) - {"path", "member", "time_index"}:
                raise ValueError("initial_state supports path, member and time_index")
            source = (self.repo_root / initial_state["path"]).resolve()
            initial_state["path"] = str(source)
            if source.is_file():
                identities.append(fingerprint(source))
            else:
                issues.append(
                    _issue(
                        "initial_state.path",
                        f"input does not exist: {source}",
                        "Select an existing NetCDF state.",
                        "prerequisite",
                    )
                )
        native_edits = apply_native_overrides(config, native_overrides, self.repo_root)
        if config["model"]["name"] == "neural_surrogate":
            self._surrogate(config, initial_state, plan_dir, identities, issues)
        self._selected_readiness(config, environment, issues, identities)
        prerequisites = self.capabilities(environment)
        backend = config["model"]["name"]
        recorded_environment = _environment_snapshot(backend)
        selected_backends = {backend}
        for _, active_model in _active_models(config):
            active_backend = str(active_model.get("_target_", "")).split(".")[0]
            if active_backend == "neural_surrogates":
                active_backend = "neural_surrogate"
            if active_backend in prerequisites["backends"]:
                selected_backends.add(active_backend)
        prerequisite_paths: set[str] = set()
        for selected_backend in sorted(selected_backends):
            readiness = prerequisites["backends"][selected_backend]
            prerequisite_paths.update(readiness["prerequisite_paths"])
            for message in readiness["missing"]:
                issues.append(
                    _issue(
                        f"backend.{selected_backend}",
                        message,
                        "Provision the selected worker environment/backend before launch.",
                        "prerequisite",
                    )
                )
        for prerequisite_path in sorted(prerequisite_paths):
            if Path(prerequisite_path).is_file():
                identities.append(fingerprint(prerequisite_path))
        # Validate write-path names and symlink containment without creating writes.
        bind_job_paths(config, plan_dir / "prospective_run")
        staged = plan_dir / "inputs"
        if staged.exists():
            identities.append(fingerprint(staged))
        plan = {
            "schema_version": 1,
            "plan_id": plan_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(self.repo_root),
            "environment": environment,
            "job_owned_environment": {
                "PYLBM_LBM_PATH": None,
                "PYLBM_BUILD_ROOT": "<run_root>/build/lbm",
                "PYPALM_FAST_IO_CATALOG": "<run_root>/fast_io/palm",
                "PYPALM_SKIP_AUTOINSTALL": "1",
            },
            "backend": backend,
            "config": config,
            "requested_config": composition["requested_config"],
            "overrides": composition["overrides"],
            "initial_state": initial_state,
            "native_overrides": native_overrides or {},
            "native_edits": native_edits,
            "normalized_paths": normalized,
            "diff_from_defaults": _diff(defaults, composition["config"]),
            "resources": resources,
            "limits": limits,
            "validation": {
                "configuration_valid": not any(
                    i["kind"] == "configuration" for i in issues
                ),
                "prerequisites_present": not any(
                    i["kind"] == "prerequisite" for i in issues
                ),
                "smoke_tested": False,
                "issues": issues,
            },
            "provenance": {
                "choices": composition["choices"],
                "sources": composition["sources"],
                "overrides": composition["overrides"],
                "code": code_identity(self.repo_root, backend),
                "inputs": identities,
                "environment": recorded_environment,
                "cuda_activation_library": _cuda_activation_library(
                    self.repo_root, environment, recorded_environment
                ),
            },
            "expected_artifacts": [
                "run_manifest.yaml",
                "artifact_index.json",
                "state member/window NetCDF",
                "sampled parameter member/window NetCDF",
            ],
        }
        plan["digest"] = hashlib.sha256(_canonical(plan)).hexdigest()
        destination = plan_dir / "plan.json"
        with destination.open("x") as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
        destination.chmod(0o400)
        return plan

    def _selected_readiness(
        self,
        config: dict[str, Any],
        environment: str,
        issues: list[dict[str, str]],
        identities: list[dict[str, Any]],
    ) -> None:
        prefix = self.repo_root / ".pixi" / "envs" / environment
        for path, model in _active_models(config):
            backend = str(model.get("_target_", "")).split(".")[0]
            if backend == "pylbm":
                requested = model.get("cuda", "auto")
                compilers = sorted(
                    (prefix / ".nvhpc").glob("Linux_x86_64/*/compilers/bin/nvfortran")
                )
                cuda = requested is True or (requested == "auto" and bool(compilers))
                if requested not in (True, False, "auto"):
                    issues.append(
                        _issue(
                            f"{path}.cuda",
                            "must be true, false or auto",
                            "Choose a supported CUDA mode.",
                        )
                    )
                if cuda:
                    if not compilers:
                        issues.append(
                            _issue(
                                f"{path}.cuda",
                                "selected worker environment has no NVHPC compiler",
                                "Set model.forward_model.cuda=false or provision environment=cuda.",
                                "prerequisite",
                            )
                        )
                    if not Path("/dev/nvidiactl").exists() or os.environ.get(
                        "CUDA_VISIBLE_DEVICES"
                    ) in ("", "-1"):
                        issues.append(
                            _issue(
                                f"{path}.cuda",
                                "no visible CUDA device",
                                "Set model.forward_model.cuda=false or use a GPU host.",
                                "prerequisite",
                            )
                        )
                    netcdf_root = Path(
                        os.environ.get(
                            "NETCDF_FORTRAN_ROOT", str(prefix / ".nvhpc/netcdf-fortran")
                        )
                    )
                    dependency = netcdf_root / "include/netcdf.mod"
                    if dependency.exists():
                        identities.append(fingerprint(dependency))
                    else:
                        issues.append(
                            _issue(
                                f"{path}.cuda",
                                "NVHPC-compatible NetCDF must be installed before a managed run",
                                "Run CUDA environment setup first; shared cache installation is not a job operation.",
                                "prerequisite",
                            )
                        )
            if backend == "pypalm" and (
                config["model"].get("compile")
                or (
                    path.endswith("spinup_forward_model")
                    and config["model"].get("prepare", {}).get("compile", True)
                )
            ):
                issues.append(
                    _issue(
                        "model.compile",
                        "PALM compilation mutates the shared installation",
                        "Build PALM during setup, then set model.compile=false (model.prepare.compile=false for spin-up).",
                    )
                )
            if backend == "pyudales" and not model.get("precomputed_geom_dir"):
                tools = (
                    self.repo_root
                    / "libs/pyudales/u-dales/tools/IBM/IBM_preproc_fortran"
                )
                if not tools.is_dir():
                    issues.append(
                        _issue(
                            f"{path}.case_dir",
                            f"missing preprocessing sources {tools}",
                            "Provision uDALES preprocessing or supply precomputed_geom_dir.",
                            "prerequisite",
                        )
                    )
            if (
                backend == "neural_surrogates"
                and str(model.get("device", "cpu")).startswith("cuda")
                and os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1")
            ):
                issues.append(
                    _issue(
                        f"{path}.device",
                        "CUDA_VISIBLE_DEVICES disables all GPUs",
                        "Set model.forward_model.device=cpu or select a visible GPU.",
                        "prerequisite",
                    )
                )

    def _surrogate(
        self,
        config: dict[str, Any],
        initial_state: Any,
        plan_dir: Path,
        identities: list,
        issues: list,
    ) -> None:
        model = config["model"]["forward_model"]
        if model.get("spinup_source") == "generative":
            generator = model.get("generative_spinup") or {}
            for key in ("model_dir", "template_path"):
                if not generator.get(key):
                    issues.append(
                        _issue(
                            f"model.forward_model.generative_spinup.{key}",
                            "required for generative spin-up",
                            "Select a trained generator artifact and matching template.",
                        )
                    )
            if generator.get("model_dir"):
                exported_generator = Path(generator["model_dir"]) / "config.yaml"
                if exported_generator.is_file():
                    generator_config = OmegaConf.to_container(
                        OmegaConf.load(exported_generator), resolve=False
                    )
                    if isinstance(generator_config, dict):
                        _artifact_architecture(
                            cast(dict[str, Any], generator_config), self.repo_root
                        )
                if not (Path(generator["model_dir"]) / "weights.pt").is_file():
                    issues.append(
                        _issue(
                            "model.forward_model.generative_spinup.model_dir",
                            "missing trained generator weights.pt",
                            "Select a complete generator export.",
                            "prerequisite",
                        )
                    )
        if model.get("allow_uninitialized_weights"):
            issues.append(
                _issue(
                    "model.forward_model.allow_uninitialized_weights",
                    "untrained weights are not a valid inference run",
                    "Use a trained model export.",
                )
            )
        if model.get("spinup_source") == "training_data" and initial_state is None:
            issues.append(
                _issue(
                    "initial_state",
                    "training_data spin-up needs an explicit initial-state selection",
                    "Supply initial_state.path or choose a provisioned forward_model/generative spin-up.",
                )
            )
        if (
            str(model.get("device", "cpu")).startswith("cuda")
            and not Path("/dev/nvidiactl").exists()
        ):
            issues.append(
                _issue(
                    "model.forward_model.device",
                    "no local CUDA device found",
                    "Set model.forward_model.device=cpu or use a GPU host.",
                    "prerequisite",
                )
            )
        directory = Path(model.get("model_dir", ""))
        exported = directory / "config.yaml"
        if not exported.is_file():
            issues.append(
                _issue(
                    "model.forward_model.model_dir",
                    "missing exported config.yaml",
                    "Select a trained export.",
                    "prerequisite",
                )
            )
            return
        trained = OmegaConf.to_container(OmegaConf.load(exported), resolve=False)
        if not isinstance(trained, dict):
            raise ValueError("surrogate config.yaml must be a mapping")
        trained["architecture"] = _artifact_architecture(
            cast(dict[str, Any], trained), self.repo_root
        )
        dataset = trained.get("dataset", {})
        data_root = (self.repo_root / str(dataset.get("root_dir", ""))).resolve()
        dataset["root_dir"] = str(data_root)
        data_cfg_path = data_root / "config.yaml"
        if data_cfg_path.is_file():
            identities.append(fingerprint(data_cfg_path))
            metadata = OmegaConf.to_container(
                OmegaConf.load(data_cfg_path), resolve=True
            )
            if isinstance(metadata, dict):
                for key in ("nx", "ny", "nz", "bounds"):
                    if metadata.get("domain", {}).get(key) != config["domain"].get(key):
                        issues.append(
                            _issue(
                                f"domain.{key}",
                                "does not match the trained grid",
                                "Use the export's trained domain.",
                            )
                        )
                cadence = metadata.get("training_data", {}).get(
                    "output_frequency", metadata.get("time", {}).get("output_frequency")
                )
                if cadence is not None and float(cadence) != float(
                    config["time"]["output_frequency"]
                ):
                    issues.append(
                        _issue(
                            "time.output_frequency",
                            "does not match trained output cadence",
                            "Use the trained cadence.",
                        )
                    )
        else:
            issues.append(
                _issue(
                    "model.forward_model.model_dir",
                    f"missing training metadata {data_cfg_path}",
                    "Restore the export's referenced training config.",
                    "prerequisite",
                )
            )
        weights = Path(model.get("weights_path") or directory / "weights.pt")
        if not weights.is_file():
            issues.append(
                _issue(
                    "model.forward_model.weights_path",
                    f"missing weights {weights}",
                    "Supply the trained weights.",
                    "prerequisite",
                )
            )
        else:
            model["weights_path"] = str(weights.resolve())
        staged = plan_dir / "inputs" / "model_export"
        staged.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(trained), staged / "config.yaml")
        model["model_dir"] = str(staged)

    def load(self, plan_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{32}", plan_id):
            raise ValueError("invalid plan ID")
        destination = self.store_root / "plans" / plan_id / "plan.json"
        if not destination.resolve().is_relative_to(self.store_root):
            raise ValueError("plan path escapes the store")
        plan = json.loads(destination.read_text())
        unsigned = {key: value for key, value in plan.items() if key != "digest"}
        if hashlib.sha256(_canonical(unsigned)).hexdigest() != plan.get("digest"):
            raise ValueError("prepared plan was modified; prepare a new plan")
        if (
            plan.get("repo_root") != str(self.repo_root)
            or plan.get("plan_id") != plan_id
        ):
            raise ValueError("plan belongs to a different checkout or identity")
        return cast(dict[str, Any], plan)

    def verify(self, plan_id: str, *, check_environment: bool = True) -> dict[str, Any]:
        plan = self.load(plan_id)
        if code_identity(self.repo_root, plan["backend"]) != plan["provenance"]["code"]:
            raise ValueError(
                "code or configuration changed since preparation; prepare again"
            )
        for identity in plan["provenance"]["inputs"]:
            try:
                current = (
                    _case_fingerprint(
                        identity["path"],
                        self.repo_root,
                        self.store_root,
                        plan["limits"],
                    )
                    if identity.get("case_input")
                    else fingerprint(identity["path"])
                )
            except OSError as error:
                raise ValueError(
                    f"prepared input disappeared: {identity['path']}"
                ) from error
            if current != identity:
                raise ValueError(
                    f"prepared input changed: {identity['path']}; prepare again"
                )
        current_env = _environment_snapshot(plan["backend"])
        if check_environment and current_env != plan["provenance"]["environment"]:
            raise ValueError("relevant execution environment changed; prepare again")
        return plan

    def capabilities(self, environment: str = "dev") -> dict[str, Any]:
        bin_dir = self.repo_root / ".pixi" / "envs" / environment / "bin"
        definitions: dict[str, list[Path]] = {
            "pylbm": [
                self.repo_root / "libs/pylbm/LBM/src",
                bin_dir / "nf-config",
                bin_dir / "gfortran",
            ],
            "pyudales": [
                self.repo_root / "libs/pyudales/u-dales/.git",
                bin_dir / "mpirun",
                bin_dir / "mpif90",
                bin_dir / "cmake",
                bin_dir / "nc-config",
                bin_dir / "nf-config",
            ],
            "pypalm": [
                Path(
                    os.environ.get(
                        "PALM_ROOT",
                        str(self.repo_root / "libs/pypalm/palm_model_system"),
                    )
                )
                / "MAKE_DEPOSITORY_default/palm",
                bin_dir / "mpirun",
            ],
            "neural_surrogate": [],
        }
        backends = {}
        for backend, prerequisites in definitions.items():
            missing = [
                f"Missing {path}"
                for path in [bin_dir / "python", *prerequisites]
                if not path.exists()
            ]
            backends[backend] = {
                "prerequisites_present": not missing,
                "missing": missing,
                "smoke_tested": False,
                "native_coverage": native_coverage(backend),
                "prerequisite_paths": [
                    str(path) for path in [bin_dir / "python", *prerequisites]
                ],
            }
        return {
            "repo_root": str(self.repo_root),
            "environment": environment,
            "python": str(bin_dir / "python"),
            "backends": backends,
            "limits": self.limits,
            "modes": ["single", "ensemble", "static", "dynamic", "rollout"],
            "streaming": False,
        }


def _diff(default: Any, requested: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(default, dict) and isinstance(requested, dict):
        result = []
        for key in sorted(set(default) | set(requested)):
            result.extend(
                _diff(
                    default.get(key), requested.get(key), f"{prefix}.{key}".lstrip(".")
                )
            )
        return result
    return (
        []
        if default == requested
        else [{"field": prefix, "default": default, "requested": requested}]
    )

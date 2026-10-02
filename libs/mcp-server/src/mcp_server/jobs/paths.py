"""Bind every supported managed write location to a private run directory."""

from __future__ import annotations

import copy
import os
import re
import stat
from pathlib import Path
from typing import Any


def ensure_private_directory(path: str | Path) -> Path:
    """Create a dedicated private store, refusing to alter existing permissions."""
    directory = Path(path)
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    metadata = directory.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise ValueError(
            f"Job storage must be a private directory owned by the current user: {directory}. "
            "Choose a new dedicated store path, or restrict this directory to mode 0700 before retrying."
        )
    return directory


def owned_path(root: str | Path, relative: str | Path) -> Path:
    base = Path(root).resolve()
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"path escapes the owned job directory: {relative}")
    return candidate


def bind_job_paths(
    config: dict[str, Any], run_root: str | Path
) -> tuple[dict[str, Any], dict[str, str]]:
    root = Path(run_root).resolve()
    cfg = copy.deepcopy(config)
    scratch = owned_path(root, "scratch")
    results = owned_path(root, "artifacts")
    cfg.setdefault("paths", {}).update(
        results_dir=str(results), experiment_dir=str(scratch)
    )
    # Keep every finished window, so a failed or cancelled run has results.
    cfg.setdefault("forward", {})["save_windows"] = True

    def bind(node: Any, label: str) -> None:
        if not isinstance(node, dict):
            return
        target = node.get("_target_", "")
        backend = target.split(".", 1)[0] if isinstance(target, str) else ""
        if backend in {"pylbm", "pyudales", "pypalm"} and target.endswith(
            ".ForwardModel"
        ):
            base = owned_path(scratch, label)
            node["temp_dir"] = str(base)
            node["verbose"] = True
            # User additions may override constructor defaults; always own these.
            for field, leaf in (
                ("experiment_base_dir", "experiment"),
                ("output_dir", "outputs"),
                ("results_dir", "results"),
            ):
                if field in node:
                    node[field] = (
                        str(owned_path(base, leaf)) if node[field] is not None else None
                    )
            # Names are path fragments in native wrappers.
            if "experiment_name" in node:
                name = str(node["experiment_name"])
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
                    raise ValueError(
                        "model.forward_model.experiment_name must be a plain name"
                    )
        if backend in {"pylbm", "pyudales", "pypalm", "neural_surrogates"} and (
            target.endswith("EnsembleForwardModel")
            or target.endswith("NeuralSurrogateForwardModel")
        ):
            base = owned_path(scratch, label)
            if target.endswith("EnsembleForwardModel"):
                node["temp_dir"] = str(base)
            if node.get("results_dir") is not None:
                node["results_dir"] = str(owned_path(base, "results"))
        for key, value in node.items():
            if isinstance(value, dict):
                bind(value, f"{label}/{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    bind(child, f"{label}/{key}/{index}")

    bind(cfg, "config")
    environment = {
        "PYLBM_BUILD_ROOT": str(owned_path(root, "build/lbm")),
        "PYPALM_FAST_IO_CATALOG": str(owned_path(root, "fast_io/palm")),
        "TMPDIR": str(owned_path(root, "tmp")),
        "MPLCONFIGDIR": str(owned_path(root, "cache/matplotlib")),
    }
    return cfg, environment

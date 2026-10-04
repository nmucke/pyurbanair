"""Backend-free, serialized Hydra composition of configs/forward.yaml."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Iterable, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

_COMPOSITION_LOCK = threading.RLock()
_RESOLVER = re.compile(r"\$\{\s*([\w.]+)\s*:")
_TARGET = re.compile(r"^\s*_target_:\s*['\"]?([^\s'\"#]+)", re.MULTILINE)
OPTION_GROUPS = ("case", "model", "params", "visualization")


def _walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from _walk(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}.{index}"
            yield path, child
            yield from _walk(child, path)


def trusted_targets(repo_root: str | Path) -> set[str]:
    """Only targets declared by the owner-controlled files under configs/."""
    targets: set[str] = set()
    for source in (Path(repo_root).resolve() / "configs").rglob("*.yaml"):
        targets.update(_TARGET.findall(source.read_text()))
    return targets


def validate_targets(config: dict[str, Any], repo_root: str | Path) -> None:
    validate_resolvers(config)
    allowed = trusted_targets(repo_root)
    executables = {
        value.strip("\"'")
        for source in (Path(repo_root) / "configs/model").rglob("*.yaml")
        for value in re.findall(
            r"^\s*matlab_bin:\s*([^\s#]+)", source.read_text(), re.MULTILINE
        )
    }
    for path, value in _walk(config):
        if path.rsplit(".", 1)[-1] == "_target_" and (
            not isinstance(value, str) or value not in allowed
        ):
            raise ValueError(f"{path}: untrusted executable target {value!r}")
        if (
            path.rsplit(".", 1)[-1] == "matlab_bin"
            and value is not None
            and (not isinstance(value, str) or value not in executables)
        ):
            raise ValueError(
                f"{path}: executable paths must be registered in repository model configuration"
            )


def validate_resolvers(config: Any) -> None:
    """Check resolver expressions before resolving configs or exported artifacts."""
    for path, value in _walk({"config": config}):
        if not isinstance(value, str):
            continue
        for match in _RESOLVER.finditer(value):
            if match.group(1) not in {"oc.env", "oc.select"}:
                raise ValueError(f"{path}: custom resolvers are not permitted")
            # PWD is the checkout; USER and SLURM_JOB_ID name cluster scratch;
            # PYURBANAIR_MACHINE and PYURBANAIR_RESULTS_ROOT are set by the job
            # scripts. A default is a plain literal (no nested resolver).
            if match.group(1) == "oc.env" and not re.match(
                r"\$\{\s*oc\.env\s*:\s*(PWD|USER|SLURM_JOB_ID|PYURBANAIR_MACHINE"
                r"|PYURBANAIR_RESULTS_ROOT)(,\s*[^}\s$]+)?\s*\}",
                value[match.start() :],
            ):
                raise ValueError(
                    f"{path}: only the PWD, USER, SLURM_JOB_ID, PYURBANAIR_MACHINE "
                    "and PYURBANAIR_RESULTS_ROOT env resolvers are permitted"
                )


def validate_overrides(overrides: Iterable[str]) -> list[str]:
    result = list(overrides)
    for override in result:
        if not isinstance(override, str) or len(override) > 16384:
            raise ValueError("overrides must be bounded Hydra strings")
        key = override.split("=", 1)[0].lstrip("+~")
        if (
            key.startswith(("hydra", "defaults"))
            or "@hydra" in key
            or "searchpath" in key
            or "plugin" in key
            or "resolver" in key
        ):
            raise ValueError(
                f"{key}: Hydra plugins, launchers and search paths are job-owned"
            )
        if _RESOLVER.search(override):
            raise ValueError(
                f"{key}: resolver expressions are not accepted in overrides"
            )
    if len(result) > 256:
        raise ValueError("at most 256 ordered overrides are supported")
    return result


def compose_forward_config(
    repo_root: str | Path, overrides: Iterable[str] = ()
) -> dict[str, Any]:
    """Resolve a forward config without importing constructors or solver packages."""
    root = Path(repo_root).resolve(strict=True)
    arguments = validate_overrides(overrides)
    with _COMPOSITION_LOCK:
        with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
            cfg = compose(
                config_name="forward", overrides=arguments, return_hydra_config=True
            )
        choices = OmegaConf.to_container(cfg.hydra.runtime.choices, resolve=True)
        del cfg.hydra
        # Resolve only the selected machine's scratch dir (the others need $USER).
        scratch = OmegaConf.to_container(cfg.paths.scratch, resolve=False)
        assert isinstance(scratch, dict)
        cfg.paths.scratch = {cfg.paths.machine: scratch[cfg.paths.machine]}
        raw = OmegaConf.to_container(cfg, resolve=False)
        assert isinstance(raw, dict)
        validate_targets(cast(dict[str, Any], raw), root)
        # Resolve PWD against the explicit checkout, not a desktop app's cwd.
        serialized = OmegaConf.to_yaml(cfg).replace("${oc.env:PWD}", str(root))
        cfg = cast(DictConfig, OmegaConf.create(serialized))
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        assert isinstance(resolved, dict)
        validate_targets(cast(dict[str, Any], resolved), root)
    # common.yaml is an ungrouped defaults entry, absent from runtime.choices.
    sources = [root / "configs" / "forward.yaml", root / "configs" / "common.yaml"]
    assert isinstance(choices, dict)
    for group_value, choice in choices.items():
        group = str(group_value)
        if not group.startswith("hydra/") and choice is not None:
            source = root / "configs" / group.split("@", 1)[0] / f"{choice}.yaml"
            if source.is_file():
                sources.append(source)
    return {
        "config": resolved,
        "requested_config": raw,
        "overrides": arguments,
        "choices": {
            str(k): v for k, v in choices.items() if not str(k).startswith("hydra/")
        },
        "sources": [str(path) for path in sorted(set(sources))],
    }


def list_config_options(
    repo_root: str | Path,
    group: str | None = None,
    search: str | None = None,
    page: int = 0,
    page_size: int = 50,
) -> dict[str, Any]:
    """The forward config groups (case, model, params) and the render presets."""
    root = Path(repo_root).resolve() / "configs"
    if page < 0 or not 1 <= page_size <= 100:
        raise ValueError(
            "page must be nonnegative; page_size must be between 1 and 100"
        )
    entries = []
    for option_group in OPTION_GROUPS:
        if group and option_group != group:
            continue
        for path in sorted((root / option_group).glob("*.yaml")):
            relative = path.relative_to(root)
            if search and search.casefold() not in str(relative).casefold():
                continue
            entries.append(
                {"group": option_group, "name": path.stem, "source": str(relative)}
            )
    start = page * page_size
    return {
        "options": entries[start : start + page_size],
        "total": len(entries),
        "page": page,
    }


def inspect_config(
    repo_root: str | Path, overrides: Iterable[str] = (), subtree: str | None = None
) -> dict[str, Any]:
    result = compose_forward_config(repo_root, overrides)
    if subtree:
        selected: Any = result["config"]
        for segment in subtree.split("."):
            if isinstance(selected, list):
                selected = selected[int(segment)]
            elif isinstance(selected, dict) and segment in selected:
                selected = selected[segment]
            else:
                raise ValueError(f"unknown configuration subtree {subtree!r}")
        result["config"] = selected
    result["source_yaml"] = {
        str(Path(source).relative_to(Path(repo_root).resolve())): Path(
            source
        ).read_text()
        for source in result["sources"]
    }
    result["field_documentation"] = {
        "domain": "Grid counts and physical bounds in metres; owned by case.",
        "time": "Simulation, spin-up, output and knot intervals in seconds.",
        "forward.rollout_steps": "Additional windows; total windows = 1 + forward.rollout_steps.",
        "forward.ensemble": "Run ensemble.ensemble_size members instead of one.",
        "model.forward_model.sgs_constant": (
            "Backend-specific: dimensionless for LBM/uDALES, m²/s constant diffusivity for PALM."
        ),
        "paths": "Effective output and scratch paths are assigned by the job service.",
    }
    return result

"""Render a cinematic animation of an LES state file (Unreal Engine + Blender).

Point it at a case folder (or a state file); it writes a render bundle --
geometry, OpenVDB volumes, particle/pathline caches, isosurfaces, slice
textures, camera shots, HUD overlays and ``manifest.json`` -- then renders a
Blender preview mp4 and stages the Unreal Editor scripts. See
docs/les_render.md for the bundle contract and the Unreal workflow.

Runs in the lightweight ``viz`` pixi environment (openvdb, usd-core)::

    pixi run -e viz python scripts/visualization/render_les.py \\
        input=training_data/pyudales_idealized/state/train/sample_0000.nc

    # other presets / quick look / UE-only export
    ... render_preset=vortex time.duration=10
    ... blender.engine=cycles blender.samples=128 render.width=3840 render.height=2160
    ... stages.blender=false stages.video=false stages.alembic=true
"""

from __future__ import annotations

import logging
import pathlib
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from les_render.case import discover_case
from les_render.export import build_bundle
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def resolve_bundle_dir(cfg: DictConfig) -> pathlib.Path:
    """``output_dir`` if set, else ``<results_dir>/<case>/<preset>``.

    (Not ``resolve_output_dir``: that helper imports jax and every backend,
    which the ``viz`` environment deliberately does not carry.)
    """
    if cfg.output_dir:
        return pathlib.Path(str(cfg.output_dir))
    case = discover_case(cfg.input, state=cfg.state)
    return pathlib.Path(str(cfg.results_dir)) / case.name / str(cfg.render_preset.name)


def apply_case_overrides(
    cfg: DictConfig, overrides: dict, cli_overrides: list[str]
) -> DictConfig:
    """Merge a case folder's ``render.yaml`` over ``cfg``, then re-apply the
    command-line overrides so the command line always wins.

    Precedence (lowest -> highest): preset/config defaults, ``render.yaml``,
    command line. ``render.yaml`` can override preset keys (``render_preset:
    {look: daylight}``) but cannot switch presets by name.
    """
    if isinstance(overrides.get("render_preset"), str):
        raise ValueError(
            "render.yaml cannot switch presets by name "
            f"(render_preset: {overrides['render_preset']!r}); pass "
            "render_preset=<name> on the command line, or override preset keys "
            "as a mapping"
        )
    merged = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    # Plain `key=value` overrides only: group choices (render_preset=vortex) are
    # already reflected in cfg, and +/~ edits are not dotlist syntax.
    dotlist = [
        o
        for o in cli_overrides
        if "=" in o and o[0] not in "+~" and o.split("=", 1)[0] != "render_preset"
    ]
    if dotlist:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(dotlist))
    assert isinstance(merged, DictConfig)
    return merged


def run(cfg: DictConfig) -> pathlib.Path:
    case = discover_case(
        cfg.input, state=cfg.state, geometry=cfg.geometry, params=cfg.params
    )
    if case.overrides:
        log.info("merging per-case overrides from render.yaml")
        cli = (
            list(HydraConfig.get().overrides.task) if HydraConfig.initialized() else []
        )
        cfg = apply_case_overrides(cfg, case.overrides, cli)
    out_dir = resolve_bundle_dir(cfg)
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    build_bundle(container, out_dir)  # type: ignore[arg-type, unused-ignore]
    log.info("bundle: %s", out_dir)
    return out_dir


@hydra.main(  # type: ignore[misc, unused-ignore]
    version_base=None, config_path="../../conf", config_name="render_les"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    sys.exit(main())

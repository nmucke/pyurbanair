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


def run(cfg: DictConfig) -> pathlib.Path:
    case = discover_case(
        cfg.input, state=cfg.state, geometry=cfg.geometry, params=cfg.params
    )
    if case.overrides:
        log.info("merging per-case overrides from render.yaml")
        merged = OmegaConf.merge(cfg, OmegaConf.create(case.overrides))
        assert isinstance(merged, DictConfig)
        cfg = merged
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

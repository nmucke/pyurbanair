import os
from collections.abc import Iterator, Sequence

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

os.environ.setdefault("MPLBACKEND", "Agg")

# Test smoke shape: the smallest, fastest run the solvers accept — a tiny
# [0,20]^2 x [0,10] domain, a 3 s window and a 2-member ensemble. Applied to
# every composed test config so the suite exercises the code paths without
# producing a meaningful flow (formerly the deleted `+scale=test` overlay).
_SMOKE_OVERRIDES = [
    "domain.nx=20",
    "domain.ny=20",
    "domain.nz=4",
    "domain.bounds=[[0.0,20.0],[0.0,20.0],[0.0,10.0]]",
    "time.simulation_time=3.0",
    "time.output_frequency=1.0",
    "time.spinup_time=3.0",
    "time.seconds_per_knot=1.5",
    "ensemble.ensemble_size=2",
    "ensemble.num_parallel_processes=1",
]


# run_esmda.yaml is the one entry point that gets retuned for whatever
# production run is in flight — it has shipped machine-specific scratch roots
# (/export/...) and ``case: barcelona``, whose precomputed uDALES geometry
# bundle only matches the Barcelona grid, not the smoke domain above. Pin the
# test-friendly xie_and_castro case (the default of the other entry points) and
# the in-repo output roots so the suite never inherits either.
_ESMDA_OVERRIDES = [
    "case=xie_and_castro",
    "paths.results_dir=.temp/${truth_model.name}_to_${assim_model.name}",
    "paths.experiment_dir=${oc.env:PWD}/.temp",
]


def _compose_test_cfg(
    overrides: Sequence[str] | None = None,
    config_name: str = "run_forward_model",
) -> DictConfig:
    # ``config_name`` selects the primary config (entry point). Forward-model
    # tests use ``run_forward_model``; ESMDA tests use ``run_esmda`` (the single
    # primary config for scripts/esmda/run_esmda.py) and pick the smoother via the
    # ``esmda/smoother`` group override.
    esmda_overrides = _ESMDA_OVERRIDES if config_name == "run_esmda" else []
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name=config_name,
            overrides=[*_SMOKE_OVERRIDES, *esmda_overrides, *(overrides or [])],
        )
    _fit_nudging_to_smoke_domain(cfg)
    return cfg


# `nnudge_meters` is the height below which nudging is NOT applied, so it has to
# leave at least one nudged level above it. The backends set it for a real
# domain (tens of metres); the smoke shape above is 10 m tall, and anything at
# or above its top cell center makes the solver raise. Scale it down instead of
# holding the production configs to the test domain's height.
_SMOKE_NNUDGE_METERS = 4.0


def _fit_nudging_to_smoke_domain(cfg: DictConfig) -> None:
    # Only the mounts that actually carry a nudging_config — pylbm has none, and
    # run_esmda/run_filtering mount two models rather than one.
    for mount in ("model", "truth_model", "assim_model"):
        nudging = cfg.get(mount, {}).get("forward_model", {}).get("nudging_config")
        if nudging is not None and "nnudge_meters" in nudging:
            nudging.nnudge_meters = _SMOKE_NNUDGE_METERS


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _restore_hydra_config_singleton() -> Iterator[None]:
    """Keep ``HydraConfig`` from leaking a composed config across test files.

    ``HydraConfig`` is a process-wide singleton, so a test that primes it with
    ``HydraConfig.instance().set_config(cfg)`` leaves it populated for the rest
    of the session. A config from bare ``compose()`` has no
    ``hydra.runtime.output_dir`` (it is ``???``), so any later test that reaches
    ``resolve_output_dir`` takes its ``HydraConfig.initialized()`` branch and
    dies on MissingMandatoryValue instead of falling back to
    ``paths.base_results_dir``. Whether that happens comes down to file
    collection order, which makes it a nasty failure to place.
    """
    from hydra.core.hydra_config import HydraConfig

    previous = HydraConfig.instance().cfg
    yield
    HydraConfig.instance().cfg = previous


@pytest.fixture
def compose_test_cfg():
    return _compose_test_cfg


@pytest.fixture
def surrogate_model_dir_factory():
    """Build a minimal trained-surrogate folder (config.yaml + weights.pt).

    Mirrors what ``scripts/neural_surrogate/train_neural_surrogate.py`` writes: a model
    ``config.yaml`` holding the architecture and dataset (state_vars /
    param_vars / root_dir), a sibling ``weights.pt`` matching that
    architecture, and a training-data ``config.yaml`` (under ``root_dir``)
    carrying the trained ``domain`` and ``time`` so the forward model can
    derive its trained grid and output frequency. No real data or training
    needed — callers point the surrogate at the returned folder.
    """
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    def _build(
        tmp_path,
        *,
        domain: dict,
        time: dict,
        state_vars=("u", "v", "w"),
        param_vars=("inflow_angle", "velocity_magnitude"),
        architecture: dict | None = None,
    ):
        architecture = architecture or {
            "_target_": "neural_surrogates.UNetConvNeXt",
            "base_channels": 4,
            "channel_mults": [1, 2],
            "depths": [1, 1],
            "kernel_size": 3,
            "expansion": 2,
        }
        root_dir = tmp_path / "training_data"
        root_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create({"domain": domain, "time": time}),
            root_dir / "config.yaml",
        )

        model_dir = tmp_path / "model_dir"
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create(
                {
                    "architecture": architecture,
                    "dataset": {
                        "root_dir": str(root_dir),
                        "state_vars": list(state_vars),
                        "param_vars": list(param_vars),
                    },
                }
            ),
            model_dir / "config.yaml",
        )
        model = instantiate(
            architecture,
            n_state_channels=len(state_vars),
            n_params=len(param_vars),
        )
        torch.save(model.state_dict(), model_dir / "weights.pt")
        return model_dir

    return _build


@pytest.fixture(scope="module")
def compose_module_cfg():
    """Module-scoped variant of ``compose_test_cfg``.

    Composing inside ``hydra.initialize`` is cheap, but each call still
    opens and closes a ``GlobalHydra`` instance. Module-scoped fixtures
    (e.g. those that compile pylbm once for a whole test module) need a
    composer that can be invoked outside the function-scoped fixture
    lifecycle. This returns the same callable so test code looks
    identical to the function-scoped path.
    """
    return _compose_test_cfg

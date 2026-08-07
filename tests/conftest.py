import itertools
import os
import pathlib
import tempfile
from collections.abc import Callable, Iterator, Sequence

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

os.environ.setdefault("MPLBACKEND", "Agg")

# Test smoke shape: the smallest, fastest run the solvers accept — a tiny
# [0,20]^2 x [0,10] domain, a 3 s window and a 2-member ensemble. Applied to
# every composed test config so the suite exercises the code paths without
# producing a meaningful flow (formerly the deleted `+scale=test` overlay).
#
# DO NOT shrink this further expecting the suite to get faster — that was
# measured and it does not. At this size the e2e suite is dominated by FIXED
# per-test overhead (process spawn, imports, JAX/solver setup, the per-model
# compile check, NetCDF round-trips), not by the flow solve, which is already
# only a few thousand cell-updates. A/B on three representative tests with a
# warm build tree: halving nx/ny to 10 and cutting spinup_time to 1.0 moved the
# uDALES case 16.6s -> 16.3s (noise) and the whole 3-test subset 131s -> 115s,
# while the FULL suite came out SLOWER (672s -> 772s) — i.e. inside this
# machine's run-to-run spread. The coarser grid also costs real coverage: at
# dx=2 m the Xie & Castro blocks in this corner of the array (5–10 m across, x
# edges at 5/15, y edges every 5 m) are down to ~2 cells, close to voxelizing
# away entirely and leaving an empty channel. Not worth it.
#
# Each knob is at a floor for a reason:
#
# * ``bounds`` are fixed by the TESTS, not the physics — test_run_esmda and
#   test_run_filtering place sensors at x=18, so a narrower domain would put
#   the observations outside the grid.
# * ``nx``/``ny`` must stay EVEN (PALM's poisfft rejects an odd number of grid
#   points along a cyclic direction, PAC0071/PAC0072).
# * ``simulation_time`` is pinned to 3.0 by the e2e tests' own
#   ``obs.interval_seconds=3.0``; one interval is the minimum the interval-mode
#   observation operator can score.
# * ``output_frequency`` 1.0 -> 4 frames per window. Time interpolation and the
#   temporal observation operator need more than a single frame.
# * ``ensemble_size`` 2 — the ESMDA update needs >=2 members.
# * ``spinup_time`` is only prepended to the FIRST window (see
#   base_rollout_forward_model.disable_spinup); it keeps the spin-up-and-trim
#   path alive.
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
    # The case's held-out sensors sit at the real geometry's coordinates, all of
    # which fall OUTSIDE the 20x20x10 smoke box, so the validation sensor set was
    # present but scored nothing inside the domain. Pin one held-out sensor into
    # the smoke box — (16, 18) is a fluid street-level cell just downstream of the
    # single block the smoke domain crops (x=[5,15], y=[5,15], roof at the domain
    # top) — so the validation branches (sensor_statistics.validation, the S5
    # validation columns, the D1 held-out histograms) run under CI.
    # ``++`` because these keys exist only in the xie_and_castro case: barcelona
    # defines no held-out sensors, and a plain assignment would make any
    # ``compose_test_cfg(["case=barcelona"])` die with a Hydra "no match in
    # config" error instead of composing.
    "++obs.validation_x_points=[16.0]",
    "++obs.validation_y_points=[18.0]",
    "++obs.validation_z_points=[2.0]",
]


# run_esmda.yaml is the one entry point that gets retuned for whatever
# production run is in flight — it has shipped machine-specific scratch roots
# (/export/...) and ``case: barcelona``, whose precomputed uDALES geometry
# bundle only matches the Barcelona grid, not the smoke domain above. Pin the
# test-friendly xie_and_castro case (the default of the other entry points) so
# the suite never inherits it.
_ESMDA_OVERRIDES = [
    "case=xie_and_castro",
]


# --- Output isolation -----------------------------------------------------------
#
# EVERY entry point defaults its output roots to a scratch directory INSIDE the
# repo, and those are the exact directories a real run writes to:
#
#   run_esmda.yaml      results_dir=.temp/${truth_model.name}_to_${assim_model.name}
#                       experiment_dir=$PWD/.temp   base_results_dir=.temp_lbm
#   run_forward_model   results_dir=results/${model.name}
#                       experiment_dir=$PWD/.temp_${model.name}
#                       base_results_dir=.temp_${model.name}
#   run_filtering.yaml  results_dir=.temp/filtering_${truth}_to_${assim}
#                       experiment_dir=$PWD/.temp   base_results_dir=.temp_lbm
#
# A default pyudales→pyudales test run therefore lands on the SAME path as the
# production run of the same shape, and the suite overwrites a 32-member
# production output with 2-member smoke artifacts — silently, because the
# scripts only ever mkdir(exist_ok=True) and write. This has already destroyed a
# real run directory.
#
# So every composed config gets all three roots rewritten into a pytest-managed
# temp tree, and a FRESH subdirectory per compose call so two tests composing the
# same config cannot collide with each other either. The values are literal
# absolute paths, which also sidesteps the ${model.name} / ${truth_model.name}
# interpolations the defaults carry.
_OUTPUT_ROOT: pathlib.Path | None = None
_RUN_COUNTER = itertools.count()

# The pylbm build tree is deliberately NOT isolated per run. It lives at
# ``<experiment_dir>/lbm_build`` by default, so isolating ``experiment_dir``
# alone would force a full Fortran rebuild for every single test. ``pylbm``
# supports pointing the tree elsewhere via ``PYLBM_BUILD_ROOT``; we park it at a
# stable per-user location OUTSIDE the repo, which keeps the compiled binary
# cached across tests *and* across sessions (what ``.temp/lbm_build`` used to do)
# while still writing nothing into the repository. Set PYLBM_BUILD_ROOT yourself
# to override; concurrent pytest sessions must still be serialized (they share
# this tree, exactly as they used to share ``.temp``).
_LBM_BUILD_CACHE = (
    pathlib.Path(tempfile.gettempdir()) / f"pyurbanair-pytest-lbm-build-{os.getuid()}"
)


def _output_root() -> pathlib.Path:
    """Session-wide root every composed test config writes under.

    ``_compose_test_cfg`` is a plain function shared by a function-scoped and a
    **module-scoped** fixture, so it cannot take ``tmp_path``. The session-scoped
    autouse fixture below fills this in from ``tmp_path_factory``; the mkdtemp
    fallback keeps the composer usable if it is ever called outside a test.
    """
    global _OUTPUT_ROOT
    if _OUTPUT_ROOT is None:
        _OUTPUT_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="pyurbanair-tests-"))
    return _OUTPUT_ROOT


@pytest.fixture(scope="session", autouse=True)  # type: ignore[misc]
def _isolate_run_outputs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point the composer's output root at pytest's temp tree, session-wide."""
    global _OUTPUT_ROOT
    if _OUTPUT_ROOT is None:
        _OUTPUT_ROOT = tmp_path_factory.mktemp("run_outputs")
    os.environ.setdefault("PYLBM_BUILD_ROOT", str(_LBM_BUILD_CACHE))
    yield


def _isolated_path_overrides() -> list[str]:
    """A fresh, repo-external home for one composed config's three output roots."""
    run_dir = _output_root() / f"run_{next(_RUN_COUNTER):04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        f"paths.results_dir={run_dir / 'results'}",
        f"paths.experiment_dir={run_dir / 'experiment'}",
        f"++paths.base_results_dir={run_dir / 'base_results'}",
    ]


def _compose_test_cfg(
    overrides: Sequence[str] | None = None,
    config_name: str = "run_forward_model",
) -> DictConfig:
    # ``config_name`` selects the primary config (entry point). Forward-model
    # tests use ``run_forward_model``; ESMDA tests use ``run_esmda`` (the single
    # primary config for scripts/esmda/run_esmda.py) and pick the smoother via the
    # ``esmda/smoother`` group override.
    # ``run_probe_series`` inherits ``/run_esmda``'s defaults, including the
    # ``case`` that gets retuned per production run, so it needs the same pin.
    esmda_overrides = (
        _ESMDA_OVERRIDES if config_name in ("run_esmda", "run_probe_series") else []
    )
    # The path overrides go before the caller's, so a test that wants its own
    # (already isolated) tmp_path for one of the roots still wins.
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                *_SMOKE_OVERRIDES,
                *esmda_overrides,
                *_isolated_path_overrides(),
                *(overrides or []),
            ],
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


@pytest.fixture  # type: ignore[misc]
def compose_test_cfg() -> Callable[..., DictConfig]:
    return _compose_test_cfg


@pytest.fixture  # type: ignore[misc]
def surrogate_model_dir_factory() -> Callable[..., pathlib.Path]:
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
        tmp_path: pathlib.Path,
        *,
        domain: dict,
        time: dict,
        state_vars: Sequence[str] = ("u", "v", "w"),
        param_vars: Sequence[str] = ("inflow_angle", "velocity_magnitude"),
        architecture: dict | None = None,
    ) -> pathlib.Path:
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


@pytest.fixture(scope="module")  # type: ignore[misc]
def compose_module_cfg() -> Callable[..., DictConfig]:
    """Module-scoped variant of ``compose_test_cfg``.

    Composing inside ``hydra.initialize`` is cheap, but each call still
    opens and closes a ``GlobalHydra`` instance. Module-scoped fixtures
    (e.g. those that compile pylbm once for a whole test module) need a
    composer that can be invoked outside the function-scoped fixture
    lifecycle. This returns the same callable so test code looks
    identical to the function-scoped path.
    """
    return _compose_test_cfg

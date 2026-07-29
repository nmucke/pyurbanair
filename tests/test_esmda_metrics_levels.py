"""``run.metrics`` config block and the metrics-level resolver (WP1.0).

Composition-only tests: they check that the entry point ships the documented
knobs and that ``resolve_metrics_settings`` gates on them, including the
old-run-dir case (a saved config with no ``run.metrics`` key at all). The
end-to-end behaviour of each level is covered by the ESMDA pipeline test, which
is far slower — nothing here runs a solver.
"""

from collections.abc import Callable

import pytest
from omegaconf import DictConfig, OmegaConf

from scripts.esmda.compute_esmda_metrics import (
    DEFAULT_METRICS,
    METRICS_LEVELS,
    MetricsSettings,
    resolve_metrics_settings,
)


def test_run_esmda_config_exposes_metrics_block(
    compose_test_cfg: Callable[..., DictConfig],
) -> None:
    """The entry point ships the documented knobs, and they match the code."""
    cfg = compose_test_cfg(config_name="run_esmda")

    metrics = OmegaConf.to_container(cfg.run.metrics, resolve=True)
    # Same keys AND same values as the resolver's fallbacks: the whole
    # old-run-dir story rests on those two sets not drifting apart.
    assert metrics == DEFAULT_METRICS

    assert cfg.run.metrics.level in METRICS_LEVELS
    assert isinstance(cfg.run.metrics.n_z_slices, int)
    assert isinstance(cfg.run.metrics.mean_field_stride, int)
    assert isinstance(cfg.run.metrics.bootstrap_blocks, int)
    assert cfg.run.metrics.stations is None


def test_resolver_reads_the_configured_level() -> None:
    cfg = OmegaConf.create({"run": {"metrics": {"level": "basic", "n_z_slices": 2}}})

    settings = resolve_metrics_settings(cfg)

    assert settings.level == "basic"
    assert settings.n_z_slices == 2
    # Unset siblings still fall back to the shipped defaults.
    assert settings.bootstrap_blocks == DEFAULT_METRICS["bootstrap_blocks"]


def test_cli_override_beats_the_configured_level() -> None:
    cfg = OmegaConf.create({"run": {"metrics": {"level": "basic"}}})

    assert resolve_metrics_settings(cfg, "standard").level == "standard"
    # An explicit `None` override means "no override", not "no level".
    assert resolve_metrics_settings(cfg, None).level == "basic"


@pytest.mark.parametrize(  # type: ignore[misc]
    "cfg,override",
    [
        (OmegaConf.create({"run": {"metrics": {"level": "verbose"}}}), None),
        (OmegaConf.create({"run": {"metrics": {"level": "basic"}}}), "everything"),
    ],
)
def test_unknown_level_is_rejected_loudly(
    cfg: DictConfig, override: str | None
) -> None:
    with pytest.raises(ValueError, match="unknown run.metrics.level"):
        resolve_metrics_settings(cfg, override)


@pytest.mark.parametrize(  # type: ignore[misc]
    "key",
    ["n_z_slices", "mean_field_stride", "bootstrap_blocks"],
)
@pytest.mark.parametrize("bad", [0, -5])  # type: ignore[misc]
def test_non_positive_counts_are_rejected_loudly(key: str, bad: int) -> None:
    """Same reasoning as the level check: a typo must fail here, not mid-stream.

    Every one of these is a "how many of X" count, so zero or negative is never
    a request anyone can mean; left unchecked it surfaces as an empty slice or a
    divide-by-zero deep inside a streaming pass that has already read GBs.
    """
    cfg = OmegaConf.create({"run": {"metrics": {key: bad}}})

    with pytest.raises(ValueError, match=f"run.metrics.{key} must be >= 1"):
        resolve_metrics_settings(cfg)


def test_the_smallest_meaningful_counts_are_accepted() -> None:
    """1 is a legitimate value for all three -- the guard is `>= 1`, not `> 1`."""
    cfg = OmegaConf.create(
        {
            "run": {
                "metrics": {
                    "n_z_slices": 1,
                    "mean_field_stride": 1,
                    "bootstrap_blocks": 1,
                }
            }
        }
    )

    settings = resolve_metrics_settings(cfg)

    assert (settings.n_z_slices, settings.mean_field_stride) == (1, 1)
    assert settings.bootstrap_blocks == 1


@pytest.mark.parametrize(  # type: ignore[misc]
    "cfg",
    [
        # Old run dirs: their saved config predates the `run.metrics` block.
        OmegaConf.create({"run": {"skip_viz": False}}),
        # Defensive: no `run` block at all.
        OmegaConf.create({}),
        # An explicit null block behaves like an absent one.
        OmegaConf.create({"run": {"metrics": None}}),
    ],
    ids=["no_metrics_key", "no_run_key", "null_metrics"],
)
def test_missing_config_block_falls_back_to_defaults(cfg: DictConfig) -> None:
    settings = resolve_metrics_settings(cfg)

    assert settings.level == DEFAULT_METRICS["level"] == "standard"
    assert settings.n_z_slices == DEFAULT_METRICS["n_z_slices"]
    assert settings.mean_field_stride == DEFAULT_METRICS["mean_field_stride"]
    assert settings.bootstrap_blocks == DEFAULT_METRICS["bootstrap_blocks"]
    assert settings.stations is None


def test_missing_block_still_honours_the_cli_override() -> None:
    """An old run dir can be re-processed at `basic` without editing its config."""
    settings = resolve_metrics_settings(OmegaConf.create({"run": {}}), "basic")

    assert settings.level == "basic"


def test_stations_are_normalized_to_float_pairs() -> None:
    cfg = OmegaConf.create({"run": {"metrics": {"stations": [[1, 2], [3.5, 4.5]]}}})

    settings = resolve_metrics_settings(cfg)

    assert settings.stations == [[1.0, 2.0], [3.5, 4.5]]
    assert all(isinstance(v, float) for station in settings.stations for v in station)


def test_at_least_orders_the_levels() -> None:
    def level(name: str) -> MetricsSettings:
        return resolve_metrics_settings(OmegaConf.create({}), name)

    assert not level("basic").at_least("standard")
    assert level("basic").at_least("basic")
    assert level("standard").at_least("standard")
    # `full` is accepted today and buys nothing beyond `standard` yet.
    assert level("full").at_least("standard")
    assert not level("standard").at_least("full")

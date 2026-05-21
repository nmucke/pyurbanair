"""Architecture-agnostic training entry point (``docs/neural_surrogate_plan.md`` §6, §7).

``run(cfg)`` is the testable core: open a corpus, fit/load normalization, build
the architecture named in the config, train with the **pushforward curriculum**
(growing horizon ``H`` with a stop-gradient warm-up) and optional off-manifold
IC injection, log rollout-error-vs-horizon, and save a full checkpoint. ``main``
is the thin ``@hydra.main`` wrapper.

The loop never inspects the architecture — it only calls the
``SurrogateArchitecture`` interface via ``rollout`` and the field-space loss.
"""

from __future__ import annotations

import logging
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..architectures.registry import resolve_architecture
from ..data.dataset import WindowDataset, iterate_batches
from ..data.generate import open_corpus
from ..data.normalization import Normalization, fit_normalization
from ..utils.schema import ContractSchema
from . import loop as loop_mod
from .checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


def _plain(cfg: Any) -> Any:
    """Convert an OmegaConf node to plain Python (no-op for dict/list)."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(cfg, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return cfg


def _build_optimizer(opt_cfg: dict) -> optax.GradientTransformation:
    name = opt_cfg.get("name", "adam")
    lr = float(opt_cfg.get("learning_rate", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return optax.adam(lr)
    if name == "adamw":
        return optax.adamw(lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name!r}.")


def _resolve_horizon(schedule: list[dict], epoch: int) -> tuple[int, int]:
    """Pick (horizon, warmup) for an epoch from the curriculum schedule."""
    horizon, warmup = 1, 0
    for stage in sorted(schedule, key=lambda s: int(s["epoch"])):
        if epoch >= int(stage["epoch"]):
            horizon = int(stage["horizon"])
            warmup = int(stage.get("warmup", 0))
    return horizon, warmup


def run(cfg: Any) -> dict:
    """Train a surrogate and write a checkpoint. Returns the final metrics."""
    cfg = _plain(cfg)
    seed = int(cfg.get("seed", 0))
    key = jax.random.PRNGKey(seed)

    corpus = open_corpus(cfg["corpus_path"])
    var_names = corpus.var_names
    n_channels = len(var_names)
    schema = corpus.contract.param_schema
    param_dim = schema.conditioning_dim
    static = jnp.asarray(corpus.static_channels)
    mask = np.asarray(corpus.geometry_mask)
    fluid = 1.0 - jnp.asarray(mask)

    # Normalization: fit on the train split only (§6.3), persist for inference.
    norm = fit_normalization(
        (corpus.load_fields(i) for i in corpus.split_ids("train")),
        var_names, mask=mask,
    )

    history_len = int(cfg.get("history_len", 1))
    arch_cfg = dict(cfg["arch"])
    arch_name = arch_cfg.pop("name")
    arch_cfg.update(
        in_state_channels=n_channels,
        static_channels=int(static.shape[0]),
        history_len=history_len,
        param_dim=param_dim,
    )
    key, arch_key = jax.random.split(key)
    arch = resolve_architecture(arch_name, arch_cfg, key=arch_key)

    optimizer = _build_optimizer(dict(cfg.get("optimizer", {})))
    opt_state = optimizer.init(eqx.filter(arch, eqx.is_inexact_array))
    train_step = loop_mod.make_train_step(optimizer)

    batch_size = int(cfg.get("batch_size", 4))
    stride = int(cfg.get("stride", 1))
    num_epochs = int(cfg.get("num_epochs", 1))
    schedule = cfg.get("horizon_schedule") or [{"epoch": 0, "horizon": 1, "warmup": 0}]
    noise_std = float(cfg.get("off_manifold_noise_std", 0.0))

    train_ds = WindowDataset(
        corpus, "train", history_len, horizon=1, stride=stride, normalization=norm
    )
    val_ds = WindowDataset(
        corpus, "val", history_len, horizon=1, stride=stride, normalization=norm
    )
    rng = np.random.default_rng(seed)

    metrics: dict[str, Any] = {"train_loss": [], "val_loss": []}
    for epoch in range(num_epochs):
        horizon, warmup = _resolve_horizon(schedule, epoch)
        train_ds.set_horizon(horizon)
        if len(train_ds) == 0:
            logger.warning("No training windows at horizon %d; skipping epoch.", horizon)
            continue

        epoch_losses = []
        for batch in iterate_batches(train_ds, batch_size, rng=rng):
            batch = {k: jnp.asarray(v) for k, v in batch.items()}
            if noise_std > 0.0:
                key, sub = jax.random.split(key)
                batch = loop_mod.inject_off_manifold(batch, sub, noise_std)
            arch, opt_state, loss = train_step(
                arch, opt_state, batch, static, fluid, horizon, warmup
            )
            epoch_losses.append(float(loss))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        metrics["train_loss"].append(train_loss)

        val_loss = _evaluate(arch, val_ds, static, fluid, horizon, batch_size)
        metrics["val_loss"].append(val_loss)
        logger.info(
            "epoch %d  H=%d  train=%.4e  val=%s", epoch, horizon, train_loss, val_loss
        )

    # rollout-error-vs-horizon on val (the metric that matters, §6.5)
    final_h = _resolve_horizon(schedule, num_epochs - 1)[0]
    metrics["rollout_error_vs_horizon"] = _rollout_error(
        arch, val_ds, static, fluid, final_h, batch_size
    )

    checkpoint_dir = cfg.get("checkpoint_dir")
    if checkpoint_dir is not None:
        contract = ContractSchema(
            source_solver_name=corpus.contract.source_solver_name,
            param_schema=schema,
            state_var_names=var_names,
        )
        ic_bank = _build_ic_bank(corpus)
        save_checkpoint(
            checkpoint_dir, arch, arch_name=arch_name,
            arch_config=arch_cfg, history_len=history_len,
            normalization=norm, grid=corpus.grid, geometry_mask=mask,
            static_channels=np.asarray(corpus.static_channels), schema=contract,
            metrics={k: v for k, v in metrics.items() if k != "rollout_error_vs_horizon"},
            manifest_extra={"corpus_path": str(cfg["corpus_path"])},
            ic_bank=ic_bank,
        )
        logger.info("Saved checkpoint to %s", checkpoint_dir)

    return metrics


def _build_ic_bank(corpus) -> dict:
    """Canned IC bank (§4): the first frame of each train trajectory, keyed by
    its per-frame encoded conditioning. ``state=None`` at inference picks the
    nearest entry — cheap, unit-testable, and matches how window-0 ICs arise.
    """
    params, fields = [], []
    for traj_id in corpus.split_ids("train"):
        f = corpus.load_fields(traj_id)
        p = corpus.load_params(traj_id)
        fields.append(np.asarray(f[0]))  # [C, Z, Y, X]
        params.append(np.asarray(p[0]))  # [P]
    return {"params": np.stack(params), "fields": np.stack(fields)}


def _evaluate(arch, dataset, static, fluid, horizon, batch_size) -> float:
    dataset.set_horizon(horizon)
    if len(dataset) == 0:
        return float("nan")
    losses = []
    for batch in iterate_batches(dataset, batch_size, shuffle=False):
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        losses.append(float(loop_mod.eval_loss(arch, batch, static, fluid, horizon)))
    return float(np.mean(losses))


def _rollout_error(arch, dataset, static, fluid, horizon, batch_size) -> list[float]:
    dataset.set_horizon(horizon)
    if len(dataset) == 0:
        return []
    accum = None
    n = 0
    for batch in iterate_batches(dataset, batch_size, shuffle=False):
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        err = np.asarray(
            loop_mod.per_horizon_sq_error(arch, batch, static, fluid, horizon)
        )
        accum = err if accum is None else accum + err
        n += 1
    return (accum / max(n, 1)).tolist()


__all__ = ["run"]

"""§11.1 CPU smoke-training stage — plumbing, not accuracy.

Proves the training code runs end-to-end on a tiny synthetic corpus in RAM
(no solver, no GPU). Parametrized over shape, since mismatched dimensions are
the likeliest break. Asserts plumbing invariants only (shape + finiteness +
params actually change). One combination round-trips a checkpoint.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from neural_surrogates.architectures.registry import resolve_architecture
from neural_surrogates.data.dataset import InMemoryCorpus, WindowDataset, iterate_batches
from neural_surrogates.data.grid import GridMeta, build_static_channels
from neural_surrogates.data.normalization import fit_normalization
from neural_surrogates.training.checkpoint import load_checkpoint, save_checkpoint
from neural_surrogates.training.loop import inject_off_manifold, make_train_step
from neural_surrogates.utils.schema import ContractSchema, ParamSchema


def _build_corpus(T: int, grid_shape, n_channels: int, n_traj: int = 4):
    z, y, x = grid_shape
    rng = np.random.default_rng(0)
    var_names = ("u", "v", "w", "pres")[:n_channels]
    schema = ParamSchema(names=("inflow_angle", "velocity_magnitude"))
    p = schema.conditioning_dim
    grid = GridMeta(nx=x, ny=y, nz=z, bounds=((0.0, x), (0.0, y), (0.0, z)))

    mask = np.zeros((z, y, x), dtype=np.float32)
    mask[: max(1, z // 2), : max(1, y // 2), : max(1, x // 2)] = 1.0  # a solid block
    static = build_static_channels(mask, grid=grid)

    fields, params, splits = {}, {}, {"train": [], "val": []}
    for i in range(n_traj):
        tid = f"t{i}"
        fields[tid] = rng.standard_normal((T, n_channels, z, y, x)).astype(np.float32)
        params[tid] = rng.standard_normal((T, p)).astype(np.float32)
        (splits["train"] if i < n_traj - 1 else splits["val"]).append(tid)

    corpus = InMemoryCorpus(
        fields=fields, params=params, grid=grid, param_schema=schema,
        var_names=var_names, static_channels=static, splits=splits,
    )
    return corpus, mask, static, schema, var_names, grid, p


# Curated scenarios covering each axis (T / grid / K / H / C) without the full
# 72-cell cross-product, since each shape JIT-compiles a fresh UNet.
@pytest.mark.parametrize(
    "T,grid_shape,K,H,n_channels",
    [
        (2, (4, 4, 4), 1, 1, 3),    # T too short for K+H beyond one window; Markov
        (3, (3, 5, 7), 3, 1, 3),    # short history (history_len < K), anisotropic grid
        (7, (2, 4, 8), 3, 3, 4),    # pushforward H>1, +pres channel, non-cubic
        (20, (4, 4, 4), 3, 3, 3),   # many interior windows
        (7, (4, 4, 4), 1, 3, 3),    # K=1 Markov + multi-step pushforward
    ],
)
def test_training_plumbing(T, grid_shape, K, H, n_channels) -> None:
    corpus, mask, static, schema, var_names, grid, p = _build_corpus(
        T, grid_shape, n_channels
    )
    norm = fit_normalization(
        (corpus.load_fields(i) for i in corpus.split_ids("train")),
        var_names, mask=mask,
    )

    dataset = WindowDataset(corpus, "train", history_len=K, horizon=H, normalization=norm)
    # Windows may not exist if T is too short for K-anchoring + H targets.
    if len(dataset) == 0:
        assert T - 1 - H < 0
        return

    record = dataset[0]
    assert record.hist_fields.shape == (K, n_channels, *grid_shape)
    assert record.future_params.shape == (H, p)
    assert record.target_fields.shape == (H, n_channels, *grid_shape)
    assert record.hist_mask.shape == (K,)

    arch = resolve_architecture(
        "unet3d",
        dict(in_state_channels=n_channels, static_channels=static.shape[0],
             history_len=K, param_dim=p, base_channels=4,
             channel_multipliers=[1, 2], num_res_blocks_per_level=1, embed_dim=8),
        key=jax.random.PRNGKey(0),
    )

    optimizer = optax.adam(1e-3)
    import equinox as eqx

    opt_state = optimizer.init(eqx.filter(arch, eqx.is_inexact_array))
    train_step = make_train_step(optimizer)

    static_j = jnp.asarray(static)
    fluid = 1.0 - jnp.asarray(mask)

    before = jax.tree_util.tree_leaves(eqx.filter(arch, eqx.is_inexact_array))
    losses = []
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(1)
    for batch in iterate_batches(dataset, batch_size=2, rng=rng):
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        key, sub = jax.random.split(key)
        batch = inject_off_manifold(batch, sub, noise_std=0.05)
        arch, opt_state, loss = train_step(arch, opt_state, batch, static_j, fluid, H, 0)
        losses.append(float(loss))

    assert all(np.isfinite(loss) for loss in losses)
    after = jax.tree_util.tree_leaves(eqx.filter(arch, eqx.is_inexact_array))
    # gradients flowed: at least one parameter array changed
    changed = any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(before, after))
    assert changed


def test_checkpoint_roundtrip(tmp_path) -> None:
    """One wired combination round-trips a throwaway checkpoint (§11.1)."""
    corpus, mask, static, schema, var_names, grid, p = _build_corpus(
        T=8, grid_shape=(4, 4, 4), n_channels=3
    )
    norm = fit_normalization(
        (corpus.load_fields(i) for i in corpus.split_ids("train")), var_names, mask=mask
    )
    arch_config = dict(
        in_state_channels=3, static_channels=static.shape[0], history_len=2,
        param_dim=p, base_channels=4, channel_multipliers=[1, 2], embed_dim=8,
    )
    arch = resolve_architecture("unet3d", arch_config, key=jax.random.PRNGKey(0))
    contract = ContractSchema(
        source_solver_name="pylbm", param_schema=schema, state_var_names=var_names
    )

    ckpt_dir = tmp_path / "run0"
    save_checkpoint(
        ckpt_dir, arch, arch_name="unet3d", arch_config=arch_config, history_len=2,
        normalization=norm, grid=grid, geometry_mask=mask, static_channels=static,
        schema=contract, metrics={"val_loss": 0.1},
    )

    loaded = load_checkpoint(ckpt_dir, expected_architecture="unet3d")
    assert loaded.arch_name == "unet3d"
    assert loaded.history_len == 2
    assert loaded.schema.source_solver_name == "pylbm"
    assert loaded.grid.matches(grid)

    # restored weights equal the saved ones
    import equinox as eqx

    a = jax.tree_util.tree_leaves(eqx.filter(arch, eqx.is_inexact_array))
    b = jax.tree_util.tree_leaves(eqx.filter(loaded.arch, eqx.is_inexact_array))
    for x_arr, y_arr in zip(a, b):
        np.testing.assert_allclose(np.asarray(x_arr), np.asarray(y_arr))

    with pytest.raises(ValueError):
        load_checkpoint(ckpt_dir, expected_architecture="upt")

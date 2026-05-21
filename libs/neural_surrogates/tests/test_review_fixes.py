"""Tests for the branch-review fixes (#3 dims, #4 hist_mask, #5 pressure mask)."""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from neural_surrogates.architectures.registry import resolve_architecture
from neural_surrogates.data.grid import GridMeta
from neural_surrogates.forward_model import ForwardModel
from neural_surrogates.utils import state_io
from neural_surrogates.utils.schema import ContractSchema, ParamSchema


def test_to_collocated_dims_renames_udales_centers() -> None:
    # uDALES collocated output uses zt/yt/xt; state_to_tensor must accept it.
    grid = GridMeta(nx=5, ny=4, nz=3, bounds=((0.0, 5.0), (0.0, 4.0), (0.0, 3.0)))
    rng = np.random.default_rng(0)
    ds = xarray.Dataset(
        {
            v: (("time", "zt", "yt", "xt"), rng.standard_normal((2, 3, 4, 5)))
            for v in ("u", "v", "w")
        },
        coords={"time": [0, 1], "zt": grid.z, "yt": grid.y, "xt": grid.x},
    )
    tensor = state_io.state_to_tensor(ds, grid, ("u", "v", "w"))
    assert tensor.shape == (2, 3, 3, 4, 5)


def test_unet_init_carry_fills_padded_slots_with_first_real_frame() -> None:
    arch = resolve_architecture(
        "unet3d",
        dict(in_state_channels=3, static_channels=0, history_len=3, param_dim=2,
             base_channels=4, channel_multipliers=[1, 2], embed_dim=8),
        key=jax.random.PRNGKey(0),
    )
    real = jax.random.normal(jax.random.PRNGKey(1), (1, 3, 4, 4, 4))
    tensor = np.asarray(real)
    hist, mask = state_io.extract_history(tensor, history_len=3)  # 2 pad + 1 real
    carry = arch.init_carry(jnp.asarray(hist), jnp.zeros((3, 2)), jnp.asarray(mask), jnp.zeros((0, 4, 4, 4)))
    # all three carry slots should equal the single real frame (no zero frames)
    for k in range(3):
        np.testing.assert_allclose(np.asarray(carry[k]), tensor[0], atol=1e-6)


def _stub_forward_model(var_names, mask) -> ForwardModel:
    fm = ForwardModel.__new__(ForwardModel)
    schema = ContractSchema("pylbm", ParamSchema(("inflow_angle", "velocity_magnitude")), var_names)
    fm._ckpt = types.SimpleNamespace(geometry_mask=mask, schema=schema)
    return fm


def test_reapply_mask_zeros_velocity_but_not_pressure() -> None:
    z, y, x = 2, 2, 2
    mask = np.zeros((z, y, x), dtype=np.float32)
    mask[0, 0, 0] = 1.0  # one solid cell
    fm = _stub_forward_model(("u", "v", "w", "pres"), mask)

    preds = np.ones((1, 4, z, y, x), dtype=np.float32)
    out = fm._reapply_mask(preds)

    # velocity channels zeroed in the solid cell
    assert out[0, 0, 0, 0, 0] == 0.0  # u
    assert out[0, 2, 0, 0, 0] == 0.0  # w
    # pressure channel untouched in the solid cell
    assert out[0, 3, 0, 0, 0] == 1.0
    # fluid cells untouched everywhere
    assert out[0, 0, 1, 1, 1] == 1.0

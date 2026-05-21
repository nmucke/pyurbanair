"""Data-parallel sharding helpers (``docs/neural_surrogate_plan.md`` §6.4, P4).

A 1-D JAX ``Mesh`` over local devices: shard the batch axis, replicate params.
The training step function stays **mesh-agnostic** (it just sees arrays), so the
same step works for any architecture and degrades to a single device when only
one is present. Model/tensor parallelism is intentionally *not* built — a hook,
not a feature (§6.4).
"""

from __future__ import annotations

from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

BATCH_AXIS = "batch"


def make_data_parallel_mesh() -> Mesh:
    """1-D mesh over all local devices on the ``batch`` axis."""
    devices = jax.devices()
    return Mesh(devices, axis_names=(BATCH_AXIS,))


def shard_batch(batch: dict[str, Any], mesh: Mesh) -> dict[str, Any]:
    """Shard each batch array along its leading (batch) axis across the mesh."""
    sharding = NamedSharding(mesh, PartitionSpec(BATCH_AXIS))
    return {k: jax.device_put(v, sharding) for k, v in batch.items()}


def replicate(tree: Any, mesh: Mesh) -> Any:
    """Replicate a PyTree (e.g. model params) across all devices."""
    sharding = NamedSharding(mesh, PartitionSpec())
    return jax.device_put(tree, sharding)

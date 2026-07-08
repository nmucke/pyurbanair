"""Signed-distance-field geometry features for the neural surrogates.

A single source of truth, used by *both* the training dataloader
(:class:`~neural_surrogates.datasets.transition.TransitionDataset`) and the
model (:class:`~neural_surrogates.architectures.P3D`), so the two paths can
never drift apart.

From the binary fluid mask ``m`` (``1`` = fluid, ``0`` = solid -- buildings +
ground) it derives 4 extra channels on the voxel grid:

``sdf``   = ``EDT(m) - EDT(1 - m)``    signed distance in **cells**
                                        (``> 0`` in fluid, ``< 0`` in solid)
``sdf_n`` = ``clip(sdf, -L, L) / L``   clamped SDF, ``L = clamp_cells``
``g_z, g_y, g_x`` = ``∇sdf / max(‖∇sdf‖, eps)``   unit gradient of the
                                        **unclamped** SDF

All four channels are bounded in ``[-1, 1]`` by construction (like the mask,
they enter the network stem raw -- no z-scoring). Distances are in **cells**,
not physical units: the model only ever sees a voxel mask at inference and has
no grid spacing, and the strict domain check already pins the trained cell
spacing to the deployed one. An optional ``spacing`` argument (mapped to
scipy's ``sampling``) is kept for future anisotropic-grid use but defaults off.

``scipy.ndimage.distance_transform_edt`` is ``O(N)`` in grid cells and
independent of building count, so the two EDTs + one gradient cost tens of
milliseconds on a ~128^3 grid -- paid once per geometry (never per rollout
step).
"""

from __future__ import annotations

import numpy as np
import torch

# Number of derived channels in the FULL stack: sdf_n + (g_z, g_y, g_x).
N_SDF_FEATURE_CHANNELS = 4

# Channel order of the full stack, and the sub-selection each feature mode keeps
# (indices into that order). ``sdf`` -> the clamped signed distance only, ``grad``
# -> the 3 unit-gradient components only, ``both`` -> all 4, ``none`` -> disabled.
# This dict is the single source of truth for *which* channels a mode selects and
# *how many* (its length); the dataset and the model both key off it, so the two
# paths can never disagree on channel count or order.
_SDF_MODE_INDICES: dict[str, tuple[int, ...]] = {
    "none": (),
    "sdf": (0,),
    "grad": (1, 2, 3),
    "both": (0, 1, 2, 3),
}


def normalize_sdf_mode(mode: bool | str) -> str:
    """Canonicalise an ``sdf_features`` value to one of the mode strings.

    Accepts the mode strings (``"none" | "sdf" | "grad" | "both"``) and, for
    backward compatibility with the old boolean flag (and already-trained
    ``config.yaml`` files that stored it), ``True`` -> ``"both"`` and ``False``
    -> ``"none"``. Raises ``ValueError`` on anything else.
    """
    if isinstance(mode, bool):
        return "both" if mode else "none"
    if isinstance(mode, str) and mode in _SDF_MODE_INDICES:
        return mode
    raise ValueError(
        f"sdf_features must be one of {sorted(_SDF_MODE_INDICES)} (or a bool), "
        f"got {mode!r}"
    )


def n_sdf_feature_channels(mode: bool | str) -> int:
    """Number of SDF feature channels selected by ``mode`` (0 when disabled)."""
    return len(_SDF_MODE_INDICES[normalize_sdf_mode(mode)])


def _sdf_features_single(
    mask: np.ndarray,
    clamp_cells: float,
    spacing: tuple[float, float, float] | None,
    eps: float,
    indices: tuple[int, ...],
) -> np.ndarray:
    """Compute the feature stack for one 3-D mask (numpy).

    The full ``[sdf_n, g_z, g_y, g_x]`` stack is always computed (the gradient
    needs the unclamped SDF regardless), then sliced down to ``indices`` -- so
    the returned shape is ``(len(indices), *grid)``.
    """
    from scipy.ndimage import distance_transform_edt

    fluid = mask.astype(bool)
    # distance_transform_edt returns, per non-zero cell, the distance to the
    # nearest zero cell. So EDT(fluid) is the distance to the nearest solid
    # (zero inside solids) and EDT(~fluid) the distance to the nearest fluid
    # (zero inside fluid); their difference is the signed distance.
    d_out = distance_transform_edt(fluid, sampling=spacing)
    d_in = distance_transform_edt(~fluid, sampling=spacing)
    sdf = np.asarray(d_out, dtype=np.float64) - np.asarray(d_in, dtype=np.float64)

    sdf_n = np.clip(sdf, -clamp_cells, clamp_cells) / clamp_cells

    # Central differences of the UNCLAMPED sdf with replicated edges, then
    # normalise each voxel's gradient to unit length. Working on the unclamped
    # field keeps direction information alive beyond the clamp radius; the
    # explicit normalisation cleans up the finite-difference approximation and
    # any plateau cells. Sign convention: positive-in-fluid sdf means the
    # gradient points AWAY from the nearest wall.
    padded = np.pad(sdf, 1, mode="edge")
    grads = []
    for axis in range(sdf.ndim):
        hi = [slice(1, -1)] * sdf.ndim
        lo = [slice(1, -1)] * sdf.ndim
        hi[axis] = slice(2, None)
        lo[axis] = slice(0, -2)
        step = 2.0 if spacing is None else 2.0 * spacing[axis]
        grads.append((padded[tuple(hi)] - padded[tuple(lo)]) / step)
    grad = np.stack(grads, axis=0)  # (ndim, *grid)
    norm = np.sqrt(np.sum(grad**2, axis=0, keepdims=True))
    grad_unit = grad / np.maximum(norm, eps)

    full = np.concatenate([sdf_n[None], grad_unit], axis=0)  # (4, *grid)
    return np.ascontiguousarray(full[list(indices)])  # (len(indices), *grid)


def sdf_features(
    mask: torch.Tensor,
    clamp_cells: float = 32.0,
    mode: bool | str = "both",
    spacing: tuple[float, float, float] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Signed-distance-field feature channels for a fluid mask.

    Args:
        mask: ``(*grid,)`` or ``(B, *grid)`` tensor, ``1`` = fluid, ``0`` =
            solid. ``*grid`` must be 3-D (``z, y, x``).
        clamp_cells: clamp radius ``L`` (in cells) for the normalised SDF
            channel; the SDF is clipped to ``[-L, L]`` and divided by ``L``.
        mode: which channels of the full ``[sdf_n, g_z, g_y, g_x]`` stack to
            return -- ``"both"`` (all 4), ``"sdf"`` (the clamped SDF only, 1
            channel), or ``"grad"`` (the 3 unit-gradient components only).
            ``True`` is accepted as an alias for ``"both"``. Only these
            channel-producing values are valid here: ``"none"`` (and its alias
            ``False``) select zero channels and **raise** -- this low-level
            helper is never asked to compute an empty stack (callers gate on the
            mode first). Defaults to ``"both"`` so external callers of
            ``sdf_features(mask)`` keep the original full-stack behaviour; note
            the dataset/model constructors instead default to ``"none"`` (off).
        spacing: optional ``(dz, dy, dx)`` cell spacing forwarded to scipy's
            EDT ``sampling`` (and the gradient step). ``None`` (default) works
            in cell units -- the trained/deployed convention.
        eps: gradient-norm floor so plateau cells give a zero (not NaN) unit
            gradient.

    Returns:
        ``(C, *grid)`` for an unbatched mask, or ``(B, C, *grid)`` for a
        batched one, where ``C = n_sdf_feature_channels(mode)``, on the input's
        device and dtype. Channel order follows the full stack
        ``[sdf_n, g_z, g_y, g_x]``, restricted to the selected channels.

    The EDT runs on CPU via numpy/scipy; the single ``.cpu()`` round-trip is
    paid once per geometry, never per rollout step.
    """
    indices = _SDF_MODE_INDICES[normalize_sdf_mode(mode)]
    if not indices:
        raise ValueError("sdf_features called with mode 'none' (no channels)")
    if mask.dim() not in (3, 4):
        raise ValueError(
            "sdf_features expects a (z, y, x) or (B, z, y, x) mask, got shape "
            f"{tuple(mask.shape)}"
        )
    batched = mask.dim() == 4
    arr = mask.detach().to("cpu").numpy()
    if batched:
        feats = np.stack(
            [
                _sdf_features_single(arr[b], clamp_cells, spacing, eps, indices)
                for b in range(arr.shape[0])
            ],
            axis=0,
        )
    else:
        feats = _sdf_features_single(arr, clamp_cells, spacing, eps, indices)
    return torch.from_numpy(feats).to(device=mask.device, dtype=mask.dtype)

"""Inference-time y-periodicity for trained surrogates (no retraining).

uDALES domains are periodic in y, but the steppers were trained with every
convolution zero-padding y, so rollouts grow a seam at the y edges.
:func:`make_periodic_y` patches a trained model in place so it sees a y-periodic
domain: every ``Conv3d`` pads y circularly (z and x keep their zero padding) and
the SDF geometry features are computed on a y-periodic mask
(``neural_surrogates.sdf.sdf_features(..., periodic_y=True)``).

Supported: ``TadpoleTimeStepper`` / ``TadpoleAE`` in ``spatial_mode`` global (ny
a multiple of 16) or local with one crop spanning y (ny == crop y), ``P3D``
without shifted windows, ``UNetConvNeXt``, ``SimpleConv``, and LoRA-wrapped
convs (peft's ``lora_A`` conv is patched like its base). Already y-periodic
parts are left as they are: the baselines built with ``periodic_axes``
containing ``"y"`` (``periodic_y`` attribute ``True``) are returned unchanged,
and the convs of ``UNetConvNeXt`` / ``P3D`` built with ``"y"`` in
``periodic_axes`` are skipped (a ``P3D``'s self-computed SDF still turns
y-periodic).

Raises instead of half-patching: ``UPT`` (point cloud, no y grid),
``TadpoleLatentGenerator`` (absolute positions on the latent grid),
``DomainDecomposed`` (configure ``decomposition.periodic_axes``), Tadpole halo
mode (edge regions lack wrapped context), shifted-window attention that masks
the y edges, baselines built non-periodic in y, padded Conv1d/Conv2d,
ConvTranspose3d with y padding and convs with a non-zero, non-circular padding
mode. Left as is (measured harmless on ``dft_b``): window attention (the Tadpole AE's
and ``P3D``'s ``maybe_pad``) zero-pads its token grid up to the window size, and
the Tadpole latent transformer adds absolute sin-cos positions.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn.functional as F
from neural_surrogates.architectures import (
    P3D,
    UPT,
    DomainDecomposed,
    TadpoleLatentGenerator,
    UNetConvNeXt,
)
from neural_surrogates.architectures._tadpole_field_io import _TadpoleFieldIO
from torch import nn
from torch.nn.modules.conv import _ConvNd

_UNSUPPORTED = {
    UPT: "it works on point clouds, not a y-padded grid",
    TadpoleLatentGenerator: "its velocity net adds absolute latent-grid positions",
    DomainDecomposed: "set decomposition.periodic_axes instead",
}


def _circular_y_pad(p: int, module: nn.Module, args: tuple[torch.Tensor, ...]) -> Any:
    """Forward pre-hook: pad the (positional) input circularly by ``p`` in y.
    Module level (bound with ``functools.partial``) so a patched model pickles."""
    return (F.pad(args[0], (0, 0, p, p, 0, 0), mode="circular"), *args[1:])


def _check_module(name: str, m: nn.Module) -> None:
    """Raise if module ``m`` has a y-aware part the patch cannot make periodic."""
    for cls, why in _UNSUPPORTED.items():
        if isinstance(m, cls):
            raise ValueError(f"make_periodic_y: {cls.__name__} {name!r}: {why}")
    if isinstance(m, _TadpoleFieldIO) and m.spatial_mode == "halo":
        raise ValueError(
            f"make_periodic_y: {name!r} uses spatial_mode='halo', whose edge "
            "regions get no wrapped context; use 'global'"
        )
    # Shifted-window attention (Tadpole's and p3d_surrogate's P3DStage) masks
    # the wrap-around of every non-periodic axis.
    if getattr(m, "shift_size", 0) > 0 and hasattr(m, "periodic"):
        periodic = m.periodic
        y_periodic = periodic[1] if isinstance(periodic, (list, tuple)) else periodic
        if not y_periodic:
            raise ValueError(
                f"make_periodic_y: {name!r} uses shifted-window attention that "
                "masks the y edges"
            )


def _conv_action(name: str, m: _ConvNd) -> str | None:
    """How to make conv ``m`` y-periodic: ``"hook"``, ``"mode"`` or ``None``."""
    if m.padding_mode == "circular":  # already wraps every axis
        return None
    pads = m.padding
    if isinstance(pads, str):
        if pads == "valid":
            return None
        raise ValueError(f"make_periodic_y: conv {name!r} has padding={pads!r}")
    if len(m.kernel_size) != 3:
        if any(pads):
            raise ValueError(
                f"make_periodic_y: padded {type(m).__name__} {name!r} (unknown axes)"
            )
        return None
    if pads[1] == 0:
        return None
    if m.transposed:
        raise ValueError(f"make_periodic_y: {name!r} is transposed with y padding")
    if isinstance(m.padding_mode, tuple):  # p3d_surrogate's Conv3dVariablePad
        return None if m.padding_mode[1] == "circular" else "mode"
    if m.padding_mode != "zeros" or not isinstance(m, nn.Conv3d):
        raise ValueError(
            f"make_periodic_y: {type(m).__name__} {name!r} with padding_mode="
            f"{m.padding_mode!r}"
        )
    return "hook"


def make_periodic_y(model: nn.Module) -> nn.Module:
    """In place: circular y padding in every 3-D conv (z/x stay zero-padded)
    and y-periodic self-computed SDF features; returns ``model``. Idempotent.

    Everything is checked before anything is changed, so a raise leaves the
    model untouched. Tensors are ``(B, C, z, y, x)``, y is dim -2; patched
    ``Conv3d`` take their input positionally. SDF features a caller passes in
    (``geom_features``) are used as given: compute them with the patched
    model's ``_sdf_features`` (as ``NeuralSurrogateForwardModel`` does) or
    ``sdf_features(..., periodic_y=True)``, not from a dataset.
    """
    own = getattr(model, "periodic_y", None)
    if own:
        return model
    if own is not None:  # the model's own switch (the baselines), built off
        raise ValueError(
            f"make_periodic_y: {type(model).__name__} was built non-periodic in "
            "y; rebuild it with 'y' in periodic_axes"
        )
    actions = []
    for name, m in model.named_modules():
        _check_module(name, m)
        if isinstance(m, _ConvNd):
            action = _conv_action(name, m)
            if action is not None:
                actions.append((action, m))

    for action, m in actions:
        pz, py, px = m.padding
        if action == "hook":
            m.register_forward_pre_hook(functools.partial(_circular_y_pad, py))
            m.padding = (pz, 0, px)
        else:
            mz, _, mx = m.padding_mode
            m.padding_mode = (mz, "circular", mx)
    for m in model.modules():
        if isinstance(m, (_TadpoleFieldIO, P3D)):
            # read by their SDF path (Tadpole: also the y-padding check)
            setattr(m, "periodic_y", True)
        if isinstance(m, (P3D, UNetConvNeXt)):  # raises on a y they would pad
            pz_, _, px_ = m._periodic
            m._periodic = (pz_, True, px_)
        # Features cached before the patch are not y-periodic.
        if getattr(m, "_sdf_cache", None) is not None:
            m._sdf_cache = None
        if getattr(m, "_branch_cache", None) is not None:
            m._branch_cache = None
    model.periodic_y = True
    return model

"""Patch discriminator + GAN loss helpers for Tadpole autoencoder pre-training.

``TadpoleAE`` (:mod:`neural_surrogates.architectures.tadpole_ae`) is trained with
a masked-MSE + KL objective, which is exactly the loss family that produces
blurry reconstructions: MSE is minimised by the conditional *mean*, so the small
high-wavenumber structure of an urban flow (shear layers, wakes, the sharp
gradients hugging building faces) is averaged away. The latent-diffusion /
VQGAN recipe fixes this with a **patch discriminator**: a small convolutional
critic that scores local patches of the reconstruction against the ground truth,
adding an adversarial term that rewards texture the pixel loss cannot see.

This module wraps the vendored ``P3DDiscriminator``
(:mod:`neural_surrogates.architectures._tadpole.architecture.discriminator`,
from `tum-pbs/Tadpole <https://github.com/tum-pbs/Tadpole>`_ — vendored for this
purpose but until now unused) behind this repo's conventions, and ships the three
plain-function loss helpers the trainer needs. It is a deliberately small,
**optional** extension of the AE pre-training loss: no EMA, no spectral norm, no
gradient penalty, no multi-scale critic.

Deliberate deviation from upstream — do not "fix" this back
------------------------------------------------------------
Tadpole's paper (Appendix C.1) builds the critic from the same encoder
architecture as the autoencoder (minus its final projection), and feeds it the
same **folded, single-channel** crops the encoder sees: ``(B, C, X, Y, Z)`` is
rearranged to ``(B*C, 1, 64, 64, 64)`` with ``in_channels=1``. Its discriminator
therefore never sees state and geometry — or even two state components —
together.

We deliberately do **not** fold. This repo needs the critic to judge the
reconstruction *in the presence of its geometry*: obstacle-adjacent sharpness is
exactly the artefact an urban-flow AE gets wrong, and scoring every channel in
isolation makes that structurally invisible. So we feed the **unfolded,
multi-channel** field on the full grid, with the geometry mask (and any SDF
feature channels) concatenated as extra input channels —
``in_channels = n_state_channels + n_geometry_channels``. The critic can then
police cross-channel, geometry-aware structure (is this wake consistent across
``u, v, w``? is the boundary layer at this wall as sharp as a real one?), which a
per-channel view cannot express.

Design notes
------------
* **Patch logits, not a scalar.** The vendored critic is a P3D encoder backbone
  followed by a ``1x1x1`` ``Conv3d`` to one channel, so ``forward`` returns a
  logit *map* ``(B, 1, d, h, w)`` (one logit per receptive-field patch, the
  backbone downsamples spatially by 16). The paper's scalar "belief" is the
  **mean** of that map, which is what the hinge losses below already take.
* **Size.** The paper pairs the critic with an autoencoder of the *same* size, so
  ``size`` should normally match the ``TadpoleAE`` it polices (the default
  ``"S"`` matches the default AE).
* **Padding.** The backbone's total downsampling is 16, so each spatial dim is
  zero-padded up to the next multiple of 16 inside :meth:`forward`. There is no
  crop-back to do (the output is a patch map, not a field), so this uses a small
  local helper rather than ``_TadpoleFieldIO._pad_to_crop_multiple``: that one is
  driven by the host's ``encoder_crop_size`` attribute and returns the original
  spatial shape for the crop-back the AE needs — machinery with no meaning here.

The heavy vendored stack (``diffusers`` / ``timm``) is imported lazily inside
``__init__`` so ``import neural_surrogates`` stays light, mirroring
``tadpole_ae.py``.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from neural_surrogates.sdf import n_sdf_feature_channels, normalize_sdf_mode
from torch import nn

# The vendored P3D backbone ships four size presets; the discriminator is a
# critic, not the model under training, so "S" is the sensible default.
_SIZES = ("S", "B", "L", "XL")

# Total spatial downsampling of the P3D encoder backbone (conv stem + transformer
# stages). Inputs must be a multiple of this on every spatial axis.
_DOWNSAMPLE = 16


class TadpoleDiscriminator(nn.Module):
    """3-D patch critic over a Tadpole AE's working-space field.

    Parameters
    ----------
    n_state_channels:
        Number of state channels ``C`` the autoencoder reconstructs (e.g. 3 for
        ``u, v, w``).
    size:
        Backbone size ``"S"`` / ``"B"`` / ``"L"`` / ``"XL"`` (the vendored P3D
        encoder presets). The paper pairs the critic with a same-size
        autoencoder, so this should normally match the ``TadpoleAE``'s ``size``
        (``"S"`` is ~3.2M params for a 5-channel input).
    encode_geometry:
        Whether the field handed to the critic carries the autoencoder's geometry
        block (mask + any SDF channels) alongside the state channels. Must match
        the ``TadpoleAE``'s own ``encode_geometry`` so the channel counts line up.
    sdf_features:
        Which SDF channels ride along with the mask: ``"none"`` / ``"sdf"`` (+1) /
        ``"grad"`` (+3) / ``"both"`` (+4) (``True``/``False`` alias
        ``"both"``/``"none"``). Requires ``encode_geometry=True``, exactly as in
        :class:`~neural_surrogates.architectures.tadpole_ae.TadpoleAE`.

    Attributes
    ----------
    n_input_channels:
        ``n_state_channels + (1 + n_sdf_feature_channels(mode))`` when geometry is
        encoded, else ``n_state_channels`` — the channel count :meth:`forward`
        expects.
    """

    def __init__(
        self,
        n_state_channels: int,
        size: str = "S",
        encode_geometry: bool = True,
        sdf_features: bool | str = "none",
    ) -> None:
        super().__init__()

        # Lazy import: keep `import neural_surrogates` free of the heavy
        # diffusers/timm stack pulled in by the vendored backbone. A clean
        # ImportError here is what lets test suites importorskip the deps.
        try:
            from neural_surrogates.architectures._tadpole.architecture.discriminator import (  # noqa: E501
                P3DDiscriminator,
            )
        except ImportError as exc:  # pragma: no cover - exercised only when absent
            raise ImportError(
                "TadpoleDiscriminator requires the vendored autoencoder's runtime "
                "deps ('diffusers', 'timm', 'einops'); install "
                "`neural_surrogates[tadpole]`."
            ) from exc

        if size not in _SIZES:
            raise ValueError(f"size must be one of {list(_SIZES)}, got {size!r}")

        self.n_state_channels = int(n_state_channels)
        self.size = size
        self.encode_geometry = bool(encode_geometry)

        self.sdf_feature_mode = normalize_sdf_mode(sdf_features)
        self.sdf_features_enabled = self.sdf_feature_mode != "none"
        self.n_geom_feature_channels = n_sdf_feature_channels(self.sdf_feature_mode)
        if self.sdf_features_enabled and not self.encode_geometry:
            raise ValueError(
                "sdf_features requires encode_geometry=True (the SDF channels are "
                "appended alongside the geometry mask)."
            )

        # Geometry block: the mask (+1) plus any SDF channels, mirroring the AE's
        # working-space layout so a working-space reconstruction can be fed here
        # unchanged.
        self.n_geometry_channels = (
            1 + self.n_geom_feature_channels if self.encode_geometry else 0
        )
        self.n_input_channels = self.n_state_channels + self.n_geometry_channels

        # The vendored backbone types `size` as a Literal; it is validated above.
        self.disc = P3DDiscriminator(
            cast(Literal["S", "B", "L", "XL"], size),
            in_channels=self.n_input_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score ``x`` ``(B, n_input_channels, D, H, W)`` -> logits ``(B, 1, d, h, w)``.

        Positive logits mean "real"; the map is one logit per patch (the backbone
        downsamples by 16 on each axis). Spatial dims are zero-padded up to a
        multiple of 16 first — the padded shell simply contributes a few extra
        patches, which both the real and the fake pass see identically.
        """
        if x.dim() != 5:
            raise ValueError(
                f"expected a 5-D (B, C, D, H, W) field, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.n_input_channels:
            raise ValueError(
                f"expected {self.n_input_channels} input channels "
                f"({self.n_state_channels} state + {self.n_geometry_channels} "
                f"geometry), got {x.shape[1]}"
            )
        # `PatchDiscriminator.forward` splats *args into the backbone, whose own
        # forward takes only `x` -- always call it with a single argument.
        logits: torch.Tensor = self.disc(_pad_to_multiple(x, _DOWNSAMPLE))
        return logits


def _pad_to_multiple(x: torch.Tensor, mult: int) -> torch.Tensor:
    """Zero-pad the trailing ``(D, H, W)`` up to a multiple of ``mult``."""
    d, h, w = x.shape[-3:]
    pad_d, pad_h, pad_w = ((mult - s % mult) % mult for s in (d, h, w))
    if pad_d or pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
    return x


# --------------------------------------------------------------------------- #
# Loss helpers (plain functions -- the trainer owns the schedule around them).
# --------------------------------------------------------------------------- #


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge discriminator loss (the VQGAN / latent-diffusion convention).

    ``0.5 * (mean(relu(1 - logits_real)) + mean(relu(1 + logits_fake)))`` -- zero
    once the critic separates the two sides by the unit margin, which is what
    keeps it from running away from the generator.
    """
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating hinge generator loss (VQGAN convention): ``-mean(logits_fake)``.

    Unlike the critic's side there is no margin: the generator always gains from
    pushing its logits higher, so the gradient never vanishes.
    """
    return -logits_fake.mean()


def adaptive_adv_weight(
    recon_loss: torch.Tensor,
    adv_loss: torch.Tensor,
    last_layer: torch.nn.Parameter,
    *,
    eps: float = 1e-4,
    max_weight: float = 1.0,
) -> torch.Tensor:
    """VQGAN/LDM adaptive adversarial weight, as a detached scalar.

    ``||d recon/d last_layer|| / (||d adv/d last_layer|| + eps)``, clamped to
    ``[0, max_weight]``: it balances the two gradients arriving at the decoder's
    output layer, so the adversarial term never overwhelms reconstruction (and
    needs no per-dataset hand tuning). Falls back to ``1.0`` whenever the ratio is
    undefined -- either loss disconnected from ``last_layer``, or a non-finite
    ratio -- so a degenerate step costs a neutral weight, not a crash.

    ``max_weight`` defaults to ``1.0`` (not taming-transformers' ``1e4``) because
    the trainer multiplies this factor by its configured ``adv_weight``, whose
    default is ``1e-4``: the ceiling then puts the *effective* adversarial
    coefficient in ``[0, 1e-4]``, matching Tadpole's Appendix C.2 ("a
    gradient-based scale strategy (Esser et al., 2021) ... with a maximum scale
    value of 1e-4"). It stays a parameter so the ceiling remains tunable.
    """
    fallback = torch.ones((), device=last_layer.device, dtype=last_layer.dtype)
    try:
        # allow_unused: a term not reaching `last_layer` yields None, not a raise;
        # the except covers the harder case of a loss with no graph at all.
        grads: list[torch.Tensor | None] = [
            torch.autograd.grad(loss, last_layer, retain_graph=True, allow_unused=True)[
                0
            ]
            for loss in (recon_loss, adv_loss)
        ]
    except RuntimeError:  # e.g. a fully detached loss (no grad_fn to walk)
        return fallback
    recon_grad, adv_grad = grads
    if recon_grad is None or adv_grad is None:
        return fallback
    weight: torch.Tensor = recon_grad.norm() / (adv_grad.norm() + eps)
    if not torch.isfinite(weight):
        return fallback
    return weight.clamp(0.0, max_weight).detach()


def resolve_last_decoder_layer(model: nn.Module) -> nn.Parameter | None:
    """Weight of the final convolution of a wrapped Tadpole decoder, or ``None``.

    :func:`adaptive_adv_weight` needs the generator's output layer. For
    :class:`~neural_surrogates.architectures.tadpole_ae.TadpoleAE` that is
    ``model.ae.decoder.conv_decoder.decompress`` -- the ``Conv3d`` that maps the
    last upsampled feature stack to the output channel in the vendored
    ``ConditionedDecoder3D``. That explicit path is tried first.

    The fallback is the *last* ``nn.Conv3d`` leaf under ``model.ae.decoder``, for
    a decoder whose output conv has been re-wrapped (e.g. by PEFT) so the
    attribute path no longer resolves to a plain ``Conv3d``. Note it is only a
    best-effort: ``ConditionedDecoder3D`` registers ``decompress`` *before* its
    upsampling blocks, so on the unmodified vendored tree the fallback would pick
    an upsampling conv instead -- which is why the explicit path comes first.

    ``None`` (no ``ae.decoder``, no conv at all) is the trainer's signal to fall
    back to a fixed adversarial weight.
    """
    decoder = getattr(getattr(model, "ae", None), "decoder", None)
    if decoder is None:
        return None

    conv = getattr(getattr(decoder, "conv_decoder", None), "decompress", None)
    if not isinstance(conv, nn.Conv3d):
        convs = [m for m in decoder.modules() if isinstance(m, nn.Conv3d)]
        conv = convs[-1] if convs else None
    if conv is None:
        return None

    weight = conv.weight
    return weight if isinstance(weight, nn.Parameter) else None

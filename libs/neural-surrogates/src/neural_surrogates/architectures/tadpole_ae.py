"""Tadpole autoencoder wrapper (plan 02: foundation-model pre-training).

``TadpoleAE`` wraps the vendored ``TadpoleAutoencoder``
(:mod:`neural_surrogates.architectures._tadpole`, from Liu et al.,
`tum-pbs/Tadpole <https://github.com/tum-pbs/Tadpole>`_) behind this repo's
conventions so it can be pre-trained on our urban-flow snapshots as a
representation-learning (V)AE -- **no** next-step objective. It is trained by
:class:`neural_surrogates.training.AutoencoderTrainer` and is **never** an ESMDA
forward model (plan 03 turns a pre-trained AE into a time-stepper).

What the wrapper adds around the raw autoencoder
------------------------------------------------
* **Per-channel z-score normalisation** (``normalize=True``) with buffered
  training statistics installed via :meth:`set_normalization` -- the same
  contract as :class:`~neural_surrogates.architectures.p3d.P3D` /
  ``UPT``, so the pre-train script's ``get_normalization_stats`` +
  ``set_normalization`` path just works. Standardising *before* the autoencoder
  folds channels into the batch means each folded **state** crop is ~``N(0, 1)``,
  matching Tadpole's pre-training statistics (this is what matters when starting
  from the HF ``thuerey-group/Tadpole`` weights, whose encoder was trained on
  standardised single-channel fields). Note this holds only for the **state**
  crops: the geometry-block crops below are fed **raw** (see the next bullet), so
  the HF encoder does see out-of-distribution inputs on those few auxiliary
  channels -- acceptable because they carry a bounded, near-constant geometry cue
  the AE learns from scratch, not primary flow statistics, and the encoder adapts
  to them during our continued pre-training.
* **Geometry handling.** Urban flow has obstacles; Tadpole's pre-training data
  does not. The input state is masked (``state * geometry``, obstacle cells
  zeroed) exactly like ``P3D``. With ``encode_geometry=True`` the geometry mask
  (in ``{0, 1}``) and, when ``sdf_features`` is on, the clamped-SDF / gradient
  channels (bounded in ``[-1, 1]``) are appended **raw** (no z-scoring -- they are
  already bounded, and a 0/1 mask has no meaningful mean/std to standardise) as
  **extra folded channels** through the same single-channel encoder. They are fed
  in *and reconstructed* on purpose: with reconstruction loss on the state alone,
  the encoder is free to drop geometry from the latent, so making it reconstruct
  the geometry block is the supervision that forces geometry *into* the latent --
  and those geometry-bearing latents are exactly what the DFT sub-network (plan
  03) attends over to see the obstacle field. On a **single-geometry** corpus this
  re-encodes a constant on every snapshot (wasted capacity); that cost is
  intended for the multi-geometry foundation-model regime where geometry varies,
  and ``encode_geometry=False`` (state channels only) is the A/B / single-geometry
  escape hatch.
* **Padding.** Each spatial dim is zero-padded up to a multiple of
  ``encoder_crop_size`` so the autoencoder tiles cleanly into
  ``encoder_crop_size``-sized crops, then the reconstruction is cropped back.
  (The upstream fold requires spatial dims divisible by the crop size; anything
  smaller becomes a single crop.)

Reconstruction target / loss space
-----------------------------------
The autoencoder operates in the wrapper's **working space**: state channels
z-scored, geometry/SDF channels raw. :meth:`forward` returns the state
reconstruction in **physical units** (the clean public contract, used by plan 03
and analysis notebooks); the trainer instead calls it with
``working_space=True`` to get the full-channel working-space reconstruction and
its target in one shot (so all normalisation logic stays inside the model).

The heavy vendored stack (``diffusers`` / ``timm``) is imported lazily inside
``__init__`` so ``import neural_surrogates`` stays light, mirroring ``p3d.py``.
"""

from __future__ import annotations

import torch
from neural_surrogates.architectures._tadpole_field_io import _TadpoleFieldIO
from neural_surrogates.sdf import n_sdf_feature_channels, normalize_sdf_mode
from torch import nn

_SIZES = ("S", "B", "L")


class TadpoleAE(_TadpoleFieldIO, nn.Module):
    """Tadpole (V)AE wrapper for snapshot pre-training.

    Parameters
    ----------
    n_state_channels:
        Number of state channels ``C`` (e.g. 3 for ``u, v, w``); Hydra-injected.
    n_params:
        Accepted for signature parity with the other architectures (the pre-train
        script derives it from the dataset) but **unused** -- physical parameters
        condition dynamics, not single-snapshot appearance, so they are not an AE
        input (they enter in plan 03). Kept so ``set_normalization`` can be handed
        param stats and simply ignore them.
    size:
        Upstream autoencoder size ``"S"`` / ``"B"`` / ``"L"`` (8.8M / 38.1M /
        152.1M params; latent compression 16 / 8 / 4).
    latent_type:
        ``"sample"`` (VAE-proper; sample the latent) or ``"mode"`` (use the
        latent mean -- the deterministic-AE ablation).
    encoder_crop_size:
        Spatial crop size the field is tiled into internally. Must be a positive
        multiple of 16 (the encoder's total downsampling; smaller values make the
        upstream decoder over-upsample). Choose one that divides the grid to
        avoid padding, or rely on the padding fallback.
    max_internal_batchsize:
        Cap on how many folded crops the autoencoder processes at once (chunks
        the internal batch to bound memory); ``None`` processes them together.
    pretrained:
        ``"none"`` (random init), ``"hf"`` (load the HF ``thuerey-group/Tadpole``
        encoder/decoder weights for this size), or a mapping ``{"encoder": path,
        "decoder": path}`` of local state-dict / safetensors files.
    encode_geometry:
        Append the geometry mask (and SDF channels, if any) as extra folded
        encoder channels (see the module docstring).
    sdf_features:
        Which signed-distance-field channels to append alongside the geometry
        mask when ``encode_geometry=True``: ``"none"`` / ``"sdf"`` (+1) /
        ``"grad"`` (+3) / ``"both"`` (+4) (``True``/``False`` alias
        ``"both"``/``"none"``). Requires ``encode_geometry=True``.
    sdf_clamp_cells:
        Clamp radius ``L`` (cells) for the normalised SDF channel; must match the
        dataset's value (the pre-train script cross-checks it).
    normalize:
        Standardise state channels with buffered statistics (install via
        :meth:`set_normalization`).
    """

    def __init__(
        self,
        n_state_channels: int,
        n_params: int = 0,
        size: str = "S",
        latent_type: str = "sample",
        encoder_crop_size: int = 64,
        max_internal_batchsize: int | None = None,
        pretrained: str | dict = "none",
        encode_geometry: bool = True,
        sdf_features: bool | str = "none",
        sdf_clamp_cells: float = 32.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        # Lazy import: keep `import neural_surrogates` free of the heavy
        # diffusers/timm stack pulled in by the vendored autoencoder. A clean
        # ImportError here is what lets test suites importorskip the deps.
        try:
            from neural_surrogates.architectures._tadpole import TadpoleAutoencoder
        except ImportError as exc:  # pragma: no cover - exercised only when absent
            raise ImportError(
                "TadpoleAE requires the vendored autoencoder's runtime deps "
                "('diffusers', 'timm', 'einops'); install "
                "`neural_surrogates[tadpole]`."
            ) from exc

        if size not in _SIZES:
            raise ValueError(f"size must be one of {list(_SIZES)}, got {size!r}")
        if latent_type not in ("sample", "mode"):
            raise ValueError(
                f"latent_type must be 'sample' or 'mode', got {latent_type!r}"
            )
        if encoder_crop_size < 16 or encoder_crop_size % 16 != 0:
            # The encoder downsamples by 16 total; a smaller/indivisible crop
            # makes the upstream decoder over-upsample (output != input shape).
            raise ValueError(
                "encoder_crop_size must be a positive multiple of 16, got "
                f"{encoder_crop_size}"
            )

        self.n_state_channels = int(n_state_channels)
        self.n_params = int(n_params)
        self.size = size
        self.latent_type = latent_type
        self.encoder_crop_size = int(encoder_crop_size)
        self.normalize = normalize
        self.encode_geometry = bool(encode_geometry)

        self.sdf_feature_mode = normalize_sdf_mode(sdf_features)
        self.sdf_features_enabled = self.sdf_feature_mode != "none"
        self.sdf_clamp_cells = float(sdf_clamp_cells)
        self.n_geom_feature_channels = n_sdf_feature_channels(self.sdf_feature_mode)
        if self.sdf_features_enabled and not self.encode_geometry:
            raise ValueError(
                "sdf_features requires encode_geometry=True (the SDF channels are "
                "appended alongside the encoded geometry mask)."
            )

        # Extra folded channels the encoder also reconstructs when encoding the
        # geometry: the mask (+1) and any SDF channels. Zero when encode_geometry
        # is off (state channels only).
        self.n_geometry_channels = (
            1 + self.n_geom_feature_channels if self.encode_geometry else 0
        )

        weight_encoder, weight_decoder = self._resolve_pretrained(pretrained, size)

        # Freeze nothing here -- the AE trainer trains the whole autoencoder from
        # (optionally pretrained) init; LoRA-on-frozen is plan 03's DFT stage.
        self.ae = TadpoleAutoencoder(
            size=size,
            weight_encoder=weight_encoder,
            weight_decoder=weight_decoder,
            encoder_ft_state="FPFT",
            decoder_ft_state="FPFT",
            latent_type=latent_type,
            encoder_crop_size=self.encoder_crop_size,
            max_internal_batchsize=max_internal_batchsize,
        )

        # Standardisation statistics (identity until set_normalization is called);
        # buffers so they travel with the checkpoint. Only state channels are
        # normalised -- geometry/SDF channels are raw and bounded by construction.
        if normalize:
            self.register_buffer("state_mean", torch.zeros(n_state_channels))
            self.register_buffer("state_std", torch.ones(n_state_channels))

    # -- pretrained-weight resolution -------------------------------------- #

    @staticmethod
    def _resolve_pretrained(pretrained: str | dict, size: str):
        """Map the ``pretrained`` knob to ``(weight_encoder, weight_decoder)``.

        ``"none"`` -> ``(None, None)`` (random init). A mapping ``{"encoder",
        "decoder"}`` -> those paths (forwarded to the autoencoder's own
        ``load_weights``). ``"hf"`` -> download the size's encoder/decoder state
        dicts from the HF ``thuerey-group/Tadpole`` repo (lazy ``huggingface_hub``
        import so the base install and the ``"none"`` path stay dependency-light).
        """
        if pretrained is None or pretrained == "none":
            return None, None
        if isinstance(pretrained, dict):
            return pretrained.get("encoder"), pretrained.get("decoder")
        if pretrained == "hf":
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "pretrained='hf' needs 'huggingface_hub'; install it or pass "
                    "explicit {encoder, decoder} paths / use 'none'."
                ) from exc
            repo = "thuerey-group/Tadpole"
            enc = hf_hub_download(repo, f"tadpole_{size}_encoder.safetensors")
            dec = hf_hub_download(repo, f"tadpole_{size}_decoder.safetensors")
            return enc, dec
        raise ValueError(
            "pretrained must be 'none', 'hf', or a {encoder, decoder} mapping; "
            f"got {pretrained!r}"
        )

    # -- normalisation ----------------------------------------------------- #

    @torch.no_grad()
    def set_normalization(
        self,
        state_mean,
        state_std,
        param_mean=None,
        param_std=None,
        eps: float = 1e-6,
    ) -> None:
        """Install per-channel state standardisation statistics.

        ``param_mean`` / ``param_std`` are accepted (so the shared
        ``get_normalization_stats`` -> ``set_normalization`` call site works
        unchanged) but ignored -- params are not an AE input.
        """
        if not self.normalize:
            print("TadpoleAE(normalize=False): ignoring normalization stats")
            return

        self.state_mean.copy_(self._to_buffer(self.state_mean, state_mean))
        self.state_std.copy_(self._to_buffer(self.state_std, state_std, eps=eps))

    # The mask/normalise/assemble/pad helpers (``_pad_to_crop_multiple``,
    # ``_geometry_channels``, ``_sdf_features``, ``_normalize_state``,
    # ``_denormalize_state``, ``_fold_dims``, ``_assemble_working_input``,
    # ``_batched_mask``) are inherited unchanged from ``_TadpoleFieldIO`` and
    # shared with ``TadpoleTimeStepper``.

    # -- encode / decode passthroughs (plan 03 + analysis) ----------------- #

    def encode(
        self,
        state: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
        *,
        latent_type: str | None = None,
    ) -> torch.Tensor:
        """Latent for one working-space input (state z-scored + geometry block).

        ``latent_type`` overrides the module's ``latent_type`` for this call
        (e.g. ``"mode"`` for a deterministic latent in analysis)."""
        x = self._assemble_working_input(state, geometry, geom_features)
        x, _ = self._pad_to_crop_multiple(x)
        b, c, u, v, w = self._fold_dims(x)
        from einops import rearrange

        folded = rearrange(
            x, "B C (U Xc) (V Yc) (W Zc) -> (B C U V W) 1 Xc Yc Zc", U=u, V=v, W=w
        )
        return self.ae.encoder(folded, latent_type or self.latent_type)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode folded latents back to folded single-channel crops (the inverse
        of :meth:`encode`'s fold is left to the caller / plan 03)."""
        return self.ae.decoder(latent)

    # -- forward ----------------------------------------------------------- #

    def forward(
        self,
        state: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
        *,
        return_kl_element: bool = False,
        working_space: bool = False,
    ) -> torch.Tensor | tuple:
        """Reconstruct ``state``.

        Default (``working_space=False``): return the **physical-units** state
        reconstruction ``(B, C, *grid)`` (obstacle cells zeroed). With
        ``working_space=True`` return the tuple ``(recon, target)`` of the
        **full-channel working-space** reconstruction and its input target
        ``(B, Cin, *grid)`` -- what :class:`AutoencoderTrainer` needs to compute
        the split (state / geometry) reconstruction loss without recomputing the
        assembled input. ``return_kl_element=True`` appends the per-crop KL
        element for the VAE loss.
        """
        x = self._assemble_working_input(state, geometry, geom_features)
        x_pad, orig = self._pad_to_crop_multiple(x)
        recon_pad, kl_elem = self.ae(x_pad, return_kl_element=True)
        d, h, w = orig
        recon = recon_pad[..., :d, :h, :w]

        if working_space:
            out: tuple = (recon, x)
        else:
            state_recon = self._denormalize_state(recon[:, : self.n_state_channels])
            mask = self._batched_mask(geometry, state)
            out = (state_recon * mask,)
        if return_kl_element:
            out = out + (kl_elem,)
        return out[0] if len(out) == 1 else out

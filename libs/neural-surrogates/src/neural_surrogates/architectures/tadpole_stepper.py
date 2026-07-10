"""Tadpole AE -> time-stepper wrapper (plan 03).

``TadpoleTimeStepper`` turns a **pre-trained** Tadpole autoencoder (plan 02,
:class:`~neural_surrogates.architectures.tadpole_ae.TadpoleAE`) into an ESMDA
forward model that predicts the next state ``u_{t+dt}`` from the current state,
its physical parameters, and the (fixed) obstacle geometry.

It wraps the vendored :class:`~neural_surrogates.architectures._tadpole.model.dft.TadpoleDFT`
("Dynamic Fine-Tuning" head): a frozen encoder/decoder with **zero-initialised
skip connections** (``gamma`` scales) and a **zero-initialised latent subnetwork**
that together act as a trainable increment *around* the frozen AE reconstruction.
Fine-tuning (Phase 2A) trains the subnetwork + skip scales + a small amount of
LoRA on the encoder/decoder; the frozen AE is loaded via the DFT's own
``weight_encoder``/``weight_decoder`` load-before-skip-wrap path.

The state/geometry pre- and post-processing (masking, per-channel z-scoring,
geometry+SDF channel assembly, crop-multiple padding) is shared with
``TadpoleAE`` through :class:`~neural_surrogates.architectures._tadpole_field_io._TadpoleFieldIO`.

Identity-at-init
----------------
At construction the subnetwork output is exactly ``0`` (``init_zero_proj``), the
skip ``gamma`` scales are ``0`` and ``latent_residual_scale == 1.0``, so the DFT
output is **identical to the plain autoencoder reconstruction** of the same
working-space input. :meth:`_ae_reference_recon` exposes that pure-AE
reconstruction cheaply (it runs the same encode/decode with the subnetwork
bypassed and the skip residuals zeroed), and Phase 2B asserts
``stepper(state) == stepper._ae_reference_recon(state, geometry)`` exactly at
init -- the single most informative test of the DFT wiring. Note this is *not*
``state_next == state`` (that only holds for a perfectly-reconstructing AE, a
training outcome, not a wiring invariant).

Residual convention
--------------------
Output is ``state_next = dft_state * mask`` -- the DFT directly predicts the next
state (it morphs its own reconstruction toward ``u_{t+dt}``); there is no
separate ``state + increment`` term. ``predict_residual`` is kept for API parity
with the other architectures but is a no-op framing here
(``state + (dft_state - state) == dft_state``).
"""

from __future__ import annotations

import os
import warnings

import torch
from neural_surrogates.architectures._tadpole_field_io import _TadpoleFieldIO
from neural_surrogates.sdf import n_sdf_feature_channels, normalize_sdf_mode
from torch import nn

_SIZES = ("S", "B", "L")

# Subnetwork geometry per model size. ``latent_mult`` is the encoder's latent
# channel count Cl (= hidden_size * 2**(len(depth)//2): 256 / 512 / 1024 for
# S / B / L), so ``in_dim = input_channels * latent_mult`` matches the channel
# count the DFT reshapes the folded latent to. The n_layers / num_heads /
# hidden_size triples are exactly upstream ``default_subnetwork``'s, but built
# with ``attention_method="naive"`` (no Triton).
_SUBNET_SIZES: dict[str, dict[str, int]] = {
    "S": dict(latent_mult=256, n_layers=4, num_heads=8, hidden_size=144),
    "B": dict(latent_mult=512, n_layers=6, num_heads=8, hidden_size=176),
    "L": dict(latent_mult=1024, n_layers=8, num_heads=8, hidden_size=224),
}


class ParamConditionedSubnetwork(nn.Module):
    """Latent subnetwork for :class:`TadpoleTimeStepper`, conditioned on params.

    Wraps a vendored ``SequentialModel`` (transformer over the folded latent
    tokens, ``attention_method="naive"``, ``init_zero_proj=True`` so its output
    is exactly ``0`` at init) and applies parameter conditioning to the tokens
    fed to it:

    * ``param_conditioning="film"`` (default): a small MLP maps the (already
      z-scored) params ``(B, P)`` to per-channel ``(scale, shift)`` with the
      **output layer zero-initialised**, applied as ``x * (1 + scale) + shift``.
      At init ``scale == shift == 0`` so the tokens are unchanged.
    * ``param_conditioning="token"``: an additive per-channel parameter
      embedding (adaLN-style shift), output layer zero-initialised, added to the
      tokens (``x + shift``). (This is an additive-embedding realisation of the
      param-token idea; it is zero at init.)
    * ``param_conditioning="none"`` or ``P == 0``: **no** conditioning module is
      built at all (the repo no-op rule) -- ``forward`` is just the
      ``SequentialModel``, and the module tree is identical to a param-free
      build.

    Because ``SequentialModel``'s output projection is zero-initialised, the
    subnetwork's *total* output is exactly ``0`` at init regardless of the
    conditioning branch -- the DFT already carries the identity via
    ``latent_residual_scale * latent``, so this module returns only the
    (zero-at-init) increment.
    """

    def __init__(
        self,
        in_dim: int,
        n_params: int = 0,
        n_layers: int = 4,
        num_heads: int = 8,
        hidden_size: int = 144,
        param_conditioning: str = "film",
        mlp_ratio: int = 4,
        film_hidden: int = 128,
        use_checkpoint: bool = False,
        in_context_patches: int = -1,
    ) -> None:
        super().__init__()
        try:
            from neural_surrogates.architectures._tadpole.architecture.downstream import (
                SequentialModel,
            )
        except ImportError as exc:  # pragma: no cover - exercised only when absent
            raise ImportError(
                "ParamConditionedSubnetwork requires the vendored Tadpole runtime "
                "deps ('diffusers', 'timm', 'einops'); install "
                "`neural_surrogates[tadpole]`."
            ) from exc

        if param_conditioning not in ("film", "token", "none"):
            raise ValueError(
                "param_conditioning must be 'film', 'token' or 'none', got "
                f"{param_conditioning!r}"
            )

        self.in_dim = int(in_dim)
        self.n_params = int(n_params)
        # No param machinery when conditioning is off or there are no params.
        self.param_conditioning = param_conditioning if self.n_params > 0 else "none"

        self.seqmodel = SequentialModel(
            in_dim=in_dim,
            n_layers=n_layers,
            attention_method="naive",
            num_heads=num_heads,
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            init_zero_proj=True,
            use_checkpoint=use_checkpoint,
            in_context_patches=in_context_patches,
        )

        self.param_mlp: nn.Module | None = None
        if self.param_conditioning != "none":
            out_dim = 2 * in_dim if self.param_conditioning == "film" else in_dim
            self.param_mlp = nn.Sequential(
                nn.Linear(self.n_params, film_hidden),
                nn.SiLU(),
                nn.Linear(film_hidden, out_dim),
            )
            # Zero-init the output layer -> scale/shift are 0 at init, so
            # conditioning is identity and the subnetwork output stays 0.
            nn.init.zeros_(self.param_mlp[-1].weight)
            nn.init.zeros_(self.param_mlp[-1].bias)

    def forward(
        self, x: torch.Tensor, params: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (B, in_dim, X', Y', Z') folded latent tokens.
        if self.param_mlp is not None and params is not None:
            cond = self.param_mlp(params)  # (B, out_dim)
            view = (cond.shape[0], self.in_dim) + (1, 1, 1)
            if self.param_conditioning == "film":
                scale, shift = cond.chunk(2, dim=-1)
                x = x * (1.0 + scale.reshape(view)) + shift.reshape(view)
            else:  # "token": additive param embedding
                x = x + cond.reshape(view)
        return self.seqmodel(x)


class TadpoleTimeStepper(_TadpoleFieldIO, nn.Module):
    """Pre-trained Tadpole AE turned into a next-state predictor (plan 03).

    Parameters
    ----------
    n_state_channels:
        Number of state channels ``C`` (Hydra-injected from the dataset).
    n_params:
        Number of physical parameters ``P`` used to condition the dynamics.
    size:
        Encoder/decoder size ``"S"`` / ``"B"`` / ``"L"`` (must match the AE).
    pretrained_ae_dir:
        Directory of a pre-trained :class:`TadpoleAE` export
        (``encoder.pt`` / ``decoder.pt`` / ``weights.pt``). Loaded into the
        frozen DFT encoder/decoder unless ``skip_pretrained_load``.
    subnetwork:
        ``"default"`` builds a :class:`ParamConditionedSubnetwork`; ``None``
        builds no subnetwork (pure frozen AE, no time-stepping increment).
    param_conditioning:
        ``"film"`` (default) / ``"token"`` / ``"none"`` (see
        :class:`ParamConditionedSubnetwork`).
    latent_type:
        ``"mode"`` (deterministic latent, the default here) or ``"sample"``.
    encoder_crop_size:
        Spatial crop the field is tiled into (positive multiple of 16).
    max_internal_batchsize:
        Cap on folded crops processed at once (``None`` = together).
    predict_residual:
        Kept for API parity; a no-op framing (see module docstring).
    normalize:
        Standardise state channels + z-score params (buffers installed via
        :meth:`set_normalization`; state stats inherited from the AE).
    encode_geometry:
        Append the geometry mask (+ SDF channels) as extra folded channels.
    sdf_features / sdf_clamp_cells:
        SDF geometry-feature selection (see ``TadpoleAE``).
    skip_pretrained_load:
        Build encoder/decoder randomly and skip reading the AE dir -- used at
        ESMDA deploy time, where the merged ``weights.pt`` already carries every
        weight. The saved fine-tune config sets this so deployment never depends
        on the AE dir existing.
    subnetwork_cfg:
        Optional overrides for the subnetwork ``{n_layers, num_heads,
        hidden_size, ...}`` on top of the size defaults.
    require_ae_state_stats:
        When loading a pre-trained AE (``pretrained_ae_dir`` set,
        ``skip_pretrained_load=False``), fail loud if the AE export carries no
        ``state_mean``/``state_std`` -- running the frozen encoder with identity
        stats feeds it out-of-distribution inputs and wastes the whole run. The
        fine-tune script flips this to ``False`` only when
        ``recompute_normalization=true`` will install fresh stats over the
        inherited ones anyway.
    """

    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        size: str = "S",
        pretrained_ae_dir: str | None = None,
        subnetwork: str | None = "default",
        param_conditioning: str = "film",
        latent_type: str = "mode",
        encoder_crop_size: int = 64,
        max_internal_batchsize: int | None = None,
        predict_residual: bool = True,
        normalize: bool = True,
        encode_geometry: bool = True,
        sdf_features: bool | str = "none",
        sdf_clamp_cells: float = 32.0,
        skip_pretrained_load: bool = False,
        subnetwork_cfg: dict | None = None,
        require_ae_state_stats: bool = True,
    ) -> None:
        super().__init__()

        try:
            from neural_surrogates.architectures._tadpole.model.dft import TadpoleDFT
        except ImportError as exc:  # pragma: no cover - exercised only when absent
            raise ImportError(
                "TadpoleTimeStepper requires the vendored Tadpole runtime deps "
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
            raise ValueError(
                "encoder_crop_size must be a positive multiple of 16, got "
                f"{encoder_crop_size}"
            )
        if param_conditioning not in ("film", "token", "none"):
            raise ValueError(
                "param_conditioning must be 'film', 'token' or 'none', got "
                f"{param_conditioning!r}"
            )
        if subnetwork not in ("default", None):
            raise ValueError(
                f"subnetwork must be 'default' or None, got {subnetwork!r}"
            )
        # The residual is *intrinsic* to the DFT (state_next = dft_state; the
        # zero-init subnetwork/gamma are the increment around the frozen AE
        # reconstruction -- see the class docstring). The knob exists only for
        # signature parity with the other architectures; predict_residual=False
        # has no meaningful behaviour here, so reject it rather than silently
        # ignore it.
        if not predict_residual:
            raise ValueError(
                "TadpoleTimeStepper only supports predict_residual=True: the "
                "residual is intrinsic to the DFT (the next state is the AE "
                "reconstruction morphed by the zero-init subnetwork/gamma), so "
                "there is no separate non-residual mode to select."
            )

        self.n_state_channels = int(n_state_channels)
        self.n_params = int(n_params)
        self.size = size
        self.latent_type = latent_type
        self.encoder_crop_size = int(encoder_crop_size)
        self.max_internal_batchsize = max_internal_batchsize
        self.normalize = bool(normalize)
        self.predict_residual = bool(predict_residual)
        self.encode_geometry = bool(encode_geometry)
        # ``param_conditioning`` collapses to "none" when there are no params
        # (the repo no-op rule) so the module tree matches a param-free build.
        self.param_conditioning = param_conditioning if self.n_params > 0 else "none"

        self.sdf_feature_mode = normalize_sdf_mode(sdf_features)
        self.sdf_features_enabled = self.sdf_feature_mode != "none"
        self.sdf_clamp_cells = float(sdf_clamp_cells)
        self.n_geom_feature_channels = n_sdf_feature_channels(self.sdf_feature_mode)
        if self.sdf_features_enabled and not self.encode_geometry:
            raise ValueError(
                "sdf_features requires encode_geometry=True (the SDF channels are "
                "appended alongside the encoded geometry mask)."
            )
        self.n_geometry_channels = (
            1 + self.n_geom_feature_channels if self.encode_geometry else 0
        )
        # Folded channel count the DFT reshapes the latent for.
        input_channels = self.n_state_channels + self.n_geometry_channels

        # Build the param-conditioned latent subnetwork (or none).
        sub_module: nn.Module | None = None
        if subnetwork == "default":
            table = _SUBNET_SIZES[size]
            cfg = dict(
                n_layers=table["n_layers"],
                num_heads=table["num_heads"],
                hidden_size=table["hidden_size"],
            )
            if subnetwork_cfg:
                overrides = dict(subnetwork_cfg)
                if "hidden" in overrides:  # accept "hidden" as a hidden_size alias
                    overrides["hidden_size"] = overrides.pop("hidden")
                cfg.update(overrides)
            sub_module = ParamConditionedSubnetwork(
                in_dim=input_channels * table["latent_mult"],
                n_params=self.n_params,
                param_conditioning=self.param_conditioning,
                **cfg,
            )

        # Resolve pretrained encoder/decoder weights (load-before-skip-wrap via
        # the DFT's own weight_encoder/weight_decoder kwargs).
        weight_encoder = weight_decoder = None
        if pretrained_ae_dir is not None and not skip_pretrained_load:
            weight_encoder = os.path.join(pretrained_ae_dir, "encoder.pt")
            weight_decoder = os.path.join(pretrained_ae_dir, "decoder.pt")

        self.dft = TadpoleDFT(
            size=size,
            input_channels=input_channels,
            subnetwork=sub_module,
            weight_encoder=weight_encoder,
            weight_decoder=weight_decoder,
            encoder_ft_state="frozen",
            decoder_ft_state="frozen",
            latent_type=latent_type,
            encoder_crop_size=self.encoder_crop_size,
            max_internal_batchsize=max_internal_batchsize,
        )

        # Standardisation buffers (identity until set_normalization). param
        # buffers use max(n_params, 1) so the zero-param case stays valid.
        if normalize:
            self.register_buffer("state_mean", torch.zeros(n_state_channels))
            self.register_buffer("state_std", torch.ones(n_state_channels))
            self.register_buffer("param_mean", torch.zeros(max(self.n_params, 1)))
            self.register_buffer("param_std", torch.ones(max(self.n_params, 1)))
            # Inherit the frozen AE's state statistics so the encoder sees the
            # distribution it was pre-trained on.
            if pretrained_ae_dir is not None and not skip_pretrained_load:
                self._load_ae_state_stats(
                    pretrained_ae_dir, require=require_ae_state_stats
                )

    # -- pretrained state-stat inheritance --------------------------------- #

    @torch.no_grad()
    def _load_ae_state_stats(
        self, pretrained_ae_dir: str, require: bool = True
    ) -> None:
        """Copy ``state_mean``/``state_std`` from the AE export's ``weights.pt``.

        The frozen encoder was pre-trained on the AE's standardised distribution,
        so missing/absent stats would leave identity normalisation (zeros/ones)
        and feed the encoder out-of-distribution inputs -> a full wasted training
        run on garbage. Repo convention is fail-loud: raise unless ``require`` is
        ``False``, which the fine-tune script sets only when
        ``recompute_normalization=true`` will install fresh stats over these
        anyway (in that case warn and continue -- the inherited stats are dead).
        """
        path = os.path.join(pretrained_ae_dir, "weights.pt")
        if not os.path.exists(path):
            msg = (
                f"TadpoleTimeStepper: no weights.pt at {path!r}, so the frozen "
                "encoder's state normalisation stats are unavailable. The encoder "
                "was pre-trained on the AE's standardised distribution; running "
                "with identity stats (zeros/ones) feeds it out-of-distribution "
                "inputs and wastes the whole run. Point pretrained_ae_dir at a "
                "real AE export, or set recompute_normalization=true to compute "
                "fresh stats on the fine-tune split."
            )
            if require:
                raise FileNotFoundError(msg)
            warnings.warn(msg, stacklevel=2)
            return
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_mean" in sd and "state_std" in sd:
            self.state_mean.copy_(sd["state_mean"].reshape(-1).to(self.state_mean))
            self.state_std.copy_(sd["state_std"].reshape(-1).to(self.state_std))
        else:
            msg = (
                f"TadpoleTimeStepper: {path!r} has no state_mean/state_std "
                "buffers, so the frozen encoder's state normalisation stats are "
                "unavailable (it was pre-trained on the AE's standardised "
                "distribution; identity stats feed it out-of-distribution "
                "inputs). Re-export the AE with normalization buffers, or set "
                "recompute_normalization=true to compute fresh stats on the "
                "fine-tune split."
            )
            if require:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

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
        """Install per-channel state and per-param standardisation statistics.

        State stats are normally the AE's (passed through unchanged by the
        fine-tune script); ``param_mean``/``param_std`` are the fine-tune
        dataset's param statistics used to z-score params before conditioning.
        """
        if not self.normalize:
            print("TadpoleTimeStepper(normalize=False): ignoring normalization stats")
            return
        if state_mean is not None:
            self.state_mean.copy_(self._to_buffer(self.state_mean, state_mean))
        if state_std is not None:
            self.state_std.copy_(self._to_buffer(self.state_std, state_std, eps))
        if self.n_params > 0:
            if param_mean is not None:
                self.param_mean.copy_(self._to_buffer(self.param_mean, param_mean))
            if param_std is not None:
                self.param_std.copy_(self._to_buffer(self.param_std, param_std, eps))

    def _zscore_params(self, params: torch.Tensor | None) -> torch.Tensor | None:
        """Z-score params for conditioning, or ``None`` when unconditioned."""
        if self.param_conditioning == "none" or self.n_params == 0 or params is None:
            return None
        if self.normalize:
            params = (params - self.param_mean) / self.param_std
        return params

    # -- encode / decode passthroughs (analysis) --------------------------- #

    def encode(
        self,
        state: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
        *,
        latent_type: str | None = None,
    ):
        """Folded latent + skip residuals for one working-space input."""
        from einops import rearrange

        x = self._assemble_working_input(state, geometry, geom_features)
        x, _ = self._pad_to_crop_multiple(x)
        _, _, u, v, w = self._fold_dims(x)
        folded = rearrange(
            x, "B C (U Xc) (V Yc) (W Zc) -> (B C U V W) 1 Xc Yc Zc", U=u, V=v, W=w
        )
        return self.dft.encoder(folded, latent_type=latent_type or self.latent_type)

    def decode(self, latent: torch.Tensor, residuals) -> torch.Tensor:
        """Decode folded latents (+ skip residuals) to folded crops."""
        return self.dft.decoder(latent, residuals)

    # -- forward ----------------------------------------------------------- #

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor | None,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next state ``(B, C, *grid)`` (obstacle cells zeroed)."""
        # A conditioned model must be given params -- otherwise FiLM/token
        # conditioning silently degrades to unconditioned (the no-op rule only
        # applies when there is genuinely nothing to condition on).
        if self.param_conditioning != "none" and params is None:
            raise ValueError(
                f"param_conditioning={self.param_conditioning!r} requires params, "
                "but got params=None; pass the parameter vector (B, P)."
            )
        x = self._assemble_working_input(state, geometry, geom_features)
        x_pad, orig = self._pad_to_crop_multiple(x)
        p = self._zscore_params(params)
        recon_pad = self.dft(x_pad, params=p)
        d, h, w = orig
        recon = recon_pad[..., :d, :h, :w]
        dft_state = self._denormalize_state(recon[:, : self.n_state_channels])
        mask = self._batched_mask(geometry, state)
        # DFT directly predicts the next state; residual framing is intrinsic.
        return dft_state * mask

    # -- identity-at-init reference ---------------------------------------- #

    @torch.no_grad()
    def _ae_reference_recon(
        self,
        state: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pure-AE reconstruction of the working input: the same encode/decode
        path with the **subnetwork bypassed and skip residuals zeroed**.

        At init (zero subnetwork, ``gamma == 0``, ``latent_residual_scale == 1``)
        this is bit-identical to :meth:`forward`; Phase 2B locks that equality
        down as the DFT-wiring invariant.
        """
        from einops import rearrange

        x = self._assemble_working_input(state, geometry, geom_features)
        x_pad, orig = self._pad_to_crop_multiple(x)

        dft = self.dft
        c = x_pad.shape[1]
        cs = dft.encoder_crop_size
        fu = max(x_pad.shape[2] // cs, 1)
        fv = max(x_pad.shape[3] // cs, 1)
        fw = max(x_pad.shape[4] // cs, 1)
        folded = rearrange(
            x_pad,
            "B C (U Xc) (V Yc) (W Zc) -> (B C U V W) 1 Xc Yc Zc",
            U=fu,
            V=fv,
            W=fw,
        )
        latent, res = dft.encoder(folded, latent_type=dft.latent_type)
        # Zero the skip residuals so the decoder is the plain (skip-free) decoder
        # regardless of the trained gamma scales -> the pure-AE reconstruction.
        zres = [
            [torch.zeros_like(t) for t in res[0]],
            [torch.zeros_like(t) for t in res[1]],
        ]
        recon = dft.decoder(latent, zres)
        recon_pad = rearrange(
            recon,
            "(B C U V W) 1 Xc Yc Zc -> B C (U Xc) (V Yc) (W Zc)",
            C=c,
            U=fu,
            V=fv,
            W=fw,
        )
        d, h, w = orig
        recon_pad = recon_pad[..., :d, :h, :w]
        ref_state = self._denormalize_state(recon_pad[:, : self.n_state_channels])
        mask = self._batched_mask(geometry, state)
        return ref_state * mask

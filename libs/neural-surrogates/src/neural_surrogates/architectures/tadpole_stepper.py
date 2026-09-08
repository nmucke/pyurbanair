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

Geometry branch
---------------
With ``geometry_branch={...}`` the geometry is no longer folded through the
encoder as extra channels: a small (frozen, pre-trained) ``GeometryBranch``
turns the geometry block into a 4-level feature pyramid that is *added* into the
frozen encoder/decoder through their zero-init 1x1x1 projections, and into the
latent subnetwork through a zero-init spatial FiLM. Every injection is zero at
init, so the identity-at-init invariant below holds in branch mode too (with
:meth:`_ae_reference_recon` applying the same branch features, since the
projections belong to the frozen AE).

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
from typing import TYPE_CHECKING

import torch
from neural_surrogates.architectures._tadpole_field_io import _TadpoleFieldIO
from neural_surrogates.sdf import n_sdf_feature_channels, normalize_sdf_mode
from torch import nn

if TYPE_CHECKING:  # heavy/optional import: only needed for the annotations below
    from neural_surrogates.architectures.tadpole_geometry_branch import GeometryBranch

_SIZES = ("S", "B", "L")

# Total spatial stride of the Tadpole encoder (a 16^3 crop -> a 1^3 latent), i.e.
# the stride of the geometry branch's level-3 feature.
_LATENT_STRIDE = 16

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

    ``geom_cond_dim > 0`` (geometry-branch mode) additionally builds a
    **spatial** FiLM: a zero-init ``Conv3d(geom_cond_dim, 2 * in_dim, 1)`` maps
    the branch's stride-16 feature map (on the latent grid) to a per-token
    ``(scale, shift)`` applied as ``x * (1 + scale) + shift`` *after* the param
    FiLM. ``geom_cond_dim == 0`` (default) builds nothing (the no-op rule).

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
        geom_cond_dim: int = 0,
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

        # Spatial (per-token) FiLM from the geometry branch's latent-grid
        # feature. Built only in geometry-branch mode (no-op rule) and zero-init,
        # so it is the identity at construction.
        self.geom_cond_dim = int(geom_cond_dim)
        self.geom_film: nn.Conv3d | None = None
        if self.geom_cond_dim > 0:
            self.geom_film = nn.Conv3d(self.geom_cond_dim, 2 * in_dim, kernel_size=1)
            nn.init.zeros_(self.geom_film.weight)
            nn.init.zeros_(self.geom_film.bias)

    def forward(
        self,
        x: torch.Tensor,
        params: torch.Tensor | None = None,
        geom_cond: torch.Tensor | None = None,
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
        if self.geom_film is not None and geom_cond is not None:
            # geom_cond: (B, geom_cond_dim, X', Y', Z') on the same latent grid.
            g = self.geom_film(geom_cond)
            scale, shift = g.chunk(2, dim=1)
            x = x * (1.0 + scale) + shift
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
        Mutually exclusive with ``geometry_branch``.
    sdf_features / sdf_clamp_cells:
        SDF geometry-feature selection (see ``TadpoleAE``). Allowed with either
        ``encode_geometry`` or ``geometry_branch`` (in branch mode the SDF
        channels feed the branch instead of the encoder).
    geometry_branch:
        ``None`` (default) = today's behaviour. A mapping (e.g. ``{"width": 32}``)
        switches on **geometry-branch conditioning**: a small
        :class:`~neural_surrogates.architectures.tadpole_geometry_branch.GeometryBranch`
        (built as ``GeometryBranch(in_channels=1 + n_sdf, **geometry_branch)``)
        maps the geometry block to a 4-level feature pyramid that is injected
        into the frozen encoder/decoder through their zero-init projections and
        into the latent subnetwork through a zero-init spatial FiLM. The branch
        is part of the frozen AE: it is loaded from
        ``<pretrained_ae_dir>/geometry_branch.pt`` (fail loud if missing, unless
        ``skip_pretrained_load``) and frozen. In branch mode geometry is no
        longer folded through the encoder (``encode_geometry`` must be ``False``,
        ``n_geometry_channels == 0``) and it is not reconstructed -- the branch
        *conditions* the AE rather than being predicted by it. Must match the
        pre-trained AE (cross-checked by the fine-tune script).
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
    num_history_steps:
        Accepted for signature parity with the next-step architectures (so a
        Hydra node carrying the key instantiates), but only ``1`` is supported:
        the frozen Tadpole AE encodes exactly ``C`` state channels (+ geometry)
        per crop, so a history-flattened input would need a different (and
        re-pretrained) encoder. ``> 1`` raises :class:`NotImplementedError`.
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
        num_history_steps: int = 1,
        geometry_branch: dict | None = None,
    ) -> None:
        super().__init__()

        if int(num_history_steps) != 1:
            raise NotImplementedError(
                "TadpoleTimeStepper does not support state history "
                f"(num_history_steps={num_history_steps}). The frozen AE "
                "encoder is pre-trained on exactly C state channels (+ the "
                "geometry block) per crop; use a plain next-step architecture "
                "(UNetConvNeXt / P3D / UPT / SimpleConv) for H > 1."
            )

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
        # History is not supported here (see above); the attributes exist so the
        # trainer / forward model can read them uniformly across architectures.
        self.num_history_steps = 1
        self.n_input_state_channels = self.n_state_channels
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

        # The geometry branch and the folded geometry channels are two mutually
        # exclusive ways of getting geometry into the (same) frozen AE: in branch
        # mode geometry conditions the encoder/decoder through their zero-init
        # projections instead of riding along as extra folded channels.
        # ``is not None`` (not truthiness) so an empty mapping means "branch with
        # default kwargs", exactly as ``TadpoleAE`` reads the same knob.
        self.geometry_branch_cfg = (
            dict(geometry_branch) if geometry_branch is not None else None
        )
        if self.geometry_branch_cfg is not None and self.encode_geometry:
            raise ValueError(
                "geometry_branch and encode_geometry are mutually exclusive: in "
                "branch mode the geometry is fed to the branch (and injected into "
                "the frozen encoder/decoder), not folded through the encoder as "
                "extra reconstructed channels. Set encode_geometry=false."
            )

        self.sdf_feature_mode = normalize_sdf_mode(sdf_features)
        self.sdf_features_enabled = self.sdf_feature_mode != "none"
        self.sdf_clamp_cells = float(sdf_clamp_cells)
        self.n_geom_feature_channels = n_sdf_feature_channels(self.sdf_feature_mode)
        if self.sdf_features_enabled and not (
            self.encode_geometry or self.geometry_branch_cfg is not None
        ):
            raise ValueError(
                "sdf_features requires encode_geometry=True or a geometry_branch "
                "(the SDF channels ride alongside the encoded geometry mask, or "
                "feed the geometry branch)."
            )
        # Zero in branch mode (encode_geometry is False there): the working input
        # is the state channels only, and the geometry is never reconstructed.
        self.n_geometry_channels = (
            1 + self.n_geom_feature_channels if self.encode_geometry else 0
        )
        # Folded channel count the DFT reshapes the latent for.
        input_channels = self.n_state_channels + self.n_geometry_channels

        # Build (and load + freeze) the geometry branch before the subnetwork /
        # DFT: both need its feature dims.
        branch = self._build_geometry_branch(pretrained_ae_dir, skip_pretrained_load)
        self.geometry_branch: GeometryBranch | None = branch
        geom_in_dims = None if branch is None else tuple(branch.out_dims)

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
                # Spatial FiLM from the branch's stride-16 (latent-grid) feature;
                # 0 => no module at all (no-op rule).
                geom_cond_dim=0 if geom_in_dims is None else int(geom_in_dims[3]),
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
            geom_in_dims=geom_in_dims,
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

    # -- geometry branch ---------------------------------------------------- #

    def _build_geometry_branch(
        self, pretrained_ae_dir: str | None, skip_pretrained_load: bool
    ) -> GeometryBranch | None:
        """Build, load and freeze the (pre-trained) geometry branch, or ``None``.

        The branch is part of the frozen autoencoder -- it was trained *with* the
        encoder/decoder projections it feeds -- so a mismatched (randomly
        initialised) branch would silently feed the frozen AE features it has
        never seen. Missing ``geometry_branch.pt`` therefore raises, exactly like
        the missing-state-stats case above; ``skip_pretrained_load`` (the ESMDA
        deploy build, where the merged ``weights.pt`` carries the branch as a
        submodule) is the only sanctioned way past it.
        """
        if self.geometry_branch_cfg is None:
            return None
        try:
            from neural_surrogates.architectures.tadpole_geometry_branch import (
                GeometryBranch,
            )
        except ImportError as exc:  # pragma: no cover - exercised only when absent
            raise ImportError(
                "geometry_branch requires neural_surrogates.architectures."
                "tadpole_geometry_branch.GeometryBranch."
            ) from exc

        branch = GeometryBranch(
            in_channels=1 + self.n_geom_feature_channels, **self.geometry_branch_cfg
        )
        if pretrained_ae_dir is not None and not skip_pretrained_load:
            path = os.path.join(pretrained_ae_dir, "geometry_branch.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"TadpoleTimeStepper: no geometry_branch.pt at {path!r}. The "
                    "geometry branch is part of the frozen AE (it was pre-trained "
                    "together with the encoder/decoder projections that consume "
                    "its features), so a random branch would feed the frozen AE "
                    "out-of-distribution conditioning. Point pretrained_ae_dir at "
                    "an AE exported with geometry_branch=... , or drop the "
                    "geometry_branch knob."
                )
            branch.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
        # Frozen like the rest of the AE (the fine-tune script's
        # `trainable_modules` never lists it, so it stays frozen end to end).
        for p in branch.parameters():
            p.requires_grad = False
        return branch

    def _geom_branch_kwargs(
        self,
        state: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None,
    ) -> dict:
        """``{}`` (no branch) or ``{"geom_feats": [...], "geom_cond": ...}``.

        The features are computed once per call on the padded grid: the folded
        pyramid for the encoder/decoder projections and the unfolded stride-16
        level for the subnetwork's spatial FiLM.
        """
        if self.geometry_branch is None:
            return {}
        feats = self._branch_features(geometry, geom_features, state)
        if feats is None:  # pragma: no cover - defensive (branch is not None here)
            return {}
        # Only the state channels are folded in branch mode
        # (n_geometry_channels == 0), so the fold expands over C == n_state_channels.
        return {
            "geom_feats": self._fold_geom_feats(feats, self.n_state_channels),
            "geom_cond": feats[3],
        }

    def _check_geom_cond_grid(
        self, geom_cond: torch.Tensor, x_pad: torch.Tensor
    ) -> None:
        """The FiLM feature must live on the DFT's *unfolded* latent grid.

        The DFT reshapes the folded latents back to ``(U Xl) (V Yl) (W Zl)``,
        which is the padded grid divided by the encoder's total stride (16), and
        that is exactly the branch's level-3 stride -- a mismatch here means the
        branch pyramid and the encoder disagree and would broadcast silently."""
        expected = tuple(int(s) // _LATENT_STRIDE for s in x_pad.shape[2:])
        got = tuple(int(s) for s in geom_cond.shape[2:])
        if got != expected:
            raise ValueError(
                f"geometry branch level-3 feature has spatial shape {got} but the "
                f"DFT's latent grid is {expected} (padded grid "
                f"{tuple(int(s) for s in x_pad.shape[2:])} / {_LATENT_STRIDE}); the "
                "branch's stride-16 level must match the encoder's total stride."
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
        # The encoder/decoder only take the folded pyramid; ``geom_cond`` is
        # the subnetwork's business (and the subnetwork is bypassed here).
        branch = self._geom_branch_kwargs(state, geometry, geom_features)
        enc_kwargs = {k: v for k, v in branch.items() if k == "geom_feats"}
        return self.dft.encoder(
            folded, latent_type=latent_type or self.latent_type, **enc_kwargs
        )

    def decode(
        self,
        latent: torch.Tensor,
        residuals: list,
        geom_feats: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Decode folded latents (+ skip residuals) to folded crops.

        ``geom_feats`` is the folded geometry-branch pyramid (what
        :meth:`_geom_branch_kwargs` returns); pass it in branch mode so the
        decoder's projections see the same conditioning :meth:`encode` used."""
        dec_kwargs = {} if geom_feats is None else {"geom_feats": geom_feats}
        return self.dft.decoder(latent, residuals, **dec_kwargs)

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
        branch = self._geom_branch_kwargs(state, geometry, geom_features)
        if branch:
            self._check_geom_cond_grid(branch["geom_cond"], x_pad)
        recon_pad = self.dft(x_pad, params=p, **branch)
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
        # The geometry-branch projections live INSIDE the frozen encoder/decoder,
        # so they are part of the AE reference reconstruction (only the DFT's own
        # additions -- the subnetwork and the gamma skips -- are bypassed here).
        branch = self._geom_branch_kwargs(state, geometry, geom_features)
        enc_kwargs = {k: v for k, v in branch.items() if k == "geom_feats"}
        latent, res = dft.encoder(folded, latent_type=dft.latent_type, **enc_kwargs)
        # Zero the skip residuals so the decoder is the plain (skip-free) decoder
        # regardless of the trained gamma scales -> the pure-AE reconstruction.
        zres = [
            [torch.zeros_like(t) for t in res[0]],
            [torch.zeros_like(t) for t in res[1]],
        ]
        recon = dft.decoder(latent, zres, **enc_kwargs)
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

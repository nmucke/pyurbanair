"""P3D neural surrogate: a wrapper around the ``p3d_surrogate`` package.

P3D (Holzschuh et al., ICLR 2026, ``tum-pbs/P3D``) is a hierarchical U-shaped
3D model combining convolutional blocks with windowed self-attention. This
wraps the upstream ``P3D_S`` / ``P3D_B`` / ``P3D_L`` factories behind the same
``forward(state, params, geometry) -> next_state`` contract used by the other
architectures (``upt.py``, ``unet_convnext.py``, ``simple_conv.py``).

Mapping the upstream model onto this contract
---------------------------------------------
* **Geometry** is concatenated to the state as one extra input channel (so the
  underlying model's ``channel_size = n_state_channels + 1 [+ n_params]``), and
  the final output is multiplied pointwise by the geometry mask so obstacle
  cells stay exactly zero -- identical to ``UPT`` / ``UNetConvNeXt``. With
  a non-``"none"`` ``sdf_features`` mode, further geometry channels (the clamped
  signed distance field ``sdf_n`` and/or its unit gradient ``g_z, g_y, g_x``) are
  inserted right after the mask -- ``"both"`` gives the full block
  ``[geometry, sdf_n, g_z, g_y, g_x]`` (+4), ``"sdf"`` only ``sdf_n`` (+1),
  ``"grad"`` only the gradient (+3) -- and ``channel_size`` grows by that many.
  During training these arrive precomputed via the
  ``geom_features`` argument (never an EDT inside the training step); at
  inference the model computes them itself, once per rollout, from a small
  identity-keyed cache (see :meth:`forward`).
* **Parameters.** The upstream model *does* expose native conditioning, but only
  through a single scalar per sample (``pde_parameters`` of shape ``(B,)``,
  sinusoidally embedded and applied as adaLN modulation in every block). It has
  no native length-``P`` parameter-vector input. By default
  (``param_conditioning="native"``) the params (reduced to one scalar via a
  learned linear when ``P > 1``) are routed through this upstream
  ``pde_parameters`` adaLN path. ``param_conditioning="channels"`` instead
  follows the pattern of the other architectures here: z-score the inflow params
  and broadcast each as a constant input channel (higher-capacity for
  multi-param conditioning, at the cost of ``P`` extra stem channels).
* **Spatial size.** The upstream window attention self-pads internally, but the
  U-structure downsamples 4x in the wrapper encoder and 4x in the backbone, so
  the input ``(D, H, W)`` must be divisible by 16. Non-periodic axes are
  zero-padded up to a multiple of 16 and cropped back; **periodic** axes cannot
  be zero-padded without breaking the circular convolutions (same restriction as
  ``UNetConvNeXt``), so a periodic axis must already be divisible by 16.

With ``normalize=True`` the model standardises state and params with buffered
training-split statistics (installed via :meth:`set_normalization`, saved with
the weights); with ``predict_residual=True`` the head predicts the state
increment. Both mirror ``UNetConvNeXt`` so a P3D model is a true drop-in.

The heavy ``p3d_surrogate`` dependency (it pulls in ``transformers`` /
``diffusers``) is imported lazily inside ``__init__`` so importing
``neural_surrogates`` does not require it; install it with
``pip install p3d-surrogate``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from neural_surrogates.sdf import n_sdf_feature_channels, normalize_sdf_mode
from neural_surrogates.sdf import sdf_features as compute_sdf_features
from torch import nn

# Total spatial downsampling factor of the upstream U-structure for the default
# S/B/L presets: wrapper encoder (num_downsampling_layers=2 -> /4) times backbone
# (len(depth)//2 = 2 -> /4). Inputs must be divisible by this along every axis.
_DOWNSAMPLE_FACTOR = 16

_FACTORIES = {"S": "P3D_S", "B": "P3D_B", "L": "P3D_L"}


class P3D(nn.Module):
    """P3D surrogate (wraps ``p3d_surrogate``'s ``P3D_S`` / ``P3D_B`` / ``P3D_L``).

    Parameters
    ----------
    n_state_channels:
        Number of state channels ``C`` (e.g. 3 for ``u, v, w``); Hydra-injected.
    n_params:
        Number of inflow parameters ``P``; Hydra-injected.
    size:
        Which upstream size factory to use: ``"S"`` (hidden 64), ``"B"`` (128)
        or ``"L"`` (256).
    window_size:
        Self-attention window size. Only ``P3D_S`` exposes this upstream; it is
        ignored (with a fixed value of 4) for ``"B"`` / ``"L"``.
    partition_size:
        Upstream region-partition count (the global-context "messenger" split).
        ``1`` (default) means a single global region and imposes no extra
        divisibility constraint; any larger value must divide the padded grid.
    shift, mending, skip_connections_active:
        Forwarded to the upstream factory unchanged.
    periodic_axes:
        Spatial axes (subset of ``"z", "y", "x"``; tensor layout ``(B, C, z, y,
        x)``) with periodic boundaries. The upstream convolutions wrap
        circularly along these axes. A periodic axis must already be divisible by
        16 (it cannot be zero-padded without corrupting the wrap).
    param_conditioning:
        ``"native"`` (default): params routed through the upstream scalar
        ``pde_parameters`` adaLN conditioning (reduced to one scalar via a
        learned linear when ``P > 1``). ``"channels"``: z-scored params
        broadcast as constant input channels.
    normalize:
        Standardise state and params with buffered training statistics (install
        via :meth:`set_normalization`); the output is mapped back to physical
        units.
    predict_residual:
        Predict the state increment rather than the next state; the identity
        rollout becomes the zero-output solution.
    sdf_features:
        Which signed-distance-field geometry features to add to the stem,
        inserted **right after** the geometry mask: ``"none"`` (off), ``"sdf"``
        (the clamped SDF only, +1 channel), ``"grad"`` (the 3 unit-gradient
        components only, +3), or ``"both"`` (SDF + ∇SDF, +4). ``True``/``False``
        are accepted as aliases for ``"both"``/``"none"``. ``in_channels`` grows
        by :attr:`n_geom_feature_channels`, which the trainer keys off to fail
        loud when the dataset ships no features. The default ``"none"`` keeps
        ``in_channels`` -- and the whole state dict -- byte-identical to today's
        models, so existing checkpoints load unchanged.
    sdf_clamp_cells:
        Clamp radius ``L`` (in cells) for the normalised SDF channel; forwarded
        to :func:`neural_surrogates.sdf.sdf_features`. Must match the value the
        training dataset used (the train script cross-checks this).
    extra_in_channels:
        Number of *extra* raw input channels appended to the stem input
        **after** the geometry mask (and any param channels). Passed to
        :meth:`forward` via the ``extra`` argument and concatenated raw -- no
        normalisation, no geometry masking -- so the domain-decomposition
        wrapper (:class:`~neural_surrogates.architectures.DomainDecomposed`) can
        use P3D as the per-patch fine net, feeding the prolonged coarse context
        and positional encodings straight in. The default ``0`` keeps the stem
        (and the whole state dict) byte-identical to a standard P3D, so a plain
        (non-DD) model and its existing checkpoints are unaffected.
    """

    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        size: str = "S",
        window_size: int = 4,
        partition_size: int = 1,
        shift: bool = False,
        mending: bool = False,
        skip_connections_active: bool = True,
        periodic_axes: Sequence[str] = (),
        param_conditioning: str = "native",
        normalize: bool = True,
        predict_residual: bool = True,
        sdf_features: bool | str = "none",
        sdf_clamp_cells: float = 32.0,
        extra_in_channels: int = 0,
    ) -> None:
        super().__init__()

        # Lazy import: keep `import neural_surrogates` free of the heavy
        # transformers/diffusers stack pulled in by p3d_surrogate. A clean
        # ImportError here is what lets test suites importorskip the dep.
        try:
            import p3d_surrogate
        except ImportError as exc:  # pragma: no cover - exercised only when absent
            raise ImportError(
                "The P3D architecture requires the 'p3d_surrogate' package; "
                "install it with `pip install p3d-surrogate`."
            ) from exc

        if size not in _FACTORIES:
            raise ValueError(f"size must be one of {sorted(_FACTORIES)}, got {size!r}")
        if param_conditioning not in ("channels", "native"):
            raise ValueError(
                "param_conditioning must be 'channels' or 'native', "
                f"got {param_conditioning!r}"
            )

        axes = tuple(periodic_axes or ())
        unknown = set(axes) - {"z", "y", "x"}
        if unknown:
            raise ValueError(
                f"periodic_axes entries must be in ('z', 'y', 'x'), got {sorted(unknown)}"
            )

        if extra_in_channels < 0:
            raise ValueError("extra_in_channels must be >= 0")

        # torch.compile hint honoured by neural_surrogates.Trainer: P3D's
        # internal window padding (P3DStage.maybe_pad) computes pad sizes from
        # `x.shape % window_size`, so dynamo introduces symbolic spatial dims
        # even on the first trace and inductor then mis-infers the backward
        # strides of the reshape/PixelShuffle ops, crashing the compiled
        # backward with `assert_size_stride`. Compiling with dynamic=False
        # specialises on the (fixed, already padded-to-16) grid and folds the
        # padding arithmetic away. The wrapper pads inputs to a multiple of 16,
        # so maybe_pad is a no-op at runtime regardless.
        self.compile_dynamic = False

        self.n_state_channels = n_state_channels
        self.n_params = n_params
        self.size = size
        self.param_conditioning = param_conditioning
        self.normalize = normalize
        self.predict_residual = predict_residual
        # SDF / ∇SDF geometry features (see neural_surrogates.sdf). ``sdf_features``
        # selects which channels: "none" (off), "sdf" (clamped SDF, 1), "grad"
        # (unit-gradient, 3) or "both" (4); ``True``/``False`` alias "both"/"none".
        # When on, the selected channels are inserted right after the geometry
        # mask; the trainer keys off ``n_geom_feature_channels`` to decide whether
        # it must ship them. The default ("none") leaves ``in_channels`` -- and the
        # whole state dict -- byte-identical to a standard P3D, so existing
        # checkpoints are unaffected (the repo's no-op-when-absent rule).
        self.sdf_feature_mode = normalize_sdf_mode(sdf_features)
        self.sdf_features_enabled = self.sdf_feature_mode != "none"
        self.sdf_clamp_cells = float(sdf_clamp_cells)
        self.n_geom_feature_channels = n_sdf_feature_channels(self.sdf_feature_mode)
        # Size-1 inference cache for the self-computed features: a
        # ``(geometry_tensor, features)`` pair. Holding a reference to the keyed
        # geometry tensor keeps its storage alive, so its address can't be
        # recycled by a later allocation and alias a stale hit (see
        # ``_sdf_features_for``). A plain attribute (NOT a buffer) so it never
        # enters the state dict.
        self._sdf_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        # ``extra_in_channels`` are raw input-only channels appended to the stem
        # input AFTER geometry (and any param channels): the domain-decomposition
        # wrapper feeds the prolonged coarse context + positional encodings in
        # this way. The default 0 keeps the upstream net's ``channel_size`` (and
        # hence the whole state dict) byte-identical to a model built without it,
        # so a standard (non-DD) P3D and its existing checkpoints are unaffected.
        self.extra_in_channels = int(extra_in_channels)
        self._periodic = tuple(a in axes for a in ("z", "y", "x"))
        self._downsample_factor = _DOWNSAMPLE_FACTOR

        # Standardisation statistics (identity until set_normalization is called).
        # Registered as buffers so they travel with the checkpoint. param buffers
        # use max(n_params, 1) so the zero-param case still has a valid buffer.
        if normalize:
            self.register_buffer("state_mean", torch.zeros(n_state_channels))
            self.register_buffer("state_std", torch.ones(n_state_channels))
            self.register_buffer("param_mean", torch.zeros(max(n_params, 1)))
            self.register_buffer("param_std", torch.ones(max(n_params, 1)))

        # Input channels: state + geometry, plus the SDF feature block (4) when
        # enabled, plus one channel per param in the channel-conditioning mode,
        # plus any raw extra (DD context/positional). Order at the stem is
        # [state, geometry, sdf features, param channels, extra].
        in_channels = n_state_channels + 1
        in_channels += self.n_geom_feature_channels
        if param_conditioning == "channels":
            in_channels += n_params
        in_channels += self.extra_in_channels
        self.in_channels = in_channels

        # native mode: reduce the P-vector to one scalar for pde_parameters.
        self.param_to_scalar = None
        if param_conditioning == "native" and n_params > 1:
            self.param_to_scalar = nn.Linear(n_params, 1)

        factory = getattr(p3d_surrogate, _FACTORIES[size])
        # periodic is positional in p3d (applied to spatial dims z, y, x in order).
        periodic = list(self._periodic)
        kwargs = dict(
            channel_size=in_channels,
            channel_size_out=n_state_channels,
            partition_size=partition_size,
            shift=shift,
            mending=mending,
            skip_connections_active=skip_connections_active,
            periodic=periodic,
            drop_class_labels=True,  # we never use class conditioning
        )
        if size == "S":
            kwargs["window_size"] = window_size
        self.net = factory(**kwargs)

        # Disable upstream classifier-free-guidance label dropout: with
        # drop_class_labels=True the class label is always 0, but the upstream
        # LabelEmbedder still randomly maps it to the null token in train mode,
        # which would inject nondeterministic noise into the conditioning. Zero
        # it everywhere so training is deterministic.
        for module in self.net.modules():
            if hasattr(module, "dropout_prob"):
                module.dropout_prob = 0.0

    # -- normalisation -----------------------------------------------------

    @torch.no_grad()
    def set_normalization(
        self,
        state_mean,
        state_std,
        param_mean=None,
        param_std=None,
        eps: float = 1e-6,
    ) -> None:
        """Install per-channel standardisation statistics (see ``normalize``).

        ``state_mean``/``state_std`` are length-``C`` and
        ``param_mean``/``param_std`` length-``P``. Standard deviations are
        floored at ``eps`` so a constant channel/param maps to 0 rather than
        dividing by zero.
        """
        if not self.normalize:
            print("P3D(normalize=False): ignoring normalization stats")
            return

        def _to(buf: torch.Tensor, value) -> torch.Tensor:
            t = torch.as_tensor(
                np.asarray(value), dtype=buf.dtype, device=buf.device
            ).reshape(-1)
            if t.numel() != buf.numel():
                raise ValueError(f"expected {buf.numel()} values, got {t.numel()}")
            return t

        self.state_mean.copy_(_to(self.state_mean, state_mean))
        self.state_std.copy_(_to(self.state_std, state_std).clamp_min(eps))
        if self.n_params > 0:
            if param_mean is not None:
                self.param_mean.copy_(_to(self.param_mean, param_mean))
            if param_std is not None:
                self.param_std.copy_(_to(self.param_std, param_std).clamp_min(eps))

    # -- spatial padding ---------------------------------------------------

    def _pad_to_multiple(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Zero-pad (D, H, W) up to a multiple of the downsample factor.

        Periodic axes cannot be zero-padded without breaking the circular
        convolutions, so they must already be divisible (mirrors
        ``UNetConvNeXt._pad_to_multiple``).
        """
        mult = self._downsample_factor
        d, h, w = x.shape[-3:]
        pads = tuple((mult - s % mult) % mult for s in (d, h, w))
        for name, size, pad, per in zip(
            ("z", "y", "x"), (d, h, w), pads, self._periodic
        ):
            if per and pad:
                raise ValueError(
                    f"size along periodic axis '{name}' ({size}) must be "
                    f"divisible by {mult}"
                )
        pad_d, pad_h, pad_w = pads
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        return x, (d, h, w)

    # -- SDF geometry features (inference self-compute path) ---------------

    def _sdf_features_for(self, geometry: torch.Tensor) -> torch.Tensor:
        """Fetch (or compute-and-cache) the SDF features for ``geometry``.

        ``NeuralSurrogateForwardModel.rollout_batched`` builds the stacked
        geometry tensor **once** and passes the *same object* to every step, so
        step 1 computes the EDT and steps ``2..T`` hit the cache. The cache is
        size-1 (a rollout has a single geometry; a new key evicts the old).

        Two-tier lookup, both safe against ``data_ptr`` aliasing:

        * **identity** (``is``) -- the hot path for the rollout's stable tensor
          object, O(1). The cache *holds a reference* to that object, so its
          storage stays alive and its address cannot be recycled underneath us.
        * **content** (``torch.equal``) -- a distinct but identical mask (e.g. a
          freshly built object for the same domain) reuses the cache after one
          off-hot-path compare. This is the actual correctness guarantee against
          a new mask that merely happens to reuse a freed address, and mirrors
          ``UPT._geom_cache``'s revalidation (``upt.py``).

        This runs ``scipy``'s EDT, so it must **never** be called from inside a
        ``torch.compile``'d region. It isn't: training features always arrive
        via ``geom_features`` (so this branch is unreachable under compile), and
        the inference rollout is uncompiled. If an inference rollout is ever
        compiled, this lookup must sit in a thin uncompiled wrapper.
        """
        cached = self._sdf_cache
        if cached is not None:
            cached_geom, feat = cached
            if cached_geom is geometry:
                return feat
            if (
                cached_geom.shape == geometry.shape
                and cached_geom.device == geometry.device
                and torch.equal(cached_geom, geometry)
            ):
                return feat
        feat = self._compute_sdf_features(geometry)
        self._sdf_cache = (geometry, feat)
        return feat

    def _compute_sdf_features(self, geometry: torch.Tensor) -> torch.Tensor:
        """Compute ``(B, C, *grid)`` SDF features from a geometry mask.

        ``C`` is :attr:`n_geom_feature_channels` (set by the feature mode).
        ``geometry`` may be ``(*grid,)``, ``(B, *grid)`` or ``(B, 1, *grid)``.
        For a batched mask, member 0's features are reused for every member
        whose mask equals member 0's (an ``O(B·N)`` compare, once per rollout):
        the ensemble stacks identical masks per window, so this collapses ``B``
        EDTs to 1, falling back to a per-member EDT only when they differ.
        """
        g = geometry
        if g.dim() == 5:  # (B, 1, z, y, x)
            g = g[:, 0]
        clamp = self.sdf_clamp_cells
        mode = self.sdf_feature_mode
        if g.dim() == 3:  # single (z, y, x)
            return compute_sdf_features(g, clamp_cells=clamp, mode=mode).unsqueeze(0)
        base = compute_sdf_features(g[0], clamp_cells=clamp, mode=mode)  # (C, *grid)
        feats = [base]
        for i in range(1, g.shape[0]):
            if torch.equal(g[i], g[0]):
                feats.append(base)
            else:
                feats.append(compute_sdf_features(g[i], clamp_cells=clamp, mode=mode))
        return torch.stack(feats, dim=0)  # (B, C, *grid)

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
        params: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (extra is None) != (self.extra_in_channels == 0):
            raise ValueError(
                "extra must be provided iff extra_in_channels > 0 "
                f"(extra_in_channels={self.extra_in_channels}, "
                f"extra={'None' if extra is None else 'tensor'})"
            )
        if extra is not None and extra.shape[1] != self.extra_in_channels:
            raise ValueError(
                f"extra has {extra.shape[1]} channels, expected "
                f"{self.extra_in_channels}"
            )

        # Resolve the SDF geometry features BEFORE ``geometry`` is reshaped/cast
        # below, so the inference cache keys on the caller's stable geometry
        # tensor identity (see ``_sdf_features_for``). Two paths:
        #   * training -> features arrive precomputed via ``geom_features``;
        #   * inference -> features absent, self-computed (cached) from the mask.
        if geom_features is not None and not self.sdf_features_enabled:
            raise ValueError(
                "geom_features was provided but this model was built with "
                "sdf_features=False; it has no stem channels for them."
            )
        feat = None
        if self.sdf_features_enabled:
            if geom_features is not None:
                if geom_features.shape[1] != self.n_geom_feature_channels:
                    raise ValueError(
                        f"geom_features has {geom_features.shape[1]} channels, "
                        f"expected {self.n_geom_feature_channels}"
                    )
                feat = geom_features
            else:
                feat = self._sdf_features_for(geometry)

        if geometry.dim() == state.dim() - 1:
            geometry = geometry.unsqueeze(1)
        geometry = geometry.to(dtype=state.dtype)

        # Zero out obstacle interiors (uDALES fielddumps carry junk there).
        state = state * geometry

        if self.normalize:
            ch = (1, -1) + (1,) * (state.dim() - 2)
            x = (state - self.state_mean.view(ch)) / self.state_std.view(ch)
            x = x * geometry
            if self.n_params > 0:
                params = (params - self.param_mean) / self.param_std
        else:
            x = state

        # Assemble the stem input, in order:
        #   [state, geometry, sdf_n, g_z, g_y, g_x, (param channels), (extra)].
        # The 4 SDF features sit RIGHT AFTER the mask so the "geometry block" is
        # contiguous; params/extra keep their relative positions. Like the mask
        # and ``extra`` they enter RAW -- bounded in [-1, 1] by construction, so
        # no normalisation and no geometry masking (mirrors UNetConvNeXt).
        pde_parameters = None
        pieces = [x, geometry]
        if feat is not None:
            feat = feat.to(dtype=x.dtype)
            # Self-compute on a single shared mask yields a (1, 4, *grid) tensor;
            # broadcast it (a view) to the state batch before concatenation.
            if feat.shape[0] == 1 and state.shape[0] != 1:
                feat = feat.expand(state.shape[0], *feat.shape[1:])
            pieces.append(feat)
        if self.param_conditioning == "channels" and self.n_params > 0:
            b, _, d, h, w = state.shape
            params_b = params[:, :, None, None, None].expand(b, self.n_params, d, h, w)
            pieces.append(params_b.to(dtype=x.dtype))
        elif self.param_conditioning == "native" and self.n_params > 0:
            if self.param_to_scalar is not None:
                pde_parameters = self.param_to_scalar(params).squeeze(-1)
            else:
                pde_parameters = params[:, 0]
            pde_parameters = pde_parameters.to(dtype=x.dtype)
        if extra is not None:
            pieces.append(extra)
        x = torch.cat(pieces, dim=1)

        x = x.contiguous()
        x, orig_spatial = self._pad_to_multiple(x)

        out = self.net(x, pde_parameters=pde_parameters).sample

        d, h, w = orig_spatial
        out = out[..., :d, :h, :w]

        # Map the head output back to physical units. With ``predict_residual``
        # the network predicts the (normalised) increment, so only the std
        # scales back; otherwise the mean is added too.
        if self.normalize:
            out = out * self.state_std.view(ch)
            if not self.predict_residual:
                out = out + self.state_mean.view(ch)
        if self.predict_residual:
            out = state + out

        return out * geometry

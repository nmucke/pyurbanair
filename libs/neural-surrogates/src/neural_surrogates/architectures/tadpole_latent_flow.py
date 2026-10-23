"""Conditional latent flow matching on a frozen Tadpole AE (plan 07).

``TadpoleLatentGenerator`` learns to *generate* a statistically developed flow
state -- conditioned on the obstacle geometry and a short history of the inflow
parameters -- so a surrogate rollout can start without a CFD spin-up. It is
**not** a time-stepper: it owns a frozen, pre-trained
:class:`~neural_surrogates.architectures.tadpole_ae.TadpoleAE` and trains a
single flow-matching velocity network in that AE's latent space. Decoding goes
through the plain AE decoder; none of the DFT machinery (skips, gates, LoRA,
skip mixers) is involved, because a generated latent has no input state whose
encoder skips could be reused.

Latent representation
---------------------
One canonical **full-domain latent grid** is used for all three AE spatial
modes (``local`` / ``global`` / ``halo``), so the velocity network always sees
one ``(B, D, Zl, Yl, Xl)`` token grid with ``D = n_state_channels * Cl``
(``Cl`` = the AE's latent feature count per folded channel, 256 / 512 / 1024
for S / B / L):

1. the AE's own field-IO helpers assemble the masked, z-scored working input
   and pad it with the inherited policy (global -> stride 16, local/halo -> the
   crop size);
2. :func:`~neural_surrogates.architectures._tadpole_spatial.encode_spatial`
   runs the plain AE encoder for **every** mode -- local regions simply have a
   zero halo -- and gathers one latent per central latent cell, so local mode
   produces exactly the crops the AE's own fold would, assembled on the full
   grid instead of left folded;
3. the ``(B*C_work, Cl, ...)`` result is reshaped to ``(B, C_work*Cl, ...)``,
   z-scored per latent channel with buffered statistics, and split into the
   generated **state** latents (``D`` channels) and the known **geometry**
   latents (``D_geom = n_geometry_channels*Cl``, folded-geometry AEs only).

Geometry enters the velocity network as *conditioning* only, through the
spatial FiLM of :class:`~neural_surrogates.architectures.tadpole_stepper.ParamConditionedSubnetwork`
(or, with ``velocity_conditioning="adaln"``, a per-token projection added to
the hidden tokens): a folded-geometry AE (``encode_geometry=True``) supplies
its deterministic, normalised geometry latents; a geometry-branch AE supplies the branch's
stride-16 feature (its full pyramid is handed to the decoder). An AE with
neither geometry path is rejected -- there would be nothing to condition on.

Velocity network
----------------
The DFT's fixed ``_SUBNET_SIZES`` widths are deliberately **not** inherited.
With a hidden width ``H < D`` the final ``Linear(H, D)`` confines every
velocity to a fixed ``H``-dimensional subspace, so the ODE could never remove
the initial noise in the orthogonal complement. Here ``hidden_size`` defaults
to ``D`` (rounded up to a multiple of ``num_heads``) and an explicit value
below ``D`` is rejected -- unless another path reaches every direction: the
per-token head (``head_hidden``), the ``D``-wide residual stream of
``"adaln_bottleneck"``, or the Gaussian skip (``gaussian_skip``, the exact flow
of a full-covariance Gaussian added to the net's output). The output
projection keeps its zero initialisation: the hidden activations are nonzero,
so it receives gradients on the very first step; conditioning layers (param FiLM, geometry FiLM) start receiving
gradients once it has moved.

Conditioning vector = ``concat(z-scored flattened params_hist (Hp*P),
sinusoidal embedding of the flow time tau (time_embed_dim))``. With
``velocity_conditioning="input_film"`` (default) it is fed to the subnetwork's
input FiLM, once; with ``"adaln"`` the velocity network is an
:class:`~neural_surrogates.architectures.latent_velocity_adaln.AdaLNVelocityNet`
that modulates every block from it (DiT adaLN-Zero). ``tau`` is an artificial
integration coordinate in ``[0, 1]``; it has nothing to do with physical time
or the parameter-history cadence.

Flow objective and sampling
---------------------------
With normalised target latents ``z1`` and ``z0 ~ N(0, I)``, ``tau ~ U(0, 1)``
per sample: ``z_tau = (1 - tau) z0 + tau z1``, ``v_target = z1 - z0``, and the
trainer minimises ``MSE(velocity(z_tau, tau, history, geometry), v_target)``
over all state-latent channels (padded positions included -- no voxel mask is
meaningful in latent space). :meth:`TadpoleLatentGenerator.sample` integrates
``dz/dtau = velocity`` from 0 to 1 in fp32 -- explicit Euler (default) or Heun
(explicit trapezoidal, two velocity evaluations per step) -- and decodes,
optionally with a different decoder spatial policy than the AE was trained in
(an evaluation probe; the latent grid stays the trained one).

Precision: the frozen encoder, geometry conditioning and latent statistics
always run in fp32 with autocast explicitly disabled; only the velocity
network runs under a caller's autocast during training.

Artifact contract
-----------------
The generator is self-contained: the AE's resolved constructor kwargs
(``ae_kwargs``, plain YAML-serialisable types, ``pretrained: "none"``) and a
sha256 fingerprint of its ``weights.pt`` are recorded at build time, and the
generator's own ``state_dict`` carries the frozen ``ae.*`` weights plus every
normalisation buffer. Deployment rebuilds with ``skip_pretrained_load=True,
ae_kwargs=...`` (no AE directory, no HF download) and loads one ``weights.pt``.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from neural_surrogates.architectures._tadpole_crop import CropShape
from neural_surrogates.architectures._tadpole_spatial import (
    STRIDE,
    decode_spatial,
    encode_spatial,
    validate_spatial_mode,
)
from neural_surrogates.architectures.tadpole_ae import TadpoleAE
from neural_surrogates.architectures.tadpole_stepper import ParamConditionedSubnetwork
from torch import nn

# Flow time tau in [0, 1] is scaled onto the integer-timestep range the standard
# sinusoidal embedding was designed for (as SD3 / rectified-flow models do), so
# the embedding resolves small tau differences instead of collapsing them.
_TAU_SCALE = 1000.0

# ODE samplers of :meth:`TadpoleLatentGenerator.sample` -> velocity evaluations
# (NFE) per step: Euler 1, Heun 2 (predictor + corrector on every interval).
SAMPLER_NFE_PER_STEP = {"euler": 1, "heun": 2}

# prediction="data" turns the clean-sample estimate x into the velocity
# (x - z_tau) / (1 - tau); the denominator is floored here near tau = 1.
_DATA_PREDICTION_MIN_SIGMA = 0.05

# Hydra bookkeeping keys that may sit in an exported ``architecture`` node.
_HYDRA_KEYS = ("_target_", "_partial_", "_recursive_", "_convert_", "_args_")


@dataclass
class LatentEncoding:
    """Everything :meth:`TadpoleLatentGenerator.decode_latents` /
    :meth:`~TadpoleLatentGenerator.velocity` need besides the state latents.

    Attributes
    ----------
    z:
        Normalised state latents ``(B, D, Zl, Yl, Xl)`` -- what the flow
        generates. ``None`` when produced by
        :meth:`TadpoleLatentGenerator.geometry_condition`.
    geom_cond:
        Spatial-FiLM conditioning on the latent grid ``(B, G, Zl, Yl, Xl)``:
        normalised geometry latents (folded-geometry AE) or the geometry
        branch's stride-16 feature (branch AE).
    decoder_geom_feats:
        The branch's full feature pyramid in the full-grid ``(B*C, F, ...)``
        layout the spatial decoder expects; ``None`` for a folded-geometry AE.
    orig_shape:
        Unpadded physical grid ``(d, h, w)`` the decode crops back to.
    geom_latents:
        **Raw** (un-normalised) geometry latents ``(B, D_geom, Zl, Yl, Xl)``
        for a folded-geometry AE -- re-appended to the decoder input as-is;
        ``None`` in branch mode.
    mask:
        Fluid mask ``(B, 1, d, h, w)`` applied to the decoded physical state.
    """

    z: torch.Tensor | None
    geom_cond: torch.Tensor | None
    decoder_geom_feats: list[torch.Tensor] | None
    orig_shape: tuple[int, int, int]
    geom_latents: torch.Tensor | None
    mask: torch.Tensor


def _sinusoidal_time_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """``(B,)`` flow times -> ``(B, dim)`` sin/cos embedding (``dim`` even)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=tau.device)
        / half
    )
    args = tau.to(torch.float32)[:, None] * _TAU_SCALE * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def time_shift_warp(sigma: Any, alpha: float) -> Any:
    """Noise-level warp ``sigma' = alpha*sigma / (1 + (alpha-1)*sigma)``.

    Monotone on ``[0, 1]`` with ``0 -> 0`` and ``1 -> 1``; ``alpha > 1`` spends
    more of the flow time near the noise end (SD3 / RAE time shift). Works on
    floats and tensors alike.
    """
    return alpha * sigma / (1.0 + (alpha - 1.0) * sigma)


def _gaussian_gain(
    tau: torch.Tensor, eig: torch.Tensor, prediction: str
) -> torch.Tensor:
    """Per-direction gain of the exact flow of a Gaussian ``N(mu, P diag(eig) P^T)``.

    With ``s = tau^2 eig + (1-tau)^2`` (the variance of ``z_tau`` along each
    eigendirection): ``"velocity"`` -> ``c = (tau eig - (1-tau)) / s`` (the
    coefficient of ``E[z1 - z0 | z_tau]``), ``"data"`` -> ``g = tau eig / s``
    (that of ``E[z1 | z_tau]``). ``tau`` broadcasts against ``eig``.
    """
    one_minus = 1.0 - tau
    denom = tau * tau * eig + one_minus * one_minus
    if prediction == "data":
        return tau * eig / denom
    return (tau * eig - one_minus) / denom


def gaussian_flow(
    z: torch.Tensor,
    tau: torch.Tensor,
    basis: torch.Tensor,
    eig: torch.Tensor,
    mean: torch.Tensor,
    prediction: str = "velocity",
) -> torch.Tensor:
    """Exact posterior mean of the linear flow ``z_tau = (1-tau) z0 + tau z1``
    with ``z0 ~ N(0, I)`` and ``z1 ~ N(mean, basis diag(eig) basis^T)``.

    ``z`` is token-last ``(..., D)``; ``tau`` broadcasts against ``z[..., :1]``.
    With ``y = P^T (z - tau mean)`` and ``m = P^T mean``:

    * ``prediction="velocity"``: ``E[z1 - z0 | z] = P (m + c(tau) * y)``;
    * ``prediction="data"``:     ``E[z1 | z] = mean + P (g(tau) * y)``.

    No ``1/(1-tau)`` appears, so both are finite on all of ``[0, 1]``.
    """
    y = (z - tau * mean) @ basis
    gain = _gaussian_gain(tau, eig, prediction)
    if prediction == "data":
        return mean + (gain * y) @ basis.T
    m = mean @ basis
    return (m + gain * y) @ basis.T


def _plain(obj: Any) -> Any:
    """Recursively turn OmegaConf containers into plain ``dict`` / ``list``."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:  # pragma: no cover - omegaconf is a hard dep of the repo
        pass
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TadpoleLatentGenerator(nn.Module):
    """Conditional latent flow-matching generator on a frozen ``TadpoleAE``.

    Parameters
    ----------
    n_state_channels:
        Number of state channels ``C`` (Hydra-injected from the dataset; must
        equal the AE export's state count).
    n_params:
        Number of physical parameters ``P`` per history step.
    param_history_steps:
        History length ``Hp >= 1``; ``params_hist`` is ``(B, Hp, P)``, oldest
        first.
    pretrained_ae_dir:
        Directory of a :class:`TadpoleAE` export (``config.yaml`` +
        ``weights.pt``). Its ``architecture`` node supplies the AE kwargs and
        its ``weights.pt`` is loaded strictly (unless ``skip_pretrained_load``).
    ae_kwargs:
        Explicit AE constructor kwargs instead of a directory (the deployment
        path; the saved generator config carries the resolved kwargs inline).
        Exactly one of ``pretrained_ae_dir`` / ``ae_kwargs`` must be given.
    skip_pretrained_load:
        Build the AE randomly without reading any weights -- the deploy build,
        where the generator's own ``weights.pt`` carries ``ae.*``.
    hidden_size:
        Velocity-net width; ``None`` -> ``D`` rounded up to a multiple of
        ``num_heads``. An explicit value below ``D`` is rejected (see the
        module docstring) unless another path reaches every latent direction:
        ``head_hidden``, ``velocity_conditioning="adaln_bottleneck"`` or
        ``gaussian_skip``.
    n_layers, num_heads, mlp_ratio, film_hidden:
        Velocity-net transformer / FiLM-MLP sizing (with ``"adaln"``,
        ``film_hidden`` is the width of the conditioning embedding).
    velocity_conditioning:
        ``"input_film"`` (default): a ``ParamConditionedSubnetwork`` that
        applies the conditioning vector once, as a FiLM on the input tokens,
        and the geometry as a spatial FiLM. ``"adaln"``: an
        ``AdaLNVelocityNet`` with per-block adaLN-Zero conditioning and
        additive per-token geometry. ``"adaln_bottleneck"``: an
        ``AdaLNBottleneckVelocityNet`` whose residual stream stays ``D`` wide,
        with attention at width ``hidden_size`` and a per-token MLP of width
        ``mlp_hidden``.
    head_hidden, head_layers:
        ``"adaln"`` only: a per-token adaLN MLP head of width ``head_hidden``
        with ``head_layers`` blocks on ``concat(hidden, z)``, added to the
        output projection. ``None`` builds no head.
    mlp_hidden:
        MLP width of the ``"adaln_bottleneck"`` blocks (required there,
        rejected elsewhere).
    gaussian_skip:
        Add the exact flow of a per-token Gaussian ``N(mu, Sigma)`` of the
        z-scored state latents (buffers ``latent_pca_basis`` /
        ``latent_pca_eig`` / ``latent_pca_mean``, installed with the latent
        statistics) to the network's output, so the untrained model already
        samples that Gaussian (see :func:`gaussian_flow`). ``False`` builds no
        buffers.
    gaussian_eig_floor:
        Floor of the covariance eigenvalues ``latent_pca_eig``.
    prediction:
        ``"velocity"`` (default): the network predicts the velocity.
        ``"data"``: it predicts the clean sample ``x`` (plus the Gaussian's
        ``E[z1 | z_tau]`` with ``gaussian_skip``), turned into the velocity
        ``(x - z_tau) / max(1 - tau, 0.05)``; the loss stays the velocity MSE.
    time_shift:
        ``alpha`` of :func:`time_shift_warp`: training draws ``sigma = 1 - tau``
        uniformly and warps it, and :meth:`sample` steps on the same warped
        grid. ``1`` (default) is the plain uniform schedule.
    use_checkpoint:
        Recompute velocity-transformer blocks during backward to reduce
        activation memory.
    time_embed_dim:
        Width of the sinusoidal ``tau`` embedding (even).
    num_sampling_steps:
        Default Euler step count for :meth:`sample`.
    latent_eps:
        Floor for the per-channel latent standard deviations.
    max_latent_tokens:
        Attention budget: :meth:`velocity` refuses ``B * Zl*Yl*Xl`` above it
        (naive attention allocates ``B*heads*N*N``). ``None`` = unlimited.
    normalize:
        Z-score ``params_hist`` with buffered statistics
        (:meth:`set_normalization`). State statistics are always the AE's and
        latent statistics are always applied (they are part of the objective).
    """

    def __init__(
        self,
        n_state_channels: int,
        n_params: int,
        param_history_steps: int,
        pretrained_ae_dir: str | None = None,
        ae_kwargs: dict | None = None,
        skip_pretrained_load: bool = False,
        hidden_size: int | None = None,
        n_layers: int = 4,
        num_heads: int = 8,
        time_embed_dim: int = 64,
        film_hidden: int = 128,
        mlp_ratio: int = 4,
        use_checkpoint: bool = False,
        velocity_conditioning: str = "input_film",
        num_sampling_steps: int = 50,
        latent_eps: float = 1e-6,
        max_latent_tokens: int | None = None,
        normalize: bool = True,
        head_hidden: int | None = None,
        head_layers: int = 2,
        mlp_hidden: int | None = None,
        gaussian_skip: bool = False,
        gaussian_eig_floor: float = 1e-4,
        prediction: str = "velocity",
        time_shift: float = 1.0,
    ) -> None:
        super().__init__()

        if int(param_history_steps) < 1:
            raise ValueError(
                f"param_history_steps must be >= 1, got {param_history_steps}"
            )
        if int(n_params) < 0:
            raise ValueError(f"n_params must be >= 0, got {n_params}")
        if int(time_embed_dim) < 2 or int(time_embed_dim) % 2:
            raise ValueError(
                f"time_embed_dim must be a positive even integer, got {time_embed_dim}"
            )
        if int(num_sampling_steps) < 1:
            raise ValueError(
                f"num_sampling_steps must be >= 1, got {num_sampling_steps}"
            )
        if float(latent_eps) <= 0:
            raise ValueError(f"latent_eps must be > 0, got {latent_eps}")
        if velocity_conditioning not in ("input_film", "adaln", "adaln_bottleneck"):
            raise ValueError(
                "velocity_conditioning must be 'input_film', 'adaln' or "
                f"'adaln_bottleneck', got {velocity_conditioning!r}"
            )
        if head_hidden is not None and velocity_conditioning != "adaln":
            raise ValueError(
                "head_hidden (the per-token head) needs velocity_conditioning="
                f"'adaln', got {velocity_conditioning!r}"
            )
        if (mlp_hidden is None) != (velocity_conditioning != "adaln_bottleneck"):
            raise ValueError(
                "mlp_hidden is required with velocity_conditioning="
                "'adaln_bottleneck' and only used there; got mlp_hidden="
                f"{mlp_hidden!r} with {velocity_conditioning!r}"
            )
        if prediction not in ("velocity", "data"):
            raise ValueError(
                f"prediction must be 'velocity' or 'data', got {prediction!r}"
            )
        if not float(time_shift) > 0:
            raise ValueError(f"time_shift must be > 0, got {time_shift}")
        if not float(gaussian_eig_floor) > 0:
            raise ValueError(
                f"gaussian_eig_floor must be > 0, got {gaussian_eig_floor}"
            )
        if max_latent_tokens is not None and int(max_latent_tokens) < 1:
            raise ValueError(
                f"max_latent_tokens must be None or >= 1, got {max_latent_tokens}"
            )

        self.n_state_channels = int(n_state_channels)
        self.n_params = int(n_params)
        self.param_history_steps = int(param_history_steps)
        self.time_embed_dim = int(time_embed_dim)
        self.num_sampling_steps = int(num_sampling_steps)
        self.latent_eps = float(latent_eps)
        self.max_latent_tokens = (
            None if max_latent_tokens is None else int(max_latent_tokens)
        )
        self.normalize = bool(normalize)
        self.pretrained_ae_dir = pretrained_ae_dir
        self.skip_pretrained_load = bool(skip_pretrained_load)

        # -- frozen AE ----------------------------------------------------- #
        self.ae_kwargs, self.ae_export_latent_type = self._resolve_ae_kwargs(
            pretrained_ae_dir, ae_kwargs
        )
        self.ae = TadpoleAE(**self.ae_kwargs)
        self.ae_fingerprint: str | None = None
        if pretrained_ae_dir is not None and not self.skip_pretrained_load:
            self.ae_fingerprint = self._load_ae_weights(pretrained_ae_dir)
        if self.ae.geometry_branch is None and not self.ae.encode_geometry:
            raise ValueError(
                "TadpoleLatentGenerator needs a geometry path to condition on: "
                "the AE export has neither encode_geometry=True (folded geometry "
                "latents) nor a geometry_branch. Re-export the AE with one of them."
            )
        # Deterministic latents (plan 07 v1): the wrapper AND the vendored
        # autoencoder both read this attribute, so pin both.
        self.ae.latent_type = "mode"
        self.ae.ae.latent_type = "mode"
        for p in self.ae.parameters():
            p.requires_grad_(False)
        self.ae.eval()

        # Inherited spatial policy (recorded in ae_kwargs; immutable here).
        self.spatial_mode: str = self.ae.spatial_mode
        self.encoder_crop_size: CropShape = self.ae.encoder_crop_size
        self.halo_size: int = self.ae.halo_size

        # Latent bookkeeping: Cl per folded channel, D generated state channels,
        # D_geom known geometry channels (folded-geometry AEs only).
        self.latent_channels: int = int(self.ae.ae.decoder.latent_size)
        self.n_geometry_channels: int = int(self.ae.n_geometry_channels)
        self.n_working_channels: int = self.n_state_channels + self.n_geometry_channels
        self.state_latent_dim: int = self.n_state_channels * self.latent_channels
        self.geom_latent_dim: int = self.n_geometry_channels * self.latent_channels
        self.working_latent_dim: int = self.state_latent_dim + self.geom_latent_dim
        if self.ae.geometry_branch is not None:
            branch: Any = self.ae.geometry_branch
            self.geom_cond_dim: int = int(branch.out_dims[3])
        else:
            self.geom_cond_dim = self.geom_latent_dim

        # -- velocity network ----------------------------------------------- #
        d = self.state_latent_dim
        self.head_hidden = None if head_hidden is None else int(head_hidden)
        self.head_layers = int(head_layers)
        self.mlp_hidden = None if mlp_hidden is None else int(mlp_hidden)
        self.gaussian_skip = bool(gaussian_skip)
        self.gaussian_eig_floor = float(gaussian_eig_floor)
        self.prediction = prediction
        self.time_shift = float(time_shift)
        full_reach = (
            self.head_hidden is not None
            or velocity_conditioning == "adaln_bottleneck"
            or self.gaussian_skip
        )
        if hidden_size is not None and int(hidden_size) < d and not full_reach:
            raise ValueError(
                f"hidden_size={hidden_size} is below the latent width D={d}: the "
                "final Linear(hidden_size, D) would confine every velocity to a "
                f"{hidden_size}-dimensional subspace and the flow could never "
                "remove the initial noise in its complement. Use hidden_size=None "
                f"(-> D) or a value >= {d}, or add a path that reaches every "
                "direction: head_hidden (with 'adaln'), velocity_conditioning="
                "'adaln_bottleneck', or gaussian_skip=True."
            )
        h = d if hidden_size is None else int(hidden_size)
        heads = int(num_heads)
        if heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        h = int(math.ceil(h / heads)) * heads  # round UP to a head multiple
        self.hidden_size = h
        self.n_layers = int(n_layers)
        self.num_heads = heads
        self.mlp_ratio = int(mlp_ratio)
        self.use_checkpoint = bool(use_checkpoint)
        self.velocity_conditioning = velocity_conditioning
        self.cond_dim = self.param_history_steps * self.n_params + self.time_embed_dim
        self.velocity_net: nn.Module
        if velocity_conditioning == "adaln":
            from neural_surrogates.architectures.latent_velocity_adaln import (
                AdaLNVelocityNet,
            )

            # The head kwargs only when a head is asked for: absent, the net
            # is built exactly as before.
            head: dict[str, Any] = (
                {}
                if self.head_hidden is None
                else {"head_hidden": self.head_hidden, "head_layers": self.head_layers}
            )
            self.velocity_net = AdaLNVelocityNet(
                in_dim=d,
                cond_dim=self.cond_dim,
                geom_cond_dim=self.geom_cond_dim,
                n_layers=self.n_layers,
                num_heads=heads,
                hidden_size=h,
                mlp_ratio=self.mlp_ratio,
                film_hidden=int(film_hidden),
                use_checkpoint=self.use_checkpoint,
                **head,
            )
        elif velocity_conditioning == "adaln_bottleneck":
            from neural_surrogates.architectures.latent_velocity_adaln import (
                AdaLNBottleneckVelocityNet,
            )

            assert self.mlp_hidden is not None
            self.velocity_net = AdaLNBottleneckVelocityNet(
                in_dim=d,
                cond_dim=self.cond_dim,
                geom_cond_dim=self.geom_cond_dim,
                n_layers=self.n_layers,
                num_heads=heads,
                hidden_size=h,
                mlp_hidden=self.mlp_hidden,
                film_hidden=int(film_hidden),
                use_checkpoint=self.use_checkpoint,
            )
        else:
            self.velocity_net = ParamConditionedSubnetwork(
                in_dim=d,
                n_params=self.cond_dim,
                n_layers=self.n_layers,
                num_heads=heads,
                hidden_size=h,
                param_conditioning="film",
                mlp_ratio=int(mlp_ratio),
                film_hidden=int(film_hidden),
                use_checkpoint=self.use_checkpoint,
                geom_cond_dim=self.geom_cond_dim,
            )

        # -- normalisation buffers ------------------------------------------ #
        self.register_buffer("latent_mean", torch.zeros(self.working_latent_dim))
        self.register_buffer("latent_std", torch.ones(self.working_latent_dim))
        self.register_buffer("latent_stats_installed", torch.tensor(False))
        self.register_buffer("param_mean", torch.zeros(self.n_params))
        self.register_buffer("param_std", torch.ones(self.n_params))
        if self.gaussian_skip:
            # N(mean, basis diag(eig) basis^T) of the z-scored state latents,
            # installed together with the latent statistics.
            self.register_buffer("latent_pca_basis", torch.eye(d))
            self.register_buffer("latent_pca_eig", torch.ones(d))
            self.register_buffer("latent_pca_mean", torch.zeros(d))

        # -- physical conditioning schema ------------------------------------ #
        # Plain attributes, NOT buffers: they are provenance the artifact's
        # config.yaml owns (the training script installs them before export and
        # the deploy loader re-installs them from physical_schema), so they must
        # not travel in -- or widen -- weights.pt's strict state dict.
        self.param_names: tuple[str, ...] | None = None
        self.history_dt_seconds: float | None = None

    # Buffer annotations for the type checker (registered above).
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    latent_stats_installed: torch.Tensor
    param_mean: torch.Tensor
    param_std: torch.Tensor

    # -- AE resolution ------------------------------------------------------ #

    def _resolve_ae_kwargs(
        self, pretrained_ae_dir: str | None, ae_kwargs: dict | None
    ) -> tuple[dict, str]:
        """``(plain AE kwargs, the export's latent_type)``.

        The kwargs come from the export's ``config.yaml`` ``architecture`` node
        or from an explicit ``ae_kwargs`` (never both); Hydra keys are dropped,
        unknown keys fail loud, and ``n_state_channels`` / ``pretrained="none"``
        / ``latent_type="mode"`` are pinned so the stored kwargs rebuild the
        very same (deterministic, download-free) AE anywhere.
        """
        if (pretrained_ae_dir is None) == (ae_kwargs is None):
            raise ValueError(
                "TadpoleLatentGenerator needs exactly one of pretrained_ae_dir "
                "(training: resolve the AE from its export) or ae_kwargs "
                "(deployment: the resolved kwargs saved with the generator)."
            )
        if ae_kwargs is not None:
            raw = _plain(ae_kwargs)
        else:
            assert pretrained_ae_dir is not None
            from omegaconf import OmegaConf

            cfg_path = os.path.join(pretrained_ae_dir, "config.yaml")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(
                    f"TadpoleLatentGenerator: no config.yaml at {cfg_path!r}; "
                    "pretrained_ae_dir must be a TadpoleAE export."
                )
            cfg: Any = OmegaConf.load(cfg_path)
            if "architecture" not in cfg:
                raise ValueError(f"{cfg_path!r} has no 'architecture' node")
            raw = _plain(cfg["architecture"])
            dataset_node = cfg.get("dataset")
            saved_vars = (
                None if dataset_node is None else dataset_node.get("state_vars")
            )
            if saved_vars is not None and len(saved_vars) != self.n_state_channels:
                raise ValueError(
                    f"AE export {pretrained_ae_dir!r} was pre-trained on "
                    f"{len(saved_vars)} state variables {list(saved_vars)}, but "
                    f"n_state_channels={self.n_state_channels}; the generator "
                    "dataset's ordered state variables must match the AE export."
                )
        if not isinstance(raw, dict):
            raise TypeError(f"AE kwargs must be a mapping, got {type(raw).__name__}")

        allowed = set(inspect.signature(TadpoleAE.__init__).parameters) - {"self"}
        kwargs = {k: v for k, v in raw.items() if k not in _HYDRA_KEYS}
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ValueError(
                f"AE kwargs contain keys TadpoleAE does not accept: {unknown}"
            )
        export_latent_type = str(kwargs.get("latent_type", "sample"))
        kwargs["n_state_channels"] = self.n_state_channels
        kwargs["pretrained"] = "none"
        kwargs["latent_type"] = "mode"
        return kwargs, export_latent_type

    def _load_ae_weights(self, pretrained_ae_dir: str) -> str:
        """Strictly load the export's full ``weights.pt`` into ``self.ae`` and
        return its sha256 fingerprint."""
        path = os.path.join(pretrained_ae_dir, "weights.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"TadpoleLatentGenerator: no weights.pt at {path!r}. The frozen "
                "AE (and its state normalisation) come from the export's full "
                "state dict; point pretrained_ae_dir at a TadpoleAE export."
            )
        sd = torch.load(path, map_location="cpu", weights_only=True)
        self.ae.load_state_dict(sd, strict=True)
        return _sha256_of_file(path)

    # -- module plumbing ---------------------------------------------------- #

    def train(self, mode: bool = True) -> "TadpoleLatentGenerator":
        """Like ``nn.Module.train`` but the frozen AE always stays in eval."""
        super().train(mode)
        self.ae.eval()
        return self

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Actual parameter count (default: the trainable velocity net only)."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def extra_repr(self) -> str:
        return (
            f"D={self.state_latent_dim}, hidden_size={self.hidden_size}, "
            f"n_layers={self.n_layers}, num_heads={self.num_heads}, "
            f"velocity_conditioning={self.velocity_conditioning!r}, "
            f"geom_cond_dim={self.geom_cond_dim}, "
            f"spatial_mode={self.spatial_mode!r}, "
            f"trainable_params={self.count_parameters():,}" + self._knobs_repr()
        )

    def _knobs_repr(self) -> str:
        """The full-reach / objective knobs that are set (none by default)."""
        knobs = []
        if self.head_hidden is not None:
            knobs += [
                f"head_hidden={self.head_hidden}",
                f"head_layers={self.head_layers}",
            ]
        if self.mlp_hidden is not None:
            knobs.append(f"mlp_hidden={self.mlp_hidden}")
        if self.gaussian_skip:
            knobs.append(
                f"gaussian_skip=True, gaussian_eig_floor={self.gaussian_eig_floor:g}"
            )
        if self.prediction != "velocity":
            knobs.append(f"prediction={self.prediction!r}")
        if self.time_shift != 1.0:
            knobs.append(f"time_shift={self.time_shift:g}")
        return "".join(f", {k}" for k in knobs)

    @property
    def _device(self) -> torch.device:
        return self.latent_mean.device

    @staticmethod
    def _autocast_off(device: torch.device) -> torch.autocast:
        """Context manager disabling autocast on ``device``'s backend."""
        return torch.autocast(device_type=device.type, enabled=False)

    # -- normalisation ------------------------------------------------------ #

    @torch.no_grad()  # type: ignore[misc, unused-ignore]  # torch is untyped in the pre-commit mypy env
    def set_normalization(
        self,
        state_mean: Any = None,
        state_std: Any = None,
        param_mean: Any = None,
        param_std: Any = None,
        eps: float = 1e-6,
    ) -> None:
        """Install **parameter** standardisation statistics (``P`` entries).

        ``state_mean`` / ``state_std`` are accepted for call-site parity with the
        other architectures but ignored: the frozen AE owns the state
        statistics (they travel in its ``weights.pt`` and are what its encoder
        was pre-trained on), so re-installing them here would silently feed the
        encoder a different distribution.
        """
        if not self.normalize:
            print("TadpoleLatentGenerator(normalize=False): ignoring param stats")
            return
        if self.n_params == 0:
            return
        if param_mean is not None:
            self.param_mean.copy_(self._to_buffer(self.param_mean, param_mean))
        if param_std is not None:
            self.param_std.copy_(self._to_buffer(self.param_std, param_std, eps))

    @staticmethod
    def _to_buffer(
        buf: torch.Tensor, value: Any, eps: float | None = None
    ) -> torch.Tensor:
        t = torch.as_tensor(value, dtype=buf.dtype, device=buf.device).reshape(-1)
        if t.numel() != buf.numel():
            raise ValueError(f"expected {buf.numel()} values, got {t.numel()}")
        if not torch.isfinite(t).all():
            raise ValueError("normalisation statistics must be finite")
        if eps is not None:
            t = t.clamp_min(eps)
        return t

    @torch.no_grad()  # type: ignore[misc, unused-ignore]  # torch is untyped in the pre-commit mypy env
    def set_latent_normalization(
        self,
        mean: Any,
        std: Any,
        *,
        eps: float | None = None,
        second_moment: Any = None,
    ) -> None:
        """Install per-channel latent statistics for all ``D + D_geom`` working
        latent channels (state first, then geometry) and mark them installed.

        With ``gaussian_skip``, ``second_moment`` -- the raw ``E[x x^T]``
        ``(D, D)`` of the **state** latents over the same cells as ``mean`` --
        is required, and the Gaussian buffers are filled from it (see
        :meth:`_set_latent_gaussian`); without it, it is ignored.
        """
        if self.gaussian_skip and second_moment is None:
            raise ValueError(
                "gaussian_skip=True needs the raw second moment of the state "
                "latents (second_moment=E[x x^T], (D, D)) with the latent "
                "statistics: use compute_latent_normalization, or a latent "
                "cache that stores it (prepared by this version of the code)."
            )
        eps = self.latent_eps if eps is None else float(eps)
        self.latent_mean.copy_(self._to_buffer(self.latent_mean, mean))
        self.latent_std.copy_(self._to_buffer(self.latent_std, std, eps))
        if self.gaussian_skip:
            self._set_latent_gaussian(mean, second_moment)
        self.latent_stats_installed.fill_(True)

    def _set_latent_gaussian(self, mean: Any, second_moment: Any) -> None:
        """Fill ``latent_pca_*`` with the z-scored state-latent Gaussian.

        ``Sigma_z = S^-1 (E[x x^T] - m m^T) S^-1`` with ``S = diag(std)`` the
        installed (floored) state std, eigendecomposed in float64 with the
        eigenvalues floored at ``gaussian_eig_floor``; the mean is
        ``(m - latent_mean) / latent_std`` (zero up to the buffers' fp32
        rounding).
        """
        d = self.state_latent_dim
        m = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)[:d].cpu()
        m2 = torch.as_tensor(second_moment, dtype=torch.float64).cpu()
        if tuple(m2.shape) != (d, d):
            raise ValueError(
                f"second_moment must be (D, D) = ({d}, {d}), got {tuple(m2.shape)}"
            )
        if not torch.isfinite(m2).all():
            raise ValueError("second_moment must be finite")
        std = self.latent_std[:d].to(torch.float64).cpu()
        cov = (m2 - torch.outer(m, m)) / torch.outer(std, std)
        eig, basis = torch.linalg.eigh(0.5 * (cov + cov.T))
        eig = eig.clamp_min(self.gaussian_eig_floor)
        z_mean = (m - self.latent_mean[:d].to(torch.float64).cpu()) / std
        self.latent_pca_basis.copy_(basis)
        self.latent_pca_eig.copy_(eig)
        self.latent_pca_mean.copy_(z_mean)

    @torch.no_grad()  # type: ignore[misc, unused-ignore]  # torch is untyped in the pre-commit mypy env
    def compute_latent_normalization(
        self,
        batches: Iterable[Sequence[Any]],
        max_batches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate and install raw-latent statistics over ``batches``.

        ``batches`` yields ``(state, geometry, geom_features)`` tuples already on
        the model device (what ``BaseTraining._prepare_snapshot_batch``
        produces). Raw (un-normalised) latents are encoded in fp32 and their
        sums / sums of squares accumulated in float64 over the batch and
        spatial axes for **every** working latent channel; standard deviations
        are floored at ``latent_eps`` and near-constant channels are reported.
        With ``gaussian_skip`` the sum of the state latents' outer products is
        accumulated as well (float64), for the Gaussian buffers.
        Returns the installed ``(mean, std)``.
        """
        n_work = self.working_latent_dim
        total = torch.zeros(n_work, dtype=torch.float64, device=self._device)
        total_sq = torch.zeros_like(total)
        d = self.state_latent_dim
        outer = (
            torch.zeros(d, d, dtype=torch.float64, device=self._device)
            if self.gaussian_skip
            else None
        )
        count = 0
        n_batches = 0
        for batch in batches:
            if max_batches is not None and n_batches >= max_batches:
                break
            state, geometry, geom_features = batch[0], batch[1], batch[2]
            with self._autocast_off(self._device):
                z_raw, geom_raw, *_ = self._encode_raw(state, geometry, geom_features)
            work = z_raw if geom_raw is None else torch.cat([z_raw, geom_raw], dim=1)
            w64 = work.to(torch.float64)
            total += w64.sum(dim=(0, 2, 3, 4))
            total_sq += (w64 * w64).sum(dim=(0, 2, 3, 4))
            if outer is not None:
                tokens = w64[:, :d].transpose(0, 1).reshape(d, -1)
                outer += tokens @ tokens.T
            count += w64.shape[0] * w64.shape[2] * w64.shape[3] * w64.shape[4]
            n_batches += 1
        if count == 0:
            raise ValueError(
                "compute_latent_normalization received no batches (empty loader?)"
            )
        mean = total / count
        var = (total_sq / count - mean * mean).clamp_min(0.0)
        std = var.sqrt()
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ValueError("latent statistics are not finite; check the AE/data")
        near_constant = int((std < 10.0 * self.latent_eps).sum().item())
        if near_constant:
            warnings.warn(
                f"compute_latent_normalization: {near_constant}/{n_work} latent "
                f"channels are near-constant (std < {10.0 * self.latent_eps:g}); "
                f"their std is floored at latent_eps={self.latent_eps:g}.",
                stacklevel=2,
            )
        print(
            f"TadpoleLatentGenerator: latent stats over {n_batches} batches, "
            f"{count} latent cells, {near_constant} near-constant channel(s)"
        )
        self.set_latent_normalization(
            mean, std, second_moment=None if outer is None else outer / count
        )
        return self.latent_mean.clone(), self.latent_std.clone()

    def _require_latent_stats(self, what: str) -> None:
        if not bool(self.latent_stats_installed):
            raise RuntimeError(
                f"{what} needs latent normalisation statistics, but none are "
                "installed: call compute_latent_normalization(...) (or "
                "set_latent_normalization) first, or load a trained state dict."
            )

    # -- latent grid bookkeeping -------------------------------------------- #

    def _padded_shape(self, grid: Sequence[int]) -> tuple[int, int, int]:
        multiples = (
            (STRIDE, STRIDE, STRIDE)
            if self.spatial_mode == "global"
            else self.encoder_crop_size
        )
        d, h, w = (int(s) for s in grid)
        return tuple(
            size + (multiple - size % multiple) % multiple
            for size, multiple in zip((d, h, w), multiples)
        )  # type: ignore[return-value]

    def latent_grid_for(self, grid: Sequence[int]) -> tuple[int, int, int]:
        """Latent grid ``(Zl, Yl, Xl)`` for a physical grid ``(d, h, w)``."""
        return tuple(s // STRIDE for s in self._padded_shape(grid))  # type: ignore[return-value]

    def _check_latent_grid(
        self, t: torch.Tensor, expected: tuple[int, int, int], what: str
    ) -> None:
        """Mirror of ``TadpoleTimeStepper._check_geom_cond_grid``: a latent-grid
        tensor must sit on the padded grid / 16, or it would broadcast silently."""
        got = tuple(int(s) for s in t.shape[2:])
        if got != expected:
            raise ValueError(
                f"{what} has latent grid {got} but the geometry's padded grid / "
                f"{STRIDE} is {expected}"
            )

    # -- encoding ----------------------------------------------------------- #

    def _full_grid_feats(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Branch pyramid ``(B, F, ...)`` -> the ``(B*C, F, ...)`` full-grid
        layout ``encode_spatial`` / ``decode_spatial`` slice regions from (the
        same expansion ``_fold_geom_feats`` does outside local mode; local mode
        must NOT use that helper's crop fold here)."""
        c = self.n_state_channels
        return [
            f.unsqueeze(1).expand(-1, c, -1, -1, -1, -1).flatten(0, 1) for f in feats
        ]

    def _geometry_conditioning(
        self,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[list[torch.Tensor] | None, torch.Tensor | None, torch.Tensor | None]:
        """``(decoder_geom_feats, branch_cond, geom_latents_raw)`` for a
        ``(B, *grid)`` mask -- the single code path both :meth:`encode_latents`
        and :meth:`geometry_condition` go through, so the two agree exactly."""
        b = geometry.shape[0]
        grid = geometry.shape[1:]
        block = self.ae._geometry_channels(geometry, geom_features, b, grid, dtype)
        block, _ = self.ae._pad_to_crop_multiple(block)
        branch = self.ae.geometry_branch
        if branch is not None:
            feats: list[torch.Tensor] = branch(block)
            return self._full_grid_feats(feats), feats[3], None
        latent, _, _ = encode_spatial(
            self.ae.ae,
            block,
            self.spatial_mode,
            self.encoder_crop_size,
            self.halo_size,
            None,
            latent_type="mode",
        )  # (B*n_geom, Cl, Zl, Yl, Xl)
        geom_raw = latent.reshape(b, self.geom_latent_dim, *latent.shape[2:])
        return None, None, geom_raw

    @torch.no_grad()  # type: ignore[misc, unused-ignore]  # torch is untyped in the pre-commit mypy env
    def _encode_raw(
        self,
        state: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.Tensor] | None,
        torch.Tensor,
        tuple[int, int, int],
    ]:
        """Raw (un-normalised) full-grid latents of ``state`` in fp32.

        Returns ``(z_raw (B, D, ...), geom_raw (B, D_geom, ...) | None,
        branch_cond | None, decoder_geom_feats | None, mask (B, 1, *grid),
        orig_shape)``. The state channels and the geometry block are encoded
        separately (folded channels are independent anyway), so the geometry
        part is bit-identical to :meth:`geometry_condition`.
        """
        if state.dim() != 5 or state.shape[1] != self.n_state_channels:
            raise ValueError(
                f"state must be (B, {self.n_state_channels}, d, h, w), got "
                f"{tuple(state.shape)}"
            )
        state = state.to(torch.float32)
        geometry = self.ae._expand_geometry(geometry, state).to(state.device)
        mask = geometry.unsqueeze(1).to(dtype=state.dtype)
        x = self.ae._normalize_state(state, mask)
        x_pad, orig = self.ae._pad_to_crop_multiple(x)
        dec_feats, branch_cond, geom_raw = self._geometry_conditioning(
            geometry, geom_features, state.dtype
        )
        latent, _, _ = encode_spatial(
            self.ae.ae,
            x_pad,
            self.spatial_mode,
            self.encoder_crop_size,
            self.halo_size,
            dec_feats,
            latent_type="mode",
        )  # (B*C, Cl, Zl, Yl, Xl)
        b = state.shape[0]
        z_raw = latent.reshape(b, self.state_latent_dim, *latent.shape[2:])
        return z_raw, geom_raw, branch_cond, dec_feats, mask, orig

    def _normalize_latents(self, raw: torch.Tensor, start: int) -> torch.Tensor:
        n = raw.shape[1]
        view = (1, n, 1, 1, 1)
        mean = self.latent_mean[start : start + n].view(view)
        std = self.latent_std[start : start + n].view(view)
        return (raw - mean) / std

    def _denormalize_latents(self, z: torch.Tensor, start: int) -> torch.Tensor:
        n = z.shape[1]
        view = (1, n, 1, 1, 1)
        mean = self.latent_mean[start : start + n].view(view)
        std = self.latent_std[start : start + n].view(view)
        return z * std + mean

    def encode_latents(
        self,
        state: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
    ) -> LatentEncoding:
        """Deterministic normalised state latents + geometry conditioning."""
        self._require_latent_stats("encode_latents")
        with torch.no_grad(), self._autocast_off(state.device):
            z_raw, geom_raw, branch_cond, dec_feats, mask, orig = self._encode_raw(
                state, geometry, geom_features
            )
            z = self._normalize_latents(z_raw, 0)
            geom_cond = (
                branch_cond
                if geom_raw is None
                else self._normalize_latents(geom_raw, self.state_latent_dim)
            )
        return LatentEncoding(
            z=z,
            geom_cond=geom_cond,
            decoder_geom_feats=dec_feats,
            orig_shape=orig,
            geom_latents=geom_raw,
            mask=mask,
        )

    def geometry_condition(
        self,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
        batch_size: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> LatentEncoding:
        """The conditioning :meth:`encode_latents` would derive, without a state.

        ``geometry`` is ``(*grid,)`` or ``(B, *grid)``; ``batch_size`` (default:
        the geometry's own batch, 1 if unbatched) expands a single mask to the
        member batch. ``dtype`` defaults to fp32 (the frozen path's precision)
        and ``device`` to the model's.
        """
        self._require_latent_stats("geometry_condition")
        device = self._device if device is None else torch.device(device)
        dtype = torch.float32 if dtype is None else dtype
        g = torch.as_tensor(geometry, device=device)
        if g.dim() == 3:
            g = g.unsqueeze(0)
        if g.dim() != 4:
            raise ValueError(
                f"geometry must be (*grid,) or (B, *grid), got {tuple(g.shape)}"
            )
        b = g.shape[0] if batch_size is None else int(batch_size)
        if b < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if g.shape[0] == 1 and b != 1:
            g = g.expand(b, *g.shape[1:])
        elif g.shape[0] != b:
            raise ValueError(
                f"geometry batch {g.shape[0]} does not match batch_size={b}"
            )
        with torch.no_grad(), self._autocast_off(device):
            mask = g.unsqueeze(1).to(dtype=torch.float32)
            dec_feats, branch_cond, geom_raw = self._geometry_conditioning(
                g, None if geom_features is None else geom_features.to(device), dtype
            )
            geom_cond = (
                branch_cond
                if geom_raw is None
                else self._normalize_latents(geom_raw, self.state_latent_dim)
            )
        d, h, w = (int(s) for s in g.shape[1:])
        return LatentEncoding(
            z=None,
            geom_cond=geom_cond,
            decoder_geom_feats=dec_feats,
            orig_shape=(d, h, w),
            geom_latents=geom_raw,
            mask=mask,
        )

    # -- decoding ----------------------------------------------------------- #

    def _resolve_decode_mode(
        self, mode: str | None, halo_size: int | None
    ) -> tuple[str, int]:
        """``(spatial_mode, halo_size)`` :meth:`decode_latents` runs the frozen
        decoder with.

        ``None`` inherits the AE's trained ``spatial_mode`` / ``halo_size``. An
        override changes the DECODER only: the latent grid (and so the flow
        model's input) stays the one the trained mode's padding defines, and
        the decoder's geometry features live on that same padded grid. ``global``
        decodes the whole latent grid at once; ``local`` tiles it into
        ``encoder_crop_size`` cores and ``halo`` adds ``halo_size`` cells of
        latent context around each core (clipped at the grid edge). When the
        padded grid is not a multiple of the crop (an AE trained ``global``),
        the last tile along an axis is simply shorter -- no latents are
        invented to fill it. The decoder computes a different function in each
        mode; only the trained one reproduces the AE's reconstruction.
        """
        resolved = self.spatial_mode if mode is None else mode
        halo = self.halo_size if halo_size is None else halo_size
        validate_spatial_mode(resolved, halo)
        return resolved, halo

    def decode_latents(
        self,
        z: torch.Tensor,
        cond: LatentEncoding,
        *,
        spatial_mode: str | None = None,
        halo_size: int | None = None,
    ) -> torch.Tensor:
        """Normalised state latents ``(B, D, Zl, Yl, Xl)`` -> physical state
        ``(B, C, d, h, w)`` on the original grid, obstacle cells zeroed.

        ``spatial_mode`` (``local`` / ``halo`` / ``global``) and ``halo_size``
        (halo mode; a nonnegative multiple of 16) override the decoder's
        spatial processing for this call (see :meth:`_resolve_decode_mode`);
        ``None`` keeps the AE's."""
        expected = self.latent_grid_for(cond.orig_shape)
        if z.dim() != 5 or z.shape[1] != self.state_latent_dim:
            raise ValueError(
                f"z must be (B, {self.state_latent_dim}, Zl, Yl, Xl), got "
                f"{tuple(z.shape)}"
            )
        self._check_latent_grid(z, expected, "z")
        mode, halo = self._resolve_decode_mode(spatial_mode, halo_size)
        b = z.shape[0]
        c, cl = self.n_state_channels, self.latent_channels
        with self._autocast_off(z.device):
            state_raw = self._denormalize_latents(z.to(torch.float32), 0)
            if cond.geom_latents is not None:
                # Folded-geometry AE: hand the decoder the full C_work fold in
                # working-channel order (state, then geometry). The geometry
                # latents are raw already (never normalised).
                if cond.geom_latents.shape[0] != b:
                    raise ValueError(
                        f"cond.geom_latents batch {cond.geom_latents.shape[0]} "
                        f"does not match z batch {b}"
                    )
                full = torch.cat([state_raw, cond.geom_latents], dim=1)
            else:
                full = state_raw
            n_work = full.shape[1] // cl
            folded = full.reshape(b * n_work, cl, *full.shape[2:]).contiguous()
            decoded = decode_spatial(
                self.ae.ae,
                folded,
                mode,
                self.encoder_crop_size,
                halo,
                cond.decoder_geom_feats,
            )  # (B*C_work, 1, Dp, Hp, Wp)
            d, h, w = cond.orig_shape
            recon = decoded.reshape(b, n_work, *decoded.shape[2:])[:, :c, :d, :h, :w]
            state = self.ae._denormalize_state(recon)
            return state * cond.mask.to(state)

    # -- conditioning schema ------------------------------------------------- #

    def set_conditioning_schema(
        self, param_names: Sequence[str], history_dt_seconds: float | None
    ) -> None:
        """Install the physical meaning of the ``params_hist`` columns.

        ``params_hist`` is a bare ``(B, Hp, P)`` tensor: reordering its columns
        or feeding it a history saved at another cadence is invisible to every
        shape check, yet conditions the flow on something else entirely. The
        training script installs the dataset's ordered ``param_names`` and
        median saved cadence before exporting, and the deployment loader
        re-installs them from the artifact's ``physical_schema``, so a caller
        that states its own schema (see :meth:`sample` / :meth:`velocity`) is
        checked against the one the weights were trained with.

        ``history_dt_seconds=None`` records "cadence unknown"; a caller that
        then supplies one raises rather than being silently accepted.
        """
        names = tuple(str(n) for n in param_names)
        if len(names) != self.n_params:
            raise ValueError(
                f"param_names has {len(names)} entries {list(names)} but the "
                f"model conditions on n_params={self.n_params} columns."
            )
        if history_dt_seconds is not None:
            dt = float(history_dt_seconds)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError(
                    f"history_dt_seconds must be a positive, finite number or "
                    f"None, got {history_dt_seconds!r}."
                )
            self.history_dt_seconds = dt
        else:
            self.history_dt_seconds = None
        self.param_names = names

    def _check_conditioning_schema(
        self,
        param_names: Sequence[str] | None,
        history_dt_seconds: float | None,
        what: str,
    ) -> None:
        """Validate a caller-supplied schema against the installed one.

        Supplying nothing skips the check (the training path, which owns the
        dataset the schema came from). Supplying something while no schema is
        installed is an error, not a pass: there would be nothing to check the
        claim against.
        """
        if param_names is None and history_dt_seconds is None:
            return
        if self.param_names is None:
            raise ValueError(
                f"{what}: a conditioning schema was supplied (param_names="
                f"{None if param_names is None else list(param_names)}, "
                f"history_dt_seconds={history_dt_seconds!r}) but this model "
                "carries none; call set_conditioning_schema(...) first (the "
                "deployment loader does it from the artifact's physical_schema)."
            )
        if param_names is not None:
            given = tuple(str(n) for n in param_names)
            if given != self.param_names:
                raise ValueError(
                    f"{what}: param_names {list(given)} do not match the model's "
                    f"conditioning schema {list(self.param_names)} (order "
                    "matters -- params_hist columns are positional)."
                )
        if history_dt_seconds is not None:
            dt = float(history_dt_seconds)
            if self.history_dt_seconds is None:
                raise ValueError(
                    f"{what}: history_dt_seconds={dt!r} was supplied but the "
                    "model's schema records no cadence to check it against."
                )
            if abs(dt - self.history_dt_seconds) > 1e-6 * abs(self.history_dt_seconds):
                raise ValueError(
                    f"{what}: history_dt_seconds={dt:.6g} s does not match the "
                    f"model's trained cadence {self.history_dt_seconds:.6g} s; "
                    f"Hp={self.param_history_steps} rows would span a different "
                    "physical duration."
                )

    # -- velocity ----------------------------------------------------------- #

    def _check_params_hist(self, params_hist: torch.Tensor, b: int) -> None:
        hp, p = self.param_history_steps, self.n_params
        if params_hist.dim() != 3 or tuple(params_hist.shape) != (b, hp, p):
            raise ValueError(
                f"params_hist must be (B={b}, Hp={hp}, P={p}), got "
                f"{tuple(params_hist.shape)}"
            )
        if not torch.isfinite(params_hist).all():
            raise ValueError("params_hist contains non-finite values")

    def _conditioning_vector(
        self, tau: torch.Tensor, params_hist: torch.Tensor
    ) -> torch.Tensor:
        """``concat(z-scored flattened history, sinusoidal tau)`` -> ``(B, cond_dim)``."""
        p = params_hist.to(torch.float32)
        if self.normalize and self.n_params > 0:
            p = (p - self.param_mean) / self.param_std
        t_emb = _sinusoidal_time_embedding(tau, self.time_embed_dim)
        return torch.cat([p.flatten(1), t_emb], dim=1)

    def velocity(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        params_hist: torch.Tensor,
        cond: LatentEncoding,
        *,
        param_names: Sequence[str] | None = None,
        history_dt_seconds: float | None = None,
    ) -> torch.Tensor:
        """Flow velocity ``dz/dtau`` at ``(z, tau)`` -> ``(B, D, Zl, Yl, Xl)``.

        ``params_hist`` is in **raw physical units** (z-scored here); ``tau`` is
        ``(B,)``. Raises when ``B * Zl*Yl*Xl`` exceeds ``max_latent_tokens``.
        ``param_names`` / ``history_dt_seconds`` state what the columns of
        ``params_hist`` mean and are checked against the installed schema (see
        :meth:`set_conditioning_schema`).
        """
        self._check_conditioning_schema(param_names, history_dt_seconds, "velocity")
        return self._velocity(z, tau, params_hist, cond.geom_cond)

    def _velocity(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        params_hist: torch.Tensor,
        geom_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        """:meth:`velocity` past the schema check, on the bare ``geom_cond``."""
        if z.dim() != 5 or z.shape[1] != self.state_latent_dim:
            raise ValueError(
                f"z must be (B, {self.state_latent_dim}, Zl, Yl, Xl), got "
                f"{tuple(z.shape)}"
            )
        b = z.shape[0]
        if tau.dim() != 1 or tau.shape[0] != b:
            raise ValueError(f"tau must be (B={b},), got {tuple(tau.shape)}")
        self._check_params_hist(params_hist, b)
        if geom_cond is None:
            raise ValueError("geom_cond is required (geometry conditioning)")
        if tuple(geom_cond.shape) != (b, self.geom_cond_dim, *z.shape[2:]):
            raise ValueError(
                f"geom_cond must be (B={b}, G={self.geom_cond_dim}, "
                f"{tuple(z.shape[2:])}), got {tuple(geom_cond.shape)}"
            )
        n_tokens = b * int(z.shape[2]) * int(z.shape[3]) * int(z.shape[4])
        if self.max_latent_tokens is not None and n_tokens > self.max_latent_tokens:
            raise ValueError(
                f"latent attention budget exceeded: B*N = {b} * "
                f"{tuple(int(s) for s in z.shape[2:])} = {n_tokens} tokens > "
                f"max_latent_tokens={self.max_latent_tokens} (naive attention "
                "allocates B*heads*N*N). Reduce the batch or the domain, or raise "
                "the budget after profiling."
            )
        cond_vec = self._conditioning_vector(tau, params_hist)
        v: torch.Tensor = self.velocity_net(z, cond_vec, geom_cond=geom_cond)
        if not self.gaussian_skip and self.prediction == "velocity":
            return v
        return self._velocity_from_net(z, tau, v)

    def _velocity_from_net(
        self, z: torch.Tensor, tau: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """The velocity from the net's output ``out`` under ``gaussian_skip`` /
        ``prediction="data"``, in fp32 outside autocast.

        ``"velocity"``: ``v = v_G + out``. ``"data"``: ``x = x_G + out`` (``x =
        out`` without the skip) and ``v = (x - z) / max(1 - tau, 0.05)``.
        ``v_G`` / ``x_G`` are :func:`gaussian_flow` of the installed Gaussian.
        """
        with self._autocast_off(z.device):
            z32 = z.to(torch.float32)
            x = out.to(torch.float32)
            if self.gaussian_skip:
                tokens = z32.flatten(2).transpose(1, 2)  # (B, N, D)
                base = gaussian_flow(
                    tokens,
                    tau.to(torch.float32).view(-1, 1, 1),
                    self.latent_pca_basis,
                    self.latent_pca_eig,
                    self.latent_pca_mean,
                    self.prediction,
                )
                x = x + base.transpose(1, 2).reshape(z32.shape)
            if self.prediction == "velocity":
                return x
            t = tau.to(torch.float32).view(-1, 1, 1, 1, 1)
            return (x - z32) / (1.0 - t).clamp_min(_DATA_PREDICTION_MIN_SIGMA)

    # -- training objective ------------------------------------------------- #

    def forward(
        self,
        state: torch.Tensor,
        params_hist: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One flow-matching draw: ``(v_pred, v_target)``, both ``(B, D, ...)``.

        The frozen encoding runs in fp32 without autocast; only the velocity
        net sees the caller's autocast context. ``generator`` makes the noise /
        flow-time draws reproducible (e.g. a fixed validation set).
        """
        self._check_params_hist(params_hist, state.shape[0])
        cond = self.encode_latents(state, geometry, geom_features)
        assert cond.z is not None
        return self._flow_matching_draw(cond.z, cond.geom_cond, params_hist, generator)

    def forward_cached(
        self,
        z_raw: torch.Tensor,
        geom: torch.Tensor,
        params_hist: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`forward` from cached raw latents, skipping the frozen encoder.

        ``z_raw`` is ``_encode_raw``'s raw state latent ``(B, D, Zl, Yl, Xl)``;
        ``geom`` the raw geometry conditioning: the branch feature (used as-is)
        or the geometry latent (normalised here, as :meth:`encode_latents`
        does). Given the same inputs and ``generator`` it equals :meth:`forward`.
        """
        self._check_params_hist(params_hist, z_raw.shape[0])
        self._require_latent_stats("forward_cached")
        with torch.no_grad(), self._autocast_off(z_raw.device):
            z = self._normalize_latents(z_raw.to(torch.float32), 0)
            geom_cond = geom.to(torch.float32)
            if self.ae.geometry_branch is None:
                geom_cond = self._normalize_latents(geom_cond, self.state_latent_dim)
        return self._flow_matching_draw(z, geom_cond, params_hist, generator)

    def _flow_matching_draw(
        self,
        z1: torch.Tensor,
        geom_cond: torch.Tensor | None,
        params_hist: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw ``z0`` / ``tau`` for normalised latents ``z1`` -> ``(v_pred, v_target)``."""
        z0 = torch.randn(
            z1.shape, generator=generator, device=z1.device, dtype=z1.dtype
        )
        tau = torch.rand(
            z1.shape[0], generator=generator, device=z1.device, dtype=z1.dtype
        )
        if self.time_shift != 1.0:  # sigma = 1 - tau uniform, then warped
            tau = 1.0 - time_shift_warp(1.0 - tau, self.time_shift)
        t = tau.view(-1, 1, 1, 1, 1)
        z_tau = (1.0 - t) * z0 + t * z1
        v_target = z1 - z0
        v_pred = self._velocity(z_tau, tau, params_hist, geom_cond)
        return v_pred, v_target

    # -- sampling ----------------------------------------------------------- #

    def sampling_times(self, steps: int) -> list[float]:
        """The ``steps + 1`` flow times ``0 = tau_0 < ... < tau_steps = 1`` of
        :meth:`sample`: ``i / steps``, or with ``time_shift`` the training
        warp ``tau_i = 1 - warp(1 - i / steps)``."""
        dt = 1.0 / steps
        if self.time_shift == 1.0:
            return [i * dt for i in range(steps + 1)]
        return [
            1.0 - time_shift_warp(1.0 - i / steps, self.time_shift)
            for i in range(steps + 1)
        ]

    @torch.no_grad()  # type: ignore[misc, unused-ignore]  # torch is untyped in the pre-commit mypy env
    def sample(
        self,
        params_hist: torch.Tensor,
        geometry: torch.Tensor,
        geom_features: torch.Tensor | None = None,
        *,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int | None = None,
        param_names: Sequence[str] | None = None,
        history_dt_seconds: float | None = None,
        decode_spatial_mode: str | None = None,
        decode_halo_size: int | None = None,
        sampler: str = "euler",
        return_latents: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Generate physical states ``(B, C, d, h, w)`` for ``params_hist``.

        Integrates from ``tau = 0`` to ``1`` in ``num_steps`` intervals
        (equal, or warped by ``time_shift``; see :meth:`sampling_times`) in
        fp32 (autocast disabled), then :meth:`decode_latents`. ``sampler``:
        ``"euler"`` (explicit Euler, ``num_steps`` velocity evaluations) or
        ``"heun"`` (explicit trapezoidal: an Euler predictor, then the average
        of the velocities at both ends of the interval -- the full step on
        every interval, ``tau = 1`` included, so ``2 * num_steps`` evaluations;
        see ``SAMPLER_NFE_PER_STEP``). Pass ``initial_noise`` ``(B, D, Zl, Yl,
        Xl)`` for reproducible per-member noise **or** a ``generator`` -- never
        both. ``param_names`` / ``history_dt_seconds`` state what the caller
        believes the ``params_hist`` columns and their spacing are; they are
        checked once here against the installed schema (see
        :meth:`set_conditioning_schema`) rather than per Euler step.
        ``decode_spatial_mode`` / ``decode_halo_size`` override the decoder's
        spatial processing (``None``: the AE's trained policy; see
        :meth:`_resolve_decode_mode`); the sampled latents are the same.
        ``return_latents=True`` returns ``(states, z)`` with ``z`` the
        normalised latents ``(B, D, Zl, Yl, Xl)`` after the last step, the
        ones that were decoded.
        """
        self._check_conditioning_schema(param_names, history_dt_seconds, "sample")
        if initial_noise is not None and generator is not None:
            raise ValueError("pass either initial_noise or generator, not both")
        self._require_latent_stats("sample")
        if params_hist.dim() != 3:
            raise ValueError(
                f"params_hist must be (B, Hp, P), got {tuple(params_hist.shape)}"
            )
        b = params_hist.shape[0]
        self._check_params_hist(params_hist, b)
        device = self._device
        params_hist = params_hist.to(device)
        steps = self.num_sampling_steps if num_steps is None else int(num_steps)
        if steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if sampler not in SAMPLER_NFE_PER_STEP:
            raise ValueError(
                f"sampler must be one of {sorted(SAMPLER_NFE_PER_STEP)}, got {sampler!r}"
            )
        self._resolve_decode_mode(decode_spatial_mode, decode_halo_size)

        cond = self.geometry_condition(
            geometry, geom_features, batch_size=b, device=device
        )
        latent_grid = self.latent_grid_for(cond.orig_shape)
        assert cond.geom_cond is not None
        self._check_latent_grid(cond.geom_cond, latent_grid, "geometry conditioning")
        shape = (b, self.state_latent_dim, *latent_grid)
        if initial_noise is not None:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"initial_noise must be {shape}, got {tuple(initial_noise.shape)}"
                )
            z = initial_noise.to(device=device, dtype=torch.float32)
        else:
            z = torch.randn(shape, generator=generator, device=device)

        with self._autocast_off(device):
            taus = self.sampling_times(steps)
            for i in range(steps):
                # Uniform grid: the constant 1/steps, as before time_shift.
                dt = 1.0 / steps if self.time_shift == 1.0 else taus[i + 1] - taus[i]
                tau = torch.full((b,), taus[i], device=device, dtype=torch.float32)
                if sampler == "euler":
                    z = z + dt * self.velocity(z, tau, params_hist, cond)
                    continue
                v = self.velocity(z, tau, params_hist, cond)
                tau_next = torch.full(
                    (b,), taus[i + 1], device=device, dtype=torch.float32
                )
                v_next = self.velocity(z + dt * v, tau_next, params_hist, cond)
                z = z + (0.5 * dt) * (v + v_next)
            fields = self.decode_latents(
                z,
                cond,
                spatial_mode=decode_spatial_mode,
                halo_size=decode_halo_size,
            )
        return (fields, z) if return_latents else fields

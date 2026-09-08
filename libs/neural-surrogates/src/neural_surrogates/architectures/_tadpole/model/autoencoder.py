"""Tadpole autoencoder (vendored from Tadpole).

pyurbanair edit -- **geometry-branch conditioning**: an optional ``geom_in_dims``
(the 4 ``GeometryBranch.out_dims``) is forwarded to the encoder/decoder, which
then own the zero-init conditioning projections (stride 1/2/4 in the conv stem
and its mirror, stride 16 on the decoder's latent input; see
``architecture/p3d/conv.py`` and ``core.py``), and ``forward`` takes the folded
branch features as ``geom_feats``. When the internal batch is chunked by
``max_internal_batchsize`` the features are chunked alongside it. With
``geom_in_dims=None`` / ``geom_feats=None`` (the defaults) this is upstream.
"""

import torch
from torch.nn import Module
from typing import Union, Literal,Dict,Optional,Literal,Sequence
from collections import OrderedDict
from einops import rearrange
from ..architecture.p3d import _KLP3DEncoder, _P3DDecoder
from ..architecture.p3d.kl import DiagonalGaussianDistribution
from ..utils import load_weights

# GIFt is upstream Tadpole's fine-tuning library; it is used ONLY by the
# integer-rank (GIFt-LoRA) ``*_ft_state`` path. pyurbanair drives fine-tuning
# through HF PEFT instead (see the master plan's "unified LoRA strategy") and
# always constructs the autoencoder with ``*_ft_state`` in {"frozen", "FPFT"},
# so this import is never exercised. Keep it optional so the vendored module
# imports without the (unvendored) GIFt dependency; a genuine int-rank request
# then fails loud below with an actionable message.
try:  # pragma: no cover - GIFt is intentionally not a dependency here
    from GIFt.strategies.lora import LoRAAllFineTuningStrategy
    from GIFt import enable_fine_tuning
except ImportError:  # pragma: no cover
    LoRAAllFineTuningStrategy = None
    enable_fine_tuning = None

def _load_module_weights(module: Module, state_dict, geom_in_dims) -> None:
    """Load ``state_dict`` into ``module``; strict unless conditioned.

    Without ``geom_in_dims`` this is upstream's plain strict ``load_state_dict``.
    With it, the checkpoint may predate the (zero-init) geometry projections, so
    load non-strictly and fail loud unless every missing key is one of those and
    nothing is unexpected.
    """
    if geom_in_dims is None:
        module.load_state_dict(state_dict)
        return
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    stray = [k for k in missing if "geom_proj" not in k and "geom_latent_proj" not in k]
    if stray or unexpected:
        raise RuntimeError(
            "geometry-conditioned weight load failed: missing "
            f"{stray}, unexpected {list(unexpected)} (only the zero-init "
            "geom_proj / geom_latent_proj keys may be absent from a checkpoint)."
        )


class TadpoleAutoencoder(Module):

    def __init__(self, 
                 size: Literal["S", "B", "L"],
                weight_encoder: Optional[Union[str,Dict,OrderedDict]] = None,
                weight_decoder: Optional[Union[str,Dict,OrderedDict]] = None,
                encoder_ft_state: Union[Literal["frozen","FPFT"],int] = "FPFT",
                decoder_ft_state: Union[Literal["frozen","FPFT"],int] = "FPFT",
                latent_type: Literal["sample", "mode"] = "sample",
                encoder_crop_size: int = 64,
                max_internal_batchsize: Optional[int] = None,
                geom_in_dims: Optional[Sequence[int]] = None,
                ):
        super().__init__()
        
        """
        Tadpole Autoencoder Model.
        Input shape: (B, C, X, Y, Z); Output shape: (B, C, X, Y, Z)
        
        Args:
            size (Literal["S", "B", "L"]): Size of the model, one of "S", "B", or "L".
            weight_encoder (Optional[Union[str,Dict,OrderedDict]]): Path to encoder weights or state dict for encoder. If None, encoder will be randomly initialized. Default is None.
            weight_decoder (Optional[Union[str,Dict,OrderedDict]]): Path to decoder weights or state dict for decoder. If None, decoder will be randomly initialized. Default is None.
            encoder_ft_state (Union[Literal["frozen","FPFT"],int]): Fine-tuning state for encoder. Can be a positive integer indicating the rank for LoRA fine-tuning, "frozen" to freeze the encoder weights, or "FPFT" to enable full-parameter fine-tuning. Default is "FPFT".
            decoder_ft_state (Union[Literal["frozen","FPFT"],int]): Fine-tuning state for decoder. Can be a positive integer indicating the rank for LoRA fine-tuning, "frozen" to freeze the decoder weights, or "FPFT" to enable full-parameter fine-tuning. Default is "FPFT".
            latent_type (Literal["sample", "mode"]): How to sample from the latent distribution, either "sample" or "mode". Default is "sample".
            encoder_crop_size (int): Size to crop input for encoder. If None, no cropping will be applied and the entire input will be processed as a single crop. Default is 64.
            max_internal_batchsize (Optional[int]): Maximum batch size for internal processing. If None, all crops will be processed in a single batch. Default is None.
        """
        
        assert size in ["S", "B", "L"], "size must be one of 'S', 'B', 'L'"
        self.encoder = _KLP3DEncoder(size, geom_in_dims=geom_in_dims)
        self.decoder = _P3DDecoder(size, geom_in_dims=geom_in_dims)
        # Pretrained weights predate the geometry-branch projections, so a
        # conditioned build loads non-strictly and then asserts that the ONLY
        # thing missing is those (zero-init) projections -- keeping the HF /
        # local warm start available in branch mode without ever silently
        # tolerating a genuinely mismatched checkpoint.
        if weight_encoder is not None:
            _load_module_weights(
                self.encoder, load_weights(weight_encoder, "encoder"), geom_in_dims
            )
        if weight_decoder is not None:
            _load_module_weights(
                self.decoder, load_weights(weight_decoder, "decoder"), geom_in_dims
            )
        assert latent_type in ["sample", "mode"], "latent_type must be one of 'sample' or 'mode'"
        # set fine-tuning states for encoder and decoder
        if isinstance(encoder_ft_state, int):
            assert encoder_ft_state > 0, "encoder_ft_state must be a positive integer or 'eval' or 'FPFT'"
            if enable_fine_tuning is None:
                raise ImportError(
                    "integer (GIFt-LoRA) encoder_ft_state requires the 'GIFt' "
                    "package, which is not vendored; use 'frozen'/'FPFT' and drive "
                    "LoRA through neural_surrogates.finetuning (PEFT) instead."
                )
            enable_fine_tuning(self.encoder, LoRAAllFineTuningStrategy(encoder_ft_state,large_rank_warning=False))
        else:
            assert encoder_ft_state in ["frozen", "FPFT"], "encoder_ft_state must be a positive integer or 'eval' or 'FPFT'"
            if encoder_ft_state == "frozen":
                for param in self.encoder.parameters():
                    param.requires_grad = False
        if isinstance(decoder_ft_state, int):
            assert decoder_ft_state > 0, "decoder_ft_state must be a positive integer or 'eval' or 'FPFT'"
            if enable_fine_tuning is None:
                raise ImportError(
                    "integer (GIFt-LoRA) decoder_ft_state requires the 'GIFt' "
                    "package, which is not vendored; use 'frozen'/'FPFT' and drive "
                    "LoRA through neural_surrogates.finetuning (PEFT) instead."
                )
            enable_fine_tuning(self.decoder, LoRAAllFineTuningStrategy(decoder_ft_state,large_rank_warning=False))
        else:
            assert decoder_ft_state in ["frozen", "FPFT"], "decoder_ft_state must be a positive integer or 'eval' or 'FPFT'"
            if decoder_ft_state == "frozen":
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.latent_type = latent_type
        self.encoder_crop_size = encoder_crop_size
        self.max_internal_batchsize = max_internal_batchsize
        
    def latent_sample(self, dist: DiagonalGaussianDistribution) -> torch.Tensor:
        # pyurbanair: hand the decoder a CONTIGUOUS latent. ``dist.mode()`` /
        # ``dist.sample()`` are strided views into the encoder's (mean, logvar)
        # tensor; the DFT path (``model/dft.py``) reaches the decoder with a
        # fresh contiguous tensor instead. Kernel selection can depend on the
        # input layout, and on some CPUs (observed on AVX-512 CI runners) the two
        # layouts round differently, breaking the bit-exact identity-at-init
        # parity between the autoencoder and the DFT. Canonicalising here keeps
        # every path on the same kernels; the copy is a small latent.
        if self.latent_type == "sample":
            return dist.sample().contiguous()
        elif self.latent_type == "mode":
            return dist.mode().contiguous()
        else:
            raise ValueError(f"Unknown latent_type: {self.latent_type}")

    def forward(self, 
                x: torch.Tensor,
                return_kl_element: bool = False,
                geom_feats: Optional[list] = None,) -> torch.Tensor:
        # x: (B, C, X, Y, Z); geom_feats: the 4 FOLDED geometry-branch features
        # (batch dim == the folded batch), or None for the unconditioned path.
        kl_elem = None
        b, c, u, v, w = (
            x.shape[0],
            x.shape[1],
            max(x.shape[2] // self.encoder_crop_size,1),
            max(x.shape[3] // self.encoder_crop_size,1),
            max(x.shape[4] // self.encoder_crop_size,1),
        )
        x = rearrange(
            x, "B C (U Xc) (V Yc) (W Zc) -> (B C U V W) 1 Xc Yc Zc", U=u, V=v, W=w
        )
        if (
            self.max_internal_batchsize is None
            or x.shape[0] <= self.max_internal_batchsize
        ):
            dist = self.encoder(x, "distribution", geom_feats)
            x = self.latent_sample(dist)
            if return_kl_element:
                kl_elem=dist.kl_elem()
            x = self.decoder(x, geom_feats)
        else:
            n_chunks = (x.shape[0] // self.max_internal_batchsize) + 1
            x_chunks = torch.chunk(x, chunks=n_chunks, dim=0)
            # Same chunk count over the same folded batch size => the geometry
            # features split exactly like x, level by level.
            geom_chunks = (
                None
                if geom_feats is None
                else [torch.chunk(g, chunks=n_chunks, dim=0) for g in geom_feats]
            )
            x_out_chunks = []
            kl_elem_chunks = []
            for i, x_chunk in enumerate(x_chunks):
                geom_chunk = (
                    None if geom_chunks is None else [g[i] for g in geom_chunks]
                )
                x_out_chunk_dist = self.encoder(x_chunk, "distribution", geom_chunk)
                x_out_chunk = self.latent_sample(x_out_chunk_dist)
                if return_kl_element:
                    kl_elem_chunks.append(x_out_chunk_dist.kl_elem())
                x_out_chunk = self.decoder(x_out_chunk, geom_chunk)
                x_out_chunks.append(x_out_chunk)
            x = torch.cat(x_out_chunks, dim=0)
            if return_kl_element:
                kl_elem = torch.cat(kl_elem_chunks, dim=0)
        x = rearrange(
            x,
            "(B C U V W) 1 Xc Yc Zc -> B C (U Xc) (V Yc) (W Zc)",
            B=b,
            C=c,
            U=u,
            V=v,
            W=w,
        )
        if return_kl_element:
            return x, kl_elem
        return x
    
    def save_separate_weights(self, encoder_path: str, decoder_path: str):
        """Export encoder/decoder state dicts (incl. any geometry projections)."""
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
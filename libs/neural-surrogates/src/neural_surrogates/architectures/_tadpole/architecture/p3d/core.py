"""P3D encoder / decoder cores (vendored from Tadpole).

pyurbanair edit -- **geometry-branch conditioning**. Both cores take an optional
``geom_in_dims`` (the 4 ``GeometryBranch.out_dims``) which they forward to their
conv stem / up-path (levels 0/1/2 at strides 1/2/4, see ``conv.py``), and
:class:`P3DDecoder` additionally owns ``geom_latent_proj``: the zero-init
``1x1x1`` conv that adds the level-3 feature (stride 16 == the latent grid) to
the decoder's **latent input**, before the transformer decoder. ``forward``
gained a ``geom_feats=None`` argument carrying the folded branch features.

With ``geom_in_dims=None`` (the default) nothing is created and the behaviour --
including the ``state_dict`` key set -- is upstream's.
"""

import torch
from collections import OrderedDict
from typing import Optional, Sequence, Union
from diffusers import ModelMixin
from .conv import ConditionedEncoder3D, ConditionedDecoder3D, _zero_proj
from .transformer import P3DTransformerEncoder, P3DTransformerDecoder


class P3DEncoder(ModelMixin):

    def __init__(
        self,
        window_size: Union[int, Sequence[int]] = 8,
        hidden_size: int = 1152,
        max_hidden_size: int = 2048,
        depth=(2, 4, 4, 6, 4, 4, 2),
        num_heads: Union[int, Sequence[int]] = 16,
        mlp_ratio: float = 4.0,
        periodic: bool = False,
        shift: bool = False,
        feature_embedding_dim: Union[int, Sequence[int]] = 64,
        num_downsampling_layers: int = 3,
        time_embedding_dim: int = 64,
        num_groups: int = 32,
        repetitions: int = 1,
        ckpt_path: Optional[str] = None,
        ckpt_prefix: str = "model.encoder.",
        in_channels: int = 1,
        geom_in_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        # "hidden_size must be equal to the last element of feature_embedding_dim"
        self.transformer_encoder = P3DTransformerEncoder(
            window_size=window_size,
            hidden_size=hidden_size,
            max_hidden_size=max_hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            periodic=periodic,
            shift=shift,
        )
        self.num_downsampling_layers = num_downsampling_layers
        self.conv_encoder = ConditionedEncoder3D(
            in_channels=in_channels,
            feature_embedding_dim=feature_embedding_dim,
            num_downsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            num_groups=num_groups,
            repetitions=repetitions,
            geom_in_dims=geom_in_dims,
        )
        self.latent_size = self.transformer_encoder.latent_size
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, prefix=ckpt_prefix)

    def forward(
        self,
        x: torch.Tensor,
        geom_feats=None,
    ):
        ## Conv encoding
        x = self.conv_encoder(x, geom_feats)
        x = self.transformer_encoder(x)
        ## Sequence Modeling
        return x

    def init_from_ckpt(self, 
                       pretrained_weights: Union[str,OrderedDict],
                       prefix: str = "model.encoder."
                       ):
        if isinstance(pretrained_weights,str):
            pretrained_weights = torch.load(pretrained_weights, map_location="cpu")["state_dict"]
        if prefix != "":
            encoder_weights = {k.replace(prefix, ""): v for k, v in pretrained_weights.items() if prefix in k}
        self.load_state_dict(encoder_weights, strict=True)

class P3DDecoder(ModelMixin):

    def __init__(
        self,
        window_size: Union[int, Sequence[int]] = 8,
        hidden_size: int = 1152,
        max_hidden_size: int = 2048,
        depth=(2, 4, 4, 6, 4, 4, 2),
        num_heads: Union[int, Sequence[int]] = 16,
        mlp_ratio: float = 4.0,
        periodic: bool = False,
        shift: bool = False,
        feature_embedding_dim: Union[int, Sequence[int]] = 64,
        num_downsampling_layers: int = 3,
        time_embedding_dim: int = 64,
        num_groups: int = 32,
        repetitions: int = 1,
        ckpt_path: Optional[str] = None,
        ckpt_prefix: str = "model.decoder.",
        out_channels: int = 1,
        geom_in_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.transformer_decoder = P3DTransformerDecoder(
            window_size=window_size,
            hidden_size=hidden_size,
            max_hidden_size=max_hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            periodic=periodic,
            shift=shift,
        )
        self.num_downsampling_layers = num_downsampling_layers
        self.latent_size = hidden_size * 2 ** (len(depth) // 2)
        self.conv_decoder = ConditionedDecoder3D(
            out_channels=out_channels,
            feature_embedding_dim=feature_embedding_dim[::-1],
            num_upsampling_layers=num_downsampling_layers,
            embedding_dim=time_embedding_dim,
            features_first_layer=feature_embedding_dim[-1],
            num_groups=num_groups,
            repetitions=repetitions,
            geom_in_dims=geom_in_dims,
        )
        # pyurbanair: level-3 (stride 16) conditioning of the latent input.
        self.geom_latent_proj = None
        if geom_in_dims is not None:
            self.geom_latent_proj = _zero_proj(geom_in_dims[3], self.latent_size)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, prefix=ckpt_prefix)

    def forward(self, x, geom_feats=None):
        # region partition
        if geom_feats is not None and self.geom_latent_proj is not None:
            x = x + self.geom_latent_proj(geom_feats[3])
        x = self.transformer_decoder(x)
        reconstructed = self.conv_decoder(x, geom_feats)
        return reconstructed
    
    def init_from_ckpt(self, 
                       pretrained_weights: Union[str,OrderedDict],
                       prefix: str = "model.decoder."
                       ):
        if isinstance(pretrained_weights,str):
            pretrained_weights = torch.load(pretrained_weights, map_location="cpu")["state_dict"]
        if prefix != "":
            decoder_weights = {k.replace(prefix, ""): v for k, v in pretrained_weights.items() if prefix in k}
        self.load_state_dict(decoder_weights, strict=True)
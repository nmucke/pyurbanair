"""Conv stem / up-path of the P3D encoder-decoder (vendored from Tadpole).

pyurbanair edit -- **geometry-branch injection points**. Both classes take an
optional ``geom_in_dims`` (the 4 ``GeometryBranch.out_dims``); when it is given
they build zero-initialised ``1x1x1`` ``Conv3d`` projections and ``forward``
accepts the folded branch features, adding level ``i`` at the matching stride:

* :class:`ConditionedEncoder3D` -- level 0 after ``feature_embed`` (stride 1),
  level 1 after ``downsampling_layers[0]`` (stride 2), level 2 after the last
  ``downsampling_layers`` (stride 4). ``geom_proj[i]`` is indexed by branch level.
* :class:`ConditionedDecoder3D` -- mirrored: level 2 at the conv input
  (stride 4), level 1 after ``upsampling_layers[0]`` (stride 2), level 0 after
  the last ``upsampling_layers``, i.e. just before ``decompress`` (stride 1).

With ``geom_in_dims=None`` (the default) **no** sub-module is created and
``forward`` is the upstream one, so the ``state_dict`` key set -- and hence
strict loading of the HF ``thuerey-group/Tadpole`` weights -- is unchanged.
"""

import torch.nn as nn
from typing import Optional, Sequence, Union
from .modules import PixelShuffle3d


def _check_geom_in_dims(geom_in_dims: Sequence[int], num_layers: int) -> None:
    """The injection strides (1, 2, 4) assume the shipped 2-downsampling stem."""
    if len(geom_in_dims) != 4:
        raise ValueError(
            f"geom_in_dims must have 4 entries (branch levels), got "
            f"{len(geom_in_dims)}"
        )
    if num_layers != 2:
        raise ValueError(
            "geometry-branch conditioning assumes num_downsampling_layers=2 (the "
            f"strides 1/2/4 of the shipped P3D configs), got {num_layers}"
        )


def _zero_proj(in_channels: int, out_channels: int) -> nn.Conv3d:
    """Zero-initialised ``1x1x1`` conditioning projection (identity at init)."""
    conv = nn.Conv3d(in_channels, out_channels, 1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class ConditionedEncoder3DBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.gn_1 = nn.GroupNorm(num_groups, in_channels)
        self.activation_1 = nn.GELU()
        self.conv_1 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)

        self.gn_2 = nn.GroupNorm(num_groups, in_channels)
        self.activation_2 = nn.GELU()
        self.conv_2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x_res = x
        x = self.gn_1(x)
        x = self.activation_1(x)
        x = self.conv_1(x)
        x = self.gn_2(x)
        x = self.activation_2(x)
        x = self.conv_2(x)
        x = x + x_res
        return x


class ConditionedEncoder3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        feature_embedding_dim: Union[int, Sequence[int]],
        num_downsampling_layers: int,
        embedding_dim: int, #need to remove, useless
        repetitions: int = 1,
        num_groups: int = 32,
        geom_in_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels

        if isinstance(feature_embedding_dim, Sequence):
            self.feature_embedding_dim = feature_embedding_dim
            if len(self.feature_embedding_dim) != num_downsampling_layers + 1:
                raise ValueError(
                    "Length of feature_embedding_dim sequence must be equal to num_downsampling_layers + 1"
                )
        else:
            self.feature_embedding_dim = [
                feature_embedding_dim * 2**i for i in range(num_downsampling_layers + 1)
            ]

        self.repetitions = repetitions
        self.num_downsampling_layers = num_downsampling_layers
        self.embedding_dim = embedding_dim
        self.feature_embed = nn.Conv3d(
            in_channels, self.feature_embedding_dim[0], 3, 1, 1
        )
        self.downsampling_layers = nn.ModuleList()
        for i in range(num_downsampling_layers):
            self.downsampling_layers.append(
                nn.Conv3d(
                    self.feature_embedding_dim[i],
                    self.feature_embedding_dim[i + 1],
                    3,
                    2,
                    1,
                )
            )
        self.blocks = nn.ModuleList()
        for i in range(num_downsampling_layers - 1):
            self.blocks.extend(
                [
                    ConditionedEncoder3DBlock(
                        self.feature_embedding_dim[i + 1],
                        num_groups=num_groups,
                    )
                    for _ in range(repetitions)
                ]
            )

        # pyurbanair: geometry-branch projections (levels 0/1/2 = strides 1/2/4).
        # Only created when conditioning is requested -- see the module docstring.
        self.geom_proj = None
        if geom_in_dims is not None:
            _check_geom_in_dims(geom_in_dims, num_downsampling_layers)
            self.geom_proj = nn.ModuleList(
                [
                    _zero_proj(geom_in_dims[i], self.feature_embedding_dim[i])
                    for i in range(3)
                ]
            )

    def _add_geom(self, x, geom_feats, level):
        """Add the projected branch feature for ``level``, if conditioning is on."""
        if geom_feats is None:
            return x
        if self.geom_proj is None:
            raise ValueError(
                "geom_feats were passed but this encoder was built without "
                "geom_in_dims (no conditioning projections exist)."
            )
        return x + self.geom_proj[level](geom_feats[level])

    def forward(self, x, geom_feats=None):
        x = self.feature_embed(x)
        x = self._add_geom(x, geom_feats, 0)
        x = self.downsampling_layers[0](x)
        x = self._add_geom(x, geom_feats, 1)
        for i in range(self.num_downsampling_layers - 1):
            for j in range(self.repetitions):
                x = self.blocks[i * self.repetitions + j](x)
            x = self.downsampling_layers[i + 1](x)
        x = self._add_geom(x, geom_feats, 2)
        return x


ConditionedDecoder3DBlock = ConditionedEncoder3DBlock


class DecoderUpsamplingBlock(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, factor: Optional[int] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear_conv = nn.Conv3d(in_channels, out_channels * 8, 1)
        self.shuffle = PixelShuffle3d(2)

    def forward(self, x):
        x = self.linear_conv(x)
        x = self.shuffle(x)
        return x


class ConditionedDecoder3D(nn.Module):

    def __init__(
        self,
        out_channels: int,
        feature_embedding_dim: Union[int, Sequence[int]],
        num_upsampling_layers: int,
        embedding_dim: int,# need to remove, useless
        repetitions: int = 1,
        features_first_layer: int = None,
        num_groups: int = 32,
        geom_in_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.out_channels = out_channels

        if isinstance(feature_embedding_dim, Sequence):
            self.feature_embedding_dim = feature_embedding_dim
            if len(self.feature_embedding_dim) != num_upsampling_layers + 1:
                raise ValueError(
                    "Length of feature_embedding_dim sequence must be equal to num_upsampling_layers + 1"
                )
        else:
            self.feature_embedding_dim = [
                feature_embedding_dim * 2 ** (num_upsampling_layers - i)
                for i in range(num_upsampling_layers + 1)
            ]

        self.num_upsampling_layers = num_upsampling_layers
        self.embedding_dim = embedding_dim
        self.repetitions = repetitions

        self.decompress = nn.Conv3d(
            self.feature_embedding_dim[-1], out_channels, 3, 1, 1
        )

        self.blocks = nn.ModuleList()
        for i in range(num_upsampling_layers - 1):
            self.blocks.extend(
                [
                    ConditionedDecoder3DBlock(
                        self.feature_embedding_dim[i + 1],
                        num_groups=num_groups,
                    )
                    for _ in range(self.repetitions)
                ]
            )

        if features_first_layer is None:
            features_first_layer = self.feature_embedding_dim[0]

        self.upsampling_layers = nn.ModuleList()

        local_feature_dim = self.feature_embedding_dim[1]
        self.upsampling_layers.append(
            DecoderUpsamplingBlock(
                features_first_layer, local_feature_dim
            )
        )
        for i in range(num_upsampling_layers - 1):
            local_feature_dim = self.feature_embedding_dim[i + 1]
            self.upsampling_layers.append(
                DecoderUpsamplingBlock(
                    local_feature_dim,
                    self.feature_embedding_dim[i + 2],
                )
            )

        # pyurbanair: geometry-branch projections, mirroring the encoder stem --
        # level 2 (stride 4) at the conv input, level 1 (stride 2) after the first
        # upsampling, level 0 (stride 1) just before `decompress`. Indexed by
        # branch level, so `geom_proj[i]` matches the encoder's `geom_proj[i]`.
        self.geom_proj = None
        if geom_in_dims is not None:
            _check_geom_in_dims(geom_in_dims, num_upsampling_layers)
            self.geom_proj = nn.ModuleList(
                [
                    _zero_proj(geom_in_dims[0], self.feature_embedding_dim[-1]),
                    _zero_proj(geom_in_dims[1], self.feature_embedding_dim[1]),
                    _zero_proj(geom_in_dims[2], features_first_layer),
                ]
            )

    def _add_geom(self, x, geom_feats, level):
        """Add the projected branch feature for ``level``, if conditioning is on."""
        if geom_feats is None:
            return x
        if self.geom_proj is None:
            raise ValueError(
                "geom_feats were passed but this decoder was built without "
                "geom_in_dims (no conditioning projections exist)."
            )
        return x + self.geom_proj[level](geom_feats[level])

    def forward(self, x, geom_feats=None):

        x = self._add_geom(x, geom_feats, 2)

        x = self.upsampling_layers[0](x)
        x = self._add_geom(x, geom_feats, 1)

        for i in range(self.num_upsampling_layers - 1):
            for j in range(self.repetitions):
                x = self.blocks[i * self.repetitions + j](x)
            x = self.upsampling_layers[i + 1](x)

        x = self._add_geom(x, geom_feats, 0)

        x = self.decompress(x)

        return x

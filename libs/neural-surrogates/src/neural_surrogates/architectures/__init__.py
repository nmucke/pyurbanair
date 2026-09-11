from neural_surrogates.architectures.domain_decomposed import DomainDecomposed
from neural_surrogates.architectures.p3d import P3D
from neural_surrogates.architectures.simple_conv import SimpleConv
from neural_surrogates.architectures.tadpole_ae import TadpoleAE
from neural_surrogates.architectures.tadpole_discriminator import TadpoleDiscriminator
from neural_surrogates.architectures.tadpole_geometry_branch import GeometryBranch
from neural_surrogates.architectures.tadpole_latent_flow import (
    LatentEncoding,
    TadpoleLatentGenerator,
)
from neural_surrogates.architectures.tadpole_stepper import (
    ParamConditionedSubnetwork,
    TadpoleTimeStepper,
)
from neural_surrogates.architectures.unet_convnext import UNetConvNeXt
from neural_surrogates.architectures.upt import UPT

__all__ = [
    "SimpleConv",
    "UNetConvNeXt",
    "UPT",
    "P3D",
    "DomainDecomposed",
    "TadpoleAE",
    "TadpoleDiscriminator",
    "GeometryBranch",
    "TadpoleTimeStepper",
    "ParamConditionedSubnetwork",
    "TadpoleLatentGenerator",
    "LatentEncoding",
]
